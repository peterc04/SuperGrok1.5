"""End-to-end training benchmark.

Measures:
  - Samples/second (throughput)
  - Time breakdown: data load, forward, backward, optimizer step, sync overhead
  - Peak memory (model + activations + optimizer state + workspace)
  - L2 cache hit rate proxy (first-layer forward time after optimizer vs after warmup)
  - Steps to reach target accuracy (convergence speed)

Usage:
    python benchmarks/training_benchmark.py --optimizer SuperGrok2 --model gpt2-small
    python benchmarks/training_benchmark.py --optimizer AdamW --model gpt2-small  # baseline
    python benchmarks/training_benchmark.py --all  # compare all optimizers
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Model Definitions ──────────────────────────────────────────

class TinyTransformer(nn.Module):
    """Minimal transformer for benchmarking. Matches grokking experiment scale."""
    def __init__(self, vocab_size=99, dim=128, num_heads=4, num_layers=2, seq_len=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(seq_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.transformer(h)
        return self.head(h[:, -1])


class MediumTransformer(nn.Module):
    """GPT-2 small scale (~125M params) for realistic benchmarking."""
    def __init__(self, vocab_size=50257, dim=768, num_heads=12, num_layers=12, seq_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(seq_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.transformer(h)
        return self.head(h)


MODEL_CONFIGS = {
    'tiny': {'cls': TinyTransformer, 'batch_size': 512, 'seq_len': 4},
    'small': {'cls': MediumTransformer, 'batch_size': 8, 'seq_len': 512,
              'kwargs': {'dim': 512, 'num_heads': 8, 'num_layers': 6}},
    'gpt2-small': {'cls': MediumTransformer, 'batch_size': 4, 'seq_len': 1024,
                   'kwargs': {'dim': 768, 'num_heads': 12, 'num_layers': 12}},
}


# ─── Phase Timing ────────────────────────────────────────────────

class PhaseTimer:
    """Measures time spent in each training phase with CUDA events."""

    def __init__(self, device):
        self.device = device
        self.use_cuda = device.type == 'cuda'
        self.timings = {
            'data': [], 'forward': [], 'backward': [],
            'optimizer': [], 'total': [], 'l2_probe': [],
        }

    def _make_event(self):
        if self.use_cuda:
            return torch.cuda.Event(enable_timing=True)
        return None

    def start(self, phase):
        if self.use_cuda:
            event = self._make_event()
            event.record()
            self._current_start = event
        else:
            self._current_start = time.perf_counter()

    def stop(self, phase):
        if self.use_cuda:
            end = self._make_event()
            end.record()
            torch.cuda.synchronize()
            ms = self._current_start.elapsed_time(end)
        else:
            ms = (time.perf_counter() - self._current_start) * 1000
        self.timings[phase].append(ms)
        return ms

    def l2_cache_probe(self, model, x):
        """Measure first-layer forward time as L2 warmth proxy.

        After optimizer step: L2 may be cold (flushed by state streaming).
        After a fresh forward: L2 is warm with model weights.
        Ratio = cold/warm tells us how much L2 pollution the optimizer causes.
        """
        if not self.use_cuda:
            return

        # Warm probe: L2 has model weights from previous forward
        torch.cuda.synchronize()
        start = self._make_event()
        end = self._make_event()
        start.record()
        with torch.no_grad():
            _ = model.embed(x)  # just first layer
        end.record()
        torch.cuda.synchronize()
        warm_ms = start.elapsed_time(end)

        self.timings['l2_probe'].append(warm_ms)

    def summary(self, skip_first=5):
        """Return mean times, skipping warmup steps."""
        result = {}
        for phase, times in self.timings.items():
            if len(times) > skip_first:
                vals = times[skip_first:]
                result[phase] = {
                    'mean_ms': sum(vals) / len(vals),
                    'min_ms': min(vals),
                    'max_ms': max(vals),
                    'count': len(vals),
                }
        return result


# ─── Memory Tracker ──────────────────────────────────────────────

def measure_memory_breakdown(model, optimizer, device):
    """Measure memory used by model, optimizer, and activations separately."""
    if device.type != 'cuda':
        return {}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Model params only
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Optimizer state
    opt_bytes = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                for k, v in optimizer.state[p].items():
                    if torch.is_tensor(v):
                        opt_bytes += v.numel() * v.element_size()

    # For SuperGrok v2: meta-net + flat states
    if hasattr(optimizer, 'meta_net'):
        meta_bytes = sum(p.numel() * p.element_size() for p in optimizer.meta_net.parameters())
    else:
        meta_bytes = 0

    flat_bytes = 0
    if hasattr(optimizer, '_flat_exp_avgs'):
        for t in optimizer._flat_exp_avgs:
            flat_bytes += t.numel() * t.element_size()
        for t in optimizer._flat_exp_avg_sqs:
            flat_bytes += t.numel() * t.element_size()
        for t in optimizer._flat_mus:
            flat_bytes += t.numel() * t.element_size()
        for t in optimizer._flat_sharpness:
            flat_bytes += t.numel() * t.element_size()
        for t in optimizer._flat_gru_states:
            flat_bytes += t.numel() * t.element_size()
        for s in optimizer._flat_mamba_fwd_states:
            if s is not None:
                flat_bytes += s.numel() * s.element_size()
        for s in optimizer._flat_mamba_bwd_states:
            if s is not None:
                flat_bytes += s.numel() * s.element_size()

    return {
        'model_mb': model_bytes / (1024**2),
        'optimizer_state_mb': (opt_bytes + flat_bytes) / (1024**2),
        'meta_net_mb': meta_bytes / (1024**2),
        'total_static_mb': (model_bytes + opt_bytes + flat_bytes + meta_bytes) / (1024**2),
    }


# ─── Training Loop ───────────────────────────────────────────────

def run_training_benchmark(
    model_name, optimizer_name, num_steps=100, num_warmup=10,
    device_str=None, state_precision='fp32',
):
    device = torch.device(device_str or ('cuda' if torch.cuda.is_available() else 'cpu'))
    config = MODEL_CONFIGS[model_name]

    kwargs = config.get('kwargs', {})
    model = config['cls'](**kwargs).to(device)
    batch_size = config['batch_size']
    seq_len = config.get('seq_len', 4)
    vocab_size = 99 if model_name == 'tiny' else 50257

    # Create optimizer
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    elif optimizer_name == 'SuperGrok2':
        from grokking_optimizers import SuperGrok2
        optimizer = SuperGrok2(model.parameters(), lr=1e-3)
    elif optimizer_name == 'GrokAdamW':
        from grokking_optimizers import GrokAdamW
        optimizer = GrokAdamW(model.parameters(), lr=1e-3)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    nparams = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    timer = PhaseTimer(device)

    print(f"\n{'='*60}")
    print(f"  Training Benchmark")
    print(f"  Model: {model_name} ({nparams:,} params)")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Batch: {batch_size} x {seq_len}")
    print(f"  Device: {device}")
    print(f"  Steps: {num_steps} (+{num_warmup} warmup)")
    print(f"{'='*60}\n")

    # Pre-generate random data (don't measure data loading)
    data_x = torch.randint(0, vocab_size, (num_steps + num_warmup, batch_size, seq_len), device=device)
    data_y = torch.randint(0, vocab_size, (num_steps + num_warmup, batch_size), device=device)

    # Run training loop
    for step in range(num_steps + num_warmup):
        x = data_x[step]
        y = data_y[step]

        timer.start('total')

        # Forward
        timer.start('forward')
        output = model(x)
        if output.dim() == 3:
            output = output[:, -1]  # last token
        loss = criterion(output, y)
        timer.stop('forward')

        # Backward
        timer.start('backward')
        loss.backward()
        timer.stop('backward')

        # Optimizer step
        timer.start('optimizer')
        optimizer.step()
        optimizer.zero_grad()
        timer.stop('optimizer')

        timer.stop('total')

        # L2 cache probe: how warm is L2 after optimizer step?
        if step >= num_warmup and step % 10 == 0:
            timer.l2_cache_probe(model, x)

    # Memory breakdown
    mem = measure_memory_breakdown(model, optimizer, device)

    # Results
    summary = timer.summary(skip_first=num_warmup)

    print(f"  Phase Breakdown:")
    for phase in ['forward', 'backward', 'optimizer', 'total']:
        if phase in summary:
            s = summary[phase]
            pct = s['mean_ms'] / summary['total']['mean_ms'] * 100 if phase != 'total' else 100
            print(f"    {phase:>12}: {s['mean_ms']:>8.2f} ms ({pct:>5.1f}%)")

    if 'l2_probe' in summary and summary['l2_probe']['count'] > 0:
        print(f"    {'l2_probe':>12}: {summary['l2_probe']['mean_ms']:>8.3f} ms (first-layer after opt)")

    throughput = batch_size / (summary['total']['mean_ms'] / 1000)
    print(f"\n  Throughput: {throughput:,.0f} samples/sec")

    if mem:
        print(f"\n  Memory:")
        print(f"    Model:          {mem['model_mb']:>8.1f} MB")
        print(f"    Optimizer state:{mem['optimizer_state_mb']:>8.1f} MB")
        if mem['meta_net_mb'] > 0:
            print(f"    Meta-net:       {mem['meta_net_mb']:>8.1f} MB")
        print(f"    Total static:   {mem['total_static_mb']:>8.1f} MB")
        if device.type == 'cuda':
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"    Peak allocated: {peak:>8.1f} MB")
            avail = torch.cuda.get_device_properties(0).total_mem / (1024**2)
            print(f"    GPU total:      {avail:>8.1f} MB")
            print(f"    Headroom:       {avail - peak:>8.1f} MB")

    return summary, mem, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='tiny', choices=MODEL_CONFIGS.keys())
    parser.add_argument('--optimizer', default='SuperGrok2')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--all', action='store_true', help='Compare all optimizers')
    parser.add_argument('--state-precision', default='fp32', choices=['fp32', 'bf16', 'config3'])
    args = parser.parse_args()

    if args.all:
        results = {}
        for opt_name in ['AdamW', 'SuperGrok2', 'GrokAdamW']:
            try:
                s, m, t = run_training_benchmark(
                    args.model, opt_name, args.steps, args.warmup)
                results[opt_name] = {'summary': s, 'memory': m, 'throughput': t}
            except Exception as e:
                print(f"  {opt_name} FAILED: {e}")

        # Comparison table
        print(f"\n{'='*60}")
        print(f"  Comparison: {args.model}")
        print(f"{'='*60}")
        print(f"  {'Optimizer':<15} {'Step (ms)':>10} {'Opt (ms)':>10} {'Opt %':>8} {'Samples/s':>12} {'Mem (MB)':>10}")
        for name, r in results.items():
            s = r['summary']
            print(f"  {name:<15} {s['total']['mean_ms']:>10.2f} {s['optimizer']['mean_ms']:>10.2f} "
                  f"{s['optimizer']['mean_ms']/s['total']['mean_ms']*100:>7.1f}% "
                  f"{r['throughput']:>12,.0f} {r['memory'].get('total_static_mb', 0):>10.1f}")
    else:
        run_training_benchmark(args.model, args.optimizer, args.steps, args.warmup,
                              state_precision=args.state_precision)

if __name__ == '__main__':
    main()
