#!/usr/bin/env python3
"""Performance benchmark for all optimizers across configurations.

Reports: step time (ms), peak memory (MB), throughput (params/sec).
Supports per-tier benchmarking via FORCE_ARCH and memory breakdown.

Usage:
    python benchmarks/benchmark_supergrok2.py                          # all optimizers
    python benchmarks/benchmark_supergrok2.py --optimizer SuperGrok2   # single optimizer
    python benchmarks/benchmark_supergrok2.py --model-size 512         # larger model
    python benchmarks/benchmark_supergrok2.py --include-bilevel        # include bilevel timing
    python benchmarks/benchmark_supergrok2.py --per-tier               # compare FORCE_ARCH tiers
    SUPERGROK_FORCE_ARCH=80 python benchmarks/benchmark_supergrok2.py  # specific tier
"""

import argparse
import os
import time
import torch
import torch.nn as nn


def make_model(hidden_dim, device):
    """Create benchmark model."""
    return nn.Sequential(
        nn.Linear(16, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 10),
    ).to(device)


def make_optimizer(name, model):
    """Create optimizer by name, matching actual constructor APIs."""
    from grokking_optimizers import (
        SuperGrok2, SuperGrok15, SuperGrok11, NeuralGrok,
        GrokAdamW, Lion, Grokfast, Prodigy, Muon, LookSAM,
    )

    if name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=1e-3)
    elif name == 'SuperGrok2':
        return SuperGrok2(model.parameters(), lr=1e-3)
    elif name == 'SuperGrok15':
        return SuperGrok15(model.parameters(), lr=1e-3)
    elif name == 'SuperGrok11':
        return SuperGrok11(model.parameters(), lr=1e-3)
    elif name == 'NeuralGrok':
        return NeuralGrok(model.parameters(), lr=1e-3)
    elif name == 'GrokAdamW':
        return GrokAdamW(model.parameters(), lr=1e-3)
    elif name == 'Lion':
        return Lion(model.parameters(), lr=1e-4)
    elif name == 'Grokfast':
        # Grokfast wraps a base optimizer
        base = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return Grokfast(base)
    elif name == 'Prodigy':
        return Prodigy(model.parameters(), lr=1e-3)
    elif name == 'Muon':
        params_2d = [p for p in model.parameters() if p.dim() >= 2]
        params_1d = [p for p in model.parameters() if p.dim() < 2]
        return Muon(params_2d, params_1d, lr=0.02)
    elif name == 'LookSAM':
        # LookSAM wraps a base optimizer
        base = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return LookSAM(base, model)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def benchmark_step(model, optimizer, device, num_warmup=10, num_steps=100):
    """Benchmark optimizer step time."""
    x = torch.randn(32, 16, device=device)

    # Warmup
    for _ in range(num_warmup):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        t0 = time.perf_counter()

    for _ in range(num_steps):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device.type == 'cuda':
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000

    step_ms = elapsed_ms / num_steps
    total_params = sum(p.numel() for p in model.parameters())
    throughput = total_params / (step_ms / 1000)  # params/sec

    return step_ms, throughput


def benchmark_memory(model, optimizer, device, num_steps=20):
    """Measure peak GPU memory during optimizer step."""
    if device.type != 'cuda':
        return 0.0, {}

    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(32, 16, device=device)

    for _ in range(num_steps):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Memory breakdown
    breakdown = {}
    total_params = sum(p.numel() for p in model.parameters())
    breakdown['model_params'] = total_params * 4 / (1024 * 1024)  # FP32

    # Optimizer states (approximate)
    if hasattr(optimizer, 'state'):
        opt_state_mb = 0
        for state in optimizer.state.values():
            if isinstance(state, dict):
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        opt_state_mb += v.numel() * v.element_size()
        breakdown['optimizer_states'] = opt_state_mb / (1024 * 1024)

    # Meta-net weights (for SuperGrok optimizers)
    if hasattr(optimizer, 'meta_net'):
        meta_params = sum(p.numel() * p.element_size()
                          for p in optimizer.meta_net.parameters())
        breakdown['meta_net_weights'] = meta_params / (1024 * 1024)

    return peak_mb, breakdown


def benchmark_bilevel(model, optimizer, device, num_steps=10):
    """Benchmark bilevel meta-learning step time (SuperGrok v2 only)."""
    if not hasattr(optimizer, 'bilevel_step'):
        return None

    x = torch.randn(32, 16, device=device)

    # Warmup
    for _ in range(3):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    for _ in range(num_steps):
        loss = model(x).sum()
        loss.backward()
        try:
            optimizer.bilevel_step()
        except Exception:
            return None
        optimizer.zero_grad()

    if device.type == 'cuda':
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_steps
    return None


def run_single_tier(args, device, force_arch=None):
    """Run benchmarks for a single FORCE_ARCH tier."""
    if force_arch is not None:
        os.environ['SUPERGROK_FORCE_ARCH'] = str(force_arch)

    from grokking_optimizers.dispatch import get_arch_label

    tier_label = f"FORCE_ARCH={force_arch}" if force_arch else "native"
    print(f"\n  --- Tier: {tier_label} ({get_arch_label()}) ---\n")

    all_optimizers = [
        'AdamW', 'SuperGrok2', 'SuperGrok15', 'SuperGrok11',
        'NeuralGrok', 'GrokAdamW', 'Lion', 'Grokfast',
        'Prodigy', 'Muon', 'LookSAM',
    ]

    if args.optimizer:
        all_optimizers = [args.optimizer]

    header = f"  {'Optimizer':<20} {'Step (ms)':>10} {'Params/s':>12} {'Mem (MB)':>10}"
    if args.include_bilevel:
        header += f" {'Bilevel (ms)':>12}"
    print(header)
    print(f"  {'─'*20} {'─'*10} {'─'*12} {'─'*10}" +
          (f" {'─'*12}" if args.include_bilevel else ""))

    results = []
    for name in all_optimizers:
        try:
            model = make_model(args.model_size, device)
            opt = make_optimizer(name, model)

            step_ms, throughput = benchmark_step(
                model, opt, device, args.num_warmup, args.num_steps)
            peak_mb, breakdown = benchmark_memory(model, opt, device)

            line = f"  {name:<20} {step_ms:>10.2f} {throughput:>12,.0f} {peak_mb:>10.1f}"

            bilevel_ms = None
            if args.include_bilevel:
                bilevel_ms = benchmark_bilevel(model, opt, device)
                if bilevel_ms is not None:
                    line += f" {bilevel_ms:>12.2f}"
                else:
                    line += f" {'N/A':>12}"

            print(line)

            # Print memory breakdown if verbose
            if args.verbose and breakdown:
                for k, v in breakdown.items():
                    print(f"    {k}: {v:.1f} MB")

            results.append({
                'optimizer': name,
                'step_ms': step_ms,
                'throughput': throughput,
                'peak_mb': peak_mb,
                'bilevel_ms': bilevel_ms,
                'breakdown': breakdown,
            })
        except Exception as e:
            print(f"  {name:<20} {'ERROR':>10} {str(e)[:40]}")
            results.append({'optimizer': name, 'error': str(e)})

    if force_arch is not None:
        del os.environ['SUPERGROK_FORCE_ARCH']

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark grokking optimizers')
    parser.add_argument('--optimizer', type=str, default=None,
                        help='Single optimizer to benchmark')
    parser.add_argument('--model-size', type=int, default=128,
                        help='Model hidden dim')
    parser.add_argument('--num-steps', type=int, default=100,
                        help='Steps to benchmark')
    parser.add_argument('--num-warmup', type=int, default=10,
                        help='Warmup steps')
    parser.add_argument('--include-bilevel', action='store_true',
                        help='Include bilevel meta-learning timing')
    parser.add_argument('--per-tier', action='store_true',
                        help='Benchmark across FORCE_ARCH tiers (75, 80, 90)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show memory breakdown per optimizer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from grokking_optimizers.dispatch import get_arch_label

    print(f"\n{'='*70}")
    print(f"  Performance Benchmark")
    print(f"  Device: {device} ({get_arch_label()})")
    print(f"  FORCE_ARCH: {os.environ.get('SUPERGROK_FORCE_ARCH', 'native')}")
    print(f"  Model: Linear(16->{args.model_size}->10)")
    print(f"  Steps: {args.num_steps} (+ {args.num_warmup} warmup)")
    print(f"{'='*70}")

    if args.per_tier and device.type == 'cuda':
        tiers = [75, 80, 90]
        all_results = {}
        for tier in tiers:
            all_results[tier] = run_single_tier(args, device, force_arch=tier)

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"  Cross-Tier Comparison (step time in ms)")
        print(f"{'='*70}\n")

        opt_names = set()
        for results in all_results.values():
            for r in results:
                if 'error' not in r:
                    opt_names.add(r['optimizer'])

        header = f"  {'Optimizer':<20}"
        for tier in tiers:
            header += f" {'ARCH=' + str(tier):>12}"
        print(header)
        print(f"  {'─'*20}" + f" {'─'*12}" * len(tiers))

        for name in sorted(opt_names):
            line = f"  {name:<20}"
            for tier in tiers:
                matching = [r for r in all_results[tier]
                            if r.get('optimizer') == name and 'error' not in r]
                if matching:
                    line += f" {matching[0]['step_ms']:>12.2f}"
                else:
                    line += f" {'ERR':>12}"
            print(line)
    else:
        run_single_tier(args, device)

    print()


if __name__ == '__main__':
    main()
