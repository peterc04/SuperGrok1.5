#!/usr/bin/env python3
"""Performance benchmark for all optimizers across configurations.

Reports: step time (ms), peak memory (MB), throughput (params/sec).

Usage:
    python benchmarks/benchmark_supergrok2.py                          # all optimizers
    python benchmarks/benchmark_supergrok2.py --optimizer SuperGrok2   # single optimizer
    python benchmarks/benchmark_supergrok2.py --model-size 512         # larger model
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
        return Grokfast(model.parameters(), lr=1e-3)
    elif name == 'Prodigy':
        return Prodigy(model.parameters(), lr=1e-3)
    elif name == 'Muon':
        params_2d = [p for p in model.parameters() if p.dim() >= 2]
        params_1d = [p for p in model.parameters() if p.dim() < 2]
        return Muon(params_2d, params_1d, lr=0.02)
    elif name == 'LookSAM':
        return LookSAM(model.parameters(), lr=1e-3)
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
        return 0.0

    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(32, 16, device=device)

    for _ in range(num_steps):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return peak_mb


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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from grokking_optimizers.dispatch import get_arch_label

    print(f"\n{'='*70}")
    print(f"  Performance Benchmark")
    print(f"  Device: {device} ({get_arch_label()})")
    print(f"  FORCE_ARCH: {os.environ.get('SUPERGROK_FORCE_ARCH', 'native')}")
    print(f"  Model: Linear(16->{args.model_size}->10)")
    print(f"  Steps: {args.num_steps} (+ {args.num_warmup} warmup)")
    print(f"{'='*70}\n")

    all_optimizers = [
        'AdamW', 'SuperGrok2', 'SuperGrok15', 'SuperGrok11',
        'NeuralGrok', 'GrokAdamW', 'Lion', 'Grokfast',
        'Prodigy', 'Muon', 'LookSAM',
    ]

    if args.optimizer:
        all_optimizers = [args.optimizer]

    print(f"  {'Optimizer':<20} {'Step (ms)':>10} {'Params/s':>12} {'Mem (MB)':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*12} {'─'*10}")

    for name in all_optimizers:
        try:
            model = make_model(args.model_size, device)
            opt = make_optimizer(name, model)

            step_ms, throughput = benchmark_step(
                model, opt, device, args.num_warmup, args.num_steps)
            peak_mb = benchmark_memory(model, opt, device)

            print(f"  {name:<20} {step_ms:>10.2f} {throughput:>12,.0f} {peak_mb:>10.1f}")
        except Exception as e:
            print(f"  {name:<20} {'ERROR':>10} {str(e)[:40]}")

    print()


if __name__ == '__main__':
    main()
