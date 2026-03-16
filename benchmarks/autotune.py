#!/usr/bin/env python3
"""Auto-tune kernel launch configurations per GPU model.

Profiles step time, bilevel time, and memory for the current compiled
configuration. Compares FP32 vs TF32 vs BF16 projection precision
(where available). Saves results as JSON and a human-readable report.

Note on scan block size: PSCAN_BLOCK is a constexpr in types.h
(currently 512). Autotuning requires recompilation with different
constexpr values. This script profiles the current compiled
configuration — it does NOT change PSCAN_BLOCK at runtime.

Usage:
    python benchmarks/autotune.py              # profile current GPU
    python benchmarks/autotune.py --dry-run    # show GPU key only
    python benchmarks/autotune.py --force      # re-profile even if cached
"""

import argparse
import hashlib
import json
import os
import time

import torch


def gpu_key():
    """Generate a unique key for the current GPU model + driver."""
    if not torch.cuda.is_available():
        return "cpu"

    props = torch.cuda.get_device_properties(0)
    raw = f"{props.name}|sm_{props.major}{props.minor}|{props.total_mem}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def cache_path():
    """Path to the cached autotune results."""
    cache_dir = os.path.expanduser("~/.cache/supergrok")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"autotune_{gpu_key()}.json")


def report_path():
    """Path to the human-readable report."""
    cache_dir = os.path.expanduser("~/.cache/supergrok")
    return os.path.join(cache_dir, f"autotune_{gpu_key()}_report.txt")


def load_cache():
    """Load cached autotune results, or return None."""
    path = cache_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_cache(results):
    """Save autotune results to cache."""
    path = cache_path()
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved JSON: {path}")


def save_report(results):
    """Save human-readable report alongside JSON cache."""
    path = report_path()
    lines = []
    lines.append("SuperGrok Auto-Tune Report")
    lines.append("=" * 50)
    lines.append(f"GPU: {results.get('_device', 'unknown')}")
    lines.append(f"Architecture: {results.get('_sm', 'unknown')}")
    lines.append(f"Timestamp: {results.get('_timestamp', 'unknown')}")
    lines.append(f"PSCAN_BLOCK: 512 (compile-time constexpr)")
    lines.append("")

    for section, data in results.items():
        if section.startswith("_"):
            continue
        lines.append(f"[{section}]")
        if isinstance(data, dict):
            if "error" in data:
                lines.append(f"  Error: {data['error']}")
            else:
                for k, v in data.items():
                    if isinstance(v, dict):
                        lines.append(f"  {k}:")
                        for k2, v2 in v.items():
                            lines.append(f"    {k2}: {v2}")
                    else:
                        lines.append(f"  {k}: {v}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved report: {path}")


def profile_scan_block_size(sizes=(128, 256, 512), num_steps=50):
    """Profile scan-related kernel throughput at different problem sizes.

    Note: This profiles the CURRENT compiled PSCAN_BLOCK (512).
    Changing PSCAN_BLOCK requires recompilation with a different constexpr.
    """
    import torch.nn as nn
    from grokking_optimizers import SuperGrok2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for hidden in sizes:
        model = nn.Sequential(
            nn.Linear(16, hidden), nn.ReLU(), nn.Linear(hidden, 10),
        ).to(device)
        opt = SuperGrok2(model.parameters(), lr=1e-3)
        x = torch.randn(32, 16, device=device)

        # Warmup
        for _ in range(10):
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        for _ in range(num_steps):
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        if device.type == 'cuda':
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            elapsed_ms = 0.0

        step_ms = elapsed_ms / num_steps
        total_params = sum(p.numel() for p in model.parameters())

        # Memory
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            for _ in range(5):
                loss = model(x).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_mb = 0.0

        results[str(hidden)] = {
            "step_ms": round(step_ms, 3),
            "total_params": total_params,
            "peak_mb": round(peak_mb, 1),
        }
        print(f"    hidden={hidden:<6}  step={step_ms:.3f} ms  "
              f"params={total_params}  mem={peak_mb:.1f} MB")

    return results


def profile_elem_block_size(sizes=(64, 128, 256, 512), num_steps=50):
    """Profile element-wise kernel throughput at different param counts."""
    import torch.nn as nn
    from grokking_optimizers import SuperGrok15

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for width in sizes:
        layers = []
        dim = 16
        for _ in range(4):
            layers.extend([nn.Linear(dim, width), nn.ReLU()])
            dim = width
        layers.append(nn.Linear(dim, 10))
        model = nn.Sequential(*layers).to(device)
        opt = SuperGrok15(model.parameters(), lr=1e-3)
        x = torch.randn(32, 16, device=device)

        total_params = sum(p.numel() for p in model.parameters())

        # Warmup
        for _ in range(10):
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        for _ in range(num_steps):
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        if device.type == 'cuda':
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            elapsed_ms = 0.0

        step_ms = elapsed_ms / num_steps
        throughput = total_params / (step_ms / 1000) if step_ms > 0 else 0
        results[str(width)] = {
            "step_ms": round(step_ms, 3),
            "total_params": total_params,
            "throughput": round(throughput),
        }
        print(f"    width={width:<6}  params={total_params:<8}  "
              f"step={step_ms:.3f} ms  throughput={throughput:,.0f} p/s")

    return results


def profile_projection_precision(num_steps=30):
    """Compare FP32 vs auto projection precision (TF32/BF16/FP8 where available)."""
    import torch.nn as nn
    from grokking_optimizers import SuperGrok2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        return {"error": "CUDA required"}

    results = {}
    for precision in ['fp32', 'auto']:
        try:
            model = nn.Sequential(
                nn.Linear(16, 256), nn.ReLU(), nn.Linear(256, 10),
            ).to(device)
            opt = SuperGrok2(model.parameters(), lr=1e-3,
                             projection_precision=precision)
            x = torch.randn(32, 16, device=device)

            for _ in range(10):
                loss = model(x).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            for _ in range(num_steps):
                loss = model(x).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()

            end.record()
            torch.cuda.synchronize()
            step_ms = start.elapsed_time(end) / num_steps
            results[precision] = round(step_ms, 3)
            print(f"    precision={precision:<6}  step={step_ms:.3f} ms")
        except Exception as e:
            results[precision] = {"error": str(e)}
            print(f"    precision={precision:<6}  ERROR: {e}")

    return results


def profile_overall(num_steps=30):
    """Profile overall step time, bilevel time, and memory."""
    import torch.nn as nn
    from grokking_optimizers import SuperGrok2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    model = nn.Sequential(
        nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 10),
    ).to(device)
    opt = SuperGrok2(model.parameters(), lr=1e-3)
    x = torch.randn(32, 16, device=device)

    # Warmup
    for _ in range(10):
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    for _ in range(num_steps):
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    if device.type == 'cuda':
        end.record()
        torch.cuda.synchronize()
        step_ms = start.elapsed_time(end) / num_steps
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        step_ms = 0.0
        peak_mb = 0.0

    results["step_ms"] = round(step_ms, 3)
    results["peak_mb"] = round(peak_mb, 1)
    results["total_params"] = sum(p.numel() for p in model.parameters())

    print(f"    step={step_ms:.3f} ms  mem={peak_mb:.1f} MB  "
          f"params={results['total_params']}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Auto-tune SuperGrok kernels')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show GPU key and cache path only')
    parser.add_argument('--force', action='store_true',
                        help='Re-profile even if cached results exist')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='Steps per profile run')
    args = parser.parse_args()

    key = gpu_key()
    path = cache_path()

    print(f"\n{'='*60}")
    print(f"  SuperGrok Auto-Tune")
    print(f"  GPU key: {key}")
    print(f"  Cache:   {path}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  Device:  {props.name} (sm_{props.major}{props.minor})")
        print(f"  Memory:  {props.total_mem / 1e9:.1f} GB")
    else:
        print(f"  Device:  CPU")

    print(f"\n  Note: PSCAN_BLOCK is a compile-time constexpr (512).")
    print(f"  This script profiles the current compiled configuration.")
    print(f"{'='*60}\n")

    if args.dry_run:
        cached = load_cache()
        if cached:
            print(f"  Cached results found ({len(cached)} entries)")
        else:
            print(f"  No cached results")
        return

    cached = load_cache()
    if cached and not args.force:
        print(f"  Using cached results. Run with --force to re-profile.\n")
        for section, data in cached.items():
            if section.startswith("_"):
                continue
            print(f"  {section}:")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"    {k}: {v}")
            print()
        return

    results = {
        "_gpu_key": key,
        "_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        results["_device"] = props.name
        results["_sm"] = f"sm_{props.major}{props.minor}"

    print("  Profiling overall performance (SuperGrok v2)...")
    try:
        results["overall"] = profile_overall(num_steps=args.num_steps)
    except Exception as e:
        print(f"    SKIP: {e}")
        results["overall"] = {"error": str(e)}

    print()
    print("  Profiling scan kernel (SuperGrok v2)...")
    try:
        results["scan_profile"] = profile_scan_block_size(
            num_steps=args.num_steps)
    except Exception as e:
        print(f"    SKIP: {e}")
        results["scan_profile"] = {"error": str(e)}

    print()
    print("  Profiling elem_step kernel (SuperGrok v1.5)...")
    try:
        results["elem_profile"] = profile_elem_block_size(
            num_steps=args.num_steps)
    except Exception as e:
        print(f"    SKIP: {e}")
        results["elem_profile"] = {"error": str(e)}

    print()
    print("  Profiling projection precision (FP32 vs auto)...")
    try:
        results["precision_profile"] = profile_projection_precision(
            num_steps=min(args.num_steps, 30))
    except Exception as e:
        print(f"    SKIP: {e}")
        results["precision_profile"] = {"error": str(e)}

    print()
    save_cache(results)
    save_report(results)
    print()


if __name__ == '__main__':
    main()
