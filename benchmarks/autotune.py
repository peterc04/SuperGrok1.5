#!/usr/bin/env python3
"""Auto-tune kernel launch configurations per GPU model.

Profiles block sizes for scan and elem_step kernels, caches results
in ~/.cache/supergrok/autotune_<gpu_key>.json.

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
    print(f"  Saved: {path}")


def profile_scan_block_size(sizes=(128, 256, 512), num_steps=50):
    """Profile scan-related kernel throughput at different problem sizes.

    Tests optimizer step time with varying model hidden dimensions to
    determine optimal configurations for the Mamba-3 scan kernels.
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
        results[str(hidden)] = round(step_ms, 3)
        print(f"    hidden={hidden:<6}  step={step_ms:.3f} ms")

    return results


def profile_elem_block_size(sizes=(64, 128, 256, 512), num_steps=50):
    """Profile element-wise kernel throughput at different param counts.

    Tests optimizer step with varying numbers of parameters to determine
    optimal block sizes for the fused element-step kernel.
    """
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
    save_cache(results)
    print()


if __name__ == '__main__':
    main()
