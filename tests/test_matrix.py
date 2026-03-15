#!/usr/bin/env python3
"""Cross-platform test matrix.

Runs the optimizer step for every supported (backend, precision, optimizer)
combination. Reports PASS/FAIL/SKIP for each.

Usage:
    python tests/test_matrix.py                    # test current GPU
    SUPERGROK_FORCE_ARCH=80 python tests/test_matrix.py  # test Ampere tier
    SUPERGROK_FORCE_ARCH=0 python tests/test_matrix.py   # test CPU path
"""

import os
import sys
import time
import traceback
import torch
import torch.nn as nn

# ─── Configuration ──────────────────────────────────────────────────

OPTIMIZERS = [
    'SuperGrok2', 'SuperGrok15', 'SuperGrok11', 'NeuralGrok',
    'GrokAdamW', 'Lion', 'Grokfast', 'Prodigy', 'Muon', 'LookSAM',
]

# ─── Helpers ────────────────────────────────────────────────────────

def make_model(device='cpu', dtype=torch.float32):
    """Tiny model for testing."""
    m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10))
    return m.to(device=device, dtype=dtype)


def make_optimizer(name, model):
    """Create an optimizer by name, matching actual constructor APIs."""
    from grokking_optimizers import (
        SuperGrok2, SuperGrok15, SuperGrok11, NeuralGrok,
        GrokAdamW, Lion, Grokfast, Prodigy, Muon, LookSAM,
    )

    if name == 'SuperGrok2':
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
        # Muon takes params_2d (2D weights) and params_1d (biases etc.)
        params_2d = [p for p in model.parameters() if p.dim() >= 2]
        params_1d = [p for p in model.parameters() if p.dim() < 2]
        return Muon(params_2d, params_1d, lr=0.02)
    elif name == 'LookSAM':
        return LookSAM(model.parameters(), lr=1e-3)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def run_single_test(opt_name, device, num_steps=5):
    """Run one optimizer. Returns (status, time_ms, error)."""
    try:
        model = make_model(device=device)
        opt = make_optimizer(opt_name, model)

        t0 = time.time()
        for step in range(num_steps):
            x = torch.randn(32, 16, device=device)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        elapsed = (time.time() - t0) * 1000  # ms

        # Check for NaN
        for p in model.parameters():
            if torch.isnan(p).any():
                return 'NaN', elapsed, 'NaN in parameters'

        return 'PASS', elapsed, ''

    except Exception as e:
        tb = traceback.format_exc()
        if 'not supported' in str(e).lower() or 'not available' in str(e).lower():
            return 'SKIP', 0, str(e)
        return 'FAIL', 0, f"{e}\n{tb}"


# ─── Main ───────────────────────────────────────────────────────────

def run_matrix():
    """Run the full test matrix."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from grokking_optimizers.dispatch import get_arch_label

    print(f"\n{'='*70}")
    print(f"  Cross-Platform Test Matrix")
    print(f"  Device: {device} ({get_arch_label()})")
    print(f"  FORCE_ARCH: {os.environ.get('SUPERGROK_FORCE_ARCH', 'not set')}")
    print(f"{'='*70}\n")

    results = []
    total = passed = failed = skipped = nan_count = 0

    for opt_name in OPTIMIZERS:
        total += 1
        status, elapsed, error = run_single_test(opt_name, device)
        results.append((opt_name, status, elapsed, error))

        if status == 'PASS':
            passed += 1
            print(f"  PASS  {opt_name:<50} {elapsed:>7.1f}ms")
        elif status == 'SKIP':
            skipped += 1
            print(f"  SKIP  {opt_name:<50} {error[:50]}")
        elif status == 'NaN':
            nan_count += 1
            print(f"  NaN!  {opt_name:<50} {error}")
        else:
            failed += 1
            print(f"  FAIL  {opt_name:<50} {error[:80]}")

    # Summary
    print(f"\n{'─'*70}")
    print(f"  Total: {total}  |  Pass: {passed}  |  Fail: {failed}  |  "
          f"Skip: {skipped}  |  NaN: {nan_count}")
    print(f"{'─'*70}\n")

    return failed == 0 and nan_count == 0


if __name__ == '__main__':
    success = run_matrix()
    sys.exit(0 if success else 1)
