#!/usr/bin/env python3
"""Test all NVIDIA architecture tiers on one GPU using FORCE_ARCH.

Each tier sets SUPERGROK_FORCE_ARCH and runs the test matrix.

Usage:
    python tests/test_all_tiers.py
"""

import subprocess
import sys
import os

TIERS = [
    ('Generic (sm_75)', '75'),
    ('Ampere (sm_80)',  '80'),
    ('Hopper (sm_90)',  '90'),
]


def run_tier(name, arch):
    """Run test_matrix.py with FORCE_ARCH set."""
    env = os.environ.copy()
    env['SUPERGROK_FORCE_ARCH'] = arch

    print(f"\n{'='*60}")
    print(f"  TIER: {name}  (SUPERGROK_FORCE_ARCH={arch})")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), 'test_matrix.py')],
        env=env,
        capture_output=False,
    )
    return result.returncode == 0


def main():
    all_pass = True
    tier_results = []

    for name, arch in TIERS:
        ok = run_tier(name, arch)
        tier_results.append((name, ok))
        if not ok:
            all_pass = False

    print(f"\n{'='*60}")
    print(f"  Tier Summary")
    print(f"{'='*60}")
    for name, ok in tier_results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")

    if all_pass:
        print(f"\n  ALL TIERS PASSED")
    else:
        print(f"\n  SOME TIERS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
