#!/usr/bin/env python3
"""Test all supported arches on one GPU using FORCE_ARCH.

Drives test_matrix.py once per supported arch in {sm_80, sm_90, sm_100, gfx942}.
Skips arches whose binding is not built into the local extension.

Replaces the old test_all_tiers.py which tested a generic-tier fallback;
under the all-specialized policy there is no generic tier.

Usage:
    python tests/test_all_arches.py
"""

import os
import subprocess
import sys

# Supported set per the all-specialized refactor. See dispatch.py /
# csrc/kernels/README.md / REFACTOR_PLAN.md.
ARCHES = [
    ('Ampere family (sm_80 binding)',           '80'),
    ('Ada / RTX 40, L40 (sm_89)',               '89'),
    ('Hopper (sm_90)',                          '90'),
    ('Datacenter Blackwell (sm_100)',           '100'),
    ('Blackwell Ultra / B300 (sm_103)',         '103'),
    ('Consumer Blackwell / RTX 50 (sm_120)',    '120'),
    ('AMD MI300X (gfx942)',                     '942'),
    ('AMD MI350X / MI355X (gfx950)',            '950'),
]


def _arch_available(arch: str) -> bool:
    """Probe whether the binding for `arch` is in the local extension.

    Pragmatic check: try to import _ops, set FORCE_ARCH, and read back
    detect_arch(). On any failure (binding missing, no GPU, etc.) skip.
    """
    env = os.environ.copy()
    env['FORCE_ARCH'] = arch
    probe = (
        "import sys;"
        "from grokking_optimizers._ops_loader import get_ops;"
        "ops = get_ops();"
        f"sys.exit(0 if ops.detect_arch() == {int(arch)} else 1)"
    )
    try:
        r = subprocess.run([sys.executable, "-c", probe], env=env,
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def run_arch(name: str, arch: str) -> tuple[bool, str]:
    """Run test_matrix.py with FORCE_ARCH set."""
    env = os.environ.copy()
    env['FORCE_ARCH'] = arch

    print(f"\n{'='*60}")
    print(f"  ARCH: {name}  (FORCE_ARCH={arch})")
    print(f"{'='*60}")

    if not _arch_available(arch):
        print(f"  SKIP: binding for sm_{arch} not in local extension")
        return True, "skipped"

    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), 'test_matrix.py')],
        env=env,
        capture_output=False,
    )
    return result.returncode == 0, "pass" if result.returncode == 0 else "fail"


def main():
    all_pass = True
    summary = []

    for name, arch in ARCHES:
        ok, status = run_arch(name, arch)
        summary.append((name, status))
        if not ok:
            all_pass = False

    print(f"\n{'='*60}")
    print(f"  Per-arch summary")
    print(f"{'='*60}")
    for name, status in summary:
        print(f"  {status:8s}  {name}")

    if all_pass:
        print("\n  ALL AVAILABLE ARCHES PASSED")
    else:
        print("\n  SOME ARCHES FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
