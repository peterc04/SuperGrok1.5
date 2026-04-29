#!/usr/bin/env python3
"""Autotune entry point.

Runs the per-kernel parameter grid for the requested arch, picks the
median-fastest config per (kernel, shape bucket), and writes the result
to ``csrc/common/tuned_configs.h``.

Usage:
    python autotune/tune.py --arch sm_90
    python autotune/tune.py --arch all
    python autotune/tune.py --kernel grokadamw_step --arch sm_90 --dry-run
    python autotune/tune.py --output csrc/common/tuned_configs.h

Requires hardware of the target arch (or FORCE_ARCH override). Output is
committed to the repo.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from autotune.grids import GRIDS  # noqa: E402

SUPPORTED_ARCHES = ("sm_80", "sm_90", "sm_100", "gfx942")


def expand_grid(axes: dict) -> list[dict]:
    """Cartesian product of the axis dict."""
    keys = list(axes.keys())
    combos = [dict(zip(keys, values))
              for values in itertools.product(*axes.values())]
    return combos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="sm_90",
                   help="Arch to tune for: sm_80, sm_90, sm_100, gfx942, or 'all'")
    p.add_argument("--kernel", default=None,
                   help="Tune a single kernel; default is all")
    p.add_argument("--output", default="csrc/common/tuned_configs.h",
                   help="Output header path")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the grid without running")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def select_arches(arch: str) -> list[str]:
    if arch == "all":
        return list(SUPPORTED_ARCHES)
    if arch in SUPPORTED_ARCHES:
        return [arch]
    raise SystemExit(f"--arch must be one of {SUPPORTED_ARCHES} or 'all'")


def select_kernels(kernel: str | None) -> list[str]:
    if kernel:
        if kernel not in GRIDS:
            raise SystemExit(
                f"unknown kernel {kernel!r}; available: {sorted(GRIDS)}")
        return [kernel]
    return sorted(GRIDS.keys())


def grid_size(spec: dict) -> int:
    if isinstance(spec.get("axes"), str):
        return -1  # delegated to cutlass profiler
    n = 1
    for vs in spec["axes"].values():
        n *= len(vs)
    return n * len(spec["shape_buckets"])


def main() -> int:
    args = parse_args()
    arches = select_arches(args.arch)
    kernels = select_kernels(args.kernel)

    print(f"# arches:  {arches}")
    print(f"# kernels: {len(kernels)}")
    print(f"# output:  {args.output}")

    total_configs = 0
    for k in kernels:
        spec = GRIDS[k]
        size = grid_size(spec)
        total_configs += size if size > 0 else 0
        kind = "cutlass" if size < 0 else f"{size} configs"
        print(f"  - {k:40s} ({kind})")

    if args.dry_run:
        print(f"\n# total non-cutlass grid points: {total_configs}")
        return 0

    print("\nERROR: tune.py is scaffolding. The runner needs a hardware-equipped "
          "session to compile and time the per-arch kernel templates. See "
          "autotune/README.md for the full design.\n"
          "Wire this script up by:\n"
          "  1. Implementing autotune/runner.py:bench() against torch.utils.cpp_extension\n"
          "  2. Implementing autotune/cutlass_profile.py:profile_gemm() against the\n"
          "     cutlass_profiler binary\n"
          "  3. Implementing the writer that emits csrc/common/tuned_configs.h\n"
          "Until then, the build uses the default configs in tuned_configs.h.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
