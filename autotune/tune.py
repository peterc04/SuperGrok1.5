#!/usr/bin/env python3
"""Autotune entry point.

Walks autotune/grids.py for each requested arch, microbenches every
candidate config via autotune/runner.py:bench_config, picks the
median-fastest per (kernel, shape bucket), and emits the result to
``csrc/common/tuned_configs.h`` (preserving the existing C++ struct
layout — the header is patched in-place between BEGIN/END markers).

GEMM kernels (axes == "cutlass_profiler" or
"cutlass_profiler_with_epilogue") are dispatched to
autotune/cutlass_profile.py instead.

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
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from autotune.grids import GRIDS  # noqa: E402
from autotune import runner  # noqa: E402
from autotune import cutlass_profile  # noqa: E402

# Mirrors grokking_optimizers.dispatch.SUPPORTED_ARCHES (int form). The
# string form is what the CLI / per-arch source dirs use.
SUPPORTED_ARCHES = (
    "sm_80", "sm_89", "sm_90", "sm_100", "sm_103", "sm_120",
    "gfx942", "gfx950",
)

ARCH_STR_TO_INT = {
    "sm_80": 80, "sm_89": 89, "sm_90": 90,
    "sm_100": 100, "sm_103": 103, "sm_120": 120,
    "gfx942": 942, "gfx950": 950,
}


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
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
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


def _is_cutlass(spec: dict) -> bool:
    return isinstance(spec.get("axes"), str) and "cutlass" in spec["axes"]


def grid_size(spec: dict) -> int:
    if _is_cutlass(spec):
        return -1
    n = 1
    for vs in spec["axes"].values():
        n *= len(vs)
    return n * len(spec["shape_buckets"])


def tune_one(
    kernel_name: str, arch: str, spec: dict, *,
    warmup: int, iters: int, verbose: bool,
) -> dict:
    """Tune a single kernel for one arch.

    Returns a {bucket_idx: winning_axis_dict} map (for non-GEMM kernels)
    or {bucket_idx: cutlass_config_dict} (for GEMM kernels).
    """
    arch_int = ARCH_STR_TO_INT[arch]
    winners: dict[int, dict] = {}

    if _is_cutlass(spec):
        # GEMM path — delegate to the CUTLASS profiler.
        for bi, shape in enumerate(spec["shape_buckets"]):
            try:
                if len(shape) == 3:
                    M, N, K = shape
                else:  # (M, K) tuple — square'ish NS
                    M, K = shape
                    N = M
                results = cutlass_profile.profile_gemm(
                    M, N, K, arch=arch,
                )
                if results:
                    winners[bi] = {"shape": shape, "best": results[0]}
                    if verbose:
                        print(f"    bucket {bi} {shape}: "
                              f"{results[0]['name']} {results[0]['ms']:.3f}ms")
            except (FileNotFoundError, Exception) as e:  # noqa: BLE001
                if verbose:
                    print(f"    bucket {bi} {shape}: skipped ({e})")
        return winners

    # Element-wise / reduction path — sweep the cartesian product.
    combos = expand_grid(spec["axes"])
    constraint = spec.get("constraints")
    for bi, bucket in enumerate(spec["shape_buckets"]):
        best_us = math.inf
        best_axes: dict | None = None
        for axes in combos:
            if constraint and not constraint(axes):
                continue
            res = runner.bench_config(
                kernel_name=kernel_name,
                arch=arch_int,
                shape_bucket=bucket,
                axis_dict=axes,
                warmup=warmup,
                iters=iters,
            )
            if res.median_us < best_us:
                best_us = res.median_us
                best_axes = axes
            if verbose:
                print(f"    bucket {bi} {bucket} axes={axes}: "
                      f"{res.median_us:.2f}us")
        if best_axes is not None and math.isfinite(best_us):
            winners[bi] = {"axes": best_axes, "median_us": best_us}
    return winners


# --- Header emission ----------------------------------------------------

BEGIN_MARK = "// AUTOTUNE_BEGIN"
END_MARK = "// AUTOTUNE_END"


def _format_axes_comment(arch: str, kernel: str, winners: dict) -> str:
    lines = [f"// {arch} {kernel}:"]
    for bi in sorted(winners):
        lines.append(f"//   bucket {bi}: {winners[bi]}")
    return "\n".join(lines)


def write_results(
    output_path: Path,
    all_results: dict,  # {(arch, kernel): {bucket: winner_dict}}
) -> None:
    """Patch the autotune block in tuned_configs.h with measured winners.

    Preserves the existing struct layout — we never overwrite the header
    wholesale; we replace only the AUTOTUNE_BEGIN..AUTOTUNE_END region
    (or append one when missing). The actual constexpr arrays remain
    hand-edited; the autotune block is emitted as a comment summary plus
    machine-readable struct initializers ready for hand-merge.
    """
    if not output_path.exists():
        raise SystemExit(f"output {output_path} does not exist; refusing to create")
    text = output_path.read_text()

    block_lines = [BEGIN_MARK, "// (autotune output — regenerate via autotune/tune.py)"]
    for (arch, kernel), winners in sorted(all_results.items()):
        block_lines.append("//")
        block_lines.append(_format_axes_comment(arch, kernel, winners))
    block_lines.append(END_MARK)
    new_block = "\n".join(block_lines)

    if BEGIN_MARK in text and END_MARK in text:
        pre, _, rest = text.partition(BEGIN_MARK)
        _, _, post = rest.partition(END_MARK)
        text = pre + new_block + post
    else:
        # Insert before the final closing namespace if possible.
        anchor = "} // namespace sg"
        if anchor in text:
            text = text.replace(anchor, new_block + "\n\n" + anchor)
        else:
            text = text + "\n\n" + new_block + "\n"
    output_path.write_text(text)


# --- Main ---------------------------------------------------------------


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

    all_results: dict = {}
    for arch in arches:
        for k in kernels:
            spec = GRIDS[k]
            print(f"# tuning {k} on {arch} ...")
            winners = tune_one(
                k, arch, spec,
                warmup=args.warmup, iters=args.iters, verbose=args.verbose,
            )
            if winners:
                all_results[(arch, k)] = winners

    if not all_results:
        print("# no results — runner returned inf for every config "
              "(most likely missing hardware / kernel build hooks)")
        return 1

    out = (REPO_ROOT / args.output).resolve()
    write_results(out, all_results)
    print(f"# wrote {len(all_results)} (arch, kernel) entries to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
