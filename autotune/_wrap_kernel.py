#!/usr/bin/env python3
"""
Helper for the all-specialized refactor.

Reads a generic kernel from csrc/cuda/generic/, wraps its body in
'namespace sg::<arch> { ... }', adds a banner, and writes to the
target arch directory.

The namespace wrapping isolates per-arch kernel and launcher symbols
so the four arch translation units don't collide at link time.

Usage: python3 _wrap_kernel.py <src> <dst> <arch> <other_archs>
  e.g.  python3 _wrap_kernel.py csrc/cuda/generic/grokadamw_kernels.cu \\
                                csrc/kernels/cuda/sm_90/grokadamw_sm90.cu \\
                                sm90 "sm_80, sm_100, gfx942"
"""
import sys
from pathlib import Path

src_path = Path(sys.argv[1])
dst_path = Path(sys.argv[2])
arch = sys.argv[3]              # sm90 / sm80 / sm100 / gfx942
other_archs_str = sys.argv[4]   # "sm_80, sm_100, gfx942"

src = src_path.read_text()
lines = src.split("\n")

# Find the last #include in the first 150 lines (before kernel code starts).
last_include_idx = -1
for i, line in enumerate(lines[:150]):
    if line.strip().startswith("#include"):
        last_include_idx = i

insert_after = last_include_idx + 1 if last_include_idx >= 0 else 0

banner = [
    "",
    "// =====================================================================",
    f"//  BASELINE COPY -- {arch} variant of {src_path.name}",
    "//",
    f"//  Math is identical to the other arch variants ({other_archs_str}).",
    "//  Arch-specific primitive divergence (cp.async / TMA / MFMA / FP8 /",
    "//  warp specialization) is deferred to hardware-validated tuning passes.",
    "//  Cross-arch numerical agreement is guarded by",
    "//  tests/test_cross_arch_agreement.py.",
    "//",
    f"//  Origin: csrc/cuda/generic/{src_path.name}",
    "// =====================================================================",
    "",
    f"namespace sg {{ namespace {arch} {{",
    "",
]
footer = ["", f"}} }} // namespace sg::{arch}", ""]

new_lines = lines[:insert_after] + banner + lines[insert_after:] + footer
dst_path.parent.mkdir(parents=True, exist_ok=True)
dst_path.write_text("\n".join(new_lines))
print(f"wrote {dst_path} ({len(new_lines)} lines)")
