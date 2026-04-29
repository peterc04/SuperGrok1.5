"""Wrapper around the CUTLASS profiler for SG2 + Muon GEMMs.

CUTLASS (NVIDIA) provides a profiler binary that explores tile shapes,
pipeline stages, and split-K configurations. We use it for:

  - SG v2 projection GEMMs (in_proj, dt_proj+softplus, B_proj, C_proj)
  - Muon Newton-Schulz GEMMs (X · Xᵀ · X)

These are the only GEMM-heavy paths in the codebase; everything else is
hand-rolled element-wise / per-element MLP work where CUTLASS provides no
benefit.

This file is scaffolding. The actual workflow:
  1. Locate cutlass_profiler binary (in $CUTLASS_HOME or from a build).
  2. For each (operation, shape, dtype), invoke the profiler with the
     appropriate flags.
  3. Parse the CSV output, find the highest-throughput config that meets
     the numerical-accuracy requirement.
  4. Emit the winning config into csrc/common/tuned_configs.h alongside
     hand-rolled kernel configs.

For the dt_proj softplus epilogue, CUTLASS does not have a stock softplus
epilogue, so the autotuner emits a custom epilogue header next to
tuned_configs.h that declares the fused-softplus variant; the SG2
forward kernel `#include`s both.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CutlassConfig:
    operation: str
    shape: tuple
    dtype_in: str
    dtype_out: str
    dtype_acc: str
    threadblock_shape: tuple
    warp_shape: tuple
    instruction_shape: tuple
    stages: int
    split_k_slices: int
    epilogue: str = "linear_combination"


def find_cutlass_profiler() -> Path:
    """Find the cutlass_profiler binary."""
    cutlass_home = os.environ.get("CUTLASS_HOME")
    if cutlass_home:
        candidate = Path(cutlass_home) / "build" / "tools" / "profiler" / "cutlass_profiler"
        if candidate.exists():
            return candidate
    found = shutil.which("cutlass_profiler")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "cutlass_profiler not found. Set CUTLASS_HOME or add the binary to PATH. "
        "Build CUTLASS with -DCUTLASS_ENABLE_TOOLS=ON.")


def profile_gemm(
    M: int, N: int, K: int,
    *,
    dtype_in: str = "f16",
    dtype_out: str = "f16",
    dtype_acc: str = "f32",
    arch: int = 90,
    timeout_s: int = 600,
) -> list[CutlassConfig]:
    """Run cutlass_profiler over the GEMM tile-shape space and return all
    candidate configs sorted by GFLOPS, fastest first.

    Supported arch targets:
      80  -- Ampere (TF32, FP16, BF16; cuBLAS or CUTLASS sm_80)
      89  -- Ada (FP8 E4M3/E5M2; CUTLASS sm_89; no TMA, no thread block clusters)
      90  -- Hopper (FP8 + warp specialization; CUTLASS sm_90)
      100 -- Datacenter Blackwell (NVFP4 + TMA; CUTLASS sm_100)
      103 -- Blackwell Ultra (NVFP4 with 1.5x throughput; CUTLASS sm_103a)
      120 -- Consumer Blackwell (CUTLASS sm_120a; smaller shared memory)
    """
    SUPPORTED = {80, 89, 90, 100, 103, 120}
    if arch not in SUPPORTED:
        raise ValueError(f"arch={arch} not in CUTLASS supported set {SUPPORTED}")
    raise NotImplementedError(
        "autotune/cutlass_profile.py is scaffolding. Wire to cutlass_profiler "
        "and parse its CSV output when running on hardware.")


def profile_gemm_with_softplus_epilogue(
    M: int, N: int, K: int, **kwargs
) -> list[CutlassConfig]:
    """SG v2 dt_proj uses a fused softplus(x + bias) epilogue.

    CUTLASS does not have a stock softplus epilogue. Two paths:
      (a) Define a custom EpilogueOp using cutlass::epilogue::thread::* and
          register it before invoking the profiler.
      (b) Use the linear_combination epilogue and append a tiny pointwise
          softplus_bias kernel.

    The autotuner picks (a) if the custom op compiles successfully on the
    target arch (sm_90 / sm_100), else falls back to (b).
    """
    raise NotImplementedError(
        "autotune/cutlass_profile.py: SG2 dt_proj softplus epilogue is "
        "scaffolding. See REFACTOR_PLAN.md §5 for the migration plan.")
