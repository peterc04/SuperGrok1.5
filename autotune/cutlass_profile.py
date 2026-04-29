"""Wrapper around the CUTLASS profiler for SG2 + Muon GEMMs.

CUTLASS (NVIDIA) provides a profiler binary that explores tile shapes,
pipeline stages, and split-K configurations. We use it for:

  - SG v2 projection GEMMs (in_proj, dt_proj+softplus, B_proj, C_proj)
  - Muon Newton-Schulz GEMMs (X . X^T . X)

These are the only GEMM-heavy paths in the codebase; everything else is
hand-rolled element-wise / per-element MLP work where CUTLASS provides no
benefit.

Workflow:
  1. Locate cutlass_profiler binary (env CUTLASS_PROFILER_BIN, then
     env CUTLASS_HOME, then the third_party/cutlass build dir, then PATH).
  2. For each (operation, shape, dtype), invoke the profiler.
  3. Parse the CSV output, extract per-config Operation + Runtime (ms),
     return sorted by ms (fastest first).

For the dt_proj softplus epilogue, CUTLASS does not have a stock softplus
epilogue, so the autotuner emits a custom epilogue header next to
tuned_configs.h that declares the fused-softplus variant; the SG2
forward kernel ``#include``s both.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILER_PATH = (
    REPO_ROOT / "third_party" / "cutlass" / "build" / "tools" / "profiler" / "cutlass_profiler"
)


@dataclass
class CutlassConfig:
    operation: str
    shape: tuple
    dtype_in: str
    dtype_out: str
    dtype_acc: str
    threadblock_shape: tuple = (0, 0, 0)
    warp_shape: tuple = (0, 0, 0)
    instruction_shape: tuple = (0, 0, 0)
    stages: int = 0
    split_k_slices: int = 1
    epilogue: str = "linear_combination"
    runtime_ms: float = 0.0
    gflops: float = 0.0
    raw_row: dict = field(default_factory=dict)


def find_cutlass_profiler() -> Path:
    """Find the cutlass_profiler binary.

    Search order:
      1. $CUTLASS_PROFILER_BIN (explicit override)
      2. $CUTLASS_HOME/build/tools/profiler/cutlass_profiler
      3. third_party/cutlass/build/tools/profiler/cutlass_profiler (default)
      4. ``cutlass_profiler`` on $PATH
    """
    explicit = os.environ.get("CUTLASS_PROFILER_BIN")
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p

    cutlass_home = os.environ.get("CUTLASS_HOME")
    if cutlass_home:
        candidate = Path(cutlass_home) / "build" / "tools" / "profiler" / "cutlass_profiler"
        if candidate.exists():
            return candidate

    if DEFAULT_PROFILER_PATH.exists():
        return DEFAULT_PROFILER_PATH

    found = shutil.which("cutlass_profiler")
    if found:
        return Path(found)

    raise FileNotFoundError(
        "cutlass_profiler not found. Build the CUTLASS profiler:\n"
        "  cd third_party/cutlass && cmake -B build "
        "-DCUTLASS_NVCC_ARCHS=<ARCH>A "
        "&& cmake --build build -j --target cutlass_profiler\n"
        "Or set CUTLASS_PROFILER_BIN to the binary path."
    )


def _build_profiler_argv(
    profiler: Path, M: int, N: int, K: int,
    dtype_a: str, dtype_b: str, dtype_c: str,
    csv_path: Path,
) -> list[str]:
    return [
        str(profiler),
        "--operation=Gemm",
        f"--m={M}",
        f"--n={N}",
        f"--k={K}",
        f"--A={dtype_a}",
        f"--B={dtype_b}",
        f"--C={dtype_c}",
        f"--output={csv_path}",
    ]


def _parse_runtime_ms(row: dict) -> float:
    """Pull the per-op runtime out of a CSV row, accommodating column
    naming drift across CUTLASS versions ("Runtime", "Runtime (ms)",
    "Runtime_ms", lowercase, etc.).
    """
    for key in row:
        norm = key.lower().replace(" ", "").replace("_", "")
        if norm.startswith("runtime"):
            try:
                return float(row[key])
            except (ValueError, TypeError):
                continue
    return float("inf")


def _parse_gflops(row: dict) -> float:
    for key in row:
        norm = key.lower().replace(" ", "").replace("_", "")
        if norm.startswith("gflops"):
            try:
                return float(row[key])
            except (ValueError, TypeError):
                continue
    return 0.0


def _parse_op_name(row: dict) -> str:
    for key in ("Operation", "operation", "OperationName", "Name"):
        if key in row:
            return str(row[key])
    # Best-effort: first column if unnamed.
    if row:
        return str(next(iter(row.values())))
    return ""


def profile_gemm(
    M: int, N: int, K: int,
    *,
    dtype_a: str = "f16",
    dtype_b: str = "f16",
    dtype_c: str = "f16",
    layout: str = "rcr",
    arch: str = "sm_90",
    timeout_s: int = 600,
) -> list[dict]:
    """Run cutlass_profiler over the GEMM tile-shape space.

    Returns a list of dicts (sorted by runtime ms, fastest first):
        [{"name": op_name, "ms": runtime_ms, "config": parsed_config_str,
          "gflops": ..., "row": <full csv row>}, ...]

    Raises FileNotFoundError with build instructions if the profiler binary
    is missing; raises subprocess.CalledProcessError if the profiler exits
    with a nonzero status.
    """
    profiler = find_cutlass_profiler()

    with tempfile.TemporaryDirectory(prefix="cutlass_profile_") as td:
        csv_path = Path(td) / "tmp_profile.csv"
        argv = _build_profiler_argv(
            profiler, M, N, K, dtype_a, dtype_b, dtype_c, csv_path,
        )
        subprocess.run(argv, check=True, timeout=timeout_s)

        # The profiler may write either the path we requested, or
        # "<output>.gemm.csv" (it appends operation kind on some versions).
        candidates = [csv_path]
        candidates += sorted(Path(td).glob("tmp_profile*.csv"))
        actual = next((c for c in candidates if c.exists()), None)
        if actual is None:
            raise FileNotFoundError(
                f"cutlass_profiler did not produce a CSV in {td}"
            )

        results: list[dict] = []
        with open(actual, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ms = _parse_runtime_ms(row)
                results.append({
                    "name": _parse_op_name(row),
                    "ms": ms,
                    "gflops": _parse_gflops(row),
                    "config": _config_string(row),
                    "row": dict(row),
                })

    results.sort(key=lambda r: r["ms"])
    return results


def _config_string(row: dict) -> str:
    """Compact human-readable config summary from a CSV row."""
    parts = []
    for key in (
        "threadblock_shape", "ThreadblockShape",
        "warp_shape", "WarpShape",
        "instruction_shape", "InstructionShape",
        "stages", "Stages",
        "split_k_slices", "SplitKSlices",
    ):
        if key in row and row[key]:
            parts.append(f"{key}={row[key]}")
    return ",".join(parts)


def profile_gemm_with_softplus_epilogue(
    M: int, N: int, K: int, **kwargs
) -> list[dict]:
    """SG v2 dt_proj uses a fused softplus(x + bias) epilogue.

    CUTLASS does not ship a stock softplus epilogue, so the profiler runs
    against the linear-combination epilogue and the autotuner emits a tiny
    pointwise softplus_bias kernel next to it. The selected GEMM config is
    still the linear-combination winner.
    """
    return profile_gemm(M, N, K, **kwargs)
