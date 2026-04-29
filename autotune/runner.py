"""Microbenchmark runner for autotune.

Given a (kernel_name, arch, shape_bucket, axis_dict), build a tiny test
extension that compiles the kernel under those parameters and times it
under representative inputs. Returns the median latency in microseconds.

This file is the harness. The actual kernel template is in the per-arch
.cu file (e.g. ``csrc/kernels/cuda/sm_90/grokadamw_sm90.cu``); the
template parameters are populated from the axis_dict.

NOTE: this is the autotune scaffolding, not the actual benchmark code.
The runner needs hardware to run; this stub establishes the interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchResult:
    median_us: float
    min_us: float
    max_us: float
    iterations: int


def bench(
    kernel_name: str,
    arch: int,
    shape_bucket: Any,
    axis_dict: dict,
    *,
    warmup: int = 10,
    iters: int = 100,
) -> BenchResult:
    """Run a single (kernel, arch, shape, axes) configuration.

    The host must have a GPU of the requested arch (or the autotuner
    must be running under FORCE_ARCH).

    The flow:
      1. Pick the per-arch source file based on `arch`.
      2. Generate a small C++ wrapper that calls the kernel with the
         template parameters from `axis_dict`.
      3. Build via torch.utils.cpp_extension (or a CUTLASS profiler
         invocation for the GEMM grids).
      4. Allocate test tensors of `shape_bucket`.
      5. Warmup, then time `iters` iterations with CUDA events.
      6. Return median / min / max latency in microseconds.

    For now, this is a stub that documents the interface.
    """
    raise NotImplementedError(
        "autotune/runner.py is scaffolding. Wire it to torch.utils.cpp_extension "
        "and the per-arch kernel templates when running on hardware."
    )


def bench_many(grid: list, *, warmup: int = 10, iters: int = 100) -> list:
    """Run the entire grid; return list of (config, BenchResult)."""
    results = []
    for cfg in grid:
        r = bench(
            kernel_name=cfg["kernel_name"],
            arch=cfg["arch"],
            shape_bucket=cfg["shape_bucket"],
            axis_dict=cfg["axes"],
            warmup=warmup,
            iters=iters,
        )
        results.append((cfg, r))
    return results
