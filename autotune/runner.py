"""Microbenchmark runner for autotune.

Times a kernel callable under a candidate `LaunchConfig`, returning the
median latency in microseconds. Uses CUDA events with a single CPU sync
per measurement; warmup of 3-5 iters is recommended.

The high-level entry points:

    bench(kernel_fn, args, iters=20, warmup=5) -> float
        Microseconds, median across ``iters`` (inf on CUDA failure).

    bench_config(kernel_name, arch, shape_bucket, axis_dict, ...) -> BenchResult
        Higher-level wrapper used by the orchestrator. Builds the kernel
        callable + test inputs from a (kernel_name, arch, shape_bucket,
        axis_dict) tuple, then delegates to ``bench``.

The kernel-callable construction (cpp_extension build with the axis_dict
template parameters) is delegated to ``_build_kernel`` which is a hook the
build-system layer fills in once the per-arch templates land. Until that
hook returns a real callable, ``bench_config`` falls back to ``inf`` so
``tune.py`` can iterate the grid without crashing.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass
class BenchResult:
    median_us: float
    min_us: float
    max_us: float
    iterations: int


def bench(
    kernel_fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    iters: int = 20,
    warmup: int = 5,
) -> float:
    """Time ``kernel_fn(*args)`` and return the median latency in microseconds.

    Implementation:
      * ``warmup`` un-timed iterations, then a single ``cuda.synchronize()``.
      * One ``(start, stop)`` CUDA event pair per iteration; events are
        recorded back-to-back without intervening syncs.
      * After all iterations are submitted, a single ``cuda.synchronize()``
        and then ``elapsed_time`` is read off each pair.
      * Returns ``median(per_iter_ms) * 1000.0``.

    On CUDA RuntimeError (illegal config, OOM, etc.) returns ``math.inf``
    so the caller can skip and try the next candidate.
    """
    try:
        import torch
    except ImportError:
        return math.inf

    if not torch.cuda.is_available():
        return math.inf

    if iters <= 0:
        return math.inf
    if warmup < 0:
        warmup = 0

    try:
        # Warmup
        for _ in range(warmup):
            kernel_fn(*args)
        torch.cuda.synchronize()

        # Allocate event pairs upfront so that recording is cheap.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            starts[i].record()
            kernel_fn(*args)
            stops[i].record()

        torch.cuda.synchronize()

        per_iter_ms = [s.elapsed_time(t) for s, t in zip(starts, stops)]
        median_ms = statistics.median(per_iter_ms)
        return float(median_ms * 1000.0)

    except RuntimeError:
        # CUDA OOM, illegal launch, etc. Skip this config.
        try:
            import torch as _t  # noqa: F401
            _t.cuda.synchronize()
        except Exception:
            pass
        return math.inf


def _build_kernel(
    kernel_name: str,
    arch: int,
    axis_dict: dict,
) -> Callable[..., Any] | None:
    """Compile and return the kernel callable for these template params.

    This is the integration seam with the build system. Returns ``None``
    until the per-arch cpp_extension wrappers are wired up; ``bench_config``
    will then return an ``inf`` result so the orchestrator can keep moving.
    """
    return None


def _make_inputs(kernel_name: str, shape_bucket: Any) -> tuple | None:
    """Allocate representative test tensors for this kernel + shape."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    # Concrete tensor allocation is delegated to the per-kernel harness
    # once the cpp_extension wrappers exist. Stub returns None so
    # bench_config short-circuits to inf.
    return None


def bench_config(
    kernel_name: str,
    arch: int,
    shape_bucket: Any,
    axis_dict: dict,
    *,
    warmup: int = 5,
    iters: int = 20,
) -> BenchResult:
    """Build + time a single (kernel, arch, shape, axes) config."""
    fn = _build_kernel(kernel_name, arch, axis_dict)
    args = _make_inputs(kernel_name, shape_bucket)
    if fn is None or args is None:
        return BenchResult(math.inf, math.inf, math.inf, 0)

    # Run a single bench() for the median, plus a tiny separate min/max pass.
    median_us = bench(fn, args, iters=iters, warmup=warmup)
    if not math.isfinite(median_us):
        return BenchResult(math.inf, math.inf, math.inf, 0)

    # min/max from one extra timing block.
    try:
        import torch
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn(*args)
            stops[i].record()
        torch.cuda.synchronize()
        ms = [s.elapsed_time(t) for s, t in zip(starts, stops)]
        return BenchResult(
            median_us=median_us,
            min_us=float(min(ms) * 1000.0),
            max_us=float(max(ms) * 1000.0),
            iterations=iters,
        )
    except RuntimeError:
        return BenchResult(median_us, median_us, median_us, iters)


def bench_many(grid: list, *, warmup: int = 5, iters: int = 20) -> list:
    """Run the entire grid; return list of (config, BenchResult)."""
    results = []
    for cfg in grid:
        r = bench_config(
            kernel_name=cfg["kernel_name"],
            arch=cfg["arch"],
            shape_bucket=cfg["shape_bucket"],
            axis_dict=cfg["axes"],
            warmup=warmup,
            iters=iters,
        )
        results.append((cfg, r))
    return results
