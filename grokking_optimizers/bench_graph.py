"""grokking_optimizers.bench_graph — CUDA / HIP graph-replay timing.

Captures one ``opt.step()`` into a graph and replays it N times under a
single CUDA / HIP event pair. Eliminates per-iteration launch overhead
so the measured median reflects kernel work, not Python.

Functions return ``{"timing_ms", "min_ms", "max_ms", "n"}`` — the same
shape ``compile.py:_time_variant`` returns — so the timing worker can
swap event-based and graph-based timing transparently.

Importing this module does not require a GPU; the functions themselves
require ``torch.cuda.is_available()``. HIP is exposed via the same
``torch.cuda`` namespace in ROCm builds.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _build_param(opt_class_name: str, size: int) -> Any:
    import torch
    torch.manual_seed(0)
    p = torch.nn.Parameter(
        torch.randn(size, size, device="cuda", dtype=torch.float32))
    g = torch.randn_like(p)
    p.grad = g
    return p, g


def cuda_graph_median_ms(opt_class: str, *, size: int = 4096,
                         warmup: int = 5, iters: int = 21) -> Dict[str, float]:
    """Time ``opt.step()`` under a CUDA graph replay."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_graph_median_ms requires torch.cuda.is_available()")
    from grokking_optimizers import _ops  # noqa: F401 — must be loaded first
    from importlib import import_module
    grok = import_module("grokking_optimizers")
    OptCls = getattr(grok, opt_class)

    p, g = _build_param(opt_class, size)
    opt = OptCls([p], lr=1e-3)

    # Warmup the optimizer state (and the runtime allocator) outside the graph
    for _ in range(max(1, warmup)):
        p.grad = g.clone()
        opt.step()
    torch.cuda.synchronize()

    # Replay a single opt.step() N times. Each replay reuses the captured
    # work; we just need the gradient refreshed.
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        p.grad = g.clone()
        opt.step()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        p.grad.copy_(g)
        opt.step()

    timings = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))
    timings.sort()
    return {
        "timing_ms": float(timings[len(timings) // 2]),
        "min_ms":    float(timings[0]),
        "max_ms":    float(timings[-1]),
        "n":         len(timings),
    }


def hip_graph_median_ms(opt_class: str, *, size: int = 4096,
                        warmup: int = 5, iters: int = 21) -> Dict[str, float]:
    """HIP analogue.

    On ROCm-built PyTorch, ``torch.cuda.CUDAGraph`` is the official HIP
    graph wrapper (the namespace was kept stable for portability). For
    that reason the implementation reuses ``cuda_graph_median_ms`` —
    they differ only in the kernel sources baked into ``_ops``, not in
    the bench harness.
    """
    return cuda_graph_median_ms(opt_class, size=size, warmup=warmup, iters=iters)


def event_median_ms(opt_class: str, *, size: int = 4096,
                    warmup: int = 5, iters: int = 21) -> Dict[str, float]:
    """Fallback timer for archs / backends that do not support graph capture
    (e.g. when CUDAGraph capture fails for the kernel in question)."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("event_median_ms requires torch.cuda.is_available()")
    from importlib import import_module
    grok = import_module("grokking_optimizers")
    OptCls = getattr(grok, opt_class)

    p, g = _build_param(opt_class, size)
    for _ in range(max(1, warmup)):
        p.grad = g.clone()
        opt = OptCls([p], lr=1e-3)
        opt.step()
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        p.grad = g.clone()
        opt = OptCls([p], lr=1e-3)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        opt.step()
        e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))
    timings.sort()
    return {
        "timing_ms": float(timings[len(timings) // 2]),
        "min_ms":    float(timings[0]),
        "max_ms":    float(timings[-1]),
        "n":         len(timings),
    }


__all__ = ["cuda_graph_median_ms", "hip_graph_median_ms", "event_median_ms"]
