"""examples/autotune_demo/tune_hook.py

Portable autotune hook for the NAIVE tiled-GEMM demo. Implements the SAME
schema-agnostic contract that compile.py's portable tune-hook seam expects
(grokking_optimizers/compile.py ``_hook_capture`` /
``_DEFAULT_PROJECT_CONFIG["project"]["tune_hook"]`` /
grokking_optimizers/tune_hook.py)::

    def run(*, so_path, model, optimizer, arch, regime, seed)
            -> {"output": np.ndarray (float32), "elapsed_ms": float}

compile.py invokes this in a FRESH subprocess per built ``.so`` (each artifact
registers ``TORCH_LIBRARY(sg_demo, ...)``, and a TORCH_LIBRARY namespace can be
registered only once per process — so two GEMM variant ``.so``s cannot coexist
in one process; a fresh subprocess per ``.so`` is mandatory). The demo driver
``run_autotune.py`` drives this hook through the IDENTICAL subprocess protocol.

What this hook does, end to end:
  1. Loads the built artifact with ``torch.ops.load_library(so_path)`` (the op
     self-registers ``torch.ops.sg_demo.gemm`` via TORCH_LIBRARY, so no
     Python-module import is needed) — exactly the "torch_op" load path.
  2. Synthesizes deterministic, seeded fp32 matrices A[M,K], B[K,N] (so the
     strict reference build and every variant build see bit-identical inputs),
     with magnitude scaled by ``regime`` (normal/large/small/adversarial) to
     mirror compile.py's adversarial-regime sweep.
  3. Runs ``torch.ops.sg_demo.gemm(C, A, B)`` for a few warmup iters, then N
     CUDA-event-timed iters, and returns the MEDIAN elapsed ms.
  4. Returns ``{"output": C.float().cpu().numpy(), "elapsed_ms": median_ms}``.

``model`` / ``optimizer`` are part of the generic contract but UNUSED here (a
GEMM has neither) — that is the point: the seam is generic enough that a
project without grokking's (optimizer, model) structure ignores those fields.
The GEMM MATH is identical for every (TILE_M,TILE_N,TILE_K,BLOCK) baked into
``so_path`` — only ``elapsed_ms`` varies, which is what the autotuner minimizes;
the output tensor is what compile.py's strict-vs-variant oracle compares.
"""
from __future__ import annotations

# Problem size. M=N=K=2048 keeps a full sweep iterable in minutes on an H100
# (even while sharing the GPU) while being large enough that the tiling knobs
# move the wall time substantially. fp32.
GEMM_M = 2048
GEMM_N = 2048
GEMM_K = 2048

WARMUP_ITERS = 5
TIMED_ITERS = 30


def _regime_scale(regime: str) -> float:
    """Magnitude scale for the synthesized inputs, mirroring compile.py's
    adversarial-regime magnitudes. For an exact fp32 GEMM no scale changes
    *relative* correctness, but honoring the regime keeps the hook faithful to
    the generic contract (a variant that only diverged under a large dynamic
    range would be caught by the cross-regime compare)."""
    return {
        "normal": 1.0,
        "large": 1e2,
        "small": 1e-2,
        "adversarial": 1e3,
    }.get(str(regime), 1.0)


def run(*, so_path, model, optimizer, arch, regime, seed):
    """Portable autotune hook entry point. See module docstring for the
    contract. ``model`` / ``optimizer`` are accepted (generic contract) but
    unused for this GEMM demo."""
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "autotune_demo tune_hook requires a CUDA device "
            "(torch.cuda.is_available() == False)")

    # 1. Load the built torch custom-op artifact (self-registers via
    #    TORCH_LIBRARY; load_library is all that is needed).
    torch.ops.load_library(str(so_path))

    device = torch.device("cuda")
    scale = _regime_scale(regime)

    # 2. Deterministic, seeded inputs (reference + every variant see identical
    #    A, B for a given (regime, seed)). Generate on CPU for cross-process
    #    determinism, then move to device.
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    A = (torch.randn(GEMM_M, GEMM_K, generator=gen, dtype=torch.float32) * scale
         ).to(device=device)
    B = (torch.randn(GEMM_K, GEMM_N, generator=gen, dtype=torch.float32) * scale
         ).to(device=device)
    C = torch.empty(GEMM_M, GEMM_N, device=device, dtype=torch.float32)

    gemm = torch.ops.sg_demo.gemm

    # 3a. Warmup (kernel cache, allocator, clocks).
    for _ in range(WARMUP_ITERS):
        gemm(C, A, B)
    torch.cuda.synchronize()

    # 3b. Timed iters with CUDA events -> median ms (robust to the GPU-sharing
    #     noise from any co-resident process on this device).
    times_ms = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(TIMED_ITERS):
        start_evt.record()
        gemm(C, A, B)
        end_evt.record()
        torch.cuda.synchronize()
        times_ms.append(start_evt.elapsed_time(end_evt))
    times_ms.sort()
    median_ms = float(times_ms[len(times_ms) // 2])

    # 4. Return the (regime-seeded) output for numerical validation against the
    #    strict reference build, plus the timing objective.
    output = C.detach().to("cpu", dtype=torch.float32).numpy().astype(np.float32)
    return {"output": output, "elapsed_ms": median_ms}


if __name__ == "__main__":
    # Tiny self-exercise: build the naive-default kernel once and drive the hook
    # directly, the way compile.py's _hook_capture would (minus the subprocess).
    # Run from the repo root:  python -m examples.autotune_demo.tune_hook
    import os
    import numpy as np
    import torch
    import torch.utils.cpp_extension as cpp_ext

    here = os.path.dirname(os.path.abspath(__file__))
    cu = os.path.join(here, "gemm_kernel.cu")
    cpp_ext.load(name="sg_demo_gemm_selfcheck", sources=[cu],
                 extra_cuda_cflags=["-O3"], verbose=False, is_python_module=False)
    bd = cpp_ext._get_build_directory("sg_demo_gemm_selfcheck", False)
    so = next(p for p in os.listdir(bd) if p.endswith(".so"))
    res = run(so_path=os.path.join(bd, so), model="none", optimizer="none",
              arch="sm_90a", regime="normal", seed=0)
    out = res["output"]
    # Re-derive the expected GEMM on the same seeded inputs to confirm.
    gen = torch.Generator(device="cpu").manual_seed(0)
    A = torch.randn(GEMM_M, GEMM_K, generator=gen, dtype=torch.float32)
    B = torch.randn(GEMM_K, GEMM_N, generator=gen, dtype=torch.float32)
    expected = (A @ B).numpy()
    ok = np.allclose(out, expected, rtol=1e-3, atol=1e-3)
    print(f"hook self-check: elapsed_ms={res['elapsed_ms']:.4f} "
          f"shape={out.shape} correct={ok}")
