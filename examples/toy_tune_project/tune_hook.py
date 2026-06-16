"""examples/toy_tune_project/tune_hook.py

Portable autotune hook for the toy SAXPY project. Implements the SAME
schema-agnostic contract that compile.py's portable tune-hook seam expects
(see ``_hook_capture`` / ``_DEFAULT_PROJECT_CONFIG["project"]["tune_hook"]`` in
grokking_optimizers/compile.py):

    def run(*, so_path, model, optimizer, arch, regime, seed)
            -> {"output": np.ndarray (float), "elapsed_ms": float}

compile.py invokes this in a FRESH subprocess per built ``.so`` (so each torch
custom-op artifact loads in its own process — two ``.so``s that both register
``TORCH_LIBRARY(toy, ...)`` cannot coexist in one process), via a dotted path
of the form ``"pkg.module:fn"`` declared in compile_config.toml as
``[project] tune_hook = "tune_hook:run"``.

What this hook does, end to end:
  1. Loads the built artifact with ``torch.ops.load_library(so_path)`` — the
     simplest "torch_op"-style load (the artifact registers torch.ops.toy.saxpy
     via TORCH_LIBRARY, so no Python-module import is needed).
  2. Synthesizes a deterministic input tensor ``x`` seeded by ``seed`` (so the
     reference build and every variant build see bit-identical inputs), with the
     magnitude scaled by ``regime`` to mirror compile.py's adversarial-regime
     sweep ("normal" / "large" / "small" / "adversarial").
  3. Runs ``torch.ops.toy.saxpy(y, x, 2.0, 1.0)`` for a few warmup iters, then N
     timed iters using CUDA events, and returns the MEDIAN elapsed ms.
  4. Returns ``{"output": y.float().cpu().numpy(), "elapsed_ms": median_ms}``.

The ``model`` and ``optimizer`` arguments are part of the generic contract but
are UNUSED for this toy project (SAXPY has neither a model nor an optimizer) —
that is fine and is exactly the point: the contract is generic enough that a
project which doesn't have grokking's (optimizer, model) structure can ignore
those fields. Correctness (y == a*x + b) is invariant to the TOY_TUNED_BLOCK
knob baked into ``so_path``; only ``elapsed_ms`` varies, which is what the
autotuner minimizes.
"""

from __future__ import annotations

# A 1-D problem big enough that the block-size knob actually moves the wall
# time on an H100, but small enough that a tuning sweep stays seconds-scale.
TOY_N = 1 << 22          # 4,194,304 elements
WARMUP_ITERS = 10
TIMED_ITERS = 50

# SAXPY coefficients — fixed so the reference build and every variant build
# compute the identical mathematical result (y = A*x + B).
SAXPY_A = 2.0
SAXPY_B = 1.0


def _regime_scale(regime: str) -> float:
    """Magnitude scale for the synthesized input, mirroring compile.py's
    ``_regime_tensor_expr`` magnitudes so a variant that only diverges under a
    large dynamic range would be caught by the cross-regime validation. For
    this exactly-linear toy op no scale ever changes correctness, but honoring
    the regime keeps the hook faithful to the generic contract."""
    return {
        "normal": 1.0,
        "large": 1e4,
        "small": 1e-4,
        "adversarial": 1e6,
    }.get(regime, 1.0)


def run(*, so_path, model, optimizer, arch, regime, seed):
    """Portable autotune hook entry point. See module docstring for the
    contract. ``model`` / ``optimizer`` are accepted (generic contract) but
    unused for this toy project."""
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "toy tune_hook requires a CUDA device (torch.cuda.is_available() "
            "== False)")

    # 1. Load the built torch custom-op artifact. is_python_module-free: the
    #    op self-registers via TORCH_LIBRARY, so load_library is all we need.
    torch.ops.load_library(str(so_path))

    device = torch.device("cuda")

    # 2. Deterministic, seeded input (so reference and variant see identical x).
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x_cpu = torch.randn(TOY_N, generator=gen, dtype=torch.float32)
    x = (x_cpu * _regime_scale(str(regime))).to(device=device,
                                                 dtype=torch.float32)
    y = torch.empty_like(x)

    saxpy = torch.ops.toy.saxpy

    # 3a. Warmup (kernel JIT/cache, allocator, clocks).
    for _ in range(WARMUP_ITERS):
        saxpy(y, x, SAXPY_A, SAXPY_B)
    torch.cuda.synchronize()

    # 3b. Timed iters with CUDA events → median ms (robust to outliers).
    times_ms = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(TIMED_ITERS):
        start_evt.record()
        saxpy(y, x, SAXPY_A, SAXPY_B)
        end_evt.record()
        torch.cuda.synchronize()
        times_ms.append(start_evt.elapsed_time(end_evt))
    times_ms.sort()
    median_ms = float(times_ms[len(times_ms) // 2])

    # 4. Return the (regime-seeded) output for numerical validation against the
    #    strict reference build, plus the timing objective. float32 → np float.
    output = y.detach().to("cpu", dtype=torch.float32).numpy().astype(
        np.float32)
    return {"output": output, "elapsed_ms": median_ms}


if __name__ == "__main__":
    # Tiny self-exercise: build the kernel once and drive the hook directly,
    # the same way compile.py's _hook_capture would (minus the subprocess).
    # Run from the repo root:
    #   python -m examples.toy_tune_project.tune_hook
    # or from this directory:
    #   python tune_hook.py
    import os
    import numpy as np
    import torch.utils.cpp_extension as cpp_ext

    here = os.path.dirname(os.path.abspath(__file__))
    cu = os.path.join(here, "toy_kernel.cu")
    cpp_ext.load(name="toy_saxpy_selfcheck", sources=[cu],
                 extra_cuda_cflags=["-DTOY_TUNED_BLOCK=256"],
                 verbose=False, is_python_module=False)
    bd = cpp_ext._get_build_directory("toy_saxpy_selfcheck", False)
    so = next(p for p in os.listdir(bd) if p.endswith(".so"))
    res = run(so_path=os.path.join(bd, so), model="none", optimizer="none",
              arch="sm_90a", regime="normal", seed=0)
    out = res["output"]
    # Re-derive the expected SAXPY on the same seeded input to confirm.
    gen = __import__("torch").Generator(device="cpu").manual_seed(0)
    x = __import__("torch").randn(TOY_N, generator=gen, dtype=__import__(
        "torch").float32).numpy()
    expected = SAXPY_A * x + SAXPY_B
    ok = np.allclose(out, expected, rtol=1e-5, atol=1e-5)
    print(f"hook self-check: elapsed_ms={res['elapsed_ms']:.5f} "
          f"n={out.size} correct={ok}")
