"""H100 (sm_90a) DECODER TENSOR-CORE variant gates.

Validates csrc/fused/sm_90/model_stage_decoder_tc.cuh — the batch-tiled bf16
wgmma fwd+bwd for the decoder cell (DESIGN-TC-PIPELINE.md Fork B), the TUNED
VARIANT compiled alongside the scalar path and selected by SG_TUNED_GEMM_IMPL.

TWO LAYERS of gates (DESIGN §10):

  PART 1 — GEMM-orientation micro-gates (decoder_tc_selftest.cu, JIT):
    the engine in the three operand orientations the decoder fwd+bwd use, with
    the transposed HBM->smem staging that keeps every wgmma issue on the
    substrate-validated TransA=0/TransB=0 path:
      (fwd) Y = X @ W^T          (no transpose)
      (dX)  dX = dY @ W          (W transposed-staged)
      (dW)  dW = dY^T @ X, K=T   (BOTH transposed-staged — multi-k-step, the
                                  stride-bug gate; A=I alone would miss it)
    Each vs an fp64 reference of the bf16-rounded inputs.

  PART 2 — full-cell gates (the built extension, real fused step; SKIPPED here
    when the extension's SG_TUNED_GEMM_IMPL is not the wgmma path — those run
    against the shipped .so and live in test_megakernel_vs_eager-style harness;
    this file provides the bf16-rounded oracle helpers + tolerances they use).

Tolerances (DESIGN §5.2, derived NOT guessed):
  * vs the bf16-ROUNDED oracle the bf16 INPUT noise cancels (both sides rounded)
    → we gate fp32 ACCUMULATION-ORDER only. The fp32 round-off bound is
    ≈ K · ε_fp32 · max|partial|, ε_fp32 = 2^-24 ≈ 6e-8. The substrate's 2e-3 was
    derived for K<=512; the dW contraction is K=T (here a few thousand) so the
    bound scales — we re-derive per-K below (atol grows ∝ K).
  * vs the TRUE fp64 oracle the bf16 input rounding SHOWS → DESIGN's rel <= 2e-2.

Run:  CUDA_MPS_PIPE_DIRECTORY=/nonexistent python -m pytest tests/hw/test_decoder_tc.py -v -s
  or: CUDA_MPS_PIPE_DIRECTORY=/nonexistent python tests/hw/test_decoder_tc.py
"""
import os
import sys
import glob
import subprocess

import pytest

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

_THIS = os.path.abspath(__file__)
_REPO = os.path.abspath(os.path.join(os.path.dirname(_THIS), "..", ".."))
_SELFTEST_CU = os.path.join(_REPO, "csrc", "fused", "sm_90", "decoder_tc_selftest.cu")


def _sm90a_available():
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


_GATE = pytest.mark.skipif(
    not _sm90a_available(),
    reason="decoder TC gates need an sm_90 (Hopper) GPU + bf16")

_MODULE = None
_BUILD_ERR = None


def _build_module():
    global _MODULE, _BUILD_ERR
    if _MODULE is not None or _BUILD_ERR is not None:
        return _MODULE
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    from torch.utils.cpp_extension import load
    try:
        _MODULE = load(
            name="decoder_tc_selftest",
            sources=[_SELFTEST_CU],
            extra_include_paths=[_REPO],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                "-gencode=arch=compute_90a,code=sm_90a",
                "-gencode=arch=compute_90a,code=compute_90a",
                "-lineinfo",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=True,
        )
    except Exception as e:
        _BUILD_ERR = e
        raise
    return _MODULE


def _disable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _k_scaled_atol(K):
    # fp32 accumulation-order bound ≈ K·ε·scale with ε=2^-24. Unit-scale random
    # operands, well-conditioned. Substrate used 2e-3 at K<=512; scale linearly
    # with a floor. (This is the bf16-rounded-oracle accumulation-order gate; the
    # bf16 input noise cancels because both sides are bf16-rounded.)
    return max(2e-3, 2e-3 * (K / 512.0))


# ═════════════════════════════════════════════════════════════════════════
#  PART 1 — GEMM orientation micro-gates.
# ═════════════════════════════════════════════════════════════════════════

def _bf16(t):
    return t.to(torch.bfloat16)


@_GATE
def test_fwd_identity_localization():
    """(fwd) A=I localization: X = [I | 0] (TILE_M x Kin, first Kin rows identity
    over the K=Kin cols) ⇒ Y[m,n] = W[n,m] for m<Kin. Isolates fragment-layout /
    descriptor / TILE_M-stacking bugs."""
    mod = _build_module()
    _disable_tf32()
    TILE_M = mod.TILE_M
    N, Kin = 128, 128
    X = torch.zeros(TILE_M, Kin, device="cuda", dtype=torch.float32)
    for i in range(min(Kin, TILE_M)):
        X[i, i] = 1.0
    W = (torch.arange(N * Kin, device="cuda", dtype=torch.float32).reshape(N, Kin) / 97.0)
    Xb, Wb = _bf16(X), _bf16(W)
    D = mod.gemm_fwd(Xb.reshape(-1).contiguous(), Wb.reshape(-1).contiguous(), N, Kin, N)
    # ref: Y[m,n] = Σ_k Xb[m,k]·Wb[n,k]
    ref = torch.einsum("mk,nk->mn", Xb.double().cpu(), Wb.double().cpu())
    err = (D.double().cpu() - ref).abs().max().item()
    assert err < _k_scaled_atol(Kin), (
        f"fwd A=I localization max_err={err:.3e} — layout/descriptor/stacking bug\n"
        f"D[:4,:4]=\n{D[:4,:4].cpu()}\nref[:4,:4]=\n{ref[:4,:4]}")
    print(f"[fwd-loc] A=I TILE_M={TILE_M} N={N} K={Kin} max_err={err:.3e}  OK")


@_GATE
@pytest.mark.parametrize("Kin,Nout", [(128, 128), (512, 128), (128, 384), (128, 512)])
def test_fwd_random(Kin, Nout):
    """(fwd) random X,W over the design's shapes: K depths (in/out K=128, ff2
    K=512) AND N widths (out 128, in_proj 3d=384, ff0 dff=512 — the N-tiled
    cases the caller loop must cover)."""
    mod = _build_module()
    _disable_tf32()
    TILE_M = mod.TILE_M
    N = 128                                    # wgmma atom width (caller N-tiles)
    g = torch.Generator(device="cuda").manual_seed(1000 + Kin * 1000 + Nout)
    X = torch.randn(TILE_M, Kin, generator=g, device="cuda")
    W = torch.randn(Nout, Kin, generator=g, device="cuda")
    Xb, Wb = _bf16(X), _bf16(W)
    D = mod.gemm_fwd(Xb.reshape(-1).contiguous(), Wb.reshape(-1).contiguous(), N, Kin, Nout)
    ref = torch.einsum("mk,nk->mn", Xb.double().cpu(), Wb.double().cpu())
    err = (D.double().cpu() - ref).abs().max().item()
    tol = _k_scaled_atol(Kin) * max(1.0, ref.abs().max().item())
    assert err < tol, f"fwd random K={Kin} Nout={Nout} max_err={err:.3e} tol={tol:.3e}"
    print(f"[fwd] random Nout={Nout} (N-tiled by {N}) K={Kin} "
          f"max_err={err:.3e} tol={tol:.3e}  OK")


@_GATE
@pytest.mark.parametrize("Nout", [128, 512])
def test_dx_random(Nout):
    """(dX) dX = dY @ W with W TRANSPOSED-staged. N(wgmma)=in_dim=128, K=Nout
    (in_proj-dX has Nout=3d=384; out_proj 128; ff0-dX Nout=dff=512; ff2-dX 128).
    Exercises Nout in {128 (out/ff2), 512 (ff0)} — the transposed-W orientation."""
    mod = _build_module()
    _disable_tf32()
    TILE_M = mod.TILE_M
    N = 128                                   # in_dim
    g = torch.Generator(device="cuda").manual_seed(2000 + Nout)
    dY = torch.randn(TILE_M, Nout, generator=g, device="cuda")
    W = torch.randn(Nout, N, generator=g, device="cuda")   # [Nout, in_dim]
    dYb, Wb = _bf16(dY), _bf16(W)
    D = mod.gemm_dx(dYb.reshape(-1).contiguous(), Wb.reshape(-1).contiguous(), N, Nout)
    # ref dX[m,i] = Σ_o dY[m,o]·W[o,i]
    ref = torch.einsum("mo,oi->mi", dYb.double().cpu(), Wb.double().cpu())
    err = (D.double().cpu() - ref).abs().max().item()
    tol = _k_scaled_atol(Nout) * max(1.0, ref.abs().max().item())
    assert err < tol, f"dX Nout={Nout} max_err={err:.3e} tol={tol:.3e}"
    print(f"[dX] random N={N} Nout(K)={Nout} max_err={err:.3e} tol={tol:.3e}  OK")


@_GATE
@pytest.mark.parametrize("T,Nout", [(128, 128), (2048, 128), (2048, 384), (2048, 512)])
def test_dw_random_multistep(T, Nout):
    """(dW) dW = dY^T @ X with BOTH operands TRANSPOSED-staged, K=T multi-k-step.
    THE stride-bug gate (model_stages_decoder.cuh:619-625 lesson): a single tile
    would pass with a wrong k-stride. T=2048 = 128 k-steps. Nout in {128 (out
    weight), 384 (in_proj weight = 6 M-atoms), 512 (ff weight = 8 M-atoms)} —
    exercises the M-atom-block loop the helper must cover (NOT just Nout=TILE_M).
    in_dim N=128."""
    mod = _build_module()
    _disable_tf32()
    N = 128
    g = torch.Generator(device="cuda").manual_seed(3000 + T + Nout * 7)
    dY = torch.randn(T, Nout, generator=g, device="cuda")     # [T, out]
    X = torch.randn(T, N, generator=g, device="cuda")         # [T, in]
    dYb, Xb = _bf16(dY), _bf16(X)
    D = mod.gemm_dw(dYb.reshape(-1).contiguous(), Xb.reshape(-1).contiguous(), N, Nout, T)
    ref = torch.einsum("to,ti->oi", dYb.double().cpu(), Xb.double().cpu())
    err = (D.double().cpu() - ref).abs().max().item()
    tol = _k_scaled_atol(T) * max(1.0, ref.abs().max().item())
    assert err < tol, f"dW T(K)={T} Nout={Nout} max_err={err:.3e} tol={tol:.3e}"
    print(f"[dW] random Nout={Nout} in={N} K=T={T} ({T//16} steps, "
          f"{(Nout+63)//64} M-atoms) max_err={err:.3e} tol={tol:.3e}  OK")


@_GATE
def test_dw_identity_who_owns_what():
    """(dW) A=I-style: dY = [I_Nout | 0] over T rows (dY[t,o]=1 iff t==o for
    t<Nout else 0). Then dW[o,i] = Σ_t dY[t,o]·X[t,i] = X[o,i] for o<Nout. Pins
    the (out,t)->fragment mapping for the doubly-transposed staging."""
    mod = _build_module()
    _disable_tf32()
    Nout, N, T = 128, 128, 256
    dY = torch.zeros(T, Nout, device="cuda", dtype=torch.float32)
    for o in range(min(Nout, T)):
        dY[o, o] = 1.0
    X = (torch.arange(T * N, device="cuda", dtype=torch.float32).reshape(T, N) / 131.0)
    dYb, Xb = _bf16(dY), _bf16(X)
    D = mod.gemm_dw(dYb.reshape(-1).contiguous(), Xb.reshape(-1).contiguous(), N, Nout, T)
    ref = Xb[:Nout, :].double().cpu()      # dW[o,i] = X[o,i]
    err = (D.double().cpu() - ref).abs().max().item()
    assert err < _k_scaled_atol(T), (
        f"dW A=I localization max_err={err:.3e} — transposed-staging mapping bug\n"
        f"D[:4,:4]=\n{D[:4,:4].cpu()}\nref[:4,:4]=\n{ref[:4,:4]}")
    print(f"[dW-loc] A=I Nout={Nout} in={N} T={T} max_err={err:.3e}  OK")


@_GATE
def test_determinism_bitwise():
    """Same inputs twice → bit-identical (fixed tile + ascending-k, no atomics)."""
    mod = _build_module()
    _disable_tf32()
    Nout, N, T = 128, 128, 512
    g = torch.Generator(device="cuda").manual_seed(31337)
    dY = _bf16(torch.randn(T, Nout, generator=g, device="cuda")).reshape(-1).contiguous()
    X = _bf16(torch.randn(T, N, generator=g, device="cuda")).reshape(-1).contiguous()
    D1 = mod.gemm_dw(dY, X, N, Nout, T)
    D2 = mod.gemm_dw(dY, X, N, Nout, T)
    D3 = mod.gemm_dw(dY, X, N, Nout, T)
    assert torch.equal(D1, D2) and torch.equal(D2, D3), "non-deterministic dW"
    print("[determinism] dW A/A/A bit-identical  OK")


# ── direct-run report ────────────────────────────────────────────────────
def _main():
    if not _sm90a_available():
        print("SKIP: no sm_90 GPU.")
        return 0
    print("Building decoder TC self-test (JIT, sm_90a)…")
    _build_module()
    fails = []

    def _try(name, fn, *a):
        try:
            fn(*a)
        except Exception as e:
            fails.append((name, repr(e)))
            print(f"  [FAIL] {name}: {e!r}")

    _try("fwd-loc(A=I)", test_fwd_identity_localization)
    for (K, Nout) in ((128, 128), (512, 128), (128, 384), (128, 512)):
        _try(f"fwd random K={K} Nout={Nout}", test_fwd_random, K, Nout)
    for Nout in (128, 512):
        _try(f"dX random Nout={Nout}", test_dx_random, Nout)
    for (T, Nout) in ((128, 128), (2048, 128), (2048, 384), (2048, 512)):
        _try(f"dW random T={T} Nout={Nout}", test_dw_random_multistep, T, Nout)
    _try("dW-loc(A=I)", test_dw_identity_who_owns_what)
    _try("determinism", test_determinism_bitwise)

    print("\n==================== SUMMARY ====================")
    if not fails:
        print("ALL PART-1 GEMM-ORIENTATION GATES PASSED")
        return 0
    print(f"{len(fails)} FAILURE(S):")
    for n, e in fails:
        print(f"  - {n}: {e}")
    return 1


if __name__ == "__main__":
    sys.exit(_main())
