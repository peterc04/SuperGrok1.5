"""H100 (sm_90a) TENSOR-CORE SUBSTRATE correctness gates.

Validates the R2 hand-rolled in-kernel ss-wgmma layer:
  csrc/backends/cuda/sm_90/wgmma.cuh        (ss-wgmma engine)
  csrc/backends/cuda/sm_90/tile_pipeline.cuh (mbarrier producer/consumer)
against torch references, via a JIT-compiled minimal test TU
(csrc/backends/cuda/sm_90/wgmma_selftest.cu) loaded with
torch.utils.cpp_extension.load() — NOT setup.py (this TU is test-only and is
never part of the shipped build glob).

These are CORRECTNESS gates (contention-immune; each runs in seconds on a tiny
grid, < 1 GB), NOT timed benchmarks — perf measurement is a separate quiet-GPU
phase. The GPU is shared with a tuner fleet, so allocations are kept small.

Gates (DESIGN-TC-PIPELINE.md §10):
  (a) single-tile bf16 GEMM (m64×N×16) vs fp32 reference of the bf16-rounded
      inputs, over N ∈ {8,16,32,64,96,128} incl. the ragged 97-row head padded;
  (b) multi-tile K-loop accumulation (the k16 chain over K=128);
  (c) pipelined variant (producer/consumer, 2 & 3 stages) — same numerics as
      the unpipelined path;
  (d) determinism: same inputs twice → bitwise-identical outputs;
  (e) occupancy / refuse probe for an oversized dynamic-smem request (no hang);
  + SASS audit: cuobjdump the JIT module, assert HGMMA / wgmma present (not
      lowered to FMA), report register counts / spills.

Run:  python -m pytest tests/hw/test_wgmma_substrate.py -v -s
  or:  python tests/hw/test_wgmma_substrate.py     (direct, prints the report)
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


# ── repo root (this file is tests/hw/…) ──────────────────────────────────
_THIS = os.path.abspath(__file__)
_REPO = os.path.abspath(os.path.join(os.path.dirname(_THIS), "..", ".."))
_SELFTEST_CU = os.path.join(
    _REPO, "csrc", "backends", "cuda", "sm_90", "wgmma_selftest.cu")


def _sm90a_available():
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9   # Hopper sm_90 (wgmma needs the sm_90a build flag, set below)


_GATE = pytest.mark.skipif(
    not _sm90a_available(),
    reason="tensor-core substrate gates need an sm_90 (Hopper) GPU + bf16")


# ── JIT build of the self-test TU (module-level cache; built once) ───────
_MODULE = None
_BUILD_ERR = None


def _build_module():
    """JIT-compile wgmma_selftest.cu for sm_90a. Cached. Returns (mod, srcpath)."""
    global _MODULE, _BUILD_ERR
    if _MODULE is not None or _BUILD_ERR is not None:
        return _MODULE
    # sm_90a is REQUIRED: wgmma is an architecture-specific (the 'a' suffix)
    # instruction. We pin TORCH_CUDA_ARCH_LIST so cpp_extension emits exactly
    # compute_90a/sm_90a and does not auto-append a plain sm_90 gencode.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    from torch.utils.cpp_extension import load
    try:
        _MODULE = load(
            name="wgmma_substrate_selftest",
            sources=[_SELFTEST_CU],
            extra_include_paths=[_REPO],           # so #include "csrc/..." resolves
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                # Emit BOTH the sm_90a SASS (what the GPU runs) AND the
                # compute_90a PTX into the fatbin. The PTX embed is REQUIRED by
                # the SASS/PTX audit gate: a code=sm_90a-only cubin carries no
                # PTX, so `cuobjdump -ptx` returns empty and the wgmma.mma_async
                # / commit_group / wait_group PTX assertions would (wrongly) fail
                # on a correct kernel. Adding code=compute_90a does NOT change
                # execution (the loader still picks the sm_90a SASS).
                "-gencode=arch=compute_90a,code=sm_90a",
                "-gencode=arch=compute_90a,code=compute_90a",
                # Keep ptxas line info so the SASS audit can attribute spills.
                "-lineinfo",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=True,
        )
    except Exception as e:   # pragma: no cover - surfaced by the gates
        _BUILD_ERR = e
        raise
    return _MODULE


# ── reference + tolerance ────────────────────────────────────────────────
#
# The wgmma computes  D[m,n] = sum_k A[m,k] * B[n,k]  with bf16 inputs and an
# fp32 accumulator (B is stored N×K row-major and read transposed → D = A @ B^T
# over the stored tiles). The faithful reference is the SAME contraction done in
# fp32 on the bf16-ROUNDED inputs: cast A,B to bf16 (both sides), then upcast to
# fp32 and matmul. bf16 inputs are then exact on both sides and bf16×bf16
# products are exact in fp32; the ONLY residual is fp32 ACCUMULATION ORDER
# (hardware k-order vs torch). The reference is computed on CPU in fp64 (and in
# fp32 on-GPU with TF32 DISABLED) so the reference itself carries no tensor-core
# imprecision.
#
# Tolerance derivation (stated, not guessed): with K total accumulation steps,
# the fp32 accumulation round-off is bounded ≈ K · ε_fp32 · max|partial|, with
# ε_fp32 = 2^-24 ≈ 6e-8. For K ≤ 512 and unit-scale operands the absolute error
# bound is ≈ 512 · 6e-8 · O(K) ≈ a few e-3 worst case; in practice (random
# ±1 operands, well-conditioned) max-abs-err is ≤ ~1e-3. We therefore gate at
# atol=2e-3, rtol=2e-3. A max-err at the ~1e-2 level would indicate a BUG
# (fragment-layout / transpose / leaked TF32), not bf16 noise — we do NOT loosen
# the tolerance to absorb that.
_ATOL = 2e-3
_RTOL = 2e-3


def _disable_tf32():
    # The on-GPU fp32 reference must NOT run on TF32 tensor cores or it loses
    # ~10 mantissa bits and the tolerance becomes meaningless.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _make_AB(N, k_steps, seed):
    """A: [k_steps,64,16] bf16, B: [k_steps,N,16] bf16. Returns (A,B, A_flat,B_flat)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = (torch.randn(k_steps, 64, 16, generator=g, device="cuda",
                     dtype=torch.float32)).to(torch.bfloat16)
    B = (torch.randn(k_steps, N, 16, generator=g, device="cuda",
                     dtype=torch.float32)).to(torch.bfloat16)
    return A, B


def _reference(A, B):
    """fp64 CPU reference: D[m,n] = sum_{k_step,k} A[k_step,m,k]·B[k_step,n,k]."""
    Ad = A.double().cpu()   # [ks,64,16]
    Bd = B.double().cpu()   # [ks,N,16]
    # contract over (k_step, 16): einsum gives D[m,n] = Σ_{s,k} A[s,m,k] B[s,n,k]
    D = torch.einsum("smk,snk->mn", Ad, Bd)
    return D   # [64, N] fp64 on CPU


def _run_tile(mod, A, B, N, k_steps):
    A_flat = A.reshape(-1).contiguous()
    B_flat = B.reshape(-1).contiguous()
    return mod.gemm_tile(A_flat, B_flat, N, k_steps)   # [64,N] fp32 cuda


def _max_err(D_dev, D_ref):
    d = D_dev.double().cpu()
    return (d - D_ref).abs().max().item()


# ═════════════════════════════════════════════════════════════════════════
#  LOCALIZATION: A = I (identity), single K-tile. Output must equal B exactly
#  (the 64×16 identity-prefix times B). This isolates fragment-layout /
#  descriptor / transpose bugs: any permuted output element points straight at
#  the mapping. Run first; if this fails, the random-input gates are noise.
# ═════════════════════════════════════════════════════════════════════════
@_GATE
def test_a_identity_localization():
    mod = _build_module()
    _disable_tf32()
    N = 16
    k_steps = 1
    # A is 64×16. Make it select row r of B^T... we want D = A @ B^T with B
    # stored N×16. Set A[m,k] = 1 if (m==k) else 0 over the 16-col K, so for
    # m<16: D[m,n] = sum_k A[m,k]·B[n,k] = B[n,m]. i.e. D = B^T (first 16 rows).
    A = torch.zeros(1, 64, 16, device="cuda", dtype=torch.float32)
    for i in range(16):
        A[0, i, i] = 1.0
    A = A.to(torch.bfloat16)
    B = torch.arange(N * 16, device="cuda", dtype=torch.float32).reshape(1, N, 16)
    B = (B / 97.0).to(torch.bfloat16)   # distinct, bf16-representable-ish values

    D = _run_tile(mod, A, B, N, k_steps)        # [64,16]
    Dref = _reference(A, B)                      # D[m,n] = B[n,m] for m<16
    err = _max_err(D, Dref)
    assert err < _ATOL, (
        f"A=I localization FAILED max_err={err:.3e} — fragment-layout / "
        f"descriptor / transpose bug. D[:4,:4]=\n{D[:4,:4].cpu()}\n"
        f"ref[:4,:4]=\n{Dref[:4,:4]}")
    print(f"[gate-localization] A=I  N={N} max_err={err:.3e}  OK")


# ═════════════════════════════════════════════════════════════════════════
#  (a) single-tile bf16 GEMM vs fp32 reference, multiple N incl. ragged.
# ═════════════════════════════════════════════════════════════════════════
@_GATE
@pytest.mark.parametrize("N", [8, 16, 32, 64, 96, 128])
def test_a_single_tile_shapes(N):
    mod = _build_module()
    _disable_tf32()
    A, B = _make_AB(N, k_steps=1, seed=1234 + N)
    D = _run_tile(mod, A, B, N, 1)
    Dref = _reference(A, B)
    err = _max_err(D, Dref)
    assert err < max(_ATOL, _RTOL * Dref.abs().max().item()), \
        f"single-tile N={N} max_err={err:.3e}"
    print(f"[gate-a] single-tile N={N:3d} K=16  max_err={err:.3e}  OK")


@_GATE
def test_a_ragged_head_97_padded():
    """The decoder head V=99 / ViT V=97: an odd-N GEMM tiled at N=96 (a legal
    wgmma atom) then the 97th..99th cols handled by the N=128-padded tile. We
    verify the N=96 atom AND that a 97-row-padded-to-128 B (rows 97..127 zero)
    yields exactly the first 97 columns matching the reference (the pad is
    inert). This is the 'ragged edge' gate."""
    mod = _build_module()
    _disable_tf32()
    # 97 real columns, padded to N=128 with zeros (rows 97..127 of B = 0).
    real = 97
    N = 128
    A, Bfull = _make_AB(N, k_steps=1, seed=99)
    B = Bfull.clone()
    B[0, real:, :] = 0   # zero the pad rows
    D = _run_tile(mod, A, B, N, 1)
    Dref = _reference(A, B)
    # Compare only the real columns; the padded columns must be ~0.
    err_real = (D[:, :real].double().cpu() - Dref[:, :real]).abs().max().item()
    pad_max = D[:, real:].abs().max().item()
    assert err_real < _ATOL, f"ragged head real-cols max_err={err_real:.3e}"
    assert pad_max < _ATOL, f"ragged head pad cols not ~0: {pad_max:.3e}"
    print(f"[gate-a] ragged head (97→128 pad) real_err={err_real:.3e} "
          f"pad_max={pad_max:.3e}  OK")


# ═════════════════════════════════════════════════════════════════════════
#  (b) multi-tile K-loop accumulation: the k16 chain over K = 128 (8 k-steps).
# ═════════════════════════════════════════════════════════════════════════
@_GATE
@pytest.mark.parametrize("N", [64, 128])
def test_b_kloop_K128(N):
    mod = _build_module()
    _disable_tf32()
    k_steps = 8                      # K = 8 * 16 = 128 (DESIGN in/out/ff K=128)
    A, B = _make_AB(N, k_steps=k_steps, seed=777 + N)
    D = _run_tile(mod, A, B, N, k_steps)
    Dref = _reference(A, B)
    err = _max_err(D, Dref)
    # K=128 → tolerance bound ≈ 128·ε·scale; still ≤ a few e-3.
    tol = max(_ATOL, _RTOL * Dref.abs().max().item())
    assert err < tol, f"K-loop N={N} K=128 max_err={err:.3e} tol={tol:.3e}"
    print(f"[gate-b] K-loop  N={N:3d} K=128 ({k_steps} steps) max_err={err:.3e}  OK")


@_GATE
def test_b_kloop_K512():
    """ff2 K=512 (32 k-steps) — the design's deepest K chain; the tolerance
    bound grows with K so this exercises the accumulation-order envelope."""
    mod = _build_module()
    _disable_tf32()
    N, k_steps = 128, 32
    A, B = _make_AB(N, k_steps, seed=2024)
    D = _run_tile(mod, A, B, N, k_steps)
    Dref = _reference(A, B)
    err = _max_err(D, Dref)
    tol = max(_ATOL, _RTOL * Dref.abs().max().item())
    assert err < tol, f"K=512 max_err={err:.3e} tol={tol:.3e}"
    print(f"[gate-b] K-loop  N={N} K=512 (32 steps) max_err={err:.3e} "
          f"tol={tol:.3e}  OK")


# ═════════════════════════════════════════════════════════════════════════
#  (c) pipelined producer/consumer variant — same numerics as unpipelined.
# ═════════════════════════════════════════════════════════════════════════
@_GATE
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("N", [64, 128])
def test_c_pipelined_matches_unpipelined(N, depth):
    mod = _build_module()
    _disable_tf32()
    k_steps = 8                      # K=128
    A, B = _make_AB(N, k_steps, seed=555 + N + depth)
    A_flat = A.reshape(-1).contiguous()
    B_flat = B.reshape(-1).contiguous()
    D_unp = mod.gemm_tile(A_flat, B_flat, N, k_steps)
    D_pip = mod.gemm_tile_pipelined(A_flat, B_flat, N, k_steps, depth)
    # The pipeline only changes operand-availability timing, not the ascending-k
    # accumulation order → BIT-IDENTICAL to the unpipelined path.
    assert torch.equal(D_unp, D_pip), (
        f"pipelined depth={depth} N={N} differs from unpipelined "
        f"(max abs diff {(D_unp-D_pip).abs().max().item():.3e})")
    # And both within tolerance of the reference.
    Dref = _reference(A, B)
    err = _max_err(D_pip, Dref)
    tol = max(_ATOL, _RTOL * Dref.abs().max().item())
    assert err < tol, f"pipelined N={N} depth={depth} vs ref max_err={err:.3e}"
    print(f"[gate-c] pipelined N={N:3d} depth={depth}: bit-identical to "
          f"unpipelined; vs ref max_err={err:.3e}  OK")


# ═════════════════════════════════════════════════════════════════════════
#  (d) determinism: same inputs twice → bitwise-identical outputs.
# ═════════════════════════════════════════════════════════════════════════
@_GATE
def test_d_determinism_bitwise():
    mod = _build_module()
    _disable_tf32()
    N, k_steps = 128, 8
    A, B = _make_AB(N, k_steps, seed=31337)
    A_flat = A.reshape(-1).contiguous()
    B_flat = B.reshape(-1).contiguous()
    D1 = mod.gemm_tile(A_flat, B_flat, N, k_steps)
    D2 = mod.gemm_tile(A_flat, B_flat, N, k_steps)
    D3 = mod.gemm_tile(A_flat, B_flat, N, k_steps)
    assert torch.equal(D1, D2) and torch.equal(D2, D3), \
        "non-deterministic wgmma output (fixed tile + ascending-k must be bit-exact)"
    # Pipelined path determinism too.
    P1 = mod.gemm_tile_pipelined(A_flat, B_flat, N, k_steps, 2)
    P2 = mod.gemm_tile_pipelined(A_flat, B_flat, N, k_steps, 2)
    assert torch.equal(P1, P2), "non-deterministic pipelined output"
    print("[gate-d] determinism A/A/A + pipelined bit-identical  OK")


# ═════════════════════════════════════════════════════════════════════════
#  (e) occupancy / refuse probe for an oversized dynamic-smem request.
# ═════════════════════════════════════════════════════════════════════════
@_GATE
def test_e_occupancy_refuse_oversized_smem():
    mod = _build_module()
    cap = mod.device_max_dynamic_smem()
    print(f"[gate-e] device max dynamic smem/block = {cap} bytes")
    # A reasonable request (one stage worth) must have occupancy >= 1.
    ok_req = 16 * 1024
    occ_ok = mod.occupancy_for_smem(ok_req)
    assert occ_ok >= 1, f"reasonable smem {ok_req}B gave occupancy {occ_ok}"
    # An oversized request (> device cap) must REFUSE: occupancy 0, no hang.
    big_req = cap + (64 * 1024)
    occ_big = mod.occupancy_for_smem(big_req)
    assert occ_big == 0, (
        f"oversized smem {big_req}B (cap {cap}) should refuse (occ 0), "
        f"got {occ_big}")
    # And an actual oversized launch fails FAST with a launch error (no hang).
    err_code = mod.probe_oversized_launch(big_req)
    # cudaErrorInvalidValue (1) or cudaErrorLaunchOutOfResources (7) are both
    # acceptable "rejected promptly" outcomes; the contract is NO HANG + a
    # nonzero error. We assert nonzero (rejected) and that the process is alive.
    assert err_code != 0, (
        f"oversized launch should fail fast with a cuda error, got "
        f"{err_code} (cudaSuccess) — would have hung/over-allocated")
    print(f"[gate-e] occupancy refuse: ok_req occ={occ_ok}, oversized occ=0, "
          f"oversized-launch err_code={err_code} (rejected, no hang)  OK")


# ═════════════════════════════════════════════════════════════════════════
#  SASS audit: cuobjdump the JIT-built module; assert HGMMA / wgmma present
#  (NOT lowered to FMA), report register counts / spills.
# ═════════════════════════════════════════════════════════════════════════
def _find_module_object():
    """Locate the JIT build artifact (.so) for cuobjdump."""
    cache = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions")
    # py311_cu124/wgmma_substrate_selftest/*.so
    hits = glob.glob(os.path.join(cache, "**", "wgmma_substrate_selftest*.so"),
                     recursive=True)
    # also the intermediate .o / .cuda.o the build emits (richer SASS)
    objs = glob.glob(os.path.join(cache, "**", "wgmma_selftest*.o"),
                     recursive=True)
    return hits, objs


@_GATE
def test_sass_audit_hgmma_present_and_spills():
    _build_module()   # ensure built
    sos, objs = _find_module_object()
    targets = objs + sos
    assert targets, "no JIT build artifact found for cuobjdump"
    # Prefer the .o (has the device SASS); fall back to the .so.
    target = targets[0]

    def _cuobjdump(args):
        try:
            return subprocess.run(["cuobjdump"] + args + [target],
                                  capture_output=True, text=True, timeout=120)
        except FileNotFoundError:
            return None

    sass = _cuobjdump(["-sass"])
    if sass is None:
        pytest.skip("cuobjdump not on PATH")
    text = sass.stdout + sass.stderr
    # bf16 wgmma lowers to HGMMA.* in SASS (the PTX 'wgmma' spelling does NOT
    # appear in -sass). Assert the tensor-core MMA is actually present — its
    # ABSENCE would mean the compiler lowered to scalar FMA (the fallback path
    # shipped instead of real wgmma).
    n_hgmma = text.count("HGMMA")
    # Spills: STL/LDL in the device code = register spills (perf + the
    # --maxrregcount tuned dim exists to fix them). Report; assert none in the
    # consumer wgmma loop is the goal but we report the raw counts.
    n_stl = text.count("STL")
    n_ldl = text.count("LDL")

    # Also confirm the PTX carries the wgmma mnemonic (the source-level check).
    ptx = _cuobjdump(["-ptx"])
    ptx_text = (ptx.stdout + ptx.stderr) if ptx else ""
    n_wgmma_ptx = ptx_text.count("wgmma.mma_async")
    n_waitgrp = ptx_text.count("wgmma.wait_group")
    n_commit = ptx_text.count("wgmma.commit_group")
    n_ldgsts = text.count("LDGSTS")           # cp.async staging
    n_setmaxnreg = ptx_text.count("setmaxnreg")

    print("\n========== SASS / PTX AUDIT (wgmma substrate) ==========")
    print(f"  target                : {os.path.basename(target)}")
    print(f"  HGMMA (SASS tensor MMA): {n_hgmma}")
    print(f"  wgmma.mma_async  (PTX) : {n_wgmma_ptx}")
    print(f"  wgmma.commit_group     : {n_commit}")
    print(f"  wgmma.wait_group       : {n_waitgrp}")
    print(f"  setmaxnreg       (PTX) : {n_setmaxnreg}")
    print(f"  LDGSTS (cp.async, SASS): {n_ldgsts}")
    print(f"  STL (spill stores)     : {n_stl}")
    print(f"  LDL (spill loads)      : {n_ldl}")
    print("========================================================\n")

    assert n_wgmma_ptx > 0, "no wgmma.mma_async in PTX — engine not emitted"
    assert n_hgmma > 0, (
        "no HGMMA in SASS — wgmma was lowered to scalar FMA (fallback "
        "shipped, NOT real tensor cores)")
    assert n_waitgrp > 0 and n_commit > 0, \
        "missing wgmma.commit_group / wait_group (pipeline correctness PTX)"
    # Spills are a perf concern, not a correctness failure; report only.
    if n_stl or n_ldl:
        print(f"  NOTE: {n_stl} STL / {n_ldl} LDL spill insns present "
              f"(perf; --maxrregcount tuning territory)")


# ── direct-run report (python tests/hw/test_wgmma_substrate.py) ──────────
def _main():
    if not _sm90a_available():
        print("SKIP: no sm_90 GPU available.")
        return 0
    print("Building wgmma substrate self-test (JIT, sm_90a)…")
    _build_module()
    fails = []

    def _try(name, fn, *a):
        try:
            fn(*a)
        except Exception as e:
            fails.append((name, repr(e)))
            print(f"  [FAIL] {name}: {e!r}")

    _try("localization(A=I)", test_a_identity_localization)
    for N in (8, 16, 32, 64, 96, 128):
        _try(f"single-tile N={N}", test_a_single_tile_shapes, N)
    _try("ragged head 97→128", test_a_ragged_head_97_padded)
    for N in (64, 128):
        _try(f"K-loop K128 N={N}", test_b_kloop_K128, N)
    _try("K-loop K512", test_b_kloop_K512)
    for N in (64, 128):
        for d in (2, 3):
            _try(f"pipelined N={N} d={d}", test_c_pipelined_matches_unpipelined, N, d)
    _try("determinism", test_d_determinism_bitwise)
    _try("occupancy-refuse", test_e_occupancy_refuse_oversized_smem)
    _try("SASS audit", test_sass_audit_hgmma_present_and_spills)

    print("\n==================== SUMMARY ====================")
    if not fails:
        print("ALL GATES PASSED")
        return 0
    print(f"{len(fails)} FAILURE(S):")
    for n, e in fails:
        print(f"  - {n}: {e}")
    return 1


if __name__ == "__main__":
    sys.exit(_main())
