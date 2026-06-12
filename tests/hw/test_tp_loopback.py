"""tests/hw/test_tp_loopback.py — TENSOR-PARALLEL loopback gate (1 GPU).

IMPLEMENTS /workspace/.parallelism_design.md §5.1/§5.2 via §7 (task #25 phase-2,
Lane C): the Megatron column→row TP pair — col-parallel wgmma GEMM, local gelu,
row-parallel partial GEMM, publish → rendezvous → FIXED-ORDER ascending-pe fp32
all-reduce (fwd comm point ②), the conjugate backward (bwd comm point ②'), and
the exact-slice dW/db shards — on TP ∈ {2,4} VIRTUAL ranks sharing one GPU via
``LoopbackTransport`` (tests/hw/tp_loopback_binding.cu). The math runs on the
PRODUCTION wgmma tile GEMMs (dectc_gemm_fwd_f32 / dectc_gemm_dx_f32) at the
production ffn shapes (d=128, dff=512, kTileM=128).

THE TRANSPORT SWAP POINT: this gate validates everything about in-kernel TP
EXCEPT the physical transport. Swapping ``LoopbackTransport`` →
``NvshmemTransport`` (csrc/fused/sm_90/tp_transport.cuh, -DSG_HAS_NVSHMEM) is
the 8×H100-window task (design §5.4 go/no-go); NO math in the binding changes.

THE FIVE ASSERTS:
  (a) CROSS-RANK IDENTITY: all P virtual ranks' reduced Y1/dX are BIT-IDENTICAL
      (the fixed-order reduce gives every pe the same bits).
  (b) A/A/A: 3 reruns of the whole TP kernel are BIT-IDENTICAL (the reduce
      order is structural, never timing-dependent).
  (c) TRANSPORT-NEUTRALITY: the TP+transport result == the SERIAL chunked-order
      reference (same shard GEMMs summed ascending-pe in one CTA) BIT-EXACT —
      the transport contributes exactly zero numerical effect.
  (d) SLICE-EXACTNESS: every dW0/db0 (col-parallel) shard == the corresponding
      ROW-slice of the unsharded dW0/db0 BIT-EXACT; every dW1 (row-parallel)
      shard == the COL-slice BIT-EXACT; db1 (replicated) identical on every pe
      and == the unsharded db1 BIT-EXACT. This is what lets P2's dW machinery
      carry over to TP unchanged (tp_layer.cuh).
  (e) vs the UNSHARDED full-K reference: NOT bitwise (the K-contraction is
      reassociated into P chunks) — asserted within the campaign parity tol
      (3e-5 rel, /workspace/.campaign_plan.md:12) and the actual delta printed.

JIT BUILD: own build dir under /workspace/.torch_ext (never the production
_ops). Run (GPU box):
    PYTHONPATH=. python -m pytest tests/hw/test_tp_loopback.py -q -s
    PYTHONPATH=. python tests/hw/test_tp_loopback.py
"""
from __future__ import annotations

import os
import sys

import pytest

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

_THIS = os.path.abspath(__file__)
_REPO = os.path.abspath(os.path.join(os.path.dirname(_THIS), "..", ".."))
_BINDING_CU = os.path.join(_REPO, "tests", "hw", "tp_loopback_binding.cu")
_TORCH_EXT_DIR = os.environ.get("SG_TORCH_EXT_DIR", "/workspace/.torch_ext")
_BUILD_DIR = os.path.join(_TORCH_EXT_DIR, "tp_loopback_bind")

_PARITY_TOL = 3e-5     # campaign fp32 parity baseline (assert (e))


def _sm90a_available() -> bool:
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


_GATE = pytest.mark.skipif(
    not _sm90a_available(),
    reason="TP loopback gate needs an sm_90 (Hopper) GPU + bf16 wgmma")

_MODULE = None


def _build_binding():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.makedirs(_BUILD_DIR, exist_ok=True)
    from torch.utils.cpp_extension import load
    # SAME flags as the production TC TUs (and sharded_optimizer_binding) so the
    # wgmma tile math compiles identically — bit-parity depends on it.
    _MODULE = load(
        name="tp_loopback_bind",
        sources=[_BINDING_CU],
        extra_include_paths=[_REPO],
        build_directory=_BUILD_DIR,
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
            "-DWITH_CUDA",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-gencode=arch=compute_90a,code=compute_90a",
            "-lineinfo",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )
    return _MODULE


def _inputs(mod, seed=20260612):
    """Deterministic problem instance: X/dY1/W0/b0/W1/b1 (CPU fp32, fixed seed),
    sized to the binding's production shapes."""
    g = torch.Generator().manual_seed(seed)
    M, D, DFF = mod.M, mod.D, mod.DFF
    X = torch.randn(M * D, generator=g) * 0.5
    dY1 = torch.randn(M * D, generator=g) * 0.05
    W0 = torch.randn(DFF * D, generator=g) * (1.0 / D ** 0.5)
    b0 = torch.randn(DFF, generator=g) * 0.02
    W1 = torch.randn(D * DFF, generator=g) * (1.0 / DFF ** 0.5)
    b1 = torch.randn(D, generator=g) * 0.02
    return X, dY1, W0, b0, W1, b1


def _run_tp(mod, P, X, dY1, W0s, b0s, W1s, b1, dev):
    M, D, DFF = mod.M, mod.D, mod.DFF
    Y1 = torch.zeros(P * M * D, device=dev)
    dX = torch.zeros(P * M * D, device=dev)
    dW0 = torch.zeros(DFF * D, device=dev)
    db0 = torch.zeros(DFF, device=dev)
    dW1 = torch.zeros(D * DFF, device=dev)
    db1 = torch.zeros(P * D, device=dev)
    mod.tp_loopback_run(P, X.to(dev), dY1.to(dev),
                        W0s.to(dev), b0s.to(dev), W1s.to(dev), b1.to(dev),
                        Y1, dX, dW0, db0, dW1, db1)
    torch.cuda.synchronize()
    return Y1, dX, dW0, db0, dW1, db1


def run_tp_loopback(P, verbose=True):
    """One full TP=P loopback gate. Returns (ok, detail)."""
    mod = _build_binding()
    dev = torch.device("cuda")
    M, D, DFF = mod.M, mod.D, mod.DFF
    F = DFF // P
    X, dY1, W0, b0, W1, b1 = _inputs(mod)
    W0s, b0s, W1s = mod.tp_pack_pair_shards(P, W0, b0, W1)

    # ── TP runs ×3 (A/A/A). ──
    runs = [_run_tp(mod, P, X, dY1, W0s, b0s, W1s, b1, dev) for _ in range(3)]
    Y1, dX, dW0, db0, dW1, db1 = runs[0]

    # (b) A/A/A bit-identical across the 3 reruns (every output buffer).
    aaa = all(all(torch.equal(a, b) for a, b in zip(runs[0], r)) for r in runs[1:])

    # (a) cross-rank identity of the reduced outputs.
    Y1v = Y1.view(P, M * D)
    dXv = dX.view(P, M * D)
    db1v = db1.view(P, D)
    cross = all(torch.equal(Y1v[0], Y1v[p]) for p in range(1, P)) and \
            all(torch.equal(dXv[0], dXv[p]) for p in range(1, P)) and \
            all(torch.equal(db1v[0], db1v[p]) for p in range(1, P))

    # ── references (unsharded full-K + serial chunked-order). ──
    Y1_ref = torch.zeros(M * D, device=dev)
    dX_ref = torch.zeros(M * D, device=dev)
    dW0_ref = torch.zeros(DFF * D, device=dev)
    db0_ref = torch.zeros(DFF, device=dev)
    dW1_ref = torch.zeros(D * DFF, device=dev)
    db1_ref = torch.zeros(D, device=dev)
    Y1_ch = torch.zeros(M * D, device=dev)
    dX_ch = torch.zeros(M * D, device=dev)
    mod.tp_reference_run(P, X.to(dev), dY1.to(dev),
                         W0.to(dev), b0.to(dev), W1.to(dev), b1.to(dev),
                         W0s.to(dev), b0s.to(dev), W1s.to(dev),
                         Y1_ref, dX_ref, dW0_ref, db0_ref, dW1_ref, db1_ref,
                         Y1_ch, dX_ch)
    torch.cuda.synchronize()

    # (c) transport-neutrality: TP result == serial chunked-order, BIT-EXACT.
    chunked = torch.equal(Y1v[0], Y1_ch) and torch.equal(dXv[0], dX_ch)

    # (d) slice-exactness of the dW/db shards vs the UNSHARDED reference.
    #     col-parallel dW0/db0: rank r owns full rows [r*F,(r+1)*F) — the dense
    #     concat IS the full [DFF, D] in row order, so equality is direct.
    slice_w0 = torch.equal(dW0, dW0_ref)
    slice_b0 = torch.equal(db0, db0_ref)
    #     row-parallel dW1: rank r owns full cols [r*F,(r+1)*F); reassemble.
    dW1_re = torch.empty(D, DFF, device=dev)
    dW1v = dW1.view(P, D, F)
    for r in range(P):
        dW1_re[:, r * F:(r + 1) * F] = dW1v[r]
    slice_w1 = torch.equal(dW1_re.reshape(-1), dW1_ref)
    slice_b1 = torch.equal(db1v[0], db1_ref)

    # (e) vs the unsharded full-K reference (chunk reassociation) — tol, report.
    def _rel(a, b):
        d = (a - b).abs().max().item()
        n = b.abs().max().item() + 1e-30
        return d, d / n
    dY1_abs, dY1_rel = _rel(Y1v[0], Y1_ref)
    dDX_abs, dDX_rel = _rel(dXv[0], dX_ref)
    parity = dY1_rel < _PARITY_TOL and dDX_rel < _PARITY_TOL

    ok = aaa and cross and chunked and slice_w0 and slice_b0 and slice_w1 and slice_b1 and parity
    if verbose:
        print(f"  [TP={P}] (a) cross-rank bit-eq={cross} | (b) A/A/A={aaa} | "
              f"(c) ==chunked-serial={chunked} | (d) dW slices: W0={slice_w0} "
              f"b0={slice_b0} W1={slice_w1} b1={slice_b1} | "
              f"(e) vs unsharded: Y1 rel={dY1_rel:.3e} dX rel={dDX_rel:.3e} "
              f"(tol {_PARITY_TOL:.0e}) parity={parity}", flush=True)
    return ok, dict(aaa=aaa, cross=cross, chunked=chunked,
                    slice_w0=slice_w0, slice_b0=slice_b0, slice_w1=slice_w1,
                    slice_b1=slice_b1, parity=parity,
                    y1_rel=dY1_rel, dx_rel=dDX_rel)


_TP_DEGREES = [2, 4]


@_GATE
@pytest.mark.parametrize("P", _TP_DEGREES)
def test_tp_loopback_block(P):
    """TP=P virtual ranks over LoopbackTransport: cross-rank identity + A/A/A +
    transport-neutrality (bit-exact vs serial chunked-order) + dW slice-exactness
    + parity vs unsharded. Design §5.1/§5.2 (§7 pre-work)."""
    ok, d = run_tp_loopback(P, verbose=True)
    assert d["cross"], f"TP={P}: virtual ranks disagree — the fixed-order reduce is broken"
    assert d["aaa"], f"TP={P}: A/A/A failed — a timing-dependent order leaked in"
    assert d["chunked"], (
        f"TP={P}: TP+transport != serial chunked-order reference. The transport "
        f"changed the numerics — it must be pure data movement + ascending-pe sum.")
    assert d["slice_w0"] and d["slice_b0"] and d["slice_w1"] and d["slice_b1"], (
        f"TP={P}: a dW/db shard is not an exact slice of the unsharded grad "
        f"(W0={d['slice_w0']} b0={d['slice_b0']} W1={d['slice_w1']} b1={d['slice_b1']}) "
        f"— the P2-carries-over-unchanged property is violated.")
    assert d["parity"], (
        f"TP={P}: chunk-reassociation delta exceeds the campaign tol "
        f"(Y1 rel={d['y1_rel']:.3e}, dX rel={d['dx_rel']:.3e} vs {_PARITY_TOL:.0e}).")
    assert ok


def main() -> int:
    if not _sm90a_available():
        print("[tp-loopback] SKIP: no sm_90 GPU")
        return 0
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    fails = []
    for P in _TP_DEGREES:
        ok, d = run_tp_loopback(P, verbose=True)
        if not ok:
            fails.append((P, d))
    if fails:
        print(f"[tp-loopback] FAIL: {[(p,) for p, _ in fails]}")
        return 1
    print(f"[tp-loopback] OK: TP={_TP_DEGREES} — cross-rank identity, A/A/A, "
          f"transport-neutrality, dW slice-exactness all bit-exact; parity vs "
          f"unsharded within tol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
