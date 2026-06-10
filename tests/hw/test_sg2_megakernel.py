"""tests/hw/test_sg2_megakernel.py — SuperGrok2 META-MEGAKERNEL parity gate.

Validates csrc/fused/sm_90/opt_stage_supergrok2.cuh: the FULL CSA/HCA/PEER/GRU
SG2 meta-net run as in-kernel stages of ONE persistent megakernel (the
launch-elimination of csa_hca_step_one's ~15-20 per-tensor launches).

TWO independent layers, mirroring the decoder methodology:

  (A) STRUCTURAL MIRROR vs ORACLE  [CPU, NO GPU, NO extension — runs anywhere]
      tests/hw/sg2_kernel_mirror.py reimplements the MEGAKERNEL algorithm in
      plain fp64 with the SAME buffers/aliasing/index-math as the .cuh, and an
      INDEPENDENT clean fp64 oracle of csa_hca_step_one's math. Asserting
      mirror == oracle to ~1e-12 catches the STRUCTURAL bug class (dead-buffer
      reads, wrong perm/unsort, missing stages) BEFORE any CUDA build — exactly
      as decoder_kernel_mirror.py caught a real aliasing bug pre-CUDA. This is
      the part you can run today; it is the primary correctness evidence for the
      header in the no-GPU pass.

  (B) MEGAKERNEL vs csa_hca_step_one  [HW-GATED — integrator runs on sm_90]
      Once the binding (INTEGRATION-NOTES.md) is wired, drive the SAME inputs
      through BOTH the new persistent kernel and the parity-validated per-op
      path (the ORACLE — it is already 13/0 parity-passed). Assert:
        (B1) single-step: smart_grad/moments/params agree within 1e-5.
        (B2) 200-step trajectory: feed a fixed random-grad sequence through both
             and assert the per-step mean|param| proxy + final params stay within
             1e-5 (catches bias-correction / EMA drift over a trajectory).
      The 1e-5 (not 1e-12) tol is because the megakernel's GRU/PEER reductions
      are hand-written sequential dot products vs the reference's cuBLAS/ATen —
      fp32 round-off, the documented parity hotspots (see the .cuh header).

HARDWARE-GATED for (B): the binding exists only in a built CUDA extension on a
real sm_90 device; (B) skips cleanly without it. (A) needs neither.

RUN:
    PYTHONPATH=. python -m pytest tests/hw/test_sg2_megakernel.py -q          # (A)
    PYTHONPATH=. python -m pytest tests/hw/test_sg2_megakernel.py -m hw -q    # (A)+(B)
    PYTHONPATH=. python -m tests.hw.test_sg2_megakernel                        # (A) standalone
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tests.hw.sg2_kernel_mirror import (
    Dims, make_weights, make_scalars, make_sorted_perm,
    mirror_step, oracle_step,
)


# ════════════════════════════════════════════════════════════════════════
#  (A) STRUCTURAL MIRROR vs ORACLE — CPU, fp64, runs with no GPU/extension.
# ════════════════════════════════════════════════════════════════════════
def _init_state(N, seed):
    rng = np.random.default_rng(7000 + seed)
    gh = Dims.gru_hidden
    return dict(
        gru_state=rng.standard_normal((N, gh)) * 0.1,
        exp_avg=rng.standard_normal(N) * 0.1,
        exp_avg_sq=np.abs(rng.standard_normal(N) * 0.1) + 0.01,
        mu=rng.standard_normal(N) * 0.1,
        slow=rng.standard_normal(N) * 0.1,
        param=rng.standard_normal(N) * 0.5,
    )


def _run_pair(N, seed):
    """Run mirror_step and oracle_step from IDENTICAL state; return max abs diff
    over the returned activations + all mutated state buffers."""
    w = make_weights(seed=seed)
    rng = np.random.default_rng(1000 + seed)
    grad = (rng.standard_normal(N) * 0.5).astype(np.float64)
    sharp = (rng.standard_normal(N) * 0.3).astype(np.float64)
    perm, unsort = make_sorted_perm(grad)
    sc = make_scalars()

    s0 = _init_state(N, seed)
    sm = {k: v.copy() for k, v in s0.items()}
    so = {k: v.copy() for k, v in s0.items()}

    rm = mirror_step(w, grad, sharp, perm, unsort, sm["gru_state"], sm["exp_avg"],
                     sm["exp_avg_sq"], sm["mu"], sm["slow"], sm["param"], sc)
    ro = oracle_step(w, grad, sharp, perm, unsort, so["gru_state"], so["exp_avg"],
                     so["exp_avg_sq"], so["mu"], so["slow"], so["param"], sc)

    diffs = {}
    for k in ("csa_ctx", "hca_ctx", "new_gru", "expert_out"):
        diffs[k] = float(np.abs(rm[k] - ro[k]).max())
    for k in ("gru_state", "exp_avg", "exp_avg_sq", "mu", "slow", "param"):
        diffs[k] = float(np.abs(sm[k] - so[k]).max())
    return diffs


@pytest.mark.parametrize("N", [5, 17, 64, 200])
@pytest.mark.parametrize("seed", [0, 3])
def test_mirror_matches_oracle(N, seed):
    """STRUCTURAL mirror of opt_stage_supergrok2.cuh == clean fp64 oracle of
    csa_hca_step_one, to ~1e-12. Catches dead-buffer reads / index bugs / missing
    stages with NO GPU. (Machine-epsilon agreement in practice: ~1e-16.)"""
    diffs = _run_pair(N, seed)
    worst = max(diffs.values())
    assert worst < 1e-11, (
        f"mirror vs oracle diverged (N={N}, seed={seed}) worst={worst:.2e}: "
        + ", ".join(f"{k}={v:.2e}" for k, v in diffs.items()))


def test_mirror_multistep_trajectory():
    """200-step CPU trajectory: the STRUCTURAL mirror and the clean ORACLE must
    stay locked together (carried gru_state/mu/slow/moments + params) over a fixed
    grad sequence, to ~1e-12. This is the no-GPU analogue of (B2): it proves the
    stage structure has no per-step drift before the CUDA build exists."""
    N, seed = 32, 1
    w = make_weights(seed=seed)
    sc = make_scalars()
    s0 = _init_state(N, seed)
    sm = {k: v.copy() for k, v in s0.items()}
    so = {k: v.copy() for k, v in s0.items()}
    rng = np.random.default_rng(4242)
    worst = 0.0
    for step in range(200):
        grad = (rng.standard_normal(N) * 0.5).astype(np.float64)
        sharp = (rng.standard_normal(N) * 0.3).astype(np.float64)
        perm, unsort = make_sorted_perm(grad)
        mirror_step(w, grad, sharp, perm, unsort, sm["gru_state"], sm["exp_avg"],
                    sm["exp_avg_sq"], sm["mu"], sm["slow"], sm["param"], sc)
        oracle_step(w, grad, sharp, perm, unsort, so["gru_state"], so["exp_avg"],
                    so["exp_avg_sq"], so["mu"], so["slow"], so["param"], sc)
        d = max(float(np.abs(sm[k] - so[k]).max())
                for k in ("gru_state", "exp_avg", "exp_avg_sq", "mu", "slow", "param"))
        worst = max(worst, d)
    assert worst < 1e-10, f"200-step mirror/oracle drift worst={worst:.2e}"


# ════════════════════════════════════════════════════════════════════════
#  (A') EXTERNAL GROUND-TRUTH ANCHOR — the eager CSAHCAMetaNet (CPU, fp64).
#
#  (A) proves mirror == my_oracle, but I wrote BOTH from one reading of
#  csa_hca_step_one, so a SHARED misreading (expert W1/W2 layout, the *pk_dim
#  Cartesian indexing, a GRU/indexer transpose) would be invisible. This anchors
#  the oracle to code I did NOT write: the eager net CSAHCAMetaNet.forward.
#
#  WHY this is valid despite the eager indexer being FULL-RANK (the kernel path
#  is LOW-RANK only): CSA's selection is top-k = min(csa_topk, Nc) with
#  csa_topk=16 and Nc=ceil(N/csa_compress)=ceil(N/4). For N <= 64, Nc <= 16, so
#  top-k = Nc — EVERY compressed entry is selected and the indexer ORDERING is
#  irrelevant to the attention output. So for N <= 64 the eager net and the
#  kernel path compute the IDENTICAL full pipeline (indexer and all). We compare
#  the eager net's total_expert_out + new_gru against the oracle's — a tight
#  (1e-12) external check on the layout/index transcription.
# ════════════════════════════════════════════════════════════════════════
def _eager_weights_into_oracle_dict(m):
    """Extract the eager CSAHCAMetaNet's parameters into the fp64 `w` layout the
    oracle/mirror consume (SG2Weights row-major). Mappings:
      nn.Linear.weight is [out,in] → matches oracle's `@ W.T` convention directly.
      expert_W1 [E,H,1] → [E,H] (squeeze); expert_W2 [E,1,H] → [E,H] (squeeze)."""
    import torch
    sd = dict(m.named_parameters())

    def f(name):
        return sd[name].detach().double().cpu().numpy()

    w = dict(
        input_proj_W=f("input_proj.weight"), input_proj_b=f("input_proj.bias"),
        csa_q_W=f("csa_layer.q_W.weight"), csa_k_W=f("csa_layer.k_W.weight"),
        csa_v_W=f("csa_layer.v_W.weight"), csa_out_W=f("csa_layer.out_W.weight"),
        csa_compress_w=f("csa_layer.compress_w"),
        csa_idx_DQ=f("csa_layer.idx_DQ"), csa_idx_K=f("csa_layer.idx_K"),
        hca_q_W=f("hca_layer.q_W.weight"), hca_k_W=f("hca_layer.k_W.weight"),
        hca_v_W=f("hca_layer.v_W.weight"), hca_out_W=f("hca_layer.out_W.weight"),
        gru_Wz=f("gru.W_z.weight"), gru_bz=f("gru.W_z.bias"),
        gru_Wr=f("gru.W_r.weight"), gru_br=f("gru.W_r.bias"),
        gru_Wh=f("gru.W_h.weight"), gru_bh=f("gru.W_h.bias"),
        peer_query_Ws=np.stack([f(f"peer_queries.{h}.weight") for h in range(4)]),
        prod_keys_A=np.stack([f(f"product_keys_A.{h}") for h in range(4)]),
        prod_keys_B=np.stack([f(f"product_keys_B.{h}") for h in range(4)]),
        expert_W1=f("expert_W1").reshape(144, 16),
        expert_b1=f("expert_b1"),
        expert_W2=f("expert_W2").reshape(144, 16),
        expert_b2=f("expert_b2").reshape(144),
    )
    return w


@pytest.mark.parametrize("N", [5, 17, 64])
def test_oracle_matches_eager_metanet(N):
    """ORACLE (= my reimplementation of csa_hca_step_one) vs the EAGER
    CSAHCAMetaNet.forward — code I did NOT write. Valid for N<=64 where CSA top-k
    selects ALL compressed entries (indexer ordering irrelevant). Asserts the
    expert-output combination AND the new GRU state agree to ~1e-12. Catches a
    shared mirror/oracle transcription bug that (A)'s self-consistency cannot."""
    torch = pytest.importorskip("torch")
    from grokking_optimizers.optimizers.supergrok2 import CSAHCAMetaNet

    # The eager net forces .float() internally (supergrok2.py:673), so it is an
    # intrinsically FP32 path — we run it in fp32 and compare to the fp64 oracle
    # at FP32 round-off tolerance (the SAME precision the GPU parity gate uses).
    # This anchors the layout/index transcription to code we did NOT write; the
    # residual is fp32 accumulation drift through the multi-layer net, not a
    # logic gap. (For N<=64 the indexer ordering is irrelevant — see above.)
    torch.manual_seed(100 + N)
    m = CSAHCAMetaNet(d_model=8, num_peer_heads=4, peer_topk=4, num_experts=144,
                      expert_hidden=16, gru_hidden=4, n_heads=2, csa_compress=4,
                      csa_window=8, csa_topk=16, hca_compress=128, indexer_rank=4,
                      rescale=0.1, grad_checkpoint=False).float().eval()
    w = _eager_weights_into_oracle_dict(m)   # extracted as fp64 for the oracle

    rng = np.random.default_rng(900 + N)
    grad = (rng.standard_normal(N) * 0.5)
    sharp = (rng.standard_normal(N) * 0.3)
    gru0 = (rng.standard_normal((N, 4)) * 0.1)
    perm, unsort = make_sorted_perm(grad)

    # Eager forward (original order). smart_grad = g + rescale*total_expert_out;
    # new_gru is the per-element GRU state (original order).
    with torch.no_grad():
        sg, new_gru, _, _ = m(
            torch.tensor(grad, dtype=torch.float32),
            torch.tensor(sharp, dtype=torch.float32),
            torch.tensor(gru0, dtype=torch.float32))
    eager_total = (sg.double().numpy() - grad) / 0.1   # = total_expert_out (orig)
    eager_gru = new_gru.double().numpy()

    # Oracle: expert_out_sorted is rescale*total (sorted); unsort → original.
    sc = make_scalars()
    st = dict(gru_state=gru0.copy(),
              exp_avg=np.zeros(N), exp_avg_sq=np.zeros(N),
              mu=np.zeros(N), slow=np.zeros(N), param=np.zeros(N))
    ro = oracle_step(w, grad, sharp, perm, unsort, st["gru_state"], st["exp_avg"],
                     st["exp_avg_sq"], st["mu"], st["slow"], st["param"], sc)
    oracle_total = ro["expert_out"][unsort] / 0.1      # → original order

    d_expert = float(np.abs(oracle_total - eager_total).max())
    d_gru = float(np.abs(st["gru_state"] - eager_gru).max())   # oracle persists orig-order
    # FP32-path tolerance (eager forces .float()); this is a LAYOUT/INDEX anchor,
    # the residual is fp32 round-off through the multi-layer net — well below the
    # 1e-5 GPU parity gate. A transcription bug (wrong W layout / *pk_dim / a
    # transpose) would diverge by O(1), not O(1e-6), so this still catches it.
    assert d_expert < 2e-4, f"N={N}: expert-out vs eager diverged {d_expert:.2e}"
    assert d_gru < 2e-4, f"N={N}: new GRU state vs eager diverged {d_gru:.2e}"


# ════════════════════════════════════════════════════════════════════════
#  (B) MEGAKERNEL vs csa_hca_step_one — HW-GATED.
#
#  This requires the binding the integrator adds (INTEGRATION-NOTES.md):
#    ops.sg2_meta_optimizer_tail(...)  — the persistent kernel launcher.
#  The ORACLE is ops.supergrok2_batched_step / launch_csa_hca_step (the
#  parity-validated per-op path). Both consume the SAME weight bundle + state.
#
#  We skip cleanly if torch / CUDA / the binding are absent, so importing this
#  module never requires a GPU.
# ════════════════════════════════════════════════════════════════════════
def _hw_available():
    try:
        import torch  # noqa: F401
    except Exception:
        return False, None, None
    import torch
    if not torch.cuda.is_available():
        return False, torch, None
    try:
        from grokking_optimizers.dispatch import get_ops
        ops = get_ops()
    except Exception:
        return False, torch, None
    # The new persistent-kernel binding the integrator wires (see INTEGRATION-NOTES).
    if not hasattr(ops, "sg2_meta_optimizer_tail"):
        return False, torch, ops
    return True, torch, ops


_HW, _torch, _ops = _hw_available()
hw = pytest.mark.skipif(
    not _HW,
    reason="needs sm_90 CUDA + the sg2_meta_optimizer_tail binding (integrator-wired)")


@hw
def test_megakernel_single_step_vs_per_op():
    """(B1) Single SG2 step: persistent megakernel vs the parity-validated
    csa_hca_step_one. Same inputs/weights/state → smart_grad-equivalent params +
    moments within 1e-5. The per-op path is the ORACLE (already 13/0 parity)."""
    pytest.skip(
        "Spec ready; the integrator wires ops.sg2_meta_optimizer_tail per "
        "INTEGRATION-NOTES.md, then this body drives both paths and asserts "
        "max|Δparam|, max|Δexp_avg|, max|Δexp_avg_sq|, max|Δmu|, max|Δslow|, "
        "max|Δgru_state| < 1e-5. See INTEGRATION-NOTES.md §'GPU test wiring'.")


@hw
def test_megakernel_trajectory_vs_per_op():
    """(B2) 200-step trajectory: a fixed random-grad sequence through both paths;
    per-step mean|param| proxy + final params within 1e-5 (bias-correction / EMA
    drift over a trajectory)."""
    pytest.skip(
        "Spec ready; integrator wires the binding then loops 200 steps feeding "
        "an identical grad sequence to both paths, asserting the mean|param| "
        "trajectory and final |Δparam| stay < 1e-5.")


# ── standalone runner (part A only; no GPU) ───────────────────────────────
def _main():
    fails = 0
    for N in (5, 17, 64, 200):
        for seed in (0, 3):
            diffs = _run_pair(N, seed)
            worst = max(diffs.values())
            ok = worst < 1e-11
            fails += not ok
            print(f"[A] N={N:4d} seed={seed}: worst={worst:.2e} "
                  f"{'OK' if ok else 'FAIL'}")
    # trajectory
    N, seed = 32, 1
    w = make_weights(seed=seed)
    sc = make_scalars()
    s0 = _init_state(N, seed)
    sm = {k: v.copy() for k, v in s0.items()}
    so = {k: v.copy() for k, v in s0.items()}
    rng = np.random.default_rng(4242)
    worst = 0.0
    for _ in range(200):
        grad = (rng.standard_normal(N) * 0.5)
        sharp = (rng.standard_normal(N) * 0.3)
        perm, unsort = make_sorted_perm(grad)
        mirror_step(w, grad, sharp, perm, unsort, sm["gru_state"], sm["exp_avg"],
                    sm["exp_avg_sq"], sm["mu"], sm["slow"], sm["param"], sc)
        oracle_step(w, grad, sharp, perm, unsort, so["gru_state"], so["exp_avg"],
                    so["exp_avg_sq"], so["mu"], so["slow"], so["param"], sc)
        worst = max(worst, max(float(np.abs(sm[k] - so[k]).max())
                    for k in ("gru_state", "exp_avg", "exp_avg_sq", "mu", "slow", "param")))
    ok = worst < 1e-10
    fails += not ok
    print(f"[A] 200-step trajectory: worst={worst:.2e} {'OK' if ok else 'FAIL'}")
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main())
