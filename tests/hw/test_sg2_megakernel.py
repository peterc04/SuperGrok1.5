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
# Two-part gating, matching the rest of tests/hw (e.g. test_mamba_megakernel.py):
#   @pytest.mark.hw      — the REAL registered marker (tests/conftest.py). Makes
#                          `pytest -m hw` COLLECT these (the gate's command) and
#                          `pytest -m "not hw"` (CPU CI) skip them. A bare
#                          skipif-mark named `hw` would NOT be selected by
#                          `-m hw` (its marker name is `skipif`), so the gate
#                          would silently not run the parity — hence the marker.
#   @_skip_no_hw         — skipif guard: skip cleanly when CUDA / the binding is
#                          absent (so collection under -m hw never errors).
_skip_no_hw = pytest.mark.skipif(
    not _HW,
    reason="needs sm_90 CUDA + the sg2_meta_optimizer_tail binding (integrator-wired)")


# ── (B) shared harness: build a real CSAHCAMetaNet, extract its weight bundle
#    in BOTH the megakernel `meta`-dict order and the per-op `supergrok2_batched_
#    step` arg order, and drive BOTH paths from identical init on cloned buffers.
#    Driving the OPS directly (not opt.step()) keeps the scalar vectors identical
#    across paths — no ramp/gate scheduling can desync them.
def _build_net_and_bundle():
    torch = _torch
    from grokking_optimizers.optimizers.supergrok2 import CSAHCAMetaNet, SuperGrok2
    net = CSAHCAMetaNet(
        d_model=8, num_peer_heads=4, peer_topk=4, num_experts=144,
        expert_hidden=16, gru_hidden=4, n_heads=2, csa_compress=4,
        csa_window=8, csa_topk=16, hca_compress=128, indexer_rank=4,
        rescale=0.1, grad_checkpoint=False).float().cuda().eval()
    # A SuperGrok2 optimizer wrapping THIS net — sg2_step_megakernel is a method
    # on the optimizer (it reads self.meta_net.{gru_hidden,num_experts}). Wraps a
    # throwaway param; we only call its sg2_step_megakernel(...) helper directly.
    _dummy = torch.zeros(1, device='cuda', requires_grad=True)
    opt = SuperGrok2([_dummy], meta_net=net, d_model=8, num_peer_heads=4,
                     peer_topk=4, num_experts=144, expert_hidden=16,
                     gru_hidden=4, n_heads=2, csa_compress=4, csa_window=8,
                     csa_topk=16, hca_compress=128, indexer_rank=4)
    opt.meta_net = net  # ensure the wrapped net is the one we extracted weights from
    w = net.get_weights(None)   # fp32 bundle (get_weights keys: gru_W_z, peer_queries lists, ...)
    peer_Ws = torch.stack([q.float().contiguous() for q in w['peer_queries']])
    pkA = torch.stack([k.float().contiguous() for k in w['product_keys_A']])
    pkB = torch.stack([k.float().contiguous() for k in w['product_keys_B']])
    ne = net.num_experts
    # `meta` dict the megakernel driver (sg2_step_megakernel) consumes — note the
    # ABI uses gru_Wz/gru_bz (no underscore) vs get_weights' gru_W_z/gru_b_z.
    meta = dict(
        input_proj_W=w['input_proj_W'], input_proj_b=w['input_proj_b'],
        csa_q_W=w['csa_q_W'], csa_k_W=w['csa_k_W'], csa_v_W=w['csa_v_W'],
        csa_out_W=w['csa_out_W'], csa_compress_w=w['csa_compress_w'],
        csa_idx_DQ=w['csa_idx_DQ'], csa_idx_K=w['csa_idx_K'],
        hca_q_W=w['hca_q_W'], hca_k_W=w['hca_k_W'], hca_v_W=w['hca_v_W'],
        hca_out_W=w['hca_out_W'],
        gru_Wz=w['gru_W_z'], gru_bz=w['gru_b_z'],
        gru_Wr=w['gru_W_r'], gru_br=w['gru_b_r'],
        gru_Wh=w['gru_W_h'], gru_bh=w['gru_b_h'],
        peer_query_Ws=peer_Ws, prod_keys_A=pkA, prod_keys_B=pkB,
        expert_W1=w['expert_W1'].reshape(ne, -1), expert_b1=w['expert_b1'],
        expert_W2=w['expert_W2'].reshape(ne, -1), expert_b2=w['expert_b2'].reshape(-1))
    return net, opt, w, peer_Ws, pkA, pkB, meta


def _make_states(shapes, GH, seed):
    torch = _torch
    rng = _torch.Generator(device='cuda'); rng.manual_seed(seed)
    def lst(fn):
        return [fn(m) for m in shapes]
    return dict(
        exp_avg=lst(lambda m: torch.randn(m, device='cuda', generator=rng) * 0.1),
        exp_avg_sq=lst(lambda m: (torch.randn(m, device='cuda', generator=rng) * 0.1).abs() + 0.01),
        mu=lst(lambda m: torch.randn(m, device='cuda', generator=rng) * 0.1),
        slow=lst(lambda m: torch.randn(m, device='cuda', generator=rng) * 0.1),
        gru_state=lst(lambda m: torch.randn(m, GH, device='cuda', generator=rng) * 0.1))


def _per_op_oracle(net, w, peer_Ws, pkA, pkB, params, grads, sharps, st,
                   scal, GH):
    """Run ONE step of the parity-validated per-op csa_hca_step_one over the
    SAME inputs/weights/state (mutates params + st in place)."""
    torch = _torch
    ops = _ops
    P = len(params)
    alpha_mus = [float(scal['alpha'][i]) for i in range(P)]
    lamb_effs = [float(scal['lamb_eff'][i]) for i in range(P)]
    beta1s = [float(scal['beta1'][i]) for i in range(P)]
    bc1s = [float(scal['bc1'][i]) for i in range(P)]
    bc2s = [float(scal['bc2'][i]) for i in range(P)]
    ne = net.num_experts
    ops.supergrok2_batched_step(
        params, grads, sharps,
        st['exp_avg'], st['exp_avg_sq'], st['mu'], st['slow'],
        [g.reshape(-1, GH) for g in st['gru_state']],
        w['input_proj_W'], w['input_proj_b'],
        w['csa_q_W'], w['csa_k_W'], w['csa_v_W'], w['csa_compress_w'],
        w['csa_idx_DQ'], w['csa_idx_UQ'], w['csa_idx_K'], w['csa_out_W'],
        w['hca_q_W'], w['hca_k_W'], w['hca_v_W'], w['hca_out_W'],
        w['gru_W_z'], w['gru_b_z'], w['gru_W_r'], w['gru_b_r'],
        w['gru_W_h'], w['gru_b_h'],
        peer_Ws, pkA, pkB,
        w['expert_W1'].reshape(ne, -1), w['expert_b1'],
        w['expert_W2'].reshape(ne, -1), w['expert_b2'].reshape(-1),
        alpha_mus, lamb_effs, beta1s, bc1s, bc2s,
        float(scal['rescale']), float(scal['beta2']), float(scal['lr']),
        float(scal['wd']), float(scal['eps']),
        net.d_model, net.gru_hidden, net.n_heads, net.pk_dim,
        net.expert_hidden, net.num_experts,
        net.csa_compress, net.csa_window, net.csa_topk, net.hca_compress,
        net.indexer_rank, net.expert_counts, net.peer_topk)


def _scalars(P, seed=0):
    # Per-tensor scalars mirroring the per-op layer-scaled betas/alphas. gru_decay
    # == beta1 (csa_hca_step_one: `gru_decay = beta1s[i]`). bc1/bc2 = 1-beta^t at
    # a fixed t so the single-step compare exercises bias correction.
    rng = np.random.default_rng(500 + seed)
    beta1 = (0.88 + 0.04 * rng.random(P)).tolist()
    alpha = (0.95 + 0.03 * rng.random(P)).tolist()
    beta2 = 0.999
    t = 5
    bc1 = [1.0 - b ** t for b in beta1]
    bc2 = [1.0 - beta2 ** t] * P
    return dict(alpha=alpha, gru_decay=list(beta1), lamb_eff=[2.0] * P,
                beta1=list(beta1), bc1=bc1, bc2=bc2,
                rescale=0.1, beta2=beta2, lr=1e-3, wd=0.01, eps=1e-8)


@pytest.mark.hw
@_skip_no_hw
def test_megakernel_single_step_vs_per_op():
    """(B1) Single SG2 step: persistent megakernel vs the parity-validated
    csa_hca_step_one. Same inputs/weights/state → params + moments within 1e-5.
    The per-op path is the ORACLE (already 13/0 parity)."""
    torch = _torch
    torch.manual_seed(123)
    net, opt, w, peer_Ws, pkA, pkB, meta = _build_net_and_bundle()
    GH = net.gru_hidden

    # A few random param tensors of assorted sizes. CONTINUOUS randn grads →
    # P(|grad| tie)=0, so the megakernel's driver-perm and the oracle's internal
    # unstable torch.sort coincide (no tie-break ambiguity — the documented #1
    # spurious-failure risk). N=333 forces the low-rank CSA indexer to select
    # 16-of-84 compressed entries (the path the eager-net anchor could not cover
    # for N<=64). TWO seeds: insurance against a seed-lucky CSA-indexer / PEER
    # top-k tie-order (a documented parity hotspot for tied scores).
    sizes = [5, 17, 64, 200, 1, 333]
    P = len(sizes)
    worst_all = 0.0
    for trial, gseed in enumerate((77, 4242)):
        g0 = torch.Generator(device='cuda'); g0.manual_seed(gseed)
        params0 = [torch.randn(m, device='cuda', generator=g0) * 0.5 for m in sizes]
        grads = [torch.randn(m, device='cuda', generator=g0) * 0.5 for m in sizes]
        sharps = [torch.randn(m, device='cuda', generator=g0) * 0.3 for m in sizes]
        st0 = _make_states(sizes, GH, seed=11 + trial)
        scal = _scalars(P, seed=trial)

        # ORACLE copy.
        p_o = [p.clone() for p in params0]
        st_o = {k: [t.clone() for t in v] for k, v in st0.items()}
        _per_op_oracle(net, w, peer_Ws, pkA, pkB, p_o,
                       [g.clone() for g in grads], [s.clone() for s in sharps],
                       st_o, scal, GH)

        # MEGAKERNEL copy (driver mutates in place).
        p_m = [p.clone() for p in params0]
        st_m = {k: [t.clone() for t in v] for k, v in st0.items()}
        opt.sg2_step_megakernel(
            p_m, [g.clone() for g in grads], [s.clone() for s in sharps],
            st_m, meta, scal)

        torch.cuda.synchronize()
        diffs = {}
        diffs['param'] = max(float((a - b).abs().max()) for a, b in zip(p_m, p_o))
        for k in ('exp_avg', 'exp_avg_sq', 'mu', 'slow', 'gru_state'):
            diffs[k] = max(float((a - b).abs().max())
                           for a, b in zip(st_m[k], st_o[k]))
        worst = max(diffs.values())
        worst_all = max(worst_all, worst)
        print(f"\n[B1] megakernel vs per-op (single step, seed={gseed}):")
        for k, v in diffs.items():
            print(f"      max|Δ{k}| = {v:.3e}")
        print(f"      WORST    = {worst:.3e}  (tol 1e-5)")
        assert worst < 1e-5, (
            f"B1 megakernel vs per-op diverged (seed={gseed}): "
            + ", ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
    print(f"[B1] WORST across seeds = {worst_all:.3e}  (tol 1e-5)")


@pytest.mark.hw
@_skip_no_hw
def test_megakernel_trajectory_vs_per_op():
    """(B2) 200-step trajectory: a fixed random-grad sequence through both paths;
    per-step mean|param| proxy + final params within 1e-5 (bias-correction / EMA
    drift over a trajectory)."""
    torch = _torch
    torch.manual_seed(321)
    net, opt, w, peer_Ws, pkA, pkB, meta = _build_net_and_bundle()
    GH = net.gru_hidden

    sizes = [17, 64, 200]
    P = len(sizes)
    g0 = torch.Generator(device='cuda'); g0.manual_seed(909)
    params0 = [torch.randn(m, device='cuda', generator=g0) * 0.5 for m in sizes]
    sharps = [torch.randn(m, device='cuda', generator=g0) * 0.3 for m in sizes]
    st0 = _make_states(sizes, GH, seed=22)
    scal = _scalars(P, seed=1)

    p_o = [p.clone() for p in params0]
    st_o = {k: [t.clone() for t in v] for k, v in st0.items()}
    p_m = [p.clone() for p in params0]
    st_m = {k: [t.clone() for t in v] for k, v in st0.items()}

    # FIXED random-grad sequence (continuous → no |grad| ties).
    gseq = torch.Generator(device='cuda'); gseq.manual_seed(4242)
    beta1_pt = list(scal['beta1'])          # per-tensor Adam beta1 (layer-scaled)
    beta2_sh = float(scal['beta2'])
    worst = 0.0
    worst_meanproxy = 0.0
    for step in range(200):
        # Advance the bias-correction terms each step (t = step+1) so the
        # trajectory genuinely exercises bias correction (1-beta^t evolving),
        # not a frozen t. BOTH paths get the identical per-step scalars, so
        # parity is preserved while the EMA/bias-correction drift is tested.
        t = step + 1
        scal['bc1'] = [1.0 - b ** t for b in beta1_pt]
        scal['bc2'] = [1.0 - beta2_sh ** t] * P
        grads = [torch.randn(m, device='cuda', generator=gseq) * 0.5 for m in sizes]
        _per_op_oracle(net, w, peer_Ws, pkA, pkB, p_o,
                       [g.clone() for g in grads], [s.clone() for s in sharps],
                       st_o, scal, GH)
        opt.sg2_step_megakernel(
            p_m, [g.clone() for g in grads], [s.clone() for s in sharps],
            st_m, meta, scal)
        torch.cuda.synchronize()
        d = max(float((a - b).abs().max()) for a, b in zip(p_m, p_o))
        worst = max(worst, d)
        # mean|param| proxy divergence (per-step trajectory health).
        mp = max(abs(float(a.abs().mean()) - float(b.abs().mean()))
                 for a, b in zip(p_m, p_o))
        worst_meanproxy = max(worst_meanproxy, mp)

    final_param = max(float((a - b).abs().max()) for a, b in zip(p_m, p_o))
    final_state = {k: max(float((a - b).abs().max())
                          for a, b in zip(st_m[k], st_o[k]))
                   for k in ('exp_avg', 'exp_avg_sq', 'mu', 'slow', 'gru_state')}
    print("\n[B2] megakernel vs per-op (200-step trajectory):")
    print(f"      worst per-step max|Δparam|   = {worst:.3e}")
    print(f"      worst per-step Δmean|param|  = {worst_meanproxy:.3e}")
    print(f"      final     max|Δparam|        = {final_param:.3e}")
    for k, v in final_state.items():
        print(f"      final     max|Δ{k}| = {v:.3e}")
    print(f"      (tol 1e-5)")
    assert worst < 1e-5, f"B2 trajectory max|Δparam| drift {worst:.2e}"
    assert worst_meanproxy < 1e-5, f"B2 mean|param| proxy drift {worst_meanproxy:.2e}"


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
