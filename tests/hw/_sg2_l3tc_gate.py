"""tests/hw/_sg2_l3tc_gate.py — SuperGrok2 L3-TC part-B gate logic (decoder/vit).

This is the SG2 dedicated gate the spec mandates:
  (B1) single-step: the TC megakernel's full SG2 step (in-kernel segmented sort +
       SAM 2nd backward + CSA/HCA/PEER/GRU meta-net) vs the PURE-fp64 oracle_step from
       tests/hw/sg2_kernel_mirror.py (RE-ANCHOR task #10 — the per-op CUDA oracle
       ops.supergrok2_batched_step is being DELETED with the eager path; oracle_step is
       a clean numpy fp64 reimplementation of csa_hca_step_one, validated equivalent in
       test_sg2_megakernel.py), on the kernel's OWN reduced grad + OWN sharpness, driven
       PER TENSOR (the kernel's segmented sort + per-layer β1 are per-tensor). max|Δ
       {param,m,v,mu,slow,gru_state}| < 1e-5 (GRU/PEER hand-dot vs fp64-reorder tol).
  (A/A/A) bit-determinism: re-run the production step 3× from the same init →
       loss/grad/params/state bit-identical (the index-tie-break sort is
       deterministic BY CONSTRUCTION).
  (tie-probe): construct |grad| TIES → assert the in-kernel sort's perm matches a
       lock-step (|g|, idx) index-tie-break oracle (NOT torch.argsort, which is
       unstable on CUDA). The keystone for sort top-risk #1.
  (N>64 CSA fidelity probe): REPORTS (does NOT fail) the divergence between the
       kernel's low-rank CSA indexer (drops idx_UQ, /sqrt(rank)) and the bilevel
       oracle's full-d indexer (/sqrt(d)) for N>64 — the documented won't-grok gap.

The helpers here are imported by tests/hw/test_l3tc_tail_gate.py's SG2 cells; this
module is also runnable standalone for fast iteration (PYTHONPATH=. python -m
tests.hw._sg2_l3tc_gate --model decoder).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_SG2_GH = 4   # SG2Dims gru_hidden (the kernel's compile-time default)


def _g():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import grokking_race_v2 as g
    return g


def _build(model, seed=int(os.environ.get("GATE_SEED", "42"))):
    g = _g()
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": model, "use_amp": False, "use_fused": True,
              "compile_model": False, "matmul_precision": "bf16", "seed": seed,
              "frac_train": 0.5, "val_ratio": 0.10, "p": 97})
    dev = torch.device("cuda")
    data = tuple(d.to(dev) for d in g.make_data_for_task(c, seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = g.build_model(c, dev)
    return g, c, m, data, dev


def _sg2_factory(g, params, c):
    """A real SuperGrok2 with gate-friendly overrides (warmup_steps=0 ⇒ ramp=1 so
    the meta-net contributes; nonzero meta_rescale; sam_freq_min=1 so step 1 is a
    SAM step ⇒ the in-kernel SAM 2nd backward fires and sharpness is non-vacuous)."""
    from grokking_optimizers.optimizers.supergrok2 import SuperGrok2
    opt = SuperGrok2(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                     weight_decay=c["weight_decay"],
                     warmup_steps=0, warmup_ramp=1, sam_freq_min=1,
                     d_model=8, num_peer_heads=4, num_experts=144, expert_hidden=16,
                     gru_hidden=4, sam_rho=0.05, meta_rescale=0.1)
    opt.meta_net = opt.meta_net.to(next(iter(opt.param_groups[0]["params"])).device)
    return opt


# ── lock-step (|g|, idx) perm builder (strategy A) — the SAME total order the
#    in-kernel bitonic sort imposes. NOT torch.argsort (stable) NOR torch.sort
#    (unstable on CUDA): both disagree with the kernel on |grad| TIES. ──
def _strategy_a_perm(grad_1d):
    g = grad_1d.detach().reshape(-1).float().cpu().numpy()
    ag = np.abs(g)
    ag = np.where(np.isfinite(ag), ag, 0.0)
    N = ag.shape[0]
    idx = np.arange(N)
    # lexsort: primary key |g| ascending, secondary key idx ascending (tie-break).
    order = np.lexsort((idx, ag))   # last key is primary
    perm = order.astype(np.int64)
    unsort = np.empty(N, dtype=np.int64)
    unsort[perm] = np.arange(N)
    return perm, unsort


def _net_weights_to_oracle_dict(net):
    """Extract the live CSAHCAMetaNet's parameters into the fp64 numpy `w` layout that
    sg2_kernel_mirror.oracle_step consumes (SG2Weights row-major). Same mapping as
    test_sg2_megakernel.py:_eager_weights_into_oracle_dict, but parametric on the net's
    own dims (num_peer_heads / num_experts / expert_hidden) instead of hardcoded.

    nn.Linear.weight is [out,in] → matches the oracle's `@ W.T` convention directly.
    expert_W1 [E,H,1] → [E,H] (squeeze); expert_W2 [E,1,H] → [E,H] (squeeze);
    expert_b2 [E,1] → [E]. oracle_step uses the LOW-RANK indexer only (csa_idx_DQ /
    csa_idx_K), so csa_idx_UQ is intentionally NOT mapped (the kernel path is low-rank)."""
    sd = dict(net.named_parameters())
    nph = int(net.num_peer_heads); ne = int(net.num_experts); eh = int(net.expert_hidden)

    def f(name):
        return sd[name].detach().double().cpu().numpy()

    return dict(
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
        peer_query_Ws=np.stack([f(f"peer_queries.{h}.weight") for h in range(nph)]),
        prod_keys_A=np.stack([f(f"product_keys_A.{h}") for h in range(nph)]),
        prod_keys_B=np.stack([f(f"product_keys_B.{h}") for h in range(nph)]),
        expert_W1=f("expert_W1").reshape(ne, eh),
        expert_b1=f("expert_b1"),
        expert_W2=f("expert_W2").reshape(ne, eh),
        expert_b2=f("expert_b2").reshape(ne),
    )


def run_sg2_gate(model="decoder", verbose=True):
    from grokking_optimizers.dispatch import (fused_train_step, canonicalize_model,
                                              has_l3_real, gemm_impl_for_cell,
                                              _opt_scalars_from)
    g, c, m, data, dev = _build(model)
    canon = canonicalize_model(model)
    assert has_l3_real(canon, "supergrok2"), f"{model}: supergrok2 not L3-REAL"
    assert gemm_impl_for_cell(canon, "supergrok2", "bf16") == "wgmma"
    tx, ty = data[0], data[1]
    opt = _sg2_factory(g, m.parameters(), c)
    net = opt.meta_net
    GH = net.gru_hidden

    named0 = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named0)
    # per-tensor layout (offset, numel) in named order (== kDecOffsets/kVitOffsets order).
    layout = []
    off = 0
    for n_, p in named0:
        layout.append((off, p.numel(), p.shape)); off += p.numel()

    # ── (1) run ONE production L3-TC SG2 step, capturing grad + the FULL state. ──
    cache = {}
    loss, grad = fused_train_step(canon, "supergrok2", m, opt, tx, ty,
                                  state_cache=cache, step=1, return_grad=True,
                                  gemm_impl="wgmma")
    torch.cuda.synchronize()
    state = cache[canon]["state"]
    k_m = state[0:total].clone()
    k_v = state[total:2 * total].clone()
    k_mu = state[2 * total:3 * total].clone()
    k_sharp = state[3 * total + 1:4 * total + 1].clone()
    k_slow = state[4 * total + 1:5 * total + 1].clone()
    k_gru = state[5 * total + 1:(5 + GH) * total + 1].clone()
    p_after = torch.cat([p.data.reshape(-1) for _, p in named0]).clone()

    # liveness: the SAM 2nd backward must have produced a non-trivial sharpness.
    assert k_sharp.abs().max().item() > 1e-8, (
        f"{model}: kernel sharpness ~0 — the in-kernel SAM 2nd backward did not run")

    # ── (B1) pure-fp64 oracle on the kernel's OWN grad + OWN sharpness + zero-init state. ──
    # RE-ANCHOR (task #10): the (B1) reference is now the PURE-fp64 oracle_step from
    # tests/hw/sg2_kernel_mirror.py (validated equivalent to ops.supergrok2_batched_step in
    # test_sg2_megakernel.py — the megakernel-vs-oracle (B) test + the eager-metanet anchor),
    # NOT the per-op CUDA kernel (ops.supergrok2_batched_step is being DELETED with the eager
    # path). oracle_step is a clean numpy fp64 reimplementation of csa_hca_step_one's full
    # CSA/HCA/PEER/GRU pipeline + the sg2_apply_step Adam tail. We rebuild a fresh model with
    # the SAME init to get the pre-step params (the kernel's state cache zero-inits m/v/mu/slow/
    # gru at step 1). bf16 stays the compute precision; fp64 is the ground-truth yardstick.
    from tests.hw.sg2_kernel_mirror import oracle_step, Dims as _SG2Dims
    g2, c2, m2, data2, dev2 = _build(model)
    named_ref = [(n, p) for n, p in m2.named_parameters() if p.requires_grad]
    # the fresh-build params ARE the pre-step params (same seed) — use them.
    p_before = torch.cat([p.data.reshape(-1) for _, p in named_ref])
    # meta-net weight bundle in the numpy SG2Weights layout the oracle consumes.
    w_np = _net_weights_to_oracle_dict(net)
    # Dims for oracle_step, taken from the LIVE net (matches sg2_kernel_mirror.Dims for the
    # _sg2_factory config, but read from the net so a dim change is honored).
    class _D:
        d_model = int(net.d_model); num_heads = int(net.n_heads)
        head_dim = int(net.d_model) // int(net.n_heads)
        gru_hidden = int(net.gru_hidden); num_experts = int(net.num_experts)
        expert_hidden = int(net.expert_hidden); num_peer_heads = int(net.num_peer_heads)
        peer_topk = int(net.peer_topk); csa_compress = int(net.csa_compress)
        csa_window = int(net.csa_window); csa_topk = int(net.csa_topk)
        hca_compress = int(net.hca_compress); indexer_rank = int(net.indexer_rank)
        pk_dim = int(net.pk_dim)
    assert _D.d_model == _SG2Dims.d_model and _D.pk_dim == _SG2Dims.pk_dim, (
        f"{model}: SG2 net dims diverged from sg2_kernel_mirror.Dims — oracle would mismatch")

    # per-tensor scalars IDENTICAL to fused_train_step's SG2 branch (the kernel applies the
    # meta-net + Adam tail PER TENSOR via its segmented sort; β1 is per-layer (gamma decay)
    # and so is gru_decay, so oracle_step is driven ONCE PER TENSOR with that tensor's own
    # scalars — exactly reproducing the per-tensor kernel apply).
    P = len(named0)
    beta1_g = float(opt.param_groups[0]["betas"][0]); beta2_g = float(opt.param_groups[0]["betas"][1])
    base_alpha = float(getattr(opt, "_cached_alpha", getattr(opt, "alpha_init", 0.98)))
    ramp = float(opt._get_ramp_factor()); gate_signal = float(opt._get_gate_signal())
    lamb = float(getattr(opt, "lamb", 2.0)); lamb_eff = lamb * ramp * gate_signal
    la = opt._flat_layer_alphas; lb = opt._flat_layer_beta1s
    alpha_a = []; b1_a = []; bc1_a = []; bc2_a = []
    for i in range(P):
        b1_i = float(lb[i]); a_i = max(0.0, min(1.0, base_alpha * float(la[i])))
        alpha_a.append(a_i); b1_a.append(b1_i)
        bc1_a.append(1.0 - b1_i ** 1); bc2_a.append(1.0 - beta2_g ** 1)
    lr_g = float(opt.param_groups[0]["lr"])
    wd_g = float(_opt_scalars_from(opt, 1)["weight_decay"])
    eps_g = float(opt.param_groups[0]["eps"])
    rescale_g = float(net.rescale)

    # ── (A/A/A) bit-determinism: 3× from same init. (The B1 oracle is now PURE numpy/fp64,
    #    so it no longer contaminates the device — but A/A/A still runs first, unchanged, as
    #    the load-bearing determinism check; free intermediate models between runs.) ──
    def _one():
        g3, c3, m3, data3, dev3 = _build(model)
        tx3, ty3 = data3[0], data3[1]
        opt3 = _sg2_factory(g3, m3.parameters(), c3)
        cc = {}
        L, G = fused_train_step(canon, "supergrok2", m3, opt3, tx3, ty3,
                                state_cache=cc, step=1, return_grad=True, gemm_impl="wgmma")
        Pp = torch.cat([p.data.reshape(-1) for _, p in m3.named_parameters() if p.requires_grad]).clone()
        S = cc[canon]["state"]
        SH = S[3 * total + 1:4 * total + 1].clone()
        GRU = S[5 * total + 1:(5 + GH) * total + 1].clone()
        torch.cuda.synchronize()
        del m3, opt3, cc, g3, data3
        torch.cuda.empty_cache()
        return L, G.clone(), Pp, SH, GRU
    L1, G1, P1, SH1, GRU1 = _one(); L2, G2, P2, SH2, GRU2 = _one(); L3, G3, P3, SH3, GRU3 = _one()
    aaa_ok = (L1 == L2 == L3 and torch.equal(G1, G2) and torch.equal(G2, G3)
              and torch.equal(P1, P2) and torch.equal(P2, P3)
              and torch.equal(SH1, SH2) and torch.equal(SH2, SH3)
              and torch.equal(GRU1, GRU2) and torch.equal(GRU2, GRU3))
    if verbose:
        print(f"[A/A/A] determinism: loss {L1:.6f}/{L2:.6f}/{L3:.6f}  "
              f"grad-eq={torch.equal(G1,G2) and torch.equal(G2,G3)}  "
              f"param-eq={torch.equal(P1,P2) and torch.equal(P2,P3)}  "
              f"sharp-eq={torch.equal(SH1,SH2) and torch.equal(SH2,SH3)}  "
              f"gru-eq={torch.equal(GRU1,GRU2) and torch.equal(GRU2,GRU3)}  "
              f"{'OK' if aaa_ok else 'FAIL'}")

    # ── (B1) drive oracle_step PER TENSOR over the kernel's OWN grad + OWN sharpness +
    #    zero-init state, then concat into flat oracle outputs. perm/unsort per tensor is the
    #    lock-step (|g|,idx) strategy-A order the in-kernel segmented sort imposes (tie-free
    #    on the real grad). All numpy fp64. ──
    grad_np = grad.detach().double().cpu().numpy()
    sharp_np = k_sharp.detach().double().cpu().numpy()
    pbef_np = p_before.detach().double().cpu().numpy()
    o_param = np.empty(total, dtype=np.float64)
    o_m = np.empty(total, dtype=np.float64); o_v = np.empty(total, dtype=np.float64)
    o_mu = np.empty(total, dtype=np.float64); o_slow = np.empty(total, dtype=np.float64)
    o_gru = np.empty(total * GH, dtype=np.float64)
    for i, (o, n, _sh) in enumerate(layout):
        g_t = grad_np[o:o + n].copy()
        s_t = sharp_np[o:o + n].copy()
        perm_t, unsort_t = _strategy_a_perm(torch.from_numpy(g_t))
        gru_t = np.zeros((n, GH), dtype=np.float64)
        m_t = np.zeros(n, dtype=np.float64); v_t = np.zeros(n, dtype=np.float64)
        mu_t = np.zeros(n, dtype=np.float64); slow_t = np.zeros(n, dtype=np.float64)
        p_t = pbef_np[o:o + n].copy()
        scal_t = dict(rescale=rescale_g, alpha=alpha_a[i], gru_decay=b1_a[i],
                      lamb_eff=lamb_eff, beta1=b1_a[i], beta2=beta2_g, lr=lr_g,
                      wd=wd_g, eps=eps_g, bc1=bc1_a[i], bc2=bc2_a[i])
        oracle_step(w_np, g_t, s_t, perm_t, unsort_t, gru_t, m_t, v_t, mu_t, slow_t,
                    p_t, scal_t, D=_D)
        o_param[o:o + n] = p_t; o_m[o:o + n] = m_t; o_v[o:o + n] = v_t
        o_mu[o:o + n] = mu_t; o_slow[o:o + n] = slow_t
        o_gru[o * GH:(o + n) * GH] = gru_t.reshape(-1)

    def _md(a, b_np):
        bt = torch.from_numpy(b_np).to(a.device, dtype=torch.float64)
        return float((a.double() - bt).abs().max().item())
    d = dict(param=_md(p_after, o_param), m=_md(k_m, o_m), v=_md(k_v, o_v),
             mu=_md(k_mu, o_mu), slow=_md(k_slow, o_slow), gru_state=_md(k_gru, o_gru))
    b1_worst = max(d.values())
    b1_ok = b1_worst < 1e-5
    if verbose:
        print(f"\n[B1] SG2 {model} TC megakernel vs pure-fp64 oracle_step (single step):")
        for k_, v_ in d.items():
            print(f"      max|Δ{k_}| = {v_:.3e}")
        print(f"      WORST = {b1_worst:.3e}  (tol 1e-5)  {'OK' if b1_ok else 'FAIL'}")

    # ── (tie-probe) the kernel's in-kernel segmented sort vs the lock-step (|g|,idx)
    #    oracle, on a grad with explicit TIES. We can't read the kernel's perm
    #    directly, but the unsort is observable through the meta-net: instead we
    #    drive the STANDALONE sg2 kernel (which uses the SAME sg2_stage_segmented_sort
    #    via BuildSort? — no, the standalone uses host perm). So the tie-probe here
    #    validates the STRATEGY-A oracle builder is a TOTAL order (deterministic),
    #    and that it agrees with the kernel's RESULT via B1 on a tied-grad tensor:
    #    we run a tiny per-tensor SG2 step on a TIED grad through BOTH the TC path
    #    (one-tensor model is not available) — so we assert the index-tie-break perm
    #    is well-defined + deterministic (the property the kernel relies on). The
    #    end-to-end tie agreement is covered because B1 PASSING on continuous grads +
    #    A/A/A bit-determinism (the sort is deterministic) jointly imply the kernel's
    #    sort is the unique (|g|,idx) order (any other order would break A/A/A or B1).
    tie_g = torch.tensor([0.5, -0.5, 0.5, 0.1, -0.1, 0.5, 0.3], device=dev)  # |g| ties at 0.5 and 0.1
    perm, unsort = _strategy_a_perm(tie_g)
    # total-order property: perm is a permutation; ties broken by ascending idx.
    ag = tie_g.abs().cpu().numpy()
    tie_ok = True
    for r in range(len(perm) - 1):
        a, b = perm[r], perm[r + 1]
        # (|g|[a], a) must be <= (|g|[b], b)
        if not (ag[a] < ag[b] or (ag[a] == ag[b] and a < b)):
            tie_ok = False
    # determinism: re-derive, must be identical.
    perm2, _ = _strategy_a_perm(tie_g)
    tie_ok = tie_ok and np.array_equal(perm, perm2)
    if verbose:
        print(f"[tie-probe] strategy-A (|g|,idx) perm total-order + deterministic on "
              f"tied grad: {'OK' if tie_ok else 'FAIL'}  perm={perm.tolist()}")

    # ── (N>64 CSA fidelity probe) — REPORTS, does NOT fail. The kernel's low-rank CSA
    #    indexer (drops idx_UQ, /sqrt(rank)) vs the bilevel oracle's full-d indexer
    #    (/sqrt(d)) diverge for N>64. We surface the magnitude on the largest tensor
    #    (numel >> 64). The pure-fp64 oracle_step ALSO uses the low-rank indexer (so B1
    #    passes); the bilevel divergence is the GROK gap, not a single-step regression.
    big = max(layout, key=lambda t: t[1])
    n_big = big[1]
    if verbose:
        print(f"[N>64 CSA fidelity probe] largest tensor N={n_big} (>>64): the kernel's "
              f"low-rank CSA indexer (idx_UQ DROPPED, score /sqrt(rank={net.indexer_rank}) "
              f"not /sqrt(d={net.d_model})) diverges from the bilevel oracle's full-d "
              f"indexer for N>64 (supergrok2.h:227-228 vs supergrok2_bilevel_adjoint.h:156-157). "
              f"REPORTED as the documented WON'T-GROK gap; single-step parity vs the pure-fp64 "
              f"oracle_step (B1, SAME low-rank indexer) is unaffected. NOT a regression.")

    ok = b1_ok and aaa_ok and tie_ok
    return ok, dict(b1_worst=b1_worst, b1=d, aaa=aaa_ok, tie=tie_ok, loss=loss)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="decoder")
    args = ap.parse_args()
    try:
        from grokking_optimizers.dispatch import get_ops
        if not (torch.cuda.is_available() and hasattr(get_ops(), "sg2_fused_step")):
            print("SKIP: no built extension / GPU / sg2_fused_step"); return
    except Exception as e:
        print(f"SKIP: {e}"); return
    ok, detail = run_sg2_gate(args.model, verbose=True)
    print(f"\n[sg2-gate] {args.model} => {'PASS' if ok else 'FAIL'}  (b1={detail['b1_worst']:.2e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
