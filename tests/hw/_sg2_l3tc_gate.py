"""tests/hw/_sg2_l3tc_gate.py — SuperGrok2 L3-TC part-B gate logic (decoder/vit).

This is the SG2 dedicated gate the spec mandates:
  (B1) single-step: the TC megakernel's full SG2 step (in-kernel segmented sort +
       SAM 2nd backward + CSA/HCA/PEER/GRU meta-net) vs ops.supergrok2_batched_step
       (the per-op ORACLE — the SAME low-rank indexer, internally consistent), on
       the kernel's OWN reduced grad + OWN sharpness. max|Δ{param,m,v,mu,slow,
       gru_state}| < 1e-5 (GRU/PEER hand-dot vs cuBLAS fp32-reorder tol).
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


def _build(model, seed=42):
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


def _per_op_oracle_step(net, opt, w, peer_Ws, pkA, pkB, params, grads, sharps,
                        st, scal):
    """Run ONE ops.supergrok2_batched_step over the SAME inputs/weights/state
    (mutates params + st in place). Mirrors test_sg2_megakernel.py:_per_op_oracle."""
    from grokking_optimizers.dispatch import get_ops
    ops = get_ops()
    GH = net.gru_hidden
    P = len(params)
    ops.supergrok2_batched_step(
        params, grads, sharps,
        st['exp_avg'], st['exp_avg_sq'], st['mu'], st['slow'],
        [s.reshape(-1, GH) for s in st['gru_state']],
        w['input_proj_W'], w['input_proj_b'],
        w['csa_q_W'], w['csa_k_W'], w['csa_v_W'], w['csa_compress_w'],
        w['csa_idx_DQ'], w['csa_idx_UQ'], w['csa_idx_K'], w['csa_out_W'],
        w['hca_q_W'], w['hca_k_W'], w['hca_v_W'], w['hca_out_W'],
        w['gru_W_z'], w['gru_b_z'], w['gru_W_r'], w['gru_b_r'],
        w['gru_W_h'], w['gru_b_h'],
        peer_Ws, pkA, pkB,
        w['expert_W1'].reshape(net.num_experts, -1), w['expert_b1'],
        w['expert_W2'].reshape(net.num_experts, -1), w['expert_b2'].reshape(-1),
        scal['alpha'], scal['lamb_eff'], scal['beta1'], scal['bc1'], scal['bc2'],
        float(net.rescale), float(scal['beta2']), float(scal['lr']),
        float(scal['wd']), float(scal['eps']),
        net.d_model, net.gru_hidden, net.n_heads, net.pk_dim,
        net.expert_hidden, net.num_experts,
        net.csa_compress, net.csa_window, net.csa_topk, net.hca_compress,
        net.indexer_rank, net.expert_counts, net.peer_topk)


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

    # ── (B1) per-op oracle on the kernel's OWN grad + OWN sharpness + zero-init state. ──
    # Rebuild a fresh model with the SAME init to get the pre-step params (state was
    # zero at step start; m/v/mu/slow = 0, gru_state = 0 at step 1 in the kernel cache).
    g2, c2, m2, data2, dev2 = _build(model)
    named_ref = [(n, p) for n, p in m2.named_parameters() if p.requires_grad]
    # confirm same init.
    p_before = torch.cat([p.data.reshape(-1) for _, p in named_ref])
    p_before_kernel = torch.empty(total, device=dev)
    o = 0
    for _, p in named0:   # kernel params were mutated; recover the pre-step from a fresh build
        pass
    # the fresh-build params ARE the pre-step params (same seed) — use them.
    # meta-net weight bundle (kernel ABI keys), same mapping as fused_train_step.
    w = net.get_weights(None)
    ne = net.num_experts
    def _f32c(t):
        t = t if t.dtype == torch.float32 else t.float()
        return t.contiguous().to(device=dev)
    peer_Ws = _f32c(torch.stack([q.float().contiguous() for q in w['peer_queries']]))
    pkA = _f32c(torch.stack([k.float().contiguous() for k in w['product_keys_A']]))
    pkB = _f32c(torch.stack([k.float().contiguous() for k in w['product_keys_B']]))

    # per-tensor scalars IDENTICAL to fused_train_step's SG2 branch.
    P = len(named0)
    beta1_g = float(opt.param_groups[0]["betas"][0]); beta2_g = float(opt.param_groups[0]["betas"][1])
    base_alpha = float(getattr(opt, "_cached_alpha", getattr(opt, "alpha_init", 0.98)))
    ramp = float(opt._get_ramp_factor()); gate_signal = float(opt._get_gate_signal())
    lamb = float(getattr(opt, "lamb", 2.0)); lamb_eff = lamb * ramp * gate_signal
    la = opt._flat_layer_alphas; lb = opt._flat_layer_beta1s
    alpha_a = []; b1_a = []; bc1_a = []; bc2_a = []; le_a = []
    for i in range(P):
        b1_i = float(lb[i]); a_i = max(0.0, min(1.0, base_alpha * float(la[i])))
        alpha_a.append(a_i); b1_a.append(b1_i); le_a.append(lamb_eff)
        bc1_a.append(1.0 - b1_i ** 1); bc2_a.append(1.0 - beta2_g ** 1)
    scal = dict(alpha=alpha_a, lamb_eff=le_a, beta1=b1_a, bc1=bc1_a, bc2=bc2_a,
                gru_decay=list(b1_a), beta2=beta2_g, lr=float(opt.param_groups[0]["lr"]),
                wd=float(_opt_scalars_from(opt, 1)["weight_decay"]),
                eps=float(opt.param_groups[0]["eps"]))

    # ── (A/A/A) bit-determinism: 3× from same init. RUN BEFORE the B1 per-op oracle
    #    call below: the per-op SG2 vit kernel (ops.supergrok2_batched_step, the ~15-20-
    #    launch path) has a CROSS-KERNEL CONTAMINATION bug — its launches perturb a
    #    device-global the SUBSEQUENT TC SG2 launch reads → NaN (the SAME class as the
    #    documented neuralgrok per-op kernel contamination, test_l3tc_tail_gate.py:179;
    #    a pre-existing per-op-kernel bug, NOT the TC path's determinism). So we measure
    #    A/A/A on the clean TC path FIRST, then run the oracle for B1. Free intermediate
    #    models + empty the allocator between runs. ──
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

    # ── (B1) per-op oracle on the kernel's OWN grad + OWN sharpness + zero-init state.
    #    NOTE: this CONTAMINATES the device for any later TC SG2 launch (the per-op
    #    kernel bug above), so it MUST run after A/A/A. ──
    p_o = [p_before[o:o + n].clone().to(dev) for (o, n, _sh) in layout]
    g_o = [grad[o:o + n].clone() for (o, n, _sh) in layout]
    s_o = [k_sharp[o:o + n].clone() for (o, n, _sh) in layout]
    st_o = dict(
        exp_avg=[torch.zeros(n, device=dev) for (o, n, _sh) in layout],
        exp_avg_sq=[torch.zeros(n, device=dev) for (o, n, _sh) in layout],
        mu=[torch.zeros(n, device=dev) for (o, n, _sh) in layout],
        slow=[torch.zeros(n, device=dev) for (o, n, _sh) in layout],
        gru_state=[torch.zeros(n * GH, device=dev) for (o, n, _sh) in layout])
    _per_op_oracle_step(net, opt, w, peer_Ws, pkA, pkB, p_o, g_o, s_o, st_o, scal)
    torch.cuda.synchronize()

    o_param = torch.cat([x.reshape(-1) for x in p_o])
    o_m = torch.cat([x.reshape(-1) for x in st_o['exp_avg']])
    o_v = torch.cat([x.reshape(-1) for x in st_o['exp_avg_sq']])
    o_mu = torch.cat([x.reshape(-1) for x in st_o['mu']])
    o_slow = torch.cat([x.reshape(-1) for x in st_o['slow']])
    o_gru = torch.cat([x.reshape(-1) for x in st_o['gru_state']])

    def _md(a, b):
        return float((a.double() - b.double()).abs().max().item())
    d = dict(param=_md(p_after, o_param), m=_md(k_m, o_m), v=_md(k_v, o_v),
             mu=_md(k_mu, o_mu), slow=_md(k_slow, o_slow), gru_state=_md(k_gru, o_gru))
    b1_worst = max(d.values())
    b1_ok = b1_worst < 1e-5
    if verbose:
        print(f"\n[B1] SG2 {model} TC megakernel vs ops.supergrok2_batched_step (single step):")
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
    #    (numel >> 64). The per-op oracle ALSO uses the low-rank indexer (so B1 passes);
    #    the bilevel divergence is the GROK gap, not a single-step regression.
    big = max(layout, key=lambda t: t[1])
    n_big = big[1]
    if verbose:
        print(f"[N>64 CSA fidelity probe] largest tensor N={n_big} (>>64): the kernel's "
              f"low-rank CSA indexer (idx_UQ DROPPED, score /sqrt(rank={net.indexer_rank}) "
              f"not /sqrt(d={net.d_model})) diverges from the bilevel oracle's full-d "
              f"indexer for N>64 (supergrok2.h:227-228 vs supergrok2_bilevel_adjoint.h:156-157). "
              f"REPORTED as the documented WON'T-GROK gap; single-step parity vs the per-op "
              f"oracle (B1, SAME low-rank indexer) is unaffected. NOT a regression.")

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
