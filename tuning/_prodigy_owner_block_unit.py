"""Prodigy P2.6 owner-block UNIT test — the d-MOVEMENT arithmetic the multi-step
parity cannot reach.

WHY THIS EXISTS. The in-kernel Prodigy d-estimate (P2.6, fused_{vit,decoder}_
megakernel_tc) reduces cross-ALL-tensors (r,s), decays the PERSISTED (r_ema,s_ema)
by beta3, folds d_coef into the candidate, and updates d = max(d_prev,
d_coef·r_ema/|s_ema|). At production d0=1e-6/d_coef=1.0 this owner-block is INERT
for BOTH decoder + vit — the slow real trajectory <g,p0−p> never trips the
candidate over d0 in <80 steps (eager's d does not move here either). And
trajectory-FORCING is unstable (small d0 won't bootstrap; large d0·lr blows the
model to NaN). So the multi-step PARITY, while green, does NOT exercise the
r_ema/s_ema/d_coef/persist arithmetic: when d is inert, max(d_prev,~0)=d_prev
regardless of whether the persist line is right (a kernel that wrongly persisted
the d_coef-SCALED r_ema would pass every inert gate). INTEGRATION-OPTSTAGES §8's
CPU oracle only covers the INSTANTANEOUS phaseA/phaseB; the beta3-EMA + d_coef +
persist is the §2 host-side part reimplemented in-kernel — novel code unseen there.

THE TEST. Snapshot the step-2-ENTRY params p_e, then INJECT a controlled anchor
param_init = p_e + delta (so p0 − p_e = delta is nonzero + EXACT), reset the
persisted scalars to the cold start (r_ema=s_ema=0, d_lr=d0), run ONE more kernel
step, and read the kernel's PERSISTED [r_ema | s_ema | d_lr]. Compare to the
canonical prodigy.h math in fp64 on the SAME captured TC-reduced grad g_e and the
SAME delta and d_prev=d0:
    r     = Σ_i d0²·g_e[i]·delta[i]               (prodigy.h:11; degree-2 scale-free)
    s     = Σ_i d0²·|g_e[i]|                       (the L1 norm)
    r_ema = beta3·0 + r,  s_ema = beta3·0 + s      (persisted scalar EMA, fresh)
    d     = max(d0, d_coef·r_ema/|s_ema|)          (prodigy_update_d; d_coef on candidate)
d_coef ≠ 1 makes the "persist UNSCALED, scale only the candidate" line LOAD-BEARING;
the controlled delta makes d genuinely MOVE off d0, exercising the max() branch.

Run: CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. python \
        tuning/_prodigy_owner_block_unit.py --model vit
     (--model decoder validates the sibling lane's SAME owner-block.)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

TOL = 5e-4   # fp32 reorder over a cross-ALL-tensors d²-scaled reduction


def main(model="vit", d_coef=3.0, d0=1e-6, delta_scale=1e-2):
    from tests.hw.test_l3tc_tail_gate import _build_cell
    from grokking_optimizers.dispatch import (has_l3_real, gemm_impl_for_cell,
                                              canonicalize_model, fused_train_step)
    canon = canonicalize_model(model)
    assert has_l3_real(canon, "prodigy"), f"{model}×prodigy is not L3-REAL"
    assert gemm_impl_for_cell(canon, "prodigy", "bf16") == "wgmma", "engine != wgmma"

    g, c, m, data, dev = _build_cell(model)
    tx, ty = data[0], data[1]
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named)

    opt = g.Prodigy(m.parameters(), lr=c.get("prodigy_lr", 1.0),
                    weight_decay=c["weight_decay"], betas=(c["beta1"], c["beta2"]),
                    d0=d0, d_coef=d_coef)
    beta3 = float(c["beta2"]) ** 0.5

    # Step 1: populate the cache (seeds param_init=params_0). d inert (r=0).
    cache = {}
    fused_train_step(canon, "prodigy", m, opt, tx, ty, state_cache=cache, step=1,
                     return_grad=True, gemm_impl="wgmma")
    state = cache[canon]["state"]
    # p_e = the step-2-ENTRY params = the params AFTER step 1 (the cache `flat` now holds
    # the updated params; the kernel will re-pack the SAME current params at step-2 entry,
    # so p_e == flat here, AND == the live module params m.named_parameters()).
    p_e = cache[canon]["flat"].clone()

    # INJECT param_init = p_e + delta (controlled, reproducible). Reset persisted
    # scalars to the cold start so step 2 is a clean first d-update from d0.
    gen = torch.Generator(device=dev).manual_seed(12345)
    delta = (torch.rand(total, generator=gen, device=dev) - 0.5) * 2.0 * delta_scale
    state[3 * total + 1:4 * total + 1].copy_(p_e + delta)
    cache["param_init_seeded"] = True
    state[4 * total + 1] = 0.0     # r_ema
    state[4 * total + 2] = 0.0     # s_ema
    state[4 * total + 3] = d0      # d_lr (persisted d_prev)

    # Step 2: the kernel re-packs the CURRENT params (== p_e) into flat, so p0 − p = delta
    # EXACTLY for the P2.6 reduction. Capture g_e = the reduced grad the P2.6 used.
    _loss2, g_e = fused_train_step(canon, "prodigy", m, opt, tx, ty,
                                   state_cache=cache, step=2, return_grad=True,
                                   gemm_impl="wgmma")
    k_r = float(state[4 * total + 1].item())
    k_s = float(state[4 * total + 2].item())
    k_d = float(state[4 * total + 3].item())

    # CANONICAL reference (fp64) — r/s with d_prev=d0, EMA from a fresh (0,0).
    ge = g_e.double()
    dl = delta.double()
    d0d = float(d0)
    r = (d0d * d0d) * float((ge * dl).sum().item())
    s = (d0d * d0d) * float(ge.abs().sum().item())
    r_ema = beta3 * 0.0 + r
    s_ema = beta3 * 0.0 + s
    cand = float(d_coef) * r_ema / (abs(s_ema) + 0.0 if s_ema != 0 else 1.0)
    d_ref = max(d0d, cand)

    def rel(a, b):
        return abs(a - b) / (abs(b) + 1e-30)

    r_rel = rel(k_r, r_ema)
    s_rel = rel(k_s, s_ema)
    d_rel = rel(k_d, d_ref)
    d_moved = k_d > d0 * (1.0 + 1e-6)

    print(f"[prodigy-owner-unit] model={model} d0={d0:.1e} d_coef={d_coef} "
          f"delta_scale={delta_scale:.1e}", flush=True)
    print(f"  kernel persisted: r_ema={k_r:.6e} s_ema={k_s:.6e} d_lr={k_d:.6e}", flush=True)
    print(f"  canonical ref:    r_ema={r_ema:.6e} s_ema={s_ema:.6e} d={d_ref:.6e}", flush=True)
    print(f"  rel: r={r_rel:.3e} s={s_rel:.3e} d={d_rel:.3e}  (tol {TOL:.1e})", flush=True)
    print(f"  d MOVED off d0 (owner-block branch exercised): {d_moved} "
          f"(candidate d_coef·r/|s| = {cand:.6e} vs d0 {d0:.1e})", flush=True)

    ok = (r_rel < TOL and s_rel < TOL and d_rel < TOL and d_moved
          and k_r == k_r and k_s == k_s and k_d == k_d)   # finite
    print(f"  => {'PASS — in-kernel d-update (beta3-EMA + d_coef + persist) matches canonical' if ok else 'FAIL'}",
          flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit", help="vit | decoder")
    ap.add_argument("--d-coef", type=float, default=3.0)
    ap.add_argument("--d0", type=float, default=1e-6)
    ap.add_argument("--delta-scale", type=float, default=1e-2)
    a = ap.parse_args()
    main(a.model, d_coef=a.d_coef, d0=a.d0, delta_scale=a.delta_scale)
