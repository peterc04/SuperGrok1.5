#!/usr/bin/env python3
"""H100 (sm_90) single-step fused-vs-reference PARITY GATE.

This gate validates the kernels touched by the H100 audit against canonical
references, and crucially EXERCISES the fixed code paths (not a neutralized
Adam tail that would be blind to the meta-net / d-adaptation bugs):

  * adamw / lion / muon : sanity anchors (known-good) + Muon weight-decay
                          re-validation. If these pass at ~1e-6 the harness
                          wiring is sound.
  * prodigy             : the adaptive d no longer CATAPULTS. The pre-fix
                          degree-(-1) estimator blew d from its 1e-6 init to
                          ~0.2 in a single step (1e6x); the scale-free fix keeps
                          d growing smoothly. Cross-checked vs external
                          prodigyopt when installed.
  * supergrok11/15      : the meta-net output mu == rescale*GELU-MLP(grad). The
                          pre-fix kernel hard-coded H=64 and read 32 floats OOB
                          (canonical hidden_dim=32), corrupting mu. We zero the
                          sharpness column of W1 so mu is a pure, exactly
                          reproducible function of grad.
  * supergrok2          : per-head PEER routing RUNS (pre-fix it raised
                          `shape '[-1, 44]' is invalid for input of size 192`).

A real kernel bug diverges by 1e-1..1e+3; a correct fp32 kernel matches the
fp64 reference at ~1e-6. Tolerances are drawn well inside that gap.

Run:  PYTHONPATH=. python3 tests/hw/parity_gate_h100.py
Exit code 0 iff every gate passes.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")
from tests.hw.test_reference_parity import (  # noqa: E402
    ref_adamw_step, ref_grokadamw_step, ref_lion_step, ref_muon_step,
    ref_sg11_step,
)
import grokking_optimizers as go            # noqa: E402,F401  (ensures _ops loads)
import grokking_optimizers._ops as ops      # noqa: E402

assert torch.cuda.is_available(), "parity gate requires a CUDA H100"
DEV = torch.device("cuda")
torch.manual_seed(0)

_RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    _RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name:28s} {detail}")


# ===========================================================================
# Section 1 — Adam-family single-step parity (sanity anchors + Muon item-7)
# ===========================================================================
def section1_anchors() -> None:
    print("\n== Section 1: Adam-family single-step parity (fp32 fused vs fp64 ref) ==")

    # adamw
    p = torch.randn(64, 64, device=DEV)
    g = torch.randn_like(p)
    ea = torch.zeros_like(p)
    eas = torch.zeros_like(p)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 1e-2
    p0 = p.double().clone()
    steps = [1]
    ops.fused_adamw_simple_step([p], [g], [ea], [eas], steps, b1, b2, lr, wd, eps)
    torch.cuda.synchronize()
    z = torch.zeros_like(p0)
    exp, _, _ = ref_adamw_step(p0, g.double(), z, z, lr=lr, beta1=b1, beta2=b2,
                               eps=eps, wd=wd, t=1)
    d = (p.double() - exp).abs().max().item()
    record("adamw", d < 1e-5, f"max|dp|={d:.2e} (tol 1e-5)")

    # lion
    p = torch.randn(64, 64, device=DEV)
    g = torch.randn_like(p)
    ea = torch.zeros_like(p)
    lr, b1, b2, wd = 3e-4, 0.9, 0.99, 0.1
    p0 = p.double().clone()
    ops.lion_fused_step([p], [g], [ea], lr, b1, b2, wd)
    torch.cuda.synchronize()
    exp, _ = ref_lion_step(p0, g.double(), torch.zeros_like(p0),
                           lr=lr, beta1=b1, beta2=b2, wd=wd)
    d = (p.double() - exp).abs().max().item()
    record("lion", d < 1e-5, f"max|dp|={d:.2e} (tol 1e-5)")

    # muon (item-7 re-validation: weight-decay sign + Newton-Schulz)
    p = torch.randn(96, 128, device=DEV)
    g = torch.randn_like(p)
    buf = torch.zeros_like(p)
    mom, lr, wd, ns = 0.95, 0.02, 1.0, 5
    p0 = p.double().clone()
    ops.muon_fused_step([p], [g], [buf], mom, lr, wd, ns)
    torch.cuda.synchronize()
    exp, _ = ref_muon_step(p0, g.double(), torch.zeros_like(p0),
                           momentum=mom, lr=lr, wd=wd, ns_steps=ns)
    d = (p.double() - exp).abs().max().item()
    # NS matmuls in fp32 vs fp64 accumulate more error; 1e-4 is the oracle floor.
    record("muon (item-7 wd+NS)", d < 1e-4, f"max|dp|={d:.2e} (tol 1e-4)")


# ===========================================================================
# Section 2 — Prodigy: the adaptive d must NOT catapult, and must adapt.
# ===========================================================================
def section2_prodigy() -> None:
    print("\n== Section 2: Prodigy d-adaptation (scale-free; no 1e6 catapult) ==")

    def run_prodigy(opt_factory, n_steps=60):
        torch.manual_seed(7)
        target = torch.randn(512, device=DEV)
        x = torch.nn.Parameter(torch.zeros(512, device=DEV))  # p_init = 0
        opt = opt_factory([x])
        d_traj = []
        init_loss = None
        for _ in range(n_steps):
            opt.zero_grad()
            loss = 0.5 * ((x - target) ** 2).sum()
            loss.backward()
            opt.step()
            if init_loss is None:
                init_loss = loss.item()
            d_traj.append(_read_d(opt))
        return d_traj, loss.item(), init_loss

    d_traj, final_loss, init_loss = run_prodigy(
        lambda ps: go.Prodigy(ps, lr=1.0, weight_decay=0.0))
    d0 = d_traj[0]
    dmax_early = max(d_traj[:5])
    # Catapult signature: pre-fix d jumps 1e-6 -> ~0.2 by step 2 (ratio ~1e5).
    early_ratio = dmax_early / max(d0, 1e-30)
    no_catapult = early_ratio < 50.0
    record("prodigy no-catapult",
           no_catapult,
           f"d[0]={d0:.2e} max(d[:5])={dmax_early:.2e} ratio={early_ratio:.1f} (tol<50)")
    # d must actually adapt upward over the run (it estimates the LR).
    grew = d_traj[-1] > d_traj[0] * 1.5
    record("prodigy adapts (d grows)", grew,
           f"d[0]={d_traj[0]:.2e} -> d[-1]={d_traj[-1]:.2e}")
    # And the run must converge (loss drops >=90%) — the catapult diverged
    # training (loss stayed at its initial value or grew).
    drop = 1.0 - final_loss / max(init_loss, 1e-30)
    record("prodigy converges", drop > 0.90,
           f"loss {init_loss:.1f} -> {final_loss:.3e} ({100*drop:.1f}% drop, tol>90%)")

    # Cross-check vs external prodigyopt 1.1.2 if available (advisor: validate
    # against the EXTERNAL reference, not the repo's own derived reference).
    try:
        import prodigyopt  # noqa: F401
        ref_traj, _, _ = run_prodigy(
            lambda ps: prodigyopt.Prodigy(ps, lr=1.0, weight_decay=0.0))
        # Both should grow smoothly and stay the same order of magnitude.
        ratio = d_traj[-1] / max(ref_traj[-1], 1e-30)
        close_order = 0.05 < ratio < 20.0
        record("prodigy vs prodigyopt", close_order,
               f"d[-1] ours={d_traj[-1]:.2e} prodigyopt={ref_traj[-1]:.2e} "
               f"ratio={ratio:.2f} (tol 0.05..20)")
    except Exception as exc:  # prodigyopt not installed -> rely on the above
        print(f"  [skip] prodigyopt cross-check unavailable: {exc}")


def _read_d(opt) -> float:
    for attr in ("d_lr", "_d_lr"):
        v = getattr(opt, attr, None)
        if v is not None:
            return float(v() if callable(v) else v)
    # prodigyopt stores d in param_groups
    try:
        return float(opt.param_groups[0]["d"])
    except Exception:
        return float("nan")


# ===========================================================================
# Section 3 — SuperGrok11/15 meta-net mu parity (H=32 OOB fix).
#   mu = rescale * MLP(grad, sharpness); MLP = Linear(2,H) -> GELU -> Linear(H,1)
#   We zero W1[:,1] (sharpness column) so mu depends only on grad and is exactly
#   reproducible without knowing the internal sharpness cache.
# ===========================================================================
def _meta_net_mu_ref(grad: torch.Tensor, W1, b1, W2, b2, rescale) -> torch.Tensor:
    g = grad.double().reshape(-1)
    W1 = W1.double()          # [H,2]
    pre = g.unsqueeze(1) * W1[:, 0].unsqueeze(0) + b1.double().unsqueeze(0)  # [N,H]
    h = torch.nn.functional.gelu(pre)
    out = h @ W2.double().reshape(-1) + float(b2.double().reshape(()))       # [N]
    return rescale * out


def _run_sg_metanet(raw_step, N=4096, H=32):
    torch.manual_seed(11)
    p = torch.randn(N, device=DEV)
    g = torch.randn(N, device=DEV)
    ea = torch.zeros(N, device=DEV)
    eas = torch.zeros(N, device=DEV)
    mu = torch.zeros(N, device=DEV)
    sharp = torch.randn(N, device=DEV)  # sharpness cache (made irrelevant below)
    # Non-trivial weights so the MLP output is meaningful; zero the sharpness col.
    W1 = (torch.randn(H, 2, device=DEV) * 0.5)
    W1[:, 1] = 0.0
    b1 = torch.randn(H, device=DEV) * 0.5
    W2 = torch.randn(1, H, device=DEV) * 0.5
    b2 = torch.randn(1, device=DEV) * 0.5
    rescale = 0.5
    raw_step(p, g, ea, eas, mu, sharp, W1, b1, W2, b2, rescale, H)
    torch.cuda.synchronize()
    mu_ref = _meta_net_mu_ref(g, W1, b1, W2, b2, rescale)
    return mu.double(), mu_ref, rescale


def section3_supergrok_metanet() -> None:
    print("\n== Section 3: SuperGrok11/15 meta-net mu parity (H=32 OOB fix) ==")

    def sg11_step(p, g, ea, eas, mu, sharp, W1, b1, W2, b2, rescale, H):
        ops.supergrok11_fused_step(
            [p], [g], [ea], [eas], [mu], [sharp], [1], [0.98], [0.9],
            W1, b1, W2, b2, rescale, H,
            0.999, 1e-3, 0.0, 1e-8, 5.0, 0.0, 5.0, 1.0)

    def sg15_step(p, g, ea, eas, mu, sharp, W1, b1, W2, b2, rescale, H):
        ops.supergrok15_fused_step(
            [p], [g], [ea], [eas], [mu], [sharp], [1], [0.98], [0.9],
            W1, b1, W2, b2, rescale, H,
            0.999, 1e-3, 0.0, 1e-8, 5.0, 0.0, 0.0, 1.0)

    for name, step in (("supergrok11", sg11_step), ("supergrok15", sg15_step)):
        try:
            mu_k, mu_ref, rescale = _run_sg_metanet(step)
            d_resc = (mu_k - mu_ref).abs().max().item()
            d_noresc = (mu_k - mu_ref / rescale).abs().max().item()
            ref_scale = mu_ref.abs().max().item()
            # The kernel should match rescale*MLP. If it instead matches MLP
            # (rescale dropped), d_noresc will be the small one — surface that.
            passed = d_resc < 1e-3
            note = ""
            if not passed and d_noresc < 1e-3:
                note = " [matches MLP w/o rescale -> rescale dropped in kernel]"
            record(f"{name} meta-net mu", passed,
                   f"max|mu-rescale*MLP|={d_resc:.2e} (tol 1e-3, |mu_ref|~{ref_scale:.2f}){note}")
        except Exception as exc:
            record(f"{name} meta-net mu", False, f"raised: {type(exc).__name__}: {exc}")


# ===========================================================================
# Section 3b — SuperGrok11 FULL param-update parity (the cosine-gated mixing).
#   The pre-fix fused path collapsed a memorized solution because it applied mu
#   TWICE: mu_metanet formed smart_grad = g + alpha*mu, then adam_decay added
#   AGAIN g_eff = smart_grad + (ramp*gate*lamb)*mu — a coefficient that GREW
#   with the gate (wrong polarity), destroying the solution as rescale trained.
#   Canonical (supergrok11.h:101 / Pallas ref / ref_sg11_step) is the SINGLE
#   gated correction smart_grad = g + (1-gate)*alpha*mu with
#   gate=sigmoid(gate_temp*cos(g,mu)), which SHRINKS as alignment grows so the
#   solution HOLDS.
#
#   Section 3 only validated the mu buffer. This validates the ENTIRE param /
#   moment update against the fp64 oracle. Crucially the reference gate is
#   recomputed INDEPENDENTLY in fp64 from (grad, mu_buffer) — NOT read from the
#   kernel — so the gate FAILS if the kernel regresses to cos(smart_grad,mu),
#   reintroduces the lamb*mu second add, or uses the wrong polarity.
# ===========================================================================
def _cosine_gate_ref(grad: torch.Tensor, mu: torch.Tensor,
                     gate_temp: float = 5.0) -> float:
    """sigmoid(gate_temp * cos(grad, mu)) in fp64 — matches compute_cosine_gate_fused
       (cos = num/sqrt(den_g*den_m + 1e-12); gate = sigmoid(gate_temp*cos), the
       canonical sg11_finalize_gate). The cosine SIGNAL is preserved; only the final
       squashing is the temperature-scaled sigmoid (no longer a bare clamp)."""
    g = grad.double().reshape(-1)
    m = mu.double().reshape(-1)
    num = (g * m).sum()
    den = torch.sqrt((g * g).sum() * (m * m).sum() + 1e-12)
    cos = (num / den) if float(den) > 0.0 else torch.tensor(0.0, dtype=torch.float64)
    return float(torch.sigmoid(gate_temp * cos))


def section3b_supergrok11_full_update() -> None:
    print("\n== Section 3b: SuperGrok11 full param update parity (cosine-gated mixing) ==")
    try:
        torch.manual_seed(1107)
        N, H = 4096, 32
        # NON-trivial moments so the Adam tail is meaningfully exercised, and a
        # higher step t so bias correction is non-degenerate.
        p = torch.randn(N, device=DEV)
        g = torch.randn(N, device=DEV)
        ea = torch.randn(N, device=DEV) * 0.1
        eas = (torch.randn(N, device=DEV) * 0.1).abs()  # exp_avg_sq >= 0
        mu = torch.zeros(N, device=DEV)
        sharp = torch.randn(N, device=DEV)
        # Weights chosen so phi (hence mu) CORRELATES with grad: keep the grad
        # column positive-mean and zero the sharpness column so mu is a smooth
        # monotone-ish function of g -> the cosine gate lands in (0,1), which is
        # exactly the regime the second-mu bug corrupted. (gate==0 or 1 would
        # not exercise the gate scaling.)
        W1 = torch.empty(H, 2, device=DEV)
        W1[:, 0] = (torch.rand(H, device=DEV) * 0.4 + 0.1)   # +0.1..+0.5
        W1[:, 1] = 0.0
        b1 = torch.randn(H, device=DEV) * 0.2
        W2 = (torch.rand(1, H, device=DEV) * 0.4 + 0.1)      # +0.1..+0.5
        b2 = torch.randn(1, device=DEV) * 0.2
        rescale = 0.7

        alpha = 0.98
        beta1 = 0.9
        beta2 = 0.999
        lr = 1e-3
        eps = 1e-8
        wd = 0.01
        t = 5
        gate_temp = 5.0

        p0 = p.double().clone()
        ea0 = ea.double().clone()
        eas0 = eas.double().clone()
        g0 = g.double().clone()

        ops.supergrok11_fused_step(
            [p], [g], [ea], [eas], [mu], [sharp], [t], [alpha], [beta1],
            W1, b1, W2, b2, rescale, H,
            beta2, lr, wd, eps,
            # lamb, ramp, gate_temperature, grad_clip_norm.
            # FIX 2 (Option B): lamb/ramp are now LIVE multipliers on the single
            # canonical (1-gate)*alpha term (lamb_eff = ramp*lamb*(1-gate)*alpha).
            # Pass lamb=ramp=1.0 here so lamb_eff == (1-gate)*alpha, the exact
            # identity ref_sg11_step validates (the live-lamb scaling is checked
            # separately below). grad_clip_norm=0 disables the in-place grad clip
            # so the fp64 reference grad matches exactly.
            1.0, 1.0, gate_temp, 0.0)
        torch.cuda.synchronize()

        # Independent fp64 reference. gate = sigmoid(gate_temp*cos(grad, mu_buffer))
        # where mu_buffer is the kernel's own mu output (= rescale*MLP, validated in §3).
        gate_ref = _cosine_gate_ref(g0, mu.double(), gate_temp)
        p_exp, m_exp, v_exp = ref_sg11_step(
            p0, g0, ea0, eas0, mu.double(),
            gate=gate_ref, alpha=alpha, lr=lr, beta1=beta1, beta2=beta2,
            eps=eps, wd=wd, t=t)

        dp = (p.double() - p_exp).abs().max().item()
        dm = (ea.double() - m_exp).abs().max().item()
        dv = (eas.double() - v_exp).abs().max().item()
        # The gate MUST be exercised in a NON-degenerate sigmoid range, else this
        # test is vacuous. gate = sigmoid(gate_temp*cos) with a positive cosine
        # (the W1 grad column is positive-mean, sharpness zeroed) ⇒ cos in (0,1) ⇒
        # gate in (0.5, ~0.993): assert it cleared the 0.5 floor (cos>0 truly fed in)
        # and is not saturated at 1.
        gate_ok = 0.5 < gate_ref < 0.999
        passed = dp < 1e-4 and dm < 1e-4 and dv < 1e-4 and gate_ok
        record("supergrok11 full update", passed,
               f"gate={gate_ref:.3f} (want 0.5..0.999, sigmoid(gate_temp*cos)) "
               f"max|dp|={dp:.2e} |dm|={dm:.2e} |dv|={dv:.2e} (tol 1e-4)")

        # FIX 2 teeth: lamb is now a LIVE multiplier. Re-run identically but with
        # lamb=L≠1; the correction (hence the first moment from a zero init)
        # must scale by L*ramp. ea2 = (1-beta1)*(g + L*(1-gate)*alpha*mu) at the
        # zero moment init, compared to the fp64 prediction. If lamb were still
        # (void)-ed/ignored, ea2 would equal the lamb=1 moment ⇒ FAIL.
        L = 2.0
        p2 = p0.clone().float()
        ea2 = torch.zeros(N, device=DEV)
        eas2 = torch.zeros(N, device=DEV)
        mu2 = torch.zeros(N, device=DEV)
        ops.supergrok11_fused_step(
            [torch.nn.Parameter(p2)], [g], [ea2], [eas2], [mu2], [sharp],
            [t], [alpha], [beta1], W1, b1, W2, b2, rescale, H,
            beta2, lr, wd, eps, L, 1.0, gate_temp, 0.0)
        torch.cuda.synchronize()
        gate2 = _cosine_gate_ref(g0, mu2.double(), gate_temp)  # mu2 == mu (same inputs)
        smart2 = g0 + L * (1.0 - gate2) * alpha * mu2.double()
        m2_exp = (1.0 - beta1) * smart2  # zero-init first moment
        dm2 = (ea2.double() - m2_exp).abs().max().item()
        # Non-vacuous: the lamb=1 moment must DIFFER from the lamb=L moment.
        m1 = (1.0 - beta1) * (g0 + 1.0 * (1.0 - gate2) * alpha * mu2.double())
        live_diff = (m2_exp - m1).abs().max().item()
        lamb_live_ok = dm2 < 1e-4 and live_diff > 1e-4
        record("supergrok11 lamb is live", lamb_live_ok,
               f"L={L}: max|dm2|={dm2:.2e} (tol 1e-4), "
               f"lamb=1-vs-L moment gap={live_diff:.2e} (want >1e-4)")
    except Exception as exc:
        record("supergrok11 full update", False,
               f"raised: {type(exc).__name__}: {exc}")


# ===========================================================================
# Section 4 — SuperGrok2 full param-update parity (restored grokfast term).
#   The kernel apply sg2_apply_step (csrc/algorithms/supergrok2.h) was missing
#   the grokfast amplification: a prior refactor (void)-ed lamb_eff in the sm_90
#   launchers, so smart_grad lost its `lamb_eff*slow_new` term. The fix restores
#   it as the single source: smart_grad = g + alpha*mu_new + lamb_eff*slow_new,
#   with mu_new = gru_decay*mu + (1-gru_decay)*expert_out the expert-output EMA
#   and slow_new = alpha*slow + (1-alpha)*g the slow-gradient EMA.
#
#   Like §3b for SG11, this validates the WHOLE param/moment update against the
#   fp64 oracle ref_sg2_apply_step. Crucially it does NOT reconstruct the heavy
#   PEER/CSA/HCA meta-net: on a FRESH optimizer (mu=slow=0) one step leaves
#   mu_buf = (1-gru_decay)*expert_out = mu_new and slow_buf = (1-alpha)*g =
#   slow_new in the persisted buffers, so the kernel's OWN post-step mu/slow
#   buffers recover the exact smart_grad it used — and ref_sg2_apply_step with a
#   NONZERO lamb_eff reproduces the param iff the kernel actually consumed
#   lamb_eff*slow_new. If the kernel regresses to the (void)-ed drop, the
#   reference (which adds lamb_eff*slow_buf) diverges and this FAILS.
# ===========================================================================
def section4_supergrok2() -> None:
    print("\n== Section 4: SuperGrok2 full param update parity (grokfast term) ==")
    # 4a: still assert the per-head PEER routing path RUNS (no reshape crash).
    try:
        torch.manual_seed(3)
        p = torch.nn.Parameter(torch.randn(256, device=DEV))
        opt = go.SuperGrok2([p], lr=1e-3)
        p.grad = torch.randn_like(p)
        opt.step()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(p.detach()).all())
        record("supergrok2 runs", finite,
               f"step completed, params finite={finite}")
    except Exception as exc:
        record("supergrok2 runs", False, f"raised: {type(exc).__name__}: {exc}")

    # 4b: numeric parity of the kernel apply against the fp64 reference, with a
    # NONZERO lamb_eff so the restored slow term is exercised.
    try:
        from tests.hw.test_reference_parity import ref_sg2_apply_step
        torch.manual_seed(4)
        N = 512
        p = torch.nn.Parameter(torch.randn(N, device=DEV))
        # warmup_steps=0, warmup_ramp=1 ⇒ ramp=1.0 after the first step; a high
        # train_acc drives the accuracy gate ≈1 ⇒ lamb_eff = lamb*ramp*gate > 0.
        # gradient_clipping huge ⇒ the kernel's in-step global-norm clip is a
        # no-op, so the fp64 reference can use the RAW grad (||g||≈sqrt(N)≫1
        # would otherwise be scaled inside the kernel and break the comparison).
        opt = go.SuperGrok2([p], lr=1e-3, warmup_steps=0, warmup_ramp=1,
                            lamb=2.0, gradient_clipping=1e9)
        p0 = p.detach().double().clone()
        g = torch.randn_like(p)
        g0 = g.double().clone()
        p.grad = g.clone()
        # train_acc=1.0 ⇒ gate_signal = sigmoid(gate_scale*(1.0-gate_thresh)).
        opt.step(train_acc=1.0)
        torch.cuda.synchronize()

        # Recover the EXACT scalars the kernel used for this (single) param.
        ramp = opt._get_ramp_factor()
        gate_signal = opt._get_gate_signal()
        lamb_eff = float(opt.lamb) * float(ramp) * float(gate_signal)
        base_alpha = opt._cached_alpha
        alpha = max(0.0, min(1.0, base_alpha * float(opt._flat_layer_alphas[0])))
        gru_decay = float(opt._flat_layer_beta1s[0])  # kernel gru_decay = beta1
        beta2 = opt.param_groups[0]["betas"][1]
        lr = opt.param_groups[0]["lr"]
        eps = opt.param_groups[0]["eps"]
        wd_eff = opt._get_effective_wd(opt.param_groups[0]["weight_decay"])
        t = int(opt._flat_steps[0])  # = 1

        # Post-step persisted EMAs ARE mu_new / slow_new (mu0=slow0=0). Feeding
        # them as (mu_state=0, expert_out, slow_state=0) to the reference would
        # need expert_out; instead recover the contribution directly: with
        # mu_state=mu_buf/gru-form already applied, run the reference with
        # mu_state=0 and expert_out chosen so (1-gru_decay)*expert_out = mu_buf,
        # and slow_state=0 so (1-alpha)*g = slow_buf must hold (it does by
        # construction). Simplest: pass expert_out = mu_buf/(1-gru_decay).
        mu_buf = opt._flat_mus[0].double()
        slow_buf = opt._flat_slows[0].double()
        ea_buf = opt._flat_exp_avgs[0].double()       # = (1-beta1)*smart_grad
        easq_buf = opt._flat_exp_avg_sqs[0].double()  # = (1-beta2)*smart_grad^2
        # Recover the PEER expert_out contribution from the post-step expert-EMA
        # buffer (mu0=0 ⇒ mu_buf = (1-gru_decay)*expert_out). Feeding this back
        # reproduces mu_new exactly; the discriminating check is on the MOMENTS
        # (and slow), which carry the lamb_eff*slow_new term linearly.
        expert_out = mu_buf / max(1e-9, (1.0 - gru_decay))

        p_exp, m_exp, v_exp, mu_new_exp, slow_new_exp = ref_sg2_apply_step(
            p0, g0,
            torch.zeros(N, dtype=torch.float64, device=DEV),   # exp_avg0 = 0
            torch.zeros(N, dtype=torch.float64, device=DEV),   # exp_avg_sq0 = 0
            torch.zeros(N, dtype=torch.float64, device=DEV),   # mu_state0 = 0
            torch.zeros(N, dtype=torch.float64, device=DEV),   # slow_state0 = 0
            expert_out=expert_out, alpha=alpha, gru_decay=gru_decay,
            lamb_eff=lamb_eff, lr=lr, beta1=gru_decay, beta2=beta2, eps=eps,
            wd=wd_eff, t=t)

        # The Adam param update at t=1 saturates to ≈sign(smart_grad), so |dp| is
        # a WEAK probe of the lamb term. The first/second moments are
        # linear/quadratic in smart_grad from the zero init, so |dm|/|dv| carry
        # the FULL lamb_eff*slow_new contribution — these are the teeth: if the
        # kernel regressed to the (void)-ed drop (smart_grad = g + alpha*mu_new,
        # no slow), |dm| diverges by O(lamb_eff*|slow|) ≫ 1e-4.
        dp = (p.double() - p_exp).abs().max().item()
        dm = (ea_buf - m_exp).abs().max().item()
        dv = (easq_buf - v_exp).abs().max().item()
        dslow = (slow_buf - slow_new_exp).abs().max().item()
        # lamb_eff MUST be exercised (nonzero), else the slow term is vacuous.
        lamb_ok = lamb_eff > 1e-3
        passed = (dm < 1e-4 and dv < 1e-4 and dslow < 1e-4
                  and dp < 1e-4 and lamb_ok)
        record("supergrok2 full update", passed,
               f"lamb_eff={lamb_eff:.3f} (want >1e-3) "
               f"max|dm|={dm:.2e} |dv|={dv:.2e} |dslow|={dslow:.2e} "
               f"|dp|={dp:.2e} (tol 1e-4)")
    except Exception as exc:
        record("supergrok2 full update", False,
               f"raised: {type(exc).__name__}: {exc}")


# ===========================================================================
# Section 5 — GrokAdamW PUBLISHED full update: per-tensor LAYER-WISE β1 +
#   amplified slow-grad (nonzero lamb), validated against the fp64 oracle.
#   Mirrors §3b (SG11): two tensors with DISTINCT β1_i (β1_0 != β1_1, the
#   layer-wise decay) must each match ref_grokadamw_step under their OWN β1,
#   AND the resulting moments must DIFFER between the two — proving the kernel
#   consumed the per-tensor β1 vector (not a single global β1). lamb>0 keeps the
#   EMA amplification live; nonzero ema/moments + t>1 exercise the full tail.
# ===========================================================================
def section_grokadamw() -> None:
    print("\n== Section 5: GrokAdamW layer-wise β1 + amplification parity ==")
    try:
        torch.manual_seed(515)
        N = 4096
        alpha, lamb = 0.9, 2.0          # lamb>0 ⇒ slow-grad amplification live
        beta1, beta2 = 0.9, 0.999
        gamma = 0.1
        lr, eps, wd = 1e-3, 1e-8, 0.01
        t = 5
        # Layer-wise β1: β1_0 = β1 (i=0), β1_1 = β1*(1-gamma) (i=1) — the
        # published β1_i = β1*(1-gamma)**i. DISTINCT so the per-tensor vector
        # path is meaningfully exercised.
        beta1s = [beta1, beta1 * (1.0 - gamma)]

        # IDENTICAL grad / ema / moment seeds for both tensors, so any divergence
        # in the OUTPUT moments is attributable SOLELY to the differing β1.
        g = torch.randn(N, device=DEV)
        ema_seed = torch.randn(N, device=DEV) * 0.1
        ea_seed = torch.randn(N, device=DEV) * 0.1
        eas_seed = (torch.randn(N, device=DEV) * 0.1).abs()  # exp_avg_sq >= 0

        params = [torch.randn(N, device=DEV), torch.randn(N, device=DEV)]
        grads = [g.clone(), g.clone()]
        exp_avgs = [ea_seed.clone(), ea_seed.clone()]
        exp_avg_sqs = [eas_seed.clone(), eas_seed.clone()]
        ema_bufs = [ema_seed.clone(), ema_seed.clone()]
        steps = [t, t]
        # fp64 snapshots of the inputs BEFORE the in-place kernel mutates them.
        snaps = [(params[i].double().clone(), exp_avgs[i].double().clone(),
                  exp_avg_sqs[i].double().clone(), ema_bufs[i].double().clone())
                 for i in range(2)]

        ops.grokadamw_fused_step(
            params, grads, exp_avgs, exp_avg_sqs, ema_bufs, steps,
            beta1s,                      # PUBLISHED: per-tensor layer-β1 vector
            alpha, lamb, beta1, beta2, lr, wd, eps, 0.0)  # grad_clip=0 ⇒ off
        torch.cuda.synchronize()

        ok = True
        details = []
        first_moments = []
        for i in range(2):
            p0, ea0, eas0, ema0 = snaps[i]
            p_exp, m_exp, v_exp, ema_exp = ref_grokadamw_step(
                p0, g.double(), ea0, eas0, ema0,
                alpha=alpha, lamb=lamb, lr=lr, beta1=beta1s[i], beta2=beta2,
                eps=eps, wd=wd, t=t)
            dp = (params[i].double() - p_exp).abs().max().item()
            dm = (exp_avgs[i].double() - m_exp).abs().max().item()
            dv = (exp_avg_sqs[i].double() - v_exp).abs().max().item()
            de = (ema_bufs[i].double() - ema_exp).abs().max().item()
            ok = ok and dp < 1e-4 and dm < 1e-4 and dv < 1e-4 and de < 1e-4
            details.append(f"L{i}(β1={beta1s[i]:.3f}) "
                           f"|dp|={dp:.1e}|dm|={dm:.1e}|dv|={dv:.1e}|de|={de:.1e}")
            first_moments.append(exp_avgs[i].double())
        # Teeth: identical inputs but different β1 ⇒ the two first moments MUST
        # differ, proving the kernel consumed the per-tensor β1 (not one global).
        moment_gap = (first_moments[0] - first_moments[1]).abs().max().item()
        teeth_ok = moment_gap > 1e-4
        record("grokadamw layer-wise β1", ok and teeth_ok,
               "; ".join(details) + f"; L0-vs-L1 moment gap={moment_gap:.1e} "
               f"(want >1e-4) (tol 1e-4)")
    except Exception as exc:
        record("grokadamw layer-wise β1", False,
               f"raised: {type(exc).__name__}: {exc}")


def main() -> int:
    print(f"H100 PARITY GATE — device={torch.cuda.get_device_name(0)}")
    section1_anchors()
    section2_prodigy()
    section3_supergrok_metanet()
    section3b_supergrok11_full_update()
    section4_supergrok2()
    section_grokadamw()
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print(f"\n=== PARITY GATE: {n_pass} pass, {n_fail} fail ===")
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"    FAIL {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
