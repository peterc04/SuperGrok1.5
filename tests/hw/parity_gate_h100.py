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
    ref_adamw_step, ref_lion_step, ref_muon_step, ref_sg11_step,
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
#   gated correction smart_grad = g + (1-gate)*alpha*mu with gate=cos(g,mu),
#   which SHRINKS as alignment grows so the solution HOLDS.
#
#   Section 3 only validated the mu buffer. This validates the ENTIRE param /
#   moment update against the fp64 oracle. Crucially the reference gate is
#   recomputed INDEPENDENTLY in fp64 from (grad, mu_buffer) — NOT read from the
#   kernel — so the gate FAILS if the kernel regresses to cos(smart_grad,mu),
#   reintroduces the lamb*mu second add, or uses the wrong polarity.
# ===========================================================================
def _cosine_gate_ref(grad: torch.Tensor, mu: torch.Tensor) -> float:
    """clamp(cos(grad, mu), 0, 1) in fp64 — matches compute_cosine_gate_fused
       (num/sqrt(den_g*den_m + 1e-12))."""
    g = grad.double().reshape(-1)
    m = mu.double().reshape(-1)
    num = (g * m).sum()
    den = torch.sqrt((g * g).sum() * (m * m).sum() + 1e-12)
    gate = (num / den) if float(den) > 0.0 else torch.tensor(0.0, dtype=torch.float64)
    return float(torch.clamp(gate, 0.0, 1.0))


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
            # lamb/ramp are now unused in the mixing; grad_clip_norm=0 disables
            # the in-place grad clip so the fp64 reference grad matches exactly.
            5.0, 1.0, gate_temp, 0.0)
        torch.cuda.synchronize()

        # Independent fp64 reference. gate = clamp(cos(grad, mu_buffer),0,1) where
        # mu_buffer is the kernel's own mu output (= rescale*MLP, validated in §3).
        gate_ref = _cosine_gate_ref(g0, mu.double())
        p_exp, m_exp, v_exp = ref_sg11_step(
            p0, g0, ea0, eas0, mu.double(),
            gate=gate_ref, alpha=alpha, lr=lr, beta1=beta1, beta2=beta2,
            eps=eps, wd=wd, t=t)

        dp = (p.double() - p_exp).abs().max().item()
        dm = (ea.double() - m_exp).abs().max().item()
        dv = (eas.double() - v_exp).abs().max().item()
        # The gate MUST be exercised in (0,1), else this test is vacuous.
        gate_ok = 0.01 < gate_ref < 0.99
        passed = dp < 1e-4 and dm < 1e-4 and dv < 1e-4 and gate_ok
        record("supergrok11 full update", passed,
               f"gate={gate_ref:.3f} (want 0.01..0.99) "
               f"max|dp|={dp:.2e} |dm|={dm:.2e} |dv|={dv:.2e} (tol 1e-4)")
    except Exception as exc:
        record("supergrok11 full update", False,
               f"raised: {type(exc).__name__}: {exc}")


# ===========================================================================
# Section 4 — SuperGrok2 per-head PEER routing must RUN (no reshape crash).
# ===========================================================================
def section4_supergrok2() -> None:
    print("\n== Section 4: SuperGrok2 per-head PEER routing runs (no crash) ==")
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


def main() -> int:
    print(f"H100 PARITY GATE — device={torch.cuda.get_device_name(0)}")
    section1_anchors()
    section2_prodigy()
    section3_supergrok_metanet()
    section3b_supergrok11_full_update()
    section4_supergrok2()
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print(f"\n=== PARITY GATE: {n_pass} pass, {n_fail} fail ===")
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"    FAIL {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
