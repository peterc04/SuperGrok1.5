"""tests/hw/test_multistep_parity.py — the LOAD-BEARING multi-step parity gates
for the converted grokadamw + prodigy L3-TC cells (audit finding #19a).

WHY THIS FILE EXISTS. tests/hw/test_l3tc_tail_gate.py's single-step gate is
NECESSARY but NOT SUFFICIENT for two cells, and its own comments (~:303-339) name
these multi-step checks as the load-bearing complement — but they were NEVER
implemented as a runnable test. This file implements them:

  * grokadamw: the single-step gate is BLIND to the GLOBAL grad-norm clip — at
    step 1 ‖g‖₂≈0.72 < grad_clip=1.0 so the kernel's P2.5 clip is INERT and a
    dropped clip would still pass. The clip FIRES later (the documented onset is
    ~step 50 for the decoder), so only a multi-step run that crosses the clip
    threshold exercises it. test_grokadamw_multistep_parity drives ~60 fused
    steps and an eager GrokAdamW reference SIDE-BY-SIDE on the kernel's own
    TC-reduced grad, asserts params match the gate's tol at {1, 50, 60}, asserts
    the clip condition ‖g‖>max_norm ACTUALLY occurred on the eager side by step
    50 (so the test cannot vacuously pass on a coincidentally-small norm), and
    re-runs the whole 60-step fused trajectory twice more bit-identical (A/A/A).

  * prodigy: the single-step gate is BLIND to the d-adaptation — at step 1 the
    trajectory anchor param_init == params so r = Σ d²·<g, p0−p> = 0 ⇒ d stays
    frozen at d0=1e-6 (matching eager _d_lr=d0). The DEFINING prodigy mechanism
    (the adaptive d grown from the cumulative parameter trajectory) only fires at
    step≥2. At the PRODUCTION d0=1e-6/d_coef=1.0 d is INERT for the decoder (the
    slow real <g,p0−p> never trips the candidate over d0; natural trajectory
    forcing is unstable — small d0 won't bootstrap, large d0·lr → NaN, per
    tuning/_prodigy_owner_block_unit.py). So test_prodigy_multistep_parity uses
    the STABLE controlled-anchor technique that owner-block unit validated:
    inject a known param_init = p_e + delta into BOTH the kernel state slot and
    the eager state IDENTICALLY at step 2 (so p0−p = delta is nonzero + EXACT on
    both sides) with d_coef>1 (makes the "persist UNSCALED, scale only the
    candidate" line load-bearing) — d then GROWS off d0 deterministically. It
    asserts the kernel's persisted d (read from state[4*total+3]; layout in
    grokking_optimizers/dispatch.py ~:1828-1836) tracks the eager opt._d_lr each
    step, that d actually grew off d0, that params match tol at the final step,
    and that a d-FROZEN control (d_coef=0 ⇒ candidate=0 ⇒ d=max(d_prev,0)=d0
    forever) DIVERGES from the adaptive run by step 50 — proving the d-adaptation
    is load-bearing (the gate has teeth).

REUSE. The model/data/init builder (_build_cell) and the grokadamw factory
(_grokadamw_factory) are IMPORTED from tests.hw.test_l3tc_tail_gate so this gate
drives the EXACT same model, init, hyperparameters, and determinism pinning as
the single-step gate (no copy-paste that can drift). The fused-step driver is
grokking_optimizers.dispatch.fused_train_step (the SAME production route the race
and the single-step gate take). The only locally-mirrored helper is _mk_prodigy
(a thin Prodigy builder that mirrors test_l3tc_tail_gate._prodigy_factory ~:108-116
but ALSO accepts d0/d_coef overrides — _prodigy_factory hard-codes the production
defaults and cannot force the d-adaptation; the override is the whole point here).
The _gpu_ready skipif predicate is imported from the tail gate and applied with
the SAME @pytest.mark.hw + skipif pattern, so these skip cleanly off-GPU and run
from HARDWARE_VALIDATION.md on real accelerators.

Run:
  PYTHONPATH=. python -m pytest tests/hw/test_multistep_parity.py -q
  CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. python -m pytest \
        tests/hw/test_multistep_parity.py -m hw -q
"""
from __future__ import annotations

import pytest
import torch

# REUSE the single-step gate's builders/helpers BY IMPORT (do NOT copy-paste logic
# that can drift): the same model+data+init builder, the same grokadamw factory
# (production hyperparameters), and the same GPU-ready skipif predicate. These all
# import lazily inside the functions, so importing them off-GPU is safe (no CUDA
# touched at import time).
from tests.hw.test_l3tc_tail_gate import (
    _build_cell,
    _grokadamw_factory,
    _gpu_ready,
)


# ── Tolerances (mirror tuning/_grokadamw_multistep_parity.py + _prodigy_multistep_
# parity.py — the same single-source numbers the reference probes calibrated). TOL
# is the kernel-vs-eager fp32-reorder budget; it MUST be tighter than the control
# divergence or the gate cannot fail a dropped mechanism. The single-step gate uses
# 1e-4 for params/state, so we use the same so "matches the gate's tol" is literal.
TOL = 1e-4
CTRL_MIN_DIVERGENCE = 1e-3   # a load-bearing mechanism's control must diverge ≥ this

# grokadamw multi-step run length. ~60 steps takes the trajectory PAST the
# documented ~step-50 clip onset for the decoder (the single-step-gate-blind
# regime). Parity is asserted at {1, 50, 60}; the clip must have fired by 50.
GA_N_STEPS = 60
GA_CLIP_DEADLINE = 50        # the clip MUST have fired by this step (‖g‖>max_norm)
GA_PARITY_STEPS = (1, 50, 60)

# prodigy multi-step run length + the controlled-anchor forcing that makes d GROW.
# d_coef>1 makes the "persist r_ema UNSCALED, scale only the candidate" line
# load-bearing (at d_coef=1 it is invisible — a wrongly-scaled-persist kernel would
# pass); the injected anchor (delta) makes p0−p nonzero+EXACT so the candidate
# clears d_prev and d moves off d0 deterministically (the STABLE technique from
# tuning/_prodigy_owner_block_unit.py — natural trajectory forcing is NaN-unstable).
PR_N_STEPS = 50
PR_D_COEF = 3.0              # owner-block unit's validated forcing coefficient
PR_D0 = 1e-6                 # production d0; the controlled anchor (not a raised d0) drives d
PR_DELTA_SCALE = 1e-2        # owner-block unit's validated anchor magnitude
PR_ANCHOR_INJECT_STEP = 2    # inject p0 = p_e + delta at step-2 entry (d fires at step≥2)
PR_CTRL_DEADLINE = 50        # the d-frozen control MUST have diverged by this step


def _flat(named):
    """Flat concat of params in named_parameters() order (the L3-REAL layout)."""
    return torch.cat([p.data.reshape(-1) for _, p in named])


def _inject_grad(named, flat_grad):
    """Scatter a flat reduced grad into each param's .grad (layout order) so the
    eager reference consumes the BYTE-IDENTICAL kernel grad — isolating the
    optimizer TAIL across steps (the SAME injection the single-step gate uses)."""
    off = 0
    for _, p in named:
        n = p.numel()
        p.grad = flat_grad[off:off + n].reshape(p.shape).clone()
        off += n


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _gather_adamw_state(opt, named, total, dev):
    """Gather eager GrokAdamW state into flat [m|v|ema] (layout order) for the (1b)
    state parity (mirrors tuning/_grokadamw_multistep_parity.py::_gather_state)."""
    m = torch.zeros(total, device=dev)
    v = torch.zeros(total, device=dev)
    ema = torch.zeros(total, device=dev)
    off = 0
    for _, p in named:
        n = p.numel()
        st = opt.state.get(p, {})
        if "exp_avg" in st:
            m[off:off + n].copy_(st["exp_avg"].reshape(-1))
        if "exp_avg_sq" in st:
            v[off:off + n].copy_(st["exp_avg_sq"].reshape(-1))
        if "ema" in st:
            ema[off:off + n].copy_(st["ema"].reshape(-1))
        off += n
    return m, v, ema


def _gather_prodigy_state(opt, named, total, dev):
    """Gather eager Prodigy state into flat [m|v|s] (layout order); s is the
    trajectory accumulator the kernel persists in its 3rd slice (mirrors
    tuning/_prodigy_multistep_parity.py::_gather_state)."""
    m = torch.zeros(total, device=dev)
    v = torch.zeros(total, device=dev)
    s = torch.zeros(total, device=dev)
    off = 0
    for _, p in named:
        n = p.numel()
        st = opt.state.get(p, {})
        if "exp_avg" in st:
            m[off:off + n].copy_(st["exp_avg"].reshape(-1))
        if "exp_avg_sq" in st:
            v[off:off + n].copy_(st["exp_avg_sq"].reshape(-1))
        if "s" in st:
            s[off:off + n].copy_(st["s"].reshape(-1))
        off += n
    return m, v, s


def _mk_prodigy(g, params, c, d0, d_coef):
    """Build the REAL eager Prodigy carrying the production hyperparameters, with
    d0/d_coef OVERRIDABLE so the d-adaptation can be forced to fire.

    MIRRORS tests.hw.test_l3tc_tail_gate._prodigy_factory (~:108-116) — the SAME
    lr=prodigy_lr (1.0), weight_decay, betas=(beta1,beta2) — but _prodigy_factory
    hard-codes the production d0=1e-6/d_coef=1.0 (at which d is INERT for the
    decoder), so it cannot exercise the d-update branch. The override is the whole
    reason this local builder exists; the d_coef=0 case is the d-frozen control."""
    return g.Prodigy(params, lr=c.get("prodigy_lr", 1.0),
                     weight_decay=c["weight_decay"],
                     betas=(c["beta1"], c["beta2"]),
                     d0=d0, d_coef=d_coef)


# ───────────────────────────── grokadamw ─────────────────────────────
@pytest.mark.hw
@pytest.mark.skipif(not _gpu_ready(), reason="needs built extension on GPU")
def test_grokadamw_multistep_parity():
    """~60 steps of the L3-TC fused decoder×grokadamw cell vs an eager GrokAdamW
    reference on identical init/data, both fed the kernel's own TC-reduced grad.

    Asserts: (a) params match the gate's tol (1e-4) at steps {1, 50, 60}, with the
    GLOBAL grad-norm clip PROVABLY fired by step 50 (‖g‖>max_norm on the eager
    side — not a vacuous pass); (b) A/A/A — the full 60-step fused trajectory is
    bit-identical across three runs from the same init.

    State parity ([m|v|ema] vs eager) is also asserted the whole way; a dropped
    clip or a dropped/garbage state blows past TOL once the norm crosses max_norm.
    """
    from grokking_optimizers.dispatch import (has_l3_real, gemm_impl_for_cell,
                                              canonicalize_model, fused_train_step)
    model, opt = "decoder", "grokadamw"
    canon = canonicalize_model(model)
    assert has_l3_real(canon, opt), f"{model}×{opt} is not L3-REAL (register it first)"
    assert gemm_impl_for_cell(canon, opt, "bf16") == "wgmma", "bf16 engine != wgmma"

    # ── (a) parity: drive the kernel (m_k) and an eager reference (m_ref) from the
    # SAME seeded init; the eager consumes the kernel's reduced grad each step.
    g, c, m_k, data, dev = _build_cell(model)
    tx, ty = data[0], data[1]
    named_k = [(n, p) for n, p in m_k.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named_k)
    max_norm = float(c.get("grokadamw_grad_clip", 1.0))

    g2, c2, m_ref, data2, dev2 = _build_cell(model)
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    assert torch.equal(_flat(named_k), _flat(named_ref)), "init mismatch (seed)"
    opt_ref = _grokadamw_factory(g2, m_ref.parameters(), c2)

    cache = {}
    clip_fired_step = None
    parity_at = {}            # step -> (param_rel, m_rel, v_rel, ema_rel)
    worst_param = worst_m = worst_v = worst_ema = 0.0

    for step in range(1, GA_N_STEPS + 1):
        # (1) ONE real TC megakernel step (real fwd+bwd+grokadamw tail), wgmma. The
        #     kernel reads the live scalars via _opt_scalars_from(opt_ref).
        loss, grad = fused_train_step(canon, opt, m_k, opt_ref, tx, ty,
                                      state_cache=cache, step=step,
                                      return_grad=True, gemm_impl="wgmma")
        # NaN-FAIL-LOUD (never let a NaN slide into a masked max()-reduction that
        # false-greens): a NaN loss/grad is a hard failure at the offending step.
        assert loss == loss and not torch.isnan(grad).any(), (
            f"grokadamw multi-step DIVERGED at step {step}: loss={loss} "
            f"grad_has_nan={bool(torch.isnan(grad).any())}")

        # The clip-fire DISCRIMINANT: the kernel's P2.5 GLOBAL grad-norm clip bites
        # exactly when the global L2 of the reduced grad exceeds max_norm. Record the
        # first step it does (this is what the single-step gate at step 1 is blind to).
        gnorm = grad.norm().item()      # global L2 over all tensors (the clip's norm)
        if clip_fired_step is None and gnorm > max_norm:
            clip_fired_step = step

        # (2) eager reference fed the SAME reduced grad → opt_ref.step() applies the
        #     SAME canonical grokadamw update (layer-wise β1 + the global clip).
        _inject_grad(named_ref, grad)
        opt_ref.step()

        # (3) compare kernel params + [m|v|ema] state vs the eager reference.
        p_k = _flat(named_k)
        p_ref = _flat(named_ref)
        param_rel = _rel(p_k, p_ref)
        kstate = cache[canon]["state"]
        k_m = kstate[0:total]
        k_v = kstate[total:2 * total]
        k_ema = kstate[2 * total:3 * total]
        r_m, r_v, r_ema = _gather_adamw_state(opt_ref, named_ref, total, dev)
        m_rel = _rel(k_m, r_m)
        v_rel = _rel(k_v, r_v)
        ema_rel = _rel(k_ema, r_ema)
        worst_param = max(worst_param, param_rel)
        worst_m = max(worst_m, m_rel)
        worst_v = max(worst_v, v_rel)
        worst_ema = max(worst_ema, ema_rel)
        if step in GA_PARITY_STEPS:
            parity_at[step] = (param_rel, m_rel, v_rel, ema_rel)

    # ── ASSERT (a): the clip ACTUALLY fired (no vacuous pass), by the deadline.
    assert clip_fired_step is not None, (
        f"grokadamw multi-step: the GLOBAL grad-norm clip NEVER fired in "
        f"{GA_N_STEPS} steps (‖g‖ never exceeded max_norm={max_norm}); the gate "
        f"would be vacuous — the kernel's P2.5 clip is the mechanism the single-step "
        f"gate is blind to, and this run must cross the threshold to exercise it.")
    assert clip_fired_step <= GA_CLIP_DEADLINE, (
        f"grokadamw multi-step: the clip fired at step {clip_fired_step} > "
        f"{GA_CLIP_DEADLINE} (documented onset ~50); the {GA_N_STEPS}-step window "
        f"may be mis-sized or the grad scale changed.")

    # ── ASSERT (a): params (and state) match the gate's tol at the checkpoints. The
    # {50, 60} checks are PAST the clip onset, so a dropped clip fails HERE.
    for s in GA_PARITY_STEPS:
        pr, mr, vr, er = parity_at[s]
        assert pr < TOL, (f"grokadamw param parity FAIL at step {s}: rel={pr:.3e} "
                          f">= tol {TOL:.0e} (clip fired at step {clip_fired_step}; "
                          f"a dropped global grad-norm clip diverges past the onset)")
        assert mr < TOL and vr < TOL and er < TOL, (
            f"grokadamw STATE parity FAIL at step {s}: m={mr:.3e} v={vr:.3e} "
            f"ema={er:.3e} (tol {TOL:.0e}) — kernel state != eager GrokAdamW")
    # whole-trajectory worst (defence in depth; the checkpoints already cover it).
    assert worst_param < TOL and worst_m < TOL and worst_v < TOL and worst_ema < TOL, (
        f"grokadamw multi-step worst-over-{GA_N_STEPS}: param={worst_param:.3e} "
        f"m={worst_m:.3e} v={worst_v:.3e} ema={worst_ema:.3e} (tol {TOL:.0e})")

    # ── (b) A/A/A: re-run the WHOLE 60-step fused trajectory twice more from the
    # SAME seeded init; the post-step-N flat params (+ the [m|v|ema] state) must be
    # BIT-identical across runs. A race introduced anywhere in the 60 fused steps
    # (including the clip's P2.5 reduction once it fires) trips this.
    def _run_trajectory():
        g_a, c_a, m_a, data_a, dev_a = _build_cell(model)
        tx_a, ty_a = data_a[0], data_a[1]
        opt_a = _grokadamw_factory(g_a, m_a.parameters(), c_a)
        cc = {}
        for st in range(1, GA_N_STEPS + 1):
            fused_train_step(canon, opt, m_a, opt_a, tx_a, ty_a, state_cache=cc,
                             step=st, return_grad=True, gemm_impl="wgmma")
        P = torch.empty(total, device=dev_a)
        o = 0
        for _, p in m_a.named_parameters():
            if p.requires_grad:
                k = p.numel()
                P[o:o + k].copy_(p.data.reshape(-1))
                o += k
        S = cc[canon]["state"][:3 * total].clone()   # [m|v|ema] at the end
        return P, S
    P1, S1 = _run_trajectory()
    P2, S2 = _run_trajectory()
    P3, S3 = _run_trajectory()
    assert torch.equal(P1, P2) and torch.equal(P2, P3), (
        f"grokadamw A/A/A FAILED: post-{GA_N_STEPS}-step params not bit-identical "
        f"across 3 runs (max|Δ| 1v2={ (P1-P2).abs().max().item():.3e}, "
        f"2v3={(P2-P3).abs().max().item():.3e}) — a nondeterministic race in the "
        f"60-step fused trajectory (clip reduction / grad reduce).")
    assert torch.equal(S1, S2) and torch.equal(S2, S3), (
        "grokadamw A/A/A FAILED: post-trajectory [m|v|ema] state not bit-identical "
        "across 3 runs.")


# ───────────────────────────── prodigy ─────────────────────────────
@pytest.mark.hw
@pytest.mark.skipif(not _gpu_ready(), reason="needs built extension on GPU")
def test_prodigy_multistep_parity():
    """~50 steps of the L3-TC fused decoder×prodigy cell vs an eager Prodigy
    reference on identical init/data, both fed the kernel's own TC-reduced grad,
    with the d-adaptation FORCED to fire via a controlled anchor (the stable
    technique from tuning/_prodigy_owner_block_unit.py — at production
    d0=1e-6/d_coef=1.0 d is inert for the decoder, and natural trajectory forcing
    is NaN-unstable).

    Asserts: (a) the kernel's PERSISTED d-estimate (state[4*total+3]) tracks the
    eager opt._d_lr each step to TOL; (b) d actually GREW off d0=1e-6 (the
    adaptation fired — not frozen); (c) params match the gate's tol at the final
    step; (d) a d-FROZEN control (d_coef=0 ⇒ d pinned at d0) DIVERGES from the
    adaptive run by step 50 (proving the d-adaptation is load-bearing).
    """
    from grokking_optimizers.dispatch import (has_l3_real, gemm_impl_for_cell,
                                              canonicalize_model, fused_train_step)
    model, opt = "decoder", "prodigy"
    canon = canonicalize_model(model)
    assert has_l3_real(canon, opt), f"{model}×{opt} is not L3-REAL (register it first)"
    assert gemm_impl_for_cell(canon, opt, "bf16") == "wgmma", "bf16 engine != wgmma"

    # The kernel's PERSISTED d lives at state[4*total+3] (the LAST scalar of the
    # prodigy state layout [m | v | s_track | loss | param_init(total) | r_ema |
    # s_ema | d_lr] = 4*total+4; see grokking_optimizers/dispatch.py ~:1828-1836).
    def _kernel_d(state):
        return float(state[4 * total + 3].item())

    # ── (a)+(c) parity reference (d adapts) + (d) d-frozen control, identical init.
    g, c, m_k, data, dev = _build_cell(model)
    tx, ty = data[0], data[1]
    named_k = [(n, p) for n, p in m_k.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named_k)

    g2, c2, m_ref, data2, dev2 = _build_cell(model)
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    assert torch.equal(_flat(named_k), _flat(named_ref)), "init mismatch (seed)"
    opt_ref = _mk_prodigy(g2, m_ref.parameters(), c2, d0=PR_D0, d_coef=PR_D_COEF)

    # d-FROZEN control: d_coef=0 ⇒ candidate d_coef·r/|s| = 0 ⇒ d = max(d_prev,0) =
    # d0 forever. SAME init, SAME fed grads, independent model. If it DIVERGES from
    # the d-adapting reference, the d-adaptation is load-bearing (the gate has teeth).
    g3, c3, m_ctl, data3, dev3 = _build_cell(model)
    named_ctl = [(n, p) for n, p in m_ctl.named_parameters() if p.requires_grad]
    opt_ctl = _mk_prodigy(g3, m_ctl.parameters(), c3, d0=PR_D0, d_coef=0.0)

    # A controlled anchor delta (reproducible, device-pinned) injected IDENTICALLY
    # into the kernel state slot AND the eager state at step-2 entry, so p0−p = delta
    # is nonzero + EXACT on every side → the candidate clears d_prev and d MOVES off
    # d0 deterministically (the stable owner-block-unit technique).
    #
    # The SIGN of delta is load-bearing for d-MOVEMENT, not just its magnitude. The
    # Prodigy d-candidate is d_coef·r_ema/|s_ema| with r_ema ∝ <g, p0−p> = <g, delta>;
    # since d = max(d_prev, candidate), d only MOVES off d0 when that inner product is
    # POSITIVE. A sign-RANDOM delta gives <g, delta> ≈ 0 with a coin-flip sign — and
    # for the decoder's (and vit's) actual step-1 reduced grad it comes out NEGATIVE
    # (measured <g, delta>≈−3.6e-3 decoder / −3.2e-3 vit at seed 12345), so the
    # candidate is negative and d stays frozen at d0 → the vacuity guard (assert (b))
    # fires even though the kernel d-update is correct. (This was the latent bug in
    # the sign-random construction shared with tuning/_prodigy_owner_block_unit.py.)
    # Fix: build a reproducible MAGNITUDE and ALIGN its sign with the step-1 reduced
    # grad at injection time so <g, delta> = Σ|g|·|delta_mag| > 0 STRONGLY (≈+1.0),
    # which moves d deterministically off d0 while keeping the anchor EXACT + identical
    # on the kernel and both eager sides. delta is filled in-place at the inject step.
    gen = torch.Generator(device=dev).manual_seed(12345)
    delta_mag = torch.rand(total, generator=gen, device=dev) * PR_DELTA_SCALE  # |δ|∈[0,scale)
    delta = torch.zeros(total, device=dev)   # filled = delta_mag·sign(g) at inject step

    def _inject_anchor_eager(opt_obj, named):
        """Overwrite each param's eager param_init with p_e+delta and reset the
        persisted d scalars to cold-start (d=d0, r_ema=s_ema=0). MIRRORS the kernel
        injection below. The eager Prodigy seeds param_init on its first .step()
        (prodigy.py:127), so this runs AFTER step 1; we also invalidate the static
        cache so the next step re-reads the overwritten param_init buffers."""
        off = 0
        for _, p in named:
            n = p.numel()
            st = opt_obj.state[p]
            st["param_init"] = (p.detach().float().reshape(-1)
                                + delta[off:off + n]).reshape(p.shape).clone()
            off += n
        opt_obj._d_lr = PR_D0
        opt_obj._r_ema = 0.0
        opt_obj._s_ema = 0.0
        opt_obj._static_cache = {}      # force re-read of the overwritten param_init

    def _inject_anchor_kernel(state):
        """Overwrite the kernel's param_init slice (state[3*total+1:4*total+1]) with
        flat(p_e)+delta and reset the persisted [r_ema|s_ema|d_lr] to cold-start.
        flat(p_e) == the cache `flat` (current params at step-2 entry)."""
        flat = cache[canon]["flat"]
        state[3 * total + 1:4 * total + 1].copy_(flat + delta)
        state[4 * total + 1] = 0.0      # r_ema
        state[4 * total + 2] = 0.0      # s_ema
        state[4 * total + 3] = PR_D0    # d_lr (persisted d_prev)

    cache = {}
    worst_param = worst_m = worst_v = worst_s = worst_d = 0.0
    d_grew_step = None
    ctl_div_at_deadline = 0.0
    final_param_rel = None

    for step in range(1, PR_N_STEPS + 1):
        # (1) ONE real TC megakernel step (real fwd+bwd+prodigy tail), wgmma. The
        #     kernel reads d0/d_coef/beta3 via _opt_scalars_from(opt_ref).
        loss, grad = fused_train_step(canon, opt, m_k, opt_ref, tx, ty,
                                      state_cache=cache, step=step,
                                      return_grad=True, gemm_impl="wgmma")
        assert loss == loss and not torch.isnan(grad).any(), (
            f"prodigy multi-step DIVERGED at step {step}: loss={loss} "
            f"grad_has_nan={bool(torch.isnan(grad).any())}")

        # (2) eager reference (d adapts) fed the SAME reduced grad.
        _inject_grad(named_ref, grad)
        opt_ref.step()

        # (3) d-frozen control (d_coef=0) fed the SAME reduced grad.
        _inject_grad(named_ctl, grad)
        opt_ctl.step()

        # ── controlled-anchor injection at step-2 entry: AFTER step 1 has seeded
        # both sides' param_init (= the cold-start anchor) and applied one update,
        # overwrite param_init = p_e+delta IDENTICALLY on the kernel + eager + control
        # so the step≥2 d-update sees a known nonzero p0−p and d moves off d0. (The
        # control is anchored too so its ONLY difference from the reference is d_coef.)
        if step == PR_ANCHOR_INJECT_STEP - 1:
            # Fill delta = |delta_mag|·sign(g) using THIS step's reduced grad so the
            # step≥2 d-update sees <g, delta> = Σ|g|·|delta_mag| > 0 and d MOVES off d0
            # (a sign-random delta gives a coin-flip inner product that is NEGATIVE here
            # → d frozen at d0 → vacuous gate). delta is filled in-place so the kernel +
            # both eager anchor injectors (which close over `delta`) inject the IDENTICAL,
            # now-sign-aligned, anchor. The grad is byte-identical across kernel/eager.
            delta.copy_(delta_mag * grad.sign())
            _inject_anchor_kernel(cache[canon]["state"])
            _inject_anchor_eager(opt_ref, named_ref)
            _inject_anchor_eager(opt_ctl, named_ctl)

        # (4) compare kernel params/state/d vs the d-adapting reference.
        p_k = _flat(named_k)
        p_ref = _flat(named_ref)
        param_rel = _rel(p_k, p_ref)
        worst_param = max(worst_param, param_rel)
        final_param_rel = param_rel

        kstate = cache[canon]["state"]
        k_m = kstate[0:total]
        k_v = kstate[total:2 * total]
        k_s = kstate[2 * total:3 * total]
        k_d = _kernel_d(kstate)
        r_m, r_v, r_s = _gather_prodigy_state(opt_ref, named_ref, total, dev)
        worst_m = max(worst_m, _rel(k_m, r_m))
        worst_v = max(worst_v, _rel(k_v, r_v))
        worst_s = max(worst_s, _rel(k_s, r_s))
        eager_d = float(opt_ref._d_lr)
        worst_d = max(worst_d, abs(k_d - eager_d) / (abs(eager_d) + 1e-30))
        if d_grew_step is None and eager_d > PR_D0 * 1.0001:
            d_grew_step = step

        # (5) control divergence: d-frozen vs the d-adapting reference.
        ctl_div = _rel(_flat(named_ctl), p_ref)
        if step <= PR_CTRL_DEADLINE:
            ctl_div_at_deadline = max(ctl_div_at_deadline, ctl_div)

    # ── ASSERT (a): the kernel's persisted d tracks the eager opt._d_lr the whole
    # way (and m/v/s state too) to fp32-reorder tol.
    finite = all(w == w for w in (worst_param, worst_m, worst_v, worst_s, worst_d))
    assert finite, "prodigy multi-step: a NaN entered the worst-reduction (run diverged)"
    assert worst_d < TOL, (
        f"prodigy multi-step: kernel persisted d does NOT track eager opt._d_lr: "
        f"worst d_rel={worst_d:.3e} >= tol {TOL:.0e} — the in-kernel d-estimate "
        f"(beta3-EMA + d_coef + persist) diverged from the canonical eager d.")
    assert worst_m < TOL and worst_v < TOL and worst_s < TOL, (
        f"prodigy multi-step STATE parity FAIL: m={worst_m:.3e} v={worst_v:.3e} "
        f"s={worst_s:.3e} (tol {TOL:.0e}) — kernel [m|v|s_track] != eager Prodigy")

    # ── ASSERT (b): d ACTUALLY grew off d0 (the adaptation fired — not frozen). If
    # the controlled anchor failed to move d, this run never exercised the d-update
    # branch and the whole gate would be vacuous.
    assert d_grew_step is not None, (
        f"prodigy multi-step: the eager d NEVER grew off d0={PR_D0:.0e} in "
        f"{PR_N_STEPS} steps — the controlled anchor (d_coef={PR_D_COEF}, "
        f"delta_scale={PR_DELTA_SCALE:.0e}) did not trip the candidate over d_prev, "
        f"so the d-adaptation never fired and the gate is vacuous.")

    # ── ASSERT (c): params within tol at the FINAL step (the adaptive d has been
    # live since step 2, so this is parity UNDER an active d-adaptation).
    assert final_param_rel is not None and final_param_rel < TOL, (
        f"prodigy param parity FAIL at the final step ({PR_N_STEPS}): "
        f"rel={final_param_rel:.3e} >= tol {TOL:.0e} (d adapting since step 2)")

    # ── ASSERT (d): the d-FROZEN control DIVERGES from the adaptive run by step 50,
    # proving the d-adaptation is load-bearing (the test has teeth — a kernel that
    # FROZE d would match this control, not the d-adapting reference asserted above).
    assert ctl_div_at_deadline > CTRL_MIN_DIVERGENCE, (
        f"prodigy multi-step: the d-FROZEN control (d_coef=0) did NOT diverge from "
        f"the d-adapting reference by step {PR_CTRL_DEADLINE} "
        f"(max ctl divergence={ctl_div_at_deadline:.3e} <= {CTRL_MIN_DIVERGENCE:.0e}) "
        f"— the d-adaptation is then not demonstrably load-bearing and the kernel's "
        f"d-tracking (a) could be a frozen-d coincidence.")
