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
    steps and an fp64 GrokAdamW reference SIDE-BY-SIDE on the kernel's own
    TC-reduced grad, asserts params match the gate's tol at {1, 50, 60}, asserts
    the clip condition ‖g‖>max_norm ACTUALLY occurred by step 50 (so the test
    cannot vacuously pass on a coincidentally-small norm), and re-runs the whole
    60-step fused trajectory twice more bit-identical (A/A/A).

  * prodigy: the single-step gate is BLIND to the d-adaptation — at step 1 the
    trajectory anchor param_init == params so r = Σ d²·<g, p0−p> = 0 ⇒ d stays
    frozen at d0=1e-6 (the cold start). The DEFINING prodigy mechanism (the
    adaptive d grown from the cumulative parameter trajectory) only fires at
    step≥2. At the PRODUCTION d0=1e-6/d_coef=1.0 d is INERT for the decoder (the
    slow real <g,p0−p> never trips the candidate over d0; natural trajectory
    forcing is unstable — small d0 won't bootstrap, large d0·lr → NaN, per
    tuning/_prodigy_owner_block_unit.py). So test_prodigy_multistep_parity uses
    the STABLE controlled-anchor technique that owner-block unit validated:
    inject a known param_init = p_e + delta into BOTH the kernel state slot and
    the fp64 reference state IDENTICALLY at step 2 (so p0−p = delta is nonzero +
    EXACT on both sides) with d_coef>1 (makes the "persist UNSCALED, scale only
    the candidate" line load-bearing) — d then GROWS off d0 deterministically. It
    asserts the kernel's persisted d (read from state[4*total+3]; layout in
    grokking_optimizers/dispatch.py ~:1828-1836) tracks the fp64 d each step,
    that d actually grew off d0, that params match tol at the final step, and
    that a d-FROZEN control (d_coef=0 ⇒ candidate=0 ⇒ d=max(d_prev,0)=d0 forever)
    DIVERGES from the adaptive run by step 50 — proving the d-adaptation is
    load-bearing (the gate has teeth).

THE REFERENCE IS PURE fp64 (NOT an eager per-op kernel). The in-house eager
per-op kernels (grokadamw_fused_step / prodigy_fused_step) are being removed, so
the side-by-side reference is now an fp64 MULTI-STEP DRIVER built on the canonical
single-source refs in tests/hw/test_reference_parity.py (the fp64 transcription of
csrc/algorithms/<opt>.h, the single source of truth):

  * grokadamw → ref_grokadamw_step driven PER-TENSOR over the N steps (each tensor
    carries its PUBLISHED layer-wise β1 = β1·(1−γ)^layer, so its bias correction
    bc1 = 1−β1_i^t is per-tensor too — exactly what the kernel's P3 does, see
    csrc/bindings/bindings.cpp:229-244). The kernel's P2.5 GLOBAL grad-norm clip
    (clip_coef = grad_clip/(‖g‖₂+1e-6) when ‖g‖₂>grad_clip, applied to the flat
    reduced grad BEFORE the EMA/amplify/Adam tail — csrc/bindings/helpers.h:311) is
    reproduced on the reference grad so the reference tracks the kernel through the
    clip onset. The gradient-EMA is seeded with the FIRST step's grad (matching the
    kernel + eager: state["ema"] = grads_by_id[id(p)], grokadamw.py:303), and α
    stays at α_init (the multistep run feeds no train/val losses, so the grokking
    signal is inactive and α = α_init = 0.98 the whole way).

  * prodigy → the d-trajectory is driven with ref_prodigy_partials (the per-step
    GLOBAL r,s partials Σ d_prev²·<g,p0−p> / Σ d_prev²·|g|), the β3-EMA decay +
    d_coef + ref_prodigy_update_d (= max(d_prev, d_coef·r_ema/|s_ema|)), then
    ref_prodigy_apply_step (AdamW with step size d_new). The EXACT order is
    byte-faithful to the kernel cell (csrc/fused/sm_90/fused_decoder_megakernel.cuh
    :1009-1052, which itself byte-matches the removed eager
    launch_multi_tensor_prodigy_fused_reduce_step):
      d_prev  = persisted d_lr (step 1: d0 cold-start)
      r_step,s_step = Σ d_prev²·<g,p0−p>, Σ d_prev²·|g|      (at the CURRENT params)
      r_ema  <- beta3·r_ema + r_step    (decay UNSCALED persisted r_ema, then add)
      s_ema  <- beta3·s_ema + s_step
      d_new   = max(d_prev, d_coef·r_ema/|s_ema|)            (d_coef scales ONLY the
                                                              candidate; r_ema UNSCALED)
      apply AdamW with step size d_new; persist (r_ema UNSCALED, s_ema, d_new)
    beta3 = sqrt(beta2) (prodigy.py:178). The controlled anchor the test injects is
    threaded into the fp64 p0 identically to the kernel slot.

REUSE. The model/data/init builder (_build_cell), the grokadamw factory
(_grokadamw_factory), and the GPU-ready skipif predicate are IMPORTED from
tests.hw.test_l3tc_tail_gate so this gate drives the EXACT same model, init,
hyperparameters, and determinism pinning as the single-step gate (no copy-paste
that can drift). The fused-step driver is grokking_optimizers.dispatch.
fused_train_step (the SAME production route the race and the single-step gate
take). The GrokAdamW / Prodigy optimizer object is STILL constructed — but ONLY as
the live SCALAR SOURCE the kernel-under-test reads through _opt_scalars_from inside
fused_train_step (lr/betas/eps/wd/α/γ/grad_clip for grokadamw; d0/d_coef/beta3 for
prodigy). Its .step() is NO LONGER called: the reference update is the fp64 driver
above. _mk_prodigy (a thin Prodigy builder that mirrors test_l3tc_tail_gate.
_prodigy_factory but ALSO accepts d0/d_coef overrides — _prodigy_factory hard-codes
the production defaults and cannot force the d-adaptation; the override is the whole
point here, and the d_coef=0 case is the d-frozen control) is the only locally-
mirrored builder. The _gpu_ready skipif predicate is imported from the tail gate
and applied with the SAME @pytest.mark.hw + skipif pattern, so these skip cleanly
off-GPU and run from HARDWARE_VALIDATION.md on real accelerators.

Run:
  PYTHONPATH=. python -m pytest tests/hw/test_multistep_parity.py -q
  CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. python -m pytest \
        tests/hw/test_multistep_parity.py -m hw -q
"""
from __future__ import annotations

import math

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

# The CANONICAL fp64 single-source refs (transcribed 1:1 from csrc/algorithms/<opt>.h
# in tests/hw/test_reference_parity.py — the single source of truth). These ARE the
# reference now (the in-house eager per-op kernels are being removed).
from tests.hw.test_reference_parity import (
    ref_grokadamw_step,
    ref_prodigy_apply_step,
    ref_prodigy_partials,
    ref_prodigy_update_d,
)


# ── Tolerances (mirror tuning/_grokadamw_multistep_parity.py + _prodigy_multistep_
# parity.py — the same single-source numbers the reference probes calibrated). TOL
# is the kernel-vs-fp64 fp32-reorder budget; it MUST be tighter than the control
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


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


# ── fp64 GrokAdamW reference state (flat, in named_parameters() order). ─────────
class _GrokAdamWRef:
    """Pure-fp64 multi-step GrokAdamW driver (the side-by-side reference that
    REPLACES the removed eager opt.step()).

    Holds the fp64 [m | v | ema] state flat in named-parameters() order and drives
    the canonical ref_grokadamw_step PER TENSOR each step (each tensor with its
    PUBLISHED layer-wise β1_i = β1·(1−γ)^layer_index, so bc1 = 1−β1_i^t is per-tensor
    too — matching the kernel's P3). The GLOBAL grad-norm clip is applied to the flat
    reduced grad BEFORE the per-tensor tail (matching the kernel's P2.5 and the eager
    clip_grad_norms_device_side: clip_coef = grad_clip/(‖g‖₂+1e-6) when ‖g‖₂>max_norm).

    The gradient-EMA is seeded with the FIRST step's (unclipped) grad — exactly the
    kernel + eager seed (grokadamw.py:303 seeds state["ema"] = grads_by_id[id(p)] from
    the raw grad BEFORE the in-kernel clip; at step 1 the clip is inert for the decoder
    so seed==clipped anyway). α is α_init the whole way (no train/val signal is fed in
    this run, so _signal_active stays False — see _opt_scalars_from / _alpha_for_group).
    """

    def __init__(self, named, c):
        self.sizes = [int(p.numel()) for _, p in named]
        self.shapes = [tuple(p.shape) for _, p in named]
        self.dev = named[0][1].device
        total = sum(self.sizes)
        # fp64 per-tensor state, flat (named order).
        self.m = [torch.zeros(s, dtype=torch.float64, device=self.dev) for s in self.sizes]
        self.v = [torch.zeros(s, dtype=torch.float64, device=self.dev) for s in self.sizes]
        self.ema = None    # seeded with the first grad (kernel/eager-faithful)
        # PUBLISHED hyperparameters (the SAME values _grokadamw_factory passes; eps is
        # the GrokAdamW constructor default 1e-8, which is also what _opt_scalars_from
        # forwards to the kernel from the optimizer's group).
        self.beta1 = float(c["beta1"])
        self.beta2 = float(c["beta2"])
        self.gamma = float(c.get("grokadamw_gamma", 0.1))
        self.alpha = float(c.get("grokadamw_alpha", 0.98))   # α_init (no signal ⇒ stays here)
        self.lamb = float(c.get("grokadamw_lamb", 2.0))
        self.lr = float(c["lr"])
        self.wd = float(c["weight_decay"])
        self.eps = 1e-8
        self.max_norm = float(c.get("grokadamw_grad_clip", 1.0))
        # PUBLISHED layer-wise β1 = β1·(1−γ)^i across the FLAT param enumeration (the
        # SAME global-order formula as grokadamw.py:_layer_beta1_by_id and
        # _opt_scalars_from's per-tensor β1). gamma=0 ⇒ a single global β1.
        self.beta1s = [self.beta1 * ((1.0 - self.gamma) ** i)
                       for i in range(len(self.sizes))]

    def _clip(self, flat_grad):
        """GLOBAL grad-norm clip on the flat reduced grad (kernel P2.5 / eager
        clip_grad_norms_device_side). Returns the (possibly clipped) flat grad and
        the pre-clip global L2 norm (the clip discriminant the test records)."""
        g64 = flat_grad.detach().to(torch.float64)
        gnorm = g64.norm().item()
        if self.max_norm > 0.0 and gnorm > self.max_norm:
            g64 = g64 * (self.max_norm / (gnorm + 1e-6))
        return g64, gnorm

    def step(self, flat_grad, t):
        """One fp64 GrokAdamW step on the flat reduced grad. Returns the pre-clip
        global grad norm (the clip discriminant)."""
        g_raw = flat_grad.detach().to(torch.float64)        # UNCLIPPED (the ema seed)
        g_clip, gnorm = self._clip(flat_grad)               # CLIPPED (filter + amp + tail)
        # Per-tensor split of BOTH the clipped and the raw flat grad (named order).
        gslices, graw_slices = [], []
        off = 0
        for n in self.sizes:
            gslices.append(g_clip[off:off + n])
            graw_slices.append(g_raw[off:off + n])
            off += n
        if self.ema is None:
            # COLD-START: seed the gradient-EMA with the FIRST step's UNCLIPPED grad —
            # BYTE-FAITHFUL to the kernel + eager (opt_components.cuh:451 `if (step==1)
            # st.ema[idx] = grad[idx]`; grokadamw.py:303 clones state["ema"] from the
            # RAW grad BEFORE the in-place clip). The EMA FILTER + amplification below
            # then run on the CLIPPED grad (gc), so seeding the unclipped grad is what
            # keeps the m/ema state matching the kernel when the clip FIRES (when it is
            # inert, raw==clipped and this is identical). ref_grokadamw_step computes
            # ema_new = alpha*ema_prev + (1-alpha)*g with g==gc, so passing the clipped
            # gs as g and the unclipped seed as ema_prev reproduces the kernel exactly.
            self.ema = [gr.clone() for gr in graw_slices]
        for i, gs in enumerate(gslices):
            # The canonical per-tensor fp64 GrokAdamW step with THIS tensor's β1_i
            # (and hence bc1_i = 1−β1_i^t computed inside ref_grokadamw_step).
            p_new, m_new, v_new, ema_new = ref_grokadamw_step(
                self._params[i], gs, self.m[i], self.v[i], self.ema[i],
                alpha=self.alpha, lamb=self.lamb, lr=self.lr,
                beta1=self.beta1s[i], beta2=self.beta2, eps=self.eps,
                wd=self.wd, t=t)
            self._params[i].data.copy_(p_new.reshape(self.shapes[i]))
            self.m[i] = m_new
            self.v[i] = v_new
            self.ema[i] = ema_new
        return gnorm

    def bind_params(self, named):
        """Bind the live (fp64) param tensors this driver updates in place."""
        self._params = [p for _, p in named]

    def flat_state(self, total):
        """Flat [m | v | ema] (fp64) for the (1b) state parity comparison."""
        m = torch.cat([x.reshape(-1) for x in self.m])
        v = torch.cat([x.reshape(-1) for x in self.v])
        ema = torch.cat([x.reshape(-1) for x in self.ema])
        return m, v, ema


def _mk_prodigy(g, params, c, d0, d_coef):
    """Build the REAL Prodigy carrying the production hyperparameters, with
    d0/d_coef OVERRIDABLE. Used here ONLY as the live SCALAR SOURCE the kernel reads
    through _opt_scalars_from (d0/d_coef/beta3) — its .step() is NOT called; the
    reference d-trajectory is the fp64 driver below.

    MIRRORS tests.hw.test_l3tc_tail_gate._prodigy_factory (~:128-135) — the SAME
    lr=prodigy_lr (1.0), weight_decay, betas=(beta1,beta2) — but _prodigy_factory
    hard-codes the production d0=1e-6/d_coef=1.0 (at which d is INERT for the
    decoder), so it cannot exercise the d-update branch. The override is the whole
    reason this local builder exists; the d_coef=0 case is the d-frozen control."""
    return g.Prodigy(params, lr=c.get("prodigy_lr", 1.0),
                     weight_decay=c["weight_decay"],
                     betas=(c["beta1"], c["beta2"]),
                     d0=d0, d_coef=d_coef)


class _ProdigyRef:
    """Pure-fp64 multi-step Prodigy d-trajectory driver (the reference that REPLACES
    the removed eager opt.step()).

    Drives the canonical ref_prodigy_partials + β3-EMA + d_coef + ref_prodigy_update_d
    + ref_prodigy_apply_step in the EXACT byte-faithful order of the kernel cell
    (fused_decoder_megakernel.cuh:1009-1052):
      d_prev = persisted d_lr (step 1: d0 cold-start)
      r_step,s_step = Σ d_prev²·<g,p0−p>, Σ d_prev²·|g|   (GLOBAL, at CURRENT params)
      r_ema  <- beta3·r_ema + r_step      (decay UNSCALED persisted r_ema, then add)
      s_ema  <- beta3·s_ema + s_step
      d_new   = max(d_prev, d_coef·r_ema/|s_ema|)         (d_coef scales ONLY candidate)
      apply AdamW with step size d_new; persist (r_ema UNSCALED, s_ema, d_new)

    No grad-norm clip (Prodigy has no grad_clip; the eager prodigy_fused_step does not
    clip). The controlled anchor is injected into self.p_init identically to the kernel
    slot via reset_anchor()."""

    def __init__(self, named, c, d0, d_coef):
        self.sizes = [int(p.numel()) for _, p in named]
        self.shapes = [tuple(p.shape) for _, p in named]
        self.dev = named[0][1].device
        self.m = [torch.zeros(s, dtype=torch.float64, device=self.dev) for s in self.sizes]
        self.v = [torch.zeros(s, dtype=torch.float64, device=self.dev) for s in self.sizes]
        self.s_track = [torch.zeros(s, dtype=torch.float64, device=self.dev) for s in self.sizes]
        # Trajectory anchor p0: seeded = the CURRENT params at the first step (eager-
        # faithful cold start, prodigy.py:127), so r = Σ d²·<g,p0−p> = 0 at step 1.
        self.p_init = None
        self.beta1 = float(c["beta1"])
        self.beta2 = float(c["beta2"])
        self.beta3 = math.sqrt(self.beta2)     # prodigy.py:178
        self.lr = float(c.get("prodigy_lr", 1.0))   # Prodigy step is d, not lr; kept for parity
        self.wd = float(c["weight_decay"])
        self.eps = 1e-8
        self.d0 = float(d0)
        self.d_coef = float(d_coef)
        # Persisted estimator scalars (mirror prodigy_persist[0/1/2] = r_ema/s_ema/d_lr).
        self.r_ema = 0.0
        self.s_ema = 0.0
        self.d_lr = float(d0)

    def bind_params(self, named):
        self._params = [p for _, p in named]

    def _seed_anchor_if_needed(self):
        if self.p_init is None:
            self.p_init = [p.detach().to(torch.float64).reshape(-1).clone()
                           for p in self._params]

    def reset_anchor(self, delta):
        """Overwrite p0 = p_e + delta (the controlled anchor) and reset the persisted
        estimator to cold-start (d=d0, r_ema=s_ema=0). MIRRORS the kernel injection
        (_inject_anchor_kernel below). p_e == the CURRENT params at the inject step.
        delta is a FLAT fp64 tensor (whole-vector, named order)."""
        self.p_init = []
        off = 0
        for i, p in enumerate(self._params):
            n = self.sizes[i]
            base = p.detach().to(torch.float64).reshape(-1)
            self.p_init.append((base + delta[off:off + n].to(torch.float64)).clone())
            off += n
        self.d_lr = self.d0
        self.r_ema = 0.0
        self.s_ema = 0.0

    def step(self, flat_grad, t):
        """One fp64 Prodigy step on the flat reduced grad. Returns d_new (the
        persisted d after this step)."""
        self._seed_anchor_if_needed()
        g64 = flat_grad.detach().to(torch.float64)
        gslices = []
        off = 0
        for n in self.sizes:
            gslices.append(g64[off:off + n])
            off += n
        d_prev = self.d_lr
        # (a) GLOBAL r,s partials at the CURRENT params vs p0, with d_prev (sum over
        #     ALL tensors — ref_prodigy_partials already returns per-tensor sums).
        r_step = torch.zeros((), dtype=torch.float64, device=self.dev)
        s_step = torch.zeros((), dtype=torch.float64, device=self.dev)
        for i, gs in enumerate(gslices):
            p_flat = self._params[i].detach().to(torch.float64).reshape(-1)
            r_i, s_i = ref_prodigy_partials(p_flat, self.p_init[i], gs, d_prev=d_prev)
            r_step = r_step + r_i
            s_step = s_step + s_i
        # (b) decay persisted (UNSCALED) r_ema/s_ema by beta3, then add this step's Σ.
        r_ema = self.beta3 * self.r_ema + float(r_step.item())
        s_ema = self.beta3 * self.s_ema + float(s_step.item())
        # (c) d = max(d_prev, d_coef·r_ema/|s_ema|). d_coef scales ONLY the candidate
        #     (ref_prodigy_update_d does max(d_prev, r/|s|) verbatim, so fold d_coef
        #     into the numerator copy — persisted r_ema stays UNSCALED, eager parity).
        d_new = ref_prodigy_update_d(d_prev, self.d_coef * r_ema, s_ema)
        self.r_ema = r_ema        # persist UNSCALED
        self.s_ema = s_ema
        self.d_lr = d_new
        # (d) apply AdamW with step size d_new (per tensor; same d across the cell).
        for i, gs in enumerate(gslices):
            p_flat = self._params[i].detach().to(torch.float64).reshape(-1)
            p_new, m_new, v_new, s_new = ref_prodigy_apply_step(
                p_flat, gs, self.m[i], self.v[i], self.s_track[i],
                d=d_new, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                wd=self.wd, t=t)
            self._params[i].data.copy_(p_new.reshape(self.shapes[i]))
            self.m[i] = m_new
            self.v[i] = v_new
            self.s_track[i] = s_new
        return d_new

    def flat_state(self):
        m = torch.cat([x.reshape(-1) for x in self.m])
        v = torch.cat([x.reshape(-1) for x in self.v])
        s = torch.cat([x.reshape(-1) for x in self.s_track])
        return m, v, s


# ───────────────────────────── grokadamw ─────────────────────────────
@pytest.mark.hw
@pytest.mark.skipif(not _gpu_ready(), reason="needs built extension on GPU")
def test_grokadamw_multistep_parity():
    """~60 steps of the L3-TC fused decoder×grokadamw cell vs a PURE fp64 GrokAdamW
    reference on identical init/data, both fed the kernel's own TC-reduced grad.

    Asserts: (a) params match the gate's tol (1e-4) at steps {1, 50, 60}, with the
    GLOBAL grad-norm clip PROVABLY fired by step 50 (‖g‖>max_norm on the reduced
    grad — not a vacuous pass); (b) A/A/A — the full 60-step fused trajectory is
    bit-identical across three runs from the same init.

    State parity ([m|v|ema] vs the fp64 reference) is also asserted the whole way; a
    dropped clip or a dropped/garbage state blows past TOL once the norm crosses
    max_norm. The fp64 reference uses the SAME published layer-wise β1 + the SAME
    GLOBAL clip the kernel does (see _GrokAdamWRef).
    """
    from grokking_optimizers.dispatch import (has_l3_real, gemm_impl_for_cell,
                                              canonicalize_model, fused_train_step)
    model, opt = "decoder", "grokadamw"
    canon = canonicalize_model(model)
    assert has_l3_real(canon, opt), f"{model}×{opt} is not L3-REAL (register it first)"
    assert gemm_impl_for_cell(canon, opt, "bf16") == "wgmma", "bf16 engine != wgmma"

    # ── (a) parity: drive the kernel (m_k) and an fp64 reference (m_ref) from the
    # SAME seeded init; the fp64 driver consumes the kernel's reduced grad each step.
    g, c, m_k, data, dev = _build_cell(model)
    tx, ty = data[0], data[1]
    named_k = [(n, p) for n, p in m_k.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named_k)
    max_norm = float(c.get("grokadamw_grad_clip", 1.0))

    g2, c2, m_ref, data2, dev2 = _build_cell(model)
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    assert torch.equal(_flat(named_k), _flat(named_ref)), "init mismatch (seed)"
    # The GrokAdamW object is the live SCALAR SOURCE the KERNEL reads through
    # _opt_scalars_from inside fused_train_step (lr/betas/eps/wd/α/γ/grad_clip). Its
    # .step() is NOT called — the reference is the fp64 driver bound to m_ref below.
    opt_scalars = _grokadamw_factory(g, m_k.parameters(), c)
    ref = _GrokAdamWRef(named_ref, c2)
    ref.bind_params(named_ref)

    cache = {}
    clip_fired_step = None
    parity_at = {}            # step -> (param_rel, m_rel, v_rel, ema_rel)
    worst_param = worst_m = worst_v = worst_ema = 0.0

    for step in range(1, GA_N_STEPS + 1):
        # (1) ONE real TC megakernel step (real fwd+bwd+grokadamw tail), wgmma. The
        #     kernel reads the live scalars via _opt_scalars_from(opt_scalars).
        loss, grad = fused_train_step(canon, opt, m_k, opt_scalars, tx, ty,
                                      state_cache=cache, step=step,
                                      return_grad=True, gemm_impl="wgmma")
        # NaN-FAIL-LOUD (never let a NaN slide into a masked max()-reduction that
        # false-greens): a NaN loss/grad is a hard failure at the offending step.
        assert loss == loss and not torch.isnan(grad).any(), (
            f"grokadamw multi-step DIVERGED at step {step}: loss={loss} "
            f"grad_has_nan={bool(torch.isnan(grad).any())}")

        # (2) fp64 reference fed the SAME reduced grad → the SAME canonical grokadamw
        #     update (per-tensor layer-wise β1 + the GLOBAL clip). Returns the pre-clip
        #     global grad norm — the clip-fire DISCRIMINANT (the kernel's P2.5 clip
        #     bites when the global L2 of the reduced grad exceeds max_norm). Record
        #     the first step it does (what the single-step gate at step 1 is blind to).
        gnorm = ref.step(grad, step)
        if clip_fired_step is None and gnorm > max_norm:
            clip_fired_step = step

        # (3) compare kernel params + [m|v|ema] state vs the fp64 reference.
        p_k = _flat(named_k)
        p_ref = _flat(named_ref)
        param_rel = _rel(p_k, p_ref)
        kstate = cache[canon]["state"]
        k_m = kstate[0:total]
        k_v = kstate[total:2 * total]
        k_ema = kstate[2 * total:3 * total]
        r_m, r_v, r_ema = ref.flat_state(total)
        m_rel = _rel(k_m.double(), r_m)
        v_rel = _rel(k_v.double(), r_v)
        ema_rel = _rel(k_ema.double(), r_ema)
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
            f"ema={er:.3e} (tol {TOL:.0e}) — kernel state != fp64 GrokAdamW")
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
    """~50 steps of the L3-TC fused decoder×prodigy cell vs a PURE fp64 Prodigy
    reference on identical init/data, both fed the kernel's own TC-reduced grad,
    with the d-adaptation FORCED to fire via a controlled anchor (the stable
    technique from tuning/_prodigy_owner_block_unit.py — at production
    d0=1e-6/d_coef=1.0 d is inert for the decoder, and natural trajectory forcing
    is NaN-unstable).

    Asserts: (a) the kernel's PERSISTED d-estimate (state[4*total+3]) tracks the
    fp64 d each step to TOL; (b) d actually GREW off d0=1e-6 (the adaptation fired —
    not frozen); (c) params match the gate's tol at the final step; (d) a d-FROZEN
    control (d_coef=0 ⇒ d pinned at d0) DIVERGES from the adaptive run by step 50
    (proving the d-adaptation is load-bearing).
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
    # The Prodigy object is the live SCALAR SOURCE the KERNEL reads through
    # _opt_scalars_from inside fused_train_step (d0/d_coef/beta3 + lr/betas/eps/wd).
    # Its .step() is NOT called; the reference is the fp64 d-trajectory driver below.
    opt_scalars = _mk_prodigy(g, m_k.parameters(), c, d0=PR_D0, d_coef=PR_D_COEF)
    ref = _ProdigyRef(named_ref, c2, d0=PR_D0, d_coef=PR_D_COEF)
    ref.bind_params(named_ref)

    # d-FROZEN control: d_coef=0 ⇒ candidate d_coef·r/|s| = 0 ⇒ d = max(d_prev,0) =
    # d0 forever. SAME init, SAME fed grads, independent fp64 driver. If it DIVERGES
    # from the d-adapting reference, the d-adaptation is load-bearing (gate has teeth).
    g3, c3, m_ctl, data3, dev3 = _build_cell(model)
    named_ctl = [(n, p) for n, p in m_ctl.named_parameters() if p.requires_grad]
    ctl = _ProdigyRef(named_ctl, c3, d0=PR_D0, d_coef=0.0)
    ctl.bind_params(named_ctl)

    # A controlled anchor delta (reproducible, device-pinned) injected IDENTICALLY
    # into the kernel state slot AND the fp64 references at step-2 entry, so p0−p =
    # delta is nonzero + EXACT on every side → the candidate clears d_prev and d MOVES
    # off d0 deterministically (the stable owner-block-unit technique).
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
    # on the kernel and both fp64 sides. delta is filled in-place at the inject step.
    gen = torch.Generator(device=dev).manual_seed(12345)
    delta_mag = torch.rand(total, generator=gen, device=dev) * PR_DELTA_SCALE  # |δ|∈[0,scale)
    delta = torch.zeros(total, device=dev)   # filled = delta_mag·sign(g) at inject step

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
        #     kernel reads d0/d_coef/beta3 via _opt_scalars_from(opt_scalars).
        loss, grad = fused_train_step(canon, opt, m_k, opt_scalars, tx, ty,
                                      state_cache=cache, step=step,
                                      return_grad=True, gemm_impl="wgmma")
        assert loss == loss and not torch.isnan(grad).any(), (
            f"prodigy multi-step DIVERGED at step {step}: loss={loss} "
            f"grad_has_nan={bool(torch.isnan(grad).any())}")

        # (2) fp64 reference (d adapts) fed the SAME reduced grad → the canonical
        #     d-trajectory (partials → β3-EMA → d_coef·max → AdamW-with-d). Returns d.
        eager_d = ref.step(grad, step)

        # (3) d-frozen control (d_coef=0) fed the SAME reduced grad.
        ctl.step(grad, step)

        # ── controlled-anchor injection at step-2 entry: AFTER step 1 has applied one
        # update on EVERY side (kernel + both fp64 refs), so p_e == the post-step-1
        # params on each side (matching the removed reference's order: kernel.step →
        # ref.step → ctl.step → inject). Fill delta = |delta_mag|·sign(g) using THIS
        # (step-1) reduced grad so the step≥2 d-update sees <g, delta> = Σ|g|·|delta_mag|
        # > 0 and d MOVES off d0 (a sign-random delta gives a coin-flip inner product
        # that is NEGATIVE here → d frozen at d0 → vacuous gate). Then overwrite p0 =
        # p_e + delta IDENTICALLY on the kernel state slot + both fp64 references (the
        # grad is byte-identical across all three), and reset the persisted (d=d0,
        # r_ema=s_ema=0). The kernel's `flat` holds the post-step-1 params here (the
        # megakernel updated it in place), so its p0 == post-step-1 + delta, matching
        # each fp64 reset_anchor (which reads its now-post-step-1 params). The step-2
        # reduction on every side then sees p0−p = delta and d moves off d0.
        if step == PR_ANCHOR_INJECT_STEP - 1:
            delta.copy_(delta_mag * grad.sign())
            _inject_anchor_kernel(cache[canon]["state"])
            ref.reset_anchor(delta)
            ctl.reset_anchor(delta)

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
        r_m, r_v, r_s = ref.flat_state()
        worst_m = max(worst_m, _rel(k_m.double(), r_m))
        worst_v = max(worst_v, _rel(k_v.double(), r_v))
        worst_s = max(worst_s, _rel(k_s.double(), r_s))
        worst_d = max(worst_d, abs(k_d - eager_d) / (abs(eager_d) + 1e-30))
        if d_grew_step is None and eager_d > PR_D0 * 1.0001:
            d_grew_step = step

        # (5) control divergence: d-frozen vs the d-adapting reference.
        ctl_div = _rel(_flat(named_ctl), p_ref)
        if step <= PR_CTRL_DEADLINE:
            ctl_div_at_deadline = max(ctl_div_at_deadline, ctl_div)

    # ── ASSERT (a): the kernel's persisted d tracks the fp64 d the whole way (and
    # m/v/s state too) to fp32-reorder tol.
    finite = all(w == w for w in (worst_param, worst_m, worst_v, worst_s, worst_d))
    assert finite, "prodigy multi-step: a NaN entered the worst-reduction (run diverged)"
    assert worst_d < TOL, (
        f"prodigy multi-step: kernel persisted d does NOT track the fp64 d: "
        f"worst d_rel={worst_d:.3e} >= tol {TOL:.0e} — the in-kernel d-estimate "
        f"(beta3-EMA + d_coef + persist) diverged from the canonical fp64 d.")
    assert worst_m < TOL and worst_v < TOL and worst_s < TOL, (
        f"prodigy multi-step STATE parity FAIL: m={worst_m:.3e} v={worst_v:.3e} "
        f"s={worst_s:.3e} (tol {TOL:.0e}) — kernel [m|v|s_track] != fp64 Prodigy")

    # ── ASSERT (b): d ACTUALLY grew off d0 (the adaptation fired — not frozen). If
    # the controlled anchor failed to move d, this run never exercised the d-update
    # branch and the whole gate would be vacuous.
    assert d_grew_step is not None, (
        f"prodigy multi-step: the fp64 d NEVER grew off d0={PR_D0:.0e} in "
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
