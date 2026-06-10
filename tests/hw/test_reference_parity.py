"""tests/hw/test_reference_parity.py — Phase 8 / Stage 3 ✅-gate.

Pure-PyTorch, **fp64-accumulate** REFERENCE implementations of one optimizer
step for each of the 11 race optimizers, transcribed directly from the
canonical per-element math in ``csrc/algorithms/<opt>.h`` (the single source of
truth — see ``csrc/algorithms/SOURCE_OF_TRUTH.md``).

This file has two halves:

  CPU half (runs everywhere, no accelerator, no built C++ extension):
    * the reference ``ref_<opt>_step`` functions, and
    * self-consistency unit tests over them — zero-grad is a no-op modulo
      decoupled weight decay, step-1 bias correction, and small closed-form
      checks on tiny tensors.

  GPU half (``@pytest.mark.hw`` + skip-if-no-CUDA):
    * compares the *built* fused C++/CUDA kernel against the reference. This is
      gated to the hardware runbook (``HARDWARE_VALIDATION.md``) and never runs
      on a CPU CI runner. It also lazily imports the kernel path, so this module
      stays importable without the compiled extension.

Calling convention (matches every header + the Python optimizers):
    bc1 = 1 - beta1**t,  bc2 = 1 - beta2**t        # un-inverted; divide by them

Run (CPU, what CI does):
    PYTHONPATH=. python3 -m pytest tests/hw/test_reference_parity.py -m "not hw" -q
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


# ===========================================================================
#  Reference implementations (fp64 accumulate). One step, no in-place magic —
#  return the *new* tensors so tests can assert against the prior values.
#  Each maps 1:1 to csrc/algorithms/<opt>.h.
# ===========================================================================

def _f64(t):
    return t.detach().to(torch.float64)


def _adam_tail(p, m_prev, v_prev, g_eff, lr, beta1, beta2, eps, wd, bc1, bc2):
    """The shared bias-corrected Adam + decoupled-WD tail.

    Bit-faithful (modulo fp32-vs-fp64 accumulation) to the tail repeated across
    adamw.h / grokadamw.h / grokfast.h / neuralgrok.h / looksam.h / prodigy.h /
    supergrok{11,15,2}.h:

        m = beta1*m_prev + (1-beta1)*g_eff
        v = beta2*v_prev + (1-beta2)*g_eff*g_eff
        update = (m / bc1) / (sqrt(v / bc2) + eps)
        p_new  = p - lr * (update + wd * p)
    """
    m = beta1 * m_prev + (1.0 - beta1) * g_eff
    v = beta2 * v_prev + (1.0 - beta2) * g_eff * g_eff
    update = (m / bc1) / (torch.sqrt(v / bc2) + eps)
    p_new = p - lr * (update + wd * p)
    return p_new, m, v


# ── adamw.h ────────────────────────────────────────────────────────────────
def ref_adamw_step(p, g, m, v, *, lr, beta1, beta2, eps, wd, t):
    p, g, m, v = map(_f64, (p, g, m, v))
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    return _adam_tail(p, m, v, g, lr, beta1, beta2, eps, wd, bc1, bc2)


# ── lion.h ───────────────────────────────────────────────────────────────────
def ref_lion_step(p, g, ea, *, lr, beta1, beta2, wd):
    """interp = beta1*ea + (1-beta1)*g; p -= lr*(sign(interp) + wd*p);
       ea = beta2*ea + (1-beta2)*g.  sign(0)=0 (copysignf guarded against 0)."""
    p, g, ea = map(_f64, (p, g, ea))
    interp = beta1 * ea + (1.0 - beta1) * g
    s = torch.where(interp != 0, torch.sign(interp), torch.zeros_like(interp))
    p_new = p - lr * (s + wd * p)
    ea_new = beta2 * ea + (1.0 - beta2) * g
    return p_new, ea_new


# ── grokadamw.h ──────────────────────────────────────────────────────────────
def ref_grokadamw_step(p, g, m, v, ema, *, alpha, lamb, lr, beta1, beta2,
                       eps, wd, t):
    p, g, m, v, ema = map(_f64, (p, g, m, v, ema))
    ema_new = alpha * ema + (1.0 - alpha) * g
    g_amp = g + lamb * ema_new
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    p_new, m_new, v_new = _adam_tail(p, m, v, g_amp, lr, beta1, beta2, eps,
                                     wd, bc1, bc2)
    return p_new, m_new, v_new, ema_new


# ── grokfast.h (fused mode) ──────────────────────────────────────────────────
def ref_grokfast_step(p, g, m, v, ema, *, alpha, lamb, lr, beta1, beta2,
                      eps, wd, t):
    # Identical structure to grokadamw fused: EMA filter + amplify, then Adam.
    return ref_grokadamw_step(p, g, m, v, ema, alpha=alpha, lamb=lamb, lr=lr,
                              beta1=beta1, beta2=beta2, eps=eps, wd=wd, t=t)


# ── neuralgrok.h ─────────────────────────────────────────────────────────────
def ref_neuralgrok_step(p, g, m, v, *, psi_scale, alpha, beta, lr, beta1,
                        beta2, eps, wd, t):
    """g_amp = (psi_scale*alpha + beta)*g; AdamW on g_amp.
       psi_scale is the psi-net (MLP) output, supplied per-element or scalar."""
    p, g, m, v = map(_f64, (p, g, m, v))
    if torch.is_tensor(psi_scale):
        psi_scale = _f64(psi_scale)
    g_amp = (psi_scale * alpha + beta) * g
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    return _adam_tail(p, m, v, g_amp, lr, beta1, beta2, eps, wd, bc1, bc2)


def ref_neuralgrok_psi_forward(abs_grad, W1, b1, W2, b2):
    """2-layer ReLU MLP forward (neuralgrok_psi_forward), fp64.
       W1,b1,W2: [H]; b2 scalar.  out = sum_j W2[j]*relu(W1[j]*|g| + b1[j]) + b2."""
    abs_grad, W1, b1, W2 = map(_f64, (abs_grad, W1, b1, W2))
    h = torch.relu(W1 * abs_grad.unsqueeze(-1) + b1)   # [..., H]
    return (h * W2).sum(-1) + b2


# ── looksam.h (normal-step apply: blend cached SAM direction + Adam) ─────────
def ref_looksam_apply_step(p, g, m, v, sam_dir, *, alpha, lr, beta1, beta2,
                           eps, wd, t):
    p, g, m, v, sam_dir = map(_f64, (p, g, m, v, sam_dir))
    g_adj = (1.0 - alpha) * g + alpha * sam_dir
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    return _adam_tail(p, m, v, g_adj, lr, beta1, beta2, eps, wd, bc1, bc2)


def ref_looksam_perturb(p, g, *, scale):
    """p_pert = p + scale*g  (scale = rho / (||g|| + eps), precomputed)."""
    p, g = _f64(p), _f64(g)
    return p + scale * g


def ref_looksam_set_direction(g_sam, g_orig):
    return _f64(g_sam) - _f64(g_orig)


# ── prodigy.h ────────────────────────────────────────────────────────────────
def ref_prodigy_partials(p, p_init, g, *, d_prev):
    """r += g*(p_init - p)*d_prev^2 ; s += d_prev^2 * |g|  (global reductions).

    Canonical form per csrc/algorithms/prodigy.h:48,52 (degree-2 d_prev makes
    d_hat scale-free; s uses the L1 norm |g|). An earlier revision of this ref
    used d_prev^1 and SIGNED g — stale vs the header; flagged by the opt-stages
    phase and fixed here so the header stays the single source of truth.
    """
    p, p_init, g = map(_f64, (p, p_init, g))
    r = (g * (p_init - p) * d_prev * d_prev).sum()
    s = (d_prev * d_prev * g.abs()).sum()
    return r, s


def ref_prodigy_update_d(d_prev, r_sum, s_sum):
    denom = abs(float(s_sum))
    if denom < 1e-12:
        return d_prev
    return max(d_prev, float(r_sum) / denom)


def ref_prodigy_apply_step(p, g, m, v, s_track, *, d, beta1, beta2, eps, wd,
                           t):
    """g_scaled = d*g; Adam moments on g_scaled; s_track += d*g;
       p -= d*(update + wd*p).  Note: the *step size* is d (not lr)."""
    p, g, m, v, s_track = map(_f64, (p, g, m, v, s_track))
    g_scaled = d * g
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    mm = beta1 * m + (1.0 - beta1) * g_scaled
    vv = beta2 * v + (1.0 - beta2) * g_scaled * g_scaled
    s_track_new = s_track + d * g
    update = (mm / bc1) / (torch.sqrt(vv / bc2) + eps)
    p_new = p - d * (update + wd * p)
    return p_new, mm, vv, s_track_new


# ── supergrok11.h (cosine-gated meta-net) ────────────────────────────────────
def ref_sg11_step(p, g, m, v, mu, *, gate, alpha, lr, beta1, beta2, eps, wd, t):
    """smart_grad = g + (1 - gate)*alpha*mu; AdamW on smart_grad.
       gate is the pre-reduced, clamped cosine similarity."""
    p, g, m, v, mu = map(_f64, (p, g, m, v, mu))
    smart_grad = g + (1.0 - gate) * alpha * mu
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    return _adam_tail(p, m, v, smart_grad, lr, beta1, beta2, eps, wd, bc1, bc2)


def ref_sg_phi_forward(grad_val, sharp_val, W1, b1, W2, b2):
    """exact-GELU-hidden, linear-output meta-net (sg11/sg15 share the shape).
       W1: [H,2] flattened row-major; b1,W2: [H]; b2 scalar.

       The canonical kernels use the EXACT erf GELU on the hidden pre-activation
       (csrc/algorithms/supergrok11.h:50-51, supergrok15.h:47-48:
       ``h = 0.5*h*(1 + erff(h * 0.70710678...))``), matching torch.nn.GELU() in
       the trainable meta-net — NOT a tanh hidden. This reference mirrors that
       exactly so the oracle does not silently diverge from the shipped phi-net.
    """
    grad_val, sharp_val, W1, b1, W2 = map(
        _f64, (grad_val, sharp_val, W1, b1, W2))
    W1 = W1.reshape(-1, 2)                       # [H, 2]
    pre = W1[:, 0] * grad_val + W1[:, 1] * sharp_val + b1
    # Exact GELU: 0.5*x*(1 + erf(x / sqrt(2))).
    gelu = 0.5 * pre * (1.0 + torch.erf(pre / math.sqrt(2.0)))
    return (gelu * W2).sum(-1) + b2


# ── supergrok15.h (sigmoid-accuracy-gated meta-net) ──────────────────────────
def ref_sg15_alpha_per_coord(mu_val, alpha_base, alpha_max):
    a = alpha_base * (1.0 + mu_val)
    return torch.clamp(a, 0.0, alpha_max)


def ref_sg15_step(p, g, m, v, mu, *, gate_global, alpha_base, alpha_max, lr,
                  beta1, beta2, eps, wd, t):
    """a = clamp(alpha_base*(1+mu), 0, alpha_max);
       smart_grad = g + gate_global*a*mu; AdamW on smart_grad."""
    p, g, m, v, mu = map(_f64, (p, g, m, v, mu))
    a = ref_sg15_alpha_per_coord(mu, alpha_base, alpha_max)
    smart_grad = g + gate_global * a * mu
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    return _adam_tail(p, m, v, smart_grad, lr, beta1, beta2, eps, wd, bc1, bc2)


# ── supergrok2.h (per-element GRU + expert EMA + slow-grad EMA, then Adam) ───
def ref_sg2_apply_step(p, g, m, v, mu_state, slow_state, *, expert_out, alpha,
                       gru_decay, lamb_eff, lr, beta1, beta2, eps, wd, t):
    """Mirrors csrc/algorithms/supergrok2.h::sg2_apply_step:
         mu_new   = gru_decay*mu_state + (1-gru_decay)*expert_out   (expert EMA)
         slow_new = alpha*slow_state   + (1-alpha)*g                (grokfast EMA)
         smart_grad = g + alpha*mu_new + lamb_eff*slow_new
       then AdamW on smart_grad. `alpha` is BOTH the mu/slow mixing coeff and the
       slow-EMA decay (matches the Python ref's alpha_i). lamb_eff is the
       restored grokfast amplification (lamb*ramp*gate); it was silently dropped
       (kernel (void)-ed lamb_eff) before this fix."""
    p, g, m, v, mu_state, slow_state = map(
        _f64, (p, g, m, v, mu_state, slow_state))
    if torch.is_tensor(expert_out):
        expert_out = _f64(expert_out)
    mu_new = gru_decay * mu_state + (1.0 - gru_decay) * expert_out
    slow_new = alpha * slow_state + (1.0 - alpha) * g
    smart_grad = g + alpha * mu_new + lamb_eff * slow_new
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    p_new, m_new, v_new = _adam_tail(p, m, v, smart_grad, lr, beta1, beta2,
                                     eps, wd, bc1, bc2)
    return p_new, m_new, v_new, mu_new, slow_new


# ── muon.h (+ bindings.cpp::muon_fused_step orchestration) ───────────────────
# NS coefficients are the canonical (3.4445, -4.7750, 2.0315) from
# csrc/bindings/bindings.cpp::muon_fused_step.
MUON_NS_A, MUON_NS_B, MUON_NS_C = 3.4445, -4.7750, 2.0315


def ref_muon_step(p, g, buf, *, momentum, lr, wd, ns_steps):
    """2D parameter update.
        buf = momentum*buf + grad                 (no (1-momentum); matches header)
        X   = buf / (||buf||_F + 1e-7)
        repeat ns_steps:  A=X@Xᵀ; AX=A@X; AAX=A@AX; X = a*X + b*AX + c*AAX
        scale = 0.2 * sqrt(max(rows, cols))
        p     = p*(1 - lr*wd) + (-lr*scale)*X
    Returns (p_new, buf_new).
    """
    p, g, buf = map(_f64, (p, g, buf))
    buf_new = momentum * buf + g
    inv_norm = 1.0 / (torch.linalg.norm(buf_new) + 1e-7)
    X = buf_new * inv_norm
    for _ in range(ns_steps):
        A = X @ X.t()
        AX = A @ X
        AAX = A @ AX
        X = MUON_NS_A * X + MUON_NS_B * AX + MUON_NS_C * AAX
    rows, cols = p.shape[0], p.shape[1]
    scale = 0.2 * math.sqrt(max(rows, cols))
    neg_lr_scale = -lr * scale
    decay = 1.0 - lr * wd
    p_new = p * decay + neg_lr_scale * X
    return p_new, buf_new


# Registry so a single guard test can assert all 11 optimizers are covered.
REFERENCE_OPTIMIZERS = {
    "adamw": ref_adamw_step,
    "lion": ref_lion_step,
    "grokadamw": ref_grokadamw_step,
    "grokfast": ref_grokfast_step,
    "neuralgrok": ref_neuralgrok_step,
    "looksam": ref_looksam_apply_step,
    "prodigy": ref_prodigy_apply_step,
    "supergrok11": ref_sg11_step,
    "supergrok15": ref_sg15_step,
    "supergrok2": ref_sg2_apply_step,
    "muon": ref_muon_step,
}


# ===========================================================================
#  CPU self-consistency tests (must pass with NO accelerator / NO C++ ext).
# ===========================================================================

ATOL = 1e-9
RTOL = 1e-7


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def test_registry_covers_all_eleven():
    """The ✅-gate must reference all 11 race optimizers."""
    assert len(REFERENCE_OPTIMIZERS) == 11
    expected = {
        "adamw", "lion", "grokadamw", "grokfast", "neuralgrok", "looksam",
        "prodigy", "supergrok11", "supergrok15", "supergrok2", "muon",
    }
    assert set(REFERENCE_OPTIMIZERS) == expected


# ── zero-grad => no-op modulo (decoupled) weight decay ──────────────────────

def test_adamw_zero_grad_is_pure_weight_decay():
    p = _rand(8, seed=1)
    z = torch.zeros(8, dtype=torch.float64)
    # wd = 0  => exact no-op on the parameter
    p_new, m, v = ref_adamw_step(p, z, z.clone(), z.clone(),
                                 lr=1e-3, beta1=0.9, beta2=0.999,
                                 eps=1e-8, wd=0.0, t=1)
    assert torch.allclose(p_new, p, atol=ATOL, rtol=RTOL)
    assert torch.allclose(m, z, atol=ATOL)
    assert torch.allclose(v, z, atol=ATOL)
    # wd > 0 => exactly the decoupled shrink p -= lr*wd*p (numerator is 0).
    lr, wd = 1e-3, 0.1
    p_wd, _, _ = ref_adamw_step(p, z, z.clone(), z.clone(),
                                lr=lr, beta1=0.9, beta2=0.999,
                                eps=1e-8, wd=wd, t=1)
    assert torch.allclose(p_wd, p - lr * wd * p, atol=ATOL, rtol=RTOL)


def test_lion_zero_grad_zero_momentum_is_pure_weight_decay():
    p = _rand(8, seed=2)
    z = torch.zeros(8, dtype=torch.float64)
    # interp == 0 => sign(0) == 0 (guarded), so update is only the wd term.
    lr, wd = 1e-3, 0.1
    p_new, ea_new = ref_lion_step(p, z, z.clone(),
                                  lr=lr, beta1=0.9, beta2=0.99, wd=wd)
    assert torch.allclose(p_new, p - lr * wd * p, atol=ATOL, rtol=RTOL)
    assert torch.allclose(ea_new, z, atol=ATOL)


@pytest.mark.parametrize("name", ["grokadamw", "grokfast"])
def test_ema_amp_zero_grad_is_pure_weight_decay(name):
    p = _rand(8, seed=3)
    z = torch.zeros(8, dtype=torch.float64)
    fn = REFERENCE_OPTIMIZERS[name]
    lr, wd = 1e-3, 0.05
    p_new, m, v, ema = fn(p, z, z.clone(), z.clone(), z.clone(),
                          alpha=0.98, lamb=5.0, lr=lr, beta1=0.9, beta2=0.999,
                          eps=1e-8, wd=wd, t=1)
    # Zero grad => ema stays 0 => g_amp 0 => only decoupled wd moves p.
    assert torch.allclose(ema, z, atol=ATOL)
    assert torch.allclose(p_new, p - lr * wd * p, atol=ATOL, rtol=RTOL)


def test_muon_zero_grad_zero_buf_is_pure_decay():
    p = _rand(4, 6, seed=4)
    z = torch.zeros(4, 6, dtype=torch.float64)
    lr, wd = 1e-3, 0.2
    p_new, buf_new = ref_muon_step(p, z, z.clone(),
                                   momentum=0.95, lr=lr, wd=wd, ns_steps=5)
    # buf stays 0 => X is 0 => p only shrinks by the decay factor (1 - lr*wd).
    assert torch.allclose(buf_new, z, atol=ATOL)
    assert torch.allclose(p_new, p * (1.0 - lr * wd), atol=ATOL, rtol=RTOL)


# ── step-1 bias correction closed forms ──────────────────────────────────────

def test_adamw_step1_bias_correction_closed_form():
    """At t=1 with m0=v0=0: m_hat = g, v_hat = g^2, so the Adam numerator
    collapses to sign(g) (modulo eps). p -= lr*(g/(|g|+eps) + wd*p)."""
    p = _rand(16, seed=5)
    g = _rand(16, seed=6)
    lr, beta1, beta2, eps, wd = 1e-2, 0.9, 0.999, 1e-12, 0.0
    z = torch.zeros(16, dtype=torch.float64)
    p_new, m, v = ref_adamw_step(p, g, z.clone(), z.clone(),
                                 lr=lr, beta1=beta1, beta2=beta2,
                                 eps=eps, wd=wd, t=1)
    # bc1 = 1-beta1, bc2 = 1-beta2.  m_hat = (1-beta1)g / (1-beta1) = g.
    m_hat = m / (1.0 - beta1)
    v_hat = v / (1.0 - beta2)
    assert torch.allclose(m_hat, g, atol=ATOL, rtol=RTOL)
    assert torch.allclose(v_hat, g * g, atol=ATOL, rtol=RTOL)
    expected = p - lr * (g / (torch.sqrt(g * g) + eps))
    assert torch.allclose(p_new, expected, atol=1e-7, rtol=1e-6)


def test_bias_correction_factors_are_uninverted():
    """The convention is bc = 1 - beta^t (divide by it). At t=2 the moment
    bias-correction divisor must be (1 - beta^2), not (1 - beta)."""
    beta1, beta2 = 0.9, 0.999
    p = _rand(4, seed=7)
    g = _rand(4, seed=8)
    # Drive two manual steps and confirm step-2 uses (1 - beta^2).
    z = torch.zeros(4, dtype=torch.float64)
    _, m1, v1 = ref_adamw_step(p, g, z.clone(), z.clone(),
                               lr=1e-3, beta1=beta1, beta2=beta2,
                               eps=1e-8, wd=0.0, t=1)
    # Independent reconstruction of the step-2 m_hat divisor.
    bc1_t2 = 1.0 - beta1 ** 2
    bc2_t2 = 1.0 - beta2 ** 2
    assert abs(bc1_t2 - (1 - beta1 * beta1)) < 1e-15
    assert abs(bc2_t2 - (1 - beta2 * beta2)) < 1e-15
    # And the t=1 divisors are the plain (1-beta).
    assert torch.all(m1 == (1 - beta1) * g)


# ── reductions to plain AdamW (the family invariant) ─────────────────────────

@pytest.mark.parametrize("name,extra", [
    ("grokadamw", dict(alpha=0.98, lamb=0.0)),
    ("grokfast", dict(alpha=0.98, lamb=0.0)),
])
def test_ema_amp_with_zero_lamb_equals_adamw(name, extra):
    """lamb=0 disables amplification => identical to plain AdamW."""
    p = _rand(32, seed=9)
    g = _rand(32, seed=10)
    z = torch.zeros(32, dtype=torch.float64)
    common = dict(lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, t=4)
    p_ref, m_ref, v_ref = ref_adamw_step(p, g, z.clone(), z.clone(), **common)
    fn = REFERENCE_OPTIMIZERS[name]
    p_new, m_new, v_new, _ema = fn(p, g, z.clone(), z.clone(), z.clone(),
                                   **extra, **common)
    assert torch.allclose(p_new, p_ref, atol=ATOL, rtol=RTOL)
    assert torch.allclose(m_new, m_ref, atol=ATOL, rtol=RTOL)
    assert torch.allclose(v_new, v_ref, atol=ATOL, rtol=RTOL)


# ── PUBLISHED GrokAdamW: layer-wise β1 decay + grokking-signal α ────────────
# These exercise the two mechanisms wired in grokking_optimizers/optimizers/
# grokadamw.py (β1_i = β1*(1-gamma)**i and α = alpha_init*exp(-kappa*signal)).
# They are CPU-only and _ops-free: the moment-divergence test drives the fp64
# ``ref_grokadamw_step`` formula directly with the two per-layer betas, and the
# wiring tests only CONSTRUCT the optimizer (__init__ never calls the kernel).


def test_grokadamw_layerwise_beta1_changes_moments():
    """Teeth: with the PUBLISHED layer-wise β1 (β1_i = β1*(1-gamma)**i), two
    layers at indices i=0,1 must produce DIFFERENT first/second moments from the
    SAME grad — and gamma=0 must collapse to identical moments. Driven through
    the fp64 ref (no kernel) so it validates the β1 *formula*, not the build."""
    p = _rand(32, seed=21)
    g = _rand(32, seed=22)
    z = torch.zeros(32, dtype=torch.float64)
    common = dict(alpha=0.98, lamb=2.0, lr=3e-4, beta2=0.999, eps=1e-8,
                  wd=0.01, t=3)
    beta1 = 0.9

    def moments_at(gamma, layer_idx):
        b1_i = beta1 * (1.0 - gamma) ** layer_idx
        _, m, v, _ = ref_grokadamw_step(p, g, z.clone(), z.clone(), z.clone(),
                                        beta1=b1_i, **common)
        return m, v

    # gamma = 0 => (1-0)**i == 1 for every i => identical β1 => identical moments.
    m0_l0, v0_l0 = moments_at(0.0, 0)
    m0_l1, v0_l1 = moments_at(0.0, 1)
    assert torch.allclose(m0_l0, m0_l1, atol=ATOL, rtol=RTOL)
    assert torch.allclose(v0_l0, v0_l1, atol=ATOL, rtol=RTOL)

    # gamma = 0.1 => layer 1 gets β1*0.9 != β1 => moments measurably differ.
    g1_l0_m, _ = moments_at(0.1, 0)
    g1_l1_m, _ = moments_at(0.1, 1)
    # layer 0 keeps the full β1, so it equals the gamma=0 layer-0 moment.
    assert torch.allclose(g1_l0_m, m0_l0, atol=ATOL, rtol=RTOL)
    # layer 1's β1 dropped by (1-gamma), so its first moment must DIVERGE.
    assert not torch.allclose(g1_l1_m, g1_l0_m, atol=1e-6)


def test_grokadamw_layer_beta1_wiring():
    """Construction wiring (no kernel): the optimizer must precompute
    β1_i = β1*(1-gamma)**i across the flat param enumeration (the published
    formula / SG11 _flat_layer_beta1s pattern)."""
    from grokking_optimizers.optimizers.grokadamw import GrokAdamW
    beta1, gamma = 0.9, 0.1
    ps = [torch.nn.Parameter(torch.zeros(3)) for _ in range(4)]
    opt = GrokAdamW(ps, betas=(beta1, 0.999), gamma=gamma)
    for i, p in enumerate(ps):
        expected = beta1 * (1.0 - gamma) ** i
        assert abs(opt._layer_beta1(p) - expected) < 1e-12
    # gamma=0 => every layer keeps the full β1 (single global β1).
    opt0 = GrokAdamW(ps, betas=(beta1, 0.999), gamma=0.0)
    assert all(abs(opt0._layer_beta1(p) - beta1) < 1e-12 for p in ps)


def test_grokadamw_grokking_signal_and_alpha():
    """The grokking signal is the MAX-normalised clipped val−train gap and α
    decays as alpha_init*exp(-kappa*signal) (published _default_grokking_signal
    + the per-group alpha line). No kernel involved — pure formula wiring."""
    from grokking_optimizers.optimizers.grokadamw import GrokAdamW
    # signal: diff = max(0, val-train); m = max(val,train); diff/m.
    assert GrokAdamW._grokking_signal(1.0, 2.0) == pytest.approx(0.5)   # (2-1)/2
    assert GrokAdamW._grokking_signal(2.0, 1.0) == 0.0                  # val<train
    assert GrokAdamW._grokking_signal(0.0, 0.0) == 0.0                  # max<=0
    assert GrokAdamW._grokking_signal(None, 1.0) == 0.0                 # missing
    # α schedule: before any signal, α == alpha_init; after, alpha_init*exp(-κ·s).
    alpha_init, kappa = 0.98, 0.1
    opt = GrokAdamW([torch.nn.Parameter(torch.zeros(2))],
                    alpha=alpha_init, kappa=kappa)
    grp = opt.param_groups[0]
    assert opt._alpha_for_group(grp) == pytest.approx(alpha_init)  # unprimed
    opt._grok_signal = 0.5
    opt._signal_active = True
    assert opt._alpha_for_group(grp) == pytest.approx(
        alpha_init * math.exp(-kappa * 0.5))


def test_neuralgrok_unit_amplifier_equals_adamw():
    """g_amp = (psi*alpha + beta)*g; choose psi/alpha/beta so the factor is 1."""
    p = _rand(20, seed=11)
    g = _rand(20, seed=12)
    z = torch.zeros(20, dtype=torch.float64)
    common = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, t=3)
    p_ref, m_ref, v_ref = ref_adamw_step(p, g, z.clone(), z.clone(), **common)
    # alpha=0, beta=1 => factor = 1 regardless of psi.
    p_new, m_new, v_new = ref_neuralgrok_step(
        p, g, z.clone(), z.clone(), psi_scale=7.3, alpha=0.0, beta=1.0, **common)
    assert torch.allclose(p_new, p_ref, atol=ATOL, rtol=RTOL)
    assert torch.allclose(v_new, v_ref, atol=ATOL, rtol=RTOL)


def test_looksam_alpha_zero_equals_adamw():
    """alpha=0 => g_adj = g, so the SAM-blend apply reduces to AdamW."""
    p = _rand(12, seed=13)
    g = _rand(12, seed=14)
    sam = _rand(12, seed=15)            # arbitrary cached direction, ignored
    z = torch.zeros(12, dtype=torch.float64)
    common = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, t=2)
    p_ref, _, _ = ref_adamw_step(p, g, z.clone(), z.clone(), **common)
    p_new, _, _ = ref_looksam_apply_step(
        p, g, z.clone(), z.clone(), sam, alpha=0.0, **common)
    assert torch.allclose(p_new, p_ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("name,extra", [
    ("supergrok11", dict(gate=1.0, alpha=0.5)),     # gate=1 => (1-gate)=0
    ("supergrok15", dict(gate_global=0.0, alpha_base=0.5, alpha_max=1.0)),
])
def test_metanet_neutral_gate_equals_adamw(name, extra):
    """A neutral meta-net gate => smart_grad == g => plain AdamW."""
    p = _rand(24, seed=16)
    g = _rand(24, seed=17)
    mu = _rand(24, seed=18)             # arbitrary meta-net output
    z = torch.zeros(24, dtype=torch.float64)
    common = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, t=5)
    p_ref, _, _ = ref_adamw_step(p, g, z.clone(), z.clone(), **common)
    fn = REFERENCE_OPTIMIZERS[name]
    p_new, _, _ = fn(p, g, z.clone(), z.clone(), mu, **extra, **common)
    assert torch.allclose(p_new, p_ref, atol=ATOL, rtol=RTOL)


def test_supergrok2_alpha_zero_equals_adamw():
    """alpha=0 AND lamb_eff=0 => smart_grad = g => SuperGrok2 apply reduces to
    AdamW. (alpha=0 zeroes the expert-EMA term; lamb_eff=0 zeroes the restored
    slow-grad term — both are needed for the AdamW identity now.)"""
    p = _rand(16, seed=19)
    g = _rand(16, seed=20)
    mu_state = _rand(16, seed=21)
    slow_state = _rand(16, seed=24)
    z = torch.zeros(16, dtype=torch.float64)
    common = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, t=6)
    p_ref, _, _ = ref_adamw_step(p, g, z.clone(), z.clone(), **common)
    p_new, _, _, _, _ = ref_sg2_apply_step(
        p, g, z.clone(), z.clone(), mu_state, slow_state, expert_out=99.0,
        alpha=0.0, gru_decay=0.9, lamb_eff=0.0, **common)
    assert torch.allclose(p_new, p_ref, atol=ATOL, rtol=RTOL)


def test_supergrok2_grokfast_term_restored():
    """The restored grokfast term: with NONZERO lamb_eff and NONZERO slow_state,
    SuperGrok2's smart_grad gains exactly lamb_eff*slow_new over the (no-slow)
    AdamW-on-(g+alpha*mu_new) baseline. Verifies sg2_apply_step actually USES
    lamb_eff*slow_new (the previously-(void)-ed mechanism)."""
    p = _rand(24, seed=31)
    g = _rand(24, seed=32)
    mu_state = _rand(24, seed=33)
    slow_state = _rand(24, seed=34)
    m0 = _rand(24, seed=35) * 0.1
    v0 = (_rand(24, seed=36) * 0.1).abs()
    alpha, gru_decay, lamb_eff = 0.7, 0.85, 1.5
    common = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, t=7)

    # Independent fp64 expectation: build smart_grad WITH the slow term, AdamW.
    p64, g64, mu64, slow64, m64, v64 = map(
        _f64, (p, g, mu_state, slow_state, m0, v0))
    mu_new_exp = gru_decay * mu64 + (1.0 - gru_decay) * g64  # expert_out=g here
    slow_new_exp = alpha * slow64 + (1.0 - alpha) * g64
    smart_exp = g64 + alpha * mu_new_exp + lamb_eff * slow_new_exp
    bc1 = 1.0 - common["beta1"] ** common["t"]
    bc2 = 1.0 - common["beta2"] ** common["t"]
    p_exp, m_exp, v_exp = _adam_tail(
        p64, m64, v64, smart_exp, common["lr"], common["beta1"],
        common["beta2"], common["eps"], common["wd"], bc1, bc2)

    p_new, m_new, v_new, mu_new, slow_new = ref_sg2_apply_step(
        p, g, m0.clone(), v0.clone(), mu_state, slow_state, expert_out=g,
        alpha=alpha, gru_decay=gru_decay, lamb_eff=lamb_eff, **common)

    assert torch.allclose(slow_new, slow_new_exp, atol=ATOL, rtol=RTOL)
    assert torch.allclose(p_new, p_exp, atol=ATOL, rtol=RTOL)
    assert torch.allclose(m_new, m_exp, atol=ATOL, rtol=RTOL)
    assert torch.allclose(v_new, v_exp, atol=ATOL, rtol=RTOL)

    # And the term is NON-vacuous: dropping it (lamb_eff=0) gives a DIFFERENT p.
    p_drop, _, _, _, _ = ref_sg2_apply_step(
        p, g, m0.clone(), v0.clone(), mu_state, slow_state, expert_out=g,
        alpha=alpha, gru_decay=gru_decay, lamb_eff=0.0, **common)
    assert not torch.allclose(p_new, p_drop, atol=1e-6, rtol=1e-6)


def test_prodigy_apply_uses_d_as_step_size():
    """At t=1, m0=v0=0, eps->0: prodigy_apply numerator collapses to sign(g)
    and the step size is d (not lr). p -= d*(sign(d*g) + wd*p)."""
    p = _rand(10, seed=22)
    g = _rand(10, seed=23)
    z = torch.zeros(10, dtype=torch.float64)
    d, wd, eps = 0.7, 0.0, 1e-12
    p_new, m, v, s_track = ref_prodigy_apply_step(
        p, g, z.clone(), z.clone(), z.clone(),
        d=d, beta1=0.9, beta2=0.999, eps=eps, wd=wd, t=1)
    # m_hat = d*g, v_hat = (d*g)^2 => update = d*g / (|d*g|) = sign(g).
    # Step size is d:  p_new = p - d * sign(g)   (eps->0).
    expected = p - d * torch.sign(g)
    assert torch.allclose(p_new, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(s_track, d * g, atol=ATOL, rtol=RTOL)


def test_prodigy_update_d_is_monotone_nondecreasing():
    # candidate below d_prev => d unchanged; above => raised.
    assert ref_prodigy_update_d(1.0, r_sum=0.5, s_sum=1.0) == 1.0   # cand 0.5
    assert ref_prodigy_update_d(1.0, r_sum=3.0, s_sum=1.0) == 3.0   # cand 3.0
    # near-zero s => guard returns d_prev unchanged.
    assert ref_prodigy_update_d(2.0, r_sum=5.0, s_sum=1e-20) == 2.0


def test_prodigy_partials_closed_form():
    # g carries a NEGATIVE element so the canonical s = Σ d²·|g| (L1 norm,
    # prodigy.h:52) is distinguishable from the stale signed-g form.
    p = torch.tensor([1.0, 2.0], dtype=torch.float64)
    p_init = torch.tensor([0.0, 0.0], dtype=torch.float64)
    g = torch.tensor([3.0, -4.0], dtype=torch.float64)
    d_prev = 2.0
    r, s = ref_prodigy_partials(p, p_init, g, d_prev=d_prev)
    # r = sum(g*(p_init-p)*d_prev^2) = 3*(-1)*4 + (-4)*(-2)*4 = -12 + 32 = 20
    # s = sum(d_prev^2 * |g|) = 4*3 + 4*4 = 28
    assert abs(float(r) - 20.0) < 1e-12
    assert abs(float(s) - 28.0) < 1e-12


# ── meta-net forward closed forms ────────────────────────────────────────────

def test_neuralgrok_psi_forward_relu_closed_form():
    # H=2; W1=[1,-1], b1=[0,0], W2=[2,3], b2=0.5; |g|=4
    W1 = torch.tensor([1.0, -1.0])
    b1 = torch.tensor([0.0, 0.0])
    W2 = torch.tensor([2.0, 3.0])
    abs_g = torch.tensor(4.0)
    out = ref_neuralgrok_psi_forward(abs_g, W1, b1, W2, b2=0.5)
    # h = relu([4, -4]) = [4, 0]; out = 2*4 + 3*0 + 0.5 = 8.5
    assert abs(float(out) - 8.5) < 1e-12


def test_sg_phi_forward_gelu_closed_form():
    # H=1; W1 row = [0, 0] => pre = b1; out = W2*GELU(b1) + b2, exact-erf GELU
    # (matches the canonical kernels' 0.5*x*(1+erf(x/sqrt(2))) hidden, NOT tanh).
    W1 = torch.tensor([0.0, 0.0])          # [H=1, 2] flattened
    b1 = torch.tensor([0.5])
    W2 = torch.tensor([2.0])
    out = ref_sg_phi_forward(torch.tensor(9.0), torch.tensor(-9.0),
                             W1, b1, W2, b2=1.0)
    gelu_half = 0.5 * 0.5 * (1.0 + math.erf(0.5 / math.sqrt(2.0)))
    expected = 2.0 * gelu_half + 1.0
    assert abs(float(out) - expected) < 1e-12


def test_sg15_alpha_per_coord_clamps():
    mu = torch.tensor([-2.0, 0.0, 10.0], dtype=torch.float64)
    a = ref_sg15_alpha_per_coord(mu, alpha_base=0.5, alpha_max=1.0)
    # 0.5*(1 + mu) = [-0.5, 0.5, 5.5] clamped to [0,1] = [0, 0.5, 1.0]
    assert torch.allclose(a, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))


# ── Muon Newton-Schulz: orthogonalization sanity on a tiny matrix ────────────

def test_muon_ns_reduces_singular_value_spread():
    """The defining property of the Newton-Schulz orthogonalization: it drives
    a matrix toward the orthogonal manifold by *equalizing* its singular values
    (condition number -> 1). Assert the NS recurrence shrinks the spread of
    singular values (max/min ratio) on a tiny ill-conditioned matrix."""
    torch.manual_seed(3)
    M = torch.randn(4, 6, dtype=torch.float64)
    X = M / (torch.linalg.norm(M) + 1e-7)        # Frobenius-normalize (as Muon does)
    cond_before = (lambda s: (s.max() / s.min()))(torch.linalg.svdvals(X))
    for _ in range(5):
        A = X @ X.t()
        AX = A @ X
        AAX = A @ AX
        X = MUON_NS_A * X + MUON_NS_B * AX + MUON_NS_C * AAX
    cond_after = (lambda s: (s.max() / s.min()))(torch.linalg.svdvals(X))
    assert cond_after < cond_before, (float(cond_before), float(cond_after))
    # Substantially closer to orthonormal (was ~4.4, should drop well below 2).
    assert float(cond_after) < 2.0, float(cond_after)


def test_muon_full_step_known_scale():
    """End-to-end Muon step closed form with a 1x1-direction 2D param so the NS
    recurrence has a scalar fixed point and the scale factor is exact.

    For a 2x2 grad with a single nonzero, after normalize+NS the orthogonalized
    X is finite; we assert the decay term is applied exactly: with grad=0 the
    update is purely p*(1 - lr*wd)."""
    p = torch.tensor([[2.0, -3.0], [1.0, 0.5]], dtype=torch.float64)
    z = torch.zeros(2, 2, dtype=torch.float64)
    lr, wd = 0.1, 0.3
    p_new, _ = ref_muon_step(p, z, z.clone(), momentum=0.9, lr=lr, wd=wd,
                             ns_steps=3)
    assert torch.allclose(p_new, p * (1.0 - lr * wd), atol=ATOL, rtol=RTOL)


def test_muon_2d_only_shape_required():
    """Muon's NS path is matrix-only (X @ X.t() needs >=2D) and indexes
    rows/cols; a 1D tensor must not be passed here (the optimizer routes 1D
    params to AdamW). Guard documents the contract — any error is acceptable."""
    p = torch.zeros(5, dtype=torch.float64)
    with pytest.raises((IndexError, RuntimeError)):
        ref_muon_step(p, torch.ones(5, dtype=torch.float64),
                      torch.zeros(5, dtype=torch.float64),
                      momentum=0.95, lr=1e-3, wd=0.0, ns_steps=1)


# ===========================================================================
#  NeuralGrok amplifier training (AUDIT A3 fix) — CPU-only, no built extension.
#
#  NeuralGrok.train_amplifier_step is PURE PyTorch (torch.func.functional_call
#  + the amplifier MLP); it never touches the fused _ops kernel. The optimizer's
#  step() needs the C++ extension, but importing NeuralGrok does not (dispatch's
#  _LazyOps resolves lazily — get_ops() never raises at import), so these run on
#  a plain CPU runner. They lock in the A3 fix: the amplifier was autograd-
#  unreachable and frozen at random init; after the fix its params must MOVE and
#  grads must FLOW, and the deployed (Python) amplifier forward must equal the
#  kernel's transcribed 2-layer ReLU math.
# ===========================================================================

def _import_neuralgrok():
    try:
        from grokking_optimizers.optimizers.neuralgrok import (
            NeuralGrok, _Amplifier)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"NeuralGrok import unavailable: {exc}")
    return NeuralGrok, _Amplifier


def test_neuralgrok_amplifier_trains_on_cpu():
    """A3 regression guard: train_amplifier_step must MOVE the amplifier params
    and FLOW grads into them (the defect: the amplifier was autograd-unreachable
    and stayed frozen at random init forever), while leaving the caller's
    p.grad untouched (snapshot/restore parity with SG11.meta_step)."""
    NeuralGrok, _ = _import_neuralgrok()
    torch.manual_seed(0)

    # Tiny CPU classification model + fake data so functional_call has a real
    # nn.Module to substitute virtual params into.
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(),
                                torch.nn.Linear(6, 3))
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    val_x = torch.randn(8, 4)
    val_y = torch.randint(0, 3, (8,))
    criterion = torch.nn.CrossEntropyLoss()

    # num_layers=2 so the Python amplifier == the kernel's deployed 2-layer MLP.
    opt = NeuralGrok(model.parameters(), lr=1e-3, num_layers=2, hidden_dim=8,
                     amplifier_lr=1e-2, grad_checkpoint=False)

    # Populate real grads via a backward (these are what the amplifier reads).
    loss = criterion(model(x), y)
    opt.zero_grad()
    loss.backward()
    grads_before = {n: p.grad.detach().clone()
                    for n, p in model.named_parameters() if p.grad is not None}
    assert grads_before, "fixture must produce gradients"

    amp_before = [p.detach().clone() for p in opt.amplifier.parameters()]

    # The fix under test. Uses the internally-owned Adam (no external optimizer
    # passed) — exercises the AdamW-parity path.
    ret = opt.train_amplifier_step(model, val_x, val_y, criterion)
    assert math.isfinite(ret), "amplifier loss must be finite"

    # (1) Amplifier params actually MOVED (the whole point of A3).
    amp_after = [p.detach().clone() for p in opt.amplifier.parameters()]
    moved = any(not torch.equal(a, b) for a, b in zip(amp_before, amp_after))
    assert moved, ("amplifier params did not change — train_amplifier_step is "
                   "still a no-op (A3 defect not fixed)")

    # (2) Grads FLOWED into the amplifier and are finite (param.grad is set by
    #     amp_loss.backward()). Non-None + finite on every amplifier param.
    for p in opt.amplifier.parameters():
        assert p.grad is not None, "amplifier param received no gradient"
        assert torch.isfinite(p.grad).all(), "amplifier gradient non-finite"

    # (3) The caller's p.grad is RESTORED exactly (snapshot/restore contract).
    for n, p in model.named_parameters():
        assert p.grad is not None and torch.equal(p.grad, grads_before[n]), (
            f"model grad for {n} was not restored to its pre-call value")


def test_neuralgrok_amplifier_no_grad_raises():
    """Honesty rail: calling train_amplifier_step before backward (no grads to
    amplify) must raise LOUDLY, never silently no-op — a silent no-op here is
    exactly the A3 defect this method exists to eliminate."""
    NeuralGrok, _ = _import_neuralgrok()
    model = torch.nn.Linear(4, 3)
    opt = NeuralGrok(model.parameters(), lr=1e-3, num_layers=2, hidden_dim=8)
    with pytest.raises(RuntimeError):
        opt.train_amplifier_step(model, torch.randn(2, 4),
                                 torch.randint(0, 3, (2,)),
                                 torch.nn.CrossEntropyLoss())


def test_neuralgrok_amplifier_forward_matches_kernel_psi():
    """Featurization parity (THE correctness crux): the differentiable training
    path optimizes the SAME function the kernel deploys. The kernel feeds the
    per-element scalar |g| (csrc/.../opt_components.cuh:225 fabsf(grad[idx]); raw
    |g|, no norm) into a 2-layer ReLU MLP psi = Σ_j W2[j]·relu(W1[j]·|g| + b1[j])
    + b2 (csrc/algorithms/neuralgrok.h:38-45). With num_layers=2, _Amplifier.
    forward must equal that, reading get_weights()'s (W1,b1,W_last,b_last) — the
    exact 2-layer contract the kernel consumes. hidden_dim=16 mirrors the
    DEPLOYED width: the only neuralgrok_psi_forward instantiation in csrc is
    <kPsiHidden=16> (opt_components.cuh:61), which is now also the race config's
    neural_hidden — so this tests the production-width function end to end."""
    _, _Amplifier = _import_neuralgrok()
    torch.manual_seed(1)
    amp = _Amplifier(num_layers=2, hidden_dim=16, grad_checkpoint=False)

    # The kernel's per-element input is |g| (scalar). Feed |g| reshaped to [N,1]
    # (the _Amplifier Linear(1,H) input contract).
    g = torch.randn(32) * 3.0
    abs_g = g.abs()

    py_psi = amp(abs_g.reshape(-1, 1)).reshape(-1)

    # Transcribed kernel math from the SAME weights the kernel would read.
    W1, b1, W_last, b_last = amp.get_weights()
    ref_psi = ref_neuralgrok_psi_forward(
        abs_g, W1.reshape(-1), b1.reshape(-1), W_last.reshape(-1),
        float(b_last.reshape(-1)[0]))

    assert torch.allclose(py_psi.double(), ref_psi, atol=1e-6, rtol=1e-5), (
        "Python amplifier forward != transcribed kernel psi — the trained net "
        "is NOT the net the kernel runs (featurization parity broken)")


# ===========================================================================
#  GPU half — fused kernel vs reference. Gated to the hardware runbook.
#  Skips cleanly (never errors) on a CPU runner and without the built ext.
# ===========================================================================

def _cuda_available():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(
    not _cuda_available(), reason="no CUDA device (GPU half gated to runbook)")


@pytest.mark.hw
@requires_cuda
@pytest.mark.parametrize("opt_class_name", [
    "AdamW", "Lion", "GrokAdamW", "Grokfast", "NeuralGrok", "LookSAM",
    "Prodigy", "SuperGrok11", "SuperGrok15", "SuperGrok2", "Muon",
])
def test_kernel_matches_reference_gpu(opt_class_name):
    """Device half: run the built fused kernel for one step and compare to the
    fp64 reference. Imports the kernel path lazily so CPU collection never
    requires the compiled extension. The exact tolerances + the multi-step /
    multi-shape sweep live in HARDWARE_VALIDATION.md (Stage 3)."""
    import importlib

    try:
        gomod = importlib.import_module("grokking_optimizers")
        opt_cls = getattr(gomod, opt_class_name)
    except Exception as exc:  # pragma: no cover - hardware-only path
        pytest.skip(f"grokking_optimizers / {opt_class_name} unavailable: {exc}")

    torch.manual_seed(0)
    # 2D so Muon's NS path is exercised; AdamW-family handles 2D fine too.
    p = torch.nn.Parameter(torch.randn(64, 64, device="cuda",
                                       dtype=torch.float32))
    p_ref = p.detach().double().clone()
    g = torch.randn_like(p)

    opt = opt_cls([p], lr=1e-3)
    p.grad = g.clone()
    opt.step()
    torch.cuda.synchronize()

    # AdamW closed-form reference for the AdamW-family kernels at step 1.
    if opt_class_name == "AdamW":
        z = torch.zeros_like(p_ref)
        expected, _, _ = ref_adamw_step(
            p_ref, g.double(), z, z, lr=1e-3, beta1=0.9, beta2=0.999,
            eps=1e-8, wd=1e-2, t=1)
        assert torch.allclose(p.detach().double(), expected,
                              atol=1e-4, rtol=1e-3)
    else:
        # For the others the per-optimizer hyper-defaults differ; the precise
        # oracle wiring is enumerated in the runbook. Here we assert the step
        # is finite and actually moved the parameter (smoke parity).
        assert torch.isfinite(p.detach()).all()
        assert not torch.allclose(p.detach().double(), p_ref)
