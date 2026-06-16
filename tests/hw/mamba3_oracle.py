"""tests/hw/mamba3_oracle.py — fp64 ORACLE for the Mamba-3 (SISO) block, with a
VERIFIED hand-derived (NON-autograd) backward.

WHY THIS EXISTS (and how it now matches mamba_oracle.py's rigor):
  This is the *Mamba-3* REVERSIBLE-FOUNDATION oracle and the line-for-line
  reference the CUDA megakernel will transcribe. Like the Mamba-1 oracle
  (mamba_oracle.py), it ships a hand-derived MANUAL backward — the reverse-time
  recurrence for the complex exponential-trapezoidal selective scan (Eq 25) plus
  all the projections — and asserts it matches torch.autograd to ~1e-12 in fp64
  for the loss grad w.r.t. EVERY parameter AND the input. Converting the
  un-runnable question "is my hand-derived complex-trapezoidal scan backward
  correct?" into the runnable "does my plain-PyTorch manual backward match
  autograd?" is the whole point: a mismatch is a real bug the kernel would
  inherit, so we FIX the derivation, never loosen the tolerance.

  It ALSO retains the original foundation properties:
    (A) FORWARD is FINITE and DETERMINISTIC (bit-identical across two runs).
    (B) The model is FULLY DIFFERENTIABLE: EVERY parameter receives a FINITE
        gradient via torch.autograd.
    (C) The fp64 run is NUMERICALLY STABLE (loss + all grads finite in float64).
    (D) The float32 forward MATCHES the float64 forward within a sane tolerance.
    (E) Param count is printed (toy gate config + the 1.5B canonical config:
        now the FULL Llama-style mixer+SwiGLU model, reaching ~1.5B at the
        paper's d=2048/nl=24/MLP-inner-4096 with the Llama-3.1 vocab + tied
        embeddings).
    (F) MANUAL backward == autograd to <=1e-10 (target ~1e-12) rel, per param
        (mixer AND SwiGLU MLP AND the two per-block pre-norms) + the input, at
        two configs/seeds + the toy config — the headline gate.

ARCHITECTURE UNDER TEST (grokking_optimizers/mamba3_block.py):
  FULL Llama-style Mamba-3 model (Section 3.4: "alternating Mamba-3 and SwiGLU
  blocks with pre-norm"). Each Mamba3Block is one Llama layer:
        h = h + Mamba3Mixer( RMSNorm_mix(h) )      # mixer residual=False
        h = h + SwiGLU_MLP ( RMSNorm_mlp(h) )      # down(SiLU(gate)*up), no bias
  ``nl`` counts the MIXER blocks; each is paired with a SwiGLU MLP (nl of each).
  Mamba3Model(p=97, ntok=99, seq_len=8, d=128, nl=2)   — toy grokking-gate config
    tok = Embedding(99,128); pos = Embedding(8,128)
    nl x Mamba3Block( mixer=Mamba3Layer(d=128, state=128, head_dim=64, exp=2),
                      SwiGLU_MLP(d=128, d_ff=256) )  (n_heads=4)
    final RMSNorm(128); out = Linear(128, 97)
    forward: out(norm(stack)[:, -1, :])   (LAST token, head width p=97)
  Loss: F.cross_entropy(logits, targets)  (mean over batch).
  The manual backward covers BOTH sub-blocks + the two per-block pre-norms +
  the block residuals (block_forward/block_backward, swiglu_forward/backward).

  Mamba3Layer (mixer) forward (see mamba3_block.py docstring for exact equations
  and MAMBA3_REFERENCE.md for the per-head/head-shared/per-channel split):
    NO conv1d, NO SiLU on the SSM input (both dropped — Sec 3.4 obviates the
    short conv "and its accompanying activation function").
    Complex state materialized as 2x2 rotation blocks on a real state of size
    state_dim (= 2*complex_dim). Exponential-trapezoidal 3-term recurrence
      h_t = alpha_t R_t h_{t-1} + beta_t R_t (Bbar_{t-1} x_{t-1}) + gamma_t (Bbar_t x_t)
      y_t = Cbar_t^T h_t       (Prop 1 Eq 5-6; Prop 2/4 Eq 25).
    PER HEAD: dt, A_real, lambda (the head_dim channels in a head share them).
    HEAD-SHARED (all heads): theta, B, Bhat, C, Chat (multi-value attention).
    PER CHANNEL: x (SSM input), D skip, gate z. Rotation angle phi = dt*theta is
    per-head per-coord [B,L,n_heads,complex_dim] (EXACT, no mean approximation).
    BCNorm (RMSNorm) + all-ones B,C biases (Appendix F). Gated output Y(.)SiLU(Z).

This file imports torch only; it needs NO GPU and NO compiled extension.
Run:  cd <repo> && source /workspace/venv/bin/activate && export PYTHONPATH="$PWD"
      python tests/hw/mamba3_oracle.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch

# make the repo root importable when run as a script
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grokking_optimizers.mamba3_block import (  # noqa: E402
    Mamba3Block,
    Mamba3Layer,
    Mamba3Model,
)


# ---------------------------------------------------------------------------
# Toy "grokking gate" config (mirrors MambaModel defaults: tiny d, seq_len=8).
# ---------------------------------------------------------------------------
TOY = dict(p=97, ntok=99, seq_len=8, d=128, nl=2, state_dim=128, head_dim=64, expand_factor=2)
# Canonical 1.5B-scale Mamba-3 SISO config (Appendix D / Table 12: d_model=2048,
# state=128, head_dim=64, expand=2, 24 layers; SISO MLP inner = 4096 = 2*d,
# Appendix C Table). The model is now the FULL Llama-style architecture
# (alternating Mamba-3 mixer + SwiGLU blocks, Sec 3.4): nl=24 mixers + 24 SwiGLU
# MLPs. With this small grokking vocab (ntok=99/p=97) the layer stack alone is
# ~1.27B; the paper's ~1.5B is reached with the real Llama-3.1 vocab (128256)
# + tied embeddings (see MAMBA3_REFERENCE.md §3 / §7), which the BIG_LLAMA_VOCAB
# config below reports explicitly. mlp_ratio=2 -> d_ff=4096 at d=2048.
BIG = dict(p=97, ntok=99, seq_len=8, d=2048, nl=24, state_dim=128, head_dim=64,
           expand_factor=2, mlp_ratio=2)
# Same canonical config but with the paper's Llama-3.1 tokenizer vocab so the
# headline ~1.5B number is reproduced (tied embeddings: head shares the embed,
# so we count tok embedding once and subtract the tiny grokking head; the real
# model ties out.weight to tok.weight). p here is set to the vocab too.
BIG_LLAMA_VOCAB = dict(p=128256, ntok=128256, seq_len=8, d=2048, nl=24,
                       state_dim=128, head_dim=64, expand_factor=2, mlp_ratio=2)

# Small configs used for the manual-vs-autograd gate (cheap, distinct shapes).
GATE_A = dict(p=11, ntok=13, seq_len=6, d=16, nl=2, state_dim=8, head_dim=4, expand_factor=2)
GATE_B = dict(p=7, ntok=9, seq_len=5, d=24, nl=3, state_dim=12, head_dim=6, expand_factor=2)

BATCH = 4
ATOL_F32 = 2e-2   # fp32-vs-fp64 forward tolerance (single-precision recurrence)
RTOL_F32 = 2e-2
MANUAL_RTOL = 1e-10   # manual-vs-autograd gate (target ~1e-12)


def _build(cfg, dtype, seed=0):
    torch.manual_seed(seed)
    model = Mamba3Model(**cfg).to(dtype)
    return model


def _make_batch(cfg, seed=1234):
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, cfg["ntok"], (BATCH, cfg["seq_len"]), generator=g)
    tgts = torch.randint(0, cfg["p"], (BATCH,), generator=g)
    return toks, tgts


def _loss(model, toks, tgts):
    logits = model(toks)
    return torch.nn.functional.cross_entropy(logits, tgts), logits


def param_count(cfg) -> int:
    m = Mamba3Model(**cfg)
    return sum(p.numel() for p in m.parameters())


def param_count_breakdown(cfg):
    """Return (mixer, swiglu_mlp, other) param totals. 'mixer' = all
    layers.*.mixer.* + layers.*.mixer_norm; 'swiglu_mlp' = layers.*.mlp.* +
    layers.*.mlp_norm; 'other' = embeddings + final norm + head."""
    m = Mamba3Model(**cfg)
    mixer = mlp = other = 0
    for name, p in m.named_parameters():
        n = p.numel()
        if ".mixer." in name or name.endswith(".mixer_norm.weight"):
            mixer += n
        elif ".mlp." in name or name.endswith(".mlp_norm.weight"):
            mlp += n
        else:
            other += n
    return mixer, mlp, other


# ===========================================================================
#  ELEMENTWISE PRIMITIVES + DERIVATIVES (mirror torch.nn.functional exactly).
# ===========================================================================
def silu(x):
    return x * torch.sigmoid(x)


def silu_grad(x):
    # d/dx [x*sig(x)] = sig(x) * (1 + x*(1-sig(x))).
    s = torch.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def softplus_grad(x):
    # d/dx log(1+exp(x)) = sigmoid(x).
    return torch.sigmoid(x)


# ===========================================================================
#  RMSNorm fwd/bwd over the last dim (size D). NO fp32 downcast (matches the
#  block's RMSNorm after the precision fix): y = x * rsqrt(mean(x^2)+eps) * w.
#    r = rsqrt(mean(x^2)+eps)  (per row), xhat = x*r, y = xhat*w.
#  Backward (dy upstream):
#    dw   = sum_rows (dy * xhat)
#    dxhat= dy * w
#    dx   = r * (dxhat - xhat * (sum_lastdim(dxhat*xhat) / D))
# ===========================================================================
def rmsnorm_forward(x, w, eps=1e-5):
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    r = torch.rsqrt(ms + eps)            # [..., 1]
    xhat = x * r
    y = xhat * w
    return y, (xhat, r, w)


def rmsnorm_backward(dy, cache):
    xhat, r, w = cache
    D = xhat.shape[-1]
    reduce_axes = tuple(range(dy.dim() - 1))
    dw = (dy * xhat).sum(dim=reduce_axes)
    dxhat = dy * w
    corr = (dxhat * xhat).sum(dim=-1, keepdim=True) / D
    dx = r * (dxhat - xhat * corr)
    return dx, dw


# ===========================================================================
#  Linear fwd/bwd: y = x @ W^T (+ b). W [out,in]. x [...,in].
# ===========================================================================
def linear_forward(x, W, b=None):
    y = x @ W.t()
    if b is not None:
        y = y + b
    return y


def linear_backward(dy, x, W, has_bias):
    in_dim = W.shape[1]
    out_dim = W.shape[0]
    x2 = x.reshape(-1, in_dim)
    dy2 = dy.reshape(-1, out_dim)
    dW = dy2.t() @ x2                                # [out,in]
    db = dy2.sum(0) if has_bias else None
    dx = (dy @ W).reshape(x.shape)
    return dx, dW, db


# ===========================================================================
#  Cross-entropy (mean over batch) fwd/bwd on last-token logits.
#  dlogits = (softmax - onehot)/B.
# ===========================================================================
def cross_entropy_forward(logits, targets):
    m = logits.max(dim=-1, keepdim=True).values
    e = torch.exp(logits - m)
    logz = m.squeeze(-1) + torch.log(e.sum(dim=-1))
    nll = logz - logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    return nll.mean()


def cross_entropy_backward(logits, targets):
    B = logits.shape[0]
    m = logits.max(dim=-1, keepdim=True).values
    e = torch.exp(logits - m)
    sm = e / e.sum(dim=-1, keepdim=True)
    onehot = torch.zeros_like(sm)
    onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    return (sm - onehot) / B


# ===========================================================================
#  COMPLEX EXPONENTIAL-TRAPEZOIDAL SELECTIVE SCAN — forward + MANUAL backward.
#
#  This is the heart of the kernel transcription. Per (batch b, head h, channel
#  p within head, complex coord c), the state is a 2-vector h_t[b,h,p,c,:].
#  Coefficients are PER HEAD scalars; theta/B/C are head-shared; x is per
#  channel. (See MAMBA3_REFERENCE.md sec 1.)
#
#  FORWARD (Eq 25), with per-head scalars
#      alpha_t[h] = exp(dt[t,h]*A[t,h]),  beta_t[h] = (1-lam)*dt*alpha,
#      gamma_t[h] = lam*dt,  phi_t[h,c] = dt[t,h]*theta[t,c],
#      R_t[h,c] = [[cos,-sin],[sin,cos]],  v_t = Bbar_t (.) x_t :
#
#      h_t = alpha_t * (R_t @ h_{t-1}) + beta_t * (R_t @ v_{t-1}) + gamma_t * v_t
#      y_t[b,h,p] = sum_{c,d} Cbar[t,c,d] * h_t[b,h,p,c,d]              (h_{-1}=0, v_{-1}=0)
#
#  with R(phi) @ (w0,w1) = (cos*w0 - sin*w1,  sin*w0 + cos*w1).
#
#  BACKWARD (reverse time t = L-1..0). Carry TWO adjoints:
#    gh[b,h,p,c,:] = dL/dh_t   (flows back through h_{t+1}=alpha R h_t + ...)
#    gv[b,h,p,c,:] = dL/dv_t   accumulated from step t+1's beta-term
#                              beta_{t+1} R_{t+1} v_t  (the WIDTH-2 coupling).
#  At step t:
#   1) y_t -> gh += Cbar_t * dy_t ;  dCbar[t] += sum_{b,h,p} h_t * dy_t .
#      Now gh = full dL/dh_t.
#   2) v_t's TOTAL grad: dv_t = gv + gamma_t * gh   (gv from t+1 beta; gamma at t).
#      v_t = Bbar_t (.) x_t  ->
#        dx[t,h,p] += sum_{c,d} dv_t[...,c,d] * Bbar[t,c,d]
#        dBbar[t,c,d] += sum_{b,h,p} dv_t[...,c,d] * x[t,h,p]
#   3) Rotated terms (need Rh = R_t @ h_{t-1}, Rv = R_t @ v_{t-1}):
#        dalpha[b,t,h] += sum_{p,c,d} Rh * gh
#        dbeta [b,t,h] += sum_{p,c,d} Rv * gh
#        dgamma[b,t,h] += sum_{p,c,d} v_t * gh
#      phi grad via R'(phi) [[-sin,-cos],[cos,-sin]] applied to h_{t-1} and v_{t-1}:
#        for a term coef*R@w with adjoint g:
#          dphi += coef*( g0*(-sin*w0 - cos*w1) + g1*(cos*w0 - sin*w1) )
#        dphi[b,t,h,c] += sum_p [ alpha*term(h_{t-1}) + beta*term(v_{t-1}) ]
#   4) Propagate to the previous step (R orthogonal -> adjoint uses R^T):
#        gv_next(=dL/dv_{t-1}) = beta_t * (R_t^T @ gh)     (carried to step t-1)
#        gh_next(=dL/dh_{t-1}) = alpha_t * (R_t^T @ gh)
#  After t=0 both carries multiply into h_{-1}=v_{-1}=0 and are discarded.
#
#  Then fold the coefficient Jacobians (all per head):
#    dt(post-softplus): ddt = dalpha*(A*alpha) + dbeta*((1-lam)*alpha*(1+dt*A))
#                            + dgamma*lam + sum_c dphi[...,c]*theta[t,c]
#    dlam            : dlam = dbeta*(-dt*alpha) + dgamma*dt
#    dA_real         : dA_real = dalpha*(dt*alpha) + dbeta*((1-lam)*dt^2*alpha)
#    dtheta (shared) : dtheta[t,c] = sum_h dphi[t,h,c]*dt[t,h]
#  R'(phi): d/dphi [cos,-sin;sin,cos] = [-sin,-cos;cos,-sin].  alpha=exp(dt*A):
#  dalpha/d(dt)=A*alpha, dalpha/dA=dt*alpha. beta=(1-lam)*dt*alpha. gamma=lam*dt.
# ===========================================================================
def scan_forward(x_h, dt, A_real, theta, lam, Bbar, Cbar, H, P, Nc):
    """x_h: [B,L,H,P] SSM input (no SiLU). dt,A_real,lam: [B,L,H] (per head).
    theta: [B,L,Nc] (head-shared). Bbar,Cbar: [B,L,Nc,2]. Returns (y, cache).
    y: [B,L,H,P]."""
    B, L, _, _ = x_h.shape
    dev, dt_t = x_h.device, x_h.dtype
    alpha = torch.exp(dt * A_real)              # [B,L,H]
    gamma = lam * dt                            # [B,L,H]
    beta = (1.0 - lam) * dt * alpha             # [B,L,H]
    phi = dt.unsqueeze(-1) * theta.unsqueeze(-2)  # [B,L,H,Nc]
    cos = torch.cos(phi)                        # [B,L,H,Nc]
    sin = torch.sin(phi)

    h = torch.zeros(B, H, P, Nc, 2, dtype=dt_t, device=dev)
    v_prev = torch.zeros(B, H, P, Nc, 2, dtype=dt_t, device=dev)
    ys, h_hist, v_hist = [], [], []
    for t in range(L):
        x_t = x_h[:, t].view(B, H, P, 1, 1)            # [B,H,P,1,1]
        Bbar_t = Bbar[:, t].view(B, 1, 1, Nc, 2)       # [B,1,1,Nc,2]
        Cbar_t = Cbar[:, t].view(B, 1, 1, Nc, 2)
        v_t = Bbar_t * x_t                             # [B,H,P,Nc,2]
        cos_t = cos[:, t].view(B, H, 1, Nc)            # [B,H,1,Nc]
        sin_t = sin[:, t].view(B, H, 1, Nc)
        a_t = alpha[:, t].view(B, H, 1, 1, 1)
        b_t = beta[:, t].view(B, H, 1, 1, 1)
        g_t = gamma[:, t].view(B, H, 1, 1, 1)
        # rotate h_{t-1} and v_{t-1}
        hr, hi = h[..., 0], h[..., 1]
        Rh = torch.stack((cos_t * hr - sin_t * hi, sin_t * hr + cos_t * hi), -1)
        vr, vi = v_prev[..., 0], v_prev[..., 1]
        Rv = torch.stack((cos_t * vr - sin_t * vi, sin_t * vr + cos_t * vi), -1)
        h = a_t * Rh + b_t * Rv + g_t * v_t            # [B,H,P,Nc,2]
        h_hist.append(h)
        v_hist.append(v_t)
        y_t = (Cbar_t * h).sum(dim=(3, 4))             # [B,H,P]
        ys.append(y_t)
        v_prev = v_t
    y = torch.stack(ys, dim=1)                         # [B,L,H,P]
    cache = dict(x_h=x_h, dt=dt, A_real=A_real, theta=theta, lam=lam,
                 Bbar=Bbar, Cbar=Cbar, alpha=alpha, beta=beta, gamma=gamma,
                 cos=cos, sin=sin, h_hist=h_hist, v_hist=v_hist, H=H, P=P, Nc=Nc)
    return y, cache


def scan_backward(dy, cache):
    """dy: [B,L,H,P] = dL/dy. Returns dict of grads w.r.t. the scan INPUTS:
    dx_h [B,L,H,P], ddt/dA_real/dlam [B,L,H], dtheta [B,L,Nc],
    dBbar/dCbar [B,L,Nc,2]. (ddt is grad w.r.t. POST-softplus dt; dlam w.r.t.
    lambda=sigmoid(u); dA_real w.r.t. the post-softplus-negated real part.)"""
    x_h, dt = cache["x_h"], cache["dt"]
    A_real, theta, lam = cache["A_real"], cache["theta"], cache["lam"]
    Bbar, Cbar = cache["Bbar"], cache["Cbar"]
    alpha, beta, gamma = cache["alpha"], cache["beta"], cache["gamma"]
    cos, sin = cache["cos"], cache["sin"]
    h_hist, v_hist = cache["h_hist"], cache["v_hist"]
    H, P, Nc = cache["H"], cache["P"], cache["Nc"]
    B, L, _, _ = x_h.shape
    dev, dtp = x_h.device, x_h.dtype

    dx_h = torch.zeros_like(x_h)
    dalpha = torch.zeros_like(alpha)
    dbeta = torch.zeros_like(beta)
    dgamma = torch.zeros_like(gamma)
    dphi = torch.zeros(B, L, H, Nc, dtype=dtp, device=dev)
    dBbar = torch.zeros_like(Bbar)
    dCbar = torch.zeros_like(Cbar)

    gh = torch.zeros(B, H, P, Nc, 2, dtype=dtp, device=dev)   # dL/dh_t
    gv = torch.zeros(B, H, P, Nc, 2, dtype=dtp, device=dev)   # dL/dv_t (from t+1 beta)

    for t in reversed(range(L)):
        h_t = h_hist[t]
        v_t = v_hist[t]
        h_tm1 = h_hist[t - 1] if t > 0 else torch.zeros_like(h_t)
        v_tm1 = v_hist[t - 1] if t > 0 else torch.zeros_like(v_t)
        Cbar_t = Cbar[:, t].view(B, 1, 1, Nc, 2)
        Bbar_t = Bbar[:, t].view(B, 1, 1, Nc, 2)
        cos_t = cos[:, t].view(B, H, 1, Nc)
        sin_t = sin[:, t].view(B, H, 1, Nc)
        a_t = alpha[:, t].view(B, H, 1, 1, 1)
        b_t = beta[:, t].view(B, H, 1, 1, 1)
        g_t = gamma[:, t].view(B, H, 1, 1, 1)
        dy_t = dy[:, t].view(B, H, P, 1, 1)              # [B,H,P,1,1]

        # (1) y_t = sum_{c,d} Cbar_t * h_t  ->  gh, dCbar
        gh = gh + Cbar_t * dy_t                          # [B,H,P,Nc,2]
        dCbar[:, t] = (h_t * dy_t).sum(dim=(1, 2))       # sum over H,P (shared)

        # (2) v_t total grad = gv (from t+1 beta) + gamma_t * gh (gamma at t)
        dv_t = gv + g_t * gh                             # [B,H,P,Nc,2]
        # v_t = Bbar_t (.) x_t : dx (sum c,d), dBbar (sum b,h,p)
        x_t = x_h[:, t].view(B, H, P, 1, 1)
        dx_h[:, t] = (dv_t * Bbar_t).sum(dim=(3, 4))     # [B,H,P]
        dBbar[:, t] = (dv_t * x_t).sum(dim=(1, 2))       # [B,Nc,2]

        # (3) rotated terms: Rh = R_t h_{t-1}, Rv = R_t v_{t-1}
        hr, hi = h_tm1[..., 0], h_tm1[..., 1]
        Rh = torch.stack((cos_t * hr - sin_t * hi, sin_t * hr + cos_t * hi), -1)
        vr, vi = v_tm1[..., 0], v_tm1[..., 1]
        Rv = torch.stack((cos_t * vr - sin_t * vi, sin_t * vr + cos_t * vi), -1)
        # per-head coefficient grads (sum over p,c,d within head)
        dalpha[:, t] = (Rh * gh).sum(dim=(2, 3, 4))      # [B,H]
        dbeta[:, t] = (Rv * gh).sum(dim=(2, 3, 4))
        dgamma[:, t] = (v_t * gh).sum(dim=(2, 3, 4))
        # phi grad via R'(phi)=[[-sin,-cos],[cos,-sin]] on h_{t-1} (coef alpha)
        # and v_{t-1} (coef beta). For coef*R@w with adjoint g=(g0,g1):
        #   dphi += coef*( g0*(-sin*w0 - cos*w1) + g1*(cos*w0 - sin*w1) )
        g0, g1 = gh[..., 0], gh[..., 1]                  # [B,H,P,Nc]
        dRh = g0 * (-sin_t * hr - cos_t * hi) + g1 * (cos_t * hr - sin_t * hi)
        dRv = g0 * (-sin_t * vr - cos_t * vi) + g1 * (cos_t * vr - sin_t * vi)
        a2 = alpha[:, t].view(B, H, 1, 1)
        b2 = beta[:, t].view(B, H, 1, 1)
        dphi[:, t] = (a2 * dRh + b2 * dRv).sum(dim=2)    # sum over P -> [B,H,Nc]

        # (4) propagate to previous step (R^T adjoint; R^T(phi)=R(-phi))
        #   R^T @ g = (cos*g0 + sin*g1, -sin*g0 + cos*g1)
        RTg = torch.stack((cos_t * g0 + sin_t * g1,
                           -sin_t * g0 + cos_t * g1), -1)  # [B,H,P,Nc,2]
        gv = b_t * RTg                                   # dL/dv_{t-1} (beta term)
        gh = a_t * RTg                                   # dL/dh_{t-1}

    # ---- fold the per-head coefficient Jacobians ----
    # alpha=exp(dt*A): dalpha/d(dt)=A*alpha, dalpha/dA=dt*alpha.
    # beta=(1-lam)*dt*alpha: dbeta/d(dt)=(1-lam)*alpha*(1+dt*A),
    #                        dbeta/dlam=-dt*alpha, dbeta/dA=(1-lam)*dt^2*alpha.
    # gamma=lam*dt: dgamma/d(dt)=lam, dgamma/dlam=dt.
    # phi=dt*theta: dphi/d(dt)=theta (sum c), dphi/dtheta=dt (sum h).
    dt_phi = (dphi * theta.unsqueeze(-2)).sum(dim=-1)    # [B,L,H]
    ddt = (dalpha * (A_real * alpha)
           + dbeta * ((1.0 - lam) * alpha * (1.0 + dt * A_real))
           + dgamma * lam
           + dt_phi)                                     # [B,L,H] (post-softplus)
    dlam = dbeta * (-dt * alpha) + dgamma * dt           # [B,L,H] (w.r.t. lambda)
    dA_real = dalpha * (dt * alpha) + dbeta * ((1.0 - lam) * dt * dt * alpha)
    dtheta = (dphi * dt.unsqueeze(-1)).sum(dim=2)        # [B,L,Nc] (sum over H)

    return dict(dx_h=dx_h, ddt=ddt, dlam=dlam, dA_real=dA_real,
                dtheta=dtheta, dBbar=dBbar, dCbar=dCbar)


# ===========================================================================
#  ONE Mamba3Layer forward + MANUAL backward, reading the live module's params.
#  Param order mirrors named_parameters(): A_log, D, B_bias, Bhat_bias, C_bias,
#  Chat_bias, in_proj.w, x_proj.w, dt_proj.w, dt_proj.b, B_norm.w, Bhat_norm.w,
#  C_norm.w, Chat_norm.w, out_proj.w  (own params first, then submodules).
# ===========================================================================
def layer_forward(layer: Mamba3Layer, P_: Dict[str, torch.Tensor], x):
    H, P, Nc = layer.n_heads, layer.head_dim, layer.complex_dim
    di, dt_rank = layer.d_inner, layer.dt_rank
    eps = layer.rmsnorm_eps
    B, L, _ = x.shape
    residual = x

    xz = linear_forward(x, P_["in_proj.weight"])         # [B,L,2*di]
    x_main, z = xz.split(di, dim=-1)                     # each [B,L,di]
    x_in = x_main                                        # NO SiLU, NO conv

    proj = linear_forward(x_in, P_["x_proj.weight"])
    dt_lr, A_mod, theta, u_lam, Br, Bi, Cr, Ci = torch.split(
        proj, [dt_rank, H, Nc, H, Nc, Nc, Nc, Nc], dim=-1)

    dt_pre = linear_forward(dt_lr, P_["dt_proj.weight"], P_["dt_proj.bias"])  # [B,L,H]
    dt = torch.nn.functional.softplus(dt_pre)            # [B,L,H]
    base_rate = torch.exp(P_["A_log"])                   # [H]
    A_arg = A_mod + base_rate.view(1, 1, -1)             # [B,L,H]
    A_real = -torch.nn.functional.softplus(A_arg)        # [B,L,H]
    lam = torch.sigmoid(u_lam)                           # [B,L,H]

    Brn, Bcache = rmsnorm_forward(Br, P_["B_norm.weight"], eps)
    Bin, Bicache = rmsnorm_forward(Bi, P_["Bhat_norm.weight"], eps)
    Crn, Ccache = rmsnorm_forward(Cr, P_["C_norm.weight"], eps)
    Cin, Cicache = rmsnorm_forward(Ci, P_["Chat_norm.weight"], eps)
    Br2 = Brn + P_["B_bias"]
    Bi2 = Bin + P_["Bhat_bias"]
    Cr2 = Crn + P_["C_bias"]
    Ci2 = Cin + P_["Chat_bias"]
    Bbar = torch.stack((Br2, Bi2), dim=-1)               # [B,L,Nc,2]
    Cbar = torch.stack((Cr2, -Ci2), dim=-1)              # [B,L,Nc,2]

    x_h = x_in.view(B, L, H, P)
    y_h, scan_cache = scan_forward(x_h, dt, A_real, theta, lam, Bbar, Cbar, H, P, Nc)
    y = y_h.reshape(B, L, di)                            # [B,L,di]

    y_skip = y + x_in * P_["D"].view(1, 1, di)
    sz = silu(z)
    y_gated = y_skip * sz
    out = linear_forward(y_gated, P_["out_proj.weight"])
    # In the Llama-style Mamba3Block the mixer's internal residual is OFF
    # (residual added at the block level around the pre-norm); honor the live
    # module's flag so the manual backward matches autograd exactly.
    r = out + residual if layer.residual else out

    cache = dict(x=x, xz=xz, x_main=x_main, z=z, x_in=x_in, dt_lr=dt_lr,
                 A_mod=A_mod, theta=theta, u_lam=u_lam, dt_pre=dt_pre, dt=dt,
                 A_arg=A_arg, A_real=A_real, base_rate=base_rate, lam=lam,
                 Bcache=Bcache, Bicache=Bicache, Ccache=Ccache, Cicache=Cicache,
                 Bbar=Bbar, Cbar=Cbar, scan_cache=scan_cache, y=y,
                 y_skip=y_skip, sz=sz, y_gated=y_gated, H=H, P=P, Nc=Nc, di=di,
                 dt_rank=dt_rank, residual=layer.residual)
    return r, cache


def layer_backward(layer: Mamba3Layer, P_: Dict[str, torch.Tensor], cache, dr):
    g = {}
    H, P, Nc, di = cache["H"], cache["P"], cache["Nc"], cache["di"]
    B, L = cache["x"].shape[0], cache["x"].shape[1]

    # r = out (+ residual if the mixer keeps its own residual). In the Llama-style
    # Mamba3Block residual is OFF, so dr is dL/d(out) only and the input residual
    # path is added at the block level (see block_backward), NOT here.
    dout = dr
    dx_residual = dr.clone() if cache["residual"] else torch.zeros_like(dr)
    # out = y_gated @ out_proj^T
    dy_gated, g["out_proj.weight"], _ = linear_backward(
        dout, cache["y_gated"], P_["out_proj.weight"], False)
    # y_gated = y_skip * sz
    dy_skip = dy_gated * cache["sz"]
    dsz = dy_gated * cache["y_skip"]
    dz = dsz * silu_grad(cache["z"])                     # sz = silu(z)
    # y_skip = y + x_in * D
    dy = dy_skip
    g["D"] = (dy_skip * cache["x_in"]).sum(dim=(0, 1))   # [di]
    dx_in = dy_skip * P_["D"].view(1, 1, di)             # D-skip path into x_in

    # scan backward: dy [B,L,di] -> [B,L,H,P]
    dy_h = dy.view(B, L, H, P)
    sg = scan_backward(dy_h, cache["scan_cache"])
    dx_in = dx_in + sg["dx_h"].reshape(B, L, di)         # scan path into x_in

    # Cbar = (Cr2, -Ci2) ; Bbar = (Br2, Bi2)
    dBr2 = sg["dBbar"][..., 0]
    dBi2 = sg["dBbar"][..., 1]
    dCr2 = sg["dCbar"][..., 0]
    dCi2 = -sg["dCbar"][..., 1]
    # +bias then RMSNorm. bias grad = sum over (b,l).
    g["B_bias"] = dBr2.sum(dim=(0, 1))
    g["Bhat_bias"] = dBi2.sum(dim=(0, 1))
    g["C_bias"] = dCr2.sum(dim=(0, 1))
    g["Chat_bias"] = dCi2.sum(dim=(0, 1))
    dBr, g["B_norm.weight"] = rmsnorm_backward(dBr2, cache["Bcache"])
    dBi, g["Bhat_norm.weight"] = rmsnorm_backward(dBi2, cache["Bicache"])
    dCr, g["C_norm.weight"] = rmsnorm_backward(dCr2, cache["Ccache"])
    dCi, g["Chat_norm.weight"] = rmsnorm_backward(dCi2, cache["Cicache"])

    # dt: ddt (post-softplus) -> dt_pre via softplus' -> dt_proj backward
    ddt_pre = sg["ddt"] * softplus_grad(cache["dt_pre"])
    ddt_lr, g["dt_proj.weight"], g["dt_proj.bias"] = linear_backward(
        ddt_pre, cache["dt_lr"], P_["dt_proj.weight"], True)

    # A_real = -softplus(A_arg), A_arg = A_mod + base_rate, base_rate=exp(A_log)
    dA_arg = sg["dA_real"] * (-softplus_grad(cache["A_arg"]))   # [B,L,H]
    dA_mod = dA_arg
    g["A_log"] = (dA_arg * cache["base_rate"].view(1, 1, -1)).sum(dim=(0, 1))  # [H]

    # lambda = sigmoid(u_lam) -> du_lam
    du_lam = sg["dlam"] * (cache["lam"] * (1.0 - cache["lam"]))

    # theta is a direct slice of the x_proj output (head-shared)
    dtheta = sg["dtheta"]

    # reassemble x_proj output grad: [dt_lr | A_mod | theta | u_lam | Br|Bi|Cr|Ci]
    dproj = torch.cat([ddt_lr, dA_mod, dtheta, du_lam, dBr, dBi, dCr, dCi], dim=-1)
    dx_in3, g["x_proj.weight"], _ = linear_backward(
        dproj, cache["x_in"], P_["x_proj.weight"], False)
    dx_in = dx_in + dx_in3                               # x_proj path into x_in

    # x_in = x_main (no SiLU) -> dx_main = dx_in
    dx_main = dx_in
    # xz = [x_main | z] ; xz = in_proj(x)
    dxz = torch.cat([dx_main, dz], dim=-1)
    dx_inproj, g["in_proj.weight"], _ = linear_backward(
        dxz, cache["x"], P_["in_proj.weight"], False)
    dx = dx_inproj + dx_residual                         # x fans to in_proj + residual
    return dx, g


# ===========================================================================
#  SwiGLU MLP forward + MANUAL backward.   (Llama feed-forward, NO biases.)
#    g_pre = x @ Wgate^T ;  u = x @ Wup^T
#    s     = SiLU(g_pre) ;  prod = s * u
#    out   = prod @ Wdown^T
#  Backward (dout):
#    (dprod, dWdown) = linear_bwd(dout, prod, Wdown)
#    ds = dprod * u ;  du = dprod * s
#    dg_pre = ds * SiLU'(g_pre)
#    (dx_g, dWgate) = linear_bwd(dg_pre, x, Wgate)
#    (dx_u, dWup)   = linear_bwd(du,     x, Wup)
#    dx = dx_g + dx_u
# ===========================================================================
def swiglu_forward(x, Wgate, Wup, Wdown):
    g_pre = linear_forward(x, Wgate)        # [B,L,d_ff]
    u = linear_forward(x, Wup)              # [B,L,d_ff]
    s = silu(g_pre)                         # [B,L,d_ff]
    prod = s * u                            # [B,L,d_ff]
    out = linear_forward(prod, Wdown)       # [B,L,d]
    cache = dict(x=x, g_pre=g_pre, u=u, s=s, prod=prod)
    return out, cache


def swiglu_backward(dout, cache, Wgate, Wup, Wdown):
    x, g_pre, u, s, prod = (cache["x"], cache["g_pre"], cache["u"],
                            cache["s"], cache["prod"])
    g = {}
    dprod, g["down_proj.weight"], _ = linear_backward(dout, prod, Wdown, False)
    ds = dprod * u
    du = dprod * s
    dg_pre = ds * silu_grad(g_pre)
    dx_g, g["gate_proj.weight"], _ = linear_backward(dg_pre, x, Wgate, False)
    dx_u, g["up_proj.weight"], _ = linear_backward(du, x, Wup, False)
    dx = dx_g + dx_u
    return dx, g


# ===========================================================================
#  ONE Mamba3Block forward + MANUAL backward — the Llama-style pre-norm layer:
#    h1 = x  + mixer( RMSNorm_mix(x)  )      # residual=False mixer
#    h2 = h1 + mlp  ( RMSNorm_mlp(h1) )      # SwiGLU MLP
#  Param keys here are RELATIVE to the block (e.g. "mixer.in_proj.weight",
#  "mixer_norm.weight", "mlp.gate_proj.weight", "mlp_norm.weight").
# ===========================================================================
def block_forward(block, P_: Dict[str, torch.Tensor], x):
    eps = block.mixer_norm.eps
    # --- mixer sub-block (pre-norm + residual) ---
    xn_mix, mixn_cache = rmsnorm_forward(x, P_["mixer_norm.weight"], eps)
    Pm = {k[len("mixer."):]: v for k, v in P_.items() if k.startswith("mixer.")}
    mix_out, mix_cache = layer_forward(block.mixer, Pm, xn_mix)
    h1 = x + mix_out
    # --- SwiGLU MLP sub-block (pre-norm + residual) ---
    eps2 = block.mlp_norm.eps
    h1n, mlpn_cache = rmsnorm_forward(h1, P_["mlp_norm.weight"], eps2)
    mlp_out, mlp_cache = swiglu_forward(
        h1n, P_["mlp.gate_proj.weight"], P_["mlp.up_proj.weight"],
        P_["mlp.down_proj.weight"])
    h2 = h1 + mlp_out
    cache = dict(mixn_cache=mixn_cache, mix_cache=mix_cache,
                 mlpn_cache=mlpn_cache, mlp_cache=mlp_cache)
    return h2, cache


def block_backward(block, P_: Dict[str, torch.Tensor], cache, dh2):
    g = {}
    # h2 = h1 + mlp(mlp_norm(h1))
    dmlp_out = dh2
    dh1 = dh2.clone()                                    # residual path
    dh1n, mlp_g = swiglu_backward(
        dmlp_out, cache["mlp_cache"], P_["mlp.gate_proj.weight"],
        P_["mlp.up_proj.weight"], P_["mlp.down_proj.weight"])
    for k, v in mlp_g.items():
        g[f"mlp.{k}"] = v
    dh1_mlpnorm, g["mlp_norm.weight"] = rmsnorm_backward(dh1n, cache["mlpn_cache"])
    dh1 = dh1 + dh1_mlpnorm                              # h1 fans to residual + mlp_norm

    # h1 = x + mixer(mixer_norm(x))
    dmix_out = dh1
    dx = dh1.clone()                                     # residual path
    Pm = {k[len("mixer."):]: v for k, v in P_.items() if k.startswith("mixer.")}
    dxn_mix, mix_g = layer_backward(block.mixer, Pm, cache["mix_cache"], dmix_out)
    for k, v in mix_g.items():
        g[f"mixer.{k}"] = v
    dx_mixnorm, g["mixer_norm.weight"] = rmsnorm_backward(dxn_mix, cache["mixn_cache"])
    dx = dx + dx_mixnorm                                 # x fans to residual + mixer_norm
    return dx, g


# ===========================================================================
#  FULL MODEL forward + manual backward (reads the live Mamba3Model params).
# ===========================================================================
def model_forward(model: Mamba3Model, tokens, targets):
    dev = model.tok.weight.device
    S = tokens.shape[1]
    pos_ids = torch.arange(S, device=dev)
    named = dict(model.named_parameters())
    h = model.tok.weight[tokens] + model.pos.weight[pos_ids].unsqueeze(0)  # [B,S,d]
    layer_caches = []
    for li, block in enumerate(model.layers):
        P_ = {k[len(f"layers.{li}."):]: v for k, v in named.items()
              if k.startswith(f"layers.{li}.")}
        h, lc = block_forward(block, P_, h)
        layer_caches.append((P_, lc))
    h_last = h[:, -1, :]
    hn, norm_cache = rmsnorm_forward(h_last, named["norm.weight"], model.norm.eps)
    logits = hn @ named["out.weight"].t() + named["out.bias"]
    loss = cross_entropy_forward(logits, targets)
    cache = dict(tokens=tokens, targets=targets, layer_caches=layer_caches,
                 norm_cache=norm_cache, logits=logits, hn=hn, h_shape=h.shape,
                 d=model.tok.weight.shape[1])
    return loss, cache


def model_backward(model: Mamba3Model, cache):
    named = dict(model.named_parameters())
    grads: Dict[str, torch.Tensor] = {}
    tokens, targets = cache["tokens"], cache["targets"]
    d = cache["d"]

    dlogits = cross_entropy_backward(cache["logits"], targets)   # [B,p]
    grads["out.weight"] = dlogits.t() @ cache["hn"]
    grads["out.bias"] = dlogits.sum(dim=0)
    dhn = dlogits @ named["out.weight"]
    dh_last, grads["norm.weight"] = rmsnorm_backward(dhn, cache["norm_cache"])

    B, S, _ = cache["h_shape"]
    dh = torch.zeros(B, S, d, device=dhn.device, dtype=dhn.dtype)
    dh[:, -1, :] = dh_last
    for li in reversed(range(len(model.layers))):
        P_, lc = cache["layer_caches"][li]
        dh, lg = block_backward(model.layers[li], P_, lc, dh)
        for k, v in lg.items():
            grads[f"layers.{li}.{k}"] = v

    # embeddings: h0 = tok[tokens] + pos[pos_ids]
    dtok = torch.zeros_like(named["tok.weight"])
    dtok.index_add_(0, tokens.reshape(-1), dh.reshape(-1, d))
    grads["tok.weight"] = dtok
    grads["pos.weight"] = dh.sum(dim=0)                          # [S,d]
    return grads


def manual_input_grad(model: Mamba3Model, tokens, targets):
    """Manual dL/d(embedding-input). Returns dh0 [B,S,d] (the grad w.r.t.
    tok+pos sum), to compare against autograd on the embedded input."""
    loss, cache = model_forward(model, tokens, targets)
    named = dict(model.named_parameters())
    d = cache["d"]
    dlogits = cross_entropy_backward(cache["logits"], targets)
    dhn = dlogits @ named["out.weight"]
    dh_last, _ = rmsnorm_backward(dhn, cache["norm_cache"])
    B, S, _ = cache["h_shape"]
    dh = torch.zeros(B, S, d, device=dhn.device, dtype=dhn.dtype)
    dh[:, -1, :] = dh_last
    for li in reversed(range(len(model.layers))):
        P_, lc = cache["layer_caches"][li]
        dh, _ = block_backward(model.layers[li], P_, lc, dh)
    return dh


# ===========================================================================
#  ORACLE — the original foundation properties (A)-(E).
# ===========================================================================
def run_oracle() -> bool:
    print("=" * 78)
    print("Mamba-3 (SISO) fp64 ORACLE  —  forward + manual-backward ground truth")
    print("=" * 78)

    cfg = TOY
    toks, tgts = _make_batch(cfg)

    model64 = _build(cfg, torch.float64, seed=0)
    for p in model64.parameters():
        p.requires_grad_(True)
    loss64, logits64 = _loss(model64, toks, tgts)
    assert torch.isfinite(loss64).all(), "fp64 loss not finite"
    assert logits64.dtype == torch.float64
    assert torch.isfinite(logits64).all(), "fp64 logits not finite"
    model64.zero_grad(set_to_none=True)
    loss64.backward()

    n_params = 0
    n_grad_params = 0
    bad = []
    zero_leaves = []
    for name, p in model64.named_parameters():
        n_params += 1
        if p.grad is None:
            bad.append(f"{name}: grad is None")
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(f"{name}: grad has NaN/Inf")
            continue
        n_grad_params += 1
        if p.grad.abs().max().item() == 0.0:
            zero_leaves.append(name)

    print(f"\n[C] fp64 forward loss            = {loss64.item():.16e}  (finite={bool(torch.isfinite(loss64))})")
    print(f"[B] parameters total             = {n_params}")
    print(f"[B] parameters w/ finite grad    = {n_grad_params}")
    if bad:
        print("[B] FAIL — params with bad grads:")
        for b in bad:
            print("      " + b)
    if zero_leaves:
        print(f"[B] NOTE — params with ALL-ZERO grad ({len(zero_leaves)}): {zero_leaves}")

    grads_ok = (not bad)
    if zero_leaves:
        grads_ok = False

    model64.eval()
    with torch.no_grad():
        l_a = model64(toks)
        l_b = model64(toks)
    deterministic = torch.equal(l_a, l_b)
    print(f"\n[A] forward deterministic (bit-eq across 2 runs) = {deterministic}")

    model32 = _build(cfg, torch.float32, seed=0)
    with torch.no_grad():
        logits32 = model32(toks)
        loss32, _ = _loss(model32, toks, tgts)
    max_abs = (logits32.double() - logits64.detach()).abs().max().item()
    rel = max_abs / (logits64.detach().abs().max().item() + 1e-12)
    f32_match = torch.allclose(logits32.double(), logits64.detach(), atol=ATOL_F32, rtol=RTOL_F32)
    print(f"\n[D] fp32 vs fp64 forward: max|abs diff| = {max_abs:.3e}  (rel = {rel:.3e})")
    print(f"[D] fp32 loss = {loss32.item():.8e}   fp64 loss = {loss64.item():.8e}")
    print(f"[D] within tol (atol={ATOL_F32}, rtol={RTOL_F32}) = {f32_match}")

    fd_ok = _fd_spotcheck(cfg, toks, tgts)
    print(f"\n[B*] finite-difference spot check on out.bias[0]: {'PASS' if fd_ok else 'FAIL'}")

    toy_pc = param_count(TOY)
    print(f"\n[E] TOY param count  (d={TOY['d']}, state={TOY['state_dim']}, nl={TOY['nl']}, "
          f"mixer+SwiGLU) = {toy_pc:,}")
    mix_t, mlp_t, oth_t = param_count_breakdown(TOY)
    print(f"    TOY breakdown: mixer={mix_t:,}  swiglu-mlp={mlp_t:,}  embed/head/norm={oth_t:,}")

    big_pc = param_count(BIG)
    mix_b, mlp_b, oth_b = param_count_breakdown(BIG)
    print(f"\n[E] 1.5B-scale param count (d={BIG['d']}, state={BIG['state_dim']}, nl={BIG['nl']}, "
          f"MLP inner={2*BIG['d']}, mixer+SwiGLU)")
    print(f"    raw (small grokking vocab ntok={BIG['ntok']}) = {big_pc:,}  ({big_pc/1e9:.4f}B)")
    print(f"    breakdown: 24 mixers={mix_b:,} ({mix_b/1e9:.4f}B)  "
          f"24 swiglu={mlp_b:,} ({mlp_b/1e9:.4f}B)  embed/head/norm={oth_b:,}")
    layers_only = mix_b + mlp_b
    print(f"    layer stack (mixers + SwiGLU, vocab-independent) = {layers_only:,} "
          f"({layers_only/1e9:.4f}B)")

    # Headline ~1.5B: the paper's config with the Llama-3.1 vocab (128256) and
    # TIED embeddings (out.weight tied to tok.weight, the Mamba LM convention).
    big_llama = param_count(BIG_LLAMA_VOCAB)
    V = BIG_LLAMA_VOCAB["ntok"]
    d = BIG_LLAMA_VOCAB["d"]
    # param_count() builds an UNtied model (separate out.weight + out.bias). The
    # real LM ties the head to the embedding, so the tied total removes one V*d
    # head matrix and the V-length bias.
    tied = big_llama - (V * d) - V
    print(f"\n[E] CANONICAL 1.5B (Llama-3.1 vocab V={V}, d={d}, nl=24, MLP inner=4096):")
    print(f"    untied (sep. head)            = {big_llama:,}  ({big_llama/1e9:.4f}B)")
    print(f"    TIED embeddings (paper conv.) = {tied:,}  ({tied/1e9:.4f}B)  <-- ~1.5B target")

    ok = (
        bool(torch.isfinite(loss64))
        and grads_ok
        and deterministic
        and f32_match
        and fd_ok
    )
    print("\n[A-E] foundation result:", "PASS" if ok else "FAIL")
    return ok


def _fd_spotcheck(cfg, toks, tgts, eps=1e-6) -> bool:
    model = _build(cfg, torch.float64, seed=0)
    for p in model.parameters():
        p.requires_grad_(True)
    loss, _ = _loss(model, toks, tgts)
    model.zero_grad(set_to_none=True)
    loss.backward()
    g_auto = model.out.bias.grad[0].item()
    with torch.no_grad():
        orig = model.out.bias[0].item()
        model.out.bias[0] = orig + eps
        lp, _ = _loss(model, toks, tgts)
        model.out.bias[0] = orig - eps
        lm, _ = _loss(model, toks, tgts)
        model.out.bias[0] = orig
    g_fd = (lp.item() - lm.item()) / (2 * eps)
    err = abs(g_auto - g_fd) / (abs(g_fd) + 1e-12)
    print(f"     autograd={g_auto:.10e}  fd={g_fd:.10e}  rel_err={err:.3e}")
    return err < 1e-5


# ===========================================================================
#  (F) MANUAL BACKWARD vs AUTOGRAD — the headline gate (<=1e-10 rel, fp64).
# ===========================================================================
def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a-b| / (max|b| + tiny). Both fp64."""
    num = (a - b).abs().max().item()
    den = b.abs().max().item()
    return num / (den + 1e-300)


def manual_vs_autograd(cfg, seed_model=0, seed_data=1234, label="") -> Tuple[bool, float]:
    torch.manual_seed(seed_model)
    model = Mamba3Model(**cfg).double()
    toks, tgts = _make_batch(cfg, seed=seed_data)

    # --- autograd reference (loss grad w.r.t. every param + the embedded input)
    for p in model.parameters():
        p.requires_grad_(True)
    dev = model.tok.weight.device
    S = toks.shape[1]
    pos_ids = torch.arange(S, device=dev)
    h0 = model.tok.weight[toks] + model.pos.weight[pos_ids].unsqueeze(0)
    h0.retain_grad()
    h = h0
    for layer in model.layers:
        h = layer(h)
    logits = model.out(model.norm(h[:, -1, :]))
    loss_auto = torch.nn.functional.cross_entropy(logits, tgts)
    model.zero_grad(set_to_none=True)
    loss_auto.backward()
    auto = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    auto_input = h0.grad.detach().clone()

    # --- manual fwd+bwd
    with torch.no_grad():
        loss_man, cache = model_forward(model, toks, tgts)
        man = model_backward(model, cache)
        man_input = manual_input_grad(model, toks, tgts)

    # loss agreement
    loss_rel = abs(loss_man.item() - loss_auto.item()) / (abs(loss_auto.item()) + 1e-300)

    # per-parameter table
    print(f"\n--- manual vs autograd  [{label}]  cfg={cfg}")
    print(f"    loss(manual)={loss_man.item():.16e}  loss(autograd)={loss_auto.item():.16e}")
    print(f"    loss rel-err = {loss_rel:.3e}")
    print(f"    {'parameter':32s} {'shape':18s} {'max-rel-err':>12s}")
    worst = loss_rel
    names = list(auto.keys())
    for name in names:
        a = auto[name]
        m = man.get(name)
        assert m is not None, f"manual backward missing grad for {name}"
        assert m.shape == a.shape, f"{name}: shape {tuple(m.shape)} != {tuple(a.shape)}"
        re = _rel_err(m, a)
        worst = max(worst, re)
        flag = "" if re <= MANUAL_RTOL else "   <-- OVER TOL"
        print(f"    {name:32s} {str(tuple(a.shape)):18s} {re:12.3e}{flag}")
    # the input grad
    re_in = _rel_err(man_input, auto_input)
    worst = max(worst, re_in)
    flag = "" if re_in <= MANUAL_RTOL else "   <-- OVER TOL"
    print(f"    {'[input] dL/d(embed)':32s} {str(tuple(auto_input.shape)):18s} {re_in:12.3e}{flag}")
    print(f"    WORST rel-err (incl. loss + input) = {worst:.3e}   (tol {MANUAL_RTOL:.0e})")
    return (worst <= MANUAL_RTOL), worst


def run_manual_gate() -> bool:
    print("\n" + "=" * 78)
    print("(F) MANUAL backward vs torch.autograd  —  fp64, <=1e-10 rel (target ~1e-12)")
    print("=" * 78)
    ok_a, w_a = manual_vs_autograd(GATE_A, seed_model=0, seed_data=1234, label="seed0 / shapeA")
    ok_b, w_b = manual_vs_autograd(GATE_B, seed_model=7, seed_data=99, label="seed7 / shapeB")
    # also run the manual gate at the TOY config (the real foundation shape)
    ok_t, w_t = manual_vs_autograd(TOY, seed_model=0, seed_data=1234, label="TOY config")
    ok = ok_a and ok_b and ok_t
    print(f"\n[F] manual-vs-autograd gate: {'PASS' if ok else 'FAIL'} "
          f"(worst rel-err = {max(w_a, w_b, w_t):.3e})")
    return ok


def main() -> bool:
    found_ok = run_oracle()
    manual_ok = run_manual_gate()
    ok = found_ok and manual_ok
    print("\n" + "=" * 78)
    print(f"ORACLE RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
