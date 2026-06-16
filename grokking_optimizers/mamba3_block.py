"""mamba3_block.py — reference PyTorch implementation of the **Mamba-3** (SISO)
sequence-model block (arXiv 2603.15569, ICLR 2026, Lahoti, Li, Chen, Wang, Bick,
Kolter, Dao, Gu — "Mamba-3: Improved Sequence Modeling using State Space
Principles").

PURPOSE / SCOPE
  This is the *reversible foundation* reference for the new ``mamba`` cell. It is
  a clean, autograd-only PyTorch module that a CUDA megakernel will later be
  transcribed from line-for-line. It is a **drop-in replacement** for the
  Mamba-1 ``SelectiveSSMLayer`` in grokking_race_v2.py: same ``__init__(d, ...)``
  signature shape and the same ``forward(x:[B,L,d]) -> [B,L,d]`` residual
  contract, so ``MambaModel`` can swap one for the other unchanged.

  We implement the **base SISO** Mamba-3 layer ONLY (the MIMO variant of
  Section 3.3 / Appendix C is a documented later option, see ``mimo_rank`` note
  at the bottom of this docstring). Complex arithmetic is materialized as
  explicit real/imag pairs (2x2 rotation blocks on a *real* state vector) per
  Proposition 2 — NOT torch.complex — so it is fully autograd-friendly today and
  portable to a CUDA kernel that has no complex type.

================================================================================
THE EXACT FORWARD EQUATIONS WE IMPLEMENT  (paper section / equation numbers)
================================================================================

(0) MODEL + BLOCK LAYOUT  (Section 3.4 "Mamba-3 Architecture", Fig. 2):
    MODEL LEVEL — the overall architecture follows Llama, "alternating Mamba-3
    and SwiGLU blocks with pre-norm" (Sec 3.4). Each ``Mamba3Block`` is one
    Llama-style layer = a Mamba-3 MIXER sub-block then a SwiGLU MLP sub-block,
    each pre-normed + residual:
        h = h + Mamba3Mixer( RMSNorm(h) )
        h = h + SwiGLU_MLP ( RMSNorm(h) )
    ``nl`` counts the MIXER blocks; each is paired with a SwiGLU MLP (so ``nl``
    of each — the Mamba-2 convention the paper follows, Appendix D). At the 1.5B
    scale: d_model=2048, nl=24, MLP inner 4096 (=2*d). A single final RMSNorm
    precedes the LM head (the Mamba-2 post-gate norm is removed; BCNorm
    stabilizes — Sec 3.4). See ``Mamba3Block`` / ``Mamba3Model`` /
    ``SwiGLU_MLP`` below and MAMBA3_REFERENCE.md §1.6/§7.

    MIXER BLOCK — the Mamba-2 mixer layout is retained, with these Mamba-3
    changes:
      * the SSD/SSM core is the complex-valued exponential-trapezoidal SSM below;
      * data-dependent RoPE is applied to B, C (the complex part of A);
      * **the short causal conv1d is DROPPED** — the exponential-trapezoidal
        recurrence induces an implicit width-2 convolution on the state-input
        B_t*x_t (Section 3.1.2, Remark 4) and, combined with the explicit B,C
        biases (Section 3.4 "B, C Biases", Appendix F), empirically obviates the
        external short conv (Section 4.2, Table 5a). So there is NO nn.Conv1d in
        this block — see AMBIGUITY note in MAMBA3_REFERENCE.md.
      * BCNorm (RMSNorm on B and C, a.k.a. QKNorm) is added after the B,C
        projections (Section 3.4 "BC / QK Normalization").
      * For the *pure* Mamba-3 model the post-gate RMSNorm of Mamba-2 is REMOVED
        (BCNorm stabilizes training); we follow that default. (Section 3.4.)

    forward(x), x:[B,L,d]:
      residual = x
      xz = in_proj(x)                       # [B,L,2*d_inner]      (no bias)
      x_main, z = xz.chunk(2, -1)           # each [B,L,d_inner]
      # NO conv1d here (dropped, see above) and NO SiLU on the SSM input: the
      # paper (Sec 3.4) obviates the short conv "and its accompanying activation
      # function", so x_main feeds the SSM/projections DIRECTLY (RESOLVED below).
      (dt, A_real, theta, lam, B, C) = projections(x_main) # all data-dependent
      y = mamba3_ssm(x_main, dt, A_real, theta, lam, B, C) # the recurrence (1)-(4)
      y = y + x_main * D                    # D skip / residual (per-channel), Eq (4) of M1
      y = y * SiLU(z)                        # gated output, phi(Y,Z)=Y(.)SiLU(Z)  (App. C, Eq C)
      out = out_proj(y)                     # [B,L,d]              (no bias)
      return out (+ residual if self.residual)  # in Mamba3Block residual is OFF;
                                            # the block adds h + Mixer(RMSNorm(h))

  NOTE on D skip: Mamba-1/-2 keep a per-channel D "skip"/feedthrough term
  y += D (.) x. Mamba-3 (Fig. 2) keeps the same block scaffold; we retain D as a
  learnable per-(d_inner) skip — flagged in MAMBA3_REFERENCE.md.

(1) EXPONENTIAL-TRAPEZOIDAL DISCRETIZATION  (Section 3.1, Proposition 1, Eq 5-6;
    Table 1 last row):
    The continuous LTV SSM  h'(t) = A(t) h(t) + B(t) x(t),  y = C(t)^T h(t)
    is discretized with the *exponential-trapezoidal* rule to the 3-term
    recurrence (Eq 6):

        h_t = alpha_t * h_{t-1}  +  beta_t * B_{t-1} x_{t-1}  +  gamma_t * B_t x_t

    with the discretization coefficients (Eq 5, Prop 1; Table 1):

        alpha_t = exp(dt_t * A_t)                       # state-transition decay
        beta_t  = (1 - lambda_t) * dt_t * exp(dt_t*A_t) = (1 - lambda_t) * dt_t * alpha_t
        gamma_t = lambda_t * dt_t

    where lambda_t in [0,1] is a **data-dependent** convex-combination gate.
    - lambda_t = 1   recovers Mamba-2's exponential-Euler (Remark 2).
    - lambda_t = 1/2 recovers the classical trapezoid rule (Remark 2).
    Per Appendix A.3 (Table 8) the *default* parameterization (best ppl) is
        lambda_t = sigmoid(u_t),   u_t a learned linear projection of the token,
    and the O(dt) constraint lambda_t = 1/2 + O(dt) is intentionally NOT enforced
    (it hurts empirical perf). We use that default.

(2) COMPLEX-VALUED STATE via the "RoPE trick"  (Section 3.2; Proposition 2,4;
    Eq 8, 9, 11, 25):
    The SSM is viewed as complex (Eq 8): the transition is Diag(A(t) + i*theta(t))
    and the input/output projections are complex (B + i*Bhat), (C + i*Chat).
    Proposition 2 (Eq 9) shows the discretized complex SSM of complex-state-dim
    N_c is EQUIVALENT to a REAL SSM of state dim N = 2*N_c whose transition is a
    scalar decay times a block-diagonal of 2x2 rotations:

        R_t = blockdiag_{i=1..N_c} R(dt_t * theta_t[i]),
              R(phi) = [[cos phi, -sin phi], [sin phi, cos phi]]
        Bbar_t = [B_t ; Bhat_t]  in R^N        (real/imag stacked per complex coord)
        Cbar_t = [C_t ; -Chat_t] in R^N

    Combining the complex transition with the exponential-trapezoidal rule gives
    the REAL recurrence we actually run (Proposition 4, Eq 25 — the
    *direct-rotation* form; we use Eq 25 rather than the algebraically-identical
    cumulative-product RoPE form Eq 11/24 because per-step rotation is the
    cleanest sequential form for a CUDA kernel to transcribe):

        h_t = alpha_t * R_t @ h_{t-1}
              + beta_t  * R_t @ (Bbar_{t-1} * x_{t-1})
              + gamma_t *        (Bbar_t     * x_t)                       (Eq 25)
        y_t = Cbar_t^T @ h_t

    (Here per *complex coordinate* i the 2-vector h_t[i] := (h_t[2i], h_t[2i+1])
    is rotated by R(dt_t*theta_t[i]); B,C act as scalars on x_t. The decay
    alpha_t and angle phi_t = dt_t*theta_t are both data-dependent.)

    State-tracking: a pure rotation (alpha->1) can implement e.g. parity via
    h_t = R(pi*x_t) h_{t-1} (Section 3.2), which real-eigenvalue SSMs cannot —
    this is the capability the complexification buys.

(3) DATA-DEPENDENT A  (Remark 1, Section 2.2):
    All SSM params are data-dependent in Mamba-3. The real part A_t is a SCALAR
    per head (Eq 8/9/17: "A_t in R is a scalar so that exp(dt*A_t) commutes with
    rotations", paper line ~1431; Mamba-2's scalar-identity transition A_t=A_t*I).
    It is produced from a base per-head A_log parameter modulated by a
    data-dependent per-head log-rate; we follow Mamba-1/-2's stable
    parameterization A_t = -softplus(...) <= 0 so alpha_t = exp(dt_t*A_t) in
    (0,1], with dt_t = softplus(dt_proj(.)) > 0. The imaginary part theta_t is a
    free data-dependent linear projection (the rotation frequency); the per-step
    rotation angle is phi_t = dt_t * theta_t.

    PER-HEAD vs PER-CHANNEL (RESOLVED — paper Sec 3.3 line 501-504 + App C
    line 1661-1663): within a head the P=head_dim channels SHARE the same
    alpha_t, dt_t, B_t, C_t ("stacking the SISO recurrence ... with P copies
    sharing the same alpha_t, dt_t and B_t"). B,C,theta are shared across ALL
    heads (multi-value attention, MVA, line 600). So:
      * dt, A_real, lambda  -> PER HEAD          [B,L,n_heads]
      * theta, B, Bhat, C, Chat -> head-shared    [B,L,complex_dim]
      * x (SSM input), D, z (gate) -> per channel [B,L,d_inner]
    The rotation angle phi = dt * theta is therefore PER-HEAD per-coord
    [B,L,n_heads,complex_dim] — EXACT, not the old per-channel-mean(dt)
    approximation. The state h is [B, n_heads, head_dim, complex_dim, 2]; within
    a head all head_dim channels evolve with the same alpha_t and R_t and the
    head-shared Bbar_t, differing only through the per-channel input x.

(4) BCNorm + B,C BIASES  (Section 3.4 "BC / QK Normalization", "B, C Biases";
    Appendix F, Table 10):
    After the B and C projections we apply RMSNorm (BCNorm/QKNorm) over the
    state dim, then add learnable, data-INDEPENDENT, channel-wise biases to BOTH
    B and C (added AFTER the norm), initialized to all-ones (Appendix F default;
    Table 10a). These biases inject the convolution-like data-independent
    component that (with the trapezoidal conv) replaces the short conv.

================================================================================
SHAPES / CONFIG
================================================================================
  d            model dim D                              (canonical 1.5B: 2048)
  expand       inner expansion factor                   (default 2 -> d_inner=2*d)
  d_inner      = expand * d                              (the SISO channel count P*nheads)
  state_dim    REAL SSM state size N per channel         (default 128; in {64,128})
               -> complex-state-dim N_c = state_dim // 2 (must be even state_dim)
  head_dim     P, SSM head dim                            (default 64; D/head structure)
  n_heads      = d_inner // head_dim
  dt_rank      low-rank dt projection rank               (default max(d//16,1), Mamba-1 style)

  B,C,theta are SHARED across heads (Mamba multi-value-attention head structure,
  Section 3.3 "MIMO Instantiation", line 600): one (n_heads-shared) B,C of
  complex-dim N_c per token -> real Bbar,Cbar of size N=state_dim. dt, A_real and
  lambda are PER HEAD (the P=head_dim channels in a head share them, line 501).
  x (the SSM input), the D skip and the gate z are per d_inner channel.

This file imports torch (+nn,F) only; needs NO GPU and NO compiled extension.
The forward materializes every intermediate tensor explicitly (clarity over
cleverness) so the kernel port and the fp64 oracle can mirror it 1:1.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm over the last dim (used for BCNorm / QKNorm). No bias (RMS, not LN).
# rms(x) = x / sqrt(mean(x^2, -1) + eps) * weight.  (Standard RMSNorm.)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in the INPUT dtype (no fp32 downcast). This keeps the fp64
        # oracle exact (an internal .float() would cap precision at ~1e-7 and
        # break the 1e-12 manual-vs-autograd gate), and is what the CUDA kernel
        # does in its working precision. rms(x) = x * rsqrt(mean(x^2)+eps) * w.
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def apply_rotation(h_pair: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the block-diagonal 2x2 rotation R(phi) per complex coordinate.

    h_pair : [..., N_c, 2]  (last axis = (real, imag) of each complex coord)
    cos,sin: [..., N_c]     (= cos/sin of phi = dt*theta for that coord)
    R(phi) [[c,-s],[s,c]] @ (hr, hi) = (c*hr - s*hi,  s*hr + c*hi)
    Returns [..., N_c, 2].
    """
    hr = h_pair[..., 0]
    hi = h_pair[..., 1]
    out_r = cos * hr - sin * hi
    out_i = sin * hr + cos * hi
    return torch.stack((out_r, out_i), dim=-1)


class SwiGLU_MLP(nn.Module):
    """SwiGLU feed-forward block — the Llama MLP that Mamba-3 alternates with the
    mixer (Section 3.4: "the overall architecture follows Llama, alternating
    Mamba-3 and SwiGLU blocks with pre-norm").

        SwiGLU(x) = down( SiLU(gate(x)) * up(x) )

    Three Linear layers, NO biases (Llama convention; the paper does not mention
    MLP biases and follows Llama, which uses bias-free MLP projections). The
    inner ("hidden"/MLP) dimension is ``d_ff``; at the canonical 1.5B scale
    (d_model=2048) the paper's SISO MLP dim is **4096 = 2 * d_model** (Appendix C
    Table — "SISO MLP dim ... 1.5B: 4,096"; Appendix D fixes expand factor 2), so
    the default ``mlp_ratio=2`` reproduces 4096 at d=2048. (The MIMO variant
    shrinks this to 3824 to parameter-match; SISO uses the full 4096.)

    Note: SwiGLU's gated structure means three projections of width ``d_ff`` cost
    the same as a plain MLP of inner width ``1.5 * d_ff`` (the classic Llama
    "2/3" trick is folded into the chosen 4096 here — the paper states 4096
    directly, so we use it directly and do NOT re-apply a 2/3 factor).
    """

    def __init__(self, d: int, d_ff: Optional[int] = None, mlp_ratio: int = 2):
        super().__init__()
        self.d = d
        self.d_ff = d_ff if d_ff is not None else mlp_ratio * d
        # Llama-style bias-free projections.
        self.gate_proj = nn.Linear(d, self.d_ff, bias=False)   # W_gate
        self.up_proj = nn.Linear(d, self.d_ff, bias=False)     # W_up
        self.down_proj = nn.Linear(self.d_ff, d, bias=False)   # W_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] -> [B, L, d]  (NO residual; caller adds it Llama-style)."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Mamba3Layer(nn.Module):
    """Base SISO Mamba-3 layer (complex-valued exponential-trapezoidal SSM).

    Drop-in for Mamba-1 SelectiveSSMLayer: __init__(d, ...) / forward(x)->x.
    Real-valued I/O; complex state materialized as 2x2 rotation blocks on a real
    state of size ``state_dim`` (=2 * complex_dim). NO conv1d (see module
    docstring (0)).

    Parameters
    ----------
    d : int                model dim D
    state_dim : int        REAL SSM state size N (default 128, must be EVEN;
                           complex dim = state_dim//2)
    dt_rank : int | None   low-rank dt projection rank (default max(d//16,1))
    expand_factor : int    inner expansion (default 2 -> d_inner = 2*d)
    head_dim : int         SSM head dim P (default 64); n_heads = d_inner//head_dim
    rmsnorm_eps : float    eps for BCNorm (default 1e-5)
    residual : bool        if True (default) the block adds its own input
                           residual (``out + x``) — the Mamba-1/-2 drop-in
                           contract. In the Llama-style ``Mamba3Model`` the
                           residual + pre-norm are done at the BLOCK level
                           (``h = h + Mixer(RMSNorm(h))``), so the mixer is
                           constructed with ``residual=False`` and returns the
                           bare ``out``. (Section 3.4 "follows Llama ...
                           with pre-norm".)
    """

    def __init__(
        self,
        d: int,
        state_dim: int = 128,
        dt_rank: Optional[int] = None,
        expand_factor: int = 2,
        head_dim: int = 64,
        rmsnorm_eps: float = 1e-5,
        residual: bool = True,
    ):
        super().__init__()
        assert state_dim % 2 == 0, "state_dim must be even (complex dim = state_dim/2)"
        self.d = d
        self.state_dim = state_dim                 # N (real state size)
        self.complex_dim = state_dim // 2          # N_c
        self.expand_factor = expand_factor
        self.d_inner = d * expand_factor           # SISO channel count
        self.dt_rank = dt_rank or max(d // 16, 1)
        # head structure: n_heads channels of width head_dim. If d_inner is not a
        # multiple of head_dim (tiny-d grokking gate), fall back to head_dim=d_inner
        # (single head) so the toy config still runs. (See MAMBA3_REFERENCE.md.)
        if self.d_inner % head_dim != 0:
            head_dim = self.d_inner
        self.head_dim = head_dim
        self.n_heads = self.d_inner // head_dim
        self.rmsnorm_eps = rmsnorm_eps
        self.residual = residual

        # --- input projection (no conv): x_main + gate z, like Mamba-1/-2 ------
        self.in_proj = nn.Linear(d, self.d_inner * 2, bias=False)

        # --- SSM parameter projections from the main path (no SiLU) ------------
        # All SSM params are DATA-DEPENDENT (Remark 1). PER-HEAD vs head-shared
        # vs per-channel follows the paper (Sec 3.3 / App C — see module docstring
        # (3)):
        #   dt : low-rank (dt_rank) -> PER HEAD (n_heads) (Mamba-1 style low-rank)
        #   A_real (log-rate): PER HEAD (n_heads) data-dependent modulation
        #   theta (rotation freq, imaginary part): per complex_dim (shared across heads)
        #   lambda (trapezoidal gate u_t): PER HEAD (n_heads) -> sigmoid
        #   B, C : complex_dim each for real part + complex_dim each for imag part
        #          (Bhat, Chat) -> shared across heads (multi-value attention).
        # x_proj emits the concatenation, then we split.
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank          # dt (low-rank)
            + self.n_heads        # A_real log-rate modulation (per head)
            + self.complex_dim    # theta (rotation freq)
            + self.n_heads        # lambda gate u_t (per head)
            + 2 * self.complex_dim  # B real + B imag (Bhat)
            + 2 * self.complex_dim,  # C real + C imag (Chat)
            bias=False,
        )
        # dt_proj maps the low-rank dt to PER-HEAD dt (n_heads), not per-channel.
        self.dt_proj = nn.Linear(self.dt_rank, self.n_heads, bias=True)

        # Base A_log parameter — PER HEAD scalar real part (Mamba-2 style init
        # A=arange(1..n_heads)); A_t = -softplus(this base + data-dependent
        # per-head log-rate). The real part of A is a scalar per head (the
        # rotation R_t carries the per-coord structure), so this is [n_heads],
        # NOT a per-coord spectrum.
        A = torch.arange(1, self.n_heads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).contiguous())   # [n_heads]

        # per-channel D skip / feedthrough (Mamba-1/-2 retained, Fig.2 scaffold)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # --- BCNorm (QKNorm): RMSNorm on B and C over the complex_dim ----------
        self.B_norm = RMSNorm(self.complex_dim, eps=rmsnorm_eps)   # on B real-part stream
        self.Bhat_norm = RMSNorm(self.complex_dim, eps=rmsnorm_eps)
        self.C_norm = RMSNorm(self.complex_dim, eps=rmsnorm_eps)
        self.Chat_norm = RMSNorm(self.complex_dim, eps=rmsnorm_eps)

        # --- learnable, data-INDEPENDENT, channel-wise B,C biases (Appendix F) -
        # added AFTER BCNorm, init all-ones. Head-specific in the MIMO/multi-head
        # case; here B,C are head-shared so the bias is per complex_dim.
        self.B_bias = nn.Parameter(torch.ones(self.complex_dim))
        self.Bhat_bias = nn.Parameter(torch.ones(self.complex_dim))
        self.C_bias = nn.Parameter(torch.ones(self.complex_dim))
        self.Chat_bias = nn.Parameter(torch.ones(self.complex_dim))

        # --- output projection -------------------------------------------------
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)

    # -----------------------------------------------------------------------
    # The complex exponential-trapezoidal selective scan (Eq 25 + Eq 5-6).
    # All intermediates materialized explicitly for the CUDA transcription.
    # -----------------------------------------------------------------------
    def _mamba3_scan(
        self,
        x: torch.Tensor,        # [B,L,d_inner]   SSM input (no SiLU)
        dt: torch.Tensor,       # [B,L,n_heads]   > 0   (post-softplus, per head)
        A_real: torch.Tensor,   # [B,L,n_heads]   < 0   (per-head real part)
        phi: torch.Tensor,      # [B,L,n_heads,complex_dim]  rotation angle = dt*theta
        lam: torch.Tensor,      # [B,L,n_heads]   in (0,1)  per-head trapezoidal gate
        Bbar: torch.Tensor,     # [B,L,complex_dim,2]   (B real, Bhat) per complex coord
        Cbar: torch.Tensor,     # [B,L,complex_dim,2]   (C real, -Chat) per complex coord
    ) -> torch.Tensor:
        """Returns y : [B,L,d_inner].

        Per Eq (5)-(6),(25). State h is real, shape
        [B, n_heads, head_dim, complex_dim, 2] (a 2-vector per complex coord per
        (head, channel-within-head)). Within a head the head_dim channels SHARE
        alpha_t, R_t (rotation), beta_t, gamma_t and the head-shared Bbar_t; they
        differ ONLY through the per-channel SSM input x. B,C are shared across ALL
        heads (MVA). y_t = Cbar_t^T h_t per channel.
        """
        Bt, L, di = x.shape
        Nc = self.complex_dim
        H = self.n_heads
        P = self.head_dim
        device = x.device

        # x reshaped to [B, n_heads, head_dim] per step (di = H*P).
        x_h = x.view(Bt, L, H, P)

        # alpha_t = exp(dt * A_real)              [B,L,H]
        alpha = torch.exp(dt * A_real)
        # gamma_t = lambda * dt                   [B,L,H]
        gamma = lam * dt
        # beta_t  = (1 - lambda) * dt * alpha     [B,L,H]
        beta = (1.0 - lam) * dt * alpha
        # cos/sin of rotation angle phi          [B,L,H,Nc]
        cos = torch.cos(phi)
        sin = torch.sin(phi)

        # state h_{t-1}: [B, H, P, Nc, 2]
        h = torch.zeros(Bt, H, P, Nc, 2, dtype=x.dtype, device=device)
        # previous state-input term Bbar_{t-1} * x_{t-1}, used by the beta term.
        # v_t := Bbar_t * x_t  has shape [B, H, P, Nc, 2] (Bbar shared over H,P).
        v_prev = torch.zeros(Bt, H, P, Nc, 2, dtype=x.dtype, device=device)

        ys = []
        for t in range(L):
            x_t = x_h[:, t, :, :].view(Bt, H, P, 1, 1)           # [B,H,P,1,1]
            Bbar_t = Bbar[:, t, :, :].view(Bt, 1, 1, Nc, 2)      # [B,1,1,Nc,2] (shared H,P)
            Cbar_t = Cbar[:, t, :, :].view(Bt, 1, 1, Nc, 2)      # [B,1,1,Nc,2]
            v_t = Bbar_t * x_t                                   # [B,H,P,Nc,2] state-input
            cos_t = cos[:, t, :, :].view(Bt, H, 1, Nc)           # [B,H,1,Nc] (shared over P)
            sin_t = sin[:, t, :, :].view(Bt, H, 1, Nc)
            alpha_t = alpha[:, t, :].view(Bt, H, 1, 1, 1)        # [B,H,1,1,1]
            beta_t = beta[:, t, :].view(Bt, H, 1, 1, 1)
            gamma_t = gamma[:, t, :].view(Bt, H, 1, 1, 1)

            # R_t @ h_{t-1}  and  R_t @ v_{t-1}   (block-diag 2x2 rotation per coord)
            Rh = apply_rotation(h, cos_t, sin_t)                 # [B,H,P,Nc,2]
            Rv_prev = apply_rotation(v_prev, cos_t, sin_t)       # [B,H,P,Nc,2]

            # Eq (25): h_t = alpha*R h_{t-1} + beta*R (Bbar_{t-1} x_{t-1}) + gamma*(Bbar_t x_t)
            h = alpha_t * Rh + beta_t * Rv_prev + gamma_t * v_t  # [B,H,P,Nc,2]

            # y_t = Cbar_t^T h_t  =  sum over complex coords of (Cr*hr + Ci*hi)
            #   (real part of complex inner product; Cbar already = [C, -Chat])
            y_t = (Cbar_t * h).sum(dim=(3, 4)).reshape(Bt, di)   # [B,d_inner]
            ys.append(y_t)

            v_prev = v_t  # carry for next step's beta term

        return torch.stack(ys, dim=1)                            # [B,L,d_inner]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] -> [B, L, d].

        If ``self.residual`` is True the input residual ``out + x`` is added
        here (Mamba-1/-2 drop-in). If False (the Llama-style ``Mamba3Model``
        path) the bare ``out`` is returned and the caller adds the residual
        around the pre-norm.
        """
        residual = x
        B_, L, _ = x.shape

        # input projection + split into main path & gate
        xz = self.in_proj(x)                                     # [B,L,2*d_inner]
        x_main, z = xz.chunk(2, dim=-1)                          # each [B,L,d_inner]

        # NO conv1d and NO SiLU on the SSM input: the paper (Sec 3.4, line 678)
        # obviates the short conv "AND its accompanying activation function", so
        # x_main feeds the projections and the scan directly. (RESOLVED.)
        x_in = x_main                                            # [B,L,d_inner]

        # SSM parameter projections (all data-dependent)
        proj = self.x_proj(x_in)
        dt_lr, A_mod, theta, u_lam, Br, Bi, Cr, Ci = torch.split(
            proj,
            [
                self.dt_rank,
                self.n_heads,        # A_real log-rate modulation (per head)
                self.complex_dim,    # theta (rotation freq, head-shared)
                self.n_heads,        # lambda gate u_t (per head)
                self.complex_dim,    # B real
                self.complex_dim,    # B imag (Bhat)
                self.complex_dim,    # C real
                self.complex_dim,    # C imag (Chat)
            ],
            dim=-1,
        )

        # dt = softplus(dt_proj(dt_lr))  > 0     [B,L,n_heads]  (PER HEAD)
        dt = F.softplus(self.dt_proj(dt_lr))

        # A_real < 0 : per-head scalar base rate (exp(A_log), [n_heads]) made
        # data-dependent by A_mod (per head), then negated through softplus so
        # alpha=exp(dt*A_real) in (0,1]. A_real shape [B,L,n_heads]. The real part
        # of A is a SCALAR per head (Eq 9/17, line ~1431); the rotation R_t (built
        # from theta) carries the per-coordinate structure.
        base_rate = torch.exp(self.A_log)                        # [n_heads]
        A_real = -F.softplus(A_mod + base_rate.view(1, 1, -1))   # [B,L,n_heads] < 0

        # theta -> rotation angle phi = dt * theta. dt is PER HEAD [B,L,H], theta
        # is head-shared [B,L,Nc]; the exact per-head angle is their outer product
        # phi[b,l,h,c] = dt[b,l,h] * theta[b,l,c]  ->  [B,L,H,Nc]. No mean approx.
        phi = dt.unsqueeze(-1) * theta.unsqueeze(-2)             # [B,L,H,Nc]

        # lambda_t = sigmoid(u_t)  in (0,1)      [B,L,n_heads]  (Appendix A.3 default)
        lam = torch.sigmoid(u_lam)

        # BCNorm (RMSNorm) then add channel-wise all-ones-init biases (Appendix F):
        Br = self.B_norm(Br) + self.B_bias                       # [B,L,complex_dim]
        Bi = self.Bhat_norm(Bi) + self.Bhat_bias
        Cr = self.C_norm(Cr) + self.C_bias
        Ci = self.Chat_norm(Ci) + self.Chat_bias

        # Build Bbar = [B ; Bhat], Cbar = [C ; -Chat] per complex coord (Prop 2).
        Bbar = torch.stack((Br, Bi), dim=-1)                     # [B,L,complex_dim,2]
        Cbar = torch.stack((Cr, -Ci), dim=-1)                    # [B,L,complex_dim,2]

        # the complex exponential-trapezoidal scan (Eq 25)
        y = self._mamba3_scan(x_in, dt, A_real, phi, lam, Bbar, Cbar)  # [B,L,d_inner]

        # D skip + gated output (phi(Y,Z) = Y (.) SiLU(Z), Appendix C, Eq C)
        y = y + x_in * self.D.view(1, 1, -1)
        y = y * F.silu(z)

        # output projection + (optional) block residual
        out = self.out_proj(y)                                   # [B,L,d]
        if self.residual:
            return out + residual
        return out


class Mamba3Block(nn.Module):
    """One Llama-style layer of the Mamba-3 architecture: a Mamba-3 **mixer**
    sub-block followed by a **SwiGLU MLP** sub-block, each wrapped in pre-norm +
    residual (Section 3.4: "the overall architecture follows Llama, alternating
    Mamba-3 and SwiGLU blocks with pre-norm"):

        h = h + Mamba3Mixer( RMSNorm_mix(h) )
        h = h + SwiGLU_MLP ( RMSNorm_mlp(h) )

    The mixer is the existing ``Mamba3Layer`` constructed with ``residual=False``
    (its own internal residual is OFF; the residual is added here around the
    pre-norm, the Llama convention). This is the faithful Mamba-3 block: the
    mixer keeps its full internal scaffold (in_proj/scan/gate/out_proj, no conv,
    no SSM-input SiLU — Section 3.4) and the SwiGLU MLP is the new addition that
    the mixer-only reference was missing.
    """

    def __init__(
        self,
        d: int,
        state_dim: int = 128,
        dt_rank: Optional[int] = None,
        expand_factor: int = 2,
        head_dim: int = 64,
        d_ff: Optional[int] = None,
        mlp_ratio: int = 2,
        rmsnorm_eps: float = 1e-5,
    ):
        super().__init__()
        self.mixer_norm = RMSNorm(d, eps=rmsnorm_eps)
        self.mixer = Mamba3Layer(
            d,
            state_dim=state_dim,
            dt_rank=dt_rank,
            expand_factor=expand_factor,
            head_dim=head_dim,
            rmsnorm_eps=rmsnorm_eps,
            residual=False,                       # residual added at block level
        )
        self.mlp_norm = RMSNorm(d, eps=rmsnorm_eps)
        self.mlp = SwiGLU_MLP(d, d_ff=d_ff, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.mixer(self.mixer_norm(x))    # pre-norm Mamba-3 mixer + residual
        h = h + self.mlp(self.mlp_norm(h))        # pre-norm SwiGLU MLP + residual
        return h


class Mamba3Model(nn.Module):
    """Full Llama-style Mamba-3 model: token+pos embeddings -> ``nl``
    ``Mamba3Block`` layers (each = a Mamba-3 mixer sub-block + a SwiGLU MLP
    sub-block, pre-norm + residual) -> final RMSNorm -> linear head on the LAST
    token. This is the COMPLETE Mamba-3 architecture of Section 3.4 ("alternating
    Mamba-3 and SwiGLU blocks with pre-norm"); the previous reference was
    mixer-only and is now superseded by this faithful Llama-style stack.

    ALTERNATION CONVENTION (RESOLVED — see MAMBA3_REFERENCE.md §1.6): ``nl`` is
    the number of Mamba-3 **mixer** blocks, and EACH is paired with a SwiGLU MLP,
    so there are ``nl`` mixers + ``nl`` SwiGLU MLPs ("24 of each" at the 1.5B
    scale). This is the Mamba-2 convention the paper follows (Appendix D:
    "procedures follow Dao and Gu 2024"), and is the convention that makes the
    canonical config (d_model=2048, nl=24, MLP inner 4096, Llama-3.1 vocab,
    tied embeddings) land at ~1.5B. The previous mixer-only stack was 0.661B.

    The post-gate RMSNorm of Mamba-2 is removed (BCNorm stabilizes, Section 3.4);
    a single final RMSNorm precedes the LM head.

    Args mirror MambaModel: p (head width), ntok (vocab), seq_len, d, nl. Extra
    SSM knobs (state_dim, expand, dt_rank, head_dim) and the MLP inner dim
    (``d_ff`` / ``mlp_ratio``, default ``2*d`` = 4096 at d=2048) are
    config-parametrized.
    """

    def __init__(
        self,
        p: int = 97,
        ntok: int = 99,
        seq_len: int = 8,
        d: int = 128,
        nl: int = 2,
        state_dim: int = 128,
        expand_factor: int = 2,
        dt_rank: Optional[int] = None,
        head_dim: int = 64,
        d_ff: Optional[int] = None,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        self.tok = nn.Embedding(ntok, d)
        self.pos = nn.Embedding(seq_len, d)
        self.layers = nn.ModuleList([
            Mamba3Block(
                d,
                state_dim=state_dim,
                dt_rank=dt_rank,
                expand_factor=expand_factor,
                head_dim=head_dim,
                d_ff=d_ff,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(nl)
        ])
        self.norm = RMSNorm(d)
        self.out = nn.Linear(d, p)
        self.register_buffer("pos_ids", torch.arange(seq_len).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok(x) + self.pos(self.pos_ids)
        for layer in self.layers:
            h = layer(h)
        return self.out(self.norm(h)[:, -1, :])
