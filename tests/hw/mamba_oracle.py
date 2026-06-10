"""tests/hw/mamba_oracle.py — runnable PyTorch ORACLE for the real Mamba (SSM)
forward+backward that the PHASE-2 L3-REAL megakernel transcribes.

WHY THIS EXISTS (the rigor substitute for an un-compilable, un-runnable .cu):
  PHASE 2 adds the REAL Mamba forward+backward as in-kernel stages of the
  persistent megakernel. That CUDA backward — in particular the SELECTIVE-SCAN
  backward (a reverse-time recurrence for dA, dB, dC, dD, ddt, dx) — is
  hand-derived and the implementing agent cannot compile or run it. So we first
  write the SAME fwd+bwd here in plain PyTorch using ONLY primitive ops — and a
  MANUAL backward (NO autograd) — then assert it matches torch.autograd's loss
  and every parameter .grad to ~1e-12 (fp64). The CUDA megakernel is then ported
  line-for-line from this verified reference, converting "is my hand-derived
  scan backward correct?" (unrunnable) into "does my manual Python match
  autograd?" (runnable now).

EXACT ARCHITECTURE TRANSCRIBED (grokking_race_v2.py, _raw_model -> MambaModel):
  MambaModel(p=97, ntok=99, seq_len=8, d=128, nl=2)            # :451-462, :477
    tok = nn.Embedding(99, 128)                                # :454
    pos = nn.Embedding(8, 128); pos_ids = arange(8)            # :454, :457
    layers = 2 x SelectiveSSMLayer(d=128)                      # :455
    norm = nn.LayerNorm(128)                                   # :456 (final)
    out  = nn.Linear(128, 97)                                  # :456 head (p=97!)
    forward: h = tok(x) + pos(pos_ids)                         # :460
             for l in layers: h = l(h)                         # :461
             return out(norm(h[:, -1, :]))                     # :462  LAST token
  Loss: F.cross_entropy(logits, targets)  (mean over batch; :745)

  SelectiveSSMLayer(d=128, state_dim=16, expand_factor=2):     # :407-449
    d_inner = d*2 = 256 ; dt_rank = max(d//16,1) = 8 ; state_dim = 16
    in_proj  = nn.Linear(128, 2*256=512, bias=False)          # :412
    conv1d   = nn.Conv1d(256,256,k=3,pad=1,groups=256,bias=True)  # :413-414  (NON-causal depthwise)
    x_proj   = nn.Linear(256, dt_rank+2*state_dim = 8+32 = 40, bias=False)  # :415
    dt_proj  = nn.Linear(8, 256, bias=True)                   # :416
    A_log    = Parameter(log(arange(1,17)).expand(256,16))    # :417-418   A = -exp(A_log)
    D        = Parameter(ones(256))                           # :419
    out_proj = nn.Linear(256, 128, bias=False)               # :420
    norm     = nn.LayerNorm(128)                              # :421 (per-layer, POST)
    forward (:442-449):
      residual = x
      xz = in_proj(x); x_main, z = xz.chunk(2, -1)            # each [B,L,256]
      x_main = SiLU(conv1d(x_main.transpose(1,2)).transpose(1,2))   # depthwise, NON-causal
      x_dbc = x_proj(x_main)                                  # [B,L,40]
      dt, B, C = x_dbc.split([8, 16, 16], -1)
      y = _selective_scan(x_main, dt_proj(dt), B, C)          # dt_proj INCLUDES its bias
      y = out_proj((y + x_main * D) * SiLU(z))
      return norm(y + residual)

  _selective_scan(x, dt, B, C)  (:422-441 — the PYTHON FALLBACK, the ground truth
  the race runs since mamba_scan_kernel.cu is absent and CPU forces is_cuda=False):
      A = -exp(A_log)                  # [256,16]
      dt = softplus(dt)                # softplus applied INSIDE the scan
      h = 0  [B, 256, 16]
      for t in 0..L-1:
        dt_t = dt[:,t,:].unsqueeze(-1)                        # [B,256,1]
        h = exp(dt_t * A) * h + (dt_t * B[:,t,:].unsqueeze(1)) * x[:,t,:].unsqueeze(-1)
        y_t = (h * C[:,t,:].unsqueeze(1)).sum(-1)             # [B,256]
      return stack(y_t)                                       # [B,L,256]

NUMERICS (each verified against autograd here, and the values the CUDA must use):
  * SiLU(x) = x * sigmoid(x); SiLU'(x) = sigmoid(x) * (1 + x*(1-sigmoid(x))).
  * softplus(x) = log(1+exp(x)); softplus'(x) = sigmoid(x).
  * Conv1d is NON-CAUSAL depthwise k=3 pad=1: y[t] = b + W0*x[t-1] + W1*x[t]
    + W2*x[t+1] (zero-pad outside [0,L)). The future-peek (W2*x[t+1]) is the #1
    index-bug site — the kernel mirror exists to catch it.
  * A = -exp(A_log): the (-exp) Jacobian folds into dA_log = dA * A  (since
    dA/dA_log = -exp(A_log) = A).
  * LayerNorm eps = 1e-5 (torch default). Backward dx uses the two
    mean-correction terms.
  * CrossEntropy MEAN over B: dlogits = (softmax - onehot) / B. Head width p=97
    (distinct from the 99-token embedding / 99-row tok table).

GRADIENT-TOPOLOGY (multi-path sums — each easy to half-implement):
  * x_main fans out to THREE consumers: x_proj (:445), the scan's x (:447),
    and the D-skip (:448 `x_main * D`). dx_main = sum of all three paths.
  * the layer input x fans out to TWO: in_proj (:443) and the end residual
    (:449 `norm(y + residual)`). dx = in_proj-path + residual-path.
  * softplus is INSIDE _selective_scan; dt_proj (with bias) is OUTSIDE. The
    scan's ddt is w.r.t. POST-softplus dt; multiply by sigmoid(pre-softplus) to
    reach dt_proj's output, then backprop dt_proj's weight/bias.

This file imports torch only; it needs NO GPU and NO built extension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Architecture constants (single source — mirrored by the C++ layout header
# mamba3_layout.cuh and asserted == the eager named_parameters() in the test).
# ---------------------------------------------------------------------------
VOCAB = 99       # ntok (num_tokens) — token range [0,98], the tok embedding rows
P_HEAD = 97      # p — head width (out = Linear(d, p)); DISTINCT from VOCAB
D_MODEL = 128    # d (MODEL_SCALES["small"])
N_LAYERS = 2     # nl
SEQ = 8          # seq_len (MambaModel default; chain_length=3 -> 1+2*3+1 = 8 toks)
EXPAND = 2       # expand_factor
D_INNER = D_MODEL * EXPAND          # 256
STATE = 16       # state_dim
DT_RANK = max(D_MODEL // 16, 1)     # 8
DBC = DT_RANK + 2 * STATE           # 40  (x_proj out width)
CONV_K = 3       # conv1d kernel size (depthwise, pad=1, NON-causal)
LN_EPS = 1e-5    # torch.nn.LayerNorm default eps


# ---------------------------------------------------------------------------
# Elementwise activations + derivatives (the REAL F.silu / F.softplus).
# ---------------------------------------------------------------------------
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def silu_grad(x: torch.Tensor) -> torch.Tensor:
    # d/dx [x*sig(x)] = sig(x) + x*sig(x)*(1-sig(x)) = sig(x)*(1 + x*(1-sig(x))).
    s = torch.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def softplus_grad(x: torch.Tensor) -> torch.Tensor:
    # d/dx log(1+exp(x)) = sigmoid(x).
    return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# LayerNorm fwd/bwd (eps=1e-5), over the last dim (size d). Affine (gamma,beta).
# (Same formulation as decoder_oracle.layernorm_*.)
# ---------------------------------------------------------------------------
def layernorm_forward(x, gamma, beta, eps=LN_EPS):
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(dim=-1, keepdim=True)
    inv_std = 1.0 / torch.sqrt(var + eps)
    xhat = xc * inv_std
    y = xhat * gamma + beta
    return y, (xhat, inv_std, gamma)


def layernorm_backward(dy, cache):
    xhat, inv_std, gamma = cache
    d = xhat.shape[-1]
    reduce_axes = tuple(range(dy.dim() - 1))
    dbeta = dy.sum(dim=reduce_axes)
    dgamma = (dy * xhat).sum(dim=reduce_axes)
    dxhat = dy * gamma
    sum_dxhat = dxhat.sum(dim=-1, keepdim=True)
    sum_dxhat_xhat = (dxhat * xhat).sum(dim=-1, keepdim=True)
    dx = inv_std * (dxhat - (sum_dxhat + xhat * sum_dxhat_xhat) / d)
    return dx, dgamma, dbeta


# ---------------------------------------------------------------------------
# Parameter spec — mirrors named_parameters() ORDER exactly (28 tensors).
# CRITICAL: PyTorch yields a module's OWN nn.Parameters before its submodules,
# so within each SelectiveSSMLayer A_log + D come BEFORE in_proj (NOT __init__
# visual order). Verified against the live module's named_parameters().
# ---------------------------------------------------------------------------
def mamba_param_spec() -> List[Tuple[str, Tuple[int, ...]]]:
    spec: List[Tuple[str, Tuple[int, ...]]] = [
        ("tok.weight", (VOCAB, D_MODEL)),
        ("pos.weight", (SEQ, D_MODEL)),
    ]
    for li in range(N_LAYERS):
        p = f"layers.{li}."
        spec += [
            (p + "A_log", (D_INNER, STATE)),
            (p + "D", (D_INNER,)),
            (p + "in_proj.weight", (2 * D_INNER, D_MODEL)),
            (p + "conv1d.weight", (D_INNER, 1, CONV_K)),
            (p + "conv1d.bias", (D_INNER,)),
            (p + "x_proj.weight", (DBC, D_INNER)),
            (p + "dt_proj.weight", (D_INNER, DT_RANK)),
            (p + "dt_proj.bias", (D_INNER,)),
            (p + "out_proj.weight", (D_MODEL, D_INNER)),
            (p + "norm.weight", (D_MODEL,)),
            (p + "norm.bias", (D_MODEL,)),
        ]
    spec += [
        ("norm.weight", (D_MODEL,)),
        ("norm.bias", (D_MODEL,)),
        ("out.weight", (P_HEAD, D_MODEL)),
        ("out.bias", (P_HEAD,)),
    ]
    return spec


def mamba_param_layout() -> Dict[str, object]:
    """Flat-buffer layout: offsets/sizes per tensor (the megakernel ABI view).

    Returns dict with: names, offsets (elements into the flat buffer), sizes,
    shapes, total (== 259425), n_tensors (== 28). The offsets follow
    named_parameters() order so the flat blob == torch.cat([p.reshape(-1) for
    _, p in model.named_parameters()]).
    """
    spec = mamba_param_spec()
    names, offsets, sizes = [], [], []
    off = 0
    for name, shape in spec:
        n = 1
        for s in shape:
            n *= s
        names.append(name)
        offsets.append(off)
        sizes.append(n)
        off += n
    return dict(names=names, offsets=offsets, sizes=sizes,
                shapes=[s for _, s in spec], total=off, n_tensors=len(spec))


@dataclass
class MambaWeights:
    """All 28 Mamba weights as named tensors. Built either from a flat buffer
    (the megakernel view) or from a torch nn module's named_parameters()."""
    tok: torch.Tensor
    pos: torch.Tensor
    layers: List[Dict[str, torch.Tensor]]
    norm_w: torch.Tensor
    norm_b: torch.Tensor
    out_w: torch.Tensor
    out_b: torch.Tensor

    @staticmethod
    def from_named(named: Dict[str, torch.Tensor]) -> "MambaWeights":
        layers = []
        for li in range(N_LAYERS):
            p = f"layers.{li}."
            layers.append(dict(
                A_log=named[p + "A_log"],
                D=named[p + "D"],
                in_w=named[p + "in_proj.weight"],
                conv_w=named[p + "conv1d.weight"],   # [d_inner,1,3]
                conv_b=named[p + "conv1d.bias"],
                x_proj_w=named[p + "x_proj.weight"],
                dt_proj_w=named[p + "dt_proj.weight"],
                dt_proj_b=named[p + "dt_proj.bias"],
                out_w=named[p + "out_proj.weight"],
                n_w=named[p + "norm.weight"], n_b=named[p + "norm.bias"],
            ))
        return MambaWeights(
            tok=named["tok.weight"], pos=named["pos.weight"], layers=layers,
            norm_w=named["norm.weight"], norm_b=named["norm.bias"],
            out_w=named["out.weight"], out_b=named["out.bias"])

    @staticmethod
    def from_flat(flat: torch.Tensor) -> "MambaWeights":
        lay = mamba_param_layout()
        named = {}
        for name, off, sz, shape in zip(lay["names"], lay["offsets"],
                                        lay["sizes"], lay["shapes"]):
            named[name] = flat[off:off + sz].reshape(shape)
        return MambaWeights.from_named(named)


# ===========================================================================
#  SELECTIVE SCAN — forward + MANUAL backward (the hardest piece).
#  Operates per (batch, channel j in [0,d_inner)) with a state vector h[STATE].
#
#  FORWARD recurrence (eager :437-440, with dt already = softplus(dt_pre)):
#    A[j,s] = -exp(A_log[j,s])
#    h_{t}[s] = exp(dt[t,j] * A[j,s]) * h_{t-1}[s] + dt[t,j] * B[t,s] * x[t,j]
#    y[t,j]   = sum_s C[t,s] * h_{t}[s]                       (h_{-1} = 0)
#
#  Let  a_t[s] = exp(dt[t,j]*A[j,s])  (the per-step decay),
#       g_t[s] = dt[t,j]*B[t,s]*x[t,j]  (the per-step input).
#  Then h_t = a_t * h_{t-1} + g_t  and  y_t = <C_t, h_t>.
#
#  BACKWARD DERIVATION (reverse-time; inputs: dy[t,j] = dL/dy[t,j]).
#  Carry an adjoint  gh[s] = dL/dh_t[s]  flowing back through time. Process t
#  from L-1 down to 0. At step t we already hold gh = dL/dh_t accumulated from
#  all FUTURE timesteps' dependence on h_t THROUGH the recurrence
#  h_{t+1}=a_{t+1} h_t + g_{t+1} (i.e. the part carried in from t+1). Add this
#  step's direct contribution from y_t = <C_t, h_t>:
#       gh[s] += C[t,s] * dy[t,j]                              (y_t depends on h_t)
#  Now gh[s] = full dL/dh_t[s]. Emit the leaf grads at t:
#       dC[t,s]  = h_t[s] * dy[t,j]                            (from y_t=<C,h_t>)
#       dB[t,s]  = dt[t,j] * x[t,j] * gh[s]                    (g_t = dt*B*x)
#       dx[t,j] += sum_s B[t,s] * dt[t,j] * gh[s]              (g_t depends on x)
#       ddt[t,j]+= sum_s ( A[j,s]*a_t[s]*h_{t-1}[s] + B[t,s]*x[t,j] ) * gh[s]
#                  (a_t = exp(dt*A) -> da_t/ddt = A*a_t; g_t -> dg/ddt = B*x)
#       dA[j,s] += dt[t,j]*a_t[s]*h_{t-1}[s] * gh[s]           (da_t/dA = dt*a_t)
#  Propagate the adjoint to the previous state via h_t = a_t h_{t-1} + g_t:
#       gh[s] <- a_t[s] * gh[s]                                (= dL/dh_{t-1}[s])
#  After t=0, gh = dL/dh_{-1} = 0 contribution (h_{-1}=0, discarded). Finally
#       dA_log[j,s] = dA[j,s] * A[j,s]   (A = -exp(A_log) -> dA/dA_log = A).
#
#  This MATCHES csrc/scan/mamba_scan_adapter.cuh::scan_backward_kernel (cross-
#  checked): there grad_C=h_t*gy, grad_h+=cv*gy, grad_B=dt*x*grad_h,
#  grad_x+=B*dt*grad_h, grad_dt+=(A*A_bar*h_{t-1}+B*x)*grad_h,
#  grad_A_acc+=dt*A*A_bar*h_{t-1}*grad_h, grad_h=A_bar*grad_h, then
#  grad_A_log = grad_A_acc (it stores dA already folded with A in the +=,
#  because A_bar uses A and the A-derivative term carries the extra A). NOTE the
#  adapter folds (-exp) by carrying A inside the accumulation; here we fold it
#  explicitly as dA_log = dA*A at the end — algebraically identical.
# ===========================================================================
def selective_scan_forward(x, dt_pre, B, C, A_log):
    """x,dt_pre: [Bt,L,d_inner]; B,C: [Bt,L,state]; A_log: [d_inner,state].
    dt_pre is dt_proj(dt) BEFORE softplus (softplus is applied here, as eager
    does inside _selective_scan). Returns (y, cache)."""
    Bt, L, di = x.shape
    A = -torch.exp(A_log)                       # [di, state]
    dt = softplus(dt_pre)                        # [Bt,L,di]
    h = torch.zeros(Bt, di, STATE, dtype=x.dtype, device=x.device)
    ys = []
    h_hist = []   # h_hist[t] = h AFTER step t (i.e. h_t)   [Bt,di,state]
    for t in range(L):
        dt_t = dt[:, t, :].unsqueeze(-1)         # [Bt,di,1]
        a_t = torch.exp(dt_t * A.unsqueeze(0))   # [Bt,di,state]
        g_t = (dt_t * B[:, t, :].unsqueeze(1)) * x[:, t, :].unsqueeze(-1)
        h = a_t * h + g_t
        h_hist.append(h.clone())
        ys.append((h * C[:, t, :].unsqueeze(1)).sum(-1))   # [Bt,di]
    y = torch.stack(ys, dim=1)                   # [Bt,L,di]
    cache = dict(x=x, dt_pre=dt_pre, dt=dt, B=B, C=C, A_log=A_log, A=A,
                 h_hist=h_hist)
    return y, cache


def selective_scan_backward(dy, cache):
    """dy: [Bt,L,d_inner] = dL/dy. Returns (dx, ddt_pre, dB, dC, dA_log).
    ddt_pre is the grad w.r.t. the PRE-softplus dt (dt_proj output) — softplus's
    Jacobian (sigmoid) is folded in here so callers backprop dt_proj directly."""
    x, dt_pre, dt = cache["x"], cache["dt_pre"], cache["dt"]
    B, C, A = cache["B"], cache["C"], cache["A"]
    h_hist = cache["h_hist"]
    Bt, L, di = x.shape
    dx = torch.zeros_like(x)
    ddt = torch.zeros_like(dt)                   # grad w.r.t. POST-softplus dt
    dB = torch.zeros_like(B)
    dC = torch.zeros_like(C)
    dA = torch.zeros_like(A)                     # grad w.r.t. A (=-exp(A_log))
    gh = torch.zeros(Bt, di, STATE, dtype=x.dtype, device=x.device)  # dL/dh_t carry
    for t in reversed(range(L)):
        dt_t = dt[:, t, :].unsqueeze(-1)         # [Bt,di,1]
        a_t = torch.exp(dt_t * A.unsqueeze(0))   # [Bt,di,state]
        h_t = h_hist[t]                          # [Bt,di,state]
        h_tm1 = h_hist[t - 1] if t > 0 else torch.zeros_like(h_t)
        dy_t = dy[:, t, :].unsqueeze(-1)         # [Bt,di,1]
        # y_t = <C_t, h_t> contributes to dC and to dL/dh_t:
        dC[:, t, :] = (h_t * dy_t).sum(1)        # sum over di? NO — per (b,state)
        # ^ careful: y[t,j] = sum_s C[t,s]*h_t[j,s]; C is shared across j (per b).
        #   dC[b,t,s] = sum_j h_t[b,j,s]*dy[b,t,j]. So sum over di (dim=1). OK.
        gh = gh + C[:, t, :].unsqueeze(1) * dy_t  # [Bt,di,state]  full dL/dh_t
        # leaf grads at t:
        Bt_t = B[:, t, :].unsqueeze(1)           # [Bt,1,state]
        x_t = x[:, t, :].unsqueeze(-1)           # [Bt,di,1]
        # dB[b,t,s] = sum_j dt[b,t,j]*x[b,t,j]*gh[b,j,s]
        dB[:, t, :] = (dt_t * x_t * gh).sum(1)
        # dx[b,t,j] = sum_s B[b,t,s]*dt[b,t,j]*gh[b,j,s]
        dx[:, t, :] = (Bt_t * dt_t * gh).sum(-1)
        # ddt[b,t,j] = sum_s ( A[j,s]*a_t*h_{t-1} + B[t,s]*x[t,j] ) * gh
        ddt[:, t, :] = ((A.unsqueeze(0) * a_t * h_tm1 + Bt_t * x_t) * gh).sum(-1)
        # dA[j,s] += sum_b dt*a_t*h_{t-1}*gh   (A shared across batch)
        dA = dA + (dt_t * a_t * h_tm1 * gh).sum(0)
        # propagate adjoint to h_{t-1}: h_t = a_t h_{t-1} + g_t
        gh = a_t * gh
    # A = -exp(A_log) -> dA_log = dA * dA/dA_log = dA * (-exp(A_log)) = dA * A.
    dA_log = dA * A
    # softplus: dt = softplus(dt_pre) -> ddt_pre = ddt * sigmoid(dt_pre).
    ddt_pre = ddt * softplus_grad(dt_pre)
    return dx, ddt_pre, dB, dC, dA_log


# ===========================================================================
#  Depthwise NON-CAUSAL conv1d (k=3, pad=1) fwd/bwd, per channel over time.
#    fwd: y[b,t,c] = conv_b[c] + sum_{k=0..2} W[c,0,k] * x[b, t+k-1, c]
#         (x out of [0,L) is zero — pad=1; k=0->t-1, k=1->t, k=2->t+1).
#    bwd: dW[c,0,k] = sum_{b,t} dy[b,t,c]*x[b,t+k-1,c]
#         db[c]     = sum_{b,t} dy[b,t,c]
#         dx[b,tt,c]= sum_{k: tt=t+k-1 in-range} W[c,0,k]*dy[b,t,c]
#                   = sum_{k} W[c,0,k]*dy[b, tt-k+1, c]  (dy index in [0,L))
# ===========================================================================
def conv1d_forward(x, W, b):
    """x: [Bt,L,d_inner]; W: [d_inner,1,3]; b: [d_inner]. Returns y [Bt,L,di]."""
    Bt, L, di = x.shape
    y = torch.zeros_like(x)
    for k in range(CONV_K):
        shift = k - 1   # tap k reads x[t+shift]
        Wk = W[:, 0, k]                          # [di]
        # x shifted so that y[t] += Wk * x[t+shift]; zero-pad ends.
        xs = torch.zeros_like(x)
        if shift == 0:
            xs = x
        elif shift < 0:           # read x[t-1]: valid for t>=1
            xs[:, -shift:, :] = x[:, :L + shift, :]
        else:                     # shift>0, read x[t+1]: valid for t<=L-2
            xs[:, :L - shift, :] = x[:, shift:, :]
        y = y + xs * Wk.view(1, 1, di)
    y = y + b.view(1, 1, di)
    return y


def conv1d_backward(dy, x, W):
    """dy,x: [Bt,L,di]; W: [di,1,3]. Returns (dx, dW, db)."""
    Bt, L, di = x.shape
    dx = torch.zeros_like(x)
    dW = torch.zeros_like(W)
    db = dy.sum(dim=(0, 1))                      # [di]
    for k in range(CONV_K):
        shift = k - 1
        Wk = W[:, 0, k].view(1, 1, di)           # [1,1,di]
        # dW[c,0,k] = sum_{b,t} dy[b,t,c]*x[b,t+shift,c]
        xs = torch.zeros_like(x)
        if shift == 0:
            xs = x
        elif shift < 0:
            xs[:, -shift:, :] = x[:, :L + shift, :]
        else:
            xs[:, :L - shift, :] = x[:, shift:, :]
        dW[:, 0, k] = (dy * xs).sum(dim=(0, 1))
        # dx contribution: x[t+shift] receives Wk*dy[t]; i.e. dx[tt] += Wk*dy[tt-shift]
        dyk = torch.zeros_like(dy)
        if shift == 0:
            dyk = dy
        elif shift < 0:           # tt = t-1 -> t = tt+1 ; dx[tt]+=Wk*dy[tt+1]
            dyk[:, :L + shift, :] = dy[:, -shift:, :]
        else:                     # tt = t+1 -> t = tt-1 ; dx[tt]+=Wk*dy[tt-1]
            dyk[:, shift:, :] = dy[:, :L - shift, :]
        dx = dx + dyk * Wk
    return dx, dW, db


# ===========================================================================
#  Linear fwd/bwd helpers: y = x @ W^T (+ b). W [out,in]. x [...,in].
# ===========================================================================
def linear_forward(x, W, b=None):
    y = x @ W.t()
    if b is not None:
        y = y + b
    return y


def linear_backward(dy, x, W, has_bias):
    """dy [...,out], x [...,in], W [out,in]. Returns (dx, dW, db|None).
    Reduces dW/db over all leading axes."""
    in_dim = W.shape[1]
    out_dim = W.shape[0]
    x2 = x.reshape(-1, in_dim)
    dy2 = dy.reshape(-1, out_dim)
    dW = dy2.t() @ x2                            # [out,in]
    db = dy2.sum(0) if has_bias else None
    dx = (dy @ W).reshape(x.shape)
    return dx, dW, db


# ===========================================================================
#  ONE SelectiveSSMLayer forward (caches everything the manual backward needs).
# ===========================================================================
def ssm_layer_forward(L: Dict[str, torch.Tensor], x):
    """x: [Bt,Lseq,d_model]. Returns (out, cache). Mirrors eager :442-449."""
    residual = x
    xz = linear_forward(x, L["in_w"])            # [Bt,L,2*d_inner]
    x_main_raw, z = xz.split(D_INNER, dim=-1)    # each [Bt,L,d_inner]
    # depthwise NON-causal conv1d then SiLU
    conv = conv1d_forward(x_main_raw, L["conv_w"], L["conv_b"])  # [Bt,L,d_inner]
    x_main = silu(conv)
    # x_proj -> dt, B, C
    x_dbc = linear_forward(x_main, L["x_proj_w"])               # [Bt,L,40]
    dt_raw, Bmat, Cmat = x_dbc.split([DT_RANK, STATE, STATE], dim=-1)
    dt_pre = linear_forward(dt_raw, L["dt_proj_w"], L["dt_proj_b"])  # [Bt,L,d_inner]
    y_scan, scan_cache = selective_scan_forward(x_main, dt_pre, Bmat, Cmat,
                                                L["A_log"])
    # skip + gate: y = (y_scan + x_main * D) * silu(z)
    sz = silu(z)
    y_skip = y_scan + x_main * L["D"].view(1, 1, D_INNER)
    y_gated = y_skip * sz
    y_out = linear_forward(y_gated, L["out_w"])  # [Bt,L,d_model]
    r = y_out + residual
    out, ln_cache = layernorm_forward(r, L["n_w"], L["n_b"])
    cache = dict(x=x, xz=xz, x_main_raw=x_main_raw, z=z, conv=conv,
                 x_main=x_main, x_dbc=x_dbc, dt_raw=dt_raw, Bmat=Bmat,
                 Cmat=Cmat, dt_pre=dt_pre, y_scan=y_scan, scan_cache=scan_cache,
                 sz=sz, y_skip=y_skip, y_gated=y_gated, y_out=y_out, r=r,
                 ln_cache=ln_cache)
    return out, cache


def ssm_layer_backward(L: Dict[str, torch.Tensor], cache, dout):
    """dout: [Bt,Lseq,d_model] = dL/dout. Returns (dx, grads_dict) where
    grads_dict has the 11 per-layer parameter grads."""
    g = {}
    # out = LayerNorm(r) -> dr, dnorm_w, dnorm_b
    dr, dn_w, dn_b = layernorm_backward(dout, cache["ln_cache"])
    g["norm.weight"] = dn_w
    g["norm.bias"] = dn_b
    # r = y_out + residual(=x) -> dy_out = dr ; dx_residual = dr
    dy_out = dr
    dx_residual = dr.clone()
    # y_out = y_gated @ out_w^T -> dW_out, dy_gated
    dy_gated, dW_out, _ = linear_backward(dy_out, cache["y_gated"], L["out_w"], False)
    g["out_proj.weight"] = dW_out
    # y_gated = y_skip * sz -> dy_skip = dy_gated*sz ; dsz = dy_gated*y_skip
    sz = cache["sz"]
    dy_skip = dy_gated * sz
    dsz = dy_gated * cache["y_skip"]
    # sz = silu(z) -> dz = dsz * silu'(z)
    dz = dsz * silu_grad(cache["z"])
    # y_skip = y_scan + x_main * D -> dy_scan = dy_skip ; dD path + dx_main path
    dy_scan = dy_skip
    Dpar = L["D"].view(1, 1, D_INNER)
    dx_main = dy_skip * Dpar                      # path: D-skip
    g["D"] = (dy_skip * cache["x_main"]).sum(dim=(0, 1))   # [d_inner]
    # selective scan backward -> dx_main(scan path), ddt_pre, dB, dC, dA_log
    dx_scan, ddt_pre, dBmat, dCmat, dA_log = selective_scan_backward(
        dy_scan, cache["scan_cache"])
    g["A_log"] = dA_log
    dx_main = dx_main + dx_scan                   # path: scan x
    # dt_pre = dt_proj(dt_raw) -> dW_dt, db_dt, ddt_raw
    ddt_raw, dW_dt, db_dt = linear_backward(ddt_pre, cache["dt_raw"],
                                            L["dt_proj_w"], True)
    g["dt_proj.weight"] = dW_dt
    g["dt_proj.bias"] = db_dt
    # x_dbc = concat[dt_raw, Bmat, Cmat] -> dx_dbc
    dx_dbc = torch.cat([ddt_raw, dBmat, dCmat], dim=-1)        # [Bt,L,40]
    # x_dbc = x_proj(x_main) -> dW_xproj, dx_main path 3
    dx_main3, dW_xproj, _ = linear_backward(dx_dbc, cache["x_main"],
                                            L["x_proj_w"], False)
    g["x_proj.weight"] = dW_xproj
    dx_main = dx_main + dx_main3                  # path: x_proj
    # x_main = silu(conv) -> dconv = dx_main * silu'(conv)
    dconv = dx_main * silu_grad(cache["conv"])
    # conv1d backward -> dx_main_raw, dW_conv, db_conv
    dx_main_raw, dW_conv, db_conv = conv1d_backward(dconv, cache["x_main_raw"],
                                                    L["conv_w"])
    g["conv1d.weight"] = dW_conv
    g["conv1d.bias"] = db_conv
    # xz = [x_main_raw | z] -> dxz ; xz = in_proj(x) -> dW_in, dx_inproj
    dxz = torch.cat([dx_main_raw, dz], dim=-1)                # [Bt,L,2*d_inner]
    dx_inproj, dW_in, _ = linear_backward(dxz, cache["x"], L["in_w"], False)
    g["in_proj.weight"] = dW_in
    # x fans out to in_proj AND residual:
    dx = dx_inproj + dx_residual
    return dx, g


# ===========================================================================
#  FULL MODEL forward + manual backward.
# ===========================================================================
def mamba_forward(W: MambaWeights, tokens: torch.Tensor, targets: torch.Tensor):
    """tokens: [B, SEQ] int64 in [0,VOCAB). targets: [B] int64 in [0,P_HEAD).
    Returns (loss_scalar, cache). loss = mean-over-batch CE on the LAST-position
    logits (head width P_HEAD)."""
    Bt, S = tokens.shape
    device = W.tok.device
    pos_ids = torch.arange(S, device=device)
    h = W.tok[tokens] + W.pos[pos_ids].unsqueeze(0)           # [B,S,d]
    layer_caches = []
    for li in range(N_LAYERS):
        h, lc = ssm_layer_forward(W.layers[li], h)
        layer_caches.append(lc)
    h_last = h[:, -1, :]                                       # [B,d]
    hn, norm_cache = layernorm_forward(h_last, W.norm_w, W.norm_b)
    logits = hn @ W.out_w.t() + W.out_b                        # [B,P_HEAD]
    m = logits.max(dim=-1, keepdim=True).values
    logz = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))
    nll = logz - logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    loss = nll.mean()
    cache = dict(Bt=Bt, S=S, tokens=tokens, targets=targets,
                 layer_caches=layer_caches, h_last=h_last, hn=hn,
                 norm_cache=norm_cache, logits=logits)
    return loss, cache


def mamba_backward(W: MambaWeights, cache) -> Dict[str, torch.Tensor]:
    Bt, S = cache["Bt"], cache["S"]
    targets = cache["targets"]
    tokens = cache["tokens"]
    grads: Dict[str, torch.Tensor] = {}
    # CE backward: dlogits = (softmax - onehot)/B
    logits = cache["logits"]
    m = logits.max(dim=-1, keepdim=True).values
    e = torch.exp(logits - m)
    sm = e / e.sum(dim=-1, keepdim=True)
    onehot = torch.zeros_like(sm)
    onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    dlogits = (sm - onehot) / Bt                              # [B,P_HEAD]
    # head: logits = hn @ out_w^T + out_b
    hn = cache["hn"]
    grads["out.weight"] = dlogits.t() @ hn                    # [P_HEAD,d]
    grads["out.bias"] = dlogits.sum(dim=0)
    dhn = dlogits @ W.out_w                                   # [B,d]
    # final norm backward
    dh_last, dnorm_w, dnorm_b = layernorm_backward(dhn, cache["norm_cache"])
    grads["norm.weight"] = dnorm_w
    grads["norm.bias"] = dnorm_b
    # upstream into last layer output: only LAST position has grad.
    dh = torch.zeros(Bt, S, D_MODEL, device=hn.device, dtype=hn.dtype)
    dh[:, -1, :] = dh_last
    # per-layer backward (reverse)
    for li in reversed(range(N_LAYERS)):
        dh, lg = ssm_layer_backward(W.layers[li], cache["layer_caches"][li], dh)
        for k, v in lg.items():
            grads[f"layers.{li}.{k}"] = v
    # embedding backward: h0 = tok[tokens] + pos[pos_ids]
    dtok = torch.zeros_like(W.tok)
    dtok.index_add_(0, tokens.reshape(-1), dh.reshape(-1, D_MODEL))
    grads["tok.weight"] = dtok
    grads["pos.weight"] = dh.sum(dim=0)                       # [S,d]
    return grads


def oracle_loss_and_grads(named_params: Dict[str, torch.Tensor],
                          tokens: torch.Tensor, targets: torch.Tensor):
    """Run the manual fwd+bwd over a dict of named parameter tensors.
    Returns (loss_float, grads_named) — the 28 Mamba tensors."""
    W = MambaWeights.from_named(named_params)
    loss, cache = mamba_forward(W, tokens, targets)
    grads = mamba_backward(W, cache)
    return loss.item(), grads
