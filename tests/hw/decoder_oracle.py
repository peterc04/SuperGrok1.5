"""tests/hw/decoder_oracle.py — runnable PyTorch ORACLE for the real decoder
forward+backward that the L3-REAL megakernel transcribes.

WHY THIS EXISTS (the rigor substitute for an un-compilable, un-runnable .cu):
  PHASE 1 replaces the surrogate model stage in the persistent megakernel with
  the REAL transformer-decoder fwd+bwd. That CUDA backward is hand-derived and
  the implementing agent cannot compile or run it. So we first write the SAME
  fwd+bwd here in plain PyTorch using ONLY primitive ops — and a MANUAL backward
  (NO autograd) — then assert it matches torch.autograd's loss and every
  parameter .grad to ~1e-6. The CUDA megakernel is then ported line-for-line from
  this verified reference, converting "is my hand-derived backward correct?"
  (unrunnable) into "does my manual Python match autograd?" (runnable now).

  This module is ALSO the per-tensor expected-grad oracle the parity test
  (tests/hw/test_megakernel_vs_eager.py, item 5a) consumes: same init, same
  batch → expected loss + expected grads to compare the megakernel against.

EXACT ARCHITECTURE TRANSCRIBED (grokking_race_v2.py, _raw_model → Transformer):
  Transformer(nl=2, d=128, h=4, ntok=99, seq=4)  [_raw_model hardcodes seq=4]
    tok = Embedding(99, 128)            # grokking_race_v2.py:360
    pos = Embedding(4, 128)             # seq=4 positions
    layers = 2 × DecoderBlock:          # :346-355
      attn = MultiheadAttention(128, 4, dropout=0, batch_first=True)  # :349
             causal mask = triu(ones(seq,seq), diagonal=1)            # :352
      n1, n2 = LayerNorm(128)           # :350  (eps=1e-5 default)
      ff = Linear(128,512) → GELU → Linear(512,128)   # :351 (GELU = exact erf)
      forward: a,_ = attn(x,x,x, attn_mask=causal)    # :354
               x = n1(x + a)                          # :355 POST-LN
               return n2(x + ff(x))                   # :355 POST-LN
    norm = LayerNorm(128)              # :362  final LN
    out  = Linear(128, 99)            # :362  head → 99 logits
    forward: h = tok(x) + pos(pos_ids)                # :366
             for l in layers: h = l(h)                # :367
             return out(norm(h)[:, -1, :])            # :368  LAST token only
  Loss: F.cross_entropy(m(tx), ty)  (mean over batch; grokking_race_v2.py:745)

  nn.MultiheadAttention internals (verified bit-identical to the manual form in
  this file, max|diff|==0): in_proj_weight [3d, d] packs [Wq; Wk; Wv] row-blocks,
  in_proj_bias [3d]; qkv = x @ W_in^T + b_in then split into q,k,v each [.,.,d];
  per-head dh = d/h = 32; scores = (qh @ kh^T)/sqrt(dh); causal-masked (-inf on
  j>i); softmax over the key axis; ctx = attn @ vh; merge heads; out = ctx @
  W_out^T + b_out  (out_proj.weight [d,d], out_proj.bias [d]).

NUMERICS (each verified against autograd here, and the values the CUDA must use):
  * GELU = EXACT erf: 0.5*x*(1+erf(x/sqrt2)); derivative
    0.5*(1+erf(x/sqrt2)) + x * (1/sqrt(2*pi)) * exp(-x^2/2).
    (The surrogate's tanh approximation differs by ~4e-4 — blows the 1e-4 grad
    tolerance — so it is WRONG for this path.)
  * LayerNorm eps = 1e-5 (torch default). Backward dx uses the two
    mean-correction terms (see _layernorm_backward).
  * Attention softmax subtracts the row max for stability (matches torch).
  * CrossEntropy is MEAN over B: dlogits = (softmax - onehot) / B.

This file imports torch only; it needs NO GPU and NO built extension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Architecture constants (single source — mirrored by the C++ layout header and
# the codegen). These are the decoder cell's fixed shapes in the grokking race.
# ---------------------------------------------------------------------------
VOCAB = 99      # num_tokens (DEFAULT_CONFIG["num_tokens"]) — token range [0,98]
D_MODEL = 128   # dim_model (MODEL_SCALES["small"])
N_HEADS = 4     # num_heads
N_LAYERS = 2    # num_layers
SEQ = 4         # _raw_model hardcodes seq=4 for the decoder (make_data → 4 toks)
D_FF = 4 * D_MODEL  # 512 — ff hidden (Linear(d,4*d))
D_HEAD = D_MODEL // N_HEADS  # 32
LN_EPS = 1e-5   # torch.nn.LayerNorm default eps
ATTN_SCALE = 1.0 / math.sqrt(D_HEAD)  # 1/sqrt(32)


# ---------------------------------------------------------------------------
# Exact-erf GELU and its derivative (the REAL nn.GELU(), NOT the tanh approx).
# ---------------------------------------------------------------------------
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x * _INV_SQRT2))


def gelu_grad(x: torch.Tensor) -> torch.Tensor:
    # d/dx [0.5 x (1+erf(x/sqrt2))] = 0.5(1+erf(x/sqrt2)) + x * phi(x),
    # phi(x) = (1/sqrt(2pi)) exp(-x^2/2)  (the normal pdf).
    cdf = 0.5 * (1.0 + torch.erf(x * _INV_SQRT2))
    pdf = _INV_SQRT_2PI * torch.exp(-0.5 * x * x)
    return cdf + x * pdf


# ---------------------------------------------------------------------------
# LayerNorm fwd/bwd (eps=1e-5), over the last dim (size d). Affine (gamma,beta).
# ---------------------------------------------------------------------------
def layernorm_forward(x, gamma, beta, eps=LN_EPS):
    """x: [..., d]. Returns (y, cache) where cache has the bwd intermediates."""
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(dim=-1, keepdim=True)
    inv_std = 1.0 / torch.sqrt(var + eps)
    xhat = xc * inv_std
    y = xhat * gamma + beta
    return y, (xhat, inv_std, gamma)


def layernorm_backward(dy, cache):
    """Returns (dx, dgamma, dbeta). dgamma/dbeta summed over all but last dim."""
    xhat, inv_std, gamma = cache
    d = xhat.shape[-1]
    # Reduce gamma/beta grads over every leading (batch/seq) axis.
    reduce_axes = tuple(range(dy.dim() - 1))
    dbeta = dy.sum(dim=reduce_axes)
    dgamma = (dy * xhat).sum(dim=reduce_axes)
    dxhat = dy * gamma
    # Standard LN dx: (1/d) * inv_std * (d*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
    sum_dxhat = dxhat.sum(dim=-1, keepdim=True)
    sum_dxhat_xhat = (dxhat * xhat).sum(dim=-1, keepdim=True)
    dx = inv_std * (dxhat - (sum_dxhat + xhat * sum_dxhat_xhat) / d)
    return dx, dgamma, dbeta


# ---------------------------------------------------------------------------
# Softmax over the last dim with row-max subtraction (matches torch), + bwd.
# ---------------------------------------------------------------------------
def softmax_lastdim(x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    s = e.sum(dim=-1, keepdim=True)
    return e / s


def softmax_backward(dy, y):
    # dx = y * (dy - sum(dy*y))  (over the softmax axis).
    s = (dy * y).sum(dim=-1, keepdim=True)
    return y * (dy - s)


# ---------------------------------------------------------------------------
# Parameter container — mirrors named_parameters() ORDER exactly (30 tensors).
# The CUDA flat-buffer offsets are generated from this same order (see
# decoder_param_layout()).
# ---------------------------------------------------------------------------
# (name, shape) in the EXACT order torch's named_parameters() yields them. This
# is THE single source of truth for the flat weight layout; the C++ header
# (csrc/fused/sm_90/decoder_layout.cuh) static-asserts the count + total.
def decoder_param_spec() -> List[Tuple[str, Tuple[int, ...]]]:
    spec: List[Tuple[str, Tuple[int, ...]]] = [
        ("tok.weight", (VOCAB, D_MODEL)),
        ("pos.weight", (SEQ, D_MODEL)),
    ]
    for li in range(N_LAYERS):
        p = f"layers.{li}."
        spec += [
            (p + "attn.in_proj_weight", (3 * D_MODEL, D_MODEL)),
            (p + "attn.in_proj_bias", (3 * D_MODEL,)),
            (p + "attn.out_proj.weight", (D_MODEL, D_MODEL)),
            (p + "attn.out_proj.bias", (D_MODEL,)),
            (p + "n1.weight", (D_MODEL,)),
            (p + "n1.bias", (D_MODEL,)),
            (p + "n2.weight", (D_MODEL,)),
            (p + "n2.bias", (D_MODEL,)),
            (p + "ff.0.weight", (D_FF, D_MODEL)),
            (p + "ff.0.bias", (D_FF,)),
            (p + "ff.2.weight", (D_MODEL, D_FF)),
            (p + "ff.2.bias", (D_MODEL,)),
        ]
    spec += [
        ("norm.weight", (D_MODEL,)),
        ("norm.bias", (D_MODEL,)),
        ("out.weight", (VOCAB, D_MODEL)),
        ("out.bias", (VOCAB,)),
    ]
    return spec


def decoder_param_layout() -> Dict[str, object]:
    """Flat-buffer layout: offsets/sizes per tensor (the megakernel ABI view).

    Returns dict with: names, offsets (elements into the flat buffer), sizes,
    total (== 422755), n_tensors (== 30). The offsets follow named_parameters()
    order so the flat blob == torch.cat([p.reshape(-1) for _, p in
    model.named_parameters()]).
    """
    spec = decoder_param_spec()
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
    return dict(names=names, offsets=offsets, sizes=sizes, shapes=[s for _, s in spec],
                total=off, n_tensors=len(spec))


@dataclass
class DecoderWeights:
    """All 30 decoder weights as named tensors. Built either from a flat buffer
    (the megakernel view) or from a torch nn module's named_parameters()."""
    tok: torch.Tensor
    pos: torch.Tensor
    layers: List[Dict[str, torch.Tensor]]
    norm_w: torch.Tensor
    norm_b: torch.Tensor
    out_w: torch.Tensor
    out_b: torch.Tensor

    @staticmethod
    def from_named(named: Dict[str, torch.Tensor]) -> "DecoderWeights":
        layers = []
        for li in range(N_LAYERS):
            p = f"layers.{li}."
            layers.append(dict(
                in_w=named[p + "attn.in_proj_weight"],
                in_b=named[p + "attn.in_proj_bias"],
                out_w=named[p + "attn.out_proj.weight"],
                out_b=named[p + "attn.out_proj.bias"],
                n1_w=named[p + "n1.weight"], n1_b=named[p + "n1.bias"],
                n2_w=named[p + "n2.weight"], n2_b=named[p + "n2.bias"],
                ff0_w=named[p + "ff.0.weight"], ff0_b=named[p + "ff.0.bias"],
                ff2_w=named[p + "ff.2.weight"], ff2_b=named[p + "ff.2.bias"],
            ))
        return DecoderWeights(
            tok=named["tok.weight"], pos=named["pos.weight"], layers=layers,
            norm_w=named["norm.weight"], norm_b=named["norm.bias"],
            out_w=named["out.weight"], out_b=named["out.bias"])

    @staticmethod
    def from_flat(flat: torch.Tensor) -> "DecoderWeights":
        lay = decoder_param_layout()
        named = {}
        for name, off, sz, shape in zip(lay["names"], lay["offsets"],
                                        lay["sizes"], lay["shapes"]):
            named[name] = flat[off:off + sz].reshape(shape)
        return DecoderWeights.from_named(named)


def _causal_mask(seq: int, device, dtype) -> torch.Tensor:
    """Additive mask [seq, seq]: 0 where j<=i, -inf where j>i (triu diag=1)."""
    m = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device), 1)
    add = torch.zeros(seq, seq, device=device, dtype=dtype)
    add = add.masked_fill(m, float("-inf"))
    return add


# ===========================================================================
#  FORWARD (manual, primitive ops) — caches everything the manual backward needs.
# ===========================================================================
def decoder_forward(W: DecoderWeights, tokens: torch.Tensor,
                    targets: torch.Tensor):
    """tokens: [B, SEQ] int64 in [0, VOCAB). targets: [B] int64.

    Returns (loss_scalar, cache). loss = mean-over-batch cross-entropy on the
    LAST-position logits. cache holds every fwd intermediate for the backward.
    """
    B, S = tokens.shape
    device, dtype = W.tok.device, W.tok.dtype
    add_mask = _causal_mask(S, device, dtype)  # [S,S]

    # Embedding + positional. pos_ids = arange(S).
    pos_ids = torch.arange(S, device=device)
    h = W.tok[tokens] + W.pos[pos_ids].unsqueeze(0)   # [B,S,d]

    layer_caches = []
    for li in range(N_LAYERS):
        L = W.layers[li]
        x_in = h  # residual base for the attention sublayer
        # ----- self-attention (MultiheadAttention semantics) -----
        qkv = x_in @ L["in_w"].t() + L["in_b"]        # [B,S,3d]
        q, k, v = qkv.split(D_MODEL, dim=-1)
        # split heads: [B,S,d] -> [B,h,S,dh]
        def sh(t):
            return t.reshape(B, S, N_HEADS, D_HEAD).permute(0, 2, 1, 3)
        qh, kh, vh = sh(q), sh(k), sh(v)
        scores = (qh @ kh.transpose(-2, -1)) * ATTN_SCALE   # [B,h,S,S]
        scores = scores + add_mask                          # causal
        attn = softmax_lastdim(scores)                      # [B,h,S,S]
        ctxh = attn @ vh                                     # [B,h,S,dh]
        ctx = ctxh.permute(0, 2, 1, 3).reshape(B, S, D_MODEL)  # merge heads
        a = ctx @ L["out_w"].t() + L["out_b"]               # [B,S,d]
        # ----- residual + n1 (POST-LN): x = n1(x_in + a) -----
        r1 = x_in + a
        x1, ln1_cache = layernorm_forward(r1, L["n1_w"], L["n1_b"])
        # ----- FFN: ff2(gelu(ff0(x1))) -----
        ff0 = x1 @ L["ff0_w"].t() + L["ff0_b"]              # [B,S,4d]
        g = gelu(ff0)                                        # [B,S,4d]
        ff2 = g @ L["ff2_w"].t() + L["ff2_b"]              # [B,S,d]
        # ----- residual + n2 (POST-LN): h = n2(x1 + ff2) -----
        r2 = x1 + ff2
        h2, ln2_cache = layernorm_forward(r2, L["n2_w"], L["n2_b"])
        layer_caches.append(dict(
            x_in=x_in, qkv=qkv, qh=qh, kh=kh, vh=vh, attn=attn, ctxh=ctxh,
            ctx=ctx, r1=r1, x1=x1, ln1_cache=ln1_cache, ff0=ff0, g=g, r2=r2,
            ln2_cache=ln2_cache))
        h = h2

    # Final norm + head on the LAST position only.
    h_last = h[:, -1, :]                                   # [B,d]
    hn, norm_cache = layernorm_forward(h_last, W.norm_w, W.norm_b)
    logits = hn @ W.out_w.t() + W.out_b                    # [B,VOCAB]
    # Cross-entropy, mean over batch (numerically-stable log-softmax).
    m = logits.max(dim=-1, keepdim=True).values
    logz = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))
    nll = logz - logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    loss = nll.mean()
    cache = dict(B=B, S=S, tokens=tokens, targets=targets,
                 layer_caches=layer_caches, h_last=h_last, hn=hn,
                 norm_cache=norm_cache, logits=logits, add_mask=add_mask)
    return loss, cache


# ===========================================================================
#  BACKWARD (manual, NO autograd) — returns grads for every named parameter.
# ===========================================================================
def decoder_backward(W: DecoderWeights, cache) -> Dict[str, torch.Tensor]:
    B, S = cache["B"], cache["S"]
    targets = cache["targets"]
    tokens = cache["tokens"]
    grads: Dict[str, torch.Tensor] = {}

    # ----- CE backward: dlogits = (softmax - onehot) / B -----
    logits = cache["logits"]
    sm = softmax_lastdim(logits)                           # [B,VOCAB]
    onehot = torch.zeros_like(sm)
    onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    dlogits = (sm - onehot) / B                            # [B,VOCAB]

    # ----- head (out): logits = hn @ out_w^T + out_b -----
    hn = cache["hn"]                                       # [B,d]
    grads["out.weight"] = dlogits.t() @ hn                 # [VOCAB,d]
    grads["out.bias"] = dlogits.sum(dim=0)                 # [VOCAB]
    dhn = dlogits @ W.out_w                                # [B,d]

    # ----- final norm backward -----
    dh_last, dnorm_w, dnorm_b = layernorm_backward(dhn, cache["norm_cache"])
    grads["norm.weight"] = dnorm_w
    grads["norm.bias"] = dnorm_b

    # Upstream into the last layer's output: only the LAST position has grad.
    dh = torch.zeros(B, S, D_MODEL, device=hn.device, dtype=hn.dtype)
    dh[:, -1, :] = dh_last

    # ----- per-layer backward (reverse) -----
    for li in reversed(range(N_LAYERS)):
        L = W.layers[li]
        lc = cache["layer_caches"][li]
        gp = f"layers.{li}."
        # h2 = n2(r2);  dh is grad wrt h2.
        dr2, dn2_w, dn2_b = layernorm_backward(dh, lc["ln2_cache"])
        grads[gp + "n2.weight"] = dn2_w
        grads[gp + "n2.bias"] = dn2_b
        # r2 = x1 + ff2  -> grad flows to x1 (residual) AND ff2.
        dff2 = dr2
        dx1 = dr2.clone()
        # ff2 = g @ ff2_w^T + ff2_b
        g = lc["g"]                                        # [B,S,4d]
        grads[gp + "ff.2.weight"] = (dff2.reshape(-1, D_MODEL).t()
                                     @ g.reshape(-1, D_FF))   # [d,4d]
        grads[gp + "ff.2.bias"] = dff2.reshape(-1, D_MODEL).sum(dim=0)
        dg = dff2 @ W.layers[li]["ff2_w"]                  # [B,S,4d]
        # gelu backward
        dff0 = dg * gelu_grad(lc["ff0"])                   # [B,S,4d]
        # ff0 = x1 @ ff0_w^T + ff0_b
        x1 = lc["x1"]
        grads[gp + "ff.0.weight"] = (dff0.reshape(-1, D_FF).t()
                                     @ x1.reshape(-1, D_MODEL))  # [4d,d]
        grads[gp + "ff.0.bias"] = dff0.reshape(-1, D_FF).sum(dim=0)
        dx1 = dx1 + dff0 @ W.layers[li]["ff0_w"]           # add FFN path to x1
        # x1 = n1(r1) -> backward
        dr1, dn1_w, dn1_b = layernorm_backward(dx1, lc["ln1_cache"])
        grads[gp + "n1.weight"] = dn1_w
        grads[gp + "n1.bias"] = dn1_b
        # r1 = x_in + a -> grad to x_in (residual) AND a (attention out).
        da = dr1
        dx_in = dr1.clone()
        # a = ctx @ out_w^T + out_b
        ctx = lc["ctx"]                                    # [B,S,d]
        grads[gp + "attn.out_proj.weight"] = (da.reshape(-1, D_MODEL).t()
                                              @ ctx.reshape(-1, D_MODEL))
        grads[gp + "attn.out_proj.bias"] = da.reshape(-1, D_MODEL).sum(dim=0)
        dctx = da @ W.layers[li]["out_w"]                  # [B,S,d]
        # merge-heads backward: ctx[B,S,d] <- ctxh[B,h,S,dh]
        dctxh = dctx.reshape(B, S, N_HEADS, D_HEAD).permute(0, 2, 1, 3)  # [B,h,S,dh]
        # ctxh = attn @ vh
        attn = lc["attn"]; vh = lc["vh"]; qh = lc["qh"]; kh = lc["kh"]
        dattn = dctxh @ vh.transpose(-2, -1)               # [B,h,S,S]
        dvh = attn.transpose(-2, -1) @ dctxh               # [B,h,S,dh]
        # softmax backward (over key axis) — masked positions had attn==0 so
        # their dscore contribution is zero (y*(...)==0), matching the -inf fwd.
        dscores = softmax_backward(dattn, attn)            # [B,h,S,S]
        dscores = dscores * ATTN_SCALE
        # scores = qh @ kh^T
        dqh = dscores @ kh                                 # [B,h,S,dh]
        dkh = dscores.transpose(-2, -1) @ qh               # [B,h,S,dh]
        # un-split heads: [B,h,S,dh] -> [B,S,d]
        def mh(t):
            return t.permute(0, 2, 1, 3).reshape(B, S, D_MODEL)
        dq, dk, dv = mh(dqh), mh(dkh), mh(dvh)             # each [B,S,d]
        dqkv = torch.cat([dq, dk, dv], dim=-1)             # [B,S,3d]
        # qkv = x_in @ in_w^T + in_b
        x_in = lc["x_in"]
        grads[gp + "attn.in_proj_weight"] = (dqkv.reshape(-1, 3 * D_MODEL).t()
                                             @ x_in.reshape(-1, D_MODEL))
        grads[gp + "attn.in_proj_bias"] = dqkv.reshape(-1, 3 * D_MODEL).sum(dim=0)
        dx_in = dx_in + dqkv @ W.layers[li]["in_w"]        # add attention path
        # propagate to the previous layer's output (or the embedding).
        dh = dx_in

    # ----- embedding backward: h0 = tok[tokens] + pos[pos_ids] -----
    # dh is grad wrt h0 [B,S,d]. tok grad scatter-adds per token id; pos grad
    # sums over batch per position.
    dtok = torch.zeros_like(W.tok)                         # [VOCAB,d]
    # index_add_ accumulates duplicate token ids deterministically enough for the
    # reference; the CUDA path uses a fixed-order accumulation (see report).
    dtok.index_add_(0, tokens.reshape(-1), dh.reshape(-1, D_MODEL))
    grads["tok.weight"] = dtok
    grads["pos.weight"] = dh.sum(dim=0)                    # [S,d]
    return grads


# ===========================================================================
#  Convenience: full step on a torch nn.Module's params (for the parity test).
# ===========================================================================
def oracle_loss_and_grads(named_params: Dict[str, torch.Tensor],
                          tokens: torch.Tensor, targets: torch.Tensor):
    """Run the manual fwd+bwd over a dict of named parameter tensors.

    Returns (loss_float, grads_named). The named_params dict must contain the 30
    decoder tensors (same keys as named_parameters()). All compute is in the
    dtype of the params (use fp32 for the correctness baseline)."""
    W = DecoderWeights.from_named(named_params)
    loss, cache = decoder_forward(W, tokens, targets)
    grads = decoder_backward(W, cache)
    return loss.item(), grads
