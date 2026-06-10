"""tests/hw/vit_oracle.py — runnable PyTorch ORACLE for the real ViT
forward+backward that the L3-REAL ViT megakernel transcribes.

WHY THIS EXISTS (the rigor substitute for an un-compilable, un-runnable .cu):
  PHASE 2 adds the REAL Vision-Transformer fwd+bwd as in-kernel stages of the
  persistent megakernel (the ViT counterpart of PHASE 1's decoder). That CUDA
  backward is hand-derived and the implementing agent cannot compile or run it.
  So we first write the SAME fwd+bwd here in plain PyTorch using ONLY primitive
  ops — and a MANUAL backward (NO autograd) — then assert it matches
  torch.autograd's loss and every parameter .grad to ~1e-12 (fp64). The CUDA
  megakernel is then ported line-for-line from this verified reference,
  converting "is my hand-derived backward correct?" (unrunnable) into "does my
  manual Python match autograd?" (runnable now).

  This module is ALSO the per-tensor expected-grad oracle the parity test
  (tests/hw/test_vit_megakernel.py) consumes: same init, same batch → expected
  loss + expected grads to compare the megakernel against.

EXACT ARCHITECTURE TRANSCRIBED (grokking_race_v2.py, _raw_model → ViT):
  ViT(p=97, patch_dim=49, num_patches=16, d=128, h=4, nl=2)   # :389-404
    patch_proj = nn.Linear(49, 128)        # :392  per-patch embed (16 patches)
    cls_token  = nn.Parameter(1,1,128)*.02 # :393  prepended at position 0
    pos        = nn.Embedding(16+1, 128)   # :394  17 learned position embeds
    layers     = 2 × EncoderBlock:         # :379-387
      attn = MultiheadAttention(128, 4, dropout=0, batch_first=True)  # :382
             NO mask — FULL (bidirectional) attention (contrast: decoder causal)
      n1, n2 = LayerNorm(128)              # :383  (eps=1e-5 default)
      ff = Linear(128,512) → GELU → Linear(512,128)   # :384 (GELU = exact erf)
      forward: a,_ = attn(x,x,x)           # :386  (no attn_mask)
               x = n1(x + a)               # :387  POST-LN
               return n2(x + ff(x))        # :387  POST-LN
    norm = LayerNorm(128)                  # :396  final LN
    out  = nn.Linear(128, 97)              # :396  head → 97 logits (== p)
    forward: h = patch_proj(x)                          # :400  [B,16,d]
             h = cat([cls.expand(B,-1,-1), h], dim=1)   # :401  [B,17,d]
             h = h + pos(arange(17))                    # :402
             for l in layers: h = l(h)                  # :403
             return out(norm(h[:, 0, :]))               # :404  CLS token (pos 0)
  Loss: F.cross_entropy(m(tx), ty)  (mean over batch; grokking_race_v2.py:967…)

  Input x is FLOAT image patches [B, 16, 49] (NOT integer tokens) — see
  make_mnist_addition_data (:287-317): a 28×14 stacked-digit image is unfolded
  into 16 patches of 49 (7×7) pixels. targets ty are int64 in [0, 97).

  nn.MultiheadAttention internals (verified bit-identical to the manual form in
  this file, max|diff|==0): in_proj_weight [3d, d] packs [Wq; Wk; Wv] row-blocks,
  in_proj_bias [3d]; qkv = x @ W_in^T + b_in then split into q,k,v each [.,.,d];
  per-head dh = d/h = 32; scores = (qh @ kh^T)/sqrt(dh); NO causal mask; softmax
  over the key axis; ctx = attn @ vh; merge heads; out = ctx @ W_out^T + b_out
  (out_proj.weight [d,d], out_proj.bias [d]).

NUMERICS (each verified against autograd here, and the values the CUDA must use):
  * GELU = EXACT erf: 0.5*x*(1+erf(x/sqrt2)); derivative
    0.5*(1+erf(x/sqrt2)) + x * (1/sqrt(2*pi)) * exp(-x^2/2).
  * LayerNorm eps = 1e-5 (torch default). Backward dx uses the two
    mean-correction terms (see _layernorm_backward).
  * Attention softmax subtracts the row max for stability (matches torch).
  * CrossEntropy is MEAN over B: dlogits = (softmax - onehot) / B.

This file imports torch only; it needs NO GPU and NO built extension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Architecture constants (single source — mirrored by the C++ layout header and
# the codegen). These are the ViT cell's fixed shapes in the grokking race
# (MODEL_SCALES["small"]; ViT defaults patch_dim=49, num_patches=16, p=97).
# ---------------------------------------------------------------------------
VOCAB = 97       # p (head Linear(d, p)); targets in [0, 97)
D_MODEL = 128    # dim_model (MODEL_SCALES["small"])
N_HEADS = 4      # num_heads
N_LAYERS = 2     # num_layers
PATCH_DIM = 49   # patch pixel count (7×7)
NUM_PATCHES = 16 # image patches
SEQ = NUM_PATCHES + 1  # 17 — CLS token at position 0, then 16 patch positions
D_FF = 4 * D_MODEL     # 512 — ff hidden (Linear(d, 4*d))
D_HEAD = D_MODEL // N_HEADS  # 32
LN_EPS = 1e-5    # torch.nn.LayerNorm default eps
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
# (Identical to the decoder oracle — shared math.)
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
    reduce_axes = tuple(range(dy.dim() - 1))
    dbeta = dy.sum(dim=reduce_axes)
    dgamma = (dy * xhat).sum(dim=reduce_axes)
    dxhat = dy * gamma
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
# Parameter container — mirrors named_parameters() ORDER exactly (32 tensors).
# The CUDA flat-buffer offsets are generated from this same order (see
# vit_param_layout()).
# ---------------------------------------------------------------------------
# (name, shape) in the EXACT order torch's named_parameters() yields them. THE
# single source of truth for the flat weight layout; the C++ header
# (csrc/fused/sm_90/vit_layout.cuh) static-asserts the count + total.
#
# ORDER NOTE (verified against the live model's named_parameters): the module
# attributes are registered in __init__ order — patch_proj, cls_token, pos, ...
# — BUT cls_token is an nn.Parameter (a leaf) while patch_proj is a submodule,
# and PyTorch yields a module's OWN parameters (here cls_token) before recursing
# into child modules (patch_proj.*). So the head of the list is:
#   cls_token, patch_proj.weight, patch_proj.bias, pos.weight, <layers...>.
def vit_param_spec() -> List[Tuple[str, Tuple[int, ...]]]:
    spec: List[Tuple[str, Tuple[int, ...]]] = [
        ("cls_token", (1, 1, D_MODEL)),
        ("patch_proj.weight", (D_MODEL, PATCH_DIM)),
        ("patch_proj.bias", (D_MODEL,)),
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


def vit_param_layout() -> Dict[str, object]:
    """Flat-buffer layout: offsets/sizes per tensor (the megakernel ABI view).

    Returns dict with: names, offsets (elements into the flat buffer), sizes,
    total (== 418017), n_tensors (== 32). The offsets follow named_parameters()
    order so the flat blob == torch.cat([p.reshape(-1) for _, p in
    model.named_parameters()]).
    """
    spec = vit_param_spec()
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
class ViTWeights:
    """All 32 ViT weights as named tensors. Built either from a flat buffer
    (the megakernel view) or from a torch nn module's named_parameters()."""
    cls: torch.Tensor          # [1,1,d]  -> used as [d]
    patch_w: torch.Tensor      # [d, patch_dim]
    patch_b: torch.Tensor      # [d]
    pos: torch.Tensor          # [SEQ, d]
    layers: List[Dict[str, torch.Tensor]]
    norm_w: torch.Tensor
    norm_b: torch.Tensor
    out_w: torch.Tensor
    out_b: torch.Tensor

    @staticmethod
    def from_named(named: Dict[str, torch.Tensor]) -> "ViTWeights":
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
        return ViTWeights(
            cls=named["cls_token"], patch_w=named["patch_proj.weight"],
            patch_b=named["patch_proj.bias"], pos=named["pos.weight"],
            layers=layers,
            norm_w=named["norm.weight"], norm_b=named["norm.bias"],
            out_w=named["out.weight"], out_b=named["out.bias"])

    @staticmethod
    def from_flat(flat: torch.Tensor) -> "ViTWeights":
        lay = vit_param_layout()
        named = {}
        for name, off, sz, shape in zip(lay["names"], lay["offsets"],
                                        lay["sizes"], lay["shapes"]):
            named[name] = flat[off:off + sz].reshape(shape)
        return ViTWeights.from_named(named)


# ===========================================================================
#  FORWARD (manual, primitive ops) — caches everything the manual backward needs.
# ===========================================================================
def vit_forward(W: ViTWeights, patches: torch.Tensor, targets: torch.Tensor):
    """patches: [B, NUM_PATCHES, PATCH_DIM] float. targets: [B] int64.

    Returns (loss_scalar, cache). loss = mean-over-batch cross-entropy on the
    CLS-position (pos 0) logits. cache holds every fwd intermediate for the bwd.
    """
    B, P, _ = patches.shape
    device, dtype = W.patch_w.device, W.patch_w.dtype
    assert P == NUM_PATCHES

    # Patch embed (positions 1..16) + CLS token (position 0); add pos embeds.
    proj = patches @ W.patch_w.t() + W.patch_b          # [B, 16, d]
    cls = W.cls.reshape(1, 1, D_MODEL).expand(B, 1, D_MODEL)  # [B,1,d]
    h = torch.cat([cls, proj], dim=1)                   # [B, 17, d]
    pos_ids = torch.arange(SEQ, device=device)
    h = h + W.pos[pos_ids].unsqueeze(0)                 # [B, 17, d]

    layer_caches = []
    for li in range(N_LAYERS):
        L = W.layers[li]
        x_in = h  # residual base for the attention sublayer
        # ----- self-attention (MultiheadAttention semantics; NO causal mask) --
        qkv = x_in @ L["in_w"].t() + L["in_b"]          # [B,S,3d]
        q, k, v = qkv.split(D_MODEL, dim=-1)

        def sh(t):
            return t.reshape(B, SEQ, N_HEADS, D_HEAD).permute(0, 2, 1, 3)
        qh, kh, vh = sh(q), sh(k), sh(v)
        scores = (qh @ kh.transpose(-2, -1)) * ATTN_SCALE   # [B,h,S,S]
        attn = softmax_lastdim(scores)                      # [B,h,S,S] (full)
        ctxh = attn @ vh                                     # [B,h,S,dh]
        ctx = ctxh.permute(0, 2, 1, 3).reshape(B, SEQ, D_MODEL)  # merge heads
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

    # Final norm + head on the CLS position (pos 0) only.
    h_cls = h[:, 0, :]                                     # [B,d]
    hn, norm_cache = layernorm_forward(h_cls, W.norm_w, W.norm_b)
    logits = hn @ W.out_w.t() + W.out_b                    # [B,VOCAB]
    # Cross-entropy, mean over batch (numerically-stable log-softmax).
    m = logits.max(dim=-1, keepdim=True).values
    logz = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))
    nll = logz - logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    loss = nll.mean()
    cache = dict(B=B, patches=patches, targets=targets,
                 layer_caches=layer_caches, h_cls=h_cls, hn=hn,
                 norm_cache=norm_cache, logits=logits)
    return loss, cache


# ===========================================================================
#  BACKWARD (manual, NO autograd) — returns grads for every named parameter.
# ===========================================================================
def vit_backward(W: ViTWeights, cache) -> Dict[str, torch.Tensor]:
    B = cache["B"]
    targets = cache["targets"]
    patches = cache["patches"]
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
    dh_cls, dnorm_w, dnorm_b = layernorm_backward(dhn, cache["norm_cache"])
    grads["norm.weight"] = dnorm_w
    grads["norm.bias"] = dnorm_b

    # Upstream into the last layer's output: only the CLS position (0) has grad.
    dh = torch.zeros(B, SEQ, D_MODEL, device=hn.device, dtype=hn.dtype)
    dh[:, 0, :] = dh_cls

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
        dctxh = dctx.reshape(B, SEQ, N_HEADS, D_HEAD).permute(0, 2, 1, 3)
        # ctxh = attn @ vh
        attn = lc["attn"]; vh = lc["vh"]; qh = lc["qh"]; kh = lc["kh"]
        dattn = dctxh @ vh.transpose(-2, -1)               # [B,h,S,S]
        dvh = attn.transpose(-2, -1) @ dctxh               # [B,h,S,dh]
        # softmax backward (over key axis) — NO mask, full S.
        dscores = softmax_backward(dattn, attn)            # [B,h,S,S]
        dscores = dscores * ATTN_SCALE
        # scores = qh @ kh^T
        dqh = dscores @ kh                                 # [B,h,S,dh]
        dkh = dscores.transpose(-2, -1) @ qh               # [B,h,S,dh]
        # un-split heads: [B,h,S,dh] -> [B,S,d]
        def mh(t):
            return t.permute(0, 2, 1, 3).reshape(B, SEQ, D_MODEL)
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

    # ----- embedding backward -----
    # dh is grad wrt h0 [B,S,d] (post-cat, pre-pos-add input to layer 0).
    # h0 = cat([cls, patch_proj(patches)], 1) + pos.
    #   pos.weight[s] grad = sum over batch of dh[:, s, :]   (all 17 positions)
    #   cls_token grad     = sum over batch of dh[:, 0, :]   (position 0 only)
    #   patch_proj: positions 1..16 are patch_proj(patches[:, i, :]):
    #     dW += sum_b sum_i dh[b, 1+i, :]^T ⊗ patches[b, i, :]
    #     db += sum_b sum_i dh[b, 1+i, :]
    grads["pos.weight"] = dh.sum(dim=0)                    # [SEQ, d]
    grads["cls_token"] = dh[:, 0, :].sum(dim=0).reshape(1, 1, D_MODEL)
    dh_patches = dh[:, 1:, :]                              # [B, 16, d]
    # dW_patch [d, patch_dim] = sum over (B,patch) of dh_patches ⊗ patches
    grads["patch_proj.weight"] = (dh_patches.reshape(-1, D_MODEL).t()
                                  @ patches.reshape(-1, PATCH_DIM))   # [d,patch]
    grads["patch_proj.bias"] = dh_patches.reshape(-1, D_MODEL).sum(dim=0)  # [d]
    return grads


# ===========================================================================
#  Convenience: full step on a torch nn.Module's params (for the parity test).
# ===========================================================================
def oracle_loss_and_grads(named_params: Dict[str, torch.Tensor],
                          patches: torch.Tensor, targets: torch.Tensor):
    """Run the manual fwd+bwd over a dict of named parameter tensors.

    Returns (loss_float, grads_named). The named_params dict must contain the 32
    ViT tensors (same keys as named_parameters()). All compute is in the dtype
    of the params (use fp32/fp64 for the correctness baseline)."""
    W = ViTWeights.from_named(named_params)
    loss, cache = vit_forward(W, patches, targets)
    grads = vit_backward(W, cache)
    return loss.item(), grads
