"""tests/hw/vit_kernel_mirror.py — single-threaded STRUCTURAL mirror of the
CUDA ViT megakernel (csrc/fused/sm_90/model_stage_vit.cuh).

WHY (the gate the math-oracle cannot provide): vit_oracle.py proves the ViT
fwd/bwd MATH is correct (matched to autograd ~1e-15). It does NOT exercise the
CUDA kernel's *structure* — the smem buffer reuse/aliasing, the recompute-in-
backward, the head-split index arithmetic, the owner-thread grad accumulation,
the per-sample sequential reduction, the cls/patch embedding-grad scatter. That
structure is exactly where a hand-written, un-runnable backward hides missing
terms / index bugs (e.g. summing dv over the causal triangle when ViT attention
is FULL, a wrong head-split offset, a dropped residual path, a recompute that
diverges from the forward).

This module re-implements the kernel ALGORITHM in plain Python — same loop nests
and index math, same recompute-in-backward, same per-sample sequential weight-grad
accumulation into a CTA-style partial — so that asserting it against vit_oracle's
grads (~1e-12 on a tiny batch) catches that STRUCTURAL bug class.

VALIDATION BOUNDARY — read before trusting a green run (the directive's three
distinct checks, kept honestly separate):
  * vit_oracle.py        : the MATH is correct (manual fwd+bwd == autograd ~1e-15).
  * THIS mirror          : the kernel's INDEX ARITHMETIC + RECOMPUTE STRUCTURE are
                           correct (== the oracle ~1e-12). The `_recompute_layer`
                           assertion (recompute reproduces the forward) is a real,
                           load-bearing check.
  * MANUAL INSPECTION ONLY (NOT covered by this mirror):
      - __syncthreads / grid-barrier placement (this mirror is single-threaded);
      - BUFFER OVERWRITE-ORDERING, i.e. the smem ALIASING SCHEDULE. This mirror
        holds each backward intermediate (dr2, dx1, dctx, dqkv, dsc, …) in its OWN
        fresh array, whereas model_stage_vit.cuh ALIASES them onto shared buffers
        (dr2→ctx, dx1→fn_xhat, dctx→ff0, dqkv→gact, …). So a DEAD-BUFFER READ in
        the .cu's actual aliasing order (reading a buffer after a later store has
        clobbered it) would NOT be caught here — the mirror's grads match the
        oracle regardless of overwrite order. That schedule is verified by reading
        each store/read against the kernel header's documented alias map. (The
        decoder mirror has the same boundary; an alias-faithful arena variant — one
        flat buffer with views at the .cu's relative offsets — is the way to close
        it on CPU and is noted as a follow-up in the report.)
  These three together are what "oracle/mirror agreement" means in the report —
  NOT "the kernel is validated end-to-end."

The mirror follows the .cuh line-by-line: vit_forward_sample,
vit_recompute_layer, vit_backward_sample, vit_linear/_bwd, vit_layernorm_*, and
the three-pass FULL-attention backward (A: dv, B: dscores, C: dq/dk).

STRUCTURAL DECISIONS MIRRORED (must match model_stage_vit.cuh exactly):
  * ONE layer live at a time; the backward RECOMPUTES each layer's forward from
    the cached layer INPUT (layer_in[li]) — only layer_in (+ the per-sample patch
    pixels) survive across layers.
  * Both ff0 (pre-GELU, for gelu') and gact (post-GELU g, for dW_ff2) are kept
    live within the layer being processed (dynamic smem, ~184 KB total budget —
    see the .cuh smem note). (NOTE: which array holds which intermediate is the
    aliasing schedule the mirror does NOT enforce — see the boundary above.)
  * FULL attention (no causal mask): dv sums over ALL qi; dq/dk over ALL kj/qi.
  * Embedding backward: pos.weight over all 17 positions; cls_token from pos 0;
    patch_proj.{weight,bias} from positions 1..16 against the live patch pixels.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from tests.hw.vit_oracle import (
    ViTWeights, vit_param_layout,
    VOCAB, D_MODEL, N_HEADS, N_LAYERS, PATCH_DIM, NUM_PATCHES, SEQ, D_FF,
    D_HEAD, LN_EPS, ATTN_SCALE, gelu, gelu_grad,
)

_F64 = torch.float64  # mirror in fp64 so the only diff vs oracle is structure


# --- primitive ops, mirroring the .cuh device functions exactly --------------
def _linear(x, W, b, in_dim, out_dim, n_rows=SEQ):
    # y[s,o] = sum_k x[s,k]*W[o,k] + b[o]   (W row-major [out,in])
    y = torch.zeros(n_rows, out_dim, dtype=_F64)
    for s in range(n_rows):
        for o in range(out_dim):
            acc = b[o].item() if b is not None else 0.0
            for k in range(in_dim):
                acc += x[s, k].item() * W[o, k].item()
            y[s, o] = acc
    return y


def _layernorm_fwd(x, gamma, beta):
    y = torch.zeros(SEQ, D_MODEL, dtype=_F64)
    xhat = torch.zeros(SEQ, D_MODEL, dtype=_F64)
    inv = torch.zeros(SEQ, dtype=_F64)
    for s in range(SEQ):
        mean = x[s].mean()
        var = ((x[s] - mean) ** 2).mean()
        i = 1.0 / math.sqrt(var.item() + LN_EPS)
        inv[s] = i
        for j in range(D_MODEL):
            xh = (x[s, j] - mean) * i
            xhat[s, j] = xh
            y[s, j] = xh * gamma[j] + beta[j]
    return y, xhat, inv


# --- the per-sample forward (mirrors vit_forward_sample) ----------------------
def _forward_sample(W: ViTWeights, patch_s, target):
    """patch_s: [NUM_PATCHES, PATCH_DIM] float (this sample's image patches)."""
    sm = {}
    # Embedding: pos 0 = cls + pos[0]; pos 1..16 = patch_proj(patch_i) + pos[1+i].
    h = torch.zeros(SEQ, D_MODEL, dtype=_F64)
    cls = W.cls.reshape(D_MODEL)
    for j in range(D_MODEL):
        h[0, j] = cls[j] + W.pos[0, j]
    # patch_proj applied to the 16 patches -> rows 1..16.
    proj = _linear(patch_s, W.patch_w, W.patch_b, PATCH_DIM, D_MODEL,
                   n_rows=NUM_PATCHES)   # [16, d]
    for i in range(NUM_PATCHES):
        for j in range(D_MODEL):
            h[1 + i, j] = proj[i, j] + W.pos[1 + i, j]
    sm["layer_in"] = [h.clone(), None]
    for li in range(N_LAYERS):
        L = W.layers[li]
        hin = sm["layer_in"][li]
        qkv = _linear(hin, L["in_w"], L["in_b"], D_MODEL, 3 * D_MODEL)
        attn = torch.zeros(N_HEADS, SEQ, SEQ, dtype=_F64)
        ctx = torch.zeros(SEQ, D_MODEL, dtype=_F64)
        for hh in range(N_HEADS):
            qoff = hh * D_HEAD
            for qi in range(SEQ):
                sc = [0.0 for _ in range(SEQ)]
                maxs = -math.inf
                for kj in range(SEQ):  # FULL attention: every kj
                    dot = 0.0
                    for t in range(D_HEAD):
                        dot += qkv[qi, qoff + t].item() * qkv[kj, D_MODEL + qoff + t].item()
                    sc[kj] = dot * ATTN_SCALE
                    maxs = max(maxs, sc[kj])
                denom = 0.0
                for kj in range(SEQ):
                    e = math.exp(sc[kj] - maxs)
                    sc[kj] = e
                    denom += e
                for kj in range(SEQ):
                    attn[hh, qi, kj] = sc[kj] / denom
                for t in range(D_HEAD):
                    acc = 0.0
                    for kj in range(SEQ):  # FULL: all kj contribute to ctx
                        acc += attn[hh, qi, kj].item() * qkv[kj, 2 * D_MODEL + qoff + t].item()
                    ctx[qi, qoff + t] = acc
        a = _linear(ctx, L["out_w"], L["out_b"], D_MODEL, D_MODEL)
        r1 = hin + a
        x1, n1_xhat, n1_inv = _layernorm_fwd(r1, L["n1_w"], L["n1_b"])
        ff0 = _linear(x1, L["ff0_w"], L["ff0_b"], D_MODEL, D_FF)
        gact = gelu(ff0)
        ff2 = _linear(gact, L["ff2_w"], L["ff2_b"], D_FF, D_MODEL)
        r2 = x1 + ff2
        h2, n2_xhat, n2_inv = _layernorm_fwd(r2, L["n2_w"], L["n2_b"])
        sm[f"L{li}"] = dict(qkv=qkv, attn=attn, ctx=ctx, x1=x1, ff0=ff0,
                            gact=gact, n1_xhat=n1_xhat, n1_inv=n1_inv,
                            n2_xhat=n2_xhat, n2_inv=n2_inv)
        if li + 1 < N_LAYERS:
            sm["layer_in"][li + 1] = h2
        else:
            sm["final_in"] = h2
    # final norm (CLS pos 0) + head -> logits
    hcls = sm["final_in"][0]
    mean = hcls.mean()
    var = ((hcls - mean) ** 2).mean()
    inv = 1.0 / math.sqrt(var.item() + LN_EPS)
    fn_xhat = torch.zeros(D_MODEL, dtype=_F64)
    hn = torch.zeros(D_MODEL, dtype=_F64)
    for j in range(D_MODEL):
        xh = (hcls[j] - mean) * inv
        fn_xhat[j] = xh
        hn[j] = xh * W.norm_w[j] + W.norm_b[j]
    logits = torch.zeros(VOCAB, dtype=_F64)
    for o in range(VOCAB):
        acc = W.out_b[o].item()
        for k in range(D_MODEL):
            acc += hn[k].item() * W.out_w[o, k].item()
        logits[o] = acc
    sm["fn_xhat"] = fn_xhat
    sm["fn_inv"] = inv
    sm["logits"] = logits
    return sm


# --- recompute one layer's fwd intermediates (mirrors vit_recompute_layer) ----
def _recompute_layer(W, li, sm):
    """RE-RUN the layer forward from the cached layer INPUT (sm['layer_in'][li]),
    exactly as the CUDA vit_recompute_layer does (it does NOT reuse a per-layer
    cache — only layer_in survives). Exercises the recompute path's index math +
    that it reproduces the forward values; we assert it equals the original
    forward cache to catch a recompute divergence (a wrong residual or a clobbered
    input)."""
    L = W.layers[li]
    hin = sm["layer_in"][li]
    qkv = _linear(hin, L["in_w"], L["in_b"], D_MODEL, 3 * D_MODEL)
    attn = torch.zeros(N_HEADS, SEQ, SEQ, dtype=_F64)
    ctx = torch.zeros(SEQ, D_MODEL, dtype=_F64)
    for hh in range(N_HEADS):
        qoff = hh * D_HEAD
        for qi in range(SEQ):
            sc = [0.0 for _ in range(SEQ)]
            maxs = -math.inf
            for kj in range(SEQ):
                dot = 0.0
                for t in range(D_HEAD):
                    dot += qkv[qi, qoff + t].item() * qkv[kj, D_MODEL + qoff + t].item()
                sc[kj] = dot * ATTN_SCALE
                maxs = max(maxs, sc[kj])
            denom = 0.0
            for kj in range(SEQ):
                e = math.exp(sc[kj] - maxs)
                sc[kj] = e
                denom += e
            for kj in range(SEQ):
                attn[hh, qi, kj] = sc[kj] / denom
            for t in range(D_HEAD):
                acc = 0.0
                for kj in range(SEQ):
                    acc += attn[hh, qi, kj].item() * qkv[kj, 2 * D_MODEL + qoff + t].item()
                ctx[qi, qoff + t] = acc
    a = _linear(ctx, L["out_w"], L["out_b"], D_MODEL, D_MODEL)
    r1 = hin + a
    x1, n1_xhat, n1_inv = _layernorm_fwd(r1, L["n1_w"], L["n1_b"])
    ff0 = _linear(x1, L["ff0_w"], L["ff0_b"], D_MODEL, D_FF)
    gact = gelu(ff0)
    ff2 = _linear(gact, L["ff2_w"], L["ff2_b"], D_FF, D_MODEL)
    r2 = x1 + ff2
    _, n2_xhat, n2_inv = _layernorm_fwd(r2, L["n2_w"], L["n2_b"])
    rc = dict(qkv=qkv, attn=attn, ctx=ctx, x1=x1, ff0=ff0, gact=gact,
              n1_xhat=n1_xhat, n1_inv=n1_inv, n2_xhat=n2_xhat, n2_inv=n2_inv)
    # Recompute MUST reproduce the original forward exactly (same inputs/ops).
    orig = sm[f"L{li}"]
    for k in ("qkv", "ctx", "x1", "ff0", "gact", "n1_xhat", "n2_xhat"):
        d = (rc[k] - orig[k]).abs().max().item()
        assert d < 1e-9, f"recompute layer {li} field {k} diverged by {d:.2e}"
    return rc


# --- the per-sample backward (mirrors vit_backward_sample) --------------------
def _backward_sample(W: ViTWeights, g: Dict[str, torch.Tensor],
                     patch_s, target, B, sm):
    # CE backward: dlogits = (softmax - onehot)/B
    logits = sm["logits"]
    mx = logits.max().item()
    es = sum(math.exp(logits[o].item() - mx) for o in range(VOCAB))
    dlogits = torch.zeros(VOCAB, dtype=_F64)
    for o in range(VOCAB):
        smo = math.exp(logits[o].item() - mx) / es
        dlogits[o] = (smo - (1.0 if o == target else 0.0)) / B
    # head bwd: dW_out += dlogits ⊗ hn ; db_out += dlogits ; dhn = dlogits @ out_w
    fn_xhat = sm["fn_xhat"]
    hn = torch.zeros(D_MODEL, dtype=_F64)
    for j in range(D_MODEL):
        hn[j] = fn_xhat[j] * W.norm_w[j] + W.norm_b[j]
    for o in range(VOCAB):
        g["out.bias"][o] += dlogits[o]
        for j in range(D_MODEL):
            g["out.weight"][o, j] += dlogits[o] * hn[j]
    dhn = torch.zeros(D_MODEL, dtype=_F64)
    for j in range(D_MODEL):
        acc = 0.0
        for o in range(VOCAB):
            acc += dlogits[o].item() * W.out_w[o, j].item()
        dhn[j] = acc
    # final-norm bwd (single row = CLS position 0).
    dy = dhn
    for j in range(D_MODEL):
        g["norm.bias"][j] += dy[j]
        g["norm.weight"][j] += dy[j] * fn_xhat[j]
    sda = sum((dy[j] * W.norm_w[j]).item() for j in range(D_MODEL))
    sdax = sum((dy[j] * W.norm_w[j] * fn_xhat[j]).item() for j in range(D_MODEL))
    inv_s = sm["fn_inv"]
    dh = torch.zeros(SEQ, D_MODEL, dtype=_F64)
    for j in range(D_MODEL):
        dxhat = dy[j] * W.norm_w[j]
        dh[0, j] = inv_s * (dxhat - (sda + fn_xhat[j] * sdax) / D_MODEL)
    # dh now = grad wrt the last layer's OUTPUT [SEQ,d] (only pos 0 nonzero).
    # per-layer backward (reverse)
    for li in reversed(range(N_LAYERS)):
        L = W.layers[li]
        gp = f"layers.{li}."
        c = _recompute_layer(W, li, sm)
        qkv, attn, ctx_fwd = c["qkv"], c["attn"], c["ctx"]
        x1, ff0, gact = c["x1"], c["ff0"], c["gact"]
        n1_xhat, n1_inv = c["n1_xhat"], c["n1_inv"]
        n2_xhat, n2_inv = c["n2_xhat"], c["n2_inv"]
        # n2 bwd: dh -> dr2 ; accumulate n2 gamma/beta.
        dr2 = torch.zeros(SEQ, D_MODEL, dtype=_F64)
        for j in range(D_MODEL):
            for s in range(SEQ):
                d = dh[s, j]
                g[gp + "n2.bias"][j] += d
                g[gp + "n2.weight"][j] += d * n2_xhat[s, j]
        for s in range(SEQ):
            sda = sum((dh[s, j] * L["n2_w"][j]).item() for j in range(D_MODEL))
            sdax = sum((dh[s, j] * L["n2_w"][j] * n2_xhat[s, j]).item()
                       for j in range(D_MODEL))
            for j in range(D_MODEL):
                dxhat = dh[s, j] * L["n2_w"][j]
                dr2[s, j] = n2_inv[s] * (dxhat - (sda + n2_xhat[s, j] * sdax) / D_MODEL)
        # dx1 := dr2 (residual). dff2 = dr2.
        dx1 = dr2.clone()
        # ff.2 bwd: dW += dr2^T @ gact ; db += sum dr2 ; dff0 = (dr2 @ ff2_w)*gelu'(ff0)
        for o in range(D_MODEL):
            for i in range(D_FF):
                acc = 0.0
                for s in range(SEQ):
                    acc += dr2[s, o].item() * gact[s, i].item()
                g[gp + "ff.2.weight"][o, i] += acc
            acc = 0.0
            for s in range(SEQ):
                acc += dr2[s, o].item()
            g[gp + "ff.2.bias"][o] += acc
        dff0 = torch.zeros(SEQ, D_FF, dtype=_F64)
        ggrad = gelu_grad(ff0)
        for s in range(SEQ):
            for i in range(D_FF):
                dg = 0.0
                for o in range(D_MODEL):
                    dg += dr2[s, o].item() * L["ff2_w"][o, i].item()
                dff0[s, i] = dg * ggrad[s, i]
        # ff.0 bwd: dW += dff0^T @ x1 ; db += sum dff0 ; dx1 += dff0 @ ff0_w
        for o in range(D_FF):
            for i in range(D_MODEL):
                acc = 0.0
                for s in range(SEQ):
                    acc += dff0[s, o].item() * x1[s, i].item()
                g[gp + "ff.0.weight"][o, i] += acc
            acc = 0.0
            for s in range(SEQ):
                acc += dff0[s, o].item()
            g[gp + "ff.0.bias"][o] += acc
        for s in range(SEQ):
            for i in range(D_MODEL):
                acc = 0.0
                for o in range(D_FF):
                    acc += dff0[s, o].item() * L["ff0_w"][o, i].item()
                dx1[s, i] += acc
        # n1 bwd: dx1 -> dr1 ; accumulate n1 gamma/beta.
        dr1 = torch.zeros(SEQ, D_MODEL, dtype=_F64)
        for j in range(D_MODEL):
            for s in range(SEQ):
                d = dx1[s, j]
                g[gp + "n1.bias"][j] += d
                g[gp + "n1.weight"][j] += d * n1_xhat[s, j]
        for s in range(SEQ):
            sda = sum((dx1[s, j] * L["n1_w"][j]).item() for j in range(D_MODEL))
            sdax = sum((dx1[s, j] * L["n1_w"][j] * n1_xhat[s, j]).item()
                       for j in range(D_MODEL))
            for j in range(D_MODEL):
                dxhat = dx1[s, j] * L["n1_w"][j]
                dr1[s, j] = n1_inv[s] * (dxhat - (sda + n1_xhat[s, j] * sdax) / D_MODEL)
        # r1 = x_in + a -> da = dr1 ; dx_in := dr1 (residual).
        da = dr1
        dx_in = dr1.clone()
        # out_proj bwd: dW += da^T @ ctx_fwd ; db += sum da ; dctx = da @ out_w
        for o in range(D_MODEL):
            for i in range(D_MODEL):
                acc = 0.0
                for s in range(SEQ):
                    acc += da[s, o].item() * ctx_fwd[s, i].item()
                g[gp + "attn.out_proj.weight"][o, i] += acc
            acc = 0.0
            for s in range(SEQ):
                acc += da[s, o].item()
            g[gp + "attn.out_proj.bias"][o] += acc
        dctx = torch.zeros(SEQ, D_MODEL, dtype=_F64)
        for s in range(SEQ):
            for i in range(D_MODEL):
                acc = 0.0
                for o in range(D_MODEL):
                    acc += da[s, o].item() * L["out_w"][o, i].item()
                dctx[s, i] = acc
        # --- attention backward (3 passes: A dv, B dscores, C dq/dk); FULL ---
        dqkv = torch.zeros(SEQ, 3 * D_MODEL, dtype=_F64)
        # A: dv[kj,qoff+t] = sum_{qi} attn[hh,qi,kj]*dctx[qi,qoff+t]  (ALL qi)
        for kj in range(SEQ):
            for hh in range(N_HEADS):
                qoff = hh * D_HEAD
                for t in range(D_HEAD):
                    acc = 0.0
                    for qi in range(SEQ):
                        acc += attn[hh, qi, kj].item() * dctx[qi, qoff + t].item()
                    dqkv[kj, 2 * D_MODEL + qoff + t] = acc
        # B: dsc[hh,qi,kj] = attn*(datt - sum_k datt*attn)*scale   (FULL, all kj)
        dsc = torch.zeros(N_HEADS, SEQ, SEQ, dtype=_F64)
        for hh in range(N_HEADS):
            qoff = hh * D_HEAD
            for qi in range(SEQ):
                datt = [0.0] * SEQ
                for kj in range(SEQ):
                    acc = 0.0
                    for t in range(D_HEAD):
                        acc += dctx[qi, qoff + t].item() * qkv[kj, 2 * D_MODEL + qoff + t].item()
                    datt[kj] = acc
                dot = 0.0
                for kj in range(SEQ):
                    dot += datt[kj] * attn[hh, qi, kj].item()
                for kj in range(SEQ):
                    a = attn[hh, qi, kj].item()
                    dsc[hh, qi, kj] = a * (datt[kj] - dot) * ATTN_SCALE
        # C: dq[pos,qoff+t]=sum_kj dsc[hh,pos,kj]*k[kj]; dk[pos]=sum_qi dsc[hh,qi,pos]*q[qi]
        for pos in range(SEQ):
            for hh in range(N_HEADS):
                qoff = hh * D_HEAD
                for t in range(D_HEAD):
                    dq = 0.0
                    dk = 0.0
                    for kj in range(SEQ):
                        dq += dsc[hh, pos, kj].item() * qkv[kj, D_MODEL + qoff + t].item()
                        dk += dsc[hh, kj, pos].item() * qkv[kj, qoff + t].item()
                    dqkv[pos, qoff + t] = dq
                    dqkv[pos, D_MODEL + qoff + t] = dk
        # in_proj bwd: dW += dqkv^T @ x_in ; db += sum dqkv ; dx_in += dqkv @ in_w
        x_in = sm["layer_in"][li]
        for o in range(3 * D_MODEL):
            for i in range(D_MODEL):
                acc = 0.0
                for s in range(SEQ):
                    acc += dqkv[s, o].item() * x_in[s, i].item()
                g[gp + "attn.in_proj_weight"][o, i] += acc
            acc = 0.0
            for s in range(SEQ):
                acc += dqkv[s, o].item()
            g[gp + "attn.in_proj_bias"][o] += acc
        for s in range(SEQ):
            for i in range(D_MODEL):
                acc = 0.0
                for o in range(3 * D_MODEL):
                    acc += dqkv[s, o].item() * L["in_w"][o, i].item()
                dx_in[s, i] += acc
        dh = dx_in
    # embedding bwd.  dh = grad wrt h0 [SEQ,d] (post-cat, pre-pos-add input to L0).
    #   pos.weight[s] += dh[s]   (ALL 17 positions; owner: thread per (s,j) col)
    #   cls_token     += dh[0]   (position 0 only)
    #   patch_proj: rows 1..16 are patch_proj(patch_i):
    #     dW_patch[o,i] += sum_{p=0..15} dh[1+p, o] * patch_s[p, i]
    #     db_patch[o]   += sum_{p=0..15} dh[1+p, o]
    for j in range(D_MODEL):
        for s in range(SEQ):
            g["pos.weight"][s, j] += dh[s, j]
        g["cls_token"][0, 0, j] += dh[0, j]
    # patch_proj weight/bias: owner thread per (o,i) output / per o bias; sum over
    # the 16 patch positions (sequential), reading the live patch pixels.
    for o in range(D_MODEL):
        for i in range(PATCH_DIM):
            acc = 0.0
            for p in range(NUM_PATCHES):
                acc += dh[1 + p, o].item() * patch_s[p, i].item()
            g["patch_proj.weight"][o, i] += acc
        acc = 0.0
        for p in range(NUM_PATCHES):
            acc += dh[1 + p, o].item()
        g["patch_proj.bias"][o] += acc


def mirror_loss_and_grads(named_params: Dict[str, torch.Tensor],
                          patches: torch.Tensor, targets: torch.Tensor):
    """Run the STRUCTURAL kernel mirror over the whole (small) batch.

    Mirrors the CTA loop: per sample, forward then immediately backward,
    accumulating weight grads into one partial (here a single CTA == whole batch,
    which is what determinism requires per element). Returns (loss, grads)."""
    named = {k: v.to(_F64) for k, v in named_params.items()}
    W = ViTWeights.from_named(named)
    lay = vit_param_layout()
    g = {name: torch.zeros(shape, dtype=_F64)
         for name, shape in zip(lay["names"], lay["shapes"])}
    B = patches.shape[0]
    nll_sum = 0.0
    for b in range(B):
        patch_s = patches[b]                      # [16, 49] float
        tgt = int(targets[b])
        sm = _forward_sample(W, patch_s, tgt)
        logits = sm["logits"]
        mx = logits.max().item()
        es = sum(math.exp(logits[o].item() - mx) for o in range(VOCAB))
        nll_sum += (mx + math.log(es)) - logits[tgt].item()
        _backward_sample(W, g, patch_s, tgt, B, sm)
    return nll_sum / B, g
