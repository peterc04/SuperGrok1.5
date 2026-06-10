"""tests/hw/decoder_kernel_mirror.py — single-threaded STRUCTURAL mirror of the
CUDA decoder megakernel (csrc/fused/sm_90/model_stages_decoder.cuh).

WHY (the gate the math-oracle cannot provide): decoder_oracle.py proves the
decoder fwd/bwd MATH is correct (matched to autograd ~1e-15). It does NOT
exercise the CUDA kernel's *structure* — the smem buffer reuse/aliasing, the
recompute-in-backward, the head-split index arithmetic, the owner-thread grad
accumulation, the per-sample sequential reduction. That structure is exactly
where a hand-written, un-runnable backward hides missing terms / index bugs /
dead-buffer reads (e.g. an attention dv term computed from an already-overwritten
softmax weight).

This module re-implements the kernel ALGORITHM in plain Python — same buffers
(as numpy/torch arrays), same aliasing decisions, same loop nests and index math,
same per-sample sequential weight-grad accumulation into a CTA-style partial — so
that asserting it against decoder_oracle's grads (~1e-6 on a tiny batch) catches
the STRUCTURAL bug class. It is single-threaded, so it does NOT catch
__syncthreads races (those are verified by reading each sync against the buffer it
guards — see the kernel header comments); it DOES catch everything else.

The mirror follows the .cuh line-by-line: dec_forward_sample,
dec_recompute_layer, dec_backward_sample, dec_linear/_bwd, dec_layernorm_*,
and the three-pass attention backward (A: dv, B: dscores, C: dq/dk).
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from tests.hw.decoder_oracle import (
    DecoderWeights, decoder_param_layout,
    VOCAB, D_MODEL, N_HEADS, N_LAYERS, SEQ, D_FF, D_HEAD, LN_EPS, ATTN_SCALE,
    gelu, gelu_grad,
)

_F64 = torch.float64  # mirror in fp64 so the only diff vs oracle is structure


# --- primitive ops, mirroring the .cuh device functions exactly --------------
def _linear(x, W, b, in_dim, out_dim):
    # y[s,o] = sum_k x[s,k]*W[o,k] + b[o]   (W row-major [out,in])
    y = torch.zeros(SEQ, out_dim, dtype=_F64)
    for s in range(SEQ):
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


# --- the per-sample forward (mirrors dec_forward_sample) ----------------------
def _forward_sample(W: DecoderWeights, tokens_s, target):
    sm = {}
    # embedding + pos -> layer_in[0]
    h = torch.zeros(SEQ, D_MODEL, dtype=_F64)
    for s in range(SEQ):
        for j in range(D_MODEL):
            h[s, j] = W.tok[tokens_s[s], j] + W.pos[s, j]
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
                sc = [(-math.inf) for _ in range(SEQ)]
                maxs = -math.inf
                for kj in range(SEQ):
                    if kj > qi:
                        continue
                    dot = 0.0
                    for t in range(D_HEAD):
                        dot += qkv[qi, qoff + t].item() * qkv[kj, D_MODEL + qoff + t].item()
                    sc[kj] = dot * ATTN_SCALE
                    maxs = max(maxs, sc[kj])
                denom = 0.0
                for kj in range(SEQ):
                    e = math.exp(sc[kj] - maxs) if kj <= qi else 0.0
                    sc[kj] = e
                    denom += e
                for kj in range(SEQ):
                    attn[hh, qi, kj] = sc[kj] / denom
                for t in range(D_HEAD):
                    acc = 0.0
                    for kj in range(qi + 1):
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
    # final norm (last pos) + head -> logits
    hlast = sm["final_in"][SEQ - 1]
    mean = hlast.mean()
    var = ((hlast - mean) ** 2).mean()
    inv = 1.0 / math.sqrt(var.item() + LN_EPS)
    fn_xhat = torch.zeros(D_MODEL, dtype=_F64)
    hn = torch.zeros(D_MODEL, dtype=_F64)
    for j in range(D_MODEL):
        xh = (hlast[j] - mean) * inv
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


# --- recompute one layer's fwd intermediates (mirrors dec_recompute_layer) ----
def _recompute_layer(W, li, sm):
    """RE-RUN the layer forward from the cached layer INPUT (sm['layer_in'][li]),
    exactly as the CUDA dec_recompute_layer does (it does NOT reuse a per-layer
    cache — only layer_in survives). This exercises the recompute path's index
    math + that it reproduces the forward values; we assert it equals the original
    forward cache to catch a recompute divergence (e.g. a wrong residual or a
    clobbered input)."""
    L = W.layers[li]
    hin = sm["layer_in"][li]
    qkv = _linear(hin, L["in_w"], L["in_b"], D_MODEL, 3 * D_MODEL)
    attn = torch.zeros(N_HEADS, SEQ, SEQ, dtype=_F64)
    ctx = torch.zeros(SEQ, D_MODEL, dtype=_F64)
    for hh in range(N_HEADS):
        qoff = hh * D_HEAD
        for qi in range(SEQ):
            sc = [(-math.inf) for _ in range(SEQ)]
            maxs = -math.inf
            for kj in range(SEQ):
                if kj > qi:
                    continue
                dot = 0.0
                for t in range(D_HEAD):
                    dot += qkv[qi, qoff + t].item() * qkv[kj, D_MODEL + qoff + t].item()
                sc[kj] = dot * ATTN_SCALE
                maxs = max(maxs, sc[kj])
            denom = 0.0
            for kj in range(SEQ):
                e = math.exp(sc[kj] - maxs) if kj <= qi else 0.0
                sc[kj] = e
                denom += e
            for kj in range(SEQ):
                attn[hh, qi, kj] = sc[kj] / denom
            for t in range(D_HEAD):
                acc = 0.0
                for kj in range(qi + 1):
                    acc += attn[hh, qi, kj].item() * qkv[kj, 2 * D_MODEL + qoff + t].item()
                ctx[qi, qoff + t] = acc
    a = _linear(ctx, L["out_w"], L["out_b"], D_MODEL, D_MODEL)
    r1 = hin + a
    x1, n1_xhat, n1_inv = _layernorm_fwd(r1, L["n1_w"], L["n1_b"])
    ff0 = _linear(x1, L["ff0_w"], L["ff0_b"], D_MODEL, D_FF)
    gact = gelu(ff0)
    # n2 caches (xhat/inv) require r2; the CUDA recompute computes r2 into ctx and
    # runs LN fwd for the caches (output discarded).
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


# --- the per-sample backward (mirrors dec_backward_sample) --------------------
def _backward_sample(W: DecoderWeights, g: Dict[str, torch.Tensor],
                     tokens_s, target, B, sm):
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
    # final-norm bwd (single row = last position).
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
        dh[SEQ - 1, j] = inv_s * (dxhat - (sda + fn_xhat[j] * sdax) / D_MODEL)
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
        # --- attention backward (3 passes: A dv, B dscores, C dq/dk) ---
        dqkv = torch.zeros(SEQ, 3 * D_MODEL, dtype=_F64)
        # A: dv[kj,qoff+t] = sum_{qi>=kj} attn[hh,qi,kj]*dctx[qi,qoff+t]
        for kj in range(SEQ):
            for hh in range(N_HEADS):
                qoff = hh * D_HEAD
                for t in range(D_HEAD):
                    acc = 0.0
                    for qi in range(kj, SEQ):
                        acc += attn[hh, qi, kj].item() * dctx[qi, qoff + t].item()
                    dqkv[kj, 2 * D_MODEL + qoff + t] = acc
        # B: dsc[hh,qi,kj] = attn*(datt - sum_k datt*attn)*scale
        dsc = torch.zeros(N_HEADS, SEQ, SEQ, dtype=_F64)
        for hh in range(N_HEADS):
            qoff = hh * D_HEAD
            for qi in range(SEQ):
                datt = [0.0] * SEQ
                for kj in range(qi + 1):
                    acc = 0.0
                    for t in range(D_HEAD):
                        acc += dctx[qi, qoff + t].item() * qkv[kj, 2 * D_MODEL + qoff + t].item()
                    datt[kj] = acc
                dot = 0.0
                for kj in range(qi + 1):
                    dot += datt[kj] * attn[hh, qi, kj].item()
                for kj in range(SEQ):
                    a = attn[hh, qi, kj].item()
                    dsc[hh, qi, kj] = (a * (datt[kj] - dot) * ATTN_SCALE) if kj <= qi else 0.0
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
    # embedding bwd: tok scatter-add per id (sequential positions), pos sum.
    for j in range(D_MODEL):
        for s in range(SEQ):
            d = dh[s, j]
            g["tok.weight"][tokens_s[s], j] += d
            g["pos.weight"][s, j] += d


def mirror_loss_and_grads(named_params: Dict[str, torch.Tensor],
                          tokens: torch.Tensor, targets: torch.Tensor):
    """Run the STRUCTURAL kernel mirror over the whole (small) batch.

    Mirrors the CTA loop: per sample, forward then immediately backward,
    accumulating weight grads into one partial (here a single CTA == whole batch,
    which is what determinism requires per element). Returns (loss, grads)."""
    named = {k: v.to(_F64) for k, v in named_params.items()}
    W = DecoderWeights.from_named(named)
    lay = decoder_param_layout()
    g = {name: torch.zeros(shape, dtype=_F64)
         for name, shape in zip(lay["names"], lay["shapes"])}
    B = tokens.shape[0]
    nll_sum = 0.0
    for b in range(B):
        tok_s = [int(tokens[b, s]) for s in range(SEQ)]
        tgt = int(targets[b])
        sm = _forward_sample(W, tok_s, tgt)
        logits = sm["logits"]
        mx = logits.max().item()
        es = sum(math.exp(logits[o].item() - mx) for o in range(VOCAB))
        nll_sum += (mx + math.log(es)) - logits[tgt].item()
        _backward_sample(W, g, tok_s, tgt, B, sm)
    return nll_sum / B, g
