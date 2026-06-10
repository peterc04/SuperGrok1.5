"""tests/hw/mamba_kernel_mirror.py — single-threaded STRUCTURAL mirror of the
CUDA Mamba megakernel (csrc/fused/sm_90/model_stage_mamba3.cuh).

WHY (the gate the math-oracle cannot provide): mamba_oracle.py proves the Mamba
fwd/bwd MATH is correct (matched to autograd ~1e-15). It does NOT exercise the
CUDA kernel's *structure* — the per-channel REGISTER scan recurrence (h[STATE]
unrolled over seq=8), the scan BACKWARD's forward-recompute + reverse-scan, the
NON-causal conv1d transpose index math, the THREE-path dx_main accumulation
(D-skip + scan + x_proj), the softplus'-folded-in-scan, the per-(t,s) dB/dC
REDUCTION over channels, and the owner-thread weight-grad accumulation. That
structure is exactly where a hand-written, un-runnable backward hides missing
terms / index bugs / dead-buffer reads (e.g. a conv future-peek transposed the
wrong way, or a dx_main path dropped).

This module re-implements the kernel ALGORITHM in plain Python — same per-channel
scan loops, same forward-recompute-in-scan-backward, same conv transpose, same
3-path dx_main sum, same per-sample sequential weight-grad accumulation into a
CTA-style partial — so that asserting it against mamba_oracle's grads (~1e-12 on
a tiny batch) catches the STRUCTURAL bug class. It is single-threaded, so it does
NOT catch __syncthreads races (those are verified by reading each sync against
the buffer it guards — see the kernel header comments); it DOES catch everything
else.

The mirror follows the .cuh line-by-line: mb_forward_sample, mb_scan_fwd,
mb_conv1d_fwd, the gate/skip/out_proj, and mb_backward_sample's order 1..13
(LN bwd, out_proj bwd, gate+skip bwd, scan bwd, dt_proj bwd, x_proj bwd, conv
bwd, in_proj bwd, embedding bwd).
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from tests.hw.mamba_oracle import (
    MambaWeights, mamba_param_layout,
    VOCAB, P_HEAD, D_MODEL, N_LAYERS, SEQ, D_INNER, STATE, DT_RANK, DBC, CONV_K,
    LN_EPS,
)

_F64 = torch.float64  # mirror in fp64 so the only diff vs oracle is structure


# --- elementwise (mirror the .cuh device fns) --------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _silu_grad(x):
    s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def _softplus(x):
    return x if x > 20.0 else math.log1p(math.exp(x))


def _softplus_grad(x):
    return _sigmoid(x)


# --- primitive linear (mirror mb_linear) -------------------------------------
def _linear(x, W, b, in_dim, out_dim):
    # y[s,o] = sum_k x[s,k]*W[o,k] + b[o]
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


# --- NON-causal depthwise conv1d (mirror mb_conv1d_fwd) ----------------------
def _conv1d_fwd(inp, W, b):
    # out[t,c] = b[c] + sum_k W[c,0,k]*in[t+k-1,c]   (zero-pad outside [0,SEQ))
    out = torch.zeros(SEQ, D_INNER, dtype=_F64)
    for t in range(SEQ):
        for c in range(D_INNER):
            acc = b[c].item()
            for k in range(CONV_K):
                tt = t + k - 1
                if 0 <= tt < SEQ:
                    acc += W[c, 0, k].item() * inp[tt, c].item()
            out[t, c] = acc
    return out


# --- per-channel register scan forward (mirror mb_scan_fwd) ------------------
def _scan_fwd(A_log, x_main, dt_pre, Bmat, Cmat):
    """Per channel j: h[STATE] in 'registers', unroll t. dt_pre is PRE-softplus.
    Returns y_scan [SEQ, D_INNER]."""
    y = torch.zeros(SEQ, D_INNER, dtype=_F64)
    for j in range(D_INNER):
        A = [-math.exp(A_log[j, s].item()) for s in range(STATE)]
        h = [0.0] * STATE
        for t in range(SEQ):
            xv = x_main[t, j].item()
            dt_t = _softplus(dt_pre[t, j].item())
            yacc = 0.0
            for s in range(STATE):
                adec = math.exp(dt_t * A[s])
                h[s] = adec * h[s] + dt_t * Bmat[t, s].item() * xv
                yacc += Cmat[t, s].item() * h[s]
            y[t, j] = yacc
    return y


# --- per-channel scan backward (mirror mb_scan_bwd) --------------------------
def _scan_bwd(A_log, x_main, dt_pre, Bmat, Cmat, dy_scan):
    """Per channel j: recompute fwd (h_hist in 'registers'), reverse-scan.
    Returns (dx_main_scan [SEQ,D_INNER], ddt_pre [SEQ,D_INNER],
             dBmat [SEQ,STATE], dCmat [SEQ,STATE], dA_log [D_INNER,STATE]).
    dBmat/dCmat are SUMMED over channels (reduced); dA_log is per-channel."""
    dx_main_scan = torch.zeros(SEQ, D_INNER, dtype=_F64)
    ddt_pre = torch.zeros(SEQ, D_INNER, dtype=_F64)
    dBmat = torch.zeros(SEQ, STATE, dtype=_F64)
    dCmat = torch.zeros(SEQ, STATE, dtype=_F64)
    dA_log = torch.zeros(D_INNER, STATE, dtype=_F64)
    for j in range(D_INNER):
        A = [-math.exp(A_log[j, s].item()) for s in range(STATE)]
        # forward recompute -> hh[t+1] = h_t (hh[0] = h_{-1} = 0)
        hh = [[0.0] * STATE for _ in range(SEQ + 1)]
        dt_save = [0.0] * SEQ
        a_save = [[0.0] * STATE for _ in range(SEQ)]
        for t in range(SEQ):
            xv = x_main[t, j].item()
            dt_t = _softplus(dt_pre[t, j].item())
            dt_save[t] = dt_t
            for s in range(STATE):
                adec = math.exp(dt_t * A[s])
                a_save[t][s] = adec
                hh[t + 1][s] = adec * hh[t][s] + dt_t * Bmat[t, s].item() * xv
        # reverse scan
        gh = [0.0] * STATE
        dA = [0.0] * STATE
        for t in reversed(range(SEQ)):
            xv = x_main[t, j].item()
            dt_t = dt_save[t]
            dyv = dy_scan[t, j].item()
            dx_acc = 0.0
            ddt_acc = 0.0
            for s in range(STATE):
                h_t = hh[t + 1][s]
                h_tm1 = hh[t][s]
                adec = a_save[t][s]
                cc = Cmat[t, s].item()
                bb = Bmat[t, s].item()
                dCmat[t, s] += h_t * dyv         # reduced over channels
                gh[s] += cc * dyv
                dBmat[t, s] += dt_t * xv * gh[s] # reduced over channels
                dx_acc += bb * dt_t * gh[s]
                ddt_acc += (A[s] * adec * h_tm1 + bb * xv) * gh[s]
                dA[s] += dt_t * adec * h_tm1 * gh[s]
                gh[s] = adec * gh[s]
            dx_main_scan[t, j] = dx_acc
            ddt_pre[t, j] = ddt_acc * _softplus_grad(dt_pre[t, j].item())
        for s in range(STATE):
            dA_log[j, s] = dA[s] * A[s]
    return dx_main_scan, ddt_pre, dBmat, dCmat, dA_log


# --- per-sample forward (mirror mb_forward_sample) ---------------------------
def _forward_sample(W: MambaWeights, tokens_s):
    sm = {}
    h = torch.zeros(SEQ, D_MODEL, dtype=_F64)
    for s in range(SEQ):
        for j in range(D_MODEL):
            h[s, j] = W.tok[tokens_s[s], j] + W.pos[s, j]
    sm["layer_in"] = [h.clone(), None]
    for li in range(N_LAYERS):
        L = W.layers[li]
        hin = sm["layer_in"][li]
        # in_proj -> x_main_raw | z
        xz = _linear(hin, L["in_w"], None, D_MODEL, 2 * D_INNER)
        x_main_raw = xz[:, :D_INNER].clone()
        z = xz[:, D_INNER:].clone()
        # conv1d (non-causal) -> conv (pre-silu); x_main = silu(conv)
        conv = _conv1d_fwd(x_main_raw, L["conv_w"], L["conv_b"])
        x_main = torch.zeros(SEQ, D_INNER, dtype=_F64)
        for t in range(SEQ):
            for c in range(D_INNER):
                x_main[t, c] = _silu(conv[t, c].item())
        # x_proj -> dt_raw, Bmat, Cmat
        x_dbc = _linear(x_main, L["x_proj_w"], None, D_INNER, DBC)
        dt_raw = x_dbc[:, :DT_RANK].clone()
        Bmat = x_dbc[:, DT_RANK:DT_RANK + STATE].clone()
        Cmat = x_dbc[:, DT_RANK + STATE:].clone()
        # dt_pre = dt_proj(dt_raw)+bias
        dt_pre = _linear(dt_raw, L["dt_proj_w"], L["dt_proj_b"], DT_RANK, D_INNER)
        # selective scan
        y_scan = _scan_fwd(L["A_log"], x_main, dt_pre, Bmat, Cmat)
        # gate+skip+out_proj
        y_gated = torch.zeros(SEQ, D_INNER, dtype=_F64)
        for t in range(SEQ):
            for c in range(D_INNER):
                yskip = y_scan[t, c].item() + x_main[t, c].item() * L["D"][c].item()
                y_gated[t, c] = yskip * _silu(z[t, c].item())
        y_out = _linear(y_gated, L["out_w"], None, D_INNER, D_MODEL)
        r = y_out + hin
        h2, ln_xhat, ln_inv = _layernorm_fwd(r, L["n_w"], L["n_b"])
        sm[f"L{li}"] = dict(x_main_raw=x_main_raw, z=z, conv=conv, x_main=x_main,
                            dt_pre=dt_pre, Bmat=Bmat, Cmat=Cmat, y_scan=y_scan,
                            y_gated=y_gated, ln_xhat=ln_xhat, ln_inv=ln_inv)
        if li + 1 < N_LAYERS:
            sm["layer_in"][li + 1] = h2
        else:
            sm["final_in"] = h2
    # final norm (last pos) + head
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
    logits = torch.zeros(P_HEAD, dtype=_F64)
    for o in range(P_HEAD):
        acc = W.out_b[o].item()
        for k in range(D_MODEL):
            acc += hn[k].item() * W.out_w[o, k].item()
        logits[o] = acc
    sm["fn_xhat"] = fn_xhat
    sm["fn_inv"] = inv
    sm["logits"] = logits
    return sm


# --- per-sample backward (mirror mb_backward_sample, order 1..13) ------------
def _backward_sample(W: MambaWeights, g: Dict[str, torch.Tensor], tokens_s,
                     target, B, sm):
    # CE backward
    logits = sm["logits"]
    mx = logits.max().item()
    es = sum(math.exp(logits[o].item() - mx) for o in range(P_HEAD))
    dlogits = torch.zeros(P_HEAD, dtype=_F64)
    for o in range(P_HEAD):
        smo = math.exp(logits[o].item() - mx) / es
        dlogits[o] = (smo - (1.0 if o == target else 0.0)) / B
    # head bwd
    fn_xhat = sm["fn_xhat"]
    hn = torch.zeros(D_MODEL, dtype=_F64)
    for j in range(D_MODEL):
        hn[j] = fn_xhat[j] * W.norm_w[j] + W.norm_b[j]
    for o in range(P_HEAD):
        g["out.bias"][o] += dlogits[o]
        for j in range(D_MODEL):
            g["out.weight"][o, j] += dlogits[o] * hn[j]
    dhn = torch.zeros(D_MODEL, dtype=_F64)
    for j in range(D_MODEL):
        acc = 0.0
        for o in range(P_HEAD):
            acc += dlogits[o].item() * W.out_w[o, j].item()
        dhn[j] = acc
    # final-norm bwd (single row last)
    for j in range(D_MODEL):
        g["norm.bias"][j] += dhn[j]
        g["norm.weight"][j] += dhn[j] * fn_xhat[j]
    sda = sum((dhn[j] * W.norm_w[j]).item() for j in range(D_MODEL))
    sdax = sum((dhn[j] * W.norm_w[j] * fn_xhat[j]).item() for j in range(D_MODEL))
    inv_s = sm["fn_inv"]
    dh = torch.zeros(SEQ, D_MODEL, dtype=_F64)
    for j in range(D_MODEL):
        dxhat = dhn[j] * W.norm_w[j]
        dh[SEQ - 1, j] = inv_s * (dxhat - (sda + fn_xhat[j] * sdax) / D_MODEL)
    # per-layer backward (reverse)
    for li in reversed(range(N_LAYERS)):
        L = W.layers[li]
        gp = f"layers.{li}."
        c = sm[f"L{li}"]
        hin = sm["layer_in"][li]
        x_main_raw, z, conv = c["x_main_raw"], c["z"], c["conv"]
        x_main, dt_pre = c["x_main"], c["dt_pre"]
        Bmat, Cmat, y_scan, y_gated = c["Bmat"], c["Cmat"], c["y_scan"], c["y_gated"]
        ln_xhat, ln_inv = c["ln_xhat"], c["ln_inv"]
        # (1) LN bwd: dh -> dr ; accumulate norm g/b
        dr = torch.zeros(SEQ, D_MODEL, dtype=_F64)
        for j in range(D_MODEL):
            for s in range(SEQ):
                d = dh[s, j]
                g[gp + "norm.bias"][j] += d
                g[gp + "norm.weight"][j] += d * ln_xhat[s, j]
        for s in range(SEQ):
            sda = sum((dh[s, j] * L["n_w"][j]).item() for j in range(D_MODEL))
            sdax = sum((dh[s, j] * L["n_w"][j] * ln_xhat[s, j]).item()
                       for j in range(D_MODEL))
            for j in range(D_MODEL):
                dxhat = dh[s, j] * L["n_w"][j]
                dr[s, j] = ln_inv[s] * (dxhat - (sda + ln_xhat[s, j] * sdax) / D_MODEL)
        # (2) r = y_out + residual: dy_out = dr ; dx_residual = dr
        dy_out = dr
        dx_residual = dr.clone()
        # (3) out_proj bwd: dW_out += dy_out^T @ y_gated ; dy_gated = dy_out @ out_w
        for o in range(D_MODEL):
            for cc in range(D_INNER):
                acc = 0.0
                for s in range(SEQ):
                    acc += dy_out[s, o].item() * y_gated[s, cc].item()
                g[gp + "out_proj.weight"][o, cc] += acc
        dy_gated = torch.zeros(SEQ, D_INNER, dtype=_F64)
        for s in range(SEQ):
            for cc in range(D_INNER):
                acc = 0.0
                for o in range(D_MODEL):
                    acc += dy_out[s, o].item() * L["out_w"][o, cc].item()
                dy_gated[s, cc] = acc
        # (4-6) gate+skip bwd: dz, dy_scan, dx_main(D-path), dD
        dz = torch.zeros(SEQ, D_INNER, dtype=_F64)
        dy_scan = torch.zeros(SEQ, D_INNER, dtype=_F64)
        dx_main = torch.zeros(SEQ, D_INNER, dtype=_F64)   # accumulator (3 paths)
        for cc in range(D_INNER):
            dDc = 0.0
            for s in range(SEQ):
                sz = _silu(z[s, cc].item())
                xm = x_main[s, cc].item()
                yskip = y_scan[s, cc].item() + xm * L["D"][cc].item()
                dyg = dy_gated[s, cc].item()
                dyskip = dyg * sz
                dsz = dyg * yskip
                dz[s, cc] = dsz * _silu_grad(z[s, cc].item())
                dDc += dyskip * xm
                dy_scan[s, cc] = dyskip
                dx_main[s, cc] += dyskip * L["D"][cc].item()   # D-path
            g[gp + "D"][cc] += dDc
        # (7) scan bwd -> dx_main(scan path), ddt_pre, dB, dC, dA_log
        dx_scan, ddt_pre, dBmat, dCmat, dA_log = _scan_bwd(
            L["A_log"], x_main, dt_pre, Bmat, Cmat, dy_scan)
        for j in range(D_INNER):
            for s in range(STATE):
                g[gp + "A_log"][j, s] += dA_log[j, s]
        for s in range(SEQ):
            for cc in range(D_INNER):
                dx_main[s, cc] += dx_scan[s, cc]               # scan path
        # (8) dt_proj bwd: recompute dt_raw = (x_proj(x_main))[:, :dt_rank]
        dt_raw = torch.zeros(SEQ, DT_RANK, dtype=_F64)
        for s in range(SEQ):
            for o in range(DT_RANK):
                acc = 0.0
                for k in range(D_INNER):
                    acc += x_main[s, k].item() * L["x_proj_w"][o, k].item()
                dt_raw[s, o] = acc
        for o in range(D_INNER):
            for i in range(DT_RANK):
                acc = 0.0
                for s in range(SEQ):
                    acc += ddt_pre[s, o].item() * dt_raw[s, i].item()
                g[gp + "dt_proj.weight"][o, i] += acc
            acc = 0.0
            for s in range(SEQ):
                acc += ddt_pre[s, o].item()
            g[gp + "dt_proj.bias"][o] += acc
        ddt_raw = torch.zeros(SEQ, DT_RANK, dtype=_F64)
        for s in range(SEQ):
            for i in range(DT_RANK):
                acc = 0.0
                for o in range(D_INNER):
                    acc += ddt_pre[s, o].item() * L["dt_proj_w"][o, i].item()
                ddt_raw[s, i] = acc
        # (9) dx_dbc = cat[ddt_raw, dBmat, dCmat] ; x_proj bwd
        dx_dbc = torch.zeros(SEQ, DBC, dtype=_F64)
        for s in range(SEQ):
            for i in range(DT_RANK):
                dx_dbc[s, i] = ddt_raw[s, i]
            for k in range(STATE):
                dx_dbc[s, DT_RANK + k] = dBmat[s, k]
                dx_dbc[s, DT_RANK + STATE + k] = dCmat[s, k]
        for o in range(DBC):
            for i in range(D_INNER):
                acc = 0.0
                for s in range(SEQ):
                    acc += dx_dbc[s, o].item() * x_main[s, i].item()
                g[gp + "x_proj.weight"][o, i] += acc
        for s in range(SEQ):
            for i in range(D_INNER):
                acc = 0.0
                for o in range(DBC):
                    acc += dx_dbc[s, o].item() * L["x_proj_w"][o, i].item()
                dx_main[s, i] += acc                           # x_proj path
        # (10-11) dconv = dx_main * silu'(conv) ; conv bwd
        dconv = torch.zeros(SEQ, D_INNER, dtype=_F64)
        for s in range(SEQ):
            for cc in range(D_INNER):
                dconv[s, cc] = dx_main[s, cc].item() * _silu_grad(conv[s, cc].item())
        dx_main_raw = torch.zeros(SEQ, D_INNER, dtype=_F64)
        for cc in range(D_INNER):
            dbc_local = 0.0
            for t in range(SEQ):
                dyv = dconv[t, cc].item()
                dbc_local += dyv
                for k in range(CONV_K):
                    tt = t + k - 1
                    if 0 <= tt < SEQ:
                        g[gp + "conv1d.weight"][cc, 0, k] += dyv * x_main_raw[tt, cc].item()
            g[gp + "conv1d.bias"][cc] += dbc_local
        for tt in range(SEQ):
            for cc in range(D_INNER):
                acc = 0.0
                for k in range(CONV_K):
                    t = tt - k + 1   # tt = t+k-1 -> t = tt-k+1
                    if 0 <= t < SEQ:
                        acc += L["conv_w"][cc, 0, k].item() * dconv[t, cc].item()
                dx_main_raw[tt, cc] = acc
        # (12-13) in_proj bwd: dxz = cat[dx_main_raw, dz] ; dx = dx_inproj + dx_residual
        for o in range(2 * D_INNER):
            dyo = dx_main_raw if o < D_INNER else dz
            oo = o if o < D_INNER else (o - D_INNER)
            for i in range(D_MODEL):
                acc = 0.0
                for s in range(SEQ):
                    acc += dyo[s, oo].item() * hin[s, i].item()
                g[gp + "in_proj.weight"][o, i] += acc
        dh_new = torch.zeros(SEQ, D_MODEL, dtype=_F64)
        for s in range(SEQ):
            for i in range(D_MODEL):
                acc = 0.0
                for o in range(D_INNER):
                    acc += dx_main_raw[s, o].item() * L["in_w"][o, i].item()
                    acc += dz[s, o].item() * L["in_w"][D_INNER + o, i].item()
                dh_new[s, i] = acc + dx_residual[s, i].item()
        dh = dh_new
    # embedding bwd
    for j in range(D_MODEL):
        for s in range(SEQ):
            d = dh[s, j]
            g["tok.weight"][tokens_s[s], j] += d
            g["pos.weight"][s, j] += d


def mirror_loss_and_grads(named_params: Dict[str, torch.Tensor],
                          tokens: torch.Tensor, targets: torch.Tensor):
    """Run the STRUCTURAL kernel mirror over the whole (small) batch.

    Mirrors the CTA loop: per sample, forward then immediately backward,
    accumulating weight grads into one partial (single CTA == whole batch, which
    is what determinism requires per element). Returns (loss, grads)."""
    named = {k: v.to(_F64) for k, v in named_params.items()}
    W = MambaWeights.from_named(named)
    lay = mamba_param_layout()
    g = {name: torch.zeros(shape, dtype=_F64)
         for name, shape in zip(lay["names"], lay["shapes"])}
    B = tokens.shape[0]
    nll_sum = 0.0
    for b in range(B):
        tok_s = [int(tokens[b, s]) for s in range(SEQ)]
        tgt = int(targets[b])
        sm = _forward_sample(W, tok_s)
        logits = sm["logits"]
        mx = logits.max().item()
        es = sum(math.exp(logits[o].item() - mx) for o in range(P_HEAD))
        nll_sum += (mx + math.log(es)) - logits[tgt].item()
        _backward_sample(W, g, tok_s, tgt, B, sm)
    return nll_sum / B, g
