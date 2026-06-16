"""tests/hw/sg2_kernel_mirror.py — single-threaded STRUCTURAL mirror of the
SuperGrok2 meta-net MEGAKERNEL (csrc/fused/sm_90/opt_stage_supergrok2.cuh).

WHY (the gate the math-oracle cannot provide): the per-op SG2 path
(grokking_optimizers/kernels/sm_90/supergrok2_sm90.cuh :: csa_hca_step_one) is
parity-validated and is the MATH oracle. It does NOT exercise the MEGAKERNEL's
*structure* — the per-CTA workspace carving + REUSE (q staged with g/s then
copied into win_k/win_v; concat reused csa→hca), the sorted/original index
juggling (perm/unsort), the hand-written sequential GRU/PEER reductions, the
low-rank-only indexer, the insertion-sort top-k. That structure is exactly where
a hand-written, un-runnable kernel hides a dead-buffer read / an index bug / a
missing unsort (e.g. attention reading a `concat` slot HCA already overwrote, or
PEER reading `new_gru` from the wrong row order).

This module re-implements the KERNEL ALGORITHM in plain Python — SAME buffers
(as a dict of fp64 arrays), SAME aliasing decisions (q→g/s→win_k/win_v, concat
csa→hca reuse), SAME loop nests / index math (sorted-row owner-computes,
perm/unsort gather) — and asserts it against an INDEPENDENT fp64 ORACLE that
reimplements csa_hca_step_one's math cleanly (different buffers, vectorized).
Agreement to ~1e-12 (fp64) catches the STRUCTURAL bug class. Single-threaded, so
it does NOT catch __syncthreads races (those are verified by reading each sync
against the buffer it guards — see the .cuh stage comments); it DOES catch
everything else.

IMPORTANT — the ORACLE is csa_hca_step_one, NOT the eager CSAHCAMetaNet.forward.
The eager net's lightning indexer is FULL-RANK ( (x@idx_DQ)@idx_UQ ), while the
kernel path is LOW-RANK only ( qI=x@idx_DQ, kI=pooled(x)@idx_K, rank-dim dot ) —
they select different top-k by construction. The parity-validated path is the
CUDA kernel path, so that is the 1e-12 target here. The transcendentals match the
kernel: ptx_expf (≈ accurate in fp64) is just exp in the mirror; the kernel uses
accurate expf in GRU/PEER and ptx_expf in attention/compress, both of which are
ordinary exp at fp64 precision, so the mirror/oracle agree exactly in fp64.
"""

from __future__ import annotations

import math
import numpy as np

_F64 = np.float64

# Toggle for the vectorized hot-path rewrite of oracle_step. When False, the
# original (slow, per-scalar) reference implementations are used instead. The
# vectorized and reference paths are mathematically identical to fp64 rounding;
# the reference path is retained for the self-test (--selftest) and as the
# authoritative semantics. Default True (fast path).
_USE_VECTORIZED = True


# ============================================================================
#  Dims (supergrok2.py constructor defaults + OPTIMIZER_CONFIGS["supergrok2"]).
# ============================================================================
class Dims:
    d_model = 8
    num_heads = 2
    head_dim = 4
    gru_hidden = 4
    num_experts = 144
    expert_hidden = 16
    num_peer_heads = 4
    peer_topk = 4
    csa_compress = 4
    csa_window = 8
    csa_topk = 16
    hca_compress = 128
    indexer_rank = 4
    pk_dim = 12
    gru_in = 2 + 2 * 8          # 18
    peer_in = 4 + 2 * 8 + 2     # 22


# ============================================================================
#  Weight bundle — random fp64 arrays in the EXACT row-major layouts the kernel
#  reads (SG2Weights). One generator so mirror + oracle share identical weights.
# ============================================================================
def make_weights(seed=0, D=Dims):
    rng = np.random.default_rng(seed)
    d, rk, gh, gi, pin = D.d_model, D.indexer_rank, D.gru_hidden, D.gru_in, D.peer_in
    nph, pkd, ne, eh, half = D.num_peer_heads, D.pk_dim, D.num_experts, D.expert_hidden, D.d_model // 2
    xin = gi + gh

    def r(*shape, scale=0.1):
        return (rng.standard_normal(shape) * scale).astype(_F64)

    w = dict(
        input_proj_W=r(d, 2), input_proj_b=r(d),
        csa_q_W=r(d, d), csa_k_W=r(d, d), csa_v_W=r(d, d), csa_out_W=r(d, d),
        csa_compress_w=r(D.csa_window),
        csa_idx_DQ=r(d, rk), csa_idx_K=r(d, rk),
        hca_q_W=r(d, d), hca_k_W=r(d, d), hca_v_W=r(d, d), hca_out_W=r(d, d),
        gru_Wz=r(gh, xin), gru_bz=r(gh),
        gru_Wr=r(gh, xin), gru_br=r(gh),
        gru_Wh=r(gh, xin), gru_bh=r(gh),
        peer_query_Ws=r(nph, d, pin),
        prod_keys_A=r(nph, pkd, half), prod_keys_B=r(nph, pkd, half),
        expert_W1=r(ne, eh), expert_b1=r(ne, eh),
        expert_W2=r(ne, eh), expert_b2=r(ne),
    )
    return w


# ============================================================================
#  Per-element building blocks — fp64 transcriptions of csrc/algorithms/
#  supergrok2.h. (ptx_expf/expf are both `exp` at fp64; ptx_fma is `a*b+c`.)
# ============================================================================
def _csa_pooled(x_seq, compress_w, j, dd, N, d, m, win):
    """sg2_csa_compress_kv: online-softmax weighted pool over the window."""
    base = j * m
    run_max = -math.inf
    for wq in range(win):
        if base + wq >= N:
            break
        run_max = max(run_max, compress_w[wq])
    if not math.isfinite(run_max):
        return 0.0
    denom = 0.0
    acc = 0.0
    for wq in range(win):
        t = base + wq
        if t >= N:
            break
        e = math.exp(compress_w[wq] - run_max)
        denom += e
        acc += e * x_seq[t, dd]
    return acc / denom if denom > 0.0 else 0.0


def _hca_pooled(x_seq, j, dd, N, d, m):
    """sg2_hca_compress_kv (mean pool, hca_w=nullptr branch)."""
    base = j * m
    acc = 0.0
    cnt = 0
    for wq in range(m):
        t = base + wq
        if t >= N:
            break
        acc += x_seq[t, dd]
        cnt += 1
    return acc / cnt if cnt > 0 else 0.0


def _attn_accum(q, k, v, run_max, run_denom, acc, scale, hd):
    """sg2_attention_score_and_accumulate (online softmax)."""
    dot = 0.0
    for e in range(hd):
        dot += q[e] * k[e]
    s = dot * scale
    m_old = run_max
    m_new = max(m_old, s)
    corr = math.exp(m_old - m_new)
    p = math.exp(s - m_new)
    run_denom = run_denom * corr + p
    for e in range(hd):
        acc[e] = acc[e] * corr + p * v[e]
    return m_new, run_denom


def _online_attn_reference(qh, kh, vh, sel_idx, winkh, winvh, scale, D):
    """REFERENCE (slow, per-scalar) online-softmax attention — the EXACT body
    of the original ``online_attn`` closure inside ``oracle_step``, lifted to a
    module-level function so the self-test can compare the vectorized path
    against it. Combines, per (query t, head h), an online softmax over the
    UNION of (a) the selected key rows ``sel_idx[t]`` (or ALL key rows when
    ``sel_idx is None``, the HCA case), filtered to ``0<=s<M``, and (b) the
    causal window rows ``range(max(t-csa_window+1,0), t+1)`` scored against
    ``winkh``/``winvh``. ``qh:[N,H,hd]``, ``kh``/``vh:[M,H,hd]``,
    ``winkh``/``winvh:[N,H,hd]``."""
    N, H, hd = qh.shape
    ctx = np.zeros((N, H, hd), _F64)
    for t in range(N):
        for h in range(H):
            rm = -math.inf
            rd = 0.0
            acc = np.zeros(hd, _F64)
            keys = sel_idx[t] if sel_idx is not None else range(kh.shape[0])
            for s in keys:
                if s < 0 or s >= kh.shape[0]:
                    continue
                sc = float(qh[t, h] @ kh[s, h]) * scale
                nm = max(rm, sc)
                corr = math.exp(rm - nm)
                p = math.exp(sc - nm)
                rd = rd * corr + p
                acc = acc * corr + p * vh[s, h]
                rm = nm
            # window over raw-token projections (win_k/win_v).
            w0 = max(t - D.csa_window + 1, 0)
            for s in range(w0, t + 1):
                sc = float(qh[t, h] @ winkh[s, h]) * scale
                nm = max(rm, sc)
                corr = math.exp(rm - nm)
                p = math.exp(sc - nm)
                rd = rd * corr + p
                acc = acc * corr + p * winvh[s, h]
                rm = nm
            ctx[t, h] = acc / rd if rd > 0 else acc
    return ctx.reshape(N, D.d_model)


def _online_attn_vectorized(qh, kh, vh, sel_idx_arr, winkh, winvh, scale, D):
    """VECTORIZED equivalent of ``_online_attn_reference``. Mathematically
    identical to fp64 rounding (softmax is permutation-invariant, so a single
    numerically-stable softmax over the combined {selected-key, window} logits
    reproduces the streaming online softmax exactly).

    ``sel_idx_arr`` is either ``None`` (HCA: every key row is a key for every
    query) or an int array ``[N, topk]`` of selected compressed-key rows (the
    CSA case; entries ``<0`` or ``>=M`` are masked out, matching the reference's
    ``if s<0 or s>=M: continue``). The exact key SET and the exact value vector
    each key contributes are preserved: selected keys contribute ``vh[s]`` and
    window keys contribute ``winvh[s]``, kept as DISTINCT entries (no dedup)."""
    N, H, hd = qh.shape
    M = kh.shape[0]
    neg_inf = -np.inf

    # ---- selected-key (or all-key) block ----
    if sel_idx_arr is None:
        # HCA: every one of the M key rows is a key for every query.
        # logits_sel[t, j, h] = (qh[t,h] . kh[j,h]) * scale
        logits_sel = np.einsum("thd,jhd->tjh", qh, kh, optimize=True) * scale  # [N,M,H]
        val_sel = vh                                                            # [M,H,hd]
        sel_gather = None
        sel_valid = None  # all valid
    else:
        sel = sel_idx_arr                                   # [N, topk] int
        sel_valid = (sel >= 0) & (sel < M)                  # [N, topk]
        sel_clamped = np.where(sel_valid, sel, 0)           # safe gather index
        # gather keys/values for each (t, slot): [N, topk, H, hd]
        kh_sel = kh[sel_clamped]
        val_sel = vh[sel_clamped]                           # [N, topk, H, hd]
        # logits[t, slot, h] = sum_hd qh[t,h,:] * kh_sel[t,slot,h,:]
        logits_sel = np.einsum("thd,tshd->tsh", qh, kh_sel, optimize=True) * scale  # [N,topk,H]
        logits_sel = np.where(sel_valid[:, :, None], logits_sel, neg_inf)
        sel_gather = val_sel

    # ---- causal window block ----
    W = D.csa_window
    t_idx = np.arange(N)
    # window rows for query t are [t-W+1, ..., t]; columns ordered ascending.
    win_cols = t_idx[:, None] - np.arange(W - 1, -1, -1)[None, :]   # [N, W]
    win_valid = win_cols >= 0                                       # [N, W]
    win_clamped = np.where(win_valid, win_cols, 0)
    winkh_w = winkh[win_clamped]                                    # [N, W, H, hd]
    winvh_w = winvh[win_clamped]
    logits_win = np.einsum("thd,twhd->twh", qh, winkh_w, optimize=True) * scale  # [N,W,H]
    logits_win = np.where(win_valid[:, :, None], logits_win, neg_inf)

    # ---- combined numerically-stable softmax over the union ----
    if sel_idx_arr is None:
        m_sel = logits_sel.max(axis=1)                              # [N,H]
    else:
        m_sel = logits_sel.max(axis=1)                              # [N,H] (>=neg_inf)
    m_win = logits_win.max(axis=1)                                  # [N,H]
    m = np.maximum(m_sel, m_win)                                    # [N,H] global max
    m_safe = np.where(np.isfinite(m), m, 0.0)                       # avoid -inf - -inf

    p_sel = np.exp(logits_sel - m_safe[:, None, :])                # [N,S,H]; masked->0
    p_win = np.exp(logits_win - m_safe[:, None, :])                # [N,W,H]; masked->0

    rd = p_sel.sum(axis=1) + p_win.sum(axis=1)                      # [N,H]

    # numerator: sum_s p_sel * val_sel  +  sum_w p_win * winvh_w
    if sel_idx_arr is None:
        num = np.einsum("tjh,jhd->thd", p_sel, val_sel, optimize=True)        # [N,H,hd]
    else:
        num = np.einsum("tsh,tshd->thd", p_sel, sel_gather, optimize=True)    # [N,H,hd]
    num = num + np.einsum("twh,twhd->thd", p_win, winvh_w, optimize=True)     # [N,H,hd]

    rd_safe = np.where(rd > 0.0, rd, 1.0)
    ctx = np.where(rd[:, :, None] > 0.0, num / rd_safe[:, :, None], num)      # [N,H,hd]
    return ctx.reshape(N, D.d_model)


# ============================================================================
#  THE STRUCTURAL MIRROR — follows opt_stage_supergrok2.cuh stage-for-stage,
#  including the workspace REUSE/ALIASING. `ws` is a dict of fp64 arrays mirroring
#  the carved SG2Work slots; we reuse them exactly as the kernel does so a
#  dead-buffer read shows up as a mismatch vs the oracle.
# ============================================================================
def mirror_step(w, grad, sharp, perm, unsort, gru_state, exp_avg, exp_avg_sq,
                mu, slow, param, scalars, D=Dims):
    """Mirror ONE tensor's full pipeline (S0..S5). Mutates gru_state/exp_avg/
    exp_avg_sq/mu/slow/param IN PLACE (as the kernel does), returns expert_out
    (sorted), csa_ctx, hca_ctx, new_gru (sorted) for assertions."""
    N = grad.shape[0]
    d = D.d_model
    hd = D.head_dim
    rk = D.indexer_rank
    gh = D.gru_hidden
    Nc = (N + D.csa_compress - 1) // D.csa_compress
    Nh = (N + D.hca_compress - 1) // D.hca_compress
    topk = min(D.csa_topk, Nc)
    topk1 = max(topk, 1)
    scale = 1.0 / math.sqrt(hd)

    # --- carve the per-CTA workspace (exact SG2Work slots) ---
    ws = dict(
        x_sorted=np.zeros((N, d), _F64),
        csa_ctx=np.zeros((N, d), _F64),
        hca_ctx=np.zeros((N, d), _F64),
        q=np.zeros((N, d), _F64),
        c_k=np.zeros((Nc, d), _F64),
        c_v=np.zeros((Nc, d), _F64),
        win_k=np.zeros((N, d), _F64),   # NOTE: also reused as g_sorted/[N] later
        win_v=np.zeros((N, d), _F64),   # NOTE: also reused as s_sorted/[N] later
        qI=np.zeros((N, rk), _F64),
        kI=np.zeros((Nc, rk), _F64),
        sel=np.full((N, topk1), -1, np.int64),
        concat=np.zeros((N, d), _F64),
        new_gru=np.zeros((N, gh), _F64),
        expert_out=np.zeros(N, _F64),
    )

    # S0: input projection + gather sorted features (fused: sorted-row owner).
    for r in range(N):
        src = perm[r]
        g = grad[src]
        s = sharp[src]
        if not np.isfinite(g):
            g = 0.0
        if not np.isfinite(s):
            s = 0.0
        for dd in range(d):
            ws["x_sorted"][r, dd] = (w["input_proj_W"][dd, 0] * g
                                     + w["input_proj_W"][dd, 1] * s
                                     + w["input_proj_b"][dd])

    # S1: CSA context -----------------------------------------------------
    # q = x_sorted @ csa_q_W.T  (project: out[t,oc]=sum_k W[oc,k]*x[t,k])
    for t in range(N):
        for oc in range(d):
            acc = 0.0
            for k in range(d):
                acc += w["csa_q_W"][oc, k] * ws["x_sorted"][t, k]
            ws["q"][t, oc] = acc
    # compress c_k, c_v (csa)
    for j in range(Nc):
        for oc in range(d):
            ak = av = 0.0
            for k in range(d):
                p = _csa_pooled(ws["x_sorted"], w["csa_compress_w"], j, k, N, d,
                                D.csa_compress, D.csa_window)
                ak += w["csa_k_W"][oc, k] * p
                av += w["csa_v_W"][oc, k] * p
            ws["c_k"][j, oc] = ak
            ws["c_v"][j, oc] = av
    # window k, v = x_sorted @ {k_W, v_W}.T
    for t in range(N):
        for oc in range(d):
            ak = av = 0.0
            for k in range(d):
                ak += w["csa_k_W"][oc, k] * ws["x_sorted"][t, k]
                av += w["csa_v_W"][oc, k] * ws["x_sorted"][t, k]
            ws["win_k"][t, oc] = ak
            ws["win_v"][t, oc] = av
    # indexer qI = x_sorted @ idx_DQ ; kI = pooled(x_sorted) @ idx_K
    for t in range(N):
        for r in range(rk):
            acc = 0.0
            for m in range(d):
                acc += ws["x_sorted"][t, m] * w["csa_idx_DQ"][m, r]
            ws["qI"][t, r] = acc
    for s in range(Nc):
        for r in range(rk):
            acc = 0.0
            for m in range(d):
                p = _csa_pooled(ws["x_sorted"], w["csa_compress_w"], s, m, N, d,
                                D.csa_compress, D.csa_window)
                acc += p * w["csa_idx_K"][m, r]
            ws["kI"][s, r] = acc
    # top-k (insertion sort, descending) — sg2_csa_index_score = qI·kI/sqrt(rank)
    inv_rk = 1.0 / math.sqrt(rk)
    for t in range(N):
        best_s = [-math.inf] * D.csa_topk
        best_i = [-1] * D.csa_topk
        K = min(topk1, D.csa_topk)
        for s in range(Nc):
            sc = 0.0
            for r in range(rk):
                sc += ws["qI"][t, r] * ws["kI"][s, r]
            sc *= inv_rk
            if sc > best_s[K - 1]:
                p = K - 1
                while p > 0 and best_s[p - 1] < sc:
                    best_s[p] = best_s[p - 1]
                    best_i[p] = best_i[p - 1]
                    p -= 1
                best_s[p] = sc
                best_i[p] = s
        for i in range(K):
            ws["sel"][t, i] = best_i[i]
    # CSA attention (per t,h) → concat
    for t in range(N):
        for h in range(D.num_heads):
            hoff = h * hd
            acc = [0.0] * hd
            run_max = -math.inf
            run_denom = 0.0
            qv = ws["q"][t, hoff:hoff + hd]
            for i in range(topk1):
                s = int(ws["sel"][t, i])
                if s < 0 or s >= Nc:
                    continue
                run_max, run_denom = _attn_accum(
                    qv, ws["c_k"][s, hoff:hoff + hd], ws["c_v"][s, hoff:hoff + hd],
                    run_max, run_denom, acc, scale, hd)
            w0 = max(t - D.csa_window + 1, 0)
            for s in range(w0, t + 1):
                run_max, run_denom = _attn_accum(
                    qv, ws["win_k"][s, hoff:hoff + hd], ws["win_v"][s, hoff:hoff + hd],
                    run_max, run_denom, acc, scale, hd)
            inv = (1.0 / run_denom) if run_denom > 0.0 else 0.0
            for e in range(hd):
                ws["concat"][t, hoff + e] = acc[e] * inv
    # out proj: csa_ctx = concat @ out_W.T
    for t in range(N):
        for oc in range(d):
            acc = 0.0
            for k in range(d):
                acc += w["csa_out_W"][oc, k] * ws["concat"][t, k]
            ws["csa_ctx"][t, oc] = acc

    # S2: HCA context (REUSE q/c_k/c_v/win_k/win_v/concat) ----------------
    for t in range(N):
        for oc in range(d):
            acc = 0.0
            for k in range(d):
                acc += w["hca_q_W"][oc, k] * ws["x_sorted"][t, k]
            ws["q"][t, oc] = acc
    # NOTE: c_k/c_v carved at Nc (csa); HCA uses Nh <= Nc entries — write the
    # first Nh rows, mirroring the kernel writing c_out[0:Nh] into the same buffer.
    for j in range(Nh):
        for oc in range(d):
            ak = av = 0.0
            for k in range(d):
                p = _hca_pooled(ws["x_sorted"], j, k, N, d, D.hca_compress)
                ak += w["hca_k_W"][oc, k] * p
                av += w["hca_v_W"][oc, k] * p
            ws["c_k"][j, oc] = ak
            ws["c_v"][j, oc] = av
    for t in range(N):
        for oc in range(d):
            ak = av = 0.0
            for k in range(d):
                ak += w["hca_k_W"][oc, k] * ws["x_sorted"][t, k]
                av += w["hca_v_W"][oc, k] * ws["x_sorted"][t, k]
            ws["win_k"][t, oc] = ak
            ws["win_v"][t, oc] = av
    for t in range(N):
        for h in range(D.num_heads):
            hoff = h * hd
            acc = [0.0] * hd
            run_max = -math.inf
            run_denom = 0.0
            qv = ws["q"][t, hoff:hoff + hd]
            for s in range(Nh):
                run_max, run_denom = _attn_accum(
                    qv, ws["c_k"][s, hoff:hoff + hd], ws["c_v"][s, hoff:hoff + hd],
                    run_max, run_denom, acc, scale, hd)
            w0 = max(t - D.csa_window + 1, 0)
            for s in range(w0, t + 1):
                run_max, run_denom = _attn_accum(
                    qv, ws["win_k"][s, hoff:hoff + hd], ws["win_v"][s, hoff:hoff + hd],
                    run_max, run_denom, acc, scale, hd)
            inv = (1.0 / run_denom) if run_denom > 0.0 else 0.0
            for e in range(hd):
                ws["concat"][t, hoff + e] = acc[e] * inv
    for t in range(N):
        for oc in range(d):
            acc = 0.0
            for k in range(d):
                acc += w["hca_out_W"][oc, k] * ws["concat"][t, k]
            ws["hca_ctx"][t, oc] = acc

    # g_sorted/s_sorted materialization via the EXACT kernel aliasing:
    #   stage into q[:,0]/q[:,1], then copy to win_k[:,0]/win_v[:,0] contiguous.
    for r in range(N):
        src = perm[r]
        g = grad[src]
        s = sharp[src]
        if not np.isfinite(g):
            g = 0.0
        if not np.isfinite(s):
            s = 0.0
        ws["q"][r, 0] = g
        ws["q"][r, 1] = s
    for r in range(N):
        ws["win_k"][r, 0] = ws["q"][r, 0]   # g_sorted contiguous (col 0)
        ws["win_v"][r, 0] = ws["q"][r, 1]   # s_sorted contiguous (col 0)
    g_sorted = ws["win_k"][:, 0]
    s_sorted = ws["win_v"][:, 0]

    # S3: matrix-GRU forward (sorted), persist gru_state in ORIGINAL order ----
    gi = D.gru_in
    xin = gi + gh
    for r in range(N):
        src = perm[r]
        xb = np.zeros(xin, _F64)
        xb[0] = g_sorted[r]
        xb[1] = s_sorted[r]
        for k in range(d):
            xb[2 + k] = ws["csa_ctx"][r, k]
        for k in range(d):
            xb[2 + d + k] = ws["hca_ctx"][r, k]
        h_old = gru_state[src].copy()
        for k in range(gh):
            xb[gi + k] = h_old[k]
        zg = np.zeros(gh, _F64)
        rg = np.zeros(gh, _F64)
        for o in range(gh):
            az = w["gru_bz"][o]
            ar = w["gru_br"][o]
            for k in range(xin):
                az += w["gru_Wz"][o, k] * xb[k]
                ar += w["gru_Wr"][o, k] * xb[k]
            zg[o] = 1.0 / (1.0 + math.exp(-az))
            rg[o] = 1.0 / (1.0 + math.exp(-ar))
        for k in range(gh):
            xb[gi + k] = rg[k] * h_old[k]
        for o in range(gh):
            ah = w["gru_bh"][o]
            for k in range(xin):
                ah += w["gru_Wh"][o, k] * xb[k]
            ht = math.tanh(ah)
            hn = (1.0 - zg[o]) * h_old[o] + zg[o] * ht
            ws["new_gru"][r, o] = hn
            gru_state[src, o] = hn

    # S4: PEER routing + experts (sorted) → expert_out *= rescale -------------
    half = d // 2
    pkd = D.pk_dim
    K = D.peer_topk
    T = 10.0
    for r in range(N):
        pin = np.zeros(D.peer_in, _F64)
        o = 0
        for k in range(gh):
            pin[o] = ws["new_gru"][r, k]; o += 1
        for k in range(d):
            pin[o] = ws["csa_ctx"][r, k]; o += 1
        for k in range(d):
            pin[o] = ws["hca_ctx"][r, k]; o += 1
        pin[o] = g_sorted[r]; o += 1
        pin[o] = s_sorted[r]; o += 1
        gval = g_sorted[r]
        head_sum = 0.0
        for h in range(D.num_peer_heads):
            Wq = w["peer_query_Ws"][h]      # [d, peer_in]
            A = w["prod_keys_A"][h]          # [pk_dim, half]
            B = w["prod_keys_B"][h]
            query = np.zeros(d, _F64)
            for oc in range(d):
                acc = 0.0
                for k in range(D.peer_in):
                    acc += Wq[oc, k] * pin[k]
                query[oc] = acc
            sa = np.zeros(pkd, _F64)
            sb = np.zeros(pkd, _F64)
            for n in range(pkd):
                aa = bb = 0.0
                for k in range(half):
                    aa += query[k] * A[n, k]
                    bb += query[half + k] * B[n, k]
                sa[n] = aa
                sb[n] = bb
            ka = min(K, pkd)
            kb = ka
            ta_v = [-math.inf] * K; ta_i = [0] * K
            tb_v = [-math.inf] * K; tb_i = [0] * K
            for n in range(pkd):
                if sa[n] > ta_v[ka - 1]:
                    p = ka - 1
                    while p > 0 and ta_v[p - 1] < sa[n]:
                        ta_v[p] = ta_v[p - 1]; ta_i[p] = ta_i[p - 1]; p -= 1
                    ta_v[p] = sa[n]; ta_i[p] = n
                if sb[n] > tb_v[kb - 1]:
                    p = kb - 1
                    while p > 0 and tb_v[p - 1] < sb[n]:
                        tb_v[p] = tb_v[p - 1]; tb_i[p] = tb_i[p - 1]; p -= 1
                    tb_v[p] = sb[n]; tb_i[p] = n
            ma = ta_v[0]; mb = tb_v[0]
            soft_a = [0.0] * ka; soft_b = [0.0] * kb
            za = zb = 0.0
            for i in range(ka):
                soft_a[i] = math.exp((ta_v[i] - ma) * T); za += soft_a[i]
            for i in range(kb):
                soft_b[i] = math.exp((tb_v[i] - mb) * T); zb += soft_b[i]
            for i in range(ka):
                soft_a[i] /= za
            for i in range(kb):
                soft_b[i] /= zb
            head_out = 0.0
            for ia in range(ka):
                for ib in range(kb):
                    eidx = ta_i[ia] * pkd + tb_i[ib]
                    eidx = max(0, min(eidx, D.num_experts - 1))
                    rw = soft_a[ia] * soft_b[ib]
                    mlp = w["expert_b2"][eidx]
                    for hdt in range(D.expert_hidden):
                        z = w["expert_W1"][eidx, hdt] * gval + w["expert_b1"][eidx, hdt]
                        if z < 0.0:
                            z = 0.0
                        mlp += w["expert_W2"][eidx, hdt] * z
                    head_out += rw * mlp
            head_sum += head_out
        head_sum /= D.num_peer_heads
        ws["expert_out"][r] = head_sum * scalars["rescale"]

    # S5: unsort + sg2_apply_step (canonical math) ---------------------------
    bc1 = max(scalars["bc1"], 1e-30)
    bc2 = max(scalars["bc2"], 1e-30)
    for i in range(N):
        eo = ws["expert_out"][unsort[i]]
        g = grad[i]
        p = param[i]
        mu_new = scalars["gru_decay"] * mu[i] + (1.0 - scalars["gru_decay"]) * eo
        mu[i] = mu_new
        slow_new = scalars["alpha"] * slow[i] + (1.0 - scalars["alpha"]) * g
        slow[i] = slow_new
        smart = g + scalars["alpha"] * mu_new + scalars["lamb_eff"] * slow_new
        m = scalars["beta1"] * exp_avg[i] + (1.0 - scalars["beta1"]) * smart
        v = scalars["beta2"] * exp_avg_sq[i] + (1.0 - scalars["beta2"]) * smart * smart
        exp_avg[i] = m
        exp_avg_sq[i] = v
        update = (m / bc1) / (math.sqrt(v / bc2) + scalars["eps"])
        param[i] = p - scalars["lr"] * (update + scalars["wd"] * p)

    return dict(expert_out=ws["expert_out"].copy(),
                csa_ctx=ws["csa_ctx"].copy(),
                hca_ctx=ws["hca_ctx"].copy(),
                new_gru=ws["new_gru"].copy())


# ============================================================================
#  THE INDEPENDENT ORACLE — reimplements csa_hca_step_one's math CLEANLY
#  (vectorized numpy, fresh buffers, NO kernel aliasing). Same math, different
#  structure → asserting mirror == oracle isolates STRUCTURE bugs.
# ============================================================================
def _csa_topk_select(qI, kI, inv_sqrt_rk, topk):
    """Per-query top-`topk` compressed keys from the CSA low-rank indexer scores
    S = (qI @ kI.T) * inv_sqrt_rk, returned in DESCENDING score order ([N, k]).

    This O(N·Nc) selection (Nc = kI.shape[0] ≈ N/csa_compress) is the single
    dominant cost of the oracle at real param-tensor sizes (N up to ~65K ⇒ an N×Nc
    ≈ 1e9-element score matrix); the naive full per-row stable argsort made the SG2
    gate cells effectively hang. The SELECTION is a discrete ranking (WHICH
    compressed keys) — NOT the meta-net math — and the production kernel itself
    computes this indexer in bf16/fp32 on the GPU, so doing the score matmul + top-k
    on the GPU (or in fp32 on the CPU) is faithful to the kernel. The fp64
    ground-truth that matters (attention weights, GRU/PEER/Adam) is untouched, and
    for tie-free scores the selected SET is identical across fp64/fp32/GPU; the
    downstream attention softmax over that set is permutation-invariant, so neither
    the dtype of the ranking nor the intra-k order changes csa_ctx.

    Fast path: torch on CUDA (matmul + torch.topk, fp32) — milliseconds on an H100.
    Fallback: chunked numpy argpartition (fp32 scores, bounded peak memory) when
    CUDA/torch is unavailable (e.g. the standalone CPU `--selftest`)."""
    N = qI.shape[0]
    Nc = kI.shape[0]
    k = min(topk, Nc)
    # --- GPU fast path (the gate runs with torch+CUDA available) ---
    try:
        import torch
        if torch.cuda.is_available():
            with torch.no_grad():
                q = torch.from_numpy(np.ascontiguousarray(qI, dtype=np.float32)).cuda()
                kk = torch.from_numpy(np.ascontiguousarray(kI, dtype=np.float32)).cuda()
                sc = (q @ kk.T) * float(inv_sqrt_rk)              # [N, Nc] fp32 (GPU)
                idx = torch.topk(sc, k, dim=1, largest=True, sorted=True).indices
                out = idx.to("cpu").numpy().astype(np.intp)
                del q, kk, sc, idx
                return out
    except Exception:
        pass  # fall through to the pure-numpy path
    # --- numpy fallback: chunked top-k via argpartition (no full O(Nc·logNc) sort) ---
    qI32 = np.ascontiguousarray(qI, dtype=np.float32)
    kI32T = np.ascontiguousarray(kI.T, dtype=np.float32)
    inv32 = np.float32(inv_sqrt_rk)
    sel = np.empty((N, k), dtype=np.intp)
    _BLK = 16384
    for _s in range(0, N, _BLK):
        _e = min(_s + _BLK, N)
        blk = (qI32[_s:_e] @ kI32T) * inv32                      # [b, Nc] fp32
        if k >= Nc:
            sel[_s:_e] = np.argsort(-blk, axis=1, kind="stable")[:, :k]
        else:
            part = np.argpartition(blk, Nc - k, axis=1)[:, Nc - k:]  # unordered top-k
            ps = np.take_along_axis(blk, part, axis=1)
            order = np.argsort(-ps, axis=1, kind="stable")          # order the k (cheap)
            sel[_s:_e] = np.take_along_axis(part, order, axis=1)
    return sel


def oracle_step(w, grad, sharp, perm, unsort, gru_state, exp_avg, exp_avg_sq,
                mu, slow, param, scalars, D=Dims):
    N = grad.shape[0]
    d = D.d_model
    hd = D.head_dim
    rk = D.indexer_rank
    gh = D.gru_hidden
    Nc = (N + D.csa_compress - 1) // D.csa_compress
    Nh = (N + D.hca_compress - 1) // D.hca_compress
    topk = max(min(D.csa_topk, Nc), 1)
    scale = 1.0 / math.sqrt(hd)
    H = D.num_heads

    g_all = grad.astype(_F64)
    s_all = sharp.astype(_F64)
    g_all = np.where(np.isfinite(g_all), g_all, 0.0)
    s_all = np.where(np.isfinite(s_all), s_all, 0.0)

    # input proj on sorted rows.
    g_sorted = g_all[perm]
    s_sorted = s_all[perm]
    inp = np.stack([g_sorted, s_sorted], axis=-1)              # [N,2]
    x_sorted = inp @ w["input_proj_W"].T + w["input_proj_b"]   # [N,d]

    def softmax_pool_csa():
        # weighted pooled x for each (j, channel) via online softmax over window.
        cw = w["csa_compress_w"]
        out = np.zeros((Nc, d), _F64)
        for j in range(Nc):
            base = j * D.csa_compress
            taps = [base + wq for wq in range(D.csa_window) if base + wq < N]
            if not taps:
                continue
            logits = np.array([cw[t - base] for t in taps], _F64)
            m = logits.max()
            e = np.exp(logits - m)
            wts = e / e.sum()
            out[j] = (wts[:, None] * x_sorted[taps]).sum(0)
        return out

    def meanpool_hca():
        out = np.zeros((Nh, d), _F64)
        for j in range(Nh):
            base = j * D.hca_compress
            taps = [base + wq for wq in range(D.hca_compress) if base + wq < N]
            if taps:
                out[j] = x_sorted[taps].mean(0)
        return out

    pooled_csa = softmax_pool_csa()       # [Nc, d]  (pooled raw features)

    def heads(t):
        return t.reshape(t.shape[0], H, hd)

    def online_attn(qh, kh, vh, sel_idx, window_only=False):
        # qh:[N,H,hd]; kh/vh:[M,H,hd]; sel_idx: per-query key rows (compressed),
        # an int array [N,topk] (CSA) or None (HCA: all M key rows). winkh/winvh
        # are read from the enclosing oracle_step scope (the window projections).
        # Dispatch to vectorized or reference body; both are fp64-identical.
        if _USE_VECTORIZED:
            return _online_attn_vectorized(qh, kh, vh, sel_idx, winkh, winvh,
                                           scale, D)
        # Reference path needs sel_idx as a per-row iterable; the vectorized
        # path passes an array, the reference accepts either (None or [N,topk]).
        return _online_attn_reference(qh, kh, vh, sel_idx, winkh, winvh,
                                      scale, D)

    # CSA
    q_csa = heads(x_sorted @ w["csa_q_W"].T)
    c_k = (pooled_csa @ w["csa_k_W"].T)
    c_v = (pooled_csa @ w["csa_v_W"].T)
    c_kh = heads(c_k); c_vh = heads(c_v)
    winkh = heads(x_sorted @ w["csa_k_W"].T)
    winvh = heads(x_sorted @ w["csa_v_W"].T)
    qI = x_sorted @ w["csa_idx_DQ"]                       # [N, rk]
    kI = pooled_csa @ w["csa_idx_K"]                      # [Nc, rk]
    # CSA low-rank indexer top-k (the O(N·Nc) dominant cost — see _csa_topk_select).
    # Reproduces the same per-row top-`topk` of (qI@kI.T)/sqrt(rk) in descending
    # score; GPU-accelerated when available, chunked-numpy fallback otherwise.
    inv_sqrt_rk = 1.0 / math.sqrt(rk)
    sel_idx = _csa_topk_select(qI, kI, inv_sqrt_rk, topk)   # [N, topk] descending
    csa_ctx = online_attn(q_csa, c_kh, c_vh, sel_idx)
    csa_ctx = csa_ctx @ w["csa_out_W"].T

    # HCA
    pooled_hca = meanpool_hca()
    q_hca = heads(x_sorted @ w["hca_q_W"].T)
    hk = heads(pooled_hca @ w["hca_k_W"].T)
    hv = heads(pooled_hca @ w["hca_v_W"].T)
    winkh = heads(x_sorted @ w["hca_k_W"].T)
    winvh = heads(x_sorted @ w["hca_v_W"].T)
    hca_ctx = online_attn(q_hca, hk, hv, None)
    hca_ctx = hca_ctx @ w["hca_out_W"].T

    # matrix-GRU (sorted)
    h_old = gru_state[perm].astype(_F64)                 # [N, gh]
    gru_input = np.concatenate(
        [g_sorted[:, None], s_sorted[:, None], csa_ctx, hca_ctx], axis=-1)
    xh = np.concatenate([gru_input, h_old], axis=-1)
    z = 1.0 / (1.0 + np.exp(-(xh @ w["gru_Wz"].T + w["gru_bz"])))
    r = 1.0 / (1.0 + np.exp(-(xh @ w["gru_Wr"].T + w["gru_br"])))
    xrh = np.concatenate([gru_input, r * h_old], axis=-1)
    h_tilde = np.tanh(xrh @ w["gru_Wh"].T + w["gru_bh"])
    new_gru = (1.0 - z) * h_old + z * h_tilde            # [N, gh] sorted
    # persist in original order
    new_gru_orig = np.zeros_like(new_gru)
    new_gru_orig[perm] = new_gru
    gru_state[...] = new_gru_orig

    # PEER (sorted)
    peer_input = np.concatenate(
        [new_gru, csa_ctx, hca_ctx, g_sorted[:, None], s_sorted[:, None]], axis=-1)
    half = d // 2
    pkd = D.pk_dim
    K = D.peer_topk
    T = 10.0
    total = np.zeros(N, _F64)
    for h in range(D.num_peer_heads):
        query = peer_input @ w["peer_query_Ws"][h].T     # [N, d]
        qa = query[:, :half]
        qb = query[:, half:]
        sa = qa @ w["prod_keys_A"][h].T                  # [N, pk_dim]
        sb = qb @ w["prod_keys_B"][h].T
        ka = min(K, pkd)
        kb = ka
        ta_i = np.argsort(-sa, axis=1, kind="stable")[:, :ka]
        tb_i = np.argsort(-sb, axis=1, kind="stable")[:, :kb]
        ta_v = np.take_along_axis(sa, ta_i, axis=1)
        tb_v = np.take_along_axis(sb, tb_i, axis=1)
        soft_a = np.exp((ta_v - ta_v.max(1, keepdims=True)) * T)
        soft_a /= soft_a.sum(1, keepdims=True)
        soft_b = np.exp((tb_v - tb_v.max(1, keepdims=True)) * T)
        soft_b /= soft_b.sum(1, keepdims=True)
        if _USE_VECTORIZED:
            # Vectorize over t and the ka*kb (ia,ib) expert pairs. The (ia,ib)
            # accumulation is an order-independent sum, so a batched reduction
            # reproduces head_out[t] exactly (fp64 rounding).
            # eidx[t,ia,ib] = clip(ta_i[t,ia]*pkd + tb_i[t,ib], 0, num_experts-1)
            eidx = ta_i[:, :, None] * pkd + tb_i[:, None, :]          # [N,ka,kb]
            eidx = np.clip(eidx, 0, D.num_experts - 1)
            rw = soft_a[:, :, None] * soft_b[:, None, :]             # [N,ka,kb]
            # expert MLP on the scalar g_sorted[t]:
            #   z = relu(expert_W1[e]*g + expert_b1[e]) ; expert_W1/b1: [.,eh]
            #   mlp = expert_W2[e] @ z + expert_b2[e]   ; expert_W2: [.,eh]
            W1 = w["expert_W1"][eidx]                                # [N,ka,kb,eh]
            b1 = w["expert_b1"][eidx]                                # [N,ka,kb,eh]
            W2 = w["expert_W2"][eidx]                                # [N,ka,kb,eh]
            b2 = w["expert_b2"][eidx]                                # [N,ka,kb]
            gcol = g_sorted[:, None, None, None]                     # [N,1,1,1]
            zrelu = np.maximum(W1 * gcol + b1, 0.0)                  # [N,ka,kb,eh]
            mlp = np.einsum("abce,abce->abc", W2, zrelu, optimize=True) + b2  # [N,ka,kb]
            head_out = (rw * mlp).sum(axis=(1, 2))                   # [N]
        else:
            head_out = np.zeros(N, _F64)
            for t in range(N):
                for ia in range(ka):
                    for ib in range(kb):
                        eidx = int(ta_i[t, ia]) * pkd + int(tb_i[t, ib])
                        eidx = max(0, min(eidx, D.num_experts - 1))
                        rw = soft_a[t, ia] * soft_b[t, ib]
                        zrelu = np.maximum(
                            w["expert_W1"][eidx] * g_sorted[t] + w["expert_b1"][eidx], 0.0)
                        mlp = float(w["expert_W2"][eidx] @ zrelu) + w["expert_b2"][eidx]
                        head_out[t] += rw * mlp
        total += head_out
    total /= D.num_peer_heads
    expert_out_sorted = total * scalars["rescale"]

    # apply (unsort), sg2_apply_step
    bc1 = max(scalars["bc1"], 1e-30)
    bc2 = max(scalars["bc2"], 1e-30)
    eo = expert_out_sorted[unsort]                       # back to original order
    g = g_all
    p = param.astype(_F64)
    mu_new = scalars["gru_decay"] * mu + (1.0 - scalars["gru_decay"]) * eo
    mu[...] = mu_new
    slow_new = scalars["alpha"] * slow + (1.0 - scalars["alpha"]) * g
    slow[...] = slow_new
    smart = g + scalars["alpha"] * mu_new + scalars["lamb_eff"] * slow_new
    m = scalars["beta1"] * exp_avg + (1.0 - scalars["beta1"]) * smart
    v = scalars["beta2"] * exp_avg_sq + (1.0 - scalars["beta2"]) * smart * smart
    exp_avg[...] = m
    exp_avg_sq[...] = v
    update = (m / bc1) / (np.sqrt(v / bc2) + scalars["eps"])
    param[...] = p - scalars["lr"] * (update + scalars["wd"] * p)

    return dict(expert_out=expert_out_sorted, csa_ctx=csa_ctx, hca_ctx=hca_ctx,
                new_gru=new_gru)


def make_scalars():
    return dict(rescale=0.1, alpha=0.98, gru_decay=0.9, lamb_eff=2.0,
                beta1=0.9, beta2=0.999, lr=1e-3, wd=0.01, eps=1e-8,
                bc1=1.0 - 0.9 ** 5, bc2=1.0 - 0.999 ** 5)


def make_sorted_perm(grad):
    """Reproduce csa_hca_step_one's sort: |grad| ASCENDING, stable.
       perm[r] = original row of the r-th sorted row;
       unsort[i] = sorted position of original row i."""
    perm = np.argsort(np.abs(grad), kind="stable").astype(np.int64)
    unsort = np.argsort(perm, kind="stable").astype(np.int64)
    return perm, unsort


# ============================================================================
#  SELF-TEST — vectorized oracle_step vs the retained reference (slow) path.
#  Asserts every output (in-place-mutated state + returned dict) agrees to
#  < 1e-9 (max-abs and max-rel) for a range of N, and that the vectorized
#  N=65536 path runs in < 1.0s. Pure numpy, CPU-only, no GPU/torch.
# ============================================================================
def _selftest(Ns=(16, 256, 1024, 16384, 65536), D=Dims, tol=1e-9, verbose=True):
    import copy as _copy
    import time as _time

    global _USE_VECTORIZED
    gh = D.gru_hidden
    w = make_weights(seed=0, D=D)        # shared fp64 weights (Dims-consistent)
    scalars = make_scalars()

    def _build_inputs(N, rng):
        # random grad/sharp + zero-init state (as the gate drives it), all fp64.
        grad = rng.standard_normal(N).astype(_F64)
        sharp = (rng.standard_normal(N) * 0.5).astype(_F64)
        perm, unsort = make_sorted_perm(grad)
        gru_state = rng.standard_normal((N, gh)).astype(_F64) * 0.1
        exp_avg = rng.standard_normal(N).astype(_F64) * 0.1
        exp_avg_sq = (np.abs(rng.standard_normal(N)) * 0.1).astype(_F64)
        mu = rng.standard_normal(N).astype(_F64) * 0.1
        slow = rng.standard_normal(N).astype(_F64) * 0.1
        param = rng.standard_normal(N).astype(_F64)
        return dict(grad=grad, sharp=sharp, perm=perm, unsort=unsort,
                    gru_state=gru_state, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq,
                    mu=mu, slow=slow, param=param)

    def _run(inp):
        # oracle_step mutates gru_state/exp_avg/exp_avg_sq/mu/slow/param in place.
        ret = oracle_step(w, inp["grad"], inp["sharp"], inp["perm"], inp["unsort"],
                          inp["gru_state"], inp["exp_avg"], inp["exp_avg_sq"],
                          inp["mu"], inp["slow"], inp["param"], scalars, D=D)
        out = dict(
            param=inp["param"], m=inp["exp_avg"], v=inp["exp_avg_sq"],
            mu=inp["mu"], slow=inp["slow"], gru_state=inp["gru_state"],
            expert_out=ret["expert_out"], csa_ctx=ret["csa_ctx"],
            hca_ctx=ret["hca_ctx"], new_gru=ret["new_gru"],
        )
        return out

    def _err(a, b):
        a = np.asarray(a, _F64); b = np.asarray(b, _F64)
        diff = np.abs(a - b)
        max_abs = float(diff.max()) if diff.size else 0.0
        denom = np.maximum(np.abs(a), np.abs(b))
        rel = np.where(denom > 0, diff / denom, 0.0)
        max_rel = float(rel.max()) if rel.size else 0.0
        return max_abs, max_rel

    all_ok = True
    for N in Ns:
        rng = np.random.default_rng(0)
        inp = _build_inputs(N, rng)

        # reference (slow) path on a deep copy.
        _USE_VECTORIZED = False
        ref_out = _run(_copy.deepcopy(inp))

        # vectorized path on a fresh deep copy + timing.
        _USE_VECTORIZED = True
        vinp = _copy.deepcopy(inp)
        t0 = _time.perf_counter()
        vec_out = _run(vinp)
        dt = _time.perf_counter() - t0

        worst_abs = 0.0
        worst_rel = 0.0
        worst_key = ""
        for k in ref_out:
            ma, mr = _err(vec_out[k], ref_out[k])
            if ma > worst_abs:
                worst_abs, worst_key = ma, k
            worst_rel = max(worst_rel, mr)
        ok = (worst_abs < tol) and (worst_rel < tol)
        all_ok = all_ok and ok
        if N == 65536:
            ok = ok and (dt < 1.0)
            all_ok = all_ok and (dt < 1.0)
        if verbose:
            print(f"  N={N:6d}  max|Δ|={worst_abs:.3e} ({worst_key:10s})  "
                  f"max relΔ={worst_rel:.3e}  vec_time={dt:7.4f}s  "
                  f"{'OK' if ok else 'FAIL'}"
                  + ("  (<1.0s req)" if N == 65536 else ""))

    _USE_VECTORIZED = True   # restore default fast path
    if verbose:
        print(f"\n  RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
              f"(tol {tol:.0e}, N=65536 must be < 1.0s)")
    return all_ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="sg2_kernel_mirror self-test")
    ap.add_argument("--selftest", action="store_true",
                    help="run vectorized-vs-reference oracle_step self-test")
    args = ap.parse_args()
    if args.selftest:
        ok = _selftest()
        raise SystemExit(0 if ok else 1)
    ap.print_help()
