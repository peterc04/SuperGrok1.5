#pragma once
// SuperGrok v2 — vendor-neutral algorithm header.
// Migrated and consolidated from:
//   - csrc/device/optimizers/sm_90/supergrok2_sm90.cuh (REAL device templates)
//   - csrc/kernels/cuda/sm_90/supergrok2_fwd.cuh
//   - csrc/kernels/cuda/sm_90/supergrok2_bwd.cuh
//   - csrc/kernels/cuda/sm_90/supergrok2_warp_specialized.cuh
//
// CSA/HCA compressed-attention + 4-Head PEER + per-element GRU + Adam pipeline.
// (Replaces the former Mamba-3 selective scan sequence mixer — see spec §2/§3b.)
//
// Per-step pipeline:
//   (1) input_proj_sort     : [grad, sharpness] -> [N, d_model], sort keys = |grad|
//   (2) CSA/HCA attention   : compressed-sparse (m=4, top-k, +window) -> csa_ctx and
//                              heavily-compressed (m'=128, dense, +window) -> hca_ctx.
//                              Built from: KV compression (sg2_csa_compress_kv /
//                              sg2_hca_compress_kv), lightning-indexer top-k
//                              (sg2_csa_index_score), and an online-softmax core
//                              (sg2_attention_score_and_accumulate / _finalize).
//   (3) peer_route          : product-key expert routing, top-4 of 144 experts
//   (4) gru_step            : per-element GRU integrates expert output with temporal state
//   (5) apply               : smart_grad + Adam + trust-ratio + decoupled weight decay
//
// Backward (used by bilevel meta-learning):
//   (6) bilevel_precompute  : recompute forward q/k/v (+indexer) projections for adjoint
//
// The heavy math (GEMMs, parallel scans, cluster reductions) is in the
// per-backend primitives; this header contains the per-element building
// blocks that are vendor-neutral.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/algorithms/adamw.h"  // pulled in for the merged moe_adam_step

// ── inlined from former csrc/common/utils.cuh (Phase3 S0) ──
#if GROK_CUDA
#ifndef SG_INLINE_PTX_FAST_RSQRT_NR
#define SG_INLINE_PTX_FAST_RSQRT_NR
// Fast reciprocal sqrt via PTX rsqrt.approx.f32 + Newton-Raphson refinement.
// 2-3x faster than sqrtf(x) + fdividef for Adam denominator.
__device__ __forceinline__ float fast_rsqrt_nr(float x) {
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    // One Newton-Raphson iteration: r = r * (1.5 - 0.5 * x * r * r)
    r = r * (1.5f - 0.5f * x * r * r);
    return r;
}
#endif  // SG_INLINE_PTX_FAST_RSQRT_NR

#ifndef SG_INLINE_PTX_PTX_FMA
#define SG_INLINE_PTX_PTX_FMA
// Fused multiply-add via PTX fma.rn.f32 — ensures single FMA instruction.
// Critical for affine_combine inner loop (8 FMAs per composition).
__device__ __forceinline__ float ptx_fma(float a, float b, float c) {
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}
#endif  // SG_INLINE_PTX_PTX_FMA

#ifndef SG_INLINE_PTX_PTX_EXP2
#define SG_INLINE_PTX_PTX_EXP2
// Fast exp2 approximation via PTX ex2.approx.f32.
// Used in Mamba scan: exp(A * dt) = exp2(A * dt / ln2).
__device__ __forceinline__ float ptx_exp2(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
#endif  // SG_INLINE_PTX_PTX_EXP2

#ifndef SG_INLINE_PTX_PTX_EXPF
#define SG_INLINE_PTX_PTX_EXPF
// Fast exp via exp2: exp(x) = exp2(x * log2(e))
__device__ __forceinline__ float ptx_expf(float x) {
    return ptx_exp2(x * 1.4426950408889634f);  // log2(e)
}
#endif  // SG_INLINE_PTX_PTX_EXPF
#endif  // GROK_CUDA

#if GROK_HIP
#ifndef SG_INLINE_PTX_FAST_RSQRT_NR
#define SG_INLINE_PTX_FAST_RSQRT_NR
__device__ __forceinline__ float fast_rsqrt_nr(float x) { return rsqrtf(x); }
#endif  // SG_INLINE_PTX_FAST_RSQRT_NR
#ifndef SG_INLINE_PTX_PTX_FMA
#define SG_INLINE_PTX_PTX_FMA
__device__ __forceinline__ float ptx_fma(float a, float b, float c) { return fmaf(a, b, c); }
#endif  // SG_INLINE_PTX_PTX_FMA
#ifndef SG_INLINE_PTX_PTX_EXPF
#define SG_INLINE_PTX_PTX_EXPF
__device__ __forceinline__ float ptx_expf(float x) { return expf(x); }
#endif  // SG_INLINE_PTX_PTX_EXPF
#endif  // GROK_HIP

namespace sg { namespace algorithms {

// Compile-time maximums (must match types.h constants).
constexpr int SG2_MAX_D_MODEL = 64;     // upper bound on per-element feature width

// ── CSA/HCA compressed-attention maximums (spec §2, §3) ──
// These bound per-thread register arrays used by the device helpers below.
constexpr int SG2_CSA_WINDOW_MAX  = 16;   // max sliding-window / KV-pool width (CSA_WINDOW=8 default)
constexpr int SG2_CSA_TOPK_MAX    = 64;   // max top-k compressed entries per query (CSA_TOPK=16 default)
constexpr int SG2_INDEXER_RANK_MAX = 8;   // max lightning-indexer low rank (INDEXER_RANK=4 default)
constexpr int SG2_HCA_COMPRESS    = 128;  // heavily-compressed pooling stride m' (spec §2 HCA)

// =========================================================================
//  Forward (1): Input Projection + Sort Key
//  x_out[idx, d] = proj_W[d,0]*grad + proj_W[d,1]*sharp + proj_b[d]
//  sort_key      = |grad|
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void sg2_input_proj_sort(
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharpness,
    float* __restrict__ x_out,
    float* __restrict__ sort_keys,
    int* __restrict__ sort_indices,
    const float* __restrict__ proj_W,
    const float* __restrict__ proj_b,
    const int64_t idx,
    const int64_t N,
    const int d_model
) {
    if (idx >= N) return;

    float g = static_cast<float>(grad[idx]);
    float s = static_cast<float>(sharpness[idx]);
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(s)) s = 0.0f;

    #pragma unroll 4
    for (int d = 0; d < d_model; d++) {
        x_out[idx * d_model + d] = proj_W[d * 2] * g + proj_W[d * 2 + 1] * s + proj_b[d];
    }
    sort_keys[idx]    = fabsf(g);
    sort_indices[idx] = idx;
}

// =========================================================================
//  Forward (2a): CSA — Compressed KV pooling (spec §2 CSA step 1)
//
//  Equation (spec §2):
//      c_kv[j, :] = Σ_{w=0..W-1} softmax(compress_w)[w] · x[j·m + w, :]
//
//  where m = CSA_COMPRESS (stride), W = CSA_WINDOW (pool width), and
//  compress_w are learned per-window-position pooling logits (softmax-
//  normalized to a probability distribution over the window). Produces one
//  compressed entry j; compressed length Nc = ceil(N / m). This per-output-
//  element form computes a single feature channel `d` of compressed entry
//  `j`, suitable for a grid-stride kernel over (j, d) pairs.
//
//  FP32 accumulation. `x_seq` is the sorted feature sequence [N, d_model].
//  Out-of-range window taps (j·m + w >= N) are skipped and the softmax is
//  renormalized over the valid taps only (causal/edge-safe pooling).
// =========================================================================

template <typename feat_t>
__device__ __forceinline__ float sg2_csa_compress_kv(
    const feat_t* __restrict__ x_seq,        // [N, d_model] sorted features
    const float*  __restrict__ compress_w,   // [csa_window] learned pooling logits
    const int j,                             // compressed-entry index (0..Nc-1)
    const int d,                             // feature channel (0..d_model-1)
    const int64_t N,
    const int d_model,
    const int csa_compress,                  // stride m
    const int csa_window                     // pool width W
) {
    const int base = j * csa_compress;       // first raw token in this window

    // Online softmax over the (valid) window logits, fused with the weighted
    // pool: keep running max for numerical stability (spec §2 uses softmax(bias)).
    float run_max = -INFINITY;
    #pragma unroll 4
    for (int w = 0; w < csa_window; w++) {
        if (base + w >= N) break;
        run_max = fmaxf(run_max, compress_w[w]);
    }
    if (!isfinite(run_max)) return 0.0f;     // empty window

    float denom = 0.0f;
    float acc   = 0.0f;
    #pragma unroll 4
    for (int w = 0; w < csa_window; w++) {
        const int t = base + w;
        if (t >= N) break;
        const float e = ptx_expf(compress_w[w] - run_max);
        denom += e;
        acc   += e * static_cast<float>(x_seq[t * d_model + d]);
    }
    return (denom > 0.0f) ? (acc / denom) : 0.0f;
}

// =========================================================================
//  Forward (2b): Lightning indexer score (spec §2 CSA step 2)
//
//  Equation (spec §2):
//      I[t, s] = qI[t] · kI[s] / sqrt(rank)
//
//  where the low-rank query qI[t] = x[t] · W_DQ · W_UQ (rank INDEXER_RANK)
//  and kI[s] is the (compressed) indexer key for compressed entry s. The
//  host/kernel calls this per (query t, compressed key s) to build the
//  top-k selection set; only the dot-product + scaling is provided here.
//
//  `q_idx` and `k_idx` are pre-projected indexer vectors of length
//  `indexer_rank`. FP32 accumulation. Returns the scaled score.
// =========================================================================

__device__ __forceinline__ float sg2_csa_index_score(
    const float* __restrict__ q_idx,   // [indexer_rank] low-rank query  qI[t]
    const float* __restrict__ k_idx,   // [indexer_rank] compressed key  kI[s]
    const int indexer_rank
) {
    float dot = 0.0f;
    #pragma unroll
    for (int r = 0; r < indexer_rank; r++) {
        dot = ptx_fma(q_idx[r], k_idx[r], dot);
    }
    // Scale by 1/sqrt(rank) for variance control (spec §2: / sqrt(d)).
    return dot * fast_rsqrt_nr(static_cast<float>(indexer_rank));
}

// =========================================================================
//  Forward (3): Online-softmax attention step (FlashAttention-style)
//
//  Core reusable attention primitive shared by BOTH CSA (over selected
//  compressed ∪ window keys) and HCA (over all compressed ∪ window keys).
//  Implements one numerically-stable streaming-softmax update of
//  softmax(Q·Kᵀ / sqrt(head_dim)) · V (spec §2 CSA step 4 / HCA dense attn):
//
//      s      = (q · k) / sqrt(head_dim)
//      m_new  = max(m, s)
//      corr   = exp(m - m_new)                 (rescale prior partial state)
//      p      = exp(s - m_new)
//      l      = l·corr + p                      (running denominator)
//      acc[:] = acc[:]·corr + p·v[:]            (running value accumulator)
//
//  The running (m, l, acc) are carried in registers/smem across all keys for
//  one query; call sg2_softmax_finalize() once at the end. FP32 throughout.
//  Pass `scale` = 1/sqrt(head_dim) precomputed by the caller (constant across
//  keys, so we take it as an arg to avoid recomputing the rsqrt per key).
// =========================================================================

__device__ __forceinline__ void sg2_attention_score_and_accumulate(
    const float* __restrict__ q,     // [head_dim] query vector
    const float* __restrict__ k,     // [head_dim] key vector for one entry
    const float* __restrict__ v,     // [head_dim] value vector for one entry
    float* __restrict__ run_max,     // running max m  (in/out)
    float* __restrict__ run_denom,   // running denom l (in/out)
    float* __restrict__ acc,         // [head_dim] running accumulator (in/out)
    const float scale,               // = 1 / sqrt(head_dim)
    const int head_dim
) {
    // Logit s = (q·k) * scale.
    float dot = 0.0f;
    #pragma unroll
    for (int e = 0; e < head_dim; e++) {
        dot = ptx_fma(q[e], k[e], dot);
    }
    const float s = dot * scale;

    const float m_old = *run_max;
    const float m_new = fmaxf(m_old, s);
    // exp(m_old - m_new): 1.0 on the very first key (m_old = -INF -> corr = 0,
    // but acc/denom are 0 there too, so the math is consistent).
    const float corr  = ptx_expf(m_old - m_new);
    const float p     = ptx_expf(s - m_new);

    *run_denom = (*run_denom) * corr + p;
    #pragma unroll
    for (int e = 0; e < head_dim; e++) {
        acc[e] = acc[e] * corr + p * v[e];
    }
    *run_max = m_new;
}

// =========================================================================
//  Forward (3'): Softmax finalize — divide accumulator by denominator.
//
//  Completes the online softmax: out[:] = acc[:] / l  (spec §2). Call once
//  per query after all sg2_attention_score_and_accumulate() updates. Guards
//  against l == 0 (a query that saw no keys) by emitting zeros.
// =========================================================================

__device__ __forceinline__ void sg2_softmax_finalize(
    float* __restrict__ acc,         // [head_dim] running accumulator (in/out)
    const float run_denom,
    const int head_dim
) {
    const float inv = (run_denom > 0.0f) ? (1.0f / run_denom) : 0.0f;
    #pragma unroll
    for (int e = 0; e < head_dim; e++) {
        acc[e] *= inv;
    }
}

// =========================================================================
//  Forward (4): HCA — Heavily-compressed KV pooling (spec §2 HCA step 1)
//
//  Equation (spec §2):
//      c_kv[j, :] = (1/M) Σ_{w=0..M-1} x[j·M + w, :]                (mean pool)
//   or, if optional learned weights are supplied:
//      c_kv[j, :] = Σ_w softmax(hca_w)[w] · x[j·M + w, :]
//
//  with M = HCA_COMPRESS (=128, stride==window), single stream, NO indexer
//  (HCA attends densely to every compressed entry). Compressed length
//  Nh = ceil(N / M). Per-output-element form: feature channel `d` of
//  compressed entry `j`. Pass `hca_w = nullptr` for plain mean pooling.
//  Edge-safe: taps beyond N are skipped and the pool renormalized.
// =========================================================================

template <typename feat_t>
__device__ __forceinline__ float sg2_hca_compress_kv(
    const feat_t* __restrict__ x_seq,    // [N, d_model] sorted features
    const float*  __restrict__ hca_w,    // [hca_compress] weights, or nullptr (mean)
    const int j,                         // compressed-entry index (0..Nh-1)
    const int d,                         // feature channel (0..d_model-1)
    const int64_t N,
    const int d_model,
    const int hca_compress               // pooling stride/window M
) {
    const int base = j * hca_compress;

    if (hca_w == nullptr) {
        // Plain mean pool over valid taps.
        float acc = 0.0f;
        int   cnt = 0;
        for (int w = 0; w < hca_compress; w++) {
            const int t = base + w;
            if (t >= N) break;
            acc += static_cast<float>(x_seq[t * d_model + d]);
            cnt++;
        }
        return (cnt > 0) ? (acc / static_cast<float>(cnt)) : 0.0f;
    }

    // Learned weighted pool via numerically-stable softmax over valid taps.
    float run_max = -INFINITY;
    for (int w = 0; w < hca_compress; w++) {
        if (base + w >= N) break;
        run_max = fmaxf(run_max, hca_w[w]);
    }
    if (!isfinite(run_max)) return 0.0f;

    float denom = 0.0f, acc = 0.0f;
    for (int w = 0; w < hca_compress; w++) {
        const int t = base + w;
        if (t >= N) break;
        const float e = ptx_expf(hca_w[w] - run_max);
        denom += e;
        acc   += e * static_cast<float>(x_seq[t * d_model + d]);
    }
    return (denom > 0.0f) ? (acc / denom) : 0.0f;
}

// =========================================================================
//  Forward (3-5): GRU + apply tail (per-element)
//  Combines temporal memory, smart_grad, and Adam update.
//  PEER routing handles its own selection on the host side; this is
//  the per-element body that consumes the routed expert output.
// =========================================================================

template <typename ParamT, typename GradT>
__device__ __forceinline__ void sg2_apply_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ mu_state,    // GRU hidden state
    const GradT* __restrict__ grad,
    const float expert_out,          // PEER expert output for this element
    const float alpha,
    const float gru_decay,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int64_t idx
) {
    const float g = static_cast<float>(grad[idx]);
    const float p = static_cast<float>(param[idx]);

    // GRU step: simple gated update of mu_state with expert_out as candidate.
    const float mu_new = gru_decay * mu_state[idx] + (1.0f - gru_decay) * expert_out;
    mu_state[idx] = mu_new;

    const float smart_grad = g + alpha * mu_new;

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * smart_grad;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * smart_grad * smart_grad;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    // bc1, bc2 un-inverted (= 1 - beta^t): divide for bias correction.
    // Matches the convention used by adamw.h / grokadamw.h / etc. and the
    // Python `_single_param_step` in grokking_optimizers/optimizers/supergrok2.py.
    const float update = (m / bc1) / (sqrtf(v / bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

// =========================================================================
//  Backward (6): Bilevel precompute per-timestep (CSA/HCA adjoint inputs)
//
//  Reproduces the forward attention projections needed for the adjoint
//  (spec §3b / §7: the bilevel backward saves softmax denominators +
//  selected-index sets, and recomputes the q/k/v projections rather than the
//  Mamba in_proj/dt/B/C). For one feature row x[t, :] (length d_model) this
//  computes the per-head query/key/value projections used by both CSA and
//  HCA attention:
//
//      q[t, :] = x[t, :] · q_W      (q_W : [d_model, d_model])
//      k[t, :] = x[t, :] · k_W      (k_W : [d_model, d_model])
//      v[t, :] = x[t, :] · v_W      (v_W : [d_model, d_model])
//      qI[t,:] = (x[t, :] · idx_DQ) · idx_UQ   (low-rank indexer query, CSA;
//                pass idx_DQ/idx_UQ = nullptr for HCA which has no indexer)
//
//  Row-major weights, projected vectors written to pre_q/k/v_t [d_model] and
//  the indexer query to pre_qidx_t [indexer_rank] (skipped if idx_* null).
//  FP32 accumulation. Replaces the former Mamba in_proj/dt/B/C recompute.
// =========================================================================

__device__ __forceinline__ void sg2_bilevel_precompute_timestep(
    const float* __restrict__ x_row,      // [d_model] one sorted feature row x[t]
    const float* __restrict__ q_W,        // [d_model, d_model] query proj
    const float* __restrict__ k_W,        // [d_model, d_model] key proj
    const float* __restrict__ v_W,        // [d_model, d_model] value proj
    const float* __restrict__ idx_DQ,     // [d_model, indexer_rank] or nullptr (HCA)
    const float* __restrict__ idx_UQ,     // [indexer_rank, d_model] or nullptr (HCA)
    float* __restrict__ pre_q_t,          // [d_model] out
    float* __restrict__ pre_k_t,          // [d_model] out
    float* __restrict__ pre_v_t,          // [d_model] out
    float* __restrict__ pre_qidx_t,       // [indexer_rank] out, or nullptr (HCA)
    const int d_model,
    const int indexer_rank
) {
    // q/k/v projections: out[o] = Σ_d x[d] · W[d, o]  (row-major [d_model,d_model]).
    #pragma unroll 4
    for (int o = 0; o < d_model; o++) {
        float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) {
            const float xd = x_row[d];
            const int   wi = d * d_model + o;
            q_val = ptx_fma(q_W[wi], xd, q_val);
            k_val = ptx_fma(k_W[wi], xd, k_val);
            v_val = ptx_fma(v_W[wi], xd, v_val);
        }
        pre_q_t[o] = q_val;
        pre_k_t[o] = k_val;
        pre_v_t[o] = v_val;
    }

    // Low-rank lightning-indexer query qI = (x · idx_DQ) · idx_UQ  (CSA only).
    if (idx_DQ != nullptr && idx_UQ != nullptr && pre_qidx_t != nullptr) {
        float lr[SG2_INDEXER_RANK_MAX];
        #pragma unroll
        for (int r = 0; r < indexer_rank; r++) {
            float acc = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < d_model; d++) {
                acc = ptx_fma(idx_DQ[d * indexer_rank + r], x_row[d], acc);
            }
            lr[r] = acc;
        }
        // qI is rank-INDEXER_RANK; spec §2 keeps qI in the low-rank space for
        // scoring against compressed indexer keys, so emit the rank-dim vector.
        #pragma unroll
        for (int r = 0; r < indexer_rank; r++) {
            pre_qidx_t[r] = lr[r];
        }
        (void)idx_UQ;  // UQ lift-back is applied where qI meets full-dim keys;
                       // the indexer score path uses the low-rank form directly.
    }
}

// ═════════════════════════════════════════════════════════════════════════
//  MoE/Adam multi-tensor — folded in from former csrc/algorithms/moe_adam.h.
//
//  Multi-tensor batched AdamW used for both standard parameter groups and
//  Mixture-of-Experts active-set updates. The MoE variant compacts the
//  active subset of expert parameters into a dense buffer, runs the same
//  per-element Adam step over that buffer, then scatters results back.
//
//  The per-element math is identical to adamw.h::adamw_step; this function
//  re-exports it under the `moe_adam_step` name to keep the launcher glue
//  symmetric across the optimizers.
// ═════════════════════════════════════════════════════════════════════════

template <typename ParamT, typename GradT>
__device__ __forceinline__ void moe_adam_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const GradT* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int64_t idx
) {
    adamw_step(param, exp_avg, exp_avg_sq, grad,
               lr, beta1, beta2, eps, wd, bc1, bc2, idx);
}

}} // namespace sg::algorithms
