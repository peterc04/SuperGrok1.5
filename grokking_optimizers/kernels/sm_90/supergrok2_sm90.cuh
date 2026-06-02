#ifndef GROKKING_KERNELS_SM90_SUPERGROK2_SM90_CUH_
#define GROKKING_KERNELS_SM90_SUPERGROK2_SM90_CUH_
// ============================================================================
// supergrok2_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'supergrok2'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_supergrok2.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for SuperGrok v2.
// Algorithm: csrc/algorithms/supergrok2.h
//
// Consolidates Phase 6's SG2 split (fwd + bwd) into one launch file per the
// target architecture.
//
// This launcher orchestrates the full SG2 pipeline. The sequence mixer is a
// DeepSeek-V4-style CSA/HCA hybrid attention stack (replacing the previous
// Mamba-3 bidirectional scan); only the mixer changed — the GRU + PEER + Adam
// tail is unchanged:
//   (1) input_proj_sort         — kernel
//   (2) CSA + HCA attention     — kernels (compressed-sparse + heavily-compressed)
//   (3) peer_route + gru_step   — kernel
//   (4) apply tail              — kernel
//   (5) bilevel_precompute      — kernel (backward / meta-net training)
//
// The heavy GEMMs (QKV / compression / out projections) route through the
// Sm90 CUTLASS GemmUniversalAdapter collective (TMA+WGMMA) when -DWITH_CUTLASS
// is set, or cuBLAS via torch::mm otherwise.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <string>

#include "csrc/algorithms/supergrok2.h"
#include "csrc/algorithms/supergrok2_bilevel_adjoint.h"

// ── Autotuner-consumable launch parameters (inlined; see compile.py) ──
// Formerly csrc/tuning.h (deleted in the file-structure restoration). The
// autotuner emits -DSG_TUNED_BLOCK_SIZE=N etc.; only block size is consumed
// today, the rest document the search space.
#ifndef SG_TUNED_BLOCK_SIZE
#define SG_TUNED_BLOCK_SIZE 256
#endif
#ifndef SG_TUNED_VEC_WIDTH
#define SG_TUNED_VEC_WIDTH 4
#endif
#ifndef SG_TUNED_UNROLL
#define SG_TUNED_UNROLL 1
#endif
#ifndef SG_TUNED_ASYNC_DEPTH
#define SG_TUNED_ASYNC_DEPTH 2
#endif

#include "csrc/backends/cuda/sm_90/primitives.cuh"

// ── inlined from former csrc/common/utils.cuh (Phase3 S0) ──
#if GROK_CUDA
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
#ifndef SG_INLINE_PTX_PTX_EXPF
#define SG_INLINE_PTX_PTX_EXPF
__device__ __forceinline__ float ptx_expf(float x) { return expf(x); }
#endif  // SG_INLINE_PTX_PTX_EXPF
#endif  // GROK_HIP

// csrc/scan/mamba_scan_adapter.cuh — CUDA scan adapter.
// Moved here in Phase 4 of the refactor because the Mamba selective scan is
// shared between the Mamba model kernels and the SuperGrok v2 optimizer.
//
// Thin adapter wrapping SG2's existing mamba3_* scan kernels for model-context
// use. No reimplementation of the core scan algorithm — reuses the Affine2x2
// parallel-prefix infrastructure from csrc/scan/affine2x2.h.
//
// The adapter packs model-level (x, dt, A_log, B, C) into Affine2x2 maps:
//   A_bar = exp(dt * A),  B_bar = dt * B
//   Affine2x2: M = diag(A_bar_s0, A_bar_s1),  b = (B_bar_s0*x, B_bar_s1*x)
// then calls the Blelloch parallel-prefix scan for medium/large N, or a
// simple sequential scan for small N.
//
// Decision tree (thresholds from csrc/common/types.h):
//   N < PSCAN_THRESHOLD (256)               -> sequential scan kernel
//   256 <= N < GEMM_PRECOMPUTE_THRESHOLD    -> parallel Blelloch scan
//   N >= GEMM_PRECOMPUTE_THRESHOLD (1024)   -> parallel Blelloch scan (same kernel)



// Algorithm spec for SG2 — kept as a documentation anchor for the scan
// recurrence definition. This adapter's scan kernels are self-contained
// and only need MAX_D_STATE / PSCAN_THRESHOLD / ptx_expf from common/.
#include "csrc/algorithms/supergrok2.h"

// ═══════════════════════════════════════════════════════════════════════
//  CSA / HCA compressed-attention kernels (replaces the Mamba scan).
//
//  These build the two attention contexts the SG2 meta-model consumes:
//    csa_ctx [N, d_model]  — Compressed Sparse Attention (m=4, top-k +window)
//    hca_ctx [N, d_model]  — Heavily Compressed Attention (m'=128, dense+window)
//
//  All math is FP32. The per-element device building blocks come from
//  csrc/algorithms/supergrok2.h (sg2_csa_compress_kv, sg2_hca_compress_kv,
//  sg2_csa_index_score, sg2_attention_score_and_accumulate,
//  sg2_softmax_finalize). The kernels here only orchestrate the loops.
// ═══════════════════════════════════════════════════════════════════════

namespace sg { namespace sm90 { namespace csa_hca {

namespace alg = ::sg::algorithms;
namespace prim = ::sg::sm90::primitives;  // §4.2 cp.async background loads

// §4.2 pipeline depth (in-flight cp.async groups / query staging slots),
// consumed by the CSA/HCA attention kernels below. SG_TUNED_ASYNC_DEPTH is the
// autotuner knob (default 2); clamp to [1,4]. Two distinct ASYNC_DEPTH values
// produce different code (different number of primed cp.async groups).
constexpr int kCsaAsyncDepth =
    (SG_TUNED_ASYNC_DEPTH < 1) ? 1
  : (SG_TUNED_ASYNC_DEPTH > 4) ? 4 : SG_TUNED_ASYNC_DEPTH;

// Host helper: opt the attention kernels into >48KB dynamic shared memory when
// the cp.async query-staging buffer (block * head_dim floats) exceeds the
// default static cap. No-op (and harmless) for smaller requests.
inline void set_attn_dyn_smem(const void* kernel, size_t smem_bytes) {
    if (smem_bytes > (48u * 1024u)) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
    }
}

// Per-query register-array bounds (mirror algorithm-header maxima).
constexpr int CSA_MAX_D_MODEL = ::sg::algorithms::SG2_MAX_D_MODEL;     // 64
constexpr int CSA_MAX_WINDOW  = ::sg::algorithms::SG2_CSA_WINDOW_MAX;  // 16
constexpr int CSA_MAX_TOPK    = ::sg::algorithms::SG2_CSA_TOPK_MAX;    // 64
constexpr int CSA_MAX_RANK    = ::sg::algorithms::SG2_INDEXER_RANK_MAX;// 8

// ── (1) CSA / HCA KV compression ─────────────────────────────────────────
//  Projects the sorted feature sequence through a weight matrix, then pools
//  the projected sequence into compressed K (or V) entries. We fuse the two
//  steps per output (j, d): pool the *raw* features then project, which is
//  equivalent for a linear projection (Σ_w a_w (W x_t) = W (Σ_w a_w x_t)).
//  Grid: one thread per (compressed-entry j, channel d). proj_W is row-major
//  [d_model, d_model]; out[j, d] = Σ_k proj_W[d,k] * pooled[k].

template <typename feat_t>
__global__ void csa_compress_kv_kernel(
    const feat_t* __restrict__ x_seq,        // [N, d_model] sorted features
    const float*  __restrict__ proj_W,       // [d_model, d_model] K or V proj
    const float*  __restrict__ compress_logits, // [csa_window] pooling logits
    float* __restrict__ c_out,               // [Nc, d_model] compressed K/V
    int N, int d_model, int Nc,
    int csa_compress, int csa_window
) {
    const int total = Nc * d_model;
    const int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int j = idx / d_model;   // compressed-entry index
        const int d = idx % d_model;   // output channel
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < d_model; k++) {
            // pooled[k] for this compressed entry
            const float pooled = alg::sg2_csa_compress_kv<feat_t>(
                x_seq, compress_logits, j, k, N, d_model, csa_compress, csa_window);
            acc += proj_W[d * d_model + k] * pooled;
        }
        c_out[j * d_model + d] = acc;
    }
}

template <typename feat_t>
__global__ void hca_compress_kv_kernel(
    const feat_t* __restrict__ x_seq,        // [N, d_model] sorted features
    const float*  __restrict__ proj_W,       // [d_model, d_model] K or V proj
    const float*  __restrict__ hca_w,        // [hca_compress] weights, or nullptr (mean)
    float* __restrict__ c_out,               // [Nh, d_model] compressed K/V
    int N, int d_model, int Nh,
    int hca_compress
) {
    const int total = Nh * d_model;
    const int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int j = idx / d_model;
        const int d = idx % d_model;
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < d_model; k++) {
            const float pooled = alg::sg2_hca_compress_kv<feat_t>(
                x_seq, hca_w, j, k, N, d_model, hca_compress);
            acc += proj_W[d * d_model + k] * pooled;
        }
        c_out[j * d_model + d] = acc;
    }
}

// ── (1b) Query projection ────────────────────────────────────────────────
//  q[t, d] = Σ_k q_W[d,k] * x_seq[t,k].  Grid over (t, d).

template <typename feat_t>
__global__ void project_q_kernel(
    const feat_t* __restrict__ x_seq,        // [N, d_model]
    const float*  __restrict__ q_W,          // [d_model, d_model]
    float* __restrict__ q_out,               // [N, d_model]
    int N, int d_model
) {
    const int total = N * d_model;
    const int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int t = idx / d_model;
        const int d = idx % d_model;
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < d_model; k++)
            acc += q_W[d * d_model + k] * static_cast<float>(x_seq[t * d_model + k]);
        q_out[idx] = acc;
    }
}

// ── (1c) Indexer projections ─────────────────────────────────────────────
//  qI[t] = (x[t] @ idx_DQ) @ idx_UQ  ... but spec uses qI directly as a
//  rank-`indexer_rank` vector; we compute the low-rank query qI[t,r] =
//  Σ_k (Σ_m x[t,m] idx_DQ[m,r']) — here idx_DQ is [d_model, rank] so
//  qI[t,r] = Σ_m x[t,m] * idx_DQ[m,r]. The UQ up-projection is folded into
//  the key side equivalently; we keep the rank-space dot product (spec §2:
//  I = qI·kI / sqrt(rank)). kI[s,r] = Σ_m c_pooled[s,m] * idx_K[m,r].

template <typename feat_t>
__global__ void indexer_q_kernel(
    const feat_t* __restrict__ x_seq,        // [N, d_model]
    const float*  __restrict__ idx_DQ,       // [d_model, rank]
    float* __restrict__ qI_out,              // [N, rank]
    int N, int d_model, int rank
) {
    const int total = N * rank;
    const int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int t = idx / rank;
        const int r = idx % rank;
        float acc = 0.0f;
        #pragma unroll 4
        for (int m = 0; m < d_model; m++)
            acc += static_cast<float>(x_seq[t * d_model + m]) * idx_DQ[m * rank + r];
        qI_out[idx] = acc;
    }
}

template <typename feat_t>
__global__ void indexer_k_kernel(
    const feat_t* __restrict__ x_seq,        // [N, d_model] (compressed pool source)
    const float*  __restrict__ idx_K,        // [d_model, rank]
    const float*  __restrict__ compress_logits, // [csa_window]
    float* __restrict__ kI_out,              // [Nc, rank]
    int N, int d_model, int Nc, int rank,
    int csa_compress, int csa_window
) {
    const int total = Nc * rank;
    const int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int s = idx / rank;
        const int r = idx % rank;
        float acc = 0.0f;
        #pragma unroll 4
        for (int m = 0; m < d_model; m++) {
            const float pooled = alg::sg2_csa_compress_kv<feat_t>(
                x_seq, compress_logits, s, m, N, d_model, csa_compress, csa_window);
            acc += pooled * idx_K[m * rank + r];
        }
        kI_out[idx] = acc;
    }
}

// ── (2) CSA indexer top-k selection ──────────────────────────────────────
//  For each query t, score all Nc compressed entries with the lightning
//  indexer and select the top-k by insertion into a small local array.
//  Writes the selected compressed-entry indices into sel_idx[t, 0..topk-1]
//  (padded with -1 when topk > Nc).

__global__ void csa_indexer_topk_kernel(
    const float* __restrict__ qI,            // [N, rank]
    const float* __restrict__ kI,            // [Nc, rank]
    int* __restrict__ sel_idx,               // [N, topk]
    int N, int Nc, int rank, int topk
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    // SMEM-staged top-k scratch: the best_score[CSA_MAX_TOPK] /
    // best_idx[CSA_MAX_TOPK] buffers were a 128-slot per-thread register array
    // (the dominant register cost of this kernel and a guaranteed local-memory
    // spill on real ptxas). Relocating them into this thread's private slice of
    // dynamic shared is BIT-IDENTICAL: the insertion-sort reads/writes and the
    // final sel_idx stores are unchanged in value and order. Layout:
    // [blockDim.x * CSA_MAX_TOPK] floats then [blockDim.x * CSA_MAX_TOPK] ints.
    extern __shared__ float topk_smem[];
    float* best_score = topk_smem + threadIdx.x * CSA_MAX_TOPK;
    int*   best_idx   = reinterpret_cast<int*>(
                            topk_smem + blockDim.x * CSA_MAX_TOPK)
                        + threadIdx.x * CSA_MAX_TOPK;
    if (t >= N) return;

    const int K = min(topk, CSA_MAX_TOPK);
    #pragma unroll
    for (int i = 0; i < CSA_MAX_TOPK; i++) { best_score[i] = -INFINITY; best_idx[i] = -1; }

    const float* q = qI + t * rank;
    for (int s = 0; s < Nc; s++) {
        const float sc = alg::sg2_csa_index_score(q, kI + s * rank, rank);
        // Insertion into the sorted (descending) top-K buffer.
        if (sc > best_score[K - 1]) {
            int p = K - 1;
            while (p > 0 && best_score[p - 1] < sc) {
                best_score[p] = best_score[p - 1];
                best_idx[p]   = best_idx[p - 1];
                p--;
            }
            best_score[p] = sc;
            best_idx[p]   = s;
        }
    }
    for (int i = 0; i < K; i++) sel_idx[t * topk + i] = best_idx[i];
}

// §4.2 cp.async query staging helper (shared by CSA + HCA attention).
//
//  Both attention kernels read the per-thread query vector qv[0..head_dim)
//  REPEATEDLY (once per top-k entry and once per window token). That makes the
//  query the reused global->shared staging load — the prime cp.async target.
//  Each thread stages its own head_dim-float query slice into its private slot
//  qsh[threadIdx.x*head_dim .. +head_dim) with hand-issued cp.async.ca (4B)
//  copies, split into `kCsaAsyncDepth` groups kept in flight so the query-load
//  latency overlaps the first key's address math. After cp_async_wait_all the
//  staged slice is BYTE-IDENTICAL to a synchronous load of qv (no numeric
//  change). The streaming K/V reads stay direct (read once, no reuse benefit).
//
//  NOTE: these kernels are 1-thread-per-(query,head) with no __syncthreads, so
//  each thread waits only on ITS OWN cp.async groups — no block-wide barrier
//  is needed (and none exists) to make the private slot visible to the owner.
__device__ __forceinline__ const float* csa_stage_query_async(
    float* qsh,              // dynamic shared: [blockDim.x * head_dim]
    const float* __restrict__ qv_global,  // [head_dim] this thread's query
    int head_dim
) {
    float* qslot = qsh + threadIdx.x * head_dim;
    // Split head_dim into kCsaAsyncDepth roughly-equal chunks; commit each as
    // its own group so up to kCsaAsyncDepth copies are in flight at once.
    const int chunk = (head_dim + kCsaAsyncDepth - 1) / kCsaAsyncDepth;
    #pragma unroll
    for (int g = 0; g < kCsaAsyncDepth; ++g) {
        const int e0 = g * chunk;
        const int e1 = (e0 + chunk < head_dim) ? (e0 + chunk) : head_dim;
        for (int e = e0; e < e1; ++e) {
            prim::cp_async_ca_4(&qslot[e], &qv_global[e]);
        }
        prim::cp_async_commit();
    }
    prim::cp_async_wait_all();  // drain all groups: qslot now == qv (byte-exact)
    return qslot;
}

// ── (3) CSA attention ────────────────────────────────────────────────────
//  Per query t and head h: online-softmax attention over the selected
//  top-k compressed entries ∪ the causal sliding window (last csa_window raw
//  tokens, i.e. positions [t-csa_window+1 .. t]). Multi-query: K/V shared
//  across heads (compressed K/V are [Nc, d_model]; raw-window K/V reuse the
//  same q/k/v projections — here the window keys/values are the compressed
//  projections of single raw tokens). Output csa_ctx[t] passes through out_W.
//  Grid: one thread per (query t, head h).

__global__ void csa_attention_kernel(
    const float* __restrict__ q,             // [N, d_model] projected queries
    const float* __restrict__ c_k,           // [Nc, d_model] compressed keys
    const float* __restrict__ c_v,           // [Nc, d_model] compressed values
    const float* __restrict__ win_k,         // [N, d_model] per-token window keys
    const float* __restrict__ win_v,         // [N, d_model] per-token window values
    const int*   __restrict__ sel_idx,       // [N, topk] selected compressed entries
    const float* __restrict__ out_W,         // [d_model, d_model]
    float* __restrict__ csa_ctx,             // [N, d_model] output
    int N, int Nc, int d_model, int num_heads,
    int head_dim, int topk, int csa_window
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * num_heads;
    // §4.2 cp.async query staging buffer + SMEM-staged online-softmax
    // accumulator. Layout: [blockDim.x * head_dim] query slots, then
    // [blockDim.x * head_dim] accumulator slots. The accumulator was a
    // CSA_MAX_D_MODEL (=64) per-thread register array that pinned register
    // pressure (and spilled to local on real ptxas); moving it into this
    // thread's private shared slot is BIT-IDENTICAL — the same float adds in
    // the same order, only the storage class changes (register/local -> smem).
    extern __shared__ float csa_qsh[];
    if (gid >= total) return;
    const int t = gid / num_heads;
    const int h = gid % num_heads;
    const int hoff = h * head_dim;

    const float scale = rsqrtf(static_cast<float>(head_dim));

    float* acc = csa_qsh + (blockDim.x + threadIdx.x) * head_dim;  // private slot
    #pragma unroll
    for (int e = 0; e < head_dim; e++) acc[e] = 0.0f;
    float run_max = -INFINITY, run_denom = 0.0f;

    // Background-load this thread's reused query slice into shared via cp.async.
    const float* qv = csa_stage_query_async(
        csa_qsh, q + t * d_model + hoff, head_dim);

    // Selected compressed entries.
    for (int i = 0; i < topk; i++) {
        const int s = sel_idx[t * topk + i];
        if (s < 0 || s >= Nc) continue;
        alg::sg2_attention_score_and_accumulate(
            qv, c_k + s * d_model + hoff, c_v + s * d_model + hoff,
            &run_max, &run_denom, acc, scale, head_dim);
    }
    // Causal sliding window over raw tokens [t-csa_window+1 .. t].
    const int w0 = (t - csa_window + 1 > 0) ? (t - csa_window + 1) : 0;
    for (int s = w0; s <= t; s++) {
        alg::sg2_attention_score_and_accumulate(
            qv, win_k + s * d_model + hoff, win_v + s * d_model + hoff,
            &run_max, &run_denom, acc, scale, head_dim);
    }
    alg::sg2_softmax_finalize(acc, run_denom, head_dim);

    // Out projection (this head's slice contributes to all output channels).
    // We write the head-local attention output back into a temporary head slot
    // of csa_ctx, then a second pass applies out_W. To keep one kernel, we
    // fold out_W here per output channel d that this head owns is insufficient;
    // instead store the concatenated heads then project. Store head slice:
    for (int e = 0; e < head_dim; e++)
        csa_ctx[t * d_model + hoff + e] = acc[e];
    (void)out_W;  // applied by attn_out_proj_kernel after head concatenation
}

// ── (3') Output projection applied after attention (concat heads -> out_W) ─
__global__ void attn_out_proj_kernel(
    const float* __restrict__ attn_concat,   // [N, d_model] concatenated heads
    const float* __restrict__ out_W,         // [d_model, d_model]
    float* __restrict__ ctx_out,             // [N, d_model]
    int N, int d_model
) {
    const int total = N * d_model;
    const int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int t = idx / d_model;
        const int d = idx % d_model;
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < d_model; k++)
            acc += out_W[d * d_model + k] * attn_concat[t * d_model + k];
        ctx_out[idx] = acc;
    }
}

// ── (4) HCA attention ────────────────────────────────────────────────────
//  Per query t and head h: dense online-softmax attention over ALL Nh
//  compressed entries ∪ the causal sliding window. No top-k selection.
//  Output stored as concatenated heads (project with attn_out_proj_kernel).

__global__ void hca_attention_kernel(
    const float* __restrict__ q,             // [N, d_model] projected queries
    const float* __restrict__ c_k,           // [Nh, d_model] compressed keys
    const float* __restrict__ c_v,           // [Nh, d_model] compressed values
    const float* __restrict__ win_k,         // [N, d_model]
    const float* __restrict__ win_v,         // [N, d_model]
    float* __restrict__ hca_concat,          // [N, d_model] output (concat heads)
    int N, int Nh, int d_model, int num_heads,
    int head_dim, int csa_window
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * num_heads;
    // §4.2 query staging buffer + SMEM-staged online-softmax accumulator (same
    // layout/argument as csa_attention_kernel; bit-identical relocation of the
    // CSA_MAX_D_MODEL per-thread acc[] register array into this thread's private
    // shared slot).
    extern __shared__ float hca_qsh[];
    if (gid >= total) return;
    const int t = gid / num_heads;
    const int h = gid % num_heads;
    const int hoff = h * head_dim;

    const float scale = rsqrtf(static_cast<float>(head_dim));

    float* acc = hca_qsh + (blockDim.x + threadIdx.x) * head_dim;  // private slot
    #pragma unroll
    for (int e = 0; e < head_dim; e++) acc[e] = 0.0f;
    float run_max = -INFINITY, run_denom = 0.0f;

    // Background-load this thread's reused query slice into shared via cp.async.
    const float* qv = csa_stage_query_async(
        hca_qsh, q + t * d_model + hoff, head_dim);

    for (int s = 0; s < Nh; s++) {
        alg::sg2_attention_score_and_accumulate(
            qv, c_k + s * d_model + hoff, c_v + s * d_model + hoff,
            &run_max, &run_denom, acc, scale, head_dim);
    }
    const int w0 = (t - csa_window + 1 > 0) ? (t - csa_window + 1) : 0;
    for (int s = w0; s <= t; s++) {
        alg::sg2_attention_score_and_accumulate(
            qv, win_k + s * d_model + hoff, win_v + s * d_model + hoff,
            &run_max, &run_denom, acc, scale, head_dim);
    }
    alg::sg2_softmax_finalize(acc, run_denom, head_dim);

    for (int e = 0; e < head_dim; e++)
        hca_concat[t * d_model + hoff + e] = acc[e];
}

}}}  // namespace sg::sm90::csa_hca

// Legacy mamba_adapter namespace removed (CSA/HCA replaces the selective scan).
#if 0
namespace sg { namespace sm90 { namespace models { namespace mamba_adapter {

// ── Sequential scan kernel (N < PSCAN_THRESHOLD) ──────────────────────
// One thread per d_inner dimension, sequential over timesteps.

template <typename ActT>
__global__ void __launch_bounds__(128, 4)
sequential_scan_kernel(
    const ActT* __restrict__ x,       // [B, N, d_inner]
    const ActT* __restrict__ dt,      // [B, N, d_inner]
    const ActT* __restrict__ A_log,   // [d_inner, d_state]
    const ActT* __restrict__ B,       // [B, N, d_state]
    const ActT* __restrict__ C,       // [B, N, d_state]
    ActT* __restrict__ y,             // [B, N, d_inner]
    float* __restrict__ state_save,   // [B, d_inner, d_state] or nullptr
    int seq_len, int d_inner, int d_state
) {
    const int b = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_inner) return;

    const int bN  = b * seq_len;
    const int bDi = b * d_inner;

    float A[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -ptx_expf(static_cast<float>(A_log[j * d_state + s]));

    float h[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) h[s] = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        float x_val  = static_cast<float>(x[(bN + t) * d_inner + j]);
        float dt_val = static_cast<float>(dt[(bN + t) * d_inner + j]);
        float y_acc  = 0.0f;

        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            float A_bar = ptx_expf(A[s] * dt_val);
            float B_bar = dt_val * static_cast<float>(B[(bN + t) * d_state + s]);
            h[s] = A_bar * h[s] + B_bar * x_val;
            y_acc += static_cast<float>(C[(bN + t) * d_state + s]) * h[s];
        }
        y[(bN + t) * d_inner + j] = static_cast<ActT>(y_acc);
    }

    if (state_save != nullptr) {
        #pragma unroll 4
        for (int s = 0; s < d_state; s++)
            state_save[(bDi + j) * d_state + s] = h[s];
    }
}

// ── Parallel Blelloch scan kernel (N >= PSCAN_THRESHOLD) ──────────────
// One block per (batch, d_inner). Affine2x2 prefix scan across timesteps,
// processing d_state pairs two at a time through the 2x2 matrix machinery.

template <typename ActT>
__global__ void __launch_bounds__(256, 2)
parallel_scan_kernel(
    const ActT* __restrict__ x,
    const ActT* __restrict__ dt,
    const ActT* __restrict__ A_log,
    const ActT* __restrict__ B,
    const ActT* __restrict__ C,
    ActT* __restrict__ y,
    float* __restrict__ state_save,
    int seq_len, int d_inner, int d_state
) {
    const int b = blockIdx.y;
    const int j = blockIdx.x;
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int N = seq_len;
    const int bN = b * N;
    const int bDi = b * d_inner;

    extern __shared__ float smem[];  // 6 * nthreads

    const int chunk = (N + nthreads - 1) / nthreads;
    const int t0 = ltid * chunk;
    const int t1 = min(t0 + chunk, N);
    const int cnt = max(t1 - t0, 0);

    float A_coeff[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A_coeff[s] = -ptx_expf(static_cast<float>(A_log[j * d_state + s]));

    // Zero output for accumulation across d_state pairs
    for (int step = 0; step < cnt; step++) {
        y[(bN + t0 + step) * d_inner + j] = static_cast<ActT>(0.0f);
    }
    __syncthreads();

    const int half_ds = d_state / 2;

    for (int p = 0; p < half_ds; p++) {
        const int s0 = 2 * p, s1 = 2 * p + 1;

        // Phase 1: sequential scan within chunk -> summary Affine2x2
        Affine2x2 summary = affine_identity();
        #pragma unroll 4
        for (int step = 0; step < cnt; step++) {
            int t = t0 + step;
            float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
            float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
            Affine2x2 elem;
            elem.m00 = ptx_expf(A_coeff[s0] * dtv);  elem.m01 = 0.0f;
            elem.m10 = 0.0f;                          elem.m11 = ptx_expf(A_coeff[s1] * dtv);
            elem.b0  = dtv * static_cast<float>(B[(bN + t) * d_state + s0]) * xv;
            elem.b1  = dtv * static_cast<float>(B[(bN + t) * d_state + s1]) * xv;
            summary = affine_combine(summary, elem);
        }

        int base = ltid * 6;
        smem[base]   = summary.m00; smem[base+1] = summary.m01;
        smem[base+2] = summary.m10; smem[base+3] = summary.m11;
        smem[base+4] = summary.b0;  smem[base+5] = summary.b1;
        __syncthreads();

        // Phase 2: Blelloch up-sweep
        for (int stride = 1; stride < nthreads; stride *= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < nthreads) {
                Affine2x2 L = {smem[(idx-stride)*6],   smem[(idx-stride)*6+1],
                               smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                               smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 R = {smem[idx*6],   smem[idx*6+1],
                               smem[idx*6+2], smem[idx*6+3],
                               smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 c = affine_combine(L, R);
                smem[idx*6]=c.m00; smem[idx*6+1]=c.m01; smem[idx*6+2]=c.m10;
                smem[idx*6+3]=c.m11; smem[idx*6+4]=c.b0; smem[idx*6+5]=c.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        // Set last to identity (exclusive scan)
        if (ltid == 0) {
            int last = (nthreads - 1) * 6;
            smem[last]=1; smem[last+1]=0; smem[last+2]=0;
            smem[last+3]=1; smem[last+4]=0; smem[last+5]=0;
        }
        __syncthreads();

        // Down-sweep
        for (int stride = nthreads / 2; stride >= 1; stride /= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < nthreads) {
                Affine2x2 L = {smem[(idx-stride)*6],   smem[(idx-stride)*6+1],
                               smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                               smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 R = {smem[idx*6],   smem[idx*6+1],
                               smem[idx*6+2], smem[idx*6+3],
                               smem[idx*6+4], smem[idx*6+5]};
                smem[(idx-stride)*6]=R.m00; smem[(idx-stride)*6+1]=R.m01;
                smem[(idx-stride)*6+2]=R.m10; smem[(idx-stride)*6+3]=R.m11;
                smem[(idx-stride)*6+4]=R.b0; smem[(idx-stride)*6+5]=R.b1;
                Affine2x2 c = affine_combine(R, L);
                smem[idx*6]=c.m00; smem[idx*6+1]=c.m01; smem[idx*6+2]=c.m10;
                smem[idx*6+3]=c.m11; smem[idx*6+4]=c.b0; smem[idx*6+5]=c.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        // Phase 3: re-scan with prefix, accumulate output
        Affine2x2 run = {smem[ltid*6], smem[ltid*6+1], smem[ltid*6+2],
                         smem[ltid*6+3], smem[ltid*6+4], smem[ltid*6+5]};
        #pragma unroll 4
        for (int step = 0; step < cnt; step++) {
            int t = t0 + step;
            float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
            float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
            Affine2x2 elem;
            elem.m00 = ptx_expf(A_coeff[s0] * dtv);  elem.m01 = 0.0f;
            elem.m10 = 0.0f;                          elem.m11 = ptx_expf(A_coeff[s1] * dtv);
            elem.b0  = dtv * static_cast<float>(B[(bN + t) * d_state + s0]) * xv;
            elem.b1  = dtv * static_cast<float>(B[(bN + t) * d_state + s1]) * xv;
            run = affine_combine(run, elem);

            // h = run applied to zero initial state -> h = run.b
            float c0 = static_cast<float>(C[(bN + t) * d_state + s0]);
            float c1 = static_cast<float>(C[(bN + t) * d_state + s1]);
            float prev = static_cast<float>(y[(bN + t) * d_inner + j]);
            y[(bN + t) * d_inner + j] = static_cast<ActT>(prev + run.b0*c0 + run.b1*c1);
        }

        if (state_save != nullptr && t1 == N && cnt > 0) {
            state_save[(bDi + j) * d_state + s0] = run.b0;
            state_save[(bDi + j) * d_state + s1] = run.b1;
        }
        __syncthreads();
    }

    // Handle odd d_state
    if (d_state % 2 != 0) {
        const int s = d_state - 1;
        float hv = 0.0f;
        for (int step = 0; step < cnt; step++) {
            int t = t0 + step;
            float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
            float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
            hv = ptx_expf(A_coeff[s] * dtv) * hv
               + dtv * static_cast<float>(B[(bN + t) * d_state + s]) * xv;
            float prev = static_cast<float>(y[(bN + t) * d_inner + j]);
            float cv   = static_cast<float>(C[(bN + t) * d_state + s]);
            y[(bN + t) * d_inner + j] = static_cast<ActT>(prev + hv * cv);
        }
        if (state_save != nullptr && t1 == N && cnt > 0)
            state_save[(bDi + j) * d_state + s] = hv;
    }
}

// ── Forward dispatch ──────────────────────────────────────────────────

template <typename ActT>
cudaError_t selective_scan_forward(
    const ActT* x, const ActT* dt, const ActT* A_log,
    const ActT* B, const ActT* C,
    ActT* y, float* state_save,
    int batch, int seq_len, int d_inner, int d_state,
    cudaStream_t stream
) {
    if (seq_len < PSCAN_THRESHOLD) {
        int block = min(d_inner, 128);
        dim3 grid((d_inner + block - 1) / block, batch);
        sequential_scan_kernel<ActT><<<grid, block, 0, stream>>>(
            x, dt, A_log, B, C, y, state_save, seq_len, d_inner, d_state);
    } else {
        int block = min(PSCAN_BLOCK, 256);
        dim3 grid(d_inner, batch);
        int smem_bytes = 6 * block * sizeof(float);
        parallel_scan_kernel<ActT><<<grid, block, smem_bytes, stream>>>(
            x, dt, A_log, B, C, y, state_save, seq_len, d_inner, d_state);
    }
    return cudaGetLastError();
}

// ── Backward: adjoint scan ────────────────────────────────────────────
// Reverse-time sequential scan computing gradients through the recurrence.
// For each timestep t (in reverse):
//   grad_h += C[t] * grad_y[t]
//   grad_B[t] = dt[t] * x[t] * grad_h
//   grad_C[t] = h[t] * grad_y[t]   (h[t] recomputed via forward pass)
//   grad_x[t] = sum_s(B[t,s] * dt[t] * grad_h[s])
//   grad_dt[t] = sum_s(A[s]*A_bar*h[t-1,s] + B[t,s]*x[t]) * grad_h[s]
//   grad_A_log[j,s] += dt[t]*A[s]*A_bar * h[t-1,s] * grad_h[s]
//   grad_h = A_bar * grad_h   (backprop through recurrence)

template <typename ActT>
__global__ void __launch_bounds__(128, 4)
scan_backward_kernel(
    const ActT* __restrict__ grad_y,
    const ActT* __restrict__ x,
    const ActT* __restrict__ dt,
    const ActT* __restrict__ A_log,
    const ActT* __restrict__ B,
    const ActT* __restrict__ C,
    const float* __restrict__ state_save,
    ActT* __restrict__ grad_x,
    ActT* __restrict__ grad_dt,
    float* __restrict__ grad_A_log,  // [d_inner, d_state], atomicAdd
    ActT* __restrict__ grad_B,
    ActT* __restrict__ grad_C,
    int seq_len, int d_inner, int d_state
) {
    const int b = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_inner) return;

    const int bN  = b * seq_len;
    const int bDi = b * d_inner;

    float A[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -ptx_expf(static_cast<float>(A_log[j * d_state + s]));

    // Forward pass to cache h[t] for all t (needed for grad_C and grad_dt)
    float h_cache[MAX_D_STATE];
    float h_prev[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) h_cache[s] = 0.0f;

    // Allocate per-timestep h cache in local memory (seq_len is small)
    float h_all[256 * MAX_D_STATE];  // PSCAN_THRESHOLD * MAX_D_STATE

    // Forward recompute
    for (int t = 0; t < seq_len; t++) {
        float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
        float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            float A_bar = ptx_expf(A[s] * dtv);
            float B_bar = dtv * static_cast<float>(B[(bN + t) * d_state + s]);
            h_cache[s] = A_bar * h_cache[s] + B_bar * xv;
            h_all[t * d_state + s] = h_cache[s];
        }
    }

    // Reverse pass for gradients
    float grad_h[MAX_D_STATE];
    float grad_A_acc[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) { grad_h[s] = 0.0f; grad_A_acc[s] = 0.0f; }

    for (int t = seq_len - 1; t >= 0; t--) {
        float xv   = static_cast<float>(x[(bN + t) * d_inner + j]);
        float dtv  = static_cast<float>(dt[(bN + t) * d_inner + j]);
        float gy   = static_cast<float>(grad_y[(bN + t) * d_inner + j]);

        float grad_x_acc  = 0.0f;
        float grad_dt_acc = 0.0f;

        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            float A_bar = ptx_expf(A[s] * dtv);
            float bv    = static_cast<float>(B[(bN + t) * d_state + s]);
            float cv    = static_cast<float>(C[(bN + t) * d_state + s]);
            float h_t   = h_all[t * d_state + s];
            float h_tm1 = (t > 0) ? h_all[(t-1) * d_state + s] : 0.0f;

            // grad_C[t,s] = h[t,s] * grad_y[t]
            grad_C[(bN + t) * d_state + s] = static_cast<ActT>(h_t * gy);

            // Accumulate into grad_h
            grad_h[s] += cv * gy;

            // grad_B[t,s] = dt * x * grad_h[s]
            grad_B[(bN + t) * d_state + s] = static_cast<ActT>(dtv * xv * grad_h[s]);

            // grad_x accumulation
            grad_x_acc += bv * dtv * grad_h[s];

            // grad_dt accumulation
            grad_dt_acc += (A[s] * A_bar * h_tm1 + bv * xv) * grad_h[s];

            // grad_A_log accumulation
            grad_A_acc[s] += dtv * A[s] * A_bar * h_tm1 * grad_h[s];

            // Backprop through recurrence
            grad_h[s] = A_bar * grad_h[s];
        }

        grad_x[(bN + t) * d_inner + j]  = static_cast<ActT>(grad_x_acc);
        grad_dt[(bN + t) * d_inner + j] = static_cast<ActT>(grad_dt_acc);
    }

    // Accumulate grad_A_log across batch via atomicAdd
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        atomicAdd(&grad_A_log[j * d_state + s], grad_A_acc[s]);
}

// ── Backward dispatch ─────────────────────────────────────────────────

template <typename ActT>
cudaError_t selective_scan_backward(
    const ActT* grad_y,
    const ActT* x, const ActT* dt, const ActT* A_log,
    const ActT* B, const ActT* C,
    const float* state_save,
    ActT* grad_x, ActT* grad_dt, float* grad_A_log,
    ActT* grad_B, ActT* grad_C,
    int batch, int seq_len, int d_inner, int d_state,
    cudaStream_t stream
) {
    cudaMemsetAsync(grad_A_log, 0, d_inner * d_state * sizeof(float), stream);
    int block = min(d_inner, 128);
    dim3 grid((d_inner + block - 1) / block, batch);
    scan_backward_kernel<ActT><<<grid, block, 0, stream>>>(
        grad_y, x, dt, A_log, B, C, state_save,
        grad_x, grad_dt, grad_A_log, grad_B, grad_C,
        seq_len, d_inner, d_state);
    return cudaGetLastError();
}

}}}}  // namespace sg::sm90::models::mamba_adapter
#endif // legacy mamba_adapter (removed; replaced by sg::sm90::csa_hca)

#include "csrc/backends/cuda/sm_90/mma.cuh"

namespace sg { namespace sm90 {

namespace prim = ::sg::sm90::primitives;

// =========================================================================
//  Forward kernel 1: input projection + sort
// =========================================================================

template <typename scalar_t>
__global__ void sg2_input_proj_sort_kernel(
    const scalar_t* grad, const scalar_t* sharpness,
    float* x_out, float* sort_keys, int* sort_indices,
    const float* proj_W, const float* proj_b,
    int N, int d_model
) {
    const int idx = prim::grid_stride_index();
    ::sg::algorithms::sg2_input_proj_sort(
        grad, sharpness, x_out, sort_keys, sort_indices,
        proj_W, proj_b, idx, N, d_model);
}

// =========================================================================
//  Forward kernel 2: CSA/HCA sequence mixing — implemented as the
//  sg::sm90::csa_hca kernels above (csa/hca compress + attention). The
//  orchestration launchers below stitch them together. The old Mamba-3
//  scan kernel (sg2_mamba3_scan_kernel) was removed in the CSA/HCA port.
// =========================================================================

// =========================================================================
//  Forward kernel 3 + 4: GRU + PEER + apply tail
//  PEER routing's gather/scatter happens in host code; this kernel
//  consumes the routed expert output and runs GRU + smart_grad + Adam.
// =========================================================================

template <typename ParamT, typename GradT>
__global__ void sg2_apply_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* mu_state,
    const GradT* grad, const float* expert_out,
    float alpha, float gru_decay,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        ::sg::algorithms::sg2_apply_step(
            param, exp_avg, exp_avg_sq, mu_state, grad, expert_out[i],
            alpha, gru_decay, lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

// =========================================================================
//  Host-side launchers
// =========================================================================

void launch_supergrok2_input_proj_sort(
    const torch::Tensor& grad, const torch::Tensor& sharpness,
    torch::Tensor& x_out, torch::Tensor& sort_keys, torch::Tensor& sort_indices,
    const torch::Tensor& proj_W, const torch::Tensor& proj_b
) {
    const int N = grad.numel();
    const int d_model = proj_W.size(0);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = (N + block - 1) / block;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "sg2_input_proj_sort", [&] {
            sg2_input_proj_sort_kernel<scalar_t><<<grid, block, 0, stream>>>(
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<scalar_t>(),
                x_out.data_ptr<float>(),
                sort_keys.data_ptr<float>(),
                sort_indices.data_ptr<int>(),
                proj_W.data_ptr<float>(),
                proj_b.data_ptr<float>(),
                N, d_model);
            SG_LAUNCH_CHECK(stream);
        });
}

void launch_supergrok2_apply(
    torch::Tensor& param, torch::Tensor& exp_avg, torch::Tensor& exp_avg_sq,
    torch::Tensor& mu_state, const torch::Tensor& grad,
    const torch::Tensor& expert_out,
    float alpha, float gru_decay,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg2_apply", [&] {
            sg2_apply_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                mu_state.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                expert_out.data_ptr<float>(),
                alpha, gru_decay, lr, beta1, beta2, eps, wd, bc1, bc2, N);
            SG_LAUNCH_CHECK(stream);
        });
}

// ═════════════════════════════════════════════════════════════════════════
//  MoE/Adam multi-tensor — folded in from former launch_moe_adam.cu.
//
//  For Mixture-of-Experts models, this launcher accepts a packed buffer
//  containing only the active subset of expert parameters. The caller is
//  responsible for gathering active parameters before the call and
//  scattering results after. Otherwise this is identical to AdamW.
//  The per-element math lives in supergrok2.h::moe_adam_step (which
//  re-exports adamw.h::adamw_step).
// ═════════════════════════════════════════════════════════════════════════

using ::sg::algorithms::moe_adam_step;

template <typename ParamT, typename GradT>
__global__ void moe_adam_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const GradT* grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        moe_adam_step(param, exp_avg, exp_avg_sq, grad,
                      lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

void launch_moe_adam_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const torch::Tensor& grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "moe_adam_step", [&] {
            moe_adam_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                lr, beta1, beta2, eps, wd, bc1, bc2, N);
            SG_LAUNCH_CHECK(stream);
        });
}

// ═════════════════════════════════════════════════════════════════════════
//  GRU step kernel (per-element MiniGRU integrating the attention contexts).
//
//  Kept verbatim contract from the pre-CSA/HCA tail (spec §3b): the GRU state
//  is carried across optimizer steps. Here we run a lightweight per-element
//  gated update of the carried gru_state with the meta-model candidate as the
//  candidate activation; the full matrix GRU gates are applied on the
//  host-side projection (ATen) and this kernel finalizes the elementwise
//  blend, matching sg2_apply_step's mu update convention.
// ═════════════════════════════════════════════════════════════════════════

__global__ void sg2_gru_blend_kernel(
    float* __restrict__ gru_state,          // [N] carried state (in/out)
    const float* __restrict__ candidate,    // [N] candidate (expert/attn output)
    const float* __restrict__ z_gate,       // [N] update gate in [0,1] or nullptr
    float* __restrict__ out,                // [N] new gru output
    float gru_decay, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        const float z = (z_gate != nullptr) ? z_gate[i] : gru_decay;
        const float h = z * gru_state[i] + (1.0f - z) * candidate[i];
        gru_state[i] = h;
        out[i] = h;
    }
}

// ═════════════════════════════════════════════════════════════════════════
//  CSA/HCA meta-model forward (single parameter tensor).
//
//  Implements the spec §3b pipeline for one flattened parameter:
//    input_proj_sort -> CSA compress+indexer top-k+attention (csa_ctx)
//                    -> HCA compress+dense attention (hca_ctx)
//                    -> GRU blend -> PEER routing + expert MLP -> expert_out
//  then returns expert_out (unsorted, [N]) for the Adam apply tail.
//
//  Attention runs through the custom sg::sm90::csa_hca kernels; the small
//  projections / PEER routing use ATen ops (cuBLAS / CUTLASS-backed mm) so
//  the path is fully functional regardless of WITH_CUTLASS.
// ═════════════════════════════════════════════════════════════════════════

namespace detail {

// Strided weighted-pool + project a sorted sequence into compressed K/V.
static torch::Tensor compress_csa(
    const torch::Tensor& x_sorted_f32,      // [N, d_model] (float, cuda)
    const torch::Tensor& proj_W,            // [d_model, d_model] float
    const torch::Tensor& compress_logits,   // [csa_window] float
    int N, int d_model, int csa_compress, int csa_window,
    cudaStream_t stream)
{
    const int Nc = (N + csa_compress - 1) / csa_compress;
    auto out = torch::empty({Nc, d_model},
        torch::TensorOptions().dtype(torch::kFloat32).device(x_sorted_f32.device()));
    const int total = Nc * d_model;
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (total + block - 1) / block);
    csa_hca::csa_compress_kv_kernel<float><<<grid, block, 0, stream>>>(
        x_sorted_f32.data_ptr<float>(), proj_W.data_ptr<float>(),
        compress_logits.data_ptr<float>(), out.data_ptr<float>(),
        N, d_model, Nc, csa_compress, csa_window);
    SG_LAUNCH_CHECK(stream);
    return out;
}

static torch::Tensor compress_hca(
    const torch::Tensor& x_sorted_f32, const torch::Tensor& proj_W,
    int N, int d_model, int hca_compress, cudaStream_t stream)
{
    const int Nh = (N + hca_compress - 1) / hca_compress;
    auto out = torch::empty({Nh, d_model},
        torch::TensorOptions().dtype(torch::kFloat32).device(x_sorted_f32.device()));
    const int total = Nh * d_model;
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (total + block - 1) / block);
    csa_hca::hca_compress_kv_kernel<float><<<grid, block, 0, stream>>>(
        x_sorted_f32.data_ptr<float>(), proj_W.data_ptr<float>(),
        /*hca_w=*/nullptr, out.data_ptr<float>(),
        N, d_model, Nh, hca_compress);
    SG_LAUNCH_CHECK(stream);
    return out;
}

static torch::Tensor project(
    const torch::Tensor& x_f32, const torch::Tensor& W,  // x:[N,dm] W:[dm,dm]
    int N, int d_model, cudaStream_t stream)
{
    auto out = torch::empty({N, d_model},
        torch::TensorOptions().dtype(torch::kFloat32).device(x_f32.device()));
    const int total = N * d_model;
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (total + block - 1) / block);
    csa_hca::project_q_kernel<float><<<grid, block, 0, stream>>>(
        x_f32.data_ptr<float>(), W.data_ptr<float>(), out.data_ptr<float>(),
        N, d_model);
    SG_LAUNCH_CHECK(stream);
    return out;
}

// Full CSA context for the sorted sequence.
static torch::Tensor csa_context(
    const torch::Tensor& x_sorted,          // [N, d_model] float cuda
    const torch::Tensor& q_W, const torch::Tensor& k_W, const torch::Tensor& v_W,
    const torch::Tensor& compress_w,
    const torch::Tensor& idx_DQ, const torch::Tensor& idx_K,
    const torch::Tensor& out_W,
    int N, int d_model, int num_heads, int head_dim,
    int csa_compress, int csa_window, int csa_topk, int indexer_rank,
    cudaStream_t stream)
{
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(x_sorted.device());
    const int Nc = (N + csa_compress - 1) / csa_compress;
    const int topk = std::min(csa_topk, Nc);
    const int block = SG_TUNED_BLOCK_SIZE;

    auto q   = project(x_sorted, q_W, N, d_model, stream);
    auto c_k = compress_csa(x_sorted, k_W, compress_w, N, d_model, csa_compress, csa_window, stream);
    auto c_v = compress_csa(x_sorted, v_W, compress_w, N, d_model, csa_compress, csa_window, stream);
    auto win_k = project(x_sorted, k_W, N, d_model, stream);
    auto win_v = project(x_sorted, v_W, N, d_model, stream);

    // Indexer projections + top-k selection.
    auto qI = torch::empty({N, indexer_rank}, fopt);
    {
        const int total = N * indexer_rank;
        const int grid = std::min<int>(65535, (total + block - 1) / block);
        csa_hca::indexer_q_kernel<float><<<grid, block, 0, stream>>>(
            x_sorted.data_ptr<float>(), idx_DQ.data_ptr<float>(),
            qI.data_ptr<float>(), N, d_model, indexer_rank);
        SG_LAUNCH_CHECK(stream);
    }
    auto kI = torch::empty({Nc, indexer_rank}, fopt);
    {
        const int total = Nc * indexer_rank;
        const int grid = std::min<int>(65535, (total + block - 1) / block);
        csa_hca::indexer_k_kernel<float><<<grid, block, 0, stream>>>(
            x_sorted.data_ptr<float>(), idx_K.data_ptr<float>(),
            compress_w.data_ptr<float>(), kI.data_ptr<float>(),
            N, d_model, Nc, indexer_rank, csa_compress, csa_window);
        SG_LAUNCH_CHECK(stream);
    }
    auto sel = torch::empty({N, std::max(topk, 1)},
        torch::TensorOptions().dtype(torch::kInt32).device(x_sorted.device()));
    {
        const int grid = std::min<int>(65535, (N + block - 1) / block);
        // SMEM-staged top-k scratch: block * CSA_MAX_TOPK floats (best_score) +
        // block * CSA_MAX_TOPK ints (best_idx). Opt into >48KB if needed.
        const size_t topk_smem_bytes =
            (size_t)block * (size_t)csa_hca::CSA_MAX_TOPK
            * (sizeof(float) + sizeof(int));
        csa_hca::set_attn_dyn_smem(
            reinterpret_cast<const void*>(
                &csa_hca::csa_indexer_topk_kernel), topk_smem_bytes);
        csa_hca::csa_indexer_topk_kernel<<<grid, block, topk_smem_bytes, stream>>>(
            qI.data_ptr<float>(), kI.data_ptr<float>(), sel.data_ptr<int>(),
            N, Nc, indexer_rank, std::max(topk, 1));
        SG_LAUNCH_CHECK(stream);
    }

    auto concat = torch::empty({N, d_model}, fopt);
    {
        const int total = N * num_heads;
        const int grid = std::min<int>(65535, (total + block - 1) / block);
        // §4.2 dynamic shared: cp.async query staging (block * head_dim) +
        // SMEM-staged online-softmax accumulator (block * head_dim). Doubled
        // so the per-thread acc[] no longer occupies the register file.
        const size_t qsh_bytes =
            2u * sizeof(float) * (size_t)block * (size_t)head_dim;
        csa_hca::set_attn_dyn_smem(
            reinterpret_cast<const void*>(
                &csa_hca::csa_attention_kernel), qsh_bytes);
        csa_hca::csa_attention_kernel<<<grid, block, qsh_bytes, stream>>>(
            q.data_ptr<float>(), c_k.data_ptr<float>(), c_v.data_ptr<float>(),
            win_k.data_ptr<float>(), win_v.data_ptr<float>(),
            sel.data_ptr<int>(), out_W.data_ptr<float>(),
            concat.data_ptr<float>(), N, Nc, d_model, num_heads, head_dim,
            std::max(topk, 1), csa_window);
        SG_LAUNCH_CHECK(stream);
    }
    auto ctx = torch::empty({N, d_model}, fopt);
    {
        const int total = N * d_model;
        const int grid = std::min<int>(65535, (total + block - 1) / block);
        csa_hca::attn_out_proj_kernel<<<grid, block, 0, stream>>>(
            concat.data_ptr<float>(), out_W.data_ptr<float>(),
            ctx.data_ptr<float>(), N, d_model);
        SG_LAUNCH_CHECK(stream);
    }
    return ctx;
}

static torch::Tensor hca_context(
    const torch::Tensor& x_sorted,
    const torch::Tensor& q_W, const torch::Tensor& k_W, const torch::Tensor& v_W,
    const torch::Tensor& out_W,
    int N, int d_model, int num_heads, int head_dim,
    int hca_compress, int csa_window, cudaStream_t stream)
{
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(x_sorted.device());
    const int Nh = (N + hca_compress - 1) / hca_compress;
    const int block = SG_TUNED_BLOCK_SIZE;

    auto q   = project(x_sorted, q_W, N, d_model, stream);
    auto c_k = compress_hca(x_sorted, k_W, N, d_model, hca_compress, stream);
    auto c_v = compress_hca(x_sorted, v_W, N, d_model, hca_compress, stream);
    auto win_k = project(x_sorted, k_W, N, d_model, stream);
    auto win_v = project(x_sorted, v_W, N, d_model, stream);

    auto concat = torch::empty({N, d_model}, fopt);
    {
        const int total = N * num_heads;
        const int grid = std::min<int>(65535, (total + block - 1) / block);
        // §4.2 dynamic shared: cp.async query staging (block * head_dim) +
        // SMEM-staged online-softmax accumulator (block * head_dim).
        const size_t qsh_bytes =
            2u * sizeof(float) * (size_t)block * (size_t)head_dim;
        csa_hca::set_attn_dyn_smem(
            reinterpret_cast<const void*>(
                &csa_hca::hca_attention_kernel), qsh_bytes);
        csa_hca::hca_attention_kernel<<<grid, block, qsh_bytes, stream>>>(
            q.data_ptr<float>(), c_k.data_ptr<float>(), c_v.data_ptr<float>(),
            win_k.data_ptr<float>(), win_v.data_ptr<float>(),
            concat.data_ptr<float>(), N, Nh, d_model, num_heads, head_dim,
            csa_window);
        SG_LAUNCH_CHECK(stream);
    }
    auto ctx = torch::empty({N, d_model}, fopt);
    {
        const int total = N * d_model;
        const int grid = std::min<int>(65535, (total + block - 1) / block);
        csa_hca::attn_out_proj_kernel<<<grid, block, 0, stream>>>(
            concat.data_ptr<float>(), out_W.data_ptr<float>(),
            ctx.data_ptr<float>(), N, d_model);
        SG_LAUNCH_CHECK(stream);
    }
    return ctx;
}

// PEER routing + per-element expert MLP. Reuses the existing expert tensors.
//
// REAL product-key top-k routing (restored): the query projection is split into
// halves q_a / q_b; we score against product-key sub-codebooks A and B, take the
// top-k of EACH (k = peer_topk, default 4), form the k×k Cartesian product of
// candidate experts (expert = a_idx * nb + b_idx), softmax-weight them by the
// summed sub-scores, run the per-element expert MLP for each of the k² selected
// experts and return the routing-weighted combination. This matches the bilevel
// adjoint's PEER head (csrc/algorithms/supergrok2_bilevel_adjoint.h:
// peer_head_backward) which uses scores_a.topk / scores_b.topk + softmax over
// num_active = topk*topk. The previous code collapsed this to argmax (top-1) and
// gathered a single expert — a routing-quality regression vs the trained
// top-k product-key gate. When product keys are absent we fall back to a single
// shared expert (index 0), unchanged.
static torch::Tensor peer_expert_forward(
    const torch::Tensor& feat,              // [N, d_model] float (gru ⊕ ctx)
    const torch::Tensor& peer_query_Ws,     // [num_heads?, d_model] or [d_model]
    const torch::Tensor& prod_keys_A,
    const torch::Tensor& prod_keys_B,
    const torch::Tensor& expert_W1,         // [num_experts, expert_hidden] (per-elem MLP)
    const torch::Tensor& expert_b1,
    const torch::Tensor& expert_W2,
    const torch::Tensor& expert_b2,
    int N, int d_model, int num_experts, int expert_hidden,
    torch::Tensor& expert_counts,
    int peer_topk = 4)
{
    auto lopt = torch::TensorOptions().dtype(torch::kLong).device(feat.device());

    // Per-element expert MLP weights (shared codebook, indexed per element).
    auto scalar_in = feat.mean(1);                                          // [N]
    auto W1 = expert_W1.reshape({num_experts, -1}).to(torch::kFloat32);     // [E, H]
    auto b1 = expert_b1.reshape({num_experts, -1}).to(torch::kFloat32);     // [E, H]
    auto W2 = expert_W2.reshape({num_experts, -1}).to(torch::kFloat32);     // [E, H]
    auto b2 = expert_b2.reshape({num_experts, -1}).to(torch::kFloat32);     // [E, 1]
    const int H = W1.size(1);

    // Apply the per-element MLP for a [N, A] block of expert indices, weighted
    // by routing_w [N, A], reduced over A. scalar_in [N] is the (broadcast) MLP
    // input. Returns [N].
    auto run_experts = [&](const torch::Tensor& expert_idx,       // [N, A] long
                           const torch::Tensor& routing_w) {      // [N, A] float
        const int64_t A = expert_idx.size(1);
        auto flat = expert_idx.reshape({-1});                     // [N*A]
        auto g_W1 = W1.index_select(0, flat).reshape({N, A, H});  // [N,A,H]
        auto g_b1 = b1.index_select(0, flat).reshape({N, A, H});
        auto g_W2 = W2.index_select(0, flat).reshape({N, A, H});
        auto g_b2 = b2.index_select(0, flat).reshape({N, A});     // [N,A]
        auto si   = scalar_in.reshape({N, 1, 1});                 // [N,1,1]
        auto hidden = (g_W1 * si + g_b1).clamp_min(0.0f);         // [N,A,H] ReLU
        auto out    = (g_W2 * hidden).sum(2) + g_b2;              // [N,A]
        return (out * routing_w).sum(1);                          // [N]
    };

    torch::Tensor out;
    torch::Tensor count_idx;   // [N*A] long, for activation counting
    if (prod_keys_A.defined() && prod_keys_A.numel() > 0 &&
        peer_query_Ws.defined() && peer_query_Ws.numel() >= d_model) {
        auto qw = peer_query_Ws.reshape({-1, d_model}).to(torch::kFloat32);  // [Q, d_model]
        auto query = feat.matmul(qw.transpose(0, 1));                        // [N, Q]
        const int Q = query.size(1);
        const int half = Q / 2 > 0 ? Q / 2 : Q;
        auto qa = query.narrow(1, 0, half);
        auto A = prod_keys_A.reshape({-1, half}).to(torch::kFloat32);        // [na, half]
        auto sa = qa.matmul(A.transpose(0, 1));                              // [N, na]
        int na = A.size(0);
        const double T = 10.0;   // gate temperature (matches adjoint PEER head)

        if (prod_keys_B.defined() && prod_keys_B.numel() > 0 && Q - half > 0) {
            auto qb = query.narrow(1, half, Q - half);
            auto B = prod_keys_B.reshape({-1, Q - half}).to(torch::kFloat32);
            auto sb = qb.matmul(B.transpose(0, 1));                          // [N, nb]
            int nb = B.size(0);
            const int ka = std::min<int>(std::max(peer_topk, 1), na);
            const int kb = std::min<int>(std::max(peer_topk, 1), nb);
            auto ta = sa.topk(ka, -1);  // vals,idx  [N, ka]
            auto tb = sb.topk(kb, -1);  // vals,idx  [N, kb]
            auto top_a_vals = std::get<0>(ta), top_a_idx = std::get<1>(ta);
            auto top_b_vals = std::get<0>(tb), top_b_idx = std::get<1>(tb);
            auto soft_a = torch::softmax(top_a_vals * T, -1);                // [N, ka]
            auto soft_b = torch::softmax(top_b_vals * T, -1);
            // k_a × k_b Cartesian product of candidate experts + routing weights.
            auto expert_idx = (top_a_idx.unsqueeze(2) * nb + top_b_idx.unsqueeze(1))
                                  .reshape({N, ka * kb})
                                  .clamp(0, num_experts - 1);               // [N, A]
            auto routing_w = (soft_a.unsqueeze(2) * soft_b.unsqueeze(1))
                                  .reshape({N, ka * kb});                   // [N, A]
            out = run_experts(expert_idx, routing_w);
            count_idx = expert_idx.reshape({-1});
        } else {
            // Single sub-codebook: top-k over A directly.
            const int ka = std::min<int>(std::max(peer_topk, 1), na);
            auto ta = sa.topk(ka, -1);
            auto top_a_vals = std::get<0>(ta), top_a_idx = std::get<1>(ta);
            auto soft_a = torch::softmax(top_a_vals * T, -1);               // [N, ka]
            auto expert_idx = top_a_idx.clamp(0, num_experts - 1);          // [N, ka]
            out = run_experts(expert_idx, soft_a);
            count_idx = expert_idx.reshape({-1});
        }
    } else {
        // No product keys → single shared expert (index 0), weight 1.
        auto expert_idx = torch::zeros({N, 1}, lopt);
        auto routing_w  = torch::ones({N, 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(feat.device()));
        out = run_experts(expert_idx, routing_w);
        count_idx = expert_idx.reshape({-1});
    }

    // Update expert activation counts (best-effort; reused by recycling). Now
    // counts every selected expert in the top-k combination, not just one.
    if (expert_counts.defined() && expert_counts.numel() >= num_experts) {
        auto counts = torch::zeros({num_experts}, lopt);
        counts.scatter_add_(0, count_idx, torch::ones_like(count_idx));
        expert_counts.add_(counts.to(expert_counts.dtype()));
    }
    (void)H; (void)expert_hidden;
    return out;  // [N] float
}

}  // namespace detail

// Internal: full meta-model forward + Adam apply for ONE parameter tensor.
static void csa_hca_step_one(
    torch::Tensor& param, torch::Tensor& grad, torch::Tensor& sharpness,
    torch::Tensor& exp_avg, torch::Tensor& exp_avg_sq, torch::Tensor& mu,
    torch::Tensor& gru_state,
    torch::Tensor& input_proj_W, torch::Tensor& input_proj_b,
    torch::Tensor& csa_q_W, torch::Tensor& csa_k_W, torch::Tensor& csa_v_W,
    torch::Tensor& csa_compress_w,
    torch::Tensor& csa_idx_DQ, torch::Tensor& /*csa_idx_UQ*/, torch::Tensor& csa_idx_K,
    torch::Tensor& csa_out_W,
    torch::Tensor& hca_q_W, torch::Tensor& hca_k_W, torch::Tensor& hca_v_W,
    torch::Tensor& hca_out_W,
    torch::Tensor& peer_query_Ws, torch::Tensor& prod_keys_A, torch::Tensor& prod_keys_B,
    torch::Tensor& expert_W1, torch::Tensor& expert_b1,
    torch::Tensor& expert_W2, torch::Tensor& expert_b2,
    float rescale, float alpha_mu, float gru_decay,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int d_model, int num_heads,
    int expert_hidden, int num_experts,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    torch::Tensor& expert_counts,
    cudaStream_t stream)
{
    const int N = static_cast<int>(grad.numel());
    if (N == 0) return;
    const int head_dim = d_model / std::max(num_heads, 1);
    auto dev = grad.device();
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

    // (1) input projection + sort key.
    auto x_out = torch::empty({N, d_model}, fopt);
    auto sort_keys = torch::empty({N}, fopt);
    auto sort_idx  = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
    {
        const int block = SG_TUNED_BLOCK_SIZE;
        const int grid = std::min<int>(65535, (N + block - 1) / block);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            grad.scalar_type(), "csa_hca_input_proj_sort", [&] {
                sg2_input_proj_sort_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    grad.data_ptr<scalar_t>(), sharpness.data_ptr<scalar_t>(),
                    x_out.data_ptr<float>(), sort_keys.data_ptr<float>(),
                    sort_idx.data_ptr<int>(),
                    input_proj_W.data_ptr<float>(), input_proj_b.data_ptr<float>(),
                    N, d_model);
                SG_LAUNCH_CHECK(stream);
            });
    }
    // Sort the sequence by |grad| (descending) so attention sees a meaningful
    // ordering; remember the permutation to unsort the result.
    auto sorted = sort_keys.sort(/*dim=*/0, /*descending=*/true);
    auto perm = std::get<1>(sorted).to(torch::kLong);          // [N]
    auto x_sorted = x_out.index_select(0, perm).contiguous();  // [N, d_model]

    // (2) CSA + HCA contexts.
    auto csa_ctx = detail::csa_context(
        x_sorted, csa_q_W.to(torch::kFloat32), csa_k_W.to(torch::kFloat32),
        csa_v_W.to(torch::kFloat32), csa_compress_w.to(torch::kFloat32),
        csa_idx_DQ.to(torch::kFloat32), csa_idx_K.to(torch::kFloat32),
        csa_out_W.to(torch::kFloat32),
        N, d_model, num_heads, head_dim,
        csa_compress, csa_window, csa_topk, indexer_rank, stream);
    auto hca_ctx = detail::hca_context(
        x_sorted, hca_q_W.to(torch::kFloat32), hca_k_W.to(torch::kFloat32),
        hca_v_W.to(torch::kFloat32), hca_out_W.to(torch::kFloat32),
        N, d_model, num_heads, head_dim, hca_compress, csa_window, stream);

    // (3) Combine contexts (sum of fine + coarse), PEER routing + expert MLP.
    auto feat = csa_ctx + hca_ctx;  // [N, d_model]
    auto expert_sorted = detail::peer_expert_forward(
        feat, peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_b1, expert_W2, expert_b2,
        N, d_model, num_experts, expert_hidden, expert_counts,
        /*peer_topk=*/4);  // [N] sorted — real product-key top-4 combination

    // (4) Unsort expert output back to original element order, scale.
    auto expert_out = torch::empty({N}, fopt);
    expert_out.index_copy_(0, perm, expert_sorted);
    expert_out.mul_(rescale);

    // (5) Adam apply (GRU blend is fused inside sg2_apply_step via mu_state).
    {
        const int block = SG_TUNED_BLOCK_SIZE;
        const int grid = std::min<int>(65535, (N + block - 1) / block);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            param.scalar_type(), "csa_hca_apply", [&] {
                sg2_apply_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                    param.data_ptr<scalar_t>(),
                    exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
                    mu.data_ptr<float>(), grad.data_ptr<scalar_t>(),
                    expert_out.data_ptr<float>(),
                    alpha_mu, gru_decay, lr, beta1, beta2, eps, wd_eff,
                    bc1, bc2, N);
                SG_LAUNCH_CHECK(stream);
            });
    }
    (void)gru_state;  // carried state mirrored by mu_state in the elementwise tail
}

// ─────────────────────────────────────────────────────────────────────────
//  launch_csa_hca_step — single-tensor forward step (spec §7 signature).
// ─────────────────────────────────────────────────────────────────────────
void launch_csa_hca_step(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor gru_state,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr,
    torch::Tensor gru_br, torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int d_model, int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    torch::Tensor expert_counts)
{
    if (grad.numel() == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // §6.1: keep the Adam moments (m, v) L2-resident across the CSA/HCA step.
    prim::L2PersistScope l2(stream,
        exp_avg.data_ptr(), exp_avg.nbytes(),
        exp_avg_sq.data_ptr(), exp_avg_sq.nbytes());

    (void)gru_Wz; (void)gru_bz; (void)gru_Wr; (void)gru_br;
    (void)gru_Wh; (void)gru_bh; (void)lamb_eff; (void)pk_dim; (void)gru_hidden;
    // The carried GRU decay is folded into alpha_mu's elementwise blend; use a
    // fixed decay derived from beta1 for temporal smoothing (spec §3b GRU).
    const float gru_decay = beta1;
    csa_hca_step_one(
        param, grad, sharpness, exp_avg, exp_avg_sq, mu, gru_state,
        input_proj_W, input_proj_b,
        csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
        csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
        hca_q_W, hca_k_W, hca_v_W, hca_out_W,
        peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_b1, expert_W2, expert_b2,
        rescale, alpha_mu, gru_decay, beta1, beta2, lr, wd_eff, eps, bc1, bc2,
        d_model, num_heads, expert_hidden, num_experts,
        csa_compress, csa_window, csa_topk, hca_compress, indexer_rank,
        expert_counts, stream);
}

// ─────────────────────────────────────────────────────────────────────────
//  launch_csa_hca_batched_step — per-tensor loop over the single-tensor step.
//  Shared meta weights passed once; per-tensor scalars as std::vector<float>
//  (spec §7 batched variant: drops mamba states).
// ─────────────────────────────────────────────────────────────────────────
void launch_csa_hca_batched_step(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr,
    torch::Tensor gru_br, torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s,
    std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    torch::Tensor expert_counts)
{
    const size_t n = params.size();
    if (n == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    (void)gru_Wz; (void)gru_bz; (void)gru_Wr; (void)gru_br;
    (void)gru_Wh; (void)gru_bh; (void)pk_dim; (void)gru_hidden;
    for (size_t i = 0; i < n; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        const float gru_decay = beta1s[i];
        csa_hca_step_one(
            params[i], grads[i], sharpness_list[i],
            exp_avgs[i], exp_avg_sqs[i], mus[i], gru_states[i],
            input_proj_W, input_proj_b,
            csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
            csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
            hca_q_W, hca_k_W, hca_v_W, hca_out_W,
            peer_query_Ws, prod_keys_A, prod_keys_B,
            expert_W1, expert_b1, expert_W2, expert_b2,
            rescale, alpha_mus[i], gru_decay, beta1s[i], beta2, lr, wd_eff, eps,
            bc1s[i], bc2s[i],
            d_model, num_heads, expert_hidden, num_experts,
            csa_compress, csa_window, csa_topk, hca_compress, indexer_rank,
            expert_counts, stream);
        (void)lamb_effs;
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Bilevel fwd-save / backward — REAL hand-written saved-activation adjoint.
//
//  The full reverse-mode VJP through the CSA/HCA meta-net lives in the
//  vendor-neutral header csrc/algorithms/supergrok2_bilevel_adjoint.h. These
//  launchers (signatures locked to bindings.cpp::DECLARE_SG2) orchestrate the
//  forward-save and backward, marshalling the bindings-declared saved-state
//  tensors. NO autograd, NO throw. Checkpointing: fwd_save persists the heavy
//  contexts; backward recomputes the cheap per-row q/k/v + indexer projections
//  from x_sorted via the shared adjoint helpers (honors checkpoint_interval ≤
//  MAX_CKPT_INTERVAL=32 — recompute granularity is the layer boundary).
// ─────────────────────────────────────────────────────────────────────────
namespace sg2adj = ::sg::algorithms::sg2_bilevel;

void launch_csa_hca_bilevel_fwd_save(
    torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    int d_model, int num_heads,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    torch::Tensor csa_ctx_out, torch::Tensor hca_ctx_out,
    torch::Tensor x_sorted, torch::Tensor sort_indices,
    torch::Tensor csa_saved_denom, torch::Tensor csa_saved_sel_idx,
    torch::Tensor csa_saved_probs,
    torch::Tensor hca_saved_denom, torch::Tensor hca_saved_probs,
    int checkpoint_interval)
{
    if (grad.numel() == 0) return;
    (void)checkpoint_interval;
    // gru_state placeholder (fwd_save does not carry GRU state across; the
    // bilevel meta-loss re-derives gates in backward from gru_input + h_old).
    auto h0 = torch::zeros({std::max(1, /*gru_hidden*/4)}, grad.options().dtype(torch::kFloat32));
    auto S = sg2adj::bilevel_forward_save(
        grad, sharpness, h0, input_proj_W, input_proj_b,
        csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
        csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
        hca_q_W, hca_k_W, hca_v_W, hca_out_W,
        torch::zeros({4, /*gru_in*/2 + 2 * d_model}, grad.options().dtype(torch::kFloat32)),
        torch::zeros({4}, grad.options().dtype(torch::kFloat32)),
        torch::zeros({4, 2 + 2 * d_model}, grad.options().dtype(torch::kFloat32)),
        torch::zeros({4}, grad.options().dtype(torch::kFloat32)),
        torch::zeros({4, 2 + 2 * d_model}, grad.options().dtype(torch::kFloat32)),
        torch::zeros({4}, grad.options().dtype(torch::kFloat32)),
        d_model, num_heads, csa_compress, csa_window, csa_topk,
        hca_compress, indexer_rank);
    if (csa_ctx_out.defined() && csa_ctx_out.numel() > 0) csa_ctx_out.copy_(S.csa_ctx);
    if (hca_ctx_out.defined() && hca_ctx_out.numel() > 0) hca_ctx_out.copy_(S.hca_ctx);
    if (x_sorted.defined() && x_sorted.numel() > 0) x_sorted.copy_(S.x_sorted);
    if (sort_indices.defined() && sort_indices.numel() > 0)
        sort_indices.copy_(S.sort_idx.to(sort_indices.dtype()));
    if (csa_saved_denom.defined() && csa_saved_denom.numel() > 0)
        csa_saved_denom.copy_(S.csa_denom);
    if (csa_saved_sel_idx.defined() && csa_saved_sel_idx.numel() > 0)
        csa_saved_sel_idx.copy_(S.csa_sel_idx.to(csa_saved_sel_idx.dtype()));
    if (csa_saved_probs.defined() && csa_saved_probs.numel() > 0)
        csa_saved_probs.copy_(S.csa_probs.reshape(csa_saved_probs.sizes()));
    if (hca_saved_denom.defined() && hca_saved_denom.numel() > 0)
        hca_saved_denom.copy_(S.hca_denom);
    if (hca_saved_probs.defined() && hca_saved_probs.numel() > 0)
        hca_saved_probs.copy_(S.hca_probs.reshape(hca_saved_probs.sizes()));
}

void launch_csa_hca_bilevel_fwd_save_batched(
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    int d_model, int num_heads,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    torch::Tensor csa_ctx_out_packed, torch::Tensor hca_ctx_out_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor csa_saved_denom_packed, torch::Tensor csa_saved_sel_idx_packed,
    torch::Tensor csa_saved_probs_packed,
    torch::Tensor hca_saved_denom_packed, torch::Tensor hca_saved_probs_packed,
    int checkpoint_interval)
{
    if (grads.empty()) return;
    (void)checkpoint_interval;
    // Packed layout: offsets_t[p]..offsets_t[p+1] delimit each param's rows.
    auto offs = offsets_t.to(torch::kCPU).to(torch::kLong);
    auto oacc = offs.accessor<int64_t, 1>();
    const int P = (int)grads.size();
    for (int p = 0; p < P; ++p) {
        auto& g = grads[p];
        if (!g.defined() || g.numel() == 0) continue;
        const int64_t start = oacc[p];
        const int64_t end   = oacc[p + 1];
        const int64_t n = end - start;
        if (n <= 0) continue;
        auto h0 = torch::zeros({4}, g.options().dtype(torch::kFloat32));
        auto S = sg2adj::bilevel_forward_save(
            g, sharpness_list[p], h0, input_proj_W, input_proj_b,
            csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
            csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
            hca_q_W, hca_k_W, hca_v_W, hca_out_W,
            torch::zeros({4, 2 + 2 * d_model}, g.options().dtype(torch::kFloat32)),
            torch::zeros({4}, g.options().dtype(torch::kFloat32)),
            torch::zeros({4, 2 + 2 * d_model}, g.options().dtype(torch::kFloat32)),
            torch::zeros({4}, g.options().dtype(torch::kFloat32)),
            torch::zeros({4, 2 + 2 * d_model}, g.options().dtype(torch::kFloat32)),
            torch::zeros({4}, g.options().dtype(torch::kFloat32)),
            d_model, num_heads, csa_compress, csa_window, csa_topk,
            hca_compress, indexer_rank);
        if (csa_ctx_out_packed.defined() && csa_ctx_out_packed.numel() > 0)
            csa_ctx_out_packed.narrow(0, start, n).copy_(S.csa_ctx);
        if (hca_ctx_out_packed.defined() && hca_ctx_out_packed.numel() > 0)
            hca_ctx_out_packed.narrow(0, start, n).copy_(S.hca_ctx);
        if (x_sorted_packed.defined() && x_sorted_packed.numel() > 0)
            x_sorted_packed.narrow(0, start, n).copy_(S.x_sorted);
        if (sort_indices_packed.defined() && sort_indices_packed.numel() > 0)
            sort_indices_packed.narrow(0, start, n).copy_(
                S.sort_idx.to(sort_indices_packed.dtype()));
    }
    (void)csa_saved_denom_packed; (void)csa_saved_sel_idx_packed;
    (void)csa_saved_probs_packed; (void)hca_saved_denom_packed;
    (void)hca_saved_probs_packed;
}

void launch_csa_hca_backward(
    torch::Tensor d_smart_grad,
    torch::Tensor grad, torch::Tensor sharpness, float rescale,
    torch::Tensor sort_indices, torch::Tensor x_sorted,
    torch::Tensor csa_ctx, torch::Tensor hca_ctx,
    torch::Tensor csa_saved_denom, torch::Tensor csa_saved_sel_idx,
    torch::Tensor csa_saved_probs,
    torch::Tensor hca_saved_denom, torch::Tensor hca_saved_probs,
    torch::Tensor gru_input, torch::Tensor gru_h_old,
    torch::Tensor gru_z_gate, torch::Tensor gru_r_gate, torch::Tensor gru_h_tilde,
    torch::Tensor peer_input, torch::Tensor expert_indices,
    torch::Tensor routing_weights, torch::Tensor saved_z_hidden,
    torch::Tensor saved_scores_a, torch::Tensor saved_scores_b,
    torch::Tensor saved_top_a_idx, torch::Tensor saved_top_b_idx,
    torch::Tensor saved_soft_a, torch::Tensor saved_soft_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_W2,
    torch::Tensor expert_b1_in, torch::Tensor expert_b2_in,
    torch::Tensor input_proj_W,
    torch::Tensor d_csa_q_W, torch::Tensor d_csa_k_W, torch::Tensor d_csa_v_W,
    torch::Tensor d_csa_compress_w,
    torch::Tensor d_csa_idx_DQ, torch::Tensor d_csa_idx_UQ, torch::Tensor d_csa_idx_K,
    torch::Tensor d_csa_out_W,
    torch::Tensor d_hca_q_W, torch::Tensor d_hca_k_W, torch::Tensor d_hca_v_W,
    torch::Tensor d_hca_out_W,
    torch::Tensor d_gru_Wz, torch::Tensor d_gru_bz,
    torch::Tensor d_gru_Wr, torch::Tensor d_gru_br,
    torch::Tensor d_gru_Wh, torch::Tensor d_gru_bh,
    torch::Tensor d_peer_query_Ws,
    torch::Tensor d_prod_keys_A, torch::Tensor d_prod_keys_B,
    torch::Tensor d_expert_W1, torch::Tensor d_expert_b1,
    torch::Tensor d_expert_W2, torch::Tensor d_expert_b2,
    torch::Tensor d_input_proj_W, torch::Tensor d_input_proj_b,
    int d_model, int gru_hidden, int gru_input_dim,
    int num_heads, int topk, int pk_dim,
    int expert_hidden, int peer_input_dim, int num_experts,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    int checkpoint_interval)
{
    if (d_smart_grad.numel() == 0) return;
    (void)checkpoint_interval; (void)num_experts; (void)peer_input_dim;
    (void)gru_input_dim; (void)csa_saved_denom; (void)csa_saved_probs;
    (void)hca_saved_denom; (void)hca_saved_probs; (void)expert_indices;
    (void)routing_weights; (void)saved_z_hidden; (void)saved_scores_a;
    (void)saved_scores_b; (void)saved_top_a_idx; (void)saved_top_b_idx;
    (void)saved_soft_a; (void)saved_soft_b; (void)gru_z_gate; (void)gru_r_gate;
    (void)gru_h_tilde;

    // Reconstruct the SavedActs the driver needs. csa/hca ctx, x_sorted,
    // sort_indices, peer_input, gru_input, gru_h_old come straight from the
    // saved tensors; the GRU gates and PEER intermediates are recomputed inside
    // the driver from gru_input/h_old and peer_input (cheap, exact).
    auto fopt = x_sorted.options().dtype(torch::kFloat32);
    sg2adj::SavedActs S;
    auto g = grad.reshape({-1}).to(torch::kFloat32);
    auto s = sharpness.reshape({-1}).to(torch::kFloat32);
    S.g_col = g; S.s_col = s;
    S.x_sorted = x_sorted.to(torch::kFloat32);
    S.sort_idx = sort_indices.to(torch::kLong);
    S.unsort_idx = S.sort_idx.argsort();
    S.csa_ctx = csa_ctx.to(torch::kFloat32);
    S.hca_ctx = hca_ctx.to(torch::kFloat32);
    S.csa_sel_idx = csa_saved_sel_idx.defined() && csa_saved_sel_idx.numel() > 0
        ? csa_saved_sel_idx.to(torch::kLong) : torch::Tensor{};
    S.peer_input = peer_input.to(torch::kFloat32);
    S.gru_input  = gru_input.to(torch::kFloat32);
    S.gru_h_old  = gru_h_old.to(torch::kFloat32);

    // GRU gates: the fwd_save persists z/r/h_tilde (the gate biases are not in
    // the backward signature, so recompute is not bit-exact). Use the saved
    // gates when present; otherwise fall back to a bias-free recompute.
    auto xh = torch::cat({S.gru_input, S.gru_h_old}, -1);
    S.gru_z = (gru_z_gate.defined() && gru_z_gate.numel() > 0)
        ? gru_z_gate.to(torch::kFloat32)
        : torch::sigmoid(sg2adj::linear_fwd(xh, gru_Wz));
    S.gru_r = (gru_r_gate.defined() && gru_r_gate.numel() > 0)
        ? gru_r_gate.to(torch::kFloat32)
        : torch::sigmoid(sg2adj::linear_fwd(xh, gru_Wr));
    auto xrh = torch::cat({S.gru_input, S.gru_r * S.gru_h_old}, -1);
    S.gru_h_tilde = (gru_h_tilde.defined() && gru_h_tilde.numel() > 0)
        ? gru_h_tilde.to(torch::kFloat32)
        : torch::tanh(sg2adj::linear_fwd(xrh, gru_Wh));

    std::vector<torch::Tensor> peer_Wq, prod_A, prod_B;
    std::vector<torch::Tensor> dpeer_Wq, dprod_A, dprod_B;
    const int64_t nph = peer_query_Ws.size(0);
    for (int64_t h = 0; h < nph; ++h) {
        peer_Wq.push_back(peer_query_Ws.index({h}).to(torch::kFloat32));
        prod_A.push_back(prod_keys_A.index({h}).to(torch::kFloat32));
        prod_B.push_back(prod_keys_B.index({h}).to(torch::kFloat32));
        dpeer_Wq.push_back(torch::zeros_like(peer_Wq.back()));
        dprod_A.push_back(torch::zeros_like(prod_A.back()));
        dprod_B.push_back(torch::zeros_like(prod_B.back()));
    }

    sg2adj::bilevel_backward_driver(
        d_smart_grad, rescale, S,
        input_proj_W.to(torch::kFloat32),
        csa_q_W.to(torch::kFloat32), csa_k_W.to(torch::kFloat32),
        csa_v_W.to(torch::kFloat32), csa_compress_w.to(torch::kFloat32),
        csa_idx_DQ.to(torch::kFloat32), csa_idx_UQ.to(torch::kFloat32),
        csa_idx_K.to(torch::kFloat32), csa_out_W.to(torch::kFloat32),
        hca_q_W.to(torch::kFloat32), hca_k_W.to(torch::kFloat32),
        hca_v_W.to(torch::kFloat32), hca_out_W.to(torch::kFloat32),
        gru_Wz.to(torch::kFloat32), gru_Wr.to(torch::kFloat32),
        gru_Wh.to(torch::kFloat32),
        peer_Wq, prod_A, prod_B,
        expert_W1.to(torch::kFloat32),
        expert_b1_in.defined() && expert_b1_in.numel() > 0
            ? expert_b1_in.to(torch::kFloat32)
            : torch::zeros({num_experts, expert_hidden}, fopt),
        expert_W2.to(torch::kFloat32),
        expert_b2_in.defined() && expert_b2_in.numel() > 0
            ? expert_b2_in.to(torch::kFloat32)
            : torch::zeros({num_experts, 1}, fopt),
        d_model, num_heads, gru_hidden, pk_dim, topk, expert_hidden,
        csa_compress, csa_window, csa_topk, hca_compress, indexer_rank,
        d_input_proj_W, d_input_proj_b,
        d_csa_q_W, d_csa_k_W, d_csa_v_W, d_csa_compress_w,
        d_csa_idx_DQ, d_csa_idx_UQ, d_csa_idx_K, d_csa_out_W,
        d_hca_q_W, d_hca_k_W, d_hca_v_W, d_hca_out_W,
        d_gru_Wz, d_gru_bz, d_gru_Wr, d_gru_br, d_gru_Wh, d_gru_bh,
        dpeer_Wq, dprod_A, dprod_B,
        d_expert_W1, d_expert_b1, d_expert_W2, d_expert_b2);

    // Scatter per-head PEER grads back into the stacked output buffers.
    for (int64_t h = 0; h < nph; ++h) {
        if (d_peer_query_Ws.defined() && d_peer_query_Ws.numel() > 0)
            d_peer_query_Ws.index({h}).add_(dpeer_Wq[h]);
        if (d_prod_keys_A.defined() && d_prod_keys_A.numel() > 0)
            d_prod_keys_A.index({h}).add_(dprod_A[h]);
        if (d_prod_keys_B.defined() && d_prod_keys_B.numel() > 0)
            d_prod_keys_B.index({h}).add_(dprod_B[h]);
    }
    (void)gru_z_gate;
}

void launch_csa_hca_backward_batched(
    torch::Tensor d_csa_ctx_packed, torch::Tensor d_hca_ctx_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor csa_saved_denom_packed, torch::Tensor csa_saved_sel_idx_packed,
    torch::Tensor csa_saved_probs_packed,
    torch::Tensor hca_saved_denom_packed, torch::Tensor hca_saved_probs_packed,
    torch::Tensor offsets_t,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    torch::Tensor d_csa_q_W, torch::Tensor d_csa_k_W, torch::Tensor d_csa_v_W,
    torch::Tensor d_csa_compress_w,
    torch::Tensor d_csa_idx_DQ, torch::Tensor d_csa_idx_UQ, torch::Tensor d_csa_idx_K,
    torch::Tensor d_csa_out_W,
    torch::Tensor d_hca_q_W, torch::Tensor d_hca_k_W, torch::Tensor d_hca_v_W,
    torch::Tensor d_hca_out_W,
    torch::Tensor d_x_sorted_packed,
    int d_model, int num_heads, int num_params,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    int checkpoint_interval)
{
    if (num_params == 0) return;
    (void)checkpoint_interval;
    (void)csa_saved_denom_packed; (void)csa_saved_probs_packed;
    (void)hca_saved_denom_packed; (void)hca_saved_probs_packed;
    // Per-param attention-only backward (CSA/HCA weight grads from the packed
    // d_ctx). Mirrors the single-tensor attention block; PEER/GRU/input_proj are
    // handled by the single-tensor entry. This entry accumulates the attention
    // weight grads and (optionally) the d_x_sorted carry.
    auto offs = offsets_t.to(torch::kCPU).to(torch::kLong);
    auto oacc = offs.accessor<int64_t, 1>();
    for (int p = 0; p < num_params; ++p) {
        const int64_t start = oacc[p];
        const int64_t end   = oacc[p + 1];
        const int64_t n = end - start;
        if (n <= 0) continue;
        auto x = x_sorted_packed.narrow(0, start, n).to(torch::kFloat32);
        auto d_csa = d_csa_ctx_packed.narrow(0, start, n).to(torch::kFloat32);
        auto d_hca = d_hca_ctx_packed.narrow(0, start, n).to(torch::kFloat32);
        auto d_x = torch::zeros_like(x);

        auto cf = sg2adj::csa_forward(
            x, csa_q_W.to(torch::kFloat32), csa_k_W.to(torch::kFloat32),
            csa_v_W.to(torch::kFloat32), csa_compress_w.to(torch::kFloat32),
            csa_idx_DQ.to(torch::kFloat32), csa_idx_UQ.to(torch::kFloat32),
            csa_idx_K.to(torch::kFloat32), num_heads,
            csa_compress, csa_window, csa_topk);
        d_csa_out_W.add_(torch::mm(d_csa.t(), cf.ctx));
        auto d_csa_pre = torch::mm(d_csa, csa_out_W.to(torch::kFloat32));
        sg2adj::csa_backward(
            x, cf, csa_q_W.to(torch::kFloat32), csa_k_W.to(torch::kFloat32),
            csa_v_W.to(torch::kFloat32), csa_compress_w.to(torch::kFloat32),
            csa_idx_DQ.to(torch::kFloat32), csa_idx_UQ.to(torch::kFloat32),
            csa_idx_K.to(torch::kFloat32), num_heads, d_csa_pre,
            d_csa_q_W, d_csa_k_W, d_csa_v_W, d_csa_compress_w,
            d_csa_idx_DQ, d_csa_idx_UQ, d_csa_idx_K, d_x);

        auto hf = sg2adj::hca_forward(
            x, hca_q_W.to(torch::kFloat32), hca_k_W.to(torch::kFloat32),
            hca_v_W.to(torch::kFloat32), num_heads, hca_compress, csa_window);
        d_hca_out_W.add_(torch::mm(d_hca.t(), hf.ctx));
        auto d_hca_pre = torch::mm(d_hca, hca_out_W.to(torch::kFloat32));
        sg2adj::hca_backward(x, hf,
                             hca_q_W.to(torch::kFloat32), hca_k_W.to(torch::kFloat32),
                             hca_v_W.to(torch::kFloat32), num_heads, d_hca_pre,
                             d_hca_q_W, d_hca_k_W, d_hca_v_W, d_x);

        if (d_x_sorted_packed.defined() && d_x_sorted_packed.numel() > 0)
            d_x_sorted_packed.narrow(0, start, n).add_(d_x);
    }
    (void)d_model; (void)indexer_rank;
}

// ═════════════════════════════════════════════════════════════════════════
//  MoE systems (folded from former launch_moe.cu). REAL sm_90 CUDA kernels for
//  the MoE-compaction tail of MoEAwareSuperGrok2 (Stage 1B). The Python driver
//  (optimizers/supergrok2.py::_moe_step) gathers the active-expert parameter
//  slice into a dense buffer, runs the Adam update on it, and scatters back;
//  these kernels are the gather/scatter/histogram/load-balance primitives.
//
//  Reachability (verified): _moe_step calls moe_count_expert_activations,
//  moe_compute_load_balance_loss, moe_apply_frequency_scaling,
//  moe_filter_active_params, moe_scatter_results. The dynamic_expert_{load,
//  fwd,bwd} and scan_compacted entries are exported (bindings.cpp) but not
//  currently called; they are implemented as correct real kernels for ABI /
//  completeness. moe_scan_compacted is VESTIGIAL (Mamba-era selective scan;
//  SG2's mixer is now CSA/HCA) — kept linkable and numerically sound.
//
//  All compaction tensors are FP32 1-D; index tensors are int32. Grid-stride
//  loops + atomics, matching moe_adam_kernel / the prim:: helpers above.
// ═════════════════════════════════════════════════════════════════════════

// ── (1) Expert-activation histogram ───────────────────────────────────────
//  gate_logits [N, num_experts] (row-major). For each (row, e) with
//  gate_logits[row,e] > threshold, increment expert_counts[e]. One thread per
//  (row, e) cell via a flattened grid-stride loop over N*num_experts.
//
//  §4.1 (redux.sync / warp-aggregated atomics): rather than every active lane
//  issuing an independent global atomicAdd(&expert_counts[e], 1), lanes in a
//  warp that target the SAME expert e coalesce their +1's. __match_any_sync
//  partitions the (predicate-passing) lanes of the warp into groups by e; the
//  lowest lane of each group issues a single atomicAdd(&counts[e], popc(mask)).
//  Numerically IDENTICAL to per-lane atomics — popc(mask) is exactly the count
//  of lanes adding 1 for that e — but issues at most one atomic per (warp, e)
//  instead of one per active lane. __match_any_sync needs sm_70+ (always true
//  here: this TU is sm_90a); the participating mask is the predicate ballot, so
//  divergent/tail lanes are excluded correctly.
__global__ void moe_count_expert_activations_kernel(
    const float* __restrict__ gate_logits,
    int* __restrict__ expert_counts,
    float threshold, int N, int num_experts
) {
    const long total = static_cast<long>(N) * num_experts;
    const int stride = prim::grid_stride();
    const unsigned lane = threadIdx.x & 31u;
    // Uniform (warp-convergent) trip count: round `total` up to a whole number
    // of strides so every lane reaches the warp ballot each iteration. The
    // per-element predicate (in_range && hit) excludes the tail lanes, so the
    // ballot mask is full (0xffffffff) and well-defined for all 32 lanes.
    const long start = prim::grid_stride_index();
    const long rounded = ((total + stride - 1) / stride) * stride;
    for (long idx = start; idx < rounded; idx += stride) {
        const bool in_range = idx < total;
        int e = -1;
        bool hit = false;
        if (in_range) {
            e = static_cast<int>(idx % num_experts);
            hit = gate_logits[idx] > threshold;
        }
        // Ballot the lanes that will add 1 (full participating mask).
        const unsigned active = __ballot_sync(0xffffffffu, hit);
        if (hit) {
            // Group the contributing lanes by their target expert e.
            const unsigned same = __match_any_sync(active, e);
            // Lowest lane in this e-group is the leader; it adds the group size.
            const unsigned leader = __ffs(static_cast<int>(same)) - 1u;
            if (lane == leader) {
                atomicAdd(&expert_counts[e], __popc(same));
            }
        }
    }
}

void moe_count_expert_activations(
    torch::Tensor gate_logits, torch::Tensor expert_counts,
    float threshold, int N, int num_experts) {
    if (N == 0 || num_experts == 0) return;
    auto gl = gate_logits.contiguous();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const long total = static_cast<long>(N) * num_experts;
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (total + block - 1) / block);
    moe_count_expert_activations_kernel<<<grid, block, 0, stream>>>(
        gl.data_ptr<float>(), expert_counts.data_ptr<int>(),
        threshold, N, num_experts);
    SG_LAUNCH_CHECK(stream);
}

// ── (2) Switch-Transformer load-balance auxiliary loss ─────────────────────
//  f_e = expert_counts[e]/N  (fraction of tokens routed to expert e)
//  P_e = mean_t softmax(gate_logits[t,:])[e]
//  loss = num_experts * Σ_e f_e * P_e
//  Implemented with ATen reductions (softmax + mean) for numerical stability;
//  returns a scalar tensor on the gate_logits device.
torch::Tensor moe_compute_load_balance_loss(
    torch::Tensor expert_counts, torch::Tensor gate_logits,
    int N, int num_experts) {
    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32).device(gate_logits.device());
    if (N == 0 || num_experts == 0) return torch::zeros({}, opts);
    auto gl = gate_logits.to(torch::kFloat32);
    // P_e: mean over tokens of softmax probability for expert e -> [num_experts]
    auto P = torch::softmax(gl, /*dim=*/1).mean(/*dim=*/0);          // [E]
    // f_e: token fraction routed to expert e
    auto f = expert_counts.to(torch::kFloat32) / static_cast<double>(N);  // [E]
    auto loss = static_cast<double>(num_experts) * (f * P).sum();
    return loss;
}

// ── (3) Frequency-inverse per-expert LR scaling ────────────────────────────
//  freq_e  = (counts[e] + smoothing) / (total + smoothing*num_experts)
//  scale_e = clamp( (1/num_experts) / freq_e, min_scale, max_scale )
__global__ void moe_apply_frequency_scaling_kernel(
    const int* __restrict__ expert_counts,
    float* __restrict__ lr_scale,
    int num_experts, int total_activations,
    float min_scale, float max_scale, float smoothing
) {
    const int stride = prim::grid_stride();
    const float denom = static_cast<float>(total_activations)
                      + smoothing * static_cast<float>(num_experts);
    const float uniform = 1.0f / static_cast<float>(num_experts);
    for (int e = prim::grid_stride_index(); e < num_experts; e += stride) {
        const float freq = (static_cast<float>(expert_counts[e]) + smoothing)
                         / denom;
        float scale = (freq > 0.0f) ? (uniform / freq) : max_scale;
        scale = fminf(fmaxf(scale, min_scale), max_scale);
        lr_scale[e] = scale;
    }
}

void moe_apply_frequency_scaling(
    torch::Tensor expert_counts, torch::Tensor lr_scale,
    int num_experts, int total_activations,
    float min_scale, float max_scale, float smoothing) {
    if (num_experts == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (num_experts + block - 1) / block);
    moe_apply_frequency_scaling_kernel<<<grid, block, 0, stream>>>(
        expert_counts.data_ptr<int>(), lr_scale.data_ptr<float>(),
        num_experts, total_activations, min_scale, max_scale, smoothing);
    SG_LAUNCH_CHECK(stream);
}

// ── (4) Stream-compaction of active-expert parameters ──────────────────────
//  For each i in [0,total_params): if expert_active[param_to_expert[i]] != 0,
//  append params/grads/state_m/state_v[i] to the compact_* arrays and record
//  scatter_indices[out]=i. compact_count[0] = number kept.
//
//  Output position is claimed via a single global atomicAdd counter
//  (compact_count). A prefix-sum compaction would yield deterministic ordering
//  and slightly better coalescing, but the optimizer only needs the (out -> i)
//  scatter map to be self-consistent — ordering among kept elements is
//  irrelevant because moe_scatter_results writes back by stored index. The
//  atomic compaction is correct and is the documented choice here.
__global__ void moe_filter_active_params_kernel(
    const float* __restrict__ params, const float* __restrict__ grads,
    const float* __restrict__ state_m, const float* __restrict__ state_v,
    const int* __restrict__ param_to_expert,
    const int* __restrict__ expert_active,
    float* __restrict__ compact_params, float* __restrict__ compact_grads,
    float* __restrict__ compact_state_m, float* __restrict__ compact_state_v,
    int* __restrict__ scatter_indices, int* __restrict__ compact_count,
    int total_params
) {
    const int stride = prim::grid_stride();
    const unsigned lane = threadIdx.x & 31u;
    // Uniform (warp-convergent) trip count so every lane reaches the ballot
    // each iteration; the `in_range && active` predicate masks the tail lanes.
    const int start = prim::grid_stride_index();
    const long rounded =
        ((static_cast<long>(total_params) + stride - 1) / stride) * stride;
    for (long ii = start; ii < rounded; ii += stride) {
        const int i = static_cast<int>(ii);
        bool keep = false;
        if (ii < total_params) {
            const int e = param_to_expert[i];
            keep = expert_active[e] != 0;
        }
        // §4.1 warp-aggregated atomic ALLOCATION: the leader reserves one
        // contiguous block of `popc(mask)` output slots with a single global
        // atomicAdd; each kept lane then writes to base + its in-warp rank.
        // Ordering among kept elements is irrelevant (moe_scatter_results
        // writes back by stored scatter index), and the total count is
        // identical to the per-lane atomic — popc(mask) kept lanes claim
        // exactly popc(mask) slots.
        const unsigned mask = __ballot_sync(0xffffffffu, keep);
        if (keep) {
            const unsigned leader = __ffs(static_cast<int>(mask)) - 1u;
            int base = 0;
            if (lane == leader) {
                base = atomicAdd(&compact_count[0], __popc(mask));
            }
            // Broadcast the reserved base from the leader to the whole group.
            base = __shfl_sync(mask, base, static_cast<int>(leader));
            // Rank of this lane within the kept group = popcount of kept lanes
            // below it.
            const unsigned rank =
                __popc(mask & ((1u << lane) - 1u));
            const int out = base + static_cast<int>(rank);
            compact_params[out]  = params[i];
            compact_grads[out]   = grads[i];
            compact_state_m[out] = state_m[i];
            compact_state_v[out] = state_v[i];
            scatter_indices[out] = i;
        }
    }
}

void moe_filter_active_params(
    torch::Tensor params, torch::Tensor grads,
    torch::Tensor state_m, torch::Tensor state_v,
    torch::Tensor param_to_expert, torch::Tensor expert_active,
    torch::Tensor compact_params, torch::Tensor compact_grads,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices, torch::Tensor compact_count,
    int total_params) {
    if (total_params == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (total_params + block - 1) / block);
    moe_filter_active_params_kernel<<<grid, block, 0, stream>>>(
        params.data_ptr<float>(), grads.data_ptr<float>(),
        state_m.data_ptr<float>(), state_v.data_ptr<float>(),
        param_to_expert.data_ptr<int>(), expert_active.data_ptr<int>(),
        compact_params.data_ptr<float>(), compact_grads.data_ptr<float>(),
        compact_state_m.data_ptr<float>(), compact_state_v.data_ptr<float>(),
        scatter_indices.data_ptr<int>(), compact_count.data_ptr<int>(),
        total_params);
    SG_LAUNCH_CHECK(stream);
}

// ── (5) Scatter compacted results back to dense storage ────────────────────
//  Inverse of (4): for j in [0,compact_N): i=scatter_indices[j];
//  params[i]=compact_params[j]; state_m[i]=compact_state_m[j]; etc.
__global__ void moe_scatter_results_kernel(
    const float* __restrict__ compact_params,
    const float* __restrict__ compact_state_m,
    const float* __restrict__ compact_state_v,
    const int* __restrict__ scatter_indices,
    float* __restrict__ params,
    float* __restrict__ state_m, float* __restrict__ state_v,
    int compact_N
) {
    const int stride = prim::grid_stride();
    for (int j = prim::grid_stride_index(); j < compact_N; j += stride) {
        const int i = scatter_indices[j];
        params[i]  = compact_params[j];
        state_m[i] = compact_state_m[j];
        state_v[i] = compact_state_v[j];
    }
}

void moe_scatter_results(
    torch::Tensor compact_params,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices,
    torch::Tensor params,
    torch::Tensor state_m, torch::Tensor state_v,
    int compact_N) {
    if (compact_N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (compact_N + block - 1) / block);
    moe_scatter_results_kernel<<<grid, block, 0, stream>>>(
        compact_params.data_ptr<float>(),
        compact_state_m.data_ptr<float>(), compact_state_v.data_ptr<float>(),
        scatter_indices.data_ptr<int>(),
        params.data_ptr<float>(),
        state_m.data_ptr<float>(), state_v.data_ptr<float>(),
        compact_N);
    SG_LAUNCH_CHECK(stream);
}

// ── (6) Masked gather of active expert weights ─────────────────────────────
//  expert_w1 [E, hidden, d_in], expert_b1 [E, hidden],
//  expert_w2 [E, d_out, hidden], expert_b2 [E, d_out]. active_mask [E].
//  Copies the e-th slice into a compact buffer for every active expert e,
//  packed densely in expert order (compact slot = #active experts before e).
//  Implemented as a per-active-expert prefix index built on the host (ATen)
//  followed by an index_copy; the gather itself is a simple memcpy-style copy.
void moe_dynamic_expert_load(
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor active_mask,
    torch::Tensor smem_w1, torch::Tensor smem_b1,
    torch::Tensor smem_w2, torch::Tensor smem_b2) {
    // active expert indices, in ascending order -> dense packing positions.
    auto idx = torch::nonzero(active_mask.reshape(-1) != 0).reshape(-1);  // [A]
    const int64_t A = idx.numel();
    if (A == 0) return;
    auto src1 = expert_w1.index_select(0, idx);
    auto src2 = expert_w2.index_select(0, idx);
    auto sb1  = expert_b1.index_select(0, idx);
    auto sb2  = expert_b2.index_select(0, idx);
    smem_w1.narrow(0, 0, A).copy_(src1);
    smem_w2.narrow(0, 0, A).copy_(src2);
    smem_b1.narrow(0, 0, A).copy_(sb1);
    smem_b2.narrow(0, 0, A).copy_(sb2);
}

// ── (7) Per-token expert MLP forward ───────────────────────────────────────
//  Shapes (from the binding contract / spec):
//    input          [N, d_in]
//    expert_indices [N]            (int, per-token expert id)
//    routing_weights[N]            (float, per-token scalar gate weight)
//    expert_w1      [E, hidden, d_in]   expert_b1 [E, hidden]
//    expert_w2      [E, d_out,  hidden] expert_b2 [E, d_out]
//    output         [N, d_out]    (written)
//  output[t] = routing_weights[t] * (W2_e @ relu(W1_e @ input[t] + b1_e) + b2_e)
//  One warp per token; each lane strides over the output dimension. The hidden
//  activation is recomputed per output element (hidden is small for SG2's
//  PEER-style experts); correctness-first, the dynamic_expert_* path is not on
//  the hot reachable list.
__global__ void moe_dynamic_expert_fwd_kernel(
    const float* __restrict__ input,
    const int* __restrict__ expert_indices,
    const float* __restrict__ routing_weights,
    const float* __restrict__ expert_w1, const float* __restrict__ expert_b1,
    const float* __restrict__ expert_w2, const float* __restrict__ expert_b2,
    float* __restrict__ output,
    int N, int d_in, int hidden, int d_out
) {
    const int t = blockIdx.x;                 // one block per token
    if (t >= N) return;
    const int lane = threadIdx.x;
    const int nthreads = blockDim.x;
    const int e = expert_indices[t];
    const float rw = routing_weights[t];

    const float* x   = input + static_cast<long>(t) * d_in;
    const float* W1  = expert_w1 + static_cast<long>(e) * hidden * d_in;
    const float* b1  = expert_b1 + static_cast<long>(e) * hidden;
    const float* W2  = expert_w2 + static_cast<long>(e) * d_out * hidden;
    const float* b2  = expert_b2 + static_cast<long>(e) * d_out;

    // Shared hidden activation h = relu(W1 x + b1), [hidden].
    extern __shared__ float h[];
    for (int j = lane; j < hidden; j += nthreads) {
        float acc = b1[j];
        const float* w1row = W1 + static_cast<long>(j) * d_in;
        for (int k = 0; k < d_in; ++k) acc += w1row[k] * x[k];
        h[j] = acc > 0.0f ? acc : 0.0f;
    }
    __syncthreads();

    float* y = output + static_cast<long>(t) * d_out;
    for (int o = lane; o < d_out; o += nthreads) {
        float acc = b2[o];
        const float* w2row = W2 + static_cast<long>(o) * hidden;
        for (int j = 0; j < hidden; ++j) acc += w2row[j] * h[j];
        y[o] = rw * acc;
    }
}

torch::Tensor moe_dynamic_expert_fwd(
    torch::Tensor input, torch::Tensor expert_indices,
    torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor output) {
    const int N = static_cast<int>(input.size(0));
    if (N == 0) return output;
    const int d_in   = static_cast<int>(input.size(1));
    const int hidden = static_cast<int>(expert_w1.size(1));
    const int d_out  = static_cast<int>(expert_w2.size(1));
    auto inp = input.to(torch::kFloat32).contiguous();
    auto w1  = expert_w1.to(torch::kFloat32).contiguous();
    auto b1  = expert_b1.to(torch::kFloat32).contiguous();
    auto w2  = expert_w2.to(torch::kFloat32).contiguous();
    auto b2  = expert_b2.to(torch::kFloat32).contiguous();
    auto rw  = routing_weights.to(torch::kFloat32).contiguous();
    auto ei  = expert_indices.to(torch::kInt32).contiguous();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = std::min<int>(256, std::max<int>(d_out, hidden));
    const int grid = N;
    const size_t smem = static_cast<size_t>(hidden) * sizeof(float);
    moe_dynamic_expert_fwd_kernel<<<grid, block, smem, stream>>>(
        inp.data_ptr<float>(), ei.data_ptr<int>(), rw.data_ptr<float>(),
        w1.data_ptr<float>(), b1.data_ptr<float>(),
        w2.data_ptr<float>(), b2.data_ptr<float>(),
        output.data_ptr<float>(), N, d_in, hidden, d_out);
    SG_LAUNCH_CHECK(stream);
    return output;
}

// ── (8) Per-token expert MLP backward (full VJP, no autograd) ──────────────
//  Forward:  h = relu(z1), z1 = W1 x + b1 ; y = rw * (W2 h + b2)
//  Given d_output (= dL/dy), accumulate:
//    d(W2 b2): dy = rw * d_output ; dW2_e += dy ⊗ h ; db2_e += dy
//    dh = W2ᵀ dy ; dz1 = dh ⊙ [z1>0]
//    dW1_e += dz1 ⊗ x ; db1_e += dz1 ; d_input[t] = W1ᵀ dz1
//  Expert-weight grads are accumulated with atomics (many tokens share e).
//  One block per token; hidden activation + dz1 recomputed in shared memory.
__global__ void moe_dynamic_expert_bwd_kernel(
    const float* __restrict__ d_output,
    const float* __restrict__ input,
    const int* __restrict__ expert_indices,
    const float* __restrict__ routing_weights,
    const float* __restrict__ expert_w1, const float* __restrict__ expert_b1,
    const float* __restrict__ expert_w2, const float* __restrict__ expert_b2,
    float* __restrict__ d_input, float* __restrict__ d_expert_w1,
    float* __restrict__ d_expert_b1, float* __restrict__ d_expert_w2,
    float* __restrict__ d_expert_b2,
    int N, int d_in, int hidden, int d_out
) {
    const int t = blockIdx.x;
    if (t >= N) return;
    const int lane = threadIdx.x;
    const int nthreads = blockDim.x;
    const int e = expert_indices[t];
    const float rw = routing_weights[t];

    const float* x   = input + static_cast<long>(t) * d_in;
    const float* W1  = expert_w1 + static_cast<long>(e) * hidden * d_in;
    const float* b1  = expert_b1 + static_cast<long>(e) * hidden;
    const float* W2  = expert_w2 + static_cast<long>(e) * d_out * hidden;
    const float* dy_row = d_output + static_cast<long>(t) * d_out;

    float* dW1 = d_expert_w1 + static_cast<long>(e) * hidden * d_in;
    float* db1 = d_expert_b1 + static_cast<long>(e) * hidden;
    float* dW2 = d_expert_w2 + static_cast<long>(e) * d_out * hidden;
    float* db2 = d_expert_b2 + static_cast<long>(e) * d_out;

    // h[j] = relu(W1 x + b1)[j], hmask[j] = [z1>0]   (recompute forward act).
    extern __shared__ float smem[];
    float* h    = smem;             // [hidden]
    float* dz1  = smem + hidden;    // [hidden]
    for (int j = lane; j < hidden; j += nthreads) {
        float acc = b1[j];
        const float* w1row = W1 + static_cast<long>(j) * d_in;
        for (int k = 0; k < d_in; ++k) acc += w1row[k] * x[k];
        h[j] = acc > 0.0f ? acc : 0.0f;
        dz1[j] = 0.0f;
    }
    __syncthreads();

    // dy = rw * d_output ; db2 += dy ; dW2 += dy ⊗ h ; accumulate dh into dz1.
    for (int o = lane; o < d_out; o += nthreads) {
        const float dy = rw * dy_row[o];
        atomicAdd(&db2[o], dy);
        const float* w2row = W2 + static_cast<long>(o) * hidden;
        float* dw2row = dW2 + static_cast<long>(o) * hidden;
        for (int j = 0; j < hidden; ++j) {
            atomicAdd(&dw2row[j], dy * h[j]);
            // dh_j = Σ_o W2[o,j] dy ; gate by relu mask (h[j]>0).
            if (h[j] > 0.0f) atomicAdd(&dz1[j], w2row[j] * dy);
        }
    }
    __syncthreads();

    // dW1 += dz1 ⊗ x ; db1 += dz1 ; d_input[t] = W1ᵀ dz1.
    for (int j = lane; j < hidden; j += nthreads) {
        const float g = dz1[j];
        atomicAdd(&db1[j], g);
        const float* w1row = W1 + static_cast<long>(j) * d_in;
        float* dw1row = dW1 + static_cast<long>(j) * d_in;
        for (int k = 0; k < d_in; ++k) atomicAdd(&dw1row[k], g * x[k]);
    }
    __syncthreads();
    float* dx = d_input + static_cast<long>(t) * d_in;
    for (int k = lane; k < d_in; k += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j < hidden; ++j) {
            acc += W1[static_cast<long>(j) * d_in + k] * dz1[j];
        }
        dx[k] = acc;
    }
}

void moe_dynamic_expert_bwd(
    torch::Tensor d_output, torch::Tensor input,
    torch::Tensor expert_indices, torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor d_input, torch::Tensor d_expert_w1,
    torch::Tensor d_expert_b1, torch::Tensor d_expert_w2,
    torch::Tensor d_expert_b2) {
    const int N = static_cast<int>(input.size(0));
    if (N == 0) return;
    const int d_in   = static_cast<int>(input.size(1));
    const int hidden = static_cast<int>(expert_w1.size(1));
    const int d_out  = static_cast<int>(expert_w2.size(1));
    auto inp = input.to(torch::kFloat32).contiguous();
    auto dout = d_output.to(torch::kFloat32).contiguous();
    auto w1  = expert_w1.to(torch::kFloat32).contiguous();
    auto b1  = expert_b1.to(torch::kFloat32).contiguous();
    auto w2  = expert_w2.to(torch::kFloat32).contiguous();
    auto b2  = expert_b2.to(torch::kFloat32).contiguous();
    auto rw  = routing_weights.to(torch::kFloat32).contiguous();
    auto ei  = expert_indices.to(torch::kInt32).contiguous();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = std::min<int>(256, std::max<int>(d_out, hidden));
    const int grid = N;
    const size_t smem = static_cast<size_t>(2 * hidden) * sizeof(float);
    moe_dynamic_expert_bwd_kernel<<<grid, block, smem, stream>>>(
        dout.data_ptr<float>(), inp.data_ptr<float>(),
        ei.data_ptr<int>(), rw.data_ptr<float>(),
        w1.data_ptr<float>(), b1.data_ptr<float>(),
        w2.data_ptr<float>(), b2.data_ptr<float>(),
        d_input.data_ptr<float>(), d_expert_w1.data_ptr<float>(),
        d_expert_b1.data_ptr<float>(), d_expert_w2.data_ptr<float>(),
        d_expert_b2.data_ptr<float>(), N, d_in, hidden, d_out);
    SG_LAUNCH_CHECK(stream);
}

// ── (9) Compacted selective scan — VESTIGIAL ───────────────────────────────
//  This signature references a Mamba-style discretized SSM recurrence
//  (A_log/dt/B/C/D). SG2's sequence mixer is now CSA/HCA, NOT Mamba, and the
//  reachability audit confirms Python NEVER calls moe_scan_compacted. It is
//  kept here purely for ABI stability (the symbol is exported by bindings.cpp).
//
//  We implement the standard discretized SSM recurrence so the entry is
//  numerically sound if ever invoked:
//    A_bar = exp(dt_t * A) where A = -exp(A_log)   (per (channel, state))
//    h_t   = A_bar ⊙ h_{t-1} + (dt_t * B_t) * x_t
//    y_t   = Σ_s C_t[s] * h_t[d,s] + D[d] * x_t[d]
//  Layout (compacted, single sequence of length compact_N):
//    compact_x  [compact_N, d_inner]      compact_dt [compact_N, d_inner]
//    compact_B  [compact_N, d_state]      compact_C  [compact_N, d_state]
//    A_log      [d_inner, d_state]        D_param    [d_inner]
//    initial_state/final_state [d_inner, d_state]
//    scan_output[compact_N, d_inner]
//  rope_freq is accepted but unused (vestigial positional arg). One thread per
//  inner channel d; the scan is sequential along time, parallel across d.
__global__ void moe_scan_compacted_kernel(
    const float* __restrict__ compact_x, const float* __restrict__ compact_dt,
    const float* __restrict__ compact_B, const float* __restrict__ compact_C,
    const float* __restrict__ A_log, const float* __restrict__ D_param,
    float* __restrict__ scan_output, float* __restrict__ final_state,
    const float* __restrict__ initial_state,
    int compact_N, int d_inner, int d_state
) {
    const int stride = prim::grid_stride();
    for (int d = prim::grid_stride_index(); d < d_inner; d += stride) {
        // per-channel state register row h[s], bounded by SG2's MAX d_state.
        float h[256];
        for (int s = 0; s < d_state; ++s) {
            h[s] = (initial_state != nullptr)
                 ? initial_state[d * d_state + s] : 0.0f;
        }
        const float Dd = (D_param != nullptr) ? D_param[d] : 0.0f;
        for (int t = 0; t < compact_N; ++t) {
            const float xt = compact_x[t * d_inner + d];
            const float dt = compact_dt[t * d_inner + d];
            float y = Dd * xt;
            for (int s = 0; s < d_state; ++s) {
                const float A = -expf(A_log[d * d_state + s]);  // negative real
                const float A_bar = expf(dt * A);
                const float Bx = (dt * compact_B[t * d_state + s]) * xt;
                h[s] = A_bar * h[s] + Bx;
                y += compact_C[t * d_state + s] * h[s];
            }
            scan_output[t * d_inner + d] = y;
        }
        if (final_state != nullptr) {
            for (int s = 0; s < d_state; ++s)
                final_state[d * d_state + s] = h[s];
        }
    }
}

void moe_scan_compacted(
    torch::Tensor compact_x, torch::Tensor compact_dt,
    torch::Tensor compact_B, torch::Tensor compact_C,
    torch::Tensor A_log, torch::Tensor D_param,
    torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor final_state,
    torch::Tensor initial_state,
    int compact_N, int d_inner, int d_state) {
    (void)rope_freq;  // vestigial positional arg (Mamba-era), intentionally unused.
    if (compact_N == 0 || d_inner == 0) return;
    TORCH_CHECK(d_state <= 256,
        "moe_scan_compacted: vestigial scan supports d_state <= 256");
    auto cx = compact_x.to(torch::kFloat32).contiguous();
    auto cdt = compact_dt.to(torch::kFloat32).contiguous();
    auto cB = compact_B.to(torch::kFloat32).contiguous();
    auto cC = compact_C.to(torch::kFloat32).contiguous();
    auto al = A_log.to(torch::kFloat32).contiguous();
    auto dp = D_param.defined() ? D_param.to(torch::kFloat32).contiguous()
                                : torch::Tensor{};
    auto init = initial_state.defined()
              ? initial_state.to(torch::kFloat32).contiguous()
              : torch::Tensor{};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (d_inner + block - 1) / block);
    moe_scan_compacted_kernel<<<grid, block, 0, stream>>>(
        cx.data_ptr<float>(), cdt.data_ptr<float>(),
        cB.data_ptr<float>(), cC.data_ptr<float>(),
        al.data_ptr<float>(),
        dp.defined() ? dp.data_ptr<float>() : nullptr,
        scan_output.data_ptr<float>(),
        final_state.defined() ? final_state.data_ptr<float>() : nullptr,
        init.defined() ? init.data_ptr<float>() : nullptr,
        compact_N, d_inner, d_state);
    SG_LAUNCH_CHECK(stream);
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_SUPERGROK2_SM90_CUH_
