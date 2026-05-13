// csrc/kernels/cuda/sm_90/models/attention.cuh
// Shared attention kernel for sm_90 (Hopper). Serves both Decoder (causal,
// seq_len=4) and ViT (non-causal, seq_len=17) via the kCausal template flag.
//
// At these tiny sequence lengths FlashAttention's block-wise tiling adds
// overhead with no benefit. Instead we compute the full QK^T score matrix
// in SMEM/registers, run softmax in-place, then multiply by V.
//
// BF16 activations, FP32 accumulation for matmuls.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace sm90 { namespace models { namespace attention {

// ── Launch configuration descriptor ────────────────────────────────────
template <typename ActT, int kHeadDim, bool kCausal>
struct AttentionLaunchConfig {
    int block;
    int cluster_size;
    int smem_bytes;
    bool use_fa3;
    bool use_cutlass_fmha;
};

// ── Forward declaration for external backends ──────────────────────────
#ifdef WITH_FLASH_ATTN_3
template <typename ActT, int kHeadDim, bool kCausal>
cudaError_t flash_attn3_forward(
    const ActT* q, const ActT* k, const ActT* v, ActT* out, ActT* softmax_lse,
    int batch, int n_heads, int seq_len, float scale, cudaStream_t stream);
#endif

#ifdef WITH_CUTLASS
template <typename ActT, int kHeadDim, bool kCausal>
cudaError_t cutlass_fmha_forward(
    const ActT* q, const ActT* k, const ActT* v, ActT* out, ActT* softmax_lse,
    int batch, int n_heads, int seq_len, float scale, cudaStream_t stream);
#endif

// ── SMEM-based attention kernel (default for tiny seq_len) ─────────────
// One block per (batch, head). Each thread cooperates on the matmul.
// Max seq_len supported: 32 (17x17 scores = 289 floats in SMEM).

template <typename ActT, int kHeadDim, bool kCausal>
__global__ void __launch_bounds__(128, 4)
smem_attention_fwd_kernel(
    const ActT* __restrict__ q,     // [B, H, N, D]
    const ActT* __restrict__ k,
    const ActT* __restrict__ v,
    ActT* __restrict__ out,
    float* __restrict__ softmax_lse, // [B, H, N] or nullptr
    int seq_len, float scale
) {
    const int bh = blockIdx.x;              // flattened (batch, head) index
    const int tid = threadIdx.x;
    const int N = seq_len;
    const int D = kHeadDim;
    const int base = bh * N * D;

    // Shared memory: scores[N][N] + row_max[N] + row_sum[N]
    extern __shared__ float smem[];
    float* scores  = smem;                  // N * N
    float* row_max = scores + N * N;        // N
    float* row_sum = row_max + N;           // N

    // Step 1: Compute S = Q K^T * scale, with optional causal mask
    for (int idx = tid; idx < N * N; idx += blockDim.x) {
        int i = idx / N;  // query row
        int j = idx % N;  // key column
        if constexpr (kCausal) {
            if (j > i) { scores[idx] = -1e9f; continue; }
        }
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += static_cast<float>(q[base + i * D + d])
                 * static_cast<float>(k[base + j * D + d]);
        }
        scores[idx] = dot * scale;
    }
    __syncthreads();

    // Step 2: Row-wise softmax (online stable via max subtraction)
    for (int i = tid; i < N; i += blockDim.x) {
        float m = -1e9f;
        for (int j = 0; j < N; j++) m = fmaxf(m, scores[i * N + j]);
        row_max[i] = m;
        float s = 0.0f;
        for (int j = 0; j < N; j++) {
            float e = ptx_expf(scores[i * N + j] - m);
            scores[i * N + j] = e;
            s += e;
        }
        float inv_s = 1.0f / fmaxf(s, 1e-12f);
        for (int j = 0; j < N; j++) scores[i * N + j] *= inv_s;
        row_sum[i] = s;  // keep for lse
        if (softmax_lse != nullptr)
            softmax_lse[bh * N + i] = m + logf(fmaxf(s, 1e-12f));
    }
    __syncthreads();

    // Step 3: Out = Softmax(S) * V
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int i = idx / D;
        int d = idx % D;
        float acc = 0.0f;
        for (int j = 0; j < N; j++) {
            acc += scores[i * N + j] * static_cast<float>(v[base + j * D + d]);
        }
        out[base + idx] = static_cast<ActT>(acc);
    }
}

// ── Forward dispatch ───────────────────────────────────────────────────
template <typename ActT, int kHeadDim, bool kCausal>
cudaError_t attention_forward(
    const ActT* q, const ActT* k, const ActT* v,
    ActT* out,
    ActT* softmax_lse_act,  // only used to locate the float buffer
    int batch, int n_heads, int seq_len,
    float scale,
    cudaStream_t stream
) {
    // softmax_lse is always FP32 regardless of ActT
    float* softmax_lse = reinterpret_cast<float*>(softmax_lse_act);

#ifdef WITH_FLASH_ATTN_3
    return flash_attn3_forward<ActT, kHeadDim, kCausal>(
        q, k, v, out, reinterpret_cast<ActT*>(softmax_lse),
        batch, n_heads, seq_len, scale, stream);
#elif defined(WITH_CUTLASS)
    return cutlass_fmha_forward<ActT, kHeadDim, kCausal>(
        q, k, v, out, reinterpret_cast<ActT*>(softmax_lse),
        batch, n_heads, seq_len, scale, stream);
#else
    int grid = batch * n_heads;
    int block = 128;
    int N = seq_len;
    int smem_bytes = (N * N + 2 * N) * sizeof(float);
    smem_attention_fwd_kernel<ActT, kHeadDim, kCausal>
        <<<grid, block, smem_bytes, stream>>>(
            q, k, v, out, softmax_lse, seq_len, scale);
    return cudaGetLastError();
#endif
}

// ── Backward kernel ────────────────────────────────────────────────────
// Recomputes attention weights from softmax_lse (log-sum-exp saved in fwd).
// dV = A^T dO, dA = dO V^T, backprop through softmax, dQ = dA' K * scale,
// dK = dA'^T Q * scale. All in SMEM for these tiny sequence lengths.

template <typename ActT, int kHeadDim, bool kCausal>
__global__ void __launch_bounds__(128, 4)
smem_attention_bwd_kernel(
    const ActT* __restrict__ grad_out,   // [B, H, N, D]
    const ActT* __restrict__ q,
    const ActT* __restrict__ k,
    const ActT* __restrict__ v,
    const ActT* __restrict__ attn_out,   // saved forward output
    const float* __restrict__ softmax_lse, // [B, H, N]
    ActT* __restrict__ grad_q,
    ActT* __restrict__ grad_k,
    ActT* __restrict__ grad_v,
    int seq_len, float scale
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int N = seq_len;
    const int D = kHeadDim;
    const int base = bh * N * D;

    extern __shared__ float smem[];
    float* scores = smem;               // N * N  (attention weights)
    float* dA     = scores + N * N;     // N * N  (grad through attn weights)

    // Recompute attention weights from softmax_lse
    for (int idx = tid; idx < N * N; idx += blockDim.x) {
        int i = idx / N, j = idx % N;
        if constexpr (kCausal) {
            if (j > i) { scores[idx] = 0.0f; continue; }
        }
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
            dot += static_cast<float>(q[base + i * D + d])
                 * static_cast<float>(k[base + j * D + d]);
        float lse = softmax_lse[bh * N + i];
        scores[idx] = ptx_expf(dot * scale - lse);
    }
    __syncthreads();

    // dV = A^T dO  (accumulated directly to global)
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int j = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int i = 0; i < N; i++) acc += scores[i * N + j]
            * static_cast<float>(grad_out[base + i * D + d]);
        grad_v[base + idx] = static_cast<ActT>(acc);
    }
    __syncthreads();

    // dA = dO V^T
    for (int idx = tid; idx < N * N; idx += blockDim.x) {
        int i = idx / N, j = idx % N;
        float acc = 0.0f;
        for (int d = 0; d < D; d++)
            acc += static_cast<float>(grad_out[base + i * D + d])
                 * static_cast<float>(v[base + j * D + d]);
        dA[idx] = acc;
    }
    __syncthreads();

    // Backprop through softmax: dS_ij = A_ij * (dA_ij - sum_k(A_ik * dA_ik))
    for (int i = tid; i < N; i += blockDim.x) {
        float dot_sum = 0.0f;
        for (int j = 0; j < N; j++) dot_sum += scores[i * N + j] * dA[i * N + j];
        for (int j = 0; j < N; j++) {
            float ds = scores[i * N + j] * (dA[i * N + j] - dot_sum) * scale;
            if constexpr (kCausal) { if (j > i) ds = 0.0f; }
            dA[i * N + j] = ds;   // reuse dA storage for dS
        }
    }
    __syncthreads();

    // dQ = dS K
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int i = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int j = 0; j < N; j++)
            acc += dA[i * N + j] * static_cast<float>(k[base + j * D + d]);
        grad_q[base + idx] = static_cast<ActT>(acc);
    }

    // dK = dS^T Q
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int j = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int i = 0; i < N; i++)
            acc += dA[i * N + j] * static_cast<float>(q[base + i * D + d]);
        grad_k[base + idx] = static_cast<ActT>(acc);
    }
}

// ── Backward dispatch ──────────────────────────────────────────────────
template <typename ActT, int kHeadDim, bool kCausal>
cudaError_t attention_backward(
    const ActT* grad_out,
    const ActT* q, const ActT* k, const ActT* v,
    const ActT* out, const ActT* softmax_lse_act,
    ActT* grad_q, ActT* grad_k, ActT* grad_v,
    int batch, int n_heads, int seq_len,
    float scale,
    cudaStream_t stream
) {
    const float* softmax_lse = reinterpret_cast<const float*>(softmax_lse_act);
    int grid = batch * n_heads;
    int block = 128;
    int N = seq_len;
    int smem_bytes = 2 * N * N * sizeof(float);  // scores + dA
    smem_attention_bwd_kernel<ActT, kHeadDim, kCausal>
        <<<grid, block, smem_bytes, stream>>>(
            grad_out, q, k, v, out, softmax_lse,
            grad_q, grad_k, grad_v, seq_len, scale);
    return cudaGetLastError();
}

}}}}  // namespace sg::sm90::models::attention
