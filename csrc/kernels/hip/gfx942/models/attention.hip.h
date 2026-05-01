// csrc/kernels/hip/gfx942/models/attention.hip.h
// Shared attention kernel for gfx942 (CDNA3 / MI300X). Serves both Decoder
// (causal, seq_len=4) and ViT (non-causal, seq_len=17) via kCausal template.
//
// Key differences from sm_90:
//   - Wave size 64: reductions use __shfl_xor with strides {32,16,8,4,2,1}
//   - No WGMMA/TMA: BF16 MFMA via __builtin_amdgcn_mfma_f32_16x16x16bf16
//   - 64 KB LDS per CU (not 228 KB SMEM)
//   - FULL_WARP_MASK is 0 on HIP (all 64 lanes lockstep)
//   - CK FMHA gated behind WITH_CK; fallback to hand-written MFMA path
//   - Occupancy via GROK_WAVES_PER_EU / GROK_FLAT_WORK_GROUP_SIZE
//   - warp_reduce_sum from utils.cuh (already wave-64 aware)
//
// For grokking shapes (d_head=32, seq_len=4/17), QK^T fits in regs/LDS.
// One workgroup per (batch, head) pair.
#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/tuned_configs.h"
#ifdef WITH_CK
#include <ck_tile/ops/fmha.hpp>
#endif

namespace sg { namespace gfx942 { namespace models { namespace attention {

constexpr int kMaxLdsBytes = 65536;  // 64 KB LDS per CU on gfx942

template <typename ActT, int kHeadDim, bool kCausal>
struct AttentionLaunchConfig {
    int block;
    int lds_bytes;
    int waves_per_eu;       // occupancy hint for CDNA3 scheduler
    bool use_ck_fmha;
    bool use_aiter;
};

// -- Wave-64 XOR butterfly reductions ----------------------------------------
__device__ __forceinline__ float wave64_reduce_sum(float val) {
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

__device__ __forceinline__ float wave64_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor(val, 32));
    val = fmaxf(val, __shfl_xor(val, 16));
    val = fmaxf(val, __shfl_xor(val, 8));
    val = fmaxf(val, __shfl_xor(val, 4));
    val = fmaxf(val, __shfl_xor(val, 2));
    val = fmaxf(val, __shfl_xor(val, 1));
    return val;
}

// -- External backend forward declarations ------------------------------------
#ifdef WITH_CK
template <typename ActT, int kHeadDim, bool kCausal>
hipError_t ck_fmha_forward(
    const ActT* q, const ActT* k, const ActT* v, ActT* out, ActT* softmax_lse,
    int batch, int n_heads, int seq_len, float scale, hipStream_t stream);
#endif
#ifdef WITH_AITER
template <typename ActT, int kHeadDim, bool kCausal>
hipError_t aiter_fmha_forward(
    const ActT* q, const ActT* k, const ActT* v, ActT* out, ActT* softmax_lse,
    int batch, int n_heads, int seq_len, float scale, hipStream_t stream);
#endif

// -- LDS-based attention kernel (default for tiny seq_len) ---------------------
// One block per (batch, head). Wave-64 cooperative matmul. Max seq_len: 32.
template <typename ActT, int kHeadDim, bool kCausal>
__global__ void
GROK_FLAT_WORK_GROUP_SIZE(64, 256)
GROK_WAVES_PER_EU(1, 4)
lds_attention_fwd_kernel(
    const ActT* __restrict__ q,      // [B, H, N, D]
    const ActT* __restrict__ k,
    const ActT* __restrict__ v,
    ActT* __restrict__ out,
    float* __restrict__ softmax_lse,  // [B, H, N] or nullptr
    int seq_len, float scale
) {
    const int bh = blockIdx.x, tid = threadIdx.x;
    const int N = seq_len, D = kHeadDim, base = bh * N * D;
    extern __shared__ float lds[];
    float* scores  = lds;
    float* row_max = scores + N * N;
    float* row_sum = row_max + N;
    // S = Q K^T * scale, optional causal mask
    for (int idx = tid; idx < N * N; idx += blockDim.x) {
        int i = idx / N, j = idx % N;
        if constexpr (kCausal) {
            if (j > i) { scores[idx] = -1e9f; continue; }
        }
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
            dot += static_cast<float>(q[base + i * D + d])
                 * static_cast<float>(k[base + j * D + d]);
        scores[idx] = dot * scale;
    }
    __syncthreads();
    // Row-wise stable softmax
    for (int i = tid; i < N; i += blockDim.x) {
        float m = -1e9f;
        for (int j = 0; j < N; j++) m = fmaxf(m, scores[i * N + j]);
        row_max[i] = m;
        float s = 0.0f;
        for (int j = 0; j < N; j++) {
            float e = expf(scores[i * N + j] - m);
            scores[i * N + j] = e;
            s += e;
        }
        float inv_s = 1.0f / fmaxf(s, 1e-12f);
        for (int j = 0; j < N; j++) scores[i * N + j] *= inv_s;
        row_sum[i] = s;
        if (softmax_lse != nullptr)
            softmax_lse[bh * N + i] = m + logf(fmaxf(s, 1e-12f));
    }
    __syncthreads();
    // Out = Softmax(S) * V
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int i = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int j = 0; j < N; j++)
            acc += scores[i * N + j] * static_cast<float>(v[base + j * D + d]);
        out[base + idx] = static_cast<ActT>(acc);
    }
}

// -- Forward dispatch ----------------------------------------------------------
template <typename ActT, int kHeadDim, bool kCausal>
hipError_t attention_forward(
    const ActT* q, const ActT* k, const ActT* v,
    ActT* out, ActT* softmax_lse_act,
    int batch, int n_heads, int seq_len, float scale,
    hipStream_t stream
) {
    float* softmax_lse = reinterpret_cast<float*>(softmax_lse_act);
#ifdef WITH_CK
    return ck_fmha_forward<ActT, kHeadDim, kCausal>(
        q, k, v, out, reinterpret_cast<ActT*>(softmax_lse),
        batch, n_heads, seq_len, scale, stream);
#elif defined(WITH_AITER)
    return aiter_fmha_forward<ActT, kHeadDim, kCausal>(
        q, k, v, out, reinterpret_cast<ActT*>(softmax_lse),
        batch, n_heads, seq_len, scale, stream);
#else
    int grid = batch * n_heads;
    int block = WARP_SIZE * 2;  // 128 threads = 2 waves on CDNA3
    int N = seq_len;
    int lds_bytes = (N * N + 2 * N) * sizeof(float);
    lds_attention_fwd_kernel<ActT, kHeadDim, kCausal>
        <<<grid, block, lds_bytes, stream>>>(
            q, k, v, out, softmax_lse, seq_len, scale);
    return hipGetLastError();
#endif
}

// -- Backward kernel ----------------------------------------------------------
// Recomputes attention weights from softmax_lse (log-sum-exp saved in fwd).
// dV = A^T dO, dA = dO V^T, backprop through softmax, dQ = dA' K * scale,
// dK = dA'^T Q * scale. All in LDS for these tiny sequence lengths.

template <typename ActT, int kHeadDim, bool kCausal>
__global__
GROK_FLAT_WORK_GROUP_SIZE(64, 256)
GROK_WAVES_PER_EU(1, 4)
void lds_attention_bwd_kernel(
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
    const int bh = blockIdx.x, tid = threadIdx.x;
    const int N = seq_len, D = kHeadDim, base = bh * N * D;

    extern __shared__ float lds[];
    float* scores = lds;             // N * N (attention weights)
    float* dA     = scores + N * N;  // N * N (grad through attn weights)

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

    // dV = A^T dO
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int j = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int i = 0; i < N; i++)
            acc += scores[i * N + j]
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
        for (int j = 0; j < N; j++)
            dot_sum += scores[i * N + j] * dA[i * N + j];
        for (int j = 0; j < N; j++) {
            float ds = scores[i * N + j] * (dA[i * N + j] - dot_sum) * scale;
            if constexpr (kCausal) { if (j > i) ds = 0.0f; }
            dA[i * N + j] = ds;  // reuse dA for dS
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

// -- Backward dispatch --------------------------------------------------------
template <typename ActT, int kHeadDim, bool kCausal>
hipError_t attention_backward(
    const ActT* grad_out,
    const ActT* q, const ActT* k, const ActT* v,
    const ActT* out, const ActT* softmax_lse_act,
    ActT* grad_q, ActT* grad_k, ActT* grad_v,
    int batch, int n_heads, int seq_len,
    float scale, hipStream_t stream
) {
    const float* softmax_lse = reinterpret_cast<const float*>(softmax_lse_act);
    int grid = batch * n_heads;
    int block = WARP_SIZE * 2;  // 128 threads = 2 waves on CDNA3
    int N = seq_len;
    int lds_bytes = 2 * N * N * sizeof(float);  // scores + dA
    lds_attention_bwd_kernel<ActT, kHeadDim, kCausal>
        <<<grid, block, lds_bytes, stream>>>(
            grad_out, q, k, v, out, softmax_lse,
            grad_q, grad_k, grad_v, seq_len, scale);
    return hipGetLastError();
}

}}}}  // namespace sg::gfx942::models::attention
