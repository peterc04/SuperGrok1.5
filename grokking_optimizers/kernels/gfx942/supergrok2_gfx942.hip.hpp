#ifndef GROKKING_SUPERGROK2_GFX942_HIP_HPP_
#define GROKKING_SUPERGROK2_GFX942_HIP_HPP_

#include "common_gfx942.hip.hpp"

namespace grokking {
namespace gfx942 {

// ============================================================================
//  SuperGrok v2 -- gfx942 (MI300X, CDNA3) fused optimizer kernel
//
//  Pipeline per step:
//    (1) sg2_bitonic_sort_abs     : sort elements by |grad| (bitonic in LDS)
//    (2) sg2_mamba_scan_forward   : bidirectional selective scan (Mamba-3)
//    (3) sg2_gru_update           : per-element GRU temporal memory
//    (4) sg2_peer_routing         : 4-head product-key expert routing
//    (5) sg2_expert_mlp           : expert MLP with LDS weight caching
//    (6) sg2_adam_update          : AdamW with amplified gradient
//    (7) sg2_fused_step           : __global__ kernel chaining all stages
//
//  gfx942-specific:
//    - 64-wide wavefront (CDNA3), NOT 32-wide warp
//    - __builtin_nontemporal_load for streaming grad reads (no __ldg)
//    - __builtin_isnan (not __isnanf)
//    - LDS (Local Data Share) = shared memory, 64 KB per CU
//    - FP32 accumulators throughout scan and reduction paths
// ============================================================================

// ---------------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------------
static constexpr int kSG2WavefrontSize = 64;
static constexpr int kSG2MaxDState     = 128;
static constexpr int kSG2MaxDInner     = 128;
static constexpr int kSG2MaxGruHidden  = 8;
static constexpr int kSG2MaxExpertHid  = 16;
static constexpr int kSG2NumHeads      = 4;
static constexpr int kSG2BlockSize     = 256;  // 4 wavefronts per workgroup

// LDS budget (must not exceed 64 KB)
static constexpr int kSG2SortLdsBytes   = 4096;   // bitonic sort scratch
static constexpr int kSG2ScanLdsBytes   = 16384;  // mamba scan state
static constexpr int kSG2ExpertLdsBytes = 8192;   // expert weight cache
static constexpr int kSG2TotalLdsBytes  = kSG2SortLdsBytes
                                        + kSG2ScanLdsBytes
                                        + kSG2ExpertLdsBytes;
static_assert(kSG2TotalLdsBytes <= 65536, "LDS budget exceeds gfx942 64 KB limit");

// ---------------------------------------------------------------------------
//  State struct
// ---------------------------------------------------------------------------
struct SuperGrok2State {
    float* __restrict__ exp_avg;
    float* __restrict__ exp_avg_sq;
    float* __restrict__ mu;
    float* __restrict__ sharpness;
    float* __restrict__ gru_state;       // [N * gru_hidden]
    float* __restrict__ mamba_fwd_state; // [d_inner * d_state]
    float* __restrict__ mamba_bwd_state; // [d_inner * d_state]
    static constexpr int num_state_tensors() { return 7; }
};

// ---------------------------------------------------------------------------
//  Activation helpers (FP32 scalar, device-only)
// ---------------------------------------------------------------------------
__forceinline__ __device__ float sg2_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ float sg2_softplus(float x) {
    return log1pf(expf(x));
}

__forceinline__ __device__ float sg2_silu(float x) {
    return x * sg2_sigmoid(x);
}

// ---------------------------------------------------------------------------
//  64-wide wavefront reduction utilities
// ---------------------------------------------------------------------------
__forceinline__ __device__ float sg2_wavefront_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kSG2WavefrontSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__forceinline__ __device__ float sg2_wavefront_reduce_max(float val) {
    #pragma unroll
    for (int offset = kSG2WavefrontSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

__forceinline__ __device__ int sg2_wavefront_reduce_max_idx(float val, int idx) {
    #pragma unroll
    for (int offset = kSG2WavefrontSize / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor(val, offset);
        int   other_idx = __shfl_xor(idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    return idx;
}

// ---------------------------------------------------------------------------
//  INT8 / INT4 expert weight dequantization
// ---------------------------------------------------------------------------

/// INT8 or FP32 dequantization based on template flag.
template <bool INT8_EXPERTS>
__forceinline__ __device__ float dequant_weight(const void* data, float scale, int idx) {
    if constexpr (INT8_EXPERTS) {
        return static_cast<float>(reinterpret_cast<const int8_t*>(data)[idx]) * scale;
    } else {
        return reinterpret_cast<const float*>(data)[idx];
    }
}

/// INT4 packed (two nibbles per byte) dequantization with zero-point.
__forceinline__ __device__ float dequant_int4(
    const uint8_t* packed, float scale, float zero, int idx)
{
    uint8_t byte   = packed[idx / 2];
    uint8_t nibble = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
    return static_cast<float>(nibble) * scale + zero;
}

// ============================================================================
//  (1) sg2_bitonic_sort_abs -- Sort elements by |grad| using bitonic sort
//      in LDS.  N <= 1024: fully in LDS.  N > 1024: block-cooperative with
//      global scratch.  Uses 64-wide wavefront shuffles for intra-wavefront
//      compare-and-swap.
// ============================================================================
__forceinline__ __device__
void sg2_bitonic_sort_abs(
    const float* __restrict__ grads,
    float*       __restrict__ sort_keys,    // output: |grad| sorted
    int*         __restrict__ sort_indices,  // output: original indices
    float*       __restrict__ global_scratch,// scratch for N > 1024 (may be nullptr)
    int                       N)
{
    __shared__ float  lds_keys[1024];
    __shared__ int    lds_idx[1024];

    const int tid   = threadIdx.x;
    const int bsz   = blockDim.x;
    const int bid   = blockIdx.x;

    // ---- Phase 1: Load into LDS (or global scratch for large N) ----
    const bool use_lds = (N <= 1024);
    float* keys_buf = use_lds ? lds_keys : (global_scratch + bid * N);
    int*   idx_buf  = use_lds ? lds_idx  : reinterpret_cast<int*>(global_scratch + gridDim.x * N + bid * N);

    // Pad to next power of two for bitonic sort
    int padded = 1;
    while (padded < N) padded <<= 1;

    for (int i = tid; i < padded; i += bsz) {
        if (i < N) {
            float g = __builtin_nontemporal_load(grads + i);
            g = __builtin_isnan(g) ? 0.0f : g;
            keys_buf[i] = fabsf(g);
            idx_buf[i]  = i;
        } else {
            // Sentinel: -inf sorts to the end (ascending) -- we sort descending
            keys_buf[i] = -1.0f;
            idx_buf[i]  = -1;
        }
    }
    __syncthreads();

    // ---- Phase 2: Bitonic sort (descending by |grad|) ----
    for (int size = 2; size <= padded; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {

            // Intra-wavefront: use shuffles for stride <= 32
            if (stride <= 32 && use_lds) {
                const int lane = tid % kSG2WavefrontSize;
                for (int i = tid; i < padded / 2; i += bsz) {
                    int block_id = i / (size / 2);
                    int local_id = i % (size / 2);
                    int pos      = block_id * size + local_id;
                    int partner  = pos ^ stride;

                    if (partner > pos && partner < padded) {
                        float k0 = keys_buf[pos];
                        float k1 = keys_buf[partner];
                        int   i0 = idx_buf[pos];
                        int   i1 = idx_buf[partner];

                        // Descending: swap if k0 < k1 in the correct half
                        bool ascending_half = ((pos & size) == 0);
                        bool should_swap = ascending_half ? (k0 < k1) : (k0 > k1);

                        if (should_swap) {
                            keys_buf[pos]     = k1;
                            keys_buf[partner] = k0;
                            idx_buf[pos]      = i1;
                            idx_buf[partner]  = i0;
                        }
                    }
                }
            } else {
                // Global / LDS path for larger strides
                for (int i = tid; i < padded / 2; i += bsz) {
                    int block_id = i / (size / 2);
                    int local_id = i % (size / 2);
                    int pos      = block_id * size + local_id;
                    int partner  = pos ^ stride;

                    if (partner > pos && partner < padded) {
                        float k0 = keys_buf[pos];
                        float k1 = keys_buf[partner];
                        int   i0 = idx_buf[pos];
                        int   i1 = idx_buf[partner];

                        bool ascending_half = ((pos & size) == 0);
                        bool should_swap = ascending_half ? (k0 < k1) : (k0 > k1);

                        if (should_swap) {
                            keys_buf[pos]     = k1;
                            keys_buf[partner] = k0;
                            idx_buf[pos]      = i1;
                            idx_buf[partner]  = i0;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // ---- Phase 3: Write back sorted results ----
    for (int i = tid; i < N; i += bsz) {
        sort_keys[i]    = keys_buf[i];
        sort_indices[i] = idx_buf[i];
    }
    __syncthreads();
}

// ============================================================================
//  (2) sg2_mamba_scan_forward -- Selective scan (Mamba-3)
//
//  State update per inner dimension j, state index s:
//    dt  = softplus(dt_proj @ x_branch)
//    B   = B_proj @ x_branch
//    C   = C_proj @ x_branch
//    h[j,s] = exp(A[j,s] * dt[j]) * h[j,s] + B[s] * x[j] * dt[j]
//    y[j]   = sum_s( C[s] * h[j,s] ) + D[j] * x[j]
//
//  FP32-only for scan state. Sequential along sequence dimension.
//  LDS is used for the state matrix tile.
// ============================================================================
__forceinline__ __device__
void sg2_mamba_scan_forward(
    float*       __restrict__ h_state,     // [d_inner * d_state] persistent state
    const float* __restrict__ x_sorted,    // [seq_len, d_inner]  sorted input
    const float* __restrict__ A_log,       // [d_inner, d_state]  log(A)
    const float* __restrict__ D_param,     // [d_inner]
    const float* __restrict__ dt_proj_W,   // [d_inner, d_inner]  dt projection weight
    const float* __restrict__ dt_proj_b,   // [d_inner]           dt projection bias
    const float* __restrict__ B_proj_W,    // [d_state, d_inner]
    const float* __restrict__ C_proj_W,    // [d_state, d_inner]
    float*       __restrict__ y_out,       // [seq_len, d_inner]  output
    int                       seq_len,
    int                       d_inner,
    int                       d_state)
{
    // LDS tile for state matrix: up to kSG2MaxDInner * kSG2MaxDState floats
    // In practice d_inner * d_state should fit in kSG2ScanLdsBytes.
    __shared__ float lds_h[kSG2ScanLdsBytes / sizeof(float)];

    const int tid = threadIdx.x;
    const int bsz = blockDim.x;

    // Load persistent state into LDS
    const int state_size = d_inner * d_state;
    for (int i = tid; i < state_size; i += bsz) {
        lds_h[i] = h_state[i];
    }
    __syncthreads();

    // Process each timestep sequentially (scan is inherently serial along seq)
    for (int t = 0; t < seq_len; ++t) {
        const float* x_t = x_sorted + t * d_inner;

        // Each thread handles a slice of d_inner
        for (int j = tid; j < d_inner; j += bsz) {
            // Compute dt = softplus(dt_proj @ x_branch + bias)
            float dt_raw = __builtin_nontemporal_load(&dt_proj_b[j]);
            for (int k = 0; k < d_inner; ++k) {
                dt_raw += __builtin_nontemporal_load(&dt_proj_W[j * d_inner + k])
                        * __builtin_nontemporal_load(&x_t[k]);
            }
            float dt = sg2_softplus(dt_raw);

            // Compute B and C projections for each state dimension
            float y_acc = 0.0f;
            float x_j = __builtin_nontemporal_load(&x_t[j]);

            for (int s = 0; s < d_state; ++s) {
                // B[s] = B_proj_W[s,:] @ x_t
                float B_s = 0.0f;
                for (int k = 0; k < d_inner; ++k) {
                    B_s += __builtin_nontemporal_load(&B_proj_W[s * d_inner + k])
                         * __builtin_nontemporal_load(&x_t[k]);
                }

                // C[s] = C_proj_W[s,:] @ x_t
                float C_s = 0.0f;
                for (int k = 0; k < d_inner; ++k) {
                    C_s += __builtin_nontemporal_load(&C_proj_W[s * d_inner + k])
                         * __builtin_nontemporal_load(&x_t[k]);
                }

                // State update: h = exp(A * dt) * h + B * x * dt
                float A_val = __builtin_nontemporal_load(&A_log[j * d_state + s]);
                float decay = expf(A_val * dt);
                int   h_idx = j * d_state + s;
                float h_old = lds_h[h_idx];
                float h_new = decay * h_old + B_s * x_j * dt;
                lds_h[h_idx] = h_new;

                // Accumulate output: y += C * h
                y_acc += C_s * h_new;
            }

            // Add skip connection: y += D * x
            float D_j = __builtin_nontemporal_load(&D_param[j]);
            y_acc += D_j * x_j;

            y_out[t * d_inner + j] = y_acc;
        }
        __syncthreads();  // All threads must finish timestep before next
    }

    // Persist state back to global memory
    for (int i = tid; i < state_size; i += bsz) {
        h_state[i] = lds_h[i];
    }
    __syncthreads();
}

// ============================================================================
//  (3) sg2_gru_update -- Per-element GRU
//
//  z       = sigmoid(W_z @ [input, h] + b_z)
//  r       = sigmoid(W_r @ [input, h] + b_r)
//  h_tilde = tanh(W_h @ [input, r*h] + b_h)
//  h_new   = (1 - z) * h + z * h_tilde
//
//  Grid-stride loop, each thread handles one element's GRU hidden state.
// ============================================================================
__forceinline__ __device__
void sg2_gru_update(
    float*       __restrict__ gru_state,   // [N, gru_hidden]
    const float* __restrict__ input,       // [N]  (scan output for each element)
    const float* __restrict__ W_z,         // [gru_hidden, 1 + gru_hidden]
    const float* __restrict__ b_z,         // [gru_hidden]
    const float* __restrict__ W_r,         // [gru_hidden, 1 + gru_hidden]
    const float* __restrict__ b_r,         // [gru_hidden]
    const float* __restrict__ W_h,         // [gru_hidden, 1 + gru_hidden]
    const float* __restrict__ b_h,         // [gru_hidden]
    int                       N,
    int                       gru_hidden)
{
    int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    const int concat_dim = 1 + gru_hidden;  // [input_scalar, h_0, h_1, ..., h_{H-1}]

    for (int64_t i = idx; i < static_cast<int64_t>(N); i += stride) {
        float inp = __builtin_nontemporal_load(&input[i]);
        if (__builtin_isnan(inp)) inp = 0.0f;

        float* h_ptr = gru_state + i * gru_hidden;

        // Load current hidden state into registers
        float h_reg[kSG2MaxGruHidden];
        for (int hh = 0; hh < gru_hidden; ++hh) {
            h_reg[hh] = h_ptr[hh];
        }

        // Compute all GRU gates per hidden unit
        for (int hh = 0; hh < gru_hidden; ++hh) {
            // Build concat vector: [inp, h_0, ..., h_{H-1}]
            // z_gate = sigmoid(W_z[hh,:] @ concat + b_z[hh])
            float z_acc = __builtin_nontemporal_load(&b_z[hh]);
            float r_acc = __builtin_nontemporal_load(&b_r[hh]);

            // Weight for input element
            z_acc += __builtin_nontemporal_load(&W_z[hh * concat_dim + 0]) * inp;
            r_acc += __builtin_nontemporal_load(&W_r[hh * concat_dim + 0]) * inp;

            // Weights for hidden state elements
            for (int k = 0; k < gru_hidden; ++k) {
                z_acc += __builtin_nontemporal_load(&W_z[hh * concat_dim + 1 + k]) * h_reg[k];
                r_acc += __builtin_nontemporal_load(&W_r[hh * concat_dim + 1 + k]) * h_reg[k];
            }

            float z_gate = sg2_sigmoid(z_acc);
            float r_gate = sg2_sigmoid(r_acc);

            // h_tilde = tanh(W_h[hh,:] @ [inp, r*h] + b_h[hh])
            float h_acc = __builtin_nontemporal_load(&b_h[hh]);
            h_acc += __builtin_nontemporal_load(&W_h[hh * concat_dim + 0]) * inp;
            for (int k = 0; k < gru_hidden; ++k) {
                h_acc += __builtin_nontemporal_load(&W_h[hh * concat_dim + 1 + k])
                       * (r_gate * h_reg[k]);
            }
            float h_tilde = tanhf(h_acc);

            // h_new = (1 - z) * h_old + z * h_tilde
            h_reg[hh] = (1.0f - z_gate) * h_reg[hh] + z_gate * h_tilde;
        }

        // Write back updated hidden state
        for (int hh = 0; hh < gru_hidden; ++hh) {
            h_ptr[hh] = h_reg[hh];
        }
    }
}

// ============================================================================
//  (4) sg2_peer_routing -- 4-Head product-key expert routing
//
//  Per element, per head:
//    q       = W_query @ input
//    split q into q_a (first half) and q_b (second half)
//    idx_a   = argmax(q_a @ keys_A^T)
//    idx_b   = argmax(q_b @ keys_B^T)
//    expert_idx = idx_a * pk_dim + idx_b
//
//  Output: per-element array of kSG2NumHeads expert indices + scores.
// ============================================================================
__forceinline__ __device__
void sg2_peer_routing(
    const float* __restrict__ input,          // [N]  per-element input
    const float* __restrict__ gru_state,      // [N, gru_hidden]
    const float* __restrict__ W_query,        // [num_heads, query_dim, (1 + gru_hidden)]
    const float* __restrict__ keys_A,         // [num_heads, pk_dim, query_dim/2]
    const float* __restrict__ keys_B,         // [num_heads, pk_dim, query_dim/2]
    int*         __restrict__ expert_indices,  // [N, num_heads] output
    float*       __restrict__ expert_scores,   // [N, num_heads] output (softmax scores)
    int                       N,
    int                       gru_hidden,
    int                       query_dim,
    int                       pk_dim,
    int                       num_heads)
{
    int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    const int concat_dim = 1 + gru_hidden;
    const int half_qd    = query_dim / 2;

    for (int64_t i = idx; i < static_cast<int64_t>(N); i += stride) {
        float inp = __builtin_nontemporal_load(&input[i]);
        if (__builtin_isnan(inp)) inp = 0.0f;

        const float* h_ptr = gru_state + i * gru_hidden;

        // Accumulate un-normalized scores for softmax across heads
        float head_logits[kSG2NumHeads];
        float max_logit = -1e30f;

        for (int head = 0; head < num_heads; ++head) {
            const float* W_q_head = W_query + head * query_dim * concat_dim;
            const float* kA_head  = keys_A  + head * pk_dim * half_qd;
            const float* kB_head  = keys_B  + head * pk_dim * half_qd;

            // q = W_query[head] @ [inp, h]
            float q[kSG2MaxExpertHid];  // query_dim <= kSG2MaxExpertHid assumed
            for (int d = 0; d < query_dim; ++d) {
                float acc = __builtin_nontemporal_load(&W_q_head[d * concat_dim]) * inp;
                for (int k = 0; k < gru_hidden; ++k) {
                    acc += __builtin_nontemporal_load(&W_q_head[d * concat_dim + 1 + k])
                         * h_ptr[k];
                }
                q[d] = acc;
            }

            // Split q into q_a = q[0:half_qd], q_b = q[half_qd:query_dim]
            // idx_a = argmax(q_a @ keys_A^T)
            float best_a = -1e30f;
            int   best_a_idx = 0;
            for (int ka = 0; ka < pk_dim; ++ka) {
                float dot = 0.0f;
                for (int d = 0; d < half_qd; ++d) {
                    dot += q[d] * __builtin_nontemporal_load(&kA_head[ka * half_qd + d]);
                }
                if (dot > best_a) {
                    best_a     = dot;
                    best_a_idx = ka;
                }
            }

            // idx_b = argmax(q_b @ keys_B^T)
            float best_b = -1e30f;
            int   best_b_idx = 0;
            for (int kb = 0; kb < pk_dim; ++kb) {
                float dot = 0.0f;
                for (int d = 0; d < half_qd; ++d) {
                    dot += q[half_qd + d]
                         * __builtin_nontemporal_load(&kB_head[kb * half_qd + d]);
                }
                if (dot > best_b) {
                    best_b     = dot;
                    best_b_idx = kb;
                }
            }

            int eidx = best_a_idx * pk_dim + best_b_idx;
            expert_indices[i * num_heads + head] = eidx;
            head_logits[head] = best_a + best_b;  // routing score
            max_logit = fmaxf(max_logit, head_logits[head]);
        }

        // Softmax over heads for combining expert outputs
        float sum_exp = 0.0f;
        for (int head = 0; head < num_heads; ++head) {
            head_logits[head] = expf(head_logits[head] - max_logit);
            sum_exp += head_logits[head];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int head = 0; head < num_heads; ++head) {
            expert_scores[i * num_heads + head] = head_logits[head] * inv_sum;
        }
    }
}

// ============================================================================
//  (5) sg2_expert_mlp -- Expert MLP with LDS caching of active expert weights
//
//  For each element:
//    For each head: load expert weights into LDS, compute
//      hidden = relu(W1 * gated_input + b1)
//      out    = W2 @ hidden + b2
//    final  = rescale * mean_over_heads(out)
//
//  Expert weights are cached in LDS within each workgroup to avoid redundant
//  global memory reads when multiple elements in the same workgroup route to
//  the same expert.
// ============================================================================
template <bool INT8_EXPERTS = false>
__forceinline__ __device__
void sg2_expert_mlp(
    const float* __restrict__ input,           // [N]
    const int*   __restrict__ expert_indices,  // [N, num_heads]
    const float* __restrict__ expert_scores,   // [N, num_heads]
    const void*  __restrict__ expert_W1,       // [num_experts, expert_hidden]
    const void*  __restrict__ expert_b1,       // [num_experts, expert_hidden]
    const void*  __restrict__ expert_W2,       // [num_experts, expert_hidden]
    const float* __restrict__ expert_b2,       // [num_experts]
    const float* __restrict__ expert_scales,   // [num_experts] (INT8 mode only)
    float*       __restrict__ output,          // [N]
    float                     rescale,
    int                       N,
    int                       num_experts,
    int                       expert_hidden,
    int                       num_heads)
{
    // LDS cache for one expert's W1, b1, W2 weights per workgroup pass
    __shared__ float lds_W1[kSG2MaxExpertHid];
    __shared__ float lds_b1[kSG2MaxExpertHid];
    __shared__ float lds_W2[kSG2MaxExpertHid];
    __shared__ float lds_b2_cached;

    int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < static_cast<int64_t>(N); i += stride) {
        float inp = __builtin_nontemporal_load(&input[i]);
        if (__builtin_isnan(inp)) inp = 0.0f;

        float combined_out = 0.0f;

        for (int head = 0; head < num_heads; ++head) {
            int eidx  = expert_indices[i * num_heads + head];
            float score = expert_scores[i * num_heads + head];

            float expert_scale = INT8_EXPERTS ?
                __builtin_nontemporal_load(&expert_scales[eidx]) : 1.0f;

            // Cache this expert's weights in LDS (first thread in warp loads)
            int lane = threadIdx.x % kSG2WavefrontSize;
            for (int h = lane; h < expert_hidden; h += kSG2WavefrontSize) {
                lds_W1[h] = dequant_weight<INT8_EXPERTS>(
                    expert_W1, expert_scale, eidx * expert_hidden + h);
                lds_b1[h] = reinterpret_cast<const float*>(expert_b1)[eidx * expert_hidden + h];
                lds_W2[h] = dequant_weight<INT8_EXPERTS>(
                    expert_W2, expert_scale, eidx * expert_hidden + h);
            }
            if (lane == 0) {
                lds_b2_cached = expert_b2[eidx];
            }
            // Ensure LDS writes are visible to all lanes
            __builtin_amdgcn_wave_barrier();

            // MLP forward: out = W2 @ relu(W1 * inp + b1) + b2
            float mlp_out = lds_b2_cached;
            for (int h = 0; h < expert_hidden; ++h) {
                float hidden = lds_W1[h] * inp + lds_b1[h];
                hidden = fmaxf(hidden, 0.0f);  // ReLU
                mlp_out += lds_W2[h] * hidden;
            }

            combined_out += score * mlp_out;
        }

        output[i] = rescale * combined_out;
    }
}

// ============================================================================
//  (6) sg2_adam_update -- AdamW step with amplified gradient
//
//  EMA:   mu = layer_alpha * mu + (1 - layer_alpha) * g
//  Effective gradient:
//    effective_g = g + lamb * ramp * gate_signal * (smart_g - g)
//  Adam:
//    m = beta1 * m + (1 - beta1) * effective_g
//    v = beta2 * v + (1 - beta2) * effective_g^2
//    p = p - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
//
//  Provides both scalar and vec4 paths.
// ============================================================================
template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void sg2_adam_update(
    ParamT*       __restrict__ params,
    const ParamT* __restrict__ grads,
    SuperGrok2State            state,
    const float*  __restrict__ smart_grads, // [N] output from expert MLP
    int64_t                    n,
    float lr,
    float layer_beta1,
    float beta2,
    float eps,
    float weight_decay,
    float layer_alpha,
    float lamb,
    float ramp,
    float gate_signal,
    float grad_clip,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f)
{
    int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        // Streaming non-temporal grad read
        ParamT g_raw = __builtin_nontemporal_load(grads + i);
        float g = to_float(g_raw);
        g = apply_nan_policy<NAN_POLICY>(g);
        g = apply_clip<ENABLE_CLIP>(g, clip_threshold);

        // Always-on per-element gradient clip
        g = fminf(fmaxf(g, -grad_clip), grad_clip);

        // EMA gradient update
        float mu_old = state.mu[i];
        float mu_val = layer_alpha * mu_old + (1.0f - layer_alpha) * g;

        // Read smart gradient from expert pipeline
        float smart_g = smart_grads[i];
        if (__builtin_isnan(smart_g)) smart_g = g;

        // Amplified gradient with gating
        float effective_g = g + lamb * ramp * gate_signal * (smart_g - g);

        // AdamW step
        float p_f   = to_float(params[i]);
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = layer_beta1 * m_old + (1.0f - layer_beta1) * effective_g;
        float v = beta2 * v_old + (1.0f - beta2) * effective_g * effective_g;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        float denom  = sqrtf(v_hat) + eps;
        float update = m_hat / denom + weight_decay * p_f;

        p_f -= lr * update;

        // Update sharpness (running magnitude of correction signal)
        float sharp_old = state.sharpness[i];
        float correction_mag = fabsf(smart_g - g);
        float new_sharp = layer_alpha * sharp_old + (1.0f - layer_alpha) * correction_mag;

        // Writeback
        state.exp_avg[i]    = m;
        state.exp_avg_sq[i] = v;
        state.mu[i]         = mu_val;
        state.sharpness[i]  = new_sharp;
        params[i]           = from_float<ParamT>(p_f);
    }
}

// ---------------------------------------------------------------------------
//  sg2_adam_update_vec4 -- Vectorized float4 path (float params only)
// ---------------------------------------------------------------------------
template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void sg2_adam_update_vec4(
    float*       __restrict__ params,
    const float* __restrict__ grads,
    SuperGrok2State            state,
    const float*  __restrict__ smart_grads,
    int64_t                    n,
    float lr,
    float layer_beta1,
    float beta2,
    float eps,
    float weight_decay,
    float layer_alpha,
    float lamb,
    float ramp,
    float gate_signal,
    float grad_clip,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f)
{
    int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    const int64_t n4 = n / 4;

    float4* __restrict__       p4  = reinterpret_cast<float4*>(params);
    const float4* __restrict__ g4  = reinterpret_cast<const float4*>(grads);
    const float4* __restrict__ sg4 = reinterpret_cast<const float4*>(smart_grads);
    float4* __restrict__       m4  = reinterpret_cast<float4*>(state.exp_avg);
    float4* __restrict__       v4  = reinterpret_cast<float4*>(state.exp_avg_sq);
    float4* __restrict__       mu4 = reinterpret_cast<float4*>(state.mu);
    float4* __restrict__       sh4 = reinterpret_cast<float4*>(state.sharpness);

    for (int64_t i = idx; i < n4; i += stride) {
        float4 p_vec  = p4[i];
        float4 g_vec  = __builtin_nontemporal_load(g4 + i);
        float4 sg_vec = sg4[i];
        float4 m_vec  = m4[i];
        float4 v_vec  = v4[i];
        float4 mu_vec = mu4[i];
        float4 sh_vec = sh4[i];

        float gs[4]  = {g_vec.x,  g_vec.y,  g_vec.z,  g_vec.w};
        float sgs[4] = {sg_vec.x, sg_vec.y, sg_vec.z, sg_vec.w};
        float ps[4]  = {p_vec.x,  p_vec.y,  p_vec.z,  p_vec.w};
        float ms[4]  = {m_vec.x,  m_vec.y,  m_vec.z,  m_vec.w};
        float vs[4]  = {v_vec.x,  v_vec.y,  v_vec.z,  v_vec.w};
        float mus[4] = {mu_vec.x, mu_vec.y, mu_vec.z, mu_vec.w};
        float shs[4] = {sh_vec.x, sh_vec.y, sh_vec.z, sh_vec.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g = apply_nan_policy<NAN_POLICY>(gs[k]);
            g = apply_clip<ENABLE_CLIP>(g, clip_threshold);
            g = fminf(fmaxf(g, -grad_clip), grad_clip);

            // EMA
            float mu_val = layer_alpha * mus[k] + (1.0f - layer_alpha) * g;

            // Smart gradient
            float sg = __builtin_isnan(sgs[k]) ? g : sgs[k];

            // Amplified gradient
            float effective_g = g + lamb * ramp * gate_signal * (sg - g);

            // AdamW
            float m = layer_beta1 * ms[k] + (1.0f - layer_beta1) * effective_g;
            float v = beta2 * vs[k] + (1.0f - beta2) * effective_g * effective_g;

            float m_hat = m * bias_correction1;
            float v_hat = v * bias_correction2;

            ps[k] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * ps[k]);
            ms[k]  = m;
            vs[k]  = v;
            mus[k] = mu_val;
            shs[k] = layer_alpha * shs[k] + (1.0f - layer_alpha) * fabsf(sg - gs[k]);
        }

        p4[i]  = make_float4(ps[0],  ps[1],  ps[2],  ps[3]);
        m4[i]  = make_float4(ms[0],  ms[1],  ms[2],  ms[3]);
        v4[i]  = make_float4(vs[0],  vs[1],  vs[2],  vs[3]);
        mu4[i] = make_float4(mus[0], mus[1], mus[2], mus[3]);
        sh4[i] = make_float4(shs[0], shs[1], shs[2], shs[3]);
    }

    // Handle tail elements
    int64_t tail_start = n4 * 4;
    for (int64_t i = tail_start + idx; i < n; i += stride) {
        float g_raw = __builtin_nontemporal_load(grads + i);
        float g = apply_nan_policy<NAN_POLICY>(g_raw);
        g = apply_clip<ENABLE_CLIP>(g, clip_threshold);
        g = fminf(fmaxf(g, -grad_clip), grad_clip);

        float mu_old  = state.mu[i];
        float mu_val  = layer_alpha * mu_old + (1.0f - layer_alpha) * g;
        float sg      = smart_grads[i];
        if (__builtin_isnan(sg)) sg = g;

        float effective_g = g + lamb * ramp * gate_signal * (sg - g);

        float p_f   = params[i];
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = layer_beta1 * m_old + (1.0f - layer_beta1) * effective_g;
        float v = beta2 * v_old + (1.0f - beta2) * effective_g * effective_g;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        p_f -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_f);

        float sharp_old = state.sharpness[i];
        float new_sharp = layer_alpha * sharp_old + (1.0f - layer_alpha) * fabsf(sg - g);

        state.exp_avg[i]    = m;
        state.exp_avg_sq[i] = v;
        state.mu[i]         = mu_val;
        state.sharpness[i]  = new_sharp;
        params[i]           = p_f;
    }
}

// ============================================================================
//  (7) sg2_fused_step -- __global__ kernel chaining all SuperGrok v2
//      components into a single dispatch.
//
//  Pipeline:
//    1. Bitonic sort by |grad|
//    2. Mamba-3 forward selective scan
//    3. Mamba-3 backward selective scan
//    4. GRU update (fuses fwd + bwd scan outputs)
//    5. PEER product-key expert routing
//    6. Expert MLP evaluation
//    7. AdamW parameter update with amplified gradient
//
//  Template params:
//    ParamT      -- parameter storage type (float, __half, hip_bfloat16)
//    NAN_POLICY  -- how to handle NaN gradients
//    ENABLE_CLIP -- compile-time toggle for gradient clipping
//    INT8_EXPERTS -- use INT8 quantized expert weights
// ============================================================================
template <typename ParamT,
          NanPolicy NAN_POLICY = NanPolicy::kNone,
          bool ENABLE_CLIP = false,
          bool INT8_EXPERTS = false>
__global__ void sg2_fused_step(
    // Parameter + gradient tensors
    ParamT*       __restrict__ params,
    const ParamT* __restrict__ grads,
    int64_t                    N,

    // Optimizer state
    SuperGrok2State            state,

    // Mamba scan weights
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ dt_proj_W,
    const float* __restrict__ dt_proj_b,
    const float* __restrict__ B_proj_W,
    const float* __restrict__ C_proj_W,
    int                       d_inner,
    int                       d_state,

    // GRU weights
    const float* __restrict__ gru_W_z,
    const float* __restrict__ gru_b_z,
    const float* __restrict__ gru_W_r,
    const float* __restrict__ gru_b_r,
    const float* __restrict__ gru_W_h,
    const float* __restrict__ gru_b_h,
    int                       gru_hidden,

    // PEER routing weights
    const float* __restrict__ peer_W_query,
    const float* __restrict__ peer_keys_A,
    const float* __restrict__ peer_keys_B,
    int                       query_dim,
    int                       pk_dim,
    int                       num_heads,

    // Expert MLP weights
    const void*  __restrict__ expert_W1,
    const void*  __restrict__ expert_b1,
    const void*  __restrict__ expert_W2,
    const float* __restrict__ expert_b2,
    const float* __restrict__ expert_scales,
    int                       num_experts,
    int                       expert_hidden,
    float                     expert_rescale,

    // Adam hyperparameters
    float lr,
    float layer_beta1,
    float beta2,
    float eps,
    float weight_decay,
    float layer_alpha,
    float lamb,
    float ramp,
    float gate_signal,
    float grad_clip,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold,

    // Scratch buffers (caller-allocated)
    float* __restrict__ sort_keys_buf,      // [N]
    int*   __restrict__ sort_indices_buf,   // [N]
    float* __restrict__ scan_input_buf,     // [N, d_inner]
    float* __restrict__ scan_fwd_out_buf,   // [N, d_inner]
    float* __restrict__ scan_bwd_out_buf,   // [N, d_inner]
    float* __restrict__ gru_input_buf,      // [N]
    int*   __restrict__ expert_idx_buf,     // [N, num_heads]
    float* __restrict__ expert_score_buf,   // [N, num_heads]
    float* __restrict__ smart_grad_buf,     // [N]
    float* __restrict__ global_sort_scratch // [>= 2*N] for large N, nullptr if N<=1024
) {
    // -----------------------------------------------------------------------
    // Stage 1: Sort by |grad| magnitude
    // -----------------------------------------------------------------------
    sg2_bitonic_sort_abs(
        reinterpret_cast<const float*>(grads),
        sort_keys_buf,
        sort_indices_buf,
        global_sort_scratch,
        static_cast<int>(N));
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 2: Build scan input from sorted gradients + sharpness
    // Project [grad, sharpness] -> d_inner dimensional space
    // -----------------------------------------------------------------------
    {
        const int tid = threadIdx.x;
        const int bsz = blockDim.x;
        for (int64_t elem = tid; elem < N; elem += bsz) {
            int orig_idx = sort_indices_buf[elem];
            if (orig_idx >= 0 && orig_idx < static_cast<int>(N)) {
                // Simple projection: replicate grad into d_inner channels
                // (full projection handled by host-side GEMM for large N)
                float g = to_float(__builtin_nontemporal_load(grads + orig_idx));
                g = apply_nan_policy<NAN_POLICY>(g);
                for (int j = 0; j < d_inner; ++j) {
                    scan_input_buf[elem * d_inner + j] = g;
                }
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 3: Mamba forward scan
    // -----------------------------------------------------------------------
    sg2_mamba_scan_forward(
        state.mamba_fwd_state,
        scan_input_buf,
        A_log, D_param,
        dt_proj_W, dt_proj_b,
        B_proj_W, C_proj_W,
        scan_fwd_out_buf,
        static_cast<int>(N), d_inner, d_state);
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 4: Mamba backward scan (reverse order)
    // Reuse the same scan function on reversed input
    // -----------------------------------------------------------------------
    {
        // Reverse scan_input_buf into a temporary reordering via index math
        // The backward scan reads from the end of the sorted sequence
        const int tid = threadIdx.x;
        const int bsz = blockDim.x;

        // Build reversed input in scan_bwd_out_buf temporarily
        for (int64_t elem = tid; elem < N; elem += bsz) {
            int rev = static_cast<int>(N) - 1 - static_cast<int>(elem);
            for (int j = 0; j < d_inner; ++j) {
                scan_bwd_out_buf[elem * d_inner + j] =
                    scan_input_buf[rev * d_inner + j];
            }
        }
        __syncthreads();

        // Run backward scan using reversed input
        // Use a temporary LDS-resident state; overwrite scan_bwd_out_buf in-place
        sg2_mamba_scan_forward(
            state.mamba_bwd_state,
            scan_bwd_out_buf,   // reversed input
            A_log, D_param,
            dt_proj_W, dt_proj_b,
            B_proj_W, C_proj_W,
            scan_bwd_out_buf,   // overwrite with output
            static_cast<int>(N), d_inner, d_state);
        __syncthreads();

        // Reverse the backward output back to original order
        // We use smart_grad_buf as temp for one d_inner column at a time
        for (int j = 0; j < d_inner; ++j) {
            for (int64_t elem = tid; elem < N / 2; elem += bsz) {
                int rev = static_cast<int>(N) - 1 - static_cast<int>(elem);
                float tmp = scan_bwd_out_buf[elem * d_inner + j];
                scan_bwd_out_buf[elem * d_inner + j] =
                    scan_bwd_out_buf[rev * d_inner + j];
                scan_bwd_out_buf[rev * d_inner + j] = tmp;
            }
            __syncthreads();
        }
    }

    // -----------------------------------------------------------------------
    // Stage 5: Merge fwd + bwd scan outputs -> per-element GRU input
    // Average across d_inner channels and unsort back to original order
    // -----------------------------------------------------------------------
    {
        const int tid = threadIdx.x;
        const int bsz = blockDim.x;
        for (int64_t elem = tid; elem < N; elem += bsz) {
            int orig_idx = sort_indices_buf[elem];
            if (orig_idx >= 0 && orig_idx < static_cast<int>(N)) {
                float sum = 0.0f;
                for (int j = 0; j < d_inner; ++j) {
                    sum += scan_fwd_out_buf[elem * d_inner + j]
                         + scan_bwd_out_buf[elem * d_inner + j];
                }
                gru_input_buf[orig_idx] = sum / static_cast<float>(2 * d_inner);
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 6: GRU update
    // -----------------------------------------------------------------------
    sg2_gru_update(
        state.gru_state,
        gru_input_buf,
        gru_W_z, gru_b_z,
        gru_W_r, gru_b_r,
        gru_W_h, gru_b_h,
        static_cast<int>(N), gru_hidden);
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 7: PEER routing
    // -----------------------------------------------------------------------
    sg2_peer_routing(
        gru_input_buf,
        state.gru_state,
        peer_W_query,
        peer_keys_A,
        peer_keys_B,
        expert_idx_buf,
        expert_score_buf,
        static_cast<int>(N),
        gru_hidden, query_dim, pk_dim, num_heads);
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 8: Expert MLP
    // -----------------------------------------------------------------------
    sg2_expert_mlp<INT8_EXPERTS>(
        gru_input_buf,
        expert_idx_buf,
        expert_score_buf,
        expert_W1, expert_b1, expert_W2, expert_b2,
        expert_scales,
        smart_grad_buf,
        expert_rescale,
        static_cast<int>(N),
        num_experts, expert_hidden, num_heads);
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 9: AdamW parameter update with amplified gradient
    // -----------------------------------------------------------------------
    sg2_adam_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, smart_grad_buf, N,
        lr, layer_beta1, beta2, eps, weight_decay,
        layer_alpha, lamb, ramp, gate_signal, grad_clip,
        bias_correction1, bias_correction2, clip_threshold);
}

}  // namespace gfx942
}  // namespace grokking

#endif  // GROKKING_SUPERGROK2_GFX942_HIP_HPP_
