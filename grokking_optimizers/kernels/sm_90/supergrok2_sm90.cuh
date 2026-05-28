#ifndef GROKKING_SUPERGROK2_SM90_CUH_
#define GROKKING_SUPERGROK2_SM90_CUH_

#include "common_sm90.cuh"

namespace grokking { namespace sm90 {

// ============================================================================
// SuperGrok v2 (SG2) -- Mamba-3 + 4-Head PEER + GRU Meta-Net Optimizer
//
// Architecture overview (per-parameter fused step):
//   1. Gradient validation, clipping, and radix sort by |grad|
//   2. Forward Mamba selective scan on sorted gradients
//   3. Backward Mamba selective scan on reversed sorted gradients
//   4. Unsort to original order
//   5. GRU temporal memory update
//   6. PEER 4-head product-key expert routing
//   7. Expert MLP evaluation
//   8. AdamW update with smart (amplified) gradients
// ============================================================================

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
namespace sg2_constants {
    constexpr int MAX_BLOCK_SORT_N   = 1024;  // Max N for intra-block bitonic sort
    constexpr int RADIX_BITS         = 8;
    constexpr int RADIX_BUCKETS      = 1 << RADIX_BITS;  // 256
    constexpr int MAMBA_TILE_SIZE    = 32;
    constexpr int DEFAULT_D_INNER    = 16;
    constexpr int DEFAULT_D_STATE    = 16;
    constexpr int DEFAULT_GRU_HIDDEN = 8;
    constexpr int DEFAULT_NUM_HEADS  = 4;
    constexpr int DEFAULT_PK_DIM     = 32;    // Product-key sub-dimension
    constexpr int DEFAULT_NUM_EXPERTS = 1024;  // pk_dim * pk_dim
    constexpr int DEFAULT_EXPERT_HIDDEN = 16;

    constexpr int SORT_SMEM_BYTES       = MAX_BLOCK_SORT_N * (sizeof(float) + sizeof(int));
    constexpr int MAMBA_SCAN_SMEM_BYTES = 2 * DEFAULT_D_INNER * DEFAULT_D_STATE * sizeof(float);
    constexpr int GRU_SMEM_BYTES        = 0;
    constexpr int PEER_SMEM_BYTES       = 4 * DEFAULT_PK_DIM * sizeof(float);  // Per-head top-k scratch
    constexpr int EXPERT_SMEM_BYTES     = DEFAULT_EXPERT_HIDDEN * 2 * sizeof(float);  // W1, b1 tile
}

// ---------------------------------------------------------------------------
// State struct: 7 persistent state tensors per parameter group
// ---------------------------------------------------------------------------
struct SuperGrok2State {
    float* __restrict__ exp_avg;           // [N]   Adam first moment
    float* __restrict__ exp_avg_sq;        // [N]   Adam second moment
    float* __restrict__ mu;                // [N]   EMA gradient
    float* __restrict__ sharpness;         // [N]   Gradient correction magnitude EMA
    float* __restrict__ gru_state;         // [N * gru_hidden]
    float* __restrict__ mamba_fwd_state;   // [d_inner * d_state] per param group
    float* __restrict__ mamba_bwd_state;   // [d_inner * d_state] per param group

    static constexpr int num_state_tensors() { return 7; }
    static constexpr int state_bytes_per_element(int gru_hidden, int d_inner, int d_state) {
        return 4 * sizeof(float)                     // exp_avg, exp_avg_sq, mu, sharpness
             + gru_hidden * sizeof(float)            // gru_state
             + 2 * d_inner * d_state * sizeof(float); // mamba fwd + bwd (amortized)
    }
};

// ---------------------------------------------------------------------------
// Hyperparameter struct passed to the fused kernel (avoids >30 scalar args)
// ---------------------------------------------------------------------------
struct SG2Hyperparams {
    float lr;
    float layer_beta1;
    float beta2;
    float eps;
    float effective_wd;
    float layer_alpha;
    float lamb;
    float ramp;
    float gate_signal;
    float grad_clip;
    float bias_correction1;
    float bias_correction2;
    float clip_threshold;
    int   d_inner;
    int   d_state;
    int   gru_hidden;
    int   gru_input_dim;
    int   num_heads;
    int   pk_dim;
    int   num_experts;
    int   expert_hidden;
    int   peer_input_dim;
};

// ---------------------------------------------------------------------------
// Mamba weight pointers
// ---------------------------------------------------------------------------
struct SG2MambaWeights {
    const float* __restrict__ A_log;       // [d_inner, d_state]
    const float* __restrict__ B_proj;      // [d_inner, d_state] projection
    const float* __restrict__ C_proj;      // [d_inner, d_state] projection
    const float* __restrict__ D;           // [d_inner]
    const float* __restrict__ dt_proj_W;   // [d_inner]  (simplified: per-channel dt bias)
};

// ---------------------------------------------------------------------------
// GRU weight pointers
// ---------------------------------------------------------------------------
struct SG2GRUWeights {
    const float* __restrict__ W_z;  // [gru_hidden, gru_input_dim + gru_hidden]
    const float* __restrict__ W_r;  // [gru_hidden, gru_input_dim + gru_hidden]
    const float* __restrict__ W_h;  // [gru_hidden, gru_input_dim + gru_hidden]
    const float* __restrict__ b_z;  // [gru_hidden]
    const float* __restrict__ b_r;  // [gru_hidden]
    const float* __restrict__ b_h;  // [gru_hidden]
};

// ---------------------------------------------------------------------------
// PEER weight pointers (per-head)
// ---------------------------------------------------------------------------
struct SG2PEERWeights {
    const float* __restrict__ query_W;   // [num_heads, peer_input_dim, pk_dim*2]
    const float* __restrict__ keys_A;    // [num_heads, pk_dim, pk_dim]
    const float* __restrict__ keys_B;    // [num_heads, pk_dim, pk_dim]
};

// ---------------------------------------------------------------------------
// Expert MLP weight pointers
// ---------------------------------------------------------------------------
struct SG2ExpertWeights {
    const void*  __restrict__ expert_W1;  // [num_experts, expert_hidden] (may be quantized)
    const float* __restrict__ expert_b1;  // [num_experts, expert_hidden]
    const void*  __restrict__ expert_W2;  // [num_experts, expert_hidden] (may be quantized)
    const float* __restrict__ expert_b2;  // [num_experts]
    const float* __restrict__ scales_W1;  // [num_experts] INT8/INT4 scales (nullptr if FP32)
    const float* __restrict__ scales_W2;  // [num_experts] INT8/INT4 scales (nullptr if FP32)
    float rescale;                         // Output rescaling factor
};

// ============================================================================
// Quantized expert weight dequantization
// ============================================================================

template <bool INT8_EXPERTS>
__forceinline__ __device__
float dequant_expert_weight(const void* data, float scale, int idx);

// FP32 passthrough (INT8_EXPERTS = false)
template <>
__forceinline__ __device__
float dequant_expert_weight<false>(const void* data, float scale, int idx) {
    return __ldg(&reinterpret_cast<const float*>(data)[idx]);
}

// INT8 dequantization
template <>
__forceinline__ __device__
float dequant_expert_weight<true>(const void* data, float scale, int idx) {
    int8_t q = __ldg(&reinterpret_cast<const int8_t*>(data)[idx]);
    return static_cast<float>(q) * scale;
}

// INT4 dequantization (packed: 2 weights per byte, low nibble first)
__forceinline__ __device__
float dequant_expert_weight_int4(const void* data, float scale, int idx) {
    uint8_t packed = __ldg(&reinterpret_cast<const uint8_t*>(data)[idx / 2]);
    int8_t val;
    if (idx & 1) {
        val = static_cast<int8_t>((packed >> 4) & 0x0F);
    } else {
        val = static_cast<int8_t>(packed & 0x0F);
    }
    // Sign-extend from 4 bits
    if (val & 0x08) val |= 0xF0;
    return static_cast<float>(val) * scale;
}

// ============================================================================
// 1. sg2_radix_sort_by_abs -- Sort gradient elements by |grad| magnitude
// ============================================================================

// Shared-memory bitonic sort for N <= 1024 (intra-block)
__forceinline__ __device__
void sg2_bitonic_sort_smem(
    float* __restrict__       keys_out,     // [N] sorted |grad| values
    int*   __restrict__       indices_out,  // [N] original indices
    const float* __restrict__ grads,        // [N] input gradients
    int                       N,
    float* __restrict__       smem_keys,    // shared mem [N]
    int*   __restrict__       smem_idx      // shared mem [N]
) {
    // Phase 1: Load |grad| and indices into shared memory
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        smem_keys[i] = fabsf(__ldg(&grads[i]));
        smem_idx[i]  = i;
    }
    // Pad to next power of 2 with +inf (sorts to end)
    int N_padded = 1;
    while (N_padded < N) N_padded <<= 1;
    for (int i = N + threadIdx.x; i < N_padded; i += blockDim.x) {
        smem_keys[i] = __int_as_float(0x7F800000);  // +inf
        smem_idx[i]  = -1;
    }
    __syncthreads();

    // Phase 2: Bitonic sort (descending -- largest |grad| first)
    for (int size = 2; size <= N_padded; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < N_padded / 2; i += blockDim.x) {
                int block_id = i / (stride);
                int offset   = i % (stride);
                int left     = 2 * (block_id / (size / stride)) * (size / 2)
                             + (block_id % (size / stride)) * stride + offset;

                // Simplified: pair indices for bitonic network
                int grp     = i / (size / 2);
                int pos     = i % (size / 2);
                int half    = stride;
                int lo_bit  = pos % half;
                int blk_off = (pos / half) * (2 * half);
                int idx0    = grp * size + blk_off + lo_bit;
                int idx1    = idx0 + half;

                if (idx1 < N_padded) {
                    // Descending: swap if keys[idx0] < keys[idx1]
                    bool should_swap = smem_keys[idx0] < smem_keys[idx1];
                    if (should_swap) {
                        float tmp_k     = smem_keys[idx0];
                        smem_keys[idx0] = smem_keys[idx1];
                        smem_keys[idx1] = tmp_k;
                        int tmp_i       = smem_idx[idx0];
                        smem_idx[idx0]  = smem_idx[idx1];
                        smem_idx[idx1]  = tmp_i;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Phase 3: Write back (only valid entries)
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        keys_out[i]    = smem_keys[i];
        indices_out[i] = smem_idx[i];
    }
}

// Multi-block radix sort for large N using global memory scratch
__forceinline__ __device__
void sg2_radix_sort_global(
    float* __restrict__       keys_out,
    int*   __restrict__       indices_out,
    const float* __restrict__ grads,
    int                       N,
    float* __restrict__       scratch_keys,  // [N] global scratch
    int*   __restrict__       scratch_idx,   // [N] global scratch
    int*   __restrict__       histograms     // [gridDim.x * 256] global scratch
) {
    // Single-pass radix sort on float-as-uint32 with sign-flip for descending
    // Each block handles a tile of elements, computes local histogram,
    // then scatters using prefix-summed offsets.

    const int TILE = (N + gridDim.x - 1) / gridDim.x;
    const int tile_start = blockIdx.x * TILE;
    const int tile_end   = min(tile_start + TILE, N);

    // Initialize: convert |grad| to radix-sortable uint32 (descending)
    for (int i = tile_start + threadIdx.x; i < tile_end; i += blockDim.x) {
        float absval = fabsf(__ldg(&grads[i]));
        // Flip bits for descending sort: negate the float-as-uint representation
        unsigned int key = __float_as_uint(absval);
        key = ~key;  // Descending: complement bits
        scratch_keys[i] = __uint_as_float(key);
        scratch_idx[i]  = i;
    }
    __syncthreads();

    // For each radix pass (4 passes for 32-bit keys, 8-bit radix)
    for (int pass = 0; pass < 4; ++pass) {
        int shift = pass * sg2_constants::RADIX_BITS;

        // Local histogram
        int* my_hist = histograms + blockIdx.x * sg2_constants::RADIX_BUCKETS;
        for (int b = threadIdx.x; b < sg2_constants::RADIX_BUCKETS; b += blockDim.x) {
            my_hist[b] = 0;
        }
        __syncthreads();

        for (int i = tile_start + threadIdx.x; i < tile_end; i += blockDim.x) {
            unsigned int key = __float_as_uint(scratch_keys[i]);
            int bucket = (key >> shift) & (sg2_constants::RADIX_BUCKETS - 1);
            atomicAdd(&my_hist[bucket], 1);
        }
        __syncthreads();

        // Compute scatter offsets using exclusive prefix sum on histogram
        // (Simplified: single-block prefix sum within tile)
        if (threadIdx.x == 0) {
            int running = tile_start;
            for (int b = 0; b < sg2_constants::RADIX_BUCKETS; ++b) {
                int count = my_hist[b];
                my_hist[b] = running;
                running += count;
            }
        }
        __syncthreads();

        // Scatter into keys_out / indices_out using atomicAdd on offsets
        for (int i = tile_start + threadIdx.x; i < tile_end; i += blockDim.x) {
            unsigned int key = __float_as_uint(scratch_keys[i]);
            int bucket = (key >> shift) & (sg2_constants::RADIX_BUCKETS - 1);
            int dst = atomicAdd(&my_hist[bucket], 1);
            if (dst < N) {
                keys_out[dst]    = scratch_keys[i];
                indices_out[dst] = scratch_idx[i];
            }
        }
        __syncthreads();

        // Swap buffers: sorted output becomes input for next pass
        for (int i = tile_start + threadIdx.x; i < tile_end; i += blockDim.x) {
            scratch_keys[i] = keys_out[i];
            scratch_idx[i]  = indices_out[i];
        }
        __syncthreads();
    }

    // Final: convert keys back to float magnitudes
    for (int i = tile_start + threadIdx.x; i < tile_end; i += blockDim.x) {
        unsigned int key = __float_as_uint(keys_out[i]);
        key = ~key;  // Reverse the complement
        keys_out[i] = __uint_as_float(key);
    }
}

// Top-level sort dispatcher
__forceinline__ __device__
void sg2_radix_sort_by_abs(
    float* __restrict__       keys_out,
    int*   __restrict__       indices_out,
    const float* __restrict__ grads,
    int                       N,
    float* __restrict__       smem_or_scratch_keys,
    int*   __restrict__       smem_or_scratch_idx,
    int*   __restrict__       histograms
) {
    if (N <= sg2_constants::MAX_BLOCK_SORT_N) {
        sg2_bitonic_sort_smem(
            keys_out, indices_out, grads, N,
            smem_or_scratch_keys,
            smem_or_scratch_idx);
    } else {
        sg2_radix_sort_global(
            keys_out, indices_out, grads, N,
            smem_or_scratch_keys,
            smem_or_scratch_idx,
            histograms);
    }
}

// ============================================================================
// 2. sg2_mamba_scan_forward -- Single-direction Mamba selective scan
//
// Simplified sequential-within-warp version. Each thread processes one element
// in the parameter-gradient sequence. State is maintained in shared memory.
//
// State update per element i:
//   dt = softplus(dt_proj_W[c] + input_dependent_bias)
//   A_bar = exp(-exp(A_log[c,n]) * dt)  for each state dim n
//   B_bar = B_proj[c,n] * dt * x[i]
//   h[c,n] = A_bar * h[c,n] + B_bar
//   y[i] += C_proj[c,n]^T @ h[:,n] + D[c] * x[i]
// ============================================================================

__forceinline__ __device__
void sg2_mamba_scan_forward(
    float* __restrict__       output,     // [N] mamba-processed gradient signal
    const float* __restrict__ input,      // [N] (sorted) gradient magnitudes
    float* __restrict__       state,      // [d_inner * d_state] persistent scan state
    const SG2MambaWeights&    weights,
    int                       N,
    int                       d_inner,
    int                       d_state,
    float* __restrict__       smem_state  // [d_inner * d_state] shared memory
) {
    // Load persistent state into shared memory
    for (int j = threadIdx.x; j < d_inner * d_state; j += blockDim.x) {
        smem_state[j] = state[j];
    }
    __syncthreads();

    // Sequential scan over the N elements
    // Each thread cooperates: thread t processes elements t, t+blockDim, ...
    // but the scan is inherently sequential, so we serialize by element index.
    // For practical N < 100K, we tile: each warp processes a channel slice.

    for (int i = 0; i < N; ++i) {
        float x_val = input[i];
        float y_val = 0.0f;

        // Each thread handles a subset of (d_inner, d_state) pairs
        for (int cn = threadIdx.x; cn < d_inner * d_state; cn += blockDim.x) {
            int c = cn / d_state;
            int n = cn % d_state;

            // Discretization
            float dt_bias = __ldg(&weights.dt_proj_W[c]);
            // Softplus: log(1 + exp(x))
            float dt = log1pf(__expf(dt_bias + x_val * 0.01f));  // input-dependent modulation

            float A_val = __expf(__ldg(&weights.A_log[c * d_state + n]));
            float A_bar = __expf(-A_val * dt);
            float B_val = __ldg(&weights.B_proj[c * d_state + n]);
            float B_bar = B_val * dt;
            float C_val = __ldg(&weights.C_proj[c * d_state + n]);

            // State update
            float h_old = smem_state[cn];
            float h_new = A_bar * h_old + B_bar * x_val;
            smem_state[cn] = h_new;

            // Output accumulation: each (c,n) contributes C_val * h_new / d_state
            float contrib = C_val * h_new;
            // Warp-reduce the contribution for this element
            atomicAdd(&output[i], contrib);
        }
        __syncthreads();

        // Add skip connection: D * x
        if (threadIdx.x == 0) {
            float d_skip = 0.0f;
            for (int c = 0; c < d_inner; ++c) {
                d_skip += __ldg(&weights.D[c]);
            }
            output[i] += d_skip * x_val / static_cast<float>(d_inner);
        }
        __syncthreads();
    }

    // Write updated state back to global memory
    for (int j = threadIdx.x; j < d_inner * d_state; j += blockDim.x) {
        state[j] = smem_state[j];
    }
}

// Backward scan: same structure but processes input in reverse order
__forceinline__ __device__
void sg2_mamba_scan_backward(
    float* __restrict__       output,
    const float* __restrict__ input,
    float* __restrict__       state,
    const SG2MambaWeights&    weights,
    int                       N,
    int                       d_inner,
    int                       d_state,
    float* __restrict__       smem_state
) {
    // Load persistent state into shared memory
    for (int j = threadIdx.x; j < d_inner * d_state; j += blockDim.x) {
        smem_state[j] = state[j];
    }
    __syncthreads();

    // Process in reverse order
    for (int i = N - 1; i >= 0; --i) {
        float x_val = input[i];
        float y_val = 0.0f;

        for (int cn = threadIdx.x; cn < d_inner * d_state; cn += blockDim.x) {
            int c = cn / d_state;
            int n = cn % d_state;

            float dt_bias = __ldg(&weights.dt_proj_W[c]);
            float dt = log1pf(__expf(dt_bias + x_val * 0.01f));

            float A_val = __expf(__ldg(&weights.A_log[c * d_state + n]));
            float A_bar = __expf(-A_val * dt);
            float B_val = __ldg(&weights.B_proj[c * d_state + n]);
            float B_bar = B_val * dt;
            float C_val = __ldg(&weights.C_proj[c * d_state + n]);

            float h_old = smem_state[cn];
            float h_new = A_bar * h_old + B_bar * x_val;
            smem_state[cn] = h_new;

            float contrib = C_val * h_new;
            atomicAdd(&output[i], contrib);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float d_skip = 0.0f;
            for (int c = 0; c < d_inner; ++c) {
                d_skip += __ldg(&weights.D[c]);
            }
            output[i] += d_skip * x_val / static_cast<float>(d_inner);
        }
        __syncthreads();
    }

    // Write updated state back
    for (int j = threadIdx.x; j < d_inner * d_state; j += blockDim.x) {
        state[j] = smem_state[j];
    }
}

// ============================================================================
// 3. sg2_gru_update -- Per-element GRU temporal memory
//
// For each element i:
//   concat = [input[i, :gru_input_dim], h_old[i, :gru_hidden]]
//   z = sigmoid(W_z @ concat + b_z)
//   r = sigmoid(W_r @ concat + b_r)
//   h_tilde = tanh(W_h @ [input[i,:], r * h_old[i,:]] + b_h)
//   h_new[i,:] = (1 - z) * h_old[i,:] + z * h_tilde
// ============================================================================

__forceinline__ __device__
void sg2_gru_update(
    float* __restrict__       h_new,      // [N, gru_hidden]
    const float* __restrict__ input,      // [N, gru_input_dim]
    const float* __restrict__ h_old,      // [N, gru_hidden]
    const SG2GRUWeights&      weights,
    int                       N,
    int                       gru_hidden,
    int                       gru_input_dim
) {
    const int concat_dim = gru_input_dim + gru_hidden;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * (int64_t)blockDim.x) {

        const float* inp_i = input + i * gru_input_dim;
        const float* h_old_i = h_old + i * gru_hidden;
        float* h_new_i = h_new + i * gru_hidden;

        for (int j = 0; j < gru_hidden; ++j) {
            // Compute z_j = sigmoid(W_z[j,:] @ [input, h_old] + b_z[j])
            float z_val = __ldg(&weights.b_z[j]);
            float r_val = __ldg(&weights.b_r[j]);

            for (int k = 0; k < gru_input_dim; ++k) {
                float inp_k = inp_i[k];
                z_val += __ldg(&weights.W_z[j * concat_dim + k]) * inp_k;
                r_val += __ldg(&weights.W_r[j * concat_dim + k]) * inp_k;
            }
            for (int k = 0; k < gru_hidden; ++k) {
                float h_k = h_old_i[k];
                z_val += __ldg(&weights.W_z[j * concat_dim + gru_input_dim + k]) * h_k;
                r_val += __ldg(&weights.W_r[j * concat_dim + gru_input_dim + k]) * h_k;
            }

            // Sigmoid gates
            z_val = 1.0f / (1.0f + __expf(-z_val));
            r_val = 1.0f / (1.0f + __expf(-r_val));

            // h_tilde_j = tanh(W_h[j,:] @ [input, r * h_old] + b_h[j])
            float h_tilde = __ldg(&weights.b_h[j]);
            for (int k = 0; k < gru_input_dim; ++k) {
                h_tilde += __ldg(&weights.W_h[j * concat_dim + k]) * inp_i[k];
            }
            for (int k = 0; k < gru_hidden; ++k) {
                float rh = r_val * h_old_i[k];
                h_tilde += __ldg(&weights.W_h[j * concat_dim + gru_input_dim + k]) * rh;
            }
            h_tilde = tanhf(h_tilde);

            // GRU output
            float h_old_j = h_old_i[j];
            h_new_i[j] = (1.0f - z_val) * h_old_j + z_val * h_tilde;
        }
    }
}

// ============================================================================
// 4. sg2_peer_routing -- 4-Head Product-key Expert Routing
//
// For each element i and each head h:
//   q = query_W[h] @ input[i,:]          shape: [pk_dim*2]
//   q_a = q[:pk_dim], q_b = q[pk_dim:]
//   scores_a = q_a @ keys_A[h]^T         shape: [pk_dim]
//   scores_b = q_b @ keys_B[h]^T         shape: [pk_dim]
//   idx_a = argmax(scores_a)
//   idx_b = argmax(scores_b)
//   expert_idx = idx_a * pk_dim + idx_b
//   routing_weight = softmax_score (max_a + max_b normalized)
// ============================================================================

__forceinline__ __device__
void sg2_peer_routing(
    int*   __restrict__       expert_indices,   // [N, num_heads]
    float* __restrict__       routing_weights,  // [N, num_heads]
    const float* __restrict__ peer_input,       // [N, peer_input_dim]
    const SG2PEERWeights&     weights,
    int                       N,
    int                       num_heads,
    int                       peer_input_dim,
    int                       pk_dim,
    float* __restrict__       smem_scores       // [2 * pk_dim] shared memory per thread
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * (int64_t)blockDim.x) {

        const float* inp_i = peer_input + i * peer_input_dim;
        float total_score = 0.0f;
        float head_scores[4];  // max 4 heads, stored in registers

        for (int h = 0; h < num_heads; ++h) {
            // Per-head query projection offset
            const int qw_offset = h * peer_input_dim * pk_dim * 2;

            // Compute q_a and q_b via matmul with query_W
            float max_a = -1e30f;
            int   idx_a = 0;
            float max_b = -1e30f;
            int   idx_b = 0;

            // q_a: first pk_dim output dimensions
            for (int d = 0; d < pk_dim; ++d) {
                float q_val = 0.0f;
                for (int k = 0; k < peer_input_dim; ++k) {
                    q_val += __ldg(&weights.query_W[qw_offset + d * peer_input_dim + k])
                           * inp_i[k];
                }
                // Score against key_A[h][d, :] -- dot product with key column
                float score_a = 0.0f;
                const int ka_offset = h * pk_dim * pk_dim;
                for (int m = 0; m < pk_dim; ++m) {
                    // keys_A[h] is [pk_dim, pk_dim]; row d, compute dot with q_val
                    // Simplified: score = q_val * keys_A[h][d][d] (diagonal approx)
                    // Full: score_a[d] = sum_k q_a[k] * keys_A[h][k][d]
                    score_a += 0.0f;  // placeholder, computed below
                }
                // Actually: compute scores_a = q_a @ keys_A^T
                // q_a is pk_dim-dimensional; keys_A[h] is [pk_dim, pk_dim]
                // scores_a[d] = sum_k q_a[k] * keys_A[h][k][d]
                // We need to accumulate q_a first, then multiply.
                // Restructure: accumulate full q_a vector first.
                (void)score_a;  // will compute below
                (void)q_val;
            }

            // Restructured: compute full q vector, then product-key lookup
            float q_a_vec[64];  // max pk_dim = 64
            float q_b_vec[64];
            int pk_d = (pk_dim < 64) ? pk_dim : 64;

            // Compute q = query_W[h] @ input[i]
            for (int d = 0; d < pk_d; ++d) {
                float qa = 0.0f;
                float qb = 0.0f;
                for (int k = 0; k < peer_input_dim; ++k) {
                    qa += __ldg(&weights.query_W[qw_offset + d * peer_input_dim + k])
                        * inp_i[k];
                    qb += __ldg(&weights.query_W[qw_offset + (pk_d + d) * peer_input_dim + k])
                        * inp_i[k];
                }
                q_a_vec[d] = qa;
                q_b_vec[d] = qb;
            }

            // scores_a[m] = q_a @ keys_A[h][:, m] = sum_d q_a[d] * keys_A[h][d][m]
            max_a = -1e30f;
            idx_a = 0;
            const int ka_base = h * pk_dim * pk_dim;
            for (int m = 0; m < pk_d; ++m) {
                float score = 0.0f;
                for (int d = 0; d < pk_d; ++d) {
                    score += q_a_vec[d] * __ldg(&weights.keys_A[ka_base + d * pk_dim + m]);
                }
                if (score > max_a) {
                    max_a = score;
                    idx_a = m;
                }
            }

            // scores_b[m] = q_b @ keys_B[h][:, m]
            max_b = -1e30f;
            idx_b = 0;
            const int kb_base = h * pk_dim * pk_dim;
            for (int m = 0; m < pk_d; ++m) {
                float score = 0.0f;
                for (int d = 0; d < pk_d; ++d) {
                    score += q_b_vec[d] * __ldg(&weights.keys_B[kb_base + d * pk_dim + m]);
                }
                if (score > max_b) {
                    max_b = score;
                    idx_b = m;
                }
            }

            int expert_idx = idx_a * pk_dim + idx_b;
            float head_score = max_a + max_b;

            expert_indices[i * num_heads + h] = expert_idx;
            head_scores[h] = head_score;
            total_score += __expf(head_score);
        }

        // Softmax normalization of routing weights across heads
        float inv_total = 1.0f / (total_score + 1e-8f);
        for (int h = 0; h < num_heads; ++h) {
            routing_weights[i * num_heads + h] = __expf(head_scores[h]) * inv_total;
        }
    }
}

// ============================================================================
// 5. sg2_expert_mlp -- Expert MLP evaluation with SMEM expert weights
//
// For each element i:
//   out[i] = rescale * mean_over_heads(
//     routing_weight[h] * (W2[expert_h] @ relu(W1[expert_h] * grad[i] + b1[expert_h]) + b2[expert_h])
//   )
// ============================================================================

template <bool INT8_EXPERTS = false>
__forceinline__ __device__
void sg2_expert_mlp(
    float* __restrict__       output,          // [N] expert MLP output
    const float* __restrict__ input,           // [N] gradient values
    const int*   __restrict__ expert_indices,  // [N, num_heads]
    const float* __restrict__ routing_weights, // [N, num_heads]
    const SG2ExpertWeights&   expert_weights,
    int                       N,
    int                       num_heads,
    int                       num_experts,
    int                       expert_hidden
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * (int64_t)blockDim.x) {

        float grad_val = input[i];
        float accum = 0.0f;

        for (int h = 0; h < num_heads; ++h) {
            int eidx = expert_indices[i * num_heads + h];
            float rw = routing_weights[i * num_heads + h];

            // Bounds check expert index
            if (eidx < 0 || eidx >= num_experts) continue;

            // Expert MLP: hidden = relu(W1[eidx] * grad + b1[eidx])
            //             out_h  = W2[eidx] @ hidden + b2[eidx]
            float expert_out = __ldg(&expert_weights.expert_b2[eidx]);

            for (int j = 0; j < expert_hidden; ++j) {
                // W1 shape: [num_experts, expert_hidden]
                // For SG2, W1 is a scalar-input MLP: W1[eidx, j] * grad + b1[eidx, j]
                int w1_idx = eidx * expert_hidden + j;
                float w1_val;
                if constexpr (INT8_EXPERTS) {
                    float scale = __ldg(&expert_weights.scales_W1[eidx]);
                    w1_val = dequant_expert_weight<true>(expert_weights.expert_W1, scale, w1_idx);
                } else {
                    w1_val = dequant_expert_weight<false>(expert_weights.expert_W1, 0.0f, w1_idx);
                }
                float b1_val = __ldg(&expert_weights.expert_b1[w1_idx]);

                float hidden = w1_val * grad_val + b1_val;
                hidden = fmaxf(hidden, 0.0f);  // ReLU

                // W2 shape: [num_experts, expert_hidden]
                int w2_idx = eidx * expert_hidden + j;
                float w2_val;
                if constexpr (INT8_EXPERTS) {
                    float scale = __ldg(&expert_weights.scales_W2[eidx]);
                    w2_val = dequant_expert_weight<true>(expert_weights.expert_W2, scale, w2_idx);
                } else {
                    w2_val = dequant_expert_weight<false>(expert_weights.expert_W2, 0.0f, w2_idx);
                }

                expert_out += w2_val * hidden;
            }

            accum += rw * expert_out;
        }

        output[i] = expert_weights.rescale * accum;
    }
}

// INT4 expert specialization
__forceinline__ __device__
void sg2_expert_mlp_int4(
    float* __restrict__       output,
    const float* __restrict__ input,
    const int*   __restrict__ expert_indices,
    const float* __restrict__ routing_weights,
    const SG2ExpertWeights&   expert_weights,
    int                       N,
    int                       num_heads,
    int                       num_experts,
    int                       expert_hidden
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * (int64_t)blockDim.x) {

        float grad_val = input[i];
        float accum = 0.0f;

        for (int h = 0; h < num_heads; ++h) {
            int eidx = expert_indices[i * num_heads + h];
            float rw = routing_weights[i * num_heads + h];
            if (eidx < 0 || eidx >= num_experts) continue;

            float expert_out = __ldg(&expert_weights.expert_b2[eidx]);

            for (int j = 0; j < expert_hidden; ++j) {
                int w1_idx = eidx * expert_hidden + j;
                float scale1 = __ldg(&expert_weights.scales_W1[eidx]);
                float w1_val = dequant_expert_weight_int4(expert_weights.expert_W1, scale1, w1_idx);
                float b1_val = __ldg(&expert_weights.expert_b1[w1_idx]);

                float hidden = fmaxf(w1_val * grad_val + b1_val, 0.0f);

                int w2_idx = eidx * expert_hidden + j;
                float scale2 = __ldg(&expert_weights.scales_W2[eidx]);
                float w2_val = dequant_expert_weight_int4(expert_weights.expert_W2, scale2, w2_idx);

                expert_out += w2_val * hidden;
            }

            accum += rw * expert_out;
        }

        output[i] = expert_weights.rescale * accum;
    }
}

// ============================================================================
// 6. sg2_adam_update -- Final AdamW step with amplified (smart) gradients
//
// Per-element:
//   mu = layer_alpha * mu + (1 - layer_alpha) * grad
//   effective_grad = grad + lamb * ramp * gate_signal * (smart_grad - grad)
//   m = beta1 * m + (1 - beta1) * effective_grad
//   v = beta2 * v + (1 - beta2) * effective_grad^2
//   m_hat = m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)
//   param -= lr * (m_hat / (sqrt(v_hat) + eps) + effective_wd * param)
//   sharpness = layer_alpha * sharpness + (1 - layer_alpha) * |smart_grad - grad|
// ============================================================================

template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void sg2_adam_update(
    ParamT* __restrict__       params,
    const float* __restrict__  grads_f32,    // Already-converted FP32 grads
    const float* __restrict__  smart_grads,  // Output from expert MLP pipeline
    SuperGrok2State            state,
    int64_t                    N,
    float lr,
    float layer_beta1,
    float beta2,
    float eps,
    float effective_wd,
    float layer_alpha,
    float lamb,
    float ramp,
    float gate_signal,
    float bias_correction1,
    float bias_correction2
) {
    const float blend = ramp * lamb * gate_signal;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * (int64_t)blockDim.x) {

        float g = grads_f32[i];
        float smart_g = smart_grads[i];

        // NaN handling on the smart gradient
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__isnanf(smart_g)) smart_g = g;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__isnanf(smart_g)) continue;
        }

        // EMA gradient update
        float mu_old = state.mu[i];
        float mu_val = layer_alpha * mu_old + (1.0f - layer_alpha) * g;

        // Blend raw and smart gradient
        float effective_g = g + blend * (smart_g - g);

        // AdamW step
        float p_f   = to_float<ParamT>(params[i]);
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = layer_beta1 * m_old + (1.0f - layer_beta1) * effective_g;
        float v = beta2 * v_old + (1.0f - beta2) * effective_g * effective_g;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        float denom  = sqrtf(v_hat) + eps;
        float update = m_hat / denom + effective_wd * p_f;

        p_f -= lr * update;

        // Update sharpness (running EMA of correction magnitude)
        float correction = smart_g - g;
        float sharp_old = state.sharpness[i];
        float new_sharp = layer_alpha * sharp_old + (1.0f - layer_alpha) * fabsf(correction);

        // Writeback
        state.exp_avg[i]    = m;
        state.exp_avg_sq[i] = v;
        state.mu[i]         = mu_val;
        state.sharpness[i]  = new_sharp;
        params[i]           = from_float<ParamT>(p_f);
    }
}

// ============================================================================
// 7. sg2_fused_step -- Top-level global kernel
//
// Chains all SG2 components for a single parameter tensor in one launch.
// Requires sufficient shared memory and scratch global memory.
// ============================================================================

template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP, bool INT8_EXPERTS = false>
__global__ void sg2_fused_step(
    // Parameter data
    ParamT* __restrict__       params,
    const ParamT* __restrict__ grads,
    int64_t                    N,
    // Optimizer state
    SuperGrok2State            state,
    // Hyperparameters
    SG2Hyperparams             hparams,
    // Component weights
    SG2MambaWeights            mamba_fwd_weights,
    SG2MambaWeights            mamba_bwd_weights,
    SG2GRUWeights              gru_weights,
    SG2PEERWeights             peer_weights,
    SG2ExpertWeights           expert_weights,
    // Scratch buffers (caller-allocated global memory)
    float* __restrict__        scratch_grads_f32,   // [N]
    float* __restrict__        scratch_sorted_keys,  // [N]
    int*   __restrict__        scratch_sorted_idx,   // [N]
    float* __restrict__        scratch_mamba_fwd,     // [N]
    float* __restrict__        scratch_mamba_bwd,     // [N]
    float* __restrict__        scratch_mamba_combined, // [N]
    float* __restrict__        scratch_gru_input,     // [N, gru_input_dim]
    float* __restrict__        scratch_gru_out,       // [N, gru_hidden]
    int*   __restrict__        scratch_expert_idx,    // [N, num_heads]
    float* __restrict__        scratch_routing_wts,   // [N, num_heads]
    float* __restrict__        scratch_expert_out,    // [N]
    float* __restrict__        scratch_smart_grads,   // [N]
    int*   __restrict__        scratch_radix_hist     // [gridDim.x * 256]
) {
    extern __shared__ char shared_mem[];

    // ====================================================================
    // Step (a): Validate and clip gradients, convert to FP32
    // ====================================================================
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * (int64_t)blockDim.x) {
        float g = to_float<ParamT>(__ldg(&grads[i]));

        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__isnanf(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__isnanf(g)) { scratch_grads_f32[i] = 0.0f; continue; }
        }

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -hparams.clip_threshold), hparams.clip_threshold);
        }

        g = fminf(fmaxf(g, -hparams.grad_clip), hparams.grad_clip);
        scratch_grads_f32[i] = g;
    }
    __syncthreads();

    // For the remaining cooperative steps, only block 0 performs the
    // sequential-scan portions. Other blocks handle embarrassingly parallel work.

    // ====================================================================
    // Step (b): Sort gradients by |grad| magnitude (descending)
    // ====================================================================
    if (N <= sg2_constants::MAX_BLOCK_SORT_N && blockIdx.x == 0) {
        float* smem_keys = reinterpret_cast<float*>(shared_mem);
        int*   smem_idx  = reinterpret_cast<int*>(shared_mem + N * sizeof(float));

        sg2_bitonic_sort_smem(
            scratch_sorted_keys, scratch_sorted_idx,
            scratch_grads_f32, static_cast<int>(N),
            smem_keys, smem_idx);
    } else if (N > sg2_constants::MAX_BLOCK_SORT_N) {
        sg2_radix_sort_global(
            scratch_sorted_keys, scratch_sorted_idx,
            scratch_grads_f32, static_cast<int>(N),
            scratch_sorted_keys,  // in-place scratch
            scratch_sorted_idx,
            scratch_radix_hist);
    }
    // Grid-wide barrier via cooperative groups would be needed here;
    // in practice the fused kernel is launched with a single block for
    // small-to-medium parameters, or split into multi-kernel for large.
    __syncthreads();

    // ====================================================================
    // Step (c): Forward Mamba scan on sorted gradients
    // ====================================================================
    if (blockIdx.x == 0) {
        // Initialize output buffer
        for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
            scratch_mamba_fwd[i] = 0.0f;
        }
        __syncthreads();

        float* mamba_smem = reinterpret_cast<float*>(shared_mem);
        sg2_mamba_scan_forward(
            scratch_mamba_fwd,
            scratch_sorted_keys,
            state.mamba_fwd_state,
            mamba_fwd_weights,
            static_cast<int>(N),
            hparams.d_inner,
            hparams.d_state,
            mamba_smem);
    }
    __syncthreads();

    // ====================================================================
    // Step (d): Backward Mamba scan on reversed sorted gradients
    // ====================================================================
    if (blockIdx.x == 0) {
        for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
            scratch_mamba_bwd[i] = 0.0f;
        }
        __syncthreads();

        float* mamba_smem = reinterpret_cast<float*>(shared_mem);
        sg2_mamba_scan_backward(
            scratch_mamba_bwd,
            scratch_sorted_keys,
            state.mamba_bwd_state,
            mamba_bwd_weights,
            static_cast<int>(N),
            hparams.d_inner,
            hparams.d_state,
            mamba_smem);
    }
    __syncthreads();

    // ====================================================================
    // Step (e): Combine forward + backward and unsort to original order
    // ====================================================================
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * (int64_t)blockDim.x) {
        // Combine bidirectional Mamba outputs (mean)
        float combined = 0.5f * (scratch_mamba_fwd[i] + scratch_mamba_bwd[i]);
        scratch_mamba_combined[i] = combined;
    }
    __syncthreads();

    // Unsort: write combined signal back to original element order
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * (int64_t)blockDim.x) {
        int orig_idx = scratch_sorted_idx[i];
        if (orig_idx >= 0 && orig_idx < N) {
            // Build GRU input: [grad, mamba_signal, sharpness, mu]
            float g_val = scratch_grads_f32[orig_idx];
            float mamba_val = scratch_mamba_combined[i];
            float sharp_val = state.sharpness[orig_idx];
            float mu_val = state.mu[orig_idx];

            // Pack into gru_input (dim = gru_input_dim, typically 4)
            int gid = hparams.gru_input_dim;
            scratch_gru_input[orig_idx * gid + 0] = g_val;
            if (gid > 1) scratch_gru_input[orig_idx * gid + 1] = mamba_val;
            if (gid > 2) scratch_gru_input[orig_idx * gid + 2] = sharp_val;
            if (gid > 3) scratch_gru_input[orig_idx * gid + 3] = mu_val;
        }
    }
    __syncthreads();

    // ====================================================================
    // Step (f): GRU temporal memory update
    // ====================================================================
    sg2_gru_update(
        scratch_gru_out,
        scratch_gru_input,
        state.gru_state,
        static_cast<int>(N),
        hparams.gru_hidden,
        hparams.gru_input_dim);
    __syncthreads();

    // Copy GRU output to persistent state
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N * hparams.gru_hidden;
         i += gridDim.x * (int64_t)blockDim.x) {
        state.gru_state[i] = scratch_gru_out[i];
    }
    __syncthreads();

    // ====================================================================
    // Step (g): PEER expert routing
    //
    // Build PEER input from GRU output (reduce gru_hidden -> peer_input_dim)
    // For simplicity, use first peer_input_dim dims of GRU state
    // ====================================================================
    // Construct peer_input from GRU hidden state
    float* peer_input_buf = scratch_gru_input;  // Reuse buffer
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * (int64_t)blockDim.x) {
        for (int d = 0; d < hparams.peer_input_dim; ++d) {
            if (d < hparams.gru_hidden) {
                peer_input_buf[i * hparams.peer_input_dim + d] =
                    scratch_gru_out[i * hparams.gru_hidden + d];
            } else {
                // Pad with gradient-derived features
                peer_input_buf[i * hparams.peer_input_dim + d] =
                    scratch_grads_f32[i];
            }
        }
    }
    __syncthreads();

    float* peer_smem = reinterpret_cast<float*>(shared_mem);
    sg2_peer_routing(
        scratch_expert_idx,
        scratch_routing_wts,
        peer_input_buf,
        peer_weights,
        static_cast<int>(N),
        hparams.num_heads,
        hparams.peer_input_dim,
        hparams.pk_dim,
        peer_smem);
    __syncthreads();

    // ====================================================================
    // Step (h): Expert MLP evaluation
    // ====================================================================
    if constexpr (INT8_EXPERTS) {
        sg2_expert_mlp<true>(
            scratch_expert_out,
            scratch_grads_f32,
            scratch_expert_idx,
            scratch_routing_wts,
            expert_weights,
            static_cast<int>(N),
            hparams.num_heads,
            hparams.num_experts,
            hparams.expert_hidden);
    } else {
        sg2_expert_mlp<false>(
            scratch_expert_out,
            scratch_grads_f32,
            scratch_expert_idx,
            scratch_routing_wts,
            expert_weights,
            static_cast<int>(N),
            hparams.num_heads,
            hparams.num_experts,
            hparams.expert_hidden);
    }
    __syncthreads();

    // Build smart gradients: original grad + expert correction
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * (int64_t)blockDim.x) {
        scratch_smart_grads[i] = scratch_grads_f32[i] + scratch_expert_out[i];
    }
    __syncthreads();

    // ====================================================================
    // Step (i): AdamW update with smart gradients
    // ====================================================================
    sg2_adam_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params,
        scratch_grads_f32,
        scratch_smart_grads,
        state,
        N,
        hparams.lr,
        hparams.layer_beta1,
        hparams.beta2,
        hparams.eps,
        hparams.effective_wd,
        hparams.layer_alpha,
        hparams.lamb,
        hparams.ramp,
        hparams.gate_signal,
        hparams.bias_correction1,
        hparams.bias_correction2);
}

// ============================================================================
// Individual __global__ launchers for autotuner testing
// ============================================================================

// Sort launcher
__global__ void sg2_sort_kernel(
    float* __restrict__       keys_out,
    int*   __restrict__       indices_out,
    const float* __restrict__ grads,
    int                       N,
    float* __restrict__       scratch_keys,
    int*   __restrict__       scratch_idx,
    int*   __restrict__       histograms
) {
    extern __shared__ char smem[];
    if (N <= sg2_constants::MAX_BLOCK_SORT_N && blockIdx.x == 0) {
        float* sk = reinterpret_cast<float*>(smem);
        int*   si = reinterpret_cast<int*>(smem + N * sizeof(float));
        sg2_bitonic_sort_smem(keys_out, indices_out, grads, N, sk, si);
    } else {
        sg2_radix_sort_global(keys_out, indices_out, grads, N,
                              scratch_keys, scratch_idx, histograms);
    }
}

// Mamba forward scan launcher
__global__ void sg2_mamba_forward_kernel(
    float* __restrict__       output,
    const float* __restrict__ input,
    float* __restrict__       state,
    SG2MambaWeights           weights,
    int                       N,
    int                       d_inner,
    int                       d_state
) {
    extern __shared__ char smem[];
    float* smem_state = reinterpret_cast<float*>(smem);

    // Zero output
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = 0.0f;
    }
    __syncthreads();

    sg2_mamba_scan_forward(output, input, state, weights, N, d_inner, d_state, smem_state);
}

// Mamba backward scan launcher
__global__ void sg2_mamba_backward_kernel(
    float* __restrict__       output,
    const float* __restrict__ input,
    float* __restrict__       state,
    SG2MambaWeights           weights,
    int                       N,
    int                       d_inner,
    int                       d_state
) {
    extern __shared__ char smem[];
    float* smem_state = reinterpret_cast<float*>(smem);

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = 0.0f;
    }
    __syncthreads();

    sg2_mamba_scan_backward(output, input, state, weights, N, d_inner, d_state, smem_state);
}

// GRU update launcher
__global__ void sg2_gru_kernel(
    float* __restrict__       h_new,
    const float* __restrict__ input,
    const float* __restrict__ h_old,
    SG2GRUWeights             weights,
    int                       N,
    int                       gru_hidden,
    int                       gru_input_dim
) {
    sg2_gru_update(h_new, input, h_old, weights, N, gru_hidden, gru_input_dim);
}

// PEER routing launcher
__global__ void sg2_peer_routing_kernel(
    int*   __restrict__       expert_indices,
    float* __restrict__       routing_weights,
    const float* __restrict__ peer_input,
    SG2PEERWeights            weights,
    int                       N,
    int                       num_heads,
    int                       peer_input_dim,
    int                       pk_dim
) {
    extern __shared__ char smem[];
    float* smem_scores = reinterpret_cast<float*>(smem);
    sg2_peer_routing(expert_indices, routing_weights, peer_input, weights,
                     N, num_heads, peer_input_dim, pk_dim, smem_scores);
}

// Expert MLP launcher (FP32 experts)
__global__ void sg2_expert_mlp_kernel(
    float* __restrict__       output,
    const float* __restrict__ input,
    const int*   __restrict__ expert_indices,
    const float* __restrict__ routing_weights,
    SG2ExpertWeights          expert_weights,
    int                       N,
    int                       num_heads,
    int                       num_experts,
    int                       expert_hidden
) {
    sg2_expert_mlp<false>(output, input, expert_indices, routing_weights,
                          expert_weights, N, num_heads, num_experts, expert_hidden);
}

// Expert MLP launcher (INT8 experts)
__global__ void sg2_expert_mlp_int8_kernel(
    float* __restrict__       output,
    const float* __restrict__ input,
    const int*   __restrict__ expert_indices,
    const float* __restrict__ routing_weights,
    SG2ExpertWeights          expert_weights,
    int                       N,
    int                       num_heads,
    int                       num_experts,
    int                       expert_hidden
) {
    sg2_expert_mlp<true>(output, input, expert_indices, routing_weights,
                         expert_weights, N, num_heads, num_experts, expert_hidden);
}

// Expert MLP launcher (INT4 experts)
__global__ void sg2_expert_mlp_int4_kernel(
    float* __restrict__       output,
    const float* __restrict__ input,
    const int*   __restrict__ expert_indices,
    const float* __restrict__ routing_weights,
    SG2ExpertWeights          expert_weights,
    int                       N,
    int                       num_heads,
    int                       num_experts,
    int                       expert_hidden
) {
    sg2_expert_mlp_int4(output, input, expert_indices, routing_weights,
                        expert_weights, N, num_heads, num_experts, expert_hidden);
}

// Adam update launcher
template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void sg2_adam_update_kernel(
    ParamT* __restrict__       params,
    const float* __restrict__  grads_f32,
    const float* __restrict__  smart_grads,
    SuperGrok2State            state,
    int64_t                    N,
    float lr,
    float layer_beta1,
    float beta2,
    float eps,
    float effective_wd,
    float layer_alpha,
    float lamb,
    float ramp,
    float gate_signal,
    float bias_correction1,
    float bias_correction2
) {
    sg2_adam_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads_f32, smart_grads, state, N,
        lr, layer_beta1, beta2, eps, effective_wd,
        layer_alpha, lamb, ramp, gate_signal,
        bias_correction1, bias_correction2);
}

// ============================================================================
// Scratch memory calculator -- tells the host how much global scratch to alloc
// ============================================================================

struct SG2ScratchSizes {
    static int64_t compute(int64_t N, const SG2Hyperparams& hp, int num_blocks) {
        int64_t total = 0;
        total += N * sizeof(float);                    // grads_f32
        total += N * sizeof(float);                    // sorted_keys
        total += N * sizeof(int);                      // sorted_idx
        total += N * sizeof(float);                    // mamba_fwd
        total += N * sizeof(float);                    // mamba_bwd
        total += N * sizeof(float);                    // mamba_combined
        total += N * hp.gru_input_dim * sizeof(float); // gru_input
        total += N * hp.gru_hidden * sizeof(float);    // gru_out
        total += N * hp.num_heads * sizeof(int);       // expert_idx
        total += N * hp.num_heads * sizeof(float);     // routing_wts
        total += N * sizeof(float);                    // expert_out
        total += N * sizeof(float);                    // smart_grads
        total += num_blocks * sg2_constants::RADIX_BUCKETS * sizeof(int);  // radix hist
        return total;
    }

    static int shared_mem_bytes(int64_t N, const SG2Hyperparams& hp) {
        int sort_bytes = (N <= sg2_constants::MAX_BLOCK_SORT_N)
            ? static_cast<int>(N) * (sizeof(float) + sizeof(int))
            : 0;
        int mamba_bytes = hp.d_inner * hp.d_state * sizeof(float);
        int peer_bytes  = 2 * hp.pk_dim * sizeof(float);
        // Return the max across all phases
        int max_bytes = sort_bytes;
        if (mamba_bytes > max_bytes) max_bytes = mamba_bytes;
        if (peer_bytes > max_bytes)  max_bytes = peer_bytes;
        return max_bytes;
    }
};

}} // namespace grokking::sm90

#endif // GROKKING_SUPERGROK2_SM90_CUH_
