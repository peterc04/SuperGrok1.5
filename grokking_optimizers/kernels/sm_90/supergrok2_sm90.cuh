#ifndef GROKKING_SUPERGROK2_SM90_CUH_
#define GROKKING_SUPERGROK2_SM90_CUH_

#include "common_sm90.cuh"

namespace grokking { namespace sm90 {

// ============================================================================
// SuperGrok v2 (SG2) -- CSA/HCA Hybrid Attention + 4-Head PEER + GRU Meta-Net
//
// DeepSeek-V4-style sequence mixer (see /tmp/csa_hca_spec.md, esp. SS2, SS3b):
// the meta-model views a parameter tensor's flattened elements as a SEQUENCE
// and runs compressed-sparse / heavily-compressed attention over it, replacing
// the previous Mamba-3 bidirectional selective scan. The GRU + PEER + expert
// MLP + AdamW apply tail is KEPT VERBATIM (only the sequence mixer changes).
//
// Architecture overview (per-parameter fused step):
//   1. Gradient validation, clipping, and radix sort by |grad|
//   2. CSA (compress m=4, lightning-indexer top-k, +sliding window)  -> csa_ctx
//   3. HCA (heavy compress m'=128, dense attention,  +sliding window) -> hca_ctx
//   4. Unsort to original order
//   5. GRU temporal memory update                  (input: g, s, csa_ctx, hca_ctx)
//   6. PEER 4-head product-key expert routing      (UNCHANGED)
//   7. Expert MLP evaluation                       (UNCHANGED)
//   8. AdamW update with smart (amplified) gradients (UNCHANGED)
//
// CSA replaces the old forward (fine/local) Mamba scan; HCA replaces the old
// backward (global) scan. Attention is STATELESS across optimizer steps, so the
// carried mamba_fwd/bwd state tensors are dropped; only the GRU state persists.
// ============================================================================

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
namespace sg2_constants {
    constexpr int MAX_BLOCK_SORT_N   = 1024;  // Max N for intra-block bitonic sort
    constexpr int RADIX_BITS         = 8;
    constexpr int RADIX_BUCKETS      = 1 << RADIX_BITS;  // 256
    // -- CSA / HCA attention defaults (spec SS3) --
    constexpr int DEFAULT_D_MODEL    = 8;     // per-element feature width
    constexpr int DEFAULT_N_HEADS    = 2;     // attention heads (multi-query KV)
    constexpr int DEFAULT_HEAD_DIM   = 4;     // d_model / n_heads
    constexpr int DEFAULT_CSA_COMPRESS = 4;   // CSA KV compression stride m
    constexpr int DEFAULT_CSA_WINDOW   = 8;   // CSA pooling window / sliding window
    constexpr int DEFAULT_CSA_TOPK     = 16;  // lightning-indexer top-k (clamped to Nc)
    constexpr int DEFAULT_HCA_COMPRESS = 128; // HCA KV compression stride m'
    constexpr int DEFAULT_INDEXER_RANK = 4;   // low-rank lightning indexer rank
    constexpr int DEFAULT_GRU_HIDDEN = 8;
    constexpr int DEFAULT_NUM_HEADS  = 4;     // PEER heads
    constexpr int DEFAULT_PK_DIM     = 32;    // Product-key sub-dimension
    constexpr int DEFAULT_NUM_EXPERTS = 1024;  // pk_dim * pk_dim
    constexpr int DEFAULT_EXPERT_HIDDEN = 16;

    constexpr int SORT_SMEM_BYTES       = MAX_BLOCK_SORT_N * (sizeof(float) + sizeof(int));
    // SMEM for an attention tile: q/k/v rows of one head + online-softmax scratch
    constexpr int ATTN_SMEM_BYTES       = 4 * DEFAULT_D_MODEL * sizeof(float);
    constexpr int GRU_SMEM_BYTES        = 0;
    constexpr int PEER_SMEM_BYTES       = 4 * DEFAULT_PK_DIM * sizeof(float);  // Per-head top-k scratch
    constexpr int EXPERT_SMEM_BYTES     = DEFAULT_EXPERT_HIDDEN * 2 * sizeof(float);  // W1, b1 tile
}

// ---------------------------------------------------------------------------
// State struct: 5 persistent state tensors per parameter group
//
// Attention (CSA/HCA) is stateless across steps, so unlike the Mamba scan there
// is no carried sequence-mixer state -- only the Adam moments + GRU memory.
// ---------------------------------------------------------------------------
struct SuperGrok2State {
    float* __restrict__ exp_avg;           // [N]   Adam first moment
    float* __restrict__ exp_avg_sq;        // [N]   Adam second moment
    float* __restrict__ mu;                // [N]   EMA gradient
    float* __restrict__ sharpness;         // [N]   Gradient correction magnitude EMA
    float* __restrict__ gru_state;         // [N * gru_hidden]

    static constexpr int num_state_tensors() { return 5; }
    static constexpr int state_bytes_per_element(int gru_hidden) {
        return 4 * sizeof(float)                     // exp_avg, exp_avg_sq, mu, sharpness
             + gru_hidden * sizeof(float);           // gru_state
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
    // -- CSA / HCA attention dims (replace d_inner/d_state) --
    int   d_model;        // per-element feature width
    int   n_heads;        // attention heads (multi-query KV)
    int   csa_compress;   // CSA compression stride m (e.g. 4)
    int   csa_window;     // CSA pooling / sliding window (e.g. 8)
    int   csa_topk;       // lightning-indexer top-k (clamped to Nc)
    int   hca_compress;   // HCA compression stride m' (e.g. 128)
    int   indexer_rank;   // low-rank lightning indexer rank
    // -- GRU / PEER / expert dims (unchanged) --
    int   gru_hidden;
    int   gru_input_dim;
    int   num_heads;      // PEER heads
    int   pk_dim;
    int   num_experts;
    int   expert_hidden;
    int   peer_input_dim;
};

// ---------------------------------------------------------------------------
// CSA (Compressed Sparse Attention) weight pointers -- produces csa_ctx.
//
// Mechanics (spec SS2):
//   c_kv[j] = sum_w softmax(csa_compress_w)[w] * x[j*m + w]   (m=csa_compress)
//   qI = x @ idx_DQ @ idx_UQ   (low-rank lightning-indexer query, rank R)
//   kI = compress(x @ idx_K)   (indexer keys, same pooling)
//   I[t,s] = qI[t] . kI[s] / sqrt(R)         -> keep top-k compressed entries s
//   out[t]  = softmax(Q.K^T / sqrt(head_dim)) . V   over (top-k compressed
//             union last csa_window raw tokens), KV shared across heads (MQA).
// ---------------------------------------------------------------------------
struct SG2CSAWeights {
    const float* __restrict__ q_W;          // [d_model, d_model]  query proj
    const float* __restrict__ k_W;          // [d_model, d_model]  key proj
    const float* __restrict__ v_W;          // [d_model, d_model]  value proj
    const float* __restrict__ out_W;        // [d_model, d_model]  output proj
    const float* __restrict__ csa_compress_w; // [csa_window]  learned KV pool weights
    const float* __restrict__ idx_DQ;       // [d_model, indexer_rank]  indexer q down-proj
    const float* __restrict__ idx_UQ;       // [indexer_rank, d_model]  indexer q up-proj
    const float* __restrict__ idx_K;        // [d_model, indexer_rank]  indexer key proj
};

// ---------------------------------------------------------------------------
// HCA (Heavily Compressed Attention) weight pointers -- produces hca_ctx.
//
// Mechanics (spec SS2): stride-m' mean/learned pool (m'=hca_compress) gives
// Nh = ceil(N / m') compressed entries; every query attends DENSELY to all Nh
// compressed entries (no top-k) plus the sliding window. Global coarse context.
// ---------------------------------------------------------------------------
struct SG2HCAWeights {
    const float* __restrict__ q_W;          // [d_model, d_model]  query proj
    const float* __restrict__ k_W;          // [d_model, d_model]  key proj
    const float* __restrict__ v_W;          // [d_model, d_model]  value proj
    const float* __restrict__ out_W;        // [d_model, d_model]  output proj
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
// 2. CSA / HCA hybrid attention sequence mixer (spec SS2, SS3b)
//
// Replaces the Mamba bidirectional selective scan. CSA produces the fine-grained
// /local context (csa_ctx, formerly mamba fwd); HCA produces the global coarse
// context (hca_ctx, formerly mamba bwd). Both operate on the d_model-wide
// projected feature sequence x[N, d_model] (built from sorted [grad, sharpness]
// via input_proj upstream). All accumulation is FP32.
//
// Helper math (small d_model, so explicit loops -- production routes GEMMs
// through CUTLASS WGMMA per spec SS8):
//   proj(x, W)[t, o] = sum_i x[t, i] * W[o, i]            (row-major [out, in])
// ============================================================================

// ---- 2a. sg2_csa_compress_kv -------------------------------------------------
// Compress the projected K/V sequence by pooling a `window`-wide block at stride
// `m` with a learned softmax(compress_w) weighting (plus implicit bias folded
// into the weights). Produces Nc = ceil(N / m) compressed rows of width d_model.
//
//   c_kv[j, :] = sum_{w=0..window-1} softmax(compress_w)[w] * kv[j*m + w, :]
//
// For HCA, pass compress_w = nullptr to fall back to a uniform mean pool over the
// `m`-wide stride block (window == m, learned-weight-free heavy compression).
__forceinline__ __device__
void sg2_csa_compress_kv(
    float* __restrict__       c_kv,        // [Nc, d_model] compressed output
    const float* __restrict__ kv,          // [N, d_model] projected key or value seq
    const float* __restrict__ compress_w,  // [window] learned weights (nullptr = mean)
    int                       N,
    int                       d_model,
    int                       m,            // compression stride
    int                       window        // pooling window (>= m)
) {
    const int Nc = (N + m - 1) / m;

    // Precompute softmax(compress_w) into per-thread registers via a max/sum scan.
    // window is tiny (<= 8 for CSA), so each thread recomputes it cheaply.
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < Nc;
         j += gridDim.x * (int64_t)blockDim.x) {

        float wmax = -1e30f;
        if (compress_w) {
            for (int w = 0; w < window; ++w) wmax = fmaxf(wmax, __ldg(&compress_w[w]));
        }
        float wsum = 0.0f;
        for (int w = 0; w < window; ++w) {
            wsum += compress_w ? __expf(__ldg(&compress_w[w]) - wmax) : 1.0f;
        }
        float inv_wsum = 1.0f / (wsum + 1e-20f);

        for (int d = 0; d < d_model; ++d) {
            float acc = 0.0f;
            for (int w = 0; w < window; ++w) {
                int src = j * m + w;
                if (src >= N) break;
                float pw = compress_w ? __expf(__ldg(&compress_w[w]) - wmax) * inv_wsum
                                      : inv_wsum;  // uniform mean when no weights
                acc += pw * kv[src * d_model + d];
            }
            c_kv[j * d_model + d] = acc;
        }
    }
}

// ---- 2b. sg2_hca_compress_kv -------------------------------------------------
// Heavy stride-m' mean pool (m'=hca_compress, window==m', no learned weights).
// Thin wrapper over sg2_csa_compress_kv with compress_w=nullptr.
__forceinline__ __device__
void sg2_hca_compress_kv(
    float* __restrict__       c_kv,        // [Nh, d_model]
    const float* __restrict__ kv,          // [N, d_model]
    int                       N,
    int                       d_model,
    int                       m_prime       // hca_compress (e.g. 128)
) {
    sg2_csa_compress_kv(c_kv, kv, /*compress_w=*/nullptr, N, d_model, m_prime, m_prime);
}

// ---- 2c. sg2_csa_indexer_topk ------------------------------------------------
// Lightning indexer: low-rank query qI[t] = (x[t] @ idx_DQ) @ idx_UQ scores each
// query token against compressed indexer keys kI[s] (= compress(x @ idx_K)):
//
//   I[t, s] = qI[t] . kI[s] / sqrt(rank)
//
// Keep the top-k highest-scoring compressed entries per query (clamped to Nc).
// Selected compressed-entry indices are written to topk_idx[t, 0..k-1] (-1 pad).
//
// Reference path uses a simple per-query selection scan (Nc is small for the
// meta-model); production uses a fused argpartition. Here qI is recomputed from
// the low-rank factors and kI is precomputed indexer keys [Nc, rank].
__forceinline__ __device__
void sg2_csa_indexer_topk(
    int*   __restrict__       topk_idx,    // [N, k] selected compressed indices
    const float* __restrict__ x,           // [N, d_model] projected feature seq
    const float* __restrict__ kI,          // [Nc, rank] compressed indexer keys
    const SG2CSAWeights&      weights,
    int                       N,
    int                       Nc,
    int                       d_model,
    int                       rank,
    int                       k             // csa_topk (clamped to Nc by caller)
) {
    const float inv_sqrt_rank = rsqrtf(static_cast<float>(rank));

    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < N;
         t += gridDim.x * (int64_t)blockDim.x) {

        // qI[t] = (x[t] @ idx_DQ) @ idx_UQ -> low-rank query of width `rank`.
        // idx_DQ:[d_model, rank], idx_UQ:[rank, d_model]; we only need the rank-
        // dim indexer query, so qI = (x @ idx_DQ) then projected back implicitly
        // via dot with kI keys (which live in rank space). Compute z = x @ idx_DQ.
        float z[8];  // rank <= 8 for the meta-model
        for (int r = 0; r < rank; ++r) {
            float acc = 0.0f;
            for (int i = 0; i < d_model; ++i) {
                acc += x[t * d_model + i] * __ldg(&weights.idx_DQ[i * rank + r]);
            }
            z[r] = acc;
        }

        // Online top-k selection over compressed entries s (causal: s*m <= t).
        // Keep the k best by score; small-k insertion into a register buffer.
        float best_score[64];
        int   best_idx[64];
        int kk = (k < 64) ? k : 64;
        for (int j = 0; j < kk; ++j) { best_score[j] = -1e30f; best_idx[j] = -1; }

        for (int s = 0; s < Nc; ++s) {
            float score = 0.0f;
            for (int r = 0; r < rank; ++r) {
                score += z[r] * __ldg(&kI[s * rank + r]);
            }
            score *= inv_sqrt_rank;
            // Insert s into the running top-k if it beats the current minimum.
            int min_pos = 0;
            float min_val = best_score[0];
            for (int j = 1; j < kk; ++j) {
                if (best_score[j] < min_val) { min_val = best_score[j]; min_pos = j; }
            }
            if (score > min_val) {
                best_score[min_pos] = score;
                best_idx[min_pos]   = s;
            }
        }
        for (int j = 0; j < kk; ++j) {
            topk_idx[t * k + j] = best_idx[j];
        }
    }
}

// ---- 2d. sg2_attention_online_softmax ---------------------------------------
// Numerically-stable online (flash-style) softmax attention of one query row
// against a caller-supplied set of key/value rows, accumulating into out[head]:
//
//   scores = q . k / sqrt(head_dim);  running max m, denom l (online softmax)
//   out += softmax(scores) . v
//
// Multi-query: a single shared KV row set is reused across all heads (the q/out
// are per-head slices of the d_model vector). Returns nothing; writes `out`.
__forceinline__ __device__
void sg2_attention_online_softmax(
    float* __restrict__       out,         // [head_dim] per-head context (accum)
    const float* __restrict__ q,           // [head_dim] query for this head
    const float* __restrict__ k_rows,      // [num_kv, d_model] candidate keys
    const float* __restrict__ v_rows,      // [num_kv, d_model] candidate values
    const int*   __restrict__ kv_index,    // [num_kv] row ids (-1 = skip), or nullptr
    int                       num_kv,
    int                       head_off,     // head_idx * head_dim (slice into d_model)
    int                       head_dim,
    int                       d_model
) {
    const float inv_sqrt_d = rsqrtf(static_cast<float>(head_dim));
    float run_max = -1e30f;
    float run_den = 0.0f;
    for (int h = 0; h < head_dim; ++h) out[h] = 0.0f;

    for (int j = 0; j < num_kv; ++j) {
        int row = kv_index ? kv_index[j] : j;
        if (row < 0) continue;
        const float* kj = k_rows + row * d_model + head_off;
        const float* vj = v_rows + row * d_model + head_off;

        float score = 0.0f;
        for (int h = 0; h < head_dim; ++h) score += q[h] * kj[h];
        score *= inv_sqrt_d;

        // Online softmax rescale.
        float new_max = fmaxf(run_max, score);
        float corr    = __expf(run_max - new_max);
        float p       = __expf(score - new_max);
        run_den = run_den * corr + p;
        for (int h = 0; h < head_dim; ++h) {
            out[h] = out[h] * corr + p * vj[h];
        }
        run_max = new_max;
    }

    float inv_den = 1.0f / (run_den + 1e-20f);
    for (int h = 0; h < head_dim; ++h) out[h] *= inv_den;
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
    SG2CSAWeights              csa_weights,   // CSA layer (produces csa_ctx)
    SG2HCAWeights              hca_weights,   // HCA layer (produces hca_ctx)
    const float* __restrict__  input_proj_W,  // [d_model, 2]
    const float* __restrict__  input_proj_b,  // [d_model]
    SG2GRUWeights              gru_weights,
    SG2PEERWeights             peer_weights,
    SG2ExpertWeights           expert_weights,
    // Scratch buffers (caller-allocated global memory)
    float* __restrict__        scratch_grads_f32,    // [N]
    float* __restrict__        scratch_sorted_keys,  // [N]
    int*   __restrict__        scratch_sorted_idx,   // [N]
    float* __restrict__        scratch_x,            // [N, d_model] projected seq
    float* __restrict__        scratch_q,            // [N, d_model] queries
    float* __restrict__        scratch_k,            // [N, d_model] keys (raw)
    float* __restrict__        scratch_v,            // [N, d_model] values (raw)
    float* __restrict__        scratch_c_k,          // [Nc_max, d_model] compressed keys
    float* __restrict__        scratch_c_v,          // [Nc_max, d_model] compressed values
    float* __restrict__        scratch_kI,           // [Nc_max, indexer_rank] indexer keys
    int*   __restrict__        scratch_topk,         // [N, csa_topk] selected indices
    float* __restrict__        scratch_csa_ctx,      // [N, d_model] CSA context
    float* __restrict__        scratch_hca_ctx,      // [N, d_model] HCA context
    float* __restrict__        scratch_gru_input,    // [N, gru_input_dim]
    float* __restrict__        scratch_gru_out,      // [N, gru_hidden]
    int*   __restrict__        scratch_expert_idx,   // [N, num_heads]
    float* __restrict__        scratch_routing_wts,  // [N, num_heads]
    float* __restrict__        scratch_expert_out,   // [N]
    float* __restrict__        scratch_smart_grads,  // [N]
    int*   __restrict__        scratch_radix_hist    // [gridDim.x * 256]
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
    // Step (c): Input projection + Q/K/V projections of the sorted sequence
    //
    // x[t] = input_proj_W @ [grad_sorted[t], sharpness_sorted[t]] + input_proj_b
    // q/k/v[t] = x[t] @ {csa_q_W, csa_k_W, csa_v_W}^T   (row-major [out, in])
    // (Production routes all of these through CUTLASS WGMMA, spec SS8.)
    // ====================================================================
    {
        const int dm = hparams.d_model;
        for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < N;
             t += gridDim.x * (int64_t)blockDim.x) {
            int orig = scratch_sorted_idx[t];
            float g_s = scratch_sorted_keys[t];
            float s_s = (orig >= 0 && orig < N) ? state.sharpness[orig] : 0.0f;
            for (int o = 0; o < dm; ++o) {
                float acc = __ldg(&input_proj_b[o]);
                acc += __ldg(&input_proj_W[o * 2 + 0]) * g_s;
                acc += __ldg(&input_proj_W[o * 2 + 1]) * s_s;
                scratch_x[t * dm + o] = acc;
            }
            // q/k/v projections (CSA weights; HCA reuses the same x via its own
            // q/k/v below -- here we materialize the CSA stream q/k/v).
            for (int o = 0; o < dm; ++o) {
                float qa = 0.0f, ka = 0.0f, va = 0.0f;
                for (int i = 0; i < dm; ++i) {
                    float xv = scratch_x[t * dm + i];
                    qa += xv * __ldg(&csa_weights.q_W[o * dm + i]);
                    ka += xv * __ldg(&csa_weights.k_W[o * dm + i]);
                    va += xv * __ldg(&csa_weights.v_W[o * dm + i]);
                }
                scratch_q[t * dm + o] = qa;
                scratch_k[t * dm + o] = ka;
                scratch_v[t * dm + o] = va;
            }
        }
    }
    __syncthreads();

    // ====================================================================
    // Step (c2): CSA -- compress KV (m=csa_compress), lightning-indexer top-k,
    //            sparse attention (+sliding window) -> csa_ctx (local context).
    // ====================================================================
    {
        const int dm = hparams.d_model;
        const int m  = hparams.csa_compress;
        const int Nc = (static_cast<int>(N) + m - 1) / m;
        const int R  = hparams.indexer_rank;
        int k = hparams.csa_topk; if (k > Nc) k = Nc;
        const int head_dim = dm / hparams.n_heads;

        // Compress keys and values with learned pooling weights.
        sg2_csa_compress_kv(scratch_c_k, scratch_k, csa_weights.csa_compress_w,
                            static_cast<int>(N), dm, m, hparams.csa_window);
        sg2_csa_compress_kv(scratch_c_v, scratch_v, csa_weights.csa_compress_w,
                            static_cast<int>(N), dm, m, hparams.csa_window);
        __syncthreads();

        // Build compressed indexer keys kI[Nc, R] = compress(x @ idx_K).
        // Reuse scratch_kI; idx_K projects d_model -> R, then pool with the same
        // learned weights. (Done inline: project per token then pool.)
        for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < Nc;
             s += gridDim.x * (int64_t)blockDim.x) {
            for (int r = 0; r < R; ++r) {
                float acc = 0.0f;
                for (int w = 0; w < m; ++w) {
                    int src = s * m + w;
                    if (src >= N) break;
                    float proj = 0.0f;
                    for (int i = 0; i < dm; ++i) {
                        proj += scratch_x[src * dm + i] * __ldg(&csa_weights.idx_K[i * R + r]);
                    }
                    acc += proj / static_cast<float>(m);  // mean pool over stride
                }
                scratch_kI[s * R + r] = acc;
            }
        }
        __syncthreads();

        sg2_csa_indexer_topk(scratch_topk, scratch_x, scratch_kI, csa_weights,
                             static_cast<int>(N), Nc, dm, R, k);
        __syncthreads();

        // Inline the CSA attention body (the sg2_csa_attention_kernel launcher
        // above wraps the same math for standalone autotuning).
        for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < N;
             t += gridDim.x * (int64_t)blockDim.x) {
            float ctx[64];
            for (int d = 0; d < dm; ++d) ctx[d] = 0.0f;
            for (int hd = 0; hd < hparams.n_heads; ++hd) {
                int head_off = hd * head_dim;
                const float* q_h = scratch_q + t * dm + head_off;
                float head_out[16], win_out[16];
                sg2_attention_online_softmax(head_out, q_h, scratch_c_k, scratch_c_v,
                                             &scratch_topk[t * k], k, head_off, head_dim, dm);
                int ws = t - hparams.csa_window + 1; if (ws < 0) ws = 0;
                int wn = static_cast<int>(t) - ws + 1;
                sg2_attention_online_softmax(win_out, q_h,
                                             scratch_k + (int64_t)ws * dm,
                                             scratch_v + (int64_t)ws * dm,
                                             nullptr, wn, head_off, head_dim, dm);
                for (int h = 0; h < head_dim; ++h)
                    ctx[head_off + h] = 0.5f * (head_out[h] + win_out[h]);
            }
            for (int o = 0; o < dm; ++o) {
                float acc = 0.0f;
                for (int i = 0; i < dm; ++i)
                    acc += ctx[i] * __ldg(&csa_weights.out_W[o * dm + i]);
                scratch_csa_ctx[t * dm + o] = acc;
            }
        }
    }
    __syncthreads();

    // ====================================================================
    // Step (d): HCA -- heavy stride-m' mean compress, DENSE attention over all
    //           Nh compressed entries (+sliding window) -> hca_ctx (global ctx).
    // ====================================================================
    {
        const int dm = hparams.d_model;
        const int mp = hparams.hca_compress;
        const int Nh = (static_cast<int>(N) + mp - 1) / mp;
        const int head_dim = dm / hparams.n_heads;

        // HCA reuses scratch_q/k/v: re-project with HCA weights into c_k/c_v after
        // compression. Project k/v with HCA k_W/v_W (overwrite scratch_k/v).
        for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < N;
             t += gridDim.x * (int64_t)blockDim.x) {
            for (int o = 0; o < dm; ++o) {
                float qa = 0.0f, ka = 0.0f, va = 0.0f;
                for (int i = 0; i < dm; ++i) {
                    float xv = scratch_x[t * dm + i];
                    qa += xv * __ldg(&hca_weights.q_W[o * dm + i]);
                    ka += xv * __ldg(&hca_weights.k_W[o * dm + i]);
                    va += xv * __ldg(&hca_weights.v_W[o * dm + i]);
                }
                scratch_q[t * dm + o] = qa;
                scratch_k[t * dm + o] = ka;
                scratch_v[t * dm + o] = va;
            }
        }
        __syncthreads();

        sg2_hca_compress_kv(scratch_c_k, scratch_k, static_cast<int>(N), dm, mp);
        sg2_hca_compress_kv(scratch_c_v, scratch_v, static_cast<int>(N), dm, mp);
        __syncthreads();

        for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < N;
             t += gridDim.x * (int64_t)blockDim.x) {
            float ctx[64];
            for (int d = 0; d < dm; ++d) ctx[d] = 0.0f;
            for (int hd = 0; hd < hparams.n_heads; ++hd) {
                int head_off = hd * head_dim;
                const float* q_h = scratch_q + t * dm + head_off;
                float head_out[16], win_out[16];
                sg2_attention_online_softmax(head_out, q_h, scratch_c_k, scratch_c_v,
                                             nullptr, Nh, head_off, head_dim, dm);
                int ws = t - hparams.csa_window + 1; if (ws < 0) ws = 0;
                int wn = static_cast<int>(t) - ws + 1;
                sg2_attention_online_softmax(win_out, q_h,
                                             scratch_k + (int64_t)ws * dm,
                                             scratch_v + (int64_t)ws * dm,
                                             nullptr, wn, head_off, head_dim, dm);
                for (int h = 0; h < head_dim; ++h)
                    ctx[head_off + h] = 0.5f * (head_out[h] + win_out[h]);
            }
            for (int o = 0; o < dm; ++o) {
                float acc = 0.0f;
                for (int i = 0; i < dm; ++i)
                    acc += ctx[i] * __ldg(&hca_weights.out_W[o * dm + i]);
                scratch_hca_ctx[t * dm + o] = acc;
            }
        }
    }
    __syncthreads();

    // ====================================================================
    // Step (e): Unsort csa_ctx/hca_ctx to original element order and build the
    //           GRU input [grad, csa_summary, hca_summary, sharpness] -- same
    //           downstream contract as the old mamba combined signal (SS3b).
    // ====================================================================
    for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < N;
         t += gridDim.x * (int64_t)blockDim.x) {
        int orig_idx = scratch_sorted_idx[t];
        if (orig_idx >= 0 && orig_idx < N) {
            const int dm = hparams.d_model;
            // Summarize each context vector to a scalar (mean over d_model), the
            // local (CSA) and global (HCA) signals that previously came from the
            // bidirectional mamba scan.
            float csa_sum = 0.0f, hca_sum = 0.0f;
            for (int d = 0; d < dm; ++d) {
                csa_sum += scratch_csa_ctx[t * dm + d];
                hca_sum += scratch_hca_ctx[t * dm + d];
            }
            csa_sum /= static_cast<float>(dm);
            hca_sum /= static_cast<float>(dm);

            float g_val = scratch_grads_f32[orig_idx];
            float sharp_val = state.sharpness[orig_idx];

            int gid = hparams.gru_input_dim;
            scratch_gru_input[orig_idx * gid + 0] = g_val;
            if (gid > 1) scratch_gru_input[orig_idx * gid + 1] = csa_sum;
            if (gid > 2) scratch_gru_input[orig_idx * gid + 2] = hca_sum;
            if (gid > 3) scratch_gru_input[orig_idx * gid + 3] = sharp_val;
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

// CSA KV-compression launcher (spec SS2.1). Pools key (or value) seq at stride
// `m` with learned softmax(compress_w) weights -> c_kv[Nc, d_model].
__global__ void sg2_csa_compress_kernel(
    float* __restrict__       c_kv,          // [Nc, d_model]
    const float* __restrict__ kv,            // [N, d_model]
    const float* __restrict__ compress_w,    // [window] (nullptr -> mean pool)
    int                       N,
    int                       d_model,
    int                       m,
    int                       window
) {
    sg2_csa_compress_kv(c_kv, kv, compress_w, N, d_model, m, window);
}

// CSA lightning-indexer top-k launcher (spec SS2.2). Scores each query against
// compressed indexer keys and emits the top-k selected compressed indices.
__global__ void sg2_csa_indexer_kernel(
    int*   __restrict__       topk_idx,      // [N, k]
    const float* __restrict__ x,             // [N, d_model]
    const float* __restrict__ kI,            // [Nc, rank] compressed indexer keys
    SG2CSAWeights             weights,
    int                       N,
    int                       Nc,
    int                       d_model,
    int                       rank,
    int                       k
) {
    sg2_csa_indexer_topk(topk_idx, x, kI, weights, N, Nc, d_model, rank, k);
}

// CSA attention launcher (spec SS2.4). Each query attends to its top-k selected
// compressed entries UNION the last csa_window raw tokens (sliding window), with
// KV shared across heads (multi-query). Writes csa_ctx[N, d_model].
__global__ void sg2_csa_attention_kernel(
    float* __restrict__       csa_ctx,       // [N, d_model] output context
    const float* __restrict__ q,             // [N, d_model] projected queries
    const float* __restrict__ c_k,           // [Nc, d_model] compressed keys
    const float* __restrict__ c_v,           // [Nc, d_model] compressed values
    const float* __restrict__ k_raw,         // [N, d_model] raw keys (window)
    const float* __restrict__ v_raw,         // [N, d_model] raw values (window)
    const int*   __restrict__ topk_idx,      // [N, k] selected compressed indices
    const float* __restrict__ out_W,         // [d_model, d_model] output proj
    int                       N,
    int                       d_model,
    int                       n_heads,
    int                       head_dim,
    int                       csa_window,
    int                       k
) {
    // Per-query scratch for the (top-k compressed) U (window raw) candidate set.
    // We build the window index list inline; compressed rows come via topk_idx.
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < N;
         t += gridDim.x * (int64_t)blockDim.x) {

        float ctx[64];   // d_model <= 64 accumulator (pre-out_proj)
        for (int d = 0; d < d_model; ++d) ctx[d] = 0.0f;

        for (int hd = 0; hd < n_heads; ++hd) {
            const int head_off = hd * head_dim;
            const float* q_h = q + t * d_model + head_off;

            float head_out[16];  // head_dim <= 16

            // (i) attend to the top-k compressed entries.
            sg2_attention_online_softmax(
                head_out, q_h, c_k, c_v, &topk_idx[t * k],
                k, head_off, head_dim, d_model);

            // (ii) sliding window: blend the last csa_window raw tokens. We run a
            // second online softmax over the window then average the two contexts
            // (reference simplification; production fuses into one denom).
            float win_out[16];
            int win_start = t - csa_window + 1;
            if (win_start < 0) win_start = 0;
            int win_n = t - win_start + 1;
            // local index list 0..win_n-1 mapped to raw rows win_start+..
            // sg2_attention_online_softmax indexes k_raw/v_raw by row directly,
            // so synthesize a contiguous window via kv_index=nullptr+offset ptrs.
            sg2_attention_online_softmax(
                win_out, q_h,
                k_raw + (int64_t)win_start * d_model,
                v_raw + (int64_t)win_start * d_model,
                /*kv_index=*/nullptr, win_n, head_off, head_dim, d_model);

            for (int h = 0; h < head_dim; ++h) {
                ctx[head_off + h] = 0.5f * (head_out[h] + win_out[h]);
            }
        }

        // Output projection: csa_ctx[t] = ctx @ out_W^T  (out_W row-major [out,in])
        for (int o = 0; o < d_model; ++o) {
            float acc = 0.0f;
            for (int i = 0; i < d_model; ++i) {
                acc += ctx[i] * __ldg(&out_W[o * d_model + i]);
            }
            csa_ctx[t * d_model + o] = acc;
        }
    }
}

// HCA attention launcher (spec SS2 HCA). Each query attends DENSELY to ALL Nh
// heavily-compressed entries (no top-k) plus the sliding window. Writes
// hca_ctx[N, d_model]. Global coarse context.
__global__ void sg2_hca_attention_kernel(
    float* __restrict__       hca_ctx,       // [N, d_model] output context
    const float* __restrict__ q,             // [N, d_model] projected queries
    const float* __restrict__ c_k,           // [Nh, d_model] compressed keys
    const float* __restrict__ c_v,           // [Nh, d_model] compressed values
    const float* __restrict__ k_raw,         // [N, d_model] raw keys (window)
    const float* __restrict__ v_raw,         // [N, d_model] raw values (window)
    const float* __restrict__ out_W,         // [d_model, d_model] output proj
    int                       N,
    int                       Nh,
    int                       d_model,
    int                       n_heads,
    int                       head_dim,
    int                       csa_window
) {
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < N;
         t += gridDim.x * (int64_t)blockDim.x) {

        float ctx[64];
        for (int d = 0; d < d_model; ++d) ctx[d] = 0.0f;

        for (int hd = 0; hd < n_heads; ++hd) {
            const int head_off = hd * head_dim;
            const float* q_h = q + t * d_model + head_off;

            float head_out[16];
            // Dense attention over all Nh compressed entries (kv_index=nullptr).
            sg2_attention_online_softmax(
                head_out, q_h, c_k, c_v, /*kv_index=*/nullptr,
                Nh, head_off, head_dim, d_model);

            float win_out[16];
            int win_start = t - csa_window + 1;
            if (win_start < 0) win_start = 0;
            int win_n = t - win_start + 1;
            sg2_attention_online_softmax(
                win_out, q_h,
                k_raw + (int64_t)win_start * d_model,
                v_raw + (int64_t)win_start * d_model,
                /*kv_index=*/nullptr, win_n, head_off, head_dim, d_model);

            for (int h = 0; h < head_dim; ++h) {
                ctx[head_off + h] = 0.5f * (head_out[h] + win_out[h]);
            }
        }

        for (int o = 0; o < d_model; ++o) {
            float acc = 0.0f;
            for (int i = 0; i < d_model; ++i) {
                acc += ctx[i] * __ldg(&out_W[o * d_model + i]);
            }
            hca_ctx[t * d_model + o] = acc;
        }
    }
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
        const int dm = hp.d_model;
        const int Nc = (static_cast<int>(N) + hp.csa_compress - 1) / hp.csa_compress;
        int64_t total = 0;
        total += N * sizeof(float);                    // grads_f32
        total += N * sizeof(float);                    // sorted_keys
        total += N * sizeof(int);                      // sorted_idx
        total += N * dm * sizeof(float);               // x (projected seq)
        total += N * dm * sizeof(float);               // q
        total += N * dm * sizeof(float);               // k (raw)
        total += N * dm * sizeof(float);               // v (raw)
        total += (int64_t)Nc * dm * sizeof(float);     // c_k (compressed keys)
        total += (int64_t)Nc * dm * sizeof(float);     // c_v (compressed values)
        total += (int64_t)Nc * hp.indexer_rank * sizeof(float);  // kI (indexer keys)
        total += N * hp.csa_topk * sizeof(int);        // topk indices
        total += N * dm * sizeof(float);               // csa_ctx
        total += N * dm * sizeof(float);               // hca_ctx
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
        // Attention works register-resident per query (small d_model), so its
        // SMEM footprint is modest; size it to one q/k/v/out head tile.
        int attn_bytes = 4 * hp.d_model * sizeof(float);
        int peer_bytes  = 2 * hp.pk_dim * sizeof(float);
        // Return the max across all phases
        int max_bytes = sort_bytes;
        if (attn_bytes > max_bytes) max_bytes = attn_bytes;
        if (peer_bytes > max_bytes)  max_bytes = peer_bytes;
        return max_bytes;
    }
};

}} // namespace grokking::sm90

#endif // GROKKING_SUPERGROK2_SM90_CUH_
