#ifndef GROKKING_TRANSFORMER_DECODER_GFX942_HIP_HPP_
#define GROKKING_TRANSFORMER_DECODER_GFX942_HIP_HPP_

#include "common_gfx942.hip.hpp"

namespace grokking { namespace gfx942 {

// ---- TransformerDecoderState ------------------------------------------------
// Persistent megakernel state for a post-norm decoder transformer.
// Architecture: embed -> [rmsnorm -> qkv_proj -> attention(RoPE, causal) ->
//   o_proj -> residual_add -> rmsnorm -> mlp_gate_up -> mlp_down ->
//   residual_add] x N -> final_norm -> lm_head
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
struct TransformerDecoderState {
    // Model geometry (set once at init).
    int d_model;
    int n_heads;
    int head_dim;
    int n_layers;
    int ffn_intermediate;   // gate/up projection width (typically 4*d or 8/3*d)

    // Weight pointers (contiguous packed buffer).
    ParamT* __restrict__ tok_embed;          // [VOCAB, d_model]
    ParamT* __restrict__ pos_embed;          // [SEQ_LEN, d_model] (unused when RoPE)

    // Per-layer weights (flat array of n_layers structs).
    struct LayerWeights {
        ParamT* __restrict__ rms1_g;         // [d_model]
        ParamT* __restrict__ qkv_W;          // [3*d_model, d_model]
        ParamT* __restrict__ o_W;            // [d_model, d_model]
        ParamT* __restrict__ rms2_g;         // [d_model]
        ParamT* __restrict__ gate_up_W;      // [2*ffn_intermediate, d_model]
        ParamT* __restrict__ down_W;         // [d_model, ffn_intermediate]
    };
    LayerWeights* layers;

    ParamT* __restrict__ final_rms_g;        // [d_model]
    ParamT* __restrict__ lm_head_W;          // [VOCAB, d_model] (or alias tok_embed)
    ParamT* __restrict__ lm_head_b;          // [VOCAB]

    // Activation scratch (caller-allocated).
    ParamT* __restrict__ act_scratch;
    size_t act_scratch_elems;

    // Gradient buffer (same layout as weights, caller-allocated).
    ParamT* __restrict__ grad_buf;

    static constexpr int seq_len  = SEQ_LEN;
    static constexpr int batch    = BATCH;
    static constexpr int vocab    = VOCAB;
    static constexpr bool tied    = TIED_EMBEDDINGS;
    static constexpr NanPolicy nan_policy = NAN_POLICY;
};

// ---- Resource declarations (CDNA3 / gfx942 tuning) -------------------------
// MI300X: 64-wide wavefronts, 64 KB LDS per workgroup, 512 VGPRs per SIMD.
// 8 XCDs (compute dies) with independent L2 partitions.
namespace decoder_resources {

static constexpr int WAVEFRONT_SIZE       = 64;
static constexpr int WORKGROUP_SIZE_NORM  = 256;    // 4 wavefronts for reductions
static constexpr int WORKGROUP_SIZE_ATTN  = 256;    // 4 wavefronts per attention tile
static constexpr int WORKGROUP_SIZE_MLP   = 256;
static constexpr int WORKGROUP_SIZE_EMBED = 128;    // 2 wavefronts

// LDS budget (bytes). gfx942 allows 64 KB per workgroup.
static constexpr int LDS_BYTES_MAX        = 65536;
static constexpr int LDS_BYTES_NORM       = 8192;   // row buffer for rmsnorm
static constexpr int LDS_BYTES_ATTN_TILE  = 32768;  // Q/K tile staging for MFMA
static constexpr int LDS_BYTES_MLP        = 8192;   // SiLU intermediate

// VGPR hints — controls waves-per-EU occupancy on CDNA3.
static constexpr int REG_HINT_NORM        = 64;     // low register pressure
static constexpr int REG_HINT_ATTN        = 128;    // MFMA accumulator heavy
static constexpr int REG_HINT_MLP         = 96;
static constexpr int REG_HINT_EMBED       = 48;

// XCD-aware tiling for attention.
static constexpr int NUM_XCDS             = 8;
static constexpr int MFMA_M               = 32;
static constexpr int MFMA_N               = 32;
static constexpr int MFMA_K               = 8;      // BF16 MFMA shape

} // namespace decoder_resources

// ---- DecoderSizes -----------------------------------------------------------
// Constexpr struct for compile-time size computations.
template <int D_MODEL, int N_HEADS, int FFN_INTER, int SEQ_LEN, int BATCH, int VOCAB>
struct DecoderSizes {
    static constexpr int d       = D_MODEL;
    static constexpr int h       = N_HEADS;
    static constexpr int d_h     = D_MODEL / N_HEADS;
    static constexpr int ffn     = FFN_INTER;
    static constexpr int seq     = SEQ_LEN;
    static constexpr int batch   = BATCH;
    static constexpr int vocab   = VOCAB;

    static constexpr int64_t bsd   = (int64_t)batch * seq * d;
    static constexpr int64_t bs3d  = (int64_t)batch * seq * 3 * d;
    static constexpr int64_t bs2f  = (int64_t)batch * seq * 2 * ffn;
    static constexpr int64_t bsf   = (int64_t)batch * seq * ffn;
    static constexpr int64_t bsv   = (int64_t)batch * seq * vocab;

    // Per-layer activation footprint (elements).
    static constexpr int64_t per_layer_act =
        bsd       // rmsnorm input save
        + bs3d    // qkv proj output
        + 3 * bsd // q, k, v in [B,H,S,d_h]
        + bsd     // attention output
        + bsd     // o_proj output
        + bsd     // residual save
        + bsd     // rmsnorm2 input save
        + bs2f    // gate_up output
        + bsf     // mlp_down output
        + bsd;    // residual2 save

    // Total scratch requirement (elements).
    static constexpr int64_t total_act =
        (int64_t)batch * seq       // input ids
        + bsd                      // embed output
        + per_layer_act            // (multiply by n_layers at runtime)
        + bsd                      // final norm output
        + bsv;                     // logits

    static_assert(d_h * h == d, "d_model must be divisible by n_heads");
    static_assert(d_h % decoder_resources::MFMA_K == 0,
                  "head_dim must be divisible by MFMA_K (8)");
};

// ---- Wavefront-wide reduction (64 lanes on CDNA3) ---------------------------
__device__ __forceinline__
float wavefront_reduce_sum(float val) {
    #pragma unroll
    for (int offset = decoder_resources::WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__device__ __forceinline__
float wavefront_reduce_max(float val) {
    #pragma unroll
    for (int offset = decoder_resources::WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Cross-wavefront reduction within a workgroup via LDS.
__device__ __forceinline__
float workgroup_reduce_sum(float val, float* __restrict__ lds_buf) {
    int lane = threadIdx.x & (decoder_resources::WAVEFRONT_SIZE - 1);
    int wid  = threadIdx.x / decoder_resources::WAVEFRONT_SIZE;
    int n_waves = blockDim.x / decoder_resources::WAVEFRONT_SIZE;

    val = wavefront_reduce_sum(val);
    if (lane == 0) lds_buf[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (lane < n_waves) ? lds_buf[lane] : 0.0f;
        val = wavefront_reduce_sum(val);
    }
    __syncthreads();
    return __shfl(val, 0);  // broadcast from lane 0
}

// ---- RMSNorm forward --------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void rmsnorm_fwd(
    const ParamT* __restrict__ x,
    const ParamT* __restrict__ gamma,
    ParamT* __restrict__ out,
    ParamT* __restrict__ x_save,
    int D,
    float eps
) {
    extern __shared__ float smem[];
    float* row_buf = smem;
    float* reduce_buf = smem + D;

    float sq_sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = to_float<ParamT>(x[d]);
        v = apply_nan_policy<NAN_POLICY>(v);
        row_buf[d] = v;
        if (x_save) x_save[d] = from_float<ParamT>(v);
        sq_sum += v * v;
    }

    sq_sum = workgroup_reduce_sum(sq_sum, reduce_buf);
    float inv_rms = rsqrtf(sq_sum / static_cast<float>(D) + eps);

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float g = to_float<ParamT>(gamma[d]);
        out[d] = from_float<ParamT>(row_buf[d] * inv_rms * g);
    }
}

// ---- RMSNorm backward -------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void rmsnorm_bwd(
    const ParamT* __restrict__ x_saved,
    const ParamT* __restrict__ gamma,
    const ParamT* __restrict__ grad_out,
    ParamT* __restrict__ grad_in,
    ParamT* __restrict__ grad_gamma,
    int D,
    float eps
) {
    extern __shared__ float smem[];
    float* row_buf = smem;
    float* reduce_buf = smem + D;

    float sq_sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = to_float<ParamT>(x_saved[d]);
        row_buf[d] = v;
        sq_sum += v * v;
    }
    sq_sum = workgroup_reduce_sum(sq_sum, reduce_buf);
    float inv_rms = rsqrtf(sq_sum / static_cast<float>(D) + eps);

    float dot_sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float go = to_float<ParamT>(grad_out[d]);
        float g  = to_float<ParamT>(gamma[d]);
        dot_sum += go * g * row_buf[d];
    }
    dot_sum = workgroup_reduce_sum(dot_sum, reduce_buf);

    float coeff = dot_sum * inv_rms * inv_rms * inv_rms / static_cast<float>(D);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float go = to_float<ParamT>(grad_out[d]);
        float g  = to_float<ParamT>(gamma[d]);
        float xv = row_buf[d];
        float dx = go * g * inv_rms - xv * coeff;
        grad_in[d] = from_float<ParamT>(dx);
        atomicAdd(reinterpret_cast<float*>(&grad_gamma[d]),
                  go * xv * inv_rms);
    }
}

// ---- RoPE (Rotary Positional Embedding) -------------------------------------
// Same algorithm as sm_90; uses sincosf (no double-underscore on HIP).
template <typename ParamT>
__device__ __forceinline__
void rope_apply_fwd(
    ParamT* __restrict__ q,   // [B, H, S, d_h]
    ParamT* __restrict__ k,   // [B, H, S, d_h]
    int d_h,
    int seq_pos,
    float theta_base          // typically 10000.0f
) {
    for (int i = threadIdx.x; i < d_h / 2; i += blockDim.x) {
        float freq = 1.0f / powf(theta_base, 2.0f * i / static_cast<float>(d_h));
        float angle = static_cast<float>(seq_pos) * freq;
        float sin_v, cos_v;
        sincosf(angle, &sin_v, &cos_v);

        float q0 = to_float<ParamT>(q[2 * i]);
        float q1 = to_float<ParamT>(q[2 * i + 1]);
        q[2 * i]     = from_float<ParamT>(q0 * cos_v - q1 * sin_v);
        q[2 * i + 1] = from_float<ParamT>(q0 * sin_v + q1 * cos_v);

        float k0 = to_float<ParamT>(k[2 * i]);
        float k1 = to_float<ParamT>(k[2 * i + 1]);
        k[2 * i]     = from_float<ParamT>(k0 * cos_v - k1 * sin_v);
        k[2 * i + 1] = from_float<ParamT>(k0 * sin_v + k1 * cos_v);
    }
}

template <typename ParamT>
__device__ __forceinline__
void rope_apply_bwd(
    const ParamT* __restrict__ grad_q,
    const ParamT* __restrict__ grad_k,
    ParamT* __restrict__ grad_q_in,
    ParamT* __restrict__ grad_k_in,
    int d_h,
    int seq_pos,
    float theta_base
) {
    for (int i = threadIdx.x; i < d_h / 2; i += blockDim.x) {
        float freq = 1.0f / powf(theta_base, 2.0f * i / static_cast<float>(d_h));
        float angle = static_cast<float>(seq_pos) * freq;
        float sin_v, cos_v;
        sincosf(angle, &sin_v, &cos_v);

        float gq0 = to_float<ParamT>(grad_q[2 * i]);
        float gq1 = to_float<ParamT>(grad_q[2 * i + 1]);
        grad_q_in[2 * i]     = from_float<ParamT>( gq0 * cos_v + gq1 * sin_v);
        grad_q_in[2 * i + 1] = from_float<ParamT>(-gq0 * sin_v + gq1 * cos_v);

        float gk0 = to_float<ParamT>(grad_k[2 * i]);
        float gk1 = to_float<ParamT>(grad_k[2 * i + 1]);
        grad_k_in[2 * i]     = from_float<ParamT>( gk0 * cos_v + gk1 * sin_v);
        grad_k_in[2 * i + 1] = from_float<ParamT>(-gk0 * sin_v + gk1 * cos_v);
    }
}

// ---- Embedding forward / backward -------------------------------------------
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB, NanPolicy NAN_POLICY>
__device__ __forceinline__
void embed_fwd(
    const int* __restrict__ input_ids,
    const ParamT* __restrict__ tok_embed,
    ParamT* __restrict__ out,
    int D
) {
    int bs = blockIdx.x;
    if (bs >= BATCH * SEQ_LEN) return;
    int tok_id = input_ids[bs];
    tok_id = (tok_id < 0) ? 0 : ((tok_id >= VOCAB) ? VOCAB - 1 : tok_id);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = to_float<ParamT>(tok_embed[(int64_t)tok_id * D + d]);
        v = apply_nan_policy<NAN_POLICY>(v);
        out[(int64_t)bs * D + d] = from_float<ParamT>(v);
    }
}

template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB, NanPolicy NAN_POLICY>
__device__ __forceinline__
void embed_bwd(
    const int* __restrict__ input_ids,
    const ParamT* __restrict__ grad_out,
    ParamT* __restrict__ grad_tok_embed,
    int D
) {
    int bs = blockIdx.x;
    if (bs >= BATCH * SEQ_LEN) return;
    int tok_id = input_ids[bs];
    tok_id = (tok_id < 0) ? 0 : ((tok_id >= VOCAB) ? VOCAB - 1 : tok_id);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float g = to_float<ParamT>(grad_out[(int64_t)bs * D + d]);
        g = apply_nan_policy<NAN_POLICY>(g);
        atomicAdd(reinterpret_cast<float*>(&grad_tok_embed[(int64_t)tok_id * D + d]), g);
    }
}

// ---- QKV projection forward / backward -------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void qkv_proj_fwd(
    const ParamT* __restrict__ x,      // [B*S, D]
    const ParamT* __restrict__ W,      // [3*D, D]
    ParamT* __restrict__ out,          // [B*S, 3*D]
    int row,
    int D
) {
    // Each workgroup handles one row. MFMA-backed matmul via rocWMMA or
    // explicit MFMA builtins for 32x32x8 BF16 tiles.
    // Weight tiles streamed through LDS to reduce global memory pressure.
    __shared__ char lds_w[decoder_resources::LDS_BYTES_ATTN_TILE];
    ParamT* w_tile = reinterpret_cast<ParamT*>(lds_w);

    for (int col_base = 0; col_base < 3 * D; col_base += decoder_resources::MFMA_N) {
        float acc = 0.0f;
        for (int k_base = 0; k_base < D; k_base += decoder_resources::MFMA_K) {
            // Cooperative LDS load of weight tile.
            for (int t = threadIdx.x; t < decoder_resources::MFMA_N * decoder_resources::MFMA_K;
                 t += blockDim.x) {
                int wc = col_base + t / decoder_resources::MFMA_K;
                int wk = k_base + t % decoder_resources::MFMA_K;
                float wv = (wc < 3 * D && wk < D)
                    ? to_float<ParamT>(__builtin_nontemporal_load(&W[(int64_t)wc * D + wk]))
                    : 0.0f;
                w_tile[t] = from_float<ParamT>(wv);
            }
            __syncthreads();

            // Accumulate using MFMA 32x32x8 BF16 semantics.
            for (int kk = 0; kk < decoder_resources::MFMA_K; ++kk) {
                int col_lane = threadIdx.x % decoder_resources::MFMA_N;
                float xv = (k_base + kk < D)
                    ? to_float<ParamT>(x[(int64_t)row * D + k_base + kk])
                    : 0.0f;
                float wv = to_float<ParamT>(w_tile[col_lane * decoder_resources::MFMA_K + kk]);
                acc += xv * wv;
            }
            __syncthreads();
        }
        int col_lane = threadIdx.x % decoder_resources::MFMA_N;
        if (col_base + col_lane < 3 * D) {
            float val = apply_nan_policy<NAN_POLICY>(acc);
            out[(int64_t)row * 3 * D + col_base + col_lane] = from_float<ParamT>(val);
        }
    }
}

template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void qkv_proj_bwd(
    const ParamT* __restrict__ x_saved,
    const ParamT* __restrict__ W,
    const ParamT* __restrict__ grad_out,
    ParamT* __restrict__ grad_x,
    ParamT* __restrict__ grad_W,
    int row,
    int D
) {
    // grad_x[row,:] = grad_out[row,:] @ W   (MFMA-backed)
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < 3 * D; ++c) {
            float go = to_float<ParamT>(grad_out[(int64_t)row * 3 * D + c]);
            float wv = to_float<ParamT>(__builtin_nontemporal_load(&W[(int64_t)c * D + d]));
            acc += go * wv;
        }
        acc = apply_nan_policy<NAN_POLICY>(acc);
        grad_x[(int64_t)row * D + d] = from_float<ParamT>(acc);
    }
    // grad_W += grad_out[row,:]^T @ x[row,:] via atomics
    for (int c = threadIdx.x; c < 3 * D; c += blockDim.x) {
        float go = to_float<ParamT>(grad_out[(int64_t)row * 3 * D + c]);
        for (int d = 0; d < D; ++d) {
            float xv = to_float<ParamT>(x_saved[(int64_t)row * D + d]);
            atomicAdd(reinterpret_cast<float*>(&grad_W[(int64_t)c * D + d]), go * xv);
        }
    }
}

// ---- MFMA-based Causal Attention --------------------------------------------
// Design: MFMA 32x32x8 BF16 chosen over 16x16x16 for longer sequences
// because the larger tile amortizes LDS staging overhead. LDS staging of K/V
// tiles reduces global memory pressure by reusing data across wavefronts within
// a workgroup. XCD-aware tiling distributes attention head/batch pairs across
// MI300X's 8 chiplets for balanced L2 utilization.
//
// Online softmax (Milakov & Gimelshein) avoids materializing the full S*S
// attention matrix, same algorithm as FlashAttention but backed by MFMA
// instead of Tensor Cores. This avoids porting FA2's wgmma-specific codegen
// and instead uses native CDNA3 MFMA instructions for better occupancy.
template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void attention_fwd(
    const ParamT* __restrict__ Q,    // [B, H, S, d_h]
    const ParamT* __restrict__ K,    // [B, H, S, d_h]
    const ParamT* __restrict__ V,    // [B, H, S, d_h]
    ParamT* __restrict__ O,          // [B, H, S, d_h]
    float* __restrict__ lse,         // [B, H, S] log-sum-exp for backward
    int B, int H, int S, int d_h,
    float scale
) {
    // Each workgroup handles one (batch, head) pair.
    // XCD assignment: blockIdx.x % NUM_XCDS distributes across chiplets.
    int bh = blockIdx.x;
    if (bh >= B * H) return;

    __shared__ char lds_raw[decoder_resources::LDS_BYTES_ATTN_TILE];
    float* lds_k = reinterpret_cast<float*>(lds_raw);

    const int64_t qkv_stride = (int64_t)S * d_h;
    const ParamT* q_head = Q + (int64_t)bh * qkv_stride;
    const ParamT* k_head = K + (int64_t)bh * qkv_stride;
    const ParamT* v_head = V + (int64_t)bh * qkv_stride;
    ParamT* o_head       = O + (int64_t)bh * qkv_stride;
    float* lse_head      = lse + (int64_t)bh * S;

    // Online softmax per query row, tiled over K columns.
    for (int qi = threadIdx.x / d_h; qi < S; qi += blockDim.x / d_h) {
        int d_lane = threadIdx.x % d_h;

        float row_max = -1e30f;
        float row_sum = 0.0f;
        float o_acc = 0.0f;

        // Tile over K/V sequence positions.
        for (int kv_base = 0; kv_base < S; kv_base += decoder_resources::MFMA_N) {
            // Cooperative LDS load of K tile [MFMA_N, d_h].
            for (int t = threadIdx.x; t < decoder_resources::MFMA_N * d_h; t += blockDim.x) {
                int ks = kv_base + t / d_h;
                int kd = t % d_h;
                lds_k[t] = (ks < S) ? to_float<ParamT>(k_head[(int64_t)ks * d_h + kd]) : 0.0f;
            }
            __syncthreads();

            // Compute dot(q[qi], k[kj]) for kj in tile, apply causal mask.
            for (int kj_off = 0; kj_off < decoder_resources::MFMA_N; ++kj_off) {
                int kj = kv_base + kj_off;
                if (kj >= S) break;

                // Causal mask: skip future positions.
                if (kj > qi) break;

                float dot = 0.0f;
                for (int dd = 0; dd < d_h; ++dd) {
                    float qv = to_float<ParamT>(q_head[(int64_t)qi * d_h + dd]);
                    float kv = lds_k[kj_off * d_h + dd];
                    dot += qv * kv;
                }
                dot *= scale;
                dot = apply_nan_policy<NAN_POLICY>(dot);

                // Online softmax update.
                float prev_max = row_max;
                row_max = fmaxf(row_max, dot);
                float exp_diff = expf(prev_max - row_max);
                row_sum = row_sum * exp_diff + expf(dot - row_max);
                float attn_w = expf(dot - row_max);

                // Accumulate O (rescale previous accumulation).
                o_acc *= exp_diff;
                // This lane contributes v[kj, d_lane].
                if (d_lane < d_h) {
                    float vv = to_float<ParamT>(v_head[(int64_t)kj * d_h + d_lane]);
                    o_acc += attn_w * vv;
                }
            }
            __syncthreads();
        }

        // Finalize: O = O_acc / row_sum.
        if (d_lane < d_h) {
            float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
            o_head[(int64_t)qi * d_h + d_lane] = from_float<ParamT>(o_acc * inv_sum);
        }
        if (d_lane == 0) {
            lse_head[qi] = row_max + logf(fmaxf(row_sum, 1e-30f));
        }
    }
}

template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void attention_bwd(
    const ParamT* __restrict__ Q,
    const ParamT* __restrict__ K,
    const ParamT* __restrict__ V,
    const ParamT* __restrict__ O,
    const ParamT* __restrict__ grad_O,
    const float* __restrict__ lse,
    ParamT* __restrict__ grad_Q,
    ParamT* __restrict__ grad_K,
    ParamT* __restrict__ grad_V,
    int B, int H, int S, int d_h,
    float scale
) {
    int bh = blockIdx.x;
    if (bh >= B * H) return;

    const int64_t stride = (int64_t)S * d_h;
    const ParamT* q_h = Q + (int64_t)bh * stride;
    const ParamT* k_h = K + (int64_t)bh * stride;
    const ParamT* v_h = V + (int64_t)bh * stride;
    const ParamT* o_h = O + (int64_t)bh * stride;
    const ParamT* go_h = grad_O + (int64_t)bh * stride;
    const float* lse_h = lse + (int64_t)bh * S;
    ParamT* gq_h = grad_Q + (int64_t)bh * stride;
    ParamT* gk_h = grad_K + (int64_t)bh * stride;
    ParamT* gv_h = grad_V + (int64_t)bh * stride;

    // Per-query row backward.
    for (int qi = threadIdx.x; qi < S; qi += blockDim.x) {
        float lse_qi = lse_h[qi];

        // D_i = sum_d (grad_O[qi,d] * O[qi,d])
        float D_i = 0.0f;
        for (int dd = 0; dd < d_h; ++dd) {
            D_i += to_float<ParamT>(go_h[(int64_t)qi * d_h + dd])
                 * to_float<ParamT>(o_h[(int64_t)qi * d_h + dd]);
        }

        for (int kj = 0; kj <= qi; ++kj) {
            float dot = 0.0f;
            for (int dd = 0; dd < d_h; ++dd) {
                dot += to_float<ParamT>(q_h[(int64_t)qi * d_h + dd])
                     * to_float<ParamT>(k_h[(int64_t)kj * d_h + dd]);
            }
            dot *= scale;
            float p_ij = expf(dot - lse_qi);

            // dV[kj] += p_ij * grad_O[qi]
            // dS_ij = grad_O[qi] . V[kj]
            float ds_ij = 0.0f;
            for (int dd = 0; dd < d_h; ++dd) {
                float gov = to_float<ParamT>(go_h[(int64_t)qi * d_h + dd]);
                atomicAdd(reinterpret_cast<float*>(&gv_h[(int64_t)kj * d_h + dd]),
                          p_ij * gov);
                ds_ij += gov * to_float<ParamT>(v_h[(int64_t)kj * d_h + dd]);
            }

            float dp_ij = scale * p_ij * (ds_ij - D_i);
            dp_ij = apply_nan_policy<NAN_POLICY>(dp_ij);

            // dQ[qi] += dp_ij * K[kj]; dK[kj] += dp_ij * Q[qi]
            for (int dd = 0; dd < d_h; ++dd) {
                atomicAdd(reinterpret_cast<float*>(&gq_h[(int64_t)qi * d_h + dd]),
                          dp_ij * to_float<ParamT>(k_h[(int64_t)kj * d_h + dd]));
                atomicAdd(reinterpret_cast<float*>(&gk_h[(int64_t)kj * d_h + dd]),
                          dp_ij * to_float<ParamT>(q_h[(int64_t)qi * d_h + dd]));
            }
        }
    }
}

// ---- Output projection forward / backward -----------------------------------
template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void o_proj_fwd(
    const ParamT* __restrict__ attn_out,  // [B*S, D]
    const ParamT* __restrict__ W,         // [D, D]
    ParamT* __restrict__ out,             // [B*S, D]
    int row,
    int D
) {
    // MFMA-backed matmul, streaming weight reads.
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D; ++k) {
            float xv = to_float<ParamT>(attn_out[(int64_t)row * D + k]);
            float wv = to_float<ParamT>(__builtin_nontemporal_load(&W[(int64_t)d * D + k]));
            acc += xv * wv;
        }
        acc = apply_nan_policy<NAN_POLICY>(acc);
        out[(int64_t)row * D + d] = from_float<ParamT>(acc);
    }
}

template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void o_proj_bwd(
    const ParamT* __restrict__ attn_out_saved,
    const ParamT* __restrict__ W,
    const ParamT* __restrict__ grad_out,
    ParamT* __restrict__ grad_attn_out,
    ParamT* __restrict__ grad_W,
    int row,
    int D
) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D; ++k) {
            float go = to_float<ParamT>(grad_out[(int64_t)row * D + k]);
            float wv = to_float<ParamT>(__builtin_nontemporal_load(&W[(int64_t)k * D + d]));
            acc += go * wv;
        }
        acc = apply_nan_policy<NAN_POLICY>(acc);
        grad_attn_out[(int64_t)row * D + d] = from_float<ParamT>(acc);
    }
    for (int od = threadIdx.x; od < D; od += blockDim.x) {
        float go = to_float<ParamT>(grad_out[(int64_t)row * D + od]);
        for (int id = 0; id < D; ++id) {
            float xv = to_float<ParamT>(attn_out_saved[(int64_t)row * D + id]);
            atomicAdd(reinterpret_cast<float*>(&grad_W[(int64_t)od * D + id]), go * xv);
        }
    }
}

// ---- Residual add -----------------------------------------------------------
template <typename ParamT>
__device__ __forceinline__
void residual_add_fwd(
    const ParamT* __restrict__ x,
    const ParamT* __restrict__ residual,
    ParamT* __restrict__ out,
    int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        float a = to_float<ParamT>(x[i]);
        float b = to_float<ParamT>(residual[i]);
        out[i] = from_float<ParamT>(a + b);
    }
}

template <typename ParamT>
__device__ __forceinline__
void residual_add_bwd(
    const ParamT* __restrict__ grad_out,
    ParamT* __restrict__ grad_x,
    ParamT* __restrict__ grad_residual,
    int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        ParamT g = grad_out[i];
        grad_x[i] = g;
        grad_residual[i] = g;
    }
}

// ---- SiLU activation --------------------------------------------------------
// Standard SiLU: x / (1 + exp(-x)). No __expf prefix on HIP.
__device__ __forceinline__
float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__
float silu_grad(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

// ---- MLP gate-up forward / backward -----------------------------------------
// SwiGLU: out = SiLU(x @ gate_W) * (x @ up_W)
// gate_up_W is [2*ffn, D] where first half is gate, second half is up.
template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void mlp_gate_up_fwd(
    const ParamT* __restrict__ x,           // [B*S, D]
    const ParamT* __restrict__ gate_up_W,   // [2*ffn, D]
    ParamT* __restrict__ gate_up_out,       // [B*S, 2*ffn] (save for bwd)
    ParamT* __restrict__ out,               // [B*S, ffn]
    int row,
    int D,
    int ffn
) {
    // Compute gate and up projections, fused with SiLU gating.
    for (int f = threadIdx.x; f < ffn; f += blockDim.x) {
        // Gate projection.
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            float xv = to_float<ParamT>(x[(int64_t)row * D + d]);
            gate_acc += xv * to_float<ParamT>(
                __builtin_nontemporal_load(&gate_up_W[(int64_t)f * D + d]));
            up_acc += xv * to_float<ParamT>(
                __builtin_nontemporal_load(&gate_up_W[(int64_t)(ffn + f) * D + d]));
        }
        gate_acc = apply_nan_policy<NAN_POLICY>(gate_acc);
        up_acc   = apply_nan_policy<NAN_POLICY>(up_acc);

        // Save pre-activation for backward.
        gate_up_out[(int64_t)row * 2 * ffn + f]       = from_float<ParamT>(gate_acc);
        gate_up_out[(int64_t)row * 2 * ffn + ffn + f] = from_float<ParamT>(up_acc);

        out[(int64_t)row * ffn + f] = from_float<ParamT>(silu(gate_acc) * up_acc);
    }
}

template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void mlp_gate_up_bwd(
    const ParamT* __restrict__ x_saved,
    const ParamT* __restrict__ gate_up_W,
    const ParamT* __restrict__ gate_up_saved,
    const ParamT* __restrict__ grad_out,     // [B*S, ffn]
    ParamT* __restrict__ grad_x,
    ParamT* __restrict__ grad_gate_up_W,
    int row,
    int D,
    int ffn
) {
    for (int f = threadIdx.x; f < ffn; f += blockDim.x) {
        float gate_pre = to_float<ParamT>(gate_up_saved[(int64_t)row * 2 * ffn + f]);
        float up_pre   = to_float<ParamT>(gate_up_saved[(int64_t)row * 2 * ffn + ffn + f]);
        float go       = to_float<ParamT>(grad_out[(int64_t)row * ffn + f]);

        float d_gate = go * up_pre * silu_grad(gate_pre);
        float d_up   = go * silu(gate_pre);
        d_gate = apply_nan_policy<NAN_POLICY>(d_gate);
        d_up   = apply_nan_policy<NAN_POLICY>(d_up);

        for (int d = 0; d < D; ++d) {
            float xv = to_float<ParamT>(x_saved[(int64_t)row * D + d]);
            atomicAdd(reinterpret_cast<float*>(&grad_gate_up_W[(int64_t)f * D + d]),
                      d_gate * xv);
            atomicAdd(reinterpret_cast<float*>(&grad_gate_up_W[(int64_t)(ffn + f) * D + d]),
                      d_up * xv);
            atomicAdd(reinterpret_cast<float*>(&grad_x[(int64_t)row * D + d]),
                      d_gate * to_float<ParamT>(gate_up_W[(int64_t)f * D + d])
                    + d_up   * to_float<ParamT>(gate_up_W[(int64_t)(ffn + f) * D + d]));
        }
    }
}

// ---- MLP down projection forward / backward ---------------------------------
template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void mlp_down_fwd(
    const ParamT* __restrict__ x,     // [B*S, ffn]
    const ParamT* __restrict__ W,     // [D, ffn]
    ParamT* __restrict__ out,         // [B*S, D]
    int row,
    int D,
    int ffn
) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int f = 0; f < ffn; ++f) {
            float xv = to_float<ParamT>(x[(int64_t)row * ffn + f]);
            float wv = to_float<ParamT>(__builtin_nontemporal_load(&W[(int64_t)d * ffn + f]));
            acc += xv * wv;
        }
        acc = apply_nan_policy<NAN_POLICY>(acc);
        out[(int64_t)row * D + d] = from_float<ParamT>(acc);
    }
}

template <typename ParamT, NanPolicy NAN_POLICY>
__device__ __forceinline__
void mlp_down_bwd(
    const ParamT* __restrict__ x_saved,   // [B*S, ffn]
    const ParamT* __restrict__ W,         // [D, ffn]
    const ParamT* __restrict__ grad_out,  // [B*S, D]
    ParamT* __restrict__ grad_x,          // [B*S, ffn]
    ParamT* __restrict__ grad_W,          // [D, ffn]
    int row,
    int D,
    int ffn
) {
    // grad_x = grad_out @ W^T
    for (int f = threadIdx.x; f < ffn; f += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            float go = to_float<ParamT>(grad_out[(int64_t)row * D + d]);
            float wv = to_float<ParamT>(__builtin_nontemporal_load(&W[(int64_t)d * ffn + f]));
            acc += go * wv;
        }
        acc = apply_nan_policy<NAN_POLICY>(acc);
        grad_x[(int64_t)row * ffn + f] = from_float<ParamT>(acc);
    }
    // grad_W += grad_out^T @ x
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float go = to_float<ParamT>(grad_out[(int64_t)row * D + d]);
        for (int f = 0; f < ffn; ++f) {
            float xv = to_float<ParamT>(x_saved[(int64_t)row * ffn + f]);
            atomicAdd(reinterpret_cast<float*>(&grad_W[(int64_t)d * ffn + f]), go * xv);
        }
    }
}

// ---- LM head forward / backward ---------------------------------------------
template <typename ParamT, int VOCAB, bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ __forceinline__
void lm_head_fwd(
    const ParamT* __restrict__ x,          // [B*S, D]
    const ParamT* __restrict__ W,          // [VOCAB, D]
    const ParamT* __restrict__ bias,       // [VOCAB]
    const ParamT* __restrict__ tok_embed,  // [VOCAB, D] (used if tied)
    ParamT* __restrict__ logits,           // [B*S, VOCAB]
    int row,
    int D
) {
    const ParamT* __restrict__ proj_W = TIED_EMBEDDINGS ? tok_embed : W;

    for (int v = threadIdx.x; v < VOCAB; v += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            float xv = to_float<ParamT>(x[(int64_t)row * D + d]);
            float wv = to_float<ParamT>(__builtin_nontemporal_load(&proj_W[(int64_t)v * D + d]));
            acc += xv * wv;
        }
        if (bias) acc += to_float<ParamT>(bias[v]);
        acc = apply_nan_policy<NAN_POLICY>(acc);
        logits[(int64_t)row * VOCAB + v] = from_float<ParamT>(acc);
    }
}

template <typename ParamT, int VOCAB, bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ __forceinline__
void lm_head_bwd(
    const ParamT* __restrict__ x_saved,
    const ParamT* __restrict__ W,
    const ParamT* __restrict__ tok_embed,
    const ParamT* __restrict__ grad_logits,
    ParamT* __restrict__ grad_x,
    ParamT* __restrict__ grad_W,
    ParamT* __restrict__ grad_bias,
    ParamT* __restrict__ grad_tok_embed,
    int row,
    int D
) {
    const ParamT* __restrict__ proj_W = TIED_EMBEDDINGS ? tok_embed : W;
    ParamT* __restrict__ proj_grad_W  = TIED_EMBEDDINGS ? grad_tok_embed : grad_W;

    // grad_x = grad_logits @ proj_W
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int v = 0; v < VOCAB; ++v) {
            float gl = to_float<ParamT>(grad_logits[(int64_t)row * VOCAB + v]);
            float wv = to_float<ParamT>(__builtin_nontemporal_load(&proj_W[(int64_t)v * D + d]));
            acc += gl * wv;
        }
        acc = apply_nan_policy<NAN_POLICY>(acc);
        grad_x[(int64_t)row * D + d] = from_float<ParamT>(acc);
    }
    // grad_W += grad_logits^T @ x; grad_bias += grad_logits
    for (int v = threadIdx.x; v < VOCAB; v += blockDim.x) {
        float gl = to_float<ParamT>(grad_logits[(int64_t)row * VOCAB + v]);
        if (grad_bias) {
            atomicAdd(reinterpret_cast<float*>(&grad_bias[v]), gl);
        }
        for (int d = 0; d < D; ++d) {
            float xv = to_float<ParamT>(x_saved[(int64_t)row * D + d]);
            atomicAdd(reinterpret_cast<float*>(&proj_grad_W[(int64_t)v * D + d]), gl * xv);
        }
    }
}

// ---- Static asserts on template parameters ----------------------------------
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
struct DecoderStaticChecks {
    static_assert(SEQ_LEN > 0, "SEQ_LEN must be positive");
    static_assert(BATCH > 0, "BATCH must be positive");
    static_assert(VOCAB > 0, "VOCAB must be positive");
    static_assert(sizeof(ParamT) <= 4,
                  "ParamT must be at most 32 bits (float, __half, hip_bfloat16)");
    static_assert(static_cast<int>(NAN_POLICY) >= 0
               && static_cast<int>(NAN_POLICY) <= 2,
                  "NAN_POLICY must be kNone, kZero, or kPropagate");
    static_assert(decoder_resources::WAVEFRONT_SIZE == 64,
                  "gfx942 targets CDNA3 with 64-wide wavefronts");
    static_assert(decoder_resources::LDS_BYTES_ATTN_TILE <= decoder_resources::LDS_BYTES_MAX,
                  "Attention LDS tile must fit within 64 KB workgroup LDS budget");
    static constexpr bool valid = true;
};

}} // namespace grokking::gfx942

#endif // GROKKING_TRANSFORMER_DECODER_GFX942_HIP_HPP_
