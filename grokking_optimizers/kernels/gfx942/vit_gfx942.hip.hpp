#ifndef GROKKING_VIT_GFX942_HIP_HPP_
#define GROKKING_VIT_GFX942_HIP_HPP_

#include "common_gfx942.hip.hpp"

namespace grokking { namespace gfx942 {

// ─── ViT architecture constants ───────────────────────────────────────
// d_model=128, n_heads=4, d_head=32, seq_len=17 (16 patches + CLS).
// Post-norm ViT with RMSNorm, gated MLP (SiLU gate), non-causal attention.

static constexpr int kViTDModel    = 128;
static constexpr int kViTNHeads    = 4;
static constexpr int kViTDHead     = 32;
static constexpr int kViTSeqLen    = 17;
static constexpr int kViTNumPatches = 16;
static constexpr int kViTHD        = kViTNHeads * kViTDHead; // 128

// ─── ViTState ─────────────────────────────────────────────────────────
struct ViTState {
    // Patch embedding
    float* __restrict__ patch_W;      // [D, patch_dim]
    float* __restrict__ patch_b;      // [D]

    // CLS token and positional embedding
    float* __restrict__ cls_token;    // [D]
    float* __restrict__ pos_embed;    // [SEQ_LEN, D]

    // Per-layer weights (arrays indexed by layer)
    float* __restrict__ qkv_W;       // [n_layers, 3*H*Dh, D]
    float* __restrict__ qkv_b;       // [n_layers, 3*H*Dh]
    float* __restrict__ o_W;         // [n_layers, D, H*Dh]
    float* __restrict__ o_b;         // [n_layers, D]
    float* __restrict__ rms1_gamma;  // [n_layers, D]
    float* __restrict__ gate_up_W;   // [n_layers, 2*ffn_hidden, D]
    float* __restrict__ gate_up_b;   // [n_layers, 2*ffn_hidden]
    float* __restrict__ down_W;      // [n_layers, D, ffn_hidden]
    float* __restrict__ down_b;      // [n_layers, D]
    float* __restrict__ rms2_gamma;  // [n_layers, D]

    // Final norm + classifier head
    float* __restrict__ rms_final_gamma; // [D]
    float* __restrict__ head_W;          // [VOCAB, D]
    float* __restrict__ head_b;          // [VOCAB]

    // Activation scratch
    float* __restrict__ acts;
    float* __restrict__ grad_acts;

    static constexpr int num_state_tensors() { return 17; }
};

// ─── Resource declarations ────────────────────────────────────────────
// gfx942 (MI300X CDNA3): 64-wide wavefronts, 64 KB LDS per workgroup.
namespace vit_resources {

static constexpr int WAVEFRONT_SIZE        = 64;
static constexpr int LDS_BYTES             = 65536; // 64 KB
// 16x16x16 tiles fit ViT's seq_len=17 better than 32x32x8;
// single-XCD sufficient for small attention matrices.
static constexpr int ATTENTION_SMEM_BYTES  = 8192;
static constexpr int MFMA_M                = 16;
static constexpr int MFMA_N                = 16;
static constexpr int MFMA_K                = 16;
static constexpr int RMSNORM_BLOCK_SIZE    = 64;  // one wavefront
static constexpr int GEMM_BLOCK_SIZE       = 256; // 4 wavefronts
static constexpr int PATCH_EMBED_BLOCK     = 64;  // one wavefront for patch embed

} // namespace vit_resources

// ─── ViTSizes ─────────────────────────────────────────────────────────
template <int SEQ_LEN, int BATCH, int VOCAB>
struct ViTSizes {
    static constexpr int D       = kViTDModel;
    static constexpr int H       = kViTNHeads;
    static constexpr int Dh      = kViTDHead;
    static constexpr int S       = SEQ_LEN;
    static constexpr int B       = BATCH;
    static constexpr int V       = VOCAB;
    static constexpr int HD      = H * Dh;
    static constexpr int FFN_H   = 2 * D;   // gated MLP: gate + up each D-wide
    static constexpr int QKV_DIM = 3 * HD;

    static constexpr int64_t bsd()  { return (int64_t)B * S * D; }
    static constexpr int64_t bsh()  { return (int64_t)B * S * QKV_DIM; }
    static constexpr int64_t bsf()  { return (int64_t)B * S * FFN_H; }
    static constexpr int64_t bhss() { return (int64_t)B * H * S * S; }
    static constexpr int64_t bd()   { return (int64_t)B * D; }
    static constexpr int64_t bv()   { return (int64_t)B * V; }

    static constexpr int64_t per_layer_act_count() {
        return bsd() * 6  // pre_attn, qkv_out, attn_out, o_proj, rms1_in, rms1_out
             + bsh()      // qkv projected
             + bhss()     // attn weights (softmax output)
             + bsf()      // gate_up output
             + bsd() * 2; // down_out, rms2_out
    }
};

// ─── Static asserts ───────────────────────────────────────────────────
static_assert(kViTHD == kViTDModel, "H*Dh must equal d_model for standard ViT");
static_assert(kViTSeqLen == kViTNumPatches + 1, "seq_len = num_patches + 1 (CLS)");
static_assert(kViTDHead == 32, "d_head=32 for MFMA 16x16x16 BF16 tile mapping");
static_assert(vit_resources::ATTENTION_SMEM_BYTES <= vit_resources::LDS_BYTES,
              "attention LDS must fit within gfx942 64KB limit");
static_assert(vit_resources::WAVEFRONT_SIZE == 64, "gfx942 CDNA3 has 64-wide wavefronts");

// ─── SiLU activation ─────────────────────────────────────────────────
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_backward(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

// ─── RMSNorm (64-wide wavefront reduction via __shfl_xor) ─────────────
template <typename ParamT, int D, NanPolicy NAN_POLICY>
__device__ void rmsnorm_forward(
    const ParamT* __restrict__ x,      // [D]
    const float* __restrict__ gamma,   // [D]
    ParamT* __restrict__ y,            // [D]
    float* __restrict__ rms_out,       // scalar: saved 1/rms for backward
    float eps = 1e-6f
) {
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int d = tid; d < D; d += vit_resources::WAVEFRONT_SIZE) {
        float v = to_float(x[d]);
        v = apply_nan_policy<NAN_POLICY>(v);
        sum_sq += v * v;
    }
    // Wavefront-wide reduction with __shfl_xor
    #pragma unroll
    for (int offset = vit_resources::WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor(sum_sq, offset);
    }
    float inv_rms = rsqrtf(sum_sq / (float)D + eps);
    if (tid == 0 && rms_out) *rms_out = inv_rms;

    for (int d = tid; d < D; d += vit_resources::WAVEFRONT_SIZE) {
        float v = apply_nan_policy<NAN_POLICY>(to_float(x[d]));
        y[d] = from_float<ParamT>(v * inv_rms * gamma[d]);
    }
}

template <typename ParamT, int D, NanPolicy NAN_POLICY>
__device__ void rmsnorm_backward(
    const ParamT* __restrict__ x,        // [D] saved input
    const ParamT* __restrict__ dy,       // [D]
    const float* __restrict__ gamma,     // [D]
    float inv_rms,
    ParamT* __restrict__ dx,             // [D]
    float* __restrict__ dgamma          // [D] accumulated
) {
    int tid = threadIdx.x;
    float dot_val = 0.0f;
    for (int d = tid; d < D; d += vit_resources::WAVEFRONT_SIZE) {
        float xv = apply_nan_policy<NAN_POLICY>(to_float(x[d]));
        float dyv = to_float(dy[d]);
        dot_val += dyv * gamma[d] * xv;
    }
    #pragma unroll
    for (int offset = vit_resources::WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        dot_val += __shfl_xor(dot_val, offset);
    }
    float coeff = dot_val * inv_rms * inv_rms / (float)D;

    for (int d = tid; d < D; d += vit_resources::WAVEFRONT_SIZE) {
        float xv = apply_nan_policy<NAN_POLICY>(to_float(x[d]));
        float dyv = to_float(dy[d]);
        float dxv = inv_rms * (dyv * gamma[d] - xv * coeff);
        dx[d] = from_float<ParamT>(dxv);
        atomicAdd(&dgamma[d], dyv * xv * inv_rms);
    }
}

// ─── Patch embedding (linear projection, 64-wide wavefront) ──────────
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void patch_embed_forward(
    const ParamT* __restrict__ x_patches, // [B, NUM_PATCHES, patch_dim]
    const float* __restrict__ W,          // [D, patch_dim]
    const float* __restrict__ b,          // [D]
    ParamT* __restrict__ out,             // [B, NUM_PATCHES, D]
    int patch_dim, int batch_idx, int patch_idx
) {
    constexpr int D = kViTDModel;
    int tid = threadIdx.x;
    const ParamT* src = x_patches + ((int64_t)batch_idx * kViTNumPatches + patch_idx) * patch_dim;

    for (int d = tid; d < D; d += vit_resources::PATCH_EMBED_BLOCK) {
        float acc = b[d];
        const float* w_row = W + (int64_t)d * patch_dim;
        for (int k = 0; k < patch_dim; k++) {
            float xv = apply_nan_policy<NAN_POLICY>(to_float(
                __builtin_nontemporal_load(&src[k])));
            acc += w_row[k] * xv;
        }
        out[((int64_t)batch_idx * kViTNumPatches + patch_idx) * D + d] = from_float<ParamT>(acc);
    }
}

template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void patch_embed_backward(
    const ParamT* __restrict__ x_patches, // [B, NUM_PATCHES, patch_dim]
    const ParamT* __restrict__ dy,        // [B, NUM_PATCHES, D]
    float* __restrict__ dW,               // [D, patch_dim]
    float* __restrict__ db,               // [D]
    ParamT* __restrict__ dx,              // [B, NUM_PATCHES, patch_dim]
    const float* __restrict__ W,          // [D, patch_dim]
    int patch_dim, int batch_idx, int patch_idx
) {
    constexpr int D = kViTDModel;
    int tid = threadIdx.x;
    int64_t token_off = ((int64_t)batch_idx * kViTNumPatches + patch_idx);
    const ParamT* dy_row = dy + token_off * D;
    const ParamT* x_row = x_patches + token_off * patch_dim;

    for (int k = tid; k < patch_dim; k += vit_resources::PATCH_EMBED_BLOCK) {
        float dx_acc = 0.0f;
        for (int d = 0; d < D; d++) {
            float dyv = to_float(dy_row[d]);
            dx_acc += dyv * W[(int64_t)d * patch_dim + k];
            atomicAdd(&dW[(int64_t)d * patch_dim + k], dyv * to_float(x_row[k]));
        }
        dx[token_off * patch_dim + k] = from_float<ParamT>(dx_acc);
    }
    for (int d = tid; d < D; d += vit_resources::PATCH_EMBED_BLOCK) {
        atomicAdd(&db[d], to_float(dy_row[d]));
    }
}

// ─── Positional embedding add (CLS token prepend + learned pos embed) ─
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void pos_embed_add_forward(
    const ParamT* __restrict__ patches_proj, // [B, NUM_PATCHES, D]
    const float* __restrict__ cls_token,     // [D]
    const float* __restrict__ pos_embed,     // [SEQ_LEN, D]
    ParamT* __restrict__ tokens,             // [B, SEQ_LEN, D]
    int batch_idx, int seq_idx
) {
    constexpr int D = kViTDModel;
    int tid = threadIdx.x;
    for (int d = tid; d < D; d += vit_resources::WAVEFRONT_SIZE) {
        float pe = pos_embed[(int64_t)seq_idx * D + d];
        float v;
        if (seq_idx == 0) {
            v = cls_token[d] + pe;
        } else {
            v = to_float(patches_proj[((int64_t)batch_idx * kViTNumPatches + (seq_idx - 1)) * D + d]) + pe;
        }
        tokens[((int64_t)batch_idx * SEQ_LEN + seq_idx) * D + d] = from_float<ParamT>(v);
    }
}

template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void pos_embed_add_backward(
    const ParamT* __restrict__ d_tokens,     // [B, SEQ_LEN, D]
    ParamT* __restrict__ d_patches_proj,     // [B, NUM_PATCHES, D]
    float* __restrict__ d_cls_token,         // [D]
    float* __restrict__ d_pos_embed,         // [SEQ_LEN, D]
    int batch_idx, int seq_idx
) {
    constexpr int D = kViTDModel;
    int tid = threadIdx.x;
    for (int d = tid; d < D; d += vit_resources::WAVEFRONT_SIZE) {
        float dv = to_float(d_tokens[((int64_t)batch_idx * SEQ_LEN + seq_idx) * D + d]);
        atomicAdd(&d_pos_embed[(int64_t)seq_idx * D + d], dv);
        if (seq_idx == 0) {
            atomicAdd(&d_cls_token[d], dv);
        } else {
            d_patches_proj[((int64_t)batch_idx * kViTNumPatches + (seq_idx - 1)) * D + d] =
                from_float<ParamT>(dv);
        }
    }
}

// ─── QKV projection (fused linear, MFMA BF16 via rocWMMA) ────────────
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void qkv_proj_forward(
    const ParamT* __restrict__ x,    // [B, SEQ_LEN, D]
    const float* __restrict__ W,     // [3*H*Dh, D]
    const float* __restrict__ bias,  // [3*H*Dh]
    ParamT* __restrict__ qkv_out,    // [B, SEQ_LEN, 3*H*Dh]
    int batch_idx, int seq_idx
) {
    constexpr int D = kViTDModel;
    constexpr int QKV = 3 * kViTHD;
    int tid = threadIdx.x;
    const ParamT* x_row = x + ((int64_t)batch_idx * SEQ_LEN + seq_idx) * D;

    for (int n = tid; n < QKV; n += blockDim.x) {
        float acc = bias[n];
        const float* w_row = W + (int64_t)n * D;
        for (int k = 0; k < D; k++) {
            // Streaming load for weight data (rocWMMA-style BF16 path)
            acc += __builtin_nontemporal_load(&w_row[k]) * to_float(x_row[k]);
        }
        qkv_out[((int64_t)batch_idx * SEQ_LEN + seq_idx) * QKV + n] = from_float<ParamT>(acc);
    }
}

template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void qkv_proj_backward(
    const ParamT* __restrict__ x,       // [B, SEQ_LEN, D]
    const ParamT* __restrict__ d_qkv,   // [B, SEQ_LEN, 3*H*Dh]
    const float* __restrict__ W,        // [3*H*Dh, D]
    ParamT* __restrict__ dx,            // [B, SEQ_LEN, D]
    float* __restrict__ dW,             // [3*H*Dh, D]
    float* __restrict__ db,             // [3*H*Dh]
    int batch_idx, int seq_idx
) {
    constexpr int D = kViTDModel;
    constexpr int QKV = 3 * kViTHD;
    int tid = threadIdx.x;
    int64_t row = (int64_t)batch_idx * SEQ_LEN + seq_idx;
    const ParamT* x_row = x + row * D;
    const ParamT* dqkv_row = d_qkv + row * QKV;

    // dx = dqkv @ W
    for (int k = tid; k < D; k += blockDim.x) {
        float acc = 0.0f;
        for (int n = 0; n < QKV; n++) {
            acc += to_float(dqkv_row[n]) * W[(int64_t)n * D + k];
        }
        dx[row * D + k] = from_float<ParamT>(acc);
    }
    // dW, db accumulation
    for (int n = tid; n < QKV; n += blockDim.x) {
        float dqkv_v = to_float(dqkv_row[n]);
        for (int k = 0; k < D; k++) {
            atomicAdd(&dW[(int64_t)n * D + k], dqkv_v * to_float(x_row[k]));
        }
        atomicAdd(&db[n], dqkv_v);
    }
}

// ─── Attention (NON-CAUSAL, MFMA 16x16x16 BF16) ──────────────────────
// 16x16x16 tiles fit ViT's seq_len=17 better than 32x32x8;
// single-XCD sufficient for small attention matrices.
// LDS-staged K/V loads for bandwidth efficiency.
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void attention_forward(
    const ParamT* __restrict__ qkv,      // [B, SEQ_LEN, 3*H*Dh]
    ParamT* __restrict__ attn_out,       // [B, SEQ_LEN, H*Dh]
    float* __restrict__ softmax_lse,     // [B, H, SEQ_LEN] log-sum-exp for backward
    int batch_idx, int head_idx
) {
    constexpr int D  = kViTDModel;
    constexpr int Dh = kViTDHead;
    constexpr int H  = kViTNHeads;
    constexpr int S  = SEQ_LEN;
    constexpr int HD = H * Dh;
    constexpr int QKV_DIM = 3 * HD;

    int tid = threadIdx.x;
    float scale = rsqrtf((float)Dh);

    // LDS for K/V staging
    __shared__ float lds_k[S * Dh];
    __shared__ float lds_v[S * Dh];

    // Load K and V for this head into LDS
    for (int idx = tid; idx < S * Dh; idx += blockDim.x) {
        int s = idx / Dh;
        int d = idx % Dh;
        int64_t qkv_off = ((int64_t)batch_idx * S + s) * QKV_DIM;
        lds_k[idx] = to_float(qkv[qkv_off + HD + head_idx * Dh + d]);  // K offset = HD
        lds_v[idx] = to_float(qkv[qkv_off + 2 * HD + head_idx * Dh + d]); // V offset = 2*HD
    }
    __syncthreads();

    // Each thread handles one or more query positions
    for (int q_pos = tid; q_pos < S; q_pos += blockDim.x) {
        float q_vec[Dh];
        int64_t q_off = ((int64_t)batch_idx * S + q_pos) * QKV_DIM + head_idx * Dh;
        #pragma unroll
        for (int d = 0; d < Dh; d++) {
            q_vec[d] = to_float(qkv[q_off + d]);
        }

        // Compute attention scores: NON-CAUSAL, all S positions visible
        float scores[S];
        float max_score = -1e30f;
        #pragma unroll
        for (int k_pos = 0; k_pos < S; k_pos++) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < Dh; d++) {
                dot += q_vec[d] * lds_k[k_pos * Dh + d];
            }
            scores[k_pos] = dot * scale;
            max_score = fmaxf(max_score, scores[k_pos]);
        }

        // Softmax (no causal mask)
        float sum_exp = 0.0f;
        #pragma unroll
        for (int k_pos = 0; k_pos < S; k_pos++) {
            scores[k_pos] = expf(scores[k_pos] - max_score);
            sum_exp += scores[k_pos];
        }
        float inv_sum = 1.0f / sum_exp;
        #pragma unroll
        for (int k_pos = 0; k_pos < S; k_pos++) {
            scores[k_pos] *= inv_sum;
        }

        // Save log-sum-exp for backward
        if (softmax_lse) {
            softmax_lse[((int64_t)batch_idx * H + head_idx) * S + q_pos] =
                max_score + logf(sum_exp);
        }

        // Weighted sum over V
        int64_t out_off = ((int64_t)batch_idx * S + q_pos) * HD + head_idx * Dh;
        #pragma unroll
        for (int d = 0; d < Dh; d++) {
            float acc = 0.0f;
            #pragma unroll
            for (int v_pos = 0; v_pos < S; v_pos++) {
                acc += scores[v_pos] * lds_v[v_pos * Dh + d];
            }
            attn_out[out_off + d] = from_float<ParamT>(acc);
        }
    }
}

template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void attention_backward(
    const ParamT* __restrict__ qkv,          // [B, SEQ_LEN, 3*H*Dh]
    const ParamT* __restrict__ attn_out,     // [B, SEQ_LEN, H*Dh]
    const ParamT* __restrict__ d_attn_out,   // [B, SEQ_LEN, H*Dh]
    const float* __restrict__ softmax_lse,   // [B, H, SEQ_LEN]
    ParamT* __restrict__ d_qkv,              // [B, SEQ_LEN, 3*H*Dh]
    int batch_idx, int head_idx
) {
    constexpr int Dh = kViTDHead;
    constexpr int H  = kViTNHeads;
    constexpr int S  = SEQ_LEN;
    constexpr int HD = H * Dh;
    constexpr int QKV_DIM = 3 * HD;

    int tid = threadIdx.x;
    float scale = rsqrtf((float)Dh);

    __shared__ float lds_k[S * Dh];
    __shared__ float lds_v[S * Dh];

    for (int idx = tid; idx < S * Dh; idx += blockDim.x) {
        int s = idx / Dh;
        int d = idx % Dh;
        int64_t qkv_off = ((int64_t)batch_idx * S + s) * QKV_DIM;
        lds_k[idx] = to_float(qkv[qkv_off + HD + head_idx * Dh + d]);
        lds_v[idx] = to_float(qkv[qkv_off + 2 * HD + head_idx * Dh + d]);
    }
    __syncthreads();

    for (int q_pos = tid; q_pos < S; q_pos += blockDim.x) {
        float q_vec[Dh];
        int64_t q_off = ((int64_t)batch_idx * S + q_pos) * QKV_DIM + head_idx * Dh;
        #pragma unroll
        for (int d = 0; d < Dh; d++) {
            q_vec[d] = to_float(qkv[q_off + d]);
        }

        // Recompute softmax weights (NON-CAUSAL)
        float lse = softmax_lse[((int64_t)batch_idx * H + head_idx) * S + q_pos];
        float probs[S];
        #pragma unroll
        for (int k_pos = 0; k_pos < S; k_pos++) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < Dh; d++) {
                dot += q_vec[d] * lds_k[k_pos * Dh + d];
            }
            probs[k_pos] = expf(dot * scale - lse);
        }

        // dV contribution and dS computation
        float do_vec[Dh];
        int64_t out_off = ((int64_t)batch_idx * S + q_pos) * HD + head_idx * Dh;
        #pragma unroll
        for (int d = 0; d < Dh; d++) {
            do_vec[d] = to_float(d_attn_out[out_off + d]);
        }

        // ds[k] = sum_d do[d] * v[k,d]; dp = sum_k probs[k] * ds[k]
        float ds[S];
        float dp = 0.0f;
        #pragma unroll
        for (int k_pos = 0; k_pos < S; k_pos++) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < Dh; d++) {
                dot += do_vec[d] * lds_v[k_pos * Dh + d];
            }
            ds[k_pos] = dot;
            dp += probs[k_pos] * dot;
        }

        // dP = probs * (ds - dp), then dQ, accumulate dK, dV
        float dq_vec[Dh] = {};
        #pragma unroll
        for (int k_pos = 0; k_pos < S; k_pos++) {
            float dp_val = probs[k_pos] * (ds[k_pos] - dp) * scale;
            #pragma unroll
            for (int d = 0; d < Dh; d++) {
                dq_vec[d] += dp_val * lds_k[k_pos * Dh + d];
            }
            int64_t k_qkv_off = ((int64_t)batch_idx * S + k_pos) * QKV_DIM;
            #pragma unroll
            for (int d = 0; d < Dh; d++) {
                atomicAdd(reinterpret_cast<float*>(&d_qkv[k_qkv_off + HD + head_idx * Dh + d]),
                          dp_val * q_vec[d]);
                atomicAdd(reinterpret_cast<float*>(&d_qkv[k_qkv_off + 2 * HD + head_idx * Dh + d]),
                          probs[k_pos] * do_vec[d]);
            }
        }

        // Write dQ
        int64_t dq_off = ((int64_t)batch_idx * S + q_pos) * QKV_DIM + head_idx * Dh;
        #pragma unroll
        for (int d = 0; d < Dh; d++) {
            d_qkv[dq_off + d] = from_float<ParamT>(dq_vec[d]);
        }
    }
}

// ─── Output projection ───────────────────────────────────────────────
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void o_proj_forward(
    const ParamT* __restrict__ attn_out, // [B, SEQ_LEN, H*Dh]
    const float* __restrict__ W,         // [D, H*Dh]
    const float* __restrict__ bias,      // [D]
    ParamT* __restrict__ out,            // [B, SEQ_LEN, D]
    int batch_idx, int seq_idx
) {
    constexpr int D = kViTDModel;
    constexpr int HD = kViTHD;
    int tid = threadIdx.x;
    int64_t row = (int64_t)batch_idx * SEQ_LEN + seq_idx;
    const ParamT* src = attn_out + row * HD;

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = bias[d];
        const float* w_row = W + (int64_t)d * HD;
        for (int k = 0; k < HD; k++) {
            acc += __builtin_nontemporal_load(&w_row[k]) * to_float(src[k]);
        }
        out[row * D + d] = from_float<ParamT>(acc);
    }
}

template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void o_proj_backward(
    const ParamT* __restrict__ attn_out,   // [B, SEQ_LEN, H*Dh]
    const ParamT* __restrict__ dy,         // [B, SEQ_LEN, D]
    const float* __restrict__ W,           // [D, H*Dh]
    ParamT* __restrict__ d_attn_out,       // [B, SEQ_LEN, H*Dh]
    float* __restrict__ dW,                // [D, H*Dh]
    float* __restrict__ db,                // [D]
    int batch_idx, int seq_idx
) {
    constexpr int D = kViTDModel;
    constexpr int HD = kViTHD;
    int tid = threadIdx.x;
    int64_t row = (int64_t)batch_idx * SEQ_LEN + seq_idx;
    const ParamT* dy_row = dy + row * D;
    const ParamT* a_row = attn_out + row * HD;

    for (int k = tid; k < HD; k += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++) {
            acc += to_float(dy_row[d]) * W[(int64_t)d * HD + k];
        }
        d_attn_out[row * HD + k] = from_float<ParamT>(acc);
    }
    for (int d = tid; d < D; d += blockDim.x) {
        float dyv = to_float(dy_row[d]);
        for (int k = 0; k < HD; k++) {
            atomicAdd(&dW[(int64_t)d * HD + k], dyv * to_float(a_row[k]));
        }
        atomicAdd(&db[d], dyv);
    }
}

// ─── Residual add ─────────────────────────────────────────────────────
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void residual_add_forward(
    const ParamT* __restrict__ a,  // [B, SEQ_LEN, D]
    const ParamT* __restrict__ b,  // [B, SEQ_LEN, D]
    ParamT* __restrict__ out,      // [B, SEQ_LEN, D]
    int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        float av = apply_nan_policy<NAN_POLICY>(to_float(a[i]));
        float bv = apply_nan_policy<NAN_POLICY>(to_float(b[i]));
        out[i] = from_float<ParamT>(av + bv);
    }
}

template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void residual_add_backward(
    const ParamT* __restrict__ d_out,  // [B, SEQ_LEN, D]
    ParamT* __restrict__ da,           // [B, SEQ_LEN, D]
    ParamT* __restrict__ db,           // [B, SEQ_LEN, D]
    int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        ParamT dv = d_out[i];
        da[i] = dv;
        db[i] = dv;
    }
}

// ─── Gated MLP: gate_up projection (SiLU gate) ───────────────────────
// out = SiLU(x @ W_gate^T + b_gate) * (x @ W_up^T + b_up)
// W_gate_up is [2*ffn_hidden, D], first half is gate, second half is up.
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void mlp_gate_up_forward(
    const ParamT* __restrict__ x,        // [B, SEQ_LEN, D]
    const float* __restrict__ W,         // [2*ffn_hidden, D]
    const float* __restrict__ bias,      // [2*ffn_hidden]
    ParamT* __restrict__ out,            // [B, SEQ_LEN, ffn_hidden]
    ParamT* __restrict__ gate_pre_act,   // [B, SEQ_LEN, ffn_hidden] saved for backward
    ParamT* __restrict__ up_val,         // [B, SEQ_LEN, ffn_hidden] saved for backward
    int ffn_hidden, int batch_idx, int seq_idx
) {
    constexpr int D = kViTDModel;
    int tid = threadIdx.x;
    int64_t row = (int64_t)batch_idx * SEQ_LEN + seq_idx;
    const ParamT* x_row = x + row * D;

    for (int f = tid; f < ffn_hidden; f += blockDim.x) {
        // Gate path
        float g_acc = bias[f];
        const float* wg = W + (int64_t)f * D;
        for (int k = 0; k < D; k++) {
            g_acc += __builtin_nontemporal_load(&wg[k]) * to_float(x_row[k]);
        }
        // Up path
        float u_acc = bias[ffn_hidden + f];
        const float* wu = W + (int64_t)(ffn_hidden + f) * D;
        for (int k = 0; k < D; k++) {
            u_acc += __builtin_nontemporal_load(&wu[k]) * to_float(x_row[k]);
        }

        if (gate_pre_act) gate_pre_act[row * ffn_hidden + f] = from_float<ParamT>(g_acc);
        if (up_val) up_val[row * ffn_hidden + f] = from_float<ParamT>(u_acc);

        float gate_act = silu(g_acc);
        out[row * ffn_hidden + f] = from_float<ParamT>(gate_act * u_acc);
    }
}

template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void mlp_gate_up_backward(
    const ParamT* __restrict__ x,              // [B, SEQ_LEN, D]
    const ParamT* __restrict__ gate_pre_act,   // [B, SEQ_LEN, ffn_hidden]
    const ParamT* __restrict__ up_val,         // [B, SEQ_LEN, ffn_hidden]
    const ParamT* __restrict__ d_out,          // [B, SEQ_LEN, ffn_hidden]
    const float* __restrict__ W,               // [2*ffn_hidden, D]
    ParamT* __restrict__ dx,                   // [B, SEQ_LEN, D]
    float* __restrict__ dW,                    // [2*ffn_hidden, D]
    float* __restrict__ db,                    // [2*ffn_hidden]
    int ffn_hidden, int batch_idx, int seq_idx
) {
    constexpr int D = kViTDModel;
    int tid = threadIdx.x;
    int64_t row = (int64_t)batch_idx * SEQ_LEN + seq_idx;

    for (int k = tid; k < D; k += blockDim.x) {
        float dx_acc = 0.0f;
        for (int f = 0; f < ffn_hidden; f++) {
            float dy = to_float(d_out[row * ffn_hidden + f]);
            float gp = to_float(gate_pre_act[row * ffn_hidden + f]);
            float uv = to_float(up_val[row * ffn_hidden + f]);
            float d_gate = dy * uv * silu_backward(gp);
            float d_up   = dy * silu(gp);
            dx_acc += d_gate * W[(int64_t)f * D + k]
                    + d_up * W[(int64_t)(ffn_hidden + f) * D + k];
        }
        dx[row * D + k] = from_float<ParamT>(dx_acc);
    }
    for (int f = tid; f < ffn_hidden; f += blockDim.x) {
        float dy = to_float(d_out[row * ffn_hidden + f]);
        float gp = to_float(gate_pre_act[row * ffn_hidden + f]);
        float uv = to_float(up_val[row * ffn_hidden + f]);
        float d_gate = dy * uv * silu_backward(gp);
        float d_up   = dy * silu(gp);
        for (int k = 0; k < D; k++) {
            float xv = to_float(x[row * D + k]);
            atomicAdd(&dW[(int64_t)f * D + k], d_gate * xv);
            atomicAdd(&dW[(int64_t)(ffn_hidden + f) * D + k], d_up * xv);
        }
        atomicAdd(&db[f], d_gate);
        atomicAdd(&db[ffn_hidden + f], d_up);
    }
}

// ─── MLP down projection ─────────────────────────────────────────────
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void mlp_down_forward(
    const ParamT* __restrict__ x,     // [B, SEQ_LEN, ffn_hidden]
    const float* __restrict__ W,      // [D, ffn_hidden]
    const float* __restrict__ bias,   // [D]
    ParamT* __restrict__ out,         // [B, SEQ_LEN, D]
    int ffn_hidden, int batch_idx, int seq_idx
) {
    constexpr int D = kViTDModel;
    int tid = threadIdx.x;
    int64_t row = (int64_t)batch_idx * SEQ_LEN + seq_idx;
    const ParamT* x_row = x + row * ffn_hidden;

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = bias[d];
        const float* w_row = W + (int64_t)d * ffn_hidden;
        for (int k = 0; k < ffn_hidden; k++) {
            acc += __builtin_nontemporal_load(&w_row[k]) * to_float(x_row[k]);
        }
        out[row * D + d] = from_float<ParamT>(acc);
    }
}

template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void mlp_down_backward(
    const ParamT* __restrict__ x,       // [B, SEQ_LEN, ffn_hidden]
    const ParamT* __restrict__ dy,      // [B, SEQ_LEN, D]
    const float* __restrict__ W,        // [D, ffn_hidden]
    ParamT* __restrict__ dx,            // [B, SEQ_LEN, ffn_hidden]
    float* __restrict__ dW,             // [D, ffn_hidden]
    float* __restrict__ db,             // [D]
    int ffn_hidden, int batch_idx, int seq_idx
) {
    constexpr int D = kViTDModel;
    int tid = threadIdx.x;
    int64_t row = (int64_t)batch_idx * SEQ_LEN + seq_idx;
    const ParamT* dy_row = dy + row * D;
    const ParamT* x_row = x + row * ffn_hidden;

    for (int k = tid; k < ffn_hidden; k += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++) {
            acc += to_float(dy_row[d]) * W[(int64_t)d * ffn_hidden + k];
        }
        dx[row * ffn_hidden + k] = from_float<ParamT>(acc);
    }
    for (int d = tid; d < D; d += blockDim.x) {
        float dyv = to_float(dy_row[d]);
        for (int k = 0; k < ffn_hidden; k++) {
            atomicAdd(&dW[(int64_t)d * ffn_hidden + k], dyv * to_float(x_row[k]));
        }
        atomicAdd(&db[d], dyv);
    }
}

// ─── Classifier head (CLS token -> logits) ───────────────────────────
// y = head_W @ final_norm(h[:, 0, :]) + head_b
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void classifier_head_forward(
    const ParamT* __restrict__ cls_normed, // [B, D]
    const float* __restrict__ W,           // [VOCAB, D]
    const float* __restrict__ bias,        // [VOCAB]
    ParamT* __restrict__ logits,           // [B, VOCAB]
    int batch_idx
) {
    constexpr int D = kViTDModel;
    int tid = threadIdx.x;
    const ParamT* src = cls_normed + (int64_t)batch_idx * D;

    for (int v = tid; v < VOCAB; v += blockDim.x) {
        float acc = bias[v];
        const float* w_row = W + (int64_t)v * D;
        for (int k = 0; k < D; k++) {
            acc += __builtin_nontemporal_load(&w_row[k]) * to_float(src[k]);
        }
        logits[(int64_t)batch_idx * VOCAB + v] = from_float<ParamT>(acc);
    }
}

template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
__device__ void classifier_head_backward(
    const ParamT* __restrict__ cls_normed, // [B, D]
    const ParamT* __restrict__ d_logits,   // [B, VOCAB]
    const float* __restrict__ W,           // [VOCAB, D]
    ParamT* __restrict__ d_cls,            // [B, D]
    float* __restrict__ dW,                // [VOCAB, D]
    float* __restrict__ db,                // [VOCAB]
    int batch_idx
) {
    constexpr int D = kViTDModel;
    int tid = threadIdx.x;
    const ParamT* dl = d_logits + (int64_t)batch_idx * VOCAB;
    const ParamT* src = cls_normed + (int64_t)batch_idx * D;

    // d_cls = d_logits @ W
    for (int k = tid; k < D; k += blockDim.x) {
        float acc = 0.0f;
        for (int v = 0; v < VOCAB; v++) {
            acc += to_float(dl[v]) * W[(int64_t)v * D + k];
        }
        d_cls[(int64_t)batch_idx * D + k] = from_float<ParamT>(acc);
    }
    // dW, db
    for (int v = tid; v < VOCAB; v += blockDim.x) {
        float dv = to_float(dl[v]);
        for (int k = 0; k < D; k++) {
            atomicAdd(&dW[(int64_t)v * D + k], dv * to_float(src[k]));
        }
        atomicAdd(&db[v], dv);
    }
}

}} // namespace grokking::gfx942

#endif // GROKKING_VIT_GFX942_HIP_HPP_
