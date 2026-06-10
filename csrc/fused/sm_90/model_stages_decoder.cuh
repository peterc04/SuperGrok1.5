#ifndef SG_FUSED_SM90_MODEL_STAGES_DECODER_CUH_
#define SG_FUSED_SM90_MODEL_STAGES_DECODER_CUH_
// ============================================================================
// csrc/fused/sm_90/model_stages_decoder.cuh — PHASE 1 of the TRUE L3 fused
// megakernel: the REAL transformer-decoder forward + backward as in-kernel
// stages of the persistent megakernel (replaces the element-local SURROGATE in
// model_stages.cuh for the (transformer_decoder × adamw) cell).
//
// HONESTY: this is the genuine eager architecture (grokking_race_v2.py
// _raw_model → Transformer), transcribed LINE-FOR-LINE from the verified
// PyTorch oracle in tests/hw/decoder_oracle.py, which is asserted bit-identical
// (fp64, ~1e-15 rel) to torch.autograd for the loss and EVERY one of the 30
// parameter gradients. There is NO placeholder math on this path. The one
// deliberate, documented deviation from the oracle is execution mechanics, not
// numerics: the backward RECOMPUTES each layer's forward from the cached layer
// INPUT (rather than caching every intermediate) to keep only one layer's
// activations live in smem at a time — see the smem-budget note below. The math
// computed is identical; only the storage strategy differs.
//
// ARCHITECTURE (exact; cites grokking_race_v2.py lines):
//   Transformer(nl=2, d=128, h=4, ntok=99, seq=4)  [_raw_model hardcodes seq=4]
//     tok = Embedding(99,128) (:360) + pos = Embedding(4,128) (:360)
//     2 × DecoderBlock (:346):
//       attn = MultiheadAttention(128,4,batch_first) (:349), causal mask
//              triu(diag=1) (:352): pos i attends j<=i
//       n1,n2 = LayerNorm(128) (:350), eps=1e-5 (torch default)
//       ff = Linear(128,512) → GELU(exact erf) → Linear(512,128) (:351)
//       POST-LN: x=n1(x+attn(x)); h=n2(x+ff(x))  (:354-355)
//     norm = LayerNorm(128) (:362); out = Linear(128,99) (:362)
//     forward: h=tok+pos; for l: h=l(h); return out(norm(h)[:,-1,:]) (:366-368)
//   loss = cross_entropy(logits, targets) (mean over batch; :745).
//
// NUMERICS (verified in the oracle; see decoder_oracle.py):
//   * GELU = exact erf  0.5x(1+erf(x/√2)); grad 0.5(1+erf(x/√2))+x·φ(x),
//     φ(x)=(1/√(2π))e^{-x²/2}.  (The surrogate's tanh approx is ~4e-4 off and
//     would blow the 1e-4 grad tolerance — NOT used here.)
//   * LayerNorm eps = 1e-5; dx uses the two mean-correction terms.
//   * attention scale 1/√dh = 1/√32; softmax subtracts the row max.
//   * cross-entropy MEAN over B → dlogits = (softmax−onehot)/B.
//
// PARALLELIZATION (locked design — Option B, batch-parallel, deterministic):
//   * Each CTA owns a FIXED contiguous batch slice (by blockIdx.x — NOT the
//     work-stealing task queue, which is param-tensor-parallel and would vary
//     the batch→CTA grouping; fp32 sums are not associative so a varying
//     grouping is non-deterministic). The whole CTA cooperates on ONE sample at
//     a time and iterates its slice SEQUENTIALLY.
//   * Per-sample, the CTA accumulates that sample's weight-grad contribution
//     into the CTA's OWN partial buffer in the global workspace
//     (gw + cta*total). Each dW element has a SINGLE owner thread that does
//     partial[e] += contrib across the slice's samples in fixed order
//     (__syncthreads between samples) — single-writer, no atomics, deterministic.
//   * One grid barrier, then a deterministic cross-CTA reduce: sum
//     partial[0..nCTA) in ASCENDING CTA index into the global grad buffer
//     (determinism lives in the per-element summation ORDER, not who computes it
//     — so the reduce REUSES the work-steal queue: whichever CTA grabs a param
//     element-range sums the partials for it in ascending CTA order).
//   * Another grid barrier, then the existing apply_optimizer<AdamW> tail
//     consumes the reduced grad in place.
//   * Loss: each CTA sums its slice's NLL (fp32) into a per-CTA loss slot; the
//     reduce sums those in fp64 and divides by B → one device float (the 1e-5
//     loss rel-tol is the tightest gate; fp32 atomic-summing 4191 terms misses it).
//
// SMEM BUDGET PER CTA (one sample, one layer live; recompute-in-backward):
//   The dominant per-sample buffer is the FFN hidden ff0/g [S=4 × 4d=512] =
//   2048 floats = 8 KB. With the QKV [4×384], ctx/x1/residual [4×128] and the LN
//   caches, one layer's live working set is ~25 KB — UNDER the 48 KB static-smem
//   cliff, so NO dynamic-smem opt-in is needed and the occupancy≥1 launch guard
//   in fused_megakernel.cuh (dynamicSMemBytes=0) is unchanged. Caching BOTH
//   layers (~50 KB) would cross 48 KB and force cudaFuncSetAttribute — avoided by
//   the per-layer recompute. The flat smem arena below is sized accordingly.
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/decoder_layout.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>

namespace sg { namespace fused { namespace sm90 {

// ── Decoder compile-time shape constants (mirror tests/hw/decoder_oracle.py and
//    csrc/fused/sm_90/decoder_layout.cuh; static_asserts there guard the count
//    + total against named_parameters()). ─────────────────────────────────────
namespace dec {
constexpr int kVocab  = SG_DEC_VOCAB;   // 99
constexpr int kD      = SG_DEC_D;       // 128
constexpr int kHeads  = SG_DEC_HEADS;   // 4
constexpr int kLayers = SG_DEC_LAYERS;  // 2
constexpr int kSeq    = SG_DEC_SEQ;     // 4
constexpr int kDff    = SG_DEC_DFF;     // 512
constexpr int kDhead  = kD / kHeads;    // 32
constexpr float kLnEps = 1e-5f;         // torch LayerNorm default
// Attention scale 1/sqrt(dhead). rsqrtf is fine; keep it a constexpr-ish value
// computed once on host-visible compile (1/sqrt(32)).
__device__ __forceinline__ float attn_scale() { return rsqrtf((float)kDhead); }
}  // namespace dec

// ── Exact-erf GELU + derivative (the real nn.GELU(), NOT the tanh surrogate). ──
__device__ __forceinline__ float dec_gelu(float x) {
    // 0.5 x (1 + erf(x/sqrt2)).  M_SQRT1_2 = 1/sqrt(2).
    return 0.5f * x * (1.0f + erff(x * (float)M_SQRT1_2));
}
__device__ __forceinline__ float dec_gelu_grad(float x) {
    // 0.5(1+erf(x/√2)) + x · φ(x), φ(x) = (1/√(2π)) e^{-x²/2}.
    const float cdf = 0.5f * (1.0f + erff(x * (float)M_SQRT1_2));
    const float inv_sqrt_2pi = 0.39894228040143270286f;  // 1/sqrt(2*pi)
    const float pdf = inv_sqrt_2pi * __expf(-0.5f * x * x);
    return cdf + x * pdf;
}

// ── Per-CTA scratch for one sample. POD held in smem (one instance per CTA).
//    Sizes chosen so the live set is one layer (recompute in backward). The
//    arrays that must survive fwd→bwd within a layer are kept; the rest are
//    recomputed. All [kSeq × width] row-major (row = position). ───────────────
struct DecSampleSmem {
    // Per-layer cached INPUT to each layer (h before the layer) — the only
    // cross-layer state kept, so the backward can recompute each layer's fwd.
    float layer_in[dec::kLayers][dec::kSeq][dec::kD];   // 2*4*128 = 1024 f
    // The final-norm input (last layer output, all positions) for bwd.
    float final_in[dec::kSeq][dec::kD];                 // 4*128 = 512 f
    // Scratch reused within a single layer's fwd or bwd (one layer live):
    float qkv[dec::kSeq][3 * dec::kD];                  // 4*384 = 1536 f
    float ctx[dec::kSeq][dec::kD];                      // 4*128 = 512 f
    float x1[dec::kSeq][dec::kD];                       // 4*128 = 512 f (post n1)
    float ff0[dec::kSeq][dec::kDff];                    // 4*512 = 2048 f (pre-gelu)
    float gact[dec::kSeq][dec::kDff];                   // 4*512 = 2048 f (post-gelu)
    float attn[dec::kHeads][dec::kSeq][dec::kSeq];      // 4*4*4 = 64 f
    // LayerNorm caches (xhat + inv_std) for n1, n2, final — per position.
    float n1_xhat[dec::kSeq][dec::kD];
    float n1_inv[dec::kSeq];
    float n2_xhat[dec::kSeq][dec::kD];
    float n2_inv[dec::kSeq];
    float fn_xhat[dec::kSeq][dec::kD];
    float fn_inv[dec::kSeq];
    // Backward adjoints carried across sub-steps (one layer live), per position.
    float dh[dec::kSeq][dec::kD];                       // running grad wrt block out
    float logits[dec::kVocab];                          // last-pos logits (bwd)
    // Dedicated attention-backward dscores buffer (NOT aliased over `attn`, so
    // the forward softmax weights stay live while dv/dq/dk read them). 64 f.
    float dsc[dec::kHeads][dec::kSeq][dec::kSeq];
    // Reductions across threads for row stats (mean / max / dot). 256 threads.
    float red[256];
};

// A typed view over the flat weight blob using the generated offsets. All
// pointers are into the single global `params` buffer (read-only in fwd/bwd).
struct DecWeights {
    const float* tok;     // [kVocab, kD]
    const float* pos;     // [kSeq, kD]
    struct Layer {
        const float* in_w;   // [3d, d]
        const float* in_b;   // [3d]
        const float* out_w;  // [d, d]
        const float* out_b;  // [d]
        const float* n1_w;   // [d]
        const float* n1_b;   // [d]
        const float* n2_w;   // [d]
        const float* n2_b;   // [d]
        const float* ff0_w;  // [4d, d]
        const float* ff0_b;  // [4d]
        const float* ff2_w;  // [d, 4d]
        const float* ff2_b;  // [d]
    } layer[dec::kLayers];
    const float* norm_w;  // [d]
    const float* norm_b;  // [d]
    const float* out_w;   // [kVocab, d]
    const float* out_b;   // [kVocab]
};

// Bind DecWeights to the flat params using the generated element offsets. The
// offsets array (csrc/fused/sm_90/decoder_layout.cuh::kDecOffsets) is the C++
// mirror of tests/hw/decoder_oracle.py::decoder_param_layout(), static-asserted.
__device__ __forceinline__ DecWeights dec_bind(const float* p) {
    DecWeights w;
    int i = 0;
    w.tok = p + kDecOffsets[i++];
    w.pos = p + kDecOffsets[i++];
    for (int li = 0; li < dec::kLayers; ++li) {
        w.layer[li].in_w  = p + kDecOffsets[i++];
        w.layer[li].in_b  = p + kDecOffsets[i++];
        w.layer[li].out_w = p + kDecOffsets[i++];
        w.layer[li].out_b = p + kDecOffsets[i++];
        w.layer[li].n1_w  = p + kDecOffsets[i++];
        w.layer[li].n1_b  = p + kDecOffsets[i++];
        w.layer[li].n2_w  = p + kDecOffsets[i++];
        w.layer[li].n2_b  = p + kDecOffsets[i++];
        w.layer[li].ff0_w = p + kDecOffsets[i++];
        w.layer[li].ff0_b = p + kDecOffsets[i++];
        w.layer[li].ff2_w = p + kDecOffsets[i++];
        w.layer[li].ff2_b = p + kDecOffsets[i++];
    }
    w.norm_w = p + kDecOffsets[i++];
    w.norm_b = p + kDecOffsets[i++];
    w.out_w  = p + kDecOffsets[i++];
    w.out_b  = p + kDecOffsets[i++];
    return w;
}

// Identical helper for the per-CTA grad PARTIAL buffer (writable, same layout).
struct DecGrad {
    float* tok; float* pos;
    struct Layer { float* in_w; float* in_b; float* out_w; float* out_b;
        float* n1_w; float* n1_b; float* n2_w; float* n2_b;
        float* ff0_w; float* ff0_b; float* ff2_w; float* ff2_b; } layer[dec::kLayers];
    float* norm_w; float* norm_b; float* out_w; float* out_b;
};
__device__ __forceinline__ DecGrad dec_bind_grad(float* p) {
    DecGrad w; int i = 0;
    w.tok = p + kDecOffsets[i++];
    w.pos = p + kDecOffsets[i++];
    for (int li = 0; li < dec::kLayers; ++li) {
        w.layer[li].in_w  = p + kDecOffsets[i++];
        w.layer[li].in_b  = p + kDecOffsets[i++];
        w.layer[li].out_w = p + kDecOffsets[i++];
        w.layer[li].out_b = p + kDecOffsets[i++];
        w.layer[li].n1_w  = p + kDecOffsets[i++];
        w.layer[li].n1_b  = p + kDecOffsets[i++];
        w.layer[li].n2_w  = p + kDecOffsets[i++];
        w.layer[li].n2_b  = p + kDecOffsets[i++];
        w.layer[li].ff0_w = p + kDecOffsets[i++];
        w.layer[li].ff0_b = p + kDecOffsets[i++];
        w.layer[li].ff2_w = p + kDecOffsets[i++];
        w.layer[li].ff2_b = p + kDecOffsets[i++];
    }
    w.norm_w = p + kDecOffsets[i++];
    w.norm_b = p + kDecOffsets[i++];
    w.out_w  = p + kDecOffsets[i++];
    w.out_b  = p + kDecOffsets[i++];
    return w;
}

// ── Block-wide reductions (256 threads) over a value `v`. The smem `red` slot
//    is the DecSampleSmem::red array. All threads must call. ──────────────────
__device__ __forceinline__ float dec_block_sum(float v, float* red) {
    const unsigned full = 0xffffffffu;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(full, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nwarp = (blockDim.x + 31) >> 5;
    float r = 0.0f;
    if (warp == 0) {
        float p = (lane < nwarp) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(full, p, o);
        if (lane == 0) red[0] = p;
    }
    __syncthreads();
    r = red[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float dec_block_max(float v, float* red) {
    const unsigned full = 0xffffffffu;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_down_sync(full, v, o);
        v = fmaxf(v, other);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nwarp = (blockDim.x + 31) >> 5;
    float r;
    if (warp == 0) {
        float p = (lane < nwarp) ? red[lane] : -CUDART_INF_F;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float other = __shfl_down_sync(full, p, o);
            p = fmaxf(p, other);
        }
        if (lane == 0) red[0] = p;
    }
    __syncthreads();
    r = red[0];
    __syncthreads();
    return r;
}

// ────────────────────────────────────────────────────────────────────────────
//  Primitive in-kernel ops, CTA-cooperative over one sample. Inputs/outputs are
//  smem [kSeq, width] tiles. Threads stride over the (position, feature) grid.
//  These mirror the oracle's primitive ops exactly.
// ────────────────────────────────────────────────────────────────────────────

// y[s, :out] = x[s, :in] @ W[out, in]^T + b[out].  W row-major [out, in].
// CTA-cooperative: one thread per (s, o) output element, strided.
__device__ __forceinline__ void dec_linear(
        const float* __restrict__ x, int in_dim,
        const float* __restrict__ W, const float* __restrict__ b, int out_dim,
        float* __restrict__ y) {
    const int total = dec::kSeq * out_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int s = idx / out_dim;
        const int o = idx % out_dim;
        const float* xr = x + s * in_dim;
        const float* Wr = W + (int64_t)o * in_dim;
        float acc = (b != nullptr) ? b[o] : 0.0f;
        #pragma unroll 4
        for (int k = 0; k < in_dim; ++k) acc += xr[k] * Wr[k];
        y[s * out_dim + o] = acc;
    }
    __syncthreads();
}

// LayerNorm forward over the last dim (width=d). Caches xhat + inv_std for bwd.
// gamma/beta are [d]. Per position: mean/var across d (cooperative reduce).
__device__ __forceinline__ void dec_layernorm_fwd(
        const float* __restrict__ x, const float* __restrict__ gamma,
        const float* __restrict__ beta, float* __restrict__ y,
        float* __restrict__ xhat_out, float* __restrict__ inv_out,
        float* red) {
    // Process positions sequentially (kSeq small); each is a full-CTA reduce.
    for (int s = 0; s < dec::kSeq; ++s) {
        const float* xr = x + s * dec::kD;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) sum += xr[j];
        float mean = dec_block_sum(sum, red) / (float)dec::kD;
        float vs = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float c = xr[j] - mean; vs += c * c;
        }
        float var = dec_block_sum(vs, red) / (float)dec::kD;
        float inv = rsqrtf(var + dec::kLnEps);
        if (threadIdx.x == 0) inv_out[s] = inv;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float xh = (xr[j] - mean) * inv;
            xhat_out[s * dec::kD + j] = xh;
            y[s * dec::kD + j] = xh * gamma[j] + beta[j];
        }
        __syncthreads();
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  FORWARD for one sample (CTA-cooperative). `tokens_s` is this sample's kSeq
//  token ids (smem). Writes the per-layer input cache + final-norm input +
//  last-position logits into `sm`, and returns the sample NLL (all threads see
//  the same value). Recompute-friendly: the layer-local scratch (qkv/ctx/x1/
//  ff0/gact/attn + LN caches) reflects the LAST layer processed; backward
//  recomputes per layer from layer_in.
// ────────────────────────────────────────────────────────────────────────────
__device__ float dec_forward_sample(const DecWeights& w, const int* tokens_s,
                                     int target, DecSampleSmem* sm) {
    // Embedding + positional: h0[s, j] = tok[tok_s][j] + pos[s][j].
    for (int idx = threadIdx.x; idx < dec::kSeq * dec::kD; idx += blockDim.x) {
        const int s = idx / dec::kD, j = idx % dec::kD;
        sm->layer_in[0][s][j] = w.tok[(int64_t)tokens_s[s] * dec::kD + j]
                              + w.pos[(int64_t)s * dec::kD + j];
    }
    __syncthreads();

    for (int li = 0; li < dec::kLayers; ++li) {
        const DecWeights::Layer& L = w.layer[li];
        float* hin = &sm->layer_in[li][0][0];   // [kSeq, kD]
        // qkv = hin @ in_w^T + in_b  -> [kSeq, 3d]
        dec_linear(hin, dec::kD, L.in_w, L.in_b, 3 * dec::kD, &sm->qkv[0][0]);
        // Attention per head: scores = q·k^T * scale (+ causal), softmax, ctx.
        // q,k,v live in qkv columns [0,d),[d,2d),[2d,3d). ctx -> sm->ctx [kSeq,d].
        const float scale = dec::attn_scale();
        // One thread per (head, qpos) computes that row's softmax + ctx slice.
        // kSeq*kHeads = 16 rows; threads stride. Each thread loops d_head.
        for (int r = threadIdx.x; r < dec::kHeads * dec::kSeq; r += blockDim.x) {
            const int hh = r / dec::kSeq;     // head
            const int qi = r % dec::kSeq;     // query position
            const int qoff = hh * dec::kDhead;        // head's slice in q block
            const float* qrow = &sm->qkv[qi][qoff];   // [dhead]
            // scores over key positions kj<=qi (causal); softmax; ctx accum.
            float maxs = -CUDART_INF_F;
            float sc[dec::kSeq];
            #pragma unroll
            for (int kj = 0; kj < dec::kSeq; ++kj) {
                if (kj > qi) { sc[kj] = -CUDART_INF_F; continue; }
                const float* krow = &sm->qkv[kj][dec::kD + qoff];  // k block
                float dot = 0.0f;
                #pragma unroll
                for (int t = 0; t < dec::kDhead; ++t) dot += qrow[t] * krow[t];
                sc[kj] = dot * scale;
                maxs = fmaxf(maxs, sc[kj]);
            }
            float denom = 0.0f;
            #pragma unroll
            for (int kj = 0; kj < dec::kSeq; ++kj) {
                float e = (kj <= qi) ? __expf(sc[kj] - maxs) : 0.0f;
                sc[kj] = e; denom += e;
            }
            float invd = 1.0f / denom;
            // store attn weights (for bwd) + compute ctx slice for this (head,qi).
            #pragma unroll
            for (int kj = 0; kj < dec::kSeq; ++kj) {
                float a = sc[kj] * invd;
                sm->attn[hh][qi][kj] = a;
            }
            // ctx[qi, hh*dhead + t] = sum_kj attn * v[kj, hh*dhead+t]
            #pragma unroll
            for (int t = 0; t < dec::kDhead; ++t) {
                float acc = 0.0f;
                #pragma unroll
                for (int kj = 0; kj <= qi; ++kj) {
                    const float vv = sm->qkv[kj][2 * dec::kD + qoff + t];
                    acc += sm->attn[hh][qi][kj] * vv;
                }
                sm->ctx[qi][qoff + t] = acc;
            }
        }
        __syncthreads();
        // a = ctx @ out_w^T + out_b  (reuse qkv[:, :d] as the 'a' scratch).
        // We need a [kSeq,d]; write into sm->x1 temporarily then add residual.
        dec_linear(&sm->ctx[0][0], dec::kD, L.out_w, L.out_b, dec::kD,
                   &sm->x1[0][0]);   // x1 holds 'a' for now
        // r1 = hin + a ; n1(r1) -> x1 (overwrite). Stage r1 into ff0 scratch row.
        for (int idx = threadIdx.x; idx < dec::kSeq * dec::kD; idx += blockDim.x) {
            const int s = idx / dec::kD, j = idx % dec::kD;
            sm->ctx[s][j] = hin[s * dec::kD + j] + sm->x1[s][j];  // ctx := r1
        }
        __syncthreads();
        dec_layernorm_fwd(&sm->ctx[0][0], L.n1_w, L.n1_b, &sm->x1[0][0],
                          &sm->n1_xhat[0][0], &sm->n1_inv[0], sm->red);
        // FFN: ff0 = x1 @ ff0_w^T + ff0_b ; g = gelu(ff0) ; ff2 = g @ ff2_w^T+b.
        dec_linear(&sm->x1[0][0], dec::kD, L.ff0_w, L.ff0_b, dec::kDff,
                   &sm->ff0[0][0]);
        for (int idx = threadIdx.x; idx < dec::kSeq * dec::kDff; idx += blockDim.x)
            sm->gact[idx / dec::kDff][idx % dec::kDff] =
                dec_gelu(sm->ff0[idx / dec::kDff][idx % dec::kDff]);
        __syncthreads();
        // ff2 -> ctx scratch (reuse), then r2 = x1 + ff2, n2(r2) -> next layer_in.
        dec_linear(&sm->gact[0][0], dec::kDff, L.ff2_w, L.ff2_b, dec::kD,
                   &sm->ctx[0][0]);   // ctx := ff2
        for (int idx = threadIdx.x; idx < dec::kSeq * dec::kD; idx += blockDim.x) {
            const int s = idx / dec::kD, j = idx % dec::kD;
            sm->ctx[s][j] = sm->x1[s][j] + sm->ctx[s][j];   // ctx := r2
        }
        __syncthreads();
        // n2(r2) -> output of this layer. Destination: next layer's input cache,
        // or the final_in buffer if this is the last layer.
        float* dst = (li + 1 < dec::kLayers) ? &sm->layer_in[li + 1][0][0]
                                             : &sm->final_in[0][0];
        dec_layernorm_fwd(&sm->ctx[0][0], L.n2_w, L.n2_b, dst,
                          &sm->n2_xhat[0][0], &sm->n2_inv[0], sm->red);
    }

    // Final norm on the LAST position only, then head → logits → NLL.
    // hn = norm(final_in[last]); logits = hn @ out_w^T + out_b.
    const float* hlast = &sm->final_in[dec::kSeq - 1][0];
    // LayerNorm of a single row (cooperative reduce).
    float sum = 0.0f;
    for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) sum += hlast[j];
    float mean = dec_block_sum(sum, sm->red) / (float)dec::kD;
    float vs = 0.0f;
    for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
        float c = hlast[j] - mean; vs += c * c;
    }
    float var = dec_block_sum(vs, sm->red) / (float)dec::kD;
    float inv = rsqrtf(var + dec::kLnEps);
    if (threadIdx.x == 0) sm->fn_inv[0] = inv;
    // Store hn into fn_xhat row 0's gamma/beta-applied value? We need both xhat
    // (for LN bwd) and hn (for head). Keep xhat in fn_xhat[last], compute hn into
    // x1 row 0 scratch.
    for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
        float xh = (hlast[j] - mean) * inv;
        sm->fn_xhat[dec::kSeq - 1][j] = xh;
        sm->x1[0][j] = xh * w.norm_w[j] + w.norm_b[j];   // x1[0] := hn
    }
    __syncthreads();
    // logits[o] = hn · out_w[o] + out_b[o]   (o over kVocab).
    for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) {
        const float* Wr = w.out_w + (int64_t)o * dec::kD;
        float acc = w.out_b[o];
        #pragma unroll 4
        for (int k = 0; k < dec::kD; ++k) acc += sm->x1[0][k] * Wr[k];
        sm->logits[o] = acc;
    }
    __syncthreads();
    // NLL (mean handled by /B in the loss reduce): logz - logits[target].
    float lmax = -CUDART_INF_F;
    for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x)
        lmax = fmaxf(lmax, sm->logits[o]);
    lmax = dec_block_max(lmax, sm->red);
    float es = 0.0f;
    for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x)
        es += __expf(sm->logits[o] - lmax);
    es = dec_block_sum(es, sm->red);
    float logz = lmax + __logf(es);
    return logz - sm->logits[target];   // sample NLL
}

// ── Atomic-free per-CTA grad accumulation. Each dW element has a SINGLE owner
//    thread within the CTA across the slice's samples, so a plain += into the
//    CTA's partial buffer is race-free AS LONG AS the same thread owns the same
//    element every sample. We guarantee that by always striding output elements
//    by threadIdx.x with the SAME total/stride, and __syncthreads() between
//    samples. acc_* helpers therefore use ordinary global +=. ─────────────────

// Recompute one layer's forward intermediates into sm (qkv, x1, ff0, gact, attn,
// and ctx := r2), reading from sm->layer_in[li]. Mirrors the fwd layer body but
// also leaves r2 in sm->ctx and x1 in sm->x1 for the backward chain.
__device__ void dec_recompute_layer(const DecWeights& w, int li,
                                     DecSampleSmem* sm) {
    const DecWeights::Layer& L = w.layer[li];
    float* hin = &sm->layer_in[li][0][0];
    dec_linear(hin, dec::kD, L.in_w, L.in_b, 3 * dec::kD, &sm->qkv[0][0]);
    const float scale = dec::attn_scale();
    for (int r = threadIdx.x; r < dec::kHeads * dec::kSeq; r += blockDim.x) {
        const int hh = r / dec::kSeq, qi = r % dec::kSeq;
        const int qoff = hh * dec::kDhead;
        const float* qrow = &sm->qkv[qi][qoff];
        float maxs = -CUDART_INF_F; float sc[dec::kSeq];
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            if (kj > qi) { sc[kj] = -CUDART_INF_F; continue; }
            const float* krow = &sm->qkv[kj][dec::kD + qoff];
            float dot = 0.0f;
            #pragma unroll
            for (int t = 0; t < dec::kDhead; ++t) dot += qrow[t] * krow[t];
            sc[kj] = dot * scale; maxs = fmaxf(maxs, sc[kj]);
        }
        float denom = 0.0f;
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            float e = (kj <= qi) ? __expf(sc[kj] - maxs) : 0.0f;
            sc[kj] = e; denom += e;
        }
        float invd = 1.0f / denom;
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) sm->attn[hh][qi][kj] = sc[kj] * invd;
        #pragma unroll
        for (int t = 0; t < dec::kDhead; ++t) {
            float acc = 0.0f;
            #pragma unroll
            for (int kj = 0; kj <= qi; ++kj)
                acc += sm->attn[hh][qi][kj] * sm->qkv[kj][2 * dec::kD + qoff + t];
            sm->ctx[qi][qoff + t] = acc;
        }
    }
    __syncthreads();
    // a -> x1 scratch, r1 -> separate; we need r1 for n1 bwd? n1 bwd needs xhat
    // (cached) + inv (cached); it does NOT need r1. So compute n1 fwd to fill
    // caches and leave x1 = n1(r1).
    dec_linear(&sm->ctx[0][0], dec::kD, L.out_w, L.out_b, dec::kD, &sm->x1[0][0]);
    for (int idx = threadIdx.x; idx < dec::kSeq * dec::kD; idx += blockDim.x) {
        const int s = idx / dec::kD, j = idx % dec::kD;
        sm->ctx[s][j] = hin[s * dec::kD + j] + sm->x1[s][j];   // ctx := r1
    }
    __syncthreads();
    dec_layernorm_fwd(&sm->ctx[0][0], L.n1_w, L.n1_b, &sm->x1[0][0],
                      &sm->n1_xhat[0][0], &sm->n1_inv[0], sm->red);  // x1 := n1(r1)
    dec_linear(&sm->x1[0][0], dec::kD, L.ff0_w, L.ff0_b, dec::kDff, &sm->ff0[0][0]);
    for (int idx = threadIdx.x; idx < dec::kSeq * dec::kDff; idx += blockDim.x) {
        const int s = idx / dec::kDff, j = idx % dec::kDff;
        sm->gact[s][j] = dec_gelu(sm->ff0[s][j]);
    }
    __syncthreads();
    // ff2 -> tmp, r2 = x1 + ff2 -> ctx (so ctx holds r2 for n2 bwd caches).
    // We use sm->ctx for r2; n2 caches (xhat/inv) are filled by the LN fwd below.
    // Stash ff2 in dh scratch (unused until bwd) to avoid clobbering x1.
    dec_linear(&sm->gact[0][0], dec::kDff, L.ff2_w, L.ff2_b, dec::kD, &sm->dh[0][0]);
    for (int idx = threadIdx.x; idx < dec::kSeq * dec::kD; idx += blockDim.x) {
        const int s = idx / dec::kD, j = idx % dec::kD;
        sm->ctx[s][j] = sm->x1[s][j] + sm->dh[s][j];   // ctx := r2
    }
    __syncthreads();
    // Fill n2 caches (we don't need n2's output value for bwd, only xhat/inv).
    // Write the (discarded) output into dh scratch.
    dec_layernorm_fwd(&sm->ctx[0][0], L.n2_w, L.n2_b, &sm->dh[0][0],
                      &sm->n2_xhat[0][0], &sm->n2_inv[0], sm->red);
    __syncthreads();
}

// LayerNorm backward: given dy [kSeq,d] and cached (xhat,inv) for these
// positions, returns dx [kSeq,d] into dx_out, and ACCUMULATES dgamma/dbeta
// (summed over positions) into the CTA grad partials gw/gb [d]. Owner-thread
// rule for gw/gb: thread owns feature j (strided), sums over positions locally.
__device__ void dec_layernorm_bwd(
        const float* __restrict__ dy, const float* __restrict__ xhat,
        const float* __restrict__ inv, const float* __restrict__ gamma,
        float* __restrict__ dx_out, float* __restrict__ gw, float* __restrict__ gb,
        float* red, bool accum_affine) {
    // dgamma/dbeta: feature-parallel, sum over positions (single owner per j).
    if (accum_affine) {
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dgw = 0.0f, dgb = 0.0f;
            #pragma unroll
            for (int s = 0; s < dec::kSeq; ++s) {
                float d = dy[s * dec::kD + j];
                dgb += d; dgw += d * xhat[s * dec::kD + j];
            }
            gw[j] += dgw; gb[j] += dgb;
        }
    }
    // dx per position: needs sum(dxhat) and sum(dxhat*xhat) over d (reduce).
    for (int s = 0; s < dec::kSeq; ++s) {
        const float* dyr = dy + s * dec::kD;
        const float* xhr = xhat + s * dec::kD;
        float sda = 0.0f, sdax = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dxhat = dyr[j] * gamma[j];
            sda += dxhat; sdax += dxhat * xhr[j];
        }
        sda = dec_block_sum(sda, red);
        sdax = dec_block_sum(sdax, red);
        float inv_s = inv[s];
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dxhat = dyr[j] * gamma[j];
            dx_out[s * dec::kD + j] =
                inv_s * (dxhat - (sda + xhr[j] * sdax) / (float)dec::kD);
        }
        __syncthreads();
    }
}

// dW += dY^T @ X  for a linear Y = X @ W^T + b. dY [kSeq,out], X [kSeq,in],
// dW [out,in]. Owner-thread rule: thread owns (o,i) output element (strided over
// out*in), sums over the kSeq positions locally. db[o] += sum_s dY[s,o].
// Also returns dX [kSeq,in] += dY @ W into dx_out (caller decides add vs set).
//
// ldY/ldX/ldDX are the ROW STRIDES (leading dimensions) of dY/X/dx_out. They
// are EXPLICIT (no defaults) because the smem scratch buffers this is called on
// are 2D arrays whose declared width (e.g. kDff=512 for ff0/gact) differs from
// the logical row width (out_dim/in_dim). The original packed-stride version
// silently read/wrote the wrong rows at both attention-seam call sites — the
// forward stayed exact while every grad upstream of attention was garbage
// (params decayed to zero via weight decay; train acc pinned at 1/97). The CPU
// structural mirror did NOT catch it (it indexes buffers semantically, not by
// flat offset), so the stride is now part of the call contract.
__device__ void dec_linear_bwd(
        const float* __restrict__ dY, const float* __restrict__ X,
        const float* __restrict__ W, int in_dim, int out_dim,
        float* __restrict__ dW, float* __restrict__ db,
        float* __restrict__ dx_out, bool set_dx,
        int ldY, int ldX, int ldDX) {
    // dW [out,in] + db [out].
    const int wtot = out_dim * in_dim;
    for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
        const int o = idx / in_dim, i = idx % in_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < dec::kSeq; ++s)
            acc += dY[s * ldY + o] * X[s * ldX + i];
        dW[(int64_t)o * in_dim + i] += acc;
    }
    if (db != nullptr) {
        for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
            float acc = 0.0f;
            #pragma unroll
            for (int s = 0; s < dec::kSeq; ++s) acc += dY[s * ldY + o];
            db[o] += acc;
        }
    }
    // dX [kSeq,in] = dY @ W  (W [out,in]).
    const int xtot = dec::kSeq * in_dim;
    for (int idx = threadIdx.x; idx < xtot; idx += blockDim.x) {
        const int s = idx / in_dim, i = idx % in_dim;
        float acc = 0.0f;
        for (int o = 0; o < out_dim; ++o)
            acc += dY[s * ldY + o] * W[(int64_t)o * in_dim + i];
        if (set_dx) dx_out[s * ldDX + i] = acc;
        else dx_out[s * ldDX + i] += acc;
    }
    __syncthreads();
}

// ────────────────────────────────────────────────────────────────────────────
//  BACKWARD for one sample (CTA-cooperative). Assumes dec_forward_sample has run
//  for THIS sample (so layer_in/final_in/fn caches/logits are populated and
//  reflect this sample). Accumulates every weight grad into the CTA's partial
//  buffer `g` (DecGrad over gw_cta). Uses extra smem scratch passed via sm.
//
//  Transcribed from tests/hw/decoder_oracle.py::decoder_backward.
// ────────────────────────────────────────────────────────────────────────────
__device__ void dec_backward_sample(const DecWeights& w, const DecGrad& g,
                                     const int* tokens_s, int target, int B,
                                     DecSampleSmem* sm) {
    // ── CE backward: dlogits = (softmax - onehot)/B. (logits cached in fwd.) ──
    float lmax = -CUDART_INF_F;
    for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x)
        lmax = fmaxf(lmax, sm->logits[o]);
    lmax = dec_block_max(lmax, sm->red);
    float es = 0.0f;
    for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x)
        es += __expf(sm->logits[o] - lmax);
    es = dec_block_sum(es, sm->red);
    float inv_es = 1.0f / es;
    // overwrite logits[] with dlogits in place.
    for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) {
        float sm_o = __expf(sm->logits[o] - lmax) * inv_es;
        float dl = (sm_o - ((o == target) ? 1.0f : 0.0f)) / (float)B;
        sm->logits[o] = dl;
    }
    __syncthreads();
    // ── head (out): logits = hn @ out_w^T + out_b. hn was stashed in x1[0]
    //    during fwd, but x1 has since been reused — recompute hn from the
    //    cached final-norm: hn[j] = fn_xhat[last][j]*norm_w[j] + norm_b[j]. ──
    // dW_out[o,:] += dlogits[o] * hn ; db_out[o] += dlogits[o].
    // dhn[j] = sum_o dlogits[o] * out_w[o,j].
    for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) {
        float dl = sm->logits[o];
        g.out_b[o] += dl;
        float* gwrow = g.out_w + (int64_t)o * dec::kD;
        const float* xh = &sm->fn_xhat[dec::kSeq - 1][0];
        #pragma unroll 4
        for (int j = 0; j < dec::kD; ++j) {
            float hn_j = xh[j] * w.norm_w[j] + w.norm_b[j];
            gwrow[j] += dl * hn_j;
        }
    }
    __syncthreads();
    // dhn[j] (feature-parallel) into dh row 0 scratch.
    for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
        float acc = 0.0f;
        for (int o = 0; o < dec::kVocab; ++o)
            acc += sm->logits[o] * w.out_w[(int64_t)o * dec::kD + j];
        sm->dh[0][j] = acc;   // dhn
    }
    __syncthreads();
    // ── final norm backward: dy=dhn (row last), xhat=fn_xhat[last], inv=fn_inv.
    //    Produces dh_last into dh[last]; accumulate dnorm_w/dnorm_b. Use a single
    //    "position" (last). Build a 1-row dy view: reuse sm->ctx row last = dhn. ─
    for (int j = threadIdx.x; j < dec::kD; j += blockDim.x)
        sm->ctx[dec::kSeq - 1][j] = sm->dh[0][j];   // dy at last position
    __syncthreads();
    // dnorm_w/dnorm_b: only the last position contributes (others have dy=0).
    for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
        float d = sm->ctx[dec::kSeq - 1][j];
        float xh = sm->fn_xhat[dec::kSeq - 1][j];
        g.norm_b[j] += d; g.norm_w[j] += d * xh;
    }
    __syncthreads();
    // dx for the last position (LN backward, single row).
    {
        const float* dyr = &sm->ctx[dec::kSeq - 1][0];
        const float* xhr = &sm->fn_xhat[dec::kSeq - 1][0];
        float sda = 0.0f, sdax = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dxhat = dyr[j] * w.norm_w[j];
            sda += dxhat; sdax += dxhat * xhr[j];
        }
        sda = dec_block_sum(sda, sm->red);
        sdax = dec_block_sum(sdax, sm->red);
        float inv_s = sm->fn_inv[0];
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dxhat = dyr[j] * w.norm_w[j];
            // dh[s,:] = 0 for s<last; last gets the LN dx.
            for (int s = 0; s < dec::kSeq; ++s) sm->dh[s][j] = 0.0f;
            sm->dh[dec::kSeq - 1][j] =
                inv_s * (dxhat - (sda + xhr[j] * sdax) / (float)dec::kD);
        }
        __syncthreads();
    }
    // dh now = grad wrt the last layer's OUTPUT [kSeq,d] (only last pos nonzero).

    // ── per-layer backward (reverse). Each layer: recompute fwd intermediates,
    //    then apply the chain. We use sm scratch heavily; the running adjoint
    //    `dh` is preserved across the recompute (recompute writes qkv/x1/ff0/
    //    gact/attn/ctx + LN caches; it temporarily uses sm->dh as ff2 scratch —
    //    so SAVE dh into final_in (free after fwd) around the recompute). ───────
    for (int li = dec::kLayers - 1; li >= 0; --li) {
        const DecWeights::Layer& L = w.layer[li];
        const DecGrad::Layer& G = g.layer[li];
        // Save the incoming adjoint dh into final_in (unused now) — recompute
        // clobbers sm->dh.
        for (int idx = threadIdx.x; idx < dec::kSeq * dec::kD; idx += blockDim.x) {
            const int s = idx / dec::kD, j = idx % dec::kD;
            sm->final_in[s][j] = sm->dh[s][j];
        }
        __syncthreads();
        dec_recompute_layer(w, li, sm);   // fills qkv,x1,ff0,gact,attn,ctx(=r2),caches
        // Restore dh (grad wrt h2 = n2 output).
        for (int idx = threadIdx.x; idx < dec::kSeq * dec::kD; idx += blockDim.x) {
            const int s = idx / dec::kD, j = idx % dec::kD;
            sm->dh[s][j] = sm->final_in[s][j];
        }
        __syncthreads();
        // n2 backward: dh -> dr2, accumulate n2 gamma/beta. Put dr2 into ctx
        // (currently r2 — we no longer need r2's value, only its caches).
        dec_layernorm_bwd(&sm->dh[0][0], &sm->n2_xhat[0][0], &sm->n2_inv[0],
                          L.n2_w, &sm->ctx[0][0], G.n2_w, G.n2_b, sm->red, true);
        // r2 = x1 + ff2 → both branches get dr2: dx1 (residual) and dff2.
        // SMEM map (one layer live; values consumed before each reuse — verified
        // alias-safe by the structural mirror): dx1 lives in `fn_xhat` (final-norm
        // caches are done), dr2 stays in `ctx`. x_in = layer_in[li] is NOT touched
        // (in_proj bwd needs it); qkv is NOT touched (attention bwd needs q/k/v).
        for (int idx = threadIdx.x; idx < dec::kSeq * dec::kD; idx += blockDim.x) {
            const int s = idx / dec::kD, j = idx % dec::kD;
            sm->fn_xhat[s][j] = sm->ctx[s][j];   // dx1 := dr2 (residual part)
        }
        __syncthreads();
        // ff.2 backward. dgact = dr2 @ ff2_w is needed only to form
        // dff0 = dgact * gelu'(ff0); since `ff0` (pre-gelu) is needed for gelu'(),
        // we compute dgact and immediately fold gelu' so dff0 is written OVER
        // `gact` (its forward value g is consumed by the dW_ff2 GEMM first). No
        // extra scratch buffer is needed.
        //   dW_ff2 += dr2^T @ g ; db_ff2 += Σ dr2.
        {
            const float* dY = &sm->ctx[0][0];   // dr2 [kSeq,d]
            const float* X = &sm->gact[0][0];   // g [kSeq,4d]
            // dW ff2 [d,4d]
            const int wtot = dec::kD * dec::kDff;
            for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
                const int o = idx / dec::kDff, i = idx % dec::kDff;
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < dec::kSeq; ++s)
                    acc += dY[s * dec::kD + o] * X[s * dec::kDff + i];
                G.ff2_w[(int64_t)o * dec::kDff + i] += acc;
            }
            for (int o = threadIdx.x; o < dec::kD; o += blockDim.x) {
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < dec::kSeq; ++s) acc += dY[s * dec::kD + o];
                G.ff2_b[o] += acc;
            }
            __syncthreads();
            // dff0 = (dr2 @ ff2_w) * gelu'(ff0), written OVER gact.
            const int xtot = dec::kSeq * dec::kDff;
            for (int idx = threadIdx.x; idx < xtot; idx += blockDim.x) {
                const int s = idx / dec::kDff, i = idx % dec::kDff;
                float dg = 0.0f;
                for (int o = 0; o < dec::kD; ++o)
                    dg += dY[s * dec::kD + o] * L.ff2_w[(int64_t)o * dec::kDff + i];
                sm->gact[s][i] = dg * dec_gelu_grad(sm->ff0[s][i]);  // gact := dff0
            }
            __syncthreads();
        }
        // ff.0 backward: dY=dff0 (=gact), X=x1 -> dW ff0, db ff0, dx1 += dff0 @ ff0_w.
        // dx1 currently in fn_xhat (residual part); ADD the FFN path.
        {
            const float* dY = &sm->gact[0][0];   // dff0 [kSeq,4d]
            const float* X = &sm->x1[0][0];      // x1 [kSeq,d]
            const int wtot = dec::kDff * dec::kD;
            for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
                const int o = idx / dec::kD, i = idx % dec::kD;
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < dec::kSeq; ++s)
                    acc += dY[s * dec::kDff + o] * X[s * dec::kD + i];
                G.ff0_w[(int64_t)o * dec::kD + i] += acc;
            }
            for (int o = threadIdx.x; o < dec::kDff; o += blockDim.x) {
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < dec::kSeq; ++s) acc += dY[s * dec::kDff + o];
                G.ff0_b[o] += acc;
            }
            __syncthreads();
            const int xtot = dec::kSeq * dec::kD;
            for (int idx = threadIdx.x; idx < xtot; idx += blockDim.x) {
                const int s = idx / dec::kD, i = idx % dec::kD;
                float acc = 0.0f;
                for (int o = 0; o < dec::kDff; ++o)
                    acc += dY[s * dec::kDff + o] * L.ff0_w[(int64_t)o * dec::kD + i];
                sm->fn_xhat[s][i] += acc;   // dx1 += FFN path
            }
            __syncthreads();
        }
        // dx1 (in fn_xhat) -> n1 backward -> dr1 (into ctx). accumulate n1 g/b.
        dec_layernorm_bwd(&sm->fn_xhat[0][0], &sm->n1_xhat[0][0], &sm->n1_inv[0],
                          L.n1_w, &sm->ctx[0][0], G.n1_w, G.n1_b, sm->red, true);
        // r1 = x_in + a -> da = dr1 ; dx_in = dr1 (residual). dx_in accumulates
        // both the residual AND the attention-input path; stash dx_in into
        // fn_xhat (free again). da stays in ctx.
        for (int idx = threadIdx.x; idx < dec::kSeq * dec::kD; idx += blockDim.x) {
            const int s = idx / dec::kD, j = idx % dec::kD;
            sm->fn_xhat[s][j] = sm->ctx[s][j];   // dx_in := dr1 (residual)
        }
        __syncthreads();
        // attn out_proj backward: a = ctx_fwd @ out_w^T + out_b. But ctx_fwd
        // (the attention context) was overwritten by the recompute's r1/r2 reuse.
        // Recompute ctx_fwd from attn + v (cheap): we still have attn[] and qkv[]
        // (v lives in qkv[:,2d:3d]). Rebuild ctx_fwd into x1 scratch (x1 no longer
        // needed — ff bwd done).
        for (int r = threadIdx.x; r < dec::kSeq * dec::kD; r += blockDim.x) {
            const int qi = r / dec::kD, col = r % dec::kD;
            const int hh = col / dec::kDhead, t = col % dec::kDhead;
            float acc = 0.0f;
            #pragma unroll
            for (int kj = 0; kj <= qi; ++kj)
                acc += sm->attn[hh][qi][kj]
                     * sm->qkv[kj][2 * dec::kD + hh * dec::kDhead + t];
            sm->x1[qi][col] = acc;   // x1 := ctx_fwd
        }
        __syncthreads();
        // da is in ctx. dW_out += da^T @ ctx_fwd(x1); db_out += sum da;
        // dctx = da @ out_w -> store into ff0 scratch (free). dctx MUST be
        // written at ff0's DECLARED row stride (kDff) — passes A/B below read it
        // as sm->ff0[qi][col]; a packed (kD-stride) write only fills row 0's
        // first 512 floats and rows 1..3 read stale recomputed pre-GELU values.
        dec_linear_bwd(&sm->ctx[0][0], &sm->x1[0][0], L.out_w, dec::kD, dec::kD,
                       G.out_w, G.out_b, &sm->ff0[0][0], /*set_dx=*/true, // dctx->ff0
                       /*ldY=*/dec::kD, /*ldX=*/dec::kD, /*ldDX=*/dec::kDff);
        // ── merge-heads + attention backward (3 synced passes; the forward
        //    softmax weights `attn` are read by A and B and NEVER overwritten —
        //    dscores go to the dedicated `dsc` buffer). dctx [kSeq,d] is in ff0;
        //    v lives in qkv[:,2d:3d], k in qkv[:,d:2d], q in qkv[:,0:d]. dqkv is
        //    built into gact[:, :3d]. Transcribed from the oracle's attention bwd:
        //      dvh   = attn^T @ dctxh
        //      dattn = dctxh @ vh^T ; dscores = softmax_bwd(dattn,attn) * scale
        //      dqh   = dscores @ kh ; dkh = dscores^T @ qh                       ─
        const float scale = dec::attn_scale();
        // A: dv into gact[:, 2d:3d].  dv[kj,qoff+t] = Σ_{qi>=kj} attn[hh][qi][kj]
        //    * dctx[qi, qoff+t].  (Owner: thread per (kj, head, t).)
        for (int r = threadIdx.x; r < dec::kSeq * dec::kHeads * dec::kDhead;
             r += blockDim.x) {
            const int kj = r / (dec::kHeads * dec::kDhead);
            const int rem = r % (dec::kHeads * dec::kDhead);
            const int hh = rem / dec::kDhead, t = rem % dec::kDhead;
            const int qoff = hh * dec::kDhead;
            float acc = 0.0f;
            #pragma unroll
            for (int qi = kj; qi < dec::kSeq; ++qi)   // qi>=kj (causal: kj<=qi)
                acc += sm->attn[hh][qi][kj] * sm->ff0[qi][qoff + t];
            sm->gact[kj][2 * dec::kD + qoff + t] = acc;   // dv block
        }
        __syncthreads();
        // B: dscores into dsc.  datt[hh,qi,kj] = Σ_t dctx[qi,qoff+t]*v[kj,qoff+t];
        //    ds = attn*(datt - Σ_k datt*attn)*scale  (masked kj>qi → 0).
        for (int r = threadIdx.x; r < dec::kHeads * dec::kSeq; r += blockDim.x) {
            const int hh = r / dec::kSeq, qi = r % dec::kSeq;
            const int qoff = hh * dec::kDhead;
            float datt[dec::kSeq];
            #pragma unroll
            for (int kj = 0; kj < dec::kSeq; ++kj) {
                if (kj > qi) { datt[kj] = 0.0f; continue; }
                float acc = 0.0f;
                #pragma unroll
                for (int t = 0; t < dec::kDhead; ++t)
                    acc += sm->ff0[qi][qoff + t]
                         * sm->qkv[kj][2 * dec::kD + qoff + t];
                datt[kj] = acc;
            }
            float dot = 0.0f;
            #pragma unroll
            for (int kj = 0; kj <= qi; ++kj) dot += datt[kj] * sm->attn[hh][qi][kj];
            #pragma unroll
            for (int kj = 0; kj < dec::kSeq; ++kj) {
                float a = sm->attn[hh][qi][kj];
                sm->dsc[hh][qi][kj] = (kj <= qi) ? a * (datt[kj] - dot) * scale
                                                 : 0.0f;
            }
        }
        __syncthreads();
        // C: dq into gact[:,0:d], dk into gact[:,d:2d].
        //    dq[qi,qoff+t] = Σ_kj dsc[hh][qi][kj]*k[kj,qoff+t];
        //    dk[kj,qoff+t] = Σ_qi dsc[hh][qi][kj]*q[qi,qoff+t].  (Owner: pos,head,t.)
        for (int r = threadIdx.x; r < dec::kSeq * dec::kHeads * dec::kDhead;
             r += blockDim.x) {
            const int pos = r / (dec::kHeads * dec::kDhead);
            const int rem = r % (dec::kHeads * dec::kDhead);
            const int hh = rem / dec::kDhead, t = rem % dec::kDhead;
            const int qoff = hh * dec::kDhead;
            float dq = 0.0f, dk = 0.0f;
            #pragma unroll
            for (int kj = 0; kj < dec::kSeq; ++kj) {
                dq += sm->dsc[hh][pos][kj] * sm->qkv[kj][dec::kD + qoff + t];
                dk += sm->dsc[hh][kj][pos] * sm->qkv[kj][qoff + t];
            }
            sm->gact[pos][qoff + t] = dq;            // dq block (gact[:,0:d])
            sm->gact[pos][dec::kD + qoff + t] = dk;  // dk block (gact[:,d:2d])
        }
        __syncthreads();
        // in_proj backward: dqkv (gact[:, :3d]) -> dW in, db in, dx_in += dqkv @ in_w.
        // x_in = layer_in[li]. dqkv was written 2D (sm->gact[pos][col]) so dY's
        // row stride is gact's DECLARED width kDff — a packed (3d=384) read
        // walks off row 0 into the stale dff0 values left in gact[0][384:512].
        dec_linear_bwd(&sm->gact[0][0], &sm->layer_in[li][0][0], L.in_w,
                       dec::kD, 3 * dec::kD, G.in_w, G.in_b,
                       &sm->fn_xhat[0][0], /*set_dx=*/false,   // dx_in += attn path
                       /*ldY=*/dec::kDff, /*ldX=*/dec::kD, /*ldDX=*/dec::kD);
        // dh for the previous layer = dx_in (in fn_xhat). Move to dh.
        for (int idx = threadIdx.x; idx < dec::kSeq * dec::kD; idx += blockDim.x) {
            const int s = idx / dec::kD, j = idx % dec::kD;
            sm->dh[s][j] = sm->fn_xhat[s][j];
        }
        __syncthreads();
    }

    // ── embedding backward: dh = grad wrt h0 [kSeq,d]. tok scatter-add per token
    //    id; pos sum over the sample's positions. Owner-thread rule: thread owns
    //    column j; loops positions SEQUENTIALLY so colliding token ids accumulate
    //    deterministically (mirrors the oracle's index_add_). ──────────────────
    for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
        #pragma unroll
        for (int s = 0; s < dec::kSeq; ++s) {
            float d = sm->dh[s][j];
            g.tok[(int64_t)tokens_s[s] * dec::kD + j] += d;   // single owner: col j
            g.pos[(int64_t)s * dec::kD + j] += d;
        }
    }
    __syncthreads();
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGES_DECODER_CUH_
