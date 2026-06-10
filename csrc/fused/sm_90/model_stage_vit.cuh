#ifndef SG_FUSED_SM90_MODEL_STAGE_VIT_CUH_
#define SG_FUSED_SM90_MODEL_STAGE_VIT_CUH_
// ============================================================================
// csrc/fused/sm_90/model_stage_vit.cuh — PHASE 2 of the TRUE L3 fused
// megakernel: the REAL Vision-Transformer forward + backward as in-kernel stages
// of the persistent megakernel (the ViT counterpart of PHASE 1's decoder
// model_stages_decoder.cuh). ONE self-contained header per the portability
// contract (csrc/fused/COMPONENT_CONTRACT.md): provides vit forward/backward
// stages, includes ONLY the substrate header + CUDA, no globals, no repo build
// flags.
//
// HONESTY: this is the genuine eager architecture (grokking_race_v2.py ViT,
// lines 379-404), transcribed LINE-FOR-LINE from the verified PyTorch oracle in
// tests/hw/vit_oracle.py, which is asserted bit-identical (fp64, ~1e-15 rel) to
// torch.autograd for the loss and EVERY one of the 32 parameter gradients, and
// structurally mirrored on CPU in tests/hw/vit_kernel_mirror.py. There is NO
// placeholder math on this path. The one deliberate, documented deviation from
// the oracle is execution mechanics, not numerics: the backward RECOMPUTES each
// layer's forward from the cached layer INPUT (rather than caching every
// intermediate) to keep only one layer's activations live in smem at a time —
// see the smem-budget note below. The math computed is identical; only the
// storage strategy differs.
//
// ARCHITECTURE (exact; cites grokking_race_v2.py lines):
//   ViT(p=97, patch_dim=49, num_patches=16, d=128, h=4, nl=2)
//     patch_proj = Linear(49,128) (:392)  — per-patch embed (16 patches)
//     cls_token  = Parameter[1,1,128]*.02 (:393) — prepended at position 0
//     pos        = Embedding(16+1,128) (:394) — 17 learned position embeds
//     2 × EncoderBlock (:379):
//       attn = MultiheadAttention(128,4,batch_first) (:382), NO MASK — FULL
//              (bidirectional) attention (contrast: decoder is causal).
//       n1,n2 = LayerNorm(128) (:383), eps=1e-5 (torch default)
//       ff = Linear(128,512) → GELU(exact erf) → Linear(512,128) (:384)
//       POST-LN: x=n1(x+attn(x)); h=n2(x+ff(x))  (:386-387)
//     norm = LayerNorm(128) (:396); out = Linear(128,97) (:396)
//     forward: h=patch_proj(x); h=cat([cls,h]); h+=pos; for l: h=l(h);
//              return out(norm(h[:,0,:]))  (:400-404)  — CLS token (pos 0).
//   loss = cross_entropy(logits, targets) (mean over batch; :967…).
//   INPUT is FLOAT image patches [B,16,49] (NOT int tokens); targets int [B].
//
// NUMERICS (verified in the oracle; see vit_oracle.py):
//   * GELU = exact erf  0.5x(1+erf(x/√2)); grad 0.5(1+erf(x/√2))+x·φ(x),
//     φ(x)=(1/√(2π))e^{-x²/2}.
//   * LayerNorm eps = 1e-5; dx uses the two mean-correction terms.
//   * attention scale 1/√dh = 1/√32; softmax subtracts the row max; FULL (no
//     causal triangle — every key position contributes for every query).
//   * cross-entropy MEAN over B → dlogits = (softmax−onehot)/B.
//
// PARALLELIZATION (locked design — Option B, batch-parallel, deterministic;
//   identical to the decoder's, see model_stages_decoder.cuh):
//   * Each CTA owns a FIXED contiguous batch slice (by blockIdx.x). The whole
//     CTA cooperates on ONE sample at a time and iterates its slice SEQUENTIALLY.
//   * Per-sample, the CTA accumulates that sample's weight-grad contribution into
//     the CTA's OWN partial buffer (gw + cta*total). Each dW element has a SINGLE
//     owner thread that does partial[e] += contrib across the slice's samples in
//     fixed order (__syncthreads between samples) — single-writer, no atomics,
//     deterministic.
//   * The cross-CTA reduce + optimizer tail live in fused_vit_megakernel.cuh
//     (deterministic ascending-CTA summation order).
//
// SMEM BUDGET PER CTA — DYNAMIC SHARED MEMORY (one sample, one layer live):
//   At seq=17 the per-sample working set is ≈ sizeof(VitSampleSmem) ≈ 180 KB —
//   FAR over the 48 KB STATIC __shared__ cap (a static `__shared__ VitSampleSmem`
//   would NOT COMPILE). So ViT uses EXTERN/DYNAMIC shared memory: the launcher
//   calls cudaFuncSetAttribute(MaxDynamicSharedMemorySize, sizeof(VitSampleSmem))
//   and passes dynamicSMemBytes=sizeof(VitSampleSmem) at <<<>>>. 180 KB ≪ the
//   sm_90 per-block dynamic-smem cap (227 KB), and the persistent megakernel is
//   one-CTA-per-SM BY DESIGN (gridDim=#SMs, grid barrier over #SMs), so occupancy
//   =1 IS the design point — a 180 KB block at occ=1 is exactly intended, NOT a
//   regression. The dominant buffers are ff0 (pre-GELU) + gact (post-GELU), 34 KB
//   each; both kept live within the processed layer (ff0 for gelu'(), gact for
//   the dW_ff2 GEMM). The exact byte count is static_asserted in vit_layout.cuh.
//   (TODO[bf16]: a bf16-activation variant would roughly halve this and is a
//   flagged follow-up defaulting to THIS fp32 path — consistent with the decoder.)
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/vit_layout.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>

namespace sg { namespace fused { namespace sm90 {

// ── ViT compile-time shape constants (mirror tests/hw/vit_oracle.py and
//    csrc/fused/sm_90/vit_layout.cuh; static_asserts there guard the count +
//    total against named_parameters()). ────────────────────────────────────────
namespace vit {
constexpr int kVocab   = SG_VIT_VOCAB;   // 97  (== p; head Linear(d, p))
constexpr int kD       = SG_VIT_D;       // 128
constexpr int kHeads   = SG_VIT_HEADS;   // 4
constexpr int kLayers  = SG_VIT_LAYERS;  // 2
constexpr int kPatch   = SG_VIT_PATCH;   // 49 (patch pixel count, 7×7)
constexpr int kNPatch  = SG_VIT_NPATCH;  // 16 image patches
constexpr int kSeq     = kNPatch + 1;    // 17 (CLS at 0, then 16 patches)
constexpr int kDff     = SG_VIT_DFF;     // 512
constexpr int kDhead   = kD / kHeads;    // 32
constexpr float kLnEps = 1e-5f;          // torch LayerNorm default
__device__ __forceinline__ float attn_scale() { return rsqrtf((float)kDhead); }
}  // namespace vit

// ── Exact-erf GELU + derivative (the real nn.GELU(), NOT the tanh surrogate). ──
__device__ __forceinline__ float vit_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * (float)M_SQRT1_2));
}
__device__ __forceinline__ float vit_gelu_grad(float x) {
    const float cdf = 0.5f * (1.0f + erff(x * (float)M_SQRT1_2));
    const float inv_sqrt_2pi = 0.39894228040143270286f;  // 1/sqrt(2*pi)
    const float pdf = inv_sqrt_2pi * __expf(-0.5f * x * x);
    return cdf + x * pdf;
}

// ── Per-CTA scratch for one sample. POD placed in DYNAMIC smem (one instance per
//    CTA; sizeof is static_asserted in vit_layout.cuh). The arrays that must
//    survive fwd→bwd within a layer are kept; the rest are recomputed. All
//    [kSeq × width] row-major (row = position). ────────────────────────────────
struct VitSampleSmem {
    // This sample's image patches (FLOAT) — kept live so the embedding backward
    // can form patch_proj's weight/bias grad. [kNPatch × kPatch] = 16*49.
    float patch[vit::kNPatch][vit::kPatch];             // 16*49 = 784 f
    // Per-layer cached INPUT to each layer (h before the layer) — the only
    // cross-layer state kept, so the backward can recompute each layer's fwd.
    float layer_in[vit::kLayers][vit::kSeq][vit::kD];   // 2*17*128 = 4352 f
    // The final-norm input (last layer output, all positions) for bwd.
    float final_in[vit::kSeq][vit::kD];                 // 17*128 = 2176 f
    // Scratch reused within a single layer's fwd or bwd (one layer live):
    float qkv[vit::kSeq][3 * vit::kD];                  // 17*384 = 6528 f
    float ctx[vit::kSeq][vit::kD];                      // 17*128 = 2176 f
    float x1[vit::kSeq][vit::kD];                       // 17*128 = 2176 f (post n1)
    float ff0[vit::kSeq][vit::kDff];                    // 17*512 = 8704 f (pre-gelu)
    float gact[vit::kSeq][vit::kDff];                   // 17*512 = 8704 f (post-gelu)
    float attn[vit::kHeads][vit::kSeq][vit::kSeq];      // 4*17*17 = 1156 f
    // LayerNorm caches (xhat + inv_std) for n1, n2, final — per position.
    float n1_xhat[vit::kSeq][vit::kD];
    float n1_inv[vit::kSeq];
    float n2_xhat[vit::kSeq][vit::kD];
    float n2_inv[vit::kSeq];
    float fn_xhat[vit::kSeq][vit::kD];
    float fn_inv[vit::kSeq];
    // Backward adjoints carried across sub-steps (one layer live), per position.
    float dh[vit::kSeq][vit::kD];                       // running grad wrt block out
    float logits[vit::kVocab];                          // CLS-pos logits (bwd)
    // Dedicated attention-backward dscores buffer (NOT aliased over `attn`, so the
    // forward softmax weights stay live while dv/dq/dk read them).
    float dsc[vit::kHeads][vit::kSeq][vit::kSeq];
    // Reductions across threads for row stats (mean / max / dot). 256 threads.
    float red[256];
};

// A typed view over the flat weight blob using the generated offsets. All
// pointers are into the single global `params` buffer (read-only in fwd/bwd).
struct VitWeights {
    const float* cls;     // [1,1,d] -> used as [d]
    const float* patch_w; // [d, patch]
    const float* patch_b; // [d]
    const float* pos;     // [kSeq, d]
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
    } layer[vit::kLayers];
    const float* norm_w;  // [d]
    const float* norm_b;  // [d]
    const float* out_w;   // [kVocab, d]
    const float* out_b;   // [kVocab]
};

// Bind VitWeights to the flat params using the generated element offsets. The
// offsets array (csrc/fused/sm_90/vit_layout.cuh::kVitOffsets) is the C++ mirror
// of tests/hw/vit_oracle.py::vit_param_layout(), static-asserted.
__device__ __forceinline__ VitWeights vit_bind(const float* p) {
    VitWeights w;
    int i = 0;
    w.cls     = p + kVitOffsets[i++];
    w.patch_w = p + kVitOffsets[i++];
    w.patch_b = p + kVitOffsets[i++];
    w.pos     = p + kVitOffsets[i++];
    for (int li = 0; li < vit::kLayers; ++li) {
        w.layer[li].in_w  = p + kVitOffsets[i++];
        w.layer[li].in_b  = p + kVitOffsets[i++];
        w.layer[li].out_w = p + kVitOffsets[i++];
        w.layer[li].out_b = p + kVitOffsets[i++];
        w.layer[li].n1_w  = p + kVitOffsets[i++];
        w.layer[li].n1_b  = p + kVitOffsets[i++];
        w.layer[li].n2_w  = p + kVitOffsets[i++];
        w.layer[li].n2_b  = p + kVitOffsets[i++];
        w.layer[li].ff0_w = p + kVitOffsets[i++];
        w.layer[li].ff0_b = p + kVitOffsets[i++];
        w.layer[li].ff2_w = p + kVitOffsets[i++];
        w.layer[li].ff2_b = p + kVitOffsets[i++];
    }
    w.norm_w = p + kVitOffsets[i++];
    w.norm_b = p + kVitOffsets[i++];
    w.out_w  = p + kVitOffsets[i++];
    w.out_b  = p + kVitOffsets[i++];
    return w;
}

// Identical helper for the per-CTA grad PARTIAL buffer (writable, same layout).
struct VitGrad {
    float* cls; float* patch_w; float* patch_b; float* pos;
    struct Layer { float* in_w; float* in_b; float* out_w; float* out_b;
        float* n1_w; float* n1_b; float* n2_w; float* n2_b;
        float* ff0_w; float* ff0_b; float* ff2_w; float* ff2_b; } layer[vit::kLayers];
    float* norm_w; float* norm_b; float* out_w; float* out_b;
};
__device__ __forceinline__ VitGrad vit_bind_grad(float* p) {
    VitGrad w; int i = 0;
    w.cls     = p + kVitOffsets[i++];
    w.patch_w = p + kVitOffsets[i++];
    w.patch_b = p + kVitOffsets[i++];
    w.pos     = p + kVitOffsets[i++];
    for (int li = 0; li < vit::kLayers; ++li) {
        w.layer[li].in_w  = p + kVitOffsets[i++];
        w.layer[li].in_b  = p + kVitOffsets[i++];
        w.layer[li].out_w = p + kVitOffsets[i++];
        w.layer[li].out_b = p + kVitOffsets[i++];
        w.layer[li].n1_w  = p + kVitOffsets[i++];
        w.layer[li].n1_b  = p + kVitOffsets[i++];
        w.layer[li].n2_w  = p + kVitOffsets[i++];
        w.layer[li].n2_b  = p + kVitOffsets[i++];
        w.layer[li].ff0_w = p + kVitOffsets[i++];
        w.layer[li].ff0_b = p + kVitOffsets[i++];
        w.layer[li].ff2_w = p + kVitOffsets[i++];
        w.layer[li].ff2_b = p + kVitOffsets[i++];
    }
    w.norm_w = p + kVitOffsets[i++];
    w.norm_b = p + kVitOffsets[i++];
    w.out_w  = p + kVitOffsets[i++];
    w.out_b  = p + kVitOffsets[i++];
    return w;
}

// ── Block-wide reductions (256 threads) over a value `v`. The smem `red` slot is
//    the VitSampleSmem::red array. All threads must call. ──────────────────────
__device__ __forceinline__ float vit_block_sum(float v, float* red) {
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
__device__ __forceinline__ float vit_block_max(float v, float* red) {
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
//  smem [n_rows, width] tiles. Threads stride over the (row, feature) grid.
//  These mirror the oracle's primitive ops exactly.
// ────────────────────────────────────────────────────────────────────────────

// y[s, :out] = x[s, :in] @ W[out, in]^T + b[out], for s in [0,n_rows). W row-major.
__device__ __forceinline__ void vit_linear(
        const float* __restrict__ x, int in_dim,
        const float* __restrict__ W, const float* __restrict__ b, int out_dim,
        float* __restrict__ y, int n_rows) {
    const int total = n_rows * out_dim;
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
__device__ __forceinline__ void vit_layernorm_fwd(
        const float* __restrict__ x, const float* __restrict__ gamma,
        const float* __restrict__ beta, float* __restrict__ y,
        float* __restrict__ xhat_out, float* __restrict__ inv_out, float* red) {
    for (int s = 0; s < vit::kSeq; ++s) {
        const float* xr = x + s * vit::kD;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) sum += xr[j];
        float mean = vit_block_sum(sum, red) / (float)vit::kD;
        float vs = 0.0f;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float c = xr[j] - mean; vs += c * c;
        }
        float var = vit_block_sum(vs, red) / (float)vit::kD;
        float inv = rsqrtf(var + vit::kLnEps);
        if (threadIdx.x == 0) inv_out[s] = inv;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float xh = (xr[j] - mean) * inv;
            xhat_out[s * vit::kD + j] = xh;
            y[s * vit::kD + j] = xh * gamma[j] + beta[j];
        }
        __syncthreads();
    }
}

// Self-attention (FULL — no causal mask) for one layer, CTA-cooperative. Reads
// q,k,v from sm->qkv columns [0,d),[d,2d),[2d,3d); writes softmax weights into
// sm->attn and the merged context into sm->ctx. Mirrors the oracle's attention.
__device__ __forceinline__ void vit_attention(VitSampleSmem* sm) {
    const float scale = vit::attn_scale();
    // One thread per (head, qpos) computes that row's softmax + ctx slice.
    // kSeq*kHeads = 68 rows; threads stride. Each thread loops d_head.
    for (int r = threadIdx.x; r < vit::kHeads * vit::kSeq; r += blockDim.x) {
        const int hh = r / vit::kSeq;     // head
        const int qi = r % vit::kSeq;     // query position
        const int qoff = hh * vit::kDhead;
        const float* qrow = &sm->qkv[qi][qoff];   // [dhead]
        float maxs = -CUDART_INF_F;
        float sc[vit::kSeq];
        #pragma unroll
        for (int kj = 0; kj < vit::kSeq; ++kj) {   // FULL: every key position
            const float* krow = &sm->qkv[kj][vit::kD + qoff];  // k block
            float dot = 0.0f;
            #pragma unroll
            for (int t = 0; t < vit::kDhead; ++t) dot += qrow[t] * krow[t];
            sc[kj] = dot * scale;
            maxs = fmaxf(maxs, sc[kj]);
        }
        float denom = 0.0f;
        #pragma unroll
        for (int kj = 0; kj < vit::kSeq; ++kj) {
            float e = __expf(sc[kj] - maxs);
            sc[kj] = e; denom += e;
        }
        float invd = 1.0f / denom;
        #pragma unroll
        for (int kj = 0; kj < vit::kSeq; ++kj)
            sm->attn[hh][qi][kj] = sc[kj] * invd;
        // ctx[qi, qoff+t] = sum_kj attn * v[kj, qoff+t]  (FULL: all kj)
        #pragma unroll
        for (int t = 0; t < vit::kDhead; ++t) {
            float acc = 0.0f;
            #pragma unroll
            for (int kj = 0; kj < vit::kSeq; ++kj)
                acc += sm->attn[hh][qi][kj] * sm->qkv[kj][2 * vit::kD + qoff + t];
            sm->ctx[qi][qoff + t] = acc;
        }
    }
    __syncthreads();
}

// ────────────────────────────────────────────────────────────────────────────
//  FORWARD for one sample (CTA-cooperative). The sample's patches must already
//  be loaded into sm->patch. Writes the per-layer input cache + final-norm input
//  + CLS-position logits into `sm`, and returns the sample NLL (all threads see
//  the same value). Recompute-friendly: the layer-local scratch reflects the LAST
//  layer processed; backward recomputes per layer from layer_in.
// ────────────────────────────────────────────────────────────────────────────
__device__ float vit_forward_sample(const VitWeights& w, int target,
                                     VitSampleSmem* sm) {
    // Embedding: position 0 = cls + pos[0]; positions 1..16 = patch_proj(patch_i)
    //   + pos[1+i].  Build h0 into layer_in[0].
    // patch_proj -> rows 1..16 of layer_in[0] (write into a temporary region of
    // qkv, then add pos; but simplest: compute proj directly into layer_in[0]
    // rows 1.. via a linear over the 16 patch rows, with pos added after).
    // Step 1: cls row (position 0).
    for (int j = threadIdx.x; j < vit::kD; j += blockDim.x)
        sm->layer_in[0][0][j] = w.cls[j] + w.pos[j];   // pos[0] == w.pos + 0
    // Step 2: patch_proj over the 16 patch rows into ctx scratch [16,d].
    vit_linear(&sm->patch[0][0], vit::kPatch, w.patch_w, w.patch_b, vit::kD,
               &sm->ctx[0][0], vit::kNPatch);   // ctx[0..15] := proj
    // Step 3: add pos[1+i] -> layer_in[0] rows 1..16.
    for (int idx = threadIdx.x; idx < vit::kNPatch * vit::kD; idx += blockDim.x) {
        const int i = idx / vit::kD, j = idx % vit::kD;
        sm->layer_in[0][1 + i][j] = sm->ctx[i][j]
                                  + w.pos[(int64_t)(1 + i) * vit::kD + j];
    }
    __syncthreads();

    for (int li = 0; li < vit::kLayers; ++li) {
        const VitWeights::Layer& L = w.layer[li];
        float* hin = &sm->layer_in[li][0][0];   // [kSeq, kD]
        // qkv = hin @ in_w^T + in_b -> [kSeq, 3d]
        vit_linear(hin, vit::kD, L.in_w, L.in_b, 3 * vit::kD, &sm->qkv[0][0],
                   vit::kSeq);
        // FULL attention -> sm->ctx, sm->attn.
        vit_attention(sm);
        // a = ctx @ out_w^T + out_b -> x1 (temporary 'a').
        vit_linear(&sm->ctx[0][0], vit::kD, L.out_w, L.out_b, vit::kD,
                   &sm->x1[0][0], vit::kSeq);   // x1 holds 'a' for now
        // r1 = hin + a -> ctx (reuse). n1(r1) -> x1.
        for (int idx = threadIdx.x; idx < vit::kSeq * vit::kD; idx += blockDim.x) {
            const int s = idx / vit::kD, j = idx % vit::kD;
            sm->ctx[s][j] = hin[s * vit::kD + j] + sm->x1[s][j];   // ctx := r1
        }
        __syncthreads();
        vit_layernorm_fwd(&sm->ctx[0][0], L.n1_w, L.n1_b, &sm->x1[0][0],
                          &sm->n1_xhat[0][0], &sm->n1_inv[0], sm->red);
        // FFN: ff0 = x1 @ ff0_w^T + ff0_b; g = gelu(ff0); ff2 = g @ ff2_w^T + b.
        vit_linear(&sm->x1[0][0], vit::kD, L.ff0_w, L.ff0_b, vit::kDff,
                   &sm->ff0[0][0], vit::kSeq);
        for (int idx = threadIdx.x; idx < vit::kSeq * vit::kDff; idx += blockDim.x)
            sm->gact[idx / vit::kDff][idx % vit::kDff] =
                vit_gelu(sm->ff0[idx / vit::kDff][idx % vit::kDff]);
        __syncthreads();
        // ff2 -> ctx scratch (reuse), then r2 = x1 + ff2, n2(r2) -> next layer_in.
        vit_linear(&sm->gact[0][0], vit::kDff, L.ff2_w, L.ff2_b, vit::kD,
                   &sm->ctx[0][0], vit::kSeq);   // ctx := ff2
        for (int idx = threadIdx.x; idx < vit::kSeq * vit::kD; idx += blockDim.x) {
            const int s = idx / vit::kD, j = idx % vit::kD;
            sm->ctx[s][j] = sm->x1[s][j] + sm->ctx[s][j];   // ctx := r2
        }
        __syncthreads();
        float* dst = (li + 1 < vit::kLayers) ? &sm->layer_in[li + 1][0][0]
                                             : &sm->final_in[0][0];
        vit_layernorm_fwd(&sm->ctx[0][0], L.n2_w, L.n2_b, dst,
                          &sm->n2_xhat[0][0], &sm->n2_inv[0], sm->red);
    }

    // Final norm on the CLS position (0) only, then head → logits → NLL.
    const float* hcls = &sm->final_in[0][0];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) sum += hcls[j];
    float mean = vit_block_sum(sum, sm->red) / (float)vit::kD;
    float vs = 0.0f;
    for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
        float c = hcls[j] - mean; vs += c * c;
    }
    float var = vit_block_sum(vs, sm->red) / (float)vit::kD;
    float inv = rsqrtf(var + vit::kLnEps);
    if (threadIdx.x == 0) sm->fn_inv[0] = inv;
    for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
        float xh = (hcls[j] - mean) * inv;
        sm->fn_xhat[0][j] = xh;                    // CLS xhat stored at row 0
        sm->x1[0][j] = xh * w.norm_w[j] + w.norm_b[j];   // x1[0] := hn
    }
    __syncthreads();
    // logits[o] = hn · out_w[o] + out_b[o]  (o over kVocab).
    for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x) {
        const float* Wr = w.out_w + (int64_t)o * vit::kD;
        float acc = w.out_b[o];
        #pragma unroll 4
        for (int k = 0; k < vit::kD; ++k) acc += sm->x1[0][k] * Wr[k];
        sm->logits[o] = acc;
    }
    __syncthreads();
    // NLL (mean handled by /B in the loss reduce): logz - logits[target].
    float lmax = -CUDART_INF_F;
    for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x)
        lmax = fmaxf(lmax, sm->logits[o]);
    lmax = vit_block_max(lmax, sm->red);
    float es = 0.0f;
    for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x)
        es += __expf(sm->logits[o] - lmax);
    es = vit_block_sum(es, sm->red);
    float logz = lmax + __logf(es);
    return logz - sm->logits[target];   // sample NLL
}

// ── Recompute one layer's forward intermediates into sm (qkv, x1, ff0, gact,
//    attn, and ctx := r2), reading from sm->layer_in[li]. Mirrors the fwd layer
//    body but leaves r2 in sm->ctx and x1 in sm->x1 for the backward chain. ─────
__device__ void vit_recompute_layer(const VitWeights& w, int li,
                                     VitSampleSmem* sm) {
    const VitWeights::Layer& L = w.layer[li];
    float* hin = &sm->layer_in[li][0][0];
    vit_linear(hin, vit::kD, L.in_w, L.in_b, 3 * vit::kD, &sm->qkv[0][0],
               vit::kSeq);
    vit_attention(sm);   // fills sm->attn + sm->ctx (== attention context)
    vit_linear(&sm->ctx[0][0], vit::kD, L.out_w, L.out_b, vit::kD, &sm->x1[0][0],
               vit::kSeq);   // x1 := a
    for (int idx = threadIdx.x; idx < vit::kSeq * vit::kD; idx += blockDim.x) {
        const int s = idx / vit::kD, j = idx % vit::kD;
        sm->ctx[s][j] = hin[s * vit::kD + j] + sm->x1[s][j];   // ctx := r1
    }
    __syncthreads();
    vit_layernorm_fwd(&sm->ctx[0][0], L.n1_w, L.n1_b, &sm->x1[0][0],
                      &sm->n1_xhat[0][0], &sm->n1_inv[0], sm->red);  // x1 := n1(r1)
    vit_linear(&sm->x1[0][0], vit::kD, L.ff0_w, L.ff0_b, vit::kDff, &sm->ff0[0][0],
               vit::kSeq);
    for (int idx = threadIdx.x; idx < vit::kSeq * vit::kDff; idx += blockDim.x) {
        const int s = idx / vit::kDff, j = idx % vit::kDff;
        sm->gact[s][j] = vit_gelu(sm->ff0[s][j]);
    }
    __syncthreads();
    // ff2 -> tmp (dh scratch, unused until bwd), r2 = x1 + ff2 -> ctx.
    vit_linear(&sm->gact[0][0], vit::kDff, L.ff2_w, L.ff2_b, vit::kD, &sm->dh[0][0],
               vit::kSeq);
    for (int idx = threadIdx.x; idx < vit::kSeq * vit::kD; idx += blockDim.x) {
        const int s = idx / vit::kD, j = idx % vit::kD;
        sm->ctx[s][j] = sm->x1[s][j] + sm->dh[s][j];   // ctx := r2
    }
    __syncthreads();
    // Fill n2 caches (we don't need n2's output for bwd, only xhat/inv).
    vit_layernorm_fwd(&sm->ctx[0][0], L.n2_w, L.n2_b, &sm->dh[0][0],
                      &sm->n2_xhat[0][0], &sm->n2_inv[0], sm->red);
    __syncthreads();
}

// LayerNorm backward (same as decoder): given dy [kSeq,d] and cached (xhat,inv),
// returns dx [kSeq,d] into dx_out and ACCUMULATES dgamma/dbeta into gw/gb [d].
// Owner-thread rule for gw/gb: thread owns feature j (strided), sums over rows.
__device__ void vit_layernorm_bwd(
        const float* __restrict__ dy, const float* __restrict__ xhat,
        const float* __restrict__ inv, const float* __restrict__ gamma,
        float* __restrict__ dx_out, float* __restrict__ gw, float* __restrict__ gb,
        float* red, bool accum_affine) {
    if (accum_affine) {
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float dgw = 0.0f, dgb = 0.0f;
            #pragma unroll
            for (int s = 0; s < vit::kSeq; ++s) {
                float d = dy[s * vit::kD + j];
                dgb += d; dgw += d * xhat[s * vit::kD + j];
            }
            gw[j] += dgw; gb[j] += dgb;
        }
    }
    for (int s = 0; s < vit::kSeq; ++s) {
        const float* dyr = dy + s * vit::kD;
        const float* xhr = xhat + s * vit::kD;
        float sda = 0.0f, sdax = 0.0f;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float dxhat = dyr[j] * gamma[j];
            sda += dxhat; sdax += dxhat * xhr[j];
        }
        sda = vit_block_sum(sda, red);
        sdax = vit_block_sum(sdax, red);
        float inv_s = inv[s];
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float dxhat = dyr[j] * gamma[j];
            dx_out[s * vit::kD + j] =
                inv_s * (dxhat - (sda + xhr[j] * sdax) / (float)vit::kD);
        }
        __syncthreads();
    }
}

// dW += dY^T @ X for a linear Y = X @ W^T + b. dY [kSeq,out], X [kSeq,in],
// dW [out,in]. db[o] += sum_s dY[s,o]. dX [kSeq,in] = dY @ W into dx_out
// (set or accumulate per set_dx).
//
// ldY/ldX/ldDX are the ROW STRIDES (leading dimensions) of dY/X/dx_out. They are
// EXPLICIT (no defaults) — IDENTICAL contract to the decoder's dec_linear_bwd.
// The smem scratch buffers this is called on are 2D arrays whose DECLARED width
// (kDff=512 for ff0/gact) DIFFERS from the logical row width passed as
// out_dim/in_dim (dctx=kD=128, dqkv=3*kD=384). The original packed-stride form
// (dY[s*out_dim+o], X[s*in_dim+i], dx_out[s*in_dim+i]) silently read/wrote the
// WRONG rows at both attention-seam call sites: only row 0 partially landed,
// rows 1.. hit stale recomputed values — forward stayed exact while every grad
// upstream of attention was garbage (params decayed to zero under weight decay;
// train acc pinned at 1/97). The CPU structural mirror (vit_kernel_mirror.py)
// does NOT catch it (it indexes buffers semantically, not by flat offset), so the
// stride is now part of the call contract. W/dW stay at in_dim — those are the
// genuinely-packed param/grad tensors, not smem scratch.
__device__ void vit_linear_bwd(
        const float* __restrict__ dY, const float* __restrict__ X,
        const float* __restrict__ W, int in_dim, int out_dim,
        float* __restrict__ dW, float* __restrict__ db,
        float* __restrict__ dx_out, bool set_dx,
        int ldY, int ldX, int ldDX) {
    const int wtot = out_dim * in_dim;
    for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
        const int o = idx / in_dim, i = idx % in_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < vit::kSeq; ++s)
            acc += dY[s * ldY + o] * X[s * ldX + i];
        dW[(int64_t)o * in_dim + i] += acc;
    }
    if (db != nullptr) {
        for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
            float acc = 0.0f;
            #pragma unroll
            for (int s = 0; s < vit::kSeq; ++s) acc += dY[s * ldY + o];
            db[o] += acc;
        }
    }
    const int xtot = vit::kSeq * in_dim;
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
//  BACKWARD for one sample (CTA-cooperative). Assumes vit_forward_sample has run
//  for THIS sample (so layer_in/final_in/fn caches/logits + sm->patch are live).
//  Accumulates every weight grad into the CTA's partial buffer `g`.
//
//  Transcribed from tests/hw/vit_oracle.py::vit_backward (structurally mirrored
//  in tests/hw/vit_kernel_mirror.py).
// ────────────────────────────────────────────────────────────────────────────
__device__ void vit_backward_sample(const VitWeights& w, const VitGrad& g,
                                     int target, int B, VitSampleSmem* sm) {
    // ── CE backward: dlogits = (softmax - onehot)/B. ──
    float lmax = -CUDART_INF_F;
    for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x)
        lmax = fmaxf(lmax, sm->logits[o]);
    lmax = vit_block_max(lmax, sm->red);
    float es = 0.0f;
    for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x)
        es += __expf(sm->logits[o] - lmax);
    es = vit_block_sum(es, sm->red);
    float inv_es = 1.0f / es;
    for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x) {
        float sm_o = __expf(sm->logits[o] - lmax) * inv_es;
        float dl = (sm_o - ((o == target) ? 1.0f : 0.0f)) / (float)B;
        sm->logits[o] = dl;   // overwrite logits[] with dlogits in place
    }
    __syncthreads();
    // ── head (out): logits = hn @ out_w^T + out_b. hn recomputed from the cached
    //    final-norm: hn[j] = fn_xhat[0][j]*norm_w[j] + norm_b[j]. ──
    for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x) {
        float dl = sm->logits[o];
        g.out_b[o] += dl;
        float* gwrow = g.out_w + (int64_t)o * vit::kD;
        const float* xh = &sm->fn_xhat[0][0];
        #pragma unroll 4
        for (int j = 0; j < vit::kD; ++j) {
            float hn_j = xh[j] * w.norm_w[j] + w.norm_b[j];
            gwrow[j] += dl * hn_j;
        }
    }
    __syncthreads();
    // dhn[j] (feature-parallel) into dh row 0 scratch.
    for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
        float acc = 0.0f;
        for (int o = 0; o < vit::kVocab; ++o)
            acc += sm->logits[o] * w.out_w[(int64_t)o * vit::kD + j];
        sm->dh[0][j] = acc;   // dhn
    }
    __syncthreads();
    // ── final norm backward (single CLS row 0): dy=dhn, xhat=fn_xhat[0],
    //    inv=fn_inv[0]. Produces dh into dh[0]; zeros all other rows; accumulate
    //    dnorm_w/dnorm_b. Stage dy into ctx row 0. ──
    for (int j = threadIdx.x; j < vit::kD; j += blockDim.x)
        sm->ctx[0][j] = sm->dh[0][j];   // dy at CLS position
    __syncthreads();
    for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
        float d = sm->ctx[0][j];
        float xh = sm->fn_xhat[0][j];
        g.norm_b[j] += d; g.norm_w[j] += d * xh;
    }
    __syncthreads();
    {
        const float* dyr = &sm->ctx[0][0];
        const float* xhr = &sm->fn_xhat[0][0];
        float sda = 0.0f, sdax = 0.0f;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float dxhat = dyr[j] * w.norm_w[j];
            sda += dxhat; sdax += dxhat * xhr[j];
        }
        sda = vit_block_sum(sda, sm->red);
        sdax = vit_block_sum(sdax, sm->red);
        float inv_s = sm->fn_inv[0];
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float dxhat = dyr[j] * w.norm_w[j];
            for (int s = 0; s < vit::kSeq; ++s) sm->dh[s][j] = 0.0f;
            sm->dh[0][j] = inv_s * (dxhat - (sda + xhr[j] * sdax) / (float)vit::kD);
        }
        __syncthreads();
    }
    // dh now = grad wrt the last layer's OUTPUT [kSeq,d] (only CLS pos 0 nonzero).

    // ── per-layer backward (reverse). Each layer: recompute fwd intermediates,
    //    then apply the chain. The running adjoint `dh` is preserved across the
    //    recompute by SAVING it into final_in (free after fwd) around the
    //    recompute (which temporarily uses sm->dh as ff2 scratch). ─────────────
    for (int li = vit::kLayers - 1; li >= 0; --li) {
        const VitWeights::Layer& L = w.layer[li];
        const VitGrad::Layer& G = g.layer[li];
        // Save incoming adjoint dh into final_in — recompute clobbers sm->dh.
        for (int idx = threadIdx.x; idx < vit::kSeq * vit::kD; idx += blockDim.x) {
            const int s = idx / vit::kD, j = idx % vit::kD;
            sm->final_in[s][j] = sm->dh[s][j];
        }
        __syncthreads();
        vit_recompute_layer(w, li, sm);   // fills qkv,x1,ff0,gact,attn,ctx(=r2),caches
        // Restore dh (grad wrt h2 = n2 output).
        for (int idx = threadIdx.x; idx < vit::kSeq * vit::kD; idx += blockDim.x) {
            const int s = idx / vit::kD, j = idx % vit::kD;
            sm->dh[s][j] = sm->final_in[s][j];
        }
        __syncthreads();
        // n2 backward: dh -> dr2 (into ctx), accumulate n2 gamma/beta.
        vit_layernorm_bwd(&sm->dh[0][0], &sm->n2_xhat[0][0], &sm->n2_inv[0],
                          L.n2_w, &sm->ctx[0][0], G.n2_w, G.n2_b, sm->red, true);
        // r2 = x1 + ff2 -> dx1 (residual) := dr2 into fn_xhat; dr2 stays in ctx.
        for (int idx = threadIdx.x; idx < vit::kSeq * vit::kD; idx += blockDim.x) {
            const int s = idx / vit::kD, j = idx % vit::kD;
            sm->fn_xhat[s][j] = sm->ctx[s][j];   // dx1 := dr2 (residual part)
        }
        __syncthreads();
        // ff.2 backward. dW_ff2 += dr2^T @ g; db_ff2 += Σ dr2; then dff0 written
        // OVER gact (its forward value consumed by the dW GEMM first).
        {
            const float* dY = &sm->ctx[0][0];   // dr2 [kSeq,d]
            const float* X = &sm->gact[0][0];   // g [kSeq,4d]
            const int wtot = vit::kD * vit::kDff;
            for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
                const int o = idx / vit::kDff, i = idx % vit::kDff;
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < vit::kSeq; ++s)
                    acc += dY[s * vit::kD + o] * X[s * vit::kDff + i];
                G.ff2_w[(int64_t)o * vit::kDff + i] += acc;
            }
            for (int o = threadIdx.x; o < vit::kD; o += blockDim.x) {
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < vit::kSeq; ++s) acc += dY[s * vit::kD + o];
                G.ff2_b[o] += acc;
            }
            __syncthreads();
            const int xtot = vit::kSeq * vit::kDff;
            for (int idx = threadIdx.x; idx < xtot; idx += blockDim.x) {
                const int s = idx / vit::kDff, i = idx % vit::kDff;
                float dg = 0.0f;
                for (int o = 0; o < vit::kD; ++o)
                    dg += dY[s * vit::kD + o] * L.ff2_w[(int64_t)o * vit::kDff + i];
                sm->gact[s][i] = dg * vit_gelu_grad(sm->ff0[s][i]);  // gact := dff0
            }
            __syncthreads();
        }
        // ff.0 backward: dY=dff0(=gact), X=x1 -> dW ff0, db ff0, dx1 += dff0 @ ff0_w.
        {
            const float* dY = &sm->gact[0][0];   // dff0 [kSeq,4d]
            const float* X = &sm->x1[0][0];      // x1 [kSeq,d]
            const int wtot = vit::kDff * vit::kD;
            for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
                const int o = idx / vit::kD, i = idx % vit::kD;
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < vit::kSeq; ++s)
                    acc += dY[s * vit::kDff + o] * X[s * vit::kD + i];
                G.ff0_w[(int64_t)o * vit::kD + i] += acc;
            }
            for (int o = threadIdx.x; o < vit::kDff; o += blockDim.x) {
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < vit::kSeq; ++s) acc += dY[s * vit::kDff + o];
                G.ff0_b[o] += acc;
            }
            __syncthreads();
            const int xtot = vit::kSeq * vit::kD;
            for (int idx = threadIdx.x; idx < xtot; idx += blockDim.x) {
                const int s = idx / vit::kD, i = idx % vit::kD;
                float acc = 0.0f;
                for (int o = 0; o < vit::kDff; ++o)
                    acc += dY[s * vit::kDff + o] * L.ff0_w[(int64_t)o * vit::kD + i];
                sm->fn_xhat[s][i] += acc;   // dx1 += FFN path
            }
            __syncthreads();
        }
        // dx1 (in fn_xhat) -> n1 backward -> dr1 (into ctx). accumulate n1 g/b.
        vit_layernorm_bwd(&sm->fn_xhat[0][0], &sm->n1_xhat[0][0], &sm->n1_inv[0],
                          L.n1_w, &sm->ctx[0][0], G.n1_w, G.n1_b, sm->red, true);
        // r1 = x_in + a -> da = dr1; dx_in := dr1 (residual) into fn_xhat. da in ctx.
        for (int idx = threadIdx.x; idx < vit::kSeq * vit::kD; idx += blockDim.x) {
            const int s = idx / vit::kD, j = idx % vit::kD;
            sm->fn_xhat[s][j] = sm->ctx[s][j];   // dx_in := dr1 (residual)
        }
        __syncthreads();
        // Rebuild ctx_fwd (attention context) from attn + v into x1 (x1 free —
        // ff bwd done). ctx_fwd[qi,col] = Σ_kj attn[hh][qi][kj] * v[kj, col].
        for (int r = threadIdx.x; r < vit::kSeq * vit::kD; r += blockDim.x) {
            const int qi = r / vit::kD, col = r % vit::kD;
            const int hh = col / vit::kDhead, t = col % vit::kDhead;
            float acc = 0.0f;
            #pragma unroll
            for (int kj = 0; kj < vit::kSeq; ++kj)   // FULL: all kj
                acc += sm->attn[hh][qi][kj]
                     * sm->qkv[kj][2 * vit::kD + hh * vit::kDhead + t];
            sm->x1[qi][col] = acc;   // x1 := ctx_fwd
        }
        __syncthreads();
        // da is in ctx. out_proj bwd: dW_out += da^T @ ctx_fwd(x1); db_out += Σ da;
        // dctx = da @ out_w -> ff0 scratch (free). dctx (logical width kD) MUST be
        // written at ff0's DECLARED row stride (kDff) — the attention-bwd readers
        // below (A at 824, B at 839) read it as sm->ff0[qi][col]; a packed
        // (kD-stride) write would fill only row 0's first 128 floats and leave
        // rows 1..16 reading stale recomputed pre-GELU values.
        vit_linear_bwd(&sm->ctx[0][0], &sm->x1[0][0], L.out_w, vit::kD, vit::kD,
                       G.out_w, G.out_b, &sm->ff0[0][0], /*set_dx=*/true, // dctx->ff0
                       /*ldY=*/vit::kD, /*ldX=*/vit::kD, /*ldDX=*/vit::kDff);
        // ── FULL attention backward (3 synced passes; `attn` read by A & B, never
        //    overwritten — dscores go to `dsc`). dctx [kSeq,d] in ff0; v/k/q in
        //    qkv[:,2d:3d]/[:,d:2d]/[:,0:d]; dqkv built into gact[:, :3d]. ────────
        const float scale = vit::attn_scale();
        // A: dv into gact[:, 2d:3d]. dv[kj,qoff+t]=Σ_{qi} attn[hh][qi][kj]
        //    * dctx[qi, qoff+t].  (FULL: all qi. Owner: thread per (kj, head, t).)
        for (int r = threadIdx.x; r < vit::kSeq * vit::kHeads * vit::kDhead;
             r += blockDim.x) {
            const int kj = r / (vit::kHeads * vit::kDhead);
            const int rem = r % (vit::kHeads * vit::kDhead);
            const int hh = rem / vit::kDhead, t = rem % vit::kDhead;
            const int qoff = hh * vit::kDhead;
            float acc = 0.0f;
            #pragma unroll
            for (int qi = 0; qi < vit::kSeq; ++qi)   // FULL (no causal qi>=kj)
                acc += sm->attn[hh][qi][kj] * sm->ff0[qi][qoff + t];
            sm->gact[kj][2 * vit::kD + qoff + t] = acc;   // dv block
        }
        __syncthreads();
        // B: dscores into dsc. datt[hh,qi,kj]=Σ_t dctx[qi,qoff+t]*v[kj,qoff+t];
        //    ds = attn*(datt - Σ_k datt*attn)*scale  (FULL: all kj).
        for (int r = threadIdx.x; r < vit::kHeads * vit::kSeq; r += blockDim.x) {
            const int hh = r / vit::kSeq, qi = r % vit::kSeq;
            const int qoff = hh * vit::kDhead;
            float datt[vit::kSeq];
            #pragma unroll
            for (int kj = 0; kj < vit::kSeq; ++kj) {
                float acc = 0.0f;
                #pragma unroll
                for (int t = 0; t < vit::kDhead; ++t)
                    acc += sm->ff0[qi][qoff + t]
                         * sm->qkv[kj][2 * vit::kD + qoff + t];
                datt[kj] = acc;
            }
            float dot = 0.0f;
            #pragma unroll
            for (int kj = 0; kj < vit::kSeq; ++kj) dot += datt[kj] * sm->attn[hh][qi][kj];
            #pragma unroll
            for (int kj = 0; kj < vit::kSeq; ++kj) {
                float a = sm->attn[hh][qi][kj];
                sm->dsc[hh][qi][kj] = a * (datt[kj] - dot) * scale;
            }
        }
        __syncthreads();
        // C: dq into gact[:,0:d], dk into gact[:,d:2d].  (FULL: all kj/qi.)
        for (int r = threadIdx.x; r < vit::kSeq * vit::kHeads * vit::kDhead;
             r += blockDim.x) {
            const int pos = r / (vit::kHeads * vit::kDhead);
            const int rem = r % (vit::kHeads * vit::kDhead);
            const int hh = rem / vit::kDhead, t = rem % vit::kDhead;
            const int qoff = hh * vit::kDhead;
            float dq = 0.0f, dk = 0.0f;
            #pragma unroll
            for (int kj = 0; kj < vit::kSeq; ++kj) {
                dq += sm->dsc[hh][pos][kj] * sm->qkv[kj][vit::kD + qoff + t];
                dk += sm->dsc[hh][kj][pos] * sm->qkv[kj][qoff + t];
            }
            sm->gact[pos][qoff + t] = dq;            // dq block (gact[:,0:d])
            sm->gact[pos][vit::kD + qoff + t] = dk;  // dk block (gact[:,d:2d])
        }
        __syncthreads();
        // in_proj backward: dqkv (gact[:, :3d]) -> dW in, db in, dx_in += dqkv @ in_w.
        // dqkv was written 2D (sm->gact[pos][col]) at gact's DECLARED width kDff, so
        // dY's row stride is kDff — a packed (3d=384) read would walk off row 0 into
        // the stale dq/dk/dv values left in gact[0][384:512]. X/dx_out (layer_in,
        // fn_xhat) are genuinely kD-wide, so ldX=ldDX=kD.
        vit_linear_bwd(&sm->gact[0][0], &sm->layer_in[li][0][0], L.in_w,
                       vit::kD, 3 * vit::kD, G.in_w, G.in_b,
                       &sm->fn_xhat[0][0], /*set_dx=*/false,   // dx_in += attn path
                       /*ldY=*/vit::kDff, /*ldX=*/vit::kD, /*ldDX=*/vit::kD);
        // dh for the previous layer = dx_in (in fn_xhat). Move to dh.
        for (int idx = threadIdx.x; idx < vit::kSeq * vit::kD; idx += blockDim.x) {
            const int s = idx / vit::kD, j = idx % vit::kD;
            sm->dh[s][j] = sm->fn_xhat[s][j];
        }
        __syncthreads();
    }

    // ── embedding backward: dh = grad wrt h0 [kSeq,d] (post-cat, pre-pos-add).
    //    pos.weight[s] += dh[s] (ALL 17 positions); cls_token += dh[0]; patch_proj
    //    weight/bias from rows 1..16 against the live patch pixels. Owner-thread:
    //    thread owns column j (for pos/cls) or (o,i)/(o) (for patch). ───────────
    for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
        #pragma unroll
        for (int s = 0; s < vit::kSeq; ++s)
            g.pos[(int64_t)s * vit::kD + j] += sm->dh[s][j];   // pos (all rows)
        g.cls[j] += sm->dh[0][j];                              // cls_token (row 0)
    }
    __syncthreads();
    // patch_proj.weight [d, patch] and bias [d]: owner thread per (o,i) / per o;
    // sum over the 16 patch positions (sequential), reading sm->patch.
    const int pwtot = vit::kD * vit::kPatch;
    for (int idx = threadIdx.x; idx < pwtot; idx += blockDim.x) {
        const int o = idx / vit::kPatch, i = idx % vit::kPatch;
        float acc = 0.0f;
        #pragma unroll
        for (int p = 0; p < vit::kNPatch; ++p)
            acc += sm->dh[1 + p][o] * sm->patch[p][i];
        g.patch_w[(int64_t)o * vit::kPatch + i] += acc;
    }
    for (int o = threadIdx.x; o < vit::kD; o += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int p = 0; p < vit::kNPatch; ++p) acc += sm->dh[1 + p][o];
        g.patch_b[o] += acc;
    }
    __syncthreads();
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGE_VIT_CUH_
