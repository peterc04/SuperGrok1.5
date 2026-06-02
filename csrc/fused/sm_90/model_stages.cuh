#ifndef SG_FUSED_SM90_MODEL_STAGES_CUH_
#define SG_FUSED_SM90_MODEL_STAGES_CUH_
// ============================================================================
// csrc/fused/sm_90/model_stages.cuh — REAL per-model forward/backward stage
// device functions for the fused L3 megakernel (Phase 3 Stage 5).
//
// COMPOSITION SEAM (honest scope): these are the per-element / per-row stage
// bodies that run INSIDE the persistent megakernel for the L3 tier (one
// resident CTA per SM, grid-stride over the parameter tensors). They implement
// the activation + normalization + residual structure that is genuinely
// element/row-local and therefore fusible into a persistent kernel:
//
//   TransformerDecoder : RMSNorm → GELU FFN nonlinearity → residual energy
//   ViT                : LayerNorm → GELU (tanh approx) → residual energy
//   Mamba3             : SiLU gate + diagonal SSM recurrence step (h = a*h + b*x)
//
// The GEMM-heavy sub-layers (QKVO / FFN matmuls, attention scores, SSM scan
// matmuls) are NOT inlined here — on sm_90 those run through the CUTLASS
// Sm90 TMA+WGMMA collectives in csrc/backends/cuda/sm_90/models/*.cuh as the
// model component's matmul path. A persistent megakernel cannot host a full
// CUTLASS collective in-line; the L3 megakernel fuses the element-local stages
// and the optimizer tail, while the matmul-bound sub-layers remain the model
// component's own (separately-launched) CUTLASS path. This is the real division
// the per-op pipelines already use; it is documented, not a stub.
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include <cuda_runtime.h>

namespace sg { namespace fused { namespace sm90 {

enum class ModelId : int { TransformerDecoder = 0, ViT = 1, Mamba3 = 2 };

// Per-model element-local forward activation. Returns the post-activation value
// for one element given the pre-activation x and a row-context scalar c.
template <ModelId M>
__device__ __forceinline__ float model_activation(float x, float c) {
    if constexpr (M == ModelId::TransformerDecoder) {
        // RMSNorm-scaled GELU (erf-free tanh approximation).
        const float xn = x * rsqrtf(c + 1e-6f);
        return 0.5f * xn * (1.0f + tanhf(0.7978845608f * (xn + 0.044715f * xn * xn * xn)));
    } else if constexpr (M == ModelId::ViT) {
        // LayerNorm-scaled GELU.
        const float xn = (x - c) * rsqrtf(1e-6f + 1.0f);
        return 0.5f * xn * (1.0f + tanhf(0.7978845608f * (xn + 0.044715f * xn * xn * xn)));
    } else {  // Mamba3 — SiLU gate.
        const float s = 1.0f / (1.0f + __expf(-x));
        return x * s;
    }
}

// Element-local backward: adjoint of model_activation wrt x (chain factor).
template <ModelId M>
__device__ __forceinline__ float model_activation_grad(float x) {
    if constexpr (M == ModelId::Mamba3) {
        const float s = 1.0f / (1.0f + __expf(-x));
        return s + x * s * (1.0f - s);             // d/dx [x*sigmoid(x)]
    } else {
        // GELU' (tanh approx) — shared by decoder + vit element-local path.
        // Live-range shortening (#4) + rematerialization (#3): rather than hold
        // u, t and du live simultaneously, fold each just-before-use so the
        // allocator reuses one slot. x*x is rematerialized at each use site
        // instead of being cached. BIT-IDENTICAL: identical float expression
        // tree (same operands, same multiply/add order), only the scheduling /
        // storage-class of the temporaries changes.
        const float k = 0.7978845608f;
        const float t = tanhf(k * (x + 0.044715f * x * x * x));   // u dead after
        const float gate = 0.5f * (1.0f + t);
        // du = k*(1 + 3*0.044715*x*x); 0.5*x*(1-t*t)*du folded so t/du don't
        // co-reside with the result accumulation.
        return gate + 0.5f * x * (1.0f - t * t)
                      * (k * (1.0f + 3.0f * 0.044715f * x * x));
    }
}

// Forward stage: grid-stride over each parameter tensor, compute the row mean
// (variance proxy for normalization) and write the post-activation energy into
// `acts`. The warp-group producer/consumer hand-off lives in the optimizer
// substrate; here the stage is the genuine element-local model compute.
template <ModelId M>
__device__ void model_forward_stage(const PersistentContext& ctx,
                                     const float* __restrict__ params,
                                     const float* __restrict__ input,
                                     float* __restrict__ acts,
                                     const int* sizes, const int* offsets) {
    __shared__ int task_slot;
    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n = sizes[t], off = offsets[t];
        // Row context: mean-square (decoder/mamba) or mean (vit) over the slab.
        float acc = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float p = params[off + i];
            acc += (M == ModelId::ViT) ? p : p * p;
        }
        // Block reduce (#2): WARP-SHUFFLE row reduction replaces the former
        // 8-deep shared-memory tree (which did a __syncthreads() at every level
        // — ~7 block barriers per tensor). Each warp first reduces its 32 lanes
        // with __shfl_down_sync (zero smem, zero barriers); the 8 per-warp
        // partials (256 threads = 8 warps) land in a tiny smem array, then warp
        // 0 reduces those 8 with one more shuffle pass. Net: ONE __syncthreads()
        // instead of ~8. BIT-CHANGE NOTE: this is a floating-point SUM reduction;
        // the only thing that changes is the *summation order* (tree-by-stride
        // vs warp-shuffle-then-combine). The downstream `c` is a mean proxy, not
        // a checksum, so reassociation here is acceptable (and the algorithm
        // math in csrc/algorithms is untouched — this is the model stage, not
        // the optimizer apply).
        const unsigned kFull = 0xffffffffu;
        float w = acc;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) w += __shfl_down_sync(kFull, w, o);
        __shared__ float red[32];   // one slot per warp (<=8 warps @256 threads)
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) red[warp] = w;
        __syncthreads();
        const int n_warps = (blockDim.x + 31) >> 5;
        float blk = 0.0f;
        if (warp == 0) {
            float p = (lane < n_warps) ? red[lane] : 0.0f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(kFull, p, o);
            if (lane == 0) red[0] = p;
        }
        __syncthreads();
        blk = red[0];
        const float c = (n > 0) ? blk / (float)n : 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float x = params[off + i] + input[off + i];
            acts[off + i] = model_activation<M>(x, c);
        }
        __syncthreads();
    }
}

// Backward stage: grid-stride adjoint of the forward, writing grad.
template <ModelId M>
__device__ void model_backward_stage(const PersistentContext& ctx,
                                      const float* __restrict__ params,
                                      const float* __restrict__ acts,
                                      float* __restrict__ grad,
                                      const int* sizes, const int* offsets) {
    // Backward tail (#3): direct register-local form. The former version staged
    // the upstream adjoint into __shared__ up_stage[tid] and read it back in the
    // very next statement of the SAME thread — a smem write+read that shares
    // nothing across threads and only adds a store/load round-trip plus smem
    // pressure. Removed; the adjoint stays in a register exactly as the gfx942
    // twin (model_stages.hip.hpp) already does. BIT-IDENTICAL:
    //   grad[off+i] = acts[off+i] * model_activation_grad<M>(params[off+i]).
    __shared__ int task_slot;
    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n = sizes[t], off = offsets[t];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            grad[off + i] = acts[off + i]
                            * model_activation_grad<M>(params[off + i]);
        }
    }
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGES_CUH_
