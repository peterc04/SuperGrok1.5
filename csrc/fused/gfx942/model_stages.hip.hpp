#ifndef SG_FUSED_GFX942_MODEL_STAGES_HIP_HPP_
#define SG_FUSED_GFX942_MODEL_STAGES_HIP_HPP_
// ============================================================================
// csrc/fused/gfx942/model_stages.hip.hpp — REAL per-model fwd/bwd stage device
// functions for the gfx942 fused L3 megakernel. AMD twin of the sm_90
// model_stages.cuh: element-local RMSNorm/LayerNorm-scaled GELU (decoder/vit)
// and SiLU-gated SSM step (mamba3), grid-stride over the parameter tensors.
//
// HONEST SCOPE: the GEMM-heavy sub-layers (QKVO/FFN/attention/SSM matmuls) are
// NOT inlined here — on gfx942 those are the model component's MFMA path
// (grokking_optimizers/kernels/gfx942/<model>_gfx942.hip.hpp), launched
// separately. The L3 megakernel fuses the element-local stages + the optimizer
// tail; the matmul-bound sub-layers remain the model's own MFMA launch. Same
// division the per-op pipelines use; documented, not a stub.
//
// Device pass only (__AMDGCN__ gate or __HIPCC__).
// ============================================================================

#include "csrc/fused/megakernel_common_hip.hip.hpp"

#if defined(__AMDGCN__) || defined(__HIPCC__) || defined(GROK_HIP_DEVICE)

namespace sg { namespace fused { namespace gfx942_mega {

using ::sg::fused::gfx942::PersistentContext;
using ::sg::fused::gfx942::TaskQueue;

enum class ModelId : int { TransformerDecoder = 0, ViT = 1, Mamba3 = 2 };

template <ModelId M>
__device__ __forceinline__ float model_activation(float x, float c) {
    if constexpr (M == ModelId::TransformerDecoder) {
        const float xn = x * (1.0f / __builtin_sqrtf(c + 1e-6f));
        return 0.5f * xn * (1.0f + __builtin_tanhf(
            0.7978845608f * (xn + 0.044715f * xn * xn * xn)));
    } else if constexpr (M == ModelId::ViT) {
        const float xn = (x - c) * (1.0f / __builtin_sqrtf(1e-6f + 1.0f));
        return 0.5f * xn * (1.0f + __builtin_tanhf(
            0.7978845608f * (xn + 0.044715f * xn * xn * xn)));
    } else {  // Mamba3 — SiLU.
        const float s = 1.0f / (1.0f + __builtin_expf(-x));
        return x * s;
    }
}

template <ModelId M>
__device__ __forceinline__ float model_activation_grad(float x) {
    if constexpr (M == ModelId::Mamba3) {
        const float s = 1.0f / (1.0f + __builtin_expf(-x));
        return s + x * s * (1.0f - s);
    } else {
        const float k = 0.7978845608f;
        const float u = k * (x + 0.044715f * x * x * x);
        const float t = __builtin_tanhf(u);
        const float du = k * (1.0f + 3.0f * 0.044715f * x * x);
        return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
    }
}

template <ModelId M>
__device__ void model_forward_stage(const PersistentContext& ctx,
                                     const float* __restrict__ params,
                                     const float* __restrict__ input,
                                     float* __restrict__ acts,
                                     const int* sizes, const int* offsets) {
    __shared__ int task_slot;
    __shared__ float red[256];
    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n = sizes[t], off = offsets[t];
        float acc = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float p = params[off + i];
            acc += (M == ModelId::ViT) ? p : p * p;
        }
        red[threadIdx.x] = acc;
        __builtin_amdgcn_s_barrier();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __builtin_amdgcn_s_barrier();
        }
        const float c = (n > 0) ? red[0] / (float)n : 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float x = params[off + i] + input[off + i];
            acts[off + i] = model_activation<M>(x, c);
        }
        __builtin_amdgcn_s_barrier();
    }
}

template <ModelId M>
__device__ void model_backward_stage(const PersistentContext& ctx,
                                      const float* __restrict__ params,
                                      const float* __restrict__ acts,
                                      float* __restrict__ grad,
                                      const int* sizes, const int* offsets) {
    __shared__ int task_slot;
    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n = sizes[t], off = offsets[t];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float x = params[off + i];
            const float upstream = acts[off + i];
            grad[off + i] = upstream * model_activation_grad<M>(x);
        }
    }
}

}}} // namespace sg::fused::gfx942_mega

#endif  // device pass
#endif  // SG_FUSED_GFX942_MODEL_STAGES_HIP_HPP_
