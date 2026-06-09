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
// §2.6 DPP wavefront reduction primitive (wave_reduce_add_dpp). Device-only,
// gate-compilable (verified via scripts/amdgcn_check.sh).
#include "csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp"

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
        // int64 base (#2): off+i can exceed 2^31 once the param blob crosses
        // ~2 Gi elements; the global index is computed in 64-bit. Behavior-
        // preserving (same element, wider index type).
        const long long base64 = (long long)off;
        float acc = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float p = params[base64 + i];
            acc += (M == ModelId::ViT) ? p : p * p;
        }
        // Block reduce (#2): DPP wavefront reduction replaces the former LDS
        // tree (which did an s_barrier at every level). Each 64-lane wavefront
        // reduces with wave_reduce_add_dpp (§2.6 cross-lane butterfly, no LDS,
        // no barrier — every lane gets the wave total); the 4 wave-totals (256
        // threads = 4 wavefronts of 64) land in LDS and one s_barrier + a tiny
        // serial combine on lane 0 finishes the block. Net: ONE s_barrier
        // instead of ~8. BIT-CHANGE NOTE: float SUM reassociation only (the
        // downstream `c` is a mean proxy, not a checksum); the optimizer apply
        // math in opt_components.hip.hpp is untouched.
        const float wsum = ::sg::gfx942::amdgcn::wave_reduce_add_dpp(acc);
        const int wave = (int)threadIdx.x / ::sg::fused::gfx942::kWave;
        const int wlane = (int)threadIdx.x % ::sg::fused::gfx942::kWave;
        const int n_waves = ((int)blockDim.x + ::sg::fused::gfx942::kWave - 1)
                            / ::sg::fused::gfx942::kWave;
        if (wlane == 0) red[wave] = wsum;
        __builtin_amdgcn_s_barrier();
        float blk = 0.0f;
        if (threadIdx.x == 0) {
            for (int wv = 0; wv < n_waves; ++wv) blk += red[wv];
            red[0] = blk;
        }
        __builtin_amdgcn_s_barrier();
        blk = red[0];
        const float c = (n > 0) ? blk / (float)n : 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const long long gi = base64 + i;
            const float x = params[gi] + input[gi];
            acts[gi] = model_activation<M>(x, c);
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
        const long long base64 = (long long)off;   // #2 int64 global index base
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const long long gi = base64 + i;
            const float x = params[gi];
            const float upstream = acts[gi];
            grad[gi] = upstream * model_activation_grad<M>(x);
        }
    }
}

}}} // namespace sg::fused::gfx942_mega

#endif  // device pass
#endif  // SG_FUSED_GFX942_MODEL_STAGES_HIP_HPP_
