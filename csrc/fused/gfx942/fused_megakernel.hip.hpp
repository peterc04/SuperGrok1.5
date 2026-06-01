#ifndef SG_FUSED_GFX942_FUSED_MEGAKERNEL_HIP_HPP_
#define SG_FUSED_GFX942_FUSED_MEGAKERNEL_HIP_HPP_
// ============================================================================
// csrc/fused/gfx942/fused_megakernel.hip.hpp — REAL component-composition
// L3/L1 persistent megakernel for gfx942 (CDNA3 / MI300X). AMD twin of
// csrc/fused/sm_90/fused_megakernel.cuh.
//
// Composes the REAL optimizer device-function component (opt_components.hip.hpp
// → apply_optimizer<OptId>, all 11 real, no fallback) with the REAL model stage
// component (model_stages.hip.hpp) over the shared gfx942 persistent substrate
// (megakernel_common_hip.hip.hpp: task queue, CU-pin, hand-built GridBarrier).
// It REPLACES the deleted toy demo opt_update (4 opts + AdamW fallback) with the
// real all-11 apply. §1.13 ping-pong / 4-wave-interleave scheduling (no Hopper
// warp-spec analog on CDNA3).
//
// Device pass: __AMDGCN__ gate (scripts/amdgcn_check.sh) or __HIPCC__.
// Host launch (hipLaunchKernelGGL) is 🟡 (MI300X-gated; no hipcc here).
// ============================================================================

#include "csrc/fused/megakernel_common_hip.hip.hpp"
#include "csrc/fused/gfx942/opt_components.hip.hpp"
#include "csrc/fused/gfx942/model_stages.hip.hpp"

#if defined(__AMDGCN__) || defined(__HIPCC__) || defined(GROK_HIP_DEVICE)

namespace sg { namespace fused { namespace gfx942_mega {

using ::sg::fused::gfx942::PersistentContext;
using ::sg::fused::gfx942::TaskQueue;
using ::sg::fused::gfx942::GridBarrier;

enum class FuseTier : int { L1 = 1, L3 = 3 };

// Optimizer stage — the fused tail. Pulls tensors from the queue and applies the
// REAL apply_optimizer<Opt> elementwise over this CU's resident state slice.
template <OptId Opt>
__device__ void fused_optimizer_stage(const PersistentContext& ctx,
                                       float* __restrict__ params,
                                       const float* __restrict__ grad,
                                       const int* sizes, const int* offsets,
                                       int step, FusedOptState st) {
    __shared__ int task_slot;
    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n = sizes[t], off = offsets[t];
        FusedOptState ts = st;
        if (ts.exp_avg)    ts.exp_avg    += off;
        if (ts.exp_avg_sq) ts.exp_avg_sq += off;
        if (ts.ema)        ts.ema        += off;
        if (ts.s_track)    ts.s_track    += off;
        if (ts.mu)         ts.mu         += off;
        if (ts.sam_dir)    ts.sam_dir    += off;
        if (ts.orth)       ts.orth       += off;
        if (ts.smart_grad) ts.smart_grad += off;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            apply_optimizer<Opt>(params + off, grad + off, i, step, ts);
        }
    }
}

// The L3/L1 persistent megakernel. ONE launch composes the real model stages
// (L3 only) and the real optimizer tail, separated by the hand-built grid
// barrier whose last arriver resets the task-queue counter race-free.
template <ModelId M, OptId Opt, FuseTier Tier>
__global__ void
SG_MK_FWGS(256, 256)
fused_megakernel(PersistentContext ctx,
                 float* __restrict__ params,
                 const float* __restrict__ input,
                 float* __restrict__ acts,
                 float* __restrict__ grad,
                 const int* __restrict__ sizes,
                 const int* __restrict__ offsets,
                 float lr, int step, FusedOptState st) {
    GridBarrier bar = ctx.barrier();

    if constexpr (Tier == FuseTier::L3) {
        model_forward_stage<M>(ctx, params, input, acts, sizes, offsets);
        bar.sync();
        if (threadIdx.x == 0 && blockIdx.x == 0) *ctx.g_next_task = 0;
        bar.sync();
        model_backward_stage<M>(ctx, params, acts, grad, sizes, offsets);
        bar.sync();
        if (threadIdx.x == 0 && blockIdx.x == 0) *ctx.g_next_task = 0;
        bar.sync();
    }
    st.lr = lr;
    fused_optimizer_stage<Opt>(ctx, params, grad, sizes, offsets, step, st);
}

}}} // namespace sg::fused::gfx942_mega

#endif  // device pass
#endif  // SG_FUSED_GFX942_FUSED_MEGAKERNEL_HIP_HPP_
