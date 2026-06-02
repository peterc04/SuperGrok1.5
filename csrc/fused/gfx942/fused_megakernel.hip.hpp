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
// #6 — per-Opt pointer rebase via `if constexpr` (AMD twin of the sm_90 form):
// only the pointers the chosen optimizer dereferences are rebased to the tensor
// slice; the dead rebases are compiled out, lowering the tail's VGPR/local
// footprint. `ts` copies the const& `st` (scalar hyperparams must be present).
template <OptId Opt>
__device__ __forceinline__ FusedOptState rebase_state(const FusedOptState& st,
                                                       int off) {
    FusedOptState ts = st;
    if constexpr (Opt == OptId::AdamW || Opt == OptId::Lion ||
                  Opt == OptId::Grokfast || Opt == OptId::GrokAdamW ||
                  Opt == OptId::LookSAM || Opt == OptId::Prodigy ||
                  Opt == OptId::NeuralGrok || Opt == OptId::SuperGrok11 ||
                  Opt == OptId::SuperGrok15 || Opt == OptId::SuperGrok2) {
        if (ts.exp_avg)    ts.exp_avg    += off;
    }
    if constexpr (Opt != OptId::Lion && Opt != OptId::Muon) {
        if (ts.exp_avg_sq) ts.exp_avg_sq += off;
    }
    if constexpr (Opt == OptId::Grokfast || Opt == OptId::GrokAdamW) {
        if (ts.ema)        ts.ema        += off;
    }
    if constexpr (Opt == OptId::Prodigy) {
        if (ts.s_track)    ts.s_track    += off;
    }
    if constexpr (Opt == OptId::SuperGrok11 || Opt == OptId::SuperGrok15) {
        if (ts.mu)         ts.mu         += off;
    }
    if constexpr (Opt == OptId::LookSAM) {
        if (ts.sam_dir)    ts.sam_dir    += off;
    }
    if constexpr (Opt == OptId::Muon) {
        if (ts.orth)       ts.orth       += off;
    }
    if constexpr (Opt == OptId::SuperGrok2) {
        if (ts.smart_grad) ts.smart_grad += off;
    }
    return ts;
}

template <OptId Opt>
__device__ void fused_optimizer_stage(const PersistentContext& ctx,
                                       float* __restrict__ params,
                                       const float* __restrict__ grad,
                                       const int* sizes, const int* offsets,
                                       int step, const FusedOptState& st) {
    __shared__ int task_slot;
    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n = sizes[t], off = offsets[t];
        const FusedOptState ts = rebase_state<Opt>(st, off);
        float* __restrict__ p = params + off;
        const float* __restrict__ g = grad + off;
        // #1 — float4 vectorization: each thread strides over groups of 4
        // CONSECUTIVE indices and calls the (transcribed canonical) per-element
        // apply 4× on [4j..4j+3]; the contiguous __restrict__ accesses coalesce
        // into 128-bit (dwordx4) global transactions. Math is CALLED, never
        // re-inlined; only the access pattern changes. Scalar remainder tail.
        const int n4 = n >> 2;
        for (int j = threadIdx.x; j < n4; j += blockDim.x) {
            const int base = j << 2;
            apply_optimizer<Opt>(p, g, base + 0, step, ts);
            apply_optimizer<Opt>(p, g, base + 1, step, ts);
            apply_optimizer<Opt>(p, g, base + 2, step, ts);
            apply_optimizer<Opt>(p, g, base + 3, step, ts);
        }
        for (int i = (n4 << 2) + (int)threadIdx.x; i < n; i += blockDim.x) {
            apply_optimizer<Opt>(p, g, i, step, ts);
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
        // #4: fold the next-phase task-queue reset into the grid barrier's
        // last-arriver critical section (sync_reset) — 4 grid barriers → 2 per
        // L3 step, identical ordering/visibility (reset before generation bump).
        model_forward_stage<M>(ctx, params, input, acts, sizes, offsets);
        bar.sync_reset(ctx.g_next_task);
        model_backward_stage<M>(ctx, params, acts, grad, sizes, offsets);
        bar.sync_reset(ctx.g_next_task);
    }
    st.lr = lr;
    fused_optimizer_stage<Opt>(ctx, params, grad, sizes, offsets, step, st);
}

}}} // namespace sg::fused::gfx942_mega

#endif  // device pass
#endif  // SG_FUSED_GFX942_FUSED_MEGAKERNEL_HIP_HPP_
