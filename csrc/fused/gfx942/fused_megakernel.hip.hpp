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
                                                       long long off) {
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
        const int n = sizes[t];
        // #2 int64: off (tensor element offset into the global blob) can exceed
        // 2^31; rebase the slice pointers in 64-bit. Behavior-preserving.
        const long long off = (long long)offsets[t];
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
            const long long base = (long long)j << 2;   // #2 int64 within-tensor
            apply_optimizer<Opt>(p, g, base + 0, step, ts);
            apply_optimizer<Opt>(p, g, base + 1, step, ts);
            apply_optimizer<Opt>(p, g, base + 2, step, ts);
            apply_optimizer<Opt>(p, g, base + 3, step, ts);
        }
        for (int i = (n4 << 2) + (int)threadIdx.x; i < n; i += blockDim.x) {
            apply_optimizer<Opt>(p, g, (long long)i, step, ts);
        }
    }
}

// Stage 5 — TRUE cross-phase fusion (bwd→opt), LDS-resident grad tile (AMD twin
// of the sm_90 fused_backward_optimizer_stage; see that header for the full
// honest-scope note). REALIZED: the backward grad and the optimizer update of
// the SAME tensor are fused into one queue-pulled body; grad is computed into a
// per-workgroup __shared__ (LDS) tile and the optimizer reads it from LDS — no
// global grad round-trip, and the bwd→opt grid barrier is eliminated (2 barriers
// → 1 per L3 step). Correct because bwd (grad = acts*act'(param)) and every
// apply_optimizer<Opt> are strictly element-local for a tensor, and the same
// workgroup owns the tensor end-to-end in one queue pull (no work-steal split).
// DEFERRED (documented): `acts` stays global — fwd→bwd is separated by a grid
// barrier + a queue reset, so under work-stealing a different workgroup may own
// the tensor in bwd; a workgroup-local LDS acts tile cannot cross that barrier
// safely. Math is still CALLED via apply_optimizer (drift-guard safe); only the
// grad source pointer changes to LDS. Grad value is bit-identical. 🟡 perf.
// SG_TUNED_GRAD_TILE — floats per LDS grad tile (AMD twin; sizes __shared__
// smem_grad[]). Default 1024 → byte-identical untuned build. NEEDS-PARITY for a
// non-default winner (LDS footprint + chunk bounds). Not registered on gfx942.
#ifndef SG_TUNED_GRAD_TILE
#define SG_TUNED_GRAD_TILE 1024
#endif
static constexpr int kGradTile = SG_TUNED_GRAD_TILE;   // floats / LDS grad tile

template <ModelId M, OptId Opt>
__device__ void fused_backward_optimizer_stage(
        const PersistentContext& ctx,
        float* __restrict__ params,
        const float* __restrict__ acts,
        const int* sizes, const int* offsets,
        int step, const FusedOptState& st) {
    __shared__ int task_slot;
    __shared__ float smem_grad[kGradTile];
    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n = sizes[t];
        const long long off = (long long)offsets[t];   // #2 int64 base
        for (int c0 = 0; c0 < n; c0 += kGradTile) {
            const int chunk = (n - c0 < kGradTile) ? (n - c0) : kGradTile;
            const long long chunk_base = off + c0;
            for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
                const long long gi = chunk_base + i;
                smem_grad[i] = acts[gi] * model_activation_grad<M>(params[gi]);
            }
            __builtin_amdgcn_s_barrier();
            const FusedOptState ts = rebase_state<Opt>(st, chunk_base);
            float* __restrict__ p = params + chunk_base;
            const float* __restrict__ g = smem_grad;   // LDS-resident grad
            const int n4 = chunk >> 2;
            for (int j = threadIdx.x; j < n4; j += blockDim.x) {
                const long long base = (long long)j << 2;
                apply_optimizer<Opt>(p, g, base + 0, step, ts);
                apply_optimizer<Opt>(p, g, base + 1, step, ts);
                apply_optimizer<Opt>(p, g, base + 2, step, ts);
                apply_optimizer<Opt>(p, g, base + 3, step, ts);
            }
            for (int i = (n4 << 2) + (int)threadIdx.x; i < chunk;
                 i += blockDim.x) {
                apply_optimizer<Opt>(p, g, (long long)i, step, ts);
            }
            __builtin_amdgcn_s_barrier();   // tile reuse fence
        }
    }
}

// The L3/L1 persistent megakernel. ONE launch composes the real model stages
// (L3 only) and the real optimizer tail, separated by the hand-built grid
// barrier whose last arriver resets the task-queue counter race-free.
template <ModelId M, OptId Opt, FuseTier Tier>
__global__ void
SG_MK_FWGS(SG_TUNED_MEGA_BLOCK, SG_TUNED_MEGA_BLOCK)
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
        // last-arriver critical section (sync_reset). Stage 5: bwd+opt fused, so
        // the L3 step is fwd → sync_reset → (bwd+opt) = 1 grid barrier.
        model_forward_stage<M>(ctx, params, input, acts, sizes, offsets);
        bar.sync_reset(ctx.g_next_task);
        st.lr = lr;
        fused_backward_optimizer_stage<M, Opt>(ctx, params, acts, sizes,
                                               offsets, step, st);
        (void)grad;   // grad is LDS-resident on L3; no global round-trip
    } else {
        st.lr = lr;
        fused_optimizer_stage<Opt>(ctx, params, grad, sizes, offsets, step, st);
    }
}

}}} // namespace sg::fused::gfx942_mega

#endif  // device pass

// =========================================================================
//  HOST launcher (hipcc host pass only; 🟡 MI300X — not the bare amdgcn gate).
//  AMD twin of the sm_90 launch_fused_megakernel hang-freedom fix (Stage 1).
//
//  Same hazard: the L3 kernel rendezvouses ALL launched workgroups at
//  GridBarrier::sync_reset, which spins until exactly ctx.n_groups workgroups
//  arrive. hipLaunchKernelGGL gives no co-residency guarantee, so if the kernel
//  cannot place even one workgroup per CU (VGPR/LDS pressure) the resident
//  workgroups spin forever. MECHANISM: hipOccupancyMaxActiveBlocksPerMultiproc
//  caps the grid so one workgroup per CU is guaranteed resident; if occupancy is
//  0 we REFUSE to launch (hipErrorOutOfMemory) rather than hang. The barrier +
//  task counters are device-zeroed before each launch (GridBarrier reuse assumes
//  a zeroed start). Cells call this instead of inlining their own launch.
// =========================================================================
#if defined(__HIPCC__) && !defined(__AMDGCN__)
#include <hip/hip_runtime.h>
#include <cassert>

namespace sg { namespace fused { namespace gfx942_mega {

template <ModelId M, OptId Opt, FuseTier Tier>
hipError_t launch_fused_megakernel(
        PersistentContext ctx,
        float* params, const float* input, float* acts, float* grad,
        const int* sizes, const int* offsets,
        float lr, int step, FusedOptState st, hipStream_t stream) {
    int dev = 0;
    hipError_t err = hipGetDevice(&dev);
    if (err != hipSuccess) return err;
    int n_cus = 0;
    err = hipDeviceGetAttribute(&n_cus, hipDeviceAttributeMultiprocessorCount,
                                dev);
    if (err != hipSuccess) return err;

    // Co-residency guarantee: at least one workgroup of THIS kernel must fit on
    // one CU, or the grid barrier of n_groups workgroups can never complete.
    int occ_blocks_per_cu = 0;
    err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_blocks_per_cu,
        reinterpret_cast<const void*>(
            &fused_megakernel<M, Opt, Tier>),
        /*blockSize=*/SG_TUNED_MEGA_BLOCK, /*dynamicSMemBytes=*/0);
    if (err != hipSuccess) return err;
    assert(occ_blocks_per_cu >= 1 &&
           "fused_megakernel: 0 workgroups/CU occupancy — GridBarrier would "
           "hang; reduce VGPR/LDS footprint or run this cell L1.");
    if (occ_blocks_per_cu < 1) return hipErrorOutOfMemory;

    // §1.3: pin exactly ONE persistent workgroup per CU; occ>=1 guarantees it is
    // co-resident, so all n_cus workgroups rendezvous at the barrier — no hang.
    const unsigned launch_groups = (unsigned)n_cus;
    ctx.n_groups = launch_groups;

    if (ctx.g_arrived != nullptr) {
        err = hipMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream);
        if (err != hipSuccess) return err;
    }
    if (ctx.g_generation != nullptr) {
        err = hipMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream);
        if (err != hipSuccess) return err;
    }
    if (ctx.g_next_task != nullptr) {
        err = hipMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream);
        if (err != hipSuccess) return err;
    }

    st.lr = lr;
    dim3 grid(launch_groups), block(SG_TUNED_MEGA_BLOCK);
    hipLaunchKernelGGL((fused_megakernel<M, Opt, Tier>), grid, block, 0, stream,
                       ctx, params, input, acts, grad, sizes, offsets,
                       lr, step, st);
    return hipGetLastError();
}

}}} // namespace sg::fused::gfx942_mega
#endif  // hipcc host pass

#endif  // SG_FUSED_GFX942_FUSED_MEGAKERNEL_HIP_HPP_
