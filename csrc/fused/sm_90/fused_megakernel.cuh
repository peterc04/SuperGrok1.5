#ifndef SG_FUSED_SM90_FUSED_MEGAKERNEL_CUH_
#define SG_FUSED_SM90_FUSED_MEGAKERNEL_CUH_
// ============================================================================
// csrc/fused/sm_90/fused_megakernel.cuh — REAL component-composition L3/L1
// persistent megakernel substrate (Phase 3 Stage 5).
//
// This is the composition seam. It pulls the REAL optimizer device-function
// component (opt_components.cuh → csrc/algorithms/<opt>.h) and the REAL model
// stage device functions (model_stages.cuh) and composes the solver-chosen
// (model, optimizer) into ONE persistent __global__ over the shared substrate
// (task queue, %smid SM-pin, hand-built GridBarrier, warp-spec hand-off).
//
// It REPLACES megakernel_demo.cu's toy `opt_update<Opt>` (4 optimizers + an
// AdamW fallback for the other 7). Every cell that includes this file gets the
// real, distinct per-optimizer math via apply_optimizer<OptId> — no fallback,
// no template-of-one-optimizer-from-another, no toy substitute.
//
// TIERING (matches grokking_optimizers/megakernel.py solver):
//   FuseTier::L3 — fuse forward + backward + optimizer in one launch.
//   FuseTier::L1 — fuse ONLY the optimizer tail (model fwd/bwd are the
//                  framework's own / the per-op CUTLASS launches). 46 of the 99
//                  cells are L1 on the GPU arches (register-pressure bound).
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/backends/cuda/sm_90/warp_specialize.cuh"
#include "csrc/fused/sm_90/opt_components.cuh"
#include "csrc/fused/sm_90/model_stages.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

namespace sg { namespace fused { namespace sm90 {

namespace wgs = ::sg::sm90::wgs;

enum class FuseTier : int { L1 = 1, L3 = 3 };

// Per-warp-group register targets for the producer/consumer split (§3.4).
// On Hopper setmaxnreg is warp-group granular (128 threads / 4 warps). The
// fused kernel launches 256 threads = TWO warp-groups so the split is well
// formed: warp-group 0 is the PRODUCER (staging / queue / row-context loads,
// few registers) and warp-group 1 is the CONSUMER (the heavy WGMMA-class
// compute + optimizer tail). Partitioning lets the consumer claim a large
// register file WITHOUT the producer's allocation counting against the L3
// footprint, so no single warp-group needs the full fused register budget.
// SG_TUNED_PROD_REGS / SG_TUNED_CONS_REGS — the setmaxnreg targets for the
// producer / consumer warp-groups, exposed as autotuner dims. The SAME macro
// tokens drive the warp-specialized tile pipeline (csrc/backends/cuda/sm_90/
// tile_pipeline.cuh); a single -DSG_TUNED_{PROD,CONS}_REGS=N flag therefore
// tunes whichever sm_90 register-split path the unit being built compiles.
// Defaults equal the prior literals (32 / 200) so an untuned / no-JSON build is
// byte-identical (verified via nvcc -E). NOTE: tile_pipeline.cuh's #ifndef
// defaults differ (40 / 232); that is harmless ONLY because the two headers are
// never co-included in one TU (fused_megakernel reaches setmaxnreg via
// warp_specialize.cuh, not via wgmma.cuh→tile_pipeline.cuh — the latter is
// pulled solely by decoder_tc_selftest.cu). If a future TU includes both, the
// divergent #ifndef defaults make include order decide the value — then they
// must be reconciled to one default (see report).
#ifndef SG_TUNED_PROD_REGS
#define SG_TUNED_PROD_REGS 32
#endif
#ifndef SG_TUNED_CONS_REGS
#define SG_TUNED_CONS_REGS 200
#endif
static constexpr int kProducerRegsF = SG_TUNED_PROD_REGS;
static constexpr int kConsumerRegsF = SG_TUNED_CONS_REGS;
// Warp-group size on Hopper (threads). 256-thread block => 2 warp-groups.
static constexpr int kWarpGroupThreads = 128;
// SG_TUNED_MEGA_BLOCK (the per-CTA thread count) is defined in
// megakernel_common.cuh (#include'd above) so the fused megakernel and the
// model drivers share one tunable; default 256 → byte-identical untuned build.
// NEEDS-PARITY before a non-default winner ships (2-warp-group split + shared
// buffer sizing assumptions; see the macro's definition site).
static constexpr int kFusedBlockThreads = SG_TUNED_MEGA_BLOCK;

// =========================================================================
//  Optimizer stage — the fused tail. Pulls tensors from the queue and applies
//  the REAL apply_optimizer<Opt> elementwise over this SM's resident state
//  slice (kept L2-warm via the %smid pin, §1.3).
// =========================================================================
// #6 — per-Opt pointer rebase via `if constexpr`. Rather than unconditionally
// bumping all 8 optimizer-state pointers per task (8 predicated adds + the
// chance the allocator keeps every field live), only the pointers the CHOSEN
// optimizer actually dereferences are rebased to the tensor slice; the rest are
// compiled out. `ts` starts as a copy of the const& `st` (the scalar
// hyperparams must be present), but the dead-pointer rebases vanish, lowering
// the local/register footprint of the tail.
template <OptId Opt>
__device__ __forceinline__ FusedOptState rebase_state(const FusedOptState& st,
                                                       int64_t off) {
    FusedOptState ts = st;
    if constexpr (Opt == OptId::AdamW || Opt == OptId::Lion ||
                  Opt == OptId::Grokfast || Opt == OptId::GrokAdamW ||
                  Opt == OptId::LookSAM || Opt == OptId::Prodigy ||
                  Opt == OptId::NeuralGrok || Opt == OptId::SuperGrok11 ||
                  Opt == OptId::SuperGrok15 || Opt == OptId::SuperGrok2) {
        if (ts.exp_avg)    ts.exp_avg    += off;   // m (all Adam-family tails)
    }
    if constexpr (Opt != OptId::Lion && Opt != OptId::Muon) {
        if (ts.exp_avg_sq) ts.exp_avg_sq += off;   // v (everyone but Lion/Muon)
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
        // #2 int64: off is the tensor's element offset into the global param
        // blob and can exceed 2^31; rebase the slice pointers in 64-bit so the
        // base address is correct past 2 Gi elements. The within-tensor index
        // (base/i below) is bounded by n (an int) but apply_optimizer takes an
        // int64_t idx so the addition never truncates. Behavior-preserving.
        const int64_t off = (int64_t)offsets[t];
        // #6: rebase only the pointers Opt uses to this tensor's slice.
        const FusedOptState ts = rebase_state<Opt>(st, off);
        float* __restrict__ p = params + off;
        const float* __restrict__ g = grad + off;
        // #1 — float4 vectorization of the per-element tail. Each thread strides
        // over groups of 4 CONSECUTIVE indices and calls the canonical per-
        // element optimizer fn 4× on [4j .. 4j+3]; because the four indices are
        // contiguous and the buffers are __restrict__, the loads/stores of
        // params/grad/m/v coalesce into 128-bit float4 transactions. The
        // canonical algorithm math (csrc/algorithms/<opt>.h via apply_optimizer)
        // is CALLED, never re-inlined — only the loop's access pattern changes.
        const int n4 = n >> 2;                  // # of full float4 groups
        for (int j = threadIdx.x; j < n4; j += blockDim.x) {
            const int64_t base = (int64_t)j << 2;   // #2 int64 within-tensor idx
            apply_optimizer<Opt>(p, g, base + 0, step, ts);
            apply_optimizer<Opt>(p, g, base + 1, step, ts);
            apply_optimizer<Opt>(p, g, base + 2, step, ts);
            apply_optimizer<Opt>(p, g, base + 3, step, ts);
        }
        // Scalar tail for the remainder (n not a multiple of 4).
        for (int i = (n4 << 2) + (int)threadIdx.x; i < n; i += blockDim.x) {
            apply_optimizer<Opt>(p, g, (int64_t)i, step, ts);
        }
    }
}

// =========================================================================
//  Stage 5 — TRUE cross-phase fusion (bwd→opt), SMEM-resident grad tile.
//
//  HONEST SCOPE / what is realized vs deferred:
//   * REALIZED: the backward grad of a tensor and the optimizer update of the
//     SAME tensor are fused into ONE queue-pulled body. The grad is computed
//     into a per-CTA __shared__ tile and the optimizer reads it from SMEM — the
//     grad NEVER round-trips through the global `grad` buffer, and the grid
//     barrier that used to separate bwd from opt is ELIMINATED (3 phases / 2
//     barriers → 2 phases / 1 barrier per L3 step). This is correct because
//     both the activation backward (grad = acts*act'(param)) and every
//     apply_optimizer<Opt> are strictly ELEMENT-LOCAL for a given tensor: no
//     cross-element and no cross-tensor dependency couples bwd to opt, so no
//     grid-wide sync is needed between them for the same tensor, and the SAME
//     CTA owns the tensor end-to-end within one queue pull (no work-stealing
//     hand-off can split a tensor's bwd from its opt).
//   * STILL GLOBAL (deferred, with reason): the forward `acts` buffer. fwd and
//     bwd are split by a grid barrier AND the task queue is reset between them,
//     so under work-stealing (§1.2) a different CTA may own a tensor in bwd than
//     owned it in fwd. A CTA-local SMEM acts tile therefore CANNOT be carried
//     across the fwd→bwd barrier safely. Making acts SMEM-resident would require
//     a stable tensor→CTA partition (abandoning work-stealing) or DSMEM cluster
//     sharing — out of scope for a no-silicon, behavior-preserving change. So
//     acts stays global; that round-trip is documented, not hidden.
//   * The apply math is still CALLED via apply_optimizer<Opt> (drift-guard safe);
//     only the grad SOURCE pointer changes from global to the SMEM tile, and the
//     params/state pointers are rebased per chunk so the canonical algorithm
//     addresses the correct global state. 🟡 perf (no silicon to measure).
//
//  The grad value itself is BIT-IDENTICAL to the previous two-phase path:
//  grad = acts[gi] * model_activation_grad<M>(params[gi]) — same expression,
//  same operand order; only its storage class (SMEM vs global) changed.
// =========================================================================
// SG_TUNED_GRAD_TILE — floats per SMEM grad tile (the cross-phase bwd→opt
// fusion buffer). Default 1024 (4 KB → byte-identical untuned build). NEEDS-
// PARITY before shipping a non-default winner: it sizes the __shared__
// smem_grad[] array, so a swept value changes the static SMEM footprint and the
// chunk loop bounds; correctness (no overrun, matching the global-grad path)
// must be re-checked on the H100 parity gate, not assumed by the autotuner.
#ifndef SG_TUNED_GRAD_TILE
#define SG_TUNED_GRAD_TILE 1024
#endif
static constexpr int kGradTile = SG_TUNED_GRAD_TILE;   // floats / SMEM grad tile

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
        const int64_t off = (int64_t)offsets[t];   // #2 int64 base
        // Tile the tensor; each tile's grad lives in SMEM, consumed in place by
        // the optimizer (no global grad write/read for this tensor).
        for (int c0 = 0; c0 < n; c0 += kGradTile) {
            const int chunk = (n - c0 < kGradTile) ? (n - c0) : kGradTile;
            const int64_t chunk_base = off + c0;
            // Backward: compute grad of this chunk into SMEM (element-local).
            for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
                const int64_t gi = chunk_base + i;
                smem_grad[i] = acts[gi] * model_activation_grad<M>(params[gi]);
            }
            __syncthreads();
            // Optimizer: read grad from the SMEM tile. Rebase params + state to
            // this chunk so apply_optimizer's chunk-local idx addresses the
            // correct global params/m/v while grad comes from SMEM.
            const FusedOptState ts = rebase_state<Opt>(st, chunk_base);
            float* __restrict__ p = params + chunk_base;
            const float* __restrict__ g = smem_grad;   // SMEM-resident grad
            const int n4 = chunk >> 2;
            for (int j = threadIdx.x; j < n4; j += blockDim.x) {
                const int64_t base = (int64_t)j << 2;
                apply_optimizer<Opt>(p, g, base + 0, step, ts);
                apply_optimizer<Opt>(p, g, base + 1, step, ts);
                apply_optimizer<Opt>(p, g, base + 2, step, ts);
                apply_optimizer<Opt>(p, g, base + 3, step, ts);
            }
            for (int i = (n4 << 2) + (int)threadIdx.x; i < chunk;
                 i += blockDim.x) {
                apply_optimizer<Opt>(p, g, (int64_t)i, step, ts);
            }
            __syncthreads();   // tile reuse: all opt reads done before next fill
        }
    }
}

// =========================================================================
//  The L3/L1 persistent megakernel. ONE launch composes the real model stages
//  (L3 only) and the real optimizer tail, separated by the hand-built grid
//  barrier (§1.4) whose last arriver resets the task-queue counter race-free.
//  Launch config (host): gridDim.x = #SMs, blockDim.x = 256 = TWO Hopper
//  warp-groups (§3.4). Warp-group 0 (threads 0..127) is the producer and
//  deallocs its register file down to kProducerRegsF; warp-group 1 (threads
//  128..255) is the consumer and allocs up to kConsumerRegsF. setmaxnreg is
//  warp-group-uniform, so the call is issued by all 128 lanes of each group.
//
//  Behavior-preserving: setmaxnreg only repartitions the SM register file
//  between the two warp-groups; it does NOT change any computed value. All 256
//  threads still execute the same grid-stride element-local stages (whose
//  shared reduction / staging buffers are sized for 256), so __syncthreads is
//  well-formed and the results are identical to the single-warp-group launch.
// =========================================================================
template <ModelId M, OptId Opt, FuseTier Tier>
__global__ void __launch_bounds__(kFusedBlockThreads)
fused_megakernel(PersistentContext ctx,
                 float* __restrict__ params,
                 const float* __restrict__ input,
                 float* __restrict__ acts,
                 float* __restrict__ grad,
                 const int* __restrict__ sizes,
                 const int* __restrict__ offsets,
                 float lr, int step, FusedOptState st) {
    GridBarrier bar = ctx.barrier();

    // §3.4 warp-group register repartition: producer WG (0) gives registers
    // back; consumer WG (1) claims the large file. Uniform within each WG.
    const int warp_group = threadIdx.x / kWarpGroupThreads;
    if (warp_group == 0) {
        wgs::warpgroup_reg_dealloc<kProducerRegsF>();
    } else {
        wgs::warpgroup_reg_alloc<kConsumerRegsF>();
    }

    if constexpr (Tier == FuseTier::L3) {
        // ── Phase 1: real model forward (writes acts to global; see the Stage 5
        //    note above for why acts stays global across the fwd→bwd barrier). ─
        // #4: the task-queue reset for the NEXT phase is folded into the grid
        // barrier's last-arriver critical section (sync_reset). The reset lands
        // before the generation bump, so every CTA crossing into the fused
        // backward+optimizer phase sees a zeroed counter.
        model_forward_stage<M>(ctx, params, input, acts, sizes, offsets);
        bar.sync_reset(ctx.g_next_task);
        // ── Phase 2 (Stage 5 TRUE FUSION): backward + optimizer in ONE pass,
        //    grad held in a SMEM tile (no global grad round-trip), with the
        //    bwd→opt grid barrier eliminated (see fused_backward_optimizer_stage).
        st.lr = lr;
        fused_backward_optimizer_stage<M, Opt>(ctx, params, acts, sizes,
                                               offsets, step, st);
        // grad (param) unused on the L3 path now — the grad is SMEM-resident.
        (void)grad;
    } else {
        // ── L1: model fwd/bwd are the framework's own launches; this kernel is
        //    the fused optimizer tail only (reads grad from global). ───────────
        st.lr = lr;
        fused_optimizer_stage<Opt>(ctx, params, grad, sizes, offsets, step, st);
    }
}

// =========================================================================
//  Host launcher — one persistent CTA per SM, 256 threads/CTA (2 warp-groups).
//
//  HANG-FREEDOM (Stage 1 fix) — the L3 kernel rendezvouses ALL launched CTAs at
//  GridBarrier::sync_reset, which spins until exactly ctx.n_ctas CTAs arrive.
//  With an ordinary <<<>>> launch that has NO co-residency guarantee: if the
//  kernel's per-SM occupancy is < 1 block (register/smem pressure — the L3 cells
//  run ~168 regs / ~65 KB smem), some of the gridDim.x==#SMs CTAs never become
//  resident, never arrive, and the resident CTAs spin FOREVER. (cudaLaunch
//  Cooperative would error -1 in that case, but the plain launch just hangs.)
//
//  MECHANISM CHOSEN: occupancy-capped grid + co-residency assert (mechanism (a)
//  in the task). We query cudaOccupancyMaxActiveBlocksPerMultiprocessor for THIS
//  exact kernel instantiation at the launch block size and cap gridDim.x to
//  (occ_blocks_per_sm * #SMs) — but, because every persistent CTA must hold a
//  full warm optimizer-state slice and we pin ONE per SM (§1.3), we launch
//  exactly min(#SMs, occ * #SMs) == #SMs ONLY WHEN occ >= 1. If the kernel
//  cannot place even one block per SM (occ == 0) we REFUSE to launch (return
//  cudaErrorLaunchOutOfResources) instead of hanging. The assert documents the
//  invariant in debug builds; the runtime check enforces it in release builds.
//  We deliberately keep at most 1 CTA/SM resident (the §1.3 pin) so the barrier
//  set size equals the number of co-resident CTAs — never more than can fit.
//
//  This is preferred over cudaLaunchCooperativeKernel here because (a) it needs
//  no kernel attribute / driver cooperative-launch capability probe, (b) it
//  preserves the existing reusable hand-built GridBarrier (§1.4) which is the
//  whole point of NOT requiring a cooperative grid, and (c) it is fully compile-
//  checkable without silicon. The barrier counters (g_arrived / g_generation)
//  and the task counter (g_next_task) are device-zeroed before each launch so a
//  stale generation/arrival from a prior step can never desync the rendezvous.
// =========================================================================
template <ModelId M, OptId Opt, FuseTier Tier>
cudaError_t launch_fused_megakernel(
        PersistentContext ctx,
        float* params, const float* input, float* acts, float* grad,
        const int* sizes, const int* offsets,
        float lr, int step, FusedOptState st, cudaStream_t stream) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 0;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;

    // Co-residency guarantee: how many blocks of THIS kernel actually fit on one
    // SM at the launch block size? If < 1, the grid barrier could never complete
    // (a non-resident CTA never arrives) — so we must NOT launch a gridDim that
    // relies on a CTA that cannot be placed.
    int occ_blocks_per_sm = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_blocks_per_sm,
        (const void*)&fused_megakernel<M, Opt, Tier>,
        kFusedBlockThreads, /*dynamicSMemBytes=*/0);
    if (err != cudaSuccess) return err;
    // Hang-freedom invariant: at least one block per SM must be resident, or the
    // GridBarrier rendezvous of n_ctas CTAs can never be satisfied.
    assert(occ_blocks_per_sm >= 1 &&
           "fused_megakernel: 0 blocks/SM occupancy — GridBarrier would hang; "
           "reduce register/smem footprint or this cell must run L1 (no grid "
           "barrier).");
    if (occ_blocks_per_sm < 1) {
        // Release-build enforcement: refuse rather than hang.
        return cudaErrorLaunchOutOfResources;
    }

    // §1.3: pin exactly ONE persistent CTA per SM (keeps each SM's optimizer-
    // state slice L2-warm). occ>=1 guarantees that single CTA per SM is co-
    // resident, so all n_sms CTAs rendezvous at the barrier — no hang.
    const unsigned launch_ctas = (unsigned)n_sms;
    ctx.n_ctas = launch_ctas;

    // Device-zero the barrier + task counters before EACH launch so no stale
    // generation/arrival from a prior step can desync the rendezvous (the
    // GridBarrier reuse argument in megakernel_common.cuh assumes a zeroed start).
    if (ctx.g_arrived != nullptr) {
        err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream);
        if (err != cudaSuccess) return err;
    }
    if (ctx.g_generation != nullptr) {
        err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream);
        if (err != cudaSuccess) return err;
    }
    if (ctx.g_next_task != nullptr) {
        err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream);
        if (err != cudaSuccess) return err;
    }

    // 256 threads = two Hopper warp-groups for the §3.4 producer/consumer
    // register split (kProducerRegsF / kConsumerRegsF).
    dim3 grid(launch_ctas), block(kFusedBlockThreads);
    fused_megakernel<M, Opt, Tier><<<grid, block, 0, stream>>>(
        ctx, params, input, acts, grad, sizes, offsets, lr, step, st);
    return cudaGetLastError();
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_FUSED_MEGAKERNEL_CUH_
