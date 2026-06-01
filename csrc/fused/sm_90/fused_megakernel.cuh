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

namespace sg { namespace fused { namespace sm90 {

namespace wgs = ::sg::sm90::wgs;

enum class FuseTier : int { L1 = 1, L3 = 3 };

// Per-warp-group register targets for the producer/consumer split (§3.4).
static constexpr int kProducerRegsF = 32;
static constexpr int kConsumerRegsF = 200;

// =========================================================================
//  Optimizer stage — the fused tail. Pulls tensors from the queue and applies
//  the REAL apply_optimizer<Opt> elementwise over this SM's resident state
//  slice (kept L2-warm via the %smid pin, §1.3).
// =========================================================================
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
        // Rebase this tensor's optimizer-state pointers to its slice.
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

// =========================================================================
//  The L3/L1 persistent megakernel. ONE launch composes the real model stages
//  (L3 only) and the real optimizer tail, separated by the hand-built grid
//  barrier (§1.4) whose last arriver resets the task-queue counter race-free.
//  Launch config (host): gridDim.x = #SMs, blockDim.x = 128 (warp 0 producer,
//  1–3 consumer) so the §3.4 warp-group split is well-formed.
// =========================================================================
template <ModelId M, OptId Opt, FuseTier Tier>
__global__ void __launch_bounds__(128)
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
        // ── Phase 1: real model forward ──────────────────────────────────
        model_forward_stage<M>(ctx, params, input, acts, sizes, offsets);
        bar.sync();
        if (threadIdx.x == 0 && blockIdx.x == 0) *ctx.g_next_task = 0;
        bar.sync();
        // ── Phase 2: real model backward ─────────────────────────────────
        model_backward_stage<M>(ctx, params, acts, grad, sizes, offsets);
        bar.sync();
        if (threadIdx.x == 0 && blockIdx.x == 0) *ctx.g_next_task = 0;
        bar.sync();
    }
    // ── Phase 3 (L3) / sole phase (L1): real fused optimizer tail ─────────
    st.lr = lr;
    fused_optimizer_stage<Opt>(ctx, params, grad, sizes, offsets, step, st);
}

// =========================================================================
//  Host launcher — one persistent CTA per SM, 128 threads/CTA.
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
    ctx.n_ctas = (unsigned)n_sms;
    dim3 grid((unsigned)n_sms), block(128);
    fused_megakernel<M, Opt, Tier><<<grid, block, 0, stream>>>(
        ctx, params, input, acts, grad, sizes, offsets, lr, step, st);
    return cudaGetLastError();
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_FUSED_MEGAKERNEL_CUH_
