// csrc/fused/sm_90/megakernel_demo.cu
// ─────────────────────────────────────────────────────────────────────────
// Stage 6 — representative sm_90 L3 persistent megakernel (the "demo" TU).
//
// This is ONE parameterized/templated L3 megakernel that COMPOSES, in a single
// persistent kernel launch:
//
//     (forward stage) ─grid-barrier─▶ (backward stage) ─grid-barrier─▶
//                                                  (fused optimizer step)
//
// for a given (Model, Optimizer) chosen at instantiation time. It is the
// structural template the Python generator (grokking_optimizers/megakernel_
// codegen.py) emits per solver-chosen cell — you do NOT hand-write 99 kernels;
// you instantiate THIS for each cell. The forward/backward stages here are
// lightweight REPRESENTATIVE bodies (grid-stride compute over the parameter
// tensor with the correct barrier/hand-off structure); the point is that the
// L3 fusion SHAPE — task queue, SM-pinning, hand-built grid barrier, and the
// fused-optimizer tail — all compiles and is structurally correct, using the
// §3.4 warp-specialization primitives for the producer/consumer split.
//
// §1.7  SCOPE: the SG2 SAM / bilevel (CSA-HCA outer loop) stays OUTSIDE this
//   per-step megakernel — it is a *second* forward/backward (the SAM ascent and
//   the bilevel adjoint) that the host schedules as its own launch(es) around
//   this one. The megakernel here is the INNER per-step fwd+bwd+opt; the demo
//   instantiates the elementwise-tail optimizers, and documents that the SG2
//   meta-net tail is emitted by the generator into its own cell.
//
// Compile gate: scripts/compile_to_object.sh csrc/fused/sm_90/megakernel_demo.cu
// ─────────────────────────────────────────────────────────────────────────

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/backends/cuda/sm_90/warp_specialize.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace sg { namespace fused { namespace sm90 {

// =========================================================================
//  Compile-time model / optimizer selectors. The generator instantiates the
//  megakernel template with the (Model, Optimizer) enum pair the solver chose
//  for a cell. Using enums (not strings) keeps everything a compile-time
//  constant so the producer/consumer register split (setmaxnreg, which needs an
//  immediate) and the per-optimizer math both specialize with zero overhead.
// =========================================================================
enum class Model    : int { TransformerDecoder = 0, ViT = 1, Mamba3 = 2 };
enum class Optimizer: int { AdamW = 0, Lion = 1, Muon = 2, SuperGrok15 = 3 };

namespace wgs = ::sg::sm90::wgs;

// Per-warp-group register targets for the producer/consumer split (§3.4).
// The producer (TMA/load) warp-group deallocates down; the consumer (compute)
// warp-group allocates up. Immediates required by setmaxnreg.
static constexpr int kProducerRegs = 32;
static constexpr int kConsumerRegs = 240;

// Bytes a representative forward tile would stage through shared for the
// load→compute mbarrier hand-off. Kept small + fixed so the demo is a clean
// structural proof of the hand-off, not a tuned tile.
static constexpr int kStageBytes = 4096;

// -------------------------------------------------------------------------
//  Optimizer tail — the fused per-element update applied after backward. This
//  is the L3 "+ optimizer" that distinguishes a megakernel from a plain
//  fwd/bwd fusion. Specialized per Optimizer at compile time. `m`/`v` are this
//  parameter's optimizer-state slice (kept L2-warm on this SM via §1.3).
// -------------------------------------------------------------------------
template <Optimizer Opt>
__device__ __forceinline__ float opt_update(float p, float g, float& m,
                                             float& v, float lr, int step) {
    if constexpr (Opt == Optimizer::AdamW) {
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
        m = b1 * m + (1.f - b1) * g;
        v = b2 * v + (1.f - b2) * g * g;
        float bc1 = 1.f - __powf(b1, (float)step);
        float bc2 = 1.f - __powf(b2, (float)step);
        float mhat = m / bc1, vhat = v / bc2;
        return p - lr * (mhat / (sqrtf(vhat) + eps) + wd * p);
    } else if constexpr (Opt == Optimizer::Lion) {
        const float b1 = 0.9f, b2 = 0.99f, wd = 0.01f;
        float upd = copysignf(1.0f, b1 * m + (1.f - b1) * g);
        m = b2 * m + (1.f - b2) * g;
        return p - lr * (upd + wd * p);
    } else if constexpr (Opt == Optimizer::Muon) {
        // The orthogonalized direction is produced by the (host-side) Newton–
        // Schulz pipeline; the in-kernel tail is the momentum + scaled step.
        const float mu = 0.95f;
        m = mu * m + g;
        return p - lr * (m + 0.1f * v);
    } else {  // SuperGrok15 — elementwise meta-gated tail (representative).
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        m = b1 * m + (1.f - b1) * g;
        v = b2 * v + (1.f - b2) * g * g;
        float gate = 1.f / (1.f + __expf(-(m * m)));   // grok meta-gate stand-in
        return p - lr * gate * (m / (sqrtf(v) + eps));
    }
}

// -------------------------------------------------------------------------
//  Forward stage (representative). One persistent CTA per SM pulls parameter
//  tensors from the task queue and runs a grid-stride pass producing an
//  activation-norm proxy into `acts`. The producer/consumer warp-group split
//  is exercised: warp-group 0 is the producer (elects a leader, declares the
//  staged bytes on an mbarrier, deallocs regs), warp-group 1+ is the consumer
//  (allocs regs, waits the mbarrier, computes). This is the §3.4 load→compute
//  hand-off shape the real fwd tile uses, with a trivial compute body.
// -------------------------------------------------------------------------
template <Model M>
__device__ void forward_stage(const PersistentContext& ctx,
                              const float* __restrict__ params,
                              const float* __restrict__ input,
                              float* __restrict__ acts,
                              const int* sizes, const int* offsets) {
    __shared__ unsigned long long mbar_storage;     // mbarrier state (8B)
    __shared__ int task_slot;                       // task broadcast slot
    __shared__ float stage[kStageBytes / sizeof(float)];

    wgs::Mbarrier mbar(&mbar_storage);
    const int warp = threadIdx.x / 32;
    const bool producer = (warp == 0);

    // One-time mbarrier init by a single thread, then a block sync.
    if (threadIdx.x == 0) mbar.init(/*expected_arrivals=*/1);
    __syncthreads();

    TaskQueue q = ctx.queue();
    unsigned phase = 0;
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n   = sizes[t];
        const int off = offsets[t];

        // Producer warp-group: lower its register footprint, stage a slab into
        // shared, and signal the consumer via the mbarrier transaction barrier.
        if (producer) {
            wgs::warpgroup_reg_dealloc<kProducerRegs>();
            if (wgs::elect_one_sync()) {
                mbar.arrive_expect_tx(kStageBytes);
            }
            // Representative "TMA": cooperatively copy a slab to shared. (The
            // real kernel issues cp.async.bulk; here a plain copy stands in.)
            for (int i = threadIdx.x;
                 i < (int)(kStageBytes / sizeof(float)) && i < n; i += 32) {
                stage[i] = input[off + i];
            }
            wgs::fence_async_proxy();     // exactly one async-proxy fence (§1.14)
        } else {
            // Consumer warp-group: take up registers, wait the hand-off, then
            // run the grid-stride forward proxy over the whole tensor.
            wgs::warpgroup_reg_alloc<kConsumerRegs>();
            mbar.wait(phase & 1u);
            float local = 0.f;
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                float x = params[off + i] + (i < (int)(kStageBytes / sizeof(float))
                                                 ? stage[i] : 0.f);
                // A tiny model-flavored nonlinearity so the stage isn't a no-op.
                if constexpr (M == Model::Mamba3)      x = x - tanhf(x) * x;
                else if constexpr (M == Model::ViT)    x = x * 0.5f * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                else                                   x = fmaxf(x, 0.f);
                local += x * x;
            }
            if (threadIdx.x < n) acts[off + (threadIdx.x % (n > 0 ? n : 1))] = local;
        }
        __syncthreads();
        ++phase;
    }
}

// -------------------------------------------------------------------------
//  Backward stage (representative). Same task-queue + grid-stride shape; here
//  the consumer computes a gradient proxy from params+acts into `grad`. No new
//  primitives — it reuses the queue and (in the real kernel) the same staged
//  load path. Kept lean to keep register pressure inside the L3 budget.
// -------------------------------------------------------------------------
template <Model M>
__device__ void backward_stage(const PersistentContext& ctx,
                               const float* __restrict__ params,
                               const float* __restrict__ acts,
                               float* __restrict__ grad,
                               const int* sizes, const int* offsets) {
    __shared__ int task_slot;
    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n   = sizes[t];
        const int off = offsets[t];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float p = params[off + i];
            float a = acts[off + i];
            // Representative adjoint: d/dp of the forward proxy's local energy.
            float g = 2.f * p * (1.f + a) * 1e-3f;
            if constexpr (M == Model::Mamba3) g *= 1.0f - 0.5f * tanhf(p) * tanhf(p);
            grad[off + i] = g;
        }
    }
}

// -------------------------------------------------------------------------
//  Optimizer stage (the fused L3 tail). Pulls tensors from the queue and
//  applies opt_update<Opt> elementwise, reading/writing this SM's resident
//  optimizer-state slice (m, v) — kept warm in L2 via the SM-pin (§1.3).
// -------------------------------------------------------------------------
template <Optimizer Opt>
__device__ void optimizer_stage(const PersistentContext& ctx,
                               float* __restrict__ params,
                               const float* __restrict__ grad,
                               float* __restrict__ m, float* __restrict__ v,
                               const int* sizes, const int* offsets,
                               float lr, int step) {
    __shared__ int task_slot;
    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n   = sizes[t];
        const int off = offsets[t];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            int idx = off + i;
            float mm = m[idx], vv = v[idx];
            params[idx] = opt_update<Opt>(params[idx], grad[idx], mm, vv, lr, step);
            m[idx] = mm; v[idx] = vv;
        }
    }
}

// =========================================================================
//  The L3 persistent megakernel. ONE launch composes the three stages,
//  separated by the hand-built grid barrier (§1.4). The task queue counter is
//  reset between phases by the host? No — it must be reset *inside* the kernel
//  at a point where all CTAs are quiesced, which is exactly at the grid
//  barrier. The last arriver of each barrier resets the queue for the next
//  phase. We thread that through a tiny helper so the reset is race-free.
//
//  Launch config (host): gridDim.x = #SMs (one persistent CTA per SM, §1.3),
//  blockDim.x = 128 (4 warps: warp 0 producer, warps 1–3 consumer), so the
//  §3.4 warp-group split is well-formed.
// =========================================================================
template <Model M, Optimizer Opt>
__global__ void __launch_bounds__(128)
l3_megakernel(PersistentContext ctx,
              float* __restrict__ params,
              const float* __restrict__ input,
              float* __restrict__ acts,
              float* __restrict__ grad,
              float* __restrict__ m, float* __restrict__ v,
              const int* __restrict__ sizes,
              const int* __restrict__ offsets,
              float lr, int step) {
    GridBarrier bar = ctx.barrier();

    // ── Phase 1: forward ────────────────────────────────────────────────
    forward_stage<M>(ctx, params, input, acts, sizes, offsets);
    // Reset the task queue for the next phase from the last-arriver of the
    // barrier so every CTA has drained the forward queue first.
    if (blockIdx.x == 0 && threadIdx.x == 0) { /* reset handled post-barrier */ }
    bar.sync();
    if (threadIdx.x == 0 && blockIdx.x == 0) *ctx.g_next_task = 0;
    bar.sync();

    // ── Phase 2: backward ───────────────────────────────────────────────
    backward_stage<M>(ctx, params, acts, grad, sizes, offsets);
    bar.sync();
    if (threadIdx.x == 0 && blockIdx.x == 0) *ctx.g_next_task = 0;
    bar.sync();

    // ── Phase 3: fused optimizer step ───────────────────────────────────
    optimizer_stage<Opt>(ctx, params, grad, m, v, sizes, offsets, lr, step);
}

// =========================================================================
//  Host launcher. Sets one persistent CTA per SM, 128 threads/CTA, and passes
//  the zero-initialized PersistentContext scratch. The generator emits one
//  launcher symbol per instantiated cell; the demo provides a uniform entry
//  the dispatch can call by (model, optimizer).
// =========================================================================
template <Model M, Optimizer Opt>
cudaError_t launch_l3_megakernel(
        PersistentContext ctx,
        float* params, const float* input, float* acts, float* grad,
        float* m, float* v, const int* sizes, const int* offsets,
        float lr, int step, cudaStream_t stream) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 0;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
    ctx.n_ctas = (unsigned)n_sms;   // one persistent CTA per SM (§1.3)

    dim3 grid((unsigned)n_sms), block(128);
    l3_megakernel<M, Opt><<<grid, block, 0, stream>>>(
        ctx, params, input, acts, grad, m, v, sizes, offsets, lr, step);
    return cudaGetLastError();
}

// =========================================================================
//  The three WIRED cells (instantiations + non-templated launchers) live ONLY
//  in this canonical demo TU. A GENERATED cell .cu (megakernel_codegen.py)
//  defines SG_MEGAKERNEL_DEMO_TEMPLATE_ONLY before including this file so it
//  pulls in JUST the template machinery above (comdat-folded across TUs) and
//  defines its own unique launcher — never a second copy of these three
//  non-templated symbols (which would be a duplicate-symbol link error).
// =========================================================================
#ifndef SG_MEGAKERNEL_DEMO_TEMPLATE_ONLY

// =========================================================================
//  Explicit instantiations for the representative cells (§1.5/§1.6). These are
//  the (model, optimizer) pairs WIRED into fused_step. The generator emits the
//  remaining cells by re-instantiating this template at the solver's tier.
//
//    (Mamba3, AdamW)         — an L3-fitting cell (mamba3 fwd/bwd + cheap AdamW
//                              tail comfortably inside the sm_90 reg/smem budget)
//    (TransformerDecoder, Lion) — L3-fitting, different model + optimizer tail
//    (ViT, SuperGrok15)      — L3-fitting, exercises the meta-gated tail
//
//  SG2's heavy CSA/HCA+PEER meta-net tail is NOT instantiated here (§1.7): it is
//  a falls-back cell on tight arches and its SAM/bilevel outer loop is a
//  separate launch; the generator emits it into its own cell.
// =========================================================================
template cudaError_t launch_l3_megakernel<Model::Mamba3, Optimizer::AdamW>(
    PersistentContext, float*, const float*, float*, float*, float*, float*,
    const int*, const int*, float, int, cudaStream_t);
template cudaError_t launch_l3_megakernel<Model::TransformerDecoder, Optimizer::Lion>(
    PersistentContext, float*, const float*, float*, float*, float*, float*,
    const int*, const int*, float, int, cudaStream_t);
template cudaError_t launch_l3_megakernel<Model::ViT, Optimizer::SuperGrok15>(
    PersistentContext, float*, const float*, float*, float*, float*, float*,
    const int*, const int*, float, int, cudaStream_t);

// -------------------------------------------------------------------------
//  Non-templated host entry points the dispatch links against (one per wired
//  cell). They translate the (model, optimizer) selection the host made into
//  the right template instantiation. This is the symbol fused_step calls.
// -------------------------------------------------------------------------
cudaError_t mega_mamba3_adamw(
        PersistentContext ctx, float* params, const float* input, float* acts,
        float* grad, float* m, float* v, const int* sizes, const int* offsets,
        float lr, int step, cudaStream_t stream) {
    return launch_l3_megakernel<Model::Mamba3, Optimizer::AdamW>(
        ctx, params, input, acts, grad, m, v, sizes, offsets, lr, step, stream);
}
cudaError_t mega_decoder_lion(
        PersistentContext ctx, float* params, const float* input, float* acts,
        float* grad, float* m, float* v, const int* sizes, const int* offsets,
        float lr, int step, cudaStream_t stream) {
    return launch_l3_megakernel<Model::TransformerDecoder, Optimizer::Lion>(
        ctx, params, input, acts, grad, m, v, sizes, offsets, lr, step, stream);
}
cudaError_t mega_vit_supergrok15(
        PersistentContext ctx, float* params, const float* input, float* acts,
        float* grad, float* m, float* v, const int* sizes, const int* offsets,
        float lr, int step, cudaStream_t stream) {
    return launch_l3_megakernel<Model::ViT, Optimizer::SuperGrok15>(
        ctx, params, input, acts, grad, m, v, sizes, offsets, lr, step, stream);
}

#endif  // !SG_MEGAKERNEL_DEMO_TEMPLATE_ONLY

}}} // namespace sg::fused::sm90
