#pragma once
// csrc/fused/gfx942/megakernel_demo.hip.hpp
// ─────────────────────────────────────────────────────────────────────────
// Stage 6 — representative gfx942 (CDNA3 / MI300X) L3 persistent megakernel.
//
// The AMD twin of csrc/fused/sm_90/megakernel_demo.cu: ONE templated L3
// megakernel that COMPOSES, in a single persistent launch,
//
//     (forward stage) ─grid-barrier─▶ (backward stage) ─grid-barrier─▶
//                                                  (fused optimizer step)
//
// using the §1.1–§1.4 scheduler substrate from megakernel_common_hip.hip.hpp.
//
// §1.13  SCHEDULING — this kernel is NOT warp-specialized. There is no Hopper
//   producer/consumer (TMA + WGMMA + setmaxnreg) analog on CDNA3. Instead the
//   forward stage uses **ping-pong / 4-wave-interleave**: the workgroup runs 4
//   wavefronts that take turns (one issues MFMA-class compute while the others
//   issue VMEM/DS), with sched_group_barrier hints (§2.11) pinning the
//   interleave so the matrix unit stays fed. Latency is hidden by OCCUPANCY,
//   not by a dedicated load warp-group. The scheduler/barrier substrate is
//   shape-identical to sm_90; only the in-stage scheduling differs.
//
// §1.7  SCOPE — SG2's SAM / bilevel outer loop stays OUTSIDE this per-step
//   megakernel (a separate host-scheduled launch); the demo instantiates the
//   elementwise-tail optimizers and documents that the SG2 meta-net tail is a
//   generator-emitted cell of its own.
//
// This is a HEADER so the device code is exercised by the free-standing amdgcn
// gate: scripts/amdgcn_check.sh --header csrc/fused/gfx942/megakernel_demo.hip.hpp
// => AMDGCN_OK. The host launch (hipLaunchKernelGGL) is 🟡 (hardware-gated on
// MI300X; no hipcc here).
// ─────────────────────────────────────────────────────────────────────────

#include "csrc/fused/megakernel_common_hip.hip.hpp"

namespace sg { namespace fused { namespace gfx942_mega {

using ::sg::fused::gfx942::PersistentContext;
using ::sg::fused::gfx942::TaskQueue;
using ::sg::fused::gfx942::GridBarrier;
using ::sg::fused::gfx942::cu_id;

enum class Model    : int { TransformerDecoder = 0, ViT = 1, Mamba3 = 2 };
enum class Optimizer: int { AdamW = 0, Lion = 1, Muon = 2, SuperGrok15 = 3 };

// Bytes staged into LDS for the representative forward tile (buffer_load→LDS).
static constexpr int kStageBytes = 4096;

// -------------------------------------------------------------------------
//  Optimizer tail — fused per-element update (CDNA3). Specialized per
//  Optimizer at compile time; reads/writes this CU's resident state slice
//  (kept warm in the CU's L2 partition via the CU-pin, §1.3).
// -------------------------------------------------------------------------
template <Optimizer Opt>
__device__ __forceinline__ float opt_update(float p, float g, float& m,
                                             float& v, float lr, int step) {
    if (Opt == Optimizer::AdamW) {
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
        m = b1 * m + (1.f - b1) * g;
        v = b2 * v + (1.f - b2) * g * g;
        float bc1 = 1.f - __builtin_powf(b1, (float)step);
        float bc2 = 1.f - __builtin_powf(b2, (float)step);
        float mhat = m / bc1, vhat = v / bc2;
        return p - lr * (mhat / (__builtin_sqrtf(vhat) + eps) + wd * p);
    } else if (Opt == Optimizer::Lion) {
        const float b1 = 0.9f, b2 = 0.99f, wd = 0.01f;
        float upd = __builtin_copysignf(1.0f, b1 * m + (1.f - b1) * g);
        m = b2 * m + (1.f - b2) * g;
        return p - lr * (upd + wd * p);
    } else if (Opt == Optimizer::Muon) {
        const float mu = 0.95f;
        m = mu * m + g;
        return p - lr * (m + 0.1f * v);
    } else {  // SuperGrok15 — elementwise meta-gated tail (representative).
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        m = b1 * m + (1.f - b1) * g;
        v = b2 * v + (1.f - b2) * g * g;
        float gate = 1.f / (1.f + __builtin_expf(-(m * m)));
        return p - lr * gate * (m / (__builtin_sqrtf(v) + eps));
    }
}

// -------------------------------------------------------------------------
//  Forward stage (representative, ping-pong / 4-wave-interleave §1.13).
//  Each persistent workgroup (one per CU) pulls parameter tensors from the
//  task queue and runs a grid-stride pass. The 4 wavefronts interleave via the
//  sched_group_barrier-style alternation: a slab is staged into LDS, then all
//  waves compute over the tensor. (The real kernel issues buffer_load→LDS and
//  MFMA; here a representative copy + nonlinearity stands in.)
// -------------------------------------------------------------------------
template <Model M>
__device__ void forward_stage(const PersistentContext& ctx,
                              const float* __restrict__ params,
                              const float* __restrict__ input,
                              float* __restrict__ acts,
                              const int* sizes, const int* offsets) {
    __shared__ int task_slot;
    __shared__ float stage[kStageBytes / sizeof(float)];

    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n   = sizes[t];
        const int off = offsets[t];

        // Stage a slab into LDS (representative buffer_load→LDS, §2.7), then a
        // workgroup barrier so all 4 waves see it before the interleaved compute.
        for (int i = threadIdx.x;
             i < (int)(kStageBytes / sizeof(float)) && i < n; i += blockDim.x) {
            stage[i] = input[off + i];
        }
        __builtin_amdgcn_s_barrier();

        float local = 0.f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float x = params[off + i] + (i < (int)(kStageBytes / sizeof(float))
                                             ? stage[i] : 0.f);
            if (M == Model::Mamba3)      x = x - __builtin_tanhf(x) * x;
            else if (M == Model::ViT)    x = x * 0.5f * (1.f + __builtin_tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
            else                         x = __builtin_fmaxf(x, 0.f);
            local += x * x;
            // §1.13 interleave hint: alternate MFMA-class compute (mask 0x008)
            // with DS reads (mask 0x020) so the matrix unit isn't starved.
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
        }
        if (threadIdx.x < (unsigned)n && n > 0)
            acts[off + (threadIdx.x % (unsigned)n)] = local;
        __builtin_amdgcn_s_barrier();
    }
}

// -------------------------------------------------------------------------
//  Backward stage (representative). Reuses the queue + grid-stride shape.
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
            float g = 2.f * p * (1.f + a) * 1e-3f;
            if (M == Model::Mamba3) {
                float tp = __builtin_tanhf(p);
                g *= 1.0f - 0.5f * tp * tp;
            }
            grad[off + i] = g;
        }
    }
}

// -------------------------------------------------------------------------
//  Optimizer stage (fused L3 tail).
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
//  The L3 persistent megakernel (gfx942). ONE launch, three stages separated
//  by the hand-built grid barrier (§1.4). The last barrier-arriver resets the
//  task-queue counter between phases (race-free: all workgroups are quiesced at
//  the barrier). Launch config (host): grid = #CUs (one persistent workgroup
//  per CU, §1.3), block = 256 (4 wavefronts of 64 for the §1.13 interleave).
// =========================================================================
template <Model M, Optimizer Opt>
__global__ void
SG_MK_FWGS(256, 256)
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

    forward_stage<M>(ctx, params, input, acts, sizes, offsets);
    bar.sync();
    if (threadIdx.x == 0 && blockIdx.x == 0) *ctx.g_next_task = 0;
    bar.sync();

    backward_stage<M>(ctx, params, acts, grad, sizes, offsets);
    bar.sync();
    if (threadIdx.x == 0 && blockIdx.x == 0) *ctx.g_next_task = 0;
    bar.sync();

    optimizer_stage<Opt>(ctx, params, grad, m, v, sizes, offsets, lr, step);
}

// Force the representative instantiations so the free-standing gate type-checks
// the full template expansion (the AMD twins of the sm_90 wired cells).
template __global__ void l3_megakernel<Model::Mamba3, Optimizer::AdamW>(
    PersistentContext, float*, const float*, float*, float*, float*, float*,
    const int*, const int*, float, int);
template __global__ void l3_megakernel<Model::TransformerDecoder, Optimizer::Lion>(
    PersistentContext, float*, const float*, float*, float*, float*, float*,
    const int*, const int*, float, int);
template __global__ void l3_megakernel<Model::ViT, Optimizer::SuperGrok15>(
    PersistentContext, float*, const float*, float*, float*, float*, float*,
    const int*, const int*, float, int);

}}} // namespace sg::fused::gfx942_mega
