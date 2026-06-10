#ifndef SG_FUSED_SM90_FUSED_VIT_MEGAKERNEL_CUH_
#define SG_FUSED_SM90_FUSED_VIT_MEGAKERNEL_CUH_
// ============================================================================
// csrc/fused/sm_90/fused_vit_megakernel.cuh — PHASE 2 of the TRUE L3 fused
// megakernel. ONE persistent kernel per training step runs the REAL Vision-
// Transformer forward+backward AND the optimizer math, separated only by
// in-kernel grid barriers — real model math, real optimizer math, ZERO
// intermediate kernel launches. The ViT counterpart of PHASE 1's
// fused_decoder_megakernel.cuh.
//
// This composes:
//   * the REAL ViT fwd/bwd stages (model_stage_vit.cuh — transcribed
//     line-for-line from the verified PyTorch oracle, asserted bit-identical to
//     autograd, and structurally mirrored on CPU),
//   * the existing persistent substrate (megakernel_common.cuh: task queue, the
//     hand-built sense-reversing GridBarrier),
//   * the existing REAL optimizer tail (opt_components.cuh::apply_optimizer<Opt>).
//
// STAGE / BARRIER LAYOUT (identical to the decoder — 5 phases, B0..B2 + the
// fused sync_reset; see the per-stage comments):
//   P0  each CTA zeroes its OWN grad-partial slice + loss slot.
//   --- grid barrier B0 ---
//   P1  BATCH-PARALLEL fwd+bwd: each CTA owns a FIXED contiguous batch slice (by
//       blockIdx.x), processes its samples ONE AT A TIME (CTA-cooperative),
//       accumulating each sample's weight-grad into the CTA's partial with a
//       single-owner-thread-per-element rule (no atomics → deterministic), and
//       sums its slice's NLL (fp32).
//   --- grid barrier B1 ---
//   P2  DETERMINISTIC cross-CTA reduce: sum partial[0..nCTA) in ASCENDING CTA
//       index into the global grad (no float atomics; order fixed → reuses the
//       work-steal queue to pick WHO reduces which tensor). Loss: fp64 ordered
//       sum → loss/B → a device float the host reads back (by CTA 0).
//   --- grid barrier B2 (sync_reset: also resets the queue for P3) ---
//   P3  the REAL apply_optimizer<Opt> tail consumes the reduced grad in place.
//
// DYNAMIC SHARED MEMORY (the ONE thing the decoder path did NOT need): ViT's
// per-sample VitSampleSmem is ≈ 183.67 KB (seq=17), which CANNOT be a static
// __shared__ (48 KB cap). So this kernel declares `extern __shared__` and the
// LAUNCHER must, all three of: (1) cudaFuncSetAttribute(kernel,
// cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(VitSampleSmem)); (2) pass
// dynamicSMemBytes=sizeof(VitSampleSmem) to cudaOccupancyMaxActiveBlocksPerMSM
// (so the occ≥1 hang-freedom check is honest); (3) pass it at <<<...>>>. Missing
// ANY of these makes the kernel silently fail to launch — see the launcher.
//
// HONESTY: no placeholder math anywhere on this path. fp32 compute is the
// correctness baseline; a bf16-compute follow-up would be a flag defaulting to
// THIS fp32 path (not yet wired — see the report / TODO in model_stage_vit.cuh).
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/opt_components.cuh"
#include "csrc/fused/sm_90/vit_layout.cuh"
#include "csrc/fused/sm_90/model_stage_vit.cuh"
#include "csrc/backends/cuda/sm_90/warp_specialize.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

namespace sg { namespace fused { namespace sm90 {

// Compile-time guard: the byte budget the launcher uses (sizeof(VitSampleSmem))
// MUST equal the literal the layout header documents + bounds-checks (227 KB cap).
static_assert(sizeof(VitSampleSmem) == (size_t)vit_layout_check::kVitSampleSmemBytes,
              "fused_vit_megakernel: sizeof(VitSampleSmem) != the documented "
              "kVitSampleSmemBytes in vit_layout.cuh — update both together.");

// Rebase a FusedOptState's per-element state pointers to a parameter-tensor slice
// at `off` within the flat [m|v|extra] layout. Per-TENSOR fields and all scalars
// pass through unchanged. (Identical to the decoder's rebase_state; redefined here
// under a vit_ name so this header stays self-contained — the decoder file is
// owned by a sibling agent and must not be included.)
template <OptId Opt>
__device__ __forceinline__ FusedOptState
vit_rebase_state(const FusedOptState& s, int64_t off) {
    FusedOptState t = s;  // scalars + per-tensor pointers copy as-is
    if (t.exp_avg)    t.exp_avg    += off;
    if (t.exp_avg_sq) t.exp_avg_sq += off;
    if (t.ema)        t.ema        += off;
    if (t.sam_dir)    t.sam_dir    += off;
    if (t.s_track)    t.s_track    += off;
    if (t.mu)         t.mu         += off;
    if (t.orth)       t.orth       += off;
    if (t.smart_grad) t.smart_grad += off;
    return t;
}

// The L3-REAL ViT megakernel needs the FLOAT patch path + a grad-partial
// workspace. Kept a SEPARATE kernel + launcher (not folded into the surrogate
// fused_megakernel) so the surrogate path is untouched.
//
// Workspace layout (one flat float buffer the host allocates + the kernel owns):
//   [0 .. nCTA*total)              : per-CTA grad partials (cta-major)
//   [nCTA*total .. nCTA*total+nCTA): per-CTA loss partials (NLL sum per slice)
//   [.. +1)                        : the reduced scalar loss (loss/B) the host reads
// total == kVitTotalElems == 418017.
struct ViTInputCtx {
    const float* patches;  // [B, kNPatch, kPatch] fp32 image patches
    const int*   targets;  // [B]                  int32 target ids in [0,kVocab)
    int          B;        // batch size
    float*       workspace; // grad partials + loss partials + reduced loss
    float*       loss_out;  // device float the kernel writes the mean loss into
};

// ── The persistent megakernel (L3-REAL). gridDim.x = #SMs (one CTA/SM), 256
//    threads/CTA. VitSampleSmem (~183.67 KB) lives in DYNAMIC smem (extern). ────
// sizes/offsets are NOT host-passed: per-tensor numel/offset live in the
// __constant__ tables kVitSizes/kVitOffsets (vit_layout.cuh), read directly by
// the reduce + optimizer phases.
template <OptId Opt>
__global__ void __launch_bounds__(256)
fused_vit_megakernel(PersistentContext ctx,
                     float* __restrict__ params,
                     ViTInputCtx in,
                     float* __restrict__ grad,        // reduced grad [total]
                     float lr, int step, FusedOptState st) {
    extern __shared__ char vit_smem_raw[];
    VitSampleSmem& sm = *reinterpret_cast<VitSampleSmem*>(vit_smem_raw);
    GridBarrier bar = ctx.barrier();
    const int cta = blockIdx.x;
    const int nCTA = (int)ctx.n_ctas;
    const int64_t total = kVitTotalElems;
    float* my_partial = in.workspace + (int64_t)cta * total;   // this CTA's dW
    float* loss_part = in.workspace + (int64_t)nCTA * total;   // [nCTA]

    // §3.4 register repartition (producer WG gives back, consumer WG claims).
    const int wg = threadIdx.x / 128;
    if (wg == 0) ::sg::sm90::wgs::warpgroup_reg_dealloc<32>();
    else         ::sg::sm90::wgs::warpgroup_reg_alloc<200>();

    // ── P0: zero this CTA's grad-partial slice + its loss slot. ───────────────
    for (int64_t i = threadIdx.x; i < total; i += blockDim.x) my_partial[i] = 0.0f;
    if (threadIdx.x == 0) loss_part[cta] = 0.0f;
    bar.sync();   // B0

    // ── P1: batch-parallel fwd+bwd. Fixed contiguous slice [b0,b1) for this CTA.
    VitWeights w = vit_bind(params);
    VitGrad g = vit_bind_grad(my_partial);
    const int B = in.B;
    const int base = B / nCTA, rem = B % nCTA;
    const int b0 = cta * base + (cta < rem ? cta : rem);
    const int cnt = base + (cta < rem ? 1 : 0);
    const int b1 = b0 + cnt;
    __shared__ int tgt_s;
    float nll_acc = 0.0f;              // fp32 slice accumulator (thread-0 holds it)
    const int patch_elems = vit::kNPatch * vit::kPatch;   // 16*49 = 784
    for (int b = b0; b < b1; ++b) {
        // Load this sample's float patches into smem (cooperative).
        const float* src = in.patches + (int64_t)b * patch_elems;
        for (int i = threadIdx.x; i < patch_elems; i += blockDim.x)
            sm.patch[i / vit::kPatch][i % vit::kPatch] = src[i];
        if (threadIdx.x == 0) tgt_s = in.targets[b];
        __syncthreads();
        float nll = vit_forward_sample(w, tgt_s, &sm);
        vit_backward_sample(w, g, tgt_s, B, &sm);
        if (threadIdx.x == 0) nll_acc += nll;   // fixed-order fp32 sum within slice
        __syncthreads();   // sample boundary: all grad writes done before reuse
    }
    if (threadIdx.x == 0) loss_part[cta] = nll_acc;
    bar.sync();   // B1: all CTA partials + loss slots complete

    // ── P2: deterministic cross-CTA reduce. Work-steal the param ELEMENT-RANGES
    //    (one task = one parameter tensor; sum its elements across CTAs in
    //    ascending CTA index). Summation ORDER (ascending cta) is fixed →
    //    deterministic regardless of which CTA grabs the task. ─────────────────
    {
        __shared__ int task_slot;
        TaskQueue q = ctx.queue();
        for (int t = q.next_block(&task_slot); t < kVitNumTensors;
             t = q.next_block(&task_slot)) {
            const int n = kVitSizes[t];
            const int64_t off = (int64_t)kVitOffsets[t];
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                float acc = 0.0f;
                for (int c = 0; c < nCTA; ++c)
                    acc += in.workspace[(int64_t)c * total + off + i];
                grad[off + i] = acc;
            }
        }
    }
    // Loss reduction (fp64 ordered) by CTA 0 only — the loss rel-tol is the
    // tightest gate; fp32 atomic-summing many terms can miss it.
    if (cta == 0 && threadIdx.x == 0) {
        double s = 0.0;
        for (int c = 0; c < nCTA; ++c) s += (double)loss_part[c];
        float mean = (float)(s / (double)B);
        *in.loss_out = mean;
    }
    bar.sync_reset(ctx.g_next_task);   // B2: reduced grad ready; reset queue for P3

    // ── P3: the REAL optimizer tail over the reduced grad (work-steal the 32
    //    tensors). apply_optimizer<Opt> is the canonical csrc/algorithms math. ──
    st.lr = lr;
    {
        __shared__ int task_slot;
        TaskQueue q = ctx.queue();
        for (int t = q.next_block(&task_slot); t < kVitNumTensors;
             t = q.next_block(&task_slot)) {
            const int n = kVitSizes[t];
            const int64_t off = (int64_t)kVitOffsets[t];
            const FusedOptState ts = vit_rebase_state<Opt>(st, off);
            float* __restrict__ p = params + off;
            const float* __restrict__ gg = grad + off;
            for (int i = threadIdx.x; i < n; i += blockDim.x)
                apply_optimizer<Opt>(p, gg, (int64_t)i, step, ts);
        }
    }
}

// ── Host launcher — one persistent CTA per SM, 256 threads/CTA (2 warp-groups).
//    Mirrors launch_fused_decoder_megakernel's hang-freedom contract (occupancy≥1
//    or refuse), PLUS the DYNAMIC-SMEM opt-in that the ViT footprint requires:
//      (1) cudaFuncSetAttribute(MaxDynamicSharedMemorySize, sizeof(VitSampleSmem))
//      (2) dynamicSMemBytes = sizeof(VitSampleSmem) in the occ≥1 query
//      (3) dynamicSMemBytes = sizeof(VitSampleSmem) at <<<...>>>
//    All three are mandatory; missing any one makes the launch silently fail. ───
template <OptId Opt>
cudaError_t launch_fused_vit_megakernel(
        PersistentContext ctx, float* params, ViTInputCtx in,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 0;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;

    const int dyn_smem = (int)sizeof(VitSampleSmem);   // ≈ 188080 B (183.67 KB)

    // (1) Opt in to >48 KB dynamic smem for THIS kernel. Without this the launch
    //     fails with cudaErrorInvalidValue (the static 48 KB default applies).
    err = cudaFuncSetAttribute(
        (const void*)&fused_vit_megakernel<Opt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
    if (err != cudaSuccess) return err;

    // (2) Occupancy with the REAL dynamic-smem request (hang-freedom is honest).
    int occ = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (const void*)&fused_vit_megakernel<Opt>, 256,
        /*dynamicSMemBytes=*/dyn_smem);
    if (err != cudaSuccess) return err;
    // At least one CTA per SM must be resident or the grid barrier can never be
    // satisfied. ~183.67 KB dynamic smem fits the sm_90 227 KB cap at occ=1 (the
    // persistent megakernel is one-CTA-per-SM by design); if it cannot place one
    // block/SM, REFUSE rather than hang.
    assert(occ >= 1 &&
           "fused_vit_megakernel: 0 blocks/SM — GridBarrier would hang. The ViT "
           "per-sample smem (~184KB dynamic) + regs exceed one-block-per-SM "
           "occupancy on this device; reduce footprint or fall back to eager.");
    if (occ < 1) return cudaErrorLaunchOutOfResources;

    const unsigned launch_ctas = (unsigned)n_sms;
    ctx.n_ctas = launch_ctas;

    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }

    // (3) Launch WITH the dynamic-smem byte count.
    dim3 grid(launch_ctas), block(256);
    fused_vit_megakernel<Opt><<<grid, block, dyn_smem, stream>>>(
        ctx, params, in, grad, lr, step, st);
    return cudaGetLastError();
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_FUSED_VIT_MEGAKERNEL_CUH_
