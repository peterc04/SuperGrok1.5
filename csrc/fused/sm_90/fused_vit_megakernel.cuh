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
// STAGED-optimizer in-kernel precompute stages (Prodigy d-reduction). Pulls in the
// canonical prodigy.h reduction math + the deterministic block reductions; the vit
// TC megakernel's Prodigy branch runs an INLINE FIXED-PARTITION (r,s) reduce + an EMA
// owner-update between B2 and P3 (the ViT twin of the decoder's P2.6 block; the
// work-steal phaseA/B helpers in opt_stages_precompute.cuh are the older form).
#include "csrc/fused/sm_90/opt_stages_precompute.cuh"
#include "csrc/fused/sm_90/vit_layout.cuh"
#include "csrc/fused/sm_90/model_stage_vit.cuh"
#include "csrc/backends/cuda/sm_90/warp_specialize.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

// ============================================================================
//  SG_TUNED_GEMM_IMPL — the per-cell GEMM-engine selector (mirrors the decoder
//  twin fused_decoder_megakernel.cuh). The scalar fp32 owner-computes path
//  (model_stage_vit.cuh) and the bf16 wgmma tensor-core path
//  (model_stage_vit_tc.cuh) are compiled ALONGSIDE each other; the autotuner
//  injects -DSG_TUNED_GEMM_IMPL=<token> per-TU to pick one. Absent the macro,
//  the SCALAR path compiles verbatim (CONTRACT rule 3) — the shipped default
//  that all of test_vit_megakernel.py's L3-REAL gates exercise; adding this seam
//  must leave those numbers BIT-IDENTICAL (the macros below are only #defined
//  here; the scalar kernel body does not reference them).
//
//  Tokens are integers (the preprocessor cannot compare strings):
//      SG_GEMM_IMPL_SCALAR = 0   (default; the verbatim fp32 path below)
//      SG_GEMM_IMPL_WGMMA  = 1   (the Fork-B bf16 tensor-core ViT cell)
//
//  The wgmma token pulls in the validated GEMM engine (model_stage_vit_tc.cuh)
//  and the phase-restructured fwd+bwd+AdamW persistent kernel at the BOTTOM of
//  this file (fused_vit_megakernel_tc / launch_fused_vit_megakernel_tc). The
//  scalar default path is UNCHANGED and its gates stay bit-identical; this is a
//  PARALLEL kernel, not an edit of the scalar one.
// ============================================================================
#ifndef SG_GEMM_IMPL_SCALAR
#define SG_GEMM_IMPL_SCALAR 0
#endif
#ifndef SG_GEMM_IMPL_WGMMA
#define SG_GEMM_IMPL_WGMMA  1
#endif
#ifndef SG_TUNED_GEMM_IMPL
#define SG_TUNED_GEMM_IMPL SG_GEMM_IMPL_SCALAR
#endif
#if (SG_TUNED_GEMM_IMPL != SG_GEMM_IMPL_SCALAR) && \
    (SG_TUNED_GEMM_IMPL != SG_GEMM_IMPL_WGMMA)
#error "SG_TUNED_GEMM_IMPL must be SG_GEMM_IMPL_SCALAR (0) or SG_GEMM_IMPL_WGMMA (1)"
#endif
#if (SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA)
#include "csrc/fused/sm_90/model_stage_vit_tc.cuh"
// SuperGrok2 FULL CSA/HCA/PEER/GRU meta-net stages (composed as the optimizer
// phase) — the ViT twin of the decoder's SG2 composition. Pulls in sg2_meta_stages
// + the in-kernel segmented sort (STAGE -1). The SG2 weight bundle is read from HBM
// (it does not fit smem alongside ViT's dynamic VitSampleSmem), so the composed
// path instantiates sg2_meta_stages with WeightsT==SG2Weights + BuildSort=true.
#include "csrc/fused/sm_90/opt_stage_supergrok2.cuh"
#endif

// ============================================================================
//  SG_VIT_SCALAR_MEGAKERNEL — compile-gate for the LEGACY fp32 SCALAR ViT
//  megakernel (fused_vit_megakernel<Opt> + launch_fused_vit_megakernel<Opt> below,
//  which drive the per-sample VitSampleSmem path in model_stage_vit.cuh). The ViT
//  twin of the decoder's SG_DEC_SCALAR_MEGAKERNEL gate (commit 555c0bc/79d3840).
//
//  WHY A GATE (the d-scaling decouple): VitSampleSmem is sized [kSeq×kD]/[kSeq×kDff]
//  arrays, so its DYNAMIC smem GROWS with SG_VIT_D and at the d-scaled bench width
//  (d=1024) it is ≈ 1.4 MB — far over the sm_90 227 KB per-block dynamic-smem cap, so
//  cudaFuncSetAttribute(MaxDynamicSharedMemorySize) would REFUSE it and the scalar
//  kernel could never launch. The TC engine (fused_vit_megakernel_tc, below) uses a
//  small d-INDEPENDENT static VitTcSmem and builds/launches at every d. Gating the
//  dead scalar kernel OFF (-DSG_VIT_SCALAR_MEGAKERNEL=0) lets the WHOLE extension
//  compile at the bench width (the TC path is what's measured). DEFAULT 1 (ON) → the
//  d=128 production build is byte-for-byte unchanged: the scalar fp32 fallback stays
//  wired and the 33/33 wiring gate stays green. The flag changes NOTHING the TC path
//  touches.
#ifndef SG_VIT_SCALAR_MEGAKERNEL
#define SG_VIT_SCALAR_MEGAKERNEL 1
#endif

namespace sg { namespace fused { namespace sm90 {

#if SG_VIT_SCALAR_MEGAKERNEL
// Compile-time guard: the byte budget the launcher uses (sizeof(VitSampleSmem))
// MUST equal the literal the layout header documents + bounds-checks (227 KB cap).
static_assert(sizeof(VitSampleSmem) == (size_t)vit_layout_check::kVitSampleSmemBytes,
              "fused_vit_megakernel: sizeof(VitSampleSmem) != the documented "
              "kVitSampleSmemBytes in vit_layout.cuh — update both together.");
#endif  // SG_VIT_SCALAR_MEGAKERNEL

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
    if (t.param_init) t.param_init += off;   // Prodigy trajectory anchor p0
    if (t.sharpness)  t.sharpness  += off;   // SuperGrok11/15 (g_sam−g)² (per element)
    // SuperGrok11/15 phi weights (sg_phi_W1/b1/W2) are a per-TENSOR weight SET (NOT a
    // per-element slice) — so NOT rebased (staged to SMEM once, like NeuralGrok's psi).
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

#if SG_VIT_SCALAR_MEGAKERNEL
// ── The persistent megakernel (L3-REAL). gridDim.x = #SMs (one CTA/SM), 256
//    threads/CTA. VitSampleSmem (~183.67 KB) lives in DYNAMIC smem (extern). ────
//    GATED by SG_VIT_SCALAR_MEGAKERNEL (this fp32 path's VitSampleSmem grows with
//    SG_VIT_D and overflows the 227 KB dynamic-smem cap at the bench width; OFF lets
//    the TC path build at scaled d — see the flag note above). ────────────────────
// sizes/offsets are NOT host-passed: per-tensor numel/offset live in the
// __constant__ tables kVitSizes/kVitOffsets (vit_layout.cuh), read directly by
// the reduce + optimizer phases.
template <OptId Opt>
__global__ void __launch_bounds__(SG_TUNED_MEGA_BLOCK)
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
        &occ, (const void*)&fused_vit_megakernel<Opt>, SG_TUNED_MEGA_BLOCK,
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
    dim3 grid(launch_ctas), block(SG_TUNED_MEGA_BLOCK);
    fused_vit_megakernel<Opt><<<grid, block, dyn_smem, stream>>>(
        ctx, params, in, grad, lr, step, st);
    return cudaGetLastError();
}
#endif  // SG_VIT_SCALAR_MEGAKERNEL

// ════════════════════════════════════════════════════════════════════════════
//  WGMMA CELL DRIVER (DESIGN-TC-PIPELINE.md Fork B, the ViT twin of the decoder
//  R2.3 driver). Compiled only under the wgmma token; the scalar path above is
//  untouched. The TC path's per-CTA tile scratch lives in HBM (NOT dynamic smem)
//  — the only smem the TC kernel needs is the tiny wgmma staging arena (one
//  A(64×16) + one B(N×16) bf16 tile + a 256-float reduction + the 10 dW specs),
//  which is STATIC (≈ 7 KB), so this kernel — unlike the scalar ViT one — needs
//  NO dynamic-smem opt-in. The big ViT activation footprint (which forced the
//  scalar path to 184 KB dynamic smem) becomes HBM acts here (Fork B), so the
//  occupancy story is the decoder TC one: static smem, 0 dynamic.
// ════════════════════════════════════════════════════════════════════════════
#if (SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA)

// TC megakernel threads: 256 (the engine's producer+consumer warpgroup layout;
// wgmma is warpgroup-scoped on threads 0..127). We do NOT apply the scalar
// path's asymmetric setmaxnreg (it would STARVE the MMA warpgroup); the
// validated selftest ran no split. We match it.
#define SG_VIT_TC_MEGA_BLOCK 256

// ── smem arena for the unpipelined engine. The M-atom-interleaved wgmma pipeline
//    (task #24, decoder H1+H3 port) stages kVitAtomsPerSlot A(64×16) tiles per
//    k-step (one per stacked m64 atom in an interleave GROUP, issued back-to-back
//    into independent fp32 fragments so the tensor pipe overlaps them; atom-in-
//    group ai at sA + ai·kVitTcSmemA1) + ONE shared B(N×16) tile. kVitAtomsPerSlot
//    = kVitMaxIL (the interleave cap, default 2). Plus the 256-float reduction slot
//    + the 10 dW specs (identical for all threads → shared, not per-thread stack).
//    16B-aligned. N = SG_TUNED_TILE_N. ──
struct VitTcSmem {
    __nv_bfloat16 sA[vittc::kVitAtomsPerSlot * 64 * 16];
    __nv_bfloat16 sB[SG_TUNED_TILE_N * 16];
    float red[256];
    vittc::VitDwSpec spec[10];
};

// ── TC workspace layout (carved from in.workspace; the host sizes it). Regions
//    (float units):
//      [0 .. acts_f)                      : VitActs bf16 region (reinterpreted)
//      [acts_f .. acts_f + nCTA*scratch)  : per-CTA tile scratch (f32)
//      [.. + nCTA*kLnVecElems)            : per-CTA LN-vec partials (f32)
//      [.. + nCTA)                        : per-CTA loss slots (f32)
//      [.. + 1)                           : reduced scalar loss (host reads it)
//    acts_f = ceil(acts_bf16 / 2). ──
__host__ __device__ __forceinline__ int64_t vit_tc_acts_floats(int T, int B) {
    return (vittc::vit_acts_bf16_count(T, B) + 1) / 2;
}
// dW split-K partial floats (0 when G==1 → no extra scratch, the single-CTA path).
__host__ __device__ __forceinline__ int64_t vit_tc_dw_part_floats() {
    return vittc::vit_dw_part_floats(vittc::kVitDwSplitK);
}
// The d-scaled BENCH layout (SG_VIT_BENCH_LAYOUT=1) is adamw-ONLY (vit_bench.py /
// scripts/roofline_bench.py drive OptId::AdamW exclusively — none of the four staged
// optimizers ever runs in the bench TU), so carving them there is pure dead weight.
// At bench width the SG2 region in particular is pathological: vit_sg2_ws_stride_floats()
// is O(Nmax·d_model) PER CTA and Nmax=4d²·… grows with d ⇒ ~565 GB over 132 CTAs at
// d=2048, OOMing the 80 GB H100 (the SAME deep limit the decoder twin documents at
// dec_tc_sg2_floats — SG2's per-CTA workspace DESIGN doesn't scale; a chunked/streamed
// restructure is the documented deep item, out of scope here). So we GATE the four
// regions OFF at bench width: honest scoping (the bench never touches them), NOT
// correctness suppression. PRODUCTION is byte-identical — kVitStagedOptScratch is true
// there, so every gated helper folds back to its original value (constexpr bool, -O3
// inline). This mirrors the decoder's kDecStagedOptScratch (fused_decoder_megakernel.cuh).
#if SG_VIT_BENCH_LAYOUT
constexpr bool kVitStagedOptScratch = false;   // bench: adamw-only → elide the 4 staged-opt carves
#else
constexpr bool kVitStagedOptScratch = true;    // production: carve all 4 (opt-agnostic launcher)
#endif
// STAGED-optimizer cross-CTA reduction scratch (Prodigy d-estimate). The Prodigy
// stage publishes per-CTA (r,s) slots (2*nCTA) + a reduced-d slot (1) — an owner-
// computes tree (opt_stages_precompute.cuh), NO float atomic. Sized for the LARGEST
// nCTA (one CTA/SM = #SMs); tiny (≤ 2*132+1 floats) and carved UNCONDITIONALLY so
// the opt-agnostic launcher allocates one workspace fitting every OptId. Unused by
// AdamW/Lion/… (their P3 never touches it) → those cells stay byte-identical.
__host__ __device__ __forceinline__ int64_t vit_tc_opt_reduce_floats(int nCTA) {
    if (!kVitStagedOptScratch) return 0;     // bench (adamw-only): Prodigy never runs
    return (int64_t)2 * nCTA + 1;            // [r slots | s slots | reduced d]
}
// ── Muon (STAGED grid-cooperative Newton-Schulz) per-matrix scratch. The NS chain
//    runs ONE 2D weight at a time over all CTAs; the scratch holds X/A/AX/AAX/orth
//    for the LARGEST 2D weight + the per-CTA Frobenius-norm partials + inv_norm. The
//    momentum buffer (muon_buf) is NOT here — it PERSISTS across steps as optimizer
//    state, bound to the m slice (st.exp_avg). Largest vit 2D weight: ff0/ff2 = 512×
//    128 = 65536 numel; largest rows = 512 ⇒ A = 512×512. Carved UNCONDITIONALLY
//    (≈ 4·65536 + 512² + nCTA + 1 floats ≈ 2 MB) so the opt-agnostic launcher fits
//    every OptId; unused by every non-Muon cell (byte-identical). ──
//    kVitMuonMaxNumel = max over 2D weights of rows*cols; kVitMuonMaxRows = max rows.
static constexpr int kVitMuonMaxNumel = vit::kDff * vit::kD;     // 512*128 = 65536 (ff0/ff2)
static constexpr int kVitMuonMaxRows  = vit::kDff;               // 512 (ff0 rows)
__host__ __device__ __forceinline__ int64_t vit_tc_muon_floats(int nCTA) {
    if (!kVitStagedOptScratch) return 0;     // bench (adamw-only): Muon never runs
    // X + AX + AAX + orth (each maxNumel) + A (maxRows²) + nrm_partials(nCTA) + inv_norm(1)
    return (int64_t)4 * kVitMuonMaxNumel
         + (int64_t)kVitMuonMaxRows * kVitMuonMaxRows
         + nCTA + 1;
}
// ── LookSAM (STAGED in-kernel SAM 2nd backward) transient scratch — the ViT twin of
//    the decoder's dec_tc_looksam_floats (fused_decoder_megakernel.cuh). The SAM step
//    perturbs p in place, runs a SECOND fwd+bwd at p', and needs (a) a param BACKUP for
//    the exact restore (eager clones p before perturbing; an fp32 add is not bit-
//    reversible, so save — don't subtract back) + (b) a SEPARATE g_sam grad buffer so
//    the 2nd backward does NOT clobber the first grad `grad` (sam_dir = g_sam − grad
//    reads both). Both are total-sized + transient (NOT persistent state — sam_dir
//    itself persists in the `extra` state slice, bound by the cell). Carved
//    UNCONDITIONALLY (≈ 2·kVitTotalElems ≈ 3.3 MB) so the opt-agnostic launcher fits
//    every OptId; unused by every non-LookSAM cell (its P2.4 phase is if-constexpr'd
//    out → byte-identical). The ‖g‖ reduction reuses the loss workspace (sq_part/ron_bc,
//    as GrokAdamW's P2.5 / the decoder twin do), so no extra here. ──
__host__ __device__ __forceinline__ int64_t vit_tc_looksam_floats() {
    if (!kVitStagedOptScratch) return 0;      // bench (adamw-only): LookSAM never runs
    return (int64_t)2 * kVitTotalElems;       // [sam_backup | sam_grad]
}
// ── SuperGrok2 — the ViT twin of dec_tc_sg2_floats. Weights from HBM; the only SG2
//    workspace is the per-CTA meta-net scratch (sized for the largest tensor) + the
//    row_off64 staging prefix (kVitNumTensors int64 = 2*N floats — the adapter that
//    builds SG2State.row_off (const int64_t*) from __constant__ int kVitOffsets). ──
using VitSG2Dims = SG2Dims<>;
// max(kVitSizes), re-derived from the layout table (vit_layout.cuh) so the SAME SG2
// tail is correctly sized at ANY ladder width — was a d=128-pinned 65536.
constexpr int kVitSG2Nmax = kVitMaxTensorNumel;   // == vit_layout_check::max_size()
__host__ __device__ __forceinline__ int64_t vit_sg2_ws_stride_floats() {
    return (int64_t)2 * kVitNumTensors
         + sg2_ws_stride<VitSG2Dims>((int64_t)kVitSG2Nmax);
}
__host__ __device__ __forceinline__ int64_t vit_tc_sg2_floats(int nCTA) {
    if (!kVitStagedOptScratch) return 0;     // bench (adamw-only): SG2 never runs — and
        // its honestly-derived per-CTA stride (O(Nmax·d_model)) would OOM at large d.
    // +1 for the 8-byte realignment slack of sg2_ws_base (kVitTotalElems is odd).
    return (int64_t)nCTA * vit_sg2_ws_stride_floats() + 1;
}
__host__ __device__ __forceinline__ int64_t vit_tc_workspace_floats(int T, int B, int nCTA) {
    return vit_tc_acts_floats(T, B)
         + (int64_t)nCTA * vittc::vit_tile_scratch_total_f32()
         + (int64_t)nCTA * vittc::kLnVecElems
         + nCTA + 1
         + vit_tc_dw_part_floats()           // split-K dW partials (G>1)
         + vit_tc_opt_reduce_floats(nCTA)    // STAGED-opt (Prodigy) reduce slots
         + vit_tc_muon_floats(nCTA)          // STAGED-opt (Muon) NS per-matrix scratch
         + vit_tc_looksam_floats()           // STAGED-opt (LookSAM) SAM 2nd-bwd scratch
         + vit_tc_sg2_floats(nCTA);          // SuperGrok2 meta-net per-CTA scratch (carve-LAST)
}

#ifdef SG_VIT_PROFILE
// Diagnostic-only (behind SG_VIT_PROFILE; NEVER on the shipped path): per-phase
// clock64() deltas, max across CTAs (= critical-path duration per phase). Slots:
// [0]=P1 fwd, [1]=P1 bwd, [2]=B1 barrier wait, [3]=P2 dW-GEMM loop,
// [4]=P2 grad-assembly (biases+clspos+lnvec+loss), [5]=P3 opt tail. Read
// host-side via cudaMemcpyFromSymbol. clock64() is per-SM; a CTA stays on one SM
// so its own deltas are valid; atomicMax across CTAs gives the slowest CTA per
// phase.
__device__ unsigned long long g_vit_prof_max[6];
#endif

template <OptId Opt>
__global__ void __launch_bounds__(SG_VIT_TC_MEGA_BLOCK)
fused_vit_megakernel_tc(PersistentContext ctx,
                        float* __restrict__ params,
                        ViTInputCtx in,
                        float* __restrict__ grad,
                        float lr, int step, FusedOptState st) {
    __shared__ VitTcSmem sm;
    GridBarrier bar = ctx.barrier();
    const int cta = blockIdx.x;
    const int nCTA = (int)ctx.n_ctas;
    const int B = in.B;
    const int T = B * vit::kSeq;
    const int Tp = B * vit::kNPatch;

    // Workspace partition.
    float* ws = in.workspace;
    const int64_t acts_f = vit_tc_acts_floats(T, B);
    __nv_bfloat16* acts_base = reinterpret_cast<__nv_bfloat16*>(ws);
    float* scratch_base = ws + acts_f;
    const int64_t scratch_per = vittc::vit_tile_scratch_total_f32();
    float* lnvec_base = scratch_base + (int64_t)nCTA * scratch_per;
    float* loss_part  = lnvec_base + (int64_t)nCTA * vittc::kLnVecElems;
    float* loss_out   = loss_part + nCTA;
    // Split-K dW partials (G>1): the (gt,kc) 64×N partial tiles, carved AFTER the
    // loss slot (matches vit_tc_workspace_floats's term order). G==1 → dw_part unused.
    float* dw_part    = loss_out + 1;
    const int kDwG    = vittc::kVitDwSplitK;
    // STAGED-opt cross-CTA reduce slots (Prodigy d), carved AFTER the dW partials
    // (matches vit_tc_workspace_floats's term order). Unused unless Opt==Prodigy.
    float* opt_reduce = dw_part + vit_tc_dw_part_floats();
    // Muon NS per-matrix scratch, carved AFTER the Prodigy reduce slots (matches
    // vit_tc_workspace_floats's term order). Unused unless Opt==Muon. Layout:
    //   [muon_X | muon_AX | muon_AAX | muon_orth] (each kVitMuonMaxNumel)
    //   [muon_A (kVitMuonMaxRows²)] [nrm_partials (nCTA)] [inv_norm (1)]
    float* muon_base = opt_reduce + vit_tc_opt_reduce_floats(nCTA);
    // LookSAM SAM 2nd-backward scratch, carved AFTER the Muon scratch (term order
    // matches vit_tc_workspace_floats — carving it LAST keeps every prior region's
    // offset unchanged, so the already-green cells are byte-identical). Layout:
    //   [sam_backup (total)] [sam_grad (total)]. Unused unless Opt==LookSAM.
    float* sam_backup = muon_base + vit_tc_muon_floats(nCTA);
    float* sam_grad   = sam_backup + kVitTotalElems;
    // SuperGrok2 meta-net per-CTA scratch, carved AFTER the LookSAM sam_grad (term
    // order matches vit_tc_workspace_floats — carve-LAST keeps every prior region's
    // offset unchanged, so the already-green cells are byte-identical). ALIGN to 8
    // bytes (the per-CTA slice fronts an int64 row_off64 staging array; kVitTotalElems
    // is odd so sam_grad+total lands on an odd float offset → an int64 read there is
    // misaligned). Round up to the next even float. Unused unless Opt==SuperGrok2.
    float* sg2_ws_base = sam_grad + kVitTotalElems;
    if (((uintptr_t)sg2_ws_base & 0x7) != 0) sg2_ws_base += 1;   // → 8-byte aligned
    (void)sg2_ws_base;   // referenced only by the SuperGrok2 P3-SG2 phase

    vittc::VitActs acts = vittc::vit_acts_bind(acts_base, T, B);
    vittc::VitTileScratch sc = vittc::vit_tile_scratch_bind(scratch_base + (int64_t)cta * scratch_per);
    float* my_lnvec = lnvec_base + (int64_t)cta * vittc::kLnVecElems;

    VitWeights w = vit_bind(params);

    // ── P0: zero this CTA's LN-vec partials + loss slot (dW/cls/pos/patch grads
    //    are written-once → no pre-zero). ──
    for (int i = threadIdx.x; i < vittc::kLnVecElems; i += blockDim.x) my_lnvec[i] = 0.0f;
    if (threadIdx.x == 0) loss_part[cta] = 0.0f;
    // STAGED-opt (Prodigy) reduce slots: INTEGRATION-OPTSTAGES.md §1 REQUIRES
    // prodigy_partials + prodigy_d zero-initialized before launch (the owner-computes
    // tree reads EVERY slot in ascending order; a CTA that claims no tensor in phaseA
    // still has its (r,s) slot read, and the persistent device workspace is NOT
    // re-zeroed between launches — so a stale slot from a prior step's reduction, or
    // garbage from the caching allocator, would make the cross-tensor d non-determi-
    // nistic / NaN). phaseA's block_reduce DOES write each slot, but zeroing here is
    // the contract guarantee + defends the prodigy_d broadcast slot. Guarded to the
    // Prodigy instantiation so every other cell is byte-identical (no extra work).
    if constexpr (Opt == OptId::Prodigy) {
        const int64_t n_opt = vit_tc_opt_reduce_floats(nCTA);   // 2*nCTA + 1
        for (int64_t i = (int64_t)cta * blockDim.x + threadIdx.x; i < n_opt;
             i += (int64_t)nCTA * blockDim.x)
            opt_reduce[i] = 0.0f;
    }
    bar.sync();   // B0

    // ── P1: token-tile-parallel fwd+bwd. Each CTA grid-strides over (sub-)tiles
    //    of kVitP1SubtileRows rows; for its tile it runs fwd (→ acts X, NLL) then
    //    bwd (→ acts dY, dh0, LN-vec partials). Barrier-free within the tile. The
    //    B1 load-imbalance knob (SG_TUNED_VIT_P1_SUBTILE_S) sets the sub-tile to S
    //    whole samples (kSeq·S rows): S=64 ⇒ kVitP1SubtileRows==kTileM==1088 ⇒
    //    this loop is BYTE-IDENTICAL to the whole-1088-tile schedule (n_tiles=16
    //    at B=1024); S<64 grows n_tiles ~64/S× so the idle CTAs (116/132 at B1024)
    //    pick up P1 work and the B1 spin collapses. The tile boundary stays a
    //    whole-sample boundary (attention stays in-tile). ──
    const int nrows_tile = vittc::kVitP1SubtileRows;
    const int n_tiles = (T + nrows_tile - 1) / nrows_tile;
    float nll_acc = 0.0f;
#ifdef SG_VIT_PROFILE
    unsigned long long prof_fwd = 0, prof_bwd = 0;
#endif
    for (int ti = cta; ti < n_tiles; ti += nCTA) {
        const int g0 = ti * nrows_tile;
        const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
#ifdef SG_VIT_PROFILE
        __syncthreads(); unsigned long long _c0 = clock64();
#endif
        float nll = vittc::vittc_forward_tile(w, g0, nrows, acts, sc, in.patches, in.targets,
                                              sm.sA, sm.sB, sm.red);
#ifdef SG_VIT_PROFILE
        __syncthreads(); unsigned long long _c1 = clock64();
#endif
        vittc::vittc_backward_tile(w, g0, nrows, B, acts, sc, in.targets,
                                   my_lnvec, sc.work2, sm.sA, sm.sB, sm.red);
#ifdef SG_VIT_PROFILE
        __syncthreads(); unsigned long long _c2 = clock64();
        if (threadIdx.x == 0) { prof_fwd += _c1 - _c0; prof_bwd += _c2 - _c1; }
#endif
        if (threadIdx.x == 0) nll_acc += nll;
        __syncthreads();
    }
    if (threadIdx.x == 0) loss_part[cta] = nll_acc;
#ifdef SG_VIT_PROFILE
    if (threadIdx.x == 0) {
        atomicMax(&g_vit_prof_max[0], prof_fwd);
        atomicMax(&g_vit_prof_max[1], prof_bwd);
    }
    unsigned long long _b1a = (threadIdx.x == 0) ? clock64() : 0;
#endif
    bar.sync();   // B1: all acts (X + dY) + LN-vec partials complete
#ifdef SG_VIT_PROFILE
    if (threadIdx.x == 0) { unsigned long long _b1b = clock64(); atomicMax(&g_vit_prof_max[2], _b1b - _b1a); }
#endif

    // ── P2: assemble all 32 grads into `grad`. dW output-stationary (gt % nCTA),
    //    biases, cls/pos owner-scan, LN-vec reduce. No partials. The 10 dW specs
    //    are built into SHARED smem (thread 0; identical for all). ──
    if (threadIdx.x == 0) vittc::vittc_build_dw_specs(acts, B, T, Tp, sm.spec);
    __syncthreads();
    vittc::VitDwSpec* spec = sm.spec;
    const int n_dw = vittc::vittc_dw_total_tiles<SG_TUNED_TILE_N>(spec);
#ifdef SG_VIT_PROFILE
    __syncthreads(); unsigned long long _dwa = (threadIdx.x == 0) ? clock64() : 0;
#endif
    if (kDwG > 1) {
        // SPLIT-K (multi-CTA tiling): fan (n_dw·G) (tile,chunk) partials over the
        // grid so the ~60% idle SMs do work; deterministic ascending-chunk reduce.
        for (int item = cta; item < n_dw * kDwG; item += nCTA) {
            const int gt = item / kDwG, kc = item % kDwG;
            vittc::vittc_dw_run_tile_splitk<SG_TUNED_TILE_N>(spec, gt, kc, kDwG, acts.dh0, dw_part, sm.sA, sm.sB);
        }
        bar.sync();   // all (gt,kc) partials complete before the reduce reads them
        vittc::vittc_dw_reduce_splitk<SG_TUNED_TILE_N>(spec, n_dw, kDwG, dw_part, grad, cta, nCTA);
    } else {
        for (int gt = cta; gt < n_dw; gt += nCTA)
            vittc::vittc_dw_run_tile<SG_TUNED_TILE_N>(spec, gt, acts.dh0, grad, sm.sA, sm.sB);
    }
#ifdef SG_VIT_PROFILE
    __syncthreads();
    if (threadIdx.x == 0) { unsigned long long _dwb = clock64(); atomicMax(&g_vit_prof_max[3], _dwb - _dwa); }
    unsigned long long _gaa = (threadIdx.x == 0) ? clock64() : 0;
#endif
    vittc::vittc_dw_biases(spec, acts.dh0, grad, cta, nCTA);
    vittc::vittc_clspos_owner_scan(acts, T, grad, cta, nCTA);
    vittc::vittc_lnvec_reduce(lnvec_base, grad, nCTA, cta);
    // Loss reduce (fp64) by CTA 0.
    if (cta == 0 && threadIdx.x == 0) {
        double s = 0.0;
        for (int c = 0; c < nCTA; ++c) s += (double)loss_part[c];
        *in.loss_out = (float)(s / (double)B);
    }
    (void)loss_out;
#ifdef SG_VIT_PROFILE
    if (threadIdx.x == 0) { unsigned long long _gab = clock64(); atomicMax(&g_vit_prof_max[4], _gab - _gaa); }
#endif
    bar.sync_reset(ctx.g_next_task);   // B2: reduced grad ready; reset queue

    // ── P2.4 (LookSAM ONLY, SAM steps): the MODEL-COUPLED SAM 2nd backward that
    //    produces st.sam_dir = g_sam − g (INTEGRATION-OPTSTAGES §6). PORT of the
    //    decoder twin's P2.4 (fused_decoder_megakernel.cuh) onto the ViT tile fns.
    //    Eager LookSAM, every k steps (looksam.py sam_step + looksam.h):
    //      ‖g‖     = GLOBAL L2 norm over ALL reduced grads; rho_over_norm = rho/‖g‖
    //      p'      = p + rho_over_norm·g          (perturb in the grad direction;
    //                backup = p.clone() for the exact restore — fp32 add is not
    //                bit-reversible, so we MUST save, not subtract back)
    //      g_sam   = ∇L(p')                       (a FULL SECOND in-kernel fwd+bwd +
    //                deterministic grad assembly at the perturbed weights, written to
    //                the SEPARATE `sam_grad` buffer so `grad` (the first grad the apply
    //                tail blends from) is untouched)
    //      sam_dir = g_sam − g                    (cached in the PERSISTENT `extra`
    //                state slice == st.sam_dir; reused verbatim on the k−1 intervening
    //                steps where looksam_sam==0, so NO 2nd pass runs then)
    //      restore p = backup
    //    The apply tail (P3, apply_optimizer<LookSAM>) then blends g_adj=(1−α)g+α·
    //    sam_dir and runs AdamW — UNCHANGED. DETERMINISM: the ‖g‖ reduction is the
    //    IDENTICAL deterministic ascending-CTA shape as GrokAdamW's P2.5 (fixed
    //    contiguous per-CTA range → per-CTA partial → CTA0 ascending sum), and the
    //    2nd fwd+bwd+assembly reuses the SAME A/A/A-clean machinery as the first pass
    //    (fixed tile ownership, ascending-k reductions), so the whole phase is
    //    deterministic BY CONSTRUCTION. Guarded so every other opt's path is byte-
    //    identical (no extra barrier / work). On a NON-SAM step (looksam_sam==0) this
    //    whole block is skipped — st.sam_dir already holds the cached direction.
    //
    //    SuperGrok11/15 SHARE this exact SAM 2nd-backward machinery (INTEGRATION-
    //    OPTSTAGES §4/§5): IDENTICAL perturb→2nd in-kernel fwd+bwd→restore; only the
    //    elementwise WRITE in (d) differs — LookSAM writes sam_dir = g_sam − g, SG11/15
    //    write sharpness = (g_sam − g)² (supergrok11_sm90.cuh:246 / supergrok15_sm90.cuh
    //    :315), the 2nd MLP input the P2.45 meta-net mu precompute reads. Same st.looksam_sam
    //    SAM-step gate + st.rho radius; cached sharpness reused on non-SAM steps.
    if constexpr (Opt == OptId::LookSAM || Opt == OptId::SuperGrok11 ||
                  Opt == OptId::SuperGrok15 || Opt == OptId::SuperGrok2) {
        if (st.looksam_sam != 0.0f) {
            // (a) GLOBAL ‖g‖ over the reduced grad, deterministic (reuse the loss
            //     workspace as P2.5 does: loss_part[nCTA] = per-CTA Σg², loss_out[1]
            //     broadcasts rho_over_norm). The reduced loss is already in
            //     *in.loss_out (host state), so this is free scratch.
            float* sq_part = loss_part;          // [nCTA] per-CTA Σ g²  (ascending reduce)
            float* ron_bc  = loss_out;           // [1] broadcast rho_over_norm
            const int64_t total_e = kVitTotalElems;
            {
                const int64_t base = total_e / nCTA, rem = total_e % nCTA;
                const int64_t e0 = (int64_t)cta * base + (cta < rem ? cta : rem);
                const int64_t ecnt = base + (cta < rem ? 1 : 0);
                float tsum = 0.0f;
                for (int64_t i = threadIdx.x; i < ecnt; i += blockDim.x) {
                    const float gv = grad[e0 + i];
                    tsum += gv * gv;
                }
                float* red = sm.red;
                red[threadIdx.x] = tsum;
                __syncthreads();
                for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
                    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
                    __syncthreads();
                }
                if (threadIdx.x == 0) sq_part[cta] = red[0];
                bar.sync();   // B2.4a: all per-CTA Σg² partials complete
                if (cta == 0 && threadIdx.x == 0) {
                    double ss = 0.0;
                    for (int c = 0; c < nCTA; ++c) ss += (double)sq_part[c];
                    const float gnorm = sqrtf((float)ss);
                    // rho / ‖g‖ (matches bindings.cpp looksam_perturb_all). gnorm>0 in
                    // any real training step; guard div-by-0 → 0 (no perturb) to be safe.
                    *ron_bc = (gnorm > 0.0f) ? (st.rho / gnorm) : 0.0f;
                }
                bar.sync();   // B2.4b: rho_over_norm broadcast ready
            }
            const float rho_over_norm = *ron_bc;
            // (b) Backup + perturb p in place: p' = p + rho_over_norm·g. Grid-strided
            //     over the flat param vector (each element owned once → no race).
            {
                const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
                for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
                     i < total_e; i += gstride) {
                    sam_backup[i] = params[i];
                    params[i] = params[i] + rho_over_norm * grad[i];
                }
            }
            bar.sync();   // B2.4c: all params perturbed before the 2nd forward reads them
            // (c) SECOND fwd+bwd at the perturbed weights → g_sam in `sam_grad`. Mirror
            //     P1 + P2 EXACTLY, but: re-zero the lnvec partials (they accumulate), do
            //     NOT write the loss (keep the first-pass loss), and assemble into
            //     `sam_grad` (NOT `grad`). `w` wraps `params` (now perturbed) so the tile
            //     fns read the perturbed weights with no re-bind needed.
            for (int i = threadIdx.x; i < vittc::kLnVecElems; i += blockDim.x) my_lnvec[i] = 0.0f;
            bar.sync();   // B2.4d: lnvec partials cleared before the 2nd backward accumulates
            for (int ti = cta; ti < n_tiles; ti += nCTA) {
                const int g0 = ti * nrows_tile;
                const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
                vittc::vittc_forward_tile(w, g0, nrows, acts, sc, in.patches, in.targets,
                                          sm.sA, sm.sB, sm.red);
                vittc::vittc_backward_tile(w, g0, nrows, B, acts, sc, in.targets,
                                           my_lnvec, sc.work2, sm.sA, sm.sB, sm.red);
                __syncthreads();
            }
            bar.sync();   // B2.4e: all 2nd-pass acts (X + dY) + LN-vec partials complete
            // Re-build the dW specs from the 2nd-pass acts (dY adjoints changed).
            if (threadIdx.x == 0) vittc::vittc_build_dw_specs(acts, B, T, Tp, sm.spec);
            __syncthreads();
            vittc::VitDwSpec* spec2 = sm.spec;
            const int n_dw2 = vittc::vittc_dw_total_tiles<SG_TUNED_TILE_N>(spec2);
            if (kDwG > 1) {
                for (int item = cta; item < n_dw2 * kDwG; item += nCTA) {
                    const int gt = item / kDwG, kc = item % kDwG;
                    vittc::vittc_dw_run_tile_splitk<SG_TUNED_TILE_N>(spec2, gt, kc, kDwG, acts.dh0, dw_part, sm.sA, sm.sB);
                }
                bar.sync();
                vittc::vittc_dw_reduce_splitk<SG_TUNED_TILE_N>(spec2, n_dw2, kDwG, dw_part, sam_grad, cta, nCTA);
            } else {
                for (int gt = cta; gt < n_dw2; gt += nCTA)
                    vittc::vittc_dw_run_tile<SG_TUNED_TILE_N>(spec2, gt, acts.dh0, sam_grad, sm.sA, sm.sB);
            }
            vittc::vittc_dw_biases(spec2, acts.dh0, sam_grad, cta, nCTA);
            vittc::vittc_clspos_owner_scan(acts, T, sam_grad, cta, nCTA);
            vittc::vittc_lnvec_reduce(lnvec_base, sam_grad, nCTA, cta);
            bar.sync();   // B2.4f: g_sam fully assembled in sam_grad
            // (d) WRITE the SAM side-channel (persistent state slice) + restore p.
            //     LookSAM → sam_dir = g_sam − g; SuperGrok11/15 → sharpness = (g_sam − g)².
            {
                const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
                for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
                     i < total_e; i += gstride) {
                    const float diff = sam_grad[i] - grad[i];
                    if constexpr (Opt == OptId::LookSAM) {
                        const_cast<float*>(st.sam_dir)[i] = diff;
                    } else {
                        const_cast<float*>(st.sharpness)[i] = diff * diff;
                    }
                    params[i] = sam_backup[i];   // restore the ORIGINAL weights
                }
            }
            bar.sync_reset(ctx.g_next_task);   // B2.4g: params restored + side-channel ready; reset queue for P3
        }
    }

    // ── P2.5 (GrokAdamW ONLY): GLOBAL grad-norm clip coefficient. Identical to the
    //    decoder twin (fused_decoder_megakernel.cuh P2.5). Eager grokadamw clips the
    //    WHOLE grad set to grad_clip via a GLOBAL L2 norm (clip_grad_norms_device_side
    //    → total_norm = sqrt(Σ_i ‖g_i‖²), clip_coef = grad_clip/(total_norm+1e-6) when
    //    total_norm>grad_clip, else 1) BEFORE the apply. We replicate it on the REDUCED
    //    grad with a deterministic ascending reduction (no float atomics): each CTA sums
    //    a contiguous element-range into a per-CTA partial slot, CTA0 sums the partials
    //    in ascending CTA order → total_norm → clip_coef, broadcast via a workspace
    //    slot. The grad buffer is NOT mutated (the return_grad oracle + the eager-side
    //    clip must both see the unclipped reduced grad); the coefficient is applied
    //    per-element inside apply_optimizer<GrokAdamW>. Guarded so every other opt's P3
    //    is byte-identical (no extra barrier/work). Reuses loss_part/loss_out as free
    //    scratch (the reduced loss is already in *in.loss_out; the dW partials are
    //    consumed → that whole region after loss_out is also free).
    //    EXTENDED to NeuralGrok (same eager global clip; consumed in apply_optimizer<NeuralGrok>).
    if constexpr (Opt == OptId::GrokAdamW || Opt == OptId::NeuralGrok) {
        float* sq_part = loss_part;          // [nCTA] per-CTA Σ g²  (ascending reduce)
        float* coef_bc = loss_out;           // [1] broadcast clip_coef
        const int64_t total_e = kVitTotalElems;
        const int64_t base = total_e / nCTA, rem = total_e % nCTA;
        const int64_t e0 = (int64_t)cta * base + (cta < rem ? cta : rem);
        const int64_t ecnt = base + (cta < rem ? 1 : 0);
        float tsum = 0.0f;
        for (int64_t i = threadIdx.x; i < ecnt; i += blockDim.x) {
            const float gv = grad[e0 + i];
            tsum += gv * gv;
        }
        // Block reduction via the smem the TC GEMM already owns (sm.red, fp32).
        float* red = sm.red;
        red[threadIdx.x] = tsum;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) sq_part[cta] = red[0];
        bar.sync();   // B2.5a: all per-CTA sum-of-squares partials complete
        if (cta == 0 && threadIdx.x == 0) {
            double ss = 0.0;
            for (int c = 0; c < nCTA; ++c) ss += (double)sq_part[c];
            const float total_norm = sqrtf((float)ss);
            float coef = 1.0f;
            if (st.grad_clip > 0.0f && total_norm > st.grad_clip)
                coef = st.grad_clip / (total_norm + 1e-6f);
            *coef_bc = coef;
        }
        bar.sync();   // B2.5b: clip_coef broadcast slot ready for all CTAs
        st.clip_coef = *coef_bc;   // every CTA reads the single global coefficient
    }

    // ── P2.6 (Prodigy ONLY): STAGED global-d estimate (cross-ALL-tensors reduce).
    //    Identical to the decoder twin's P2.6. The apply tail (apply_optimizer
    //    <Prodigy>) reads st.d_factor (the effective LR scale d), which is a single
    //    cross-tensor reduction (NOT per-element) → an IN-KERNEL phase here, BYTE-
    //    FAITHFUL to the live eager multi-tensor path (prodigy_sm90.cuh →
    //    launch_multi_tensor_prodigy_fused_reduce_step, the order prodigy.py executes):
    //      d_prev  = persisted d_lr  (step 1: d0 cold-start — the zero-init state slot
    //                is 0 but eager inits _d_lr=d0=1e-6, so seed it here)
    //      r_ema  <- beta3·r_ema + Σ d_prev²·<g, p0−p>     (decay persisted SCALAR,
    //      s_ema  <- beta3·s_ema + Σ d_prev²·|g|            then add this step's Σ)
    //      d       = max(d_prev, d_coef·r_ema/|s_ema|)      (prodigy_update_d; d_coef
    //                scales ONLY the candidate — persisted r_ema stays UNSCALED)
    //    phaseA publishes per-CTA (r,s) into opt_reduce slots (owner-computes tree,
    //    no float atomic, deterministic); a grid barrier; CTA0 owner-sums ascending,
    //    EMA-decays, updates d, broadcasts. Guarded so every other opt is byte-
    //    identical (no extra barrier/work). p0 (param_init) is the trajectory anchor.
    // DETERMINISM — FIXED-PARTITION reduction (mirrors the A/A/A-CLEAN GrokAdamW
    // P2.5 above + the decoder twin's P2.6), NOT the work-steal task-queue drain.
    // WHY THIS SHAPE (the A/A/A determinism leg): the work-steal phaseA drained the
    // global TaskQueue (atomicAdd(g_next_task)) in timing-dependent claim order, so
    // each CTA's (r,s) slot could be a different tensor subset run-to-run; with non-
    // associative fp32 add the partials then regroup differently and the owner's d can
    // differ at ULP level from step>=2 — a COMPONENT_CONTRACT deterministic-reduction
    // violation (not guaranteed reproducible BY CONSTRUCTION, even where empirically
    // bit-exact on a given GPU). A FIXED contiguous element range per CTA over the
    // FLAT [0,kVitTotalElems) arrays (params/param_init/grad are parallel flat blobs
    // → flat Σ == cross-tensor Σ) with a plain sync() is the IDENTICAL shape as the
    // A/A/A-CLEAN GrokAdamW P2.5 — deterministic by construction. Per-element math
    // still CALLS canonical prodigy_partials_step (prodigy.h, single-source).
    // g_next_task is UNTOUCHED (B2 reset it), so P3's drain is unchanged — no
    // sync_reset in P2.6.
    if constexpr (Opt == OptId::Prodigy) {
        float* r_part = opt_reduce;                  // [nCTA] per-CTA Σ d²·<g,p0−p>
        float* s_part = opt_reduce + nCTA;           // [nCTA] per-CTA Σ d²·|g|
        float* d_bc   = opt_reduce + 2 * nCTA;       // [1] reduced-d broadcast
        const float d_prev = (step == 1) ? st.d0 : st.prodigy_persist[2];
        const int64_t total_p = kVitTotalElems;
        const int64_t pbase = total_p / nCTA, prem = total_p % nCTA;
        const int64_t pe0 = (int64_t)cta * pbase + (cta < prem ? cta : prem);
        const int64_t pecnt = pbase + (cta < prem ? 1 : 0);
        float r_acc = 0.0f, s_acc = 0.0f;
        for (int64_t i = threadIdx.x; i < pecnt; i += blockDim.x) {
            algo::prodigy_partials_step(params, st.param_init, grad,
                                        d_prev, pe0 + i, r_acc, s_acc);
        }
        float r_block, s_block;
        prim::block_reduce_sum2_f32(r_acc, s_acc, r_block, s_block);
        if (threadIdx.x == 0) { r_part[cta] = r_block; s_part[cta] = s_block; }
        bar.sync();   // B2.6a: all per-CTA (r,s) partials published & visible
        if (cta == 0 && threadIdx.x == 0) {
            float r_step = 0.0f, s_step = 0.0f;     // ascending-CTA owner-sum
            for (int c = 0; c < nCTA; ++c) { r_step += r_part[c]; s_step += s_part[c]; }
            const float r_ema = st.beta3 * st.prodigy_persist[0] + r_step;
            const float s_ema = st.beta3 * st.prodigy_persist[1] + s_step;
            const float d_new = algo::prodigy_update_d(d_prev, st.d_coef * r_ema, s_ema);
            st.prodigy_persist[0] = r_ema;          // persist UNSCALED EMA
            st.prodigy_persist[1] = s_ema;
            st.prodigy_persist[2] = d_new;          // persisted d_lr for next step
            d_bc[0]               = d_new;          // broadcast to all CTAs
        }
        bar.sync();   // B2.6b: d visible to every CTA before the apply
        st.d_factor = d_bc[0];                      // the reduced d the tail reads
    }

    // ── P2.7 (Muon ONLY): grid-cooperative Newton-Schulz orthogonalization of the
    //    2D weights (INTEGRATION-OPTSTAGES §3). For EACH 2D matrix (kVitMuon2D) all
    //    CTAs cooperate: buf=μ·buf+g (buf is the PERSISTENT m-slice — momentum state,
    //    NOT transient), ‖buf‖_F via per-CTA partials → inv_norm, X=buf·inv_norm,
    //    then ns_steps × { A=XXᵀ → AX=A·X → AAX=A·AX → orth=a·X+b·AX+c·AAX, swap },
    //    then the canonical muon_update_step apply (decay·p + neg_lr_scale·orth). The
    //    1D / non-2D weights (cls_token, biases, LN γ/β) take the AdamW tail in P3.
    //    The elementwise bodies CALL muon.h (muon_momentum_normalize_step via the
    //    phaseA buf body, muon_ns_combine_step, muon_update_step); the matmuls are the
    //    cited new device code (the eager path delegates to torch::mm/cuBLAS). Guarded
    //    so every other opt is byte-identical (no extra barriers / work).
    if constexpr (Opt == OptId::Muon) {
        // Carve the per-matrix NS scratch (sized for the largest 2D weight).
        PrecomputeWorkspace pw{};
        pw.muon_X            = muon_base;
        pw.muon_AX           = pw.muon_X   + kVitMuonMaxNumel;
        pw.muon_AAX          = pw.muon_AX  + kVitMuonMaxNumel;
        pw.muon_orth         = pw.muon_AAX + kVitMuonMaxNumel;
        pw.muon_A            = pw.muon_orth + kVitMuonMaxNumel;
        pw.muon_nrm_partials = pw.muon_A   + (int64_t)kVitMuonMaxRows * kVitMuonMaxRows;
        pw.muon_inv_norm     = pw.muon_nrm_partials + nCTA;
        const float momentum = st.beta1;          // Muon momentum (eager Muon: betas[0])
        const int   ns_steps = 5;                 // bindings.cpp default
        for (int mi = 0; mi < vittc::kVitNumMuon2D; ++mi) {
            const vittc::VitMuon2D M = vittc::kVitMuon2D[mi];
            const int rows = M.rows, cols = M.cols;
            const int64_t numel = (int64_t)rows * cols;
            const int64_t off   = (int64_t)kVitOffsets[M.tidx];
            // buf = the PERSISTENT momentum slice for this matrix (st.exp_avg+off).
            pw.muon_buf = st.exp_avg + off;
            // phaseA: buf=μ·buf+g, publish per-CTA ‖buf‖_F² → reduce → inv_norm → X.
            muon_momentum_norm_phaseA(grad + off, numel, momentum, pw);
            bar.sync();
            muon_norm_reduce_phaseB(ctx, pw);
            bar.sync();
            muon_scale_X(numel, pw);
            bar.sync();
            // NS iterations. After each combine, orth holds the new iterate; swap so
            // the next iteration reads it as X. ns_steps swaps → final result is in
            // muon_X iff ns_steps even else muon_orth (we track the parity below).
            for (int s = 0; s < ns_steps; ++s) {
                // A = X Xᵀ  (M=rows,N=rows,K=cols; B transposed).
                muon_matmul(pw.muon_X, pw.muon_X, pw.muon_A, rows, rows, cols, cols, cols, /*bT=*/true);
                bar.sync();
                // AX = A X   (M=rows,N=cols,K=rows).
                muon_matmul(pw.muon_A, pw.muon_X, pw.muon_AX, rows, cols, rows, rows, cols, /*bT=*/false);
                bar.sync();
                // AAX = A (AX) (M=rows,N=cols,K=rows).
                muon_matmul(pw.muon_A, pw.muon_AX, pw.muon_AAX, rows, cols, rows, rows, cols, /*bT=*/false);
                bar.sync();
                muon_ns_combine_phase(numel, pw);   // orth = a·X + b·AX + c·AAX
                bar.sync();
                // swap X <-> orth (next iter reads X; final result tracked by parity).
                float* tmp = pw.muon_X; pw.muon_X = pw.muon_orth; pw.muon_orth = tmp;
            }
            // After ns_steps swaps the final orthogonalized matrix is in muon_X (the
            // last combine wrote muon_orth, then we swapped → muon_X). Apply Muon:
            //   p = decay_factor·p + neg_lr_scale·orth   (muon.h:63-73)
            const float scale        = 0.2f * sqrtf((float)(rows > cols ? rows : cols));
            const float neg_lr_scale = -lr * scale;
            const float decay_factor = 1.0f - lr * st.wd;
            float* __restrict__ p = params + off;
            const float* __restrict__ orth_final = pw.muon_X;   // post-swap result
            const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
            for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += gstride)
                p[i] = decay_factor * p[i] + neg_lr_scale * orth_final[i];
            bar.sync();   // matrix done; all CTAs synchronized before the next matrix
        }
    }

    // ── P2.45 (SuperGrok11/15 ONLY) staged into P3: the per-TENSOR meta-net mu
    //    precompute (INTEGRATION-OPTSTAGES §4/§5) — ViT twin of the decoder's. SINGLE
    //    task-queue drain, NO grid barrier (each tensor is self-contained: mu(T) +
    //    SG11's per-tensor cosine gate(T) depend ONLY on T's elements). The phi weights
    //    (per-TENSOR weight set) stage to SMEM ONCE per block before the drain; SG15's
    //    gate is the host scalar st.gate. sharpness was produced by P2.4 (or cached).
    __shared__ float sg_sW1[kSgPhiHidden * 2];
    __shared__ float sg_sb1[kSgPhiHidden];
    __shared__ float sg_sW2[kSgPhiHidden];
    if constexpr (Opt == OptId::SuperGrok11 || Opt == OptId::SuperGrok15) {
        sg_stage_phi_weights<kSgPhiHidden>(st.sg_phi_W1, st.sg_phi_b1, st.sg_phi_W2,
                                           sg_sW1, sg_sb1, sg_sW2);   // syncs
    }

    // ── P3-SG2 (SuperGrok2 ONLY) — the ViT twin of the decoder's P3-SG2 phase. The
    //    FULL CSA/HCA/PEER/GRU meta-net as the optimizer phase, run INSTEAD of the
    //    per-element apply_optimizer<SuperGrok2> stub. Each CTA work-steals WHOLE
    //    tensors and runs sg2_meta_stages (STAGE -1 in-kernel segmented sort → S0..S5)
    //    reading st.sharpness (P2.4 SAM 2nd backward) + the HBM meta-net bundle. The
    //    kernel returns after this. if-constexpr'd to SuperGrok2 ⇒ every other cell is
    //    byte-identical.
    if constexpr (Opt == OptId::SuperGrok2) {
        SG2Weights w2{
            st.sg2_input_proj_W, st.sg2_input_proj_b,
            st.sg2_csa_q_W, st.sg2_csa_k_W, st.sg2_csa_v_W, st.sg2_csa_out_W,
            st.sg2_csa_compress_w, st.sg2_csa_idx_DQ, st.sg2_csa_idx_K,
            st.sg2_hca_q_W, st.sg2_hca_k_W, st.sg2_hca_v_W, st.sg2_hca_out_W,
            st.sg2_gru_Wz, st.sg2_gru_bz, st.sg2_gru_Wr, st.sg2_gru_br,
            st.sg2_gru_Wh, st.sg2_gru_bh,
            st.sg2_peer_query_Ws, st.sg2_prod_keys_A, st.sg2_prod_keys_B,
            st.sg2_expert_W1, st.sg2_expert_b1, st.sg2_expert_W2, st.sg2_expert_b2};
        SG2Scalars sc2{
            st.sg2_alpha, st.sg2_gru_decay, st.sg2_lamb_eff,
            st.sg2_beta1, st.sg2_bc1, st.sg2_bc2,
            st.sg2_rescale, st.beta2, lr, st.wd, st.eps};
        float* sg2_base = sg2_ws_base + (int64_t)blockIdx.x * vit_sg2_ws_stride_floats();
        int64_t* row_off64 = reinterpret_cast<int64_t*>(sg2_base);
        for (int t = threadIdx.x; t < kVitNumTensors; t += blockDim.x)
            row_off64[t] = (int64_t)kVitOffsets[t];
        float* sg2_meta_ws = sg2_base + 2 * kVitNumTensors;
        __syncthreads();
        SG2State stt{};
        stt.exp_avg     = st.exp_avg;
        stt.exp_avg_sq  = st.exp_avg_sq;
        stt.mu          = const_cast<float*>(st.mu);
        stt.slow        = st.sg2_slow;
        stt.gru_state   = st.sg2_gru_state;
        stt.perm        = nullptr;
        stt.unsort      = nullptr;
        stt.workspace   = sg2_meta_ws;
        stt.ws_stride   = sg2_ws_stride<VitSG2Dims>((int64_t)kVitSG2Nmax);
        stt.n_tensors   = kVitNumTensors;
        stt.n           = kVitSizes;
        stt.row_off     = row_off64;
        __shared__ int task_slot2;
        TaskQueue q2 = ctx.queue();
        for (int t = q2.next_block(&task_slot2); t < kVitNumTensors;
             t = q2.next_block(&task_slot2)) {
            sg2_meta_stages<VitSG2Dims, SG2Weights, float, float, /*BuildSort=*/true>(
                w2, t, stt, sc2, params, grad, st.sharpness, sg2_meta_ws);
        }
        return;   // SG2 owns the whole optimizer phase; skip P3.
    }

    // ── P3: scalar optimizer tail over the reduced grad (REUSED verbatim). ──
    st.lr = lr;
#ifdef SG_VIT_PROFILE
    unsigned long long _p3a = (threadIdx.x == 0) ? clock64() : 0;
#endif
    {
        __shared__ int task_slot;
        TaskQueue q = ctx.queue();
        for (int t = q.next_block(&task_slot); t < kVitNumTensors;
             t = q.next_block(&task_slot)) {
            const int n = kVitSizes[t];
            const int64_t off = (int64_t)kVitOffsets[t];
            // MUON: the 2D matrices were orthogonalized + applied in P2.7; P3 handles
            //   ONLY the 1D / non-2D weights, which take the AdamW tail (muon.h:75-76,
            //   the eager Muon auto-split). Skip the 2D ones here (already done).
            if constexpr (Opt == OptId::Muon) {
                if (vittc::vit_is_muon_2d(t)) continue;
            }
            FusedOptState ts = vit_rebase_state<Opt>(st, off);
            // (i) PER-TENSOR LAYER-WISE β1 (GrokAdamW only): β1_i = β1·(1-γ)^t,
            //     t == the tensor's flat named_parameters() layer index (the
            //     work-steal task id maps 1:1 to kVitOffsets order == the eager
            //     enumeration order — cls_token is t=0). bc1 must be rebased TOO
            //     (= 1-β1_i^step) or m_hat=m/bc1 mismatches eager; bc2 stays global.
            if constexpr (Opt == OptId::GrokAdamW) {
                const float b1 = st.beta1 * powf(1.0f - st.gamma, (float)t);
                ts.beta1 = b1;
                ts.bc1   = 1.0f - powf(b1, (float)step);
            }
            // SuperGrok11/15 meta-net mu (+ SG11 per-tensor gate) precompute for THIS
            // tensor, BEFORE the apply reads ts.mu/ts.gate (ViT twin of the decoder's).
            if constexpr (Opt == OptId::SuperGrok11) {
                const float b2 = (st.sg_phi_W2 != nullptr) ? st.sg_phi_W2[kSgPhiHidden]
                                                           : st.sg_phi_b2;
                // st.exp_avg = start-of-step momentum (base ptr, indexed by off+i) —
                // the gate's 2nd cosine operand: sigmoid(gate_temp·cos(grad,momentum)).
                // Precompute runs BEFORE the apply (sg11_sweep_b_step) writes exp_avg.
                const float g8 = sg11_precompute_mu_and_gate_for_tensor<kSgPhiHidden>(
                    st.mu, grad, st.sharpness, st.exp_avg, sg_sW1, sg_sb1, sg_sW2,
                    b2, st.sg_rescale, off, n, st.gate_temp);
                ts.gate = g8;            // per-tensor gate=sigmoid(gate_temp·cos) the apply tail reads
                __syncthreads();         // mu(T) fully written + gate broadcast before apply
            } else if constexpr (Opt == OptId::SuperGrok15) {
                const float b2 = (st.sg_phi_W2 != nullptr) ? st.sg_phi_W2[kSgPhiHidden]
                                                           : st.sg_phi_b2;
                sg15_precompute_mu_for_tensor<kSgPhiHidden>(
                    st.mu, grad, st.sharpness, sg_sW1, sg_sb1, sg_sW2,
                    b2, st.sg_rescale, off, n);
                // ts.gate stays = st.gate (the host sigmoid(accuracy) scalar).
                __syncthreads();         // mu(T) visible before the apply reads it
            }
            float* __restrict__ p = params + off;
            const float* __restrict__ gg = grad + off;
            if constexpr (Opt == OptId::Muon) {
                // 1D / non-2D weights: the canonical AdamW tail (muon.py:115-125 routes
                // non-matrix params to a SEPARATE AdamW group with INDEPENDENT
                // hyperparameters — adamw_lr/adamw_betas — NOT the 2D Muon lr/momentum).
                // ts.lr/ts.beta1 carry the 2D-group's (lr=0.02, momentum=0.95), so the
                // 1D tail MUST instead use the aux_* fields (eager adamw_lr/adamw_betas).
                // weight_decay is SHARED across both eager groups (muon.py:122) → ts.wd
                // stays. eps is the eager adamw_eps (= ts.eps, mapped by _opt_scalars_from).
                // bc1/bc2 are device-computed here from aux_beta^step (kept out of the
                // mirror to shrink the host ABI surface; fp32 powf, tol 2e-3 is ample).
                // apply_optimizer<Muon> would deref st.orth (NS dir, only valid for 2D),
                // so route directly to adamw_step with the aux hyperparameters.
                const float a_bc1 = 1.0f - powf(st.aux_beta1, (float)step);
                const float a_bc2 = 1.0f - powf(st.aux_beta2, (float)step);
                for (int i = threadIdx.x; i < n; i += blockDim.x)
                    algo::adamw_step<float, float>(
                        p, ts.exp_avg, ts.exp_avg_sq, gg,
                        st.aux_lr, st.aux_beta1, st.aux_beta2, ts.eps, ts.wd,
                        a_bc1, a_bc2, (int64_t)i);
            } else {
                for (int i = threadIdx.x; i < n; i += blockDim.x)
                    apply_optimizer<Opt>(p, gg, (int64_t)i, step, ts);
            }
        }
    }
#ifdef SG_VIT_PROFILE
    if (threadIdx.x == 0) { unsigned long long _p3b = clock64(); atomicMax(&g_vit_prof_max[5], _p3b - _p3a); }
#endif
}

// ncta_cap (default 0): if >0, launch min(n_sms, ncta_cap) CTAs instead of one
// per SM. The grid barrier rendezvous is over the LAUNCHED count (ctx.n_ctas);
// any cap is hang-safe as long as the workspace + dW-tile/cls/pos ownership use
// the SAME nCTA (they read ctx.n_ctas). The shipped path passes 0 (full
// saturation); a memory-constrained TEST passes a small cap so the per-CTA
// scratch (nCTA × slab) fits. Determinism is preserved per fixed nCTA.
template <OptId Opt>
cudaError_t launch_fused_vit_megakernel_tc(
        PersistentContext ctx, float* params, ViTInputCtx in,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream,
        int ncta_cap = 0) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 0;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;

    // Hang-freedom: certify one block/SM occupancy with the ACTUAL dynamic smem
    // (0 here — VitTcSmem is static). If the static smem + 200-reg consumer can't
    // place one block/SM, REFUSE (the grid barrier would hang).
    int occ = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (const void*)&fused_vit_megakernel_tc<Opt>, SG_VIT_TC_MEGA_BLOCK,
        /*dynamicSMemBytes=*/0);
    if (err != cudaSuccess) return err;
    if (occ < 1) return cudaErrorLaunchOutOfResources;

    unsigned launch_ctas = (unsigned)n_sms;
    if (ncta_cap > 0 && (unsigned)ncta_cap < launch_ctas) launch_ctas = (unsigned)ncta_cap;
    ctx.n_ctas = launch_ctas;
    // B%16 required (the dW K-loop contracts K=T=B*kSeq and K=B in 16-step atoms).
    if ((in.B % 16) != 0) return cudaErrorInvalidValue;

    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }

    dim3 grid(launch_ctas), block(SG_VIT_TC_MEGA_BLOCK);
    fused_vit_megakernel_tc<Opt><<<grid, block, 0, stream>>>(
        ctx, params, in, grad, lr, step, st);
    return cudaGetLastError();
}

#endif  // SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_FUSED_VIT_MEGAKERNEL_CUH_
