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
#endif

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

// ── smem arena for the unpipelined engine: one A(64×16) + one B(N×16) bf16 tile
//    + the 256-float reduction slot + the 10 dW specs (identical for all threads
//    → shared, not per-thread stack). 16B-aligned. N = SG_TUNED_TILE_N. ──
struct VitTcSmem {
    __nv_bfloat16 sA[64 * 16];
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
__host__ __device__ __forceinline__ int64_t vit_tc_workspace_floats(int T, int B, int nCTA) {
    return vit_tc_acts_floats(T, B)
         + (int64_t)nCTA * vittc::vit_tile_scratch_total_f32()
         + (int64_t)nCTA * vittc::kLnVecElems
         + nCTA + 1;
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

    vittc::VitActs acts = vittc::vit_acts_bind(acts_base, T, B);
    vittc::VitTileScratch sc = vittc::vit_tile_scratch_bind(scratch_base + (int64_t)cta * scratch_per);
    float* my_lnvec = lnvec_base + (int64_t)cta * vittc::kLnVecElems;

    VitWeights w = vit_bind(params);

    // ── P0: zero this CTA's LN-vec partials + loss slot (dW/cls/pos/patch grads
    //    are written-once → no pre-zero). ──
    for (int i = threadIdx.x; i < vittc::kLnVecElems; i += blockDim.x) my_lnvec[i] = 0.0f;
    if (threadIdx.x == 0) loss_part[cta] = 0.0f;
    bar.sync();   // B0

    // ── P1: token-tile-parallel fwd+bwd. Each CTA grid-strides over tiles of
    //    kTileM rows; for its tile it runs fwd (→ acts X, NLL) then bwd (→ acts
    //    dY, dh0, LN-vec partials). Barrier-free within the tile. ──
    const int nrows_tile = vittc::kTileM;
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
    for (int gt = cta; gt < n_dw; gt += nCTA)
        vittc::vittc_dw_run_tile<SG_TUNED_TILE_N>(spec, gt, acts.dh0, grad, sm.sA, sm.sB);
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
            const FusedOptState ts = vit_rebase_state<Opt>(st, off);
            float* __restrict__ p = params + off;
            const float* __restrict__ gg = grad + off;
            for (int i = threadIdx.x; i < n; i += blockDim.x)
                apply_optimizer<Opt>(p, gg, (int64_t)i, step, ts);
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
