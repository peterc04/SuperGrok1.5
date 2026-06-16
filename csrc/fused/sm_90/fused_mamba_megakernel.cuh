#ifndef SG_FUSED_SM90_FUSED_MAMBA_MEGAKERNEL_CUH_
#define SG_FUSED_SM90_FUSED_MAMBA_MEGAKERNEL_CUH_
// ============================================================================
// csrc/fused/sm_90/fused_mamba_megakernel.cuh — PHASE 2 of the TRUE L3 fused
// megakernel. ONE persistent kernel per training step runs the REAL Mamba
// (selective-SSM) forward+backward AND the optimizer math, separated only by
// in-kernel grid barriers — real model math, real optimizer math, ZERO
// intermediate kernel launches. The Mamba counterpart of PHASE 1's
// fused_decoder_megakernel.cuh (and the PHASE-2 fused_vit_megakernel.cuh).
//
// This composes:
//   * the REAL Mamba fwd/bwd stages (model_stage_mamba3.cuh — transcribed
//     line-for-line from the verified PyTorch oracle, asserted bit-identical to
//     autograd INCLUDING the selective-scan reverse-time backward, and
//     structurally mirrored on CPU),
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
//       broadcasting the sample's kSeq int tokens + target into smem, running
//       mb_forward_sample + mb_backward_sample, accumulating each sample's
//       weight-grad into the CTA's partial with a single-owner-thread-per-element
//       rule (no atomics → deterministic), and summing its slice's NLL (fp32).
//   --- grid barrier B1 ---
//   P2  DETERMINISTIC cross-CTA reduce: sum partial[0..nCTA) in ASCENDING CTA
//       index into the global grad (no float atomics; order fixed → reuses the
//       work-steal queue to pick WHO reduces which tensor). Loss: fp64 ordered
//       sum → loss/B → a device float the host reads back (by CTA 0).
//   --- grid barrier B2 (sync_reset: also resets the queue for P3) ---
//   P3  the REAL apply_optimizer<Opt> tail consumes the reduced grad in place.
//
// DYNAMIC SHARED MEMORY (the same thing the ViT path did; the decoder did NOT
// need it): Mamba's per-sample MambaSampleSmem is ≈ 145124 B (≈141.72 KB; both
// layers' forward activations cached, d_inner=256), which CANNOT be a static
// __shared__ (48 KB cap). So this kernel declares `extern __shared__` and the
// LAUNCHER must, all three of: (1) cudaFuncSetAttribute(kernel,
// cudaFuncAttributeMaxDynamicSharedMemorySize, kMambaSmemBytes); (2) pass
// dynamicSMemBytes=kMambaSmemBytes to cudaOccupancyMaxActiveBlocksPerMSM (so the
// occ≥1 hang-freedom check is honest); (3) pass it at <<<...>>>. Missing ANY of
// these makes the kernel silently fail to launch — see the launcher. The scan
// state stays in per-thread REGISTERS (the seq=8 exploit), so smem does NOT
// explode to the ~128 KB an h-in-smem scan would need — see model_stage_mamba3.cuh.
//
// HONESTY: no placeholder math anywhere on this path. fp32 compute is the
// correctness baseline; a bf16-compute follow-up would be a flag defaulting to
// THIS fp32 path (not yet wired — see the TODO(bf16) in model_stage_mamba3.cuh).
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/opt_components.cuh"
// STAGED-optimizer in-kernel precompute stages (Prodigy d-reduction). Pulls in the
// canonical prodigy.h reduction math + the deterministic per-CTA owner-computes tree;
// the TC megakernel's Prodigy branch (P2.6) runs an INLINE FIXED-PARTITION (r,s) reduce
// + an EMA-decay/d-update owner block, BYTE-FAITHFUL to the eager multi-tensor estimator
// (the work-steal prodigy_precompute_reduce_phaseA/B helpers in opt_stages_precompute.cuh
// are the older form). NB: mamba×prodigy is still BLOCKED on a SEPARATE mamba scan/forward
// determinism issue (see dispatch.cpp's carve-out), so this branch is landed-dormant.
// Header-only, self-contained per COMPONENT_CONTRACT (substrate + algorithm headers).
#include "csrc/fused/sm_90/opt_stages_precompute.cuh"
#include "csrc/fused/sm_90/mamba3_layout.cuh"
#include "csrc/fused/sm_90/model_stage_mamba3.cuh"
// NOTE: warp_specialize.cuh (warpgroup_reg_alloc/dealloc) is intentionally NOT
// included — the Mamba megakernel does NOT do the decoder/vit producer/consumer
// register repartition (the symmetric register-resident scan needs the uniform
// __launch_bounds__ budget; see the P0 comment in fused_mamba_megakernel below).

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

// ── GEMM-impl seam (the decoder/vit pattern, owner directive: BOTH paths
//    compiled, the tuner picks). The SCALAR default body above is the live path
//    and its gates stay bit-identical; selecting the wgmma token pulls in the
//    Fork-B tensor-core CELL DRIVER (model_stage_mamba_tc.cuh + the _tc kernel/
//    launcher at the bottom). This is a PARALLEL kernel — nothing here edits the
//    scalar path. ──
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
#include "csrc/fused/sm_90/model_stage_mamba_tc.cuh"
// SuperGrok2 FULL CSA/HCA/PEER/GRU meta-net stages (composed as the optimizer phase)
// — the mamba twin of the decoder/vit SG2 include. Pulls in sg2_meta_stages + the
// in-kernel segmented sort (STAGE -1) + SG2Dims/SG2State/SG2Scalars/SG2Weights. Only
// under the wgmma token. The SG2 weight bundle is read from HBM (it does NOT fit smem
// alongside MbTcSmem); the composed path instantiates sg2_meta_stages with
// WeightsT==SG2Weights + BuildSort=true. The SG11/15 phi helpers ride
// opt_stages_precompute.cuh (already included above).
#include "csrc/fused/sm_90/opt_stage_supergrok2.cuh"
#endif

// ============================================================================
//  SG_MB_SCALAR_MEGAKERNEL — compile-gate for the LEGACY fp32 SCALAR Mamba
//  megakernel (fused_mamba_megakernel<Opt> + launch_fused_mamba_megakernel<Opt>
//  below, which drive the per-sample MambaSampleSmem path in model_stage_mamba3.cuh).
//  The mamba twin of the decoder's SG_DEC_SCALAR_MEGAKERNEL gate (commit 555c0bc).
//
//  WHY A GATE (the d-scaling decouple): MambaSampleSmem is sized [kSeq×kD]/[kSeq×
//  kDInner] arrays, so its DYNAMIC smem GROWS with SG_MB_D and at the d-scaled bench
//  width (d=1024) it is ≈ 1.1 MB — far over the sm_90 ~228 KB/SM budget, so
//  cudaFuncSetAttribute(MaxDynamicSharedMemorySize) would REFUSE it and the scalar
//  kernel could never place one block/SM. The TC engine (fused_mamba_megakernel_tc,
//  below) uses a small d-INDEPENDENT static MbTcSmem and builds/launches at every d.
//  Gating the dead scalar kernel OFF (-DSG_MB_SCALAR_MEGAKERNEL=0) lets the WHOLE
//  extension compile at the bench width (the TC path is what's measured). DEFAULT 1
//  (ON) → the d=128 production build is byte-for-byte unchanged: the scalar fp32
//  fallback stays wired and the 33/33 wiring gate stays green. The flag changes
//  NOTHING the TC path touches.
#ifndef SG_MB_SCALAR_MEGAKERNEL
#define SG_MB_SCALAR_MEGAKERNEL 1
#endif

namespace sg { namespace fused { namespace sm90 {

#if SG_MB_SCALAR_MEGAKERNEL
// Compile-time guard: the byte budget the launcher uses (kMambaSmemBytes) MUST
// equal sizeof(MambaSampleSmem). model_stage_mamba3.cuh already static_asserts
// the same equality (it pins kMambaSmemFloats to the actual struct); we restate
// it here so this header is self-contained and a drift fails loudly at this TU.
static_assert((int64_t)sizeof(MambaSampleSmem) == kMambaSmemBytes,
              "fused_mamba_megakernel: sizeof(MambaSampleSmem) != the documented "
              "kMambaSmemBytes in mamba3_layout.cuh — update kMambaSmemFloats.");
#endif  // SG_MB_SCALAR_MEGAKERNEL

// Rebase a FusedOptState's per-element state pointers to a parameter-tensor slice
// at `off` within the flat [m|v|extra] layout. Per-TENSOR fields and all scalars
// pass through unchanged. (Identical to the decoder's rebase_state; redefined here
// under a mamba_ name so this header stays self-contained — the decoder file is
// owned by a sibling agent and must not be included.)
template <OptId Opt>
__device__ __forceinline__ FusedOptState
mamba_rebase_state(const FusedOptState& s, int64_t off) {
    FusedOptState t = s;  // scalars + per-tensor pointers copy as-is
    if (t.exp_avg)    t.exp_avg    += off;
    if (t.exp_avg_sq) t.exp_avg_sq += off;
    if (t.ema)        t.ema        += off;
    if (t.sam_dir)    t.sam_dir    += off;
    if (t.s_track)    t.s_track    += off;
    if (t.mu)         t.mu         += off;
    if (t.orth)       t.orth       += off;
    if (t.smart_grad) t.smart_grad += off;
    if (t.param_init) t.param_init += off;   // Prodigy trajectory anchor p0 (per-tensor
                                             // slice). prodigy_persist is a GLOBAL 3-
                                             // scalar [r_ema|s_ema|d_lr] — NOT rebased.
    if (t.sharpness)  t.sharpness  += off;   // SuperGrok11/15 (g_sam−g)² (per element;
                                             // DORMANT for mamba — gated out, no SG case).
    // SuperGrok11/15 phi weights are a per-TENSOR weight set (NOT rebased).
    return t;
}

// The L3-REAL Mamba megakernel needs the int TOKEN path + a grad-partial
// workspace. Kept a SEPARATE kernel + launcher (not folded into the surrogate
// fused_megakernel) so the surrogate path is untouched. MambaTokenCtx mirrors the
// decoder's DecoderTokenCtx (int tokens, NOT ViT's float patches).
//
// Workspace layout (one flat float buffer the host allocates + the kernel owns):
//   [0 .. nCTA*total)              : per-CTA grad partials (cta-major)
//   [nCTA*total .. nCTA*total+nCTA): per-CTA loss partials (NLL sum per slice)
//   [.. +1)                        : the reduced scalar loss (loss/B) the host reads
// total == kMambaTotalElems == 259425.
struct MambaTokenCtx {
    const int* tokens;   // [B, kSeq] int32 token ids in [0, kVocab)
    const int* targets;  // [B]       int32 target ids in [0, kPHead)
    int        B;        // batch size
    float*     workspace; // grad partials + loss partials + reduced loss
    float*     loss_out;  // device float the kernel writes the mean loss into
};

#if SG_MB_SCALAR_MEGAKERNEL
// ── The persistent megakernel (L3-REAL). gridDim.x = #SMs (one CTA/SM), 256
//    threads/CTA. MambaSampleSmem (~141.72 KB) lives in DYNAMIC smem (extern). ──
//    GATED by SG_MB_SCALAR_MEGAKERNEL (this fp32 path's MambaSampleSmem grows with
//    SG_MB_D and overflows the ~228 KB/SM budget at the bench width; OFF lets the TC
//    path build at scaled d — see the flag note above). ────────────────────────────
// sizes/offsets are NOT host-passed: per-tensor numel/offset live in the
// __constant__ tables kMambaSizes/kMambaOffsets (mamba3_layout.cuh), read directly
// by the reduce + optimizer phases.
template <OptId Opt>
__global__ void __launch_bounds__(SG_TUNED_MEGA_BLOCK)
fused_mamba_megakernel(PersistentContext ctx,
                       float* __restrict__ params,
                       MambaTokenCtx tok,
                       float* __restrict__ grad,        // reduced grad [total]
                       float lr, int step, FusedOptState st) {
    extern __shared__ char mamba_smem_raw[];
    MambaSampleSmem& sm = *reinterpret_cast<MambaSampleSmem*>(mamba_smem_raw);
    GridBarrier bar = ctx.barrier();
    const int cta = blockIdx.x;
    const int nCTA = (int)ctx.n_ctas;
    const int64_t total = kMambaTotalElems;
    float* my_partial = tok.workspace + (int64_t)cta * total;   // this CTA's dW
    float* loss_part = tok.workspace + (int64_t)nCTA * total;   // [nCTA]

    // NO warp-group register repartition here (UNLIKE the decoder/vit seams).
    // The decoder/vit copy a §3.4 producer/consumer setmaxnreg (dealloc<32> on
    // warpgroup 0, alloc<200> on warpgroup 1) — valid ONLY when warpgroup 0 needs
    // few registers. Mamba is NOT warp-specialized that way: ALL 256 threads do
    // SYMMETRIC selective-scan work, each thread owning a channel j∈[0,256) and
    // holding large register-resident scan state (fwd A[16]+h[16]; bwd adds
    // hh[kSeq+1][kState]+a_save[kSeq][kState]+dB_loc/dC_loc[kSeq][kState]+gh+dA),
    // so a dealloc<32> would spill it. __launch_bounds__(SG_TUNED_MEGA_BLOCK) sets the uniform
    // budget; the compiler allocates symmetrically across all 256 threads.

    // ── P0: zero this CTA's grad-partial slice + its loss slot. ───────────────
    for (int64_t i = threadIdx.x; i < total; i += blockDim.x) my_partial[i] = 0.0f;
    if (threadIdx.x == 0) loss_part[cta] = 0.0f;
    bar.sync();   // B0

    // ── P1: batch-parallel fwd+bwd. Fixed contiguous slice [b0,b1) for this CTA.
    MambaWeights w = mb_bind(params);
    MambaGrad g = mb_bind_grad(my_partial);
    const int B = tok.B;
    const int base = B / nCTA, rem = B % nCTA;
    const int b0 = cta * base + (cta < rem ? cta : rem);
    const int cnt = base + (cta < rem ? 1 : 0);
    const int b1 = b0 + cnt;
    __shared__ int tok_s[mb::kSeq];   // this sample's token ids (broadcast)
    __shared__ int tgt_s;
    float nll_acc = 0.0f;              // fp32 slice accumulator (thread-0 holds it)
    for (int b = b0; b < b1; ++b) {
        if (threadIdx.x < mb::kSeq) tok_s[threadIdx.x] = tok.tokens[(int64_t)b * mb::kSeq + threadIdx.x];
        if (threadIdx.x == 0) tgt_s = tok.targets[b];
        __syncthreads();
        float nll = mb_forward_sample(w, tok_s, tgt_s, &sm);
        mb_backward_sample(w, g, tok_s, tgt_s, B, &sm);
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
        for (int t = q.next_block(&task_slot); t < kMambaNumTensors;
             t = q.next_block(&task_slot)) {
            const int n = kMambaSizes[t];
            const int64_t off = (int64_t)kMambaOffsets[t];
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                float acc = 0.0f;
                for (int c = 0; c < nCTA; ++c)
                    acc += tok.workspace[(int64_t)c * total + off + i];
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
        *tok.loss_out = mean;
    }
    bar.sync_reset(ctx.g_next_task);   // B2: reduced grad ready; reset queue for P3

    // ── P3: the REAL optimizer tail over the reduced grad (work-steal the 28
    //    tensors). apply_optimizer<Opt> is the canonical csrc/algorithms math. ──
    st.lr = lr;
    {
        __shared__ int task_slot;
        TaskQueue q = ctx.queue();
        for (int t = q.next_block(&task_slot); t < kMambaNumTensors;
             t = q.next_block(&task_slot)) {
            const int n = kMambaSizes[t];
            const int64_t off = (int64_t)kMambaOffsets[t];
            const FusedOptState ts = mamba_rebase_state<Opt>(st, off);
            float* __restrict__ p = params + off;
            const float* __restrict__ gg = grad + off;
            for (int i = threadIdx.x; i < n; i += blockDim.x)
                apply_optimizer<Opt>(p, gg, (int64_t)i, step, ts);
        }
    }
}

// ── Host launcher — one persistent CTA per SM, 256 threads/CTA (2 warp-groups).
//    Mirrors launch_fused_decoder_megakernel's hang-freedom contract (occupancy≥1
//    or refuse), PLUS the DYNAMIC-SMEM opt-in that the Mamba footprint requires
//    (exactly as launch_fused_vit_megakernel does):
//      (1) cudaFuncSetAttribute(MaxDynamicSharedMemorySize, kMambaSmemBytes)
//      (2) dynamicSMemBytes = kMambaSmemBytes in the occ≥1 query
//      (3) dynamicSMemBytes = kMambaSmemBytes at <<<...>>>
//    All three are mandatory; missing any one makes the launch silently fail. ───
template <OptId Opt>
cudaError_t launch_fused_mamba_megakernel(
        PersistentContext ctx, float* params, MambaTokenCtx tok,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 0;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;

    const int dyn_smem = (int)kMambaSmemBytes;   // 145124 B (≈141.72 KB)

    // (1) Opt in to >48 KB dynamic smem for THIS kernel. Without this the launch
    //     fails with cudaErrorInvalidValue (the static 48 KB default applies).
    err = cudaFuncSetAttribute(
        (const void*)&fused_mamba_megakernel<Opt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
    if (err != cudaSuccess) return err;

    // (2) Occupancy with the REAL dynamic-smem request (hang-freedom is honest).
    int occ = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (const void*)&fused_mamba_megakernel<Opt>, SG_TUNED_MEGA_BLOCK,
        /*dynamicSMemBytes=*/dyn_smem);
    if (err != cudaSuccess) return err;
    // At least one CTA per SM must be resident or the grid barrier can never be
    // satisfied. ~141.72 KB dynamic smem fits the sm_90 ~227 KB cap at occ=1 (the
    // persistent megakernel is one-CTA-per-SM by design); if it cannot place one
    // block/SM, REFUSE rather than hang.
    assert(occ >= 1 &&
           "fused_mamba_megakernel: 0 blocks/SM — GridBarrier would hang. The "
           "Mamba per-sample smem (~142KB dynamic) + regs exceed one-block-per-SM "
           "occupancy on this device; reduce footprint or fall back to eager.");
    if (occ < 1) return cudaErrorLaunchOutOfResources;

    const unsigned launch_ctas = (unsigned)n_sms;
    ctx.n_ctas = launch_ctas;

    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }

    // (3) Launch WITH the dynamic-smem byte count.
    dim3 grid(launch_ctas), block(SG_TUNED_MEGA_BLOCK);
    fused_mamba_megakernel<Opt><<<grid, block, dyn_smem, stream>>>(
        ctx, params, tok, grad, lr, step, st);
    return cudaGetLastError();
}
#endif  // SG_MB_SCALAR_MEGAKERNEL

// ════════════════════════════════════════════════════════════════════════════
//  WGMMA CELL DRIVER (Fork B, R2 Mamba TC). Compiled only under the wgmma token;
//  the scalar path above is untouched. UNLIKE the scalar Mamba megakernel, this
//  uses STATIC smem (the per-(layer,sample) activation cache lives in HBM tile-
//  scratch, NOT smem — so NO dynamic-smem opt-in; the decoder TC launcher's
//  contract, not the scalar mamba launcher's three-step opt-in).
// ════════════════════════════════════════════════════════════════════════════
#if (SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA)

// TC megakernel threads: 256 (the scan owns channels 0..255; the wgmma is
// warpgroup-scoped on threads 0..127). No asymmetric setmaxnreg.
#define SG_MB_TC_MEGA_BLOCK 256

// Min co-resident blocks/SM for __launch_bounds__. DEFAULT 1 = the shipped
// register-maximal (255 reg/thread → occ=1) one-CTA-per-SM config — which the
// PersistentContext GridBarrier (megakernel_common.cuh §1.4) is DESIGNED for
// ("occupancy=1 IS the design point", INTEGRATION-MAMBA §2). Setting 2 caps regs to
// ~128 so TWO CTAs co-reside per SM (occupancy-fill launches occ·n_sms CTAs below),
// which MEASURED 1.14× faster (B=16384 quiet, 113→99ms; the projection GEMMs are
// 0.1%-of-wall so the register trade is nearly free) BUT IS DETERMINISM-UNSAFE: the
// A/A/A gate proves occ=2 grads are NOT bit-identical across runs (max|Δ|=2.3e-10,
// timing-dependent), while occ=1 is exactly bit-identical — the hand-built grid
// barrier's two-fence visibility model assumes one CTA per SM, and co-residency
// exposes a P1-partial→P2-reduce visibility race. The barrier substrate fix lives in
// megakernel_common.cuh (NOT this lane); until it lands, occ=1 is mandatory (the
// no-suppression determinism gate is load-bearing). 0 → omit the bound (ptxas chooses).
#ifndef SG_MBTC_MIN_BLOCKS_PER_SM
#define SG_MBTC_MIN_BLOCKS_PER_SM 1
#endif
#if SG_MBTC_MIN_BLOCKS_PER_SM >= 1
#define SG_MBTC_LAUNCH_BOUNDS __launch_bounds__(SG_MB_TC_MEGA_BLOCK, SG_MBTC_MIN_BLOCKS_PER_SM)
#else
#define SG_MBTC_LAUNCH_BOUNDS __launch_bounds__(SG_MB_TC_MEGA_BLOCK)
#endif

// ── PHASE PROFILER (default OFF; a separate measurement TU defines
//    SG_MB_TC_PROFILE to bracket P1-fwd/P1-bwd/P2/P3 with clock64() and write
//    per-CTA cumulative cycles to a profiling buffer carved from the workspace
//    tail). Production builds never define it → zero added code, zero ABI change.
#ifndef SG_MB_TC_PROFILE
#define SG_MB_TC_PROFILE 0
#endif
#if SG_MB_TC_PROFILE
#define SG_MBTC_PROF_SLOTS 8   // [p1fwd,p1bwd,p2,p3, p2-dwgemm,p2-reduce,p2-embed, witness]
#define SG_MBTC_PROF_TIC(v) do { __syncthreads(); if (threadIdx.x == 0) (v) = clock64(); } while (0)
#define SG_MBTC_PROF_ACC(slot, t0) do { __syncthreads(); \
    if (threadIdx.x == 0) prof[(int64_t)cta * SG_MBTC_PROF_SLOTS + (slot)] += (double)(clock64() - (t0)); } while (0)
#else
#define SG_MBTC_PROF_TIC(v) do {} while (0)
#define SG_MBTC_PROF_ACC(slot, t0) do {} while (0)
#endif

// ── Static smem arena. The M-atom-interleaved wgmma pipeline (task #24, decoder
//    H1+H3 port) stages kMbAtomsPerSlot A(64×16) tiles per ring slot (one per
//    stacked m64 atom in an interleave GROUP, issued back-to-back into independent
//    fp32 fragments so the tensor pipe overlaps them; slot s, atom-in-group ai at
//    sA + (s*kMbAtomsPerSlot + ai)*64*16) + ONE shared B(N×16) tile per slot (slot s
//    at sB + s*N*16). kMbAtomsPerSlot = min(kAtomsM, kMbMaxIL) (fwd/dX = 2; dW = 2).
//    Plus the 256-float reduction slot + the scan-bwd cross-channel reduce targets
//    (dBmat/dCmat, [kSeq×kState] each) + the 8 dW specs (shared, not per-thread
//    stack). At S=2 + =2 + N=128 the ring is 2·(2·2KB+4KB)=16KB; MbTcSmem total
//    ~16.6KB ≪ the 48KB static cap (the TC launcher uses static smem). ──
struct MbTcSmem {
    __nv_bfloat16 sA[mbtc::kMbTcStages * mbtc::kMbAtomsPerSlot * 64 * 16];
    __nv_bfloat16 sB[mbtc::kMbTcStages * SG_TUNED_TILE_N * 16];
    float red[256];
    float dBmat[mb::kSeq * mb::kState];
    float dCmat[mb::kSeq * mb::kState];
    mbtc::MbDwSpec spec[8];
};

// TC workspace layout (carved from tok.workspace; host sizes it):
//   [0 .. acts_f)                          : MbActs bf16 region
//   [acts_f .. + nCTA*scratch)             : per-CTA tile scratch (f32)
//   [.. + nCTA*kPartElems)                 : per-CTA non-GEMM grad partials (f32)
//   [.. + nCTA)                            : per-CTA loss slots
//   [.. + 1)                               : reduced scalar loss
//   [.. + mb_dw_part_floats(G))            : split-K dW partials (G>1 only; the
//                                            (gt,kc) 64×N partial tiles P2 reduces)
__host__ __device__ __forceinline__ int64_t mb_tc_acts_floats(int T) {
    return mbtc::mb_acts_floats(T);
}
// dW split-K partial floats (0 when G==1 → no extra scratch, the single-CTA path).
__host__ __device__ __forceinline__ int64_t mb_tc_dw_part_floats() {
    return (mbtc::kMbDwSplitK > 1) ? mbtc::mb_dw_part_floats(mbtc::kMbDwSplitK) : 0;
}
// STAGED-optimizer cross-CTA reduction scratch (Prodigy d-estimate). Mirrors the
// decoder's dec_tc_opt_reduce_floats: the Prodigy P2.6 stage publishes per-CTA (r,s)
// slots (2*nCTA) + a reduced-d broadcast slot (1) — an owner-computes tree
// (opt_stages_precompute.cuh), NO float atomic. Sized for the LARGEST nCTA (one
// CTA/SM = #SMs); tiny (≤ 2*132+1 ≈ 1 KB) and carved UNCONDITIONALLY so the opt-
// agnostic cached launcher workspace fits every OptId. Unused by AdamW/Lion/… (their
// P3 never touches this region), so adding it leaves those cells byte-identical.
__host__ __device__ __forceinline__ int64_t mb_tc_opt_reduce_floats(int nCTA) {
    return (int64_t)2 * nCTA + 1;            // [r slots | s slots | reduced d]
}
// ── LookSAM (STAGED in-kernel SAM 2nd backward) transient scratch — the Mamba twin of
//    the decoder's dec_tc_looksam_floats. The SAM step perturbs p in place, runs a
//    SECOND fwd+bwd at p', and needs (a) a param BACKUP for the exact restore (eager
//    clones p before perturbing; an fp32 add is not bit-reversible, so save — don't
//    subtract back) + (b) a SEPARATE g_sam grad buffer so the 2nd backward does NOT
//    clobber the first grad `grad` (sam_dir = g_sam − grad reads both). Both are total-
//    sized + transient (NOT persistent state — sam_dir itself persists in the `extra`
//    state slice, bound by the cell). Carved UNCONDITIONALLY (≈ 2·kMambaTotalElems)
//    so the opt-agnostic launcher fits every OptId; unused by every non-LookSAM cell
//    (its P2.4 phase is if-constexpr'd out → byte-identical). The ‖g‖ reduction reuses
//    the loss workspace (sq_part/ron_bc, as GrokAdamW's P2.5 / the decoder twin do), so
//    no extra here. ──
__host__ __device__ __forceinline__ int64_t mb_tc_looksam_floats() {
    return (int64_t)2 * kMambaTotalElems;    // [sam_backup | sam_grad]
}
// ── Muon (STAGED grid-cooperative Newton-Schulz) per-matrix scratch — the Mamba twin of
//    the decoder's dec_tc_muon_floats / the vit's vit_tc_muon_floats. The NS chain runs ONE
//    2D weight at a time over all CTAs; the scratch holds X/A/AX/AAX/orth for the LARGEST 2D
//    weight + the per-CTA Frobenius-norm partials + inv_norm. The momentum buffer (muon_buf)
//    is NOT here — it PERSISTS across steps as optimizer state, bound to the m slice
//    (st.exp_avg). Largest mamba 2D weight: in_proj = 512×128 = 65536 numel; largest rows =
//    512 ⇒ A = 512×512 (mbtc::kMbMuonMaxNumel/kMbMuonMaxRows). Carved UNCONDITIONALLY (≈
//    4·65536 + 512² + nCTA + 1 floats ≈ 2 MB) so the opt-agnostic cached launcher workspace
//    fits every OptId; unused by every non-Muon cell (its P2.7/P3-2D branches are if-
//    constexpr'd out → byte-identical). ──
__host__ __device__ __forceinline__ int64_t mb_tc_muon_floats(int nCTA) {
    // X + AX + AAX + orth (each maxNumel) + A (maxRows²) + nrm_partials(nCTA) + inv_norm(1)
    return (int64_t)4 * mbtc::kMbMuonMaxNumel
         + (int64_t)mbtc::kMbMuonMaxRows * mbtc::kMbMuonMaxRows
         + nCTA + 1;
}
// ── SuperGrok2 compile-time dims = the race config (== SG2Dims defaults). The mamba
//    twin of dec_sg2_ws_stride_floats / dec_tc_sg2_floats. The composed megakernel
//    reads the meta-net weights from HBM, so the only SG2 workspace is the per-CTA
//    meta-net scratch (sg2_ws_stride, sized for the LARGEST tensor — the 65536-element
//    in_proj weight — which bounds every tensor's intermediates + the in-kernel
//    segmented-sort key/idx/perm/unsort), PLUS a row_off64 staging prefix
//    (kMambaNumTensors int64 = 2*N floats — the adapter that builds SG2State.row_off
//    (const int64_t*) from the __constant__ int kMambaOffsets). Carved UNCONDITIONALLY
//    (carve-LAST) so the opt-agnostic cached launcher workspace fits every OptId;
//    unused by every non-SG2 cell (its P3-SG2 phase is if-constexpr'd out →
//    byte-identical). ──
using MbSG2Dims = SG2Dims<>;
// max(kMambaSizes), re-derived from the layout table (mamba3_layout.cuh) so the SAME
// SG2 tail is correctly sized at ANY ladder width — was a d=128-pinned 65536 (in_proj
// [512,128] at d=128, the per-CTA carve bound).
constexpr int kMbSG2Nmax = kMambaMaxTensorNumel;   // == mamba_layout_check::max_size()
__host__ __device__ __forceinline__ int64_t mb_sg2_ws_stride_floats() {
    return (int64_t)2 * kMambaNumTensors
         + sg2_ws_stride<MbSG2Dims>((int64_t)kMbSG2Nmax);
}
__host__ __device__ __forceinline__ int64_t mb_tc_sg2_floats(int nCTA) {
    // +1 for the 8-byte realignment slack of sg2_ws_base (kMambaTotalElems is odd, so
    // the per-CTA int64 row_off64 prefix may need a +1 bump to land 8-byte aligned).
    return (int64_t)nCTA * mb_sg2_ws_stride_floats() + 1;
}
__host__ __device__ __forceinline__ int64_t mb_tc_workspace_floats(int T, int nCTA) {
    return mb_tc_acts_floats(T)
         + (int64_t)nCTA * mbtc::mb_tile_scratch_floats()
         + (int64_t)nCTA * mbtc::kPartElems
         + nCTA + 1
         + mb_tc_dw_part_floats()                       // split-K dW partials (G>1)
         + mb_tc_opt_reduce_floats(nCTA)                // STAGED-opt (Prodigy) reduce slots
         + mb_tc_looksam_floats()                       // STAGED-opt (LookSAM) SAM 2nd-bwd scratch
         + mb_tc_muon_floats(nCTA)                      // STAGED-opt (Muon) NS per-matrix scratch
         + mb_tc_sg2_floats(nCTA)                       // SuperGrok2 meta-net per-CTA scratch (carve-LAST)
#if SG_MB_TC_PROFILE
         + (int64_t)nCTA * SG_MBTC_PROF_SLOTS * 2 + 2  // phase-profiler (doubles=2 floats) + align pad
#endif
         ;
}

template <OptId Opt>
__global__ void SG_MBTC_LAUNCH_BOUNDS
fused_mamba_megakernel_tc(PersistentContext ctx,
                          float* __restrict__ params,
                          MambaTokenCtx tok,
                          float* __restrict__ grad,
                          float lr, int step, FusedOptState st) {
    __shared__ MbTcSmem sm;
    GridBarrier bar = ctx.barrier();
    // SAM-coupled cells run the tile fwd+bwd TWICE (P1 + P2.4); ONLY they route
    // through the out-of-line shims (one shared frame). Single-pass cells now
    // use the inline bodies (the original noinline was unconditional and taxed
    // them ~5.7 KB of callee spill; campaign C2 scopes it).
    constexpr bool kSamCoupled = (Opt == OptId::LookSAM || Opt == OptId::SuperGrok11 ||
                                  Opt == OptId::SuperGrok15 || Opt == OptId::SuperGrok2);
    const int cta = blockIdx.x;
    const int nCTA = (int)ctx.n_ctas;
    const int B = tok.B;
    const int T = B * mb::kSeq;

    // Workspace partition.
    float* ws = tok.workspace;
    const int64_t acts_f = mb_tc_acts_floats(T);
    __nv_bfloat16* acts_base = reinterpret_cast<__nv_bfloat16*>(ws);
    float* scratch_base = ws + acts_f;
    const int64_t scratch_per = mbtc::mb_tile_scratch_floats();
    float* part_base = scratch_base + (int64_t)nCTA * scratch_per;
    float* loss_part = part_base + (int64_t)nCTA * mbtc::kPartElems;
    float* loss_out  = loss_part + nCTA;
    // Split-K dW partials (G>1): the (gt,kc) 64×N partial tiles, carved AFTER the
    // loss slot (matches mb_tc_workspace_floats's term order). G==1 → dw_part unused.
    float* dw_part   = loss_out + 1;
    const int kDwG   = mbtc::kMbDwSplitK;
    // STAGED-opt (Prodigy P2.6) cross-CTA reduce slots: [r slots | s slots | reduced d]
    // = 2*nCTA+1, carved AFTER the split-K dW partials (matches mb_tc_workspace_floats's
    // term order). Unused by every non-Prodigy tail (their P3 never touches it), so this
    // pointer derivation leaves those cells byte-identical.
    float* opt_reduce = dw_part + mb_tc_dw_part_floats();
    // LookSAM SAM 2nd-backward scratch, carved AFTER the Prodigy reduce slots (mamba has
    // NO Muon region — unlike the decoder/vit twins — so this follows opt_reduce; term
    // order matches mb_tc_workspace_floats). Carving it here keeps every prior region's
    // offset unchanged, so the already-green cells are byte-identical. Layout:
    //   [sam_backup (total)] [sam_grad (total)]. Unused unless Opt==LookSAM.
    float* sam_backup = opt_reduce + mb_tc_opt_reduce_floats(nCTA);
    float* sam_grad   = sam_backup + kMambaTotalElems;
    // Muon NS per-matrix scratch, carved AFTER the LookSAM sam scratch (term order matches
    // mb_tc_workspace_floats — carving it LAST [before the optional profiler] keeps every
    // prior region's offset unchanged, so the already-green mamba cells are byte-identical).
    // Unused unless Opt==Muon. Layout (mirrors the decoder/vit twins):
    //   [muon_X | muon_AX | muon_AAX | muon_orth] (each kMbMuonMaxNumel)
    //   [muon_A (kMbMuonMaxRows²)] [nrm_partials (nCTA)] [inv_norm (1)]
    float* muon_base  = sam_grad + kMambaTotalElems;
    // SuperGrok2 meta-net per-CTA scratch, carved AFTER the Muon NS scratch (term order
    // matches mb_tc_workspace_floats — carving it LAST [before the optional profiler]
    // keeps every prior region's offset unchanged, so the already-green mamba cells are
    // byte-identical). Each CTA owns mb_sg2_ws_stride_floats() (row_off64 staging +
    // meta-net scratch). ALIGN to 8 bytes (even float offset): the per-CTA slice fronts
    // an int64 row_off64 staging array; kMambaTotalElems is odd so sam_grad+total lands
    // on an odd float offset (4-byte) → an int64 read there is misaligned. Round up to
    // the next even float (the carve reserves +1 slack). Unused unless Opt==SuperGrok2.
    float* sg2_ws_base = muon_base + mb_tc_muon_floats(nCTA);
    if (((uintptr_t)sg2_ws_base & 0x7) != 0) sg2_ws_base += 1;   // → 8-byte aligned
    (void)sg2_ws_base;   // referenced only by the SuperGrok2 P3-SG2 phase
#if SG_MB_TC_PROFILE
    // 8-byte align the double accumulator: round the float offset up to an even
    // count so reinterpret_cast<double*> is aligned (past the SG2 scratch — the
    // phase-profiler is the LAST workspace term, mirroring mb_tc_workspace_floats).
    float* prof_f = sg2_ws_base + mb_tc_sg2_floats(nCTA);
    uintptr_t _pa = reinterpret_cast<uintptr_t>(prof_f);
    if (_pa & 0x7u) prof_f = reinterpret_cast<float*>((_pa + 7u) & ~uintptr_t(7u));
    double* prof = reinterpret_cast<double*>(prof_f);   // [nCTA*SLOTS], host-zeroed
    long long _pt = 0;
#endif

    mbtc::MbActs acts = mbtc::mb_acts_bind(acts_base, T);
    mbtc::MbTileScratch sc = mbtc::mb_tile_scratch_bind(scratch_base + (int64_t)cta * scratch_per);
    float* my_part = part_base + (int64_t)cta * mbtc::kPartElems;
    mbtc::MbPartial part = mbtc::mb_partial_bind(my_part);

    MambaWeights w = mb_bind(params);

    // ── P0: zero this CTA's non-GEMM grad partial + loss slot. ──
    for (int64_t i = threadIdx.x; i < mbtc::kPartElems; i += blockDim.x) my_part[i] = 0.0f;
    if (threadIdx.x == 0) loss_part[cta] = 0.0f;
    bar.sync();   // B0

    // ── P1: sample-tile-parallel fwd+bwd. Each CTA grid-strides over tiles of
    //    kTileM rows; fwd (→ acts X, NLL) then bwd (→ acts dY, dh0, partial). ──
    const int nrows_tile = mbtc::kTileM;
    const int n_tiles = (T + nrows_tile - 1) / nrows_tile;
    float nll_acc = 0.0f;
    for (int ti = cta; ti < n_tiles; ti += nCTA) {
        const int g0 = ti * nrows_tile;
        const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
        SG_MBTC_PROF_TIC(_pt);
        float nll;
        if constexpr (kSamCoupled)
            nll = mbtc::mbtc_forward_tile_outlined(w, g0, nrows, acts, sc, tok.tokens,
                                                   tok.targets, sm.sA, sm.sB, sm.red);
        else
            nll = mbtc::mbtc_forward_tile(w, g0, nrows, acts, sc, tok.tokens, tok.targets,
                                          sm.sA, sm.sB, sm.red);
        SG_MBTC_PROF_ACC(0, _pt);
        SG_MBTC_PROF_TIC(_pt);
        if constexpr (kSamCoupled)
            mbtc::mbtc_backward_tile_outlined(w, g0, nrows, B, acts, sc, part, tok.tokens,
                                              tok.targets, sm.sA, sm.sB, sm.red, sm.dBmat, sm.dCmat);
        else
            mbtc::mbtc_backward_tile(w, g0, nrows, B, acts, sc, part, tok.tokens, tok.targets,
                                     sm.sA, sm.sB, sm.red, sm.dBmat, sm.dCmat);
        SG_MBTC_PROF_ACC(1, _pt);
        if (threadIdx.x == 0) nll_acc += nll;
        __syncthreads();
    }
    if (threadIdx.x == 0) loss_part[cta] = nll_acc;
    bar.sync();   // B1: all acts (X + dY + dh0) + partials complete
    SG_MBTC_PROF_TIC(_pt);

    // ── P2: assemble all 28 grads. 8 projection dW output-stationary (gt %
    //    nCTA), dt_proj.bias, non-GEMM partial-reduce, embedding owner-scan. ──
    if (threadIdx.x == 0) mbtc::mbtc_build_dw_specs(acts, T, sm.spec);
    __syncthreads();
    mbtc::MbDwSpec* spec = sm.spec;
    const int n_dw = mbtc::mbtc_dw_total_tiles<SG_TUNED_TILE_N>(spec);
    SG_MBTC_PROF_TIC(_pt);
#if !SG_MBTC_BYPASS_DW_GEMM
    if (kDwG > 1) {
        // SPLIT-K (multi-CTA tiling): fan (n_dw·G) (tile,chunk) partials over the
        // grid so the ~73% idle SMs do work; deterministic ascending-chunk reduce.
        for (int item = cta; item < n_dw * kDwG; item += nCTA) {
            const int gt = item / kDwG, kc = item % kDwG;
            mbtc::mbtc_dw_run_tile_splitk<SG_TUNED_TILE_N>(spec, gt, kc, kDwG, dw_part, sm.sA, sm.sB);
        }
        bar.sync();   // all (gt,kc) partials complete before the reduce reads them
        mbtc::mbtc_dw_reduce_splitk<SG_TUNED_TILE_N>(spec, n_dw, kDwG, dw_part, grad, cta, nCTA);
    } else {
        for (int gt = cta; gt < n_dw; gt += nCTA)
            mbtc::mbtc_dw_run_tile<SG_TUNED_TILE_N>(spec, gt, grad, sm.sA, sm.sB);
    }
#else
    (void)n_dw; (void)dw_part; (void)kDwG;
#endif
    SG_MBTC_PROF_ACC(4, _pt);          // P2-a: projection dW GEMMs (K=T)
    SG_MBTC_PROF_TIC(_pt);
    mbtc::mbtc_dw_biases(spec, grad, cta, nCTA);
    mbtc::mbtc_partial_reduce(part_base, grad, nCTA, cta);
    SG_MBTC_PROF_ACC(5, _pt);          // P2-b: biases + non-GEMM partial reduce
    SG_MBTC_PROF_TIC(_pt);
#if !SG_MBTC_BYPASS_EMBED_SCAN
    mbtc::mbtc_embed_owner_scan(acts, tok.tokens, T, grad, cta, nCTA);
#endif
    SG_MBTC_PROF_ACC(6, _pt);          // P2-c: embedding owner-scan (O(vocab·T))
    // Loss reduce (fp64 ordered) by CTA 0.
    if (cta == 0 && threadIdx.x == 0) {
        double s = 0.0;
        for (int c = 0; c < nCTA; ++c) s += (double)loss_part[c];
        *tok.loss_out = (float)(s / (double)B);
    }
    (void)loss_out;
    SG_MBTC_PROF_ACC(2, _pt);          // close P2 (dW GEMM + reduce + embed + loss)
    bar.sync_reset(ctx.g_next_task);   // B2: reduced grad ready; reset queue
    SG_MBTC_PROF_TIC(_pt);             // open P3 (optimizer tail)

    // ── P2.4 (LookSAM ONLY, SAM steps): the MODEL-COUPLED SAM 2nd backward that
    //    produces st.sam_dir = g_sam − g (INTEGRATION-OPTSTAGES §6). PORT of the
    //    decoder twin's P2.4 (fused_decoder_megakernel.cuh) onto the Mamba tile fns.
    //    ⚠ LANDED-DORMANT / NOT PRODUCTION-ROUTED: mamba×looksam is BLOCKED on the SAME
    //    PRE-EXISTING shared mamba scan/forward A/A/A race that blocks mamba×prodigy (this
    //    port is faithful; the failure is NOT in this code — it shows EVEN on a SAM-OFF
    //    step). dispatch.cpp's mb_looksam carve-out (env SG_MAMBA_LOOKSAM_PROBE to lift) +
    //    the Python registries keep this branch unreachable from the race until the shared
    //    mamba forward (megakernel_common.cuh GridBarrier / model_stage_mamba3.cuh) is fixed
    //    by its owner. Kept here (if-constexpr'd, proven byte-identical for every non-LookSAM
    //    Opt — mamba×{adamw,lion,grokfast} stay bit-identical A/A/A) as the ready start point
    //    for that fix. decoder/vit looksam ARE A/A/A-clean + registered-converted.
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
    //    contiguous per-CTA range → per-CTA partial → CTA0 ascending sum), and the 2nd
    //    fwd+bwd+assembly reuses the SAME machinery as the first pass (fixed tile
    //    ownership, the deterministic ascending-CTA partial-reduce), so the whole phase
    //    is deterministic to the SAME degree as the first pass. Guarded so every other
    //    opt's path is byte-identical (no extra barrier / work). On a NON-SAM step
    //    (looksam_sam==0) this whole block is skipped — st.sam_dir already holds the
    //    cached direction.
    //    SuperGrok11/15 SHARE this exact SAM 2nd-backward machinery (INTEGRATION-
    //    OPTSTAGES §4/§5): the ONLY difference is the side-channel write — LookSAM writes
    //    st.sam_dir[i] = g_sam − g, SG11/15 write st.sharpness[i] = (g_sam − g)²
    //    (supergrok11_sm90.cuh:246 / supergrok15_sm90.cuh:315). sharpness is then the 2nd
    //    MLP input the P2.45 meta-net mu precompute reads. SuperGrok2 ALSO shares it (its
    //    meta-net's 2nd MLP input is the SAME sharpness signal, read by sg2_meta_stages).
    //    The SAM-step gate (st.looksam_sam) and the perturbation radius (st.rho) are the
    //    same every-k cadence; on a non-SAM step the cached side-channel is reused verbatim.
    //    The now-FIXED register-pressure wgmma-accumulator-spill race (commit 0b57f7e)
    //    un-dormanted the mamba SAM 2nd-pass (looksam/mamba A/A/A bit-exact), so porting
    //    SG11/15/SG2 onto the SAME machinery is a feature port (not a determinism fix);
    //    the gate re-verifies A/A/A for each new SG cell.
    if constexpr (Opt == OptId::LookSAM || Opt == OptId::SuperGrok11 ||
                  Opt == OptId::SuperGrok15 || Opt == OptId::SuperGrok2) {
        if (st.looksam_sam != 0.0f) {
            // (a) GLOBAL ‖g‖ over the reduced grad, deterministic (reuse the loss
            //     workspace as P2.5 does: loss_part[nCTA] = per-CTA Σg², loss_out[1]
            //     broadcasts rho_over_norm). The reduced loss is already in
            //     *tok.loss_out (host state), so this is free scratch.
            float* sq_part = loss_part;          // [nCTA] per-CTA Σ g²  (ascending reduce)
            float* ron_bc  = loss_out;           // [1] broadcast rho_over_norm
            const int64_t total_e = kMambaTotalElems;
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
            //     P1 + P2 EXACTLY, but: re-zero the per-CTA non-GEMM partial (it
            //     accumulates), do NOT write the loss (keep the first-pass loss), and
            //     assemble into `sam_grad` (NOT `grad`). `w` wraps `params` (now
            //     perturbed) so the tile fns read the perturbed weights with no re-bind.
            for (int64_t i = threadIdx.x; i < mbtc::kPartElems; i += blockDim.x) my_part[i] = 0.0f;
            bar.sync();   // B2.4d: partials cleared before the 2nd backward accumulates
            for (int ti = cta; ti < n_tiles; ti += nCTA) {
                const int g0 = ti * nrows_tile;
                const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
                mbtc::mbtc_forward_tile_outlined(w, g0, nrows, acts, sc, tok.tokens,
                                                 tok.targets, sm.sA, sm.sB, sm.red);
                mbtc::mbtc_backward_tile_outlined(w, g0, nrows, B, acts, sc, part, tok.tokens,
                                                  tok.targets, sm.sA, sm.sB, sm.red, sm.dBmat, sm.dCmat);
                __syncthreads();
            }
            bar.sync();   // B2.4e: all 2nd-pass acts (X + dY + dh0) + partials complete
            // Re-build the dW specs from the 2nd-pass acts (dY adjoints changed).
            if (threadIdx.x == 0) mbtc::mbtc_build_dw_specs(acts, T, sm.spec);
            __syncthreads();
            mbtc::MbDwSpec* spec2 = sm.spec;
            const int n_dw2 = mbtc::mbtc_dw_total_tiles<SG_TUNED_TILE_N>(spec2);
            if (kDwG > 1) {
                for (int item = cta; item < n_dw2 * kDwG; item += nCTA) {
                    const int gt = item / kDwG, kc = item % kDwG;
                    mbtc::mbtc_dw_run_tile_splitk<SG_TUNED_TILE_N>(spec2, gt, kc, kDwG, dw_part, sm.sA, sm.sB);
                }
                bar.sync();
                mbtc::mbtc_dw_reduce_splitk<SG_TUNED_TILE_N>(spec2, n_dw2, kDwG, dw_part, sam_grad, cta, nCTA);
            } else {
                for (int gt = cta; gt < n_dw2; gt += nCTA)
                    mbtc::mbtc_dw_run_tile<SG_TUNED_TILE_N>(spec2, gt, sam_grad, sm.sA, sm.sB);
            }
            mbtc::mbtc_dw_biases(spec2, sam_grad, cta, nCTA);
            mbtc::mbtc_partial_reduce(part_base, sam_grad, nCTA, cta);
            mbtc::mbtc_embed_owner_scan(acts, tok.tokens, T, sam_grad, cta, nCTA);
            bar.sync();   // B2.4f: g_sam fully assembled in sam_grad
            // (d) WRITE the SAM side-channel (into the persistent state slice) + restore p.
            //     LookSAM → sam_dir = g_sam − g; SuperGrok11/15/SuperGrok2 → sharpness =
            //     (g_sam − g)² (SG2 shares the SAM 2nd-backward machinery: its meta-net's
            //     2nd MLP input is the SAME sharpness signal, read by sg2_meta_stages).
            //     Mirrors the decoder/vit twins' unified P2.4 write.
            {
                const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
                for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
                     i < total_e; i += gstride) {
                    const float diff = sam_grad[i] - grad[i];
                    if constexpr (Opt == OptId::LookSAM) {
                        // st.sam_dir is the extra-slice base (host-bound). g_sam − g.
                        const_cast<float*>(st.sam_dir)[i] = diff;
                    } else {
                        // SuperGrok11/15/SuperGrok2: sharpness = (g_sam − g)² (state slice).
                        const_cast<float*>(st.sharpness)[i] = diff * diff;
                    }
                    params[i] = sam_backup[i];   // restore the ORIGINAL weights
                }
            }
            bar.sync_reset(ctx.g_next_task);   // B2.4g: params restored + sam_dir ready; reset queue for P3
        }
    }

    // ── P2.5 (GrokAdamW ONLY): GLOBAL grad-norm clip coefficient. PORT of the
    //    decoder/vit TC kernels' P2.5 (fused_decoder_megakernel.cuh:461-512) — the
    //    eager grokadamw clips the WHOLE grad set to grad_clip via a GLOBAL L2 norm
    //    (clip_grad_norms_device_side → total_norm = sqrt(Σ_i ‖g_i‖²), clip_coef =
    //    grad_clip/(total_norm+1e-6) when total_norm>grad_clip, else 1) BEFORE the
    //    apply. Replicated on the REDUCED grad with a deterministic ascending
    //    reduction (NO float atomic — COMPONENT_CONTRACT): each CTA sums a contiguous
    //    element-range into a per-CTA partial slot, CTA0 sums the partials in
    //    ASCENDING CTA order → total_norm → clip_coef, broadcast via a workspace slot.
    //    grad is NOT mutated (the return_grad oracle + the eager-side clip both see
    //    the unclipped reduced grad); the coefficient is applied per-element inside
    //    apply_optimizer<GrokAdamW>. Guarded so every other opt's P3 is byte-identical
    //    (no extra barrier/work). loss_part (nCTA) + loss_out (1) are free scratch
    //    here: the reduced loss is already in *tok.loss_out (a separate pointer).
    //    EXTENDED to NeuralGrok (same eager global clip; consumed in apply_optimizer<NeuralGrok>).
    if constexpr (Opt == OptId::GrokAdamW || Opt == OptId::NeuralGrok) {
        float* sq_part = loss_part;          // [nCTA] per-CTA Σ g²  (ascending reduce)
        float* coef_bc = loss_out;           // [1] broadcast clip_coef
        const int64_t total = kMambaTotalElems;
        const int64_t base = total / nCTA, rem = total % nCTA;
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

    // ── P2.6 (PRODIGY ONLY): STAGED cross-ALL-tensors d-estimate. PORT of the
    //    decoder/vit TC kernels' P2.6 (fused_decoder_megakernel.cuh:514-568) onto the
    //    mamba constant tables.
    //    ⚠ LANDED-DORMANT / NOT PRODUCTION-ROUTED: mamba×prodigy is BLOCKED on a
    //    PRE-EXISTING A/A/A determinism failure SHARED by the decoder/vit prodigy P2.6
    //    (this port faithfully reproduces them; the failure is NOT in this code). The
    //    dispatch.cpp mamba carve-out (optimizer!="prodigy") + the Python has_l3_real
    //    gate keep this branch unreachable from the race until the shared prodigy P2.6
    //    (opt_stages_precompute.cuh) / GridBarrier (megakernel_common.cuh) is fixed by
    //    its owner. Kept here (if-constexpr'd, proven byte-identical for every non-Prodigy
    //    Opt) as the ready starting point for that fix. See dispatch.py's ⚠ escalation.
    //    The apply tail (apply_optimizer<Prodigy>) reads
    //    st.d_factor (the effective LR scale d), a GLOBAL reduction over EVERY element
    //    of EVERY parameter tensor. We compute it here, BYTE-FAITHFUL to the live eager
    //    multi-tensor path (prodigy_sm90.cuh:465-544, the order prodigy.py →
    //    _ops.prodigy_fused_step actually executes):
    //      d_prev  = persisted d_lr  (step 1: d0 cold-start — the zero-init state slot
    //                would give d_prev=0 ⇒ d=0 ⇒ frozen params; eager inits _d_lr=d0,
    //                so seed it here, the grokfast-style step-1 fix)
    //      r_ema  <- beta3·r_ema + Σ d_prev²·<g, p0−p>     (decay persisted SCALAR,
    //      s_ema  <- beta3·s_ema + Σ d_prev²·|g|            then add this step's Σ)
    //      d       = max(d_prev, d_coef·r_ema/|s_ema|)      (prodigy_update_d; d_coef
    //                scales ONLY the candidate — persisted r_ema stays UNSCALED)
    //    DETERMINISM (COMPONENT_CONTRACT): NO float atomic. Each CTA publishes its
    //    (r,s) into per-CTA slots (opt_reduce) → grid barrier → CTA0 owner-sums in
    //    ascending index order → writes d back to the persisted slot + a broadcast
    //    slot. The decay is on the persisted SCALARS (not the per-CTA partials): the
    //    work-steal queue reassigns tensors to CTAs across steps, so a per-CTA EMA is
    //    undefined — the live form is a scalar EMA (prodigy_sm90.cuh:488). Guarded so
    //    every other opt's P3 is byte-identical (no extra barrier/work).
    // DETERMINISM — FIXED-PARTITION reduction (mirrors the A/A/A-CLEAN GrokAdamW
    // P2.5 + the decoder/vit twins' P2.6), NOT the work-steal task-queue drain.
    // WHY THIS SHAPE (the A/A/A determinism leg): the work-steal phaseA drained the
    // global TaskQueue (atomicAdd(g_next_task)) in timing-dependent claim order, so
    // each CTA's (r,s) slot could be a different tensor subset run-to-run; with non-
    // associative fp32 add the partials regroup differently and the owner's d can
    // differ at ULP level from step>=2 — a COMPONENT_CONTRACT deterministic-reduction
    // violation (not reproducible BY CONSTRUCTION). A FIXED contiguous element range
    // per CTA over the FLAT [0,kMambaTotalElems) arrays (params/param_init/grad are
    // parallel flat blobs → flat Σ == cross-tensor Σ) with a plain sync() is the
    // IDENTICAL shape as the A/A/A-CLEAN GrokAdamW P2.5 — deterministic by
    // construction. Per-element math still CALLS canonical prodigy_partials_step
    // (prodigy.h, single-source). g_next_task is UNTOUCHED (B2 reset it), so P3's
    // drain is unchanged — no sync_reset in P2.6. (mamba×prodigy is dispatch-gated to
    // eager today on a SEPARATE mamba scan/forward determinism issue; this keeps the
    // landed-dormant P2.6 correct + deterministic.)
    if constexpr (Opt == OptId::Prodigy) {
        float* r_part = opt_reduce;                  // [nCTA] per-CTA Σ d²·<g,p0−p>
        float* s_part = opt_reduce + nCTA;           // [nCTA] per-CTA Σ d²·|g|
        float* d_bc   = opt_reduce + 2 * nCTA;       // [1] reduced-d broadcast
        const float d_prev = (step == 1) ? st.d0 : st.prodigy_persist[2];
        const int64_t total_p = kMambaTotalElems;
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
    //    2D weights (INTEGRATION-OPTSTAGES §3). PORT of the decoder/vit twins' P2.7
    //    (fused_decoder_megakernel.cuh / fused_vit_megakernel.cuh) onto the mamba
    //    constant tables — same shared opt_stages_precompute.cuh helpers, same barrier
    //    sequence; only the per-model 2D table (mbtc::kMbMuon2D, 13 matrices incl.
    //    tok[99,128]/pos[8,128]/A_log[256,16]/dt_proj[256,8]/x_proj[40,256]) and offset
    //    array (kMambaOffsets) differ. For EACH 2D matrix all CTAs cooperate: buf=μ·buf+g
    //    (buf is the PERSISTENT m-slice — momentum state, NOT transient), ‖buf‖_F via
    //    per-CTA partials → inv_norm, X=buf·inv_norm, then ns_steps × { A=XXᵀ → AX=A·X →
    //    AAX=A·AX → orth=a·X+b·AX+c·AAX, swap }, then the canonical muon_update_step apply
    //    (decay·p + neg_lr_scale·orth). The 1D / non-2D weights (D, conv1d.weight/bias,
    //    dt_proj.bias, LN γ/β, out.bias) take the AdamW aux tail in P3. The elementwise
    //    bodies CALL muon.h (muon_momentum_normalize_step via the phaseA buf body,
    //    muon_ns_combine_step, muon_update_step); the matmuls are the cited new device
    //    code (the eager path delegates to torch::mm/cuBLAS). Guarded so every other opt is
    //    byte-identical (no extra barriers / work).
    //    DETERMINISM (the A/A/A leg the gate verifies): muon/mamba is a SINGLE forward +
    //    NS precompute — it does NOT re-run the mamba forward (unlike the SAM-2nd-pass
    //    mamba cells looksam/SG11/15 which hit the shared mamba-forward A/A/A race), and the
    //    P2.7 reductions are fixed-ownership ascending-CTA (muon_norm_reduce_phaseB sums
    //    nrm_partials[0..nCTA) in ascending index) + the NS matmuls are deterministic
    //    grid-cooperative tiles — so the phase is deterministic BY CONSTRUCTION. The gate
    //    confirms A/A/A bit-identity; if it ever fails, the cell is landed-dormant (see
    //    the dispatch.cpp mb_muon carve-out scaffold).
    if constexpr (Opt == OptId::Muon) {
        // Carve the per-matrix NS scratch (sized for the largest 2D weight).
        PrecomputeWorkspace pw{};
        pw.muon_X            = muon_base;
        pw.muon_AX           = pw.muon_X   + mbtc::kMbMuonMaxNumel;
        pw.muon_AAX          = pw.muon_AX  + mbtc::kMbMuonMaxNumel;
        pw.muon_orth         = pw.muon_AAX + mbtc::kMbMuonMaxNumel;
        pw.muon_A            = pw.muon_orth + mbtc::kMbMuonMaxNumel;
        pw.muon_nrm_partials = pw.muon_A   + (int64_t)mbtc::kMbMuonMaxRows * mbtc::kMbMuonMaxRows;
        pw.muon_inv_norm     = pw.muon_nrm_partials + nCTA;
        const float momentum = st.beta1;          // Muon momentum (eager Muon: betas[0])
        const int   ns_steps = 5;                 // bindings.cpp default
        for (int mi = 0; mi < mbtc::kMbNumMuon2D; ++mi) {
            const mbtc::MbMuon2D M = mbtc::kMbMuon2D[mi];
            const int rows = M.rows, cols = M.cols;
            const int64_t numel = (int64_t)rows * cols;
            const int64_t off   = (int64_t)kMambaOffsets[M.tidx];
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
            // muon_X (the last combine wrote muon_orth, then we swapped → muon_X).
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
                float* tmp = pw.muon_X; pw.muon_X = pw.muon_orth; pw.muon_orth = tmp;
            }
            // Apply Muon: p = decay_factor·p + neg_lr_scale·orth  (muon.h:63-73).
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
    //    precompute (INTEGRATION-OPTSTAGES §4/§5) — PORT of the decoder/vit twins'
    //    P2.45 onto the mamba constant tables. SINGLE task-queue drain, NO grid
    //    barrier: each tensor T is SELF-CONTAINED — mu(T) = sg_rescale·phi(g[T],
    //    sharpness[T]) (per element) and, for SG11, the per-tensor gate(T) =
    //    sigmoid(gate_temp·cos(g[T], mu[T])) depend ONLY on T's own elements, so mu(+gate)→
    //    apply fuse into ONE body the SAME CTA owns end-to-end inside the EXISTING P3
    //    drain. The phi weights (per-TENSOR weight set) are staged to SMEM ONCE per
    //    block before the drain. SG15's gate is the host scalar st.gate (no cosine
    //    stage); SG11 binds the per-tensor gate the helper returns. sharpness (the 2nd
    //    MLP input) was produced by P2.4 (or the cached prior step).
    __shared__ float sg_sW1[kSgPhiHidden * 2];
    __shared__ float sg_sb1[kSgPhiHidden];
    __shared__ float sg_sW2[kSgPhiHidden];
    if constexpr (Opt == OptId::SuperGrok11 || Opt == OptId::SuperGrok15) {
        sg_stage_phi_weights<kSgPhiHidden>(st.sg_phi_W1, st.sg_phi_b1, st.sg_phi_W2,
                                           sg_sW1, sg_sb1, sg_sW2);   // syncs
    }

    // ── P3-SG2 (SuperGrok2 ONLY): the FULL CSA/HCA/PEER/GRU meta-net as the optimizer
    //    phase — PORT of the decoder/vit twins' P3-SG2 onto the mamba constant tables.
    //    Run INSTEAD of the per-element apply_optimizer<SuperGrok2> (which is only the
    //    Adam-on-smart_grad stub). Each CTA work-steals WHOLE tensors from the queue
    //    (reset at B2 / B2.4g) and runs sg2_meta_stages for each tensor end-to-end:
    //    STAGE -1 in-kernel segmented sort (|grad| ascending, index tie-break — strategy
    //    A) → S0..S5 (input-proj, CSA, HCA, GRU, PEER, apply). The SAM-written
    //    st.sharpness (P2.4, (g_sam−g)²) is the meta-net's 2nd MLP input (sharp_base).
    //    The meta-net WEIGHTS come from HBM — sg2_meta_stages is instantiated with
    //    WeightsT==SG2Weights + BuildSort=true. The per-tensor intermediates + the
    //    segmented-sort scratch live in this CTA's slice of the SG2 workspace (carve-
    //    LAST, after the Muon NS scratch). This whole phase is if-constexpr'd to
    //    SuperGrok2, so every other cell is byte-identical (no extra barrier/work). The
    //    kernel returns after this — no trailing grid barrier (each tensor is CTA-local).
    //    ⚠ A/A/A: the SAM double-forward + segmented sort re-exercise the shared mamba
    //    forward (the .sg2_spec.md-flagged risk). The gate re-verifies A/A/A; if it trips
    //    the race, mamba×supergrok2 lands DORMANT (dispatch-gated to eager).
    if constexpr (Opt == OptId::SuperGrok2) {
        // Reconstruct the SG2 weight bundle (HBM pointers) from the threaded fields.
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
        // Stage row_off (int64) for the SG2State adapter once into the FRONT of this
        // CTA's SG2 workspace slice (kMambaOffsets is __constant__ int; SG2State.row_off
        // wants const int64_t*). All CTAs stage their own copy (cheap, kMambaNumTensors).
        float* sg2_base = sg2_ws_base + (int64_t)blockIdx.x * mb_sg2_ws_stride_floats();
        int64_t* row_off64 = reinterpret_cast<int64_t*>(sg2_base);
        for (int t = threadIdx.x; t < kMambaNumTensors; t += blockDim.x)
            row_off64[t] = (int64_t)kMambaOffsets[t];
        // The meta-net scratch starts AFTER the row_off64 staging block: reserve
        // kMambaNumTensors int64 = 2*N floats.
        float* sg2_meta_ws = sg2_base + 2 * kMambaNumTensors;
        __syncthreads();
        SG2State stt{};
        stt.exp_avg     = st.exp_avg;
        stt.exp_avg_sq  = st.exp_avg_sq;
        stt.mu          = const_cast<float*>(st.mu);
        stt.slow        = st.sg2_slow;
        stt.gru_state   = st.sg2_gru_state;
        stt.perm        = nullptr;          // built in-kernel (BuildSort=true)
        stt.unsort      = nullptr;
        stt.workspace   = sg2_meta_ws;
        stt.ws_stride   = sg2_ws_stride<MbSG2Dims>((int64_t)kMbSG2Nmax);
        stt.n_tensors   = kMambaNumTensors;
        stt.n           = kMambaSizes;      // __constant__ int[]
        stt.row_off     = row_off64;
        __shared__ int task_slot2;
        TaskQueue q2 = ctx.queue();
        for (int t = q2.next_block(&task_slot2); t < kMambaNumTensors;
             t = q2.next_block(&task_slot2)) {
            sg2_meta_stages<MbSG2Dims, SG2Weights, float, float, /*BuildSort=*/true>(
                w2, t, stt, sc2, params, grad, st.sharpness, sg2_meta_ws);
        }
        SG_MBTC_PROF_ACC(3, _pt);          // close P3 (SG2 owns the optimizer phase)
        return;   // SG2 owns the whole optimizer phase; skip P3.
    }

    // ── P3: the REAL optimizer tail over the reduced grad (work-steal 28). ──
    st.lr = lr;
    {
        __shared__ int task_slot;
        TaskQueue q = ctx.queue();
        for (int t = q.next_block(&task_slot); t < kMambaNumTensors;
             t = q.next_block(&task_slot)) {
            const int n = kMambaSizes[t];
            const int64_t off = (int64_t)kMambaOffsets[t];
            // MUON: the 2D matrices were orthogonalized + applied in P2.7; P3 handles
            //   ONLY the 1D / non-2D weights, which take the AdamW tail (muon.h:75-76,
            //   the eager Muon auto-split). Skip the 2D ones here (already done).
            if constexpr (Opt == OptId::Muon) {
                if (mbtc::mb_is_muon_2d(t)) continue;
            }
            FusedOptState ts = mamba_rebase_state<Opt>(st, off);
            // (i) PER-TENSOR LAYER-WISE β1 (GrokAdamW only) — PORT of the decoder
            //     P3 (fused_decoder_megakernel.cuh:587-591): β1_i = β1·(1-γ)^t, where
            //     t == the tensor's flat named_parameters() layer index (the work-steal
            //     task id maps 1:1 to kMambaOffsets order == the eager enumeration
            //     order, so t IS the eager layer index). bc1 must be rebased TOO
            //     (= 1-β1_i^step) or m_hat=m/bc1 mismatches eager; bc2 stays global
            //     (β2 is not layer-wise). This is the mechanism that fails the STEP-1
            //     STATE gate when dropped (decoder observed m-rel 0.895).
            if constexpr (Opt == OptId::GrokAdamW) {
                const float b1 = st.beta1 * powf(1.0f - st.gamma, (float)t);
                ts.beta1 = b1;
                ts.bc1   = 1.0f - powf(b1, (float)step);
            }
            // SuperGrok11/15 meta-net mu (+ SG11 per-tensor gate) precompute for THIS
            // tensor, BEFORE the apply reads ts.mu/ts.gate — PORT of the decoder/vit P3.
            // mu(T)=sg_rescale·phi(g,sharpness) over [off,off+n); SG11 also computes the
            // gate(T)=sigmoid(gate_temp·cos) (block-uniform __syncthreads/reduce — the whole CTA
            // owns t). SG15's gate is st.gate (host sigmoid(accuracy)). Helpers index
            // grad/sharpness/mu by off+i, so pass the BASE pointers (st.mu, grad,
            // st.sharpness) + off; the apply then reads ts.mu (== st.mu+off). phi_b2 read
            // on-device from sg_phi_W2[H].
            if constexpr (Opt == OptId::SuperGrok11) {
                const float b2 = (st.sg_phi_W2 != nullptr) ? st.sg_phi_W2[kSgPhiHidden]
                                                           : st.sg_phi_b2;
                const float g8 = sg11_precompute_mu_and_gate_for_tensor<kSgPhiHidden>(
                    st.mu, grad, st.sharpness, sg_sW1, sg_sb1, sg_sW2,
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
                // 1D / non-2D weights: the canonical AdamW tail (muon.py:99-125 routes
                // non-matrix params to a SEPARATE AdamW group with INDEPENDENT
                // hyperparameters — adamw_lr/adamw_betas — NOT the 2D Muon lr/momentum).
                // ts.lr/ts.beta1 carry the 2D-group's (lr=0.02, momentum=0.95), so the
                // 1D tail MUST instead use the aux_* fields (eager adamw_lr/adamw_betas).
                // weight_decay is SHARED across both eager groups (muon.py:122) → ts.wd
                // stays. eps is the eager adamw_eps (= ts.eps, mapped by _opt_scalars_from).
                // bc1/bc2 are device-computed from aux_beta^step (kept out of the mirror
                // to shrink the host ABI; fp32 powf, tol 2e-3 ample). apply_optimizer<Muon>
                // would deref st.orth (NS dir, only valid for 2D), so route directly to
                // adamw_step with the aux hyperparameters — IDENTICAL to the decoder/vit twins.
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
    SG_MBTC_PROF_ACC(3, _pt);          // close P3
}

// Launcher (static-smem; the decoder TC contract). ncta_cap>0 caps the launched
// CTAs (the per-CTA scratch is nCTA×slab; a memory-tight TEST caps it). Grid
// barrier rendezvous is over the LAUNCHED count → hang-safe; determinism is per
// fixed nCTA (dW-tile / partial / embed owner maps read ctx.n_ctas). B%16 req.
// Host helper: the EXACT CTA count launch_fused_mamba_megakernel_tc<Opt> will use
// (occ·n_sms with occupancy-fill, or the cap). The caller MUST size the workspace
// with this — the per-CTA scratch/partials are nCTA·slab, so sizing for n_sms while
// the launcher runs occ·n_sms would overflow. Returns 0 on a CUDA-attr failure (the
// caller should treat that as "fall back to a conservative larger size").
template <OptId Opt>
int mb_tc_launched_nctas(int dev, int ncta_cap) {
    int n_sms = 0;
    if (cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 0;
    int occ = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, (const void*)&fused_mamba_megakernel_tc<Opt>, SG_MB_TC_MEGA_BLOCK, 0) != cudaSuccess)
        return 0;
    if (occ < 1) return 0;
    unsigned waves = (SG_MBTC_MIN_BLOCKS_PER_SM >= 2 && occ > 1) ? (unsigned)occ : 1u;
    unsigned launch_ctas = (unsigned)n_sms * waves;
    if (ncta_cap > 0 && (unsigned)ncta_cap < launch_ctas) launch_ctas = (unsigned)ncta_cap;
    return (int)launch_ctas;
}

template <OptId Opt>
cudaError_t launch_fused_mamba_megakernel_tc(
        PersistentContext ctx, float* params, MambaTokenCtx tok,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream,
        int ncta_cap = 0) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 0;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;

    int occ = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (const void*)&fused_mamba_megakernel_tc<Opt>, SG_MB_TC_MEGA_BLOCK,
        /*dynamicSMemBytes=*/0);
    if (err != cudaSuccess) return err;
    if (occ < 1) return cudaErrorLaunchOutOfResources;

    // OCCUPANCY-FILL: at min-blocks≥2 the kernel is built ≤128 reg/thread so `occ`
    // CTAs co-reside per SM. Launch occ·n_sms CTAs (one full residency set) so the
    // co-resident CTAs hide each other's HBM latency across the scan/conv/LN phases
    // that run fully exposed at occ=1 — the persistent grid barrier rendezvous is
    // over the LAUNCHED count and occ·n_sms are guaranteed simultaneously resident,
    // so it stays hang-free. Determinism is per fixed launched nCTA (ascending-CTA
    // reduce). At min-blocks==1 (shipped default) occ==1 → identical to before.
    unsigned waves = (SG_MBTC_MIN_BLOCKS_PER_SM >= 2 && occ > 1) ? (unsigned)occ : 1u;
    unsigned launch_ctas = (unsigned)n_sms * waves;
    if (ncta_cap > 0 && (unsigned)ncta_cap < launch_ctas) launch_ctas = (unsigned)ncta_cap;
    ctx.n_ctas = launch_ctas;
    // B%16: the dW K-loop contracts K=T=B·8 in 16-step atoms. It ALSO guarantees
    // FULL tiles (T%kTileM==0 for kTileM∈{64,128}), which the projection GEMMs
    // REQUIRE — they process a compile-time M-extent of kTileM rows (kAtomsM
    // atoms), NOT the runtime `nrows`; a partial final tile would feed garbage
    // pad rows to the wgmma while the scalar scan/conv loops process only nsamp.
    // (A future relaxation to B%2 would satisfy the K-atom reason but silently
    // break partial tiles — keep B%16.)
    if ((tok.B % 16) != 0) return cudaErrorInvalidValue;
    // SPLIT-K dW (G>1) needs equal 16-aligned K-chunks: G must divide k_steps_total
    // = T/16 = B·kSeq/16 = B/2 (kSeq=8). With B%16==0, B/2%4==0, so G∈{1,2,4} are
    // always legal; a larger/odd G that doesn't divide B/2 would silently DROP the
    // remainder k-steps from every dW (a correctness bug) — REFUSE loudly instead.
    if (mbtc::kMbDwSplitK > 1 && (((int64_t)tok.B * mb::kSeq / wgs::kWgmmaAtomK) % mbtc::kMbDwSplitK) != 0)
        return cudaErrorInvalidValue;

    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }

    dim3 grid(launch_ctas), block(SG_MB_TC_MEGA_BLOCK);
    fused_mamba_megakernel_tc<Opt><<<grid, block, 0, stream>>>(
        ctx, params, tok, grad, lr, step, st);
    return cudaGetLastError();
}

#endif  // SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_FUSED_MAMBA_MEGAKERNEL_CUH_
