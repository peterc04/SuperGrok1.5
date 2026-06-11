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
// the TC megakernel's Prodigy branch (P2.6) drives prodigy_precompute_reduce_phaseA +
// an EMA-decay/d-update owner block, BYTE-FAITHFUL to the eager multi-tensor estimator.
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
#endif

namespace sg { namespace fused { namespace sm90 {

// Compile-time guard: the byte budget the launcher uses (kMambaSmemBytes) MUST
// equal sizeof(MambaSampleSmem). model_stage_mamba3.cuh already static_asserts
// the same equality (it pins kMambaSmemFloats to the actual struct); we restate
// it here so this header is self-contained and a drift fails loudly at this TU.
static_assert((int64_t)sizeof(MambaSampleSmem) == kMambaSmemBytes,
              "fused_mamba_megakernel: sizeof(MambaSampleSmem) != the documented "
              "kMambaSmemBytes in mamba3_layout.cuh — update kMambaSmemFloats.");

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

// ── The persistent megakernel (L3-REAL). gridDim.x = #SMs (one CTA/SM), 256
//    threads/CTA. MambaSampleSmem (~141.72 KB) lives in DYNAMIC smem (extern). ──
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

// ── Static smem arena: kMbTcStages A(64×16) + kMbTcStages B(N×16) bf16 tiles
//    (the GEMM K-loop double-buffer ring — slot s at sA + s*64*16 / sB +
//    s*N*16) + the 256-float reduction slot + the scan-bwd cross-channel reduce
//    targets (dBmat/dCmat, [kSeq×kState] each) + the 8 dW specs (shared, not
//    per-thread stack). At S=2 + N=128 the ring is 2·(2KB+4KB)=12KB; MbTcSmem
//    total ~14.6KB ≪ the 48KB static cap (the TC launcher uses static smem). ──
struct MbTcSmem {
    __nv_bfloat16 sA[mbtc::kMbTcStages * 64 * 16];
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
__host__ __device__ __forceinline__ int64_t mb_tc_workspace_floats(int T, int nCTA) {
    return mb_tc_acts_floats(T)
         + (int64_t)nCTA * mbtc::mb_tile_scratch_floats()
         + (int64_t)nCTA * mbtc::kPartElems
         + nCTA + 1
         + mb_tc_dw_part_floats()                       // split-K dW partials (G>1)
         + mb_tc_opt_reduce_floats(nCTA)                // STAGED-opt (Prodigy) reduce slots
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
#if SG_MB_TC_PROFILE
    // 8-byte align the double accumulator: round the float offset up to an even
    // count so reinterpret_cast<double*> is aligned (past the opt-reduce region — the
    // phase-profiler is the LAST workspace term, mirroring mb_tc_workspace_floats).
    float* prof_f = opt_reduce + mb_tc_opt_reduce_floats(nCTA);
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
        float nll = mbtc::mbtc_forward_tile(w, g0, nrows, acts, sc, tok.tokens, tok.targets,
                                            sm.sA, sm.sB, sm.red);
        SG_MBTC_PROF_ACC(0, _pt);
        SG_MBTC_PROF_TIC(_pt);
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
    if constexpr (Opt == OptId::GrokAdamW) {
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
    if constexpr (Opt == OptId::Prodigy) {
        PrecomputeWorkspace pw{};
        pw.prodigy_partials = opt_reduce;            // [r slots | s slots]
        pw.prodigy_d        = opt_reduce + 2 * nCTA; // reduced-d broadcast slot
        // d_prev: persisted d_lr (slot 2 of prodigy_persist), or d0 at step 1.
        const float d_prev = (step == 1) ? st.d0 : st.prodigy_persist[2];
        st.d_factor = d_prev;   // phaseA reads st.d_factor as d_prev (prodigy.h)
        // Phase A: each CTA accumulates Σ d_prev²·<g,p0−p> / Σ d_prev²·|g| over its
        // claimed tensors → per-CTA (r,s) slots. Drains the task queue (the P3
        // re-drain below needs a queue reset, done at the barrier). Reads the mamba
        // __constant__ layout tables (kMambaSizes/kMambaOffsets), n_tasks=kMambaNumTensors.
        prodigy_precompute_reduce_phaseA(ctx, params, st.param_init, grad,
                                         kMambaSizes, kMambaOffsets, d_prev, pw);
        bar.sync_reset(ctx.g_next_task);   // B2.6a: slots published; reset queue for P3
        // Owner block (CTA0 thread0): EMA decay + accumulate + d_coef + update_d,
        // byte-matching launch_multi_tensor_prodigy_fused_reduce_step.
        if (cta == 0 && threadIdx.x == 0) {
            float r_step = 0.0f, s_step = 0.0f;     // ascending-CTA owner-sum
            for (int c = 0; c < nCTA; ++c) {
                r_step += pw.prodigy_partials[c];
                s_step += pw.prodigy_partials[nCTA + c];
            }
            // Decay persisted scalars by beta3, then add this step's reduction.
            const float r_ema = st.beta3 * st.prodigy_persist[0] + r_step;
            const float s_ema = st.beta3 * st.prodigy_persist[1] + s_step;
            // d = max(d_prev, d_coef·r_ema/|s_ema|). prodigy_update_d does
            // max(d_prev, r/|s|) verbatim, so fold d_coef into the numerator copy
            // (persisted r_ema stays UNSCALED — returned/persisted, eager parity).
            const float d_new = algo::prodigy_update_d(d_prev, st.d_coef * r_ema, s_ema);
            st.prodigy_persist[0] = r_ema;          // persist UNSCALED EMA
            st.prodigy_persist[1] = s_ema;
            st.prodigy_persist[2] = d_new;          // persisted d_lr for next step
            pw.prodigy_d[0]       = d_new;          // broadcast to all CTAs
        }
        bar.sync();   // B2.6b: d visible to every CTA before the apply
        st.d_factor = pw.prodigy_d[0];              // the reduced d the tail reads
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
            float* __restrict__ p = params + off;
            const float* __restrict__ gg = grad + off;
            for (int i = threadIdx.x; i < n; i += blockDim.x)
                apply_optimizer<Opt>(p, gg, (int64_t)i, step, ts);
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
