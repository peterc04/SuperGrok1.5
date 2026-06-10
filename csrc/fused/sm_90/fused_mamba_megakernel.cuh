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

// ── Static smem arena: one A(64×16) + one B(N×16) bf16 tile + the 256-float
//    reduction slot + the scan-bwd cross-channel reduce targets (dBmat/dCmat,
//    [kSeq×kState] each) + the 8 dW specs (shared, not per-thread stack). ──
struct MbTcSmem {
    __nv_bfloat16 sA[64 * 16];
    __nv_bfloat16 sB[SG_TUNED_TILE_N * 16];
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
__host__ __device__ __forceinline__ int64_t mb_tc_acts_floats(int T) {
    return mbtc::mb_acts_floats(T);
}
__host__ __device__ __forceinline__ int64_t mb_tc_workspace_floats(int T, int nCTA) {
    return mb_tc_acts_floats(T)
         + (int64_t)nCTA * mbtc::mb_tile_scratch_floats()
         + (int64_t)nCTA * mbtc::kPartElems
         + nCTA + 1;
}

template <OptId Opt>
__global__ void __launch_bounds__(SG_MB_TC_MEGA_BLOCK)
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
        float nll = mbtc::mbtc_forward_tile(w, g0, nrows, acts, sc, tok.tokens, tok.targets,
                                            sm.sA, sm.sB, sm.red);
        mbtc::mbtc_backward_tile(w, g0, nrows, B, acts, sc, part, tok.tokens, tok.targets,
                                 sm.sA, sm.sB, sm.red, sm.dBmat, sm.dCmat);
        if (threadIdx.x == 0) nll_acc += nll;
        __syncthreads();
    }
    if (threadIdx.x == 0) loss_part[cta] = nll_acc;
    bar.sync();   // B1: all acts (X + dY + dh0) + partials complete

    // ── P2: assemble all 28 grads. 8 projection dW output-stationary (gt %
    //    nCTA), dt_proj.bias, non-GEMM partial-reduce, embedding owner-scan. ──
    if (threadIdx.x == 0) mbtc::mbtc_build_dw_specs(acts, T, sm.spec);
    __syncthreads();
    mbtc::MbDwSpec* spec = sm.spec;
    const int n_dw = mbtc::mbtc_dw_total_tiles<SG_TUNED_TILE_N>(spec);
    for (int gt = cta; gt < n_dw; gt += nCTA)
        mbtc::mbtc_dw_run_tile<SG_TUNED_TILE_N>(spec, gt, grad, sm.sA, sm.sB);
    mbtc::mbtc_dw_biases(spec, grad);
    mbtc::mbtc_partial_reduce(part_base, grad, nCTA, cta);
    mbtc::mbtc_embed_owner_scan(acts, tok.tokens, T, grad, cta, nCTA);
    // Loss reduce (fp64 ordered) by CTA 0.
    if (cta == 0 && threadIdx.x == 0) {
        double s = 0.0;
        for (int c = 0; c < nCTA; ++c) s += (double)loss_part[c];
        *tok.loss_out = (float)(s / (double)B);
    }
    (void)loss_out;
    bar.sync_reset(ctx.g_next_task);   // B2: reduced grad ready; reset queue

    // ── P3: the REAL optimizer tail over the reduced grad (work-steal 28). ──
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

// Launcher (static-smem; the decoder TC contract). ncta_cap>0 caps the launched
// CTAs (the per-CTA scratch is nCTA×slab; a memory-tight TEST caps it). Grid
// barrier rendezvous is over the LAUNCHED count → hang-safe; determinism is per
// fixed nCTA (dW-tile / partial / embed owner maps read ctx.n_ctas). B%16 req.
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

    unsigned launch_ctas = (unsigned)n_sms;
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
