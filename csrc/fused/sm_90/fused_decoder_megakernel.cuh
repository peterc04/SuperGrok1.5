#ifndef SG_FUSED_SM90_FUSED_DECODER_MEGAKERNEL_CUH_
#define SG_FUSED_SM90_FUSED_DECODER_MEGAKERNEL_CUH_
// ============================================================================
// csrc/fused/sm_90/fused_decoder_megakernel.cuh — PHASE 1 of the TRUE L3 fused
// megakernel. ONE persistent kernel per training step runs the REAL transformer-
// decoder forward+backward AND the AdamW optimizer math, separated only by
// in-kernel grid barriers — real model math, real optimizer math, ZERO
// intermediate kernel launches.
//
// This composes:
//   * the REAL decoder fwd/bwd stages (model_stages_decoder.cuh — transcribed
//     line-for-line from the verified PyTorch oracle, asserted bit-identical to
//     autograd, and structurally mirrored on CPU),
//   * the existing persistent substrate (megakernel_common.cuh: task queue, the
//     hand-built sense-reversing GridBarrier),
//   * the existing REAL optimizer tail (opt_components.cuh::apply_optimizer<Opt>).
//
// STAGE / BARRIER LAYOUT (5 phases, 4 grid barriers — see the per-stage comments):
//   P0  zero the per-CTA grad-partial workspace + per-CTA loss slots (each CTA
//       zeroes its OWN slice — no cross-CTA contention, no host memset needed).
//   --- grid barrier (B0): all partials zeroed before any accumulation ---
//   P1  BATCH-PARALLEL fwd+bwd: each CTA owns a FIXED contiguous batch slice
//       (by blockIdx.x), processes its samples ONE AT A TIME (CTA-cooperative),
//       accumulating each sample's weight-grad contribution into the CTA's OWN
//       partial buffer (gw + cta*total) with a single-owner-thread-per-element
//       rule (no atomics → deterministic), and summing its slice's NLL (fp32)
//       into the CTA's loss slot.
//   --- grid barrier (B1): all CTAs' partials + loss complete ---
//   P2  DETERMINISTIC cross-CTA reduce: sum partial[0..nCTA) in ASCENDING CTA
//       index into the global `grad` buffer (no float atomics; the per-element
//       summation ORDER is fixed, so the work-steal queue MAY pick who reduces
//       which element-range — determinism is in the order, not the picker). The
//       loss slots are summed in fp64 → one device float (loss/B), by CTA 0.
//   --- grid barrier (B2): reduced grad ready in global ---
//   P3  the existing apply_optimizer<AdamW> tail consumes the reduced grad
//       in-place over the flat param (work-steal queue over the 30 tensors).
//   (no barrier after P3 — kernel exits; the step is complete.)
//
// HONESTY: no placeholder math anywhere on this path. fp32 compute is the
// correctness baseline; a bf16-compute follow-up would be a flag defaulting to
// THIS fp32 path (not yet wired — see the report).
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/opt_components.cuh"
#include "csrc/fused/sm_90/decoder_layout.cuh"
#include "csrc/fused/sm_90/model_stages_decoder.cuh"
#include "csrc/backends/cuda/sm_90/warp_specialize.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

// ============================================================================
//  SG_TUNED_GEMM_IMPL — the per-cell GEMM-engine selector (DESIGN-TC-PIPELINE.md
//  §9 / the owner "BOTH paths compiled, the tuner picks" directive). The scalar
//  fp32 owner-computes path (model_stages_decoder.cuh) and the bf16 wgmma
//  tensor-core path (model_stage_decoder_tc.cuh) are compiled ALONGSIDE each
//  other; the kernel-autotuner injects -DSG_TUNED_GEMM_IMPL=<token> per-TU to
//  pick one. Absent the macro, the SCALAR path compiles verbatim (CONTRACT
//  rule 3: an untuned build is a correct build) — this is the shipped default
//  and the path all of test_megakernel_vs_eager.py's L3-REAL decoder gates
//  exercise; adding this seam must leave those numbers BIT-IDENTICAL.
//
//  Tokens are integers (the C preprocessor cannot compare strings, and the
//  autotuner's -D injection passes an integer literal):
//      SG_GEMM_IMPL_SCALAR = 0   (default; the verbatim fp32 path below)
//      SG_GEMM_IMPL_WGMMA  = 1   (the Fork-B bf16 tensor-core cell)
//
//  WGMMA-PATH STATUS (honest, no-suppression — MEMORY.md "fix components, never
//  disable them"): the validated unit today is the GEMM ENGINE
//  (model_stage_decoder_tc.cuh::tc_gemm_block_unpipelined + the wgmma.cuh /
//  tile_pipeline.cuh substrate), silicon-gated 13/13 by test_decoder_tc.py
//  (fwd/dX/dW vs the bf16-rounded fp64 oracle, A=I localizations exactly 0.0,
//  dW A/A/A bit-identical) on a validated 18/18 substrate. The full Fork-B
//  fwd+bwd CELL DRIVER (the phase-restructured P0→P1(fwd+bwd, acts→HBM)→
//  optimizer megakernel that REPLACES the body below) is DESIGN §11 work item
//  R2.3 and is NOT yet authored. Selecting the wgmma token therefore FAILS THE
//  COMPILE LOUDLY (the #error below) rather than silently shipping the scalar
//  body under a wgmma name — a wgmma-requested cell that secretly ran scalar
//  would be exactly the functionality suppression the owner forbids. The seam
//  is in place and dormant; wiring the driver into it is the bounded next step.
// ============================================================================
#define SG_GEMM_IMPL_SCALAR 0
#define SG_GEMM_IMPL_WGMMA  1
#ifndef SG_TUNED_GEMM_IMPL
#define SG_TUNED_GEMM_IMPL SG_GEMM_IMPL_SCALAR
#endif
#if (SG_TUNED_GEMM_IMPL != SG_GEMM_IMPL_SCALAR) && \
    (SG_TUNED_GEMM_IMPL != SG_GEMM_IMPL_WGMMA)
#error "SG_TUNED_GEMM_IMPL must be SG_GEMM_IMPL_SCALAR (0) or SG_GEMM_IMPL_WGMMA (1)"
#endif
// WGMMA-PATH (R2.3, NOW LANDED): the Fork-B tensor-core CELL DRIVER. Selecting
// the wgmma token pulls in the validated GEMM engine (model_stage_decoder_tc
// .cuh, test_decoder_tc.py 13/13 on the 18/18 substrate) and the
// phase-restructured fwd+bwd+AdamW persistent kernel below (fused_decoder_
// megakernel_tc / launch_fused_decoder_megakernel_tc). The scalar default path
// is UNCHANGED and its gates stay bit-identical; this is a PARALLEL kernel.
#if (SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA)
#include "csrc/fused/sm_90/model_stage_decoder_tc.cuh"
#endif

namespace sg { namespace fused { namespace sm90 {

// Rebase a FusedOptState's per-element state pointers to a parameter-tensor
// slice at `off` within the flat [m|v|extra] layout. Per-TENSOR fields (the
// NeuralGrok psi MLP weights) and all scalars pass through unchanged.
template <OptId Opt>
__device__ __forceinline__ FusedOptState
rebase_state(const FusedOptState& s, int64_t off) {
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

// The L3-REAL decoder megakernel needs the token path + a grad-partial workspace
// in addition to the FuseTier::{L1,L3} ABI. We keep it a SEPARATE kernel +
// launcher (not folded into fused_megakernel) so the surrogate path is untouched.
//
// Workspace layout (one flat float buffer the host allocates + the kernel owns):
//   [0 .. nCTA*total)              : per-CTA grad partials (cta-major)
//   [nCTA*total .. nCTA*total+nCTA): per-CTA loss partials (NLL sum per slice)
//   [.. +1)                        : the reduced scalar loss (loss/B) the host reads
// total == kDecTotalElems == 422755.

struct DecoderTokenCtx {
    const int* tokens;   // [B, kSeq] int32 token ids in [0, kVocab)
    const int* targets;  // [B]       int32 target ids
    int        B;        // batch size (full-batch in the race ≈ 4191)
    float*     workspace; // grad partials + loss partials + reduced loss
    float*     loss_out;  // device float the kernel writes the mean loss into
};

// ── The persistent megakernel (L3-REAL). gridDim.x = #SMs (one CTA/SM), 256
//    threads/CTA. The smem holds ONE DecSampleSmem (≈42 KB, < 48 KB static cap —
//    no dynamic-smem opt-in, so the occupancy≥1 guard with dynamicSMemBytes=0 in
//    the launcher is unchanged). ────────────────────────────────────────────────
// sizes/offsets are NOT host-passed: the per-tensor numel/offset live in the
// generated __constant__ tables kDecSizes/kDecOffsets (decoder_layout.cuh), so
// the reduce + optimizer phases read them directly. This also lets the host side
// (dispatch.cpp) avoid building any layout tensors.
template <OptId Opt>
__global__ void __launch_bounds__(SG_TUNED_MEGA_BLOCK)
fused_decoder_megakernel(PersistentContext ctx,
                         float* __restrict__ params,
                         DecoderTokenCtx tok,
                         float* __restrict__ grad,        // reduced grad [total]
                         float lr, int step, FusedOptState st) {
    __shared__ DecSampleSmem sm;
    GridBarrier bar = ctx.barrier();
    const int cta = blockIdx.x;
    const int nCTA = (int)ctx.n_ctas;
    const int64_t total = kDecTotalElems;
    float* my_partial = tok.workspace + (int64_t)cta * total;   // this CTA's dW
    float* loss_part = tok.workspace + (int64_t)nCTA * total;   // [nCTA]

    // §3.4 register repartition (producer WG gives back, consumer WG claims).
    const int wg = threadIdx.x / 128;
    // warp-specialize prims live in sg::sm90::wgs (this file is sg::fused::sm90)
    if (wg == 0) ::sg::sm90::wgs::warpgroup_reg_dealloc<32>();
    else         ::sg::sm90::wgs::warpgroup_reg_alloc<200>();

    // ── P0: zero this CTA's grad-partial slice + its loss slot. ───────────────
    for (int64_t i = threadIdx.x; i < total; i += blockDim.x) my_partial[i] = 0.0f;
    if (threadIdx.x == 0) loss_part[cta] = 0.0f;
    bar.sync();   // B0

    // ── P1: batch-parallel fwd+bwd. Fixed contiguous slice [b0,b1) for this CTA.
    DecWeights w = dec_bind(params);
    DecGrad g = dec_bind_grad(my_partial);
    // Even split with remainder to the low CTAs (contiguous, deterministic).
    const int B = tok.B;
    const int base = B / nCTA, rem = B % nCTA;
    const int b0 = cta * base + (cta < rem ? cta : rem);
    const int cnt = base + (cta < rem ? 1 : 0);
    const int b1 = b0 + cnt;
    __shared__ int tok_s[dec::kSeq];   // this sample's token ids (broadcast)
    __shared__ int tgt_s;
    float nll_acc = 0.0f;              // fp32 slice accumulator (thread-0 holds it)
    for (int b = b0; b < b1; ++b) {
        if (threadIdx.x < dec::kSeq) tok_s[threadIdx.x] = tok.tokens[(int64_t)b * dec::kSeq + threadIdx.x];
        if (threadIdx.x == 0) tgt_s = tok.targets[b];
        __syncthreads();
        float nll = dec_forward_sample(w, tok_s, tgt_s, &sm);
        dec_backward_sample(w, g, tok_s, tgt_s, B, &sm);
        if (threadIdx.x == 0) nll_acc += nll;   // fixed-order fp32 sum within slice
        __syncthreads();   // sample boundary: all grad writes done before reuse
    }
    if (threadIdx.x == 0) loss_part[cta] = nll_acc;
    bar.sync();   // B1: all CTA partials + loss slots complete

    // ── P2: deterministic cross-CTA reduce. Work-steal the param ELEMENT-RANGES
    //    (one task = one parameter tensor; sum its elements across CTAs in
    //    ascending CTA index). The summation ORDER (ascending cta) is fixed, so
    //    the result is deterministic regardless of which CTA grabs the task. ────
    {
        __shared__ int task_slot;
        TaskQueue q = ctx.queue();
        for (int t = q.next_block(&task_slot); t < kDecNumTensors;
             t = q.next_block(&task_slot)) {
            const int n = kDecSizes[t];
            const int64_t off = (int64_t)kDecOffsets[t];
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                float acc = 0.0f;
                // ascending CTA index → fixed fp32 summation order (deterministic).
                for (int c = 0; c < nCTA; ++c)
                    acc += tok.workspace[(int64_t)c * total + off + i];
                grad[off + i] = acc;
            }
        }
    }
    // Loss reduction (fp64 ordered) by CTA 0 only — the 1e-5 loss rel-tol is the
    // tightest gate; fp32 atomic-summing ~4191 terms can miss it.
    if (cta == 0 && threadIdx.x == 0) {
        double s = 0.0;
        for (int c = 0; c < nCTA; ++c) s += (double)loss_part[c];
        float mean = (float)(s / (double)B);
        *tok.loss_out = mean;
    }
    bar.sync_reset(ctx.g_next_task);   // B2: reduced grad ready; reset queue for P3

    // ── P3: the REAL optimizer tail over the reduced grad (work-steal the 30
    //    tensors). apply_optimizer<Opt> is the canonical csrc/algorithms math. ──
    st.lr = lr;
    {
        __shared__ int task_slot;
        TaskQueue q = ctx.queue();
        for (int t = q.next_block(&task_slot); t < kDecNumTensors;
             t = q.next_block(&task_slot)) {
            const int n = kDecSizes[t];
            const int64_t off = (int64_t)kDecOffsets[t];
            const FusedOptState ts = rebase_state<Opt>(st, off);
            float* __restrict__ p = params + off;
            const float* __restrict__ gg = grad + off;
            for (int i = threadIdx.x; i < n; i += blockDim.x)
                apply_optimizer<Opt>(p, gg, (int64_t)i, step, ts);
        }
    }
}

// ── Host launcher — one persistent CTA per SM, 256 threads/CTA (2 warp-groups).
//    Mirrors launch_fused_megakernel's hang-freedom contract: occupancy≥1 or
//    refuse (the GridBarrier rendezvous of n_ctas CTAs can never complete if a
//    CTA can't be placed). Zeroes the barrier+task counters per launch. ─────────
template <OptId Opt>
cudaError_t launch_fused_decoder_megakernel(
        PersistentContext ctx, float* params, DecoderTokenCtx tok,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 0;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;

    int occ = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (const void*)&fused_decoder_megakernel<Opt>, SG_TUNED_MEGA_BLOCK,
        /*dynamicSMemBytes=*/0);
    if (err != cudaSuccess) return err;
    // Hang-freedom: at least one CTA per SM must be resident or the grid barrier
    // can never be satisfied. The L3-REAL kernel uses ≈42 KB static smem + a high
    // register count; if it cannot place one block/SM, REFUSE rather than hang.
    assert(occ >= 1 &&
           "fused_decoder_megakernel: 0 blocks/SM — GridBarrier would hang. The "
           "decoder smem (~42KB) + regs exceed one-block-per-SM occupancy; "
           "reduce footprint or fall back to the L1 per-op path.");
    if (occ < 1) return cudaErrorLaunchOutOfResources;

    const unsigned launch_ctas = (unsigned)n_sms;
    ctx.n_ctas = launch_ctas;

    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }

    dim3 grid(launch_ctas), block(SG_TUNED_MEGA_BLOCK);
    fused_decoder_megakernel<Opt><<<grid, block, 0, stream>>>(
        ctx, params, tok, grad, lr, step, st);
    return cudaGetLastError();
}

// ════════════════════════════════════════════════════════════════════════════
//  WGMMA CELL DRIVER (DESIGN-TC-PIPELINE.md Fork B, R2.3). Compiled only under
//  the wgmma token; the scalar path above is untouched.
// ════════════════════════════════════════════════════════════════════════════
#if (SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA)

// TC megakernel threads: 256 (the engine's producer+consumer warpgroup layout;
// wgmma is warpgroup-scoped on threads 0..127). We do NOT apply the scalar
// path's asymmetric setmaxnreg (dealloc<32> on WG0) — that would STARVE the MMA
// warpgroup (WG0 issues the wgmma; a 128-wide accumulator needs 64 fp32 regs/
// thread). The validated selftest ran no split; we match it.
#define SG_TC_MEGA_BLOCK 256

// ── smem arena for the unpipelined engine: one A(64×16) + one B(N×16) bf16 tile
//    + the 256-float reduction slot. 16B-aligned. N = SG_TUNED_TILE_N. The 9 dW
//    specs live here too (identical for all threads → shared, not per-thread
//    stack: keeps the launch's local-memory reservation small so the persistent
//    kernel places on a memory-tight GPU). ──
struct DecTcSmem {
    __nv_bfloat16 sA[64 * 16];
    __nv_bfloat16 sB[SG_TUNED_TILE_N * 16];
    float red[256];
    dectc::DecDwSpec spec[9];
};

// ── TC workspace layout (carved from tok.workspace; the host sizes it for the
//    TC path — see dec_tc_workspace_floats). Regions (float units):
//      [0 .. acts_f)                      : DecActs bf16 region (reinterpreted)
//      [acts_f .. acts_f + nCTA*scratch)  : per-CTA tile scratch (f32)
//      [.. + nCTA*kLnVecElems)            : per-CTA LN-vec partials (f32)
//      [.. + nCTA)                        : per-CTA loss slots (f32)
//      [.. + 1)                           : reduced scalar loss (host reads it)
//    acts_f = ceil(acts_bf16 / 2). ──
__host__ __device__ __forceinline__ int64_t dec_tc_acts_floats(int T, int B) {
    // mirror dectc::dec_acts_bind's running offset (bf16 elems) → floats.
    const int64_t d = dec::kD, dff = dec::kDff, V = dec::kVocab, L = dec::kLayers;
    const int64_t Td = (int64_t)T * d, T3d = (int64_t)T * 3 * d, Tff = (int64_t)T * dff;
    int64_t bf = 0;
    for (int li = 0; li < L; ++li) bf += Td + Td + Td + Tff + T3d + Td + Tff + Td;
    bf += (int64_t)B * d + (int64_t)B * V + Td;     // X_hn + dY_logits + dh0
    return (bf + 1) / 2;
}
__host__ __device__ __forceinline__ int64_t dec_tc_workspace_floats(int T, int B, int nCTA) {
    return dec_tc_acts_floats(T, B)
         + (int64_t)nCTA * dectc::dec_tile_scratch_total_f32()
         + (int64_t)nCTA * dectc::kLnVecElems
         + nCTA + 1;
}

template <OptId Opt>
__global__ void __launch_bounds__(SG_TC_MEGA_BLOCK)
fused_decoder_megakernel_tc(PersistentContext ctx,
                            float* __restrict__ params,
                            DecoderTokenCtx tok,
                            float* __restrict__ grad,
                            float lr, int step, FusedOptState st) {
    __shared__ DecTcSmem sm;
    GridBarrier bar = ctx.barrier();
    const int cta = blockIdx.x;
    const int nCTA = (int)ctx.n_ctas;
    const int B = tok.B;
    const int T = B * dec::kSeq;

    // Workspace partition.
    float* ws = tok.workspace;
    const int64_t acts_f = dec_tc_acts_floats(T, B);
    __nv_bfloat16* acts_base = reinterpret_cast<__nv_bfloat16*>(ws);
    float* scratch_base = ws + acts_f;
    const int64_t scratch_per = dectc::dec_tile_scratch_total_f32();
    float* lnvec_base = scratch_base + (int64_t)nCTA * scratch_per;
    float* loss_part  = lnvec_base + (int64_t)nCTA * dectc::kLnVecElems;
    float* loss_out   = loss_part + nCTA;

    dectc::DecActs acts = dectc::dec_acts_bind(acts_base, T, B);
    dectc::DecTileScratch sc = dectc::dec_tile_scratch_bind(scratch_base + (int64_t)cta * scratch_per);
    float* my_lnvec = lnvec_base + (int64_t)cta * dectc::kLnVecElems;

    DecWeights w = dec_bind(params);

    // ── P0: zero this CTA's LN-vec partials + loss slot (dW/embed grads are
    //    written-once → no pre-zero). ──
    for (int i = threadIdx.x; i < dectc::kLnVecElems; i += blockDim.x) my_lnvec[i] = 0.0f;
    if (threadIdx.x == 0) loss_part[cta] = 0.0f;
    bar.sync();   // B0

    // ── P1: token-tile-parallel fwd+bwd. Each CTA grid-strides over tiles of
    //    kTileM rows; for its tile it runs fwd (→ acts X, NLL) then bwd (→ acts
    //    dY, dh0, LN-vec partials). Barrier-free within the tile. ──
    const int nrows_tile = dectc::kTileM;
    const int n_tiles = (T + nrows_tile - 1) / nrows_tile;
    float nll_acc = 0.0f;
    for (int ti = cta; ti < n_tiles; ti += nCTA) {
        const int g0 = ti * nrows_tile;
        const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
        float nll = dectc::dectc_forward_tile(w, g0, nrows, acts, sc, tok.tokens, tok.targets,
                                              sm.sA, sm.sB, sm.red);
        dectc::dectc_backward_tile(w, g0, nrows, B, acts, sc, tok.targets,
                                   my_lnvec, sc.work2, sm.sA, sm.sB, sm.red);
        if (threadIdx.x == 0) nll_acc += nll;
        __syncthreads();
    }
    if (threadIdx.x == 0) loss_part[cta] = nll_acc;
    bar.sync();   // B1: all acts (X + dY) + LN-vec partials complete

    // ── P2: assemble all 30 grads into `grad`. dW output-stationary (gt %
    //    nCTA), biases, embedding owner-scan, LN-vec reduce. No partials. The 9
    //    dW specs are built into SHARED smem (thread 0; identical for all) so the
    //    9-spec array is NOT on every thread's stack (shrinks the launch's local
    //    reservation — the persistent kernel must place on a memory-tight GPU). ──
    if (threadIdx.x == 0) dectc::dectc_build_dw_specs(acts, B, T, sm.spec);
    __syncthreads();
    dectc::DecDwSpec* spec = sm.spec;
    const int n_dw = dectc::dectc_dw_total_tiles<SG_TUNED_TILE_N>(spec);
    for (int gt = cta; gt < n_dw; gt += nCTA)
        dectc::dectc_dw_run_tile<SG_TUNED_TILE_N>(spec, gt, grad, sm.sA, sm.sB);
    dectc::dectc_dw_biases(spec, grad);
    dectc::dectc_embed_owner_scan(acts, tok.tokens, T, grad, cta, nCTA);
    dectc::dectc_lnvec_reduce(lnvec_base, grad, nCTA, cta);
    // Loss reduce (fp64) by CTA 0.
    if (cta == 0 && threadIdx.x == 0) {
        double s = 0.0;
        for (int c = 0; c < nCTA; ++c) s += (double)loss_part[c];
        *tok.loss_out = (float)(s / (double)B);
    }
    (void)loss_out;
    bar.sync_reset(ctx.g_next_task);   // B2: reduced grad ready; reset queue

    // ── P3: scalar optimizer tail over the reduced grad (REUSED verbatim). ──
    st.lr = lr;
    {
        __shared__ int task_slot;
        TaskQueue q = ctx.queue();
        for (int t = q.next_block(&task_slot); t < kDecNumTensors;
             t = q.next_block(&task_slot)) {
            const int n = kDecSizes[t];
            const int64_t off = (int64_t)kDecOffsets[t];
            const FusedOptState ts = rebase_state<Opt>(st, off);
            float* __restrict__ p = params + off;
            const float* __restrict__ gg = grad + off;
            for (int i = threadIdx.x; i < n; i += blockDim.x)
                apply_optimizer<Opt>(p, gg, (int64_t)i, step, ts);
        }
    }
}

// ncta_cap (default 0): if >0, launch min(n_sms, ncta_cap) CTAs instead of one
// per SM. The grid barrier rendezvous is over the LAUNCHED count (ctx.n_ctas),
// so any cap is hang-safe as long as the workspace + dW-tile/embedding ownership
// use the SAME nCTA (they read ctx.n_ctas). The shipped path passes 0 (full
// saturation); a memory-constrained TEST passes a small cap so the per-CTA
// scratch fits (the scratch is nCTA×slab). Determinism is preserved per fixed
// nCTA (the dW/embedding owner maps are functions of nCTA).
template <OptId Opt>
cudaError_t launch_fused_decoder_megakernel_tc(
        PersistentContext ctx, float* params, DecoderTokenCtx tok,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream,
        int ncta_cap = 0) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 0;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;

    // Hang-freedom: certify one block/SM occupancy with the ACTUAL dynamic smem
    // (0 here — DecTcSmem is static). If the static smem + 200-reg consumer
    // can't place one block/SM, REFUSE (the grid barrier would hang).
    int occ = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (const void*)&fused_decoder_megakernel_tc<Opt>, SG_TC_MEGA_BLOCK,
        /*dynamicSMemBytes=*/0);
    if (err != cudaSuccess) return err;
    if (occ < 1) return cudaErrorLaunchOutOfResources;

    unsigned launch_ctas = (unsigned)n_sms;
    if (ncta_cap > 0 && (unsigned)ncta_cap < launch_ctas) launch_ctas = (unsigned)ncta_cap;
    ctx.n_ctas = launch_ctas;
    // B%16 required (the dW K-loop contracts K=T=B*kSeq and K=B in 16-step atoms).
    if ((tok.B % 16) != 0) return cudaErrorInvalidValue;

    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }

    dim3 grid(launch_ctas), block(SG_TC_MEGA_BLOCK);
    fused_decoder_megakernel_tc<Opt><<<grid, block, 0, stream>>>(
        ctx, params, tok, grad, lr, step, st);
    return cudaGetLastError();
}

#endif  // SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_FUSED_DECODER_MEGAKERNEL_CUH_
