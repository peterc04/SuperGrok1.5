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
// STAGED-optimizer in-kernel precompute stages (Prodigy d-reduction, …). Pulls in
// the canonical prodigy.h reduction math + the deterministic block reductions; the
// TC megakernel's Prodigy branch drives prodigy_precompute_reduce_phaseA + an EMA
// d-update owner block between B2 and P3. Included only under the wgmma token (the
// scalar default path has no STAGED tail).
#include "csrc/fused/sm_90/opt_stages_precompute.cuh"
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
    if (t.param_init) t.param_init += off;   // Prodigy trajectory anchor p0
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
    // kDecTcStages A(64×16) + kDecTcStages B(N×16) bf16 tiles — the GEMM K-loop
    // double-buffer ring (slot s at sA + s*64*16 / sB + s*N*16). At S=2 + N=128
    // the ring is 2·(2KB+4KB)=12KB; DecTcSmem total ~13.5KB ≪ the 48KB static cap.
    __nv_bfloat16 sA[dectc::kDecTcStages * 64 * 16];
    __nv_bfloat16 sB[dectc::kDecTcStages * SG_TUNED_TILE_N * 16];
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
// dW split-K partial floats (0 when G==1 → no extra scratch, single-CTA path).
__host__ __device__ __forceinline__ int64_t dec_tc_dw_part_floats() {
    return (dectc::kDecDwSplitK > 1) ? dectc::dec_dw_part_floats(dectc::kDecDwSplitK) : 0;
}
// STAGED-optimizer cross-CTA reduction scratch (Prodigy d-estimate). The Prodigy
// stage publishes per-CTA (r,s) slots (2*nCTA) + a reduced-d slot (1) — an owner-
// computes tree (opt_stages_precompute.cuh), NO float atomic. Sized for the
// LARGEST nCTA (one CTA/SM = #SMs); tiny (≤ 2*132+1 ≈ 1 KB) and carved
// UNCONDITIONALLY so the opt-agnostic host launcher (dec_tc_launcher_scratch)
// allocates one workspace that fits every OptId. Unused by AdamW/Lion/… (their
// P3 never touches this region), so adding it leaves those cells byte-identical.
__host__ __device__ __forceinline__ int64_t dec_tc_opt_reduce_floats(int nCTA) {
    return (int64_t)2 * nCTA + 1;            // [r slots | s slots | reduced d]
}
// ── Muon (STAGED grid-cooperative Newton-Schulz) per-matrix scratch. IDENTICAL in
//    shape to the vit twin (vit_tc_muon_floats): the NS chain runs ONE 2D weight at
//    a time over all CTAs; the scratch holds X/A/AX/AAX/orth for the LARGEST 2D weight
//    + the per-CTA Frobenius-norm partials + inv_norm. The momentum buffer (muon_buf)
//    is NOT here — it PERSISTS across steps as optimizer state, bound to the m slice
//    (st.exp_avg). Largest decoder 2D weight: ff.0/ff.2 = 512×128 = 65536 numel; largest
//    rows = 512 ⇒ A = 512×512 (dectc::kDecMuonMaxNumel/kDecMuonMaxRows). Carved
//    UNCONDITIONALLY (≈ 4·65536 + 512² + nCTA + 1 floats ≈ 2 MB) so the opt-agnostic
//    launcher (dec_tc_launcher_scratch) fits every OptId; unused by every non-Muon cell
//    (its P2.7/P3-2D branches are if-constexpr'd out → byte-identical). ──
__host__ __device__ __forceinline__ int64_t dec_tc_muon_floats(int nCTA) {
    // X + AX + AAX + orth (each maxNumel) + A (maxRows²) + nrm_partials(nCTA) + inv_norm(1)
    return (int64_t)4 * dectc::kDecMuonMaxNumel
         + (int64_t)dectc::kDecMuonMaxRows * dectc::kDecMuonMaxRows
         + nCTA + 1;
}
__host__ __device__ __forceinline__ int64_t dec_tc_workspace_floats(int T, int B, int nCTA) {
    return dec_tc_acts_floats(T, B)
         + (int64_t)nCTA * dectc::dec_tile_scratch_total_f32()
         + (int64_t)nCTA * dectc::kLnVecElems
         + nCTA + 1
         + dec_tc_dw_part_floats()            // split-K dW partials (G>1)
         + dec_tc_opt_reduce_floats(nCTA)     // STAGED-opt (Prodigy) reduce slots
         + dec_tc_muon_floats(nCTA);          // STAGED-opt (Muon) NS per-matrix scratch
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
    // Split-K dW partials (G>1): the (gt,kc) 64×N partial tiles, carved AFTER the
    // loss slot (matches dec_tc_workspace_floats's term order). G==1 → dw_part unused.
    float* dw_part    = loss_out + 1;
    const int kDwG    = dectc::kDecDwSplitK;
    // STAGED-opt cross-CTA reduce slots (Prodigy d), carved AFTER the dW partials
    // (matches dec_tc_workspace_floats's term order). Unused unless Opt==Prodigy.
    float* opt_reduce = dw_part + dec_tc_dw_part_floats();
    // Muon NS per-matrix scratch, carved AFTER the Prodigy reduce slots (matches
    // dec_tc_workspace_floats's term order — carving it LAST keeps every prior
    // region's offset unchanged, so the 5 already-green cells are byte-identical).
    // Unused unless Opt==Muon. Layout (mirrors the vit twin):
    //   [muon_X | muon_AX | muon_AAX | muon_orth] (each kDecMuonMaxNumel)
    //   [muon_A (kDecMuonMaxRows²)] [nrm_partials (nCTA)] [inv_norm (1)]
    float* muon_base = opt_reduce + dec_tc_opt_reduce_floats(nCTA);

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
    if (kDwG > 1) {
        // SPLIT-K (multi-CTA tiling): fan (n_dw·G) (tile,chunk) partials over the
        // grid so the ~62% idle SMs do work; deterministic ascending-chunk reduce.
        for (int item = cta; item < n_dw * kDwG; item += nCTA) {
            const int gt = item / kDwG, kc = item % kDwG;
            dectc::dectc_dw_run_tile_splitk<SG_TUNED_TILE_N>(spec, gt, kc, kDwG, dw_part, sm.sA, sm.sB);
        }
        bar.sync();   // all (gt,kc) partials complete before the reduce reads them
        dectc::dectc_dw_reduce_splitk<SG_TUNED_TILE_N>(spec, n_dw, kDwG, dw_part, grad, cta, nCTA);
    } else {
        for (int gt = cta; gt < n_dw; gt += nCTA)
            dectc::dectc_dw_run_tile<SG_TUNED_TILE_N>(spec, gt, grad, sm.sA, sm.sB);
    }
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

    // ── P2.5 (GrokAdamW ONLY): GLOBAL grad-norm clip coefficient. Eager
    //    grokadamw clips the WHOLE grad set to grad_clip via a GLOBAL L2 norm
    //    (clip_grad_norms_device_side → total_norm = sqrt(Σ_i ‖g_i‖²),
    //    clip_coef = grad_clip/(total_norm+1e-6) when total_norm>grad_clip,
    //    else 1) BEFORE the apply. We replicate it on the REDUCED grad with a
    //    deterministic ascending reduction (no float atomics): each CTA sums a
    //    contiguous element-range into a per-CTA partial slot, CTA0 sums the
    //    partials in ascending CTA order → total_norm → clip_coef, broadcast via
    //    a workspace slot. The grad buffer is NOT mutated (the return_grad oracle
    //    + the eager-side clip must both see the unclipped reduced grad); the
    //    coefficient is applied per-element inside apply_optimizer<GrokAdamW>.
    //    Guarded so every other opt's P3 is byte-identical (no extra barrier/work).
    if constexpr (Opt == OptId::GrokAdamW) {
        // Reuse the (now-consumed) loss workspace: loss_part[nCTA] holds per-CTA
        // partial sum-of-squares; loss_out (1 float) broadcasts clip_coef. The
        // reduced loss is already in *tok.loss_out (state), so this is free scratch.
        float* sq_part = loss_part;          // [nCTA] per-CTA Σ g²  (ascending reduce)
        float* coef_bc = loss_out;           // [1] broadcast clip_coef
        // Per-CTA contiguous element range over `grad` (the reduced grad, [total]).
        const int64_t total = kDecTotalElems;
        const int64_t base = total / nCTA, rem = total % nCTA;
        const int64_t e0 = (int64_t)cta * base + (cta < rem ? cta : rem);
        const int64_t ecnt = base + (cta < rem ? 1 : 0);
        // Thread-local partial → block reduce (fixed tree) → thread0 writes the slot.
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
        // CTA0 ascending-order reduce → total_norm → clip_coef (deterministic).
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

    // ── P2.6 (PRODIGY ONLY): STAGED cross-ALL-tensors d-estimate. The apply tail
    //    (apply_optimizer<Prodigy>) reads st.d_factor (the effective LR scale d),
    //    a GLOBAL reduction over EVERY element of EVERY tensor. We compute it here,
    //    BYTE-FAITHFUL to the live eager multi-tensor path (prodigy_sm90.cuh:465-
    //    544, the order prodigy.py → _ops.prodigy_fused_step actually executes):
    //      d_prev  = persisted d_lr  (step 1: d0 cold-start — the zero-init state
    //                slot would give d_prev=0 ⇒ d=0 ⇒ frozen params; eager inits
    //                _d_lr=d0=1e-6, so seed it here, the grokfast-style step-1 fix)
    //      r_ema  <- beta3·r_ema + Σ d_prev²·<g, p0−p>     (decay persisted SCALAR,
    //      s_ema  <- beta3·s_ema + Σ d_prev²·|g|            then add this step's Σ)
    //      d       = max(d_prev, d_coef·r_ema/|s_ema|)      (prodigy_update_d; d_coef
    //                scales ONLY the candidate — persisted r_ema stays UNSCALED)
    //    DETERMINISM (COMPONENT_CONTRACT): NO float atomic. Each CTA publishes its
    //    (r,s) into per-CTA slots (opt_reduce) → grid barrier → CTA0 owner-sums in
    //    ascending index order → writes d back to the persisted slot + a broadcast
    //    slot. The decay is on the persisted SCALARS (not the per-CTA partials):
    //    the work-steal queue reassigns tensors to CTAs across steps, so a per-CTA
    //    EMA is undefined — the live form is a scalar EMA (prodigy_sm90.cuh:488).
    //    Guarded so every other opt's P3 is byte-identical (no extra barrier/work).
    if constexpr (Opt == OptId::Prodigy) {
        PrecomputeWorkspace pw{};
        pw.prodigy_partials = opt_reduce;            // [r slots | s slots]
        pw.prodigy_d        = opt_reduce + 2 * nCTA; // reduced-d broadcast slot
        // d_prev: persisted d_lr (slot 2 of prodigy_persist), or d0 at step 1.
        const float d_prev = (step == 1) ? st.d0 : st.prodigy_persist[2];
        st.d_factor = d_prev;   // phaseA reads st.d_factor as d_prev (prodigy.h)
        // Phase A: each CTA accumulates Σ d_prev²·<g,p0−p> / Σ d_prev²·|g| over its
        // claimed tensors → per-CTA (r,s) slots. Drains the task queue (the P3
        // re-drain below needs a queue reset, done at the barrier).
        prodigy_precompute_reduce_phaseA(ctx, params, st.param_init, grad,
                                         kDecSizes, kDecOffsets, d_prev, pw);
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

    // ── P2.7 (Muon ONLY): grid-cooperative Newton-Schulz orthogonalization of the
    //    2D weights (INTEGRATION-OPTSTAGES §3). IDENTICAL to the vit twin
    //    (fused_vit_megakernel.cuh P2.7) — same shared opt_stages_precompute.cuh
    //    helpers, same barrier sequence — only the per-model 2D table (dectc::
    //    kDecMuon2D, 11 matrices including tok[99,128]/pos[4,128]) and offset array
    //    (kDecOffsets) differ. For EACH 2D matrix all CTAs cooperate: buf=μ·buf+g
    //    (buf is the PERSISTENT m-slice — momentum state, NOT transient), ‖buf‖_F via
    //    per-CTA partials → inv_norm, X=buf·inv_norm, then ns_steps × { A=XXᵀ → AX=A·X
    //    → AAX=A·AX → orth=a·X+b·AX+c·AAX, swap }, then the canonical muon_update_step
    //    apply (decay·p + neg_lr_scale·orth). The 1D / non-2D weights (biases, LN γ/β)
    //    take the AdamW aux tail in P3. The elementwise bodies CALL muon.h
    //    (muon_momentum_normalize_step via the phaseA buf body, muon_ns_combine_step,
    //    muon_update_step); the matmuls are the cited new device code (the eager path
    //    delegates to torch::mm/cuBLAS). Guarded so every other opt is byte-identical
    //    (no extra barriers / work).
    if constexpr (Opt == OptId::Muon) {
        // Carve the per-matrix NS scratch (sized for the largest 2D weight).
        PrecomputeWorkspace pw{};
        pw.muon_X            = muon_base;
        pw.muon_AX           = pw.muon_X   + dectc::kDecMuonMaxNumel;
        pw.muon_AAX          = pw.muon_AX  + dectc::kDecMuonMaxNumel;
        pw.muon_orth         = pw.muon_AAX + dectc::kDecMuonMaxNumel;
        pw.muon_A            = pw.muon_orth + dectc::kDecMuonMaxNumel;
        pw.muon_nrm_partials = pw.muon_A   + (int64_t)dectc::kDecMuonMaxRows * dectc::kDecMuonMaxRows;
        pw.muon_inv_norm     = pw.muon_nrm_partials + nCTA;
        const float momentum = st.beta1;          // Muon momentum (eager Muon: betas[0])
        const int   ns_steps = 5;                 // bindings.cpp default
        for (int mi = 0; mi < dectc::kDecNumMuon2D; ++mi) {
            const dectc::DecMuon2D M = dectc::kDecMuon2D[mi];
            const int rows = M.rows, cols = M.cols;
            const int64_t numel = (int64_t)rows * cols;
            const int64_t off   = (int64_t)kDecOffsets[M.tidx];
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
            // MUON: the 2D matrices were orthogonalized + applied in P2.7; P3 handles
            //   ONLY the 1D / non-2D weights, which take the AdamW tail (muon.h:75-76,
            //   the eager Muon auto-split). Skip the 2D ones here (already done).
            if constexpr (Opt == OptId::Muon) {
                if (dectc::dec_is_muon_2d(t)) continue;
            }
            FusedOptState ts = rebase_state<Opt>(st, off);
            // (i) PER-TENSOR LAYER-WISE β1 (GrokAdamW only): β1_i = β1·(1-γ)^t,
            //     t == the tensor's flat named_parameters() layer index (the
            //     work-steal task id maps 1:1 to kDecOffsets order == the eager
            //     enumeration order, so t IS the eager layer index). bc1 must be
            //     rebased TOO (= 1-β1_i^step) or m_hat=m/bc1 mismatches eager
            //     (~9.6× on the deepest layer); bc2 stays global (β2 not layer-wise).
            if constexpr (Opt == OptId::GrokAdamW) {
                const float b1 = st.beta1 * powf(1.0f - st.gamma, (float)t);
                ts.beta1 = b1;
                ts.bc1   = 1.0f - powf(b1, (float)step);
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
                // adamw_step with the aux hyperparameters — IDENTICAL to the vit twin.
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
    // B%16 required (the dW K-loop contracts K=T=B*kSeq and K=B in 16-step atoms,
    // AND it guarantees full token tiles for the projections). NO G-divisibility
    // guard: the split-K dW uses a FLOOR-BALANCED K-partition (dectc_dw_run_tile_
    // splitk) that sums to KS exactly for any KS≥G, so it works at the production
    // truncated B (e.g. 4176, where head KS=B/16=261 is NOT divisible by G=4).
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
