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
#include "csrc/fused/sm_90/dec_weights.cuh"
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
// TC megakernel's Prodigy branch runs an INLINE FIXED-PARTITION (r,s) reduce + an EMA
// d-update owner block between B2 and P3 (see the P2.6 block; the work-steal
// prodigy_precompute_reduce_phaseA/B helpers in opt_stages_precompute.cuh are the
// older form, retained for the dormant mamba path). Included only under the wgmma
// token (the scalar default path has no STAGED tail).
#include "csrc/fused/sm_90/opt_stages_precompute.cuh"
// SuperGrok2 FULL CSA/HCA/PEER/GRU meta-net stages (composed as the optimizer
// phase). Pulls in sg2_meta_stages + the in-kernel segmented sort (STAGE -1) +
// SG2Dims/SG2State/SG2Scalars/SG2Weights. Only under the wgmma token (the scalar
// default path has no SG2 tail). The SG2 weight bundle is read from HBM (it does
// NOT fit smem alongside DecTcSmem under the 48KB cap), so the composed path
// instantiates sg2_meta_stages with WeightsT==SG2Weights + BuildSort=true.
#include "csrc/fused/sm_90/opt_stage_supergrok2.cuh"
#endif

// ============================================================================
//  SG_DEC_SCALAR_MEGAKERNEL — compile-gate for the LEGACY fp32 SCALAR decoder
//  megakernel (fused_decoder_megakernel<Opt> + launch_fused_decoder_megakernel
//  <Opt> below, and the per-sample DecSampleSmem path in model_stages_decoder
//  .cuh). It is the fp32 fallback cell (dispatch.cpp's !want_wgmma branch, adamw-
//  only) and is NEVER on the production bf16 wgmma path that wiring_check's 33/33
//  exercise — the TC engine (fused_decoder_megakernel_tc, below) carries every
//  routed cell.
//
//  WHY A GATE (the d-scaling decouple): DecSampleSmem is sized [kSeq×kD] /
//  [kSeq×kDff] arrays, so its static smem GROWS with SG_DEC_D and ptxas HARD-STOPS
//  ("uses too much shared data") at large d (≥768) — a hard compile failure on the
//  scalar __global__, even though the TC engine's smem (DecTcSmem ~13.5 KB) is
//  d-INDEPENDENT and builds at every d. Gating the dead scalar kernel OFF
//  (-DSG_DEC_SCALAR_MEGAKERNEL=0) lets the WHOLE extension compile at scaled d
//  (the TC path is what's measured). DEFAULT 1 (ON) → the d=128 production build
//  is byte-for-byte unchanged: the scalar fp32 fallback stays wired and 33/33 stay
//  green. The flag changes NOTHING the TC path touches.
#ifndef SG_DEC_SCALAR_MEGAKERNEL
#define SG_DEC_SCALAR_MEGAKERNEL 1
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
    if (t.sharpness)  t.sharpness  += off;   // SuperGrok11/15 (g_sam−g)² (per element)
    // SuperGrok11/15 phi weights (sg_phi_W1/b1/W2) are a per-TENSOR weight SET (same
    // for every element), NOT a per-element slice — so they are NOT rebased (the same
    // pointer is staged to SMEM once per block, exactly like NeuralGrok's psi pointers).
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

#if SG_DEC_SCALAR_MEGAKERNEL
// ── The persistent megakernel (L3-REAL). gridDim.x = #SMs (one CTA/SM), 256
//    threads/CTA. The smem holds ONE DecSampleSmem (≈42 KB, < 48 KB static cap —
//    no dynamic-smem opt-in, so the occupancy≥1 guard with dynamicSMemBytes=0 in
//    the launcher is unchanged). ────────────────────────────────────────────────
//    GATED by SG_DEC_SCALAR_MEGAKERNEL (this legacy fp32 path's DecSampleSmem
//    grows with SG_DEC_D and ptxas-fails at large d; OFF lets the TC path build
//    at scaled d — see the flag note above). ────────────────────────────────────
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
#endif  // SG_DEC_SCALAR_MEGAKERNEL

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
    // GEMM K-loop ring. The M-atom-interleaved wgmma pipeline stages up to
    // kDecAtomsPerSlot A(64×16) tiles per slot (one per stacked m64 atom in an
    // interleave GROUP, issued back-to-back into independent fp32 fragments so the
    // tensor pipe overlaps them) + ONE shared B(N×16) tile per slot. Layout:
    // A-tiles packed [slot][atom-in-group] (slot s, atom ai at
    // sA + (s*kDecAtomsPerSlot + ai)*64*16); B [slot] at sB + s*N*16.
    // kDecAtomsPerSlot = min(kAtomsM, kDecMaxIL) = the widest interleave group any
    // call site uses (fwd/dX = kAtomsM=2; dW = 1). At S=2, =2, N=128 the ring is
    // 2·(2·2KB + 4KB)=16KB; DecTcSmem total ~17.5KB ≪ the 48KB static cap.
    static constexpr int kDecAtomsPerSlot =
        (dectc::kAtomsM < dectc::kDecMaxIL) ? dectc::kAtomsM : dectc::kDecMaxIL;
    // RING DEPTH for the smem allocation = kDecRingStagesMax = max(fwd/dX ring
    // depth, dW ring depth). When SG_TUNED_DEC_FWD_PIPE=0 this == kDecTcStages
    // (the shipped 2), so sA/sB are BYTE-IDENTICAL to every shipped build → the
    // kernel's static smem (and thus the launcher's cudaOccupancyMaxActiveBlocks
    // certification) is unchanged. When PIPE=1 the deeper fwd/dX ring needs
    // kDecFwdStages (2..4) slots, so the ring smem grows ∝ depth — this is the
    // OCCUPANCY COST of the lever: at N=128, IL=2, each extra slot adds
    // (2·64·16 + 128·16)·2 B = 8 KB (sA +4KB, sB +4KB). The launcher REFUSES
    // (cudaErrorLaunchOutOfResources) if the larger static smem drops occupancy
    // below 1 CTA/SM (the persistent grid-barrier requires ≥1) — so the operator
    // backs off the depth knob if a chosen depth won't place. NO register growth
    // (the WgmmaAccum<N> fragments are independent of ring depth), so the ceiling
    // is purely the 228 KB/SM smem budget (≈ depth 4 fits comfortably at d=2048).
    static constexpr int kDecRingStages = dectc::kDecRingStagesMax;
    // alignas(16): the cp.async RING (fwd/dX staging) lands 16-byte LDGSTS
    // chunks here, which require a 16B-aligned smem destination. Every
    // tile-internal offset is a multiple of 8 bf16 (16B) and the sA block size
    // is a multiple of 16B at every legal (stages, IL, TILE_N), so the alignas
    // changes NO member offset — it only pins the guarantee the ring needs.
    alignas(16) __nv_bfloat16 sA[kDecRingStages * kDecAtomsPerSlot * 64 * 16];
    alignas(16) __nv_bfloat16 sB[kDecRingStages * SG_TUNED_TILE_N * 16];
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
// CONTIGUOUS-TRANSPOSE dW scratch floats (SG_TUNED_DEC_DW_STAGE=1 only; 0 on the
// scalar default → the workspace is byte-identical to every shipped build). Holds
// the packed K-major transpose (dYt+Xt) of the 9 weights (bf16 elems → ceil/2
// floats). Carved LAST (after the embed lists) so every prior region's offset is
// unchanged when this is enabled, and entirely absent when it is 0.
__host__ __device__ __forceinline__ int64_t dec_tc_dw_transpose_floats(int T, int B) {
    const int64_t e = dectc::dec_dw_transpose_elems(B, T);   // bf16 elems (0 if stage!=1)
    return (e + 1) / 2;                                      // → floats (round up)
}
// ── STAGED-opt scratch gate (bench-layout-only carve elision) ────────────────
// The four STAGED-optimizer scratch regions below (Prodigy reduce / Muon NS /
// LookSAM 2nd-bwd / SuperGrok2 meta-net) are carved UNCONDITIONALLY on the
// production path so the opt-agnostic launcher fits every OptId in one workspace.
// The d-scaled BENCH layout (SG_DEC_BENCH_LAYOUT=1) is adamw-ONLY (decoder_bench.py
// drives OptId::AdamW exclusively — none of those four optimizers ever runs in the
// bench TU), so carving them there is pure dead weight. At bench width the SG2
// region in particular is pathological: dec_sg2_ws_stride_floats() is O(Nmax·d_model)
// PER CTA and Nmax=4d²=4,194,304 at d=1024 ⇒ ~199 GB over 132 CTAs, OOMing the 80 GB
// H100 (see the KNOWN DEEP LIMIT note at dec_tc_sg2_floats — SG2's per-CTA workspace
// DESIGN doesn't scale; a chunked/streamed restructure is the documented deep item,
// out of scope here per 7656ea6/campaign notes). So we GATE the four regions OFF at
// bench width: honest scoping (the bench never touches them), NOT correctness
// suppression. PRODUCTION is byte-identical — kDecStagedOptScratch is true there, so
// every gated helper folds back to its original value (constexpr bool, -O3 inline).
#if SG_DEC_BENCH_LAYOUT
constexpr bool kDecStagedOptScratch = false;   // bench: adamw-only → elide the 4 staged-opt carves
#else
constexpr bool kDecStagedOptScratch = true;    // production: carve all 4 (opt-agnostic launcher)
#endif
// STAGED-optimizer cross-CTA reduction scratch (Prodigy d-estimate). The Prodigy
// stage publishes per-CTA (r,s) slots (2*nCTA) + a reduced-d slot (1) — an owner-
// computes tree (opt_stages_precompute.cuh), NO float atomic. Sized for the
// LARGEST nCTA (one CTA/SM = #SMs); tiny (≤ 2*132+1 ≈ 1 KB) and carved
// UNCONDITIONALLY so the opt-agnostic host launcher (dec_tc_launcher_scratch)
// allocates one workspace that fits every OptId. Unused by AdamW/Lion/… (their
// P3 never touches this region), so adding it leaves those cells byte-identical.
__host__ __device__ __forceinline__ int64_t dec_tc_opt_reduce_floats(int nCTA) {
    if (!kDecStagedOptScratch) return 0;     // bench (adamw-only): Prodigy never runs
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
    if (!kDecStagedOptScratch) return 0;     // bench (adamw-only): Muon NS never runs
    // X + AX + AAX + orth (each maxNumel) + A (maxRows²) + nrm_partials(nCTA) + inv_norm(1)
    return (int64_t)4 * dectc::kDecMuonMaxNumel
         + (int64_t)dectc::kDecMuonMaxRows * dectc::kDecMuonMaxRows
         + nCTA + 1;
}
// ── LookSAM (STAGED in-kernel SAM 2nd backward) transient scratch. The SAM step
//    perturbs p in place, runs a SECOND fwd+bwd at p', and needs (a) a param BACKUP
//    for the exact restore (eager clones p before perturbing) + (b) a SEPARATE g_sam
//    grad buffer so the 2nd backward does NOT clobber the first grad `grad` (sam_dir
//    = g_sam − grad reads both). Both are total-sized + transient (NOT persistent
//    state — sam_dir itself persists in the `extra` state slice, bound by the cell).
//    Carved UNCONDITIONALLY (≈ 2·422755 ≈ 3.4 MB) so the opt-agnostic launcher fits
//    every OptId; unused by every non-LookSAM cell (its phase is if-constexpr'd out
//    → byte-identical). nrm_partials(nCTA)+inv_norm(1) for the ‖g‖ reduction reuse
//    the loss workspace (sq_part/coef_bc, as GrokAdamW's P2.5 does), so no extra here. ──
__host__ __device__ __forceinline__ int64_t dec_tc_looksam_floats() {
    if (!kDecStagedOptScratch) return 0;      // bench (adamw-only): LookSAM never runs
    return (int64_t)2 * kDecTotalElems;       // [sam_backup | sam_grad]
}
// ── SuperGrok2 compile-time dims = the race config (== SG2Dims defaults). The
//    composed megakernel reads weights from HBM, so the only SG2 workspace is the
//    per-CTA meta-net scratch (sg2_ws_stride, sized for the LARGEST tensor — the
//    ff weights at d=128 — which bounds every tensor's intermediates + the
//    in-kernel segmented-sort key/idx/perm/unsort). ──
using DecSG2Dims = SG2Dims<>;
// max(kDecSizes), re-derived from the layout table (decoder_layout.cuh) so the SAME
// SG2 tail is correctly sized at ANY ladder width — was a d=128-pinned 65536. The
// per-CTA carve upper bound: the largest per-tensor numel.
constexpr int kDecSG2Nmax = kDecMaxTensorNumel;   // == dec_layout_check::max_size()
// ── KNOWN DEEP LIMIT (task #24 Part B): dec_sg2_ws_stride_floats() is
//    O(kDecSG2Nmax · SG2Dims::d_model) ≈ 50·Nmax floats PER CTA, and dec_tc_sg2_floats
//    carves it for nCTA (one CTA/SM, 132 on H100). At the d=1024 bench width Nmax=4d²
//    =4,194,304 ⇒ ~377 M floats/CTA × 132 ≈ 199 GB — over the 80 GB H100 HBM, so the
//    decoder d=1024 bench OOMs on this carve (empirically: "Tried to allocate 191.97
//    GiB"). The d=128 pin (65536) hid this by UNDER-sizing a region adamw never
//    touches; the correct de-derivation EXPOSES it. This is a STRUCTURAL property of
//    the SG2 per-CTA workspace (it materializes the largest tensor's full CSA/HCA/PEER
//    intermediates), NOT a constexpr fix — making the SG2 cell run at d=1024 needs the
//    workspace CHUNKED/streamed over the tensor (a tail restructure, intentionally out
//    of scope per the Part-B directive). The de-pin itself is correct (production d=128
//    unchanged: Nmax==65536; fits at d≤512: ~50 GB). Same limit applies to
//    vit_sg2_ws_stride_floats / mb_sg2_ws_stride_floats (kVitSG2Nmax/kMbSG2Nmax).
// Per-CTA SG2 slice = row_off64 staging (kDecNumTensors int64 = 2*N floats) + the
// meta-net scratch (sized for the largest tensor). The row_off64 prefix lets the
// SG2State.row_off (const int64_t*) be built on-device from the __constant__ int
// kDecOffsets (the adapter for the int-typed layout tables).
__host__ __device__ __forceinline__ int64_t dec_sg2_ws_stride_floats() {
    return (int64_t)2 * kDecNumTensors
         + sg2_ws_stride<DecSG2Dims>((int64_t)kDecSG2Nmax);
}
__host__ __device__ __forceinline__ int64_t dec_tc_sg2_floats(int nCTA) {
    if (!kDecStagedOptScratch) return 0;     // bench (adamw-only): SG2 never runs — and
        // its honestly-derived per-CTA stride (4d²·d_model floats) would OOM at d=1024.
    // +1 for the 8-byte realignment slack of sg2_ws_base (the per-CTA slice fronts an
    // int64 row_off64 array; kDecTotalElems is odd, so the base may need a +1 bump).
    return (int64_t)nCTA * dec_sg2_ws_stride_floats() + 1;
}
// AGGREGATE of the four STAGED-opt scratch regions (Prodigy reduce | Muon NS |
// LookSAM 2nd-bwd | SuperGrok2 meta-net), in the kernel's carve order. This is the
// SINGLE source the host workspace size (dec_tc_workspace_floats) AND the kernel's
// post-staged-opt offset (embed_ws) are derived from, so the host carve and kernel
// offsets stay byte-consistent under the kDecStagedOptScratch gate: at bench width
// all four collapse to 0 (so the embed lists carve right after the dW partials, and
// the kernel's pointer chain — built from the SAME four gated helpers — lands embed_ws
// at the identical place); in production each returns its full size (byte-identical).
__host__ __device__ __forceinline__ int64_t dec_tc_staged_opt_floats(int nCTA) {
    return dec_tc_opt_reduce_floats(nCTA)     // STAGED-opt (Prodigy) reduce slots
         + dec_tc_muon_floats(nCTA)           // STAGED-opt (Muon) NS per-matrix scratch
         + dec_tc_looksam_floats()            // STAGED-opt (LookSAM) SAM 2nd-bwd scratch
         + dec_tc_sg2_floats(nCTA);           // SuperGrok2 meta-net per-CTA scratch
}
__host__ __device__ __forceinline__ int64_t dec_tc_workspace_floats(int T, int B, int nCTA) {
    return dec_tc_acts_floats(T, B)
         + (int64_t)nCTA * dectc::dec_tile_scratch_total_f32()
         + (int64_t)nCTA * dectc::kLnVecElems
         + nCTA + 1
         + dec_tc_dw_part_floats()            // split-K dW partials (G>1)
         + dec_tc_staged_opt_floats(nCTA)     // STAGED-opt scratch (Prodigy|Muon|LookSAM|SG2);
                                              //   gated to 0 at bench width (adamw-only)
         + dectc::dec_wbf_floats()            // bf16 weight pre-stage cache (C1 + C1-T transposed section)
         + dectc::dec_embed_lists_floats(T)   // embedding token lists (row_start+perm)
         + dec_tc_dw_transpose_floats(T, B)   // CONTIGUOUS-TRANSPOSE dW scratch (carve-LAST; 0 unless
                                              //   SG_TUNED_DEC_DW_STAGE=1 → byte-identical when off)
         + 4                                  // realign slack: <=3 floats to 16B-align the wbf cache
                                              //   base (cp.async RING) + 1 for the int32-lists realign
         + (dectc::kDecDwTransposeActive ? 8 : 0); // +<=7 floats to 16B-align the dW-transpose base
                                              //   (ONLY when active → the scalar default's size +
                                              //   every offset is byte-identical to before)
}

#ifdef SG_DEC_PROFILE
// Diagnostic-only (behind SG_DEC_PROFILE; NEVER on the shipped path — the flag is
// set ONLY by the bench/profile TU, default OFF, so the production _ops is byte-
// identical). Per-phase clock64() deltas, max across CTAs (= the critical-path
// duration per phase = the slowest CTA, which is what the host wall sees because
// the trailing grid barrier waits for the last CTA). Slots (decoder AdamW path):
//   [0]=P1 fwd, [1]=P1 bwd, [2]=B1 barrier wait, [3]=P2 dW-GEMM loop (+split-K),
//   [4]=P2 grad-assembly (biases+embed owner-scan+lnvec reduce+loss),
//   [5]=P3 opt tail, [6]=B2 barrier wait (sync_reset P2->P3), [7]=B0 barrier wait.
// clock64() is per-SM; a CTA stays on one SM so its own deltas are valid;
// atomicMax across CTAs gives the slowest CTA per phase. Read host-side via
// cudaMemcpyFromSymbol (see mega_decoder_real_adamw_tc.cu tc_profile_read).
__device__ unsigned long long g_dec_prof_max[8];
#endif

template <OptId Opt>
__global__ void __launch_bounds__(SG_TC_MEGA_BLOCK)
fused_decoder_megakernel_tc(PersistentContext ctx,
                            float* __restrict__ params,
                            DecoderTokenCtx tok,
                            float* __restrict__ grad,
                            float lr, int step, FusedOptState st) {
    __shared__ DecTcSmem sm;
    GridBarrier bar = ctx.barrier();
    // SAM-coupled cells run the tile fwd+bwd TWICE (P1 + P2.4); route BOTH their
    // passes through the out-of-line shims (one shared frame, campaign C2). The
    // single-pass cells keep the inline bodies -- byte-identical allocation.
    constexpr bool kSamCoupled = (Opt == OptId::LookSAM || Opt == OptId::SuperGrok11 ||
                                  Opt == OptId::SuperGrok15 || Opt == OptId::SuperGrok2);
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
    // LookSAM SAM 2nd-backward scratch, carved AFTER the Muon scratch (term order
    // matches dec_tc_workspace_floats — carving it LAST keeps every prior region's
    // offset unchanged, so the already-green cells are byte-identical). Layout:
    //   [sam_backup (total)] [sam_grad (total)]. Unused unless Opt==LookSAM.
    float* sam_backup = muon_base + dec_tc_muon_floats(nCTA);
    float* sam_grad   = sam_backup + kDecTotalElems;
    // SuperGrok2 meta-net per-CTA scratch, carved AFTER the LookSAM sam_grad (term
    // order matches dec_tc_workspace_floats — carving it LAST keeps every prior
    // region's offset unchanged, so the already-green cells are byte-identical).
    // Each CTA owns dec_sg2_ws_stride_floats() (row_off64 staging + meta-net scratch).
    // ALIGN to 8 bytes (even float offset): the per-CTA slice fronts an int64
    // row_off64 staging array; kDecTotalElems is odd so sam_grad+total lands on an
    // odd float offset (4-byte) → an int64 read there is a misaligned address. Round
    // up to the next even float. The carve term dec_tc_sg2_floats reserves the stride
    // per CTA; the +1 alignment slack fits within the acts/scratch headroom (the
    // workspace is over-sized) and dec_sg2_ws_stride_floats() is even, so every CTA
    // base stays 8-byte aligned. Unused unless Opt==SuperGrok2.
    float* sg2_ws_base = sam_grad + kDecTotalElems;
    if (((uintptr_t)sg2_ws_base & 0x7) != 0) sg2_ws_base += 1;   // → 8-byte aligned
    (void)sg2_ws_base;   // referenced only by the SuperGrok2 P3-SG2 phase
    // Embedding token-list scratch (counting-sort row_start[V+1] + perm[T]), carved
    // AFTER the SG2 region (term order matches dec_tc_workspace_floats — carving it
    // LAST keeps every prior region's offset unchanged → the green cells' non-embed
    // regions are byte-identical). int32 views over the float tail; 8-byte aligned so
    // the (future) widening to int is safe. Built in the P1 window (cta 0), consumed
    // in P2 — fenced by the already-present B1 barrier (no new barrier).
#if SG_DEC_BENCH_LAYOUT
    // Bench (adamw-only): the four staged-opt regions are gated to 0, so embed_ws is
    // the staged-opt block START (== opt_reduce) plus the aggregate (== 0 here). We
    // derive it from the SAME dec_tc_staged_opt_floats aggregate the host carve
    // (dec_tc_workspace_floats) uses, so the host size and this offset are provably the
    // identical expression at bench width — no reliance on the (collapsed) sg2_ws_base
    // chain. (opt_reduce == dw_part + dec_tc_dw_part_floats(); see its carve above.)
    float* wbf_f = opt_reduce + dec_tc_staged_opt_floats(nCTA);
    // 16B-align the bf16 cache base: the cp.async RING streams it in 16-byte
    // chunks (gmem source alignment requirement). Bump <= 3 floats, covered by
    // dec_tc_workspace_floats' slack term; deterministic (the workspace base is
    // >=256B-aligned, so the bump depends only on the carve offsets).
    while (((uintptr_t)wbf_f & 0xF) != 0) wbf_f += 1;
    float* embed_ws = wbf_f + dectc::dec_wbf_floats();
#else
    // Production: carve AFTER the (full-size) SG2 region via the existing chain
    // (the sg2_ws_base align bump + the dec_tc_sg2_floats stride are load-bearing
    // for the SG2 int64 row_off64 reads). The bf16 weight pre-stage cache (C1)
    // is interposed here; embed lists stay carve-LAST.
    float* wbf_f = sg2_ws_base + dec_tc_sg2_floats(nCTA);
    // 16B-align the bf16 cache base for the cp.async RING (see the bench-branch
    // note; bump <= 3 floats, covered by dec_tc_workspace_floats' slack term).
    while (((uintptr_t)wbf_f & 0xF) != 0) wbf_f += 1;
    float* embed_ws = wbf_f + dectc::dec_wbf_floats();
#endif
    if (((uintptr_t)embed_ws & 0x7) != 0) embed_ws += 1;         // → 8-byte aligned
    int* embed_row_start = reinterpret_cast<int*>(embed_ws);     // [V+1]
    int* embed_perm      = embed_row_start + (dec::kVocab + 1);  // [T]
    // CONTIGUOUS-TRANSPOSE dW scratch (SG_TUNED_DEC_DW_STAGE=1), carved AFTER the
    // embed lists (term order matches dec_tc_workspace_floats — carve-LAST keeps
    // every prior region's offset unchanged). #if-guarded (not a runtime ternary)
    // so the scalar default emits ZERO extra carve code → the kernel's register
    // allocation + PTX are BYTE-IDENTICAL to every shipped build (PTX-verified).
    // 16B-align the base: the cp.async ring streams it in 16-byte chunks (gmem
    // source alignment). Each transposed ROW is K-contiguous and K is a multiple
    // of kWgmmaAtomK=16 (⇒ 8 bf16 = 16B), so once the base is 16B-aligned every
    // chunk16 half-offset (mn·8 / N·8 / kbase) is too. Bump ≤ 7 floats (covered
    // by dec_tc_workspace_floats' +8 active-only slack term).
#if SG_TUNED_DEC_DW_STAGE
    float* dwt_f = reinterpret_cast<float*>(embed_perm + (int64_t)T);
    while (((uintptr_t)dwt_f & 0xF) != 0) dwt_f += 1;
    __nv_bfloat16* dwt_base = dectc::kDecDwTransposeActive
        ? reinterpret_cast<__nv_bfloat16*>(dwt_f) : nullptr;
#endif

    dectc::DecActs acts = dectc::dec_acts_bind(acts_base, T, B);
    dectc::DecTileScratch sc = dectc::dec_tile_scratch_bind(scratch_base + (int64_t)cta * scratch_per);
    float* my_lnvec = lnvec_base + (int64_t)cta * dectc::kLnVecElems;

    DecWeights w = dec_bind(params);
    // bf16 weight pre-stage cache (C1): bound over its workspace carve; FILLED in
    // P0 below (fenced by B0 before any P1 GEMM stages from it).
    __nv_bfloat16* wbf_cache = reinterpret_cast<__nv_bfloat16*>(wbf_f);
    dectc::DecWBf wb = dectc::dec_wbf_bind(wbf_cache);

    // ── P0: zero this CTA's LN-vec partials + loss slot (dW/embed grads are
    //    written-once → no pre-zero). ──
    for (int i = threadIdx.x; i < dectc::kLnVecElems; i += blockDim.x) my_lnvec[i] = 0.0f;
    if (threadIdx.x == 0) loss_part[cta] = 0.0f;
    // C1: fill the bf16 weight cache (grid-strided, element-owned, deterministic);
    // B0 fences it before any P1 GEMM stages from it.
    dectc::dectc_wbf_convert(params, wbf_cache, cta, nCTA);
#ifdef SG_DEC_PROFILE
    unsigned long long _b0a = (threadIdx.x == 0) ? clock64() : 0;
#endif
    bar.sync();   // B0
#ifdef SG_DEC_PROFILE
    if (threadIdx.x == 0) { unsigned long long _b0b = clock64(); atomicMax(&g_dec_prof_max[7], _b0b - _b0a); }
#endif

    // ── Build the embedding token lists (counting sort over tok_ids) ONCE on cta 0,
    //    HERE in the P1 window so it overlaps the other CTAs' fwd/bwd and the
    //    already-present B1 barrier fences it before the P2 consume — NO new barrier.
    //    cta 0 then joins P1. The build is O(T), parallel over its worker lanes, so
    //    it measures ~0.3 ms (≪ P1) — far below the embed-consume time it unlocks. ──
    if (cta == 0)
        dectc::dectc_embed_build_lists(tok.tokens, T, embed_row_start, embed_perm);

    // ── P1: token-tile-parallel fwd+bwd. Each CTA grid-strides over tiles of
    //    kTileM rows; for its tile it runs fwd (→ acts X, NLL) then bwd (→ acts
    //    dY, dh0, LN-vec partials). Barrier-free within the tile. ──
    const int nrows_tile = dectc::kTileM;
    const int n_tiles = (T + nrows_tile - 1) / nrows_tile;
    float nll_acc = 0.0f;
#ifdef SG_DEC_PROFILE
    unsigned long long prof_fwd = 0, prof_bwd = 0;
#endif
    for (int ti = cta; ti < n_tiles; ti += nCTA) {
        const int g0 = ti * nrows_tile;
        const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
#ifdef SG_DEC_PROFILE
        __syncthreads(); unsigned long long _c0 = clock64();
#endif
        float nll;
        if constexpr (kSamCoupled)
            nll = dectc::dectc_forward_tile_outlined(w, wb, g0, nrows, acts, sc, tok.tokens,
                                                     tok.targets, sm.sA, sm.sB, sm.red);
        else
            nll = dectc::dectc_forward_tile(w, wb, g0, nrows, acts, sc, tok.tokens, tok.targets,
                                            sm.sA, sm.sB, sm.red);
#ifdef SG_DEC_PROFILE
        __syncthreads(); unsigned long long _c1 = clock64();
#endif
        if constexpr (kSamCoupled)
            dectc::dectc_backward_tile_outlined(w, wb, g0, nrows, B, acts, sc, tok.targets,
                                                my_lnvec, sc.work2, sm.sA, sm.sB, sm.red);
        else
            dectc::dectc_backward_tile(w, wb, g0, nrows, B, acts, sc, tok.targets,
                                       my_lnvec, sc.work2, sm.sA, sm.sB, sm.red);
#ifdef SG_DEC_PROFILE
        __syncthreads(); unsigned long long _c2 = clock64();
        if (threadIdx.x == 0) { prof_fwd += _c1 - _c0; prof_bwd += _c2 - _c1; }
#endif
        if (threadIdx.x == 0) nll_acc += nll;
        __syncthreads();
    }
    if (threadIdx.x == 0) loss_part[cta] = nll_acc;
#ifdef SG_DEC_PROFILE
    if (threadIdx.x == 0) {
        atomicMax(&g_dec_prof_max[0], prof_fwd);
        atomicMax(&g_dec_prof_max[1], prof_bwd);
    }
    unsigned long long _b1a = (threadIdx.x == 0) ? clock64() : 0;
#endif
    bar.sync();   // B1: all acts (X + dY) + LN-vec partials complete
#ifdef SG_DEC_PROFILE
    if (threadIdx.x == 0) { unsigned long long _b1b = clock64(); atomicMax(&g_dec_prof_max[2], _b1b - _b1a); }
#endif

    // ── P2: assemble all 30 grads into `grad`. dW output-stationary (gt %
    //    nCTA), biases, embedding owner-scan, LN-vec reduce. No partials. The 9
    //    dW specs are built into SHARED smem (thread 0; identical for all) so the
    //    9-spec array is NOT on every thread's stack (shrinks the launch's local
    //    reservation — the persistent kernel must place on a memory-tight GPU). ──
    // Spec build: the 5-arg form (with dwt_base) binds the transpose scratch when
    // SG_TUNED_DEC_DW_STAGE=1; the scalar default uses the original 4-arg overload
    // VERBATIM (no extra arg in the PTX) → byte-identical.
    if (threadIdx.x == 0) {
#if SG_TUNED_DEC_DW_STAGE
        dectc::dectc_build_dw_specs(acts, B, T, sm.spec, dwt_base);
#else
        dectc::dectc_build_dw_specs(acts, B, T, sm.spec);
#endif
    }
    __syncthreads();
    dectc::DecDwSpec* spec = sm.spec;
    const int n_dw = dectc::dectc_dw_total_tiles<SG_TUNED_TILE_N>(spec);
    // CONTIGUOUS-TRANSPOSE staging (SG_TUNED_DEC_DW_STAGE=1): pre-transpose the 9
    // weights' dY/X into the K-major scratch the specs bind, so the SINGLE-CTA dW
    // GEMM (dectc_dw_run_tile, the proof wrapper) reads them via the proven
    // cp.async ring (DecGmemTileSrc*). Runs AFTER B1 (all acts complete) + ONE
    // grid barrier so a CTA's dW tile reads dYt/Xt written by ANY CTA (the
    // transpose is grid-strided, owner ≠ the dW tile owner). GATED to kDwG==1
    // (the wired proof path): when split-K is active the _splitk path keeps the
    // lambda gather, so the transpose would be wasted work — skip it (and the
    // barrier). #if-guarded so the scalar default emits NEITHER the transpose call
    // NOR the extra barrier (byte-identical control flow + grid-barrier count). ──
#if SG_TUNED_DEC_DW_STAGE
    if constexpr (dectc::kDecDwTransposeActive) {
        dectc::dectc_dw_transpose_operands(spec, cta, nCTA);
        bar.sync();   // B1b: all transposed operands visible before any dW GEMM read
    }
#endif
#ifdef SG_DEC_PROFILE
    __syncthreads(); unsigned long long _dwa = (threadIdx.x == 0) ? clock64() : 0;
#endif
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
#ifdef SG_DEC_PROFILE
    __syncthreads();
    if (threadIdx.x == 0) { unsigned long long _dwb = clock64(); atomicMax(&g_dec_prof_max[3], _dwb - _dwa); }
    unsigned long long _gaa = (threadIdx.x == 0) ? clock64() : 0;
#endif
    dectc::dectc_dw_biases(spec, grad, cta, nCTA);
    dectc::dectc_embed_owner_scan(acts, embed_row_start, embed_perm, T, grad, cta, nCTA);
    dectc::dectc_lnvec_reduce(lnvec_base, grad, nCTA, cta);
    // Loss reduce (fp64) by CTA 0.
    if (cta == 0 && threadIdx.x == 0) {
        double s = 0.0;
        for (int c = 0; c < nCTA; ++c) s += (double)loss_part[c];
        *tok.loss_out = (float)(s / (double)B);
    }
#ifdef SG_DEC_PROFILE
    if (threadIdx.x == 0) { unsigned long long _gab = clock64(); atomicMax(&g_dec_prof_max[4], _gab - _gaa); }
#endif
    (void)loss_out;
#ifdef SG_DEC_PROFILE
    unsigned long long _b2a = (threadIdx.x == 0) ? clock64() : 0;
#endif
    bar.sync_reset(ctx.g_next_task);   // B2: reduced grad ready; reset queue
#ifdef SG_DEC_PROFILE
    if (threadIdx.x == 0) { unsigned long long _b2b = clock64(); atomicMax(&g_dec_prof_max[6], _b2b - _b2a); }
#endif

    // ── P2.4 (LookSAM ONLY, SAM steps): the MODEL-COUPLED SAM 2nd backward that
    //    produces st.sam_dir = g_sam − g (INTEGRATION-OPTSTAGES §6). Eager LookSAM,
    //    every k steps (looksam.py sam_step + looksam.h:27-59):
    //      ‖g‖     = GLOBAL L2 norm over ALL reduced grads (compute_sam_grad_norm_
    //                device_side); rho_over_norm = rho/‖g‖
    //      p'      = p + rho_over_norm·g          (perturb in the grad direction;
    //                backup = p.clone() for the exact restore — fp32 add is not
    //                bit-reversible, so we MUST save, not subtract back)
    //      g_sam   = ∇L(p')                       (a FULL SECOND in-kernel fwd+bwd
    //                + deterministic grad assembly at the perturbed weights, written
    //                to the SEPARATE `sam_grad` buffer so `grad` (the first grad the
    //                apply tail blends from) is untouched)
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
    //    deterministic BY CONSTRUCTION. Guarded so every other opt's path is
    //    byte-identical (no extra barrier / work). On a NON-SAM step (looksam_sam==0)
    //    this whole block is skipped — st.sam_dir already holds the cached direction.
    //
    //    SuperGrok11/15 SHARE this exact SAM 2nd-backward machinery (INTEGRATION-
    //    OPTSTAGES §4/§5): the perturb→2nd in-kernel fwd+bwd→restore pipeline is
    //    IDENTICAL; only the elementwise WRITE in step (d) differs — LookSAM writes
    //    st.sam_dir[i] = g_sam − g, SG11/15 write st.sharpness[i] = (g_sam − g)²
    //    (supergrok11_sm90.cuh:246 / supergrok15_sm90.cuh:315). sharpness is then the
    //    2nd MLP input the P2.45 meta-net mu precompute reads. The SAM-step gate is the
    //    SAME st.looksam_sam host scalar (the SG sam_step cadence); on a non-SAM step
    //    the cached sharpness is reused verbatim. SG reuses st.rho as the perturbation
    //    radius (its own sam_rho), identical reduction/perturb math.
    if constexpr (Opt == OptId::LookSAM || Opt == OptId::SuperGrok11 ||
                  Opt == OptId::SuperGrok15 || Opt == OptId::SuperGrok2) {
        if (st.looksam_sam != 0.0f) {
            // (a) GLOBAL ‖g‖ over the reduced grad, deterministic (reuse the loss
            //     workspace as P2.5 does: loss_part[nCTA] = per-CTA Σg², loss_out[1]
            //     broadcasts rho_over_norm). The reduced loss is already in
            //     *tok.loss_out (host state), so this is free scratch.
            float* sq_part = loss_part;          // [nCTA] per-CTA Σ g²  (ascending reduce)
            float* ron_bc  = loss_out;           // [1] broadcast rho_over_norm
            const int64_t total = kDecTotalElems;
            {
                const int64_t base = total / nCTA, rem = total % nCTA;
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
                     i < total; i += gstride) {
                    sam_backup[i] = params[i];
                    params[i] = params[i] + rho_over_norm * grad[i];
                }
            }
            bar.sync();   // B2.4c: all params perturbed before the 2nd forward reads them
            // C1: re-fill the bf16 weight cache from the PERTURBED params (the 2nd
            // fwd/bwd GEMMs must consume bf16(p'), exactly what the on-read path saw).
            // Element-owned + barrier-fenced -> deterministic. (After the restore in
            // (d) the cache is stale; nothing reads it again this step -- the next
            // step's P0 re-converts from the restored params.)
            dectc::dectc_wbf_convert(params, wbf_cache, cta, nCTA);
            bar.sync();   // B2.4c2: bf16 weight cache (p') complete before the 2nd forward
            // (c) SECOND fwd+bwd at the perturbed weights → g_sam in `sam_grad`. Mirror
            //     P1 + P2 EXACTLY, but: re-zero the lnvec partials (they accumulate),
            //     do NOT write the loss (keep the first-pass loss), and assemble into
            //     `sam_grad` (NOT `grad`). `w` wraps `params` (now perturbed) so the
            //     tile fns read the perturbed weights with no re-bind needed.
            for (int i = threadIdx.x; i < dectc::kLnVecElems; i += blockDim.x) my_lnvec[i] = 0.0f;
            bar.sync();   // B2.4d: lnvec partials cleared before the 2nd backward accumulates
            for (int ti = cta; ti < n_tiles; ti += nCTA) {
                const int g0 = ti * nrows_tile;
                const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
                dectc::dectc_forward_tile_outlined(w, wb, g0, nrows, acts, sc, tok.tokens,
                                                   tok.targets, sm.sA, sm.sB, sm.red);
                dectc::dectc_backward_tile_outlined(w, wb, g0, nrows, B, acts, sc, tok.targets,
                                                    my_lnvec, sc.work2, sm.sA, sm.sB, sm.red);
                __syncthreads();
            }
            bar.sync();   // B2.4e: all 2nd-pass acts (X + dY) + LN-vec partials complete
            // Re-build the dW specs from the 2nd-pass acts (dY adjoints changed).
            if (threadIdx.x == 0) dectc::dectc_build_dw_specs(acts, B, T, sm.spec);
            __syncthreads();
            dectc::DecDwSpec* spec2 = sm.spec;
            const int n_dw2 = dectc::dectc_dw_total_tiles<SG_TUNED_TILE_N>(spec2);
            if (kDwG > 1) {
                for (int item = cta; item < n_dw2 * kDwG; item += nCTA) {
                    const int gt = item / kDwG, kc = item % kDwG;
                    dectc::dectc_dw_run_tile_splitk<SG_TUNED_TILE_N>(spec2, gt, kc, kDwG, dw_part, sm.sA, sm.sB);
                }
                bar.sync();
                dectc::dectc_dw_reduce_splitk<SG_TUNED_TILE_N>(spec2, n_dw2, kDwG, dw_part, sam_grad, cta, nCTA);
            } else {
                for (int gt = cta; gt < n_dw2; gt += nCTA)
                    dectc::dectc_dw_run_tile<SG_TUNED_TILE_N>(spec2, gt, sam_grad, sm.sA, sm.sB);
            }
            dectc::dectc_dw_biases(spec2, sam_grad, cta, nCTA);
            // Reuse the P1-built token lists: the SAM 2nd pass perturbs WEIGHTS, not
            // tok_ids, so the token→row mapping (row_start/perm) is unchanged.
            dectc::dectc_embed_owner_scan(acts, embed_row_start, embed_perm, T, sam_grad, cta, nCTA);
            dectc::dectc_lnvec_reduce(lnvec_base, sam_grad, nCTA, cta);
            bar.sync();   // B2.4f: g_sam fully assembled in sam_grad
            // (d) WRITE the SAM side-channel (into the persistent state slice) + restore p.
            //     LookSAM → sam_dir = g_sam − g; SuperGrok11/15/SuperGrok2 → sharpness =
            //     (g_sam − g)² (SG2 shares the SAM 2nd-backward machinery: its meta-net's
            //     2nd MLP input is the SAME sharpness signal, read by sg2_meta_stages).
            {
                const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
                for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
                     i < total; i += gstride) {
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
            bar.sync_reset(ctx.g_next_task);   // B2.4g: params restored + side-channel ready; reset queue for P3
        }
    }

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
    //    EXTENDED to NeuralGrok: eager neuralgrok applies the SAME global grad-norm
    //    clip (clip_grad_norms_device_side, grad_clip=1.0) before psi+amp — reuse
    //    this exact performant+deterministic machinery (apply_optimizer<NeuralGrok>
    //    consumes st.clip_coef). Every non-{GrokAdamW,NeuralGrok} opt stays byte-identical.
    if constexpr (Opt == OptId::GrokAdamW || Opt == OptId::NeuralGrok) {
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
    //
    //    DETERMINISM — FIXED-PARTITION reduction (mirrors the A/A/A-CLEAN GrokAdamW
    //    P2.5 grad-norm reduce above), NOT the work-steal task-queue drain. WHY THIS
    //    SHAPE: the previous form drained the global TaskQueue (q.next_block →
    //    atomicAdd(g_next_task)) inside phaseA and reset that SAME counter in a
    //    sync_RESET barrier (B2.6a). The work-steal claim order is timing-dependent,
    //    so each CTA's (r,s) slot holds a potentially DIFFERENT tensor subset run-to-
    //    run; because fp32 add is non-associative, the per-CTA partials then regroup
    //    the same values differently and the owner's d can differ at ULP level from
    //    step>=2 (a COMPONENT_CONTRACT deterministic-reduction violation — even where
    //    it happens to be empirically bit-exact on a given GPU/workload, it is not
    //    deterministic BY CONSTRUCTION). GrokAdamW's P2.5 reduces a FIXED contiguous
    //    element range per CTA with a plain sync() and is A/A/A-CLEAN; prodigy now
    //    uses the IDENTICAL shape — deterministic by construction, independent of CTA
    //    timing. Each CTA owns [e0,e0+ecnt) of the FLAT [0,total) arrays —
    //    params/param_init/grad are parallel flat blobs, so a contiguous flat range
    //    sum == the cross-all-TENSORS sum (tensor boundaries are irrelevant to a
    //    global Σ). The per-element contribution still CALLS the canonical
    //    prodigy_partials_step (prodigy.h:28-53, single-source). NO float atomic;
    //    per-CTA (r,s) slot publish → ONE plain grid barrier → CTA0 owner-sums
    //    ascending → EMA-decay/update_d → broadcast → ONE plain grid barrier.
    //    g_next_task is UNTOUCHED here (B2 already reset it to 0), so P3's queue
    //    drain runs unchanged — no sync_reset needed in P2.6.
    //    Guarded so every other opt's P3 is byte-identical (no extra barrier/work).
    if constexpr (Opt == OptId::Prodigy) {
        // Per-CTA (r,s) slots in opt_reduce ([r slots | s slots]) + the reduced-d
        // broadcast slot. d_prev: persisted d_lr (slot 2), or d0 cold-start at step 1.
        float* r_part = opt_reduce;                  // [nCTA] per-CTA Σ d²·<g,p0−p>
        float* s_part = opt_reduce + nCTA;           // [nCTA] per-CTA Σ d²·|g|
        float* d_bc   = opt_reduce + 2 * nCTA;       // [1] reduced-d broadcast
        const float d_prev = (step == 1) ? st.d0 : st.prodigy_persist[2];
        // FIXED contiguous element range over the flat [0,total_p) arrays (same split
        // as GrokAdamW P2.5: even base + remainder to the low CTAs → deterministic).
        const int64_t total_p = kDecTotalElems;
        const int64_t pbase = total_p / nCTA, prem = total_p % nCTA;
        const int64_t pe0 = (int64_t)cta * pbase + (cta < prem ? cta : prem);
        const int64_t pecnt = pbase + (cta < prem ? 1 : 0);
        float r_acc = 0.0f, s_acc = 0.0f;
        for (int64_t i = threadIdx.x; i < pecnt; i += blockDim.x) {
            // Canonical per-element partials (prodigy.h:28-53): r carries d²·<g,p0−p>,
            // s carries d²·|g| (the L1-norm denominator). params/param_init/grad are
            // flat blobs → index pe0+i is the same element in all three.
            algo::prodigy_partials_step(params, st.param_init, grad,
                                        d_prev, pe0 + i, r_acc, s_acc);
        }
        // Deterministic in-CTA two-value reduction (fixed thread count) → thread0
        // publishes this CTA's (r,s) slot. No atomic.
        float r_block, s_block;
        prim::block_reduce_sum2_f32(r_acc, s_acc, r_block, s_block);
        if (threadIdx.x == 0) { r_part[cta] = r_block; s_part[cta] = s_block; }
        bar.sync();   // B2.6a: all per-CTA (r,s) partials published & visible
        // Owner block (CTA0 thread0): ascending owner-sum + EMA decay + accumulate +
        // d_coef + update_d, byte-matching launch_multi_tensor_prodigy_fused_reduce_step.
        if (cta == 0 && threadIdx.x == 0) {
            float r_step = 0.0f, s_step = 0.0f;     // ascending-CTA owner-sum
            for (int c = 0; c < nCTA; ++c) { r_step += r_part[c]; s_step += s_part[c]; }
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
            d_bc[0]               = d_new;          // broadcast to all CTAs
        }
        bar.sync();   // B2.6b: d visible to every CTA before the apply
        st.d_factor = d_bc[0];                      // the reduced d the tail reads
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

    // ── P2.45 (SuperGrok11/15 ONLY) staged into P3: the per-TENSOR meta-net mu
    //    precompute (INTEGRATION-OPTSTAGES §4/§5). SINGLE task-queue drain, NO grid
    //    barrier: each tensor T is SELF-CONTAINED — mu(T) = sg_rescale·phi(g[T],
    //    sharpness[T]) (per element) and, for SG11, the per-tensor gate(T) =
    //    sigmoid(gate_temp·cos(g[T], mu[T])) depend ONLY on T's own elements (never on any
    //    other tensor's mu), so mu(+gate)→apply fuse into ONE body the SAME CTA owns
    //    end-to-end inside the EXISTING P3 drain (every thread of a CTA claims the same
    //    tensor t via the block-level task_slot, so the helper's __syncthreads/block-
    //    reduce are block-uniform and safe). NOT a separate full-queue mu pre-pass +
    //    a second apply drain (that would silently no-op — see the SG11 block in
    //    opt_stages_precompute.cuh). The phi weights (per-TENSOR weight set) are staged
    //    to SMEM ONCE per block before the drain. SG15's gate is the host scalar st.gate
    //    (no cosine stage); SG11 binds the per-tensor gate the helper returns. sharpness
    //    (the 2nd MLP input) was produced by P2.4 (or the cached prior step).
    __shared__ float sg_sW1[kSgPhiHidden * 2];
    __shared__ float sg_sb1[kSgPhiHidden];
    __shared__ float sg_sW2[kSgPhiHidden];
    if constexpr (Opt == OptId::SuperGrok11 || Opt == OptId::SuperGrok15) {
        sg_stage_phi_weights<kSgPhiHidden>(st.sg_phi_W1, st.sg_phi_b1, st.sg_phi_W2,
                                           sg_sW1, sg_sb1, sg_sW2);   // syncs
    }

    // ── P3-SG2 (SuperGrok2 ONLY): the FULL CSA/HCA/PEER/GRU meta-net as the
    //    optimizer phase, run INSTEAD of the per-element apply_optimizer<SuperGrok2>
    //    (which is only the Adam-on-smart_grad stub). Each CTA work-steals WHOLE
    //    tensors from the queue (reset at B2 / B2.4g) and runs sg2_meta_stages for
    //    each tensor end-to-end: STAGE -1 in-kernel segmented sort (|grad| ascending,
    //    index tie-break — strategy A) → S0..S5 (input-proj, CSA, HCA, GRU, PEER,
    //    apply). The SAM-written st.sharpness (P2.4, (g_sam−g)²) is the meta-net's 2nd
    //    MLP input (sharp_base). The meta-net WEIGHTS come from HBM (the 35 KB bundle
    //    does NOT fit smem alongside DecTcSmem under the 48 KB cap) — sg2_meta_stages
    //    is instantiated with WeightsT==SG2Weights + BuildSort=true. The per-tensor
    //    intermediates + the segmented-sort scratch live in this CTA's slice of the
    //    SG2 workspace (carve-LAST, after the LookSAM sam_grad). This whole phase is
    //    if-constexpr'd to SuperGrok2, so every other cell is byte-identical (no extra
    //    barrier/work). The kernel returns after this — no trailing grid barrier (each
    //    tensor is CTA-local; the standalone megakernel's trailing bar.sync was only to
    //    let a composing stage reset the queue, which there is none here).
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
        // CTA's SG2 workspace slice (kDecOffsets is __constant__ int; SG2State.row_off
        // wants const int64_t*). All CTAs stage their own copy (cheap, kDecNumTensors).
        float* sg2_base = sg2_ws_base + (int64_t)blockIdx.x * dec_sg2_ws_stride_floats();
        int64_t* row_off64 = reinterpret_cast<int64_t*>(sg2_base);
        for (int t = threadIdx.x; t < kDecNumTensors; t += blockDim.x)
            row_off64[t] = (int64_t)kDecOffsets[t];
        // The meta-net scratch starts AFTER the row_off64 staging block (rounded to
        // keep the carve's float alignment): reserve kDecNumTensors int64 = 2*N floats.
        float* sg2_meta_ws = sg2_base + 2 * kDecNumTensors;
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
        stt.ws_stride   = sg2_ws_stride<DecSG2Dims>((int64_t)kDecSG2Nmax);
        stt.n_tensors   = kDecNumTensors;
        stt.n           = kDecSizes;        // __constant__ int[]
        stt.row_off     = row_off64;
        __shared__ int task_slot2;
        TaskQueue q2 = ctx.queue();
        for (int t = q2.next_block(&task_slot2); t < kDecNumTensors;
             t = q2.next_block(&task_slot2)) {
            sg2_meta_stages<DecSG2Dims, SG2Weights, float, float, /*BuildSort=*/true>(
                w2, t, stt, sc2, params, grad, st.sharpness, sg2_meta_ws);
        }
        return;   // SG2 owns the whole optimizer phase; skip P3.
    }

    // ── P3: the REAL optimizer tail over the reduced grad (work-steal the 30
    //    tensors). apply_optimizer<Opt> is the canonical csrc/algorithms math. ──
    st.lr = lr;
#ifdef SG_DEC_PROFILE
    unsigned long long _p3a = (threadIdx.x == 0) ? clock64() : 0;
#endif
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
            // SuperGrok11/15 meta-net mu (+ SG11 per-tensor gate) precompute for THIS
            // tensor, BEFORE the apply reads ts.mu/ts.gate. mu(T)=sg_rescale·phi(g,
            // sharpness) over [off,off+n); SG11 also computes the gate(T)=sigmoid(gate_temp·cos)
            // (block-uniform __syncthreads/reduce — the whole CTA owns t). SG15's gate is
            // st.gate (host sigmoid(accuracy)). Helpers index grad/sharpness/mu by off+i,
            // so pass the BASE pointers (st.mu, grad, st.sharpness) + off; the apply then
            // reads ts.mu (== st.mu+off). phi_b2 read on-device from sg_phi_W2[H].
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
#ifdef SG_DEC_PROFILE
    if (threadIdx.x == 0) { unsigned long long _p3b = clock64(); atomicMax(&g_dec_prof_max[5], _p3b - _p3a); }
#endif
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
