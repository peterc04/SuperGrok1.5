#ifndef SG_FUSED_SM90_MODEL_STAGE_DECODER_TC_CUH_
#define SG_FUSED_SM90_MODEL_STAGE_DECODER_TC_CUH_
// ============================================================================
// csrc/fused/sm_90/model_stage_decoder_tc.cuh — R2 TENSOR-CORE variant of the
// L3-REAL transformer-decoder fwd+bwd. This is the batch-tiled bf16 wgmma path
// (DESIGN-TC-PIPELINE.md Fork B), a TUNED VARIANT compiled ALONGSIDE the scalar
// model_stages_decoder.cuh and selected per-cell by SG_TUNED_GEMM_IMPL (the
// owner directive: BOTH paths compiled, the tuner picks). The scalar path's
// math + gates are UNCHANGED; nothing here edits model_stages_decoder.cuh.
//
// WHY A NEW BODY (not an in-place edit): the wgmma atom is m64nNk16 — it needs
// >= 64 M-rows per issue, but one decoder sample is kSeq=4 rows. So the scalar
// "one CTA owns a batch slice, one sample at a time" model CANNOT use tensor
// cores. The TC path batches across samples: M = (sample x position) TOKEN rows.
// Each CTA owns a contiguous tile of SG_TUNED_TILE_M token rows (default 128 =
// 32 samples x 4 positions; the tile boundary lands on a sample boundary so each
// sample's 4x4 attention stays fully within one tile). This is a genuine
// rewrite, which is why DESIGN hands it a separate header.
//
// FORK B (DESIGN §2/§3, dW-output-stationary — the Q2 deliverable):
//   * P1 token-tile-parallel fwd + bwd-dX through ALL layers, barrier-free
//     within the tile (no per-layer grid barrier — DESIGN explicitly rejects
//     that). The per-token activations the cross-tile dW owners need (the linear
//     INPUTS X and OUTPUT adjoints dY, plus the embedding-input adjoint dh0) are
//     written to an HBM bf16 acts buffer carved from the SAME workspace the
//     scalar path used for its 223 MB grad partials (which Fork B eliminates).
//   * P2 dW-output-stationary: each weight-matrix dW tile is owned by ONE CTA
//     (tile_id % nCTA) which contracts the FULL token dimension T itself
//     (K_g=T, ascending-t, no float atomics → deterministic), streaming dY and X
//     from HBM. No [nCTA x total] partial, no cross-CTA dW reduce.
//
// GRAD OWNERSHIP (all 30 tensors — DESIGN §3.1/§3.4; every grad is a Σ-over-T):
//   * 9 weight MATRICES  → output-stationary dW GEMM (wgmma, K=T)            [P2]
//   * 9 BIASES db=Σ_t dY → folded into the dW-owner's dY stream (free)       [P2]
//   * 2 EMBEDDINGS       → owner-scan over full T (owner = row % nCTA)       [P2]
//   * 10 LN affine (γ/β) → tile-local in P1 into a TINY per-CTA partials
//        buffer (132 x 1280 floats ≈ 0.68 MB), then a deterministic
//        ascending-CTA reduce                                               [P2]
//
// PRECISION (DESIGN §5.1 — torch-autocast boundary):
//   * the six linear families (in/out/ff0/ff2/head + all dX/dW): bf16 operands,
//     wgmma, fp32 accumulator, bf16 acts / fp32 grad.
//   * attention scores/softmax, LayerNorm, GELU, cross-entropy: fp32 (kept
//     identical to the scalar oracle math). S=4 is tiny so scores/ctx stay the
//     per-sample fp32 special-case (DESIGN §3.1 — too small for wgmma).
//
// VALIDATION: the wgmma engine + pipeline are silicon-validated by
//   tests/hw/test_wgmma_substrate.py. This header is gated by
//   tests/hw/test_decoder_tc.py (per-orientation micro-gates + full-cell grad
//   parity vs the bf16-rounded oracle + determinism + grok-floor). The scalar
//   path's fp32 gates are untouched.
//
// PORTABILITY: arch-guarded on __CUDA_ARCH__ >= 900. The substrate falls back to
//   scalar pre-sm_90; the cell driver only selects this body on sm_90 builds.
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/decoder_layout.cuh"
#include "csrc/fused/sm_90/dec_weights.cuh"   // reuse DecWeights/DecGrad/bind + fp32 helpers
#include "csrc/fused/sm_90/parallel_config.cuh"   // par::SingleGPU default tmpl arg (EDIT D)
#include "csrc/fused/sm_90/tp_transport.cuh"      // LoopbackTransport default tmpl arg + reduce (EDIT D)
#include "csrc/backends/cuda/sm_90/wgmma.cuh"
#include "csrc/backends/cuda/sm_90/tile_pipeline.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math_constants.h>

namespace sg { namespace fused { namespace sm90 {

namespace wgs = ::sg::sm90::wgs;

// ── Tunable knobs (DESIGN §9). #ifndef defaults compose a correct untuned
//    kernel (CONTRACT rule 3). SG_TUNED_TILE_M / SG_TUNED_TILE_N / depth are
//    shared with the substrate headers (same macro names). ──────────────────
#ifndef SG_TUNED_TILE_M
#define SG_TUNED_TILE_M 128
#endif
#ifndef SG_TUNED_TILE_N
#define SG_TUNED_TILE_N 128
#endif

// ── GEMM K-loop double-buffer depth + dW split-K factor (the validated mamba TC
//    perf fixes, carried here: model_stage_mamba_tc.cuh). S=2 stages the next
//    K-tile into the OTHER smem slot while the wgmma on the current tile is
//    async-resident (HBM operand-latency hiding); S=1 reproduces the old serial
//    path BIT-FOR-BIT. SPLIT-K (G) chunks the dW K=T contraction across CTAs so
//    the ~62% idle SMs in P2 (decoder has ~50 dW tiles / 132 SMs) do real work;
//    G=1 == the single-CTA-per-tile path. Both preserve ascending-k fp32
//    accumulation → parity + A/A/A determinism UNCHANGED (mamba 5/5 confirms). ──
#ifndef SG_TUNED_DEC_GEMM_STAGES
#define SG_TUNED_DEC_GEMM_STAGES 2
#endif
#ifndef SG_TUNED_DEC_DW_SPLITK
// Roofline ratchet (2026-06-16, d=2048/B=4096): G=2 beats the prior G=4 default by −2.5% on
// 3 seeds (G=8 was +2.6% SLOWER — at d=2048 the dW tiles already fill the grid, so MORE split-K
// only adds partial-reduce + workspace traffic). The deterministic ascending-chunk reduce is
// order-stable for any G, so parity + A/A/A determinism hold. G==1 = single-CTA path.
// SUPERSEDED 2026-06-16 → default now G=1: with DW_STAGE=1 (contiguous-layout staging) the
// single-CTA dW is 2.05× faster than G=2-scalar (920.7 vs 1889.8 ms @ d=2048, gate-green ×3 seeds).
// split-K was a SCALAR-dW grid-fill mitigation — obsolete now that staging, not the grid, is the lever.
#define SG_TUNED_DEC_DW_SPLITK 1
#endif

// ── M-atom INTERLEAVE width cap (task #13 hill-climb win). The GEMM microkernel
//    processes stacked m64 atoms in groups of min(MaxAtomsM, this); within a group
//    the per-k wgmmas (one per atom) issue back-to-back into independent fp32
//    fragments sharing ONE staged B-tile → the tensor pipe overlaps the MMAs AND
//    the (HBM-bound) weight B-tile is staged once per group instead of per atom.
//    Capped (default 2) so the accumulator-register + A-smem cost stays bounded
//    regardless of m_atoms (a dW tile can be 8 atoms; an 8-wide interleave would
//    need 8×(N/2) fp32 accumulator regs). 1 = no interleave (the old serial path,
//    bit-for-bit). Production fwd/dX use kAtomsM=2 → full 2-wide interleave.
#ifndef SG_TUNED_DEC_GEMM_INTERLEAVE
#define SG_TUNED_DEC_GEMM_INTERLEAVE 2
#endif

// ── dW STAGING METHOD (Track E P0 redirect; campaign 2026-06-16). The dW GEMM
//    (dW = dYᵀ·X, K=T) is STAGING-bound, not drain-bound: ~97% of each dW K-step
//    is the scalar transposed-strided operand gather in stage_kmajor_tile, only
//    ~3% is the wgmma (vit_findings.md clock64). The reverted P0 pipelined the
//    MMA — the wrong lever (you cannot hide 97% staging behind 3% MMA). The REAL
//    lever is faster STAGING. This macro selects HOW the dW operands reach the
//    wgmma; it changes NEITHER the reduction order NOR the math (only transport),
//    so fp64-oracle parity + A/A/A determinism are preserved BY CONSTRUCTION.
//
//      0 (DEFAULT, revert-safe): the proven scalar path. dectc_dw_run_tile feeds
//        the engine LAMBDA sources doing the transposed-strided gmem read
//        dY[k·Nout+m] / X[k·Kin+nn]; the engine's cp.async-ring gate
//        (DecTileSrcIsGmem) is false for lambdas → synchronous 2-byte
//        LDG→reg→STS stage_kmajor_tile. Byte-identical to every shipped build.
//
//      1 (CONTIGUOUS-LAYOUT, Track E option B): a CHEAP grid-cooperative
//        pre-transpose (dectc_dw_transpose_operands) writes each weight's dY/X
//        ONCE per step into a CONTIGUOUS K-major gmem scratch (dYt[Nout,K],
//        Xt[Kin,K]; rows = the staged mn axis, K-CONTIGUOUS + 16B-aligned). The
//        dW then builds DecGmemTileSrcA/B over that scratch → the engine's
//        EXISTING, silicon-validated kRingAsync cp.async ring (the −14.2% fwd/dX
//        win) streams the operands in 16-byte coalesced cp.async copies instead
//        of the 2-byte strided gather. NO new MMA, NO TMA substrate, NO new
//        descriptor path — pure reuse of the proven fast path. The pre-transpose
//        is a pure bf16 copy (dYt[m,k]=dY[k,m]: no arithmetic) reused across ALL
//        M-atom-groups × N-tiles × split-K chunks of that weight, so its O(T·N)
//        one-touch cost amortizes against the dW's many re-reads of each operand.
#ifndef SG_TUNED_DEC_DW_STAGE
#define SG_TUNED_DEC_DW_STAGE 1
#endif

// ── fwd/dX DEEPER cp.async RING (campaign P1-fwd/dX lever; 2026-06-16). The
//    P1_fwd (28.8%) + P1_bwd-dX (27.7%) GEMMs are 56.5% of the step. Unlike dW
//    (97% scalar-gather staging → STAGING-bound, where the reverted P0 producer/
//    consumer split REGRESSED −11.6%), fwd/dX are bf16-CONTIGUOUS: they already
//    run the proven kRingAsync cp.async double-buffer ring (16B coalesced LDGSTS,
//    no scalar gather), so they are plausibly DRAIN/LATENCY-bound — the regime
//    where DEEPER prefetch SHOULD help. This knob deepens ONLY the fwd/dX ring
//    (the lowest-risk mechanism (a): reuses the SAME silicon-validated ring
//    machinery, the SAME M-atom-interleave (kIL), the SAME all-256-threads-stage /
//    WG0-consume structure). It does NOT touch dW (just-fixed staging), the
//    optimizer tail, attention/scan, or the scalar production path.
//
//    SG_TUNED_DEC_FWD_PIPE (0/1): master enable. 0 (DEFAULT, revert-safe) →
//      the fwd/dX ring depth == SG_TUNED_DEC_GEMM_STAGES (the shipped S=2 double-
//      buffer), so the engine is BYTE-IDENTICAL to every shipped build (PTX-
//      verified). 1 → the fwd/dX ring uses SG_TUNED_DEC_FWD_STAGES slots.
//    SG_TUNED_DEC_FWD_STAGES (2..4): fwd/dX ring depth when PIPE=1. S slots keep
//      S-1 cp.async groups IN FLIGHT (vs the S=2 ring's single prefetch tile),
//      so more HBM operand latency is hidden behind the in-flight wgmmas.
//
//    PARITY BY CONSTRUCTION: the deeper ring changes ONLY when each operand tile
//    becomes available (MEMORY-LOAD reorder) — the wgmma ISSUE sequence is
//    UNCHANGED (ascending-k, k=0 overwrite / k>0 accumulate, per-k commit +
//    wgmma_wait_group<0> drain), so the fp32 accumulation order is bit-identical
//    → fp64-oracle parity + A/A/A determinism hold (the same invariant the S=2
//    ring and the dW contiguous-staging KEEP already preserve). At S=2 the deeper
//    loop COLLAPSES to the current ring exactly (cp_async_wait_group<S-2> ==
//    <0>), which is why PIPE=0 (S maps to the GEMM_STAGES default) is byte-clean.
//
//    OCCUPANCY/REGISTER CEILING: the deeper ring adds NO accumulator registers
//    (the per-atom WgmmaAccum<N> fragments are independent of S); it adds only
//    smem (DecTcSmem.sA/sB grow ∝ ring depth). The persistent grid-barrier
//    REQUIRES ≥1 CTA/SM (the launcher REFUSES via cudaErrorLaunchOutOfResources
//    if occupancy<1) — so the ring depth is smem-capped, not reg-capped. This is
//    the OPPOSITE of the P0 failure mode (P0 stole MMA threads + regs for a
//    producer/consumer split on a staging-bound GEMM). DecTcSmem sizes sA/sB for
//    max(fwd-ring, dW-ring) depth; see the smem-budget note there.
#ifndef SG_TUNED_DEC_FWD_PIPE
#define SG_TUNED_DEC_FWD_PIPE 1   // BAKED default: gate verdict — entry-a deeper cp.async ring (+1.49x); entry-b PIPE=2 LOST (857 vs 618ms)
#endif
#ifndef SG_TUNED_DEC_FWD_STAGES
#define SG_TUNED_DEC_FWD_STAGES 4 // BAKED with PIPE=1 (STAGES>2 → dynamic-smem path; gated 11/11 x 3 seeds @ 618ms/6.477%)
#endif

// ── fwd/dX FINE sub-phase profiler (campaign P1-fwd/dX diagnosis; 2026-06-16).
//    Splits the coarse P1_fwd (slot 0) / P1_bwd (slot 1) clock64 regions into
//    fine sub-counters INSIDE the GEMM engine's cp.async ring, so the main loop
//    can read WHERE the time goes (drain-bound vs compute/epilogue-bound) before
//    committing the deeper-ring lever. ONLY meaningful when SG_DEC_PROFILE is
//    also set (the fine array + the engine stamps are #if-gated on BOTH). OFF by
//    default → ZERO overhead, byte-identical when off (PTX-verified). See
//    g_dec_prof_fwd_fine + the SG_DEC_FWD_FINE_* slot enum in
//    fused_decoder_megakernel.cuh.
#ifndef SG_DEC_PROFILE_FWD_FINE
#define SG_DEC_PROFILE_FWD_FINE 0
#endif

namespace dectc {

// dW staging-method selector (see the SG_TUNED_DEC_DW_STAGE macro doc above).
// 0 = scalar transposed-strided gather (default, byte-identical); 1 = contiguous
// K-major pre-transpose + the proven cp.async ring fast path.
constexpr int kDecDwStage = SG_TUNED_DEC_DW_STAGE;
static_assert(kDecDwStage == 0 || kDecDwStage == 1,
              "SG_TUNED_DEC_DW_STAGE must be 0 (scalar) or 1 (contiguous-transpose)");

constexpr int kDecMaxIL = SG_TUNED_DEC_GEMM_INTERLEAVE;
static_assert(kDecMaxIL >= 1 && kDecMaxIL <= 4,
              "SG_TUNED_DEC_GEMM_INTERLEAVE must be 1 (serial) .. 4");
constexpr int kDecTcStages = SG_TUNED_DEC_GEMM_STAGES;
static_assert(kDecTcStages >= 1 && kDecTcStages <= 2,
              "SG_TUNED_DEC_GEMM_STAGES must be 1 (serial) or 2 (double-buffer)");

// fwd/dX DEEPER-ring depth (SG_TUNED_DEC_FWD_PIPE / SG_TUNED_DEC_FWD_STAGES). When
// the master enable is OFF, the fwd/dX ring depth == kDecTcStages (the shipped
// double-buffer) so the engine is byte-identical. When ON it is the FWD_STAGES
// knob (2..4). The dW path is UNAFFECTED (it never takes the ring — lambda
// sources, DecTileSrcIsGmem=false — so it keeps kDecTcStages regardless).
//
// ── DESIGN-TOURNAMENT: PIPE is now a 3-way SELECTOR (0/1/2), NOT a scale. The
//    three modes are INDEPENDENT ALTERNATIVES the main loop GPU-gates and keeps
//    the FASTER (they are never merged):
//      PIPE=0  OFF — the shipped S=2 ring VERBATIM → byte-identical PTX.
//      PIPE=1  entry (a): the SAME all-threads-stage / WG0-compute ring DEEPENED
//              to FWD_STAGES slots (more cp.async groups in flight; the deeper-
//              prefetch lever). Loose overlap — staging + MMA share the 256
//              threads; the cp.async WAIT is still on the consumer's critical path.
//      PIPE=2  entry (b): the validated producer/consumer mbarrier ENGINE
//              (tile_pipeline.cuh choreography, transcribed here over the EXISTING
//              ring tiles + the M-atom-interleave + the bit-identical issue_k).
//              WG0 is a PURE producer (cp.async + mbarrier arrive), WG1 a PURE
//              consumer (mbarrier wait + wgmma). The producer runs up to Depth
//              stages AHEAD of the consumer, so the cp.async drain is HIDDEN
//              behind the consumer's in-flight wgmmas (TRUE drain-hiding — the
//              regime the fine profiler proved fwd/dX live in: ring WAIT 43% fwd /
//              56% dX dominates WGMMA). NOT a repeat of the P0 dW failure: dW was
//              ~97%-staging / ~3%-MMA (nothing to hide behind, and the split stole
//              the staging threads); fwd/dX are the OPPOSITE (drain/latency-bound,
//              contiguous cp.async, real MMA share). PARITY: see the engine body —
//              the overlap reorders LOADS only; the wgmma ISSUE sequence and the
//              ascending-k fp32 accumulation are bit-identical to PIPE=0.
constexpr int kDecFwdPipe = SG_TUNED_DEC_FWD_PIPE;
static_assert(kDecFwdPipe == 0 || kDecFwdPipe == 1 || kDecFwdPipe == 2,
              "SG_TUNED_DEC_FWD_PIPE must be 0 (off, byte-identical), 1 (entry-a deeper "
              "fwd/dX ring), or 2 (entry-b producer/consumer mbarrier engine)");
// Is the producer/consumer mbarrier engine selected (entry b)? A compile-time bool
// the engine + DecTcSmem branch on; false (PIPE 0|1) → not one mbarrier byte emitted.
constexpr bool kDecFwdPipeEngine = (kDecFwdPipe == 2);
// PIPE=0 → inherit kDecTcStages VERBATIM (1 or 2; preserves the serial S=1 build
// and the byte-identical S=2 default). PIPE=1 OR 2 → the FWD_STAGES knob (asserted
// 2..4 below; <2 is not a real ring, >4 blows the smem budget). Both deeper modes
// share the SAME ring DEPTH (the producer/consumer engine's Depth == the deeper
// ring's slot count), so the sA/sB allocation is identical for PIPE 1 vs 2.
constexpr int kDecFwdStages = kDecFwdPipe ? SG_TUNED_DEC_FWD_STAGES : kDecTcStages;
static_assert(kDecFwdPipe == 0 || (kDecFwdStages >= 2 && kDecFwdStages <= 4),
              "SG_TUNED_DEC_FWD_STAGES must be 2..4 when SG_TUNED_DEC_FWD_PIPE=1|2 "
              "(smem-capped; >1 for a real ring). PIPE=0 inherits GEMM_STAGES (1|2).");
// The widest ring any call site uses, for the DecTcSmem sA/sB allocation: fwd/dX
// take kDecFwdStages; dW takes kDecTcStages. Spell the max out (covers the S=1
// build, where both are 1, AND the PIPE=1 deeper case where fwd > dW).
constexpr int kDecRingStagesMax =
    (kDecFwdStages > kDecTcStages) ? kDecFwdStages : kDecTcStages;

constexpr int kDecTcSmemA1 = wgs::kWgmmaAtomM * wgs::kWgmmaAtomK;   // 64*16 bf16
constexpr int kDecTcSmemB1 = SG_TUNED_TILE_N * wgs::kWgmmaAtomK;    // N*16 bf16
constexpr int kDecDwSplitK = SG_TUNED_DEC_DW_SPLITK;
static_assert(kDecDwSplitK >= 1, "SG_TUNED_DEC_DW_SPLITK must be >= 1");

// ════════════════════════════════════════════════════════════════════════
//  fwd/dX FINE sub-phase profiler (SG_DEC_PROFILE && SG_DEC_PROFILE_FWD_FINE).
//  Diagnostic-only; NEVER on the shipped path (both flags default OFF; the
//  production _ops sets neither). When OFF, NONE of this — nor any engine stamp
//  — is compiled (the engine's fine-stamp blocks are #if-gated on the SAME pair),
//  so the kernel's PTX/regalloc is BYTE-IDENTICAL to every shipped build.
//
//  The coarse profiler (g_dec_prof_max[0]=P1_fwd, [1]=P1_bwd) wraps the WHOLE
//  tile fwd / bwd call (GEMMs + LN/softmax/attention/elementwise). This array
//  splits the time SPENT INSIDE THE GEMM ENGINE's cp.async ring into 5 fine
//  sub-counters, separately for the fwd ring (phase 0) and the dX ring (phase 1):
//    [phase*kDecFwdFineSub + sub], with sub ∈ {
//      0 ISSUE   = cp.async LDGSTS issue (stage_k_async: the 16B coalesced copies)
//      1 WAIT    = cp.async drain (cp_async_wait_group<...>) — the DRAIN/latency cost
//      2 WGMMA   = wgmma issue + commit + wait_group<0> (the MMA compute+drain)
//      3 EPI     = epilogue fragment-decode + bf16/fp32 store
//      4 BARRIER = fence_async_proxy + __syncthreads (the ring publish barrier)
//    }
//  Stamped thread-0-only, atomicMax across CTAs (= the slowest CTA = the
//  critical path the host wall sees), the SAME idiom as g_dec_prof_max. The dW
//  GEMM never enters the ring branch (lambda sources → !kRingAsync), so it never
//  stamps these — the fine counts are PURELY fwd/dX, as intended.
#if defined(SG_DEC_PROFILE) && SG_DEC_PROFILE_FWD_FINE
constexpr int kDecFwdFineSub    = 5;   // ISSUE, WAIT, WGMMA, EPI, BARRIER
constexpr int kDecFwdFinePhases = 2;   // 0 = fwd ring, 1 = dX ring
constexpr int kDecFwdFineSlots  = kDecFwdFinePhases * kDecFwdFineSub;   // 10
enum DecFwdFineSub { kFineIssue = 0, kFineWait = 1, kFineWgmma = 2,
                     kFineEpi = 3, kFineBarrier = 4 };
// Plain __device__ (matches g_dec_prof_max). Each TU that enables the pair is its
// own JIT extension (.so) — no cross-TU linkage, so no ODR concern. The .cu TU's
// reader (tc_profile_read_fwd_fine) copies+resets it via cudaMemcpyFromSymbol.
__device__ unsigned long long g_dec_prof_fwd_fine[kDecFwdFineSlots];
// Accumulate a fine delta into [phase][sub] (thread-0-only; atomicMax across CTAs).
__device__ __forceinline__ void dec_fwd_fine_acc(int phase, int sub,
                                                 unsigned long long delta) {
    atomicMax(&g_dec_prof_fwd_fine[phase * kDecFwdFineSub + sub], delta);
}
#endif

// fwd/dX engine call-site phase tags. The fwd wrappers append SG_DEC_FINE_FWD (the
// `prof_phase=0` arg), the dX wrappers SG_DEC_FINE_DX (`=1`) — but ONLY when the
// fine-profiler pair is set. When OFF these expand to NOTHING, so the wrapper's
// engine call is the original 9-arg form → byte-identical PTX (and the selftest's
// 9-arg calls are unaffected; prof_phase keeps its -1 default there).
#if defined(SG_DEC_PROFILE) && SG_DEC_PROFILE_FWD_FINE
#define SG_DEC_FINE_FWD , /*prof_phase=fwd*/ 0
#define SG_DEC_FINE_DX  , /*prof_phase=dX */ 1
#else
#define SG_DEC_FINE_FWD
#define SG_DEC_FINE_DX
#endif

// Token-tile rows a CTA owns. Must be a multiple of 64 (wgmma atom M) AND of
// kSeq (so a tile boundary is a sample boundary — attention stays in-tile).
constexpr int kTileM = SG_TUNED_TILE_M;
static_assert(kTileM % wgs::kWgmmaAtomM == 0,
              "SG_TUNED_TILE_M must be a multiple of 64 (wgmma m64 atom)");
static_assert(kTileM % dec::kSeq == 0,
              "SG_TUNED_TILE_M must be a multiple of kSeq=4 (tile=sample boundary)");
constexpr int kAtomsM = kTileM / wgs::kWgmmaAtomM;   // stacked m64 atoms per tile
constexpr int kSamplesPerTile = kTileM / dec::kSeq;  // 32 for TILE_M=128

// ── LN vector-grad partials layout (the tile-local γ/β grads). Dense order:
//    4 slots/layer (n1.w,n1.b,n2.w,n2.b) for li∈[0,L), then norm.w,norm.b. At L=2
//    this is {6,7,8,9,18,19,20,21,26,27} (the original 10-slot table). We store
//    them densely [kNumLnVec x kD] per CTA; the P2 reduce maps them back by tensor
//    index via dec_lnvec_tensor_idx (a formula — a __constant__ array can't be
//    filled by a loop at L=48). L-GENERAL: kNumLnVec = 4*L+2. ──
constexpr int kNumLnVec = 4 * dec::kLayers + 2;   // n1_w,n1_b,n2_w,n2_b ×L + norm_w,norm_b
constexpr int kLnVecElems = kNumLnVec * dec::kD;  // (4*L+2)*kD  (1280 at L=2)
// dec_layout tensor index of LN-vector dense slot v. v∈[0,4L): li=v/4, kind=v%4 →
// 6+12*li+kind (n1.w/b,n2.w/b are tensor 6..9 of each 12-tensor layer block);
// v=4L → norm.w (2+12*L); v=4L+1 → norm.b (2+12*L+1). At L=2 reproduces the old
// kLnVecTensorIdx[10] EXACTLY ({6,7,8,9,18,19,20,21,26,27}).
__host__ __device__ __forceinline__ int dec_lnvec_tensor_idx(int v) {
    const int Lx4 = 4 * dec::kLayers;
    if (v < Lx4) return 6 + 12 * (v / 4) + (v % 4);
    return 2 + 12 * dec::kLayers + (v - Lx4);     // 2+12L (norm.w), 2+12L+1 (norm.b)
}

// ── Muon 2D-weight table (the matrices Newton-Schulz orthogonalizes). The eager
//    Muon auto-splits params by p.ndim: ndim==2 → NS, everything else → AdamW
//    (muon.py:91-98 _split_by_ndim; muon.h:75-76). For the small decoder the
//    ndim==2 weights are exactly these 11 (the flat named_parameters() tensor
//    index + rows[dim0] + cols[dim1], matching the LIVE model's p.shape EXACTLY —
//    verified against m.named_parameters()). NOTE vs ViT: the decoder's `tok`
//    (Embedding [99,128]) and `pos` (Embedding [4,128]) ARE 2D, so they take the
//    NS path (ViT's cls_token is ndim==3 → AdamW). All biases + LayerNorm γ/β
//    (ndim==1) take the AdamW 1D tail. The kernel's Muon P2.7 loops THIS table
//    running the grid-cooperative NS per matrix; P3 routes tensor t to the NS
//    apply iff it is in the table, else the AdamW aux tail. Indices MUST match
//    decoder_layout / named_parameters() order. ──
// kDecNumMuon2D = tok + pos + 4 weights/layer (in_proj,out_proj,ff0,ff2) + head.out
//   = 2 + 4*L + 1  (= 11 at L=2). The table is now a FORMULA (dec_muon_2d) — a
// __device__ __constant__ array can't be loop-filled to 195 entries at L=48.
constexpr int kDecNumMuon2D = 2 + 4 * dec::kLayers + 1;
struct DecMuon2D { int tidx; int rows; int cols; };
// The mi-th Muon 2D matrix (tensor index + rows/cols), L-general. Dense order:
//   mi=0 tok[V,d]; mi=1 pos[seq,d];
//   mi∈[2,2+4L): li=(mi-2)/4, kind=(mi-2)%4 →
//     kind0 in_proj  tidx 2 +12li  [3d,d]
//     kind1 out_proj tidx 4 +12li  [d, d]
//     kind2 ff0      tidx 10+12li  [dff,d]
//     kind3 ff2      tidx 12+12li  [d, dff]
//   mi=2+4L head out.weight tidx 2+12L+2 [V,d].
// At L=2 reproduces the old kDecMuon2D[11] EXACTLY (tidx {0,1,2,4,10,12,14,16,22,24,28}).
__host__ __device__ __forceinline__ DecMuon2D dec_muon_2d(int mi) {
    if (mi == 0)                         return { 0, dec::kVocab, dec::kD };   // tok
    if (mi == 1)                         return { 1, dec::kSeq,   dec::kD };   // pos
    if (mi == 2 + 4 * dec::kLayers)      return { 2 + 12 * dec::kLayers + 2, dec::kVocab, dec::kD }; // head.out
    const int li   = (mi - 2) / 4;
    const int kind = (mi - 2) % 4;
    if (kind == 0) return { 2  + 12 * li, 3 * dec::kD, dec::kD   };  // in_proj
    if (kind == 1) return { 4  + 12 * li, dec::kD,     dec::kD   };  // out_proj
    if (kind == 2) return { 10 + 12 * li, dec::kDff,   dec::kD   };  // ff0
    return            { 12 + 12 * li, dec::kD,     dec::kDff };      // ff2
}
// Is tensor index `t` a Muon 2D matrix (orthogonalized in P2.7)? P3 routes only
// the 1D / non-2D weights to the AdamW aux tail. Closed-form (no table scan):
//   t∈{0,1} (tok/pos), OR t==head.out (2+12L+2), OR a per-layer 2D weight
//   (t∈[2,2+12L) and (t-2)%12 ∈ {0,2,8,10} = in_w/out_w/ff0_w/ff2_w).
__device__ __host__ __forceinline__ bool dec_is_muon_2d(int t) {
    if (t == 0 || t == 1) return true;
    if (t == 2 + 12 * dec::kLayers + 2) return true;       // head out.weight
    if (t >= 2 && t < 2 + 12 * dec::kLayers) {
        const int r = (t - 2) % 12;
        return (r == 0 || r == 2 || r == 8 || r == 10);
    }
    return false;
}
// Largest 2D weight (numel) + largest #rows over the table — sizes the per-matrix
// NS scratch (the stage runs ONE matrix at a time, reusing the buffers). ff.0
// [512,128]=65536 is the largest numel; ff.0 rows=512 is the largest #rows (A=XXᵀ
// is rows×rows). Mirrors vit's kVitMuonMaxNumel/kVitMuonMaxRows.
constexpr int kDecMuonMaxNumel = dec::kDff * dec::kD;   // 512*128 = 65536 (ff.0/ff.2)
constexpr int kDecMuonMaxRows  = dec::kDff;             // 512 (ff.0 rows)

// ════════════════════════════════════════════════════════════════════════
//  HBM bf16 ACTS buffer (Fork B). Carved from the FRONT of the workspace the
//  host already allocates (float[nCTA*total + nCTA]); Fork B does not use that
//  space for the eliminated 223 MB grad partials, so it is free real estate.
//  Reinterpreted as __nv_bfloat16. Total 80,546,592 bf16 (161 MB) << the 223 MB
//  workspace; 62 MB headroom. All regions row-major [rows x width].
//
//  Offsets are RUNTIME (depend on T = B*kSeq, B is a host arg), computed by
//  DecActs::bind. The struct holds base pointers per region.
// ════════════════════════════════════════════════════════════════════════
struct DecActs {
    // Per-layer linear INPUTS X (needed by dW = dY^T @ X):
    __nv_bfloat16* X_in[dec::kLayers];    // [T, d]    in_proj input (= layer input)
    __nv_bfloat16* X_ctx[dec::kLayers];   // [T, d]    out_proj input (attn context)
    __nv_bfloat16* X_x1[dec::kLayers];    // [T, d]    ff0 input (n1 output)
    __nv_bfloat16* X_gact[dec::kLayers];  // [T, dff]  ff2 input (gelu output)
    // Per-layer linear OUTPUT adjoints dY (needed by dW + bias db = Σ_t dY):
    __nv_bfloat16* dY_qkv[dec::kLayers];  // [T, 3d]   in_proj output adjoint
    __nv_bfloat16* dY_a[dec::kLayers];    // [T, d]    out_proj output adjoint
    __nv_bfloat16* dY_ff0[dec::kLayers];  // [T, dff]  ff0 output adjoint
    __nv_bfloat16* dY_ff2[dec::kLayers];  // [T, d]    ff2 output adjoint
    // Head (B rows — last position only):
    __nv_bfloat16* X_hn;                  // [B, d]    head input
    __nv_bfloat16* dY_logits;             // [B, V]    head output adjoint (dlogits)
    // Embedding-input adjoint (needed by tok/pos owner-scan):
    __nv_bfloat16* dh0;                   // [T, d]
};

__device__ __forceinline__ DecActs dec_acts_bind(__nv_bfloat16* p, int T, int B) {
    DecActs a;
    int64_t off = 0;
    const int64_t Td = (int64_t)T * dec::kD;
    const int64_t T3d = (int64_t)T * 3 * dec::kD;
    const int64_t Tff = (int64_t)T * dec::kDff;
    for (int li = 0; li < dec::kLayers; ++li) {
        a.X_in[li]   = p + off; off += Td;
        a.X_ctx[li]  = p + off; off += Td;
        a.X_x1[li]   = p + off; off += Td;
        a.X_gact[li] = p + off; off += Tff;
        a.dY_qkv[li] = p + off; off += T3d;
        a.dY_a[li]   = p + off; off += Td;
        a.dY_ff0[li] = p + off; off += Tff;
        a.dY_ff2[li] = p + off; off += Td;
    }
    a.X_hn      = p + off; off += (int64_t)B * dec::kD;
    a.dY_logits = p + off; off += (int64_t)B * dec::kVocab;
    a.dh0       = p + off; off += Td;
    return a;
}

// ════════════════════════════════════════════════════════════════════════
//  bf16 WEIGHT PRE-STAGE (reg-pressure campaign C1 + cp.async-ring blocker (a)).
//  The fwd/dX GEMMs' B operand was the fp32 params blob CONVERTED ON READ
//  (__float2bfloat16 inside the staging accessor). That conversion web is the
//  measured marginal register demand of the K-loop (the pure-bf16 dW path
//  allocates spill-free at the same accumulator width while the fp32-read
//  fwd/dX paths spill ~2.2 KB when isolated), and it is the documented blocker
//  (a) of the cp.async/TMA ring: an async copy cannot convert, so the dominant
//  operand could not stream. Fix: convert ONCE per step into a bf16 weight
//  cache carved from the workspace, and stage the GEMM B operand from the
//  cache. cache[i] = __float2bfloat16(params[i]) is the IDENTICAL deterministic
//  per-element rounding the on-read path performed -> every GEMM consumes
//  BIT-IDENTICAL operand values; numerics/parity/A-A-A are unchanged BY
//  CONSTRUCTION (pure caching of a pure function; no reorder, no reassociation).
//
//  Only the 8 per-layer GEMM weight matrices (in_w/out_w/ff0_w/ff2_w x layers)
//  are cached: they are the ONLY fp32->bf16 on-read GEMM operands (the head
//  runs scalar fp32 per the oracle; embeddings/biases/LN are scalar fp32; the
//  dW GEMM operands are bf16 acts already). All sizes derive from the layout
//  constants -- no problem-specific hardcoding; the bench layout scales it.
//
//  C1-T (RING): a SECOND section of the cache holds the same 8 matrices
//  TRANSPOSED (W^T [Kin,Nout]). The dX GEMM reads W transposed; from the
//  row-major cache that staging is a k-stride 2-byte gather, which no 4/8/16B
//  cp.async can express (the same structural shape as ring blocker (b)).
//  Staging dX's B operand from W^T instead makes its tile rows K-CONTIGUOUS,
//  so the cp.async double-buffered ring streams BOTH fwd and dX weight
//  operands. WT[r,c] == W-cache[c,r] bit-identically (same fp32 element, same
//  deterministic rounding, written by exactly one thread) -> operand values,
//  parity and A/A/A are unchanged by construction. Cost: the cache doubles
//  (1.57 MB at d=128, 101 MB at d=1024 -- workspace headroom verified at both).
// ════════════════════════════════════════════════════════════════════════
struct DecWBf {
    const __nv_bfloat16* in_w[dec::kLayers];    // [3d, d]  per layer
    const __nv_bfloat16* out_w[dec::kLayers];   // [d, d]   per layer
    const __nv_bfloat16* ff0_w[dec::kLayers];   // [dff, d] per layer
    const __nv_bfloat16* ff2_w[dec::kLayers];   // [d, dff] per layer
    // ── C1-T (RING section): the SAME four matrices stored TRANSPOSED, W^T
    //    [Kin, Nout] row-major. WT[r,c] == W-cache[c,r] BIT-IDENTICAL (the same
    //    __float2bfloat16 of the same fp32 element, stored twice). Purpose: the
    //    dX GEMM's B operand reads W TRANSPOSED (srcB(n=kin,k=out) = W[out,kin],
    //    a k-stride-Kin 2-byte gather the cp.async ring cannot stream — the same
    //    structural shape as ring blocker (b)). Staging from W^T instead makes
    //    the dX B-tile rows K-CONTIGUOUS (16B-chunkable), unblocking the ring on
    //    the dX path with UNCHANGED operand values → numerics/parity/A-A-A
    //    identical by construction. ──
    const __nv_bfloat16* in_wT[dec::kLayers];   // [d, 3d]  per layer
    const __nv_bfloat16* out_wT[dec::kLayers];  // [d, d]   per layer
    const __nv_bfloat16* ff0_wT[dec::kLayers];  // [d, dff] per layer
    const __nv_bfloat16* ff2_wT[dec::kLayers];  // [dff, d] per layer
};
constexpr int64_t kWbfInW        = (int64_t)3 * dec::kD * dec::kD;
constexpr int64_t kWbfOutW       = (int64_t)dec::kD * dec::kD;
constexpr int64_t kWbfFf0W       = (int64_t)dec::kDff * dec::kD;
constexpr int64_t kWbfFf2W       = (int64_t)dec::kD * dec::kDff;
constexpr int64_t kWbfLayerElems = kWbfInW + kWbfOutW + kWbfFf0W + kWbfFf2W;
constexpr int64_t kWbfTotalElems = (int64_t)dec::kLayers * kWbfLayerElems;
// Full cache: straight section [kWbfTotalElems] + transposed section (C1-T,
// same element count — every matrix size is a multiple of 8 so both sections
// and every per-matrix base stay 16B-aligned for the cp.async ring).
constexpr int64_t kWbfCacheElems = 2 * kWbfTotalElems;
// Workspace floats the cache occupies (bf16 elems -> float units, rounded up).
__host__ __device__ __forceinline__ int64_t dec_wbf_floats() {
    return (kWbfCacheElems + 1) / 2;
}
__device__ __forceinline__ DecWBf dec_wbf_bind(const __nv_bfloat16* c) {
    DecWBf wb;
    #pragma unroll
    for (int li = 0; li < dec::kLayers; ++li) {
        const __nv_bfloat16* b = c + (int64_t)li * kWbfLayerElems;
        wb.in_w[li]  = b;
        wb.out_w[li] = b + kWbfInW;
        wb.ff0_w[li] = b + kWbfInW + kWbfOutW;
        wb.ff2_w[li] = b + kWbfInW + kWbfOutW + kWbfFf0W;
        // transposed section (C1-T): same per-layer/per-matrix offsets, based
        // kWbfTotalElems further in (element counts identical per matrix).
        const __nv_bfloat16* t = b + kWbfTotalElems;
        wb.in_wT[li]  = t;
        wb.out_wT[li] = t + kWbfInW;
        wb.ff0_wT[li] = t + kWbfInW + kWbfOutW;
        wb.ff2_wT[li] = t + kWbfInW + kWbfOutW + kWbfFf0W;
    }
    return wb;
}
// Grid-strided fp32->bf16 convert of the 8 matrices into the cache. Element-
// owned (each cache index written by exactly one thread), no atomics ->
// deterministic. Caller fences with the existing grid barrier before any GEMM
// reads the cache (P0->B0; the SAM re-convert gets its own barrier). The
// kDecOffsets index of [in_w,out_w,ff0_w,ff2_w] for layer li is {2,4,10,12} +
// 12*li (the 12-tensors-per-layer stride of the generated layout -- the same
// indices dectc_build_dw_specs walks).
__device__ __forceinline__ void dectc_wbf_convert(
        const float* __restrict__ params, __nv_bfloat16* __restrict__ cache,
        int cta, int nCTA) {
    const int64_t stride = (int64_t)nCTA * blockDim.x;
    for (int64_t i = (int64_t)cta * blockDim.x + threadIdx.x;
         i < kWbfCacheElems; i += stride) {
        // Section split: [0, kWbfTotalElems) = straight copies; the tail is the
        // C1-T TRANSPOSED section (the dX ring's B operand). Both sections are
        // element-owned (one writer per cache index) -> deterministic.
        const bool   tr = (i >= kWbfTotalElems);
        const int64_t ii = tr ? (i - kWbfTotalElems) : i;
        const int   li = (int)(ii / kWbfLayerElems);
        const int64_t r = ii % kWbfLayerElems;
        int wi; int64_t off, nout, kin;
        if      (r < kWbfInW)                       { wi = 2;  off = r;                                  nout = 3 * dec::kD; kin = dec::kD;   }
        else if (r < kWbfInW + kWbfOutW)            { wi = 4;  off = r - kWbfInW;                        nout = dec::kD;     kin = dec::kD;   }
        else if (r < kWbfInW + kWbfOutW + kWbfFf0W) { wi = 10; off = r - kWbfInW - kWbfOutW;             nout = dec::kDff;   kin = dec::kD;   }
        else                                        { wi = 12; off = r - kWbfInW - kWbfOutW - kWbfFf0W;  nout = dec::kD;     kin = dec::kDff; }
        // straight: source element `off` of W [Nout,Kin]. transposed: `off` is
        // element (rr,cc) = (off/Nout, off%Nout) of W^T [Kin,Nout]; its source is
        // W[cc,rr] = element cc*Kin + rr — the SAME fp32 through the SAME
        // deterministic rounding, so WT[rr,cc] is bit-identical to W-cache[cc,rr].
        const int64_t src = tr ? ((off % nout) * kin + (off / nout)) : off;
        cache[i] = __float2bfloat16(params[(int64_t)kDecOffsets[wi + li * 12] + src]);
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Canonical Major-K smem stager. The ss-wgmma operand smem tile (MN rows x
//  K=16 bf16) MUST be in the CUTLASS Major-K INTERLEAVE layout (wgmma.cuh):
//      idx(mn,k) = (k/8)*(MN*8) + mn*8 + (k%8)
//  This helper writes ONE such MN x 16 tile, pulling element (mn, k) from a
//  caller-provided accessor `src(mn, kbase + k)`. Routing the source axis
//  through the accessor is what lets the SAME wgmma issue (TransA=0/TransB=0,
//  the substrate-validated orientation) serve fwd / dX / dW: the staging loop
//  transposes physically; the engine never leaves the gated path.
//
//  Cooperative over `nthreads` threads starting at thread `t0` (a warpgroup).
// ════════════════════════════════════════════════════════════════════════
template <int MN, typename Src>
__device__ __forceinline__ void stage_kmajor_tile(
        __nv_bfloat16* smem_tile, int kbase, Src src, int t0, int nthreads) {
    // MN*16 elements, each thread strides.
    #pragma unroll 1
    for (int i = t0; i < MN * wgs::kWgmmaAtomK; i += nthreads) {
        const int mn = i / wgs::kWgmmaAtomK;
        const int k  = i % wgs::kWgmmaAtomK;
        const int dst = (k >> 3) * (MN * 8) + mn * 8 + (k & 7);
        smem_tile[dst] = src(mn, kbase + k);
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Flat-gmem K-major tile SOURCES (the cp.async RING enabler). The generic
//  engine takes accessor-shaped sources `src(mn, k)`; when the source is a
//  FLAT row-major gmem region whose rows are K-CONTIGUOUS and 16B-aligned
//  (base/ld multiples of 8 bf16 — true of every fwd/dX operand: the bf16 acts
//  and the C1/C1-T weight cache), the staging can be issued as 16-byte
//  cp.async copies instead of the 2-byte LDG→reg→STS element loop. These two
//  POD sources carry exactly the information that decision needs:
//    operator()(mn,k)        — the scalar accessor (identical semantics to the
//                              lambdas they replace; the S=1 serial path and
//                              any non-ring build still stage through it).
//    chunk16(mn,kbase,half)  — gmem address of the 16B half-row (8 bf16) that
//                              lands at smem offset half·(MN·8)+mn·8 of the
//                              canonical Major-K interleave tile.
//    row_valid(mn)           — false ⇒ the row is PAD (ragged N-tile): the
//                              ring zero-fills smem and must NOT read gmem
//                              (mirrors the `nn < Nout ? .. : 0` guard).
//  The A-form has no guard (the engine's A rows are always readable — the
//  caller guarantees the region, exactly as the current lambdas assume); the
//  B-form guards rows beyond `rows_valid`. A trait gates the engine's ring
//  branch at COMPILE TIME, so the dW path (lambda sources; transposed-strided
//  reads — ring blocker (b), out of scope) keeps the synchronous staging
//  unchanged.
// ════════════════════════════════════════════════════════════════════════
namespace decprim = ::sg::sm90::primitives;

struct DecGmemTileSrcA {
    const __nv_bfloat16* base;   // row 0, k 0 (16B-aligned)
    int ld;                      // row stride in ELEMENTS (multiple of 8)
    __device__ __forceinline__ __nv_bfloat16 operator()(int mn, int k) const {
        return base[(int64_t)mn * ld + k];
    }
    __device__ __forceinline__ const void* chunk16(int mn, int kbase, int half) const {
        return base + (int64_t)mn * ld + kbase + half * 8;
    }
    __device__ __forceinline__ bool row_valid(int) const { return true; }
};

struct DecGmemTileSrcB {
    const __nv_bfloat16* base;   // row n0, k 0 (16B-aligned)
    int ld;                      // row stride in ELEMENTS (multiple of 8)
    int rows_valid;              // rows >= this are pad (zero, never read)
    __device__ __forceinline__ __nv_bfloat16 operator()(int mn, int k) const {
        return mn < rows_valid ? base[(int64_t)mn * ld + k] : __float2bfloat16(0.f);
    }
    __device__ __forceinline__ const void* chunk16(int mn, int kbase, int half) const {
        return base + (int64_t)mn * ld + kbase + half * 8;
    }
    __device__ __forceinline__ bool row_valid(int mn) const { return mn < rows_valid; }
};

template <typename T> struct DecTileSrcIsGmem { static constexpr bool value = false; };
template <> struct DecTileSrcIsGmem<DecGmemTileSrcA> { static constexpr bool value = true; };
template <> struct DecTileSrcIsGmem<DecGmemTileSrcB> { static constexpr bool value = true; };

// ════════════════════════════════════════════════════════════════════════
//  Unpipelined single-CTA batch GEMM: D[M,N] = Σ_k A[m,k]·B[n,k], bf16 operands
//  (accessor-sourced + Major-K staged), fp32 accumulator, ascending-k. M is
//  kTileM (kAtomsM stacked m64 atoms — the substrate's unexercised TILE_M
//  stacking, gated here). N is a compile-time wgmma atom width. K=k_steps*16.
//
//  The consumer warpgroup (the FULL CTA's 256 threads act as producer+consumer
//  here in the SIMPLE unpipelined form: all 256 stage, then warpgroup 0's 128
//  threads — actually wgmma is warpgroup-scoped (.aligned, 128 threads). We use
//  ONE warpgroup (threads 0..127) for the MMA; staging uses all 256 threads.
//
//  smem: one A tile (64x16) + one B tile (Nx16) per stacked atom row, reused
//  across k-steps (unpipelined: stage k, mma k, repeat). Two buffers (A,B) of
//  the largest atom. Accumulators live in registers (kAtomsM fragments).
//
//  ACC OUTPUT: written via accessor `out(m_global, n_local, value)` so the
//  caller routes the fp32 result (the fragment decode gives (row,col) within the
//  64xN atom; the caller adds the atom's M-base).
//
//  GENERALITY (the M and N must be caller-parameterized, NOT hardwired to the
//  token tile — a dW GEMM has M = Nout in {99,128,384,512} and the fwd GEMMs
//  have N in {128,384,512}). This helper computes ONE M-block of `m_atoms`
//  stacked m64 atoms (rows [mbase0, mbase0 + m_atoms*64)) for ONE N-tile of the
//  compile-time atom width N. The CALLER loops M-atom-blocks × N-tiles to cover
//  arbitrary (M, N). MaxAtomsM bounds the register accumulator array
//  (compile-time); only `m_atoms` (<= MaxAtomsM) atoms are issued at runtime.
//  `n_real <= N` lets the caller mark the valid column count for a ragged N-tile
//  (e.g. head V=99 in a N=128 tile) so the epilogue suppresses pad columns; the
//  wgmma still runs the full N (pad operands are zero so pad outputs are inert).
//
//  Determinism: ascending-k, one CTA owns the tile end-to-end, no atomics.
// ════════════════════════════════════════════════════════════════════════
// `prof_phase` (default -1) is the FINE-profiler phase tag (0 = fwd ring, 1 = dX
// ring) used ONLY when SG_DEC_PROFILE && SG_DEC_PROFILE_FWD_FINE is set — the
// fwd wrappers pass 0, the dX wrappers pass 1, dW leaves the default. It is a
// trailing defaulted scalar that is NEVER referenced unless that pair is set
// (then it selects which g_dec_prof_fwd_fine phase the engine stamps), so the
// selftest's existing 9-arg calls still compile AND the shipped PTX is byte-
// identical (an unreferenced trailing defaulted arg is dropped by the optimizer;
// PTX-verified OFF-vs-baseline).
// `pipeBars` (default nullptr) is the producer/consumer mbarrier-words base for the
// PIPE=2 engine (entry b) — 2·RS `unsigned long long` words (full[RS] then empty[RS]),
// carved from the kernel's DecTcSmem::pipe_bars. It is dereferenced ONLY inside the
// `if constexpr (kDecFwdPipeEngine)` sub-branch (PIPE=2), which is compiled OUT for
// PIPE 0|1 — so the nullptr the wrappers pass on the shipped build is UNREFERENCED
// and dropped by the optimizer (byte-identical PTX, like prof_phase). dW + the TP
// fp32-W overloads + the selftest's 9-arg calls leave it null (they never take the
// ring → !kRingAsync → the engine sub-branch is unreachable for them).
template <int N, int MaxAtomsM, typename SrcA, typename SrcB, typename Out>
__device__ void tc_gemm_block_unpipelined(
        int mbase0, int m_atoms, int n_real, int k_steps,
        SrcA srcA, SrcB srcB, Out out,
        __nv_bfloat16* smemA, __nv_bfloat16* smemB,
        unsigned long long* pipeBars = nullptr,
        int prof_phase = -1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    (void)prof_phase; (void)pipeBars;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;            // 256
    const bool in_wg0 = (tid < 128);
    const int tid_wg = tid & 127;
    constexpr int S = kDecTcStages;

    // ── M-ATOM-INTERLEAVED wgmma pipeline (overlaps the tensor pipe + HALVES the
    //    redundant B-tile staging; THE hill-climb win, task #13). The OLD body ran
    //    each stacked m64 atom's k-chain to completion SEQUENTIALLY (atom0 chain →
    //    atom0 epilogue → atom1 chain → …): every wgmma was followed by a per-issue
    //    wait, the shared B (weight) tile was re-staged ONCE PER ATOM, and the atom-a
    //    epilogue stores interleaved with atom-a+1's first wgmma (ptxas C7515).
    //
    //    Here, atoms are processed in GROUPS of kIL (= min(MaxAtomsM, kDecMaxIL)).
    //    Within a group, EACH k-step issues the kIL wgmmas (one per atom) BACK-TO-
    //    BACK into their OWN fp32 fragments, sharing ONE staged B-tile, before the
    //    single per-k wait. Two wins: (1) the kIL atoms are INDEPENDENT (distinct
    //    M-rows / accumulators) → the tensor pipe overlaps their MMA execution
    //    instead of paying each latency raw; (2) the B-tile is staged ONCE for the
    //    whole group instead of kIL times → the (HBM-bound) weight-operand traffic
    //    drops kIL×. Measured: d=1024 B=16384 step 2084→1624 ms (+28% TF/s).
    //
    //    kIL is CAPPED (kDecMaxIL=2) so the register/smem cost is bounded regardless
    //    of m_atoms (the dW micro-gate runs Nout=512 → 8 atoms; an 8-wide interleave
    //    would need 8×64 accumulator regs). Groups reuse the SAME ring slots
    //    sequentially (like the old atom loop). Per-atom accumulation stays
    //    ASCENDING-k (k=0 overwrite, k>0 add) → numerics bit-identical + A/A/A
    //    determinism UNCHANGED. Ring stages kIL A-tiles (slot sl, atom-in-group ai at
    //    +ai·kDecTcSmemA1) + ONE shared B-tile; smem{A} must hold kIL·kDecTcStages
    //    tiles (DecTcSmem sizes it for kAtomsM=production max).
    constexpr int kIL = (MaxAtomsM < kDecMaxIL) ? MaxAtomsM : kDecMaxIL;
    wgs::WgmmaAccum<N> acc[kIL];                 // kIL live fragments per group

    // Stage k-tile for a group of `g_atoms` (<= kIL) atoms based at `gbase`: the
    // g_atoms A-tiles (rows gbase+ai·64) + the shared B-tile, into ring slot k % S.
    auto stage_k = [&] (int gbase, int g_atoms, int k) {
        const int sl = k % S;
        for (int ai = 0; ai < g_atoms; ++ai) {
            const int mbase = gbase + ai * wgs::kWgmmaAtomM;
            stage_kmajor_tile<wgs::kWgmmaAtomM>(
                smemA + ((int64_t)sl * kIL + ai) * kDecTcSmemA1, k * wgs::kWgmmaAtomK,
                [&] (int mn, int kk) { return srcA(mbase + mn, kk); }, tid, nthreads);
        }
        stage_kmajor_tile<N>(
            smemB + (int64_t)sl * kDecTcSmemB1, k * wgs::kWgmmaAtomK,
            [&] (int mn, int kk) { return srcB(mn, kk); }, tid, nthreads);
    };
    // Issue the group's g_atoms wgmmas for staged slot k (k=0 overwrite else accum).
    // `rs` is the ring-slot modulus (kDecTcStages for the scalar/dW path; the
    // deeper kDecFwdStages for the fwd/dX cp.async ring). Both are compile-time at
    // every call site so the % folds; the wgmma ISSUE SEQUENCE (ascending k, k=0
    // overwrite / k>0 accumulate) is identical regardless of rs — only the slot
    // the operands were staged into differs, so the fp32 accumulation order is
    // bit-identical (parity / A/A/A preserved).
    auto issue_k = [&] (int g_atoms, int k, int rs) {
        const int sl = k % rs;
        wgs::SmemDesc dB = wgs::make_desc_B_kmajor<N, wgs::kSwizzleNone>(
            smemB + (int64_t)sl * kDecTcSmemB1);
        #pragma unroll
        for (int ai = 0; ai < kIL; ++ai) {
            if (ai >= g_atoms) break;
            wgs::SmemDesc dA = wgs::make_desc_A_kmajor<wgs::kWgmmaAtomM, wgs::kSwizzleNone>(
                smemA + ((int64_t)sl * kIL + ai) * kDecTcSmemA1);
            if (k == 0) wgs::wgmma_m64nNk16_bf16<N, /*ScaleD=*/0, 0, 0>(acc[ai], dA, dB);
            else        wgs::wgmma_m64nNk16_bf16<N, /*ScaleD=*/1, 0, 0>(acc[ai], dA, dB);
        }
    };

    // ── cp.async DOUBLE-BUFFERED RING (campaign RING; unblocked by C1/C1-T).
    //    Selected at COMPILE TIME when BOTH sources are flat-gmem K-major rows
    //    (DecGmemTileSrc*, i.e. the fwd/dX wrappers) and the ring has >= 2 slots.
    //    Same slot ring, same ascending-k issue order, same __syncthreads
    //    choreography as the synchronous path — ONLY the staging TRANSPORT
    //    changes: the per-element 2-byte LDG→reg→STS loop (≈16 dependent
    //    load/store pairs per thread per k-tile, latency exposed in register-
    //    window-limited chunks) becomes 2 fire-and-forget 16-byte
    //    cp.async.cg copies per thread (LDGSTS: gmem→smem direct, no register
    //    round-trip, the WHOLE tile in flight at once) issued while wg0's
    //    wgmmas on the PREVIOUS tile execute in the tensor pipe. The
    //    per-thread cp_async_wait_group<0> + ONE fence.proxy.async + the
    //    existing __syncthreads publish the slot before its first wgmma — the
    //    primitives.cuh contract (issue→commit→wait→barrier→read), i.e. the
    //    silicon-validated tile_pipeline handshake.
    //    NUMERICS: identical bytes land in identical smem offsets in the same
    //    k-order; the wgmma sequence is untouched → results bit-identical to
    //    the synchronous staging (fp64-oracle parity / A/A/A by construction).
    //    dW keeps the synchronous staging (lambda sources: transposed-strided
    //    acts reads = ring blocker (b), needs TMA-with-transpose; out of scope).
    constexpr bool kRingAsync = (S > 1)
        && DecTileSrcIsGmem<SrcA>::value && DecTileSrcIsGmem<SrcB>::value;

    // ── FINE-profiler accumulators (SG_DEC_PROFILE && SG_DEC_PROFILE_FWD_FINE).
    //    Thread-0-only clock64 deltas, summed across ALL M-atom groups + k-steps
    //    of THIS engine call, atomicMax'd into g_dec_prof_fwd_fine ONCE after the
    //    group loop (= the slowest CTA's critical path per sub-phase). Compiled
    //    out entirely (no vars, no clock64, no atomic) unless the pair is set →
    //    byte-identical when off. Only the kRingAsync (fwd/dX) path stamps; the dW
    //    !kRingAsync path never reaches these TICs.
#if defined(SG_DEC_PROFILE) && SG_DEC_PROFILE_FWD_FINE
    unsigned long long _f_issue = 0, _f_wait = 0, _f_wgmma = 0, _f_epi = 0, _f_bar = 0;
    unsigned long long _t0 = 0;
    const bool _prof_t0 = (tid == 0);
    #define SG_FINE_TIC() do { if (_prof_t0) _t0 = clock64(); } while (0)
    #define SG_FINE_ACC(acc) do { if (_prof_t0) (acc) += clock64() - _t0; } while (0)
#else
    #define SG_FINE_TIC() do {} while (0)
    #define SG_FINE_ACC(acc) do {} while (0)
#endif

    // Loop M-atom GROUPS of kIL; each group runs its own k-chain into kIL fragments.
    #pragma unroll 1
    for (int g0 = 0; g0 < m_atoms; g0 += kIL) {
        const int gbase = mbase0 + g0 * wgs::kWgmmaAtomM;
        const int g_atoms = (m_atoms - g0) < kIL ? (m_atoms - g0) : kIL;
        if constexpr (kRingAsync) {
            (void)stage_k;   // the scalar stager is the !ring branch's transport
            // RING DEPTH for the fwd/dX path. PIPE=0 → RS==kDecTcStages (the
            // shipped S=2 double-buffer), and the RS==2 sub-branch below is the
            // ORIGINAL ring VERBATIM → byte-identical PTX. PIPE=1 → RS=2..4 keeps
            // RS-1 cp.async groups in flight (deeper prefetch; the deeper sub-branch).
            constexpr int RS = kDecFwdStages;
            // Stage the group's k-tile `kk` into ring slot kk % RS via 16B cp.async:
            // g_atoms A(64×16) tiles + the ONE shared B(N×16) tile (the M-atom-
            // interleave structure, verbatim). Chunk space is FLAT and fixed
            // (kIL·64·2 A-halves then N·2 B-halves) so every thread's share is
            // static; halves of a dead atom (ai >= g_atoms) are skipped (their
            // smem is never read — issue_k stops at g_atoms). Each 16B half-row
            // (mn, k 8·half..8·half+7) lands at the canonical Major-K interleave
            // offset half·(MN·8)+mn·8 — the same map stage_kmajor_tile writes and
            // the same chunking tile_pipeline.cuh::pipeline_produce_ktile issues.
            auto stage_k_async = [&] (int kk) {
                const int sl = kk % RS;
                const int kb = kk * wgs::kWgmmaAtomK;
                constexpr int kAChunks = kIL * wgs::kWgmmaAtomM * 2;
                constexpr int kChunks  = kAChunks + N * 2;
                #pragma unroll 1
                for (int v = tid; v < kChunks; v += nthreads) {
                    if (v < kAChunks) {
                        const int ai = v / (wgs::kWgmmaAtomM * 2);
                        if (ai < g_atoms) {
                            const int r  = v - ai * (wgs::kWgmmaAtomM * 2);
                            const int mn = r >> 1, half = r & 1;
                            decprim::cp_async_cg_16(
                                smemA + ((int64_t)sl * kIL + ai) * kDecTcSmemA1
                                      + half * (wgs::kWgmmaAtomM * 8) + mn * 8,
                                srcA.chunk16(gbase + ai * wgs::kWgmmaAtomM + mn, kb, half));
                        }
                    } else {
                        const int r  = v - kAChunks;
                        const int mn = r >> 1, half = r & 1;
                        __nv_bfloat16* dst = smemB + (int64_t)sl * kDecTcSmemB1
                                           + half * (N * 8) + mn * 8;
                        if (srcB.row_valid(mn)) {
                            decprim::cp_async_cg_16(dst, srcB.chunk16(mn, kb, half));
                        } else {
                            // PAD row of a ragged N-tile: zero smem, NO gmem read
                            // (the epilogue drops pad cols; zeros keep them inert).
                            uint4 z; z.x = z.y = z.z = z.w = 0u;
                            *reinterpret_cast<uint4*>(dst) = z;
                        }
                    }
                }
                decprim::cp_async_commit();
            };
            if constexpr (kDecFwdPipeEngine) {
                // ══════════════════════════════════════════════════════════════
                //  DESIGN-TOURNAMENT entry (b): PRODUCER/CONSUMER mbarrier ENGINE
                //  (SG_TUNED_DEC_FWD_PIPE=2). The validated tile_pipeline.cuh
                //  choreography, transcribed here over the EXISTING ring tiles
                //  (smemA/smemB), the M-atom-interleave (acc[kIL]), and the
                //  bit-identical issue_k — so the only thing that changes vs the
                //  PIPE=0|1 ring is the THREAD SPLIT and the cross-warpgroup
                //  handoff: WG0 is the CONSUMER (wgmma into acc[]; owns the
                //  epilogue, like every other branch's `in_wg0`), WG1 is the
                //  PURE PRODUCER (cp.async-stages each k-tile + arrives `full`).
                //  The producer runs up to RS stages AHEAD of the consumer, gated
                //  only by the `empty` barrier, so the cp.async DRAIN is hidden
                //  behind the consumer's in-flight wgmmas (DRAIN-HIDING — the
                //  regime the fine profiler proved fwd/dX live in).
                //
                //  PARITY BY CONSTRUCTION: the consumer issues issue_k(g_atoms,k,RS)
                //  in ASCENDING k (k=0 overwrite / k>0 accumulate) into the SAME
                //  acc[ai] fragments as PIPE=0 — the wgmma ISSUE SEQUENCE and the
                //  fp32 accumulation order are bit-identical. The producer stages
                //  the SAME 16B chunks into the SAME canonical Major-K smem offsets
                //  the PIPE=0 stage_k_async writes; only WHEN a tile lands changes
                //  (a LOAD reorder). No ragged/atomic reduce: each acc[ai] is owned
                //  by one consumer thread, summed serially. dW/TP/selftest never
                //  reach here (!kRingAsync). Identical to the unpipelined kchain
                //  because the consumer issues the SAME ss-wgmma sequence in the
                //  SAME order (tile_pipeline.cuh DESIGN gate c).
                //
                //  Barriers: 2·RS words in pipeBars — full[0..RS) then empty[0..RS).
                //  Each expects a FULL WARPGROUP (128) of arrivals per phase (the
                //  tile_pipeline contract: a per-warp elect_one would give 4
                //  arrivals on a count-1 barrier and DEADLOCK the parity wait). The
                //  consumer arrives `empty` (releases a slot); the producer arrives
                //  `full` (publishes a slot) after its own cp_async_wait_group<0> +
                //  one fence.proxy.async (the portable wait_group handoff — the
                //  mbarrier is the cross-warpgroup ready SIGNAL, not the tx counter).
                auto bar_full  = [&] (int s) { return wgs::Mbarrier(pipeBars + s); };
                auto bar_empty = [&] (int s) { return wgs::Mbarrier(pipeBars + RS + s); };
                // Init the 2·RS barriers (thread 0), each expecting 128 arrivals.
                if (tid == 0) {
                    #pragma unroll
                    for (int s = 0; s < RS; ++s) { bar_full(s).init(128); bar_empty(s).init(128); }
                }
                __syncthreads();   // barriers initialized + visible before any arrive

                // Producer-side stager: stage k-tile `kk` into slot kk%RS using ONLY
                // WG1's 128 threads (the producer warpgroup). SAME chunk math /
                // SAME canonical Major-K destination offsets as stage_k_async — only
                // the thread stride is the 128-wide producer group (tid_wg ∈ 0..127),
                // NOT all 256. (stage_k_async strides over 256 threads, so it is the
                // PIPE 0|1 transport; the producer/consumer split needs a 128-wide
                // producer.) Followed by cp_async_commit() (the caller drains+arrives).
                auto produce_ktile = [&] (int kk) {
                    const int sl = kk % RS;
                    const int kb = kk * wgs::kWgmmaAtomK;
                    constexpr int kAChunks = kIL * wgs::kWgmmaAtomM * 2;
                    constexpr int kChunks  = kAChunks + N * 2;
                    #pragma unroll 1
                    for (int v = tid_wg; v < kChunks; v += 128) {
                        if (v < kAChunks) {
                            const int ai = v / (wgs::kWgmmaAtomM * 2);
                            if (ai < g_atoms) {
                                const int r  = v - ai * (wgs::kWgmmaAtomM * 2);
                                const int mn = r >> 1, half = r & 1;
                                decprim::cp_async_cg_16(
                                    smemA + ((int64_t)sl * kIL + ai) * kDecTcSmemA1
                                          + half * (wgs::kWgmmaAtomM * 8) + mn * 8,
                                    srcA.chunk16(gbase + ai * wgs::kWgmmaAtomM + mn, kb, half));
                            }
                        } else {
                            const int r  = v - kAChunks;
                            const int mn = r >> 1, half = r & 1;
                            __nv_bfloat16* dst = smemB + (int64_t)sl * kDecTcSmemB1
                                               + half * (N * 8) + mn * 8;
                            if (srcB.row_valid(mn)) {
                                decprim::cp_async_cg_16(dst, srcB.chunk16(mn, kb, half));
                            } else {
                                uint4 z; z.x = z.y = z.z = z.w = 0u;
                                *reinterpret_cast<uint4*>(dst) = z;
                            }
                        }
                    }
                    decprim::cp_async_commit();
                };

                if (!in_wg0) {
                    // ── PRODUCER (WG1): stage every k-tile in ascending k. With RS
                    //    buffering it runs up to RS stages ahead; for kk>=RS the slot
                    //    was used by kk-RS, so wait on `empty` before overwriting it
                    //    (WAR-safe: the consumer arrives empty only after wgmma kk-RS
                    //    drained). Each stage: produce → drain own cp.async →
                    //    fence.proxy.async → all-128-producer arrive `full`.
                    unsigned emptyp[RS];
                    #pragma unroll
                    for (int s = 0; s < RS; ++s) emptyp[s] = 0u;
                    #pragma unroll 1
                    for (int kk = 0; kk < k_steps; ++kk) {
                        const int sl = kk % RS;
                        if (kk >= RS) { bar_empty(sl).wait(emptyp[sl]); emptyp[sl] ^= 1u; }
                        SG_FINE_TIC(); produce_ktile(kk);                  SG_FINE_ACC(_f_issue);
                        SG_FINE_TIC(); decprim::cp_async_wait_group<0>();  SG_FINE_ACC(_f_wait);
                        wgs::fence_async_proxy();
                        bar_full(sl).arrive();   // publish slot sl (all 128 producers)
                    }
                } else {
                    // ── CONSUMER (WG0): wgmma over each staged k-tile in ascending k
                    //    into acc[] (k=0 overwrite / k>0 accumulate — bit-identical to
                    //    PIPE=0). Wait `full` before issuing, drain the wgmma, then
                    //    all-128-consumer arrive `empty` to release the slot. ONE
                    //    wgmma_fence before the chain (accumulators were just touched).
                    unsigned fullp[RS];
                    #pragma unroll
                    for (int s = 0; s < RS; ++s) fullp[s] = 0u;
                    wgs::wgmma_fence();
                    #pragma unroll 1
                    for (int k = 0; k < k_steps; ++k) {
                        const int sl = k % RS;
                        SG_FINE_TIC(); bar_full(sl).wait(fullp[sl]); fullp[sl] ^= 1u; SG_FINE_ACC(_f_wait);
                        SG_FINE_TIC();
                        issue_k(g_atoms, k, RS);
                        wgs::wgmma_commit_group();
                        wgs::wgmma_wait_group<0>();   // drain wgmma k → slot k%RS free
                        SG_FINE_ACC(_f_wgmma);
                        bar_empty(sl).arrive();       // release slot sl (all 128 consumers)
                    }
                }
                // The shared epilogue below stores acc[] from `in_wg0` = the CONSUMER
                // warpgroup (WG0 owns the fragments here), so it is correct as-is.
            } else if constexpr (RS <= 2) {
                // ── ORIGINAL S=2 DOUBLE-BUFFER RING (verbatim; byte-identical when
                //    PIPE=0). One prefetch tile in flight; full drain + barrier per k.
                //    (RS<=2 also covers the dead RS=1 case — kRingAsync needs S>1, so
                //    this block is only ever reached with RS>=2; the <=2 guard just
                //    keeps the deeper else-branch's cp_async_wait_group<RS-2> from
                //    being instantiated with a negative immediate.)
                // Prologue: tile 0 in flight → drain own copies → ONE async-proxy
                // fence → publish (barrier). Fence choreography then matches the
                // synchronous path (wgmma_fence once, wg0).
                SG_FINE_TIC(); stage_k_async(0);                 SG_FINE_ACC(_f_issue);
                SG_FINE_TIC(); decprim::cp_async_wait_group<0>(); SG_FINE_ACC(_f_wait);
                SG_FINE_TIC(); wgs::fence_async_proxy(); __syncthreads(); SG_FINE_ACC(_f_bar);
                if (in_wg0) wgs::wgmma_fence();
                // Steady state: (1) wg0 issues the slot-k wgmmas (async tensor pipe);
                // (2) ALL threads fire tile k+1's copies into the OTHER slot — the
                // gmem→smem transfers overlap the in-flight wgmmas (slot (k+1)%2 was
                // last READ by wgmma k-1, drained at the end of iteration k-1 →
                // WAR-safe); (3) drain tensor pipe + own copies, fence, ONE barrier.
                #pragma unroll 1
                for (int k = 0; k < k_steps; ++k) {
                    SG_FINE_TIC();
                    if (in_wg0) { issue_k(g_atoms, k, RS); wgs::wgmma_commit_group(); }
                    SG_FINE_ACC(_f_wgmma);
                    SG_FINE_TIC(); if (k + 1 < k_steps) stage_k_async(k + 1); SG_FINE_ACC(_f_issue);
                    SG_FINE_TIC(); if (in_wg0) wgs::wgmma_wait_group<0>();     SG_FINE_ACC(_f_wgmma);
                    SG_FINE_TIC(); decprim::cp_async_wait_group<0>();          SG_FINE_ACC(_f_wait);
                    SG_FINE_TIC(); wgs::fence_async_proxy(); __syncthreads();  SG_FINE_ACC(_f_bar);
                }
            } else {
                // ── DEEPER cp.async RING (RS=3..4; SG_TUNED_DEC_FWD_PIPE=1). Keeps
                //    RS-1 cp.async groups IN FLIGHT so more HBM operand latency is
                //    hidden behind the in-flight wgmmas (the drain-bound lever).
                //    PARITY: the wgmma ISSUE order is IDENTICAL to the RS==2 ring
                //    (ascending k, per-k commit + wgmma_wait_group<0> drain) — only
                //    the cp.async prefetch DISTANCE (and thus WHEN each tile lands)
                //    changes. fp32 accumulation order bit-identical.
                //
                //    WAR safety: a refill for tile k+RS-1 writes slot (k+RS-1)%RS ==
                //    (k-1)%RS, which tile k-1 occupied and wgmma k-1 finished reading
                //    (drained via wgmma_wait_group<0> in iter k-1, ordered before this
                //    iter's top __syncthreads). The refill goes to slot (k-1)%RS while
                //    wgmma k reads slot k%RS (distinct) → no live-slot clobber.
                //
                //    FIFO drain discipline (cp_async_wait_group<N> is a compile-time
                //    immediate, FIFO-ordered): in the STEADY region (a refill happens
                //    this iter ⇒ exactly RS-1 groups in flight) wait_group<RS-2> drains
                //    exactly the oldest (tile k). In the TAIL (refills exhausted) there
                //    is no future copy to overlap, so wait_group<0> (drain all
                //    remaining) is free and keeps the wait depth a valid immediate.
                const int n_pre = (RS - 1) < k_steps ? (RS - 1) : k_steps;
                #pragma unroll 1
                for (int p = 0; p < n_pre; ++p) { SG_FINE_TIC(); stage_k_async(p); SG_FINE_ACC(_f_issue); }
                if (in_wg0) wgs::wgmma_fence();
                #pragma unroll 1
                for (int k = 0; k < k_steps; ++k) {
                    const bool will_refill = (k + (RS - 1) < k_steps);
                    // Land tile k. Steady: keep RS-2 in flight. Tail: drain all.
                    SG_FINE_TIC();
                    if (will_refill) decprim::cp_async_wait_group<RS - 2>();
                    else             decprim::cp_async_wait_group<0>();
                    SG_FINE_ACC(_f_wait);
                    SG_FINE_TIC(); wgs::fence_async_proxy(); __syncthreads(); SG_FINE_ACC(_f_bar);
                    SG_FINE_TIC();
                    if (in_wg0) {
                        issue_k(g_atoms, k, RS);
                        wgs::wgmma_commit_group();
                        wgs::wgmma_wait_group<0>();   // drain wgmma k → slot k%RS free
                    }
                    SG_FINE_ACC(_f_wgmma);
                    // Refill the ring (WAR-safe; see above). No post-wgmma barrier
                    // needed: the refill targets slot (k-1)%RS, freed in iter k-1.
                    SG_FINE_TIC(); if (will_refill) stage_k_async(k + (RS - 1)); SG_FINE_ACC(_f_issue);
                }
            }
        } else {
        // Prologue: stage tile 0 (the group's atoms); make visible; fence ONCE.
        stage_k(gbase, g_atoms, 0);
        __syncthreads();
        if (in_wg0) wgs::wgmma_fence();
        // Steady state (S=2 single group in flight): issue the g_atoms wgmmas for
        // slot k%S (async, overlapping in the tensor pipe), THEN stage tile k+1 into
        // the OTHER slot (HBM loads overlap the MMAs), THEN wait_group<0> + sync. S=1
        // collapses to staging into the single slot AFTER the wait (serial, exact).
        #pragma unroll 1
        for (int k = 0; k < k_steps; ++k) {
            if (in_wg0) { issue_k(g_atoms, k, S); wgs::wgmma_commit_group(); }
            if (S > 1) {
                if (k + 1 < k_steps) stage_k(gbase, g_atoms, k + 1);
                if (in_wg0) wgs::wgmma_wait_group<0>();
                __syncthreads();
            } else {
                if (in_wg0) wgs::wgmma_wait_group<0>();
                __syncthreads();
                if (k + 1 < k_steps) { stage_k(gbase, g_atoms, k + 1); __syncthreads(); }
            }
        }
        }
        // Epilogue: warpgroup 0 owns the fp32 fragments; decode + emit (real cols).
        // All reads happen AFTER the final wait_group<0> — no overlap with any wgmma.
        // EPI fine-stamp only on the ring (fwd/dX) path (constexpr-guarded so the dW
        // !kRingAsync path emits no stamp); the store accessor cost is the EPI bucket.
        if constexpr (kRingAsync) SG_FINE_TIC();
        if (in_wg0) {
            #pragma unroll
            for (int ai = 0; ai < kIL; ++ai) {
                if (ai >= g_atoms) break;
                const int mbase = gbase + ai * wgs::kWgmmaAtomM;
                #pragma unroll
                for (int i = 0; i < wgs::WgmmaAccum<N>::kRegs; ++i) {
                    int row, col;
                    wgs::wgmma_frag_decode(tid_wg, i, N, row, col);
                    if (col < n_real) out(mbase + row, col, acc[ai].c[i]);
                }
            }
        }
        if constexpr (kRingAsync) SG_FINE_ACC(_f_epi);
        __syncthreads();
    }
    // Stamp the fine sub-phases ONCE (thread-0-only, atomicMax across CTAs). Only
    // the ring (fwd/dX) path accumulated nonzero deltas (the dW path never TIC'd);
    // guard on kRingAsync so the dW engine emits no atomic. prof_phase: 0 = fwd
    // ring, 1 = dX ring (the wrapper passed it); -1 (defensive) → phase 0.
#if defined(SG_DEC_PROFILE) && SG_DEC_PROFILE_FWD_FINE
    if constexpr (kRingAsync) {
        if (_prof_t0) {
            const int ph = (prof_phase == 1) ? 1 : 0;
            dec_fwd_fine_acc(ph, kFineIssue,   _f_issue);
            dec_fwd_fine_acc(ph, kFineWait,    _f_wait);
            dec_fwd_fine_acc(ph, kFineWgmma,   _f_wgmma);
            dec_fwd_fine_acc(ph, kFineEpi,     _f_epi);
            dec_fwd_fine_acc(ph, kFineBarrier, _f_bar);
        }
    }
#endif
#undef SG_FINE_TIC
#undef SG_FINE_ACC
#else
    (void)mbase0; (void)m_atoms; (void)n_real; (void)k_steps;
    (void)srcA; (void)srcB; (void)out; (void)smemA; (void)smemB;
    (void)prof_phase; (void)pipeBars;
#endif
}

// ════════════════════════════════════════════════════════════════════════
//  THIN ORIENTATION WRAPPERS over tc_gemm_block_unpipelined. These reproduce
//  the THREE accessor patterns the engine is silicon-validated on
//  (decoder_tc_selftest.cu / test_decoder_tc.py 13/13): fwd (Y=X·Wᵀ, no
//  transpose), dX (dX=dY·W, W transposed-staged), dW (dW=dYᵀ·X, BOTH
//  transposed-staged, K=T). The driver calls THESE — it never re-derives the
//  staging (the no-suppression / reuse-the-validated-unit discipline).
//
//  All operands are HBM bf16 row-major. The caller passes one A(64×16) + one
//  B(Nmax×16) smem staging pair (the engine reuses them across k-steps). N is
//  the compile-time wgmma atom width (128 for in/out/ff dX-N=d; the fwd loops
//  N-tiles internally). Accumulation is fp32; output written via the accessor.
// ════════════════════════════════════════════════════════════════════════

// (fwd) Y[M,Nout] = X[M,Kin] @ W[Nout,Kin]ᵀ.  Tiles N over [0,Nout) in width-N
// atoms (Nout∈{d=128, 3d=384, dff=512}). M = kTileM (kAtomsM stacked atoms).
// Y is written row-major [M, Nout] at base `Yout` with row stride Nout.
// NOTE on weights: W is the PRE-STAGED bf16 cache (C1, dec_wbf_bind) — the
// fp32→bf16 conversion happened ONCE per step in dectc_wbf_convert with the
// identical deterministic rounding the old on-read accessor performed, so the
// staged operand VALUES are bit-identical. Both operands are flat-gmem K-major
// → the engine's cp.async double-buffered ring stages them (RING).
// `pipeBars` (default nullptr): the PIPE=2 producer/consumer mbarrier-words base
// (DecTcSmem::pipe_bars). Forwarded to the engine; dereferenced ONLY in its PIPE=2
// sub-branch (compiled out for PIPE 0|1 → the nullptr default is unreferenced and
// dropped → byte-identical). The fwd/dX driver threads sm.pipe_bars under
// #if SG_TUNED_DEC_FWD_PIPE == 2; on the shipped build it passes nothing (default).
template <int N>
__device__ __forceinline__ void dectc_gemm_fwd(
        const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
        __nv_bfloat16* __restrict__ Yout, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB, unsigned long long* pipeBars = nullptr) {
    const int k_steps = Kin / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        // A: token rows of X (bf16 acts), K-contiguous. B: rows n0.. of the
        // PRE-STAGED bf16 cache (C1) -- a pure bf16 copy, values bit-identical
        // to the on-read path. Both flat-gmem K-major -> the engine selects the
        // cp.async double-buffered ring (RING); same accessor semantics as the
        // lambdas these replace (incl. the `nn < Nout ? .. : 0` pad guard).
        DecGmemTileSrcA srcA{X, Kin};
        DecGmemTileSrcB srcB{W + (int64_t)n0 * Kin, Kin, Nout - n0};
        auto out  = [&] (int m, int n, float v) {
            Yout[(int64_t)m * Nout + n0 + n] = __float2bfloat16(v); };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB, pipeBars SG_DEC_FINE_FWD);
    }
}

// Same as dectc_gemm_fwd but emits the fp32 result (no bf16 round) — for the
// few fwd outputs consumed by fp32 elementwise stages directly. Writes [M,Nout]
// fp32 at `Yf32`.
template <int N>
__device__ __forceinline__ void dectc_gemm_fwd_f32(
        const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
        float* __restrict__ Yf32, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB, unsigned long long* pipeBars = nullptr) {
    const int k_steps = Kin / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        // Same flat-gmem K-major sources as dectc_gemm_fwd (C1 cache B operand;
        // RING-staged); only the fp32 output accessor differs.
        DecGmemTileSrcA srcA{X, Kin};
        DecGmemTileSrcB srcB{W + (int64_t)n0 * Kin, Kin, Nout - n0};
        auto out  = [&] (int m, int n, float v) { Yf32[(int64_t)m * Nout + n0 + n] = v; };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB, pipeBars SG_DEC_FINE_FWD);
    }
}

// (dX) dX[M,Kin] = dY[M,Nout] @ W[Nout,Kin].  N(wgmma) = Kin (the in_dim, tiled
// by width N). K = Nout (the contracted out_dim). The B operand is staged from
// the TRANSPOSED cache section (C1-T, dec_wbf_bind's *_wT): srcB(n=kin, k=out)
// = WT[kin·Nout + out] == W[out·Kin + kin] BIT-IDENTICALLY (the transposed copy
// holds the same deterministically-rounded bf16 values) — but K-CONTIGUOUS, so
// the cp.async ring streams the dX weight operand too (the old row-major W read
// was k-strided: a 2-byte gather no 4/8/16B async copy can express). Writes
// fp32 dX [M,Kin] (LN/elementwise bwd consume it fp32).
template <int N>
__device__ __forceinline__ void dectc_gemm_dx_f32(
        const __nv_bfloat16* __restrict__ dY, const __nv_bfloat16* __restrict__ WT,
        float* __restrict__ dXf32, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB, unsigned long long* pipeBars = nullptr) {
    const int k_steps = Nout / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Kin; n0 += N) {
        const int n_real = (Kin - n0) < N ? (Kin - n0) : N;
        DecGmemTileSrcA srcA{dY, Nout};
        DecGmemTileSrcB srcB{WT + (int64_t)n0 * Nout, Nout, Kin - n0};
        auto out  = [&] (int m, int n, float v) { dXf32[(int64_t)m * Kin + n0 + n] = v; };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB, pipeBars SG_DEC_FINE_DX);
    }
}

// ── fp32-W OVERLOADS (TP-path compatibility; pre-C1 convert-on-read bodies).
//    The C1 weight pre-stage scoped the megakernel's GEMM B operand to the bf16
//    cache, but the TENSOR-PARALLEL layer (csrc/fused/sm_90/tp_layer.cuh and its
//    JIT test binding tests/hw/tp_loopback_binding.cu) stages PER-RANK fp32
//    weight SHARDS that live outside the megakernel workspace — no C1 cache
//    exists there. These overloads carry the EXACT pre-C1 accessor bodies
//    (`__float2bfloat16(W[...])` on read — deterministic, bit-identical to what
//    the TP path always consumed), selected by the fp32 pointer type. The
//    megakernel never calls them (its W is the bf16 cache). ──
template <int N>
__device__ __forceinline__ void dectc_gemm_fwd_f32(
        const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
        float* __restrict__ Yf32, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    const int k_steps = Kin / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return X[(int64_t)m * Kin + k]; };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return nn < Nout ? __float2bfloat16(W[(int64_t)nn * Kin + k]) : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) { Yf32[(int64_t)m * Nout + n0 + n] = v; };
        // fp32-W (TP) lambda sources → !kRingAsync (no stamp; never reaches the
        // PIPE=2 engine) → pipeBars=nullptr; tag kept for intent.
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB, /*pipeBars=*/nullptr SG_DEC_FINE_FWD);
    }
}
template <int N>
__device__ __forceinline__ void dectc_gemm_dx_f32(
        const __nv_bfloat16* __restrict__ dY, const float* __restrict__ W,
        float* __restrict__ dXf32, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    const int k_steps = Nout / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Kin; n0 += N) {
        const int n_real = (Kin - n0) < N ? (Kin - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return dY[(int64_t)m * Nout + k]; };
        // B[n=kin, k=out] = W[out, kin]  (transposed read; fp32 → bf16).
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return nn < Kin ? __float2bfloat16(W[(int64_t)k * Kin + nn]) : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) { dXf32[(int64_t)m * Kin + n0 + n] = v; };
        // fp32-W (TP) lambda sources → !kRingAsync (no stamp; never reaches the
        // PIPE=2 engine) → pipeBars=nullptr; tag kept for intent.
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB, /*pipeBars=*/nullptr SG_DEC_FINE_DX);
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Per-CTA TILE SCRATCH (HBM). One tile's forward intermediates the backward
//  reads, reused across the tiles a CTA grid-strides over (a CTA finishes a
//  tile's fwd+bwd before the next tile, so one slab per CTA suffices). Sized
//  for kTileM rows. The X-inputs / dY-adjoints that OTHER CTAs read in P2's dW
//  do NOT live here — they live in the full-T DecActs (cross-CTA). This holds
//  only the within-tile fwd state + the running adjoint so the backward needs
//  NO recompute.
//
//  DEDICATED, NON-ALIASED buffers (the TC test sizes its own workspace; HBM is
//  NOT scarce, so hand-managed aliasing — the stride/alias bug class that bit
//  the scalar path at model_stages_decoder.cuh:619-625, which the CPU mirror
//  does NOT cover for tile-batched reuse — is avoided entirely). Each fp32
//  intermediate gets its own slot.
// ════════════════════════════════════════════════════════════════════════

// nSamp samples per tile, H*S*S attention entries each; V logits each.
constexpr int kNSampPerTile  = kTileM / dec::kSeq;
constexpr int kAttnPerTile   = kNSampPerTile * dec::kHeads * dec::kSeq * dec::kSeq;
constexpr int kLogitsPerTile = kNSampPerTile * dec::kVocab;

// CRITICAL: the forward runs ALL layers, THEN the backward runs ALL layers (NOT
// interleaved per-layer). So every forward intermediate the backward reads PER
// LAYER must be stored PER LAYER — a single-buffered cache holds only the LAST
// layer's values and the earlier layers' backward reads garbage (the "forward
// exact, layer-0 grads wrong, error compounds backward" bug). qkv / ff0pre /
// attn / n1·n2 LN caches are therefore [kLayers]-indexed. fnx/fni (final norm,
// one instance) + the transient dh/x1/finalin/logits/work/work2/dsc stay single.
struct DecTileScratch {
    __nv_bfloat16* qkv[dec::kLayers];     // [kTileM, 3d]  per layer
    __nv_bfloat16* ff0pre[dec::kLayers];  // [kTileM, dff] per layer
    float* attn[dec::kLayers];            // [kAttnPerTile] per layer
    float* n1x[dec::kLayers]; float* n1i[dec::kLayers];
    float* n2x[dec::kLayers]; float* n2i[dec::kLayers];
    float* dsc;             // [kAttnPerTile] attention dscores (transient, bwd-only)
    float* fnx; float* fni; // final-norm LN caches (single)
    float* dh;              // [kTileM, d]    running adjoint wrt block output
    float* x1;              // [kTileM, d]    n1 output (fp32, residual base for r2)
    float* finalin;         // [kTileM, d]    last-layer n2 output (fp32, head input)
    float* logits;          // [kLogitsPerTile] per-sample last-pos logits (fp32)
    float* work;            // [kTileM, dff]  GEMM output / general fp32 scratch
    float* work2;           // [kTileM, dff]  second fp32 scratch (bwd dx1/dqkv)
};

// Bytes one CTA's scratch occupies (for host sizing of the workspace tail).
__host__ __device__ __forceinline__ int64_t dec_tile_scratch_bf16_count() {
    // (qkv + ff0pre) per layer.
    return (int64_t)dec::kLayers * ((int64_t)kTileM * 3 * dec::kD + (int64_t)kTileM * dec::kDff);
}
__host__ __device__ __forceinline__ int64_t dec_tile_scratch_f32_count() {
    return (int64_t)dec::kLayers * (                     // per-layer:
             (int64_t)kAttnPerTile                       //   attn
           + 2 * ((int64_t)kTileM * dec::kD + kTileM))    //   n1+n2 xhat+inv
         + (int64_t)kAttnPerTile                          // dsc (single)
         + ((int64_t)kTileM * dec::kD + kTileM)           // fn xhat+inv (single)
         + (int64_t)kTileM * dec::kD                      // dh
         + (int64_t)kTileM * dec::kD                      // x1
         + (int64_t)kTileM * dec::kD                      // finalin
         + (int64_t)kLogitsPerTile                        // logits
         + 2 * (int64_t)kTileM * dec::kDff;               // work + work2
}
__host__ __device__ __forceinline__ int64_t dec_tile_scratch_total_f32() {
    return (dec_tile_scratch_bf16_count() + 1) / 2 + dec_tile_scratch_f32_count();
}

__device__ __forceinline__ DecTileScratch dec_tile_scratch_bind(float* slab) {
    DecTileScratch s;
    __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(slab);
    int64_t bo = 0;
    for (int li = 0; li < dec::kLayers; ++li) { s.qkv[li]    = b + bo; bo += (int64_t)kTileM * 3 * dec::kD; }
    for (int li = 0; li < dec::kLayers; ++li) { s.ff0pre[li] = b + bo; bo += (int64_t)kTileM * dec::kDff; }
    float* f = slab + (dec_tile_scratch_bf16_count() + 1) / 2;
    int64_t fo = 0;
    for (int li = 0; li < dec::kLayers; ++li) { s.attn[li] = f + fo; fo += kAttnPerTile; }
    for (int li = 0; li < dec::kLayers; ++li) { s.n1x[li] = f + fo; fo += (int64_t)kTileM * dec::kD; s.n1i[li] = f + fo; fo += kTileM; }
    for (int li = 0; li < dec::kLayers; ++li) { s.n2x[li] = f + fo; fo += (int64_t)kTileM * dec::kD; s.n2i[li] = f + fo; fo += kTileM; }
    s.dsc  = f + fo; fo += kAttnPerTile;
    s.fnx  = f + fo; fo += (int64_t)kTileM * dec::kD;
    s.fni  = f + fo; fo += kTileM;
    s.dh   = f + fo; fo += (int64_t)kTileM * dec::kD;
    s.x1   = f + fo; fo += (int64_t)kTileM * dec::kD;
    s.finalin = f + fo; fo += (int64_t)kTileM * dec::kD;
    s.logits  = f + fo; fo += kLogitsPerTile;
    s.work    = f + fo; fo += (int64_t)kTileM * dec::kDff;
    s.work2   = f + fo; fo += (int64_t)kTileM * dec::kDff;
    return s;
}

// ════════════════════════════════════════════════════════════════════════
//  TILE-AWARE SCALAR ELEMENTWISE STAGES (fp32, CTA-cooperative over kTileM
//  rows). These mirror the scalar oracle's per-row math (model_stages_decoder
//  .cuh) but operate on a whole tile of `nrows` rows at once, reading/writing
//  HBM [rows×width] buffers. Reductions reuse the validated dec_block_sum /
//  dec_block_max helpers (whole-block, looped per row — LN/softmax are ≪1% of
//  FLOPs, so the sequential row loop is not a bottleneck). `red` is a 256-float
//  smem reduction slot (from the engine's smem arena).
// ════════════════════════════════════════════════════════════════════════

// LayerNorm fwd over the last dim d, for `nrows` rows. y, xhat are fp32 HBM
// [rows×d]; inv is fp32 HBM [rows]. gamma/beta are fp32 [d] (params). Caches
// xhat+inv for the bwd (identical to dec_layernorm_fwd but tiled).
__device__ __forceinline__ void dectc_ln_fwd_tile(
        const float* __restrict__ x, const float* __restrict__ gamma,
        const float* __restrict__ beta, int nrows,
        float* __restrict__ y, float* __restrict__ xhat, float* __restrict__ inv,
        float* red) {
    for (int s = 0; s < nrows; ++s) {
        const float* xr = x + (int64_t)s * dec::kD;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) sum += xr[j];
        float mean = dec_block_sum(sum, red) / (float)dec::kD;
        float vs = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) { float c = xr[j] - mean; vs += c * c; }
        float var = dec_block_sum(vs, red) / (float)dec::kD;
        float iv = rsqrtf(var + dec::kLnEps);
        if (threadIdx.x == 0) inv[s] = iv;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float xh = (xr[j] - mean) * iv;
            xhat[(int64_t)s * dec::kD + j] = xh;
            y[(int64_t)s * dec::kD + j] = xh * gamma[j] + beta[j];
        }
        __syncthreads();
    }
}

// LayerNorm bwd for `nrows` rows: dy [rows×d] fp32, cached xhat/inv → dx [rows×d]
// fp32; ACCUMULATES dgamma/dbeta (summed over the tile's rows) into a per-CTA
// LN-vec partial slot gw/gb [d] (plain += : single owner thread per feature j
// across rows, deterministic — same rule as the scalar dec_layernorm_bwd).
__device__ __forceinline__ void dectc_ln_bwd_tile(
        const float* __restrict__ dy, const float* __restrict__ xhat,
        const float* __restrict__ inv, const float* __restrict__ gamma, int nrows,
        float* __restrict__ dx, float* __restrict__ gw, float* __restrict__ gb,
        float* red) {
    for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
        float dgw = 0.0f, dgb = 0.0f;
        for (int s = 0; s < nrows; ++s) {
            float d = dy[(int64_t)s * dec::kD + j];
            dgb += d; dgw += d * xhat[(int64_t)s * dec::kD + j];
        }
        gw[j] += dgw; gb[j] += dgb;
    }
    for (int s = 0; s < nrows; ++s) {
        const float* dyr = dy + (int64_t)s * dec::kD;
        const float* xhr = xhat + (int64_t)s * dec::kD;
        float sda = 0.0f, sdax = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dxhat = dyr[j] * gamma[j]; sda += dxhat; sdax += dxhat * xhr[j];
        }
        sda = dec_block_sum(sda, red);
        sdax = dec_block_sum(sdax, red);
        float iv = inv[s];
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dxhat = dyr[j] * gamma[j];
            dx[(int64_t)s * dec::kD + j] = iv * (dxhat - (sda + xhr[j] * sdax) / (float)dec::kD);
        }
        __syncthreads();
    }
}

// Per-sample causal self-attention FORWARD over a tile. qkv is bf16 HBM
// [rows×3d] (q|k|v). Writes ctx fp32 HBM [rows×d] and attn weights fp32 to
// `attn_w` [nSamp×H×S×S]. Each (sample,head,qpos) row is owned by one thread —
// identical math to dec_forward_sample's attention block, looped over samples.
__device__ __forceinline__ void dectc_attn_fwd_tile(
        const __nv_bfloat16* __restrict__ qkv, int nrows,
        float* __restrict__ ctx, float* __restrict__ attn_w) {
    const int nsamp = nrows / dec::kSeq;
    const float scale = dec::attn_scale();
    const int rows_per = nsamp * dec::kHeads * dec::kSeq;   // (sample,head,qpos)
    for (int r = threadIdx.x; r < rows_per; r += blockDim.x) {
        const int si = r / (dec::kHeads * dec::kSeq);
        const int rem = r % (dec::kHeads * dec::kSeq);
        const int hh = rem / dec::kSeq, qi = rem % dec::kSeq;
        const int qoff = hh * dec::kDhead;
        const int rbase = si * dec::kSeq;        // first row of this sample
        const __nv_bfloat16* qrow = qkv + (int64_t)(rbase + qi) * 3 * dec::kD + qoff;
        float maxs = -CUDART_INF_F; float sc[dec::kSeq];
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            if (kj > qi) { sc[kj] = -CUDART_INF_F; continue; }
            const __nv_bfloat16* krow = qkv + (int64_t)(rbase + kj) * 3 * dec::kD + dec::kD + qoff;
            float dot = 0.0f;
            #pragma unroll
            for (int t = 0; t < dec::kDhead; ++t)
                dot += __bfloat162float(qrow[t]) * __bfloat162float(krow[t]);
            sc[kj] = dot * scale; maxs = fmaxf(maxs, sc[kj]);
        }
        float denom = 0.0f;
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            float e = (kj <= qi) ? __expf(sc[kj] - maxs) : 0.0f; sc[kj] = e; denom += e;
        }
        float invd = 1.0f / denom;
        float* aw = attn_w + ((int64_t)(si * dec::kHeads + hh) * dec::kSeq + qi) * dec::kSeq;
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) aw[kj] = sc[kj] * invd;
        #pragma unroll
        for (int t = 0; t < dec::kDhead; ++t) {
            float acc = 0.0f;
            #pragma unroll
            for (int kj = 0; kj <= qi; ++kj) {
                float vv = __bfloat162float(qkv[(int64_t)(rbase + kj) * 3 * dec::kD + 2 * dec::kD + qoff + t]);
                acc += aw[kj] * vv;
            }
            ctx[(int64_t)(rbase + qi) * dec::kD + qoff + t] = acc;
        }
    }
    __syncthreads();
}

// ────────────────────────────────────────────────────────────────────────────
//  TP HEAD-LOCALIZED attention FORWARD (tp_kernel.md §6/§12 NOTE). IDENTICAL
//  per-head math to dectc_attn_fwd_tile, but over the rank's LOCAL head shard:
//  on the kTPComm column-parallel in_proj path the qkv buffer is the rank's
//  dense [q_own | k_own | v_own] concatenation (tp_layer.cuh §"QKV 3-BLOCK
//  SHARD"), so the per-row stride is 3*Dloc and the q|k|v blocks start at
//  0 / Dloc / 2*Dloc, with Hloc = kHeads/P whole heads of width kDhead each
//  (Dloc == Hloc*kDhead == kD/P). `ctx` is the rank's own [nrows, Dloc] context
//  (recombined to full width by the row-parallel out_proj all-reduce, reduce
//  point ①). At TP=1 (Hloc==kHeads, Dloc==kD) every literal here equals the
//  SingleGPU dectc_attn_fwd_tile EXACTLY — but this function is ONLY ever
//  instantiated under `if constexpr (Par::kTPComm)`, so the default path keeps
//  calling the byte-identical dectc_attn_fwd_tile above. attn_w is indexed with
//  Hloc heads (self-consistent fwd-write / bwd-read).
//
//  Hloc/Dloc are passed in (the caller derives them from dec::kHeads/Par::kTP);
//  if kHeads%P != 0 the caller passes Hloc==0 ⇒ this loop is a NO-OP (the ColQKV
//  per-head split is geometrically impossible for that {layout,P} — head-whole
//  is the documented precondition, tp_layer.cuh:158-161). The {1,8} dispatch is
//  the head-divisible allow-list.
__device__ __forceinline__ void dectc_attn_fwd_tile_tp(
        const __nv_bfloat16* __restrict__ qkv, int nrows,
        float* __restrict__ ctx, float* __restrict__ attn_w,
        int Hloc, int Dloc) {
    const int nsamp = nrows / dec::kSeq;
    const float scale = dec::attn_scale();
    const int stride3 = 3 * Dloc;                          // local per-row qkv stride
    const int rows_per = nsamp * Hloc * dec::kSeq;         // (sample,local-head,qpos)
    for (int r = threadIdx.x; r < rows_per; r += blockDim.x) {
        const int si = r / (Hloc * dec::kSeq);
        const int rem = r % (Hloc * dec::kSeq);
        const int hh = rem / dec::kSeq, qi = rem % dec::kSeq;
        const int qoff = hh * dec::kDhead;
        const int rbase = si * dec::kSeq;        // first row of this sample
        const __nv_bfloat16* qrow = qkv + (int64_t)(rbase + qi) * stride3 + qoff;
        float maxs = -CUDART_INF_F; float sc[dec::kSeq];
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            if (kj > qi) { sc[kj] = -CUDART_INF_F; continue; }
            const __nv_bfloat16* krow = qkv + (int64_t)(rbase + kj) * stride3 + Dloc + qoff;
            float dot = 0.0f;
            #pragma unroll
            for (int t = 0; t < dec::kDhead; ++t)
                dot += __bfloat162float(qrow[t]) * __bfloat162float(krow[t]);
            sc[kj] = dot * scale; maxs = fmaxf(maxs, sc[kj]);
        }
        float denom = 0.0f;
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            float e = (kj <= qi) ? __expf(sc[kj] - maxs) : 0.0f; sc[kj] = e; denom += e;
        }
        float invd = 1.0f / denom;
        float* aw = attn_w + ((int64_t)(si * Hloc + hh) * dec::kSeq + qi) * dec::kSeq;
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) aw[kj] = sc[kj] * invd;
        #pragma unroll
        for (int t = 0; t < dec::kDhead; ++t) {
            float acc = 0.0f;
            #pragma unroll
            for (int kj = 0; kj <= qi; ++kj) {
                float vv = __bfloat162float(qkv[(int64_t)(rbase + kj) * stride3 + 2 * Dloc + qoff + t]);
                acc += aw[kj] * vv;
            }
            ctx[(int64_t)(rbase + qi) * Dloc + qoff + t] = acc;
        }
    }
    __syncthreads();
}

// Global SAMPLE index of the si-th sample in a tile whose first token row is g0.
__device__ __forceinline__ int si_global(int g0, int si) { return g0 / dec::kSeq + si; }

// ════════════════════════════════════════════════════════════════════════
//  FORWARD over one TOKEN TILE (nrows = nsamp samples × kSeq positions), global
//  token rows [g0, g0+nrows). Tile-batched: the four per-layer linears
//  (in_proj/out_proj/ff0/ff2) run on wgmma (M=nrows, N-tiled); attention/LN/
//  GELU are scalar fp32 over the tile; head/CE are scalar per-sample (M=nsamp<
//  64). Writes the DecActs X-inputs (bf16 dW operands), the per-CTA tile scratch
//  (qkv/ff0pre/attn/LN caches/x1/finalin/logits the bwd needs), and returns the
//  tile's summed NLL (thread 0 holds it). `tok_ids`/`tgt_ids` are HBM int32.
//
//  DATAFLOW (DecActs X-regions = bf16 dW operands AND inter-stage operands;
//  dedicated fp32 scratch for residuals/LN; weights convert fp32→bf16 on read):
//    X_in[li]  := layer input (bf16)           [embedding for li=0]
//    qkv(bf16) := X_in @ in_w^T + in_b   (in_b folded fp32 → re-round bf16)
//    ctx       := attn(qkv) → X_ctx[li] (bf16, out_proj input + dW operand)
//    a(work)   := X_ctx @ out_w^T ; r1=X_in+a+out_b (work); n1(r1)→x1(fp32)→X_x1[li]
//    ff0(work) := X_x1 @ ff0_w^T ; (ff0+ff0_b)→ff0pre(bf16); gelu(ff0+ff0_b)→X_gact[li](bf16)
//    ff2(work) := X_gact @ ff2_w^T ; r2=x1+ff2+ff2_b (work); n2(r2)→X_in[li+1]
//                 (last layer: n2→finalin fp32, the head input)
//  BIASES (in/out/ff0/ff2) are folded in fp32 at these points (the oracle adds
//  them in fp32 after the bf16 matmul); LN β + head out_b were already applied.
// ════════════════════════════════════════════════════════════════════════
__device__ float dectc_forward_tile(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tok_ids,
        const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        unsigned long long* pipeBars = nullptr) {
    const int nsamp = nrows / dec::kSeq;
    // ── Embedding: X_in[0][r] = tok[token_id[g0+r]] + pos[(g0+r)%S]. bf16. ──
    for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
        const int r = idx / dec::kD, j = idx % dec::kD;
        const int g = g0 + r;
        const int tid = tok_ids[g];
        const int sp = g % dec::kSeq;            // position within the sample
        float v = w.tok[(int64_t)tid * dec::kD + j] + w.pos[(int64_t)sp * dec::kD + j];
        acts.X_in[0][(int64_t)g * dec::kD + j] = __float2bfloat16(v);
    }
    __syncthreads();

    for (int li = 0; li < dec::kLayers; ++li) {
        const DecWeights::Layer& L = w.layer[li];
        const __nv_bfloat16* Xin = acts.X_in[li] + (int64_t)g0 * dec::kD;        // [nrows,d]
        // qkv = Xin @ in_w^T + in_b   (N=3d, K=d). bf16 → scratch.qkv[li].
        dectc_gemm_fwd<SG_TUNED_TILE_N>(Xin, wb.in_w[li], sc.qkv[li], dec::kD, 3 * dec::kD, sA, sB, pipeBars);
        __syncthreads();
        // add in_b (the fwd GEMM did W only; bias folded in scalar here for qkv —
        // matches the bf16-faithful oracle qkv = bf(x_in @ bf(in_w)^T + in_b)).
        for (int idx = threadIdx.x; idx < nrows * 3 * dec::kD; idx += blockDim.x) {
            const int j = idx % (3 * dec::kD);
            float v = __bfloat162float(sc.qkv[li][idx]) + L.in_b[j];
            sc.qkv[li][idx] = __float2bfloat16(v);
        }
        __syncthreads();
        // attention → ctx (work fp32) + attn[li] weights.
        dectc_attn_fwd_tile(sc.qkv[li], nrows, sc.work, sc.attn[li]);
        // ctx bf16 → X_ctx[li] (out_proj input + its dW operand).
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            const int r = idx / dec::kD, j = idx % dec::kD;
            acts.X_ctx[li][(int64_t)(g0 + r) * dec::kD + j] = __float2bfloat16(sc.work[(int64_t)r * dec::kD + j]);
        }
        __syncthreads();
        // a = X_ctx @ out_w^T (+ out_b)  (N=d, K=d). fp32 → work.
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_ctx[li] + (int64_t)g0 * dec::kD, wb.out_w[li],
                                            sc.work, dec::kD, dec::kD, sA, sB, pipeBars);
        __syncthreads();
        // r1 = Xin + a + out_b → work (fp32). out_b folded here (the GEMM did W only)
        // — matches the oracle a = ctx_b @ out_w^T + out_b kept fp32 through r1.
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            const int r = idx / dec::kD, j = idx % dec::kD;
            sc.work[(int64_t)r * dec::kD + j] += __bfloat162float(Xin[(int64_t)r * dec::kD + j]) + L.out_b[j];
        }
        __syncthreads();
        // n1(r1) → x1 (fp32) + caches[li]; then bf16 → X_x1[li] (ff0 input + dW operand).
        dectc_ln_fwd_tile(sc.work, L.n1_w, L.n1_b, nrows, sc.x1, sc.n1x[li], sc.n1i[li], red);
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            const int r = idx / dec::kD, j = idx % dec::kD;
            acts.X_x1[li][(int64_t)(g0 + r) * dec::kD + j] = __float2bfloat16(sc.x1[(int64_t)r * dec::kD + j]);
        }
        __syncthreads();
        // ff0 = X_x1 @ ff0_w^T (+ ff0_b)  (N=dff, K=d). fp32 → work; (pre+b) bf16 →
        // ff0pre; gelu(pre+b) → X_gact[li] (bf16, ff2 input + dW operand). ff0_b folded
        // into pre (fp32) — matches the oracle ff0pre=bf(ff0+b), gact=bf(gelu(ff0+b)).
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_x1[li] + (int64_t)g0 * dec::kD, wb.ff0_w[li],
                                            sc.work, dec::kD, dec::kDff, sA, sB, pipeBars);
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * dec::kDff; idx += blockDim.x) {
            const int r = idx / dec::kDff, j = idx % dec::kDff;
            float pre = sc.work[(int64_t)r * dec::kDff + j] + L.ff0_b[j];
            sc.ff0pre[li][(int64_t)r * dec::kDff + j] = __float2bfloat16(pre);
            acts.X_gact[li][(int64_t)(g0 + r) * dec::kDff + j] = __float2bfloat16(dec_gelu(pre));
        }
        __syncthreads();
        // ff2 = X_gact @ ff2_w^T (+ ff2_b) (N=d, K=dff). fp32 → work.
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_gact[li] + (int64_t)g0 * dec::kDff, wb.ff2_w[li],
                                            sc.work, dec::kDff, dec::kD, sA, sB, pipeBars);
        __syncthreads();
        // r2 = x1 + ff2 + ff2_b → work (fp32). x1 lives in the dedicated fp32 buffer
        // (no bf16 round). ff2_b folded here — matches the oracle r2 = x1 + (ff2 + ff2_b).
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            const int r = idx / dec::kD, j = idx % dec::kD;
            sc.work[(int64_t)r * dec::kD + j] += sc.x1[(int64_t)r * dec::kD + j] + L.ff2_b[j];
        }
        __syncthreads();
        if (li + 1 < dec::kLayers) {
            // n2(r2) → finalin (fp32 reused) + n2 caches[li]; bf16 → X_in[li+1].
            dectc_ln_fwd_tile(sc.work, L.n2_w, L.n2_b, nrows, sc.finalin, sc.n2x[li], sc.n2i[li], red);
            for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
                const int r = idx / dec::kD, j = idx % dec::kD;
                acts.X_in[li + 1][(int64_t)(g0 + r) * dec::kD + j] = __float2bfloat16(sc.finalin[(int64_t)r * dec::kD + j]);
            }
            __syncthreads();
        } else {
            // last layer: n2(r2) → finalin (fp32, all positions; head reads last pos) + n2 caches[li].
            dectc_ln_fwd_tile(sc.work, L.n2_w, L.n2_b, nrows, sc.finalin, sc.n2x[li], sc.n2i[li], red);
        }
    }

    // ── Final norm + head + CE, scalar PER-SAMPLE on the LAST position only.
    //    finalin holds the last-layer n2 output [nrows,d] fp32. ──
    float nll_acc = 0.0f;
    for (int si = 0; si < nsamp; ++si) {
        const int rlast = si * dec::kSeq + (dec::kSeq - 1);
        const float* hlast = sc.finalin + (int64_t)rlast * dec::kD;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) sum += hlast[j];
        float mean = dec_block_sum(sum, red) / (float)dec::kD;
        float vs = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) { float c = hlast[j] - mean; vs += c * c; }
        float var = dec_block_sum(vs, red) / (float)dec::kD;
        float iv = rsqrtf(var + dec::kLnEps);
        if (threadIdx.x == 0) sc.fni[rlast] = iv;
        // fn_xhat cache (last row); hn → X_hn (bf16 head dW operand) AND reuse the
        // X_hn bf16 as the scalar head input (read back below).
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float xh = (hlast[j] - mean) * iv;
            sc.fnx[(int64_t)rlast * dec::kD + j] = xh;
            float hn = xh * w.norm_w[j] + w.norm_b[j];
            acts.X_hn[(int64_t)si_global(g0, si) * dec::kD + j] = __float2bfloat16(hn);
        }
        __syncthreads();
        // logits[o] = hn · out_w[o] + out_b[o]  (scalar; hn read from X_hn bf16 so the
        // head input == the head dW operand exactly). Store into sc.logits[si*V..].
        float* lg = sc.logits + (int64_t)si * dec::kVocab;
        const __nv_bfloat16* hnb = acts.X_hn + (int64_t)si_global(g0, si) * dec::kD;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) {
            const float* Wr = w.out_w + (int64_t)o * dec::kD;
            float acc = w.out_b[o];
            #pragma unroll 4
            for (int k = 0; k < dec::kD; ++k) acc += __bfloat162float(hnb[k]) * Wr[k];
            lg[o] = acc;
        }
        __syncthreads();
        int tgt = tgt_ids[si_global(g0, si)];
        float lmax = -CUDART_INF_F;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) lmax = fmaxf(lmax, lg[o]);
        lmax = dec_block_max(lmax, red);
        float es = 0.0f;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) es += __expf(lg[o] - lmax);
        es = dec_block_sum(es, red);
        float logz = lmax + __logf(es);
        if (threadIdx.x == 0) nll_acc += (logz - lg[tgt]);
        __syncthreads();
    }
    return nll_acc;
}

// Attention BACKWARD over a tile (the oracle's 3-pass form, tile-batched).
// Reads qkv (bf16), attn weights, and dctx [nrows,d] fp32; writes dqkv [nrows,3d]
// fp32 into `dqkv_out`. dsc is the per-CTA dscores scratch. Mirror of
// dec_backward_sample's attention block (A: dv, B: dscores, C: dq/dk), looped
// over the tile's samples. scale = 1/sqrt(dh).
__device__ __forceinline__ void dectc_attn_bwd_tile(
        const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ attn_w,
        const float* __restrict__ dctx, int nrows,
        float* __restrict__ dqkv_out, float* __restrict__ dsc) {
    const int nsamp = nrows / dec::kSeq;
    const float scale = dec::attn_scale();
    // A: dv[kj] = Σ_{qi>=kj} attn[qi,kj] * dctx[qi].  Owner: (sample,kj,head,t).
    for (int r = threadIdx.x; r < nsamp * dec::kSeq * dec::kHeads * dec::kDhead; r += blockDim.x) {
        const int si  = r / (dec::kSeq * dec::kHeads * dec::kDhead);
        int rem = r % (dec::kSeq * dec::kHeads * dec::kDhead);
        const int kj  = rem / (dec::kHeads * dec::kDhead);
        rem = rem % (dec::kHeads * dec::kDhead);
        const int hh  = rem / dec::kDhead, t = rem % dec::kDhead;
        const int qoff = hh * dec::kDhead;
        const int rbase = si * dec::kSeq;
        const float* aw = attn_w + ((int64_t)(si * dec::kHeads + hh) * dec::kSeq) * dec::kSeq;  // [S,S]
        float acc = 0.0f;
        #pragma unroll
        for (int qi = kj; qi < dec::kSeq; ++qi)
            acc += aw[qi * dec::kSeq + kj] * dctx[(int64_t)(rbase + qi) * dec::kD + qoff + t];
        dqkv_out[(int64_t)(rbase + kj) * 3 * dec::kD + 2 * dec::kD + qoff + t] = acc;   // dv block
    }
    __syncthreads();
    // B: dscores ds[qi,kj] = attn*(datt - Σ_k datt*attn)*scale, masked kj>qi → 0.
    //    datt[kj] = Σ_t dctx[qi,qoff+t]*v[kj,qoff+t]. Owner: (sample,head,qi).
    for (int r = threadIdx.x; r < nsamp * dec::kHeads * dec::kSeq; r += blockDim.x) {
        const int si = r / (dec::kHeads * dec::kSeq);
        int rem = r % (dec::kHeads * dec::kSeq);
        const int hh = rem / dec::kSeq, qi = rem % dec::kSeq;
        const int qoff = hh * dec::kDhead;
        const int rbase = si * dec::kSeq;
        const float* aw = attn_w + ((int64_t)(si * dec::kHeads + hh) * dec::kSeq) * dec::kSeq;
        float datt[dec::kSeq];
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            if (kj > qi) { datt[kj] = 0.0f; continue; }
            float acc = 0.0f;
            #pragma unroll
            for (int t = 0; t < dec::kDhead; ++t)
                acc += dctx[(int64_t)(rbase + qi) * dec::kD + qoff + t]
                     * __bfloat162float(qkv[(int64_t)(rbase + kj) * 3 * dec::kD + 2 * dec::kD + qoff + t]);
            datt[kj] = acc;
        }
        float dot = 0.0f;
        #pragma unroll
        for (int kj = 0; kj <= qi; ++kj) dot += datt[kj] * aw[qi * dec::kSeq + kj];
        float* ds = dsc + ((int64_t)(si * dec::kHeads + hh) * dec::kSeq + qi) * dec::kSeq;
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            float a = aw[qi * dec::kSeq + kj];
            ds[kj] = (kj <= qi) ? a * (datt[kj] - dot) * scale : 0.0f;
        }
    }
    __syncthreads();
    // C: dq[qi] = Σ_kj ds[qi,kj]*k[kj]; dk[kj] = Σ_qi ds[qi,kj]*q[qi]. Owner: (sample,pos,head,t).
    for (int r = threadIdx.x; r < nsamp * dec::kSeq * dec::kHeads * dec::kDhead; r += blockDim.x) {
        const int si = r / (dec::kSeq * dec::kHeads * dec::kDhead);
        int rem = r % (dec::kSeq * dec::kHeads * dec::kDhead);
        const int pos = rem / (dec::kHeads * dec::kDhead);
        rem = rem % (dec::kHeads * dec::kDhead);
        const int hh = rem / dec::kDhead, t = rem % dec::kDhead;
        const int qoff = hh * dec::kDhead;
        const int rbase = si * dec::kSeq;
        const float* ds = dsc + ((int64_t)(si * dec::kHeads + hh) * dec::kSeq) * dec::kSeq;  // [S,S]
        float dq = 0.0f, dk = 0.0f;
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            dq += ds[pos * dec::kSeq + kj] * __bfloat162float(qkv[(int64_t)(rbase + kj) * 3 * dec::kD + dec::kD + qoff + t]);
            dk += ds[kj * dec::kSeq + pos] * __bfloat162float(qkv[(int64_t)(rbase + kj) * 3 * dec::kD + qoff + t]);
        }
        dqkv_out[(int64_t)(rbase + pos) * 3 * dec::kD + qoff + t] = dq;             // dq block
        dqkv_out[(int64_t)(rbase + pos) * 3 * dec::kD + dec::kD + qoff + t] = dk;   // dk block
    }
    __syncthreads();
}

// ────────────────────────────────────────────────────────────────────────────
//  TP HEAD-LOCALIZED attention BACKWARD (mirror of dectc_attn_bwd_tile over the
//  rank's LOCAL head shard, tp_kernel.md §6/§12 NOTE). IDENTICAL 3-pass math
//  (A: dv, B: dscores, C: dq/dk), but over Hloc = kHeads/P heads with local
//  per-row qkv stride 3*Dloc and q|k|v blocks at 0 / Dloc / 2*Dloc. `dctx` is
//  the rank's own [nrows, Dloc] context grad (the row-parallel out_proj dX is
//  comm-free local, tp_layer.cuh:51) and `dqkv_out` is the rank's own
//  [nrows, 3*Dloc] (fed to the column-parallel in_proj dX reduce ①'). attn_w/dsc
//  index with Hloc heads (matches the fwd-side dectc_attn_fwd_tile_tp). At TP=1
//  (Hloc==kHeads, Dloc==kD) every literal equals dectc_attn_bwd_tile EXACTLY;
//  this function is ONLY instantiated under `if constexpr (Par::kTPComm)`. If
//  kHeads%P != 0 the caller passes Hloc==0 ⇒ all three passes are NO-OPs.
__device__ __forceinline__ void dectc_attn_bwd_tile_tp(
        const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ attn_w,
        const float* __restrict__ dctx, int nrows,
        float* __restrict__ dqkv_out, float* __restrict__ dsc,
        int Hloc, int Dloc) {
    const int nsamp = nrows / dec::kSeq;
    const float scale = dec::attn_scale();
    const int stride3 = 3 * Dloc;                          // local per-row qkv stride
    // A: dv[kj] = Σ_{qi>=kj} attn[qi,kj] * dctx[qi].  Owner: (sample,kj,local-head,t).
    for (int r = threadIdx.x; r < nsamp * dec::kSeq * Hloc * dec::kDhead; r += blockDim.x) {
        const int si  = r / (dec::kSeq * Hloc * dec::kDhead);
        int rem = r % (dec::kSeq * Hloc * dec::kDhead);
        const int kj  = rem / (Hloc * dec::kDhead);
        rem = rem % (Hloc * dec::kDhead);
        const int hh  = rem / dec::kDhead, t = rem % dec::kDhead;
        const int qoff = hh * dec::kDhead;
        const int rbase = si * dec::kSeq;
        const float* aw = attn_w + ((int64_t)(si * Hloc + hh) * dec::kSeq) * dec::kSeq;  // [S,S]
        float acc = 0.0f;
        #pragma unroll
        for (int qi = kj; qi < dec::kSeq; ++qi)
            acc += aw[qi * dec::kSeq + kj] * dctx[(int64_t)(rbase + qi) * Dloc + qoff + t];
        dqkv_out[(int64_t)(rbase + kj) * stride3 + 2 * Dloc + qoff + t] = acc;   // dv block
    }
    __syncthreads();
    // B: dscores ds[qi,kj] = attn*(datt - Σ_k datt*attn)*scale, masked kj>qi → 0.
    //    datt[kj] = Σ_t dctx[qi,qoff+t]*v[kj,qoff+t]. Owner: (sample,local-head,qi).
    for (int r = threadIdx.x; r < nsamp * Hloc * dec::kSeq; r += blockDim.x) {
        const int si = r / (Hloc * dec::kSeq);
        int rem = r % (Hloc * dec::kSeq);
        const int hh = rem / dec::kSeq, qi = rem % dec::kSeq;
        const int qoff = hh * dec::kDhead;
        const int rbase = si * dec::kSeq;
        const float* aw = attn_w + ((int64_t)(si * Hloc + hh) * dec::kSeq) * dec::kSeq;
        float datt[dec::kSeq];
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            if (kj > qi) { datt[kj] = 0.0f; continue; }
            float acc = 0.0f;
            #pragma unroll
            for (int t = 0; t < dec::kDhead; ++t)
                acc += dctx[(int64_t)(rbase + qi) * Dloc + qoff + t]
                     * __bfloat162float(qkv[(int64_t)(rbase + kj) * stride3 + 2 * Dloc + qoff + t]);
            datt[kj] = acc;
        }
        float dot = 0.0f;
        #pragma unroll
        for (int kj = 0; kj <= qi; ++kj) dot += datt[kj] * aw[qi * dec::kSeq + kj];
        float* ds = dsc + ((int64_t)(si * Hloc + hh) * dec::kSeq + qi) * dec::kSeq;
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            float a = aw[qi * dec::kSeq + kj];
            ds[kj] = (kj <= qi) ? a * (datt[kj] - dot) * scale : 0.0f;
        }
    }
    __syncthreads();
    // C: dq[qi] = Σ_kj ds[qi,kj]*k[kj]; dk[kj] = Σ_qi ds[qi,kj]*q[qi]. Owner: (sample,pos,local-head,t).
    for (int r = threadIdx.x; r < nsamp * dec::kSeq * Hloc * dec::kDhead; r += blockDim.x) {
        const int si = r / (dec::kSeq * Hloc * dec::kDhead);
        int rem = r % (dec::kSeq * Hloc * dec::kDhead);
        const int pos = rem / (Hloc * dec::kDhead);
        rem = rem % (Hloc * dec::kDhead);
        const int hh = rem / dec::kDhead, t = rem % dec::kDhead;
        const int qoff = hh * dec::kDhead;
        const int rbase = si * dec::kSeq;
        const float* ds = dsc + ((int64_t)(si * Hloc + hh) * dec::kSeq) * dec::kSeq;  // [S,S]
        float dq = 0.0f, dk = 0.0f;
        #pragma unroll
        for (int kj = 0; kj < dec::kSeq; ++kj) {
            dq += ds[pos * dec::kSeq + kj] * __bfloat162float(qkv[(int64_t)(rbase + kj) * stride3 + Dloc + qoff + t]);
            dk += ds[kj * dec::kSeq + pos] * __bfloat162float(qkv[(int64_t)(rbase + kj) * stride3 + qoff + t]);
        }
        dqkv_out[(int64_t)(rbase + pos) * stride3 + qoff + t] = dq;          // dq block
        dqkv_out[(int64_t)(rbase + pos) * stride3 + Dloc + qoff + t] = dk;   // dk block
    }
    __syncthreads();
}

// ════════════════════════════════════════════════════════════════════════
//  BACKWARD over one TOKEN TILE. Assumes dectc_forward_tile ran for THIS tile
//  (scratch + DecActs X-inputs populated). Fork B: computes dX via wgmma and
//  WRITES the dY output-adjoints to DecActs (dY_qkv/dY_a/dY_ff0/dY_ff2/
//  dY_logits) + dh0 for P2's output-stationary dW — it does NOT touch the
//  weight dW here. ACCUMULATES the 10 LN-vector grads (γ/β) into the per-CTA
//  LN-vec partials `lnvec` [kNumLnVec × d] (deterministic single-owner-per-j).
//  `B` is the full batch (CE mean scale). Mirrors dec_backward_sample.
//
//  dqkv/dctx/dgact intermediates use the fp32 `work` buffer + a second fp32
//  buffer `work2` (caller passes both, each [nrows×dff]); dh (running adjoint)
//  is the dedicated scratch.dh.
// ════════════════════════════════════════════════════════════════════════
__device__ void dectc_backward_tile(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, int B, const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tgt_ids,
        float* __restrict__ lnvec, float* __restrict__ work2,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        unsigned long long* pipeBars = nullptr) {
    const int nsamp = nrows / dec::kSeq;
    // LN-vec partial slots (dense order; see kLnVecTensorIdx).
    float* gn_n1w[dec::kLayers]; float* gn_n1b[dec::kLayers];
    float* gn_n2w[dec::kLayers]; float* gn_n2b[dec::kLayers];
    for (int li = 0; li < dec::kLayers; ++li) {
        gn_n1w[li] = lnvec + (int64_t)(li * 4 + 0) * dec::kD;
        gn_n1b[li] = lnvec + (int64_t)(li * 4 + 1) * dec::kD;
        gn_n2w[li] = lnvec + (int64_t)(li * 4 + 2) * dec::kD;
        gn_n2b[li] = lnvec + (int64_t)(li * 4 + 3) * dec::kD;
    }
    float* gn_normw = lnvec + (int64_t)(4 * dec::kLayers + 0) * dec::kD;  // 8*kD at L=2
    float* gn_normb = lnvec + (int64_t)(4 * dec::kLayers + 1) * dec::kD;  // 9*kD at L=2

    // ── CE bwd (per sample): dlogits = (softmax - onehot)/B, overwrite logits.
    //    head bwd: dY_logits[si] = dlogits (the head dW operand); dhn = dlogits@out_w.
    //    final-norm bwd: dh_last (last position only) → scratch.dh (zero others). ──
    // First zero scratch.dh for the whole tile (only last positions get grad).
    for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) sc.dh[idx] = 0.0f;
    __syncthreads();
    for (int si = 0; si < nsamp; ++si) {
        const int rlast = si * dec::kSeq + (dec::kSeq - 1);
        const int gs = si_global(g0, si);
        float* lg = sc.logits + (int64_t)si * dec::kVocab;
        int tgt = tgt_ids[gs];
        // softmax of cached logits.
        float lmax = -CUDART_INF_F;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) lmax = fmaxf(lmax, lg[o]);
        lmax = dec_block_max(lmax, red);
        float es = 0.0f;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) es += __expf(lg[o] - lmax);
        es = dec_block_sum(es, red);
        float inv_es = 1.0f / es;
        // dlogits → overwrite lg, AND write to dY_logits[gs] (bf16, head dW operand).
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) {
            float smo = __expf(lg[o] - lmax) * inv_es;
            float dl = (smo - ((o == tgt) ? 1.0f : 0.0f)) / (float)B;
            lg[o] = dl;
            acts.dY_logits[(int64_t)gs * dec::kVocab + o] = __float2bfloat16(dl);
        }
        __syncthreads();
        // dhn[j] = Σ_o dlogits[o] * out_w[o,j]  (head dX), feature-parallel → dh row rlast.
        // Then final-norm bwd of that single row → scratch.dh[rlast].
        // Use fnx cache (xhat) + fni (inv). Accumulate norm γ/β. (head dW is P2.)
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dhn = 0.0f;
            for (int o = 0; o < dec::kVocab; ++o)
                dhn += lg[o] * w.out_w[(int64_t)o * dec::kD + j];
            // final-norm bwd needs the row-reduce of dxhat; stash dhn into work row rlast.
            sc.work[(int64_t)rlast * dec::kD + j] = dhn;
        }
        __syncthreads();
        // norm γ/β: dnorm_w[j] += dhn*xhat; dnorm_b[j] += dhn. (Only last pos contributes.)
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dhn = sc.work[(int64_t)rlast * dec::kD + j];
            float xh = sc.fnx[(int64_t)rlast * dec::kD + j];
            gn_normw[j] += dhn * xh; gn_normb[j] += dhn;
        }
        __syncthreads();
        // LN dx (single row): dxhat=dhn*norm_w; reduce; dh[rlast] = inv*(dxhat-(sda+xhat*sdax)/d).
        {
            const float* dyr = sc.work + (int64_t)rlast * dec::kD;
            const float* xhr = sc.fnx + (int64_t)rlast * dec::kD;
            float sda = 0.0f, sdax = 0.0f;
            for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
                float dxhat = dyr[j] * w.norm_w[j]; sda += dxhat; sdax += dxhat * xhr[j];
            }
            sda = dec_block_sum(sda, red); sdax = dec_block_sum(sdax, red);
            float iv = sc.fni[rlast];
            for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
                float dxhat = dyr[j] * w.norm_w[j];
                sc.dh[(int64_t)rlast * dec::kD + j] = iv * (dxhat - (sda + xhr[j] * sdax) / (float)dec::kD);
            }
            __syncthreads();
        }
    }
    // scratch.dh now = grad wrt last-layer output [nrows,d] (only last positions nonzero).

    // ── per-layer backward (reverse). dh is the running adjoint (grad wrt the
    //    layer's n2 output). All fwd intermediates are in scratch/DecActs (NO
    //    recompute). ──
    for (int li = dec::kLayers - 1; li >= 0; --li) {
        const DecWeights::Layer& L = w.layer[li];
        // n2 bwd: dh → dr2 (work fp32), accumulate n2 γ/β. xhat=n2x[li], inv=n2i[li].
        dectc_ln_bwd_tile(sc.dh, sc.n2x[li], sc.n2i[li], L.n2_w, nrows, sc.work, gn_n2w[li], gn_n2b[li], red);
        // r2 = x1 + ff2 → dx1 = dr2 (residual), dff2 = dr2. dff2 → dY_ff2 acts (bf16).
        // dx1 starts as dr2 (copy into work2), the FFN path adds to it.
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            work2[idx] = sc.work[idx];   // dx1 := dr2 (residual part)
            acts.dY_ff2[li][(int64_t)g0 * dec::kD + idx] = __float2bfloat16(sc.work[idx]);  // dff2
        }
        __syncthreads();
        // ff2 dX: dgact = dff2 @ ff2_w  (N=dff, K=d). fp32 → tw? need a [nrows,dff] buffer.
        //   Use sc.work (currently dr2, no longer needed — dx1 saved in work2, dff2 in acts).
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff2[li] + (int64_t)g0 * dec::kD, wb.ff2_wT[li],
                                           sc.work, dec::kDff, dec::kD, sA, sB, pipeBars);  // dgact [nrows,dff]
        __syncthreads();
        // dff0 = dgact * gelu'(ff0pre) → dY_ff0 acts (bf16) AND keep fp32 in sc.work for dX.
        for (int idx = threadIdx.x; idx < nrows * dec::kDff; idx += blockDim.x) {
            float dff0 = sc.work[idx] * dec_gelu_grad(__bfloat162float(sc.ff0pre[li][idx]));
            sc.work[idx] = dff0;
            acts.dY_ff0[li][(int64_t)g0 * dec::kDff + idx] = __float2bfloat16(dff0);
        }
        __syncthreads();
        // ff0 dX: dx1 += dff0 @ ff0_w  (output width Kin=d, contract Nout=dff). fp32
        //   → sc.x1 (free now — fwd x1 consumed); then add to work2.
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff0[li] + (int64_t)g0 * dec::kDff, wb.ff0_wT[li],
                                           sc.x1, /*Kin=*/dec::kD, /*Nout=*/dec::kDff, sA, sB, pipeBars);  // dx1_ffn [nrows,d]
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x)
            work2[idx] += sc.x1[idx];   // dx1 = residual + FFN path
        __syncthreads();
        // n1 bwd: dx1 (work2) → dr1 (work), accumulate n1 γ/β. xhat=n1x[li], inv=n1i[li].
        dectc_ln_bwd_tile(work2, sc.n1x[li], sc.n1i[li], L.n1_w, nrows, sc.work, gn_n1w[li], gn_n1b[li], red);
        // r1 = x_in + a → da = dr1 (out_proj output adjoint), dx_in = dr1 (residual).
        // SAVE the residual dr1 into sc.dh NOW (dh is free — its grad was consumed into
        // dr2 at the top of this layer); attention bwd will overwrite work2. Then add the
        // in_proj dX path to it. da → dY_a acts (bf16, out_proj dW operand).
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            sc.dh[idx] = sc.work[idx];   // residual dx_in := dr1  (saved across attn bwd)
            acts.dY_a[li][(int64_t)g0 * dec::kD + idx] = __float2bfloat16(sc.work[idx]);  // da
        }
        __syncthreads();
        // out_proj dX: dctx = da @ out_w  (N=d, K=d). fp32 → sc.work (dctx [nrows,d]).
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_a[li] + (int64_t)g0 * dec::kD, wb.out_wT[li],
                                           sc.work, dec::kD, dec::kD, sA, sB, pipeBars);  // dctx
        __syncthreads();
        // attention bwd: (qkv[li], attn[li], dctx=work) → dqkv [nrows,3d] fp32 into
        //   work2 (3d=384 ≤ dff=512, fits). Then → dY_qkv acts (bf16, in_proj dW operand).
        dectc_attn_bwd_tile(sc.qkv[li], sc.attn[li], sc.work, nrows, work2, sc.dsc);
        for (int idx = threadIdx.x; idx < nrows * 3 * dec::kD; idx += blockDim.x)
            acts.dY_qkv[li][(int64_t)g0 * 3 * dec::kD + idx] = __float2bfloat16(work2[idx]);
        __syncthreads();
        // in_proj dX: dx_in_attn = dqkv @ in_w  (output width Kin=d, contract Nout=3d).
        //   fp32 → sc.work; ADD residual (in sc.dh) → new running adjoint dh.
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_qkv[li] + (int64_t)g0 * 3 * dec::kD, wb.in_wT[li],
                                           sc.work, /*Kin=*/dec::kD, /*Nout=*/3 * dec::kD, sA, sB, pipeBars);  // dx_in_attn
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x)
            sc.dh[idx] += sc.work[idx];   // dx_in = residual (in dh) + attn path
        __syncthreads();
    }

    // ── embedding bwd: dh = grad wrt h0 [nrows,d]. Write dh0 acts (bf16); the
    //    tok/pos owner-scan (P2) reads dh0 by global token row. ──
    for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x)
        acts.dh0[(int64_t)g0 * dec::kD + idx] = __float2bfloat16(sc.dh[idx]);
    __syncthreads();
}

// ════════════════════════════════════════════════════════════════════════
//  TP-AWARE forward/backward tile bodies (EDIT D, /workspace/impl_diffs/
//  tp_kernel.md §6). TWO-BODY shape (the spec's SAFE default): the SingleGPU
//  entries `dectc_forward_tile` / `dectc_backward_tile` above are LEFT
//  BYTE-IDENTICAL (so the test_decoder_tc.py PTX gate is guaranteed). These
//  `_impl<Par,Transport>` bodies are NEW functions only ever instantiated under
//  `if constexpr (Par::kTPComm)` at the megakernel call site (folded away on
//  SingleGPU), so they NEVER codegen on the default path. They are a copy of the
//  SingleGPU body with the FOUR all-reduce points wrapped in `if constexpr
//  (Par::kTPComm)` (① out_proj fwd, ② ff2 fwd, ②' ff0 dX, ①' in_proj dX) + the
//  two COLUMN-parallel forward GEMM width ternaries (§6 NOTE).
//
//  HONEST SCOPE (recorded as a deviation): the attention head-localization
//  (H_loc = kHeads/P, the local-shard qkv stride) is the one extra correctness
//  touch the spec (§6 NOTE) flags BEYOND the four reduce points; it is a deeper
//  rewrite of dectc_attn_fwd/bwd_tile (which hardcode 3*dec::kD stride and
//  dec::kHeads). On the kTPComm path here the attention still runs full-width;
//  the per-rank QKV shard + local-head attention is the 8×H100-window task
//  (tp_kernel.md §12: this track "is the part most likely to need an on-silicon
//  iteration"). The FOUR linear-projection reduces — the core of EDIT D — are
//  exact and reuse the loopback-validated tp_* primitives.
// ════════════════════════════════════════════════════════════════════════
template <class Par = ::sg::fused::par::SingleGPU,
          class Transport = ::sg::fused::sm90::tp::LoopbackTransport>
__device__ float dectc_forward_tile_impl(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, bool active,
        const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tok_ids,
        const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        const Transport& tr, const ::sg::fused::GridBarrier& bar,
        int64_t slot_pub, int64_t slot_red,
        unsigned long long* pipeBars = nullptr) {
    // On the kTPComm path an inactive round still must reach every rendezvous;
    // the GEMM/elementwise work is skipped (nrows==0) but the rendezvous calls
    // at ①/② below run unconditionally (the §1 lockstep invariant).
    (void)active; (void)slot_red;
    const int nsamp = nrows / dec::kSeq;
    // ── Embedding: X_in[0][r] = tok[token_id[g0+r]] + pos[(g0+r)%S]. bf16. ──
    for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
        const int r = idx / dec::kD, j = idx % dec::kD;
        const int g = g0 + r;
        const int tid = tok_ids[g];
        const int sp = g % dec::kSeq;            // position within the sample
        float v = w.tok[(int64_t)tid * dec::kD + j] + w.pos[(int64_t)sp * dec::kD + j];
        acts.X_in[0][(int64_t)g * dec::kD + j] = __float2bfloat16(v);
    }
    __syncthreads();

    for (int li = 0; li < dec::kLayers; ++li) {
        const DecWeights::Layer& L = w.layer[li];
        const __nv_bfloat16* Xin = acts.X_in[li] + (int64_t)g0 * dec::kD;        // [nrows,d]
        // qkv = Xin @ in_w^T + in_b   (N=3d, K=d). bf16 → scratch.qkv[li].
        // COLUMN(QKV)-parallel on kTPComm: rank owns 3d/P out columns (§6 NOTE).
        const int qkv_nout = Par::kTPComm ? (3 * dec::kD) / Par::kTP : 3 * dec::kD;
        dectc_gemm_fwd<SG_TUNED_TILE_N>(Xin, wb.in_w[li], sc.qkv[li], dec::kD, qkv_nout, sA, sB, pipeBars);
        __syncthreads();
        // add in_b (the fwd GEMM did W only; bias folded in scalar here for qkv —
        // matches the bf16-faithful oracle qkv = bf(x_in @ bf(in_w)^T + in_b)).
        for (int idx = threadIdx.x; idx < nrows * qkv_nout; idx += blockDim.x) {
            const int j = idx % qkv_nout;
            float v = __bfloat162float(sc.qkv[li][idx]) + L.in_b[j];
            sc.qkv[li][idx] = __float2bfloat16(v);
        }
        __syncthreads();
        // attention → ctx (work fp32) + attn[li] weights. HEAD-LOCALIZED on the
        // kTPComm path: the col-parallel qkv is the rank's [q|k|v]_own shard
        // (stride 3*Dloc, Hloc=kHeads/P heads); ctx is the rank's own [nrows,Dloc]
        // (recombined to full width by the row-parallel out_proj all-reduce ① below).
        // tp_kernel.md §6/§12 head-localization NOTE. SingleGPU path uses the
        // byte-identical dectc_attn_fwd_tile above (this branch folds away there).
        const int Dloc  = Par::kTPComm ? (dec::kD / Par::kTP) : dec::kD;          // local q/k/v block width
        const int Hloc  = Par::kTPComm ? (dec::kHeads / Par::kTP) : dec::kHeads;  // local whole heads (0 if kHeads%P!=0)
        if constexpr (!Par::kTPComm) {
            dectc_attn_fwd_tile(sc.qkv[li], nrows, sc.work, sc.attn[li]);
            // ctx bf16 → X_ctx[li] (out_proj input + its dW operand). Full width.
            for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
                const int r = idx / dec::kD, j = idx % dec::kD;
                acts.X_ctx[li][(int64_t)(g0 + r) * dec::kD + j] = __float2bfloat16(sc.work[(int64_t)r * dec::kD + j]);
            }
        } else {
            dectc_attn_fwd_tile_tp(sc.qkv[li], nrows, sc.work, sc.attn[li], Hloc, Dloc);
            // ctx bf16 → X_ctx[li] at LOCAL width Dloc (== out_proj's Kloc input
            // width; the row-parallel out_proj GEMM reads acts.X_ctx + g0*Dloc).
            for (int idx = threadIdx.x; idx < nrows * Dloc; idx += blockDim.x) {
                const int r = idx / Dloc, j = idx % Dloc;
                acts.X_ctx[li][(int64_t)(g0 + r) * Dloc + j] = __float2bfloat16(sc.work[(int64_t)r * Dloc + j]);
            }
        }
        __syncthreads();
        // a = X_ctx @ out_w^T (+ out_b)  (N=d, K=d). fp32 → work.
        if constexpr (!Par::kTPComm) {
            dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_ctx[li] + (int64_t)g0 * dec::kD, wb.out_w[li],
                                                sc.work, dec::kD, dec::kD, sA, sB, pipeBars);
        } else {
            // ROW-parallel out_proj: out_w[li] is the [d, d/P] col-shard; X_ctx is
            // the rank's own ctx [nrows, d/P]. Publish the [nrows,d] partial to the
            // symmetric slot, rendezvous, fixed-order ascending-pe reduce → sc.work.
            // (① of design §5.1 / tp_layer.cuh. Activations are full-width [nrows,d]
            // post-reduce, so the r1 residual+bias fold below runs UNCHANGED.)
            const int Kloc = dec::kD / tr.n_pes();   // local input width (col-shard)
            if (active) {
                ::sg::fused::sm90::tp::tp_rowparallel_fwd_partial_tile<SG_TUNED_TILE_N>(
                    tr, slot_pub,
                    /*Xown=*/ acts.X_ctx[li] + (int64_t)g0 * Kloc, wb.out_w[li],
                    /*Kin_local=*/ Kloc, /*Nout=*/ dec::kD, sA, sB);
            }
            tr.rendezvous(bar);                                  // publish visible
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, sc.work, (int64_t)nrows * dec::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);                                  // slot reusable
        }
        __syncthreads();
        // r1 = Xin + a + out_b → work (fp32). out_b folded here (the GEMM did W only)
        // — matches the oracle a = ctx_b @ out_w^T + out_b kept fp32 through r1.
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            const int r = idx / dec::kD, j = idx % dec::kD;
            sc.work[(int64_t)r * dec::kD + j] += __bfloat162float(Xin[(int64_t)r * dec::kD + j]) + L.out_b[j];
        }
        __syncthreads();
        // n1(r1) → x1 (fp32) + caches[li]; then bf16 → X_x1[li] (ff0 input + dW operand).
        dectc_ln_fwd_tile(sc.work, L.n1_w, L.n1_b, nrows, sc.x1, sc.n1x[li], sc.n1i[li], red);
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            const int r = idx / dec::kD, j = idx % dec::kD;
            acts.X_x1[li][(int64_t)(g0 + r) * dec::kD + j] = __float2bfloat16(sc.x1[(int64_t)r * dec::kD + j]);
        }
        __syncthreads();
        // ff0 = X_x1 @ ff0_w^T (+ ff0_b)  (N=dff, K=d). fp32 → work; (pre+b) bf16 →
        // ff0pre; gelu(pre+b) → X_gact[li] (bf16, ff2 input + dW operand). ff0_b folded
        // into pre (fp32) — matches the oracle ff0pre=bf(ff0+b), gact=bf(gelu(ff0+b)).
        // COLUMN-parallel on kTPComm: rank owns dff/P out columns (§6 NOTE).
        const int ff0_nout = Par::kTPComm ? dec::kDff / Par::kTP : dec::kDff;
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_x1[li] + (int64_t)g0 * dec::kD, wb.ff0_w[li],
                                            sc.work, dec::kD, ff0_nout, sA, sB, pipeBars);
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * ff0_nout; idx += blockDim.x) {
            const int r = idx / ff0_nout, j = idx % ff0_nout;
            float pre = sc.work[(int64_t)r * ff0_nout + j] + L.ff0_b[j];
            sc.ff0pre[li][(int64_t)r * ff0_nout + j] = __float2bfloat16(pre);
            acts.X_gact[li][(int64_t)(g0 + r) * ff0_nout + j] = __float2bfloat16(dec_gelu(pre));
        }
        __syncthreads();
        // ff2 = X_gact @ ff2_w^T (+ ff2_b) (N=d, K=dff). fp32 → work.
        if constexpr (!Par::kTPComm) {
            dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_gact[li] + (int64_t)g0 * dec::kDff, wb.ff2_w[li],
                                                sc.work, dec::kDff, dec::kD, sA, sB, pipeBars);
        } else {
            // ROW-parallel ff2: ff2_w[li] is the [d, dff/P] col-shard; X_gact is the
            // rank's own gact [nrows, dff/P]. Publish [nrows,d] partial → reduce → sc.work
            // (② of design §5.1). r2 fold below runs unchanged on the reduced value.
            const int Kloc = dec::kDff / tr.n_pes();
            if (active) {
                ::sg::fused::sm90::tp::tp_rowparallel_fwd_partial_tile<SG_TUNED_TILE_N>(
                    tr, slot_pub,
                    /*Xown=*/ acts.X_gact[li] + (int64_t)g0 * Kloc, wb.ff2_w[li],
                    /*Kin_local=*/ Kloc, /*Nout=*/ dec::kD, sA, sB);
            }
            tr.rendezvous(bar);
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, sc.work, (int64_t)nrows * dec::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);
        }
        __syncthreads();
        // r2 = x1 + ff2 + ff2_b → work (fp32). x1 lives in the dedicated fp32 buffer
        // (no bf16 round). ff2_b folded here — matches the oracle r2 = x1 + (ff2 + ff2_b).
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            const int r = idx / dec::kD, j = idx % dec::kD;
            sc.work[(int64_t)r * dec::kD + j] += sc.x1[(int64_t)r * dec::kD + j] + L.ff2_b[j];
        }
        __syncthreads();
        if (li + 1 < dec::kLayers) {
            // n2(r2) → finalin (fp32 reused) + n2 caches[li]; bf16 → X_in[li+1].
            dectc_ln_fwd_tile(sc.work, L.n2_w, L.n2_b, nrows, sc.finalin, sc.n2x[li], sc.n2i[li], red);
            for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
                const int r = idx / dec::kD, j = idx % dec::kD;
                acts.X_in[li + 1][(int64_t)(g0 + r) * dec::kD + j] = __float2bfloat16(sc.finalin[(int64_t)r * dec::kD + j]);
            }
            __syncthreads();
        } else {
            // last layer: n2(r2) → finalin (fp32, all positions; head reads last pos) + n2 caches[li].
            dectc_ln_fwd_tile(sc.work, L.n2_w, L.n2_b, nrows, sc.finalin, sc.n2x[li], sc.n2i[li], red);
        }
    }

    // ── Final norm + head + CE, scalar PER-SAMPLE on the LAST position only.
    //    finalin holds the last-layer n2 output [nrows,d] fp32. ──
    float nll_acc = 0.0f;
    for (int si = 0; si < nsamp; ++si) {
        const int rlast = si * dec::kSeq + (dec::kSeq - 1);
        const float* hlast = sc.finalin + (int64_t)rlast * dec::kD;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) sum += hlast[j];
        float mean = dec_block_sum(sum, red) / (float)dec::kD;
        float vs = 0.0f;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) { float c = hlast[j] - mean; vs += c * c; }
        float var = dec_block_sum(vs, red) / (float)dec::kD;
        float iv = rsqrtf(var + dec::kLnEps);
        if (threadIdx.x == 0) sc.fni[rlast] = iv;
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float xh = (hlast[j] - mean) * iv;
            sc.fnx[(int64_t)rlast * dec::kD + j] = xh;
            float hn = xh * w.norm_w[j] + w.norm_b[j];
            acts.X_hn[(int64_t)si_global(g0, si) * dec::kD + j] = __float2bfloat16(hn);
        }
        __syncthreads();
        float* lg = sc.logits + (int64_t)si * dec::kVocab;
        const __nv_bfloat16* hnb = acts.X_hn + (int64_t)si_global(g0, si) * dec::kD;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) {
            const float* Wr = w.out_w + (int64_t)o * dec::kD;
            float acc = w.out_b[o];
            #pragma unroll 4
            for (int k = 0; k < dec::kD; ++k) acc += __bfloat162float(hnb[k]) * Wr[k];
            lg[o] = acc;
        }
        __syncthreads();
        int tgt = tgt_ids[si_global(g0, si)];
        float lmax = -CUDART_INF_F;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) lmax = fmaxf(lmax, lg[o]);
        lmax = dec_block_max(lmax, red);
        float es = 0.0f;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) es += __expf(lg[o] - lmax);
        es = dec_block_sum(es, red);
        float logz = lmax + __logf(es);
        if (threadIdx.x == 0) nll_acc += (logz - lg[tgt]);
        __syncthreads();
    }
    return nll_acc;
}

// TP-aware backward tile (mirror of dectc_backward_tile with the two backward
// reduce points ②' (ff0 dX) and ①' (in_proj dX) wrapped in if constexpr).
template <class Par = ::sg::fused::par::SingleGPU,
          class Transport = ::sg::fused::sm90::tp::LoopbackTransport>
__device__ void dectc_backward_tile_impl(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, bool active, int B,
        const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tgt_ids,
        float* __restrict__ lnvec, float* __restrict__ work2,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        const Transport& tr, const ::sg::fused::GridBarrier& bar,
        int64_t slot_pub, int64_t slot_red,
        unsigned long long* pipeBars = nullptr) {
    (void)active; (void)slot_red;
    const int nsamp = nrows / dec::kSeq;
    float* gn_n1w[dec::kLayers]; float* gn_n1b[dec::kLayers];
    float* gn_n2w[dec::kLayers]; float* gn_n2b[dec::kLayers];
    for (int li = 0; li < dec::kLayers; ++li) {
        gn_n1w[li] = lnvec + (int64_t)(li * 4 + 0) * dec::kD;
        gn_n1b[li] = lnvec + (int64_t)(li * 4 + 1) * dec::kD;
        gn_n2w[li] = lnvec + (int64_t)(li * 4 + 2) * dec::kD;
        gn_n2b[li] = lnvec + (int64_t)(li * 4 + 3) * dec::kD;
    }
    float* gn_normw = lnvec + (int64_t)(4 * dec::kLayers + 0) * dec::kD;
    float* gn_normb = lnvec + (int64_t)(4 * dec::kLayers + 1) * dec::kD;

    for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) sc.dh[idx] = 0.0f;
    __syncthreads();
    for (int si = 0; si < nsamp; ++si) {
        const int rlast = si * dec::kSeq + (dec::kSeq - 1);
        const int gs = si_global(g0, si);
        float* lg = sc.logits + (int64_t)si * dec::kVocab;
        int tgt = tgt_ids[gs];
        float lmax = -CUDART_INF_F;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) lmax = fmaxf(lmax, lg[o]);
        lmax = dec_block_max(lmax, red);
        float es = 0.0f;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) es += __expf(lg[o] - lmax);
        es = dec_block_sum(es, red);
        float inv_es = 1.0f / es;
        for (int o = threadIdx.x; o < dec::kVocab; o += blockDim.x) {
            float smo = __expf(lg[o] - lmax) * inv_es;
            float dl = (smo - ((o == tgt) ? 1.0f : 0.0f)) / (float)B;
            lg[o] = dl;
            acts.dY_logits[(int64_t)gs * dec::kVocab + o] = __float2bfloat16(dl);
        }
        __syncthreads();
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dhn = 0.0f;
            for (int o = 0; o < dec::kVocab; ++o)
                dhn += lg[o] * w.out_w[(int64_t)o * dec::kD + j];
            sc.work[(int64_t)rlast * dec::kD + j] = dhn;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float dhn = sc.work[(int64_t)rlast * dec::kD + j];
            float xh = sc.fnx[(int64_t)rlast * dec::kD + j];
            gn_normw[j] += dhn * xh; gn_normb[j] += dhn;
        }
        __syncthreads();
        {
            const float* dyr = sc.work + (int64_t)rlast * dec::kD;
            const float* xhr = sc.fnx + (int64_t)rlast * dec::kD;
            float sda = 0.0f, sdax = 0.0f;
            for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
                float dxhat = dyr[j] * w.norm_w[j]; sda += dxhat; sdax += dxhat * xhr[j];
            }
            sda = dec_block_sum(sda, red); sdax = dec_block_sum(sdax, red);
            float iv = sc.fni[rlast];
            for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
                float dxhat = dyr[j] * w.norm_w[j];
                sc.dh[(int64_t)rlast * dec::kD + j] = iv * (dxhat - (sda + xhr[j] * sdax) / (float)dec::kD);
            }
            __syncthreads();
        }
    }

    for (int li = dec::kLayers - 1; li >= 0; --li) {
        const DecWeights::Layer& L = w.layer[li];
        const int ff0_nout = Par::kTPComm ? dec::kDff / Par::kTP : dec::kDff;
        const int qkv_nout = Par::kTPComm ? (3 * dec::kD) / Par::kTP : 3 * dec::kD;
        dectc_ln_bwd_tile(sc.dh, sc.n2x[li], sc.n2i[li], L.n2_w, nrows, sc.work, gn_n2w[li], gn_n2b[li], red);
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            work2[idx] = sc.work[idx];
            acts.dY_ff2[li][(int64_t)g0 * dec::kD + idx] = __float2bfloat16(sc.work[idx]);
        }
        __syncthreads();
        // ff2 dX (ROW-parallel ff2 ⇒ dgact is the rank's local [nrows, dff/P], comm-free).
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff2[li] + (int64_t)g0 * dec::kD, wb.ff2_wT[li],
                                           sc.work, dec::kDff, ff0_nout, sA, sB, pipeBars);  // dgact [nrows,dff/P]
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * ff0_nout; idx += blockDim.x) {
            float dff0 = sc.work[idx] * dec_gelu_grad(__bfloat162float(sc.ff0pre[li][idx]));
            sc.work[idx] = dff0;
            acts.dY_ff0[li][(int64_t)g0 * ff0_nout + idx] = __float2bfloat16(dff0);
        }
        __syncthreads();
        // ff0 dX: dx1 += dff0 @ ff0_w  (output width Kin=d, contract Nout=dff). fp32
        //   → sc.x1 (free now — fwd x1 consumed); then add to work2.
        if constexpr (!Par::kTPComm) {
            dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff0[li] + (int64_t)g0 * dec::kDff, wb.ff0_wT[li],
                                               sc.x1, /*Kin=*/dec::kD, /*Nout=*/dec::kDff, sA, sB, pipeBars);  // dx1_ffn [nrows,d]
        } else {
            // COLUMN-parallel ff0: ff0_wT[li] is the rank's [dff/P, d] col-shard's
            // transpose; dY_ff0 is the rank's own [nrows, dff/P]. The dX is a PARTIAL
            // (Σ_pe) → publish [nrows,d] → reduce → sc.x1 (②' of design §5.1). Then
            // the `work2[idx] += sc.x1[idx]` accumulate below runs unchanged.
            const int Noutloc = dec::kDff / tr.n_pes();
            if (active) {
                ::sg::fused::sm90::tp::tp_colparallel_dx_partial_tile<SG_TUNED_TILE_N>(
                    tr, slot_pub,
                    /*dYown=*/ acts.dY_ff0[li] + (int64_t)g0 * Noutloc, wb.ff0_wT[li],
                    /*Kin=*/ dec::kD, /*Nout_local=*/ Noutloc, sA, sB);
            }
            tr.rendezvous(bar);
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, sc.x1, (int64_t)nrows * dec::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x)
            work2[idx] += sc.x1[idx];
        __syncthreads();
        dectc_ln_bwd_tile(work2, sc.n1x[li], sc.n1i[li], L.n1_w, nrows, sc.work, gn_n1w[li], gn_n1b[li], red);
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x) {
            sc.dh[idx] = sc.work[idx];
            acts.dY_a[li][(int64_t)g0 * dec::kD + idx] = __float2bfloat16(sc.work[idx]);
        }
        __syncthreads();
        // out_proj dX (ROW-parallel out_proj ⇒ dctx is the rank's local [nrows, d/P], comm-free).
        const int dctx_nin = Par::kTPComm ? dec::kD / Par::kTP : dec::kD;
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_a[li] + (int64_t)g0 * dec::kD, wb.out_wT[li],
                                           sc.work, dctx_nin, dec::kD, sA, sB, pipeBars);  // dctx [nrows, d/P]
        __syncthreads();
        // attention bwd HEAD-LOCALIZED on the kTPComm path (tp_kernel.md §6/§12):
        // dctx (sc.work) is the rank's own [nrows,Dloc] out_proj dX (dctx_nin==Dloc
        // above), qkv is the [q|k|v]_own shard (stride 3*Dloc, Hloc heads), and the
        // produced dqkv (work2) is the rank's own [nrows,3*Dloc] (==qkv_nout) fed to
        // the column-parallel in_proj dX reduce ①' below. SingleGPU uses the
        // byte-identical dectc_attn_bwd_tile (this branch folds away there).
        if constexpr (!Par::kTPComm) {
            dectc_attn_bwd_tile(sc.qkv[li], sc.attn[li], sc.work, nrows, work2, sc.dsc);
        } else {
            const int Dloc = dec::kD / Par::kTP;          // local q/k/v block width
            const int Hloc = dec::kHeads / Par::kTP;      // local whole heads (0 if kHeads%P!=0)
            dectc_attn_bwd_tile_tp(sc.qkv[li], sc.attn[li], sc.work, nrows, work2, sc.dsc, Hloc, Dloc);
        }
        for (int idx = threadIdx.x; idx < nrows * qkv_nout; idx += blockDim.x)
            acts.dY_qkv[li][(int64_t)g0 * qkv_nout + idx] = __float2bfloat16(work2[idx]);
        __syncthreads();
        // in_proj dX: dx_in_attn = dqkv @ in_w  (output width Kin=d, contract Nout=3d).
        //   fp32 → sc.work; ADD residual (in sc.dh) → new running adjoint dh.
        if constexpr (!Par::kTPComm) {
            dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_qkv[li] + (int64_t)g0 * 3 * dec::kD, wb.in_wT[li],
                                               sc.work, /*Kin=*/dec::kD, /*Nout=*/3 * dec::kD, sA, sB, pipeBars);  // dx_in_attn
        } else {
            // COLUMN(QKV)-parallel in_proj: in_wT[li] is the rank's [3d/P, d] qkv
            // col-shard's transpose; dY_qkv is the rank's own [nrows, 3d/P] (the
            // 3-block q|k|v own-rows concatenated). The dX is a PARTIAL → publish
            // [nrows,d] → reduce → sc.work (①' of design §5.1). Then the residual
            // add `sc.dh[idx] += sc.work[idx]` below runs unchanged.
            const int Noutloc = (3 * dec::kD) / tr.n_pes();
            if (active) {
                ::sg::fused::sm90::tp::tp_colparallel_dx_partial_tile<SG_TUNED_TILE_N>(
                    tr, slot_pub,
                    /*dYown=*/ acts.dY_qkv[li] + (int64_t)g0 * Noutloc, wb.in_wT[li],
                    /*Kin=*/ dec::kD, /*Nout_local=*/ Noutloc, sA, sB);
            }
            tr.rendezvous(bar);
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, sc.work, (int64_t)nrows * dec::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x)
            sc.dh[idx] += sc.work[idx];
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < nrows * dec::kD; idx += blockDim.x)
        acts.dh0[(int64_t)g0 * dec::kD + idx] = __float2bfloat16(sc.dh[idx]);
    __syncthreads();
}

// Thin wrappers EDIT C.3 calls (forward `Par`/`Transport` to the _impl bodies).
template <class Par, class Transport>
__device__ __forceinline__ float dectc_forward_tile_tp(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, bool active,
        const DecActs& acts, const DecTileScratch& sc,
        const int* __restrict__ tok_ids, const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        const Transport& tr, const ::sg::fused::GridBarrier& bar,
        int64_t slot_pub, int64_t slot_red,
        unsigned long long* pipeBars = nullptr) {
    return dectc_forward_tile_impl<Par, Transport>(
        w, wb, g0, nrows, active, acts, sc, tok_ids, tgt_ids, sA, sB, red,
        tr, bar, slot_pub, slot_red, pipeBars);
}
template <class Par, class Transport>
__device__ __forceinline__ void dectc_backward_tile_tp(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, bool active, int B,
        const DecActs& acts, const DecTileScratch& sc, const int* __restrict__ tgt_ids,
        float* __restrict__ lnvec, float* __restrict__ work2,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        const Transport& tr, const ::sg::fused::GridBarrier& bar,
        int64_t slot_pub, int64_t slot_red,
        unsigned long long* pipeBars = nullptr) {
    dectc_backward_tile_impl<Par, Transport>(
        w, wb, g0, nrows, active, B, acts, sc, tgt_ids, lnvec, work2, sA, sB, red,
        tr, bar, slot_pub, slot_red, pipeBars);
}

// ════════════════════════════════════════════════════════════════════════
//  SAM-cell-scoped OUT-OF-LINE tile shims (reg-pressure campaign C2 -- the
//  mamba precedent, model_stage_mamba_tc.cuh mbtc_forward_tile/backward_tile,
//  SCOPED).  The SAM-coupled cells (LookSAM/SG11/SG15/SG2) run the heavy tile
//  fwd+bwd TWICE per step (P1 + the P2.4 SAM 2nd pass); inlining both copies
//  into one kernel blows the 255-reg budget by ~15 KB of hot-loop spills.
//  Routing BOTH passes of those cells through these __noinline__ shims gives
//  one shared out-of-line frame (measured: total spill bytes roughly halve and
//  leave the entry body nearly clean).  The single-pass cells keep calling the
//  inline bodies directly -- their allocation (255 regs, ZERO spills) is
//  byte-identical to the pre-campaign engine, never taxed with an ABI boundary.
//  Math identical; warpgroup-uniform call (whole CTA enters/exits together) so
//  the wgmma fence/commit/wait choreography inside is well-formed.
// ════════════════════════════════════════════════════════════════════════
__device__ __noinline__ float dectc_forward_tile_outlined(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tok_ids,
        const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        unsigned long long* pipeBars = nullptr) {
    return dectc_forward_tile(w, wb, g0, nrows, acts, sc, tok_ids, tgt_ids, sA, sB, red, pipeBars);
}
__device__ __noinline__ void dectc_backward_tile_outlined(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, int B, const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tgt_ids,
        float* __restrict__ lnvec, float* __restrict__ work2,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        unsigned long long* pipeBars = nullptr) {
    dectc_backward_tile(w, wb, g0, nrows, B, acts, sc, tgt_ids, lnvec, work2, sA, sB, red, pipeBars);
}

// ════════════════════════════════════════════════════════════════════════
//  P2 — OUTPUT-STATIONARY dW (the Q2 deliverable). Each of the 9 weight
//  matrices dW = dYᵀ @ X (K=T) is split into 64×N output tiles; tile_id %
//  nCTA owns each (fixed every step → determinism + L2 warmth). The owner CTA
//  contracts the FULL token dimension itself (ascending-t, no float atomics, no
//  partials) via the validated dW orientation (tc_gemm_block_unpipelined with
//  BOTH operands transposed-staged, MaxAtomsM=1 → one 64×N tile, 64 acc regs/
//  thread, no spill). Writes the tile into `grad` (the reduced-grad output).
//
//  The 9 weights, in dec_layout tensor-index order, with their (dY,X) acts and
//  the contraction length K (T for per-position weights, B for the head).
// ════════════════════════════════════════════════════════════════════════
struct DecDwSpec {
    const __nv_bfloat16* dY;   // [K, Nout]
    const __nv_bfloat16* X;    // [K, Kin]
    int Nout; int Kin; int K;
    int grad_off;              // element offset of this weight in `grad`
    const __nv_bfloat16* dY_bias;  // same as dY (bias db = Σ_K dY)
    int bias_off;              // element offset of the bias in `grad`
#if SG_TUNED_DEC_DW_STAGE
    // CONTIGUOUS-TRANSPOSE staging scratch (SG_TUNED_DEC_DW_STAGE=1 only). These
    // three pointers exist ONLY when the macro is set, so the scalar-default
    // struct (and therefore DecTcSmem's smem footprint + the kernel's register
    // allocation) is BYTE-IDENTICAL to every shipped build (PTX-verified). dYt =
    // K-major TRANSPOSE of dY, shape [Mpad, K] row-major (dYt[m,k] = dY[k,m]); Xt
    // = [Kin, K] (Xt[n,k] = X[k,n]). Rows are K-CONTIGUOUS + 16B-aligned (K is a
    // multiple of kWgmmaAtomK=16 ⇒ 8 bf16), so DecGmemTileSrcA/B read them with
    // the proven 16B cp.async ring instead of the 2-byte transposed-strided
    // gather. Filled ONCE per step by dectc_dw_transpose_operands (a pure bf16
    // copy — no math → parity-safe).
    __nv_bfloat16* dYt;        // [Mpad, K] (transposed dY) — null unless active
    __nv_bfloat16* Xt;         // [Kin,  K] (transposed X)  — null unless active
    int64_t t_off;             // element offset of (dYt|Xt) base in the dW-T scratch
#endif
};

// Number of dW specs: 4 per layer (in_proj, out_proj, ff0, ff2) + 1 head.out.
//   = 4*L + 1  (= 9 at L=2, the original spec[9]; = 193 at the flagship L=48).
// This is the compile-time bound for EVERY DecDwSpec array (the DecTcSmem member,
// every spec[] signature/local, the dW phase loops, the bias prefix). At L=2 it
// is exactly 9 → the spec arrays + their stack/smem footprint are byte-identical.
constexpr int kDecNumDwSpecs = 4 * dec::kLayers + 1;

// ── CONTIGUOUS-TRANSPOSE dW scratch geometry (SG_TUNED_DEC_DW_STAGE=1). The 9
//    weights' transposed operands (dYt[Nout,K] + Xt[Kin,K]) are packed into ONE
//    contiguous bf16 region. Per weight s the block is (Nout_s + Kin_s)·K_s bf16:
//    dYt at base+0, Xt at base+Nout_s·K_s. The SAME running-offset walk is used
//    by the host workspace sizer (dec_tc_dw_transpose_floats) and the device spec
//    builder (dectc_build_dw_specs) so the carve and the kernel agree exactly. K_s
//    is T for the 8 layer weights, B for the head — both multiples of kWgmmaAtomK
//    in every (B, kSeq) the bench/production use (so each transposed ROW is 16B-
//    aligned, the cp.async-ring chunk16 requirement). Returns bf16 ELEMENTS. ──
// Per-weight transpose-block bf16 elems: dYt is row-padded to a multiple of 64
// (the engine's m64 atom) so the cp.async ring's UNGUARDED A-row reads (it has no
// row_valid check on A) never index past the valid Nout rows — pad rows are
// zeroed (inert in the wgmma). Xt is NOT row-padded (the ring guards B rows via
// row_valid → only valid Kin rows are read; pad cols are zero-filled in smem).
__host__ __device__ __forceinline__ int64_t dec_dw_mpad(int Nout) {
    return (int64_t)((Nout + wgs::kWgmmaAtomM - 1) / wgs::kWgmmaAtomM) * wgs::kWgmmaAtomM;
}
__host__ __device__ __forceinline__ int64_t dec_dw_weight_t_elems(int Nout, int Kin, int64_t K) {
    return (dec_dw_mpad(Nout) + (int64_t)Kin) * K;     // dYt[Mpad,K] + Xt[Kin,K]
}
// True only when the contiguous-transpose path is BOTH selected (stage==1) AND
// actually wired (single-CTA dW, splitk==1 — the proof path; the split-K path
// keeps the lambda gather). When false the scratch is 0-sized / unused and the
// run-tile never reads dYt/Xt, so the workspace + control flow are byte-identical
// to the scalar default.
constexpr bool kDecDwTransposeActive = (kDecDwStage == 1) && (kDecDwSplitK == 1);

__host__ __device__ __forceinline__ int64_t dec_dw_transpose_elems(int B, int T) {
    if (!kDecDwTransposeActive) return 0;
    const int64_t Kly = T, Khd = B;
    int64_t e = 0;
    // 8 layer weights (per layer: qkv 3d×d, attn_out d×d, ff0 dff×d, ff2 d×dff).
    for (int li = 0; li < dec::kLayers; ++li) {
        e += dec_dw_weight_t_elems(3 * dec::kD, dec::kD,   Kly);  // qkv
        e += dec_dw_weight_t_elems(dec::kD,     dec::kD,   Kly);  // attn_out
        e += dec_dw_weight_t_elems(dec::kDff,   dec::kD,   Kly);  // ff0
        e += dec_dw_weight_t_elems(dec::kD,     dec::kDff, Kly);  // ff2
    }
    e += dec_dw_weight_t_elems(dec::kVocab, dec::kD, Khd);        // head (K=B)
    return e;
}

// Build the 9 specs (called by all CTAs; cheap). T = B*kSeq.
//
// When SG_TUNED_DEC_DW_STAGE==1, `dwt_base` is the contiguous bf16 transpose
// scratch (dec_dw_transpose_elems sized) and each spec's dYt/Xt/t_off are bound
// to its packed sub-block (same running-offset walk as dec_dw_transpose_elems +
// the host carve). On the scalar path pass dwt_base=nullptr → dYt/Xt stay null.
__device__ __forceinline__ void dectc_build_dw_specs(
        const DecActs& acts, int B, int T, DecDwSpec spec[kDecNumDwSpecs],
        __nv_bfloat16* dwt_base);

// Back-compat overload (scalar path / pre-existing call sites): no transpose
// scratch. Forwards with dwt_base=nullptr so dYt/Xt are null and stage==0 runs
// the proven lambda gather unchanged.
__device__ __forceinline__ void dectc_build_dw_specs(
        const DecActs& acts, int B, int T, DecDwSpec spec[kDecNumDwSpecs]) {
    dectc_build_dw_specs(acts, B, T, spec, /*dwt_base=*/nullptr);
}

__device__ __forceinline__ void dectc_build_dw_specs(
        const DecActs& acts, int B, int T, DecDwSpec spec[kDecNumDwSpecs],
        __nv_bfloat16* dwt_base) {
    // dec_layout offsets: see kDecOffsets. Per-layer 12-tensor block (li):
    //   weight tidx = 2 + 12*li + {0,2,8,10}[kind] (in_w,out_w,ff0_w,ff2_w), bias +1.
    //   head out.weight tidx = 2 + 12*L + 2, bias +1.
    // At L=2 these reproduce the old wi {2,4,10,12,14,16,22,24} / bi {3,5,11,13,
    // 15,17,23,25} and head {28,29} EXACTLY. 4*L layer specs + 1 head = kDecNumDwSpecs.
    for (int s = 0; s < 4 * dec::kLayers; ++s) {
        const int li = s / 4, kind = s % 4;
        const int woff = (kind == 0) ? 0 : (kind == 1) ? 2 : (kind == 2) ? 8 : 10;
        const int wi = 2 + 12 * li + woff;   // weight tensor index
        const int bi = wi + 1;               // bias tensor index
        DecDwSpec& sp = spec[s];
        sp.K = T; sp.grad_off = kDecOffsets[wi]; sp.bias_off = kDecOffsets[bi];
        if (kind == 0)      { sp.dY = acts.dY_qkv[li]; sp.X = acts.X_in[li];  sp.Nout = 3 * dec::kD; sp.Kin = dec::kD;   }
        else if (kind == 1) { sp.dY = acts.dY_a[li];   sp.X = acts.X_ctx[li]; sp.Nout = dec::kD;     sp.Kin = dec::kD;   }
        else if (kind == 2) { sp.dY = acts.dY_ff0[li]; sp.X = acts.X_x1[li];  sp.Nout = dec::kDff;   sp.Kin = dec::kD;   }
        else                { sp.dY = acts.dY_ff2[li]; sp.X = acts.X_gact[li];sp.Nout = dec::kD;     sp.Kin = dec::kDff; }
        sp.dY_bias = sp.dY;
    }
    DecDwSpec& hd = spec[4 * dec::kLayers];   // head spec (was spec[8] at L=2)
    hd.dY = acts.dY_logits; hd.X = acts.X_hn; hd.Nout = dec::kVocab; hd.Kin = dec::kD; hd.K = B;
    hd.grad_off = kDecOffsets[2 + 12 * dec::kLayers + 2];           // out.weight (28 at L=2)
    hd.bias_off = kDecOffsets[2 + 12 * dec::kLayers + 3];           // out.bias   (29 at L=2)
    hd.dY_bias = hd.dY;

    // CONTIGUOUS-TRANSPOSE bind (stage==1): pack each weight's dYt[Mpad,K] then
    // Xt[Kin,K] into dwt_base via the SAME running-offset walk dec_dw_transpose_elems
    // uses (so the kernel offsets == the host carve). On the scalar path (dwt_base
    // null / stage==0) leave dYt/Xt null → dectc_dw_run_tile takes the lambda gather.
    // #if-guarded (not if-constexpr): the dYt/Xt/t_off fields only EXIST when the
    // macro is set, so the scalar default's struct + smem are byte-identical.
#if SG_TUNED_DEC_DW_STAGE
    if (kDecDwTransposeActive && dwt_base != nullptr) {
        int64_t e = 0;
        for (int s = 0; s < kDecNumDwSpecs; ++s) {
            DecDwSpec& sp = spec[s];
            sp.t_off = e;
            sp.dYt   = dwt_base + e;                                  // [Mpad, K]
            sp.Xt    = dwt_base + e + dec_dw_mpad(sp.Nout) * sp.K;    // [Kin,  K]
            e += dec_dw_weight_t_elems(sp.Nout, sp.Kin, sp.K);
        }
    } else {
        for (int s = 0; s < kDecNumDwSpecs; ++s) { spec[s].dYt = nullptr; spec[s].Xt = nullptr; spec[s].t_off = 0; }
    }
#else
    (void)dwt_base;
#endif
}

// ── CONTIGUOUS-TRANSPOSE PASS (SG_TUNED_DEC_DW_STAGE=1). Grid-cooperatively
//    transpose each weight's dY[K,Nout] → dYt[Nout,K] and X[K,Kin] → Xt[Kin,K]
//    into the packed dW-T scratch the specs bind. ELEMENT-OWNED (one writer per
//    output element, fixed mapping) → DETERMINISTIC and A/A/A-safe; PURE bf16
//    COPY (dYt[m,k]=dY[k,m] — NO arithmetic, NO reduction) → fp64-oracle parity
//    is trivially preserved (the wgmma later consumes the SAME bf16 operand
//    values in the SAME ascending-k order; only their MEMORY LAYOUT changed from
//    transposed-strided gmem to K-contiguous gmem). Called by ALL CTAs after the
//    backward writes dY/X to acts and BEFORE the dW phase, fenced by the existing
//    B1 grid barrier (no new barrier). The total work is Σ_s (Nout_s+Kin_s)·K_s
//    bf16 stores, touched ONCE per step and reused across every dW work item +
//    split-K chunk of that weight. ──
__device__ __forceinline__ void dectc_dw_transpose_operands(
        const DecDwSpec spec[kDecNumDwSpecs], int cta, int nCTA) {
#if SG_TUNED_DEC_DW_STAGE
    if constexpr (!kDecDwTransposeActive) { (void)spec; (void)cta; (void)nCTA; return; }
    const int tpb = blockDim.x;
    const int64_t stride = (int64_t)nCTA * tpb;
    const int64_t lane0  = (int64_t)cta * tpb + threadIdx.x;
    for (int s = 0; s < kDecNumDwSpecs; ++s) {
        const DecDwSpec& sp = spec[s];
        if (sp.dYt == nullptr) continue;
        const int64_t K = sp.K, Nout = sp.Nout, Kin = sp.Kin;
        const int64_t Mpad = dec_dw_mpad(sp.Nout);
        // dYt[m,k] = dY[k,m] for m<Nout; pad rows [Nout,Mpad) = 0 (inert — the
        // ring reads A rows unguarded, so the row dim is padded to a 64-atom
        // boundary). Linear out-index i = m*K+k over the FULL padded [Mpad,K).
        const int64_t ndy = Mpad * K;
        for (int64_t i = lane0; i < ndy; i += stride) {
            const int64_t m = i / K, k = i % K;
            sp.dYt[i] = (m < Nout) ? sp.dY[k * Nout + m] : __float2bfloat16(0.f);
        }
        // Xt[n,k] = X[k,n], n in [0,Kin), k in [0,K). Linear out-index i = n*K+k.
        // (No row pad: the ring guards B rows via row_valid → pad N-tile cols are
        // zero-filled in smem, never read from Xt.)
        const int64_t nx = Kin * K;
        for (int64_t i = lane0; i < nx; i += stride) {
            const int64_t n = i / K, k = i % K;
            sp.Xt[i] = sp.X[k * Kin + n];
        }
    }
#else
    (void)spec; (void)cta; (void)nCTA;   // scalar default: no transpose pass
#endif
}

// ── dW M-atom INTERLEAVE group width (task #13 H3). A dW output "tile" (work
//    item) owns a GROUP of up to kDecDwIL consecutive m64 atoms instead of one,
//    so the engine runs kDecDwIL wgmmas/k-step into independent fragments sharing
//    ONE staged X (B-operand) tile → the H1 win (overlap the MMAs + kDecDwIL× less
//    X-operand HBM traffic) now applies to the dW K=T contraction (the post-H2
//    dominant phase). kDecDwIL == the engine's interleave cap kDecMaxIL so the
//    ring smem (DecTcSmem.sA, sized kDecAtomsPerSlot=min(kAtomsM,kDecMaxIL) A-tiles
//    per slot) and the selftest-validated dW path (gemm_dw_kernel: MaxAtomsM=8,
//    kIL=min(8,2)=2) match exactly. Ascending-k per atom is unchanged → numerics
//    bit-identical + A/A/A determinism. Atoms-per-weight-N-tile / groups: ──
constexpr int kDecDwIL = kDecMaxIL;
__device__ __host__ __forceinline__ int dec_dw_atoms(int Nout)  { return (Nout + 63) / 64; }
__device__ __host__ __forceinline__ int dec_dw_groups(int Nout) {
    return (dec_dw_atoms(Nout) + kDecDwIL - 1) / kDecDwIL;
}

// Total number of dW output WORK ITEMS (M-atom groups × N-tiles) across the 9
// weights (the tile loop count). Each item is a group of <= kDecDwIL m64 atoms.
template <int N>
__device__ __forceinline__ int dectc_dw_total_tiles(const DecDwSpec spec[kDecNumDwSpecs]) {
    int n = 0;
    for (int s = 0; s < kDecNumDwSpecs; ++s)
        n += dec_dw_groups(spec[s].Nout) * ((spec[s].Kin + N - 1) / N);
    return n;
}

// Run ONE global dW work item `gt` (if it belongs to this CTA): decode (weight,
// M-atom GROUP, N-tile), then contract K via the dW GEMM into grad[grad_off]. The
// group is up to kDecDwIL stacked m64 atoms → the engine interleaves them (kIL=
// kDecDwIL): kDecDwIL wgmmas/k-step into independent fragments sharing ONE staged
// X B-tile (H1's win, now on the dW K=T contraction).
template <int N>
__device__ __forceinline__ void dectc_dw_run_tile(
        const DecDwSpec spec[kDecNumDwSpecs], int gt, float* __restrict__ grad,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    // Decode gt → (s, m_group, n_tile).
    int acc = 0, s = 0, m_group = 0, n_tile = 0;
    for (s = 0; s < kDecNumDwSpecs; ++s) {
        const int ng = dec_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; break; }
        acc += ng * nt;
    }
    const DecDwSpec& sp = spec[s];
    const int n_atoms = dec_dw_atoms(sp.Nout);
    const int a0 = m_group * kDecDwIL;               // first atom of the group
    const int g_atoms = (n_atoms - a0) < kDecDwIL ? (n_atoms - a0) : kDecDwIL;
    const int mbase = a0 * 64;
    const int n0 = n_tile * N;
    const int n_real = (sp.Kin - n0) < N ? (sp.Kin - n0) : N;
    const int k_steps = sp.K / wgs::kWgmmaAtomK;     // K = T or B (must be /16; padded by caller)
    const int Nout = sp.Nout, Kin = sp.Kin;
    // out(m,n,v): m is GLOBAL (engine adds mbase0=mbase); identical on both paths.
    auto out  = [&] (int m, int n, float v) {
        if (m < Nout) grad[sp.grad_off + (int64_t)m * Kin + n0 + n] = v; };
    // ── SCALAR transposed-strided gather (DEFAULT path; lambda sources). The
    //    engine (tc_gemm_block_unpipelined, mbase0=mbase) passes the GLOBAL row
    //    m = mbase + (group-local row) to srcA/out (it adds mbase0 itself), so the
    //    accessors use `m` RAW (the selftest gemm_dw_kernel passes mbase0=0 + m
    //    raw; we shard atoms into groups, pass mbase0=mbase, likewise m raw).
    //    A[m=out,k=t]=dY[t,out], B[n=in,k=t]=X[t,in] — both TRANSPOSED-strided.
    //    Lambda sources ⇒ DecTileSrcIsGmem=false ⇒ the engine's kRingAsync gate is
    //    OFF ⇒ the synchronous 2-byte LDG→reg→STS stage_kmajor_tile. ──
    auto run_scalar = [&] {
        const __nv_bfloat16* dY = sp.dY; const __nv_bfloat16* X = sp.X;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
            return m < Nout ? dY[(int64_t)k * Nout + m] : __float2bfloat16(0.f); };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return nn < Kin ? X[(int64_t)k * Kin + nn] : __float2bfloat16(0.f); };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kDecDwIL>(
            mbase, g_atoms, n_real, k_steps, srcA, srcB, out, sA, sB);
    };
#if SG_TUNED_DEC_DW_STAGE
    if constexpr (kDecDwTransposeActive) {
        // DEFENSIVE: a caller that did NOT run dectc_dw_transpose_operands (e.g.
        // the PP-stage / SAM dW phases, which build specs via the 4-arg overload →
        // dYt/Xt null) falls back to the scalar gather rather than dereferencing a
        // null transpose scratch. The wired proof path (fused_decoder_megakernel_tc)
        // always populates dYt/Xt first, so it takes the fast contiguous branch.
        if (sp.dYt == nullptr || sp.Xt == nullptr) { run_scalar(); return; }
        // ── CONTIGUOUS-LAYOUT path: the operands were PRE-TRANSPOSED into K-major
        //    gmem (dYt[Mpad,K], Xt[Kin,K]) by dectc_dw_transpose_operands, so the
        //    A/B rows are now K-CONTIGUOUS + 16B-aligned. Feeding the engine the
        //    flat-gmem DecGmemTileSrc{A,B} (the SAME POD sources the fwd/dX
        //    wrappers use) FLIPS DecTileSrcIsGmem to true → the engine's proven
        //    cp.async ring (kRingAsync) streams them in 16-byte coalesced copies
        //    instead of stage_kmajor_tile's 2-byte transposed-strided gather. The
        //    wgmma issue sequence + ascending-k accumulation are UNCHANGED (the
        //    same A[m,k]=dY[k,m] / B[n,k]=X[k,n] operand values reach the MMA in
        //    the same order) → bit-identical to the scalar path. A's row dim is
        //    padded to Mpad (64-atom boundary) so the ring's UNGUARDED A reads
        //    stay in-bounds; B uses rows_valid=Kin-n0 (the ring zero-fills pad
        //    N-cols). ld = K (the transposed row stride in elements).
        const int K = (int)sp.K;
        DecGmemTileSrcA srcA{sp.dYt, K};                       // A[m,k] = dYt[m·K + k]
        DecGmemTileSrcB srcB{sp.Xt + (int64_t)n0 * K, K, Kin - n0};  // B[n,k] = Xt[(n0+n)·K + k]
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kDecDwIL>(
            mbase, g_atoms, n_real, k_steps, srcA, srcB, out, sA, sB);
    } else {
        // stage==1 but split-K active → the _splitk path handles the dW; this
        // single-CTA wrapper keeps the scalar gather (consistent + unused).
        run_scalar();
    }
#else
    run_scalar();
#endif
}

// ════════════════════════════════════════════════════════════════════════
//  SPLIT-K dW (multi-CTA tiling — the validated mamba fix). The 9 dW yield only
//  ~50 output tiles → ~62% of 132 SMs idle in P2. Split-K turns each output tile
//  into G work items (one per K-chunk), so the grid sees (n_dw·G) items → idle
//  SMs do work. Each CTA computes a PARTIAL over its K-chunk into a per-(tile,
//  chunk) scratch slot; a grid barrier; then a DETERMINISTIC ascending-chunk
//  reduce sums the G partials into grad — no float atomics, fixed order, so
//  parity + A/A/A bit-determinism hold (each partial is the SAME ascending-k fp32
//  wgmma accumulate; Σ_chunk == full-K sum reassociated into G fp32 blocks).
//  G==1 routes to the single-CTA dectc_dw_run_tile above (no scratch). Slot (gt,
//  kc) at dw_part[((gt*G+kc)) * (kDecDwIL·64·kDecMaxTileN) + lr*kDecMaxTileN + col]
//  — each work item gt is an M-atom GROUP (up to kDecDwIL=2 stacked m64 atoms, H3),
//  so its slot holds kDecDwIL·64 group-local rows (lr). Decoder K varies per spec:
//  layer dW K=T, head dW K=B (so kc_steps reads sp.K).
// ════════════════════════════════════════════════════════════════════════
constexpr int kDecMaxTileN = SG_TUNED_TILE_N;                       // widest dW N-tile
// Floats per (gt,kc) split-K slot: a work item is an M-atom GROUP of up to kDecDwIL
// stacked m64 atoms → kDecDwIL·64 rows × N cols. (Was 64·N for the 1-atom item.)
constexpr int kDecDwTileFloats = kDecDwIL * wgs::kWgmmaAtomM * kDecMaxTileN;

// COMPILE-TIME max #dW output WORK ITEMS (the 9 dW have fixed Nout/Kin; decoder
// dims are compile-time → constant). per layer: qkv(3d×d), attn_out(d×d), ff0(dff
// ×d), ff2(d×dff), N=kDecMaxTileN; + head(V×d). The M factor is now M-atom GROUPS
// of kDecDwIL (ceil(atoms/kDecDwIL)), matching dec_dw_groups — NOT single atoms.
constexpr int kDecDwMGroups(int Nout) {                 // ceil(ceil(Nout/64)/kDecDwIL)
    return (((Nout + 63) / 64) + kDecDwIL - 1) / kDecDwIL;
}
constexpr int kDecDwTilesPerLayer =
      kDecDwMGroups(3*dec::kD) * ((dec::kD  + kDecMaxTileN - 1)/kDecMaxTileN)   // qkv
    + kDecDwMGroups(dec::kD)   * ((dec::kD  + kDecMaxTileN - 1)/kDecMaxTileN)   // attn_out
    + kDecDwMGroups(dec::kDff) * ((dec::kD  + kDecMaxTileN - 1)/kDecMaxTileN)   // ff0
    + kDecDwMGroups(dec::kD)   * ((dec::kDff+ kDecMaxTileN - 1)/kDecMaxTileN);  // ff2
constexpr int kDecDwHeadTiles =
      kDecDwMGroups(dec::kVocab) * ((dec::kD + kDecMaxTileN - 1)/kDecMaxTileN);
constexpr int kDecDwMaxTiles = dec::kLayers * kDecDwTilesPerLayer + kDecDwHeadTiles;

// Split-K dW partial-scratch float count (host carves it from the workspace tail).
__host__ __device__ __forceinline__ int64_t dec_dw_part_floats(int G) {
    return (int64_t)kDecDwMaxTiles * G * kDecDwTileFloats;
}

// Decode global dW work-item index gt → (spec index s, m_group, n_tile). Single-
// source: m_group indexes M-atom GROUPS of kDecDwIL (group's first atom = m_group·
// kDecDwIL, base row = m_group·kDecDwIL·64).
template <int N>
__device__ __forceinline__ void dectc_dw_decode(
        const DecDwSpec spec[kDecNumDwSpecs], int gt, int& s, int& m_group, int& n_tile) {
    int acc = 0;
    for (s = 0; s < kDecNumDwSpecs; ++s) {
        const int ng = dec_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; return; }
        acc += ng * nt;
    }
    s = 4 * dec::kLayers; m_group = 0; n_tile = 0;   // unreachable (gt < n_dw); keep defined (head idx)
}

// PARTIAL dW for global tile gt over K-chunk kc of G → dw_part. K-chunk uses sp.K
// (layer T / head B). FLOOR-BALANCED partition: chunk kc = [k0,k1) with
// k0=floor(kc·KS/G), k1=floor((kc+1)·KS/G) — near-equal, summing to KS EXACTLY for
// ANY KS≥G (no `G | KS` requirement → works at the production truncated B=4176,
// where the head's KS=B/16 need NOT be divisible by G). A CEIL split would leave a
// trailing EMPTY chunk whose slot stays unwritten → the reduce sums garbage (the
// determinism-blind dW bug); floor never empties a chunk for KS≥G. Fresh ScaleD=0
// per chunk → true partial. Writes the full 64×N tile (LOCAL rows) to the slot.
template <int N>
__device__ __forceinline__ void dectc_dw_run_tile_splitk(
        const DecDwSpec spec[kDecNumDwSpecs], int gt, int kc, int G, float* __restrict__ dw_part,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    int s, m_group, n_tile;
    dectc_dw_decode<N>(spec, gt, s, m_group, n_tile);
    const DecDwSpec& sp = spec[s];
    const int n_atoms = dec_dw_atoms(sp.Nout);
    const int a0 = m_group * kDecDwIL;                      // first atom of the group
    const int g_atoms = (n_atoms - a0) < kDecDwIL ? (n_atoms - a0) : kDecDwIL;
    const int mbase = a0 * 64;
    const int n0 = n_tile * N;
    const int KS = sp.K / wgs::kWgmmaAtomK;                 // total k-atoms (T/16 or B/16)
    const int k0 = (int)(((int64_t)kc       * KS) / G);     // floor-balanced chunk bounds
    const int k1 = (int)(((int64_t)(kc + 1) * KS) / G);
    const int kc_steps = k1 - k0;                          // sums to KS exactly over kc
    // Slot holds the GROUP's kDecDwIL·64 rows (lr) × N cols (kDecDwTileFloats).
    float* slot = dw_part + ((int64_t)gt * G + kc) * kDecDwTileFloats;
    // Empty-chunk guard (KS<G, i.e. B<64): a k_steps=0 GEMM would emit the
    // uninitialized accumulator → zero the WHOLE group slot + return instead of
    // running it (the reduce sums all G slots unconditionally, so an empty chunk
    // MUST be 0). Zero the full kDecDwIL·64·N slot (only g_atoms·64 rows are valid
    // but the reduce reads only valid rows; zeroing all keeps it simple + safe).
    if (kc_steps <= 0) {
        for (int i = threadIdx.x; i < kDecDwTileFloats; i += blockDim.x) slot[i] = 0.0f;
        __syncthreads();
        return;
    }
    const int Nout = sp.Nout, Kin = sp.Kin;
    const __nv_bfloat16* dY = sp.dY; const __nv_bfloat16* X = sp.X;
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
        return m < Nout ? dY[(int64_t)(k0 * wgs::kWgmmaAtomK + k) * Nout + m] : __float2bfloat16(0.f); };
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
        int nn = n0 + n; return nn < Kin ? X[(int64_t)(k0 * wgs::kWgmmaAtomK + k) * Kin + nn] : __float2bfloat16(0.f); };
    // out(mbase+row, col, v): m is GLOBAL (srcA needs it; engine adds mbase0=mbase);
    // the slot holds the GROUP's kDecDwIL·64 LOCAL rows → index by (m - mbase). The
    // engine emits group-local rows in [0, g_atoms·64); writing them by (m-mbase)
    // fills the lower g_atoms·64 rows (a ragged final group's tail rows stay as the
    // zeroed/garbage upper rows but the reduce reads only valid rows → never used).
    auto out  = [&] (int m, int n, float v) {
        const int lr = m - mbase;
        if (lr >= 0 && lr < kDecDwIL * 64 && n < N) slot[(int64_t)lr * N + n] = v; };
    tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kDecDwIL>(
        mbase, g_atoms, /*n_real=*/N, kc_steps, srcA, srcB, out, sA, sB);
}

// Deterministic reduce: output tile gt (% nCTA) sums its G chunk-partials ascending-kc
// → grad. Same (gt → geometry) decode as the partial.
template <int N>
__device__ __forceinline__ void dectc_dw_reduce_splitk(
        const DecDwSpec spec[kDecNumDwSpecs], int n_dw, int G, const float* __restrict__ dw_part,
        float* __restrict__ grad, int cta, int nCTA) {
    for (int gt = cta; gt < n_dw; gt += nCTA) {
        int s, m_group, n_tile;
        dectc_dw_decode<N>(spec, gt, s, m_group, n_tile);
        const DecDwSpec& sp = spec[s];
        const int mbase = m_group * kDecDwIL * 64;          // group's first global row
        const int n0 = n_tile * N;
        const int n_real = (sp.Kin - n0) < N ? (sp.Kin - n0) : N;
        // Valid group rows: up to kDecDwIL·64, clamped to the weight's row count.
        const int grp_rows = kDecDwIL * 64;
        const int Nrow = (sp.Nout - mbase) < grp_rows ? (sp.Nout - mbase) : grp_rows;
        const int64_t base = (int64_t)gt * G * kDecDwTileFloats;
        for (int idx = threadIdx.x; idx < Nrow * n_real; idx += blockDim.x) {
            const int row = idx / n_real, col = idx % n_real;   // row = group-local lr
            float accv = 0.0f;
            for (int kc = 0; kc < G; ++kc)
                accv += dw_part[base + (int64_t)kc * kDecDwTileFloats + (int64_t)row * N + col];
            grad[sp.grad_off + (int64_t)(mbase + row) * sp.Kin + n0 + col] = accv;
        }
    }
}

// Biases db = Σ_K dY  (column-sum of dY[K,Nout] → [Nout], per output row).
//
// WAS: each CTA strided ALL bias outputs and ran the full Σ_K reduction — i.e. the
// ENTIRE column-sum was recomputed redundantly on ALL ~132 CTAs (the comment called
// it "cheap" — true only when Nout≤3d=384 at the d=128 production width; at the
// d=1024 roofline width Nout reaches 4d=4096 and K=T=65536, so the 132× redundant
// reduction was the DOMINANT grad_asm cost — ~500 ms, eclipsing even the embedding
// scan this task set out to fix; the per-phase profiler (ga.biases sub-timer) made
// it visible). HBM traffic was 132 × Σ_s K_s·Nout_s·2 B ≈ 70 GB.
//
// NOW: SINGLE OWNER per output element across the WHOLE grid. Flatten the 9 specs'
// outputs into one global index space [0, ΣNout) and grid-stride it over all CTAs
// × threads, so each bias output is reduced EXACTLY ONCE and the work is spread
// across the full grid. Traffic collapses to Σ_s K_s·Nout_s·2 B (one pass, ≈ 130×
// less at d=1024). Reads stay COALESCED (consecutive threads → consecutive o on the
// same k → consecutive dY addresses). DETERMINISM: one owner per output + the SAME
// ascending-k fp32 accumulation → bit-identical to the old per-output sum, no atomics.
//
// PORTABLE: the "single-owner grid-stride column-sum" is the general fix for any
// bias/reduction-to-a-vector that a megakernel was recomputing per-CTA; vit/mamba
// bias grads (same db = Σ_K dY shape) reuse this verbatim.
__device__ __forceinline__ void dectc_dw_biases(
        const DecDwSpec spec[kDecNumDwSpecs], float* __restrict__ grad, int cta, int nCTA) {
    // exclusive prefix of Nout across the specs → total bias-output count.
    int pre[kDecNumDwSpecs + 1];
    pre[0] = 0;
    #pragma unroll
    for (int s = 0; s < kDecNumDwSpecs; ++s) pre[s + 1] = pre[s] + spec[s].Nout;
    const int total = pre[kDecNumDwSpecs];
    const int stride = nCTA * blockDim.x;
    for (int go = cta * blockDim.x + threadIdx.x; go < total; go += stride) {
        // decode global output index → (spec s, local row o). #specs → linear scan.
        int s = 0;
        #pragma unroll
        for (int t = 0; t < kDecNumDwSpecs; ++t) if (go >= pre[t + 1]) s = t + 1;
        const DecDwSpec& sp = spec[s];
        const int o = go - pre[s];
        float accv = 0.0f;
        for (int k = 0; k < sp.K; ++k) accv += __bfloat162float(sp.dY_bias[(int64_t)k * sp.Nout + o]);
        grad[sp.bias_off + o] = accv;
    }
}

// ════════════════════════════════════════════════════════════════════════
//  P2 — EMBEDDING grad (tok/pos). dh0 [T,d] (bf16) holds grad wrt h0.
//
//  WAS (the owner-LINEAR-SCAN, task #13 H2 target): each of V owner-CTAs scanned
//  ALL T tokens (`if tok_ids[t]==r` per token), re-reading the WHOLE dh0 stream V
//  times — O(V·d·T) work + V passes over 134 MB of dh0, where O(d·T) (ONE pass)
//  suffices. V=99<132 left 33 CTAs idle; threads beyond d idled per owner; pos
//  used only S=4 owner-CTAs scanning all T with a t%S branch.
//
//  NOW (H2 — counting-sort token lists + flat-element grid-stride):
//    BUILD (dectc_embed_build_lists, once, cta 0, in the P1 window so it overlaps
//    fwd/bwd and adds NO barrier — B1 already fences it before P2 consumes):
//    a deterministic integer counting sort over tok_ids → row_start[V+1] (CSR
//    offsets) + perm[T] (token positions bucketed by vocab row, ASCENDING t
//    within each row, because the scatter visits t ascending and appends).
//    CONSUME (dectc_embed_owner_scan): a flat grid-stride over ALL V·d output
//    elements. Element (r,j) walks ONLY row r's own tokens perm[row_start[r] ..
//    row_start[r+1]) in ascending-t order, reading dh0[perm[i]·d + j]. Total
//    work collapses to O(d·T) (each dh0 value read once in aggregate); reads are
//    COALESCED (consecutive threads = consecutive j on the same token row); ALL
//    132 CTAs + full blockDim are busy (V·d=101376 elems ≫ 132·256). pos is the
//    same flat grid-stride but needs NO list: tokens with (t%S)==p are exactly
//    t = p, p+S, …, p+(B-1)·S (T=B·S), a closed-form ascending walk of length B.
//
//  DETERMINISM: per row the accumulation order over its tokens is the SAME fixed
//  ascending-t order as the old scan (perm is built ascending; pos walk is
//  ascending) → bitwise-identical fp32 accumulation, no float atomics, no timing-
//  dependent order. The build is integer-exact (histogram counts + prefix +
//  serial ascending scatter), so row_start/perm are bit-identical every rerun.
//
//  PORTABLE: this CSR-bucket + flat-element grid-stride is the general pattern for
//  any "gather grad into a small set of embedding/lookup rows" — vit patch-proj /
//  pos-embed rows and mamba's embedding will reuse dectc_embed_build_lists +
//  this consume verbatim (only the row count V and the membership map differ; a
//  structural map like pos needs no list at all).
// ════════════════════════════════════════════════════════════════════════

// Scratch float count for the embedding token lists (host carves it from the
// workspace tail). row_start[V+1] + perm[T], stored as int32 (1 float slot each).
__host__ __device__ __forceinline__ int64_t dec_embed_lists_floats(int T) {
    return (int64_t)(dec::kVocab + 1) + (int64_t)T;
}

// BUILD the per-vocab-row token lists (counting sort). Single CTA (caller guards
// cta==0); runs in the P1 window so it overlaps fwd/bwd and the existing B1
// barrier fences it before the P2 consume — NO new barrier. `row_start` is
// [V+1] int32 (CSR offsets), `perm` is [T] int32 (token positions, ascending t
// within each vocab-row bucket). All integer ops → bit-exact + deterministic.
//
// COST: O(T), PARALLEL over kW worker lanes (HBM-latency hidden) so the build is a
// few hundred µs — far below the consume saving it enables. DETERMINISM via a fixed
// STRUCTURAL decomposition: worker w owns the CONTIGUOUS t-chunk [w·C,(w+1)·C); it
// histograms then scatters its chunk in ascending t into a per-(worker,row) slice
// of perm whose base sits AFTER all lower-w workers' slices for the same row. Lower
// w ⇒ lower t, and within a worker ascending t ⇒ each row's perm bucket is GLOBALLY
// ascending t — the SAME order the old single-cursor scatter (and the old owner-scan
// accumulation) produced, so bit-identical. All integer ops; no atomics on the hot
// path. tok_ids is HBM int32 [T]. row_start is [V+1] (CSR offsets); perm is [T].
__device__ __forceinline__ void dectc_embed_build_lists(
        const int* __restrict__ tok_ids, int T,
        int* __restrict__ row_start, int* __restrict__ perm) {
    constexpr int kW = 64;                         // worker lanes (latency hiding)
    __shared__ int wcnt[kW * dec::kVocab];         // per-(worker,row) count, then base cursor
    // floor-balanced contiguous t-chunk per worker w: [c0(w), c0(w+1)).
    auto c0 = [&] (int w) -> int { return (int)(((int64_t)w * T) / kW); };
    // 1) zero the per-worker histograms (all threads).
    for (int i = threadIdx.x; i < kW * dec::kVocab; i += blockDim.x) wcnt[i] = 0;
    __syncthreads();
    // 2) each of the first kW threads histograms its contiguous t-chunk into its own
    //    row of wcnt (no atomics — private per worker).
    if (threadIdx.x < kW) {
        const int w = threadIdx.x, e0 = c0(w), e1 = c0(w + 1);
        int* my = wcnt + (int64_t)w * dec::kVocab;
        for (int t = e0; t < e1; ++t) {
            const int r = tok_ids[t];
            if (r >= 0 && r < dec::kVocab) my[r]++;
        }
    }
    __syncthreads();
    // 3) exclusive prefix over r → row_start[V+1] (single thread; V tiny). totals[r]
    //    = Σ_w wcnt[w][r]. row_start is exclusive over rows (CSR).
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int r = 0; r < dec::kVocab; ++r) {
            row_start[r] = acc;
            for (int w = 0; w < kW; ++w) acc += wcnt[(int64_t)w * dec::kVocab + r];
        }
        row_start[dec::kVocab] = acc;   // == #tokens with a valid row (≤ T)
    }
    __syncthreads();
    // 4) per-(worker,row) base cursor: wcnt[w][r] ← row_start[r] + Σ_{w'<w} cnt[w'][r]
    //    (ascending-worker prefix WITHIN a row → lower-t chunks land first). One
    //    thread per row (V≤256 → fits blockDim) scans workers ascending.
    if (threadIdx.x < dec::kVocab) {
        const int r = threadIdx.x;
        int base = row_start[r];
        for (int w = 0; w < kW; ++w) {
            int* slot = &wcnt[(int64_t)w * dec::kVocab + r];
            const int c = *slot;
            *slot = base;        // becomes the live append cursor for (w,r)
            base += c;
        }
    }
    __syncthreads();
    // 5) scatter: each worker walks its chunk ascending t, appending t to perm at
    //    its per-row cursor. Ascending t within the chunk + ascending-worker bases
    //    ⇒ globally ascending t per row bucket.
    if (threadIdx.x < kW) {
        const int w = threadIdx.x, e0 = c0(w), e1 = c0(w + 1);
        int* cur = wcnt + (int64_t)w * dec::kVocab;
        for (int t = e0; t < e1; ++t) {
            const int r = tok_ids[t];
            if (r >= 0 && r < dec::kVocab) perm[cur[r]++] = t;
        }
    }
    __syncthreads();
}

// CONSUME: assemble tok + pos embedding grads from the prebuilt lists. Flat grid-
// stride over V·d (tok) and S·d (pos) output elements → all CTAs + threads busy,
// coalesced dh0 reads, fixed ascending-t accumulation per row (deterministic).
__device__ __forceinline__ void dectc_embed_owner_scan(
        const DecActs& acts, const int* __restrict__ row_start,
        const int* __restrict__ perm, int T,
        float* __restrict__ grad, int cta, int nCTA) {
    const int tok_off = kDecOffsets[0];   // tok.weight [V,d]
    const int pos_off = kDecOffsets[1];   // pos.weight [S,d]
    const __nv_bfloat16* __restrict__ dh0 = acts.dh0;
    const int stride = nCTA * blockDim.x;
    const int base   = cta * blockDim.x + threadIdx.x;
    // ── tok grad: element (r,j) over V·d, grid-strided. Walk ONLY row r's tokens
    //    (perm[row_start[r] .. row_start[r+1])) ascending → coalesced dh0 column
    //    read across the warp (consecutive j on one token row). ──
    const int64_t Vd = (int64_t)dec::kVocab * dec::kD;
    for (int64_t e = base; e < Vd; e += stride) {
        const int r = (int)(e / dec::kD);
        const int j = (int)(e - (int64_t)r * dec::kD);
        const int s0 = row_start[r], s1 = row_start[r + 1];
        float accv = 0.0f;
        for (int i = s0; i < s1; ++i)
            accv += __bfloat162float(dh0[(int64_t)perm[i] * dec::kD + j]);
        grad[tok_off + e] = accv;
    }
    // ── pos grad: element (p,j) over S·d, grid-strided. Tokens with (t%S)==p are
    //    t = p, p+S, …, p+(B-1)·S (T=B·S) — closed-form ascending walk, no list. ──
    const int64_t Sd = (int64_t)dec::kSeq * dec::kD;
    const int B = T / dec::kSeq;
    for (int64_t e = base; e < Sd; e += stride) {
        const int p = (int)(e / dec::kD);
        const int j = (int)(e - (int64_t)p * dec::kD);
        float accv = 0.0f;
        int t = p;
        for (int i = 0; i < B; ++i, t += dec::kSeq)
            accv += __bfloat162float(dh0[(int64_t)t * dec::kD + j]);
        grad[pos_off + e] = accv;
    }
}

// ════════════════════════════════════════════════════════════════════════
//  P2 — LN-vector grad reduce. The 10 γ/β grads were accumulated tile-locally
//    into each CTA's lnvec partials [kNumLnVec × d]; sum across CTAs in
//    ASCENDING CTA index (deterministic) into the 10 dec_layout slots of grad.
//    `lnvec_base` is the start of the [nCTA × kLnVecElems] partial region.
// ════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void dectc_lnvec_reduce(
        const float* __restrict__ lnvec_base, float* __restrict__ grad,
        int nCTA, int cta) {
    // Each CTA reduces a subset of the LN tensors (round-robin by tensor).
    for (int v = cta; v < kNumLnVec; v += nCTA) {
        const int goff = dec_lnvec_tensor_idx(v);   // was kLnVecTensorIdx[v]
        const int64_t gbase = kDecOffsets[goff];
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float accv = 0.0f;
            for (int c = 0; c < nCTA; ++c)
                accv += lnvec_base[(int64_t)c * kLnVecElems + (int64_t)v * dec::kD + j];
            grad[gbase + j] = accv;
        }
    }
}

}  // namespace dectc

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGE_DECODER_TC_CUH_
