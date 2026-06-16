#ifndef SG_FUSED_SM90_MODEL_STAGE_MAMBA_TC_CUH_
#define SG_FUSED_SM90_MODEL_STAGE_MAMBA_TC_CUH_
// ============================================================================
// csrc/fused/sm_90/model_stage_mamba_tc.cuh — R2 TENSOR-CORE variant of the
// L3-REAL Mamba (selective-SSM) fwd+bwd. This is the sample-tiled bf16 wgmma
// path (DESIGN-TC-PIPELINE.md Fork B, the Mamba counterpart of the decoder's
// model_stage_decoder_tc.cuh), a TUNED VARIANT compiled ALONGSIDE the scalar
// model_stage_mamba3.cuh and selected per-cell by SG_TUNED_GEMM_IMPL (BOTH
// paths compiled, the tuner picks). The scalar path's math + gates are
// UNCHANGED; nothing here edits model_stage_mamba3.cuh.
//
// WHAT IS TENSOR-CORE (the contract): ONLY the FOUR projection GEMMs and their
// dX/dW — in_proj (d=128 → 2*d_inner=512), x_proj (d_inner=256 → dbc=40, N
// padded to 64), dt_proj (dt_rank=8 → d_inner=256, K padded to 16), out_proj
// (d_inner=256 → d=128). The selective scan stays SCALAR register-resident (h
// in regs, t=0..7 unroll, reverse-time bwd) and the depthwise conv1d stays
// SCALAR — both REUSED verbatim from model_stage_mamba3.cuh (that reuse IS the
// "stays scalar" deliverable). LayerNorm / gate / skip / head / CE are fp32.
//
// WHY M-BATCHING WORKS (the decoder's trick, transplanted): the wgmma atom is
// m64nNk16 — it needs >= 64 M-rows, but one Mamba sample is kSeq=8 rows. So a
// tile batches kSamplesPerTile = kTileM/kSeq samples. SG_TUNED_TILE_M is the
// shared substrate knob (wgmma.cuh defaults it to 128 → 16 samples → kAtomsM=2
// stacked m64 atoms, same as the decoder); a tuner can drop it to 64 (8 samples,
// kAtomsM=1). Either is sample-aligned (kTileM % kSeq == 0). The PROJECTIONS see
// M=kTileM token rows; the SCAN/conv loop si over the tile's samples, each running the scalar
// per-sample reference over a one-sample smem LayerAct staged from HBM (so the
// 8×2-layer activation set lives in HBM tile-scratch, NOT smem — the scalar
// path's ~142 KB dynamic smem held ONE sample; eight cannot fit). Batching
// samples in M does NOT touch the scan's per-sample-sequential / per-channel-
// parallel structure, so the scan keeps its scalar register budget.
//
// FORK B (dW-output-stationary): P1 sample-tile-parallel fwd+bwd writes the
// projection X-inputs (bf16 dW operands) and dY-adjoints (bf16) to an HBM acts
// buffer; P2 owns each projection dW = dYᵀ@X tile (K = T = B·kSeq) on wgmma,
// no [nCTA×total] partial. The grads the stateful scan/conv produce (A_log, D,
// conv_w/b) plus the LN vectors + the SCALAR head (out.w/out.b) are NOT
// contractions of two stored activations — they come out of the reverse-time
// recurrence in P1, so they accumulate tile-local into a SMALL per-CTA partial
// reduced ascending-CTA in P2 (the decoder's kLnVec mechanism, widened). The
// GEMM biases reduce to ONE: dt_proj.bias = Σ_T ddt_pre (free P2 bias pass);
// in/x/out_proj have no bias (oracle linear_forward(x,W)). tok/pos via an
// owner-scan over dh0. No atomics, fixed order → deterministic A/A/A.
//
// PRECISION FORK (the bf16-faithful boundary — pinned to the oracle):
//   * the 4 projections (fwd/dX/dW): bf16 operands, wgmma, fp32 accumulator.
//   * x_main / dt_raw stay FP32 into the scan/conv/gate/D-skip; ONLY the bf16
//     COPY consumed by x_proj (and held as that GEMM's dW X-operand) is rounded.
//   * the layer input is bf16 for BOTH in_proj and the residual add (mirrors
//     the decoder's X_in bf16 at model_stage_decoder_tc.cuh:627).
//   * LayerNorm, SiLU, softplus, the selective scan, conv1d, gate, skip, head,
//     CE: fp32 (identical to the scalar oracle math).
//
// PORTABILITY: arch-guarded on __CUDA_ARCH__ >= 900 (the GEMM engine). Reuses
// the scalar header for DInner/DState math + the scan/conv/LN device fns.
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/mamba3_layout.cuh"
#include "csrc/fused/sm_90/model_stage_mamba3.cuh"   // reuse mb::dims, MambaWeights/Grad, scan/conv/LN
#include "csrc/backends/cuda/sm_90/wgmma.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math_constants.h>

namespace sg { namespace fused { namespace sm90 {

namespace wgs = ::sg::sm90::wgs;

// ── Tunable knobs. #ifndef defaults compose a correct untuned kernel. NOTE:
//    wgmma.cuh (included above) already #defines SG_TUNED_TILE_M (default 128),
//    so these #ifndef guards only fire if neither the tuner nor the substrate
//    set them. The static_asserts below pin whatever value wins to a legal tile
//    (multiple of 64 AND of kSeq=8). 64 (kAtomsM=1) is the simplest legal Mamba
//    tile; 128 (kAtomsM=2, the substrate default) is also legal. ──────────────
#ifndef SG_TUNED_TILE_M
#define SG_TUNED_TILE_M 64
#endif
#ifndef SG_TUNED_TILE_N
#define SG_TUNED_TILE_N 128
#endif

// ── PROFILING-ONLY phase-bypass switches (default 0 = OFF → shipped path
//    unchanged; the static-default kernel is byte-identical to the un-flagged
//    one). A differential-bypass profiler sets exactly one of these via -D to
//    NULL OUT a single phase's work and measure the wall it accounts for (ncu
//    SpeedOfLight is blocked in this container — ERR_NVGPUCTRPERM). These guard
//    ONLY whether the work executes; they NEVER change the shipped (all-0)
//    kernel, so they cannot affect correctness on any wired path. ─────────────
#ifndef SG_MBTC_BYPASS_DW_GEMM
#define SG_MBTC_BYPASS_DW_GEMM 0      // P2 output-stationary dW = dYᵀ@X (K=T)
#endif
#ifndef SG_MBTC_BYPASS_SCAN
#define SG_MBTC_BYPASS_SCAN 0         // per-sample selective-scan fwd+bwd
#endif
#ifndef SG_MBTC_BYPASS_EMBED_SCAN
#define SG_MBTC_BYPASS_EMBED_SCAN 0   // P2 tok/pos owner-scan (O(vocab·T))
#endif
#ifndef SG_MBTC_BYPASS_PROJ_FWD
#define SG_MBTC_BYPASS_PROJ_FWD 0     // the 4 in-tile fwd/dX projection GEMMs
#endif

namespace mbtc {

// Sample-tile rows a CTA owns. Must be a multiple of 64 (wgmma atom M) AND of
// kSeq=8 (so a tile boundary is a sample boundary). kTileM=64 → 8 samples → one
// m64 atom (kAtomsM=1) — simpler than the decoder's stacked atoms.
constexpr int kTileM = SG_TUNED_TILE_M;
static_assert(kTileM % wgs::kWgmmaAtomM == 0,
              "SG_TUNED_TILE_M must be a multiple of 64 (wgmma m64 atom)");
static_assert(kTileM % mb::kSeq == 0,
              "SG_TUNED_TILE_M must be a multiple of kSeq=8 (tile=sample boundary)");
constexpr int kAtomsM = kTileM / wgs::kWgmmaAtomM;
constexpr int kSamplesPerTile = kTileM / mb::kSeq;

// ════════════════════════════════════════════════════════════════════════
//  Canonical Major-K smem stager (identical to the decoder TC's; the decoder
//  header is sibling-owned and off-limits to include, so the validated helper
//  is restated here — same INTERLEAVE layout the wgmma.cuh make_desc assumes:
//      idx(mn,k) = (k/8)*(MN*8) + mn*8 + (k%8)
//  Routes the source axis through an accessor so the SAME TransA=0/TransB=0
//  issue serves fwd / dX / dW; the staging loop transposes physically. ──────
template <int MN, typename Src>
__device__ __forceinline__ void stage_kmajor_tile(
        __nv_bfloat16* smem_tile, int kbase, Src src, int t0, int nthreads) {
    #pragma unroll 1
    for (int i = t0; i < MN * wgs::kWgmmaAtomK; i += nthreads) {
        const int mn = i / wgs::kWgmmaAtomK;
        const int k  = i % wgs::kWgmmaAtomK;
        const int dst = (k >> 3) * (MN * 8) + mn * 8 + (k & 7);
        smem_tile[dst] = src(mn, kbase + k);
    }
}

// ════════════════════════════════════════════════════════════════════════
//  SMEM DOUBLE-BUFFER tile bytes (per operand stage). kMbTcSmem{A,B}1 is one
//  staged A (64×16) / B (N×16) bf16 tile; the kernel allocates kMbTcStages of
//  each so the K-loop can stage tile k+1 into the OTHER slot while the wgmma on
//  tile k is async-resident → the HBM operand loads (the dW K-loop is HBM-acts-
//  latency bound) overlap the MMA. S=2 = ONE wgmma in flight (per-iter
//  wait_group<0>); a 3rd stage is dead smem under that wait (a multi-in-flight
//  ring needs gated cp.async staging — a much bigger rewrite for little gain).
//  S=1 reproduces the old fully-serial path BIT-FOR-BIT (the #if escape hatch;
//  same ascending-k fp32 accumulate, so parity/determinism are UNCHANGED).
// ════════════════════════════════════════════════════════════════════════
#ifndef SG_TUNED_MB_GEMM_STAGES
#define SG_TUNED_MB_GEMM_STAGES 2
#endif
constexpr int kMbTcStages = SG_TUNED_MB_GEMM_STAGES;
static_assert(kMbTcStages >= 1 && kMbTcStages <= 2,
              "SG_TUNED_MB_GEMM_STAGES must be 1 (serial) or 2 (double-buffer)");

// ── M-atom INTERLEAVE width cap (task #24 — the decoder H1+H3 hill-climb win,
//    ported to the Mamba twin). The GEMM microkernel processes stacked m64 atoms in
//    groups of min(MaxAtomsM, this); within a group the per-k wgmmas (one per atom)
//    issue back-to-back into INDEPENDENT fp32 fragments sharing ONE staged B-tile →
//    the tensor pipe overlaps the MMAs AND the (HBM-bound) B-operand tile is staged
//    once per group instead of per atom. Capped (default 2) so the accumulator-
//    register + A-smem cost stays bounded (Mamba's production tile is kAtomsM=2 for
//    fwd/dX, and a dW tile up to 8 atoms; capping at 2 keeps acc[kIL] at 2 fragments
//    = N regs). 1 = no interleave (the old serial path, bit-for-bit). Per-atom
//    accumulation stays ASCENDING-k → numerics bit-identical + A/A/A determinism.
//    CAVEAT (task #24): the Mamba TC megakernel is occupancy-1 at 255 regs with a
//    thin spill margin — if the dW 2-wide fragment pushes ptxas past the budget into
//    refused occupancy, fall back to kIL=1 for Mamba (this knob = 1).
#ifndef SG_TUNED_MB_GEMM_INTERLEAVE
#define SG_TUNED_MB_GEMM_INTERLEAVE 2
#endif
constexpr int kMbMaxIL = SG_TUNED_MB_GEMM_INTERLEAVE;
static_assert(kMbMaxIL >= 1 && kMbMaxIL <= 4,
              "SG_TUNED_MB_GEMM_INTERLEAVE must be 1 (serial) .. 4");

// SPLIT-K factor for the P2 projection dW (the named "multi-CTA tiling"). The 8
// dW yield only ~36 output tiles → ~73% of 132 SMs idle in P2; G chunks the K=T
// contraction so the grid sees n_dw·G items. G=4 → ~144 items ≈ full grid (one
// per SM). MUST divide k_steps_total = T/16 (the launcher enforces G | (T/16);
// since T=B·8 and B%16==0, T/16=B/2 is divisible by 4 for any B%16==0). G=1 ==
// the old single-CTA-per-tile path (no scratch, no reduce — byte-identical).
#ifndef SG_TUNED_MB_DW_SPLITK
#define SG_TUNED_MB_DW_SPLITK 4
#endif
constexpr int kMbDwSplitK = SG_TUNED_MB_DW_SPLITK;
static_assert(kMbDwSplitK >= 1, "SG_TUNED_MB_DW_SPLITK must be >= 1");
constexpr int kMbTcSmemA1 = wgs::kWgmmaAtomM * wgs::kWgmmaAtomK;   // 64*16 bf16
constexpr int kMbTcSmemB1 = SG_TUNED_TILE_N * wgs::kWgmmaAtomK;    // N*16 bf16
// A-tiles the ring holds PER SLOT: kMbAtomsPerSlot = min(kAtomsM, kMbMaxIL) = the
// widest interleave group any call site uses (fwd/dX = kAtomsM=2; dW = kMbMaxIL).
// The M-atom-interleaved engine stages this many A-tiles per ring slot (slot sl,
// atom-in-group ai at smemA + (sl*kMbAtomsPerSlot + ai)*kMbTcSmemA1); the megakernel
// arena (MbTcSmem.sA) is sized kMbTcStages·kMbAtomsPerSlot tiles.
constexpr int kMbAtomsPerSlot = (kAtomsM < kMbMaxIL) ? kAtomsM : kMbMaxIL;

// ════════════════════════════════════════════════════════════════════════
//  MUON 2D-WEIGHT TABLE — the ndim==2 mamba weights orthogonalized by the
//  in-kernel grid-cooperative Newton-Schulz P2.7 phase (Muon ONLY). Mirrors
//  the decoder twin's kDecMuon2D (model_stage_decoder_tc.cuh) and the vit twin's
//  kVitMuon2D. Muon auto-splits params by p.ndim: ndim==2 → NS, everything else →
//  AdamW (muon.py:91-98 _split_by_ndim; muon.h:75-76). For the small Mamba the
//  ndim==2 weights are EXACTLY these 13 (the flat named_parameters() tensor index
//  + rows[dim0] + cols[dim1], matching the LIVE model's p.shape EXACTLY — verified
//  against tests/hw/mamba_oracle.py::mamba_param_spec, the SAME source the layout
//  header mamba3_layout.cuh static_asserts == named_parameters()). All ndim==1
//  weights (D, conv1d.bias, dt_proj.bias, norm γ/β, out.bias) AND the ndim==3
//  conv1d.weight [256,1,3] take the AdamW aux tail in P3. The kernel's Muon P2.7
//  loops THIS table running the grid-cooperative NS per matrix; P3 routes tensor t
//  to the NS apply iff it is in the table, else the AdamW aux tail. Indices MUST
//  match mamba3_layout / named_parameters() order. ──
constexpr int kMbNumMuon2D = 13;
struct MbMuon2D { int tidx; int rows; int cols; };
__device__ __constant__ MbMuon2D kMbMuon2D[kMbNumMuon2D] = {
    { 0, mb::kVocab,      mb::kD       },   // tok.weight            [99,128]
    { 1, mb::kSeq,        mb::kD       },   // pos.weight            [8,128]
    { 2, mb::kDInner,     mb::kState   },   // L0 A_log              [256,16]
    { 4, 2*mb::kDInner,   mb::kD       },   // L0 in_proj.weight     [512,128]
    { 7, mb::kDbc,        mb::kDInner  },   // L0 x_proj.weight      [40,256]
    { 8, mb::kDInner,     mb::kDtRank  },   // L0 dt_proj.weight     [256,8]
    {10, mb::kD,          mb::kDInner  },   // L0 out_proj.weight    [128,256]
    {13, mb::kDInner,     mb::kState   },   // L1 A_log              [256,16]
    {15, 2*mb::kDInner,   mb::kD       },   // L1 in_proj.weight     [512,128]
    {18, mb::kDbc,        mb::kDInner  },   // L1 x_proj.weight      [40,256]
    {19, mb::kDInner,     mb::kDtRank  },   // L1 dt_proj.weight     [256,8]
    {21, mb::kD,          mb::kDInner  },   // L1 out_proj.weight    [128,256]
    {26, mb::kPHead,      mb::kD       },   // out.weight            [97,128]
};
// Is tensor index `t` one of the Muon 2D matrices (orthogonalized in P2.7)? P3
// uses this to route ONLY the 1D / non-2D weights to the AdamW aux tail for Muon.
__device__ __forceinline__ bool mb_is_muon_2d(int t) {
    #pragma unroll
    for (int mi = 0; mi < kMbNumMuon2D; ++mi) if (kMbMuon2D[mi].tidx == t) return true;
    return false;
}
// Largest 2D weight (numel) + largest #rows over the table — sizes the per-matrix
// NS scratch (the stage runs ONE matrix at a time, reusing the buffers). in_proj
// [512,128]=65536 is the largest numel; in_proj rows=512 is the largest #rows
// (A=XXᵀ is rows×rows). Mirrors dec's kDecMuonMaxNumel/kDecMuonMaxRows.
constexpr int kMbMuonMaxNumel = 2 * mb::kDInner * mb::kD;   // 512*128 = 65536 (in_proj)
constexpr int kMbMuonMaxRows  = 2 * mb::kDInner;            // 512 (in_proj rows)

// ════════════════════════════════════════════════════════════════════════
//  SOFTWARE-PIPELINED single-CTA batch GEMM: D[M,N] = Σ_k A[m,k]·B[n,k], bf16
//  operands (accessor-sourced + Major-K staged), fp32 accumulator, ASCENDING-k.
//  One warpgroup (threads 0..127) issues the wgmma; all `nthreads` stage. The
//  staging of tile k+1 OVERLAPS the async wgmma on tile k (a kMbTcStages-deep
//  smem ring buffer hides the HBM-acts latency the dW K-loop (K=T=B·kSeq) is
//  bound by — every k-step pulls fresh operands from HBM). NUMERICS are
//  byte-identical to the old serial form: the accumulator is the SAME fp32
//  WgmmaAccum, issued in the SAME ascending-k order with ScaleD=0 on k=0 then
//  ScaleD=1 — only the OVERLAP of independent stages changes, never the math.
//  So the 0.08 per-tensor parity AND A/A/A bit-determinism hold by construction.
//  (Kept the name `tc_gemm_block_unpipelined` so the dozens of call sites + the
//  decoder-mirror lineage are untouched; "unpipelined" is now historical.)
//  MaxAtomsM bounds the register accumulator (compile-time). n_real<=N
//  suppresses ragged pad columns. smemA/smemB are the RING BASES (kMbTcStages
//  tiles each, contiguous): slot s lives at smemA + s*kMbTcSmemA1.
// ════════════════════════════════════════════════════════════════════════
template <int N, int MaxAtomsM, typename SrcA, typename SrcB, typename Out>
__device__ void tc_gemm_block_unpipelined(
        int mbase0, int m_atoms, int n_real, int k_steps,
        SrcA srcA, SrcB srcB, Out out,
        __nv_bfloat16* smemA, __nv_bfloat16* smemB) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const bool in_wg0 = (tid < 128);
    const int tid_wg = tid & 127;
    constexpr int S = kMbTcStages;

    // ── M-ATOM-INTERLEAVED wgmma pipeline (task #24, the decoder H1 win ported to
    //    the Mamba twin; the Mamba engine already had the S-stage ring so this is a
    //    near-verbatim port). The OLD body ran each stacked m64 atom's k-chain to
    //    completion SEQUENTIALLY (atom0 chain → atom0 epilogue → atom1 chain → …),
    //    re-staging the shared B (operand) tile ONCE PER ATOM. Here atoms are
    //    processed in GROUPS of kIL (= min(MaxAtomsM, kMbMaxIL)). Within a group each
    //    k-step issues the kIL wgmmas (one per atom) BACK-TO-BACK into their OWN fp32
    //    fragments, sharing ONE staged B-tile, before the single per-k wait. Two
    //    wins: (1) the kIL atoms are INDEPENDENT (distinct M-rows / accumulators) →
    //    the tensor pipe overlaps their MMA execution; (2) the B-tile is staged ONCE
    //    per group instead of kIL times → the (HBM-bound) operand traffic drops kIL×.
    //    kIL is CAPPED (kMbMaxIL=2) so the register/smem cost is bounded regardless of
    //    m_atoms (a dW tile can be 8 atoms). Groups reuse the SAME ring slots
    //    sequentially. Per-atom accumulation stays ASCENDING-k (k=0 overwrite, k>0
    //    add) → numerics bit-identical + A/A/A determinism UNCHANGED. Ring stages kIL
    //    A-tiles (slot sl, atom-in-group ai at +ai·kMbTcSmemA1 within the slot) + ONE
    //    shared B-tile; smemA must hold kMbAtomsPerSlot·kMbTcStages tiles.
    constexpr int kIL = (MaxAtomsM < kMbMaxIL) ? MaxAtomsM : kMbMaxIL;
    wgs::WgmmaAccum<N> acc[kIL];                 // kIL live fragments per group

    // Stage k-tile for a group of `g_atoms` (<= kIL) atoms based at `gbase`: the
    // g_atoms A-tiles (rows gbase+ai·64) + the shared B-tile, into ring slot k % S.
    auto stage_k = [&] (int gbase, int g_atoms, int k) {
        const int s = k % S;
        for (int ai = 0; ai < g_atoms; ++ai) {
            const int mbase = gbase + ai * wgs::kWgmmaAtomM;
            stage_kmajor_tile<wgs::kWgmmaAtomM>(
                smemA + ((int64_t)s * kMbAtomsPerSlot + ai) * kMbTcSmemA1, k * wgs::kWgmmaAtomK,
                [&] (int mn, int kk) { return srcA(mbase + mn, kk); }, tid, nthreads);
        }
        stage_kmajor_tile<N>(
            smemB + (int64_t)s * kMbTcSmemB1, k * wgs::kWgmmaAtomK,
            [&] (int mn, int kk) { return srcB(mn, kk); }, tid, nthreads);
    };
    // Issue the group's g_atoms wgmmas for staged slot k (k=0 overwrite else accum).
    auto issue_k = [&] (int g_atoms, int k) {
        const int s = k % S;
        wgs::SmemDesc dB = wgs::make_desc_B_kmajor<N, wgs::kSwizzleNone>(
            smemB + (int64_t)s * kMbTcSmemB1);
        #pragma unroll
        for (int ai = 0; ai < kIL; ++ai) {
            if (ai >= g_atoms) break;
            wgs::SmemDesc dA = wgs::make_desc_A_kmajor<wgs::kWgmmaAtomM, wgs::kSwizzleNone>(
                smemA + ((int64_t)s * kMbAtomsPerSlot + ai) * kMbTcSmemA1);
            if (k == 0) wgs::wgmma_m64nNk16_bf16<N, /*ScaleD=*/0, 0, 0>(acc[ai], dA, dB);
            else        wgs::wgmma_m64nNk16_bf16<N, /*ScaleD=*/1, 0, 0>(acc[ai], dA, dB);
        }
    };

    // Loop M-atom GROUPS of kIL; each group runs its own k-chain into kIL fragments.
    #pragma unroll 1
    for (int g0 = 0; g0 < m_atoms; g0 += kIL) {
        const int gbase = mbase0 + g0 * wgs::kWgmmaAtomM;
        const int g_atoms = (m_atoms - g0) < kIL ? (m_atoms - g0) : kIL;
        // ── Prologue: stage tile 0 (the group's atoms) into slot 0; visible; fence. ──
        stage_k(gbase, g_atoms, 0);
        __syncthreads();                 // tile 0 visible to the wgmma below
        if (in_wg0) wgs::wgmma_fence();
        // ── Steady state (S=2 single group in flight): issue the g_atoms wgmmas for
        //    slot k%S (async, overlapping in the tensor pipe), THEN stage tile k+1
        //    into the OTHER slot (its HBM loads overlap the MMAs), THEN wait_group<0>
        //    + sync (group's wgmmas done reading slot k&1; stage(k+1) visible; slot
        //    reuse-safe). S=1 collapses to staging into the single slot AFTER the
        //    wait (serial, bit-identical to the old per-k stage→sync→issue→wait). ──
        #pragma unroll 1
        for (int k = 0; k < k_steps; ++k) {
            if (in_wg0) {
                issue_k(g_atoms, k);
                wgs::wgmma_commit_group();
            }
            if (S > 1) {
                // Double-buffer: prefetch k+1 into the other slot, overlapping wgmma(k).
                if (k + 1 < k_steps) stage_k(gbase, g_atoms, k + 1);
                if (in_wg0) wgs::wgmma_wait_group<0>();   // group's wgmmas done reading slot k&1
                __syncthreads();                          // stage(k+1) visible; slot k&1 free
            } else {
                // Serial (S==1, single slot): wait, then stage next into the slot.
                if (in_wg0) wgs::wgmma_wait_group<0>();
                __syncthreads();                          // group's wgmmas done reading the slot
                if (k + 1 < k_steps) { stage_k(gbase, g_atoms, k + 1); __syncthreads(); }
            }
        }
        // Epilogue: warpgroup 0 owns the fp32 fragments; decode + emit (real cols).
        // All reads happen AFTER the final wait_group<0> — no overlap with any wgmma.
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
        __syncthreads();
    }
#else
    (void)mbase0; (void)m_atoms; (void)n_real; (void)k_steps;
    (void)srcA; (void)srcB; (void)out; (void)smemA; (void)smemB;
#endif
}

// k_steps with PADDING: K is padded UP to a multiple of 16; the staging
// accessor must return 0 for k >= K_real (the pad rows). dt_proj contracts K=8,
// x_proj-dX contracts K=dbc=40 — NEITHER a multiple of 16. ceil, not floor.
__device__ __host__ __forceinline__ int kceil16(int K) {
    return (K + wgs::kWgmmaAtomK - 1) / wgs::kWgmmaAtomK;
}

// ════════════════════════════════════════════════════════════════════════
//  THIN ORIENTATION WRAPPERS (mbtc_gemm_*). Three accessor patterns: fwd
//  (Y=X·Wᵀ), dX (dX=dY·W, W transposed-staged), dW (dW=dYᵀ·X, both transposed,
//  K=T). M = kTileM (kAtomsM stacked atoms). Weights convert fp32→bf16 ON READ
//  (deterministic; no bf16 weight buffer). PAD-AWARE: srcA/srcB return 0 for
//  k >= K_real so a K not divisible by 16 (dt_proj K=8, x_proj-dX K=40) is
//  zero-extended; n_real marks the valid output columns for a ragged N tile.
// ════════════════════════════════════════════════════════════════════════

// (fwd) Y[M,Nout] = X[M,Kin] @ W[Nout,Kin]ᵀ.  Tiles N over [0,Nout) in width-N
// atoms. Kin may be non-/16 (dt_proj Kin=dt_rank=8 → padded). Writes bf16
// Y [M,Nout] at `Yout` (the projection output, stored bf16).
template <int N>
__device__ __forceinline__ void mbtc_gemm_fwd_bf16(
        const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
        __nv_bfloat16* __restrict__ Yout, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    const int k_steps = kceil16(Kin);
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
            return k < Kin ? X[(int64_t)m * Kin + k] : __float2bfloat16(0.f); };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return (nn < Nout && k < Kin) ? __float2bfloat16(W[(int64_t)nn * Kin + k]) : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) {
            Yout[(int64_t)m * Nout + n0 + n] = __float2bfloat16(v); };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB);
    }
}

// Same as fwd but emits fp32 (no bf16 round) into `Yf32` — for projection
// outputs consumed by fp32 elementwise stages (dt_pre, the scan inputs).
template <int N>
__device__ __forceinline__ void mbtc_gemm_fwd_f32(
        const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
        float* __restrict__ Yf32, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    const int k_steps = kceil16(Kin);
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
            return k < Kin ? X[(int64_t)m * Kin + k] : __float2bfloat16(0.f); };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return (nn < Nout && k < Kin) ? __float2bfloat16(W[(int64_t)nn * Kin + k]) : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) { Yf32[(int64_t)m * Nout + n0 + n] = v; };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB);
    }
}

// (dX) dX[M,Kin] = dY[M,Nout] @ W[Nout,Kin].  N(wgmma)=Kin (tiled by N). K=Nout
// (the contracted out_dim) — may be non-/16 (x_proj dX K=dbc=40 → padded).
// W staged transposed: srcB(n=kin, k=out)=W[out·Kin+kin]. Writes fp32 dX [M,Kin].
template <int N>
__device__ __forceinline__ void mbtc_gemm_dx_f32(
        const __nv_bfloat16* __restrict__ dY, const float* __restrict__ W,
        float* __restrict__ dXf32, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    const int k_steps = kceil16(Nout);
    for (int n0 = 0; n0 < Kin; n0 += N) {
        const int n_real = (Kin - n0) < N ? (Kin - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
            return k < Nout ? dY[(int64_t)m * Nout + k] : __float2bfloat16(0.f); };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return (nn < Kin && k < Nout) ? __float2bfloat16(W[(int64_t)k * Kin + nn]) : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) { dXf32[(int64_t)m * Kin + n0 + n] = v; };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB);
    }
}

// ════════════════════════════════════════════════════════════════════════
//  HBM bf16 ACTS buffer (Fork B). The projection-dW operands (X-inputs and
//  dY-adjoints) the cross-tile P2 dW owners read. Token-row-major [T, width]
//  (= sample-major [nsamp][kSeq][width] since rows are sample-ordered). Carved
//  from the FRONT of the workspace; offsets are RUNTIME (depend on T=B·kSeq).
//
//  The 4 projections' dW = dYᵀ@X needs, per layer:
//    in_proj  dW[512,128] = dxz^T  @ layer_in : X_in    [T,128], dY_dxz   [T,512]
//    x_proj   dW[40,256]  = dxdbc^T@ x_main   : X_xmain  [T,256], dY_dxdbc [T,40]
//    dt_proj  dW[256,8]   = ddtpre^T@ dt_raw  : X_dtraw  [T,8],   dY_ddtpre[T,256]
//    out_proj dW[128,256] = dyout^T @ y_gated : X_ygated [T,256], dY_dyout [T,128]
//  (dt_proj.bias = Σ_T ddt_pre — free P2 bias pass. in/x/out_proj have no bias.)
//  The HEAD (out.w/out.b) stays scalar → no acts. dh0 [T,128] feeds tok/pos owner-scan.
// ════════════════════════════════════════════════════════════════════════
// (still in namespace mbtc — opened above)

struct MbActs {
    __nv_bfloat16* X_in[mb::kLayers];      // [T, d]        in_proj input (layer input, bf16)
    __nv_bfloat16* X_xmain[mb::kLayers];   // [T, d_inner]  x_proj input (bf16 copy of x_main)
    __nv_bfloat16* X_dtraw[mb::kLayers];   // [T, dt_rank]  dt_proj input (bf16 dt_raw)
    __nv_bfloat16* X_ygated[mb::kLayers];  // [T, d_inner]  out_proj input (bf16 y_gated)
    __nv_bfloat16* dY_dxz[mb::kLayers];    // [T, 2*d_inner] in_proj output adjoint (bf16 dxz)
    __nv_bfloat16* dY_dxdbc[mb::kLayers];  // [T, dbc]      x_proj output adjoint (bf16 dx_dbc)
    __nv_bfloat16* dY_ddtpre[mb::kLayers]; // [T, d_inner]  dt_proj output adjoint (bf16 ddt_pre)
    __nv_bfloat16* dY_dyout[mb::kLayers];  // [T, d]        out_proj output adjoint (bf16 dy_out)
    __nv_bfloat16* dh0;                    // [T, d]        embedding-input adjoint
};

// Running bf16-element offset for the acts region (host + device must agree).
__host__ __device__ __forceinline__ int64_t mb_acts_bf16_count(int T) {
    const int64_t d = mb::kD, di = mb::kDInner, dr = mb::kDtRank, dbc = mb::kDbc;
    const int64_t Td = (int64_t)T * d, Tdi = (int64_t)T * di, Tdr = (int64_t)T * dr,
                  Tdbc = (int64_t)T * dbc, T2di = (int64_t)T * 2 * di;
    int64_t bf = 0;
    for (int li = 0; li < mb::kLayers; ++li)
        bf += Td + Tdi + Tdr + Tdi + T2di + Tdbc + Tdi + Td;   // X_in,X_xmain,X_dtraw,X_ygated,dY_dxz,dY_dxdbc,dY_ddtpre,dY_dyout
    bf += Td;                                                  // dh0
    return bf;
}
__host__ __device__ __forceinline__ int64_t mb_acts_floats(int T) {
    return (mb_acts_bf16_count(T) + 1) / 2;
}

__device__ __forceinline__ MbActs mb_acts_bind(__nv_bfloat16* p, int T) {
    MbActs a;
    int64_t off = 0;
    const int64_t Td = (int64_t)T * mb::kD, Tdi = (int64_t)T * mb::kDInner,
                  Tdr = (int64_t)T * mb::kDtRank, Tdbc = (int64_t)T * mb::kDbc,
                  T2di = (int64_t)T * 2 * mb::kDInner;
    for (int li = 0; li < mb::kLayers; ++li) {
        a.X_in[li]      = p + off; off += Td;
        a.X_xmain[li]   = p + off; off += Tdi;
        a.X_dtraw[li]   = p + off; off += Tdr;
        a.X_ygated[li]  = p + off; off += Tdi;
        a.dY_dxz[li]    = p + off; off += T2di;
        a.dY_dxdbc[li]  = p + off; off += Tdbc;
        a.dY_ddtpre[li] = p + off; off += Tdi;
        a.dY_dyout[li]  = p + off; off += Td;
    }
    a.dh0 = p + off; off += Td;
    return a;
}

// ════════════════════════════════════════════════════════════════════════
//  Per-CTA TILE SCRATCH (HBM). The fwd-all-layers-then-bwd-all-layers order
//  (the decoder's line-377 lesson) means every sample's forward activations the
//  backward reads must be cached PER (layer, sample) — so `act` is typed as the
//  REAL scalar MambaSampleSmem::LayerAct, [kLayers][nsamp], letting the scalar
//  mb_scan_fwd/bwd be called VERBATIM with `a = &act[li][si]` (they deref only
//  a->dt_pre/Bmat/Cmat; no __shared__ assumption — so the cache is HBM, the
//  launcher stays STATIC-smem, no dynamic opt-in). Plus the residual chain
//  (layer_in / final_in) and the transient fp32 buffers the per-tile fwd/bwd use.
//  One slab per CTA (a CTA finishes a tile fwd+bwd before the next).
//
//  All per-sample scalar calls (scan/conv/LN) take a CONTIGUOUS [kSeq,width]
//  one-sample slice: `buf + si*kSeq*width`. Since rows are sample-major
//  (row m = si*kSeq + t), a token-row-major [T_tile,width] buffer IS that
//  sample-major slice — the TC projection's out(m,·) and the scalar fn's
//  [kSeq,width] view coincide with NO reshape.
// ════════════════════════════════════════════════════════════════════════
struct MbTileScratch {
    MambaSampleSmem::LayerAct* act[mb::kLayers];  // [nsamp] each — the bwd cache
    float* layer_in[mb::kLayers];   // [kTileM, d]      residual / layer input (fp32)
    float* final_in;                // [kTileM, d]      last-layer output (fp32, head input)
    // transient fp32 tile buffers (reused within a tile):
    float* xmain;                   // [kTileM, d_inner] silu(conv) (fp32, scan input + D-skip)
    float* work_di;                 // [kTileM, d_inner] general d_inner-wide fp32 scratch
    float* work_di2;                // [kTileM, d_inner] second d_inner-wide scratch
    float* work_d;                  // [kTileM, d]       d-wide fp32 scratch (LN in/out, residual)
    float* dbc;                     // [kTileM, dbc]     x_dbc / dx_dbc staging (fp32)
    float* fn_xhat;                 // [kTileM, d]       final-norm xhat cache (last pos used)
    float* fn_inv;                  // [kTileM]          final-norm inv cache
    float* logits;                  // [nsamp, kPHead]   per-sample last-pos logits (fp32)
};

constexpr int kNSampPerTile = kTileM / mb::kSeq;
constexpr int kLayerActFloats = (int)(sizeof(MambaSampleSmem::LayerAct) / sizeof(float));

__host__ __device__ __forceinline__ int64_t mb_tile_scratch_floats() {
    const int64_t TMd = (int64_t)kTileM * mb::kD, TMdi = (int64_t)kTileM * mb::kDInner,
                  TMdbc = (int64_t)kTileM * mb::kDbc;
    int64_t f = 0;
    f += (int64_t)mb::kLayers * kNSampPerTile * kLayerActFloats;   // act[L][nsamp]
    f += (int64_t)mb::kLayers * TMd;                               // layer_in[L]
    f += TMd;                                                      // final_in
    f += TMdi + TMdi + TMdi + TMd + TMdbc;                         // xmain,work_di,work_di2,work_d,dbc
    f += TMd + kTileM;                                             // fn_xhat,fn_inv
    f += (int64_t)kNSampPerTile * mb::kPHead;                      // logits
    return f;
}

__device__ __forceinline__ MbTileScratch mb_tile_scratch_bind(float* slab) {
    MbTileScratch s;
    int64_t o = 0;
    for (int li = 0; li < mb::kLayers; ++li) {
        s.act[li] = reinterpret_cast<MambaSampleSmem::LayerAct*>(slab + o);
        o += (int64_t)kNSampPerTile * kLayerActFloats;
    }
    for (int li = 0; li < mb::kLayers; ++li) { s.layer_in[li] = slab + o; o += (int64_t)kTileM * mb::kD; }
    s.final_in = slab + o; o += (int64_t)kTileM * mb::kD;
    s.xmain    = slab + o; o += (int64_t)kTileM * mb::kDInner;
    s.work_di  = slab + o; o += (int64_t)kTileM * mb::kDInner;
    s.work_di2 = slab + o; o += (int64_t)kTileM * mb::kDInner;
    s.work_d   = slab + o; o += (int64_t)kTileM * mb::kD;
    s.dbc      = slab + o; o += (int64_t)kTileM * mb::kDbc;
    s.fn_xhat  = slab + o; o += (int64_t)kTileM * mb::kD;
    s.fn_inv   = slab + o; o += kTileM;
    s.logits   = slab + o; o += (int64_t)kNSampPerTile * mb::kPHead;
    return s;
}

// ── LN-vector + scan/conv grad partials layout (the NON-GEMM grads Fork B keeps
//    tile-local, reduced ascending-CTA in P2). Per CTA, in this dense order:
//      [0]            A_log  [d_inner*state]   (per layer ×L)
//      ...            D      [d_inner]
//      ...            conv_w [d_inner*conv_k]
//      ...            conv_b [d_inner]
//      ...            n_w    [d]               (per-layer norm)
//      ...            n_b    [d]
//      (after L layers) head norm_w [d], norm_b [d], head out_w [pHead*d], out_b [pHead]
//    Each is summed across CTAs in P2. The per-layer block repeats L times; the
//    head block once. The dt_proj.bias is NOT here (it is the free GEMM bias).
// ════════════════════════════════════════════════════════════════════════
constexpr int kPartPerLayer =
      mb::kDInner * mb::kState   // A_log
    + mb::kDInner               // D
    + mb::kDInner * mb::kConvK  // conv_w
    + mb::kDInner               // conv_b
    + mb::kD                    // n_w
    + mb::kD;                   // n_b
constexpr int kPartHead =
      mb::kD + mb::kD                 // norm_w, norm_b
    + mb::kPHead * mb::kD + mb::kPHead; // out_w, out_b
constexpr int kPartElems = mb::kLayers * kPartPerLayer + kPartHead;

// Typed view over ONE CTA's partial slab (so the P1 accumulators write by name).
struct MbPartial {
    float* A_log[mb::kLayers]; float* D[mb::kLayers];
    float* conv_w[mb::kLayers]; float* conv_b[mb::kLayers];
    float* n_w[mb::kLayers]; float* n_b[mb::kLayers];
    float* norm_w; float* norm_b; float* out_w; float* out_b;
};
__device__ __forceinline__ MbPartial mb_partial_bind(float* base) {
    MbPartial p; int64_t o = 0;
    for (int li = 0; li < mb::kLayers; ++li) {
        p.A_log[li]  = base + o; o += mb::kDInner * mb::kState;
        p.D[li]      = base + o; o += mb::kDInner;
        p.conv_w[li] = base + o; o += mb::kDInner * mb::kConvK;
        p.conv_b[li] = base + o; o += mb::kDInner;
        p.n_w[li]    = base + o; o += mb::kD;
        p.n_b[li]    = base + o; o += mb::kD;
    }
    p.norm_w = base + o; o += mb::kD;
    p.norm_b = base + o; o += mb::kD;
    p.out_w  = base + o; o += mb::kPHead * mb::kD;
    p.out_b  = base + o; o += mb::kPHead;
    return p;
}
// The dec_layout tensor index for each partial slot, in P2 reduce order. Built at
// runtime in the reduce (functions of li) — see mbtc_partial_reduce.

// ════════════════════════════════════════════════════════════════════════
//  FORWARD over one SAMPLE TILE (nsamp = nrows/kSeq samples, kTileM rows), global
//  token rows [g0, g0+nrows). The 4 PROJECTIONS run tile-wide on wgmma (M=nrows);
//  conv1d / selective-scan / gate / skip / LN / head / CE run PER-SAMPLE scalar
//  (REUSING model_stage_mamba3.cuh's mb_conv1d_fwd / mb_scan_fwd / mb_layernorm_fwd
//  verbatim with a = &act[li][si]). Writes the MbActs projection operands (bf16)
//  + the per-(layer,sample) LayerAct caches the bwd reads; returns the tile's
//  summed NLL (thread 0 holds it). tok_ids/tgt_ids are HBM int32.
//
//  Mirrors mb_forward_sample's math EXACTLY; the precision fork: layer_in bf16
//  (in_proj + residual), x_main/dt_raw fp32 into scan/conv/gate with ONLY the
//  bf16 copy consumed by x_proj/dt_proj rounded; LN/scan/conv/gate/head fp32.
// ════════════════════════════════════════════════════════════════════════
// __noinline__ (mirrors mbtc_backward_tile): one shared out-of-line frame for the
// 2nd-pass cells that call the forward twice, bounding the megakernel's register
// footprint so the wgmma accumulator stays register-resident (A/A/A determinism).
__device__ float mbtc_forward_tile(
        const MambaWeights& w, int g0, int nrows, const MbActs& acts,
        const MbTileScratch& sc, const int* __restrict__ tok_ids,
        const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red) {
    const int nsamp = nrows / mb::kSeq;
    // ── Embedding: X_in[0][g] = tok[id] + pos[s%S]. bf16. ──
    for (int idx = threadIdx.x; idx < nrows * mb::kD; idx += blockDim.x) {
        const int r = idx / mb::kD, j = idx % mb::kD;
        const int g = g0 + r;
        const int tid = tok_ids[g];
        const int sp = g % mb::kSeq;
        float v = w.tok[(int64_t)tid * mb::kD + j] + w.pos[(int64_t)sp * mb::kD + j];
        acts.X_in[0][(int64_t)g * mb::kD + j] = __float2bfloat16(v);
    }
    __syncthreads();

    for (int li = 0; li < mb::kLayers; ++li) {
        const MambaWeights::Layer& L = w.layer[li];
        const __nv_bfloat16* Xin = acts.X_in[li] + (int64_t)g0 * mb::kD;   // [nrows,d] bf16
        // (a) in_proj (TC): xz = Xin @ in_w^T (N=2*d_inner, K=d). Route split → the
        //     LayerAct caches act[li][si].x_main_raw / .z (fp32, the scan/conv inputs).
        {
            auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return Xin[(int64_t)m * mb::kD + k]; };
            for (int n0 = 0; n0 < 2 * mb::kDInner; n0 += SG_TUNED_TILE_N) {
                const int N = SG_TUNED_TILE_N;
                const int n_real = (2 * mb::kDInner - n0) < N ? (2 * mb::kDInner - n0) : N;
                auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
                    int nn = n0 + n; return nn < 2 * mb::kDInner ? __float2bfloat16(L.in_w[(int64_t)nn * mb::kD + k]) : __float2bfloat16(0.f); };
                auto out = [&] (int m, int n, float v) {
                    const int si = m / mb::kSeq, t = m % mb::kSeq; const int o = n0 + n;
                    if (o < mb::kDInner) sc.act[li][si].x_main_raw[t][o] = v;
                    else sc.act[li][si].z[t][o - mb::kDInner] = v; };
                tc_gemm_block_unpipelined<SG_TUNED_TILE_N, /*MaxAtomsM=*/kAtomsM>(
                    0, kAtomsM, n_real, kceil16(mb::kD), srcA, srcB, out, sA, sB);
            }
        }
        __syncthreads();
        // (b) conv1d (scalar, per-sample) → act.conv (PRE-silu cache); x_main =
        //     silu(conv) → sc.xmain tile buffer (fp32) AND bf16 → X_xmain (x_proj dW operand).
        for (int si = 0; si < nsamp; ++si) {
            MambaSampleSmem::LayerAct* a = &sc.act[li][si];
            mb_conv1d_fwd(&a->x_main_raw[0][0], L.conv_w, L.conv_b, &a->conv[0][0]);
        }
        for (int idx = threadIdx.x; idx < nrows * mb::kDInner; idx += blockDim.x) {
            const int si = idx / (mb::kSeq * mb::kDInner);
            const int rem = idx % (mb::kSeq * mb::kDInner);
            const int t = rem / mb::kDInner, c = rem % mb::kDInner;
            float xm = mb_silu(sc.act[li][si].conv[t][c]);
            sc.xmain[(int64_t)(si * mb::kSeq + t) * mb::kDInner + c] = xm;
            acts.X_xmain[li][(int64_t)(g0 + si * mb::kSeq + t) * mb::kDInner + c] = __float2bfloat16(xm);
        }
        __syncthreads();
        // (c) x_proj (TC): x_dbc = x_main(bf16) @ x_proj_w^T (N=dbc=40, K=d_inner).
        //     fp32 → sc.dbc; split → dt_raw[:, :8] (→ bf16 X_dtraw + kept in dbc),
        //     Bmat/Cmat → act caches.
        mbtc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_xmain[li] + (int64_t)g0 * mb::kDInner, L.x_proj_w,
                                           sc.dbc, mb::kDInner, mb::kDbc, sA, sB);
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * mb::kDbc; idx += blockDim.x) {
            const int r = idx / mb::kDbc, o = idx % mb::kDbc;
            const int si = r / mb::kSeq, t = r % mb::kSeq;
            float v = sc.dbc[(int64_t)r * mb::kDbc + o];
            if (o < mb::kDtRank)
                acts.X_dtraw[li][(int64_t)(g0 + r) * mb::kDtRank + o] = __float2bfloat16(v);  // dt_raw bf16 (dt_proj dW operand)
            else if (o < mb::kDtRank + mb::kState) sc.act[li][si].Bmat[t][o - mb::kDtRank] = v;
            else sc.act[li][si].Cmat[t][o - mb::kDtRank - mb::kState] = v;
        }
        __syncthreads();
        // (d) dt_proj (TC): dt_pre = dt_raw(bf16) @ dt_proj_w^T + bias (N=d_inner,
        //     K=dt_rank=8 PADDED→16). Bias added in the scalar epilogue. fp32 →
        //     act.dt_pre. The GEMM emits WITHOUT bias; add it on read-back.
        mbtc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_dtraw[li] + (int64_t)g0 * mb::kDtRank, L.dt_proj_w,
                                           sc.work_di, mb::kDtRank, mb::kDInner, sA, sB);
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * mb::kDInner; idx += blockDim.x) {
            const int si = idx / (mb::kSeq * mb::kDInner);
            const int rem = idx % (mb::kSeq * mb::kDInner);
            const int t = rem / mb::kDInner, o = rem % mb::kDInner;
            sc.act[li][si].dt_pre[t][o] = sc.work_di[(int64_t)(si * mb::kSeq + t) * mb::kDInner + o] + L.dt_proj_b[o];
        }
        __syncthreads();
        // (e) selective scan (scalar, per-sample) → act.y_scan. x_main one-sample
        //     slice = sc.xmain + si*kSeq*d_inner (contiguous [kSeq,d_inner]).
#if !SG_MBTC_BYPASS_SCAN
        for (int si = 0; si < nsamp; ++si) {
            MambaSampleSmem::LayerAct* a = &sc.act[li][si];
            mb_scan_fwd(L.A_log, sc.xmain + (int64_t)si * mb::kSeq * mb::kDInner, a, &a->y_scan[0][0]);
        }
#else
        for (int idx = threadIdx.x; idx < nrows * mb::kDInner; idx += blockDim.x) {
            const int si = idx / (mb::kSeq * mb::kDInner);
            const int rem = idx % (mb::kSeq * mb::kDInner);
            const int t = rem / mb::kDInner, c = rem % mb::kDInner;
            sc.act[li][si].y_scan[t][c] = 0.0f;  // bypass: zero the scan output
        }
        __syncthreads();
#endif
        // (f) gate+skip: y_gated = (y_scan + x_main*D)*silu(z) → sc.work_di (fp32)
        //     AND bf16 → X_ygated (out_proj dW operand).
        for (int idx = threadIdx.x; idx < nrows * mb::kDInner; idx += blockDim.x) {
            const int si = idx / (mb::kSeq * mb::kDInner);
            const int rem = idx % (mb::kSeq * mb::kDInner);
            const int t = rem / mb::kDInner, c = rem % mb::kDInner;
            const int r = si * mb::kSeq + t;
            MambaSampleSmem::LayerAct* a = &sc.act[li][si];
            float yv = a->y_scan[t][c] + sc.xmain[(int64_t)r * mb::kDInner + c] * L.D[c];
            float yg = yv * mb_silu(a->z[t][c]);
            sc.work_di[(int64_t)r * mb::kDInner + c] = yg;
            acts.X_ygated[li][(int64_t)(g0 + r) * mb::kDInner + c] = __float2bfloat16(yg);
        }
        __syncthreads();
        // (g) out_proj (TC): y_out = y_gated(bf16) @ out_w^T (N=d, K=d_inner). fp32
        //     → sc.work_d; r = y_out + residual(Xin bf16→fp32) → sc.work_d.
        mbtc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_ygated[li] + (int64_t)g0 * mb::kDInner, L.out_w,
                                           sc.work_d, mb::kDInner, mb::kD, sA, sB);
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * mb::kD; idx += blockDim.x) {
            const int r = idx / mb::kD, j = idx % mb::kD;
            sc.work_d[(int64_t)r * mb::kD + j] += __bfloat162float(Xin[(int64_t)r * mb::kD + j]);
        }
        __syncthreads();
        // (h) LN (scalar, per-sample) → next layer_in (bf16) / final_in (fp32) + act.ln caches.
        float* dst = (li + 1 < mb::kLayers) ? sc.layer_in[li + 1] : sc.final_in;
        for (int si = 0; si < nsamp; ++si) {
            MambaSampleSmem::LayerAct* a = &sc.act[li][si];
            const int r0 = si * mb::kSeq;
            mb_layernorm_fwd(sc.work_d + (int64_t)r0 * mb::kD, L.n_w, L.n_b,
                             dst + (int64_t)r0 * mb::kD, &a->ln_xhat[0][0], &a->ln_inv[0], red, mb::kD);
        }
        if (li + 1 < mb::kLayers) {
            for (int idx = threadIdx.x; idx < nrows * mb::kD; idx += blockDim.x) {
                const int r = idx / mb::kD, j = idx % mb::kD;
                acts.X_in[li + 1][(int64_t)(g0 + r) * mb::kD + j] = __float2bfloat16(sc.layer_in[li + 1][(int64_t)r * mb::kD + j]);
            }
            __syncthreads();
        }
    }

    // ── Final norm + head + CE, scalar PER-SAMPLE on the LAST position only. ──
    float nll_acc = 0.0f;
    for (int si = 0; si < nsamp; ++si) {
        const int rlast = si * mb::kSeq + (mb::kSeq - 1);
        const float* hlast = sc.final_in + (int64_t)rlast * mb::kD;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) sum += hlast[j];
        float mean = mb_block_sum(sum, red) / (float)mb::kD;
        float vs = 0.0f;
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) { float c = hlast[j] - mean; vs += c * c; }
        float var = mb_block_sum(vs, red) / (float)mb::kD;
        float iv = rsqrtf(var + mb::kLnEps);
        if (threadIdx.x == 0) sc.fn_inv[rlast] = iv;
        // hn = xhat*norm_w + norm_b → sc.work_d row rlast (head input, fp32, scalar head).
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float xh = (hlast[j] - mean) * iv;
            sc.fn_xhat[(int64_t)rlast * mb::kD + j] = xh;
            sc.work_d[(int64_t)rlast * mb::kD + j] = xh * w.norm_w[j] + w.norm_b[j];
        }
        __syncthreads();
        // logits[o] = hn · out_w[o] + out_b[o] (scalar head, fp32). Store sc.logits[si].
        float* lg = sc.logits + (int64_t)si * mb::kPHead;
        const float* hn = sc.work_d + (int64_t)rlast * mb::kD;
        for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) {
            const float* Wr = w.out_w + (int64_t)o * mb::kD;
            float acc = w.out_b[o];
            #pragma unroll 4
            for (int k = 0; k < mb::kD; ++k) acc += hn[k] * Wr[k];
            lg[o] = acc;
        }
        __syncthreads();
        int tgt = tgt_ids[g0 / mb::kSeq + si];
        float lmax = -CUDART_INF_F;
        for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) lmax = fmaxf(lmax, lg[o]);
        lmax = mb_block_max(lmax, red);
        float es = 0.0f;
        for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) es += __expf(lg[o] - lmax);
        es = mb_block_sum(es, red);
        float logz = lmax + __logf(es);
        if (threadIdx.x == 0) nll_acc += (logz - lg[tgt]);
        __syncthreads();
    }
    return nll_acc;
}

// ════════════════════════════════════════════════════════════════════════
//  BACKWARD over one SAMPLE TILE. Assumes mbtc_forward_tile ran for THIS tile
//  (act caches + final/fn caches + logits populated). The 4 PROJECTIONS compute
//  dX via wgmma and WRITE the dY output-adjoints to MbActs (dY_dxz/dY_dxdbc/
//  dY_ddtpre/dY_dyout) + dh0 for P2's output-stationary dW (it does NOT touch the
//  proj dW here). The scan/conv (per-sample scalar, REUSED mb_scan_bwd /
//  mb_conv1d_bwd) + the gate/skip + LN + scalar head accumulate A_log/D/conv_w/b/
//  n_w/b/norm/out into the per-CTA partial `part`. `B` is the full batch (CE mean).
//  Mirrors mb_backward_sample's order 1..13 exactly.
//
//  smem: `dBmat_s`/`dCmat_s` [kSeq*kState] each (the scan-bwd cross-channel reduce
//  target, zeroed + consumed PER SAMPLE). `red` the reduction slot.
// ════════════════════════════════════════════════════════════════════════
// __noinline__ (A/A/A DETERMINISM, register-pressure bound): the megakernel calls
// this heavy backward from P1 AND — for the SAM-2nd-pass cells (looksam/SG) — a
// SECOND time inside the P2.4 SAM block. Inlining both copies into the megakernel
// pushes the per-thread frame past the wgmma-accumulator-spill threshold (the
// LookSAM instantiation hit STACK 3024 / spilled the 64-reg wgmma accumulator of
// the projection GEMMs in the double-buffer window → non-deterministic denormal/NaN
// grad drift; the basic single-pass cells stayed under it). Forcing ONE shared
// out-of-line frame keeps the megakernel's own register footprint small for EVERY
// instantiation, so the GEMM accumulator stays register-resident and the wgmma is
// deterministic BY CONSTRUCTION — independent of which optimizer's tail is compiled
// in. The math is unchanged (same code, just not inlined). Warpgroup-uniform call:
// the whole CTA enters/exits together, so the wgmma fence/commit/wait choreography
// inside is well-formed across the call boundary.
__device__ void mbtc_backward_tile(
        const MambaWeights& w, int g0, int nrows, int B, const MbActs& acts,
        const MbTileScratch& sc, const MbPartial& part,
        const int* __restrict__ tok_ids, const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        float* dBmat_s, float* dCmat_s) {
    const int nsamp = nrows / mb::kSeq;

    // ── CE bwd + scalar head + final-norm bwd, per sample. dh (last-layer output
    //    adjoint) → sc.work_d (zero non-last rows). ──
    for (int idx = threadIdx.x; idx < nrows * mb::kD; idx += blockDim.x) sc.work_d[idx] = 0.0f;
    __syncthreads();
    for (int si = 0; si < nsamp; ++si) {
        const int rlast = si * mb::kSeq + (mb::kSeq - 1);
        const int gs = g0 / mb::kSeq + si;
        float* lg = sc.logits + (int64_t)si * mb::kPHead;
        int tgt = tgt_ids[gs];
        float lmax = -CUDART_INF_F;
        for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) lmax = fmaxf(lmax, lg[o]);
        lmax = mb_block_max(lmax, red);
        float es = 0.0f;
        for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) es += __expf(lg[o] - lmax);
        es = mb_block_sum(es, red);
        float inv_es = 1.0f / es;
        // dlogits → overwrite lg (scalar head, fp32; NO bf16 — head is not a projection).
        for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) {
            float smo = __expf(lg[o] - lmax) * inv_es;
            lg[o] = (smo - ((o == tgt) ? 1.0f : 0.0f)) / (float)B;
        }
        __syncthreads();
        // head: out_b += dl ; out_w[o] += dl * hn  (hn = fn_xhat*norm_w + norm_b).
        const float* xh = sc.fn_xhat + (int64_t)rlast * mb::kD;
        for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) {
            float dl = lg[o];
            part.out_b[o] += dl;
            float* gwrow = part.out_w + (int64_t)o * mb::kD;
            #pragma unroll 4
            for (int j = 0; j < mb::kD; ++j) gwrow[j] += dl * (xh[j] * w.norm_w[j] + w.norm_b[j]);
        }
        __syncthreads();
        // dhn[j] = Σ_o dl[o]*out_w[o,j] → sc.work_di row 0 (transient; d-wide).
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float acc = 0.0f;
            for (int o = 0; o < mb::kPHead; ++o) acc += lg[o] * w.out_w[(int64_t)o * mb::kD + j];
            sc.work_di[j] = acc;   // dhn (reuse work_di[0..d) as transient)
        }
        __syncthreads();
        // final-norm bwd (single row last): norm γ/β → partial; dh[rlast] = LN-dx.
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float d = sc.work_di[j];
            part.norm_b[j] += d; part.norm_w[j] += d * xh[j];
        }
        __syncthreads();
        {
            float sda = 0.0f, sdax = 0.0f;
            for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
                float dxhat = sc.work_di[j] * w.norm_w[j]; sda += dxhat; sdax += dxhat * xh[j];
            }
            sda = mb_block_sum(sda, red); sdax = mb_block_sum(sdax, red);
            float iv = sc.fn_inv[rlast];
            for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
                float dxhat = sc.work_di[j] * w.norm_w[j];
                sc.work_d[(int64_t)rlast * mb::kD + j] = iv * (dxhat - (sda + xh[j] * sdax) / (float)mb::kD);
            }
            __syncthreads();
        }
    }
    // sc.work_d now = dh (grad wrt last-layer output [nrows,d], only last pos nonzero).

    for (int li = mb::kLayers - 1; li >= 0; --li) {
        const MambaWeights::Layer& L = w.layer[li];
        // (the residual operand layer_in is read by P2's in_proj dW from acts.X_in[li];
        //  the residual ADJOINT path here is the saved dr in sc.work_d.)
        // (1) LN bwd (per-sample): dh → dr (sc.work_di2 reused as dr, d-wide);
        //     accumulate per-layer norm γ/β into partial.
        for (int si = 0; si < nsamp; ++si) {
            MambaSampleSmem::LayerAct* a = &sc.act[li][si];
            const int r0 = si * mb::kSeq;
            mb_layernorm_bwd(sc.work_d + (int64_t)r0 * mb::kD, &a->ln_xhat[0][0], &a->ln_inv[0],
                             L.n_w, sc.work_di2 + (int64_t)r0 * mb::kD, part.n_w[li], part.n_b[li], red);
        }
        // dr lives in sc.work_di2[:, :d]. (2) r = y_out + residual:
        //   dy_out = dr ; dx_residual = dr (SAVE dx_residual into sc.work_d — dh free now).
        //   dy_out → dY_dyout acts (bf16, out_proj dW operand).
        for (int idx = threadIdx.x; idx < nrows * mb::kD; idx += blockDim.x) {
            float dr = sc.work_di2[idx];
            sc.work_d[idx] = dr;   // dx_residual := dr (saved)
            acts.dY_dyout[li][(int64_t)g0 * mb::kD + idx] = __float2bfloat16(dr);
        }
        __syncthreads();
        // (3) out_proj dX: dy_gated = dy_out @ out_w (N=d_inner, K=d). fp32 → sc.work_di.
        mbtc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_dyout[li] + (int64_t)g0 * mb::kD, L.out_w,
                                          sc.work_di, mb::kDInner, mb::kD, sA, sB);  // dy_gated
        __syncthreads();
        // Recompute x_main = silu(conv) → sc.xmain (needed by gate bwd + dD + conv-bwd-in
        //   is x_main_raw in act cache). conv cache lives in act[li][si].conv.
        for (int idx = threadIdx.x; idx < nrows * mb::kDInner; idx += blockDim.x) {
            const int si = idx / (mb::kSeq * mb::kDInner);
            const int rem = idx % (mb::kSeq * mb::kDInner);
            const int t = rem / mb::kDInner, c = rem % mb::kDInner;
            sc.xmain[(int64_t)(si * mb::kSeq + t) * mb::kDInner + c] = mb_silu(sc.act[li][si].conv[t][c]);
        }
        __syncthreads();
        // (4-6) gate+skip bwd (elementwise). y_gated = y_skip*silu(z);
        //   y_skip = y_scan + x_main*D. sz=silu(z); dy_skip = dy_gated*sz;
        //   dsz = dy_gated*y_skip; dz = dsz*silu'(z); dy_scan = dy_skip;
        //   dx_main(D-path) = dy_skip*D; dD += Σ dy_skip*x_main.
        //   Buffers: dz → sc.work_di2 (dr consumed; reuse, d_inner-wide); dy_scan →
        //   act.y_scan (in place); dx_main accumulator(D-path) → sc.work_di
        //   (dy_gated consumed). dD → partial (owner channel c).
        for (int c = threadIdx.x; c < mb::kDInner; c += blockDim.x) {
            float dDc = 0.0f;
            #pragma unroll
            for (int si = 0; si < nsamp; ++si) {
                MambaSampleSmem::LayerAct* a = &sc.act[li][si];
                #pragma unroll
                for (int t = 0; t < mb::kSeq; ++t) {
                    const int r = si * mb::kSeq + t;
                    float sz = mb_silu(a->z[t][c]);
                    float xm = sc.xmain[(int64_t)r * mb::kDInner + c];
                    float yskip = a->y_scan[t][c] + xm * L.D[c];
                    float dyg = sc.work_di[(int64_t)r * mb::kDInner + c];   // dy_gated
                    float dyskip = dyg * sz;
                    float dsz = dyg * yskip;
                    sc.work_di2[(int64_t)r * mb::kDInner + c] = dsz * mb_silu_grad(a->z[t][c]);  // dz
                    dDc += dyskip * xm;
                    a->y_scan[t][c] = dyskip;                               // dy_scan (in place)
                    sc.work_di[(int64_t)r * mb::kDInner + c] = dyskip * L.D[c];  // dx_main_D (overwrite dy_gated)
                }
            }
            part.D[li][c] += dDc;
        }
        __syncthreads();
        // (7) selective scan bwd (per-sample scalar). x_main = sc.xmain (untouched).
        //   dy_scan = act.y_scan. Outputs: dx_main_scan → act.z (in place; z fwd
        //   consumed as dz=work_di2); ddt_pre → act.dt_pre (in place); dB/dC →
        //   smem dBmat_s/dCmat_s (zeroed per sample), consumed into sc.dbc; dA_log
        //   → partial. Then dx_main = D-path(work_di) + scan-path(act.z) → work_di.
        for (int si = 0; si < nsamp; ++si) {
            MambaSampleSmem::LayerAct* a = &sc.act[li][si];
            const int r0 = si * mb::kSeq;
            for (int idx = threadIdx.x; idx < mb::kSeq * mb::kState; idx += blockDim.x) {
                dBmat_s[idx] = 0.0f; dCmat_s[idx] = 0.0f;
            }
            __syncthreads();
#if !SG_MBTC_BYPASS_SCAN
            mb_scan_bwd(L.A_log, sc.xmain + (int64_t)r0 * mb::kDInner, a, &a->y_scan[0][0],
                        &a->z[0][0],        // dx_main_scan SET → act.z
                        &a->dt_pre[0][0],   // ddt_pre SET → act.dt_pre (in place, safe per channel)
                        dBmat_s, dCmat_s, part.A_log[li], red);
#else
            for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
                const int t = idx / mb::kDInner, c = idx % mb::kDInner;
                a->z[t][c] = 0.0f; a->dt_pre[t][c] = 0.0f;   // bypass: zero scan-bwd outputs
            }
            __syncthreads();
#endif
            // ddt_pre → dY_ddtpre acts (bf16, dt_proj dW operand) + assemble dx_dbc's
            //   B/C halves into sc.dbc (ddt_raw filled after dt_proj dX). dt_raw also
            //   needed for dt_proj dW (X operand) — already stored bf16 in acts.X_dtraw.
            for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
                const int t = idx / mb::kDInner, o = idx % mb::kDInner;
                acts.dY_ddtpre[li][(int64_t)(g0 + r0 + t) * mb::kDInner + o] = __float2bfloat16(a->dt_pre[t][o]);
            }
            // dB/dC → sc.dbc[:, dt_rank..] for this sample's rows.
            for (int idx = threadIdx.x; idx < mb::kSeq * mb::kState; idx += blockDim.x) {
                const int t = idx / mb::kState, k = idx % mb::kState;
                const int r = r0 + t;
                sc.dbc[(int64_t)r * mb::kDbc + mb::kDtRank + k] = dBmat_s[t * mb::kState + k];
                sc.dbc[(int64_t)r * mb::kDbc + mb::kDtRank + mb::kState + k] = dCmat_s[t * mb::kState + k];
            }
            __syncthreads();
        }
        // dx_main = D-path (work_di) + scan-path (act.z) → work_di2 (dz consumed:
        //   dz was needed for the in_proj dxz — STILL NEEDED at step 12! So do NOT
        //   clobber work_di2 yet. Put dx_main into work_di in place: work_di holds
        //   dx_main_D; add scan-path act.z).
        for (int idx = threadIdx.x; idx < nrows * mb::kDInner; idx += blockDim.x) {
            const int si = idx / (mb::kSeq * mb::kDInner);
            const int rem = idx % (mb::kSeq * mb::kDInner);
            const int t = rem / mb::kDInner, c = rem % mb::kDInner;
            sc.work_di[idx] += sc.act[li][si].z[t][c];   // dx_main = D-path + scan-path (in work_di)
        }
        __syncthreads();
        // (8) dt_proj dX: ddt_raw = ddt_pre @ dt_proj_w (N=dt_rank=8, K=d_inner).
        //   fp32 → sc.dbc[:, :dt_rank] (the ddt_raw half of dx_dbc). dt_proj dW
        //   (ddt_pre^T@dt_raw) + bias (Σ ddt_pre) are P2 (both operands in acts).
        {
            auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
                return k < mb::kDInner ? acts.dY_ddtpre[li][(int64_t)(g0 + m) * mb::kDInner + k] : __float2bfloat16(0.f); };
            auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
                return (n < mb::kDtRank && k < mb::kDInner) ? __float2bfloat16(L.dt_proj_w[(int64_t)k * mb::kDtRank + n]) : __float2bfloat16(0.f); };
            auto out = [&] (int m, int n, float v) {
                if (n < mb::kDtRank) sc.dbc[(int64_t)m * mb::kDbc + n] = v; };   // ddt_raw → dbc[:, :dt_rank]
            tc_gemm_block_unpipelined<SG_TUNED_TILE_N, /*MaxAtomsM=*/kAtomsM>(
                0, kAtomsM, mb::kDtRank, kceil16(mb::kDInner), srcA, srcB, out, sA, sB);
        }
        __syncthreads();
        // dx_dbc = cat[ddt_raw, dB, dC] now complete in sc.dbc. → dY_dxdbc acts (bf16,
        //   x_proj dW operand). x_proj dW (dx_dbc^T@x_main) is P2 (both in acts).
        for (int idx = threadIdx.x; idx < nrows * mb::kDbc; idx += blockDim.x) {
            const int r = idx / mb::kDbc, o = idx % mb::kDbc;
            acts.dY_dxdbc[li][(int64_t)(g0 + r) * mb::kDbc + o] = __float2bfloat16(sc.dbc[(int64_t)r * mb::kDbc + o]);
        }
        __syncthreads();
        // (9) x_proj dX: dx_main(x_proj path) = dx_dbc @ x_proj_w (N=d_inner, K=dbc=40
        //   PADDED→48). fp32 → sc.work_di2? No — work_di2 holds dz (needed step 12).
        //   Use sc.xmain (x_main fwd consumed: scan bwd done, gate bwd done, dD done —
        //   x_main no longer needed). dx_main(x_proj) → sc.xmain; then ADD to the
        //   dx_main accumulator (work_di).
        mbtc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_dxdbc[li] + (int64_t)g0 * mb::kDbc, L.x_proj_w,
                                          sc.xmain, mb::kDInner, mb::kDbc, sA, sB);  // dx_main_xproj
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * mb::kDInner; idx += blockDim.x)
            sc.work_di[idx] += sc.xmain[idx];   // dx_main complete (work_di)
        __syncthreads();
        // (10-11) dconv = dx_main * silu'(conv) → sc.xmain (x_main consumed). conv bwd
        //   (per-sample scalar): conv_w/conv_b → partial; dx_main_raw → sc.work_di
        //   (dx_main consumed; reuse, d_inner-wide).
        for (int idx = threadIdx.x; idx < nrows * mb::kDInner; idx += blockDim.x) {
            const int si = idx / (mb::kSeq * mb::kDInner);
            const int rem = idx % (mb::kSeq * mb::kDInner);
            const int t = rem / mb::kDInner, c = rem % mb::kDInner;
            sc.xmain[idx] = sc.work_di[idx] * mb_silu_grad(sc.act[li][si].conv[t][c]);   // dconv
        }
        __syncthreads();
        for (int si = 0; si < nsamp; ++si) {
            MambaSampleSmem::LayerAct* a = &sc.act[li][si];
            const int r0 = si * mb::kSeq;
            mb_conv1d_bwd(sc.xmain + (int64_t)r0 * mb::kDInner, &a->x_main_raw[0][0], L.conv_w,
                          part.conv_w[li], part.conv_b[li], sc.work_di + (int64_t)r0 * mb::kDInner);  // dx_main_raw
        }
        // (12) dxz = cat[dx_main_raw (work_di), dz (work_di2)] → dY_dxz acts (bf16,
        //   in_proj dW operand, [T, 2*d_inner]). in_proj dW (dxz^T@layer_in) is P2.
        for (int idx = threadIdx.x; idx < nrows * mb::kDInner; idx += blockDim.x) {
            const int r = idx / mb::kDInner, c = idx % mb::kDInner;
            acts.dY_dxz[li][(int64_t)(g0 + r) * 2 * mb::kDInner + c] = __float2bfloat16(sc.work_di[idx]);              // dx_main_raw half
            acts.dY_dxz[li][(int64_t)(g0 + r) * 2 * mb::kDInner + mb::kDInner + c] = __float2bfloat16(sc.work_di2[idx]); // dz half
        }
        __syncthreads();
        // (13) in_proj dX: dx_inproj = dxz @ in_w (N=d, K=2*d_inner). fp32 → sc.work_di
        //   (dx_main_raw consumed). dh = dx_inproj + dx_residual(sc.work_d) → sc.work_d
        //   (the previous layer's output adjoint, or the embedding grad for li=0).
        mbtc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_dxz[li] + (int64_t)g0 * 2 * mb::kDInner, L.in_w,
                                          sc.work_di, mb::kD, 2 * mb::kDInner, sA, sB);  // dx_inproj [nrows,d]
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * mb::kD; idx += blockDim.x)
            sc.work_d[idx] += sc.work_di[idx];   // dh = residual + in_proj path
        __syncthreads();
    }

    // ── embedding bwd: dh = grad wrt h0 [nrows,d] (sc.work_d). → dh0 acts (bf16);
    //    tok/pos owner-scan (P2) reads dh0. ──
    for (int idx = threadIdx.x; idx < nrows * mb::kD; idx += blockDim.x)
        acts.dh0[(int64_t)g0 * mb::kD + idx] = __float2bfloat16(sc.work_d[idx]);
    __syncthreads();
}

// ════════════════════════════════════════════════════════════════════════
//  SAM-cell-scoped OUT-OF-LINE tile shims (reg-pressure campaign C2). The
//  unconditional __noinline__ on mbtc_forward_tile/mbtc_backward_tile (the
//  original LookSAM accumulator-spill fix) taxed EVERY cell with the ABI
//  out-of-line frame: the single-pass cells carried ~5.7-5.8 KB of callee
//  spill traffic that the fully-inline allocation does not have (the original
//  fix note itself records that the single-pass cells stayed under the spill
//  threshold when inline). SCOPING the outline to the SAM-coupled cells
//  (LookSAM/SG11/SG15/SG2 -- the ones that genuinely instantiate the tile body
//  TWICE) keeps their one-shared-frame fix AND returns the single-pass cells
//  to the cleaner inline allocation. Math identical; warpgroup-uniform call.
// ════════════════════════════════════════════════════════════════════════
__device__ __noinline__ float mbtc_forward_tile_outlined(
        const MambaWeights& w, int g0, int nrows, const MbActs& acts,
        const MbTileScratch& sc, const int* __restrict__ tok_ids,
        const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red) {
    return mbtc_forward_tile(w, g0, nrows, acts, sc, tok_ids, tgt_ids, sA, sB, red);
}
__device__ __noinline__ void mbtc_backward_tile_outlined(
        const MambaWeights& w, int g0, int nrows, int B, const MbActs& acts,
        const MbTileScratch& sc, const MbPartial& part,
        const int* __restrict__ tok_ids, const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        float* dBmat_s, float* dCmat_s) {
    mbtc_backward_tile(w, g0, nrows, B, acts, sc, part, tok_ids, tgt_ids, sA, sB, red,
                       dBmat_s, dCmat_s);
}


// ════════════════════════════════════════════════════════════════════════
//  P2 — OUTPUT-STATIONARY dW (the 4 projections × kLayers = 8 weight matrices,
//  dW = dYᵀ @ X, K=T). Each split into 64×N output tiles; tile_id % nCTA owns
//  each, contracting the FULL T (ascending-t, no atomics, no partials) via the
//  validated dW orientation (both operands transposed-staged, MaxAtomsM=1).
//  Writes into `grad`. The ONLY GEMM bias is dt_proj.bias = Σ_T ddt_pre.
//
//  The 8 weights, in named_parameters() tensor-index order:
//    L0: in_w=4, x_proj_w=7, dt_proj_w=8, out_w=10
//    L1: in_w=15, x_proj_w=18, dt_proj_w=19, out_w=21
//  with (dY, X) acts and Nout/Kin per the forward projection.
// ════════════════════════════════════════════════════════════════════════
// (still in namespace mbtc — opened above)

struct MbDwSpec {
    const __nv_bfloat16* dY;   // [T, Nout]
    const __nv_bfloat16* X;    // [T, Kin]
    int Nout; int Kin; int T;
    int grad_off;
    bool has_bias;             // only dt_proj
    const __nv_bfloat16* dY_bias;
    int bias_off;
};

// Build the 8 specs (cheap; all CTAs). T = B*kSeq.
__device__ __forceinline__ void mbtc_build_dw_specs(
        const MbActs& acts, int T, MbDwSpec spec[8]) {
    // per-layer tensor indices (and dt bias idx). L0 base=2 (tok=0,pos=1), 11/layer.
    for (int li = 0; li < mb::kLayers; ++li) {
        const int base = 2 + li * 11;
        const int in_w = base + 2, xp_w = base + 5, dt_w = base + 6, dt_b = base + 7, out_w = base + 8;
        MbDwSpec& s_in = spec[li * 4 + 0];
        s_in.dY = acts.dY_dxz[li];   s_in.X = acts.X_in[li];
        s_in.Nout = 2 * mb::kDInner; s_in.Kin = mb::kD; s_in.T = T;
        s_in.grad_off = kMambaOffsets[in_w]; s_in.has_bias = false;
        MbDwSpec& s_xp = spec[li * 4 + 1];
        s_xp.dY = acts.dY_dxdbc[li]; s_xp.X = acts.X_xmain[li];
        s_xp.Nout = mb::kDbc; s_xp.Kin = mb::kDInner; s_xp.T = T;
        s_xp.grad_off = kMambaOffsets[xp_w]; s_xp.has_bias = false;
        MbDwSpec& s_dt = spec[li * 4 + 2];
        s_dt.dY = acts.dY_ddtpre[li]; s_dt.X = acts.X_dtraw[li];
        s_dt.Nout = mb::kDInner; s_dt.Kin = mb::kDtRank; s_dt.T = T;
        s_dt.grad_off = kMambaOffsets[dt_w]; s_dt.has_bias = true;
        s_dt.dY_bias = acts.dY_ddtpre[li]; s_dt.bias_off = kMambaOffsets[dt_b];
        MbDwSpec& s_out = spec[li * 4 + 3];
        s_out.dY = acts.dY_dyout[li]; s_out.X = acts.X_ygated[li];
        s_out.Nout = mb::kD; s_out.Kin = mb::kDInner; s_out.T = T;
        s_out.grad_off = kMambaOffsets[out_w]; s_out.has_bias = false;
    }
}

// ── dW M-atom INTERLEAVE group width (task #24 H3 port). A dW output "tile" (work
//    item) owns a GROUP of up to kMbDwIL consecutive m64 atoms instead of one, so
//    the engine runs kMbDwIL wgmmas/k-step into independent fragments sharing ONE
//    staged X (B-operand) tile → the H1 win (overlap the MMAs + kMbDwIL× less
//    X-operand HBM traffic) now applies to the dW K=T contraction. kMbDwIL ==
//    kMbMaxIL so the ring smem (MbTcSmem.sA, sized kMbAtomsPerSlot A-tiles/slot)
//    matches. Ascending-k per atom is unchanged → numerics bit-identical + A/A/A.
//    Mamba dW atom counts: in_proj(2·dInner=512→8), x_proj(dbc=40→1 — ODD, a single
//    group of 1 atom), dt_proj(dInner=256→4), out_proj(d=128→2). The ragged-tail
//    logic (g_atoms = min) handles x_proj's lone atom correctly (NO even shortcut). ──
constexpr int kMbDwIL = kMbMaxIL;
__device__ __host__ __forceinline__ int mb_dw_atoms(int Nout)  { return (Nout + 63) / 64; }
__device__ __host__ __forceinline__ int mb_dw_groups(int Nout) {
    return (mb_dw_atoms(Nout) + kMbDwIL - 1) / kMbDwIL;
}

// Total number of dW output WORK ITEMS (M-atom groups × N-tiles) across the 8
// weights (the tile loop count). Each item is a group of <= kMbDwIL m64 atoms.
template <int N>
__device__ __forceinline__ int mbtc_dw_total_tiles(const MbDwSpec spec[8]) {
    int n = 0;
    for (int s = 0; s < 8; ++s)
        n += mb_dw_groups(spec[s].Nout) * ((spec[s].Kin + N - 1) / N);
    return n;
}

// Run ONE global dW WORK ITEM gt (if owned): decode (weight, M-atom GROUP, N-tile),
// then dW = dYᵀ @ X, contract K=T. Both operands transposed-staged. K=T is a multiple
// of 16 (host enforces B%16==0 → T=B·8). The group is up to kMbDwIL stacked m64 atoms
// → the engine interleaves them (kIL=kMbDwIL): kMbDwIL wgmmas/k-step into independent
// fragments sharing ONE staged X B-tile (H1's win on the dW K=T contraction). The
// engine adds mbase0=mbase and passes the GLOBAL row m to out (m<Nout guard handles a
// ragged final group), so the direct grad write needs no per-atom shimming here.
template <int N>
__device__ __forceinline__ void mbtc_dw_run_tile(
        const MbDwSpec spec[8], int gt, float* __restrict__ grad,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    int acc = 0, s = 0, m_group = 0, n_tile = 0;
    for (s = 0; s < 8; ++s) {
        const int ng = mb_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; break; }
        acc += ng * nt;
    }
    const MbDwSpec& sp = spec[s];
    const int n_atoms = mb_dw_atoms(sp.Nout);
    const int a0 = m_group * kMbDwIL;                // first atom of the group
    const int g_atoms = (n_atoms - a0) < kMbDwIL ? (n_atoms - a0) : kMbDwIL;
    const int mbase = a0 * 64;
    const int n0 = n_tile * N;
    const int n_real = (sp.Kin - n0) < N ? (sp.Kin - n0) : N;
    const int k_steps = sp.T / wgs::kWgmmaAtomK;
    const int Nout = sp.Nout, Kin = sp.Kin;
    const __nv_bfloat16* dY = sp.dY; const __nv_bfloat16* X = sp.X;
    // A[m=out, k=t] = dY[t,out] (transposed); B[n=in, k=t] = X[t,in] (transposed).
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
        return m < Nout ? dY[(int64_t)k * Nout + m] : __float2bfloat16(0.f); };
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
        int nn = n0 + n; return nn < Kin ? X[(int64_t)k * Kin + nn] : __float2bfloat16(0.f); };
    auto out  = [&] (int m, int n, float v) {
        if (m < Nout) grad[sp.grad_off + (int64_t)m * Kin + n0 + n] = v; };
    tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kMbDwIL>(
        mbase, g_atoms, n_real, k_steps, srcA, srcB, out, sA, sB);
}

// ════════════════════════════════════════════════════════════════════════
//  SPLIT-K dW (the "multi-CTA tiling" fix for the ~73% idle SMs in P2). The 8
//  projection dW = dYᵀ@X contract K=T (=B·kSeq, 131072 at B=16384) but yield
//  only ~36 output tiles → with 132 SMs, ~96 idle for ALL of P2 (PROFILED: the
//  K-loop is occupancy-bound, NOT latency-bound — a K-loop double-buffer moved
//  the dW phase 0%). Split-K turns each output tile into G work items (one per
//  K-chunk), so the work-steal fans (n_dw·G) items over the grid → the idle SMs
//  do real work. Each CTA computes a PARTIAL over its K-chunk into a per-
//  (tile,chunk) scratch slot; a grid barrier; then a DETERMINISTIC ascending-
//  chunk reduce sums the G partials into grad — no float atomics, fixed order,
//  so A/A/A bit-determinism + the 0.08 parity hold (each partial is the SAME
//  ascending-k fp32 wgmma accumulate; Σ_chunk partial == the full-K sum
//  reassociated into G fp32 blocks; the tol absorbs the reassociation). G==1
//  routes to the single-CTA mbtc_dw_run_tile above (no scratch, byte-identical).
//
//  Partial scratch layout (flat, host-sized): slot (gt, kc) at
//    dw_part[((int64_t)gt*G + kc) * (64*kMbMaxTileN) + row*kMbMaxTileN + col].
//  The reduce reads [0,Nrow)×[0,n_real) and scatters to grad[grad_off +
//  (mbase+row)*Kin + n0+col]. Biases (dt_proj only) stay in mbtc_dw_biases.
// ════════════════════════════════════════════════════════════════════════
constexpr int kMbMaxTileN = SG_TUNED_TILE_N;                      // widest dW N-tile
// Floats per (gt,kc) split-K slot: a work item is an M-atom GROUP of up to kMbDwIL
// stacked m64 atoms → kMbDwIL·64 rows × N cols. (Was 64·N for the 1-atom item.)
constexpr int kMbDwTileFloats = kMbDwIL * wgs::kWgmmaAtomM * kMbMaxTileN;

// COMPILE-TIME max #dW output tiles (the 8 projection dW have fixed Nout/Kin —
// mamba dims are compile-time — so n_dw is a constant; mbtc_dw_total_tiles
// returns this at runtime, but the host needs it to size the split-K scratch).
//   per layer: in(512×128), x_proj(40×256), dt_proj(256×8), out(128×256), N=kMbMaxTileN
//   The M factor is now M-atom GROUPS of kMbDwIL (ceil(ceil(Nout/64)/kMbDwIL)),
//   matching mb_dw_groups — NOT single atoms. (x_proj's 40→1-atom→1-group reserves a
//   full kMbDwIL·64-row slot of which 64 rows are valid; the reduce reads only valid
//   rows. groups·kMbDwIL·64·N == atoms·64·N only for EVEN atom counts; x_proj is odd
//   so its split-K scratch grows by one 64×N slot per N-tile per chunk — accepted.)
constexpr int kMbDwMGroups(int Nout) {                 // ceil(ceil(Nout/64)/kMbDwIL)
    return (((Nout + 63) / 64) + kMbDwIL - 1) / kMbDwIL;
}
constexpr int kMbDwTilesPerLayer =
      kMbDwMGroups(2*mb::kDInner) * ((mb::kD + kMbMaxTileN - 1)/kMbMaxTileN)      // in_proj
    + kMbDwMGroups(mb::kDbc)      * ((mb::kDInner + kMbMaxTileN - 1)/kMbMaxTileN) // x_proj
    + kMbDwMGroups(mb::kDInner)   * ((mb::kDtRank + kMbMaxTileN - 1)/kMbMaxTileN) // dt_proj
    + kMbDwMGroups(mb::kD)        * ((mb::kDInner + kMbMaxTileN - 1)/kMbMaxTileN);// out_proj
constexpr int kMbDwMaxTiles = mb::kLayers * kMbDwTilesPerLayer;

// Split-K dW partial-scratch float count (host carves it from the workspace tail):
// kMbDwMaxTiles tiles × G chunks × one 64×N slot each.
__host__ __device__ __forceinline__ int64_t mb_dw_part_floats(int G) {
    return (int64_t)kMbDwMaxTiles * G * kMbDwTileFloats;
}

// Decode global dW work-item index gt → (spec index s, m_group, n_tile). Shared by
// the partial pass + the reduce so the (gt → geometry) map is single-source.
// m_group indexes M-atom GROUPS of kMbDwIL (group's first atom = m_group·kMbDwIL).
template <int N>
__device__ __forceinline__ void mbtc_dw_decode(
        const MbDwSpec spec[8], int gt, int& s, int& m_group, int& n_tile) {
    int acc = 0;
    for (s = 0; s < 8; ++s) {
        const int ng = mb_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; return; }
        acc += ng * nt;
    }
    s = 7; m_group = 0; n_tile = 0;   // unreachable (gt < n_dw); keep defined
}

// PARTIAL dW for global work item gt over K-chunk kc of G → dw_part. K-chunk =
// [kc*kc_steps, (kc+1)*kc_steps) in 16-wide k-atoms; the launcher chooses
// G | (T/16) so chunks are equal + 16-aligned (NEVER empty → no zero guard needed).
// Fresh ScaleD=0 per chunk → a true partial. Each work item gt is an M-atom GROUP
// (up to kMbDwIL stacked m64 atoms, H3) → the engine interleaves them; the slot holds
// the group's kMbDwIL·64 group-local rows. Writes the full group tile into (gt,kc).
template <int N>
__device__ __forceinline__ void mbtc_dw_run_tile_splitk(
        const MbDwSpec spec[8], int gt, int kc, int G, float* __restrict__ dw_part,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    int s, m_group, n_tile;
    mbtc_dw_decode<N>(spec, gt, s, m_group, n_tile);
    const MbDwSpec& sp = spec[s];
    const int n_atoms = mb_dw_atoms(sp.Nout);
    const int a0 = m_group * kMbDwIL;                      // first atom of the group
    const int g_atoms = (n_atoms - a0) < kMbDwIL ? (n_atoms - a0) : kMbDwIL;
    const int mbase = a0 * 64;
    const int n0 = n_tile * N;
    const int kc_steps = (sp.T / wgs::kWgmmaAtomK) / G;   // equal chunks (G | k_steps_total)
    const int k0 = kc * kc_steps;
    const int Nout = sp.Nout, Kin = sp.Kin;
    const __nv_bfloat16* dY = sp.dY; const __nv_bfloat16* X = sp.X;
    // srcA/srcB offset the K axis by k0*16 (the chunk's first token); k local to chunk.
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
        return m < Nout ? dY[(int64_t)(k0 * wgs::kWgmmaAtomK + k) * Nout + m] : __float2bfloat16(0.f); };
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
        int nn = n0 + n; return nn < Kin ? X[(int64_t)(k0 * wgs::kWgmmaAtomK + k) * Kin + nn] : __float2bfloat16(0.f); };
    float* slot = dw_part + ((int64_t)gt * G + kc) * kMbDwTileFloats;
    // out(mbase+row, col, v): m is GLOBAL (srcA reads dY[m·Nout]; engine adds
    // mbase0=mbase), the slot holds the GROUP's kMbDwIL·64 LOCAL rows → index by
    // (m - mbase). The engine emits group-local rows in [0, g_atoms·64); a `lr<64`
    // clamp would lose atom>=1's rows (the rel~1.0 dW bug). A ragged final group's
    // tail rows stay unwritten but the reduce reads only valid rows → never used.
    auto out  = [&] (int m, int n, float v) {
        const int lr = m - mbase;
        if (lr >= 0 && lr < kMbDwIL * 64 && n < N) slot[(int64_t)lr * N + n] = v; };
    tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kMbDwIL>(
        mbase, g_atoms, /*n_real=*/N, kc_steps, srcA, srcB, out, sA, sB);
}

// Deterministic reduce: output work item gt (% nCTA) sums its G chunk-partials in
// ASCENDING kc and scatters to grad. Same (gt → geometry) decode as the partial. Each
// work item is an M-atom GROUP (up to kMbDwIL stacked m64 atoms, H3) → its slot holds
// the group's kMbDwIL·64 group-local rows; mbase is the group's first global row.
template <int N>
__device__ __forceinline__ void mbtc_dw_reduce_splitk(
        const MbDwSpec spec[8], int n_dw, int G, const float* __restrict__ dw_part,
        float* __restrict__ grad, int cta, int nCTA) {
    for (int gt = cta; gt < n_dw; gt += nCTA) {
        int s, m_group, n_tile;
        mbtc_dw_decode<N>(spec, gt, s, m_group, n_tile);
        const MbDwSpec& sp = spec[s];
        const int mbase = m_group * kMbDwIL * 64;           // group's first global row
        const int n0 = n_tile * N;
        const int n_real = (sp.Kin - n0) < N ? (sp.Kin - n0) : N;
        // Valid group rows: up to kMbDwIL·64, clamped to the weight's row count.
        const int grp_rows = kMbDwIL * 64;
        const int Nrow = (sp.Nout - mbase) < grp_rows ? (sp.Nout - mbase) : grp_rows;
        const int64_t base = (int64_t)gt * G * kMbDwTileFloats;
        for (int idx = threadIdx.x; idx < Nrow * n_real; idx += blockDim.x) {
            const int row = idx / n_real, col = idx % n_real;   // row = group-local lr
            float accv = 0.0f;
            for (int kc = 0; kc < G; ++kc)            // ascending chunk → deterministic
                accv += dw_part[base + (int64_t)kc * kMbDwTileFloats + (int64_t)row * N + col];
            grad[sp.grad_off + (int64_t)(mbase + row) * sp.Kin + n0 + col] = accv;
        }
    }
}

// dt_proj.bias db = Σ_T ddt_pre (the ONLY GEMM bias). The reduction over T is
// O(T=B·kSeq) PER output element; striping it across ALL nCTA CTAs (not letting
// every CTA redundantly recompute the full O(T) sum for every output and race-write
// the same grad slot — the measured ~19%-of-wall uniform straggler: mean≈max across
// CTAs because ALL 132 did the SAME full work) cuts that ~nCTA-fold. We enumerate a
// GLOBAL bias-output index over the (bias-carrying spec s, output o) space and
// round-robin it by (cta·blockDim + tid) so each output is summed by exactly ONE
// thread on ONE CTA — single owner, deterministic, no race, no redundancy.
__device__ __forceinline__ void mbtc_dw_biases(
        const MbDwSpec spec[8], float* __restrict__ grad, int cta, int nCTA) {
    int total_bias = 0;
    #pragma unroll
    for (int s = 0; s < 8; ++s) total_bias += spec[s].has_bias ? spec[s].Nout : 0;
    const int64_t base_thread = (int64_t)cta * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)nCTA * blockDim.x;
    for (int64_t gi = base_thread; gi < total_bias; gi += stride) {
        int s = 0, o = (int)gi;
        #pragma unroll
        for (int ss = 0; ss < 8; ++ss) {
            if (!spec[ss].has_bias) continue;
            if (o < spec[ss].Nout) { s = ss; break; }
            o -= spec[ss].Nout;
        }
        const MbDwSpec& sp = spec[s];
        float accv = 0.0f;
        for (int k = 0; k < sp.T; ++k) accv += __bfloat162float(sp.dY_bias[(int64_t)k * sp.Nout + o]);
        grad[sp.bias_off + o] = accv;
    }
}

// ════════════════════════════════════════════════════════════════════════
//  P2 — NON-GEMM grad reduce (A_log, D, conv_w/b, n_w/b per layer + head
//  norm_w/b, out_w/b). Each was accumulated tile-local into the per-CTA partial
//  [kPartElems]; sum across CTAs in ASCENDING CTA index (deterministic) into the
//  named_parameters() slots of grad. `part_base` = start of [nCTA × kPartElems].
//  Work split round-robin over the kPartElems by FLAT element (each element c is
//  owned by cta = (global_flat_index / chunk)?) — simplest deterministic split:
//  CTA strides the kPartElems by tensor (round-robin tensors), summing each
//  tensor's elements across CTAs. We enumerate the (tensor-index, partial-slot,
//  numel) triples and round-robin them by CTA.
// ════════════════════════════════════════════════════════════════════════
// Number of partial tensors to reduce: 6/layer + 4 head.
constexpr int kNumPartTensors = mb::kLayers * 6 + 4;

// Fill the (grad tensor index, partial-offset-in-CTA-slab, numel) for slot v.
__device__ __forceinline__ void mbtc_part_slot(
        int v, int& grad_idx, int64_t& part_off, int& numel) {
    // per-layer order: A_log, D, conv_w, conv_b, n_w, n_b (6); head: norm_w,norm_b,out_w,out_b.
    const int kLayerNumels[6] = {
        mb::kDInner * mb::kState, mb::kDInner, mb::kDInner * mb::kConvK, mb::kDInner, mb::kD, mb::kD };
    if (v < mb::kLayers * 6) {
        const int li = v / 6, kind = v % 6;
        const int base = 2 + li * 11;
        const int tidx[6] = { base + 0, base + 1, base + 3, base + 4, base + 9, base + 10 };  // A_log,D,conv_w,conv_b,n_w,n_b
        grad_idx = tidx[kind];
        // partial offset within a CTA slab:
        int64_t off = (int64_t)li * kPartPerLayer;
        for (int k = 0; k < kind; ++k) off += kLayerNumels[k];
        part_off = off; numel = kLayerNumels[kind];
    } else {
        const int h = v - mb::kLayers * 6;   // 0..3: norm_w,norm_b,out_w,out_b
        const int htidx[4] = { 2 + mb::kLayers * 11 + 0, 2 + mb::kLayers * 11 + 1,
                               2 + mb::kLayers * 11 + 2, 2 + mb::kLayers * 11 + 3 };
        const int hnum[4] = { mb::kD, mb::kD, mb::kPHead * mb::kD, mb::kPHead };
        grad_idx = htidx[h];
        int64_t off = (int64_t)mb::kLayers * kPartPerLayer;
        for (int k = 0; k < h; ++k) off += hnum[k];
        part_off = off; numel = hnum[h];
    }
}

// Map a flat slab index e → (grad base offset, local index) by re-deriving the
// segment via mbtc_part_slot (16 tensors; cheap, register-only — NO __shared__,
// which would collide with the kernel's MbTcSmem staging arena). The `seg` cursor
// is carried across the grid-stride loop so consecutive e only re-scan at tensor
// boundaries (warp-coherent for long runs).
__device__ __forceinline__ void mbtc_part_flat_map(
        int64_t e, int& seg, int64_t& gdst) {
    int gidx, numel; int64_t poff;
    mbtc_part_slot(seg, gidx, poff, numel);
    // advance forward (e is monotonically increasing within a thread's stride run);
    // if e fell before the cached segment (first iter / stride wrap), reset to 0.
    if (e < poff) { seg = 0; mbtc_part_slot(seg, gidx, poff, numel); }
    while (e >= poff + numel) { ++seg; mbtc_part_slot(seg, gidx, poff, numel); }
    gdst = kMambaOffsets[gidx] + (e - poff);
}

// FLAT-PARALLEL cross-CTA reduce: stripe the kPartElems flat slab elements across
// ALL nCTA CTAs × blockDim threads (not the old by-tensor split that left only
// kNumPartTensors=16 of 132 CTAs working while 116 idled at the B2 barrier — the
// measured ~19%-of-wall straggler). Each flat element e is summed across the nCTA
// per-CTA partials in ASCENDING c (deterministic, unchanged) and scattered to its
// grad tensor. Element ownership is a fixed function of e, so the reduce stays
// bit-identical across runs (gate-2 A/A/A) — same addends, same ascending-c order.
__device__ __forceinline__ void mbtc_partial_reduce(
        const float* __restrict__ part_base, float* __restrict__ grad,
        int nCTA, int cta) {
    const int64_t stride = (int64_t)nCTA * blockDim.x;
    int seg = 0;
    for (int64_t e = (int64_t)cta * blockDim.x + threadIdx.x; e < kPartElems; e += stride) {
        int64_t gdst;
        mbtc_part_flat_map(e, seg, gdst);
        float accv = 0.0f;
        for (int c = 0; c < nCTA; ++c)
            accv += part_base[(int64_t)c * kPartElems + e];
        grad[gdst] = accv;
    }
}

// ════════════════════════════════════════════════════════════════════════
//  P2 — EMBEDDING owner-scan (tok/pos). dh0 [T,d] (bf16) holds grad wrt h0.
//    tok grad row r (token id) → owner CTA (r % nCTA), scans ALL T tokens
//    ascending-t accumulating dh0[t,:] for ids == r. pos row p → owner, sums
//    dh0[t,:] over t with (t%S)==p. Fixed t-order + single owner → deterministic.
// ════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void mbtc_embed_owner_scan(
        const MbActs& acts, const int* __restrict__ tok_ids, int T,
        float* __restrict__ grad, int cta, int nCTA) {
    const int tok_off = kMambaOffsets[0];
    const int pos_off = kMambaOffsets[1];
    for (int r = cta; r < mb::kVocab; r += nCTA) {
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float accv = 0.0f;
            for (int t = 0; t < T; ++t)
                if (tok_ids[t] == r) accv += __bfloat162float(acts.dh0[(int64_t)t * mb::kD + j]);
            grad[tok_off + (int64_t)r * mb::kD + j] = accv;
        }
    }
    for (int p = cta; p < mb::kSeq; p += nCTA) {
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float accv = 0.0f;
            for (int t = 0; t < T; ++t)
                if ((t % mb::kSeq) == p) accv += __bfloat162float(acts.dh0[(int64_t)t * mb::kD + j]);
            grad[pos_off + (int64_t)p * mb::kD + j] = accv;
        }
    }
}

}  // namespace mbtc

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGE_MAMBA_TC_CUH_
