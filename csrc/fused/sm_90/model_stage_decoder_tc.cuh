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
#include "csrc/fused/sm_90/model_stages_decoder.cuh"   // reuse DecWeights/DecGrad/bind + fp32 helpers
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
#define SG_TUNED_DEC_DW_SPLITK 4
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

namespace dectc {

constexpr int kDecMaxIL = SG_TUNED_DEC_GEMM_INTERLEAVE;
static_assert(kDecMaxIL >= 1 && kDecMaxIL <= 4,
              "SG_TUNED_DEC_GEMM_INTERLEAVE must be 1 (serial) .. 4");
constexpr int kDecTcStages = SG_TUNED_DEC_GEMM_STAGES;
static_assert(kDecTcStages >= 1 && kDecTcStages <= 2,
              "SG_TUNED_DEC_GEMM_STAGES must be 1 (serial) or 2 (double-buffer)");
constexpr int kDecTcSmemA1 = wgs::kWgmmaAtomM * wgs::kWgmmaAtomK;   // 64*16 bf16
constexpr int kDecTcSmemB1 = SG_TUNED_TILE_N * wgs::kWgmmaAtomK;    // N*16 bf16
constexpr int kDecDwSplitK = SG_TUNED_DEC_DW_SPLITK;
static_assert(kDecDwSplitK >= 1, "SG_TUNED_DEC_DW_SPLITK must be >= 1");

// Token-tile rows a CTA owns. Must be a multiple of 64 (wgmma atom M) AND of
// kSeq (so a tile boundary is a sample boundary — attention stays in-tile).
constexpr int kTileM = SG_TUNED_TILE_M;
static_assert(kTileM % wgs::kWgmmaAtomM == 0,
              "SG_TUNED_TILE_M must be a multiple of 64 (wgmma m64 atom)");
static_assert(kTileM % dec::kSeq == 0,
              "SG_TUNED_TILE_M must be a multiple of kSeq=4 (tile=sample boundary)");
constexpr int kAtomsM = kTileM / wgs::kWgmmaAtomM;   // stacked m64 atoms per tile
constexpr int kSamplesPerTile = kTileM / dec::kSeq;  // 32 for TILE_M=128

// ── LN vector-grad partials layout (the 10 tile-local γ/β grads). Order MUST
//    match dec_layout tensor indices {6,7,8,9,18,19,20,21,26,27}. We store them
//    densely [10 x kD] per CTA; the P2 reduce maps them back by tensor index. ──
constexpr int kNumLnVec = 10;                 // n1_w,n1_b,n2_w,n2_b ×L + norm_w,norm_b
constexpr int kLnVecElems = kNumLnVec * dec::kD;   // 10 * 128 = 1280
// The dec_layout tensor index of each LN-vector slot, in our dense order.
__device__ __constant__ int kLnVecTensorIdx[kNumLnVec] = {
    6, 7, 8, 9,        // L0 n1.w, n1.b, n2.w, n2.b
    18, 19, 20, 21,    // L1 n1.w, n1.b, n2.w, n2.b
    26, 27             // norm.w, norm.b
};

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
constexpr int kDecNumMuon2D = 11;
struct DecMuon2D { int tidx; int rows; int cols; };
__device__ __constant__ DecMuon2D kDecMuon2D[kDecNumMuon2D] = {
    { 0, dec::kVocab,  dec::kD     },   // tok.weight          [99,128]
    { 1, dec::kSeq,    dec::kD     },   // pos.weight          [4,128]
    { 2, 3*dec::kD,    dec::kD     },   // L0 in_proj_weight   [384,128]
    { 4, dec::kD,      dec::kD     },   // L0 out_proj.weight  [128,128]
    {10, dec::kDff,    dec::kD     },   // L0 ff.0.weight      [512,128]
    {12, dec::kD,      dec::kDff   },   // L0 ff.2.weight      [128,512]
    {14, 3*dec::kD,    dec::kD     },   // L1 in_proj_weight   [384,128]
    {16, dec::kD,      dec::kD     },   // L1 out_proj.weight  [128,128]
    {22, dec::kDff,    dec::kD     },   // L1 ff.0.weight      [512,128]
    {24, dec::kD,      dec::kDff   },   // L1 ff.2.weight      [128,512]
    {28, dec::kVocab,  dec::kD     },   // out.weight          [99,128]
};
// Is tensor index `t` one of the Muon 2D matrices (orthogonalized in P2.7)? P3
// uses this to route ONLY the 1D / non-2D weights to the AdamW aux tail for Muon.
__device__ __forceinline__ bool dec_is_muon_2d(int t) {
    #pragma unroll
    for (int mi = 0; mi < kDecNumMuon2D; ++mi) if (kDecMuon2D[mi].tidx == t) return true;
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
template <int N, int MaxAtomsM, typename SrcA, typename SrcB, typename Out>
__device__ void tc_gemm_block_unpipelined(
        int mbase0, int m_atoms, int n_real, int k_steps,
        SrcA srcA, SrcB srcB, Out out,
        __nv_bfloat16* smemA, __nv_bfloat16* smemB) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
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
    auto issue_k = [&] (int g_atoms, int k) {
        const int sl = k % S;
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

    // Loop M-atom GROUPS of kIL; each group runs its own k-chain into kIL fragments.
    #pragma unroll 1
    for (int g0 = 0; g0 < m_atoms; g0 += kIL) {
        const int gbase = mbase0 + g0 * wgs::kWgmmaAtomM;
        const int g_atoms = (m_atoms - g0) < kIL ? (m_atoms - g0) : kIL;
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
            if (in_wg0) { issue_k(g_atoms, k); wgs::wgmma_commit_group(); }
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
// NOTE on weights: the params blob is fp32; the wgmma engine needs bf16 B
// operands. We convert ON READ in the accessor (`__float2bfloat16(W[...])`) —
// deterministic, so every read yields the same bf16 (determinism-safe), and it
// needs NO bf16 weight buffer or cross-CTA conversion phase. Cost: weight bytes
// read as fp32 (2×) in fwd/dX — a perf-phase optimization, not a correctness gate.
template <int N>
__device__ __forceinline__ void dectc_gemm_fwd(
        const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
        __nv_bfloat16* __restrict__ Yout, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    const int k_steps = Kin / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return X[(int64_t)m * Kin + k]; };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return nn < Nout ? __float2bfloat16(W[(int64_t)nn * Kin + k]) : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) {
            Yout[(int64_t)m * Nout + n0 + n] = __float2bfloat16(v); };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB);
    }
}

// Same as dectc_gemm_fwd but emits the fp32 result (no bf16 round) — for the
// few fwd outputs consumed by fp32 elementwise stages directly. Writes [M,Nout]
// fp32 at `Yf32`.
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
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB);
    }
}

// (dX) dX[M,Kin] = dY[M,Nout] @ W[Nout,Kin].  N(wgmma) = Kin (the in_dim, tiled
// by width N). K = Nout (the contracted out_dim). W is staged transposed:
// srcB(n=kin, k=out) = W[out·Kin + kin] (fp32 → bf16 on read). Writes fp32 dX
// [M,Kin] (LN/elementwise bwd consume it fp32).
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
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB);
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
        const DecWeights& w, int g0, int nrows, const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tok_ids,
        const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red) {
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
        dectc_gemm_fwd<SG_TUNED_TILE_N>(Xin, L.in_w, sc.qkv[li], dec::kD, 3 * dec::kD, sA, sB);
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
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_ctx[li] + (int64_t)g0 * dec::kD, L.out_w,
                                            sc.work, dec::kD, dec::kD, sA, sB);
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
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_x1[li] + (int64_t)g0 * dec::kD, L.ff0_w,
                                            sc.work, dec::kD, dec::kDff, sA, sB);
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * dec::kDff; idx += blockDim.x) {
            const int r = idx / dec::kDff, j = idx % dec::kDff;
            float pre = sc.work[(int64_t)r * dec::kDff + j] + L.ff0_b[j];
            sc.ff0pre[li][(int64_t)r * dec::kDff + j] = __float2bfloat16(pre);
            acts.X_gact[li][(int64_t)(g0 + r) * dec::kDff + j] = __float2bfloat16(dec_gelu(pre));
        }
        __syncthreads();
        // ff2 = X_gact @ ff2_w^T (+ ff2_b) (N=d, K=dff). fp32 → work.
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_gact[li] + (int64_t)g0 * dec::kDff, L.ff2_w,
                                            sc.work, dec::kDff, dec::kD, sA, sB);
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
        const DecWeights& w, int g0, int nrows, int B, const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tgt_ids,
        float* __restrict__ lnvec, float* __restrict__ work2,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red) {
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
    float* gn_normw = lnvec + (int64_t)8 * dec::kD;
    float* gn_normb = lnvec + (int64_t)9 * dec::kD;

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
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff2[li] + (int64_t)g0 * dec::kD, L.ff2_w,
                                           sc.work, dec::kDff, dec::kD, sA, sB);  // dgact [nrows,dff]
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
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff0[li] + (int64_t)g0 * dec::kDff, L.ff0_w,
                                           sc.x1, /*Kin=*/dec::kD, /*Nout=*/dec::kDff, sA, sB);  // dx1_ffn [nrows,d]
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
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_a[li] + (int64_t)g0 * dec::kD, L.out_w,
                                           sc.work, dec::kD, dec::kD, sA, sB);  // dctx
        __syncthreads();
        // attention bwd: (qkv[li], attn[li], dctx=work) → dqkv [nrows,3d] fp32 into
        //   work2 (3d=384 ≤ dff=512, fits). Then → dY_qkv acts (bf16, in_proj dW operand).
        dectc_attn_bwd_tile(sc.qkv[li], sc.attn[li], sc.work, nrows, work2, sc.dsc);
        for (int idx = threadIdx.x; idx < nrows * 3 * dec::kD; idx += blockDim.x)
            acts.dY_qkv[li][(int64_t)g0 * 3 * dec::kD + idx] = __float2bfloat16(work2[idx]);
        __syncthreads();
        // in_proj dX: dx_in_attn = dqkv @ in_w  (output width Kin=d, contract Nout=3d).
        //   fp32 → sc.work; ADD residual (in sc.dh) → new running adjoint dh.
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_qkv[li] + (int64_t)g0 * 3 * dec::kD, L.in_w,
                                           sc.work, /*Kin=*/dec::kD, /*Nout=*/3 * dec::kD, sA, sB);  // dx_in_attn
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
};

// Build the 9 specs (called by all CTAs; cheap). T = B*kSeq.
__device__ __forceinline__ void dectc_build_dw_specs(
        const DecActs& acts, int B, int T, DecDwSpec spec[9]) {
    // dec_layout offsets: see kDecOffsets. Weight tensor indices (and bias idx):
    //   L0: in_w=2 (in_b=3), out_w=4 (out_b=5), ff0_w=10 (ff0_b=11), ff2_w=12 (ff2_b=13)
    //   L1: in_w=14(15), out_w=16(17), ff0_w=22(23), ff2_w=24(25)
    //   head: out_w=28 (out_b=29)
    const int wi[9]  = {2,4,10,12, 14,16,22,24, 28};
    const int bi[9]  = {3,5,11,13, 15,17,23,25, 29};
    for (int s = 0; s < 8; ++s) {
        const int li = s / 4, kind = s % 4;
        DecDwSpec& sp = spec[s];
        sp.K = T; sp.grad_off = kDecOffsets[wi[s]]; sp.bias_off = kDecOffsets[bi[s]];
        if (kind == 0)      { sp.dY = acts.dY_qkv[li]; sp.X = acts.X_in[li];  sp.Nout = 3 * dec::kD; sp.Kin = dec::kD;   }
        else if (kind == 1) { sp.dY = acts.dY_a[li];   sp.X = acts.X_ctx[li]; sp.Nout = dec::kD;     sp.Kin = dec::kD;   }
        else if (kind == 2) { sp.dY = acts.dY_ff0[li]; sp.X = acts.X_x1[li];  sp.Nout = dec::kDff;   sp.Kin = dec::kD;   }
        else                { sp.dY = acts.dY_ff2[li]; sp.X = acts.X_gact[li];sp.Nout = dec::kD;     sp.Kin = dec::kDff; }
        sp.dY_bias = sp.dY;
    }
    DecDwSpec& hd = spec[8];
    hd.dY = acts.dY_logits; hd.X = acts.X_hn; hd.Nout = dec::kVocab; hd.Kin = dec::kD; hd.K = B;
    hd.grad_off = kDecOffsets[28]; hd.bias_off = kDecOffsets[29]; hd.dY_bias = hd.dY;
}

// Total number of 64×N dW output tiles across the 9 weights (for the tile loop).
template <int N>
__device__ __forceinline__ int dectc_dw_total_tiles(const DecDwSpec spec[9]) {
    int n = 0;
    for (int s = 0; s < 9; ++s)
        n += ((spec[s].Nout + 63) / 64) * ((spec[s].Kin + N - 1) / N);
    return n;
}

// Run ONE global dW tile `gt` (if it belongs to this CTA): decode (weight, M-atom,
// N-tile), then contract K via the dW GEMM into grad[grad_off]. MaxAtomsM=1.
template <int N>
__device__ __forceinline__ void dectc_dw_run_tile(
        const DecDwSpec spec[9], int gt, float* __restrict__ grad,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    // Decode gt → (s, m_atom, n_tile).
    int acc = 0, s = 0, m_atom = 0, n_tile = 0;
    for (s = 0; s < 9; ++s) {
        const int ma = (spec[s].Nout + 63) / 64;
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ma * nt) { int loc = gt - acc; m_atom = loc / nt; n_tile = loc % nt; break; }
        acc += ma * nt;
    }
    const DecDwSpec& sp = spec[s];
    const int mbase = m_atom * 64;
    const int n0 = n_tile * N;
    const int n_real = (sp.Kin - n0) < N ? (sp.Kin - n0) : N;
    const int k_steps = sp.K / wgs::kWgmmaAtomK;     // K = T or B (must be /16; padded by caller)
    const int Nout = sp.Nout, Kin = sp.Kin;
    const __nv_bfloat16* dY = sp.dY; const __nv_bfloat16* X = sp.X;
    // The engine (tc_gemm_block_unpipelined, mbase0=mbase, m_atoms=1) passes the
    // GLOBAL row m = mbase + mn to srcA/out (it adds mbase0 itself), so the
    // accessors use `m` RAW — adding mbase again would double-count (the selftest
    // passes mbase0=0 and uses m raw; we pass mbase0=mbase and likewise use m raw).
    // A[m=out, k=t] = dY[t,out]  (transposed read); B[n=in, k=t] = X[t,in] (transposed).
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
        return m < Nout ? dY[(int64_t)k * Nout + m] : __float2bfloat16(0.f); };
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
        int nn = n0 + n; return nn < Kin ? X[(int64_t)k * Kin + nn] : __float2bfloat16(0.f); };
    auto out  = [&] (int m, int n, float v) {
        if (m < Nout) grad[sp.grad_off + (int64_t)m * Kin + n0 + n] = v; };
    tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/1>(
        mbase, /*m_atoms=*/1, n_real, k_steps, srcA, srcB, out, sA, sB);
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
//  kc) at dw_part[((gt*G+kc)) * (64*kDecMaxTileN) + row*kDecMaxTileN + col].
//  Decoder K varies per spec: layer dW K=T, head dW K=B (so kc_steps reads sp.K).
// ════════════════════════════════════════════════════════════════════════
constexpr int kDecMaxTileN = SG_TUNED_TILE_N;                       // widest dW N-tile
constexpr int kDecDwTileFloats = wgs::kWgmmaAtomM * kDecMaxTileN;   // 64*N per (gt,kc) slot

// COMPILE-TIME max #dW output tiles (the 9 dW have fixed Nout/Kin; decoder dims
// are compile-time → constant). per layer: qkv(3d×d), attn_out(d×d), ff0(dff×d),
// ff2(d×dff), N=kDecMaxTileN; + head(V×d).
constexpr int kDecDwTilesPerLayer =
      ((3*dec::kD + 63)/64) * ((dec::kD  + kDecMaxTileN - 1)/kDecMaxTileN)   // qkv
    + ((dec::kD   + 63)/64) * ((dec::kD  + kDecMaxTileN - 1)/kDecMaxTileN)   // attn_out
    + ((dec::kDff + 63)/64) * ((dec::kD  + kDecMaxTileN - 1)/kDecMaxTileN)   // ff0
    + ((dec::kD   + 63)/64) * ((dec::kDff+ kDecMaxTileN - 1)/kDecMaxTileN);  // ff2
constexpr int kDecDwHeadTiles =
      ((dec::kVocab + 63)/64) * ((dec::kD + kDecMaxTileN - 1)/kDecMaxTileN);
constexpr int kDecDwMaxTiles = dec::kLayers * kDecDwTilesPerLayer + kDecDwHeadTiles;

// Split-K dW partial-scratch float count (host carves it from the workspace tail).
__host__ __device__ __forceinline__ int64_t dec_dw_part_floats(int G) {
    return (int64_t)kDecDwMaxTiles * G * kDecDwTileFloats;
}

// Decode global dW tile index gt → (spec index s, m_atom, n_tile). Single-source.
template <int N>
__device__ __forceinline__ void dectc_dw_decode(
        const DecDwSpec spec[9], int gt, int& s, int& m_atom, int& n_tile) {
    int acc = 0;
    for (s = 0; s < 9; ++s) {
        const int ma = (spec[s].Nout + 63) / 64;
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ma * nt) { int loc = gt - acc; m_atom = loc / nt; n_tile = loc % nt; return; }
        acc += ma * nt;
    }
    s = 8; m_atom = 0; n_tile = 0;   // unreachable (gt < n_dw); keep defined
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
        const DecDwSpec spec[9], int gt, int kc, int G, float* __restrict__ dw_part,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    int s, m_atom, n_tile;
    dectc_dw_decode<N>(spec, gt, s, m_atom, n_tile);
    const DecDwSpec& sp = spec[s];
    const int mbase = m_atom * 64;
    const int n0 = n_tile * N;
    const int KS = sp.K / wgs::kWgmmaAtomK;                 // total k-atoms (T/16 or B/16)
    const int k0 = (int)(((int64_t)kc       * KS) / G);     // floor-balanced chunk bounds
    const int k1 = (int)(((int64_t)(kc + 1) * KS) / G);
    const int kc_steps = k1 - k0;                          // sums to KS exactly over kc
    float* slot = dw_part + ((int64_t)gt * G + kc) * kDecDwTileFloats;
    // Empty-chunk guard (KS<G, i.e. B<64): a k_steps=0 GEMM would emit the
    // uninitialized accumulator → zero the slot + return instead of running it
    // (the reduce sums all G slots unconditionally, so an empty chunk MUST be 0).
    if (kc_steps <= 0) {
        for (int i = threadIdx.x; i < 64 * N; i += blockDim.x) slot[i] = 0.0f;
        __syncthreads();
        return;
    }
    const int Nout = sp.Nout, Kin = sp.Kin;
    const __nv_bfloat16* dY = sp.dY; const __nv_bfloat16* X = sp.X;
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
        return m < Nout ? dY[(int64_t)(k0 * wgs::kWgmmaAtomK + k) * Nout + m] : __float2bfloat16(0.f); };
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
        int nn = n0 + n; return nn < Kin ? X[(int64_t)(k0 * wgs::kWgmmaAtomK + k) * Kin + nn] : __float2bfloat16(0.f); };
    // out(mbase+row, col, v): m is GLOBAL (srcA needs it); the slot holds only this
    // atom's 64 LOCAL rows → index by (m - mbase). A `m<64` guard would never fire
    // for m_atom>=1 → that slot stays UNWRITTEN (the rel~1.0 dW bug). Local-row fills it.
    auto out  = [&] (int m, int n, float v) {
        const int lr = m - mbase;
        if (lr >= 0 && lr < 64 && n < N) slot[(int64_t)lr * N + n] = v; };
    tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/1>(
        mbase, /*m_atoms=*/1, /*n_real=*/N, kc_steps, srcA, srcB, out, sA, sB);
}

// Deterministic reduce: output tile gt (% nCTA) sums its G chunk-partials ascending-kc
// → grad. Same (gt → geometry) decode as the partial.
template <int N>
__device__ __forceinline__ void dectc_dw_reduce_splitk(
        const DecDwSpec spec[9], int n_dw, int G, const float* __restrict__ dw_part,
        float* __restrict__ grad, int cta, int nCTA) {
    for (int gt = cta; gt < n_dw; gt += nCTA) {
        int s, m_atom, n_tile;
        dectc_dw_decode<N>(spec, gt, s, m_atom, n_tile);
        const DecDwSpec& sp = spec[s];
        const int mbase = m_atom * 64;
        const int n0 = n_tile * N;
        const int n_real = (sp.Kin - n0) < N ? (sp.Kin - n0) : N;
        const int Nrow = (sp.Nout - mbase) < 64 ? (sp.Nout - mbase) : 64;
        const int64_t base = (int64_t)gt * G * kDecDwTileFloats;
        for (int idx = threadIdx.x; idx < Nrow * n_real; idx += blockDim.x) {
            const int row = idx / n_real, col = idx % n_real;
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
        const DecDwSpec spec[9], float* __restrict__ grad, int cta, int nCTA) {
    // exclusive prefix of Nout across the 9 specs → total bias-output count.
    int pre[10];
    pre[0] = 0;
    #pragma unroll
    for (int s = 0; s < 9; ++s) pre[s + 1] = pre[s] + spec[s].Nout;
    const int total = pre[9];
    const int stride = nCTA * blockDim.x;
    for (int go = cta * blockDim.x + threadIdx.x; go < total; go += stride) {
        // decode global output index → (spec s, local row o). 9 specs → linear scan.
        int s = 0;
        #pragma unroll
        for (int t = 0; t < 9; ++t) if (go >= pre[t + 1]) s = t + 1;
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
    // Each CTA reduces a subset of the 10 LN tensors (round-robin by tensor).
    for (int v = cta; v < kNumLnVec; v += nCTA) {
        const int goff = kLnVecTensorIdx[v];
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
