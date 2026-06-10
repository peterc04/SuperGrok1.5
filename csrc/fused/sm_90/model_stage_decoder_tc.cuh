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

namespace dectc {

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

    // One fp32 accumulator fragment per stacked m64 atom (compile-time bound).
    wgs::WgmmaAccum<N> acc[MaxAtomsM];

    // For each stacked atom (M-base = mbase0 + a*64), run the full k-chain.
    #pragma unroll 1
    for (int a = 0; a < m_atoms; ++a) {
        const int mbase = mbase0 + a * wgs::kWgmmaAtomM;
        if (in_wg0) wgs::wgmma_fence();
        #pragma unroll 1
        for (int k = 0; k < k_steps; ++k) {
            // Stage A (64x16) for THIS atom's rows [mbase, mbase+64) and B (Nx16),
            // cooperatively over all 256 threads, into the Major-K smem tiles.
            stage_kmajor_tile<wgs::kWgmmaAtomM>(
                smemA, k * wgs::kWgmmaAtomK,
                [&] (int mn, int kk) { return srcA(mbase + mn, kk); },
                tid, nthreads);
            stage_kmajor_tile<N>(
                smemB, k * wgs::kWgmmaAtomK,
                [&] (int mn, int kk) { return srcB(mn, kk); },
                tid, nthreads);
            __syncthreads();   // staged tile visible to the whole CTA
            if (in_wg0) {
                wgs::SmemDesc dA = wgs::make_desc_A_kmajor<wgs::kWgmmaAtomM, wgs::kSwizzleNone>(smemA);
                wgs::SmemDesc dB = wgs::make_desc_B_kmajor<N, wgs::kSwizzleNone>(smemB);
                if (k == 0)
                    wgs::wgmma_m64nNk16_bf16<N, /*ScaleD=*/0, 0, 0>(acc[a], dA, dB);
                else
                    wgs::wgmma_m64nNk16_bf16<N, /*ScaleD=*/1, 0, 0>(acc[a], dA, dB);
                wgs::wgmma_commit_group();
                wgs::wgmma_wait_group<0>();
            }
            __syncthreads();   // MMA done reading smem before next stage overwrites
        }
        // Epilogue: warpgroup 0 owns the fp32 fragment; decode + emit (real cols).
        if (in_wg0) {
            #pragma unroll
            for (int i = 0; i < wgs::WgmmaAccum<N>::kRegs; ++i) {
                int row, col;
                wgs::wgmma_frag_decode(tid_wg, i, N, row, col);
                if (col < n_real) out(mbase + row, col, acc[a].c[i]);
            }
        }
        __syncthreads();
    }
#else
    (void)mbase0; (void)m_atoms; (void)n_real; (void)k_steps;
    (void)srcA; (void)srcB; (void)out; (void)smemA; (void)smemB;
#endif
}

}  // namespace dectc

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGE_DECODER_TC_CUH_
