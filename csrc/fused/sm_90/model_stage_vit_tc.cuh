#ifndef SG_FUSED_SM90_MODEL_STAGE_VIT_TC_CUH_
#define SG_FUSED_SM90_MODEL_STAGE_VIT_TC_CUH_
// ============================================================================
// csrc/fused/sm_90/model_stage_vit_tc.cuh — R2 TENSOR-CORE variant of the
// L3-REAL Vision-Transformer fwd+bwd. The batch-tiled bf16 wgmma path
// (DESIGN-TC-PIPELINE.md Fork B), the ViT TWIN of model_stage_decoder_tc.cuh.
// It is a TUNED VARIANT compiled ALONGSIDE the scalar model_stage_vit.cuh and
// selected per-cell by SG_TUNED_GEMM_IMPL (the owner directive: BOTH paths
// compiled, the tuner picks). The scalar path's math + gates are UNCHANGED;
// nothing here edits model_stage_vit.cuh.
//
// WHY A NEW BODY (not an in-place edit): the wgmma atom is m64nNk16 — it needs
// >= 64 M-rows per issue, but one ViT sample is kSeq=17 rows. So the scalar
// "one CTA owns a batch slice, one sample at a time" model CANNOT use tensor
// cores. The TC path batches across samples: M = (sample × position) TOKEN rows.
// Each CTA owns a contiguous tile of SG_TUNED_TILE_M token rows; the tile
// boundary lands on a SAMPLE boundary (kTileM is a multiple of kSeq) so each
// sample's 17×17 attention stays fully within one tile. This is a genuine
// rewrite, which is why DESIGN hands it a separate header.
//
// THE FOUR ViT DELTAS vs the decoder TC twin (the contract DELTAS):
//   1. FULL (bidirectional) attention — NO causal mask. Every key position kj
//      contributes for every query qi (the decoder masked kj>qi). The fwd/bwd
//      attention-tile helpers drop the triangle.
//   2. PATCH-PROJ Linear(49→128) REPLACES the token embedding. Per-patch embed
//      (16 patches/sample) with its OWN fwd / dX(none — input is data) / dW GEMM
//      (K=49 padded to 64). The CLS token (a learned [d] vector) is prepended at
//      position 0; pos[17] is a learned table added to all 17 positions.
//   3. CLS pos-0 HEAD — the final-norm + head + CE run on position 0 of each
//      sample (the decoder used the LAST position, kSeq-1).
//   4. kSeq=17 → kTileM a MULTIPLE OF 17 (and of 64, the wgmma atom). The least
//      such is LCM(17,64)=1088 (= 17 stacked m64 atoms = 64 samples × 17 pos).
//
// FORK B (DESIGN §2/§3, dW-output-stationary):
//   * P1 token-tile-parallel fwd + bwd-dX through ALL layers, barrier-free
//     within the tile. The per-token activations the cross-tile dW owners need
//     (linear INPUTS X and OUTPUT adjoints dY, plus the embedding-input adjoint
//     dh0) are written to an HBM bf16 acts buffer (DecActs→VitActs pattern).
//   * P2 dW-output-stationary: each weight-matrix dW tile is owned by ONE CTA
//     (tile_id % nCTA) which contracts the FULL token dimension T itself
//     (ascending-t, no float atomics → deterministic), streaming dY and X from
//     HBM. No [nCTA × total] partial, no cross-CTA dW reduce.
//
// GRAD OWNERSHIP (all 32 tensors — every grad is a Σ-over-T/Tp):
//   * patch_proj.weight  → output-stationary dW GEMM (dh0_patch^T @ patches,
//        K=Tp=nsamp·16, the patch rows only)                                [P2]
//   * 9 weight MATRICES (in/out/ff0/ff2 ×L + head) → dW GEMM (wgmma, K=T)    [P2]
//   * 10 BIASES db=Σ dY (8 linear + patch_proj.bias + head bias)            [P2]
//   * cls_token grad = Σ_samp dh0[CLS row]   (owner-scan over CLS rows)      [P2]
//   * pos.weight grad = Σ_t dh0 by within-sample position                   [P2]
//   * 10 LN affine (γ/β) → tile-local in P1 into a per-CTA partials buffer,
//        then a deterministic ascending-CTA reduce                          [P2]
//
// PRECISION (DESIGN §5.1 — torch-autocast boundary):
//   * the six linear families (patch_proj/in/out/ff0/ff2/head + all dX/dW):
//     bf16 operands, wgmma, fp32 accumulator, bf16 acts / fp32 grad.
//   * attention scores/softmax, LayerNorm, GELU, cross-entropy: fp32 (kept
//     identical to the scalar oracle math). S=17 is tiny so scores/ctx stay the
//     per-sample fp32 special-case.
//
// VALIDATION: the wgmma engine + pipeline are silicon-validated by
//   tests/hw/test_wgmma_substrate.py + tests/hw/test_decoder_tc.py (the engine
//   is model-agnostic). This header is gated by tests/hw/test_vit_tc.py
//   (per-orientation micro-gates + full-cell grad parity vs the bf16-rounded
//   oracle + determinism + grok-floor). The scalar path's fp32 gates are
//   untouched.
//
// PORTABILITY: arch-guarded on __CUDA_ARCH__ >= 900. The substrate falls back to
//   scalar pre-sm_90; the cell driver only selects this body on sm_90 builds.
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/vit_layout.cuh"
#include "csrc/fused/sm_90/model_stage_vit.cuh"   // reuse VitWeights/VitGrad/bind + fp32 helpers
#include "csrc/backends/cuda/sm_90/wgmma.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math_constants.h>

namespace sg { namespace fused { namespace sm90 {

namespace wgs = ::sg::sm90::wgs;

// ── Tunable knobs (DESIGN §9). #ifndef defaults compose a correct untuned
//    kernel (CONTRACT rule 3). SG_TUNED_TILE_N is shared with the substrate;
//    SG_TUNED_VIT_TILE_M is ViT-specific (the LCM(17,64)=1088 token tile —
//    distinct from the decoder's 128, since the tile boundary MUST be a sample
//    boundary and a ViT sample is 17 rows). ─────────────────────────────────
#ifndef SG_TUNED_TILE_N
#define SG_TUNED_TILE_N 128
#endif
#ifndef SG_TUNED_VIT_TILE_M
#define SG_TUNED_VIT_TILE_M 1088
#endif
// ── dW split-K factor (the validated decoder/mamba multi-CTA-tiling win, carried
//    to ViT). The 10 dW yield only ~52 output tiles → ~60% of 132 SMs idle in the
//    P2 dW phase. Split-K chunks each tile's K-contraction across G CTAs so the
//    grid sees (n_dw·G) work items → idle SMs do work; a deterministic ascending-
//    chunk reduce sums the G partials (no float atomics, fixed order → parity +
//    A/A/A bit-determinism hold). G=4 → 52·4≈208 items ≥ 132 SMs (full saturation,
//    matching SG_TUNED_DEC_DW_SPLITK). G==1 routes to the single-CTA path (no
//    scratch) — the byte-identical pre-split behaviour. ─────────────────────────
#ifndef SG_TUNED_VIT_DW_SPLITK
#define SG_TUNED_VIT_DW_SPLITK 4
#endif

namespace vittc {

constexpr int kVitDwSplitK = SG_TUNED_VIT_DW_SPLITK;
static_assert(kVitDwSplitK >= 1, "SG_TUNED_VIT_DW_SPLITK must be >= 1");

// ── Muon 2D-weight table (the matrices Newton-Schulz orthogonalizes). The eager
//    Muon auto-splits params by p.ndim: ndim==2 → NS, everything else → AdamW
//    (muon.py _split_by_ndim; muon.h:75-76). For the small ViT the ndim==2 weights
//    are exactly these 11 (the flat named_parameters() tensor index + rows + cols);
//    cls_token (ndim==3), all biases + LayerNorm γ/β (ndim==1) take the AdamW 1D
//    tail. The kernel's Muon P2.7 loops THIS table running the grid-cooperative NS
//    per matrix; P3 routes tensor t to the NS apply iff it is in the table, else the
//    AdamW tail. Indices MUST match vit_layout / named_parameters() order. ──
constexpr int kVitNumMuon2D = 11;
struct VitMuon2D { int tidx; int rows; int cols; };
__device__ __constant__ VitMuon2D kVitMuon2D[kVitNumMuon2D] = {
    { 1, vit::kD,      vit::kPatch },   // patch_proj.weight  [128,49]
    { 3, vit::kSeq,    vit::kD     },   // pos.weight         [17,128]
    { 4, 3*vit::kD,    vit::kD     },   // L0 in_proj_weight  [384,128]
    { 6, vit::kD,      vit::kD     },   // L0 out_proj.weight [128,128]
    {12, vit::kDff,    vit::kD     },   // L0 ff.0.weight     [512,128]
    {14, vit::kD,      vit::kDff   },   // L0 ff.2.weight     [128,512]
    {16, 3*vit::kD,    vit::kD     },   // L1 in_proj_weight  [384,128]
    {18, vit::kD,      vit::kD     },   // L1 out_proj.weight [128,128]
    {24, vit::kDff,    vit::kD     },   // L1 ff.0.weight     [512,128]
    {26, vit::kD,      vit::kDff   },   // L1 ff.2.weight     [128,512]
    {30, vit::kVocab,  vit::kD     },   // out.weight         [97,128]
};
// Is tensor index `t` one of the Muon 2D matrices (orthogonalized in P2.7)? P3 uses
// this to route ONLY the 1D / non-2D weights to the AdamW tail for Muon.
__device__ __forceinline__ bool vit_is_muon_2d(int t) {
    #pragma unroll
    for (int mi = 0; mi < kVitNumMuon2D; ++mi) if (kVitMuon2D[mi].tidx == t) return true;
    return false;
}

#ifdef SG_VIT_PROFILE
// Diagnostic-only (SG_VIT_PROFILE; never shipped): summed clock64 cycles spent in
// the per-sample head/CE loops (fwd [0], bwd [1]), accumulated across all tiles a
// CTA runs, max across CTAs (atomicMax). Read host-side via cudaMemcpyFromSymbol.
__device__ unsigned long long g_vit_prof_head[2];
#endif

// Token-tile rows a CTA owns. Must be a multiple of 64 (wgmma atom M) AND of
// kSeq=17 (so a tile boundary is a sample boundary — attention stays in-tile).
constexpr int kTileM = SG_TUNED_VIT_TILE_M;
static_assert(kTileM % wgs::kWgmmaAtomM == 0,
              "SG_TUNED_VIT_TILE_M must be a multiple of 64 (wgmma m64 atom)");
static_assert(kTileM % vit::kSeq == 0,
              "SG_TUNED_VIT_TILE_M must be a multiple of kSeq=17 (tile=sample boundary)");
constexpr int kAtomsM = kTileM / wgs::kWgmmaAtomM;   // stacked m64 atoms per tile (17)
constexpr int kSamplesPerTile = kTileM / vit::kSeq;  // 64 for TILE_M=1088
constexpr int kPatchPerTile = kSamplesPerTile * vit::kNPatch;  // patch rows per tile

// ── LN vector-grad partials layout (the 10 tile-local γ/β grads). Order MUST
//    match the vit_layout tensor indices of {n1.w, n1.b, n2.w, n2.b}×L plus
//    {norm.w, norm.b}. We store them densely [10 × kD] per CTA; the P2 reduce
//    maps them back by tensor index. (vit_layout order: per-layer block starts
//    at 4 + li*12, with n1.w/b at +4/+5, n2.w/b at +6/+7; norm.w/b at 28/29.) ─
constexpr int kNumLnVec = 10;                  // n1_w,n1_b,n2_w,n2_b ×L + norm_w,norm_b
constexpr int kLnVecElems = kNumLnVec * vit::kD;   // 10 * 128 = 1280
// The vit_layout tensor index of each LN-vector slot, in our dense order.
__device__ __constant__ int kLnVecTensorIdx[kNumLnVec] = {
    8, 9, 10, 11,      // L0 n1.w, n1.b, n2.w, n2.b  (4 + 0*12 + {4,5,6,7})
    20, 21, 22, 23,    // L1 n1.w, n1.b, n2.w, n2.b  (4 + 1*12 + {4,5,6,7})
    28, 29             // norm.w, norm.b
};

// ════════════════════════════════════════════════════════════════════════
//  HBM bf16 ACTS buffer (Fork B). Carved from the FRONT of the workspace the
//  host allocates. Reinterpreted as __nv_bfloat16. All regions row-major
//  [rows × width]. Offsets are RUNTIME (depend on T = B*kSeq and B), computed
//  by vit_acts_bind. The struct holds base pointers per region.
//
//  ViT ADDS X_patch (the float→bf16 patch input the patch_proj dW reads) and
//  REPLACES the decoder's tok/pos embedding adjoint plumbing with dh0 (the
//  post-cat, pre-pos grad) whose CLS rows feed cls_token and whose patch rows
//  feed patch_proj; pos is Σ over all rows by position.
// ════════════════════════════════════════════════════════════════════════
struct VitActs {
    // Patch-proj input (the image patches as bf16, laid out as PATCH token rows
    // [Tp, kPatch] where Tp = nsamp*kNPatch — patch_proj dW = dh0_patch^T @ this):
    __nv_bfloat16* X_patch;               // [Tp, kPatch]
    // Per-layer linear INPUTS X (needed by dW = dY^T @ X):
    __nv_bfloat16* X_in[vit::kLayers];    // [T, d]    in_proj input (= layer input)
    __nv_bfloat16* X_ctx[vit::kLayers];   // [T, d]    out_proj input (attn context)
    __nv_bfloat16* X_x1[vit::kLayers];    // [T, d]    ff0 input (n1 output)
    __nv_bfloat16* X_gact[vit::kLayers];  // [T, dff]  ff2 input (gelu output)
    // Per-layer linear OUTPUT adjoints dY (needed by dW + bias db = Σ_t dY):
    __nv_bfloat16* dY_qkv[vit::kLayers];  // [T, 3d]   in_proj output adjoint
    __nv_bfloat16* dY_a[vit::kLayers];    // [T, d]    out_proj output adjoint
    __nv_bfloat16* dY_ff0[vit::kLayers];  // [T, dff]  ff0 output adjoint
    __nv_bfloat16* dY_ff2[vit::kLayers];  // [T, d]    ff2 output adjoint
    // Head (B rows — CLS position only):
    __nv_bfloat16* X_hn;                  // [B, d]    head input
    __nv_bfloat16* dY_logits;             // [B, V]    head output adjoint (dlogits)
    // Embedding-input adjoint (post-cat, pre-pos): cls/patch/pos owners read it:
    __nv_bfloat16* dh0;                   // [T, d]
};

__device__ __forceinline__ VitActs vit_acts_bind(__nv_bfloat16* p, int T, int B) {
    VitActs a;
    int64_t off = 0;
    const int Tp = (T / vit::kSeq) * vit::kNPatch;
    const int64_t Td = (int64_t)T * vit::kD;
    const int64_t T3d = (int64_t)T * 3 * vit::kD;
    const int64_t Tff = (int64_t)T * vit::kDff;
    a.X_patch = p + off; off += (int64_t)Tp * vit::kPatch;
    for (int li = 0; li < vit::kLayers; ++li) {
        a.X_in[li]   = p + off; off += Td;
        a.X_ctx[li]  = p + off; off += Td;
        a.X_x1[li]   = p + off; off += Td;
        a.X_gact[li] = p + off; off += Tff;
        a.dY_qkv[li] = p + off; off += T3d;
        a.dY_a[li]   = p + off; off += Td;
        a.dY_ff0[li] = p + off; off += Tff;
        a.dY_ff2[li] = p + off; off += Td;
    }
    a.X_hn      = p + off; off += (int64_t)B * vit::kD;
    a.dY_logits = p + off; off += (int64_t)B * vit::kVocab;
    a.dh0       = p + off; off += Td;
    return a;
}

// Total bf16 element count of the acts region (host sizing mirror).
__host__ __device__ __forceinline__ int64_t vit_acts_bf16_count(int T, int B) {
    const int64_t d = vit::kD, dff = vit::kDff, V = vit::kVocab, L = vit::kLayers;
    const int Tp = (T / vit::kSeq) * vit::kNPatch;
    const int64_t Td = (int64_t)T * d, T3d = (int64_t)T * 3 * d, Tff = (int64_t)T * dff;
    int64_t bf = (int64_t)Tp * vit::kPatch;                // X_patch
    for (int li = 0; li < L; ++li) bf += Td + Td + Td + Tff + T3d + Td + Tff + Td;
    bf += (int64_t)B * d + (int64_t)B * V + Td;            // X_hn + dY_logits + dh0
    return bf;
}

// ════════════════════════════════════════════════════════════════════════
//  Canonical Major-K smem stager (LIFTED verbatim from the decoder TC twin —
//  the wgmma engine is model-agnostic). Writes ONE MN×16 tile in the CUTLASS
//  Major-K INTERLEAVE layout (wgmma.cuh):
//      idx(mn,k) = (k/8)*(MN*8) + mn*8 + (k%8)
//  pulling element (mn, k) from a caller accessor `src(mn, kbase + k)`. Routing
//  the source axis through the accessor lets the SAME wgmma issue (TransA=0/
//  TransB=0, the substrate-validated orientation) serve fwd / dX / dW.
//  Cooperative over `nthreads` threads starting at thread `t0`.
// ════════════════════════════════════════════════════════════════════════
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
//  Unpipelined single-CTA batch GEMM: D[M,N] = Σ_k A[m,k]·B[n,k], bf16 operands
//  (accessor-sourced + Major-K staged), fp32 accumulator, ascending-k. LIFTED
//  verbatim from the decoder TC twin (the engine is model-agnostic; only the
//  MaxAtomsM bound differs at the call sites — ViT's token tile is 17 atoms).
//
//  This helper computes ONE M-block of `m_atoms` stacked m64 atoms (rows
//  [mbase0, mbase0 + m_atoms*64)) for ONE N-tile of the compile-time atom width
//  N. The CALLER loops M-atom-blocks × N-tiles to cover arbitrary (M, N).
//  MaxAtomsM bounds the register accumulator array (compile-time); only
//  `m_atoms` (<= MaxAtomsM) atoms are issued at runtime. `n_real <= N` marks
//  the valid column count for a ragged N-tile (pad operands are zero → inert).
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

    // ── CODEGEN FIX (correctness/codegen, NOT a perf knob): use ONE accumulator
    // reused per m-atom, NOT `WgmmaAccum<N> acc[MaxAtomsM]`. Each m-atom's
    // accumulator is FULLY CONSUMED (stored via out()) at the end of its
    // iteration before the next m-atom's k-loop begins (k==0 uses ScaleD=0 →
    // overwrite, no carry across atoms), so a single live accumulator is
    // numerically identical. The array version indexed `acc[a]` under the
    // `#pragma unroll 1` runtime-`a` loop, which GPU registers cannot address
    // dynamically → ptxas placed the 17-wide (vit kAtomsM) array in LOCAL memory
    // (5024 B stack frame) and emitted C7515: "wgmma.mma_async serialized due to
    // non-wgmma instructions DEFINING accumulator registers" — i.e. every wgmma
    // was forced to fully serialize around the local-mem round-trip of acc[a].
    // The single register-resident accumulator removes that serialization. The
    // ascending-k order + one-CTA tile ownership (determinism) are unchanged.
    #pragma unroll 1
    for (int a = 0; a < m_atoms; ++a) {
        wgs::WgmmaAccum<N> acc;
        const int mbase = mbase0 + a * wgs::kWgmmaAtomM;
        if (in_wg0) wgs::wgmma_fence();
        #pragma unroll 1
        for (int k = 0; k < k_steps; ++k) {
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
                    wgs::wgmma_m64nNk16_bf16<N, /*ScaleD=*/0, 0, 0>(acc, dA, dB);
                else
                    wgs::wgmma_m64nNk16_bf16<N, /*ScaleD=*/1, 0, 0>(acc, dA, dB);
                wgs::wgmma_commit_group();
                wgs::wgmma_wait_group<0>();
            }
            __syncthreads();   // MMA done reading smem before next stage overwrites
        }
        if (in_wg0) {
            #pragma unroll
            for (int i = 0; i < wgs::WgmmaAccum<N>::kRegs; ++i) {
                int row, col;
                wgs::wgmma_frag_decode(tid_wg, i, N, row, col);
                if (col < n_real) out(mbase + row, col, acc.c[i]);
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
//  THIN ORIENTATION WRAPPERS over tc_gemm_block_unpipelined — the THREE
//  accessor patterns the engine is silicon-validated on (fwd / dX / dW). The
//  driver calls THESE; it never re-derives the staging. N is the compile-time
//  wgmma atom width; the fwd/dX loops N-tiles internally. Weights convert
//  fp32→bf16 ON READ (deterministic; no bf16 weight buffer needed). Kin is
//  padded to a multiple of kWgmmaAtomK by the caller (49→64 for patch_proj).
// ════════════════════════════════════════════════════════════════════════

// (fwd) Y[M,Nout] = X[M,Kpad] @ W[Nout,Kin]ᵀ.  Tiles N over [0,Nout) in width-N
// atoms. M = m_atoms stacked atoms (kAtomsM for token tiles; fewer for the last
// ragged tile / patch tiles). Y bf16 row-major [M, Nout]. `Kreal` is the true
// contracted dim (<= Kpad); pad-K reads return 0 so they are inert.
template <int N>
__device__ __forceinline__ void vittc_gemm_fwd(
        const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
        __nv_bfloat16* __restrict__ Yout, int Kreal, int Kpad, int Nout,
        int m_atoms, __nv_bfloat16* sA, __nv_bfloat16* sB) {
    const int k_steps = Kpad / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
            return k < Kreal ? X[(int64_t)m * Kreal + k] : __float2bfloat16(0.f); };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return (nn < Nout && k < Kreal) ? __float2bfloat16(W[(int64_t)nn * Kreal + k]) : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) {
            Yout[(int64_t)m * Nout + n0 + n] = __float2bfloat16(v); };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, m_atoms, n_real, k_steps, srcA, srcB, out, sA, sB);
    }
}

// Same as vittc_gemm_fwd but emits the fp32 result (no bf16 round) — for fwd
// outputs consumed by fp32 elementwise stages. Writes [M,Nout] fp32 at `Yf32`.
// (K is never padded for these — only the in/out/ff GEMMs use it, all K∈{d,dff}
// already multiples of 16.)
template <int N>
__device__ __forceinline__ void vittc_gemm_fwd_f32(
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

// (dX) dX[M,Kin] = dY[M,Nout] @ W[Nout,Kin].  N(wgmma)=Kin (tiled by width N),
// K=Nout. W staged transposed: srcB(n=kin,k=out)=W[out·Kin+kin] (fp32→bf16).
// Writes fp32 dX [M,Kin].
template <int N>
__device__ __forceinline__ void vittc_gemm_dx_f32(
        const __nv_bfloat16* __restrict__ dY, const float* __restrict__ W,
        float* __restrict__ dXf32, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    const int k_steps = Nout / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Kin; n0 += N) {
        const int n_real = (Kin - n0) < N ? (Kin - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return dY[(int64_t)m * Nout + k]; };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return nn < Kin ? __float2bfloat16(W[(int64_t)k * Kin + nn]) : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) { dXf32[(int64_t)m * Kin + n0 + n] = v; };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB);
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Per-CTA TILE SCRATCH (HBM). One tile's forward intermediates the backward
//  reads, reused across the tiles a CTA grid-strides over. Sized for kTileM
//  rows. DEDICATED, NON-ALIASED buffers per fp32 intermediate (HBM is not
//  scarce; the aliasing bug class is avoided). qkv/ff0pre/attn/n1·n2 caches are
//  [kLayers]-indexed (the fwd runs ALL layers then bwd runs ALL layers, so each
//  layer's bwd-read state must persist per layer — the "layer-0 grads wrong"
//  bug). fnx/fni (final norm) + transient dh/x1/finalin/logits/work/work2/dsc
//  stay single.
// ════════════════════════════════════════════════════════════════════════
constexpr int kNSampPerTile  = kTileM / vit::kSeq;
constexpr int kAttnPerTile   = kNSampPerTile * vit::kHeads * vit::kSeq * vit::kSeq;
constexpr int kLogitsPerTile = kNSampPerTile * vit::kVocab;

struct VitTileScratch {
    __nv_bfloat16* qkv[vit::kLayers];     // [kTileM, 3d]  per layer
    __nv_bfloat16* ff0pre[vit::kLayers];  // [kTileM, dff] per layer
    float* attn[vit::kLayers];            // [kAttnPerTile] per layer
    float* n1x[vit::kLayers]; float* n1i[vit::kLayers];
    float* n2x[vit::kLayers]; float* n2i[vit::kLayers];
    float* dsc;             // [kAttnPerTile] attention dscores (transient, bwd-only)
    float* fnx; float* fni; // final-norm LN caches (single)
    float* dh;              // [kTileM, d]    running adjoint wrt block output
    float* x1;              // [kTileM, d]    n1 output (fp32, residual base for r2)
    float* finalin;         // [kTileM, d]    last-layer n2 output (fp32, head input)
    float* logits;          // [kLogitsPerTile] per-sample CLS-pos logits (fp32)
    float* work;            // [kTileM, dff]  GEMM output / general fp32 scratch
    float* work2;           // [kTileM, dff]  second fp32 scratch (bwd dx1/dqkv)
};

__host__ __device__ __forceinline__ int64_t vit_tile_scratch_bf16_count() {
    return (int64_t)vit::kLayers * ((int64_t)kTileM * 3 * vit::kD + (int64_t)kTileM * vit::kDff);
}
__host__ __device__ __forceinline__ int64_t vit_tile_scratch_f32_count() {
    return (int64_t)vit::kLayers * (
             (int64_t)kAttnPerTile
           + 2 * ((int64_t)kTileM * vit::kD + kTileM))
         + (int64_t)kAttnPerTile                          // dsc (single)
         + ((int64_t)kTileM * vit::kD + kTileM)           // fn xhat+inv (single)
         + (int64_t)kTileM * vit::kD                      // dh
         + (int64_t)kTileM * vit::kD                      // x1
         + (int64_t)kTileM * vit::kD                      // finalin
         + (int64_t)kLogitsPerTile                        // logits
         + 2 * (int64_t)kTileM * vit::kDff;               // work + work2
}
__host__ __device__ __forceinline__ int64_t vit_tile_scratch_total_f32() {
    return (vit_tile_scratch_bf16_count() + 1) / 2 + vit_tile_scratch_f32_count();
}

__device__ __forceinline__ VitTileScratch vit_tile_scratch_bind(float* slab) {
    VitTileScratch s;
    __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(slab);
    int64_t bo = 0;
    for (int li = 0; li < vit::kLayers; ++li) { s.qkv[li]    = b + bo; bo += (int64_t)kTileM * 3 * vit::kD; }
    for (int li = 0; li < vit::kLayers; ++li) { s.ff0pre[li] = b + bo; bo += (int64_t)kTileM * vit::kDff; }
    float* f = slab + (vit_tile_scratch_bf16_count() + 1) / 2;
    int64_t fo = 0;
    for (int li = 0; li < vit::kLayers; ++li) { s.attn[li] = f + fo; fo += kAttnPerTile; }
    for (int li = 0; li < vit::kLayers; ++li) { s.n1x[li] = f + fo; fo += (int64_t)kTileM * vit::kD; s.n1i[li] = f + fo; fo += kTileM; }
    for (int li = 0; li < vit::kLayers; ++li) { s.n2x[li] = f + fo; fo += (int64_t)kTileM * vit::kD; s.n2i[li] = f + fo; fo += kTileM; }
    s.dsc  = f + fo; fo += kAttnPerTile;
    s.fnx  = f + fo; fo += (int64_t)kTileM * vit::kD;
    s.fni  = f + fo; fo += kTileM;
    s.dh   = f + fo; fo += (int64_t)kTileM * vit::kD;
    s.x1   = f + fo; fo += (int64_t)kTileM * vit::kD;
    s.finalin = f + fo; fo += (int64_t)kTileM * vit::kD;
    s.logits  = f + fo; fo += kLogitsPerTile;
    s.work    = f + fo; fo += (int64_t)kTileM * vit::kDff;
    s.work2   = f + fo; fo += (int64_t)kTileM * vit::kDff;
    return s;
}

// ════════════════════════════════════════════════════════════════════════
//  TILE-AWARE SCALAR ELEMENTWISE STAGES (fp32, CTA-cooperative over kTileM
//  rows). Mirror the scalar oracle's per-row math (model_stage_vit.cuh) but
//  operate on a whole tile of `nrows` rows at once over HBM [rows×width].
//  Reductions reuse vit_block_sum / vit_block_max (whole-block, looped per row).
//  `red` is a 256-float smem reduction slot.
// ════════════════════════════════════════════════════════════════════════

// LayerNorm fwd over the last dim d, for `nrows` rows. y/xhat fp32 HBM [rows×d];
// inv fp32 HBM [rows]. gamma/beta fp32 [d]. Caches xhat+inv for the bwd.
__device__ __forceinline__ void vittc_ln_fwd_tile(
        const float* __restrict__ x, const float* __restrict__ gamma,
        const float* __restrict__ beta, int nrows,
        float* __restrict__ y, float* __restrict__ xhat, float* __restrict__ inv,
        float* red) {
    for (int s = 0; s < nrows; ++s) {
        const float* xr = x + (int64_t)s * vit::kD;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) sum += xr[j];
        float mean = vit_block_sum(sum, red) / (float)vit::kD;
        float vs = 0.0f;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) { float c = xr[j] - mean; vs += c * c; }
        float var = vit_block_sum(vs, red) / (float)vit::kD;
        float iv = rsqrtf(var + vit::kLnEps);
        if (threadIdx.x == 0) inv[s] = iv;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float xh = (xr[j] - mean) * iv;
            xhat[(int64_t)s * vit::kD + j] = xh;
            y[(int64_t)s * vit::kD + j] = xh * gamma[j] + beta[j];
        }
        __syncthreads();
    }
}

// LayerNorm bwd for `nrows` rows: dy [rows×d] fp32, cached xhat/inv → dx [rows×d]
// fp32; ACCUMULATES dgamma/dbeta (summed over the tile's rows) into a per-CTA
// LN-vec partial slot gw/gb [d] (plain += : single owner thread per feature j
// across rows, deterministic — same rule as the scalar vit_layernorm_bwd).
__device__ __forceinline__ void vittc_ln_bwd_tile(
        const float* __restrict__ dy, const float* __restrict__ xhat,
        const float* __restrict__ inv, const float* __restrict__ gamma, int nrows,
        float* __restrict__ dx, float* __restrict__ gw, float* __restrict__ gb,
        float* red) {
    for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
        float dgw = 0.0f, dgb = 0.0f;
        for (int s = 0; s < nrows; ++s) {
            float d = dy[(int64_t)s * vit::kD + j];
            dgb += d; dgw += d * xhat[(int64_t)s * vit::kD + j];
        }
        gw[j] += dgw; gb[j] += dgb;
    }
    for (int s = 0; s < nrows; ++s) {
        const float* dyr = dy + (int64_t)s * vit::kD;
        const float* xhr = xhat + (int64_t)s * vit::kD;
        float sda = 0.0f, sdax = 0.0f;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float dxhat = dyr[j] * gamma[j]; sda += dxhat; sdax += dxhat * xhr[j];
        }
        sda = vit_block_sum(sda, red);
        sdax = vit_block_sum(sdax, red);
        float iv = inv[s];
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float dxhat = dyr[j] * gamma[j];
            dx[(int64_t)s * vit::kD + j] = iv * (dxhat - (sda + xhr[j] * sdax) / (float)vit::kD);
        }
        __syncthreads();
    }
}

// Per-sample FULL (bidirectional) self-attention FORWARD over a tile. qkv is
// bf16 HBM [rows×3d] (q|k|v). Writes ctx fp32 HBM [rows×d] and attn weights fp32
// to `attn_w` [nSamp×H×S×S]. Each (sample,head,qpos) row owned by one thread —
// identical math to vit_attention, looped over samples. DELTA 1 vs decoder: NO
// causal mask — every key kj contributes for every query qi.
__device__ __forceinline__ void vittc_attn_fwd_tile(
        const __nv_bfloat16* __restrict__ qkv, int nrows,
        float* __restrict__ ctx, float* __restrict__ attn_w) {
    const int nsamp = nrows / vit::kSeq;
    const float scale = vit::attn_scale();
    const int rows_per = nsamp * vit::kHeads * vit::kSeq;   // (sample,head,qpos)
    for (int r = threadIdx.x; r < rows_per; r += blockDim.x) {
        const int si = r / (vit::kHeads * vit::kSeq);
        const int rem = r % (vit::kHeads * vit::kSeq);
        const int hh = rem / vit::kSeq, qi = rem % vit::kSeq;
        const int qoff = hh * vit::kDhead;
        const int rbase = si * vit::kSeq;        // first row of this sample
        const __nv_bfloat16* qrow = qkv + (int64_t)(rbase + qi) * 3 * vit::kD + qoff;
        float maxs = -CUDART_INF_F; float sc[vit::kSeq];
        #pragma unroll
        for (int kj = 0; kj < vit::kSeq; ++kj) {   // FULL: every key position
            const __nv_bfloat16* krow = qkv + (int64_t)(rbase + kj) * 3 * vit::kD + vit::kD + qoff;
            float dot = 0.0f;
            #pragma unroll
            for (int t = 0; t < vit::kDhead; ++t)
                dot += __bfloat162float(qrow[t]) * __bfloat162float(krow[t]);
            sc[kj] = dot * scale; maxs = fmaxf(maxs, sc[kj]);
        }
        float denom = 0.0f;
        #pragma unroll
        for (int kj = 0; kj < vit::kSeq; ++kj) { float e = __expf(sc[kj] - maxs); sc[kj] = e; denom += e; }
        float invd = 1.0f / denom;
        float* aw = attn_w + ((int64_t)(si * vit::kHeads + hh) * vit::kSeq + qi) * vit::kSeq;
        #pragma unroll
        for (int kj = 0; kj < vit::kSeq; ++kj) aw[kj] = sc[kj] * invd;
        #pragma unroll
        for (int t = 0; t < vit::kDhead; ++t) {
            float acc = 0.0f;
            #pragma unroll
            for (int kj = 0; kj < vit::kSeq; ++kj) {  // FULL: all kj
                float vv = __bfloat162float(qkv[(int64_t)(rbase + kj) * 3 * vit::kD + 2 * vit::kD + qoff + t]);
                acc += aw[kj] * vv;
            }
            ctx[(int64_t)(rbase + qi) * vit::kD + qoff + t] = acc;
        }
    }
    __syncthreads();
}

// Global SAMPLE index of the si-th sample in a tile whose first token row is g0.
__device__ __forceinline__ int si_global(int g0, int si) { return g0 / vit::kSeq + si; }

// ════════════════════════════════════════════════════════════════════════
//  FORWARD over one TOKEN TILE (nrows = nsamp samples × kSeq positions), global
//  token rows [g0, g0+nrows). Tile-batched: the four per-layer linears run on
//  wgmma (M=nrows, N-tiled); attention/LN/GELU scalar fp32; head/CE scalar
//  per-sample. Writes VitActs X-inputs (bf16 dW operands), the per-CTA tile
//  scratch, and returns the tile's summed NLL (thread 0 holds it).
//
//  DELTA 2 (patch_proj embedding): X_in[0][CLS row] = cls + pos[0];
//    X_in[0][patch row 1+i] = patch_proj(patch_i) + pos[1+i]. The patch_proj is
//    a wgmma fwd GEMM over the tile's patch rows (M = nsamp*16 = kPatchPerTile,
//    K=49 padded to 64). The float patches are staged into X_patch (bf16) first,
//    then patch_proj reads them.
//  `patches` is HBM float [B, 16, 49]; `tgt_ids` HBM int32 [B].
// ════════════════════════════════════════════════════════════════════════
__device__ float vittc_forward_tile(
        const VitWeights& w, int g0, int nrows, const VitActs& acts,
        const VitTileScratch& sc, const float* __restrict__ patches,
        const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red) {
    const int nsamp = nrows / vit::kSeq;
    const int npatch_rows = nsamp * vit::kNPatch;
    const int gp0 = (g0 / vit::kSeq) * vit::kNPatch;   // first PATCH row of this tile
    // ── Stage this tile's patches into X_patch (bf16) as patch token rows.
    //    Sample si's patch p is global patch row gp0 + si*kNPatch + p; the source
    //    is patches[(g0/kSeq + si)*kNPatch*kPatch + p*kPatch + c]. ──
    for (int idx = threadIdx.x; idx < npatch_rows * vit::kPatch; idx += blockDim.x) {
        const int pr = idx / vit::kPatch, c = idx % vit::kPatch;
        const int si = pr / vit::kNPatch, p = pr % vit::kNPatch;
        const int gs = si_global(g0, si);
        float v = patches[((int64_t)gs * vit::kNPatch + p) * vit::kPatch + c];
        acts.X_patch[(int64_t)(gp0 + pr) * vit::kPatch + c] = __float2bfloat16(v);
    }
    __syncthreads();
    // ── patch_proj: proj[pr, :] = X_patch[pr,:49] @ patch_w[d,49]ᵀ + patch_b.
    //    fp32 → sc.work (patch rows). K=49 padded to 64 (one m64 atom block per
    //    64 patch rows). M = npatch_rows (= nsamp*16) issued as ceil/64 atoms. ──
    {
        const int pm_atoms = (npatch_rows + wgs::kWgmmaAtomM - 1) / wgs::kWgmmaAtomM;
        const __nv_bfloat16* Xp = acts.X_patch + (int64_t)gp0 * vit::kPatch;
        // fwd into sc.work [npatch_rows, d] fp32 (Kpad=64, Kreal=49).
        const int Kpad = ((vit::kPatch + wgs::kWgmmaAtomK - 1) / wgs::kWgmmaAtomK) * wgs::kWgmmaAtomK; // 64
        const int N = SG_TUNED_TILE_N;
        const int k_steps = Kpad / wgs::kWgmmaAtomK;
        for (int n0 = 0; n0 < vit::kD; n0 += N) {
            const int n_real = (vit::kD - n0) < N ? (vit::kD - n0) : N;
            auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
                return k < vit::kPatch ? Xp[(int64_t)m * vit::kPatch + k] : __float2bfloat16(0.f); };
            auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
                int nn = n0 + n; return (nn < vit::kD && k < vit::kPatch) ? __float2bfloat16(w.patch_w[(int64_t)nn * vit::kPatch + k]) : __float2bfloat16(0.f); };
            auto out  = [&] (int m, int n, float v) { sc.work[(int64_t)m * vit::kD + n0 + n] = v; };
            tc_gemm_block_unpipelined<SG_TUNED_TILE_N, /*MaxAtomsM=*/kAtomsM>(
                0, pm_atoms, n_real, k_steps, srcA, srcB, out, sA, sB);
        }
    }
    __syncthreads();
    // ── Build X_in[0]: CLS rows (pos 0 of each sample) = cls + pos[0]; patch
    //    rows (pos 1+i) = proj(patch_i) + pos[1+i]. bf16. ──
    for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x) {
        const int r = idx / vit::kD, j = idx % vit::kD;
        const int g = g0 + r;
        const int sp = g % vit::kSeq;            // position within the sample
        float v;
        if (sp == 0) {
            v = w.cls[j] + w.pos[j];             // CLS + pos[0]
        } else {
            const int si = r / vit::kSeq;
            const int p = sp - 1;                // patch index 0..15
            const int pr = si * vit::kNPatch + p;
            v = sc.work[(int64_t)pr * vit::kD + j] + w.pos[(int64_t)sp * vit::kD + j] + w.patch_b[j];
        }
        acts.X_in[0][(int64_t)g * vit::kD + j] = __float2bfloat16(v);
    }
    __syncthreads();

    for (int li = 0; li < vit::kLayers; ++li) {
        const VitWeights::Layer& L = w.layer[li];
        const __nv_bfloat16* Xin = acts.X_in[li] + (int64_t)g0 * vit::kD;        // [nrows,d]
        // qkv = Xin @ in_w^T + in_b   (N=3d, K=d). bf16 → scratch.qkv[li].
        vittc_gemm_fwd<SG_TUNED_TILE_N>(Xin, L.in_w, sc.qkv[li], vit::kD, vit::kD, 3 * vit::kD, kAtomsM, sA, sB);
        __syncthreads();
        // add in_b (the fwd GEMM did W only; biases folded in scalar here for qkv).
        for (int idx = threadIdx.x; idx < nrows * 3 * vit::kD; idx += blockDim.x) {
            const int j = idx % (3 * vit::kD);
            float v = __bfloat162float(sc.qkv[li][idx]) + L.in_b[j];
            sc.qkv[li][idx] = __float2bfloat16(v);
        }
        __syncthreads();
        // attention (FULL) → ctx (work fp32) + attn[li] weights.
        vittc_attn_fwd_tile(sc.qkv[li], nrows, sc.work, sc.attn[li]);
        // ctx bf16 → X_ctx[li] (out_proj input + its dW operand).
        for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x) {
            const int r = idx / vit::kD, j = idx % vit::kD;
            acts.X_ctx[li][(int64_t)(g0 + r) * vit::kD + j] = __float2bfloat16(sc.work[(int64_t)r * vit::kD + j]);
        }
        __syncthreads();
        // a = X_ctx @ out_w^T (+ out_b). fp32 → work.
        vittc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_ctx[li] + (int64_t)g0 * vit::kD, L.out_w,
                                            sc.work, vit::kD, vit::kD, sA, sB);
        __syncthreads();
        // r1 = Xin + a + out_b → work (fp32).
        for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x) {
            const int r = idx / vit::kD, j = idx % vit::kD;
            sc.work[(int64_t)r * vit::kD + j] += __bfloat162float(Xin[(int64_t)r * vit::kD + j]) + L.out_b[j];
        }
        __syncthreads();
        // n1(r1) → x1 (fp32) + caches[li]; then bf16 → X_x1[li].
        vittc_ln_fwd_tile(sc.work, L.n1_w, L.n1_b, nrows, sc.x1, sc.n1x[li], sc.n1i[li], red);
        for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x) {
            const int r = idx / vit::kD, j = idx % vit::kD;
            acts.X_x1[li][(int64_t)(g0 + r) * vit::kD + j] = __float2bfloat16(sc.x1[(int64_t)r * vit::kD + j]);
        }
        __syncthreads();
        // ff0 = X_x1 @ ff0_w^T (+ ff0_b). fp32 → work; pre-gelu bf16 → ff0pre;
        // gelu(pre+b) → X_gact[li] (bf16).
        vittc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_x1[li] + (int64_t)g0 * vit::kD, L.ff0_w,
                                            sc.work, vit::kD, vit::kDff, sA, sB);
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * vit::kDff; idx += blockDim.x) {
            const int r = idx / vit::kDff, j = idx % vit::kDff;
            float pre = sc.work[(int64_t)r * vit::kDff + j] + L.ff0_b[j];
            sc.ff0pre[li][(int64_t)r * vit::kDff + j] = __float2bfloat16(pre);
            acts.X_gact[li][(int64_t)(g0 + r) * vit::kDff + j] = __float2bfloat16(vit_gelu(pre));
        }
        __syncthreads();
        // ff2 = X_gact @ ff2_w^T (+ ff2_b). fp32 → work.
        vittc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_gact[li] + (int64_t)g0 * vit::kDff, L.ff2_w,
                                            sc.work, vit::kDff, vit::kD, sA, sB);
        __syncthreads();
        // r2 = x1 + ff2 + ff2_b → work (fp32).
        for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x) {
            const int r = idx / vit::kD, j = idx % vit::kD;
            sc.work[(int64_t)r * vit::kD + j] += sc.x1[(int64_t)r * vit::kD + j] + L.ff2_b[j];
        }
        __syncthreads();
        if (li + 1 < vit::kLayers) {
            // n2(r2) → finalin (fp32 reused) + n2 caches[li]; bf16 → X_in[li+1].
            vittc_ln_fwd_tile(sc.work, L.n2_w, L.n2_b, nrows, sc.finalin, sc.n2x[li], sc.n2i[li], red);
            for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x) {
                const int r = idx / vit::kD, j = idx % vit::kD;
                acts.X_in[li + 1][(int64_t)(g0 + r) * vit::kD + j] = __float2bfloat16(sc.finalin[(int64_t)r * vit::kD + j]);
            }
            __syncthreads();
        } else {
            // last layer: n2(r2) → finalin (fp32, all positions; head reads pos 0).
            vittc_ln_fwd_tile(sc.work, L.n2_w, L.n2_b, nrows, sc.finalin, sc.n2x[li], sc.n2i[li], red);
        }
    }

    // ── Final norm + head + CE, scalar PER-SAMPLE on the CLS position (0) only.
    //    DELTA 3: CLS at pos 0 (row si*kSeq), not the last position. ──
    float nll_acc = 0.0f;
#ifdef SG_VIT_PROFILE
    __syncthreads(); unsigned long long _ha = (threadIdx.x == 0) ? clock64() : 0;
#endif
    for (int si = 0; si < nsamp; ++si) {
        const int rcls = si * vit::kSeq;                  // CLS row (pos 0)
        const float* hcls = sc.finalin + (int64_t)rcls * vit::kD;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) sum += hcls[j];
        float mean = vit_block_sum(sum, red) / (float)vit::kD;
        float vs = 0.0f;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) { float c = hcls[j] - mean; vs += c * c; }
        float var = vit_block_sum(vs, red) / (float)vit::kD;
        float iv = rsqrtf(var + vit::kLnEps);
        if (threadIdx.x == 0) sc.fni[rcls] = iv;
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float xh = (hcls[j] - mean) * iv;
            sc.fnx[(int64_t)rcls * vit::kD + j] = xh;
            float hn = xh * w.norm_w[j] + w.norm_b[j];
            acts.X_hn[(int64_t)si_global(g0, si) * vit::kD + j] = __float2bfloat16(hn);
        }
        __syncthreads();
        // logits[o] = hn · out_w[o] + out_b[o]  (scalar; hn read from X_hn bf16).
        float* lg = sc.logits + (int64_t)si * vit::kVocab;
        const __nv_bfloat16* hnb = acts.X_hn + (int64_t)si_global(g0, si) * vit::kD;
        for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x) {
            const float* Wr = w.out_w + (int64_t)o * vit::kD;
            float acc = w.out_b[o];
            #pragma unroll 4
            for (int k = 0; k < vit::kD; ++k) acc += __bfloat162float(hnb[k]) * Wr[k];
            lg[o] = acc;
        }
        __syncthreads();
        int tgt = tgt_ids[si_global(g0, si)];
        float lmax = -CUDART_INF_F;
        for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x) lmax = fmaxf(lmax, lg[o]);
        lmax = vit_block_max(lmax, red);
        float es = 0.0f;
        for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x) es += __expf(lg[o] - lmax);
        es = vit_block_sum(es, red);
        float logz = lmax + __logf(es);
        if (threadIdx.x == 0) nll_acc += (logz - lg[tgt]);
        __syncthreads();
    }
#ifdef SG_VIT_PROFILE
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(&g_vit_prof_head[0], clock64() - _ha);
#endif
    return nll_acc;
}

// Attention BACKWARD over a tile (FULL — no causal triangle). Reads qkv (bf16),
// attn weights, dctx [nrows,d] fp32; writes dqkv [nrows,3d] fp32 into `dqkv_out`.
// dsc is the per-CTA dscores scratch. Mirror of vit_backward_sample's attention
// (A: dv, B: dscores, C: dq/dk), looped over samples. scale = 1/sqrt(dh).
__device__ __forceinline__ void vittc_attn_bwd_tile(
        const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ attn_w,
        const float* __restrict__ dctx, int nrows,
        float* __restrict__ dqkv_out, float* __restrict__ dsc) {
    const int nsamp = nrows / vit::kSeq;
    const float scale = vit::attn_scale();
    // A: dv[kj] = Σ_{qi} attn[qi,kj] * dctx[qi].  Owner: (sample,kj,head,t). FULL.
    for (int r = threadIdx.x; r < nsamp * vit::kSeq * vit::kHeads * vit::kDhead; r += blockDim.x) {
        const int si  = r / (vit::kSeq * vit::kHeads * vit::kDhead);
        int rem = r % (vit::kSeq * vit::kHeads * vit::kDhead);
        const int kj  = rem / (vit::kHeads * vit::kDhead);
        rem = rem % (vit::kHeads * vit::kDhead);
        const int hh  = rem / vit::kDhead, t = rem % vit::kDhead;
        const int qoff = hh * vit::kDhead;
        const int rbase = si * vit::kSeq;
        const float* aw = attn_w + ((int64_t)(si * vit::kHeads + hh) * vit::kSeq) * vit::kSeq;  // [S,S]
        float acc = 0.0f;
        #pragma unroll
        for (int qi = 0; qi < vit::kSeq; ++qi)   // FULL: all qi
            acc += aw[qi * vit::kSeq + kj] * dctx[(int64_t)(rbase + qi) * vit::kD + qoff + t];
        dqkv_out[(int64_t)(rbase + kj) * 3 * vit::kD + 2 * vit::kD + qoff + t] = acc;   // dv block
    }
    __syncthreads();
    // B: dscores ds[qi,kj] = attn*(datt - Σ_k datt*attn)*scale (FULL, all kj).
    //    datt[kj] = Σ_t dctx[qi,qoff+t]*v[kj,qoff+t]. Owner: (sample,head,qi).
    for (int r = threadIdx.x; r < nsamp * vit::kHeads * vit::kSeq; r += blockDim.x) {
        const int si = r / (vit::kHeads * vit::kSeq);
        int rem = r % (vit::kHeads * vit::kSeq);
        const int hh = rem / vit::kSeq, qi = rem % vit::kSeq;
        const int qoff = hh * vit::kDhead;
        const int rbase = si * vit::kSeq;
        const float* aw = attn_w + ((int64_t)(si * vit::kHeads + hh) * vit::kSeq) * vit::kSeq;
        float datt[vit::kSeq];
        #pragma unroll
        for (int kj = 0; kj < vit::kSeq; ++kj) {
            float acc = 0.0f;
            #pragma unroll
            for (int t = 0; t < vit::kDhead; ++t)
                acc += dctx[(int64_t)(rbase + qi) * vit::kD + qoff + t]
                     * __bfloat162float(qkv[(int64_t)(rbase + kj) * 3 * vit::kD + 2 * vit::kD + qoff + t]);
            datt[kj] = acc;
        }
        float dot = 0.0f;
        #pragma unroll
        for (int kj = 0; kj < vit::kSeq; ++kj) dot += datt[kj] * aw[qi * vit::kSeq + kj];
        float* ds = dsc + ((int64_t)(si * vit::kHeads + hh) * vit::kSeq + qi) * vit::kSeq;
        #pragma unroll
        for (int kj = 0; kj < vit::kSeq; ++kj) {
            float a = aw[qi * vit::kSeq + kj];
            ds[kj] = a * (datt[kj] - dot) * scale;
        }
    }
    __syncthreads();
    // C: dq[qi]=Σ_kj ds[qi,kj]*k[kj]; dk[kj]=Σ_qi ds[qi,kj]*q[qi]. FULL. Owner: (sample,pos,head,t).
    for (int r = threadIdx.x; r < nsamp * vit::kSeq * vit::kHeads * vit::kDhead; r += blockDim.x) {
        const int si = r / (vit::kSeq * vit::kHeads * vit::kDhead);
        int rem = r % (vit::kSeq * vit::kHeads * vit::kDhead);
        const int pos = rem / (vit::kHeads * vit::kDhead);
        rem = rem % (vit::kHeads * vit::kDhead);
        const int hh = rem / vit::kDhead, t = rem % vit::kDhead;
        const int qoff = hh * vit::kDhead;
        const int rbase = si * vit::kSeq;
        const float* ds = dsc + ((int64_t)(si * vit::kHeads + hh) * vit::kSeq) * vit::kSeq;  // [S,S]
        float dq = 0.0f, dk = 0.0f;
        #pragma unroll
        for (int kj = 0; kj < vit::kSeq; ++kj) {
            dq += ds[pos * vit::kSeq + kj] * __bfloat162float(qkv[(int64_t)(rbase + kj) * 3 * vit::kD + vit::kD + qoff + t]);
            dk += ds[kj * vit::kSeq + pos] * __bfloat162float(qkv[(int64_t)(rbase + kj) * 3 * vit::kD + qoff + t]);
        }
        dqkv_out[(int64_t)(rbase + pos) * 3 * vit::kD + qoff + t] = dq;             // dq block
        dqkv_out[(int64_t)(rbase + pos) * 3 * vit::kD + vit::kD + qoff + t] = dk;   // dk block
    }
    __syncthreads();
}

// ════════════════════════════════════════════════════════════════════════
//  BACKWARD over one TOKEN TILE. Assumes vittc_forward_tile ran for THIS tile.
//  Fork B: computes dX via wgmma and WRITES the dY output-adjoints to VitActs
//  (dY_qkv/dY_a/dY_ff0/dY_ff2/dY_logits) + dh0 for P2's dW — it does NOT touch
//  the weight dW here. ACCUMULATES the 10 LN-vector grads into `lnvec`
//  [kNumLnVec × d] (deterministic single-owner-per-j). `B` is the full batch
//  (CE mean scale). DELTA 3: head/final-norm bwd on the CLS row (pos 0).
// ════════════════════════════════════════════════════════════════════════
__device__ void vittc_backward_tile(
        const VitWeights& w, int g0, int nrows, int B, const VitActs& acts,
        const VitTileScratch& sc, const int* __restrict__ tgt_ids,
        float* __restrict__ lnvec, float* __restrict__ work2,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red) {
    const int nsamp = nrows / vit::kSeq;
    float* gn_n1w[vit::kLayers]; float* gn_n1b[vit::kLayers];
    float* gn_n2w[vit::kLayers]; float* gn_n2b[vit::kLayers];
    for (int li = 0; li < vit::kLayers; ++li) {
        gn_n1w[li] = lnvec + (int64_t)(li * 4 + 0) * vit::kD;
        gn_n1b[li] = lnvec + (int64_t)(li * 4 + 1) * vit::kD;
        gn_n2w[li] = lnvec + (int64_t)(li * 4 + 2) * vit::kD;
        gn_n2b[li] = lnvec + (int64_t)(li * 4 + 3) * vit::kD;
    }
    float* gn_normw = lnvec + (int64_t)8 * vit::kD;
    float* gn_normb = lnvec + (int64_t)9 * vit::kD;

    // ── CE bwd (per sample): dlogits = (softmax - onehot)/B → dY_logits (bf16);
    //    head dX → dhn; final-norm bwd → dh on the CLS row (pos 0), zero others. ──
    for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x) sc.dh[idx] = 0.0f;
    __syncthreads();
#ifdef SG_VIT_PROFILE
    unsigned long long _hb = (threadIdx.x == 0) ? clock64() : 0;
#endif
    for (int si = 0; si < nsamp; ++si) {
        const int rcls = si * vit::kSeq;          // CLS row (pos 0)
        const int gs = si_global(g0, si);
        float* lg = sc.logits + (int64_t)si * vit::kVocab;
        int tgt = tgt_ids[gs];
        float lmax = -CUDART_INF_F;
        for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x) lmax = fmaxf(lmax, lg[o]);
        lmax = vit_block_max(lmax, red);
        float es = 0.0f;
        for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x) es += __expf(lg[o] - lmax);
        es = vit_block_sum(es, red);
        float inv_es = 1.0f / es;
        for (int o = threadIdx.x; o < vit::kVocab; o += blockDim.x) {
            float smo = __expf(lg[o] - lmax) * inv_es;
            float dl = (smo - ((o == tgt) ? 1.0f : 0.0f)) / (float)B;
            lg[o] = dl;
            acts.dY_logits[(int64_t)gs * vit::kVocab + o] = __float2bfloat16(dl);
        }
        __syncthreads();
        // dhn[j] = Σ_o dlogits[o] * out_w[o,j] → stash into work row rcls.
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float dhn = 0.0f;
            for (int o = 0; o < vit::kVocab; ++o)
                dhn += lg[o] * w.out_w[(int64_t)o * vit::kD + j];
            sc.work[(int64_t)rcls * vit::kD + j] = dhn;
        }
        __syncthreads();
        // norm γ/β: dnorm_w[j] += dhn*xhat; dnorm_b[j] += dhn. (Only CLS row.)
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float dhn = sc.work[(int64_t)rcls * vit::kD + j];
            float xh = sc.fnx[(int64_t)rcls * vit::kD + j];
            gn_normw[j] += dhn * xh; gn_normb[j] += dhn;
        }
        __syncthreads();
        // LN dx (single CLS row): dxhat=dhn*norm_w; reduce; dh[rcls] = inv*(...).
        {
            const float* dyr = sc.work + (int64_t)rcls * vit::kD;
            const float* xhr = sc.fnx + (int64_t)rcls * vit::kD;
            float sda = 0.0f, sdax = 0.0f;
            for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
                float dxhat = dyr[j] * w.norm_w[j]; sda += dxhat; sdax += dxhat * xhr[j];
            }
            sda = vit_block_sum(sda, red); sdax = vit_block_sum(sdax, red);
            float iv = sc.fni[rcls];
            for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
                float dxhat = dyr[j] * w.norm_w[j];
                sc.dh[(int64_t)rcls * vit::kD + j] = iv * (dxhat - (sda + xhr[j] * sdax) / (float)vit::kD);
            }
            __syncthreads();
        }
    }
#ifdef SG_VIT_PROFILE
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(&g_vit_prof_head[1], clock64() - _hb);
#endif
    // scratch.dh now = grad wrt last-layer output [nrows,d] (only CLS positions nonzero).

    for (int li = vit::kLayers - 1; li >= 0; --li) {
        const VitWeights::Layer& L = w.layer[li];
        // n2 bwd: dh → dr2 (work fp32), accumulate n2 γ/β. xhat=n2x[li], inv=n2i[li].
        vittc_ln_bwd_tile(sc.dh, sc.n2x[li], sc.n2i[li], L.n2_w, nrows, sc.work, gn_n2w[li], gn_n2b[li], red);
        // r2 = x1 + ff2 → dx1 = dr2 (residual), dff2 = dr2. dff2 → dY_ff2 (bf16).
        for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x) {
            work2[idx] = sc.work[idx];   // dx1 := dr2 (residual part)
            acts.dY_ff2[li][(int64_t)g0 * vit::kD + idx] = __float2bfloat16(sc.work[idx]);  // dff2
        }
        __syncthreads();
        // ff2 dX: dgact = dff2 @ ff2_w  (N=dff, K=d). fp32 → sc.work (dr2 no longer needed).
        vittc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff2[li] + (int64_t)g0 * vit::kD, L.ff2_w,
                                           sc.work, vit::kDff, vit::kD, sA, sB);  // dgact [nrows,dff]
        __syncthreads();
        // dff0 = dgact * gelu'(ff0pre) → dY_ff0 (bf16) AND keep fp32 in sc.work for dX.
        for (int idx = threadIdx.x; idx < nrows * vit::kDff; idx += blockDim.x) {
            float dff0 = sc.work[idx] * vit_gelu_grad(__bfloat162float(sc.ff0pre[li][idx]));
            sc.work[idx] = dff0;
            acts.dY_ff0[li][(int64_t)g0 * vit::kDff + idx] = __float2bfloat16(dff0);
        }
        __syncthreads();
        // ff0 dX: dx1 += dff0 @ ff0_w  (output width Kin=d, contract Nout=dff). fp32 → sc.x1.
        vittc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff0[li] + (int64_t)g0 * vit::kDff, L.ff0_w,
                                           sc.x1, /*Kin=*/vit::kD, /*Nout=*/vit::kDff, sA, sB);  // dx1_ffn [nrows,d]
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x)
            work2[idx] += sc.x1[idx];   // dx1 = residual + FFN path
        __syncthreads();
        // n1 bwd: dx1 (work2) → dr1 (work), accumulate n1 γ/β.
        vittc_ln_bwd_tile(work2, sc.n1x[li], sc.n1i[li], L.n1_w, nrows, sc.work, gn_n1w[li], gn_n1b[li], red);
        // r1 = x_in + a → da = dr1 (out_proj output adjoint), dx_in = dr1 (residual).
        // SAVE residual dr1 into sc.dh (free); da → dY_a (bf16).
        for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x) {
            sc.dh[idx] = sc.work[idx];   // residual dx_in := dr1 (saved across attn bwd)
            acts.dY_a[li][(int64_t)g0 * vit::kD + idx] = __float2bfloat16(sc.work[idx]);  // da
        }
        __syncthreads();
        // out_proj dX: dctx = da @ out_w  (N=d, K=d). fp32 → sc.work (dctx [nrows,d]).
        vittc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_a[li] + (int64_t)g0 * vit::kD, L.out_w,
                                           sc.work, vit::kD, vit::kD, sA, sB);  // dctx
        __syncthreads();
        // attention bwd (FULL): (qkv[li], attn[li], dctx=work) → dqkv [nrows,3d]
        //   fp32 into work2 (3d=384 ≤ dff=512, fits). Then → dY_qkv (bf16).
        vittc_attn_bwd_tile(sc.qkv[li], sc.attn[li], sc.work, nrows, work2, sc.dsc);
        for (int idx = threadIdx.x; idx < nrows * 3 * vit::kD; idx += blockDim.x)
            acts.dY_qkv[li][(int64_t)g0 * 3 * vit::kD + idx] = __float2bfloat16(work2[idx]);
        __syncthreads();
        // in_proj dX: dx_in_attn = dqkv @ in_w (output width Kin=d, contract Nout=3d).
        //   fp32 → sc.work; ADD residual (in sc.dh) → new running adjoint dh.
        vittc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_qkv[li] + (int64_t)g0 * 3 * vit::kD, L.in_w,
                                           sc.work, /*Kin=*/vit::kD, /*Nout=*/3 * vit::kD, sA, sB);  // dx_in_attn
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x)
            sc.dh[idx] += sc.work[idx];   // dx_in = residual (in dh) + attn path
        __syncthreads();
    }

    // ── embedding bwd: dh = grad wrt h0 [nrows,d]. Write dh0 (bf16); the
    //    cls/patch/pos owners (P2) read dh0 by global token row. ──
    for (int idx = threadIdx.x; idx < nrows * vit::kD; idx += blockDim.x)
        acts.dh0[(int64_t)g0 * vit::kD + idx] = __float2bfloat16(sc.dh[idx]);
    __syncthreads();
}

// ════════════════════════════════════════════════════════════════════════
//  P2 — OUTPUT-STATIONARY dW. Each of the 10 weight matrices (patch_proj +
//  in/out/ff0/ff2 ×L + head) dW = dYᵀ @ X is split into 64×N output tiles;
//  tile_id % nCTA owns each. The owner CTA contracts the FULL token dim itself
//  (ascending-t, no float atomics, no partials) via the validated dW
//  orientation. Writes the tile into `grad`.
//
//  patch_proj dW = dh0_patch^T @ X_patch (K = Tp = nsamp·16; the patch rows of
//  dh0, gathered against the patch token rows). The 9 transformer dW use K=T (B
//  for the head). To unify, each spec carries an `X_is_patch` flag: when set,
//  the dW contracts over the Tp PATCH rows, reading the dY adjoint from dh0's
//  patch rows (via a global→patch-row map) and X from X_patch.
// ════════════════════════════════════════════════════════════════════════
struct VitDwSpec {
    const __nv_bfloat16* dY;   // [K, Nout]  (for patch_proj: nullptr — uses dh0)
    const __nv_bfloat16* X;    // [K, Kin]
    int Nout; int Kin; int K;  // Kin = REAL in-dim; K = contraction length
    int Kpad;                  // Kin padded to a multiple of 16 (=64 for patch_proj's Kin=49)... see note
    int grad_off;              // element offset of this weight in `grad`
    int bias_off;              // element offset of the bias in `grad`
    int kind;                  // 0=transformer (dY/X both token rows), 1=patch_proj
};

// Build the 10 specs (called by all CTAs; cheap). T = B*kSeq; Tp = B*kNPatch.
__device__ __forceinline__ void vittc_build_dw_specs(
        const VitActs& acts, int B, int T, int Tp, VitDwSpec spec[10]) {
    // vit_layout weight tensor indices (and bias idx). Per-layer block base = 4 + li*12:
    //   in_w = base+0 (in_b base+1), out_w base+2 (out_b base+3),
    //   ff0_w base+8 (ff0_b base+9), ff2_w base+10 (ff2_b base+11).
    //   patch_proj.weight=1 (bias=2); head out.weight=30 (out.bias=31).
    // spec[0] = patch_proj; spec[1..8] = transformer (li,kind); spec[9] = head.
    {
        VitDwSpec& sp = spec[0];
        sp.dY = nullptr; sp.X = acts.X_patch; sp.Nout = vit::kD; sp.Kin = vit::kPatch;
        sp.K = Tp; sp.Kpad = vit::kPatch;  // K (contraction over patch rows) IS Tp; Kin padding handled in run_tile
        sp.grad_off = kVitOffsets[1]; sp.bias_off = kVitOffsets[2]; sp.kind = 1;
    }
    for (int s = 0; s < 8; ++s) {
        const int li = s / 4, kk = s % 4;
        const int base = 4 + li * 12;
        VitDwSpec& sp = spec[1 + s];
        sp.K = T; sp.kind = 0; sp.Kpad = 0;
        if (kk == 0)      { sp.dY = acts.dY_qkv[li]; sp.X = acts.X_in[li];  sp.Nout = 3 * vit::kD; sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 0]; sp.bias_off = kVitOffsets[base + 1]; }
        else if (kk == 1) { sp.dY = acts.dY_a[li];   sp.X = acts.X_ctx[li]; sp.Nout = vit::kD;     sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 2]; sp.bias_off = kVitOffsets[base + 3]; }
        else if (kk == 2) { sp.dY = acts.dY_ff0[li]; sp.X = acts.X_x1[li];  sp.Nout = vit::kDff;   sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 8]; sp.bias_off = kVitOffsets[base + 9]; }
        else              { sp.dY = acts.dY_ff2[li]; sp.X = acts.X_gact[li];sp.Nout = vit::kD;     sp.Kin = vit::kDff; sp.grad_off = kVitOffsets[base + 10]; sp.bias_off = kVitOffsets[base + 11]; }
    }
    VitDwSpec& hd = spec[9];
    hd.dY = acts.dY_logits; hd.X = acts.X_hn; hd.Nout = vit::kVocab; hd.Kin = vit::kD; hd.K = B; hd.kind = 0; hd.Kpad = 0;
    hd.grad_off = kVitOffsets[30]; hd.bias_off = kVitOffsets[31];
}

// Total number of 64×N dW output tiles across the 10 weights (for the tile loop).
// The N-tiling is over Kin (the in-dim), padded so the GEMM N covers it.
template <int N>
__device__ __forceinline__ int vittc_dw_total_tiles(const VitDwSpec spec[10]) {
    int n = 0;
    for (int s = 0; s < 10; ++s)
        n += ((spec[s].Nout + 63) / 64) * ((spec[s].Kin + N - 1) / N);
    return n;
}

// Run ONE global dW tile `gt` (if it belongs to this CTA). MaxAtomsM=1 (one 64×N
// tile). For patch_proj (kind==1) the dY adjoint is dh0's PATCH rows: dh0 is
// [T,d] token-row-major; its patch rows for sample si, patch p are at token row
// si*kSeq + (1+p). The dW contracts over the Tp patch rows in ascending patch
// order (k=0..Tp-1 maps to si=k/kNPatch, p=k%kNPatch → token row si*kSeq+1+p).
template <int N>
__device__ __forceinline__ void vittc_dw_run_tile(
        const VitDwSpec spec[10], int gt, const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ grad, __nv_bfloat16* sA, __nv_bfloat16* sB) {
    int acc = 0, s = 0, m_atom = 0, n_tile = 0;
    for (s = 0; s < 10; ++s) {
        const int ma = (spec[s].Nout + 63) / 64;
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ma * nt) { int loc = gt - acc; m_atom = loc / nt; n_tile = loc % nt; break; }
        acc += ma * nt;
    }
    const VitDwSpec& sp = spec[s];
    const int mbase = m_atom * 64;
    const int n0 = n_tile * N;
    const int n_real = (sp.Kin - n0) < N ? (sp.Kin - n0) : N;
    const int Nout = sp.Nout, Kin = sp.Kin;
    const __nv_bfloat16* dY = sp.dY; const __nv_bfloat16* X = sp.X;
    if (sp.kind == 1) {
        // patch_proj: K = Tp patch rows (ascending), padded UP to a multiple of 16.
        const int Tp = sp.K;
        const int Kpad = ((Tp + wgs::kWgmmaAtomK - 1) / wgs::kWgmmaAtomK) * wgs::kWgmmaAtomK;
        const int k_steps = Kpad / wgs::kWgmmaAtomK;
        // A[m=out, k=patchrow] = dh0[token row of patchrow, out]  (transposed read).
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
            if (m >= Nout || k >= Tp) return __float2bfloat16(0.f);
            const int si = k / vit::kNPatch, p = k % vit::kNPatch;
            const int trow = si * vit::kSeq + (1 + p);
            return dh0[(int64_t)trow * vit::kD + m]; };
        // B[n=in, k=patchrow] = X_patch[patchrow, in].
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; if (nn >= Kin || k >= Tp) return __float2bfloat16(0.f);
            return X[(int64_t)k * Kin + nn]; };
        auto out  = [&] (int m, int n, float v) {
            if (m < Nout) grad[sp.grad_off + (int64_t)m * Kin + n0 + n] = v; };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/1>(
            mbase, /*m_atoms=*/1, n_real, k_steps, srcA, srcB, out, sA, sB);
        return;
    }
    const int k_steps = sp.K / wgs::kWgmmaAtomK;     // K = T or B (must be /16; padded by caller)
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
//  SPLIT-K dW (multi-CTA tiling — the validated decoder/mamba win, ported to
//  ViT). The 10 dW yield only ~52 output tiles → ~60% of 132 SMs idle in P2.
//  Split-K turns each output tile into G work items (one per K-chunk): each CTA
//  computes a PARTIAL over its K-chunk into a per-(tile,chunk) scratch slot; a
//  grid barrier; then a DETERMINISTIC ascending-chunk reduce sums the G partials
//  into grad — no float atomics, fixed order, so parity + A/A/A bit-determinism
//  hold (each partial is the SAME ascending-k fp32 wgmma accumulate; Σ_chunk ==
//  full-K sum reassociated into G fp32 blocks). G==1 routes to the single-CTA
//  vittc_dw_run_tile above (no scratch). Slot (gt,kc) at
//  dw_part[(gt*G+kc) * (64*kVitMaxTileN) + row*kVitMaxTileN + col].
//
//  ViT delta vs the decoder split-K: the patch_proj tile (kind==1) contracts over
//  K=Tp patch rows (not a multiple of 16); the SAME floor-balanced atom partition
//  over KS=ceil(Tp/16) atoms works because the patch srcA/srcB already zero-guard
//  k>=Tp (padded atoms contribute 0). kind==0 tiles contract over K=T or K=B
//  (exact multiples of 16), exactly like the decoder.
// ════════════════════════════════════════════════════════════════════════
constexpr int kVitMaxTileN = SG_TUNED_TILE_N;                       // widest dW N-tile
constexpr int kVitDwTileFloats = wgs::kWgmmaAtomM * kVitMaxTileN;   // 64*N per (gt,kc) slot

// COMPILE-TIME max #dW output tiles (the 10 dW have fixed Nout/Kin; ViT dims are
// compile-time → constant). Mirrors vittc_dw_total_tiles at N=kVitMaxTileN.
//   patch_proj(d×patch) + per-layer[qkv(3d×d),attn_out(d×d),ff0(dff×d),ff2(d×dff)]
//   ×kLayers + head(V×d).
constexpr int kVitDwTilesPerLayer =
      ((3*vit::kD + 63)/64) * ((vit::kD   + kVitMaxTileN - 1)/kVitMaxTileN)   // qkv
    + ((vit::kD   + 63)/64) * ((vit::kD   + kVitMaxTileN - 1)/kVitMaxTileN)   // attn_out
    + ((vit::kDff + 63)/64) * ((vit::kD   + kVitMaxTileN - 1)/kVitMaxTileN)   // ff0
    + ((vit::kD   + 63)/64) * ((vit::kDff + kVitMaxTileN - 1)/kVitMaxTileN);  // ff2
constexpr int kVitDwPatchTiles =
      ((vit::kD + 63)/64) * ((vit::kPatch + kVitMaxTileN - 1)/kVitMaxTileN);  // patch_proj
constexpr int kVitDwHeadTiles =
      ((vit::kVocab + 63)/64) * ((vit::kD + kVitMaxTileN - 1)/kVitMaxTileN);  // head
constexpr int kVitDwMaxTiles =
      kVitDwPatchTiles + vit::kLayers * kVitDwTilesPerLayer + kVitDwHeadTiles;

// Split-K dW partial-scratch float count (host carves it from the workspace tail).
// 0 when G==1 → no extra scratch (the single-CTA path is byte-identical).
__host__ __device__ __forceinline__ int64_t vit_dw_part_floats(int G) {
    return (G > 1) ? (int64_t)kVitDwMaxTiles * G * kVitDwTileFloats : 0;
}

// Decode global dW tile index gt → (spec index s, m_atom, n_tile). Single-source
// (the SAME walk vittc_dw_run_tile + vittc_dw_total_tiles use).
template <int N>
__device__ __forceinline__ void vittc_dw_decode(
        const VitDwSpec spec[10], int gt, int& s, int& m_atom, int& n_tile) {
    int acc = 0;
    for (s = 0; s < 10; ++s) {
        const int ma = (spec[s].Nout + 63) / 64;
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ma * nt) { int loc = gt - acc; m_atom = loc / nt; n_tile = loc % nt; return; }
        acc += ma * nt;
    }
    s = 9; m_atom = 0; n_tile = 0;   // unreachable (gt < n_dw); keep defined
}

// PARTIAL dW for global tile gt over K-chunk kc of G → dw_part. FLOOR-BALANCED
// atom partition: chunk kc = [k0,k1) atoms with k0=floor(kc·KS/G),
// k1=floor((kc+1)·KS/G) — near-equal, summing to KS EXACTLY for ANY KS≥G (no
// `G | KS` requirement, so it works at the production batch where B/16 or
// ceil(Tp/16) need not divide G). A CEIL split would leave a trailing EMPTY chunk
// whose slot stays unwritten → the reduce sums garbage; floor never empties a
// chunk for KS≥G, and a KS<G empty chunk is explicitly zeroed. Fresh ScaleD=0 per
// chunk → true partial; writes the full 64×N tile (LOCAL rows) to the slot.
template <int N>
__device__ __forceinline__ void vittc_dw_run_tile_splitk(
        const VitDwSpec spec[10], int gt, int kc, int G,
        const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ dw_part, __nv_bfloat16* sA, __nv_bfloat16* sB) {
    int s, m_atom, n_tile;
    vittc_dw_decode<N>(spec, gt, s, m_atom, n_tile);
    const VitDwSpec& sp = spec[s];
    const int mbase = m_atom * 64;
    const int n0 = n_tile * N;
    const int Nout = sp.Nout, Kin = sp.Kin;
    // KS = total k-atoms. kind==1 (patch_proj): K=Tp patch rows padded UP to /16.
    // kind==0: K=T or K=B (already multiples of 16).
    const int KS = (sp.kind == 1)
        ? (((sp.K + wgs::kWgmmaAtomK - 1) / wgs::kWgmmaAtomK))   // ceil(Tp/16)
        : (sp.K / wgs::kWgmmaAtomK);
    const int k0 = (int)(((int64_t)kc       * KS) / G);          // floor-balanced
    const int k1 = (int)(((int64_t)(kc + 1) * KS) / G);
    const int kc_steps = k1 - k0;
    float* slot = dw_part + ((int64_t)gt * G + kc) * kVitDwTileFloats;
    // Empty-chunk guard (KS<G): a k_steps=0 GEMM would emit the uninitialized
    // accumulator → zero the slot + return (the reduce sums all G slots).
    if (kc_steps <= 0) {
        for (int i = threadIdx.x; i < 64 * N; i += blockDim.x) slot[i] = 0.0f;
        __syncthreads();
        return;
    }
    auto out = [&] (int m, int n, float v) {
        const int lr = m - mbase;
        if (lr >= 0 && lr < 64 && n < N) slot[(int64_t)lr * N + n] = v; };
    if (sp.kind == 1) {
        // patch_proj: A[m=out,k=patchrow]=dh0[token row of patchrow, out] (transposed);
        // B[n=in,k=patchrow]=X_patch[patchrow,in]. The k index here is the LOCAL chunk
        // atom-step; the global patch row is (k0*16 + k). Pad rows (>=Tp) → 0.
        const int Tp = sp.K;
        const __nv_bfloat16* X = sp.X;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
            const int gk = k0 * wgs::kWgmmaAtomK + k;
            if (m >= Nout || gk >= Tp) return __float2bfloat16(0.f);
            const int si = gk / vit::kNPatch, p = gk % vit::kNPatch;
            const int trow = si * vit::kSeq + (1 + p);
            return dh0[(int64_t)trow * vit::kD + m]; };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            const int gk = k0 * wgs::kWgmmaAtomK + k;
            int nn = n0 + n; if (nn >= Kin || gk >= Tp) return __float2bfloat16(0.f);
            return X[(int64_t)gk * Kin + nn]; };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/1>(
            mbase, /*m_atoms=*/1, /*n_real=*/N, kc_steps, srcA, srcB, out, sA, sB);
        return;
    }
    // kind==0: A[m=out,k=t]=dY[t,out]; B[n=in,k=t]=X[t,in] (both transposed reads).
    const __nv_bfloat16* dY = sp.dY; const __nv_bfloat16* X = sp.X;
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
        return m < Nout ? dY[(int64_t)(k0 * wgs::kWgmmaAtomK + k) * Nout + m] : __float2bfloat16(0.f); };
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
        int nn = n0 + n; return nn < Kin ? X[(int64_t)(k0 * wgs::kWgmmaAtomK + k) * Kin + nn] : __float2bfloat16(0.f); };
    tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/1>(
        mbase, /*m_atoms=*/1, /*n_real=*/N, kc_steps, srcA, srcB, out, sA, sB);
}

// Deterministic reduce: output tile gt (% nCTA) sums its G chunk-partials
// ascending-kc → grad. SAME (gt → geometry) decode as the partial.
template <int N>
__device__ __forceinline__ void vittc_dw_reduce_splitk(
        const VitDwSpec spec[10], int n_dw, int G, const float* __restrict__ dw_part,
        float* __restrict__ grad, int cta, int nCTA) {
    for (int gt = cta; gt < n_dw; gt += nCTA) {
        int s, m_atom, n_tile;
        vittc_dw_decode<N>(spec, gt, s, m_atom, n_tile);
        const VitDwSpec& sp = spec[s];
        const int mbase = m_atom * 64;
        const int n0 = n_tile * N;
        const int n_real = (sp.Kin - n0) < N ? (sp.Kin - n0) : N;
        const int Nrow = (sp.Nout - mbase) < 64 ? (sp.Nout - mbase) : 64;
        const int64_t base = (int64_t)gt * G * kVitDwTileFloats;
        for (int idx = threadIdx.x; idx < Nrow * n_real; idx += blockDim.x) {
            const int row = idx / n_real, col = idx % n_real;
            float accv = 0.0f;
            for (int kc = 0; kc < G; ++kc)
                accv += dw_part[base + (int64_t)kc * kVitDwTileFloats + (int64_t)row * N + col];
            grad[sp.grad_off + (int64_t)(mbase + row) * sp.Kin + n0 + col] = accv;
        }
    }
}

// Biases db. For the 8 transformer linears + head: db = Σ_K dY (per output row).
// For patch_proj.bias: db[o] = Σ_{patch rows} dh0[token row, o]. Single owner per
// output element → no atomics.
// PERF FIX (structural, parity-preserved — gated by the K-scaled grad tol, which is
// designed for fp32 accumulation-order differences at large K):
//   (1) The ORIGINAL ran the FULL Σ_K (K up to T=278528) for EVERY bias element on
//       EVERY CTA (no cta/nCTA guard) — 132× redundant work, the dominant cost of P2
//       grad-assembly (measured 59.5% of the step). Columns are now PARTITIONED across
//       CTAs (each global bias column owned by one CTA), removing the 132× redundancy.
//   (2) The K-reduction per column was SERIAL on ONE thread (≈278528 iters). It is now
//       WARP-PARALLEL: a WARP (32 lanes) co-reduces one column — each lane sums a
//       strided 1/32 slice of K (ascending within the lane), then a shuffle tree sums
//       the 32 lane-partials. 32× parallelism per column. The lane reads dY[k*Nout+o]
//       at fixed o with k = lane, lane+32, … (stride 32·Nout); writes by lane 0 only.
__device__ __forceinline__ void vittc_dw_biases(
        const VitDwSpec spec[10], const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ grad, int cta, int nCTA) {
    const int warp = threadIdx.x >> 5;            // 0..(blockDim/32 - 1)
    const int lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    int gcol_base = 0;
    for (int s = 0; s < 10; ++s) {
        const VitDwSpec& sp = spec[s];
        const int Nout = sp.Nout;
        const bool patch = (sp.kind == 1);
        const int K = sp.K;
        // Each warp owns columns o = (its rank among this CTA's warps), strided. A
        // global column gcol is owned by CTA (gcol % nCTA); within the CTA the warps
        // round-robin the owned columns.
        for (int o = 0; o < Nout; ++o) {
            const int gcol = gcol_base + o;
            if ((gcol % nCTA) != cta) continue;       // not this CTA's column
            // round-robin owned columns across warps: count owned-so-far for THIS s.
            // Cheap: derive the warp owner from the o-th owned index. We instead let
            // every warp test (owned_index % nwarps == warp); compute owned_index by
            // a strided scan-free rule: owned columns appear every nCTA in gcol, so
            // the k-th owned column of this CTA has o such that (gcol_base+o)%nCTA==cta.
            // Assign by (o / 1) round-robin is fine since the inner reduce dominates:
            // use (o-th column's position) — approximate with o itself for balance.
            if (((o) % nwarps) != warp) continue;     // warp owns this column
            float part = 0.0f;
            if (patch) {
                for (int k = lane; k < K; k += 32) {
                    const int si = k / vit::kNPatch, p = k % vit::kNPatch;
                    const int trow = si * vit::kSeq + (1 + p);
                    part += __bfloat162float(dh0[(int64_t)trow * vit::kD + o]);
                }
            } else {
                const __nv_bfloat16* dY = sp.dY;
                for (int k = lane; k < K; k += 32)
                    part += __bfloat162float(dY[(int64_t)k * Nout + o]);
            }
            // Warp-reduce the 32 lane-partials (shuffle tree).
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                part += __shfl_down_sync(0xffffffffu, part, off);
            if (lane == 0) grad[sp.bias_off + o] = part;
        }
        gcol_base += Nout;
    }
}

// ════════════════════════════════════════════════════════════════════════
//  P2 — CLS + POS owner-scan. dh0 [T,d] (bf16) holds grad wrt h0 (post-cat,
//    pre-pos). cls_token grad = Σ_samp dh0[CLS row (pos 0)]; pos.weight[p] grad
//    = Σ_t dh0[t] over t with (t%kSeq)==p (ALL 17 positions). Fixed t-order +
//    single owner per element → deterministic, atomic-free. The owner of a
//    feature j is round-robin (j % ... is implicit via thread stride; only ONE
//    CTA — cta 0 — writes these tiny ([d] and [17×d]) tensors to avoid a
//    cross-CTA race, since they are not tiled by gt). Mirrors the decoder
//    embed owner-scan, ViT variant.
// ════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void vittc_clspos_owner_scan(
        const VitActs& acts, int T, float* __restrict__ grad, int cta, int nCTA) {
    const int cls_off = kVitOffsets[0];   // cls_token [1,1,d] → [d]
    const int pos_off = kVitOffsets[3];   // pos.weight [kSeq, d]
    const int nsamp = T / vit::kSeq;
    // cls_token: owned by cta 0 (tiny [d]); each thread owns feature j.
    if (cta == 0) {
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float accv = 0.0f;
            for (int si = 0; si < nsamp; ++si)
                accv += __bfloat162float(acts.dh0[(int64_t)(si * vit::kSeq) * vit::kD + j]);
            grad[cls_off + j] = accv;
        }
    }
    // pos rows: round-robin by position p across CTAs (p in [0,kSeq)).
    // PERF FIX: the ORIGINAL looped ALL T tokens with `if (t%kSeq)==p`, doing 17×
    // (kSeq) the necessary reads (keeping only 1/17). Token rows with t%kSeq==p are
    // exactly t = p, p+kSeq, p+2·kSeq, … so stride DIRECTLY by kSeq (the same
    // ascending-t order → bit-identical fp32 sum). 17× fewer reads.
    for (int p = cta; p < vit::kSeq; p += nCTA) {
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float accv = 0.0f;
            for (int t = p; t < T; t += vit::kSeq)
                accv += __bfloat162float(acts.dh0[(int64_t)t * vit::kD + j]);
            grad[pos_off + (int64_t)p * vit::kD + j] = accv;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
//  P2 — LN-vector grad reduce. The 10 γ/β grads were accumulated tile-locally
//    into each CTA's lnvec partials [kNumLnVec × d]; sum across CTAs in
//    ASCENDING CTA index (deterministic) into the 10 vit_layout slots of grad.
//    `lnvec_base` is the start of the [nCTA × kLnVecElems] partial region.
// ════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void vittc_lnvec_reduce(
        const float* __restrict__ lnvec_base, float* __restrict__ grad,
        int nCTA, int cta) {
    for (int v = cta; v < kNumLnVec; v += nCTA) {
        const int goff = kLnVecTensorIdx[v];
        const int64_t gbase = kVitOffsets[goff];
        for (int j = threadIdx.x; j < vit::kD; j += blockDim.x) {
            float accv = 0.0f;
            for (int c = 0; c < nCTA; ++c)
                accv += lnvec_base[(int64_t)c * kLnVecElems + (int64_t)v * vit::kD + j];
            grad[gbase + j] = accv;
        }
    }
}

}  // namespace vittc

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGE_VIT_TC_CUH_
