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

// ── M-atom INTERLEAVE width cap (task #24 — the decoder H1+H3 hill-climb win,
//    ported to the ViT twin). The GEMM microkernel processes stacked m64 atoms in
//    groups of min(MaxAtomsM, this); within a group the per-k wgmmas (one per atom)
//    issue back-to-back into INDEPENDENT fp32 fragments sharing ONE staged B-tile →
//    the tensor pipe overlaps the MMAs AND the (HBM-bound) B-operand tile is staged
//    once per group instead of per atom. Capped (default 2) so the accumulator-
//    register + A-smem cost stays bounded regardless of m_atoms (a ViT token tile is
//    kAtomsM=17 atoms, and a dW tile up to 8; an N-wide interleave needs N·(width/2)
//    fp32 accumulator regs). 1 = no interleave (the old serial path, bit-for-bit).
//    Per-atom accumulation stays ASCENDING-k → numerics bit-identical + A/A/A
//    determinism unchanged. (ViT's TILE_M=1088 / LCM-1088 constraint is UNTOUCHED —
//    this only regroups the m64-atom issue order, not the tile size.)
#ifndef SG_TUNED_VIT_GEMM_INTERLEAVE
#define SG_TUNED_VIT_GEMM_INTERLEAVE 2
#endif

// ── P1 token-tile SUB-TILING (the B1 grid-barrier load-imbalance lever — #1 ViT
//    perf item, 51.2% of the step at d2048/B1024). The token tile is locked at
//    kTileM = LCM(kSeq=17, 64) = 1088 rows because a tile boundary must be BOTH a
//    wgmma-m64-atom boundary AND a whole-sample (kSeq) boundary. At B=1024 that
//    yields T=17408 → only n_tiles=16 P1 tiles, so 116 of 132 CTAs do ZERO P1
//    work and spin at B1 the whole fwd+bwd. It is tile-count-vs-SM quantization,
//    NOT intra-tile skew.
//
//    This knob decouples the P1 SCHEDULING granularity from the 1088 GEMM lock:
//    P1 grid-strides over sub-tiles of S WHOLE samples (= kSeq·S rows) instead of
//    whole 1088-row tiles, so n_tiles grows ~(64/S)× and the per-sample scalar
//    surface (LN / head-CE / attention — the dominant cost; dW is only ~3%) is
//    spread across up to ~64/S× more SMs. S MUST stay a whole-sample count
//    (never split a 17-row sample across CTAs — that corrupts attention) and the
//    sub-tile row count kSeq·S is in general NOT a multiple of 64, so the small
//    bf16 GEMMs pad the M-dim up to ceil(rows/64)·64 atoms (the m_real row-guard
//    below makes the pad rows inert + race-free).
//
//    DEFAULT 64 == kSamplesPerTile for TILE_M=1088 ⇒ sub-tile rows == kTileM ⇒
//    the P1 loop, n_tiles, m_atoms (=17, a multiple of 64 rows), and the GEMM
//    engine are ALL identical to the pre-knob kernel — the OFF build is
//    BYTE-IDENTICAL (the m_real guard + runtime-m_atoms code is #if'd out
//    entirely at S==64). Values {16,8,4} ENGAGE sub-tiling (S=16→64 active CTAs,
//    S=8→128, S=4→all 132); sweep {16,8,4} (U-shaped curve — avoid S=1).
//
//    PARITY (knob ON, S<64): dW / biases / cls / pos grads are BYTE-IDENTICAL —
//    P2 reads activations by GLOBAL token row and contracts ascending-k
//    independent of P1 tiling, and each acts row is a pure per-row (sample-local,
//    in-tile) function preserved by keeping the whole-sample invariant. The 10
//    LN-affine (γ/β) grads + the loss are NOT byte-identical: they are VALID fp32
//    REASSOCIATIONS — per-CTA partials are built tile-locally (gw[j]+=dgw / the
//    per-tile nll_acc), so changing S regroups which global rows land in which
//    CTA, hence the fp32 grouping of those sums shifts in the low bits. That is
//    absorbed by the gate's 1e-4 fp32-reorder tol (the decoder already ships
//    TILE_M 64/256 through the same gate). A/A/A bit-identity HOLDS because S is
//    fixed across the 3 runs (a fixed deterministic partition of (cta,nCTA,
//    n_tiles,T)). No atomics, no ragged reduce. ⇒ OFF = byte-identical; ON = a
//    NEEDS-PARITY swept dim, the GPU fp64 + A/A/A gate is the arbiter.
#ifndef SG_TUNED_VIT_P1_SUBTILE_S
// BAKED default 64->8 (2026-06-17): gate verdict from the B1 flip-gate sweep.
// S=8 is parity-clean (11/11 ViT cells x 3 seeds {42,7,123}; A/A/A held — S is a
// fixed deterministic partition) AND fastest: 4.02x vs S=64 OFF at d=2048/B=1024
// (B1_barrier 51.2%->41.9%; the U-curve inflects at 8 — S=4 ticks back +1.2%).
// The #1 perf lever (was 51% of the ViT step). Final confirm-build on return.
#define SG_TUNED_VIT_P1_SUBTILE_S 8
#endif

namespace vittc {

constexpr int kVitDwSplitK = SG_TUNED_VIT_DW_SPLITK;
static_assert(kVitDwSplitK >= 1, "SG_TUNED_VIT_DW_SPLITK must be >= 1");

constexpr int kVitMaxIL = SG_TUNED_VIT_GEMM_INTERLEAVE;
static_assert(kVitMaxIL >= 1 && kVitMaxIL <= 4,
              "SG_TUNED_VIT_GEMM_INTERLEAVE must be 1 (serial) .. 4");

// ── Muon 2D-weight table (the matrices Newton-Schulz orthogonalizes). The eager
//    Muon auto-splits params by p.ndim: ndim==2 → NS, everything else → AdamW
//    (muon.py _split_by_ndim; muon.h:75-76). For the small ViT the ndim==2 weights
//    are exactly these 11 (the flat named_parameters() tensor index + rows + cols);
//    cls_token (ndim==3), all biases + LayerNorm γ/β (ndim==1) take the AdamW 1D
//    tail. The kernel's Muon P2.7 loops THIS table running the grid-cooperative NS
//    per matrix; P3 routes tensor t to the NS apply iff it is in the table, else the
//    AdamW tail. Indices MUST match vit_layout / named_parameters() order. ──
// kVitNumMuon2D = patch_proj + pos + 4 weights/layer (in_proj,out_proj,ff0,ff2)
//   + head.out = 2 + 4*L + 1  (= 11 at L=2). The table is now a FORMULA
// (vit_muon_2d) — a __device__ __constant__ array can't be loop-filled to 195
// entries at L=48.
constexpr int kVitNumMuon2D = 2 + 4 * vit::kLayers + 1;
struct VitMuon2D { int tidx; int rows; int cols; };
// The mi-th Muon 2D matrix (tensor index + rows/cols), L-general. Dense order:
//   mi=0 patch_proj.weight tidx 1  [d, patch];
//   mi=1 pos.weight        tidx 3  [seq, d];
//   mi∈[2,2+4L): li=(mi-2)/4, kind=(mi-2)%4, base=4+12*li →
//     kind0 in_proj  tidx base+0 [3d,d]
//     kind1 out_proj tidx base+2 [d, d]
//     kind2 ff0      tidx base+8 [dff,d]
//     kind3 ff2      tidx base+10 [d, dff]
//   mi=2+4L head out.weight tidx 4+12*L+2 [V,d].
// At L=2 reproduces the old kVitMuon2D[11] EXACTLY (tidx {1,3,4,6,12,14,16,18,24,26,30}).
__host__ __device__ __forceinline__ VitMuon2D vit_muon_2d(int mi) {
    if (mi == 0)                     return { 1, vit::kD,    vit::kPatch }; // patch_proj
    if (mi == 1)                     return { 3, vit::kSeq,  vit::kD     }; // pos
    if (mi == 2 + 4 * vit::kLayers)  return { 4 + 12 * vit::kLayers + 2, vit::kVocab, vit::kD }; // head.out
    const int li   = (mi - 2) / 4;
    const int kind = (mi - 2) % 4;
    const int base = 4 + 12 * li;
    if (kind == 0) return { base + 0,  3 * vit::kD, vit::kD   };  // in_proj
    if (kind == 1) return { base + 2,  vit::kD,     vit::kD   };  // out_proj
    if (kind == 2) return { base + 8,  vit::kDff,   vit::kD   };  // ff0
    return            { base + 10, vit::kD,     vit::kDff };      // ff2
}
// Is tensor index `t` a Muon 2D matrix (orthogonalized in P2.7)? P3 routes only
// the 1D / non-2D weights to the AdamW tail. Closed-form (no table scan):
//   t∈{1,3} (patch_proj/pos), OR t==head.out (4+12L+2), OR a per-layer 2D weight
//   (t∈[4,4+12L) and (t-4)%12 ∈ {0,2,8,10} = in_w/out_w/ff0_w/ff2_w).
__device__ __forceinline__ bool vit_is_muon_2d(int t) {
    if (t == 1 || t == 3) return true;
    if (t == 4 + 12 * vit::kLayers + 2) return true;       // head out.weight
    if (t >= 4 && t < 4 + 12 * vit::kLayers) {
        const int r = (t - 4) % 12;
        return (r == 0 || r == 2 || r == 8 || r == 10);
    }
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

// ── P1 sub-tile granularity (the B1 load-imbalance knob, see the SG_TUNED_VIT_
//    P1_SUBTILE_S block above). kVitP1SubtileS = WHOLE samples a P1 sub-tile
//    spans; kVitP1SubtileRows = kSeq·S = the sub-tile row count the megakernel
//    grid-strides over. S==kSamplesPerTile (default 64) ⇒ rows==kTileM ⇒ the P1
//    loop is byte-identical to the whole-1088-tile schedule (and the m_real
//    row-guard / runtime-m_atoms paths #if out entirely). S<64 sub-tiles P1.
constexpr int kVitP1SubtileS    = SG_TUNED_VIT_P1_SUBTILE_S;
constexpr int kVitP1SubtileRows = vit::kSeq * kVitP1SubtileS;
static_assert(kVitP1SubtileS >= 1 && kVitP1SubtileS <= kSamplesPerTile,
              "SG_TUNED_VIT_P1_SUBTILE_S must be in [1, kSamplesPerTile] whole samples "
              "(never split a kSeq-row sample across CTAs)");
// Max m64 atoms ANY P1 (sub-)tile issues: ceil(kSeq·S / 64). At S=64 this is
// kAtomsM (=17); for S<64 it bounds the runtime m_atoms the wrappers thread.
constexpr int kVitP1MaxAtomsM = (kVitP1SubtileRows + wgs::kWgmmaAtomM - 1) / wgs::kWgmmaAtomM;

// One staged A(64×16) tile (bf16 element count). The M-atom-interleaved engine
// stages kIL of these per k-step (atom-in-group ai at smemA + ai·kVitTcSmemA1)
// sharing ONE B(N×16) tile; the smem arena (VitTcSmem.sA / the selftest arena)
// holds kVitAtomsPerSlot of them. kVitAtomsPerSlot = the widest interleave group
// any call site uses (fwd/dX = min(kAtomsM,kVitMaxIL); dW = min(8,kVitMaxIL)) —
// both cap at kVitMaxIL, so it IS kVitMaxIL whenever the model has >= that many
// atoms (ViT token tiles are 17, dW up to 8, so always >= 2 for the default).
constexpr int kVitTcSmemA1 = wgs::kWgmmaAtomM * wgs::kWgmmaAtomK;   // 64*16 bf16
constexpr int kVitAtomsPerSlot = kVitMaxIL;                        // A-tiles the arena holds

// ── LN vector-grad partials layout (the tile-local γ/β grads). Dense order:
//    4 slots/layer (n1.w,n1.b,n2.w,n2.b) for li∈[0,L), then norm.w,norm.b. At L=2
//    this is {8,9,10,11,20,21,22,23,28,29} (the original 10-slot table). We store
//    them densely [kNumLnVec × kD] per CTA; the P2 reduce maps them back by tensor
//    index via vit_lnvec_tensor_idx (a formula — a __constant__ array can't be
//    filled by a loop at L=48). L-GENERAL: kNumLnVec = 4*L+2. (vit_layout order:
//    per-layer block starts at 4 + li*12, with n1.w/b at +4/+5, n2.w/b at +6/+7;
//    norm.w/b at 4+12L / 4+12L+1.) ─
constexpr int kNumLnVec = 4 * vit::kLayers + 2;    // n1_w,n1_b,n2_w,n2_b ×L + norm_w,norm_b
constexpr int kLnVecElems = kNumLnVec * vit::kD;   // (4*L+2)*kD  (1280 at L=2)
// vit_layout tensor index of LN-vector dense slot v. v∈[0,4L): li=v/4, kind=v%4 →
// 8+12*li+kind (n1.w/b,n2.w/b are tensors 4..7 of each 12-tensor layer block, base
// 4+12*li); v=4L → norm.w (4+12*L); v=4L+1 → norm.b (4+12*L+1). At L=2 reproduces
// the old kLnVecTensorIdx[10] EXACTLY ({8,9,10,11,20,21,22,23,28,29}).
__host__ __device__ __forceinline__ int vit_lnvec_tensor_idx(int v) {
    const int Lx4 = 4 * vit::kLayers;
    if (v < Lx4) return 8 + 12 * (v / 4) + (v % 4);
    return 4 + 12 * vit::kLayers + (v - Lx4);     // 4+12L (norm.w), 4+12L+1 (norm.b)
}

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
//
//  M-ROW GUARD (`m_real`, the SHARED B1-subtile fix): for ANY P1 sub-tile of
//  fewer than kSamplesPerTile whole samples the row count kSeq·S is NOT a
//  multiple of 64, so m_atoms = ceil(rows/64) pads the M-dim. Without a guard
//  srcA is read for ALL m_atoms*64 rows (the pad rows read into the NEXT tile's
//  acts region — a cross-CTA read-before-write — and the last sub-tile reads
//  past T), while `out` is gated only by col<n_real. `m_real` (the real row
//  count, relative to mbase0==0 for the P1 callers) makes pad-row srcA reads
//  return 0 and pad-row `out` writes skip → both the race and the over-read are
//  eliminated cleanly. CRITICAL: this engine is SHARED with the dW (P2) callers
//  and (at S==64) every P1 caller. The guard is gated on SG_TUNED_VIT_P1_SUBTILE_S
//  < 64 so when the knob is OFF the parameter and both guard sites VANISH from
//  the source — every existing caller (and the byte-identical S=64 build)
//  compiles EXACTLY as before. When ON, the P1 wrappers pass m_real = real rows;
//  the dW callers (which already self-guard `out` with m<Nout and read a
//  transposed-K operand, not row-padded acts) would pass m_real = m_atoms*64 ⇒
//  no-op — but they are never recompiled with the guard active in practice since
//  only P1 wrappers add the parameter (see vittc_gemm_* below).
// ════════════════════════════════════════════════════════════════════════
#if SG_TUNED_VIT_P1_SUBTILE_S < 64
  // knob ON: real M-row guard threaded from the P1 wrappers
  #define SG_VIT_GEMM_MREAL_PARAM      , int m_real
  #define SG_VIT_GEMM_SRCA(r, kk)      (((r) < m_real) ? srcA((r), (kk)) : __float2bfloat16(0.f))
  #define SG_VIT_GEMM_OUTVALID(rr)     (col < n_real && ((rr)) < m_real)
#else
  // knob OFF (default S=64): guard erased → byte-identical to the pre-knob engine
  #define SG_VIT_GEMM_MREAL_PARAM
  #define SG_VIT_GEMM_SRCA(r, kk)      srcA((r), (kk))
  #define SG_VIT_GEMM_OUTVALID(rr)     (col < n_real)
#endif
template <int N, int MaxAtomsM, typename SrcA, typename SrcB, typename Out>
__device__ void tc_gemm_block_unpipelined(
        int mbase0, int m_atoms, int n_real, int k_steps SG_VIT_GEMM_MREAL_PARAM,
        SrcA srcA, SrcB srcB, Out out,
        __nv_bfloat16* smemA, __nv_bfloat16* smemB) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;            // 256
    const bool in_wg0 = (tid < 128);
    const int tid_wg = tid & 127;

    // ── M-ATOM-INTERLEAVED wgmma pipeline (task #24, the decoder H1 win ported to
    //    the ViT twin). The OLD body ran each stacked m64 atom's k-chain to
    //    completion SEQUENTIALLY into ONE reused accumulator (atom0 chain → atom0
    //    store → atom1 chain → …), re-staging the shared B (operand) tile ONCE PER
    //    ATOM. (That single-accumulator form was itself a codegen fix: the original
    //    `WgmmaAccum<N> acc[MaxAtomsM]` indexed by the runtime atom `a` placed the
    //    17-wide ViT array in LOCAL memory — 5024 B stack — and serialized every
    //    wgmma via C7515 round-trips. A single register-resident accumulator removed
    //    that; but it also re-staged B per atom and left the MMAs back-to-back-
    //    serialized within the same fp32 fragment.)
    //
    //    Here atoms are processed in GROUPS of kIL (= min(MaxAtomsM, kVitMaxIL),
    //    compile-time, default 2). Within a group each k-step stages the kIL A-tiles
    //    (atom-in-group ai at smemA + ai·kVitTcSmemA1) + ONE shared B-tile, then
    //    issues the kIL wgmmas (one per atom) BACK-TO-BACK into their OWN fp32
    //    fragments before the single per-k wait. Two wins: (1) the kIL atoms are
    //    INDEPENDENT (distinct M-rows / accumulators) → the tensor pipe overlaps
    //    their MMA execution instead of paying each latency raw; (2) the B-tile is
    //    staged ONCE per group instead of kIL times → the (HBM-bound) operand
    //    traffic drops kIL×. kIL stays CAPPED at 2 so the accumulator array
    //    `acc[kIL]` (= kIL·N/2 fp32 regs) is register-resident (NOT the spilling
    //    17-wide array) and ptxas can address it (compile-time unrolled). Per-atom
    //    accumulation stays ASCENDING-k (k=0 ScaleD=0 overwrite, k>0 ScaleD=1 add)
    //    → numerics bit-identical + A/A/A determinism UNCHANGED. (Unpipelined: this
    //    engine has no S-stage ring; the win is purely the within-group overlap +
    //    B-staging-sharing.)
    constexpr int kIL = (MaxAtomsM < kVitMaxIL) ? MaxAtomsM : kVitMaxIL;
    #pragma unroll 1
    for (int g0 = 0; g0 < m_atoms; g0 += kIL) {
        const int gbase = mbase0 + g0 * wgs::kWgmmaAtomM;
        const int g_atoms = (m_atoms - g0) < kIL ? (m_atoms - g0) : kIL;
        wgs::WgmmaAccum<N> acc[kIL];            // kIL live fragments per group
        if (in_wg0) wgs::wgmma_fence();
        #pragma unroll 1
        for (int k = 0; k < k_steps; ++k) {
            #pragma unroll
            for (int ai = 0; ai < kIL; ++ai) {
                if (ai >= g_atoms) break;
                const int mbase = gbase + ai * wgs::kWgmmaAtomM;
                stage_kmajor_tile<wgs::kWgmmaAtomM>(
                    smemA + (int64_t)ai * kVitTcSmemA1, k * wgs::kWgmmaAtomK,
                    [&] (int mn, int kk) { return SG_VIT_GEMM_SRCA(mbase + mn, kk); },
                    tid, nthreads);
            }
            stage_kmajor_tile<N>(
                smemB, k * wgs::kWgmmaAtomK,
                [&] (int mn, int kk) { return srcB(mn, kk); },
                tid, nthreads);
            __syncthreads();   // staged tiles visible to the whole CTA
            if (in_wg0) {
                wgs::SmemDesc dB = wgs::make_desc_B_kmajor<N, wgs::kSwizzleNone>(smemB);
                #pragma unroll
                for (int ai = 0; ai < kIL; ++ai) {
                    if (ai >= g_atoms) break;
                    wgs::SmemDesc dA = wgs::make_desc_A_kmajor<wgs::kWgmmaAtomM, wgs::kSwizzleNone>(
                        smemA + (int64_t)ai * kVitTcSmemA1);
                    if (k == 0)
                        wgs::wgmma_m64nNk16_bf16<N, /*ScaleD=*/0, 0, 0>(acc[ai], dA, dB);
                    else
                        wgs::wgmma_m64nNk16_bf16<N, /*ScaleD=*/1, 0, 0>(acc[ai], dA, dB);
                }
                wgs::wgmma_commit_group();
                wgs::wgmma_wait_group<0>();
            }
            __syncthreads();   // MMAs done reading smem before next stage overwrites
        }
        if (in_wg0) {
            #pragma unroll
            for (int ai = 0; ai < kIL; ++ai) {
                if (ai >= g_atoms) break;
                const int mbase = gbase + ai * wgs::kWgmmaAtomM;
                #pragma unroll
                for (int i = 0; i < wgs::WgmmaAccum<N>::kRegs; ++i) {
                    int row, col;
                    wgs::wgmma_frag_decode(tid_wg, i, N, row, col);
                    if (SG_VIT_GEMM_OUTVALID(mbase + row)) out(mbase + row, col, acc[ai].c[i]);
                }
            }
        }
        __syncthreads();
    }
#else
    (void)mbase0; (void)m_atoms; (void)n_real; (void)k_steps;
    (void)srcA; (void)srcB; (void)out; (void)smemA; (void)smemB;
  #if SG_TUNED_VIT_P1_SUBTILE_S < 64
    (void)m_real;
  #endif
#endif
}
#undef SG_VIT_GEMM_MREAL_PARAM
#undef SG_VIT_GEMM_SRCA
#undef SG_VIT_GEMM_OUTVALID

// ════════════════════════════════════════════════════════════════════════
//  THIN ORIENTATION WRAPPERS over tc_gemm_block_unpipelined — the THREE
//  accessor patterns the engine is silicon-validated on (fwd / dX / dW). The
//  driver calls THESE; it never re-derives the staging. N is the compile-time
//  wgmma atom width; the fwd/dX loops N-tiles internally. Weights convert
//  fp32→bf16 ON READ (deterministic; no bf16 weight buffer needed). Kin is
//  padded to a multiple of kWgmmaAtomK by the caller (49→64 for patch_proj).
//
//  P1 SUB-TILE m_real PLUMBING: when the B1 sub-tile knob is ON (S<64) the P1
//  callers pass a runtime m_real (= the sub-tile's real row count = kSeq·S, or
//  the last sub-tile's residual) so the engine's M-row guard makes the pad rows
//  inert. When OFF (S==64) the param + arg vanish ⇒ byte-identical signatures
//  and engine calls. (The dW/patch-proj callers below do NOT use these wrappers;
//  they call the engine directly with self-guarded `out` lambdas.)
// ════════════════════════════════════════════════════════════════════════
#if SG_TUNED_VIT_P1_SUBTILE_S < 64
  #define SG_VIT_P1_MREAL_PARAM         , int m_real
  #define SG_VIT_P1_MREAL_ARG           , m_real
  // fwd_f32 / dx_f32 historically hardcoded the compile-time kAtomsM; when ON they
  // take the runtime m_atoms (ceil(sub-tile rows / 64)) instead.
  #define SG_VIT_P1_MATOMS_PARAM_F32    , int m_atoms
  #define SG_VIT_P1_MATOMS_ARG_F32      m_atoms
#else
  #define SG_VIT_P1_MREAL_PARAM
  #define SG_VIT_P1_MREAL_ARG
  #define SG_VIT_P1_MATOMS_PARAM_F32
  #define SG_VIT_P1_MATOMS_ARG_F32      kAtomsM
#endif

// ── P1 tile-function CALL-SITE arg packs. A P1 sub-tile owns `nrows` rows; the
//    GEMMs over it issue m_atoms_rt = ceil(nrows/64) atoms with m_real = nrows.
//    At S==64 nrows==kTileM (a multiple of 64) so m_atoms_rt==kAtomsM and the
//    guard is a no-op — these expand to the EXACT pre-knob call sites (literal
//    kAtomsM for the bf16 qkv call; no extra args for fwd_f32 / dx_f32).
#if SG_TUNED_VIT_P1_SUBTILE_S < 64
  #define SG_VIT_P1_BF16_MA_MR   m_atoms_rt, nrows   // vittc_gemm_fwd: m_atoms, m_real
  #define SG_VIT_P1_F32_MA_MR    , m_atoms_rt, nrows // vittc_gemm_*_f32: , m_atoms, m_real
  // DIRECT engine calls (not via a wrapper). m_real is in ABSOLUTE row coords
  // (the engine guards mbase0+row). patch_proj has mbase0==0 ⇒ m_real = the real
  // patch-row count. The dW callers pass a nonzero mbase + already self-guard
  // `out` with m<Nout and read transposed-K operands (never row-padded acts), so
  // their guard is a strict no-op: m_real = 0x7fffffff keeps every existing dW row.
  #define SG_VIT_PATCH_MREAL_ARG , npatch_rows
  #define SG_VIT_DW_MREAL_ARG    , 0x7fffffff
#else
  #define SG_VIT_P1_BF16_MA_MR   kAtomsM
  #define SG_VIT_P1_F32_MA_MR
  #define SG_VIT_PATCH_MREAL_ARG
  #define SG_VIT_DW_MREAL_ARG
#endif

// (fwd) Y[M,Nout] = X[M,Kpad] @ W[Nout,Kin]ᵀ.  Tiles N over [0,Nout) in width-N
// atoms. M = m_atoms stacked atoms (kAtomsM for token tiles; fewer for the last
// ragged tile / patch tiles). Y bf16 row-major [M, Nout]. `Kreal` is the true
// contracted dim (<= Kpad); pad-K reads return 0 so they are inert.
template <int N>
__device__ __forceinline__ void vittc_gemm_fwd(
        const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
        __nv_bfloat16* __restrict__ Yout, int Kreal, int Kpad, int Nout,
        int m_atoms SG_VIT_P1_MREAL_PARAM, __nv_bfloat16* sA, __nv_bfloat16* sB) {
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
            0, m_atoms, n_real, k_steps SG_VIT_P1_MREAL_ARG, srcA, srcB, out, sA, sB);
    }
}

// Same as vittc_gemm_fwd but emits the fp32 result (no bf16 round) — for fwd
// outputs consumed by fp32 elementwise stages. Writes [M,Nout] fp32 at `Yf32`.
// (K is never padded for these — only the in/out/ff GEMMs use it, all K∈{d,dff}
// already multiples of 16.)
template <int N>
__device__ __forceinline__ void vittc_gemm_fwd_f32(
        const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
        float* __restrict__ Yf32, int Kin, int Nout
        SG_VIT_P1_MATOMS_PARAM_F32 SG_VIT_P1_MREAL_PARAM,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    const int k_steps = Kin / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return X[(int64_t)m * Kin + k]; };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return nn < Nout ? __float2bfloat16(W[(int64_t)nn * Kin + k]) : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) { Yf32[(int64_t)m * Nout + n0 + n] = v; };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, SG_VIT_P1_MATOMS_ARG_F32, n_real, k_steps SG_VIT_P1_MREAL_ARG, srcA, srcB, out, sA, sB);
    }
}

// (dX) dX[M,Kin] = dY[M,Nout] @ W[Nout,Kin].  N(wgmma)=Kin (tiled by width N),
// K=Nout. W staged transposed: srcB(n=kin,k=out)=W[out·Kin+kin] (fp32→bf16).
// Writes fp32 dX [M,Kin].
template <int N>
__device__ __forceinline__ void vittc_gemm_dx_f32(
        const __nv_bfloat16* __restrict__ dY, const float* __restrict__ W,
        float* __restrict__ dXf32, int Kin, int Nout
        SG_VIT_P1_MATOMS_PARAM_F32 SG_VIT_P1_MREAL_PARAM,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    const int k_steps = Nout / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Kin; n0 += N) {
        const int n_real = (Kin - n0) < N ? (Kin - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return dY[(int64_t)m * Nout + k]; };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return nn < Kin ? __float2bfloat16(W[(int64_t)k * Kin + nn]) : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) { dXf32[(int64_t)m * Kin + n0 + n] = v; };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, SG_VIT_P1_MATOMS_ARG_F32, n_real, k_steps SG_VIT_P1_MREAL_ARG, srcA, srcB, out, sA, sB);
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
#if SG_TUNED_VIT_P1_SUBTILE_S < 64
    // B1 sub-tile: nrows = kSeq·S is generally not a multiple of 64 → the linear
    // GEMMs issue ceil(nrows/64) m64 atoms and guard pad rows with m_real=nrows.
    const int m_atoms_rt = (nrows + wgs::kWgmmaAtomM - 1) / wgs::kWgmmaAtomM;
#endif
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
                0, pm_atoms, n_real, k_steps SG_VIT_PATCH_MREAL_ARG, srcA, srcB, out, sA, sB);
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
        vittc_gemm_fwd<SG_TUNED_TILE_N>(Xin, L.in_w, sc.qkv[li], vit::kD, vit::kD, 3 * vit::kD, SG_VIT_P1_BF16_MA_MR, sA, sB);
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
                                            sc.work, vit::kD, vit::kD SG_VIT_P1_F32_MA_MR, sA, sB);
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
                                            sc.work, vit::kD, vit::kDff SG_VIT_P1_F32_MA_MR, sA, sB);
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
                                            sc.work, vit::kDff, vit::kD SG_VIT_P1_F32_MA_MR, sA, sB);
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
#if SG_TUNED_VIT_P1_SUBTILE_S < 64
    // B1 sub-tile: dX GEMMs issue ceil(nrows/64) m64 atoms with m_real=nrows.
    const int m_atoms_rt = (nrows + wgs::kWgmmaAtomM - 1) / wgs::kWgmmaAtomM;
#endif
    float* gn_n1w[vit::kLayers]; float* gn_n1b[vit::kLayers];
    float* gn_n2w[vit::kLayers]; float* gn_n2b[vit::kLayers];
    for (int li = 0; li < vit::kLayers; ++li) {
        gn_n1w[li] = lnvec + (int64_t)(li * 4 + 0) * vit::kD;
        gn_n1b[li] = lnvec + (int64_t)(li * 4 + 1) * vit::kD;
        gn_n2w[li] = lnvec + (int64_t)(li * 4 + 2) * vit::kD;
        gn_n2b[li] = lnvec + (int64_t)(li * 4 + 3) * vit::kD;
    }
    float* gn_normw = lnvec + (int64_t)(4 * vit::kLayers + 0) * vit::kD;  // 8*kD at L=2
    float* gn_normb = lnvec + (int64_t)(4 * vit::kLayers + 1) * vit::kD;  // 9*kD at L=2

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
                                           sc.work, vit::kDff, vit::kD SG_VIT_P1_F32_MA_MR, sA, sB);  // dgact [nrows,dff]
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
                                           sc.x1, /*Kin=*/vit::kD, /*Nout=*/vit::kDff SG_VIT_P1_F32_MA_MR, sA, sB);  // dx1_ffn [nrows,d]
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
                                           sc.work, vit::kD, vit::kD SG_VIT_P1_F32_MA_MR, sA, sB);  // dctx
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
                                           sc.work, /*Kin=*/vit::kD, /*Nout=*/3 * vit::kD SG_VIT_P1_F32_MA_MR, sA, sB);  // dx_in_attn
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

// Number of dW specs: 1 patch_proj + 4 per layer (in_proj,out_proj,ff0,ff2) + 1
//   head.out = 2 + 4*L  (= 10 at L=2, the original spec[10]; = 194 at flagship L=48).
// This is the compile-time bound for EVERY VitDwSpec array (the VitTcSmem member,
// every spec[] signature/local, the dW phase loops). At L=2 it is exactly 10 → the
// spec arrays + their smem footprint are byte-identical.
constexpr int kVitNumDwSpecs = 2 + 4 * vit::kLayers;

// Build the dW specs (called by all CTAs; cheap). T = B*kSeq; Tp = B*kNPatch.
// kVitNumDwSpecs = 2 + 4*L (patch + 4/layer + head); 10 at L=2.
__device__ __forceinline__ void vittc_build_dw_specs(
        const VitActs& acts, int B, int T, int Tp, VitDwSpec spec[kVitNumDwSpecs]) {
    // vit_layout weight tensor indices (and bias idx). Per-layer block base = 4 + li*12:
    //   in_w = base+0 (in_b base+1), out_w base+2 (out_b base+3),
    //   ff0_w base+8 (ff0_b base+9), ff2_w base+10 (ff2_b base+11).
    //   patch_proj.weight=1 (bias=2); head out.weight = 4+12*L+2 (out.bias +1; 30/31 at L=2).
    // spec[0] = patch_proj; spec[1..4L] = transformer (li,kind); spec[1+4L] = head.
    {
        VitDwSpec& sp = spec[0];
        sp.dY = nullptr; sp.X = acts.X_patch; sp.Nout = vit::kD; sp.Kin = vit::kPatch;
        sp.K = Tp; sp.Kpad = vit::kPatch;  // K (contraction over patch rows) IS Tp; Kin padding handled in run_tile
        sp.grad_off = kVitOffsets[1]; sp.bias_off = kVitOffsets[2]; sp.kind = 1;
    }
    for (int s = 0; s < 4 * vit::kLayers; ++s) {
        const int li = s / 4, kk = s % 4;
        const int base = 4 + li * 12;
        VitDwSpec& sp = spec[1 + s];
        sp.K = T; sp.kind = 0; sp.Kpad = 0;
        if (kk == 0)      { sp.dY = acts.dY_qkv[li]; sp.X = acts.X_in[li];  sp.Nout = 3 * vit::kD; sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 0]; sp.bias_off = kVitOffsets[base + 1]; }
        else if (kk == 1) { sp.dY = acts.dY_a[li];   sp.X = acts.X_ctx[li]; sp.Nout = vit::kD;     sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 2]; sp.bias_off = kVitOffsets[base + 3]; }
        else if (kk == 2) { sp.dY = acts.dY_ff0[li]; sp.X = acts.X_x1[li];  sp.Nout = vit::kDff;   sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 8]; sp.bias_off = kVitOffsets[base + 9]; }
        else              { sp.dY = acts.dY_ff2[li]; sp.X = acts.X_gact[li];sp.Nout = vit::kD;     sp.Kin = vit::kDff; sp.grad_off = kVitOffsets[base + 10]; sp.bias_off = kVitOffsets[base + 11]; }
    }
    VitDwSpec& hd = spec[1 + 4 * vit::kLayers];   // head spec (was spec[9] at L=2)
    hd.dY = acts.dY_logits; hd.X = acts.X_hn; hd.Nout = vit::kVocab; hd.Kin = vit::kD; hd.K = B; hd.kind = 0; hd.Kpad = 0;
    hd.grad_off = kVitOffsets[4 + 12 * vit::kLayers + 2];   // out.weight (30 at L=2)
    hd.bias_off = kVitOffsets[4 + 12 * vit::kLayers + 3];   // out.bias   (31 at L=2)
}

// Total number of 64×N dW output tiles across the 10 weights (for the tile loop).
// ── dW M-atom INTERLEAVE group width (task #24 H3 port). A dW output "tile"
//    (work item) owns a GROUP of up to kVitDwIL consecutive m64 atoms instead of
//    one, so the engine runs kVitDwIL wgmmas/k-step into independent fragments
//    sharing ONE staged X (B-operand) tile → the H1 win (overlap the MMAs +
//    kVitDwIL× less X-operand HBM traffic) now applies to the dW K=T contraction.
//    kVitDwIL == the engine's interleave cap kVitMaxIL so the ring smem
//    (VitTcSmem.sA, sized kVitAtomsPerSlot=kVitMaxIL A-tiles/slot) and the
//    selftest-validated dW path (gemm_dw_kernel: MaxAtomsM=8, kIL=2) match exactly.
//    Ascending-k per atom is unchanged → numerics bit-identical + A/A/A
//    determinism. ViT dW atom counts: patch_proj(d=128→2), qkv(3d=384→6),
//    attn_out(d=128→2), ff0(dff=512→8), ff2(d=128→2), head(V=97→2) — all EVEN, but
//    the ragged-tail logic (g_atoms = min) is kept for safety (no even shortcut). ──
constexpr int kVitDwIL = kVitMaxIL;
__device__ __host__ __forceinline__ int vit_dw_atoms(int Nout)  { return (Nout + 63) / 64; }
__device__ __host__ __forceinline__ int vit_dw_groups(int Nout) {
    return (vit_dw_atoms(Nout) + kVitDwIL - 1) / kVitDwIL;
}

// Total number of dW output WORK ITEMS (M-atom groups × N-tiles) across the 10
// weights. The N-tiling is over Kin (the in-dim), padded so the GEMM N covers it.
template <int N>
__device__ __forceinline__ int vittc_dw_total_tiles(const VitDwSpec spec[kVitNumDwSpecs]) {
    int n = 0;
    for (int s = 0; s < kVitNumDwSpecs; ++s)
        n += vit_dw_groups(spec[s].Nout) * ((spec[s].Kin + N - 1) / N);
    return n;
}

// Run ONE global dW WORK ITEM `gt` (if it belongs to this CTA): decode (weight,
// M-atom GROUP, N-tile), then contract K via the dW GEMM into grad. The group is
// up to kVitDwIL stacked m64 atoms → the engine interleaves them (kIL=kVitDwIL):
// kVitDwIL wgmmas/k-step into independent fragments sharing ONE staged X B-tile
// (H1's win, now on the dW K contraction). For patch_proj (kind==1) the dY adjoint
// is dh0's PATCH rows: dh0 is [T,d] token-row-major; its patch rows for sample si,
// patch p are at token row si*kSeq + (1+p). The dW contracts over the Tp patch rows
// in ascending patch order (k=0..Tp-1 maps to si=k/kNPatch, p=k%kNPatch → token row
// si*kSeq+1+p). The engine adds mbase0=mbase and passes the GLOBAL row m to the
// accessors/out; the out lambda writes grad by global m (m<Nout guard handles a
// ragged final group), so multi-atom groups need no per-atom shimming here.
template <int N>
__device__ __forceinline__ void vittc_dw_run_tile(
        const VitDwSpec spec[kVitNumDwSpecs], int gt, const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ grad, __nv_bfloat16* sA, __nv_bfloat16* sB) {
    int acc = 0, s = 0, m_group = 0, n_tile = 0;
    for (s = 0; s < kVitNumDwSpecs; ++s) {
        const int ng = vit_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; break; }
        acc += ng * nt;
    }
    const VitDwSpec& sp = spec[s];
    const int n_atoms = vit_dw_atoms(sp.Nout);
    const int a0 = m_group * kVitDwIL;               // first atom of the group
    const int g_atoms = (n_atoms - a0) < kVitDwIL ? (n_atoms - a0) : kVitDwIL;
    const int mbase = a0 * 64;
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
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kVitDwIL>(
            mbase, g_atoms, n_real, k_steps SG_VIT_DW_MREAL_ARG, srcA, srcB, out, sA, sB);
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
    tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kVitDwIL>(
        mbase, g_atoms, n_real, k_steps SG_VIT_DW_MREAL_ARG, srcA, srcB, out, sA, sB);
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
// Floats per (gt,kc) split-K slot: a work item is an M-atom GROUP of up to kVitDwIL
// stacked m64 atoms → kVitDwIL·64 rows × N cols. (Was 64·N for the 1-atom item.)
constexpr int kVitDwTileFloats = kVitDwIL * wgs::kWgmmaAtomM * kVitMaxTileN;

// COMPILE-TIME max #dW output WORK ITEMS (the 10 dW have fixed Nout/Kin; ViT dims
// are compile-time → constant). Mirrors vittc_dw_total_tiles at N=kVitMaxTileN. The
// M factor is M-atom GROUPS of kVitDwIL (ceil(ceil(Nout/64)/kVitDwIL)), matching
// vit_dw_groups — NOT single atoms.
//   patch_proj(d×patch) + per-layer[qkv(3d×d),attn_out(d×d),ff0(dff×d),ff2(d×dff)]
//   ×kLayers + head(V×d).
constexpr int kVitDwMGroups(int Nout) {                 // ceil(ceil(Nout/64)/kVitDwIL)
    return (((Nout + 63) / 64) + kVitDwIL - 1) / kVitDwIL;
}
constexpr int kVitDwTilesPerLayer =
      kVitDwMGroups(3*vit::kD) * ((vit::kD   + kVitMaxTileN - 1)/kVitMaxTileN)   // qkv
    + kVitDwMGroups(vit::kD)   * ((vit::kD   + kVitMaxTileN - 1)/kVitMaxTileN)   // attn_out
    + kVitDwMGroups(vit::kDff) * ((vit::kD   + kVitMaxTileN - 1)/kVitMaxTileN)   // ff0
    + kVitDwMGroups(vit::kD)   * ((vit::kDff + kVitMaxTileN - 1)/kVitMaxTileN);  // ff2
constexpr int kVitDwPatchTiles =
      kVitDwMGroups(vit::kD) * ((vit::kPatch + kVitMaxTileN - 1)/kVitMaxTileN);  // patch_proj
constexpr int kVitDwHeadTiles =
      kVitDwMGroups(vit::kVocab) * ((vit::kD + kVitMaxTileN - 1)/kVitMaxTileN);  // head
constexpr int kVitDwMaxTiles =
      kVitDwPatchTiles + vit::kLayers * kVitDwTilesPerLayer + kVitDwHeadTiles;

// Split-K dW partial-scratch float count (host carves it from the workspace tail).
// 0 when G==1 → no extra scratch (the single-CTA path is byte-identical). The total
// (groups·kVitDwIL·64·N) == (atoms·64·N) for even atom counts; for ViT all 10 dW
// have EVEN atom counts (2/6/2/8/2/2) so it is exactly unchanged vs the per-atom form.
__host__ __device__ __forceinline__ int64_t vit_dw_part_floats(int G) {
    return (G > 1) ? (int64_t)kVitDwMaxTiles * G * kVitDwTileFloats : 0;
}

// Decode global dW work-item index gt → (spec index s, m_group, n_tile). Single-
// source (the SAME walk vittc_dw_run_tile + vittc_dw_total_tiles use): m_group
// indexes M-atom GROUPS of kVitDwIL (group's first atom = m_group·kVitDwIL).
template <int N>
__device__ __forceinline__ void vittc_dw_decode(
        const VitDwSpec spec[kVitNumDwSpecs], int gt, int& s, int& m_group, int& n_tile) {
    int acc = 0;
    for (s = 0; s < kVitNumDwSpecs; ++s) {
        const int ng = vit_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; return; }
        acc += ng * nt;
    }
    s = 1 + 4 * vit::kLayers; m_group = 0; n_tile = 0;   // unreachable (gt < n_dw); keep defined (head idx)
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
        const VitDwSpec spec[kVitNumDwSpecs], int gt, int kc, int G,
        const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ dw_part, __nv_bfloat16* sA, __nv_bfloat16* sB) {
    int s, m_group, n_tile;
    vittc_dw_decode<N>(spec, gt, s, m_group, n_tile);
    const VitDwSpec& sp = spec[s];
    const int n_atoms = vit_dw_atoms(sp.Nout);
    const int a0 = m_group * kVitDwIL;                      // first atom of the group
    const int g_atoms = (n_atoms - a0) < kVitDwIL ? (n_atoms - a0) : kVitDwIL;
    const int mbase = a0 * 64;
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
    // Slot holds the GROUP's kVitDwIL·64 local rows (lr) × N cols (kVitDwTileFloats).
    float* slot = dw_part + ((int64_t)gt * G + kc) * kVitDwTileFloats;
    // Empty-chunk guard (KS<G): a k_steps=0 GEMM would emit the uninitialized
    // accumulator → zero the WHOLE group slot + return (the reduce sums all G slots;
    // an empty chunk MUST be 0). Only g_atoms·64 rows are valid but zeroing the full
    // kVitDwIL·64·N slot keeps it simple + safe (the reduce reads only valid rows).
    if (kc_steps <= 0) {
        for (int i = threadIdx.x; i < kVitDwTileFloats; i += blockDim.x) slot[i] = 0.0f;
        __syncthreads();
        return;
    }
    // out(mbase+row, col, v): m is GLOBAL (srcA needs it; engine adds mbase0=mbase);
    // the slot holds the GROUP's kVitDwIL·64 LOCAL rows → index by (m - mbase). A
    // ragged final group's tail rows stay zeroed/unwritten but the reduce reads only
    // valid rows → never used. (A `lr<64` clamp would lose atom>=1's rows → the dW bug.)
    auto out = [&] (int m, int n, float v) {
        const int lr = m - mbase;
        if (lr >= 0 && lr < kVitDwIL * 64 && n < N) slot[(int64_t)lr * N + n] = v; };
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
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kVitDwIL>(
            mbase, g_atoms, /*n_real=*/N, kc_steps SG_VIT_DW_MREAL_ARG, srcA, srcB, out, sA, sB);
        return;
    }
    // kind==0: A[m=out,k=t]=dY[t,out]; B[n=in,k=t]=X[t,in] (both transposed reads).
    const __nv_bfloat16* dY = sp.dY; const __nv_bfloat16* X = sp.X;
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
        return m < Nout ? dY[(int64_t)(k0 * wgs::kWgmmaAtomK + k) * Nout + m] : __float2bfloat16(0.f); };
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
        int nn = n0 + n; return nn < Kin ? X[(int64_t)(k0 * wgs::kWgmmaAtomK + k) * Kin + nn] : __float2bfloat16(0.f); };
    tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kVitDwIL>(
        mbase, g_atoms, /*n_real=*/N, kc_steps SG_VIT_DW_MREAL_ARG, srcA, srcB, out, sA, sB);
}

// Deterministic reduce: output work item gt (% nCTA) sums its G chunk-partials
// ascending-kc → grad. SAME (gt → geometry) decode as the partial. Each work item
// is an M-atom GROUP (up to kVitDwIL stacked m64 atoms, H3), so its slot holds the
// group's kVitDwIL·64 group-local rows (row); mbase is the group's first global row.
template <int N>
__device__ __forceinline__ void vittc_dw_reduce_splitk(
        const VitDwSpec spec[kVitNumDwSpecs], int n_dw, int G, const float* __restrict__ dw_part,
        float* __restrict__ grad, int cta, int nCTA) {
    for (int gt = cta; gt < n_dw; gt += nCTA) {
        int s, m_group, n_tile;
        vittc_dw_decode<N>(spec, gt, s, m_group, n_tile);
        const VitDwSpec& sp = spec[s];
        const int mbase = m_group * kVitDwIL * 64;          // group's first global row
        const int n0 = n_tile * N;
        const int n_real = (sp.Kin - n0) < N ? (sp.Kin - n0) : N;
        // Valid group rows: up to kVitDwIL·64, clamped to the weight's row count.
        const int grp_rows = kVitDwIL * 64;
        const int Nrow = (sp.Nout - mbase) < grp_rows ? (sp.Nout - mbase) : grp_rows;
        const int64_t base = (int64_t)gt * G * kVitDwTileFloats;
        for (int idx = threadIdx.x; idx < Nrow * n_real; idx += blockDim.x) {
            const int row = idx / n_real, col = idx % n_real;   // row = group-local lr
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
        const VitDwSpec spec[kVitNumDwSpecs], const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ grad, int cta, int nCTA) {
    const int warp = threadIdx.x >> 5;            // 0..(blockDim/32 - 1)
    const int lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    int gcol_base = 0;
    for (int s = 0; s < kVitNumDwSpecs; ++s) {
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
        const int goff = vit_lnvec_tensor_idx(v);   // was kLnVecTensorIdx[v]
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
