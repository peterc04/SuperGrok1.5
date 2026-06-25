#ifndef SG_FUSED_SM90_MODEL_STAGE_MAMBA_TC_CUH_
#define SG_FUSED_SM90_MODEL_STAGE_MAMBA_TC_CUH_
// ============================================================================
// csrc/fused/sm_90/model_stage_mamba_tc.cuh — Mamba-3 "TC" support header.
//
// MAMBA-3 NOTE: the Mamba-3 mixer is SCAN-DOMINATED (the complex
// exponential-trapezoidal selective scan is the wall; the 7 projections are a
// tiny share). So the production "wgmma" Mamba megakernel
// (fused_mamba_megakernel_tc) runs the VALIDATED scalar per-sample fwd+bwd from
// model_stage_mamba3.cuh (mb_forward_sample / mb_backward_sample — matched to the
// fp64 oracle to ~2e-6) batch-parallel into a per-CTA full-grad partial, NOT the
// old Mamba-1 dW-output-stationary wgmma tile machinery. This header therefore no
// longer carries the MbActs / MbTileScratch / MbDwSpec tile/dW kernels (they were
// Mamba-1-shaped and are obsolete on the scalar path). It keeps ONLY the symbols
// the megakernel + launcher still reference:
//   * the GEMM-substrate tile constants (used by the dormant MbTcSmem struct);
//   * the per-model Muon 2D-weight table (kMbMuon2D / mb_is_muon_2d) — the P2.7
//     Newton-Schulz phase orthogonalizes these matrices;
//   * tiny workspace-sizing stubs (acts/dW partials are zero-width now);
//   * mbtc::kTileM (the standalone bench TU references it).
//
// The optimizer tails (AdamW/Lion/.../Muon/SG2) + the SAM 2nd backward are
// model-math-agnostic and live in fused_mamba_megakernel.cuh; they operate on the
// flat reduced grad + params, so this header needs no tile kernels for them.
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/mamba3_layout.cuh"
#include "csrc/fused/sm_90/model_stage_mamba3.cuh"   // mb:: dims, MambaWeights/Grad, scan/RMSNorm/SwiGLU
#include "csrc/backends/cuda/sm_90/wgmma.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sg { namespace fused { namespace sm90 {

namespace wgs = ::sg::sm90::wgs;

#ifndef SG_TUNED_TILE_M
#define SG_TUNED_TILE_M 64
#endif
#ifndef SG_TUNED_TILE_N
#define SG_TUNED_TILE_N 128
#endif

namespace mbtc {

// ── GEMM-substrate tile constants (the dormant MbTcSmem ring sizing reads these;
//    kept so that struct + any tuner-derived sizing still compiles). ───────────
constexpr int kTileM = SG_TUNED_TILE_M;
static_assert(kTileM % wgs::kWgmmaAtomM == 0,
              "SG_TUNED_TILE_M must be a multiple of 64 (wgmma m64 atom)");
constexpr int kAtomsM = kTileM / wgs::kWgmmaAtomM;
#ifndef SG_MBTC_STAGES
#define SG_MBTC_STAGES 2
#endif
constexpr int kMbTcStages = SG_MBTC_STAGES;
constexpr int kMbMaxIL = 2;
constexpr int kMbAtomsPerSlot = (kAtomsM < kMbMaxIL) ? kAtomsM : kMbMaxIL;
constexpr int kMbDwSplitK = 1;

// Dummy dW spec (the dormant MbTcSmem holds an array of these; unused on the
// scalar path). Kept minimal so the struct + any reference compiles.
struct MbDwSpec {
    const __nv_bfloat16* dY; const __nv_bfloat16* X;
    int Nout; int Kin; int T; int grad_off;
    bool has_bias; const __nv_bfloat16* dY_bias; int bias_off;
};
// Dormant dW-spec count for the (unused-on-the-scalar-path) MbTcSmem::spec[] array.
// FIXED at 8 (the old Mamba-1 Fork-B 4-GEMM×2-layer count) — NOT layer-scaled: the
// scalar Mamba TC path never populates/reads these specs, so keeping it constant
// holds MbTcSmem BYTE-IDENTICAL at every L (no dormant-smem growth at the flagship
// L=24). Named (vs a bare literal) to mirror the decoder's kDecNumDwSpecs and to be
// the single knob if a Mamba TC dW path is ever revived.
constexpr int kMbNumDwSpecs = 8;

// Workspace-sizing stubs — the scalar path's per-CTA partial is the FULL grad
// (sized in mb_tc_workspace_floats), so the old acts / split-K dW regions are
// zero-width.
__host__ __device__ __forceinline__ int64_t mb_acts_floats(int /*T*/) { return 0; }
__host__ __device__ __forceinline__ int64_t mb_dw_part_floats(int /*G*/) { return 0; }

// ── Per-model Muon 2D-weight set (the ndim==2 parameters Muon's P2.7 Newton-Schulz
//    orthogonalizes; ndim==1/other weights take the AdamW aux tail). Now a FORMULA
//    (mb_muon_2d / mb_is_muon_2d), L-general: 2 + 7*L + 1 matrices (tok, pos, 7
//    2D weights/layer, head.out). Flat tensor index (named_parameters() order) +
//    rows[dim0] + cols[dim1]; VERIFIED value-identical at L=2 against the live
//    Mamba3Model (the original 17-matrix toy table). A_log is [n_heads] (ndim==1) →
//    NOT here, unlike Mamba-1. ──────────────────────────────────────────────────
// kMbNumMuon2D = tok + pos + 7 weights/layer (in_proj,x_proj,dt_proj,out_proj,
//   gate,up,down) + head.out = 2 + 7*L + 1 (= 17 at L=2). The table is now a
// FORMULA (mb_muon_2d) — a __device__ __constant__ array can't be loop-filled to
// 171 entries at the flagship L=24.
constexpr int kMbNumMuon2D = 2 + 7 * mb::kLayers + 1;
struct MbMuon2D { int tidx; int rows; int cols; };
// The mi-th Muon 2D matrix (tensor index + rows/cols), L-general. Per-layer
// 20-tensor block (li) starts at flat tidx 2+20*li; the 7 2D weights are at
// block-offsets {7,8,9,15,17,18,19} = in_proj,x_proj,dt_proj_w,out_proj,gate,up,down.
// Dense order:
//   mi=0 tok[V,d]; mi=1 pos[seq,d];
//   mi∈[2,2+7L): li=(mi-2)/7, kind=(mi-2)%7 →
//     kind0 in_proj  tidx 2 +20li+7  [2*d_inner, d]
//     kind1 x_proj   tidx 2 +20li+8  [x_proj_out, d_inner]
//     kind2 dt_proj  tidx 2 +20li+9  [n_heads, dt_rank]
//     kind3 out_proj tidx 2 +20li+15 [d, d_inner]
//     kind4 gate     tidx 2 +20li+17 [d_ff, d]
//     kind5 up       tidx 2 +20li+18 [d_ff, d]
//     kind6 down     tidx 2 +20li+19 [d, d_ff]
//   mi=2+7L head out.weight tidx 2+20*L+1 [phead, d].
// At L=2 reproduces the old kMbMuon2D[17] EXACTLY (tidx {0,1,9,10,11,17,19,20,21,
// 29,30,31,37,39,40,41,43}).
__host__ __device__ __forceinline__ MbMuon2D mb_muon_2d(int mi) {
    if (mi == 0)                       return { 0, mb::kVocab, mb::kD };   // tok
    if (mi == 1)                       return { 1, mb::kSeq,   mb::kD };   // pos
    if (mi == 2 + 7 * mb::kLayers)     return { 2 + 20 * mb::kLayers + 1, mb::kPHead, mb::kD }; // head.out
    const int li   = (mi - 2) / 7;
    const int kind = (mi - 2) % 7;
    const int base = 2 + 20 * li;
    if (kind == 0) return { base + 7,  2 * mb::kDInner, mb::kD       };  // in_proj
    if (kind == 1) return { base + 8,  mb::kXProj,      mb::kDInner  };  // x_proj
    if (kind == 2) return { base + 9,  mb::kNHeads,     mb::kDtRank  };  // dt_proj
    if (kind == 3) return { base + 15, mb::kD,          mb::kDInner  };  // out_proj
    if (kind == 4) return { base + 17, mb::kDff,        mb::kD       };  // gate
    if (kind == 5) return { base + 18, mb::kDff,        mb::kD       };  // up
    return            { base + 19, mb::kD,          mb::kDff     };      // down
}
// Is tensor index `t` a Muon 2D matrix (orthogonalized in P2.7)? P3 routes only the
// 1D / non-2D weights to the AdamW aux tail. Closed-form (no table scan):
//   t∈{0,1} (tok/pos), OR t==head.out (2+20L+1), OR a per-layer 2D weight
//   (t∈[2,2+20L) and (t-2)%20 ∈ {7,8,9,15,17,18,19}).
__device__ __forceinline__ bool mb_is_muon_2d(int t) {
    if (t == 0 || t == 1) return true;
    if (t == 2 + 20 * mb::kLayers + 1) return true;       // head out.weight
    if (t >= 2 && t < 2 + 20 * mb::kLayers) {
        const int r = (t - 2) % 20;
        return (r == 7 || r == 8 || r == 9 || r == 15 || r == 17 || r == 18 || r == 19);
    }
    return false;
}
// Largest 2D weight (numel) + largest #rows over the Muon-2D set — sizes the per-
// matrix NS scratch (mb_tc_muon_floats carves 4*kMbMuonMaxNumel + kMbMuonMaxRows²).
// kMbMuonMaxNumel is derived from the LAYOUT's max_size() (the largest of ALL
// tensors). VERIFIED at every config the global-max tensor is a 2D Muon weight, so
// max_size() is an exact, width-safe bound: prod d=128 → x_proj 86016; bench
// d=1024 → in_proj 4194304; flagship d=2048 → in_proj 16777216. (The old literal
// mb::kXProj*mb::kDInner was a d=128 COINCIDENCE — x_proj is NOT the largest 2D
// weight at the flagship, where in_proj 2*d_inner*d dominates; that literal would
// under-size the NS scratch ~7× → Muon-cell OOB.) At d=128 this equals 86016
// (== the old literal) → byte-identical workspace carve on prod/bench. kMbMuonMaxRows
// = 2*d_inner (in_proj rows) is the largest #rows over the 2D set at every width;
// A = X Xᵀ is rows×rows.
constexpr int kMbMuonMaxNumel = kMambaMaxTensorNumel;      // == mamba_layout_check::max_size()
constexpr int kMbMuonMaxRows  = 2 * mb::kDInner;           // 512 at d=128 (in_proj rows)

}  // namespace mbtc

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGE_MAMBA_TC_CUH_
