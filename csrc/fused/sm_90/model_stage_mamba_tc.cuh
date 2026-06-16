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

// Workspace-sizing stubs — the scalar path's per-CTA partial is the FULL grad
// (sized in mb_tc_workspace_floats), so the old acts / split-K dW regions are
// zero-width.
__host__ __device__ __forceinline__ int64_t mb_acts_floats(int /*T*/) { return 0; }
__host__ __device__ __forceinline__ int64_t mb_dw_part_floats(int /*G*/) { return 0; }

// ── Per-model Muon 2D-weight table (the ndim==2 parameters Muon's P2.7
//    Newton-Schulz orthogonalizes; ndim==1/other weights take the AdamW aux tail).
//    Flat tensor index (named_parameters() order) + rows[dim0] + cols[dim1],
//    VERIFIED against the live Mamba3Model (17 matrices at the toy config).
//    A_log is now [n_heads] (ndim==1) → NOT here, unlike Mamba-1. ──────────────
constexpr int kMbNumMuon2D = 17;
struct MbMuon2D { int tidx; int rows; int cols; };
__device__ __constant__ MbMuon2D kMbMuon2D[kMbNumMuon2D] = {
    {  0, mb::kVocab,      mb::kD       },   // tok.weight            [99,128]
    {  1, mb::kSeq,        mb::kD       },   // pos.weight            [8,128]
    {  9, 2 * mb::kDInner, mb::kD       },   // L0 in_proj.weight     [512,128]
    { 10, mb::kXProj,      mb::kDInner  },   // L0 x_proj.weight      [336,256]
    { 11, mb::kNHeads,     mb::kDtRank  },   // L0 dt_proj.weight     [4,8]
    { 17, mb::kD,          mb::kDInner  },   // L0 out_proj.weight    [128,256]
    { 19, mb::kDff,        mb::kD       },   // L0 gate_proj.weight   [256,128]
    { 20, mb::kDff,        mb::kD       },   // L0 up_proj.weight     [256,128]
    { 21, mb::kD,          mb::kDff     },   // L0 down_proj.weight   [128,256]
    { 29, 2 * mb::kDInner, mb::kD       },   // L1 in_proj.weight
    { 30, mb::kXProj,      mb::kDInner  },   // L1 x_proj.weight
    { 31, mb::kNHeads,     mb::kDtRank  },   // L1 dt_proj.weight
    { 37, mb::kD,          mb::kDInner  },   // L1 out_proj.weight
    { 39, mb::kDff,        mb::kD       },   // L1 gate_proj.weight
    { 40, mb::kDff,        mb::kD       },   // L1 up_proj.weight
    { 41, mb::kD,          mb::kDff     },   // L1 down_proj.weight
    { 43, mb::kPHead,      mb::kD       },   // out.weight            [97,128]
};
__device__ __forceinline__ bool mb_is_muon_2d(int t) {
    #pragma unroll
    for (int mi = 0; mi < kMbNumMuon2D; ++mi) if (kMbMuon2D[mi].tidx == t) return true;
    return false;
}
// Largest 2D weight (numel) + largest #rows over the table — sizes the per-matrix
// NS scratch. x_proj [336,256]=86016 is the largest numel; in_proj rows=512 is the
// largest #rows (A=XXᵀ is rows×rows).
constexpr int kMbMuonMaxNumel = mb::kXProj * mb::kDInner;   // 336*256 = 86016
constexpr int kMbMuonMaxRows  = 2 * mb::kDInner;            // 512 (in_proj rows)

}  // namespace mbtc

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGE_MAMBA_TC_CUH_
