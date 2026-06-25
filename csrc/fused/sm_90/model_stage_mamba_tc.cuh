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
#include "csrc/fused/sm_90/parallel_config.cuh"       // par::ParConfig / par::SingleGPU (TP shard table default arg — EDIT M)
#include "csrc/fused/sm_90/tp_transport.cuh"          // tp:: transport seam + tp_allreduce_sum_fixed_order (EDIT M)

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

// ── Layer-streaming HBM scratch helpers: DEFINED in model_stage_mamba3.cuh (at
//    sg::fused::sm90 scope, since the fwd/bwd sample fns there reference them and
//    that header is parsed first). Re-export the workspace-sizing helpers into the
//    mbtc:: namespace so fused_mamba_megakernel.cuh's mb_tc_workspace_floats keeps
//    its mbtc:: spelling. ──
using ::sg::fused::sm90::MbActsHbm;
using ::sg::fused::sm90::mb_acts_stride_floats;
using ::sg::fused::sm90::mb_acts_perlayer_floats;
using ::sg::fused::sm90::mb_acts_transient_floats;
using ::sg::fused::sm90::mb_acts_bind;

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

// ════════════════════════════════════════════════════════════════════════════
//  MAMBA-3 TENSOR-PARALLEL (Megatron) SHARD — the analogue of the decoder's
//  kDecTpShard / tp_layer.cuh geometry, scoped to the SCALAR Mamba path.
//
//  WHY A NEW TABLE (NOT a reuse of tp_layer.cuh): tp_layer.cuh's partial-GEMM
//  helpers are built on the decoder wgmma tile engine (dectc::dectc_gemm_*,
//  dec:: dims) — they do NOT apply to the scan-dominated scalar Mamba path,
//  which projects with the per-(s,o)-owner-computes mb_linear, not wgmma tiles.
//  So the Mamba TP reuses ONLY the transport-agnostic primitives from
//  tp_transport.cuh (LoopbackTransport/NvshmemTransport, the fixed-order
//  ascending-pe reduce, make_transport_from_comm) + parallel_config.cuh
//  (ParConfig/CommCtx). The sharded partial here is a SCALAR row-parallel
//  mb_linear writing fp32 directly into the symmetric slot.
//
//  THE MEGATRON SPLIT ON MAMBA'S 45-TENSOR LAYOUT (row-major [Nout,Kin],
//  named_parameters() order; per-layer 20-block at flat 2+20*li; block-offsets
//  r∈{7=in_proj,8=x_proj,9=dt_proj_w,15=out_proj,17=gate,18=up,19=down}):
//
//   weight (block r)     split          rank owns                    comm
//   ------------------------------------------------------------------------
//   in_proj  (7)         COLUMN         rows [r·2di/P,(r+1)·2di/P)    none (fwd)
//   out_proj (15)        ROW            cols [r·di/P,(r+1)·di/P)      fwd all-reduce ①
//   gate     (17)        COLUMN         rows [r·dff/P,(r+1)·dff/P)    none (fwd)
//   up       (18)        COLUMN         same rows                     none
//   down     (19)        ROW            cols [r·dff/P,(r+1)·dff/P)    fwd all-reduce ②
//   x_proj/dt_proj/scan internals + B/C norms + biases + D + A_log + the
//   RMSNorms + tok/pos/head + norm           REPLICATED               none
//
//  SCOPING NOTE (the Mamba-specific honest boundary — analogue of the decoder's
//  "attention head-shard is the one model-specific touch"): the decoder TP
//  localizes attention to H/P heads on the column-parallel in_proj path. Mamba
//  has NO standard attention, so that touch is precisely SCOPED OUT. The Mamba
//  SSM (x_proj→dt/A/lam/theta/B/C→selective scan) consumes the FULL d_inner
//  x_main with HEAD-SHARED (theta/B/C over Nc) and per-head (dt/A/lam) terms
//  that do NOT decompose into independent per-rank channel shards the way the
//  decoder's per-head attention does (theta/B/C are shared across heads). So
//  the Mamba TP shards the PROJECTION recombination boundaries (the row-parallel
//  out_proj/down all-reduces + their column-parallel dX conjugates) — exactly
//  the decoder's 4 reduce points minus the head localization — and keeps the
//  SSM math replicated. A full head-sharded selective scan (n_heads/P per rank,
//  head-local theta/B/C) is the documented deep follow-up, NOT this mechanical
//  mirror. The 4 reduce points below are the faithful, loopback-checkable core.
// ════════════════════════════════════════════════════════════════════════════
enum class MbTpSplit : int8_t {
    Replicated = 0,   // full tensor on every rank (grads bit-identical, no comm)
    Col        = 1,   // split along Nout (output rows) — column-parallel
    Row        = 2,   // split along Kin  (input cols)  — row-parallel (+ all-reduce)
};
struct MbTpTensorShard { int tidx; MbTpSplit split; };

// The 45-tensor Mamba shard table (mb_bind named order). Per-layer 20-block at
// flat 2+20*li; only {in_proj(+7)=Col, out_proj(+15)=Row, gate(+17)=Col,
// up(+18)=Col, down(+19)=Row} are sharded; everything else replicated.
__device__ __forceinline__ MbTpSplit mb_tp_split_of(int t) {
    if (t < 2) return MbTpSplit::Replicated;                       // tok, pos
    if (t >= 2 + 20 * mb::kLayers) return MbTpSplit::Replicated;   // norm, head w/b
    const int r = (t - 2) % 20;
    if (r == 7)  return MbTpSplit::Col;   // in_proj  [2*d_inner, d]
    if (r == 15) return MbTpSplit::Row;   // out_proj [d, d_inner]
    if (r == 17) return MbTpSplit::Col;   // gate     [d_ff, d]
    if (r == 18) return MbTpSplit::Col;   // up       [d_ff, d]
    if (r == 19) return MbTpSplit::Row;   // down     [d, d_ff]
    return MbTpSplit::Replicated;
}

// Contiguous own-range along a split extent (caller asserts extent % P == 0).
__host__ __device__ __forceinline__ void mb_tp_own_range(int extent, int P, int r,
                                                         int* lo, int* hi) {
    const int per = extent / P;
    *lo = r * per; *hi = *lo + per;
}

// ── SCALAR row-parallel forward partial (the Mamba analogue of
//    tp_rowparallel_fwd_partial_tile, but owner-computes not wgmma). Each PE
//    computes Ypart[s, o] = Σ_{k in own col-shard} Xown[s,k]·Wshard[o,k] over
//    its [Nout, Kin/P] dense col-shard and writes the fp32 partial DIRECTLY into
//    this PE's symmetric slot (the publish). Caller then rendezvous + fixed-order
//    ascending-pe reduce → the full Σ_pe Ypart (all-reduce point ①/②). No bias
//    here — the replicated bias is added by the caller AFTER the reduce (matching
//    the decoder: out_b/ff2_b are post-reduce adds). ──
__device__ __forceinline__ void mb_tp_rowparallel_fwd_partial(
        float* __restrict__ slot,                 // tr.local(slot_off): [kSeq*Nout] fp32
        const float* __restrict__ Xown, int ldX,  // [kSeq, Kin_local] (rank's own input shard)
        const float* __restrict__ Wshard,         // [Nout, Kin_local] dense row-major col-shard
        int Kin_local, int Nout) {
    const int total = mb::kSeq * Nout;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int s = idx / Nout, o = idx % Nout;
        const float* xr = Xown + (int64_t)s * ldX;
        const float* Wr = Wshard + (int64_t)o * Kin_local;
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < Kin_local; ++k) acc += xr[k] * Wr[k];
        slot[(int64_t)s * Nout + o] = acc;
    }
    __syncthreads();
}

// ── SCALAR column-parallel backward dX partial (the conjugate comm point ①'/②').
//    dXpart[s,i] = Σ_{o in own row-shard} dYown[s,o]·Wshard[o,i], the rank's
//    contribution to the FULL [kSeq, Kin] dX; published to the symmetric slot,
//    caller rendezvous + reduce ⇒ dX = Σ_pe dXpart. Wshard is the rank's
//    [Nout_local, Kin] dense row-shard. ──
__device__ __forceinline__ void mb_tp_colparallel_dx_partial(
        float* __restrict__ slot,                 // tr.local(slot_off): [kSeq*Kin] fp32
        const float* __restrict__ dYown, int ldY, // [kSeq, Nout_local]
        const float* __restrict__ Wshard,         // [Nout_local, Kin] dense row-major row-shard
        int Kin, int Nout_local) {
    const int total = mb::kSeq * Kin;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int s = idx / Kin, i = idx % Kin;
        const float* dyr = dYown + (int64_t)s * ldY;
        float acc = 0.0f;
        for (int o = 0; o < Nout_local; ++o) acc += dyr[o] * Wshard[(int64_t)o * Kin + i];
        slot[(int64_t)s * Kin + i] = acc;
    }
    __syncthreads();
}

// ── Symmetric-slot budget for one Mamba TP reduce, in floats. The largest
//    reduced payload per sample is max(kSeq·d [out_proj/down fwd reduce, and the
//    in_proj/gate-up dX reduce → kSeq·d]). All four reduces are [kSeq, d]. The
//    canonical heap plan (one publish + one reduced slot per CTA-in-PE):
//      slot 0: partial  [kSeq · d]   (publish target)
//      slot 1: reduced  [kSeq · d]   (reduce output)
//    The persistent kernel runs ONE sample per CTA at a time ⇒ heap stride =
//    ctas_per_pe · 2 · kSeq · d floats. (Mirrors tp_layer.cuh::tp_tile_slot_floats
//    / tp_heap_stride_floats, with the Mamba sample shape kSeq·d in place of the
//    decoder tile shape kTileM·d.) ──
__host__ __device__ __forceinline__ int64_t mb_tp_slot_floats() {
    return (int64_t)mb::kSeq * mb::kD;
}
__host__ __device__ __forceinline__ int64_t mb_tp_heap_stride_floats(int ctas_per_pe) {
    return (int64_t)ctas_per_pe * 2 * mb_tp_slot_floats();
}

}  // namespace mbtc

// ════════════════════════════════════════════════════════════════════════════
//  TENSOR-PARALLEL Mamba sample fwd/bwd (EDIT M-D) — the analogue of the
//  decoder's dectc_forward_tile_impl / dectc_backward_tile_impl with the four
//  if-constexpr(Par::kTPComm) reduce points, scoped to the scalar Mamba path.
//
//  SHAPE: mb_forward_sample_tp<Par,Transport> / mb_backward_sample_tp<Par,
//  Transport> share the EXACT body of mb_forward_sample / mb_backward_sample
//  (model_stage_mamba3.cuh) on the SingleGPU (!kTPComm) path — they CALL those
//  functions verbatim, so SingleGPU is byte-identical by construction (the
//  wrappers add no code on the folded path). On the kTPComm path they run a
//  TP-aware mixer/swiglu where the row-parallel out_proj/down forward produce a
//  partial → publish → fixed-order ascending-pe reduce (①/②), and the
//  column-parallel in_proj/gate-up backward dX is a partial → reduce (①'/②').
//
//  GRID-LOCKSTEP CONTRACT (mirrors tp_kernel.md §1 / the decoder C.3): these are
//  called ONLY from the kTPComm grid-lockstep-over-samples P1 (EDIT M-C), where
//  every CTA on a GPU cooperates on the SAME sample each round so tr.rendezvous
//  (bar) at the 4 reduce points is reached a GRID-UNIFORM number of times. The
//  `active` flag lets an empty round still reach every rendezvous (publishing a
//  zero partial / skipping the math) so the arrival count stays uniform — the
//  §1 deadlock fix. `slot_pub`/`slot_red` are this CTA's lanes in the symmetric
//  heap (publish + reduced).
//
//  ⚠ SSM REPLICATION (the honest scope boundary, see mbtc:: shard-table NOTE):
//  the selective scan + x_proj/dt internals are REPLICATED on every rank (the
//  full d_inner x_main is reconstructed post-in_proj-reduce-free since in_proj is
//  column-parallel and the SSM is head-shared). So the kTPComm mixer below runs
//  the unsharded mb_mixer_fwd for the SSM body and applies the TP reduce ONLY at
//  the out_proj/down recombination — exactly the decoder's reduce-point set
//  minus the head localization that Mamba's head-shared SSM cannot take. The
//  loopback transport makes this honestly checkable: with P virtual PEs each
//  holding the full (replicated) weights, the reduce of P identical partials is
//  P× the single value — which is why a TRUE multi-rank shard needs the sharded
//  out_w/down_w col-shards the launcher's per-rank `params` provides (EDIT M-E,
//  E.3 upstream host plan). The SCAFFOLD (template/transport/reduce/sym-heap/
//  dispatch) is the faithful decoder mirror; the per-rank weight shard is the
//  upstream host plan, out of this kernel track (decoder E.3 parallel).
// ════════════════════════════════════════════════════════════════════════════

// Forward: SingleGPU forwards verbatim (byte-identical). kTPComm runs the mixer
// with out_proj row-parallel reduce + swiglu with down row-parallel reduce.
template <class Par = ::sg::fused::par::SingleGPU,
          class Transport = ::sg::fused::sm90::tp::LoopbackTransport>
__device__ inline float mb_forward_sample_tp(
        const MambaWeights& w, const int* tokens_s, int target, bool active,
        MambaSampleSmem* sm, const Transport& tr, const ::sg::fused::GridBarrier& bar,
        int64_t slot_pub, int64_t slot_red) {
    if constexpr (!Par::kTPComm) {
        (void)active; (void)tr; (void)bar; (void)slot_pub; (void)slot_red;
        return mb_forward_sample(w, tokens_s, target, sm);
    } else {
        // SCOPE: the kTPComm path indexes sm->layer_in[li]/sm->act[li] for ALL layers
        // (no streaming), so it is incompatible with the streamed (flagship) smem. TP at
        // flagship-streamed scale would need this body streamed too (out of this track).
        // (Par-dependent so it only fires when THIS branch is actually instantiated.)
        static_assert(!Par::kTPComm || !kMbStreamSmem,
                      "mb_forward_sample_tp: kTPComm + layer-streamed smem not supported "
                      "(the TP body caches all layers; stream it before TP-at-flagship).");
        // ── kTPComm forward. The SSM body is replicated (unsharded mb_* device
        //    fns); the two ROW-parallel projections (mixer out_proj, swiglu down)
        //    publish a partial → rendezvous → fixed-order reduce. The replicated
        //    bias is folded post-reduce. The recombination is the decoder ①/② set.
        const int P = tr.n_pes();
        float nll_acc = 0.0f;
        // Embedding + positional -> layer_in[0] (replicated, only when active).
        if (active) {
            for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
                const int s = idx / mb::kD, j = idx % mb::kD;
                sm->layer_in[0][s][j] = w.tok[(int64_t)tokens_s[s] * mb::kD + j]
                                      + w.pos[(int64_t)s * mb::kD + j];
            }
            __syncthreads();
        }
        for (int li = 0; li < mb::kLayers; ++li) {
            const MambaWeights::Layer& L = w.layer[li];
            MambaSampleSmem::LayerAct* a = &sm->act[li];
            float* hin = &sm->layer_in[li][0][0];
            // --- mixer sub-block (SSM replicated; out_proj ROW-parallel reduce ①) ---
            if (active) {
                mb_rmsnorm_fwd(hin, L.mixn_w, &sm->dr[0][0], &sm->adj_c[0][0], &a->mixn_r[0],
                               sm->red, mb::kD, mb::kD);   // xn -> sm->dr
                // mixer through y_gated (everything before out_proj); y_gated -> sm->adj_a.
                mb_mixer_fwd_pre_outproj(L, &sm->dr[0][0], a, sm);
            }
            // out_proj ROW-parallel: out_w is the rank's [d, d_inner/P] col-shard;
            // y_gated (sm->adj_a) is the rank's own [kSeq, d_inner/P]. Publish [kSeq,d]
            // partial -> reduce -> mix_out (sm->adj_b, kD-strided). (① of the shard table.)
            const int Kloc = mb::kDInner / P;
            if (active) {
                mbtc::mb_tp_rowparallel_fwd_partial(
                    tr.local(slot_pub), &sm->adj_a[0][0], mb::kDInner, L.out_w, Kloc, mb::kD);
            }
            tr.rendezvous(bar);
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, &sm->adj_b[0][0], (int64_t)mb::kSeq * mb::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);
            if (active) {
                float* mo = &sm->adj_b[0][0];
                for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
                    const int s = idx / mb::kD, j = idx % mb::kD;
                    a->h1[s][j] = hin[s * mb::kD + j] + mo[s * mb::kD + j];   // h1 = x + mixer_out
                }
                __syncthreads();
                // --- SwiGLU sub-block (gate/up COLUMN replicated; down ROW reduce ②) ---
                mb_rmsnorm_fwd(&a->h1[0][0], L.mlpn_w, &sm->dr[0][0], &sm->adj_c[0][0], &a->mlpn_r[0],
                               sm->red, mb::kD, mb::kD);   // h1n -> sm->dr
                // swiglu through prod (silu(gate)*up) -> sm->wff_a (down_proj input).
                mb_swiglu_fwd_pre_down(L, &sm->dr[0][0], a, sm);
            }
            // down ROW-parallel: down_w is the rank's [d, d_ff/P] col-shard; prod
            // (sm->wff_a) is the rank's own [kSeq, d_ff/P]. Publish -> reduce -> mlp_out. (②)
            const int Kloc2 = mb::kDff / P;
            if (active) {
                mbtc::mb_tp_rowparallel_fwd_partial(
                    tr.local(slot_pub), &sm->wff_a[0][0], mb::kDff, L.down_w, Kloc2, mb::kD);
            }
            tr.rendezvous(bar);
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, &sm->adj_b[0][0], (int64_t)mb::kSeq * mb::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);
            if (active) {
                float* mo = &sm->adj_b[0][0];
                float* dst = (li + 1 < mb::kLayers) ? &sm->layer_in[li + 1][0][0] : &sm->final_in[0][0];
                for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
                    const int s = idx / mb::kD, j = idx % mb::kD;
                    dst[s * mb::kD + j] = a->h1[s][j] + mo[s * mb::kD + j];   // h2 = h1 + mlp_out
                }
                __syncthreads();
            }
        }
        // Final RMSNorm + head + NLL (replicated, only when active).
        if (active) {
            const float* hlast = &sm->final_in[mb::kSeq - 1][0];
            float ss = 0.0f;
            for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) { float v = hlast[j]; ss += v * v; }
            float ms = mb_block_sum(ss, sm->red) / (float)mb::kD;
            float r = rsqrtf(ms + mb::kRmsEps);
            if (threadIdx.x == 0) sm->fn_r[mb::kSeq - 1] = r;
            for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
                float xh = hlast[j] * r;
                sm->fn_xhat[mb::kSeq - 1][j] = xh;
                sm->adj_a[0][j] = xh * w.norm_w[j];
            }
            __syncthreads();
            for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) {
                const float* Wr = w.out_w + (int64_t)o * mb::kD;
                float acc = w.out_b[o];
                #pragma unroll 4
                for (int k = 0; k < mb::kD; ++k) acc += sm->adj_a[0][k] * Wr[k];
                sm->logits[o] = acc;
            }
            __syncthreads();
            float lmax = -CUDART_INF_F;
            for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) lmax = fmaxf(lmax, sm->logits[o]);
            lmax = mb_block_max(lmax, sm->red);
            float es = 0.0f;
            for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) es += __expf(sm->logits[o] - lmax);
            es = mb_block_sum(es, sm->red);
            float logz = lmax + __logf(es);
            nll_acc = logz - sm->logits[target];
        }
        return nll_acc;
    }
}

// Backward: SingleGPU forwards verbatim (byte-identical). kTPComm runs the mixer
// with in_proj column-parallel dX reduce + swiglu with gate/up column dX reduce.
template <class Par = ::sg::fused::par::SingleGPU,
          class Transport = ::sg::fused::sm90::tp::LoopbackTransport>
__device__ inline void mb_backward_sample_tp(
        const MambaWeights& w, const MambaGrad& g, const int* tokens_s, int target,
        int B, bool active, MambaSampleSmem* sm, const Transport& tr,
        const ::sg::fused::GridBarrier& bar, int64_t slot_pub, int64_t slot_red) {
    if constexpr (!Par::kTPComm) {
        (void)active; (void)tr; (void)bar; (void)slot_pub; (void)slot_red;
        mb_backward_sample(w, g, tokens_s, target, B, sm);
    } else {
        // ── kTPComm backward. The replicated head/final-norm + the SSM body bwd run
        //    the unsharded mb_* device fns; the two COLUMN-parallel projections
        //    (in_proj, gate+up) have a PARTIAL dX → publish → reduce (①'/②').
        const int P = tr.n_pes();
        if (active) {
            // CE bwd + head + final RMSNorm bwd (replicated) → sm->dh (last pos nonzero).
            float lmax = -CUDART_INF_F;
            for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) lmax = fmaxf(lmax, sm->logits[o]);
            lmax = mb_block_max(lmax, sm->red);
            float es = 0.0f;
            for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) es += __expf(sm->logits[o] - lmax);
            es = mb_block_sum(es, sm->red);
            float inv_es = 1.0f / es;
            for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) {
                float smo = __expf(sm->logits[o] - lmax) * inv_es;
                sm->logits[o] = (smo - ((o == target) ? 1.0f : 0.0f)) / (float)B;
            }
            __syncthreads();
            const float* xh = &sm->fn_xhat[mb::kSeq - 1][0];
            for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) {
                float dl = sm->logits[o];
                g.out_b[o] += dl;
                float* gwrow = g.out_w + (int64_t)o * mb::kD;
                #pragma unroll 4
                for (int j = 0; j < mb::kD; ++j) gwrow[j] += dl * (xh[j] * w.norm_w[j]);
            }
            __syncthreads();
            for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
                float acc = 0.0f;
                for (int o = 0; o < mb::kPHead; ++o) acc += sm->logits[o] * w.out_w[(int64_t)o * mb::kD + j];
                sm->dr[0][j] = acc;
            }
            __syncthreads();
            for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) g.norm_w[j] += sm->dr[0][j] * xh[j];
            __syncthreads();
            {
                float sdax = 0.0f;
                for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
                    float dxhat = sm->dr[0][j] * w.norm_w[j]; sdax += dxhat * xh[j];
                }
                sdax = mb_block_sum(sdax, sm->red);
                float corr = sdax / (float)mb::kD;
                float rs = sm->fn_r[mb::kSeq - 1];
                for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
                    float dxhat = sm->dr[0][j] * w.norm_w[j];
                    for (int s = 0; s < mb::kSeq; ++s) sm->dh[s][j] = 0.0f;
                    sm->dh[mb::kSeq - 1][j] = rs * (dxhat - xh[j] * corr);
                }
                __syncthreads();
            }
        }
        for (int li = mb::kLayers - 1; li >= 0; --li) {
            const MambaWeights::Layer& L = w.layer[li];
            const MambaGrad::Layer& G = g.layer[li];
            MambaSampleSmem::LayerAct* a = &sm->act[li];
            float* hin = &sm->layer_in[li][0][0];
            // h2 = h1 + mlp(mlp_norm(h1)). dh2 = sm->dh.
            if (active) {
                for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
                    const int s = idx / mb::kD, j = idx % mb::kD;
                    sm->dr[s][j] = (a->h1[s][j] * a->mlpn_r[s]) * L.mlpn_w[j];   // h1n
                }
                __syncthreads();
                // swiglu bwd: dW(down/gate/up) accumulate + dX = dx_gate + dx_up into
                // sm->fn_xhat. On the COLUMN-parallel shard each rank owns dff/P rows of
                // gate/up, so its dX is a PARTIAL Σ over its own rows; publish it to the
                // symmetric slot for the fixed-order reduce (②' of the shard table). In the
                // loopback (replicated weights) this publishes the full dX (the documented
                // P× loopback artifact — see the SSM-REPLICATION NOTE).
                mb_swiglu_bwd(L, G, &sm->dr[0][0], a, &sm->dh[0][0], &sm->fn_xhat[0][0], sm);
                for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x)
                    tr.local(slot_pub)[idx] = sm->fn_xhat[idx / mb::kD][idx % mb::kD];
                __syncthreads();
            }
            // gate+up COLUMN-parallel dX reduce (②'): the partial dh1n is in tr.local(slot_pub).
            // Reduce → sm->fn_xhat.
            tr.rendezvous(bar);
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, &sm->fn_xhat[0][0], (int64_t)mb::kSeq * mb::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);
            if (active) {
                mb_rmsnorm_bwd_rawx(&sm->fn_xhat[0][0], &a->h1[0][0], &a->mlpn_r[0], L.mlpn_w,
                                    &sm->dr[0][0], G.mlpn_w, sm->red, mb::kD, mb::kD);
                for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
                    const int s = idx / mb::kD, j = idx % mb::kD;
                    sm->dh[s][j] += sm->dr[s][j];   // dh1 = dh2 + mlp_norm path
                }
                __syncthreads();
                // mixer bwd through the SSM body to the in_proj dxz; dxn (mixer_norm out
                // grad) is the COLUMN-parallel in_proj dX partial; reduced below (①').
                for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
                    const int s = idx / mb::kD, j = idx % mb::kD;
                    sm->dr[s][j] = (hin[s * mb::kD + j] * a->mixn_r[s]) * L.mixn_w[j];   // xn
                }
                __syncthreads();
                // mixer bwd: dW(out/x_proj/dt/in/scan/...) accumulate + dX = in_proj dX
                // into sm->fn_xhat. in_proj is COLUMN-parallel, so its dX is a PARTIAL Σ
                // over the rank's own 2*d_inner/P rows; publish for the reduce (①').
                mb_mixer_bwd(L, G, &sm->dr[0][0], a, &sm->dh[0][0], &sm->fn_xhat[0][0], sm);
                for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x)
                    tr.local(slot_pub)[idx] = sm->fn_xhat[idx / mb::kD][idx % mb::kD];
                __syncthreads();
            }
            // in_proj COLUMN-parallel dX reduce (①'): partial dxn in tr.local(slot_pub).
            tr.rendezvous(bar);
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, &sm->fn_xhat[0][0], (int64_t)mb::kSeq * mb::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);
            if (active) {
                mb_rmsnorm_bwd_rawx(&sm->fn_xhat[0][0], hin, &a->mixn_r[0], L.mixn_w,
                                    &sm->dr[0][0], G.mixn_w, sm->red, mb::kD, mb::kD);
                for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
                    const int s = idx / mb::kD, j = idx % mb::kD;
                    sm->dh[s][j] += sm->dr[s][j];   // dx = dh1 + mixer_norm path
                }
                __syncthreads();
            }
        }
        // embedding backward (replicated).
        if (active) {
            for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
                #pragma unroll
                for (int s = 0; s < mb::kSeq; ++s) {
                    float d = sm->dh[s][j];
                    g.tok[(int64_t)tokens_s[s] * mb::kD + j] += d;
                    g.pos[(int64_t)s * mb::kD + j] += d;
                }
            }
            __syncthreads();
        }
        (void)P; (void)slot_red;
    }
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGE_MAMBA_TC_CUH_
