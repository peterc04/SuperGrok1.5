#ifndef SG_FUSED_SM90_TP_LAYER_CUH_
#define SG_FUSED_SM90_TP_LAYER_CUH_
// ============================================================================
// csrc/fused/sm_90/tp_layer.cuh — TENSOR-PARALLEL math for the L3-TC decoder
// megakernel: the Megatron column/row split geometry, the sharded wgmma tile
// GEMMs, and the two in-step all-reduce points.
//
// IMPLEMENTS: /workspace/.parallelism_design.md §5.1 (where the all-reduce
// lands — col-parallel in_proj/ff0, row-parallel out_proj/ff2, TWO all-reduces
// per layer) + §5.2 (fixed-order in-kernel reduce, via tp_transport.cuh).
// Authored 1-GPU per design §7 (task #25 phase-2, Lane C). The TP MATH here is
// the durable deliverable; the transport behind it is the swap point
// (tp_transport.cuh — loopback now, NVSHMEM at the 8× window).
//
// ──────────────────────────────────────────────────────────────────────────
//  THE MEGATRON SPLIT ON THIS KERNEL'S WEIGHT LAYOUT (all row-major [Nout,Kin],
//  matching DecWeights / decoder_layout.cuh):
//
//   weight (tidx)        split          rank r owns                 comm
//   ------------------------------------------------------------------------
//   in_proj  (2,14)      COLUMN(QKV)    rows {q,k,v}×[r·d/P,(r+1)·d/P)   none (fwd)
//   in_proj_b(3,15)      COLUMN(QKV)    same 3-block rows                none
//   out_proj (4,16)      ROW            cols [r·d/P,(r+1)·d/P)       fwd all-reduce ①
//   out_b    (5,17)      REPLICATED     (added after reduce ①, identical on all)
//   ff.0     (10,22)     COLUMN         rows [r·dff/P,(r+1)·dff/P)       none (fwd)
//   ff.0_b   (11,23)     COLUMN         same rows                        none
//   ff.2     (12,24)     ROW            cols [r·dff/P,(r+1)·dff/P)   fwd all-reduce ②
//   ff.2_b   (13,25)     REPLICATED     (added after reduce ②)
//   tok/pos  (0,1)       REPLICATED     —                                 none*
//   LN γ/β   (6..9,18..21,26,27) REPLICATED (LN runs on full-width, post-reduce)
//   head w/b (28,29)     REPLICATED     (V=99 — vocab-parallel head is the
//                                        documented future extension, not built)
//
//   * replicated tensors receive BIT-IDENTICAL grads on every TP rank: their
//     producing adjoints sit downstream of fixed-order all-reduces (dh0 is the
//     reduced in_proj dX; LN/head inputs are full-width post-reduce activations)
//     — so no grad comm is needed for them and determinism is structural.
//
//  THE QKV 3-BLOCK SHARD: in_proj packs [q(d)|k(d)|v(d)] along Nout=3d, heads
//  inside each block. Rank r owns heads [r·H/P,(r+1)·H/P) ⇒ for EACH of q,k,v
//  the row range [r·d/P,(r+1)·d/P). The rank's DENSE shard buffer is the three
//  blocks concatenated: [q_own | k_own | v_own], 3·d/P rows of Kin=d — so the
//  rank-local attention reads its qkv exactly like the unsharded kernel reads
//  the full qkv, just with H_loc = H/P heads (attention is per-head ⇒ the
//  unsharded per-head math runs UNCHANGED on the local heads).
//
//  BACKWARD (the conjugate points, design §5.1): the all-reduce moves to the
//  dX of every COLUMN-parallel linear —
//      dX(in_proj)  = Σ_r dqkv_own @ in_w_own   → bwd all-reduce ①' (→ dh0/residual)
//      dX(ff.0)     = Σ_r dff0_own @ ff0_w_own  → bwd all-reduce ②' (→ dx1)
//  Row-parallel dX is comm-free (dX_own = dY @ W_own is already the rank's own
//  K-shard). dW/db NEVER need comm: every rank's dW_own is an EXACT row-slice
//  (column-parallel) or col-slice (row-parallel) of the full dW — the loopback
//  test asserts this slice-exactness bitwise.
//
//  WHERE THE REDUCES SIT IN THE FUSED STEP (the §5.2 insertion map for the
//  megakernel builder; the two marked points in dectc_forward_tile).
//  NB: line numbers below are ANCHORED TO THE CALL-SITE COMMENT TEXT (stable),
//  not absolute lines — the production header grows as the kernel track edits
//  it (was ~1100 lines at the Lane-C authoring, 1946 as of 2026-06-16). Grep the
//  quoted comment if a number has drifted. Lines verified current 2026-06-16:
//    ① model_stage_decoder_tc.cuh ~1085-1087 — the `a = X_ctx @ out_w^T (+ out_b)`
//      out_proj GEMM into sc.work (the `// a = X_ctx @ out_w^T` comment), residual
//      fold at ~1093: publish the [nrows,d] partial to the TP slot, rendezvous,
//      fixed-order-reduce into sc.work, rendezvous; THEN the r1 residual+bias fold
//      proceeds unchanged on the reduced value.
//    ② model_stage_decoder_tc.cuh ~1116-1118 — the `ff2 = X_gact @ ff2_w^T (+ ff2_b)`
//      GEMM into sc.work (the `// ff2 = X_gact @ ff2_w^T` comment), r2 fold at ~1124:
//      same publish/reduce on the [nrows,d] partial before the r2 fold.
//    ①' / ②' mirror in dectc_backward_tile: ①' at ~1421-1423 (the in_proj dX
//      `dx_in_attn = dqkv @ in_w` GEMM, residual fold `sc.dh += sc.work` at ~1427)
//      and ②' at ~1392-1395 (the ff0 dX `dx1 += dff0 @ ff0_w` GEMM).
//      NOTE the rendezvous is GRID-WIDE: with TP inside the
//      persistent kernel every CTA participates (the GridBarrier IS the
//      rendezvous fabric, design §5.2) — i.e. P1's "barrier-free within a
//      tile" relaxes to "barrier at the 4 reduce points" on the TP>1 path
//      ONLY (if constexpr (Par::kTPComm)); the SingleGPU instantiation keeps
//      today's barrier-free P1 byte-identical (design §1.2).
//
//  THIS HEADER deliberately does NOT edit the production stage header — the
//  per-tile TP block functions below are the exact bodies the builder inserts
//  behind `if constexpr (Par::kTPComm)` at the marked lines (that edit is a
//  tracked-file change → staged as a .phase2 patch when the transport choice
//  lands). They are honestly testable TODAY via tests/hw/tp_loopback_binding.cu,
//  which drives them with the SAME wgmma tile GEMMs the production kernel uses.
// ──────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "csrc/fused/sm_90/model_stage_decoder_tc.cuh"   // dectc_gemm_* (the wgmma tiles), dec:: dims
#include "csrc/fused/sm_90/tp_transport.cuh"             // the transport seam + fixed-order reduce
#include "csrc/fused/sm_90/parallel_config.cuh"          // ParConfig (kTPComm gate)

namespace sg { namespace fused { namespace sm90 { namespace tp {

// ─────────────────────────────────────────────────────────────────────────
//  Shard geometry. Contiguous block ownership along the split dimension; the
//  TP degree must divide the split extents AND keep heads whole (static checks
//  where compile-time, runtime asserts in the host plan otherwise).
// ─────────────────────────────────────────────────────────────────────────
enum class TpSplit : int8_t {
    Replicated = 0,   // full tensor on every rank (grads bit-identical, no comm)
    Col        = 1,   // split along Nout (output features) — rank owns a row block
    Row        = 2,   // split along Kin (input features)   — rank owns a col block
    ColQKV     = 3,   // in_proj 3-block column split (q|k|v each split by heads)
};

struct TpTensorShard { int tidx; TpSplit split; };

// The 30-tensor decoder shard table (dec_layout named_parameters() order — the
// single source decoder_layout.cuh mirrors). See the header table.
constexpr int kDecTpNumTensors = 30;
__device__ __constant__ TpTensorShard kDecTpShard[kDecTpNumTensors] = {
    { 0, TpSplit::Replicated},  // tok.weight
    { 1, TpSplit::Replicated},  // pos.weight
    { 2, TpSplit::ColQKV    },  // L0 in_proj_weight
    { 3, TpSplit::ColQKV    },  // L0 in_proj_bias
    { 4, TpSplit::Row       },  // L0 out_proj.weight
    { 5, TpSplit::Replicated},  // L0 out_proj.bias  (post-reduce add)
    { 6, TpSplit::Replicated},  // L0 n1.weight
    { 7, TpSplit::Replicated},  // L0 n1.bias
    { 8, TpSplit::Replicated},  // L0 n2.weight
    { 9, TpSplit::Replicated},  // L0 n2.bias
    {10, TpSplit::Col       },  // L0 ff.0.weight
    {11, TpSplit::Col       },  // L0 ff.0.bias
    {12, TpSplit::Row       },  // L0 ff.2.weight
    {13, TpSplit::Replicated},  // L0 ff.2.bias  (post-reduce add)
    {14, TpSplit::ColQKV    },  // L1 in_proj_weight
    {15, TpSplit::ColQKV    },  // L1 in_proj_bias
    {16, TpSplit::Row       },  // L1 out_proj.weight
    {17, TpSplit::Replicated},  // L1 out_proj.bias
    {18, TpSplit::Replicated},  // L1 n1.weight
    {19, TpSplit::Replicated},  // L1 n1.bias
    {20, TpSplit::Replicated},  // L1 n2.weight
    {21, TpSplit::Replicated},  // L1 n2.bias
    {22, TpSplit::Col       },  // L1 ff.0.weight
    {23, TpSplit::Col       },  // L1 ff.0.bias
    {24, TpSplit::Row       },  // L1 ff.2.weight
    {25, TpSplit::Replicated},  // L1 ff.2.bias
    {26, TpSplit::Replicated},  // norm.weight
    {27, TpSplit::Replicated},  // norm.bias
    {28, TpSplit::Replicated},  // out.weight  (head — vocab-parallel = future)
    {29, TpSplit::Replicated},  // out.bias
};

// Contiguous own-range along a split extent. Requires P | extent (TP degrees
// are powers of two and d/dff/3d at every ladder width are multiples of 8·64;
// the host plan asserts before launch — no silent remainder handling).
__host__ __device__ __forceinline__ void tp_own_range(
        int extent, int P, int r, int* lo, int* hi) {
    const int per = extent / P;     // caller asserts extent % P == 0
    *lo = r * per;
    *hi = *lo + per;
}

// Heads-per-rank for the QKV/head-aligned splits. Requires P | H so each rank's
// column block is whole heads (the local attention precondition).
__host__ __device__ __forceinline__ int tp_heads_per_rank(int H, int P) {
    return H / P;                   // caller asserts H % P == 0
}

// ─────────────────────────────────────────────────────────────────────────
//  Host-side dense shard packing maps (how a rank STORES its shard). The rank
//  keeps its shard DENSE — [Nout/P, Kin] for Col, [Nout, Kin/P] for Row, and
//  for ColQKV the q|k|v own-blocks concatenated [3·d/P, d] — so every sharded
//  GEMM below reads it exactly like the unsharded kernel reads a full weight
//  (row-major, no strides). These index helpers define the pack ONCE (host
//  packers + tests both call them; no duplicated arithmetic).
//    Col   : dense row i  ↔ full row (own_lo + i)
//    Row   : dense col j  ↔ full col (own_lo + j)
//    ColQKV: dense row i  ↔ full row (block(i)·d + own_lo + i % (d/P)),
//            block(i) = i / (d/P) ∈ {0,1,2} = {q,k,v}
// ─────────────────────────────────────────────────────────────────────────
__host__ __device__ __forceinline__ int tp_col_full_row(int own_lo, int i) {
    return own_lo + i;
}
__host__ __device__ __forceinline__ int tp_row_full_col(int own_lo, int j) {
    return own_lo + j;
}
__host__ __device__ __forceinline__ int tp_colqkv_full_row(int d, int P, int r, int i) {
    const int per = d / P;          // rows per (q|k|v) block on one rank
    const int blk = i / per;        // 0=q, 1=k, 2=v
    const int off = i % per;
    return blk * d + r * per + off;
}

// ─────────────────────────────────────────────────────────────────────────
//  THE PER-TILE TP BLOCK FUNCTIONS (the megakernel-tiling math). All reuse the
//  PRODUCTION wgmma tile GEMMs (dectc_gemm_fwd_f32 / dectc_gemm_dx_f32 — fp32
//  accumulate, kTileM=128-row tiles, identical staging) on the rank's DENSE
//  shard; zero new GEMM math. `Wshard` is fp32 (the params blob dtype; the GEMM
//  converts to bf16 on read exactly like the unsharded path).
//
//  COLUMN-parallel forward:  Y_own[M, Nout/P] = X[M,Kin] @ Wshard^T.
//  Comm-free; the output IS the rank's feature shard.
// ─────────────────────────────────────────────────────────────────────────
template <int N>
__device__ __forceinline__ void tp_colparallel_fwd_tile(
        const __nv_bfloat16* __restrict__ X, const float* __restrict__ Wshard,
        float* __restrict__ Yown, int Kin, int Nout_local,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    dectc::dectc_gemm_fwd_f32<N>(X, Wshard, Yown, Kin, Nout_local, sA, sB);
}

// ROW-parallel forward PARTIAL: Ypart[M, Nout] = X_own[M, Kin/P] @ Wshard^T,
// written DIRECTLY into this PE's symmetric slot (the publish). The caller then
// crosses the rendezvous and runs the fixed-order reduce (all-reduce point ①/②).
template <int N, class Transport>
__device__ __forceinline__ void tp_rowparallel_fwd_partial_tile(
        const Transport& tr, int64_t slot_off,
        const __nv_bfloat16* __restrict__ Xown, const float* __restrict__ Wshard,
        int Kin_local, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    dectc::dectc_gemm_fwd_f32<N>(Xown, Wshard, tr.local(slot_off),
                                 Kin_local, Nout, sA, sB);
}

// ROW-parallel backward dX (comm-free): dXown[M, Kin/P] = dY[M, Nout] @ Wshard.
// dY is the FULL-width adjoint (the row-parallel output was all-reduced, so its
// adjoint is replicated); Wshard is the rank's [Nout, Kin/P] dense col-shard.
template <int N>
__device__ __forceinline__ void tp_rowparallel_dx_tile(
        const __nv_bfloat16* __restrict__ dY, const float* __restrict__ Wshard,
        float* __restrict__ dXown, int Kin_local, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    dectc::dectc_gemm_dx_f32<N>(dY, Wshard, dXown, Kin_local, Nout, sA, sB);
}

// COLUMN-parallel backward dX PARTIAL (the conjugate comm point ①'/②'):
// dXpart[M, Kin] = dYown[M, Nout/P] @ Wshard, published to the symmetric slot;
// caller rendezvous + fixed-order reduce ⇒ dX = Σ_pe dXpart.
template <int N, class Transport>
__device__ __forceinline__ void tp_colparallel_dx_partial_tile(
        const Transport& tr, int64_t slot_off,
        const __nv_bfloat16* __restrict__ dYown, const float* __restrict__ Wshard,
        int Kin, int Nout_local,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    dectc::dectc_gemm_dx_f32<N>(dYown, Wshard, tr.local(slot_off),
                                Kin, Nout_local, sA, sB);
}

// ─────────────────────────────────────────────────────────────────────────
//  dW under TP — the exact-slice property (NO comm, the P2 machinery carries
//  over unchanged). For a COLUMN-parallel weight, the rank's
//      dW_own[Nout/P, Kin] = dY_own^T @ X        (X full/replicated)
//  is bit-for-bit rows [own_lo, own_hi) of the unsharded dW (same ascending-K
//  fp32 contraction over the same T). For a ROW-parallel weight,
//      dW_own[Nout, Kin/P] = dY^T @ X_own        (dY full/replicated)
//  is bit-for-bit cols [own_lo, own_hi). So P2's output-stationary dW spec for
//  a TP rank is the SAME DecDwSpec with (dY, Nout) or (X, Kin) swapped for the
//  rank-local stream/extent — this helper performs that transform. Biases:
//  col-parallel db_own = Σ_K dY_own (the own rows of db); row-parallel /
//  replicated db = Σ_K dY computed identically on every rank.
// ─────────────────────────────────────────────────────────────────────────
__device__ __host__ __forceinline__ void tp_shard_extents(
        TpSplit split, int P, int Nout_full, int Kin_full,
        int* Nout_local, int* Kin_local) {
    switch (split) {
        case TpSplit::Col:
        case TpSplit::ColQKV: *Nout_local = Nout_full / P; *Kin_local = Kin_full;     break;
        case TpSplit::Row:    *Nout_local = Nout_full;     *Kin_local = Kin_full / P; break;
        default:              *Nout_local = Nout_full;     *Kin_local = Kin_full;     break;
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Symmetric-slot budget for one decoder layer's TP comm, in floats, per tile
//  row block of kTileM rows: the largest reduced payload is [kTileM, d] (both
//  fwd reduces and both bwd reduces are [rows, d]). The canonical heap plan
//  (the loopback binding + the future NVSHMEM bootstrap both size with this):
//      slot 0: partial  [kTileM · d]      (publish target)
//      slot 1: reduced  [kTileM · d]      (reduce output, when not fused into
//                                          a local consumer buffer)
//  per CONCURRENT tile in flight per PE. The persistent kernel runs ONE tile
//  per CTA at a time ⇒ heap stride = n_ctas_per_pe · 2 · kTileM · d floats.
// ─────────────────────────────────────────────────────────────────────────
__host__ __device__ __forceinline__ int64_t tp_tile_slot_floats() {
    return (int64_t)dectc::kTileM * dec::kD;
}
__host__ __device__ __forceinline__ int64_t tp_heap_stride_floats(int ctas_per_pe) {
    return (int64_t)ctas_per_pe * 2 * tp_tile_slot_floats();
}

}}}}  // namespace sg::fused::sm90::tp

#endif  // SG_FUSED_SM90_TP_LAYER_CUH_
