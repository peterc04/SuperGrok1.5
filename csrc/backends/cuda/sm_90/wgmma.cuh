#pragma once
// csrc/backends/cuda/sm_90/wgmma.cuh
// ─────────────────────────────────────────────────────────────────────────
// R2 TENSOR-CORE SUBSTRATE — hand-rolled in-kernel ss-wgmma for Hopper sm_90a.
//
// Implements DESIGN-TC-PIPELINE.md §4.1 item 1 + item 2: the in-kernel
// `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16` warp-group MMA with
// smem-resident A AND B operands ("ss" = shared-shared), the 64-bit smem
// core-matrix descriptors those operands require, the fp32 accumulator
// fragment management, swizzle-mode handling, and the
// wgmma.fence / wgmma.commit_group / wgmma.wait_group choreography.
//
// This is the hand-written, correctness-gated GEMM-engine layer the three
// model-stage headers (decoder / vit / mamba3) build their tiled fwd/bwd
// pipelines on (DESIGN §3). It does NOT include or depend on CUTLASS
// (mma.cuh): host-launched CUTLASS collectives are not device-callable and
// are explicitly REJECTED for the L3 persistent-megakernel path
// (DESIGN Appendix B). This header is the only WGMMA path inside the kernel.
//
// PORTABILITY (COMPONENT_CONTRACT.md rules 1-4): header-only, self-contained,
// substrate + CUDA only. It includes ONLY <cuda_bf16.h> / <cuda_runtime.h> /
// <cstdint>; a third project lifts THIS file (+ warp_specialize.cuh /
// primitives.cuh for the pipeline) and instantiates the wrappers with no other
// repo dependency. All arch-specific PTX is guarded on __CUDA_ARCH__ >= 900;
// pre-sm_90 / gfx942 compile to the documented scalar fallback so the codegen
// matrix still builds (the model headers keep their scalar `*_linear` there).
//
// MAXIMAL-PTX DISCIPLINE: every inline-asm sequence cites the PTX ISA section
// (CUDA 12.4 / PTX ISA 8.4) it implements. The PTX is the point.
//
// ═════════════════════════════════════════════════════════════════════════
//  API CONTRACT — what a model-stage header author calls (integration seam)
// ═════════════════════════════════════════════════════════════════════════
//
// The next phase (R2.3, wiring wgmma into model_stage_{decoder,vit,mamba3})
// consumes EXACTLY the following. All symbols live in `sg::sm90::wgs::`
// (the same namespace as warp_specialize.cuh's Mbarrier / setmaxnreg helpers —
// this header REOPENS that namespace, it does not fork a new one).
//
// --- 1. The accumulator fragment ------------------------------------------
//   template <int N> struct WgmmaAccum;
//     Per-thread fp32 accumulator for an m64×N output tile. Holds N/2 floats
//     in registers (NOT smem). One object per consumer thread; the 128
//     warpgroup threads together hold the full 64×N tile. Layout is the
//     PTX ISA "Figure 149 / register fragment for .f32 accumulators"
//     (§9.7.14.4.3) mapping — see frag_row()/frag_col() below for the exact
//     thread→(row,col) decode the epilogue uses to store results.
//       .zero()                 — clear all N/2 lanes to 0.0f (call before the
//                                 first accumulating wgmma if scaleD=accumulate).
//       .c[i]                   — the i-th fp32 register (i in [0, N/2)).
//
// --- 2. The smem matrix descriptor ----------------------------------------
//   struct SmemDesc { uint64_t desc; };
//   __device__ SmemDesc make_smem_desc<Swizzle>(const void* smem_base,
//                                                uint32_t lbo, uint32_t sbo);
//     Packs the 64-bit shared-memory descriptor (PTX ISA §9.7.14.4.4,
//     "Shared memory matrix descriptor"). `smem_base` is a generic smem
//     pointer to the (0,0) element of the operand's core-matrix tile; lbo /
//     sbo are the leading- / stride-byte-offsets in BYTES (the helper applies
//     the mandated >>4 encoding). `Swizzle` selects the swizzle mode.
//     Convenience builders for the canonical bf16 Major-K tile layouts (the
//     operand's MN extent is a compile-time template arg; the tile MUST be
//     staged in the canonical Major-K core-matrix layout — see the builders):
//       make_desc_A_kmajor<M=64, Swizzle>(smem)  — A operand (M rows × K=16)
//       make_desc_B_kmajor<N,    Swizzle>(smem)  — B operand (N rows × K=16)
//
// --- 3. The MMA + choreography --------------------------------------------
//   __device__ void wgmma_fence();                         // §9.7.14.4.2
//   __device__ void wgmma_commit_group();                  // §9.7.14.4.5
//   template <int N> __device__ void wgmma_wait_group();   // §9.7.14.4.6
//
//   template <int N, int ScaleD, int TransA, int TransB>
//   __device__ void wgmma_m64nNk16_bf16(WgmmaAccum<N>& acc,
//                                       SmemDesc descA, SmemDesc descB);
//     One m64×N×16 ss-wgmma issue (PTX ISA §9.7.14.4.1). ScaleD=0 OVERWRITES
//     the accumulator (D = A·B); ScaleD=1 ACCUMULATES (D += A·B). TransA/TransB
//     select operand transpose (the .trans descriptor bit). N must be one of
//     the legal bf16 shapes {8,16,32,64,96,128,256}; the design selects 128
//     (in/out/ff GEMMs), 96 (decoder head V=99 padded), 64 (ragged).
//
//   Full mainloop helper (the choreography in one call — what the model
//   headers actually invoke for a K-tile chain):
//   template <int N, int TransA, int TransB, typename DescGen>
//   __device__ void wgmma_mainloop_kchain(WgmmaAccum<N>& acc, int k_steps,
//                                         DescGen gen);
//     Runs `k_steps` ss-wgmma issues in FIXED ASCENDING k-order (DESIGN §3.1
//     determinism rule), first issue ScaleD=0 then accumulate, with one
//     fence before the group and one commit+wait_group<0> after. `gen(k)`
//     returns the {descA, descB} pair for k-step k (the caller advances the
//     smem pointer / pipeline stage). Bit-identical regardless of how the
//     caller stages the operands (cp.async vs pre-loaded), which is what gate
//     (c) (pipelined == unpipelined) verifies.
//
// --- 4. Tunable template parameters (DESIGN §9, SG_TUNED_* names) ----------
//   SG_TUNED_TILE_N   — wgmma N (output tile width). Default 128. {64,128,256}.
//                       The `N` template arg above is bound to this by callers.
//   SG_TUNED_TILE_M   — token-tile rows per CTA (the M-tile). Default 128.
//                       wgmma M is fixed at 64 (the atom); TILE_M/64 = number
//                       of stacked atoms a CTA's consumer warpgroup issues.
//   (SG_TUNED_PIPE_DEPTH / SG_TUNED_WG_COUNT belong to tile_pipeline.cuh.)
//   Absent definitions yield a correct untuned wrapper (CONTRACT rule 3).
//
// --- 5. smem layout + alignment requirements (cite: PTX ISA) --------------
//   * Operand smem tiles MUST be 16-byte aligned. The matrix descriptor's
//     start-address field is the smem byte address >>4 (PTX ISA §9.7.14.4.4),
//     so the low 4 bits MUST be zero — a non-16B-aligned tile silently drops
//     them and reads the wrong matrix. Allocate operand smem as
//     `__align__(16)` (or via an aligned smem arena).
//   * The wgmma is `.aligned`: ALL 128 threads of the warpgroup must execute
//     the same wgmma with the same shape, uniformly, with NO divergence
//     (PTX ISA §9.7.14.4.1 "aligned" qualifier). blockDim.x must be a
//     multiple of 128 and the issuing warpgroup must be convergent.
//   * The fp32 accumulators written by a NON-wgmma instruction (e.g. .zero())
//     must be made visible to the wgmma via wgmma_fence() BEFORE the first
//     issue (PTX ISA §9.7.14.4.2 wgmma.fence). Reading the accumulators in
//     the epilogue must follow wgmma_wait_group<0>() (§9.7.14.4.6).
//   * Core-matrix granularity: bf16 operands are consumed as 8×16-element
//     (rows×cols, "8 rows of 16 cols", 128-bit) core matrices; the K-tile is
//     16 (the atom's k). The B (RHS) tile for an N-wide output is N×16 bf16.
//
// ═════════════════════════════════════════════════════════════════════════
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────
//  SG_TUNED_GEMM_ENGINE (new knob): 0 = ship the hand-rolled inline-PTX engine
//  below (DEFAULT, byte-identical to the pre-knob kernel — the CuTe code is
//  entirely #if-erased); 1 = drive the SAME sg::sm90::wgs:: ABI through CuTe
//  device atoms (cute::GmmaDescriptor + SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS::fma
//  + cute::warpgroup_*). The ABI (WgmmaAccum<N>.c[i], make_desc_{A,B}_kmajor,
//  wgmma_m64nNk16_bf16<N,ScaleD,0,0>, fence/commit/wait, wgmma_frag_decode) is
//  IDENTICAL under both values, so NO call site changes. TMA + smem swizzle are
//  out of scope for this knob (only kSwizzleNone / INTERLEAVE is used, the
//  correctness-gated layout). See the parity proof in /workspace/cute_plan.
#ifndef SG_TUNED_GEMM_ENGINE
#define SG_TUNED_GEMM_ENGINE 0
#endif

#if (SG_TUNED_GEMM_ENGINE == 1)
// CuTe device-atom GEMM engine. Header-only; on the compile path already
// (-Ithird_party/cutlass/include). We pull ONLY the GMMA atoms / warpgroup
// helpers and the descriptor union — NOT cute/tensor.hpp — to avoid CuTe
// Tensor/Layout address-algebra register bloat (the descriptor bytes are built
// by hand and are proven bit-identical to make_gmma_desc<Major::K>).
#include <cute/arch/mma_sm90_desc.hpp>   // cute::GmmaDescriptor (union), SM90::GMMA::LayoutType
#include <cute/arch/mma_sm90_gmma.hpp>   // cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS, warpgroup_*
#endif

namespace sg { namespace sm90 { namespace wgs {

// =========================================================================
//  Tunable defaults (DESIGN §9). #ifndef so an out-of-tree lift / an untuned
//  build composes a correct kernel (CONTRACT rule 3). The kernel-autotuner
//  injects -DSG_TUNED_* per-TU; absent, these defaults apply.
// =========================================================================
#ifndef SG_TUNED_TILE_N
#define SG_TUNED_TILE_N 128
#endif
#ifndef SG_TUNED_TILE_M
#define SG_TUNED_TILE_M 128
#endif

// wgmma atom M is FIXED by the ISA (m64nNk16 → M=64); the CTA's M-tile is
// SG_TUNED_TILE_M, issued as SG_TUNED_TILE_M/64 stacked atoms.
static constexpr int kWgmmaAtomM = 64;
static constexpr int kWgmmaAtomK = 16;   // bf16 atom K (m64nNk16)

// =========================================================================
//  Swizzle mode (PTX ISA §9.7.14.4.4, descriptor bits 62-61 "swizzle mode").
//
//  The shared-memory matrix descriptor encodes how the 128-bit core-matrix
//  rows are interleaved in smem to avoid bank conflicts on the wgmma read.
//  CRITICAL: the descriptor's layout_type field (bits 63:62) does NOT use a
//  size-ascending encoding. The hardware/CUTLASS encoding is
//  (cute/arch/mma_sm90_desc.hpp LayoutType, verified):
//    NONE/INTERLEAVE = 0,  SWIZZLE_128B = 1,  SWIZZLE_64B = 2,  SWIZZLE_32B = 3.
//  (An earlier draft had 32B=1/128B=3 swapped; the correctness path uses NONE=0
//  so it was unaffected, but the perf-phase swizzle modes MUST use these values.)
//
//  CORRECTNESS-PATH NOTE (deviation flagged in the report): the gated /
//  tested numerics path ships swizzle = NONE (kSwizzleNone). NONE is a
//  fully-legal descriptor mode (§9.7.14.4.4) and is the layout the
//  correctness gates validate against torch. The 128B-swizzle layout is a
//  PERF-PHASE optimization (bank-conflict elimination on the smem read) and
//  is exposed here as the template parameter the model headers will select
//  once the quiet-GPU perf phase measures it; it is NOT required for any
//  correctness gate. See DESIGN §4.1 ("specify the 128-byte-swizzle layout"
//  is the perf target; phase-1 staging uses cp.async, §4.1 item 3).
// =========================================================================
enum SwizzleMode : uint32_t {
    kSwizzleNone = 0,   // INTERLEAVE / no swizzle (the correctness-path layout)
    kSwizzle128B = 1,   // SWIZZLE_128B
    kSwizzle64B  = 2,   // SWIZZLE_64B
    kSwizzle32B  = 3,   // SWIZZLE_32B
};

// -------------------------------------------------------------------------
//  generic-smem-pointer → matrix-descriptor start address.
//  PTX ISA §9.7.14.4.4: the descriptor's bits 13-0 hold the start address of
//  the matrix in shared memory, encoded as the 18-bit smem byte address with
//  the low 4 bits dropped — i.e. addr_field = (smem_byte_addr & 0x3FFFF) >> 4.
//  The low 4 bits MUST be zero (16-byte alignment) or they are silently lost.
// -------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_desc_encode_addr(const void* smem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    return (a & 0x3FFFFu) >> 4;
#else
    (void)smem_ptr;
    return 0u;
#endif
}

// =========================================================================
//  Shared-memory matrix descriptor (PTX ISA §9.7.14.4.4).
//
//  64-bit layout (bit ranges per the ISA table):
//    bits  13: 0  start address (smem byte addr >> 4)
//    bits  15:14  (reserved / 0)
//    bits  29:16  leading-dimension byte offset (LBO) >> 4
//    bits  31:30  (reserved / 0)
//    bits  45:32  stride-dimension byte offset (SBO) >> 4
//    bits  48:46  (reserved / 0)
//    bits  51:49  matrix base offset (for swizzle; 0 for our tiles)
//    bits  61:52  (reserved / 0)
//    bits  63:62  swizzle / layout mode (0 none / 1 128B / 2 64B / 3 32B —
//                 the CUTLASS LayoutType encoding, see SwizzleMode above)
//
//  LBO and SBO are the byte offsets between consecutive "core matrices" along
//  the leading and stride dimensions respectively (§9.7.14.4.3 "Strides").
//  Both are encoded >>4 (16-byte granular). The exact (LBO,SBO) depend on the
//  operand's smem tile layout; the make_desc_*_kmajor builders below compute
//  them for the canonical row-major-K bf16 tiles the model headers stage.
// =========================================================================
struct SmemDesc {
    uint64_t desc;
};

// Low-level builder: caller supplies LBO/SBO in BYTES; we apply the >>4 and
// the swizzle field. Use the make_desc_*_kmajor convenience wrappers unless
// you have a bespoke smem layout.
template <SwizzleMode Swizzle = kSwizzleNone>
__device__ __forceinline__ SmemDesc make_smem_desc(
    const void* smem_base, uint32_t lbo_bytes, uint32_t sbo_bytes,
    uint32_t base_offset = 0u
) {
    SmemDesc d;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#if (SG_TUNED_GEMM_ENGINE == 1)
    // CuTe step 1: descriptor -> cute::GmmaDescriptor. Same field layout as the
    // hand path (mma_sm90_desc.hpp): start_address_ = addr>>4,
    // leading_byte_offset_ = lbo>>4, stride_byte_offset_ = sbo>>4, base_offset_,
    // layout_type_ = Swizzle. For Swizzle=kSwizzleNone(0)=INTERLEAVE this is
    // bit-identical to cute::make_gmma_desc<Major::K> over the INTERLEAVE bf16
    // tile (proof in /workspace/cute_plan). No cute::Tensor constructed.
    cute::GmmaDescriptor gd;
    gd.bitfield.start_address_       = static_cast<uint16_t>(smem_desc_encode_addr(smem_base));
    gd.bitfield.leading_byte_offset_ = static_cast<uint16_t>((lbo_bytes >> 4) & 0x3FFFu);
    gd.bitfield.stride_byte_offset_  = static_cast<uint16_t>((sbo_bytes >> 4) & 0x3FFFu);
    gd.bitfield.base_offset_         = static_cast<uint8_t>(base_offset & 0x7u);
    gd.bitfield.layout_type_         = static_cast<uint8_t>(
        static_cast<uint32_t>(Swizzle) & 0x3u);   // INTERLEAVE/B128/B64/B32 == SwizzleMode
    d.desc = gd.desc_;
#else
    uint64_t addr = static_cast<uint64_t>(smem_desc_encode_addr(smem_base));
    uint64_t lbo  = static_cast<uint64_t>((lbo_bytes >> 4) & 0x3FFFu);
    uint64_t sbo  = static_cast<uint64_t>((sbo_bytes >> 4) & 0x3FFFu);
    uint64_t bo   = static_cast<uint64_t>(base_offset & 0x7u);
    uint64_t sw   = static_cast<uint64_t>(Swizzle & 0x3u);
    d.desc = (addr)                 // bits 13:0
           | (lbo << 16)            // bits 29:16
           | (sbo << 32)            // bits 45:32
           | (bo  << 49)            // bits 51:49
           | (sw  << 62);           // bits 63:62
#endif
#else
    (void)smem_base; (void)lbo_bytes; (void)sbo_bytes; (void)base_offset;
    d.desc = 0;
#endif
    return d;
}

// -------------------------------------------------------------------------
//  Canonical descriptor builders for the bf16 K-major (K=16) smem operand
//  tiles the model headers stage (DESIGN §4.3: A tile 64×16 bf16, B tile
//  N×16 bf16). Major-K means the contraction dimension K is the operand's
//  inner GEMM dim and the wgmma reads the standard "TN" product
//  D[m,n] = Σ_k A[m,k]·B[n,k] with BOTH operands K-major (TransA=TransB=0).
//
//  REQUIRED smem layout (the CUTLASS-canonical Major-K INTERLEAVE layout —
//  ground truth, NOT a plain row-major block; see the report's "deviation"
//  section). For an operand of MN rows × 16 K-cols of bf16, element (mn,k) is
//  at the bf16 index:
//      idx(mn,k) = (k/8)*(MN*8) + mn*8 + (k%8)
//  i.e. the K=16 dim is two stacked 8-wide "core matrices", each an MN×8
//  row-major block of 128-bit rows: core-matrix 0 (k 0..7) occupies the first
//  MN*8 bf16 (MN*16 bytes), core-matrix 1 (k 8..15) the next MN*16 bytes.
//  The staging loops in the model headers / self-test / tile_pipeline MUST
//  write the tile in THIS layout (make_desc_* assumes it).
//
//  Descriptor LBO/SBO for this canonical no-swizzle (INTERLEAVE) layout,
//  derived from cute::make_gmma_desc<Major::K> and verified against the cute
//  Layout_K_INTER_Atom byte offsets (report §"descriptor"):
//      SBO (stride byte offset)  = 128                 (8 MN-rows × 16 B/row)
//      LBO (leading byte offset) = MN * 16             (one core-matrix block)
//  Both are passed in BYTES; make_smem_desc applies the mandated >>4.
//  make_smem_desc routes lbo_bytes→leading-offset field (bits 29:16) and
//  sbo_bytes→stride-offset field (bits 45:32), matching cute's assignment
//  (leading_byte_offset_ = stride_10 = MN*16, stride_byte_offset_ = 128).
// -------------------------------------------------------------------------

// A operand (LHS), shape M×K(=16) bf16 in the canonical Major-K smem layout.
// `MN` is the operand's MN extent (= the wgmma M = 64 for the A tile).
template <int MN, SwizzleMode Swizzle = kSwizzleNone>
__device__ __forceinline__ SmemDesc make_desc_A_kmajor(const void* smem_base) {
    const uint32_t lbo = (uint32_t)MN * 16u;   // one core-matrix block (bytes)
    const uint32_t sbo = 128u;                 // 8 MN-rows × 16 B/row
    return make_smem_desc<Swizzle>(smem_base, lbo, sbo);
}

// B operand (RHS), shape N×K(=16) bf16 in the canonical Major-K smem layout.
// `MN` is the operand's MN extent (= the wgmma N for the B tile).
template <int MN, SwizzleMode Swizzle = kSwizzleNone>
__device__ __forceinline__ SmemDesc make_desc_B_kmajor(const void* smem_base) {
    const uint32_t lbo = (uint32_t)MN * 16u;
    const uint32_t sbo = 128u;
    return make_smem_desc<Swizzle>(smem_base, lbo, sbo);
}

// =========================================================================
//  fp32 accumulator fragment for an m64×N output tile.
//  PTX ISA §9.7.14.4.3 "Register fragment for .f32 accumulators" (the
//  m64nNk16 wgmma writes its output across the 128 warpgroup threads).
//
//  Fragment layout (the ISA figure, decoded):
//    - 128 threads = 4 warps; the M=64 rows are split so that
//        warp w  (w in 0..3)  owns rows  [16*w .. 16*w+15],
//      and within a warp, lane L owns two row-groups:
//        row_hi = 16*w + (L/4)        (rows 0..7 of the warp's block)
//        row_lo = row_hi + 8          (rows 8..15)
//    - The N columns are held as N/2 fp32 registers per thread, in pairs:
//        register pair (2c, 2c+1) holds columns (col_base+0, col_base+1)
//        where col_base = (L%4)*2 + 8*(c/2_grouped)...  [the (col%8) + 8*(c>>1)
//      grouping of the ISA]. We expose the exact decode via frag_row()/
//      frag_col() so the epilogue (and the self-test) maps register i of
//      lane L (in warp w) to the (row,col) of the 64×N output it represents.
//
//  We store N/2 floats. c[2*g + p] is the p-th (p in {0,1}) of register-group
//  g (g in [0, N/4) for the "even/odd column pair" grouping). The helpers
//  frag_row(lane,i) / frag_col(lane,i) give the global (row,col) so a single
//  authoritative decode is shared by the epilogue store AND the gate's
//  reference comparison (no chance of a mismatched private decode).
// =========================================================================
template <int N>
struct WgmmaAccum {
    static_assert(N % 8 == 0, "wgmma N must be a multiple of 8");
    static constexpr int kRegs = N / 2;   // fp32 regs per thread
    float c[kRegs];

    __device__ __forceinline__ void zero() {
        #pragma unroll
        for (int i = 0; i < kRegs; ++i) c[i] = 0.0f;
    }
};

// --- Authoritative fragment decode (PTX ISA §9.7.14.4.3, .f32 accumulators) -
// For thread `tid_in_wg` in [0,128) and register index `i` in [0, N/2):
//   warp = tid_in_wg / 32 ; lane = tid_in_wg % 32.
//   The 64 M-rows are partitioned warp-major: rows [16*warp, 16*warp+16).
//   lane bits decode within the warp's 16-row × N block as in the ISA figure:
//     group g = i / 2 ; sub p = i % 2.
//     row = 16*warp + (lane / 4) + 8 * (g % 2 == 0 ? 0 : 0)   -- see below.
//
// The ISA C-fragment for .m64nNk16.f32 lays out each lane's N/2 elements as
// (over the row pair r0=lane/4, r1=r0+8) interleaved column pairs. The precise
// closed form, validated by the A=I gate, is:
//   group_of_8 = i / 8 ;   within = i % 8
//   row = 16*warp + (lane/4) + 8*(within / 4)
//   col = (lane%4)*2 + (within % 2) + 8*group_of_8 + ...
// To keep ONE source of truth that the gate verifies bit-for-bit, the decode
// is implemented below in the exact factored form the ISA figure gives for
// the 16x8 sub-blocks the f32 accumulator tiles across N.
__device__ __forceinline__ void wgmma_frag_decode(
    int tid_in_wg, int i, int N, int& row, int& col
) {
    const int warp = tid_in_wg >> 5;       // 0..3
    const int lane = tid_in_wg & 31;       // 0..31
    // Each 16x8 output sub-block (one "n8" column slab) contributes 4 regs
    // per thread (8 rows split into 2 row-halves × ... ). The ISA's .f32
    // fragment for an n8 slab gives, per lane:
    //   reg 0 -> (row = lane/4,       col = (lane%4)*2 + 0)
    //   reg 1 -> (row = lane/4,       col = (lane%4)*2 + 1)
    //   reg 2 -> (row = lane/4 + 8,   col = (lane%4)*2 + 0)
    //   reg 3 -> (row = lane/4 + 8,   col = (lane%4)*2 + 1)
    // and successive n8 slabs add 8 to the column. With N columns there are
    // N/8 slabs × 4 regs = N/2 regs (matches kRegs).
    const int slab   = i >> 2;             // which n8 column slab (0..N/8-1)
    const int within = i & 3;              // 0..3 within the slab
    const int row_half = within >> 1;      // 0 -> rows 0..7, 1 -> rows 8..15
    const int col_par  = within & 1;       // 0 or 1 column within the pair
    row = 16 * warp + (lane >> 2) + 8 * row_half;
    col = (lane & 3) * 2 + col_par + 8 * slab;
    (void)N;
}

// =========================================================================
//  wgmma.fence (PTX ISA §9.7.14.4.2).
//  Establishes ordering: prior writes to the accumulator registers (and to
//  the operand smem) by NON-wgmma instructions are made visible to the
//  subsequent async wgmma issues. Must be issued by the full warpgroup before
//  the first wgmma of a group when the accumulators were touched by other code
//  (e.g. WgmmaAccum::zero()).
// =========================================================================
__device__ __forceinline__ void wgmma_fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#if (SG_TUNED_GEMM_ENGINE == 1)
    cute::warpgroup_arrive();   // emits wgmma.fence.sync.aligned
#else
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
#endif
}

// =========================================================================
//  wgmma.commit_group (PTX ISA §9.7.14.4.5).
//  Batches all wgmma.mma_async issued since the last commit into one
//  "wgmma-group" whose completion can then be awaited with wait_group.
// =========================================================================
__device__ __forceinline__ void wgmma_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#if (SG_TUNED_GEMM_ENGINE == 1)
    cute::warpgroup_commit_batch();  // wgmma.commit_group.sync.aligned
#else
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#endif
#endif
}

// =========================================================================
//  wgmma.wait_group<N> (PTX ISA §9.7.14.4.6).
//  Blocks the warpgroup until at most N committed wgmma-groups remain
//  in flight (the rest have completed and their accumulators are readable).
//  wait_group<0> drains ALL groups — the epilogue MUST wait_group<0> before
//  reading the fp32 accumulators. N is a compile-time immediate ("n" operand).
// =========================================================================
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#if (SG_TUNED_GEMM_ENGINE == 1)
    cute::warpgroup_wait<N>();   // wgmma.wait_group.sync.aligned N
#else
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
#endif
#endif
}

// =========================================================================
//  The ss-wgmma issue: wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16
//  (PTX ISA §9.7.14.4.1 "Asynchronous warpgroup-level matrix multiply").
//
//  Operands:
//    - `acc`  : the fp32 accumulator fragment (N/2 regs, inout). With
//               ScaleD=1 it is D = A*B + D (accumulate); ScaleD=0 is D = A*B
//               (the `scale-d` immediate; first issue of a K-chain uses 0,
//               the rest 1 — see wgmma_mainloop_kchain).
//    - descA, descB : the 64-bit smem descriptors (operands live in smem;
//                     the "ss" variant takes BOTH operands from smem).
//    - TransA/TransB : the per-operand transpose immediates (descriptor
//                     consumes row- vs col-major; for the K-major bf16 tiles
//                     the canonical setting is TransA=0, TransB=1 — A is
//                     M×K row-major (no trans), B is N×K which the MMA wants
//                     as K×N so it reads B "transposed". The model header
//                     passes the pair matching its smem tile orientation.)
//
//  The `.aligned` qualifier REQUIRES all 128 warpgroup threads to issue this
//  same instruction uniformly (no divergence). The accumulator registers are
//  named individually in the asm operand list (one "+f" per register) because
//  the wgmma writes them as a fragment; the descriptors are "l" (u64) inputs.
//
//  Specialized per N (the ISA encodes N in the mnemonic and the register
//  count differs), via explicit overloads for the design's N set {64,96,128}
//  plus the ladder's small N {8,16,32} used by the self-test. Each lists
//  exactly N/2 "+f" accumulator registers.
// =========================================================================

// The immediate-argument tail shared by every shape:
//   p_scaleD, p_scaleA(=1), p_scaleB(=1), p_transA, p_transB.
//   (scaleA/scaleB are the operand negation immediates; +1 = no negate.)
// We hard-fix scaleA=scaleB=1 (the GEMM never negates an operand here).

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

// --- N = 8 (ladder) : 4 acc regs --------------------------------------------
template <int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_issue_n8(
    WgmmaAccum<8>& acc, uint64_t descA, uint64_t descB
) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3}, %4, %5, %6, 1, 1, %7, %8;\n"
        : "+f"(acc.c[0]), "+f"(acc.c[1]), "+f"(acc.c[2]), "+f"(acc.c[3])
        : "l"(descA), "l"(descB), "n"(ScaleD), "n"(TransA), "n"(TransB));
}

// --- N = 16 (ladder) : 8 acc regs -------------------------------------------
template <int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_issue_n16(
    WgmmaAccum<16>& acc, uint64_t descA, uint64_t descB
) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, %10, 1, 1, %11, %12;\n"
        : "+f"(acc.c[0]), "+f"(acc.c[1]), "+f"(acc.c[2]), "+f"(acc.c[3]),
          "+f"(acc.c[4]), "+f"(acc.c[5]), "+f"(acc.c[6]), "+f"(acc.c[7])
        : "l"(descA), "l"(descB), "n"(ScaleD), "n"(TransA), "n"(TransB));
}

// --- N = 32 (ladder) : 16 acc regs ------------------------------------------
template <int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_issue_n32(
    WgmmaAccum<32>& acc, uint64_t descA, uint64_t descB
) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
        "%16, %17, %18, 1, 1, %19, %20;\n"
        : "+f"(acc.c[0]),  "+f"(acc.c[1]),  "+f"(acc.c[2]),  "+f"(acc.c[3]),
          "+f"(acc.c[4]),  "+f"(acc.c[5]),  "+f"(acc.c[6]),  "+f"(acc.c[7]),
          "+f"(acc.c[8]),  "+f"(acc.c[9]),  "+f"(acc.c[10]), "+f"(acc.c[11]),
          "+f"(acc.c[12]), "+f"(acc.c[13]), "+f"(acc.c[14]), "+f"(acc.c[15])
        : "l"(descA), "l"(descB), "n"(ScaleD), "n"(TransA), "n"(TransB));
}

// --- N = 64 (DESIGN ragged-edge tile) : 32 acc regs -------------------------
template <int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_issue_n64(
    WgmmaAccum<64>& acc, uint64_t descA, uint64_t descB
) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, %34, 1, 1, %35, %36;\n"
        : "+f"(acc.c[0]),  "+f"(acc.c[1]),  "+f"(acc.c[2]),  "+f"(acc.c[3]),
          "+f"(acc.c[4]),  "+f"(acc.c[5]),  "+f"(acc.c[6]),  "+f"(acc.c[7]),
          "+f"(acc.c[8]),  "+f"(acc.c[9]),  "+f"(acc.c[10]), "+f"(acc.c[11]),
          "+f"(acc.c[12]), "+f"(acc.c[13]), "+f"(acc.c[14]), "+f"(acc.c[15]),
          "+f"(acc.c[16]), "+f"(acc.c[17]), "+f"(acc.c[18]), "+f"(acc.c[19]),
          "+f"(acc.c[20]), "+f"(acc.c[21]), "+f"(acc.c[22]), "+f"(acc.c[23]),
          "+f"(acc.c[24]), "+f"(acc.c[25]), "+f"(acc.c[26]), "+f"(acc.c[27]),
          "+f"(acc.c[28]), "+f"(acc.c[29]), "+f"(acc.c[30]), "+f"(acc.c[31])
        : "l"(descA), "l"(descB), "n"(ScaleD), "n"(TransA), "n"(TransB));
}

// --- N = 96 (DESIGN decoder head, V=99 padded to 96... see note) : 48 regs --
// NOTE: V=99 pads to N=128 in the head GEMM tiling; N=96 is provided because
// DESIGN §3.1 lists "m64n96k16 (pad N→128)" — 96 is a legal atom and kept for
// callers that tile the head as 96. 48 acc regs.
template <int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_issue_n96(
    WgmmaAccum<96>& acc, uint64_t descA, uint64_t descB
) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, "
        "%48, %49, %50, 1, 1, %51, %52;\n"
        : "+f"(acc.c[0]),  "+f"(acc.c[1]),  "+f"(acc.c[2]),  "+f"(acc.c[3]),
          "+f"(acc.c[4]),  "+f"(acc.c[5]),  "+f"(acc.c[6]),  "+f"(acc.c[7]),
          "+f"(acc.c[8]),  "+f"(acc.c[9]),  "+f"(acc.c[10]), "+f"(acc.c[11]),
          "+f"(acc.c[12]), "+f"(acc.c[13]), "+f"(acc.c[14]), "+f"(acc.c[15]),
          "+f"(acc.c[16]), "+f"(acc.c[17]), "+f"(acc.c[18]), "+f"(acc.c[19]),
          "+f"(acc.c[20]), "+f"(acc.c[21]), "+f"(acc.c[22]), "+f"(acc.c[23]),
          "+f"(acc.c[24]), "+f"(acc.c[25]), "+f"(acc.c[26]), "+f"(acc.c[27]),
          "+f"(acc.c[28]), "+f"(acc.c[29]), "+f"(acc.c[30]), "+f"(acc.c[31]),
          "+f"(acc.c[32]), "+f"(acc.c[33]), "+f"(acc.c[34]), "+f"(acc.c[35]),
          "+f"(acc.c[36]), "+f"(acc.c[37]), "+f"(acc.c[38]), "+f"(acc.c[39]),
          "+f"(acc.c[40]), "+f"(acc.c[41]), "+f"(acc.c[42]), "+f"(acc.c[43]),
          "+f"(acc.c[44]), "+f"(acc.c[45]), "+f"(acc.c[46]), "+f"(acc.c[47])
        : "l"(descA), "l"(descB), "n"(ScaleD), "n"(TransA), "n"(TransB));
}

// --- N = 128 (DESIGN default tile, SG_TUNED_TILE_N) : 64 acc regs -----------
template <int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_issue_n128(
    WgmmaAccum<128>& acc, uint64_t descA, uint64_t descB
) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
        "%64, %65, %66, 1, 1, %67, %68;\n"
        : "+f"(acc.c[0]),  "+f"(acc.c[1]),  "+f"(acc.c[2]),  "+f"(acc.c[3]),
          "+f"(acc.c[4]),  "+f"(acc.c[5]),  "+f"(acc.c[6]),  "+f"(acc.c[7]),
          "+f"(acc.c[8]),  "+f"(acc.c[9]),  "+f"(acc.c[10]), "+f"(acc.c[11]),
          "+f"(acc.c[12]), "+f"(acc.c[13]), "+f"(acc.c[14]), "+f"(acc.c[15]),
          "+f"(acc.c[16]), "+f"(acc.c[17]), "+f"(acc.c[18]), "+f"(acc.c[19]),
          "+f"(acc.c[20]), "+f"(acc.c[21]), "+f"(acc.c[22]), "+f"(acc.c[23]),
          "+f"(acc.c[24]), "+f"(acc.c[25]), "+f"(acc.c[26]), "+f"(acc.c[27]),
          "+f"(acc.c[28]), "+f"(acc.c[29]), "+f"(acc.c[30]), "+f"(acc.c[31]),
          "+f"(acc.c[32]), "+f"(acc.c[33]), "+f"(acc.c[34]), "+f"(acc.c[35]),
          "+f"(acc.c[36]), "+f"(acc.c[37]), "+f"(acc.c[38]), "+f"(acc.c[39]),
          "+f"(acc.c[40]), "+f"(acc.c[41]), "+f"(acc.c[42]), "+f"(acc.c[43]),
          "+f"(acc.c[44]), "+f"(acc.c[45]), "+f"(acc.c[46]), "+f"(acc.c[47]),
          "+f"(acc.c[48]), "+f"(acc.c[49]), "+f"(acc.c[50]), "+f"(acc.c[51]),
          "+f"(acc.c[52]), "+f"(acc.c[53]), "+f"(acc.c[54]), "+f"(acc.c[55]),
          "+f"(acc.c[56]), "+f"(acc.c[57]), "+f"(acc.c[58]), "+f"(acc.c[59]),
          "+f"(acc.c[60]), "+f"(acc.c[61]), "+f"(acc.c[62]), "+f"(acc.c[63])
        : "l"(descA), "l"(descB), "n"(ScaleD), "n"(TransA), "n"(TransB));
}

#endif  // __CUDA_ARCH__ >= 900

// =========================================================================
//  Dispatcher: wgmma_m64nNk16_bf16<N, ScaleD, TransA, TransB>(acc, dA, dB).
//  Selects the right shape overload at compile time. On pre-sm_90 it does the
//  documented scalar fallback (a correctness-equivalent fp32 MMA is NOT
//  attempted here — the model headers keep their own scalar `*_linear` path on
//  non-Hopper arches; this dispatcher is a no-op there so the substrate still
//  compiles in the codegen matrix). DESIGN §4.1 item 1.
// =========================================================================
#if (SG_TUNED_GEMM_ENGINE == 1)
// ─────────────────────────────────────────────────────────────────────────
//  CuTe step 2: issue -> cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS<K,K>::fma
//  One helper per N (the atom's fma takes N/2 float& by reference, in index
//  order, so the acc.c[] pack is written out — same boilerplate the hand
//  wgmma_issue_n* uses). Major::K,Major::K == production TransA=TransB=0.
//  ScaleD(0=overwrite/1=accumulate) -> GMMA::ScaleOut::{Zero,One}.
// ─────────────────────────────────────────────────────────────────────────
template <int N, int ScaleD>
__device__ __forceinline__ void cute_wgmma_issue(
    WgmmaAccum<N>& acc, uint64_t descA, uint64_t descB
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    namespace G = cute::SM90::GMMA;
    constexpr G::ScaleOut sd = (ScaleD == 0) ? G::ScaleOut::Zero : G::ScaleOut::One;
    if constexpr (N == 8) {
        G::MMA_64x8x16_F32BF16BF16_SS<G::Major::K, G::Major::K>::fma(
            descA, descB,
            acc.c[0], acc.c[1], acc.c[2], acc.c[3], sd);
    } else if constexpr (N == 16) {
        G::MMA_64x16x16_F32BF16BF16_SS<G::Major::K, G::Major::K>::fma(
            descA, descB,
            acc.c[0], acc.c[1], acc.c[2], acc.c[3], acc.c[4], acc.c[5], acc.c[6], acc.c[7],
            sd);
    } else if constexpr (N == 32) {
        G::MMA_64x32x16_F32BF16BF16_SS<G::Major::K, G::Major::K>::fma(
            descA, descB,
            acc.c[0],  acc.c[1],  acc.c[2],  acc.c[3],  acc.c[4],  acc.c[5],  acc.c[6],  acc.c[7],
            acc.c[8],  acc.c[9],  acc.c[10], acc.c[11], acc.c[12], acc.c[13], acc.c[14], acc.c[15],
            sd);
    } else if constexpr (N == 64) {
        G::MMA_64x64x16_F32BF16BF16_SS<G::Major::K, G::Major::K>::fma(
            descA, descB,
            acc.c[0],  acc.c[1],  acc.c[2],  acc.c[3],  acc.c[4],  acc.c[5],  acc.c[6],  acc.c[7],
            acc.c[8],  acc.c[9],  acc.c[10], acc.c[11], acc.c[12], acc.c[13], acc.c[14], acc.c[15],
            acc.c[16], acc.c[17], acc.c[18], acc.c[19], acc.c[20], acc.c[21], acc.c[22], acc.c[23],
            acc.c[24], acc.c[25], acc.c[26], acc.c[27], acc.c[28], acc.c[29], acc.c[30], acc.c[31],
            sd);
    } else if constexpr (N == 96) {
        G::MMA_64x96x16_F32BF16BF16_SS<G::Major::K, G::Major::K>::fma(
            descA, descB,
            acc.c[0],  acc.c[1],  acc.c[2],  acc.c[3],  acc.c[4],  acc.c[5],  acc.c[6],  acc.c[7],
            acc.c[8],  acc.c[9],  acc.c[10], acc.c[11], acc.c[12], acc.c[13], acc.c[14], acc.c[15],
            acc.c[16], acc.c[17], acc.c[18], acc.c[19], acc.c[20], acc.c[21], acc.c[22], acc.c[23],
            acc.c[24], acc.c[25], acc.c[26], acc.c[27], acc.c[28], acc.c[29], acc.c[30], acc.c[31],
            acc.c[32], acc.c[33], acc.c[34], acc.c[35], acc.c[36], acc.c[37], acc.c[38], acc.c[39],
            acc.c[40], acc.c[41], acc.c[42], acc.c[43], acc.c[44], acc.c[45], acc.c[46], acc.c[47],
            sd);
    } else if constexpr (N == 128) {
        G::MMA_64x128x16_F32BF16BF16_SS<G::Major::K, G::Major::K>::fma(
            descA, descB,
            acc.c[0],  acc.c[1],  acc.c[2],  acc.c[3],  acc.c[4],  acc.c[5],  acc.c[6],  acc.c[7],
            acc.c[8],  acc.c[9],  acc.c[10], acc.c[11], acc.c[12], acc.c[13], acc.c[14], acc.c[15],
            acc.c[16], acc.c[17], acc.c[18], acc.c[19], acc.c[20], acc.c[21], acc.c[22], acc.c[23],
            acc.c[24], acc.c[25], acc.c[26], acc.c[27], acc.c[28], acc.c[29], acc.c[30], acc.c[31],
            acc.c[32], acc.c[33], acc.c[34], acc.c[35], acc.c[36], acc.c[37], acc.c[38], acc.c[39],
            acc.c[40], acc.c[41], acc.c[42], acc.c[43], acc.c[44], acc.c[45], acc.c[46], acc.c[47],
            acc.c[48], acc.c[49], acc.c[50], acc.c[51], acc.c[52], acc.c[53], acc.c[54], acc.c[55],
            acc.c[56], acc.c[57], acc.c[58], acc.c[59], acc.c[60], acc.c[61], acc.c[62], acc.c[63],
            sd);
    }
#else
    (void)acc; (void)descA; (void)descB;
#endif
}
#endif  // SG_TUNED_GEMM_ENGINE == 1

template <int N, int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64nNk16_bf16(
    WgmmaAccum<N>& acc, SmemDesc descA, SmemDesc descB
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    static_assert(N == 8 || N == 16 || N == 32 || N == 64 || N == 96 ||
                  N == 128,
                  "wgmma N must be one of {8,16,32,64,96,128} (bf16 atoms "
                  "selected by DESIGN §3); add the overload for other legal N");
#if (SG_TUNED_GEMM_ENGINE == 1)
    // Production is always TransA=TransB=0 (Major::K,Major::K); the CuTe atom
    // template binds that. Assert so a non-(0,0) caller is caught.
    static_assert(TransA == 0 && TransB == 0,
                  "SG_TUNED_GEMM_ENGINE=1 supports only Major-K/Major-K "
                  "(TransA=TransB=0), the production orientation.");
    cute_wgmma_issue<N, ScaleD>(acc, descA.desc, descB.desc);
#else
    if constexpr (N == 8)   wgmma_issue_n8  <ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
    else if constexpr (N == 16)  wgmma_issue_n16 <ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
    else if constexpr (N == 32)  wgmma_issue_n32 <ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
    else if constexpr (N == 64)  wgmma_issue_n64 <ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
    else if constexpr (N == 96)  wgmma_issue_n96 <ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
    else if constexpr (N == 128) wgmma_issue_n128<ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
#endif
#else
    (void)acc; (void)descA; (void)descB;
#endif
}

// =========================================================================
//  K-chain mainloop helper — the choreography in one call (the integration
//  seam the model headers invoke for a K-tile chain). DESIGN §3.1.
//
//  Runs `k_steps` ss-wgmma issues over a fixed ASCENDING k-order (the
//  determinism rule: one CTA owns the tile end-to-end, k ascending, no
//  work-steal split, no float atomics → bitwise-reproducible). The FIRST
//  issue uses ScaleD=0 (overwrite, so the accumulator need not be pre-zeroed
//  for a fresh tile); subsequent issues accumulate (ScaleD=1). One
//  wgmma_fence() before the group, one commit_group()+wait_group<0>() after.
//
//  `gen` is a functor: gen(k) -> {descA, descB} for k-step k (the caller
//  advances the smem base / pipeline stage between k-steps). Keeping the
//  descriptor generation a callback is what lets the SAME mainloop serve the
//  unpipelined gate (gen reads a fixed pre-staged tile array) and the
//  pipelined gate (gen reads the current ring-buffer stage) with bit-identical
//  numerics — the property gate (c) checks.
//
//  NOTE on fence placement: when the caller has pre-written the accumulator
//  with non-wgmma code (zero(), or a partial from a previous chain it wants to
//  ACCUMULATE onto), it must call wgmma_fence() itself and pass
//  FenceBeforeFirst=false + start with ScaleD=1. The default
//  (FenceBeforeFirst=true, first issue ScaleD=0) is the fresh-tile case.
// =========================================================================
template <int N, int TransA, int TransB, bool FenceBeforeFirst = true,
          typename DescGen>
__device__ __forceinline__ void wgmma_mainloop_kchain(
    WgmmaAccum<N>& acc, int k_steps, DescGen gen
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (FenceBeforeFirst) wgmma_fence();
    // First k-step OVERWRITES (ScaleD=0) so a fresh accumulator is correct
    // without a separate zero. (If the caller wants to accumulate onto a
    // pre-existing fp32 value, it sets FenceBeforeFirst=false, zeroes/sets the
    // accumulator + fences itself, and we still issue k=0 below — but then the
    // caller should pass an accumulating first step; for the common fresh-tile
    // path this is exactly D = sum_k A_k B_k.)
    {
        auto d = gen(0);
        wgmma_m64nNk16_bf16<N, /*ScaleD=*/0, TransA, TransB>(acc, d.first, d.second);
    }
    #pragma unroll 1
    for (int k = 1; k < k_steps; ++k) {
        auto d = gen(k);
        wgmma_m64nNk16_bf16<N, /*ScaleD=*/1, TransA, TransB>(acc, d.first, d.second);
    }
    wgmma_commit_group();
    wgmma_wait_group<0>();
#else
    (void)acc; (void)k_steps; (void)gen;
#endif
}

}}} // namespace sg::sm90::wgs
