# APPLY-READY SPEC — CuTe step 4.2: wire the gated TMA load into the megakernel fwd/dX staging

AREA: `csrc/backends/cuda/sm_90/{wgmma.cuh, tile_pipeline.cuh}` +
`csrc/fused/sm_90/model_stage_decoder_tc.cuh` (the fwd/dX producer + the
orientation wrappers + the tile fns) + `csrc/fused/megakernel_common.cuh`
(`PersistentContext`) + `csrc/fused/sm_90/fused_decoder_megakernel.cuh` (the
carve + the shared launcher) + `csrc/bindings/dispatch.cpp` (the by-value
`PersistentContext` mirror) + a NEW host-only header
`csrc/backends/cuda/sm_90/cute_tma_desc.h`.

GOAL: turn the DOCUMENTED-BUT-DEFERRED §5 of `/workspace/impl_diffs/tma.md` into
APPLY-READY exact edits. This closes LEVER ① (the GEMM at 6.5% of roofline / 72%
of step): the fwd/dX A/B operands are staged into smem by the Hopper TMA
(`cp.async.bulk.tensor.2d`) instead of the hand cp.async ring, driven by
host-built `CUtensorMap` descriptors over the STEP-STABLE workspace operand
bases. dW keeps cp.async (its transposed-strided gather is not TMA-reachable
unless `SG_TUNED_DEC_DW_STAGE=1`; out of scope here).

TRANSPORT-ONLY: TMA only changes WHERE/HOW the bf16 bytes land in smem. After the
tile lands, the SAME `make_desc_*_kmajor` + `wgmma_m64nNk16_bf16` run in the SAME
ascending-k order, so the wgmma fp32 accumulation is bit-identical → fp64 parity
(`_TC_LOSS_REL=1e-4`, grad rel ≤0.15w/0.08b) + A/A/A bit-determinism are preserved
BY CONSTRUCTION.

HARD GATE: **byte-identical when `SG_TUNED_GEMM_TMA=0`** (the shipped default and
the default of both gate builds). Every device-side change is `#if (SG_TUNED_GEMM_TMA
== 1)`-erased to the pre-knob code; every host-side change is gated likewise or is
a trailing-defaulted struct member that OFF builds never read. gfx942/tpu
untouched (all new device code is sm_90-only and CUDA-only).

```
gate_commands:
  CUDA_VISIBLE_DEVICES=0 bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1 -DSG_TUNED_GEMM_ENGINE=1 -DSG_TUNED_GEMM_TMA=1
  CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_decoder_tc.py -q
```

---

## 0. HONESTY — what this spec applies, and the one thing the lead must verify on silicon

The two device primitives (`tma_load_kmajor_tile`, `tma_prefetch_desc`) and the
host descriptor builder (`cute_tma_desc.h`) ARE the §2-4 unit that `tma.md`
specified but did not apply to the live tree (verified this session:
`grep SG_TUNED_GEMM_TMA csrc/backends/cuda/sm_90/wgmma.cuh` → no hits;
`cute_tma_desc.h` does not exist). So this spec INCLUDES those §2-4 edits
(Sections 1-3 below) re-anchored VERBATIM against the LIVE post-steps-1-3
`wgmma.cuh`, then adds the §5 wiring (Sections 4-8). Applying this spec alone
produces a complete, compilable, gate-passing build.

What is mechanically exact and provably byte-identical-when-OFF (Sections 1-8):
every device-side `#if SG_TUNED_GEMM_TMA` block, the trailing-defaulted struct
members, the trailing-defaulted function params, and the `SG_DEC_TMA_*_ARG` macro
idiom (a verbatim clone of the existing `SG_DEC_PIPE_BARS_ARG` /
`SG_DEC_FINE_*` idioms the codebase already proves byte-identical-when-OFF).

The one thing the lead MUST verify on silicon (it cannot be proven read-only):
the TMA `complete_tx::bytes` count and the canonical-Major-K smem layout the TMA
writes must equal the cp.async layout BYTE-FOR-BYTE. Section 3 builds the
descriptor with `GMMA::Layout_K_INTER_Atom<bf16>` tiled to `Shape<TILE_MN,16>`
(no swizzle / INTERLEAVE) — the SAME layout `make_desc_*_kmajor<…,kSwizzleNone>`
reads and the SAME canonical map `idx(mn,k)=(k/8)*(MN*8)+mn*8+(k%8)` the cp.async
ring writes (`stage_kmajor_tile`, model_stage_decoder_tc.cuh:601). The decoder
parity gate (gate 2 with `-DSG_TUNED_GEMM_TMA=1` injected — see Section 9) is the
silicon proof; if it fails, the TMA box/swizzle is the suspect, not the wgmma.

---

## 1. EDIT `wgmma.cuh` — add the `SG_TUNED_GEMM_TMA` knob + the TMA includes (tma.md §2)

The live engine-knob block is at wgmma.cuh:136-148 (post-steps-1-3; the trailing
comment line reads `/workspace/cute_plan`, NOT what tma.md §2's OLD quoted —
match the LIVE text below).

### OLD (wgmma.cuh:126-148 — copy VERBATIM)

```cpp
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
```

### NEW (replace the block above with)

```cpp
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

// ─────────────────────────────────────────────────────────────────────────
//  SG_TUNED_GEMM_TMA (CuTe step 4 sub-knob): 0 = stage A/B into smem via the
//  current hand cp.async ring (DEFAULT, byte-identical — the entire TMA block
//  below is #if-erased, so PTX/smem/regs equal the pre-knob kernel); 1 = expose
//  the device TMA-load primitive (cute::SM90_TMA_LOAD::copy =
//  cp.async.bulk.tensor.2d ...) so the producer can stage A/B from a host-built
//  CUtensorMap instead of cp.async. ONLY meaningful when SG_TUNED_GEMM_ENGINE==1
//  (TMA composes with the CuTe MMA-atom engine; the hand-PTX engine keeps
//  cp.async). TRANSPORT-ONLY: the wgmma ascending-k fp32 accumulation is
//  untouched, so fp64 parity + A/A/A are preserved. The host descriptors are
//  built over STEP-STABLE workspace bases (see cute_tma_desc.h); the megakernel
//  passes a device CUtensorMap* through PersistentContext. TMA's single-CTA
//  mbarrier (warp_specialize.cuh Mbarrier::arrive_expect_tx / try_wait) is a
//  CTA-LOCAL barrier — it does NOT interact with the cross-CTA hand GridBarrier,
//  and no thread-block cluster is launched (cp.async.bulk.tensor's
//  shared::cluster scope is valid at the implicit cluster-size-1). dW's
//  transposed-strided gather is NOT covered (stays cp.async) — see tma_wire.md.
#ifndef SG_TUNED_GEMM_TMA
#define SG_TUNED_GEMM_TMA 0
#endif
#if (SG_TUNED_GEMM_TMA == 1) && (SG_TUNED_GEMM_ENGINE != 1)
#error "SG_TUNED_GEMM_TMA=1 requires SG_TUNED_GEMM_ENGINE=1 (TMA composes with the CuTe MMA atoms; the hand-PTX engine stays on cp.async)."
#endif

#if (SG_TUNED_GEMM_ENGINE == 1)
// CuTe device-atom GEMM engine. Header-only; on the compile path already
// (-Ithird_party/cutlass/include). We pull ONLY the GMMA atoms / warpgroup
// helpers and the descriptor union — NOT cute/tensor.hpp — to avoid CuTe
// Tensor/Layout address-algebra register bloat (the descriptor bytes are built
// by hand and are proven bit-identical to make_gmma_desc<Major::K>).
#include <cute/arch/mma_sm90_desc.hpp>   // cute::GmmaDescriptor (union), SM90::GMMA::LayoutType
#include <cute/arch/mma_sm90_gmma.hpp>   // cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS, warpgroup_*
#if (SG_TUNED_GEMM_TMA == 1)
// CuTe step 4: the TMA bulk-tensor load atom + the device tensormap fence/
// prefetch helpers. SM90_TMA_LOAD::copy emits exactly
//   cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
// (copy_sm90_tma.hpp). copy_sm90_desc.hpp gives CUtensorMap (= cute::TmaDescriptor),
// prefetch_tma_descriptor, and the mbarrier helpers — all CUTE_HOST_DEVICE, no
// cute::Tensor needed. Guarded by CUTE_ARCH_TMA_SM90_ENABLED (active on sm_90a
// via cutlass/arch/config.h; config.hpp activates it for compute_90a).
#include <cute/arch/copy_sm90_tma.hpp>   // cute::SM90_TMA_LOAD::copy (cp.async.bulk.tensor)
#include <cute/arch/copy_sm90_desc.hpp>  // cute::TmaDescriptor (CUtensorMap), prefetch_tma_descriptor
#endif
#endif
```

NOTE: `copy_sm90_desc.hpp` includes `<cuda.h>` (for `CUtensorMap`) under
`!defined(__CUDACC_RTC__)`. The gate compiles `mega_decoder_real_adamw_tc.cu` by
nvcc (not RTC), `-Ithird_party/cutlass/include` is on the line
(compile_to_object.sh:11), and `CUDACC_VER_MAJOR>=12` holds — so the include
resolves and `CUtensorMap` is a real type.

---

## 2. EDIT `wgmma.cuh` — add the device TMA-load primitive (tma.md §3)

Insert the TMA-load helper AFTER the `cute_wgmma_issue` dispatcher's closing
`#endif  // SG_TUNED_GEMM_ENGINE == 1` and BEFORE the `wgmma_m64nNk16_bf16`
template, so it lives inside the engine-on region. Live anchor: wgmma.cuh:687-693.

### OLD (wgmma.cuh:687-697 — copy VERBATIM)

```cpp
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
```

### NEW (replace the block above with)

```cpp
#else
    (void)acc; (void)descA; (void)descB;
#endif
}

#if (SG_TUNED_GEMM_TMA == 1)
// ─────────────────────────────────────────────────────────────────────────
//  CuTe step 4: TMA bulk-tensor LOAD of one canonical Major-K bf16 tile.
//
//  TRANSPORT-ONLY replacement for the cp.async staging of ONE operand tile (the
//  A 64×16 or B N×16 half-row scatter the decoder ring does in stage_k_async,
//  model_stage_decoder_tc.cuh:848-881). It does NOT touch the wgmma issue: after
//  the tile lands in smem the SAME make_desc_*_kmajor + wgmma_m64nNk16_bf16 run
//  in the SAME ascending-k order, so the fp32 accumulation is bit-identical
//  (parity + A/A/A by construction).
//
//  CONTRACT (the caller owns the mbarrier + the parity handshake; this header
//  stays free of warp_specialize.cuh):
//    * tma_desc  — a host-built CUtensorMap over the operand's FULL row-major
//      [rows, K] bf16 tensor (K = the contraction axis, Major-K). Lives in
//      global memory (a device CUtensorMap*); pass &desc. Built by
//      cute_tma_desc.h::sg_build_tma_desc_kmajor on the host over the STEP-STABLE
//      workspace base (DecActs / DecWBf region). One descriptor per (base, rows,
//      K); it covers every n0 tile + every k-step via the (crd_mn, crd_k) coords.
//    * mbar      — a uint64_t* to a CTA-LOCAL smem mbarrier the CALLER has
//      mbarrier.init'd and on which the CALLER issues
//      Mbarrier::arrive_expect_tx(tile_bytes) BEFORE this call and
//      Mbarrier::try_wait(parity) AFTER (warp_specialize.cuh:98,117). TMA
//      auto-counts the transaction bytes into this mbarrier
//      (mbarrier::complete_tx::bytes).
//    * smem_tile — the 16B-aligned smem destination (DecTcSmem.sA/sB slot). The
//      CUtensorMap box (built host-side as Shape<TILE_MN,16>, Major-K INTERLEAVE
//      = no swizzle) lands the tile in the EXACT canonical Major-K layout
//      idx(mn,k)=(k/8)*(MN*8)+mn*8+(k%8) the cp.async path writes and
//      make_desc_*_kmajor reads — byte-identical smem.
//    * crd_mn    — the tile's row origin in the global tensor (= n0 for B, the
//      tile's token row origin g0+ai*64 for A; the leading-coord of the 2D box).
//    * crd_k     — k*16, the k-step's contraction origin (the box's K coord).
//
//  Single-CTA / no cluster: SM90_TMA_LOAD's PTX is shared::cluster-scoped but
//  valid at the implicit cluster-size-1 (descriptor built with Cluster_Size=
//  Int<1>); cache_hint 0 = default. No ClusterBarrier, no cooperative launch —
//  composes with the hand GridBarrier.
//  ─────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void tma_load_kmajor_tile(
    const void* tma_desc, unsigned long long* mbar, void* smem_tile,
    int crd_mn, int crd_k
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // cute::SM90_TMA_LOAD::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr,
    //                           crd0, crd1) -> the 2D overload (copy_sm90_tma.hpp).
    // crd0 is the tensor's leading (row/MN) coord, crd1 the K coord; the box
    // shape (TILE_MN×16) is baked into the descriptor.
    cute::SM90_TMA_LOAD::copy(
        tma_desc,
        reinterpret_cast<uint64_t*>(mbar),
        /*cache_hint=*/0ull,
        smem_tile,
        crd_mn, crd_k);
#else
    (void)tma_desc; (void)mbar; (void)smem_tile; (void)crd_mn; (void)crd_k;
#endif
}

// Prefetch a tensormap into the descriptor cache (cheap, once before the first
// TMA over a freshly-uploaded descriptor — copy_sm90_desc.hpp). The caller issues
// this from one thread after the descriptor is visible in global.
__device__ __forceinline__ void tma_prefetch_desc(const void* tma_desc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cute::prefetch_tma_descriptor(
        reinterpret_cast<const cute::TmaDescriptor*>(tma_desc));
#else
    (void)tma_desc;
#endif
}
#endif  // SG_TUNED_GEMM_TMA == 1

#endif  // SG_TUNED_GEMM_ENGINE == 1

template <int N, int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64nNk16_bf16(
    WgmmaAccum<N>& acc, SmemDesc descA, SmemDesc descB
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
```

This is the ONLY change to `wgmma.cuh`'s code body. With `SG_TUNED_GEMM_TMA==0`
the entire `#if (SG_TUNED_GEMM_TMA == 1)` block is erased → byte-identical to the
pre-knob file. With both flags `1` it compiles: the helpers use only
`cute::SM90_TMA_LOAD::copy` / `cute::prefetch_tma_descriptor` /
`cute::TmaDescriptor`, all from the two includes added in Section 1.

---

## 3. NEW FILE — `csrc/backends/cuda/sm_90/cute_tma_desc.h` (host-only, tma.md §4)

The host-side `CUtensorMap` builder over a step-stable row-major `[rows, K]` bf16
operand. HOST + CUDA only; includes `cute/tensor.hpp` (host algebra) — this is the
ONLY place a `cute::Tensor` is constructed, and it is HOST code (no device
register cost). It is included ONLY by the device TU's host-side launcher
function (`launch_fused_decoder_megakernel_tc`, which is `__host__` and lives in
the same .cu); the device kernel does NOT include it (it only needs the device
`tma_load_kmajor_tile` in wgmma.cuh + a `const void*` descriptor pointer).

Write the file VERBATIM:

```cpp
#pragma once
// csrc/backends/cuda/sm_90/cute_tma_desc.h
// ─────────────────────────────────────────────────────────────────────────
// CuTe step 4 — HOST-SIDE CUtensorMap builders for the persistent decoder
// megakernel's TMA staging (paired with wgmma.cuh's tma_load_kmajor_tile).
//
// The persistent megakernel carves its A/B operands from a runtime workspace,
// but the bases are STEP-STABLE (tok.workspace is cudaMalloc'd once per B and
// reused; T=B*kSeq and nCTA are fixed for a fixed B/device — see tma.md §1's
// step-stability audit). So the host can compute every fwd/dX operand base from
// (workspace_base, T, B, nCTA) and build ONE CUtensorMap per (base, rows, K)
// tensor; the descriptor covers all n0 tiles + all k-steps via TMA coordinates.
//
// CuTe builds CUtensorMap host-side ONLY (cuTensorMapEncodeTiled inside
// make_tma_copy). The raw 128-byte CUtensorMap is extracted and uploaded to a
// device buffer; the kernel reads it from global (a device CUtensorMap* in
// PersistentContext).
//
// HOST + CUDA only. Guarded so a stray device/RTC include is inert. Included
// ONLY by the device TU's host launcher; the megakernel device code never sees it.
// ─────────────────────────────────────────────────────────────────────────
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#if !defined(__CUDACC_RTC__)
#include <cuda.h>            // CUtensorMap
#include <cute/tensor.hpp>   // HOST: make_tensor / make_layout / make_tma_copy
#include <cute/arch/copy_sm90_tma.hpp>          // cute::SM90_TMA_LOAD
#include <cute/atom/copy_traits_sm90_tma.hpp>   // cute::make_tma_copy
#include <cute/arch/mma_sm90_gmma.hpp>          // GMMA::Layout_K_INTER_Atom
#endif

namespace sg { namespace sm90 { namespace wgs {

// 128-byte POD wrapper so a launcher can keep an array of descriptors without
// dragging CUtensorMap's header into non-CUDA TUs. Layout-compatible with
// CUtensorMap (alignas(64), 128 bytes) so memcpy/upload is a straight copy.
struct TmaDescBytes { alignas(64) unsigned char bytes[128]; };

#if !defined(__CUDACC_RTC__) && (__CUDACC_VER_MAJOR__ >= 12)

// Build a single-CTA Major-K (no-swizzle/INTERLEAVE) TMA descriptor over a
// row-major [rows, K] bf16 operand. TILE_MN = the smem tile's MN extent (64 for
// A, the wgmma N for B); the box is Shape<TILE_MN, 16> (the canonical k16 atom
// tile). `base` is the operand's element (0,0) (the DecActs/DecWBf region base);
// `rows`/`K` are the FULL tensor shape (K-contiguous rows, ld = K).
//
// HOST function. Returns the raw CUtensorMap (128-byte POD) to upload to device.
template <int TILE_MN>
inline CUtensorMap sg_build_tma_desc_kmajor(
    const __nv_bfloat16* base, int rows, int K
) {
    using namespace cute;
    // Global tensor: row-major [rows, K], stride (K, 1) — K-contiguous rows.
    auto gtensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<const cute::bfloat16_t*>(base)),
        make_layout(make_shape(rows, K), make_stride(K, 1)));
    // smem layout = the no-swizzle Major-K INTERLEAVE atom tiled to (TILE_MN,16)
    // — the SAME layout make_desc_*_kmajor<TILE_MN, kSwizzleNone> reads, so the
    // landed smem bytes are byte-identical to the cp.async path.
    auto slayout = tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        Shape<Int<TILE_MN>, Int<16>>{});
    auto cta_tiler = Shape<Int<TILE_MN>, Int<16>>{};
    // Cluster_Size = Int<1> ⇒ single-CTA descriptor, no multicast.
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout, cta_tiler, Int<1>{});
    // get_tma_descriptor() returns a CUtensorMap CONST POINTER (&tma_desc_) —
    // deref to copy out the 128-byte POD.
    return *tma.get_tma_descriptor();   // raw 128-byte CUtensorMap POD (by value)
}

// Convenience: build + memcpy into a launcher's POD buffer.
template <int TILE_MN>
inline TmaDescBytes sg_build_tma_desc_bytes(
    const __nv_bfloat16* base, int rows, int K
) {
    CUtensorMap m = sg_build_tma_desc_kmajor<TILE_MN>(base, rows, K);
    TmaDescBytes out;
    static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap must be 128 bytes");
    __builtin_memcpy(out.bytes, &m, 128);
    return out;
}

#endif  // host + CUDA >= 12

}}} // namespace sg::sm90::wgs
```

---

## 4. EDIT `csrc/fused/megakernel_common.cuh` — add the TMA descriptor fields to `PersistentContext`

Trailing, defaulted; OFF builds never read them. Live anchor: megakernel_common.cuh:263-276.

### OLD (megakernel_common.cuh:263-276 — copy VERBATIM)

```cpp
struct PersistentContext {
    int*      g_next_task;     // §1.1 TaskQueue counter
    unsigned* g_arrived;       // §1.4 GridBarrier arrival count
    unsigned* g_generation;    // §1.4 GridBarrier generation
    int       n_tasks;         // parameter tensors this step
    unsigned  n_ctas;          // persistent CTAs (== #SMs, one per SM, §1.3)

    __device__ __forceinline__ TaskQueue queue() const {
        return TaskQueue(g_next_task, n_tasks);
    }
    __device__ __forceinline__ GridBarrier barrier() const {
        return GridBarrier(g_arrived, g_generation, n_ctas);
    }
};
```

### NEW (replace with)

```cpp
struct PersistentContext {
    int*      g_next_task;     // §1.1 TaskQueue counter
    unsigned* g_arrived;       // §1.4 GridBarrier arrival count
    unsigned* g_generation;    // §1.4 GridBarrier generation
    int       n_tasks;         // parameter tensors this step
    unsigned  n_ctas;          // persistent CTAs (== #SMs, one per SM, §1.3)
    // ── CuTe step 4 (SG_TUNED_GEMM_TMA): the device CUtensorMap[] for the fwd/dX
    //    operands + its count. null/0 ⇒ TMA OFF (the producer keeps cp.async).
    //    TRAILING + DEFAULTED so every existing 5-value aggregate-init site
    //    (mega_{decoder,vit,mamba}_real_adamw_tc.cu, dispatch.cpp) stays valid
    //    (the two new members default to null/0) and the by-value struct ABI is a
    //    pure trailing-append: OFF builds never read these, so adding them leaves
    //    the live numerics + A/A/A unchanged. The dispatch.cpp CUDA mirror
    //    (dispatch.cpp:185) MUST be appended in lock-step (Section 5) so the
    //    by-value pass across the launcher TU boundary agrees on layout.
    const void* g_tma_desc = nullptr;   // device CUtensorMap[n_tma_desc]; null ⇒ TMA off
    int         n_tma_desc = 0;

    __device__ __forceinline__ TaskQueue queue() const {
        return TaskQueue(g_next_task, n_tasks);
    }
    __device__ __forceinline__ GridBarrier barrier() const {
        return GridBarrier(g_arrived, g_generation, n_ctas);
    }
};
```

NOTE on byte-identity: adding defaulted trailing members keeps `PersistentContext`
a non-`__device__`-readable POD whose first 5 fields are at the SAME offsets. The
device kernel reads `ctx.g_tma_desc` ONLY inside `#if SG_TUNED_GEMM_TMA` (Section 7),
so an OFF device build's PTX is unchanged (the field exists but is never loaded).
gfx942's `megakernel_common_hip.hip.hpp::PersistentContext` (line 285) is a
SEPARATE struct used only by HIP launchers — NOT edited (gfx942 untouched).

---

## 5. EDIT `csrc/bindings/dispatch.cpp` — append the same two fields to the CUDA mirror

`PersistentContext` is passed BY VALUE from dispatch.cpp into the launcher TUs
(`mega_decoder_real_adamw_tc(::sg::fused::PersistentContext ctx, …)`,
dispatch.cpp:272-273). The dispatch.cpp CUDA mirror (line 185) MUST stay
layout-identical to the real struct (Section 4) or the by-value ABI mismatches.
Append the two fields. Live anchor: dispatch.cpp:185-191.

### OLD (dispatch.cpp:185-191 — copy VERBATIM)

```cpp
struct PersistentContext {
 int* g_next_task;
 unsigned* g_arrived;
 unsigned* g_generation;
 int n_tasks;
 unsigned n_ctas;
};
```

### NEW (replace with)

```cpp
struct PersistentContext {
 int* g_next_task;
 unsigned* g_arrived;
 unsigned* g_generation;
 int n_tasks;
 unsigned n_ctas;
 // CuTe step 4 (SG_TUNED_GEMM_TMA) — MUST mirror megakernel_common.cuh's
 // trailing-appended fields so the by-value PersistentContext passed into the
 // launcher TUs agrees on layout. dispatch.cpp NEVER sets these (its 5-value
 // aggregate-inits leave them default null/0); the launcher TU
 // (launch_fused_decoder_megakernel_tc) populates them under #if SG_TUNED_GEMM_TMA.
 const void* g_tma_desc = nullptr;
 int n_tma_desc = 0;
};
```

NOTE: the `#if defined(WITH_HIP)` gfx942 mirror (dispatch.cpp:367) is NOT edited —
it mirrors the HIP struct and is used only on the HIP build (gfx942 untouched).

---

## 6. EDIT `fused_decoder_megakernel.cuh` — descriptor enumeration + host build + the shared launcher

The fwd/dX operands are: per layer `li`, four B (weight) matrices from the C1
cache + their four C1-T transposed mates, and the per-operand A (acts) tensors.
We enumerate them with a compile-time index so the host build and the device
producer agree on WHICH descriptor each GEMM uses, computed from the SAME carve
math (`dec_acts_bind` / `dec_wbf_bind` / the workspace offset chain).

### 6.1 — A `__host__ __device__` helper for the C1 cache base offset (factored from the kernel carve)

The kernel computes `wbf_f` (the C1 cache base) at fused_decoder_megakernel.cuh:
760-776 via the full carve chain. The host needs the SAME offset. Factor it into
a `__host__ __device__` helper next to `dec_tc_workspace_floats`. Insert AFTER
`dec_tc_workspace_floats`'s closing brace (live anchor: the function ends at
fused_decoder_megakernel.cuh:655, `}` then a blank line then `#ifdef SG_DEC_PROFILE`).

### OLD (fused_decoder_megakernel.cuh:655-657 — copy VERBATIM)

```cpp
}

#ifdef SG_DEC_PROFILE
```

### NEW (replace with)

```cpp
}

// ── CuTe step 4 (SG_TUNED_GEMM_TMA): host/device offset (in FLOATS, from the
//    workspace base) of the C1 bf16 weight cache base `wbf_f`. This is the SAME
//    carve chain the kernel walks at fused_decoder_megakernel_tc (acts + per-CTA
//    scratch + lnvec + loss + dW partials + staged-opt + … → wbf_f), expressed
//    once so the HOST TMA descriptor build (launch_fused_decoder_megakernel_tc)
//    and the DEVICE carve land on the SAME bf16 base. PRODUCTION layout only
//    (SG_DEC_BENCH_LAYOUT=0): mirrors the #else branch at lines 768-776. The host
//    then 16B-aligns the byte address exactly as the kernel does
//    (`while ((uintptr_t)wbf_f & 0xF) wbf_f += 1`). Used ONLY under
//    #if SG_TUNED_GEMM_TMA — on OFF builds this helper is unreferenced and
//    dropped (byte-identical). The acts base is float-offset 0 (ws front).
__host__ __device__ __forceinline__ int64_t dec_tc_wbf_base_floats(int T, int B, int nCTA) {
    // SAME aggregate the workspace size uses up to (but excluding) the wbf cache:
    // acts + per-CTA scratch + lnvec + (nCTA loss parts + 1 reduced loss) +
    // dW split-K partials + staged-opt (Prodigy|Muon|LookSAM|SG2). The SG2 region
    // is INSIDE dec_tc_staged_opt_floats (via dec_tc_sg2_floats, which already
    // carries its +1 align slack), so this is the kernel's float distance from
    // the workspace base to `sg2_ws_base + dec_tc_sg2_floats(nCTA)` == the kernel
    // `wbf_f` PRE-16B-align. The caller re-applies the same 16B byte-align.
    return dec_tc_acts_floats(T, B)
         + (int64_t)nCTA * dectc::dec_tile_scratch_total_f32()
         + (int64_t)nCTA * dectc::kLnVecElems
         + nCTA + 1
         + dec_tc_dw_part_floats()
         + dec_tc_staged_opt_floats(nCTA);
}

#ifdef SG_DEC_PROFILE
```

CRITICAL CORRECTNESS NOTE for the lead: the kernel's `wbf_f` is
`sg2_ws_base + dec_tc_sg2_floats(nCTA)` where `sg2_ws_base = sam_grad + total`
THEN a `+1` 8-byte realign bump, and `dec_tc_sg2_floats` ALREADY includes that
`+1` slack (line 622). The float-offset chain up to (but excluding) the sg2 +1
realign is exactly `acts + per-CTA-scratch + lnvec + loss + dw_part +
staged_opt`. Because `dec_tc_staged_opt_floats` already sums
`opt_reduce+muon+looksam+sg2` and `dec_tc_sg2_floats` carries the `+1`, the
helper above must equal the kernel's pre-16B-align `wbf_f` float offset EXACTLY.
The `* 0` term is a documentation placeholder; **the lead must diff this against
the kernel carve at compile-apply time** by adding a device-side
`assert((uintptr_t)wbf_f == (uintptr_t)ws + dec_tc_wbf_base_floats(T,B,nCTA)*4
 (post-align))` in a debug build, OR (preferred) refactor the kernel's carve
(lines 704-776) to CALL this helper for `wbf_f` so there is a SINGLE source —
see 6.1b. This is the one offset that read-only inspection cannot
byte-verify against the live carve's sg2 `+1`/16B-align bumps; it is gated to
TMA-ON only, so OFF builds are unaffected regardless.

### 6.1b (PREFERRED, removes the duplicate-carve risk) — make the kernel carve call the helper

Rather than duplicate the chain, change the kernel's production `wbf_f` carve to
derive from the helper so host and device are provably identical. Live anchor:
fused_decoder_megakernel.cuh:768-776 (the `#else` production branch).

OLD (fused_decoder_megakernel.cuh:767-776 — copy VERBATIM):

```cpp
#else
    // Production: carve AFTER the (full-size) SG2 region via the existing chain
    // (the sg2_ws_base align bump + the dec_tc_sg2_floats stride are load-bearing
    // for the SG2 int64 row_off64 reads). The bf16 weight pre-stage cache (C1)
    // is interposed here; embed lists stay carve-LAST.
    float* wbf_f = sg2_ws_base + dec_tc_sg2_floats(nCTA);
    // 16B-align the bf16 cache base for the cp.async RING (see the bench-branch
    // note; bump <= 3 floats, covered by dec_tc_workspace_floats' slack term).
    while (((uintptr_t)wbf_f & 0xF) != 0) wbf_f += 1;
    float* embed_ws = wbf_f + dectc::dec_wbf_floats();
#endif
```

NEW:

```cpp
#else
    // Production: carve AFTER the (full-size) SG2 region via the existing chain
    // (the sg2_ws_base align bump + the dec_tc_sg2_floats stride are load-bearing
    // for the SG2 int64 row_off64 reads). The bf16 weight pre-stage cache (C1)
    // is interposed here; embed lists stay carve-LAST.
    float* wbf_f = sg2_ws_base + dec_tc_sg2_floats(nCTA);
    // 16B-align the bf16 cache base for the cp.async RING (see the bench-branch
    // note; bump <= 3 floats, covered by dec_tc_workspace_floats' slack term).
    while (((uintptr_t)wbf_f & 0xF) != 0) wbf_f += 1;
    float* embed_ws = wbf_f + dectc::dec_wbf_floats();
#if (SG_TUNED_GEMM_TMA == 1)
    // CuTe step 4: assert the host TMA-descriptor base helper lands on the SAME
    // bf16 cache base the carve above produced (post-16B-align). This makes the
    // single-source guarantee a COMPILE-once / RUN-once check rather than a
    // read-only claim; it is gated to TMA-ON so OFF builds emit no assert (byte-
    // identical). `ws` is the workspace base local; dec_tc_wbf_base_floats is the
    // pre-align float offset, so re-apply the same 16B bump here.
    {
        float* hb = ws + dec_tc_wbf_base_floats(T, B, nCTA);
        while (((uintptr_t)hb & 0xF) != 0) hb += 1;
        // (debug aid; compiles to nothing under -DNDEBUG)
        assert(hb == wbf_f && "TMA wbf base helper disagrees with kernel carve");
    }
#endif
#endif
```

(`assert` is available in CUDA device code via `<cassert>`/`<assert.h>`, already
transitively included; under `-DNDEBUG` it is a no-op so even a TMA-ON release
build pays nothing. This is the honest closure of the one offset read-only review
cannot fully verify.)

### 6.2 — the operand-descriptor enumeration (compile-time index ↔ operand)

Insert a small enum + count next to the carve helpers (e.g. right before
`launch_fused_decoder_megakernel_tc`, inside `namespace sg::fused::sm90`, under
`#if SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA`). Index layout (per layer `li`,
`L = dec::kLayers`):

```
  B (fwd weights, from the C1 straight section, rows=Nout, K=Kin, TILE_MN=SG_TUNED_TILE_N):
    in_w[li]  : 0*L + li     rows=3d  K=d
    out_w[li] : 1*L + li     rows=d   K=d
    ff0_w[li] : 2*L + li     rows=dff K=d
    ff2_w[li] : 3*L + li     rows=d   K=dff
  B (dX weights, from the C1-T transposed section, rows=Kin, K=Nout, TILE_MN=SG_TUNED_TILE_N):
    in_wT[li] : 4*L + li     rows=d   K=3d
    out_wT[li]: 5*L + li     rows=d   K=d
    ff0_wT[li]: 6*L + li     rows=d   K=dff
    ff2_wT[li]: 7*L + li     rows=dff K=d
  A (acts, from DecActs, rows=T, K=the operand width, TILE_MN=64):
    X_in[li]  : 8*L  + li    K=d
    X_ctx[li] : 9*L  + li    K=d
    X_x1[li]  : 10*L + li    K=d
    X_gact[li]: 11*L + li    K=dff
    dY_qkv[li]: 12*L + li    K=3d
    dY_a[li]  : 13*L + li    K=d
    dY_ff0[li]: 14*L + li    K=dff
    dY_ff2[li]: 15*L + li    K=d
  Total descriptors: kDecTmaNumDesc = 16 * L.
```

Define the constant + index helpers ONCE, in the `dectc` namespace in
`model_stage_decoder_tc.cuh` (so BOTH the launcher in 6.3 and the producer/tile-fn
in Section 7 share the SAME formula — the single host/device agreement the scheme
rests on). Place next to the `SG_DEC_FINE_*` macro block (Section 7.1):

```cpp
#if (SG_TUNED_GEMM_TMA == 1)
// CuTe step 4: number of fwd/dX TMA descriptors = 16 per layer (8 weight + 8
// acts operands; the enumeration in tma_wire.md §6.2). One CUtensorMap each,
// covering all n0 tiles + all k-steps via TMA coordinates. dW is NOT here.
constexpr int kDecTmaNumDesc = 16 * dec::kLayers;
// Compile-time index helpers (host build + device fetch use the SAME formula).
__host__ __device__ __forceinline__ int dec_tma_idx_Bfwd(int kind, int li) { return (kind)     * dec::kLayers + li; } // kind 0..3 = in/out/ff0/ff2
__host__ __device__ __forceinline__ int dec_tma_idx_BdX (int kind, int li) { return (4 + kind) * dec::kLayers + li; } // kind 0..3 = in/out/ff0/ff2 (transposed)
__host__ __device__ __forceinline__ int dec_tma_idx_A   (int kind, int li) { return (8 + kind) * dec::kLayers + li; } // kind 0..7 = X_in,X_ctx,X_x1,X_gact,dY_qkv,dY_a,dY_ff0,dY_ff2
#endif
```

The launcher (6.3) calls these as `dectc::dec_tma_idx_*`; the tile-fn macros (7.4)
call them unqualified (they are already in the `dectc` namespace). There is NO
second copy — 7.4's earlier mention of "model-header-local mirrors" is satisfied
by THESE helpers (one definition, two namespaces of caller).

### 6.2b — make `dec_acts_bind` / `dec_wbf_bind` host-callable (the host build calls them)

`dec_acts_bind` (model_stage_decoder_tc.cuh:441) and `dec_wbf_bind` (528) are
currently `__device__ __forceinline__`. The host descriptor build (6.3) calls them
on the HOST to recover the per-operand bases, so they must become
`__host__ __device__ __forceinline__`. Their bodies are PURE pointer arithmetic
(no device intrinsics, no `__shared__`, no warp ops — verified), so this is a
pure qualifier widening: every existing device caller is unaffected (the device
codegen is byte-identical) and the host gains a callable overload. Two one-token
edits:

OLD (model_stage_decoder_tc.cuh:441):
```cpp
__device__ __forceinline__ DecActs dec_acts_bind(__nv_bfloat16* p, int T, int B) {
```
NEW:
```cpp
__host__ __device__ __forceinline__ DecActs dec_acts_bind(__nv_bfloat16* p, int T, int B) {
```

OLD (model_stage_decoder_tc.cuh:528):
```cpp
__device__ __forceinline__ DecWBf dec_wbf_bind(const __nv_bfloat16* c) {
```
NEW:
```cpp
__host__ __device__ __forceinline__ DecWBf dec_wbf_bind(const __nv_bfloat16* c) {
```

NOTE: these two structs (`DecActs`, `DecWBf`) and their `bind` fns are used on the
device unconditionally — but the `__host__` arm is only ever INSTANTIATED on the
host under `#if SG_TUNED_GEMM_TMA` (the 6.3 build), so an OFF build never emits the
host overload. Even so, widening the qualifier is byte-identical for the device
(the qualifier set is a superset; `__device__` codegen is unchanged). The
`dec_tma_idx_*` helpers in 6.2 and the `dectc_tma_*_idx` mirrors in 7.4 must use
the IDENTICAL index formula so host-built descriptor[i] == the device-fetched
descriptor[i] for every operand.

### 6.3 — the host descriptor build + upload, inside `launch_fused_decoder_megakernel_tc`

This is the SINGLE shared launch entry for ALL host paths (tc_train_step, the real
launcher, sg2 — all call it). Build once per (workspace, T, B, nCTA) into a cached
device buffer, mirroring the existing workspace cache discipline. Insert the build
just BEFORE the `<<<>>>` launch. Live anchor: the `ctx.n_ctas = launch_ctas;` line
through the memset block (fused_decoder_megakernel.cuh:1549-1559).

### OLD (fused_decoder_megakernel.cuh:1547-1559 — copy VERBATIM)

```cpp
    unsigned launch_ctas = (unsigned)n_sms;
    if (ncta_cap > 0 && (unsigned)ncta_cap < launch_ctas) launch_ctas = (unsigned)ncta_cap;
    ctx.n_ctas = launch_ctas;
    // B%16 required (the dW K-loop contracts K=T=B*kSeq and K=B in 16-step atoms,
    // AND it guarantees full token tiles for the projections). NO G-divisibility
    // guard: the split-K dW uses a FLOOR-BALANCED K-partition (dectc_dw_run_tile_
    // splitk) that sums to KS exactly for any KS≥G, so it works at the production
    // truncated B (e.g. 4176, where head KS=B/16=261 is NOT divisible by G=4).
    if ((tok.B % 16) != 0) return cudaErrorInvalidValue;

    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }
```

### NEW (replace with)

```cpp
    unsigned launch_ctas = (unsigned)n_sms;
    if (ncta_cap > 0 && (unsigned)ncta_cap < launch_ctas) launch_ctas = (unsigned)ncta_cap;
    ctx.n_ctas = launch_ctas;
    // B%16 required (the dW K-loop contracts K=T=B*kSeq and K=B in 16-step atoms,
    // AND it guarantees full token tiles for the projections). NO G-divisibility
    // guard: the split-K dW uses a FLOOR-BALANCED K-partition (dectc_dw_run_tile_
    // splitk) that sums to KS exactly for any KS≥G, so it works at the production
    // truncated B (e.g. 4176, where head KS=B/16=261 is NOT divisible by G=4).
    if ((tok.B % 16) != 0) return cudaErrorInvalidValue;

#if (SG_TUNED_GEMM_TMA == 1)
    // ── CuTe step 4: build + upload the fwd/dX CUtensorMap[] over the STEP-STABLE
    //    workspace operand bases, cached per (workspace, T, B, nCTA) like the
    //    workspace itself. OFF (SG_TUNED_GEMM_TMA=0) ⇒ this whole block is
    //    #if-erased; ctx.g_tma_desc stays null and the producer keeps cp.async →
    //    byte-identical. The descriptors are TRANSPORT-ONLY (Major-K INTERLEAVE,
    //    no swizzle) so the wgmma issue order / fp32 accumulation are unchanged.
    {
        const int T = tok.B * dec::kSeq;
        const int nCTA = (int)launch_ctas;
        // Cached device descriptor buffer (process-lived, re-uploaded only when
        // (ws, T, B, nCTA) changes — the same key set as the workspace cache).
        static const float* s_tma_ws = nullptr;   // workspace base the descs were built over
        static int s_tma_T = -1, s_tma_B = -1, s_tma_nCTA = -1, s_tma_dev = -1;
        static CUtensorMap* s_tma_dev_desc = nullptr;  // device array [dectc::kDecTmaNumDesc]
        // NOTE: this launcher is in namespace sg::fused::sm90; the index helpers +
        // count live in the nested dectc namespace, so qualify them `dectc::`.
        using dectc::kDecTmaNumDesc; using dectc::dec_tma_idx_Bfwd;
        using dectc::dec_tma_idx_BdX; using dectc::dec_tma_idx_A;
        const float* ws = tok.workspace;
        if (s_tma_ws != ws || s_tma_T != T || s_tma_B != tok.B ||
            s_tma_nCTA != nCTA || s_tma_dev != dev || s_tma_dev_desc == nullptr) {
            // Recompute the step-stable operand bases from (ws, T, B, nCTA) — the
            // SAME carve the kernel walks. acts base = ws front; C1 cache base =
            // dec_tc_wbf_base_floats (+16B align, like the kernel).
            const __nv_bfloat16* acts_base = reinterpret_cast<const __nv_bfloat16*>(ws);
            float* wbf_f = const_cast<float*>(ws) + dec_tc_wbf_base_floats(T, tok.B, nCTA);
            while (((uintptr_t)wbf_f & 0xF) != 0) wbf_f += 1;
            const __nv_bfloat16* wbf_base = reinterpret_cast<const __nv_bfloat16*>(wbf_f);
            dectc::DecActs A = dectc::dec_acts_bind(const_cast<__nv_bfloat16*>(acts_base), T, tok.B);
            dectc::DecWBf  W = dectc::dec_wbf_bind(wbf_base);
            const int d = dec::kD, dff = dec::kDff, L = dec::kLayers;
            std::vector<sg::sm90::wgs::TmaDescBytes> host_desc(kDecTmaNumDesc);
            constexpr int TN = SG_TUNED_TILE_N;
            for (int li = 0; li < L; ++li) {
                // B fwd weights (C1 straight): rows=Nout, K=Kin, TILE_MN=TN.
                host_desc[dec_tma_idx_Bfwd(0, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<TN>(W.in_w[li],  3*d, d);
                host_desc[dec_tma_idx_Bfwd(1, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<TN>(W.out_w[li],   d, d);
                host_desc[dec_tma_idx_Bfwd(2, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<TN>(W.ff0_w[li], dff, d);
                host_desc[dec_tma_idx_Bfwd(3, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<TN>(W.ff2_w[li],   d, dff);
                // B dX weights (C1-T transposed): rows=Kin, K=Nout, TILE_MN=TN.
                host_desc[dec_tma_idx_BdX(0, li)]  = sg::sm90::wgs::sg_build_tma_desc_bytes<TN>(W.in_wT[li],   d, 3*d);
                host_desc[dec_tma_idx_BdX(1, li)]  = sg::sm90::wgs::sg_build_tma_desc_bytes<TN>(W.out_wT[li],  d, d);
                host_desc[dec_tma_idx_BdX(2, li)]  = sg::sm90::wgs::sg_build_tma_desc_bytes<TN>(W.ff0_wT[li],  d, dff);
                host_desc[dec_tma_idx_BdX(3, li)]  = sg::sm90::wgs::sg_build_tma_desc_bytes<TN>(W.ff2_wT[li], dff, d);
                // A acts: rows=T, K=operand width, TILE_MN=64.
                host_desc[dec_tma_idx_A(0, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<64>(A.X_in[li],   T, d);
                host_desc[dec_tma_idx_A(1, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<64>(A.X_ctx[li],  T, d);
                host_desc[dec_tma_idx_A(2, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<64>(A.X_x1[li],   T, d);
                host_desc[dec_tma_idx_A(3, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<64>(A.X_gact[li], T, dff);
                host_desc[dec_tma_idx_A(4, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<64>(A.dY_qkv[li], T, 3*d);
                host_desc[dec_tma_idx_A(5, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<64>(A.dY_a[li],   T, d);
                host_desc[dec_tma_idx_A(6, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<64>(A.dY_ff0[li], T, dff);
                host_desc[dec_tma_idx_A(7, li)] = sg::sm90::wgs::sg_build_tma_desc_bytes<64>(A.dY_ff2[li], T, d);
            }
            if (s_tma_dev_desc == nullptr) {
                err = cudaMalloc(&s_tma_dev_desc, (size_t)kDecTmaNumDesc * sizeof(CUtensorMap));
                if (err != cudaSuccess) return err;
            }
            err = cudaMemcpy(s_tma_dev_desc, host_desc.data(),
                             (size_t)kDecTmaNumDesc * sizeof(CUtensorMap),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) return err;
            s_tma_ws = ws; s_tma_T = T; s_tma_B = tok.B; s_tma_nCTA = nCTA; s_tma_dev = dev;
        }
        ctx.g_tma_desc = (const void*)s_tma_dev_desc;
        ctx.n_tma_desc = kDecTmaNumDesc;
    }
#endif

    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }
```

This block uses `std::vector`, `cudaMalloc`, `cudaMemcpy`, `CUtensorMap` and
`sg::sm90::wgs::sg_build_tma_desc_bytes` — so the device TU
(`mega_decoder_real_adamw_tc.cu`) must include `cute_tma_desc.h` and `<vector>`
under `#if SG_TUNED_GEMM_TMA` (Section 8). The build is HOST-side
(`launch_fused_decoder_megakernel_tc` is `__host__`), so the `cute::Tensor`
algebra in `cute_tma_desc.h` never enters device code. The descriptors are
re-uploaded when the cache base changes (NOT every step — the per-step weight
re-conversion `dectc_wbf_convert` writes new bytes IN PLACE at the SAME base, and
the descriptor addresses bytes by base+shape, so it stays valid; tma.md §1).

NOTE — the C1 cache holds FRESH per-step bf16 weight values, but at a STEP-STABLE
base. TMA reads live global bytes through the descriptor, so the per-step content
change is irrelevant; the descriptor (base, shape, stride) is unchanged. Same for
the acts region (overwritten each step at the same base by fwd/bwd).

---

## 7. EDIT `model_stage_decoder_tc.cuh` — thread the descriptor to the producer (the `SG_DEC_TMA_*_ARG` idiom)

We thread a per-GEMM descriptor pointer (A-desc, B-desc) + the A row origin `g0`
through forward_tile/backward_tile → dectc_gemm_fwd/dx → tc_gemm_block_unpipelined,
using the EXACT macro idiom the codebase already proves byte-identical-when-OFF
(`SG_DEC_PIPE_BARS_ARG` at fused_decoder_megakernel.cuh:420-424; `SG_DEC_FINE_*`
at model_stage_decoder_tc.cuh:319-325). All new params are TRAILING + DEFAULTED
(`= nullptr` / `= 0`), dereferenced ONLY inside `#if SG_TUNED_GEMM_TMA`.

### 7.1 — the descriptor-arg macros (next to SG_DEC_FINE_*)

Live anchor: model_stage_decoder_tc.cuh:315-325 (the SG_DEC_FINE_* macro block).
The OLD below starts at line 321 (`#define SG_DEC_FINE_FWD , …`) — line 319 is the
`// 9-arg calls …` comment and 320 the `#if defined(SG_DEC_PROFILE) …`; do NOT
include those in the match.

### OLD (model_stage_decoder_tc.cuh:321-325 — copy VERBATIM)

```cpp
#define SG_DEC_FINE_FWD , /*prof_phase=fwd*/ 0
#define SG_DEC_FINE_DX  , /*prof_phase=dX */ 1
#else
#define SG_DEC_FINE_FWD
#define SG_DEC_FINE_DX
#endif
```

### NEW (replace with)

```cpp
#define SG_DEC_FINE_FWD , /*prof_phase=fwd*/ 0
#define SG_DEC_FINE_DX  , /*prof_phase=dX */ 1
#else
#define SG_DEC_FINE_FWD
#define SG_DEC_FINE_DX
#endif

// ── CuTe step 4 (SG_TUNED_GEMM_TMA): the engine takes a trailing (A-desc,B-desc,
//    a_row0) triple so the producer can issue tma_load_kmajor_tile from the host-
//    built CUtensorMap instead of cp.async. The wrappers expand SG_DEC_TMA_ENGINE_*
//    (the engine call's trailing args) and the tile fns expand SG_DEC_TMA_TILE_*
//    (their forwarded params). OFF ⇒ all expand to NOTHING → the calls/signatures
//    are TEXTUALLY UNCHANGED (the trailing defaulted params are unreferenced and
//    dropped → byte-identical PTX, exactly like SG_DEC_PIPE_BARS_ARG). The
//    g_tma_desc array pointer rides through PersistentContext → the tile fns
//    receive it; the wrappers pass the per-operand descriptor POINTERS (indexed
//    into that array) + the A row origin g0.
#if (SG_TUNED_GEMM_TMA == 1)
#define SG_DEC_TMA_TILE_PARAMS , const void* tmaDescA = nullptr, const void* tmaDescB = nullptr, int tmaArow0 = 0
#define SG_DEC_TMA_TILE_FWD    , tmaDescA, tmaDescB, tmaArow0
#else
#define SG_DEC_TMA_TILE_PARAMS
#define SG_DEC_TMA_TILE_FWD
#endif
```

### 7.2 — `tc_gemm_block_unpipelined`: accept the descriptors + route the RS==2 ring through TMA

Add the trailing params to the engine signature (live anchor: the param list at
model_stage_decoder_tc.cuh:709-715), and add a `#if SG_TUNED_GEMM_TMA` TMA staging
path inside the `kRingAsync` / `RS <= 2` branch (the shipped double-buffer ring,
model_stage_decoder_tc.cuh:1010-1038). dW (`!kRingAsync`) and PIPE≥1 deeper rings
are left on cp.async in this first landing (TMA is wired to the SHIPPED S=2 ring,
which is the gate's path; PIPE=1 deeper-ring TMA is a follow-on).

### OLD (model_stage_decoder_tc.cuh:709-715 — copy VERBATIM)

```cpp
template <int N, int MaxAtomsM, typename SrcA, typename SrcB, typename Out>
__device__ void tc_gemm_block_unpipelined(
        int mbase0, int m_atoms, int n_real, int k_steps,
        SrcA srcA, SrcB srcB, Out out,
        __nv_bfloat16* smemA, __nv_bfloat16* smemB,
        unsigned long long* pipeBars = nullptr,
        int prof_phase = -1) {
```

### NEW (replace with)

```cpp
template <int N, int MaxAtomsM, typename SrcA, typename SrcB, typename Out>
__device__ void tc_gemm_block_unpipelined(
        int mbase0, int m_atoms, int n_real, int k_steps,
        SrcA srcA, SrcB srcB, Out out,
        __nv_bfloat16* smemA, __nv_bfloat16* smemB,
        unsigned long long* pipeBars = nullptr,
        int prof_phase = -1
        SG_DEC_TMA_TILE_PARAMS) {
```

Then, inside the engine body, add the TMA stager + a TMA variant of the RS==2
ring. The cleanest minimal change: define a `#if SG_TUNED_GEMM_TMA` constexpr
`kTmaActive` (true only when BOTH descriptors are non-null AND the sources are the
flat-gmem K-major sources) and add a TMA staging lambda + route the prologue/steady
of the `RS <= 2` branch through it. Insert the TMA stager right AFTER the
`stage_k_async` lambda's closing `};` (live anchor: model_stage_decoder_tc.cuh:881,
the `decprim::cp_async_commit(); };` close — match the exact lines).

### OLD (model_stage_decoder_tc.cuh:880-882 — copy VERBATIM)

```cpp
                decprim::cp_async_commit();
            };
            if constexpr (kDecFwdPipeEngine) {
```

### NEW (replace with)

```cpp
                decprim::cp_async_commit();
            };
#if (SG_TUNED_GEMM_TMA == 1)
            // ── CuTe step 4: TMA staging of the group's k-tile `kk` into ring
            //    slot kk%RS. ELECTED single thread issues the kIL A-tile loads +
            //    ONE B-tile load via cute::SM90_TMA_LOAD (cp.async.bulk.tensor.2d)
            //    against the host CUtensorMaps, with one CTA-LOCAL mbarrier that
            //    auto-counts complete_tx::bytes. The destination smem offsets are
            //    the SAME canonical Major-K slots stage_k_async writes (the
            //    descriptor's box is Shape<TILE_MN,16> Major-K INTERLEAVE), so the
            //    landed bytes are byte-identical → wgmma issue order unchanged →
            //    parity / A/A/A preserved. The mbarrier word rides in DecTcSmem
            //    (tma_bar[]; see fused_decoder_megakernel.cuh). A operand crd_mn =
            //    tmaArow0 + (gbase-mbase0) + ai*64 (the GLOBAL token row of the
            //    atom — the descriptor is over the FULL [T,K] tensor, so the per-
            //    tile g0 offset is a coordinate, not a base shift). B operand
            //    crd_mn = n0 (the wrapper bakes n0 into srcB.base, but the TMA
            //    descriptor is over the FULL weight tensor, so we pass the slot's
            //    n0 = mbase-independent column origin via tmaBn0 — see 7.3). For
            //    this first landing we recover n0 from srcB.base − descBbase using
            //    the wrapper-passed tmaBn0 (threaded in 7.3); here n0 == the B
            //    descriptor's row origin for the current N-tile.
            const bool kTmaActive = (tmaDescA != nullptr) && (tmaDescB != nullptr)
                && DecTileSrcIsGmem<SrcA>::value && DecTileSrcIsGmem<SrcB>::value;
            // CTA-local TMA mbarrier (one 8-byte word in smem `red` tail is NOT
            // used — red is fp32 reduction scratch; the kernel passes a dedicated
            // tma_bar via pipeBars-adjacent storage. To avoid widening the param
            // list further, REUSE the existing pipeBars channel ONLY when PIPE!=2:
            // pipeBars is null on the shipped build, so under TMA we instead carve
            // the mbarrier from a NEW DecTcSmem member tma_bar threaded as the
            // FIRST 8 bytes of a dedicated arg — see 7.2-note. For exactness the
            // lead carves DecTcSmem::tma_bar[1] and threads &sm.tma_bar through a
            // trailing param; this comment documents the contract.)
            auto stage_k_tma = [&] (int kk, unsigned long long* tbar, unsigned& parity) {
                const int sl = kk % RS;
                const int kb = kk * wgs::kWgmmaAtomK;
                // tx bytes = kIL A-tiles (g_atoms used) + one B-tile, each
                // TILE_MN*16 bf16 = TILE_MN*32 bytes. Count only g_atoms A-tiles.
                const int a_bytes = g_atoms * (wgs::kWgmmaAtomM * wgs::kWgmmaAtomK * (int)sizeof(__nv_bfloat16));
                const int b_bytes = (N * wgs::kWgmmaAtomK * (int)sizeof(__nv_bfloat16));
                if (wgs::elect_one_sync()) {
                    wgs::Mbarrier(tbar).arrive_expect_tx(a_bytes + b_bytes);
                    for (int ai = 0; ai < g_atoms; ++ai) {
                        wgs::tma_load_kmajor_tile(
                            tmaDescA, tbar,
                            smemA + ((int64_t)sl * kIL + ai) * kDecTcSmemA1,
                            /*crd_mn=*/ tmaArow0 + (gbase - mbase0) + ai * wgs::kWgmmaAtomM,
                            /*crd_k =*/ kb);
                    }
                    wgs::tma_load_kmajor_tile(
                        tmaDescB, tbar,
                        smemB + (int64_t)sl * kDecTcSmemB1,
                        /*crd_mn=*/ n0_for_tma,   // B descriptor row origin (see 7.3)
                        /*crd_k =*/ kb);
                }
                wgs::Mbarrier(tbar).wait(parity); parity ^= 1u;
            };
            (void)stage_k_tma;
#endif
            if constexpr (kDecFwdPipeEngine) {
```

IMPORTANT (honest): the `stage_k_tma` lambda above references two symbols this
spec cannot fully resolve read-only without a compile loop: (1) a dedicated
CTA-local mbarrier word `tbar` — the lead must add `DecTcSmem::tma_bar[1]` (an
`alignas(8) unsigned long long`) and thread `&sm.tma_bar[0]` through a trailing
param exactly like `pipeBars` (a one-line DecTcSmem member + one trailing
param + the `SG_DEC_TMA_*` macro already carries the plumbing); and (2)
`n0_for_tma`, the B operand's column origin (the N-tile's `n0`), which the wrapper
must thread because the engine only sees the already-`n0`-shifted `srcB.base`.
Section 7.3 threads `n0` as `tmaBn0`. Both are mechanical trailing-param adds in
the same macro family; they are flagged here so the lead lands them in the same
edit rather than discovering them at compile. The ENTIRE block is
`#if SG_TUNED_GEMM_TMA`-gated → OFF builds never see it (byte-identical).

Then route the shipped `RS <= 2` ring's staging through TMA when `kTmaActive`.
Live anchor: model_stage_decoder_tc.cuh:1010-1038 (the `else if constexpr (RS <= 2)`
double-buffer ring). The minimal exact change replaces the two `stage_k_async(...)`
calls with a `kTmaActive ? stage_k_tma(...) : stage_k_async(...)` selection. Since
`stage_k_tma` needs the mbarrier+parity and `stage_k_async` does not, the lead
wraps the selection in a small `if (kTmaActive) { … TMA prologue+steady … } else
{ … existing cp.async ring VERBATIM … }`. Because the existing cp.async ring is
preserved VERBATIM in the `else`, and the whole `if (kTmaActive)` arm is inside
`#if SG_TUNED_GEMM_TMA`, OFF builds compile the existing ring byte-identically.

(The exact RS==2 TMA prologue/steady mirrors the cp.async ring structure 1:1 —
stage tile 0 → wgmma_fence → per-k: issue_k + commit, stage k+1, wgmma_wait<0>,
__syncthreads — but with `stage_k_tma` instead of `stage_k_async`+
`cp_async_wait_group<0>`+`fence_async_proxy`, because TMA's mbarrier IS the
completion signal. The wgmma ISSUE SEQUENCE — issue_k(g_atoms,k,RS) ascending k,
k=0 overwrite / k>0 accumulate — is UNCHANGED, which is the parity guarantee.)

### 7.3 — the orientation wrappers thread the descriptor pointers + n0 + g0

`dectc_gemm_fwd` / `dectc_gemm_fwd_f32` / `dectc_gemm_dx_f32` already compute `n0`
in their N-loop and receive the A slice. Add trailing params for the A/B
descriptor pointers + the A row origin (`a_row0`), and pass them + `n0` to the
engine. Live anchor: dectc_gemm_fwd at model_stage_decoder_tc.cuh:1181-1201.

### OLD (model_stage_decoder_tc.cuh:1181-1201 — copy VERBATIM)

```cpp
template <int N>
__device__ __forceinline__ void dectc_gemm_fwd(
        const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
        __nv_bfloat16* __restrict__ Yout, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB, unsigned long long* pipeBars = nullptr) {
    const int k_steps = Kin / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        // A: token rows of X (bf16 acts), K-contiguous. B: rows n0.. of the
        // PRE-STAGED bf16 cache (C1) -- a pure bf16 copy, values bit-identical
        // to the on-read path. Both flat-gmem K-major -> the engine selects the
        // cp.async double-buffered ring (RING); same accessor semantics as the
        // lambdas these replace (incl. the `nn < Nout ? .. : 0` pad guard).
        DecGmemTileSrcA srcA{X, Kin};
        DecGmemTileSrcB srcB{W + (int64_t)n0 * Kin, Kin, Nout - n0};
        auto out  = [&] (int m, int n, float v) {
            Yout[(int64_t)m * Nout + n0 + n] = __float2bfloat16(v); };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB, pipeBars SG_DEC_FINE_FWD);
    }
}
```

### NEW (replace with)

```cpp
template <int N>
__device__ __forceinline__ void dectc_gemm_fwd(
        const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
        __nv_bfloat16* __restrict__ Yout, int Kin, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB, unsigned long long* pipeBars = nullptr
        SG_DEC_TMA_WRAP_PARAMS) {
    const int k_steps = Kin / wgs::kWgmmaAtomK;
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        // A: token rows of X (bf16 acts), K-contiguous. B: rows n0.. of the
        // PRE-STAGED bf16 cache (C1) -- a pure bf16 copy, values bit-identical
        // to the on-read path. Both flat-gmem K-major -> the engine selects the
        // cp.async double-buffered ring (RING); same accessor semantics as the
        // lambdas these replace (incl. the `nn < Nout ? .. : 0` pad guard).
        DecGmemTileSrcA srcA{X, Kin};
        DecGmemTileSrcB srcB{W + (int64_t)n0 * Kin, Kin, Nout - n0};
        auto out  = [&] (int m, int n, float v) {
            Yout[(int64_t)m * Nout + n0 + n] = __float2bfloat16(v); };
        tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/kAtomsM>(
            0, kAtomsM, n_real, k_steps, srcA, srcB, out, sA, sB, pipeBars SG_DEC_FINE_FWD
            SG_DEC_TMA_WRAP_FWD(n0));
    }
}
```

where the new wrapper macros (add to the macro block in 7.1) are:

```cpp
#if (SG_TUNED_GEMM_TMA == 1)
// Wrapper-level: the wrapper receives the A/B descriptor POINTERS + the A row
// origin (a_row0 = g0) from the tile fn, and passes (A-desc, B-desc, a_row0,
// b_n0) to the engine. b_n0 is the N-tile column origin computed in the wrapper
// loop. (The engine's stage_k_tma uses a_row0+atom offset for A's crd_mn and
// b_n0 for B's crd_mn.)
#define SG_DEC_TMA_WRAP_PARAMS , const void* tmaDescA = nullptr, const void* tmaDescB = nullptr, int tmaArow0 = 0
#define SG_DEC_TMA_WRAP_FWD(n0_expr) , tmaDescA, tmaDescB, (tmaArow0 + (n0_expr) - (n0_expr))   /* see note */
#else
#define SG_DEC_TMA_WRAP_PARAMS
#define SG_DEC_TMA_WRAP_FWD(n0_expr)
#endif
```

HONEST NOTE on the `n0`/`tmaBn0` plumbing: the engine's `tc_gemm_block_unpipelined`
needs BOTH the A row origin (`tmaArow0`) AND the B column origin (`n0`). The macro
sketch above passes `tmaArow0` and the descriptors; the B `n0` must ALSO reach the
engine. The clean exact form is to widen `SG_DEC_TMA_TILE_PARAMS` (7.1) to a FOUR-
tuple `(tmaDescA, tmaDescB, tmaArow0, tmaBn0)` and have the wrapper pass `n0` as
`tmaBn0` (it is in scope in the N-loop). The engine's `stage_k_tma` then reads
`n0_for_tma = tmaBn0`. This is the same trailing-defaulted-param add; the macros
in 7.1 + 7.3 must be updated to carry the 4th element. (I keep the 3-tuple in the
primary sketch for readability and flag the required 4th element here so the lead
lands it; both are inside `#if SG_TUNED_GEMM_TMA` → OFF byte-identical.)

Apply the identical pattern to `dectc_gemm_fwd_f32` (model_stage_decoder_tc.cuh:
1206-1222) and `dectc_gemm_dx_f32` (1232-1246): add `SG_DEC_TMA_WRAP_PARAMS` to
the signature and `SG_DEC_TMA_WRAP_FWD(n0)` to the engine call. The wrapper-internal
threading is IDENTICAL for fwd and dX (it just forwards `tmaDescA/tmaDescB/tmaArow0`
+ the loop's `n0`); fwd-vs-dX differs ONLY in WHICH descriptors the tile fn passes
in (the `SG_DEC_TMA_FWD_CALL` vs `SG_DEC_TMA_DX_CALL` index choice, Section 7.4), so
one wrapper macro (`SG_DEC_TMA_WRAP_FWD`) serves all three wrappers. The
fp32-W TP overloads (1258+, 1276+) are NOT TMA-wired (the TP path has no C1 cache /
no descriptors) — leave them on cp.async; they pass NO TMA args (the macro is empty
for them since they are called only from the TP path, never the megakernel).

### 7.4 — the tile fns thread the per-operand descriptor pointers from `ctx.g_tma_desc`

`dectc_forward_tile` / `_backward_tile` receive the descriptor ARRAY base (a
`const void*` = `ctx.g_tma_desc`) as a trailing param, then index it per GEMM call
using the 6.2 index helpers + `li`, and pass the A row origin `g0`. Live anchor:
the forward_tile signature at model_stage_decoder_tc.cuh:1530-1535 and each GEMM
call (1552, 1571, 1591, 1602); backward_tile at 1769-1774 and its dX calls (1868,
1880, 1898, 1909).

Add a trailing param to forward_tile/backward_tile:

```cpp
// forward_tile signature: append  SG_DEC_TMA_TILEFN_PARAMS  (= , const void* tmaDesc = nullptr)
// then each fwd GEMM call appends the per-operand A+B descriptor pointers + g0:
//   in_w  (B fwd kind 0): A = X_in[li]  (A idx 0), B = in_w[li]  (Bfwd idx 0)
//   out_w (B fwd kind 1): A = X_ctx[li] (A idx 1), B = out_w[li] (Bfwd idx 1)
//   ff0_w (B fwd kind 2): A = X_x1[li]  (A idx 2), B = ff0_w[li] (Bfwd idx 2)
//   ff2_w (B fwd kind 3): A = X_gact[li](A idx 3), B = ff2_w[li] (Bfwd idx 3)
```

Define the tilefn macro + a device helper to fetch the i-th descriptor:

```cpp
#if (SG_TUNED_GEMM_TMA == 1)
#define SG_DEC_TMA_TILEFN_PARAMS , const void* tmaDesc = nullptr
// device: pointer to the i-th CUtensorMap in the array base `tmaDesc`.
__device__ __forceinline__ const void* dec_tma_desc_at(const void* base, int i) {
    return base ? (const void*)((const char*)base + (int64_t)i * 128) : nullptr;  // sizeof(CUtensorMap)=128
}
// fwd GEMM call expansion: pass (A-desc, B-desc, a_row0=g0). The wrapper bakes
// the N-tile column origin n0 (tmaBn0); see 7.2/7.3.
#define SG_DEC_TMA_FWD_CALL(a_kind, b_kind, li) \
    , dec_tma_desc_at(tmaDesc, dec_tma_idx_A(a_kind, li)), \
      dec_tma_desc_at(tmaDesc, dec_tma_idx_Bfwd(b_kind, li)), g0
#define SG_DEC_TMA_DX_CALL(a_kind, b_kind, li) \
    , dec_tma_desc_at(tmaDesc, dec_tma_idx_A(a_kind, li)), \
      dec_tma_desc_at(tmaDesc, dec_tma_idx_BdX(b_kind, li)), g0
#else
#define SG_DEC_TMA_TILEFN_PARAMS
#define SG_DEC_TMA_FWD_CALL(a_kind, b_kind, li)
#define SG_DEC_TMA_DX_CALL(a_kind, b_kind, li)
#endif
```

(The `dec_tma_idx_*` are the SINGLE definitions from Section 6.2, in the `dectc`
namespace — the launcher (6.3) calls them as `dectc::dec_tma_idx_*`, the macros
here call them unqualified since they expand inside `dectc` code. One formula,
host + device, which is the descriptor-index agreement the whole scheme rests on.)

Then each fwd GEMM call site appends the macro, e.g. the in_w call (line 1552):

OLD:
```cpp
        dectc_gemm_fwd<SG_TUNED_TILE_N>(Xin, wb.in_w[li], sc.qkv[li], dec::kD, 3 * dec::kD, sA, sB, pipeBars);
```
NEW:
```cpp
        dectc_gemm_fwd<SG_TUNED_TILE_N>(Xin, wb.in_w[li], sc.qkv[li], dec::kD, 3 * dec::kD, sA, sB, pipeBars
            SG_DEC_TMA_FWD_CALL(/*A=X_in*/0, /*B=in_w*/0, li));
```
and analogously: out_w (1571) `SG_DEC_TMA_FWD_CALL(1,1,li)`, ff0_w (1591)
`SG_DEC_TMA_FWD_CALL(2,2,li)`, ff2_w (1602) `SG_DEC_TMA_FWD_CALL(3,3,li)`; the dX
calls in backward_tile: ff2_wT (1868) `SG_DEC_TMA_DX_CALL(/*A=dY_ff2*/7, /*B=ff2_wT*/3, li)`,
ff0_wT (1880) `SG_DEC_TMA_DX_CALL(/*A=dY_ff0*/6, /*B=ff0_wT*/2, li)`, out_wT (1898)
`SG_DEC_TMA_DX_CALL(/*A=dY_a*/5, /*B=out_wT*/1, li)`, in_wT (1909)
`SG_DEC_TMA_DX_CALL(/*A=dY_qkv*/4, /*B=in_wT*/0, li)`.

Finally the kernel's two forward_tile/backward_tile call sites
(fused_decoder_megakernel.cuh:849/852 and 858/861, the SAM and non-SAM arms) append
a `SG_DEC_TMA_TILEFN_ARG` macro = `, ctx.g_tma_desc` (under `#if SG_TUNED_GEMM_TMA`,
else nothing) — identical to how `SG_DEC_PIPE_BARS_ARG` is appended there. Define
it in fused_decoder_megakernel.cuh next to SG_DEC_PIPE_BARS_ARG (line 420):

```cpp
#if (SG_TUNED_GEMM_TMA == 1)
#define SG_DEC_TMA_TILEFN_ARG , ctx.g_tma_desc
#else
#define SG_DEC_TMA_TILEFN_ARG
#endif
```

and append `SG_DEC_TMA_TILEFN_ARG` to each of the four forward_tile/backward_tile
calls (lines 849, 852, 858, 861) AND the second pair in the SAM 2nd-backward
region (lines 1061, 1063) — exactly where `SG_DEC_PIPE_BARS_ARG` already appears.

---

## 8. EDIT the device TU `mega_decoder_real_adamw_tc.cu` — include `cute_tma_desc.h` + `<vector>` under the knob

The host descriptor build (Section 6.3) lives in `launch_fused_decoder_megakernel_tc`
(in the header, instantiated in this TU), so this TU must pull the host builder +
`<vector>`. Add under `#if SG_TUNED_GEMM_TMA`, AFTER the existing includes. Live
anchor: mega_decoder_real_adamw_tc.cu:30 (`#include "…/fused_decoder_megakernel.cuh"`).

### OLD (mega_decoder_real_adamw_tc.cu:30)

```cpp
#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"
```

### NEW

```cpp
#if (SG_TUNED_GEMM_TMA == 1)
#include <vector>
#include "csrc/backends/cuda/sm_90/cute_tma_desc.h"   // HOST CUtensorMap builders (Section 6.3)
#endif
#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"
```

NOTE: `cute_tma_desc.h` must be included BEFORE `fused_decoder_megakernel.cuh`'s
`launch_fused_decoder_megakernel_tc` uses `sg::sm90::wgs::sg_build_tma_desc_bytes`,
so the include order above is required. Apply the same two-line add to the
production launcher TU `mega_decoder_real_adamw_tc_launcher.cu` (its
`#include "…/fused_decoder_megakernel.cuh"` at line 34) ONLY if that TU is ever
compiled with `-DSG_TUNED_GEMM_TMA=1` (per-TU opt-in via setup.py; the gate
compiles `mega_decoder_real_adamw_tc.cu`, so that one is the required edit).

### setup.py (build-system, follow-on): per-TU `-DSG_TUNED_GEMM_TMA=1`

Mirror the existing per-source `-DSG_TUNED_GEMM_IMPL=1` rewrite (the TC TUs already
carry it via the `#define SG_TUNED_GEMM_IMPL 1` at the top of each .cu). To ship
TMA in `_ops`, add `-DSG_TUNED_GEMM_ENGINE=1 -DSG_TUNED_GEMM_TMA=1` to the TC TUs'
nvcc flags (CUTLASS include path is already present on the TC TUs since
WITH_CUTLASS defaults on). This is OPTIONAL for the gate (the gate compiles the TU
directly with the flags); it is the production opt-in.

---

## 9. GATE EXPECTATIONS / WHY IT HOLDS

* **Gate 1** `compile_to_object.sh mega_decoder_real_adamw_tc.cu -DWITH_CUTLASS
  -DSG_TUNED_GEMM_IMPL=1 -DSG_TUNED_GEMM_ENGINE=1 -DSG_TUNED_GEMM_TMA=1`:
  - the two new CuTe TMA includes (Section 1) resolve on the existing
    `-Ithird_party/cutlass/include` line (config.hpp activates
    `CUTE_ARCH_TMA_SM90_ENABLED` for compute_90a);
  - `tma_load_kmajor_tile` / `tma_prefetch_desc` (Section 2) compile against
    `cute::SM90_TMA_LOAD::copy` / `cute::prefetch_tma_descriptor` /
    `cute::TmaDescriptor`;
  - `cute_tma_desc.h` (Section 3) compiles HOST-side in this TU (it includes
    `cute/tensor.hpp` — host algebra only; `make_tma_copy` →
    `cuTensorMapEncodeTiled`);
  - the host build block (Section 6.3) and the device producer (Section 7) compile
    against the threaded descriptor pointers;
  - the `PersistentContext` field add (Sections 4-5) keeps the by-value ABI
    consistent (the TU's own aggregate-init at mega_decoder_real_adamw_tc.cu:105
    stays a 5-value init → the 2 new fields default).
* **Gate 2** `test_decoder_tc.py` (the SHIPPED default build, `SG_TUNED_GEMM_TMA`
  undefined → 0): EVERY device-side change is `#if SG_TUNED_GEMM_TMA`-erased; the
  `PersistentContext` fields are present but never read (no PTX load); the
  `SG_DEC_TMA_*` macros expand to nothing → the producer + tile fns + wrappers are
  TEXTUALLY UNCHANGED. So `wgmma.cuh` / `model_stage_decoder_tc.cuh` /
  `fused_decoder_megakernel.cuh` are byte-identical to the validated path → fp64
  parity + A/A/A pass unchanged.
* **Gate 2 with TMA ON** (the lead injects `-DSG_TUNED_GEMM_ENGINE=1
  -DSG_TUNED_GEMM_TMA=1` into the decoder TU's nvcc flags — the real correctness
  proof of the transport): the descriptor box is no-swizzle Major-K INTERLEAVE
  (Section 3), landing the SAME canonical smem bytes the cp.async ring writes; the
  wgmma issue order (`issue_k` ascending k, k=0 overwrite / k>0 accumulate) is
  UNCHANGED (Section 7) → the fp32 accumulation is bit-identical ⇒ parity / A/A/A
  hold by construction (TMA is a load-timing + load-path reorder, not a math
  change). THIS is the run that proves the wiring; it is the recommended
  additional check below.

### Recommended additional checks (not in the required set, high-value)
* Compile the TU WITHOUT the TMA flag (default 0) to confirm OFF is byte-clean:
  `bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu
  -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1`.
* Inject `-DSG_TUNED_GEMM_ENGINE=1 -DSG_TUNED_GEMM_TMA=1` into the
  `test_decoder_tc.py` JIT build's `extra_cuda_cflags` and re-run — the keystone
  silicon proof that the TMA transport is bit-identical to cp.async (parity + the
  A/A/A determinism triple). If it fails, the suspect is the descriptor box /
  swizzle (Section 0), NOT the wgmma engine.
* `cuobjdump -sass` the TMA-ON object and grep for `UTMALDG` (the TMA bulk-tensor
  load SASS) to confirm the producer actually issues TMA (not a silent cp.async
  fallback).

---

## 10. RISKS (honest)

1. **The `wbf_f` offset helper (6.1)** duplicates the kernel's carve chain; the
   sg2 `+1` realign + the 16B-align bump make a read-only byte-match unprovable.
   MITIGATION: 6.1b refactors the kernel carve to `assert` against the helper
   (TMA-ON only, no-op under `-DNDEBUG`) — a compile-once/run-once check. The
   risk is contained to TMA-ON builds; OFF is untouched.
2. **The B operand `n0` + the CTA-local TMA mbarrier** (`tmaBn0`, `DecTcSmem::
   tma_bar`) are flagged in Section 7.2/7.3 as the two trailing-param adds the lead
   must complete in the same edit (they cannot be fully resolved read-only without
   a compile loop). They are mechanical (one DecTcSmem member, one widened macro
   tuple), all inside `#if SG_TUNED_GEMM_TMA` → OFF byte-identical.
3. **TMA descriptor box vs cp.async smem layout** must match BYTE-FOR-BYTE
   (no-swizzle INTERLEAVE). Section 3 builds it from the SAME
   `GMMA::Layout_K_INTER_Atom<bf16>` the wgmma descriptor reads; the silicon proof
   is Gate-2-with-TMA-ON. If the box ld/stride disagrees, parity fails loud (not
   silent) — the gate catches it.
4. **TMA `crd_mn` for A is a GLOBAL token row** (the descriptor is over the full
   `[T,K]` acts tensor, so the per-tile `g0` is a coordinate, not a base shift).
   The wrapper threads `a_row0=g0` and the engine adds the atom offset
   `(gbase-mbase0)+ai*64`. A wrong `g0` shows as a parity failure (wrong rows
   loaded), again caught by the gate.
5. **dW stays on cp.async** (its transposed-strided gather is not TMA-reachable
   unless `SG_TUNED_DEC_DW_STAGE=1`). This spec does NOT remove the single biggest
   hand-cp.async cost; it makes fwd/dX (the step-stable operands, LEVER ①) run on
   TMA and leaves dW for the DW_STAGE path (tma.md §1 option (a)).
6. **First landing wires TMA to the SHIPPED S=2 ring only** (RS<=2 branch); the
   PIPE=1 deeper ring + the PIPE=2 producer/consumer engine keep cp.async. TMA in
   the deeper rings is a follow-on (the same `stage_k_tma` lambda, more slots).
```
