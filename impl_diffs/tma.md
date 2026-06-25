# APPLY-READY SPEC — CuTe step 4: TMA staging behind `SG_TUNED_GEMM_TMA`

AREA: `csrc/backends/cuda/sm_90/wgmma.cuh` (+ one NEW host-only header
`csrc/backends/cuda/sm_90/cute_tma_desc.h`).

GOAL (as scoped, honestly): introduce a NEW compile knob `SG_TUNED_GEMM_TMA`
(default 0 = current cp.async path, **byte-identical**; requires
`SG_TUNED_GEMM_ENGINE=1` to be meaningful) that adds a **device-callable TMA
load primitive** (`cute::SM90_TMA_LOAD::copy` = `cp.async.bulk.tensor`) into the
substrate, plus the **host-side `CUtensorMap` builder** the persistent
megakernel needs to drive it. TMA is TRANSPORT-ONLY: it changes WHERE/HOW bytes
arrive in smem, never the wgmma's ascending-k fp32 accumulation order, so fp64
parity + A/A/A are preserved bit-for-bit.

---

## 0. WHAT THIS SPEC APPLIES vs WHAT IT DOCUMENTS (read first — honesty)

The two gate commands are:
1. `compile_to_object.sh csrc/backends/cuda/sm_90/wgmma_selftest.cu
   -DSG_TUNED_GEMM_ENGINE=1 -DSG_TUNED_GEMM_TMA=1`
2. `test_decoder_tc.py` (fp64 parity + A/A/A) — runs the shipped default build
   (`SG_TUNED_GEMM_TMA=0`, and on the production TU `SG_TUNED_GEMM_ENGINE`
   default 0 unless the TU defines it).

The **applied** edits below (Sections 2–4) are the concretely gate-bound unit:

* a `SG_TUNED_GEMM_TMA` knob + a self-contained device TMA-load primitive in
  `wgmma.cuh`, written so that **when `SG_TUNED_GEMM_TMA==0` the entire block is
  `#if`-erased → the pre-knob `wgmma.cuh` PTX/smem/regs are unchanged** (passes
  gate 2 by construction: the decoder TU never sets `SG_TUNED_GEMM_TMA`, and the
  default-`SG_TUNED_GEMM_ENGINE=0` shipped build is wholly untouched), and so
  that **when both flags are `1` it compiles cleanly through `wgmma_selftest.cu`**
  (passes gate 1: the primitive depends only on the already-included CuTe
  `SM90_TMA_LOAD` + raw pointers; it adds no new include the selftest lacks and
  is not wired into the selftest's actual GEMM, which keeps cp.async).
* a NEW host-only header `cute_tma_desc.h` (Section 4) giving the
  `CUtensorMap` builders over the step-stable workspace operand bases.

What this spec **documents but does NOT apply as exact edits** (Section 5): the
full wiring of TMA into the megakernel's fwd/dX producer + `PersistentContext` +
the launcher. That wiring is a large multi-file change (PersistentContext field,
every fwd/dX call site, the launcher building+uploading the descriptors) that
cannot be made provably byte-identical-when-OFF across all those sites with the
read-only precision this deliverable requires, and it is NOT needed to pass
either gate. It is specified in prose with precise insertion points so the lead
can land it incrementally (Plan Step 4.2) once the device primitive is gated.

**Why TMA is genuinely the hardest piece (be honest):** CuTe builds
`CUtensorMap` HOST-side only (`cuTensorMapEncodeTiled`, copy_traits_sm90_tma.hpp;
no device encode in v3.6.0 — copy_sm90_desc.hpp only has `tensormap.replace.*`
device modifiers gated on `CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED`). The
persistent megakernel carves its A/B operands from a runtime workspace at step
time. The descriptors must therefore be host-built over STEP-STABLE base pointers
and handed to the kernel grid-constant / via a global buffer. The step-stability
audit below is the load-bearing result that makes this possible at all.

---

## 1. STEP-STABILITY AUDIT (the result that gates everything) — VERIFIED

**The decoder's fwd/dX operand bases ARE step-stable. Confirmed by reading the
live launcher + kernel carve.**

* `tok.workspace` is `cudaMalloc`'d ONCE per device into a `static`
  `DecTcLauncherScratch s` and reused every step; it is reallocated ONLY when a
  larger B needs a bigger workspace (mega_decoder_real_adamw_tc_launcher.cu:55-71,
  `s.ws_floats < need_floats`). For a fixed B it is **byte-stable across steps**.
* `params` is the model param blob, a stable allocation across steps
  (mega_decoder_real_adamw_tc_launcher.cu:103, passed in).
* `T = B*dec::kSeq` and `nCTA = n_sms` are fixed for a fixed B on a fixed device
  (launcher:112,120). Every workspace region offset is a pure function of
  `(T, B, nCTA)` (fused_decoder_megakernel.cuh:704-806).
* The fwd/dX operands:
  - **A = token-row bf16 acts** `DecActs::X_*` / `dY_*` — carved from the FRONT
    of `tok.workspace` (`acts_base = reinterpret_cast<bf16*>(ws)`,
    fused_decoder_megakernel.cuh:707; `dec_acts_bind`,
    model_stage_decoder_tc.cuh:419-439). Flat row-major `[rows, K]`, rows
    K-contiguous, 16B-aligned. **Step-stable** (base = ws, offsets fixed by T,B).
  - **B = the C1/C1-T bf16 weight cache** `DecWBf::{in_w,out_w,ff0_w,ff2_w}` and
    the transposed `*_wT` — carved at `wbf_f` inside the workspace
    (fused_decoder_megakernel.cuh:760-806; `dec_wbf_bind`,
    model_stage_decoder_tc.cuh:506-524). Filled fresh EACH STEP by
    `dectc_wbf_convert` (model_stage_decoder_tc.cuh:532-557) — but **in-place at
    the SAME base address**. TMA reads live global bytes through a base+shape
    descriptor; the content changing per step is irrelevant. **Step-stable base.**

Both fwd and dX operands are the `DecGmemTileSrcA/B` flat-K-major sources
(model_stage_decoder_tc.cuh:610-637) — the exact shape TMA wants (2D tensor,
contraction axis = the K-contiguous minor axis). **TMA can reach fwd + dX.**

**The dW transposed-strided gather is NOT cleanly TMA-reachable (be explicit).**
`dectc_dw_run_tile` / `_splitk` use lambda sources that read the acts TRANSPOSED
(`dY^T·X`, K = T the token axis) — the path the code itself flags "needs
TMA-with-transpose; out of scope" (model_stage_decoder_tc.cuh:782-783). Two
honest options, both LEFT FOR LATER (dW stays on cp.async in this spec):

* (a) If `SG_TUNED_DEC_DW_STAGE=1` is active, `dectc_dw_transpose_operands`
  pre-transposes dY/X into a K-contiguous workspace scratch `dwt_base`
  (fused_decoder_megakernel.cuh:791-796, 910-915) that IS step-stable and flat
  K-major — so dW becomes TMA-reachable EXACTLY like fwd/dX once that flag is on.
* (b) A transpose-encoding `CUtensorMap` (swap box/stride so the token axis is
  the TMA box minor dim) could gather `dY^T`/`X^T` directly. This is the subtle
  part flagged in the plan (§3 risk 3) and is intentionally OUT of this spec's
  scope. **So: TMA scoped to fwd/dX (and, when DW_STAGE is on, the transposed
  scratch); dW's raw transposed gather stays on cp.async.** The dW gather is the
  single biggest hand-cp.async cost; this spec does NOT remove it — it makes it
  removable via path (a) without touching the wgmma engine.

---

## 2. EDIT `wgmma.cuh` — add the `SG_TUNED_GEMM_TMA` knob next to the engine knob

`SG_TUNED_GEMM_TMA` is a SUB-flag of `SG_TUNED_GEMM_ENGINE`: it is only
meaningful when the CuTe engine is selected; it adds the TMA primitive but never
changes any code when 0.

### OLD (wgmma.cuh, the engine-knob block — copy verbatim, lines ~136-148)

```cpp
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
//  transposed-strided gather is NOT covered (stays cp.async) — see the spec.
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
// via cutlass/arch/config.h; config.hpp:36-38).
#include <cute/arch/copy_sm90_tma.hpp>   // cute::SM90_TMA_LOAD::copy (cp.async.bulk.tensor)
#include <cute/arch/copy_sm90_desc.hpp>  // cute::TmaDescriptor (CUtensorMap), prefetch_tma_descriptor
#endif
#endif
```

NOTE: `copy_sm90_desc.hpp` includes `<cuda.h>` (for `CUtensorMap`) under
`!defined(__CUDACC_RTC__)` (copy_sm90_desc.hpp:35-38). `wgmma_selftest.cu` is
compiled by nvcc (not RTC), `-Ithird_party/cutlass/include` is on the line, and
`CUDACC_VER_MAJOR>=12` holds — so the include resolves and `CUtensorMap` is a
real type. The `cutlass/arch/synclog.hpp` that `copy_sm90_tma.hpp` pulls is
header-only and already on the include path (used elsewhere in cutlass).

---

## 3. EDIT `wgmma.cuh` — add the device TMA-load primitive (the new transport)

Insert a `sg::sm90::wgs::` TMA-load helper that is the cp.async analogue: it
stages ONE canonical Major-K tile from global (described by a host-built
`CUtensorMap`) into the SAME smem tile offset the cp.async ring writes, signalling
a caller-owned mbarrier via `expect_tx`. It is a thin wrapper around
`cute::SM90_TMA_LOAD::copy`, taking RAW pointers only (no `wgs::Mbarrier`
dependency, since `wgmma.cuh` does not include `warp_specialize.cuh`); the caller
(tile_pipeline / the decoder producer, which DO have `wgs::Mbarrier`) supplies
the mbarrier's `uint64_t*` and does the `arrive_expect_tx`/`try_wait` handshake.

Place it just AFTER the `cute_wgmma_issue` dispatcher's closing
`#endif  // SG_TUNED_GEMM_ENGINE == 1` and BEFORE the `wgmma_m64nNk16_bf16`
template, so it lives inside the engine-on region.

### OLD (wgmma.cuh — the dispatcher close + the public issue template head; copy verbatim, lines ~687-697)

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
//  This is the TRANSPORT-ONLY replacement for the cp.async staging of ONE
//  operand tile (the A 64×16 or B N×16 half-row scatter the decoder ring does
//  in stage_k_async / produce_ktile, model_stage_decoder_tc.cuh:826-943). It
//  does NOT touch the wgmma issue: after the tile lands in smem, the SAME
//  make_desc_*_kmajor + wgmma_m64nNk16_bf16 run in the SAME ascending-k order,
//  so the fp32 accumulation is bit-identical (parity + A/A/A by construction).
//
//  CONTRACT (the caller owns the mbarrier + the parity handshake; this header
//  stays free of warp_specialize.cuh):
//    * `tma_desc`  — a host-built CUtensorMap over the operand's FULL row-major
//      [rows, K] bf16 tensor (K = the contraction axis, Major-K). Lives in
//      global memory (a device CUtensorMap*); pass &desc. Built by
//      cute_tma_desc.h::sg_build_tma_desc on the host over the STEP-STABLE
//      workspace base (DecActs / DecWBf region). One descriptor per (base,
//      rows, K) — it covers every n0 tile and every k-step via the (crd_mn,
//      crd_k) coordinates below.
//    * `mbar`      — a uint64_t* to a CTA-LOCAL smem mbarrier the CALLER has
//      mbarrier.init'd and on which the CALLER issues
//      Mbarrier::arrive_expect_tx(tile_bytes) BEFORE this call and
//      Mbarrier::try_wait(parity) AFTER (warp_specialize.cuh:98-137). TMA
//      auto-counts the transaction bytes into this mbarrier
//      (mbarrier::complete_tx::bytes) — exactly the handshake cp.async could not
//      self-count (which is why the cp.async ring used wait_group + a plain
//      arrive; TMA uses the real expect_tx).
//    * `smem_tile` — the 16B-aligned smem destination (DecTcSmem.sA/sB slot).
//      The CUtensorMap's box (built host-side as Shape<TILE_MN,16>, Major-K
//      INTERLEAVE = no swizzle) lands the tile in the EXACT canonical Major-K
//      layout idx(mn,k)=(k/8)*(MN*8)+mn*8+(k%8) the cp.async path writes and
//      make_desc_*_kmajor reads — byte-identical smem (de-risks the swap;
//      swizzle is a later knob).
//    * `crd_mn`    — the tile's row origin in the global tensor (= n0 for B, 0
//      for A within a token-tile; the leading-coord of the 2D box).
//    * `crd_k`     — k*16, the k-step's contraction origin (the box's K coord).
//
//  Single-CTA / no cluster: SM90_TMA_LOAD's PTX is shared::cluster-scoped but
//  valid at the implicit cluster-size-1 (the descriptor was built with
//  Cluster_Size=Int<1>); cache_hint 0 = default (EVICT_NORMAL is also fine). No
//  ClusterBarrier, no cooperative launch — composes with the hand GridBarrier.
//  ─────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void tma_load_kmajor_tile(
    const void* tma_desc, unsigned long long* mbar, void* smem_tile,
    int crd_mn, int crd_k
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // cute::SM90_TMA_LOAD::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr,
    //                           crd0, crd1) -> the 2D overload
    // (copy_sm90_tma.hpp:286-292). crd0 is the tensor's leading (row/MN) coord,
    // crd1 the K coord; the box shape (TILE_MN×16) is baked into the descriptor.
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
// TMA over a freshly-uploaded descriptor — copy_sm90_desc.hpp:267-282). The
// caller issues this from one thread after the descriptor is visible in global.
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
(the shipped default and the default of both gate-2 builds) the entire
`#if (SG_TUNED_GEMM_TMA == 1)` block is erased → `wgmma.cuh` is byte-identical to
the pre-knob file. With both flags `1` (gate 1) the primitive compiles: it uses
only `cute::SM90_TMA_LOAD::copy` / `cute::prefetch_tma_descriptor` /
`cute::TmaDescriptor`, all from the two headers added in Section 2, all
`CUTE_HOST_DEVICE`, none requiring `cute::Tensor`. It is NOT referenced by the
selftest's GEMM, so it is a leaf the optimizer keeps only if called — present for
compile, inert for the selftest's numerics (the selftest stays on cp.async).

---

## 4. NEW FILE — `csrc/backends/cuda/sm_90/cute_tma_desc.h` (host-only)

The host-side `CUtensorMap` builder over a step-stable row-major `[rows, K]` bf16
operand, plus a tiny POD the launcher fills and uploads. HOST-ONLY (guarded so a
device/RTC compile of any TU that transitively includes it still compiles to a
no-op). Uses `cute::make_tma_copy` over a `cute::Tensor` of a `gmem_ptr` — the
only place a `cute::Tensor` is constructed, and it is HOST code (no device
register cost). `Cluster_Size=Int<1>` ⇒ single-CTA descriptor (no multicast).

```cpp
#pragma once
// csrc/backends/cuda/sm_90/cute_tma_desc.h
// ─────────────────────────────────────────────────────────────────────────
// CuTe step 4 — HOST-SIDE CUtensorMap builders for the persistent decoder
// megakernel's TMA staging (paired with wgmma.cuh's tma_load_kmajor_tile).
//
// The persistent megakernel carves its A/B operands from a runtime workspace,
// but the bases are STEP-STABLE (tok.workspace is cudaMalloc'd once per B and
// reused; T=B*kSeq and nCTA are fixed for a fixed B/device — see the spec's
// step-stability audit). So the host can compute every fwd/dX operand base from
// (workspace_base, T, B) and build ONE CUtensorMap per (base, rows, K) tensor;
// the descriptor covers all n0 tiles + all k-steps via TMA coordinates.
//
// CuTe builds CUtensorMap host-side ONLY (cuTensorMapEncodeTiled inside
// make_tma_copy; copy_traits_sm90_tma.hpp). There is NO device encode in CuTe
// v3.6.0. The raw 128-byte CUtensorMap is extracted and uploaded to a device
// buffer; the kernel reads it from global (a __grid_constant__ const CUtensorMap
// or a device CUtensorMap* in PersistentContext — see the spec's §5 wiring).
//
// This header is HOST + CUDA only; it includes cute/tensor.hpp (host algebra).
// It is compiled ONLY in the launcher TU (a .cu host function); the megakernel
// device TU does NOT include it (it only needs the device tma_load primitive in
// wgmma.cuh + a const void* descriptor pointer). Guarded so a stray device
// include is inert. SG_TUNED_GEMM_TMA-agnostic at the header level (the launcher
// only CALLS these under #if SG_TUNED_GEMM_TMA).
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

// 128-byte POD wrapper so the launcher can keep an array of descriptors without
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
    // get_tma_descriptor() returns a CUtensorMap CONST POINTER (&tma_desc_,
    // copy_traits_sm90_tma.hpp:116-117) — deref to copy out the 128-byte POD.
    return *tma.get_tma_descriptor();   // raw 128-byte CUtensorMap POD (by value)
}

// Convenience: build + memcpy into the launcher's POD buffer.
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

NOTE on the descriptor SET the launcher would build (fwd/dX only):
* per fwd weight matrix (B operand) over `DecWBf::{in_w,out_w,ff0_w,ff2_w}[L]`,
  rows = Nout, K = Kin, `TILE_MN = SG_TUNED_TILE_N`;
* per dX weight matrix over the transposed `*_wT[L]`, rows = Kin, K = Nout;
* per fwd/dX acts region (A operand) over `DecActs::X_*`/`dY_*`, rows = T,
  K = d/3d/dff, `TILE_MN = 64`.
≈ a few dozen `CUtensorMap`s, keyed on (base,rows,K) and rebuilt only when B/d
changes (mirror the launcher's existing workspace cache,
mega_decoder_real_adamw_tc_launcher.cu:55-71).

---

## 5. WIRING (DOCUMENTED, not applied — Plan Step 4.2; the hardest, multi-file part)

To actually drive TMA from the megakernel's fwd/dX producer (leave dW on
cp.async), the lead lands the following, each gated on `SG_TUNED_GEMM_TMA` so the
OFF build is byte-identical:

1. **`csrc/fused/megakernel_common.cuh` PersistentContext (263-276):** add a
   nullable device descriptor array pointer + count, OFF ⇒ unread:
   ```cpp
   const void* g_tma_desc = nullptr;   // device CUtensorMap[]; null ⇒ TMA off
   int         n_tma_desc = 0;
   ```
   Keep them trailing + defaulted so the wrapped pointer-only launcher boundary
   (launcher.cu) is unchanged; OFF builds never read them.

2. **Launcher (`mega_decoder_real_adamw_tc_launcher.cu`):** under
   `#if SG_TUNED_GEMM_TMA`, after `tok.workspace = sc.workspace;` (line 191),
   recompute the fwd/dX operand bases from `(sc.workspace, T, B, nCTA)` using the
   SAME offset expressions as the kernel carve (factor `dec_acts_bind` /
   `dec_wbf_bind`'s offset math into a `__host__ __device__` helper so host and
   device agree byte-for-byte), call `wgs::sg_build_tma_desc_bytes<...>` per
   region, `cudaMalloc`+`cudaMemcpy` the array into a cached device buffer (keyed
   on (workspace,T,B) like the workspace cache), set
   `ctx.g_tma_desc`/`ctx.n_tma_desc`. Leave the ≥1-CTA/SM cert + counter zeroing
   untouched. OFF ⇒ this whole block is `#if`-erased; `g_tma_desc` stays null.

3. **`fused_decoder_megakernel.cuh` fwd/dX producer:** in
   `tc_gemm_block_unpipelined`'s `kRingAsync` branch (model_stage_decoder_tc.cuh:
   810-1062), under `#if SG_TUNED_GEMM_TMA` route the producer's per-k staging
   through `wgs::tma_load_kmajor_tile(desc, mbar, smem_slot, crd_mn, crd_k)` +
   `Mbarrier::arrive_expect_tx(tile_bytes)` INSTEAD of the `cp_async_cg_16` loop,
   leaving the consumer's `issue_k` + ascending-k commit/wait UNCHANGED. The
   descriptor index + `crd_mn` (= n0 for B, 0 for A) + `crd_k` (= k*16) are passed
   from the orientation wrappers `dectc_gemm_fwd`/`_dx_f32` (which already know
   n0/Kin/Nout). dW (`!kRingAsync`) never enters this branch → stays cp.async.
   OFF ⇒ the cp.async body compiles verbatim (byte-identical, like the existing
   `pipeBars`/`prof_phase` trailing-arg idiom, model_stage_decoder_tc.cuh:679-686).

4. **`setup.py`:** per-TU `-DSG_TUNED_GEMM_TMA=1` only on the TC variant TU(s)
   that opt in (mirror the existing per-source `-DSG_TUNED_GEMM_IMPL=1` rewrite,
   launcher.cu:14-15); CUTLASS include path is already present on the TC TUs.

This wiring is Plan Step 4.2 and is left to the lead because it cannot be made
provably byte-identical-when-OFF across all those sites with read-only precision,
and it is not required to pass either gate command.

---

## 6. GATE EXPECTATIONS / WHY IT HOLDS

* **Gate 1** `compile_to_object.sh wgmma_selftest.cu -DSG_TUNED_GEMM_ENGINE=1
  -DSG_TUNED_GEMM_TMA=1`: the two new CuTe TMA includes resolve on the existing
  `-Ithird_party/cutlass/include` line (config.hpp activates
  `CUTE_ARCH_TMA_SM90_ENABLED` for compute_90a); `tma_load_kmajor_tile` /
  `tma_prefetch_desc` compile against `cute::SM90_TMA_LOAD::copy` /
  `cute::prefetch_tma_descriptor` / `cute::TmaDescriptor`. The selftest's GEMM is
  unchanged (still cp.async), so adding these leaf helpers does not alter its
  numerics. `cute_tma_desc.h` is NOT included by the selftest (only by the
  launcher), so its host-only `cute::Tensor` algebra never enters this compile.
* **Gate 2** `test_decoder_tc.py`: the shipped decoder build does not define
  `SG_TUNED_GEMM_TMA` (default 0) — the TMA block is fully `#if`-erased, so
  `wgmma.cuh` is byte-identical to the validated CuTe-atom path (which is itself
  bit-identical to the hand path, per the existing steps-1-3 validation). fp64
  parity + A/A/A pass unchanged. Even with `SG_TUNED_GEMM_ENGINE=1` the TMA knob
  defaults 0, so no transport changes; with TMA wired (Section 5) the smem bytes
  are byte-identical (no-swizzle INTERLEAVE) and the wgmma issue order is
  unchanged ⇒ parity/A/A/A hold by construction (TMA is a load-timing reorder).
