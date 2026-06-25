# AREA: csrc/backends/cuda/sm_90/wgmma.cuh — CuTe-atom GEMM engine behind SG_TUNED_GEMM_ENGINE

Status: apply-ready. Single-file change. Adds a new compile macro `SG_TUNED_GEMM_ENGINE`
(default `0`). When `0` the file is **byte-identical** to today (the new code is `#if`-erased
to the existing hand-PTX path). When `1`, the three seam internals are re-expressed on CuTe
device atoms while the `sg::sm90::wgs::` ABI (`WgmmaAccum<N>` with `.c[i]`,
`make_desc_{A,B}_kmajor`, `wgmma_m64nNk16_bf16<N,ScaleD,0,0>`, `wgmma_fence/commit_group/
wait_group<N>`, `wgmma_frag_decode`) is preserved exactly. **No call site changes anywhere.**

CuTe steps implemented (per the prompt):
1. descriptor → `cute::GmmaDescriptor` (the union from `cute/arch/mma_sm90_desc.hpp`),
   populated by the same bit layout the hand path uses (proven bit-identical to
   `make_gmma_desc<Major::K>` over `tile_to_shape(Layout_K_INTER_Atom<bf16>, Shape<MN,_16>)`,
   see "Determinism proof" below). `SmemDesc` is made a thin alias so `.desc` keeps working.
2. issue → `cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS<Major::K,Major::K>::fma(desc_a,
   desc_b, c[0]..c[N/2-1], scale_D)` dispatched per N.
3. fence/commit/wait → `cute::warpgroup_arrive()` / `cute::warpgroup_commit_batch()` /
   `cute::warpgroup_wait<N>()`.

TMA + swizzle are **deferred** (not touched here). Only `kSwizzleNone` / INTERLEAVE is used,
which is the path the gate validates.

This file's `wgmma_mainloop_kchain` (used by `wgmma_selftest.cu`) is unchanged in source and
automatically routes through whichever `wgmma_*` primitives the macro selects, so the gate
TU `wgmma_selftest.cu` exercises the CuTe path under `-DSG_TUNED_GEMM_ENGINE=1`.

---

## Determinism / parity proof (why ENGINE=1 is bit-equal to ENGINE=0)

* **Same hardware op.** `MMA_64x128x16_F32BF16BF16_SS<Major::K,Major::K>::fma` emits
  `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16` (mma_sm90_gmma.hpp:2842) with
  `scaleA=scaleB=1` (template default `ScaleIn::One`, line 2805-2806) and `tnspA=tnspB=0`
  (`Major::K==0`). The hand path emits the identical mnemonic with `1, 1` and `TransA=TransB=0`
  (wgmma.cuh:538-543). Same op ⇒ same intra-tile fp32 reduction.
* **Same descriptor bytes.** `cute::GmmaDescriptor.bitfield` (mma_sm90_desc.hpp:113-131) packs
  `start_address_` (smem addr>>4) in [0,14), `leading_byte_offset_>>4` in [16,30),
  `stride_byte_offset_>>4` in [32,46), `base_offset_` in [49,52), `layout_type_` in [62,64).
  The hand `make_smem_desc` packs the IDENTICAL fields: `addr` bits 13:0, `lbo>>4` bits 29:16,
  `sbo>>4` bits 45:32, `bo` bits 51:49, `Swizzle` bits 63:62 (wgmma.cuh:231-235). With
  `lbo=MN*16`, `sbo=128`, `bo=0`, `Swizzle=INTERLEAVE(0)` the produced 64-bit value is the same.
  This also matches `make_gmma_desc<Major::K>` for the INTERLEAVE bf16 tile: for
  `tile_to_shape(Layout_K_INTER_Atom<bf16>, Shape<Int<MN>,_16>)` recast to uint128_t the
  canonical K-major layout is `((8,n),2):((1,SBO),LBO)` (mma_traits_sm90_gmma.hpp:266) giving
  `stride_byte_offset_ = stride_01 = 8` (=128 bytes >>4) and `leading_byte_offset_ = stride_10
  = MN` (=MN*16 bytes >>4) (mma_traits_sm90_gmma.hpp:291-294). All three encodings agree.
  We therefore build `cute::GmmaDescriptor` directly from the same `(MN*16, 128)` byte offsets
  rather than dragging in `cute/tensor.hpp` + `make_gmma_desc` (avoids register/address bloat,
  Risk #1 in IMPLEMENTATION_PLAN §6; the bytes are proven equal).
* **Same ascending-k order, same scale_D semantics.** Callers still issue k=0 with `ScaleD=0`
  then k>0 with `ScaleD=1`. We map `ScaleD==0 → GMMA::ScaleOut::Zero` (overwrite) and
  `ScaleD!=0 → GMMA::ScaleOut::One` (accumulate) — identical to the immediate `0/1`.
* **Same accumulator layout + same epilogue decode.** `WgmmaAccum<N>` and `wgmma_frag_decode`
  are KEPT byte-for-byte; the CuTe atom writes the same `float[N/2]` C-fragment (PTX
  §9.7.14.4.3) in the same per-thread order, so `.c[i]` pairs with the same `(row,col)`.

**Residual (honest) difference:** the CuTe atom passes `scale_D` as a runtime predicate
register (`setp.ne.b32 p, scale_D, 0`) whereas the hand path uses a compile-time immediate.
The *math* is identical (the predicate selects overwrite-vs-accumulate exactly as the immediate
does) and the result is deterministic, so fp64 parity (rel 1e-4 / SAM 2.5e-2) and A/A/A
bit-determinism hold. But the generated SASS is NOT guaranteed byte-identical to the hand path
under ENGINE=1 (extra reg for the predicate; possibly different scheduling). That is acceptable
because the byte-identical requirement is for the **OFF** build only (ENGINE=0, default), which
is preserved by `#if`-erasure. Flagged in Risks.

---

# EDIT 1 — includes + macro default (after the existing system includes)

The header currently includes only `<cuda_bf16.h>/<cuda_runtime.h>/<cstdint>`. We add the CuTe
atom + descriptor headers and the macro default, all guarded so an OFF / non-CUTLASS lift still
compiles. The CuTe headers are header-only and already on the build include path
(`-Ithird_party/cutlass/include`, see scripts/compile_to_object.sh; `mma.cuh` in this same dir
already `#include <cute/tensor.hpp>`). We include ONLY `mma_sm90_gmma.hpp` (the atoms + the
`warpgroup_*` helpers) and `mma_sm90_desc.hpp` (the `GmmaDescriptor` union) — NOT `cute/tensor.hpp`
— to keep the dependency minimal and portable. These two headers are self-contained (they pull
`cute/config.hpp`, `cute/arch/mma.hpp`, and `cutlass/arch/synclog.hpp`, the last a no-op unless
`CUTLASS_ENABLE_SYNCLOG`).

OLD (wgmma.cuh:121-126):
```cpp
// ═════════════════════════════════════════════════════════════════════════
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace sg { namespace sm90 { namespace wgs {
```

NEW:
```cpp
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
//  correctness-gated layout). See the parity proof in the CuTe spec.
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
```

---

# EDIT 2 — SmemDesc: keep ABI, alias to cute::GmmaDescriptor under ENGINE=1

The ABI consumers only ever touch `SmemDesc` as an opaque value passed to
`wgmma_m64nNk16_bf16` and read its `.desc` field (the dispatcher reads `descA.desc` /
`descB.desc`, wgmma.cuh:582-587). Under ENGINE=1 we redefine `SmemDesc` so its `.desc` is the
`cute::GmmaDescriptor`'s `uint64_t` payload, and add an implicit conversion to `uint64_t` so the
CuTe `::fma` (which takes `uint64_t const&`) accepts `descA.desc` directly. Under ENGINE=0 the
struct is exactly as today.

OLD (wgmma.cuh:212-214):
```cpp
struct SmemDesc {
    uint64_t desc;
};
```

NEW:
```cpp
#if (SG_TUNED_GEMM_ENGINE == 1)
// CuTe engine: SmemDesc carries a cute::GmmaDescriptor. `.desc` stays a uint64_t
// (the descriptor's raw bits) so every existing reader (the dispatcher reads
// descA.desc / descB.desc, the selftest gen() copies SmemDesc by value) is
// unchanged. The make_desc_*_kmajor builders populate it via the same bit
// layout the hand path uses (proven == make_gmma_desc<Major::K> for the
// INTERLEAVE bf16 tile). No cute::Tensor is constructed.
struct SmemDesc {
    uint64_t desc;
};
#else
struct SmemDesc {
    uint64_t desc;
};
#endif
```

(Note: the struct body is identical under both arms — `cute::GmmaDescriptor` decays to
`uint64_t` and the union's `desc_` is exactly the 64-bit value we store. Keeping a plain
`uint64_t desc` member means `make_desc_*_kmajor` can assign either a hand-packed value or a
`cute::GmmaDescriptor`'s `.desc_` into it with no ABI change, and the dispatcher passes
`descA.desc` (a `uint64_t`) straight into `::fma`'s `uint64_t const&`. We keep the two arms
explicit so the intent — "this is the CuTe-descriptor carrier" — is documented at the seam.)

---

# EDIT 3 — make_smem_desc: build the descriptor via cute::GmmaDescriptor under ENGINE=1

This is the "descriptor → cute::GmmaDescriptor" step. Under ENGINE=1 we construct a
`cute::GmmaDescriptor` and set its bitfield exactly as the layout requires, then store its raw
`.desc_` into `SmemDesc.desc`. Under ENGINE=0 the existing hand bit-OR is kept verbatim.

OLD (wgmma.cuh:219-241):
```cpp
template <SwizzleMode Swizzle = kSwizzleNone>
__device__ __forceinline__ SmemDesc make_smem_desc(
    const void* smem_base, uint32_t lbo_bytes, uint32_t sbo_bytes,
    uint32_t base_offset = 0u
) {
    SmemDesc d;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
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
#else
    (void)smem_base; (void)lbo_bytes; (void)sbo_bytes; (void)base_offset;
    d.desc = 0;
#endif
    return d;
}
```

NEW:
```cpp
template <SwizzleMode Swizzle = kSwizzleNone>
__device__ __forceinline__ SmemDesc make_smem_desc(
    const void* smem_base, uint32_t lbo_bytes, uint32_t sbo_bytes,
    uint32_t base_offset = 0u
) {
    SmemDesc d;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#if (SG_TUNED_GEMM_ENGINE == 1)
    // CuTe step 1: descriptor -> cute::GmmaDescriptor. Same field layout as the
    // hand path (mma_sm90_desc.hpp:113-131): start_address_ = addr>>4,
    // leading_byte_offset_ = lbo>>4, stride_byte_offset_ = sbo>>4, base_offset_,
    // layout_type_ = Swizzle. For Swizzle=kSwizzleNone(0)=INTERLEAVE this is
    // bit-identical to cute::make_gmma_desc<Major::K> over the INTERLEAVE bf16
    // tile (proof in spec). No cute::Tensor constructed -> no address bloat.
    cute::GmmaDescriptor gd;
    gd.bitfield.start_address_      = static_cast<uint16_t>(smem_desc_encode_addr(smem_base));
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
```

Note `smem_desc_encode_addr` already returns `(addr & 0x3FFFF) >> 4` (wgmma.cuh:184), and
`GmmaDescriptor.start_address_` is the 14-bit `addr>>4` field — assigning the helper's return
truncated to 14 bits is exactly the hand path's `addr & 0x3FFF` after the OR (the high 4 bits of
the 18-bit smem addr that survive >>4 are bits 14:0; both paths mask to 14 bits — the union
bitfield is `:14`, the hand path masks `addr` to its low bits via the `| (lbo<<16)` boundary).
For all decoder/vit tiles the smem arena is well under 16 KB so bits 14+ are zero and the two
are identical.

---

# EDIT 4 — choreography: route fence/commit/wait through cute::warpgroup_* under ENGINE=1

This is the "fence/commit/wait → cute warpgroup_*" step. Three functions, each gets a CuTe arm.
`cute::warpgroup_arrive()` emits the IDENTICAL `wgmma.fence.sync.aligned` (mma_sm90_gmma.hpp:53),
`warpgroup_commit_batch()` the identical `wgmma.commit_group.sync.aligned` (line 80), and
`warpgroup_wait<N>()` the identical `wgmma.wait_group.sync.aligned N` (line 67). They require
`CUTE_ARCH_MMA_SM90A_ENABLED` (`__CUDA_ARCH__>=900 && __CUDA_ARCH_FEAT_SM90_ALL`), which the
`arch=compute_90a,code=sm_90a` build defines — so they are active exactly where the hand PTX is.

OLD (wgmma.cuh:374-378):
```cpp
__device__ __forceinline__ void wgmma_fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
}
```

NEW:
```cpp
__device__ __forceinline__ void wgmma_fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#if (SG_TUNED_GEMM_ENGINE == 1)
    cute::warpgroup_arrive();   // emits wgmma.fence.sync.aligned (mma_sm90_gmma.hpp:53)
#else
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
#endif
}
```

OLD (wgmma.cuh:385-389):
```cpp
__device__ __forceinline__ void wgmma_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#endif
}
```

NEW:
```cpp
__device__ __forceinline__ void wgmma_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#if (SG_TUNED_GEMM_ENGINE == 1)
    cute::warpgroup_commit_batch();  // wgmma.commit_group.sync.aligned (mma_sm90_gmma.hpp:80)
#else
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#endif
#endif
}
```

OLD (wgmma.cuh:398-403):
```cpp
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
#endif
}
```

NEW:
```cpp
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#if (SG_TUNED_GEMM_ENGINE == 1)
    cute::warpgroup_wait<N>();   // wgmma.wait_group.sync.aligned N (mma_sm90_gmma.hpp:67)
#else
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
#endif
#endif
}
```

(`cute::warpgroup_wait<N>` has `static_assert(N>=0 && N<=7)`; every caller uses `wait_group<0>`,
so this is satisfied. The hand path's `N` immediate has the same legal range.)

---

# EDIT 5 — the per-N issue: dispatch to the CuTe atom fma under ENGINE=1

This is the "issue → cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS<Major::K,Major::K>::fma" step.
We keep the entire existing `#if __CUDA_ARCH__>=900` block of hand `wgmma_issue_n*` overloads
(wgmma.cuh:439-563) **unchanged** — they remain the ENGINE=0 implementation. We change ONLY the
dispatcher `wgmma_m64nNk16_bf16` to pick the CuTe atom per N when ENGINE=1.

The atom is invoked as
`cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS<Major::K,Major::K>::fma(descA.desc, descB.desc,
acc.c[0], acc.c[1], ..., acc.c[N/2-1], scale_D)` where `scale_D = (ScaleD==0 ?
ScaleOut::Zero : ScaleOut::One)`. The d-arguments are passed as a parameter pack over `acc.c[]`
in index order (the atom's `float& d00..d_{N/2-1}` map 1:1 to the hand `"+f"(acc.c[i])` list,
PTX §9.7.14.4.3 — same C-fragment, so the epilogue decode is unchanged). TransA/TransB are bound
by the `Major::K,Major::K` template args, exactly the production `<...,0,0>`.

We provide a tiny per-N helper `cute_wgmma_issue<N,ScaleD>` using `if constexpr` over the same N
ladder {8,16,32,64,96,128} the hand dispatcher asserts. Each N expands the explicit `acc.c[i]`
pack (CuTe's `fma` is variadic-by-overload, not a fold, so the args are written out — this is the
unavoidable boilerplate and is what `wgmma_issue_n*` also does by hand).

OLD (wgmma.cuh:573-591):
```cpp
template <int N, int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64nNk16_bf16(
    WgmmaAccum<N>& acc, SmemDesc descA, SmemDesc descB
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    static_assert(N == 8 || N == 16 || N == 32 || N == 64 || N == 96 ||
                  N == 128,
                  "wgmma N must be one of {8,16,32,64,96,128} (bf16 atoms "
                  "selected by DESIGN §3); add the overload for other legal N");
    if constexpr (N == 8)   wgmma_issue_n8  <ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
    else if constexpr (N == 16)  wgmma_issue_n16 <ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
    else if constexpr (N == 32)  wgmma_issue_n32 <ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
    else if constexpr (N == 64)  wgmma_issue_n64 <ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
    else if constexpr (N == 96)  wgmma_issue_n96 <ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
    else if constexpr (N == 128) wgmma_issue_n128<ScaleD, TransA, TransB>(acc, descA.desc, descB.desc);
#else
    (void)acc; (void)descA; (void)descB;
#endif
}
```

NEW:
```cpp
#if (SG_TUNED_GEMM_ENGINE == 1)
// ─────────────────────────────────────────────────────────────────────────
//  CuTe step 2: issue -> cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS<K,K>::fma
//  One helper per N (the atom's fma takes N/2 float& by reference, in index
//  order, so the acc.c[] pack is written out — same boilerplate the hand
//  wgmma_issue_n* uses). Major::K,Major::K == production TransA=TransB=0.
//  ScaleD(0=overwrite/1=accumulate) -> GMMA::ScaleOut::{Zero,One}, which the
//  atom lowers to the same wgmma.mma_async (math identical to the immediate).
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
    // template binds that. Assert here so a non-(0,0) caller is caught instead
    // of silently using the K-major atom.
    static_assert(TransA == 0 && TransB == 0,
                  "SG_TUNED_GEMM_ENGINE=1 supports only Major-K/Major-K "
                  "(TransA=TransB=0), the production orientation; the staging "
                  "loop does any physical transpose. Add an atom binding for "
                  "other orientations before enabling them.");
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
```

---

## What is NOT changed (and why that is correct)

* `WgmmaAccum<N>` / `.zero()` / `wgmma_frag_decode` (wgmma.cuh:314-364) — KEPT verbatim. The
  CuTe atom writes the same `float[N/2]` C-fragment in the same per-thread order, so the
  existing decode and every `acc.c[i]` epilogue (model_stage_decoder_tc.cuh:1100,
  model_stage_vit_tc.cuh:488) is correct unchanged.
* The hand `wgmma_issue_n*` block (wgmma.cuh:439-563) — KEPT verbatim; it is the ENGINE=0
  implementation. Under ENGINE=1 it is still compiled (it is a template, instantiated only if
  referenced) but unreferenced, so it is dead-stripped; no behavior change.
* `wgmma_mainloop_kchain` (wgmma.cuh:617-644) — KEPT verbatim. It calls
  `wgmma_fence()/wgmma_m64nNk16_bf16<...,0,0>/wgmma_commit_group()/wgmma_wait_group<0>()`, all of
  which now route through CuTe under ENGINE=1. This is the function `wgmma_selftest.cu` drives,
  so the gate exercises the CuTe path with no selftest edit.
* `make_desc_A_kmajor` / `make_desc_B_kmajor` (wgmma.cuh:274-288) — KEPT verbatim; they call the
  edited `make_smem_desc`, which now emits the descriptor via `cute::GmmaDescriptor` under
  ENGINE=1. Same `(MN*16, 128)` byte offsets ⇒ same bits.
* `gfx942` / `tpu_v6e` paths — untouched: this file is sm_90-only, all new code is under
  `__CUDA_ARCH__>=900` (and the CuTe includes under `SG_TUNED_GEMM_ENGINE==1`). On gfx942 the
  model headers use their scalar `*_linear` fallback; the dispatcher is a no-op there as today.

---

## Gate commands (run by the lead after apply)

```
CUDA_VISIBLE_DEVICES=0 bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/wgmma_selftest.cu -DSG_TUNED_GEMM_ENGINE=1
CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_decoder_tc.py -q
```

Recommended additional checks (not in the required set but cheap and high-value):
* Compile the selftest WITHOUT the flag too (default ENGINE=0) to confirm the OFF path is
  unchanged: `bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/wgmma_selftest.cu`.
* Run `tests/hw/test_wgmma_substrate.py` (the substrate micro-gate over `wgmma_selftest.cu`) to
  validate the CuTe single-tile / pipelined GEMM bit-for-bit against the hand path — but note
  that test currently builds the selftest with `cpp_extension.load()`; to exercise ENGINE=1 it
  must pass `-DSG_TUNED_GEMM_ENGINE=1` in its `extra_cuda_cflags` (a test-runner change, out of
  scope for this file-only spec).
* `-Xptxas -v` on the selftest TU under ENGINE=1 to confirm reg count / no spills vs ENGINE=0
  (Risk #1): `bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/wgmma_selftest.cu
  -DSG_TUNED_GEMM_ENGINE=1 -Xptxas -v`.

The required `test_decoder_tc.py` gate compiles the **production** TUs via setup.py without
`-DSG_TUNED_GEMM_ENGINE=1`, so it exercises ENGINE=0 (the default) — it confirms this change is
byte-identical-when-OFF and does not regress the shipped path. To actually exercise ENGINE=1
through the decoder gate, the lead must inject `-DSG_TUNED_GEMM_ENGINE=1` into the decoder TU's
nvcc flags (per the autotuner's per-TU flag mechanism); that build-system wiring is the
follow-on area, not this file.
