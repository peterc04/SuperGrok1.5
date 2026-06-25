# CuTe Atoms + wgmma.cuh — Ground Truth Digest

**Area:** CuTe/CUTLASS atom consumption in the SuperGrok1.5 sm_90 GEMM substrate  
**Source files read:** wgmma.cuh (776 lines), 5 cute_plan digests (CUTE_GEMM_SYNTHESIS.md,  
IMPLEMENTATION_PLAN.md, INVENTORY.md, WGMMA_ENGINE_SEAM.md, GEMM_CALL_SITES.md, mma_cuh_and_build_seam.md),  
cutlass/version.h (confirmed v3.6.0)

---

## 1. CUTLASS Version — CONFIRMED

`/workspace/SuperGrok1.5/third_party/cutlass/include/cutlass/version.h`:
```
#define CUTLASS_MAJOR 3
#define CUTLASS_MINOR 6
#define CUTLASS_PATCH 0
```
Claim: v3.6.0. Status: **VERIFIED**.

---

## 2. wgmma.cuh — What It Is and Does (line-by-line)

**File:** `csrc/backends/cuda/sm_90/wgmma.cuh` (776 lines)  
**Namespace:** `sg::sm90::wgs::`  
**Role:** The ONLY wgmma (GEMM) path inside the persistent megakernel. Explicitly rejects CUTLASS host-launched CollectiveMma (wgmma.cuh:16-18). Self-contained: includes ONLY `<cuda_bf16.h>`, `<cuda_runtime.h>`, `<cstdint>` in the default path (wgmma.cuh:122-124).

### 2.1 The Dual-Engine Architecture (SG_TUNED_GEMM_ENGINE)

wgmma.cuh implements a TWO-ENGINE design gated by `SG_TUNED_GEMM_ENGINE`:
- **0 (DEFAULT):** Hand-rolled inline-PTX ss-wgmma. The byte-identical correctness path.
- **1:** CuTe device atoms (`cute::GmmaDescriptor` + `SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS::fma` + `cute::warpgroup_*`). Gated `#if (SG_TUNED_GEMM_ENGINE == 1)` throughout.

When ENGINE=1, two CuTe headers are included (wgmma.cuh:146-148):
```cpp
#include <cute/arch/mma_sm90_desc.hpp>   // cute::GmmaDescriptor, SM90::GMMA::LayoutType
#include <cute/arch/mma_sm90_gmma.hpp>   // cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS, warpgroup_*
```
**Note:** Only the GMMA atoms/warpgroup helpers and the descriptor union are pulled in — NOT `cute/tensor.hpp` — to avoid CuTe Tensor/Layout address-algebra register bloat (wgmma.cuh:143-148).

### 2.2 Core ABI Types

**`WgmmaAccum<N>` (wgmma.cuh:355-365):**
- `static constexpr int kRegs = N/2;  float c[kRegs];`
- Per-thread fp32 accumulator for m64×N output tile. 128 warpgroup threads together hold the 64×N tile.
- `static_assert(N % 8 == 0)` at compile time.
- `.zero()` clears all lanes.

**`SmemDesc` (wgmma.cuh:236-238):**
- `struct SmemDesc { uint64_t desc; };`
- 64-bit smem matrix descriptor per PTX ISA §<ip>.4.

**`SwizzleMode` (wgmma.cuh:191-196):**
```
kSwizzleNone = 0   // INTERLEAVE / no swizzle — the correctness-path layout
kSwizzle128B = 1   // SWIZZLE_128B
kSwizzle64B  = 2   // SWIZZLE_64B
kSwizzle32B  = 3   // SWIZZLE_32B
```
Matches the CuTe `LayoutType` enum in `mma_sm90_desc.hpp:53-57` exactly (INTERLEAVE=0, B128=1, B64=2, B32=3).

### 2.3 Descriptor Construction

**`make_smem_desc<Swizzle>(smem_base, lbo_bytes, sbo_bytes, base_offset)` (wgmma.cuh:243-282):**

Under ENGINE=0 (hand path, wgmma.cuh:265-276):
```cpp
uint64_t addr = (__cvta_generic_to_shared(smem_ptr) & 0x3FFFFu) >> 4;
d.desc = addr | (lbo<<16) | (sbo<<32) | (bo<<49) | (sw<<62);
```

Under ENGINE=1 (CuTe path, wgmma.cuh:251-263):
```cpp
cute::GmmaDescriptor gd;
gd.bitfield.start_address_       = smem_desc_encode_addr(smem_base);
gd.bitfield.leading_byte_offset_ = (lbo_bytes >> 4) & 0x3FFFu;
gd.bitfield.stride_byte_offset_  = (sbo_bytes >> 4) & 0x3FFFu;
gd.bitfield.base_offset_         = base_offset & 0x7u;
gd.bitfield.layout_type_         = Swizzle & 0x3u;
d.desc = gd.desc_;
```

This is **bit-identical** for the production NONE-swizzle path — verified by comment at wgmma.cuh:255-256 citing `cute::make_gmma_desc<Major::K>` over the INTERLEAVE bf16 tile.

**`make_desc_A_kmajor<MN, Swizzle>(smem)` / `make_desc_B_kmajor<MN, Swizzle>(smem)` (wgmma.cuh:315-329):**
- Both set: `lbo = MN * 16u;` (bytes), `sbo = 128u;` (bytes)
- Canonical Major-K INTERLEAVE layout: `idx(mn,k) = (k/8)*(MN*8) + mn*8 + (k%8)`
- K=16 split as two 8-wide core matrices; SBO=128, LBO=MN*16
- This is exactly `GMMA::Layout_K_INTER_Atom<bf16>` in CuTe terms.

### 2.4 Choreography Wrappers

**wgmma_fence() (wgmma.cuh:415-423):**
- ENGINE=0: `asm volatile("wgmma.fence.sync.aligned;" ::: "memory");`
- ENGINE=1: `cute::warpgroup_arrive();`

**wgmma_commit_group() (wgmma.cuh:430-438):**
- ENGINE=0: `asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");`
- ENGINE=1: `cute::warpgroup_commit_batch();`

**`wgmma_wait_group<N>()` (wgmma.cuh:447-456):**
- ENGINE=0: `asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");`
- ENGINE=1: `cute::warpgroup_wait<N>();`

### 2.5 CuTe Atom Dispatch (`cute_wgmma_issue`, ENGINE=1 only, wgmma.cuh:634-691)

```cpp
template <int N, int ScaleD>
__device__ __forceinline__ void cute_wgmma_issue(WgmmaAccum<N>& acc, uint64_t descA, uint64_t descB) {
    namespace G = cute::SM90::GMMA;
    constexpr G::ScaleOut sd = (ScaleD == 0) ? G::ScaleOut::Zero : G::ScaleOut::One;
    if constexpr (N == 8)
        G::MMA_64x8x16_F32BF16BF16_SS<G::Major::K, G::Major::K>::fma(descA, descB, acc.c[0..3], sd);
    else if constexpr (N == 16) ...
    else if constexpr (N == 32) ...
    else if constexpr (N == 64) ...
    else if constexpr (N == 96) ...
    else if constexpr (N == 128)
        G::MMA_64x128x16_F32BF16BF16_SS<G::Major::K, G::Major::K>::fma(
            descA, descB, acc.c[0]..acc.c[63], sd);
}
```

**Critical constraint at wgmma.cuh:703-707:** ENGINE=1 enforces `static_assert(TransA == 0 && TransB == 0)` — CuTe atoms only support Major::K/Major::K (TransA=TransB=0). This matches production use (all model stages call with 0,0).

### 2.6 Fragment Decode

**`wgmma_frag_decode(tid_in_wg, i, N, &row, &col)` (wgmma.cuh:384-405):**
- Single authoritative thread→(row,col) decode shared by epilogue + gate.
- Formula: `slab = i>>2; within = i&3; row_half = within>>1; col_par = within&1`
  `row = 16*warp + (lane>>2) + 8*row_half; col = (lane&3)*2 + col_par + 8*slab`
- **NOT replaced** in ENGINE=1 — kept as-is because CuTe C-fragment layout is identical.

### 2.7 K-Chain Mainloop (wgmma.cuh:746-773)

```cpp
template <int N, int TransA, int TransB, bool FenceBeforeFirst=true, typename DescGen>
__device__ void wgmma_mainloop_kchain(WgmmaAccum<N>& acc, int k_steps, DescGen gen) {
    if (FenceBeforeFirst) wgmma_fence();
    // k=0: ScaleD=0 (overwrite)
    wgmma_m64nNk16_bf16<N, 0, TransA, TransB>(acc, gen(0).first, gen(0).second);
    // k>0: ScaleD=1 (accumulate)
    for (int k = 1; k < k_steps; ++k)
        wgmma_m64nNk16_bf16<N, 1, TransA, TransB>(acc, gen(k).first, gen(k).second);
    wgmma_commit_group(); wgmma_wait_group<0>();
}
```
Fixed ascending-k, k=0 overwrite / k>0 accumulate — the determinism contract.

### 2.8 Per-N Overloads (Hand Path, wgmma.cuh:493-616)

Explicit inline-asm per N in {8, 16, 32, 64, 96, 128}, listing exactly N/2 `"+f"` accumulator registers. All have identical PTX form:
```
wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 {N/2 regs}, descA, descB, ScaleD, 1, 1, TransA, TransB
```
All are guarded `#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)`.

---

## 3. CuTe Atom Inventory (from CUTLASS v3.6.0 headers)

### 3.1 MMA Atoms Available

All in `cute/arch/mma_sm90_gmma.hpp`. The atoms consumed/referenced:

| N | Struct | Line |
|---|--------|------|
| 8  | `MMA_64x8x16_F32BF16BF16_SS<Major,Major>` | ~2247 |
| 16 | `MMA_64x16x16_F32BF16BF16_SS<Major,Major>` | ~2338 |
| 32 | `MMA_64x32x16_F32BF16BF16_SS<Major,Major>` | ~2433 |
| 64 | `MMA_64x64x16_F32BF16BF16_SS<Major,Major>` | ~2538 |
| 96 | `MMA_64x96x16_F32BF16BF16_SS<Major,Major>` | ~2663 |
| 128| `MMA_64x128x16_F32BF16BF16_SS<Major,Major>` | ~2808 |

All guarded on `CUTE_ARCH_MMA_SM90A_ENABLED` (`__CUDA_ARCH__>=900 && __CUDA_ARCH_FEAT_SM90_ALL`), **requiring `-arch=sm_90a` (not just sm_90)** — stricter than the hand path's `__CUDA_ARCH__>=900`.

### 3.2 Warpgroup Choreography (mma_sm90_gmma.hpp)

| CuTe function | PTX emitted | wgmma.cuh equivalent |
|---------------|-------------|-----------------------|
| `warpgroup_arrive()` (line ~47) | `wgmma.fence.sync.aligned` | `wgmma_fence()` |
| `warpgroup_commit_batch()` (line ~75) | `wgmma.commit_group.sync.aligned` | `wgmma_commit_group()` |
| `warpgroup_wait<N>()` (line ~61) | `wgmma.wait_group.sync.aligned N` | `wgmma_wait_group<N>()` |
| `warpgroup_fence_operand(frg)` (line ~88) | accumulator visibility | supplemental |

**CuTe caps `warpgroup_wait<N>` at N≤7** — the hand version has no such cap. Production uses wait_group<0> only so this is not an issue.

### 3.3 GmmaDescriptor (mma_sm90_desc.hpp:85-136)

Bitfield layout:
- `start_address_:14` (bits 13:0)
- `leading_byte_offset_:14` (bits 29:16)
- `stride_byte_offset_:14` (bits 45:32)
- `base_offset_:3` (bits 51:49)
- `layout_type_:2` (bits 63:62)

`LayoutType` encoding (mma_sm90_desc.hpp:53-57): `INTERLEAVE=0, B128=1, B64=2, B32=3` — matches `SwizzleMode` exactly. The bit-identical mapping is confirmed in wgmma.cuh's comment block (wgmma.cuh:174-180).

### 3.4 Smem Layout Atoms (mma_traits_sm90_gmma.hpp:74-131)

- `Layout_K_INTER_Atom<bfloat16_t>` (`Swizzle<0,4,3>`) — the NONE-swizzle correctness path. **Byte-identical to the hand-rolled `idx(mn,k)=(k/8)*(MN*8)+mn*8+(k%8)` layout** (LBO=MN*16, SBO=128).
- `Layout_K_SW128_Atom<bfloat16_t>` (`Swizzle<3,4,3>`) — 128B-swizzle perf target (flagged TODO in wgmma.cuh:183-190).

---

## 4. The TMA Host-Only-Encode Blocker

### 4.1 Status: CONFIRMED BLOCKER

`make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout, cta_tiler, Int<1>{})` is `CUTE_HOST_RTC` (copy_traits_sm90_tma.hpp:1266-1308). It calls the CUDA driver `cuTensorMapEncodeTiled` at copy_traits_sm90_tma.hpp:1008. This is **host-only** (guarded `!__CUDACC_RTC__`).

### 4.2 Current Device Path Has ZERO TMA

From `mma_cuh_and_build_seam.md` (verified by code structure): grep for `cuTensorMapEncode|CUtensorMap|make_tma_copy|cp.async.bulk` in `csrc/` returns ONLY doc comments. The persistent megakernel stages operands with **Ampere `cp.async.cg`** (`cp.async.cg.shared.global`, `primitives.cuh:497-508`), NOT TMA.

TMA is explicitly deferred:
- `tile_pipeline.cuh:199`: "this uses cp.async (.cg.16), NOT TMA — TMA is explicitly phase-2"
- `model_stage_decoder_tc.cuh:782-783`: "needs TMA-with-transpose; out of scope"

### 4.3 Workaround Design

The plan requires building `CUtensorMap` descriptors host-side (in the megakernel launcher `mega_decoder_real_adamw_tc_launcher.cu`) and passing them to the device via one of:
1. `__grid_constant__ const CUtensorMap` kernel params (preferred — lives in param/const bank)
2. Device global memory pointer in `PersistentContext` (nullable; OFF path ignores it)

The `PersistentContext` (megakernel_common.cuh:263-276) would gain: `const void* g_tma_desc = nullptr; int n_tma_desc = 0;`

---

## 5. The Production Seam: What Calls wgmma.cuh

### 5.1 Central Engine

`tc_gemm_block_unpipelined<N,MaxAtomsM,SrcA,SrcB,Out>` at `model_stage_decoder_tc.cuh:687-1130` is the ONE place all wgmma issues happen for the decoder. Key call site (wgmma.cuh consumption):

**`issue_k` lambda (model_stage_decoder_tc.cuh:750-762):**
```cpp
dB = make_desc_B_kmajor<N>(smemB + sl*kDecTcSmemB1);
for (ai in 0..kIL-1):
    dA = make_desc_A_kmajor<64>(smemA + (sl*kIL+ai)*kDecTcSmemA1);
    wgmma_m64nNk16_bf16<N, k==0?0:1, 0, 0>(acc[ai], dA, dB);
```

TransA=TransB=0 ALWAYS (both K-major). N=128 production default.

### 5.2 Model Coverage

All 3 flagship models consume the same `sg::sm90::wgs::` ABI:
- Decoder: `model_stage_decoder_tc.cuh:759-760`
- ViT: `model_stage_vit_tc.cuh:470-472`
- Mamba: `model_stage_mamba_tc.cuh` (same pattern)

### 5.3 Engine Selection in wgmma_m64nNk16_bf16 (wgmma.cuh:693-720)

```cpp
template <int N, int ScaleD, int TransA, int TransB>
__device__ void wgmma_m64nNk16_bf16(WgmmaAccum<N>& acc, SmemDesc descA, SmemDesc descB) {
    // static_assert N in {8,16,32,64,96,128}
#if (SG_TUNED_GEMM_ENGINE == 1)
    static_assert(TransA==0 && TransB==0);
    cute_wgmma_issue<N, ScaleD>(acc, descA.desc, descB.desc);
#else
    if constexpr (N==8)   wgmma_issue_n8<ScaleD,TransA,TransB>(acc, ...);
    else if constexpr ... // dispatches by N to hand-asm overloads
#endif
}
```

---

## 6. Key Facts / Claims vs Ground Truth

### 6.1 Claimed: SG_TUNED_GEMM_ENGINE exists and ENGINE=1 uses CuTe atoms
**VERIFIED.** wgmma.cuh:127-148 defines the knob; wgmma.cuh:140-148 includes the CuTe headers conditionally; wgmma.cuh:251-263, 417-418, 431-433, 449-450, 638-691 implement ENGINE=1 paths.

### 6.2 Claimed: Bit-identical between ENGINE=0 and ENGINE=1 for kSwizzleNone
**VERIFIED BY DESIGN.** The CuTe path in `make_smem_desc` (ENGINE=1) fills `cute::GmmaDescriptor.bitfield.*` fields with identical values to the hand-rolled bit packing (wgmma.cuh:255-263). The GMMA atom emits the same `wgmma.mma_async` PTX. The fragment layout is identical (PTX §<ip>.3 applied the same way). The implementation explicitly claims bit-identity (wgmma.cuh:255-256: "For Swizzle=kSwizzleNone(0)=INTERLEAVE this is bit-identical to cute::make_gmma_desc<Major::K> over the INTERLEAVE bf16 tile (proof in /workspace/cute_plan)").

### 6.3 Claimed: CUTLASS v3.6.0
**VERIFIED.** cutlass/version.h: MAJOR=3, MINOR=6, PATCH=0.

### 6.4 Claimed: TMA is "host-only-encode" blocker in CuTe v3.6.0
**VERIFIED.** copy_traits_sm90_tma.hpp:1008 calls `cuTensorMapEncodeTiled` guarded on `!__CUDACC_RTC__`. No device-side tensormap encode found. The megakernel currently has no TMA anywhere.

### 6.5 Claimed: warpgroup_wait<N> in CuTe caps N≤7
**INVENTORY.md confirms** (mma_sm90_gmma.hpp:62-65). Production uses wait_group<0> only; not a practical blocker.

### 6.6 CuTe atoms require sm_90a (not just sm_90)
**INVENTORY.md confirms:** `CUTE_ARCH_MMA_SM90A_ENABLED` requires `__CUDA_ARCH_FEAT_SM90_ALL`. The hand-rolled path guards on `__CUDA_ARCH__ >= 900` only — stricter requirement when ENGINE=1.

---

## 7. Open Items / Blockers

1. **ENGINE=1 is not the default** (SG_TUNED_GEMM_ENGINE defaults to 0, wgmma.cuh:137). The CuTe path exists in source but is **never shipped in production binaries**.

2. **TMA is entirely absent from the device path.** No `CUtensorMap`, no `cp.async.bulk` anywhere in csrc/ device code. The full TMA plan (IMPLEMENTATION_PLAN.md steps 4.1-4.5) is a synthesis/plan document, not implemented code. This is the biggest gap between the plan and ground truth.

3. **dW path cannot use cp.async ring** (model_stage_decoder_tc.cuh:782-783: "needs TMA-with-transpose; out of scope"). TMA is the fix; the fix is not implemented.

4. **sm_90a vs sm_90 compile target** — ENGINE=1 requires `-arch=sm_90a`; the build must be verified to emit this for megakernel TUs (setup.py:683-687 emits `sm_90a` when CUTLASS enabled, but the megakernel TUs currently DON'T depend on CUTLASS).

5. **CuTe register pressure risk** — `cute/tensor.hpp` is explicitly excluded from the ENGINE=1 include (wgmma.cuh:143-144: "to avoid CuTe Tensor/Layout address-algebra register bloat"). The plan notes the `cudaOccupancyMaxActiveBlocks≥1` cert as the safety net if CuTe bloats registers below 1 CTA/SM.

6. **Swizzle is TODO** — 128B-swizzle (`kSwizzle128B`) is exposed in the `SwizzleMode` enum but the correctness path ships only `kSwizzleNone`. The swizzle+stager coupling (both must change atomically: descriptor layout_type AND the staging loop `stage_kmajor_tile` at model_stage_decoder_tc.cuh:571-582) is explicitly flagged as a perf-phase step (wgmma.cuh:183-190).

7. **`cute_gemm.cuh` and `cute_tma_desc.h` do not exist yet.** IMPLEMENTATION_PLAN.md describes them as "NEW" files to be created. They are not present in the repository.

8. **PersistentContext has no TMA fields yet.** megakernel_common.cuh:263-276 does not contain `g_tma_desc` / `n_tma_desc`. The plan describes adding them.

---

## 8. Config-Derivation / Adaptivity in This Layer

The wgmma.cuh layer participates in adaptivity via the SG_TUNED_* knob system:
- `SG_TUNED_GEMM_ENGINE` (0/1): engine selection
- `SG_TUNED_TILE_N` (default 128): N dimension of the wgmma atom; legal set {64,128,256} mentioned in comments (wgmma.cuh:95)
- `SG_TUNED_TILE_M` (default 128): M-tile = TILE_M/64 stacked atoms per CTA

These are `#ifndef`-guarded defaults (wgmma.cuh:156-162), so they compose correctly without redefinition. The kernel autotuner can inject `-DSG_TUNED_*` per-TU. The N selection is a compile-time template parameter, not a runtime branch — the size-adaptive selector is at the compilation level, not a dynamic `if`.

The ENGINE=1 CuTe path further self-specializes via `if constexpr (N == ...)` at wgmma.cuh:641-686, folding in exactly the right CuTe atom struct for each N — matching the "self-specializes by deployment config" design claim.
