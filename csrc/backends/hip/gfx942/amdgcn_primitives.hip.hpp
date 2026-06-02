#pragma once
// csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp
// ─────────────────────────────────────────────────────────────────────────
// §2 AMD-native gfx942 (CDNA3 / MI300X) device primitives.
//
// Shared building blocks for the hand-written `.hip` kernels (Stage 5):
//   * MFMA matrix-core wrappers  (__builtin_amdgcn_mfma_f32_*bf16_1k)   — §2.4
//   * DPP cross-lane wavefront reductions (__builtin_amdgcn_mov_dpp)    — §2.6
//   * ds_swizzle / ds_bpermute fallbacks for patterns DPP can't express — §2.6
//   * buffer_load → LDS direct-to-scratchpad staging w/ s_waitcnt vmcnt — §2.7
//   * FP8 FNUZ (CDNA3) <-> f32 conversion (distinct from OCP E4M3/E5M2)  — §2.12
//   * sched_group_barrier hot-loop scheduling control                   — §2.11
//   * AGENT-scope global atomics for cross-chip (XCD) reductions         — §2.13
//
// CONTRACT: this header is only ever compiled by **hipcc** (the `.hip` TUs).
// It is NOT included by the host-compiler `.hip.cpp` thin TUs. Every builtin
// here requires the AMDGCN device compiler; there is no host fallback (unlike
// the portable platform.h helpers). Wavefront width on CDNA3 = 64.
//
// HARDWARE-GATED: this environment has no hipcc, so NONE of this is compile-
// verified here — it is structurally + byte reviewed against the proven
// mamba3 `#if 0` MFMA reference and the AMD CDNA3 ISA / ROCm builtin docs.
// Every kernel built on it is 🟡 until it runs on MI300X
// (see HARDWARE_VALIDATION.md Stage 5).
// ─────────────────────────────────────────────────────────────────────────

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
// <cstdint> is unavailable in the free-standing amdgcn device-only compile gate
// (scripts/amdgcn_check.sh); under real hipcc it is present. Only uint32_t is
// needed, so define it directly when the standard header isn't reachable.
#if defined(__has_include)
#  if __has_include(<cstdint>)
#    include <cstdint>
#  else
typedef unsigned int uint32_t;
#  endif
#else
typedef unsigned int uint32_t;
#endif

namespace sg { namespace gfx942 { namespace amdgcn {

// CDNA3 wavefront width.
static constexpr int kWave = 64;

// ─────────────────────────────────────────────────────────────────────────
//  64-bit grid-stride helpers (mirror of the sm_90 primitives.cuh contract).
//
//  The canonical per-element step functions in csrc/algorithms/<opt>.h take
//  `const int64_t idx`; the gfx942 device kernels that adopt the contract drive
//  them with these. blockIdx.x * blockDim.x is formed in 64-bit (cast BEFORE the
//  multiply) so a launch covering >2^31 elements cannot overflow the 32-bit
//  product. Wavefront width on CDNA3 is 64 (kWave), but the flat 1-D global
//  index is identical to the CUDA form.
//
//  Guarded on __HIPCC__: the built-in launch dims (blockIdx/blockDim/gridDim)
//  exist under real hipcc. The free-standing amdgcn compile gate
//  (scripts/amdgcn_check.sh) does NOT model those built-ins, so the helpers are
//  omitted there (it only type-checks the intrinsic wrappers, not launch glue).
// ─────────────────────────────────────────────────────────────────────────
#if defined(__HIPCC__)
__device__ __forceinline__ int64_t grid_stride_index() {
    return static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int64_t grid_stride() {
    return static_cast<int64_t>(gridDim.x) * blockDim.x;
}
#endif

// ─────────────────────────────────────────────────────────────────────────
// Occupancy attributes (WS5 — wire the dead `waves_per_eu` metadata into a
// REAL codegen attribute). `amdgpu_flat_work_group_size(lo,hi)` bounds the
// flat (1-D) block size the compiler must support, and `amdgpu_waves_per_eu(N)`
// asks the register allocator to leave room for N wavefronts per execution
// unit. Bandwidth-bound elementwise applies/reduces want HIGH occupancy to
// hide global-memory latency, so they request a generous waves_per_eu. CDNA3
// has 4 SIMDs × up to 8 (architectural) waves/SIMD per CU; the per-EU figure
// here is the per-SIMD slot count the allocator should target.
//
// These attributes must sit on the __global__ kernel DEFINITION. The bare gate
// (scripts/amdgcn_check.sh) and real hipcc both honor them (clang AMDGPU
// backend). Usage:
//   __global__ SG_LAUNCH_BOUNDS(256, 8) void my_kernel(...) { ... }
// 256 = our standard block (4 wavefronts), 8 waves/EU for bandwidth kernels.
// ─────────────────────────────────────────────────────────────────────────
// SG_KERNEL_BOUNDS(FLAT_WG, WAVES_PER_EU) is the kernel-declaration PREFIX
// (replaces a bare `__global__`): it both marks the function as an AMDGPU
// kernel AND attaches the occupancy bounds. Two lowerings:
//   * real hipcc  → `__global__ __launch_bounds__(FLAT_WG, WAVES_PER_EU)`,
//     the canonical HIP spelling (maxThreadsPerBlock, minWavesPerEU);
//   * bare gate (plain-C++ amdgcn target, scripts/amdgcn_check.sh) → a single
//     C++11 attribute list, because there the GNU `amdgpu_kernel` attribute is
//     not elevated to kernel-function status early enough for the verifier to
//     accept `amdgpu_flat_work_group_size` (clang-verified: the `[[...]]` form
//     with `clang::amdgpu_kernel` IS accepted, the trailing/GNU forms are not).
// Usage:  SG_KERNEL_BOUNDS(256, 8) void my_kernel(...) { ... }
#if defined(__HIPCC__)
#  define SG_KERNEL_BOUNDS(FLAT_WG, WAVES_PER_EU) \
       __global__ __launch_bounds__((FLAT_WG), (WAVES_PER_EU))
#else
#  define SG_KERNEL_BOUNDS(FLAT_WG, WAVES_PER_EU) \
       [[clang::amdgpu_kernel, \
         gnu::amdgpu_flat_work_group_size(1, FLAT_WG), \
         gnu::amdgpu_waves_per_eu(WAVES_PER_EU)]]
#endif

// ─────────────────────────────────────────────────────────────────────────
// §2.4  MFMA matrix-core wrappers (bf16 inputs, f32 accumulate).
//
// The two CDNA3 bf16 matrix shapes used across the models / SG2 GEMMs. Source
// fragments are 4×u32 (= 4 bf16 pairs per lane); accumulators are the native
// register-spread the ISA defines (16 floats for 32×32×8, 4 floats for
// 16×16×16). Promoted verbatim from the verified mamba3 reference.
// ─────────────────────────────────────────────────────────────────────────
// CLANG-VERIFIED operand types (clang --target=amdgcn -mcpu=gfx942): the bf16
// MFMA `_1k` builtins take a vector of 4 bf16 (= short4) per lane, NOT u32x4.
// (The former in-repo mamba3 reference used u32x4, which clang rejects — fixed
// here and verified via scripts/amdgcn_check.sh.) Accumulators are the native
// f32 register spread: 16 floats for 32×32×8, 4 floats for 16×16×16.
using bf16x4 = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using f32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using f32x4  = __attribute__((__vector_size__(4 * sizeof(float)))) float;

// 32×32×8 bf16: acc[16] += A·B. a/b are 4 bf16 each (bit-pattern in short[4]).
__device__ __forceinline__ void mfma_bf16_32x32x8(
    float acc[16], const short a[4], const short b[4]) {
    f32x16* pacc = reinterpret_cast<f32x16*>(acc);
    const bf16x4* pa = reinterpret_cast<const bf16x4*>(a);
    const bf16x4* pb = reinterpret_cast<const bf16x4*>(b);
    *pacc = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(*pa, *pb, *pacc, 0, 0, 0);
}

// 16×16×16 bf16: acc[4] += A·B.
__device__ __forceinline__ void mfma_bf16_16x16x16(
    float acc[4], const short a[4], const short b[4]) {
    f32x4* pacc = reinterpret_cast<f32x4*>(acc);
    const bf16x4* pa = reinterpret_cast<const bf16x4*>(a);
    const bf16x4* pb = reinterpret_cast<const bf16x4*>(b);
    *pacc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(*pa, *pb, *pacc, 0, 0, 0);
}

// ─────────────────────────────────────────────────────────────────────────
// §2.6  DPP cross-lane wavefront reduction.
//
// __builtin_amdgcn_mov_dpp shuffles a value across lanes by a DPP control
// pattern in ONE instruction (no LDS round-trip). The row_shr/row_bcast
// butterfly reduces all 64 lanes of a wavefront. Falls back to ds_swizzle /
// ds_bpermute for patterns DPP's fixed controls can't express (§2.6).
//
// DPP control encodings (CDNA): row_shr:n = 0x110|n ; the wave-64 reduction is
// the canonical AMD pattern (row shifts within each 16-lane row, then two
// row-broadcast steps to cross the four rows). bound_ctrl=1 zero-fills OOB.
// ─────────────────────────────────────────────────────────────────────────
// dpp_ctrl / masks must be compile-time constants (clang-verified), so they are
// template parameters.
template <int DPP_CTRL, int ROW_MASK = 0xf, int BANK_MASK = 0xf>
__device__ __forceinline__ int dpp_mov(int v) {
    return __builtin_amdgcn_mov_dpp(v, DPP_CTRL, ROW_MASK, BANK_MASK,
                                    /*bound_ctrl=*/true);
}

// Full wave-64 sum reduction via the DPP row-shift + row-broadcast butterfly.
// Every lane returns the wavefront total (broadcast by the final read-lane).
// DPP controls must be compile-time constants (clang-verified) — the butterfly
// is unrolled with literal controls via the templated dpp_mov<>.
#define SG_DPP_ADD_F32(f, CTRL) \
    do { int s_ = dpp_mov<CTRL>(__builtin_bit_cast(int, (f))); \
         (f) += __builtin_bit_cast(float, s_); } while (0)

__device__ __forceinline__ float wave_reduce_add_dpp(float val) {
    float f = val;
    SG_DPP_ADD_F32(f, 0x111);  // row_shr:1
    SG_DPP_ADD_F32(f, 0x112);  // row_shr:2
    SG_DPP_ADD_F32(f, 0x114);  // row_shr:4
    SG_DPP_ADD_F32(f, 0x118);  // row_shr:8
    // cross the four 16-lane rows via row broadcast (ds_swizzle-free DPP path).
    SG_DPP_ADD_F32(f, 0x142);  // row_bcast:15
    SG_DPP_ADD_F32(f, 0x143);  // row_bcast:31
    // the reduced total is in the top lane; broadcast it to every lane.
    int last = __builtin_amdgcn_readlane(__builtin_bit_cast(int, f), kWave - 1);
    return __builtin_bit_cast(float, last);
}

// ds_bpermute-based arbitrary cross-lane gather (for reductions/patterns DPP's
// fixed controls can't express, §2.6). `src_lane` is the lane to read from.
__device__ __forceinline__ float bpermute_lane(float val, int src_lane) {
    // ds_bpermute address is byte-addressed (lane << 2).
    int r = __builtin_amdgcn_ds_bpermute(src_lane << 2,
                                         __builtin_bit_cast(int, val));
    return __builtin_bit_cast(float, r);
}

// ds_swizzle within a 32-lane group (quad/row patterns DPP lacks). PATTERN is
// a compile-time constant (clang-verified).
template <int PATTERN>
__device__ __forceinline__ float swizzle_lane(float val) {
    int r = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, val), PATTERN);
    return __builtin_bit_cast(float, r);
}

// u32 wave reduction (PEER routing counts etc.) via DPP — done in the integer
// domain (a u32 sum reinterpreted through f32 would be meaningless).
__device__ __forceinline__ unsigned wave_reduce_add_u32_dpp(unsigned val) {
    int f = (int)val;
    f += dpp_mov<0x111>(f);
    f += dpp_mov<0x112>(f);
    f += dpp_mov<0x114>(f);
    f += dpp_mov<0x118>(f);
    f += dpp_mov<0x142>(f);
    f += dpp_mov<0x143>(f);
    int last = __builtin_amdgcn_readlane(f, kWave - 1);
    return (unsigned)last;
}

// ─────────────────────────────────────────────────────────────────────────
// §2.7  buffer_load → LDS direct-to-scratchpad staging.
//
// The CDNA `buffer_load ... lds` path streams global memory straight into LDS
// (shared) without round-tripping through VGPRs, then s_waitcnt vmcnt(0)
// tracks completion. Modeled with the global_load + explicit nontemporal hint;
// the true `lds` direct path is emitted by the compiler for these access
// patterns under -mllvm flags. Here we expose the streaming load + the wait.
// ─────────────────────────────────────────────────────────────────────────
template <typename T>
__device__ __forceinline__ T streaming_load(const T* __restrict__ p) {
    return __builtin_nontemporal_load(p);
}
template <typename T>
__device__ __forceinline__ void streaming_store(T* __restrict__ p, T v) {
    __builtin_nontemporal_store(v, p);
}

// ── CACHED (temporal) load/store ────────────────────────────────────────────
// NONTEMPORAL POLICY (WS5/§2.7). `streaming_load`/`streaming_store` are
// nontemporal: they bypass / evict L2 (the CDNA `glc/slc` hint), which is
// CORRECT for one-touch data — a grad read once per step, a param written once
// per step. They are WRONG for recurring optimizer STATE (exp_avg, exp_avg_sq,
// ema, s_track, mu, momentum): that data is read AND written every step and
// re-read on the very next step, so it should stay resident in L2. Using the
// nontemporal hint on state forces a cold reload every step (lost L2 locality).
// `cached_load`/`cached_store` are plain temporal accesses — the compiler keeps
// the normal L2 caching behavior. State buffers MUST use these; grad/param
// (one-touch) keep the streaming_* variants. (adamw_gfx942 is the exemplar:
// state via plain `[]` = cached, grad/param via streaming_*.)
template <typename T>
__device__ __forceinline__ T cached_load(const T* __restrict__ p) {
    return *p;
}
template <typename T>
__device__ __forceinline__ void cached_store(T* __restrict__ p, T v) {
    *p = v;
}
// Wait for all outstanding vector-memory loads (vmcnt(0)).
__device__ __forceinline__ void wait_vmcnt0() {
    __builtin_amdgcn_s_waitcnt(/*vmcnt=*/0);
}

// ─────────────────────────────────────────────────────────────────────────
// §2.12  FP8 FNUZ (CDNA3) <-> f32. CDNA3 uses the FNUZ variant (no -0, no inf:
// the all-zero-exp/all-one-mantissa NaN encoding differs from OCP E4M3/E5M2).
// Conversion via the native cvt builtins.
// ─────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ unsigned f32_to_fp8x4_fnuz_e4m3(
    float a, float b, float c, float d) {
    // Pack 4 floats into one u32 of e4m3 FNUZ via the CDNA3 cvt_pk builtin
    // (word_sel false→low 16b, true→high 16b of the dst u32).
    unsigned lo = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false);
    unsigned hi = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, lo, true);
    return hi;
}
// CLANG-VERIFIED: cvt_f32_fp8's byte-select must be a COMPILE-TIME CONSTANT, so
// the unpack index is a template parameter (clang rejects a runtime index).
template <int IDX>
__device__ __forceinline__ float fp8_fnuz_e4m3_to_f32(unsigned packed) {
    static_assert(IDX >= 0 && IDX < 4, "fp8 byte index 0..3");
    return __builtin_amdgcn_cvt_f32_fp8(packed, IDX);
}

// ─────────────────────────────────────────────────────────────────────────
// §2.11  Explicit instruction-scheduling control for matrix-heavy hot loops.
// sched_group_barrier(mask, size, sync_id) pins the compiler's interleave of
// MFMA vs memory vs valu in the steady-state loop (prevents the scheduler from
// clustering all loads then all MFMAs, which stalls the matrix unit).
// ─────────────────────────────────────────────────────────────────────────
template <int MASK, int SIZE, int SYNC_ID = 0>
__device__ __forceinline__ void sched_group_barrier() {
    __builtin_amdgcn_sched_group_barrier(MASK, SIZE, SYNC_ID);
}
// Hint masks (subset): 0x008 = MFMA, 0x020 = DS read, 0x040 = DS write,
// 0x002 = VALU, 0x100 = VMEM read. Callers interleave e.g. (MFMA,1),(VMEM,2).

// ─────────────────────────────────────────────────────────────────────────
// §2.13  Cross-chip (XCD / chiplet) reductions: AMD has no DSMEM analog, so
// cross-workgroup reduction uses global atomics with AGENT (device) memory
// scope — visible across all 8 XCDs of MI300X.
// ─────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void atomic_add_agent_f32(float* addr, float v) {
    __hip_atomic_fetch_add(addr, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void atomic_add_agent_u32(unsigned* addr, unsigned v) {
    __hip_atomic_fetch_add(addr, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// ── WORKGROUP / WAVEFRONT-scope atomics ─────────────────────────────────────
// AGENT scope publishes across all 8 XCDs (device-wide coherence) — necessary
// for cross-workgroup accumulators but the most expensive scope. When the
// accumulation target is an LDS / workgroup-private slot (the standard
// "DPP wave-reduce → ONE atomic per workgroup" pattern), a WORKGROUP-scope
// atomic is sufficient and far cheaper (no cross-XCD probe traffic). A
// WAVEFRONT-scope atomic narrows further to a single wavefront. The reduction
// kernels should DPP-reduce within the wave first, then issue ONE workgroup
// (or agent, when the accumulator is a global buffer shared across workgroups)
// atomic — never one atomic per element/thread. (The MoE compaction filter
// already follows this; matched here.)
__device__ __forceinline__ void atomic_add_workgroup_f32(float* addr, float v) {
    __hip_atomic_fetch_add(addr, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}
__device__ __forceinline__ void atomic_add_workgroup_u32(unsigned* addr, unsigned v) {
    __hip_atomic_fetch_add(addr, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}
#if defined(__HIP_MEMORY_SCOPE_WAVEFRONT)
__device__ __forceinline__ void atomic_add_wavefront_f32(float* addr, float v) {
    __hip_atomic_fetch_add(addr, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
}
#else
// The free-standing gate (scripts/amdgcn_check.sh) only defines the AGENT /
// WORKGROUP scope macros; fall back to workgroup scope there. Under real hipcc
// __HIP_MEMORY_SCOPE_WAVEFRONT is defined and the narrower scope is used.
__device__ __forceinline__ void atomic_add_wavefront_f32(float* addr, float v) {
    __hip_atomic_fetch_add(addr, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}
#endif

// Workgroup barrier + release/acquire fence (the LDS-handoff pattern used by
// the mamba3 reference scan).
__device__ __forceinline__ void workgroup_barrier_release() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
}
__device__ __forceinline__ void workgroup_barrier_acquire() {
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
    __builtin_amdgcn_s_barrier();
}

}}} // namespace sg::gfx942::amdgcn
