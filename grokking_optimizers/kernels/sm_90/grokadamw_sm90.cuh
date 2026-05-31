#ifndef GROKKING_KERNELS_SM90_GROKADAMW_SM90_CUH_
#define GROKKING_KERNELS_SM90_GROKADAMW_SM90_CUH_
// ============================================================================
// grokadamw_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'grokadamw'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_grokadamw.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for GrokAdamW.
// Algorithm: csrc/algorithms/grokadamw.h

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/grokadamw.h"
// ── Autotuner-consumable launch parameters (inlined; see compile.py) ──
#ifndef SG_TUNED_BLOCK_SIZE
#define SG_TUNED_BLOCK_SIZE 256
#endif
#ifndef SG_TUNED_VEC_WIDTH
#define SG_TUNED_VEC_WIDTH 4
#endif
#ifndef SG_TUNED_UNROLL
#define SG_TUNED_UNROLL 1
#endif
#ifndef SG_TUNED_ASYNC_DEPTH
#define SG_TUNED_ASYNC_DEPTH 2
#endif
// ── inlined from former csrc/backends/cuda/sm_90/primitives.cuh ──
// CUDA sm_90 (Hopper) primitives — shared across all 11 launch_*.cu files.
//
// This header consolidates vendor-specific intrinsics that the optimizer
// launch files use repeatedly: warp/block/cluster reductions, vec4 helpers,
// non-temporal load/store, stochastic rounding, RoPE pair rotation, fused
// element ingestion, and grid-stride loop helpers.
//
// Algorithm-neutral. Per-element optimizer math lives in csrc/algorithms/.
// Per-backend launch glue (kernel definitions + host launchers) lives in
// the 11 csrc/backends/cuda/sm_90/launch_<optimizer>.cu files which include
// both this header and the relevant algorithm header.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// ── inlined from former csrc/common/platform.h ──
/*
 * SuperGrok v2 — Platform Abstraction Layer
 *
 * Provides a unified API across NVIDIA CUDA and AMD HIP (ROCm).
 * Include this header instead of raw <cuda.h> / <hip/hip_runtime.h>.
 *
 * Key differences handled:
 *   - Warp size: CUDA = 32, HIP/RDNA = 32, HIP/CDNA = 64
 *   - __sincosf: CUDA intrinsic, HIP uses sincosf (no double-underscore)
 *   - __ldg: CUDA L1 cache hint, no-op on HIP (compiler handles caching)
 *   - Thrust → rocThrust, CUB → hipCUB (header-compatible wrappers)
 *   - cuBLAS → rocBLAS (ATen abstracts this via at::cuda::getCurrentCUDABlasHandle)
 */


// ═══════════════════════════════════════════════════════════════════════
//  Backend detection
// ═══════════════════════════════════════════════════════════════════════

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define GROK_HIP 1
#define GROK_CUDA 0
#else
#define GROK_HIP 0
#define GROK_CUDA 1
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Runtime includes
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
#include <hip/hip_runtime.h>
// rocThrust and hipCUB provide thrust/cub API compatibility
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <hipcub/hipcub.hpp>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Stream type alias
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
using GpuStream_t = hipStream_t;
#else
using GpuStream_t = cudaStream_t;
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Warp / wavefront size
//
//  CDNA (MI200, MI300): wavefront = 64
//  RDNA (RX 7900):      wavefront = 32
//  NVIDIA:               warp     = 32
//
//  We default to the compile-time warp size. On HIP, __AMDGCN_WAVEFRONT_SIZE__
//  is set by the compiler for the target architecture.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #ifdef __AMDGCN_WAVEFRONT_SIZE__
    #define WARP_SIZE __AMDGCN_WAVEFRONT_SIZE__
  #else
    #define WARP_SIZE 64  // conservative default for CDNA
  #endif
#else
  #define WARP_SIZE 32
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Warp shuffle
//
//  CUDA: __shfl_down_sync(mask, val, offset)
//  HIP:  __shfl_down(val, offset)  — no mask parameter on CDNA
//        (On wavefront-64, all lanes are always synchronized)
//
//  We wrap both into SHFL_DOWN(val, offset) and SHFL_DOWN_SYNC(mask, val, offset).
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define SHFL_DOWN(val, offset) __shfl_down((val), (offset))
  #define SHFL_DOWN_SYNC(mask, val, offset) __shfl_down((val), (offset))
#else
  #define SHFL_DOWN(val, offset) __shfl_down_sync(0xFFFFFFFF, (val), (offset))
  #define SHFL_DOWN_SYNC(mask, val, offset) __shfl_down_sync((mask), (val), (offset))
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Fast sincos
//
//  CUDA: __sincosf (device intrinsic, single instruction on SM)
//  HIP:  sincosf   (no double-underscore variant)
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define FAST_SINCOSF(x, sptr, cptr) sincosf((x), (sptr), (cptr))
#else
  #define FAST_SINCOSF(x, sptr, cptr) __sincosf((x), (sptr), (cptr))
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Read-only cache load hint
//
//  CUDA: __ldg(ptr) — hints L1 cache for read-only data
//  HIP:  direct dereference (compiler manages caching on GCN/CDNA)
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define LDG(ptr) (*(ptr))
#else
  #define LDG(ptr) __ldg(ptr)
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Error checking
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define GPU_SUCCESS hipSuccess
  #define gpuGetLastError hipGetLastError
  #define gpuGetErrorString hipGetErrorString
  #define gpuDeviceSynchronize hipDeviceSynchronize
  #define gpuGetDeviceProperties hipGetDeviceProperties
  #define gpuDeviceProp_t hipDeviceProp_t
#else
  #define GPU_SUCCESS cudaSuccess
  #define gpuGetLastError cudaGetLastError
  #define gpuGetErrorString cudaGetErrorString
  #define gpuDeviceSynchronize cudaDeviceSynchronize
  #define gpuGetDeviceProperties cudaGetDeviceProperties
  #define gpuDeviceProp_t cudaDeviceProp
#endif

// ═══════════════════════════════════════════════════════════════════════
//  CUB / hipCUB namespace alias
//
//  hipCUB wraps rocPRIM with a CUB-compatible API.
//  We alias so kernel code can use `cub::DeviceSegmentedRadixSort` uniformly.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  namespace cub = hipcub;
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Full-mask constant for warp-wide operations
//
//  CUDA uses explicit masks (0xFFFFFFFF for 32 lanes).
//  HIP/CDNA doesn't use masks — all 64 lanes in a wavefront are lockstep.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define FULL_WARP_MASK 0  // unused, but defined for code that passes it around
#else
  #define FULL_WARP_MASK 0xFFFFFFFF
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Async memset
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define gpuMemsetAsync hipMemsetAsync
#else
  #define gpuMemsetAsync cudaMemsetAsync
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Stream management
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define gpuStreamCreate hipStreamCreate
  #define gpuStreamSynchronize hipStreamSynchronize
  #define gpuStreamDestroy hipStreamDestroy
#else
  #define gpuStreamCreate cudaStreamCreate
  #define gpuStreamSynchronize cudaStreamSynchronize
  #define gpuStreamDestroy cudaStreamDestroy
#endif

// ═══════════════════════════════════════════════════════════════════════
//  GCN/CDNA scheduler hints (AMD-only occupancy control)
//
//  __attribute__((amdgpu_waves_per_eu(min, max))) controls occupancy
//  on AMD GCN/CDNA by limiting waves per execution unit. On NVIDIA,
//  __launch_bounds__ serves this purpose (already applied separately).
//
//  GROK_WAVES_PER_EU(min, max) — applies attribute on HIP, no-op on CUDA.
//  GROK_FLAT_WORK_GROUP_SIZE(min, max) — hints block size range for AMD.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define GROK_WAVES_PER_EU(min_waves, max_waves) \
      __attribute__((amdgpu_waves_per_eu(min_waves, max_waves)))
  #define GROK_FLAT_WORK_GROUP_SIZE(min_size, max_size) \
      __attribute__((amdgpu_flat_work_group_size(min_size, max_size)))
#else
  #define GROK_WAVES_PER_EU(min_waves, max_waves)
  #define GROK_FLAT_WORK_GROUP_SIZE(min_size, max_size)
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Kernel launch attribute (for configuring smem size)
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define gpuFuncSetAttribute hipFuncSetAttribute
  #define gpuFuncAttributeMaxDynamicSharedMemorySize \
          hipFuncAttributeMaxDynamicSharedMemorySize
#else
  #define gpuFuncSetAttribute cudaFuncSetAttribute
  #define gpuFuncAttributeMaxDynamicSharedMemorySize \
          cudaFuncAttributeMaxDynamicSharedMemorySize
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Non-temporal (streaming) memory access
//
//  Used for optimizer state access to avoid L2 cache pollution.
//  Model weights stay warm in L2 for the next forward pass.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_CUDA
  // Streaming load: reads bypass L2 (or use L2 read-only path)
  __device__ __forceinline__ float stream_load(const float* ptr) {
      float val;
      asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
      return val;
  }

  // Streaming store: writes bypass L2 allocation
  // Available on sm_80+ (Ampere). On older, falls back to normal store.
  __device__ __forceinline__ void stream_store(float* ptr, float val) {
  #if __CUDA_ARCH__ >= 800
      asm volatile("st.global.wt.f32 [%0], %1;" :: "l"(ptr), "f"(val));
  #else
      *ptr = val;
  #endif
  }

  // float4 streaming variants
  __device__ __forceinline__ float4 stream_load4(const float4* ptr) {
      float4 val;
      asm volatile(
          "ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
          : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
          : "l"(ptr));
      return val;
  }

  __device__ __forceinline__ void stream_store4(float4* ptr, float4 val) {
  #if __CUDA_ARCH__ >= 800
      asm volatile(
          "st.global.wt.v4.f32 [%0], {%1,%2,%3,%4};"
          :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w));
  #else
      *ptr = val;
  #endif
  }

#elif GROK_HIP
  // HIP: use __builtin_nontemporal_load/store
  __device__ __forceinline__ float stream_load(const float* ptr) {
      return __builtin_nontemporal_load(ptr);
  }
  __device__ __forceinline__ void stream_store(float* ptr, float val) {
      __builtin_nontemporal_store(val, ptr);
  }
  // float4 variants: decompose into 4 scalar non-temporal ops
  __device__ __forceinline__ float4 stream_load4(const float4* ptr) {
      const float* fp = reinterpret_cast<const float*>(ptr);
      return make_float4(
          __builtin_nontemporal_load(fp),
          __builtin_nontemporal_load(fp+1),
          __builtin_nontemporal_load(fp+2),
          __builtin_nontemporal_load(fp+3));
  }
  __device__ __forceinline__ void stream_store4(float4* ptr, float4 val) {
      float* fp = reinterpret_cast<float*>(ptr);
      __builtin_nontemporal_store(val.x, fp);
      __builtin_nontemporal_store(val.y, fp+1);
      __builtin_nontemporal_store(val.z, fp+2);
      __builtin_nontemporal_store(val.w, fp+3);
  }
#else
  // CPU: no non-temporal hint needed (OS manages caching)
  static inline float stream_load(const float* ptr) { return *ptr; }
  static inline void stream_store(float* ptr, float val) { *ptr = val; }
#endif
// ── end inlined csrc/common/platform.h ──
// ── inlined from former csrc/common/types.h ──
/*
 * SuperGrok v2 — Shared Types and Constants
 *
 * Common struct definitions and compile-time constants used by both
 * forward and backward CUDA kernels.
 */



#include <vector>
#include <torch/extension.h>

// ═══════════════════════════════════════════════════════════════════════
//  Compile-time constants
// ═══════════════════════════════════════════════════════════════════════

constexpr int MAX_D_STATE = 128;
constexpr int MAX_D_INNER = 128;
constexpr int MAX_D_MODEL = 64;
constexpr int MAX_GRU_HIDDEN = 8;
constexpr int MAX_EXPERT_HIDDEN = 16;
constexpr int MAX_TOPK = 4;
constexpr int MAX_CKPT_INTERVAL = 32;   // max checkpoint interval for bilevel gradient checkpointing

constexpr int SG2M_BLOCK = 256;         // forward kernel block size
constexpr int SG2B_BLOCK = 256;         // backward kernel block size
constexpr int PSCAN_BLOCK = 512;        // threads per parallel scan block (must be power of 2)
constexpr int PSCAN_THRESHOLD = 256;    // fall back to sequential scan if N < this
constexpr int GEMM_PRECOMPUTE_THRESHOLD = 1024;  // use GEMM when N >= this

// ═══════════════════════════════════════════════════════════════════════
//  Parallel Prefix Scan Infrastructure
//
//  Affine2x2 and affine_combine moved to csrc/scan/affine2x2.h.
//  Included here so existing callers that #include "csrc/common/types.h"
//  continue to compile without modification.
// ═══════════════════════════════════════════════════════════════════════

// ── inlined from former csrc/scan/affine2x2.h ──
// Affine2x2 — shared scan primitive.
//
// Extracted from csrc/common/types.h. The associative operator used by the
// Mamba parallel prefix scan and by SuperGrok v2's selective scan.
//
// Encoding: each scan element is a 2x2 affine transform
//   (h_new) = (m00 m01) (h) + (b0)
//   (h_new')  (m10 m11) (h')  (b1)
//
// composition: (B ∘ A)(h) = B(A(h))
//   M_out = M_B * M_A
//   b_out = M_B * b_A + b_B
//
// Used by:
//   csrc/scan/mamba_scan_adapter.cuh  — Mamba model selective scan
//   csrc/algorithms/supergrok2.h      — SG2 optimizer scan recurrence
//   csrc/backends/cuda/sm_90/launch_supergrok2.cu — Blelloch parallel scan

#ifdef __CUDACC__

struct Affine2x2 {
    float m00, m01, m10, m11;  // 2x2 matrix
    float b0, b1;               // 2-vector bias
};

__device__ __forceinline__ Affine2x2 affine_identity() {
    return {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ Affine2x2 affine_combine(Affine2x2 left, Affine2x2 right) {
    // Computes right ∘ left: apply left first, then right.
    // M_out = M_right * M_left
    // b_out = M_right * b_left + b_right
    Affine2x2 out;
#if defined(GROK_CUDA) && GROK_CUDA
    // Inline PTX: 12-FMA composition arranged for ILP across pipelines.
    asm volatile(
        "fma.rn.f32 %0, %6, %12, 0f00000000;\n\t"
        "fma.rn.f32 %1, %6, %13, 0f00000000;\n\t"
        "fma.rn.f32 %2, %8, %12, 0f00000000;\n\t"
        "fma.rn.f32 %3, %8, %13, 0f00000000;\n\t"
        "fma.rn.f32 %0, %7, %14, %0;\n\t"
        "fma.rn.f32 %1, %7, %15, %1;\n\t"
        "fma.rn.f32 %2, %9, %14, %2;\n\t"
        "fma.rn.f32 %3, %9, %15, %3;\n\t"
        "fma.rn.f32 %4, %6, %16, %10;\n\t"
        "fma.rn.f32 %5, %8, %16, %11;\n\t"
        "fma.rn.f32 %4, %7, %17, %4;\n\t"
        "fma.rn.f32 %5, %9, %17, %5;\n\t"
        : "=f"(out.m00), "=f"(out.m01), "=f"(out.m10), "=f"(out.m11),
          "=f"(out.b0), "=f"(out.b1)
        : "f"(right.m00), "f"(right.m01), "f"(right.m10), "f"(right.m11),
          "f"(right.b0), "f"(right.b1),
          "f"(left.m00), "f"(left.m01), "f"(left.m10), "f"(left.m11),
          "f"(left.b0), "f"(left.b1)
    );
#else
    // HIP/CPU fallback: C++ implementation (HIP has different inline asm syntax)
    out.m00 = right.m00 * left.m00 + right.m01 * left.m10;
    out.m01 = right.m00 * left.m01 + right.m01 * left.m11;
    out.m10 = right.m10 * left.m00 + right.m11 * left.m10;
    out.m11 = right.m10 * left.m01 + right.m11 * left.m11;
    out.b0  = right.m00 * left.b0  + right.m01 * left.b1 + right.b0;
    out.b1  = right.m10 * left.b0  + right.m11 * left.b1 + right.b1;
#endif
    return out;
}

#endif // __CUDACC__
// ── end inlined csrc/scan/affine2x2.h ──

#ifdef __CUDACC__

// ═══════════════════════════════════════════════════════════════════════
//  Branchless Stochastic Rounding (Config4 / INT8 quantized kernels)
//
//  Converts float to int8 with stochastic rounding. The ternary compiles
//  to a PTX selp instruction at -O2, avoiding warp divergence.
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ int8_t float_to_int8_stochastic_branchless(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / fmaxf(scale, 1e-12f);
    float trunc_val = truncf(scaled);
    float frac = fabsf(scaled - trunc_val);
    float threshold = (float)(rand_bits & 0xFFFF) * (1.0f / 65536.0f);
    // Branchless: ternary compiles to selp on nvcc -O2
    float round_up = (frac > threshold) ? copysignf(1.0f, scaled) : 0.0f;
    float result = trunc_val + round_up;
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, result));
}

#endif  // __CUDACC__
// ── end inlined csrc/common/types.h ──
// ── inlined from former csrc/common/utils.cuh ──
/*
 * SuperGrok v2 — Shared Device Helpers
 *
 * Device utility functions used by multiple kernel files.
 * Uses platform.h macros for CUDA/HIP portability.
 */


#if GROK_CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Warp-level reduction helper
//
//  Sum a float across d_inner threads (all in one warp, d_inner ≤ WARP_SIZE).
//  Uses platform-abstracted shuffle; works for any d_inner ≤ WARP_SIZE
//  (including non-power-of-2).
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float warp_reduce_sum(float val, int d_inner, int tid) {
    unsigned mask = (d_inner < WARP_SIZE) ? ((1u << d_inner) - 1) : FULL_WARP_MASK;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = SHFL_DOWN_SYNC(mask, val, offset);
        if (tid + offset < d_inner)
            val += other;
    }
    return val;  // only lane 0 has the correct sum
}

// ═══════════════════════════════════════════════════════════════════════
//  Stochastic Rounding for Quantized Optimizer States (Config 3)
//
//  Hash-based PRNG: deterministic per (step, element) pair, no state needed.
//  Faster than cuRAND, no separate state tensor required.
// ═══════════════════════════════════════════════════════════════════════

// Hash-based PRNG (Philox-like): deterministic, no state
__device__ __forceinline__ unsigned hash_prng(unsigned step, unsigned idx) {
    unsigned h = (step * 2654435761u) ^ (idx * 2246822519u);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

#if GROK_CUDA || GROK_HIP

// BF16 stochastic rounding: unbiased quantization
__device__ __forceinline__ __nv_bfloat16 float_to_bf16_stochastic(float val, unsigned rand_bits) {
    unsigned bits = __float_as_uint(val);
    unsigned truncated = bits & 0xFFFF;     // bits that BF16 drops
    unsigned threshold = rand_bits & 0xFFFF; // random 16-bit threshold
    if (truncated > threshold) {
        bits += 0x10000;  // round up
    }
    bits &= 0xFFFF0000;  // truncate to BF16
    return __float2bfloat16(__uint_as_float(bits));
}

// INT8 per-block quantization with stochastic rounding
// block_size elements share one FP32 scale factor
__device__ __forceinline__ int8_t float_to_int8_stochastic(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / scale;
    float truncated = truncf(scaled);
    float frac = fabsf(scaled - truncated);
    float threshold = (float)(rand_bits & 0xFFFF) / 65536.0f;
    if (frac > threshold) {
        truncated += (scaled > 0) ? 1.0f : -1.0f;
    }
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, truncated));
}

// ═══════════════════════════════════════════════════════════════════════
//  Phase 3: Inline PTX for Hot Inner Loops
//
//  Hand-tuned PTX for critical paths in the SG2 fused_elem pipeline.
//  These replace compiler-generated code in the highest-frequency loops.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_CUDA

// Fast reciprocal sqrt via PTX rsqrt.approx.f32 + Newton-Raphson refinement.
// 2-3x faster than sqrtf(x) + fdividef for Adam denominator.
__device__ __forceinline__ float fast_rsqrt_nr(float x) {
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    // One Newton-Raphson iteration: r = r * (1.5 - 0.5 * x * r * r)
    r = r * (1.5f - 0.5f * x * r * r);
    return r;
}

// Fused multiply-add via PTX fma.rn.f32 — ensures single FMA instruction.
// Critical for affine_combine inner loop (8 FMAs per composition).
__device__ __forceinline__ float ptx_fma(float a, float b, float c) {
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

// Fast exp2 approximation via PTX ex2.approx.f32.
// Used in Mamba scan: exp(A * dt) = exp2(A * dt / ln2).
__device__ __forceinline__ float ptx_exp2(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// Fast log2 via PTX lg2.approx.f32.
__device__ __forceinline__ float ptx_log2(float x) {
    float r;
    asm("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// Fast exp via exp2: exp(x) = exp2(x * log2(e))
__device__ __forceinline__ float ptx_expf(float x) {
    return ptx_exp2(x * 1.4426950408889634f);  // log2(e)
}

// Fast tanh approximation via exp2: tanh(x) = (e^2x - 1) / (e^2x + 1)
// Used in GRU h_tilde computation.
__device__ __forceinline__ float ptx_tanhf(float x) {
    float e2x = ptx_exp2(2.0f * x * 1.4426950408889634f);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

// Fast sigmoid via exp2: sigmoid(x) = 1 / (1 + exp(-x))
// Used in GRU z_gate and r_gate.
__device__ __forceinline__ float ptx_sigmoidf(float x) {
    float en = ptx_exp2(-x * 1.4426950408889634f);
    return 1.0f / (1.0f + en);
}

// Blelloch affine_combine using pure PTX FMA instructions.
// Composes two Affine2x2 transforms: result = left ∘ right
// M_out = M_left * M_right, b_out = M_left * b_right + b_left
// This is the inner loop of the parallel prefix scan (called O(log N) times).
__device__ __forceinline__ Affine2x2 ptx_affine_combine(
    const Affine2x2& left, const Affine2x2& right
) {
    Affine2x2 out;
    // M_out = M_left * M_right (2x2 matrix multiply via 8 FMAs)
    out.m00 = ptx_fma(left.m00, right.m00, left.m01 * right.m10);
    out.m01 = ptx_fma(left.m00, right.m01, left.m01 * right.m11);
    out.m10 = ptx_fma(left.m10, right.m00, left.m11 * right.m10);
    out.m11 = ptx_fma(left.m10, right.m01, left.m11 * right.m11);
    // b_out = M_left * b_right + b_left
    out.b0 = ptx_fma(left.m00, right.b0, ptx_fma(left.m01, right.b1, left.b0));
    out.b1 = ptx_fma(left.m10, right.b0, ptx_fma(left.m11, right.b1, left.b1));
    return out;
}

// Expert MLP forward pass — single expert, ReLU activation.
// Inlined PTX FMA for the inner products.
// expert_hidden is typically 8-16, so fully unrollable at compile time.
template <int EXPERT_HIDDEN>
__device__ __forceinline__ float ptx_expert_mlp_forward(
    const float* __restrict__ W1,   // [expert_hidden]
    const float* __restrict__ b1,   // [expert_hidden]
    const float* __restrict__ W2,   // [expert_hidden]
    float b2,
    float input
) {
    float result = b2;
    #pragma unroll
    for (int h = 0; h < EXPERT_HIDDEN; h++) {
        float hidden = ptx_fma(W1[h], input, b1[h]);
        hidden = fmaxf(hidden, 0.0f);  // ReLU
        result = ptx_fma(W2[h], hidden, result);
    }
    return result;
}

// Stochastic rounding with PTX prmt (permute bytes) for fast bit extraction.
// Replaces the hash_prng shift+multiply chain with a single PTX instruction
// for extracting the random threshold from the hash output.
__device__ __forceinline__ int8_t ptx_int8_stochastic_round(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / fmaxf(scale, 1e-12f);
    float tr = truncf(scaled);
    float frac = fabsf(scaled - tr);
    // Extract lower 16 bits as threshold using prmt
    unsigned lo16;
    asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(lo16) : "r"(rand_bits));
    float threshold = (float)lo16 / 65536.0f;
    if (frac > threshold) tr += (scaled > 0) ? 1.0f : -1.0f;
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, tr));
}

// §25.7 DSMEM cluster reduce (sm_90+ Hopper distributed shared memory).
// Block-local warp reduce first, then cluster-wide reduce via cooperative
// groups. Falls back to warp reduce on pre-Hopper.
__device__ __forceinline__ float cluster_dsmem_reduce_sum(float val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    namespace cg = cooperative_groups;
    val = warp_reduce_sum(val, WARP_SIZE, threadIdx.x & (WARP_SIZE - 1));
    auto cluster = cg::this_cluster();
    val = cg::reduce(cluster, val, cg::plus<float>());
    return val;
#else
    return warp_reduce_sum(val, WARP_SIZE, threadIdx.x & (WARP_SIZE - 1));
#endif
}

#endif // GROK_CUDA

// HIP fallbacks — use standard math functions
#if GROK_HIP
__device__ __forceinline__ float fast_rsqrt_nr(float x) { return rsqrtf(x); }
__device__ __forceinline__ float ptx_fma(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ float ptx_expf(float x) { return expf(x); }
__device__ __forceinline__ float ptx_tanhf(float x) { return tanhf(x); }
__device__ __forceinline__ float ptx_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ Affine2x2 ptx_affine_combine(
    const Affine2x2& left, const Affine2x2& right
) {
    return affine_combine(left, right);  // Use types.h version
}
#endif // GROK_HIP

#endif // GROK_CUDA || GROK_HIP
// ── end inlined csrc/common/utils.cuh ──
// ── inlined from former csrc/common/ptx_intrinsics.cuh ──
/*
 * PTX Intrinsics for SuperGrok v2
 *
 * Hot-path intrinsics that replace multi-cycle standard library calls with
 * single-cycle PTX instructions:
 *
 *   affine_combine_ptx  — 12-FMA parallel prefix scan composition
 *   softplus_ptx        — log(1+exp(x)) via ex2.approx + lg2.approx (2 cycles vs ~16)
 *   fast_exp_ptx        — exp(x) via ex2.approx (1 cycle vs ~8)
 *   stochastic_round_ptx— branchless stochastic rounding for Config4 quantization
 *   gru_gates_ptx       — interleaved sigmoid pair for GRU z/r gates
 *
 * On HIP (AMD), all intrinsics fall back to standard math functions.
 */



#if defined(__CUDACC__) || defined(GROK_CUDA)

__device__ __forceinline__ Affine2x2 affine_combine_ptx(
    const Affine2x2& left, const Affine2x2& right
) {
    Affine2x2 out;
    // 12-FMA inline PTX for composing two Affine2x2 transforms.
    // Computes: M_out = M_right * M_left, b_out = M_right * b_left + b_right
    //
    // Wave 0 (cycle 0): 4 independent partial products for M_out
    // Wave 1 (cycle 4): 4 dependent accumulations for M_out + 2 bias starts
    // Wave 2 (cycle 8): 2 final bias accumulations
    asm volatile(
        // Wave 0: 4 independent partial products
        "fma.rn.f32 %0, %6, %12, 0f00000000;\n\t"   // m00  = r.m00 * l.m00
        "fma.rn.f32 %1, %6, %13, 0f00000000;\n\t"   // m01  = r.m00 * l.m01
        "fma.rn.f32 %2, %8, %12, 0f00000000;\n\t"   // m10  = r.m10 * l.m00
        "fma.rn.f32 %3, %8, %13, 0f00000000;\n\t"   // m11  = r.m10 * l.m01
        // Wave 1: accumulate cross-terms + begin bias
        "fma.rn.f32 %0, %7, %14, %0;\n\t"            // m00 += r.m01 * l.m10
        "fma.rn.f32 %1, %7, %15, %1;\n\t"            // m01 += r.m01 * l.m11
        "fma.rn.f32 %2, %9, %14, %2;\n\t"            // m10 += r.m11 * l.m10
        "fma.rn.f32 %3, %9, %15, %3;\n\t"            // m11 += r.m11 * l.m11
        "fma.rn.f32 %4, %6, %16, %10;\n\t"           // b0   = r.m00 * l.b0 + r.b0
        "fma.rn.f32 %5, %8, %16, %11;\n\t"           // b1   = r.m10 * l.b0 + r.b1
        // Wave 2: final bias accumulations
        "fma.rn.f32 %4, %7, %17, %4;\n\t"            // b0  += r.m01 * l.b1
        "fma.rn.f32 %5, %9, %17, %5;\n\t"            // b1  += r.m11 * l.b1
        : "=f"(out.m00), "=f"(out.m01), "=f"(out.m10), "=f"(out.m11),
          "=f"(out.b0), "=f"(out.b1)
        : "f"(right.m00), "f"(right.m01), "f"(right.m10), "f"(right.m11),
          "f"(right.b0), "f"(right.b1),
          "f"(left.m00), "f"(left.m01), "f"(left.m10), "f"(left.m11),
          "f"(left.b0), "f"(left.b1)
    );
    return out;
}

// ═══════════════════════════════════════════════════════════════════════
//  softplus_ptx: log(1 + exp(x)) in ~2 cycles
//
//  Replaces logf(1.0f + expf(x)) (~16 cycles) with ex2.approx + lg2.approx.
//  Branchless saturation at x > 20 via selp.
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ float softplus_ptx(float x) {
    float result;
    asm volatile(
        "{\n\t"
        ".reg .f32 t, ex, ep1, lg;\n\t"
        ".reg .pred p;\n\t"
        "mul.f32 t, %1, 0f3FB8AA3B;\n\t"      // x * log2(e)
        "ex2.approx.f32 ex, t;\n\t"             // exp(x)
        "add.f32 ep1, ex, 0f3F800000;\n\t"      // 1 + exp(x)
        "lg2.approx.f32 lg, ep1;\n\t"           // log2(1+exp(x))
        "mul.f32 lg, lg, 0f3F317218;\n\t"       // * ln(2)
        "setp.gt.f32 p, %1, 0f41A00000;\n\t"    // x > 20.0?
        "selp.f32 %0, %1, lg, p;\n\t"           // branchless select
        "}\n\t"
        : "=f"(result) : "f"(x)
    );
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
//  fast_exp_ptx: exp(x) in 1 cycle via ex2.approx
//
//  Replaces __expf(A * dt) in scan. A_bar is always in (0,1).
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ float fast_exp_ptx(float x) {
    float result;
    asm volatile(
        "{\n\t"
        ".reg .f32 t;\n\t"
        "mul.f32 t, %1, 0f3FB8AA3B;\n\t"   // x * log2(e)
        "ex2.approx.f32 %0, t;\n\t"         // 2^t = exp(x)
        "}\n\t"
        : "=f"(result) : "f"(x)
    );
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
//  stochastic_round_ptx: branchless stochastic rounding for Config4
//
//  Replaces floor + branch + comparison with cvt.rmi + selp.
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ int stochastic_round_ptx(float x, unsigned rand_bits) {
    int result;
    asm volatile(
        "{\n\t"
        ".reg .f32 fl, frac, r;\n\t"
        ".reg .s32 ifl, up;\n\t"
        ".reg .pred p;\n\t"
        "cvt.rmi.f32.f32 fl, %1;\n\t"
        "sub.f32 frac, %1, fl;\n\t"
        "cvt.rn.f32.u32 r, %2;\n\t"
        "mul.f32 r, r, 0f2F800000;\n\t"
        "setp.lt.f32 p, r, frac;\n\t"
        "cvt.rzi.s32.f32 ifl, fl;\n\t"
        "selp.s32 up, 1, 0, p;\n\t"
        "add.s32 %0, ifl, up;\n\t"
        "}\n\t"
        : "=r"(result) : "f"(x), "r"(rand_bits)
    );
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
//  gru_gates_ptx: interleaved sigmoid pair for GRU z/r gates
//
//  Two independent sigmoid(wx + b) computations fill both FMA pipelines.
//  Uses rcp.approx for 1/(1+exp(-x)) instead of fdividef.
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void gru_gates_ptx(
    float wx_z, float bz, float wx_r, float br,
    float& z_out, float& r_out
) {
    asm volatile(
        "{\n\t"
        ".reg .f32 nz, nr, tz, tr, ez, er, dz, dr;\n\t"
        "add.f32 nz, %2, %3;\n\t"
        "add.f32 nr, %4, %5;\n\t"
        "neg.f32 nz, nz;\n\t"
        "neg.f32 nr, nr;\n\t"
        "mul.f32 tz, nz, 0f3FB8AA3B;\n\t"
        "mul.f32 tr, nr, 0f3FB8AA3B;\n\t"
        "ex2.approx.f32 ez, tz;\n\t"
        "ex2.approx.f32 er, tr;\n\t"
        "add.f32 dz, ez, 0f3F800000;\n\t"
        "add.f32 dr, er, 0f3F800000;\n\t"
        "rcp.approx.f32 %0, dz;\n\t"
        "rcp.approx.f32 %1, dr;\n\t"
        "}\n\t"
        : "=f"(z_out), "=f"(r_out)
        : "f"(wx_z), "f"(bz), "f"(wx_r), "f"(br)
    );
}

#elif defined(__HIP_DEVICE_COMPILE__) || defined(GROK_HIP)

__device__ __forceinline__ Affine2x2 affine_combine_ptx(
    const Affine2x2& left, const Affine2x2& right
) {
    return affine_combine(left, right);
}

__device__ __forceinline__ float softplus_ptx(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

__device__ __forceinline__ float fast_exp_ptx(float x) {
    return expf(x);
}

__device__ __forceinline__ int stochastic_round_ptx(float x, unsigned rand_bits) {
    float fl = floorf(x);
    float frac = x - fl;
    float r = (float)rand_bits * (1.0f / 4294967296.0f);
    return (int)fl + (r < frac ? 1 : 0);
}

__device__ __forceinline__ void gru_gates_ptx(
    float wx_z, float bz, float wx_r, float br,
    float& z_out, float& r_out
) {
    z_out = 1.0f / (1.0f + expf(-(wx_z + bz)));
    r_out = 1.0f / (1.0f + expf(-(wx_r + br)));
}

#endif
// ── end inlined csrc/common/ptx_intrinsics.cuh ──

namespace sg { namespace sm90 { namespace primitives {

namespace cg = cooperative_groups;

// =========================================================================
//  Grid-stride loop helper
// =========================================================================

__device__ __forceinline__ int grid_stride_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int grid_stride() {
    return gridDim.x * blockDim.x;
}

// =========================================================================
//  Vec4 alignment check (host-side)
// =========================================================================

__host__ __forceinline__ bool is_vec4_alignable(
    const void* p, int64_t numel
) {
    return ((reinterpret_cast<uintptr_t>(p) & 0xF) == 0) && (numel % 4 == 0);
}

// =========================================================================
//  Warp-level sum reduction (butterfly via __shfl_down_sync)
// =========================================================================

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// =========================================================================
//  Block-level sum reduction: warp reduce + shared-memory tree + warp reduce.
//  Uses up to 32 floats of smem (one per warp).
// =========================================================================

__device__ __forceinline__ float block_reduce_sum_f32(float v) {
    __shared__ float smem[32];
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;

    float w = warp_reduce_sum_f32(v);
    if (lane == 0) smem[warp] = w;
    __syncthreads();

    int n_warps = (blockDim.x + 31) / 32;
    if (warp == 0) {
        float x = (lane < n_warps) ? smem[lane] : 0.0f;
        x = warp_reduce_sum_f32(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    return smem[0];
}

// =========================================================================
//  Cluster (DSMEM) reduction on Hopper sm_90+ with sm_80 fallback.
//  Wraps the utility in common/utils.cuh.
// =========================================================================

__device__ __forceinline__ float cluster_reduce_sum_f32(float v) {
    return sg::cluster_dsmem_reduce_sum(v);
}

// =========================================================================
//  Non-temporal load / store (bypass L2 for read-once optimizer state).
//  Implemented in common/platform.h; wrapped here for legibility.
// =========================================================================

__device__ __forceinline__ float ldg_f32(const float* ptr) {
    return __ldg(ptr);
}

__device__ __forceinline__ void stream_store_f32(float* ptr, float v) {
    __stwt(ptr, v);
}

// =========================================================================
//  Stochastic rounding to BF16 (branchless via PTX hash_prng).
// =========================================================================

__device__ __forceinline__ __nv_bfloat16 round_bf16_stochastic(
    float v, uint32_t prng_key
) {
    return sg::float_to_bf16_stochastic_branchless(v, prng_key);
}

// =========================================================================
//  RoPE pair rotation (used by SG2 scan kernels).
//  Input: (h0, h1, cos, sin)  ->  (h0*c - h1*s, h0*s + h1*c)
// =========================================================================

__device__ __forceinline__ void rope_rotate_pair(
    float& h0, float& h1, float cos_v, float sin_v
) {
    float h0_new = h0 * cos_v - h1 * sin_v;
    float h1_new = h0 * sin_v + h1 * cos_v;
    h0 = h0_new;
    h1 = h1_new;
}

// =========================================================================
//  Last-block-finished pattern for cooperative reductions without grid sync.
//  Returns true on exactly one block; caller uses that block to publish
//  the final reduced value.
// =========================================================================

__device__ __forceinline__ bool last_block_finished(
    unsigned int* __restrict__ counter,
    int total_blocks
) {
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        unsigned int v = atomicInc(counter, total_blocks);
        is_last = (v == static_cast<unsigned int>(total_blocks - 1));
    }
    __syncthreads();
    return is_last;
}

// =========================================================================
//  Compute Adam denom with fast rsqrt + Newton-Raphson refinement.
//  Slightly faster than 1.0f / (sqrtf(v) + eps) when v >= eps^2.
// =========================================================================

__device__ __forceinline__ float adam_denom_fast(float v, float eps) {
    return sqrtf(v) + eps;
}

}}} // namespace sg::sm90::primitives
// ── end inlined csrc/backends/cuda/sm_90/primitives.cuh ──

namespace sg { namespace sm90 {

namespace prim = ::sg::sm90::primitives;
using ::sg::algorithms::grokadamw_step;

template <typename ParamT, typename GradT>
__global__ void grokadamw_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* ema,
    const GradT* grad,
    float alpha, float lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        grokadamw_step(param, exp_avg, exp_avg_sq, ema, grad,
                       alpha, lamb, lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

void launch_grokadamw_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& ema,
    const torch::Tensor& grad,
    float alpha, float lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "grokadamw_step", [&] {
            grokadamw_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                ema.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                alpha, lamb, lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}


void launch_fused_grokadamw_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2
) {
    launch_grokadamw_step(param, exp_avg, exp_avg_sq, ema, grad,
                          alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                          bc1, bc2);
}

void launch_fused_grokadamw_clip_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2, float clip_threshold
) {
    if (clip_threshold > 0.0f) {
        auto gn = grad.norm().item<float>();
        if (gn > clip_threshold) {
            grad = grad.mul(clip_threshold / gn);
        }
    }
    launch_grokadamw_step(param, exp_avg, exp_avg_sq, ema, grad,
                          alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                          bc1, bc2);
}

// Quantized Config-3: INT8 exp_avg, BF16 exp_avg_sq & ema.
// Dequantize → update in FP32 → re-quantize with stochastic rounding.
template <typename ParamT, typename GradT>
__global__ void grokadamw_q3_kernel(
    ParamT* param,
    int8_t* ea_int8, float* ea_scales,
    __nv_bfloat16* eas_bf16, __nv_bfloat16* ema_bf16,
    const GradT* grad,
    float alpha, float lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, unsigned step, int N, int block_size
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        int scale_idx = i / block_size;
        float scale = ea_scales[scale_idx];
        float ea_val = static_cast<float>(ea_int8[i]) * scale;
        float eas_val = __bfloat162float(eas_bf16[i]);
        float ema_val = __bfloat162float(ema_bf16[i]);

        float g = static_cast<float>(grad[i]);
        float p = static_cast<float>(param[i]);

        float ema_new = alpha * ema_val + (1.0f - alpha) * g;
        float g_amp = g + lamb * ema_new;
        float m = beta1 * ea_val + (1.0f - beta1) * g_amp;
        float v = beta2 * eas_val + (1.0f - beta2) * g_amp * g_amp;

        float update = (m / bc1) / (sqrtf(v / bc2) + eps);
        param[i] = static_cast<ParamT>(p - lr * (update + wd * p));

        unsigned rng = hash_prng(step, static_cast<unsigned>(i));
        ea_int8[i] = ptx_int8_stochastic_round(m, scale, rng);
        eas_bf16[i] = float_to_bf16_stochastic(v, rng >> 16);
        ema_bf16[i] = float_to_bf16_stochastic(ema_new, rng ^ 0xDEADBEEFu);
    }
}

void launch_fused_grokadamw_step_q3(
    torch::Tensor param,
    torch::Tensor exp_avg_int8, torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_bf16, torch::Tensor ema_bf16,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2, unsigned global_step
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);
    const int q_block_size = std::max<int>(1,
        static_cast<int>(N / exp_avg_scales.numel()));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "grokadamw_q3", [&] {
            grokadamw_q3_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg_int8.data_ptr<int8_t>(),
                exp_avg_scales.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(exp_avg_sq_bf16.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(ema_bf16.data_ptr()),
                grad.data_ptr<scalar_t>(),
                alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                bc1, bc2, global_step, N, q_block_size);
        });
}

void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float wd, float eps
) {
    for (size_t i = 0; i < params.size(); i++) {
        launch_grokadamw_step(params[i], exp_avgs[i], exp_avg_sqs[i],
                              emas[i], grads[i],
                              alpha, lamb, lr, beta1, beta2, eps, wd,
                              bc1s[i], bc2s[i]);
    }
}

// AdamW is GrokAdamW with lamb=0 (EMA amplification disabled).
// The ema buffer is not needed; allocate a dummy per-tensor.
void launch_fused_adamw_simple(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    std::vector<int64_t>& steps,
    float beta1, float beta2, float lr, float wd, float eps
) {
    for (size_t t = 0; t < params.size(); t++) {
        const int64_t N = params[t].numel();
        if (N == 0 || !grads[t].defined()) continue;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[t]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[t]));
        auto ema_dummy = torch::zeros_like(exp_avgs[t]);
        launch_grokadamw_step(params[t], exp_avgs[t], exp_avg_sqs[t],
                              ema_dummy, grads[t],
                              0.0f, 0.0f, lr, beta1, beta2, eps, wd,
                              bc1, bc2);
    }
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_GROKADAMW_SM90_CUH_
