// CUDA sm_90 launch glue for SuperGrok v2.
// Algorithm: csrc/algorithms/supergrok2.h
//
// Consolidates Phase 6's three-way SG2 split (fwd + bwd + warp-specialized)
// into one launch file per the prompt's target architecture. The
// warp-specialized path is a runtime branch (activated when uniform d_state
// is detected), not a separate compilation unit.
//
// This launcher orchestrates the full SG2 pipeline:
//   (1) input_proj_sort         — kernel
//   (2) mamba3_scan             — kernel (sequential | parallel | warp-spec)
//   (3) peer_route + gru_step   — kernel
//   (4) apply tail              — kernel
//   (5) bilevel_precompute      — kernel (backward / meta-net training)
//
// The heavy GEMMs (projections, dt_proj with fused softplus) route through
// CUTLASS (csrc/backends/cuda/sm_90/mma.cuh) when -DWITH_CUTLASS is set,
// or cuBLAS via torch::mm otherwise.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/supergrok2.h"
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
//  BatchedScanCtx — shared bookkeeping for the batched mamba peer step
//
//  Recovered verbatim from the deleted csrc/common/ops.h@682eab4^. The
//  setup helper batched_step_setup_and_sort builds one of these per call;
//  the scan + fused-elem helpers consume it. Per-arch variants in
//  csrc/kernels/{cuda,hip}/<arch>/supergrok2_fwd_*.{cu,hip.cpp}
//  expect this exact layout.
//
//  Layout (14 members): 3 int + 2 std::vector<int>
//                     + 8 torch::Tensor + 1 std::vector<torch::Tensor>.
// ═══════════════════════════════════════════════════════════════════════

struct BatchedScanCtx {
    int num_params;
    int total_N;
    int max_N;
    std::vector<int> N_vec;
    std::vector<int> seg_offsets_cpu;
    torch::Tensor x_sorted_packed;
    torch::Tensor offsets_t;
    torch::Tensor initial_fwd;
    torch::Tensor initial_bwd;
    torch::Tensor final_fwd;
    torch::Tensor final_bwd;
    torch::Tensor fwd_scan_packed;
    torch::Tensor bwd_scan_packed;
    std::vector<torch::Tensor> unsort_idx_list;
};

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

namespace sg { namespace cuda_sm90 { namespace primitives {

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

}}} // namespace sg::cuda_sm90::primitives
// ── end inlined csrc/backends/cuda/sm_90/primitives.cuh ──
// ── inlined from former csrc/scan/mamba_scan_adapter.cuh ──
// csrc/scan/mamba_scan_adapter.cuh — CUDA scan adapter.
// Moved here in Phase 4 of the refactor because the Mamba selective scan is
// shared between the Mamba model kernels and the SuperGrok v2 optimizer.
//
// Thin adapter wrapping SG2's existing mamba3_* scan kernels for model-context
// use. No reimplementation of the core scan algorithm — reuses the Affine2x2
// parallel-prefix infrastructure from csrc/scan/affine2x2.h.
//
// The adapter packs model-level (x, dt, A_log, B, C) into Affine2x2 maps:
//   A_bar = exp(dt * A),  B_bar = dt * B
//   Affine2x2: M = diag(A_bar_s0, A_bar_s1),  b = (B_bar_s0*x, B_bar_s1*x)
// then calls the Blelloch parallel-prefix scan for medium/large N, or a
// simple sequential scan for small N.
//
// Decision tree (thresholds from csrc/common/types.h):
//   N < PSCAN_THRESHOLD (256)               -> sequential scan kernel
//   256 <= N < GEMM_PRECOMPUTE_THRESHOLD    -> parallel Blelloch scan
//   N >= GEMM_PRECOMPUTE_THRESHOLD (1024)   -> parallel Blelloch scan (same kernel)


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
//  BatchedScanCtx — shared bookkeeping for the batched mamba peer step
//
//  Recovered verbatim from the deleted csrc/common/ops.h@682eab4^. The
//  setup helper batched_step_setup_and_sort builds one of these per call;
//  the scan + fused-elem helpers consume it. Per-arch variants in
//  csrc/kernels/{cuda,hip}/<arch>/supergrok2_fwd_*.{cu,hip.cpp}
//  expect this exact layout.
//
//  Layout (14 members): 3 int + 2 std::vector<int>
//                     + 8 torch::Tensor + 1 std::vector<torch::Tensor>.
// ═══════════════════════════════════════════════════════════════════════

struct BatchedScanCtx {
    int num_params;
    int total_N;
    int max_N;
    std::vector<int> N_vec;
    std::vector<int> seg_offsets_cpu;
    torch::Tensor x_sorted_packed;
    torch::Tensor offsets_t;
    torch::Tensor initial_fwd;
    torch::Tensor initial_bwd;
    torch::Tensor final_fwd;
    torch::Tensor final_bwd;
    torch::Tensor fwd_scan_packed;
    torch::Tensor bwd_scan_packed;
    std::vector<torch::Tensor> unsort_idx_list;
};

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

// Algorithm spec for SG2 — kept as a documentation anchor for the scan
// recurrence definition. This adapter's scan kernels are self-contained
// and only need MAX_D_STATE / PSCAN_THRESHOLD / ptx_expf from common/.
#include "csrc/algorithms/supergrok2.h"
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

namespace sg { namespace sm90 { namespace models { namespace mamba_adapter {

// ── Sequential scan kernel (N < PSCAN_THRESHOLD) ──────────────────────
// One thread per d_inner dimension, sequential over timesteps.

template <typename ActT>
__global__ void __launch_bounds__(128, 4)
sequential_scan_kernel(
    const ActT* __restrict__ x,       // [B, N, d_inner]
    const ActT* __restrict__ dt,      // [B, N, d_inner]
    const ActT* __restrict__ A_log,   // [d_inner, d_state]
    const ActT* __restrict__ B,       // [B, N, d_state]
    const ActT* __restrict__ C,       // [B, N, d_state]
    ActT* __restrict__ y,             // [B, N, d_inner]
    float* __restrict__ state_save,   // [B, d_inner, d_state] or nullptr
    int seq_len, int d_inner, int d_state
) {
    const int b = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_inner) return;

    const int bN  = b * seq_len;
    const int bDi = b * d_inner;

    float A[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -ptx_expf(static_cast<float>(A_log[j * d_state + s]));

    float h[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) h[s] = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        float x_val  = static_cast<float>(x[(bN + t) * d_inner + j]);
        float dt_val = static_cast<float>(dt[(bN + t) * d_inner + j]);
        float y_acc  = 0.0f;

        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            float A_bar = ptx_expf(A[s] * dt_val);
            float B_bar = dt_val * static_cast<float>(B[(bN + t) * d_state + s]);
            h[s] = A_bar * h[s] + B_bar * x_val;
            y_acc += static_cast<float>(C[(bN + t) * d_state + s]) * h[s];
        }
        y[(bN + t) * d_inner + j] = static_cast<ActT>(y_acc);
    }

    if (state_save != nullptr) {
        #pragma unroll 4
        for (int s = 0; s < d_state; s++)
            state_save[(bDi + j) * d_state + s] = h[s];
    }
}

// ── Parallel Blelloch scan kernel (N >= PSCAN_THRESHOLD) ──────────────
// One block per (batch, d_inner). Affine2x2 prefix scan across timesteps,
// processing d_state pairs two at a time through the 2x2 matrix machinery.

template <typename ActT>
__global__ void __launch_bounds__(256, 2)
parallel_scan_kernel(
    const ActT* __restrict__ x,
    const ActT* __restrict__ dt,
    const ActT* __restrict__ A_log,
    const ActT* __restrict__ B,
    const ActT* __restrict__ C,
    ActT* __restrict__ y,
    float* __restrict__ state_save,
    int seq_len, int d_inner, int d_state
) {
    const int b = blockIdx.y;
    const int j = blockIdx.x;
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int N = seq_len;
    const int bN = b * N;
    const int bDi = b * d_inner;

    extern __shared__ float smem[];  // 6 * nthreads

    const int chunk = (N + nthreads - 1) / nthreads;
    const int t0 = ltid * chunk;
    const int t1 = min(t0 + chunk, N);
    const int cnt = max(t1 - t0, 0);

    float A_coeff[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A_coeff[s] = -ptx_expf(static_cast<float>(A_log[j * d_state + s]));

    // Zero output for accumulation across d_state pairs
    for (int step = 0; step < cnt; step++) {
        y[(bN + t0 + step) * d_inner + j] = static_cast<ActT>(0.0f);
    }
    __syncthreads();

    const int half_ds = d_state / 2;

    for (int p = 0; p < half_ds; p++) {
        const int s0 = 2 * p, s1 = 2 * p + 1;

        // Phase 1: sequential scan within chunk -> summary Affine2x2
        Affine2x2 summary = affine_identity();
        #pragma unroll 4
        for (int step = 0; step < cnt; step++) {
            int t = t0 + step;
            float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
            float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
            Affine2x2 elem;
            elem.m00 = ptx_expf(A_coeff[s0] * dtv);  elem.m01 = 0.0f;
            elem.m10 = 0.0f;                          elem.m11 = ptx_expf(A_coeff[s1] * dtv);
            elem.b0  = dtv * static_cast<float>(B[(bN + t) * d_state + s0]) * xv;
            elem.b1  = dtv * static_cast<float>(B[(bN + t) * d_state + s1]) * xv;
            summary = affine_combine(summary, elem);
        }

        int base = ltid * 6;
        smem[base]   = summary.m00; smem[base+1] = summary.m01;
        smem[base+2] = summary.m10; smem[base+3] = summary.m11;
        smem[base+4] = summary.b0;  smem[base+5] = summary.b1;
        __syncthreads();

        // Phase 2: Blelloch up-sweep
        for (int stride = 1; stride < nthreads; stride *= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < nthreads) {
                Affine2x2 L = {smem[(idx-stride)*6],   smem[(idx-stride)*6+1],
                               smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                               smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 R = {smem[idx*6],   smem[idx*6+1],
                               smem[idx*6+2], smem[idx*6+3],
                               smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 c = affine_combine(L, R);
                smem[idx*6]=c.m00; smem[idx*6+1]=c.m01; smem[idx*6+2]=c.m10;
                smem[idx*6+3]=c.m11; smem[idx*6+4]=c.b0; smem[idx*6+5]=c.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        // Set last to identity (exclusive scan)
        if (ltid == 0) {
            int last = (nthreads - 1) * 6;
            smem[last]=1; smem[last+1]=0; smem[last+2]=0;
            smem[last+3]=1; smem[last+4]=0; smem[last+5]=0;
        }
        __syncthreads();

        // Down-sweep
        for (int stride = nthreads / 2; stride >= 1; stride /= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < nthreads) {
                Affine2x2 L = {smem[(idx-stride)*6],   smem[(idx-stride)*6+1],
                               smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                               smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 R = {smem[idx*6],   smem[idx*6+1],
                               smem[idx*6+2], smem[idx*6+3],
                               smem[idx*6+4], smem[idx*6+5]};
                smem[(idx-stride)*6]=R.m00; smem[(idx-stride)*6+1]=R.m01;
                smem[(idx-stride)*6+2]=R.m10; smem[(idx-stride)*6+3]=R.m11;
                smem[(idx-stride)*6+4]=R.b0; smem[(idx-stride)*6+5]=R.b1;
                Affine2x2 c = affine_combine(R, L);
                smem[idx*6]=c.m00; smem[idx*6+1]=c.m01; smem[idx*6+2]=c.m10;
                smem[idx*6+3]=c.m11; smem[idx*6+4]=c.b0; smem[idx*6+5]=c.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        // Phase 3: re-scan with prefix, accumulate output
        Affine2x2 run = {smem[ltid*6], smem[ltid*6+1], smem[ltid*6+2],
                         smem[ltid*6+3], smem[ltid*6+4], smem[ltid*6+5]};
        #pragma unroll 4
        for (int step = 0; step < cnt; step++) {
            int t = t0 + step;
            float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
            float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
            Affine2x2 elem;
            elem.m00 = ptx_expf(A_coeff[s0] * dtv);  elem.m01 = 0.0f;
            elem.m10 = 0.0f;                          elem.m11 = ptx_expf(A_coeff[s1] * dtv);
            elem.b0  = dtv * static_cast<float>(B[(bN + t) * d_state + s0]) * xv;
            elem.b1  = dtv * static_cast<float>(B[(bN + t) * d_state + s1]) * xv;
            run = affine_combine(run, elem);

            // h = run applied to zero initial state -> h = run.b
            float c0 = static_cast<float>(C[(bN + t) * d_state + s0]);
            float c1 = static_cast<float>(C[(bN + t) * d_state + s1]);
            float prev = static_cast<float>(y[(bN + t) * d_inner + j]);
            y[(bN + t) * d_inner + j] = static_cast<ActT>(prev + run.b0*c0 + run.b1*c1);
        }

        if (state_save != nullptr && t1 == N && cnt > 0) {
            state_save[(bDi + j) * d_state + s0] = run.b0;
            state_save[(bDi + j) * d_state + s1] = run.b1;
        }
        __syncthreads();
    }

    // Handle odd d_state
    if (d_state % 2 != 0) {
        const int s = d_state - 1;
        float hv = 0.0f;
        for (int step = 0; step < cnt; step++) {
            int t = t0 + step;
            float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
            float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
            hv = ptx_expf(A_coeff[s] * dtv) * hv
               + dtv * static_cast<float>(B[(bN + t) * d_state + s]) * xv;
            float prev = static_cast<float>(y[(bN + t) * d_inner + j]);
            float cv   = static_cast<float>(C[(bN + t) * d_state + s]);
            y[(bN + t) * d_inner + j] = static_cast<ActT>(prev + hv * cv);
        }
        if (state_save != nullptr && t1 == N && cnt > 0)
            state_save[(bDi + j) * d_state + s] = hv;
    }
}

// ── Forward dispatch ──────────────────────────────────────────────────

template <typename ActT>
cudaError_t selective_scan_forward(
    const ActT* x, const ActT* dt, const ActT* A_log,
    const ActT* B, const ActT* C,
    ActT* y, float* state_save,
    int batch, int seq_len, int d_inner, int d_state,
    cudaStream_t stream
) {
    if (seq_len < PSCAN_THRESHOLD) {
        int block = min(d_inner, 128);
        dim3 grid((d_inner + block - 1) / block, batch);
        sequential_scan_kernel<ActT><<<grid, block, 0, stream>>>(
            x, dt, A_log, B, C, y, state_save, seq_len, d_inner, d_state);
    } else {
        int block = min(PSCAN_BLOCK, 256);
        dim3 grid(d_inner, batch);
        int smem_bytes = 6 * block * sizeof(float);
        parallel_scan_kernel<ActT><<<grid, block, smem_bytes, stream>>>(
            x, dt, A_log, B, C, y, state_save, seq_len, d_inner, d_state);
    }
    return cudaGetLastError();
}

// ── Backward: adjoint scan ────────────────────────────────────────────
// Reverse-time sequential scan computing gradients through the recurrence.
// For each timestep t (in reverse):
//   grad_h += C[t] * grad_y[t]
//   grad_B[t] = dt[t] * x[t] * grad_h
//   grad_C[t] = h[t] * grad_y[t]   (h[t] recomputed via forward pass)
//   grad_x[t] = sum_s(B[t,s] * dt[t] * grad_h[s])
//   grad_dt[t] = sum_s(A[s]*A_bar*h[t-1,s] + B[t,s]*x[t]) * grad_h[s]
//   grad_A_log[j,s] += dt[t]*A[s]*A_bar * h[t-1,s] * grad_h[s]
//   grad_h = A_bar * grad_h   (backprop through recurrence)

template <typename ActT>
__global__ void __launch_bounds__(128, 4)
scan_backward_kernel(
    const ActT* __restrict__ grad_y,
    const ActT* __restrict__ x,
    const ActT* __restrict__ dt,
    const ActT* __restrict__ A_log,
    const ActT* __restrict__ B,
    const ActT* __restrict__ C,
    const float* __restrict__ state_save,
    ActT* __restrict__ grad_x,
    ActT* __restrict__ grad_dt,
    float* __restrict__ grad_A_log,  // [d_inner, d_state], atomicAdd
    ActT* __restrict__ grad_B,
    ActT* __restrict__ grad_C,
    int seq_len, int d_inner, int d_state
) {
    const int b = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_inner) return;

    const int bN  = b * seq_len;
    const int bDi = b * d_inner;

    float A[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -ptx_expf(static_cast<float>(A_log[j * d_state + s]));

    // Forward pass to cache h[t] for all t (needed for grad_C and grad_dt)
    float h_cache[MAX_D_STATE];
    float h_prev[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) h_cache[s] = 0.0f;

    // Allocate per-timestep h cache in local memory (seq_len is small)
    float h_all[256 * MAX_D_STATE];  // PSCAN_THRESHOLD * MAX_D_STATE

    // Forward recompute
    for (int t = 0; t < seq_len; t++) {
        float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
        float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            float A_bar = ptx_expf(A[s] * dtv);
            float B_bar = dtv * static_cast<float>(B[(bN + t) * d_state + s]);
            h_cache[s] = A_bar * h_cache[s] + B_bar * xv;
            h_all[t * d_state + s] = h_cache[s];
        }
    }

    // Reverse pass for gradients
    float grad_h[MAX_D_STATE];
    float grad_A_acc[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) { grad_h[s] = 0.0f; grad_A_acc[s] = 0.0f; }

    for (int t = seq_len - 1; t >= 0; t--) {
        float xv   = static_cast<float>(x[(bN + t) * d_inner + j]);
        float dtv  = static_cast<float>(dt[(bN + t) * d_inner + j]);
        float gy   = static_cast<float>(grad_y[(bN + t) * d_inner + j]);

        float grad_x_acc  = 0.0f;
        float grad_dt_acc = 0.0f;

        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            float A_bar = ptx_expf(A[s] * dtv);
            float bv    = static_cast<float>(B[(bN + t) * d_state + s]);
            float cv    = static_cast<float>(C[(bN + t) * d_state + s]);
            float h_t   = h_all[t * d_state + s];
            float h_tm1 = (t > 0) ? h_all[(t-1) * d_state + s] : 0.0f;

            // grad_C[t,s] = h[t,s] * grad_y[t]
            grad_C[(bN + t) * d_state + s] = static_cast<ActT>(h_t * gy);

            // Accumulate into grad_h
            grad_h[s] += cv * gy;

            // grad_B[t,s] = dt * x * grad_h[s]
            grad_B[(bN + t) * d_state + s] = static_cast<ActT>(dtv * xv * grad_h[s]);

            // grad_x accumulation
            grad_x_acc += bv * dtv * grad_h[s];

            // grad_dt accumulation
            grad_dt_acc += (A[s] * A_bar * h_tm1 + bv * xv) * grad_h[s];

            // grad_A_log accumulation
            grad_A_acc[s] += dtv * A[s] * A_bar * h_tm1 * grad_h[s];

            // Backprop through recurrence
            grad_h[s] = A_bar * grad_h[s];
        }

        grad_x[(bN + t) * d_inner + j]  = static_cast<ActT>(grad_x_acc);
        grad_dt[(bN + t) * d_inner + j] = static_cast<ActT>(grad_dt_acc);
    }

    // Accumulate grad_A_log across batch via atomicAdd
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        atomicAdd(&grad_A_log[j * d_state + s], grad_A_acc[s]);
}

// ── Backward dispatch ─────────────────────────────────────────────────

template <typename ActT>
cudaError_t selective_scan_backward(
    const ActT* grad_y,
    const ActT* x, const ActT* dt, const ActT* A_log,
    const ActT* B, const ActT* C,
    const float* state_save,
    ActT* grad_x, ActT* grad_dt, float* grad_A_log,
    ActT* grad_B, ActT* grad_C,
    int batch, int seq_len, int d_inner, int d_state,
    cudaStream_t stream
) {
    cudaMemsetAsync(grad_A_log, 0, d_inner * d_state * sizeof(float), stream);
    int block = min(d_inner, 128);
    dim3 grid((d_inner + block - 1) / block, batch);
    scan_backward_kernel<ActT><<<grid, block, 0, stream>>>(
        grad_y, x, dt, A_log, B, C, state_save,
        grad_x, grad_dt, grad_A_log, grad_B, grad_C,
        seq_len, d_inner, d_state);
    return cudaGetLastError();
}

}}}}  // namespace sg::sm90::models::mamba_adapter
// ── end inlined csrc/scan/mamba_scan_adapter.cuh ──

#ifdef WITH_CUTLASS
// ── inlined from former csrc/backends/cuda/sm_90/mma.cuh ──
// CUDA sm_90 matrix-multiply accelerator wrappers (CUTLASS).
// Renamed from csrc/kernels/cuda/_cutlass_gemm.cuh.
//
// Used by Muon (Newton-Schulz GEMMs) and SuperGrok v2 (dt_proj fused
// softplus). Gated behind -DWITH_CUTLASS. Without the flag, Muon falls
// back to cuBLAS (torch::mm) and SG2 uses cuBLAS + a separate softplus
// kernel — slightly slower but fully functional.
//
// Math equivalence: every helper computes C = A * B with FP32 accumulate
// and FP32 output, matching cuBLAS GemmEx with CUBLAS_COMPUTE_32F.

#ifdef WITH_CUTLASS

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>

namespace sg { namespace cuda_sm90 { namespace mma {

// FP16 in / FP32 acc / FP32 out, row-major A * row-major B, row-major C.
inline cudaError_t gemm_fp16(
    int M, int N, int K,
    const __half* A, const __half* B, float* C,
    cudaStream_t stream)
{
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = float;
    using ElementAcc = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAcc>;

    typename Gemm::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(A), K},
        {reinterpret_cast<const ElementB*>(B), N},
        {C, N},
        {C, N},
        {ElementAcc(1.0f), ElementAcc(0.0f)});

    Gemm op;
    cutlass::Status st = op(args, nullptr, stream);
    return (st == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// BF16 in / FP32 acc / FP32 out variant.
inline cudaError_t gemm_bf16(
    int M, int N, int K,
    const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
    cudaStream_t stream)
{
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = float;
    using ElementAcc = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAcc>;

    typename Gemm::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(A), K},
        {reinterpret_cast<const ElementB*>(B), N},
        {C, N},
        {C, N},
        {ElementAcc(1.0f), ElementAcc(0.0f)});

    Gemm op;
    cutlass::Status st = op(args, nullptr, stream);
    return (st == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// Softplus+bias post-pass (used by SG2 dt_proj fused path).
static __global__ void softplus_bias_kernel(
    float* __restrict__ C, const float* __restrict__ bias,
    int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        float val = C[idx] + bias[col];
        C[idx] = (val > 20.0f) ? val : logf(1.0f + expf(val));
    }
}

inline void launch_softplus_bias(
    float* C, const float* bias, int M, int N,
    cudaStream_t stream)
{
    if (!bias) return;
    int total = M * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    softplus_bias_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
}

// Fused dt_proj: GEMM + softplus+bias in one call.
inline cudaError_t dt_proj_fused_with_bias(
    int M, int N, int K,
    const __half* A, const __half* B,
    const float* bias, float* C,
    cudaStream_t stream)
{
    cudaError_t err = gemm_fp16(M, N, K, A, B, C, stream);
    if (err != cudaSuccess) return err;
    launch_softplus_bias(C, bias, M, N, stream);
    return cudaSuccess;
}

}}} // namespace sg::cuda_sm90::mma

#else  // !WITH_CUTLASS

#error "CUTLASS not enabled. Use cuBLAS path or build with -DWITH_CUTLASS."

#endif // WITH_CUTLASS
// ── end inlined csrc/backends/cuda/sm_90/mma.cuh ──
#endif

namespace sg { namespace cuda_sm90 {

namespace prim = ::sg::cuda_sm90::primitives;

// =========================================================================
//  Forward kernel 1: input projection + sort
// =========================================================================

template <typename scalar_t>
__global__ void sg2_input_proj_sort_kernel(
    const scalar_t* grad, const scalar_t* sharpness,
    float* x_out, float* sort_keys, int* sort_indices,
    const float* proj_W, const float* proj_b,
    int N, int d_model
) {
    const int idx = prim::grid_stride_index();
    ::sg::algorithms::sg2_input_proj_sort(
        grad, sharpness, x_out, sort_keys, sort_indices,
        proj_W, proj_b, idx, N, d_model);
}

// =========================================================================
//  Forward kernel 2: sequential mamba scan (one thread per d_inner)
// =========================================================================

__global__ void sg2_mamba3_scan_kernel(
    float* h_state, const float* A, const float* freq,
    const float* x, const float* dt, const float* B, const float* C,
    const float* D, const float* z,
    float* y_out, int T, int d_state, int d_inner
) {
    const int di = blockIdx.x * blockDim.x + threadIdx.x;
    if (di >= d_inner) return;

    float* h = h_state + di * d_state;
    for (int t = 0; t < T; t++) {
        const float x_val = x[t * d_inner + di];
        const float dt_val = dt[t * d_inner + di];
        const float* B_vals = B + t * d_state;
        const float* C_vals = C + t * d_state;
        const float D_val = D[di];
        const float z_val = z[t * d_inner + di];

        float y;
        ::sg::algorithms::sg2_mamba3_scan_step(
            h, A + di * d_state, freq,
            x_val, dt_val, B_vals, C_vals, D_val, z_val,
            d_state, t, &y);
        y_out[t * d_inner + di] = y;
    }
}

// =========================================================================
//  Forward kernel 3 + 4: GRU + PEER + apply tail
//  PEER routing's gather/scatter happens in host code; this kernel
//  consumes the routed expert output and runs GRU + smart_grad + Adam.
// =========================================================================

template <typename ParamT, typename GradT>
__global__ void sg2_apply_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* mu_state,
    const GradT* grad, const float* expert_out,
    float alpha, float gru_decay,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        ::sg::algorithms::sg2_apply_step(
            param, exp_avg, exp_avg_sq, mu_state, grad, expert_out[i],
            alpha, gru_decay, lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

// =========================================================================
//  Backward kernel: bilevel precompute (one thread per timestep)
// =========================================================================

__global__ void sg2_bilevel_precompute_kernel(
    const float* x_sorted,
    const float* in_proj_W, const float* dt_proj_W, const float* dt_proj_b,
    const float* B_proj_W, const float* C_proj_W,
    float* pre_x, float* pre_z, float* pre_dt, float* pre_B, float* pre_C,
    int T, int d_model, int d_inner, int d_state
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    ::sg::algorithms::sg2_bilevel_precompute_timestep(
        x_sorted + t * d_model,
        in_proj_W, dt_proj_W, dt_proj_b, B_proj_W, C_proj_W,
        pre_x + t * d_inner, pre_z + t * d_inner, pre_dt + t * d_inner,
        pre_B + t * d_state, pre_C + t * d_state,
        d_model, d_inner, d_state);
}

// =========================================================================
//  Host-side launchers
// =========================================================================

void launch_supergrok2_input_proj_sort(
    const torch::Tensor& grad, const torch::Tensor& sharpness,
    torch::Tensor& x_out, torch::Tensor& sort_keys, torch::Tensor& sort_indices,
    const torch::Tensor& proj_W, const torch::Tensor& proj_b
) {
    const int N = grad.numel();
    const int d_model = proj_W.size(0);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = (N + block - 1) / block;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "sg2_input_proj_sort", [&] {
            sg2_input_proj_sort_kernel<scalar_t><<<grid, block, 0, stream>>>(
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<scalar_t>(),
                x_out.data_ptr<float>(),
                sort_keys.data_ptr<float>(),
                sort_indices.data_ptr<int>(),
                proj_W.data_ptr<float>(),
                proj_b.data_ptr<float>(),
                N, d_model);
        });
}

void launch_supergrok2_apply(
    torch::Tensor& param, torch::Tensor& exp_avg, torch::Tensor& exp_avg_sq,
    torch::Tensor& mu_state, const torch::Tensor& grad,
    const torch::Tensor& expert_out,
    float alpha, float gru_decay,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg2_apply", [&] {
            sg2_apply_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                mu_state.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                expert_out.data_ptr<float>(),
                alpha, gru_decay, lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}

// ═════════════════════════════════════════════════════════════════════════
//  MoE/Adam multi-tensor — folded in from former launch_moe_adam.cu.
//
//  For Mixture-of-Experts models, this launcher accepts a packed buffer
//  containing only the active subset of expert parameters. The caller is
//  responsible for gathering active parameters before the call and
//  scattering results after. Otherwise this is identical to AdamW.
//  The per-element math lives in supergrok2.h::moe_adam_step (which
//  re-exports adamw.h::adamw_step).
// ═════════════════════════════════════════════════════════════════════════

using ::sg::algorithms::moe_adam_step;

template <typename ParamT, typename GradT>
__global__ void moe_adam_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const GradT* grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        moe_adam_step(param, exp_avg, exp_avg_sq, grad,
                      lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

void launch_moe_adam_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const torch::Tensor& grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "moe_adam_step", [&] {
            moe_adam_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}

}} // namespace sg::cuda_sm90
