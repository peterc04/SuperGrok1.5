// csrc/backends/hip/gfx942/models/mamba.hip.h
// Mamba (selective state-space) model header for gfx942 (CDNA3 / MI300X).
//
// Strategy: this is a thin wrapper around the sm_90 implementation.
// mamba.cuh is platform-portable on the cuBLAS/rocBLAS path. The
// underlying scan kernels are __device__ __forceinline__ device
// functions with no sm_90-only intrinsics (no WGMMA, no clusters, no
// TMA), so they compile cleanly under HIP. CUTLASS is gated behind
// `WITH_CUTLASS` and is NOT defined on the HIP build.
//
// The bindings (csrc/bindings/models_mamba.cpp) call
// `sg::gfx942::models::mamba::{forward,backward,selective_scan_fwd,
// selective_scan_bwd}<T>` directly when `detect_arch() == 942`.

#pragma once
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
// Bring in the full sm_90 Mamba template implementation. On HIP the
// cuBLAS includes are hipified to rocBLAS by PyTorch's CUDAExtension.
#include "csrc/backends/cuda/sm_90/models/mamba.cuh"

namespace sg { namespace gfx942 { namespace models { namespace mamba {

// -- Model configuration ------------------------------------------------------
struct ModelConfig {
    int d_model;
    int d_state;
    int d_inner;        // expansion dim (typically 2 * d_model)
    int d_conv;         // local convolution width
    int n_layers;
    int seq_len;
    int batch;
    int lds_bytes;      // LDS allocation budget
    int waves_per_eu;   // occupancy hint for CDNA3
    bool use_bf16_mfma; // BF16 MFMA fast path (d_inner >= 128)
};

// -- Forward pass (full stack) ------------------------------------------------
template <typename T>
inline cudaError_t forward(
    const T* input,
    const T* weights,
    T* output,
    T* states,
    int batch, int seq_len, int d_model, int d_state,
    int d_conv, int expand, int n_layers,
    cudaStream_t stream
) {
    return sg::sm90::models::mamba::forward<T>(
        input, weights, output, states,
        batch, seq_len, d_model, d_state,
        d_conv, expand, n_layers, stream);
}

// -- Backward pass (full stack) -----------------------------------------------
template <typename T>
inline cudaError_t backward(
    const T* grad_output,
    const T* activations_saved,
    const T* weights,
    T* grad_input,
    T* grad_weights,
    int batch, int seq_len, int d_model, int d_state,
    int d_conv, int expand, int n_layers,
    cudaStream_t stream
) {
    return sg::sm90::models::mamba::backward<T>(
        grad_output, activations_saved, weights,
        grad_input, grad_weights,
        batch, seq_len, d_model, d_state,
        d_conv, expand, n_layers, stream);
}

// -- selective_scan_fwd / bwd (component-test wrappers) -----------------------
// Signature contract (matches the binding forward declaration in
// csrc/bindings/models_mamba.cpp):
//   fwd: (u, delta, A, B,            out, state)
//   bwd: (grad_out, u, delta, A, B, state, grad_u, grad_delta, grad_A, grad_B)
template <typename T>
inline cudaError_t selective_scan_fwd(
    const T* u, const T* delta, const T* A, const T* B,
    T* out, T* state,
    int batch, int seq_len, int d_state,
    cudaStream_t stream
) {
    return sg::sm90::models::mamba::selective_scan_fwd<T>(
        u, delta, A, B, out, state,
        batch, seq_len, d_state, stream);
}

template <typename T>
inline cudaError_t selective_scan_bwd(
    const T* grad_out,
    const T* u, const T* delta, const T* A,
    const T* B, const T* state,
    T* grad_u, T* grad_delta, T* grad_A, T* grad_B,
    int batch, int seq_len, int d_state,
    cudaStream_t stream
) {
    return sg::sm90::models::mamba::selective_scan_bwd<T>(
        grad_out, u, delta, A, B, state,
        grad_u, grad_delta, grad_A, grad_B,
        batch, seq_len, d_state, stream);
}

}}}}  // namespace sg::gfx942::models::mamba
