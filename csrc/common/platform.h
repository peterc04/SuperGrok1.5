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

#pragma once

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
