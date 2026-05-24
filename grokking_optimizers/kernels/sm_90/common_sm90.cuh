#ifndef GROKKING_COMMON_SM90_CUH_
#define GROKKING_COMMON_SM90_CUH_

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>

namespace grokking { namespace sm90 {

enum class NanPolicy : int { kNone = 0, kZero = 1, kPropagate = 2 };

template <typename T>
__device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }

template <>
__device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v) { return static_cast<T>(v); }

template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half_rn(v); }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

}} // namespace grokking::sm90

#endif // GROKKING_COMMON_SM90_CUH_
