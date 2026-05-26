#ifndef GROKKING_COMMON_GFX942_HIP_HPP_
#define GROKKING_COMMON_GFX942_HIP_HPP_

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>

namespace grokking { namespace gfx942 {

enum class NanPolicy : int { kNone = 0, kZero = 1, kPropagate = 2 };

template <typename T>
__device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }

template <>
__device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }

template <>
__device__ __forceinline__ float to_float<hip_bfloat16>(hip_bfloat16 v) { return static_cast<float>(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v) { return static_cast<T>(v); }

template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half_rn(v); }

template <>
__device__ __forceinline__ hip_bfloat16 from_float<hip_bfloat16>(float v) { return hip_bfloat16(v); }

template <NanPolicy NP>
__forceinline__ __device__ float apply_nan_policy(float g) {
    if constexpr (NP == NanPolicy::kZero) {
        return __builtin_isnan(g) ? 0.0f : g;
    } else {
        return g;
    }
}

template <bool ENABLE_CLIP>
__forceinline__ __device__ float apply_clip(float g, float clip_threshold) {
    if constexpr (ENABLE_CLIP) {
        return fminf(fmaxf(g, -clip_threshold), clip_threshold);
    } else {
        (void)clip_threshold;
        return g;
    }
}

}} // namespace grokking::gfx942

#endif // GROKKING_COMMON_GFX942_HIP_HPP_
