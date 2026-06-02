#ifndef GROKKING_COMMON_GFX942_HIP_HPP_
#define GROKKING_COMMON_GFX942_HIP_HPP_

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>

namespace grokking { namespace gfx942 {

enum class NanPolicy : int { kNone = 0, kZero = 1, kPropagate = 2 };

// =========================================================================
//  Stage-1 NaN/Inf gradient sanitization (Phase 8) — gfx942 mirror of the
//  sm_90 mechanism (grokking_optimizers/kernels/sm_90/common_sm90.cuh).
//
//  When enabled, a non-finite gradient is treated as 0 BEFORE the per-element
//  step reads it (nan_to_num(nan=0,posinf=0,neginf=0) semantics at the gradient
//  read boundary only — the optimizer math stays single-sourced). The default
//  is OFF, so sg_sanitize_grad() is the identity and the device kernels are
//  byte-for-byte the prior behavior; build with -DSG_SANITIZE_NONFINITE=1 to
//  turn it on for every gfx942 optimizer apply kernel. Cheap: at most one
//  __builtin_isfinite-select per element on the hot path.
// =========================================================================
#ifndef SG_SANITIZE_NONFINITE
#define SG_SANITIZE_NONFINITE 0
#endif

// Returns g when finite, else 0. Identity (zero added instructions) when OFF.
__device__ __forceinline__ float sg_sanitize_grad(float g) {
#if SG_SANITIZE_NONFINITE
    return __builtin_isfinite(g) ? g : 0.0f;
#else
    return g;
#endif
}

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
