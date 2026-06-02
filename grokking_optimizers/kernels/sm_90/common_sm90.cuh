#ifndef GROKKING_COMMON_SM90_CUH_
#define GROKKING_COMMON_SM90_CUH_

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>
#include <type_traits>

namespace grokking { namespace sm90 {

enum class NanPolicy : int { kNone = 0, kZero = 1, kPropagate = 2 };

// =========================================================================
//  Stage-1 NaN/Inf gradient sanitization (Phase 8).
//
//  SuperGrok2 sanitizes its gradients (torch.nan_to_num) so a single Inf/NaN
//  in one element cannot poison the second moment (exp_avg_sq) and from there
//  every subsequent step. The sm_90 OPTIMIZER kernels did NOT. This adds an
//  OPTIONAL, cheap, per-element guard: when enabled, a non-finite gradient is
//  treated as 0 BEFORE the canonical per-element step reads it — exactly the
//  nan_to_num(nan=0, posinf=0, neginf=0) semantics, applied at the gradient
//  read boundary only (the optimizer math stays single-sourced in
//  csrc/algorithms/<opt>.h; nothing is re-typed here).
//
//  ENABLEMENT (compile-time): the default is ON — non-finite gradients are
//  zeroed before the per-element step, preventing a single NaN/Inf from
//  permanently poisoning exp_avg_sq or the parameter.  Build with
//  -DSG_SANITIZE_NONFINITE=0 to revert to the legacy (pass-through) behavior.
//
//  The helpers operate on the FLOAT gradient value the kernel already
//  materializes (after the static_cast<float> the canonical step would do), so
//  they are dtype-agnostic (fp32 vec4 + bf16/half scalar) and add at most one
//  isfinite-select per element on the hot path.
// =========================================================================
#ifndef SG_SANITIZE_NONFINITE
#define SG_SANITIZE_NONFINITE 1
#endif

// Scalar: returns g when finite, else 0. Identity when the guard is off.
__device__ __forceinline__ float sg_sanitize_grad(float g) {
#if SG_SANITIZE_NONFINITE
    return isfinite(g) ? g : 0.0f;
#else
    return g;
#endif
}

// float4 (vec4 fast path): sanitize all four gradient lanes in place. Identity
// (no-op) when the guard is off, so the vec4 kernels keep their exact behavior.
__device__ __forceinline__ void sg_sanitize_grad4(float4& g) {
#if SG_SANITIZE_NONFINITE
    g.x = isfinite(g.x) ? g.x : 0.0f;
    g.y = isfinite(g.y) ? g.y : 0.0f;
    g.z = isfinite(g.z) ? g.z : 0.0f;
    g.w = isfinite(g.w) ? g.w : 0.0f;
#else
    (void)g;
#endif
}

// In-place scalar sanitize at the gradient buffer (for kernels whose canonical
// step reads grad[idx] internally, so the float value cannot be intercepted in
// a register before the call). When the guard is OFF this expands to NOTHING —
// the const grad buffer is never written and the default path is byte-identical.
// When ON, the non-finite element is zeroed in the grad buffer right before the
// step reads it. `gptr` is the (possibly const) grad pointer; `gidx` the index.
#if SG_SANITIZE_NONFINITE
#define SG_SANITIZE_GRAD_INPLACE(gptr, gidx)                                   \
    do {                                                                       \
        auto* _sg_g = const_cast<                                              \
            std::remove_const_t<std::remove_pointer_t<decltype(gptr)>>*>(gptr);\
        float _sg_v = static_cast<float>(_sg_g[(gidx)]);                       \
        if (!isfinite(_sg_v)) _sg_g[(gidx)] = static_cast<                     \
            std::remove_const_t<std::remove_pointer_t<decltype(gptr)>>>(0.0f); \
    } while (0)
#else
#define SG_SANITIZE_GRAD_INPLACE(gptr, gidx) do { } while (0)
#endif

// float4 in-place variant: for vec4 fast paths whose canonical *_step_vec4
// reads grad4[i] internally (so the lanes cannot be sanitized in a register
// before the call). When OFF this expands to nothing (the const grad4 buffer is
// never touched). When ON it zeroes any non-finite lane in grad4[i] in place.
#if SG_SANITIZE_NONFINITE
#define SG_SANITIZE_GRAD4_INPLACE(g4ptr, gidx)                                 \
    do {                                                                       \
        float4* _sg_g4 = const_cast<float4*>(g4ptr);                           \
        float4 _sg_v4 = _sg_g4[(gidx)];                                        \
        ::grokking::sm90::sg_sanitize_grad4(_sg_v4);                           \
        _sg_g4[(gidx)] = _sg_v4;                                               \
    } while (0)
#else
#define SG_SANITIZE_GRAD4_INPLACE(g4ptr, gidx) do { } while (0)
#endif

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
