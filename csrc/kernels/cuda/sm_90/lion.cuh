// =====================================================================
//  csrc/kernels/cuda/sm_90/lion.cuh
//
//  sm_90 (Hopper) Lion optimizer kernel + launcher header.
//
//  Algorithm (no v-state Lion variant — 3 reads / 2 writes per element):
//     c_t   = sign(beta1 * m_{t-1} + (1 - beta1) * g)
//     theta = theta - lr * (c_t + wd * theta)            // decoupled WD
//     m_t   = beta2 * m_{t-1} + (1 - beta2) * g          // EMA uses beta2
//
//  Two launchers, namespace sg::sm90, signatures matched against
//  csrc/bindings/lion.cpp DECLARE_LION(sm90):
//     void launch_fused_lion_step(torch::Tensor, torch::Tensor,
//                                 torch::Tensor, float, float, float, float);
//     void launch_multi_tensor_lion(std::vector<torch::Tensor>&,
//                                   std::vector<torch::Tensor>&,
//                                   std::vector<torch::Tensor>&,
//                                   float, float, float, float);
//
//  Roofline: with FP32 params/state/grad the BW per element is
//     R = 3 * 4 B  reads (param, exp_avg, grad)
//     W = 2 * 4 B  writes (param, exp_avg)
//     I = 5 FLOPs (2 FMAs for interp, sign, FMA for update, FMA for ema)
//  Arithmetic intensity ~5/20 = 0.25 FLOP/byte — solidly BW-bound on H100
//  (3 TB/s HBM3, ~67 TFLOP/s FP32). The kernel is therefore tuned for
//  peak HBM throughput: vec4 loads (ld.global.nc.v4.f32), wt stores to
//  bypass L2 allocation on optimizer state, and __launch_bounds__ chosen
//  from tuned_configs.h::DEFAULT_CONFIG (BLOCK=256, MIN_BLOCKS_PER_SM=2).
//
//  Heterogeneous dtype support — all instantiations live in lion.cu:
//     ParamT in {float, __nv_bfloat16, __half}
//     StateT in {float, __nv_bfloat16}
//     GradT  in {float, __nv_bfloat16, __half,
//                __nv_fp8_e4m3, __nv_fp8_e5m2}
//  Incoherent combos (e.g. FP8 param) are rejected via static_assert.
//  All math runs in FP32; only loads/stores are typed.
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/tuned_configs.h"
#include "csrc/device/optimizers/sm_90/lion_sm90.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
  #include <cuda_fp8.h>
#endif

#include <torch/extension.h>
#include <vector>
#include <type_traits>

namespace sg { namespace sm90 { namespace lion {

// ---------------------------------------------------------------------
// Compile-time predicates for the dtype matrix.
// ---------------------------------------------------------------------

template <typename T>
struct is_param_dtype
    : std::integral_constant<bool,
        std::is_same<T, float>::value         ||
        std::is_same<T, __nv_bfloat16>::value ||
        std::is_same<T, __half>::value> {};

template <typename T>
struct is_state_dtype
    : std::integral_constant<bool,
        std::is_same<T, float>::value         ||
        std::is_same<T, __nv_bfloat16>::value> {};

template <typename T>
struct is_grad_dtype
    : std::integral_constant<bool,
        std::is_same<T, float>::value          ||
        std::is_same<T, __nv_bfloat16>::value  ||
        std::is_same<T, __half>::value         ||
        std::is_same<T, __nv_fp8_e4m3>::value  ||
        std::is_same<T, __nv_fp8_e5m2>::value> {};

// ---------------------------------------------------------------------
// Type-erased load / store helpers — all math is FP32.
// ---------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p) {
    return static_cast<float>(*p);
}
template <>
__device__ __forceinline__ float load_as_float<float>(const float* p) {
    // ld.global.nc on read-only state/grad is preferred for 3R/2W kernels.
    return LDG(p);
}
template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(LDG(p));
}
template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* p) {
    return __half2float(LDG(p));
}
template <>
__device__ __forceinline__ float load_as_float<__nv_fp8_e4m3>(const __nv_fp8_e4m3* p) {
    return static_cast<float>(*p);  // __nv_fp8_e4m3 lacks LDG overload
}
template <>
__device__ __forceinline__ float load_as_float<__nv_fp8_e5m2>(const __nv_fp8_e5m2* p) {
    return static_cast<float>(*p);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T* p, float v) {
    *p = static_cast<T>(v);
}
template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ void store_from_float<__half>(__half* p, float v) {
    *p = __float2half_rn(v);
}

// State stream-store: FP32 state goes through the wt path to bypass L2
// allocation; BF16 state has no PTX wt v1.b16 path so falls through.
__device__ __forceinline__ void store_state(float* p, float v)         { stream_store(p, v); }
__device__ __forceinline__ void store_state(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

}}} // namespace sg::sm90::lion
