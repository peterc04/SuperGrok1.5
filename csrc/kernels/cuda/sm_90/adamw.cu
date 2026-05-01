// =====================================================================
//  csrc/kernels/cuda/sm_90/adamw.cu
//
//  sm_90 (Hopper) AdamW kernel — explicit instantiation TU.
//
//  All kernel + launcher logic lives in adamw.cuh as templates. This TU
//  forces emission of the dtype matrix below into a single object file
//  so callers outside this TU can link against the per-tensor entry
//  point without dragging the whole header into a non-CUDA source.
//
//  Dtype matrix (per main spec):
//    ParamT in {float, __nv_bfloat16, __half}
//    StateT in {float, __nv_bfloat16}
//    GradT  in {float, __nv_bfloat16, __half,
//               __nv_fp8_e4m3, __nv_fp8_e5m2}
//
//  Incoherent combos (FP8 grad with FP32 param, FP8 storage on
//  param/state) are statically rejected by the templates — those rows
//  are simply absent from the instantiation list below.
//
//  The vector wrapper sg::sm90::launch_fused_adamw_simple is defined
//  `inline` in the header, so it does not need explicit instantiation.
//  ODR is preserved because the only TU including the header for the
//  sm_90 build is this one (and the vector wrapper is inline anyway).
// =====================================================================

// Force the vector wrapper body to be emitted into THIS TU (single
// strong definition; bindings/grokadamw.cpp resolves against it).
#define SG_SM90_ADAMW_DEFINE_LAUNCHER 1

#include "csrc/kernels/cuda/sm_90/adamw.cuh"

namespace sg { namespace sm90 { namespace adamw {

// ---------------------------------------------------------------------
// FP32 param family
// ---------------------------------------------------------------------
template cudaError_t launch_fused_adamw_simple_step<float, float, float>(
    float*, float*, float*, const float*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<float, float, __nv_bfloat16>(
    float*, float*, float*, const __nv_bfloat16*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<float, float, __half>(
    float*, float*, float*, const __half*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

// FP32 param + BF16 state — used for memory-pressured mid-precision setups
template cudaError_t launch_fused_adamw_simple_step<float, __nv_bfloat16, float>(
    float*, __nv_bfloat16*, __nv_bfloat16*, const float*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<float, __nv_bfloat16, __nv_bfloat16>(
    float*, __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<float, __nv_bfloat16, __half>(
    float*, __nv_bfloat16*, __nv_bfloat16*, const __half*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

// FP32 param explicitly EXCLUDES FP8 grads (rejected by static_assert
// is_coherent_combo<float, FP8>).

// ---------------------------------------------------------------------
// BF16 param family — full grad cross-section (incl. FP8 grads).
// ---------------------------------------------------------------------
template cudaError_t launch_fused_adamw_simple_step<__nv_bfloat16, float, float>(
    __nv_bfloat16*, float*, float*, const float*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__nv_bfloat16, float, __nv_bfloat16>(
    __nv_bfloat16*, float*, float*, const __nv_bfloat16*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__nv_bfloat16, float, __half>(
    __nv_bfloat16*, float*, float*, const __half*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__nv_bfloat16, float, __nv_fp8_e4m3>(
    __nv_bfloat16*, float*, float*, const __nv_fp8_e4m3*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__nv_bfloat16, float, __nv_fp8_e5m2>(
    __nv_bfloat16*, float*, float*, const __nv_fp8_e5m2*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__nv_bfloat16, __nv_bfloat16, float>(
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, const float*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__nv_bfloat16, __nv_bfloat16, __half>(
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, const __half*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3>(
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, const __nv_fp8_e4m3*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__nv_bfloat16, __nv_bfloat16, __nv_fp8_e5m2>(
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, const __nv_fp8_e5m2*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

// ---------------------------------------------------------------------
// FP16 param family — full grad cross-section.
// ---------------------------------------------------------------------
template cudaError_t launch_fused_adamw_simple_step<__half, float, float>(
    __half*, float*, float*, const float*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__half, float, __nv_bfloat16>(
    __half*, float*, float*, const __nv_bfloat16*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__half, float, __half>(
    __half*, float*, float*, const __half*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__half, float, __nv_fp8_e4m3>(
    __half*, float*, float*, const __nv_fp8_e4m3*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__half, float, __nv_fp8_e5m2>(
    __half*, float*, float*, const __nv_fp8_e5m2*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__half, __nv_bfloat16, float>(
    __half*, __nv_bfloat16*, __nv_bfloat16*, const float*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__half, __nv_bfloat16, __nv_bfloat16>(
    __half*, __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__half, __nv_bfloat16, __half>(
    __half*, __nv_bfloat16*, __nv_bfloat16*, const __half*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__half, __nv_bfloat16, __nv_fp8_e4m3>(
    __half*, __nv_bfloat16*, __nv_bfloat16*, const __nv_fp8_e4m3*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

template cudaError_t launch_fused_adamw_simple_step<__half, __nv_bfloat16, __nv_fp8_e5m2>(
    __half*, __nv_bfloat16*, __nv_bfloat16*, const __nv_fp8_e5m2*,
    float, float, float, float, float, float, float,
    int64_t, int64_t, cudaStream_t);

}}} // namespace sg::sm90::adamw
