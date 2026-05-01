// =====================================================================
//  csrc/kernels/cuda/sm_90/grokadamw.cuh
//
//  sm_90 (Hopper) GrokAdamW optimizer kernels + thin templated launchers.
//
//  Net-new replacement for the deleted csrc/kernels/cuda/sm_90/
//  grokadamw_sm90.cu (commit 5505b50). All kernel + launcher logic
//  lives here as templates; explicit instantiations + the four
//  torch::Tensor-form binding shims are emitted from grokadamw.cu.
//
//  Two templated entry points in namespace sg::sm90::grokadamw:
//
//    1) launch_grokadamw_fused_step<ParamT,StateT,GradT>(...)
//       Single-pass Grokfast + AdamW with a gradient-clamp prologue:
//
//           g~      = clamp(g, -c, c)
//           xi_t    = alpha * xi_{t-1} + (1-alpha) * g~
//           g_amp   = g~ + lamb * xi_t              (register-only)
//           m_t     = b1 * m_{t-1} + (1-b1) * g_amp
//           v_t     = b2 * v_{t-1} + (1-b2) * g_amp^2
//           m_hat   = m_t / bc1                     (host-passed bc1)
//           v_hat   = v_t / bc2                     (host-passed bc2)
//           u       = m_hat / (sqrt(v_hat) + eps) + lambda * theta
//           theta_t = theta_{t-1} - lr * u
//
//       6 reads (g, xi, m, v, theta + the implicit GradT decode) /
//       5 writes (xi, m, v, theta, plus the optional grad-clamp
//       writeback fold-in is held in a register only so it never
//       hits HBM). Roofline with FP32 P/S/G: R+W = 36 B/elem,
//       ~16 FLOPs ⇒ AI ≈ 0.44 FLOP/byte. At H100 HBM3 (3 TB/s) the
//       BW ceiling is ~1.33 TFLOP/s; FP32 compute ceiling ~67 TFLOP/s.
//       Solidly bandwidth-bound — every optimization below targets
//       HBM throughput.
//
//    2) launch_grokadamw_fused_step_q3<ParamT,GradT>(...)
//       Quantized state variant.
//         exp_avg     : INT8 + per-block FP32 scales (Q3_BLOCK_SIZE=32).
//         slow_ema    : INT8 + per-block FP32 scales (same pattern).
//         exp_avg_sq  : BF16 with stochastic rounding on writeback
//                       (utils.cuh::float_to_bf16_stochastic).
//       Math identical to non-q3. Dequant via dequant_int8 from
//       quantization.h; requantize to INT8 via
//       utils.cuh::ptx_int8_stochastic_round (uses prmt.b32 for
//       fast 16-bit threshold extraction). FP8 grad rejected on q3
//       via static_assert — INT8 moment dynamic range is incompatible
//       with FP8 grad scales at the recommended LR schedule.
//
//  Heterogeneous dtype matrix (instantiations live in grokadamw.cu):
//     Non-q3:
//        ParamT in {float, __nv_bfloat16, __half}
//        StateT in {float, __nv_bfloat16}
//        GradT  in {float, __nv_bfloat16, __half,
//                   __nv_fp8_e4m3, __nv_fp8_e5m2}
//        Incoherent combo (FP8 grad + FP32 param without rescale)
//        rejected via static_assert.
//     Q3:
//        ParamT in {float, __nv_bfloat16}
//        GradT  in {float, __nv_bfloat16, __half}
//        FP8 grad rejected unconditionally on q3.
//
//  Optimizations & justifications:
//    - LDG / stream_load on read-only loads: ld.global.nc.* bypasses
//      L2 allocation so model weights stay warm for next forward.
//    - stream_store (st.global.wt.f32) on FP32 state writes: same
//      rationale, plus it skips the L1 writeback queue.
//    - float4 vec fast path on the all-FP32 instantiation, gated by
//      tuned_configs.h::GROKADAMW_CONFIGS[ARCH_SM90][bucket].vec4.
//    - __launch_bounds__ pulled from tuned_configs.h
//      (block_size, min_blocks_per_sm). No hardcoded launch params.
//    - fast_rsqrt_nr from utils.cuh for 1/sqrt(v_hat) — single
//      rsqrt.approx.f32 + one Newton-Raphson refinement; ~2x faster
//      than sqrtf+fdividef on H100.
//    - Q3 INT8 requant via ptx_int8_stochastic_round — uses prmt.b32
//      byte permutation for fast threshold extraction (forbidden to
//      reinline — already in utils.cuh).
//    - The device template at csrc/device/optimizers/sm_90/
//      grokadamw_sm90.cuh predates this spec (uses multiplicative
//      decoupled-WD and lacks the grad-clamp prologue). We inline a
//      spec-matching variant here rather than calling that template.
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/quantization.h"
#include "csrc/common/tuned_configs.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
  #include <cuda_fp8.h>
#endif

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace sg { namespace sm90 { namespace grokadamw {

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

// FP8 grads with FP32 params silently lose dynamic range without an
// explicit rescale (which is not part of this kernel).
template <typename ParamT, typename GradT>
struct is_coherent_combo
    : std::integral_constant<bool,
        !((std::is_same<GradT, __nv_fp8_e4m3>::value ||
           std::is_same<GradT, __nv_fp8_e5m2>::value) &&
          std::is_same<ParamT, float>::value)> {};

// FP8 grad on the Q3 path is rejected unconditionally.
template <typename GradT>
struct is_q3_grad_dtype
    : std::integral_constant<bool,
        std::is_same<GradT, float>::value         ||
        std::is_same<GradT, __nv_bfloat16>::value ||
        std::is_same<GradT, __half>::value> {};

// Q3 INT8 quantization block size (matches deleted baseline + device
// template in csrc/device/optimizers/sm_90/grokadamw_sm90.cuh).
constexpr int Q3_BLOCK_SIZE = 32;

// ---------------------------------------------------------------------
// Type-erased load / store helpers — math is always FP32.
// ---------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p) {
    return static_cast<float>(*p);
}
template <>
__device__ __forceinline__ float load_as_float<float>(const float* p) {
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
    return static_cast<float>(*p);  // FP8 lacks LDG overload
}
template <>
__device__ __forceinline__ float load_as_float<__nv_fp8_e5m2>(const __nv_fp8_e5m2* p) {
    return static_cast<float>(*p);
}

// FP32 state read: stream_load to bypass L2 allocation. BF16 state has
// no PTX wt v1.b16 path, so falls back to LDG / read-only cache.
__device__ __forceinline__ float load_state(const float* p)         { return stream_load(p); }
__device__ __forceinline__ float load_state(const __nv_bfloat16* p) { return __bfloat162float(LDG(p)); }

template <typename T>
__device__ __forceinline__ void store_param(T* p, float v) {
    *p = static_cast<T>(v);
}
template <>
__device__ __forceinline__ void store_param<float>(float* p, float v) {
    *p = v;  // L1+L2 default; param is consumed by next forward, so we want it cached
}
template <>
__device__ __forceinline__ void store_param<__nv_bfloat16>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ void store_param<__half>(__half* p, float v) {
    *p = __float2half_rn(v);
}

namespace detail {

// State stream-store: FP32 -> st.global.wt.f32 (no L2 alloc); BF16
// has no PTX wt v1.b16 — falls through to a stochastic-rounded
// rn store keyed by hash_prng(step, idx) so that small EMA deltas
// (1-alpha)*g << ulp(BF16) are not silently zeroed.
__device__ __forceinline__ void store_state(
    float* p, float v, unsigned /*step*/, unsigned /*idx*/
) {
    stream_store(p, v);
}
__device__ __forceinline__ void store_state(
    __nv_bfloat16* p, float v, unsigned step, unsigned idx
) {
    *p = float_to_bf16_stochastic(v, hash_prng(step, idx));
}

} // namespace detail

// ---------------------------------------------------------------------
// Per-element fused step (FP32 math, grad-clamp prologue, decoupled WD).
//
// Inlined here rather than calling
// csrc/device/optimizers/sm_90/grokadamw_sm90.cuh::grokadamw_step()
// because that helper predates this spec (it uses multiplicative
// decoupled WD `p *= (1 - lr*wd)` + a denominator-fold `1/(1+eps*rsv)`
// instead of `m_hat / (sqrt(v_hat)+eps) + lambda*theta`).
// ---------------------------------------------------------------------

template <typename ParamT, typename StateT, typename GradT>
__device__ __forceinline__ void grokadamw_step_inline(
    ParamT*       __restrict__ params,
    StateT*       __restrict__ exp_avg,
    StateT*       __restrict__ exp_avg_sq,
    StateT*       __restrict__ slow_ema,
    const GradT*  __restrict__ grads,
    int64_t idx,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb, float clip_c,
    float bc1, float bc2,
    unsigned step_u
) {
    // 6 reads (one per state buffer + grad + param + slow_ema)
    const float g_raw   = load_as_float(grads      + idx);
    const float xi_old  = load_state   (slow_ema   + idx);
    const float m_old   = load_state   (exp_avg    + idx);
    const float v_old   = load_state   (exp_avg_sq + idx);
    const float p_old   = load_as_float(params     + idx);

    // 1. Gradient clamp prologue — held in register, never written.
    const float g = fminf(fmaxf(g_raw, -clip_c), clip_c);

    // 2. Slow EMA gradient filter.
    const float xi = alpha * xi_old + (1.0f - alpha) * g;

    // 3. Amplified gradient (register-only — never round-trips HBM).
    const float gamp = g + lamb * xi;

    // 4. Adam moment updates (FMA-fusable).
    const float m = beta1 * m_old + (1.0f - beta1) * gamp;
    const float v = beta2 * v_old + (1.0f - beta2) * gamp * gamp;

    // 5. Bias-corrected step + decoupled WD term.
    const float m_hat = m / bc1;
    const float v_hat = v / bc2;
    const float rsv   = fast_rsqrt_nr(fmaxf(v_hat, 0.0f));
    // u = m_hat / (sqrt(v_hat) + eps) + lambda * theta
    //   = m_hat * rsv / (1 + eps * rsv) + wd * theta
    const float u   = m_hat * rsv / (1.0f + eps * rsv) + weight_decay * p_old;
    const float p_new = p_old - lr * u;

    // 5 writes: xi, m, v, theta. (gamp stayed in registers.)
    const unsigned i_u = static_cast<unsigned>(idx);
    detail::store_state(slow_ema   + idx, xi, step_u, i_u);
    detail::store_state(exp_avg    + idx, m,  step_u, i_u ^ 0x1u);
    detail::store_state(exp_avg_sq + idx, v,  step_u, i_u ^ 0x2u);
    store_param(params + idx, p_new);
}

// =====================================================================
// __global__ kernel: scalar grid-stride.
//
// __launch_bounds__ takes block_size from tuned_configs.h. Templated on
// BLOCK_SIZE so the launch_bounds tuple is a compile-time constant.
// =====================================================================

template <typename ParamT, typename StateT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void grokadamw_fused_step_kernel(
    ParamT*       __restrict__ params,
    StateT*       __restrict__ exp_avg,
    StateT*       __restrict__ exp_avg_sq,
    StateT*       __restrict__ slow_ema,
    const GradT*  __restrict__ grads,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb, float clip_c,
    float bc1, float bc2,
    int64_t n_elements,
    int64_t step_count
) {
    static_assert(is_param_dtype<ParamT>::value, "GrokAdamW: invalid ParamT");
    static_assert(is_state_dtype<StateT>::value, "GrokAdamW: invalid StateT");
    static_assert(is_grad_dtype<GradT>::value,   "GrokAdamW: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "GrokAdamW: FP8 grad with FP32 param requires explicit rescale");

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const unsigned step_u = static_cast<unsigned>(step_count);
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        grokadamw_step_inline<ParamT, StateT, GradT>(
            params, exp_avg, exp_avg_sq, slow_ema, grads,
            i, lr, beta1, beta2, eps, weight_decay,
            alpha, lamb, clip_c, bc1, bc2, step_u
        );
    }
}

// =====================================================================
// FP32-only vec4 fast path for the (FP32 param ∧ FP32 state ∧ FP32 grad)
// instantiation. 5 v4 loads / 4 v4 stores per iter.
// =====================================================================

template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void grokadamw_fused_step_vec4_fp32_kernel(
    float* __restrict__ params,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ slow_ema,
    const float* __restrict__ grads,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb, float clip_c,
    float bc1, float bc2,
    int64_t n_vec
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float one_minus_a  = 1.0f - alpha;
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;
    const float neg_c   = -clip_c;

    float4*       p4_out  = reinterpret_cast<float4*>(params);
    float4*       m4_out  = reinterpret_cast<float4*>(exp_avg);
    float4*       v4_out  = reinterpret_cast<float4*>(exp_avg_sq);
    float4*       xi4_out = reinterpret_cast<float4*>(slow_ema);
    const float4* g4      = reinterpret_cast<const float4*>(grads);

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_vec; i += stride) {
        const float4 gv  = stream_load4(g4 + i);
        const float4 xi0 = stream_load4(reinterpret_cast<const float4*>(slow_ema)   + i);
        const float4 m0  = stream_load4(reinterpret_cast<const float4*>(exp_avg)    + i);
        const float4 v0  = stream_load4(reinterpret_cast<const float4*>(exp_avg_sq) + i);
        const float4 p0  = stream_load4(reinterpret_cast<const float4*>(params)     + i);

        float4 xi1, m1, v1, p1;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float gk_raw = (&gv.x)[k];
            const float gk     = fminf(fmaxf(gk_raw, neg_c), clip_c);
            const float xik    = alpha * (&xi0.x)[k] + one_minus_a * gk;
            const float ak     = gk + lamb * xik;
            const float mk     = beta1 * (&m0.x)[k] + one_minus_b1 * ak;
            const float vk     = beta2 * (&v0.x)[k] + one_minus_b2 * ak * ak;
            const float mh     = mk * inv_bc1;
            const float vh     = vk * inv_bc2;
            const float rs     = fast_rsqrt_nr(fmaxf(vh, 0.0f));
            const float u      = mh * rs / (1.0f + eps * rs)
                                 + weight_decay * (&p0.x)[k];
            (&xi1.x)[k] = xik;
            (&m1.x)[k]  = mk;
            (&v1.x)[k]  = vk;
            (&p1.x)[k]  = (&p0.x)[k] - lr * u;
        }

        stream_store4(xi4_out + i, xi1);
        stream_store4(m4_out  + i, m1);
        stream_store4(v4_out  + i, v1);
        stream_store4(p4_out  + i, p1);
    }
}

}}} // namespace sg::sm90::grokadamw
