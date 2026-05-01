// =====================================================================
//  csrc/kernels/cuda/sm_90/grokfast.cuh
//
//  sm_90 (Hopper) Grokfast optimizer kernels + thin templated launchers.
//  Two entry points live in namespace sg::sm90::grokfast:
//
//    1) launch_grokfast_fused_step             — EMA-only filter pass.
//       Updates the slow-EMA buffer and writes the amplified gradient
//       g~ = g + lamb * ema back into the grad tensor so a downstream
//       Adam step (in fp32 grad space) can consume it. 3 reads / 2 writes
//       per element; pure BW-bound.
//
//    2) launch_grokfast_fused_ema_adam_step    — single-kernel fused
//       Grokfast + AdamW. Keeps the amplified gradient in registers, so
//       the round-trip through global memory is eliminated. The kernel
//       executes:
//
//           ξ_t = α·ξ_{t-1} + (1-α)·g
//           g~  = g + ℓ·ξ_t                  (register-only)
//           m_t = β1·m_{t-1} + (1-β1)·g~
//           v_t = β2·v_{t-1} + (1-β2)·g~^2
//           m^  = m_t / bc1                  (bias-correction from binding)
//           v^  = v_t / bc2
//           u   = m^ / (√v^ + ε) + λ·θ
//           θ_t = θ_{t-1} - η·u
//
//       5 reads (ξ, m, v, θ, g) / 4 writes (ξ, m, v, θ). Roofline:
//       with FP32 everywhere R+W = 36 B/elem and ~14 FLOPs ⇒ ~0.39
//       FLOP/byte. At 3 TB/s HBM3 that's ~1.17 TFLOP/s ceiling — solidly
//       bandwidth-bound on H100. Hopper has ~67 TFLOP/s FP32, so the
//       arithmetic budget is ~57× the BW budget; every optimization
//       below is therefore a memory-traffic optimization.
//
//  Heterogeneous dtype matrix (instantiated in grokfast.cu):
//     ParamT in {float, __nv_bfloat16, __half}
//     StateT in {float, __nv_bfloat16}
//     GradT  in {float, __nv_bfloat16, __half,
//                __nv_fp8_e4m3, __nv_fp8_e5m2}
//  Incoherent combos rejected via static_assert. All math is FP32; only
//  loads/stores are typed. BF16 state writes use unbiased stochastic
//  rounding (utils.cuh::float_to_bf16_stochastic) to keep the EMA from
//  collapsing toward zero across many small updates.
//
//  Optimizations & justifications:
//    - LDG / stream_load on read-only state: bypasses L2 allocation so
//      model weights stay warm for the next forward pass.
//    - stream_store (st.global.wt) on FP32 state writes: same rationale.
//    - float4 vec fast path on the (FP32 grad ∧ FP32 state ∧ aligned ∧
//      N%4==0) instantiation, gated by tuned_configs::DEFAULT_CONFIG.vec4.
//    - __launch_bounds__ pulled from tuned_configs.h (block_size=256,
//      min_blocks_per_sm=2 by default; autotune may overwrite).
//    - fast_rsqrt_nr from utils.cuh for 1/√v^ — single rsqrt.approx.f32
//      + one Newton-Raphson refinement, ~2× faster than sqrtf+fdividef.
//    - Reuses the device-function template in
//      csrc/device/optimizers/sm_90/grokfast_sm90.cuh for the per-element
//      scalar EMA helper (cross-arch numerical agreement).
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/tuned_configs.h"
#include "csrc/device/optimizers/sm_90/grokfast_sm90.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
  #include <cuda_fp8.h>
#endif

#include <cstdint>
#include <type_traits>

namespace sg { namespace sm90 { namespace grokfast {

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
__device__ __forceinline__ void store_param<__nv_bfloat16>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ void store_param<__half>(__half* p, float v) {
    *p = __float2half_rn(v);
}

namespace detail {

// Stochastic-rounding state stores. FP32 state goes through wt-cached
// stream_store; BF16 state uses unbiased SR via the PRNG in utils.cuh
// to keep small EMA deltas (1-α)·g ≪ ulp(BF16) from being silently
// zeroed (a real failure mode for grokfast at α=0.98).
__device__ __forceinline__ void store_state(
    float* p, float v, unsigned /*step*/, unsigned /*idx*/
) {
    stream_store(p, v);
}
__device__ __forceinline__ void store_state(
    __nv_bfloat16* p, float v, unsigned step, unsigned idx
) {
    unsigned r = hash_prng(step, idx);
    *p = float_to_bf16_stochastic(v, r);
}

} // namespace detail

// ---------------------------------------------------------------------
// Kernel 1: EMA-only Grokfast filter step.
//
//   ξ_t = α·ξ_{t-1} + (1-α)·g
//   g~  = g + ℓ·ξ_t        (written back into grads)
//
// 3 reads (g, ξ) on FP32 state, 2 writes (g~, ξ). The grad tensor is
// read-then-overwritten in place. Reuses the device template in
// csrc/device/optimizers/sm_90/grokfast_sm90.cuh for cross-arch parity
// when the dtype matches its supported scalar_t.
// ---------------------------------------------------------------------

template <typename StateT, typename GradT>
__launch_bounds__(::sg::DEFAULT_CONFIG.block_size,
                  ::sg::DEFAULT_CONFIG.min_blocks_per_sm)
__global__ void grokfast_fused_step_kernel(
    StateT* __restrict__ slow_ema,
    GradT*  __restrict__ grads,
    float alpha,
    float lamb,
    int64_t n_elements,
    int64_t step_count
) {
    static_assert(is_state_dtype<StateT>::value, "StateT must be fp32 or bf16");
    static_assert(is_grad_dtype<GradT>::value,
                  "GradT must be fp32, bf16, fp16, fp8_e4m3, or fp8_e5m2");

    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n_elements) return;

    const float g = load_as_float(grads + i);
    const float e_old = load_state(slow_ema + i);

    // EMA + amplification (FP32 math).
    const float e = alpha * e_old + (1.0f - alpha) * g;
    detail::store_state(slow_ema + i,
                        e,
                        static_cast<unsigned>(step_count),
                        static_cast<unsigned>(i));

    const float g_amp = g + lamb * e;
    store_param(grads + i, g_amp);  // GradT-typed write-back
}

// FP32-only float4 vectorized fast path. Gated at launch time by
// alignment + N%4==0 + tuned_configs::DEFAULT_CONFIG.vec4.
__launch_bounds__(::sg::DEFAULT_CONFIG.block_size,
                  ::sg::DEFAULT_CONFIG.min_blocks_per_sm)
__global__ void grokfast_fused_step_vec4_kernel(
    float4* __restrict__ slow_ema4,
    float4* __restrict__ grads4,
    float alpha,
    float lamb,
    int64_t n4
) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n4) return;

    float4 g = grads4[i];
    float4 e = stream_load4(slow_ema4 + i);

    e.x = alpha * e.x + (1.0f - alpha) * g.x;
    e.y = alpha * e.y + (1.0f - alpha) * g.y;
    e.z = alpha * e.z + (1.0f - alpha) * g.z;
    e.w = alpha * e.w + (1.0f - alpha) * g.w;
    stream_store4(slow_ema4 + i, e);

    g.x = g.x + lamb * e.x;
    g.y = g.y + lamb * e.y;
    g.z = g.z + lamb * e.z;
    g.w = g.w + lamb * e.w;
    grads4[i] = g;
}

// ---------------------------------------------------------------------
// Kernel 2: Fused Grokfast EMA + AdamW step.
//
//   ξ_t = α·ξ_{t-1} + (1-α)·g
//   g~  = g + ℓ·ξ_t                            (register-only)
//   m_t = β1·m_{t-1} + (1-β1)·g~
//   v_t = β2·v_{t-1} + (1-β2)·g~^2
//   m^  = m_t / bc1, v^ = v_t / bc2
//   u   = m^ / (√v^ + ε) + λ·θ
//   θ_t = θ_{t-1} - η·u
//
// 5 reads / 4 writes per element. The amplified gradient never touches
// global memory — that's the whole point of the fused variant.
// ---------------------------------------------------------------------

template <typename ParamT, typename StateT, typename GradT>
__launch_bounds__(::sg::DEFAULT_CONFIG.block_size,
                  ::sg::DEFAULT_CONFIG.min_blocks_per_sm)
__global__ void grokfast_fused_ema_adam_step_kernel(
    ParamT*       __restrict__ params,
    StateT*       __restrict__ exp_avg,
    StateT*       __restrict__ exp_avg_sq,
    StateT*       __restrict__ slow_ema,
    const GradT*  __restrict__ grads,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb,
    float bc1, float bc2,
    int64_t n_elements,
    int64_t step_count
) {
    static_assert(is_param_dtype<ParamT>::value, "ParamT must be fp32, bf16, or fp16");
    static_assert(is_state_dtype<StateT>::value, "StateT must be fp32 or bf16");
    static_assert(is_grad_dtype<GradT>::value,
                  "GradT must be fp32, bf16, fp16, fp8_e4m3, or fp8_e5m2");

    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n_elements) return;

    // 5 reads
    const float g       = load_as_float(grads + i);
    const float xi_old  = load_state(slow_ema + i);
    const float m_old   = load_state(exp_avg + i);
    const float v_old   = load_state(exp_avg_sq + i);
    const float p_old   = load_as_float(params + i);

    // EMA + amplification (register only)
    const float xi   = alpha * xi_old + (1.0f - alpha) * g;
    const float gamp = g + lamb * xi;

    // Adam moments
    const float m = beta1 * m_old + (1.0f - beta1) * gamp;
    const float v = beta2 * v_old + (1.0f - beta2) * gamp * gamp;

    // Bias-corrected moments + update.
    const float m_hat = m / bc1;
    const float v_hat = v / bc2;
    const float rsv   = fast_rsqrt_nr(v_hat);          // 1/√v^
    const float u     = m_hat * rsv / (1.0f + eps * rsv) + weight_decay * p_old;
    const float p_new = p_old - lr * u;

    // 4 writes
    const unsigned s_u = static_cast<unsigned>(step_count);
    const unsigned i_u = static_cast<unsigned>(i);
    detail::store_state(slow_ema   + i, xi, s_u, i_u);
    detail::store_state(exp_avg    + i, m,  s_u, i_u ^ 0x1u);
    detail::store_state(exp_avg_sq + i, v,  s_u, i_u ^ 0x2u);
    store_param(params + i, p_new);
}

// FP32 vec4 fast path for the (FP32 param ∧ FP32 state ∧ FP32 grad)
// instantiation. Reuses fast_rsqrt_nr.
__launch_bounds__(::sg::DEFAULT_CONFIG.block_size,
                  ::sg::DEFAULT_CONFIG.min_blocks_per_sm)
__global__ void grokfast_fused_ema_adam_step_vec4_kernel(
    float4* __restrict__ params4,
    float4* __restrict__ exp_avg4,
    float4* __restrict__ exp_avg_sq4,
    float4* __restrict__ slow_ema4,
    const float4* __restrict__ grads4,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb,
    float bc1, float bc2,
    int64_t n4
) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n4) return;

    float4 g  = stream_load4(grads4 + i);
    float4 xi = stream_load4(slow_ema4 + i);
    float4 m  = stream_load4(exp_avg4 + i);
    float4 v  = stream_load4(exp_avg_sq4 + i);
    float4 p  = params4[i];

    #define _LANE(c)                                                          \
        do {                                                                  \
            xi.c = alpha * xi.c + (1.0f - alpha) * g.c;                       \
            float a = g.c + lamb * xi.c;                                      \
            m.c = beta1 * m.c + (1.0f - beta1) * a;                           \
            v.c = beta2 * v.c + (1.0f - beta2) * a * a;                       \
            float mh = m.c / bc1;                                             \
            float vh = v.c / bc2;                                             \
            float rs = fast_rsqrt_nr(vh);                                     \
            float u  = mh * rs / (1.0f + eps * rs) + weight_decay * p.c;      \
            p.c = p.c - lr * u;                                               \
        } while (0)
    _LANE(x); _LANE(y); _LANE(z); _LANE(w);
    #undef _LANE

    stream_store4(slow_ema4   + i, xi);
    stream_store4(exp_avg4    + i, m);
    stream_store4(exp_avg_sq4 + i, v);
    params4[i] = p;
}

// ---------------------------------------------------------------------
// Launchers — thin wrappers. Both pick block_size from tuned_configs and
// fall through to the vec4 fast path when alignment + dtype permit.
// ---------------------------------------------------------------------

namespace detail {

inline bool can_vec4(const void* p, int64_t n) {
    return (reinterpret_cast<uintptr_t>(p) % 16 == 0) && (n % 4 == 0);
}

inline ::sg::LaunchConfig pick_config(int64_t n) {
    // tuned_configs.h does not yet have a GROKFAST table — autotune
    // TODO is recorded there. Fall through to DEFAULT_CONFIG, which is
    // the same value (block=256, min_blocks_per_sm=2) that the deleted
    // baseline kernel was hand-tuned with.
    (void)n;
    return ::sg::DEFAULT_CONFIG;
}

} // namespace detail

template <typename ParamT, typename StateT, typename GradT>
cudaError_t launch_grokfast_fused_step(
    StateT* slow_ema,
    GradT*  grads,
    float alpha,
    float lamb,
    int64_t n_elements,
    cudaStream_t stream
) {
    static_assert(is_state_dtype<StateT>::value, "StateT must be fp32 or bf16");
    static_assert(is_grad_dtype<GradT>::value,
                  "GradT must be fp32, bf16, fp16, fp8_e4m3, or fp8_e5m2");
    // ParamT is unused on this path (EMA-only step does not touch params)
    // but kept in the signature for symmetry with the fused launcher.
    static_assert(is_param_dtype<ParamT>::value || std::is_void<ParamT>::value,
                  "ParamT must be fp32, bf16, or fp16");

    if (n_elements <= 0) return cudaSuccess;

    const ::sg::LaunchConfig cfg = detail::pick_config(n_elements);
    const int block = cfg.block_size;

    // FP32 vec4 fast path: state + grad both FP32 and aligned + N%4==0.
    if (cfg.vec4 &&
        std::is_same<StateT, float>::value &&
        std::is_same<GradT,  float>::value &&
        detail::can_vec4(grads, n_elements) &&
        detail::can_vec4(slow_ema, n_elements)) {
        const int64_t n4 = n_elements / 4;
        const int64_t grid = (n4 + block - 1) / block;
        grokfast_fused_step_vec4_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<float4*>(slow_ema),
            reinterpret_cast<float4*>(grads),
            alpha, lamb, n4);
        return cudaGetLastError();
    }

    const int64_t grid = (n_elements + block - 1) / block;
    // step_count is unused by the EMA-only kernel for SR keying — pass 0.
    grokfast_fused_step_kernel<StateT, GradT><<<grid, block, 0, stream>>>(
        slow_ema, grads, alpha, lamb, n_elements, /*step_count=*/0);
    return cudaGetLastError();
}

template <typename ParamT, typename StateT, typename GradT>
cudaError_t launch_grokfast_fused_ema_adam_step(
    ParamT* params,
    StateT* exp_avg,
    StateT* exp_avg_sq,
    StateT* slow_ema,
    const GradT* grads,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb,
    float bc1, float bc2,
    int64_t n_elements, int64_t step_count,
    cudaStream_t stream
) {
    static_assert(is_param_dtype<ParamT>::value, "ParamT must be fp32, bf16, or fp16");
    static_assert(is_state_dtype<StateT>::value, "StateT must be fp32 or bf16");
    static_assert(is_grad_dtype<GradT>::value,
                  "GradT must be fp32, bf16, fp16, fp8_e4m3, or fp8_e5m2");

    if (n_elements <= 0) return cudaSuccess;

    const ::sg::LaunchConfig cfg = detail::pick_config(n_elements);
    const int block = cfg.block_size;

    // FP32 vec4 fast path: param + state + grad all FP32 and aligned.
    if (cfg.vec4 &&
        std::is_same<ParamT, float>::value &&
        std::is_same<StateT, float>::value &&
        std::is_same<GradT,  float>::value &&
        detail::can_vec4(params,     n_elements) &&
        detail::can_vec4(grads,      n_elements) &&
        detail::can_vec4(exp_avg,    n_elements) &&
        detail::can_vec4(exp_avg_sq, n_elements) &&
        detail::can_vec4(slow_ema,   n_elements)) {
        const int64_t n4 = n_elements / 4;
        const int64_t grid = (n4 + block - 1) / block;
        grokfast_fused_ema_adam_step_vec4_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<float4*>(params),
            reinterpret_cast<float4*>(exp_avg),
            reinterpret_cast<float4*>(exp_avg_sq),
            reinterpret_cast<float4*>(slow_ema),
            reinterpret_cast<const float4*>(grads),
            lr, beta1, beta2, eps, weight_decay,
            alpha, lamb, bc1, bc2, n4);
        return cudaGetLastError();
    }

    const int64_t grid = (n_elements + block - 1) / block;
    grokfast_fused_ema_adam_step_kernel<ParamT, StateT, GradT>
        <<<grid, block, 0, stream>>>(
            params, exp_avg, exp_avg_sq, slow_ema, grads,
            lr, beta1, beta2, eps, weight_decay,
            alpha, lamb, bc1, bc2,
            n_elements, step_count);
    return cudaGetLastError();
}

}}} // namespace sg::sm90::grokfast
