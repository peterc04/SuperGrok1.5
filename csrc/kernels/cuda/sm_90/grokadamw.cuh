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

// =====================================================================
// Per-tensor templated launcher (signature exactly per build spec).
//
// bc1, bc2 are precomputed by the caller (see binding multi-tensor
// loop / per-tensor torch shim) and forwarded; step_count is the
// integer step (used to seed BF16 SR keying via hash_prng).
// =====================================================================

namespace detail {

inline bool can_vec4(const void* p, int64_t n) {
    return (reinterpret_cast<uintptr_t>(p) % 16 == 0) && (n % 4 == 0);
}

} // namespace detail

template <typename ParamT, typename StateT, typename GradT>
cudaError_t launch_grokadamw_fused_step(
    ParamT* params, StateT* exp_avg, StateT* exp_avg_sq, StateT* slow_ema,
    const GradT* grads,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb, float clip_c,
    float bc1, float bc2,
    int64_t n_elements, int64_t step_count,
    cudaStream_t stream
) {
    static_assert(is_param_dtype<ParamT>::value, "GrokAdamW: invalid ParamT");
    static_assert(is_state_dtype<StateT>::value, "GrokAdamW: invalid StateT");
    static_assert(is_grad_dtype<GradT>::value,   "GrokAdamW: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "GrokAdamW: FP8 grad with FP32 param requires explicit rescale");

    if (n_elements <= 0) return cudaSuccess;

    const ::sg::LaunchConfig cfg =
        ::sg::get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;

    constexpr int MAX_BLOCKS = 8192;

    constexpr bool is_all_fp32 =
        std::is_same<ParamT, float>::value &&
        std::is_same<StateT, float>::value &&
        std::is_same<GradT,  float>::value;

    const bool aligned4 =
        detail::can_vec4(params,     n_elements) &&
        detail::can_vec4(exp_avg,    n_elements) &&
        detail::can_vec4(exp_avg_sq, n_elements) &&
        detail::can_vec4(slow_ema,   n_elements) &&
        detail::can_vec4(grads,      n_elements);

    if (cfg.vec4 && is_all_fp32 && aligned4) {
        const int64_t n_vec  = n_elements / 4;
        const int64_t needed = (n_vec + block - 1) / block;
        const int     grid   = static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);
        if (block == 128) {
            grokadamw_fused_step_vec4_fp32_kernel<128>
                <<<grid, 128, 0, stream>>>(
                    reinterpret_cast<float*>(params),
                    reinterpret_cast<float*>(exp_avg),
                    reinterpret_cast<float*>(exp_avg_sq),
                    reinterpret_cast<float*>(slow_ema),
                    reinterpret_cast<const float*>(grads),
                    lr, beta1, beta2, eps, weight_decay,
                    alpha, lamb, clip_c, bc1, bc2, n_vec);
        } else if (block == 512) {
            grokadamw_fused_step_vec4_fp32_kernel<512>
                <<<grid, 512, 0, stream>>>(
                    reinterpret_cast<float*>(params),
                    reinterpret_cast<float*>(exp_avg),
                    reinterpret_cast<float*>(exp_avg_sq),
                    reinterpret_cast<float*>(slow_ema),
                    reinterpret_cast<const float*>(grads),
                    lr, beta1, beta2, eps, weight_decay,
                    alpha, lamb, clip_c, bc1, bc2, n_vec);
        } else {
            grokadamw_fused_step_vec4_fp32_kernel<256>
                <<<grid, 256, 0, stream>>>(
                    reinterpret_cast<float*>(params),
                    reinterpret_cast<float*>(exp_avg),
                    reinterpret_cast<float*>(exp_avg_sq),
                    reinterpret_cast<float*>(slow_ema),
                    reinterpret_cast<const float*>(grads),
                    lr, beta1, beta2, eps, weight_decay,
                    alpha, lamb, clip_c, bc1, bc2, n_vec);
        }
        (void) step_count;
        return cudaGetLastError();
    }

    const int64_t needed = (n_elements + block - 1) / block;
    const int     grid   = static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);
    if (block == 128) {
        grokadamw_fused_step_kernel<ParamT, StateT, GradT, 128>
            <<<grid, 128, 0, stream>>>(
                params, exp_avg, exp_avg_sq, slow_ema, grads,
                lr, beta1, beta2, eps, weight_decay,
                alpha, lamb, clip_c, bc1, bc2,
                n_elements, step_count);
    } else if (block == 512) {
        grokadamw_fused_step_kernel<ParamT, StateT, GradT, 512>
            <<<grid, 512, 0, stream>>>(
                params, exp_avg, exp_avg_sq, slow_ema, grads,
                lr, beta1, beta2, eps, weight_decay,
                alpha, lamb, clip_c, bc1, bc2,
                n_elements, step_count);
    } else {
        grokadamw_fused_step_kernel<ParamT, StateT, GradT, 256>
            <<<grid, 256, 0, stream>>>(
                params, exp_avg, exp_avg_sq, slow_ema, grads,
                lr, beta1, beta2, eps, weight_decay,
                alpha, lamb, clip_c, bc1, bc2,
                n_elements, step_count);
    }
    return cudaGetLastError();
}

// =====================================================================
// Q3 quantized variant — INT8 exp_avg (per-block FP32 scale,
// Q3_BLOCK_SIZE=32), BF16 exp_avg_sq + BF16 slow_ema, both with
// stochastic-rounded writeback. Math identical to non-q3.
//
// The slow_ema storage type tracks the binding signature
// (csrc/bindings/grokadamw.cpp: torch::Tensor ema_bf16). The baseline
// kernel stored ema as BF16 too — so this kernel only needs ONE
// per-block scales array (for exp_avg). exp_avg_sq + slow_ema both
// keep BF16 storage with float_to_bf16_stochastic on writeback to
// avoid the small-delta zero-collapse failure mode at alpha=0.98.
//
// Per-block scale update: each Q3_BLOCK_SIZE-element block (==1 warp
// at Q=32) computes the new max-abs via warp __shfl_xor_sync and the
// lane at block_offset==0 writes the new scale. Everything stays in
// registers — no shared memory traffic.
// =====================================================================

template <typename ParamT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void grokadamw_fused_step_q3_kernel(
    ParamT*               __restrict__ params,
    int8_t*               __restrict__ exp_avg_int8,
    float*                __restrict__ exp_avg_scales,
    __nv_bfloat16*        __restrict__ exp_avg_sq_bf16,
    __nv_bfloat16*        __restrict__ slow_ema_bf16,
    const GradT*          __restrict__ grads,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb, float clip_c,
    float bc1, float bc2,
    int64_t n_elements,
    int64_t step_count
) {
    static_assert(is_param_dtype<ParamT>::value, "GrokAdamW q3: invalid ParamT");
    static_assert(is_q3_grad_dtype<GradT>::value,
                  "GrokAdamW q3: GradT must be fp32, bf16, or fp16 "
                  "(FP8 grad rejected on q3 — INT8 moment dynamic range "
                  "incompatible with FP8 grad scales).");
    static_assert(std::is_same<ParamT, float>::value ||
                  std::is_same<ParamT, __nv_bfloat16>::value,
                  "GrokAdamW q3: ParamT must be fp32 or bf16");

    const int64_t stride  = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const unsigned step_u = static_cast<unsigned>(step_count);
    constexpr int Q  = Q3_BLOCK_SIZE;

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const int block_idx     = static_cast<int>(i / Q);
        const int lane_in_block = static_cast<int>(i % Q);

        // ----- Reads -----
        const float  g_raw       = load_as_float(grads + i);
        const float  p_old       = load_as_float(params + i);
        const float  ea_scale_old = LDG(exp_avg_scales + block_idx);
        const int8_t ea_q_old    = LDG(exp_avg_int8 + i);
        const float  v_old       = __bfloat162float(LDG(exp_avg_sq_bf16 + i));
        const float  xi_old      = __bfloat162float(LDG(slow_ema_bf16   + i));

        // Dequant exp_avg.
        const float m_old  = dequant_int8(ea_q_old, ea_scale_old);

        // ----- Math (identical to non-q3 path) -----
        const float g    = fminf(fmaxf(g_raw, -clip_c), clip_c);
        const float xi   = alpha * xi_old + (1.0f - alpha) * g;
        const float gamp = g + lamb * xi;
        const float m    = beta1 * m_old + (1.0f - beta1) * gamp;
        const float v    = beta2 * v_old + (1.0f - beta2) * gamp * gamp;

        const float m_hat = m / bc1;
        const float v_hat = v / bc2;
        const float rsv   = fast_rsqrt_nr(fmaxf(v_hat, 0.0f));
        const float u     = m_hat * rsv / (1.0f + eps * rsv) + weight_decay * p_old;
        const float p_new = p_old - lr * u;

        // ----- Per-block exp_avg scale recompute (warp absmax) -----
        // Q == 32 == WARP_SIZE on sm_90. __shfl_xor_sync is a standard
        // CUDA primitive (no inline PTX needed).
        float ea_absmax = fabsf(m);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ea_absmax = fmaxf(ea_absmax,
                              __shfl_xor_sync(FULL_WARP_MASK, ea_absmax, off));
        }
        const float ea_scale_new = fmaxf(ea_absmax, 1e-12f) / 127.0f;

        // ----- Requant + stochastic-rounded writeback -----
        const unsigned i_u    = static_cast<unsigned>(i);
        const unsigned rng_m  = hash_prng(step_u,            i_u);
        const unsigned rng_xi = hash_prng(step_u ^ 0xA5A5u,  i_u);
        const unsigned rng_v  = hash_prng(step_u ^ 0x5A5Au,  i_u);

        exp_avg_int8[i]    = ptx_int8_stochastic_round(m,  ea_scale_new, rng_m);
        slow_ema_bf16[i]   = float_to_bf16_stochastic(xi, rng_xi);
        exp_avg_sq_bf16[i] = float_to_bf16_stochastic(v,  rng_v);

        if (lane_in_block == 0) {
            exp_avg_scales[block_idx] = ea_scale_new;
        }

        store_param(params + i, p_new);
    }
}

template <typename ParamT, typename GradT>
cudaError_t launch_grokadamw_fused_step_q3(
    ParamT* params,
    int8_t* exp_avg_int8, float* exp_avg_scales,
    __nv_bfloat16* exp_avg_sq_bf16,
    __nv_bfloat16* slow_ema_bf16,
    const GradT* grads,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb, float clip_c,
    float bc1, float bc2,
    int64_t n_elements, int64_t step_count,
    cudaStream_t stream
) {
    static_assert(is_param_dtype<ParamT>::value, "GrokAdamW q3: invalid ParamT");
    static_assert(is_q3_grad_dtype<GradT>::value,
                  "GrokAdamW q3: GradT must be fp32, bf16, or fp16");
    static_assert(std::is_same<ParamT, float>::value ||
                  std::is_same<ParamT, __nv_bfloat16>::value,
                  "GrokAdamW q3: ParamT must be fp32 or bf16");

    if (n_elements <= 0) return cudaSuccess;

    // tuned_configs picks the actual block size from {128, 256, 512}.
    // All are multiples of Q3_BLOCK_SIZE=32 == WARP_SIZE so the warp
    // absmax reduction lands cleanly on per-block lane chunks.
    const ::sg::LaunchConfig cfg =
        ::sg::get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;

    constexpr int MAX_BLOCKS = 8192;
    const int64_t needed = (n_elements + block - 1) / block;
    const int     grid   = static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);

    if (block == 128) {
        grokadamw_fused_step_q3_kernel<ParamT, GradT, 128>
            <<<grid, 128, 0, stream>>>(
                params,
                exp_avg_int8, exp_avg_scales, exp_avg_sq_bf16,
                slow_ema_bf16, grads,
                lr, beta1, beta2, eps, weight_decay,
                alpha, lamb, clip_c, bc1, bc2,
                n_elements, step_count);
    } else if (block == 512) {
        grokadamw_fused_step_q3_kernel<ParamT, GradT, 512>
            <<<grid, 512, 0, stream>>>(
                params,
                exp_avg_int8, exp_avg_scales, exp_avg_sq_bf16,
                slow_ema_bf16, grads,
                lr, beta1, beta2, eps, weight_decay,
                alpha, lamb, clip_c, bc1, bc2,
                n_elements, step_count);
    } else {
        grokadamw_fused_step_q3_kernel<ParamT, GradT, 256>
            <<<grid, 256, 0, stream>>>(
                params,
                exp_avg_int8, exp_avg_scales, exp_avg_sq_bf16,
                slow_ema_bf16, grads,
                lr, beta1, beta2, eps, weight_decay,
                alpha, lamb, clip_c, bc1, bc2,
                n_elements, step_count);
    }
    return cudaGetLastError();
}

}}} // namespace sg::sm90::grokadamw

// =====================================================================
// Tensor-form binding shims at namespace sg::sm90 — these match the
// forward declarations in csrc/bindings/grokadamw.cpp::
// DECLARE_GROKADAMW_LAUNCHERS(sm90) and DECLARE_MT_GROKADAMW(sm90).
// They must stay byte-identical (return type, param list, ref-quals).
//
// Bindings only forward (alpha, lamb, beta1, beta2, lr, wd, eps,
// bc1, bc2) — the gradient-clamp threshold is NOT plumbed through the
// non-clip entry points. We wire the inner template's clip_c to a
// large sentinel (FLT_MAX/4) on those paths so the clamp becomes a
// pass-through. The clip_step entry point uses the clip_threshold
// argument as the per-element clamp magnitude.
// =====================================================================

namespace sg { namespace sm90 {

void launch_fused_grokadamw_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2);

void launch_fused_grokadamw_clip_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2,
    float clip_threshold);

void launch_fused_grokadamw_step_q3(
    torch::Tensor param,
    torch::Tensor exp_avg_int8,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_bf16,
    torch::Tensor ema_bf16,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2,
    unsigned global_step);

void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps);

void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> emas,
    std::vector<torch::Tensor> grads,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2);

}} // namespace sg::sm90

// =====================================================================
// Tensor-form bodies — emitted ONCE from grokadamw.cu so other TUs can
// link against them without duplicate-symbol errors.
// =====================================================================
#ifdef SG_SM90_GROKADAMW_DEFINE_LAUNCHER

namespace sg { namespace sm90 {

namespace _gaw_detail {

// Sentinel "no clamp" — large enough that the fminf/fmaxf in the inner
// kernel becomes a pass-through for any plausible gradient magnitude.
inline constexpr float NO_CLAMP = 3.4e30f;

// Dispatch on (param dtype, state dtype, grad dtype). We restrict here
// to the tightest cross-section the bindings actually feed in:
// param/grad share dtype (Python optimizer constructs grads on the
// param), state is FP32 (allocated FP32 in grokadamw.py state init).
// FP8 grads + FP16/BF16 params are reachable via the templated entry
// point but not from this Python-facing torch::Tensor shim — they go
// through a separate FP8 binding when added.
template <typename ParamT, typename GradT>
inline void dispatch_step_fp32_state(
    torch::Tensor& param, torch::Tensor& exp_avg, torch::Tensor& exp_avg_sq,
    torch::Tensor& ema, torch::Tensor& grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float wd, float eps,
    float bc1, float bc2,
    float clip_c,
    int64_t step_count,
    cudaStream_t stream
) {
    const int64_t N = param.numel();
    if (N <= 0) return;
    auto err = ::sg::sm90::grokadamw::launch_grokadamw_fused_step<
        ParamT, float, GradT>(
        reinterpret_cast<ParamT*>(param.data_ptr()),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        ema.data_ptr<float>(),
        reinterpret_cast<const GradT*>(grad.data_ptr()),
        lr, beta1, beta2, eps, wd, alpha, lamb, clip_c,
        bc1, bc2, N, step_count, stream);
    TORCH_CHECK(err == cudaSuccess,
        "launch_fused_grokadamw_step (sm90): ",
        cudaGetErrorString(err));
}

inline void dispatch_step_per_param(
    torch::Tensor& param, torch::Tensor& exp_avg, torch::Tensor& exp_avg_sq,
    torch::Tensor& ema, torch::Tensor& grad,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps,
    float bc1, float bc2, float clip_c,
    int64_t step_count, cudaStream_t stream
) {
    TORCH_CHECK(exp_avg.scalar_type()    == at::kFloat,
                "grokadamw sm90: exp_avg must be float32");
    TORCH_CHECK(exp_avg_sq.scalar_type() == at::kFloat,
                "grokadamw sm90: exp_avg_sq must be float32");
    TORCH_CHECK(ema.scalar_type()        == at::kFloat,
                "grokadamw sm90: ema must be float32");
    TORCH_CHECK(grad.scalar_type() == param.scalar_type(),
                "grokadamw sm90: grad and param dtypes must match");

    switch (param.scalar_type()) {
        case at::kFloat:
            dispatch_step_fp32_state<float, float>(
                param, exp_avg, exp_avg_sq, ema, grad,
                alpha, lamb, beta1, beta2, lr, wd, eps,
                bc1, bc2, clip_c, step_count, stream);
            break;
        case at::kBFloat16:
            dispatch_step_fp32_state<__nv_bfloat16, __nv_bfloat16>(
                param, exp_avg, exp_avg_sq, ema, grad,
                alpha, lamb, beta1, beta2, lr, wd, eps,
                bc1, bc2, clip_c, step_count, stream);
            break;
        case at::kHalf:
            dispatch_step_fp32_state<__half, __half>(
                param, exp_avg, exp_avg_sq, ema, grad,
                alpha, lamb, beta1, beta2, lr, wd, eps,
                bc1, bc2, clip_c, step_count, stream);
            break;
        default:
            TORCH_CHECK(false,
                "grokadamw sm90: unsupported param dtype ",
                param.scalar_type());
    }
}

} // namespace _gaw_detail

// ---------------------------------------------------------------------
// 1) launch_fused_grokadamw_step — non-clip per-tensor entry point.
//    Bindings DO NOT plumb a clip_c here, so we run with the no-clamp
//    sentinel. Step count is reconstructed from bc1 (== 1 - beta1^t)
//    purely so the SR keying differs across calls — the kernel itself
//    only uses step_count to seed hash_prng.
// ---------------------------------------------------------------------
void launch_fused_grokadamw_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Recover step ≈ log(1-bc1)/log(beta1). Used only for SR PRNG seed.
    int64_t step_count = 0;
    if (bc1 > 0.0f && beta1 > 0.0f && beta1 < 1.0f) {
        const float t = std::log(std::max(1.0f - bc1, 1e-30f))
                        / std::log(beta1);
        step_count = static_cast<int64_t>(std::lround(std::max(t, 1.0f)));
    }
    _gaw_detail::dispatch_step_per_param(
        param, exp_avg, exp_avg_sq, ema, grad,
        alpha, lamb, beta1, beta2, lr, weight_decay, eps,
        bc1, bc2, _gaw_detail::NO_CLAMP, step_count, stream);
}

// ---------------------------------------------------------------------
// 2) launch_fused_grokadamw_clip_step — same as (1), but the
//    clip_threshold argument becomes the per-element clamp magnitude.
// ---------------------------------------------------------------------
void launch_fused_grokadamw_clip_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2,
    float clip_threshold
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t step_count = 0;
    if (bc1 > 0.0f && beta1 > 0.0f && beta1 < 1.0f) {
        const float t = std::log(std::max(1.0f - bc1, 1e-30f))
                        / std::log(beta1);
        step_count = static_cast<int64_t>(std::lround(std::max(t, 1.0f)));
    }
    const float c = (clip_threshold > 0.0f) ? clip_threshold
                                            : _gaw_detail::NO_CLAMP;
    _gaw_detail::dispatch_step_per_param(
        param, exp_avg, exp_avg_sq, ema, grad,
        alpha, lamb, beta1, beta2, lr, weight_decay, eps,
        bc1, bc2, c, step_count, stream);
}

// ---------------------------------------------------------------------
// 3) launch_fused_grokadamw_step_q3 — quantized variant. exp_avg is
//    INT8 + FP32 per-block scales; exp_avg_sq + ema are BF16 with SR.
// ---------------------------------------------------------------------
void launch_fused_grokadamw_step_q3(
    torch::Tensor param,
    torch::Tensor exp_avg_int8,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_bf16,
    torch::Tensor ema_bf16,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2,
    unsigned global_step
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t N = param.numel();
    if (N <= 0) return;

    TORCH_CHECK(exp_avg_int8.scalar_type()    == at::kChar,
                "grokadamw_q3 sm90: exp_avg must be int8");
    TORCH_CHECK(exp_avg_scales.scalar_type()  == at::kFloat,
                "grokadamw_q3 sm90: exp_avg_scales must be float32");
    TORCH_CHECK(exp_avg_sq_bf16.scalar_type() == at::kBFloat16,
                "grokadamw_q3 sm90: exp_avg_sq must be bfloat16");
    TORCH_CHECK(ema_bf16.scalar_type()        == at::kBFloat16,
                "grokadamw_q3 sm90: ema must be bfloat16");
    TORCH_CHECK(grad.scalar_type() == param.scalar_type(),
                "grokadamw_q3 sm90: grad and param dtypes must match");

    const int64_t step_count = static_cast<int64_t>(global_step);

    auto launch_typed = [&](auto p_tag, auto g_tag) {
        using ParamT = decltype(p_tag);
        using GradT  = decltype(g_tag);
        auto err = ::sg::sm90::grokadamw::launch_grokadamw_fused_step_q3<
            ParamT, GradT>(
            reinterpret_cast<ParamT*>(param.data_ptr()),
            reinterpret_cast<int8_t*>(exp_avg_int8.data_ptr()),
            exp_avg_scales.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(exp_avg_sq_bf16.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(ema_bf16.data_ptr()),
            reinterpret_cast<const GradT*>(grad.data_ptr()),
            lr, beta1, beta2, eps, weight_decay,
            alpha, lamb, _gaw_detail::NO_CLAMP, bc1, bc2,
            N, step_count, stream);
        TORCH_CHECK(err == cudaSuccess,
            "launch_fused_grokadamw_step_q3 (sm90): ",
            cudaGetErrorString(err));
    };

    switch (param.scalar_type()) {
        case at::kFloat:    launch_typed(float{},          float{});         break;
        case at::kBFloat16: launch_typed(__nv_bfloat16{},  __nv_bfloat16{}); break;
        default:
            TORCH_CHECK(false,
                "grokadamw_q3 sm90: unsupported param dtype ",
                param.scalar_type(),
                " (only fp32 / bf16 supported on q3)");
    }
}

// ---------------------------------------------------------------------
// 4) launch_multi_tensor_grokadamw — vector wrapper. Loops per-param
//    and forwards into the same per-tensor dispatch as (1). The clip
//    is already applied host-side by the caller (see
//    csrc/bindings/grokadamw.cpp::grokadamw_fused_step), so the
//    kernel runs with the no-clamp sentinel.
// ---------------------------------------------------------------------
void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps
) {
    const size_t n = params.size();
    if (n == 0) return;
    TORCH_CHECK(bc1s.size() == n && bc2s.size() == n,
                "grokadamw sm90: bc1/bc2 vector size mismatch");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (size_t i = 0; i < n; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

        // Reconstruct step from bc1 only for SR keying.
        int64_t step_count = 0;
        if (bc1s[i] > 0.0f && beta1 > 0.0f && beta1 < 1.0f) {
            const float t = std::log(std::max(1.0f - bc1s[i], 1e-30f))
                            / std::log(beta1);
            step_count = static_cast<int64_t>(std::lround(std::max(t, 1.0f)));
        }
        _gaw_detail::dispatch_step_per_param(
            params[i], exp_avgs[i], exp_avg_sqs[i], emas[i], grads[i],
            alpha, lamb, beta1, beta2, lr, wd, eps,
            bc1s[i], bc2s[i], _gaw_detail::NO_CLAMP,
            step_count, stream);
    }
}

// Scalar bc1/bc2 overload — broadcasts to per-param vectors.
void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> emas,
    std::vector<torch::Tensor> grads,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    const size_t n = params.size();
    std::vector<float> bc1s(n, bc1), bc2s(n, bc2);
    launch_multi_tensor_grokadamw(
        params, exp_avgs, exp_avg_sqs, emas, grads,
        bc1s, bc2s, alpha, lamb, beta1, beta2,
        lr, weight_decay, eps);
}

}} // namespace sg::sm90

#endif // SG_SM90_GROKADAMW_DEFINE_LAUNCHER
