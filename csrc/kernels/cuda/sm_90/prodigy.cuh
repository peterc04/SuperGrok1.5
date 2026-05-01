// =====================================================================
//  csrc/kernels/cuda/sm_90/prodigy.cuh
//
//  sm_90 (Hopper) Prodigy optimizer kernels + per-tensor launcher.
//
//  Prodigy is a distance-aware, self-tuning Adam variant. The adaptive
//  learning rate scalar d_t is updated each step from a global reduction
//  over (g, theta - theta0, s_state):
//
//     r_t = sqrt(beta2) * r_{t-1}
//         + (1 - sqrt(beta2)) * d_{t-1}^2 * g . (theta0 - theta_{t-1})
//     s_t = sqrt(beta2) * s_{t-1}
//         + (1 - sqrt(beta2)) * d_{t-1} * g
//     d_t = max(d_{t-1}, r_t / |s_t|_1)
//
//  The fused apply step (with the new d_t) is then:
//     m_t = beta1 * m_{t-1} + (1 - beta1) * d_t * g
//     v_t = beta2 * v_{t-1} + (1 - beta2) * d_t^2 * g^2
//     theta_t = theta_{t-1}
//             - lr * d_t * (m_t / (sqrt(v_t) + d_t * eps) + wd * theta)
//
//  This header exposes three sm_90 kernels and one templated launcher:
//
//     1) prodigy_dlr_reduce_kernel<ParamT,GradT,StateT,BLOCK>
//        Per-element multi-output reduction:
//           r_partial = sum d_{t-1}^2 * g[i] * (theta0[i] - theta[i])
//           s_partial = sum |d_{t-1} * (1-sqrt(b2)) * g[i]
//                          + sqrt(b2) * s_state[i]|
//        Two-channel warp_reduce_sum + SMEM tree, grid finish via
//        atomicAdd to *r_partial_out / *s_partial_out (initialized to 0
//        on the host side before launch).
//
//     2) prodigy_d_update_kernel
//        Single-thread scalar kernel that reads (r_partial, s_partial,
//        d_lr_inout) and writes d_lr_inout = max(d_{t-1}, r/|s|_1).
//        Keeps d_t fully on device so the apply kernel can dereference
//        it directly; avoids the per-step host CPU sync.
//
//     3) prodigy_apply_kernel<ParamT,StateT,GradT,BLOCK>
//        Fused per-element apply consuming the new d_t pointer. Updates
//        m, v, r_state, s_state, theta in a single grid-stride pass.
//
//  Heterogeneous dtype matrix (instantiations live in prodigy.cu):
//     ParamT in {float, __nv_bfloat16, __half}
//     StateT in {float, __nv_bfloat16}
//     GradT  in {float, __nv_bfloat16, __half,
//                __nv_fp8_e4m3, __nv_fp8_e5m2}
//  Incoherent combos rejected via static_assert (FP8 grad with FP32
//  param requires an explicit rescale not part of this kernel).
//
//  Forbidden by the build spec: no inline PTX over helpers, no runtime
//  dtype branches inside kernels, no hardcoded launch params (block
//  size pulled from tuned_configs.h::GROKADAMW_CONFIGS[ARCH_SM90]).
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/tuned_configs.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) ? (__CUDA_ARCH__ >= 890) : 1
  #include <cuda_fp8.h>
#endif

#include <cstdint>
#include <type_traits>

namespace sg { namespace sm90 { namespace prodigy {

// ---------------------------------------------------------------------
// Compile-time predicates for the dtype matrix (mirrors adamw.cuh).
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

template <typename ParamT, typename GradT>
struct is_coherent_combo
    : std::integral_constant<bool,
        !((std::is_same<GradT, __nv_fp8_e4m3>::value ||
           std::is_same<GradT, __nv_fp8_e5m2>::value) &&
          std::is_same<ParamT, float>::value)> {};

// ---------------------------------------------------------------------
// Type-erased load / store helpers (math is always FP32).
// ---------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p);

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
    return static_cast<float>(*p);
}
template <>
__device__ __forceinline__ float load_as_float<__nv_fp8_e5m2>(const __nv_fp8_e5m2* p) {
    return static_cast<float>(*p);
}

__device__ __forceinline__ float load_state(const float* p)         { return stream_load(p); }
__device__ __forceinline__ float load_state(const __nv_bfloat16* p) { return __bfloat162float(*p); }

template <typename T>
__device__ __forceinline__ void store_from_float(T* p, float v);

template <>
__device__ __forceinline__ void store_from_float<float>(float* p, float v) {
    *p = v;
}
template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ void store_from_float<__half>(__half* p, float v) {
    *p = __float2half_rn(v);
}

__device__ __forceinline__ void store_state(float* p, float v)         { stream_store(p, v); }
__device__ __forceinline__ void store_state(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

// =====================================================================
// Kernel 1: Multi-output reduction (r_partial, s_partial).
//
// Per element:
//     r_term[i] = d_prev^2 * g[i] * (theta0[i] - theta[i])
//     s_term[i] = | d_prev * (1 - sqrt(b2)) * g[i] + sqrt(b2) * s_state[i] |
//
// Two-channel warp_reduce_sum + SMEM tree, atomicAdd grid finish.
// d_prev is read from a single-element device pointer so the launcher
// can keep d_t fully on-device across iterations (no host sync).
//
// SMEM layout: [num_warps] floats r-channel + [num_warps] s-channel.
// Block size templated so __launch_bounds__ stays a compile-time const.
// =====================================================================

template <typename ParamT, typename GradT, typename StateT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void prodigy_dlr_reduce_kernel(
    const ParamT* __restrict__ params,
    const ParamT* __restrict__ params_init,
    const StateT* __restrict__ s_state,
    const GradT*  __restrict__ grads,
    const float*  __restrict__ d_prev_ptr,
    float*        __restrict__ r_partial_out,  // accumulator (init to 0)
    float*        __restrict__ s_partial_out,  // accumulator (init to 0)
    float beta2,
    int64_t n_elements
) {
    static_assert(is_param_dtype<ParamT>::value, "Prodigy: invalid ParamT");
    static_assert(is_state_dtype<StateT>::value, "Prodigy: invalid StateT");
    static_assert(is_grad_dtype<GradT>::value,   "Prodigy: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "Prodigy: FP8 grad with FP32 param requires explicit rescale");

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float smem_r[NUM_WARPS];
    __shared__ float smem_s[NUM_WARPS];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // d_prev is broadcast across the block (and the grid). One read per
    // block via shared memory keeps L1 traffic minimal.
    __shared__ float s_d_prev;
    __shared__ float s_sqrt_b2;
    __shared__ float s_one_minus_sqrt_b2;
    if (tid == 0) {
        s_d_prev = *d_prev_ptr;
        s_sqrt_b2 = sqrtf(beta2);
        s_one_minus_sqrt_b2 = 1.0f - s_sqrt_b2;
    }
    __syncthreads();
    const float d_prev   = s_d_prev;
    const float sqrt_b2  = s_sqrt_b2;
    const float omsqrtb2 = s_one_minus_sqrt_b2;
    const float d_prev_sq = d_prev * d_prev;

    float local_r = 0.0f;
    float local_s = 0.0f;

    const int64_t stride =
        static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         i < n_elements; i += stride) {
        const float g  = load_as_float(grads + i);
        const float p  = load_as_float(params + i);
        const float p0 = load_as_float(params_init + i);
        const float sv = load_state(s_state + i);

        local_r += d_prev_sq * g * (p0 - p);
        local_s += fabsf(d_prev * omsqrtb2 * g + sqrt_b2 * sv);
    }

    // Warp-level reduction (two channels in parallel).
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_r += SHFL_DOWN(local_r, offset);
        local_s += SHFL_DOWN(local_s, offset);
    }

    if (lane_id == 0) {
        smem_r[warp_id] = local_r;
        smem_s[warp_id] = local_s;
    }
    __syncthreads();

    // Cross-warp reduction (first warp only).
    if (warp_id == 0) {
        float wr = (lane_id < NUM_WARPS) ? smem_r[lane_id] : 0.0f;
        float ws = (lane_id < NUM_WARPS) ? smem_s[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            wr += SHFL_DOWN(wr, offset);
            ws += SHFL_DOWN(ws, offset);
        }
        if (lane_id == 0) {
            atomicAdd(r_partial_out, wr);
            atomicAdd(s_partial_out, ws);
        }
    }
}

// =====================================================================
// Kernel 1b: scalar update of d_t fully on device.
//   d_t = max(d_{t-1}, r_partial / max(s_partial, eps_floor))
// Avoids per-step host sync between the reduce and apply kernels.
// =====================================================================

__global__ void prodigy_d_update_kernel(
    const float* __restrict__ r_partial,
    const float* __restrict__ s_partial,
    float*       __restrict__ d_lr_inout
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const float r = *r_partial;
        const float s = *s_partial;
        const float d_prev = *d_lr_inout;
        // |s_t|_1 should be strictly positive after the first step; guard
        // against the trivial t=0 case where the buffer was zero-init'd.
        const float denom = fmaxf(s, 1e-30f);
        const float candidate = r / denom;
        *d_lr_inout = fmaxf(d_prev, candidate);
    }
}

// =====================================================================
// Kernel 2: Fused per-element apply with the new d_t.
//
// Per element (FP32 math; only loads/stores are typed):
//     g    = grad[i]
//     m_t  = b1 * m_{t-1} + (1-b1) * d_t * g
//     v_t  = b2 * v_{t-1} + (1-b2) * d_t^2 * g^2
//     r_t  = sqrt(b2) * r_{t-1}
//          + (1 - sqrt(b2)) * d_t^2 * g * (theta0[i] - theta_{t-1}[i])
//     s_t  = sqrt(b2) * s_{t-1}
//          + (1 - sqrt(b2)) * d_t * g
//     theta_t = theta_{t-1}
//             - lr * d_t * (m_t / (sqrt(v_t) + d_t * eps) + wd * theta)
//
// d_t is read from a single-element device pointer (filled by
// prodigy_d_update_kernel). Note: bc1 / bc2 are NOT applied here —
// Prodigy's convergence is driven by the d_t schedule.
// =====================================================================

template <typename ParamT, typename StateT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void prodigy_apply_kernel(
    ParamT*       __restrict__ params,
    const ParamT* __restrict__ params_init,
    StateT*       __restrict__ exp_avg,
    StateT*       __restrict__ exp_avg_sq,
    StateT*       __restrict__ r_state,
    StateT*       __restrict__ s_state,
    const GradT*  __restrict__ grads,
    const float*  __restrict__ d_lr_ptr,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int64_t n_elements
) {
    static_assert(is_param_dtype<ParamT>::value, "Prodigy: invalid ParamT");
    static_assert(is_state_dtype<StateT>::value, "Prodigy: invalid StateT");
    static_assert(is_grad_dtype<GradT>::value,   "Prodigy: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "Prodigy: FP8 grad with FP32 param requires explicit rescale");

    // Broadcast d_t and beta-derived scalars once per block.
    __shared__ float s_d;
    __shared__ float s_sqrt_b2;
    __shared__ float s_one_minus_sqrt_b2;
    if (threadIdx.x == 0) {
        s_d = *d_lr_ptr;
        s_sqrt_b2 = sqrtf(beta2);
        s_one_minus_sqrt_b2 = 1.0f - s_sqrt_b2;
    }
    __syncthreads();
    const float d_t      = s_d;
    const float sqrt_b2  = s_sqrt_b2;
    const float omsqrtb2 = s_one_minus_sqrt_b2;
    const float d_t_sq   = d_t * d_t;
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;

    const int64_t stride =
        static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float g  = load_as_float(grads        + i);
        const float p  = load_as_float(params       + i);
        const float p0 = load_as_float(params_init  + i);
        const float m0 = load_state(exp_avg    + i);
        const float v0 = load_state(exp_avg_sq + i);
        const float r0 = load_state(r_state    + i);
        const float s0 = load_state(s_state    + i);

        const float dg   = d_t * g;
        const float dgsq = d_t_sq * g * g;

        const float m1 = beta1 * m0 + one_minus_b1 * dg;
        const float v1 = beta2 * v0 + one_minus_b2 * dgsq;
        const float r1 = sqrt_b2 * r0 + omsqrtb2 * d_t_sq * g * (p0 - p);
        const float s1 = sqrt_b2 * s0 + omsqrtb2 * dg;

        // theta update with coupled-style scaling: lr_eff = lr * d_t.
        const float rsv = fast_rsqrt_nr(fmaxf(v1, 0.0f));
        // 1 / (sqrt(v_t) + d_t * eps) = rsv / (1 + d_t * eps * rsv).
        const float denom_inv = rsv / (1.0f + d_t * eps * rsv);
        const float u  = m1 * denom_inv + weight_decay * p;
        const float p1 = p - lr * d_t * u;

        store_state(exp_avg    + i, m1);
        store_state(exp_avg_sq + i, v1);
        store_state(r_state    + i, r1);
        store_state(s_state    + i, s1);
        store_from_float(params + i, p1);
    }
}

// =====================================================================
// Per-tensor templated launcher.
//
// Signature exactly as documented in the build spec. Orchestration:
//   (a) zero-init r_partial_out / s_partial_out via cudaMemsetAsync
//   (b) prodigy_dlr_reduce_kernel  -> r_partial / s_partial (device)
//   (c) prodigy_d_update_kernel    -> updates *d_lr_inout (device)
//   (d) prodigy_apply_kernel       -> moments / r / s / params
// d_t stays fully on-device — no per-step host CPU sync.
//
// Block size is pulled from tuned_configs.h::GROKADAMW_CONFIGS[ARCH_SM90]
// (Prodigy shares the same BW-bound profile until autotune adds a row).
// =====================================================================

template <typename ParamT, typename StateT, typename GradT>
cudaError_t launch_prodigy_fused_step(
    ParamT* params, ParamT* params_init,
    StateT* exp_avg, StateT* exp_avg_sq,
    StateT* r_state, StateT* s_state,
    const GradT* grads,
    float* d_lr_inout,
    float* r_partial_out, float* s_partial_out,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int64_t n_elements, int64_t step_count,
    cudaStream_t stream
) {
    static_assert(is_param_dtype<ParamT>::value, "Prodigy: invalid ParamT");
    static_assert(is_state_dtype<StateT>::value, "Prodigy: invalid StateT");
    static_assert(is_grad_dtype<GradT>::value,   "Prodigy: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "Prodigy: FP8 grad with FP32 param requires an explicit rescale");

    if (n_elements <= 0) return cudaSuccess;

    // Block size pulled from the autotuned table; do NOT hardcode.
    const LaunchConfig cfg =
        get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;

    constexpr int MAX_BLOCKS = 8192;
    const int64_t needed = (n_elements + block - 1) / block;
    const int grid = static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);

    // (a) Zero the partial accumulators before the reduce kernel atomics.
    cudaError_t err = cudaMemsetAsync(r_partial_out, 0, sizeof(float), stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(s_partial_out, 0, sizeof(float), stream);
    if (err != cudaSuccess) return err;

    // (b) Multi-output reduce: r_partial / s_partial -> device scalars.
    auto launch_reduce = [&](auto block_constant) {
        constexpr int BS = decltype(block_constant)::value;
        prodigy_dlr_reduce_kernel<ParamT, GradT, StateT, BS>
            <<<grid, BS, 0, stream>>>(
                params, params_init, s_state, grads,
                d_lr_inout, r_partial_out, s_partial_out,
                beta2, n_elements);
    };
    if (block == 128) {
        launch_reduce(std::integral_constant<int, 128>{});
    } else if (block == 512) {
        launch_reduce(std::integral_constant<int, 512>{});
    } else {
        launch_reduce(std::integral_constant<int, 256>{});
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // (c) Device-side scalar update of d_t — preferred over host sync.
    prodigy_d_update_kernel<<<1, 1, 0, stream>>>(
        r_partial_out, s_partial_out, d_lr_inout);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // (d) Fused apply with the freshly-updated d_t.
    auto launch_apply = [&](auto block_constant) {
        constexpr int BS = decltype(block_constant)::value;
        prodigy_apply_kernel<ParamT, StateT, GradT, BS>
            <<<grid, BS, 0, stream>>>(
                params, params_init,
                exp_avg, exp_avg_sq, r_state, s_state, grads,
                d_lr_inout,
                lr, beta1, beta2, eps, weight_decay,
                n_elements);
    };
    if (block == 128) {
        launch_apply(std::integral_constant<int, 128>{});
    } else if (block == 512) {
        launch_apply(std::integral_constant<int, 512>{});
    } else {
        launch_apply(std::integral_constant<int, 256>{});
    }
    (void) step_count;  // reserved for future SR / PRNG seeding.
    return cudaGetLastError();
}

}}} // namespace sg::sm90::prodigy
