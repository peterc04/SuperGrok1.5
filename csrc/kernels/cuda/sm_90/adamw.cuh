// =====================================================================
//  csrc/kernels/cuda/sm_90/adamw.cuh
//
//  sm_90 (Hopper) AdamW optimizer kernel + launcher header.
//
//  Algorithm (single fused elementwise pass, decoupled weight decay):
//     m_t   = beta1 * m_{t-1} + (1 - beta1) * g
//     v_t   = beta2 * v_{t-1} + (1 - beta2) * g^2
//     m_hat = m_t / bc1                   // bc1 = 1 - beta1^t (host-passed)
//     v_hat = v_t / bc2                   // bc2 = 1 - beta2^t (host-passed)
//     u     = m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1}
//     theta = theta_{t-1} - lr * u
//
//  Two launchers (namespace sg::sm90::adamw):
//     1) launch_fused_adamw_simple_step<ParamT,StateT,GradT>(...)
//        per-tensor, signature exactly as documented in the build spec.
//
//  And one vector wrapper at namespace sg::sm90 (matches the forward
//  declaration in csrc/bindings/grokadamw.cpp::DECLARE_MT_GROKADAMW(sm90)
//  for SG_DISPATCH(launch_fused_adamw_simple, ...)):
//     2) launch_fused_adamw_simple(std::vector<torch::Tensor>& params,
//                                  std::vector<torch::Tensor>& exp_avgs,
//                                  std::vector<torch::Tensor>& exp_avg_sqs,
//                                  std::vector<torch::Tensor>& grads,
//                                  std::vector<int64_t>& steps,
//                                  float beta1, float beta2,
//                                  float lr, float wd, float eps);
//
//  Roofline (FP32 P/S/G, per element):
//     R = 4 * 4 B  (param, exp_avg, exp_avg_sq, grad)
//     W = 3 * 4 B  (param, exp_avg, exp_avg_sq)
//     I ~= 9 FLOPs (3 FMAs for moments, rsqrt + NR, 2 FMAs for update)
//  AI ~ 9 / 28 ~ 0.32 FLOP/byte — solidly bandwidth bound on H100
//  (3 TB/s HBM3, ~67 TFLOP/s FP32). Tuning targets peak HBM throughput:
//     - ld.global.nc.* via LDG / stream_load for read-only inputs
//     - st.global.wt.f32 via stream_store on FP32 state to bypass L2
//     - vec4 fast path (float4) for FP32 P/S/G when all aligned and
//       N is a multiple of 4 (selectable via tuned_configs.h::vec4)
//     - block_size pulled from GROKADAMW_CONFIGS[ARCH_SM90][bucket]
//
//  Heterogeneous dtype matrix (instantiations live in adamw.cu):
//     ParamT in {float, __nv_bfloat16, __half}
//     StateT in {float, __nv_bfloat16}
//     GradT  in {float, __nv_bfloat16, __half,
//                __nv_fp8_e4m3, __nv_fp8_e5m2}
//  Incoherent combos (e.g. FP8 grad + FP32 param without an explicit
//  rescale, FP8 storage on param/state) are rejected via static_assert.
//  All math runs in FP32 — only loads/stores are typed.
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

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace sg { namespace sm90 { namespace adamw {

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

// FP8 grads are only coherent when the parameter is also reduced precision
// (BF16/FP16) — full FP32 params with FP8 grads silently lose dynamic range
// without an explicit rescale, which is not part of this kernel.
template <typename ParamT, typename GradT>
struct is_coherent_combo
    : std::integral_constant<bool,
        !((std::is_same<GradT, __nv_fp8_e4m3>::value ||
           std::is_same<GradT, __nv_fp8_e5m2>::value) &&
          std::is_same<ParamT, float>::value)> {};

}}} // namespace sg::sm90::adamw

// =====================================================================
// Type-erased load / store helpers — math is always FP32.
// =====================================================================

namespace sg { namespace sm90 { namespace adamw {

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p);

template <>
__device__ __forceinline__ float load_as_float<float>(const float* p) {
    // ld.global.nc.f32 — read-only path, bypasses L1 contention with state.
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
    // FP8 lacks an LDG overload; rely on compiler to emit ld.global.nc.b8.
    return static_cast<float>(*p);
}
template <>
__device__ __forceinline__ float load_as_float<__nv_fp8_e5m2>(const __nv_fp8_e5m2* p) {
    return static_cast<float>(*p);
}

// Streaming-load FP32 state (param / exp_avg / exp_avg_sq are read AND
// written, so the wt path on the store side keeps L2 unallocated).
__device__ __forceinline__ float load_state(const float* p)         { return stream_load(p); }
__device__ __forceinline__ float load_state(const __nv_bfloat16* p) { return __bfloat162float(*p); }

// Generic typed store from FP32. ParamT FP16/BF16 use rn-rounded paths;
// state has a streaming-store fast path for FP32.
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

// State stream-store: FP32 uses st.global.wt.f32 (no L2 alloc); BF16
// state has no PTX wt v1.b16 — falls through to a normal rn store.
__device__ __forceinline__ void store_state(float* p, float v)         { stream_store(p, v); }
__device__ __forceinline__ void store_state(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

// Optional BF16 stochastic round entry point (kept distinct from the
// default rn path so callers can opt in for ParamT = BF16). Uses
// utils.cuh::float_to_bf16_stochastic + a hash PRNG seeded by step.
__device__ __forceinline__ void store_param_bf16_sr(
    __nv_bfloat16* p, float v, unsigned step, unsigned idx
) {
    *p = float_to_bf16_stochastic(v, hash_prng(step, idx));
}

// =====================================================================
// Per-element fused step (FP32 math, decoupled WD).
// Inlined here rather than calling adam_sm90.cuh because that header's
// templates are TODO stubs as of this commit.
// =====================================================================

template <typename ParamT, typename StateT, typename GradT>
__device__ __forceinline__ void adamw_step_inline(
    ParamT*       __restrict__ param_ptr,
    StateT*       __restrict__ exp_avg_ptr,
    StateT*       __restrict__ exp_avg_sq_ptr,
    const GradT*  __restrict__ grad_ptr,
    int64_t idx,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2,
    unsigned step_for_sr   // only used by the BF16-param SR overload
) {
    const float g  = load_as_float(grad_ptr  + idx);
    const float m0 = load_state(exp_avg_ptr  + idx);
    const float v0 = load_state(exp_avg_sq_ptr + idx);
    const float p0 = load_as_float(param_ptr + idx);

    // Moment updates (FMA-fusable).
    const float m1 = beta1 * m0 + (1.0f - beta1) * g;
    const float v1 = beta2 * v0 + (1.0f - beta2) * g * g;

    // Bias correction is a host-side division by bc1, bc2 (passed as args).
    const float m_hat = m1 / bc1;
    const float v_hat = v1 / bc2;

    // 1 / (sqrt(v_hat) + eps): use rsqrt+NR + reciprocal-style fold-in.
    // Equivalent to fast_rsqrt_nr(v_hat) / (1 + eps * fast_rsqrt_nr(v_hat))
    // which avoids a divide. The closed-form below uses a single rcp.
    const float rsv = fast_rsqrt_nr(fmaxf(v_hat, 0.0f));
    const float denom_inv = rsv / (1.0f + eps * rsv);

    // Decoupled weight decay: u = m_hat * denom_inv + wd * theta.
    const float u  = m_hat * denom_inv + wd * p0;
    const float p1 = p0 - lr * u;

    // Writeback. State always rn (BF16) or wt-store (FP32 via store_state).
    store_state(exp_avg_ptr    + idx, m1);
    store_state(exp_avg_sq_ptr + idx, v1);
    store_from_float(param_ptr + idx, p1);
    (void) step_for_sr;  // reserved for future BF16-param SR opt-in
}

// =====================================================================
// __global__ kernel: scalar grid-stride.
//
// __launch_bounds__ takes block_size from tuned_configs.h::GROKADAMW_CONFIGS
// (the AdamW family currently shares that table). We template the block
// size so __launch_bounds__ remains a compile-time constant.
// =====================================================================

template <typename ParamT, typename StateT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void fused_adamw_simple_step_kernel(
    ParamT*       __restrict__ params,
    StateT*       __restrict__ exp_avg,
    StateT*       __restrict__ exp_avg_sq,
    const GradT*  __restrict__ grads,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2,
    int64_t n_elements,
    int64_t step_count
) {
    static_assert(is_param_dtype<ParamT>::value, "AdamW: invalid ParamT");
    static_assert(is_state_dtype<StateT>::value, "AdamW: invalid StateT");
    static_assert(is_grad_dtype<GradT>::value,   "AdamW: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "AdamW: FP8 grad with FP32 param requires explicit rescale");

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const unsigned step_u = static_cast<unsigned>(step_count);
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        adamw_step_inline<ParamT, StateT, GradT>(
            params, exp_avg, exp_avg_sq, grads,
            i, lr, beta1, beta2, eps, wd, bc1, bc2, step_u
        );
    }
}

}}} // namespace sg::sm90::adamw

// =====================================================================
// FP32-only vec4 fast path: 1 vec load per (param, m, v, g), 1 vec store
// per (param, m, v). Halves issue-rate of memory ops on H100 over scalar.
// =====================================================================

namespace sg { namespace sm90 { namespace adamw {

template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void fused_adamw_simple_step_kernel_vec4_fp32(
    float*       __restrict__ params,
    float*       __restrict__ exp_avg,
    float*       __restrict__ exp_avg_sq,
    const float* __restrict__ grads,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2,
    int64_t n_vec    // = n_elements / 4
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;

    float4*       p4_out = reinterpret_cast<float4*>(params);
    float4*       m4_out = reinterpret_cast<float4*>(exp_avg);
    float4*       v4_out = reinterpret_cast<float4*>(exp_avg_sq);
    const float4* g4     = reinterpret_cast<const float4*>(grads);

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_vec; i += stride) {
        const float4 gv = stream_load4(g4 + i);
        const float4 m0 = stream_load4(reinterpret_cast<const float4*>(exp_avg)    + i);
        const float4 v0 = stream_load4(reinterpret_cast<const float4*>(exp_avg_sq) + i);
        const float4 p0 = stream_load4(reinterpret_cast<const float4*>(params)     + i);

        float4 m1, v1, p1;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float gk = (&gv.x)[k];
            const float mk = beta1 * (&m0.x)[k] + one_minus_b1 * gk;
            const float vk = beta2 * (&v0.x)[k] + one_minus_b2 * gk * gk;
            const float m_hat = mk * inv_bc1;
            const float v_hat = vk * inv_bc2;
            const float rsv   = fast_rsqrt_nr(fmaxf(v_hat, 0.0f));
            const float denom_inv = rsv / (1.0f + eps * rsv);
            const float u  = m_hat * denom_inv + wd * (&p0.x)[k];
            (&m1.x)[k] = mk;
            (&v1.x)[k] = vk;
            (&p1.x)[k] = (&p0.x)[k] - lr * u;
        }

        stream_store4(m4_out + i, m1);
        stream_store4(v4_out + i, v1);
        stream_store4(p4_out + i, p1);
    }
}

// =====================================================================
// Per-tensor templated launcher (signature exactly per build spec).
//
// bc1, bc2 are precomputed by the binding loop from beta and step_count;
// step_count is forwarded purely so future SR variants can seed a PRNG.
// =====================================================================

template <typename ParamT, typename StateT, typename GradT>
cudaError_t launch_fused_adamw_simple_step(
    ParamT* params, StateT* exp_avg, StateT* exp_avg_sq,
    const GradT* grads,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2,
    int64_t n_elements, int64_t step_count,
    cudaStream_t stream
) {
    static_assert(is_param_dtype<ParamT>::value, "AdamW: invalid ParamT");
    static_assert(is_state_dtype<StateT>::value, "AdamW: invalid StateT");
    static_assert(is_grad_dtype<GradT>::value,   "AdamW: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "AdamW: FP8 grad with FP32 param requires an explicit rescale");

    if (n_elements <= 0) return cudaSuccess;

    // Block size pulled from the autotuned table; do NOT hardcode.
    const LaunchConfig cfg =
        get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;

    // Cap blocks; grid-stride loop handles n > grid * block.
    constexpr int MAX_BLOCKS = 8192;

    constexpr bool is_all_fp32 =
        std::is_same<ParamT, float>::value &&
        std::is_same<StateT, float>::value &&
        std::is_same<GradT,  float>::value;

    const bool aligned4 =
        ((reinterpret_cast<uintptr_t>(params)     & 0xF) == 0) &&
        ((reinterpret_cast<uintptr_t>(exp_avg)    & 0xF) == 0) &&
        ((reinterpret_cast<uintptr_t>(exp_avg_sq) & 0xF) == 0) &&
        ((reinterpret_cast<uintptr_t>(grads)      & 0xF) == 0);

    if (cfg.vec4 && is_all_fp32 && aligned4 && (n_elements % 4 == 0)) {
        const int64_t n_vec = n_elements / 4;
        const int64_t needed = (n_vec + block - 1) / block;
        const int grid = static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);
        if (block == 128) {
            fused_adamw_simple_step_kernel_vec4_fp32<128>
                <<<grid, 128, 0, stream>>>(
                    reinterpret_cast<float*>(params),
                    reinterpret_cast<float*>(exp_avg),
                    reinterpret_cast<float*>(exp_avg_sq),
                    reinterpret_cast<const float*>(grads),
                    lr, beta1, beta2, eps, weight_decay, bc1, bc2, n_vec);
        } else if (block == 512) {
            fused_adamw_simple_step_kernel_vec4_fp32<512>
                <<<grid, 512, 0, stream>>>(
                    reinterpret_cast<float*>(params),
                    reinterpret_cast<float*>(exp_avg),
                    reinterpret_cast<float*>(exp_avg_sq),
                    reinterpret_cast<const float*>(grads),
                    lr, beta1, beta2, eps, weight_decay, bc1, bc2, n_vec);
        } else {
            fused_adamw_simple_step_kernel_vec4_fp32<256>
                <<<grid, 256, 0, stream>>>(
                    reinterpret_cast<float*>(params),
                    reinterpret_cast<float*>(exp_avg),
                    reinterpret_cast<float*>(exp_avg_sq),
                    reinterpret_cast<const float*>(grads),
                    lr, beta1, beta2, eps, weight_decay, bc1, bc2, n_vec);
        }
        (void) step_count;
        return cudaGetLastError();
    }

    // Scalar fallback: handles any dtype combo, including FP8 grads,
    // BF16/FP16 params, and BF16 state.
    const int64_t needed = (n_elements + block - 1) / block;
    const int grid = static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);
    if (block == 128) {
        fused_adamw_simple_step_kernel<ParamT, StateT, GradT, 128>
            <<<grid, 128, 0, stream>>>(
                params, exp_avg, exp_avg_sq, grads,
                lr, beta1, beta2, eps, weight_decay, bc1, bc2,
                n_elements, step_count);
    } else if (block == 512) {
        fused_adamw_simple_step_kernel<ParamT, StateT, GradT, 512>
            <<<grid, 512, 0, stream>>>(
                params, exp_avg, exp_avg_sq, grads,
                lr, beta1, beta2, eps, weight_decay, bc1, bc2,
                n_elements, step_count);
    } else {
        fused_adamw_simple_step_kernel<ParamT, StateT, GradT, 256>
            <<<grid, 256, 0, stream>>>(
                params, exp_avg, exp_avg_sq, grads,
                lr, beta1, beta2, eps, weight_decay, bc1, bc2,
                n_elements, step_count);
    }
    return cudaGetLastError();
}

}}} // namespace sg::sm90::adamw

// =====================================================================
// Vector wrapper at namespace sg::sm90 — matches the forward declaration
// in csrc/bindings/grokadamw.cpp::DECLARE_MT_GROKADAMW(sm90)::
// launch_fused_adamw_simple. Loops over the per-param vector, computes
// per-tensor (bc1, bc2) on the host from the (mutable) step counter,
// and dispatches the templated kernel above per a runtime dtype check.
//
// The dispatch here is on the param scalar_type only — state is assumed
// FP32 (the simple path used by Muon / LookSAM allocates FP32 state) and
// grad shares the param dtype (these callers do not pass FP8 grads).
// FP8/BF16-state paths are reachable via the templated entry point.
// =====================================================================

namespace sg { namespace sm90 {

inline void launch_fused_adamw_simple(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    std::vector<int64_t>& steps,
    float beta1, float beta2, float lr, float wd, float eps
) {
    const size_t n_params = params.size();
    if (n_params == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    for (size_t i = 0; i < n_params; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        steps[i] += 1;

        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));

        const int64_t N = params[i].numel();
        const int64_t step_count = steps[i];

        TORCH_CHECK(exp_avgs[i].scalar_type()    == at::kFloat,
                    "fused_adamw_simple: exp_avg must be float32");
        TORCH_CHECK(exp_avg_sqs[i].scalar_type() == at::kFloat,
                    "fused_adamw_simple: exp_avg_sq must be float32");
        TORCH_CHECK(grads[i].scalar_type() == params[i].scalar_type(),
                    "fused_adamw_simple: grad and param dtypes must match");

        switch (params[i].scalar_type()) {
            case at::kFloat: {
                auto err = sg::sm90::adamw::launch_fused_adamw_simple_step<
                    float, float, float>(
                    params[i].data_ptr<float>(),
                    exp_avgs[i].data_ptr<float>(),
                    exp_avg_sqs[i].data_ptr<float>(),
                    grads[i].data_ptr<float>(),
                    lr, beta1, beta2, eps, wd, bc1, bc2,
                    N, step_count, stream);
                TORCH_CHECK(err == cudaSuccess,
                    "fused_adamw_simple (fp32) failed: ",
                    cudaGetErrorString(err));
                break;
            }
            case at::kBFloat16: {
                auto err = sg::sm90::adamw::launch_fused_adamw_simple_step<
                    __nv_bfloat16, float, __nv_bfloat16>(
                    reinterpret_cast<__nv_bfloat16*>(params[i].data_ptr()),
                    exp_avgs[i].data_ptr<float>(),
                    exp_avg_sqs[i].data_ptr<float>(),
                    reinterpret_cast<const __nv_bfloat16*>(grads[i].data_ptr()),
                    lr, beta1, beta2, eps, wd, bc1, bc2,
                    N, step_count, stream);
                TORCH_CHECK(err == cudaSuccess,
                    "fused_adamw_simple (bf16) failed: ",
                    cudaGetErrorString(err));
                break;
            }
            case at::kHalf: {
                auto err = sg::sm90::adamw::launch_fused_adamw_simple_step<
                    __half, float, __half>(
                    reinterpret_cast<__half*>(params[i].data_ptr()),
                    exp_avgs[i].data_ptr<float>(),
                    exp_avg_sqs[i].data_ptr<float>(),
                    reinterpret_cast<const __half*>(grads[i].data_ptr()),
                    lr, beta1, beta2, eps, wd, bc1, bc2,
                    N, step_count, stream);
                TORCH_CHECK(err == cudaSuccess,
                    "fused_adamw_simple (fp16) failed: ",
                    cudaGetErrorString(err));
                break;
            }
            default:
                TORCH_CHECK(false,
                    "fused_adamw_simple: unsupported param dtype ",
                    params[i].scalar_type());
        }
    }
}

}} // namespace sg::sm90
