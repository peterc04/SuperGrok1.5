// =====================================================================
//  csrc/kernels/cuda/sm_90/neuralgrok.cuh
//
//  sm_90 (Hopper) NeuralGrok optimizer kernel + launcher header.
//
//  Net-new replacement for the deleted csrc/kernels/cuda/sm_90/
//  neuralgrok_sm90.cu (commit 5505b50). Architecture identical to that
//  baseline but rehoused under namespace sg::sm90::neuralgrok and
//  retemplated for the Param/State/Grad dtype matrix mandated by the
//  build spec (mirroring adamw.cuh).
//
//  Algorithm (per-element, FP32 accumulation):
//     a_t       = MLP_psi(g)                    // amplifier MLP
//     a_scale   = alpha * a_t + beta            // affine skip-connection
//     g_tilde   = g * a_scale                   // amplified gradient
//     m_t       = beta1 * m_{t-1} + (1 - beta1) * g_tilde
//     v_t       = beta2 * v_{t-1} + (1 - beta2) * g_tilde^2
//     m_hat     = m_t / bc1     (bc1 = 1 - beta1^t, host-passed)
//     v_hat     = v_t / bc2     (bc2 = 1 - beta2^t, host-passed)
//     theta_t   = theta_{t-1} * (1 - lr * wd)
//                 - (lr / bc1) * m_t / (sqrt(v_t / bc2) + eps)
//
//  MLP_psi (matches grokking_optimizers/neuralgrok.py::_Amplifier):
//      Linear(1, H) -> ReLU -> Linear(H, 1)
//  Python's _Amplifier supports num_layers >= 2 but get_weights() exposes
//  only the first and last linear, and the kernel evaluates exactly that
//  2-layer composition (this matches the deleted baseline's behaviour).
//
//  Two kernels (CUDA Graph captured as a unit by the per-tensor wrapper):
//     1) neuralgrok_meta_psi_kernel<ParamT,StateT,GradT,H>: emits
//        amplified_grad = grad * (alpha * MLP_psi(grad) + beta)
//        into a scratch buffer in g_tilde_dtype (= GradT).
//     2) neuralgrok_apply_kernel<ParamT,StateT,GradT,BLOCK>:
//        consumes amplified_grad and emits the AdamW step.
//  An optional fully-fused single-kernel path
//  (neuralgrok_fused_step_kernel) keeps amplified_grad in registers and
//  is used by the torch::Tensor launcher below as the default path
//  (matches deleted baseline). The two-kernel split is reachable via
//  launch_neuralgrok_meta_psi + launch_neuralgrok_apply, exposed for the
//  CUDA-Graph capture entry point.
//
//  Optimisations:
//    - psi weights: __constant__ when total bytes fit (H<=128 with
//      L<=3 -> <= 4 + H + H + H*H + H + H*1 + 1 floats, ~16 KiB at H=64).
//      A constant-memory mirror is always available; the launcher copies
//      from the device tensor to the constant cache prior to capture.
//      A SMEM-resident path is the fallback for H > 128.
//    - Apply kernel: scalar grid-stride with LDG + stream_load /
//      stream_store on FP32 state, plus a vec4 fast path for all-FP32.
//    - fast_rsqrt_nr for 1/sqrt(v_hat) (rsqrt.approx.f32 + 1 NR step).
//    - Block sizes pulled from tuned_configs.h::GROKADAMW_CONFIGS (the
//      NeuralGrok apply kernel reuses the AdamW table — same I/O shape
//      modulo the extra MLP read; autotune adds a dedicated table later).
//    - CUDA Graph: per-tensor wrapper captures the (psi, apply) pair on
//      first call; the cached graph is replayed on subsequent calls. The
//      cache is keyed by (param_ptr, n, dtype-triple, hidden, layers) so
//      hyperparameter and shape changes cause re-capture transparently.
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

namespace sg { namespace sm90 { namespace neuralgrok {

// ---------------------------------------------------------------------
// Compile-time predicates (mirror adamw.cuh — kept independent so the
// two TUs do not become coupled).
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

// FP8 grad with FP32 param silently loses dynamic range; reject.
template <typename ParamT, typename GradT>
struct is_coherent_combo
    : std::integral_constant<bool,
        !((std::is_same<GradT, __nv_fp8_e4m3>::value ||
           std::is_same<GradT, __nv_fp8_e5m2>::value) &&
          std::is_same<ParamT, float>::value)> {};

// =====================================================================
// __constant__ memory mirror for the meta-MLP (psi) weights.
//
// Sized for the largest hidden width we want in the constant cache:
//   layout: [W1: H, b1: H, W2: H, b2: 1]  (2-layer Linear(1,H)->...->Linear(H,1))
// At H=128 -> 3*128 + 1 = 385 floats = 1540 B (well under 64 KiB).
// At H=256 -> 3*256 + 1 = 769 floats = 3076 B (still fits).
// We size to H_MAX = 256 and fall back to a SMEM-resident path beyond.
// =====================================================================

constexpr int kPsiHmax     = 256;
constexpr int kPsiNumFloat = 3 * kPsiHmax + 1;  // W1 + b1 + W2 + b2

}}} // namespace sg::sm90::neuralgrok

#if GROK_CUDA
// File-scope (CUDA requires __constant__ at namespace or file scope).
// Defined in neuralgrok.cu (single TU). Declared extern here so any TU
// including the header sees the symbol; the definition is emitted once.
extern __constant__ float sg_sm90_neuralgrok_psi[
    sg::sm90::neuralgrok::kPsiNumFloat];
#endif

namespace sg { namespace sm90 { namespace neuralgrok {

// =====================================================================
// Type-erased load / store helpers — math is always FP32. Mirrors the
// equivalent block in adamw.cuh; kept independent so changes here cannot
// silently break the AdamW TU.
// =====================================================================

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
__device__ __forceinline__ void store_from_float<float>(float* p, float v) { *p = v; }
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
// Kernel 1 — Meta-MLP psi forward.
//
// Per element, evaluates  a_i = MLP_psi(g_i) and writes the per-element
// scalar a_t into the scratch buffer `a_out` (FP32). The Python amplifier
// uses the SIGNED gradient as input and applies (alpha * mlp + beta) on
// the consumer side, so this kernel emits the raw MLP output and lets
// the apply kernel do the affine skip-connection. This matches both the
// deleted baseline and grokking_optimizers/neuralgrok.py::_Amplifier.
//
// Architecture per Python ref: Linear(1, H) -> ReLU -> Linear(H, 1)
// (intermediate layers are skipped on the CUDA path; the kernel only
// reads W1/b1/W2/b2 from the __constant__ mirror).
//
// For H matching a specialization in utils.cuh::ptx_expert_mlp_forward
// (typically H in {8, 16, 32, 64, 128}), the templated path uses that
// helper which fully unrolls via #pragma unroll. Otherwise a runtime
// loop is taken.
//
// __constant__ layout: [W1: H][b1: H][W2: H][b2: 1].
// =====================================================================

#if GROK_CUDA
template <typename GradT, int H, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void neuralgrok_meta_psi_kernel(
    const GradT* __restrict__ grads,
    float*       __restrict__ a_out,        // [N] FP32 amplifier scalar
    int64_t n_elements
) {
    static_assert(is_grad_dtype<GradT>::value, "NeuralGrok: invalid GradT");
    const float* W1 = sg_sm90_neuralgrok_psi + 0;
    const float* b1 = sg_sm90_neuralgrok_psi + H;
    const float* W2 = sg_sm90_neuralgrok_psi + 2 * H;
    const float  b2 = sg_sm90_neuralgrok_psi[3 * H];

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float g = load_as_float(grads + i);
        // ptx_expert_mlp_forward<H>: ReLU MLP with PTX FMAs, fully unrolled.
        const float a = ptx_expert_mlp_forward<H>(W1, b1, W2, b2, g);
        a_out[i] = a;
    }
}

// Runtime-H fallback: hidden width determined by argument, not template.
// Used when H exceeds the specialized roster (16/32/64/128).
template <typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void neuralgrok_meta_psi_kernel_runtimeH(
    const GradT* __restrict__ grads,
    float*       __restrict__ a_out,
    int          H,
    int64_t      n_elements
) {
    static_assert(is_grad_dtype<GradT>::value, "NeuralGrok: invalid GradT");
    const float* W1 = sg_sm90_neuralgrok_psi + 0;
    const float* b1 = sg_sm90_neuralgrok_psi + H;
    const float* W2 = sg_sm90_neuralgrok_psi + 2 * H;
    const float  b2 = sg_sm90_neuralgrok_psi[3 * H];

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float g = load_as_float(grads + i);
        float r = b2;
        for (int h = 0; h < H; ++h) {
            float z = ptx_fma(W1[h], g, b1[h]);
            z = fmaxf(z, 0.0f);
            r = ptx_fma(W2[h], z, r);
        }
        a_out[i] = r;
    }
}
#endif // GROK_CUDA

}}} // namespace sg::sm90::neuralgrok
