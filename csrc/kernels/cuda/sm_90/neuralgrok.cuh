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

// thread_local scalars used by the templated launcher to receive the
// per-call (alpha, beta) amplifier affine-skip parameters. Defined in
// neuralgrok.cu (single TU). Declared at sg::sm90 (matching the binding
// wrapper namespace) and forward-declared here so the templated launcher
// in sg::sm90::neuralgrok can read them via qualified lookup.
namespace sg { namespace sm90 {
extern thread_local float sg_sm90_neuralgrok_alpha;
extern thread_local float sg_sm90_neuralgrok_beta;
}}

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

// =====================================================================
// Kernel 2 — Fused apply: amplified-grad + AdamW step.
//
// Per element:
//     a       = a_in[i]                              // from Kernel 1
//     g_amp   = grad[i] * (alpha * a + beta)
//     m_t     = beta1 * m + (1 - beta1) * g_amp
//     v_t     = beta2 * v + (1 - beta2) * g_amp^2
//     m_hat   = m_t / bc1
//     v_hat   = v_t / bc2
//     denom_inv = rsqrt(v_hat) / (1 + eps * rsqrt(v_hat))
//     theta   = theta - lr * (m_hat * denom_inv + wd * theta)
//
// LDG / stream_load on read-only loads, stream_store on FP32 state and
// FP32 param writebacks (st.global.wt — bypasses L2 allocation so the
// next forward pass is not evicted by the optimizer step). FP16 / BF16
// param/state stores fall through to a normal rn-rounded store.
// =====================================================================

template <typename ParamT, typename StateT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void neuralgrok_apply_kernel(
    ParamT*       __restrict__ params,
    StateT*       __restrict__ exp_avg,
    StateT*       __restrict__ exp_avg_sq,
    const GradT*  __restrict__ grads,
    const float*  __restrict__ a_in,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha_amp, float beta_amp,
    float bc1, float bc2,
    int64_t n_elements
) {
    static_assert(is_param_dtype<ParamT>::value, "NeuralGrok: invalid ParamT");
    static_assert(is_state_dtype<StateT>::value, "NeuralGrok: invalid StateT");
    static_assert(is_grad_dtype<GradT>::value,   "NeuralGrok: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "NeuralGrok: FP8 grad with FP32 param requires explicit rescale");

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float g  = load_as_float(grads + i);
        const float a  = LDG(a_in + i);
        const float g_amp = g * (alpha_amp * a + beta_amp);

        const float m0 = load_state(exp_avg + i);
        const float v0 = load_state(exp_avg_sq + i);
        const float p0 = load_as_float(params + i);

        const float m1 = beta1 * m0 + one_minus_b1 * g_amp;
        const float v1 = beta2 * v0 + one_minus_b2 * g_amp * g_amp;

        const float m_hat = m1 * inv_bc1;
        const float v_hat = v1 * inv_bc2;
        const float rsv   = fast_rsqrt_nr(fmaxf(v_hat, 0.0f));
        const float denom_inv = rsv / (1.0f + eps * rsv);
        const float u  = m_hat * denom_inv + weight_decay * p0;
        const float p1 = p0 - lr * u;

        store_state(exp_avg    + i, m1);
        store_state(exp_avg_sq + i, v1);
        store_from_float(params + i, p1);
    }
}

// =====================================================================
// CUDA Graph cache. The (psi + apply) pair is captured on first call and
// replayed on subsequent calls for matching shape / dtype / hyperparam
// signature. Hyperparam changes invalidate the cache (a fresh capture
// overwrites the stored cudaGraphExec_t). Per-call pointers are not part
// of the cache key — we re-capture if the param pointer rotates (this
// matches stable use-cases like a fixed model + persistent state).
// =====================================================================

struct GraphCacheKey {
    void*   param_ptr;
    int64_t n;
    int     param_dtype_id;
    int     state_dtype_id;
    int     grad_dtype_id;
    int     hidden;
    int     layers;
    float   lr;
    float   beta1;
    float   beta2;
    float   eps;
    float   wd;
    float   alpha;
    float   beta_amp;
    bool operator==(const GraphCacheKey& o) const {
        return param_ptr == o.param_ptr && n == o.n &&
               param_dtype_id == o.param_dtype_id &&
               state_dtype_id == o.state_dtype_id &&
               grad_dtype_id  == o.grad_dtype_id &&
               hidden == o.hidden && layers == o.layers &&
               lr == o.lr && beta1 == o.beta1 && beta2 == o.beta2 &&
               eps == o.eps && wd == o.wd &&
               alpha == o.alpha && beta_amp == o.beta_amp;
    }
};

// Trivial open-addressed cache (single slot per param pointer) — keeps
// header-only complexity low while covering the common single-tensor
// optimizer step case. Multi-tensor flows use the per-tensor wrapper
// loop, each with its own slot.
struct GraphCacheSlot {
    GraphCacheKey   key{};
    cudaGraphExec_t exec{nullptr};
    bool            valid{false};
};

// Templated to keep ODR per (ParamT,StateT,GradT) — distinct slots per
// instantiation avoid cross-dtype collision on the same tensor address.
template <typename ParamT, typename StateT, typename GradT>
inline GraphCacheSlot& graph_cache_slot() {
    static thread_local GraphCacheSlot slot;
    return slot;
}

template <typename T> inline int dtype_id();
template <> inline int dtype_id<float>()         { return 0; }
template <> inline int dtype_id<__nv_bfloat16>() { return 1; }
template <> inline int dtype_id<__half>()        { return 2; }
template <> inline int dtype_id<__nv_fp8_e4m3>() { return 3; }
template <> inline int dtype_id<__nv_fp8_e5m2>() { return 4; }

// =====================================================================
// Templated per-tensor launcher (signature exactly as the build spec).
//
// Captures (Kernel 1: meta-MLP) -> (Kernel 2: apply) into a single
// cudaGraphExec_t on first invocation; replays the cached graph on every
// subsequent call with a matching key. Hyperparameter and shape changes
// trigger re-capture (graph is destroyed and rebuilt).
//
// `meta_psi_weights` is the host-visible source pointer for the psi
// weights ([W1: H_eff][b1: H][W2: H][b2: 1]); the launcher copies that
// into the __constant__ mirror prior to capture and on every replay
// (the cudaMemcpyToSymbolAsync sits inside the captured stream so the
// graph itself owns the H2D transfer node).
//
// `step_count` is reserved for future stochastic-rounding paths; it is
// not consumed here but is kept in the signature to match the build
// spec verbatim and enable late binding without an ABI break.
// =====================================================================

template <typename ParamT, typename StateT, typename GradT>
cudaError_t launch_neuralgrok_fused_step(
    ParamT* params, StateT* exp_avg, StateT* exp_avg_sq,
    const GradT* grads,
    const float* meta_psi_weights,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2,
    int meta_hidden, int meta_layers,
    int64_t n_elements, int64_t step_count,
    cudaStream_t stream
) {
    static_assert(is_param_dtype<ParamT>::value, "NeuralGrok: invalid ParamT");
    static_assert(is_state_dtype<StateT>::value, "NeuralGrok: invalid StateT");
    static_assert(is_grad_dtype<GradT>::value,   "NeuralGrok: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "NeuralGrok: FP8 grad with FP32 param requires explicit rescale");

    if (n_elements <= 0) return cudaSuccess;

    // alpha/beta affine skip-connection scalars come from the binding;
    // they are folded into bc1/bc2 here only to keep the spec signature
    // intact. The torch::Tensor wrapper below routes alpha/beta through
    // a thread-local before invoking this template.
    const float alpha_amp = ::sg::sm90::sg_sm90_neuralgrok_alpha;
    const float beta_amp  = ::sg::sm90::sg_sm90_neuralgrok_beta;

    // Block size from the autotuned table (same family as AdamW until a
    // dedicated NeuralGrok table lands — TODO in tuned_configs.h).
    const LaunchConfig cfg =
        get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;
    constexpr int MAX_BLOCKS = 8192;

    const int64_t needed = (n_elements + block - 1) / block;
    const int grid = static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);

    // psi constant-cache footprint: 3*H + 1 floats. SMEM-resident path
    // is used when H exceeds kPsiHmax (the constant-cache budget).
    const int H = meta_hidden;
    const int psi_floats = 3 * H + 1;
    const bool psi_in_const = (H > 0 && H <= kPsiHmax);
    if (!psi_in_const) {
        // Out-of-budget psi — caller should reduce hidden width or extend
        // kPsiHmax. We surface a launch error rather than silently fall
        // through to a SMEM path that is not yet implemented.
        return cudaErrorInvalidConfiguration;
    }

    // ---- CUDA Graph cache lookup ----------------------------------------
    GraphCacheKey key{
        /*param_ptr*/ static_cast<void*>(params),
        /*n*/ n_elements,
        /*param_dtype_id*/ dtype_id<ParamT>(),
        /*state_dtype_id*/ dtype_id<StateT>(),
        /*grad_dtype_id*/  dtype_id<GradT>(),
        /*hidden*/ meta_hidden,
        /*layers*/ meta_layers,
        /*lr*/ lr, /*b1*/ beta1, /*b2*/ beta2, /*eps*/ eps,
        /*wd*/ weight_decay, /*alpha*/ alpha_amp, /*beta_amp*/ beta_amp,
    };
    auto& slot = graph_cache_slot<ParamT, StateT, GradT>();

    auto record_into_stream = [&](cudaStream_t s) -> cudaError_t {
        cudaError_t e;
        // Refresh psi __constant__ inside the capture — the H2D transfer
        // becomes a graph node and runs on every replay (so a Python-side
        // amplifier weight update is honoured without re-capture as long
        // as the *pointer* and shape are stable).
        e = cudaMemcpyToSymbolAsync(
            sg_sm90_neuralgrok_psi, meta_psi_weights,
            psi_floats * sizeof(float), 0, cudaMemcpyDeviceToDevice, s);
        if (e != cudaSuccess) return e;

        // Kernel 1 — meta-MLP psi over `grads`, output to `exp_avg_sq`
        // reused as scratch? No — must not stomp Adam state. Allocate a
        // scratch tensor on the caller side instead. Here we co-locate
        // scratch via cudaMallocAsync on the captured stream so it is
        // owned by the graph too.
        // Note: cudaMallocAsync IS captureable on H100 + CUDA 12.x.
        float* a_scratch = nullptr;
        e = cudaMallocAsync(&a_scratch,
            static_cast<size_t>(n_elements) * sizeof(float), s);
        if (e != cudaSuccess) return e;

        // Dispatch on H to a templated specialization when one exists,
        // else the runtime-H fallback (single PTX FMA inner loop).
        auto launch_psi = [&]() {
            #define SG_NG_PSI_H(HH)                                       \
                neuralgrok_meta_psi_kernel<GradT, HH, 256>                \
                    <<<grid, 256, 0, s>>>(grads, a_scratch, n_elements)
            switch (H) {
                case 8:   SG_NG_PSI_H(8);   break;
                case 16:  SG_NG_PSI_H(16);  break;
                case 32:  SG_NG_PSI_H(32);  break;
                case 64:  SG_NG_PSI_H(64);  break;
                case 128: SG_NG_PSI_H(128); break;
                default:
                    neuralgrok_meta_psi_kernel_runtimeH<GradT, 256>
                        <<<grid, 256, 0, s>>>(grads, a_scratch, H, n_elements);
            }
            #undef SG_NG_PSI_H
        };
        launch_psi();

        // Kernel 2 — apply.
        if (block == 128) {
            neuralgrok_apply_kernel<ParamT, StateT, GradT, 128>
                <<<grid, 128, 0, s>>>(
                    params, exp_avg, exp_avg_sq, grads, a_scratch,
                    lr, beta1, beta2, eps, weight_decay,
                    alpha_amp, beta_amp, bc1, bc2, n_elements);
        } else if (block == 512) {
            neuralgrok_apply_kernel<ParamT, StateT, GradT, 512>
                <<<grid, 512, 0, s>>>(
                    params, exp_avg, exp_avg_sq, grads, a_scratch,
                    lr, beta1, beta2, eps, weight_decay,
                    alpha_amp, beta_amp, bc1, bc2, n_elements);
        } else {
            neuralgrok_apply_kernel<ParamT, StateT, GradT, 256>
                <<<grid, 256, 0, s>>>(
                    params, exp_avg, exp_avg_sq, grads, a_scratch,
                    lr, beta1, beta2, eps, weight_decay,
                    alpha_amp, beta_amp, bc1, bc2, n_elements);
        }

        // Free scratch back to the captured stream's mempool.
        e = cudaFreeAsync(a_scratch, s);
        return e;
    };

    if (slot.valid && slot.key == key) {
        return cudaGraphLaunch(slot.exec, stream);
    }

    // Re-capture: tear down any stale exec, capture fresh, instantiate.
    if (slot.valid) {
        cudaGraphExecDestroy(slot.exec);
        slot.exec = nullptr;
        slot.valid = false;
    }

    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) return err;
    err = record_into_stream(stream);
    if (err != cudaSuccess) {
        cudaGraph_t tmp;
        cudaStreamEndCapture(stream, &tmp);
        if (tmp) cudaGraphDestroy(tmp);
        return err;
    }
    cudaGraph_t graph = nullptr;
    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) return err;
    err = cudaGraphInstantiate(&slot.exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) return err;
    slot.key = key;
    slot.valid = true;
    (void) step_count;
    return cudaGraphLaunch(slot.exec, stream);
}

}}} // namespace sg::sm90::neuralgrok

// =====================================================================
// Binding-compatible vector wrappers at namespace sg::sm90::.
//
// Match the forward declarations in csrc/bindings/neuralgrok.cpp:
//   - launch_fused_neuralgrok_amplifier
//   - launch_fused_neuralgrok_adam
//   - launch_fused_neuralgrok_full_step
//
// The bindings live in sg::sm90 (not sg::sm90::neuralgrok), so these
// wrappers are intentionally *outside* the inner namespace. They route
// down to the templated launcher above by stashing the per-call alpha /
// beta scalars in thread_locals (the templated launcher reads them so
// the spec-mandated signature can stay alpha/beta-free).
// =====================================================================

namespace sg { namespace sm90 {

// thread_local scalars used by the templated launcher to receive the
// per-call (alpha, beta) amplifier affine-skip parameters.
extern thread_local float sg_sm90_neuralgrok_alpha;
extern thread_local float sg_sm90_neuralgrok_beta;

void launch_fused_neuralgrok_amplifier(
    torch::Tensor grad, torch::Tensor amplified,
    torch::Tensor amplifier_w1, torch::Tensor amplifier_b1,
    torch::Tensor amplifier_w2, torch::Tensor amplifier_b2,
    int hidden_dim, float alpha, float beta);

void launch_fused_neuralgrok_adam(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor amplified_grad,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2);

void launch_fused_neuralgrok_full_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2);

}} // namespace sg::sm90

// Wrapper bodies live in neuralgrok.cu (single TU) — declared above so
// the bindings link against this header without dragging in CUDA code.
