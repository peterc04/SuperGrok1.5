// =====================================================================
//  csrc/kernels/cuda/sm_90/grokfast.cu
//
//  Thin instantiation TU for the sm_90 Grokfast launchers declared in
//  csrc/kernels/cuda/sm_90/grokfast.cuh. All kernel logic lives in the
//  header; this TU only forces the template instantiations the linker
//  needs for the dtype matrix:
//
//     ParamT in {float, __nv_bfloat16, __half}
//     StateT in {float, __nv_bfloat16}
//     GradT  in {float, __nv_bfloat16, __half,
//                __nv_fp8_e4m3, __nv_fp8_e5m2}
//
//  Counts:
//     - launch_grokfast_fused_ema_adam_step : 3 × 2 × 5 = 30 combos
//     - launch_grokfast_fused_step          : 1 × 2 × 5 = 10 combos
//                                             (ParamT is unused on this
//                                             path; pin to float so the
//                                             SFINAE guard is satisfied
//                                             without expanding the matrix)
//     - Total                               : 40 instantiations
//
//  Incoherent dtype combos are rejected at compile time inside the
//  header via static_assert (see is_param_dtype / is_state_dtype /
//  is_grad_dtype) so any nonsense from a future caller fails the build
//  rather than miscompiling silently.
// =====================================================================

#include "csrc/kernels/cuda/sm_90/grokfast.cuh"

namespace sg { namespace sm90 { namespace grokfast {

// X-macro over the GradT axis. Used by both launchers.
#define SG_GROKFAST_FOREACH_GRAD(_)                  \
    _(float)                                         \
    _(__nv_bfloat16)                                 \
    _(__half)                                        \
    _(__nv_fp8_e4m3)                                 \
    _(__nv_fp8_e5m2)

// X-macro over (ParamT, StateT) for the fused EMA+Adam path.
#define SG_GROKFAST_FOREACH_PARAM_STATE(_)           \
    _(float,         float)                          \
    _(float,         __nv_bfloat16)                  \
    _(__nv_bfloat16, float)                          \
    _(__nv_bfloat16, __nv_bfloat16)                  \
    _(__half,        float)                          \
    _(__half,        __nv_bfloat16)

// ---------------------------------------------------------------------
// launch_grokfast_fused_ema_adam_step — 30 instantiations
// ---------------------------------------------------------------------

#define SG_GF_INST_ADAM(PARAM_T, STATE_T, GRAD_T)                              \
    template cudaError_t                                                       \
    launch_grokfast_fused_ema_adam_step<PARAM_T, STATE_T, GRAD_T>(             \
        PARAM_T*, STATE_T*, STATE_T*, STATE_T*, const GRAD_T*,                 \
        float, float, float, float, float,                                     \
        float, float, float, float,                                            \
        int64_t, int64_t, cudaStream_t);

#define SG_GF_INST_ADAM_BIND_PARAM_STATE(PARAM_T, STATE_T)                     \
    SG_GF_INST_ADAM(PARAM_T, STATE_T, float)                                   \
    SG_GF_INST_ADAM(PARAM_T, STATE_T, __nv_bfloat16)                           \
    SG_GF_INST_ADAM(PARAM_T, STATE_T, __half)                                  \
    SG_GF_INST_ADAM(PARAM_T, STATE_T, __nv_fp8_e4m3)                           \
    SG_GF_INST_ADAM(PARAM_T, STATE_T, __nv_fp8_e5m2)

SG_GROKFAST_FOREACH_PARAM_STATE(SG_GF_INST_ADAM_BIND_PARAM_STATE)

#undef SG_GF_INST_ADAM_BIND_PARAM_STATE
#undef SG_GF_INST_ADAM

// ---------------------------------------------------------------------
// launch_grokfast_fused_step — 10 instantiations (ParamT pinned to float)
// ---------------------------------------------------------------------

#define SG_GF_INST_EMA(STATE_T, GRAD_T)                                        \
    template cudaError_t                                                       \
    launch_grokfast_fused_step<float, STATE_T, GRAD_T>(                        \
        STATE_T*, GRAD_T*, float, float, int64_t, cudaStream_t);

#define SG_GF_INST_EMA_BIND_STATE(STATE_T)                                     \
    SG_GF_INST_EMA(STATE_T, float)                                             \
    SG_GF_INST_EMA(STATE_T, __nv_bfloat16)                                     \
    SG_GF_INST_EMA(STATE_T, __half)                                            \
    SG_GF_INST_EMA(STATE_T, __nv_fp8_e4m3)                                     \
    SG_GF_INST_EMA(STATE_T, __nv_fp8_e5m2)

SG_GF_INST_EMA_BIND_STATE(float)
SG_GF_INST_EMA_BIND_STATE(__nv_bfloat16)

#undef SG_GF_INST_EMA_BIND_STATE
#undef SG_GF_INST_EMA

#undef SG_GROKFAST_FOREACH_PARAM_STATE
#undef SG_GROKFAST_FOREACH_GRAD

}}} // namespace sg::sm90::grokfast

// =====================================================================
// torch::Tensor binding shims in namespace sg::sm90.
//
// Forward-declared by csrc/bindings/grokfast.cpp DECLARE_GROKFAST(sm90).
// These accept torch::Tensors, dispatch on dtype, extract raw pointers,
// and delegate to the inner sg::sm90::grokfast:: templated launchers.
// =====================================================================

#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

namespace sg { namespace sm90 {

// ---------------------------------------------------------------------
// launch_fused_grokfast_ema — EMA-only filter pass.
//   Delegates to grokfast::launch_grokfast_fused_step<float, StateT, GradT>.
// ---------------------------------------------------------------------
void launch_fused_grokfast_ema(
    torch::Tensor grad,
    torch::Tensor ema,
    float alpha,
    float lamb
) {
    TORCH_CHECK(grad.is_cuda(),  "grokfast_ema sm90: grad must be CUDA");
    TORCH_CHECK(ema.is_cuda(),   "grokfast_ema sm90: ema must be CUDA");
    TORCH_CHECK(grad.is_contiguous(), "grokfast_ema sm90: grad must be contiguous");
    TORCH_CHECK(ema.is_contiguous(),  "grokfast_ema sm90: ema must be contiguous");
    TORCH_CHECK(grad.numel() == ema.numel(),
                "grokfast_ema sm90: grad/ema numel mismatch");

    const int64_t N = grad.numel();
    if (N == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // State (ema) is always FP32 from the Python optimizer.
    TORCH_CHECK(ema.scalar_type() == at::kFloat,
                "grokfast_ema sm90: ema must be float32");

    // Dispatch on grad dtype. ParamT is pinned to float (unused on EMA-only path).
    auto dispatch_grad = [&](auto grad_tag) {
        using GradT = decltype(grad_tag);
        auto err = ::sg::sm90::grokfast::launch_grokfast_fused_step<float, float, GradT>(
            ema.data_ptr<float>(),
            reinterpret_cast<GradT*>(grad.data_ptr()),
            alpha, lamb, N, stream);
        TORCH_CHECK(err == cudaSuccess,
            "launch_fused_grokfast_ema (sm90): ", cudaGetErrorString(err));
    };

    switch (grad.scalar_type()) {
        case at::kFloat:    dispatch_grad(float{});          break;
        case at::kBFloat16: dispatch_grad(__nv_bfloat16{}); break;
        case at::kHalf:     dispatch_grad(__half{});         break;
        default:
            TORCH_CHECK(false,
                "grokfast_ema sm90: unsupported grad dtype ", grad.scalar_type());
    }
}

// ---------------------------------------------------------------------
// launch_fused_grokfast_adam — fused Grokfast EMA + AdamW step.
//   Delegates to grokfast::launch_grokfast_fused_ema_adam_step.
// ---------------------------------------------------------------------
void launch_fused_grokfast_adam(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor ema,
    torch::Tensor grad,
    float alpha,
    float lamb,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float bc1,
    float bc2
) {
    TORCH_CHECK(param.is_cuda(),      "grokfast_adam sm90: param must be CUDA");
    TORCH_CHECK(exp_avg.is_cuda(),    "grokfast_adam sm90: exp_avg must be CUDA");
    TORCH_CHECK(exp_avg_sq.is_cuda(), "grokfast_adam sm90: exp_avg_sq must be CUDA");
    TORCH_CHECK(ema.is_cuda(),        "grokfast_adam sm90: ema must be CUDA");
    TORCH_CHECK(grad.is_cuda(),       "grokfast_adam sm90: grad must be CUDA");
    TORCH_CHECK(param.is_contiguous(),      "grokfast_adam sm90: param must be contiguous");
    TORCH_CHECK(exp_avg.is_contiguous(),    "grokfast_adam sm90: exp_avg must be contiguous");
    TORCH_CHECK(exp_avg_sq.is_contiguous(), "grokfast_adam sm90: exp_avg_sq must be contiguous");
    TORCH_CHECK(ema.is_contiguous(),        "grokfast_adam sm90: ema must be contiguous");
    TORCH_CHECK(grad.is_contiguous(),       "grokfast_adam sm90: grad must be contiguous");

    const int64_t N = param.numel();
    if (N == 0) return;

    // State tensors are FP32 from the Python optimizer.
    TORCH_CHECK(exp_avg.scalar_type()    == at::kFloat,
                "grokfast_adam sm90: exp_avg must be float32");
    TORCH_CHECK(exp_avg_sq.scalar_type() == at::kFloat,
                "grokfast_adam sm90: exp_avg_sq must be float32");
    TORCH_CHECK(ema.scalar_type()        == at::kFloat,
                "grokfast_adam sm90: ema must be float32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Recover step from bc1 for stochastic-rounding PRNG seed.
    int64_t step_count = 0;
    if (bc1 > 0.0f && beta1 > 0.0f && beta1 < 1.0f) {
        const float t = std::log(std::max(1.0f - bc1, 1e-30f))
                        / std::log(beta1);
        step_count = static_cast<int64_t>(std::lround(std::max(t, 1.0f)));
    }

    // Dispatch on param dtype. Grad dtype matches param (Python optimizer
    // constructs grads on the param), state is FP32.
    TORCH_CHECK(grad.scalar_type() == param.scalar_type(),
                "grokfast_adam sm90: grad and param dtypes must match");

    auto dispatch_typed = [&](auto param_tag) {
        using ParamT = decltype(param_tag);
        // GradT == ParamT (checked above).
        auto err = ::sg::sm90::grokfast::launch_grokfast_fused_ema_adam_step<
            ParamT, float, ParamT>(
            reinterpret_cast<ParamT*>(param.data_ptr()),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            ema.data_ptr<float>(),
            reinterpret_cast<const ParamT*>(grad.data_ptr()),
            lr, beta1, beta2, eps, weight_decay,
            alpha, lamb, bc1, bc2,
            N, step_count, stream);
        TORCH_CHECK(err == cudaSuccess,
            "launch_fused_grokfast_adam (sm90): ", cudaGetErrorString(err));
    };

    switch (param.scalar_type()) {
        case at::kFloat:    dispatch_typed(float{});          break;
        case at::kBFloat16: dispatch_typed(__nv_bfloat16{}); break;
        case at::kHalf:     dispatch_typed(__half{});         break;
        default:
            TORCH_CHECK(false,
                "grokfast_adam sm90: unsupported param dtype ",
                param.scalar_type());
    }
}

}} // namespace sg::sm90
