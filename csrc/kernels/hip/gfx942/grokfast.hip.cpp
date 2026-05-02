// =====================================================================
//  csrc/kernels/hip/gfx942/grokfast.hip.cpp
//
//  gfx942 Grokfast launchers:
//    1) launch_fused_grokfast_ema   — EMA-only filter pass
//    2) launch_fused_grokfast_adam  — fused EMA + AdamW step
//    3) launch_multi_tensor_grokfast_ema — batched EMA over a list
//
//  Grokfast EMA: ema_t = lamb * ema_{t-1} + (1 - lamb) * g
//                g'    = g + alpha * ema_t   (in-place if requested)
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/grokfast.hip.h"

namespace sg { namespace gfx942 { namespace grokfast_impl {

// ---------------------------------------------------------------------
// EMA-only kernel. ema is FP32 in storage; grad is the actual model grad
// dtype. We update ema in place; grad is amplified by alpha * ema in
// place (matching the sm_90 launcher's "fused_step" semantics).
// ---------------------------------------------------------------------

template <typename GradT>
__global__ void grokfast_ema_kernel(
    float* __restrict__ ema,
    GradT* __restrict__ grad,
    const float alpha, const float lamb,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float g = static_cast<float>(grad[idx]);
    const float new_ema = lamb * ema[idx] + (1.0f - lamb) * g;
    ema[idx] = new_ema;
    grad[idx] = static_cast<GradT>(g + alpha * new_ema);
}

template <typename GradT>
inline hipError_t launch_ema_typed(
    float* ema, GradT* grad, float alpha, float lamb,
    int64_t N, hipStream_t stream
) {
    using namespace sg::gfx942::common;
    if (N == 0) return hipSuccess;
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;
    grokfast_ema_kernel<GradT><<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
        ema, grad, alpha, lamb, N);
    return hipGetLastError();
}

// ---------------------------------------------------------------------
// Fused EMA + AdamW kernel.
// ---------------------------------------------------------------------

template <typename ParamT>
__global__ void grokfast_adam_kernel(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ ema,
    const ParamT* __restrict__ grad,
    const float alpha, const float lamb,
    const float beta1, const float beta2,
    const float lr, const float wd, const float eps,
    const float bc1, const float bc2,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float g = static_cast<float>(grad[idx]);
    const float p = static_cast<float>(param[idx]);

    const float new_ema = lamb * ema[idx] + (1.0f - lamb) * g;
    ema[idx] = new_ema;
    const float g_amp = g + alpha * new_ema;

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * g_amp;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g_amp * g_amp;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    const float m_hat = m / bc1;
    const float v_hat = v / bc2;
    const float u = m_hat / (sqrtf(v_hat) + eps) + wd * p;
    param[idx] = static_cast<ParamT>(p - lr * u);
}

template <typename ParamT>
inline hipError_t launch_adam_typed(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* ema,
    const ParamT* grad,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps, float bc1, float bc2,
    int64_t N, hipStream_t stream
) {
    using namespace sg::gfx942::common;
    if (N == 0) return hipSuccess;
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;
    grokfast_adam_kernel<ParamT><<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
        param, exp_avg, exp_avg_sq, ema, grad,
        alpha, lamb, beta1, beta2, lr, wd, eps, bc1, bc2, N);
    return hipGetLastError();
}

}}} // namespace sg::gfx942::grokfast_impl

namespace sg { namespace gfx942 {

void launch_fused_grokfast_ema(
    torch::Tensor grad, torch::Tensor ema,
    float alpha, float lamb
) {
    using namespace sg::gfx942::common;
    TORCH_CHECK(grad.is_cuda() && ema.is_cuda(),
                "grokfast_ema (gfx942): tensors must be GPU");
    TORCH_CHECK(grad.is_contiguous() && ema.is_contiguous(),
                "grokfast_ema (gfx942): tensors must be contiguous");
    TORCH_CHECK(ema.scalar_type() == at::kFloat,
                "grokfast_ema (gfx942): ema must be FP32");
    TORCH_CHECK(grad.numel() == ema.numel(),
                "grokfast_ema (gfx942): numel mismatch");
    const int64_t N = grad.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    auto disp = [&](auto tag) {
        using T = decltype(tag);
        auto err = sg::gfx942::grokfast_impl::launch_ema_typed<T>(
            ema.data_ptr<float>(),
            reinterpret_cast<T*>(grad.data_ptr()),
            alpha, lamb, N, stream);
        check_hip(err, "launch_fused_grokfast_ema");
    };
    switch (grad.scalar_type()) {
        case at::kFloat:    disp(float{});           break;
        case at::kBFloat16: disp(__hip_bfloat16{});  break;
        case at::kHalf:     disp(__half{});          break;
        default:
            TORCH_CHECK(false, "grokfast_ema (gfx942): unsupported grad dtype ",
                        grad.scalar_type());
    }
}

void launch_fused_grokfast_adam(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    using namespace sg::gfx942::common;
    TORCH_CHECK(param.is_cuda() && exp_avg.is_cuda() && exp_avg_sq.is_cuda() &&
                ema.is_cuda() && grad.is_cuda(),
                "grokfast_adam (gfx942): tensors must be GPU");
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat &&
                exp_avg_sq.scalar_type() == at::kFloat &&
                ema.scalar_type() == at::kFloat,
                "grokfast_adam (gfx942): state tensors must be FP32");
    TORCH_CHECK(grad.scalar_type() == param.scalar_type(),
                "grokfast_adam (gfx942): grad and param dtypes must match");
    const int64_t N = param.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    auto disp = [&](auto tag) {
        using T = decltype(tag);
        auto err = sg::gfx942::grokfast_impl::launch_adam_typed<T>(
            reinterpret_cast<T*>(param.data_ptr()),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            ema.data_ptr<float>(),
            reinterpret_cast<const T*>(grad.data_ptr()),
            alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
            N, stream);
        check_hip(err, "launch_fused_grokfast_adam");
    };
    switch (param.scalar_type()) {
        case at::kFloat:    disp(float{});           break;
        case at::kBFloat16: disp(__hip_bfloat16{});  break;
        case at::kHalf:     disp(__half{});          break;
        default:
            TORCH_CHECK(false, "grokfast_adam (gfx942): unsupported param dtype ",
                        param.scalar_type());
    }
}

void launch_multi_tensor_grokfast_ema(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& ema_bufs,
    float alpha, float lamb
) {
    const size_t T = grads.size();
    TORCH_CHECK(ema_bufs.size() == T,
                "grokfast multi-tensor ema (gfx942): vector size mismatch");
    for (size_t i = 0; i < T; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        launch_fused_grokfast_ema(grads[i], ema_bufs[i], alpha, lamb);
    }
}

}} // namespace sg::gfx942
