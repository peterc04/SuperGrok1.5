// =====================================================================
//  csrc/kernels/hip/gfx942/lion.hip.cpp
//
//  gfx942 Lion launchers — ATen-based reference implementation.
//  This file is compiled by the host compiler (no `__global__` / kernel
//  launch syntax). The math is identical to the sm_90 path but the
//  inner loop is written as ATen tensor ops instead of raw HIP kernels.
//
//  Lion update:
//     interp = beta1 * exp_avg + (1 - beta1) * grad
//     param -= lr * (sign(interp) + wd * param)
//     exp_avg = beta2 * exp_avg + (1 - beta2) * grad
//
//  All inputs must be on the same CUDA (HIP) device; exp_avg must be
//  FP32. param/grad accept {FP32, BF16, FP16}.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/lion.hip.h"

namespace sg { namespace gfx942 {

namespace {

inline void lion_step_aten(
    torch::Tensor& param, torch::Tensor& exp_avg, torch::Tensor& grad,
    float lr, float beta1, float beta2, float wd
) {
    // All math in FP32; cast back to param dtype at the end.
    auto p_f32 = param.to(torch::kFloat32);
    auto g_f32 = grad.to(torch::kFloat32);
    // exp_avg is FP32 by precondition.

    // interp = beta1 * exp_avg + (1 - beta1) * grad
    auto interp = exp_avg * beta1 + g_f32 * (1.0f - beta1);
    auto sign = torch::sign(interp);

    // param -= lr * (sign + wd * param)
    auto p_new = p_f32 - lr * (sign + wd * p_f32);
    param.copy_(p_new.to(param.scalar_type()));

    // exp_avg = beta2 * exp_avg + (1 - beta2) * grad
    exp_avg.mul_(beta2).add_(g_f32, 1.0f - beta2);
}

} // anonymous

void launch_fused_lion_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,
    float lr, float beta1, float beta2, float weight_decay
) {
    TORCH_CHECK(param.is_cuda() && exp_avg.is_cuda() && grad.is_cuda(),
                "lion (gfx942): tensors must be on a HIP device");
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat,
                "lion (gfx942): exp_avg must be FP32");
    TORCH_CHECK(param.numel() == grad.numel() && param.numel() == exp_avg.numel(),
                "lion (gfx942): numel mismatch");
    if (param.numel() == 0) return;
    lion_step_aten(param, exp_avg, grad, lr, beta1, beta2, weight_decay);
}

namespace {
void launch_multi_tensor_lion_impl(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& exp_avgs,
    const std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float wd
) {
    const size_t T = params.size();
    TORCH_CHECK(exp_avgs.size() == T && grads.size() == T,
                "lion multi-tensor (gfx942): vector size mismatch");
    for (size_t i = 0; i < T; ++i) {
        if (!params[i].defined() || params[i].numel() == 0) continue;
        torch::Tensor p = params[i], ea = exp_avgs[i], g = grads[i];
        launch_fused_lion_step(p, ea, g, lr, beta1, beta2, wd);
    }
}
} // anonymous

void launch_multi_tensor_lion(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float weight_decay
) {
    launch_multi_tensor_lion_impl(params, exp_avgs, grads,
                                  lr, beta1, beta2, weight_decay);
}

void launch_multi_tensor_lion(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> grads,
    float lr, float beta1, float beta2, float weight_decay
) {
    launch_multi_tensor_lion_impl(params, exp_avgs, grads,
                                  lr, beta1, beta2, weight_decay);
}

}} // namespace sg::gfx942
