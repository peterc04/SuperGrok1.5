// =====================================================================
//  csrc/kernels/hip/gfx942/adamw.hip.cpp
//
//  gfx942 fused AdamW (vector-form) launcher — ATen-based.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/adamw.hip.h"

namespace sg { namespace gfx942 {

namespace {

inline void adamw_step_aten(
    torch::Tensor& param, torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq, torch::Tensor& grad,
    float beta1, float beta2,
    float lr, float wd, float eps,
    float bc1, float bc2
) {
    auto p_f32 = param.to(torch::kFloat32);
    auto g_f32 = grad.to(torch::kFloat32);

    // m = beta1 * m + (1 - beta1) * g
    exp_avg.mul_(beta1).add_(g_f32, 1.0f - beta1);
    // v = beta2 * v + (1 - beta2) * g^2
    exp_avg_sq.mul_(beta2).addcmul_(g_f32, g_f32, 1.0f - beta2);

    auto m_hat = exp_avg / bc1;
    auto v_hat = exp_avg_sq / bc2;
    auto u = m_hat / (v_hat.sqrt() + eps) + wd * p_f32;
    auto p_new = p_f32 - lr * u;
    param.copy_(p_new.to(param.scalar_type()));
}

} // anonymous

void launch_fused_adamw_simple(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    std::vector<int64_t>& steps,
    float beta1, float beta2, float lr, float wd, float eps
) {
    const size_t n = params.size();
    if (n == 0) return;
    for (size_t i = 0; i < n; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        steps[i] += 1;
        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));
        TORCH_CHECK(exp_avgs[i].scalar_type() == at::kFloat &&
                    exp_avg_sqs[i].scalar_type() == at::kFloat,
                    "fused_adamw_simple (gfx942): state tensors must be FP32");
        adamw_step_aten(params[i], exp_avgs[i], exp_avg_sqs[i], grads[i],
                        beta1, beta2, lr, wd, eps, bc1, bc2);
    }
}

}} // namespace sg::gfx942
