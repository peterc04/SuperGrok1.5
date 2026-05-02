// =====================================================================
//  csrc/kernels/hip/gfx942/grokfast.hip.cpp — ATen-based.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/grokfast.hip.h"

namespace sg { namespace gfx942 {

void launch_fused_grokfast_ema(
    torch::Tensor grad, torch::Tensor ema,
    float alpha, float lamb
) {
    TORCH_CHECK(ema.scalar_type() == at::kFloat,
                "grokfast_ema (gfx942): ema must be FP32");
    if (grad.numel() == 0) return;
    auto g_f32 = grad.to(torch::kFloat32);
    // ema = lamb * ema + (1 - lamb) * g
    ema.mul_(lamb).add_(g_f32, 1.0f - lamb);
    // grad += alpha * ema (in-place); cast back to grad dtype.
    auto g_amp = g_f32 + alpha * ema;
    grad.copy_(g_amp.to(grad.scalar_type()));
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
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat &&
                exp_avg_sq.scalar_type() == at::kFloat &&
                ema.scalar_type() == at::kFloat,
                "grokfast_adam (gfx942): state tensors must be FP32");
    if (param.numel() == 0) return;

    auto p_f32 = param.to(torch::kFloat32);
    auto g_f32 = grad.to(torch::kFloat32);

    ema.mul_(lamb).add_(g_f32, 1.0f - lamb);
    auto g_amp = g_f32 + alpha * ema;

    exp_avg.mul_(beta1).add_(g_amp, 1.0f - beta1);
    exp_avg_sq.mul_(beta2).addcmul_(g_amp, g_amp, 1.0f - beta2);

    auto m_hat = exp_avg / bc1;
    auto v_hat = exp_avg_sq / bc2;
    auto u = m_hat / (v_hat.sqrt() + eps) + weight_decay * p_f32;
    param.copy_((p_f32 - lr * u).to(param.scalar_type()));
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
