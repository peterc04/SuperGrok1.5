// =====================================================================
//  csrc/kernels/hip/gfx942/neuralgrok.hip.cpp — ATen-based.
//
//  NeuralGrok runs a per-element 2-layer MLP (hidden -> ReLU -> 1) over
//  each grad coordinate, then performs an Adam step on the amplified
//  gradient. The per-element MLP is implemented as a vectorized broadcast
//  over the grad tensor: matmul-friendly shape conversion in pure ATen.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/neuralgrok.hip.h"

namespace sg { namespace gfx942 {

namespace {

// Compute amplified = grad + beta * mlp(alpha * grad).
// W1, b1 :: [H], W2, b2 :: [H, 1] / [1] (per-coord MLP).
inline torch::Tensor neuralgrok_mlp(
    torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    int hidden_dim, float alpha, float beta
) {
    auto g_flat = grad.flatten().to(torch::kFloat32);   // [N]
    int64_t N = g_flat.size(0);
    // h = ReLU(g * alpha * W1[None,:] + b1[None,:])    [N, H]
    auto W1_f = W1.to(torch::kFloat32).reshape({hidden_dim});
    auto b1_f = b1.to(torch::kFloat32).reshape({hidden_dim});
    auto W2_f = W2.to(torch::kFloat32).reshape({hidden_dim});
    auto b2_f = b2.to(torch::kFloat32).reshape({1});
    (void)N;
    auto pre = g_flat.unsqueeze(1) * alpha * W1_f.unsqueeze(0) + b1_f.unsqueeze(0);
    auto hidden = torch::relu(pre);                     // [N, H]
    auto out = (hidden * W2_f.unsqueeze(0)).sum(/*dim=*/1) + b2_f;  // [N]
    auto amplified = g_flat + beta * out;
    return amplified.reshape_as(grad);
}

inline void adam_step_aten(
    torch::Tensor& param, torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq, torch::Tensor& g_amp,
    float beta1, float beta2,
    float lr, float wd, float eps,
    float bc1, float bc2
) {
    auto p_f32 = param.to(torch::kFloat32);
    exp_avg.mul_(beta1).add_(g_amp, 1.0f - beta1);
    exp_avg_sq.mul_(beta2).addcmul_(g_amp, g_amp, 1.0f - beta2);
    auto m_hat = exp_avg / bc1;
    auto v_hat = exp_avg_sq / bc2;
    auto u = m_hat / (v_hat.sqrt() + eps) + wd * p_f32;
    param.copy_((p_f32 - lr * u).to(param.scalar_type()));
}

} // anonymous

void launch_fused_neuralgrok_amplifier(
    torch::Tensor grad, torch::Tensor amplified,
    torch::Tensor amplifier_w1, torch::Tensor amplifier_b1,
    torch::Tensor amplifier_w2, torch::Tensor amplifier_b2,
    int hidden_dim, float alpha, float beta
) {
    if (grad.numel() == 0) return;
    auto out = neuralgrok_mlp(grad,
        amplifier_w1, amplifier_b1, amplifier_w2, amplifier_b2,
        hidden_dim, alpha, beta);
    amplified.copy_(out.to(amplified.scalar_type()));
}

void launch_fused_neuralgrok_adam(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor amplified_grad,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat &&
                exp_avg_sq.scalar_type() == at::kFloat,
                "neuralgrok_adam (gfx942): state tensors must be FP32");
    if (param.numel() == 0) return;
    auto g = amplified_grad.to(torch::kFloat32);
    adam_step_aten(param, exp_avg, exp_avg_sq, g,
                   beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}

void launch_fused_neuralgrok_full_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat &&
                exp_avg_sq.scalar_type() == at::kFloat,
                "neuralgrok_full_step (gfx942): state tensors must be FP32");
    if (param.numel() == 0) return;
    auto g_amp = neuralgrok_mlp(grad, W1, b1, W2, b2,
                                hidden_dim, alpha_amp, beta_amp);
    adam_step_aten(param, exp_avg, exp_avg_sq, g_amp,
                   beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}

}} // namespace sg::gfx942
