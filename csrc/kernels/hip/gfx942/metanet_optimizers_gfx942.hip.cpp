/*
 * Metanet Optimizers — CDNA3 (MI300X)
 *
 * v1.5, v1.1, and NeuralGrok optimizers.
 * HONEST DELEGATION: CDNA3 metanet calls generic.
 * The meta-net MLPs use small matrices (W1 is [hidden_dim, 2], W2 is [1, hidden_dim])
 * with hidden_dim=32 by default. BF16 MFMA benefit is marginal for matrices
 * this small — the compute is done per-thread in shared memory, not via GEMM.
 */

#include <torch/extension.h>

// Forward declarations of generic launchers
void launch_fused_supergrok15_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_fused_sg11_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_fused_neuralgrok_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2);

void launch_fused_supergrok15_full_step_cdna3(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim
) {
    launch_fused_supergrok15_full_step(
        param, exp_avg, exp_avg_sq, mu, grad, sharpness, alpha,
        W1, b1, W2, b2, rescale, lamb_eff,
        beta1, beta2, lr, wd_eff, eps, bc1, bc2, hidden_dim);
}

void launch_fused_sg11_full_step_cdna3(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim
) {
    launch_fused_sg11_full_step(
        param, exp_avg, exp_avg_sq, mu, grad, sharpness, alpha,
        W1, b1, W2, b2, rescale, lamb_eff,
        beta1, beta2, lr, wd_eff, eps, bc1, bc2, hidden_dim);
}

void launch_fused_neuralgrok_full_step_cdna3(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    launch_fused_neuralgrok_full_step(
        param, exp_avg, exp_avg_sq, grad,
        W1, b1, W2, b2, alpha_amp, beta_amp, hidden_dim,
        beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}
