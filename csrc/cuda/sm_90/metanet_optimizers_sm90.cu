/*
 * Metanet Optimizers — Hopper (sm_90+)
 *
 * v1.5, v1.1, and NeuralGrok optimizers with meta-net MLPs.
 * HONEST DELEGATION: Hopper metanet calls Ampere cp.async kernels.
 * The meta-net MLPs are small (32 hidden units) — FP8 overhead exceeds
 * benefit for these tiny GEMMs. The Ampere cp.async path with TF32
 * cuBLAS mode already maximizes throughput. FP8 would help for
 * hidden_dim >= 256, which is not the default configuration.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Forward declarations of Ampere launchers
void launch_fused_supergrok15_full_step_ampere(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_fused_sg11_full_step_ampere(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_fused_neuralgrok_full_step_ampere(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2);

// Hopper launchers — delegate to Ampere cp.async path

void launch_fused_supergrok15_full_step_hopper(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim
) {
    launch_fused_supergrok15_full_step_ampere(
        param, exp_avg, exp_avg_sq, mu, grad, sharpness, alpha,
        W1, b1, W2, b2, rescale, lamb_eff,
        beta1, beta2, lr, wd_eff, eps, bc1, bc2, hidden_dim);
}

void launch_fused_sg11_full_step_hopper(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim
) {
    launch_fused_sg11_full_step_ampere(
        param, exp_avg, exp_avg_sq, mu, grad, sharpness, alpha,
        W1, b1, W2, b2, rescale, lamb_eff,
        beta1, beta2, lr, wd_eff, eps, bc1, bc2, hidden_dim);
}

void launch_fused_neuralgrok_full_step_hopper(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    launch_fused_neuralgrok_full_step_ampere(
        param, exp_avg, exp_avg_sq, grad,
        W1, b1, W2, b2, alpha_amp, beta_amp, hidden_dim,
        beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}
