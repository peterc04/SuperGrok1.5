/*
 * SuperGrok v1.5 — C++ Operations Header
 */
#pragma once
#include <torch/extension.h>
#include <vector>

// ── CUDA kernel launchers (defined in kernels.cu) ────────────────────
#ifdef WITH_CUDA
void launch_fused_mu_metanet(
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor smart_grad, float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim);

void launch_fused_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor smart_grad, torch::Tensor mu,
    float lamb_eff, float beta1, float beta2, float lr, float wd_eff,
    float eps, float bc1, float bc2);

void launch_sam_perturb(torch::Tensor param, torch::Tensor grad, float rho_over_norm);

void launch_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness, torch::Tensor backup,
    torch::Tensor sam_grad, torch::Tensor normal_grad);
#endif

// ── High-level C++ operations (called from Python) ───────────────────

// Process all parameters in one C++ call: mu update, meta-net, gating, adam, wd
void supergrok15_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& mus,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<int64_t>& steps,
    std::vector<float>& layer_alphas,
    std::vector<float>& layer_beta1s,
    // Meta-net weights (contiguous)
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim,
    // Hyperparams
    float beta2, float lr, float wd_eff, float eps,
    float lamb, float ramp, float gate_temperature,
    float grad_clip_norm);

// SAM: perturb all parameters, return backups
std::vector<torch::Tensor> supergrok15_sam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho);

// Compute sharpness + restore params from backups
void supergrok15_sharpness_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<torch::Tensor>& backups,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads);
