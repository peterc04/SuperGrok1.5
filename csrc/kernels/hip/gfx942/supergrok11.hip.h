// =====================================================================
//  csrc/kernels/hip/gfx942/supergrok11.hip.h
// =====================================================================

#pragma once

#include <torch/extension.h>

namespace sg { namespace gfx942 {

void launch_sg11_mu_metanet(
    torch::Tensor mu, torch::Tensor grad,
    torch::Tensor sharpness, torch::Tensor smart_grad,
    float alpha,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim);

void launch_sg11_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor smart_grad,
    torch::Tensor mu,
    float lamb_eff, float beta1, float beta2,
    float lr, float wd_eff, float eps, float bc1, float bc2);

void launch_sg11_sam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm);

void launch_sg11_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness,
    torch::Tensor backup,
    torch::Tensor sam_grad, torch::Tensor normal_grad);

float compute_cosine_gate_fused(
    torch::Tensor smart_grad, torch::Tensor mu, float gate_temp);

}} // namespace sg::gfx942
