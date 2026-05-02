// =====================================================================
//  csrc/kernels/hip/gfx942/supergrok15.hip.h
// =====================================================================

#pragma once

#include <torch/extension.h>

namespace sg { namespace gfx942 {

void launch_fused_supergrok15_full_step(
    torch::Tensor param,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad,
    torch::Tensor sharpness, float alpha,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2,
    float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_sam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm);

void launch_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness,
    torch::Tensor backup,
    torch::Tensor sam_grad, torch::Tensor normal_grad);

}} // namespace sg::gfx942
