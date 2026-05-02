// =====================================================================
//  csrc/kernels/hip/gfx942/neuralgrok.hip.h
// =====================================================================

#pragma once

#include <torch/extension.h>

namespace sg { namespace gfx942 {

void launch_fused_neuralgrok_amplifier(
    torch::Tensor grad, torch::Tensor amplified,
    torch::Tensor amplifier_w1, torch::Tensor amplifier_b1,
    torch::Tensor amplifier_w2, torch::Tensor amplifier_b2,
    int hidden_dim, float alpha, float beta);

void launch_fused_neuralgrok_adam(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor amplified_grad,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2);

void launch_fused_neuralgrok_full_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2);

}} // namespace sg::gfx942
