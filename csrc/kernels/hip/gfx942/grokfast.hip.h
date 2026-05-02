// =====================================================================
//  csrc/kernels/hip/gfx942/grokfast.hip.h
// =====================================================================

#pragma once

#include <torch/extension.h>
#include <vector>

namespace sg { namespace gfx942 {

void launch_fused_grokfast_ema(
    torch::Tensor grad, torch::Tensor ema,
    float alpha, float lamb);

void launch_fused_grokfast_adam(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2);

void launch_multi_tensor_grokfast_ema(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& ema_bufs,
    float alpha, float lamb);

}} // namespace sg::gfx942
