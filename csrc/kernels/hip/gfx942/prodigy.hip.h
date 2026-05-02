// =====================================================================
//  csrc/kernels/hip/gfx942/prodigy.hip.h
// =====================================================================

#pragma once

#include <torch/extension.h>
#include <vector>

namespace sg { namespace gfx942 {

void launch_fused_prodigy_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor s,
    torch::Tensor param_init, torch::Tensor grad,
    float lr, float d_lr,
    float beta1, float beta2,
    float weight_decay,
    float eps, float bc1, float bc2);

void launch_prodigy_dlr_reduce(
    torch::Tensor grad, torch::Tensor param,
    torch::Tensor param_init, torch::Tensor s,
    torch::Tensor numerator, torch::Tensor denominator,
    float eps);

void launch_multi_tensor_prodigy_fused_reduce_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& param_inits,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_bufs,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    torch::Tensor d_lr_buf,
    float beta1, float beta2, float lr, float wd, float eps);

}} // namespace sg::gfx942
