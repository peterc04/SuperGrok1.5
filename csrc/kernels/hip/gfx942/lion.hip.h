// =====================================================================
//  csrc/kernels/hip/gfx942/lion.hip.h — gfx942 Lion launcher decls.
//  Forward-declared by csrc/bindings/lion.cpp DECLARE_LION(gfx942) and
//  csrc/bindings/multi_tensor.cpp DECLARE_MT(gfx942).
// =====================================================================

#pragma once

#include <torch/extension.h>
#include <vector>

namespace sg { namespace gfx942 {

void launch_fused_lion_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,
    float lr, float beta1, float beta2, float weight_decay);

// Both by-reference (lion.cpp) and by-value (multi_tensor.cpp) overloads.
void launch_multi_tensor_lion(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float weight_decay);

void launch_multi_tensor_lion(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> grads,
    float lr, float beta1, float beta2, float weight_decay);

}} // namespace sg::gfx942
