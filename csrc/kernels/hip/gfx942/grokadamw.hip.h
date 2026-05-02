// =====================================================================
//  csrc/kernels/hip/gfx942/grokadamw.hip.h
//
//  gfx942 GrokAdamW launchers. Forward-declared by:
//    - csrc/bindings/grokadamw.cpp DECLARE_GROKADAMW_LAUNCHERS(gfx942)
//      step / clip_step / step_q3
//    - csrc/bindings/grokadamw.cpp DECLARE_MT_GROKADAMW(gfx942)
//      launch_multi_tensor_grokadamw (by-ref) + launch_fused_adamw_simple
//    - csrc/bindings/multi_tensor.cpp DECLARE_MT(gfx942)
//      launch_multi_tensor_grokadamw (by-value)
// =====================================================================

#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <vector>

namespace sg { namespace gfx942 {

void launch_fused_grokadamw_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2);

void launch_fused_grokadamw_clip_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2,
    float clip_threshold);

void launch_fused_grokadamw_step_q3(
    torch::Tensor param,
    torch::Tensor exp_avg_int8,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_bf16,
    torch::Tensor ema_bf16,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2,
    unsigned global_step);

// By-reference variant (csrc/bindings/grokadamw.cpp DECLARE_MT_GROKADAMW)
void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps);

// By-value variant (csrc/bindings/multi_tensor.cpp DECLARE_MT) — single
// shared (bc1, bc2) instead of vectors.
void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> emas,
    std::vector<torch::Tensor> grads,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2);

}} // namespace sg::gfx942
