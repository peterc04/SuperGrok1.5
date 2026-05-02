// =====================================================================
//  csrc/kernels/hip/gfx942/adamw.hip.h
//
//  gfx942 AdamW launcher header.
//
//  Note: AdamW does not have its own DECLARE in csrc/bindings/. Instead,
//  the multi-tensor AdamW vector wrapper is forward-declared by
//  csrc/bindings/grokadamw.cpp::DECLARE_MT_GROKADAMW(gfx942) and exposed
//  here as launch_fused_adamw_simple in namespace sg::gfx942.
// =====================================================================

#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <vector>

namespace sg { namespace gfx942 {

void launch_fused_adamw_simple(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    std::vector<int64_t>& steps,
    float beta1, float beta2, float lr, float wd, float eps);

}} // namespace sg::gfx942
