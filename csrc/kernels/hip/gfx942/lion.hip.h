// =====================================================================
//  csrc/kernels/hip/gfx942/lion.hip.h
//
//  gfx942 (CDNA3 / MI300X) Lion optimizer kernel + launcher header.
//
//  This is a HIP-native port of the sm_90 Lion launcher. The math is
//  identical to sm_90/lion.cuh; differences are HIP plumbing only:
//    - wave size is 64 (CDNA3) but irrelevant for elementwise Lion
//    - FP8 dtypes (e4m3, e5m2) are NOT supported on gfx942 in this
//      port; the dtype matrix is reduced to {FP32, BF16, FP16}^3
//    - hipStream_t in place of cudaStream_t (aliased by PyTorch HIP)
//    - no PTX (uses portable C++ math; LDG / stream_load are no-ops)
//
//  Public-facing torch::Tensor entry points in namespace sg::gfx942
//  (forward-declared by csrc/bindings/lion.cpp DECLARE_LION(gfx942)
//   and csrc/bindings/multi_tensor.cpp DECLARE_MT(gfx942)).
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"

#include <torch/extension.h>
#include <cstdint>
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
