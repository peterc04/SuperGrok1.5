#pragma once
// Adam (multi-tensor) -- Device-function templates for gfx942 (CDNA3 / MI300X).
// Migrated from csrc/kernels/hip/gfx942/multi_tensor_gfx942.hip.cpp
//
// Multi-tensor fused optimizer kernels:
//   1. multi_tensor_grokadamw: batched GrokAdamW across N parameters
//   2. multi_tensor_lion: batched Lion across N parameters
//   3. multi_tensor_grokfast_ema: batched Grokfast EMA
//   4. multi_tensor_prodigy: batched Prodigy step

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace gfx942 {

template <typename scalar_t>
__device__ __forceinline__ void multi_tensor_grokadamw_element(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ ema,
    const scalar_t* __restrict__ grad,
    const float alpha,
    const float lamb,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

template <typename scalar_t>
__device__ __forceinline__ void multi_tensor_lion_element(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    const scalar_t* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

template <typename scalar_t>
__device__ __forceinline__ void multi_tensor_grokfast_ema_element(
    scalar_t* __restrict__ grad,
    float* __restrict__ ema,
    const float alpha,
    const float lamb,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

template <typename scalar_t>
__device__ __forceinline__ void multi_tensor_prodigy_element(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ s,
    const scalar_t* __restrict__ grad,
    const float d_lr,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

}}} // namespace sg::device::gfx942
