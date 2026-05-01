#pragma once
// NeuralGrok -- Device-function templates for gfx942 (CDNA3 / MI300X).
// Migrated from csrc/kernels/hip/gfx942/neuralgrok_gfx942.hip.cpp
//
// Adam with learned MLP gradient amplifier:
//   1. amplifier: per-element MLP(1->H->1) that modulates gradient
//   2. adam: standard Adam update with amplified gradients

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace gfx942 {

template <typename scalar_t>
__device__ __forceinline__ void neuralgrok_amplifier(
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ amplified_grad,
    const float* __restrict__ sW1,
    const float* __restrict__ sb1,
    const float* __restrict__ sW2,
    const float* __restrict__ sb2,
    const float alpha,
    const float beta,
    const int idx,
    const int H
) {
    // TODO: Port full implementation from kernel
}

template <typename scalar_t>
__device__ __forceinline__ void neuralgrok_adam_step(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const scalar_t* __restrict__ amplified_grad,
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
