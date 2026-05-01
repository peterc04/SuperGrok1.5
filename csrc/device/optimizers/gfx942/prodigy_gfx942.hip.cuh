#pragma once
// Prodigy -- Device-function templates for gfx942 (CDNA3 / MI300X).
// Migrated from csrc/kernels/hip/gfx942/prodigy_gfx942.hip.cpp
//
// Distance-aware, self-tuning Adam variant.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace gfx942 {

template <typename scalar_t>
__device__ __forceinline__ void prodigy_step(
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

__device__ __forceinline__ void prodigy_step_vec4(
    float4* __restrict__ param4,
    float4* __restrict__ exp_avg4,
    float4* __restrict__ exp_avg_sq4,
    float4* __restrict__ s4,
    const float4* __restrict__ grad4,
    const float d_lr,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const int i
) {
    // TODO: Port full implementation from kernel
}

}}} // namespace sg::device::gfx942
