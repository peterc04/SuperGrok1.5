#pragma once
// Grokfast -- Device-function templates for gfx942 (CDNA3 / MI300X).
// Migrated from csrc/kernels/hip/gfx942/grokfast_gfx942.hip.cpp
//
// EMA gradient accumulation + amplification:
//   ema = alpha * ema + (1 - alpha) * grad
//   grad = grad + lamb * ema

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace gfx942 {

template <typename scalar_t>
__device__ __forceinline__ void grokfast_ema_step(
    scalar_t* __restrict__ grad,
    float* __restrict__ ema,
    const float alpha,
    const float lamb,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

__device__ __forceinline__ void grokfast_ema_step_vec4(
    float4* __restrict__ grad4,
    float4* __restrict__ ema4,
    const float alpha,
    const float lamb,
    const int i
) {
    // TODO: Port full implementation from kernel
}

}}} // namespace sg::device::gfx942
