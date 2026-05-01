#pragma once
// TODO: Device-function template for moe on sm_90 (Hopper).
// Migrated from csrc/kernels/cuda/sm_90/moe_sm90.cu
// Contains __device__ __forceinline__ update step logic.

#include "csrc/common/types.h"

namespace sg { namespace device { namespace sm90 {

template <typename scalar_t>
__device__ __forceinline__ void moe_step(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    float lr, int numel
) {
    // TODO: Port from kernel implementation
}

}}} // namespace sg::device::sm90
