#pragma once
// TODO: Device-function template for neuralgrok on gfx942 (CDNA3).

#include "csrc/common/types.h"

namespace sg { namespace device { namespace gfx942 {

template <typename scalar_t>
__device__ __forceinline__ void neuralgrok_step(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    float lr, int numel
) {
    // TODO: Port from kernel implementation
}

}}} // namespace sg::device::gfx942
