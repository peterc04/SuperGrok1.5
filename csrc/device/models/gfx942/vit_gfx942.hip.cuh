#pragma once
// TODO: Fused forward+backward device template for vit on gfx942.

#include "csrc/common/types.h"

namespace sg { namespace device { namespace gfx942 {

template <typename scalar_t>
__device__ __forceinline__ void vit_forward(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size, int seq_len
) {
    // TODO: Implement fused forward pass
}

template <typename scalar_t>
__device__ __forceinline__ void vit_backward(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    int batch_size, int seq_len
) {
    // TODO: Implement fused backward pass
}

}}} // namespace sg::device::gfx942
