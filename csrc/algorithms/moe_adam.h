#pragma once
// MoE/Adam multi-tensor — vendor-neutral algorithm header.
//
// Multi-tensor batched AdamW, used for both standard parameter groups and
// Mixture-of-Experts active-set updates. The MoE variant compacts the
// active subset of expert parameters into a dense buffer, runs the same
// per-element Adam step over that buffer, then scatters results back.
//
// The per-element math is identical to adamw.h::adamw_step; this header
// re-exports it under the `moe_adam_step` name to keep the launcher glue
// symmetric across the 11 optimizers.

#include "csrc/algorithms/adamw.h"

namespace sg { namespace algorithms {

template <typename ParamT, typename GradT>
__device__ __forceinline__ void moe_adam_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const GradT* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int idx
) {
    adamw_step(param, exp_avg, exp_avg_sq, grad,
               lr, beta1, beta2, eps, wd, bc1, bc2, idx);
}

}} // namespace sg::algorithms
