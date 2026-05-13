#pragma once
// Muon — vendor-neutral algorithm header.
//
// Dual-strategy optimizer:
//   2D parameters: Newton-Schulz orthogonalized momentum
//   1D parameters: standard AdamW
//
// Per-element pieces (the heavy lifting — matrix multiplications for NS —
// lives in primitives.cuh / mma.cuh per backend):
//
//   momentum_normalize:  buf = momentum * buf + (1 - momentum) * grad;
//                        X    = buf / ||buf||_F     (Frobenius-normalized)
//   ns_combine:          Y = a*X + b*A_X + c*AA_X    (polynomial combine)
//   muon_update:         param = param * decay + neg_lr_scale * orth

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/algorithms/adamw.h"

namespace sg { namespace algorithms {

// Momentum normalize: buf = momentum * buf + (1-momentum) * grad, then X = buf * inv_norm.
template <typename GradT>
__device__ __forceinline__ void muon_momentum_normalize_step(
    float* __restrict__ buf,
    float* __restrict__ X,
    const GradT* __restrict__ grad,
    const float momentum,
    const float inv_norm,
    const int idx
) {
    const float g = static_cast<float>(grad[idx]);
    const float b = momentum * buf[idx] + (1.0f - momentum) * g;
    buf[idx] = b;
    X[idx] = b * inv_norm;
}

// Polynomial combine: Y = a*X + b*AX + c*AAX  (Newton-Schulz iteration body).
__device__ __forceinline__ void muon_ns_combine_step(
    float* __restrict__ Y,
    const float* __restrict__ X,
    const float* __restrict__ AX,
    const float* __restrict__ AAX,
    const float a,
    const float b,
    const float c,
    const int idx
) {
    Y[idx] = a * X[idx] + b * AX[idx] + c * AAX[idx];
}

// Final parameter update with trust-ratio scaling.
template <typename ParamT>
__device__ __forceinline__ void muon_update_step(
    ParamT* __restrict__ param,
    const float* __restrict__ orth,
    const float neg_lr_scale,
    const float decay_factor,
    const int idx
) {
    const float p = static_cast<float>(param[idx]);
    param[idx] = static_cast<ParamT>(p * decay_factor + neg_lr_scale * orth[idx]);
}

// 1D parameters use AdamW directly.
using sg::algorithms::adamw_step;

}} // namespace sg::algorithms
