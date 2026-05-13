#pragma once
// Prodigy — vendor-neutral algorithm header.
//
// Self-tuning Adam. Estimates its own learning rate d from the cumulative
// parameter trajectory: d_new = max(d_prev, r / |s|), where r and s are
// global reductions across all parameters:
//   r += grad * (param_init - param) * d_prev
//   s += d_prev * d_prev * grad
//
// Three operations:
//   (1) reduce: compute partial r, s sums (block-level)
//   (2) update: combine partials, update d on a single device thread
//   (3) apply:  AdamW step using d (loaded from device memory)

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

// Per-element contribution to r and s partial sums.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void prodigy_partials_step(
    const ParamT* __restrict__ param,
    const ParamT* __restrict__ param_init,
    const GradT* __restrict__ grad,
    const float d_prev,
    const int idx,
    float& r_local,
    float& s_local
) {
    const float p  = static_cast<float>(param[idx]);
    const float pi = static_cast<float>(param_init[idx]);
    const float g  = static_cast<float>(grad[idx]);

    r_local += g * (pi - p) * d_prev;
    s_local += d_prev * d_prev * g;
}

// d update — runs on a single thread once r_sum and s_sum are reduced.
__device__ __forceinline__ float prodigy_update_d(
    const float d_prev,
    const float r_sum,
    const float s_sum
) {
    const float denom = fabsf(s_sum);
    if (denom < 1e-12f) return d_prev;
    const float candidate = r_sum / denom;
    return fmaxf(d_prev, candidate);
}

// Apply: AdamW with d as the effective learning rate scale.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void prodigy_apply_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ s_track,     // trajectory accumulator
    const GradT* __restrict__ grad,
    const float d,                   // adaptive learning rate
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int idx
) {
    const float g = static_cast<float>(grad[idx]);
    const float p = static_cast<float>(param[idx]);

    const float g_scaled = d * g;
    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * g_scaled;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g_scaled * g_scaled;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;
    s_track[idx]   += d * g;

    const float update = (m * bc1) / (sqrtf(v * bc2) + eps);
    param[idx] = static_cast<ParamT>(p - d * (update + wd * p));
}

}} // namespace sg::algorithms
