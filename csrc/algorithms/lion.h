#pragma once
// Lion — vendor-neutral algorithm header.
// Migrated from csrc/device/optimizers/sm_90/lion_sm90.cuh.
//
// Sign-based optimizer with interpolated momentum (EvoLved Sign Momentum):
//   update  = sign(beta1 * exp_avg + (1 - beta1) * grad)
//   param  -= lr * (update + wd * param)
//   exp_avg = beta2 * exp_avg + (1 - beta2) * grad
//
// Only one state tensor (momentum buffer). No second moment.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

template <typename ParamT, typename GradT>
__device__ __forceinline__ void lion_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    const GradT* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd,
    const int64_t idx
) {
    const float g  = static_cast<float>(grad[idx]);
    const float ea = exp_avg[idx];
    const float p  = static_cast<float>(param[idx]);

    const float interp = beta1 * ea + (1.0f - beta1) * g;
    const float s = (interp != 0.0f) ? copysignf(1.0f, interp) : 0.0f;

    param[idx]   = static_cast<ParamT>(p - lr * (s + wd * p));
    exp_avg[idx] = beta2 * ea + (1.0f - beta2) * g;
}

__device__ __forceinline__ void lion_step_vec4(
    float4* __restrict__ param4,
    float4* __restrict__ exp_avg4,
    const float4* __restrict__ grad4,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd,
    const int64_t i
) {
    float4 p  = param4[i];
    float4 ea = exp_avg4[i];
    float4 g  = grad4[i];

    float4 interp;
    interp.x = beta1 * ea.x + (1.0f - beta1) * g.x;
    interp.y = beta1 * ea.y + (1.0f - beta1) * g.y;
    interp.z = beta1 * ea.z + (1.0f - beta1) * g.z;
    interp.w = beta1 * ea.w + (1.0f - beta1) * g.w;

    const float sx = (interp.x != 0.0f) ? copysignf(1.0f, interp.x) : 0.0f;
    const float sy = (interp.y != 0.0f) ? copysignf(1.0f, interp.y) : 0.0f;
    const float sz = (interp.z != 0.0f) ? copysignf(1.0f, interp.z) : 0.0f;
    const float sw = (interp.w != 0.0f) ? copysignf(1.0f, interp.w) : 0.0f;

    p.x -= lr * (sx + wd * p.x);
    p.y -= lr * (sy + wd * p.y);
    p.z -= lr * (sz + wd * p.z);
    p.w -= lr * (sw + wd * p.w);
    param4[i] = p;

    ea.x = beta2 * ea.x + (1.0f - beta2) * g.x;
    ea.y = beta2 * ea.y + (1.0f - beta2) * g.y;
    ea.z = beta2 * ea.z + (1.0f - beta2) * g.z;
    ea.w = beta2 * ea.w + (1.0f - beta2) * g.w;
    exp_avg4[i] = ea;
}

}} // namespace sg::algorithms
