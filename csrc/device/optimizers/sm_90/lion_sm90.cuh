#pragma once
// Lion -- Device-function templates for sm_90 (Hopper).
// Migrated from csrc/kernels/cuda/sm_90/lion_sm90.cu
//
// Lion optimizer: sign-based update with interpolated momentum.
//   update  = sign(beta1 * exp_avg + (1 - beta1) * grad)
//   param  -= lr * (update + wd * param)
//   exp_avg = beta2 * exp_avg + (1 - beta2) * grad
//
// The __global__ launch wrappers remain in csrc/kernels/ (or csrc/fused/).

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace sm90 {

// =========================================================================
//  Fused Lion per-element step
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void lion_step(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    const scalar_t* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd,
    const int idx
) {
    const float g = static_cast<float>(grad[idx]);
    const float ea = exp_avg[idx];
    const float p = static_cast<float>(param[idx]);

    // Interpolated direction for update
    const float interp = beta1 * ea + (1.0f - beta1) * g;

    // Sign function
    const float s = (interp != 0.0f) ? copysignf(1.0f, interp) : 0.0f;

    // Parameter update: p -= lr * (sign(interp) + wd * p)
    param[idx] = static_cast<scalar_t>(p - lr * (s + wd * p));

    // Momentum update (FP32 state)
    exp_avg[idx] = beta2 * ea + (1.0f - beta2) * g;
}

// =========================================================================
//  FP32 vec4 variant
// =========================================================================

__device__ __forceinline__ void lion_step_vec4(
    float4* __restrict__ param4,
    float4* __restrict__ exp_avg4,
    const float4* __restrict__ grad4,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd,
    const int i
) {
    float4 p = param4[i];
    float4 ea = exp_avg4[i];
    float4 g = grad4[i];

    // Interpolated direction for update
    float4 interp;
    interp.x = beta1 * ea.x + (1.0f - beta1) * g.x;
    interp.y = beta1 * ea.y + (1.0f - beta1) * g.y;
    interp.z = beta1 * ea.z + (1.0f - beta1) * g.z;
    interp.w = beta1 * ea.w + (1.0f - beta1) * g.w;

    // Sign function
    const float sx = (interp.x != 0.0f) ? copysignf(1.0f, interp.x) : 0.0f;
    const float sy = (interp.y != 0.0f) ? copysignf(1.0f, interp.y) : 0.0f;
    const float sz = (interp.z != 0.0f) ? copysignf(1.0f, interp.z) : 0.0f;
    const float sw = (interp.w != 0.0f) ? copysignf(1.0f, interp.w) : 0.0f;

    // Parameter update: p -= lr * (sign(interp) + wd * p)
    p.x = p.x - lr * (sx + wd * p.x);
    p.y = p.y - lr * (sy + wd * p.y);
    p.z = p.z - lr * (sz + wd * p.z);
    p.w = p.w - lr * (sw + wd * p.w);
    param4[i] = p;

    // Momentum update (FP32 state)
    ea.x = beta2 * ea.x + (1.0f - beta2) * g.x;
    ea.y = beta2 * ea.y + (1.0f - beta2) * g.y;
    ea.z = beta2 * ea.z + (1.0f - beta2) * g.z;
    ea.w = beta2 * ea.w + (1.0f - beta2) * g.w;
    exp_avg4[i] = ea;
}

}}} // namespace sg::device::sm90
