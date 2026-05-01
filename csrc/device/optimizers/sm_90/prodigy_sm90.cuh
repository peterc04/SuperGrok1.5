#pragma once
// Prodigy -- Device-function templates for sm_90 (Hopper).
// Migrated from csrc/kernels/cuda/sm_90/prodigy_sm90.cu
//
// Distance-aware, self-tuning Adam variant:
//   s[i] = beta2 * s[i] + (1-beta2) * d_lr^2 * grad[i]^2
//   exp_avg[i] = beta1 * exp_avg[i] + (1-beta1) * d_lr * grad[i]
//   exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1-beta2) * d_lr^2 * grad[i]^2
//   param[i] *= (1 - lr * d_lr * weight_decay)
//   param[i] -= lr * exp_avg[i] / (bc1 * (sqrt(exp_avg_sq/bc2) + d_lr*eps))

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace sm90 {

// =========================================================================
//  Fused Prodigy per-element step
// =========================================================================

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
    // Core logic:
    //   float g = static_cast<float>(grad[idx]);
    //   float d_lr_g = d_lr * g;
    //   float d_lr_g_sq = d_lr * d_lr * g * g;
    //   s[idx] = beta2 * s[idx] + (1.0f - beta2) * d_lr_g_sq;
    //   float ea = beta1 * exp_avg[idx] + (1.0f - beta1) * d_lr_g;
    //   float easq = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * d_lr_g_sq;
    //   exp_avg[idx] = ea; exp_avg_sq[idx] = easq;
    //   float rsqrt_v = rsqrtf(easq / bc2 + 1e-30f);
    //   float p = static_cast<float>(param[idx]);
    //   p *= (1.0f - lr * d_lr * weight_decay);
    //   p -= lr * ea * rsqrt_v / (bc1 * (1.0f + d_lr * eps * rsqrt_v));
    //   param[idx] = static_cast<scalar_t>(p);
}

// =========================================================================
//  FP32 vec4 variant
// =========================================================================

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

}}} // namespace sg::device::sm90
