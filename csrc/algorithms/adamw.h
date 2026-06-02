#pragma once
// AdamW — vendor-neutral algorithm header.
//
// Reference: Loshchilov & Hutter 2017, "Decoupled Weight Decay
// Regularization" (https://arxiv.org/abs/1711.05101), Algorithm 2.
//
// Math (decoupled weight decay variant):
//   m_t = beta1 * m_{t-1} + (1 - beta1) * g
//   v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
//   m_hat = m_t / (1 - beta1^t)        [bias correction]
//   v_hat = v_t / (1 - beta2^t)        [bias correction]
//   p -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
//
// Per-element step function called from inside a grid-stride loop in the
// per-backend launch kernel. Compiles under nvcc (CUDA) and hipcc (HIP).
// Pallas/JAX implements the same math directly in launch_adamw.py.
//
// Calling convention: bc1, bc2 are passed un-inverted —
//   bc1 = 1 - beta1^t,  bc2 = 1 - beta2^t
// — matching the binding code in csrc/bindings/bindings.cpp and the Python
// reference (`bc1 = 1.0 - beta1 ** step`). The step function divides by
// them to obtain the bias-corrected moments.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

template <typename ParamT, typename GradT>
__device__ __forceinline__ void adamw_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const GradT* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,   // 1 - beta1^t (un-inverted; header divides by it)
    const float bc2,   // 1 - beta2^t (un-inverted; header divides by it)
    const int64_t idx
) {
    const float g  = static_cast<float>(grad[idx]);
    const float p  = static_cast<float>(param[idx]);
    const float m0 = exp_avg[idx];
    const float v0 = exp_avg_sq[idx];

    const float m = beta1 * m0 + (1.0f - beta1) * g;
    const float v = beta2 * v0 + (1.0f - beta2) * g * g;

    // Bias correction: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t).
    const float m_hat = m / sg_safe_bc(bc1);
    const float v_hat = v / sg_safe_bc(bc2);

    const float denom = sqrtf(v_hat) + eps;
    const float update = m_hat / denom;

    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;
    param[idx]      = static_cast<ParamT>(p - lr * (update + wd * p));
}

// FP32 vec4 fast path — process 4 elements per call.
__device__ __forceinline__ void adamw_step_vec4(
    float4* __restrict__ param4,
    float4* __restrict__ exp_avg4,
    float4* __restrict__ exp_avg_sq4,
    const float4* __restrict__ grad4,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int64_t i
) {
    float4 p  = param4[i];
    float4 m0 = exp_avg4[i];
    float4 v0 = exp_avg_sq4[i];
    float4 g  = grad4[i];

    float4 m;
    m.x = beta1 * m0.x + (1.0f - beta1) * g.x;
    m.y = beta1 * m0.y + (1.0f - beta1) * g.y;
    m.z = beta1 * m0.z + (1.0f - beta1) * g.z;
    m.w = beta1 * m0.w + (1.0f - beta1) * g.w;

    float4 v;
    v.x = beta2 * v0.x + (1.0f - beta2) * g.x * g.x;
    v.y = beta2 * v0.y + (1.0f - beta2) * g.y * g.y;
    v.z = beta2 * v0.z + (1.0f - beta2) * g.z * g.z;
    v.w = beta2 * v0.w + (1.0f - beta2) * g.w * g.w;

    // bc1, bc2 un-inverted: divide for bias correction.
    p.x -= lr * ((m.x / sg_safe_bc(bc1)) / (sqrtf(v.x / sg_safe_bc(bc2)) + eps) + wd * p.x);
    p.y -= lr * ((m.y / sg_safe_bc(bc1)) / (sqrtf(v.y / sg_safe_bc(bc2)) + eps) + wd * p.y);
    p.z -= lr * ((m.z / sg_safe_bc(bc1)) / (sqrtf(v.z / sg_safe_bc(bc2)) + eps) + wd * p.z);
    p.w -= lr * ((m.w / sg_safe_bc(bc1)) / (sqrtf(v.w / sg_safe_bc(bc2)) + eps) + wd * p.w);

    param4[i]      = p;
    exp_avg4[i]    = m;
    exp_avg_sq4[i] = v;
}

}} // namespace sg::algorithms
