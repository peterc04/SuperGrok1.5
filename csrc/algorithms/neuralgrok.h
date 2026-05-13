#pragma once
// NeuralGrok — vendor-neutral algorithm header.
//
// Adam with a learned per-element amplifier (psi-net). The amplifier is a
// 2-layer MLP that takes |grad| as input and outputs a multiplicative scale.
//
// Two-stage compute:
//   (1) psi_net forward:  s = mlp(|g|)  (per-element scaling factor)
//   (2) apply:            g_amp = (s * alpha + beta) * g
//                         AdamW step on g_amp
//
// The MLP weights live in constant memory on CUDA / LDS on HIP. The math
// here is the per-element body; the matmul is handled by the backend.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

// 2-layer MLP forward for a single element, with weights passed inline.
// W1: [H], b1: [H], W2: [H], b2: scalar.
// Activation: ReLU on hidden layer.
template <int H>
__device__ __forceinline__ float neuralgrok_psi_forward(
    const float abs_grad,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float b2
) {
    float h_acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < H; j++) {
        float h = W1[j] * abs_grad + b1[j];
        h = (h > 0.0f) ? h : 0.0f;       // ReLU
        h_acc += W2[j] * h;
    }
    return h_acc + b2;
}

// Apply: g_amp = (s * alpha + beta) * g, then run AdamW on g_amp.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void neuralgrok_apply_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const GradT* __restrict__ grad,
    const float psi_scale,        // output of psi_net forward
    const float alpha,
    const float beta,
    const float lr,
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

    const float g_amp = (psi_scale * alpha + beta) * g;

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * g_amp;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g_amp * g_amp;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    const float update = (m * bc1) / (sqrtf(v * bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

}} // namespace sg::algorithms
