#pragma once
// SuperGrok v1.1 — vendor-neutral algorithm header.
//
// Element-wise MLP gradient transformation with cosine-similarity gating.
// Same structure as v1.5 but the gating term is the cosine between grad
// and momentum (computed per-parameter), not a sigmoid of training accuracy.
//
// Pipeline per step (two cooperative sweeps):
//
//   Sweep A — meta-net forward:
//     mu = mlp_phi(grad, sharpness)
//     gate_num += grad * momentum
//     gate_den_g += grad * grad
//     gate_den_m += momentum * momentum
//
//   Sweep B — apply:
//     gate = clamp(gate_num / sqrt(gate_den_g * gate_den_m + eps), 0, 1)
//     smart_grad = grad + (1 - gate) * alpha * mu
//     AdamW step on smart_grad with trust-ratio scaling.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

// MLP forward: tanh hidden, linear output. Phi weights: W1[H,2], b1[H], W2[H], b2.
template <int H>
__device__ __forceinline__ float sg11_phi_forward(
    const float grad_val,
    const float sharp_val,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float b2
) {
    float h_acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < H; j++) {
        float h = W1[j * 2] * grad_val + W1[j * 2 + 1] * sharp_val + b1[j];
        h = tanhf(h);
        h_acc += W2[j] * h;
    }
    return h_acc + b2;
}

// Sweep A per-element: compute mu and contribute to cosine reductions.
template <typename GradT>
__device__ __forceinline__ void sg11_sweep_a_step(
    float* __restrict__ mu_out,
    const GradT* __restrict__ grad,
    const float* __restrict__ sharpness,
    const float* __restrict__ momentum,
    const float mu_val,
    const int idx,
    float& gate_num_local,
    float& gate_den_g_local,
    float& gate_den_m_local
) {
    mu_out[idx] = mu_val;
    const float g = static_cast<float>(grad[idx]);
    const float m = momentum[idx];
    gate_num_local   += g * m;
    gate_den_g_local += g * g;
    gate_den_m_local += m * m;
}

// Sweep B per-element: cosine gate + smart_grad + Adam.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void sg11_sweep_b_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const GradT* __restrict__ grad,
    const float* __restrict__ mu,
    const float gate,            // pre-reduced cosine similarity, clamped
    const float alpha,           // meta-net strength
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int idx
) {
    const float g  = static_cast<float>(grad[idx]);
    const float p  = static_cast<float>(param[idx]);
    const float u  = mu[idx];

    const float smart_grad = g + (1.0f - gate) * alpha * u;

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * smart_grad;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * smart_grad * smart_grad;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    const float update = (m * bc1) / (sqrtf(v * bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

}} // namespace sg::algorithms
