#pragma once
// Grokfast — vendor-neutral algorithm header.
//
// Simplest grokking-aware AdamW: EMA filter + amplification, then standard
// Adam. Two operational modes:
//
//   (A) ema-only:   ema = alpha * ema + (1-alpha) * g;
//                   grad_out = g + lamb * ema       (consumed by downstream Adam)
//
//   (B) fused:      same EMA update, then immediately run Adam on g_amp.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

// EMA-only path: update EMA and write amplified gradient back into grad_out.
template <typename GradT>
__device__ __forceinline__ void grokfast_ema_step(
    float* __restrict__ ema,
    GradT* __restrict__ grad_out,
    const GradT* __restrict__ grad_in,
    const float alpha,
    const float lamb,
    const int idx
) {
    const float g = static_cast<float>(grad_in[idx]);
    const float e = alpha * ema[idx] + (1.0f - alpha) * g;
    ema[idx] = e;
    grad_out[idx] = static_cast<GradT>(g + lamb * e);
}

// Fully fused: EMA + amplification + AdamW in a single per-element step.
// Keeps the amplified gradient in registers from compute through Adam.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void grokfast_fused_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ ema,
    const GradT* __restrict__ grad,
    const float grokfast_alpha,
    const float grokfast_lamb,
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

    const float e_new = grokfast_alpha * ema[idx] + (1.0f - grokfast_alpha) * g;
    ema[idx] = e_new;
    const float g_amp = g + grokfast_lamb * e_new;

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * g_amp;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g_amp * g_amp;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    const float update = (m * bc1) / (sqrtf(v * bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

}} // namespace sg::algorithms
