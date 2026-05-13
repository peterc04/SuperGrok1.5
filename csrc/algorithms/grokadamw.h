#pragma once
// GrokAdamW — vendor-neutral algorithm header.
//
// AdamW with an EMA gradient filter that amplifies persistent directions.
//
// Math:
//   ema_t = alpha * ema_{t-1} + (1 - alpha) * g
//   g_amp = g + lamb * ema_t
//   m_t   = beta1 * m_{t-1} + (1 - beta1) * g_amp
//   v_t   = beta2 * v_{t-1} + (1 - beta2) * g_amp^2
//   m_hat = m_t * bc1
//   v_hat = v_t * bc2
//   p    -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

template <typename ParamT, typename GradT>
__device__ __forceinline__ void grokadamw_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ ema,
    const GradT* __restrict__ grad,
    const float alpha,    // EMA decay (e.g. 0.98)
    const float lamb,     // amplification factor (e.g. 5.0)
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

    // EMA filter
    const float ema_new = alpha * ema[idx] + (1.0f - alpha) * g;
    ema[idx] = ema_new;

    // Amplified gradient
    const float g_amp = g + lamb * ema_new;

    // Adam moments
    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * g_amp;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g_amp * g_amp;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    // Bias-corrected update with decoupled weight decay
    const float update = (m * bc1) / (sqrtf(v * bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

}} // namespace sg::algorithms
