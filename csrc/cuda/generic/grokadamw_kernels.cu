/*
 * GrokAdamW -- Fused CUDA Kernel
 *
 * Adam with grokking-aware gradient filtering and amplification.
 * Fuses the entire optimizer step into one kernel per parameter:
 *
 *   1. EMA gradient filter:
 *        filtered = alpha * ema + (1 - alpha) * grad
 *        ema = filtered
 *
 *   2. Gradient amplification:
 *        amplified = grad + lamb * filtered
 *
 *   3. Adam update with amplified gradient:
 *        exp_avg    = beta1 * exp_avg    + (1 - beta1) * amplified
 *        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * amplified^2
 *        bc1 = 1 - beta1^step,  bc2 = 1 - beta2^step
 *        step_size = lr / bc1
 *        denom = sqrt(exp_avg_sq / bc2) + eps
 *        param = param * (1 - lr * weight_decay) - step_size * exp_avg / denom
 *
 * Gradient clipping is applied before the kernel launch; the grad tensor
 * passed here is already clipped.
 *
 * One thread per element, fully coalesced memory access.
 * Uses float accumulation internally for sqrt and bias correction
 * to preserve numerical precision with FP16/BF16 inputs.
 */

#include <torch/extension.h>

#include "platform.h"

constexpr int GROKADAMW_BLOCK_SIZE = 256;

// ===================================================================
//  Kernel: Fused GrokAdamW step
//  (EMA filter + amplify + Adam moments + decoupled weight decay)
// ===================================================================

template <typename scalar_t>
__global__ void fused_grokadamw_step_kernel(
    scalar_t* __restrict__ param,          // [N] -- updated in-place
    float* __restrict__ exp_avg,           // [N] -- first moment, updated
    float* __restrict__ exp_avg_sq,        // [N] -- second moment, updated
    float* __restrict__ ema,               // [N] -- EMA gradient filter, updated
    const scalar_t* __restrict__ grad,     // [N] -- pre-clipped gradient
    const float alpha,                     // EMA decay for gradient filter
    const float lamb,                      // Amplification factor
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,                       // 1 - beta1^step (precomputed)
    const float bc2,                       // 1 - beta2^step (precomputed)
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Read inputs -- cast to float for accumulation precision
    const float g = static_cast<float>(grad[idx]);
    const float e = ema[idx];

    // -- 1. EMA gradient filter -----------------------------------------
    const float filtered = alpha * e + (1.0f - alpha) * g;
    ema[idx] = filtered;

    // -- 2. Gradient amplification --------------------------------------
    const float amplified = g + lamb * filtered;

    // -- 3. Adam moment updates -----------------------------------------
    const float ea_old = exp_avg[idx];
    const float easq_old = exp_avg_sq[idx];

    const float ea = beta1 * ea_old + (1.0f - beta1) * amplified;
    const float easq = beta2 * easq_old + (1.0f - beta2) * amplified * amplified;

    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    // -- 4. Bias-corrected Adam step with decoupled weight decay --------
    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;

    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}


// ===================================================================
//  Kernel: Fused GrokAdamW step — float4 vectorized (FP32 only)
// ===================================================================

__global__ void fused_grokadamw_step_vec4_kernel(
    float4* __restrict__ param4,
    float4* __restrict__ exp_avg4,
    float4* __restrict__ exp_avg_sq4,
    float4* __restrict__ ema4,
    const float4* __restrict__ grad4,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float weight_decay, float eps, float bc1, float bc2,
    int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 p = param4[i];
    float4 g = grad4[i];
    float4 e = ema4[i];
    float4 ea = exp_avg4[i];
    float4 eas = exp_avg_sq4[i];

    // EMA filter
    e.x = alpha * e.x + (1.0f - alpha) * g.x;
    e.y = alpha * e.y + (1.0f - alpha) * g.y;
    e.z = alpha * e.z + (1.0f - alpha) * g.z;
    e.w = alpha * e.w + (1.0f - alpha) * g.w;
    ema4[i] = e;

    // Amplification
    float4 amp;
    amp.x = g.x + lamb * e.x;
    amp.y = g.y + lamb * e.y;
    amp.z = g.z + lamb * e.z;
    amp.w = g.w + lamb * e.w;

    // Adam moments
    ea.x = beta1 * ea.x + (1.0f - beta1) * amp.x;
    ea.y = beta1 * ea.y + (1.0f - beta1) * amp.y;
    ea.z = beta1 * ea.z + (1.0f - beta1) * amp.z;
    ea.w = beta1 * ea.w + (1.0f - beta1) * amp.w;

    eas.x = beta2 * eas.x + (1.0f - beta2) * amp.x * amp.x;
    eas.y = beta2 * eas.y + (1.0f - beta2) * amp.y * amp.y;
    eas.z = beta2 * eas.z + (1.0f - beta2) * amp.z * amp.z;
    eas.w = beta2 * eas.w + (1.0f - beta2) * amp.w * amp.w;

    exp_avg4[i] = ea;
    exp_avg_sq4[i] = eas;

    // Step
    float step_size = lr / bc1;
    float decay = 1.0f - lr * weight_decay;
    p.x = decay * p.x - step_size * ea.x / (sqrtf(eas.x / bc2) + eps);
    p.y = decay * p.y - step_size * ea.y / (sqrtf(eas.y / bc2) + eps);
    p.z = decay * p.z - step_size * ea.z / (sqrtf(eas.z / bc2) + eps);
    p.w = decay * p.w - step_size * ea.w / (sqrtf(eas.w / bc2) + eps);
    param4[i] = p;
}


// ===================================================================
//  C++ Dispatch Function (called from ops.cpp / pybind)
// ===================================================================

void launch_fused_grokadamw_step(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor ema,
    torch::Tensor grad,
    float alpha,
    float lamb,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float bc1,
    float bc2
) {
    const int N = param.numel();
    if (N == 0) return;

    // float4 fast path: FP32 params, N divisible by 4, 16-byte aligned
    if (param.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(param.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + GROKADAMW_BLOCK_SIZE - 1) / GROKADAMW_BLOCK_SIZE;
        fused_grokadamw_step_vec4_kernel<<<grid4, GROKADAMW_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            reinterpret_cast<float4*>(ema.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2, N4);
        return;
    }

    const int grid = (N + GROKADAMW_BLOCK_SIZE - 1) / GROKADAMW_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_grokadamw_step", ([&] {
            fused_grokadamw_step_kernel<scalar_t><<<grid, GROKADAMW_BLOCK_SIZE>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                ema.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                alpha,
                lamb,
                beta1,
                beta2,
                lr,
                weight_decay,
                eps,
                bc1,
                bc2,
                N
            );
        })
    );
}
