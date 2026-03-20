/*
 * Grokfast — Fused CUDA Kernel (BF16/FP16 compatible)
 *
 * Single kernel that performs both EMA gradient accumulation and gradient
 * amplification in one pass, eliminating the Python per-parameter loop.
 *
 *   grads_ema[i] = alpha * grads_ema[i] + (1 - alpha) * grad[i]
 *   grad[i]      = grad[i] + lamb * grads_ema[i]
 *
 * FP32 accumulation used internally for numerical stability.
 */

#include <torch/extension.h>

#include "platform.h"
#include "utils.cuh"

constexpr int GF_BLOCK_SIZE = 256;

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void fused_grokfast_ema_kernel(
    scalar_t* __restrict__ grad,
    float* __restrict__ ema,
    const float alpha,
    const float lamb,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float g = static_cast<float>(grad[idx]);
    const float e_old = stream_load(&ema[idx]);

    // EMA update
    const float e = alpha * e_old + (1.0f - alpha) * g;
    stream_store(&ema[idx], e);

    // Gradient amplification
    grad[idx] = static_cast<scalar_t>(g + lamb * e);
}

// ===================================================================
//  Kernel: Fused Grokfast EMA — float4 vectorized (FP32 only)
// ===================================================================

__launch_bounds__(256, 8)
__global__ void fused_grokfast_ema_vec4_kernel(
    float4* __restrict__ grad4,
    float4* __restrict__ ema4,
    float alpha, float lamb, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 g = grad4[i];
    float4 e = stream_load4(&ema4[i]);

    // EMA update
    e.x = alpha * e.x + (1.0f - alpha) * g.x;
    e.y = alpha * e.y + (1.0f - alpha) * g.y;
    e.z = alpha * e.z + (1.0f - alpha) * g.z;
    e.w = alpha * e.w + (1.0f - alpha) * g.w;
    stream_store4(&ema4[i], e);

    // Gradient amplification
    g.x = g.x + lamb * e.x;
    g.y = g.y + lamb * e.y;
    g.z = g.z + lamb * e.z;
    g.w = g.w + lamb * e.w;
    grad4[i] = g;
}

void launch_fused_grokfast_ema(
    torch::Tensor grad,
    torch::Tensor ema,
    float alpha,
    float lamb
) {
    const int N = grad.numel();
    if (N == 0) return;

    // float4 fast path: FP32 grads, N divisible by 4, 16-byte aligned
    if (grad.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(ema.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + GF_BLOCK_SIZE - 1) / GF_BLOCK_SIZE;
        fused_grokfast_ema_vec4_kernel<<<grid4, GF_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(grad.data_ptr<float>()),
            reinterpret_cast<float4*>(ema.data_ptr<float>()),
            alpha, lamb, N4);
        return;
    }

    const int grid = (N + GF_BLOCK_SIZE - 1) / GF_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_grokfast_ema", ([&] {
        fused_grokfast_ema_kernel<scalar_t><<<grid, GF_BLOCK_SIZE>>>(
            grad.data_ptr<scalar_t>(),
            ema.data_ptr<float>(),
            alpha, lamb, N
        );
    }));
}

// ═══════════════════════════════════════════════════════════════════════
//  Kernel: Fused Grokfast EMA + Adam step (single pass)
//  Keeps amplified gradient in register — no global memory round-trip.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void fused_grokfast_adam_kernel(
    scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float* __restrict__ ema,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const float alpha,
    const float lamb,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float g = static_cast<float>(grad[idx]);
    const float e_old = stream_load(&ema[idx]);

    // EMA update
    const float e = alpha * e_old + (1.0f - alpha) * g;
    stream_store(&ema[idx], e);

    // Gradient amplification (result stays in register)
    const float amplified = g + lamb * e;

    // Adam moments
    const float ea_old = stream_load(&exp_avg[idx]);
    const float easq_old = stream_load(&exp_avg_sq[idx]);

    const float ea = beta1 * ea_old + (1.0f - beta1) * amplified;
    const float easq = beta2 * easq_old + (1.0f - beta2) * amplified * amplified;

    stream_store(&exp_avg[idx], ea);
    stream_store(&exp_avg_sq[idx], easq);

    // Adam step with fast_rsqrt_nr
    const float step_size = lr / bc1;
    const float rsqrt_v = fast_rsqrt_nr(easq / bc2);

    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea * rsqrt_v / (1.0f + eps * rsqrt_v);
    param[idx] = static_cast<scalar_t>(p);
}

// Vec4 variant
__launch_bounds__(256, 8)
__global__ void fused_grokfast_adam_vec4_kernel(
    float4* __restrict__ param4,
    float4* __restrict__ grad4,
    float4* __restrict__ ema4,
    float4* __restrict__ exp_avg4,
    float4* __restrict__ exp_avg_sq4,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float weight_decay, float eps, float bc1, float bc2,
    int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 g = grad4[i];
    float4 e = stream_load4(&ema4[i]);
    float4 p = param4[i];
    float4 ea = stream_load4(&exp_avg4[i]);
    float4 eas = stream_load4(&exp_avg_sq4[i]);

    // EMA update
    e.x = alpha * e.x + (1.0f - alpha) * g.x;
    e.y = alpha * e.y + (1.0f - alpha) * g.y;
    e.z = alpha * e.z + (1.0f - alpha) * g.z;
    e.w = alpha * e.w + (1.0f - alpha) * g.w;
    stream_store4(&ema4[i], e);

    // Amplification (register only)
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

    stream_store4(&exp_avg4[i], ea);
    stream_store4(&exp_avg_sq4[i], eas);

    // Adam step with fast_rsqrt_nr
    float step_size = lr / bc1;
    float decay = 1.0f - lr * weight_decay;

    #define ADAM_STEP_RSQRT(comp) { \
        float rv = fast_rsqrt_nr(eas.comp / bc2); \
        p.comp = decay * p.comp - step_size * ea.comp * rv / (1.0f + eps * rv); \
    }
    ADAM_STEP_RSQRT(x)
    ADAM_STEP_RSQRT(y)
    ADAM_STEP_RSQRT(z)
    ADAM_STEP_RSQRT(w)
    #undef ADAM_STEP_RSQRT

    param4[i] = p;
}

void launch_fused_grokfast_adam(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor ema,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    const int N = param.numel();
    if (N == 0) return;

    if (param.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(param.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + GF_BLOCK_SIZE - 1) / GF_BLOCK_SIZE;
        fused_grokfast_adam_vec4_kernel<<<grid4, GF_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(grad.data_ptr<float>()),
            reinterpret_cast<float4*>(ema.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2, N4);
        return;
    }

    const int grid = (N + GF_BLOCK_SIZE - 1) / GF_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_grokfast_adam", ([&] {
        fused_grokfast_adam_kernel<scalar_t><<<grid, GF_BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            ema.data_ptr<float>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2, N);
    }));
}
