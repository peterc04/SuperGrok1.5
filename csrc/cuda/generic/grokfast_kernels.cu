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
__global__ __launch_bounds__(256, 2) void fused_grokfast_ema_kernel(
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
__global__ __launch_bounds__(256, 2) void fused_grokfast_ema_vec4_kernel(
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
