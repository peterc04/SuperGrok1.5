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
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int GF_BLOCK_SIZE = 256;

template <typename scalar_t>
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
    const float e_old = ema[idx];

    // EMA update
    const float e = alpha * e_old + (1.0f - alpha) * g;
    ema[idx] = e;

    // Gradient amplification
    grad[idx] = static_cast<scalar_t>(g + lamb * e);
}

void launch_fused_grokfast_ema(
    torch::Tensor grad,
    torch::Tensor ema,
    float alpha,
    float lamb
) {
    const int N = grad.numel();
    if (N == 0) return;
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
