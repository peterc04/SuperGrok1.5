/*
 * Lion — Fused CUDA Kernel (BF16/FP16 compatible)
 *
 * Lion optimizer: sign-based update with interpolated momentum.
 *
 *   update  = sign(beta1 * exp_avg + (1 - beta1) * grad)
 *   param  -= lr * (update + wd * param)
 *   exp_avg = beta2 * exp_avg + (1 - beta2) * grad
 *
 * FP32 accumulation used internally. Optimizer state kept in FP32.
 */

#include <torch/extension.h>

#include "platform.h"

constexpr int LION_BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void fused_lion_step_kernel(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,       // FP32 state
    const scalar_t* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float g = static_cast<float>(grad[idx]);
    const float ea = exp_avg[idx];
    const float p = static_cast<float>(param[idx]);

    // Interpolated direction for update
    const float interp = beta1 * ea + (1.0f - beta1) * g;

    // Sign function
    float s;
    if (interp > 0.0f)
        s = 1.0f;
    else if (interp < 0.0f)
        s = -1.0f;
    else
        s = 0.0f;

    // Parameter update: p -= lr * (sign(interp) + wd * p)
    param[idx] = static_cast<scalar_t>(p - lr * (s + wd * p));

    // Momentum update (FP32 state)
    exp_avg[idx] = beta2 * ea + (1.0f - beta2) * g;
}

void launch_fused_lion_step(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor grad,
    float lr,
    float beta1,
    float beta2,
    float wd
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + LION_BLOCK_SIZE - 1) / LION_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_lion_step", ([&] {
        fused_lion_step_kernel<scalar_t><<<grid, LION_BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            grad.data_ptr<scalar_t>(),
            lr, beta1, beta2, wd, N
        );
    }));
}
