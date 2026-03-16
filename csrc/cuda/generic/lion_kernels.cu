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
#include "utils.cuh"

constexpr int LION_BLOCK_SIZE = 256;

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ __launch_bounds__(256, 2) void fused_lion_step_kernel(
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
    const float ea = stream_load(&exp_avg[idx]);
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
    stream_store(&exp_avg[idx], beta2 * ea + (1.0f - beta2) * g);
}

// ===================================================================
//  Kernel: Fused Lion step — float4 vectorized (FP32 only)
// ===================================================================

__launch_bounds__(256, 8)
__global__ __launch_bounds__(256, 2) void fused_lion_step_vec4_kernel(
    float4* __restrict__ param4,
    float4* __restrict__ exp_avg4,
    const float4* __restrict__ grad4,
    float lr, float beta1, float beta2, float wd, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 p = param4[i];
    float4 ea = stream_load4(&exp_avg4[i]);
    float4 g = grad4[i];

    // Interpolated direction for update
    float4 interp;
    interp.x = beta1 * ea.x + (1.0f - beta1) * g.x;
    interp.y = beta1 * ea.y + (1.0f - beta1) * g.y;
    interp.z = beta1 * ea.z + (1.0f - beta1) * g.z;
    interp.w = beta1 * ea.w + (1.0f - beta1) * g.w;

    // Sign function
    float sx = (interp.x > 0.0f) ? 1.0f : ((interp.x < 0.0f) ? -1.0f : 0.0f);
    float sy = (interp.y > 0.0f) ? 1.0f : ((interp.y < 0.0f) ? -1.0f : 0.0f);
    float sz = (interp.z > 0.0f) ? 1.0f : ((interp.z < 0.0f) ? -1.0f : 0.0f);
    float sw = (interp.w > 0.0f) ? 1.0f : ((interp.w < 0.0f) ? -1.0f : 0.0f);

    // Parameter update: p -= lr * (sign(interp) + wd * p)
    p.x = p.x - lr * (sx + wd * p.x);
    p.y = p.y - lr * (sy + wd * p.y);
    p.z = p.z - lr * (sz + wd * p.z);
    p.w = p.w - lr * (sw + wd * p.w);
    param4[i] = p;

    // Momentum update (FP32 state)
    ea.x = beta2 * ea.x + (1.0f - beta2) * g.x;
    ea.y = beta2 * ea.y + (1.0f - beta2) * g.y;
    ea.z = beta2 * ea.z + (1.0f - beta2) * g.z;
    ea.w = beta2 * ea.w + (1.0f - beta2) * g.w;
    stream_store4(&exp_avg4[i], ea);
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

    // float4 fast path: FP32 params, N divisible by 4, 16-byte aligned
    if (param.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(param.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + LION_BLOCK_SIZE - 1) / LION_BLOCK_SIZE;
        fused_lion_step_vec4_kernel<<<grid4, LION_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            lr, beta1, beta2, wd, N4);
        return;
    }

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
