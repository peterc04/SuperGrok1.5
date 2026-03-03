/*
 * Lion — Fused CUDA Kernel
 *
 * Lion optimizer: sign-based update with interpolated momentum.
 * Fuses the entire optimizer step into one kernel per parameter:
 *
 *   update  = sign(beta1 * exp_avg + (1 - beta1) * grad)
 *   param  -= lr * (update + wd * param)
 *   exp_avg = beta2 * exp_avg + (1 - beta2) * grad
 *
 * No sqrt, no division — Lion is inherently simpler than Adam.
 * One thread per element, fully coalesced.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int LION_BLOCK_SIZE = 256;

// ═══════════════════════════════════════════════════════════════════════
//  Kernel: Fused Lion step (interpolation + sign + update + decay)
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void fused_lion_step_kernel(
    scalar_t* __restrict__ param,      // [N] — updated
    scalar_t* __restrict__ exp_avg,    // [N] — updated (momentum buffer)
    const scalar_t* __restrict__ grad, // [N]
    const scalar_t lr,
    const scalar_t beta1,              // For update direction (e.g. 0.9)
    const scalar_t beta2,              // For momentum tracking (e.g. 0.99)
    const scalar_t wd,                 // Weight decay
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const scalar_t g = grad[idx];
    const scalar_t ea = exp_avg[idx];
    const scalar_t p = param[idx];

    // Interpolated direction for update
    const scalar_t interp = beta1 * ea + (static_cast<scalar_t>(1) - beta1) * g;

    // Sign function: -1, 0, +1
    scalar_t s;
    if (interp > static_cast<scalar_t>(0))
        s = static_cast<scalar_t>(1);
    else if (interp < static_cast<scalar_t>(0))
        s = static_cast<scalar_t>(-1);
    else
        s = static_cast<scalar_t>(0);

    // Parameter update: p -= lr * (sign(interp) + wd * p)
    param[idx] = p - lr * (s + wd * p);

    // Momentum update (for NEXT step's interpolation)
    exp_avg[idx] = beta2 * ea + (static_cast<scalar_t>(1) - beta2) * g;
}


// ═══════════════════════════════════════════════════════════════════════
//  C++ Dispatch
// ═══════════════════════════════════════════════════════════════════════

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

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "fused_lion_step", ([&] {
        fused_lion_step_kernel<scalar_t><<<grid, LION_BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            exp_avg.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            static_cast<scalar_t>(lr),
            static_cast<scalar_t>(beta1),
            static_cast<scalar_t>(beta2),
            static_cast<scalar_t>(wd),
            N
        );
    }));
}
