/*
 * Grokfast — Fused CUDA Kernel
 *
 * Single kernel that performs both EMA gradient accumulation and gradient
 * amplification in one pass, eliminating the Python per-parameter loop.
 *
 *   grads_ema[i] = alpha * grads_ema[i] + (1 - alpha) * grad[i]
 *   grad[i]      = grad[i] + lamb * grads_ema[i]
 *
 * One thread per element, fully coalesced memory access, no shared memory.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int GF_BLOCK_SIZE = 256;

// ═══════════════════════════════════════════════════════════════════════
//  Kernel: Fused EMA update + gradient amplification
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void fused_grokfast_ema_kernel(
    scalar_t* __restrict__ grad,       // [N] — modified in-place (amplified)
    scalar_t* __restrict__ ema,        // [N] — updated in-place
    const scalar_t alpha,              // EMA decay (e.g. 0.98)
    const scalar_t lamb,               // Amplification factor (e.g. 2.0)
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const scalar_t g = grad[idx];

    // EMA update: ema = alpha * ema + (1 - alpha) * grad
    const scalar_t e = alpha * ema[idx] + (static_cast<scalar_t>(1) - alpha) * g;
    ema[idx] = e;

    // Gradient amplification: grad = grad + lamb * ema
    grad[idx] = g + lamb * e;
}


// ═══════════════════════════════════════════════════════════════════════
//  C++ Dispatch
// ═══════════════════════════════════════════════════════════════════════

void launch_fused_grokfast_ema(
    torch::Tensor grad,
    torch::Tensor ema,
    float alpha,
    float lamb
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + GF_BLOCK_SIZE - 1) / GF_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "fused_grokfast_ema", ([&] {
        fused_grokfast_ema_kernel<scalar_t><<<grid, GF_BLOCK_SIZE>>>(
            grad.data_ptr<scalar_t>(),
            ema.data_ptr<scalar_t>(),
            static_cast<scalar_t>(alpha),
            static_cast<scalar_t>(lamb),
            N
        );
    }));
}
