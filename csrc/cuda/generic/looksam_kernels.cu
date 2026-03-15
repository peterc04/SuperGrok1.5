/*
 * LookSAM — Fused CUDA Kernels (BF16/FP16 compatible)
 *
 *   1. looksam_direction_kernel: v_dir = (sam_grad - normal_grad) / ||...||
 *   2. looksam_adjust_kernel: grad += lambda * ||grad|| * v_dir
 *   3. looksam_perturb_kernel: param += rho_over_norm * grad
 *   4. looksam_restore_kernel: param = backup
 *
 * FP32 accumulation used internally for numerical stability.
 */

#include <torch/extension.h>

#include "platform.h"

constexpr int LOOKSAM_BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void looksam_direction_kernel(
    scalar_t* __restrict__ v_dir,
    const scalar_t* __restrict__ sam_grad,
    const scalar_t* __restrict__ normal_grad,
    const float inv_norm,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float sg = static_cast<float>(sam_grad[idx]);
    float ng = static_cast<float>(normal_grad[idx]);
    v_dir[idx] = static_cast<scalar_t>((sg - ng) * inv_norm);
}

template <typename scalar_t>
__global__ void looksam_adjust_kernel(
    scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ v_dir,
    const float la_times_gnorm,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = static_cast<float>(grad[idx]);
    float v = static_cast<float>(v_dir[idx]);
    grad[idx] = static_cast<scalar_t>(g + la_times_gnorm * v);
}

template <typename scalar_t>
__global__ void looksam_perturb_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    const float rho_over_norm,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float p = static_cast<float>(param[idx]);
    float g = static_cast<float>(grad[idx]);
    param[idx] = static_cast<scalar_t>(p + rho_over_norm * g);
}

template <typename scalar_t>
__global__ void looksam_restore_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ backup,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    param[idx] = backup[idx];
}

// ═══════════════════════════════════════════════════════════════════════
//  C++ Dispatch Functions
// ═══════════════════════════════════════════════════════════════════════

void launch_looksam_direction(
    torch::Tensor v_dir,
    torch::Tensor sam_grad,
    torch::Tensor normal_grad,
    float inv_norm
) {
    const int N = v_dir.numel();
    if (N == 0) return;
    const int grid = (N + LOOKSAM_BLOCK_SIZE - 1) / LOOKSAM_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        v_dir.scalar_type(), "looksam_direction", ([&] {
        looksam_direction_kernel<scalar_t><<<grid, LOOKSAM_BLOCK_SIZE>>>(
            v_dir.data_ptr<scalar_t>(),
            sam_grad.data_ptr<scalar_t>(),
            normal_grad.data_ptr<scalar_t>(),
            inv_norm, N
        );
    }));
}

void launch_looksam_adjust(
    torch::Tensor grad,
    torch::Tensor v_dir,
    float la_times_gnorm
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + LOOKSAM_BLOCK_SIZE - 1) / LOOKSAM_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "looksam_adjust", ([&] {
        looksam_adjust_kernel<scalar_t><<<grid, LOOKSAM_BLOCK_SIZE>>>(
            grad.data_ptr<scalar_t>(),
            v_dir.data_ptr<scalar_t>(),
            la_times_gnorm, N
        );
    }));
}

void launch_looksam_perturb(
    torch::Tensor param,
    torch::Tensor grad,
    float rho_over_norm
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + LOOKSAM_BLOCK_SIZE - 1) / LOOKSAM_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "looksam_perturb", ([&] {
        looksam_perturb_kernel<scalar_t><<<grid, LOOKSAM_BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            rho_over_norm, N
        );
    }));
}

void launch_looksam_restore(
    torch::Tensor param,
    torch::Tensor backup
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + LOOKSAM_BLOCK_SIZE - 1) / LOOKSAM_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "looksam_restore", ([&] {
        looksam_restore_kernel<scalar_t><<<grid, LOOKSAM_BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            backup.data_ptr<scalar_t>(),
            N
        );
    }));
}
