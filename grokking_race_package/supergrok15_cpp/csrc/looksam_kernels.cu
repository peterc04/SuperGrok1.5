/*
 * LookSAM — Fused CUDA Kernels
 *
 * Two kernels for the LookSAM optimizer's non-closure steps:
 *
 *   1. looksam_direction_kernel:
 *      Computes the normalized sharpness direction vector:
 *        v_dir = (sam_grad - normal_grad) / ||sam_grad - normal_grad||
 *      Two-pass: first compute L2 norm (block-level reduce), then normalize.
 *      Since we need a global norm, we compute per-param and normalize in C++.
 *
 *   2. looksam_adjust_kernel:
 *      Adjusts gradients using cached direction:
 *        grad += lambda * ||grad|| * v_dir
 *      Also needs per-param grad norm, computed in C++ before launch.
 *
 * SAM perturbation/restore reuses existing kernels from kernels.cu:
 *   - sam_perturb_kernel for parameter perturbation
 *   - sharpness_restore_kernel logic (simplified: just restore, no sharpness)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int LOOKSAM_BLOCK_SIZE = 256;

// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: Compute unnormalized direction = sam_grad - normal_grad
//            and write to v_dir buffer
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void looksam_direction_kernel(
    scalar_t* __restrict__ v_dir,            // [N] — output direction
    const scalar_t* __restrict__ sam_grad,   // [N]
    const scalar_t* __restrict__ normal_grad,// [N]
    const scalar_t inv_norm,                 // 1.0 / ||sam_grad - normal_grad|| (precomputed)
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    v_dir[idx] = (sam_grad[idx] - normal_grad[idx]) * inv_norm;
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: Adjust gradient using cached direction
//            grad += lambda * grad_norm * v_dir
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void looksam_adjust_kernel(
    scalar_t* __restrict__ grad,       // [N] — modified in-place
    const scalar_t* __restrict__ v_dir,// [N] — cached direction
    const scalar_t la_times_gnorm,     // lambda * ||grad|| (precomputed)
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    grad[idx] += la_times_gnorm * v_dir[idx];
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: SAM perturbation (same as in kernels.cu, duplicated here
//            so looksam can be self-contained)
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void looksam_perturb_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    const scalar_t rho_over_norm,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    param[idx] += rho_over_norm * grad[idx];
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 4: Restore parameters from backup
// ═══════════════════════════════════════════════════════════════════════

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

    AT_DISPATCH_FLOATING_TYPES(v_dir.scalar_type(), "looksam_direction", ([&] {
        looksam_direction_kernel<scalar_t><<<grid, LOOKSAM_BLOCK_SIZE>>>(
            v_dir.data_ptr<scalar_t>(),
            sam_grad.data_ptr<scalar_t>(),
            normal_grad.data_ptr<scalar_t>(),
            static_cast<scalar_t>(inv_norm),
            N
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

    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "looksam_adjust", ([&] {
        looksam_adjust_kernel<scalar_t><<<grid, LOOKSAM_BLOCK_SIZE>>>(
            grad.data_ptr<scalar_t>(),
            v_dir.data_ptr<scalar_t>(),
            static_cast<scalar_t>(la_times_gnorm),
            N
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

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "looksam_perturb", ([&] {
        looksam_perturb_kernel<scalar_t><<<grid, LOOKSAM_BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            static_cast<scalar_t>(rho_over_norm),
            N
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

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "looksam_restore", ([&] {
        looksam_restore_kernel<scalar_t><<<grid, LOOKSAM_BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            backup.data_ptr<scalar_t>(),
            N
        );
    }));
}
