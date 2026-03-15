/*
 * Muon — Fused CUDA Kernels (BF16/FP16 compatible)
 *
 * Newton-Schulz orthogonalization for 2D weight matrix updates:
 *   1. muon_momentum_normalize: buf = mom*buf + grad; X = buf/||buf||
 *   2. muon_ns_combine: X = a*X + b*AX + c*AAX
 *   3. muon_update: p += neg_lr_scale*orth; p *= decay_factor
 *
 * Matrix multiplications use cuBLAS via ATen. Element-wise ops are custom kernels.
 * FP32 accumulation used internally.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int MUON_BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void muon_momentum_normalize_kernel(
    scalar_t* __restrict__ buf,
    scalar_t* __restrict__ X,
    const scalar_t* __restrict__ grad,
    const float momentum,
    const float inv_norm,
    const int numel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float b = momentum * static_cast<float>(buf[idx]) + static_cast<float>(grad[idx]);
    buf[idx] = static_cast<scalar_t>(b);
    X[idx] = static_cast<scalar_t>(b * inv_norm);
}

template <typename scalar_t>
__global__ void muon_ns_combine_kernel(
    scalar_t* __restrict__ X_out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ AX,
    const scalar_t* __restrict__ AAX,
    const float a,
    const float b,
    const float c,
    const int numel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float x = static_cast<float>(X[idx]);
    float ax = static_cast<float>(AX[idx]);
    float aax = static_cast<float>(AAX[idx]);
    X_out[idx] = static_cast<scalar_t>(a * x + b * ax + c * aax);
}

template <typename scalar_t>
__global__ void muon_update_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ orth,
    const float neg_lr_scale,
    const float decay_factor,
    const int numel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float p = static_cast<float>(param[idx]);
    float o = static_cast<float>(orth[idx]);
    p += neg_lr_scale * o;
    p *= decay_factor;
    param[idx] = static_cast<scalar_t>(p);
}

// ═══════════════════════════════════════════════════════════════════════
//  C++ Dispatch Functions
// ═══════════════════════════════════════════════════════════════════════

void launch_muon_momentum_normalize(
    torch::Tensor buf,
    torch::Tensor X,
    torch::Tensor grad,
    float momentum,
    float inv_norm
) {
    const int N = buf.numel();
    if (N == 0) return;
    const int grid = (N + MUON_BLOCK_SIZE - 1) / MUON_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        buf.scalar_type(), "muon_momentum_normalize", ([&] {
        muon_momentum_normalize_kernel<scalar_t><<<grid, MUON_BLOCK_SIZE>>>(
            buf.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            momentum, inv_norm, N
        );
    }));
}

void launch_muon_ns_combine(
    torch::Tensor X_out,
    torch::Tensor X,
    torch::Tensor AX,
    torch::Tensor AAX,
    float a,
    float b,
    float c
) {
    const int N = X_out.numel();
    if (N == 0) return;
    const int grid = (N + MUON_BLOCK_SIZE - 1) / MUON_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        X_out.scalar_type(), "muon_ns_combine", ([&] {
        muon_ns_combine_kernel<scalar_t><<<grid, MUON_BLOCK_SIZE>>>(
            X_out.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            AX.data_ptr<scalar_t>(),
            AAX.data_ptr<scalar_t>(),
            a, b, c, N
        );
    }));
}

void launch_muon_update(
    torch::Tensor param,
    torch::Tensor orth,
    float neg_lr_scale,
    float decay_factor
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + MUON_BLOCK_SIZE - 1) / MUON_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "muon_update", ([&] {
        muon_update_kernel<scalar_t><<<grid, MUON_BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            orth.data_ptr<scalar_t>(),
            neg_lr_scale, decay_factor, N
        );
    }));
}
