/*
 * Muon — Fused CUDA Kernels
 *
 * The Muon optimizer operates on 2D weight matrices using Newton-Schulz
 * iteration for approximate matrix orthogonalization:
 *
 *   1. Momentum accumulation: buf = momentum * buf + grad
 *   2. Normalize: X = buf / ||buf||
 *   3. Newton-Schulz iteration (5 steps):
 *        A = X @ X^T
 *        X = a*X + b*(A@X) + c*(A@(A@X))
 *      where a=3.4445, b=-4.7750, c=2.0315
 *   4. Parameter update: p -= lr * scale * X / sqrt(max(rows, cols))
 *   5. Weight decay: p *= (1 - lr * wd)
 *
 * Unlike the element-wise kernels, Muon operates on matrices. The Newton-Schulz
 * iteration involves matrix multiplications (A = X @ X^T) which are best done
 * via cuBLAS. So the "kernel" here is actually a C++ function that calls
 * cuBLAS for the matmuls and custom kernels for the element-wise operations.
 *
 * Kernels provided:
 *   1. muon_momentum_normalize_kernel: buf = mom*buf + grad; X = buf/||buf||
 *   2. muon_ns_combine_kernel: X = a*X + b*AX + c*AAX (element-wise combine)
 *   3. muon_update_kernel: p -= lr*scale*X; p *= (1 - lr*wd)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int MUON_BLOCK_SIZE = 256;

// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: Momentum accumulation + normalization (element-wise)
//  buf = momentum * buf + grad
//  X = buf / ||buf||   (norm passed as scalar, precomputed)
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void muon_momentum_normalize_kernel(
    scalar_t* __restrict__ buf,         // [M*N] — updated in-place
    scalar_t* __restrict__ X,           // [M*N] — output normalized
    const scalar_t* __restrict__ grad,  // [M*N]
    const scalar_t momentum,
    const scalar_t inv_norm,            // 1.0 / (||buf_new|| + 1e-7)
    const int numel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // Momentum accumulation
    const scalar_t b = momentum * buf[idx] + grad[idx];
    buf[idx] = b;

    // Normalize for Newton-Schulz init
    X[idx] = b * inv_norm;
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: Newton-Schulz combination (element-wise)
//  X_new = a*X + b*(A@X) + c*(A@(A@X))
//
//  A@X and A@(A@X) are computed via cuBLAS before this kernel.
//  This kernel combines them element-wise.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void muon_ns_combine_kernel(
    scalar_t* __restrict__ X_out,       // [M*N] — output
    const scalar_t* __restrict__ X,     // [M*N]
    const scalar_t* __restrict__ AX,    // [M*N] — A @ X
    const scalar_t* __restrict__ AAX,   // [M*N] — A @ (A @ X)
    const scalar_t a,
    const scalar_t b,
    const scalar_t c,
    const int numel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    X_out[idx] = a * X[idx] + b * AX[idx] + c * AAX[idx];
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: Parameter update + weight decay
//  p -= lr * scale * orth
//  p *= (1 - lr * wd)
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void muon_update_kernel(
    scalar_t* __restrict__ param,       // [M*N] — updated
    const scalar_t* __restrict__ orth,  // [M*N] — orthogonalized matrix
    const scalar_t neg_lr_scale,        // -lr * 0.2 * sqrt(max(M,N)) / sqrt(max(M,N))
    const scalar_t decay_factor,        // 1 - lr * wd
    const int numel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    scalar_t p = param[idx];
    p += neg_lr_scale * orth[idx];  // add because neg_lr_scale is negative
    p *= decay_factor;
    param[idx] = p;
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

    AT_DISPATCH_FLOATING_TYPES(buf.scalar_type(), "muon_momentum_normalize", ([&] {
        muon_momentum_normalize_kernel<scalar_t><<<grid, MUON_BLOCK_SIZE>>>(
            buf.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            static_cast<scalar_t>(momentum),
            static_cast<scalar_t>(inv_norm),
            N
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

    AT_DISPATCH_FLOATING_TYPES(X_out.scalar_type(), "muon_ns_combine", ([&] {
        muon_ns_combine_kernel<scalar_t><<<grid, MUON_BLOCK_SIZE>>>(
            X_out.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            AX.data_ptr<scalar_t>(),
            AAX.data_ptr<scalar_t>(),
            static_cast<scalar_t>(a),
            static_cast<scalar_t>(b),
            static_cast<scalar_t>(c),
            N
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

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "muon_update", ([&] {
        muon_update_kernel<scalar_t><<<grid, MUON_BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            orth.data_ptr<scalar_t>(),
            static_cast<scalar_t>(neg_lr_scale),
            static_cast<scalar_t>(decay_factor),
            N
        );
    }));
}
