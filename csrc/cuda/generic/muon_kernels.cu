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

#include "platform.h"
#include "utils.cuh"

constexpr int MUON_BLOCK_SIZE = 256;

template <typename scalar_t>
__launch_bounds__(256, 8)
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
    buf[idx] = static_cast<scalar_t>(b);  // buf is reused for NS; keep in L2
    X[idx] = static_cast<scalar_t>(b * inv_norm);
}

template <typename scalar_t>
__launch_bounds__(256, 8)
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
__launch_bounds__(256, 8)
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
//  float4 vectorized kernels (FP32 only)
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void muon_momentum_normalize_vec4_kernel(
    float4* __restrict__ buf4,
    float4* __restrict__ X4,
    const float4* __restrict__ grad4,
    float momentum, float inv_norm, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 b = stream_load4(&buf4[i]);
    float4 g = grad4[i];

    b.x = momentum * b.x + g.x;
    b.y = momentum * b.y + g.y;
    b.z = momentum * b.z + g.z;
    b.w = momentum * b.w + g.w;
    stream_store4(&buf4[i], b);

    float4 x;
    x.x = b.x * inv_norm;
    x.y = b.y * inv_norm;
    x.z = b.z * inv_norm;
    x.w = b.w * inv_norm;
    X4[i] = x;
}

__launch_bounds__(256, 8)
__global__ void muon_ns_combine_vec4_kernel(
    float4* __restrict__ X_out4,
    const float4* __restrict__ X4,
    const float4* __restrict__ AX4,
    const float4* __restrict__ AAX4,
    float a, float b, float c, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 x = X4[i];
    float4 ax = AX4[i];
    float4 aax = AAX4[i];

    float4 out;
    out.x = a * x.x + b * ax.x + c * aax.x;
    out.y = a * x.y + b * ax.y + c * aax.y;
    out.z = a * x.z + b * ax.z + c * aax.z;
    out.w = a * x.w + b * ax.w + c * aax.w;
    X_out4[i] = out;
}

__launch_bounds__(256, 8)
__global__ void muon_update_vec4_kernel(
    float4* __restrict__ param4,
    const float4* __restrict__ orth4,
    float neg_lr_scale, float decay_factor, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 p = param4[i];
    float4 o = orth4[i];

    p.x = (p.x + neg_lr_scale * o.x) * decay_factor;
    p.y = (p.y + neg_lr_scale * o.y) * decay_factor;
    p.z = (p.z + neg_lr_scale * o.z) * decay_factor;
    p.w = (p.w + neg_lr_scale * o.w) * decay_factor;
    param4[i] = p;
}

// ═══════════════════════════════════════════════════════════════════════
//  Fused ns_combine + update: X_orth stays in register
//  Eliminates one N-element global write + read (X → orth)
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void muon_ns_combine_update_fused_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ AX,
    const scalar_t* __restrict__ AAX,
    const float a, const float b, const float c,
    const float neg_lr_scale,
    const float decay_factor,
    const int numel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    // ns_combine: X_orth stays in register
    float x = static_cast<float>(X[idx]);
    float ax = static_cast<float>(AX[idx]);
    float aax = static_cast<float>(AAX[idx]);
    float orth = a * x + b * ax + c * aax;
    // update: directly apply to param
    float p = static_cast<float>(param[idx]);
    p += neg_lr_scale * orth;
    p *= decay_factor;
    param[idx] = static_cast<scalar_t>(p);
}

__launch_bounds__(256, 8)
__global__ void muon_ns_combine_update_fused_vec4_kernel(
    float4* __restrict__ param4,
    const float4* __restrict__ X4,
    const float4* __restrict__ AX4,
    const float4* __restrict__ AAX4,
    float a, float b, float c,
    float neg_lr_scale, float decay_factor, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 x = X4[i];
    float4 ax = AX4[i];
    float4 aax = AAX4[i];
    float4 p = param4[i];

    // Fused: ns_combine result stays in register, applied directly to param
    p.x = (p.x + neg_lr_scale * (a * x.x + b * ax.x + c * aax.x)) * decay_factor;
    p.y = (p.y + neg_lr_scale * (a * x.y + b * ax.y + c * aax.y)) * decay_factor;
    p.z = (p.z + neg_lr_scale * (a * x.z + b * ax.z + c * aax.z)) * decay_factor;
    p.w = (p.w + neg_lr_scale * (a * x.w + b * ax.w + c * aax.w)) * decay_factor;
    param4[i] = p;
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

    // float4 fast path: FP32 params, N divisible by 4, 16-byte aligned
    if (buf.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(buf.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + MUON_BLOCK_SIZE - 1) / MUON_BLOCK_SIZE;
        muon_momentum_normalize_vec4_kernel<<<grid4, MUON_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(buf.data_ptr<float>()),
            reinterpret_cast<float4*>(X.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            momentum, inv_norm, N4);
        return;
    }

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

    // float4 fast path: FP32 params, N divisible by 4, 16-byte aligned
    if (X_out.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(X_out.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(X.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(AX.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(AAX.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + MUON_BLOCK_SIZE - 1) / MUON_BLOCK_SIZE;
        muon_ns_combine_vec4_kernel<<<grid4, MUON_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(X_out.data_ptr<float>()),
            reinterpret_cast<const float4*>(X.data_ptr<float>()),
            reinterpret_cast<const float4*>(AX.data_ptr<float>()),
            reinterpret_cast<const float4*>(AAX.data_ptr<float>()),
            a, b, c, N4);
        return;
    }

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

    // float4 fast path: FP32 params, N divisible by 4, 16-byte aligned
    if (param.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(param.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(orth.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + MUON_BLOCK_SIZE - 1) / MUON_BLOCK_SIZE;
        muon_update_vec4_kernel<<<grid4, MUON_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<const float4*>(orth.data_ptr<float>()),
            neg_lr_scale, decay_factor, N4);
        return;
    }

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

void launch_muon_ns_combine_update_fused(
    torch::Tensor param,
    torch::Tensor X,
    torch::Tensor AX,
    torch::Tensor AAX,
    float a, float b, float c,
    float neg_lr_scale,
    float decay_factor
) {
    const int N = param.numel();
    if (N == 0) return;

    if (param.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(param.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(X.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(AX.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(AAX.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + MUON_BLOCK_SIZE - 1) / MUON_BLOCK_SIZE;
        muon_ns_combine_update_fused_vec4_kernel<<<grid4, MUON_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<const float4*>(X.data_ptr<float>()),
            reinterpret_cast<const float4*>(AX.data_ptr<float>()),
            reinterpret_cast<const float4*>(AAX.data_ptr<float>()),
            a, b, c, neg_lr_scale, decay_factor, N4);
        return;
    }

    const int grid = (N + MUON_BLOCK_SIZE - 1) / MUON_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "muon_ns_combine_update_fused", ([&] {
        muon_ns_combine_update_fused_kernel<scalar_t><<<grid, MUON_BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            AX.data_ptr<scalar_t>(),
            AAX.data_ptr<scalar_t>(),
            a, b, c, neg_lr_scale, decay_factor, N
        );
    }));
}


// ═══════════════════════════════════════════════════════════════════════
//  Fused Muon Step: momentum + normalize + Newton-Schulz + update
//
//  Combines all Muon sub-operations into a single launcher.
//  Newton-Schulz uses ATen torch::mm (cuBLAS). Element-wise ops use
//  the custom kernels above (with float4 fast path when possible).
// ═══════════════════════════════════════════════════════════════════════

void launch_muon_fused_step(
    torch::Tensor param, torch::Tensor momentum_buffer, torch::Tensor grad,
    float lr, float momentum, float weight_decay, int ns_steps,
    float a, float b, float c
) {
    // 1. Momentum update + normalize
    float norm = (momentum_buffer.mul_(momentum).add_(grad)).norm().item<float>();
    float inv_norm = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;
    auto X = momentum_buffer * inv_norm;

    // 2. Newton-Schulz iterations (for 2D weight matrices)
    if (X.dim() >= 2) {
        int M = X.size(0);
        int N_dim = X.size(1);
        auto X_2d = X.view({M, N_dim});

        float neg_lr = -lr;
        float decay = 1.0f - weight_decay * lr;

        for (int i = 0; i < ns_steps; i++) {
            auto AX = torch::mm(torch::mm(X_2d.t(), X_2d), X_2d.t()).t();
            auto AAX = torch::mm(torch::mm(X_2d.t(), torch::mm(X_2d, X_2d.t())), X_2d.t()).t();

            if (i < ns_steps - 1) {
                // Intermediate iteration: just ns_combine
                launch_muon_ns_combine(X_2d, X_2d, AX, AAX, a, b, c);
            } else {
                // Last iteration: fused ns_combine + update (saves one global mem round-trip)
                launch_muon_ns_combine_update_fused(
                    param.view({M, N_dim}), X_2d, AX, AAX, a, b, c, neg_lr, decay);
                return;  // param already updated
            }
        }
        X = X_2d.view_as(param);
    }

    // 3. Fallback for 1D params or ns_steps==0: separate update
    float neg_lr = -lr;
    float decay = 1.0f - weight_decay * lr;
    launch_muon_update(param, X, neg_lr, decay);
}
