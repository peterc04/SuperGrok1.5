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
#include <type_traits>

#include "platform.h"

constexpr int LOOKSAM_BLOCK_SIZE = 256;

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ __launch_bounds__(256, 2) void looksam_direction_kernel(
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
    float result = (sg - ng) * inv_norm;
    // v_dir is optimizer state (direction EMA) — use non-temporal store
    // to avoid polluting L2 cache
    if constexpr (std::is_same_v<scalar_t, float>) {
        stream_store(reinterpret_cast<float*>(&v_dir[idx]), result);
    } else {
        v_dir[idx] = static_cast<scalar_t>(result);
    }
}

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ __launch_bounds__(256, 2) void looksam_adjust_kernel(
    scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ v_dir,
    const float la_times_gnorm,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = static_cast<float>(grad[idx]);
    // v_dir is optimizer state — non-temporal load (consumed once)
    float v;
    if constexpr (std::is_same_v<scalar_t, float>) {
        v = stream_load(reinterpret_cast<const float*>(&v_dir[idx]));
    } else {
        v = static_cast<float>(v_dir[idx]);
    }
    grad[idx] = static_cast<scalar_t>(g + la_times_gnorm * v);
}

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ __launch_bounds__(256, 2) void looksam_perturb_kernel(
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
__launch_bounds__(256, 8)
__global__ __launch_bounds__(256, 2) void looksam_restore_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ backup,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    param[idx] = backup[idx];
}

// ═══════════════════════════════════════════════════════════════════════
//  float4 vectorized kernels (FP32 only)
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ __launch_bounds__(256, 2) void looksam_direction_vec4_kernel(
    float4* __restrict__ v_dir4,
    const float4* __restrict__ sam_grad4,
    const float4* __restrict__ normal_grad4,
    float inv_norm, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 sg = sam_grad4[i];
    float4 ng = normal_grad4[i];

    float4 out;
    out.x = (sg.x - ng.x) * inv_norm;
    out.y = (sg.y - ng.y) * inv_norm;
    out.z = (sg.z - ng.z) * inv_norm;
    out.w = (sg.w - ng.w) * inv_norm;
    // v_dir is optimizer state — non-temporal store to avoid L2 pollution
    stream_store4(&v_dir4[i], out);
}

__launch_bounds__(256, 8)
__global__ __launch_bounds__(256, 2) void looksam_adjust_vec4_kernel(
    float4* __restrict__ grad4,
    const float4* __restrict__ v_dir4,
    float la_times_gnorm, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 g = grad4[i];
    // v_dir is optimizer state — non-temporal load (consumed once)
    float4 v = stream_load4(&v_dir4[i]);

    g.x = g.x + la_times_gnorm * v.x;
    g.y = g.y + la_times_gnorm * v.y;
    g.z = g.z + la_times_gnorm * v.z;
    g.w = g.w + la_times_gnorm * v.w;
    grad4[i] = g;
}

__launch_bounds__(256, 8)
__global__ __launch_bounds__(256, 2) void looksam_perturb_vec4_kernel(
    float4* __restrict__ param4,
    const float4* __restrict__ grad4,
    float rho_over_norm, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 p = param4[i];
    float4 g = grad4[i];

    p.x = p.x + rho_over_norm * g.x;
    p.y = p.y + rho_over_norm * g.y;
    p.z = p.z + rho_over_norm * g.z;
    p.w = p.w + rho_over_norm * g.w;
    param4[i] = p;
}

__launch_bounds__(256, 8)
__global__ __launch_bounds__(256, 2) void looksam_restore_vec4_kernel(
    float4* __restrict__ param4,
    const float4* __restrict__ backup4,
    int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    param4[i] = backup4[i];
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

    // float4 fast path: FP32 params, N divisible by 4, 16-byte aligned
    if (v_dir.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(v_dir.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(sam_grad.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(normal_grad.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + LOOKSAM_BLOCK_SIZE - 1) / LOOKSAM_BLOCK_SIZE;
        looksam_direction_vec4_kernel<<<grid4, LOOKSAM_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(v_dir.data_ptr<float>()),
            reinterpret_cast<const float4*>(sam_grad.data_ptr<float>()),
            reinterpret_cast<const float4*>(normal_grad.data_ptr<float>()),
            inv_norm, N4);
        return;
    }

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

    // float4 fast path: FP32 params, N divisible by 4, 16-byte aligned
    if (grad.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(v_dir.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + LOOKSAM_BLOCK_SIZE - 1) / LOOKSAM_BLOCK_SIZE;
        looksam_adjust_vec4_kernel<<<grid4, LOOKSAM_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(grad.data_ptr<float>()),
            reinterpret_cast<const float4*>(v_dir.data_ptr<float>()),
            la_times_gnorm, N4);
        return;
    }

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

    // float4 fast path: FP32 params, N divisible by 4, 16-byte aligned
    if (param.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(param.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + LOOKSAM_BLOCK_SIZE - 1) / LOOKSAM_BLOCK_SIZE;
        looksam_perturb_vec4_kernel<<<grid4, LOOKSAM_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            rho_over_norm, N4);
        return;
    }

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

    // float4 fast path: FP32 params, N divisible by 4, 16-byte aligned
    if (param.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(param.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(backup.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + LOOKSAM_BLOCK_SIZE - 1) / LOOKSAM_BLOCK_SIZE;
        looksam_restore_vec4_kernel<<<grid4, LOOKSAM_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<const float4*>(backup.data_ptr<float>()),
            N4);
        return;
    }

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
