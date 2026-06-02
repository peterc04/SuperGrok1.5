#ifndef GROKKING_KERNELS_SM90_LOOKSAM_SM90_CUH_
#define GROKKING_KERNELS_SM90_LOOKSAM_SM90_CUH_
// ============================================================================
// looksam_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'looksam'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_looksam.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for LookSAM.
// Algorithm: csrc/algorithms/looksam.h
//
// LookSAM has 4 distinct kernels: perturb, restore, set_direction (on SAM
// step), and apply (every step). This file declares all 4 host launchers.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/looksam.h"
// ── Autotuner-consumable launch parameters (inlined; see compile.py) ──
#ifndef SG_TUNED_BLOCK_SIZE
#define SG_TUNED_BLOCK_SIZE 256
#endif
#ifndef SG_TUNED_VEC_WIDTH
#define SG_TUNED_VEC_WIDTH 4
#endif
#ifndef SG_TUNED_UNROLL
#define SG_TUNED_UNROLL 1
#endif
#ifndef SG_TUNED_ASYNC_DEPTH
#define SG_TUNED_ASYNC_DEPTH 2
#endif
#include "csrc/backends/cuda/sm_90/primitives.cuh"
#include "grokking_optimizers/kernels/sm_90/common_sm90.cuh"

namespace sg { namespace sm90 {

namespace prim = ::sg::sm90::primitives;
using ::sg::algorithms::looksam_perturb_step;
using ::sg::algorithms::looksam_restore_step;
using ::sg::algorithms::looksam_set_direction;
using ::sg::algorithms::looksam_apply_step;

#ifndef SG_LOOKSAM_MIN_BLOCKS
#define SG_LOOKSAM_MIN_BLOCKS 4
#endif

template <typename ParamT, typename GradT>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_LOOKSAM_MIN_BLOCKS)
looksam_perturb_kernel(
    ParamT* param, ParamT* backup, const GradT* grad, float scale, int64_t N
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        SG_SANITIZE_GRAD_INPLACE(grad, i);
        looksam_perturb_step(param, backup, grad, scale, i);
    }
}

template <typename ParamT>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_LOOKSAM_MIN_BLOCKS)
looksam_restore_kernel(
    ParamT* param, const ParamT* backup, int64_t N
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        looksam_restore_step(param, backup, i);
    }
}

template <typename GradT>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_LOOKSAM_MIN_BLOCKS)
looksam_set_direction_kernel(
    float* sam_dir, const GradT* grad_sam, const GradT* grad_orig, int64_t N
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        looksam_set_direction(sam_dir, grad_sam, grad_orig, i);
    }
}

// SG_TUNED_UNROLL-parameterized scalar apply; calls canonical looksam_apply_step.
template <typename ParamT, typename GradT, int UNROLL>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_LOOKSAM_MIN_BLOCKS)
looksam_apply_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const float* sam_dir, const GradT* grad,
    float alpha, float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N
) {
    const int64_t stride = prim::grid_stride() * UNROLL;
    const int64_t base0 = prim::grid_stride_index() * UNROLL;
    for (int64_t base = base0; base < N; base += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int64_t i = base + u;
            if (i < N) {
                SG_SANITIZE_GRAD_INPLACE(grad, i);
                looksam_apply_step(param, exp_avg, exp_avg_sq, sam_dir, grad,
                                   alpha, lr, beta1, beta2, eps, wd,
                                   bc1, bc2, i);
            }
        }
    }
}

// FP32 vec4 fast path for the Adam apply: float4 traffic, canonical step 4×.
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_LOOKSAM_MIN_BLOCKS)
looksam_apply_kernel_vec4_fp32(
    float4* param4, float4* exp_avg4, float4* exp_avg_sq4,
    const float4* sam_dir4, const float4* grad4,
    float alpha, float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N4
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N4; i += stride) {
        SG_SANITIZE_GRAD4_INPLACE(grad4, i);
        float4 p  = prim::ld_f32v4(param4 + i);
        float4 m  = prim::ld_f32v4(exp_avg4 + i);
        float4 v  = prim::ld_f32v4(exp_avg_sq4 + i);
        float4 d  = prim::ldg_f32v4(sam_dir4 + i);
        float4 g  = prim::ldg_f32v4(grad4 + i);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            looksam_apply_step(&p.x, &m.x, &v.x, &d.x, &g.x,
                               alpha, lr, beta1, beta2, eps, wd, bc1, bc2, u);
        }
        prim::st_f32v4(param4 + i, p);
        prim::st_f32v4(exp_avg4 + i, m);
        prim::st_f32v4(exp_avg_sq4 + i, v);
    }
}

void launch_looksam_perturb(
    torch::Tensor& param, torch::Tensor& backup,
    const torch::Tensor& grad, float scale
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    // Clamp the grid in int64 (N can exceed 2^31) then cast the grid dim to int
    // (grid dims are int; the element index inside the kernel is int64_t).
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "looksam_perturb", [&] {
            looksam_perturb_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                backup.data_ptr<scalar_t>(),
                grad.data_ptr<scalar_t>(),
                scale, N);
            SG_LAUNCH_CHECK(stream);
        });
}

void launch_looksam_restore(
    torch::Tensor& param, const torch::Tensor& backup
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    // Clamp the grid in int64 (N can exceed 2^31) then cast the grid dim to int
    // (grid dims are int; the element index inside the kernel is int64_t).
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "looksam_restore", [&] {
            looksam_restore_kernel<scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                backup.data_ptr<scalar_t>(), N);
            SG_LAUNCH_CHECK(stream);
        });
}

void launch_looksam_set_direction(
    torch::Tensor& sam_dir,
    const torch::Tensor& grad_sam,
    const torch::Tensor& grad_orig
) {
    const int64_t N = sam_dir.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    // Clamp the grid in int64 (N can exceed 2^31) then cast the grid dim to int
    // (grid dims are int; the element index inside the kernel is int64_t).
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad_sam.scalar_type(), "looksam_set_direction", [&] {
            looksam_set_direction_kernel<scalar_t><<<grid, block, 0, stream>>>(
                sam_dir.data_ptr<float>(),
                grad_sam.data_ptr<scalar_t>(),
                grad_orig.data_ptr<scalar_t>(), N);
            SG_LAUNCH_CHECK(stream);
        });
}

void launch_looksam_apply(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const torch::Tensor& sam_dir,
    const torch::Tensor& grad,
    float alpha, float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // §6.1: keep the Adam moments (m, v) L2-resident across the SAM apply.
    prim::L2PersistScope l2(stream,
        exp_avg.data_ptr(), exp_avg.nbytes(),
        exp_avg_sq.data_ptr(), exp_avg_sq.nbytes());

    const int block = SG_TUNED_BLOCK_SIZE;
    // Clamp the grid in int64 (N can exceed 2^31) then cast the grid dim to int
    // (grid dims are int; the element index inside the kernel is int64_t).
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    const bool all_fp32 = param.scalar_type() == torch::kFloat32 &&
                          grad.scalar_type() == torch::kFloat32;

    if (SG_TUNED_VEC_WIDTH == 4 && all_fp32 &&
        prim::is_vec4_alignable(param.data_ptr(), N) &&
        prim::is_vec4_alignable(exp_avg.data_ptr(), N) &&
        prim::is_vec4_alignable(exp_avg_sq.data_ptr(), N) &&
        prim::is_vec4_alignable(sam_dir.data_ptr(), N) &&
        prim::is_vec4_alignable(grad.data_ptr(), N)) {
        const int64_t N4 = N / 4;
        const int grid4 = static_cast<int>(
            std::min<int64_t>(65535, (N4 + block - 1) / block));
        looksam_apply_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            reinterpret_cast<const float4*>(sam_dir.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            alpha, lr, beta1, beta2, eps, wd, bc1, bc2, N4);
        SG_LAUNCH_CHECK(stream);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "looksam_apply", [&] {
            looksam_apply_kernel<scalar_t, scalar_t, SG_TUNED_UNROLL>
                <<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                sam_dir.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                alpha, lr, beta1, beta2, eps, wd, bc1, bc2, N);
            SG_LAUNCH_CHECK(stream);
        });
}


// Fused direction-adjust: grad = (1 - lambda) * grad + lambda * grad_norm * (v_dir * inv_norm)
__global__ void looksam_direction_adjust_kernel(
    float* grad, const float* v_dir,
    float inv_norm, float lambda, float grad_norm, int64_t N
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        float g = grad[i];
        float d = v_dir[i] * inv_norm;
        grad[i] = (1.0f - lambda) * g + lambda * grad_norm * d;
    }
}

void launch_looksam_direction_adjust_fused(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor v_dir,
    float inv_norm, float lambda, float grad_norm
) {
    const int64_t N = grad.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    // Clamp the grid in int64 (N can exceed 2^31) then cast the grid dim to int
    // (grid dims are int; the element index inside the kernel is int64_t).
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));
    looksam_direction_adjust_kernel<<<grid, block, 0, stream>>>(
        grad.data_ptr<float>(), v_dir.data_ptr<float>(),
        inv_norm, lambda, grad_norm, N);
    SG_LAUNCH_CHECK(stream);
}

// Reduce: results[0] = ||sam_grad - grad||, results[1] = ||grad||.
void launch_looksam_norm_reduce(
    torch::Tensor grad, torch::Tensor sam_grad,
    torch::Tensor results
) {
    auto diff = (sam_grad - grad).to(torch::kFloat32);
    auto dn = diff.norm();
    auto gn = grad.to(torch::kFloat32).norm();
    results[0] = dn;
    results[1] = gn;
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_LOOKSAM_SM90_CUH_
