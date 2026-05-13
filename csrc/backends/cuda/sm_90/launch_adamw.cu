// CUDA sm_90 launch glue for AdamW.
//
// Includes:
//   csrc/algorithms/adamw.h           — per-element math
//   csrc/backends/cuda/sm_90/primitives.cuh — grid-stride, vec4, reductions
//
// Exposes:
//   sg::cuda_sm90::launch_adamw_step(...)   — multi-tensor launcher

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/adamw.h"
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace cuda_sm90 {

namespace prim = ::sg::cuda_sm90::primitives;
using ::sg::algorithms::adamw_step;
using ::sg::algorithms::adamw_step_vec4;

// =========================================================================
//  Scalar grid-stride kernel
// =========================================================================

template <typename ParamT, typename GradT>
__global__ void adamw_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const GradT* grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        adamw_step(param, exp_avg, exp_avg_sq, grad,
                   lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

// =========================================================================
//  FP32 vec4 fast path (when param, grad, both states are FP32 and 16B-aligned)
// =========================================================================

__global__ void adamw_kernel_vec4_fp32(
    float4* param4, float4* exp_avg4, float4* exp_avg_sq4,
    const float4* grad4,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N4
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N4; i += stride) {
        adamw_step_vec4(param4, exp_avg4, exp_avg_sq4, grad4,
                        lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

// =========================================================================
//  Host-side launcher (per-tensor)
// =========================================================================

void launch_adamw_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const torch::Tensor& grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);

    const bool all_fp32 = param.scalar_type() == torch::kFloat32 &&
                          grad.scalar_type() == torch::kFloat32;

    if (all_fp32 && prim::is_vec4_alignable(param.data_ptr(), N) &&
        prim::is_vec4_alignable(grad.data_ptr(), N)) {
        const int N4 = N / 4;
        const int grid4 = std::min<int>(8192, (N4 + block - 1) / block);
        adamw_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            lr, beta1, beta2, eps, wd, bc1, bc2, N4);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "adamw_step", [&] {
            adamw_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}

}} // namespace sg::cuda_sm90
