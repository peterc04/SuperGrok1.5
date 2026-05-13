// CUDA sm_90 launch glue for Lion.
// Algorithm: csrc/algorithms/lion.h

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/lion.h"
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace cuda_sm90 {

namespace prim = ::sg::cuda_sm90::primitives;
using ::sg::algorithms::lion_step;
using ::sg::algorithms::lion_step_vec4;

template <typename ParamT, typename GradT>
__global__ void lion_kernel(
    ParamT* param, float* exp_avg,
    const GradT* grad,
    float lr, float beta1, float beta2, float wd, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        lion_step(param, exp_avg, grad, lr, beta1, beta2, wd, i);
    }
}

__global__ void lion_kernel_vec4_fp32(
    float4* param4, float4* exp_avg4, const float4* grad4,
    float lr, float beta1, float beta2, float wd, int N4
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N4; i += stride) {
        lion_step_vec4(param4, exp_avg4, grad4, lr, beta1, beta2, wd, i);
    }
}

void launch_lion_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    const torch::Tensor& grad,
    float lr, float beta1, float beta2, float wd
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
        lion_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            lr, beta1, beta2, wd, N4);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "lion_step", [&] {
            lion_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                lr, beta1, beta2, wd, N);
        });
}

}} // namespace sg::cuda_sm90
