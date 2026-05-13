// CUDA sm_90 launch glue for GrokAdamW.
// Algorithm: csrc/algorithms/grokadamw.h

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/grokadamw.h"
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace cuda_sm90 {

namespace prim = ::sg::cuda_sm90::primitives;
using ::sg::algorithms::grokadamw_step;

template <typename ParamT, typename GradT>
__global__ void grokadamw_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* ema,
    const GradT* grad,
    float alpha, float lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        grokadamw_step(param, exp_avg, exp_avg_sq, ema, grad,
                       alpha, lamb, lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

void launch_grokadamw_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& ema,
    const torch::Tensor& grad,
    float alpha, float lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "grokadamw_step", [&] {
            grokadamw_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                ema.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                alpha, lamb, lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}

}} // namespace sg::cuda_sm90
