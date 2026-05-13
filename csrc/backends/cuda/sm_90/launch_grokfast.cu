// CUDA sm_90 launch glue for Grokfast (fused EMA + Adam path).
// Algorithm: csrc/algorithms/grokfast.h

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/grokfast.h"
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace cuda_sm90 {

namespace prim = ::sg::cuda_sm90::primitives;
using ::sg::algorithms::grokfast_fused_step;

template <typename ParamT, typename GradT>
__global__ void grokfast_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* ema,
    const GradT* grad,
    float gf_alpha, float gf_lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        grokfast_fused_step(param, exp_avg, exp_avg_sq, ema, grad,
                            gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd,
                            bc1, bc2, i);
    }
}

void launch_grokfast_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& ema,
    const torch::Tensor& grad,
    float gf_alpha, float gf_lamb,
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
        param.scalar_type(), "grokfast_step", [&] {
            grokfast_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                ema.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}

}} // namespace sg::cuda_sm90
