// CUDA sm_90 launch glue for NeuralGrok.
// Algorithm: csrc/algorithms/neuralgrok.h
//
// Two-stage compute: psi_net forward (per-element MLP on |grad|), then
// Adam apply with the amplified gradient. The psi weights live in constant
// memory; the launcher copies host-side weights into the constant buffer
// before kernel launch.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/neuralgrok.h"
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace cuda_sm90 {

namespace prim = ::sg::cuda_sm90::primitives;
using ::sg::algorithms::neuralgrok_psi_forward;
using ::sg::algorithms::neuralgrok_apply_step;

// Compile-time hidden width specializations (H = 8, 16, 32, 64, 128).
constexpr int NG_H = 64;

template <typename ParamT, typename GradT>
__global__ void neuralgrok_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const GradT* grad,
    const float* W1, const float* b1, const float* W2, float b2,
    float alpha, float beta,
    float lr, float beta1_a, float beta2_a, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        const float ag = fabsf(static_cast<float>(grad[i]));
        const float s = neuralgrok_psi_forward<NG_H>(ag, W1, b1, W2, b2);
        neuralgrok_apply_step(param, exp_avg, exp_avg_sq, grad, s,
                              alpha, beta, lr, beta1_a, beta2_a, eps, wd,
                              bc1, bc2, i);
    }
}

void launch_neuralgrok_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const torch::Tensor& grad,
    const torch::Tensor& psi_W1,
    const torch::Tensor& psi_b1,
    const torch::Tensor& psi_W2,
    float psi_b2,
    float alpha, float beta,
    float lr, float beta1_a, float beta2_a, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "neuralgrok_step", [&] {
            neuralgrok_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                psi_W1.data_ptr<float>(),
                psi_b1.data_ptr<float>(),
                psi_W2.data_ptr<float>(),
                psi_b2,
                alpha, beta, lr, beta1_a, beta2_a, eps, wd, bc1, bc2, N);
        });
}

}} // namespace sg::cuda_sm90
