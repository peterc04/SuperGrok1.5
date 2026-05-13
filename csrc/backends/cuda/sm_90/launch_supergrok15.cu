// CUDA sm_90 launch glue for SuperGrok v1.5.
// Algorithm: csrc/algorithms/supergrok15.h
//
// Same two-sweep pattern as v1.1, but the gate is a global sigmoid of
// training accuracy (passed in as gate_global), and per-coord alpha is
// computed from mu via sg15_alpha_per_coord.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/supergrok15.h"
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace cuda_sm90 {

namespace prim = ::sg::cuda_sm90::primitives;
using ::sg::algorithms::sg15_phi_forward;
using ::sg::algorithms::sg15_sweep_a_step;
using ::sg::algorithms::sg15_sweep_b_step;

constexpr int SG15_H = 64;

template <typename GradT>
__global__ void sg15_sweep_a_kernel(
    float* mu_out, const GradT* grad, const float* sharpness,
    const float* W1, const float* b1, const float* W2, float b2,
    float* sharp_partial, int N
) {
    float sl = 0.0f;
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        const float g = static_cast<float>(grad[i]);
        const float s = sharpness[i];
        const float mu_val = sg15_phi_forward<SG15_H>(g, s, W1, b1, W2, b2);
        sg15_sweep_a_step(mu_out, grad, mu_val, i, sl);
    }
    float r = prim::block_reduce_sum_f32(sl);
    if (threadIdx.x == 0) atomicAdd(sharp_partial, r);
}

template <typename ParamT, typename GradT>
__global__ void sg15_sweep_b_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const GradT* grad, const float* mu,
    float gate_global, float alpha_base, float alpha_max,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        sg15_sweep_b_step(param, exp_avg, exp_avg_sq, grad, mu,
                          gate_global, alpha_base, alpha_max,
                          lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

void launch_supergrok15_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& mu_buf,
    const torch::Tensor& grad,
    const torch::Tensor& sharpness,
    const torch::Tensor& phi_W1,
    const torch::Tensor& phi_b1,
    const torch::Tensor& phi_W2,
    float phi_b2,
    torch::Tensor& sharp_partial,
    float gate_global,
    float alpha_base, float alpha_max,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);

    sharp_partial.zero_();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "sg15_sweep_a", [&] {
            sg15_sweep_a_kernel<scalar_t><<<grid, block, 0, stream>>>(
                mu_buf.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<float>(),
                phi_W1.data_ptr<float>(),
                phi_b1.data_ptr<float>(),
                phi_W2.data_ptr<float>(),
                phi_b2,
                sharp_partial.data_ptr<float>(), N);
        });

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg15_sweep_b", [&] {
            sg15_sweep_b_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                mu_buf.data_ptr<float>(),
                gate_global, alpha_base, alpha_max,
                lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}

}} // namespace sg::cuda_sm90
