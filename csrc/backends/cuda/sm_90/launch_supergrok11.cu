// CUDA sm_90 launch glue for SuperGrok v1.1.
// Algorithm: csrc/algorithms/supergrok11.h
//
// Two-sweep cooperative pattern:
//   Sweep A: meta-net forward + cosine gate reduction
//   Sweep B: smart_grad mixing + Adam apply (uses gate from sweep A)
//
// The meta-net hidden width H is fixed at compile time. Defaults to 64.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/supergrok11.h"
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace cuda_sm90 {

namespace prim = ::sg::cuda_sm90::primitives;
using ::sg::algorithms::sg11_phi_forward;
using ::sg::algorithms::sg11_sweep_a_step;
using ::sg::algorithms::sg11_sweep_b_step;

constexpr int SG11_H = 64;

template <typename GradT>
__global__ void sg11_sweep_a_kernel(
    float* mu_out, const GradT* grad, const float* sharpness, const float* momentum,
    const float* W1, const float* b1, const float* W2, float b2,
    float* gate_num_p, float* gate_den_g_p, float* gate_den_m_p,
    int N
) {
    float gn = 0.0f, gdg = 0.0f, gdm = 0.0f;
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        const float g = static_cast<float>(grad[i]);
        const float s = sharpness[i];
        const float mu_val = sg11_phi_forward<SG11_H>(g, s, W1, b1, W2, b2);
        sg11_sweep_a_step(mu_out, grad, sharpness, momentum, mu_val, i,
                          gn, gdg, gdm);
    }
    float r1 = prim::block_reduce_sum_f32(gn);
    float r2 = prim::block_reduce_sum_f32(gdg);
    float r3 = prim::block_reduce_sum_f32(gdm);
    if (threadIdx.x == 0) {
        atomicAdd(gate_num_p, r1);
        atomicAdd(gate_den_g_p, r2);
        atomicAdd(gate_den_m_p, r3);
    }
}

template <typename ParamT, typename GradT>
__global__ void sg11_sweep_b_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const GradT* grad, const float* mu, float gate,
    float alpha, float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        sg11_sweep_b_step(param, exp_avg, exp_avg_sq, grad, mu, gate,
                          alpha, lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

void launch_supergrok11_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& mu_buf,
    const torch::Tensor& grad,
    const torch::Tensor& sharpness,
    const torch::Tensor& momentum,
    const torch::Tensor& phi_W1,
    const torch::Tensor& phi_b1,
    const torch::Tensor& phi_W2,
    float phi_b2,
    torch::Tensor& gate_partials,  // [3] {num, den_g, den_m}
    float alpha,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);

    gate_partials.zero_();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "sg11_sweep_a", [&] {
            sg11_sweep_a_kernel<scalar_t><<<grid, block, 0, stream>>>(
                mu_buf.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<float>(),
                momentum.data_ptr<float>(),
                phi_W1.data_ptr<float>(),
                phi_b1.data_ptr<float>(),
                phi_W2.data_ptr<float>(),
                phi_b2,
                gate_partials.data_ptr<float>(),
                gate_partials.data_ptr<float>() + 1,
                gate_partials.data_ptr<float>() + 2,
                N);
        });

    // Host-side reduction of the 3 scalars and gate clamp.
    auto partials_cpu = gate_partials.to(torch::kCPU);
    float gn  = partials_cpu[0].item<float>();
    float gdg = partials_cpu[1].item<float>();
    float gdm = partials_cpu[2].item<float>();
    float denom = sqrtf(gdg * gdm + 1e-12f);
    float gate = (denom > 0.0f) ? (gn / denom) : 0.0f;
    gate = fminf(fmaxf(gate, 0.0f), 1.0f);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg11_sweep_b", [&] {
            sg11_sweep_b_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                mu_buf.data_ptr<float>(),
                gate, alpha, lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}

}} // namespace sg::cuda_sm90
