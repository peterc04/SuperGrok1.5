// HIP gfx942 launch glue for NeuralGrok.
// Algorithm: csrc/algorithms/neuralgrok.h
//
// Two-stage: psi-net MLP forward (via torch::matmul + ReLU), then
// Adam apply on amplified gradient.

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace hip_gfx942 {

namespace prim = ::sg::hip_gfx942::primitives;

void launch_neuralgrok_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    const torch::Tensor& psi_W1,
    const torch::Tensor& psi_b1,
    const torch::Tensor& psi_W2,
    float psi_b2,
    float alpha, float beta,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];

        // psi forward: input is |grad| flattened
        auto ag = g.abs().to(torch::kFloat32).view({-1, 1});
        auto h = torch::relu(torch::matmul(ag, psi_W1.unsqueeze(0)) + psi_b1);
        auto s = (torch::matmul(h, psi_W2.unsqueeze(1)) + psi_b2).view_as(g);

        auto g_amp = (s * alpha + beta) * g.to(torch::kFloat32);
        prim::ema_update_inplace(m, g_amp, beta1);
        prim::ema_sq_update_inplace(v, g_amp, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::hip_gfx942
