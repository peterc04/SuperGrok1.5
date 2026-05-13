// HIP gfx942 launch glue for SuperGrok v1.5.
// Algorithm: csrc/algorithms/supergrok15.h

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace hip_gfx942 {

namespace prim = ::sg::hip_gfx942::primitives;

void launch_supergrok15_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& mu_bufs,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& sharpnesses,
    const torch::Tensor& phi_W1,
    const torch::Tensor& phi_b1,
    const torch::Tensor& phi_W2,
    float phi_b2,
    float gate_global,
    float alpha_base, float alpha_max,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& mu = mu_bufs[i];

        // Sweep A: meta-net forward
        auto x = torch::stack({g.to(torch::kFloat32).view({-1}),
                               sharpnesses[i].view({-1})}, /*dim=*/1);
        auto h = torch::tanh(torch::matmul(x, phi_W1.t()) + phi_b1);
        auto mu_flat = (torch::matmul(h, phi_W2.unsqueeze(1)) + phi_b2).view_as(g);
        mu.copy_(mu_flat);

        // Per-coord alpha, then smart_grad
        auto a_per_coord = torch::clamp(alpha_base * (1.0f + mu), 0.0f, alpha_max);
        auto smart = g.to(torch::kFloat32) + gate_global * a_per_coord * mu;

        prim::ema_update_inplace(m, smart, beta1);
        prim::ema_sq_update_inplace(v, smart, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::hip_gfx942
