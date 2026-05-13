// HIP gfx942 launch glue for Prodigy.
// Algorithm: csrc/algorithms/prodigy.h
//
// Three-stage: (1) reduce r,s partials, (2) update d, (3) apply Adam with d.
// On HIP, the device-resident d_t scalar is kept as a 1-element tensor.

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace hip_gfx942 {

namespace prim = ::sg::hip_gfx942::primitives;

void launch_prodigy_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_tracks,
    std::vector<torch::Tensor>& param_inits,
    std::vector<torch::Tensor>& grads,
    torch::Tensor& d_t,
    float d_prev,
    float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    // Reduce r, s across all parameters.
    auto r_sum = torch::zeros({}, d_t.options());
    auto s_sum = torch::zeros({}, d_t.options());
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& pi = param_inits[i];
        auto& g = grads[i];

        auto delta = (pi - p).to(torch::kFloat32);
        r_sum += (g.to(torch::kFloat32) * delta).sum() * d_prev;
        s_sum += (g.to(torch::kFloat32).abs().sum()) * (d_prev * d_prev);
    }

    // Update d (on-device scalar).
    auto candidate = r_sum / (s_sum.abs() + 1e-12f);
    d_t.copy_(torch::maximum(d_t.new_full({}, d_prev), candidate));

    float d_val = d_t.item<float>();

    // Apply Adam with d as effective lr.
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& st = s_tracks[i];

        auto g_scaled = d_val * g.to(torch::kFloat32);
        prim::ema_update_inplace(m, g_scaled, beta1);
        prim::ema_sq_update_inplace(v, g_scaled, beta2);
        st.add_(g.to(torch::kFloat32), d_val);
        prim::adam_apply_inplace(p, m, v, d_val, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::hip_gfx942
