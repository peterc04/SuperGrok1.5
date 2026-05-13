// HIP gfx942 launch glue for GrokAdamW.
// Algorithm: csrc/algorithms/grokadamw.h

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace hip_gfx942 {

namespace prim = ::sg::hip_gfx942::primitives;

void launch_grokadamw_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    float alpha, float lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& ema = emas[i];

        // EMA filter
        prim::ema_update_inplace(ema, g, alpha);
        // Amplified gradient
        auto g_amp = g.to(torch::kFloat32) + lamb * ema;
        // Adam moments on g_amp
        prim::ema_update_inplace(m, g_amp, beta1);
        prim::ema_sq_update_inplace(v, g_amp, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::hip_gfx942
