// HIP gfx942 launch glue for MoE/Adam multi-tensor.
// Algorithm: csrc/algorithms/moe_adam.h (wraps adamw.h)
//
// Caller passes pre-gathered active expert parameters. Math is identical
// to AdamW.

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace hip_gfx942 {

namespace prim = ::sg::hip_gfx942::primitives;

void launch_moe_adam_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];

        prim::ema_update_inplace(m, g, beta1);
        prim::ema_sq_update_inplace(v, g, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::hip_gfx942
