// HIP gfx942 launch glue for Lion.
// Algorithm: csrc/algorithms/lion.h

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace hip_gfx942 {

namespace prim = ::sg::hip_gfx942::primitives;

void launch_lion_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float wd
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& ea = exp_avgs[i];

        // Interpolation, sign, update
        auto interp = beta1 * ea + (1.0f - beta1) * g.to(ea.scalar_type());
        auto upd = interp.sign();
        p.add_(upd + wd * p, -lr);

        // Momentum refresh
        ea.mul_(beta2).add_(g.to(ea.scalar_type()), 1.0f - beta2);
    }
}

}} // namespace sg::hip_gfx942
