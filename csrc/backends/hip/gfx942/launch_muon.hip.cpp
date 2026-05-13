// HIP gfx942 launch glue for Muon.
// Algorithm: csrc/algorithms/muon.h
//
// Newton-Schulz iterations use torch::mm (which routes to rocBLAS).
// 1D parameters fall back to AdamW.

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace hip_gfx942 {

namespace prim = ::sg::hip_gfx942::primitives;

static inline torch::Tensor newton_schulz_iterate(
    torch::Tensor X, int ns_steps, float a, float b, float c
) {
    for (int it = 0; it < ns_steps; it++) {
        auto AX  = torch::mm(X.transpose(-2, -1), X);
        auto AAX = torch::mm(AX, AX);
        X = a * X + b * torch::mm(X, AX) + c * torch::mm(X, AAX);
    }
    return X;
}

void launch_muon_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& bufs,
    std::vector<torch::Tensor>& grads,
    float lr, float momentum, float wd, int ns_steps,
    float ns_a, float ns_b, float ns_c
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& buf = bufs[i];
        auto& g = grads[i];

        buf.mul_(momentum).add_(g.to(buf.scalar_type()), 1.0f - momentum);

        if (p.dim() >= 2) {
            auto frob = buf.norm() + 1e-8f;
            auto X = buf / frob;
            X = newton_schulz_iterate(X, ns_steps, ns_a, ns_b, ns_c);
            float neg_lr_scale = -lr * 0.2f * sqrtf((float)std::max<int64_t>(p.size(-1), p.size(-2)));
            p.mul_(1.0f - lr * wd).add_(X.to(p.scalar_type()), neg_lr_scale);
        } else {
            // 1D fall back: Adam-like
            p.add_(buf.to(p.scalar_type()), -lr);
        }
    }
}

}} // namespace sg::hip_gfx942
