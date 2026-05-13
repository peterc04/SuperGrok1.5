// HIP gfx942 launch glue for LookSAM (4 operations).
// Algorithm: csrc/algorithms/looksam.h

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace hip_gfx942 {

namespace prim = ::sg::hip_gfx942::primitives;

void launch_looksam_perturb(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& backups,
    std::vector<torch::Tensor>& grads,
    float scale
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        backups[i].copy_(params[i]);
        params[i].add_(grads[i].to(params[i].scalar_type()), scale);
    }
}

void launch_looksam_restore(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& backups
) {
    for (size_t i = 0; i < params.size(); i++) {
        params[i].copy_(backups[i]);
    }
}

void launch_looksam_set_direction(
    std::vector<torch::Tensor>& sam_dirs,
    std::vector<torch::Tensor>& grads_sam,
    std::vector<torch::Tensor>& grads_orig
) {
    for (size_t i = 0; i < sam_dirs.size(); i++) {
        sam_dirs[i].copy_(grads_sam[i] - grads_orig[i]);
    }
}

void launch_looksam_apply(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& sam_dirs,
    std::vector<torch::Tensor>& grads,
    float alpha,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];

        auto g_adj = (1.0f - alpha) * g.to(torch::kFloat32) + alpha * sam_dirs[i];
        prim::ema_update_inplace(m, g_adj, beta1);
        prim::ema_sq_update_inplace(v, g_adj, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::hip_gfx942
