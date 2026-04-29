// bindings/looksam.cpp — runtime dispatch to per-arch LookSAM launchers.

#include "_dispatch_macro.h"
#include "_helpers.h"

#include <vector>

namespace sg {

#define DECLARE_LOOKSAM(NS)                                                   \
    namespace NS {                                                            \
        void launch_looksam_perturb(                                          \
            torch::Tensor param, torch::Tensor backup, torch::Tensor grad,   \
            float rho_over_norm);                                             \
        void launch_looksam_restore(                                          \
            torch::Tensor param, torch::Tensor backup);                       \
        void launch_looksam_direction_adjust_fused(                           \
            torch::Tensor grad, torch::Tensor sam_grad,                       \
            torch::Tensor v_dir,                                              \
            float inv_norm, float lambda, float grad_norm);                   \
        void launch_looksam_norm_reduce(                                      \
            torch::Tensor grad, torch::Tensor sam_grad,                       \
            torch::Tensor results /* [diff_norm, grad_norm] */);              \
    }

DECLARE_LOOKSAM(sm80) DECLARE_LOOKSAM(sm90)
DECLARE_LOOKSAM(sm100) DECLARE_LOOKSAM(gfx942)
DECLARE_LOOKSAM(sm89) DECLARE_LOOKSAM(sm103) DECLARE_LOOKSAM(sm120) DECLARE_LOOKSAM(gfx950)
#undef DECLARE_LOOKSAM

void looksam_perturb(
    torch::Tensor param, torch::Tensor backup, torch::Tensor grad,
    float rho_over_norm)
{
    SG_DISPATCH(launch_looksam_perturb, param, backup, grad, rho_over_norm);
}

void looksam_restore(torch::Tensor param, torch::Tensor backup) {
    SG_DISPATCH(launch_looksam_restore, param, backup);
}

void looksam_direction_adjust_fused(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor v_dir,
    float inv_norm, float lambda, float grad_norm)
{
    SG_DISPATCH(launch_looksam_direction_adjust_fused,
        grad, sam_grad, v_dir, inv_norm, lambda, grad_norm);
}

void looksam_norm_reduce(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor results)
{
    SG_DISPATCH(launch_looksam_norm_reduce, grad, sam_grad, results);
}

// ---------------------------------------------------------------------
// High-level vector-signature entry points (pre-refactor ops.cpp).
// ---------------------------------------------------------------------

// Pre-refactor csrc/common/ops.cpp::looksam_perturb_all.
// Returns a vector of backups (param.clone() before perturb).
std::vector<torch::Tensor> looksam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho
) {
    float grad_norm = compute_sam_grad_norm_device_side(grads, grads.size());
    float rho_over_norm = rho / grad_norm;

    std::vector<torch::Tensor> backups;
    backups.reserve(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        backups.push_back(params[i].clone());
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        SG_DISPATCH_CALL(launch_looksam_perturb,
            params[i], backups[i], grads[i], rho_over_norm);
    }
    return backups;
}

// Pre-refactor csrc/common/ops.cpp::looksam_restore_all.
void looksam_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& backups
) {
    for (size_t i = 0; i < params.size(); i++) {
        SG_DISPATCH_CALL(launch_looksam_restore, params[i], backups[i]);
    }
}

// Pre-refactor csrc/common/ops.cpp::looksam_compute_directions_and_adjust.
// Batches the per-tensor norm reductions into 2 CPU syncs (diff + grad
// norms via torch::stack), then dispatches the fused adjust kernel.
void looksam_compute_directions_and_adjust(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads,
    float la
) {
    std::vector<size_t> valid_idx;
    for (size_t i = 0; i < grads.size(); i++) {
        if (!sam_grads[i].defined() || !normal_grads[i].defined()
            || sam_grads[i].numel() == 0) continue;
        valid_idx.push_back(i);
    }
    if (valid_idx.empty()) return;

    std::vector<torch::Tensor> diff_norms_t, grad_norms_t, diffs;
    diff_norms_t.reserve(valid_idx.size());
    grad_norms_t.reserve(valid_idx.size());
    diffs.reserve(valid_idx.size());
    for (auto i : valid_idx) {
        auto diff = (sam_grads[i] - normal_grads[i]).to(torch::kFloat32);
        diffs.push_back(diff);
        diff_norms_t.push_back(diff.norm());
        grad_norms_t.push_back(grads[i].norm());
    }
    auto diff_norms = torch::stack(diff_norms_t).cpu();
    auto grad_norms = torch::stack(grad_norms_t).cpu();
    auto* dnp = diff_norms.data_ptr<float>();
    auto* gnp = grad_norms.data_ptr<float>();

    for (size_t vi = 0; vi < valid_idx.size(); vi++) {
        size_t i = valid_idx[vi];
        float dn = dnp[vi];
        if (dn < 1e-12f) continue;
        float inv_norm = 1.0f / dn;
        float gn = gnp[vi];
        // The per-arch launcher's signature is
        //   (grad, sam_grad, v_dir, inv_norm, lambda, grad_norm)
        // — passes diff (==v_dir for this fused kernel) as the third arg.
        SG_DISPATCH_CALL(launch_looksam_direction_adjust_fused,
            grads[i], sam_grads[i], diffs[vi], inv_norm, la, gn);
    }
}

} // namespace sg
