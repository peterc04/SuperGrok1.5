// bindings/supergrok15.cpp — runtime dispatch to per-arch SG v1.5 launchers.
//
// SG v1.5 has launchers for the meta-net forward, the Adam step with
// decoupled WD, the SAM perturb, the sharpness restore (post-SAM), and a
// fused full-step that does the whole pipeline in one kernel.

#include "_dispatch_macro.h"
#include "_helpers.h"

#include <cmath>
#include <vector>

namespace sg {

#define DECLARE_SG15(NS) \
 namespace NS { \
 void launch_fused_supergrok15_full_step( \
 torch::Tensor param, \
 torch::Tensor exp_avg, torch::Tensor exp_avg_sq, \
 torch::Tensor mu, torch::Tensor grad, \
 torch::Tensor sharpness, float alpha, \
 torch::Tensor W1, torch::Tensor b1, \
 torch::Tensor W2, torch::Tensor b2, \
 float rescale, float lamb_eff, \
 float beta1, float beta2, \
 float lr, float wd_eff, float eps, \
 float bc1, float bc2, int hidden_dim); \
 void launch_sam_perturb( \
 torch::Tensor param, torch::Tensor grad, float rho_over_norm); \
 void launch_sharpness_restore( \
 torch::Tensor param, torch::Tensor sharpness, \
 torch::Tensor backup, \
 torch::Tensor sam_grad, torch::Tensor normal_grad); \
 }

 DECLARE_SG15(sm90)

DECLARE_SG15(gfx942) 
#undef DECLARE_SG15

void sg15_sam_perturb(torch::Tensor param, torch::Tensor grad, float rho_over_norm) {
 SG_DISPATCH(launch_sam_perturb, param, grad, rho_over_norm);
}

// Pre-refactor csrc/common/ops.cpp::supergrok15_fused_step.
void supergrok15_fused_step(
 std::vector<torch::Tensor>& params,
 std::vector<torch::Tensor>& grads,
 std::vector<torch::Tensor>& exp_avgs,
 std::vector<torch::Tensor>& exp_avg_sqs,
 std::vector<torch::Tensor>& mus,
 std::vector<torch::Tensor>& sharpness_cache,
 std::vector<int64_t>& steps,
 std::vector<float>& layer_alphas,
 std::vector<float>& layer_beta1s,
 torch::Tensor W1, torch::Tensor b1,
 torch::Tensor W2, torch::Tensor b2,
 float rescale, int hidden_dim,
 float beta2, float lr, float wd_eff, float eps,
 float lamb, float ramp, float gate_signal,
 float grad_clip_norm
) {
 const size_t n_params = params.size();
 clip_grad_norms_device_side(grads, n_params, grad_clip_norm);

 float lamb_eff = (ramp > 0.0f) ? (ramp * gate_signal * lamb) : 0.0f;

 for (size_t i = 0; i < n_params; i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 steps[i] += 1;
 int64_t step = steps[i];
 float alpha = layer_alphas[i];
 float beta1 = layer_beta1s[i];
 float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
 float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

 SG_DISPATCH_CALL(launch_fused_supergrok15_full_step,
 params[i], exp_avgs[i], exp_avg_sqs[i], mus[i],
 grads[i], sharpness_cache[i], alpha,
 W1, b1, W2, b2, rescale,
 lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2,
 hidden_dim);
 }
}

// Pre-refactor csrc/common/ops.cpp::supergrok15_sam_perturb_all.
std::vector<torch::Tensor> supergrok15_sam_perturb_all(
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
 SG_DISPATCH_CALL(launch_sam_perturb, params[i], grads[i], rho_over_norm);
 }
 return backups;
}

// Pre-refactor csrc/common/ops.cpp::supergrok15_sharpness_restore_all.
void supergrok15_sharpness_restore_all(
 std::vector<torch::Tensor>& params,
 std::vector<torch::Tensor>& sharpness_cache,
 std::vector<torch::Tensor>& backups,
 std::vector<torch::Tensor>& sam_grads,
 std::vector<torch::Tensor>& normal_grads
) {
 for (size_t i = 0; i < params.size(); i++) {
 if (!sam_grads[i].defined() || !normal_grads[i].defined()
 || sam_grads[i].numel() == 0)
 {
 params[i].copy_(backups[i]);
 continue;
 }
 SG_DISPATCH_CALL(launch_sharpness_restore,
 params[i], sharpness_cache[i], backups[i],
 sam_grads[i], normal_grads[i]);
 }
}

} // namespace sg
