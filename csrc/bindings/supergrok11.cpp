// bindings/supergrok11.cpp — runtime dispatch to per-arch SG v1.1 launchers.
//
// SG v1.1 mirrors v1.5 except for the cosine-similarity gate. The flow:
//   1. mu_metanet kernel produces smart_grad (and updates mu)
//   2. compute_cosine_gate_fused does a 3-quantity reduction
//      (dot, |sg|², |mu|²), syncs once to CPU, returns sigmoid(t·cos_sim)
//   3. adam_decay applies the Adam step using lamb_eff = ramp·gate·lamb

#include "_dispatch_macro.h"
#include "_helpers.h"

#include <cmath>
#include <vector>

namespace sg {

#define DECLARE_SG11(NS)                                                      \
    namespace NS {                                                            \
        void launch_sg11_mu_metanet(                                          \
            torch::Tensor mu, torch::Tensor grad,                             \
            torch::Tensor sharpness, torch::Tensor smart_grad,                \
            float alpha,                                                      \
            torch::Tensor W1, torch::Tensor b1,                               \
            torch::Tensor W2, torch::Tensor b2,                               \
            float rescale, int hidden_dim);                                   \
        void launch_sg11_adam_decay(                                          \
            torch::Tensor param, torch::Tensor exp_avg,                       \
            torch::Tensor exp_avg_sq, torch::Tensor smart_grad,               \
            torch::Tensor mu,                                                 \
            float lamb_eff, float beta1, float beta2,                         \
            float lr, float wd_eff, float eps, float bc1, float bc2);         \
        void launch_sg11_sam_perturb(                                         \
            torch::Tensor param, torch::Tensor grad, float rho_over_norm);    \
        void launch_sg11_sharpness_restore(                                   \
            torch::Tensor param, torch::Tensor sharpness,                     \
            torch::Tensor backup,                                             \
            torch::Tensor sam_grad, torch::Tensor normal_grad);               \
        float compute_cosine_gate_fused(                                      \
            torch::Tensor smart_grad, torch::Tensor mu, float gate_temp);     \
    }

DECLARE_SG11(sm80) DECLARE_SG11(sm89) DECLARE_SG11(sm90)
DECLARE_SG11(sm100) DECLARE_SG11(sm103) DECLARE_SG11(sm120)
DECLARE_SG11(gfx942) DECLARE_SG11(gfx950)
#undef DECLARE_SG11

// Per-tensor cosine-gate dispatch helper. The arch-namespaced
// compute_cosine_gate_fused is a host-side function that internally
// launches the 3-quantity reduction kernel and does the sigmoid on the
// result. Returns the gate.
static float dispatch_cosine_gate(
    torch::Tensor smart_grad, torch::Tensor mu, float gate_temp)
{
    const int sg_arch_ = ::sg::detect_arch();
    switch (sg_arch_) {
        case 80:  return ::sg::sm80::compute_cosine_gate_fused(smart_grad, mu, gate_temp);
        case 89:  return ::sg::sm89::compute_cosine_gate_fused(smart_grad, mu, gate_temp);
        case 90:  return ::sg::sm90::compute_cosine_gate_fused(smart_grad, mu, gate_temp);
        case 100: return ::sg::sm100::compute_cosine_gate_fused(smart_grad, mu, gate_temp);
        case 103: return ::sg::sm103::compute_cosine_gate_fused(smart_grad, mu, gate_temp);
        case 120: return ::sg::sm120::compute_cosine_gate_fused(smart_grad, mu, gate_temp);
        case 942: return ::sg::gfx942::compute_cosine_gate_fused(smart_grad, mu, gate_temp);
        case 950: return ::sg::gfx950::compute_cosine_gate_fused(smart_grad, mu, gate_temp);
        default:
            throw std::runtime_error(
                "compute_cosine_gate_fused: unsupported arch " +
                std::to_string(sg_arch_));
    }
}

// Pre-refactor csrc/common/ops.cpp::supergrok11_fused_step.
void supergrok11_fused_step(
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
    float lamb, float ramp, float gate_temperature,
    float grad_clip_norm
) {
    const size_t n_params = params.size();
    clip_grad_norms_device_side(grads, n_params, grad_clip_norm);

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        steps[i] += 1;
        float alpha = layer_alphas[i];
        float beta1 = layer_beta1s[i];
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));

        auto smart_grad = torch::empty_like(params[i]);
        SG_DISPATCH_CALL(launch_sg11_mu_metanet,
            mus[i], grads[i], sharpness_cache[i], smart_grad, alpha,
            W1, b1, W2, b2, rescale, hidden_dim);

        float gate = dispatch_cosine_gate(smart_grad, mus[i], gate_temperature);
        float lamb_eff = (ramp > 0.0f) ? (ramp * gate * lamb) : 0.0f;

        SG_DISPATCH_CALL(launch_sg11_adam_decay,
            params[i], exp_avgs[i], exp_avg_sqs[i], smart_grad, mus[i],
            lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2);
    }
}

// Pre-refactor: SG v1.1 SAM perturb-all reuses v1.5's logic (same math).
std::vector<torch::Tensor> supergrok11_sam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho)
{
    float grad_norm = compute_sam_grad_norm_device_side(grads, grads.size());
    float rho_over_norm = rho / grad_norm;
    std::vector<torch::Tensor> backups;
    backups.reserve(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        backups.push_back(params[i].clone());
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        SG_DISPATCH_CALL(launch_sg11_sam_perturb,
            params[i], grads[i], rho_over_norm);
    }
    return backups;
}

void supergrok11_sharpness_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<torch::Tensor>& backups,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads)
{
    for (size_t i = 0; i < params.size(); i++) {
        if (!sam_grads[i].defined() || !normal_grads[i].defined()
            || sam_grads[i].numel() == 0)
        {
            params[i].copy_(backups[i]);
            continue;
        }
        SG_DISPATCH_CALL(launch_sg11_sharpness_restore,
            params[i], sharpness_cache[i], backups[i],
            sam_grads[i], normal_grads[i]);
    }
}

} // namespace sg
