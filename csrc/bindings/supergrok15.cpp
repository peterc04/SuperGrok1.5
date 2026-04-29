// bindings/supergrok15.cpp — runtime dispatch to per-arch SG v1.5 launchers.
//
// SG v1.5 has launchers for the meta-net forward (mu_metanet), the Adam
// step with decoupled WD, the SAM perturb step, and a fused full-step.
// Templated specializations on hidden_dim happen inside the per-arch launcher.

#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_SG15(NS)                                                      \
    namespace NS {                                                            \
        void launch_fused_mu_metanet(                                         \
            torch::Tensor grad, torch::Tensor sharpness,                      \
            torch::Tensor mu, torch::Tensor smart_grad,                       \
            torch::Tensor metanet_w1, torch::Tensor metanet_b1,               \
            torch::Tensor metanet_w2, torch::Tensor metanet_b2,               \
            int hidden_dim, float alpha, float rescale);                      \
        void launch_fused_adam_decay(                                         \
            torch::Tensor param, torch::Tensor smart_grad, torch::Tensor mu, \
            torch::Tensor exp_avg, torch::Tensor exp_avg_sq,                  \
            float lamb_eff, float beta1, float beta2,                         \
            float lr, float wd_eff, float eps,                                \
            float bc1, float bc2);                                            \
        void launch_sam_perturb(                                              \
            torch::Tensor param, torch::Tensor grad, float rho_over_norm);    \
        void launch_fused_supergrok15_full_step(                              \
            torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness, \
            torch::Tensor mu, torch::Tensor exp_avg,                          \
            torch::Tensor exp_avg_sq,                                         \
            torch::Tensor metanet_w1, torch::Tensor metanet_b1,               \
            torch::Tensor metanet_w2, torch::Tensor metanet_b2,               \
            int hidden_dim, float alpha, float rescale, float lamb_eff,      \
            float beta1, float beta2, float lr, float wd_eff,                 \
            float eps, float bc1, float bc2);                                 \
    }

DECLARE_SG15(sm80) DECLARE_SG15(sm90) DECLARE_SG15(sm100) DECLARE_SG15(gfx942)
#undef DECLARE_SG15

void sg15_fused_step(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor mu, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor metanet_w1, torch::Tensor metanet_b1,
    torch::Tensor metanet_w2, torch::Tensor metanet_b2,
    int hidden_dim, float alpha, float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff,
    float eps, float bc1, float bc2)
{
    SG_DISPATCH(launch_fused_supergrok15_full_step,
        param, grad, sharpness, mu, exp_avg, exp_avg_sq,
        metanet_w1, metanet_b1, metanet_w2, metanet_b2,
        hidden_dim, alpha, rescale, lamb_eff,
        beta1, beta2, lr, wd_eff, eps, bc1, bc2);
}

void sg15_sam_perturb(torch::Tensor param, torch::Tensor grad, float rho_over_norm) {
    SG_DISPATCH(launch_sam_perturb, param, grad, rho_over_norm);
}

} // namespace sg
