// bindings/supergrok11.cpp — runtime dispatch to per-arch SG v1.1 launchers.
//
// SG v1.1 mirrors v1.5 except for the cosine-similarity gate. Adds:
// - launch_sg11_cosine_gate_reduce: fused 3-quantity reduction (dot, |sg|², |mu|²)
// - The gate is a temperature-scaled sigmoid of cos_sim computed host-side
//   from the reduction outputs.

#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_SG11(NS)                                                      \
    namespace NS {                                                            \
        void launch_sg11_mu_metanet(                                          \
            torch::Tensor grad, torch::Tensor sharpness, torch::Tensor mu,   \
            torch::Tensor smart_grad,                                         \
            torch::Tensor metanet_w1, torch::Tensor metanet_b1,               \
            torch::Tensor metanet_w2, torch::Tensor metanet_b2,               \
            int hidden_dim, float alpha, float rescale);                      \
        void launch_sg11_adam_decay(                                          \
            torch::Tensor param, torch::Tensor smart_grad, torch::Tensor mu, \
            torch::Tensor exp_avg, torch::Tensor exp_avg_sq,                  \
            float lamb_eff, float beta1, float beta2,                         \
            float lr, float wd_eff, float eps,                                \
            float bc1, float bc2);                                            \
        void launch_sg11_cosine_gate_reduce(                                  \
            torch::Tensor smart_grad, torch::Tensor mu,                       \
            torch::Tensor results /* [dot, sg_norm_sq, mu_norm_sq] */);       \
        void launch_sg11_sam_perturb(                                         \
            torch::Tensor param, torch::Tensor grad, float rho_over_norm);    \
        void launch_fused_sg11_full_step(                                     \
            torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness, \
            torch::Tensor mu, torch::Tensor exp_avg,                          \
            torch::Tensor exp_avg_sq,                                         \
            torch::Tensor metanet_w1, torch::Tensor metanet_b1,               \
            torch::Tensor metanet_w2, torch::Tensor metanet_b2,               \
            int hidden_dim, float alpha, float rescale, float lamb_eff,      \
            float beta1, float beta2, float lr, float wd_eff,                 \
            float eps, float bc1, float bc2);                                 \
    }

DECLARE_SG11(sm80) DECLARE_SG11(sm90) DECLARE_SG11(sm100) DECLARE_SG11(gfx942)
DECLARE_SG11(sm89) DECLARE_SG11(sm103) DECLARE_SG11(sm120) DECLARE_SG11(gfx950)
#undef DECLARE_SG11

void sg11_fused_step(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor mu, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor metanet_w1, torch::Tensor metanet_b1,
    torch::Tensor metanet_w2, torch::Tensor metanet_b2,
    int hidden_dim, float alpha, float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff,
    float eps, float bc1, float bc2)
{
    SG_DISPATCH(launch_fused_sg11_full_step,
        param, grad, sharpness, mu, exp_avg, exp_avg_sq,
        metanet_w1, metanet_b1, metanet_w2, metanet_b2,
        hidden_dim, alpha, rescale, lamb_eff,
        beta1, beta2, lr, wd_eff, eps, bc1, bc2);
}

void sg11_cosine_gate_reduce(
    torch::Tensor smart_grad, torch::Tensor mu, torch::Tensor results)
{
    SG_DISPATCH(launch_sg11_cosine_gate_reduce, smart_grad, mu, results);
}

} // namespace sg
