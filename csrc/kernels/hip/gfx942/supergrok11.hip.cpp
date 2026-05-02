// =====================================================================
//  csrc/kernels/hip/gfx942/supergrok11.hip.cpp
//
//  gfx942 SuperGrok v1.1 launchers. The pipeline is:
//    1) mu_metanet:       EMA on mu + 2-layer MLP -> smart_grad
//    2) cosine_gate:      gate = sigmoid(temp * cos(smart_grad, mu))
//    3) adam_decay:       Adam step using lamb_eff = ramp * gate * lamb
//    4) sam_perturb:      param += rho/|g| * grad
//    5) sharpness_restore: sharpness = |sam_grad - normal_grad|; restore
//
//  This port mirrors the sm_90 implementation, which does the math via
//  ATen tensor ops (so it's host-side and rocBLAS handles GEMMs).
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/supergrok11.hip.h"

namespace sg { namespace gfx942 {

void launch_sg11_mu_metanet(
    torch::Tensor mu, torch::Tensor grad,
    torch::Tensor sharpness, torch::Tensor smart_grad,
    float alpha,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim
) {
    (void)hidden_dim;

    // EMA: mu = (1 - alpha) * mu + alpha * grad
    mu.mul_(1.0f - alpha).add_(grad, alpha);

    auto g_flat = grad.flatten().to(torch::kFloat32);
    int64_t N = g_flat.size(0);

    // Build [N, 2] input: [grad, broadcast(sharpness)].
    float s_val = sharpness.item<float>();
    auto inp = torch::empty({N, 2}, g_flat.options());
    inp.select(1, 0).copy_(g_flat);
    inp.select(1, 1).fill_(s_val);

    auto W1_f = W1.to(torch::kFloat32);
    auto b1_f = b1.to(torch::kFloat32);
    auto W2_f = W2.to(torch::kFloat32);
    auto b2_f = b2.to(torch::kFloat32);

    auto hidden = torch::addmm(b1_f, inp, W1_f.t()).gelu();
    auto out    = torch::addmm(b2_f, hidden, W2_f.t());
    smart_grad.copy_(out.reshape_as(smart_grad).mul_(rescale));
}

float compute_cosine_gate_fused(
    torch::Tensor smart_grad, torch::Tensor mu, float gate_temp
) {
    auto sg_flat = smart_grad.flatten().to(torch::kFloat32);
    auto mu_flat = mu.flatten().to(torch::kFloat32);
    auto dot   = torch::dot(sg_flat, mu_flat);
    auto sg_n  = torch::norm(sg_flat);
    auto mu_n  = torch::norm(mu_flat);
    auto cos_sim = dot / (sg_n * mu_n + 1e-12f);
    auto gate    = torch::sigmoid(cos_sim * gate_temp);
    return gate.item<float>();
}

void launch_sg11_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor smart_grad,
    torch::Tensor mu,
    float lamb_eff, float beta1, float beta2,
    float lr, float wd_eff, float eps, float bc1, float bc2
) {
    (void)mu;
    exp_avg.mul_(beta1).add_(smart_grad, 1.0f - beta1);
    exp_avg_sq.mul_(beta2).addcmul_(smart_grad, smart_grad, 1.0f - beta2);
    auto m_hat = exp_avg / bc1;
    auto v_hat = exp_avg_sq / bc2;
    auto update = m_hat / (v_hat.sqrt() + eps);
    param.mul_(1.0f - lr * wd_eff).add_(update, -lr * lamb_eff);
}

void launch_sg11_sam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm
) {
    param.add_(grad, rho_over_norm);
}

void launch_sg11_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness,
    torch::Tensor backup,
    torch::Tensor sam_grad, torch::Tensor normal_grad
) {
    sharpness.copy_((sam_grad - normal_grad).abs());
    param.copy_(backup);
}

}} // namespace sg::gfx942
