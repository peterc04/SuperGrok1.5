/*
 * Grokking Optimizers — CPU-Only Pybind11 Module
 *
 * Minimal bindings for CPU-only builds. Exposes:
 *   - supergrok2_cpu_step: Full SuperGrok v2 forward step on CPU
 *   - CPU versions of simpler optimizers (use ATen ops directly)
 *
 * For GPU builds, csrc/common/ops.cpp is used instead.
 */

#include <torch/extension.h>
#include <vector>

// Declared in cpu_kernels.cpp
void supergrok2_cpu_step(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor gru_state,
    torch::Tensor mamba_fwd_state, torch::Tensor mamba_bwd_state,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);


// ═══════════════════════════════════════════════════════════════════
//  CPU Fallback Implementations for Other Optimizers
//
//  These use ATen operations (torch::) which dispatch to MKL/OpenBLAS
//  on CPU automatically. Simple enough to not need custom C++ kernels.
// ═══════════════════════════════════════════════════════════════════

void supergrok15_fused_step_cpu(
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
    for (size_t i = 0; i < params.size(); i++) {
        auto g = grads[i].reshape(-1).to(torch::kFloat32);

        // Gradient clipping
        float gnorm = g.norm().item<float>();
        if (gnorm > grad_clip_norm && grad_clip_norm > 0)
            g = g * (grad_clip_norm / (gnorm + 1e-12f));

        // Meta-net: simple MLP
        auto hidden = torch::relu(torch::addmv(b1, W1, g * rescale));
        auto smart_g = g + rescale * torch::mv(W2.t(), hidden);

        // Mu EMA
        float alpha = layer_alphas[i];
        mus[i].mul_(alpha).add_(g, 1.0f - alpha);

        float lamb_eff = lamb * ramp * gate_signal;
        auto effective = smart_g + lamb_eff * mus[i].reshape(-1);

        // Adam
        float b1v = layer_beta1s[i];
        steps[i] += 1;
        float bc1 = 1.0f - std::pow(b1v, (float)steps[i]);
        float bc2 = 1.0f - std::pow(beta2, (float)steps[i]);

        exp_avgs[i].reshape(-1).mul_(b1v).add_(effective, 1.0f - b1v);
        exp_avg_sqs[i].reshape(-1).mul_(beta2).addcmul_(effective, effective, 1.0f - beta2);

        float step_size = lr / bc1;
        auto denom = (exp_avg_sqs[i].reshape(-1) / bc2).sqrt().add_(eps);
        params[i].reshape(-1).mul_(1.0f - lr * wd_eff).addcdiv_(
            exp_avgs[i].reshape(-1), denom, -step_size);
    }
}

void grokadamw_fused_step_cpu(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<int64_t>& steps,
    float alpha, float lamb,
    float beta1, float beta2, float lr, float wd,
    float eps, float grad_clip_norm
) {
    for (size_t i = 0; i < params.size(); i++) {
        auto g = grads[i].reshape(-1).to(torch::kFloat32);

        float gnorm = g.norm().item<float>();
        if (gnorm > grad_clip_norm && grad_clip_norm > 0)
            g = g * (grad_clip_norm / (gnorm + 1e-12f));

        emas[i].reshape(-1).mul_(alpha).add_(g, 1.0f - alpha);
        auto effective = g + lamb * emas[i].reshape(-1);

        steps[i] += 1;
        float bc1 = 1.0f - std::pow(beta1, (float)steps[i]);
        float bc2 = 1.0f - std::pow(beta2, (float)steps[i]);

        exp_avgs[i].reshape(-1).mul_(beta1).add_(effective, 1.0f - beta1);
        exp_avg_sqs[i].reshape(-1).mul_(beta2).addcmul_(effective, effective, 1.0f - beta2);

        float step_size = lr / bc1;
        auto denom = (exp_avg_sqs[i].reshape(-1) / bc2).sqrt().add_(eps);
        params[i].reshape(-1).mul_(1.0f - lr * wd).addcdiv_(
            exp_avgs[i].reshape(-1), denom, -step_size);
    }
}

void lion_fused_step_cpu(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    float lr, float beta1, float beta2, float wd
) {
    for (size_t i = 0; i < params.size(); i++) {
        auto g = grads[i].reshape(-1).to(torch::kFloat32);
        auto m = exp_avgs[i].reshape(-1);
        auto update = (m * beta1 + g * (1.0f - beta1)).sign();
        params[i].reshape(-1).mul_(1.0f - lr * wd).add_(update, -lr);
        m.mul_(beta2).add_(g, 1.0f - beta2);
    }
}

void grokfast_fused_step_cpu(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& ema_bufs,
    float alpha, float lamb
) {
    for (size_t i = 0; i < grads.size(); i++) {
        ema_bufs[i].mul_(alpha).add_(grads[i], 1.0f - alpha);
        grads[i].add_(ema_bufs[i], lamb);
    }
}


// ═══════════════════════════════════════════════════════════════════
//  Pybind11 Module Registration
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(_ops, m) {
    m.doc() = "Grokking Optimizers — CPU Reference Kernels";

    // ── SuperGrok v2 CPU Step ────────────────────────────────────────
    m.def("supergrok2_cpu_step", &supergrok2_cpu_step,
          "SuperGrok2 CPU: full Mamba-3+PEER step (reference implementation)");

    // ── SuperGrok v1.5 CPU Step ──────────────────────────────────────
    m.def("supergrok15_fused_step", &supergrok15_fused_step_cpu,
          "SuperGrok v1.5 CPU: fused meta-net + Adam step");

    // ── GrokAdamW CPU Step ──────────────────────────────────────────
    m.def("grokadamw_fused_step", &grokadamw_fused_step_cpu,
          "GrokAdamW CPU: fused EMA + Adam step");

    // ── Lion CPU Step ───────────────────────────────────────────────
    m.def("lion_fused_step", &lion_fused_step_cpu,
          "Lion CPU: sign + momentum step");

    // ── Grokfast CPU Step ───────────────────────────────────────────
    m.def("grokfast_fused_step", &grokfast_fused_step_cpu,
          "Grokfast CPU: EMA + amplification step");
}
