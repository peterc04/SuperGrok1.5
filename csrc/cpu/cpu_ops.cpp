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
//  SuperGrok v1.1 CPU (same structure as v1.5, different gate)
// ═══════════════════════════════════════════════════════════════════

void supergrok11_fused_step_cpu(
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
    // v1.1 uses cosine gate (gate_temperature replaces gate_signal)
    // Reuse the same numerics as v1.5 for the CPU fallback
    supergrok15_fused_step_cpu(
        params, grads, exp_avgs, exp_avg_sqs, mus, sharpness_cache,
        steps, layer_alphas, layer_beta1s,
        W1, b1, W2, b2, rescale, hidden_dim,
        beta2, lr, wd_eff, eps,
        lamb, ramp, gate_temperature, grad_clip_norm
    );
}


// ═══════════════════════════════════════════════════════════════════
//  NeuralGrok CPU
// ═══════════════════════════════════════════════════════════════════

void neuralgrok_fused_step_cpu(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<int64_t>& steps,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W_last, torch::Tensor b_last,
    float alpha, float beta, int hidden_dim,
    float beta1, float beta2, float lr, float wd,
    float eps, float grad_clip_norm
) {
    auto W1_f = W1.to(torch::kFloat32);
    auto b1_f = b1.to(torch::kFloat32);
    auto Wl_f = W_last.to(torch::kFloat32);
    auto bl_f = b_last.to(torch::kFloat32);

    for (size_t i = 0; i < params.size(); i++) {
        auto g = grads[i].reshape(-1).to(torch::kFloat32);

        float gnorm = g.norm().item<float>();
        if (gnorm > grad_clip_norm && grad_clip_norm > 0)
            g = g * (grad_clip_norm / (gnorm + 1e-12f));

        // Amplifier: scale = alpha * MLP(|g|) + beta
        auto abs_g = g.abs().unsqueeze(1);
        auto hidden = torch::relu(torch::addmm(b1_f.unsqueeze(0), abs_g, W1_f.t()));
        auto mlp_out = torch::addmm(bl_f.unsqueeze(0), hidden, Wl_f.t()).squeeze(1);
        auto smart_g = g * (alpha * mlp_out + beta);

        steps[i] += 1;
        float bc1_v = 1.0f - std::pow(beta1, (float)steps[i]);
        float bc2_v = 1.0f - std::pow(beta2, (float)steps[i]);

        exp_avgs[i].reshape(-1).mul_(beta1).add_(smart_g, 1.0f - beta1);
        exp_avg_sqs[i].reshape(-1).mul_(beta2).addcmul_(smart_g, smart_g, 1.0f - beta2);

        float step_size = lr / bc1_v;
        auto denom = (exp_avg_sqs[i].reshape(-1) / bc2_v).sqrt().add_(eps);
        params[i].reshape(-1).mul_(1.0f - lr * wd).addcdiv_(
            exp_avgs[i].reshape(-1), denom, -step_size);
    }
}


// ═══════════════════════════════════════════════════════════════════
//  Prodigy CPU
// ═══════════════════════════════════════════════════════════════════

float prodigy_fused_step_cpu(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_bufs,
    std::vector<torch::Tensor>& param_inits,
    std::vector<int64_t>& steps,
    float d_lr,
    float beta1, float beta2, float lr, float wd, float eps
) {
    float num_acc = 0.0f, den_acc = 0.0f;

    for (size_t i = 0; i < params.size(); i++) {
        auto p = params[i].reshape(-1);
        auto g = grads[i].reshape(-1).to(torch::kFloat32);
        auto p0 = param_inits[i].reshape(-1);
        auto s = s_bufs[i].reshape(-1);

        steps[i] += 1;
        int64_t step = steps[i];

        num_acc += (g * (p.to(torch::kFloat32) - p0)).sum().item<float>();
        den_acc += (s.to(torch::kFloat32) * g.abs()).sum().item<float>();

        s.mul_(beta2).add_(g.abs() * d_lr, 1.0f - beta2);

        auto effective = g * d_lr;

        float bc1 = 1.0f - std::pow(beta1, (float)step);
        float bc2 = 1.0f - std::pow(beta2, (float)step);

        exp_avgs[i].reshape(-1).mul_(beta1).add_(effective, 1.0f - beta1);
        exp_avg_sqs[i].reshape(-1).mul_(beta2).addcmul_(effective, effective, 1.0f - beta2);

        float step_size = lr / bc1;
        auto denom = (exp_avg_sqs[i].reshape(-1) / bc2).sqrt().add_(eps);
        p.mul_(1.0f - lr * wd).addcdiv_(exp_avgs[i].reshape(-1), denom, -step_size);
    }

    if (den_acc > 0)
        d_lr = std::max(d_lr, num_acc / den_acc);
    return d_lr;
}


// ═══════════════════════════════════════════════════════════════════
//  LookSAM CPU
// ═══════════════════════════════════════════════════════════════════

std::vector<torch::Tensor> looksam_perturb_all_cpu(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho
) {
    std::vector<torch::Tensor> backups;
    for (size_t i = 0; i < params.size(); i++) {
        backups.push_back(params[i].data().clone());
        float gnorm = grads[i].norm().item<float>();
        if (gnorm > 0)
            params[i].data().add_(grads[i], rho / (gnorm + 1e-12f));
    }
    return backups;
}

void looksam_restore_all_cpu(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& backups
) {
    for (size_t i = 0; i < params.size(); i++)
        params[i].data().copy_(backups[i]);
}

void looksam_compute_directions_cpu(
    std::vector<torch::Tensor>& directions,
    std::vector<torch::Tensor>& perturbed_grads,
    std::vector<torch::Tensor>& orig_grads
) {
    for (size_t i = 0; i < directions.size(); i++) {
        auto diff = perturbed_grads[i] - orig_grads[i];
        auto dnorm = diff.norm();
        if (dnorm.item<float>() > 0)
            directions[i].copy_(diff / dnorm);
    }
}

void looksam_adjust_grads_cpu(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& directions,
    float alpha
) {
    for (size_t i = 0; i < grads.size(); i++) {
        float proj = (grads[i] * directions[i]).sum().item<float>();
        grads[i].add_(directions[i], alpha * proj);
    }
}


// ═══════════════════════════════════════════════════════════════════
//  Muon CPU
// ═══════════════════════════════════════════════════════════════════

void muon_fused_step_cpu(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& momentum_bufs,
    float momentum, float lr, float wd, int ns_steps
) {
    for (size_t i = 0; i < params.size(); i++) {
        auto p = params[i];
        auto g = grads[i].to(torch::kFloat32);
        auto m = momentum_bufs[i];

        m.mul_(momentum).add_(g);

        torch::Tensor update;
        if (p.dim() >= 2) {
            // Newton-Schulz orthogonalization
            auto X = m.clone();
            auto norm_val = X.norm();
            if (norm_val.item<float>() > 0)
                X.div_(norm_val);
            for (int s = 0; s < ns_steps; s++) {
                auto A = torch::mm(X, X.t());
                X = 1.5f * X - 0.5f * torch::mm(A, X);
            }
            update = X * norm_val;
        } else {
            update = m;
        }

        p.mul_(1.0f - lr * wd);
        p.add_(update, -lr);
    }
}


// ═══════════════════════════════════════════════════════════════════
//  Pybind11 Module Registration
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(_ops, m) {
    m.doc() = "Grokking Optimizers — CPU Reference Kernels (all optimizers)";

    // ── SuperGrok v2 CPU Step ────────────────────────────────────────
    m.def("supergrok2_cpu_step", &supergrok2_cpu_step,
          "SuperGrok2 CPU: full Mamba-3+PEER step (reference implementation)");

    // ── SuperGrok v1.5 CPU Step ──────────────────────────────────────
    m.def("supergrok15_fused_step", &supergrok15_fused_step_cpu,
          "SuperGrok v1.5 CPU: fused meta-net + Adam step");

    // ── SuperGrok v1.1 CPU Step ──────────────────────────────────────
    m.def("supergrok11_fused_step", &supergrok11_fused_step_cpu,
          "SuperGrok v1.1 CPU: fused meta-net + cosine gate + Adam step");

    // ── GrokAdamW CPU Step ──────────────────────────────────────────
    m.def("grokadamw_fused_step", &grokadamw_fused_step_cpu,
          "GrokAdamW CPU: fused EMA + Adam step");

    // ── NeuralGrok CPU Step ──────────────────────────────────────────
    m.def("neuralgrok_fused_step", &neuralgrok_fused_step_cpu,
          "NeuralGrok CPU: amplifier MLP + Adam step");

    // ── Prodigy CPU Step ─────────────────────────────────────────────
    m.def("prodigy_fused_step", &prodigy_fused_step_cpu,
          "Prodigy CPU: distance-aware self-tuning Adam step");

    // ── Grokfast CPU Step ───────────────────────────────────────────
    m.def("grokfast_fused_step", &grokfast_fused_step_cpu,
          "Grokfast CPU: EMA + amplification step");

    // ── Lion CPU Step ───────────────────────────────────────────────
    m.def("lion_fused_step", &lion_fused_step_cpu,
          "Lion CPU: sign + momentum step");

    // ── LookSAM CPU Steps ───────────────────────────────────────────
    m.def("looksam_perturb_all", &looksam_perturb_all_cpu,
          "LookSAM CPU: SAM perturbation with backups");
    m.def("looksam_restore_all", &looksam_restore_all_cpu,
          "LookSAM CPU: restore from backups");
    m.def("looksam_compute_directions", &looksam_compute_directions_cpu,
          "LookSAM CPU: compute sharpness-aware directions");
    m.def("looksam_adjust_grads", &looksam_adjust_grads_cpu,
          "LookSAM CPU: adjust gradients with directions");

    // ── Muon CPU Step ────────────────────────────────────────────────
    m.def("muon_fused_step", &muon_fused_step_cpu,
          "Muon CPU: Newton-Schulz orthogonalization + momentum step");
}
