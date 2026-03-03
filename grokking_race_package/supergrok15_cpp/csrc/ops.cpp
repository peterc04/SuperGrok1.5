/*
 * SuperGrok v1.5 — C++ Operations
 *
 * Eliminates Python overhead by processing all parameters in a tight C++ loop.
 * Dispatches to CUDA kernels when available, falls back to ATen ops on CPU.
 *
 * Key optimisation: the meta-net forward pass is fused into a CUDA kernel
 * that evaluates the 2→H→1 MLP per element in parallel, avoiding the
 * reshape→matmul→reshape overhead of PyTorch's nn.Linear.
 */

#include "ops.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════════
//  CPU Fallback: meta-net inference + mu update
// ═══════════════════════════════════════════════════════════════════════

static void cpu_mu_metanet(
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor smart_grad, float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int H
) {
    // mu EMA update
    mu.mul_(alpha).add_(grad, 1.0f - alpha);

    // Meta-net: reshape to (N, 2), matmul, GELU, matmul, reshape back + skip
    auto shape = grad.sizes().vec();
    auto flat_g = grad.reshape({-1, 1});
    auto flat_s = sharpness.reshape({-1, 1});
    auto inp = torch::cat({flat_g, flat_s}, /*dim=*/1);  // (N, 2)

    // Linear(2, H) → GELU → Linear(H, 1)
    auto z = torch::addmm(b1, inp, W1.t());   // (N, H)
    auto act = torch::gelu(z);                 // (N, H)
    auto out = torch::addmm(b2, act, W2.t()); // (N, 1)

    // Skip connection
    auto result = flat_g + rescale * out;
    smart_grad.copy_(result.reshape(shape));
}

// ═══════════════════════════════════════════════════════════════════════
//  CPU Fallback: fused adam + weight decay
// ═══════════════════════════════════════════════════════════════════════

static void cpu_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor smart_grad, torch::Tensor mu,
    float lamb_eff, float beta1, float beta2,
    float lr, float wd_eff, float eps, float bc1, float bc2
) {
    // final_grad = smart_grad + lamb_eff * mu
    auto fg = smart_grad + lamb_eff * mu;

    // Adam moment updates
    exp_avg.mul_(beta1).add_(fg, 1.0f - beta1);
    exp_avg_sq.mul_(beta2).addcmul_(fg, fg, 1.0f - beta2);

    // Bias-corrected step
    float step_size = lr / bc1;
    auto denom = (exp_avg_sq / bc2).sqrt_().add_(eps);

    // Weight decay + Adam step
    param.mul_(1.0f - lr * wd_eff);
    param.addcdiv_(exp_avg, denom, -step_size);
}


// ═══════════════════════════════════════════════════════════════════════
//  Main Step: Process all parameters in C++ loop
// ═══════════════════════════════════════════════════════════════════════

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
    float lamb, float ramp, float gate_temperature,
    float grad_clip_norm
) {
    const size_t n_params = params.size();

    // ── Gradient clipping (global norm) ──────────────────────────────
    if (grad_clip_norm > 0.0f) {
        float total_norm_sq = 0.0f;
        for (size_t i = 0; i < n_params; i++) {
            if (grads[i].defined() && grads[i].numel() > 0) {
                float norm = grads[i].norm().item<float>();
                total_norm_sq += norm * norm;
            }
        }
        float total_norm = std::sqrt(total_norm_sq);
        if (total_norm > grad_clip_norm) {
            float clip_coef = grad_clip_norm / (total_norm + 1e-6f);
            for (size_t i = 0; i < n_params; i++) {
                if (grads[i].defined() && grads[i].numel() > 0) {
                    grads[i].mul_(clip_coef);
                }
            }
        }
    }

    // ── Per-parameter update ─────────────────────────────────────────
    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0)
            continue;

        auto& p = params[i];
        auto& g = grads[i];
        auto& ea = exp_avgs[i];
        auto& easq = exp_avg_sqs[i];
        auto& mu = mus[i];
        auto& sharp = sharpness_cache[i];

        steps[i] += 1;
        int64_t step = steps[i];
        float alpha = layer_alphas[i];
        float beta1 = layer_beta1s[i];
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

        // Allocate smart_grad (same device/dtype)
        auto smart_grad = torch::empty_like(p);

        // ── Mu update + meta-net inference ───────────────────────────
        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = p.is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_fused_mu_metanet(
                mu, g, sharp, smart_grad, alpha,
                W1, b1, W2, b2, rescale, hidden_dim
            );
#endif
        } else {
            cpu_mu_metanet(mu, g, sharp, smart_grad, alpha,
                           W1, b1, W2, b2, rescale, hidden_dim);
        }

        // ── Soft sigmoid gating ──────────────────────────────────────
        float lamb_eff = 0.0f;
        if (ramp > 0.0f) {
            float mu_norm = mu.norm().item<float>();
            float sg_norm = smart_grad.norm().item<float>();
            if (mu_norm > 1e-12f && sg_norm > 1e-12f) {
                float cos_sim = torch::cosine_similarity(
                    smart_grad.flatten().unsqueeze(0),
                    mu.flatten().unsqueeze(0)
                ).item<float>();
                float gate = 1.0f / (1.0f + std::exp(-gate_temperature * cos_sim));
                lamb_eff = ramp * gate * lamb;
            }
        }

        // ── Fused Adam + progressive weight decay ────────────────────
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_fused_adam_decay(
                p, ea, easq, smart_grad, mu,
                lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2
            );
#endif
        } else {
            cpu_adam_decay(p, ea, easq, smart_grad, mu,
                          lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  SAM Perturbation: perturb all params, return backups
// ═══════════════════════════════════════════════════════════════════════

std::vector<torch::Tensor> supergrok15_sam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho
) {
    // Compute global gradient norm
    float total_norm_sq = 0.0f;
    for (size_t i = 0; i < grads.size(); i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            float n = grads[i].norm().item<float>();
            total_norm_sq += n * n;
        }
    }
    float grad_norm = std::sqrt(total_norm_sq) + 1e-12f;
    float rho_over_norm = rho / grad_norm;

    // Backup params and perturb
    std::vector<torch::Tensor> backups;
    backups.reserve(params.size());

    for (size_t i = 0; i < params.size(); i++) {
        backups.push_back(params[i].clone());

        if (!grads[i].defined() || grads[i].numel() == 0)
            continue;

        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = params[i].is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_sam_perturb(params[i], grads[i], rho_over_norm);
#endif
        } else {
            params[i].add_(grads[i], rho_over_norm);
        }
    }
    return backups;
}


// ═══════════════════════════════════════════════════════════════════════
//  Sharpness Compute + Restore
// ═══════════════════════════════════════════════════════════════════════

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
            // Just restore
            params[i].copy_(backups[i]);
            continue;
        }

        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = params[i].is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_sharpness_restore(
                params[i], sharpness_cache[i], backups[i],
                sam_grads[i], normal_grads[i]
            );
#endif
        } else {
            sharpness_cache[i] = (sam_grads[i] - normal_grads[i]).abs();
            params[i].copy_(backups[i]);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  pybind11 Bindings
// ═══════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SuperGrok v1.5 C++/CUDA operations";

    m.def("fused_step", &supergrok15_fused_step,
          "Fused optimizer step: mu + meta-net + gating + adam + progressive wd",
          py::arg("params"), py::arg("grads"),
          py::arg("exp_avgs"), py::arg("exp_avg_sqs"),
          py::arg("mus"), py::arg("sharpness_cache"),
          py::arg("steps"),
          py::arg("layer_alphas"), py::arg("layer_beta1s"),
          py::arg("W1"), py::arg("b1"), py::arg("W2"), py::arg("b2"),
          py::arg("rescale"), py::arg("hidden_dim"),
          py::arg("beta2"), py::arg("lr"), py::arg("wd_eff"), py::arg("eps"),
          py::arg("lamb"), py::arg("ramp"), py::arg("gate_temperature"),
          py::arg("grad_clip_norm"));

    m.def("sam_perturb_all", &supergrok15_sam_perturb_all,
          "SAM perturbation: perturb all params, return backups",
          py::arg("params"), py::arg("grads"), py::arg("rho"));

    m.def("sharpness_restore_all", &supergrok15_sharpness_restore_all,
          "Compute sharpness and restore params from backups",
          py::arg("params"), py::arg("sharpness_cache"),
          py::arg("backups"), py::arg("sam_grads"), py::arg("normal_grads"));
}
