/*
 * Grokking Optimizers — C++ Operations Header
 *
 * Declares kernel launchers and high-level optimizer step functions for:
 *   - SuperGrok v1.5 (fused mu + meta-net + gating + adam + wd)
 *   - SuperGrok v2   (sparse attention meta-net + adam + wd)
 *   - SuperGrok v1.1 (meta-net + cosine gating + adam + wd)
 *   - GrokAdamW      (fused EMA filter + amplification + adam)
 *   - NeuralGrok     (fused MLP amplifier + adam)
 *   - Prodigy        (distance-aware self-tuning adam)
 *   - Grokfast       (fused EMA + gradient amplification)
 *   - Lion           (fused momentum interp + sign + update + decay)
 *   - LookSAM        (SAM perturbation/restore + direction + adjust)
 *   - Muon           (momentum + Newton-Schulz + update)
 *
 * All kernels support FP32, FP16, and BF16 parameter tensors.
 * Optimizer state (moments, EMA buffers) is always FP32 for stability.
 */
#pragma once
#include <torch/extension.h>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════
//  CUDA Kernel Launchers (defined in respective .cu files)
// ═══════════════════════════════════════════════════════════════════════

#ifdef WITH_CUDA

// ── SuperGrok v1.5 (supergrok15_kernels.cu) ─────────────────────────
void launch_fused_mu_metanet(
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor smart_grad, float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim);

void launch_fused_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor smart_grad, torch::Tensor mu,
    float lamb_eff, float beta1, float beta2, float lr, float wd_eff,
    float eps, float bc1, float bc2);

void launch_sam_perturb(torch::Tensor param, torch::Tensor grad, float rho_over_norm);

void launch_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness, torch::Tensor backup,
    torch::Tensor sam_grad, torch::Tensor normal_grad);

void launch_fused_supergrok15_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

// ── SuperGrok v2 (supergrok2_kernels.cu) ────────────────────────────
void launch_dsa_project(
    torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor idx_q, torch::Tensor idx_k,
    torch::Tensor W_q, torch::Tensor b_q,
    torch::Tensor W_k, torch::Tensor b_k,
    torch::Tensor W_v, torch::Tensor b_v,
    torch::Tensor W_iq, torch::Tensor W_ik,
    int d_head, int n_idx_heads);

void launch_dsa_indexer_topk(
    torch::Tensor idx_q, torch::Tensor idx_k,
    torch::Tensor w_idx, torch::Tensor selected_indices,
    int n_idx_heads, int top_k);

void launch_dsa_sparse_attention(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor selected_indices,
    torch::Tensor grad, torch::Tensor smart_grad,
    torch::Tensor W_out, torch::Tensor b_out,
    float rescale, int d_head, int top_k);

// ── SuperGrok v1.1 (supergrok11_kernels.cu) ─────────────────────────
void launch_sg11_mu_metanet(
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor smart_grad, float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim);

void launch_sg11_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor smart_grad, torch::Tensor mu,
    float lamb_eff, float beta1, float beta2, float lr, float wd_eff,
    float eps, float bc1, float bc2);

void launch_sg11_sam_perturb(torch::Tensor param, torch::Tensor grad, float rho_over_norm);

void launch_sg11_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness, torch::Tensor backup,
    torch::Tensor sam_grad, torch::Tensor normal_grad);

float compute_cosine_gate(torch::Tensor smart_grad, torch::Tensor mu, float gate_temp);

// ── GrokAdamW (grokadamw_kernels.cu) ────────────────────────────────
void launch_fused_grokadamw_step(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema,
    float alpha, float lamb,
    float beta1, float beta2, float lr, float wd,
    float eps, float bc1, float bc2);

// ── NeuralGrok (neuralgrok_kernels.cu) ──────────────────────────────
void launch_fused_neuralgrok_amplifier(
    torch::Tensor grad, torch::Tensor amplified_grad,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp,
    int hidden_dim);

void launch_fused_neuralgrok_adam(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor amplified_grad,
    float beta1, float beta2, float lr, float wd,
    float eps, float bc1, float bc2);

// ── Prodigy (prodigy_kernels.cu) ────────────────────────────────────
void launch_fused_prodigy_step(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor s,
    float d_lr, float beta1, float beta2, float lr, float wd,
    float eps, float bc1, float bc2);

void launch_prodigy_dlr_reduce(
    torch::Tensor grad, torch::Tensor param, torch::Tensor param_init,
    torch::Tensor s, torch::Tensor numerator_out, torch::Tensor denominator_out);

// ── Grokfast (grokfast_kernels.cu) ──────────────────────────────────
void launch_fused_grokfast_ema(
    torch::Tensor grad, torch::Tensor ema,
    float alpha, float lamb);

// ── Lion (lion_kernels.cu) ──────────────────────────────────────────
void launch_fused_lion_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,
    float lr, float beta1, float beta2, float wd);

// ── LookSAM (looksam_kernels.cu) ───────────────────────────────────
void launch_looksam_direction(
    torch::Tensor v_dir, torch::Tensor sam_grad, torch::Tensor normal_grad,
    float inv_norm);

void launch_looksam_adjust(
    torch::Tensor grad, torch::Tensor v_dir, float la_times_gnorm);

void launch_looksam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm);

void launch_looksam_restore(
    torch::Tensor param, torch::Tensor backup);

// ── Muon (muon_kernels.cu) ─────────────────────────────────────────
void launch_muon_momentum_normalize(
    torch::Tensor buf, torch::Tensor X, torch::Tensor grad,
    float momentum, float inv_norm);

void launch_muon_ns_combine(
    torch::Tensor X_out, torch::Tensor X, torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c);

void launch_muon_update(
    torch::Tensor param, torch::Tensor orth,
    float neg_lr_scale, float decay_factor);

#endif  // WITH_CUDA


// ═══════════════════════════════════════════════════════════════════════
//  High-Level C++ Operations (called from Python via pybind11)
// ═══════════════════════════════════════════════════════════════════════

// ── SuperGrok v1.5 ───────────────────────────────────────────────────
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
    float grad_clip_norm);

std::vector<torch::Tensor> supergrok15_sam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho);

void supergrok15_sharpness_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<torch::Tensor>& backups,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads);

// ── SuperGrok v2 ────────────────────────────────────────────────────
void supergrok2_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& mus,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<int64_t>& steps,
    std::vector<float>& layer_alphas,
    std::vector<float>& layer_beta1s,
    // DSA weights
    torch::Tensor W_q, torch::Tensor b_q,
    torch::Tensor W_k, torch::Tensor b_k,
    torch::Tensor W_v, torch::Tensor b_v,
    torch::Tensor W_iq, torch::Tensor W_ik,
    torch::Tensor w_idx,
    torch::Tensor W_out, torch::Tensor b_out,
    float rescale, int d_head, int n_idx_heads, int top_k,
    float beta2, float lr, float wd_eff, float eps,
    float lamb, float ramp, float gate_signal,
    float grad_clip_norm);

// ── SuperGrok v1.1 ──────────────────────────────────────────────────
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
    float grad_clip_norm);

std::vector<torch::Tensor> supergrok11_sam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho);

void supergrok11_sharpness_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<torch::Tensor>& backups,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads);

// ── GrokAdamW ───────────────────────────────────────────────────────
void grokadamw_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<int64_t>& steps,
    float alpha, float lamb,
    float beta1, float beta2, float lr, float wd,
    float eps, float grad_clip_norm);

// ── NeuralGrok ──────────────────────────────────────────────────────
void neuralgrok_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<int64_t>& steps,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float wd,
    float eps, float grad_clip_norm);

// ── Prodigy ─────────────────────────────────────────────────────────
float prodigy_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_bufs,
    std::vector<torch::Tensor>& param_inits,
    std::vector<int64_t>& steps,
    float d_lr,
    float beta1, float beta2, float lr, float wd,
    float eps);

// ── Grokfast ────────────────────────────────────────────────────────
void grokfast_fused_step(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& ema_bufs,
    float alpha, float lamb);

// ── Lion ────────────────────────────────────────────────────────────
void lion_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    float lr, float beta1, float beta2, float wd);

// ── LookSAM ────────────────────────────────────────────────────────
std::vector<torch::Tensor> looksam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho);

void looksam_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& backups);

void looksam_compute_directions(
    std::vector<torch::Tensor>& v_dirs,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads);

void looksam_adjust_grads(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& v_dirs,
    float la);

// ── Muon ────────────────────────────────────────────────────────────
void muon_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& bufs,
    float momentum, float lr, float wd, int ns_steps);
