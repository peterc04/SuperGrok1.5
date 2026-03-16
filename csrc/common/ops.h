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
#include "dispatch.h"

// Context struct for passing intermediate state between batched scan phases.
// Used by batched_step_setup_and_sort, generic_batched_precompute, and
// batched_step_scan_and_fused_elem to avoid duplicating setup code across tiers.
struct BatchedScanCtx {
    int num_params;
    int total_N;
    int max_N;
    std::vector<int> N_vec;
    std::vector<int> seg_offsets_cpu;
    torch::Tensor x_sorted_packed;
    torch::Tensor offsets_t;
    torch::Tensor initial_fwd;
    torch::Tensor initial_bwd;
    torch::Tensor final_fwd;
    torch::Tensor final_bwd;
    torch::Tensor fwd_scan_packed;
    torch::Tensor bwd_scan_packed;
    std::vector<torch::Tensor> unsort_idx_list;
};

// ═══════════════════════════════════════════════════════════════════════
//  CUDA Kernel Launchers (defined in respective .cu files)
// ═══════════════════════════════════════════════════════════════════════

#if defined(WITH_CUDA) || defined(WITH_HIP)

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

void launch_fused_sg11_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

float compute_cosine_gate_fused(
    torch::Tensor smart_grad, torch::Tensor mu, float gate_temp);

// NOTE: launch_sg11_sam_perturb and launch_sg11_sharpness_restore are defined
// in supergrok11_kernels.cu but not called from ops.cpp (v1.1 delegates to v1.5
// implementations).

// ── GrokAdamW (grokadamw_kernels.cu) ────────────────────────────────
void launch_fused_grokadamw_step(
    torch::Tensor param,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
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

void launch_fused_neuralgrok_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2);

// ── Prodigy (prodigy_kernels.cu) ────────────────────────────────────
void launch_fused_prodigy_step(
    torch::Tensor param,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor s, torch::Tensor grad,
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
// NOTE: launch_muon_momentum_normalize is defined in muon_kernels.cu but not
// called from ops.cpp (momentum + normalize done via ATen ops inline).

void launch_muon_fused_step(
    torch::Tensor param, torch::Tensor momentum_buffer, torch::Tensor grad,
    float lr, float momentum, float weight_decay, int ns_steps,
    float a, float b, float c);

void launch_muon_ns_combine(
    torch::Tensor X_out, torch::Tensor X, torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c);

void launch_muon_update(
    torch::Tensor param, torch::Tensor orth,
    float neg_lr_scale, float decay_factor);

// ── Distributed Scan (distributed_scan_kernels.cu) ──────────────────
void distributed_scan_local_with_summary(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val, torch::Tensor pre_dt_val,
    torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor summaries,
    torch::Tensor initial_state,
    int N_local, int d_inner, int d_state, int reverse);

void distributed_scan_apply_prefix(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val, torch::Tensor pre_dt_val,
    torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor rope_freq,
    torch::Tensor prefix_transforms, torch::Tensor scan_output,
    torch::Tensor initial_state,
    int N_local, int d_inner, int d_state, int reverse);

void distributed_scan_summary_prefix(
    torch::Tensor all_summaries, torch::Tensor prefix_out,
    int K, int d_inner, int half_d_state);

void distributed_scan_local_with_summary_bwd(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val, torch::Tensor pre_dt_val,
    torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor grad_output, torch::Tensor fwd_scan_output,
    torch::Tensor grad_pre_x, torch::Tensor grad_pre_dt,
    torch::Tensor grad_pre_B, torch::Tensor grad_pre_C,
    torch::Tensor grad_D, torch::Tensor bwd_summaries,
    torch::Tensor initial_state,
    int N_local, int d_inner, int d_state, int reverse);

void distributed_scan_apply_prefix_bwd(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val, torch::Tensor pre_dt_val,
    torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor rope_freq,
    torch::Tensor grad_output, torch::Tensor bwd_prefix_transforms,
    torch::Tensor grad_pre_x, torch::Tensor grad_pre_dt,
    torch::Tensor grad_pre_B, torch::Tensor grad_pre_C,
    int N_local, int d_inner, int d_state, int reverse);

void distributed_scan_summary_prefix_bwd(
    torch::Tensor all_bwd_summaries, torch::Tensor bwd_prefix_out,
    int K, int d_inner, int half_d_state);

// ── MoE Deep (moe_deep_kernels.cu) ─────────────────────────────────
void moe_dynamic_expert_load(
    torch::Tensor gate_logits, torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor loaded_W1, torch::Tensor loaded_b1,
    torch::Tensor loaded_W2, torch::Tensor loaded_b2,
    torch::Tensor active_expert_ids, torch::Tensor num_active_per_sample,
    float threshold, int N, int num_experts, int input_dim, int expert_dim);

torch::Tensor moe_dynamic_expert_fwd(
    torch::Tensor input, torch::Tensor gate_logits,
    torch::Tensor loaded_W1, torch::Tensor loaded_b1,
    torch::Tensor loaded_W2, torch::Tensor loaded_b2,
    torch::Tensor active_expert_ids, torch::Tensor num_active_per_sample,
    int N, int num_experts, int input_dim, int expert_dim);

void moe_dynamic_expert_bwd(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor gate_logits,
    torch::Tensor loaded_W1, torch::Tensor loaded_b1,
    torch::Tensor loaded_W2, torch::Tensor loaded_b2,
    torch::Tensor active_expert_ids, torch::Tensor num_active_per_sample,
    torch::Tensor grad_gate_logits, torch::Tensor grad_W1, torch::Tensor grad_b1,
    torch::Tensor grad_W2, torch::Tensor grad_b2,
    int N, int num_experts, int input_dim, int expert_dim);

void moe_filter_active_params(
    torch::Tensor params, torch::Tensor grads,
    torch::Tensor state_m, torch::Tensor state_v,
    torch::Tensor param_to_expert, torch::Tensor expert_active,
    torch::Tensor compact_params, torch::Tensor compact_grads,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices, torch::Tensor compact_count,
    int total_params);

void moe_scan_compacted(
    torch::Tensor compact_x, torch::Tensor compact_dt,
    torch::Tensor compact_B, torch::Tensor compact_C,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor final_state,
    torch::Tensor initial_state,
    int compact_N, int d_inner, int d_state);

void moe_scatter_results(
    torch::Tensor compact_params, torch::Tensor compact_state_m,
    torch::Tensor compact_state_v, torch::Tensor scatter_indices,
    torch::Tensor params, torch::Tensor state_m, torch::Tensor state_v,
    int compact_N);

void moe_count_expert_activations(
    torch::Tensor gate_logits, torch::Tensor expert_counts,
    float threshold, int N, int num_experts);

torch::Tensor moe_compute_load_balance_loss(
    torch::Tensor expert_counts, torch::Tensor gate_logits,
    int N, int num_experts);

void moe_apply_frequency_scaling(
    torch::Tensor expert_counts, torch::Tensor lr_scale,
    int num_experts, int total_activations,
    float min_scale, float max_scale, float smoothing);

// ── CDNA4 Kernels (cdna4_kernels.hip.cpp) ───────────────────────────
void cdna4_scan_local_with_summary(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val, torch::Tensor pre_dt_val,
    torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor h_state_fp6, torch::Tensor state_scale,
    torch::Tensor scan_output, torch::Tensor summary_M, torch::Tensor summary_b,
    int N_local, int d_inner, int d_state);

void cdna4_backward_fp6(
    torch::Tensor grad_output, torch::Tensor pre_x_val, torch::Tensor pre_z_val,
    torch::Tensor pre_dt_val, torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor saved_states_fp6, torch::Tensor state_scales,
    torch::Tensor grad_pre_x, torch::Tensor grad_pre_dt,
    torch::Tensor grad_pre_B, torch::Tensor grad_pre_C,
    torch::Tensor grad_D,
    int N, int d_inner, int d_state, int checkpoint_interval);

void cdna4_dynamic_expert_fp4(
    torch::Tensor scan_output, torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor gru_state,
    torch::Tensor all_expert_W1_fp4, torch::Tensor expert_b1,
    torch::Tensor all_expert_W2_fp4, torch::Tensor expert_b2,
    torch::Tensor expert_scale, torch::Tensor active_expert_indices,
    int num_active_experts, int N,
    float lr, float beta1, float beta2, float eps, float wd, float rescale,
    int expert_hidden, int num_experts);

void cdna4_persistent_scan_fused_elem(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val, torch::Tensor pre_dt_val,
    torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor param, torch::Tensor grad, torch::Tensor sort_indices,
    torch::Tensor expert_W1_fp4, torch::Tensor expert_b1,
    torch::Tensor expert_W2_fp4, torch::Tensor expert_b2,
    torch::Tensor expert_scale,
    torch::Tensor exp_avg_fp6, torch::Tensor exp_avg_sq_fp6,
    torch::Tensor state_scale_avg, torch::Tensor state_scale_sq,
    int N, int d_inner, int d_state,
    float lr, float beta1, float beta2, float eps, float wd, float rescale,
    int expert_hidden, int num_experts);

// ── CDNA4 FP4 Expert Kernels ─────────────────────────────────────────
void cdna4_fp4_expert_load(
    torch::Tensor weights_fp4, torch::Tensor scale_factors,
    torch::Tensor weights_fp32,
    int num_experts, int weight_numel, int packed_size);

void cdna4_fp4_expert_fwd(
    torch::Tensor input, torch::Tensor W1_fp4, torch::Tensor b1,
    torch::Tensor W2_fp4, torch::Tensor b2,
    torch::Tensor scale_W1, torch::Tensor scale_W2,
    torch::Tensor expert_assign, torch::Tensor output,
    int batch_size, int d_in, int expert_hidden, int d_out,
    int packed_W1_row, int packed_W2_row);

void cdna4_fp4_expert_bwd(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor hidden_acts,
    torch::Tensor W1_fp4, torch::Tensor W2_fp4,
    torch::Tensor scale_W1, torch::Tensor scale_W2,
    torch::Tensor expert_assign,
    torch::Tensor grad_input, torch::Tensor grad_W1_accum, torch::Tensor grad_W2_accum,
    torch::Tensor grad_b1, torch::Tensor grad_b2,
    uint32_t rng_seed,
    int batch_size, int d_in, int expert_hidden, int d_out,
    int packed_W1_row, int packed_W2_row);

void cdna4_fp4_quantize_experts(
    torch::Tensor weights_fp32, torch::Tensor weights_fp4,
    torch::Tensor scale_factors, uint32_t rng_seed,
    int num_experts, int weight_numel, int packed_size);

// ── CDNA4 FP6 State Kernels ─────────────────────────────────────────
void cdna4_fp6_state_pack(
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor exp_avg_fp6, torch::Tensor exp_avg_sq_fp6,
    torch::Tensor state_scale_avg, torch::Tensor state_scale_sq,
    int N);

void cdna4_fp6_state_unpack(
    torch::Tensor exp_avg_fp6, torch::Tensor exp_avg_sq_fp6,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor state_scale_avg, torch::Tensor state_scale_sq,
    int N);

void cdna4_fp6_adam_step(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg_fp6, torch::Tensor exp_avg_sq_fp6,
    torch::Tensor state_scale_avg, torch::Tensor state_scale_sq,
    float beta1, float beta2, float lr, float eps,
    float weight_decay, float bc1, float bc2, int N);

void cdna4_fp6_lamb_step(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg_fp6, torch::Tensor exp_avg_sq_fp6,
    torch::Tensor state_scale_avg, torch::Tensor state_scale_sq,
    torch::Tensor param_norm_out, torch::Tensor update_norm_out,
    float beta1, float beta2, float lr, float eps,
    float weight_decay, float bc1, float bc2, float trust_ratio, int N);

// ── CDNA4 2:4 Sparsity Kernels ──────────────────────────────────────
void cdna4_sparse24_select(
    torch::Tensor dense, torch::Tensor sparse_values, torch::Tensor metadata, int N);

void cdna4_sparse24_apply_mask(
    torch::Tensor grad, torch::Tensor metadata, int N);

void cdna4_sparse24_project(
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor metadata, int N);

void cdna4_sparse24_densify(
    torch::Tensor sparse_values, torch::Tensor metadata, torch::Tensor dense, int N);

// ── CDNA4 Fused Kernels ─────────────────────────────────────────────
void cdna4_fp4_sparse24_fused_expert(
    torch::Tensor input, torch::Tensor W1_fp4, torch::Tensor b1,
    torch::Tensor W2_fp4, torch::Tensor b2,
    torch::Tensor scale_W1, torch::Tensor scale_W2,
    torch::Tensor W1_sparse_meta, torch::Tensor W2_sparse_meta,
    torch::Tensor expert_assign, torch::Tensor output,
    int batch_size, int d_in, int expert_hidden, int d_out,
    int packed_W1_row, int packed_W2_row);

void cdna4_supergrok15_full_step(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg_fp6, torch::Tensor exp_avg_sq_fp6,
    torch::Tensor state_scale_avg, torch::Tensor state_scale_sq,
    torch::Tensor sparse_metadata, torch::Tensor expert_fp4_out,
    torch::Tensor expert_scale,
    float beta1, float beta2, float lr, float eps,
    float weight_decay, float bc1, float bc2,
    int N, int is_sparse, int is_expert);

// ── SuperGrok v2 Mamba-3+PEER (supergrok2_mamba_peer_kernels.cu) ──
void launch_mamba3_peer_step(
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

// Batched version: all parameters at once
void launch_mamba3_peer_batched_step(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
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
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

// ── SuperGrok v2 Bilevel Backward (supergrok2_mamba_peer_backward_kernels.cu)
void launch_mamba3_peer_bilevel_fwd_save(
    torch::Tensor grad, torch::Tensor sharpness,
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
    int d_model, int d_state, int d_inner,
    torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out,
    torch::Tensor fwd_final_state, torch::Tensor bwd_final_state,
    torch::Tensor fwd_saved_states, torch::Tensor fwd_saved_x_branch,
    torch::Tensor fwd_saved_z, torch::Tensor fwd_saved_dt,
    torch::Tensor bwd_saved_states, torch::Tensor bwd_saved_x_branch,
    torch::Tensor bwd_saved_z, torch::Tensor bwd_saved_dt,
    torch::Tensor x_sorted, torch::Tensor sort_indices,
    torch::Tensor fwd_initial_state, torch::Tensor bwd_initial_state,
    int checkpoint_interval);

void launch_mamba3_peer_backward(
    torch::Tensor d_smart_grad, torch::Tensor grad, torch::Tensor sharpness,
    float rescale,
    torch::Tensor sort_indices, torch::Tensor x_sorted,
    torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out,
    torch::Tensor fwd_saved_states, torch::Tensor fwd_saved_x_branch,
    torch::Tensor fwd_saved_z, torch::Tensor fwd_saved_dt,
    torch::Tensor bwd_saved_states, torch::Tensor bwd_saved_x_branch,
    torch::Tensor bwd_saved_z, torch::Tensor bwd_saved_dt,
    torch::Tensor gru_input, torch::Tensor gru_h_old,
    torch::Tensor gru_z_gate, torch::Tensor gru_r_gate,
    torch::Tensor gru_h_tilde,
    torch::Tensor peer_input,
    torch::Tensor expert_indices, torch::Tensor routing_weights,
    torch::Tensor saved_z_hidden,
    torch::Tensor saved_scores_a, torch::Tensor saved_scores_b,
    torch::Tensor saved_top_a_idx, torch::Tensor saved_top_b_idx,
    torch::Tensor saved_soft_a, torch::Tensor saved_soft_b,
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
    torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_W2,
    torch::Tensor expert_b1_in, torch::Tensor expert_b2_in,
    torch::Tensor input_proj_W,
    torch::Tensor mamba_fwd_init_state,  // [d_inner, d_state] or empty
    torch::Tensor mamba_bwd_init_state,  // [d_inner, d_state] or empty
    torch::Tensor d_mamba_fwd_in_proj, torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b, torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj, torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_fwd_out_proj,
    torch::Tensor d_mamba_bwd_in_proj, torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b, torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj, torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_mamba_bwd_out_proj,
    torch::Tensor d_gru_Wz, torch::Tensor d_gru_bz,
    torch::Tensor d_gru_Wr, torch::Tensor d_gru_br,
    torch::Tensor d_gru_Wh, torch::Tensor d_gru_bh,
    torch::Tensor d_peer_query_Ws,
    torch::Tensor d_prod_keys_A, torch::Tensor d_prod_keys_B,
    torch::Tensor d_expert_W1, torch::Tensor d_expert_b1,
    torch::Tensor d_expert_W2, torch::Tensor d_expert_b2,
    torch::Tensor d_input_proj_W, torch::Tensor d_input_proj_b,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int gru_input_dim,
    int num_heads, int topk, int pk_dim,
    int expert_hidden, int peer_input_dim, int num_experts,
    int checkpoint_interval);

// Batched bilevel forward-save
void launch_mamba3_peer_bilevel_fwd_save_batched(
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
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
    int d_model, int d_state, int d_inner,
    torch::Tensor fwd_scan_out_packed, torch::Tensor bwd_scan_out_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int checkpoint_interval);

// Batched bilevel backward scan
void launch_mamba3_peer_backward_batched(
    torch::Tensor d_fwd_scan_out_packed, torch::Tensor d_bwd_scan_out_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor offsets_t,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor d_mamba_fwd_in_proj, torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b, torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj, torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_bwd_in_proj, torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b, torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj, torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_x_sorted_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int d_model, int d_state, int d_inner, int num_params,
    int checkpoint_interval);

#ifdef WITH_CUDA  // NVIDIA-specific tier declarations

// ── SuperGrok v2 Ampere Tier (csrc/cuda/sm_80/) ─────────────────────
void launch_mamba3_peer_step_ampere(
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

void launch_mamba3_peer_batched_step_ampere(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
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
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

void launch_mamba3_peer_bilevel_fwd_save_batched_ampere(
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
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
    int d_model, int d_state, int d_inner,
    torch::Tensor fwd_scan_out_packed, torch::Tensor bwd_scan_out_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int checkpoint_interval);

void launch_mamba3_peer_backward_batched_ampere(
    torch::Tensor d_fwd_scan_out_packed, torch::Tensor d_bwd_scan_out_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor offsets_t,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor d_mamba_fwd_in_proj, torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b, torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj, torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_bwd_in_proj, torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b, torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj, torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_x_sorted_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int d_model, int d_state, int d_inner, int num_params,
    int checkpoint_interval);

// ── SuperGrok v2 Hopper Tier (csrc/cuda/sm_90/) ─────────────────────
void launch_mamba3_peer_step_hopper(
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

void launch_mamba3_peer_batched_step_hopper(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
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
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

void launch_mamba3_peer_bilevel_fwd_save_batched_hopper(
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
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
    int d_model, int d_state, int d_inner,
    torch::Tensor fwd_scan_out_packed, torch::Tensor bwd_scan_out_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int checkpoint_interval);

void launch_mamba3_peer_backward_batched_hopper(
    torch::Tensor d_fwd_scan_out_packed, torch::Tensor d_bwd_scan_out_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor offsets_t,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor d_mamba_fwd_in_proj, torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b, torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj, torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_bwd_in_proj, torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b, torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj, torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_x_sorted_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int d_model, int d_state, int d_inner, int num_params,
    int checkpoint_interval);

// ── SuperGrok v2 Blackwell Tier (csrc/cuda/sm_100/) ──────────────────
void launch_mamba3_peer_step_blackwell(
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

void launch_mamba3_peer_batched_step_blackwell(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
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
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

void launch_mamba3_peer_bilevel_fwd_save_batched_blackwell(
    std::vector<torch::Tensor> grads, std::vector<torch::Tensor> sharpness_list,
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
    int d_model, int d_state, int d_inner,
    torch::Tensor fwd_scan_out_packed, torch::Tensor bwd_scan_out_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int checkpoint_interval);

void launch_mamba3_peer_backward_batched_blackwell(
    torch::Tensor d_fwd_scan_out_packed, torch::Tensor d_bwd_scan_out_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor offsets_t,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor d_mamba_fwd_in_proj, torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b, torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj, torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_bwd_in_proj, torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b, torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj, torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_x_sorted_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int d_model, int d_state, int d_inner, int num_params,
    int checkpoint_interval);

// ── Quantization Kernels (csrc/quantization/quantization_kernels.cu) ──
std::vector<torch::Tensor> quantize_fp8_e4m3(torch::Tensor input);
torch::Tensor dequantize_fp8_e4m3(torch::Tensor input, torch::Tensor scale, int64_t numel);

std::vector<torch::Tensor> quantize_int8(torch::Tensor input);
torch::Tensor dequantize_int8(torch::Tensor input, torch::Tensor scale, int64_t numel);

std::vector<torch::Tensor> quantize_int4(torch::Tensor input);
torch::Tensor dequantize_int4(torch::Tensor input, torch::Tensor scales, int64_t numel);

std::vector<torch::Tensor> quantize_mxfp4(torch::Tensor input);
torch::Tensor dequantize_mxfp4(torch::Tensor input, torch::Tensor block_scales, int64_t numel);

std::vector<torch::Tensor> quantize_nvfp4(torch::Tensor input);
torch::Tensor dequantize_nvfp4(torch::Tensor input, torch::Tensor block_scales, int64_t numel);

// ── Ampere Tier Metanet Wrappers (csrc/cuda/sm_80/metanet_optimizers_sm80.cu) ──
void launch_fused_supergrok15_full_step_ampere(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim,
    float alpha, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2);

void launch_fused_sg11_full_step_ampere(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim,
    float alpha, float lamb_eff, float gate_val,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2);

void launch_fused_neuralgrok_full_step_ampere(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_scale, float beta_shift,
    int hidden_dim, float rescale,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2);

// ── Muon Ampere (csrc/cuda/sm_80/muon_sm80.cu) — TF32 cuBLAS math mode ──
void launch_muon_fused_step_ampere(
    torch::Tensor param, torch::Tensor momentum_buffer, torch::Tensor grad,
    float lr, float momentum, float weight_decay, int ns_steps,
    float a, float b, float c);

// ── Muon Hopper (csrc/cuda/sm_90/muon_sm90.cu) — FP8 E4M3 Newton-Schulz GEMMs ──
void launch_muon_fused_step_hopper(
    torch::Tensor param, torch::Tensor momentum_buffer, torch::Tensor grad,
    float lr, float momentum, float weight_decay, int ns_steps,
    float a, float b, float c);

// ── Hopper Tier Metanet (csrc/cuda/sm_90/metanet_optimizers_sm90.cu) ──
// Delegates to Ampere cp.async kernels — meta-net MLPs (hidden_dim=32)
// are too small to benefit from FP8.
void launch_fused_supergrok15_full_step_hopper(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_fused_sg11_full_step_hopper(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_fused_neuralgrok_full_step_hopper(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2);

// ── Generated SG2 Kernel Dispatchers (csrc/cuda/generated/sg2_dispatch.cu) ──


void launch_sg2_fused_elem_generated(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor gru_state,
    torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out,
    torch::Tensor out_proj_fwd_W, torch::Tensor out_proj_bwd_W,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    torch::Tensor gate_logits,
    float rescale, float alpha, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, float gate_threshold, float gate_scale,
    torch::Tensor expert_counts,
    int d_model, int d_inner_val,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    int state_prec_int, int expert_prec_int,
    bool moe_sparse);

void launch_sg2_metanet_only_generated(
    torch::Tensor grad,
    torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out,
    torch::Tensor out_proj_fwd_W, torch::Tensor out_proj_bwd_W,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor gru_state,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    torch::Tensor meta_output,
    float rescale,
    torch::Tensor expert_counts,
    int d_model, int d_inner_val,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    int expert_prec_int);

void launch_sg2_adam_only_generated(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor smart_grad, torch::Tensor mu,
    float lamb_eff, float beta1, float beta2, float lr, float wd_eff,
    float eps, float bc1, float bc2,
    int N, StatePrecision state_prec);

void launch_sg2_persistent_scan_generated(
    torch::Tensor x_sorted,
    torch::Tensor in_proj_W, torch::Tensor dt_proj_W, torch::Tensor dt_proj_b,
    torch::Tensor B_proj_W, torch::Tensor C_proj_W,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor scan_state,
    torch::Tensor initial_state,
    int N, int d_model, int d_inner_val, int d_state, int reverse,
    int state_prec_int);

void launch_sg2_scan_d16(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val,
    torch::Tensor pre_dt_val, torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor final_state,
    torch::Tensor initial_state,
    int N, int d_state, int reverse);

// ── Generated GrokAdamW Q4 Dispatcher (csrc/cuda/generated/grokadamw_generated.cu) ──
void launch_grokadamw_q4_step(
    torch::Tensor param,
    torch::Tensor exp_avg_q, torch::Tensor exp_avg_sq_bf16,
    torch::Tensor ema_bf16, torch::Tensor exp_avg_scales,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2, float lr, float wd,
    float eps, float bc1, float bc2);

// ── Generated Compute Absmax Scale (csrc/cuda/generated/compute_absmax_scale_kernel.cu) ──
void launch_compute_absmax_scale(
    torch::Tensor input, torch::Tensor scale_out,
    float max_representable);

// ── Generated Muon Update with Streaming (csrc/cuda/generated/muon_update_generated.cu) ──
void launch_muon_update_stream(
    torch::Tensor param, torch::Tensor orth,
    float neg_lr_scale, float decay_factor);

#endif  // WITH_CUDA (NVIDIA-specific tiers)

#ifdef WITH_HIP  // AMD-specific tier declarations

// ── SuperGrok v2 CDNA2 Tier (csrc/hip/cdna2/) ──────────────────────
void launch_mamba3_peer_step_cdna2(
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

void launch_mamba3_peer_batched_step_cdna2(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
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
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

void launch_mamba3_peer_bilevel_fwd_save_batched_cdna2(
    std::vector<torch::Tensor> grads, std::vector<torch::Tensor> sharpness_list,
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
    int d_model, int d_state, int d_inner,
    torch::Tensor fwd_scan_out_packed, torch::Tensor bwd_scan_out_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int checkpoint_interval);

void launch_mamba3_peer_backward_batched_cdna2(
    torch::Tensor d_fwd_scan_out_packed, torch::Tensor d_bwd_scan_out_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor offsets_t,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor d_mamba_fwd_in_proj, torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b, torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj, torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_bwd_in_proj, torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b, torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj, torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_x_sorted_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int d_model, int d_state, int d_inner, int num_params,
    int checkpoint_interval);

// ── SuperGrok v2 CDNA3 Tier (csrc/hip/cdna3/) ──────────────────────
void launch_mamba3_peer_step_cdna3(
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

void launch_mamba3_peer_batched_step_cdna3(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
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
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

void launch_mamba3_peer_bilevel_fwd_save_batched_cdna3(
    std::vector<torch::Tensor> grads, std::vector<torch::Tensor> sharpness_list,
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
    int d_model, int d_state, int d_inner,
    torch::Tensor fwd_scan_out_packed, torch::Tensor bwd_scan_out_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int checkpoint_interval);

void launch_mamba3_peer_backward_batched_cdna3(
    torch::Tensor d_fwd_scan_out_packed, torch::Tensor d_bwd_scan_out_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor offsets_t,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor d_mamba_fwd_in_proj, torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b, torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj, torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_bwd_in_proj, torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b, torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj, torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_x_sorted_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int d_model, int d_state, int d_inner, int num_params,
    int checkpoint_interval);

// ── Muon CDNA3 (csrc/hip/cdna3/muon_cdna3.hip.cpp) — delegates to generic ──
// CDNA3 BF16 MFMA marginal for Newton-Schulz on typical param matrices.
void launch_muon_fused_step_cdna3(
    torch::Tensor param, torch::Tensor momentum_buffer, torch::Tensor grad,
    float lr, float momentum, float weight_decay, int ns_steps,
    float a, float b, float c);

// ── CDNA3 Tier Metanet (csrc/hip/cdna3/metanet_optimizers_cdna3.hip.cpp) ──
// Delegates to generic — meta-net MLPs too small for MFMA benefit.
void launch_fused_supergrok15_full_step_cdna3(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_fused_sg11_full_step_cdna3(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_fused_neuralgrok_full_step_cdna3(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2);

#endif  // WITH_HIP (AMD-specific tiers)

#endif  // WITH_CUDA || WITH_HIP


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

// ── SuperGrok v2 Mamba-3+PEER ────────────────────────────────────
void supergrok2_mamba_peer_step(
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

// Batched SuperGrok v2 Mamba-PEER step (multiple params in one call)
void supergrok2_mamba_peer_batched_step(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
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
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

// ── CPU Fused Scan+Elem (sg2_fused_scan_elem_cpu.cpp) ───────────────
void cpu_sg2_fused_scan_elem(
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

void cpu_sg2_fused_scan_elem_q4(
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

// ── CPU MoE Filter (csrc/cpu/moe_cpu.cpp) ───────────────────────────
void cpu_moe_filter_active_params(
    torch::Tensor all_grads, torch::Tensor active_mask,
    torch::Tensor compact_grads, torch::Tensor compact_indices,
    torch::Tensor N_active_out);

// Batched scan pipeline: setup/sort → precompute → scan+fused_elem
// These allow tier-specific precompute (FP8, BF16) with shared scan+fused_elem.
BatchedScanCtx batched_step_setup_and_sort(
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    int d_model, int d_state, int d_inner);

void generic_batched_precompute(
    const BatchedScanCtx& ctx,
    torch::Tensor in_proj_W, torch::Tensor dt_proj_W, torch::Tensor dt_proj_b,
    torch::Tensor B_proj_W, torch::Tensor C_proj_W,
    torch::Tensor& pre_x, torch::Tensor& pre_z, torch::Tensor& pre_dt,
    torch::Tensor& pre_B, torch::Tensor& pre_C,
    int d_model, int d_inner, int d_state);

void batched_step_scan_and_fused_elem(
    BatchedScanCtx& ctx,
    torch::Tensor fwd_pre_x, torch::Tensor fwd_pre_z, torch::Tensor fwd_pre_dt,
    torch::Tensor fwd_pre_B, torch::Tensor fwd_pre_C,
    torch::Tensor bwd_pre_x, torch::Tensor bwd_pre_z, torch::Tensor bwd_pre_dt,
    torch::Tensor bwd_pre_B, torch::Tensor bwd_pre_C,
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
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
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

// Ampere version: uses cp.async scan + cp.async fused_elem kernels
void ampere_batched_scan_and_fused_elem(
    BatchedScanCtx& ctx,
    torch::Tensor fwd_pre_x, torch::Tensor fwd_pre_z, torch::Tensor fwd_pre_dt,
    torch::Tensor fwd_pre_B, torch::Tensor fwd_pre_C,
    torch::Tensor bwd_pre_x, torch::Tensor bwd_pre_z, torch::Tensor bwd_pre_dt,
    torch::Tensor bwd_pre_B, torch::Tensor bwd_pre_C,
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
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
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);
