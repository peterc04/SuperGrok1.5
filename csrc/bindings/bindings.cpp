// =====================================================================
// bindings.cpp — pybind11 module + all per-optimizer dispatchers
//
// Consolidated from 16 per-optimizer dispatcher files + module.cpp.
// Each section below preserves the original file's content verbatim;
// the only changes are: (1) #include lines for the deleted internal
// headers were stripped, (2) a single set of shared #includes lives at
// the top of this file.
//
// Pybind11 registration order is preserved from the original module.cpp.
// =====================================================================

#include "helpers.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <string>
#include <stdexcept>


// ─── csrc/bindings/grokadamw.cpp ───────────────────────────────────
// =====================================================================
// bindings/grokadamw.cpp — runtime dispatch to per-arch GrokAdamW launchers
//
// Each arch has its launcher symbols inside a sg::<arch> namespace
// defined in csrc/kernels/{cuda/<sm>,hip/gfx942}/grokadamw_<arch>.cu.
// This dispatcher picks one based on detect_arch() and raises if the
// detected arch is unsupported.
//
// Pybind11 registration is done from csrc/bindings/module.cpp.
// =====================================================================


#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------
// Forward-declare the launcher signatures inside each arch namespace.
// Signatures must match the definitions in the per-arch .cu files.
// ---------------------------------------------------------------------

namespace sg {

#define DECLARE_GROKADAMW_LAUNCHERS(NS) \
 namespace NS { \
 void launch_fused_grokadamw_step( \
 torch::Tensor param, torch::Tensor exp_avg, \
 torch::Tensor exp_avg_sq, torch::Tensor ema, \
 torch::Tensor grad, \
 float alpha, float lamb, \
 float beta1, float beta2, \
 float lr, float weight_decay, \
 float eps, float bc1, float bc2); \
 void launch_fused_grokadamw_clip_step( \
 torch::Tensor param, torch::Tensor exp_avg, \
 torch::Tensor exp_avg_sq, torch::Tensor ema, \
 torch::Tensor grad, \
 float alpha, float lamb, \
 float beta1, float beta2, \
 float lr, float weight_decay, \
 float eps, float bc1, float bc2, \
 float clip_threshold); \
 void launch_fused_grokadamw_step_q3( \
 torch::Tensor param, \
 torch::Tensor exp_avg_int8, \
 torch::Tensor exp_avg_scales, \
 torch::Tensor exp_avg_sq_bf16, \
 torch::Tensor ema_bf16, \
 torch::Tensor grad, \
 float alpha, float lamb, \
 float beta1, float beta2, \
 float lr, float weight_decay, \
 float eps, float bc1, float bc2, \
 unsigned global_step); \
 }

DECLARE_GROKADAMW_LAUNCHERS(sm90)

DECLARE_GROKADAMW_LAUNCHERS(gfx942)

#undef DECLARE_GROKADAMW_LAUNCHERS

// Multi-tensor variants live in csrc/kernels/<arch>/multi_tensor_<arch>.cu.
// One declaration per supported arch.
#define DECLARE_MT_GROKADAMW(NS) \
 namespace NS { \
 void launch_multi_tensor_grokadamw( \
 std::vector<torch::Tensor>& params, \
 std::vector<torch::Tensor>& exp_avgs, \
 std::vector<torch::Tensor>& exp_avg_sqs, \
 std::vector<torch::Tensor>& emas, \
 std::vector<torch::Tensor>& grads, \
 std::vector<float>& bc1s, std::vector<float>& bc2s, \
 float alpha, float lamb, float beta1, float beta2, \
 float lr, float wd, float eps); \
 void launch_fused_adamw_simple( \
 std::vector<torch::Tensor>& params, \
 std::vector<torch::Tensor>& exp_avgs, \
 std::vector<torch::Tensor>& exp_avg_sqs, \
 std::vector<torch::Tensor>& grads, \
 std::vector<int64_t>& steps, \
 float beta1, float beta2, float lr, float wd, float eps); \
 }

DECLARE_MT_GROKADAMW(sm90) 

DECLARE_MT_GROKADAMW(gfx942) 
#undef DECLARE_MT_GROKADAMW

// ---------------------------------------------------------------------
// Public entry points called from Python.
// ---------------------------------------------------------------------

#define DISPATCH_GROKADAMW(METHOD, ...) \
 do { \
 const int a = sg::detect_arch(); \
 switch (a) { \

 case 90: return sg::sm90::METHOD(__VA_ARGS__); \

 case 942: return sg::gfx942::METHOD(__VA_ARGS__); \
 default: \
 throw std::runtime_error( \
 "GrokAdamW dispatch: unsupported arch " + \
 std::to_string(a)); \
 } \
 } while (0)

void grokadamw_step(
 torch::Tensor param, torch::Tensor exp_avg,
 torch::Tensor exp_avg_sq, torch::Tensor ema,
 torch::Tensor grad,
 float alpha, float lamb,
 float beta1, float beta2,
 float lr, float weight_decay,
 float eps, float bc1, float bc2)
{
 DISPATCH_GROKADAMW(launch_fused_grokadamw_step,
 param, exp_avg, exp_avg_sq, ema, grad,
 alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}

void grokadamw_clip_step(
 torch::Tensor param, torch::Tensor exp_avg,
 torch::Tensor exp_avg_sq, torch::Tensor ema,
 torch::Tensor grad,
 float alpha, float lamb,
 float beta1, float beta2,
 float lr, float weight_decay,
 float eps, float bc1, float bc2,
 float clip_threshold)
{
 DISPATCH_GROKADAMW(launch_fused_grokadamw_clip_step,
 param, exp_avg, exp_avg_sq, ema, grad,
 alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
 clip_threshold);
}

void grokadamw_step_q3(
 torch::Tensor param,
 torch::Tensor exp_avg_int8,
 torch::Tensor exp_avg_scales,
 torch::Tensor exp_avg_sq_bf16,
 torch::Tensor ema_bf16,
 torch::Tensor grad,
 float alpha, float lamb,
 float beta1, float beta2,
 float lr, float weight_decay,
 float eps, float bc1, float bc2,
 unsigned global_step)
{
 DISPATCH_GROKADAMW(launch_fused_grokadamw_step_q3,
 param, exp_avg_int8, exp_avg_scales, exp_avg_sq_bf16, ema_bf16,
 grad, alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
 global_step);
}

// ---------------------------------------------------------------------
// High-level vector-signature entry point — restored from pre-refactor
// csrc/common/ops.cpp::grokadamw_fused_step. Loops over per-param
// scalars (bc1, bc2) on the host, then calls the multi-tensor launcher
// in the arch-detected namespace.
// ---------------------------------------------------------------------
void grokadamw_fused_step(
 std::vector<torch::Tensor>& params,
 std::vector<torch::Tensor>& grads,
 std::vector<torch::Tensor>& exp_avgs,
 std::vector<torch::Tensor>& exp_avg_sqs,
 std::vector<torch::Tensor>& emas,
 std::vector<int64_t>& steps,
 float alpha, float lamb_grok,
 float beta1, float beta2, float lr, float wd,
 float eps, float grad_clip_norm
) {
 const size_t n_params = params.size();
 clip_grad_norms_device_side(grads, n_params, grad_clip_norm);
 if (n_params == 0) return;

 std::vector<torch::Tensor> vp, vg, vea, veas, vema;
 std::vector<float> bc1_vec, bc2_vec;
 for (size_t i = 0; i < n_params; i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 // NOTE: the Python optimizer owns the step counter (state["step"] += 1
 // before this call), matching fused_adamw_simple_step and
 // grokfast_fused_ema_adam_step. Do NOT increment here — `steps` is a
 // pybind-copied vector so an increment would not persist back to Python
 // anyway, and incrementing made bias correction permanently off-by-one
 // (bc computed with step+1).
 float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
 float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));
 vp.push_back(params[i]); vg.push_back(grads[i]);
 vea.push_back(exp_avgs[i]); veas.push_back(exp_avg_sqs[i]);
 vema.push_back(emas[i]);
 bc1_vec.push_back(bc1); bc2_vec.push_back(bc2);
 }
 if (vp.empty()) return;

 DISPATCH_GROKADAMW(launch_multi_tensor_grokadamw,
 vp, vea, veas, vema, vg, bc1_vec, bc2_vec,
 alpha, lamb_grok, beta1, beta2, lr, wd, eps);
}

// Shared simple AdamW (used by Muon and LookSAM 1D-param paths).
void fused_adamw_simple_step(
 std::vector<torch::Tensor>& params,
 std::vector<torch::Tensor>& grads,
 std::vector<torch::Tensor>& exp_avgs,
 std::vector<torch::Tensor>& exp_avg_sqs,
 std::vector<int64_t>& steps,
 float beta1, float beta2, float lr, float wd, float eps
) {
 if (params.empty()) return;
 DISPATCH_GROKADAMW(launch_fused_adamw_simple,
 params, exp_avgs, exp_avg_sqs, grads, steps,
 beta1, beta2, lr, wd, eps);
}

#undef DISPATCH_GROKADAMW

} // namespace sg


// ─── csrc/bindings/grokfast.cpp ────────────────────────────────────
// bindings/grokfast.cpp — runtime dispatch to per-arch Grokfast launchers.

#include <cmath>
#include <vector>

namespace sg {

#define DECLARE_GROKFAST(NS) \
 namespace NS { \
 void launch_fused_grokfast_ema( \
 torch::Tensor grad, torch::Tensor ema, \
 float alpha, float lamb); \
 void launch_fused_grokfast_adam( \
 torch::Tensor param, torch::Tensor exp_avg, \
 torch::Tensor exp_avg_sq, torch::Tensor ema, \
 torch::Tensor grad, \
 float alpha, float lamb, \
 float beta1, float beta2, \
 float lr, float weight_decay, \
 float eps, float bc1, float bc2); \
 }

 DECLARE_GROKFAST(sm90)
 DECLARE_GROKFAST(gfx942)

#undef DECLARE_GROKFAST

// Multi-tensor EMA launcher (in csrc/kernels/<arch>/multi_tensor_<arch>.cu).
#define DECLARE_MT_GROKFAST(NS) \
 namespace NS { \
 void launch_multi_tensor_grokfast_ema( \
 std::vector<torch::Tensor>& grads, \
 std::vector<torch::Tensor>& ema_bufs, \
 float alpha, float lamb); \
 }

 DECLARE_MT_GROKFAST(sm90)

DECLARE_MT_GROKFAST(gfx942) 
#undef DECLARE_MT_GROKFAST

void grokfast_ema(torch::Tensor grad, torch::Tensor ema, float alpha, float lamb) {
 SG_DISPATCH(launch_fused_grokfast_ema, grad, ema, alpha, lamb);
}

void grokfast_adam(
 torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
 torch::Tensor ema, torch::Tensor grad,
 float alpha, float lamb,
 float beta1, float beta2, float lr, float weight_decay,
 float eps, float bc1, float bc2)
{
 SG_DISPATCH(launch_fused_grokfast_adam,
 param, exp_avg, exp_avg_sq, ema, grad,
 alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}

// Pre-refactor csrc/common/ops.cpp::grokfast_fused_step (EMA-only).
void grokfast_fused_step(
 std::vector<torch::Tensor>& grads,
 std::vector<torch::Tensor>& ema_bufs,
 float alpha, float lamb
) {
 if (grads.empty()) return;
 std::vector<torch::Tensor> vg, ve;
 for (size_t i = 0; i < grads.size(); i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 vg.push_back(grads[i]); ve.push_back(ema_bufs[i]);
 }
 if (vg.empty()) return;
 SG_DISPATCH(launch_multi_tensor_grokfast_ema, vg, ve, alpha, lamb);
}

// Pre-refactor csrc/common/ops.cpp::grokfast_fused_ema_adam_step.
// Per-param dispatch — there's no multi-tensor variant for the
// EMA+Adam fused kernel, so we loop and call the single-tensor launcher.
void grokfast_fused_ema_adam_step(
 std::vector<torch::Tensor>& params,
 std::vector<torch::Tensor>& grads,
 std::vector<torch::Tensor>& emas,
 std::vector<torch::Tensor>& exp_avgs,
 std::vector<torch::Tensor>& exp_avg_sqs,
 std::vector<int64_t>& steps,
 float alpha, float lamb,
 float beta1, float beta2, float lr, float wd, float eps
) {
 for (size_t i = 0; i < params.size(); i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
 float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));
 grokfast_adam(params[i], exp_avgs[i], exp_avg_sqs[i], emas[i], grads[i],
 alpha, lamb, beta1, beta2, lr, wd, eps, bc1, bc2);
 }
}

} // namespace sg


// ─── csrc/bindings/lion.cpp ────────────────────────────────────────
// bindings/lion.cpp — runtime dispatch to per-arch Lion launchers.
//
// Lion has a single launcher (launch_fused_lion_step). See bindings/grokadamw.cpp
// for the worked-example dispatcher pattern.


#include <vector>

namespace sg {

#define DECLARE_LION(NS) \
 namespace NS { \
 void launch_fused_lion_step( \
 torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad, \
 float lr, float beta1, float beta2, float weight_decay); \
 void launch_multi_tensor_lion( \
 std::vector<torch::Tensor>& params, \
 std::vector<torch::Tensor>& exp_avgs, \
 std::vector<torch::Tensor>& grads, \
 float lr, float beta1, float beta2, float wd); \
 }

 DECLARE_LION(sm90) DECLARE_LION(gfx942)

#undef DECLARE_LION

// Per-tensor dispatcher (kept as internal helper).
void lion_step(
 torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,
 float lr, float beta1, float beta2, float weight_decay)
{
 SG_DISPATCH(launch_fused_lion_step,
 param, exp_avg, grad, lr, beta1, beta2, weight_decay);
}

// High-level vector-signature entry point matching the pre-refactor
// csrc/common/ops.cpp::lion_fused_step.
void lion_fused_step(
 std::vector<torch::Tensor>& params,
 std::vector<torch::Tensor>& grads,
 std::vector<torch::Tensor>& exp_avgs,
 float lr, float beta1, float beta2, float wd
) {
 if (params.empty()) return;
 std::vector<torch::Tensor> vp, vg, vea;
 for (size_t i = 0; i < params.size(); i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 vp.push_back(params[i]); vg.push_back(grads[i]); vea.push_back(exp_avgs[i]);
 }
 if (vp.empty()) return;
 SG_DISPATCH(launch_multi_tensor_lion, vp, vea, vg, lr, beta1, beta2, wd);
}

} // namespace sg


// ─── csrc/bindings/looksam.cpp ─────────────────────────────────────
// bindings/looksam.cpp — runtime dispatch to per-arch LookSAM launchers.


#include <vector>

namespace sg {

#define DECLARE_LOOKSAM(NS) \
 namespace NS { \
 void launch_looksam_perturb( \
 torch::Tensor param, torch::Tensor backup, torch::Tensor grad, \
 float rho_over_norm); \
 void launch_looksam_restore( \
 torch::Tensor param, torch::Tensor backup); \
 void launch_looksam_direction_adjust_fused( \
 torch::Tensor grad, torch::Tensor sam_grad, \
 torch::Tensor v_dir, \
 float inv_norm, float lambda, float grad_norm); \
 void launch_looksam_norm_reduce( \
 torch::Tensor grad, torch::Tensor sam_grad, \
 torch::Tensor results /* [diff_norm, grad_norm] */); \
 }

 DECLARE_LOOKSAM(sm90)
 DECLARE_LOOKSAM(gfx942)

#undef DECLARE_LOOKSAM

void looksam_perturb(
 torch::Tensor param, torch::Tensor backup, torch::Tensor grad,
 float rho_over_norm)
{
 SG_DISPATCH(launch_looksam_perturb, param, backup, grad, rho_over_norm);
}

void looksam_restore(torch::Tensor param, torch::Tensor backup) {
 SG_DISPATCH(launch_looksam_restore, param, backup);
}

void looksam_direction_adjust_fused(
 torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor v_dir,
 float inv_norm, float lambda, float grad_norm)
{
 SG_DISPATCH(launch_looksam_direction_adjust_fused,
 grad, sam_grad, v_dir, inv_norm, lambda, grad_norm);
}

void looksam_norm_reduce(
 torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor results)
{
 SG_DISPATCH(launch_looksam_norm_reduce, grad, sam_grad, results);
}

// ---------------------------------------------------------------------
// High-level vector-signature entry points (pre-refactor ops.cpp).
// ---------------------------------------------------------------------

// Pre-refactor csrc/common/ops.cpp::looksam_perturb_all.
// Returns a vector of backups (param.clone() before perturb).
std::vector<torch::Tensor> looksam_perturb_all(
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
 SG_DISPATCH_CALL(launch_looksam_perturb,
 params[i], backups[i], grads[i], rho_over_norm);
 }
 return backups;
}

// Pre-refactor csrc/common/ops.cpp::looksam_restore_all.
void looksam_restore_all(
 std::vector<torch::Tensor>& params,
 std::vector<torch::Tensor>& backups
) {
 for (size_t i = 0; i < params.size(); i++) {
 SG_DISPATCH_CALL(launch_looksam_restore, params[i], backups[i]);
 }
}

// Pre-refactor csrc/common/ops.cpp::looksam_compute_directions_and_adjust.
// Batches the per-tensor norm reductions into 2 CPU syncs (diff + grad
// norms via torch::stack), then dispatches the fused adjust kernel.
void looksam_compute_directions_and_adjust(
 std::vector<torch::Tensor>& grads,
 std::vector<torch::Tensor>& sam_grads,
 std::vector<torch::Tensor>& normal_grads,
 float la
) {
 std::vector<size_t> valid_idx;
 for (size_t i = 0; i < grads.size(); i++) {
 if (!sam_grads[i].defined() || !normal_grads[i].defined()
 || sam_grads[i].numel() == 0) continue;
 valid_idx.push_back(i);
 }
 if (valid_idx.empty()) return;

 std::vector<torch::Tensor> diff_norms_t, grad_norms_t, diffs;
 diff_norms_t.reserve(valid_idx.size());
 grad_norms_t.reserve(valid_idx.size());
 diffs.reserve(valid_idx.size());
 for (auto i : valid_idx) {
 auto diff = (sam_grads[i] - normal_grads[i]).to(torch::kFloat32);
 diffs.push_back(diff);
 diff_norms_t.push_back(diff.norm());
 grad_norms_t.push_back(grads[i].norm());
 }
 auto diff_norms = torch::stack(diff_norms_t).cpu();
 auto grad_norms = torch::stack(grad_norms_t).cpu();
 auto* dnp = diff_norms.data_ptr<float>();
 auto* gnp = grad_norms.data_ptr<float>();

 for (size_t vi = 0; vi < valid_idx.size(); vi++) {
 size_t i = valid_idx[vi];
 float dn = dnp[vi];
 if (dn < 1e-12f) continue;
 float inv_norm = 1.0f / dn;
 float gn = gnp[vi];
 // The per-arch launcher's signature is
 // (grad, sam_grad, v_dir, inv_norm, lambda, grad_norm)
 // — passes diff (==v_dir for this fused kernel) as the third arg.
 SG_DISPATCH_CALL(launch_looksam_direction_adjust_fused,
 grads[i], sam_grads[i], diffs[vi], inv_norm, la, gn);
 }
}

} // namespace sg


// ─── csrc/bindings/moe.cpp ─────────────────────────────────────────
// bindings/moe.cpp — runtime dispatch to per-arch MoE launchers.
//
// MoE covers dynamic expert load/forward/backward, active-param
// filtering, compacted scan, scatter, activation counting, load-balance
// loss, and frequency-based LR scaling. Functions in the canonical
// moe_<arch>.cu files are NOT prefixed with `launch_`; they are direct
// host-side wrappers, so the SG_DISPATCH calls below match those names.


#include <vector>

namespace sg {

#define DECLARE_MOE(NS) \
 namespace NS { \
 void moe_dynamic_expert_load( \
 torch::Tensor expert_w1, torch::Tensor expert_b1, \
 torch::Tensor expert_w2, torch::Tensor expert_b2, \
 torch::Tensor active_mask, \
 torch::Tensor smem_w1, torch::Tensor smem_b1, \
 torch::Tensor smem_w2, torch::Tensor smem_b2); \
 torch::Tensor moe_dynamic_expert_fwd( \
 torch::Tensor input, torch::Tensor expert_indices, \
 torch::Tensor routing_weights, \
 torch::Tensor expert_w1, torch::Tensor expert_b1, \
 torch::Tensor expert_w2, torch::Tensor expert_b2, \
 torch::Tensor output); \
 void moe_dynamic_expert_bwd( \
 torch::Tensor d_output, torch::Tensor input, \
 torch::Tensor expert_indices, torch::Tensor routing_weights, \
 torch::Tensor expert_w1, torch::Tensor expert_b1, \
 torch::Tensor expert_w2, torch::Tensor expert_b2, \
 torch::Tensor d_input, torch::Tensor d_expert_w1, \
 torch::Tensor d_expert_b1, torch::Tensor d_expert_w2, \
 torch::Tensor d_expert_b2); \
 void moe_filter_active_params( \
 torch::Tensor params, torch::Tensor grads, \
 torch::Tensor state_m, torch::Tensor state_v, \
 torch::Tensor param_to_expert, torch::Tensor expert_active, \
 torch::Tensor compact_params, torch::Tensor compact_grads, \
 torch::Tensor compact_state_m, torch::Tensor compact_state_v, \
 torch::Tensor scatter_indices, torch::Tensor compact_count, \
 int total_params); \
 void moe_scan_compacted( \
 torch::Tensor compact_x, torch::Tensor compact_dt, \
 torch::Tensor compact_B, torch::Tensor compact_C, \
 torch::Tensor A_log, torch::Tensor D_param, \
 torch::Tensor rope_freq, \
 torch::Tensor scan_output, torch::Tensor final_state, \
 torch::Tensor initial_state, \
 int compact_N, int d_inner, int d_state); \
 void moe_scatter_results( \
 torch::Tensor compact_params, \
 torch::Tensor compact_state_m, torch::Tensor compact_state_v, \
 torch::Tensor scatter_indices, \
 torch::Tensor params, \
 torch::Tensor state_m, torch::Tensor state_v, \
 int compact_N); \
 void moe_count_expert_activations( \
 torch::Tensor gate_logits, torch::Tensor expert_counts, \
 float threshold, int N, int num_experts); \
 torch::Tensor moe_compute_load_balance_loss( \
 torch::Tensor expert_counts, torch::Tensor gate_logits, \
 int N, int num_experts); \
 void moe_apply_frequency_scaling( \
 torch::Tensor expert_counts, torch::Tensor lr_scale, \
 int num_experts, int total_activations, \
 float min_scale, float max_scale, float smoothing); \
 }

 DECLARE_MOE(sm90)

DECLARE_MOE(gfx942) 
#undef DECLARE_MOE

// ---------------------------------------------------------------------
// Public Python entry points. Names match the m.def() registrations
// from the deleted csrc/common/ops.cpp.
// ---------------------------------------------------------------------

void moe_dynamic_expert_load(
 torch::Tensor expert_w1, torch::Tensor expert_b1,
 torch::Tensor expert_w2, torch::Tensor expert_b2,
 torch::Tensor active_mask,
 torch::Tensor smem_w1, torch::Tensor smem_b1,
 torch::Tensor smem_w2, torch::Tensor smem_b2)
{
 SG_DISPATCH(moe_dynamic_expert_load,
 expert_w1, expert_b1, expert_w2, expert_b2, active_mask,
 smem_w1, smem_b1, smem_w2, smem_b2);
}

torch::Tensor moe_dynamic_expert_fwd(
 torch::Tensor input, torch::Tensor expert_indices,
 torch::Tensor routing_weights,
 torch::Tensor expert_w1, torch::Tensor expert_b1,
 torch::Tensor expert_w2, torch::Tensor expert_b2,
 torch::Tensor output)
{
 SG_DISPATCH(moe_dynamic_expert_fwd,
 input, expert_indices, routing_weights,
 expert_w1, expert_b1, expert_w2, expert_b2, output);
}

void moe_dynamic_expert_bwd(
 torch::Tensor d_output, torch::Tensor input,
 torch::Tensor expert_indices, torch::Tensor routing_weights,
 torch::Tensor expert_w1, torch::Tensor expert_b1,
 torch::Tensor expert_w2, torch::Tensor expert_b2,
 torch::Tensor d_input, torch::Tensor d_expert_w1,
 torch::Tensor d_expert_b1, torch::Tensor d_expert_w2,
 torch::Tensor d_expert_b2)
{
 SG_DISPATCH(moe_dynamic_expert_bwd,
 d_output, input, expert_indices, routing_weights,
 expert_w1, expert_b1, expert_w2, expert_b2,
 d_input, d_expert_w1, d_expert_b1, d_expert_w2, d_expert_b2);
}

void moe_filter_active_params(
 torch::Tensor params, torch::Tensor grads,
 torch::Tensor state_m, torch::Tensor state_v,
 torch::Tensor param_to_expert, torch::Tensor expert_active,
 torch::Tensor compact_params, torch::Tensor compact_grads,
 torch::Tensor compact_state_m, torch::Tensor compact_state_v,
 torch::Tensor scatter_indices, torch::Tensor compact_count,
 int total_params)
{
 SG_DISPATCH(moe_filter_active_params,
 params, grads, state_m, state_v, param_to_expert, expert_active,
 compact_params, compact_grads, compact_state_m, compact_state_v,
 scatter_indices, compact_count, total_params);
}

void moe_scan_compacted(
 torch::Tensor compact_x, torch::Tensor compact_dt,
 torch::Tensor compact_B, torch::Tensor compact_C,
 torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
 torch::Tensor scan_output, torch::Tensor final_state,
 torch::Tensor initial_state,
 int compact_N, int d_inner, int d_state)
{
 SG_DISPATCH(moe_scan_compacted,
 compact_x, compact_dt, compact_B, compact_C,
 A_log, D_param, rope_freq,
 scan_output, final_state, initial_state,
 compact_N, d_inner, d_state);
}

void moe_scatter_results(
 torch::Tensor compact_params,
 torch::Tensor compact_state_m, torch::Tensor compact_state_v,
 torch::Tensor scatter_indices,
 torch::Tensor params,
 torch::Tensor state_m, torch::Tensor state_v,
 int compact_N)
{
 SG_DISPATCH(moe_scatter_results,
 compact_params, compact_state_m, compact_state_v,
 scatter_indices, params, state_m, state_v, compact_N);
}

void moe_count_expert_activations(
 torch::Tensor gate_logits, torch::Tensor expert_counts,
 float threshold, int N, int num_experts)
{
 SG_DISPATCH(moe_count_expert_activations,
 gate_logits, expert_counts, threshold, N, num_experts);
}

torch::Tensor moe_compute_load_balance_loss(
 torch::Tensor expert_counts, torch::Tensor gate_logits,
 int N, int num_experts)
{
 SG_DISPATCH(moe_compute_load_balance_loss,
 expert_counts, gate_logits, N, num_experts);
}

void moe_apply_frequency_scaling(
 torch::Tensor expert_counts, torch::Tensor lr_scale,
 int num_experts, int total_activations,
 float min_scale, float max_scale, float smoothing)
{
 SG_DISPATCH(moe_apply_frequency_scaling,
 expert_counts, lr_scale, num_experts, total_activations,
 min_scale, max_scale, smoothing);
}

} // namespace sg


// ─── csrc/bindings/multi_tensor.cpp ────────────────────────────────
// bindings/multi_tensor.cpp — runtime dispatch to per-arch multi-tensor launchers.
//
// The multi-tensor path fuses many small parameter tensors into one launch.
// Currently registered: GrokAdamW and Lion. (Grokfast EMA's multi-tensor
// variant is wired in csrc/bindings/grokfast.cpp; Prodigy's
// fused-reduce-step multi-tensor variant is wired in csrc/bindings/prodigy.cpp.)


#include <vector>

namespace sg {

#define DECLARE_MT(NS) \
 namespace NS { \
 void launch_multi_tensor_grokadamw( \
 std::vector<torch::Tensor> params, \
 std::vector<torch::Tensor> exp_avgs, \
 std::vector<torch::Tensor> exp_avg_sqs, \
 std::vector<torch::Tensor> emas, \
 std::vector<torch::Tensor> grads, \
 float alpha, float lamb, \
 float beta1, float beta2, \
 float lr, float weight_decay, \
 float eps, float bc1, float bc2); \
 void launch_multi_tensor_lion( \
 std::vector<torch::Tensor> params, \
 std::vector<torch::Tensor> exp_avgs, \
 std::vector<torch::Tensor> grads, \
 float lr, float beta1, float beta2, float weight_decay); \
 }

 DECLARE_MT(sm90) DECLARE_MT(gfx942)

#undef DECLARE_MT

void multi_tensor_grokadamw(
 std::vector<torch::Tensor> params, std::vector<torch::Tensor> exp_avgs,
 std::vector<torch::Tensor> exp_avg_sqs, std::vector<torch::Tensor> emas,
 std::vector<torch::Tensor> grads,
 float alpha, float lamb, float beta1, float beta2,
 float lr, float weight_decay, float eps, float bc1, float bc2)
{
 SG_DISPATCH(launch_multi_tensor_grokadamw,
 params, exp_avgs, exp_avg_sqs, emas, grads,
 alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}

void multi_tensor_lion(
 std::vector<torch::Tensor> params, std::vector<torch::Tensor> exp_avgs,
 std::vector<torch::Tensor> grads,
 float lr, float beta1, float beta2, float weight_decay)
{
 SG_DISPATCH(launch_multi_tensor_lion,
 params, exp_avgs, grads, lr, beta1, beta2, weight_decay);
}

} // namespace sg


// ─── csrc/bindings/muon.cpp ────────────────────────────────────────
// bindings/muon.cpp — runtime dispatch to per-arch Muon launchers.
//
// Per-arch namespaces define four launchers (signatures must match the
// definitions in csrc/kernels/{cuda/<sm>,hip/gfx{942,950}}/muon_<arch>.cu):
//
// launch_muon_momentum_normalize(buf, X, grad, momentum, inv_norm)
// launch_muon_ns_combine(X_out, X, AX, AAX, a, b, c)
// launch_muon_update(param, orth, neg_lr_scale, decay_factor)
// launch_muon_ns_combine_update_fused(
// param, X, AX, AAX, a, b, c, neg_lr_scale, decay_factor)
//
// The vector-API muon_fused_step in this file orchestrates the per-tensor
// pipeline (norm reduction, NS iterations, fused final step) and is the
// only Python-callable Muon entry. The Newton-Schulz coefficients
// (3.4445, -4.7750, 2.0315) come from the pre-refactor ops.cpp constants
// and are passed as a, b, c.


#include <cmath>
#include <vector>

namespace sg {

#define DECLARE_MUON(NS) \
 namespace NS { \
 void launch_muon_momentum_normalize( \
 torch::Tensor buf, torch::Tensor X, torch::Tensor grad, \
 float momentum, float inv_norm); \
 void launch_muon_ns_combine( \
 torch::Tensor X_out, torch::Tensor X, \
 torch::Tensor AX, torch::Tensor AAX, \
 float a, float b, float c); \
 void launch_muon_update( \
 torch::Tensor param, torch::Tensor orth, \
 float neg_lr_scale, float decay_factor); \
 void launch_muon_ns_combine_update_fused( \
 torch::Tensor param, torch::Tensor X, \
 torch::Tensor AX, torch::Tensor AAX, \
 float a, float b, float c, \
 float neg_lr_scale, float decay_factor); \
 }

 DECLARE_MUON(sm90)

DECLARE_MUON(gfx942) 
#undef DECLARE_MUON

// ---------------------------------------------------------------------
// Per-tensor wrappers (internal helpers).
// ---------------------------------------------------------------------

void muon_momentum_normalize(
 torch::Tensor buf, torch::Tensor X, torch::Tensor grad,
 float momentum, float inv_norm)
{
 SG_DISPATCH(launch_muon_momentum_normalize,
 buf, X, grad, momentum, inv_norm);
}

void muon_ns_combine(
 torch::Tensor X_out, torch::Tensor X,
 torch::Tensor AX, torch::Tensor AAX,
 float a, float b, float c)
{
 SG_DISPATCH(launch_muon_ns_combine, X_out, X, AX, AAX, a, b, c);
}

void muon_ns_combine_update_fused(
 torch::Tensor param, torch::Tensor X,
 torch::Tensor AX, torch::Tensor AAX,
 float a, float b, float c,
 float neg_lr_scale, float decay_factor)
{
 SG_DISPATCH(launch_muon_ns_combine_update_fused,
 param, X, AX, AAX, a, b, c, neg_lr_scale, decay_factor);
}

// ---------------------------------------------------------------------
// High-level vector-signature entry point — pre-refactor
// csrc/common/ops.cpp::muon_fused_step. The math is unchanged.
// Replaces N CPU syncs with one (torch::stack of bufs[i].norm()).
// ---------------------------------------------------------------------

void muon_fused_step(
 std::vector<torch::Tensor>& params,
 std::vector<torch::Tensor>& grads,
 std::vector<torch::Tensor>& bufs,
 float momentum, float lr, float wd, int ns_steps
) {
 constexpr float NS_A = 3.4445f;
 constexpr float NS_B = -4.7750f;
 constexpr float NS_C = 2.0315f;

 std::vector<torch::Tensor> norm_tensors;
 std::vector<size_t> valid_indices;
 norm_tensors.reserve(params.size());
 valid_indices.reserve(params.size());
 for (size_t i = 0; i < params.size(); i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 bufs[i].mul_(momentum).add_(grads[i]);
 norm_tensors.push_back(bufs[i].norm());
 valid_indices.push_back(i);
 }
 if (norm_tensors.empty()) return;

 auto norms_stacked = torch::stack(norm_tensors).cpu();
 auto* norms_ptr = norms_stacked.data_ptr<float>();

 for (size_t vi = 0; vi < valid_indices.size(); vi++) {
 size_t i = valid_indices[vi];
 auto& p = params[i];
 auto& buf = bufs[i];

 float buf_norm = norms_ptr[vi] + 1e-7f;
 float inv_norm = 1.0f / buf_norm;
 auto X = buf * inv_norm;

 int64_t rows = p.size(0);
 int64_t cols = p.size(1);
 float max_dim = static_cast<float>(std::max(rows, cols));
 float scale_factor = 0.2f * std::sqrt(max_dim);
 float neg_lr_scale = -lr * scale_factor;
 float decay_factor = 1.0f - lr * wd;

 for (int step = 0; step < ns_steps; step++) {
 auto A = torch::mm(X, X.t());
 auto AX = torch::mm(A, X);
 auto AAX = torch::mm(A, AX);
 if (step < ns_steps - 1) {
 auto X_new = torch::empty_like(X);
 SG_DISPATCH_CALL(launch_muon_ns_combine,
 X_new, X, AX, AAX, NS_A, NS_B, NS_C);
 X = X_new;
 } else {
 SG_DISPATCH_CALL(launch_muon_ns_combine_update_fused,
 p, X, AX, AAX, NS_A, NS_B, NS_C,
 neg_lr_scale, decay_factor);
 }
 }
 if (ns_steps == 0) {
 auto X_typed = X.to(p.dtype());
 SG_DISPATCH_CALL(launch_muon_update,
 p, X_typed, neg_lr_scale, decay_factor);
 }
 }
}

} // namespace sg


// ─── csrc/bindings/neuralgrok.cpp ──────────────────────────────────
// bindings/neuralgrok.cpp — runtime dispatch to per-arch NeuralGrok launchers.
//
// The amplifier MLP launchers vary by hidden-dim templating; the bindings
// expose the generic untemplated entry point and the templated
// specializations (H=16/32/64/128) are dispatched inside the per-arch
// launcher itself rather than at the binding boundary.


#include <cmath>
#include <vector>

namespace sg {

#define DECLARE_NEURALGROK(NS) \
 namespace NS { \
 void launch_fused_neuralgrok_amplifier( \
 torch::Tensor grad, torch::Tensor amplified, \
 torch::Tensor amplifier_w1, torch::Tensor amplifier_b1, \
 torch::Tensor amplifier_w2, torch::Tensor amplifier_b2, \
 int hidden_dim, float alpha, float beta); \
 void launch_fused_neuralgrok_adam( \
 torch::Tensor param, torch::Tensor exp_avg, \
 torch::Tensor exp_avg_sq, torch::Tensor amplified_grad, \
 float beta1, float beta2, \
 float lr, float weight_decay, \
 float eps, float bc1, float bc2); \
 void launch_fused_neuralgrok_full_step( \
 torch::Tensor param, torch::Tensor exp_avg, \
 torch::Tensor exp_avg_sq, torch::Tensor grad, \
 torch::Tensor W1, torch::Tensor b1, \
 torch::Tensor W2, torch::Tensor b2, \
 float alpha_amp, float beta_amp, int hidden_dim, \
 float beta1, float beta2, float lr, float weight_decay, \
 float eps, float bc1, float bc2); \
 }

 DECLARE_NEURALGROK(sm90)
 DECLARE_NEURALGROK(gfx942)

#undef DECLARE_NEURALGROK

void neuralgrok_amplifier(
 torch::Tensor grad, torch::Tensor amplified,
 torch::Tensor amplifier_w1, torch::Tensor amplifier_b1,
 torch::Tensor amplifier_w2, torch::Tensor amplifier_b2,
 int hidden_dim, float alpha, float beta)
{
 SG_DISPATCH(launch_fused_neuralgrok_amplifier,
 grad, amplified, amplifier_w1, amplifier_b1,
 amplifier_w2, amplifier_b2, hidden_dim, alpha, beta);
}

void neuralgrok_adam(
 torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
 torch::Tensor amplified_grad,
 float beta1, float beta2, float lr, float weight_decay,
 float eps, float bc1, float bc2)
{
 SG_DISPATCH(launch_fused_neuralgrok_adam,
 param, exp_avg, exp_avg_sq, amplified_grad,
 beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}

// Pre-refactor csrc/common/ops.cpp::neuralgrok_fused_step (full single
// kernel: amplifier MLP + Adam fused per-param).
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
 float eps, float grad_clip_norm
) {
 const size_t n_params = params.size();
 clip_grad_norms_device_side(grads, n_params, grad_clip_norm);

 for (size_t i = 0; i < n_params; i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
 float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));
 SG_DISPATCH_CALL(launch_fused_neuralgrok_full_step,
 params[i], exp_avgs[i], exp_avg_sqs[i], grads[i],
 W1, b1, W2, b2, alpha_amp, beta_amp, hidden_dim,
 beta1, beta2, lr, wd, eps, bc1, bc2);
 }
}

} // namespace sg


// ─── csrc/bindings/prodigy.cpp ─────────────────────────────────────
// bindings/prodigy.cpp — runtime dispatch to per-arch Prodigy launchers.

#include <cmath>
#include <vector>

namespace sg {

#define DECLARE_PRODIGY(NS) \
 namespace NS { \
 void launch_fused_prodigy_step( \
 torch::Tensor param, torch::Tensor exp_avg, \
 torch::Tensor exp_avg_sq, torch::Tensor s, \
 torch::Tensor param_init, torch::Tensor grad, \
 float lr, float d_lr, \
 float beta1, float beta2, \
 float weight_decay, \
 float eps, float bc1, float bc2); \
 void launch_prodigy_dlr_reduce( \
 torch::Tensor grad, torch::Tensor param, \
 torch::Tensor param_init, torch::Tensor s, \
 torch::Tensor numerator, torch::Tensor denominator, \
 float eps); \
 }

 DECLARE_PRODIGY(sm90)
 DECLARE_PRODIGY(gfx942)

#undef DECLARE_PRODIGY

// Multi-tensor fused-reduce-step launcher (in multi_tensor_<arch>.cu).
#define DECLARE_MT_PRODIGY(NS) \
 namespace NS { \
 void launch_multi_tensor_prodigy_fused_reduce_step( \
 std::vector<torch::Tensor>& params, \
 std::vector<torch::Tensor>& grads, \
 std::vector<torch::Tensor>& param_inits, \
 std::vector<torch::Tensor>& exp_avgs, \
 std::vector<torch::Tensor>& exp_avg_sqs, \
 std::vector<torch::Tensor>& s_bufs, \
 std::vector<float>& bc1s, std::vector<float>& bc2s, \
 torch::Tensor d_lr_buf, \
 float beta1, float beta2, float lr, float wd, float eps); \
 }
 DECLARE_MT_PRODIGY(sm90)

DECLARE_MT_PRODIGY(gfx942) 
#undef DECLARE_MT_PRODIGY

void prodigy_step(
 torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
 torch::Tensor s, torch::Tensor param_init, torch::Tensor grad,
 float lr, float d_lr, float beta1, float beta2, float weight_decay,
 float eps, float bc1, float bc2)
{
 SG_DISPATCH(launch_fused_prodigy_step,
 param, exp_avg, exp_avg_sq, s, param_init, grad,
 lr, d_lr, beta1, beta2, weight_decay, eps, bc1, bc2);
}

void prodigy_dlr_reduce(
 torch::Tensor grad, torch::Tensor param, torch::Tensor param_init,
 torch::Tensor s, torch::Tensor numerator, torch::Tensor denominator,
 float eps)
{
 SG_DISPATCH(launch_prodigy_dlr_reduce,
 grad, param, param_init, s, numerator, denominator, eps);
}

// Pre-refactor csrc/common/ops.cpp::prodigy_fused_step.
// Returns the updated d_lr (computed device-side, single CPU sync).
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
 float eps
) {
 if (params.empty()) return d_lr;

 torch::Device dev(torch::kCPU);
 for (auto& g : grads) {
 if (g.defined() && g.numel() > 0) { dev = g.device(); break; }
 }

 std::vector<torch::Tensor> vp, vg, vpi, vea, veasq, vs;
 std::vector<float> bc1_vec, bc2_vec;
 for (size_t i = 0; i < params.size(); i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
 float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));
 vp.push_back(params[i]); vg.push_back(grads[i]);
 vpi.push_back(param_inits[i]);
 vea.push_back(exp_avgs[i]); veasq.push_back(exp_avg_sqs[i]);
 vs.push_back(s_bufs[i]);
 bc1_vec.push_back(bc1); bc2_vec.push_back(bc2);
 }
 if (vp.empty()) return d_lr;

 auto d_lr_buf = torch::tensor(
 {d_lr}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));

 SG_DISPATCH_CALL(launch_multi_tensor_prodigy_fused_reduce_step,
 vp, vg, vpi, vea, veasq, vs, bc1_vec, bc2_vec, d_lr_buf,
 beta1, beta2, lr, wd, eps);
 return d_lr_buf.item<float>();
}

} // namespace sg


// ─── csrc/bindings/supergrok11.cpp ─────────────────────────────────
// bindings/supergrok11.cpp — runtime dispatch to per-arch SG v1.1 launchers.
//
// SG v1.1 mirrors v1.5 except for the cosine-similarity gate. The flow:
// 1. mu_metanet kernel produces smart_grad (and updates mu)
// 2. compute_cosine_gate_fused does a 3-quantity reduction
// (dot, |sg|², |mu|²), syncs once to CPU, returns sigmoid(t·cos_sim)
// 3. adam_decay applies the Adam step using lamb_eff = ramp·gate·lamb


#include <cmath>
#include <vector>

namespace sg {

#define DECLARE_SG11(NS) \
 namespace NS { \
 void launch_sg11_mu_metanet( \
 torch::Tensor mu, torch::Tensor grad, \
 torch::Tensor sharpness, torch::Tensor smart_grad, \
 float alpha, \
 torch::Tensor W1, torch::Tensor b1, \
 torch::Tensor W2, torch::Tensor b2, \
 float rescale, int hidden_dim); \
 void launch_sg11_adam_decay( \
 torch::Tensor param, torch::Tensor exp_avg, \
 torch::Tensor exp_avg_sq, torch::Tensor smart_grad, \
 torch::Tensor mu, \
 float lamb_eff, float beta1, float beta2, \
 float lr, float wd_eff, float eps, float bc1, float bc2); \
 void launch_sg11_sam_perturb( \
 torch::Tensor param, torch::Tensor grad, float rho_over_norm); \
 void launch_sg11_sharpness_restore( \
 torch::Tensor param, torch::Tensor sharpness, \
 torch::Tensor backup, \
 torch::Tensor sam_grad, torch::Tensor normal_grad); \
 float compute_cosine_gate_fused( \
 torch::Tensor smart_grad, torch::Tensor mu, float gate_temp); \
 }

 DECLARE_SG11(sm90)

DECLARE_SG11(gfx942) 
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

 case 90: return ::sg::sm90::compute_cosine_gate_fused(smart_grad, mu, gate_temp);

 case 942: return ::sg::gfx942::compute_cosine_gate_fused(smart_grad, mu, gate_temp);
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


// ─── csrc/bindings/supergrok15.cpp ─────────────────────────────────
// bindings/supergrok15.cpp — runtime dispatch to per-arch SG v1.5 launchers.
//
// SG v1.5 has launchers for the meta-net forward, the Adam step with
// decoupled WD, the SAM perturb, the sharpness restore (post-SAM), and a
// fused full-step that does the whole pipeline in one kernel.


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


// ─── csrc/bindings/supergrok2.cpp ──────────────────────────────────
// bindings/supergrok2.cpp — runtime dispatch to per-arch SG v2 launchers.
//
// SG v2 has the largest dispatcher surface in the codebase. The Python
// optimizer (grokking_optimizers/supergrok2.py) invokes six entry points
// that all dispatch to per-arch launchers in
// csrc/kernels/{cuda/<sm>,hip/<gfx>}/supergrok2_{fwd,bwd}_<arch>.{cu,hip.cpp}.
//
// Per-arch launcher names (signatures must match exactly across all 8
// arches — the wrap-baseline pass preserved bit-identical signatures):
//
// sg::<arch>::launch_mamba3_peer_step (forward step)
// sg::<arch>::launch_mamba3_peer_batched_step (batched fwd step)
// sg::<arch>::launch_mamba3_peer_bilevel_fwd_save (bilevel fwd-save)
// sg::<arch>::launch_mamba3_peer_bilevel_fwd_save_batched(batched fwd-save)
// sg::<arch>::launch_mamba3_peer_backward (bilevel bwd)
// sg::<arch>::launch_mamba3_peer_backward_batched (batched bwd)
//
// Public Python entry points (registered via _ops.<name> in module.cpp):
// supergrok2_mamba_peer_step
// supergrok2_mamba_peer_batched_step
// supergrok2_bilevel_fwd_save
// supergrok2_bilevel_fwd_save_batched
// supergrok2_bilevel_backward
// supergrok2_bilevel_backward_batched
//
// Reference: pre-refactor csrc/common/ops.cpp@682eab4^ (lines 908-1199 for
// forward/peer-step entries, lines 1499-1611 for bilevel registrations).
// The launcher signatures here are taken verbatim from the canonical sm_80
// translation units (supergrok2_fwd_sm80.cu / supergrok2_bwd_sm80.cu) —
// they are the linker contract.


#include <vector>

namespace sg {

// ---------------------------------------------------------------------
// Per-arch launcher forward declarations.
// Signatures verbatim from csrc/kernels/cuda/sm_80/supergrok2_{fwd,bwd}_sm80.cu.
// ---------------------------------------------------------------------

#define DECLARE_SG2(NS) \
 namespace NS { \
 void launch_mamba3_peer_step( \
 torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness, \
 torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,\
 torch::Tensor gru_state, \
 torch::Tensor mamba_fwd_state, torch::Tensor mamba_bwd_state, \
 torch::Tensor input_proj_W, torch::Tensor input_proj_b, \
 torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W, \
 torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj, \
 torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log, \
 torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope, \
 torch::Tensor mamba_fwd_out_proj, \
 torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W, \
 torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj, \
 torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log, \
 torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope, \
 torch::Tensor mamba_bwd_out_proj, \
 torch::Tensor gru_Wz, torch::Tensor gru_bz, \
 torch::Tensor gru_Wr, torch::Tensor gru_br, \
 torch::Tensor gru_Wh, torch::Tensor gru_bh, \
 torch::Tensor peer_query_Ws, \
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B, \
 torch::Tensor expert_W1, torch::Tensor expert_b1, \
 torch::Tensor expert_W2, torch::Tensor expert_b2, \
 float rescale, float alpha_mu, float lamb_eff, \
 float beta1, float beta2, float lr, float wd_eff, float eps, \
 float bc1, float bc2, \
 int d_model, int d_state, int d_inner, \
 int gru_hidden, int num_heads, int pk_dim, \
 int expert_hidden, int num_experts, \
 torch::Tensor expert_counts); \
 void launch_mamba3_peer_batched_step( \
 std::vector<torch::Tensor> params, \
 std::vector<torch::Tensor> grads, \
 std::vector<torch::Tensor> sharpness_list, \
 std::vector<torch::Tensor> exp_avgs, \
 std::vector<torch::Tensor> exp_avg_sqs, \
 std::vector<torch::Tensor> mus, \
 std::vector<torch::Tensor> gru_states, \
 std::vector<torch::Tensor> mamba_fwd_states, \
 std::vector<torch::Tensor> mamba_bwd_states, \
 torch::Tensor input_proj_W, torch::Tensor input_proj_b, \
 torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W, \
 torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj, \
 torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log, \
 torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope, \
 torch::Tensor mamba_fwd_out_proj, \
 torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W, \
 torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj, \
 torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log, \
 torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope, \
 torch::Tensor mamba_bwd_out_proj, \
 torch::Tensor gru_Wz, torch::Tensor gru_bz, \
 torch::Tensor gru_Wr, torch::Tensor gru_br, \
 torch::Tensor gru_Wh, torch::Tensor gru_bh, \
 torch::Tensor peer_query_Ws, \
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B, \
 torch::Tensor expert_W1, torch::Tensor expert_b1, \
 torch::Tensor expert_W2, torch::Tensor expert_b2, \
 std::vector<float> alpha_mus, std::vector<float> lamb_effs, \
 std::vector<float> beta1s, \
 std::vector<float> bc1s, std::vector<float> bc2s, \
 float rescale, float beta2, float lr, float wd_eff, float eps, \
 int d_model, int d_state, int d_inner, \
 int gru_hidden, int num_heads, int pk_dim, \
 int expert_hidden, int num_experts, \
 torch::Tensor expert_counts); \
 void launch_mamba3_peer_bilevel_fwd_save( \
 torch::Tensor grad, torch::Tensor sharpness, \
 torch::Tensor input_proj_W, torch::Tensor input_proj_b, \
 torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W, \
 torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj, \
 torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log, \
 torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope, \
 torch::Tensor mamba_fwd_out_proj, \
 torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W, \
 torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj, \
 torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log, \
 torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope, \
 torch::Tensor mamba_bwd_out_proj, \
 int d_model, int d_state, int d_inner, \
 torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out, \
 torch::Tensor fwd_final_state, torch::Tensor bwd_final_state, \
 torch::Tensor fwd_saved_states, \
 torch::Tensor fwd_saved_x_branch, \
 torch::Tensor fwd_saved_z, torch::Tensor fwd_saved_dt, \
 torch::Tensor bwd_saved_states, \
 torch::Tensor bwd_saved_x_branch, \
 torch::Tensor bwd_saved_z, torch::Tensor bwd_saved_dt, \
 torch::Tensor x_sorted, torch::Tensor sort_indices, \
 torch::Tensor fwd_initial_state, \
 torch::Tensor bwd_initial_state, \
 int checkpoint_interval); \
 void launch_mamba3_peer_bilevel_fwd_save_batched( \
 std::vector<torch::Tensor> grads, \
 std::vector<torch::Tensor> sharpness_list, \
 torch::Tensor input_proj_W, torch::Tensor input_proj_b, \
 torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W, \
 torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj, \
 torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log, \
 torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope, \
 torch::Tensor mamba_fwd_out_proj, \
 torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W, \
 torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj, \
 torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log, \
 torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope, \
 torch::Tensor mamba_bwd_out_proj, \
 int d_model, int d_state, int d_inner, \
 torch::Tensor fwd_scan_out_packed, \
 torch::Tensor bwd_scan_out_packed, \
 torch::Tensor fwd_saved_states_packed, \
 torch::Tensor fwd_saved_xb_packed, \
 torch::Tensor fwd_saved_z_packed, \
 torch::Tensor fwd_saved_dt_packed, \
 torch::Tensor bwd_saved_states_packed, \
 torch::Tensor bwd_saved_xb_packed, \
 torch::Tensor bwd_saved_z_packed, \
 torch::Tensor bwd_saved_dt_packed, \
 torch::Tensor x_sorted_packed, torch::Tensor offsets_t, \
 torch::Tensor sort_indices_packed, \
 torch::Tensor fwd_initial_states, \
 torch::Tensor bwd_initial_states, \
 int checkpoint_interval); \
 void launch_mamba3_peer_backward( \
 torch::Tensor d_smart_grad, \
 torch::Tensor grad, torch::Tensor sharpness, float rescale, \
 torch::Tensor sort_indices, torch::Tensor x_sorted, \
 torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out, \
 torch::Tensor fwd_saved_states, \
 torch::Tensor fwd_saved_x_branch, \
 torch::Tensor fwd_saved_z, torch::Tensor fwd_saved_dt, \
 torch::Tensor bwd_saved_states, \
 torch::Tensor bwd_saved_x_branch, \
 torch::Tensor bwd_saved_z, torch::Tensor bwd_saved_dt, \
 torch::Tensor gru_input, torch::Tensor gru_h_old, \
 torch::Tensor gru_z_gate, torch::Tensor gru_r_gate, \
 torch::Tensor gru_h_tilde, \
 torch::Tensor peer_input, torch::Tensor expert_indices, \
 torch::Tensor routing_weights, torch::Tensor saved_z_hidden, \
 torch::Tensor saved_scores_a, torch::Tensor saved_scores_b, \
 torch::Tensor saved_top_a_idx, torch::Tensor saved_top_b_idx, \
 torch::Tensor saved_soft_a, torch::Tensor saved_soft_b, \
 torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W, \
 torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj, \
 torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log, \
 torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope, \
 torch::Tensor mamba_fwd_out_proj, \
 torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W, \
 torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj, \
 torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log, \
 torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope, \
 torch::Tensor mamba_bwd_out_proj, \
 torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh, \
 torch::Tensor peer_query_Ws, \
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B, \
 torch::Tensor expert_W1, torch::Tensor expert_W2, \
 torch::Tensor expert_b1_in, torch::Tensor expert_b2_in, \
 torch::Tensor input_proj_W, \
 torch::Tensor mamba_fwd_init_state, \
 torch::Tensor mamba_bwd_init_state, \
 torch::Tensor d_mamba_fwd_in_proj, \
 torch::Tensor d_mamba_fwd_dt_W, \
 torch::Tensor d_mamba_fwd_dt_b, \
 torch::Tensor d_mamba_fwd_B_proj, \
 torch::Tensor d_mamba_fwd_C_proj, \
 torch::Tensor d_mamba_fwd_A_log, \
 torch::Tensor d_mamba_fwd_D, \
 torch::Tensor d_mamba_fwd_rope, \
 torch::Tensor d_mamba_fwd_out_proj, \
 torch::Tensor d_mamba_bwd_in_proj, \
 torch::Tensor d_mamba_bwd_dt_W, \
 torch::Tensor d_mamba_bwd_dt_b, \
 torch::Tensor d_mamba_bwd_B_proj, \
 torch::Tensor d_mamba_bwd_C_proj, \
 torch::Tensor d_mamba_bwd_A_log, \
 torch::Tensor d_mamba_bwd_D, \
 torch::Tensor d_mamba_bwd_rope, \
 torch::Tensor d_mamba_bwd_out_proj, \
 torch::Tensor d_gru_Wz, torch::Tensor d_gru_bz, \
 torch::Tensor d_gru_Wr, torch::Tensor d_gru_br, \
 torch::Tensor d_gru_Wh, torch::Tensor d_gru_bh, \
 torch::Tensor d_peer_query_Ws, \
 torch::Tensor d_prod_keys_A, torch::Tensor d_prod_keys_B, \
 torch::Tensor d_expert_W1, torch::Tensor d_expert_b1, \
 torch::Tensor d_expert_W2, torch::Tensor d_expert_b2, \
 torch::Tensor d_input_proj_W, torch::Tensor d_input_proj_b, \
 int d_model, int d_state, int d_inner, \
 int gru_hidden, int gru_input_dim, \
 int num_heads, int topk, int pk_dim, \
 int expert_hidden, int peer_input_dim, int num_experts, \
 int checkpoint_interval); \
 void launch_mamba3_peer_backward_batched( \
 torch::Tensor d_fwd_scan_out_packed, \
 torch::Tensor d_bwd_scan_out_packed, \
 torch::Tensor x_sorted_packed, \
 torch::Tensor fwd_saved_states_packed, \
 torch::Tensor fwd_saved_xb_packed, \
 torch::Tensor fwd_saved_z_packed, \
 torch::Tensor fwd_saved_dt_packed, \
 torch::Tensor bwd_saved_states_packed, \
 torch::Tensor bwd_saved_xb_packed, \
 torch::Tensor bwd_saved_z_packed, \
 torch::Tensor bwd_saved_dt_packed, \
 torch::Tensor offsets_t, \
 torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W, \
 torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj, \
 torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log, \
 torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope, \
 torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W, \
 torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj, \
 torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log, \
 torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope, \
 torch::Tensor d_mamba_fwd_in_proj, \
 torch::Tensor d_mamba_fwd_dt_W, torch::Tensor d_mamba_fwd_dt_b, \
 torch::Tensor d_mamba_fwd_B_proj, \
 torch::Tensor d_mamba_fwd_C_proj, \
 torch::Tensor d_mamba_fwd_A_log, \
 torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope, \
 torch::Tensor d_mamba_bwd_in_proj, \
 torch::Tensor d_mamba_bwd_dt_W, torch::Tensor d_mamba_bwd_dt_b, \
 torch::Tensor d_mamba_bwd_B_proj, \
 torch::Tensor d_mamba_bwd_C_proj, \
 torch::Tensor d_mamba_bwd_A_log, \
 torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope, \
 torch::Tensor d_x_sorted_packed, \
 torch::Tensor fwd_initial_states, \
 torch::Tensor bwd_initial_states, \
 int d_model, int d_state, int d_inner, int num_params, \
 int checkpoint_interval); \
 }

 DECLARE_SG2(sm90)

DECLARE_SG2(gfx942) 
#undef DECLARE_SG2

// ---------------------------------------------------------------------
// Public entry points: thin SG_DISPATCH wrappers around the launchers.
// Each one matches the pre-refactor csrc/common/ops.cpp signature so
// existing Python callers (grokking_optimizers/supergrok2.py) keep
// working unchanged. Reference: ops.cpp@682eab4^ lines 908-1199 for the
// forward/peer-step entries, lines 1499-1611 for the bilevel entries.
// ---------------------------------------------------------------------

// Pre-refactor csrc/common/ops.cpp::supergrok2_mamba_peer_step.
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
 torch::Tensor peer_query_Ws,
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
 torch::Tensor expert_W1, torch::Tensor expert_b1,
 torch::Tensor expert_W2, torch::Tensor expert_b2,
 float rescale, float alpha_mu, float lamb_eff,
 float beta1, float beta2, float lr, float wd_eff, float eps,
 float bc1, float bc2,
 int d_model, int d_state, int d_inner,
 int gru_hidden, int num_heads, int pk_dim,
 int expert_hidden, int num_experts,
 torch::Tensor expert_counts
) {
 if (grad.numel() == 0) return;
 SG_DISPATCH(launch_mamba3_peer_step,
 param, grad, sharpness, exp_avg, exp_avg_sq, mu,
 gru_state, mamba_fwd_state, mamba_bwd_state,
 input_proj_W, input_proj_b,
 mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
 mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
 mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
 mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
 mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
 mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
 gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
 peer_query_Ws, prod_keys_A, prod_keys_B,
 expert_W1, expert_b1, expert_W2, expert_b2,
 rescale, alpha_mu, lamb_eff,
 beta1, beta2, lr, wd_eff, eps, bc1, bc2,
 d_model, d_state, d_inner,
 gru_hidden, num_heads, pk_dim,
 expert_hidden, num_experts, expert_counts);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_mamba_peer_batched_step.
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
 torch::Tensor peer_query_Ws,
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
 torch::Tensor expert_W1, torch::Tensor expert_b1,
 torch::Tensor expert_W2, torch::Tensor expert_b2,
 std::vector<float> alpha_mus, std::vector<float> lamb_effs,
 std::vector<float> beta1s,
 std::vector<float> bc1s, std::vector<float> bc2s,
 float rescale, float beta2, float lr, float wd_eff, float eps,
 int d_model, int d_state, int d_inner,
 int gru_hidden, int num_heads, int pk_dim,
 int expert_hidden, int num_experts,
 torch::Tensor expert_counts
) {
 if (params.empty()) return;
 SG_DISPATCH(launch_mamba3_peer_batched_step,
 params, grads, sharpness_list, exp_avgs, exp_avg_sqs, mus,
 gru_states, mamba_fwd_states, mamba_bwd_states,
 input_proj_W, input_proj_b,
 mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
 mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
 mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
 mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
 mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
 mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
 gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
 peer_query_Ws, prod_keys_A, prod_keys_B,
 expert_W1, expert_b1, expert_W2, expert_b2,
 alpha_mus, lamb_effs, beta1s, bc1s, bc2s,
 rescale, beta2, lr, wd_eff, eps,
 d_model, d_state, d_inner,
 gru_hidden, num_heads, pk_dim,
 expert_hidden, num_experts, expert_counts);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_bilevel_fwd_save.
// Forward scan with state-saving for the meta-net's bilevel backward.
void supergrok2_bilevel_fwd_save(
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
 torch::Tensor fwd_saved_states,
 torch::Tensor fwd_saved_x_branch,
 torch::Tensor fwd_saved_z, torch::Tensor fwd_saved_dt,
 torch::Tensor bwd_saved_states,
 torch::Tensor bwd_saved_x_branch,
 torch::Tensor bwd_saved_z, torch::Tensor bwd_saved_dt,
 torch::Tensor x_sorted, torch::Tensor sort_indices,
 torch::Tensor fwd_initial_state,
 torch::Tensor bwd_initial_state,
 int checkpoint_interval
) {
 if (grad.numel() == 0) return;
 SG_DISPATCH(launch_mamba3_peer_bilevel_fwd_save,
 grad, sharpness,
 input_proj_W, input_proj_b,
 mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
 mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
 mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
 mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
 mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
 mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
 d_model, d_state, d_inner,
 fwd_scan_out, bwd_scan_out,
 fwd_final_state, bwd_final_state,
 fwd_saved_states, fwd_saved_x_branch, fwd_saved_z, fwd_saved_dt,
 bwd_saved_states, bwd_saved_x_branch, bwd_saved_z, bwd_saved_dt,
 x_sorted, sort_indices,
 fwd_initial_state, bwd_initial_state,
 checkpoint_interval);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_bilevel_fwd_save_batched.
void supergrok2_bilevel_fwd_save_batched(
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
 torch::Tensor fwd_scan_out_packed,
 torch::Tensor bwd_scan_out_packed,
 torch::Tensor fwd_saved_states_packed,
 torch::Tensor fwd_saved_xb_packed,
 torch::Tensor fwd_saved_z_packed,
 torch::Tensor fwd_saved_dt_packed,
 torch::Tensor bwd_saved_states_packed,
 torch::Tensor bwd_saved_xb_packed,
 torch::Tensor bwd_saved_z_packed,
 torch::Tensor bwd_saved_dt_packed,
 torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
 torch::Tensor sort_indices_packed,
 torch::Tensor fwd_initial_states,
 torch::Tensor bwd_initial_states,
 int checkpoint_interval
) {
 if (grads.empty()) return;
 SG_DISPATCH(launch_mamba3_peer_bilevel_fwd_save_batched,
 grads, sharpness_list,
 input_proj_W, input_proj_b,
 mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
 mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
 mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
 mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
 mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
 mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
 d_model, d_state, d_inner,
 fwd_scan_out_packed, bwd_scan_out_packed,
 fwd_saved_states_packed, fwd_saved_xb_packed,
 fwd_saved_z_packed, fwd_saved_dt_packed,
 bwd_saved_states_packed, bwd_saved_xb_packed,
 bwd_saved_z_packed, bwd_saved_dt_packed,
 x_sorted_packed, offsets_t, sort_indices_packed,
 fwd_initial_states, bwd_initial_states,
 checkpoint_interval);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_bilevel_backward.
// Full backward through the meta-net, single-tensor entry.
void supergrok2_bilevel_backward(
 torch::Tensor d_smart_grad,
 torch::Tensor grad, torch::Tensor sharpness, float rescale,
 torch::Tensor sort_indices, torch::Tensor x_sorted,
 torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out,
 torch::Tensor fwd_saved_states,
 torch::Tensor fwd_saved_x_branch,
 torch::Tensor fwd_saved_z, torch::Tensor fwd_saved_dt,
 torch::Tensor bwd_saved_states,
 torch::Tensor bwd_saved_x_branch,
 torch::Tensor bwd_saved_z, torch::Tensor bwd_saved_dt,
 torch::Tensor gru_input, torch::Tensor gru_h_old,
 torch::Tensor gru_z_gate, torch::Tensor gru_r_gate,
 torch::Tensor gru_h_tilde,
 torch::Tensor peer_input, torch::Tensor expert_indices,
 torch::Tensor routing_weights, torch::Tensor saved_z_hidden,
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
 torch::Tensor peer_query_Ws,
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
 torch::Tensor expert_W1, torch::Tensor expert_W2,
 torch::Tensor expert_b1_in, torch::Tensor expert_b2_in,
 torch::Tensor input_proj_W,
 torch::Tensor mamba_fwd_init_state,
 torch::Tensor mamba_bwd_init_state,
 torch::Tensor d_mamba_fwd_in_proj,
 torch::Tensor d_mamba_fwd_dt_W,
 torch::Tensor d_mamba_fwd_dt_b,
 torch::Tensor d_mamba_fwd_B_proj,
 torch::Tensor d_mamba_fwd_C_proj,
 torch::Tensor d_mamba_fwd_A_log,
 torch::Tensor d_mamba_fwd_D,
 torch::Tensor d_mamba_fwd_rope,
 torch::Tensor d_mamba_fwd_out_proj,
 torch::Tensor d_mamba_bwd_in_proj,
 torch::Tensor d_mamba_bwd_dt_W,
 torch::Tensor d_mamba_bwd_dt_b,
 torch::Tensor d_mamba_bwd_B_proj,
 torch::Tensor d_mamba_bwd_C_proj,
 torch::Tensor d_mamba_bwd_A_log,
 torch::Tensor d_mamba_bwd_D,
 torch::Tensor d_mamba_bwd_rope,
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
 int checkpoint_interval
) {
 if (d_smart_grad.numel() == 0) return;
 SG_DISPATCH(launch_mamba3_peer_backward,
 d_smart_grad, grad, sharpness, rescale,
 sort_indices, x_sorted,
 fwd_scan_out, bwd_scan_out,
 fwd_saved_states, fwd_saved_x_branch, fwd_saved_z, fwd_saved_dt,
 bwd_saved_states, bwd_saved_x_branch, bwd_saved_z, bwd_saved_dt,
 gru_input, gru_h_old, gru_z_gate, gru_r_gate, gru_h_tilde,
 peer_input, expert_indices, routing_weights, saved_z_hidden,
 saved_scores_a, saved_scores_b,
 saved_top_a_idx, saved_top_b_idx, saved_soft_a, saved_soft_b,
 mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
 mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
 mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
 mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
 mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
 mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
 gru_Wz, gru_Wr, gru_Wh,
 peer_query_Ws, prod_keys_A, prod_keys_B,
 expert_W1, expert_W2, expert_b1_in, expert_b2_in, input_proj_W,
 mamba_fwd_init_state, mamba_bwd_init_state,
 d_mamba_fwd_in_proj, d_mamba_fwd_dt_W, d_mamba_fwd_dt_b,
 d_mamba_fwd_B_proj, d_mamba_fwd_C_proj, d_mamba_fwd_A_log,
 d_mamba_fwd_D, d_mamba_fwd_rope, d_mamba_fwd_out_proj,
 d_mamba_bwd_in_proj, d_mamba_bwd_dt_W, d_mamba_bwd_dt_b,
 d_mamba_bwd_B_proj, d_mamba_bwd_C_proj, d_mamba_bwd_A_log,
 d_mamba_bwd_D, d_mamba_bwd_rope, d_mamba_bwd_out_proj,
 d_gru_Wz, d_gru_bz, d_gru_Wr, d_gru_br, d_gru_Wh, d_gru_bh,
 d_peer_query_Ws, d_prod_keys_A, d_prod_keys_B,
 d_expert_W1, d_expert_b1, d_expert_W2, d_expert_b2,
 d_input_proj_W, d_input_proj_b,
 d_model, d_state, d_inner,
 gru_hidden, gru_input_dim,
 num_heads, topk, pk_dim,
 expert_hidden, peer_input_dim, num_experts,
 checkpoint_interval);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_bilevel_backward_batched.
void supergrok2_bilevel_backward_batched(
 torch::Tensor d_fwd_scan_out_packed,
 torch::Tensor d_bwd_scan_out_packed,
 torch::Tensor x_sorted_packed,
 torch::Tensor fwd_saved_states_packed,
 torch::Tensor fwd_saved_xb_packed,
 torch::Tensor fwd_saved_z_packed,
 torch::Tensor fwd_saved_dt_packed,
 torch::Tensor bwd_saved_states_packed,
 torch::Tensor bwd_saved_xb_packed,
 torch::Tensor bwd_saved_z_packed,
 torch::Tensor bwd_saved_dt_packed,
 torch::Tensor offsets_t,
 torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
 torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
 torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
 torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
 torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
 torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
 torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
 torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
 torch::Tensor d_mamba_fwd_in_proj,
 torch::Tensor d_mamba_fwd_dt_W, torch::Tensor d_mamba_fwd_dt_b,
 torch::Tensor d_mamba_fwd_B_proj,
 torch::Tensor d_mamba_fwd_C_proj,
 torch::Tensor d_mamba_fwd_A_log,
 torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
 torch::Tensor d_mamba_bwd_in_proj,
 torch::Tensor d_mamba_bwd_dt_W, torch::Tensor d_mamba_bwd_dt_b,
 torch::Tensor d_mamba_bwd_B_proj,
 torch::Tensor d_mamba_bwd_C_proj,
 torch::Tensor d_mamba_bwd_A_log,
 torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
 torch::Tensor d_x_sorted_packed,
 torch::Tensor fwd_initial_states,
 torch::Tensor bwd_initial_states,
 int d_model, int d_state, int d_inner, int num_params,
 int checkpoint_interval
) {
 if (num_params == 0) return;
 SG_DISPATCH(launch_mamba3_peer_backward_batched,
 d_fwd_scan_out_packed, d_bwd_scan_out_packed,
 x_sorted_packed,
 fwd_saved_states_packed, fwd_saved_xb_packed,
 fwd_saved_z_packed, fwd_saved_dt_packed,
 bwd_saved_states_packed, bwd_saved_xb_packed,
 bwd_saved_z_packed, bwd_saved_dt_packed,
 offsets_t,
 mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
 mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
 mamba_fwd_D, mamba_fwd_rope,
 mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
 mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
 mamba_bwd_D, mamba_bwd_rope,
 d_mamba_fwd_in_proj, d_mamba_fwd_dt_W, d_mamba_fwd_dt_b,
 d_mamba_fwd_B_proj, d_mamba_fwd_C_proj, d_mamba_fwd_A_log,
 d_mamba_fwd_D, d_mamba_fwd_rope,
 d_mamba_bwd_in_proj, d_mamba_bwd_dt_W, d_mamba_bwd_dt_b,
 d_mamba_bwd_B_proj, d_mamba_bwd_C_proj, d_mamba_bwd_A_log,
 d_mamba_bwd_D, d_mamba_bwd_rope,
 d_x_sorted_packed,
 fwd_initial_states, bwd_initial_states,
 d_model, d_state, d_inner, num_params,
 checkpoint_interval);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_prepare_and_batched_step.
// Lives in csrc/kernels/{cuda,hip}/<arch>/multi_tensor_prepare_<arch>.{cu,hip.cpp}
// (not a launch_X — the per-arch host wrapper for the multi_tensor prepare
// kernel + batched-step pipeline). Each arch translates this to one
// ::sg::<arch>::supergrok2_prepare_and_batched_step.
void supergrok2_prepare_and_batched_step(
 std::vector<torch::Tensor> params,
 std::vector<torch::Tensor> grads,
 std::vector<torch::Tensor> exp_avgs,
 std::vector<torch::Tensor> exp_avg_sqs,
 std::vector<torch::Tensor> mamba_fwd_states,
 std::vector<torch::Tensor> mamba_bwd_states,
 std::vector<torch::Tensor> gru_states,
 std::vector<torch::Tensor> mus,
 std::vector<torch::Tensor> sharpnesses,
 std::vector<int64_t> steps,
 std::vector<double> layer_alphas,
 std::vector<double> layer_beta1s,
 double base_alpha, double gradient_clipping,
 double beta2, double lr, double eps, double wd,
 double lamb, double ramp, double gate_signal,
 torch::Tensor mamba_fwd_A, torch::Tensor mamba_fwd_B,
 torch::Tensor mamba_fwd_C, torch::Tensor mamba_fwd_D,
 torch::Tensor mamba_fwd_dt,
 torch::Tensor mamba_bwd_A, torch::Tensor mamba_bwd_B,
 torch::Tensor mamba_bwd_C, torch::Tensor mamba_bwd_D,
 torch::Tensor mamba_bwd_dt,
 torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh,
 torch::Tensor gru_bz, torch::Tensor gru_br, torch::Tensor gru_bh,
 torch::Tensor peer_query_Ws,
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
 torch::Tensor value_proj_W,
 int64_t d_inner, int64_t d_state, int64_t n_experts, int64_t topk
) {
 if (params.empty()) return;
 SG_DISPATCH(supergrok2_prepare_and_batched_step,
 params, grads, exp_avgs, exp_avg_sqs,
 mamba_fwd_states, mamba_bwd_states, gru_states, mus, sharpnesses,
 steps, layer_alphas, layer_beta1s,
 base_alpha, gradient_clipping, beta2, lr, eps, wd,
 lamb, ramp, gate_signal,
 mamba_fwd_A, mamba_fwd_B, mamba_fwd_C, mamba_fwd_D, mamba_fwd_dt,
 mamba_bwd_A, mamba_bwd_B, mamba_bwd_C, mamba_bwd_D, mamba_bwd_dt,
 gru_Wz, gru_Wr, gru_Wh, gru_bz, gru_br, gru_bh,
 peer_query_Ws, prod_keys_A, prod_keys_B, value_proj_W,
 d_inner, d_state, n_experts, topk);
}

} // namespace sg


// ─── csrc/bindings/models_decoder.cpp ──────────────────────────────
// =====================================================================
// bindings/models_decoder.cpp — pybind11 wrapper for Decoder Transformer
//
// Follows the SG_DISPATCH pattern used by optimizer bindings. Provides
// forward/backward entry points for the full decoder and separate
// attention entry points for component-level testing.
//
// Per-arch launchers live in csrc/kernels/{cuda/<sm>,hip/<gfx>}/ and
// are forward-declared here. The full template definitions live in
// csrc/kernels/cuda/sm_90/models/decoder.cuh and are explicitly
// instantiated in csrc/kernels/cuda/sm_90/models/decoder.cu.
//
// Pybind11 registration is done from models_module.cpp.
// =====================================================================


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------
// Forward declarations for per-arch launchers.
// (We avoid including the .cuh headers here because they contain
// __global__ kernels that don't compile under the host C++ compiler.
// Instantiations live in csrc/kernels/cuda/sm_90/models/decoder.cu.)
// ---------------------------------------------------------------------

namespace sg { namespace sm90 { namespace models { namespace attention {
 template <typename ActT, int kHeadDim, bool kCausal>
 cudaError_t attention_forward(
 const ActT* q, const ActT* k, const ActT* v,
 ActT* out, ActT* softmax_lse_act,
 int batch, int n_heads, int seq_len,
 float scale, cudaStream_t stream);
 template <typename ActT, int kHeadDim, bool kCausal>
 cudaError_t attention_backward(
 const ActT* grad_out,
 const ActT* q, const ActT* k, const ActT* v,
 const ActT* out, const ActT* softmax_lse_act,
 ActT* grad_q, ActT* grad_k, ActT* grad_v,
 int batch, int n_heads, int seq_len,
 float scale, cudaStream_t stream);
}}}}

namespace sg { namespace sm90 { namespace models { namespace decoder {
 template <typename ActT, typename WeightT>
 cudaError_t forward(const ActT*, const WeightT*, ActT*, ActT*,
 int batch, int seq_len, int d_model,
 int n_heads, int d_head, int n_layers, int vocab_size,
 int ffn_expansion, cudaStream_t);
 template <typename ActT, typename WeightT>
 cudaError_t backward(const ActT*, const ActT*, const WeightT*,
 ActT*, WeightT*,
 int batch, int seq_len, int d_model,
 int n_heads, int d_head, int n_layers, int vocab_size,
 int ffn_expansion, cudaStream_t);
}}}}

namespace sg { namespace gfx942 { namespace models { namespace decoder {
 template <typename ActT, typename WeightT>
 cudaError_t forward(const ActT*, const WeightT*, ActT*, ActT*,
 int batch, int seq_len, int d_model,
 int n_heads, int d_head, int n_layers, int vocab_size,
 int ffn_expansion, cudaStream_t);
 template <typename ActT, typename WeightT>
 cudaError_t backward(const ActT*, const ActT*, const WeightT*,
 ActT*, WeightT*,
 int batch, int seq_len, int d_model,
 int n_heads, int d_head, int n_layers, int vocab_size,
 int ffn_expansion, cudaStream_t);
}}}}

// ---------------------------------------------------------------------
// Public entry points (called from models_module.cpp registration)
// ---------------------------------------------------------------------

namespace sg {

namespace {

inline void check_cuda_err(cudaError_t err, const char* what) {
 TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

} // anonymous namespace

void decoder_forward(
 torch::Tensor input,
 torch::Tensor weights,
 torch::Tensor output,
 torch::Tensor activations,
 int batch_size, int seq_len, int d_model, int n_heads,
 int d_head, int n_layers, int vocab_size, int ffn_expansion
) {
 auto stream = at::cuda::getCurrentCUDAStream().stream();
 auto t = weights.scalar_type();
 cudaError_t err;
 if (t == torch::kFloat32) {
 err = sg::sm90::models::decoder::forward<float, float>(
 input.data_ptr<float>(), weights.data_ptr<float>(),
 output.data_ptr<float>(), activations.data_ptr<float>(),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else if (t == torch::kBFloat16) {
 err = sg::sm90::models::decoder::forward<__nv_bfloat16, __nv_bfloat16>(
 reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(activations.data_ptr()),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else if (t == torch::kFloat16) {
 err = sg::sm90::models::decoder::forward<__half, __half>(
 reinterpret_cast<const __half*>(input.data_ptr()),
 reinterpret_cast<const __half*>(weights.data_ptr()),
 reinterpret_cast<__half*>(output.data_ptr()),
 reinterpret_cast<__half*>(activations.data_ptr()),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else {
 TORCH_CHECK(false, "decoder_forward: unsupported dtype ", t);
 }
 check_cuda_err(err, "decoder_forward");
}

void decoder_backward(
 torch::Tensor grad_output,
 torch::Tensor activations_saved,
 torch::Tensor weights,
 torch::Tensor grad_input,
 torch::Tensor grad_weights,
 int batch_size, int seq_len, int d_model, int n_heads,
 int d_head, int n_layers, int vocab_size, int ffn_expansion
) {
 auto stream = at::cuda::getCurrentCUDAStream().stream();
 auto t = weights.scalar_type();
 cudaError_t err;
 if (t == torch::kFloat32) {
 err = sg::sm90::models::decoder::backward<float, float>(
 grad_output.data_ptr<float>(), activations_saved.data_ptr<float>(),
 weights.data_ptr<float>(),
 grad_input.data_ptr<float>(), grad_weights.data_ptr<float>(),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else if (t == torch::kBFloat16) {
 err = sg::sm90::models::decoder::backward<__nv_bfloat16, __nv_bfloat16>(
 reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(activations_saved.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(grad_weights.data_ptr()),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else if (t == torch::kFloat16) {
 err = sg::sm90::models::decoder::backward<__half, __half>(
 reinterpret_cast<const __half*>(grad_output.data_ptr()),
 reinterpret_cast<const __half*>(activations_saved.data_ptr()),
 reinterpret_cast<const __half*>(weights.data_ptr()),
 reinterpret_cast<__half*>(grad_input.data_ptr()),
 reinterpret_cast<__half*>(grad_weights.data_ptr()),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else {
 TORCH_CHECK(false, "decoder_backward: unsupported dtype ", t);
 }
 check_cuda_err(err, "decoder_backward");
}

void decoder_attention_forward(
 torch::Tensor q, torch::Tensor k, torch::Tensor v,
 torch::Tensor out, torch::Tensor softmax_lse,
 int batch, int n_heads, int seq_len, float scale
) {
 auto stream = at::cuda::getCurrentCUDAStream().stream();
 auto t = q.scalar_type();
 cudaError_t err;
 if (t == torch::kFloat32) {
 err = sg::sm90::models::attention::attention_forward<float, 32, true>(
 q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
 out.data_ptr<float>(),
 reinterpret_cast<float*>(softmax_lse.data_ptr()),
 batch, n_heads, seq_len, scale, stream);
 } else if (t == torch::kBFloat16) {
 err = sg::sm90::models::attention::attention_forward<__nv_bfloat16, 32, true>(
 reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(softmax_lse.data_ptr()),
 batch, n_heads, seq_len, scale, stream);
 } else if (t == torch::kFloat16) {
 err = sg::sm90::models::attention::attention_forward<__half, 32, true>(
 reinterpret_cast<const __half*>(q.data_ptr()),
 reinterpret_cast<const __half*>(k.data_ptr()),
 reinterpret_cast<const __half*>(v.data_ptr()),
 reinterpret_cast<__half*>(out.data_ptr()),
 reinterpret_cast<__half*>(softmax_lse.data_ptr()),
 batch, n_heads, seq_len, scale, stream);
 } else {
 TORCH_CHECK(false, "decoder_attention_forward: unsupported dtype ", t);
 }
 check_cuda_err(err, "decoder_attention_forward");
}

void decoder_attention_backward(
 torch::Tensor grad_out,
 torch::Tensor q, torch::Tensor k, torch::Tensor v,
 torch::Tensor out, torch::Tensor softmax_lse,
 torch::Tensor grad_q, torch::Tensor grad_k, torch::Tensor grad_v,
 int batch, int n_heads, int seq_len, float scale
) {
 auto stream = at::cuda::getCurrentCUDAStream().stream();
 auto t = q.scalar_type();
 cudaError_t err;
 if (t == torch::kFloat32) {
 err = sg::sm90::models::attention::attention_backward<float, 32, true>(
 grad_out.data_ptr<float>(),
 q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
 out.data_ptr<float>(),
 reinterpret_cast<const float*>(softmax_lse.data_ptr()),
 grad_q.data_ptr<float>(), grad_k.data_ptr<float>(), grad_v.data_ptr<float>(),
 batch, n_heads, seq_len, scale, stream);
 } else if (t == torch::kBFloat16) {
 err = sg::sm90::models::attention::attention_backward<__nv_bfloat16, 32, true>(
 reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(out.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(softmax_lse.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(grad_q.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(grad_k.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(grad_v.data_ptr()),
 batch, n_heads, seq_len, scale, stream);
 } else if (t == torch::kFloat16) {
 err = sg::sm90::models::attention::attention_backward<__half, 32, true>(
 reinterpret_cast<const __half*>(grad_out.data_ptr()),
 reinterpret_cast<const __half*>(q.data_ptr()),
 reinterpret_cast<const __half*>(k.data_ptr()),
 reinterpret_cast<const __half*>(v.data_ptr()),
 reinterpret_cast<const __half*>(out.data_ptr()),
 reinterpret_cast<const __half*>(softmax_lse.data_ptr()),
 reinterpret_cast<__half*>(grad_q.data_ptr()),
 reinterpret_cast<__half*>(grad_k.data_ptr()),
 reinterpret_cast<__half*>(grad_v.data_ptr()),
 batch, n_heads, seq_len, scale, stream);
 } else {
 TORCH_CHECK(false, "decoder_attention_backward: unsupported dtype ", t);
 }
 check_cuda_err(err, "decoder_attention_backward");
}

} // namespace sg


// ─── csrc/bindings/models_vit.cpp ──────────────────────────────────
// =====================================================================
// bindings/models_vit.cpp — pybind11 wrapper for Vision Transformer
//
// Follows the SG_DISPATCH pattern used by optimizer bindings. Provides
// forward/backward entry points for the full ViT, separate attention
// entry points, and a patch projection entry point for testing.
//
// Per-arch launchers live in csrc/kernels/{cuda/<sm>,hip/<gfx>}/ and
// are forward-declared here.
//
// Pybind11 registration is done from models_module.cpp.
// =====================================================================


#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

// ---------------------------------------------------------------------
// Forward declarations for per-arch launchers.
// ---------------------------------------------------------------------

namespace sg { namespace sm90 { namespace models { namespace vit {
 template <typename ActT, typename WeightT>
 cudaError_t forward(const ActT*, const WeightT*, ActT*, ActT*,
 int batch, int channels, int height, int width,
 int patch_size, int d_model, int n_heads, int d_head,
 int n_layers, int n_classes, int ffn_expansion,
 cudaStream_t);
 template <typename ActT, typename WeightT>
 cudaError_t backward(const ActT*, const ActT*, const WeightT*,
 ActT*, WeightT*,
 int batch, int channels, int height, int width,
 int patch_size, int d_model, int n_heads, int d_head,
 int n_layers, int n_classes, int ffn_expansion,
 cudaStream_t);
 template <typename ActT, typename WeightT>
 cudaError_t patch_project(const ActT*, const WeightT*, const WeightT*, ActT*,
 int batch, int num_patches, int patch_dim, int d_model,
 cudaStream_t);
}}}}

namespace sg { namespace sm90 { namespace models { namespace attention {
 template <typename ActT, int kHeadDim, bool kCausal>
 cudaError_t attention_forward(
 const ActT* q, const ActT* k, const ActT* v, ActT* out,
 ActT* softmax_lse_act,
 int batch, int n_heads, int seq_len, float scale, cudaStream_t);
 template <typename ActT, int kHeadDim, bool kCausal>
 cudaError_t attention_backward(
 const ActT* grad_out, const ActT* q, const ActT* k, const ActT* v,
 const ActT* out, const ActT* softmax_lse_act,
 ActT* grad_q, ActT* grad_k, ActT* grad_v,
 int batch, int n_heads, int seq_len, float scale, cudaStream_t);
}}}}

namespace sg { namespace gfx942 { namespace models { namespace vit {
 template <typename ActT, typename WeightT>
 cudaError_t forward(const ActT*, const WeightT*, ActT*, ActT*,
 int batch, int channels, int height, int width,
 int patch_size, int d_model, int n_heads, int d_head,
 int n_layers, int n_classes, int ffn_expansion,
 cudaStream_t);
 template <typename ActT, typename WeightT>
 cudaError_t backward(const ActT*, const ActT*, const WeightT*,
 ActT*, WeightT*,
 int batch, int channels, int height, int width,
 int patch_size, int d_model, int n_heads, int d_head,
 int n_layers, int n_classes, int ffn_expansion,
 cudaStream_t);
}}}}

// ---------------------------------------------------------------------
// Public entry points (called from models_module.cpp registration)
// ---------------------------------------------------------------------

namespace sg {

namespace {

inline cudaStream_t default_stream() {
 return at::cuda::getCurrentCUDAStream().stream();
}

template <typename ActT>
cudaError_t do_forward(
 const void* input, const void* weights, void* output, void* activations,
 int batch, int channels, int height, int width,
 int patch_size, int d_model, int n_heads, int d_head,
 int n_layers, int n_classes, int ffn_expansion,
 cudaStream_t stream
) {
 int a = ::sg::detect_arch();
 if (a == 90) {
 return sg::sm90::models::vit::forward<ActT, ActT>(
 static_cast<const ActT*>(input),
 static_cast<const ActT*>(weights),
 static_cast<ActT*>(output),
 static_cast<ActT*>(activations),
 batch, channels, height, width,
 patch_size, d_model, n_heads, d_head,
 n_layers, n_classes, ffn_expansion, stream);
 }
 if (a == 942) {
 return sg::gfx942::models::vit::forward<ActT, ActT>(
 static_cast<const ActT*>(input),
 static_cast<const ActT*>(weights),
 static_cast<ActT*>(output),
 static_cast<ActT*>(activations),
 batch, channels, height, width,
 patch_size, d_model, n_heads, d_head,
 n_layers, n_classes, ffn_expansion, stream);
 }
 throw std::runtime_error("vit_forward: unsupported arch " + std::to_string(a));
}

template <typename ActT>
cudaError_t do_backward(
 const void* grad_output, const void* activations_saved, const void* weights,
 void* grad_input, void* grad_weights,
 int batch, int channels, int height, int width,
 int patch_size, int d_model, int n_heads, int d_head,
 int n_layers, int n_classes, int ffn_expansion,
 cudaStream_t stream
) {
 int a = ::sg::detect_arch();
 if (a == 90) {
 return sg::sm90::models::vit::backward<ActT, ActT>(
 static_cast<const ActT*>(grad_output),
 static_cast<const ActT*>(activations_saved),
 static_cast<const ActT*>(weights),
 static_cast<ActT*>(grad_input),
 static_cast<ActT*>(grad_weights),
 batch, channels, height, width,
 patch_size, d_model, n_heads, d_head,
 n_layers, n_classes, ffn_expansion, stream);
 }
 if (a == 942) {
 return sg::gfx942::models::vit::backward<ActT, ActT>(
 static_cast<const ActT*>(grad_output),
 static_cast<const ActT*>(activations_saved),
 static_cast<const ActT*>(weights),
 static_cast<ActT*>(grad_input),
 static_cast<ActT*>(grad_weights),
 batch, channels, height, width,
 patch_size, d_model, n_heads, d_head,
 n_layers, n_classes, ffn_expansion, stream);
 }
 throw std::runtime_error("vit_backward: unsupported arch " + std::to_string(a));
}

inline void check_cuda(cudaError_t err, const char* what) {
 TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

} // anonymous namespace

void vit_forward(
 torch::Tensor input,
 torch::Tensor weights,
 torch::Tensor output,
 torch::Tensor activations,
 int batch_size, int channels, int height, int width,
 int patch_size, int d_model, int n_heads, int d_head,
 int n_layers, int n_classes, int ffn_expansion
) {
 TORCH_CHECK(input.is_cuda(), "vit_forward: input must be on CUDA");
 TORCH_CHECK(weights.scalar_type() == input.scalar_type(),
 "vit_forward: weights/input dtype mismatch");
 TORCH_CHECK(output.scalar_type() == input.scalar_type(),
 "vit_forward: output/input dtype mismatch");
 cudaStream_t stream = default_stream();
 cudaError_t err = cudaSuccess;
 auto t = input.scalar_type();
 if (t == torch::kFloat32) {
 err = do_forward<float>(input.data_ptr(), weights.data_ptr(),
 output.data_ptr(), activations.data_ptr(),
 batch_size, channels, height, width,
 patch_size, d_model, n_heads, d_head,
 n_layers, n_classes, ffn_expansion, stream);
 } else if (t == torch::kBFloat16) {
 err = do_forward<__nv_bfloat16>(input.data_ptr(), weights.data_ptr(),
 output.data_ptr(), activations.data_ptr(),
 batch_size, channels, height, width,
 patch_size, d_model, n_heads, d_head,
 n_layers, n_classes, ffn_expansion, stream);
 } else if (t == torch::kFloat16) {
 err = do_forward<__half>(input.data_ptr(), weights.data_ptr(),
 output.data_ptr(), activations.data_ptr(),
 batch_size, channels, height, width,
 patch_size, d_model, n_heads, d_head,
 n_layers, n_classes, ffn_expansion, stream);
 } else {
 TORCH_CHECK(false, "vit_forward: unsupported dtype ", t);
 }
 check_cuda(err, "vit_forward");
}

void vit_backward(
 torch::Tensor grad_output,
 torch::Tensor activations_saved,
 torch::Tensor weights,
 torch::Tensor grad_input,
 torch::Tensor grad_weights,
 int batch_size, int channels, int height, int width,
 int patch_size, int d_model, int n_heads, int d_head,
 int n_layers, int n_classes, int ffn_expansion
) {
 TORCH_CHECK(grad_output.is_cuda(), "vit_backward: grad_output must be on CUDA");
 TORCH_CHECK(weights.scalar_type() == grad_output.scalar_type(),
 "vit_backward: weights/grad_output dtype mismatch");
 cudaStream_t stream = default_stream();
 cudaError_t err = cudaSuccess;
 auto t = grad_output.scalar_type();
 if (t == torch::kFloat32) {
 err = do_backward<float>(
 grad_output.data_ptr(), activations_saved.data_ptr(), weights.data_ptr(),
 grad_input.data_ptr(), grad_weights.data_ptr(),
 batch_size, channels, height, width,
 patch_size, d_model, n_heads, d_head,
 n_layers, n_classes, ffn_expansion, stream);
 } else if (t == torch::kBFloat16) {
 err = do_backward<__nv_bfloat16>(
 grad_output.data_ptr(), activations_saved.data_ptr(), weights.data_ptr(),
 grad_input.data_ptr(), grad_weights.data_ptr(),
 batch_size, channels, height, width,
 patch_size, d_model, n_heads, d_head,
 n_layers, n_classes, ffn_expansion, stream);
 } else if (t == torch::kFloat16) {
 err = do_backward<__half>(
 grad_output.data_ptr(), activations_saved.data_ptr(), weights.data_ptr(),
 grad_input.data_ptr(), grad_weights.data_ptr(),
 batch_size, channels, height, width,
 patch_size, d_model, n_heads, d_head,
 n_layers, n_classes, ffn_expansion, stream);
 } else {
 TORCH_CHECK(false, "vit_backward: unsupported dtype ", t);
 }
 check_cuda(err, "vit_backward");
}

void vit_attention_forward(
 torch::Tensor q, torch::Tensor k, torch::Tensor v,
 torch::Tensor out, torch::Tensor softmax_lse,
 int batch, int n_heads, int seq_len, float scale
) {
 TORCH_CHECK(q.is_cuda(), "vit_attention_forward: tensors must be CUDA");
 TORCH_CHECK(q.scalar_type() == torch::kFloat32,
 "vit_attention_forward: only fp32 is wired through this entry; "
 "use vit_forward for bf16/fp16");
 cudaStream_t stream = default_stream();
 cudaError_t err = sg::sm90::models::attention::attention_forward<
 float, /*kHeadDim=*/32, /*kCausal=*/false>(
 q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
 out.data_ptr<float>(),
 softmax_lse.data_ptr<float>(),
 batch, n_heads, seq_len, scale, stream);
 check_cuda(err, "vit_attention_forward");
}

void vit_attention_backward(
 torch::Tensor grad_out,
 torch::Tensor q, torch::Tensor k, torch::Tensor v,
 torch::Tensor out, torch::Tensor softmax_lse,
 torch::Tensor grad_q, torch::Tensor grad_k, torch::Tensor grad_v,
 int batch, int n_heads, int seq_len, float scale
) {
 TORCH_CHECK(q.is_cuda(), "vit_attention_backward: tensors must be CUDA");
 TORCH_CHECK(q.scalar_type() == torch::kFloat32,
 "vit_attention_backward: only fp32 is wired through this entry");
 cudaStream_t stream = default_stream();
 cudaError_t err = sg::sm90::models::attention::attention_backward<
 float, /*kHeadDim=*/32, /*kCausal=*/false>(
 grad_out.data_ptr<float>(),
 q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
 out.data_ptr<float>(),
 softmax_lse.data_ptr<float>(),
 grad_q.data_ptr<float>(), grad_k.data_ptr<float>(), grad_v.data_ptr<float>(),
 batch, n_heads, seq_len, scale, stream);
 check_cuda(err, "vit_attention_backward");
}

void vit_patch_project(
 torch::Tensor input,
 torch::Tensor weight,
 torch::Tensor bias,
 torch::Tensor output,
 int batch, int channels, int height, int width,
 int patch_size, int d_model
) {
 TORCH_CHECK(input.is_cuda(), "vit_patch_project: input must be CUDA");
 TORCH_CHECK(height % patch_size == 0 && width % patch_size == 0,
 "vit_patch_project: image dims must be divisible by patch_size");
 int num_patches = (height / patch_size) * (width / patch_size);
 int patch_dim = patch_size * patch_size * channels;
 TORCH_CHECK(input.numel() == (int64_t)batch * num_patches * patch_dim,
 "vit_patch_project: input must be [B*num_patches*patch_dim]");
 TORCH_CHECK(output.numel() == (int64_t)batch * num_patches * d_model,
 "vit_patch_project: output must be [B*num_patches*d_model]");
 cudaStream_t stream = default_stream();
 cudaError_t err = cudaSuccess;
 auto t = input.scalar_type();
 const void* bias_ptr = bias.defined() && bias.numel() > 0 ? bias.data_ptr() : nullptr;
 int a = ::sg::detect_arch();
 if (a != 90) {
 throw std::runtime_error(
 "vit_patch_project: unsupported arch " + std::to_string(a));
 }
 if (t == torch::kFloat32) {
 err = sg::sm90::models::vit::patch_project<float, float>(
 input.data_ptr<float>(),
 weight.data_ptr<float>(),
 static_cast<const float*>(bias_ptr),
 output.data_ptr<float>(),
 batch, num_patches, patch_dim, d_model, stream);
 } else if (t == torch::kBFloat16) {
 err = sg::sm90::models::vit::patch_project<__nv_bfloat16, __nv_bfloat16>(
 reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
 static_cast<const __nv_bfloat16*>(bias_ptr),
 reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
 batch, num_patches, patch_dim, d_model, stream);
 } else if (t == torch::kFloat16) {
 err = sg::sm90::models::vit::patch_project<__half, __half>(
 reinterpret_cast<const __half*>(input.data_ptr()),
 reinterpret_cast<const __half*>(weight.data_ptr()),
 static_cast<const __half*>(bias_ptr),
 reinterpret_cast<__half*>(output.data_ptr()),
 batch, num_patches, patch_dim, d_model, stream);
 } else {
 TORCH_CHECK(false, "vit_patch_project: unsupported dtype ", t);
 }
 check_cuda(err, "vit_patch_project");
}

} // namespace sg


// ─── csrc/bindings/models_mamba.cpp ────────────────────────────────
// =====================================================================
// bindings/models_mamba.cpp — pybind11 wrapper for Mamba (SSM) model
//
// Follows the SG_DISPATCH pattern used by optimizer bindings. Provides
// forward/backward entry points for the full Mamba model, per-layer
// forward, and selective scan primitives for component-level testing.
//
// Per-arch launchers live in csrc/kernels/{cuda/<sm>,hip/<gfx>}/ and
// are forward-declared here.
//
// Pybind11 registration is done from models_module.cpp.
// =====================================================================


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------
// Forward declarations for per-arch launchers.
// ---------------------------------------------------------------------

namespace sg { namespace sm90 { namespace models { namespace mamba {
 template <typename T>
 cudaError_t forward(const T*, const T*, T*, T*,
 int batch, int seq_len, int d_model, int d_state,
 int d_conv, int expand, int n_layers, cudaStream_t);
 template <typename T>
 cudaError_t backward(const T*, const T*, const T*,
 T*, T*,
 int batch, int seq_len, int d_model, int d_state,
 int d_conv, int expand, int n_layers, cudaStream_t);
 template <typename T>
 cudaError_t selective_scan_fwd(const T*, const T*, const T*, const T*,
 T*, T*,
 int batch, int seq_len, int d_state,
 cudaStream_t);
 template <typename T>
 cudaError_t selective_scan_bwd(const T*, const T*, const T*, const T*,
 const T*, const T*,
 T*, T*, T*, T*,
 int batch, int seq_len, int d_state,
 cudaStream_t);
}}}}

namespace sg { namespace gfx942 { namespace models { namespace mamba {
 template <typename T>
 cudaError_t forward(const T*, const T*, T*, T*,
 int batch, int seq_len, int d_model, int d_state,
 int d_conv, int expand, int n_layers, cudaStream_t);
 template <typename T>
 cudaError_t backward(const T*, const T*, const T*,
 T*, T*,
 int batch, int seq_len, int d_model, int d_state,
 int d_conv, int expand, int n_layers, cudaStream_t);
 template <typename T>
 cudaError_t selective_scan_fwd(const T*, const T*, const T*, const T*,
 T*, T*,
 int batch, int seq_len, int d_state,
 cudaStream_t);
 template <typename T>
 cudaError_t selective_scan_bwd(const T*, const T*, const T*, const T*,
 const T*, const T*,
 T*, T*, T*, T*,
 int batch, int seq_len, int d_state,
 cudaStream_t);
}}}}

// ---------------------------------------------------------------------
// Helpers: cast typed pointer view of a torch tensor.
// ---------------------------------------------------------------------

namespace {

template <typename T>
inline T* tptr(torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
template <typename T>
inline const T* ctptr(const torch::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }

inline cudaStream_t cur_stream() {
 return c10::cuda::getCurrentCUDAStream().stream();
}

inline void check_cuda(cudaError_t err, const char* what) {
 TORCH_CHECK(err == cudaSuccess,
 "Mamba kernel ", what, " failed: ", cudaGetErrorString(err));
}

} // anonymous namespace

// ---------------------------------------------------------------------
// Public entry points (called from models_module.cpp registration)
// ---------------------------------------------------------------------

namespace sg {

void mamba_forward(
 torch::Tensor input,
 torch::Tensor weights,
 torch::Tensor output,
 torch::Tensor states,
 int batch, int seq_len, int d_model, int d_state,
 int d_conv, int expand, int n_layers
) {
 TORCH_CHECK(input.is_cuda() && weights.is_cuda() && output.is_cuda()
 && states.is_cuda(),
 "mamba_forward: all tensors must be on CUDA");
 TORCH_CHECK(input.scalar_type() == torch::kInt32,
 "mamba_forward: input (token ids) must be int32");
 auto stream = cur_stream();
 const int sg_arch = ::sg::detect_arch();

 auto run = [&](auto dummy) {
 using scalar_t = decltype(dummy);
 cudaError_t err;
 if (sg_arch == 90) {
 err = ::sg::sm90::models::mamba::forward<scalar_t>(
 reinterpret_cast<const scalar_t*>(input.data_ptr()),
 ctptr<scalar_t>(weights),
 tptr<scalar_t>(output),
 tptr<scalar_t>(states),
 batch, seq_len, d_model, d_state,
 d_conv, expand, n_layers, stream);
 } else if (sg_arch == 942) {
 err = ::sg::gfx942::models::mamba::forward<scalar_t>(
 reinterpret_cast<const scalar_t*>(input.data_ptr()),
 ctptr<scalar_t>(weights),
 tptr<scalar_t>(output),
 tptr<scalar_t>(states),
 batch, seq_len, d_model, d_state,
 d_conv, expand, n_layers, stream);
 } else {
 TORCH_CHECK(false, "mamba_forward: unsupported arch ", sg_arch);
 }
 check_cuda(err, "forward");
 };

 switch (weights.scalar_type()) {
 case torch::kFloat32: run(float{}); break;
 case torch::kFloat16: run(__half{}); break;
 case torch::kBFloat16: run(__nv_bfloat16{}); break;
 default:
 TORCH_CHECK(false, "mamba_forward: unsupported weight dtype ",
 weights.scalar_type());
 }
}

void mamba_backward(
 torch::Tensor grad_output,
 torch::Tensor states_saved,
 torch::Tensor weights,
 torch::Tensor grad_input,
 torch::Tensor grad_weights,
 int batch, int seq_len, int d_model, int d_state,
 int d_conv, int expand, int n_layers
) {
 TORCH_CHECK(grad_output.is_cuda() && weights.is_cuda(),
 "mamba_backward: tensors must be on CUDA");
 auto stream = cur_stream();
 const int sg_arch = ::sg::detect_arch();

 auto run = [&](auto dummy) {
 using scalar_t = decltype(dummy);
 cudaError_t err;
 if (sg_arch == 90) {
 err = ::sg::sm90::models::mamba::backward<scalar_t>(
 ctptr<scalar_t>(grad_output),
 ctptr<scalar_t>(states_saved),
 ctptr<scalar_t>(weights),
 tptr<scalar_t>(grad_input),
 tptr<scalar_t>(grad_weights),
 batch, seq_len, d_model, d_state,
 d_conv, expand, n_layers, stream);
 } else if (sg_arch == 942) {
 err = ::sg::gfx942::models::mamba::backward<scalar_t>(
 ctptr<scalar_t>(grad_output),
 ctptr<scalar_t>(states_saved),
 ctptr<scalar_t>(weights),
 tptr<scalar_t>(grad_input),
 tptr<scalar_t>(grad_weights),
 batch, seq_len, d_model, d_state,
 d_conv, expand, n_layers, stream);
 } else {
 TORCH_CHECK(false, "mamba_backward: unsupported arch ", sg_arch);
 }
 check_cuda(err, "backward");
 };

 switch (weights.scalar_type()) {
 case torch::kFloat32: run(float{}); break;
 case torch::kFloat16: run(__half{}); break;
 case torch::kBFloat16: run(__nv_bfloat16{}); break;
 default:
 TORCH_CHECK(false, "mamba_backward: unsupported weight dtype ",
 weights.scalar_type());
 }
}

void mamba_layer_forward(
 torch::Tensor input,
 torch::Tensor weights,
 torch::Tensor output,
 torch::Tensor state,
 int batch, int seq_len, int d_model, int d_state,
 int d_conv, int expand
) {
 // Per-layer forward: a single-layer forward pass routed via the same
 // stack with n_layers=1. The caller is responsible for providing weights
 // shaped for a single layer (i.e. embeddings sized for an identity
 // pass-through, or just the layer block).
 mamba_forward(input, weights, output, state,
 batch, seq_len, d_model, d_state,
 d_conv, expand, /*n_layers=*/1);
}

void mamba_selective_scan_forward(
 torch::Tensor u, torch::Tensor delta,
 torch::Tensor A, torch::Tensor B,
 torch::Tensor out, torch::Tensor state,
 int batch, int seq_len, int d_state
) {
 TORCH_CHECK(u.is_cuda() && delta.is_cuda() && A.is_cuda() && B.is_cuda()
 && out.is_cuda() && state.is_cuda(),
 "mamba_selective_scan_forward: all tensors must be on CUDA");
 auto stream = cur_stream();
 const int sg_arch = ::sg::detect_arch();

 auto run = [&](auto dummy) {
 using scalar_t = decltype(dummy);
 cudaError_t err;
 if (sg_arch == 90) {
 err = ::sg::sm90::models::mamba::selective_scan_fwd<scalar_t>(
 ctptr<scalar_t>(u), ctptr<scalar_t>(delta),
 ctptr<scalar_t>(A), ctptr<scalar_t>(B),
 tptr<scalar_t>(out), tptr<scalar_t>(state),
 batch, seq_len, d_state, stream);
 } else if (sg_arch == 942) {
 err = ::sg::gfx942::models::mamba::selective_scan_fwd<scalar_t>(
 ctptr<scalar_t>(u), ctptr<scalar_t>(delta),
 ctptr<scalar_t>(A), ctptr<scalar_t>(B),
 tptr<scalar_t>(out), tptr<scalar_t>(state),
 batch, seq_len, d_state, stream);
 } else {
 TORCH_CHECK(false,
 "mamba_selective_scan_forward: unsupported arch ", sg_arch);
 }
 check_cuda(err, "selective_scan_forward");
 };

 switch (u.scalar_type()) {
 case torch::kFloat32: run(float{}); break;
 case torch::kFloat16: run(__half{}); break;
 case torch::kBFloat16: run(__nv_bfloat16{}); break;
 default:
 TORCH_CHECK(false,
 "mamba_selective_scan_forward: unsupported dtype ",
 u.scalar_type());
 }
}

void mamba_selective_scan_backward(
 torch::Tensor grad_out,
 torch::Tensor u, torch::Tensor delta,
 torch::Tensor A, torch::Tensor B,
 torch::Tensor out, torch::Tensor state,
 torch::Tensor grad_u, torch::Tensor grad_delta,
 torch::Tensor grad_A, torch::Tensor grad_B,
 int batch, int seq_len, int d_state
) {
 (void)out; // not used: backward recomputes h from forward inputs
 TORCH_CHECK(grad_out.is_cuda() && u.is_cuda() && delta.is_cuda()
 && A.is_cuda() && B.is_cuda() && state.is_cuda(),
 "mamba_selective_scan_backward: all tensors must be on CUDA");
 auto stream = cur_stream();
 const int sg_arch = ::sg::detect_arch();

 auto run = [&](auto dummy) {
 using scalar_t = decltype(dummy);
 cudaError_t err;
 if (sg_arch == 90) {
 err = ::sg::sm90::models::mamba::selective_scan_bwd<scalar_t>(
 ctptr<scalar_t>(grad_out),
 ctptr<scalar_t>(u), ctptr<scalar_t>(delta),
 ctptr<scalar_t>(A), ctptr<scalar_t>(B),
 ctptr<scalar_t>(state),
 tptr<scalar_t>(grad_u), tptr<scalar_t>(grad_delta),
 tptr<scalar_t>(grad_A), tptr<scalar_t>(grad_B),
 batch, seq_len, d_state, stream);
 } else if (sg_arch == 942) {
 err = ::sg::gfx942::models::mamba::selective_scan_bwd<scalar_t>(
 ctptr<scalar_t>(grad_out),
 ctptr<scalar_t>(u), ctptr<scalar_t>(delta),
 ctptr<scalar_t>(A), ctptr<scalar_t>(B),
 ctptr<scalar_t>(state),
 tptr<scalar_t>(grad_u), tptr<scalar_t>(grad_delta),
 tptr<scalar_t>(grad_A), tptr<scalar_t>(grad_B),
 batch, seq_len, d_state, stream);
 } else {
 TORCH_CHECK(false,
 "mamba_selective_scan_backward: unsupported arch ", sg_arch);
 }
 check_cuda(err, "selective_scan_backward");
 };

 switch (u.scalar_type()) {
 case torch::kFloat32: run(float{}); break;
 case torch::kFloat16: run(__half{}); break;
 case torch::kBFloat16: run(__nv_bfloat16{}); break;
 default:
 TORCH_CHECK(false,
 "mamba_selective_scan_backward: unsupported dtype ",
 u.scalar_type());
 }
}

} // namespace sg


// ─── csrc/bindings/models_module.cpp ───────────────────────────────
// =====================================================================
// bindings/models_module.cpp — model bindings aggregator
//
// Registers all model entry points (decoder, vit, mamba) into the _ops
// pybind11 module under a "models" submodule, so they appear as
// _ops.models.<name> in Python.
//
// Each per-model file (csrc/bindings/models_<model>.cpp) defines public
// entry points in namespace sg::; this file binds them to pybind11.
// =====================================================================

#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ---------------------------------------------------------------------
// Forward declarations — Decoder
// ---------------------------------------------------------------------
namespace sg {

void decoder_forward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int, int);
void decoder_backward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int, int);
void decoder_attention_forward(
 torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 int, int, int, float);
void decoder_attention_backward(
 torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, float);

// ---------------------------------------------------------------------
// Forward declarations — ViT
// ---------------------------------------------------------------------

void vit_forward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int, int, int, int, int);
void vit_backward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int, int, int, int, int);
void vit_attention_forward(
 torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 int, int, int, float);
void vit_attention_backward(
 torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, float);
void vit_patch_project(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int);

// ---------------------------------------------------------------------
// Forward declarations — Mamba
// ---------------------------------------------------------------------

void mamba_forward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int);
void mamba_backward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int);
void mamba_layer_forward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int);
void mamba_selective_scan_forward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 int, int, int);
void mamba_selective_scan_backward(
 torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int);

} // namespace sg

// ---------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------

void register_model_bindings(py::module_& m) {
 auto models = m.def_submodule("models",
 "Per-arch optimized model kernels (decoder, vit, mamba)");

 // Decoder Transformer
 models.def("decoder_forward", &sg::decoder_forward,
 "Decoder Transformer forward pass");
 models.def("decoder_backward", &sg::decoder_backward,
 "Decoder Transformer backward pass");
 models.def("decoder_attention_forward", &sg::decoder_attention_forward,
 "Decoder multi-head attention forward (component test)");
 models.def("decoder_attention_backward", &sg::decoder_attention_backward,
 "Decoder multi-head attention backward (component test)");

 // Vision Transformer
 models.def("vit_forward", &sg::vit_forward,
 "Vision Transformer forward pass");
 models.def("vit_backward", &sg::vit_backward,
 "Vision Transformer backward pass");
 models.def("vit_attention_forward", &sg::vit_attention_forward,
 "ViT multi-head attention forward (component test)");
 models.def("vit_attention_backward", &sg::vit_attention_backward,
 "ViT multi-head attention backward (component test)");
 models.def("vit_patch_project", &sg::vit_patch_project,
 "ViT patch projection (component test)");

 // Mamba (SSM)
 models.def("mamba_forward", &sg::mamba_forward,
 "Mamba SSM forward pass");
 models.def("mamba_backward", &sg::mamba_backward,
 "Mamba SSM backward pass");
 models.def("mamba_layer_forward", &sg::mamba_layer_forward,
 "Mamba single-layer forward (component test)");
 models.def("mamba_selective_scan_forward", &sg::mamba_selective_scan_forward,
 "Mamba selective scan forward (component test)");
 models.def("mamba_selective_scan_backward", &sg::mamba_selective_scan_backward,
 "Mamba selective scan backward (component test)");
}


// ─── csrc/bindings/module.cpp (PYBIND11 entry) ─────────────────
// Forward declaration for model bindings aggregator (models_module.cpp)
void register_model_bindings(pybind11::module_& m);

PYBIND11_MODULE(_ops, m) {
 m.doc() = "Grokking Optimizers — specialized per-arch C++/CUDA/HIP kernels";

 m.def("detect_arch", &sg::detect_arch,
 "Returns 90 or 942 for the detected GPU (3-arch active set: "
 "sm_90, gfx942, tpu_v5p). TPU handled in Python.");

 // Fused (model, optimizer, arch) dispatch
 m.def("fused_step", &sg::fused_step,
 "Fused (model, optimizer, arch) kernel dispatch. Routes to the "
 "appropriate fused TU based on detected hardware.");

 // GrokAdamW
 m.def("grokadamw_step", &sg::grokadamw_step);
 m.def("grokadamw_clip_step", &sg::grokadamw_clip_step);
 m.def("grokadamw_step_q3", &sg::grokadamw_step_q3);
 m.def("grokadamw_fused_step", &sg::grokadamw_fused_step,
 "Pre-refactor ops.cpp::grokadamw_fused_step (vector signature)");
 m.def("fused_adamw_simple_step", &sg::fused_adamw_simple_step,
 "Shared simple AdamW step (vector signature)");

 // Lion
 m.def("lion_step", &sg::lion_step);
 m.def("lion_fused_step", &sg::lion_fused_step,
 "Pre-refactor ops.cpp::lion_fused_step (vector signature)");

 // Grokfast
 m.def("grokfast_ema", &sg::grokfast_ema);
 m.def("grokfast_adam", &sg::grokfast_adam);
 m.def("grokfast_fused_step", &sg::grokfast_fused_step,
 "EMA-only multi-tensor grokfast (vector signature)");
 m.def("grokfast_fused_ema_adam_step", &sg::grokfast_fused_ema_adam_step,
 "Per-param EMA + amplification + Adam (vector signature)");

 // Prodigy
 m.def("prodigy_step", &sg::prodigy_step);
 m.def("prodigy_dlr_reduce", &sg::prodigy_dlr_reduce);
 m.def("prodigy_fused_step", &sg::prodigy_fused_step,
 "Pre-refactor ops.cpp::prodigy_fused_step (returns updated d_lr)");

 // NeuralGrok
 m.def("neuralgrok_amplifier", &sg::neuralgrok_amplifier);
 m.def("neuralgrok_adam", &sg::neuralgrok_adam);
 m.def("neuralgrok_fused_step", &sg::neuralgrok_fused_step,
 "Pre-refactor ops.cpp::neuralgrok_fused_step (vector signature)");

 // LookSAM
 m.def("looksam_perturb", &sg::looksam_perturb);
 m.def("looksam_restore", &sg::looksam_restore);
 m.def("looksam_direction_adjust_fused", &sg::looksam_direction_adjust_fused);
 m.def("looksam_norm_reduce", &sg::looksam_norm_reduce);
 m.def("looksam_perturb_all", &sg::looksam_perturb_all,
 "Vector SAM perturb; returns clones to restore from");
 m.def("looksam_restore_all", &sg::looksam_restore_all);
 m.def("looksam_compute_directions_and_adjust",
 &sg::looksam_compute_directions_and_adjust,
 "Fused direction + adjust over a list of params");

 // Muon
 m.def("muon_momentum_normalize", &sg::muon_momentum_normalize);
 m.def("muon_ns_combine", &sg::muon_ns_combine);
 m.def("muon_ns_combine_update_fused", &sg::muon_ns_combine_update_fused);
 m.def("muon_fused_step", &sg::muon_fused_step,
 "Pre-refactor ops.cpp::muon_fused_step (vector signature)");

 // SG v1.5
 m.def("sg15_sam_perturb", &sg::sg15_sam_perturb);
 m.def("supergrok15_fused_step", &sg::supergrok15_fused_step);
 m.def("supergrok15_sam_perturb_all", &sg::supergrok15_sam_perturb_all);
 m.def("supergrok15_sharpness_restore_all", &sg::supergrok15_sharpness_restore_all);

 // SG v1.1
 m.def("supergrok11_fused_step", &sg::supergrok11_fused_step);
 m.def("supergrok11_sam_perturb_all", &sg::supergrok11_sam_perturb_all);
 m.def("supergrok11_sharpness_restore_all", &sg::supergrok11_sharpness_restore_all);

 // MoE
 m.def("moe_dynamic_expert_load", &sg::moe_dynamic_expert_load);
 m.def("moe_dynamic_expert_fwd", &sg::moe_dynamic_expert_fwd);
 m.def("moe_dynamic_expert_bwd", &sg::moe_dynamic_expert_bwd);
 m.def("moe_filter_active_params", &sg::moe_filter_active_params);
 m.def("moe_scan_compacted", &sg::moe_scan_compacted);
 m.def("moe_scatter_results", &sg::moe_scatter_results);
 m.def("moe_count_expert_activations", &sg::moe_count_expert_activations);
 m.def("moe_compute_load_balance_loss", &sg::moe_compute_load_balance_loss);
 m.def("moe_apply_frequency_scaling", &sg::moe_apply_frequency_scaling);

 // Distributed scan
 m.def("distributed_scan_phase1", &sg::distributed_scan_phase1);
 m.def("distributed_scan_phase2", &sg::distributed_scan_phase2);
 m.def("distributed_scan_phase3", &sg::distributed_scan_phase3);

 // Quantization
 m.def("fp8_e4m3_quantize", &sg::fp8_e4m3_quantize);
 m.def("int8_symmetric_quantize", &sg::int8_symmetric_quantize);
 m.def("int4_gptq_quantize", &sg::int4_gptq_quantize);

 // SuperGrok v2 — Mamba-3+PEER meta-net entries.
 // Wrappers in csrc/bindings/supergrok2.cpp dispatch to the per-arch
 // launchers in csrc/kernels/{cuda/<sm>,hip/<gfx>}/supergrok2_{fwd,bwd}_<arch>.
 m.def("supergrok2_mamba_peer_step", &sg::supergrok2_mamba_peer_step,
 "SuperGrok2: per-param Mamba-3+PEER meta-net + mu + Adam + WD");
 m.def("supergrok2_mamba_peer_batched_step", &sg::supergrok2_mamba_peer_batched_step,
 "SuperGrok2: batched Mamba-3+PEER step for all params at once");
 m.def("supergrok2_bilevel_fwd_save", &sg::supergrok2_bilevel_fwd_save,
 "SuperGrok2 bilevel: forward scan with state saving for backward");
 m.def("supergrok2_bilevel_fwd_save_batched", &sg::supergrok2_bilevel_fwd_save_batched,
 "SuperGrok2 bilevel: batched forward scan with state saving");
 m.def("supergrok2_bilevel_backward", &sg::supergrok2_bilevel_backward,
 "SuperGrok2 bilevel: full backward through meta-net");
 m.def("supergrok2_bilevel_backward_batched", &sg::supergrok2_bilevel_backward_batched,
 "SuperGrok2 bilevel: batched full backward through meta-net");
 m.def("supergrok2_prepare_and_batched_step", &sg::supergrok2_prepare_and_batched_step,
 "Fused multi-tensor grad prepare (clip, finite, bias corrections) "
 "+ batched mamba-3+PEER step. Replaces N per-parameter Python "
 "iterations with one kernel launch.");

 // Model bindings (decoder, vit, mamba) — registered as _ops.models.*
 register_model_bindings(m);
}
