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
 SG_CASE_SM90_RET(METHOD, __VA_ARGS__) \
 SG_CASE_GFX942_RET(METHOD, __VA_ARGS__) \
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
// High-level vector-signature entry point — PUBLISHED GrokAdamW.
//
// β1 is now PER-TENSOR (layer-wise momentum decay, β1_i = β1*(1-gamma)**i,
// computed Python-side and passed in `layer_beta1s`), mirroring
// supergrok11_fused_step's `layer_beta1s` plumbing. Because the multi-tensor
// launcher (launch_multi_tensor_grokadamw) takes a SINGLE scalar beta1 for the
// whole batch, a per-tensor β1 can no longer go through it; instead we LOOP and
// call the single-tensor launcher per param with its own β1_i (exactly as
// grokfast_fused_ema_adam_step loops over grokfast_adam). The bias-correction
// uses the SAME per-tensor β1_i the kernel uses for the moment update
// (bc1 = 1 - β1_i**step), matching SG11 (bindings.cpp ~L1374).
//
// `alpha` is the LIVE α_t (grokking-signal-modulated) the Python optimizer
// computed this step — a scalar, passed straight through (the slow-grad EMA +
// amplification math is unchanged: csrc/algorithms/grokadamw.h).
// ---------------------------------------------------------------------
void grokadamw_fused_step(
 std::vector<torch::Tensor>& params,
 std::vector<torch::Tensor>& grads,
 std::vector<torch::Tensor>& exp_avgs,
 std::vector<torch::Tensor>& exp_avg_sqs,
 std::vector<torch::Tensor>& emas,
 std::vector<int64_t>& steps,
 std::vector<double>& layer_beta1s,
 float alpha, float lamb_grok,
 float beta1, float beta2, float lr, float wd,
 float eps, float grad_clip_norm
) {
 const size_t n_params = params.size();
 check_params_grads(params, grads, "grokadamw_fused_step");
 check_list_len(exp_avgs, n_params, "exp_avgs", "grokadamw_fused_step");
 check_list_len(exp_avg_sqs, n_params, "exp_avg_sqs", "grokadamw_fused_step");
 check_list_len(emas, n_params, "emas", "grokadamw_fused_step");
 check_list_len(steps, n_params, "steps", "grokadamw_fused_step");
 check_list_len(layer_beta1s, n_params, "layer_beta1s", "grokadamw_fused_step");
 clip_grad_norms_device_side(grads, n_params, grad_clip_norm);
 if (n_params == 0) return;

 // NOTE: the Python optimizer owns the step counter (state["step"] += 1
 // before this call), matching fused_adamw_simple_step and
 // grokfast_fused_ema_adam_step. Do NOT increment here — `steps` is a
 // pybind-copied vector so an increment would not persist back to Python
 // anyway, and incrementing made bias correction permanently off-by-one
 // (bc computed with step+1).
 //
 // `beta1` (scalar) is retained in the signature for ABI compatibility with
 // the historical single-β1 call shape and is used ONLY as the fallback β1 for
 // any tensor whose layer_beta1s entry is non-finite/<=0 (defensive); the live
 // per-tensor value is layer_beta1s[i].
 for (size_t i = 0; i < n_params; i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 float b1 = static_cast<float>(layer_beta1s[i]);
 if (!(b1 > 0.0f) || b1 >= 1.0f) b1 = beta1;  // defensive fallback
 float bc1 = 1.0f - std::pow(b1, static_cast<float>(steps[i]));
 float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));
 // Loop through the single-tensor WRAPPER (exactly as grokfast_fused_
 // ema_adam_step loops over grokfast_adam) — NOT the DISPATCH_GROKADAMW
 // macro inline: that macro's arch cases expand to `case 90: return ...`,
 // and an inline `return` exits THIS function after tensor 0, silently
 // skipping every later parameter. Caught by the parity gate's layer-wise
 // β1 section the first time the vector ABI went live: L0 exact at 1e-7,
 // L1's moments untouched (solved per-element b1 == 1.0, spread 0).
 grokadamw_step(params[i], exp_avgs[i], exp_avg_sqs[i], emas[i], grads[i],
 alpha, lamb_grok, b1, beta2, lr, wd, eps, bc1, bc2);
 }
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
 check_params_grads(params, grads, "fused_adamw_simple_step");
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
 TORCH_CHECK(grads.size() == ema_bufs.size(), "grokfast_fused_step",
             ": grads.size() (", grads.size(), ") != ema_bufs.size() (",
             ema_bufs.size(), ")");
 std::vector<torch::Tensor> vg, ve;
 for (size_t i = 0; i < grads.size(); i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 // EMA-only entrypoint: the launcher indexes ema_bufs[i] and grads[i] as
 // flat data_ptr buffers, so they must match device/dtype/shape/contig.
 // check_param_grad validates the EMA buffer as the "param" and the grad
 // against it — the same boundary contract every other *_fused_step uses.
 check_param_grad(ema_bufs[i], grads[i], "grokfast_fused_step");
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
 check_params_grads(params, grads, "grokfast_fused_ema_adam_step");
 {
 const size_t n = params.size();
 check_list_len(emas, n, "emas", "grokfast_fused_ema_adam_step");
 check_list_len(exp_avgs, n, "exp_avgs", "grokfast_fused_ema_adam_step");
 check_list_len(exp_avg_sqs, n, "exp_avg_sqs", "grokfast_fused_ema_adam_step");
 check_list_len(steps, n, "steps", "grokfast_fused_ema_adam_step");
 }
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
 check_params_grads(params, grads, "lion_fused_step");
 check_list_len(exp_avgs, params.size(), "exp_avgs", "lion_fused_step");
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
 // Params without a grad are never perturbed below, so cloning them is
 // wasted work. Skip the clone and store the param itself as the backup:
 // since the param is left untouched, the later restore (copy backup→param)
 // is a value-preserving self-copy no-op — behavior is identical, minus the
 // clone/alloc. Backups stays index-aligned with params for restore.
 if (!grads[i].defined() || grads[i].numel() == 0) {
 backups.push_back(params[i]);
 continue;
 }
 backups.push_back(params[i].clone());
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
 // Broadcast the scalar bias-corrections into the per-tensor vectors the real
 // launcher takes (DECLARE_MT_GROKADAMW signature, defined in
 // kernels/<arch>/grokadamw_*): every param steps together here, so bc is
 // uniform. The scalar-bc DECLARE_MT overload above is defined by no backend.
 std::vector<float> bc1s(params.size(), bc1);
 std::vector<float> bc2s(params.size(), bc2);
 SG_DISPATCH(launch_multi_tensor_grokadamw,
 params, exp_avgs, exp_avg_sqs, emas, grads,
 bc1s, bc2s, alpha, lamb, beta1, beta2, lr, weight_decay, eps);
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
 check_params_grads(params, grads, "muon_fused_step");
 check_list_len(bufs, params.size(), "bufs", "muon_fused_step");
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

 // Muon's Newton-Schulz orthogonalization (X.t(), mm) and the
 // 0.2*sqrt(max(rows,cols)) RMS scale are defined only for a 2D
 // matrix param. A 1D bias or a 3D/4D conv weight would silently read
 // p.size(1) as a wrong dimension (or be undefined for dim<2) and
 // corrupt the update. The Python Muon optimizer routes non-2D params
 // to the AdamW path (fused_adamw_simple_step); enforce that contract
 // loudly here so a misrouted param fails instead of corrupting.
 TORCH_CHECK(p.dim() == 2, "muon_fused_step: param ", i,
 " has ndim=", p.dim(),
 " but the Newton-Schulz orthogonalization requires a 2D "
 "matrix. Route 1D/>=3D params to the AdamW path "
 "(fused_adamw_simple_step) in the Python wrapper.");
 int64_t rows = p.size(0);
 int64_t cols = p.size(1);
 float max_dim = static_cast<float>(std::max(rows, cols));
 float scale_factor = 0.2f * std::sqrt(max_dim);
 float neg_lr_scale = -lr * scale_factor;
 float decay_factor = 1.0f - lr * wd;

 // Double-buffer the NS combine output: allocate one scratch of X's shape
 // up front and ping-pong into it across iterations, instead of a fresh
 // torch::empty_like(X) every intermediate step. The combine launcher reads
 // X and writes a distinct buffer, so two buffers suffice; we swap handles
 // (no copies). Math is unchanged.
 torch::Tensor scratch; // lazily allocated on first combine step
 for (int step = 0; step < ns_steps; step++) {
 auto A = torch::mm(X, X.t());
 auto AX = torch::mm(A, X);
 auto AAX = torch::mm(A, AX);
 if (step < ns_steps - 1) {
 if (!scratch.defined()) scratch = torch::empty_like(X);
 SG_DISPATCH_CALL(launch_muon_ns_combine,
 scratch, X, AX, AAX, NS_A, NS_B, NS_C);
 // Ping-pong: the just-read X becomes the next iteration's scratch,
 // the freshly-written buffer becomes X.
 std::swap(X, scratch);
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
 check_params_grads(params, grads, "neuralgrok_fused_step");
 check_list_len(exp_avgs, n_params, "exp_avgs", "neuralgrok_fused_step");
 check_list_len(exp_avg_sqs, n_params, "exp_avg_sqs", "neuralgrok_fused_step");
 check_list_len(steps, n_params, "steps", "neuralgrok_fused_step");
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
// r_num_buf / s_den_buf are PERSISTENT device scalars holding the Prodigy
// D-estimate's EMA numerator/denominator (decayed by beta3 each step, NOT
// zeroed); d_coef scales the candidate estimate. (See the launcher: this is
// the canonical persistent-EMA Prodigy estimator, replacing the instantaneous
// per-step r/s that ratcheted d up without bound.)
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
 torch::Tensor r_num_buf, torch::Tensor s_den_buf, \
 float beta1, float beta2, float beta3, \
 float lr, float wd, float eps, float d_coef); \
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
// Canonical persistent-EMA Prodigy D-estimate (Mishchenko & Defazio 2023 /
// prodigyopt): the numerator r_ema and denominator s_ema are running EMAs
// decayed by beta3 = sqrt(beta2) across steps (NOT recomputed instantaneously);
// d = max(d_prev, d_coef·r_ema/|s_ema|). This makes d PLATEAU once the params
// stop drifting from init, instead of the old instantaneous form whose `max()`
// locked in post-memorization gradient-noise spikes → d blew up → collapse.
// Returns (d_lr, r_ema, s_ema) — all three persist across steps in the Python
// optimizer (like _d_lr). Single CPU sync.
std::tuple<float, float, float> prodigy_fused_step(
 std::vector<torch::Tensor>& params,
 std::vector<torch::Tensor>& grads,
 std::vector<torch::Tensor>& exp_avgs,
 std::vector<torch::Tensor>& exp_avg_sqs,
 std::vector<torch::Tensor>& s_bufs,
 std::vector<torch::Tensor>& param_inits,
 std::vector<int64_t>& steps,
 float d_lr,
 float r_ema, float s_ema,
 float beta1, float beta2, float beta3, float lr, float wd,
 float eps, float d0, float d_coef
) {
 if (params.empty()) return std::make_tuple(d_lr, r_ema, s_ema);
 check_params_grads(params, grads, "prodigy_fused_step");
 {
 const size_t n = params.size();
 check_list_len(exp_avgs, n, "exp_avgs", "prodigy_fused_step");
 check_list_len(exp_avg_sqs, n, "exp_avg_sqs", "prodigy_fused_step");
 check_list_len(s_bufs, n, "s_bufs", "prodigy_fused_step");
 check_list_len(param_inits, n, "param_inits", "prodigy_fused_step");
 check_list_len(steps, n, "steps", "prodigy_fused_step");
 }

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
 if (vp.empty()) return std::make_tuple(d_lr, r_ema, s_ema);

 auto d_lr_buf = torch::tensor(
 {d_lr}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
 // Persistent EMA numerator/denominator carried in/out as device scalars,
 // seeded from the Python-held floats. The launcher decays each by beta3 and
 // adds this step's reduction (d0 is a global constant that cancels in the
 // ratio, so it is NOT threaded into the device math — the bare d²·<g,p0−p>
 // and d²·‖g‖₁ partials already match prodigyopt up to that constant factor).
 auto r_num_buf = torch::tensor(
 {r_ema}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
 auto s_den_buf = torch::tensor(
 {s_ema}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
 (void)d0;

 SG_DISPATCH_CALL(launch_multi_tensor_prodigy_fused_reduce_step,
 vp, vg, vpi, vea, veasq, vs, bc1_vec, bc2_vec, d_lr_buf,
 r_num_buf, s_den_buf,
 beta1, beta2, beta3, lr, wd, eps, d_coef);
 return std::make_tuple(
 d_lr_buf.item<float>(), r_num_buf.item<float>(), s_den_buf.item<float>());
}

} // namespace sg


// ─── csrc/bindings/supergrok11.cpp ─────────────────────────────────
// bindings/supergrok11.cpp — runtime dispatch to per-arch SG v1.1 launchers.
//
// SG v1.1 mirrors v1.5 except for the cosine-similarity gate. The flow:
// 1. mu_metanet kernel produces smart_grad (and updates mu)
// 2. compute_cosine_gate_fused does a 3-quantity reduction
// (dot, |sg|², |mu|²), syncs once to CPU, returns sigmoid(t·cos_sim)
// 3. adam_decay applies the Adam step using the single live correction
// lamb_eff = ramp·lamb·(1-gate)·alpha (lamb/ramp are live multipliers on the
// canonical (1-gate)·alpha term; lamb=ramp=1 ⇒ the plain (1-gate)·alpha). The
// earlier note here said "lamb_eff = ramp·gate·lamb", which described a SECOND
// mu add that never matched this code and has been removed — see the detailed
// block in supergrok11_fused_step below.


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
 SG_CASE_SM90_RET(compute_cosine_gate_fused, smart_grad, mu, gate_temp)
 SG_CASE_GFX942_RET(compute_cosine_gate_fused, smart_grad, mu, gate_temp)
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
 check_params_grads(params, grads, "supergrok11_fused_step");
 check_list_len(exp_avgs, n_params, "exp_avgs", "supergrok11_fused_step");
 check_list_len(exp_avg_sqs, n_params, "exp_avg_sqs", "supergrok11_fused_step");
 check_list_len(mus, n_params, "mus", "supergrok11_fused_step");
 check_list_len(sharpness_cache, n_params, "sharpness_cache", "supergrok11_fused_step");
 check_list_len(steps, n_params, "steps", "supergrok11_fused_step");
 check_list_len(layer_alphas, n_params, "layer_alphas", "supergrok11_fused_step");
 check_list_len(layer_beta1s, n_params, "layer_beta1s", "supergrok11_fused_step");
 clip_grad_norms_device_side(grads, n_params, grad_clip_norm);

 for (size_t i = 0; i < n_params; i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 // steps[i] is pre-incremented by the Python optimizer (SuperGrok11.step):
 // pybind passes `steps` by value, so incrementing it here would NOT persist
 // back to Python and would freeze bias-correction at t=1. Read it as-is.
 float alpha = layer_alphas[i];
 float beta1 = layer_beta1s[i];
 float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
 float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));

 // Canonical SuperGrok v1.1 mixing (csrc/algorithms/supergrok11.h:101,
 // Pallas ref launch_supergrok11.py, oracle ref_sg11_step):
 //   gate       = sigmoid(gate_temp * cosine(grad, mu))  (sg11_finalize_gate)
 //   smart_grad = grad + lamb_eff * mu                  (applied ONCE)
 //   AdamW on smart_grad.
 // The base correction (1-gate)*alpha SHRINKS as grad/mu alignment (gate)
 // grows — the polarity that lets a memorized solution HOLD. We reproduce this
 // through the existing kernels by exact algebra, WITHOUT any kernel signature
 // change:
 //   * mu_metanet with alpha=0 writes mu = rescale*phi (mu is independent of
 //     alpha, kernel line ~285) AND its smart_grad output = grad + 0*mu = grad.
 //   * dispatch_cosine_gate(smart_grad==grad, mu) is therefore cosine(grad, mu).
 //   * adam_decay computes g_eff = smart_grad + lamb_eff*mu; with
 //     smart_grad==grad and the lamb_eff below this is the single live add,
 //     then the canonical Adam tail.
 //
 // OWNER DECISION (Option B): `lamb`/`ramp` are made LIVE as MULTIPLIERS on the
 // single canonical term. The pre-fix code had a SECOND mu add
 // `ramp*gate*lamb*mu` (the OLD comment: "they were the buggy SECOND mu add …
 // whose coefficient GREW with the gate and destroyed the memorized solution");
 // that second add was REMOVED as a collapse mitigation, which suppressed the
 // lamb/ramp mechanism entirely (they were (void)-ed). It is RESTORED here
 // MULTIPLICATIVELY on the ONE live, correctly-shrinking term:
 //   lamb_eff = ramp * lamb * (1 - gate) * alpha
 // so lamb=1.0 with post-warmup ramp=1.0 reproduces the previous behavior
 // EXACTLY ((1-gate)*alpha), while lamb is a genuine tuner dial that scales the
 // correction WITHOUT flipping its polarity (it still shrinks as the gate
 // grows). ramp is the warmup schedule (0 during warmup → no correction early).
 const float lamb_eff_scale = ramp * lamb;
 auto smart_grad = torch::empty_like(params[i]);
 SG_DISPATCH_CALL(launch_sg11_mu_metanet,
 mus[i], grads[i], sharpness_cache[i], smart_grad, /*alpha=*/0.0f,
 W1, b1, W2, b2, rescale, hidden_dim);

 float gate = dispatch_cosine_gate(smart_grad, mus[i], gate_temperature);
 float lamb_eff = lamb_eff_scale * (1.0f - gate) * alpha;

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
 // Grad-less params are not perturbed; skip the clone and alias the param
 // as its own backup. The restore for these entries (params[i].copy_(
 // backups[i])) is then a self-copy no-op — identical behavior, no clone.
 if (!grads[i].defined() || grads[i].numel() == 0) {
 backups.push_back(params[i]);
 continue;
 }
 backups.push_back(params[i].clone());
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
 check_params_grads(params, grads, "supergrok15_fused_step");
 check_list_len(exp_avgs, n_params, "exp_avgs", "supergrok15_fused_step");
 check_list_len(exp_avg_sqs, n_params, "exp_avg_sqs", "supergrok15_fused_step");
 check_list_len(mus, n_params, "mus", "supergrok15_fused_step");
 check_list_len(sharpness_cache, n_params, "sharpness_cache", "supergrok15_fused_step");
 check_list_len(steps, n_params, "steps", "supergrok15_fused_step");
 check_list_len(layer_alphas, n_params, "layer_alphas", "supergrok15_fused_step");
 check_list_len(layer_beta1s, n_params, "layer_beta1s", "supergrok15_fused_step");
 clip_grad_norms_device_side(grads, n_params, grad_clip_norm);

 float lamb_eff = (ramp > 0.0f) ? (ramp * gate_signal * lamb) : 0.0f;

 for (size_t i = 0; i < n_params; i++) {
 if (!grads[i].defined() || grads[i].numel() == 0) continue;
 // steps[i] is pre-incremented by the Python optimizer (SuperGrok15.step):
 // pybind passes `steps` by value, so incrementing here would not persist and
 // would freeze bias-correction at t=1. Read it as-is.
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
 // Grad-less params are not perturbed; skip the clone and alias the param
 // as its own backup. The restore for these entries (params[i].copy_(
 // backups[i])) is then a self-copy no-op — identical behavior, no clone.
 if (!grads[i].defined() || grads[i].numel() == 0) {
 backups.push_back(params[i]);
 continue;
 }
 backups.push_back(params[i].clone());
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
// SG v2 is a learned optimizer whose meta-model is a DeepSeek-V4-style
// CSA/HCA hybrid attention stack (replacing the old Mamba-3 + PEER scan).
// For each parameter tensor it views the flattened gradient elements as a
// sequence, runs:
// input_proj -> CSA (compressed sparse attention, m=4, lightning-indexer
// top-k + sliding window) -> HCA (heavily compressed
// attention, m'=128, dense) -> MiniGRU -> PEER product-key
// routing -> per-element expert MLP -> out_proj
// producing a smart gradient, then applies the unchanged Adam-style update.
//
// The CSA/HCA attention is STATELESS across optimizer steps (unlike Mamba's
// carried fwd/bwd scan state), so the launchers DROP mamba_fwd_state /
// mamba_bwd_state. The GRU state IS still carried across steps (gru_state).
//
// Per-arch launcher names (sg::sm90 / sg::gfx942). Signatures must match
// exactly across arches and with the CUDA / HIP launcher agents — the
// single-tensor step list is the locked spec §7 contract:
//
// sg::<arch>::launch_csa_hca_step (forward step)
// sg::<arch>::launch_csa_hca_batched_step (batched fwd step)
// sg::<arch>::launch_csa_hca_bilevel_fwd_save (bilevel fwd-save)
// sg::<arch>::launch_csa_hca_bilevel_fwd_save_batched(batched fwd-save)
// sg::<arch>::launch_csa_hca_backward (bilevel bwd)
// sg::<arch>::launch_csa_hca_backward_batched (batched bwd)
//
// Public Python entry points (registered via _ops.<name> in module.cpp).
// New names plus old "mamba_peer" aliases kept for back-compat:
// supergrok2_step (alias: supergrok2_mamba_peer_step)
// supergrok2_batched_step (alias: supergrok2_mamba_peer_batched_step)
// supergrok2_bilevel_fwd_save
// supergrok2_bilevel_fwd_save_batched
// supergrok2_bilevel_backward
// supergrok2_bilevel_backward_batched
// supergrok2_prepare_and_batched_step
//
// The CSA/HCA meta-model weight set (per spec §4/§7) replaces the Mamba
// weight block: csa_{q,k,v,out}_W, csa_compress_w, csa_idx_{DQ,UQ,K},
// hca_{q,k,v,out}_W. The GRU, PEER (peer_query_Ws, prod_keys_A/B),
// expert MLP, and Adam-apply tail are unchanged.


#include <vector>

namespace sg {

// ---------------------------------------------------------------------
// Per-arch launcher forward declarations.
// The launch_csa_hca_step parameter list is the AUTHORITATIVE spec §7
// contract, shared byte-for-byte with the CUDA / HIP launcher agents.
// ---------------------------------------------------------------------

#define DECLARE_SG2(NS) \
 namespace NS { \
 void launch_csa_hca_step( \
 torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness, \
 torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,\
 torch::Tensor slow, \
 torch::Tensor gru_state, \
 /* --- shared input projection --- */ \
 torch::Tensor input_proj_W, torch::Tensor input_proj_b, \
 /* --- CSA layer (produces csa_ctx) --- */ \
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, \
 torch::Tensor csa_compress_w, \
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, \
 torch::Tensor csa_idx_K, \
 torch::Tensor csa_out_W, \
 /* --- HCA layer (produces hca_ctx) --- */ \
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, \
 torch::Tensor hca_out_W, \
 /* --- GRU (carried across steps) --- */ \
 torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr, \
 torch::Tensor gru_br, torch::Tensor gru_Wh, torch::Tensor gru_bh, \
 /* --- PEER routing + experts --- */ \
 torch::Tensor peer_query_Ws, \
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B, \
 torch::Tensor expert_W1, torch::Tensor expert_b1, \
 torch::Tensor expert_W2, torch::Tensor expert_b2, \
 /* --- scalars --- */ \
 float rescale, float alpha_mu, float lamb_eff, \
 float beta1, float beta2, float lr, float wd_eff, float eps, \
 float bc1, float bc2, \
 int d_model, int gru_hidden, int num_heads, int pk_dim, \
 int expert_hidden, int num_experts, \
 int csa_compress, int csa_window, int csa_topk, \
 int hca_compress, int indexer_rank, \
 torch::Tensor expert_counts, \
 int peer_topk); \
 void launch_csa_hca_batched_step( \
 std::vector<torch::Tensor> params, \
 std::vector<torch::Tensor> grads, \
 std::vector<torch::Tensor> sharpness_list, \
 std::vector<torch::Tensor> exp_avgs, \
 std::vector<torch::Tensor> exp_avg_sqs, \
 std::vector<torch::Tensor> mus, \
 std::vector<torch::Tensor> slows, \
 std::vector<torch::Tensor> gru_states, \
 /* --- shared input projection --- */ \
 torch::Tensor input_proj_W, torch::Tensor input_proj_b, \
 /* --- CSA layer --- */ \
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, \
 torch::Tensor csa_compress_w, \
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, \
 torch::Tensor csa_idx_K, \
 torch::Tensor csa_out_W, \
 /* --- HCA layer --- */ \
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, \
 torch::Tensor hca_out_W, \
 /* --- GRU --- */ \
 torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr, \
 torch::Tensor gru_br, torch::Tensor gru_Wh, torch::Tensor gru_bh, \
 /* --- PEER routing + experts --- */ \
 torch::Tensor peer_query_Ws, \
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B, \
 torch::Tensor expert_W1, torch::Tensor expert_b1, \
 torch::Tensor expert_W2, torch::Tensor expert_b2, \
 /* --- per-tensor scalars --- */ \
 std::vector<float> alpha_mus, std::vector<float> lamb_effs, \
 std::vector<float> beta1s, \
 std::vector<float> bc1s, std::vector<float> bc2s, \
 /* --- shared scalars --- */ \
 float rescale, float beta2, float lr, float wd_eff, float eps, \
 int d_model, int gru_hidden, int num_heads, int pk_dim, \
 int expert_hidden, int num_experts, \
 int csa_compress, int csa_window, int csa_topk, \
 int hca_compress, int indexer_rank, \
 torch::Tensor expert_counts, \
 int peer_topk); \
 void launch_csa_hca_bilevel_fwd_save( \
 torch::Tensor grad, torch::Tensor sharpness, \
 torch::Tensor input_proj_W, torch::Tensor input_proj_b, \
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, \
 torch::Tensor csa_compress_w, \
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, \
 torch::Tensor csa_idx_K, \
 torch::Tensor csa_out_W, \
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, \
 torch::Tensor hca_out_W, \
 int d_model, int num_heads, \
 int csa_compress, int csa_window, int csa_topk, \
 int hca_compress, int indexer_rank, \
 torch::Tensor csa_ctx_out, torch::Tensor hca_ctx_out, \
 torch::Tensor x_sorted, torch::Tensor sort_indices, \
 /* attention adjoint saved state */ \
 torch::Tensor csa_saved_denom, \
 torch::Tensor csa_saved_sel_idx, \
 torch::Tensor csa_saved_probs, \
 torch::Tensor hca_saved_denom, \
 torch::Tensor hca_saved_probs, \
 int checkpoint_interval); \
 void launch_csa_hca_bilevel_fwd_save_batched( \
 std::vector<torch::Tensor> grads, \
 std::vector<torch::Tensor> sharpness_list, \
 torch::Tensor input_proj_W, torch::Tensor input_proj_b, \
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, \
 torch::Tensor csa_compress_w, \
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, \
 torch::Tensor csa_idx_K, \
 torch::Tensor csa_out_W, \
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, \
 torch::Tensor hca_out_W, \
 int d_model, int num_heads, \
 int csa_compress, int csa_window, int csa_topk, \
 int hca_compress, int indexer_rank, \
 torch::Tensor csa_ctx_out_packed, \
 torch::Tensor hca_ctx_out_packed, \
 torch::Tensor x_sorted_packed, torch::Tensor offsets_t, \
 torch::Tensor sort_indices_packed, \
 torch::Tensor csa_saved_denom_packed, \
 torch::Tensor csa_saved_sel_idx_packed, \
 torch::Tensor csa_saved_probs_packed, \
 torch::Tensor hca_saved_denom_packed, \
 torch::Tensor hca_saved_probs_packed, \
 int checkpoint_interval); \
 void launch_csa_hca_backward( \
 torch::Tensor d_smart_grad, \
 torch::Tensor grad, torch::Tensor sharpness, float rescale, \
 torch::Tensor sort_indices, torch::Tensor x_sorted, \
 torch::Tensor csa_ctx, torch::Tensor hca_ctx, \
 torch::Tensor csa_saved_denom, \
 torch::Tensor csa_saved_sel_idx, \
 torch::Tensor csa_saved_probs, \
 torch::Tensor hca_saved_denom, \
 torch::Tensor hca_saved_probs, \
 torch::Tensor gru_input, torch::Tensor gru_h_old, \
 torch::Tensor gru_z_gate, torch::Tensor gru_r_gate, \
 torch::Tensor gru_h_tilde, \
 torch::Tensor peer_input, torch::Tensor expert_indices, \
 torch::Tensor routing_weights, torch::Tensor saved_z_hidden, \
 torch::Tensor saved_scores_a, torch::Tensor saved_scores_b, \
 torch::Tensor saved_top_a_idx, torch::Tensor saved_top_b_idx, \
 torch::Tensor saved_soft_a, torch::Tensor saved_soft_b, \
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, \
 torch::Tensor csa_compress_w, \
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, \
 torch::Tensor csa_idx_K, \
 torch::Tensor csa_out_W, \
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, \
 torch::Tensor hca_out_W, \
 torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh, \
 torch::Tensor peer_query_Ws, \
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B, \
 torch::Tensor expert_W1, torch::Tensor expert_W2, \
 torch::Tensor expert_b1_in, torch::Tensor expert_b2_in, \
 torch::Tensor input_proj_W, \
 torch::Tensor d_csa_q_W, torch::Tensor d_csa_k_W, \
 torch::Tensor d_csa_v_W, torch::Tensor d_csa_compress_w, \
 torch::Tensor d_csa_idx_DQ, torch::Tensor d_csa_idx_UQ, \
 torch::Tensor d_csa_idx_K, torch::Tensor d_csa_out_W, \
 torch::Tensor d_hca_q_W, torch::Tensor d_hca_k_W, \
 torch::Tensor d_hca_v_W, torch::Tensor d_hca_out_W, \
 torch::Tensor d_gru_Wz, torch::Tensor d_gru_bz, \
 torch::Tensor d_gru_Wr, torch::Tensor d_gru_br, \
 torch::Tensor d_gru_Wh, torch::Tensor d_gru_bh, \
 torch::Tensor d_peer_query_Ws, \
 torch::Tensor d_prod_keys_A, torch::Tensor d_prod_keys_B, \
 torch::Tensor d_expert_W1, torch::Tensor d_expert_b1, \
 torch::Tensor d_expert_W2, torch::Tensor d_expert_b2, \
 torch::Tensor d_input_proj_W, torch::Tensor d_input_proj_b, \
 int d_model, int gru_hidden, int gru_input_dim, \
 int num_heads, int topk, int pk_dim, \
 int expert_hidden, int peer_input_dim, int num_experts, \
 int csa_compress, int csa_window, int csa_topk, \
 int hca_compress, int indexer_rank, \
 int checkpoint_interval); \
 void launch_csa_hca_backward_batched( \
 torch::Tensor d_csa_ctx_packed, \
 torch::Tensor d_hca_ctx_packed, \
 torch::Tensor x_sorted_packed, \
 torch::Tensor csa_saved_denom_packed, \
 torch::Tensor csa_saved_sel_idx_packed, \
 torch::Tensor csa_saved_probs_packed, \
 torch::Tensor hca_saved_denom_packed, \
 torch::Tensor hca_saved_probs_packed, \
 torch::Tensor offsets_t, \
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, \
 torch::Tensor csa_compress_w, \
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, \
 torch::Tensor csa_idx_K, \
 torch::Tensor csa_out_W, \
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, \
 torch::Tensor hca_out_W, \
 torch::Tensor d_csa_q_W, torch::Tensor d_csa_k_W, \
 torch::Tensor d_csa_v_W, torch::Tensor d_csa_compress_w, \
 torch::Tensor d_csa_idx_DQ, torch::Tensor d_csa_idx_UQ, \
 torch::Tensor d_csa_idx_K, torch::Tensor d_csa_out_W, \
 torch::Tensor d_hca_q_W, torch::Tensor d_hca_k_W, \
 torch::Tensor d_hca_v_W, torch::Tensor d_hca_out_W, \
 torch::Tensor d_x_sorted_packed, \
 int d_model, int num_heads, int num_params, \
 int csa_compress, int csa_window, int csa_topk, \
 int hca_compress, int indexer_rank, \
 int checkpoint_interval); \
 }

 DECLARE_SG2(sm90)

DECLARE_SG2(gfx942)
#undef DECLARE_SG2

// ---------------------------------------------------------------------
// SuperGrok2 FULL meta-net as ONE persistent megakernel (INTEGRATION-NOTES.md
// §1). The §1b host wrapper body + the ws-stride accessor are DEFINED in the
// .cu TU csrc/fused/sm_90/sg2_meta_tail.cu (nvcc-compiled, so the kernel
// `<<<...>>>` launch + the device-templated launcher are legal). bindings.cpp
// (host-only .cpp) cannot #include opt_stage_supergrok2.cuh — host gcc cannot
// parse the launch syntax — so it only extern-declares + PYBIND-registers them.
// This is a NEW op (sm_90 persistent path); it does NOT alter the existing
// supergrok2_batched_step path, which stays the parity oracle.
// ---------------------------------------------------------------------
void sg2_meta_optimizer_tail(
    torch::Tensor params_packed, torch::Tensor grads_packed, torch::Tensor sharpness_packed,
    torch::Tensor exp_avg_packed, torch::Tensor exp_avg_sq_packed,
    torch::Tensor mu_packed, torch::Tensor slow_packed,
    torch::Tensor gru_state_packed, torch::Tensor perm_packed, torch::Tensor unsort_packed,
    torch::Tensor n_per_tensor, torch::Tensor row_off,
    torch::Tensor workspace, int64_t ws_stride,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, torch::Tensor csa_out_W,
    torch::Tensor csa_compress_w, torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_K,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, torch::Tensor hca_out_W,
    torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1, torch::Tensor expert_W2, torch::Tensor expert_b2,
    torch::Tensor alpha, torch::Tensor gru_decay, torch::Tensor lamb_eff,
    torch::Tensor beta1, torch::Tensor bc1, torch::Tensor bc2,
    double rescale, double beta2, double lr, double wd, double eps,
    torch::Tensor g_next_task, torch::Tensor g_arrived, torch::Tensor g_generation);

int64_t sg2_ws_stride(int64_t Nmax);

// ---------------------------------------------------------------------
// Public entry points: thin SG_DISPATCH wrappers around the launchers.
// Each one matches the pre-refactor csrc/common/ops.cpp signature so
// existing Python callers (grokking_optimizers/supergrok2.py) keep
// working unchanged. Reference: ops.cpp@682eab4^ lines 908-1199 for the
// forward/peer-step entries, lines 1499-1611 for the bilevel entries.
// ---------------------------------------------------------------------

// SG2 single-tensor CSA/HCA step. (Renamed from supergrok2_mamba_peer_step;
// old name kept as a pybind alias.) Parameter list is spec §7.
void supergrok2_step(
 torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
 torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
 torch::Tensor slow,
 torch::Tensor gru_state,
 torch::Tensor input_proj_W, torch::Tensor input_proj_b,
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
 torch::Tensor csa_compress_w,
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ,
 torch::Tensor csa_idx_K,
 torch::Tensor csa_out_W,
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
 torch::Tensor hca_out_W,
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
 int d_model, int gru_hidden, int num_heads, int pk_dim,
 int expert_hidden, int num_experts,
 int csa_compress, int csa_window, int csa_topk,
 int hca_compress, int indexer_rank,
 torch::Tensor expert_counts,
 int peer_topk = 4
) {
 if (grad.numel() == 0) return;
 check_param_grad(param, grad, "supergrok2_step");
 SG_DISPATCH(launch_csa_hca_step,
 param, grad, sharpness, exp_avg, exp_avg_sq, mu,
 slow,
 gru_state,
 input_proj_W, input_proj_b,
 csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
 csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
 hca_q_W, hca_k_W, hca_v_W, hca_out_W,
 gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
 peer_query_Ws, prod_keys_A, prod_keys_B,
 expert_W1, expert_b1, expert_W2, expert_b2,
 rescale, alpha_mu, lamb_eff,
 beta1, beta2, lr, wd_eff, eps, bc1, bc2,
 d_model, gru_hidden, num_heads, pk_dim,
 expert_hidden, num_experts,
 csa_compress, csa_window, csa_topk,
 hca_compress, indexer_rank, expert_counts, peer_topk);
}

// SG2 batched CSA/HCA step. (Renamed from supergrok2_mamba_peer_batched_step;
// old name kept as a pybind alias.) Vectors per tensor; shared meta weights
// passed once; mamba states dropped (attention is stateless across steps).
void supergrok2_batched_step(
 std::vector<torch::Tensor> params,
 std::vector<torch::Tensor> grads,
 std::vector<torch::Tensor> sharpness_list,
 std::vector<torch::Tensor> exp_avgs,
 std::vector<torch::Tensor> exp_avg_sqs,
 std::vector<torch::Tensor> mus,
 std::vector<torch::Tensor> slows,
 std::vector<torch::Tensor> gru_states,
 torch::Tensor input_proj_W, torch::Tensor input_proj_b,
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
 torch::Tensor csa_compress_w,
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ,
 torch::Tensor csa_idx_K,
 torch::Tensor csa_out_W,
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
 torch::Tensor hca_out_W,
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
 int d_model, int gru_hidden, int num_heads, int pk_dim,
 int expert_hidden, int num_experts,
 int csa_compress, int csa_window, int csa_topk,
 int hca_compress, int indexer_rank,
 torch::Tensor expert_counts,
 int peer_topk = 4
) {
 if (params.empty()) return;
 check_params_grads(params, grads, "supergrok2_batched_step");
 {
 const size_t n = params.size();
 check_list_len(sharpness_list, n, "sharpness_list", "supergrok2_batched_step");
 check_list_len(exp_avgs, n, "exp_avgs", "supergrok2_batched_step");
 check_list_len(exp_avg_sqs, n, "exp_avg_sqs", "supergrok2_batched_step");
 check_list_len(mus, n, "mus", "supergrok2_batched_step");
 check_list_len(slows, n, "slows", "supergrok2_batched_step");
 check_list_len(gru_states, n, "gru_states", "supergrok2_batched_step");
 check_list_len(alpha_mus, n, "alpha_mus", "supergrok2_batched_step");
 check_list_len(lamb_effs, n, "lamb_effs", "supergrok2_batched_step");
 check_list_len(beta1s, n, "beta1s", "supergrok2_batched_step");
 check_list_len(bc1s, n, "bc1s", "supergrok2_batched_step");
 check_list_len(bc2s, n, "bc2s", "supergrok2_batched_step");
 }
 SG_DISPATCH(launch_csa_hca_batched_step,
 params, grads, sharpness_list, exp_avgs, exp_avg_sqs, mus,
 slows,
 gru_states,
 input_proj_W, input_proj_b,
 csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
 csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
 hca_q_W, hca_k_W, hca_v_W, hca_out_W,
 gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
 peer_query_Ws, prod_keys_A, prod_keys_B,
 expert_W1, expert_b1, expert_W2, expert_b2,
 alpha_mus, lamb_effs, beta1s, bc1s, bc2s,
 rescale, beta2, lr, wd_eff, eps,
 d_model, gru_hidden, num_heads, pk_dim,
 expert_hidden, num_experts,
 csa_compress, csa_window, csa_topk,
 hca_compress, indexer_rank, expert_counts, peer_topk);
}

// SG2 bilevel forward-save: runs the CSA/HCA attention forward and saves the
// adjoint state (softmax denominators, selected-index sets, attention probs)
// needed by the bilevel backward. Mamba scan saved-state tensors dropped.
void supergrok2_bilevel_fwd_save(
 torch::Tensor grad, torch::Tensor sharpness,
 torch::Tensor input_proj_W, torch::Tensor input_proj_b,
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
 torch::Tensor csa_compress_w,
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ,
 torch::Tensor csa_idx_K,
 torch::Tensor csa_out_W,
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
 torch::Tensor hca_out_W,
 int d_model, int num_heads,
 int csa_compress, int csa_window, int csa_topk,
 int hca_compress, int indexer_rank,
 torch::Tensor csa_ctx_out, torch::Tensor hca_ctx_out,
 torch::Tensor x_sorted, torch::Tensor sort_indices,
 torch::Tensor csa_saved_denom,
 torch::Tensor csa_saved_sel_idx,
 torch::Tensor csa_saved_probs,
 torch::Tensor hca_saved_denom,
 torch::Tensor hca_saved_probs,
 int checkpoint_interval
) {
 if (grad.numel() == 0) return;
 SG_DISPATCH(launch_csa_hca_bilevel_fwd_save,
 grad, sharpness,
 input_proj_W, input_proj_b,
 csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
 csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
 hca_q_W, hca_k_W, hca_v_W, hca_out_W,
 d_model, num_heads,
 csa_compress, csa_window, csa_topk,
 hca_compress, indexer_rank,
 csa_ctx_out, hca_ctx_out,
 x_sorted, sort_indices,
 csa_saved_denom, csa_saved_sel_idx, csa_saved_probs,
 hca_saved_denom, hca_saved_probs,
 checkpoint_interval);
}

// SG2 batched bilevel forward-save (packed across params).
void supergrok2_bilevel_fwd_save_batched(
 std::vector<torch::Tensor> grads,
 std::vector<torch::Tensor> sharpness_list,
 torch::Tensor input_proj_W, torch::Tensor input_proj_b,
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
 torch::Tensor csa_compress_w,
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ,
 torch::Tensor csa_idx_K,
 torch::Tensor csa_out_W,
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
 torch::Tensor hca_out_W,
 int d_model, int num_heads,
 int csa_compress, int csa_window, int csa_topk,
 int hca_compress, int indexer_rank,
 torch::Tensor csa_ctx_out_packed,
 torch::Tensor hca_ctx_out_packed,
 torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
 torch::Tensor sort_indices_packed,
 torch::Tensor csa_saved_denom_packed,
 torch::Tensor csa_saved_sel_idx_packed,
 torch::Tensor csa_saved_probs_packed,
 torch::Tensor hca_saved_denom_packed,
 torch::Tensor hca_saved_probs_packed,
 int checkpoint_interval
) {
 if (grads.empty()) return;
 SG_DISPATCH(launch_csa_hca_bilevel_fwd_save_batched,
 grads, sharpness_list,
 input_proj_W, input_proj_b,
 csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
 csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
 hca_q_W, hca_k_W, hca_v_W, hca_out_W,
 d_model, num_heads,
 csa_compress, csa_window, csa_topk,
 hca_compress, indexer_rank,
 csa_ctx_out_packed, hca_ctx_out_packed,
 x_sorted_packed, offsets_t, sort_indices_packed,
 csa_saved_denom_packed, csa_saved_sel_idx_packed,
 csa_saved_probs_packed,
 hca_saved_denom_packed, hca_saved_probs_packed,
 checkpoint_interval);
}

// SG2 bilevel backward through the CSA/HCA meta-net, single-tensor entry.
// Mamba weight/state block swapped for the CSA/HCA weight + attention adjoint
// saved-state block. FUNCTIONAL on BOTH arches (no throw): sm_90 and gfx942 run
// the real reverse-mode VJP via launch_csa_hca_backward →
// sg2adj::bilevel_backward_driver (vendor-neutral ATen adjoint, the LIVE CPU
// path). On a hipcc build the gfx942 device AMDGCN adjoint
// (supergrok2_bilevel_adjoint_gfx942.hip.hpp) is the live device path; numeric
// parity is 🟡 (MI300X). The earlier "forward only / throws" note was stale.
void supergrok2_bilevel_backward(
 torch::Tensor d_smart_grad,
 torch::Tensor grad, torch::Tensor sharpness, float rescale,
 torch::Tensor sort_indices, torch::Tensor x_sorted,
 torch::Tensor csa_ctx, torch::Tensor hca_ctx,
 torch::Tensor csa_saved_denom,
 torch::Tensor csa_saved_sel_idx,
 torch::Tensor csa_saved_probs,
 torch::Tensor hca_saved_denom,
 torch::Tensor hca_saved_probs,
 torch::Tensor gru_input, torch::Tensor gru_h_old,
 torch::Tensor gru_z_gate, torch::Tensor gru_r_gate,
 torch::Tensor gru_h_tilde,
 torch::Tensor peer_input, torch::Tensor expert_indices,
 torch::Tensor routing_weights, torch::Tensor saved_z_hidden,
 torch::Tensor saved_scores_a, torch::Tensor saved_scores_b,
 torch::Tensor saved_top_a_idx, torch::Tensor saved_top_b_idx,
 torch::Tensor saved_soft_a, torch::Tensor saved_soft_b,
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
 torch::Tensor csa_compress_w,
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ,
 torch::Tensor csa_idx_K,
 torch::Tensor csa_out_W,
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
 torch::Tensor hca_out_W,
 torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh,
 torch::Tensor peer_query_Ws,
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
 torch::Tensor expert_W1, torch::Tensor expert_W2,
 torch::Tensor expert_b1_in, torch::Tensor expert_b2_in,
 torch::Tensor input_proj_W,
 torch::Tensor d_csa_q_W, torch::Tensor d_csa_k_W,
 torch::Tensor d_csa_v_W, torch::Tensor d_csa_compress_w,
 torch::Tensor d_csa_idx_DQ, torch::Tensor d_csa_idx_UQ,
 torch::Tensor d_csa_idx_K, torch::Tensor d_csa_out_W,
 torch::Tensor d_hca_q_W, torch::Tensor d_hca_k_W,
 torch::Tensor d_hca_v_W, torch::Tensor d_hca_out_W,
 torch::Tensor d_gru_Wz, torch::Tensor d_gru_bz,
 torch::Tensor d_gru_Wr, torch::Tensor d_gru_br,
 torch::Tensor d_gru_Wh, torch::Tensor d_gru_bh,
 torch::Tensor d_peer_query_Ws,
 torch::Tensor d_prod_keys_A, torch::Tensor d_prod_keys_B,
 torch::Tensor d_expert_W1, torch::Tensor d_expert_b1,
 torch::Tensor d_expert_W2, torch::Tensor d_expert_b2,
 torch::Tensor d_input_proj_W, torch::Tensor d_input_proj_b,
 int d_model, int gru_hidden, int gru_input_dim,
 int num_heads, int topk, int pk_dim,
 int expert_hidden, int peer_input_dim, int num_experts,
 int csa_compress, int csa_window, int csa_topk,
 int hca_compress, int indexer_rank,
 int checkpoint_interval
) {
 if (d_smart_grad.numel() == 0) return;
 SG_DISPATCH(launch_csa_hca_backward,
 d_smart_grad, grad, sharpness, rescale,
 sort_indices, x_sorted,
 csa_ctx, hca_ctx,
 csa_saved_denom, csa_saved_sel_idx, csa_saved_probs,
 hca_saved_denom, hca_saved_probs,
 gru_input, gru_h_old, gru_z_gate, gru_r_gate, gru_h_tilde,
 peer_input, expert_indices, routing_weights, saved_z_hidden,
 saved_scores_a, saved_scores_b,
 saved_top_a_idx, saved_top_b_idx, saved_soft_a, saved_soft_b,
 csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
 csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
 hca_q_W, hca_k_W, hca_v_W, hca_out_W,
 gru_Wz, gru_Wr, gru_Wh,
 peer_query_Ws, prod_keys_A, prod_keys_B,
 expert_W1, expert_W2, expert_b1_in, expert_b2_in, input_proj_W,
 d_csa_q_W, d_csa_k_W, d_csa_v_W, d_csa_compress_w,
 d_csa_idx_DQ, d_csa_idx_UQ, d_csa_idx_K, d_csa_out_W,
 d_hca_q_W, d_hca_k_W, d_hca_v_W, d_hca_out_W,
 d_gru_Wz, d_gru_bz, d_gru_Wr, d_gru_br, d_gru_Wh, d_gru_bh,
 d_peer_query_Ws, d_prod_keys_A, d_prod_keys_B,
 d_expert_W1, d_expert_b1, d_expert_W2, d_expert_b2,
 d_input_proj_W, d_input_proj_b,
 d_model, gru_hidden, gru_input_dim,
 num_heads, topk, pk_dim,
 expert_hidden, peer_input_dim, num_experts,
 csa_compress, csa_window, csa_topk,
 hca_compress, indexer_rank,
 checkpoint_interval);
}

// SG2 batched bilevel backward (packed across params). CSA/HCA weight block
// + attention adjoint saved-state block replace the Mamba scan block.
void supergrok2_bilevel_backward_batched(
 torch::Tensor d_csa_ctx_packed,
 torch::Tensor d_hca_ctx_packed,
 torch::Tensor x_sorted_packed,
 torch::Tensor csa_saved_denom_packed,
 torch::Tensor csa_saved_sel_idx_packed,
 torch::Tensor csa_saved_probs_packed,
 torch::Tensor hca_saved_denom_packed,
 torch::Tensor hca_saved_probs_packed,
 torch::Tensor offsets_t,
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
 torch::Tensor csa_compress_w,
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ,
 torch::Tensor csa_idx_K,
 torch::Tensor csa_out_W,
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
 torch::Tensor hca_out_W,
 torch::Tensor d_csa_q_W, torch::Tensor d_csa_k_W,
 torch::Tensor d_csa_v_W, torch::Tensor d_csa_compress_w,
 torch::Tensor d_csa_idx_DQ, torch::Tensor d_csa_idx_UQ,
 torch::Tensor d_csa_idx_K, torch::Tensor d_csa_out_W,
 torch::Tensor d_hca_q_W, torch::Tensor d_hca_k_W,
 torch::Tensor d_hca_v_W, torch::Tensor d_hca_out_W,
 torch::Tensor d_x_sorted_packed,
 int d_model, int num_heads, int num_params,
 int csa_compress, int csa_window, int csa_topk,
 int hca_compress, int indexer_rank,
 int checkpoint_interval
) {
 if (num_params == 0) return;
 SG_DISPATCH(launch_csa_hca_backward_batched,
 d_csa_ctx_packed, d_hca_ctx_packed,
 x_sorted_packed,
 csa_saved_denom_packed, csa_saved_sel_idx_packed,
 csa_saved_probs_packed,
 hca_saved_denom_packed, hca_saved_probs_packed,
 offsets_t,
 csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
 csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
 hca_q_W, hca_k_W, hca_v_W, hca_out_W,
 d_csa_q_W, d_csa_k_W, d_csa_v_W, d_csa_compress_w,
 d_csa_idx_DQ, d_csa_idx_UQ, d_csa_idx_K, d_csa_out_W,
 d_hca_q_W, d_hca_k_W, d_hca_v_W, d_hca_out_W,
 d_x_sorted_packed,
 d_model, num_heads, num_params,
 csa_compress, csa_window, csa_topk,
 hca_compress, indexer_rank,
 checkpoint_interval);
}

// SG2 fused multi-tensor grad prepare (clip, finite, bias corrections) +
// batched CSA/HCA step. Keeps its name (spec §7); the per-arch impl swaps the
// Mamba weight bundle for the CSA/HCA weight set. Lives in
// csrc/kernels/{cuda,hip}/<arch>/multi_tensor_prepare_<arch>.{cu,hip.cpp}.
void supergrok2_prepare_and_batched_step(
 std::vector<torch::Tensor> params,
 std::vector<torch::Tensor> grads,
 std::vector<torch::Tensor> exp_avgs,
 std::vector<torch::Tensor> exp_avg_sqs,
 std::vector<torch::Tensor> gru_states,
 std::vector<torch::Tensor> mus,
 std::vector<torch::Tensor> slows,
 std::vector<torch::Tensor> sharpnesses,
 std::vector<int64_t> steps,
 std::vector<double> layer_alphas,
 std::vector<double> layer_beta1s,
 double base_alpha, double gradient_clipping,
 double beta2, double lr, double eps, double wd,
 double lamb, double ramp, double gate_signal,
 torch::Tensor input_proj_W, torch::Tensor input_proj_b,
 torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
 torch::Tensor csa_compress_w,
 torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ,
 torch::Tensor csa_idx_K,
 torch::Tensor csa_out_W,
 torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
 torch::Tensor hca_out_W,
 torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh,
 torch::Tensor gru_bz, torch::Tensor gru_br, torch::Tensor gru_bh,
 torch::Tensor peer_query_Ws,
 torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
 torch::Tensor expert_W1, torch::Tensor expert_b1,
 torch::Tensor expert_W2, torch::Tensor expert_b2,
 int64_t d_model, int64_t gru_hidden, int64_t num_heads, int64_t pk_dim,
 int64_t expert_hidden, int64_t num_experts,
 int64_t csa_compress, int64_t csa_window, int64_t csa_topk,
 int64_t hca_compress, int64_t indexer_rank,
 torch::Tensor expert_counts,
 int64_t peer_topk = 4
) {
 const size_t n = params.size();
 if (n == 0) return;
 check_params_grads(params, grads, "supergrok2_prepare_and_batched_step");
 // Secondary state + scalar lists are indexed in lockstep with params[i].
 check_list_len(exp_avgs, n, "exp_avgs", "supergrok2_prepare_and_batched_step");
 check_list_len(exp_avg_sqs, n, "exp_avg_sqs", "supergrok2_prepare_and_batched_step");
 check_list_len(gru_states, n, "gru_states", "supergrok2_prepare_and_batched_step");
 check_list_len(mus, n, "mus", "supergrok2_prepare_and_batched_step");
 check_list_len(slows, n, "slows", "supergrok2_prepare_and_batched_step");
 check_list_len(sharpnesses, n, "sharpnesses", "supergrok2_prepare_and_batched_step");
 check_list_len(steps, n, "steps", "supergrok2_prepare_and_batched_step");
 check_list_len(layer_alphas, n, "layer_alphas", "supergrok2_prepare_and_batched_step");
 check_list_len(layer_beta1s, n, "layer_beta1s", "supergrok2_prepare_and_batched_step");

 // Host-side global-norm gradient clipping (arch-independent prep).
 clip_grad_norms_device_side(grads, n, static_cast<float>(gradient_clipping));

 // Per-param scalars: clamped blend alpha, gated lambda, bias corrections.
 std::vector<torch::Tensor> sharp_list(n);
 std::vector<float> alpha_mus(n), lamb_effs(n), beta1s(n), bc1s(n), bc2s(n);
 const float lamb_eff = static_cast<float>(lamb * ramp * gate_signal);
 const float wd_eff = static_cast<float>(wd);
 for (size_t i = 0; i < n; ++i) {
 const float la = static_cast<float>(layer_alphas[i]);
 alpha_mus[i] = std::max(0.0f, std::min(1.0f,
 static_cast<float>(base_alpha) * la));
 lamb_effs[i] = lamb_eff;
 beta1s[i] = static_cast<float>(layer_beta1s[i]);
 bc1s[i] = 1.0f - std::pow(beta1s[i], static_cast<float>(steps[i]));
 bc2s[i] = 1.0f - std::pow(static_cast<float>(beta2),
 static_cast<float>(steps[i]));
 sharp_list[i] =
 (sharpnesses[i].defined() && sharpnesses[i].numel() == grads[i].numel())
 ? sharpnesses[i].to(grads[i].dtype())
 : torch::zeros_like(grads[i]);
 }

 // Smart-grad blend strength (rescale); matches the prior raw-lambda blend.
 const float rescale = static_cast<float>(lamb);

 SG_DISPATCH(launch_csa_hca_batched_step,
 params, grads, sharp_list, exp_avgs, exp_avg_sqs, mus, slows, gru_states,
 input_proj_W, input_proj_b,
 csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
 csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
 hca_q_W, hca_k_W, hca_v_W, hca_out_W,
 gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
 peer_query_Ws, prod_keys_A, prod_keys_B,
 expert_W1, expert_b1, expert_W2, expert_b2,
 alpha_mus, lamb_effs, beta1s, bc1s, bc2s,
 rescale, static_cast<float>(beta2),
 static_cast<float>(lr), wd_eff, static_cast<float>(eps),
 static_cast<int>(d_model), static_cast<int>(gru_hidden),
 static_cast<int>(num_heads), static_cast<int>(pk_dim),
 static_cast<int>(expert_hidden), static_cast<int>(num_experts),
 static_cast<int>(csa_compress), static_cast<int>(csa_window),
 static_cast<int>(csa_topk), static_cast<int>(hca_compress),
 static_cast<int>(indexer_rank), expert_counts,
 static_cast<int>(peer_topk));
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
#if defined(WITH_HIP)
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
#endif
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
#if defined(WITH_HIP)
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
#endif
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
 int d_conv, int expand, int n_layers, cudaStream_t,
 T* activation_cache = nullptr);
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
 }
#if defined(WITH_HIP)
 else if (sg_arch == 942) {
 err = ::sg::gfx942::models::mamba::forward<scalar_t>(
 reinterpret_cast<const scalar_t*>(input.data_ptr()),
 ctptr<scalar_t>(weights),
 tptr<scalar_t>(output),
 tptr<scalar_t>(states),
 batch, seq_len, d_model, d_state,
 d_conv, expand, n_layers, stream);
 }
#endif
 else {
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
 }
#if defined(WITH_HIP)
 else if (sg_arch == 942) {
 err = ::sg::gfx942::models::mamba::backward<scalar_t>(
 ctptr<scalar_t>(grad_output),
 ctptr<scalar_t>(states_saved),
 ctptr<scalar_t>(weights),
 tptr<scalar_t>(grad_input),
 tptr<scalar_t>(grad_weights),
 batch, seq_len, d_model, d_state,
 d_conv, expand, n_layers, stream);
 }
#endif
 else {
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
 }
#if defined(WITH_HIP)
 else if (sg_arch == 942) {
 err = ::sg::gfx942::models::mamba::selective_scan_fwd<scalar_t>(
 ctptr<scalar_t>(u), ctptr<scalar_t>(delta),
 ctptr<scalar_t>(A), ctptr<scalar_t>(B),
 tptr<scalar_t>(out), tptr<scalar_t>(state),
 batch, seq_len, d_state, stream);
 }
#endif
 else {
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
 }
#if defined(WITH_HIP)
 else if (sg_arch == 942) {
 err = ::sg::gfx942::models::mamba::selective_scan_bwd<scalar_t>(
 ctptr<scalar_t>(grad_out),
 ctptr<scalar_t>(u), ctptr<scalar_t>(delta),
 ctptr<scalar_t>(A), ctptr<scalar_t>(B),
 ctptr<scalar_t>(state),
 tptr<scalar_t>(grad_u), tptr<scalar_t>(grad_delta),
 tptr<scalar_t>(grad_A), tptr<scalar_t>(grad_B),
 batch, seq_len, d_state, stream);
 }
#endif
 else {
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

// OWNER-DECIDED REMOVAL (2026-06-10): the `_ops.models.*` submodule exported
// 13 per-op model kernels (decoder/vit/mamba fwd+bwd + attention/scan/patch
// component entries) with ZERO Python consumers anywhere (race, tuner, tests —
// verified by the cleanup inventory and re-verified at deletion). They were
// pre-megakernel scaffolding; the portable reuse surface for model math is the
// stage-header layer (csrc/fused/sm_90/model_stage*.cuh, COMPONENT_CONTRACT.md),
// not pybind exports — an exported, untested, unused API is liability, not
// portability. The underlying sg::* host functions + kernels are excised in the
// dead-code pass that accompanies the dispatch-table prune (kept this commit so
// the binding removal is independently revertible).
void register_model_bindings(py::module_& m) {
 (void)m;  // intentionally registers nothing — see removal note above
}


// ─── csrc/bindings/module.cpp (PYBIND11 entry) ─────────────────
// Forward declaration for model bindings aggregator (models_module.cpp)
void register_model_bindings(pybind11::module_& m);

// ── Exported-ABI schema version (A5-F6) ──────────────────────────────
// Single integer that versions the SET of exported pybind signatures below
// (every m.def's argument list + return type, and the secondary-list/scalar
// contracts the Python optimizers rely on). BUMP THIS on ANY exported-signature
// change: adding/removing/reordering a fused-step argument, changing a tensor
// list into a scalar (or vice-versa), changing a return tuple, or renaming an
// exported symbol. A stale prebuilt .so paired with newer Python wrappers (or
// the reverse) otherwise mis-marshals arguments silently; a version mismatch
// must fail loudly instead.
//
// The Python-side assertion (compare _ops.__abi_schema__ against the value the
// wrappers were written for) lands SEPARATELY in grokking_optimizers/dispatch.py
// (sibling-owned). Until that check exists this attribute is exported but inert
// — that is intentional and harmless: it merely makes the version observable
// now so the Python guard can be added without a coordinated ABI bump.
constexpr int GROK_ABI_SCHEMA = 1;

PYBIND11_MODULE(_ops, m) {
 m.doc() = "Grokking Optimizers — specialized per-arch C++/CUDA/HIP kernels";

 // Exported-ABI schema version (see GROK_ABI_SCHEMA above). Bump on ANY
 // exported-signature change; the Python-side compatibility assert lands in
 // dispatch.py (sibling-owned) and is inert until then.
 m.attr("__abi_schema__") = GROK_ABI_SCHEMA;

 m.def("detect_arch", &sg::detect_arch,
 "Returns 90 or 942 for the detected GPU (3-arch active set: "
 "sm_90, gfx942, tpu_v6e). TPU handled in Python.");

 // Fused (model, optimizer, arch) dispatch. The trailing scalar args carry the
 // FULL optimizer-state scalar set (C2-gap fix) so the fused tail's real math
 // runs; py::arg defaults reproduce the pre-fix inert behavior, so a short call
 // ops.fused_step(model, opt, params, input, grad, state, lr) still works.
 // bc1/bc2 are un-inverted (= 1 - beta^step). opt_only=true → L1 faithful tail.
 m.def("fused_step", &sg::fused_step,
 "Fused (model, optimizer, arch) kernel dispatch. Routes to the "
 "appropriate fused TU based on detected hardware. Carries the full "
 "optimizer scalar set (beta1/beta2/eps/weight_decay/alpha/lamb/gamma/"
 "gate/d_factor/bc1/bc2/neg_lr_scale/decay_factor/beta/alpha_max/step); "
 "opt_only=True selects the L1 real-grad optimizer tail (the faithful "
 "path), False the L3 surrogate-model fwd+bwd+opt.",
 py::arg("model"), py::arg("optimizer"), py::arg("params"),
 py::arg("input"), py::arg("grad"), py::arg("state"), py::arg("lr"),
 py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
 py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.01f,
 py::arg("alpha") = 0.98f, py::arg("lamb") = 2.0f,
 py::arg("gamma") = 0.0f, py::arg("gate") = 1.0f,
 py::arg("d_factor") = 1.0f, py::arg("bc1") = 1.0f, py::arg("bc2") = 1.0f,
 py::arg("neg_lr_scale") = 0.0f, py::arg("decay_factor") = 1.0f,
 py::arg("beta") = 0.0f, py::arg("alpha_max") = 1.0f,
 py::arg("step") = 1, py::arg("opt_only") = true,
 // GEMM-engine selector for the L3-REAL path (task 1). Default "scalar" keeps
 // the old call shape valid (back-compat); the Python wrapper passes "wgmma"
 // for decoder/vit at bf16 to take the tensor-core launchers.
 py::arg("gemm_impl") = "scalar",
 // GrokAdamW GLOBAL grad-norm clip threshold (decoder L3-TC, mechanism (ii)).
 // Trailing defaulted arg → back-compat preserved (a stale _ops without it
 // trips the caller's one-shot TypeError latch, loud degrade). ≤0 ⇒ no clip
 // (inert for every non-GrokAdamW cell); the decoder GrokAdamW cell passes the
 // optimizer's grad_clip (=1.0) so the kernel's P2.5 global-norm clip fires.
 py::arg("grad_clip") = 0.0f,
 // Prodigy estimator scalars (decoder L3-TC, STAGED global-d). Trailing defaulted
 // args → back-compat preserved (eager/inert for every non-Prodigy cell). The
 // decoder Prodigy cell passes d0/d_coef/beta3 so the kernel's P2.6 d-reduction
 // matches the eager multi-tensor estimator.
 py::arg("d0") = 1e-6f, py::arg("d_coef") = 1.0f, py::arg("beta3") = 0.0f,
 // Muon 1D-group AdamW hyperparameters (ViT L3-TC, STAGED NS). Trailing defaulted
 // args → back-compat preserved (eager Muon adamw_* defaults, inert for every
 // non-Muon cell). The vit Muon cell passes adamw_lr/adamw_betas so the kernel's
 // Muon P3 1D AdamW tail matches the eager non-2D group.
 py::arg("aux_lr") = 1e-3f, py::arg("aux_beta1") = 0.9f,
 py::arg("aux_beta2") = 0.98f,
 // LookSAM SAM 2nd-backward scalars (decoder/vit/mamba L3-TC, MODEL-COUPLED).
 // Trailing defaulted args → back-compat preserved (inert 0.0 for every non-LookSAM
 // cell; a stale _ops without them trips the caller's one-shot TypeError latch, loud
 // degrade). The LookSAM cell passes rho (perturbation radius) + looksam_sam (the
 // every-k SAM-step gate, 1.0 on SAM steps) so the kernel's P2.4 perturb→2nd
 // fwd+bwd→sam_dir=g_sam−g phase fires on the right cadence.
 py::arg("rho") = 0.0f, py::arg("looksam_sam") = 0.0f,
 // SuperGrok11/15 meta-net rescale (decoder/vit L3-TC). Trailing defaulted arg →
 // back-compat preserved (inert 0.0 for every non-SG cell; a stale _ops without it
 // trips the caller's one-shot TypeError latch, loud degrade). The SG cell passes the
 // SharpnessMetaNet rescale so the kernel's P2.45 meta-net mu precompute (mu =
 // rescale·phi(g, sharpness)) runs the live mechanism; the phi weights + sharpness
 // buffer ride the STATE buffer (cell-scattered), so only this scalar is in the ABI.
 py::arg("sg_rescale") = 0.0f,
 // SuperGrok11 cosine-gate temperature (decoder/vit/mamba L3-TC). Trailing
 // defaulted arg → back-compat (inert 1.0 for every non-SG11 cell). The SG11
 // cell passes the SharpnessMetaNet gate_temperature so the kernel's P2.45
 // finalizer computes gate = sigmoid(gate_temp · cos(grad, mu)) (the cosine
 // SIGNAL preserved; only the final squashing is the temperature-scaled sigmoid).
 py::arg("gate_temp") = 1.0f);

 // SuperGrok2 DEDICATED L3-TC entry (decoder/vit). The FULL CSA/HCA/PEER/GRU
 // meta-net as the optimizer phase: in-kernel SEGMENTED SORT (STAGE -1) + SAM 2nd
 // backward → sharpness + sg2_meta_stages. Needs the meta-net weight bundle + the
 // per-tensor scalar arrays this generic fused_step ABI cannot carry, so it is a
 // PARALLEL entry (fused_step + the 28 byte-identical cells are UNTOUCHED).
 m.def("sg2_fused_step", &sg::sg2_fused_step,
 "SuperGrok2 L3-TC fused train step (decoder/vit): in-kernel segmented sort + "
 "SAM 2nd backward + full CSA/HCA/PEER/GRU meta-net as the optimizer phase. "
 "Writes the reduced grad into `grad` and the loss into state[3*total].",
 py::arg("model"), py::arg("params"), py::arg("input"), py::arg("grad"),
 py::arg("state"),
 py::arg("input_proj_W"), py::arg("input_proj_b"),
 py::arg("csa_q_W"), py::arg("csa_k_W"), py::arg("csa_v_W"), py::arg("csa_out_W"),
 py::arg("csa_compress_w"), py::arg("csa_idx_DQ"), py::arg("csa_idx_K"),
 py::arg("hca_q_W"), py::arg("hca_k_W"), py::arg("hca_v_W"), py::arg("hca_out_W"),
 py::arg("gru_Wz"), py::arg("gru_bz"), py::arg("gru_Wr"), py::arg("gru_br"),
 py::arg("gru_Wh"), py::arg("gru_bh"),
 py::arg("peer_query_Ws"), py::arg("prod_keys_A"), py::arg("prod_keys_B"),
 py::arg("expert_W1"), py::arg("expert_b1"), py::arg("expert_W2"), py::arg("expert_b2"),
 py::arg("sc_alpha"), py::arg("sc_gru_decay"), py::arg("sc_lamb_eff"),
 py::arg("sc_beta1"), py::arg("sc_bc1"), py::arg("sc_bc2"),
 py::arg("rescale"), py::arg("beta2"), py::arg("lr"), py::arg("wd"), py::arg("eps"),
 py::arg("rho"), py::arg("sam_on"), py::arg("step"));

 // GrokAdamW
 m.def("grokadamw_step", &sg::grokadamw_step);
 m.def("grokadamw_clip_step", &sg::grokadamw_clip_step);
 m.def("grokadamw_step_q3", &sg::grokadamw_step_q3);
 m.def("grokadamw_fused_step", &sg::grokadamw_fused_step,
 "PUBLISHED GrokAdamW vector step: per-tensor layer_beta1s (layer-wise "
 "momentum decay) + live scalar alpha_t (grokking-signal-modulated).");
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
 "Prodigy fused step (persistent-EMA D-estimate); returns "
 "(d_lr, r_ema, s_ema)");

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

 // SuperGrok v2 — CSA/HCA hybrid-attention meta-net entries.
 // Wrappers in csrc/bindings/supergrok2.cpp dispatch to the per-arch
 // launchers launch_csa_hca_* in csrc/kernels/{cuda/<sm>,hip/<gfx>}/.
 // Both the new names and the old "mamba_peer" names are registered (the
 // old ones alias the same C++ fn) so existing callers/tests keep working.
 m.def("supergrok2_step", &sg::supergrok2_step,
 "SuperGrok2: per-param CSA/HCA meta-net + mu + Adam + WD");
 m.def("supergrok2_mamba_peer_step", &sg::supergrok2_step,
 "Alias of supergrok2_step (back-compat)");
 m.def("supergrok2_batched_step", &sg::supergrok2_batched_step,
 "SuperGrok2: batched CSA/HCA step for all params at once");
 m.def("supergrok2_mamba_peer_batched_step", &sg::supergrok2_batched_step,
 "Alias of supergrok2_batched_step (back-compat)");
 m.def("sg2_meta_optimizer_tail", &sg::sg2_meta_optimizer_tail,
 "SuperGrok2 full meta-net as ONE persistent megakernel (launch-elimination "
 "of csa_hca_step_one). Consumes the per-tensor PACKED flat buffers + the "
 "pre-computed |grad|-ascending sort perms; runs CSA/HCA/GRU/PEER/apply "
 "in-kernel.",
 py::arg("params_packed"),
 py::arg("grads_packed"),
 py::arg("sharpness_packed"),
 py::arg("exp_avg_packed"),
 py::arg("exp_avg_sq_packed"),
 py::arg("mu_packed"),
 py::arg("slow_packed"),
 py::arg("gru_state_packed"),
 py::arg("perm_packed"),
 py::arg("unsort_packed"),
 py::arg("n_per_tensor"),
 py::arg("row_off"),
 py::arg("workspace"),
 py::arg("ws_stride"),
 py::arg("input_proj_W"), py::arg("input_proj_b"),
 py::arg("csa_q_W"), py::arg("csa_k_W"), py::arg("csa_v_W"), py::arg("csa_out_W"),
 py::arg("csa_compress_w"), py::arg("csa_idx_DQ"), py::arg("csa_idx_K"),
 py::arg("hca_q_W"), py::arg("hca_k_W"), py::arg("hca_v_W"), py::arg("hca_out_W"),
 py::arg("gru_Wz"), py::arg("gru_bz"), py::arg("gru_Wr"), py::arg("gru_br"),
 py::arg("gru_Wh"), py::arg("gru_bh"),
 py::arg("peer_query_Ws"), py::arg("prod_keys_A"), py::arg("prod_keys_B"),
 py::arg("expert_W1"), py::arg("expert_b1"), py::arg("expert_W2"), py::arg("expert_b2"),
 py::arg("alpha"), py::arg("gru_decay"), py::arg("lamb_eff"),
 py::arg("beta1"), py::arg("bc1"), py::arg("bc2"),
 py::arg("rescale"), py::arg("beta2"), py::arg("lr"), py::arg("wd"), py::arg("eps"),
 py::arg("g_next_task"), py::arg("g_arrived"), py::arg("g_generation"));
 m.def("sg2_ws_stride", &sg::sg2_ws_stride,
 "Authoritative floats-per-CTA workspace stride for the SG2 megakernel "
 "(== sg2_ws_stride<SG2Dims<>>(Nmax)); the Python driver calls this so the "
 "host allocation can never drift from the kernel's workspace carve.",
 py::arg("Nmax"));
 m.def("supergrok2_bilevel_fwd_save", &sg::supergrok2_bilevel_fwd_save,
 "SuperGrok2 bilevel: CSA/HCA forward with adjoint state saving");
 m.def("supergrok2_bilevel_fwd_save_batched", &sg::supergrok2_bilevel_fwd_save_batched,
 "SuperGrok2 bilevel: batched CSA/HCA forward with state saving");
 m.def("supergrok2_bilevel_backward", &sg::supergrok2_bilevel_backward,
 "SuperGrok2 bilevel: full backward through CSA/HCA meta-net");
 m.def("supergrok2_bilevel_backward_batched", &sg::supergrok2_bilevel_backward_batched,
 "SuperGrok2 bilevel: batched full backward through CSA/HCA meta-net");
 m.def("supergrok2_prepare_and_batched_step", &sg::supergrok2_prepare_and_batched_step,
 "Fused multi-tensor grad prepare (clip, finite, bias corrections) "
 "+ batched CSA/HCA step. Replaces N per-parameter Python "
 "iterations with one kernel launch.");

 // Model bindings (decoder, vit, mamba) — registered as _ops.models.*
 register_model_bindings(m);
}
