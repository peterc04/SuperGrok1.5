// bindings/grokfast.cpp — runtime dispatch to per-arch Grokfast launchers.
#include "_dispatch_macro.h"
#include "_helpers.h"

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
