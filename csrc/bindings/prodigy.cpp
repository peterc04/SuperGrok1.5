// bindings/prodigy.cpp — runtime dispatch to per-arch Prodigy launchers.
#include "_dispatch_macro.h"
#include "_helpers.h"

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
