// bindings/multi_tensor.cpp — runtime dispatch to per-arch multi-tensor launchers.
//
// The multi-tensor path fuses many small parameter tensors into one launch.
// Currently registered: GrokAdamW and Lion. (Grokfast EMA's multi-tensor
// variant is wired in csrc/bindings/grokfast.cpp; Prodigy's
// fused-reduce-step multi-tensor variant is wired in csrc/bindings/prodigy.cpp.)

#include "_dispatch_macro.h"

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
