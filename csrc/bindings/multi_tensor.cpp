// bindings/multi_tensor.cpp — runtime dispatch to per-arch multi-tensor launchers.
//
// The multi-tensor path fuses many small parameter tensors into one launch.
// Supports: GrokAdamW, Lion, Grokfast EMA, Prodigy step, Prodigy fused
// reduce+step, plus the SG2 prepare-and-batched-step pre-kernel.

#include "_dispatch_macro.h"

#include <vector>

namespace sg {

#define DECLARE_MT(NS)                                                        \
    namespace NS {                                                            \
        void launch_multi_tensor_grokadamw(                                   \
            std::vector<torch::Tensor> params,                                \
            std::vector<torch::Tensor> exp_avgs,                              \
            std::vector<torch::Tensor> exp_avg_sqs,                           \
            std::vector<torch::Tensor> emas,                                  \
            std::vector<torch::Tensor> grads,                                 \
            float alpha, float lamb,                                          \
            float beta1, float beta2,                                         \
            float lr, float weight_decay,                                     \
            float eps, float bc1, float bc2);                                 \
        void launch_multi_tensor_lion(                                        \
            std::vector<torch::Tensor> params,                                \
            std::vector<torch::Tensor> exp_avgs,                              \
            std::vector<torch::Tensor> grads,                                 \
            float lr, float beta1, float beta2, float weight_decay);          \
        void launch_multi_tensor_grokfast_ema(                                \
            std::vector<torch::Tensor> grads,                                 \
            std::vector<torch::Tensor> emas,                                  \
            float alpha, float lamb);                                         \
        void launch_multi_tensor_prodigy_step(                                \
            std::vector<torch::Tensor> params,                                \
            std::vector<torch::Tensor> exp_avgs,                              \
            std::vector<torch::Tensor> exp_avg_sqs,                           \
            std::vector<torch::Tensor> ss,                                    \
            std::vector<torch::Tensor> param_inits,                           \
            std::vector<torch::Tensor> grads,                                 \
            float lr, float d_lr, float beta1, float beta2,                   \
            float weight_decay, float eps, float bc1, float bc2);             \
        void supergrok2_prepare_and_batched_step(                             \
            /* SG2 batched prepare; signature mirrors                         \
               csrc/cuda/generic/multi_tensor_prepare.cu */);                 \
    }

DECLARE_MT(sm80) DECLARE_MT(sm90) DECLARE_MT(sm100) DECLARE_MT(gfx942)
DECLARE_MT(sm89) DECLARE_MT(sm103) DECLARE_MT(sm120) DECLARE_MT(gfx950)
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

// TODO(structural-refactor): grokfast_ema and prodigy multi-tensor entry
// points follow the same DISPATCH pattern; flesh out when wiring pybind11.

} // namespace sg
