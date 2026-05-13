// HIP gfx942 launch glue for AdamW.
// Algorithm: csrc/algorithms/adamw.h (math spec)
// Primitives: csrc/backends/hip/gfx942/primitives.hpp (ATen helpers)
//
// Note: .hip.cpp files route through the host compiler (g++/clang++), NOT
// hipcc. Therefore this file cannot contain __global__ kernels. Instead
// we implement AdamW via ATen tensor ops, which PyTorch dispatches to
// hipBLAS / rocBLAS / internal HIP kernels.

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace hip_gfx942 {

namespace prim = ::sg::hip_gfx942::primitives;

void launch_adamw_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    auto pack = prim::pack_valid(params, grads, exp_avgs, exp_avg_sqs);
    for (size_t i = 0; i < pack.params.size(); i++) {
        auto& p = pack.params[i];
        auto& g = pack.grads[i];
        auto& m = pack.state_a[i];
        auto& v = pack.state_b[i];

        prim::ema_update_inplace(m, g, beta1);
        prim::ema_sq_update_inplace(v, g, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::hip_gfx942
