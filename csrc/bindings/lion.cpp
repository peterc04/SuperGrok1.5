// bindings/lion.cpp — runtime dispatch to per-arch Lion launchers.
//
// Lion has a single launcher (launch_fused_lion_step). See bindings/grokadamw.cpp
// for the worked-example dispatcher pattern.

#include "_dispatch_macro.h"
#include "_helpers.h"

#include <vector>

namespace sg {

#define DECLARE_LION(NS)                                                      \
    namespace NS {                                                            \
        void launch_fused_lion_step(                                          \
            torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,   \
            float lr, float beta1, float beta2, float weight_decay);          \
        void launch_multi_tensor_lion(                                        \
            std::vector<torch::Tensor>& params,                               \
            std::vector<torch::Tensor>& exp_avgs,                             \
            std::vector<torch::Tensor>& grads,                                \
            float lr, float beta1, float beta2, float wd);                    \
    }

DECLARE_LION(sm80) DECLARE_LION(sm90) DECLARE_LION(sm100) DECLARE_LION(gfx942)
DECLARE_LION(sm89) DECLARE_LION(sm103) DECLARE_LION(sm120) DECLARE_LION(gfx950)
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
