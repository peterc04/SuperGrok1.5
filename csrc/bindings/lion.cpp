// bindings/lion.cpp — runtime dispatch to per-arch Lion launchers.
//
// Lion has a single launcher (launch_fused_lion_step). See bindings/grokadamw.cpp
// for the worked-example dispatcher pattern.

#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_LION(NS)                                                      \
    namespace NS {                                                            \
        void launch_fused_lion_step(                                          \
            torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,   \
            float lr, float beta1, float beta2, float weight_decay);          \
    }

DECLARE_LION(sm80) DECLARE_LION(sm90) DECLARE_LION(sm100) DECLARE_LION(gfx942)
DECLARE_LION(sm89) DECLARE_LION(sm103) DECLARE_LION(sm120) DECLARE_LION(gfx950)
#undef DECLARE_LION

void lion_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,
    float lr, float beta1, float beta2, float weight_decay)
{
    SG_DISPATCH(launch_fused_lion_step,
        param, exp_avg, grad, lr, beta1, beta2, weight_decay);
}

} // namespace sg
