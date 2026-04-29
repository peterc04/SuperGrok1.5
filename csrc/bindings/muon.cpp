// bindings/muon.cpp — runtime dispatch to per-arch Muon launchers.
//
// Muon has launchers for the momentum normalize step, Newton-Schulz combine,
// fused NS + parameter update, and a top-level fused-step entry point.
// CUTLASS migration for the X·Xᵀ·X NS GEMM pattern (sm_90, sm_100) happens
// inside the per-arch launcher, not here.

#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_MUON(NS)                                                      \
    namespace NS {                                                            \
        void launch_muon_momentum_normalize(                                  \
            torch::Tensor momentum, torch::Tensor grad,                       \
            torch::Tensor X_out, float momentum_decay, float inv_norm);       \
        void launch_muon_ns_combine(                                          \
            torch::Tensor X, torch::Tensor AX, torch::Tensor AAX,             \
            torch::Tensor out, float a, float b, float c);                    \
        void launch_muon_ns_combine_update_fused(                             \
            torch::Tensor param, torch::Tensor X, torch::Tensor AX,           \
            torch::Tensor AAX,                                                \
            float a, float b, float c, float lr, float weight_decay);         \
        void launch_muon_fused_step(                                          \
            torch::Tensor param, torch::Tensor momentum, torch::Tensor grad,  \
            int ns_steps,                                                     \
            float momentum_decay, float lr, float weight_decay);              \
    }

DECLARE_MUON(sm80) DECLARE_MUON(sm90)
DECLARE_MUON(sm100) DECLARE_MUON(gfx942)
#undef DECLARE_MUON

void muon_momentum_normalize(
    torch::Tensor momentum, torch::Tensor grad, torch::Tensor X_out,
    float momentum_decay, float inv_norm)
{
    SG_DISPATCH(launch_muon_momentum_normalize,
        momentum, grad, X_out, momentum_decay, inv_norm);
}

void muon_ns_combine(
    torch::Tensor X, torch::Tensor AX, torch::Tensor AAX, torch::Tensor out,
    float a, float b, float c)
{
    SG_DISPATCH(launch_muon_ns_combine, X, AX, AAX, out, a, b, c);
}

void muon_ns_combine_update_fused(
    torch::Tensor param, torch::Tensor X, torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c, float lr, float weight_decay)
{
    SG_DISPATCH(launch_muon_ns_combine_update_fused,
        param, X, AX, AAX, a, b, c, lr, weight_decay);
}

void muon_fused_step(
    torch::Tensor param, torch::Tensor momentum, torch::Tensor grad,
    int ns_steps, float momentum_decay, float lr, float weight_decay)
{
    SG_DISPATCH(launch_muon_fused_step,
        param, momentum, grad, ns_steps, momentum_decay, lr, weight_decay);
}

} // namespace sg
