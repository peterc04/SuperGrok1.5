// bindings/looksam.cpp — runtime dispatch to per-arch LookSAM launchers.
//
// LookSAM has 6 launchers (perturb, restore, direction, adjust,
// direction_adjust_fused, norm_reduce). Stubbed here with the primary
// fused-direction-adjust path; remaining secondary launchers TODO.

#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_LOOKSAM(NS)                                                   \
    namespace NS {                                                            \
        void launch_looksam_perturb(                                          \
            torch::Tensor param, torch::Tensor backup, torch::Tensor grad,   \
            float rho_over_norm);                                             \
        void launch_looksam_restore(                                          \
            torch::Tensor param, torch::Tensor backup);                       \
        void launch_looksam_direction_adjust_fused(                           \
            torch::Tensor grad, torch::Tensor sam_grad,                       \
            torch::Tensor v_dir,                                              \
            float inv_norm, float lambda, float grad_norm);                   \
        void launch_looksam_norm_reduce(                                      \
            torch::Tensor grad, torch::Tensor sam_grad,                       \
            torch::Tensor results /* [diff_norm, grad_norm] */);              \
    }

DECLARE_LOOKSAM(sm80) DECLARE_LOOKSAM(sm90)
DECLARE_LOOKSAM(sm100) DECLARE_LOOKSAM(gfx942)
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

} // namespace sg
