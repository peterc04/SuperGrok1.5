// =====================================================================
//  csrc/kernels/hip/gfx942/looksam.hip.h
// =====================================================================

#pragma once

#include <torch/extension.h>

namespace sg { namespace gfx942 {

void launch_looksam_perturb(
    torch::Tensor param, torch::Tensor backup, torch::Tensor grad,
    float rho_over_norm);

void launch_looksam_restore(
    torch::Tensor param, torch::Tensor backup);

void launch_looksam_direction_adjust_fused(
    torch::Tensor grad, torch::Tensor sam_grad,
    torch::Tensor v_dir,
    float inv_norm, float lambda, float grad_norm);

void launch_looksam_norm_reduce(
    torch::Tensor grad, torch::Tensor sam_grad,
    torch::Tensor results /* [diff_norm, grad_norm] */);

}} // namespace sg::gfx942
