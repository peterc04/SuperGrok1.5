// =====================================================================
//  csrc/kernels/hip/gfx942/muon.hip.h
//
//  gfx942 Muon launchers. The Newton-Schulz iterations are orchestrated
//  host-side (csrc/bindings/muon.cpp::muon_fused_step) using torch::mm
//  (which routes through rocBLAS on HIP). This TU only provides the
//  per-element kernels: momentum_normalize, ns_combine, update, and the
//  fused ns_combine_update.
//
//  No CUTLASS dependency on HIP. rocBLAS is used implicitly via torch::mm.
// =====================================================================

#pragma once

#include <torch/extension.h>

namespace sg { namespace gfx942 {

void launch_muon_momentum_normalize(
    torch::Tensor buf, torch::Tensor X, torch::Tensor grad,
    float momentum, float inv_norm);

void launch_muon_ns_combine(
    torch::Tensor X_out, torch::Tensor X,
    torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c);

void launch_muon_update(
    torch::Tensor param, torch::Tensor orth,
    float neg_lr_scale, float decay_factor);

void launch_muon_ns_combine_update_fused(
    torch::Tensor param, torch::Tensor X,
    torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c,
    float neg_lr_scale, float decay_factor);

}} // namespace sg::gfx942
