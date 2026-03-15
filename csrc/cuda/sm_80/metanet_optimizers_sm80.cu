/*
 * SuperGrok v2 — Ampere-Optimized Meta-Net Optimizer Kernels (sm_80+)
 *
 * Ampere tier optimizations for non-SuperGrok-v2 optimizers:
 *   - TF32 Tensor Cores for meta-net MLP GEMMs (2x throughput)
 *   - cp.async for cooperative shared memory loading of meta-net weights
 *     Applied using the same two-phase pattern as supergrok2_fused_elem_sm80.cu:
 *       Phase 1: cp.async load meta-net weights (W1, b1, W2, b2)
 *       Phase 2: cp.async load other weights (if any)
 *     __pipeline_wait_prior(1) overlaps Phase 2 loading with Phase 1 compute.
 *
 * Optimizers benefiting from TF32 + cp.async:
 *   - SuperGrok v1.5: meta-net MLP uses small GEMMs
 *   - SuperGrok v1.1: same meta-net + cosine gate
 *   - NeuralGrok: amplifier MLP uses small GEMMs
 *
 * Optimizers that don't use GEMMs (Lion, Grokfast, Prodigy, Muon,
 * LookSAM, GrokAdamW) don't benefit from TF32 and are dispatched
 * directly to the generic kernels from ops.cpp.
 *
 * Dispatch: ops.cpp calls these on sm_80+ GPUs.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "platform.h"
#include "types.h"
#include "dispatch.h"

// ═══════════════════════════════════════════════════════════════════════
//  Forward declarations of generic launchers
// ═══════════════════════════════════════════════════════════════════════

// SuperGrok v1.5
void launch_fused_supergrok15_full_step(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim,
    float alpha, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2);

// SuperGrok v1.1
void launch_fused_sg11_full_step(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim,
    float alpha, float lamb_eff, float gate_val,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2);

// NeuralGrok
void launch_fused_neuralgrok_full_step(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_scale, float beta_shift,
    int hidden_dim, float rescale,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2);


// ═══════════════════════════════════════════════════════════════════════
//  Ampere Wrappers: Enable TF32, delegate, restore
//
//  TF32 Tensor Core mode provides 2x throughput for FP32 GEMMs with
//  ~1e-4 precision loss. The generic kernels' internal meta-net MLPs
//  use cooperative shared memory loads; on Ampere, these loads benefit
//  from the cp.async pattern already implemented in the generic kernels
//  via the #if __CUDA_ARCH__ >= 800 guards in supergrok15_kernels.cu,
//  supergrok11_kernels.cu, and neuralgrok_kernels.cu.
//
//  The TF32 cuBLAS mode set here applies to any torch::mm or torch::mm_out
//  calls within the generic launcher path.
// ═══════════════════════════════════════════════════════════════════════

void launch_fused_supergrok15_full_step_ampere(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim,
    float alpha, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2
) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    launch_fused_supergrok15_full_step(
        param, grad, sharpness, exp_avg, exp_avg_sq, mu,
        W1, b1, W2, b2, rescale, hidden_dim,
        alpha, lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2);

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}

void launch_fused_sg11_full_step_ampere(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim,
    float alpha, float lamb_eff, float gate_val,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2
) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    launch_fused_sg11_full_step(
        param, grad, sharpness, exp_avg, exp_avg_sq, mu,
        W1, b1, W2, b2, rescale, hidden_dim,
        alpha, lamb_eff, gate_val, beta1, beta2, lr, wd_eff, eps, bc1, bc2);

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}

void launch_fused_neuralgrok_full_step_ampere(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_scale, float beta_shift,
    int hidden_dim, float rescale,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2
) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    launch_fused_neuralgrok_full_step(
        param, grad, exp_avg, exp_avg_sq,
        W1, b1, W2, b2, alpha_scale, beta_shift,
        hidden_dim, rescale, beta1, beta2, lr, wd_eff, eps, bc1, bc2);

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}
