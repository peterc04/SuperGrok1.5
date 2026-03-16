/*
 * Muon — Ampere-Optimized (sm_80+)
 *
 * Sets cuBLAS TF32 Tensor Core math mode for Newton-Schulz GEMMs,
 * providing ~2x throughput on A100. Element-wise kernels (momentum
 * normalize, NS combine, update) use the generic implementations
 * as they are compute-bound on FP32 ALUs, not on Tensor Cores.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

// Forward declare generic launcher
void launch_muon_fused_step(
    torch::Tensor param, torch::Tensor momentum_buffer, torch::Tensor grad,
    float lr, float momentum, float weight_decay, int ns_steps,
    float a, float b, float c);

void launch_muon_fused_step_ampere(
    torch::Tensor param, torch::Tensor momentum_buffer, torch::Tensor grad,
    float lr, float momentum, float weight_decay, int ns_steps,
    float a, float b, float c
) {
    // Set TF32 math mode for cuBLAS — 2x throughput on Ampere Tensor Cores
    // for the Newton-Schulz matrix multiplications (torch::mm calls)
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    launch_muon_fused_step(param, momentum_buffer, grad,
                           lr, momentum, weight_decay, ns_steps, a, b, c);

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}
