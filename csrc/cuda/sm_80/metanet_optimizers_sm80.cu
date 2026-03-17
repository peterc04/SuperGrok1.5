/*
 * SuperGrok v2 — Ampere-Optimized Meta-Net Optimizer Kernels (sm_80+)
 *
 * WHAT THIS FILE DOES:
 *   - fused_supergrok15_full_step_cpasync_kernel: SuperGrok v1.5 with cp.async weight loading
 *   - fused_sg11_full_step_cpasync_kernel: SuperGrok v1.1 with cp.async weight loading
 *   - fused_neuralgrok_full_step_cpasync_kernel: NeuralGrok with cp.async weight loading
 *
 * KEY OPTIMIZATION — Two-Phase Pipelined Weight Loading:
 *   Phase 1: cp.async load W1 and b1 into shared memory, __pipeline_commit()
 *   Phase 2: cp.async load W2 and b2 into shared memory, __pipeline_commit()
 *   __pipeline_wait_prior(1) — W1,b1 ready; W2,b2 still loading in background
 *   Compute MLP layer 1 (using W1, b1 from shared memory)
 *   __pipeline_wait_prior(0) — W2,b2 now ready
 *   Compute MLP layer 2 (using W2, b2 from shared memory)
 *
 *   This overlaps the W2/b2 global->shared transfer with MLP layer 1 compute,
 *   hiding ~200 cycles of memory latency behind the GELU/ReLU MLP computation.
 *
 * WHAT IT DOES NOT DO:
 *   - Does not use FP8 (Hopper tier)
 *   - Does not modify optimizer math (identical to generic)
 *
 * Dispatch: ops.cpp calls these on sm_80+ GPUs.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "platform.h"
#include "types.h"
#include "dispatch.h"

#if GROK_CUDA
#include <cuda_pipeline.h>
#endif

// Block sizes matching the generic kernels
constexpr int BLOCK_SIZE = 256;
constexpr int NEURALGROK_BLOCK_SIZE = 256;


// ═══════════════════════════════════════════════════════════════════════
//  Forward declarations of generic launchers (fallback reference)
//
//  These are declared here for reference but NOT called — the Ampere
//  launchers below launch real cp.async kernels instead.
// ═══════════════════════════════════════════════════════════════════════

void launch_fused_supergrok15_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_fused_sg11_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim);

void launch_fused_neuralgrok_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor grad, torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2);


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: SuperGrok v1.5 with cp.async weight loading
//
//  Computation is identical to fused_supergrok15_full_step_kernel in
//  csrc/cuda/generic/supergrok15_kernels.cu.  The shared memory loading
//  uses two-phase __pipeline_memcpy_async so that W2/b2 transfer
//  overlaps with MLP layer 1 (GELU) computation.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void fused_supergrok15_full_step_cpasync_kernel(
    scalar_t* __restrict__ param,         // [N] — updated
    float* __restrict__ exp_avg,          // [N] — FP32 state
    float* __restrict__ exp_avg_sq,       // [N] — FP32 state
    scalar_t* __restrict__ mu,            // [N] — updated
    const scalar_t* __restrict__ grad,    // [N]
    const scalar_t* __restrict__ sharpness, // [N]
    const float alpha,
    const float* __restrict__ W1,         // [H, 2] (always FP32)
    const float* __restrict__ b1,         // [H]
    const float* __restrict__ W2,         // [1, H]
    const float* __restrict__ b2,         // [1]
    const float rescale,
    const float lamb_eff,
    const float beta1,
    const float beta2,
    const float lr,
    const float wd_eff,
    const float eps,
    const float bc1,
    const float bc2,
    const int N,
    const int H
) {
    // Shared memory layout: W1[H*2] | b1[H] | W2[H] | b2[1]
    extern __shared__ float smem[];
    float* sW1 = smem;
    float* sb1 = sW1 + H * 2;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // ── Phase 1: cp.async load W1 and b1 ────────────────────────────
    #pragma unroll 4
    for (int i = tid; i < H * 2; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    // ── Phase 2: cp.async load W2 and b2 ────────────────────────────
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    // Wait for Phase 1 (W1, b1 ready). Phase 2 (W2, b2) still in flight.
    __pipeline_wait_prior(1);
    __syncthreads();
#else
    // Synchronous fallback for non-Ampere
    #pragma unroll 4
    for (int i = tid; i < H * 2; i += blockDim.x) sW1[i] = W1[i];
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x) sb1[i] = b1[i];
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x) sW2[i] = W2[i];
    if (tid == 0) sb2[0] = b2[0];
    __syncthreads();
#endif

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    // ── 1. Read inputs ──────────────────────────────────────────
    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

    // ── 2. mu EMA ───────────────────────────────────────────────
    const float mu_old = static_cast<float>(mu[idx]);
    const float mu_new = alpha * mu_old + (1.0f - alpha) * g;
    mu[idx] = static_cast<scalar_t>(mu_new);

    // ── 3a. MLP Layer 1: Linear(2,H) -> GELU (uses W1, b1) ─────
    //    W2/b2 transfer overlaps with this computation.
    float hidden[128];  // register file; H <= 128 assumed
    #pragma unroll 4
    for (int h = 0; h < H; h++) {
        float z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];
        // Fast GELU: sigmoid approximation (~2.5x faster, max error ~0.004)
        hidden[h] = z / (1.0f + expf(-1.702f * z));
    }

    // ── Wait for Phase 2 (W2, b2 now ready) ─────────────────────
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_wait_prior(0);
    __syncthreads();
#endif

    // ── 3b. MLP Layer 2: Linear(H,1) (uses W2, b2) ─────────────
    float mlp_out = 0.0f;
    #pragma unroll 4
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    // ── 4. smart_grad (REGISTER ONLY — no global write) ────────
    const float smart_grad = g + rescale * mlp_out;

    // ── 5. Final gradient with gating ───────────────────────────
    const float fg = smart_grad + lamb_eff * mu_new;

    // ── 6. Adam moments ─────────────────────────────────────────
    const float ea = beta1 * exp_avg[idx] + (1.0f - beta1) * fg;
    const float easq = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * fg * fg;
    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    // ── 7. Bias-corrected step + progressive WD ─────────────────
    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: SuperGrok v1.1 with cp.async weight loading
//
//  Computation is identical to fused_sg11_full_step_kernel in
//  csrc/cuda/generic/supergrok11_kernels.cu.  Uses the same two-phase
//  pipelined cp.async pattern as Kernel 1.
//
//  Note: The cosine gate is computed host-side and folded into lamb_eff.
//  The kernel receives the pre-computed lamb_eff scalar — identical
//  structure to v1.5.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void fused_sg11_full_step_cpasync_kernel(
    scalar_t* __restrict__ param,         // [N] — updated
    float* __restrict__ exp_avg,          // [N] — FP32 state
    float* __restrict__ exp_avg_sq,       // [N] — FP32 state
    scalar_t* __restrict__ mu,            // [N] — updated
    const scalar_t* __restrict__ grad,    // [N]
    const scalar_t* __restrict__ sharpness, // [N]
    const float alpha,
    const float* __restrict__ W1,         // [H, 2] (always FP32)
    const float* __restrict__ b1,         // [H]
    const float* __restrict__ W2,         // [1, H]
    const float* __restrict__ b2,         // [1]
    const float rescale,
    const float lamb_eff,
    const float beta1,
    const float beta2,
    const float lr,
    const float wd_eff,
    const float eps,
    const float bc1,
    const float bc2,
    const int N,
    const int H
) {
    // Shared memory layout: W1[H*2] | b1[H] | W2[H] | b2[1]
    extern __shared__ float smem[];
    float* sW1 = smem;
    float* sb1 = sW1 + H * 2;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // ── Phase 1: cp.async load W1 and b1 ────────────────────────────
    #pragma unroll 4
    for (int i = tid; i < H * 2; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    // ── Phase 2: cp.async load W2 and b2 ────────────────────────────
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    // Wait for Phase 1 (W1, b1 ready). Phase 2 (W2, b2) still in flight.
    __pipeline_wait_prior(1);
    __syncthreads();
#else
    // Synchronous fallback for non-Ampere
    #pragma unroll 4
    for (int i = tid; i < H * 2; i += blockDim.x) sW1[i] = W1[i];
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x) sb1[i] = b1[i];
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x) sW2[i] = W2[i];
    if (tid == 0) sb2[0] = b2[0];
    __syncthreads();
#endif

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    // ── 1. Read inputs ──────────────────────────────────────────
    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

    // ── 2. mu EMA ───────────────────────────────────────────────
    const float mu_old = static_cast<float>(mu[idx]);
    const float mu_new = alpha * mu_old + (1.0f - alpha) * g;
    mu[idx] = static_cast<scalar_t>(mu_new);

    // ── 3a. MLP Layer 1: Linear(2,H) -> GELU (uses W1, b1) ─────
    //    W2/b2 transfer overlaps with this computation.
    float hidden[128];  // register file; H <= 128 assumed
    #pragma unroll 4
    for (int h = 0; h < H; h++) {
        float z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];
        // Fast GELU: sigmoid approximation (~2.5x faster, max error ~0.004)
        hidden[h] = z / (1.0f + expf(-1.702f * z));
    }

    // ── Wait for Phase 2 (W2, b2 now ready) ─────────────────────
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_wait_prior(0);
    __syncthreads();
#endif

    // ── 3b. MLP Layer 2: Linear(H,1) (uses W2, b2) ─────────────
    float mlp_out = 0.0f;
    #pragma unroll 4
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    // ── 4. smart_grad (REGISTER ONLY — no global write) ────────
    const float smart_grad = g + rescale * mlp_out;

    // ── 5. Final gradient with gating ───────────────────────────
    const float fg = smart_grad + lamb_eff * mu_new;

    // ── 6. Adam moments ─────────────────────────────────────────
    const float ea = beta1 * exp_avg[idx] + (1.0f - beta1) * fg;
    const float easq = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * fg * fg;
    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    // ── 7. Bias-corrected step + progressive WD ─────────────────
    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: NeuralGrok with cp.async weight loading
//
//  Computation is identical to fused_neuralgrok_full_step_kernel in
//  csrc/cuda/generic/neuralgrok_kernels.cu.  Uses the same two-phase
//  pipelined cp.async pattern.
//
//  Note: NeuralGrok uses W1[H,1] (single input) with ReLU activation,
//  vs SuperGrok's W1[H,2] (two inputs) with fast GELU.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void fused_neuralgrok_full_step_cpasync_kernel(
    scalar_t* __restrict__ param,             // [N] -- updated in-place
    float* __restrict__ exp_avg,              // [N] -- updated in-place
    float* __restrict__ exp_avg_sq,           // [N] -- updated in-place
    const scalar_t* __restrict__ grad,        // [N]
    const float* __restrict__ W1,             // [H, 1] -- row-major
    const float* __restrict__ b1,             // [H]
    const float* __restrict__ W2,             // [1, H] -- row-major
    const float* __restrict__ b2,             // [1]
    const float alpha,                        // Scale factor for MLP output
    const float beta,                         // Bias term (skip connection strength)
    const int N,
    const int H,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,                          // 1 - beta1^step
    const float bc2                           // 1 - beta2^step
) {
    // Shared memory layout: W1[H] | b1[H] | W2[H] | b2[1]
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    float* sW1 = smem;              // H elements
    float* sb1 = sW1 + H;          // H elements
    float* sW2 = sb1 + H;          // H elements
    float* sb2 = sW2 + H;          // 1 element

    const int tid = threadIdx.x;

#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // ── Phase 1: cp.async load W1 and b1 ────────────────────────────
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    // ── Phase 2: cp.async load W2 and b2 ────────────────────────────
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    // Wait for Phase 1 (W1, b1 ready). Phase 2 (W2, b2) still in flight.
    __pipeline_wait_prior(1);
    __syncthreads();
#else
    // Synchronous fallback for non-Ampere
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x)
        sW1[i] = W1[i];
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x)
        sb1[i] = b1[i];
    #pragma unroll 4
    for (int i = tid; i < H; i += blockDim.x)
        sW2[i] = W2[i];
    if (tid == 0)
        sb2[0] = b2[0];
    __syncthreads();
#endif

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    // ── MLP Layer 1: Linear(1,H) -> ReLU (uses W1, b1) ─────────
    //    W2/b2 transfer overlaps with this computation.
    const float g = static_cast<float>(grad[idx]);

    float hidden[128];  // register file; H <= 128 assumed
    #pragma unroll 4
    for (int h = 0; h < H; h++) {
        float z = sW1[h] * g + sb1[h];
        hidden[h] = (z > 0.0f) ? z : 0.0f;  // ReLU
    }

    // ── Wait for Phase 2 (W2, b2 now ready) ─────────────────────
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_wait_prior(0);
    __syncthreads();
#endif

    // ── MLP Layer 2: Linear(H,1) (uses W2, b2) ─────────────────
    float mlp_out = 0.0f;
    #pragma unroll 4
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    // amplified_grad lives only in this register -- never touches GMEM
    const float ag = g * (alpha * mlp_out + beta);

    // ── Adam update with register-held amplified_grad ───────────
    const float ea_old = exp_avg[idx];
    const float easq_old = exp_avg_sq[idx];

    const float ea = beta1 * ea_old + (1.0f - beta1) * ag;
    const float easq = beta2 * easq_old + (1.0f - beta2) * ag * ag;

    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;

    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}


// ═══════════════════════════════════════════════════════════════════════
//  Launcher 1: SuperGrok v1.5 Ampere
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
    const int N = grad.numel();
    if (N == 0) return;

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Shared memory: W1[H*2] + b1[H] + W2[H] + b2[1] = (4*H + 1) floats
    const int smem_bytes = (hidden_dim * 4 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_supergrok15_full_step_cpasync", ([&] {
        fused_supergrok15_full_step_cpasync_kernel<scalar_t><<<grid, BLOCK_SIZE, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            mu.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            sharpness.data_ptr<scalar_t>(),
            alpha,
            W1_f.data_ptr<float>(),
            b1_f.data_ptr<float>(),
            W2_f.data_ptr<float>(),
            b2_f.data_ptr<float>(),
            rescale, lamb_eff,
            beta1, beta2, lr, wd_eff, eps, bc1, bc2,
            N, hidden_dim
        );
    }));

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}


// ═══════════════════════════════════════════════════════════════════════
//  Launcher 2: SuperGrok v1.1 Ampere
// ═══════════════════════════════════════════════════════════════════════

void launch_fused_sg11_full_step_ampere(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim,
    float alpha, float lamb_eff, float gate_val,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2
) {
    const int N = grad.numel();
    if (N == 0) return;

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Shared memory: W1[H*2] + b1[H] + W2[H] + b2[1] = (4*H + 1) floats
    const int smem_bytes = (hidden_dim * 4 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    // Note: gate_val is accepted for API compatibility but the sg11 kernel
    // does not use it (cosine gate is applied at a higher level and folded
    // into lamb_eff).

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_sg11_full_step_cpasync", ([&] {
        fused_sg11_full_step_cpasync_kernel<scalar_t><<<grid, BLOCK_SIZE, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            mu.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            sharpness.data_ptr<scalar_t>(),
            alpha,
            W1_f.data_ptr<float>(),
            b1_f.data_ptr<float>(),
            W2_f.data_ptr<float>(),
            b2_f.data_ptr<float>(),
            rescale, lamb_eff,
            beta1, beta2, lr, wd_eff, eps, bc1, bc2,
            N, hidden_dim
        );
    }));

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}


// ═══════════════════════════════════════════════════════════════════════
//  Launcher 3: NeuralGrok Ampere
// ═══════════════════════════════════════════════════════════════════════

void launch_fused_neuralgrok_full_step_ampere(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_scale, float beta_shift,
    int hidden_dim, float rescale,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2
) {
    const int N = param.numel();
    if (N == 0) return;

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    const int grid = (N + NEURALGROK_BLOCK_SIZE - 1) / NEURALGROK_BLOCK_SIZE;
    // Shared memory: W1[H] + b1[H] + W2[H] + b2[1] = (3*H + 1) floats
    const int smem_elems = hidden_dim * 3 + 1;
    const int smem_bytes = smem_elems * sizeof(float);

    // Convert weights to FP32 only if needed (avoid redundant copy)
    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    // Note: rescale is accepted for API compatibility but is not used by
    // the NeuralGrok kernel (amplifier scaling is controlled by alpha_scale
    // and beta_shift).

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_neuralgrok_full_step_cpasync", ([&] {
            fused_neuralgrok_full_step_cpasync_kernel<scalar_t><<<grid, NEURALGROK_BLOCK_SIZE, smem_bytes>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                W1_f.data_ptr<float>(),
                b1_f.data_ptr<float>(),
                W2_f.data_ptr<float>(),
                b2_f.data_ptr<float>(),
                alpha_scale,
                beta_shift,
                N,
                hidden_dim,
                beta1,
                beta2,
                lr,
                wd_eff,
                eps,
                bc1,
                bc2
            );
        }));

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}


// ═══════════════════════════════════════════════════════════════════════
//  Explicit template instantiations
// ═══════════════════════════════════════════════════════════════════════

template __global__ void fused_supergrok15_full_step_cpasync_kernel<float>(
    float*, float*, float*, float*, const float*, const float*,
    const float, const float*, const float*, const float*, const float*,
    const float, const float, const float, const float, const float,
    const float, const float, const float, const float, const int, const int);
template __global__ void fused_supergrok15_full_step_cpasync_kernel<at::Half>(
    at::Half*, float*, float*, at::Half*, const at::Half*, const at::Half*,
    const float, const float*, const float*, const float*, const float*,
    const float, const float, const float, const float, const float,
    const float, const float, const float, const float, const int, const int);
template __global__ void fused_supergrok15_full_step_cpasync_kernel<at::BFloat16>(
    at::BFloat16*, float*, float*, at::BFloat16*, const at::BFloat16*, const at::BFloat16*,
    const float, const float*, const float*, const float*, const float*,
    const float, const float, const float, const float, const float,
    const float, const float, const float, const float, const int, const int);

template __global__ void fused_sg11_full_step_cpasync_kernel<float>(
    float*, float*, float*, float*, const float*, const float*,
    const float, const float*, const float*, const float*, const float*,
    const float, const float, const float, const float, const float,
    const float, const float, const float, const float, const int, const int);
template __global__ void fused_sg11_full_step_cpasync_kernel<at::Half>(
    at::Half*, float*, float*, at::Half*, const at::Half*, const at::Half*,
    const float, const float*, const float*, const float*, const float*,
    const float, const float, const float, const float, const float,
    const float, const float, const float, const float, const int, const int);
template __global__ void fused_sg11_full_step_cpasync_kernel<at::BFloat16>(
    at::BFloat16*, float*, float*, at::BFloat16*, const at::BFloat16*, const at::BFloat16*,
    const float, const float*, const float*, const float*, const float*,
    const float, const float, const float, const float, const float,
    const float, const float, const float, const float, const int, const int);

template __global__ void fused_neuralgrok_full_step_cpasync_kernel<float>(
    float*, float*, float*, const float*,
    const float*, const float*, const float*, const float*,
    const float, const float, const int, const int,
    const float, const float, const float, const float, const float,
    const float, const float);
template __global__ void fused_neuralgrok_full_step_cpasync_kernel<at::Half>(
    at::Half*, float*, float*, const at::Half*,
    const float*, const float*, const float*, const float*,
    const float, const float, const int, const int,
    const float, const float, const float, const float, const float,
    const float, const float);
template __global__ void fused_neuralgrok_full_step_cpasync_kernel<at::BFloat16>(
    at::BFloat16*, float*, float*, const at::BFloat16*,
    const float*, const float*, const float*, const float*,
    const float, const float, const int, const int,
    const float, const float, const float, const float, const float,
    const float, const float);
