/*
 * SuperGrok v1.5 — CUDA Kernels
 *
 * Four fused kernels that eliminate per-element Python/kernel-launch overhead:
 *
 *   1. fused_mu_metanet  — EMA update + element-wise MLP inference
 *   2. fused_adam_decay   — gating blend + Adam moments + progressive wd + step
 *   3. sam_perturb        — worst-case parameter perturbation
 *   4. sharpness_restore  — |sam_grad − grad| + param restore
 *
 * The meta-net kernel is the key innovation: instead of reshape→matmul→reshape,
 * each thread independently evaluates the small MLP (2→H→1) for its own element.
 * With H=32 that's 64 multiply-adds per thread — trivial compute but fully
 * parallel across all gradient elements, with weights in shared memory.
 *
 * Supports FP32, FP16, and BF16 parameter tensors.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int BLOCK_SIZE = 256;

// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: Fused mu EMA update + meta-net inference
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void fused_mu_metanet_kernel(
    scalar_t* __restrict__ mu,           // [N] — updated in-place
    const scalar_t* __restrict__ grad,   // [N]
    const scalar_t* __restrict__ sharp,  // [N]
    scalar_t* __restrict__ smart_grad,   // [N] — output
    const float alpha,                   // EMA momentum for mu
    const float* __restrict__ W1,        // [H, 2] — row-major (always FP32)
    const float* __restrict__ b1,        // [H]
    const float* __restrict__ W2,        // [1, H] — row-major
    const float* __restrict__ b2,        // [1]
    const float rescale,
    const int N,
    const int H
) {
    // Load meta-net weights into shared memory (always FP32)
    extern __shared__ float smem[];
    float* sW1 = smem;              // H * 2
    float* sb1 = sW1 + H * 2;      // H
    float* sW2 = sb1 + H;          // H
    float* sb2 = sW2 + H;          // 1

    const int tid = threadIdx.x;

    // Cooperative load of weights into shared memory
    for (int i = tid; i < H * 2; i += blockDim.x)
        sW1[i] = W1[i];
    for (int i = tid; i < H; i += blockDim.x)
        sb1[i] = b1[i];
    for (int i = tid; i < H; i += blockDim.x)
        sW2[i] = W2[i];
    if (tid == 0)
        sb2[0] = b2[0];
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    // Read in native precision, compute in FP32
    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharp[idx]);

    // ── 1. mu EMA update ─────────────────────────────────────────────
    const float mu_old = static_cast<float>(mu[idx]);
    const float mu_new = alpha * mu_old + (1.0f - alpha) * g;
    mu[idx] = static_cast<scalar_t>(mu_new);

    // ── 2. Meta-net inference: Linear(2,H) → GELU → Linear(H,1) ─────
    float mlp_out = 0.0f;

    for (int h = 0; h < H; h++) {
        // Linear(2, H): z = W1[h,0]*g + W1[h,1]*s + b1[h]
        float z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];

        // GELU(z) = z * Φ(z) ≈ z * 0.5 * (1 + tanh(sqrt(2/π) * (z + 0.044715*z³)))
        const float kSqrt2OverPi = 0.7978845608f;
        const float kCoeff = 0.044715f;
        float inner = kSqrt2OverPi * (z + kCoeff * z * z * z);
        float gelu = z * 0.5f * (1.0f + tanhf(inner));

        // Linear(H, 1): accumulate W2[0,h] * gelu
        mlp_out += sW2[h] * gelu;
    }
    mlp_out += sb2[0];

    // ── 3. Skip connection: smart_grad = grad + rescale * mlp_out ────
    smart_grad[idx] = static_cast<scalar_t>(g + rescale * mlp_out);
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: Fused gating blend + Adam moments + progressive wd + step
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void fused_adam_decay_kernel(
    scalar_t* __restrict__ param,             // [N] — updated
    float* __restrict__ exp_avg,              // [N] — FP32 state
    float* __restrict__ exp_avg_sq,           // [N] — FP32 state
    const scalar_t* __restrict__ smart_grad,  // [N]
    const scalar_t* __restrict__ mu,          // [N]
    const float lamb_eff,                     // ramp * gate * lamb
    const float beta1,
    const float beta2,
    const float lr,
    const float wd_eff,                       // progressive weight decay
    const float eps,
    const float bc1,                          // 1 - beta1^step
    const float bc2,                          // 1 - beta2^step
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Read in native precision, compute in FP32
    const float sg = static_cast<float>(smart_grad[idx]);
    const float m = static_cast<float>(mu[idx]);

    // ── Final gradient = smart_grad + lambda * mu ────────────────────
    const float fg = sg + lamb_eff * m;

    // ── Adam moment updates (FP32 state) ─────────────────────────────
    const float ea = beta1 * exp_avg[idx] + (1.0f - beta1) * fg;
    const float easq = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * fg * fg;
    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    // ── Bias-corrected step ──────────────────────────────────────────
    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;

    // ── Progressive weight decay + Adam step (fused) ─────────────────
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: SAM parameter perturbation
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void sam_perturb_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    const float rho_over_norm,   // rho / (global_grad_norm + eps)
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float p = static_cast<float>(param[idx]);
    float g = static_cast<float>(grad[idx]);
    param[idx] = static_cast<scalar_t>(p + rho_over_norm * g);
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 4: Compute sharpness + restore parameters
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void sharpness_restore_kernel(
    scalar_t* __restrict__ param,         // [N] — restored to backup
    scalar_t* __restrict__ sharpness,     // [N] — output
    const scalar_t* __restrict__ backup,  // [N]
    const scalar_t* __restrict__ sam_grad,    // [N]
    const scalar_t* __restrict__ normal_grad, // [N]
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float sg = static_cast<float>(sam_grad[idx]);
    float ng = static_cast<float>(normal_grad[idx]);
    sharpness[idx] = static_cast<scalar_t>(fabsf(sg - ng));
    param[idx] = backup[idx];
}


// ═══════════════════════════════════════════════════════════════════════
//  C++ Dispatch Functions (called from ops.cpp)
// ═══════════════════════════════════════════════════════════════════════

void launch_fused_mu_metanet(
    torch::Tensor mu,
    torch::Tensor grad,
    torch::Tensor sharpness,
    torch::Tensor smart_grad,
    float alpha,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2,
    float rescale,
    int hidden_dim
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Shared memory: (H*2 + H + H + 1) floats — always FP32
    const int smem_bytes = (hidden_dim * 4 + 1) * sizeof(float);

    // Meta-net weights are always FP32
    auto W1_f = W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_mu_metanet", ([&] {
        fused_mu_metanet_kernel<scalar_t><<<grid, BLOCK_SIZE, smem_bytes>>>(
            mu.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            sharpness.data_ptr<scalar_t>(),
            smart_grad.data_ptr<scalar_t>(),
            alpha,
            W1_f.data_ptr<float>(),
            b1_f.data_ptr<float>(),
            W2_f.data_ptr<float>(),
            b2_f.data_ptr<float>(),
            rescale,
            N,
            hidden_dim
        );
    }));
}

void launch_fused_adam_decay(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor smart_grad,
    torch::Tensor mu,
    float lamb_eff,
    float beta1,
    float beta2,
    float lr,
    float wd_eff,
    float eps,
    float bc1,
    float bc2
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_adam_decay", ([&] {
        fused_adam_decay_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            smart_grad.data_ptr<scalar_t>(),
            mu.data_ptr<scalar_t>(),
            lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2, N
        );
    }));
}

void launch_sam_perturb(
    torch::Tensor param,
    torch::Tensor grad,
    float rho_over_norm
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sam_perturb", ([&] {
        sam_perturb_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            rho_over_norm, N
        );
    }));
}

void launch_sharpness_restore(
    torch::Tensor param,
    torch::Tensor sharpness,
    torch::Tensor backup,
    torch::Tensor sam_grad,
    torch::Tensor normal_grad
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sharpness_restore", ([&] {
        sharpness_restore_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            sharpness.data_ptr<scalar_t>(),
            backup.data_ptr<scalar_t>(),
            sam_grad.data_ptr<scalar_t>(),
            normal_grad.data_ptr<scalar_t>(),
            N
        );
    }));
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 5: Fully fused mu_metanet + adam_decay (single pass)
//
//  Eliminates the smart_grad global memory round-trip between kernels 1 & 2.
//  smart_grad stays in registers — never written to / read from GMEM.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void fused_supergrok15_full_step_kernel(
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
    // Shared memory for meta-net weights
    extern __shared__ float smem[];
    float* sW1 = smem;
    float* sb1 = sW1 + H * 2;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

    // Cooperative weight load
    for (int i = tid; i < H * 2; i += blockDim.x) sW1[i] = W1[i];
    for (int i = tid; i < H; i += blockDim.x) sb1[i] = b1[i];
    for (int i = tid; i < H; i += blockDim.x) sW2[i] = W2[i];
    if (tid == 0) sb2[0] = b2[0];
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    // ── 1. Read inputs ──────────────────────────────────────────
    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

    // ── 2. mu EMA ───────────────────────────────────────────────
    const float mu_old = static_cast<float>(mu[idx]);
    const float mu_new = alpha * mu_old + (1.0f - alpha) * g;
    mu[idx] = static_cast<scalar_t>(mu_new);

    // ── 3. Meta-net: Linear(2,H) → GELU → Linear(H,1) ─────────
    float mlp_out = 0.0f;
    for (int h = 0; h < H; h++) {
        float z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];
        const float kSqrt2OverPi = 0.7978845608f;
        const float kCoeff = 0.044715f;
        float inner = kSqrt2OverPi * (z + kCoeff * z * z * z);
        float gelu = z * 0.5f * (1.0f + tanhf(inner));
        mlp_out += sW2[h] * gelu;
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

void launch_fused_supergrok15_full_step(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor mu,
    torch::Tensor grad,
    torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2,
    float rescale,
    float lamb_eff,
    float beta1,
    float beta2,
    float lr,
    float wd_eff,
    float eps,
    float bc1,
    float bc2,
    int hidden_dim
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem_bytes = (hidden_dim * 4 + 1) * sizeof(float);

    auto W1_f = W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_supergrok15_full_step", ([&] {
        fused_supergrok15_full_step_kernel<scalar_t><<<grid, BLOCK_SIZE, smem_bytes>>>(
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
}
