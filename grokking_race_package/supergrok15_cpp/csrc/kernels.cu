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
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    const scalar_t alpha,                // EMA momentum for mu
    const scalar_t* __restrict__ W1,     // [H, 2] — row-major
    const scalar_t* __restrict__ b1,     // [H]
    const scalar_t* __restrict__ W2,     // [1, H] — row-major
    const scalar_t* __restrict__ b2,     // [1]
    const scalar_t rescale,
    const int N,
    const int H
) {
    // Load meta-net weights into shared memory
    extern __shared__ char smem_raw[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* sW1 = smem;              // H * 2
    scalar_t* sb1 = sW1 + H * 2;      // H
    scalar_t* sW2 = sb1 + H;          // H
    scalar_t* sb2 = sW2 + H;          // 1

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

    const scalar_t g = grad[idx];
    const scalar_t s = sharp[idx];

    // ── 1. mu EMA update ─────────────────────────────────────────────
    const scalar_t mu_new = alpha * mu[idx] + (static_cast<scalar_t>(1) - alpha) * g;
    mu[idx] = mu_new;

    // ── 2. Meta-net inference: Linear(2,H) → GELU → Linear(H,1) ─────
    scalar_t mlp_out = static_cast<scalar_t>(0);

    for (int h = 0; h < H; h++) {
        // Linear(2, H): z = W1[h,0]*g + W1[h,1]*s + b1[h]
        scalar_t z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];

        // GELU(z) = z * Φ(z) ≈ z * 0.5 * (1 + tanh(sqrt(2/π) * (z + 0.044715*z³)))
        const scalar_t kSqrt2OverPi = static_cast<scalar_t>(0.7978845608);
        const scalar_t kCoeff = static_cast<scalar_t>(0.044715);
        scalar_t inner = kSqrt2OverPi * (z + kCoeff * z * z * z);
        scalar_t gelu = z * static_cast<scalar_t>(0.5) * (static_cast<scalar_t>(1) + tanhf(inner));

        // Linear(H, 1): accumulate W2[0,h] * gelu
        mlp_out += sW2[h] * gelu;
    }
    mlp_out += sb2[0];

    // ── 3. Skip connection: smart_grad = grad + rescale * mlp_out ────
    smart_grad[idx] = g + rescale * mlp_out;
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: Fused gating blend + Adam moments + progressive wd + step
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void fused_adam_decay_kernel(
    scalar_t* __restrict__ param,             // [N] — updated
    scalar_t* __restrict__ exp_avg,           // [N] — updated
    scalar_t* __restrict__ exp_avg_sq,        // [N] — updated
    const scalar_t* __restrict__ smart_grad,  // [N]
    const scalar_t* __restrict__ mu,          // [N]
    const scalar_t lamb_eff,                  // ramp * gate * lamb
    const scalar_t beta1,
    const scalar_t beta2,
    const scalar_t lr,
    const scalar_t wd_eff,                    // progressive weight decay
    const scalar_t eps,
    const scalar_t bc1,                       // 1 - beta1^step
    const scalar_t bc2,                       // 1 - beta2^step
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ── Final gradient = smart_grad + lambda * mu ────────────────────
    const scalar_t fg = smart_grad[idx] + lamb_eff * mu[idx];

    // ── Adam moment updates ──────────────────────────────────────────
    const scalar_t ea = beta1 * exp_avg[idx] + (static_cast<scalar_t>(1) - beta1) * fg;
    const scalar_t easq = beta2 * exp_avg_sq[idx]
                        + (static_cast<scalar_t>(1) - beta2) * fg * fg;
    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    // ── Bias-corrected step ──────────────────────────────────────────
    const scalar_t step_size = lr / bc1;
    const scalar_t denom = sqrtf(easq / bc2) + eps;

    // ── Progressive weight decay + Adam step (fused) ─────────────────
    scalar_t p = param[idx];
    p *= (static_cast<scalar_t>(1) - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = p;
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: SAM parameter perturbation
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void sam_perturb_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    const scalar_t rho_over_norm,   // rho / (global_grad_norm + eps)
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    param[idx] += rho_over_norm * grad[idx];
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
    sharpness[idx] = fabsf(sam_grad[idx] - normal_grad[idx]);
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
    // Shared memory: (H*2 + H + H + 1) floats
    const int smem_bytes = (hidden_dim * 3 + 1) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "fused_mu_metanet", ([&] {
        fused_mu_metanet_kernel<scalar_t><<<grid, BLOCK_SIZE, smem_bytes>>>(
            mu.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            sharpness.data_ptr<scalar_t>(),
            smart_grad.data_ptr<scalar_t>(),
            static_cast<scalar_t>(alpha),
            W1.data_ptr<scalar_t>(),
            b1.data_ptr<scalar_t>(),
            W2.data_ptr<scalar_t>(),
            b2.data_ptr<scalar_t>(),
            static_cast<scalar_t>(rescale),
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

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "fused_adam_decay", ([&] {
        fused_adam_decay_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            exp_avg.data_ptr<scalar_t>(),
            exp_avg_sq.data_ptr<scalar_t>(),
            smart_grad.data_ptr<scalar_t>(),
            mu.data_ptr<scalar_t>(),
            static_cast<scalar_t>(lamb_eff),
            static_cast<scalar_t>(beta1),
            static_cast<scalar_t>(beta2),
            static_cast<scalar_t>(lr),
            static_cast<scalar_t>(wd_eff),
            static_cast<scalar_t>(eps),
            static_cast<scalar_t>(bc1),
            static_cast<scalar_t>(bc2),
            N
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

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "sam_perturb", ([&] {
        sam_perturb_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            static_cast<scalar_t>(rho_over_norm),
            N
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

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "sharpness_restore", ([&] {
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
