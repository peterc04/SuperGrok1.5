/*
 * SuperGrok v1.1 — CUDA Kernels (Cosine Similarity Gating)
 *
 * Four fused kernels plus a C++ cosine-gate utility:
 *
 *   1. fused_sg11_mu_metanet  — EMA update + element-wise MLP inference
 *   2. fused_sg11_adam_cosine_gate — Adam moments + progressive wd + step
 *        (cosine similarity gating is computed host-side via ATen ops
 *         and passed as the pre-computed lamb_eff scalar)
 *   3. sg11_sam_perturb       — worst-case parameter perturbation
 *   4. sg11_sharpness_restore — |sam_grad − grad| + param restore
 *
 * Key difference from SuperGrok v1.5:
 *   v1.5 uses a sigmoid gate from a pre-computed gate_signal scalar.
 *   v1.1 computes per-parameter cosine similarity between smart_grad
 *   and mu, applies sigmoid to get the gate, then derives lamb_eff.
 *   The cosine similarity reduction is done via ATen (cuBLAS) before
 *   the Adam kernel launch, so the kernel itself receives lamb_eff
 *   as a scalar — identical structure to v1.5's fused_adam_decay_kernel.
 *
 * All kernels support FP32, FP64, FP16, and BF16 via
 * AT_DISPATCH_FLOATING_TYPES_AND2.
 */

#include <torch/extension.h>
#include <cmath>

#include "platform.h"
#include "utils.cuh"

constexpr int BLOCK_SIZE = 256;

// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: Fused mu EMA update + meta-net inference
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 4)
__global__ void fused_sg11_mu_metanet_kernel(
    scalar_t* __restrict__ mu,           // [N] — updated in-place
    const scalar_t* __restrict__ grad,   // [N]
    const scalar_t* __restrict__ sharp,  // [N]
    scalar_t* __restrict__ smart_grad,   // [N] — output
    const float alpha,                // EMA momentum for mu
    const float* __restrict__ W1,     // [H, 2] — row-major
    const float* __restrict__ b1,     // [H]
    const float* __restrict__ W2,     // [1, H] — row-major
    const float* __restrict__ b2,     // [1]
    const float rescale,
    const int N,
    const int H
) {
    // Load meta-net weights into shared memory
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
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

        // Fast GELU: sigmoid approximation (~2.5x faster, max error ~0.004)
        float gelu = z / (1.0f + expf(-1.702f * z));

        // Linear(H, 1): accumulate W2[0,h] * gelu
        mlp_out += sW2[h] * gelu;
    }
    mlp_out += sb2[0];

    // ── 3. Skip connection: smart_grad = grad + rescale * mlp_out ────
    smart_grad[idx] = static_cast<scalar_t>(g + rescale * mlp_out);
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: Fused Adam moments + progressive wd + step
//            (cosine similarity gating pre-computed as lamb_eff)
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void fused_sg11_adam_cosine_gate_kernel(
    scalar_t* __restrict__ param,             // [N] — updated
    float* __restrict__ exp_avg,           // [N] — updated
    float* __restrict__ exp_avg_sq,        // [N] — updated
    const scalar_t* __restrict__ smart_grad,  // [N]
    const scalar_t* __restrict__ mu,          // [N]
    const float lamb_eff,                  // ramp * gate * lamb (pre-computed)
    const float beta1,
    const float beta2,
    const float lr,
    const float wd_eff,                    // progressive weight decay
    const float eps,
    const float bc1,                       // 1 - beta1^step
    const float bc2,                       // 1 - beta2^step
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ── Final gradient = smart_grad + lambda * mu ────────────────────
    const float sg = static_cast<float>(smart_grad[idx]);
    const float m = static_cast<float>(mu[idx]);
    const float fg = sg + lamb_eff * m;

    // ── Adam moment updates (non-temporal to preserve L2) ─────────────
    const float ea = beta1 * stream_load(&exp_avg[idx]) + (1.0f - beta1) * fg;
    const float easq = beta2 * stream_load(&exp_avg_sq[idx])
                        + (1.0f - beta2) * fg * fg;
    stream_store(&exp_avg[idx], ea);
    stream_store(&exp_avg_sq[idx], easq);

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
__launch_bounds__(256, 8)
__global__ void sg11_sam_perturb_kernel(
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
__launch_bounds__(256, 8)
__global__ void sg11_sharpness_restore_kernel(
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
//  Vec4 Kernel 2b: Fused Adam cosine gate (float4 vectorized, FP32-only)
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void fused_sg11_adam_cosine_gate_vec4_kernel(
    float4* __restrict__ param4,
    float4* __restrict__ exp_avg4,
    float4* __restrict__ exp_avg_sq4,
    const float4* __restrict__ smart_grad4,
    const float4* __restrict__ mu4,
    float lamb_eff, float beta1, float beta2, float lr,
    float wd_eff, float eps, float bc1, float bc2, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 sg = smart_grad4[i];
    float4 m = mu4[i];

    // Final gradient = smart_grad + lambda * mu
    float4 fg;
    fg.x = sg.x + lamb_eff * m.x;
    fg.y = sg.y + lamb_eff * m.y;
    fg.z = sg.z + lamb_eff * m.z;
    fg.w = sg.w + lamb_eff * m.w;

    // Adam moment updates (non-temporal to preserve L2)
    float4 ea = stream_load4(&exp_avg4[i]);
    float4 eas = stream_load4(&exp_avg_sq4[i]);

    ea.x = beta1 * ea.x + (1.0f - beta1) * fg.x;
    ea.y = beta1 * ea.y + (1.0f - beta1) * fg.y;
    ea.z = beta1 * ea.z + (1.0f - beta1) * fg.z;
    ea.w = beta1 * ea.w + (1.0f - beta1) * fg.w;

    eas.x = beta2 * eas.x + (1.0f - beta2) * fg.x * fg.x;
    eas.y = beta2 * eas.y + (1.0f - beta2) * fg.y * fg.y;
    eas.z = beta2 * eas.z + (1.0f - beta2) * fg.z * fg.z;
    eas.w = beta2 * eas.w + (1.0f - beta2) * fg.w * fg.w;

    stream_store4(&exp_avg4[i], ea);
    stream_store4(&exp_avg_sq4[i], eas);

    // Bias-corrected step + progressive weight decay
    float step_size = lr / bc1;
    float decay = 1.0f - lr * wd_eff;
    float4 p = param4[i];
    p.x = decay * p.x - step_size * ea.x / (sqrtf(eas.x / bc2) + eps);
    p.y = decay * p.y - step_size * ea.y / (sqrtf(eas.y / bc2) + eps);
    p.z = decay * p.z - step_size * ea.z / (sqrtf(eas.z / bc2) + eps);
    p.w = decay * p.w - step_size * ea.w / (sqrtf(eas.w / bc2) + eps);
    param4[i] = p;
}


// ═══════════════════════════════════════════════════════════════════════
//  Vec4 Kernel 3b: SAM perturbation (float4 vectorized, FP32-only)
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void sg11_sam_perturb_vec4_kernel(
    float4* __restrict__ param4,
    const float4* __restrict__ grad4,
    float rho_over_norm,
    int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 p = param4[i];
    float4 g = grad4[i];
    p.x += rho_over_norm * g.x;
    p.y += rho_over_norm * g.y;
    p.z += rho_over_norm * g.z;
    p.w += rho_over_norm * g.w;
    param4[i] = p;
}


// ═══════════════════════════════════════════════════════════════════════
//  Vec4 Kernel 4b: Sharpness restore (float4 vectorized, FP32-only)
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void sg11_sharpness_restore_vec4_kernel(
    float4* __restrict__ param4,
    float4* __restrict__ sharpness4,
    const float4* __restrict__ backup4,
    const float4* __restrict__ sam_grad4,
    const float4* __restrict__ normal_grad4,
    int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 sg = sam_grad4[i];
    float4 ng = normal_grad4[i];
    float4 s;
    s.x = fabsf(sg.x - ng.x);
    s.y = fabsf(sg.y - ng.y);
    s.z = fabsf(sg.z - ng.z);
    s.w = fabsf(sg.w - ng.w);
    sharpness4[i] = s;
    param4[i] = backup4[i];
}


// ═══════════════════════════════════════════════════════════════════════
//  Cosine Similarity Gate — C++ utility (ATen / cuBLAS reduction)
// ═══════════════════════════════════════════════════════════════════════

float compute_cosine_gate(
    torch::Tensor smart_grad,
    torch::Tensor mu,
    float gate_temp
) {
    // Use ATen ops for the global reduction (cuBLAS under the hood)
    auto dot = (smart_grad.flatten() * mu.flatten()).sum().item<float>();
    auto sg_norm = smart_grad.norm().item<float>();
    auto mu_norm = mu.norm().item<float>();
    float cos_sim = dot / (sg_norm * mu_norm + 1e-8f);
    float gate = 1.0f / (1.0f + std::exp(-gate_temp * cos_sim));
    return gate;
}


// ═══════════════════════════════════════════════════════════════════════
//  Fused Cosine-Gate Reduction Kernel
//  Computes [dot(sg,mu), ||sg||², ||mu||²] in a single kernel launch
//  with warp shuffle reduction — only ONE CPU sync needed.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void cosine_gate_reduce_kernel(
    const float* __restrict__ smart_grad,  // [N]
    const float* __restrict__ mu,          // [N]
    float* __restrict__ results,           // [3] output: {dot, sg_norm_sq, mu_norm_sq}
    const int N
) {
    float dot_acc = 0.0f, sg_sq_acc = 0.0f, mu_sq_acc = 0.0f;

    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float sg = smart_grad[i];
        float m = mu[i];
        dot_acc += sg * m;
        sg_sq_acc += sg * sg;
        mu_sq_acc += m * m;
    }

    // Warp shuffle reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        dot_acc += SHFL_DOWN(dot_acc, offset);
        sg_sq_acc += SHFL_DOWN(sg_sq_acc, offset);
        mu_sq_acc += SHFL_DOWN(mu_sq_acc, offset);
    }

    // Lane 0 of each warp atomicAdds to global
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        atomicAdd(&results[0], dot_acc);
        atomicAdd(&results[1], sg_sq_acc);
        atomicAdd(&results[2], mu_sq_acc);
    }
}

float compute_cosine_gate_fused(
    torch::Tensor smart_grad,
    torch::Tensor mu,
    float gate_temp
) {
    auto sg_f = smart_grad.to(torch::kFloat32).flatten();
    auto mu_f = mu.to(torch::kFloat32).flatten();
    int N = sg_f.numel();
    auto results = torch::zeros({3}, torch::TensorOptions().device(sg_f.device()).dtype(torch::kFloat32));

    int block = 256;
    int grid = std::min((N + block - 1) / block, 1024);
    cosine_gate_reduce_kernel<<<grid, block>>>(
        sg_f.data_ptr<float>(), mu_f.data_ptr<float>(),
        results.data_ptr<float>(), N);

    // Single CPU sync to read all 3 values
    auto r = results.cpu();
    float dot = r[0].item<float>();
    float sg_norm = std::sqrt(r[1].item<float>());
    float mu_norm = std::sqrt(r[2].item<float>());
    float cos_sim = dot / (sg_norm * mu_norm + 1e-8f);
    return 1.0f / (1.0f + std::exp(-gate_temp * cos_sim));
}


// ═══════════════════════════════════════════════════════════════════════
//  C++ Dispatch Functions (called from Python / ops.cpp)
// ═══════════════════════════════════════════════════════════════════════

void launch_sg11_mu_metanet(
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
    // Shared memory: (H*2 + H + H + 1) elements
    const int smem_elems = hidden_dim * 4 + 1;
    const int smem_bytes = smem_elems * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_sg11_mu_metanet", ([&] {
        fused_sg11_mu_metanet_kernel<scalar_t><<<grid, BLOCK_SIZE, smem_bytes>>>(
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

void launch_sg11_adam_decay(
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

    // Float4 fast path: FP32, N divisible by 4, 16-byte aligned
    if (param.scalar_type() == at::ScalarType::Float &&
        (N % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(param.data_ptr<float>()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(exp_avg.data_ptr<float>()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(exp_avg_sq.data_ptr<float>()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(smart_grad.data_ptr<float>()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(mu.data_ptr<float>()) % 16 == 0))
    {
        const int N4 = N / 4;
        const int grid = (N4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fused_sg11_adam_cosine_gate_vec4_kernel<<<grid, BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            reinterpret_cast<const float4*>(smart_grad.data_ptr<float>()),
            reinterpret_cast<const float4*>(mu.data_ptr<float>()),
            lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2, N4
        );
        return;
    }

    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_sg11_adam_cosine_gate", ([&] {
        fused_sg11_adam_cosine_gate_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            smart_grad.data_ptr<scalar_t>(),
            mu.data_ptr<scalar_t>(),
            lamb_eff,
            beta1,
            beta2,
            lr,
            wd_eff,
            eps,
            bc1,
            bc2,
            N
        );
    }));
}

void launch_sg11_sam_perturb(
    torch::Tensor param,
    torch::Tensor grad,
    float rho_over_norm
) {
    const int N = param.numel();
    if (N == 0) return;

    // Float4 fast path: FP32, N divisible by 4, 16-byte aligned
    if (param.scalar_type() == at::ScalarType::Float &&
        (N % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(param.data_ptr<float>()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(grad.data_ptr<float>()) % 16 == 0))
    {
        const int N4 = N / 4;
        const int grid = (N4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sg11_sam_perturb_vec4_kernel<<<grid, BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            rho_over_norm, N4
        );
        return;
    }

    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg11_sam_perturb", ([&] {
        sg11_sam_perturb_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            rho_over_norm,
            N
        );
    }));
}

void launch_sg11_sharpness_restore(
    torch::Tensor param,
    torch::Tensor sharpness,
    torch::Tensor backup,
    torch::Tensor sam_grad,
    torch::Tensor normal_grad
) {
    const int N = param.numel();
    if (N == 0) return;

    // Float4 fast path: FP32, N divisible by 4, 16-byte aligned
    if (param.scalar_type() == at::ScalarType::Float &&
        (N % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(param.data_ptr<float>()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(sharpness.data_ptr<float>()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(backup.data_ptr<float>()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(sam_grad.data_ptr<float>()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(normal_grad.data_ptr<float>()) % 16 == 0))
    {
        const int N4 = N / 4;
        const int grid = (N4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sg11_sharpness_restore_vec4_kernel<<<grid, BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(sharpness.data_ptr<float>()),
            reinterpret_cast<const float4*>(backup.data_ptr<float>()),
            reinterpret_cast<const float4*>(sam_grad.data_ptr<float>()),
            reinterpret_cast<const float4*>(normal_grad.data_ptr<float>()),
            N4
        );
        return;
    }

    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg11_sharpness_restore", ([&] {
        sg11_sharpness_restore_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
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
//  Kernel 5: Fused full step — mu EMA + meta-net + Adam (single pass)
//
//  Eliminates the smart_grad global memory round-trip between kernels 1 & 2.
//  smart_grad stays in registers — never written to / read from GMEM.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 4)
__global__ void fused_sg11_full_step_kernel(
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
        // Fast GELU: sigmoid approximation (~2.5x faster, max error ~0.004)
        float gelu = z / (1.0f + expf(-1.702f * z));
        mlp_out += sW2[h] * gelu;
    }
    mlp_out += sb2[0];

    // ── 4. smart_grad (REGISTER ONLY — no global write) ────────
    const float smart_grad = g + rescale * mlp_out;

    // ── 5. Final gradient with gating ───────────────────────────
    const float fg = smart_grad + lamb_eff * mu_new;

    // ── 6. Adam moments (non-temporal to preserve L2) ──────────
    const float ea = beta1 * stream_load(&exp_avg[idx]) + (1.0f - beta1) * fg;
    const float easq = beta2 * stream_load(&exp_avg_sq[idx]) + (1.0f - beta2) * fg * fg;
    stream_store(&exp_avg[idx], ea);
    stream_store(&exp_avg_sq[idx], easq);

    // ── 7. Bias-corrected step + progressive WD ─────────────────
    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}

void launch_fused_sg11_full_step(
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

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_sg11_full_step", ([&] {
        fused_sg11_full_step_kernel<scalar_t><<<grid, BLOCK_SIZE, smem_bytes>>>(
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
