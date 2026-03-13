/*
 * NeuralGrok -- Fused CUDA Kernels
 *
 * Adam optimizer with a learned MLP gradient amplifier.
 * Two kernels:
 *
 *   1. fused_neuralgrok_amplifier_kernel:
 *      Per-element MLP inference that modulates each gradient element.
 *      MLP architecture:  Linear(1, H) -> ReLU -> Linear(H, 1) -> scale
 *      Output: amplified_grad = grad * (alpha * scale + beta)
 *
 *      MLP weights are loaded into shared memory cooperatively by the
 *      block, then each thread independently evaluates the tiny MLP for
 *      its own gradient element. With H=32 this is 64 multiply-adds per
 *      thread -- trivial compute but fully parallel across all elements.
 *
 *   2. fused_neuralgrok_adam_kernel:
 *      Standard Adam update using the amplified gradients:
 *        exp_avg    = beta1 * exp_avg    + (1 - beta1) * amplified_grad
 *        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * amplified_grad^2
 *        bc1 = 1 - beta1^step,  bc2 = 1 - beta2^step
 *        step_size = lr / bc1
 *        denom = sqrt(exp_avg_sq / bc2) + eps
 *        param = param * (1 - lr * wd) - step_size * exp_avg / denom
 *
 * Uses float accumulation internally for MLP inference, sqrt, and bias
 * correction to preserve numerical precision with FP16/BF16 inputs.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int NEURALGROK_BLOCK_SIZE = 256;

// ===================================================================
//  Kernel 1: MLP gradient amplifier
//  Input:  grad (1 value per element)
//  MLP:    Linear(1, H) -> ReLU -> Linear(H, 1)
//  Output: amplified = grad * (alpha * mlp_out + beta)
// ===================================================================

template <typename scalar_t>
__global__ void fused_neuralgrok_amplifier_kernel(
    const scalar_t* __restrict__ grad,           // [N]
    scalar_t* __restrict__ amplified_grad,       // [N] -- output
    const float* __restrict__ W1,                // [H, 1] -- row-major
    const float* __restrict__ b1,                // [H]
    const float* __restrict__ W2,                // [1, H] -- row-major
    const float* __restrict__ b2,                // [1]
    const float alpha,                           // Scale factor for MLP output
    const float beta,                            // Bias term (skip connection strength)
    const int N,
    const int H
) {
    // Load MLP weights into shared memory
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    float* sW1 = smem;              // H elements
    float* sb1 = sW1 + H;          // H elements
    float* sW2 = sb1 + H;          // H elements
    float* sb2 = sW2 + H;          // 1 element

    const int tid = threadIdx.x;

    // Cooperative load of weights into shared memory
    for (int i = tid; i < H; i += blockDim.x)
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

    // Read gradient and cast to float for accumulation precision
    const float g = static_cast<float>(grad[idx]);

    // -- MLP inference: Linear(1, H) -> ReLU -> Linear(H, 1) -----------
    float mlp_out = 0.0f;

    for (int h = 0; h < H; h++) {
        // Linear(1, H): z = W1[h] * g + b1[h]
        float z = sW1[h] * g + sb1[h];

        // ReLU activation
        z = (z > 0.0f) ? z : 0.0f;

        // Linear(H, 1): accumulate W2[h] * relu_out
        mlp_out += sW2[h] * z;
    }
    mlp_out += sb2[0];

    // -- Amplification: grad * (alpha * mlp_out + beta) -----------------
    const float amp = g * (alpha * mlp_out + beta);
    amplified_grad[idx] = static_cast<scalar_t>(amp);
}


// ===================================================================
//  Kernel 2: Standard Adam update with amplified gradients
//  (no mu/lambda terms -- pure AdamW with decoupled weight decay)
// ===================================================================

template <typename scalar_t>
__global__ void fused_neuralgrok_adam_kernel(
    scalar_t* __restrict__ param,             // [N] -- updated
    float* __restrict__ exp_avg,              // [N] -- updated
    float* __restrict__ exp_avg_sq,           // [N] -- updated
    const scalar_t* __restrict__ amplified_grad,  // [N]
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,                          // 1 - beta1^step
    const float bc2,                          // 1 - beta2^step
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Cast to float for accumulation precision
    const float ag = static_cast<float>(amplified_grad[idx]);
    const float ea_old = exp_avg[idx];
    const float easq_old = exp_avg_sq[idx];

    // -- Adam moment updates --------------------------------------------
    const float ea = beta1 * ea_old + (1.0f - beta1) * ag;
    const float easq = beta2 * easq_old + (1.0f - beta2) * ag * ag;

    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    // -- Bias-corrected step with decoupled weight decay ----------------
    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;

    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}


// ===================================================================
//  C++ Dispatch Functions
// ===================================================================

void launch_fused_neuralgrok_amplifier(
    torch::Tensor grad,
    torch::Tensor amplified_grad,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2,
    float alpha,
    float beta,
    int hidden_dim
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + NEURALGROK_BLOCK_SIZE - 1) / NEURALGROK_BLOCK_SIZE;
    // Shared memory: (H + H + H + 1) elements
    const int smem_elems = hidden_dim * 3 + 1;
    const int smem_bytes = smem_elems * sizeof(float);

    // Convert weights to FP32 (Python provides FP32 weights, dispatch is on grad dtype)
    auto W1_f = W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_neuralgrok_amplifier", ([&] {
            fused_neuralgrok_amplifier_kernel<scalar_t><<<grid, NEURALGROK_BLOCK_SIZE, smem_bytes>>>(
                grad.data_ptr<scalar_t>(),
                amplified_grad.data_ptr<scalar_t>(),
                W1_f.data_ptr<float>(),
                b1_f.data_ptr<float>(),
                W2_f.data_ptr<float>(),
                b2_f.data_ptr<float>(),
                alpha,
                beta,
                N,
                hidden_dim
            );
        })
    );
}

void launch_fused_neuralgrok_adam(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor amplified_grad,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float bc1,
    float bc2
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + NEURALGROK_BLOCK_SIZE - 1) / NEURALGROK_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_neuralgrok_adam", ([&] {
            fused_neuralgrok_adam_kernel<scalar_t><<<grid, NEURALGROK_BLOCK_SIZE>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                amplified_grad.data_ptr<scalar_t>(),
                beta1,
                beta2,
                lr,
                weight_decay,
                eps,
                bc1,
                bc2,
                N
            );
        })
    );
}
