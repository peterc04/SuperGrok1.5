/*
 * GrokAdamW -- Fused CUDA Kernel
 *
 * Adam with grokking-aware gradient filtering and amplification.
 * Fuses the entire optimizer step into one kernel per parameter:
 *
 *   1. EMA gradient filter:
 *        filtered = alpha * ema + (1 - alpha) * grad
 *        ema = filtered
 *
 *   2. Gradient amplification:
 *        amplified = grad + lamb * filtered
 *
 *   3. Adam update with amplified gradient:
 *        exp_avg    = beta1 * exp_avg    + (1 - beta1) * amplified
 *        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * amplified^2
 *        bc1 = 1 - beta1^step,  bc2 = 1 - beta2^step
 *        step_size = lr / bc1
 *        denom = sqrt(exp_avg_sq / bc2) + eps
 *        param = param * (1 - lr * weight_decay) - step_size * exp_avg / denom
 *
 * Gradient clipping is applied before the kernel launch; the grad tensor
 * passed here is already clipped.
 *
 * One thread per element, fully coalesced memory access.
 * Uses float accumulation internally for sqrt and bias correction
 * to preserve numerical precision with FP16/BF16 inputs.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int GROKADAMW_BLOCK_SIZE = 256;

// ===================================================================
//  Kernel: Fused GrokAdamW step
//  (EMA filter + amplify + Adam moments + decoupled weight decay)
// ===================================================================

template <typename scalar_t>
__global__ void fused_grokadamw_step_kernel(
    scalar_t* __restrict__ param,          // [N] -- updated in-place
    float* __restrict__ exp_avg,           // [N] -- first moment, updated
    float* __restrict__ exp_avg_sq,        // [N] -- second moment, updated
    float* __restrict__ ema,               // [N] -- EMA gradient filter, updated
    const scalar_t* __restrict__ grad,     // [N] -- pre-clipped gradient
    const float alpha,                     // EMA decay for gradient filter
    const float lamb,                      // Amplification factor
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,                       // 1 - beta1^step (precomputed)
    const float bc2,                       // 1 - beta2^step (precomputed)
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Read inputs -- cast to float for accumulation precision
    const float g = static_cast<float>(grad[idx]);
    const float e = ema[idx];

    // -- 1. EMA gradient filter -----------------------------------------
    const float filtered = alpha * e + (1.0f - alpha) * g;
    ema[idx] = filtered;

    // -- 2. Gradient amplification --------------------------------------
    const float amplified = g + lamb * filtered;

    // -- 3. Adam moment updates -----------------------------------------
    const float ea_old = exp_avg[idx];
    const float easq_old = exp_avg_sq[idx];

    const float ea = beta1 * ea_old + (1.0f - beta1) * amplified;
    const float easq = beta2 * easq_old + (1.0f - beta2) * amplified * amplified;

    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    // -- 4. Bias-corrected Adam step with decoupled weight decay --------
    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;

    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}


// ===================================================================
//  C++ Dispatch Function (called from ops.cpp / pybind)
// ===================================================================

void launch_fused_grokadamw_step(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor ema,
    torch::Tensor grad,
    float alpha,
    float lamb,
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
    const int grid = (N + GROKADAMW_BLOCK_SIZE - 1) / GROKADAMW_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_grokadamw_step", ([&] {
            fused_grokadamw_step_kernel<scalar_t><<<grid, GROKADAMW_BLOCK_SIZE>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                ema.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                alpha,
                lamb,
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
