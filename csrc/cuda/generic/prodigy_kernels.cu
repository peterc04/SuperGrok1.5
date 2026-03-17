/*
 * Prodigy -- Fused CUDA Kernels
 *
 * Prodigy is a distance-aware, self-tuning Adam variant that automatically
 * adapts its learning rate based on the distance the parameters have moved
 * from initialization.
 *
 * Two kernels:
 *
 *   1. fused_prodigy_step_kernel:
 *      Per-element update combining EMA of squared gradients, Adam moments,
 *      bias correction, and the parameter step:
 *
 *        s[i]       = beta2 * s[i] + (1-beta2) * d_lr^2 * grad[i]^2
 *        exp_avg[i] = beta1 * exp_avg[i] + (1-beta1) * d_lr * grad[i]
 *        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1-beta2) * d_lr^2 * grad[i]^2
 *        bc1 = 1 - beta1^step,  bc2 = 1 - beta2^step
 *        denom = sqrt(exp_avg_sq[i] / bc2) + d_lr * eps
 *        param[i] *= (1 - lr * d_lr * weight_decay)
 *        param[i] -= lr * exp_avg[i] / (bc1 * denom)
 *
 *      d_lr (the adaptive learning rate scalar) is computed in C++ from
 *      the reduction output and passed as a scalar argument.
 *
 *   2. prodigy_dlr_reduce_kernel:
 *      Block-level reduction for computing the adaptive learning rate d.
 *      Per element, accumulates two values:
 *        numerator   += grad[i] * (param[i] - param_init[i])
 *        denominator += s[i]
 *      Uses warp-shuffle reduction within each warp, then shared memory
 *      reduction across warps, and finally atomicAdd to global accumulators.
 *
 * Uses float accumulation internally for all reductions, sqrt, and bias
 * correction to preserve numerical precision with FP16/BF16 inputs.
 */

#include <torch/extension.h>

#include "platform.h"
#include "utils.cuh"

constexpr int PRODIGY_BLOCK_SIZE = 256;
constexpr int PRODIGY_WARP_SIZE = WARP_SIZE;

// ===================================================================
//  Kernel 1: Fused Prodigy per-element step
//  (s update + Adam moments + bias-corrected step + weight decay)
// ===================================================================

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void fused_prodigy_step_kernel(
    scalar_t* __restrict__ param,          // [N] -- updated in-place
    float* __restrict__ exp_avg,           // [N] -- first moment, updated
    float* __restrict__ exp_avg_sq,        // [N] -- second moment, updated
    float* __restrict__ s,                 // [N] -- EMA of d_lr-scaled sq grads, updated
    const scalar_t* __restrict__ grad,     // [N]
    const float d_lr,                      // Adaptive learning rate (precomputed in C++)
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

    // Cast to float for accumulation precision
    const float g = static_cast<float>(grad[idx]);
    const float d_lr_g = d_lr * g;
    const float d_lr_g_sq = d_lr * d_lr * g * g;

    // -- 1. Update s (EMA of d_lr-scaled squared gradients) -------------
    const float s_old = stream_load(&s[idx]);
    const float s_new = beta2 * s_old + (1.0f - beta2) * d_lr_g_sq;
    stream_store(&s[idx], s_new);

    // -- 2. Adam moment updates -----------------------------------------
    const float ea_old = stream_load(&exp_avg[idx]);
    const float easq_old = stream_load(&exp_avg_sq[idx]);

    const float ea = beta1 * ea_old + (1.0f - beta1) * d_lr_g;
    const float easq = beta2 * easq_old + (1.0f - beta2) * d_lr_g_sq;

    stream_store(&exp_avg[idx], ea);
    stream_store(&exp_avg_sq[idx], easq);

    // -- 3. Bias-corrected step with weight decay -----------------------
    const float denom = sqrtf(easq / bc2) + d_lr * eps;

    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * d_lr * weight_decay);
    p -= lr * ea / (bc1 * denom);
    param[idx] = static_cast<scalar_t>(p);
}


// ===================================================================
//  Kernel 1b: Fused Prodigy per-element step — float4 vectorized (FP32 only)
// ===================================================================

__launch_bounds__(256, 8)
__global__ void fused_prodigy_step_vec4_kernel(
    float4* __restrict__ param4,
    float4* __restrict__ exp_avg4,
    float4* __restrict__ exp_avg_sq4,
    float4* __restrict__ s4,
    const float4* __restrict__ grad4,
    float d_lr, float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2, int N4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    float4 g = grad4[i];
    float4 p = param4[i];
    float4 ea = stream_load4(&exp_avg4[i]);
    float4 eas = stream_load4(&exp_avg_sq4[i]);
    float4 sv = stream_load4(&s4[i]);

    // Precompute scaled gradient components
    float4 d_lr_g, d_lr_g_sq;
    d_lr_g.x = d_lr * g.x;  d_lr_g_sq.x = d_lr * d_lr * g.x * g.x;
    d_lr_g.y = d_lr * g.y;  d_lr_g_sq.y = d_lr * d_lr * g.y * g.y;
    d_lr_g.z = d_lr * g.z;  d_lr_g_sq.z = d_lr * d_lr * g.z * g.z;
    d_lr_g.w = d_lr * g.w;  d_lr_g_sq.w = d_lr * d_lr * g.w * g.w;

    // Update s
    sv.x = beta2 * sv.x + (1.0f - beta2) * d_lr_g_sq.x;
    sv.y = beta2 * sv.y + (1.0f - beta2) * d_lr_g_sq.y;
    sv.z = beta2 * sv.z + (1.0f - beta2) * d_lr_g_sq.z;
    sv.w = beta2 * sv.w + (1.0f - beta2) * d_lr_g_sq.w;
    stream_store4(&s4[i], sv);

    // Adam moment updates
    ea.x = beta1 * ea.x + (1.0f - beta1) * d_lr_g.x;
    ea.y = beta1 * ea.y + (1.0f - beta1) * d_lr_g.y;
    ea.z = beta1 * ea.z + (1.0f - beta1) * d_lr_g.z;
    ea.w = beta1 * ea.w + (1.0f - beta1) * d_lr_g.w;

    eas.x = beta2 * eas.x + (1.0f - beta2) * d_lr_g_sq.x;
    eas.y = beta2 * eas.y + (1.0f - beta2) * d_lr_g_sq.y;
    eas.z = beta2 * eas.z + (1.0f - beta2) * d_lr_g_sq.z;
    eas.w = beta2 * eas.w + (1.0f - beta2) * d_lr_g_sq.w;

    stream_store4(&exp_avg4[i], ea);
    stream_store4(&exp_avg_sq4[i], eas);

    // Bias-corrected step with weight decay
    float d_lr_eps = d_lr * eps;
    float decay = 1.0f - lr * d_lr * weight_decay;

    p.x = decay * p.x - lr * ea.x / (bc1 * (sqrtf(eas.x / bc2) + d_lr_eps));
    p.y = decay * p.y - lr * ea.y / (bc1 * (sqrtf(eas.y / bc2) + d_lr_eps));
    p.z = decay * p.z - lr * ea.z / (bc1 * (sqrtf(eas.z / bc2) + d_lr_eps));
    p.w = decay * p.w - lr * ea.w / (bc1 * (sqrtf(eas.w / bc2) + d_lr_eps));
    param4[i] = p;
}


// ===================================================================
//  Kernel 2: Block reduction for computing adaptive learning rate d
//
//  Accumulates two global sums:
//    numerator   = sum_i  grad[i] * (param[i] - param_init[i])
//    denominator = sum_i  s[i]
//
//  Uses warp-level shuffle reduction + shared memory cross-warp
//  reduction + atomicAdd to global output.
// ===================================================================

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void prodigy_dlr_reduce_kernel(
    const scalar_t* __restrict__ grad,         // [N]
    const scalar_t* __restrict__ param,        // [N]
    const scalar_t* __restrict__ param_init,   // [N]
    const float* __restrict__ s,               // [N] — always FP32
    float* __restrict__ out_num,               // [1] -- global numerator accumulator
    float* __restrict__ out_den,               // [1] -- global denominator accumulator
    const int N
) {
    // Shared memory for cross-warp reduction
    // Layout: [num_warps floats for numerator][num_warps floats for denominator]
    constexpr int NUM_WARPS = PRODIGY_BLOCK_SIZE / PRODIGY_WARP_SIZE;
    __shared__ float shared_num[NUM_WARPS];
    __shared__ float shared_den[NUM_WARPS];

    const int tid = threadIdx.x;
    const int warp_id = tid / PRODIGY_WARP_SIZE;
    const int lane_id = tid % PRODIGY_WARP_SIZE;

    // -- Per-thread accumulation ----------------------------------------
    float local_num = 0.0f;
    float local_den = 0.0f;

    // Grid-stride loop for large tensors
    for (int idx = blockIdx.x * blockDim.x + tid; idx < N;
         idx += gridDim.x * blockDim.x) {
        const float g = static_cast<float>(grad[idx]);
        const float p = static_cast<float>(param[idx]);
        const float p0 = static_cast<float>(param_init[idx]);
        const float s_val = static_cast<float>(s[idx]);

        local_num += g * (p - p0);
        local_den += s_val;
    }

    // -- Warp-level reduction via shuffle --------------------------------
    for (int offset = PRODIGY_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_num += SHFL_DOWN(local_num, offset);
        local_den += SHFL_DOWN(local_den, offset);
    }

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        shared_num[warp_id] = local_num;
        shared_den[warp_id] = local_den;
    }
    __syncthreads();

    // -- Cross-warp reduction (first warp only) -------------------------
    if (warp_id == 0) {
        float warp_num = (lane_id < NUM_WARPS) ? shared_num[lane_id] : 0.0f;
        float warp_den = (lane_id < NUM_WARPS) ? shared_den[lane_id] : 0.0f;

        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            warp_num += SHFL_DOWN(warp_num, offset);
            warp_den += SHFL_DOWN(warp_den, offset);
        }

        // Lane 0 of warp 0 does atomic add to global accumulators
        if (lane_id == 0) {
            atomicAdd(out_num, warp_num);
            atomicAdd(out_den, warp_den);
        }
    }
}


// ===================================================================
//  C++ Dispatch Functions
// ===================================================================

void launch_fused_prodigy_step(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor s,
    torch::Tensor grad,
    float d_lr,
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

    // float4 fast path: FP32 params, N divisible by 4, 16-byte aligned
    if (param.scalar_type() == at::ScalarType::Float &&
        N % 4 == 0 &&
        reinterpret_cast<uintptr_t>(param.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0) {
        const int N4 = N / 4;
        const int grid4 = (N4 + PRODIGY_BLOCK_SIZE - 1) / PRODIGY_BLOCK_SIZE;
        fused_prodigy_step_vec4_kernel<<<grid4, PRODIGY_BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            reinterpret_cast<float4*>(s.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            d_lr, beta1, beta2, lr, weight_decay, eps, bc1, bc2, N4);
        return;
    }

    const int grid = (N + PRODIGY_BLOCK_SIZE - 1) / PRODIGY_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_prodigy_step", ([&] {
            fused_prodigy_step_kernel<scalar_t><<<grid, PRODIGY_BLOCK_SIZE>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                s.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                d_lr,
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

void launch_prodigy_dlr_reduce(
    torch::Tensor grad,
    torch::Tensor param,
    torch::Tensor param_init,
    torch::Tensor s,
    torch::Tensor out_num,
    torch::Tensor out_den
) {
    const int N = grad.numel();
    if (N == 0) return;

    // Cap grid size for reduction kernel to avoid excessive atomics
    const int max_blocks = 1024;
    const int grid = min((N + PRODIGY_BLOCK_SIZE - 1) / PRODIGY_BLOCK_SIZE,
                         max_blocks);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "prodigy_dlr_reduce", ([&] {
            prodigy_dlr_reduce_kernel<scalar_t><<<grid, PRODIGY_BLOCK_SIZE>>>(
                grad.data_ptr<scalar_t>(),
                param.data_ptr<scalar_t>(),
                param_init.data_ptr<scalar_t>(),
                s.data_ptr<float>(),
                out_num.data_ptr<float>(),
                out_den.data_ptr<float>(),
                N
            );
        })
    );
}
