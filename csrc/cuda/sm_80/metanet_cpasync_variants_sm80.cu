#include <torch/extension.h>
#include "platform.h"
#include "utils.cuh"

#if GROK_CUDA
#include <cuda_pipeline.h>
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

constexpr int CPASYNC_BLOCK = 256;

__device__ __forceinline__ unsigned cpasync_hash_prng(unsigned step, unsigned idx) {
    unsigned h = (step * 2654435761u) ^ (idx * 2246822519u);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__ int8_t cpasync_int8_stochastic(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / fmaxf(scale, 1e-12f);
    float tr = truncf(scaled);
    float frac = fabsf(scaled - tr);
    float threshold = (float)(rand_bits & 0xFFFF) / 65536.0f;
    if (frac > threshold) tr += (scaled > 0) ? 1.0f : -1.0f;
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, tr));
}

template <typename scalar_t>
__launch_bounds__(256, 2)
__global__ __launch_bounds__(256, 2) void fused_supergrok15_full_step_cpasync_q4_kernel(
    scalar_t* __restrict__ param,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_scales,
    int8_t* __restrict__ exp_avg_sq_q,
    float* __restrict__ exp_avg_sq_scales,
    scalar_t* __restrict__ mu,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharpness,
    const float alpha,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    const float rescale,
    const float lamb_eff,
    const float beta1,
    const float beta2,
    const float lr,
    const float wd_eff,
    const float eps,
    const float bc1,
    const float bc2,
    const unsigned step,
    const int N,
    const int H
) {
    extern __shared__ float smem[];
    float* sW1 = smem;
    float* sb1 = sW1 + H * 2;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

    for (int i = tid; i < H * 2; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    __pipeline_wait_prior(1);
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

    const float mu_old = static_cast<float>(mu[idx]);
    const float mu_new = alpha * mu_old + (1.0f - alpha) * g;
    mu[idx] = static_cast<scalar_t>(mu_new);

    float hidden[128];
    for (int h = 0; h < H; h++) {
        float z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];
        hidden[h] = z / (1.0f + expf(-1.702f * z));
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float mlp_out = 0.0f;
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    const float smart_grad = g + rescale * mlp_out;
    const float fg = smart_grad + lamb_eff * mu_new;

    const int blk = idx / 8;
    float ea_scale = stream_load(&exp_avg_scales[blk]);
    float ea_old = (float)exp_avg_q[idx] * ea_scale;
    float easq_scale = stream_load(&exp_avg_sq_scales[blk]);
    float easq_old = (float)exp_avg_sq_q[idx] * easq_scale;

    const float ea = beta1 * ea_old + (1.0f - beta1) * fg;
    const float easq = beta2 * easq_old + (1.0f - beta2) * fg * fg;

    float ea_new_scale = fmaxf(fabsf(ea) / 127.0f, 1e-12f);
    float easq_new_scale = fmaxf(fabsf(easq) / 127.0f, 1e-12f);

    unsigned rng = cpasync_hash_prng(step, idx);
    exp_avg_q[idx] = cpasync_int8_stochastic(ea, ea_new_scale, rng);
    stream_store(&exp_avg_scales[blk], ea_new_scale);
    unsigned rng2 = cpasync_hash_prng(step, idx + N);
    exp_avg_sq_q[idx] = cpasync_int8_stochastic(easq, easq_new_scale, rng2);
    stream_store(&exp_avg_sq_scales[blk], easq_new_scale);

    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}

template <typename scalar_t>
__launch_bounds__(256, 2)
__global__ __launch_bounds__(256, 2) void fused_supergrok15_full_step_cpasync_moe_kernel(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    scalar_t* __restrict__ mu,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharpness,
    const bool* __restrict__ active_mask,
    const float alpha,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
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
    extern __shared__ float smem[];
    float* sW1 = smem;
    float* sb1 = sW1 + H * 2;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

    for (int i = tid; i < H * 2; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    __pipeline_wait_prior(1);
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;
    if (!active_mask[idx]) return;

    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

    const float mu_old = static_cast<float>(mu[idx]);
    const float mu_new = alpha * mu_old + (1.0f - alpha) * g;
    mu[idx] = static_cast<scalar_t>(mu_new);

    float hidden[128];
    for (int h = 0; h < H; h++) {
        float z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];
        hidden[h] = z / (1.0f + expf(-1.702f * z));
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float mlp_out = 0.0f;
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    const float smart_grad = g + rescale * mlp_out;
    const float fg = smart_grad + lamb_eff * mu_new;

    const float ea = beta1 * stream_load(&exp_avg[idx]) + (1.0f - beta1) * fg;
    const float easq = beta2 * stream_load(&exp_avg_sq[idx]) + (1.0f - beta2) * fg * fg;
    stream_store(&exp_avg[idx], ea);
    stream_store(&exp_avg_sq[idx], easq);

    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}

template <typename scalar_t>
__launch_bounds__(256, 2)
__global__ __launch_bounds__(256, 2) void fused_supergrok15_full_step_cpasync_moe_q4_kernel(
    scalar_t* __restrict__ param,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_scales,
    int8_t* __restrict__ exp_avg_sq_q,
    float* __restrict__ exp_avg_sq_scales,
    scalar_t* __restrict__ mu,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharpness,
    const bool* __restrict__ active_mask,
    const float alpha,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    const float rescale,
    const float lamb_eff,
    const float beta1,
    const float beta2,
    const float lr,
    const float wd_eff,
    const float eps,
    const float bc1,
    const float bc2,
    const unsigned step,
    const int N,
    const int H
) {
    extern __shared__ float smem[];
    float* sW1 = smem;
    float* sb1 = sW1 + H * 2;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

    for (int i = tid; i < H * 2; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    __pipeline_wait_prior(1);
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;
    if (!active_mask[idx]) return;

    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

    const float mu_old = static_cast<float>(mu[idx]);
    const float mu_new = alpha * mu_old + (1.0f - alpha) * g;
    mu[idx] = static_cast<scalar_t>(mu_new);

    float hidden[128];
    for (int h = 0; h < H; h++) {
        float z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];
        hidden[h] = z / (1.0f + expf(-1.702f * z));
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float mlp_out = 0.0f;
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    const float smart_grad = g + rescale * mlp_out;
    const float fg = smart_grad + lamb_eff * mu_new;

    const int blk = idx / 8;
    float ea_scale = stream_load(&exp_avg_scales[blk]);
    float ea_old = (float)exp_avg_q[idx] * ea_scale;
    float easq_scale = stream_load(&exp_avg_sq_scales[blk]);
    float easq_old = (float)exp_avg_sq_q[idx] * easq_scale;

    const float ea = beta1 * ea_old + (1.0f - beta1) * fg;
    const float easq = beta2 * easq_old + (1.0f - beta2) * fg * fg;

    float ea_new_scale = fmaxf(fabsf(ea) / 127.0f, 1e-12f);
    float easq_new_scale = fmaxf(fabsf(easq) / 127.0f, 1e-12f);

    unsigned rng = cpasync_hash_prng(step, idx);
    exp_avg_q[idx] = cpasync_int8_stochastic(ea, ea_new_scale, rng);
    stream_store(&exp_avg_scales[blk], ea_new_scale);
    unsigned rng2 = cpasync_hash_prng(step, idx + N);
    exp_avg_sq_q[idx] = cpasync_int8_stochastic(easq, easq_new_scale, rng2);
    stream_store(&exp_avg_sq_scales[blk], easq_new_scale);

    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}

template <typename scalar_t>
__launch_bounds__(256, 2)
__global__ __launch_bounds__(256, 2) void fused_sg11_full_step_cpasync_q4_kernel(
    scalar_t* __restrict__ param,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_scales,
    int8_t* __restrict__ exp_avg_sq_q,
    float* __restrict__ exp_avg_sq_scales,
    scalar_t* __restrict__ mu,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharpness,
    const float alpha,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    const float rescale,
    const float lamb_eff,
    const float beta1,
    const float beta2,
    const float lr,
    const float wd_eff,
    const float eps,
    const float bc1,
    const float bc2,
    const unsigned step,
    const int N,
    const int H
) {
    extern __shared__ float smem[];
    float* sW1 = smem;
    float* sb1 = sW1 + H * 2;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

    for (int i = tid; i < H * 2; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    __pipeline_wait_prior(1);
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

    const float mu_old = static_cast<float>(mu[idx]);
    const float mu_new = alpha * mu_old + (1.0f - alpha) * g;
    mu[idx] = static_cast<scalar_t>(mu_new);

    float hidden[128];
    for (int h = 0; h < H; h++) {
        float z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];
        hidden[h] = z / (1.0f + expf(-1.702f * z));
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float mlp_out = 0.0f;
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    const float smart_grad = g + rescale * mlp_out;
    const float fg = smart_grad + lamb_eff * mu_new;

    const int blk = idx / 8;
    float ea_scale = stream_load(&exp_avg_scales[blk]);
    float ea_old = (float)exp_avg_q[idx] * ea_scale;
    float easq_scale = stream_load(&exp_avg_sq_scales[blk]);
    float easq_old = (float)exp_avg_sq_q[idx] * easq_scale;

    const float ea = beta1 * ea_old + (1.0f - beta1) * fg;
    const float easq = beta2 * easq_old + (1.0f - beta2) * fg * fg;

    float ea_new_scale = fmaxf(fabsf(ea) / 127.0f, 1e-12f);
    float easq_new_scale = fmaxf(fabsf(easq) / 127.0f, 1e-12f);

    unsigned rng = cpasync_hash_prng(step, idx);
    exp_avg_q[idx] = cpasync_int8_stochastic(ea, ea_new_scale, rng);
    stream_store(&exp_avg_scales[blk], ea_new_scale);
    unsigned rng2 = cpasync_hash_prng(step, idx + N);
    exp_avg_sq_q[idx] = cpasync_int8_stochastic(easq, easq_new_scale, rng2);
    stream_store(&exp_avg_sq_scales[blk], easq_new_scale);

    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}

template <typename scalar_t>
__launch_bounds__(256, 2)
__global__ __launch_bounds__(256, 2) void fused_sg11_full_step_cpasync_moe_kernel(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    scalar_t* __restrict__ mu,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharpness,
    const bool* __restrict__ active_mask,
    const float alpha,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
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
    extern __shared__ float smem[];
    float* sW1 = smem;
    float* sb1 = sW1 + H * 2;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

    for (int i = tid; i < H * 2; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    __pipeline_wait_prior(1);
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;
    if (!active_mask[idx]) return;

    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

    const float mu_old = static_cast<float>(mu[idx]);
    const float mu_new = alpha * mu_old + (1.0f - alpha) * g;
    mu[idx] = static_cast<scalar_t>(mu_new);

    float hidden[128];
    for (int h = 0; h < H; h++) {
        float z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];
        hidden[h] = z / (1.0f + expf(-1.702f * z));
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float mlp_out = 0.0f;
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    const float smart_grad = g + rescale * mlp_out;
    const float fg = smart_grad + lamb_eff * mu_new;

    const float ea = beta1 * stream_load(&exp_avg[idx]) + (1.0f - beta1) * fg;
    const float easq = beta2 * stream_load(&exp_avg_sq[idx]) + (1.0f - beta2) * fg * fg;
    stream_store(&exp_avg[idx], ea);
    stream_store(&exp_avg_sq[idx], easq);

    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}

template <typename scalar_t>
__launch_bounds__(256, 2)
__global__ __launch_bounds__(256, 2) void fused_sg11_full_step_cpasync_moe_q4_kernel(
    scalar_t* __restrict__ param,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_scales,
    int8_t* __restrict__ exp_avg_sq_q,
    float* __restrict__ exp_avg_sq_scales,
    scalar_t* __restrict__ mu,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharpness,
    const bool* __restrict__ active_mask,
    const float alpha,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    const float rescale,
    const float lamb_eff,
    const float beta1,
    const float beta2,
    const float lr,
    const float wd_eff,
    const float eps,
    const float bc1,
    const float bc2,
    const unsigned step,
    const int N,
    const int H
) {
    extern __shared__ float smem[];
    float* sW1 = smem;
    float* sb1 = sW1 + H * 2;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

    for (int i = tid; i < H * 2; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    __pipeline_wait_prior(1);
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;
    if (!active_mask[idx]) return;

    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

    const float mu_old = static_cast<float>(mu[idx]);
    const float mu_new = alpha * mu_old + (1.0f - alpha) * g;
    mu[idx] = static_cast<scalar_t>(mu_new);

    float hidden[128];
    for (int h = 0; h < H; h++) {
        float z = sW1[h * 2] * g + sW1[h * 2 + 1] * s + sb1[h];
        hidden[h] = z / (1.0f + expf(-1.702f * z));
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float mlp_out = 0.0f;
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    const float smart_grad = g + rescale * mlp_out;
    const float fg = smart_grad + lamb_eff * mu_new;

    const int blk = idx / 8;
    float ea_scale = stream_load(&exp_avg_scales[blk]);
    float ea_old = (float)exp_avg_q[idx] * ea_scale;
    float easq_scale = stream_load(&exp_avg_sq_scales[blk]);
    float easq_old = (float)exp_avg_sq_q[idx] * easq_scale;

    const float ea = beta1 * ea_old + (1.0f - beta1) * fg;
    const float easq = beta2 * easq_old + (1.0f - beta2) * fg * fg;

    float ea_new_scale = fmaxf(fabsf(ea) / 127.0f, 1e-12f);
    float easq_new_scale = fmaxf(fabsf(easq) / 127.0f, 1e-12f);

    unsigned rng = cpasync_hash_prng(step, idx);
    exp_avg_q[idx] = cpasync_int8_stochastic(ea, ea_new_scale, rng);
    stream_store(&exp_avg_scales[blk], ea_new_scale);
    unsigned rng2 = cpasync_hash_prng(step, idx + N);
    exp_avg_sq_q[idx] = cpasync_int8_stochastic(easq, easq_new_scale, rng2);
    stream_store(&exp_avg_sq_scales[blk], easq_new_scale);

    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}

template <typename scalar_t>
__launch_bounds__(256, 2)
__global__ __launch_bounds__(256, 2) void fused_neuralgrok_full_step_cpasync_q4_kernel(
    scalar_t* __restrict__ param,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_scales,
    int8_t* __restrict__ exp_avg_sq_q,
    float* __restrict__ exp_avg_sq_scales,
    const scalar_t* __restrict__ grad,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    const float alpha,
    const float beta,
    const int N,
    const int H,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const unsigned step
) {
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    float* sW1 = smem;
    float* sb1 = sW1 + H;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    __pipeline_wait_prior(1);
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    const float g = static_cast<float>(grad[idx]);

    float hidden[128];
    for (int h = 0; h < H; h++) {
        float z = sW1[h] * g + sb1[h];
        hidden[h] = (z > 0.0f) ? z : 0.0f;
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float mlp_out = 0.0f;
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    const float ag = g * (alpha * mlp_out + beta);

    const int blk = idx / 8;
    float ea_scale = stream_load(&exp_avg_scales[blk]);
    float ea_old = (float)exp_avg_q[idx] * ea_scale;
    float easq_scale = stream_load(&exp_avg_sq_scales[blk]);
    float easq_old = (float)exp_avg_sq_q[idx] * easq_scale;

    const float ea = beta1 * ea_old + (1.0f - beta1) * ag;
    const float easq = beta2 * easq_old + (1.0f - beta2) * ag * ag;

    float ea_new_scale = fmaxf(fabsf(ea) / 127.0f, 1e-12f);
    float easq_new_scale = fmaxf(fabsf(easq) / 127.0f, 1e-12f);

    unsigned rng = cpasync_hash_prng(step, idx);
    exp_avg_q[idx] = cpasync_int8_stochastic(ea, ea_new_scale, rng);
    stream_store(&exp_avg_scales[blk], ea_new_scale);
    unsigned rng2 = cpasync_hash_prng(step, idx + N);
    exp_avg_sq_q[idx] = cpasync_int8_stochastic(easq, easq_new_scale, rng2);
    stream_store(&exp_avg_sq_scales[blk], easq_new_scale);

    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}

template <typename scalar_t>
__launch_bounds__(256, 2)
__global__ __launch_bounds__(256, 2) void fused_neuralgrok_full_step_cpasync_moe_kernel(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const scalar_t* __restrict__ grad,
    const bool* __restrict__ active_mask,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    const float alpha,
    const float beta,
    const int N,
    const int H,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2
) {
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    float* sW1 = smem;
    float* sb1 = sW1 + H;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    __pipeline_wait_prior(1);
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;
    if (!active_mask[idx]) return;

    const float g = static_cast<float>(grad[idx]);

    float hidden[128];
    for (int h = 0; h < H; h++) {
        float z = sW1[h] * g + sb1[h];
        hidden[h] = (z > 0.0f) ? z : 0.0f;
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float mlp_out = 0.0f;
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    const float ag = g * (alpha * mlp_out + beta);

    const float ea_old = stream_load(&exp_avg[idx]);
    const float easq_old = stream_load(&exp_avg_sq[idx]);

    const float ea = beta1 * ea_old + (1.0f - beta1) * ag;
    const float easq = beta2 * easq_old + (1.0f - beta2) * ag * ag;

    stream_store(&exp_avg[idx], ea);
    stream_store(&exp_avg_sq[idx], easq);

    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}

template <typename scalar_t>
__launch_bounds__(256, 2)
__global__ __launch_bounds__(256, 2) void fused_neuralgrok_full_step_cpasync_moe_q4_kernel(
    scalar_t* __restrict__ param,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_scales,
    int8_t* __restrict__ exp_avg_sq_q,
    float* __restrict__ exp_avg_sq_scales,
    const scalar_t* __restrict__ grad,
    const bool* __restrict__ active_mask,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    const float alpha,
    const float beta,
    const int N,
    const int H,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const unsigned step
) {
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    float* sW1 = smem;
    float* sb1 = sW1 + H;
    float* sW2 = sb1 + H;
    float* sb2 = sW2 + H;

    const int tid = threadIdx.x;

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW1[i], &W1[i], sizeof(float));
    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sb1[i], &b1[i], sizeof(float));
    __pipeline_commit();

    for (int i = tid; i < H; i += blockDim.x)
        __pipeline_memcpy_async(&sW2[i], &W2[i], sizeof(float));
    if (tid == 0)
        __pipeline_memcpy_async(&sb2[0], &b2[0], sizeof(float));
    __pipeline_commit();

    __pipeline_wait_prior(1);
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;
    if (!active_mask[idx]) return;

    const float g = static_cast<float>(grad[idx]);

    float hidden[128];
    for (int h = 0; h < H; h++) {
        float z = sW1[h] * g + sb1[h];
        hidden[h] = (z > 0.0f) ? z : 0.0f;
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float mlp_out = 0.0f;
    for (int h = 0; h < H; h++) {
        mlp_out += sW2[h] * hidden[h];
    }
    mlp_out += sb2[0];

    const float ag = g * (alpha * mlp_out + beta);

    const int blk = idx / 8;
    float ea_scale = stream_load(&exp_avg_scales[blk]);
    float ea_old = (float)exp_avg_q[idx] * ea_scale;
    float easq_scale = stream_load(&exp_avg_sq_scales[blk]);
    float easq_old = (float)exp_avg_sq_q[idx] * easq_scale;

    const float ea = beta1 * ea_old + (1.0f - beta1) * ag;
    const float easq = beta2 * easq_old + (1.0f - beta2) * ag * ag;

    float ea_new_scale = fmaxf(fabsf(ea) / 127.0f, 1e-12f);
    float easq_new_scale = fmaxf(fabsf(easq) / 127.0f, 1e-12f);

    unsigned rng = cpasync_hash_prng(step, idx);
    exp_avg_q[idx] = cpasync_int8_stochastic(ea, ea_new_scale, rng);
    stream_store(&exp_avg_scales[blk], ea_new_scale);
    unsigned rng2 = cpasync_hash_prng(step, idx + N);
    exp_avg_sq_q[idx] = cpasync_int8_stochastic(easq, easq_new_scale, rng2);
    stream_store(&exp_avg_sq_scales[blk], easq_new_scale);

    const float step_size = lr / bc1;
    const float denom = sqrtf(easq / bc2) + eps;
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p);
}

#endif



void launch_fused_supergrok15_full_step_cpasync_q4(
    torch::Tensor param,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_scales,
    torch::Tensor mu,
    torch::Tensor grad,
    torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2,
    float lr, float wd_eff, float eps,
    float bc1, float bc2,
    unsigned step,
    int hidden_dim
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + CPASYNC_BLOCK - 1) / CPASYNC_BLOCK;
    const int smem_bytes = (hidden_dim * 4 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_supergrok15_full_step_cpasync_q4", ([&] {
        fused_supergrok15_full_step_cpasync_q4_kernel<scalar_t><<<grid, CPASYNC_BLOCK, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg_q.data_ptr<int8_t>(),
            exp_avg_scales.data_ptr<float>(),
            exp_avg_sq_q.data_ptr<int8_t>(),
            exp_avg_sq_scales.data_ptr<float>(),
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
            step, N, hidden_dim
        );
    }));
}

void launch_fused_supergrok15_full_step_cpasync_moe(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor mu,
    torch::Tensor grad,
    torch::Tensor sharpness,
    torch::Tensor active_mask,
    float alpha,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2,
    float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int hidden_dim
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + CPASYNC_BLOCK - 1) / CPASYNC_BLOCK;
    const int smem_bytes = (hidden_dim * 4 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_supergrok15_full_step_cpasync_moe", ([&] {
        fused_supergrok15_full_step_cpasync_moe_kernel<scalar_t><<<grid, CPASYNC_BLOCK, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            mu.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            sharpness.data_ptr<scalar_t>(),
            active_mask.data_ptr<bool>(),
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

void launch_fused_supergrok15_full_step_cpasync_moe_q4(
    torch::Tensor param,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_scales,
    torch::Tensor mu,
    torch::Tensor grad,
    torch::Tensor sharpness,
    torch::Tensor active_mask,
    float alpha,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2,
    float lr, float wd_eff, float eps,
    float bc1, float bc2,
    unsigned step,
    int hidden_dim
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + CPASYNC_BLOCK - 1) / CPASYNC_BLOCK;
    const int smem_bytes = (hidden_dim * 4 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_supergrok15_full_step_cpasync_moe_q4", ([&] {
        fused_supergrok15_full_step_cpasync_moe_q4_kernel<scalar_t><<<grid, CPASYNC_BLOCK, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg_q.data_ptr<int8_t>(),
            exp_avg_scales.data_ptr<float>(),
            exp_avg_sq_q.data_ptr<int8_t>(),
            exp_avg_sq_scales.data_ptr<float>(),
            mu.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            sharpness.data_ptr<scalar_t>(),
            active_mask.data_ptr<bool>(),
            alpha,
            W1_f.data_ptr<float>(),
            b1_f.data_ptr<float>(),
            W2_f.data_ptr<float>(),
            b2_f.data_ptr<float>(),
            rescale, lamb_eff,
            beta1, beta2, lr, wd_eff, eps, bc1, bc2,
            step, N, hidden_dim
        );
    }));
}

void launch_fused_sg11_full_step_cpasync_q4(
    torch::Tensor param,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_scales,
    torch::Tensor mu,
    torch::Tensor grad,
    torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2,
    float lr, float wd_eff, float eps,
    float bc1, float bc2,
    unsigned step,
    int hidden_dim
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + CPASYNC_BLOCK - 1) / CPASYNC_BLOCK;
    const int smem_bytes = (hidden_dim * 4 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_sg11_full_step_cpasync_q4", ([&] {
        fused_sg11_full_step_cpasync_q4_kernel<scalar_t><<<grid, CPASYNC_BLOCK, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg_q.data_ptr<int8_t>(),
            exp_avg_scales.data_ptr<float>(),
            exp_avg_sq_q.data_ptr<int8_t>(),
            exp_avg_sq_scales.data_ptr<float>(),
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
            step, N, hidden_dim
        );
    }));
}

void launch_fused_sg11_full_step_cpasync_moe(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor mu,
    torch::Tensor grad,
    torch::Tensor sharpness,
    torch::Tensor active_mask,
    float alpha,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2,
    float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int hidden_dim
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + CPASYNC_BLOCK - 1) / CPASYNC_BLOCK;
    const int smem_bytes = (hidden_dim * 4 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_sg11_full_step_cpasync_moe", ([&] {
        fused_sg11_full_step_cpasync_moe_kernel<scalar_t><<<grid, CPASYNC_BLOCK, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            mu.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            sharpness.data_ptr<scalar_t>(),
            active_mask.data_ptr<bool>(),
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

void launch_fused_sg11_full_step_cpasync_moe_q4(
    torch::Tensor param,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_scales,
    torch::Tensor mu,
    torch::Tensor grad,
    torch::Tensor sharpness,
    torch::Tensor active_mask,
    float alpha,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2,
    float lr, float wd_eff, float eps,
    float bc1, float bc2,
    unsigned step,
    int hidden_dim
) {
    const int N = grad.numel();
    if (N == 0) return;
    const int grid = (N + CPASYNC_BLOCK - 1) / CPASYNC_BLOCK;
    const int smem_bytes = (hidden_dim * 4 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "fused_sg11_full_step_cpasync_moe_q4", ([&] {
        fused_sg11_full_step_cpasync_moe_q4_kernel<scalar_t><<<grid, CPASYNC_BLOCK, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg_q.data_ptr<int8_t>(),
            exp_avg_scales.data_ptr<float>(),
            exp_avg_sq_q.data_ptr<int8_t>(),
            exp_avg_sq_scales.data_ptr<float>(),
            mu.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            sharpness.data_ptr<scalar_t>(),
            active_mask.data_ptr<bool>(),
            alpha,
            W1_f.data_ptr<float>(),
            b1_f.data_ptr<float>(),
            W2_f.data_ptr<float>(),
            b2_f.data_ptr<float>(),
            rescale, lamb_eff,
            beta1, beta2, lr, wd_eff, eps, bc1, bc2,
            step, N, hidden_dim
        );
    }));
}

void launch_fused_neuralgrok_full_step_cpasync_q4(
    torch::Tensor param,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_scales,
    torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float alpha_scale, float beta_shift,
    int hidden_dim,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2,
    unsigned step
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + CPASYNC_BLOCK - 1) / CPASYNC_BLOCK;
    const int smem_bytes = (hidden_dim * 3 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_neuralgrok_full_step_cpasync_q4", ([&] {
        fused_neuralgrok_full_step_cpasync_q4_kernel<scalar_t><<<grid, CPASYNC_BLOCK, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg_q.data_ptr<int8_t>(),
            exp_avg_scales.data_ptr<float>(),
            exp_avg_sq_q.data_ptr<int8_t>(),
            exp_avg_sq_scales.data_ptr<float>(),
            grad.data_ptr<scalar_t>(),
            W1_f.data_ptr<float>(),
            b1_f.data_ptr<float>(),
            W2_f.data_ptr<float>(),
            b2_f.data_ptr<float>(),
            alpha_scale, beta_shift,
            N, hidden_dim,
            beta1, beta2, lr, weight_decay, eps, bc1, bc2,
            step
        );
    }));
}

void launch_fused_neuralgrok_full_step_cpasync_moe(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor grad,
    torch::Tensor active_mask,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float alpha_scale, float beta_shift,
    int hidden_dim,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + CPASYNC_BLOCK - 1) / CPASYNC_BLOCK;
    const int smem_bytes = (hidden_dim * 3 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_neuralgrok_full_step_cpasync_moe", ([&] {
        fused_neuralgrok_full_step_cpasync_moe_kernel<scalar_t><<<grid, CPASYNC_BLOCK, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            grad.data_ptr<scalar_t>(),
            active_mask.data_ptr<bool>(),
            W1_f.data_ptr<float>(),
            b1_f.data_ptr<float>(),
            W2_f.data_ptr<float>(),
            b2_f.data_ptr<float>(),
            alpha_scale, beta_shift,
            N, hidden_dim,
            beta1, beta2, lr, weight_decay, eps, bc1, bc2
        );
    }));
}

void launch_fused_neuralgrok_full_step_cpasync_moe_q4(
    torch::Tensor param,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_scales,
    torch::Tensor grad,
    torch::Tensor active_mask,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float alpha_scale, float beta_shift,
    int hidden_dim,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2,
    unsigned step
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + CPASYNC_BLOCK - 1) / CPASYNC_BLOCK;
    const int smem_bytes = (hidden_dim * 3 + 1) * sizeof(float);

    auto W1_f = W1.dtype() == torch::kFloat32 ? W1.contiguous() : W1.to(torch::kFloat32).contiguous();
    auto b1_f = b1.dtype() == torch::kFloat32 ? b1.contiguous() : b1.to(torch::kFloat32).contiguous();
    auto W2_f = W2.dtype() == torch::kFloat32 ? W2.contiguous() : W2.to(torch::kFloat32).contiguous();
    auto b2_f = b2.dtype() == torch::kFloat32 ? b2.contiguous() : b2.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_neuralgrok_full_step_cpasync_moe_q4", ([&] {
        fused_neuralgrok_full_step_cpasync_moe_q4_kernel<scalar_t><<<grid, CPASYNC_BLOCK, smem_bytes>>>(
            param.data_ptr<scalar_t>(),
            exp_avg_q.data_ptr<int8_t>(),
            exp_avg_scales.data_ptr<float>(),
            exp_avg_sq_q.data_ptr<int8_t>(),
            exp_avg_sq_scales.data_ptr<float>(),
            grad.data_ptr<scalar_t>(),
            active_mask.data_ptr<bool>(),
            W1_f.data_ptr<float>(),
            b1_f.data_ptr<float>(),
            W2_f.data_ptr<float>(),
            b2_f.data_ptr<float>(),
            alpha_scale, beta_shift,
            N, hidden_dim,
            beta1, beta2, lr, weight_decay, eps, bc1, bc2,
            step
        );
    }));
}

template __global__ void fused_supergrok15_full_step_cpasync_q4_kernel<float>(float*, int8_t*, float*, int8_t*, float*, float*, const float*, const float*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);
template __global__ void fused_supergrok15_full_step_cpasync_q4_kernel<at::Half>(at::Half*, int8_t*, float*, int8_t*, float*, at::Half*, const at::Half*, const at::Half*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);
template __global__ void fused_supergrok15_full_step_cpasync_q4_kernel<at::BFloat16>(at::BFloat16*, int8_t*, float*, int8_t*, float*, at::BFloat16*, const at::BFloat16*, const at::BFloat16*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);

template __global__ void fused_supergrok15_full_step_cpasync_moe_kernel<float>(float*, float*, float*, float*, const float*, const float*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const int, const int);
template __global__ void fused_supergrok15_full_step_cpasync_moe_kernel<at::Half>(at::Half*, float*, float*, at::Half*, const at::Half*, const at::Half*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const int, const int);
template __global__ void fused_supergrok15_full_step_cpasync_moe_kernel<at::BFloat16>(at::BFloat16*, float*, float*, at::BFloat16*, const at::BFloat16*, const at::BFloat16*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const int, const int);

template __global__ void fused_supergrok15_full_step_cpasync_moe_q4_kernel<float>(float*, int8_t*, float*, int8_t*, float*, float*, const float*, const float*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);
template __global__ void fused_supergrok15_full_step_cpasync_moe_q4_kernel<at::Half>(at::Half*, int8_t*, float*, int8_t*, float*, at::Half*, const at::Half*, const at::Half*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);
template __global__ void fused_supergrok15_full_step_cpasync_moe_q4_kernel<at::BFloat16>(at::BFloat16*, int8_t*, float*, int8_t*, float*, at::BFloat16*, const at::BFloat16*, const at::BFloat16*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);

template __global__ void fused_sg11_full_step_cpasync_q4_kernel<float>(float*, int8_t*, float*, int8_t*, float*, float*, const float*, const float*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);
template __global__ void fused_sg11_full_step_cpasync_q4_kernel<at::Half>(at::Half*, int8_t*, float*, int8_t*, float*, at::Half*, const at::Half*, const at::Half*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);
template __global__ void fused_sg11_full_step_cpasync_q4_kernel<at::BFloat16>(at::BFloat16*, int8_t*, float*, int8_t*, float*, at::BFloat16*, const at::BFloat16*, const at::BFloat16*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);

template __global__ void fused_sg11_full_step_cpasync_moe_kernel<float>(float*, float*, float*, float*, const float*, const float*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const int, const int);
template __global__ void fused_sg11_full_step_cpasync_moe_kernel<at::Half>(at::Half*, float*, float*, at::Half*, const at::Half*, const at::Half*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const int, const int);
template __global__ void fused_sg11_full_step_cpasync_moe_kernel<at::BFloat16>(at::BFloat16*, float*, float*, at::BFloat16*, const at::BFloat16*, const at::BFloat16*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const int, const int);

template __global__ void fused_sg11_full_step_cpasync_moe_q4_kernel<float>(float*, int8_t*, float*, int8_t*, float*, float*, const float*, const float*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);
template __global__ void fused_sg11_full_step_cpasync_moe_q4_kernel<at::Half>(at::Half*, int8_t*, float*, int8_t*, float*, at::Half*, const at::Half*, const at::Half*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);
template __global__ void fused_sg11_full_step_cpasync_moe_q4_kernel<at::BFloat16>(at::BFloat16*, int8_t*, float*, int8_t*, float*, at::BFloat16*, const at::BFloat16*, const at::BFloat16*, const bool*, const float, const float*, const float*, const float*, const float*, const float, const float, const float, const float, const float, const float, const float, const float, const float, const unsigned, const int, const int);

template __global__ void fused_neuralgrok_full_step_cpasync_q4_kernel<float>(float*, int8_t*, float*, int8_t*, float*, const float*, const float*, const float*, const float*, const float*, const float, const float, const int, const int, const float, const float, const float, const float, const float, const float, const float, const unsigned);
template __global__ void fused_neuralgrok_full_step_cpasync_q4_kernel<at::Half>(at::Half*, int8_t*, float*, int8_t*, float*, const at::Half*, const float*, const float*, const float*, const float*, const float, const float, const int, const int, const float, const float, const float, const float, const float, const float, const float, const unsigned);
template __global__ void fused_neuralgrok_full_step_cpasync_q4_kernel<at::BFloat16>(at::BFloat16*, int8_t*, float*, int8_t*, float*, const at::BFloat16*, const float*, const float*, const float*, const float*, const float, const float, const int, const int, const float, const float, const float, const float, const float, const float, const float, const unsigned);

template __global__ void fused_neuralgrok_full_step_cpasync_moe_kernel<float>(float*, float*, float*, const float*, const bool*, const float*, const float*, const float*, const float*, const float, const float, const int, const int, const float, const float, const float, const float, const float, const float, const float);
template __global__ void fused_neuralgrok_full_step_cpasync_moe_kernel<at::Half>(at::Half*, float*, float*, const at::Half*, const bool*, const float*, const float*, const float*, const float*, const float, const float, const int, const int, const float, const float, const float, const float, const float, const float, const float);
template __global__ void fused_neuralgrok_full_step_cpasync_moe_kernel<at::BFloat16>(at::BFloat16*, float*, float*, const at::BFloat16*, const bool*, const float*, const float*, const float*, const float*, const float, const float, const int, const int, const float, const float, const float, const float, const float, const float, const float);

template __global__ void fused_neuralgrok_full_step_cpasync_moe_q4_kernel<float>(float*, int8_t*, float*, int8_t*, float*, const float*, const bool*, const float*, const float*, const float*, const float*, const float, const float, const int, const int, const float, const float, const float, const float, const float, const float, const float, const unsigned);
template __global__ void fused_neuralgrok_full_step_cpasync_moe_q4_kernel<at::Half>(at::Half*, int8_t*, float*, int8_t*, float*, const at::Half*, const bool*, const float*, const float*, const float*, const float*, const float, const float, const int, const int, const float, const float, const float, const float, const float, const float, const float, const unsigned);
template __global__ void fused_neuralgrok_full_step_cpasync_moe_q4_kernel<at::BFloat16>(at::BFloat16*, int8_t*, float*, int8_t*, float*, const at::BFloat16*, const bool*, const float*, const float*, const float*, const float*, const float, const float, const int, const int, const float, const float, const float, const float, const float, const float, const float, const unsigned);
