/*
 * Multi-Tensor Optimizer Kernels
 *
 * Fuses N per-parameter kernel launches into a single launch using a 2D grid:
 *   blockIdx.y selects the parameter, threads within that row iterate over
 *   elements with grid-stride loops.
 *
 * Four optimizers:
 *   1. GrokAdamW  — Adam with grokking-aware EMA gradient filtering
 *   2. Lion       — Sign-based optimizer with EMA interpolation
 *   3. Grokfast EMA — In-place gradient filtering + amplification
 *   4. Prodigy step — Distance-aware, self-tuning Adam variant
 *
 * Pointer arrays are packed on the CPU as int64_t tensors (holding raw
 * device pointers), then transferred to GPU memory before the launch.
 * Inside the kernel, each pointer is reinterpreted to the actual type.
 *
 * Uses float accumulation internally for all arithmetic to preserve
 * numerical precision with FP16/BF16 parameter tensors.
 */

#include <torch/extension.h>
#include "platform.h"
#include "utils.cuh"

constexpr int MT_BLOCK_SIZE = 256;

// Maximum blocks per parameter dimension (x-axis) to keep grid reasonable
constexpr int MT_MAX_BLOCKS_PER_PARAM = 1024;


// ═══════════════════════════════════════════════════════════════════════
//  Helper: build a device tensor of raw pointers from a vector of tensors
//
//  Returns a 1-D int64 tensor on the same device as tensors[0], where
//  element i holds the raw data_ptr of tensors[i] cast to int64_t.
// ═══════════════════════════════════════════════════════════════════════

template <typename T>
static torch::Tensor pack_pointers(const std::vector<torch::Tensor>& tensors) {
    const int64_t n = static_cast<int64_t>(tensors.size());
    auto cpu_buf = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64));
    int64_t* dst = cpu_buf.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; i++) {
        dst[i] = reinterpret_cast<int64_t>(tensors[i].data_ptr<T>());
    }
    return cpu_buf.to(tensors[0].device(), /*non_blocking=*/true);
}

// Variant that deduces pointer type from scalar_type at runtime.
// Used for param/grad tensors whose dtype matches the AT_DISPATCH scalar_t.
static torch::Tensor pack_pointers_scalar(
    const std::vector<torch::Tensor>& tensors, at::ScalarType dtype
) {
    const int64_t n = static_cast<int64_t>(tensors.size());
    auto cpu_buf = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64));
    int64_t* dst = cpu_buf.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; i++) {
        dst[i] = reinterpret_cast<int64_t>(tensors[i].data_ptr());
    }
    return cpu_buf.to(tensors[0].device(), /*non_blocking=*/true);
}

// Helper: build a device tensor from a vector of floats
static torch::Tensor pack_scalars(
    const std::vector<float>& vals, const torch::Device& device
) {
    auto cpu_buf = torch::from_blob(
        const_cast<float*>(vals.data()),
        {static_cast<int64_t>(vals.size())},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
    return cpu_buf.to(device, /*non_blocking=*/true);
}

// Helper: compute sizes array on device
static torch::Tensor pack_sizes(
    const std::vector<torch::Tensor>& tensors, const torch::Device& device
) {
    const int64_t n = static_cast<int64_t>(tensors.size());
    auto cpu_buf = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32));
    int32_t* dst = cpu_buf.data_ptr<int32_t>();
    for (int64_t i = 0; i < n; i++) {
        dst[i] = static_cast<int32_t>(tensors[i].numel());
    }
    return cpu_buf.to(device, /*non_blocking=*/true);
}

// Helper: find max numel across tensors
static int max_numel(const std::vector<torch::Tensor>& tensors) {
    int m = 0;
    for (auto& t : tensors) {
        int n = static_cast<int>(t.numel());
        if (n > m) m = n;
    }
    return m;
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: Multi-Tensor GrokAdamW
//
//  EMA gradient filter + amplification + Adam moments + weight decay
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void multi_tensor_grokadamw_kernel(
    const int64_t* __restrict__ param_ptrs,
    const int64_t* __restrict__ exp_avg_ptrs,
    const int64_t* __restrict__ exp_avg_sq_ptrs,
    const int64_t* __restrict__ ema_ptrs,
    const int64_t* __restrict__ grad_ptrs,
    const int* __restrict__ sizes,
    const float* __restrict__ bc1,
    const float* __restrict__ bc2,
    const float alpha,
    const float lamb,
    const float beta1,
    const float beta2,
    const float lr,
    const float wd,
    const float eps
) {
    const int param_idx = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizes[param_idx]) return;

    // Per-param bias corrections
    const float bc1_val = bc1[param_idx];
    const float bc2_val = bc2[param_idx];

    // Recover typed pointers from packed int64 array
    scalar_t* param   = reinterpret_cast<scalar_t*>(param_ptrs[param_idx]);
    float*    exp_avg = reinterpret_cast<float*>(exp_avg_ptrs[param_idx]);
    float*    exp_avg_sq = reinterpret_cast<float*>(exp_avg_sq_ptrs[param_idx]);
    float*    ema     = reinterpret_cast<float*>(ema_ptrs[param_idx]);
    const scalar_t* grad = reinterpret_cast<const scalar_t*>(grad_ptrs[param_idx]);

    // -- 1. EMA gradient filter -------------------------------------------
    const float g = static_cast<float>(grad[idx]);
    const float ema_old = stream_load(&ema[idx]);
    const float ema_new = alpha * ema_old + (1.0f - alpha) * g;
    stream_store(&ema[idx], ema_new);

    // -- 2. Gradient amplification ----------------------------------------
    const float fg = g + lamb * ema_new;

    // -- 3. Adam moment updates -------------------------------------------
    const float ea = beta1 * stream_load(&exp_avg[idx]) + (1.0f - beta1) * fg;
    const float easq = beta2 * stream_load(&exp_avg_sq[idx]) + (1.0f - beta2) * fg * fg;
    stream_store(&exp_avg[idx], ea);
    stream_store(&exp_avg_sq[idx], easq);

    // -- 4. Bias-corrected step with decoupled weight decay ---------------
    const float rsqrt_v = fast_rsqrt_nr(easq / bc2_val);
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd);
    p -= lr * ea * rsqrt_v / (bc1_val * (1.0f + eps * rsqrt_v));
    param[idx] = static_cast<scalar_t>(p);
}


void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    std::vector<float>& bc1s,
    std::vector<float>& bc2s,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps
) {
    const int num_params = static_cast<int>(params.size());
    if (num_params == 0) return;

    const auto device = params[0].device();
    const auto dtype = params[0].scalar_type();

    // Pack pointer arrays and transfer to GPU
    auto d_param_ptrs     = pack_pointers_scalar(params, dtype);
    auto d_exp_avg_ptrs   = pack_pointers<float>(exp_avgs);
    auto d_exp_avg_sq_ptrs = pack_pointers<float>(exp_avg_sqs);
    auto d_ema_ptrs       = pack_pointers<float>(emas);
    auto d_grad_ptrs      = pack_pointers_scalar(grads, dtype);
    auto d_sizes          = pack_sizes(params, device);
    auto d_bc1            = pack_scalars(bc1s, device);
    auto d_bc2            = pack_scalars(bc2s, device);

    const int max_n = max_numel(params);
    if (max_n == 0) return;

    const int max_blocks = std::min(
        (max_n + MT_BLOCK_SIZE - 1) / MT_BLOCK_SIZE,
        MT_MAX_BLOCKS_PER_PARAM
    );
    dim3 grid(max_blocks, num_params);
    dim3 block(MT_BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dtype, "multi_tensor_grokadamw", ([&] {
            multi_tensor_grokadamw_kernel<scalar_t><<<grid, block>>>(
                d_param_ptrs.data_ptr<int64_t>(),
                d_exp_avg_ptrs.data_ptr<int64_t>(),
                d_exp_avg_sq_ptrs.data_ptr<int64_t>(),
                d_ema_ptrs.data_ptr<int64_t>(),
                d_grad_ptrs.data_ptr<int64_t>(),
                d_sizes.data_ptr<int32_t>(),
                d_bc1.data_ptr<float>(),
                d_bc2.data_ptr<float>(),
                alpha, lamb, beta1, beta2, lr, wd, eps
            );
        })
    );
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: Multi-Tensor Lion
//
//  Sign-based optimizer: sign(beta1 * ema + (1-beta1) * grad)
//  Decoupled weight decay applied before the update.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void multi_tensor_lion_kernel(
    const int64_t* __restrict__ param_ptrs,
    const int64_t* __restrict__ exp_avg_ptrs,
    const int64_t* __restrict__ grad_ptrs,
    const int* __restrict__ sizes,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd
) {
    const int param_idx = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizes[param_idx]) return;

    scalar_t* param   = reinterpret_cast<scalar_t*>(param_ptrs[param_idx]);
    float*    exp_avg = reinterpret_cast<float*>(exp_avg_ptrs[param_idx]);
    const scalar_t* grad = reinterpret_cast<const scalar_t*>(grad_ptrs[param_idx]);

    const float g = static_cast<float>(grad[idx]);
    const float ea = stream_load(&exp_avg[idx]);

    // Interpolation for sign computation
    const float interp = beta1 * ea + (1.0f - beta1) * g;
    const float s = (interp != 0.0f) ? copysignf(1.0f, interp) : 0.0f;

    // Weight decay + sign update
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * wd);
    p -= lr * s;
    param[idx] = static_cast<scalar_t>(p);

    // EMA update (uses beta2, not beta1)
    stream_store(&exp_avg[idx], beta2 * ea + (1.0f - beta2) * g);
}


void launch_multi_tensor_lion(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float wd
) {
    const int num_params = static_cast<int>(params.size());
    if (num_params == 0) return;

    const auto device = params[0].device();
    const auto dtype = params[0].scalar_type();

    auto d_param_ptrs   = pack_pointers_scalar(params, dtype);
    auto d_exp_avg_ptrs = pack_pointers<float>(exp_avgs);
    auto d_grad_ptrs    = pack_pointers_scalar(grads, dtype);
    auto d_sizes        = pack_sizes(params, device);

    const int max_n = max_numel(params);
    if (max_n == 0) return;

    const int max_blocks = std::min(
        (max_n + MT_BLOCK_SIZE - 1) / MT_BLOCK_SIZE,
        MT_MAX_BLOCKS_PER_PARAM
    );
    dim3 grid(max_blocks, num_params);
    dim3 block(MT_BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dtype, "multi_tensor_lion", ([&] {
            multi_tensor_lion_kernel<scalar_t><<<grid, block>>>(
                d_param_ptrs.data_ptr<int64_t>(),
                d_exp_avg_ptrs.data_ptr<int64_t>(),
                d_grad_ptrs.data_ptr<int64_t>(),
                d_sizes.data_ptr<int32_t>(),
                lr, beta1, beta2, wd
            );
        })
    );
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: Multi-Tensor Grokfast EMA
//
//  In-place gradient filtering and amplification:
//    ema = alpha * ema + (1 - alpha) * grad
//    grad = grad + lamb * ema
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void multi_tensor_grokfast_ema_kernel(
    const int64_t* __restrict__ grad_ptrs,
    const int64_t* __restrict__ ema_ptrs,
    const int* __restrict__ sizes,
    const float alpha,
    const float lamb
) {
    const int param_idx = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizes[param_idx]) return;

    scalar_t* grad = reinterpret_cast<scalar_t*>(grad_ptrs[param_idx]);
    float*    ema  = reinterpret_cast<float*>(ema_ptrs[param_idx]);

    const float g = static_cast<float>(grad[idx]);
    const float e = alpha * stream_load(&ema[idx]) + (1.0f - alpha) * g;
    stream_store(&ema[idx], e);
    grad[idx] = static_cast<scalar_t>(g + lamb * e);
}


void launch_multi_tensor_grokfast_ema(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& emas,
    float alpha, float lamb
) {
    const int num_params = static_cast<int>(grads.size());
    if (num_params == 0) return;

    const auto device = grads[0].device();
    const auto dtype = grads[0].scalar_type();

    auto d_grad_ptrs = pack_pointers_scalar(grads, dtype);
    auto d_ema_ptrs  = pack_pointers<float>(emas);
    auto d_sizes     = pack_sizes(grads, device);

    const int max_n = max_numel(grads);
    if (max_n == 0) return;

    const int max_blocks = std::min(
        (max_n + MT_BLOCK_SIZE - 1) / MT_BLOCK_SIZE,
        MT_MAX_BLOCKS_PER_PARAM
    );
    dim3 grid(max_blocks, num_params);
    dim3 block(MT_BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dtype, "multi_tensor_grokfast_ema", ([&] {
            multi_tensor_grokfast_ema_kernel<scalar_t><<<grid, block>>>(
                d_grad_ptrs.data_ptr<int64_t>(),
                d_ema_ptrs.data_ptr<int64_t>(),
                d_sizes.data_ptr<int32_t>(),
                alpha, lamb
            );
        })
    );
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 4: Multi-Tensor Prodigy Step
//
//  Distance-aware Adam variant with adaptive d_lr scaling.
//  s buffer tracks EMA of d_lr-scaled squared gradients.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void multi_tensor_prodigy_step_kernel(
    const int64_t* __restrict__ param_ptrs,
    const int64_t* __restrict__ exp_avg_ptrs,
    const int64_t* __restrict__ exp_avg_sq_ptrs,
    const int64_t* __restrict__ s_ptrs,
    const int64_t* __restrict__ grad_ptrs,
    const int* __restrict__ sizes,
    const float* __restrict__ bc1,
    const float* __restrict__ bc2,
    const float d_lr,
    const float beta1,
    const float beta2,
    const float lr,
    const float wd,
    const float eps
) {
    const int param_idx = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizes[param_idx]) return;

    const float bc1_val = bc1[param_idx];
    const float bc2_val = bc2[param_idx];

    scalar_t* param       = reinterpret_cast<scalar_t*>(param_ptrs[param_idx]);
    float*    exp_avg     = reinterpret_cast<float*>(exp_avg_ptrs[param_idx]);
    float*    exp_avg_sq  = reinterpret_cast<float*>(exp_avg_sq_ptrs[param_idx]);
    float*    s           = reinterpret_cast<float*>(s_ptrs[param_idx]);
    const scalar_t* grad  = reinterpret_cast<const scalar_t*>(grad_ptrs[param_idx]);

    const float g = static_cast<float>(grad[idx]);
    const float d_lr_g = d_lr * g;
    const float d_lr_g_sq = d_lr * d_lr * g * g;

    // -- 1. Update s (EMA of d_lr-scaled squared gradients) ---------------
    const float s_new = beta2 * stream_load(&s[idx]) + (1.0f - beta2) * d_lr_g_sq;
    stream_store(&s[idx], s_new);

    // -- 2. Adam moment updates -------------------------------------------
    const float ea = beta1 * stream_load(&exp_avg[idx]) + (1.0f - beta1) * d_lr_g;
    const float easq = beta2 * stream_load(&exp_avg_sq[idx]) + (1.0f - beta2) * d_lr_g_sq;
    stream_store(&exp_avg[idx], ea);
    stream_store(&exp_avg_sq[idx], easq);

    // -- 3. Bias-corrected step with weight decay -------------------------
    const float rsqrt_v = fast_rsqrt_nr(easq / bc2_val);
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * d_lr * wd);
    p -= lr * ea * rsqrt_v / (bc1_val * (1.0f + d_lr * eps * rsqrt_v));
    param[idx] = static_cast<scalar_t>(p);
}


void launch_multi_tensor_prodigy_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_bufs,
    std::vector<torch::Tensor>& grads,
    std::vector<float>& bc1s,
    std::vector<float>& bc2s,
    float d_lr, float beta1, float beta2,
    float lr, float wd, float eps
) {
    const int num_params = static_cast<int>(params.size());
    if (num_params == 0) return;

    const auto device = params[0].device();
    const auto dtype = params[0].scalar_type();

    auto d_param_ptrs      = pack_pointers_scalar(params, dtype);
    auto d_exp_avg_ptrs    = pack_pointers<float>(exp_avgs);
    auto d_exp_avg_sq_ptrs = pack_pointers<float>(exp_avg_sqs);
    auto d_s_ptrs          = pack_pointers<float>(s_bufs);
    auto d_grad_ptrs       = pack_pointers_scalar(grads, dtype);
    auto d_sizes           = pack_sizes(params, device);
    auto d_bc1             = pack_scalars(bc1s, device);
    auto d_bc2             = pack_scalars(bc2s, device);

    const int max_n = max_numel(params);
    if (max_n == 0) return;

    const int max_blocks = std::min(
        (max_n + MT_BLOCK_SIZE - 1) / MT_BLOCK_SIZE,
        MT_MAX_BLOCKS_PER_PARAM
    );
    dim3 grid(max_blocks, num_params);
    dim3 block(MT_BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dtype, "multi_tensor_prodigy_step", ([&] {
            multi_tensor_prodigy_step_kernel<scalar_t><<<grid, block>>>(
                d_param_ptrs.data_ptr<int64_t>(),
                d_exp_avg_ptrs.data_ptr<int64_t>(),
                d_exp_avg_sq_ptrs.data_ptr<int64_t>(),
                d_s_ptrs.data_ptr<int64_t>(),
                d_grad_ptrs.data_ptr<int64_t>(),
                d_sizes.data_ptr<int32_t>(),
                d_bc1.data_ptr<float>(),
                d_bc2.data_ptr<float>(),
                d_lr, beta1, beta2, lr, wd, eps
            );
        })
    );
}


// ═══════════════════════════════════════════════════════════════════════
//  Prodigy Fused Reduction + Step (eliminates CPU-GPU sync)
//
//  Three async kernel launches on the same stream:
//    1. Multi-tensor DLR reduction → accumulates num/den to device buffer
//    2. Tiny 1-thread kernel → computes d_lr = max(d_lr_old, |num|/(den+eps))
//    3. Multi-tensor step → reads d_lr from device buffer (not a scalar arg)
// ═══════════════════════════════════════════════════════════════════════

// Phase 1: Multi-tensor reduction across all params
template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void multi_tensor_prodigy_dlr_reduce_kernel(
    const int64_t* __restrict__ param_ptrs,
    const int64_t* __restrict__ param_init_ptrs,
    const int64_t* __restrict__ s_ptrs,
    const int64_t* __restrict__ grad_ptrs,
    const int* __restrict__ sizes,
    float* __restrict__ out_num,
    float* __restrict__ out_den
) {
    constexpr int NUM_WARPS = MT_BLOCK_SIZE / WARP_SIZE;
    __shared__ float shared_num[NUM_WARPS];
    __shared__ float shared_den[NUM_WARPS];

    const int param_idx = blockIdx.y;
    const int N = sizes[param_idx];
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const scalar_t* grad       = reinterpret_cast<const scalar_t*>(grad_ptrs[param_idx]);
    const scalar_t* param      = reinterpret_cast<const scalar_t*>(param_ptrs[param_idx]);
    const scalar_t* param_init = reinterpret_cast<const scalar_t*>(param_init_ptrs[param_idx]);
    const float*    s          = reinterpret_cast<const float*>(s_ptrs[param_idx]);

    float local_num = 0.0f;
    float local_den = 0.0f;

    #pragma unroll 4
    for (int idx = blockIdx.x * blockDim.x + tid; idx < N;
         idx += gridDim.x * blockDim.x) {
        const float g   = static_cast<float>(grad[idx]);
        const float p   = static_cast<float>(param[idx]);
        const float p0  = static_cast<float>(param_init[idx]);
        const float sv  = s[idx];
        local_num += g * (p - p0);
        local_den += sv;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_num += SHFL_DOWN(local_num, offset);
        local_den += SHFL_DOWN(local_den, offset);
    }

    if (lane_id == 0) {
        shared_num[warp_id] = local_num;
        shared_den[warp_id] = local_den;
    }
    __syncthreads();

    if (warp_id == 0) {
        float wn = (lane_id < NUM_WARPS) ? shared_num[lane_id] : 0.0f;
        float wd_val = (lane_id < NUM_WARPS) ? shared_den[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            wn += SHFL_DOWN(wn, offset);
            wd_val += SHFL_DOWN(wd_val, offset);
        }
        if (lane_id == 0) {
            atomicAdd(out_num, wn);
            atomicAdd(out_den, wd_val);
        }
    }
}

// Phase 2: Compute d_lr on device (1 thread)
__global__ void prodigy_compute_dlr_kernel(
    const float* __restrict__ num_acc,
    const float* __restrict__ den_acc,
    float* __restrict__ d_lr_buf,
    const float eps_den
) {
    float num = num_acc[0];
    float den = den_acc[0];
    float d_lr_old = d_lr_buf[0];
    if (den > eps_den) {
        float candidate = fabsf(num) / (den + 1e-12f);
        d_lr_buf[0] = fmaxf(d_lr_old, candidate);
    }
}

// Phase 3: Multi-tensor step reading d_lr from device buffer
template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void multi_tensor_prodigy_step_devptr_kernel(
    const int64_t* __restrict__ param_ptrs,
    const int64_t* __restrict__ exp_avg_ptrs,
    const int64_t* __restrict__ exp_avg_sq_ptrs,
    const int64_t* __restrict__ s_ptrs,
    const int64_t* __restrict__ grad_ptrs,
    const int* __restrict__ sizes,
    const float* __restrict__ bc1,
    const float* __restrict__ bc2,
    const float* __restrict__ d_lr_ptr,
    const float beta1,
    const float beta2,
    const float lr,
    const float wd_val,
    const float eps
) {
    const int param_idx = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizes[param_idx]) return;

    const float d_lr = d_lr_ptr[0];
    const float bc1_val = bc1[param_idx];
    const float bc2_val = bc2[param_idx];

    scalar_t* param       = reinterpret_cast<scalar_t*>(param_ptrs[param_idx]);
    float*    exp_avg     = reinterpret_cast<float*>(exp_avg_ptrs[param_idx]);
    float*    exp_avg_sq  = reinterpret_cast<float*>(exp_avg_sq_ptrs[param_idx]);
    float*    s           = reinterpret_cast<float*>(s_ptrs[param_idx]);
    const scalar_t* grad  = reinterpret_cast<const scalar_t*>(grad_ptrs[param_idx]);

    const float g = static_cast<float>(grad[idx]);
    const float d_lr_g = d_lr * g;
    const float d_lr_g_sq = d_lr * d_lr * g * g;

    const float s_new = beta2 * stream_load(&s[idx]) + (1.0f - beta2) * d_lr_g_sq;
    stream_store(&s[idx], s_new);

    const float ea = beta1 * stream_load(&exp_avg[idx]) + (1.0f - beta1) * d_lr_g;
    const float easq = beta2 * stream_load(&exp_avg_sq[idx]) + (1.0f - beta2) * d_lr_g_sq;
    stream_store(&exp_avg[idx], ea);
    stream_store(&exp_avg_sq[idx], easq);

    const float rsqrt_v = fast_rsqrt_nr(easq / bc2_val);
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * d_lr * wd_val);
    p -= lr * ea * rsqrt_v / (bc1_val * (1.0f + d_lr * eps * rsqrt_v));
    param[idx] = static_cast<scalar_t>(p);
}


// C++ launcher: fused reduction + d_lr compute + step (no CPU sync)
void launch_multi_tensor_prodigy_fused_reduce_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& param_inits,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_bufs,
    std::vector<float>& bc1s,
    std::vector<float>& bc2s,
    torch::Tensor d_lr_buf,
    float beta1, float beta2,
    float lr, float wd, float eps
) {
    const int num_params = static_cast<int>(params.size());
    if (num_params == 0) return;

    const auto device = params[0].device();
    const auto dtype = params[0].scalar_type();

    auto d_param_ptrs      = pack_pointers_scalar(params, dtype);
    auto d_param_init_ptrs = pack_pointers_scalar(param_inits, dtype);
    auto d_exp_avg_ptrs    = pack_pointers<float>(exp_avgs);
    auto d_exp_avg_sq_ptrs = pack_pointers<float>(exp_avg_sqs);
    auto d_s_ptrs          = pack_pointers<float>(s_bufs);
    auto d_grad_ptrs       = pack_pointers_scalar(grads, dtype);
    auto d_sizes           = pack_sizes(params, device);
    auto d_bc1             = pack_scalars(bc1s, device);
    auto d_bc2             = pack_scalars(bc2s, device);

    const int max_n = max_numel(params);
    if (max_n == 0) return;

    // Phase 1: Reduction
    auto num_acc = torch::zeros({1}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto den_acc = torch::zeros({1}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

    const int reduce_blocks = std::min(
        (max_n + MT_BLOCK_SIZE - 1) / MT_BLOCK_SIZE, MT_MAX_BLOCKS_PER_PARAM);
    dim3 reduce_grid(reduce_blocks, num_params);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dtype, "mt_prodigy_reduce", ([&] {
            multi_tensor_prodigy_dlr_reduce_kernel<scalar_t><<<reduce_grid, MT_BLOCK_SIZE>>>(
                d_param_ptrs.data_ptr<int64_t>(),
                d_param_init_ptrs.data_ptr<int64_t>(),
                d_s_ptrs.data_ptr<int64_t>(),
                d_grad_ptrs.data_ptr<int64_t>(),
                d_sizes.data_ptr<int32_t>(),
                num_acc.data_ptr<float>(),
                den_acc.data_ptr<float>()
            );
        })
    );

    // Phase 2: Compute d_lr on device (same stream, no sync)
    prodigy_compute_dlr_kernel<<<1, 1>>>(
        num_acc.data_ptr<float>(),
        den_acc.data_ptr<float>(),
        d_lr_buf.data_ptr<float>(),
        1e-30f
    );

    // Phase 3: Step using device d_lr (same stream, no sync)
    const int step_blocks = std::min(
        (max_n + MT_BLOCK_SIZE - 1) / MT_BLOCK_SIZE, MT_MAX_BLOCKS_PER_PARAM);
    dim3 step_grid(step_blocks, num_params);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dtype, "mt_prodigy_step_devptr", ([&] {
            multi_tensor_prodigy_step_devptr_kernel<scalar_t><<<step_grid, MT_BLOCK_SIZE>>>(
                d_param_ptrs.data_ptr<int64_t>(),
                d_exp_avg_ptrs.data_ptr<int64_t>(),
                d_exp_avg_sq_ptrs.data_ptr<int64_t>(),
                d_s_ptrs.data_ptr<int64_t>(),
                d_grad_ptrs.data_ptr<int64_t>(),
                d_sizes.data_ptr<int32_t>(),
                d_bc1.data_ptr<float>(),
                d_bc2.data_ptr<float>(),
                d_lr_buf.data_ptr<float>(),
                beta1, beta2, lr, wd, eps
            );
        })
    );
}
