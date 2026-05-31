#ifndef GROKKING_KERNELS_SM90_PRODIGY_SM90_CUH_
#define GROKKING_KERNELS_SM90_PRODIGY_SM90_CUH_
// ============================================================================
// prodigy_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'prodigy'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_prodigy.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for Prodigy.
// Algorithm: csrc/algorithms/prodigy.h
//
// Three-kernel orchestration:
//   (1) reduce  — block-reduce (r, s) partial sums
//   (2) update  — single-thread d update
//   (3) apply   — Adam with d as effective learning rate
// All on-device; d_t never round-trips through CPU.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/prodigy.h"
// ── Autotuner-consumable launch parameters (inlined; see compile.py) ──
#ifndef SG_TUNED_BLOCK_SIZE
#define SG_TUNED_BLOCK_SIZE 256
#endif
#ifndef SG_TUNED_VEC_WIDTH
#define SG_TUNED_VEC_WIDTH 4
#endif
#ifndef SG_TUNED_UNROLL
#define SG_TUNED_UNROLL 1
#endif
#ifndef SG_TUNED_ASYNC_DEPTH
#define SG_TUNED_ASYNC_DEPTH 2
#endif
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace sm90 {

namespace prim = ::sg::sm90::primitives;
using ::sg::algorithms::prodigy_partials_step;
using ::sg::algorithms::prodigy_update_d;
using ::sg::algorithms::prodigy_apply_step;

template <typename ParamT, typename GradT>
__global__ void prodigy_reduce_kernel(
    const ParamT* param, const ParamT* param_init, const GradT* grad,
    float d_prev,
    float* r_partial, float* s_partial,
    int N
) {
    float r_local = 0.0f, s_local = 0.0f;
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        prodigy_partials_step(param, param_init, grad, d_prev, i,
                              r_local, s_local);
    }
    float r_block = prim::block_reduce_sum_f32(r_local);
    float s_block = prim::block_reduce_sum_f32(s_local);
    if (threadIdx.x == 0) {
        atomicAdd(r_partial, r_block);
        atomicAdd(s_partial, s_block);
    }
}

__global__ void prodigy_update_d_kernel(
    float* d_t, const float* r_sum, const float* s_sum, float d_prev
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *d_t = prodigy_update_d(d_prev, *r_sum, *s_sum);
    }
}

template <typename ParamT, typename GradT>
__global__ void prodigy_apply_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* s_track,
    const GradT* grad, const float* d_ptr,
    float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const float d = *d_ptr;
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        prodigy_apply_step(param, exp_avg, exp_avg_sq, s_track, grad,
                           d, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

void launch_prodigy_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& s_track,
    const torch::Tensor& param_init,
    const torch::Tensor& grad,
    torch::Tensor& d_t,
    torch::Tensor& r_partial,
    torch::Tensor& s_partial,
    float d_prev,
    float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);

    r_partial.zero_();
    s_partial.zero_();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "prodigy_reduce", [&] {
            prodigy_reduce_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                param_init.data_ptr<scalar_t>(),
                grad.data_ptr<scalar_t>(),
                d_prev,
                r_partial.data_ptr<float>(),
                s_partial.data_ptr<float>(),
                N);
        });

    prodigy_update_d_kernel<<<1, 1, 0, stream>>>(
        d_t.data_ptr<float>(),
        r_partial.data_ptr<float>(),
        s_partial.data_ptr<float>(),
        d_prev);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "prodigy_apply", [&] {
            prodigy_apply_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                s_track.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                d_t.data_ptr<float>(),
                beta1, beta2, eps, wd, bc1, bc2, N);
        });
}


// Per-tensor Prodigy step: d_lr is the current adaptive learning rate.
void launch_fused_prodigy_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor s, torch::Tensor param_init, torch::Tensor grad,
    float lr, float d_lr, float beta1, float beta2,
    float weight_decay, float eps, float bc1, float bc2
) {
    auto d_t = torch::tensor({d_lr},
        torch::TensorOptions().device(param.device()).dtype(torch::kFloat32));
    auto r_partial = torch::zeros({1},
        torch::TensorOptions().device(param.device()).dtype(torch::kFloat32));
    auto s_partial = torch::zeros({1},
        torch::TensorOptions().device(param.device()).dtype(torch::kFloat32));
    launch_prodigy_step(param, exp_avg, exp_avg_sq, s, param_init, grad,
                        d_t, r_partial, s_partial, d_lr,
                        beta1, beta2, eps, weight_decay, bc1, bc2);
}

// DLR reduction: numerator += <grad, param - param_init>,
// denominator += ||s|| * d + eps.
__global__ void prodigy_dlr_reduce_kernel(
    const float* grad, const float* param, const float* param_init,
    const float* s, float* numerator, float* denominator,
    float eps, int N
) {
    float num_acc = 0.0f, den_acc = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        num_acc += grad[i] * (param[i] - param_init[i]);
        den_acc += fabsf(s[i]);
    }
    atomicAdd(numerator, num_acc);
    atomicAdd(denominator, den_acc);
}

void launch_prodigy_dlr_reduce(
    torch::Tensor grad, torch::Tensor param, torch::Tensor param_init,
    torch::Tensor s, torch::Tensor numerator, torch::Tensor denominator,
    float eps
) {
    const int64_t N = grad.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);
    prodigy_dlr_reduce_kernel<<<grid, block, 0, stream>>>(
        grad.data_ptr<float>(), param.data_ptr<float>(),
        param_init.data_ptr<float>(), s.data_ptr<float>(),
        numerator.data_ptr<float>(), denominator.data_ptr<float>(),
        eps, N);
}

// Multi-tensor fused reduce + step: accumulate d_lr across all tensors,
// then apply the Prodigy update to each.
void launch_multi_tensor_prodigy_fused_reduce_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& param_inits,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_bufs,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    torch::Tensor d_lr_buf,
    float beta1, float beta2, float lr, float wd, float eps
) {
    if (params.empty()) return;
    auto dev = params[0].device();
    auto r_partial = torch::zeros({1},
        torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    auto s_partial = torch::zeros({1},
        torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    float d_prev = d_lr_buf.item<float>();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;

    for (size_t i = 0; i < params.size(); i++) {
        const int64_t N = params[i].numel();
        if (N == 0) continue;
        const int grid = std::min<int>(65535, (N + block - 1) / block);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            params[i].scalar_type(), "prodigy_mt_reduce", [&] {
                prodigy_reduce_kernel<scalar_t, scalar_t>
                    <<<grid, block, 0, stream>>>(
                    params[i].data_ptr<scalar_t>(),
                    param_inits[i].data_ptr<scalar_t>(),
                    grads[i].data_ptr<scalar_t>(),
                    d_prev,
                    r_partial.data_ptr<float>(),
                    s_partial.data_ptr<float>(), N);
            });
    }

    prodigy_update_d_kernel<<<1, 1, 0, stream>>>(
        d_lr_buf.data_ptr<float>(),
        r_partial.data_ptr<float>(),
        s_partial.data_ptr<float>(), d_prev);

    for (size_t i = 0; i < params.size(); i++) {
        const int64_t N = params[i].numel();
        if (N == 0) continue;
        const int grid = std::min<int>(65535, (N + block - 1) / block);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            params[i].scalar_type(), "prodigy_mt_apply", [&] {
                prodigy_apply_kernel<scalar_t, scalar_t>
                    <<<grid, block, 0, stream>>>(
                    params[i].data_ptr<scalar_t>(),
                    exp_avgs[i].data_ptr<float>(),
                    exp_avg_sqs[i].data_ptr<float>(),
                    s_bufs[i].data_ptr<float>(),
                    grads[i].data_ptr<scalar_t>(),
                    d_lr_buf.data_ptr<float>(),
                    beta1, beta2, eps, wd, bc1s[i], bc2s[i], N);
            });
    }
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_PRODIGY_SM90_CUH_
