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
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace cuda_sm90 {

namespace prim = ::sg::cuda_sm90::primitives;
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
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);

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

}} // namespace sg::cuda_sm90
