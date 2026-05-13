// CUDA sm_90 launch glue for Muon.
// Algorithm: csrc/algorithms/muon.h
//
// Muon has multiple per-element kernels (momentum_normalize, ns_combine,
// update) plus matrix multiplications for the Newton-Schulz iteration,
// which are handled via CUTLASS (mma.cuh) when -DWITH_CUTLASS is set,
// or torch::mm (cuBLAS) otherwise. The orchestration loop runs host-side.
//
// 1D parameters fall back to AdamW via launch_adamw.cu.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/muon.h"
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace cuda_sm90 {

namespace prim = ::sg::cuda_sm90::primitives;
using ::sg::algorithms::muon_momentum_normalize_step;
using ::sg::algorithms::muon_ns_combine_step;
using ::sg::algorithms::muon_update_step;

template <typename GradT>
__global__ void muon_momentum_normalize_kernel(
    float* buf, float* X, const GradT* grad,
    float momentum, float inv_norm, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        muon_momentum_normalize_step(buf, X, grad, momentum, inv_norm, i);
    }
}

__global__ void muon_ns_combine_kernel(
    float* Y, const float* X, const float* AX, const float* AAX,
    float a, float b, float c, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        muon_ns_combine_step(Y, X, AX, AAX, a, b, c, i);
    }
}

template <typename ParamT>
__global__ void muon_update_kernel(
    ParamT* param, const float* orth,
    float neg_lr_scale, float decay_factor, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        muon_update_step(param, orth, neg_lr_scale, decay_factor, i);
    }
}

void launch_muon_momentum_normalize(
    torch::Tensor& buf, torch::Tensor& X, const torch::Tensor& grad,
    float momentum, float inv_norm
) {
    const int64_t N = buf.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "muon_momentum_normalize", [&] {
            muon_momentum_normalize_kernel<scalar_t><<<grid, block, 0, stream>>>(
                buf.data_ptr<float>(), X.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                momentum, inv_norm, N);
        });
}

void launch_muon_ns_combine(
    torch::Tensor& Y, const torch::Tensor& X,
    const torch::Tensor& AX, const torch::Tensor& AAX,
    float a, float b, float c
) {
    const int64_t N = Y.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);
    muon_ns_combine_kernel<<<grid, block, 0, stream>>>(
        Y.data_ptr<float>(), X.data_ptr<float>(),
        AX.data_ptr<float>(), AAX.data_ptr<float>(),
        a, b, c, N);
}

void launch_muon_update(
    torch::Tensor& param, const torch::Tensor& orth,
    float neg_lr_scale, float decay_factor
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = 256;
    const int grid = std::min<int>(8192, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "muon_update", [&] {
            muon_update_kernel<scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                orth.data_ptr<float>(),
                neg_lr_scale, decay_factor, N);
        });
}

}} // namespace sg::cuda_sm90
