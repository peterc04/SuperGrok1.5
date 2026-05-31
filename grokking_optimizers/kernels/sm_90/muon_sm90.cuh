#ifndef GROKKING_KERNELS_SM90_MUON_SM90_CUH_
#define GROKKING_KERNELS_SM90_MUON_SM90_CUH_
// ============================================================================
// muon_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'muon'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_muon.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
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
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);

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
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);
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
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "muon_update", [&] {
            muon_update_kernel<scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                orth.data_ptr<float>(),
                neg_lr_scale, decay_factor, N);
        });
}


// Fused Newton-Schulz combine + parameter update in a single pass.
// Y = a*X + b*AX + c*AAX (NS combine), then param -= lr_scale*Y + decay*param.
template <typename ParamT>
__global__ void muon_ns_combine_update_kernel(
    ParamT* param, const float* X, const float* AX, const float* AAX,
    float a, float b, float c, float neg_lr_scale, float decay_factor, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        float y = a * X[i] + b * AX[i] + c * AAX[i];
        float p = static_cast<float>(param[i]);
        param[i] = static_cast<ParamT>(p + neg_lr_scale * y - decay_factor * p);
    }
}

void launch_muon_ns_combine_update_fused(
    torch::Tensor param, torch::Tensor X, torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c, float neg_lr_scale, float decay_factor
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "muon_ns_combine_update", [&] {
            muon_ns_combine_update_kernel<scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                X.data_ptr<float>(), AX.data_ptr<float>(), AAX.data_ptr<float>(),
                a, b, c, neg_lr_scale, decay_factor, N);
        });
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_MUON_SM90_CUH_
