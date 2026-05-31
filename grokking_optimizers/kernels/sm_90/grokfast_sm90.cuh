#ifndef GROKKING_KERNELS_SM90_GROKFAST_SM90_CUH_
#define GROKKING_KERNELS_SM90_GROKFAST_SM90_CUH_
// ============================================================================
// grokfast_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'grokfast'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_grokfast.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for Grokfast (fused EMA + Adam path).
// Algorithm: csrc/algorithms/grokfast.h

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/grokfast.h"
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
using ::sg::algorithms::grokfast_fused_step;

template <typename ParamT, typename GradT>
__global__ void grokfast_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* ema,
    const GradT* grad,
    float gf_alpha, float gf_lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        grokfast_fused_step(param, exp_avg, exp_avg_sq, ema, grad,
                            gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd,
                            bc1, bc2, i);
    }
}

void launch_grokfast_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& ema,
    const torch::Tensor& grad,
    float gf_alpha, float gf_lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "grokfast_step", [&] {
            grokfast_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                ema.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}


// EMA-only update: ema = alpha * ema + (1 - alpha) * grad.
// No Adam step; used as a sub-operation by the fused path.
__global__ void grokfast_ema_kernel(
    float* grad, float* ema, float alpha, float lamb, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        float g = grad[i];
        float e = alpha * ema[i] + (1.0f - alpha) * g;
        ema[i] = e;
        grad[i] = g + lamb * e;
    }
}

void launch_fused_grokfast_ema(
    torch::Tensor grad, torch::Tensor ema, float alpha, float lamb
) {
    const int64_t N = grad.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);
    grokfast_ema_kernel<<<grid, block, 0, stream>>>(
        grad.data_ptr<float>(), ema.data_ptr<float>(), alpha, lamb, N);
}

void launch_fused_grokfast_adam(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2
) {
    launch_grokfast_step(param, exp_avg, exp_avg_sq, ema, grad,
                         alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                         bc1, bc2);
}

void launch_multi_tensor_grokfast_ema(
    std::vector<torch::Tensor>& grads, std::vector<torch::Tensor>& ema_bufs,
    float alpha, float lamb
) {
    for (size_t i = 0; i < grads.size(); i++) {
        launch_fused_grokfast_ema(grads[i], ema_bufs[i], alpha, lamb);
    }
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_GROKFAST_SM90_CUH_
