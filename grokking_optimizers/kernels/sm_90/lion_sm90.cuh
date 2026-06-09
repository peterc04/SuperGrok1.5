#ifndef GROKKING_KERNELS_SM90_LION_SM90_CUH_
#define GROKKING_KERNELS_SM90_LION_SM90_CUH_
// ============================================================================
// lion_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'lion'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_lion.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for Lion.
// Algorithm: csrc/algorithms/lion.h

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/lion.h"
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
#include "grokking_optimizers/kernels/sm_90/common_sm90.cuh"

namespace sg { namespace sm90 {

namespace prim = ::sg::sm90::primitives;
using ::sg::algorithms::lion_step;
using ::sg::algorithms::lion_step_vec4;

// Minimum resident blocks/SM for the bandwidth-bound element-wise applies.
// Caps registers so occupancy stays high on the memory-bound path.
#ifndef SG_LION_MIN_BLOCKS
#define SG_LION_MIN_BLOCKS 4
#endif

// SG_TUNED_UNROLL elements per iteration; the canonical lion_step is CALLED
// (math single-sourced) — the unroll only changes the loop structure the
// autotuner generates, not the math.
template <typename ParamT, typename GradT, int UNROLL>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_LION_MIN_BLOCKS)
lion_kernel(
    ParamT* param, float* exp_avg,
    const GradT* grad,
    float lr, float beta1, float beta2, float wd, int64_t N
) {
    const int64_t stride = prim::grid_stride() * UNROLL;
    const int64_t base0 = prim::grid_stride_index() * UNROLL;
    for (int64_t base = base0; base < N; base += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int64_t i = base + u;
            if (i < N) {
                SG_SANITIZE_GRAD_INPLACE(grad, i);
                lion_step(param, exp_avg, grad, lr, beta1, beta2, wd, i);
            }
        }
    }
}

__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_LION_MIN_BLOCKS)
lion_kernel_vec4_fp32(
    float4* param4, float4* exp_avg4, const float4* grad4,
    float lr, float beta1, float beta2, float wd, int64_t N4
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N4; i += stride) {
        SG_SANITIZE_GRAD4_INPLACE(grad4, i);
        lion_step_vec4(param4, exp_avg4, grad4, lr, beta1, beta2, wd, i);
    }
}

void launch_lion_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    const torch::Tensor& grad,
    float lr, float beta1, float beta2, float wd
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // §6.1: keep the Lion momentum buffer L2-resident across the step.
    prim::L2PersistScope l2(stream, exp_avg.data_ptr(), exp_avg.nbytes());

    const int block = SG_TUNED_BLOCK_SIZE;
    // Clamp the grid in int64 (N can exceed 2^31) then cast the grid dim to int.
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    const bool all_fp32 = param.scalar_type() == torch::kFloat32 &&
                          grad.scalar_type() == torch::kFloat32;

    if (SG_TUNED_VEC_WIDTH == 4 && all_fp32 &&
        prim::is_vec4_alignable(param.data_ptr(), N) &&
        prim::is_vec4_alignable(grad.data_ptr(), N)) {
        const int64_t N4 = N / 4;
        const int grid4 = static_cast<int>(
            std::min<int64_t>(65535, (N4 + block - 1) / block));
        lion_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            lr, beta1, beta2, wd, N4);
        SG_LAUNCH_CHECK(stream);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "lion_step", [&] {
            lion_kernel<scalar_t, scalar_t, SG_TUNED_UNROLL>
                <<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                lr, beta1, beta2, wd, N);
            SG_LAUNCH_CHECK(stream);
        });
}


void launch_fused_lion_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,
    float lr, float beta1, float beta2, float weight_decay
) {
    launch_lion_step(param, exp_avg, grad, lr, beta1, beta2, weight_decay);
}

void launch_multi_tensor_lion(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float wd
) {
    for (size_t i = 0; i < params.size(); i++) {
        launch_lion_step(params[i], exp_avgs[i], grads[i],
                         lr, beta1, beta2, wd);
    }
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_LION_SM90_CUH_
