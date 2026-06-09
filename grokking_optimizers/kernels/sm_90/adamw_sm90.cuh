#ifndef GROKKING_KERNELS_SM90_ADAMW_SM90_CUH_
#define GROKKING_KERNELS_SM90_ADAMW_SM90_CUH_
// ============================================================================
// adamw_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'adamw'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_adamw.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for AdamW.
//
// Includes:
//   csrc/algorithms/adamw.h           — per-element math
//   csrc/backends/cuda/sm_90/primitives.cuh — grid-stride, vec4, reductions
//
// Exposes:
//   sg::sm90::launch_adamw_step(...)   — multi-tensor launcher

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/adamw.h"
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
using ::sg::algorithms::adamw_step;
using ::sg::algorithms::adamw_step_vec4;

// Minimum resident blocks/SM for the bandwidth-bound element-wise applies.
// Caps registers so occupancy stays high on the memory-bound path.
#ifndef SG_ADAMW_MIN_BLOCKS
#define SG_ADAMW_MIN_BLOCKS 4
#endif

// =========================================================================
//  Scalar grid-stride kernel. SG_TUNED_UNROLL elements per iteration; the
//  canonical adamw_step is CALLED (math single-sourced) — the unroll only
//  changes the loop structure the autotuner generates, not the math.
// =========================================================================

template <typename ParamT, typename GradT, int UNROLL>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_ADAMW_MIN_BLOCKS)
adamw_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const GradT* grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N
) {
    const int64_t stride = prim::grid_stride() * UNROLL;
    const int64_t base0 = prim::grid_stride_index() * UNROLL;
    for (int64_t base = base0; base < N; base += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int64_t i = base + u;
            if (i < N) {
                SG_SANITIZE_GRAD_INPLACE(grad, i);
                adamw_step(param, exp_avg, exp_avg_sq, grad,
                           lr, beta1, beta2, eps, wd, bc1, bc2, i);
            }
        }
    }
}

// =========================================================================
//  FP32 vec4 fast path (when param, grad, both states are FP32 and 16B-aligned)
// =========================================================================

__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_ADAMW_MIN_BLOCKS)
adamw_kernel_vec4_fp32(
    float4* param4, float4* exp_avg4, float4* exp_avg_sq4,
    const float4* grad4,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N4
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N4; i += stride) {
        SG_SANITIZE_GRAD4_INPLACE(grad4, i);
        adamw_step_vec4(param4, exp_avg4, exp_avg_sq4, grad4,
                        lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

// =========================================================================
//  Host-side launcher (per-tensor)
// =========================================================================

void launch_adamw_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const torch::Tensor& grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    // Clamp the grid in int64 (N can exceed 2^31) then cast the grid dim to int
    // (grid dims are int; the element index inside the kernel is int64_t).
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    // §6.1: keep the Adam moments (m, v) L2-resident across the step.
    prim::L2PersistScope l2(stream,
        exp_avg.data_ptr(), exp_avg.nbytes(),
        exp_avg_sq.data_ptr(), exp_avg_sq.nbytes());

    const bool all_fp32 = param.scalar_type() == torch::kFloat32 &&
                          grad.scalar_type() == torch::kFloat32;

    if (SG_TUNED_VEC_WIDTH == 4 && all_fp32 &&
        prim::is_vec4_alignable(param.data_ptr(), N) &&
        prim::is_vec4_alignable(grad.data_ptr(), N)) {
        const int64_t N4 = N / 4;
        const int grid4 = static_cast<int>(
            std::min<int64_t>(65535, (N4 + block - 1) / block));
        adamw_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            lr, beta1, beta2, eps, wd, bc1, bc2, N4);
        SG_LAUNCH_CHECK(stream);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "adamw_step", [&] {
            adamw_kernel<scalar_t, scalar_t, SG_TUNED_UNROLL>
                <<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                lr, beta1, beta2, eps, wd, bc1, bc2, N);
            SG_LAUNCH_CHECK(stream);
        });
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_ADAMW_SM90_CUH_
