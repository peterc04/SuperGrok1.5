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
#include "grokking_optimizers/kernels/sm_90/common_sm90.cuh"

namespace sg { namespace sm90 {

namespace prim = ::sg::sm90::primitives;
using ::sg::algorithms::muon_momentum_normalize_step;
using ::sg::algorithms::muon_ns_combine_step;
using ::sg::algorithms::muon_update_step;

#ifndef SG_MUON_MIN_BLOCKS
#define SG_MUON_MIN_BLOCKS 4
#endif

template <typename GradT, int UNROLL>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_MUON_MIN_BLOCKS)
muon_momentum_normalize_kernel(
    float* buf, float* X, const GradT* grad,
    float momentum, float inv_norm, int64_t N
) {
    const int64_t stride = prim::grid_stride() * UNROLL;
    const int64_t base0 = prim::grid_stride_index() * UNROLL;
    for (int64_t base = base0; base < N; base += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int64_t i = base + u;
            if (i < N) {
                SG_SANITIZE_GRAD_INPLACE(grad, i);
                muon_momentum_normalize_step(buf, X, grad, momentum, inv_norm, i);
            }
        }
    }
}

// FP32 vec4 momentum-normalize: float4 traffic, canonical step CALLED 4×.
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_MUON_MIN_BLOCKS)
muon_momentum_normalize_kernel_vec4_fp32(
    float4* buf4, float4* X4, const float4* grad4,
    float momentum, float inv_norm, int64_t N4
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N4; i += stride) {
        float4 b = prim::ld_f32v4(buf4 + i);
        float4 x = prim::ld_f32v4(X4 + i);
        float4 g = prim::ldg_f32v4(grad4 + i);
        ::grokking::sm90::sg_sanitize_grad4(g);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            muon_momentum_normalize_step(&b.x, &x.x, &g.x, momentum, inv_norm, u);
        }
        prim::st_f32v4(buf4 + i, b);
        prim::st_f32v4(X4 + i, x);
    }
}

__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_MUON_MIN_BLOCKS)
muon_ns_combine_kernel(
    float* Y, const float* X, const float* AX, const float* AAX,
    float a, float b, float c, int64_t N
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        muon_ns_combine_step(Y, X, AX, AAX, a, b, c, i);
    }
}

template <typename ParamT, int UNROLL>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_MUON_MIN_BLOCKS)
muon_update_kernel(
    ParamT* param, const float* orth,
    float neg_lr_scale, float decay_factor, int64_t N
) {
    const int64_t stride = prim::grid_stride() * UNROLL;
    const int64_t base0 = prim::grid_stride_index() * UNROLL;
    for (int64_t base = base0; base < N; base += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int64_t i = base + u;
            if (i < N) {
                muon_update_step(param, orth, neg_lr_scale, decay_factor, i);
            }
        }
    }
}

// FP32 vec4 update: float4 traffic, canonical muon_update_step CALLED 4×.
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_MUON_MIN_BLOCKS)
muon_update_kernel_vec4_fp32(
    float4* param4, const float4* orth4,
    float neg_lr_scale, float decay_factor, int64_t N4
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N4; i += stride) {
        float4 p = prim::ld_f32v4(param4 + i);
        float4 o = prim::ldg_f32v4(orth4 + i);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            muon_update_step(&p.x, &o.x, neg_lr_scale, decay_factor, u);
        }
        prim::st_f32v4(param4 + i, p);
    }
}

void launch_muon_momentum_normalize(
    torch::Tensor buf, torch::Tensor X, torch::Tensor grad,
    float momentum, float inv_norm
) {
    const int64_t N = buf.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // §6.1: keep the Muon momentum buffer L2-resident across the step.
    prim::L2PersistScope l2(stream, buf.data_ptr(), buf.nbytes());

    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    if (SG_TUNED_VEC_WIDTH == 4 && grad.scalar_type() == torch::kFloat32 &&
        prim::is_vec4_alignable(buf.data_ptr(), N) &&
        prim::is_vec4_alignable(X.data_ptr(), N) &&
        prim::is_vec4_alignable(grad.data_ptr(), N)) {
        const int64_t N4 = N / 4;
        const int grid4 = static_cast<int>(
            std::min<int64_t>(65535, (N4 + block - 1) / block));
        muon_momentum_normalize_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(buf.data_ptr<float>()),
            reinterpret_cast<float4*>(X.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            momentum, inv_norm, N4);
        SG_LAUNCH_CHECK(stream);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "muon_momentum_normalize", [&] {
            muon_momentum_normalize_kernel<scalar_t, SG_TUNED_UNROLL>
                <<<grid, block, 0, stream>>>(
                buf.data_ptr<float>(), X.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                momentum, inv_norm, N);
            SG_LAUNCH_CHECK(stream);
        });
}

void launch_muon_ns_combine(
    torch::Tensor Y, torch::Tensor X,
    torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c
) {
    const int64_t N = Y.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));
    muon_ns_combine_kernel<<<grid, block, 0, stream>>>(
        Y.data_ptr<float>(), X.data_ptr<float>(),
        AX.data_ptr<float>(), AAX.data_ptr<float>(),
        a, b, c, N);
    SG_LAUNCH_CHECK(stream);
}

void launch_muon_update(
    torch::Tensor param, torch::Tensor orth,
    float neg_lr_scale, float decay_factor
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    if (SG_TUNED_VEC_WIDTH == 4 && param.scalar_type() == torch::kFloat32 &&
        prim::is_vec4_alignable(param.data_ptr(), N) &&
        prim::is_vec4_alignable(orth.data_ptr(), N)) {
        const int64_t N4 = N / 4;
        const int grid4 = static_cast<int>(
            std::min<int64_t>(65535, (N4 + block - 1) / block));
        muon_update_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<const float4*>(orth.data_ptr<float>()),
            neg_lr_scale, decay_factor, N4);
        SG_LAUNCH_CHECK(stream);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "muon_update", [&] {
            muon_update_kernel<scalar_t, SG_TUNED_UNROLL>
                <<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                orth.data_ptr<float>(),
                neg_lr_scale, decay_factor, N);
            SG_LAUNCH_CHECK(stream);
        });
}


// Fused Newton-Schulz combine + parameter update in a single pass.
// Y = a*X + b*AX + c*AAX (NS combine), then param -= lr_scale*Y + decay*param.
template <typename ParamT>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_MUON_MIN_BLOCKS)
muon_ns_combine_update_kernel(
    ParamT* param, const float* X, const float* AX, const float* AAX,
    float a, float b, float c, float neg_lr_scale, float decay_factor, int64_t N
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
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
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "muon_ns_combine_update", [&] {
            muon_ns_combine_update_kernel<scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                X.data_ptr<float>(), AX.data_ptr<float>(), AAX.data_ptr<float>(),
                a, b, c, neg_lr_scale, decay_factor, N);
            SG_LAUNCH_CHECK(stream);
        });
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_MUON_SM90_CUH_
