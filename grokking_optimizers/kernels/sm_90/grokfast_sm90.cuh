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
#include "grokking_optimizers/kernels/sm_90/common_sm90.cuh"

namespace sg { namespace sm90 {

namespace prim = ::sg::sm90::primitives;
using ::sg::algorithms::grokfast_fused_step;

// Minimum resident blocks/SM for the bandwidth-bound element-wise applies.
// Caps registers so occupancy stays high on the memory-bound path.
#ifndef SG_GROKFAST_MIN_BLOCKS
#define SG_GROKFAST_MIN_BLOCKS 4
#endif

// SG_TUNED_UNROLL-parameterized scalar grid-stride kernel. Each iteration
// processes UNROLL elements; the canonical per-element grokfast_fused_step is
// CALLED (math single-sourced in algorithms/grokfast.h) — the unroll only
// changes the loop structure the autotuner generates, not the math.
template <typename ParamT, typename GradT, int UNROLL>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_GROKFAST_MIN_BLOCKS)
grokfast_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* ema,
    const GradT* grad,
    float gf_alpha, float gf_lamb,
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
                grokfast_fused_step(param, exp_avg, exp_avg_sq, ema, grad,
                                    gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd,
                                    bc1, bc2, i);
            }
        }
    }
}

// FP32 vec4 fast path. Loads param/state/grad as float4 (one 128-bit
// transaction per 4 elements), then CALLS the canonical scalar
// grokfast_fused_step 4× on register-resident lanes — the math is NOT
// re-typed here (single-source guard), only the global traffic is widened.
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_GROKFAST_MIN_BLOCKS)
grokfast_kernel_vec4_fp32(
    float4* param4, float4* exp_avg4, float4* exp_avg_sq4, float4* ema4,
    const float4* grad4,
    float gf_alpha, float gf_lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N4
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N4; i += stride) {
        float4 p  = prim::ld_f32v4(param4 + i);
        float4 m  = prim::ld_f32v4(exp_avg4 + i);
        float4 v  = prim::ld_f32v4(exp_avg_sq4 + i);
        float4 e  = prim::ld_f32v4(ema4 + i);
        float4 g  = prim::ldg_f32v4(grad4 + i);
        ::grokking::sm90::sg_sanitize_grad4(g);
        // Call the canonical per-element step 4× on the register lanes.
        grokfast_fused_step(&p.x, &m.x, &v.x, &e.x, &g.x,
                            gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd,
                            bc1, bc2, 0);
        grokfast_fused_step(&p.x, &m.x, &v.x, &e.x, &g.x,
                            gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd,
                            bc1, bc2, 1);
        grokfast_fused_step(&p.x, &m.x, &v.x, &e.x, &g.x,
                            gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd,
                            bc1, bc2, 2);
        grokfast_fused_step(&p.x, &m.x, &v.x, &e.x, &g.x,
                            gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd,
                            bc1, bc2, 3);
        prim::st_f32v4(param4 + i, p);
        prim::st_f32v4(exp_avg4 + i, m);
        prim::st_f32v4(exp_avg_sq4 + i, v);
        prim::st_f32v4(ema4 + i, e);
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

    // §6.1: keep the Adam moments (m, v) L2-resident across the step.
    prim::L2PersistScope l2(stream,
        exp_avg.data_ptr(), exp_avg.nbytes(),
        exp_avg_sq.data_ptr(), exp_avg_sq.nbytes());

    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));

    const bool all_fp32 = param.scalar_type() == torch::kFloat32 &&
                          grad.scalar_type() == torch::kFloat32;

    // Autotuner picks the vector width: width 4 + 16B-aligned all-FP32 takes
    // the float4 fast path; otherwise the SG_TUNED_UNROLL scalar path runs.
    if (SG_TUNED_VEC_WIDTH == 4 && all_fp32 &&
        prim::is_vec4_alignable(param.data_ptr(), N) &&
        prim::is_vec4_alignable(exp_avg.data_ptr(), N) &&
        prim::is_vec4_alignable(exp_avg_sq.data_ptr(), N) &&
        prim::is_vec4_alignable(ema.data_ptr(), N) &&
        prim::is_vec4_alignable(grad.data_ptr(), N)) {
        const int64_t N4 = N / 4;
        const int grid4 = static_cast<int>(
            std::min<int64_t>(65535, (N4 + block - 1) / block));
        grokfast_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            reinterpret_cast<float4*>(ema.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd, bc1, bc2, N4);
        SG_LAUNCH_CHECK(stream);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "grokfast_step", [&] {
            grokfast_kernel<scalar_t, scalar_t, SG_TUNED_UNROLL>
                <<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                ema.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd, bc1, bc2, N);
            SG_LAUNCH_CHECK(stream);
        });
}


// EMA-only update: ema = alpha * ema + (1 - alpha) * grad.
// No Adam step; used as a sub-operation by the fused path.
__global__ void grokfast_ema_kernel(
    float* grad, float* ema, float alpha, float lamb, int64_t N
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        float g = ::grokking::sm90::sg_sanitize_grad(grad[i]);
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
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));
    grokfast_ema_kernel<<<grid, block, 0, stream>>>(
        grad.data_ptr<float>(), ema.data_ptr<float>(), alpha, lamb, N);
    SG_LAUNCH_CHECK(stream);
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
