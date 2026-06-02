#ifndef GROKKING_KERNELS_SM90_NEURALGROK_SM90_CUH_
#define GROKKING_KERNELS_SM90_NEURALGROK_SM90_CUH_
// ============================================================================
// neuralgrok_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'neuralgrok'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_neuralgrok.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for NeuralGrok.
// Algorithm: csrc/algorithms/neuralgrok.h
//
// Two-stage compute: psi_net forward (per-element MLP on |grad|), then
// Adam apply with the amplified gradient. The psi weights live in constant
// memory; the launcher copies host-side weights into the constant buffer
// before kernel launch.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/neuralgrok.h"
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
using ::sg::algorithms::neuralgrok_psi_forward;
using ::sg::algorithms::neuralgrok_apply_step;
using ::sg::algorithms::neuralgrok_adam_tail;

// Compile-time hidden width specializations (H = 8, 16, 32, 64, 128).
constexpr int NG_H = 64;

// Minimum resident blocks/SM. The psi-net forward is compute-heavy per element
// but the bound is still memory traffic on the Adam state; cap registers so
// occupancy stays high.
#ifndef SG_NEURALGROK_MIN_BLOCKS
#define SG_NEURALGROK_MIN_BLOCKS 4
#endif

// Cooperatively stage the per-element psi-net weights (W1[H], b1[H], W2[H]) into
// shared memory ONCE per block, then hand the shared pointers to the canonical
// neuralgrok_psi_forward<H>. The same weights are read by every element this
// block processes; staging them removes the per-element re-read from GLOBAL
// inside the H-wide forward loop. NO math/signature change — the fn still takes
// const float* and is single-sourced in algorithms/neuralgrok.h.
template <int H>
__device__ __forceinline__ void ng_stage_psi_weights(
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2,
    float* sW1, float* sb1, float* sW2
) {
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        sW1[j] = W1[j];
        sb1[j] = b1[j];
        sW2[j] = W2[j];
    }
    __syncthreads();
}

template <typename ParamT, typename GradT, int UNROLL>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_NEURALGROK_MIN_BLOCKS)
neuralgrok_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const GradT* grad,
    const float* W1, const float* b1, const float* W2, float b2,
    float alpha, float beta,
    float lr, float beta1_a, float beta2_a, float eps, float wd,
    float bc1, float bc2, int N
) {
    __shared__ float sW1[NG_H];
    __shared__ float sb1[NG_H];
    __shared__ float sW2[NG_H];
    ng_stage_psi_weights<NG_H>(W1, b1, W2, sW1, sb1, sW2);

    const int stride = prim::grid_stride() * UNROLL;
    const int base0 = prim::grid_stride_index() * UNROLL;
    for (int base = base0; base < N; base += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int i = base + u;
            if (i < N) {
                const float ag = fabsf(static_cast<float>(grad[i]));
                const float s = neuralgrok_psi_forward<NG_H>(ag, sW1, sb1, sW2, b2);
                neuralgrok_apply_step(param, exp_avg, exp_avg_sq, grad, s,
                                      alpha, beta, lr, beta1_a, beta2_a, eps, wd,
                                      bc1, bc2, i);
            }
        }
    }
}

// FP32 vec4 fast path. Stages the psi weights into shared once, then loads
// param/state/grad as float4 (one 128-bit transaction per 4 elements) and CALLS
// the canonical neuralgrok_psi_forward<H> + neuralgrok_apply_step 4× on the
// register lanes. The math is NOT re-typed here (single-source guard), only the
// global traffic is widened and the weights are shared-staged.
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_NEURALGROK_MIN_BLOCKS)
neuralgrok_kernel_vec4_fp32(
    float4* param4, float4* exp_avg4, float4* exp_avg_sq4,
    const float4* grad4,
    const float* W1, const float* b1, const float* W2, float b2,
    float alpha, float beta,
    float lr, float beta1_a, float beta2_a, float eps, float wd,
    float bc1, float bc2, int N4
) {
    __shared__ float sW1[NG_H];
    __shared__ float sb1[NG_H];
    __shared__ float sW2[NG_H];
    ng_stage_psi_weights<NG_H>(W1, b1, W2, sW1, sb1, sW2);

    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N4; i += stride) {
        float4 p = prim::ld_f32v4(param4 + i);
        float4 m = prim::ld_f32v4(exp_avg4 + i);
        float4 v = prim::ld_f32v4(exp_avg_sq4 + i);
        float4 g = prim::ldg_f32v4(grad4 + i);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float ag = fabsf((&g.x)[u]);
            const float s = neuralgrok_psi_forward<NG_H>(ag, sW1, sb1, sW2, b2);
            neuralgrok_apply_step(&p.x, &m.x, &v.x, &g.x, s,
                                  alpha, beta, lr, beta1_a, beta2_a, eps, wd,
                                  bc1, bc2, u);
        }
        prim::st_f32v4(param4 + i, p);
        prim::st_f32v4(exp_avg4 + i, m);
        prim::st_f32v4(exp_avg_sq4 + i, v);
    }
}

void launch_neuralgrok_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const torch::Tensor& grad,
    const torch::Tensor& psi_W1,
    const torch::Tensor& psi_b1,
    const torch::Tensor& psi_W2,
    float psi_b2,
    float alpha, float beta,
    float lr, float beta1_a, float beta2_a, float eps, float wd,
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
    const int grid = std::min<int>(65535, (N + block - 1) / block);

    const bool all_fp32 = param.scalar_type() == torch::kFloat32 &&
                          grad.scalar_type() == torch::kFloat32;

    // Autotuner picks the vector width: width 4 + 16B-aligned all-FP32 takes the
    // float4 fast path; otherwise the SG_TUNED_UNROLL scalar path runs. Both
    // stage the psi weights into shared once per block.
    if (SG_TUNED_VEC_WIDTH == 4 && all_fp32 &&
        prim::is_vec4_alignable(param.data_ptr(), N) &&
        prim::is_vec4_alignable(exp_avg.data_ptr(), N) &&
        prim::is_vec4_alignable(exp_avg_sq.data_ptr(), N) &&
        prim::is_vec4_alignable(grad.data_ptr(), N)) {
        const int N4 = N / 4;
        const int grid4 = std::min<int>(65535, (N4 + block - 1) / block);
        neuralgrok_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            psi_W1.data_ptr<float>(),
            psi_b1.data_ptr<float>(),
            psi_W2.data_ptr<float>(),
            psi_b2,
            alpha, beta, lr, beta1_a, beta2_a, eps, wd, bc1, bc2, N4);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "neuralgrok_step", [&] {
            neuralgrok_kernel<scalar_t, scalar_t, SG_TUNED_UNROLL>
                <<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                psi_W1.data_ptr<float>(),
                psi_b1.data_ptr<float>(),
                psi_W2.data_ptr<float>(),
                psi_b2,
                alpha, beta, lr, beta1_a, beta2_a, eps, wd, bc1, bc2, N);
        });
}


// Amplifier-only: amplified[i] = psi(|grad[i]|) * grad[i], where
// psi is the 1-hidden-layer network (W1,b1,W2,b2) with ReLU.
// alpha, beta scale the amplified gradient: out = alpha * psi(|g|) * g + beta * g
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_NEURALGROK_MIN_BLOCKS)
neuralgrok_amplifier_kernel(
    const float* grad, float* amplified,
    const float* W1, const float* b1, const float* W2, float b2_scalar,
    float alpha, float beta, int N
) {
    __shared__ float sW1[NG_H];
    __shared__ float sb1[NG_H];
    __shared__ float sW2[NG_H];
    ng_stage_psi_weights<NG_H>(W1, b1, W2, sW1, sb1, sW2);

    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        float g = grad[i];
        float ag = fabsf(g);
        float s = neuralgrok_psi_forward<NG_H>(ag, sW1, sb1, sW2, b2_scalar);
        amplified[i] = alpha * s * g + beta * g;
    }
}

void launch_fused_neuralgrok_amplifier(
    torch::Tensor grad, torch::Tensor amplified,
    torch::Tensor amplifier_w1, torch::Tensor amplifier_b1,
    torch::Tensor amplifier_w2, torch::Tensor amplifier_b2,
    int hidden_dim, float alpha, float beta
) {
    const int64_t N = grad.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);
    float b2_val = amplifier_b2.item<float>();
    neuralgrok_amplifier_kernel<<<grid, block, 0, stream>>>(
        grad.data_ptr<float>(), amplified.data_ptr<float>(),
        amplifier_w1.data_ptr<float>(), amplifier_b1.data_ptr<float>(),
        amplifier_w2.data_ptr<float>(), b2_val,
        alpha, beta, N);
}

// Adam-only on pre-amplified gradient (no psi network evaluation).
__global__ void neuralgrok_adam_only_kernel(
    float* param, float* exp_avg, float* exp_avg_sq,
    const float* amplified_grad,
    float beta1, float beta2, float lr, float wd, float eps,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        // g_eff is the pre-amplified gradient; the bias-corrected Adam +
        // decoupled-WD math lives once in algorithms/neuralgrok.h.
        const float g_eff = amplified_grad[i];
        neuralgrok_adam_tail(param, exp_avg, exp_avg_sq, g_eff,
                             lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

void launch_fused_neuralgrok_adam(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor amplified_grad,
    float beta1, float beta2, float lr, float weight_decay, float eps,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);
    neuralgrok_adam_only_kernel<<<grid, block, 0, stream>>>(
        param.data_ptr<float>(), exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(), amplified_grad.data_ptr<float>(),
        beta1, beta2, lr, weight_decay, eps, bc1, bc2, N);
}

// Full fused step: psi amplification + Adam update in a single kernel launch.
void launch_fused_neuralgrok_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay, float eps,
    float bc1, float bc2
) {
    float b2_val = b2.item<float>();
    launch_neuralgrok_step(param, exp_avg, exp_avg_sq, grad,
                           W1, b1, W2, b2_val,
                           alpha_amp, beta_amp,
                           lr, beta1, beta2, eps, weight_decay, bc1, bc2);
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_NEURALGROK_SM90_CUH_
