#ifndef GROKKING_KERNELS_SM90_SUPERGROK15_SM90_CUH_
#define GROKKING_KERNELS_SM90_SUPERGROK15_SM90_CUH_
// ============================================================================
// supergrok15_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'supergrok15'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_supergrok15.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for SuperGrok v1.5.
// Algorithm: csrc/algorithms/supergrok15.h
//
// Same two-sweep pattern as v1.1, but the gate is a global sigmoid of
// training accuracy (passed in as gate_global), and per-coord alpha is
// computed from mu via sg15_alpha_per_coord.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/supergrok15.h"
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
using ::sg::algorithms::sg15_phi_forward;
using ::sg::algorithms::sg15_sweep_a_step;
using ::sg::algorithms::sg15_sweep_b_step;

constexpr int SG15_H = 64;

// Minimum resident blocks/SM for the element-wise sweeps. Caps registers so
// occupancy stays high on the (memory-bound) Adam apply.
#ifndef SG_SUPERGROK15_MIN_BLOCKS
#define SG_SUPERGROK15_MIN_BLOCKS 4
#endif

// Cooperatively stage the per-element meta-net phi weights into shared memory
// ONCE per block, then hand the shared pointers to the canonical
// sg15_phi_forward<H>. W1 is [H,2], b1/W2 are [H]. Removes the per-element
// re-read from GLOBAL inside the H-wide forward loop. NO math/signature change —
// the fn still takes const float* (single-sourced in algorithms/supergrok15.h).
template <int H>
__device__ __forceinline__ void sg15_stage_phi_weights(
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2,
    float* sW1, float* sb1, float* sW2
) {
    for (int j = threadIdx.x; j < H * 2; j += blockDim.x) sW1[j] = W1[j];
    for (int j = threadIdx.x; j < H;     j += blockDim.x) { sb1[j] = b1[j]; sW2[j] = W2[j]; }
    __syncthreads();
}

template <typename GradT>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_SUPERGROK15_MIN_BLOCKS)
sg15_sweep_a_kernel(
    float* mu_out, const GradT* grad, const float* sharpness,
    const float* W1, const float* b1, const float* W2, float b2,
    float* sharp_partial, int N
) {
    __shared__ float sW1[SG15_H * 2];
    __shared__ float sb1[SG15_H];
    __shared__ float sW2[SG15_H];
    sg15_stage_phi_weights<SG15_H>(W1, b1, W2, sW1, sb1, sW2);

    float sl = 0.0f;
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        const float g = static_cast<float>(grad[i]);
        const float s = sharpness[i];
        const float mu_val = sg15_phi_forward<SG15_H>(g, s, sW1, sb1, sW2, b2);
        sg15_sweep_a_step(mu_out, grad, mu_val, i, sl);
    }
    float r = prim::block_reduce_sum_f32(sl);
    if (threadIdx.x == 0) atomicAdd(sharp_partial, r);
}

template <typename ParamT, typename GradT, int UNROLL>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_SUPERGROK15_MIN_BLOCKS)
sg15_sweep_b_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const GradT* grad, const float* mu,
    float gate_global, float alpha_base, float alpha_max,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride() * UNROLL;
    const int base0 = prim::grid_stride_index() * UNROLL;
    for (int base = base0; base < N; base += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int i = base + u;
            if (i < N) {
                sg15_sweep_b_step(param, exp_avg, exp_avg_sq, grad, mu,
                                  gate_global, alpha_base, alpha_max,
                                  lr, beta1, beta2, eps, wd, bc1, bc2, i);
            }
        }
    }
}

// FP32 vec4 sweep B: float4 traffic on param/state/grad/mu, canonical
// sg15_sweep_b_step CALLED 4× on the register lanes. Math is NOT re-typed here.
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_SUPERGROK15_MIN_BLOCKS)
sg15_sweep_b_kernel_vec4_fp32(
    float4* param4, float4* exp_avg4, float4* exp_avg_sq4,
    const float4* grad4, const float4* mu4,
    float gate_global, float alpha_base, float alpha_max,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N4
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N4; i += stride) {
        float4 p  = prim::ld_f32v4(param4 + i);
        float4 m  = prim::ld_f32v4(exp_avg4 + i);
        float4 v  = prim::ld_f32v4(exp_avg_sq4 + i);
        float4 g  = prim::ldg_f32v4(grad4 + i);
        float4 mu = prim::ldg_f32v4(mu4 + i);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            sg15_sweep_b_step(&p.x, &m.x, &v.x, &g.x, &mu.x,
                              gate_global, alpha_base, alpha_max,
                              lr, beta1, beta2, eps, wd, bc1, bc2, u);
        }
        prim::st_f32v4(param4 + i, p);
        prim::st_f32v4(exp_avg4 + i, m);
        prim::st_f32v4(exp_avg_sq4 + i, v);
    }
}

void launch_supergrok15_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& mu_buf,
    const torch::Tensor& grad,
    const torch::Tensor& sharpness,
    const torch::Tensor& phi_W1,
    const torch::Tensor& phi_b1,
    const torch::Tensor& phi_W2,
    float phi_b2,
    torch::Tensor& sharp_partial,
    float gate_global,
    float alpha_base, float alpha_max,
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
    const int grid = std::min<int>(65535, (N + block - 1) / block);

    sharp_partial.zero_();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "sg15_sweep_a", [&] {
            sg15_sweep_a_kernel<scalar_t><<<grid, block, 0, stream>>>(
                mu_buf.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<float>(),
                phi_W1.data_ptr<float>(),
                phi_b1.data_ptr<float>(),
                phi_W2.data_ptr<float>(),
                phi_b2,
                sharp_partial.data_ptr<float>(), N);
        });

    const bool all_fp32 = param.scalar_type() == torch::kFloat32 &&
                          grad.scalar_type() == torch::kFloat32;
    if (SG_TUNED_VEC_WIDTH == 4 && all_fp32 &&
        prim::is_vec4_alignable(param.data_ptr(), N) &&
        prim::is_vec4_alignable(exp_avg.data_ptr(), N) &&
        prim::is_vec4_alignable(exp_avg_sq.data_ptr(), N) &&
        prim::is_vec4_alignable(grad.data_ptr(), N) &&
        prim::is_vec4_alignable(mu_buf.data_ptr(), N)) {
        const int N4 = N / 4;
        const int grid4 = std::min<int>(65535, (N4 + block - 1) / block);
        sg15_sweep_b_kernel_vec4_fp32<<<grid4, block, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg.data_ptr<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            reinterpret_cast<const float4*>(mu_buf.data_ptr<float>()),
            gate_global, alpha_base, alpha_max,
            lr, beta1, beta2, eps, wd, bc1, bc2, N4);
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg15_sweep_b", [&] {
            sg15_sweep_b_kernel<scalar_t, scalar_t, SG_TUNED_UNROLL>
                <<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                mu_buf.data_ptr<float>(),
                gate_global, alpha_base, alpha_max,
                lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}


// Full fused SG15 step: meta-net forward (sweep_a) + Adam update (sweep_b).
void launch_fused_supergrok15_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim
) {
    float b2_val = b2.item<float>();
    auto sharp_partial = torch::zeros({1},
        torch::TensorOptions().device(param.device()).dtype(torch::kFloat32));
    float gate_global = lamb_eff;
    float alpha_base = alpha;
    float alpha_max = alpha;
    launch_supergrok15_step(param, exp_avg, exp_avg_sq, mu, grad, sharpness,
                            W1, b1, W2, b2_val, sharp_partial,
                            gate_global, alpha_base, alpha_max,
                            lr, beta1, beta2, eps, wd_eff, bc1, bc2);
}

// SAM perturbation: param += rho_over_norm * grad
template <typename ParamT>
__global__ void sg15_sam_perturb_kernel(
    ParamT* param, const ParamT* grad, float rho_over_norm, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        float p = static_cast<float>(param[i]);
        float g = static_cast<float>(grad[i]);
        param[i] = static_cast<ParamT>(p + rho_over_norm * g);
    }
}

void launch_sam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg15_sam_perturb", [&] {
            sg15_sam_perturb_kernel<scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
                rho_over_norm, N);
        });
}

// Sharpness restore: param = backup, sharpness = (sam_grad - normal_grad)^2
template <typename ParamT>
__global__ void sg15_sharpness_restore_kernel(
    ParamT* param, float* sharpness, const ParamT* backup,
    const ParamT* sam_grad, const ParamT* normal_grad, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        param[i] = backup[i];
        float diff = static_cast<float>(sam_grad[i]) -
                     static_cast<float>(normal_grad[i]);
        sharpness[i] = diff * diff;
    }
}

void launch_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness, torch::Tensor backup,
    torch::Tensor sam_grad, torch::Tensor normal_grad
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg15_sharpness_restore", [&] {
            sg15_sharpness_restore_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                sharpness.data_ptr<float>(),
                backup.data_ptr<scalar_t>(),
                sam_grad.data_ptr<scalar_t>(),
                normal_grad.data_ptr<scalar_t>(), N);
        });
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_SUPERGROK15_SM90_CUH_
