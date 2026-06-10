#ifndef GROKKING_KERNELS_SM90_SUPERGROK11_SM90_CUH_
#define GROKKING_KERNELS_SM90_SUPERGROK11_SM90_CUH_
// ============================================================================
// supergrok11_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'supergrok11'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_supergrok11.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for SuperGrok v1.1.
// Algorithm: csrc/algorithms/supergrok11.h
//
// Two-sweep cooperative pattern:
//   Sweep A: meta-net forward + cosine gate reduction
//   Sweep B: smart_grad mixing + Adam apply (uses gate from sweep A)
//
// The meta-net hidden width H is fixed at compile time. Defaults to 32
// (matches the canonical SharpnessMetaNet hidden_dim in
// grokking_optimizers/optimizers/supergrok11.py).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/supergrok11.h"
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
using ::sg::algorithms::sg11_phi_forward;
using ::sg::algorithms::sg11_adam_tail;
// NOTE: sg11_sweep_a_step / sg11_sweep_b_step were the per-element bodies of the
// two-sweep cosine-gate path (sg11_sweep_a_kernel / sg11_sweep_b_kernel /
// launch_supergrok11_step). That path was DEAD device code — no binding and no
// in-file caller launched launch_supergrok11_step (the live SG v1.1 path is
// launch_sg11_mu_metanet + compute_cosine_gate_fused + launch_sg11_adam_decay,
// orchestrated by supergrok11_fused_step in csrc/bindings/bindings.cpp). The
// sweep kernels + launcher are removed below, so these two using-declarations
// (which only fed them) are dropped as well. The canonical step bodies remain in
// csrc/algorithms/supergrok11.h, untouched.

constexpr int SG11_H = 32;

// Minimum resident blocks/SM for the element-wise sweeps. Caps registers so
// occupancy stays high on the (memory-bound) Adam apply.
#ifndef SG_TUNED_MIN_BLOCKS
#define SG_TUNED_MIN_BLOCKS 4
#endif

// Cooperatively stage the per-element meta-net phi weights into shared memory
// ONCE per block, then hand the shared pointers to the canonical
// sg11_phi_forward<H>. W1 is [H,2] (row-major, 2 inputs/hidden unit), b1/W2 are
// [H]. The same weights are read by every element this block processes; staging
// removes the per-element re-read from GLOBAL inside the H-wide forward loop. NO
// math/signature change — the fn still takes const float* (single-sourced in
// algorithms/supergrok11.h).
template <int H>
__device__ __forceinline__ void sg11_stage_phi_weights(
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2,
    float* sW1, float* sb1, float* sW2
) {
    for (int j = threadIdx.x; j < H * 2; j += blockDim.x) sW1[j] = W1[j];
    for (int j = threadIdx.x; j < H;     j += blockDim.x) { sb1[j] = b1[j]; sW2[j] = W2[j]; }
    __syncthreads();
}

// ── DEAD-CODE REMOVAL (SASS audit #5): SG v1.1 two-sweep cosine-gate path ──
//
// REMOVED here: sg11_sweep_a_kernel, sg11_sweep_b_kernel,
// sg11_sweep_b_kernel_vec4_fp32, and their sole host launcher
// launch_supergrok11_step.
//
// These were never-called device kernels. A whole-tree reference scan
// (csrc/bindings/bindings.cpp + all csrc + all Python) found ZERO references to
// any of the four C++ symbols outside this file, and no in-file caller launched
// launch_supergrok11_step. (The identically-named Python function in
// csrc/backends/pallas/launch_supergrok11.py is a separate JAX/Pallas symbol,
// not this CUDA launcher.)
//
// The LIVE SG v1.1 CUDA path is unchanged: launch_sg11_mu_metanet ->
// compute_cosine_gate_fused -> launch_sg11_adam_decay (plus launch_sg11_sam_-
// perturb / launch_sg11_sharpness_restore), orchestrated by
// supergrok11_fused_step in csrc/bindings/bindings.cpp. sg11_stage_phi_weights
// (used by the live sg11_mu_metanet_kernel) and the canonical per-element bodies
// in csrc/algorithms/supergrok11.h are untouched. This is removal of dead device
// code, NOT suppression of any reachable functionality.


// Meta-net forward: mu[i] = phi(grad[i], sharpness[i]); smart_grad = grad + alpha*mu
template <typename GradT>
__global__ void __launch_bounds__(SG_TUNED_BLOCK_SIZE, SG_TUNED_MIN_BLOCKS)
sg11_mu_metanet_kernel(
    float* mu, const GradT* grad, const float* sharpness,
    float* smart_grad, float alpha,
    const float* W1, const float* b1, const float* W2, float b2_scalar,
    float rescale, int64_t N
) {
    __shared__ float sW1[SG11_H * 2];
    __shared__ float sb1[SG11_H];
    __shared__ float sW2[SG11_H];
    sg11_stage_phi_weights<SG11_H>(W1, b1, W2, sW1, sb1, sW2);

    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        SG_SANITIZE_GRAD_INPLACE(grad, i);
        float g = static_cast<float>(grad[i]);
        float s = sharpness[i];
        float phi = sg11_phi_forward<SG11_H>(g, s, sW1, sb1, sW2, b2_scalar) * rescale;
        mu[i] = phi;
        smart_grad[i] = g + alpha * phi;
    }
}

void launch_sg11_mu_metanet(
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor smart_grad, float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim
) {
    const int64_t N = mu.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));
    float b2_val = b2.item<float>();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "sg11_mu_metanet", [&] {
            sg11_mu_metanet_kernel<scalar_t><<<grid, block, 0, stream>>>(
                mu.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<float>(),
                smart_grad.data_ptr<float>(), alpha,
                W1.data_ptr<float>(), b1.data_ptr<float>(),
                W2.data_ptr<float>(), b2_val, rescale, N);
            SG_LAUNCH_CHECK(stream);
        });
}

// Adam + decoupled WD on smart_grad, with optional lamb-scaled mu blending.
__global__ void sg11_adam_decay_kernel(
    float* param, float* exp_avg, float* exp_avg_sq,
    const float* smart_grad, const float* mu,
    float lamb_eff, float beta1, float beta2, float lr,
    float wd_eff, float eps, float bc1, float bc2, int64_t N
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        // g_eff = smart_grad + lamb_eff*mu blending; the bias-corrected Adam +
        // decoupled-WD math lives once in algorithms/supergrok11.h.
        const float g_eff = ::grokking::sm90::sg_sanitize_grad(smart_grad[i]) +
                            lamb_eff * mu[i];
        sg11_adam_tail(param, exp_avg, exp_avg_sq, g_eff,
                       lr, beta1, beta2, eps, wd_eff, bc1, bc2, i);
    }
}

void launch_sg11_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor smart_grad, torch::Tensor mu,
    float lamb_eff, float beta1, float beta2, float lr,
    float wd_eff, float eps, float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));
    sg11_adam_decay_kernel<<<grid, block, 0, stream>>>(
        param.data_ptr<float>(), exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        smart_grad.data_ptr<float>(), mu.data_ptr<float>(),
        lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2, N);
    SG_LAUNCH_CHECK(stream);
}

// SAM perturbation: param += rho_over_norm * grad
template <typename ParamT>
__global__ void sg11_sam_perturb_kernel(
    ParamT* param, const ParamT* grad, float rho_over_norm, int64_t N
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        float p = static_cast<float>(param[i]);
        float g = ::grokking::sm90::sg_sanitize_grad(static_cast<float>(grad[i]));
        param[i] = static_cast<ParamT>(p + rho_over_norm * g);
    }
}

void launch_sg11_sam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg11_sam_perturb", [&] {
            sg11_sam_perturb_kernel<scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
                rho_over_norm, N);
            SG_LAUNCH_CHECK(stream);
        });
}

// Sharpness restore: param = backup, sharpness = ||sam_grad - normal_grad||
template <typename ParamT>
__global__ void sg11_sharpness_restore_kernel(
    ParamT* param, float* sharpness, const ParamT* backup,
    const ParamT* sam_grad, const ParamT* normal_grad, int64_t N
) {
    const int64_t stride = prim::grid_stride();
    for (int64_t i = prim::grid_stride_index(); i < N; i += stride) {
        param[i] = backup[i];
        float sg = static_cast<float>(sam_grad[i]);
        float ng = static_cast<float>(normal_grad[i]);
        float diff = sg - ng;
        sharpness[i] = diff * diff;
    }
}

void launch_sg11_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness, torch::Tensor backup,
    torch::Tensor sam_grad, torch::Tensor normal_grad
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = static_cast<int>(
        std::min<int64_t>(65535, (N + block - 1) / block));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg11_sharpness_restore", [&] {
            sg11_sharpness_restore_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                sharpness.data_ptr<float>(),
                backup.data_ptr<scalar_t>(),
                sam_grad.data_ptr<scalar_t>(),
                normal_grad.data_ptr<scalar_t>(), N);
            SG_LAUNCH_CHECK(stream);
        });
}

// Cosine gate: cos_sim(smart_grad, mu) clamped to [0,1].
float compute_cosine_gate_fused(
    torch::Tensor smart_grad, torch::Tensor mu, float gate_temp
) {
    auto sg_f = smart_grad.to(torch::kFloat32).flatten();
    auto mu_f = mu.to(torch::kFloat32).flatten();
    float num = (sg_f * mu_f).sum().item<float>();
    float den_g = (sg_f * sg_f).sum().item<float>();
    float den_m = (mu_f * mu_f).sum().item<float>();
    float denom = sqrtf(den_g * den_m + 1e-12f);
    float gate = (denom > 0.0f) ? (num / denom) : 0.0f;
    return fminf(fmaxf(gate, 0.0f), 1.0f);
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_SUPERGROK11_SM90_CUH_
