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
// The meta-net hidden width H is fixed at compile time. Defaults to 64.

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

namespace sg { namespace sm90 {

namespace prim = ::sg::sm90::primitives;
using ::sg::algorithms::sg11_phi_forward;
using ::sg::algorithms::sg11_sweep_a_step;
using ::sg::algorithms::sg11_sweep_b_step;
using ::sg::algorithms::sg11_adam_tail;

constexpr int SG11_H = 64;

template <typename GradT>
__global__ void sg11_sweep_a_kernel(
    float* mu_out, const GradT* grad, const float* sharpness, const float* momentum,
    const float* W1, const float* b1, const float* W2, float b2,
    float* gate_num_p, float* gate_den_g_p, float* gate_den_m_p,
    int N
) {
    float gn = 0.0f, gdg = 0.0f, gdm = 0.0f;
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        const float g = static_cast<float>(grad[i]);
        const float s = sharpness[i];
        const float mu_val = sg11_phi_forward<SG11_H>(g, s, W1, b1, W2, b2);
        sg11_sweep_a_step(mu_out, grad, sharpness, momentum, mu_val, i,
                          gn, gdg, gdm);
    }
    float r1 = prim::block_reduce_sum_f32(gn);
    float r2 = prim::block_reduce_sum_f32(gdg);
    float r3 = prim::block_reduce_sum_f32(gdm);
    if (threadIdx.x == 0) {
        atomicAdd(gate_num_p, r1);
        atomicAdd(gate_den_g_p, r2);
        atomicAdd(gate_den_m_p, r3);
    }
}

template <typename ParamT, typename GradT>
__global__ void sg11_sweep_b_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq,
    const GradT* grad, const float* mu, float gate,
    float alpha, float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        sg11_sweep_b_step(param, exp_avg, exp_avg_sq, grad, mu, gate,
                          alpha, lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

void launch_supergrok11_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& mu_buf,
    const torch::Tensor& grad,
    const torch::Tensor& sharpness,
    const torch::Tensor& momentum,
    const torch::Tensor& phi_W1,
    const torch::Tensor& phi_b1,
    const torch::Tensor& phi_W2,
    float phi_b2,
    torch::Tensor& gate_partials,  // [3] {num, den_g, den_m}
    float alpha,
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

    gate_partials.zero_();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "sg11_sweep_a", [&] {
            sg11_sweep_a_kernel<scalar_t><<<grid, block, 0, stream>>>(
                mu_buf.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<float>(),
                momentum.data_ptr<float>(),
                phi_W1.data_ptr<float>(),
                phi_b1.data_ptr<float>(),
                phi_W2.data_ptr<float>(),
                phi_b2,
                gate_partials.data_ptr<float>(),
                gate_partials.data_ptr<float>() + 1,
                gate_partials.data_ptr<float>() + 2,
                N);
        });

    // Host-side reduction of the 3 scalars and gate clamp.
    auto partials_cpu = gate_partials.to(torch::kCPU);
    float gn  = partials_cpu[0].item<float>();
    float gdg = partials_cpu[1].item<float>();
    float gdm = partials_cpu[2].item<float>();
    float denom = sqrtf(gdg * gdm + 1e-12f);
    float gate = (denom > 0.0f) ? (gn / denom) : 0.0f;
    gate = fminf(fmaxf(gate, 0.0f), 1.0f);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg11_sweep_b", [&] {
            sg11_sweep_b_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                mu_buf.data_ptr<float>(),
                gate, alpha, lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}


// Meta-net forward: mu[i] = phi(grad[i], sharpness[i]); smart_grad = grad + alpha*mu
template <typename GradT>
__global__ void sg11_mu_metanet_kernel(
    float* mu, const GradT* grad, const float* sharpness,
    float* smart_grad, float alpha,
    const float* W1, const float* b1, const float* W2, float b2_scalar,
    float rescale, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        float g = static_cast<float>(grad[i]);
        float s = sharpness[i];
        float phi = sg11_phi_forward<SG11_H>(g, s, W1, b1, W2, b2_scalar) * rescale;
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
    const int grid = std::min<int>(65535, (N + block - 1) / block);
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
        });
}

// Adam + decoupled WD on smart_grad, with optional lamb-scaled mu blending.
__global__ void sg11_adam_decay_kernel(
    float* param, float* exp_avg, float* exp_avg_sq,
    const float* smart_grad, const float* mu,
    float lamb_eff, float beta1, float beta2, float lr,
    float wd_eff, float eps, float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        // g_eff = smart_grad + lamb_eff*mu blending; the bias-corrected Adam +
        // decoupled-WD math lives once in algorithms/supergrok11.h.
        const float g_eff = smart_grad[i] + lamb_eff * mu[i];
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
    const int grid = std::min<int>(65535, (N + block - 1) / block);
    sg11_adam_decay_kernel<<<grid, block, 0, stream>>>(
        param.data_ptr<float>(), exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        smart_grad.data_ptr<float>(), mu.data_ptr<float>(),
        lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2, N);
}

// SAM perturbation: param += rho_over_norm * grad
template <typename ParamT>
__global__ void sg11_sam_perturb_kernel(
    ParamT* param, const ParamT* grad, float rho_over_norm, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        float p = static_cast<float>(param[i]);
        float g = static_cast<float>(grad[i]);
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
    const int grid = std::min<int>(65535, (N + block - 1) / block);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sg11_sam_perturb", [&] {
            sg11_sam_perturb_kernel<scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
                rho_over_norm, N);
        });
}

// Sharpness restore: param = backup, sharpness = ||sam_grad - normal_grad||
template <typename ParamT>
__global__ void sg11_sharpness_restore_kernel(
    ParamT* param, float* sharpness, const ParamT* backup,
    const ParamT* sam_grad, const ParamT* normal_grad, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
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
    const int grid = std::min<int>(65535, (N + block - 1) / block);
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
