// =====================================================================
//  csrc/kernels/cuda/sm_90/supergrok15.cu
//
//  sm_90 (Hopper) SuperGrok v1.5 — explicit instantiation TU.
//
//  All kernel + launcher logic lives in supergrok15.cuh as templates.
//  This TU forces emission of the dtype matrix below into a single
//  object file so callers outside this TU (e.g. the per-arch shim that
//  the bindings dispatch through) can link against the per-tensor entry
//  points without dragging the cooperative-groups + fp8 headers into a
//  non-CUDA source.
//
//  Dtype matrix (per build spec):
//    ParamT in {float, __nv_bfloat16, __half}                     (3)
//    StateT in {float, __nv_bfloat16}                             (2)
//    GradT  in {float, __nv_bfloat16, __half,
//               __nv_fp8_e4m3, __nv_fp8_e5m2}                     (5)
//
//  Coherence rule (mirrors adamw.cu and supergrok11.cu):
//    FP8 grad with FP32 param is REJECTED via static_assert and is
//    therefore absent from the instantiation list below. All other
//    cells of the 3·2·5 = 30 cube are valid for fused_step (26 active);
//    SAM perturb / restore are only (ParamT, GradT) pairs (13 active
//    each). Total instantiations: 26 + 13 + 13 = 52.
// =====================================================================

#include "csrc/kernels/cuda/sm_90/supergrok15.cuh"

namespace sg { namespace sm90 { namespace supergrok15 {

// =====================================================================
// fused_step instantiations (26 total)
// =====================================================================

#define INST_FUSED(P, S, G)                                                   \
    template cudaError_t launch_supergrok15_fused_step<P, S, G>(              \
        P*, S*, S*, S*,                                                       \
        const G*, const G*,                                                   \
        const float*,                                                         \
        float, float, float, float, float,                                    \
        float, float, float, float,                                           \
        float, float,                                                         \
        int, int,                                                             \
        int64_t, int64_t,                                                     \
        cudaStream_t)

// ---------------------------------------------------------------------
// FP32 param family — FP8 grads excluded by is_coherent_combo.
// State ∈ {FP32, BF16} × Grad ∈ {FP32, BF16, FP16}  → 6 cells.
// ---------------------------------------------------------------------
INST_FUSED(float, float,         float);
INST_FUSED(float, float,         __nv_bfloat16);
INST_FUSED(float, float,         __half);
INST_FUSED(float, __nv_bfloat16, float);
INST_FUSED(float, __nv_bfloat16, __nv_bfloat16);
INST_FUSED(float, __nv_bfloat16, __half);

// ---------------------------------------------------------------------
// BF16 param family — full grad cross-section incl. FP8.
// 2 states × 5 grads = 10 cells.
// ---------------------------------------------------------------------
INST_FUSED(__nv_bfloat16, float,         float);
INST_FUSED(__nv_bfloat16, float,         __nv_bfloat16);
INST_FUSED(__nv_bfloat16, float,         __half);
INST_FUSED(__nv_bfloat16, float,         __nv_fp8_e4m3);
INST_FUSED(__nv_bfloat16, float,         __nv_fp8_e5m2);
INST_FUSED(__nv_bfloat16, __nv_bfloat16, float);
INST_FUSED(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
INST_FUSED(__nv_bfloat16, __nv_bfloat16, __half);
INST_FUSED(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3);
INST_FUSED(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e5m2);

// ---------------------------------------------------------------------
// FP16 param family — full grad cross-section incl. FP8.
// 2 states × 5 grads = 10 cells.
// ---------------------------------------------------------------------
INST_FUSED(__half, float,         float);
INST_FUSED(__half, float,         __nv_bfloat16);
INST_FUSED(__half, float,         __half);
INST_FUSED(__half, float,         __nv_fp8_e4m3);
INST_FUSED(__half, float,         __nv_fp8_e5m2);
INST_FUSED(__half, __nv_bfloat16, float);
INST_FUSED(__half, __nv_bfloat16, __nv_bfloat16);
INST_FUSED(__half, __nv_bfloat16, __half);
INST_FUSED(__half, __nv_bfloat16, __nv_fp8_e4m3);
INST_FUSED(__half, __nv_bfloat16, __nv_fp8_e5m2);

#undef INST_FUSED

// =====================================================================
// sam_perturb_all instantiations (13 total)
// =====================================================================

#define INST_SAM(P, G)                                                        \
    template cudaError_t launch_supergrok15_sam_perturb_all<P, G>(            \
        P*, const G*, float, float, int64_t, cudaStream_t)

// FP32 param — FP8 grads excluded.
INST_SAM(float, float);
INST_SAM(float, __nv_bfloat16);
INST_SAM(float, __half);

// BF16 param — full grad cross-section.
INST_SAM(__nv_bfloat16, float);
INST_SAM(__nv_bfloat16, __nv_bfloat16);
INST_SAM(__nv_bfloat16, __half);
INST_SAM(__nv_bfloat16, __nv_fp8_e4m3);
INST_SAM(__nv_bfloat16, __nv_fp8_e5m2);

// FP16 param — full grad cross-section.
INST_SAM(__half, float);
INST_SAM(__half, __nv_bfloat16);
INST_SAM(__half, __half);
INST_SAM(__half, __nv_fp8_e4m3);
INST_SAM(__half, __nv_fp8_e5m2);

#undef INST_SAM

// =====================================================================
// sharpness_restore_all instantiations (13 total — mirror of SAM)
// =====================================================================

#define INST_RESTORE(P, G)                                                    \
    template cudaError_t launch_supergrok15_sharpness_restore_all<P, G>(      \
        P*, const G*, float, float, int64_t, cudaStream_t)

// FP32 param — FP8 grads excluded.
INST_RESTORE(float, float);
INST_RESTORE(float, __nv_bfloat16);
INST_RESTORE(float, __half);

// BF16 param — full grad cross-section.
INST_RESTORE(__nv_bfloat16, float);
INST_RESTORE(__nv_bfloat16, __nv_bfloat16);
INST_RESTORE(__nv_bfloat16, __half);
INST_RESTORE(__nv_bfloat16, __nv_fp8_e4m3);
INST_RESTORE(__nv_bfloat16, __nv_fp8_e5m2);

// FP16 param — full grad cross-section.
INST_RESTORE(__half, float);
INST_RESTORE(__half, __nv_bfloat16);
INST_RESTORE(__half, __half);
INST_RESTORE(__half, __nv_fp8_e4m3);
INST_RESTORE(__half, __nv_fp8_e5m2);

#undef INST_RESTORE

}}} // namespace sg::sm90::supergrok15

// =====================================================================
// torch::Tensor binding shims — namespace sg::sm90
//
// The bindings in csrc/bindings/supergrok15.cpp declare three per-tensor
// functions in each arch namespace (e.g. sg::sm90). These operate on
// torch::Tensors and are decomposed into torch/ATen tensor operations
// (for mu EMA, MLP forward, Adam) or use simple tensor ops (for SAM
// perturb and sharpness restore).
//
// Same pattern as the SG11 shims: avoids a full dtype-dispatch switch
// by relying on ATen's internal dispatch for mixed-dtype tensor ops.
// =====================================================================

#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

namespace sg { namespace sm90 {

// -----------------------------------------------------------------
// 1. launch_fused_supergrok15_full_step
//
//    Full fused step: mu EMA + MLP forward + Adam + weight decay.
//
//    mu   ← (1 - alpha) * mu + alpha * grad          (EMA)
//    h     = GELU(inp @ W1^T + b1)                    (hidden, inp=[grad,sharpness])
//    smart_grad = (h @ W2^T + b2) * rescale            (MLP output)
//    exp_avg    = beta1 * exp_avg    + (1-beta1) * smart_grad
//    exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * smart_grad^2
//    m_hat = exp_avg / bc1;  v_hat = exp_avg_sq / bc2
//    update = m_hat / (sqrt(v_hat) + eps)
//    param  = (1 - lr * wd_eff) * param - lr * lamb_eff * update
// -----------------------------------------------------------------

void launch_fused_supergrok15_full_step(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor mu,
    torch::Tensor grad,
    torch::Tensor sharpness,
    float alpha,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2,
    float rescale,
    float lamb_eff,
    float beta1,
    float beta2,
    float lr,
    float wd_eff,
    float eps,
    float bc1,
    float bc2,
    int hidden_dim)
{
    (void)hidden_dim;   // implied by W1.size(0)

    // --- mu EMA ---
    mu.mul_(1.0f - alpha).add_(grad, alpha);

    // --- MLP forward (all math in float) ---
    auto g_flat = grad.flatten().to(torch::kFloat32);
    int64_t N = g_flat.size(0);

    // Build [N, 2] input: column 0 = grad, column 1 = sharpness scalar.
    float s_val = sharpness.item<float>();
    auto inp = torch::empty({N, 2}, g_flat.options());
    inp.select(1, 0).copy_(g_flat);
    inp.select(1, 1).fill_(s_val);

    // W1 is [H,2], b1 is [H], W2 is [1,H], b2 is [1]
    auto W1_f = W1.to(torch::kFloat32);
    auto b1_f = b1.to(torch::kFloat32);
    auto W2_f = W2.to(torch::kFloat32);
    auto b2_f = b2.to(torch::kFloat32);

    // hidden = GELU( inp @ W1^T + b1 )   => [N, H]
    auto hidden = torch::addmm(b1_f, inp, W1_f.t()).gelu();
    // out = hidden @ W2^T + b2  => [N, 1]
    auto out = torch::addmm(b2_f, hidden, W2_f.t());
    // smart_grad: reshaped + scaled
    auto smart_grad = out.reshape_as(param).mul_(rescale);

    // --- Adam step (in-place moment updates) ---
    exp_avg.mul_(beta1).add_(smart_grad, 1.0f - beta1);
    exp_avg_sq.mul_(beta2).addcmul_(smart_grad, smart_grad, 1.0f - beta2);

    // Bias-corrected moments.
    auto m_hat = exp_avg / bc1;
    auto v_hat = exp_avg_sq / bc2;

    // Adam update: m_hat / (sqrt(v_hat) + eps)
    auto update = m_hat / (v_hat.sqrt() + eps);

    // Weight decay + step.
    param.mul_(1.0f - lr * wd_eff).add_(update, -lr * lamb_eff);
}

// -----------------------------------------------------------------
// 2. launch_sam_perturb
//
//    SAM perturbation: param += rho_over_norm * grad
// -----------------------------------------------------------------

void launch_sam_perturb(
    torch::Tensor param,
    torch::Tensor grad,
    float rho_over_norm)
{
    param.add_(grad, rho_over_norm);
}

// -----------------------------------------------------------------
// 3. launch_sharpness_restore
//
//    Computes sharpness = |sam_grad - normal_grad| and restores
//    params from backup.
// -----------------------------------------------------------------

void launch_sharpness_restore(
    torch::Tensor param,
    torch::Tensor sharpness,
    torch::Tensor backup,
    torch::Tensor sam_grad,
    torch::Tensor normal_grad)
{
    // Sharpness: element-wise |sam_grad - normal_grad|
    sharpness.copy_((sam_grad - normal_grad).abs());
    // Restore params from backup.
    param.copy_(backup);
}

}} // namespace sg::sm90
