// =====================================================================
//  csrc/kernels/cuda/sm_90/neuralgrok.cu
//
//  sm_90 (Hopper) NeuralGrok kernel — explicit instantiation TU.
//
//  All kernel + launcher logic lives in neuralgrok.cuh as templates;
//  this TU emits the dtype-matrix instantiations of
//  launch_neuralgrok_fused_step plus the binding-compatible wrappers
//  (sg::sm90::launch_fused_neuralgrok_*) and the unique definitions of
//  the __constant__ psi mirror and the per-call alpha/beta thread_locals.
//
//  Net-new replacement for the deleted neuralgrok_sm90.cu (commit
//  5505b50). Coverage: ParamT in {fp32, bf16, fp16}, StateT in
//  {fp32, bf16}, GradT in {fp32, bf16, fp16, fp8_e4m3, fp8_e5m2}.
//  Incoherent combos (FP8 grad + FP32 param) are rejected by
//  is_coherent_combo and simply omitted from the instantiation list.
// =====================================================================

#include "csrc/kernels/cuda/sm_90/neuralgrok.cuh"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

// ---------------------------------------------------------------------
// Unique definitions: __constant__ psi mirror and per-call thread_locals.
// ---------------------------------------------------------------------
__constant__ float sg_sm90_neuralgrok_psi[
    sg::sm90::neuralgrok::kPsiNumFloat];

namespace sg { namespace sm90 {
thread_local float sg_sm90_neuralgrok_alpha = 0.0f;
thread_local float sg_sm90_neuralgrok_beta  = 1.0f;
}} // namespace sg::sm90

namespace sg { namespace sm90 { namespace neuralgrok {

// =====================================================================
// Explicit instantiations of launch_neuralgrok_fused_step over the dtype
// matrix. FP8-grad + FP32-param combos are rejected by static_assert
// and are absent from this list (would otherwise fail the build).
// =====================================================================

#define SG_NG_INST(P, S, G)                                              \
    template cudaError_t launch_neuralgrok_fused_step<P, S, G>(          \
        P*, S*, S*, const G*,                                            \
        const float*,                                                    \
        float, float, float, float, float, float, float,                 \
        int, int, int64_t, int64_t, cudaStream_t);

// FP32 param family (FP8 grads excluded).
SG_NG_INST(float, float, float)
SG_NG_INST(float, float, __nv_bfloat16)
SG_NG_INST(float, float, __half)
SG_NG_INST(float, __nv_bfloat16, float)
SG_NG_INST(float, __nv_bfloat16, __nv_bfloat16)
SG_NG_INST(float, __nv_bfloat16, __half)

// BF16 param family — full grad cross-section.
SG_NG_INST(__nv_bfloat16, float, float)
SG_NG_INST(__nv_bfloat16, float, __nv_bfloat16)
SG_NG_INST(__nv_bfloat16, float, __half)
SG_NG_INST(__nv_bfloat16, float, __nv_fp8_e4m3)
SG_NG_INST(__nv_bfloat16, float, __nv_fp8_e5m2)
SG_NG_INST(__nv_bfloat16, __nv_bfloat16, float)
SG_NG_INST(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16)
SG_NG_INST(__nv_bfloat16, __nv_bfloat16, __half)
SG_NG_INST(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3)
SG_NG_INST(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e5m2)

// FP16 param family — full grad cross-section.
SG_NG_INST(__half, float, float)
SG_NG_INST(__half, float, __nv_bfloat16)
SG_NG_INST(__half, float, __half)
SG_NG_INST(__half, float, __nv_fp8_e4m3)
SG_NG_INST(__half, float, __nv_fp8_e5m2)
SG_NG_INST(__half, __nv_bfloat16, float)
SG_NG_INST(__half, __nv_bfloat16, __nv_bfloat16)
SG_NG_INST(__half, __nv_bfloat16, __half)
SG_NG_INST(__half, __nv_bfloat16, __nv_fp8_e4m3)
SG_NG_INST(__half, __nv_bfloat16, __nv_fp8_e5m2)

#undef SG_NG_INST

}}} // namespace sg::sm90::neuralgrok

// =====================================================================
// Binding-compatible wrapper bodies (namespace sg::sm90).
//
// The bindings (csrc/bindings/neuralgrok.cpp) declare three entry points
// per arch namespace; the only one actually used by the public Python
// flow is launch_fused_neuralgrok_full_step (called from the per-tensor
// loop inside neuralgrok_fused_step). The other two are kept here as
// thin shims so the DECLARE_NEURALGROK(sm90) forward declarations stay
// resolvable at link time even when callers route around them.
// =====================================================================

namespace sg { namespace sm90 {

namespace {

// Pack [W1: H][b1: H][W2: H][b2: 1] into a single contiguous device
// buffer (the format the __constant__ mirror expects).
torch::Tensor pack_psi_weights(
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    int H
) {
    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(W1.device());
    auto buf = torch::empty({3 * H + 1}, opts);
    auto W1f = W1.dtype() == torch::kFloat32 ? W1.contiguous()
                                             : W1.to(torch::kFloat32).contiguous();
    auto b1f = b1.dtype() == torch::kFloat32 ? b1.contiguous()
                                             : b1.to(torch::kFloat32).contiguous();
    auto W2f = W2.dtype() == torch::kFloat32 ? W2.contiguous()
                                             : W2.to(torch::kFloat32).contiguous();
    auto b2f = b2.dtype() == torch::kFloat32 ? b2.contiguous()
                                             : b2.to(torch::kFloat32).contiguous();
    // W1 is [H, 1] from the Python ref — flatten.
    buf.narrow(0, 0,     H).copy_(W1f.view(-1));
    buf.narrow(0, H,     H).copy_(b1f.view(-1));
    buf.narrow(0, 2 * H, H).copy_(W2f.view(-1));
    buf.narrow(0, 3 * H, 1).copy_(b2f.view(-1));
    return buf;
}

template <typename ParamT, typename StateT, typename GradT>
inline cudaError_t dispatch_full_step(
    torch::Tensor& param, torch::Tensor& exp_avg, torch::Tensor& exp_avg_sq,
    torch::Tensor& grad,
    const float* psi_ptr,
    int H, int L,
    float lr, float beta1, float beta2, float wd, float eps,
    float bc1, float bc2,
    cudaStream_t stream
) {
    return sg::sm90::neuralgrok::launch_neuralgrok_fused_step<
        ParamT, StateT, GradT>(
        reinterpret_cast<ParamT*>(param.data_ptr()),
        reinterpret_cast<StateT*>(exp_avg.data_ptr()),
        reinterpret_cast<StateT*>(exp_avg_sq.data_ptr()),
        reinterpret_cast<const GradT*>(grad.data_ptr()),
        psi_ptr,
        lr, beta1, beta2, eps, wd, bc1, bc2,
        H, L, param.numel(), /*step_count*/ 0, stream);
}

} // anonymous namespace

void launch_fused_neuralgrok_full_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor grad,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;
    TORCH_CHECK(grad.scalar_type() == param.scalar_type(),
        "neuralgrok: grad and param dtypes must match");
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat,
        "neuralgrok: exp_avg must be FP32");
    TORCH_CHECK(exp_avg_sq.scalar_type() == at::kFloat,
        "neuralgrok: exp_avg_sq must be FP32");
    TORCH_CHECK(hidden_dim > 0 && hidden_dim <= sg::sm90::neuralgrok::kPsiHmax,
        "neuralgrok: hidden_dim out of __constant__ budget");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Pack psi into a single contiguous device buffer; the templated
    // launcher reads via cudaMemcpyToSymbolAsync from this pointer.
    auto psi = pack_psi_weights(W1, b1, W2, b2, hidden_dim);

    // Stash alpha/beta for the templated launcher (per-call scope).
    sg_sm90_neuralgrok_alpha = alpha_amp;
    sg_sm90_neuralgrok_beta  = beta_amp;

    // Layers count is informational here (the kernel evaluates the
    // 2-layer first/last composition matching the Python get_weights()).
    constexpr int L = 2;

    cudaError_t err = cudaSuccess;
    switch (param.scalar_type()) {
        case at::kFloat:
            err = dispatch_full_step<float, float, float>(
                param, exp_avg, exp_avg_sq, grad, psi.data_ptr<float>(),
                hidden_dim, L, lr, beta1, beta2, weight_decay, eps,
                bc1, bc2, stream);
            break;
        case at::kBFloat16:
            err = dispatch_full_step<__nv_bfloat16, float, __nv_bfloat16>(
                param, exp_avg, exp_avg_sq, grad, psi.data_ptr<float>(),
                hidden_dim, L, lr, beta1, beta2, weight_decay, eps,
                bc1, bc2, stream);
            break;
        case at::kHalf:
            err = dispatch_full_step<__half, float, __half>(
                param, exp_avg, exp_avg_sq, grad, psi.data_ptr<float>(),
                hidden_dim, L, lr, beta1, beta2, weight_decay, eps,
                bc1, bc2, stream);
            break;
        default:
            TORCH_CHECK(false,
                "neuralgrok: unsupported param dtype ", param.scalar_type());
    }
    TORCH_CHECK(err == cudaSuccess,
        "neuralgrok full_step failed: ", cudaGetErrorString(err));
}

// The amplifier-only and adam-only entry points are not exercised by
// the current Python flow (which routes through full_step). They are
// kept as link-resolvable stubs so the DECLARE_NEURALGROK(sm90) macro
// in csrc/bindings/neuralgrok.cpp does not produce an undefined symbol.
void launch_fused_neuralgrok_amplifier(
    torch::Tensor /*grad*/, torch::Tensor /*amplified*/,
    torch::Tensor /*W1*/,   torch::Tensor /*b1*/,
    torch::Tensor /*W2*/,   torch::Tensor /*b2*/,
    int /*hidden_dim*/, float /*alpha*/, float /*beta*/
) {
    TORCH_CHECK(false,
        "sg::sm90::launch_fused_neuralgrok_amplifier: split-kernel path "
        "is not exposed on sm_90; use launch_fused_neuralgrok_full_step");
}

void launch_fused_neuralgrok_adam(
    torch::Tensor /*param*/, torch::Tensor /*exp_avg*/,
    torch::Tensor /*exp_avg_sq*/, torch::Tensor /*amplified_grad*/,
    float /*beta1*/, float /*beta2*/,
    float /*lr*/, float /*weight_decay*/,
    float /*eps*/, float /*bc1*/, float /*bc2*/
) {
    TORCH_CHECK(false,
        "sg::sm90::launch_fused_neuralgrok_adam: split-kernel path "
        "is not exposed on sm_90; use launch_fused_neuralgrok_full_step");
}

}} // namespace sg::sm90
