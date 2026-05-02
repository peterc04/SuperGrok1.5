// =====================================================================
//  csrc/kernels/hip/gfx942/grokadamw.hip.cpp
//
//  gfx942 GrokAdamW launchers. Provides per-tensor step / clip_step /
//  step_q3 plus multi-tensor (by-ref and by-value overloads).
//
//  GrokAdamW: AdamW with an EMA-based "grokfast" gradient amplifier.
//     ema_t = ema_{t-1} * lamb + (1 - lamb) * g
//     g'    = g + alpha * ema_t
//     [then standard AdamW on g']
//
//  Q3 variant: int8-quantized exp_avg with per-block FP32 scales,
//  bf16 exp_avg_sq + ema. Stochastic rounding for the int8 store.
//  This port implements the dequant-update-requant path in FP32.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/grokadamw.hip.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace gfx942 { namespace grokadamw_impl {

// ---------------------------------------------------------------------
// Non-quantized GrokAdamW step (also used by the clip variant; the host
// side scales the grad by the clip coefficient before the launch).
// ---------------------------------------------------------------------

template <typename ParamT>
__global__ void grokadamw_step_kernel(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ ema,
    const ParamT* __restrict__ grad,
    const float alpha, const float lamb,
    const float beta1, const float beta2,
    const float lr, const float wd, const float eps,
    const float bc1, const float bc2,
    const float clip_scale,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = static_cast<float>(grad[idx]) * clip_scale;
    const float p = static_cast<float>(param[idx]);

    // EMA + amplification
    const float new_ema = lamb * ema[idx] + (1.0f - lamb) * g;
    ema[idx] = new_ema;
    const float g_amp = g + alpha * new_ema;

    // AdamW moments
    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * g_amp;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g_amp * g_amp;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    const float m_hat = m / bc1;
    const float v_hat = v / bc2;
    const float u = m_hat / (sqrtf(v_hat) + eps) + wd * p;
    param[idx] = static_cast<ParamT>(p - lr * u);
}

template <typename ParamT>
inline hipError_t launch_step_typed(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* ema,
    const ParamT* grad,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps,
    float bc1, float bc2, float clip_scale,
    int64_t N, hipStream_t stream
) {
    using namespace sg::gfx942::common;
    if (N == 0) return hipSuccess;
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;
    grokadamw_step_kernel<ParamT><<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
        param, exp_avg, exp_avg_sq, ema, grad,
        alpha, lamb, beta1, beta2, lr, wd, eps, bc1, bc2, clip_scale, N);
    return hipGetLastError();
}

// ---------------------------------------------------------------------
// Q3 variant: int8 exp_avg + per-block FP32 scales + bf16 v + bf16 ema.
// Per-block scale: floor(idx / kQ3Block) selects the scale.
// ---------------------------------------------------------------------

constexpr int kQ3Block = 64;

template <typename ParamT>
__global__ void grokadamw_step_q3_kernel(
    ParamT* __restrict__ param,
    int8_t* __restrict__ exp_avg_int8,
    float* __restrict__ exp_avg_scales,
    __hip_bfloat16* __restrict__ exp_avg_sq_bf16,
    __hip_bfloat16* __restrict__ ema_bf16,
    const ParamT* __restrict__ grad,
    const float alpha, const float lamb,
    const float beta1, const float beta2,
    const float lr, const float wd, const float eps,
    const float bc1, const float bc2,
    const unsigned global_step,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int64_t bidx = idx / kQ3Block;
    const float scale = exp_avg_scales[bidx];

    const float g = static_cast<float>(grad[idx]);
    const float p = static_cast<float>(param[idx]);

    // EMA (bf16 in storage; FP32 math)
    const float ema_prev = static_cast<float>(ema_bf16[idx]);
    const float new_ema = lamb * ema_prev + (1.0f - lamb) * g;
    ema_bf16[idx] = static_cast<__hip_bfloat16>(new_ema);

    const float g_amp = g + alpha * new_ema;

    // m: int8 exp_avg dequant -> FP32 update -> stochastic re-quant.
    const float m_prev = scale * static_cast<float>(exp_avg_int8[idx]);
    const float m = beta1 * m_prev + (1.0f - beta1) * g_amp;
    unsigned r = hash_prng(global_step, static_cast<unsigned>(idx));
    exp_avg_int8[idx] = float_to_int8_stochastic(m, scale + 1e-12f, r);

    // v: bf16 storage
    const float v_prev = static_cast<float>(exp_avg_sq_bf16[idx]);
    const float v = beta2 * v_prev + (1.0f - beta2) * g_amp * g_amp;
    exp_avg_sq_bf16[idx] = static_cast<__hip_bfloat16>(v);

    const float m_hat = m / bc1;
    const float v_hat = v / bc2;
    const float u = m_hat / (sqrtf(v_hat) + eps) + wd * p;
    param[idx] = static_cast<ParamT>(p - lr * u);
}

template <typename ParamT>
inline hipError_t launch_step_q3_typed(
    ParamT* param, int8_t* exp_avg_int8, float* exp_avg_scales,
    __hip_bfloat16* exp_avg_sq_bf16, __hip_bfloat16* ema_bf16,
    const ParamT* grad,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps,
    float bc1, float bc2, unsigned global_step,
    int64_t N, hipStream_t stream
) {
    using namespace sg::gfx942::common;
    if (N == 0) return hipSuccess;
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;
    grokadamw_step_q3_kernel<ParamT><<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
        param, exp_avg_int8, exp_avg_scales, exp_avg_sq_bf16, ema_bf16, grad,
        alpha, lamb, beta1, beta2, lr, wd, eps, bc1, bc2, global_step, N);
    return hipGetLastError();
}

// ---------------------------------------------------------------------
// Per-tensor host wrappers (used by single-step + clip + multi-tensor).
// ---------------------------------------------------------------------

inline void run_step(
    torch::Tensor& param, torch::Tensor& exp_avg, torch::Tensor& exp_avg_sq,
    torch::Tensor& ema, torch::Tensor& grad,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps, float bc1, float bc2, float clip_scale
) {
    using namespace sg::gfx942::common;
    TORCH_CHECK(param.is_cuda() && grad.is_cuda(),
                "grokadamw (gfx942): tensors must be GPU");
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat &&
                exp_avg_sq.scalar_type() == at::kFloat &&
                ema.scalar_type() == at::kFloat,
                "grokadamw (gfx942): state tensors must be FP32");
    TORCH_CHECK(grad.scalar_type() == param.scalar_type(),
                "grokadamw (gfx942): grad and param dtypes must match");
    const int64_t N = param.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    auto disp = [&](auto tag) {
        using T = decltype(tag);
        auto err = launch_step_typed<T>(
            reinterpret_cast<T*>(param.data_ptr()),
            exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
            ema.data_ptr<float>(),
            reinterpret_cast<const T*>(grad.data_ptr()),
            alpha, lamb, beta1, beta2, lr, wd, eps, bc1, bc2,
            clip_scale, N, stream);
        check_hip(err, "launch_fused_grokadamw_step");
    };

    switch (param.scalar_type()) {
        case at::kFloat:    disp(float{});           break;
        case at::kBFloat16: disp(__hip_bfloat16{});  break;
        case at::kHalf:     disp(__half{});          break;
        default:
            TORCH_CHECK(false, "grokadamw (gfx942): unsupported param dtype ",
                        param.scalar_type());
    }
}

}}} // namespace sg::gfx942::grokadamw_impl

namespace sg { namespace gfx942 {

// ---------------------------------------------------------------------
// Per-tensor entry: standard step.
// ---------------------------------------------------------------------
void launch_fused_grokadamw_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    sg::gfx942::grokadamw_impl::run_step(
        param, exp_avg, exp_avg_sq, ema, grad,
        alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
        /*clip_scale=*/1.0f);
}

// ---------------------------------------------------------------------
// Per-tensor entry: clip step. Computes a per-tensor grad-norm clip
// coefficient host-side, then launches the same kernel with clip_scale.
// ---------------------------------------------------------------------
void launch_fused_grokadamw_clip_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2,
    float clip_threshold
) {
    float clip_scale = 1.0f;
    if (clip_threshold > 0.0f && grad.defined() && grad.numel() > 0) {
        auto g_f32 = grad.to(torch::kFloat32).reshape(-1);
        float gn = std::sqrt(g_f32.dot(g_f32).item<float>());
        if (gn > clip_threshold) clip_scale = clip_threshold / (gn + 1e-6f);
    }
    sg::gfx942::grokadamw_impl::run_step(
        param, exp_avg, exp_avg_sq, ema, grad,
        alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
        clip_scale);
}

// ---------------------------------------------------------------------
// Per-tensor entry: q3 (int8 exp_avg + bf16 v + bf16 ema).
// ---------------------------------------------------------------------
void launch_fused_grokadamw_step_q3(
    torch::Tensor param,
    torch::Tensor exp_avg_int8,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_bf16,
    torch::Tensor ema_bf16,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2,
    unsigned global_step
) {
    using namespace sg::gfx942::common;
    TORCH_CHECK(param.is_cuda() && grad.is_cuda(),
                "grokadamw_q3 (gfx942): tensors must be GPU");
    TORCH_CHECK(exp_avg_int8.scalar_type() == at::kChar,
                "grokadamw_q3 (gfx942): exp_avg_int8 must be int8");
    TORCH_CHECK(exp_avg_scales.scalar_type() == at::kFloat,
                "grokadamw_q3 (gfx942): exp_avg_scales must be FP32");
    TORCH_CHECK(exp_avg_sq_bf16.scalar_type() == at::kBFloat16,
                "grokadamw_q3 (gfx942): exp_avg_sq_bf16 must be BF16");
    TORCH_CHECK(ema_bf16.scalar_type() == at::kBFloat16,
                "grokadamw_q3 (gfx942): ema_bf16 must be BF16");
    TORCH_CHECK(grad.scalar_type() == param.scalar_type(),
                "grokadamw_q3 (gfx942): grad and param dtypes must match");

    const int64_t N = param.numel();
    if (N == 0) return;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    auto disp = [&](auto tag) {
        using T = decltype(tag);
        auto err = sg::gfx942::grokadamw_impl::launch_step_q3_typed<T>(
            reinterpret_cast<T*>(param.data_ptr()),
            exp_avg_int8.data_ptr<int8_t>(),
            exp_avg_scales.data_ptr<float>(),
            reinterpret_cast<__hip_bfloat16*>(exp_avg_sq_bf16.data_ptr()),
            reinterpret_cast<__hip_bfloat16*>(ema_bf16.data_ptr()),
            reinterpret_cast<const T*>(grad.data_ptr()),
            alpha, lamb, beta1, beta2, lr, weight_decay, eps,
            bc1, bc2, global_step, N, stream);
        check_hip(err, "launch_fused_grokadamw_step_q3");
    };

    switch (param.scalar_type()) {
        case at::kFloat:    disp(float{});           break;
        case at::kBFloat16: disp(__hip_bfloat16{});  break;
        case at::kHalf:     disp(__half{});          break;
        default:
            TORCH_CHECK(false,
                "grokadamw_q3 (gfx942): unsupported param dtype ",
                param.scalar_type());
    }
}

// ---------------------------------------------------------------------
// Multi-tensor — by-reference (csrc/bindings/grokadamw.cpp).
// Loops over (params, ..., bc1s[i], bc2s[i]) and dispatches per-tensor.
// ---------------------------------------------------------------------
void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps
) {
    const size_t T = params.size();
    TORCH_CHECK(exp_avgs.size() == T && exp_avg_sqs.size() == T &&
                emas.size() == T && grads.size() == T &&
                bc1s.size() == T && bc2s.size() == T,
                "grokadamw multi-tensor (gfx942): vector size mismatch");
    for (size_t i = 0; i < T; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        sg::gfx942::grokadamw_impl::run_step(
            params[i], exp_avgs[i], exp_avg_sqs[i], emas[i], grads[i],
            alpha, lamb, beta1, beta2, lr, wd, eps,
            bc1s[i], bc2s[i], /*clip_scale=*/1.0f);
    }
}

// ---------------------------------------------------------------------
// Multi-tensor — by-value (csrc/bindings/multi_tensor.cpp). Single shared
// (bc1, bc2) for all tensors.
// ---------------------------------------------------------------------
void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> emas,
    std::vector<torch::Tensor> grads,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    const size_t T = params.size();
    TORCH_CHECK(exp_avgs.size() == T && exp_avg_sqs.size() == T &&
                emas.size() == T && grads.size() == T,
                "grokadamw multi-tensor (gfx942): vector size mismatch");
    for (size_t i = 0; i < T; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        sg::gfx942::grokadamw_impl::run_step(
            params[i], exp_avgs[i], exp_avg_sqs[i], emas[i], grads[i],
            alpha, lamb, beta1, beta2, lr, weight_decay, eps,
            bc1, bc2, /*clip_scale=*/1.0f);
    }
}

}} // namespace sg::gfx942
