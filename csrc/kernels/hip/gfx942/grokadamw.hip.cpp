// =====================================================================
//  csrc/kernels/hip/gfx942/grokadamw.hip.cpp
//
//  gfx942 GrokAdamW launchers — ATen-based reference implementation.
//  Provides per-tensor step / clip_step / step_q3 plus multi-tensor
//  by-ref + by-value overloads.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/grokadamw.hip.h"

namespace sg { namespace gfx942 {

namespace {

inline void grokadamw_step_aten(
    torch::Tensor& param, torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq, torch::Tensor& ema, torch::Tensor& grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float wd, float eps,
    float bc1, float bc2,
    float clip_scale
) {
    auto p_f32 = param.to(torch::kFloat32);
    auto g_f32 = grad.to(torch::kFloat32);
    if (clip_scale != 1.0f) g_f32 = g_f32 * clip_scale;

    // ema = lamb * ema + (1 - lamb) * g
    ema.mul_(lamb).add_(g_f32, 1.0f - lamb);
    // amplified = g + alpha * ema
    auto g_amp = g_f32 + alpha * ema;

    exp_avg.mul_(beta1).add_(g_amp, 1.0f - beta1);
    exp_avg_sq.mul_(beta2).addcmul_(g_amp, g_amp, 1.0f - beta2);

    auto m_hat = exp_avg / bc1;
    auto v_hat = exp_avg_sq / bc2;
    auto u = m_hat / (v_hat.sqrt() + eps) + wd * p_f32;
    param.copy_((p_f32 - lr * u).to(param.scalar_type()));
}

} // anonymous

void launch_fused_grokadamw_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2
) {
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat &&
                exp_avg_sq.scalar_type() == at::kFloat &&
                ema.scalar_type() == at::kFloat,
                "grokadamw (gfx942): state tensors must be FP32");
    if (param.numel() == 0) return;
    grokadamw_step_aten(param, exp_avg, exp_avg_sq, ema, grad,
                        alpha, lamb, beta1, beta2, lr, weight_decay, eps,
                        bc1, bc2, /*clip_scale=*/1.0f);
}

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
    if (param.numel() == 0) return;
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat &&
                exp_avg_sq.scalar_type() == at::kFloat &&
                ema.scalar_type() == at::kFloat,
                "grokadamw_clip (gfx942): state tensors must be FP32");
    grokadamw_step_aten(param, exp_avg, exp_avg_sq, ema, grad,
                        alpha, lamb, beta1, beta2, lr, weight_decay, eps,
                        bc1, bc2, clip_scale);
}

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
    (void)global_step;  // stochastic-rounding path is not enabled in the
                        // ATen reference; fall back to simple round.
    TORCH_CHECK(exp_avg_int8.scalar_type() == at::kChar,
                "grokadamw_q3 (gfx942): exp_avg_int8 must be int8");
    TORCH_CHECK(exp_avg_scales.scalar_type() == at::kFloat,
                "grokadamw_q3 (gfx942): exp_avg_scales must be FP32");
    TORCH_CHECK(exp_avg_sq_bf16.scalar_type() == at::kBFloat16,
                "grokadamw_q3 (gfx942): exp_avg_sq_bf16 must be BF16");
    TORCH_CHECK(ema_bf16.scalar_type() == at::kBFloat16,
                "grokadamw_q3 (gfx942): ema_bf16 must be BF16");
    if (param.numel() == 0) return;

    // Dequantize int8 exp_avg to FP32 using per-block scales (block=64).
    constexpr int kQ3Block = 64;
    const int64_t N = param.numel();
    auto exp_avg_f32 = exp_avg_int8.to(torch::kFloat32);
    // Broadcast per-block scale: scales shape is [ceil(N/64)].
    auto idx = torch::arange(N,
        torch::TensorOptions().device(param.device()).dtype(torch::kLong));
    auto block_idx = idx / kQ3Block;
    auto scales_bcast = exp_avg_scales.index({block_idx});
    exp_avg_f32 = exp_avg_f32 * scales_bcast.reshape_as(exp_avg_f32);

    auto exp_avg_sq_f32 = exp_avg_sq_bf16.to(torch::kFloat32);
    auto ema_f32        = ema_bf16.to(torch::kFloat32);

    grokadamw_step_aten(param, exp_avg_f32, exp_avg_sq_f32, ema_f32, grad,
                        alpha, lamb, beta1, beta2, lr, weight_decay, eps,
                        bc1, bc2, /*clip_scale=*/1.0f);

    // Re-quantize exp_avg back to int8 with the same per-block scales.
    auto requant = (exp_avg_f32 / scales_bcast.reshape_as(exp_avg_f32))
                       .clamp(-127.0f, 127.0f)
                       .round()
                       .to(torch::kChar);
    exp_avg_int8.copy_(requant);
    exp_avg_sq_bf16.copy_(exp_avg_sq_f32.to(torch::kBFloat16));
    ema_bf16.copy_(ema_f32.to(torch::kBFloat16));
}

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
        TORCH_CHECK(exp_avgs[i].scalar_type() == at::kFloat &&
                    exp_avg_sqs[i].scalar_type() == at::kFloat &&
                    emas[i].scalar_type() == at::kFloat,
                    "grokadamw multi-tensor (gfx942): state tensors must be FP32");
        grokadamw_step_aten(params[i], exp_avgs[i], exp_avg_sqs[i],
                            emas[i], grads[i],
                            alpha, lamb, beta1, beta2, lr, wd, eps,
                            bc1s[i], bc2s[i], /*clip_scale=*/1.0f);
    }
}

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
        TORCH_CHECK(exp_avgs[i].scalar_type() == at::kFloat &&
                    exp_avg_sqs[i].scalar_type() == at::kFloat &&
                    emas[i].scalar_type() == at::kFloat,
                    "grokadamw multi-tensor (gfx942): state tensors must be FP32");
        grokadamw_step_aten(params[i], exp_avgs[i], exp_avg_sqs[i],
                            emas[i], grads[i],
                            alpha, lamb, beta1, beta2, lr, weight_decay, eps,
                            bc1, bc2, /*clip_scale=*/1.0f);
    }
}

}} // namespace sg::gfx942
