// =====================================================================
//  csrc/kernels/hip/gfx942/adamw.hip.cpp
//
//  gfx942 fused AdamW (vector-form) launcher TU. Provides the symbol
//  launch_fused_adamw_simple referenced by csrc/bindings/grokadamw.cpp
//  via DECLARE_MT_GROKADAMW(gfx942).
//
//  Algorithm (decoupled WD AdamW, single fused elementwise pass):
//     m_t   = beta1 * m_{t-1} + (1 - beta1) * g
//     v_t   = beta2 * v_{t-1} + (1 - beta2) * g^2
//     m_hat = m_t / bc1
//     v_hat = v_t / bc2
//     u     = m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1}
//     theta = theta_{t-1} - lr * u
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/adamw.hip.h"

namespace sg { namespace gfx942 { namespace adamw_impl {

template <typename ParamT>
__global__ void adamw_simple_kernel(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const ParamT* __restrict__ grad,
    const float lr, const float beta1, const float beta2,
    const float eps, const float wd, const float bc1, const float bc2,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float g  = static_cast<float>(grad[idx]);
    const float p  = static_cast<float>(param[idx]);
    const float m_prev = exp_avg[idx];
    const float v_prev = exp_avg_sq[idx];

    const float m = beta1 * m_prev + (1.0f - beta1) * g;
    const float v = beta2 * v_prev + (1.0f - beta2) * g * g;

    const float m_hat = m / bc1;
    const float v_hat = v / bc2;

    const float u = m_hat / (sqrtf(v_hat) + eps) + wd * p;
    param[idx]      = static_cast<ParamT>(p - lr * u);
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;
}

template <typename ParamT>
inline hipError_t launch_typed(
    ParamT* param, float* exp_avg, float* exp_avg_sq, const ParamT* grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N, hipStream_t stream
) {
    if (N == 0) return hipSuccess;
    using namespace sg::gfx942::common;
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;
    adamw_simple_kernel<ParamT><<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
        param, exp_avg, exp_avg_sq, grad,
        lr, beta1, beta2, eps, wd, bc1, bc2, N);
    return hipGetLastError();
}

}}} // namespace sg::gfx942::adamw_impl

namespace sg { namespace gfx942 {

void launch_fused_adamw_simple(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    std::vector<int64_t>& steps,
    float beta1, float beta2, float lr, float wd, float eps
) {
    const size_t n_params = params.size();
    if (n_params == 0) return;

    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    for (size_t i = 0; i < n_params; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        steps[i] += 1;

        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));
        const int64_t N = params[i].numel();

        TORCH_CHECK(exp_avgs[i].scalar_type()    == at::kFloat,
                    "fused_adamw_simple (gfx942): exp_avg must be float32");
        TORCH_CHECK(exp_avg_sqs[i].scalar_type() == at::kFloat,
                    "fused_adamw_simple (gfx942): exp_avg_sq must be float32");
        TORCH_CHECK(grads[i].scalar_type() == params[i].scalar_type(),
                    "fused_adamw_simple (gfx942): grad and param dtypes must match");

        switch (params[i].scalar_type()) {
            case at::kFloat: {
                auto err = sg::gfx942::adamw_impl::launch_typed<float>(
                    params[i].data_ptr<float>(),
                    exp_avgs[i].data_ptr<float>(),
                    exp_avg_sqs[i].data_ptr<float>(),
                    grads[i].data_ptr<float>(),
                    lr, beta1, beta2, eps, wd, bc1, bc2, N, stream);
                sg::gfx942::common::check_hip(err, "launch_fused_adamw_simple (fp32)");
                break;
            }
            case at::kBFloat16: {
                auto err = sg::gfx942::adamw_impl::launch_typed<__hip_bfloat16>(
                    reinterpret_cast<__hip_bfloat16*>(params[i].data_ptr()),
                    exp_avgs[i].data_ptr<float>(),
                    exp_avg_sqs[i].data_ptr<float>(),
                    reinterpret_cast<const __hip_bfloat16*>(grads[i].data_ptr()),
                    lr, beta1, beta2, eps, wd, bc1, bc2, N, stream);
                sg::gfx942::common::check_hip(err, "launch_fused_adamw_simple (bf16)");
                break;
            }
            case at::kHalf: {
                auto err = sg::gfx942::adamw_impl::launch_typed<__half>(
                    reinterpret_cast<__half*>(params[i].data_ptr()),
                    exp_avgs[i].data_ptr<float>(),
                    exp_avg_sqs[i].data_ptr<float>(),
                    reinterpret_cast<const __half*>(grads[i].data_ptr()),
                    lr, beta1, beta2, eps, wd, bc1, bc2, N, stream);
                sg::gfx942::common::check_hip(err, "launch_fused_adamw_simple (fp16)");
                break;
            }
            default:
                TORCH_CHECK(false,
                    "fused_adamw_simple (gfx942): unsupported param dtype ",
                    params[i].scalar_type());
        }
    }
}

}} // namespace sg::gfx942
