// =====================================================================
//  csrc/kernels/hip/gfx942/lion.hip.cpp
//
//  gfx942 (CDNA3 / MI300X) Lion optimizer launcher TU.
//
//  Self-contained: defines the HIP __global__ kernel inline, then exposes
//  the torch::Tensor-facing launchers in namespace sg::gfx942 to satisfy
//  csrc/bindings/lion.cpp + csrc/bindings/multi_tensor.cpp forward decls.
//
//  Dtype matrix (reduced from sm_90):
//    ParamT, GradT in {float, __hip_bfloat16, __half}
//    StateT (exp_avg) is always FP32
//  FP8 grads are out of scope for the gfx942 port (no FP8 support on CDNA3
//  hardware). The runtime dispatch raises if presented an FP8 grad.
// =====================================================================

#include "csrc/common/platform.h"
#include "csrc/common/types.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace sg { namespace gfx942 { namespace lion_impl {

// ---------------------------------------------------------------------
// Per-element Lion step (HIP device-inline). All math runs in FP32; loads
// and stores are typed.
// ---------------------------------------------------------------------

template <typename ParamT, typename GradT>
__device__ __forceinline__ void lion_step_one(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    const GradT* __restrict__ grad,
    const float lr, const float beta1, const float beta2, const float wd,
    const int64_t idx
) {
    const float g  = static_cast<float>(grad[idx]);
    const float ea = exp_avg[idx];
    const float p  = static_cast<float>(param[idx]);

    const float interp = beta1 * ea + (1.0f - beta1) * g;
    const float s = (interp != 0.0f) ? copysignf(1.0f, interp) : 0.0f;

    param[idx]   = static_cast<ParamT>(p - lr * (s + wd * p));
    exp_avg[idx] = beta2 * ea + (1.0f - beta2) * g;
}

template <typename ParamT, typename GradT>
__global__ void fused_lion_step_kernel(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    const GradT* __restrict__ grad,
    const float lr, const float beta1, const float beta2, const float wd,
    const int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    lion_step_one<ParamT, GradT>(param, exp_avg, grad, lr, beta1, beta2, wd, idx);
}

constexpr int kBlockSize = 256;

template <typename ParamT, typename GradT>
inline hipError_t launch_lion_typed(
    ParamT* param, float* exp_avg, const GradT* grad,
    float lr, float beta1, float beta2, float wd,
    int64_t N, hipStream_t stream
) {
    if (N == 0) return hipSuccess;
    const int64_t grid = (N + kBlockSize - 1) / kBlockSize;
    fused_lion_step_kernel<ParamT, GradT>
        <<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
            param, exp_avg, grad, lr, beta1, beta2, wd, N);
    return hipGetLastError();
}

inline void check_hip(hipError_t e, const char* where) {
    if (e != hipSuccess) {
        throw std::runtime_error(
            std::string("lion ") + where + ": " + hipGetErrorString(e));
    }
}

template <typename Functor, typename ParamT, typename... Args>
inline void dispatch_grad(at::ScalarType g, Args&&... args) {
    switch (g) {
        case at::ScalarType::Float:
            Functor::template run<ParamT, float>(std::forward<Args>(args)...);
            break;
        case at::ScalarType::BFloat16:
            Functor::template run<ParamT, __hip_bfloat16>(std::forward<Args>(args)...);
            break;
        case at::ScalarType::Half:
            Functor::template run<ParamT, __half>(std::forward<Args>(args)...);
            break;
        default:
            throw std::runtime_error(
                std::string("lion (gfx942): unsupported grad dtype ") +
                c10::toString(g));
    }
}

template <typename Functor, typename... Args>
inline void dispatch_param_grad(at::ScalarType p, at::ScalarType g, Args&&... args) {
    switch (p) {
        case at::ScalarType::Float:
            dispatch_grad<Functor, float>(g, std::forward<Args>(args)...);
            break;
        case at::ScalarType::BFloat16:
            dispatch_grad<Functor, __hip_bfloat16>(g, std::forward<Args>(args)...);
            break;
        case at::ScalarType::Half:
            dispatch_grad<Functor, __half>(g, std::forward<Args>(args)...);
            break;
        default:
            throw std::runtime_error(
                std::string("lion (gfx942): unsupported param dtype ") +
                c10::toString(p));
    }
}

struct SingleStep {
    template <typename ParamT, typename GradT>
    static void run(
        torch::Tensor& param, torch::Tensor& exp_avg, torch::Tensor& grad,
        float lr, float beta1, float beta2, float wd,
        int64_t N, hipStream_t stream
    ) {
        auto e = launch_lion_typed<ParamT, GradT>(
            static_cast<ParamT*>(param.data_ptr()),
            static_cast<float*>(exp_avg.data_ptr()),
            static_cast<const GradT*>(grad.data_ptr()),
            lr, beta1, beta2, wd, N, stream);
        check_hip(e, "launch_fused_lion_step");
    }
};

struct MultiStep {
    template <typename ParamT, typename GradT>
    static void run(
        torch::Tensor& param, torch::Tensor& exp_avg, torch::Tensor& grad,
        float lr, float beta1, float beta2, float wd,
        int64_t N, hipStream_t stream
    ) {
        auto e = launch_lion_typed<ParamT, GradT>(
            static_cast<ParamT*>(param.data_ptr()),
            static_cast<float*>(exp_avg.data_ptr()),
            static_cast<const GradT*>(grad.data_ptr()),
            lr, beta1, beta2, wd, N, stream);
        check_hip(e, "launch_multi_tensor_lion");
    }
};

}}} // namespace sg::gfx942::lion_impl

// =====================================================================
// Public torch::Tensor launchers in namespace sg::gfx942.
// =====================================================================

namespace sg { namespace gfx942 {

void launch_fused_lion_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,
    float lr, float beta1, float beta2, float weight_decay
) {
    TORCH_CHECK(param.is_cuda(),   "lion (gfx942): param must be GPU");
    TORCH_CHECK(exp_avg.is_cuda(), "lion (gfx942): exp_avg must be GPU");
    TORCH_CHECK(grad.is_cuda(),    "lion (gfx942): grad must be GPU");
    TORCH_CHECK(param.is_contiguous(),   "lion (gfx942): param must be contiguous");
    TORCH_CHECK(exp_avg.is_contiguous(), "lion (gfx942): exp_avg must be contiguous");
    TORCH_CHECK(grad.is_contiguous(),    "lion (gfx942): grad must be contiguous");
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat,
                "lion (gfx942): exp_avg must be FP32 (BF16 state not supported on this port)");
    TORCH_CHECK(param.numel() == grad.numel(),
                "lion (gfx942): param/grad numel mismatch");
    TORCH_CHECK(param.numel() == exp_avg.numel(),
                "lion (gfx942): param/exp_avg numel mismatch");

    const int64_t N = param.numel();
    if (N == 0) return;

    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    sg::gfx942::lion_impl::dispatch_param_grad<sg::gfx942::lion_impl::SingleStep>(
        param.scalar_type(), grad.scalar_type(),
        param, exp_avg, grad, lr, beta1, beta2, weight_decay, N, stream);
}

namespace {

void launch_multi_tensor_lion_impl(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& exp_avgs,
    const std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float wd
) {
    const size_t T = params.size();
    TORCH_CHECK(exp_avgs.size() == T && grads.size() == T,
                "lion (gfx942) multi-tensor: vector size mismatch");
    if (T == 0) return;

    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    for (size_t i = 0; i < T; ++i) {
        if (!params[i].defined() || params[i].numel() == 0) continue;
        TORCH_CHECK(params[i].is_cuda() && exp_avgs[i].is_cuda() && grads[i].is_cuda(),
                    "lion (gfx942) multi-tensor: tensors must be GPU");
        TORCH_CHECK(params[i].is_contiguous() && exp_avgs[i].is_contiguous() &&
                    grads[i].is_contiguous(),
                    "lion (gfx942) multi-tensor: tensors must be contiguous");
        TORCH_CHECK(exp_avgs[i].scalar_type() == at::kFloat,
                    "lion (gfx942) multi-tensor: exp_avg must be FP32");
        TORCH_CHECK(params[i].numel() == grads[i].numel() &&
                    params[i].numel() == exp_avgs[i].numel(),
                    "lion (gfx942) multi-tensor: per-index numel mismatch");

        torch::Tensor p = params[i], ea = exp_avgs[i], g = grads[i];
        sg::gfx942::lion_impl::dispatch_param_grad<sg::gfx942::lion_impl::MultiStep>(
            p.scalar_type(), g.scalar_type(),
            p, ea, g, lr, beta1, beta2, wd, p.numel(), stream);
    }
}

} // anonymous

void launch_multi_tensor_lion(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float weight_decay
) {
    launch_multi_tensor_lion_impl(params, exp_avgs, grads,
                                  lr, beta1, beta2, weight_decay);
}

void launch_multi_tensor_lion(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> grads,
    float lr, float beta1, float beta2, float weight_decay
) {
    launch_multi_tensor_lion_impl(params, exp_avgs, grads,
                                  lr, beta1, beta2, weight_decay);
}

}} // namespace sg::gfx942
