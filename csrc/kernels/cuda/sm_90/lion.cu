// =====================================================================
//  csrc/kernels/cuda/sm_90/lion.cu
//
//  Thin instantiation TU for the sm_90 Lion optimizer kernels. All actual
//  __global__ + __device__ code lives in lion.cuh; this TU just emits
//  - explicit template instantiations for every coherent dtype combo,
//  - the torch::Tensor-facing launchers (single-tensor + multi-tensor),
//  - a runtime-ScalarType -> typed-call dispatch (3 nested switches, no
//    runtime dtype branches inside the kernel itself).
//
//  See lion.cuh header comment for the algorithm, roofline, and dtype
//  matrix. No additional optimizations are added here.
// =====================================================================

#include "csrc/kernels/cuda/sm_90/lion.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace sg { namespace sm90 { namespace lion {

// ---------------------------------------------------------------------
// Explicit template instantiations of the typed launcher.
//
// Coherent dtype matrix:
//   ParamT in {float, __nv_bfloat16, __half}                           (3)
//   StateT in {float, __nv_bfloat16}                                   (2)
//   GradT  in {float, __nv_bfloat16, __half, __nv_fp8_e4m3, __nv_fp8_e5m2} (5)
// All 30 combos pass is_coherent_combo<>; we instantiate every one.
// ---------------------------------------------------------------------

#define INSTANTIATE_LION(P, S, G)                                              \
    template cudaError_t launch_lion_typed<P, S, G>(                           \
        P*, S*, const G*, float, float, float, float, int64_t, cudaStream_t); \
    template cudaError_t launch_multi_tensor_lion_typed<P, S, G>(              \
        const LionTensorTable<P, S, G>&, int64_t,                              \
        float, float, float, float, cudaStream_t);

// ParamT = float (FP32)
INSTANTIATE_LION(float,         float,         float)
INSTANTIATE_LION(float,         float,         __nv_bfloat16)
INSTANTIATE_LION(float,         float,         __half)
INSTANTIATE_LION(float,         float,         __nv_fp8_e4m3)
INSTANTIATE_LION(float,         float,         __nv_fp8_e5m2)
INSTANTIATE_LION(float,         __nv_bfloat16, float)
INSTANTIATE_LION(float,         __nv_bfloat16, __nv_bfloat16)
INSTANTIATE_LION(float,         __nv_bfloat16, __half)
INSTANTIATE_LION(float,         __nv_bfloat16, __nv_fp8_e4m3)
INSTANTIATE_LION(float,         __nv_bfloat16, __nv_fp8_e5m2)

// ParamT = __nv_bfloat16 (BF16)
INSTANTIATE_LION(__nv_bfloat16, float,         float)
INSTANTIATE_LION(__nv_bfloat16, float,         __nv_bfloat16)
INSTANTIATE_LION(__nv_bfloat16, float,         __half)
INSTANTIATE_LION(__nv_bfloat16, float,         __nv_fp8_e4m3)
INSTANTIATE_LION(__nv_bfloat16, float,         __nv_fp8_e5m2)
INSTANTIATE_LION(__nv_bfloat16, __nv_bfloat16, float)
INSTANTIATE_LION(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16)
INSTANTIATE_LION(__nv_bfloat16, __nv_bfloat16, __half)
INSTANTIATE_LION(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3)
INSTANTIATE_LION(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e5m2)

// ParamT = __half (FP16)
INSTANTIATE_LION(__half,        float,         float)
INSTANTIATE_LION(__half,        float,         __nv_bfloat16)
INSTANTIATE_LION(__half,        float,         __half)
INSTANTIATE_LION(__half,        float,         __nv_fp8_e4m3)
INSTANTIATE_LION(__half,        float,         __nv_fp8_e5m2)
INSTANTIATE_LION(__half,        __nv_bfloat16, float)
INSTANTIATE_LION(__half,        __nv_bfloat16, __nv_bfloat16)
INSTANTIATE_LION(__half,        __nv_bfloat16, __half)
INSTANTIATE_LION(__half,        __nv_bfloat16, __nv_fp8_e4m3)
INSTANTIATE_LION(__half,        __nv_bfloat16, __nv_fp8_e5m2)

#undef INSTANTIATE_LION

// ---------------------------------------------------------------------
// Helpers visible to anything in this TU (anonymous namespace gives
// internal linkage; reachable from the public sg::sm90::launch_*
// functions defined later in the same TU).
// ---------------------------------------------------------------------

namespace {

inline void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("lion ") + where + ": " + cudaGetErrorString(e));
    }
}

} // anonymous namespace

}}} // namespace sg::sm90::lion

// =====================================================================
// Dispatch (C++17 compatible -- no C++20 generic lambdas).
//
// dispatch_lion<Functor>(p_dt, s_dt, g_dt, args...) calls
// Functor::template run<ParamT, StateT, GradT>(args...) for the
// resolved dtype triple. Three nested switch-statements; every case is a
// compile-time-typed call so the compiler emits 30 specialized paths.
// =====================================================================

namespace sg { namespace sm90 { namespace lion {

namespace {

template <typename Functor, typename ParamT, typename StateT, typename... Args>
inline void dispatch_grad(at::ScalarType g, Args&&... args) {
    switch (g) {
        case at::ScalarType::Float:
            Functor::template run<ParamT, StateT, float>(std::forward<Args>(args)...);
            break;
        case at::ScalarType::BFloat16:
            Functor::template run<ParamT, StateT, __nv_bfloat16>(std::forward<Args>(args)...);
            break;
        case at::ScalarType::Half:
            Functor::template run<ParamT, StateT, __half>(std::forward<Args>(args)...);
            break;
        case at::ScalarType::Float8_e4m3fn:
            Functor::template run<ParamT, StateT, __nv_fp8_e4m3>(std::forward<Args>(args)...);
            break;
        case at::ScalarType::Float8_e5m2:
            Functor::template run<ParamT, StateT, __nv_fp8_e5m2>(std::forward<Args>(args)...);
            break;
        default:
            throw std::runtime_error(
                std::string("lion: unsupported grad dtype ") + c10::toString(g));
    }
}

template <typename Functor, typename ParamT, typename... Args>
inline void dispatch_state(at::ScalarType s, at::ScalarType g, Args&&... args) {
    switch (s) {
        case at::ScalarType::Float:
            dispatch_grad<Functor, ParamT, float>(g, std::forward<Args>(args)...);
            break;
        case at::ScalarType::BFloat16:
            dispatch_grad<Functor, ParamT, __nv_bfloat16>(g, std::forward<Args>(args)...);
            break;
        default:
            throw std::runtime_error(
                std::string("lion: unsupported state dtype ") + c10::toString(s) +
                " (must be FP32 or BF16)");
    }
}

template <typename Functor, typename... Args>
inline void dispatch_lion(at::ScalarType p, at::ScalarType s, at::ScalarType g,
                          Args&&... args) {
    switch (p) {
        case at::ScalarType::Float:
            dispatch_state<Functor, float>(s, g, std::forward<Args>(args)...);
            break;
        case at::ScalarType::BFloat16:
            dispatch_state<Functor, __nv_bfloat16>(s, g, std::forward<Args>(args)...);
            break;
        case at::ScalarType::Half:
            dispatch_state<Functor, __half>(s, g, std::forward<Args>(args)...);
            break;
        default:
            throw std::runtime_error(
                std::string("lion: unsupported param dtype ") + c10::toString(p) +
                " (must be FP32, BF16, or FP16)");
    }
}

// ---------------------------------------------------------------------
// Functors -- each implements a templated static run<P,S,G>(args...).
// ---------------------------------------------------------------------

struct SingleStep {
    template <typename ParamT, typename StateT, typename GradT>
    static void run(
        torch::Tensor& param,
        torch::Tensor& exp_avg,
        torch::Tensor& grad,
        float lr, float beta1, float beta2, float wd,
        int64_t N,
        cudaStream_t stream
    ) {
        cudaError_t e = launch_lion_typed<ParamT, StateT, GradT>(
            static_cast<ParamT*>(param.data_ptr()),
            static_cast<StateT*>(exp_avg.data_ptr()),
            static_cast<const GradT*>(grad.data_ptr()),
            lr, beta1, beta2, wd,
            N, stream);
        check_cuda(e, "launch_fused_lion_step");
    }
};

struct MultiTensor {
    template <typename ParamT, typename StateT, typename GradT>
    static void run(
        const std::vector<torch::Tensor>& params,
        const std::vector<torch::Tensor>& exp_avgs,
        const std::vector<torch::Tensor>& grads,
        const std::vector<size_t>& idxs,
        float lr, float beta1, float beta2, float wd,
        cudaStream_t stream
    ) {
        using Tbl = LionTensorTable<ParamT, StateT, GradT>;
        constexpr int kMax = kMaxTensorsPerLaunch;
        for (size_t base = 0; base < idxs.size(); base += kMax) {
            Tbl tbl{};
            int64_t cum = 0;
            tbl.offsets[0] = 0;
            int n = 0;
            for (size_t j = base; j < idxs.size() && n < kMax; ++j, ++n) {
                const size_t i = idxs[j];
                tbl.param[n]   = static_cast<ParamT*>(params[i].data_ptr());
                tbl.exp_avg[n] = static_cast<StateT*>(exp_avgs[i].data_ptr());
                tbl.grad[n]    = static_cast<const GradT*>(grads[i].data_ptr());
                cum += params[i].numel();
                tbl.offsets[n + 1] = cum;
            }
            tbl.num_tensors = n;
            cudaError_t e =
                launch_multi_tensor_lion_typed<ParamT, StateT, GradT>(
                    tbl, cum, lr, beta1, beta2, wd, stream);
            check_cuda(e, "launch_multi_tensor_lion");
        }
    }
};

} // anonymous namespace

}}} // namespace sg::sm90::lion

// =====================================================================
// Public torch::Tensor launchers in namespace sg::sm90.
// Forward-declared by csrc/bindings/lion.cpp DECLARE_LION(sm90)
// and csrc/bindings/multi_tensor.cpp DECLARE_MT(sm90).
// =====================================================================

namespace sg { namespace sm90 {

void launch_fused_lion_step(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor grad,
    float lr,
    float beta1,
    float beta2,
    float weight_decay
) {
    TORCH_CHECK(param.is_cuda(),   "lion: param must be CUDA");
    TORCH_CHECK(exp_avg.is_cuda(), "lion: exp_avg must be CUDA");
    TORCH_CHECK(grad.is_cuda(),    "lion: grad must be CUDA");
    TORCH_CHECK(param.is_contiguous(),   "lion: param must be contiguous");
    TORCH_CHECK(exp_avg.is_contiguous(), "lion: exp_avg must be contiguous");
    TORCH_CHECK(grad.is_contiguous(),    "lion: grad must be contiguous");
    TORCH_CHECK(param.numel() == grad.numel(),    "lion: param/grad numel mismatch");
    TORCH_CHECK(param.numel() == exp_avg.numel(), "lion: param/exp_avg numel mismatch");

    const int64_t N = param.numel();
    if (N == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sg::sm90::lion::dispatch_lion<sg::sm90::lion::SingleStep>(
        param.scalar_type(), exp_avg.scalar_type(), grad.scalar_type(),
        param, exp_avg, grad, lr, beta1, beta2, weight_decay, N, stream);
}

// ---------------------------------------------------------------------
// Multi-tensor entry. Groups inputs by (param,state,grad) dtype triple,
// then launches one or more batched kernels per group (kMaxTensorsPerLaunch
// tensors per launch). All input vectors are forwarded as const refs into
// the MultiTensor functor; data_ptr() is read inside the functor with the
// resolved dtype, ensuring no runtime dtype branches inside the kernel.
// ---------------------------------------------------------------------

namespace {

struct DtypeKey {
    at::ScalarType p, s, g;
    bool operator==(const DtypeKey& o) const {
        return p == o.p && s == o.s && g == o.g;
    }
};

void launch_multi_tensor_lion_impl(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& exp_avgs,
    const std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float wd
) {
    const size_t T = params.size();
    TORCH_CHECK(exp_avgs.size() == T && grads.size() == T,
                "lion multi-tensor: vector size mismatch");
    if (T == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    std::vector<DtypeKey> keys;
    std::vector<std::vector<size_t>> groups;
    for (size_t i = 0; i < T; ++i) {
        if (!params[i].defined() || params[i].numel() == 0) continue;
        TORCH_CHECK(params[i].is_cuda() && exp_avgs[i].is_cuda() && grads[i].is_cuda(),
                    "lion multi-tensor: tensors must be CUDA");
        TORCH_CHECK(params[i].is_contiguous() && exp_avgs[i].is_contiguous() &&
                    grads[i].is_contiguous(),
                    "lion multi-tensor: tensors must be contiguous");
        TORCH_CHECK(params[i].numel() == grads[i].numel() &&
                    params[i].numel() == exp_avgs[i].numel(),
                    "lion multi-tensor: per-index numel mismatch");
        DtypeKey k{params[i].scalar_type(), exp_avgs[i].scalar_type(),
                   grads[i].scalar_type()};
        size_t gi = keys.size();
        for (size_t j = 0; j < keys.size(); ++j) {
            if (keys[j] == k) { gi = j; break; }
        }
        if (gi == keys.size()) { keys.push_back(k); groups.emplace_back(); }
        groups[gi].push_back(i);
    }

    for (size_t gi = 0; gi < keys.size(); ++gi) {
        const DtypeKey& k = keys[gi];
        sg::sm90::lion::dispatch_lion<sg::sm90::lion::MultiTensor>(
            k.p, k.s, k.g,
            params, exp_avgs, grads, groups[gi],
            lr, beta1, beta2, wd, stream);
    }
}

} // anonymous namespace

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

}} // namespace sg::sm90
