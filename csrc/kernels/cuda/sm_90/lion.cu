// =====================================================================
//  csrc/kernels/cuda/sm_90/lion.cu
//
//  Thin instantiation TU for the sm_90 Lion optimizer kernels. All actual
//  __global__ + __device__ code lives in lion.cuh; this TU just emits
//  - explicit template instantiations for every coherent dtype combo,
//  - the torch::Tensor-facing launchers (single-tensor + multi-tensor),
//  - the AT_DISPATCH glue that routes runtime ScalarType -> typed call.
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
// AT_DISPATCH glue: at::ScalarType -> typed C++ type.
// ---------------------------------------------------------------------

namespace {

// Dispatch on (param, state, grad) ScalarType. State only supports FP32 /
// BF16; we throw on anything else. ParamT is FP32 / BF16 / FP16; GradT is
// any dtype in the matrix. Bool/int dtypes are rejected.
template <typename Fn>
void dispatch_lion(
    at::ScalarType param_dtype,
    at::ScalarType state_dtype,
    at::ScalarType grad_dtype,
    Fn&& fn
) {
#define DTYPE_CASE_GRAD(P, S)                                                  \
    switch (grad_dtype) {                                                      \
        case at::ScalarType::Float:           fn.template operator()<P, S, float>();          break; \
        case at::ScalarType::BFloat16:        fn.template operator()<P, S, __nv_bfloat16>();  break; \
        case at::ScalarType::Half:            fn.template operator()<P, S, __half>();         break; \
        case at::ScalarType::Float8_e4m3fn:   fn.template operator()<P, S, __nv_fp8_e4m3>();  break; \
        case at::ScalarType::Float8_e5m2:     fn.template operator()<P, S, __nv_fp8_e5m2>();  break; \
        default: throw std::runtime_error(                                     \
            std::string("lion: unsupported grad dtype ") +                     \
            c10::toString(grad_dtype));                                        \
    }

#define DTYPE_CASE_STATE(P)                                                    \
    switch (state_dtype) {                                                     \
        case at::ScalarType::Float:    { DTYPE_CASE_GRAD(P, float)         break; } \
        case at::ScalarType::BFloat16: { DTYPE_CASE_GRAD(P, __nv_bfloat16) break; } \
        default: throw std::runtime_error(                                     \
            std::string("lion: unsupported state dtype ") +                    \
            c10::toString(state_dtype) + " (must be FP32 or BF16)");           \
    }

    switch (param_dtype) {
        case at::ScalarType::Float:    { DTYPE_CASE_STATE(float)         break; }
        case at::ScalarType::BFloat16: { DTYPE_CASE_STATE(__nv_bfloat16) break; }
        case at::ScalarType::Half:     { DTYPE_CASE_STATE(__half)        break; }
        default: throw std::runtime_error(
            std::string("lion: unsupported param dtype ") +
            c10::toString(param_dtype) + " (must be FP32, BF16, or FP16)");
    }
#undef DTYPE_CASE_STATE
#undef DTYPE_CASE_GRAD
}

inline void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("lion ") + where + ": " + cudaGetErrorString(e));
    }
}

} // anonymous namespace

}}} // namespace sg::sm90::lion

// =====================================================================
// Public torch::Tensor launchers in namespace sg::sm90.
// Forward-declared by csrc/bindings/lion.cpp DECLARE_LION(sm90).
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

    lion::dispatch_lion(
        param.scalar_type(), exp_avg.scalar_type(), grad.scalar_type(),
        [&]<typename ParamT, typename StateT, typename GradT>() {
            cudaError_t e = lion::launch_lion_typed<ParamT, StateT, GradT>(
                static_cast<ParamT*>(param.data_ptr()),
                static_cast<StateT*>(exp_avg.data_ptr()),
                static_cast<const GradT*>(grad.data_ptr()),
                lr, beta1, beta2, weight_decay,
                N, stream);
            lion::check_cuda(e, "launch_fused_lion_step");
        });
}

// ---------------------------------------------------------------------
// Multi-tensor implementation. Packs all input tensors into batches of
// kMaxTensorsPerLaunch and issues one kernel per batch. All tensors in a
// single batch must share the (param, state, grad) dtype triple; the
// outer loop groups by dtype-triple before packing.
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

    // Group indices by (param, state, grad) dtype triple.
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
        const auto& idxs = groups[gi];
        const DtypeKey& k = keys[gi];

        sg::sm90::lion::dispatch_lion(
            k.p, k.s, k.g,
            [&]<typename ParamT, typename StateT, typename GradT>() {
                using Tbl = sg::sm90::lion::LionTensorTable<ParamT, StateT, GradT>;
                constexpr int kMax = sg::sm90::lion::kMaxTensorsPerLaunch;
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
                        sg::sm90::lion::launch_multi_tensor_lion_typed<ParamT, StateT, GradT>(
                            tbl, cum, lr, beta1, beta2, wd, stream);
                    sg::sm90::lion::check_cuda(e, "launch_multi_tensor_lion");
                }
            });
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
