// =====================================================================
//  bindings/models_mamba.cpp — pybind11 wrapper for Mamba (SSM) model
//
//  Follows the SG_DISPATCH pattern used by optimizer bindings. Provides
//  forward/backward entry points for the full Mamba model, per-layer
//  forward, and selective scan primitives for component-level testing.
//
//  Per-arch launchers live in csrc/kernels/{cuda/<sm>,hip/<gfx>}/ and
//  are forward-declared here.
//
//  Pybind11 registration is done from models_module.cpp.
// =====================================================================

#include "bindings.h"
#include "_dispatch_macro.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------
// Forward declarations for per-arch launchers.
// ---------------------------------------------------------------------

namespace sg { namespace sm90 { namespace models { namespace mamba {
    template <typename T>
    cudaError_t forward(const T*, const T*, T*, T*,
                       int batch, int seq_len, int d_model, int d_state,
                       int d_conv, int expand, int n_layers, cudaStream_t);
    template <typename T>
    cudaError_t backward(const T*, const T*, const T*,
                        T*, T*,
                        int batch, int seq_len, int d_model, int d_state,
                        int d_conv, int expand, int n_layers, cudaStream_t);
    template <typename T>
    cudaError_t selective_scan_fwd(const T*, const T*, const T*, const T*,
                                  T*, T*,
                                  int batch, int seq_len, int d_state,
                                  cudaStream_t);
    template <typename T>
    cudaError_t selective_scan_bwd(const T*, const T*, const T*, const T*,
                                  const T*, const T*,
                                  T*, T*, T*, T*,
                                  int batch, int seq_len, int d_state,
                                  cudaStream_t);
}}}}

namespace sg { namespace gfx942 { namespace models { namespace mamba {
    template <typename T>
    cudaError_t forward(const T*, const T*, T*, T*,
                       int batch, int seq_len, int d_model, int d_state,
                       int d_conv, int expand, int n_layers, cudaStream_t);
    template <typename T>
    cudaError_t backward(const T*, const T*, const T*,
                        T*, T*,
                        int batch, int seq_len, int d_model, int d_state,
                        int d_conv, int expand, int n_layers, cudaStream_t);
    template <typename T>
    cudaError_t selective_scan_fwd(const T*, const T*, const T*, const T*,
                                  T*, T*,
                                  int batch, int seq_len, int d_state,
                                  cudaStream_t);
    template <typename T>
    cudaError_t selective_scan_bwd(const T*, const T*, const T*, const T*,
                                  const T*, const T*,
                                  T*, T*, T*, T*,
                                  int batch, int seq_len, int d_state,
                                  cudaStream_t);
}}}}

// ---------------------------------------------------------------------
// Helpers: cast typed pointer view of a torch tensor.
// ---------------------------------------------------------------------

namespace {

template <typename T>
inline T* tptr(torch::Tensor& t)             { return reinterpret_cast<T*>(t.data_ptr()); }
template <typename T>
inline const T* ctptr(const torch::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }

inline cudaStream_t cur_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

inline void check_cuda(cudaError_t err, const char* what) {
    TORCH_CHECK(err == cudaSuccess,
                "Mamba kernel ", what, " failed: ", cudaGetErrorString(err));
}

}  // anonymous namespace

// ---------------------------------------------------------------------
// Public entry points (called from models_module.cpp registration)
// ---------------------------------------------------------------------

namespace sg {

void mamba_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor output,
    torch::Tensor states,
    int batch, int seq_len, int d_model, int d_state,
    int d_conv, int expand, int n_layers
) {
    TORCH_CHECK(input.is_cuda() && weights.is_cuda() && output.is_cuda()
                && states.is_cuda(),
                "mamba_forward: all tensors must be on CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kInt32,
                "mamba_forward: input (token ids) must be int32");
    auto stream = cur_stream();
    const int sg_arch = ::sg::detect_arch();

    auto run = [&](auto dummy) {
        using scalar_t = decltype(dummy);
        cudaError_t err;
        if (sg_arch == 90) {
            err = ::sg::sm90::models::mamba::forward<scalar_t>(
                reinterpret_cast<const scalar_t*>(input.data_ptr()),
                ctptr<scalar_t>(weights),
                tptr<scalar_t>(output),
                tptr<scalar_t>(states),
                batch, seq_len, d_model, d_state,
                d_conv, expand, n_layers, stream);
        } else if (sg_arch == 942) {
            err = ::sg::gfx942::models::mamba::forward<scalar_t>(
                reinterpret_cast<const scalar_t*>(input.data_ptr()),
                ctptr<scalar_t>(weights),
                tptr<scalar_t>(output),
                tptr<scalar_t>(states),
                batch, seq_len, d_model, d_state,
                d_conv, expand, n_layers, stream);
        } else {
            TORCH_CHECK(false, "mamba_forward: unsupported arch ", sg_arch);
        }
        check_cuda(err, "forward");
    };

    switch (weights.scalar_type()) {
        case torch::kFloat32: run(float{}); break;
        case torch::kFloat16: run(__half{}); break;
        case torch::kBFloat16: run(__nv_bfloat16{}); break;
        default:
            TORCH_CHECK(false, "mamba_forward: unsupported weight dtype ",
                        weights.scalar_type());
    }
}

void mamba_backward(
    torch::Tensor grad_output,
    torch::Tensor states_saved,
    torch::Tensor weights,
    torch::Tensor grad_input,
    torch::Tensor grad_weights,
    int batch, int seq_len, int d_model, int d_state,
    int d_conv, int expand, int n_layers
) {
    TORCH_CHECK(grad_output.is_cuda() && weights.is_cuda(),
                "mamba_backward: tensors must be on CUDA");
    auto stream = cur_stream();
    const int sg_arch = ::sg::detect_arch();

    auto run = [&](auto dummy) {
        using scalar_t = decltype(dummy);
        cudaError_t err;
        if (sg_arch == 90) {
            err = ::sg::sm90::models::mamba::backward<scalar_t>(
                ctptr<scalar_t>(grad_output),
                ctptr<scalar_t>(states_saved),
                ctptr<scalar_t>(weights),
                tptr<scalar_t>(grad_input),
                tptr<scalar_t>(grad_weights),
                batch, seq_len, d_model, d_state,
                d_conv, expand, n_layers, stream);
        } else if (sg_arch == 942) {
            err = ::sg::gfx942::models::mamba::backward<scalar_t>(
                ctptr<scalar_t>(grad_output),
                ctptr<scalar_t>(states_saved),
                ctptr<scalar_t>(weights),
                tptr<scalar_t>(grad_input),
                tptr<scalar_t>(grad_weights),
                batch, seq_len, d_model, d_state,
                d_conv, expand, n_layers, stream);
        } else {
            TORCH_CHECK(false, "mamba_backward: unsupported arch ", sg_arch);
        }
        check_cuda(err, "backward");
    };

    switch (weights.scalar_type()) {
        case torch::kFloat32: run(float{}); break;
        case torch::kFloat16: run(__half{}); break;
        case torch::kBFloat16: run(__nv_bfloat16{}); break;
        default:
            TORCH_CHECK(false, "mamba_backward: unsupported weight dtype ",
                        weights.scalar_type());
    }
}

void mamba_layer_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor output,
    torch::Tensor state,
    int batch, int seq_len, int d_model, int d_state,
    int d_conv, int expand
) {
    // Per-layer forward: a single-layer forward pass routed via the same
    // stack with n_layers=1. The caller is responsible for providing weights
    // shaped for a single layer (i.e. embeddings sized for an identity
    // pass-through, or just the layer block).
    mamba_forward(input, weights, output, state,
                  batch, seq_len, d_model, d_state,
                  d_conv, expand, /*n_layers=*/1);
}

void mamba_selective_scan_forward(
    torch::Tensor u, torch::Tensor delta,
    torch::Tensor A, torch::Tensor B,
    torch::Tensor out, torch::Tensor state,
    int batch, int seq_len, int d_state
) {
    TORCH_CHECK(u.is_cuda() && delta.is_cuda() && A.is_cuda() && B.is_cuda()
                && out.is_cuda() && state.is_cuda(),
                "mamba_selective_scan_forward: all tensors must be on CUDA");
    auto stream = cur_stream();
    const int sg_arch = ::sg::detect_arch();

    auto run = [&](auto dummy) {
        using scalar_t = decltype(dummy);
        cudaError_t err;
        if (sg_arch == 90) {
            err = ::sg::sm90::models::mamba::selective_scan_fwd<scalar_t>(
                ctptr<scalar_t>(u),  ctptr<scalar_t>(delta),
                ctptr<scalar_t>(A),  ctptr<scalar_t>(B),
                tptr<scalar_t>(out), tptr<scalar_t>(state),
                batch, seq_len, d_state, stream);
        } else if (sg_arch == 942) {
            err = ::sg::gfx942::models::mamba::selective_scan_fwd<scalar_t>(
                ctptr<scalar_t>(u),  ctptr<scalar_t>(delta),
                ctptr<scalar_t>(A),  ctptr<scalar_t>(B),
                tptr<scalar_t>(out), tptr<scalar_t>(state),
                batch, seq_len, d_state, stream);
        } else {
            TORCH_CHECK(false,
                "mamba_selective_scan_forward: unsupported arch ", sg_arch);
        }
        check_cuda(err, "selective_scan_forward");
    };

    switch (u.scalar_type()) {
        case torch::kFloat32: run(float{}); break;
        case torch::kFloat16: run(__half{}); break;
        case torch::kBFloat16: run(__nv_bfloat16{}); break;
        default:
            TORCH_CHECK(false,
                "mamba_selective_scan_forward: unsupported dtype ",
                u.scalar_type());
    }
}

void mamba_selective_scan_backward(
    torch::Tensor grad_out,
    torch::Tensor u, torch::Tensor delta,
    torch::Tensor A, torch::Tensor B,
    torch::Tensor out, torch::Tensor state,
    torch::Tensor grad_u, torch::Tensor grad_delta,
    torch::Tensor grad_A, torch::Tensor grad_B,
    int batch, int seq_len, int d_state
) {
    (void)out;  // not used: backward recomputes h from forward inputs
    TORCH_CHECK(grad_out.is_cuda() && u.is_cuda() && delta.is_cuda()
                && A.is_cuda() && B.is_cuda() && state.is_cuda(),
                "mamba_selective_scan_backward: all tensors must be on CUDA");
    auto stream = cur_stream();
    const int sg_arch = ::sg::detect_arch();

    auto run = [&](auto dummy) {
        using scalar_t = decltype(dummy);
        cudaError_t err;
        if (sg_arch == 90) {
            err = ::sg::sm90::models::mamba::selective_scan_bwd<scalar_t>(
                ctptr<scalar_t>(grad_out),
                ctptr<scalar_t>(u), ctptr<scalar_t>(delta),
                ctptr<scalar_t>(A), ctptr<scalar_t>(B),
                ctptr<scalar_t>(state),
                tptr<scalar_t>(grad_u), tptr<scalar_t>(grad_delta),
                tptr<scalar_t>(grad_A), tptr<scalar_t>(grad_B),
                batch, seq_len, d_state, stream);
        } else if (sg_arch == 942) {
            err = ::sg::gfx942::models::mamba::selective_scan_bwd<scalar_t>(
                ctptr<scalar_t>(grad_out),
                ctptr<scalar_t>(u), ctptr<scalar_t>(delta),
                ctptr<scalar_t>(A), ctptr<scalar_t>(B),
                ctptr<scalar_t>(state),
                tptr<scalar_t>(grad_u), tptr<scalar_t>(grad_delta),
                tptr<scalar_t>(grad_A), tptr<scalar_t>(grad_B),
                batch, seq_len, d_state, stream);
        } else {
            TORCH_CHECK(false,
                "mamba_selective_scan_backward: unsupported arch ", sg_arch);
        }
        check_cuda(err, "selective_scan_backward");
    };

    switch (u.scalar_type()) {
        case torch::kFloat32: run(float{}); break;
        case torch::kFloat16: run(__half{}); break;
        case torch::kBFloat16: run(__nv_bfloat16{}); break;
        default:
            TORCH_CHECK(false,
                "mamba_selective_scan_backward: unsupported dtype ",
                u.scalar_type());
    }
}

} // namespace sg
