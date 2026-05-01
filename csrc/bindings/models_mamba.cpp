// =====================================================================
//  bindings/models_mamba.cpp — pybind11 wrapper for Mamba (SSM) model
//
//  Follows the SG_DISPATCH pattern used by optimizer bindings. Provides
//  forward/backward entry points for the full Mamba model, per-layer
//  forward, and selective scan primitives for component-level testing.
//
//  Per-arch launchers live in csrc/kernels/{cuda/<sm>,hip/<gfx>}/ and
//  are forward-declared here. Until those TUs are compiled, all entry
//  points raise with a clear diagnostic.
//
//  Pybind11 registration is done from models_module.cpp.
// =====================================================================

#include "bindings.h"
#include "_dispatch_macro.h"

#include <torch/extension.h>

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
    TORCH_CHECK(false, "mamba_forward: per-arch kernel not yet compiled");
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
    TORCH_CHECK(false, "mamba_backward: per-arch kernel not yet compiled");
}

void mamba_layer_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor output,
    torch::Tensor state,
    int batch, int seq_len, int d_model, int d_state,
    int d_conv, int expand
) {
    TORCH_CHECK(false, "mamba_layer_forward: per-arch kernel not yet compiled");
}

void mamba_selective_scan_forward(
    torch::Tensor u, torch::Tensor delta,
    torch::Tensor A, torch::Tensor B,
    torch::Tensor out, torch::Tensor state,
    int batch, int seq_len, int d_state
) {
    TORCH_CHECK(false, "mamba_selective_scan_forward: per-arch kernel not yet compiled");
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
    TORCH_CHECK(false, "mamba_selective_scan_backward: per-arch kernel not yet compiled");
}

} // namespace sg
