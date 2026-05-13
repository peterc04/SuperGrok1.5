// csrc/kernels/hip/gfx942/models/mamba.hip.h
// Mamba (selective state-space) model header for gfx942 (CDNA3 / MI300X).
//
// Strategy: this is a thin wrapper around the sm_90 implementation.
// mamba.cuh is platform-portable on the cuBLAS/rocBLAS path. The
// underlying scan kernels are __device__ __forceinline__ device
// functions with no sm_90-only intrinsics (no WGMMA, no clusters, no
// TMA), so they compile cleanly under HIP. CUTLASS is gated behind
// `WITH_CUTLASS` and is NOT defined on the HIP build.
//
// The bindings (csrc/bindings/models_mamba.cpp) call
// `sg::gfx942::models::mamba::{forward,backward,selective_scan_fwd,
// selective_scan_bwd}<T>` directly when `detect_arch() == 942`.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
// Bring in the full sm_90 Mamba template implementation. On HIP the
// cuBLAS includes are hipified to rocBLAS by PyTorch's CUDAExtension.
#include "csrc/kernels/cuda/sm_90/models/mamba.cuh"

namespace sg { namespace gfx942 { namespace models { namespace mamba {

// -- Model configuration ------------------------------------------------------
struct ModelConfig {
    int d_model;
    int d_state;
    int d_inner;        // expansion dim (typically 2 * d_model)
    int d_conv;         // local convolution width
    int n_layers;
    int seq_len;
    int batch;
    int lds_bytes;      // LDS allocation budget
    int waves_per_eu;   // occupancy hint for CDNA3
    bool use_bf16_mfma; // BF16 MFMA fast path (d_inner >= 128)
};

// -- Forward pass (full stack) ------------------------------------------------
template <typename T>
inline cudaError_t forward(
    const T* input,
    const T* weights,
    T* output,
    T* states,
    int batch, int seq_len, int d_model, int d_state,
    int d_conv, int expand, int n_layers,
    cudaStream_t stream
) {
    return sg::sm90::models::mamba::forward<T>(
        input, weights, output, states,
        batch, seq_len, d_model, d_state,
        d_conv, expand, n_layers, stream);
}

// -- Backward pass (full stack) -----------------------------------------------
template <typename T>
inline cudaError_t backward(
    const T* grad_output,
    const T* activations_saved,
    const T* weights,
    T* grad_input,
    T* grad_weights,
    int batch, int seq_len, int d_model, int d_state,
    int d_conv, int expand, int n_layers,
    cudaStream_t stream
) {
    return sg::sm90::models::mamba::backward<T>(
        grad_output, activations_saved, weights,
        grad_input, grad_weights,
        batch, seq_len, d_model, d_state,
        d_conv, expand, n_layers, stream);
}

// -- selective_scan_fwd / bwd (component-test wrappers) -----------------------
// Signature contract (matches the binding forward declaration in
// csrc/bindings/models_mamba.cpp):
//   fwd: (u, delta, A, B,            out, state)
//   bwd: (grad_out, u, delta, A, B, state, grad_u, grad_delta, grad_A, grad_B)
template <typename T>
inline cudaError_t selective_scan_fwd(
    const T* u, const T* delta, const T* A, const T* B,
    T* out, T* state,
    int batch, int seq_len, int d_state,
    cudaStream_t stream
) {
    return sg::sm90::models::mamba::selective_scan_fwd<T>(
        u, delta, A, B, out, state,
        batch, seq_len, d_state, stream);
}

template <typename T>
inline cudaError_t selective_scan_bwd(
    const T* grad_out,
    const T* u, const T* delta, const T* A,
    const T* B, const T* state,
    T* grad_u, T* grad_delta, T* grad_A, T* grad_B,
    int batch, int seq_len, int d_state,
    cudaStream_t stream
) {
    return sg::sm90::models::mamba::selective_scan_bwd<T>(
        grad_out, u, delta, A, B, state,
        grad_u, grad_delta, grad_A, grad_B,
        batch, seq_len, d_state, stream);
}

}}}}  // namespace sg::gfx942::models::mamba
