// csrc/kernels/hip/gfx942/models/mamba.hip.h
// Mamba (selective state-space) model header for gfx942 (CDNA3 / MI300X).
// Wraps mamba_scan_adapter.hip.h into a model-layer interface.
//
// Mirrors the sm_90 Mamba interface with HIP types.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/kernels/hip/gfx942/models/mamba_scan_adapter.hip.h"

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

// -- Forward pass (stub) ------------------------------------------------------
template <typename ActT>
hipError_t forward(
    const ActT* __restrict__ input,         // [B, L, D]
    const ActT* __restrict__ weights,       // layer weights (packed)
    ActT* __restrict__ output,              // [B, L, D]
    ActT* __restrict__ scan_states,         // [B, D_inner, D_state]
    ActT* __restrict__ workspace,           // scratch buffer
    const ModelConfig& cfg,
    hipStream_t stream
) {
    return hipErrorNotReady;
}

// -- Backward pass (stub) -----------------------------------------------------
template <typename ActT>
hipError_t backward(
    const ActT* __restrict__ grad_out,
    const ActT* __restrict__ input,
    const ActT* __restrict__ weights,
    const ActT* __restrict__ scan_states,
    const ActT* __restrict__ saved_activations,
    ActT* __restrict__ grad_input,
    ActT* __restrict__ grad_weights,
    ActT* __restrict__ workspace,
    const ModelConfig& cfg,
    hipStream_t stream
) {
    return hipErrorNotReady;
}

}}}}  // namespace sg::gfx942::models::mamba
