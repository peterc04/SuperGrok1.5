// csrc/kernels/hip/gfx942/models/decoder.hip.h
// Decoder model header for gfx942 (CDNA3 / MI300X).
// Uses causal attention (kCausal=true) from attention.hip.h.
//
// Mirrors the sm_90 decoder interface with HIP types.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/kernels/hip/gfx942/models/attention.hip.h"

namespace sg { namespace gfx942 { namespace models { namespace decoder {

// -- Model configuration ------------------------------------------------------
struct ModelConfig {
    int d_model;
    int n_heads;
    int head_dim;
    int n_layers;
    int seq_len;
    int vocab_size;
    int batch;
    float attn_scale;       // typically 1/sqrt(head_dim)
    int lds_bytes;          // LDS allocation budget
    int waves_per_eu;       // occupancy hint for CDNA3
};

// Causal attention alias for decoder
template <typename ActT, int kHeadDim>
using DecoderAttentionConfig =
    attention::AttentionLaunchConfig<ActT, kHeadDim, /*kCausal=*/true>;

// -- Forward pass (stub) ------------------------------------------------------
template <typename ActT, int kHeadDim>
hipError_t forward(
    const ActT* __restrict__ input,     // [B, L, D]
    const ActT* __restrict__ weights,   // layer weights (packed)
    ActT* __restrict__ output,          // [B, L, D]
    ActT* __restrict__ workspace,       // scratch buffer
    const ModelConfig& cfg,
    hipStream_t stream
) {
    return hipErrorNotReady;
}

// -- Backward pass (stub) -----------------------------------------------------
template <typename ActT, int kHeadDim>
hipError_t backward(
    const ActT* __restrict__ grad_out,
    const ActT* __restrict__ input,
    const ActT* __restrict__ weights,
    const ActT* __restrict__ saved_activations,
    ActT* __restrict__ grad_input,
    ActT* __restrict__ grad_weights,
    ActT* __restrict__ workspace,
    const ModelConfig& cfg,
    hipStream_t stream
) {
    return hipErrorNotReady;
}

}}}}  // namespace sg::gfx942::models::decoder
