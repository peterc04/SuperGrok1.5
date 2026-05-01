// csrc/kernels/hip/gfx942/models/vit.hip.h
// Vision Transformer model header for gfx942 (CDNA3 / MI300X).
// Uses non-causal attention (kCausal=false) from attention.hip.h.
//
// Mirrors the sm_90 ViT interface with HIP types.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/kernels/hip/gfx942/models/attention.hip.h"

namespace sg { namespace gfx942 { namespace models { namespace vit {

// -- Model configuration ------------------------------------------------------
struct ModelConfig {
    int d_model;
    int n_heads;
    int head_dim;
    int n_layers;
    int seq_len;        // number of patches + CLS token (e.g. 17)
    int patch_size;
    int image_size;
    int num_classes;
    int batch;
    float attn_scale;   // typically 1/sqrt(head_dim)
    int lds_bytes;      // LDS allocation budget
    int waves_per_eu;   // occupancy hint for CDNA3
};

// Non-causal attention alias for ViT
template <typename ActT, int kHeadDim>
using ViTAttentionConfig =
    attention::AttentionLaunchConfig<ActT, kHeadDim, /*kCausal=*/false>;

// -- Forward pass (stub) ------------------------------------------------------
template <typename ActT, int kHeadDim>
hipError_t forward(
    const ActT* __restrict__ input,     // [B, C, H, W] or [B, N, D]
    const ActT* __restrict__ weights,   // layer weights (packed)
    ActT* __restrict__ output,          // [B, num_classes]
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

}}}}  // namespace sg::gfx942::models::vit
