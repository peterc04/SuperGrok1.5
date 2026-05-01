// csrc/kernels/cuda/sm_90/models/mamba.cuh
// Mamba (selective state-space) model header for sm_90 (Hopper).
// Uses mamba_scan_adapter.cuh for the core selective scan.
//
// Stub implementation -- full fused forward/backward will be filled
// by optimizer-specific agents.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/kernels/cuda/sm_90/models/mamba_scan_adapter.cuh"

namespace sg { namespace sm90 { namespace models { namespace mamba {

// -- Model configuration --------------------------------------------------
struct ModelConfig {
    int batch_size;
    int seq_len;
    int d_model;
    int n_heads;           // unused by Mamba but kept for uniform interface
    int d_head;            // unused by Mamba but kept for uniform interface
    int n_layers;
    int vocab_size;
    float ffn_expansion;   // expansion factor for d_inner = d_model * ffn_expansion

    // Mamba-specific
    int d_inner;           // inner SSM dimension (typically d_model * expand)
    int d_state;           // SSM state dimension (typically 16)
    int d_conv;            // local convolution width (typically 4)
    float dt_min;          // minimum delta-t for softplus clamping
    float dt_max;          // maximum delta-t for softplus clamping
    int expand;            // expansion factor (d_inner = d_model * expand)
};

// -- Weight buffer layout -------------------------------------------------
// Opaque pointer -- layout determined by the fused kernel implementation.
// Contains embedding, per-layer in_proj (for x and z), conv1d weights,
// dt/B/C projection weights, A_log parameters, D parameter, out_proj,
// and unembedding head.

// -- Activations saved for backward --------------------------------------
// Opaque pointer -- the forward pass writes intermediate activations
// (conv state, SSM state, z gate, residuals) for backward.

// -- Forward pass ---------------------------------------------------------
// Runs the full Mamba stack:
//   embedding -> n_layers x (norm + mamba_block) -> norm -> head.
// Each mamba_block:
//   in_proj -> conv1d -> SiLU -> dt/B/C proj -> selective_scan -> mul(z) -> out_proj

template <typename ActT, typename WeightT>
cudaError_t forward(
    const ActT* input,               // [B, seq_len] token ids cast to ActT
    const WeightT* weights,          // flattened weight buffer
    ActT* output,                    // [B, seq_len, vocab_size] logits
    ActT* activations_for_backward,  // scratch for saved activations
    const ModelConfig& cfg,
    cudaStream_t stream
) {
    // TODO: Implementation outline:
    //   1. Token embedding lookup
    //   2. For each layer:
    //      a. LayerNorm / RMSNorm (pre-norm)
    //      b. in_proj: project d_model -> 2*d_inner (split into x, z)
    //      c. conv1d on x (causal, width = d_conv)
    //      d. SiLU activation on conv output
    //      e. Project x -> dt, B, C
    //      f. Softplus on dt (clamped to [dt_min, dt_max])
    //      g. mamba_adapter::selective_scan_forward<ActT>(
    //             x, dt, A_log, B, C, y, state_save, ...)
    //      h. y = y * SiLU(z)  (gating)
    //      i. out_proj: project d_inner -> d_model
    //      j. Residual connection
    //      k. Save activations needed for backward
    //   3. Final LayerNorm / RMSNorm
    //   4. Unembedding head projection -> logits
    return cudaErrorNotReady;  // stub
}

// -- Backward pass --------------------------------------------------------
// Reverse-mode through the Mamba stack.

template <typename ActT, typename WeightT>
cudaError_t backward(
    const ActT* grad_output,         // [B, seq_len, vocab_size]
    const ActT* activations_saved,   // from forward
    const WeightT* weights,
    ActT* grad_input,                // [B, seq_len] (grad w.r.t. embeddings)
    WeightT* grad_weights,           // same layout as weights
    const ModelConfig& cfg,
    cudaStream_t stream
) {
    // TODO: Reverse of forward:
    //   1. Grad through unembedding head
    //   2. Grad through final norm
    //   3. For each layer (reverse order):
    //      a. Grad through residual
    //      b. Grad through out_proj
    //      c. Grad through gating (y * SiLU(z))
    //      d. mamba_adapter::selective_scan_backward<ActT>(
    //             grad_y, x, dt, A_log, B, C, state_save,
    //             grad_x, grad_dt, grad_A_log, grad_B, grad_C, ...)
    //      e. Grad through dt/B/C projections
    //      f. Grad through SiLU + conv1d
    //      g. Grad through in_proj
    //      h. Grad through norm + residual
    //   4. Grad through embedding
    return cudaErrorNotReady;  // stub
}

}}}}  // namespace sg::sm90::models::mamba
