// csrc/kernels/cuda/sm_90/models/vit.cuh
// Vision Transformer model header for sm_90 (Hopper).
// Uses shared attention kernel with kCausal=false (bidirectional).
//
// Stub implementation -- full fused forward/backward will be filled
// by optimizer-specific agents.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/kernels/cuda/sm_90/models/attention.cuh"

namespace sg { namespace sm90 { namespace models { namespace vit {

// -- Model configuration --------------------------------------------------
struct ModelConfig {
    int batch_size;
    int seq_len;           // number of patches + 1 (CLS token), typically 17
    int d_model;
    int n_heads;
    int d_head;            // typically 32 for grokking
    int n_layers;
    int vocab_size;        // number of patch classes (or 0 if continuous)
    float ffn_expansion;   // FFN hidden = d_model * ffn_expansion

    // ViT-specific
    int image_size;        // input image resolution (square)
    int patch_size;        // patch side length; num_patches = (image_size/patch_size)^2
    int n_channels;        // input image channels (e.g. 3 for RGB)
    bool use_cls_token;    // prepend learnable [CLS] token
    float dropout_p;       // attention dropout (0 during eval)
};

// -- Weight buffer layout -------------------------------------------------
// Opaque pointer -- layout determined by the fused kernel implementation.
// Contains patch embedding conv, positional embedding, per-layer QKV
// projections, output projection, FFN weights, layer norms, and
// classification head.

// -- Activations saved for backward --------------------------------------
// Opaque pointer -- the forward pass writes intermediate activations
// (pre-norm, QKV, softmax_lse, FFN hidden, residual) for backward.

// -- Forward pass ---------------------------------------------------------
// Runs the full ViT stack:
//   patch_embed -> pos_embed -> n_layers x (attn + FFN) -> head.
// Attention uses the shared kernel with kCausal=false (bidirectional).

template <typename ActT, typename WeightT>
cudaError_t forward(
    const ActT* input,               // [B, C, H, W] image or [B, seq_len] tokens
    const WeightT* weights,          // flattened weight buffer
    ActT* output,                    // [B, n_classes] or [B, seq_len, d_model]
    ActT* activations_for_backward,  // scratch for saved activations
    const ModelConfig& cfg,
    cudaStream_t stream
) {
    // TODO: Implementation outline:
    //   1. Patch embedding (conv2d or linear projection)
    //   2. Prepend [CLS] token if use_cls_token
    //   3. Add positional embedding
    //   4. For each layer:
    //      a. LayerNorm (pre-norm)
    //      b. QKV projection
    //      c. attention::attention_forward<ActT, d_head, /*kCausal=*/false>(...)
    //      d. Output projection + residual
    //      e. LayerNorm (pre-norm)
    //      f. FFN (up-proj, activation, down-proj) + residual
    //      g. Save activations needed for backward
    //   5. Final LayerNorm
    //   6. Extract [CLS] token (or global average pool)
    //   7. Classification head projection
    return cudaErrorNotReady;  // stub
}

// -- Backward pass --------------------------------------------------------
// Reverse-mode through the ViT stack.

template <typename ActT, typename WeightT>
cudaError_t backward(
    const ActT* grad_output,         // [B, n_classes] or [B, seq_len, d_model]
    const ActT* activations_saved,   // from forward
    const WeightT* weights,
    ActT* grad_input,                // [B, C, H, W] or [B, seq_len]
    WeightT* grad_weights,           // same layout as weights
    const ModelConfig& cfg,
    cudaStream_t stream
) {
    // TODO: Reverse of forward:
    //   1. Grad through classification head
    //   2. Grad through [CLS] extraction / global pool
    //   3. Grad through final LayerNorm
    //   4. For each layer (reverse order):
    //      a. Grad through FFN + residual
    //      b. Grad through LayerNorm
    //      c. Grad through output projection
    //      d. attention::attention_backward<ActT, d_head, /*kCausal=*/false>(...)
    //      e. Grad through QKV projection
    //      f. Grad through LayerNorm + residual
    //   5. Grad through positional embedding
    //   6. Grad through patch embedding
    return cudaErrorNotReady;  // stub
}

}}}}  // namespace sg::sm90::models::vit
