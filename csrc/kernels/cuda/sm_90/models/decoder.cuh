// csrc/kernels/cuda/sm_90/models/decoder.cuh
// Autoregressive decoder model header for sm_90 (Hopper).
// Uses shared attention kernel with kCausal=true.
//
// Stub implementation — full fused forward/backward will be filled
// by optimizer-specific agents.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/kernels/cuda/sm_90/models/attention.cuh"

namespace sg { namespace sm90 { namespace models { namespace decoder {

// ── Model configuration ──────────────────────────────────────────────
struct ModelConfig {
    int batch_size;
    int seq_len;           // typically 4 for grokking
    int d_model;
    int n_heads;
    int d_head;            // typically 32 for grokking
    int n_layers;
    int vocab_size;
    float ffn_expansion;   // FFN hidden = d_model * ffn_expansion

    // Decoder-specific
    int max_seq_len;       // maximum position for positional encoding
    bool use_rope;         // apply rotary positional embeddings
    float dropout_p;       // attention dropout (0 during eval)
};

// ── Weight buffer layout ─────────────────────────────────────────────
// Opaque pointer — layout determined by the fused kernel implementation.
// Contains embedding, per-layer QKV projections, output projection,
// FFN weights, layer norms, and unembedding head.

// ── Activations saved for backward ──────────────────────────────────
// Opaque pointer — the forward pass writes intermediate activations
// (pre-norm, QKV, softmax_lse, FFN hidden, residual) for backward.

// ── Forward pass ─────────────────────────────────────────────────────
// Runs the full decoder stack: embedding -> n_layers x (attn + FFN) -> head.
// Attention uses the shared kernel with kCausal=true.

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
    //      a. LayerNorm (pre-norm)
    //      b. QKV projection
    //      c. attention::attention_forward<ActT, d_head, /*kCausal=*/true>(...)
    //      d. Output projection + residual
    //      e. LayerNorm (pre-norm)
    //      f. FFN (up-proj, activation, down-proj) + residual
    //      g. Save activations needed for backward
    //   3. Final LayerNorm
    //   4. Unembedding head projection -> logits
    return cudaErrorNotReady;  // stub
}

// ── Backward pass ────────────────────────────────────────────────────
// Reverse-mode through the decoder stack.

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
    //   2. Grad through final LayerNorm
    //   3. For each layer (reverse order):
    //      a. Grad through FFN + residual
    //      b. Grad through LayerNorm
    //      c. Grad through output projection
    //      d. attention::attention_backward<ActT, d_head, /*kCausal=*/true>(...)
    //      e. Grad through QKV projection
    //      f. Grad through LayerNorm + residual
    //   4. Grad through embedding
    return cudaErrorNotReady;  // stub
}

}}}}  // namespace sg::sm90::models::decoder
