#pragma once
// Decoder Transformer — vendor-neutral model definition.
//
// Autoregressive decoder with post-norm causal self-attention and FFN.
// Architecture for the modular-division grokking task (sequence length 4).
//
// Components per layer:
//   - QKV projection (linear)
//   - Causal self-attention with softmax
//   - Output projection (linear)
//   - Residual + LayerNorm
//   - FFN: linear -> GELU -> linear
//   - Residual + LayerNorm
//
// Per-backend implementations live in:
//   csrc/backends/cuda/sm_90/models/decoder.cu
//   csrc/backends/hip/gfx942/models/decoder.hip.cpp
//   csrc/backends/pallas/models/decoder.py
//
// This header is a contract: every backend exports the same function
// signatures in its respective `sg::<arch>::decoder` namespace.

#include "csrc/common/types.h"

namespace sg { namespace models { namespace decoder {

// Decoder configuration. Compile-time constants where helpful, runtime
// values where flexibility is needed.
struct DecoderConfig {
    int vocab_size;        // e.g. 97 (modular base p)
    int seq_len;           // e.g. 4 (a, op, b, eq)
    int d_model;           // hidden dim (small=128, medium=256, large=512)
    int n_layers;          // transformer depth (2 for grokking race)
    int n_heads;           // attention heads (d_model / 64 typical)
    int d_ff;              // FFN inner dim (4 * d_model)
    bool causal;           // true for autoregressive
};

// Weight pointer layout (one block of contiguous memory per layer).
struct DecoderLayerWeights {
    const float* qkv_W;    // [3 * d_model, d_model]
    const float* qkv_b;    // [3 * d_model]
    const float* out_W;    // [d_model, d_model]
    const float* out_b;    // [d_model]
    const float* ffn_W1;   // [d_ff, d_model]
    const float* ffn_b1;   // [d_ff]
    const float* ffn_W2;   // [d_model, d_ff]
    const float* ffn_b2;   // [d_model]
    const float* ln1_w;    // [d_model]
    const float* ln1_b;    // [d_model]
    const float* ln2_w;    // [d_model]
    const float* ln2_b;    // [d_model]
};

// The actual forward/backward functions are declared in the per-backend
// headers (which differ in tensor type, stream type, and namespace).
// See csrc/backends/<vendor>/<arch>/models/decoder.* for declarations.

}}} // namespace sg::models::decoder
