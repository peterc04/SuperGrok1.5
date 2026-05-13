#pragma once
// Vision Transformer — vendor-neutral model definition.
//
// Image classification transformer for the MNIST-addition grokking task.
// Architecture:
//   - Patch projection: 16 patches of dim 49 -> [16, d_model]
//   - CLS token prepend -> [17, d_model]
//   - Learned positional embedding addition
//   - Stacked encoder blocks (non-causal attention + FFN)
//   - Classification head on [CLS] token
//
// Per-backend implementations live in:
//   csrc/backends/cuda/sm_90/models/vit.cu
//   csrc/backends/hip/gfx942/models/vit.hip.cpp
//   csrc/backends/pallas/models/vit.py

#include "csrc/common/types.h"

namespace sg { namespace models { namespace vit {

struct ViTConfig {
    int num_classes;        // e.g. 97 (modular addition base p)
    int num_patches;        // e.g. 16
    int patch_dim;          // e.g. 49 (7x7 sub-image)
    int d_model;            // hidden dim
    int n_layers;
    int n_heads;
    int d_ff;
};

struct ViTLayerWeights {
    const float* qkv_W;
    const float* qkv_b;
    const float* out_W;
    const float* out_b;
    const float* ffn_W1;
    const float* ffn_b1;
    const float* ffn_W2;
    const float* ffn_b2;
    const float* ln1_w;
    const float* ln1_b;
    const float* ln2_w;
    const float* ln2_b;
};

struct ViTGlobalWeights {
    const float* patch_proj_W;   // [d_model, patch_dim]
    const float* patch_proj_b;   // [d_model]
    const float* cls_token;      // [d_model]
    const float* pos_embed;      // [num_patches + 1, d_model]
    const float* head_W;         // [num_classes, d_model]
    const float* head_b;         // [num_classes]
};

}}} // namespace sg::models::vit
