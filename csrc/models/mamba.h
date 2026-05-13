#pragma once
// Mamba — vendor-neutral model definition.
//
// Selective state-space model for the sequential chained-division grokking
// task. Architecture per layer:
//   - Input projection: x -> [x_branch, z_branch]
//   - 1D depthwise convolution (kernel size 4) + SiLU on x_branch
//   - x_proj: x_conv -> [B, C, dt_raw]   (selective parameters)
//   - dt_proj + softplus -> dt
//   - selective_scan(x_conv, dt, A_log, B, C, D) -> y    (uses csrc/scan/)
//   - Gating: y_gated = y * silu(z_branch)
//   - Output projection: y_gated -> [d_model]
//   - Residual + LayerNorm
//
// Per-backend implementations live in:
//   csrc/backends/cuda/sm_90/models/mamba.cu
//   csrc/backends/hip/gfx942/models/mamba.hip.cpp
//   csrc/backends/pallas/models/mamba.py
//
// The selective scan implementation is shared across the model and the
// SuperGrok v2 optimizer (csrc/scan/mamba_scan_adapter.cuh).

#include "csrc/common/types.h"

namespace sg { namespace models { namespace mamba {

struct MambaConfig {
    int vocab_size;
    int seq_len;            // sequence length (e.g. 4)
    int d_model;            // hidden dim
    int n_layers;
    int d_state;            // SSM state dim (default 16)
    int d_conv;             // 1D conv kernel size (default 4)
    int expand_factor;      // d_inner = expand_factor * d_model (default 2)
};

struct MambaLayerWeights {
    const float* in_proj_W;     // [2*d_inner, d_model]
    const float* conv1d_W;      // [d_inner, d_conv]
    const float* conv1d_b;      // [d_inner]
    const float* x_proj_W;      // [d_state + d_state + d_inner, d_inner]
    const float* dt_proj_W;     // [d_inner, d_inner]
    const float* dt_proj_b;     // [d_inner]
    const float* A_log;         // [d_inner, d_state]
    const float* D;             // [d_inner]
    const float* out_proj_W;    // [d_model, d_inner]
    const float* ln_w;          // [d_model]
    const float* ln_b;          // [d_model]
};

}}} // namespace sg::models::mamba
