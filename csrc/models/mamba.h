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

// ── inlined from former csrc/common/types.h ──
/*
 * SuperGrok v2 — Shared Types and Constants
 *
 * Common struct definitions and compile-time constants used by both
 * forward and backward CUDA kernels.
 */



#include <vector>
#include <torch/extension.h>

// ═══════════════════════════════════════════════════════════════════════
//  BatchedScanCtx — shared bookkeeping for the batched mamba peer step
//
//  Recovered verbatim from the deleted csrc/common/ops.h@682eab4^. The
//  setup helper batched_step_setup_and_sort builds one of these per call;
//  the scan + fused-elem helpers consume it. Per-arch variants in
//  csrc/kernels/{cuda,hip}/<arch>/supergrok2_fwd_*.{cu,hip.cpp}
//  expect this exact layout.
//
//  Layout (14 members): 3 int + 2 std::vector<int>
//                     + 8 torch::Tensor + 1 std::vector<torch::Tensor>.
// ═══════════════════════════════════════════════════════════════════════

struct BatchedScanCtx {
    int num_params;
    int total_N;
    int max_N;
    std::vector<int> N_vec;
    std::vector<int> seg_offsets_cpu;
    torch::Tensor x_sorted_packed;
    torch::Tensor offsets_t;
    torch::Tensor initial_fwd;
    torch::Tensor initial_bwd;
    torch::Tensor final_fwd;
    torch::Tensor final_bwd;
    torch::Tensor fwd_scan_packed;
    torch::Tensor bwd_scan_packed;
    std::vector<torch::Tensor> unsort_idx_list;
};

// ═══════════════════════════════════════════════════════════════════════
//  Compile-time constants
// ═══════════════════════════════════════════════════════════════════════

constexpr int MAX_D_STATE = 128;
constexpr int MAX_D_INNER = 128;
constexpr int MAX_D_MODEL = 64;
constexpr int MAX_GRU_HIDDEN = 8;
constexpr int MAX_EXPERT_HIDDEN = 16;
constexpr int MAX_TOPK = 4;
constexpr int MAX_CKPT_INTERVAL = 32;   // max checkpoint interval for bilevel gradient checkpointing

constexpr int SG2M_BLOCK = 256;         // forward kernel block size
constexpr int SG2B_BLOCK = 256;         // backward kernel block size
constexpr int PSCAN_BLOCK = 512;        // threads per parallel scan block (must be power of 2)
constexpr int PSCAN_THRESHOLD = 256;    // fall back to sequential scan if N < this
constexpr int GEMM_PRECOMPUTE_THRESHOLD = 1024;  // use GEMM when N >= this

// ═══════════════════════════════════════════════════════════════════════
//  Parallel Prefix Scan Infrastructure
//
//  Affine2x2 and affine_combine moved to csrc/scan/affine2x2.h.
//  Included here so existing callers that #include "csrc/common/types.h"
//  continue to compile without modification.
// ═══════════════════════════════════════════════════════════════════════


#ifdef __CUDACC__

// ═══════════════════════════════════════════════════════════════════════
//  Branchless Stochastic Rounding (Config4 / INT8 quantized kernels)
//
//  Converts float to int8 with stochastic rounding. The ternary compiles
//  to a PTX selp instruction at -O2, avoiding warp divergence.
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ int8_t float_to_int8_stochastic_branchless(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / fmaxf(scale, 1e-12f);
    float trunc_val = truncf(scaled);
    float frac = fabsf(scaled - trunc_val);
    float threshold = (float)(rand_bits & 0xFFFF) * (1.0f / 65536.0f);
    // Branchless: ternary compiles to selp on nvcc -O2
    float round_up = (frac > threshold) ? copysignf(1.0f, scaled) : 0.0f;
    float result = trunc_val + round_up;
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, result));
}

#endif  // __CUDACC__
// ── end inlined csrc/common/types.h ──

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
