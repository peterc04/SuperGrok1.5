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
