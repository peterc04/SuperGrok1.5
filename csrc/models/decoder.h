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
