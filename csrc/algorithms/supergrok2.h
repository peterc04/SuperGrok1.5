#pragma once
// SuperGrok v2 — vendor-neutral algorithm header.
// Migrated and consolidated from:
//   - csrc/device/optimizers/sm_90/supergrok2_sm90.cuh (REAL device templates)
//   - csrc/kernels/cuda/sm_90/supergrok2_fwd.cuh
//   - csrc/kernels/cuda/sm_90/supergrok2_bwd.cuh
//   - csrc/kernels/cuda/sm_90/supergrok2_warp_specialized.cuh
//
// Mamba-3 + 4-Head PEER + per-element GRU + Adam pipeline.
//
// Per-step pipeline:
//   (1) input_proj_sort     : [grad, sharpness] -> [N, d_model], sort keys = |grad|
//   (2) mamba3_scan         : selective scan with trapezoidal discretization + RoPE
//                              (sequential for small N, parallel Blelloch for larger N,
//                               warp-specialized on Hopper for uniform d_state)
//   (3) peer_route          : product-key expert routing, top-4 of 144 experts
//   (4) gru_step            : per-element GRU integrates expert output with temporal state
//   (5) apply               : smart_grad + Adam + trust-ratio + decoupled weight decay
//
// Backward (used by bilevel meta-learning):
//   (6) bilevel_precompute  : reproduce forward projections needed for adjoint scan
//
// The heavy math (GEMMs, parallel scans, cluster reductions) is in the
// per-backend primitives; this header contains the per-element building
// blocks that are vendor-neutral.

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
// ── inlined from former csrc/common/utils.cuh ──
/*
 * SuperGrok v2 — Shared Device Helpers
 *
 * Device utility functions used by multiple kernel files.
 * Uses platform.h macros for CUDA/HIP portability.
 */


#if GROK_CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Warp-level reduction helper
//
//  Sum a float across d_inner threads (all in one warp, d_inner ≤ WARP_SIZE).
//  Uses platform-abstracted shuffle; works for any d_inner ≤ WARP_SIZE
//  (including non-power-of-2).
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float warp_reduce_sum(float val, int d_inner, int tid) {
    unsigned mask = (d_inner < WARP_SIZE) ? ((1u << d_inner) - 1) : FULL_WARP_MASK;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = SHFL_DOWN_SYNC(mask, val, offset);
        if (tid + offset < d_inner)
            val += other;
    }
    return val;  // only lane 0 has the correct sum
}

// ═══════════════════════════════════════════════════════════════════════
//  Stochastic Rounding for Quantized Optimizer States (Config 3)
//
//  Hash-based PRNG: deterministic per (step, element) pair, no state needed.
//  Faster than cuRAND, no separate state tensor required.
// ═══════════════════════════════════════════════════════════════════════

// Hash-based PRNG (Philox-like): deterministic, no state
__device__ __forceinline__ unsigned hash_prng(unsigned step, unsigned idx) {
    unsigned h = (step * 2654435761u) ^ (idx * 2246822519u);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

#if GROK_CUDA || GROK_HIP

// BF16 stochastic rounding: unbiased quantization
__device__ __forceinline__ __nv_bfloat16 float_to_bf16_stochastic(float val, unsigned rand_bits) {
    unsigned bits = __float_as_uint(val);
    unsigned truncated = bits & 0xFFFF;     // bits that BF16 drops
    unsigned threshold = rand_bits & 0xFFFF; // random 16-bit threshold
    if (truncated > threshold) {
        bits += 0x10000;  // round up
    }
    bits &= 0xFFFF0000;  // truncate to BF16
    return __float2bfloat16(__uint_as_float(bits));
}

// INT8 per-block quantization with stochastic rounding
// block_size elements share one FP32 scale factor
__device__ __forceinline__ int8_t float_to_int8_stochastic(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / scale;
    float truncated = truncf(scaled);
    float frac = fabsf(scaled - truncated);
    float threshold = (float)(rand_bits & 0xFFFF) / 65536.0f;
    if (frac > threshold) {
        truncated += (scaled > 0) ? 1.0f : -1.0f;
    }
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, truncated));
}

// ═══════════════════════════════════════════════════════════════════════
//  Phase 3: Inline PTX for Hot Inner Loops
//
//  Hand-tuned PTX for critical paths in the SG2 fused_elem pipeline.
//  These replace compiler-generated code in the highest-frequency loops.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_CUDA

// Fast reciprocal sqrt via PTX rsqrt.approx.f32 + Newton-Raphson refinement.
// 2-3x faster than sqrtf(x) + fdividef for Adam denominator.
__device__ __forceinline__ float fast_rsqrt_nr(float x) {
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    // One Newton-Raphson iteration: r = r * (1.5 - 0.5 * x * r * r)
    r = r * (1.5f - 0.5f * x * r * r);
    return r;
}

// Fused multiply-add via PTX fma.rn.f32 — ensures single FMA instruction.
// Critical for affine_combine inner loop (8 FMAs per composition).
__device__ __forceinline__ float ptx_fma(float a, float b, float c) {
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

// Fast exp2 approximation via PTX ex2.approx.f32.
// Used in Mamba scan: exp(A * dt) = exp2(A * dt / ln2).
__device__ __forceinline__ float ptx_exp2(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// Fast log2 via PTX lg2.approx.f32.
__device__ __forceinline__ float ptx_log2(float x) {
    float r;
    asm("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// Fast exp via exp2: exp(x) = exp2(x * log2(e))
__device__ __forceinline__ float ptx_expf(float x) {
    return ptx_exp2(x * 1.4426950408889634f);  // log2(e)
}

// Fast tanh approximation via exp2: tanh(x) = (e^2x - 1) / (e^2x + 1)
// Used in GRU h_tilde computation.
__device__ __forceinline__ float ptx_tanhf(float x) {
    float e2x = ptx_exp2(2.0f * x * 1.4426950408889634f);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

// Fast sigmoid via exp2: sigmoid(x) = 1 / (1 + exp(-x))
// Used in GRU z_gate and r_gate.
__device__ __forceinline__ float ptx_sigmoidf(float x) {
    float en = ptx_exp2(-x * 1.4426950408889634f);
    return 1.0f / (1.0f + en);
}

// Blelloch affine_combine using pure PTX FMA instructions.
// Composes two Affine2x2 transforms: result = left ∘ right
// M_out = M_left * M_right, b_out = M_left * b_right + b_left
// This is the inner loop of the parallel prefix scan (called O(log N) times).
__device__ __forceinline__ Affine2x2 ptx_affine_combine(
    const Affine2x2& left, const Affine2x2& right
) {
    Affine2x2 out;
    // M_out = M_left * M_right (2x2 matrix multiply via 8 FMAs)
    out.m00 = ptx_fma(left.m00, right.m00, left.m01 * right.m10);
    out.m01 = ptx_fma(left.m00, right.m01, left.m01 * right.m11);
    out.m10 = ptx_fma(left.m10, right.m00, left.m11 * right.m10);
    out.m11 = ptx_fma(left.m10, right.m01, left.m11 * right.m11);
    // b_out = M_left * b_right + b_left
    out.b0 = ptx_fma(left.m00, right.b0, ptx_fma(left.m01, right.b1, left.b0));
    out.b1 = ptx_fma(left.m10, right.b0, ptx_fma(left.m11, right.b1, left.b1));
    return out;
}

// Expert MLP forward pass — single expert, ReLU activation.
// Inlined PTX FMA for the inner products.
// expert_hidden is typically 8-16, so fully unrollable at compile time.
template <int EXPERT_HIDDEN>
__device__ __forceinline__ float ptx_expert_mlp_forward(
    const float* __restrict__ W1,   // [expert_hidden]
    const float* __restrict__ b1,   // [expert_hidden]
    const float* __restrict__ W2,   // [expert_hidden]
    float b2,
    float input
) {
    float result = b2;
    #pragma unroll
    for (int h = 0; h < EXPERT_HIDDEN; h++) {
        float hidden = ptx_fma(W1[h], input, b1[h]);
        hidden = fmaxf(hidden, 0.0f);  // ReLU
        result = ptx_fma(W2[h], hidden, result);
    }
    return result;
}

// Stochastic rounding with PTX prmt (permute bytes) for fast bit extraction.
// Replaces the hash_prng shift+multiply chain with a single PTX instruction
// for extracting the random threshold from the hash output.
__device__ __forceinline__ int8_t ptx_int8_stochastic_round(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / fmaxf(scale, 1e-12f);
    float tr = truncf(scaled);
    float frac = fabsf(scaled - tr);
    // Extract lower 16 bits as threshold using prmt
    unsigned lo16;
    asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(lo16) : "r"(rand_bits));
    float threshold = (float)lo16 / 65536.0f;
    if (frac > threshold) tr += (scaled > 0) ? 1.0f : -1.0f;
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, tr));
}

// §25.7 DSMEM cluster reduce (sm_90+ Hopper distributed shared memory).
// Block-local warp reduce first, then cluster-wide reduce via cooperative
// groups. Falls back to warp reduce on pre-Hopper.
__device__ __forceinline__ float cluster_dsmem_reduce_sum(float val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    namespace cg = cooperative_groups;
    val = warp_reduce_sum(val, WARP_SIZE, threadIdx.x & (WARP_SIZE - 1));
    auto cluster = cg::this_cluster();
    val = cg::reduce(cluster, val, cg::plus<float>());
    return val;
#else
    return warp_reduce_sum(val, WARP_SIZE, threadIdx.x & (WARP_SIZE - 1));
#endif
}

#endif // GROK_CUDA

// HIP fallbacks — use standard math functions
#if GROK_HIP
__device__ __forceinline__ float fast_rsqrt_nr(float x) { return rsqrtf(x); }
__device__ __forceinline__ float ptx_fma(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ float ptx_expf(float x) { return expf(x); }
__device__ __forceinline__ float ptx_tanhf(float x) { return tanhf(x); }
__device__ __forceinline__ float ptx_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ Affine2x2 ptx_affine_combine(
    const Affine2x2& left, const Affine2x2& right
) {
    return affine_combine(left, right);  // Use types.h version
}
#endif // GROK_HIP

#endif // GROK_CUDA || GROK_HIP
// ── end inlined csrc/common/utils.cuh ──
#include "csrc/algorithms/adamw.h"  // pulled in for the merged moe_adam_step

namespace sg { namespace algorithms {

// Compile-time maximums (must match types.h constants).
constexpr int SG2_MAX_D_STATE = 128;
constexpr int SG2_MAX_D_INNER = 128;
constexpr int SG2_MAX_D_MODEL = 64;

// =========================================================================
//  Forward (1): Input Projection + Sort Key
//  x_out[idx, d] = proj_W[d,0]*grad + proj_W[d,1]*sharp + proj_b[d]
//  sort_key      = |grad|
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void sg2_input_proj_sort(
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharpness,
    float* __restrict__ x_out,
    float* __restrict__ sort_keys,
    int* __restrict__ sort_indices,
    const float* __restrict__ proj_W,
    const float* __restrict__ proj_b,
    const int idx,
    const int N,
    const int d_model
) {
    if (idx >= N) return;

    float g = static_cast<float>(grad[idx]);
    float s = static_cast<float>(sharpness[idx]);
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(s)) s = 0.0f;

    #pragma unroll 4
    for (int d = 0; d < d_model; d++) {
        x_out[idx * d_model + d] = proj_W[d * 2] * g + proj_W[d * 2 + 1] * s + proj_b[d];
    }
    sort_keys[idx]    = fabsf(g);
    sort_indices[idx] = idx;
}

// =========================================================================
//  Forward (2): Sequential Mamba-3 Scan (per-thread, single timestep)
//  Trapezoidal discretization + RoPE on state pairs.
// =========================================================================

__device__ __forceinline__ void sg2_mamba3_scan_step(
    float* __restrict__ h,           // [d_state] state (registers / smem)
    const float* __restrict__ A,     // [d_state] preloaded A coefficients
    const float* __restrict__ freq,  // [d_state/2] RoPE frequencies
    const float x_val,
    const float dt_val,
    const float* __restrict__ B_vals,
    const float* __restrict__ C_vals,
    const float D_val,
    const float z_val,
    const int d_state,
    const int step_t,
    float* __restrict__ y_out
) {
    const int half_d_state = d_state / 2;
    float y_acc = 0.0f;

    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++) {
        int s0 = p * 2;
        int s1 = s0 + 1;

        // Trapezoidal discretization
        float dt_A0 = dt_val * A[s0];
        float dt_A1 = dt_val * A[s1];
        float dA0 = (1.0f + dt_A0 * 0.5f) / (1.0f - dt_A0 * 0.5f);
        float dA1 = (1.0f + dt_A1 * 0.5f) / (1.0f - dt_A1 * 0.5f);
        float dBx0 = B_vals[s0] * x_val * dt_val;
        float dBx1 = B_vals[s1] * x_val * dt_val;

        h[s0] = dA0 * h[s0] + dBx0;
        h[s1] = dA1 * h[s1] + dBx1;

        // RoPE rotation on state pair
        float cos_r = cosf(freq[p] * step_t);
        float sin_r = sinf(freq[p] * step_t);
        float h0_rot = h[s0] * cos_r - h[s1] * sin_r;
        float h1_rot = h[s0] * sin_r + h[s1] * cos_r;

        y_acc += C_vals[s0] * h0_rot + C_vals[s1] * h1_rot;
    }

    // y + D*x gated by silu(z)
    y_acc += D_val * x_val;
    float silu_z = z_val / (1.0f + expf(-z_val));
    *y_out = y_acc * silu_z;
}

// =========================================================================
//  Forward (2'): Warp-Specialized Scan (Hopper consumer per-timestep)
//  Producer warp loads data into double-buffered smem; this is the
//  consumer-side recurrence for a single (di, state-pair) tuple.
// =========================================================================

__device__ __forceinline__ void sg2_scan_consumer_step(
    float* __restrict__ h,           // [2] state pair in registers
    const float A0,
    const float A1,
    const float D_val,
    const float rope_f,
    const float x_val,
    const float z_val,
    const float dt_val,
    const float B0_val,
    const float B1_val,
    const float C0_val,
    const float C1_val,
    const int t,
    float* __restrict__ y_out
) {
    float dA0 = expf(A0 * dt_val);
    float dA1 = expf(A1 * dt_val);
    float dBx0 = B0_val * x_val * dt_val;
    float dBx1 = B1_val * x_val * dt_val;

    h[0] = dA0 * h[0] + dBx0;
    h[1] = dA1 * h[1] + dBx1;

    float h0_rot = h[0] * cosf(rope_f * t) - h[1] * sinf(rope_f * t);
    float h1_rot = h[0] * sinf(rope_f * t) + h[1] * cosf(rope_f * t);

    float y = C0_val * h0_rot + C1_val * h1_rot + D_val * x_val;
    float silu_z = z_val / (1.0f + expf(-z_val));
    *y_out = y * silu_z;
}

// =========================================================================
//  Forward (2''): Warp-Specialized Scan, d_state=16 unrolled
//  All 8 state pairs processed in one consumer thread.
// =========================================================================

constexpr int SG2_D_STATE_16 = 16;
constexpr int SG2_D_STATE_16_PAIRS = 8;

__device__ __forceinline__ void sg2_scan_consumer_step_d16(
    float h[SG2_D_STATE_16],
    const float A_vals[SG2_D_STATE_16],
    const float rope_f[SG2_D_STATE_16_PAIRS],
    const float D_val,
    const float x_val,
    const float z_val,
    const float dt_val,
    const float B[SG2_D_STATE_16],
    const float C[SG2_D_STATE_16],
    const int t,
    float* __restrict__ y_out
) {
    float y_acc = 0.0f;
    #pragma unroll
    for (int p = 0; p < SG2_D_STATE_16_PAIRS; p++) {
        int s0 = p * 2;
        int s1 = s0 + 1;

        float dA0 = expf(A_vals[s0] * dt_val);
        float dA1 = expf(A_vals[s1] * dt_val);
        float dBx0 = B[s0] * x_val * dt_val;
        float dBx1 = B[s1] * x_val * dt_val;

        h[s0] = dA0 * h[s0] + dBx0;
        h[s1] = dA1 * h[s1] + dBx1;

        float cos_r = cosf(rope_f[p] * t);
        float sin_r = sinf(rope_f[p] * t);
        float h0_rot = h[s0] * cos_r - h[s1] * sin_r;
        float h1_rot = h[s0] * sin_r + h[s1] * cos_r;

        y_acc += C[s0] * h0_rot + C[s1] * h1_rot;
    }

    y_acc += D_val * x_val;
    float silu_z = z_val / (1.0f + expf(-z_val));
    *y_out = y_acc * silu_z;
}

// =========================================================================
//  Forward (3-5): GRU + apply tail (per-element)
//  Combines temporal memory, smart_grad, and Adam update.
//  PEER routing handles its own selection on the host side; this is
//  the per-element body that consumes the routed expert output.
// =========================================================================

template <typename ParamT, typename GradT>
__device__ __forceinline__ void sg2_apply_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ mu_state,    // GRU hidden state
    const GradT* __restrict__ grad,
    const float expert_out,          // PEER expert output for this element
    const float alpha,
    const float gru_decay,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int idx
) {
    const float g = static_cast<float>(grad[idx]);
    const float p = static_cast<float>(param[idx]);

    // GRU step: simple gated update of mu_state with expert_out as candidate.
    const float mu_new = gru_decay * mu_state[idx] + (1.0f - gru_decay) * expert_out;
    mu_state[idx] = mu_new;

    const float smart_grad = g + alpha * mu_new;

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * smart_grad;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * smart_grad * smart_grad;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    // bc1, bc2 un-inverted (= 1 - beta^t): divide for bias correction.
    // Matches the convention used by adamw.h / grokadamw.h / etc. and the
    // Python `_single_param_step` in grokking_optimizers/optimizers/supergrok2.py.
    const float update = (m / bc1) / (sqrtf(v / bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

// =========================================================================
//  Backward (6): Bilevel precompute per-timestep
//  Reproduces the forward projections needed for the adjoint scan.
// =========================================================================

__device__ __forceinline__ void sg2_bilevel_precompute_timestep(
    const float* __restrict__ x_sorted_t,
    const float* __restrict__ in_proj_W,
    const float* __restrict__ dt_proj_W,
    const float* __restrict__ dt_proj_b,
    const float* __restrict__ B_proj_W,
    const float* __restrict__ C_proj_W,
    float* __restrict__ pre_x_val_t,
    float* __restrict__ pre_z_val_t,
    float* __restrict__ pre_dt_val_t,
    float* __restrict__ pre_B_val_t,
    float* __restrict__ pre_C_val_t,
    const int d_model,
    const int d_inner,
    const int d_state
) {
    float x_branch[SG2_MAX_D_INNER];

    // Input projection: x_branch and z
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++) {
        float x_val = 0.0f, z_val = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) {
            x_val += in_proj_W[j * d_model + d] * x_sorted_t[d];
            z_val += in_proj_W[(j + d_inner) * d_model + d] * x_sorted_t[d];
        }
        x_branch[j]      = x_val;
        pre_x_val_t[j]   = x_val;
        pre_z_val_t[j]   = z_val;
    }

    // dt projection + softplus
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++) {
        float dt_raw = dt_proj_b[j];
        #pragma unroll 4
        for (int k = 0; k < d_inner; k++) {
            dt_raw += dt_proj_W[j * d_inner + k] * x_branch[k];
        }
        pre_dt_val_t[j] = logf(1.0f + expf(dt_raw));
    }

    // B and C projections
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) {
        float b_val = 0.0f, c_val = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < d_inner; j++) {
            b_val += B_proj_W[s * d_inner + j] * x_branch[j];
            c_val += C_proj_W[s * d_inner + j] * x_branch[j];
        }
        pre_B_val_t[s] = b_val;
        pre_C_val_t[s] = c_val;
    }
}

// ═════════════════════════════════════════════════════════════════════════
//  MoE/Adam multi-tensor — folded in from former csrc/algorithms/moe_adam.h.
//
//  Multi-tensor batched AdamW used for both standard parameter groups and
//  Mixture-of-Experts active-set updates. The MoE variant compacts the
//  active subset of expert parameters into a dense buffer, runs the same
//  per-element Adam step over that buffer, then scatters results back.
//
//  The per-element math is identical to adamw.h::adamw_step; this function
//  re-exports it under the `moe_adam_step` name to keep the launcher glue
//  symmetric across the optimizers.
// ═════════════════════════════════════════════════════════════════════════

template <typename ParamT, typename GradT>
__device__ __forceinline__ void moe_adam_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const GradT* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int idx
) {
    adamw_step(param, exp_avg, exp_avg_sq, grad,
               lr, beta1, beta2, eps, wd, bc1, bc2, idx);
}

}} // namespace sg::algorithms
