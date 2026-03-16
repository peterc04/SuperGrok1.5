/*
 * SuperGrok v2 — Shared Device Helpers
 *
 * Device utility functions used by multiple kernel files.
 * Uses platform.h macros for CUDA/HIP portability.
 */

#pragma once
#include "platform.h"
#include "types.h"

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
