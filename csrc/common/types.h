/*
 * SuperGrok v2 — Shared Types and Constants
 *
 * Common struct definitions and compile-time constants used by both
 * forward and backward CUDA kernels.
 */

#pragma once

#include "platform.h"

// ═══════════════════════════════════════════════════════════════════════
//  Compile-time constants
// ═══════════════════════════════════════════════════════════════════════

constexpr int MAX_D_STATE = 32;
constexpr int MAX_D_INNER = 32;
constexpr int MAX_D_MODEL = 16;
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
//  The selective scan h[t] = M[t] * h[t-1] + b[t] is an affine recurrence.
//  Affine transforms compose associatively:
//    (M2, b2) ∘ (M1, b1) = (M2 * M1, M2 * b1 + b2)
//
//  This enables Blelloch parallel prefix scan over affine transforms.
//  For paired RoPE, each pair (s_even, s_odd) uses a 2×2 matrix + 2-vector.
//
//  Identity: M = I_2, b = 0
//  Combine:  (M_out, b_out) = (M_right * M_left, M_right * b_left + b_right)
// ═══════════════════════════════════════════════════════════════════════

#ifdef __CUDACC__

struct Affine2x2 {
    float m00, m01, m10, m11;  // 2×2 matrix
    float b0, b1;               // 2-vector bias
};

__device__ __forceinline__ Affine2x2 affine_identity() {
    return {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ Affine2x2 affine_combine(Affine2x2 left, Affine2x2 right) {
    // Computes right ∘ left: apply left first, then right
    // M_out = M_right * M_left
    // b_out = M_right * b_left + b_right
    Affine2x2 out;
#if GROK_CUDA
    // Inline PTX: interleave independent FMAs for ILP across pipelines.
    // Lines 1-4 are mutually independent (4 partial products).
    // Lines 5-8 accumulate into the same outputs (depend on 1-4).
    // Lines 9-10 start bias computation; 11-12 finalize bias.
    asm volatile(
        // Cycle 0: 4 independent partial products
        "fma.rn.f32 %0, %6, %12, 0f00000000;\n\t"   // m00 = r00*l00
        "fma.rn.f32 %1, %6, %13, 0f00000000;\n\t"   // m01 = r00*l01
        "fma.rn.f32 %2, %8, %12, 0f00000000;\n\t"   // m10 = r10*l00
        "fma.rn.f32 %3, %8, %13, 0f00000000;\n\t"   // m11 = r10*l01
        // Cycle 4: 4 dependent accumulations + 2 bias starts
        "fma.rn.f32 %0, %7, %14, %0;\n\t"            // m00 += r01*l10
        "fma.rn.f32 %1, %7, %15, %1;\n\t"            // m01 += r01*l11
        "fma.rn.f32 %2, %9, %14, %2;\n\t"            // m10 += r11*l10
        "fma.rn.f32 %3, %9, %15, %3;\n\t"            // m11 += r11*l11
        "fma.rn.f32 %4, %6, %16, %10;\n\t"           // b0 = r00*lb0 + rb0
        "fma.rn.f32 %5, %8, %16, %11;\n\t"           // b1 = r10*lb0 + rb1
        // Cycle 8: final bias accumulations
        "fma.rn.f32 %4, %7, %17, %4;\n\t"            // b0 += r01*lb1
        "fma.rn.f32 %5, %9, %17, %5;\n\t"            // b1 += r11*lb1
        : "=f"(out.m00), "=f"(out.m01), "=f"(out.m10), "=f"(out.m11),
          "=f"(out.b0), "=f"(out.b1)
        : "f"(right.m00), "f"(right.m01), "f"(right.m10), "f"(right.m11),
          "f"(right.b0), "f"(right.b1),
          "f"(left.m00), "f"(left.m01), "f"(left.m10), "f"(left.m11),
          "f"(left.b0), "f"(left.b1)
    );
#else
    // HIP/CPU fallback: C++ implementation (HIP has different inline asm syntax)
    out.m00 = right.m00 * left.m00 + right.m01 * left.m10;
    out.m01 = right.m00 * left.m01 + right.m01 * left.m11;
    out.m10 = right.m10 * left.m00 + right.m11 * left.m10;
    out.m11 = right.m10 * left.m01 + right.m11 * left.m11;
    out.b0  = right.m00 * left.b0  + right.m01 * left.b1 + right.b0;
    out.b1  = right.m10 * left.b0  + right.m11 * left.b1 + right.b1;
#endif
    return out;
}

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
