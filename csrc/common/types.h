/*
 * SuperGrok v2 — Shared Types and Constants
 *
 * Common struct definitions and compile-time constants used by both
 * forward and backward CUDA kernels.
 */

#pragma once

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
    out.m00 = right.m00 * left.m00 + right.m01 * left.m10;
    out.m01 = right.m00 * left.m01 + right.m01 * left.m11;
    out.m10 = right.m10 * left.m00 + right.m11 * left.m10;
    out.m11 = right.m10 * left.m01 + right.m11 * left.m11;
    out.b0  = right.m00 * left.b0  + right.m01 * left.b1 + right.b0;
    out.b1  = right.m10 * left.b0  + right.m11 * left.b1 + right.b1;
    return out;
}

#endif  // __CUDACC__
