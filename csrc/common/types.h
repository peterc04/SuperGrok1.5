#pragma once
// Canonical header (de-inlined). Body is byte-identical to the
// formerly copy-pasted block; prerequisites are included so that
// platform macros precede their use.
#include "csrc/common/platform.h"
#include "csrc/scan/affine2x2.h"

/*
 * SuperGrok v2 — Shared Types and Constants
 *
 * Common struct definitions and compile-time constants used by both
 * forward and backward CUDA kernels.
 */



#include <vector>
#include <torch/extension.h>

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
// SG_TUNED_PSCAN_BLOCK — threads per parallel-scan block (MUST stay a power of
// two). Default 512 (= prior literal → byte-identical untuned build). NEEDS-
// PARITY before a non-default winner ships: the Blelloch up/down-sweep indexing
// and the shared scratch sizing assume a power-of-two block, and callers clamp
// it via min(PSCAN_BLOCK, 256) — the autotuner cannot prove scan correctness,
// only the mamba/SG2 parity gate can. (Macro guard added so the constant is
// -D-overridable; not yet wired into the autotuner search space — see report:
// scan dims are deferred on the per-arch cardinality budget.)
#ifndef SG_TUNED_PSCAN_BLOCK
#define SG_TUNED_PSCAN_BLOCK 512
#endif
constexpr int PSCAN_BLOCK = SG_TUNED_PSCAN_BLOCK;  // power of 2
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
