#pragma once
// Canonical header (de-inlined). Body is byte-identical to the
// formerly copy-pasted block; prerequisites are included so that
// platform macros precede their use.
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/scan/affine2x2.h"

/*
 * SuperGrok v2 — Shared Device Helpers
 *
 * Device utility functions used by multiple kernel files.
 * Uses platform.h macros for CUDA/HIP portability.
 */

// Defensive bias-correction denominator guard. bc = 1 - beta^t; at t==0 bc==0
// which divides by zero. The host contract is step>=1 so this never fires in
// normal operation; the guard is free (one fmaxf) and prevents silent NaN
// poisoning if a caller ever violates the contract.
__device__ __forceinline__ float sg_safe_bc(float bc) {
    return fmaxf(bc, 1e-30f);
}


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

// §Phase3-S0: PTX helpers (fast_rsqrt_nr/ptx_fma/ptx_exp2/ptx_expf/ptx_tanhf/ptx_sigmoidf/ptx_int8_stochastic_round) inlined into their owning component headers; drained here.

#if GROK_CUDA

// §25.7 / §3.2 DSMEM cluster reduce — SAFE FALLBACK shim.
//
// The REAL Hopper thread-block-cluster DSMEM cross-CTA reduction now lives in
// csrc/backends/cuda/sm_90/primitives.cuh as
//   sg::sm90::primitives::cluster_reduce_sum_f32_dsmem(val, cluster_smem_slot)
// which does the full thread->warp->block->cluster tree via map_shared_rank +
// cl.sync(). That helper needs (a) a cluster launch and (b) a per-block shared
// scratch slot, so new cluster-aware call sites route to it directly.
//
// This utils.cuh entry is intentionally LEFT as the arch-portable warp-reduce
// fallback: it has no shared-scratch argument and no cluster handle, so it is
// the right default for the (many) non-cluster call sites and for pre-Hopper
// builds. We do NOT change its signature (other sites depend on it). It is the
// "DSMEM off" behavior — equivalent to ENABLE_DSMEM_REDUCE==0.
__device__ __forceinline__ float cluster_dsmem_reduce_sum(float val) {
    // No cluster handle / shared slot here by design; the supported, arch-
    // portable behavior is the warp-level reduction. Cluster-aware sites that
    // want the real DSMEM tree call primitives::cluster_reduce_sum_f32_dsmem.
    return warp_reduce_sum(val, WARP_SIZE, threadIdx.x & (WARP_SIZE - 1));
}

#endif // GROK_CUDA

#endif // GROK_CUDA || GROK_HIP
