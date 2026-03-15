/*
 * SuperGrok v2 — Shared CUDA Device Helpers
 *
 * Device utility functions used by multiple kernel files.
 */

#pragma once

// ═══════════════════════════════════════════════════════════════════════
//  Warp-level reduction helper
//
//  Sum a float across d_inner threads (all in one warp, d_inner ≤ 32).
//  Uses __shfl_down_sync; works for any d_inner ≤ 32 (including non-power-of-2).
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float warp_reduce_sum(float val, int d_inner, int tid) {
    unsigned mask = (d_inner < 32) ? ((1u << d_inner) - 1) : 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, val, offset);
        if (tid + offset < d_inner)
            val += other;
    }
    return val;  // only lane 0 has the correct sum
}
