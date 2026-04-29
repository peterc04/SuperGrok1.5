/*
 * SuperGrok v2 — Shared FP4 / FP6 / Stochastic-Rounding Helpers
 *
 * Extracted verbatim from the former
 *   csrc/kernels/hip/gfx950/cdna4_kernels_gfx950.hip.cpp
 * monolith so that the per-feature TUs (fp4_expert / fp6_state /
 * sparse24 / fused_combos) can all share these helpers without
 * multiple-definition linker errors.
 *
 * Notes on linkage:
 *   - All helpers are marked `__device__ static __forceinline__` so each
 *     TU gets its own internal-linkage copy. This avoids ODR / multiple-
 *     definition errors when this header is included from multiple
 *     translation units.
 *   - The dequant LUT lives in an anonymous namespace; each TU gets its
 *     own __constant__ copy (64 bytes). Acceptable cost for clean linkage.
 *   - Math is unchanged from the original monolith.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace sg { namespace gfx950 {
namespace {

// FP4 dequantization lookup table (16 entries for 4-bit codes)
// Format: E2M1 — sign(1) exp(2) mantissa(1)
// Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (positive and negative)
__constant__ float kFP4DequantTable[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ═══════════════════════════════════════════════════════════════════════
//  FP4 helpers (E2M1: 1 sign + 2 exp + 1 mantissa, 8 packed per uint32)
// ═══════════════════════════════════════════════════════════════════════

__device__ static __forceinline__ float fp4_dequant(uint32_t packed, int idx) {
    // Extract 4-bit code at position idx (0-7) from packed uint32_t
    uint32_t code = (packed >> (idx * 4)) & 0xF;
    return kFP4DequantTable[code];
}

__device__ static __forceinline__ uint32_t fp4_quant_one(float val) {
    // Quantize single float to 4-bit FP4 code
    uint32_t sign = (val < 0.0f) ? 8u : 0u;
    float aval = fabsf(val);

    // Nearest-neighbor quantization to E2M1 representable values
    uint32_t code;
    if (aval < 0.25f)       code = 0;  // 0.0
    else if (aval < 0.75f)  code = 1;  // 0.5
    else if (aval < 1.25f)  code = 2;  // 1.0
    else if (aval < 1.75f)  code = 3;  // 1.5
    else if (aval < 2.5f)   code = 4;  // 2.0
    else if (aval < 3.5f)   code = 5;  // 3.0
    else if (aval < 5.0f)   code = 6;  // 4.0
    else                    code = 7;  // 6.0

    return sign | code;
}

__device__ static __forceinline__ uint32_t fp4_pack8(const float* vals, float scale) {
    // Pack 8 scaled float values into a single uint32_t of FP4
    uint32_t packed = 0;
    for (int i = 0; i < 8; i++) {
        uint32_t code = fp4_quant_one(vals[i] * scale);
        packed |= (code << (i * 4));
    }
    return packed;
}

// ═══════════════════════════════════════════════════════════════════════
//  FP6 (E3M2) helpers (1 sign + 3 exp + 2 mantissa, 4 packed per 3 bytes)
// ═══════════════════════════════════════════════════════════════════════

__device__ static __forceinline__ float fp6_to_fp32(uint32_t bits6) {
    // E3M2 decode: sign(1) exp(3) mantissa(2)
    uint32_t sign_bit = (bits6 >> 5) & 1;
    uint32_t exp_bits = (bits6 >> 2) & 0x7;
    uint32_t man_bits = bits6 & 0x3;

    float sign = sign_bit ? -1.0f : 1.0f;

    if (exp_bits == 0) {
        // Subnormal: value = sign * mantissa * 2^(1-bias) * 2^(-2)
        // With bias=3: 2^(-2) * 2^(-2) = 2^(-4)
        float mantissa = (float)man_bits * 0.0625f;  // man * 2^(-4)
        return sign * mantissa;
    }
    if (exp_bits == 7) {
        // Inf/NaN: treat as max finite value for optimizer stability
        float mantissa = 1.0f + (float)man_bits * 0.25f;
        return sign * mantissa * 16.0f;  // 2^(7-3) = 16
    }

    // Normal: value = sign * (1 + mantissa*2^(-2)) * 2^(exp - bias)
    float mantissa = 1.0f + (float)man_bits * 0.25f;
    int exponent = (int)exp_bits - 3;  // bias = 3
    return sign * ldexpf(mantissa, exponent);
}

__device__ static __forceinline__ uint32_t fp32_to_fp6(float val) {
    // Quantize FP32 to 6-bit E3M2
    uint32_t sign_bit = (val < 0.0f) ? 1u : 0u;
    float aval = fabsf(val);

    if (aval < 0.0625f) {
        // Subnormal region or zero
        // Subnormal: man_bits = round(aval / 0.0625)
        uint32_t man_bits = (uint32_t)(aval * 16.0f + 0.5f);
        if (man_bits > 3) man_bits = 3;
        return (sign_bit << 5) | man_bits;
    }

    // Find exponent: aval = (1 + frac) * 2^exp
    int exp_raw;
    float frac = frexpf(aval, &exp_raw);
    // frexp returns [0.5, 1.0), we need [1.0, 2.0)
    frac *= 2.0f;
    exp_raw -= 1;

    // Biased exponent (bias = 3)
    int exp_biased = exp_raw + 3;

    if (exp_biased <= 0) {
        // Underflow to subnormal
        float scaled = aval * 16.0f;  // shift to subnormal range
        uint32_t man_bits = (uint32_t)(scaled + 0.5f);
        if (man_bits > 3) man_bits = 3;
        return (sign_bit << 5) | man_bits;
    }
    if (exp_biased >= 7) {
        // Overflow: clamp to max representable (exp=6, man=3)
        return (sign_bit << 5) | (6u << 2) | 3u;
    }

    // Normal encoding
    // frac is in [1.0, 2.0), mantissa bits = round((frac - 1.0) * 4)
    uint32_t man_bits = (uint32_t)((frac - 1.0f) * 4.0f + 0.5f);
    if (man_bits > 3) {
        man_bits = 0;
        exp_biased++;
        if (exp_biased >= 7) {
            return (sign_bit << 5) | (6u << 2) | 3u;
        }
    }

    return (sign_bit << 5) | ((uint32_t)exp_biased << 2) | man_bits;
}

__device__ static __forceinline__ void fp6_pack4(const float* vals, uint8_t* out3) {
    // Pack 4 FP6 values into 3 bytes (24 bits total)
    uint32_t v0 = fp32_to_fp6(vals[0]);
    uint32_t v1 = fp32_to_fp6(vals[1]);
    uint32_t v2 = fp32_to_fp6(vals[2]);
    uint32_t v3 = fp32_to_fp6(vals[3]);

    uint32_t packed = v0 | (v1 << 6) | (v2 << 12) | (v3 << 18);
    out3[0] = (uint8_t)(packed & 0xFF);
    out3[1] = (uint8_t)((packed >> 8) & 0xFF);
    out3[2] = (uint8_t)((packed >> 16) & 0xFF);
}

__device__ static __forceinline__ void fp6_unpack4(const uint8_t* in3, float* vals) {
    // Unpack 3 bytes into 4 FP6→FP32 values
    uint32_t packed = (uint32_t)in3[0] | ((uint32_t)in3[1] << 8) | ((uint32_t)in3[2] << 16);
    vals[0] = fp6_to_fp32(packed & 0x3F);
    vals[1] = fp6_to_fp32((packed >> 6) & 0x3F);
    vals[2] = fp6_to_fp32((packed >> 12) & 0x3F);
    vals[3] = fp6_to_fp32((packed >> 18) & 0x3F);
}

// ═══════════════════════════════════════════════════════════════════════
//  Stochastic rounding helper for FP4 gradient quantization
// ═══════════════════════════════════════════════════════════════════════

__device__ static __forceinline__ uint32_t philox_hash(uint32_t counter, uint32_t key) {
    // Simplified Philox-based random number for stochastic rounding
    uint32_t state = counter * 0xD2511F53u + key;
    state ^= state >> 16;
    state *= 0x85EBCA6Bu;
    state ^= state >> 13;
    state *= 0xC2B2AE35u;
    state ^= state >> 16;
    return state;
}

} // anonymous namespace
} } // namespace sg::gfx950
