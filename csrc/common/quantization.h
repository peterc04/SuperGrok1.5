/*
 * SuperGrok v2 — Quantization Utilities
 *
 * Precision modes for projection GEMMs and expert weights.
 * Scan state accumulation always stays FP32 (numerical necessity for
 * 65K-step recurrences).
 *
 * Supported formats:
 *   FP32      — Default, all architectures
 *   TF32      — sm_80+: cuBLAS TF32 mode (transparent for GEMMs)
 *   BF16      — sm_80+ / gfx90a+: BF16 inputs with FP32 accumulation
 *   FP8_E4M3  — sm_89+/sm_90+: FP8 inputs with FP32 accumulation
 *   INT8      — Expert weight-only: symmetric per-tensor quantization
 *   INT4      — Expert weight-only: GPTQ-style packed 4-bit with group scales
 *   MXFP4     — Projection weight-only: Microscaling FP4 (E2M1) with shared exponents
 */

#pragma once

#include <torch/extension.h>
#include "platform.h"

// ═══════════════════════════════════════════════════════════════════════
//  Precision mode enum
// ═══════════════════════════════════════════════════════════════════════

enum class PrecisionMode {
    FP32      = 0,  // Default, all architectures
    TF32      = 1,  // sm_80+: cuBLAS TF32 mode (transparent for GEMMs)
    BF16      = 2,  // sm_80+: BF16 inputs with FP32 accumulation
    FP8_E4M3  = 3,  // sm_89+/sm_90+: FP8 inputs with FP32 accumulation
    INT8_SYM  = 4,  // Expert weights: symmetric per-tensor INT8
    INT4_GPTQ = 5,  // Expert weights: GPTQ-style packed 4-bit
    MXFP4     = 6,  // Projections: Microscaling FP4 (E2M1)
};

// Get the best supported precision for projection GEMMs
inline PrecisionMode get_best_projection_precision(int sm_arch) {
    if (sm_arch >= 90)  return PrecisionMode::FP8_E4M3;
    if (sm_arch >= 80)  return PrecisionMode::BF16;
    return PrecisionMode::FP32;
}

// Get the best supported precision for expert weights
inline PrecisionMode get_best_expert_precision(int sm_arch) {
    // INT8 is safe on all architectures — just weight-only dequant
    // INT4 needs careful handling but works everywhere
    return PrecisionMode::INT8_SYM;
}

// ═══════════════════════════════════════════════════════════════════════
//  INT8 Symmetric Quantization — Device helpers
//
//  Symmetric: scale = max(|w|) / 127
//  Quantize:  q = round(w / scale)  →  int8 [-127, 127]
//  Dequant:   w ≈ q * scale
//
//  Used for expert weights (144 experts × expert_hidden × 2 matrices).
//  The scale is per-tensor (one float per weight matrix).
// ═══════════════════════════════════════════════════════════════════════

struct Int8QuantizedTensor {
    torch::Tensor data;   // int8 tensor, same shape as original
    torch::Tensor scale;  // float scalar (per-tensor scale)
};

inline Int8QuantizedTensor quantize_int8_symmetric(const torch::Tensor& w) {
    auto absmax = w.abs().max();
    auto scale = absmax / 127.0f;
    scale = scale.clamp_min(1e-12f);
    auto q = (w / scale).round().clamp(-127, 127).to(torch::kInt8);
    return {q, scale};
}

// Device-side INT8 dequantization (for use inside kernels)
__device__ __forceinline__ float dequant_int8(int8_t q, float scale) {
    return static_cast<float>(q) * scale;
}

// ═══════════════════════════════════════════════════════════════════════
//  INT4 GPTQ-Style Packing — Device helpers
//
//  Two INT4 values packed into one uint8: low nibble = elem[2k], high = elem[2k+1]
//  Group quantization: one scale+zero per group of G elements (G=32 typical)
//
//  Used for expert weights when extreme compression is needed.
//  Expert MLP: (1 → expert_hidden → 1) with expert_hidden=4..16.
//  Total per expert: 2*(1*H + H*1) = 4*H weights. With H=16, 64 weights → 32 bytes.
// ═══════════════════════════════════════════════════════════════════════

struct Int4PackedTensor {
    torch::Tensor data;    // uint8 tensor, shape [..., N/2] (packed pairs)
    torch::Tensor scales;  // float tensor, shape [..., num_groups]
    torch::Tensor zeros;   // float tensor, shape [..., num_groups] (asymmetric zero-points)
    int group_size;
};

inline Int4PackedTensor quantize_int4_gptq(const torch::Tensor& w, int group_size = 32) {
    auto w_flat = w.reshape({-1}).contiguous().to(torch::kFloat32);
    int64_t N = w_flat.numel();
    // Pad to even length if needed
    int64_t N_padded = (N + 1) / 2 * 2;
    if (N_padded > N) {
        w_flat = torch::nn::functional::pad(w_flat, torch::nn::functional::PadFuncOptions({0, N_padded - N}));
    }

    // Group quantization
    int64_t num_groups = (N_padded + group_size - 1) / group_size;
    auto w_grouped = w_flat.reshape({num_groups, -1});
    auto gmax = std::get<0>(w_grouped.max(1));
    auto gmin = std::get<0>(w_grouped.min(1));

    // Asymmetric: map [min, max] → [0, 15]
    auto scales = (gmax - gmin) / 15.0f;
    scales = scales.clamp_min(1e-12f);
    auto zeros = gmin;

    // Quantize each group
    auto scales_exp = scales.unsqueeze(1).expand_as(w_grouped);
    auto zeros_exp = zeros.unsqueeze(1).expand_as(w_grouped);
    auto q = ((w_grouped - zeros_exp) / scales_exp).round().clamp(0, 15).to(torch::kUInt8);
    q = q.reshape({-1});

    // Pack pairs into uint8: low nibble = even, high nibble = odd
    auto even = q.slice(0, 0, N_padded, 2);
    auto odd = q.slice(0, 1, N_padded, 2);
    auto packed = even | (odd << 4);

    return {packed, scales, zeros, group_size};
}

// Device-side INT4 unpacking and dequantization
__device__ __forceinline__ float dequant_int4(uint8_t packed, int which, float scale, float zero) {
    // which=0: low nibble, which=1: high nibble
    int q = (which == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    return static_cast<float>(q) * scale + zero;
}

// ═══════════════════════════════════════════════════════════════════════
//  MXFP4 — Microscaling FP4 (E2M1) for Projections
//
//  OCP Microscaling format: groups of 32 elements share a single 8-bit
//  shared exponent. Each element is FP4 (1 sign + 2 exponent + 1 mantissa).
//
//  Values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
//  Effective range with shared exponent: 2^(shared_exp - bias) × fp4_value
//
//  Used for projection weight matrices when extreme bandwidth savings needed.
//  Projection sizes: d_model × d_inner (8×16 = 128 weights) → 64 bytes + 4 scale bytes.
// ═══════════════════════════════════════════════════════════════════════

struct MxFp4Tensor {
    torch::Tensor data;         // uint8 tensor (packed pairs of FP4)
    torch::Tensor shared_exp;   // uint8 tensor (one shared exponent per group of 32)
    int block_size;             // elements per shared exponent group (always 32)
};

// FP4 E2M1 encoding table (unsigned magnitude)
// 0b000=0.0, 0b001=0.5, 0b010=1.0, 0b011=1.5, 0b100=2.0, 0b101=3.0, 0b110=4.0, 0b111=6.0
__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t bits) {
    // bits is 4-bit: sign(1) | exp(2) | man(1)
    int sign = (bits >> 3) & 1;
    int exp_bits = (bits >> 1) & 0x3;
    int man_bit = bits & 1;

    float val;
    if (exp_bits == 0) {
        // Subnormal: 0 or 0.5
        val = man_bit ? 0.5f : 0.0f;
    } else {
        // Normal: (1 + man_bit * 0.5) * 2^(exp_bits - 1)
        float mantissa = 1.0f + man_bit * 0.5f;
        float exp_val = (float)(1 << (exp_bits - 1));  // 2^(exp_bits-1)
        val = mantissa * exp_val;
    }
    return sign ? -val : val;
}

// Dequantize MXFP4: value = fp4_to_float(bits) * 2^(shared_exp - 127)
__device__ __forceinline__ float dequant_mxfp4(uint8_t packed, int which, uint8_t shared_exp) {
    uint8_t bits = (which == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    float base_val = fp4_e2m1_to_float(bits);
    // shared_exp uses bias=127 (same as FP32 exponent)
    float scale = exp2f((float)shared_exp - 127.0f);
    return base_val * scale;
}

inline MxFp4Tensor quantize_mxfp4(const torch::Tensor& w, int block_size = 32) {
    auto w_flat = w.reshape({-1}).contiguous().to(torch::kFloat32);
    int64_t N = w_flat.numel();
    int64_t N_padded = ((N + block_size - 1) / block_size) * block_size;
    if (N_padded > N) {
        w_flat = torch::nn::functional::pad(w_flat, torch::nn::functional::PadFuncOptions({0, N_padded - N}));
    }

    int64_t num_blocks = N_padded / block_size;
    auto w_blocked = w_flat.reshape({num_blocks, block_size});

    // Compute shared exponent per block: floor(log2(max(|block|))) + bias
    auto block_max = w_blocked.abs().max(1).values.clamp_min(1e-12f);
    auto shared_exp_f = torch::floor(torch::log2(block_max)) + 127.0f;
    shared_exp_f = shared_exp_f.clamp(0.0f, 255.0f);
    auto shared_exp = shared_exp_f.to(torch::kUInt8);

    // Quantize: scale each block, then find nearest FP4 value
    // FP4 E2M1 values (magnitude): 0, 0.5, 1, 1.5, 2, 3, 4, 6
    auto block_scale = torch::exp2(shared_exp_f - 127.0f).unsqueeze(1);
    auto w_scaled = w_blocked / block_scale;

    // Simple nearest-value quantization to 4-bit
    // Encode sign + magnitude into 4 bits
    auto signs = (w_scaled < 0).to(torch::kUInt8);
    auto magnitudes = w_scaled.abs();

    // Lookup table for FP4 magnitudes
    auto fp4_vals = torch::tensor({0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f});
    // Find nearest FP4 value for each element
    auto diffs = (magnitudes.unsqueeze(-1) - fp4_vals.unsqueeze(0).unsqueeze(0)).abs();
    auto indices = diffs.argmin(-1).to(torch::kUInt8);  // 0-7 (3 bits)

    auto encoded = (signs << 3) | indices;  // 4 bits per element
    encoded = encoded.reshape({-1});

    // Pack pairs into uint8
    int64_t packed_len = N_padded / 2;
    auto even = encoded.slice(0, 0, N_padded, 2);
    auto odd = encoded.slice(0, 1, N_padded, 2);
    auto packed = even | (odd << 4);

    return {packed, shared_exp, block_size};
}
