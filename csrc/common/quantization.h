/*
 * SuperGrok v2 — Quantization Utilities
 *
 * Precision modes for projection GEMMs and expert weights.
 * Scan state accumulation always stays FP32 (numerical necessity for
 * 65K-step recurrences).
 *
 * Supported formats (3-arch active set):
 *   FP32      — Default, all architectures
 *   TF32      — sm_80+: cuBLAS TF32 mode (transparent for GEMMs)
 *   BF16      — sm_80+ / gfx90a+: BF16 inputs with FP32 accumulation
 *   FP8_E4M3  — sm_89+/sm_90+: FP8 inputs with FP32 accumulation
 *   INT8      — Expert weight-only: symmetric per-tensor quantization
 *   INT4      — Expert weight-only: GPTQ-style packed 4-bit with group scales
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
    FP8_E4M3  = 3,  // sm_90+: FP8 inputs with FP32 accumulation
    INT8_SYM  = 4,  // Expert weights: symmetric per-tensor INT8
    INT4_GPTQ = 5,  // Expert weights: GPTQ-style packed 4-bit
};

// Get the best supported precision for projection GEMMs
inline PrecisionMode get_best_projection_precision(int sm_arch) {
    if (sm_arch >= 90)  return PrecisionMode::FP8_E4M3;
    if (sm_arch >= 80)  return PrecisionMode::BF16;
    return PrecisionMode::FP32;
}

// Get the best supported precision for expert weights
inline PrecisionMode get_best_expert_precision(int /*sm_arch*/) {
    // INT8 is safe on all architectures — just weight-only dequant.
    return PrecisionMode::INT8_SYM;
}

// ═══════════════════════════════════════════════════════════════════════
//  INT8 Symmetric Quantization — Device helpers
//
//  Symmetric: scale = max(|w|) / 127
//  Quantize:  q = round(w / scale)  →  int8 [-127, 127]
//  Dequant:   w ≈ q * scale
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
// ═══════════════════════════════════════════════════════════════════════

struct Int4PackedTensor {
    torch::Tensor data;    // uint8 tensor, shape [..., N/2] (packed pairs)
    torch::Tensor scales;  // float tensor, shape [..., num_groups]
    torch::Tensor zeros;   // float tensor, shape [..., num_groups]
    int group_size;
};

inline Int4PackedTensor quantize_int4_gptq(const torch::Tensor& w, int group_size = 32) {
    auto w_flat = w.reshape({-1}).contiguous().to(torch::kFloat32);
    int64_t N = w_flat.numel();
    int64_t N_padded = (N + 1) / 2 * 2;
    if (N_padded > N) {
        w_flat = torch::nn::functional::pad(w_flat, torch::nn::functional::PadFuncOptions({0, N_padded - N}));
    }

    int64_t num_groups = (N_padded + group_size - 1) / group_size;
    auto w_grouped = w_flat.reshape({num_groups, -1});
    auto gmax = std::get<0>(w_grouped.max(1));
    auto gmin = std::get<0>(w_grouped.min(1));

    auto scales = (gmax - gmin) / 15.0f;
    scales = scales.clamp_min(1e-12f);
    auto zeros = gmin;

    auto scales_exp = scales.unsqueeze(1).expand_as(w_grouped);
    auto zeros_exp = zeros.unsqueeze(1).expand_as(w_grouped);
    auto q = ((w_grouped - zeros_exp) / scales_exp).round().clamp(0, 15).to(torch::kUInt8);
    q = q.reshape({-1});

    auto even = q.slice(0, 0, N_padded, 2);
    auto odd = q.slice(0, 1, N_padded, 2);
    auto packed = even | (odd << 4);

    return {packed, scales, zeros, group_size};
}

// Device-side INT4 unpacking and dequantization
__device__ __forceinline__ float dequant_int4(uint8_t packed, int which, float scale, float zero) {
    int q = (which == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    return static_cast<float>(q) * scale + zero;
}
