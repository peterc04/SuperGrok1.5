/*
 * SuperGrok v2 — Quantization Utilities
 *
 * Precision modes for projection GEMMs. Scan state accumulation
 * always stays FP32 (numerical necessity for 65K-step recurrences).
 * Expert weights stay FP32 (already in smem, bandwidth not the bottleneck).
 */

#pragma once

#include <torch/extension.h>

#ifdef WITH_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

enum class PrecisionMode {
    FP32      = 0,  // Default, all architectures
    TF32      = 1,  // sm_80+: cuBLAS TF32 mode (transparent for GEMMs)
    BF16      = 2,  // sm_80+: BF16 inputs with FP32 accumulation
    FP8_E4M3  = 3,  // sm_89+/sm_90+: FP8 inputs with FP32 accumulation
};

// Get the best supported precision for projection GEMMs
inline PrecisionMode get_best_projection_precision(int sm_arch) {
    if (sm_arch >= 90)  return PrecisionMode::FP8_E4M3;
    if (sm_arch >= 80)  return PrecisionMode::BF16;
    return PrecisionMode::FP32;
}
