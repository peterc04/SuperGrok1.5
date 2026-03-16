/*
 * SuperGrok v2 — Runtime Architecture Dispatch
 *
 * Detects GPU compute capability at runtime and provides dispatch
 * helpers for selecting the optimal kernel tier.
 *
 * Three tiers (NVIDIA):
 *   GENERIC  — sm_70, sm_75 (V100, T4): FP32, basic smem
 *   AMPERE   — sm_80, sm_86, sm_89 (A100, RTX 3090, L4, RTX 4090):
 *              TF32 Tensor Cores, cp.async, 192KB smem, BF16
 *   HOPPER   — sm_90, sm_100 (H100, B200):
 *              TMA, Thread Block Clusters, FP8 Tensor Cores, 228KB smem
 *
 * AMD (ROCm/HIP):
 *   GENERIC tier — gfx908 (MI100), gfx90a (MI200), gfx942 (MI300X)
 *   CDNA wavefront-64 with matrix cores
 */

#pragma once

#include "platform.h"
#include <cstdlib>

// ═══════════════════════════════════════════════════════════════════════
//  GPU Vendor
// ═══════════════════════════════════════════════════════════════════════

enum class GpuVendor {
    NVIDIA,
    AMD,
    NONE,
};

inline GpuVendor get_gpu_vendor() {
#if GROK_HIP
    return GpuVendor::AMD;
#elif GROK_CUDA
    return GpuVendor::NVIDIA;
#else
    return GpuVendor::NONE;
#endif
}

// ═══════════════════════════════════════════════════════════════════════
//  Architecture tier
// ═══════════════════════════════════════════════════════════════════════

enum class ArchTier {
    GENERIC,    // sm_70, sm_75 (NVIDIA) or gfx908/gfx90a/gfx942 (AMD)
    AMPERE,     // sm_80, sm_86, sm_89
    HOPPER,     // sm_90, sm_100
    BLACKWELL,  // sm_100+
};

inline int get_sm_arch() {
    static int cached = -1;
    if (cached >= 0) return cached;

    // FORCE_ARCH env var overrides hardware detection for testing.
    // Set FORCE_ARCH=75 to test generic tier, =80 for Ampere, =90 for Hopper,
    // =100 for Blackwell — regardless of actual GPU.
    const char* force = std::getenv("FORCE_ARCH");
    if (force != nullptr && force[0] != '\0') {
        cached = std::atoi(force);
        return cached;
    }

#if GROK_CUDA
    gpuDeviceProp_t prop;
    if (gpuGetDeviceProperties(&prop, 0) == GPU_SUCCESS) {
        cached = prop.major * 10 + prop.minor;
    } else {
        cached = 0;
    }
#elif GROK_HIP
    gpuDeviceProp_t prop;
    if (gpuGetDeviceProperties(&prop, 0) == GPU_SUCCESS) {
        const char* name = prop.gcnArchName;
        if (name[0] == 'g' && name[1] == 'f' && name[2] == 'x') {
            int d1 = name[3] - '0';
            int d2 = name[4] - '0';
            cached = d1 * 10 + d2;
        } else {
            cached = 0;
        }
    } else {
        cached = 0;
    }
#else
    cached = 0;
#endif
    return cached;
}

inline ArchTier get_arch_tier() {
    if (get_gpu_vendor() == GpuVendor::AMD) {
        return ArchTier::GENERIC;
    }
    int arch = get_sm_arch();
    if (arch >= 100) return ArchTier::BLACKWELL;
    if (arch >= 90) return ArchTier::HOPPER;
    if (arch >= 80) return ArchTier::AMPERE;
    return ArchTier::GENERIC;
}

// ═══════════════════════════════════════════════════════════════════════
//  AMD architecture tier
//
//  GENERIC — gfx908 (MI100): basic CDNA, wavefront-64
//  CDNA2   — gfx90a (MI250): MFMA_F32_16x16x4, 8MB L2
//  CDNA3   — gfx942 (MI300X): MFMA_F32_32x32x8_BF16, 256MB L2
// ═══════════════════════════════════════════════════════════════════════

enum class AmdTier {
    GENERIC,    // gfx908 (MI100)
    CDNA2,      // gfx90a (MI250)
    CDNA3,      // gfx942 (MI300X)
};

inline AmdTier get_amd_tier() {
#if GROK_HIP
    // FORCE_ARCH env var: 942 → CDNA3, 90 → CDNA2, anything else → GENERIC
    const char* force = std::getenv("FORCE_ARCH");
    if (force != nullptr && force[0] != '\0') {
        int arch = std::atoi(force);
        if (arch >= 942) return AmdTier::CDNA3;
        if (arch == 90)  return AmdTier::CDNA2;
        return AmdTier::GENERIC;
    }

    gpuDeviceProp_t prop;
    if (gpuGetDeviceProperties(&prop, 0) == GPU_SUCCESS) {
        const char* name = prop.gcnArchName;
        // gcnArchName format: "gfx942:sramecc+:xnack-" (may have suffixes)
        if (name[0] == 'g' && name[1] == 'f' && name[2] == 'x') {
            // Check for known CDNA arch names (strncmp ignores suffixes)
            if (name[3] == '9' && name[4] == '4' && name[5] == '2')
                return AmdTier::CDNA3;   // MI300X
            if (name[3] == '9' && name[4] == '0' && name[5] == 'a')
                return AmdTier::CDNA2;   // MI250
        }
    }
#endif
    return AmdTier::GENERIC;
}

inline const char* get_amd_tier_name() {
    AmdTier tier = get_amd_tier();
    switch (tier) {
        case AmdTier::CDNA3:   return "cdna3";
        case AmdTier::CDNA2:   return "cdna2";
        default:               return "generic";
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Optimizer precision enums (used by generated kernel dispatch)
// ═══════════════════════════════════════════════════════════════════════

enum class StatePrecision {
    FP32 = 0,
    CONFIG4 = 1,
};

enum class ExpertPrecision {
    FP32 = 0,
    INT8 = 1,
    INT4 = 2,
    MXFP4 = 3,
};

