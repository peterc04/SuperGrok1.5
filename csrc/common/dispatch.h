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
    GENERIC,  // sm_70, sm_75 (NVIDIA) or gfx908/gfx90a/gfx942 (AMD)
    AMPERE,   // sm_80, sm_86, sm_89
    HOPPER,   // sm_90, sm_100
};

inline int get_sm_arch() {
    static int cached = -1;
    if (cached >= 0) return cached;
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
        // HIP gcnArchName: e.g., "gfx90a" → 90, "gfx942" → 94
        // Map to a tier rather than a direct SM number
        const char* name = prop.gcnArchName;
        if (name[0] == 'g' && name[1] == 'f' && name[2] == 'x') {
            // Parse first two digits after "gfx"
            int d1 = name[3] - '0';
            int d2 = name[4] - '0';
            cached = d1 * 10 + d2;  // e.g., gfx90a → 90, gfx94x → 94
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
        // AMD uses GENERIC tier — no Ampere/Hopper-specific kernels
        return ArchTier::GENERIC;
    }
    int arch = get_sm_arch();
    if (arch >= 90) return ArchTier::HOPPER;
    if (arch >= 80) return ArchTier::AMPERE;
    return ArchTier::GENERIC;
}
