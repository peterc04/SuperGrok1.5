/*
 * SuperGrok v2 — Runtime Architecture Dispatch
 *
 * Detects GPU compute capability at runtime and provides dispatch
 * helpers for selecting the optimal kernel tier.
 *
 * Three tiers:
 *   GENERIC  — sm_70, sm_75 (V100, T4): FP32, basic smem
 *   AMPERE   — sm_80, sm_86, sm_89 (A100, RTX 3090, L4, RTX 4090):
 *              TF32 Tensor Cores, cp.async, 192KB smem, BF16
 *   HOPPER   — sm_90, sm_100 (H100, B200):
 *              TMA, Thread Block Clusters, FP8 Tensor Cores, 228KB smem
 */

#pragma once

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

enum class ArchTier {
    GENERIC,  // sm_70, sm_75
    AMPERE,   // sm_80, sm_86, sm_89
    HOPPER,   // sm_90, sm_100
};

inline int get_sm_arch() {
    static int cached = -1;
    if (cached >= 0) return cached;
#ifdef WITH_CUDA
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        cached = prop.major * 10 + prop.minor;
    } else {
        cached = 0;
    }
#else
    cached = 0;
#endif
    return cached;
}

inline ArchTier get_arch_tier() {
    int arch = get_sm_arch();
    if (arch >= 90) return ArchTier::HOPPER;
    if (arch >= 80) return ArchTier::AMPERE;
    return ArchTier::GENERIC;
}
