// =====================================================================
//  tuned_configs.h — autotune output (defaults until autotune is run)
//
//  Per-(kernel, arch, shape-bucket) launch parameters and tile sizes.
//  3-arch active set: sm_90 (Hopper) and gfx942 (CDNA3). tpu_v5p uses
//  XLA's autotuner and does not appear in these tables.
//
//  The values in this file are DEFAULT placeholders chosen to match the
//  current hand-coded __launch_bounds__ in the kernels (BLOCK_SIZE=256,
//  MIN_BLOCKS_PER_SM=2 or 8 depending on optimizer). Run autotune on
//  hardware to overwrite with measured winners.
// =====================================================================

#pragma once

#include <cstdint>
#include <stdexcept>

namespace sg {

// ---------------------------------------------------------------------
// Arch identifiers used to index the tables. Must match the values
// returned by csrc/bindings/dispatch.cpp::detect_arch().
// ---------------------------------------------------------------------

enum ArchId : int {
    ARCH_SM90   = 0,    // Hopper (H100, H200)
    ARCH_GFX942 = 1,    // CDNA3 (MI300X, MI300A)
    NUM_ARCHES  = 2,
};

inline ArchId arch_id_from_int(int a) {
    switch (a) {
        case 90:  return ARCH_SM90;
        case 942: return ARCH_GFX942;
        default:
            throw std::runtime_error(
                "arch_id_from_int: unsupported arch " + std::to_string(a) +
                " (3-arch active set is sm_90, gfx942, tpu_v5p)");
    }
}

// ---------------------------------------------------------------------
// LaunchConfig — what every per-arch kernel pulls in for its template
// parameters and __launch_bounds__.
// ---------------------------------------------------------------------

struct LaunchConfig {
    int  block_size;          // threads per block
    int  min_blocks_per_sm;   // hint for __launch_bounds__
    int  block_m;             // tile M dim (GEMM); 0 if N/A
    int  block_n;             // tile N dim (GEMM); 0 if N/A
    int  stages;              // pipeline depth (cp.async/TMA); 0 if N/A
    bool vec4;                // float4 vectorized fast path
};

// Default config when autotune has not been run for this kernel/arch yet.
inline constexpr LaunchConfig DEFAULT_CONFIG = {
    /* block_size        */ 256,
    /* min_blocks_per_sm */ 2,
    /* block_m           */ 0,
    /* block_n           */ 0,
    /* stages            */ 0,
    /* vec4              */ true,
};

// ---------------------------------------------------------------------
// Per-kernel tables. Indexed by [arch_id][shape_bucket].
//
// shape_bucket index: 0=N<=4096, 1=N<=65k, 2=N<=512k, 3=N<=4M, 4=N<=32M+
// ---------------------------------------------------------------------

inline constexpr LaunchConfig GROKADAMW_CONFIGS[NUM_ARCHES][5] = {
    /* sm_90   */ {DEFAULT_CONFIG, DEFAULT_CONFIG, DEFAULT_CONFIG, DEFAULT_CONFIG, DEFAULT_CONFIG},
    /* gfx942  */ {DEFAULT_CONFIG, DEFAULT_CONFIG, DEFAULT_CONFIG, DEFAULT_CONFIG, DEFAULT_CONFIG},
};

inline int bucket_for_N(int N) {
    if (N <= 4096)    return 0;
    if (N <= 65536)   return 1;
    if (N <= 524288)  return 2;
    if (N <= 4194304) return 3;
    return 4;
}

inline LaunchConfig get_grokadamw_config(int arch, int N) {
    return GROKADAMW_CONFIGS[arch_id_from_int(arch)][bucket_for_N(N)];
}

} // namespace sg
