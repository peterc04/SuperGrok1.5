#pragma once
// Autotuner-consumable defaults for kernel launch parameters.
//
// When compile.py's autotuner emits -DSG_TUNED_BLOCK_SIZE=N (etc.)
// those override these defaults. Production kernels read these macros
// instead of hardcoding values, making the autotuner search effective.
//
// Search dimensions (compile.py:1097-1620):
//   - SG_TUNED_BLOCK_SIZE:  threads per block (elementwise kernels)
//   - SG_TUNED_VEC_WIDTH:   float4/float2/float1 vectorization width
//   - SG_TUNED_UNROLL:      #pragma unroll factor for inner loops
//   - SG_TUNED_NUM_STAGES:  pipeline stages (cp.async / async copy)
//   - SG_TUNED_ASYNC_DEPTH: prefetch depth for software pipelining
//
// Arch-specific dimensions (active only when capability flags are set):
//   - SG_TUNED_WGMMA_SHAPE:          sm_90 wgmma tile shape
//   - SG_TUNED_WARP_SPECIALIZATION:   sm_90 producer/consumer warps
//   - SG_TUNED_CLUSTER_SHAPE:         sm_90 thread block cluster
//   - SG_TUNED_MFMA_SHAPE:            gfx942 MFMA tile shape
//   - SG_TUNED_WAVES_PER_EU:          gfx942 occupancy hint
//   - SG_TUNED_LDS_PADDING:           gfx942 LDS bank conflict padding

// ── Universal defaults ──────────────────────────────────────────────

#ifndef SG_TUNED_BLOCK_SIZE
#define SG_TUNED_BLOCK_SIZE 256
#endif

#ifndef SG_TUNED_VEC_WIDTH
#define SG_TUNED_VEC_WIDTH 4
#endif

#ifndef SG_TUNED_UNROLL
#define SG_TUNED_UNROLL 1
#endif

#ifndef SG_TUNED_NUM_STAGES
#define SG_TUNED_NUM_STAGES 1
#endif

#ifndef SG_TUNED_ASYNC_DEPTH
#define SG_TUNED_ASYNC_DEPTH 2
#endif

// ── CUDA sm_90 (Hopper) defaults ────────────────────────────────────

#ifndef SG_TUNED_WGMMA_SHAPE
#define SG_TUNED_WGMMA_SHAPE 0
#endif

#ifndef SG_TUNED_WARP_SPECIALIZATION
#define SG_TUNED_WARP_SPECIALIZATION 0
#endif

#ifndef SG_TUNED_CLUSTER_SHAPE
#define SG_TUNED_CLUSTER_SHAPE 1
#endif

#ifndef SG_TUNED_FP8_LAYOUT
#define SG_TUNED_FP8_LAYOUT 0
#endif

#ifndef SG_TUNED_FP4_LAYOUT
#define SG_TUNED_FP4_LAYOUT 0
#endif

#ifndef SG_TUNED_TCGEN05_VARIANT
#define SG_TUNED_TCGEN05_VARIANT 0
#endif

#ifndef SG_TUNED_SWIZZLE
#define SG_TUNED_SWIZZLE 0
#endif

#ifndef SG_TUNED_TMA_DESCRIPTORS
#define SG_TUNED_TMA_DESCRIPTORS 0
#endif

// ── AMD gfx942 (CDNA3) defaults ────────────────────────────────────

#ifndef SG_TUNED_MFMA_SHAPE
#define SG_TUNED_MFMA_SHAPE 0
#endif

#ifndef SG_TUNED_WAVES_PER_EU
#define SG_TUNED_WAVES_PER_EU 0
#endif

#ifndef SG_TUNED_LDS_PADDING
#define SG_TUNED_LDS_PADDING 0
#endif

#ifndef SG_TUNED_SCHEDULER_HINT
#define SG_TUNED_SCHEDULER_HINT 0
#endif

#ifndef SG_TUNED_FP8_MFMA_LAYOUT
#define SG_TUNED_FP8_MFMA_LAYOUT 0
#endif

#ifndef SG_TUNED_FP4_MFMA_LAYOUT
#define SG_TUNED_FP4_MFMA_LAYOUT 0
#endif

// ── Capability feature flags (emitted by compile.py:8035-8051) ──────
// These are set by compile.py when the target arch supports the feature.
// Kernels can #ifdef on them to enable arch-specific code paths.
// Defaults to 0 (disabled) when not set by the build system.

#ifndef CUDA_TMA_ENABLED
#define CUDA_TMA_ENABLED 0
#endif

#ifndef CUDA_WGMMA_ENABLED
#define CUDA_WGMMA_ENABLED 0
#endif

#ifndef CUDA_CLUSTER_ENABLED
#define CUDA_CLUSTER_ENABLED 0
#endif

#ifndef CUDA_FP8_ENABLED
#define CUDA_FP8_ENABLED 0
#endif

#ifndef CUDA_FP4_ENABLED
#define CUDA_FP4_ENABLED 0
#endif

#ifndef CUDA_TCGEN05_ENABLED
#define CUDA_TCGEN05_ENABLED 0
#endif

#ifndef AMDGPU_MFMA_ENABLED
#define AMDGPU_MFMA_ENABLED 0
#endif
