// csrc/backends/cuda/sm_90/models/decoder.cuh
// Autoregressive Decoder Transformer for sm_90 (Hopper).
//
// Reference (grokking_race_v2.py lines 318-340):
//   class DecoderBlock:
//     attn = MultiheadAttention(d, h)
//     n1 = LayerNorm(d); n2 = LayerNorm(d)
//     ff = Linear(d, 4d) -> GELU -> Linear(4d, d)
//     forward(x):
//       a, _ = attn(x, x, x, attn_mask=causal)
//       x = n1(x + a)
//       return n2(x + ff(x))
//   class Transformer:
//     tok = Embedding(ntok, d); pos = Embedding(seq, d)
//     forward(x):
//       h = tok(x) + pos(pos_ids)
//       for l in layers: h = l(h)
//       return out(norm(h)[:, -1, :])
//
// Post-norm style; output is the LAST token only.
//
// ─── Weight buffer layout (contiguous packed) ────────────────────────
//   tok_embed     [vocab, d]
//   pos_embed     [seq,   d]
//   per layer ℓ:
//     n1_g [d], n1_b [d]
//     qkv_W [3d, d], qkv_b [3d]
//     out_W [d, d],  out_b [d]
//     n2_g [d], n2_b [d]
//     ff1_W [4d, d], ff1_b [4d]
//     ff2_W [d, 4d], ff2_b [d]
//   final_g [d], final_b [d]
//   vocab_W [vocab, d], vocab_b [vocab]
//
// ─── Activation scratch layout (per call) ────────────────────────────
//   input_ids_saved    [B, S]                 (forward stashes input here)
//   embed_out          [B, S, d]
//   per layer ℓ:
//     qkv_in           [B, S, d]      (input to QKV proj == prev layer output)
//     qkv_out          [B, S, 3d]
//     attn_in_qkv      [B, S, 3d]     (q,k,v in [B,H,S,d_h] layout, packed)
//     attn_out_perhead [B, S, d]      (attention output [B,H,S,d_h])
//     attn_out         [B, S, d]      (reshaped [B,S,D])
//     attn_proj        [B, S, d]      (after out_W + bias)
//     n1_in            [B, S, d]      (qkv_in + attn_proj — layernorm input)
//     layer1_out       [B, S, d]      (LN(n1_in) — input to FFN block)
//     ffn_pre          [B, S, 4d]     (pre-GELU)
//     ffn_post         [B, S, 4d]     (post-GELU)
//     ffn_out          [B, S, d]      (after ff2)
//     n2_in            [B, S, d]      (layer1_out + ffn_out)
//     layer_out        [B, S, d]      (LN(n2_in) — block output)
//     softmax_lse      [B, H, S]      (FP32 bit-aliased)
//   final_norm_out     [B, S, d]
//   logits_full        [B, S, vocab]
//
// Total scratch (in T elements):
//   B*S + B*S*D + n_layers * per_layer_total + B*S*D + B*S*V
//
// All multi-token GEMMs go through cuBLAS; LayerNorm + residual is fused
// in a single warp-reduction kernel; GELU is a separate elementwise kernel.

#pragma once
// ── inlined from former csrc/common/platform.h ──
/*
 * SuperGrok v2 — Platform Abstraction Layer
 *
 * Provides a unified API across NVIDIA CUDA and AMD HIP (ROCm).
 * Include this header instead of raw <cuda.h> / <hip/hip_runtime.h>.
 *
 * Key differences handled:
 *   - Warp size: CUDA = 32, HIP/RDNA = 32, HIP/CDNA = 64
 *   - __sincosf: CUDA intrinsic, HIP uses sincosf (no double-underscore)
 *   - __ldg: CUDA L1 cache hint, no-op on HIP (compiler handles caching)
 *   - Thrust → rocThrust, CUB → hipCUB (header-compatible wrappers)
 *   - cuBLAS → rocBLAS (ATen abstracts this via at::cuda::getCurrentCUDABlasHandle)
 */


// ═══════════════════════════════════════════════════════════════════════
//  Backend detection
// ═══════════════════════════════════════════════════════════════════════

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define GROK_HIP 1
#define GROK_CUDA 0
#else
#define GROK_HIP 0
#define GROK_CUDA 1
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Runtime includes
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
#include <hip/hip_runtime.h>
// rocThrust and hipCUB provide thrust/cub API compatibility
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <hipcub/hipcub.hpp>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Stream type alias
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
using GpuStream_t = hipStream_t;
#else
using GpuStream_t = cudaStream_t;
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Warp / wavefront size
//
//  CDNA (MI200, MI300): wavefront = 64
//  RDNA (RX 7900):      wavefront = 32
//  NVIDIA:               warp     = 32
//
//  We default to the compile-time warp size. On HIP, __AMDGCN_WAVEFRONT_SIZE__
//  is set by the compiler for the target architecture.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #ifdef __AMDGCN_WAVEFRONT_SIZE__
    #define WARP_SIZE __AMDGCN_WAVEFRONT_SIZE__
  #else
    #define WARP_SIZE 64  // conservative default for CDNA
  #endif
#else
  #define WARP_SIZE 32
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Warp shuffle
//
//  CUDA: __shfl_down_sync(mask, val, offset)
//  HIP:  __shfl_down(val, offset)  — no mask parameter on CDNA
//        (On wavefront-64, all lanes are always synchronized)
//
//  We wrap both into SHFL_DOWN(val, offset) and SHFL_DOWN_SYNC(mask, val, offset).
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define SHFL_DOWN(val, offset) __shfl_down((val), (offset))
  #define SHFL_DOWN_SYNC(mask, val, offset) __shfl_down((val), (offset))
#else
  #define SHFL_DOWN(val, offset) __shfl_down_sync(0xFFFFFFFF, (val), (offset))
  #define SHFL_DOWN_SYNC(mask, val, offset) __shfl_down_sync((mask), (val), (offset))
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Fast sincos
//
//  CUDA: __sincosf (device intrinsic, single instruction on SM)
//  HIP:  sincosf   (no double-underscore variant)
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define FAST_SINCOSF(x, sptr, cptr) sincosf((x), (sptr), (cptr))
#else
  #define FAST_SINCOSF(x, sptr, cptr) __sincosf((x), (sptr), (cptr))
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Read-only cache load hint
//
//  CUDA: __ldg(ptr) — hints L1 cache for read-only data
//  HIP:  direct dereference (compiler manages caching on GCN/CDNA)
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define LDG(ptr) (*(ptr))
#else
  #define LDG(ptr) __ldg(ptr)
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Error checking
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define GPU_SUCCESS hipSuccess
  #define gpuGetLastError hipGetLastError
  #define gpuGetErrorString hipGetErrorString
  #define gpuDeviceSynchronize hipDeviceSynchronize
  #define gpuGetDeviceProperties hipGetDeviceProperties
  #define gpuDeviceProp_t hipDeviceProp_t
#else
  #define GPU_SUCCESS cudaSuccess
  #define gpuGetLastError cudaGetLastError
  #define gpuGetErrorString cudaGetErrorString
  #define gpuDeviceSynchronize cudaDeviceSynchronize
  #define gpuGetDeviceProperties cudaGetDeviceProperties
  #define gpuDeviceProp_t cudaDeviceProp
#endif

// ═══════════════════════════════════════════════════════════════════════
//  CUB / hipCUB namespace alias
//
//  hipCUB wraps rocPRIM with a CUB-compatible API.
//  We alias so kernel code can use `cub::DeviceSegmentedRadixSort` uniformly.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  namespace cub = hipcub;
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Full-mask constant for warp-wide operations
//
//  CUDA uses explicit masks (0xFFFFFFFF for 32 lanes).
//  HIP/CDNA doesn't use masks — all 64 lanes in a wavefront are lockstep.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define FULL_WARP_MASK 0  // unused, but defined for code that passes it around
#else
  #define FULL_WARP_MASK 0xFFFFFFFF
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Async memset
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define gpuMemsetAsync hipMemsetAsync
#else
  #define gpuMemsetAsync cudaMemsetAsync
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Stream management
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define gpuStreamCreate hipStreamCreate
  #define gpuStreamSynchronize hipStreamSynchronize
  #define gpuStreamDestroy hipStreamDestroy
#else
  #define gpuStreamCreate cudaStreamCreate
  #define gpuStreamSynchronize cudaStreamSynchronize
  #define gpuStreamDestroy cudaStreamDestroy
#endif

// ═══════════════════════════════════════════════════════════════════════
//  GCN/CDNA scheduler hints (AMD-only occupancy control)
//
//  __attribute__((amdgpu_waves_per_eu(min, max))) controls occupancy
//  on AMD GCN/CDNA by limiting waves per execution unit. On NVIDIA,
//  __launch_bounds__ serves this purpose (already applied separately).
//
//  GROK_WAVES_PER_EU(min, max) — applies attribute on HIP, no-op on CUDA.
//  GROK_FLAT_WORK_GROUP_SIZE(min, max) — hints block size range for AMD.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define GROK_WAVES_PER_EU(min_waves, max_waves) \
      __attribute__((amdgpu_waves_per_eu(min_waves, max_waves)))
  #define GROK_FLAT_WORK_GROUP_SIZE(min_size, max_size) \
      __attribute__((amdgpu_flat_work_group_size(min_size, max_size)))
#else
  #define GROK_WAVES_PER_EU(min_waves, max_waves)
  #define GROK_FLAT_WORK_GROUP_SIZE(min_size, max_size)
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Kernel launch attribute (for configuring smem size)
// ═══════════════════════════════════════════════════════════════════════

#if GROK_HIP
  #define gpuFuncSetAttribute hipFuncSetAttribute
  #define gpuFuncAttributeMaxDynamicSharedMemorySize \
          hipFuncAttributeMaxDynamicSharedMemorySize
#else
  #define gpuFuncSetAttribute cudaFuncSetAttribute
  #define gpuFuncAttributeMaxDynamicSharedMemorySize \
          cudaFuncAttributeMaxDynamicSharedMemorySize
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Non-temporal (streaming) memory access
//
//  Used for optimizer state access to avoid L2 cache pollution.
//  Model weights stay warm in L2 for the next forward pass.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_CUDA
  // Streaming load: reads bypass L2 (or use L2 read-only path)
  __device__ __forceinline__ float stream_load(const float* ptr) {
      float val;
      asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
      return val;
  }

  // Streaming store: writes bypass L2 allocation
  // Available on sm_80+ (Ampere). On older, falls back to normal store.
  __device__ __forceinline__ void stream_store(float* ptr, float val) {
  #if __CUDA_ARCH__ >= 800
      asm volatile("st.global.wt.f32 [%0], %1;" :: "l"(ptr), "f"(val));
  #else
      *ptr = val;
  #endif
  }

  // float4 streaming variants
  __device__ __forceinline__ float4 stream_load4(const float4* ptr) {
      float4 val;
      asm volatile(
          "ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
          : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
          : "l"(ptr));
      return val;
  }

  __device__ __forceinline__ void stream_store4(float4* ptr, float4 val) {
  #if __CUDA_ARCH__ >= 800
      asm volatile(
          "st.global.wt.v4.f32 [%0], {%1,%2,%3,%4};"
          :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w));
  #else
      *ptr = val;
  #endif
  }

#elif GROK_HIP
  // HIP: use __builtin_nontemporal_load/store
  __device__ __forceinline__ float stream_load(const float* ptr) {
      return __builtin_nontemporal_load(ptr);
  }
  __device__ __forceinline__ void stream_store(float* ptr, float val) {
      __builtin_nontemporal_store(val, ptr);
  }
  // float4 variants: decompose into 4 scalar non-temporal ops
  __device__ __forceinline__ float4 stream_load4(const float4* ptr) {
      const float* fp = reinterpret_cast<const float*>(ptr);
      return make_float4(
          __builtin_nontemporal_load(fp),
          __builtin_nontemporal_load(fp+1),
          __builtin_nontemporal_load(fp+2),
          __builtin_nontemporal_load(fp+3));
  }
  __device__ __forceinline__ void stream_store4(float4* ptr, float4 val) {
      float* fp = reinterpret_cast<float*>(ptr);
      __builtin_nontemporal_store(val.x, fp);
      __builtin_nontemporal_store(val.y, fp+1);
      __builtin_nontemporal_store(val.z, fp+2);
      __builtin_nontemporal_store(val.w, fp+3);
  }
#else
  // CPU: no non-temporal hint needed (OS manages caching)
  static inline float stream_load(const float* ptr) { return *ptr; }
  static inline void stream_store(float* ptr, float val) { *ptr = val; }
#endif
// ── end inlined csrc/common/platform.h ──
// ── inlined from former csrc/common/types.h ──
/*
 * SuperGrok v2 — Shared Types and Constants
 *
 * Common struct definitions and compile-time constants used by both
 * forward and backward CUDA kernels.
 */



#include <vector>
#include <torch/extension.h>

// ═══════════════════════════════════════════════════════════════════════
//  BatchedScanCtx — shared bookkeeping for the batched mamba peer step
//
//  Recovered verbatim from the deleted csrc/common/ops.h@682eab4^. The
//  setup helper batched_step_setup_and_sort builds one of these per call;
//  the scan + fused-elem helpers consume it. Per-arch variants in
//  csrc/kernels/{cuda,hip}/<arch>/supergrok2_fwd_*.{cu,hip.cpp}
//  expect this exact layout.
//
//  Layout (14 members): 3 int + 2 std::vector<int>
//                     + 8 torch::Tensor + 1 std::vector<torch::Tensor>.
// ═══════════════════════════════════════════════════════════════════════

struct BatchedScanCtx {
    int num_params;
    int total_N;
    int max_N;
    std::vector<int> N_vec;
    std::vector<int> seg_offsets_cpu;
    torch::Tensor x_sorted_packed;
    torch::Tensor offsets_t;
    torch::Tensor initial_fwd;
    torch::Tensor initial_bwd;
    torch::Tensor final_fwd;
    torch::Tensor final_bwd;
    torch::Tensor fwd_scan_packed;
    torch::Tensor bwd_scan_packed;
    std::vector<torch::Tensor> unsort_idx_list;
};

// ═══════════════════════════════════════════════════════════════════════
//  Compile-time constants
// ═══════════════════════════════════════════════════════════════════════

constexpr int MAX_D_STATE = 128;
constexpr int MAX_D_INNER = 128;
constexpr int MAX_D_MODEL = 64;
constexpr int MAX_GRU_HIDDEN = 8;
constexpr int MAX_EXPERT_HIDDEN = 16;
constexpr int MAX_TOPK = 4;
constexpr int MAX_CKPT_INTERVAL = 32;   // max checkpoint interval for bilevel gradient checkpointing

constexpr int SG2M_BLOCK = 256;         // forward kernel block size
constexpr int SG2B_BLOCK = 256;         // backward kernel block size
constexpr int PSCAN_BLOCK = 512;        // threads per parallel scan block (must be power of 2)
constexpr int PSCAN_THRESHOLD = 256;    // fall back to sequential scan if N < this
constexpr int GEMM_PRECOMPUTE_THRESHOLD = 1024;  // use GEMM when N >= this

// ═══════════════════════════════════════════════════════════════════════
//  Parallel Prefix Scan Infrastructure
//
//  Affine2x2 and affine_combine moved to csrc/scan/affine2x2.h.
//  Included here so existing callers that #include "csrc/common/types.h"
//  continue to compile without modification.
// ═══════════════════════════════════════════════════════════════════════

// ── inlined from former csrc/scan/affine2x2.h ──
// Affine2x2 — shared scan primitive.
//
// Extracted from csrc/common/types.h. The associative operator used by the
// Mamba parallel prefix scan and by SuperGrok v2's selective scan.
//
// Encoding: each scan element is a 2x2 affine transform
//   (h_new) = (m00 m01) (h) + (b0)
//   (h_new')  (m10 m11) (h')  (b1)
//
// composition: (B ∘ A)(h) = B(A(h))
//   M_out = M_B * M_A
//   b_out = M_B * b_A + b_B
//
// Used by:
//   csrc/scan/mamba_scan_adapter.cuh  — Mamba model selective scan
//   csrc/algorithms/supergrok2.h      — SG2 optimizer scan recurrence
//   csrc/backends/cuda/sm_90/launch_supergrok2.cu — Blelloch parallel scan

#ifdef __CUDACC__

struct Affine2x2 {
    float m00, m01, m10, m11;  // 2x2 matrix
    float b0, b1;               // 2-vector bias
};

__device__ __forceinline__ Affine2x2 affine_identity() {
    return {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ Affine2x2 affine_combine(Affine2x2 left, Affine2x2 right) {
    // Computes right ∘ left: apply left first, then right.
    // M_out = M_right * M_left
    // b_out = M_right * b_left + b_right
    Affine2x2 out;
#if defined(GROK_CUDA) && GROK_CUDA
    // Inline PTX: 12-FMA composition arranged for ILP across pipelines.
    asm volatile(
        "fma.rn.f32 %0, %6, %12, 0f00000000;\n\t"
        "fma.rn.f32 %1, %6, %13, 0f00000000;\n\t"
        "fma.rn.f32 %2, %8, %12, 0f00000000;\n\t"
        "fma.rn.f32 %3, %8, %13, 0f00000000;\n\t"
        "fma.rn.f32 %0, %7, %14, %0;\n\t"
        "fma.rn.f32 %1, %7, %15, %1;\n\t"
        "fma.rn.f32 %2, %9, %14, %2;\n\t"
        "fma.rn.f32 %3, %9, %15, %3;\n\t"
        "fma.rn.f32 %4, %6, %16, %10;\n\t"
        "fma.rn.f32 %5, %8, %16, %11;\n\t"
        "fma.rn.f32 %4, %7, %17, %4;\n\t"
        "fma.rn.f32 %5, %9, %17, %5;\n\t"
        : "=f"(out.m00), "=f"(out.m01), "=f"(out.m10), "=f"(out.m11),
          "=f"(out.b0), "=f"(out.b1)
        : "f"(right.m00), "f"(right.m01), "f"(right.m10), "f"(right.m11),
          "f"(right.b0), "f"(right.b1),
          "f"(left.m00), "f"(left.m01), "f"(left.m10), "f"(left.m11),
          "f"(left.b0), "f"(left.b1)
    );
#else
    // HIP/CPU fallback: C++ implementation (HIP has different inline asm syntax)
    out.m00 = right.m00 * left.m00 + right.m01 * left.m10;
    out.m01 = right.m00 * left.m01 + right.m01 * left.m11;
    out.m10 = right.m10 * left.m00 + right.m11 * left.m10;
    out.m11 = right.m10 * left.m01 + right.m11 * left.m11;
    out.b0  = right.m00 * left.b0  + right.m01 * left.b1 + right.b0;
    out.b1  = right.m10 * left.b0  + right.m11 * left.b1 + right.b1;
#endif
    return out;
}

#endif // __CUDACC__
// ── end inlined csrc/scan/affine2x2.h ──

#ifdef __CUDACC__

// ═══════════════════════════════════════════════════════════════════════
//  Branchless Stochastic Rounding (Config4 / INT8 quantized kernels)
//
//  Converts float to int8 with stochastic rounding. The ternary compiles
//  to a PTX selp instruction at -O2, avoiding warp divergence.
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ int8_t float_to_int8_stochastic_branchless(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / fmaxf(scale, 1e-12f);
    float trunc_val = truncf(scaled);
    float frac = fabsf(scaled - trunc_val);
    float threshold = (float)(rand_bits & 0xFFFF) * (1.0f / 65536.0f);
    // Branchless: ternary compiles to selp on nvcc -O2
    float round_up = (frac > threshold) ? copysignf(1.0f, scaled) : 0.0f;
    float result = trunc_val + round_up;
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, result));
}

#endif  // __CUDACC__
// ── end inlined csrc/common/types.h ──
// ── inlined from former csrc/common/utils.cuh ──
/*
 * SuperGrok v2 — Shared Device Helpers
 *
 * Device utility functions used by multiple kernel files.
 * Uses platform.h macros for CUDA/HIP portability.
 */


#if GROK_CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Warp-level reduction helper
//
//  Sum a float across d_inner threads (all in one warp, d_inner ≤ WARP_SIZE).
//  Uses platform-abstracted shuffle; works for any d_inner ≤ WARP_SIZE
//  (including non-power-of-2).
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float warp_reduce_sum(float val, int d_inner, int tid) {
    unsigned mask = (d_inner < WARP_SIZE) ? ((1u << d_inner) - 1) : FULL_WARP_MASK;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = SHFL_DOWN_SYNC(mask, val, offset);
        if (tid + offset < d_inner)
            val += other;
    }
    return val;  // only lane 0 has the correct sum
}

// ═══════════════════════════════════════════════════════════════════════
//  Stochastic Rounding for Quantized Optimizer States (Config 3)
//
//  Hash-based PRNG: deterministic per (step, element) pair, no state needed.
//  Faster than cuRAND, no separate state tensor required.
// ═══════════════════════════════════════════════════════════════════════

// Hash-based PRNG (Philox-like): deterministic, no state
__device__ __forceinline__ unsigned hash_prng(unsigned step, unsigned idx) {
    unsigned h = (step * 2654435761u) ^ (idx * 2246822519u);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

#if GROK_CUDA || GROK_HIP

// BF16 stochastic rounding: unbiased quantization
__device__ __forceinline__ __nv_bfloat16 float_to_bf16_stochastic(float val, unsigned rand_bits) {
    unsigned bits = __float_as_uint(val);
    unsigned truncated = bits & 0xFFFF;     // bits that BF16 drops
    unsigned threshold = rand_bits & 0xFFFF; // random 16-bit threshold
    if (truncated > threshold) {
        bits += 0x10000;  // round up
    }
    bits &= 0xFFFF0000;  // truncate to BF16
    return __float2bfloat16(__uint_as_float(bits));
}

// INT8 per-block quantization with stochastic rounding
// block_size elements share one FP32 scale factor
__device__ __forceinline__ int8_t float_to_int8_stochastic(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / scale;
    float truncated = truncf(scaled);
    float frac = fabsf(scaled - truncated);
    float threshold = (float)(rand_bits & 0xFFFF) / 65536.0f;
    if (frac > threshold) {
        truncated += (scaled > 0) ? 1.0f : -1.0f;
    }
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, truncated));
}

// ═══════════════════════════════════════════════════════════════════════
//  Phase 3: Inline PTX for Hot Inner Loops
//
//  Hand-tuned PTX for critical paths in the SG2 fused_elem pipeline.
//  These replace compiler-generated code in the highest-frequency loops.
// ═══════════════════════════════════════════════════════════════════════

#if GROK_CUDA

// Fast reciprocal sqrt via PTX rsqrt.approx.f32 + Newton-Raphson refinement.
// 2-3x faster than sqrtf(x) + fdividef for Adam denominator.
__device__ __forceinline__ float fast_rsqrt_nr(float x) {
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    // One Newton-Raphson iteration: r = r * (1.5 - 0.5 * x * r * r)
    r = r * (1.5f - 0.5f * x * r * r);
    return r;
}

// Fused multiply-add via PTX fma.rn.f32 — ensures single FMA instruction.
// Critical for affine_combine inner loop (8 FMAs per composition).
__device__ __forceinline__ float ptx_fma(float a, float b, float c) {
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

// Fast exp2 approximation via PTX ex2.approx.f32.
// Used in Mamba scan: exp(A * dt) = exp2(A * dt / ln2).
__device__ __forceinline__ float ptx_exp2(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// Fast log2 via PTX lg2.approx.f32.
__device__ __forceinline__ float ptx_log2(float x) {
    float r;
    asm("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// Fast exp via exp2: exp(x) = exp2(x * log2(e))
__device__ __forceinline__ float ptx_expf(float x) {
    return ptx_exp2(x * 1.4426950408889634f);  // log2(e)
}

// Fast tanh approximation via exp2: tanh(x) = (e^2x - 1) / (e^2x + 1)
// Used in GRU h_tilde computation.
__device__ __forceinline__ float ptx_tanhf(float x) {
    float e2x = ptx_exp2(2.0f * x * 1.4426950408889634f);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

// Fast sigmoid via exp2: sigmoid(x) = 1 / (1 + exp(-x))
// Used in GRU z_gate and r_gate.
__device__ __forceinline__ float ptx_sigmoidf(float x) {
    float en = ptx_exp2(-x * 1.4426950408889634f);
    return 1.0f / (1.0f + en);
}

// Blelloch affine_combine using pure PTX FMA instructions.
// Composes two Affine2x2 transforms: result = left ∘ right
// M_out = M_left * M_right, b_out = M_left * b_right + b_left
// This is the inner loop of the parallel prefix scan (called O(log N) times).
__device__ __forceinline__ Affine2x2 ptx_affine_combine(
    const Affine2x2& left, const Affine2x2& right
) {
    Affine2x2 out;
    // M_out = M_left * M_right (2x2 matrix multiply via 8 FMAs)
    out.m00 = ptx_fma(left.m00, right.m00, left.m01 * right.m10);
    out.m01 = ptx_fma(left.m00, right.m01, left.m01 * right.m11);
    out.m10 = ptx_fma(left.m10, right.m00, left.m11 * right.m10);
    out.m11 = ptx_fma(left.m10, right.m01, left.m11 * right.m11);
    // b_out = M_left * b_right + b_left
    out.b0 = ptx_fma(left.m00, right.b0, ptx_fma(left.m01, right.b1, left.b0));
    out.b1 = ptx_fma(left.m10, right.b0, ptx_fma(left.m11, right.b1, left.b1));
    return out;
}

// Expert MLP forward pass — single expert, ReLU activation.
// Inlined PTX FMA for the inner products.
// expert_hidden is typically 8-16, so fully unrollable at compile time.
template <int EXPERT_HIDDEN>
__device__ __forceinline__ float ptx_expert_mlp_forward(
    const float* __restrict__ W1,   // [expert_hidden]
    const float* __restrict__ b1,   // [expert_hidden]
    const float* __restrict__ W2,   // [expert_hidden]
    float b2,
    float input
) {
    float result = b2;
    #pragma unroll
    for (int h = 0; h < EXPERT_HIDDEN; h++) {
        float hidden = ptx_fma(W1[h], input, b1[h]);
        hidden = fmaxf(hidden, 0.0f);  // ReLU
        result = ptx_fma(W2[h], hidden, result);
    }
    return result;
}

// Stochastic rounding with PTX prmt (permute bytes) for fast bit extraction.
// Replaces the hash_prng shift+multiply chain with a single PTX instruction
// for extracting the random threshold from the hash output.
__device__ __forceinline__ int8_t ptx_int8_stochastic_round(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / fmaxf(scale, 1e-12f);
    float tr = truncf(scaled);
    float frac = fabsf(scaled - tr);
    // Extract lower 16 bits as threshold using prmt
    unsigned lo16;
    asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(lo16) : "r"(rand_bits));
    float threshold = (float)lo16 / 65536.0f;
    if (frac > threshold) tr += (scaled > 0) ? 1.0f : -1.0f;
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, tr));
}

// §25.7 DSMEM cluster reduce (sm_90+ Hopper distributed shared memory).
// Block-local warp reduce first, then cluster-wide reduce via cooperative
// groups. Falls back to warp reduce on pre-Hopper.
__device__ __forceinline__ float cluster_dsmem_reduce_sum(float val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    namespace cg = cooperative_groups;
    val = warp_reduce_sum(val, WARP_SIZE, threadIdx.x & (WARP_SIZE - 1));
    auto cluster = cg::this_cluster();
    val = cg::reduce(cluster, val, cg::plus<float>());
    return val;
#else
    return warp_reduce_sum(val, WARP_SIZE, threadIdx.x & (WARP_SIZE - 1));
#endif
}

#endif // GROK_CUDA

// HIP fallbacks — use standard math functions
#if GROK_HIP
__device__ __forceinline__ float fast_rsqrt_nr(float x) { return rsqrtf(x); }
__device__ __forceinline__ float ptx_fma(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ float ptx_expf(float x) { return expf(x); }
__device__ __forceinline__ float ptx_tanhf(float x) { return tanhf(x); }
__device__ __forceinline__ float ptx_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ Affine2x2 ptx_affine_combine(
    const Affine2x2& left, const Affine2x2& right
) {
    return affine_combine(left, right);  // Use types.h version
}
#endif // GROK_HIP

#endif // GROK_CUDA || GROK_HIP
// ── end inlined csrc/common/utils.cuh ──
#include "csrc/backends/cuda/sm_90/models/attention.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace sg { namespace sm90 { namespace models { namespace decoder {

// ─── cuBLAS dtype traits ─────────────────────────────────────────────
template <typename T> struct CublasTraits;
template <> struct CublasTraits<float> {
    static constexpr cudaDataType_t data_type = CUDA_R_32F;
    static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
};
template <> struct CublasTraits<__nv_bfloat16> {
    static constexpr cudaDataType_t data_type = CUDA_R_16BF;
    static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
};
template <> struct CublasTraits<__half> {
    static constexpr cudaDataType_t data_type = CUDA_R_16F;
    static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
};

// ─── Cast helpers ────────────────────────────────────────────────────
template <typename T>
__host__ __device__ __forceinline__ float to_float(T x) { return static_cast<float>(x); }
template <>
__host__ __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
#ifdef __CUDA_ARCH__
    return __bfloat162float(x);
#else
    return static_cast<float>(x);
#endif
}
template <>
__host__ __device__ __forceinline__ float to_float<__half>(__half x) {
#ifdef __CUDA_ARCH__
    return __half2float(x);
#else
    return static_cast<float>(x);
#endif
}
template <typename T>
__device__ __forceinline__ T from_float(float x) { return static_cast<T>(x); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

// ─── Atomic-add-on-W helpers ─────────────────────────────────────────
// FP32: builtin atomicAdd. BF16/FP16: simulate via CAS on uint16 packed
// pairs (slow path; only used for embedding/bias gradients which are
// small). Acceptable for grokking-scale (vocab=99, d=128).
__device__ __forceinline__ void atomic_add_w(float* p, float v) { atomicAdd(p, v); }
__device__ __forceinline__ void atomic_add_w(__nv_bfloat16* p, float v) {
    // Atomic on aligned uint32 holding two BF16s
    auto* p_align = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(p) & ~(size_t)0x3);
    bool is_high = (reinterpret_cast<size_t>(p) & 0x3) != 0;
    unsigned int old = *p_align, expected;
    do {
        expected = old;
        unsigned int half = is_high ? (expected >> 16) : (expected & 0xFFFF);
        __nv_bfloat16 cur = __ushort_as_bfloat16((unsigned short)half);
        float new_val = __bfloat162float(cur) + v;
        unsigned short new_bits = __bfloat16_as_ushort(__float2bfloat16(new_val));
        unsigned int packed = is_high
            ? ((expected & 0xFFFFu) | ((unsigned int)new_bits << 16))
            : ((expected & 0xFFFF0000u) | (unsigned int)new_bits);
        old = atomicCAS(p_align, expected, packed);
    } while (old != expected);
}
__device__ __forceinline__ void atomic_add_w(__half* p, float v) {
    auto* p_align = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(p) & ~(size_t)0x3);
    bool is_high = (reinterpret_cast<size_t>(p) & 0x3) != 0;
    unsigned int old = *p_align, expected;
    do {
        expected = old;
        unsigned int half = is_high ? (expected >> 16) : (expected & 0xFFFF);
        __half cur = __ushort_as_half((unsigned short)half);
        float new_val = __half2float(cur) + v;
        unsigned short new_bits = __half_as_ushort(__float2half(new_val));
        unsigned int packed = is_high
            ? ((expected & 0xFFFFu) | ((unsigned int)new_bits << 16))
            : ((expected & 0xFFFF0000u) | (unsigned int)new_bits);
        old = atomicCAS(p_align, expected, packed);
    } while (old != expected);
}

// ─── Activation layout helpers ───────────────────────────────────────
template <typename T>
struct ActLayout {
    size_t input_ids_saved;  // [B*S]
    size_t qkv_in, qkv_out, attn_in_qkv, attn_out_perhead, attn_out;
    size_t attn_proj, n1_in, layer1_out;
    size_t ffn_pre, ffn_post, ffn_out, n2_in, layer_out, softmax_lse_T;
    size_t per_layer_total;
    size_t embed_out, final_norm_out, logits_full;
};

template <typename T>
inline ActLayout<T> compute_layout(int B, int S, int D, int H, int V, int ffn_exp) {
    const size_t bsd = (size_t)B * S * D;
    const size_t bs3d = (size_t)B * S * 3 * D;
    const size_t bshd = (size_t)B * S * (size_t)ffn_exp * D;
    const size_t bhs = (size_t)B * H * S;
    const size_t bsv = (size_t)B * S * V;
    const size_t lse_T = (bhs * sizeof(float) + sizeof(T) - 1) / sizeof(T);
    ActLayout<T> L;
    L.input_ids_saved = (size_t)B * S;
    L.qkv_in = bsd;
    L.qkv_out = bs3d;
    L.attn_in_qkv = bs3d;       // packed q,k,v in [B,H,S,d] each = bsd*3
    L.attn_out_perhead = bsd;
    L.attn_out = bsd;
    L.attn_proj = bsd;
    L.n1_in = bsd;
    L.layer1_out = bsd;
    L.ffn_pre = bshd;
    L.ffn_post = bshd;
    L.ffn_out = bsd;
    L.n2_in = bsd;
    L.layer_out = bsd;
    L.softmax_lse_T = lse_T;
    L.per_layer_total =
        L.qkv_in + L.qkv_out + L.attn_in_qkv + L.attn_out_perhead +
        L.attn_out + L.attn_proj + L.n1_in + L.layer1_out +
        L.ffn_pre + L.ffn_post + L.ffn_out + L.n2_in + L.layer_out +
        L.softmax_lse_T;
    L.embed_out = bsd;
    L.final_norm_out = bsd;
    L.logits_full = bsv;
    return L;
}

template <typename T>
struct LayerScratch {
    T* qkv_in;
    T* qkv_out;
    T* attn_in_qkv;       // [3 * B*S*D] — q,k,v concatenated
    T* attn_out_perhead;  // [B*S*D] in [B,H,S,d] layout
    T* attn_out;
    T* attn_proj;
    T* n1_in;
    T* layer1_out;
    T* ffn_pre;
    T* ffn_post;
    T* ffn_out;
    T* n2_in;
    T* layer_out;
    T* softmax_lse;
};

template <typename T>
inline LayerScratch<T> slice_layer(T* base, const ActLayout<T>& L, int layer_idx) {
    T* p = base + L.input_ids_saved + L.embed_out + (size_t)layer_idx * L.per_layer_total;
    LayerScratch<T> s;
    s.qkv_in = p;            p += L.qkv_in;
    s.qkv_out = p;           p += L.qkv_out;
    s.attn_in_qkv = p;       p += L.attn_in_qkv;
    s.attn_out_perhead = p;  p += L.attn_out_perhead;
    s.attn_out = p;          p += L.attn_out;
    s.attn_proj = p;         p += L.attn_proj;
    s.n1_in = p;             p += L.n1_in;
    s.layer1_out = p;        p += L.layer1_out;
    s.ffn_pre = p;           p += L.ffn_pre;
    s.ffn_post = p;          p += L.ffn_post;
    s.ffn_out = p;           p += L.ffn_out;
    s.n2_in = p;             p += L.n2_in;
    s.layer_out = p;         p += L.layer_out;
    s.softmax_lse = p;       p += L.softmax_lse_T;
    return s;
}

// ─── Weight pointer slicing ──────────────────────────────────────────
inline size_t per_layer_weight_count(int d, int ffn_h) {
    return (size_t)d + d
         + (size_t)3*d*d + 3*d
         + (size_t)d*d + d
         + d + d
         + (size_t)ffn_h*d + ffn_h
         + (size_t)d*ffn_h + d;
}

template <typename W>
struct LayerWeights {
    const W* n1_g;  const W* n1_b;
    const W* qkv_W; const W* qkv_b;
    const W* out_W; const W* out_b;
    const W* n2_g;  const W* n2_b;
    const W* ff1_W; const W* ff1_b;
    const W* ff2_W; const W* ff2_b;
};

template <typename W>
inline LayerWeights<W> slice_layer_weights(const W* base, int d, int ffn_h) {
    LayerWeights<W> w;
    const W* p = base;
    w.n1_g = p; p += d;
    w.n1_b = p; p += d;
    w.qkv_W = p; p += (size_t)3*d*d;
    w.qkv_b = p; p += 3*d;
    w.out_W = p; p += (size_t)d*d;
    w.out_b = p; p += d;
    w.n2_g = p; p += d;
    w.n2_b = p; p += d;
    w.ff1_W = p; p += (size_t)ffn_h*d;
    w.ff1_b = p; p += ffn_h;
    w.ff2_W = p; p += (size_t)d*ffn_h;
    w.ff2_b = p; p += d;
    return w;
}

// Same struct for grad (non-const)
template <typename W>
struct LayerGrads {
    W* n1_g;  W* n1_b;
    W* qkv_W; W* qkv_b;
    W* out_W; W* out_b;
    W* n2_g;  W* n2_b;
    W* ff1_W; W* ff1_b;
    W* ff2_W; W* ff2_b;
};
template <typename W>
inline LayerGrads<W> slice_layer_grads(W* base, int d, int ffn_h) {
    LayerGrads<W> w;
    W* p = base;
    w.n1_g = p; p += d;
    w.n1_b = p; p += d;
    w.qkv_W = p; p += (size_t)3*d*d;
    w.qkv_b = p; p += 3*d;
    w.out_W = p; p += (size_t)d*d;
    w.out_b = p; p += d;
    w.n2_g = p; p += d;
    w.n2_b = p; p += d;
    w.ff1_W = p; p += (size_t)ffn_h*d;
    w.ff1_b = p; p += ffn_h;
    w.ff2_W = p; p += (size_t)d*ffn_h;
    w.ff2_b = p; p += d;
    return w;
}

// ─── Embedding kernel ────────────────────────────────────────────────
template <typename T, typename W>
__global__ void __launch_bounds__(128, 4)
embedding_kernel(
    const T* __restrict__ input_ids,
    const W* __restrict__ tok_embed,
    const W* __restrict__ pos_embed,
    T* __restrict__ out,
    int B, int S, int D, int V
) {
    const int bs = blockIdx.x;
    if (bs >= B * S) return;
    const int s = bs % S;
    const int tid = threadIdx.x;
    int tok_id = static_cast<int>(to_float<T>(input_ids[bs]));
    if (tok_id < 0) tok_id = 0;
    if (tok_id >= V) tok_id = V - 1;
    for (int d = tid; d < D; d += blockDim.x) {
        float t = to_float<W>(tok_embed[(size_t)tok_id * D + d]);
        float p = to_float<W>(pos_embed[(size_t)s * D + d]);
        out[(size_t)bs * D + d] = from_float<T>(t + p);
    }
}

template <typename T, typename W>
__global__ void __launch_bounds__(128, 4)
embedding_bwd_kernel(
    const T* __restrict__ input_ids,
    const T* __restrict__ grad_embed,
    W* __restrict__ grad_tok_embed,
    W* __restrict__ grad_pos_embed,
    int B, int S, int D, int V
) {
    const int bs = blockIdx.x;
    if (bs >= B * S) return;
    const int s = bs % S;
    int tok_id = static_cast<int>(to_float<T>(input_ids[bs]));
    if (tok_id < 0) tok_id = 0;
    if (tok_id >= V) tok_id = V - 1;
    const int tid = threadIdx.x;
    for (int d = tid; d < D; d += blockDim.x) {
        float g = to_float<T>(grad_embed[(size_t)bs * D + d]);
        atomic_add_w(&grad_pos_embed[(size_t)s * D + d], g);
        atomic_add_w(&grad_tok_embed[(size_t)tok_id * D + d], g);
    }
}

// ─── Fused residual + LayerNorm ──────────────────────────────────────
template <typename T, typename W>
__global__ void __launch_bounds__(128, 4)
residual_layernorm_kernel(
    const T* __restrict__ x, const T* __restrict__ residual,
    const W* __restrict__ gain, const W* __restrict__ bias,
    T* __restrict__ sum_out, T* __restrict__ out,
    int N, int D, float eps
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int row_off = row * D;

    extern __shared__ float smem[];
    float* sum_buf = smem;

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = to_float<T>(x[row_off + d]) + to_float<T>(residual[row_off + d]);
        sum_buf[d] = v;
        sum_out[row_off + d] = from_float<T>(v);
        local_sum += v;
    }
    __shared__ float reduce_buf[32];
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;
    local_sum = warp_reduce_sum(local_sum, WARP_SIZE, lane);
    if (lane == 0) reduce_buf[warp] = local_sum;
    __syncthreads();
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    float total_sum = (tid < n_warps) ? reduce_buf[tid] : 0.0f;
    total_sum = warp_reduce_sum(total_sum, n_warps, tid);
    if (tid == 0) reduce_buf[0] = total_sum;
    __syncthreads();
    float mean = reduce_buf[0] / (float)D;

    float local_var = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = sum_buf[d] - mean;
        local_var += v * v;
    }
    local_var = warp_reduce_sum(local_var, WARP_SIZE, lane);
    if (lane == 0) reduce_buf[warp] = local_var;
    __syncthreads();
    float total_var = (tid < n_warps) ? reduce_buf[tid] : 0.0f;
    total_var = warp_reduce_sum(total_var, n_warps, tid);
    if (tid == 0) reduce_buf[0] = total_var;
    __syncthreads();
    float inv_std = fast_rsqrt_nr(reduce_buf[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float g = to_float<W>(gain[d]);
        float b = to_float<W>(bias[d]);
        float v = (sum_buf[d] - mean) * inv_std;
        out[row_off + d] = from_float<T>(v * g + b);
    }
}

template <typename T, typename W>
__global__ void __launch_bounds__(128, 4)
layernorm_kernel(
    const T* __restrict__ x,
    const W* __restrict__ gain, const W* __restrict__ bias,
    T* __restrict__ out,
    int N, int D, float eps
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int row_off = row * D;
    extern __shared__ float smem[];
    float* buf = smem;

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = to_float<T>(x[row_off + d]);
        buf[d] = v;
        local_sum += v;
    }
    __shared__ float reduce_buf[32];
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;
    local_sum = warp_reduce_sum(local_sum, WARP_SIZE, lane);
    if (lane == 0) reduce_buf[warp] = local_sum;
    __syncthreads();
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    float total_sum = (tid < n_warps) ? reduce_buf[tid] : 0.0f;
    total_sum = warp_reduce_sum(total_sum, n_warps, tid);
    if (tid == 0) reduce_buf[0] = total_sum;
    __syncthreads();
    float mean = reduce_buf[0] / (float)D;

    float local_var = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = buf[d] - mean;
        local_var += v * v;
    }
    local_var = warp_reduce_sum(local_var, WARP_SIZE, lane);
    if (lane == 0) reduce_buf[warp] = local_var;
    __syncthreads();
    float total_var = (tid < n_warps) ? reduce_buf[tid] : 0.0f;
    total_var = warp_reduce_sum(total_var, n_warps, tid);
    if (tid == 0) reduce_buf[0] = total_var;
    __syncthreads();
    float inv_std = fast_rsqrt_nr(reduce_buf[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float g = to_float<W>(gain[d]);
        float b = to_float<W>(bias[d]);
        float v = (buf[d] - mean) * inv_std;
        out[row_off + d] = from_float<T>(v * g + b);
    }
}

// LayerNorm backward
template <typename T, typename W>
__global__ void __launch_bounds__(128, 4)
layernorm_bwd_kernel(
    const T* __restrict__ x, const T* __restrict__ grad_out,
    const W* __restrict__ gain,
    T* __restrict__ grad_in, W* __restrict__ grad_gain, W* __restrict__ grad_bias,
    int N, int D, float eps
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int row_off = row * D;
    extern __shared__ float smem[];
    float* x_hat = smem;

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x)
        local_sum += to_float<T>(x[row_off + d]);
    __shared__ float reduce_buf[32];
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    local_sum = warp_reduce_sum(local_sum, WARP_SIZE, lane);
    if (lane == 0) reduce_buf[warp] = local_sum;
    __syncthreads();
    float total_sum = (tid < n_warps) ? reduce_buf[tid] : 0.0f;
    total_sum = warp_reduce_sum(total_sum, n_warps, tid);
    if (tid == 0) reduce_buf[0] = total_sum;
    __syncthreads();
    float mean = reduce_buf[0] / (float)D;

    float local_var = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = to_float<T>(x[row_off + d]) - mean;
        local_var += v * v;
    }
    local_var = warp_reduce_sum(local_var, WARP_SIZE, lane);
    if (lane == 0) reduce_buf[warp] = local_var;
    __syncthreads();
    float total_var = (tid < n_warps) ? reduce_buf[tid] : 0.0f;
    total_var = warp_reduce_sum(total_var, n_warps, tid);
    if (tid == 0) reduce_buf[0] = total_var;
    __syncthreads();
    float inv_std = fast_rsqrt_nr(reduce_buf[0] / (float)D + eps);

    float sum_gy = 0.0f, sum_gy_xhat = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float xv = to_float<T>(x[row_off + d]);
        float xh = (xv - mean) * inv_std;
        x_hat[d] = xh;
        float g = to_float<W>(gain[d]);
        float gy = to_float<T>(grad_out[row_off + d]) * g;
        sum_gy += gy;
        sum_gy_xhat += gy * xh;
    }
    sum_gy = warp_reduce_sum(sum_gy, WARP_SIZE, lane);
    if (lane == 0) reduce_buf[warp] = sum_gy;
    __syncthreads();
    float total_gy = (tid < n_warps) ? reduce_buf[tid] : 0.0f;
    total_gy = warp_reduce_sum(total_gy, n_warps, tid);
    if (tid == 0) reduce_buf[0] = total_gy;
    __syncthreads();
    total_gy = reduce_buf[0];

    sum_gy_xhat = warp_reduce_sum(sum_gy_xhat, WARP_SIZE, lane);
    if (lane == 0) reduce_buf[warp] = sum_gy_xhat;
    __syncthreads();
    float total_gy_xhat = (tid < n_warps) ? reduce_buf[tid] : 0.0f;
    total_gy_xhat = warp_reduce_sum(total_gy_xhat, n_warps, tid);
    if (tid == 0) reduce_buf[0] = total_gy_xhat;
    __syncthreads();
    total_gy_xhat = reduce_buf[0];

    float invD = 1.0f / (float)D;
    for (int d = tid; d < D; d += blockDim.x) {
        float g = to_float<W>(gain[d]);
        float gy = to_float<T>(grad_out[row_off + d]) * g;
        float xh = x_hat[d];
        float dx = inv_std * (gy - invD * total_gy - xh * invD * total_gy_xhat);
        grad_in[row_off + d] = from_float<T>(dx);
        float go = to_float<T>(grad_out[row_off + d]);
        atomic_add_w(&grad_gain[d], go * xh);
        atomic_add_w(&grad_bias[d], go);
    }
}

// ─── GELU (tanh approximation) ───────────────────────────────────────
__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float t = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + ptx_tanhf(t));
}
__device__ __forceinline__ float gelu_tanh_grad(float x) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float x2 = x * x;
    float t = k0 * (x + k1 * x * x2);
    float th = ptx_tanhf(t);
    float sech2 = 1.0f - th * th;
    float dt = k0 * (1.0f + 3.0f * k1 * x2);
    return 0.5f * (1.0f + th) + 0.5f * x * sech2 * dt;
}

template <typename T>
__global__ void __launch_bounds__(256, 4)
gelu_fwd_kernel(const T* __restrict__ pre, T* __restrict__ post, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    post[i] = from_float<T>(gelu_tanh(to_float<T>(pre[i])));
}
template <typename T>
__global__ void __launch_bounds__(256, 4)
gelu_bwd_kernel(const T* __restrict__ pre, const T* __restrict__ grad_post,
                T* __restrict__ grad_pre, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gp = to_float<T>(grad_post[i]);
    float dx = gp * gelu_tanh_grad(to_float<T>(pre[i]));
    grad_pre[i] = from_float<T>(dx);
}

// ─── Bias add + bias bwd ─────────────────────────────────────────────
template <typename T, typename W>
__global__ void __launch_bounds__(256, 4)
bias_add_kernel(T* __restrict__ x, const W* __restrict__ bias, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    for (int n = tid; n < N; n += blockDim.x)
        x[(size_t)row * N + n] = from_float<T>(
            to_float<T>(x[(size_t)row * N + n]) + to_float<W>(bias[n]));
}
template <typename T, typename W>
__global__ void __launch_bounds__(256, 4)
bias_bwd_kernel(const T* __restrict__ grad_out, W* __restrict__ grad_bias,
                int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.0f;
    for (int m = 0; m < M; m++) acc += to_float<T>(grad_out[(size_t)m * N + n]);
    atomic_add_w(&grad_bias[n], acc);
}

// Elementwise add: dst += src
template <typename T>
__global__ void __launch_bounds__(256, 4)
add_inplace_kernel(T* __restrict__ dst, const T* __restrict__ src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = from_float<T>(to_float<T>(dst[i]) + to_float<T>(src[i]));
}

// ─── QKV split: [B, S, 3, H, d] -> q/k/v in [B, H, S, d] ─────────────
template <typename T>
__global__ void __launch_bounds__(128, 4)
qkv_split_kernel(
    const T* __restrict__ qkv,
    T* __restrict__ q, T* __restrict__ k, T* __restrict__ v,
    int B, int S, int H, int d
) {
    int bs = blockIdx.x;
    if (bs >= B * S) return;
    int b = bs / S, s = bs % S;
    int tid = threadIdx.x;
    int total = 3 * H * d;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int qkv_idx = idx / (H * d);
        int rem = idx % (H * d);
        int h = rem / d, dd = rem % d;
        T val = qkv[((size_t)bs * 3 + qkv_idx) * H * d + h * d + dd];
        size_t out_off = ((size_t)b * H + h) * S * d + (size_t)s * d + dd;
        if (qkv_idx == 0) q[out_off] = val;
        else if (qkv_idx == 1) k[out_off] = val;
        else v[out_off] = val;
    }
}

template <typename T>
__global__ void __launch_bounds__(128, 4)
qkv_merge_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ qkv,
    int B, int S, int H, int d
) {
    int bs = blockIdx.x;
    if (bs >= B * S) return;
    int b = bs / S, s = bs % S;
    int tid = threadIdx.x;
    int total = 3 * H * d;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int qkv_idx = idx / (H * d);
        int rem = idx % (H * d);
        int h = rem / d, dd = rem % d;
        size_t in_off = ((size_t)b * H + h) * S * d + (size_t)s * d + dd;
        T val = (qkv_idx == 0) ? q[in_off]
              : (qkv_idx == 1) ? k[in_off]
                                : v[in_off];
        qkv[((size_t)bs * 3 + qkv_idx) * H * d + h * d + dd] = val;
    }
}

// [B, H, S, d] -> [B, S, H*d]
template <typename T>
__global__ void __launch_bounds__(128, 4)
attn_out_reshape_kernel(const T* __restrict__ in, T* __restrict__ out,
                        int B, int S, int H, int d) {
    int bs = blockIdx.x;
    if (bs >= B * S) return;
    int b = bs / S, s = bs % S;
    int tid = threadIdx.x;
    int D = H * d;
    for (int dd = tid; dd < D; dd += blockDim.x) {
        int h = dd / d, ddi = dd % d;
        size_t in_off = ((size_t)b * H + h) * S * d + (size_t)s * d + ddi;
        out[(size_t)bs * D + dd] = in[in_off];
    }
}
template <typename T>
__global__ void __launch_bounds__(128, 4)
attn_out_inverse_reshape_kernel(const T* __restrict__ in, T* __restrict__ out,
                                int B, int S, int H, int d) {
    int bs = blockIdx.x;
    if (bs >= B * S) return;
    int b = bs / S, s = bs % S;
    int tid = threadIdx.x;
    int D = H * d;
    for (int dd = tid; dd < D; dd += blockDim.x) {
        int h = dd / d, ddi = dd % d;
        size_t out_off = ((size_t)b * H + h) * S * d + (size_t)s * d + ddi;
        out[out_off] = in[(size_t)bs * D + dd];
    }
}

// ─── Last-token extract / scatter ────────────────────────────────────
template <typename T>
__global__ void __launch_bounds__(128, 4)
last_token_kernel(const T* __restrict__ full, T* __restrict__ out,
                  int B, int S, int V) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    for (int v = tid; v < V; v += blockDim.x)
        out[(size_t)b * V + v] = full[((size_t)b * S + (S - 1)) * V + v];
}
template <typename T>
__global__ void __launch_bounds__(128, 4)
last_token_scatter_kernel(const T* __restrict__ grad_last, T* __restrict__ grad_full,
                          int B, int S, int V) {
    int bs = blockIdx.x;
    if (bs >= B * S) return;
    int b = bs / S, s = bs % S;
    int tid = threadIdx.x;
    bool is_last = (s == S - 1);
    for (int v = tid; v < V; v += blockDim.x) {
        T val = is_last ? grad_last[(size_t)b * V + v] : from_float<T>(0.0f);
        grad_full[(size_t)bs * V + v] = val;
    }
}

// ─── cuBLAS GEMM (row-major wrapper) ─────────────────────────────────
// Computes row-major: C[M,N] = alpha * op(A) * op(B) + beta * C.
// Implemented by computing C^T in column-major:
//   cublas: opB',opA' on (N, M, K) with B as first matrix, A as second.
template <typename T>
inline cublasStatus_t cublas_gemm_rm(
    cublasHandle_t handle,
    cublasOperation_t opA, cublasOperation_t opB,
    int M, int N, int K,
    float alpha, float beta,
    const T* A, int lda,
    const T* B, int ldb,
    T* C, int ldc,
    cudaStream_t stream
) {
    cublasSetStream(handle, stream);
    return cublasGemmEx(handle,
        opB, opA,
        N, M, K,
        &alpha,
        B, CublasTraits<T>::data_type, ldb,
        A, CublasTraits<T>::data_type, lda,
        &beta,
        C, CublasTraits<T>::data_type, ldc,
        CublasTraits<T>::compute_type,
        CUBLAS_GEMM_DEFAULT);
}

// ─── Forward orchestration ───────────────────────────────────────────
template <typename ActT, typename WeightT>
cudaError_t forward(
    const ActT* input,
    const WeightT* weights,
    ActT* output,
    ActT* activations,
    int batch, int seq_len, int d_model, int n_heads, int d_head,
    int n_layers, int vocab_size, int ffn_expansion,
    cudaStream_t stream
) {
    const int B = batch, S = seq_len, D = d_model, H = n_heads, V = vocab_size;
    const int FH = ffn_expansion * D;
    const float scale = 1.0f / sqrtf((float)d_head);
    const float eps = 1e-5f;
    const size_t per_layer_w = per_layer_weight_count(D, FH);
    auto L = compute_layout<ActT>(B, S, D, H, V, ffn_expansion);

    const WeightT* wp = weights;
    const WeightT* tok_embed = wp; wp += (size_t)V * D;
    const WeightT* pos_embed = wp; wp += (size_t)S * D;
    const WeightT* layers_w = wp; wp += (size_t)n_layers * per_layer_w;
    const WeightT* final_g = wp; wp += D;
    const WeightT* final_b = wp; wp += D;
    const WeightT* vocab_W = wp; wp += (size_t)V * D;
    const WeightT* vocab_b = wp; wp += V;

    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    // Stash input ids for backward
    ActT* input_ids_saved = activations;
    cudaMemcpyAsync(input_ids_saved, input, (size_t)B*S*sizeof(ActT),
                    cudaMemcpyDeviceToDevice, stream);

    // Embedding lookup
    ActT* embed_out = activations + L.input_ids_saved;
    embedding_kernel<ActT, WeightT><<<B*S, 128, 0, stream>>>(
        input, tok_embed, pos_embed, embed_out, B, S, D, V);

    ActT* layer_input = embed_out;

    // Layer stack
    for (int li = 0; li < n_layers; li++) {
        auto sl = slice_layer<ActT>(activations, L, li);
        auto lw = slice_layer_weights<WeightT>(layers_w + (size_t)li * per_layer_w, D, FH);

        // Save input to QKV (== prev layer output)
        cudaMemcpyAsync(sl.qkv_in, layer_input, (size_t)B*S*D*sizeof(ActT),
                        cudaMemcpyDeviceToDevice, stream);

        // QKV projection: [B*S, D] * [D, 3D]^T -> [B*S, 3D]
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            B*S, 3*D, D, 1.0f, 0.0f,
            (const WeightT*)sl.qkv_in, D, lw.qkv_W, D,
            (WeightT*)sl.qkv_out, 3*D, stream);
        bias_add_kernel<ActT, WeightT><<<B*S, 256, 0, stream>>>(
            sl.qkv_out, lw.qkv_b, B*S, 3*D);

        // Split into Q/K/V in [B, H, S, d_h] layout (3 contiguous slabs)
        ActT* q_buf = sl.attn_in_qkv;
        ActT* k_buf = sl.attn_in_qkv + (size_t)B*S*D;
        ActT* v_buf = sl.attn_in_qkv + (size_t)2*B*S*D;
        qkv_split_kernel<ActT><<<B*S, 128, 0, stream>>>(
            sl.qkv_out, q_buf, k_buf, v_buf, B, S, H, d_head);

        // Attention forward
        attention::attention_forward<ActT, 32, /*kCausal=*/true>(
            q_buf, k_buf, v_buf,
            sl.attn_out_perhead, sl.softmax_lse,
            B, H, S, scale, stream);

        // Reshape attention output [B, H, S, d] -> [B, S, D]
        attn_out_reshape_kernel<ActT><<<B*S, 128, 0, stream>>>(
            sl.attn_out_perhead, sl.attn_out, B, S, H, d_head);

        // Output projection
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            B*S, D, D, 1.0f, 0.0f,
            (const WeightT*)sl.attn_out, D, lw.out_W, D,
            (WeightT*)sl.attn_proj, D, stream);
        bias_add_kernel<ActT, WeightT><<<B*S, 128, 0, stream>>>(
            sl.attn_proj, lw.out_b, B*S, D);

        // n1: x = LN(qkv_in + attn_proj) [post-norm]
        residual_layernorm_kernel<ActT, WeightT>
            <<<B*S, 128, D*sizeof(float), stream>>>(
                sl.qkv_in, sl.attn_proj, lw.n1_g, lw.n1_b,
                sl.n1_in, sl.layer1_out, B*S, D, eps);

        // FFN up
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            B*S, FH, D, 1.0f, 0.0f,
            (const WeightT*)sl.layer1_out, D, lw.ff1_W, D,
            (WeightT*)sl.ffn_pre, FH, stream);
        bias_add_kernel<ActT, WeightT><<<B*S, 256, 0, stream>>>(
            sl.ffn_pre, lw.ff1_b, B*S, FH);

        // GELU
        {
            size_t n = (size_t)B*S*FH;
            int block = 256, grid = (int)((n + block - 1) / block);
            gelu_fwd_kernel<ActT><<<grid, block, 0, stream>>>(
                sl.ffn_pre, sl.ffn_post, n);
        }

        // FFN down
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            B*S, D, FH, 1.0f, 0.0f,
            (const WeightT*)sl.ffn_post, FH, lw.ff2_W, FH,
            (WeightT*)sl.ffn_out, D, stream);
        bias_add_kernel<ActT, WeightT><<<B*S, 128, 0, stream>>>(
            sl.ffn_out, lw.ff2_b, B*S, D);

        // n2: layer_out = LN(layer1_out + ffn_out) [post-norm]
        residual_layernorm_kernel<ActT, WeightT>
            <<<B*S, 128, D*sizeof(float), stream>>>(
                sl.layer1_out, sl.ffn_out, lw.n2_g, lw.n2_b,
                sl.n2_in, sl.layer_out, B*S, D, eps);

        layer_input = sl.layer_out;
    }

    // Final LayerNorm
    ActT* final_norm_out = activations + L.input_ids_saved + L.embed_out + (size_t)n_layers * L.per_layer_total;
    layernorm_kernel<ActT, WeightT>
        <<<B*S, 128, D*sizeof(float), stream>>>(
            layer_input, final_g, final_b, final_norm_out, B*S, D, eps);

    // Vocab head (full): logits[B*S, V] = norm * vocab_W^T + b
    ActT* logits_full = final_norm_out + L.final_norm_out;
    cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        B*S, V, D, 1.0f, 0.0f,
        (const WeightT*)final_norm_out, D, vocab_W, D,
        (WeightT*)logits_full, V, stream);
    bias_add_kernel<ActT, WeightT><<<B*S, 128, 0, stream>>>(
        logits_full, vocab_b, B*S, V);

    // Last-token output [B, V]
    last_token_kernel<ActT><<<B, 128, 0, stream>>>(
        logits_full, output, B, S, V);

    cublasDestroy(cublas);
    return cudaGetLastError();
}

// ─── Backward orchestration ──────────────────────────────────────────
//
// Implementation strategy: every weight-grad GEMM is computed BEFORE the
// activation that it consumes is overwritten. Where we need to overwrite
// a saved activation (to run a backward GEMM into it), we issue the
// weight-grad GEMM first.
//
// We use the activation buffer as scratch by reusing slots that are no
// longer needed. Specifically:
//   * grad_layer_out (input grad of n2): held in workspace W1
//   * grad_ffn_out  (= grad_n2_in): held in W1 (it's the output of LN bwd)
//   * grad_ffn_post: held in W2 (activation grad of GEMM2)
//   * grad_ffn_pre:  W2 (after GELU bwd)
//   * grad_layer1_out (residual + GEMM1.A grad): W1 (after add)
//   * grad_n1_in:    W1 (output of LN bwd)
//   * grad_attn_proj: W1
//   * grad_attn_out: W2
//   * grad_attn_perhead: W3 (we need a third buffer for QKV layout)
//   * grad_q/grad_k/grad_v: W1, W2, W3 (overwriting attn_in_qkv slabs)
//   * grad_qkv_out: W4 (or reuse qkv_out — we overwrite it last)
//   * grad_qkv_in: W1 (final output for layer)
//
// We allocate W1/W2/W3 by overwriting saved layer activations in the
// reverse order they're consumed. The key invariants:
//   - sl.qkv_in must be preserved until grad_qkv_W is computed.
//   - sl.layer1_out must be preserved until grad_ff1_W is computed.
//   - sl.ffn_post must be preserved until grad_ff2_W is computed.
//   - sl.attn_out must be preserved until grad_out_W is computed.
//   - sl.attn_in_qkv (q,k,v) must be preserved until attention_backward.
//   - sl.qkv_out is no longer needed once Q,K,V are split (we rebuild for
//     forward consistency, but for backward we overwrite as grad_qkv_out).
//
template <typename ActT, typename WeightT>
cudaError_t backward(
    const ActT* grad_output,
    const ActT* activations_saved,
    const WeightT* weights,
    ActT* grad_input,
    WeightT* grad_weights,
    int batch, int seq_len, int d_model, int n_heads, int d_head,
    int n_layers, int vocab_size, int ffn_expansion,
    cudaStream_t stream
) {
    const int B = batch, S = seq_len, D = d_model, H = n_heads, V = vocab_size;
    const int FH = ffn_expansion * D;
    const float scale = 1.0f / sqrtf((float)d_head);
    const float eps = 1e-5f;
    const size_t per_layer_w = per_layer_weight_count(D, FH);
    auto L = compute_layout<ActT>(B, S, D, H, V, ffn_expansion);

    const WeightT* wp = weights;
    const WeightT* tok_embed = wp; wp += (size_t)V * D;
    const WeightT* pos_embed = wp; wp += (size_t)S * D;
    const WeightT* layers_w = wp; wp += (size_t)n_layers * per_layer_w;
    const WeightT* final_g = wp; wp += D;
    /* final_b */ wp += D;
    const WeightT* vocab_W = wp; wp += (size_t)V * D;
    /* vocab_b */ wp += V;
    (void)tok_embed; (void)pos_embed;

    WeightT* gwp = grad_weights;
    WeightT* g_tok_embed = gwp; gwp += (size_t)V * D;
    WeightT* g_pos_embed = gwp; gwp += (size_t)S * D;
    WeightT* g_layers = gwp; gwp += (size_t)n_layers * per_layer_w;
    WeightT* g_final_g = gwp; gwp += D;
    WeightT* g_final_b = gwp; gwp += D;
    WeightT* g_vocab_W = gwp; gwp += (size_t)V * D;
    WeightT* g_vocab_b = gwp; gwp += V;

    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    ActT* act_base = const_cast<ActT*>(activations_saved);
    const ActT* input_ids_saved = act_base;
    ActT* final_norm_out = act_base + L.input_ids_saved + L.embed_out + (size_t)n_layers * L.per_layer_total;
    ActT* logits_full = final_norm_out + L.final_norm_out;

    // 1. Scatter grad_output into grad_logits_full (overwrites logits_full)
    last_token_scatter_kernel<ActT><<<B*S, 128, 0, stream>>>(
        grad_output, logits_full, B, S, V);

    // 2. Vocab head bwd. Compute weight grad FIRST (needs saved final_norm_out).
    // grad_vocab_W [V, D] += grad_logits^T [V, B*S] * final_norm_out [B*S, D]
    cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        V, D, B*S, 1.0f, 1.0f,
        (const WeightT*)logits_full, V,
        (const WeightT*)final_norm_out, D,
        g_vocab_W, D, stream);
    // grad_vocab_b
    bias_bwd_kernel<ActT, WeightT><<<(V+255)/256, 256, 0, stream>>>(
        logits_full, g_vocab_b, B*S, V);
    // grad_norm_out (= grad of input to vocab head) = grad_logits * vocab_W
    // Overwrite final_norm_out — no longer needed.
    cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        B*S, D, V, 1.0f, 0.0f,
        (const WeightT*)logits_full, V, vocab_W, D,
        (WeightT*)final_norm_out, D, stream);

    // 3. Final LayerNorm bwd. Input was last layer's layer_out.
    auto sl_last = slice_layer<ActT>(act_base, L, n_layers - 1);
    ActT* grad_stack_out = logits_full;  // reuse — sized [B*S*V] which is >> [B*S*D]
    layernorm_bwd_kernel<ActT, WeightT>
        <<<B*S, 128, D*sizeof(float), stream>>>(
            sl_last.layer_out, final_norm_out, final_g,
            grad_stack_out, g_final_g, g_final_b, B*S, D, eps);

    ActT* grad_y = grad_stack_out;  // grad w.r.t. last layer's output

    // 4. Per-layer backward (reverse order)
    for (int li = n_layers - 1; li >= 0; li--) {
        auto sl = slice_layer<ActT>(act_base, L, li);
        auto lw = slice_layer_weights<WeightT>(layers_w + (size_t)li * per_layer_w, D, FH);
        auto gw = slice_layer_grads<WeightT>(g_layers + (size_t)li * per_layer_w, D, FH);

        // ── n2 backward
        // input was n2_in = layer1_out + ffn_out. Grad branches both.
        // Write grad_n2_in into ffn_out (no longer needed).
        layernorm_bwd_kernel<ActT, WeightT>
            <<<B*S, 128, D*sizeof(float), stream>>>(
                sl.n2_in, grad_y, lw.n2_g,
                sl.ffn_out, gw.n2_g, gw.n2_b, B*S, D, eps);
        ActT* grad_n2_in = sl.ffn_out;     // alias (grad for both branches)

        // ── FFN-down bwd: y = ffn_post * ff2_W^T + ff2_b
        // First: weight grad uses saved ffn_post.
        // grad_ff2_W [D, FH] += grad_y^T * ffn_post
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D, FH, B*S, 1.0f, 1.0f,
            (const WeightT*)grad_n2_in, D,
            (const WeightT*)sl.ffn_post, FH,
            gw.ff2_W, FH, stream);
        bias_bwd_kernel<ActT, WeightT><<<(D+255)/256, 256, 0, stream>>>(
            grad_n2_in, gw.ff2_b, B*S, D);
        // Then: activation grad. Overwrite ffn_post (no longer needed).
        // grad_ffn_post = grad_n2_in * ff2_W
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            B*S, FH, D, 1.0f, 0.0f,
            (const WeightT*)grad_n2_in, D, lw.ff2_W, FH,
            (WeightT*)sl.ffn_post, FH, stream);
        ActT* grad_ffn_post = sl.ffn_post;

        // ── GELU bwd: in-place into ffn_pre (no longer needed after this)
        {
            size_t n = (size_t)B*S*FH;
            int block = 256, grid = (int)((n + block - 1) / block);
            gelu_bwd_kernel<ActT><<<grid, block, 0, stream>>>(
                sl.ffn_pre, grad_ffn_post, sl.ffn_pre, n);
        }
        ActT* grad_ffn_pre = sl.ffn_pre;

        // ── FFN-up bwd: y = layer1_out * ff1_W^T + ff1_b
        // Weight grad uses saved layer1_out.
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            FH, D, B*S, 1.0f, 1.0f,
            (const WeightT*)grad_ffn_pre, FH,
            (const WeightT*)sl.layer1_out, D,
            gw.ff1_W, D, stream);
        bias_bwd_kernel<ActT, WeightT><<<(FH+255)/256, 256, 0, stream>>>(
            grad_ffn_pre, gw.ff1_b, B*S, FH);
        // Activation grad: grad_layer1_ff = grad_ffn_pre * ff1_W
        // Overwrite layer1_out (no longer needed after weight grad).
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            B*S, D, FH, 1.0f, 0.0f,
            (const WeightT*)grad_ffn_pre, FH, lw.ff1_W, D,
            (WeightT*)sl.layer1_out, D, stream);
        ActT* grad_layer1_ff = sl.layer1_out;

        // grad_layer1_out total = grad_n2_in (residual branch) + grad_layer1_ff
        {
            size_t n = (size_t)B*S*D;
            int block = 256, grid = (int)((n + block - 1) / block);
            add_inplace_kernel<ActT><<<grid, block, 0, stream>>>(
                grad_layer1_ff, grad_n2_in, n);
        }
        ActT* grad_layer1_out = grad_layer1_ff;

        // ── n1 bwd. Input was n1_in = qkv_in + attn_proj. Grad branches both.
        // Write grad_n1_in into attn_proj (no longer needed).
        layernorm_bwd_kernel<ActT, WeightT>
            <<<B*S, 128, D*sizeof(float), stream>>>(
                sl.n1_in, grad_layer1_out, lw.n1_g,
                sl.attn_proj, gw.n1_g, gw.n1_b, B*S, D, eps);
        ActT* grad_n1_in = sl.attn_proj;

        // ── out projection bwd: y = attn_out * out_W^T + out_b
        // Weight grad uses saved attn_out.
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D, D, B*S, 1.0f, 1.0f,
            (const WeightT*)grad_n1_in, D,
            (const WeightT*)sl.attn_out, D,
            gw.out_W, D, stream);
        bias_bwd_kernel<ActT, WeightT><<<(D+255)/256, 256, 0, stream>>>(
            grad_n1_in, gw.out_b, B*S, D);
        // Activation grad. Overwrite attn_out.
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            B*S, D, D, 1.0f, 0.0f,
            (const WeightT*)grad_n1_in, D, lw.out_W, D,
            (WeightT*)sl.attn_out, D, stream);
        ActT* grad_attn_out = sl.attn_out;

        // Reshape [B, S, D] -> [B, H, S, d] for attention_backward
        attn_out_inverse_reshape_kernel<ActT><<<B*S, 128, 0, stream>>>(
            grad_attn_out, sl.attn_out_perhead, B, S, H, d_head);
        // After this, attn_out_perhead holds grad in [B,H,S,d] layout.

        // ── Attention bwd
        // q/k/v are stored in attn_in_qkv slabs. We allocate separate
        // grad buffers (NOT aliased with q/k/v — the bwd kernel reads
        // v multiple times).
        ActT* q_buf = sl.attn_in_qkv;
        ActT* k_buf = sl.attn_in_qkv + (size_t)B*S*D;
        ActT* v_buf = sl.attn_in_qkv + (size_t)2*B*S*D;
        // grad_q/k/v borrow ffn_pre/ffn_post (each [B,S,4D] >> [B,S,D]).
        ActT* grad_q_buf = sl.ffn_pre;
        ActT* grad_k_buf = sl.ffn_pre + (size_t)B*S*D;
        ActT* grad_v_buf = sl.ffn_post;
        // attention_backward consumes q,k,v + softmax_lse and produces grad_q/k/v.
        // The 'out' arg is unused by the SMEM kernel — pass nullptr-equivalent
        // (we pass q_buf for type-correctness; it's only touched by FA3 path).
        attention::attention_backward<ActT, 32, /*kCausal=*/true>(
            sl.attn_out_perhead,
            q_buf, k_buf, v_buf,
            q_buf,            // 'out' placeholder (unused by SMEM bwd)
            sl.softmax_lse,
            grad_q_buf, grad_k_buf, grad_v_buf,
            B, H, S, scale, stream);

        // Merge grad_q/k/v back into a packed [B, S, 3, H, d] buffer.
        // Reuse qkv_out as the merged grad buffer.
        qkv_merge_kernel<ActT><<<B*S, 128, 0, stream>>>(
            grad_q_buf, grad_k_buf, grad_v_buf, sl.qkv_out, B, S, H, d_head);
        ActT* grad_qkv_out = sl.qkv_out;

        // ── QKV projection bwd: y = qkv_in * qkv_W^T + qkv_b
        // Weight grad uses saved qkv_in.
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            3*D, D, B*S, 1.0f, 1.0f,
            (const WeightT*)grad_qkv_out, 3*D,
            (const WeightT*)sl.qkv_in, D,
            gw.qkv_W, D, stream);
        bias_bwd_kernel<ActT, WeightT><<<(3*D+255)/256, 256, 0, stream>>>(
            grad_qkv_out, gw.qkv_b, B*S, 3*D);
        // Activation grad: grad_qkv_in = grad_qkv_out * qkv_W
        // Write into qkv_in (no longer needed).
        cublas_gemm_rm<WeightT>(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            B*S, D, 3*D, 1.0f, 0.0f,
            (const WeightT*)grad_qkv_out, 3*D, lw.qkv_W, D,
            (WeightT*)sl.qkv_in, D, stream);
        ActT* grad_qkv_in = sl.qkv_in;

        // grad_layer_input = grad_n1_in (residual) + grad_qkv_in
        {
            size_t n = (size_t)B*S*D;
            int block = 256, grid = (int)((n + block - 1) / block);
            add_inplace_kernel<ActT><<<grid, block, 0, stream>>>(
                grad_qkv_in, grad_n1_in, n);
        }
        grad_y = grad_qkv_in;
    }

    // 5. Embedding bwd: grad_y is grad of embed_out. The forward pass
    //    stashed the input ids at activations_saved[0..B*S).
    embedding_bwd_kernel<ActT, WeightT><<<B*S, 128, 0, stream>>>(
        input_ids_saved, grad_y, g_tok_embed, g_pos_embed, B, S, D, V);

    // grad_input is grad w.r.t. token IDs — undefined for integer inputs.
    cudaMemsetAsync(grad_input, 0, (size_t)B*S*sizeof(ActT), stream);

    cublasDestroy(cublas);
    return cudaGetLastError();
}

}}}}  // namespace sg::sm90::models::decoder
