#pragma once
// Canonical header (de-inlined). Body is byte-identical to the
// formerly copy-pasted block; prerequisites are included so that
// platform macros precede their use.
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/scan/affine2x2.h"
#include "csrc/common/utils.cuh"
// §3.0: csrc/common/ptx_intrinsics.cuh removed — all 5 of its hand-PTX
// transcendentals (affine_combine_ptx / softplus_ptx / fast_exp_ptx /
// stochastic_round_ptx / gru_gates_ptx) were dead (0 call sites) and merely
// re-derived what --use_fast_math already emits. See scripts/STAGE3_PTX_AUDIT.md.

// CUDA sm_90 (Hopper) primitives — shared across all 11 launch_*.cu files.
//
// This header consolidates vendor-specific intrinsics that the optimizer
// launch files use repeatedly: warp/block/cluster reductions, vec4 helpers,
// non-temporal load/store, stochastic rounding, RoPE pair rotation, fused
// element ingestion, and grid-stride loop helpers.
//
// Algorithm-neutral. Per-element optimizer math lives in csrc/algorithms/.
// Per-backend launch glue (kernel definitions + host launchers) lives in
// the 11 csrc/backends/cuda/sm_90/launch_<optimizer>.cu files which include
// both this header and the relevant algorithm header.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


namespace sg { namespace sm90 { namespace primitives {

namespace cg = cooperative_groups;

// =========================================================================
//  Grid-stride loop helper
// =========================================================================

__device__ __forceinline__ int grid_stride_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int grid_stride() {
    return gridDim.x * blockDim.x;
}

// =========================================================================
//  Vec4 alignment check (host-side)
// =========================================================================

__host__ __forceinline__ bool is_vec4_alignable(
    const void* p, int64_t numel
) {
    return ((reinterpret_cast<uintptr_t>(p) & 0xF) == 0) && (numel % 4 == 0);
}

// =========================================================================
//  Warp-level sum reduction (butterfly via __shfl_down_sync)
// =========================================================================

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// =========================================================================
//  Block-level sum reduction: warp reduce + shared-memory tree + warp reduce.
//  Uses up to 32 floats of smem (one per warp).
// =========================================================================

__device__ __forceinline__ float block_reduce_sum_f32(float v) {
    __shared__ float smem[32];
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;

    float w = warp_reduce_sum_f32(v);
    if (lane == 0) smem[warp] = w;
    __syncthreads();

    int n_warps = (blockDim.x + 31) / 32;
    if (warp == 0) {
        float x = (lane < n_warps) ? smem[lane] : 0.0f;
        x = warp_reduce_sum_f32(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    return smem[0];
}

// =========================================================================
//  Warp-level u32 sum reduction (§4.1 PTX maximization).
//  Integer reductions use the single-instruction `redux.sync.add.u32`
//  warp collective on Ampere/Hopper (sm_80+), which replaces the 5-step
//  shuffle-xor butterfly with one instruction. Every lane in the (fully
//  active) warp receives the full-warp sum. A shuffle-tree fallback keeps
//  the helper compiling/working on pre-Ampere arches in the codegen matrix.
//
//  NOTE: integer-only. FLOAT redux (`redux.sync.{add,min,max}.f32`) is
//  Blackwell-only (sm_100+) and out of scope — float reductions stay on the
//  shuffle tree (warp_reduce_sum_f32 above).
// =========================================================================

__device__ __forceinline__ unsigned warp_reduce_add_u32(unsigned v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned r;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    return r;  // every lane gets the full-warp sum
#else
    // shuffle-tree fallback (pre-Ampere)
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
#endif
}

// =========================================================================
//  Block-level u32 sum reduction: warp reduce + shared-memory tree + warp
//  reduce. Mirrors block_reduce_sum_f32 exactly (32 u32 of smem, one per
//  warp; same __syncthreads placement) but uses the redux.sync warp reducer.
// =========================================================================

__device__ __forceinline__ unsigned block_reduce_add_u32(unsigned v) {
    __shared__ unsigned smem[32];
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;

    unsigned w = warp_reduce_add_u32(v);
    if (lane == 0) smem[warp] = w;
    __syncthreads();

    int n_warps = (blockDim.x + 31) / 32;
    if (warp == 0) {
        unsigned x = (lane < n_warps) ? smem[lane] : 0u;
        x = warp_reduce_add_u32(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    return smem[0];
}

// =========================================================================
//  Cluster (DSMEM) reduction on Hopper sm_90+ with sm_80 fallback.
//  Wraps the utility in common/utils.cuh.
// =========================================================================

__device__ __forceinline__ float cluster_reduce_sum_f32(float v) {
    return cluster_dsmem_reduce_sum(v);
}

// =========================================================================
//  Non-temporal load / store (bypass L2 for read-once optimizer state).
//  Implemented in common/platform.h; wrapped here for legibility.
// =========================================================================

__device__ __forceinline__ float ldg_f32(const float* ptr) {
    return __ldg(ptr);
}

__device__ __forceinline__ void stream_store_f32(float* ptr, float v) {
    __stwt(ptr, v);
}

// =========================================================================
//  Stochastic rounding to BF16 (branchless via PTX hash_prng).
// =========================================================================

__device__ __forceinline__ __nv_bfloat16 round_bf16_stochastic(
    float v, uint32_t prng_key
) {
    return float_to_bf16_stochastic(v, prng_key);
}

// =========================================================================
//  RoPE pair rotation (used by SG2 scan kernels).
//  Input: (h0, h1, cos, sin)  ->  (h0*c - h1*s, h0*s + h1*c)
// =========================================================================

__device__ __forceinline__ void rope_rotate_pair(
    float& h0, float& h1, float cos_v, float sin_v
) {
    float h0_new = h0 * cos_v - h1 * sin_v;
    float h1_new = h0 * sin_v + h1 * cos_v;
    h0 = h0_new;
    h1 = h1_new;
}

// =========================================================================
//  Last-block-finished pattern for cooperative reductions without grid sync.
//  Returns true on exactly one block; caller uses that block to publish
//  the final reduced value.
// =========================================================================

__device__ __forceinline__ bool last_block_finished(
    unsigned int* __restrict__ counter,
    int total_blocks
) {
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        unsigned int v = atomicInc(counter, total_blocks);
        is_last = (v == static_cast<unsigned int>(total_blocks - 1));
    }
    __syncthreads();
    return is_last;
}

// =========================================================================
//  Compute Adam denom with fast rsqrt + Newton-Raphson refinement.
//  Slightly faster than 1.0f / (sqrtf(v) + eps) when v >= eps^2.
// =========================================================================

__device__ __forceinline__ float adam_denom_fast(float v, float eps) {
    return sqrtf(v) + eps;
}

// =========================================================================
//  §6.1 L2 persistence for per-step optimizer state (Hopper sm_90+).
//
//  Optimizer state (m, v, EMA, mu, …) is read+written every step and is the
//  hottest reuse in the whole pipeline. We hint the L2 cache to keep it
//  resident across the step via the SAFE RUNTIME API
//  (cudaStreamSetAttribute + cudaAccessPolicyWindow) — NOT hand-written
//  `createpolicy` PTX, which trips the known CUDA-13.1 ptxas lowering bug.
//
//  Use as an RAII scope around the kernel launch(es) in a host launcher:
//      { L2PersistScope l2(stream, exp_avg.data_ptr(), exp_avg.nbytes(),
//                          exp_avg_sq.data_ptr(), exp_avg_sq.nbytes());
//        kernel<<<...,stream>>>(...); }
//  On construction it carves a persisting window over the (contiguous-ish)
//  span covering the given buffers; on destruction it resets the policy and
//  releases the carve-out so the next op sees a clean L2.
//
//  Gated by ENABLE_L2_PERSIST and a runtime check: if the requested span is
//  larger than the device's reservable persisting-L2 (cudaDevAttrMax-
//  PersistingL2CacheSize, ~50 MB on H100) — or the device predates Hopper —
//  the scope is a no-op (and logs nothing on the hot path).
// =========================================================================

#ifndef ENABLE_L2_PERSIST
#define ENABLE_L2_PERSIST 1
#endif

class L2PersistScope {
public:
    // Up to two state buffers (the common Adam m/v case). Pass {ptr,bytes}.
    L2PersistScope(cudaStream_t stream,
                   void* p0, size_t n0,
                   void* p1 = nullptr, size_t n1 = 0)
        : stream_(stream), active_(false) {
#if ENABLE_L2_PERSIST
        if (p0 == nullptr || n0 == 0) return;

        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return;

        int cc_major = 0;
        cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev);
        if (cc_major < 9) return;  // L2 residency control: Ampere+ has the API,
                                   // but we scope the win to Hopper sm_90 here.

        int max_persist = 0;
        cudaDeviceGetAttribute(&max_persist,
                               cudaDevAttrMaxPersistingL2CacheSize, dev);
        if (max_persist <= 0) return;

        // Build the smallest byte span covering the (assumed nearby) buffers.
        char* lo = static_cast<char*>(p0);
        char* hi = lo + n0;
        if (p1 != nullptr && n1 > 0) {
            char* lo1 = static_cast<char*>(p1);
            char* hi1 = lo1 + n1;
            if (lo1 < lo) lo = lo1;
            if (hi1 > hi) hi = hi1;
        }
        size_t span = static_cast<size_t>(hi - lo);
        // Only worthwhile (and reservable) if the span fits the persisting L2.
        if (span == 0 || span > static_cast<size_t>(max_persist)) return;

        // Reserve the carve-out for persisting accesses on this stream.
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, span);

        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr  = static_cast<void*>(lo);
        attr.accessPolicyWindow.num_bytes = span;
        attr.accessPolicyWindow.hitRatio  = 1.0f;
        attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        if (cudaStreamSetAttribute(
                stream_, cudaStreamAttributeAccessPolicyWindow, &attr)
            == cudaSuccess) {
            active_ = true;
        }
#else
        (void)stream; (void)p0; (void)n0; (void)p1; (void)n1;
#endif
    }

    ~L2PersistScope() {
#if ENABLE_L2_PERSIST
        if (!active_) return;
        // Reset the window (num_bytes=0 disables it) and release the carve-out
        // so subsequent ops on this stream see a clean, fully-normal L2.
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr  = nullptr;
        attr.accessPolicyWindow.num_bytes = 0;
        attr.accessPolicyWindow.hitRatio  = 0.0f;
        attr.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
        attr.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;
        cudaStreamSetAttribute(
            stream_, cudaStreamAttributeAccessPolicyWindow, &attr);
        cudaCtxResetPersistingL2Cache();
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
#endif
    }

    L2PersistScope(const L2PersistScope&) = delete;
    L2PersistScope& operator=(const L2PersistScope&) = delete;

    bool active() const { return active_; }

private:
    cudaStream_t stream_;
    bool active_;
};

// =========================================================================
//  §4.2 cp.async background loads (NVIDIA PTX maximization).
//
//  Hand-issued `cp.async.{cg,ca}.shared.global` + commit/wait pipelines for
//  memory-bound global->shared staging loads. The async copy lets the SM
//  issue the load and continue executing (address math, the previous tile's
//  compute) while the bytes arrive — overlapping the global-load latency
//  instead of stalling at a synchronous load.
//
//  Contract / correctness (CRITICAL):
//    issue (cp_async_cg_16 / cp_async_ca_4)  ->  cp_async_commit()  ->
//    cp_async_wait_group<N>()  ->  __syncthreads()  ->  read shared.
//  A correctly WAITED cp.async produces byte-identical shared data to a plain
//  synchronous load. A missing/misplaced wait = reading bytes still in flight
//  = garbage. Every issued group MUST be committed and the consume MUST be
//  preceded by a wait that drains the group(s) feeding it.
//
//  Arch guard: cp.async is sm_80+ (Ampere). On pre-Ampere arches in the
//  codegen matrix the helpers fall back to a plain synchronous copy (commit /
//  wait become no-ops), so call sites are portable and numerically identical.
//
//  Alignment/size: cp.async sources must be 4/8/16-byte aligned and the size
//  must be exactly 4/8/16. Use cp_async_cg_16 for 16B (float4, .cg cache-
//  global) and cp_async_ca_4 for 4B (single float, .ca cache-all).
// =========================================================================

// 16-byte (float4) async global->shared copy, .cg (cache-global) hint.
__device__ __forceinline__ void cp_async_cg_16(
    void* smem_ptr, const void* gmem_ptr
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned s = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(s), "l"(gmem_ptr));
#else
    *reinterpret_cast<float4*>(smem_ptr) =
        *reinterpret_cast<const float4*>(gmem_ptr);
#endif
}

// 4-byte async global->shared copy, .ca (cache-all) hint. For scalar (float)
// staging loops that are not float4-alignable.
__device__ __forceinline__ void cp_async_ca_4(
    void* smem_ptr, const void* gmem_ptr
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned s = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"(s), "l"(gmem_ptr));
#else
    *reinterpret_cast<float*>(smem_ptr) =
        *reinterpret_cast<const float*>(gmem_ptr);
#endif
}

// Commit all cp.async copies issued since the last commit into one group.
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n");
#endif
    // pre-Ampere: synchronous fallback already completed the copy; no-op.
}

// Wait until at most N committed groups remain in flight (the rest have
// landed in shared). cp_async_wait_group<0>() drains ALL outstanding groups.
// N is a compile-time immediate as required by the PTX wait_group operand.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
    // pre-Ampere: synchronous fallback already completed the copy; no-op.
}

// Drain every outstanding cp.async group (wait_group 0).
__device__ __forceinline__ void cp_async_wait_all() {
    cp_async_wait_group<0>();
}

}}} // namespace sg::sm90::primitives
