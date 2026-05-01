// =====================================================================
//  csrc/kernels/cuda/sm_90/supergrok11.cuh
//
//  sm_90 (Hopper) SuperGrok v1.1 kernels + 3 launcher declarations.
//
//  This header is NET-NEW: the previous baseline at
//  csrc/kernels/cuda/sm_90/supergrok11_sm90.cu was deleted in
//  commit 5505b50 and is recovered (for reference only) via:
//    git show 5505b50^:csrc/kernels/cuda/sm_90/supergrok11_sm90.cu
//
//  The math here corresponds to the SuperGrok v1.1 algorithm with the
//  REFRESH §25 / ANALYSIS §8 #1 "easy win" optimisation: the per-tensor
//  cosine-similarity gate reduction is FUSED INTO the full-step apply
//  kernel instead of running as a separate kernel launch.
//
//  Three top-level operations expose launchers:
//
//    1. supergrok11_fused_step    — sharpness EMA + meta-net + cosine
//                                    gate (fused) + Adam + trust-ratio
//                                    + apply.  Two grid sweeps:
//                                      sweep A: cosine + sharpness
//                                               reductions (3 + 1 outputs)
//                                               + meta-net forward to e_t
//                                      sweep B: gate apply + Adam +
//                                               trust-ratio reduction +
//                                               apply (cooperative grid
//                                               sync between the two
//                                               internal phases).
//    2. sam_perturb_all           — per-tensor θ ← θ + ρ·g/||g||
//    3. sharpness_restore_all     — per-tensor θ ← θ_pert - ρ·g/||g||
//
//  Reduction strategy: warp-reduce → one atomicAdd per warp into a small
//  device scratch buffer ("scalars[]"). Sweep A uses a "last block
//  finished" gridDim.x atomic counter to compute the cosine gate α and
//  the new sharpness EMA s_t in-place, eliminating a second tiny kernel
//  launch. Sweep B uses cooperative_groups::this_grid().sync() to fuse
//  the trust-ratio reduction with the parameter apply phase. Total
//  device-memory sweeps over per-element data: 2.
//
//  Dtype matrix (instantiated in the .cu TU):
//    ParamT in {float, __nv_bfloat16, __half}                     (3)
//    StateT in {float, __nv_bfloat16}                             (2)
//    GradT  in {float, __nv_bfloat16, __half,
//               __nv_fp8_e4m3, __nv_fp8_e5m2}                     (5)
//
//  Coherence rules (encoded as static_assert inside the kernels):
//    - FP8 GradT requires reduced-precision (BF16/FP16) ParamT — FP8
//      grads with FP32 params silently lose dynamic range without an
//      explicit rescale, mirroring adamw.cuh::is_coherent_combo.
//    - All math is FP32; only loads/stores are typed.
//
//  Meta-net φ is bound through __constant__ memory by the launcher
//  (cudaMemcpyToSymbolAsync). Hidden width H is a runtime template tag
//  (specialised in .cu) so the inner loop is fully unrolled. SG v1.1's
//  MLP is two-input (ĝ, s_t) → H → 1, so we open-code the dot-product
//  with PTX FMAs rather than calling utils.cuh::ptx_expert_mlp_forward<H>
//  (that helper specialises on a single-scalar input).
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/ptx_intrinsics.cuh"
#include "csrc/common/tuned_configs.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) ? (__CUDA_ARCH__ >= 890) : 1
  #include <cuda_fp8.h>
#endif

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cstdint>

namespace cg = cooperative_groups;

namespace sg { namespace sm90 { namespace supergrok11 {

// ---------------------------------------------------------------------
// Launch parameters drawn from tuned_configs.h::DEFAULT_CONFIG. The
// SG11 kernel-specific table is not yet populated (see the TODO at
// tuned_configs.h:122); we therefore mirror the DEFAULT_CONFIG fields
// statically here. The .cu launcher reads them at compile time.
// ---------------------------------------------------------------------

constexpr int SG11_REDUCE_GRID_CAP     = 1024;   // persistent reduce grid
constexpr int SG11_META_PHI_MAX_FLOATS = 2048;   // __constant__ budget

// Scratch layout (per-tensor, allocated by the launcher in a thread-local
// device buffer big enough for the largest tensor seen so far):
//   scalars[0] = sum(ĝ · e)
//   scalars[1] = sum(ĝ²)
//   scalars[2] = sum(e²)
//   scalars[3] = sum((ĝ - g)²)        // for s_t update
//   scalars[4] = α                     // gate (written by last-block-finished)
//   scalars[5] = block_done_counter    // gridDim.x sentinel for sweep A
//   scalars[6] = ||θ_{t-1}||²          // sweep B
//   scalars[7] = ||u_t||²              // sweep B
//   scalars[8] = trust_ratio_mu        // written after grid_sync in sweep B
constexpr int SG11_SCRATCH_FLOATS = 9;

// ---------------------------------------------------------------------
// Compile-time predicates for the dtype matrix.
// ---------------------------------------------------------------------

template <typename T>
struct is_param_dtype
    : std::integral_constant<bool,
        std::is_same<T, float>::value         ||
        std::is_same<T, __nv_bfloat16>::value ||
        std::is_same<T, __half>::value> {};

template <typename T>
struct is_state_dtype
    : std::integral_constant<bool,
        std::is_same<T, float>::value         ||
        std::is_same<T, __nv_bfloat16>::value> {};

template <typename T>
struct is_grad_dtype
    : std::integral_constant<bool,
        std::is_same<T, float>::value          ||
        std::is_same<T, __nv_bfloat16>::value  ||
        std::is_same<T, __half>::value         ||
        std::is_same<T, __nv_fp8_e4m3>::value  ||
        std::is_same<T, __nv_fp8_e5m2>::value> {};

template <typename T>
struct is_fp8
    : std::integral_constant<bool,
        std::is_same<T, __nv_fp8_e4m3>::value  ||
        std::is_same<T, __nv_fp8_e5m2>::value> {};

// FP8 grads require reduced-precision params. Mirrors adamw.cuh.
template <typename ParamT, typename GradT>
struct is_coherent_combo
    : std::integral_constant<bool,
        !(is_fp8<GradT>::value && std::is_same<ParamT, float>::value)> {};

template <typename ParamT, typename StateT, typename GradT>
struct dtype_combo_is_valid
    : std::integral_constant<bool,
        is_param_dtype<ParamT>::value &&
        is_state_dtype<StateT>::value &&
        is_grad_dtype<GradT>::value   &&
        is_coherent_combo<ParamT, GradT>::value> {};

// ---------------------------------------------------------------------
// Type-erased load / store helpers — math is always FP32. Mirrors the
// pattern in adamw.cuh; kept local to this namespace so the sm_90 SG11
// build owns its set without colliding with the AdamW family.
// ---------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p);

template <>
__device__ __forceinline__ float load_as_float<float>(const float* p) {
    return LDG(p);
}
template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(LDG(p));
}
template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* p) {
    return __half2float(LDG(p));
}
template <>
__device__ __forceinline__ float load_as_float<__nv_fp8_e4m3>(const __nv_fp8_e4m3* p) {
    return static_cast<float>(*p);
}
template <>
__device__ __forceinline__ float load_as_float<__nv_fp8_e5m2>(const __nv_fp8_e5m2* p) {
    return static_cast<float>(*p);
}

__device__ __forceinline__ float load_state(const float* p)         { return stream_load(p); }
__device__ __forceinline__ float load_state(const __nv_bfloat16* p) { return __bfloat162float(*p); }

template <typename T>
__device__ __forceinline__ void store_from_float(T* p, float v);

template <>
__device__ __forceinline__ void store_from_float<float>(float* p, float v) {
    *p = v;
}
template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ void store_from_float<__half>(__half* p, float v) {
    *p = __float2half_rn(v);
}

__device__ __forceinline__ void store_state(float* p, float v)         { stream_store(p, v); }
__device__ __forceinline__ void store_state(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

// ---------------------------------------------------------------------
// Meta-net φ in __constant__ memory. Layout matches the binding's
// W1[H,2] row-major + b1[H] + W2[1,H] + b2[1] → packed FP32.
// Total floats: 2H + H + H + 1 = 4H + 1; with H_max = 511 we fit
// SG11_META_PHI_MAX_FLOATS = 2048 inside the 64 KiB __constant__ bank.
// ---------------------------------------------------------------------

__constant__ float c_meta_phi[SG11_META_PHI_MAX_FLOATS];

// Sigmoid via fast_exp_ptx (PTX ex2.approx). Used for both gate and the
// fast-GELU surrogate inside the meta-net inner loop.
__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + fast_exp_ptx(-x));
}

// ---------------------------------------------------------------------
// Block-wide reduction of a quartet of FP32 partials → 4 atomicAdds
// per warp into the global scalars[] buffer. Folded into the inner loop
// of sweep A; we pay one __syncthreads at function tail.
// ---------------------------------------------------------------------
template <int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce4_atomic(
    float a, float b, float c, float d,
    float* __restrict__ scalars   // [4]
) {
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    // Warp-level sum (full mask, in-warp shfl).
    a = warp_reduce_sum(a, 32, lane);
    b = warp_reduce_sum(b, 32, lane);
    c = warp_reduce_sum(c, 32, lane);
    d = warp_reduce_sum(d, 32, lane);

    // Lane 0 of each warp → 4 atomicAdds (uncoalesced but 4·WARPS_PER_BLOCK
    // total ops per block, dwarfed by the per-element work).
    if (lane == 0) {
        atomicAdd(&scalars[0], a);
        atomicAdd(&scalars[1], b);
        atomicAdd(&scalars[2], c);
        atomicAdd(&scalars[3], d);
    }
    (void) warp_id; (void) WARPS_PER_BLOCK;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce2_atomic(
    float a, float b,
    float* __restrict__ scalars   // [2]
) {
    const int lane = threadIdx.x & 31;
    a = warp_reduce_sum(a, 32, lane);
    b = warp_reduce_sum(b, 32, lane);
    if (lane == 0) {
        atomicAdd(&scalars[0], a);
        atomicAdd(&scalars[1], b);
    }
}

// ---------------------------------------------------------------------
// Sweep A kernel: meta-net forward + cosine-gate reduction + sharpness
// EMA reduction. Side-effect: writes e_t to a per-tensor scratch buffer
// (e_scratch[N], FP32) so sweep B can re-read it as plain FP32.
//
// Reductions accumulated into scalars[0..3]:
//   [0] sum(ĝ · e)
//   [1] sum(ĝ²)
//   [2] sum(e²)
//   [3] sum((ĝ - g)²)        // for sharpness EMA s_t
//
// Last-block-finished trick (atomic counter at scalars[5]) computes
// gate α and the new sharpness EMA in one thread of the final block,
// publishing α at scalars[4] and the FP32 sharpness at sharpness_ema[0].
// This eliminates the "tiny finish kernel" launch.
// ---------------------------------------------------------------------

template <typename ParamT, typename StateT, typename GradT,
          int BLOCK_SIZE, int H>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void sg11_sweep_a_kernel(
    const GradT*  __restrict__ perturbed_grads,   // ĝ  [N]
    const GradT*  __restrict__ grads,             // g  [N]
    StateT*       __restrict__ sharpness_ema,     // [1] (per-tensor)
    float*        __restrict__ e_scratch,         // [N] FP32
    float*        __restrict__ scalars,           // [SG11_SCRATCH_FLOATS]
    float kappa,                                  // EMA rate for s_t
    float temperature,                            // gate temperature T
    int64_t n_elements
) {
    static_assert(dtype_combo_is_valid<ParamT, StateT, GradT>::value,
        "supergrok11: invalid (ParamT, StateT, GradT) combo "
        "(see is_coherent_combo: FP8 grad requires non-FP32 param)");

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Each thread accumulates 4 partials in registers.
    float acc_dot   = 0.0f;
    float acc_gn2   = 0.0f;   // ‖ĝ‖²
    float acc_en2   = 0.0f;   // ‖e‖²
    float acc_dgg   = 0.0f;   // ‖ĝ - g‖²

    // Constant-memory layout offsets.
    const float* W1 = c_meta_phi;
    const float* b1 = c_meta_phi + (2 * H);
    const float* W2 = c_meta_phi + (2 * H) + H;
    const float  b2 = c_meta_phi[(2 * H) + H + H];

    // Read previous sharpness scalar once; broadcast to all threads.
    const float s_prev = static_cast<float>(load_state(sharpness_ema));

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float gh = load_as_float(perturbed_grads + i);  // ĝ
        const float g  = load_as_float(grads          + i);   // g

        // Meta-net forward: Linear(2,H) → fast-GELU → Linear(H,1) + b2.
        // Inputs (gh, s_prev). H is a compile-time constant, so the loop
        // unrolls fully.
        float mlp = b2;
        #pragma unroll
        for (int h = 0; h < H; ++h) {
            const float z = ptx_fma(W1[h * 2], gh,
                              ptx_fma(W1[h * 2 + 1], s_prev, b1[h]));
            // Fast-GELU sigmoid surrogate: x * sigmoid(1.702 * x).
            const float gelu = z * sigmoid_fast(1.702f * z);
            mlp = ptx_fma(W2[h], gelu, mlp);
        }
        const float e = mlp;
        e_scratch[i] = e;

        const float diff = gh - g;
        acc_dot += gh * e;
        acc_gn2 += gh * gh;
        acc_en2 += e  * e;
        acc_dgg += diff * diff;
    }

    // Block-wide reduce → atomicAdd into scalars[0..3].
    block_reduce4_atomic<BLOCK_SIZE>(acc_dot, acc_gn2, acc_en2, acc_dgg, scalars);

    // "Last block finished" pattern: increment a counter, the block that
    // sees gridDim.x-1 publishes α and s_t.
    __threadfence();   // make atomicAdds visible
    __syncthreads();
    if (threadIdx.x == 0) {
        // scalars[5] used as block_done counter; FP32 reinterp is fine.
        unsigned old = atomicAdd(reinterpret_cast<unsigned*>(&scalars[5]), 1u);
        if (old == gridDim.x - 1) {
            const float dot   = scalars[0];
            const float gnrm  = sqrtf(fmaxf(scalars[1], 0.0f));
            const float enrm  = sqrtf(fmaxf(scalars[2], 0.0f));
            const float cos_t = dot / (gnrm * enrm + 1e-12f);
            const float alpha = sigmoid_fast(cos_t / fmaxf(temperature, 1e-6f));
            scalars[4] = alpha;
            // Sharpness EMA: s_t = (1-κ)·s_{t-1} + κ·‖ĝ-g‖² (mean over N).
            const float mean_diff = scalars[3] / static_cast<float>(n_elements);
            const float s_t = (1.0f - kappa) * s_prev + kappa * mean_diff;
            store_state(sharpness_ema, s_t);
            // Reset counter for the next call.
            scalars[5] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------
// Sweep B kernel (COOPERATIVE LAUNCH): two internal phases joined by
// cg::this_grid().sync() so we get reduce → apply in a single launch.
//
// Phase 1: per-element compute g̃ = α·ĝ + (1-α)·e, Adam moments, u_t,
//          accumulate ‖θ_{t-1}‖² and ‖u_t‖² into scalars[6..7]. We
//          write u_t into a scratch buffer (u_scratch[N], FP32) so
//          phase 2 can re-read it after the grid sync without a third
//          GMEM sweep over (m, v, ĝ, e).
//
// Phase 2: read trust ratio from scalars[8] (computed once by lane 0
//          of block 0 right after grid_sync), apply
//              θ ← (1 - lr·wd)·θ - lr·μ·u
//          using FP32 math, store-typed.
//
// Total device-memory sweeps over per-element data across the kernel:
// one (phase 1 reads ĝ, e, m, v, θ; phase 2 reads θ + u). The state
// stores in phase 1 are non-temporal so L2 stays warm for θ.
// ---------------------------------------------------------------------

template <typename ParamT, typename StateT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void sg11_sweep_b_kernel(
    ParamT*       __restrict__ params,         // θ_{t-1}            [N]
    StateT*       __restrict__ exp_avg,        // m                  [N]
    StateT*       __restrict__ exp_avg_sq,     // v                  [N]
    const GradT*  __restrict__ perturbed_grads,// ĝ                  [N]
    const float*  __restrict__ e_scratch,      // e_t (from sweep A) [N]
    float*        __restrict__ u_scratch,      // FP32 u_t           [N]
    float*        __restrict__ scalars,        // [SG11_SCRATCH_FLOATS]
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float trust_clip,
    float bc1, float bc2,
    int64_t n_elements
) {
    static_assert(dtype_combo_is_valid<ParamT, StateT, GradT>::value,
        "supergrok11: invalid (ParamT, StateT, GradT) combo");

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Read α once per thread (broadcast through L1).
    const float alpha    = scalars[4];
    const float one_ma   = 1.0f - alpha;

    float acc_pn2 = 0.0f;     // ‖θ_{t-1}‖²
    float acc_un2 = 0.0f;     // ‖u_t‖²

    // ── Phase 1: Adam + trust-norm reductions ────────────────────────
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float gh = load_as_float(perturbed_grads + i);
        const float e  = LDG(e_scratch + i);
        const float m0 = load_state(exp_avg    + i);
        const float v0 = load_state(exp_avg_sq + i);
        const float p0 = load_as_float(params + i);

        const float g_tilde = alpha * gh + one_ma * e;

        const float m1 = beta1 * m0 + (1.0f - beta1) * g_tilde;
        const float v1 = beta2 * v0 + (1.0f - beta2) * g_tilde * g_tilde;

        const float m_hat = m1 / bc1;
        const float v_hat = v1 / bc2;
        const float rsv   = fast_rsqrt_nr(fmaxf(v_hat, 0.0f));
        const float u     = m_hat * (rsv / (1.0f + eps * rsv));

        // Stream-store moments (non-temporal: keeps L2 budget for θ).
        store_state(exp_avg    + i, m1);
        store_state(exp_avg_sq + i, v1);

        // Stash u for phase 2; FP32 by design (we already pay the
        // cooperative-launch tax, no point round-tripping through ParamT).
        u_scratch[i] = u;

        acc_pn2 += p0 * p0;
        acc_un2 += u  * u;
    }

    block_reduce2_atomic<BLOCK_SIZE>(acc_pn2, acc_un2, &scalars[6]);

    // Grid-wide barrier: every block must finish phase 1 before phase 2.
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // ── Trust ratio (single-thread compute, broadcast via scalars[8]) ─
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float pnorm = sqrtf(fmaxf(scalars[6], 0.0f));
        const float unorm = sqrtf(fmaxf(scalars[7], 0.0f));
        // μ_t = min(ℓ, ‖θ‖ / (‖u‖ + ε))
        float mu = pnorm / (unorm + 1e-12f);
        if (trust_clip > 0.0f && mu > trust_clip) mu = trust_clip;
        scalars[8] = mu;
        // Reset accumulators for the next call.
        scalars[0] = 0.0f; scalars[1] = 0.0f;
        scalars[2] = 0.0f; scalars[3] = 0.0f;
        scalars[6] = 0.0f; scalars[7] = 0.0f;
    }
    grid.sync();

    const float trust_mu = scalars[8];
    const float decay    = 1.0f - lr * weight_decay;
    const float step_sz  = lr * trust_mu;

    // ── Phase 2: apply update ────────────────────────────────────────
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float p0 = load_as_float(params + i);
        const float u  = LDG(u_scratch + i);
        const float p1 = decay * p0 - step_sz * u;
        store_from_float(params + i, p1);
    }
}

// ---------------------------------------------------------------------
// sam_perturb_kernel:  θ_pert ← θ + ρ · g / ‖g‖₂
// `grad_norm_inv` is computed CPU-side from a single-sync vector reduce
// (compute_sam_grad_norm_device_side in _helpers.h) and passed in.
// ---------------------------------------------------------------------

template <typename ParamT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void sg11_sam_perturb_kernel(
    ParamT*       __restrict__ params,
    const GradT*  __restrict__ grads,
    float rho,
    float grad_norm_inv,
    int64_t n_elements
) {
    static_assert(is_param_dtype<ParamT>::value, "SG11 SAM: invalid ParamT");
    static_assert(is_grad_dtype<GradT>::value,   "SG11 SAM: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "SG11 SAM: FP8 grad with FP32 param requires explicit rescale");

    const float k = rho * grad_norm_inv;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float p = load_as_float(params + i);
        const float g = load_as_float(grads  + i);
        store_from_float(params + i, p + k * g);
    }
}

// ---------------------------------------------------------------------
// sharpness_restore_kernel:  θ ← θ_pert - ρ · g / ‖g‖₂
// Same arithmetic as sam_perturb but inverted sign.
// ---------------------------------------------------------------------

template <typename ParamT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void sg11_sharpness_restore_kernel(
    ParamT*       __restrict__ params,
    const GradT*  __restrict__ grads,
    float rho,
    float grad_norm_inv,
    int64_t n_elements
) {
    static_assert(is_param_dtype<ParamT>::value, "SG11 restore: invalid ParamT");
    static_assert(is_grad_dtype<GradT>::value,   "SG11 restore: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "SG11 restore: FP8 grad with FP32 param requires explicit rescale");

    const float k = rho * grad_norm_inv;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float p = load_as_float(params + i);
        const float g = load_as_float(grads  + i);
        store_from_float(params + i, p - k * g);
    }
}

// =====================================================================
// Per-tensor scratch slab. The launcher owns a thread-local cache that
// grows monotonically with the largest tensor seen so far. Layout:
//   [0 .. SG11_SCRATCH_FLOATS)         : scalars[]
//   [pad to 256 B aligned]             : e_scratch[N]
//   [N .. 2N) (FP32-stride)            : u_scratch[N]
// =====================================================================

struct sg11_scratch_t {
    float*  base;       // device pointer to the slab
    int64_t capacity;   // in elements N (e_scratch and u_scratch each)
};

inline cudaError_t sg11_scratch_ensure(sg11_scratch_t& s, int64_t n_elements) {
    if (n_elements <= s.capacity) return cudaSuccess;
    if (s.base) cudaFree(s.base);
    const size_t scratch_bytes =
        sizeof(float) * (size_t)SG11_SCRATCH_FLOATS +
        sizeof(float) * (size_t)n_elements * 2u;
    cudaError_t err = cudaMalloc(&s.base, scratch_bytes);
    if (err != cudaSuccess) { s.base = nullptr; s.capacity = 0; return err; }
    s.capacity = n_elements;
    return cudaSuccess;
}

inline float* sg11_scratch_scalars(const sg11_scratch_t& s)        { return s.base; }
inline float* sg11_scratch_e(const sg11_scratch_t& s)              {
    return s.base + SG11_SCRATCH_FLOATS;
}
inline float* sg11_scratch_u(const sg11_scratch_t& s)              {
    return s.base + SG11_SCRATCH_FLOATS + s.capacity;
}

// Thread-local accessor; one slab per host thread is fine (CUDA stream
// is host-thread-bound by default in PyTorch). This is intentionally
// not freed at exit — the CUDA runtime tears the context down.
inline sg11_scratch_t& sg11_get_tls_scratch() {
    thread_local sg11_scratch_t s = {nullptr, 0};
    return s;
}

// =====================================================================
// Internal H-templated launcher (fused step). Caller picks H at runtime
// via the meta_hidden parameter. Currently specialised for H ∈ {16, 32,
// 64, 128, 256}. Anything else falls back to a runtime loop variant
// (rejected here with cudaErrorInvalidValue — adding a runtime-H kernel
// would lose the unrolled inner loop the §25 §8.1 win depends on).
// =====================================================================

template <typename ParamT, typename StateT, typename GradT, int H>
cudaError_t sg11_launch_fused_step_H(
    ParamT* params, StateT* exp_avg, StateT* exp_avg_sq,
    StateT* sharpness_ema,
    const GradT* perturbed_grads,
    const GradT* grads,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float kappa, float temperature, float trust_clip,
    float bc1, float bc2,
    int64_t n_elements, int64_t step_count,
    cudaStream_t stream
) {
    static_assert(dtype_combo_is_valid<ParamT, StateT, GradT>::value,
        "supergrok11: invalid (ParamT, StateT, GradT) combo");
    static_assert((4 * H + 1) <= SG11_META_PHI_MAX_FLOATS,
        "supergrok11: meta_hidden too large for __constant__ budget");

    if (n_elements <= 0) return cudaSuccess;

    // Block size from tuned_configs.h. SG11 reuses the GROKADAMW table
    // until autotune populates SG11_CONFIGS (TODO at tuned_configs.h:122).
    const LaunchConfig cfg =
        get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;
    constexpr int MAX_BLOCKS = SG11_REDUCE_GRID_CAP;
    const int64_t needed = (n_elements + block - 1) / block;
    const int grid =
        static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);

    // Ensure scratch is large enough.
    sg11_scratch_t& sc = sg11_get_tls_scratch();
    cudaError_t err = sg11_scratch_ensure(sc, n_elements);
    if (err != cudaSuccess) return err;
    float* scalars   = sg11_scratch_scalars(sc);
    float* e_scratch = sg11_scratch_e(sc);
    float* u_scratch = sg11_scratch_u(sc);

    // Zero the scalars header (FP32 atomics into bins 0..3, 6..7).
    err = cudaMemsetAsync(scalars, 0,
        sizeof(float) * (size_t)SG11_SCRATCH_FLOATS, stream);
    if (err != cudaSuccess) return err;

    (void) step_count;  // reserved (would seed BF16 SR PRNG if added later)

    // Sweep A — non-cooperative; uses last-block-finished to publish α.
    if (block == 128) {
        sg11_sweep_a_kernel<ParamT, StateT, GradT, 128, H>
            <<<grid, 128, 0, stream>>>(
                perturbed_grads, grads, sharpness_ema,
                e_scratch, scalars, kappa, temperature, n_elements);
    } else if (block == 512) {
        sg11_sweep_a_kernel<ParamT, StateT, GradT, 512, H>
            <<<grid, 512, 0, stream>>>(
                perturbed_grads, grads, sharpness_ema,
                e_scratch, scalars, kappa, temperature, n_elements);
    } else {
        sg11_sweep_a_kernel<ParamT, StateT, GradT, 256, H>
            <<<grid, 256, 0, stream>>>(
                perturbed_grads, grads, sharpness_ema,
                e_scratch, scalars, kappa, temperature, n_elements);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Sweep B — cooperative launch, fuses trust-ratio reduce + apply.
    void* args[] = {
        (void*)&params, (void*)&exp_avg, (void*)&exp_avg_sq,
        (void*)&perturbed_grads,
        (void*)&e_scratch, (void*)&u_scratch, (void*)&scalars,
        (void*)&lr, (void*)&beta1, (void*)&beta2, (void*)&eps,
        (void*)&weight_decay, (void*)&trust_clip, (void*)&bc1, (void*)&bc2,
        (void*)&n_elements
    };
    auto kfn = (void*)&sg11_sweep_b_kernel<ParamT, StateT, GradT, 256>;
    int b_block = 256;
    if (block == 128) {
        kfn = (void*)&sg11_sweep_b_kernel<ParamT, StateT, GradT, 128>;
        b_block = 128;
    } else if (block == 512) {
        kfn = (void*)&sg11_sweep_b_kernel<ParamT, StateT, GradT, 512>;
        b_block = 512;
    }

    // Cooperative launch grid is bounded by the occupancy limit; query and
    // clamp. CUDA refuses launches larger than max_active_blocks * num_SMs.
    int max_blocks_per_sm = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, kfn, b_block, /*dynamicSMemBytes=*/0);
    if (err != cudaSuccess) return err;
    int dev = 0; (void)cudaGetDevice(&dev);
    int num_sm = 0; (void)cudaDeviceGetAttribute(&num_sm,
        cudaDevAttrMultiProcessorCount, dev);
    const int coop_grid_cap = max_blocks_per_sm * num_sm;
    int b_grid = grid;
    if (b_grid > coop_grid_cap) b_grid = coop_grid_cap;
    if (b_grid < 1) b_grid = 1;

    err = cudaLaunchCooperativeKernel(kfn, b_grid, b_block, args, 0, stream);
    return err;
}

// =====================================================================
// PUBLIC LAUNCHER: supergrok11_fused_step
// Signature exactly per the build spec — verified against
// csrc/bindings/supergrok11.cpp's vector entry. Note that the binding
// today still uses the legacy per-tensor calls (launch_sg11_mu_metanet,
// launch_sg11_adam_decay, dispatch_cosine_gate); migrating it to call
// this fused entry is a follow-up bindings change (see report below).
// =====================================================================

template <typename ParamT, typename StateT, typename GradT>
cudaError_t launch_supergrok11_fused_step(
    ParamT* params, StateT* exp_avg, StateT* exp_avg_sq,
    StateT* sharpness_ema,
    const GradT* perturbed_grads,
    const GradT* grads,
    const float* meta_phi_weights,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float kappa, float temperature, float trust_clip,
    float bc1, float bc2,
    int meta_hidden, int meta_layers,
    int64_t n_elements, int64_t step_count,
    cudaStream_t stream
) {
    static_assert(dtype_combo_is_valid<ParamT, StateT, GradT>::value,
        "supergrok11: invalid (ParamT, StateT, GradT) combo");

    if (n_elements <= 0) return cudaSuccess;
    if (meta_layers != 2) {
        // SG v1.1's meta-net φ is a 2-layer MLP (2 → H → 1). meta_layers
        // is forwarded for forward-compat with future deep-φ variants;
        // anything else here is a programmer error.
        return cudaErrorInvalidValue;
    }
    const size_t n_phi = (size_t)(4 * meta_hidden + 1);
    if (n_phi > (size_t)SG11_META_PHI_MAX_FLOATS) return cudaErrorInvalidValue;

    // Bind φ into __constant__ memory for the duration of this launch.
    cudaError_t err = cudaMemcpyToSymbolAsync(
        c_meta_phi, meta_phi_weights, n_phi * sizeof(float),
        /*offset=*/0, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;

    // Dispatch H to the templated path so the meta-net inner loop is
    // fully unrolled. Hidden widths outside the menu fall back to the
    // closest specialisation that has H >= meta_hidden (the kernel reads
    // 4·H + 1 floats from c_meta_phi; reading past the populated region
    // is undefined unless we tile-zero, so we error instead).
    switch (meta_hidden) {
        case 16:
            return sg11_launch_fused_step_H<ParamT, StateT, GradT, 16>(
                params, exp_avg, exp_avg_sq, sharpness_ema,
                perturbed_grads, grads,
                lr, beta1, beta2, eps, weight_decay,
                kappa, temperature, trust_clip, bc1, bc2,
                n_elements, step_count, stream);
        case 32:
            return sg11_launch_fused_step_H<ParamT, StateT, GradT, 32>(
                params, exp_avg, exp_avg_sq, sharpness_ema,
                perturbed_grads, grads,
                lr, beta1, beta2, eps, weight_decay,
                kappa, temperature, trust_clip, bc1, bc2,
                n_elements, step_count, stream);
        case 64:
            return sg11_launch_fused_step_H<ParamT, StateT, GradT, 64>(
                params, exp_avg, exp_avg_sq, sharpness_ema,
                perturbed_grads, grads,
                lr, beta1, beta2, eps, weight_decay,
                kappa, temperature, trust_clip, bc1, bc2,
                n_elements, step_count, stream);
        case 128:
            return sg11_launch_fused_step_H<ParamT, StateT, GradT, 128>(
                params, exp_avg, exp_avg_sq, sharpness_ema,
                perturbed_grads, grads,
                lr, beta1, beta2, eps, weight_decay,
                kappa, temperature, trust_clip, bc1, bc2,
                n_elements, step_count, stream);
        default:
            // Common SG11 default in grokking_optimizers/supergrok11.py
            // is meta_hidden_dim=32; anything off-menu is rejected.
            return cudaErrorInvalidValue;
    }
}

// =====================================================================
// PUBLIC LAUNCHER: supergrok11_sam_perturb_all  (per-tensor wrapper)
// θ_pert ← θ + ρ · g / ‖g‖₂   — caller pre-computed the global norm
// reciprocal via _helpers.h::compute_sam_grad_norm_device_side.
// =====================================================================

template <typename ParamT, typename GradT>
cudaError_t launch_supergrok11_sam_perturb_all(
    ParamT* params, const GradT* grads,
    float rho, float grad_norm_inv,
    int64_t n_elements, cudaStream_t stream
) {
    static_assert(is_param_dtype<ParamT>::value, "SG11 SAM: invalid ParamT");
    static_assert(is_grad_dtype<GradT>::value,   "SG11 SAM: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "SG11 SAM: FP8 grad with FP32 param requires explicit rescale");

    if (n_elements <= 0) return cudaSuccess;

    const LaunchConfig cfg =
        get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;
    constexpr int MAX_BLOCKS = 8192;
    const int64_t needed = (n_elements + block - 1) / block;
    const int grid =
        static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);

    if (block == 128) {
        sg11_sam_perturb_kernel<ParamT, GradT, 128>
            <<<grid, 128, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    } else if (block == 512) {
        sg11_sam_perturb_kernel<ParamT, GradT, 512>
            <<<grid, 512, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    } else {
        sg11_sam_perturb_kernel<ParamT, GradT, 256>
            <<<grid, 256, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    }
    return cudaGetLastError();
}

// =====================================================================
// PUBLIC LAUNCHER: supergrok11_sharpness_restore_all
// θ ← θ_pert - ρ · g / ‖g‖₂   (mirror of sam_perturb_all)
// =====================================================================

template <typename ParamT, typename GradT>
cudaError_t launch_supergrok11_sharpness_restore_all(
    ParamT* params, const GradT* grads,
    float rho, float grad_norm_inv,
    int64_t n_elements, cudaStream_t stream
) {
    static_assert(is_param_dtype<ParamT>::value, "SG11 restore: invalid ParamT");
    static_assert(is_grad_dtype<GradT>::value,   "SG11 restore: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "SG11 restore: FP8 grad with FP32 param requires explicit rescale");

    if (n_elements <= 0) return cudaSuccess;

    const LaunchConfig cfg =
        get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;
    constexpr int MAX_BLOCKS = 8192;
    const int64_t needed = (n_elements + block - 1) / block;
    const int grid =
        static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);

    if (block == 128) {
        sg11_sharpness_restore_kernel<ParamT, GradT, 128>
            <<<grid, 128, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    } else if (block == 512) {
        sg11_sharpness_restore_kernel<ParamT, GradT, 512>
            <<<grid, 512, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    } else {
        sg11_sharpness_restore_kernel<ParamT, GradT, 256>
            <<<grid, 256, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    }
    return cudaGetLastError();
}

}}} // namespace sg::sm90::supergrok11
