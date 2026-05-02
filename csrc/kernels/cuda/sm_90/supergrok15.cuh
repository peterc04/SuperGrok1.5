// =====================================================================
//  csrc/kernels/cuda/sm_90/supergrok15.cuh
//
//  sm_90 (Hopper) SuperGrok v1.5 kernels + 3 launcher declarations.
//
//  This header is NET-NEW: the previous baseline at
//  csrc/kernels/cuda/sm_90/supergrok15_sm90.cu was deleted in
//  commit 5505b50 and is recovered (for reference only) via:
//    git show 5505b50^:csrc/kernels/cuda/sm_90/supergrok15_sm90.cu
//
//  The math here corresponds to the SuperGrok v1.5 algorithm:
//
//    s_t   = (1-κ)·s_{t-1} + κ·‖ĝ - g‖²        (per-tensor scalar)
//    e_t   = MLP_φ(ĝ, s_t)                      (per-coord pseudo-grad)
//    α_t   = clip_{[0,1]}(α₀ + τ·e_t)           (PER-COORDINATE)
//    g̃     = α_t ⊙ ĝ + (1-α_t) ⊙ e_t           (elementwise mix)
//    Adam(g̃) → u_t                              (using bc1, bc2)
//    TrustRatio(u_t) → θ_t
//
//  Differences vs SG v1.1 (supergrok11.cuh):
//    - α is PER-COORDINATE (clipped affine of e_t), not the per-tensor
//      sigmoid-of-cosine that SG11 uses. No cosine-similarity
//      reduction, no temperature divide. One fewer reduction in sweep A.
//    - The "smart_grad" in-register pattern (recovered from the deleted
//      sm_90 baseline) keeps g̃ = α·ĝ + (1-α)·e in registers across the
//      Adam moment update, the trust-ratio reduction, and the apply.
//
//  Three top-level operations expose launchers:
//
//    1. supergrok15_fused_step    — sharpness EMA + meta-net + per-coord
//                                    α + Adam + trust-ratio + apply.
//                                    Two grid sweeps:
//                                      sweep A: meta-net forward + ‖ĝ-g‖²
//                                               reduction + e_t scratch
//                                      sweep B: smart_grad register-resident
//                                               Adam + trust-ratio reduction
//                                               + apply (cooperative grid
//                                               sync between phases).
//    2. sam_perturb_all           — per-tensor θ ← θ + ρ·g/‖g‖₂
//    3. sharpness_restore_all     — per-tensor θ ← θ_pert - ρ·g/‖g‖₂
//
//  Reduction strategy: warp-reduce → one atomicAdd per warp into a small
//  device scratch buffer ("scalars[]"). Sweep A uses a single scalar
//  reduction (‖ĝ-g‖²). Sweep B is launched cooperatively
//  (cudaLaunchCooperativeKernel) and uses cg::this_grid().sync() between
//  the moment-update / norm-reduce phase and the apply phase. Total
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
//      explicit rescale, mirroring adamw.cuh::is_coherent_combo and the
//      sibling supergrok11.cuh.
//    - All math is FP32; only loads/stores are typed.
//
//  Meta-net φ is bound through __constant__ memory by the launcher
//  (cudaMemcpyToSymbolAsync). Hidden width H is a compile-time template
//  parameter (specialised in .cu) so the inner MLP loop is fully
//  unrolled. SG v1.5's MLP is two-input (ĝ, s_t) → H → 1 with a
//  fast-GELU surrogate, identical layout to SG v1.1.
//
//  NAMESPACE NOTE: csrc/bindings/supergrok15.cpp expects per-tensor
//  entries (launch_fused_supergrok15_full_step, launch_sam_perturb,
//  launch_sharpness_restore) in namespace sg::sm90. The launchers
//  defined here follow the build-spec signatures
//  (launch_supergrok15_fused_step, ...sam_perturb_all,
//  ...sharpness_restore_all) inside namespace sg::sm90::supergrok15 —
//  matching the sibling supergrok11.cuh design. A thin per-tensor
//  shim TU (not in this PR) is required to bridge the binding to these
//  vectorized entries; mirrors the SG11 follow-up bindings change.
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

namespace sg { namespace sm90 { namespace supergrok15 {

// ---------------------------------------------------------------------
// Launch parameters drawn from tuned_configs.h::DEFAULT_CONFIG (the
// SG15 kernel-specific table is not yet populated; SG11 mirrors the
// same TODO — see tuned_configs.h:122). Block size is read at runtime
// from get_grokadamw_config() and dispatched to a small static menu of
// compile-time block sizes so __launch_bounds__ stays accurate.
// ---------------------------------------------------------------------

constexpr int SG15_REDUCE_GRID_CAP     = 1024;   // persistent reduce grid
constexpr int SG15_META_PHI_MAX_FLOATS = 2048;   // __constant__ budget

// Scratch layout (per-tensor, allocated by the launcher in a thread-local
// device buffer big enough for the largest tensor seen so far):
//   scalars[0] = sum((ĝ - g)²)         // sharpness EMA numerator
//   scalars[1] = block_done_counter    // gridDim.x sentinel for sweep A
//   scalars[2] = ‖θ_{t-1}‖²            // sweep B phase 1
//   scalars[3] = ‖u_t‖²                // sweep B phase 1
//   scalars[4] = trust_ratio_mu        // sweep B (after grid_sync)
constexpr int SG15_SCRATCH_FLOATS = 5;

// ---------------------------------------------------------------------
// Compile-time predicates for the dtype matrix. Mirror supergrok11.cuh
// so the same coherence rules apply.
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
// pattern in adamw.cuh and supergrok11.cuh; kept local to this
// namespace so the SG15 build owns its specialisations cleanly.
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
// 2H + H + H + 1 = 4H + 1 floats; with H_max ≤ 511 we fit
// SG15_META_PHI_MAX_FLOATS = 2048 inside the 64 KiB __constant__ bank.
// ---------------------------------------------------------------------

__constant__ float c_meta_phi[SG15_META_PHI_MAX_FLOATS];

// Sigmoid via fast_exp_ptx (PTX ex2.approx). Used for the fast-GELU
// surrogate inside the meta-net inner loop.
__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + fast_exp_ptx(-x));
}

// Per-coordinate α_t = clip_{[0,1]}(α₀ + τ·e_t).  No sigmoid, no
// per-tensor reductions — the whole point of v1.5.
__device__ __forceinline__ float alpha_per_coord(float alpha0, float tau, float e_t) {
    return fminf(1.0f, fmaxf(0.0f, ptx_fma(tau, e_t, alpha0)));
}

// ---------------------------------------------------------------------
// Block-wide single-scalar reduction → 1 atomicAdd per warp into
// scalars[0]. Sweep A only needs ‖ĝ-g‖² (one scalar) — the SG11 quartet
// reduce is overkill here, hence the dedicated 1-bin variant.
// ---------------------------------------------------------------------
template <int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce1_atomic(
    float a, float* __restrict__ scalars   // scalars[0]
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    a = cluster_dsmem_reduce_sum(a);
    if (threadIdx.x == 0) {
        namespace cg = cooperative_groups;
        auto cluster = cg::this_cluster();
        if (cluster.block_rank() == 0)
            atomicAdd(&scalars[0], a);
    }
#else
    const int lane = threadIdx.x & 31;
    a = warp_reduce_sum(a, 32, lane);
    if (lane == 0) atomicAdd(&scalars[0], a);
#endif
}

// Block-wide pair reduction → 2 atomicAdds per warp into a caller-supplied
// pair pointer (caller passes &scalars[2] for sweep B phase 1).
template <int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce2_atomic(
    float a, float b, float* __restrict__ pair   // pair[0..1]
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    a = cluster_dsmem_reduce_sum(a);
    b = cluster_dsmem_reduce_sum(b);
    if (threadIdx.x == 0) {
        namespace cg = cooperative_groups;
        auto cluster = cg::this_cluster();
        if (cluster.block_rank() == 0) {
            atomicAdd(&pair[0], a);
            atomicAdd(&pair[1], b);
        }
    }
#else
    const int lane = threadIdx.x & 31;
    a = warp_reduce_sum(a, 32, lane);
    b = warp_reduce_sum(b, 32, lane);
    if (lane == 0) {
        atomicAdd(&pair[0], a);
        atomicAdd(&pair[1], b);
    }
#endif
}

// ---------------------------------------------------------------------
// Sweep A kernel: per-tensor sharpness reduction + meta-net forward.
//
//   acc_dgg += (ĝ_i - g_i)²        (reduced into scalars[0])
//   e_i      = MLP_φ(ĝ_i, s_{t-1}) (written to e_scratch)
//
// Compared with SG11's sweep A this is **one fewer reduction**: SG15's
// per-coordinate α does not need cosine similarity. We keep the
// last-block-finished pattern to publish s_t in the same launch — exact
// same trick as SG11. Total atomicAdds per warp: 1 (vs SG11's 4).
// ---------------------------------------------------------------------

template <typename ParamT, typename StateT, typename GradT,
          int BLOCK_SIZE, int H>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void sg15_sweep_a_kernel(
    const GradT*  __restrict__ perturbed_grads,   // ĝ  [N]
    const GradT*  __restrict__ grads,             // g  [N]
    StateT*       __restrict__ sharpness_ema,     // [1] (per-tensor)
    float*        __restrict__ e_scratch,         // [N] FP32
    float*        __restrict__ scalars,           // [SG15_SCRATCH_FLOATS]
    float kappa,                                  // EMA rate for s_t
    int64_t n_elements
) {
    static_assert(dtype_combo_is_valid<ParamT, StateT, GradT>::value,
        "supergrok15 sweep A: invalid (ParamT, StateT, GradT) combo "
        "(see is_coherent_combo: FP8 grad requires non-FP32 param)");

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Constant-memory layout offsets — same packed layout as SG11's φ.
    const float* W1 = c_meta_phi;                           // [H,2]
    const float* b1 = c_meta_phi + (2 * H);                 // [H]
    const float* W2 = c_meta_phi + (2 * H) + H;             // [H]
    const float  b2 = c_meta_phi[(2 * H) + H + H];          // [1]

    // s_{t-1}: per-tensor scalar; broadcast to every thread via a single
    // load (loads typed by StateT but the value is FP32 in math).
    const float s_prev = static_cast<float>(load_state(sharpness_ema));

    float acc_dgg = 0.0f;

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float gh = load_as_float(perturbed_grads + i);  // ĝ
        const float g  = load_as_float(grads          + i);   // g

        // Meta-net φ: Linear(2,H) → fast-GELU → Linear(H,1) + b2.
        // Fully unrolled at compile time because H is a template arg.
        float mlp = b2;
        #pragma unroll
        for (int h = 0; h < H; ++h) {
            const float z = ptx_fma(W1[h * 2], gh,
                              ptx_fma(W1[h * 2 + 1], s_prev, b1[h]));
            const float gelu = z * sigmoid_fast(1.702f * z);
            mlp = ptx_fma(W2[h], gelu, mlp);
        }
        // e_t scratch: FP32 (sweep B re-reads as plain FP32 — no GradT
        // round-trip). Stream store so we don't pollute L2 with a
        // single-use value.
        const float e = mlp;
        stream_store(e_scratch + i, e);

        const float diff = gh - g;
        acc_dgg = ptx_fma(diff, diff, acc_dgg);
    }

    // Single-scalar reduce → scalars[0].
    block_reduce1_atomic<BLOCK_SIZE>(acc_dgg, scalars);

    // Last-block-finished trick: increment scalars[1] (block counter);
    // the block that observes gridDim.x-1 publishes s_t and resets.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned old = atomicAdd(reinterpret_cast<unsigned*>(&scalars[1]), 1u);
        if (old == gridDim.x - 1) {
            const float sum_diff  = scalars[0];
            const float mean_diff = sum_diff / static_cast<float>(n_elements);
            const float s_t = (1.0f - kappa) * s_prev + kappa * mean_diff;
            store_state(sharpness_ema, s_t);
            // Reset the per-launch state used by sweep A.
            scalars[0] = 0.0f;
            scalars[1] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------
// Sweep B kernel (COOPERATIVE LAUNCH).
//
// Two phases joined by cg::this_grid().sync(). Recovers the smart_grad
// register-resident pattern from the deleted sm_90 baseline: g̃ stays
// in a register across the Adam moment update so the kernel never
// re-reads ĝ or e_t after the first load.
//
// Phase 1 (per element):
//   ĝ, e_t   ← gmem
//   α_t      = clip_{[0,1]}(α₀ + τ·e_t)        (per-coordinate, register)
//   g̃        = α_t·ĝ + (1-α_t)·e_t             (register-resident)
//   m_1      = β1·m_0 + (1-β1)·g̃
//   v_1      = β2·v_0 + (1-β2)·g̃²
//   m̂        = m_1 / bc1, v̂ = v_1 / bc2
//   u_t      = m̂ · rsqrt(v̂) / (1 + ε·rsqrt(v̂))
//   stash u_t to u_scratch (FP32) for phase 2.
//   accumulate ‖θ_{t-1}‖², ‖u_t‖² into scalars[2..3].
//
// Grid sync, then thread (0,0) computes trust_ratio_mu = min(τ̂, ‖θ‖/(‖u‖+ε))
// and writes scalars[4]. Grid sync again.
//
// Phase 2: θ ← (1 - lr·wd)·θ - lr·μ_t · u
// ---------------------------------------------------------------------

template <typename ParamT, typename StateT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void sg15_sweep_b_kernel(
    ParamT*       __restrict__ params,         // θ_{t-1}            [N]
    StateT*       __restrict__ exp_avg,        // m                  [N]
    StateT*       __restrict__ exp_avg_sq,     // v                  [N]
    const GradT*  __restrict__ perturbed_grads,// ĝ                  [N]
    const float*  __restrict__ e_scratch,      // e_t (from sweep A) [N]
    float*        __restrict__ u_scratch,      // FP32 u_t           [N]
    float*        __restrict__ scalars,        // [SG15_SCRATCH_FLOATS]
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha0, float tau, float trust_clip,
    float bc1, float bc2,
    int64_t n_elements
) {
    static_assert(dtype_combo_is_valid<ParamT, StateT, GradT>::value,
        "supergrok15 sweep B: invalid (ParamT, StateT, GradT) combo");

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Per-thread reduction accumulators for (‖θ‖², ‖u‖²).
    float acc_pn2 = 0.0f;
    float acc_un2 = 0.0f;

    // ── Phase 1: per-coord α + g̃ in register + Adam + reductions ────
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        // Load typed -> FP32 once each. e_scratch is FP32 by construction.
        const float gh = load_as_float(perturbed_grads + i);
        const float e  = LDG(e_scratch + i);
        const float m0 = load_state(exp_avg    + i);
        const float v0 = load_state(exp_avg_sq + i);
        const float p0 = load_as_float(params + i);

        // Per-coordinate α (clipped affine). NO sigmoid, NO cosine.
        const float alpha_i = alpha_per_coord(alpha0, tau, e);

        // smart_grad pattern: g̃ stays in a register through Adam.
        // g̃ = α·ĝ + (1-α)·e  ≡  e + α·(ĝ - e)  (1 FMA, 1 sub).
        const float g_tilde = ptx_fma(alpha_i, gh - e, e);

        // Adam moment update (FP32 math, typed state).
        const float m1 = ptx_fma(beta1, m0, (1.0f - beta1) * g_tilde);
        const float v1 = ptx_fma(beta2, v0, (1.0f - beta2) * g_tilde * g_tilde);

        const float m_hat = m1 / bc1;
        const float v_hat = v1 / bc2;
        const float rsv   = fast_rsqrt_nr(fmaxf(v_hat, 0.0f));
        const float u     = m_hat * (rsv / (1.0f + eps * rsv));

        // Non-temporal moment stores: keep L2 budget for θ.
        store_state(exp_avg    + i, m1);
        store_state(exp_avg_sq + i, v1);

        // Stash u_t for phase 2; FP32 (we already pay the coop tax).
        stream_store(u_scratch + i, u);

        acc_pn2 = ptx_fma(p0, p0, acc_pn2);
        acc_un2 = ptx_fma(u,  u,  acc_un2);
    }

    // Reduce (‖θ‖², ‖u‖²) → scalars[2..3].
    block_reduce2_atomic<BLOCK_SIZE>(acc_pn2, acc_un2, &scalars[2]);

    // Grid-wide barrier: all blocks finish phase 1 before trust ratio.
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // Single-thread compute of trust ratio (clip optional).
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float pnorm = sqrtf(fmaxf(scalars[2], 0.0f));
        const float unorm = sqrtf(fmaxf(scalars[3], 0.0f));
        float mu = pnorm / (unorm + 1e-12f);
        if (trust_clip > 0.0f && mu > trust_clip) mu = trust_clip;
        scalars[4] = mu;
        // Reset for the next launch.
        scalars[2] = 0.0f;
        scalars[3] = 0.0f;
    }
    grid.sync();

    const float trust_mu = scalars[4];
    const float decay    = 1.0f - lr * weight_decay;
    const float step_sz  = lr * trust_mu;

    // ── Phase 2: apply  θ ← decay·θ - step_sz·u  ─────────────────────
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float p0 = load_as_float(params + i);
        const float u  = LDG(u_scratch + i);
        const float p1 = ptx_fma(-step_sz, u, decay * p0);
        store_from_float(params + i, p1);
    }
}

// ---------------------------------------------------------------------
// sam_perturb_kernel:  θ_pert ← θ + ρ · g / ‖g‖₂
// `grad_norm_inv` is computed CPU-side from a single-sync vector reduce
// (compute_sam_grad_norm_device_side in _helpers.h) and passed in.
// Same structure as SG v1.1 (per the build spec).
// ---------------------------------------------------------------------

template <typename ParamT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void sg15_sam_perturb_kernel(
    ParamT*       __restrict__ params,
    const GradT*  __restrict__ grads,
    float rho,
    float grad_norm_inv,
    int64_t n_elements
) {
    static_assert(is_param_dtype<ParamT>::value, "SG15 SAM: invalid ParamT");
    static_assert(is_grad_dtype<GradT>::value,   "SG15 SAM: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "SG15 SAM: FP8 grad with FP32 param requires explicit rescale");

    const float k = rho * grad_norm_inv;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float p = load_as_float(params + i);
        const float g = load_as_float(grads  + i);
        store_from_float(params + i, ptx_fma(k, g, p));
    }
}

// ---------------------------------------------------------------------
// sharpness_restore_kernel:  θ ← θ_pert - ρ · g / ‖g‖₂
// Mirror of sam_perturb with inverted sign.
// ---------------------------------------------------------------------

template <typename ParamT, typename GradT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void sg15_sharpness_restore_kernel(
    ParamT*       __restrict__ params,
    const GradT*  __restrict__ grads,
    float rho,
    float grad_norm_inv,
    int64_t n_elements
) {
    static_assert(is_param_dtype<ParamT>::value, "SG15 restore: invalid ParamT");
    static_assert(is_grad_dtype<GradT>::value,   "SG15 restore: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "SG15 restore: FP8 grad with FP32 param requires explicit rescale");

    const float k = rho * grad_norm_inv;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const float p = load_as_float(params + i);
        const float g = load_as_float(grads  + i);
        store_from_float(params + i, ptx_fma(-k, g, p));
    }
}

// =====================================================================
// Per-tensor scratch slab. Single thread-local cache that grows
// monotonically with the largest tensor seen so far. Layout:
//   [0 .. SG15_SCRATCH_FLOATS)         : scalars[]
//   [pad]                              : e_scratch[N]      (FP32)
//   [N .. 2N)                          : u_scratch[N]      (FP32)
// =====================================================================

struct sg15_scratch_t {
    float*  base;       // device pointer to the slab
    int64_t capacity;   // in elements N (e_scratch and u_scratch each)
};

inline cudaError_t sg15_scratch_ensure(sg15_scratch_t& s, int64_t n_elements) {
    if (n_elements <= s.capacity) return cudaSuccess;
    if (s.base) cudaFree(s.base);
    const size_t scratch_bytes =
        sizeof(float) * (size_t)SG15_SCRATCH_FLOATS +
        sizeof(float) * (size_t)n_elements * 2u;
    cudaError_t err = cudaMalloc(&s.base, scratch_bytes);
    if (err != cudaSuccess) { s.base = nullptr; s.capacity = 0; return err; }
    s.capacity = n_elements;
    return cudaSuccess;
}

inline float* sg15_scratch_scalars(const sg15_scratch_t& s) { return s.base; }
inline float* sg15_scratch_e(const sg15_scratch_t& s) {
    return s.base + SG15_SCRATCH_FLOATS;
}
inline float* sg15_scratch_u(const sg15_scratch_t& s) {
    return s.base + SG15_SCRATCH_FLOATS + s.capacity;
}

// One slab per host thread (CUDA stream is host-thread-bound by default in
// PyTorch). Not freed at exit — torn down with the CUDA context.
inline sg15_scratch_t& sg15_get_tls_scratch() {
    thread_local sg15_scratch_t s = {nullptr, 0};
    return s;
}

// =====================================================================
// Internal H-templated launcher. Hidden width H is a compile-time tag
// so the meta-net inner loop unrolls fully. Specialised for the
// matched-width menu {16, 32, 64, 128} per the build spec.
// =====================================================================

template <typename ParamT, typename StateT, typename GradT, int H>
cudaError_t sg15_launch_fused_step_H(
    ParamT* params, StateT* exp_avg, StateT* exp_avg_sq,
    StateT* sharpness_ema,
    const GradT* perturbed_grads,
    const GradT* grads,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float kappa, float alpha0, float tau, float trust_clip,
    float bc1, float bc2,
    int64_t n_elements, int64_t step_count,
    cudaStream_t stream
) {
    static_assert(dtype_combo_is_valid<ParamT, StateT, GradT>::value,
        "supergrok15: invalid (ParamT, StateT, GradT) combo");
    static_assert((4 * H + 1) <= SG15_META_PHI_MAX_FLOATS,
        "supergrok15: meta_hidden too large for __constant__ budget");

    if (n_elements <= 0) return cudaSuccess;

    // Block size is read from tuned_configs.h (TODO at line 122 to add
    // an SG15-specific table; until then we share the GROKADAMW table).
    const LaunchConfig cfg =
        get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;
    constexpr int MAX_BLOCKS = SG15_REDUCE_GRID_CAP;
    const int64_t needed = (n_elements + block - 1) / block;
    const int grid =
        static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);

    // Ensure scratch is large enough.
    sg15_scratch_t& sc = sg15_get_tls_scratch();
    cudaError_t err = sg15_scratch_ensure(sc, n_elements);
    if (err != cudaSuccess) return err;
    float* scalars   = sg15_scratch_scalars(sc);
    float* e_scratch = sg15_scratch_e(sc);
    float* u_scratch = sg15_scratch_u(sc);

    // Zero the scalars header (FP32 atomics into bins 0..3).
    err = cudaMemsetAsync(scalars, 0,
        sizeof(float) * (size_t)SG15_SCRATCH_FLOATS, stream);
    if (err != cudaSuccess) return err;

    (void) step_count;  // reserved (would seed BF16 SR PRNG if added later)

    // ── Sweep A: non-cooperative; last-block-finished publishes s_t ──
    if (block == 128) {
        sg15_sweep_a_kernel<ParamT, StateT, GradT, 128, H>
            <<<grid, 128, 0, stream>>>(
                perturbed_grads, grads, sharpness_ema,
                e_scratch, scalars, kappa, n_elements);
    } else if (block == 512) {
        sg15_sweep_a_kernel<ParamT, StateT, GradT, 512, H>
            <<<grid, 512, 0, stream>>>(
                perturbed_grads, grads, sharpness_ema,
                e_scratch, scalars, kappa, n_elements);
    } else {
        sg15_sweep_a_kernel<ParamT, StateT, GradT, 256, H>
            <<<grid, 256, 0, stream>>>(
                perturbed_grads, grads, sharpness_ema,
                e_scratch, scalars, kappa, n_elements);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // ── Sweep B: cooperative — fuses Adam + trust-ratio + apply ──────
    void* args[] = {
        (void*)&params, (void*)&exp_avg, (void*)&exp_avg_sq,
        (void*)&perturbed_grads,
        (void*)&e_scratch, (void*)&u_scratch, (void*)&scalars,
        (void*)&lr, (void*)&beta1, (void*)&beta2, (void*)&eps,
        (void*)&weight_decay,
        (void*)&alpha0, (void*)&tau, (void*)&trust_clip,
        (void*)&bc1, (void*)&bc2,
        (void*)&n_elements
    };
    auto kfn = (void*)&sg15_sweep_b_kernel<ParamT, StateT, GradT, 256>;
    int b_block = 256;
    if (block == 128) {
        kfn = (void*)&sg15_sweep_b_kernel<ParamT, StateT, GradT, 128>;
        b_block = 128;
    } else if (block == 512) {
        kfn = (void*)&sg15_sweep_b_kernel<ParamT, StateT, GradT, 512>;
        b_block = 512;
    }

    // Cooperative-launch grid is bounded by occupancy × #SMs.
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
// PUBLIC LAUNCHER: launch_supergrok15_fused_step
// Signature exactly per the build spec — verified against
// csrc/bindings/supergrok15.cpp's per-tensor entry. The binding today
// dispatches to a per-tensor `launch_fused_supergrok15_full_step` in
// namespace sg::sm90 with torch::Tensor args; this template lives in
// sg::sm90::supergrok15 with raw pointers per the spec. Bridging the
// binding to this entry is a follow-up (mirrors the SG11 follow-up).
// =====================================================================

template <typename ParamT, typename StateT, typename GradT>
cudaError_t launch_supergrok15_fused_step(
    ParamT* params, StateT* exp_avg, StateT* exp_avg_sq,
    StateT* sharpness_ema,
    const GradT* perturbed_grads,
    const GradT* grads,
    const float* meta_phi_weights,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float kappa, float alpha0, float tau, float trust_clip,
    float bc1, float bc2,
    int meta_hidden, int meta_layers,
    int64_t n_elements, int64_t step_count,
    cudaStream_t stream
) {
    static_assert(dtype_combo_is_valid<ParamT, StateT, GradT>::value,
        "supergrok15: invalid (ParamT, StateT, GradT) combo");

    if (n_elements <= 0) return cudaSuccess;
    if (meta_layers != 2) {
        // SG v1.5's meta-net φ is a 2-layer MLP (2 → H → 1). meta_layers
        // is forwarded for forward-compat with future deep-φ variants;
        // anything else here is a programmer error.
        return cudaErrorInvalidValue;
    }
    const size_t n_phi = (size_t)(4 * meta_hidden + 1);
    if (n_phi > (size_t)SG15_META_PHI_MAX_FLOATS) return cudaErrorInvalidValue;

    // Bind φ into __constant__ memory for the duration of this launch.
    cudaError_t err = cudaMemcpyToSymbolAsync(
        c_meta_phi, meta_phi_weights, n_phi * sizeof(float),
        /*offset=*/0, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;

    // Dispatch H to the templated path so the inner MLP loop unrolls.
    // Matched menu = {16, 32, 64, 128}. Anything else is rejected
    // (a runtime-H kernel would lose the unroll the smart_grad win
    // depends on).
    switch (meta_hidden) {
        case 16:
            return sg15_launch_fused_step_H<ParamT, StateT, GradT, 16>(
                params, exp_avg, exp_avg_sq, sharpness_ema,
                perturbed_grads, grads,
                lr, beta1, beta2, eps, weight_decay,
                kappa, alpha0, tau, trust_clip, bc1, bc2,
                n_elements, step_count, stream);
        case 32:
            return sg15_launch_fused_step_H<ParamT, StateT, GradT, 32>(
                params, exp_avg, exp_avg_sq, sharpness_ema,
                perturbed_grads, grads,
                lr, beta1, beta2, eps, weight_decay,
                kappa, alpha0, tau, trust_clip, bc1, bc2,
                n_elements, step_count, stream);
        case 64:
            return sg15_launch_fused_step_H<ParamT, StateT, GradT, 64>(
                params, exp_avg, exp_avg_sq, sharpness_ema,
                perturbed_grads, grads,
                lr, beta1, beta2, eps, weight_decay,
                kappa, alpha0, tau, trust_clip, bc1, bc2,
                n_elements, step_count, stream);
        case 128:
            return sg15_launch_fused_step_H<ParamT, StateT, GradT, 128>(
                params, exp_avg, exp_avg_sq, sharpness_ema,
                perturbed_grads, grads,
                lr, beta1, beta2, eps, weight_decay,
                kappa, alpha0, tau, trust_clip, bc1, bc2,
                n_elements, step_count, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

// =====================================================================
// PUBLIC LAUNCHER: launch_supergrok15_sam_perturb_all
// θ_pert ← θ + ρ · g / ‖g‖₂
// =====================================================================

template <typename ParamT, typename GradT>
cudaError_t launch_supergrok15_sam_perturb_all(
    ParamT* params, const GradT* grads,
    float rho, float grad_norm_inv,
    int64_t n_elements, cudaStream_t stream
) {
    static_assert(is_param_dtype<ParamT>::value, "SG15 SAM: invalid ParamT");
    static_assert(is_grad_dtype<GradT>::value,   "SG15 SAM: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "SG15 SAM: FP8 grad with FP32 param requires explicit rescale");

    if (n_elements <= 0) return cudaSuccess;

    const LaunchConfig cfg =
        get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;
    constexpr int MAX_BLOCKS = 8192;
    const int64_t needed = (n_elements + block - 1) / block;
    const int grid =
        static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);

    if (block == 128) {
        sg15_sam_perturb_kernel<ParamT, GradT, 128>
            <<<grid, 128, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    } else if (block == 512) {
        sg15_sam_perturb_kernel<ParamT, GradT, 512>
            <<<grid, 512, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    } else {
        sg15_sam_perturb_kernel<ParamT, GradT, 256>
            <<<grid, 256, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    }
    return cudaGetLastError();
}

// =====================================================================
// PUBLIC LAUNCHER: launch_supergrok15_sharpness_restore_all
// θ ← θ_pert - ρ · g / ‖g‖₂
// =====================================================================

template <typename ParamT, typename GradT>
cudaError_t launch_supergrok15_sharpness_restore_all(
    ParamT* params, const GradT* grads,
    float rho, float grad_norm_inv,
    int64_t n_elements, cudaStream_t stream
) {
    static_assert(is_param_dtype<ParamT>::value, "SG15 restore: invalid ParamT");
    static_assert(is_grad_dtype<GradT>::value,   "SG15 restore: invalid GradT");
    static_assert(is_coherent_combo<ParamT, GradT>::value,
        "SG15 restore: FP8 grad with FP32 param requires explicit rescale");

    if (n_elements <= 0) return cudaSuccess;

    const LaunchConfig cfg =
        get_grokadamw_config(/*arch=*/90, static_cast<int>(n_elements));
    const int block = cfg.block_size;
    constexpr int MAX_BLOCKS = 8192;
    const int64_t needed = (n_elements + block - 1) / block;
    const int grid =
        static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);

    if (block == 128) {
        sg15_sharpness_restore_kernel<ParamT, GradT, 128>
            <<<grid, 128, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    } else if (block == 512) {
        sg15_sharpness_restore_kernel<ParamT, GradT, 512>
            <<<grid, 512, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    } else {
        sg15_sharpness_restore_kernel<ParamT, GradT, 256>
            <<<grid, 256, 0, stream>>>(params, grads, rho, grad_norm_inv, n_elements);
    }
    return cudaGetLastError();
}

}}} // namespace sg::sm90::supergrok15
