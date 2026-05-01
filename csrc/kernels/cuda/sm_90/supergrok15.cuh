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
    const int lane = threadIdx.x & 31;
    a = warp_reduce_sum(a, 32, lane);
    if (lane == 0) atomicAdd(&scalars[0], a);
}

// Block-wide pair reduction → 2 atomicAdds per warp into a caller-supplied
// pair pointer (caller passes &scalars[2] for sweep B phase 1).
template <int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce2_atomic(
    float a, float b, float* __restrict__ pair   // pair[0..1]
) {
    const int lane = threadIdx.x & 31;
    a = warp_reduce_sum(a, 32, lane);
    b = warp_reduce_sum(b, 32, lane);
    if (lane == 0) {
        atomicAdd(&pair[0], a);
        atomicAdd(&pair[1], b);
    }
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
