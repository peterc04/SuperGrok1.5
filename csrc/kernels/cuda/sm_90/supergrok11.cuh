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
//                                      sweep B: trust-ratio reduction
//                                               + apply (2 outputs + write)
//    2. sam_perturb_all           — per-tensor θ ← θ + ρ·g/||g||
//    3. sharpness_restore_all     — per-tensor θ ← θ_pert - ρ·g/||g||
//
//  Reduction strategy: cooperative-groups grid finish via warp-reduce +
//  one atomicAdd per warp into a small device scratch buffer. The scratch
//  is then read back as `__restrict__` input by the apply pass in sweep
//  B (single grid_sync would also work, but cooperative-groups requires a
//  cooperative launch flag — the two-kernel-call pattern is portable and
//  matches the rest of the sm_90 file family in this tree).
//
//  Dtype matrix (instantiated in the .cu TU):
//    ParamT in {float, __nv_bfloat16, __half}                     (3)
//    StateT in {float, __nv_bfloat16}                             (2)
//    GradT  in {float, __nv_bfloat16, __half,
//               __nv_fp8_e4m3, __nv_fp8_e5m2}                     (5)
//
//  Coherence rules:
//    - FP8 GradT is only allowed with FP32 or BF16 StateT (FP8 cannot
//      represent state safely; the static_assert guards this).
//    - All math is FP32; only loads/stores are typed.
//
//  Meta-net φ is bound through __constant__ memory by the launcher.
//  Hidden width H is a runtime template tag (specialised in .cu) so the
//  inner loop is fully unrolled. ptx_expert_mlp_forward<H> from
//  utils.cuh is reused when the hidden width matches its single-input
//  specialisation; SG v1.1's MLP is two-input (ĝ, s_t), so we open-code
//  the dot-product with PTX FMAs for the inner loop instead.
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/ptx_intrinsics.cuh"
#include "csrc/common/tuned_configs.h"
#include "csrc/device/optimizers/sm_90/supergrok11_sm90.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
  #include <cuda_fp8.h>
#endif

#include <cuda_runtime.h>
#include <type_traits>
#include <cstdint>

namespace sg { namespace sm90 { namespace supergrok11 {

// ---------------------------------------------------------------------
// Launch parameters drawn from tuned_configs.h::DEFAULT_CONFIG. The
// SG11 kernel-specific table is not yet populated (see the TODO at
// tuned_configs.h:122); we therefore mirror the DEFAULT_CONFIG fields
// statically here. The .cu launcher reads them at compile time.
// ---------------------------------------------------------------------

constexpr int SG11_BLOCK_SIZE          = 256;
constexpr int SG11_MIN_BLOCKS_PER_SM   = 8;
constexpr int SG11_REDUCE_GRID_CAP     = 1024;   // persistent reduce grid
constexpr int SG11_META_PHI_MAX_FLOATS = 2048;   // __constant__ budget

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

// Coherence: FP8 grads still need FP32-or-BF16 state. (FP8 state would
// catastrophically truncate Adam moments.) Both StateT options are
// already FP32/BF16, so the only invalid case to forbid is "everything
// is FP8" — guarded at the kernel boundary.
template <typename ParamT, typename StateT, typename GradT>
struct dtype_combo_is_valid
    : std::integral_constant<bool,
        is_param_dtype<ParamT>::value &&
        is_state_dtype<StateT>::value &&
        is_grad_dtype<GradT>::value> {};

}}} // namespace sg::sm90::supergrok11
