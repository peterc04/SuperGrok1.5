#pragma once
// LookSAM -- Device-function templates for sm_90 (Hopper).
// Migrated from csrc/kernels/cuda/sm_90/looksam_sm90.cu
//
// Four operations:
//   1. direction: v_dir = (sam_grad - normal_grad) * inv_norm
//   2. adjust: grad += la_times_gnorm * v_dir
//   3. perturb: param += rho_over_norm * grad
//   4. restore: param = backup

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace sm90 {

// =========================================================================
//  LookSAM direction computation
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void looksam_direction(
    scalar_t* __restrict__ v_dir,
    const scalar_t* __restrict__ sam_grad,
    const scalar_t* __restrict__ normal_grad,
    const float inv_norm,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

// =========================================================================
//  LookSAM gradient adjustment
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void looksam_adjust(
    scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ v_dir,
    const float la_times_gnorm,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

// =========================================================================
//  LookSAM fused direction + adjust (v_dir stays in register)
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void looksam_direction_adjust_fused(
    scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sam_grad,
    const scalar_t* __restrict__ normal_grad,
    const float inv_norm,
    const float la_times_gnorm,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

// =========================================================================
//  LookSAM perturbation + restore
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void looksam_perturb(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    const float rho_over_norm,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

template <typename scalar_t>
__device__ __forceinline__ void looksam_restore(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ backup,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

}}} // namespace sg::device::sm90
