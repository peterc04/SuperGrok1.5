#pragma once
// LookSAM -- Device-function templates for gfx942 (CDNA3 / MI300X).
// Migrated from csrc/kernels/hip/gfx942/looksam_gfx942.hip.cpp
//
// Four operations:
//   1. direction: v_dir = (sam_grad - normal_grad) * inv_norm
//   2. adjust: grad += la_times_gnorm * v_dir
//   3. perturb: param += rho_over_norm * grad
//   4. restore: param = backup

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace gfx942 {

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

template <typename scalar_t>
__device__ __forceinline__ void looksam_adjust(
    scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ v_dir,
    const float la_times_gnorm,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

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

}}} // namespace sg::device::gfx942
