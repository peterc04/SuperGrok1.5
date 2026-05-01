#pragma once
// SuperGrok v1.5 -- Device-function templates for gfx942 (CDNA3 / MI300X).
// Migrated from csrc/kernels/hip/gfx942/supergrok15_gfx942.hip.cpp
//
// Four operations:
//   1. mu_metanet: EMA update + element-wise MLP(2->H->1) with GELU
//   2. adam_decay: gating blend + Adam moments + progressive WD + step
//   3. sam_perturb: worst-case param perturbation
//   4. sharpness_restore: |sam_grad - grad| + param restore

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace gfx942 {

template <typename scalar_t>
__device__ __forceinline__ void sg15_mu_metanet(
    scalar_t* __restrict__ mu,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharp,
    scalar_t* __restrict__ smart_grad,
    const float alpha,
    const float* __restrict__ sW1,
    const float* __restrict__ sb1,
    const float* __restrict__ sW2,
    const float* __restrict__ sb2,
    const float rescale,
    const int idx,
    const int H
) {
    // TODO: Port full implementation from kernel
}

template <typename scalar_t>
__device__ __forceinline__ void sg15_adam_decay(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const scalar_t* __restrict__ smart_grad,
    const scalar_t* __restrict__ normal_grad,
    const float gate_signal,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

template <typename scalar_t>
__device__ __forceinline__ void sg15_sam_perturb(
    scalar_t* __restrict__ param,
    scalar_t* __restrict__ backup,
    const scalar_t* __restrict__ grad,
    const float rho_over_norm,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

template <typename scalar_t>
__device__ __forceinline__ void sg15_sharpness_restore(
    scalar_t* __restrict__ param,
    scalar_t* __restrict__ sharpness,
    const scalar_t* __restrict__ backup,
    const scalar_t* __restrict__ sam_grad,
    const scalar_t* __restrict__ normal_grad,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

}}} // namespace sg::device::gfx942
