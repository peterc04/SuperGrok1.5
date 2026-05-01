#pragma once
// SuperGrok v1.1 -- Device-function templates for sm_90 (Hopper).
// Migrated from csrc/kernels/cuda/sm_90/supergrok11_sm90.cu
//
// Four operations (cosine similarity gating variant):
//   1. mu_metanet: EMA update + element-wise MLP(2->H->1) with GELU
//   2. adam_cosine_gate: Adam moments + progressive WD + step
//   3. sam_perturb: worst-case param perturbation
//   4. sharpness_restore: |sam_grad - grad| + param restore

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace sm90 {

// =========================================================================
//  SG11 mu EMA + meta-net inference
//  MLP: Linear(2,H) -> GELU -> Linear(H,1), skip connection
//  Weights must be pre-loaded into shared memory.
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void sg11_mu_metanet(
    scalar_t* __restrict__ mu,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharp,
    scalar_t* __restrict__ smart_grad,
    const float alpha,
    const float* __restrict__ sW1,   // [H*2] in shared memory
    const float* __restrict__ sb1,   // [H] in shared memory
    const float* __restrict__ sW2,   // [H] in shared memory
    const float* __restrict__ sb2,   // [1] in shared memory
    const float rescale,
    const int idx,
    const int H
) {
    // TODO: Port full implementation from kernel
}

// =========================================================================
//  SG11 Adam with cosine-gating (lamb_eff precomputed on host)
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void sg11_adam_cosine_gate(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const scalar_t* __restrict__ smart_grad,
    const float lamb_eff,
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

// =========================================================================
//  SG11 SAM perturbation + restore
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void sg11_sam_perturb(
    scalar_t* __restrict__ param,
    scalar_t* __restrict__ backup,
    const scalar_t* __restrict__ grad,
    const float rho_over_norm,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

template <typename scalar_t>
__device__ __forceinline__ void sg11_sharpness_restore(
    scalar_t* __restrict__ param,
    scalar_t* __restrict__ sharpness,
    const scalar_t* __restrict__ backup,
    const scalar_t* __restrict__ sam_grad,
    const scalar_t* __restrict__ normal_grad,
    const int idx
) {
    // TODO: Port full implementation from kernel
}

}}} // namespace sg::device::sm90
