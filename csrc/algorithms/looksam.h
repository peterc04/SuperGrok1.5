#pragma once
// LookSAM — vendor-neutral algorithm header.
//
// Reference: Liu et al. 2022, "Towards Efficient and Scalable Sharpness-
// Aware Minimization" (https://arxiv.org/abs/2203.02714), Algorithm 1
// (periodic-perturbation variant — SAM gradient is recomputed every k
// steps, then reused as a cached direction for intervening steps).
//
// AdamW with periodic Sharpness-Aware Minimization (every k steps).
//
// Four operations:
//   (1) perturb:           p_pert = p + rho * (g / ||g||)
//   (2) restore:           p = p_pert - rho * (g / ||g||)
//   (3) direction adjust:  on SAM step,    sam_dir = g_sam - g
//                          on normal step, g_adj   = (1-alpha)*g + alpha*sam_dir
//   (4) standard AdamW update with g_adj.
//
// Calling convention: bc1, bc2 are passed un-inverted —
//   bc1 = 1 - beta1^t,  bc2 = 1 - beta2^t.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

// Step (1) — perturb in gradient direction by rho / ||g||.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void looksam_perturb_step(
    ParamT* __restrict__ param,
    ParamT* __restrict__ backup,
    const GradT* __restrict__ grad,
    const float scale,        // rho / (||g|| + eps), precomputed
    const int idx
) {
    const ParamT p = param[idx];
    backup[idx] = p;
    param[idx] = static_cast<ParamT>(static_cast<float>(p) + scale * static_cast<float>(grad[idx]));
}

// Step (2) — restore parameters from backup.
template <typename ParamT>
__device__ __forceinline__ void looksam_restore_step(
    ParamT* __restrict__ param,
    const ParamT* __restrict__ backup,
    const int idx
) {
    param[idx] = backup[idx];
}

// Step (3) on SAM step: cache the gradient difference.
template <typename GradT>
__device__ __forceinline__ void looksam_set_direction(
    float* __restrict__ sam_dir,
    const GradT* __restrict__ grad_sam,
    const GradT* __restrict__ grad_orig,
    const int idx
) {
    sam_dir[idx] = static_cast<float>(grad_sam[idx]) - static_cast<float>(grad_orig[idx]);
}

// Step (3) on normal step + Step (4): blend cached SAM direction and run Adam.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void looksam_apply_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const float* __restrict__ sam_dir,
    const GradT* __restrict__ grad,
    const float alpha,         // SAM blend strength
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int idx
) {
    const float g = static_cast<float>(grad[idx]);
    const float p = static_cast<float>(param[idx]);
    const float g_adj = (1.0f - alpha) * g + alpha * sam_dir[idx];

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * g_adj;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g_adj * g_adj;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    // bc1, bc2 un-inverted (= 1 - beta^t): divide for bias correction.
    const float update = (m / bc1) / (sqrtf(v / bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

}} // namespace sg::algorithms
