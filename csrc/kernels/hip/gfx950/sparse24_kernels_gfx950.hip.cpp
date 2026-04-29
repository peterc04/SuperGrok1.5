/*
 * SuperGrok v2 — CDNA4 Structured 2:4 Sparsity Kernels (gfx950, MI350X)
 *
 * Split out from the former cdna4_kernels_gfx950.hip.cpp monolith.
 * Contains the 4 structured 2:4 sparsity kernels and host launchers:
 *   1. cdna4_sparse24_select_kernel       — pick top-2 of 4 by magnitude
 *   2. cdna4_sparse24_apply_mask_kernel   — zero pruned positions in grad
 *   3. cdna4_sparse24_project_kernel      — zero pruned positions in state
 *   4. cdna4_sparse24_densify_kernel      — reconstruct dense from sparse
 *
 * These kernels do not depend on FP4/FP6 helpers, but the file still
 * includes the shared header for namespace-consistency and to keep
 * future fused additions painless.
 *
 * Math is unchanged from the monolith.
 */

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include "platform.h"
#include "../../common/fp4_helpers.hip.h"

namespace sg { namespace gfx950 {

// ═══════════════════════════════════════════════════════════════════════
//  Kernel 9: Structured 2:4 Sparsity Select
//
//  From a dense [N] parameter vector, select the 2 largest-magnitude
//  values out of every group of 4 consecutive elements.
//
//  Output: sparse_values [N/2] — the 2 kept values per group
//          metadata [N/4] — 2-bit mask per group (which 2 of 4 kept)
//
//  Metadata encoding: each byte holds masks for 4 groups (2 bits each).
//  For a group of 4 elements [a,b,c,d], the 2-bit mask encodes which
//  pair is kept using a 6-entry lookup (C(4,2) = 6 combinations).
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_sparse24_select_kernel(
    const float*    __restrict__ dense,           // [N] — must be multiple of 4
    float*          __restrict__ sparse_values,   // [N/2]
    uint8_t*        __restrict__ metadata,        // [N/4] (4 bits per group: bitmask of which 2 kept)
    int             N
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = N / 4;

    if (group_idx >= num_groups) return;

    int base = group_idx * 4;

    // Load 4 elements
    float vals[4];
    float abs_vals[4];
    for (int i = 0; i < 4; i++) {
        vals[i] = dense[base + i];
        abs_vals[i] = fabsf(vals[i]);
    }

    // Find the 2 largest by magnitude using a sorting network
    // We need indices of the top-2
    int idx[4] = {0, 1, 2, 3};

    // Bubble the 2 smallest to positions 0,1 (keep positions 2,3 = top 2)
    // Sort by ascending absolute value
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (abs_vals[idx[i]] > abs_vals[idx[j]]) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
        }
    }

    // Top 2 are idx[2] and idx[3] (largest magnitude)
    int keep0 = (idx[2] < idx[3]) ? idx[2] : idx[3];  // lower index first
    int keep1 = (idx[2] < idx[3]) ? idx[3] : idx[2];

    // Store sparse values (2 per group)
    int sparse_base = group_idx * 2;
    sparse_values[sparse_base + 0] = vals[keep0];
    sparse_values[sparse_base + 1] = vals[keep1];

    // Encode metadata as 4-bit bitmask: bit i set if position i is kept
    uint8_t mask = (1u << keep0) | (1u << keep1);
    metadata[group_idx] = mask;
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 10: Apply 2:4 Sparsity Mask to Gradients
//
//  Zero out the 2 pruned positions in each group of 4 gradient elements,
//  using the metadata mask from the select kernel.
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_sparse24_apply_mask_kernel(
    float*          __restrict__ grad,            // [N] — modified in-place
    const uint8_t*  __restrict__ metadata,        // [N/4]
    int             N
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = N / 4;

    if (group_idx >= num_groups) return;

    int base = group_idx * 4;
    uint8_t mask = metadata[group_idx];

    // Zero out pruned positions (where bit is not set)
    for (int i = 0; i < 4; i++) {
        if (!(mask & (1u << i))) {
            grad[base + i] = 0.0f;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 11: Project Optimizer State Through 2:4 Mask
//
//  Zero the optimizer state (exp_avg, exp_avg_sq) at pruned positions.
//  Only the 2 active positions per group retain their state.
//  This prevents stale momentum from accumulating at pruned positions.
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_sparse24_project_kernel(
    float*          __restrict__ exp_avg,         // [N] — modified in-place
    float*          __restrict__ exp_avg_sq,      // [N] — modified in-place
    const uint8_t*  __restrict__ metadata,        // [N/4]
    int             N
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = N / 4;

    if (group_idx >= num_groups) return;

    int base = group_idx * 4;
    uint8_t mask = metadata[group_idx];

    for (int i = 0; i < 4; i++) {
        if (!(mask & (1u << i))) {
            exp_avg[base + i]    = 0.0f;
            exp_avg_sq[base + i] = 0.0f;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 12: Densify from Sparse 2:4
//
//  Reconstruct dense [N] output from sparse values [N/2] + metadata.
//  Pruned positions are filled with zero.
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_sparse24_densify_kernel(
    const float*    __restrict__ sparse_values,   // [N/2]
    const uint8_t*  __restrict__ metadata,        // [N/4]
    float*          __restrict__ dense,           // [N]
    int             N
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = N / 4;

    if (group_idx >= num_groups) return;

    int base = group_idx * 4;
    int sparse_base = group_idx * 2;
    uint8_t mask = metadata[group_idx];

    // Scatter sparse values into their original positions
    int sparse_idx = 0;
    for (int i = 0; i < 4; i++) {
        if (mask & (1u << i)) {
            dense[base + i] = sparse_values[sparse_base + sparse_idx];
            sparse_idx++;
        } else {
            dense[base + i] = 0.0f;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Host Launchers for 2:4 Sparsity Kernels
// ═══════════════════════════════════════════════════════════════════════

void cdna4_sparse24_select(
    torch::Tensor dense, torch::Tensor sparse_values, torch::Tensor metadata, int N
) {
    int num_groups = N / 4;
    dim3 grid((num_groups + 255) / 256);
    dim3 block(256);
    cdna4_sparse24_select_kernel<<<grid, block>>>(
        dense.data_ptr<float>(), sparse_values.data_ptr<float>(),
        metadata.data_ptr<uint8_t>(), N
    );
}

void cdna4_sparse24_apply_mask(
    torch::Tensor grad, torch::Tensor metadata, int N
) {
    int num_groups = N / 4;
    dim3 grid((num_groups + 255) / 256);
    dim3 block(256);
    cdna4_sparse24_apply_mask_kernel<<<grid, block>>>(
        grad.data_ptr<float>(), metadata.data_ptr<uint8_t>(), N
    );
}

void cdna4_sparse24_project(
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor metadata, int N
) {
    int num_groups = N / 4;
    dim3 grid((num_groups + 255) / 256);
    dim3 block(256);
    cdna4_sparse24_project_kernel<<<grid, block>>>(
        exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
        metadata.data_ptr<uint8_t>(), N
    );
}

void cdna4_sparse24_densify(
    torch::Tensor sparse_values, torch::Tensor metadata, torch::Tensor dense, int N
) {
    int num_groups = N / 4;
    dim3 grid((num_groups + 255) / 256);
    dim3 block(256);
    cdna4_sparse24_densify_kernel<<<grid, block>>>(
        sparse_values.data_ptr<float>(), metadata.data_ptr<uint8_t>(),
        dense.data_ptr<float>(), N
    );
}

} } // namespace sg::gfx950
