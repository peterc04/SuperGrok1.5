#pragma once
// MoE Deep Optimization -- Device-function templates for sm_90 (Hopper).
// Migrated from csrc/kernels/cuda/sm_90/moe_sm90.cu
//
// Nine operations for advanced Mixture-of-Experts:
//   Dynamic Expert Loading (1-3):
//     1. dynamic_expert_load: determine active experts, load weights to smem
//     2. dynamic_expert_fwd: forward through dynamically-loaded experts
//     3. dynamic_expert_bwd: backward through dynamic expert loading
//   Filtered Scan with Param Compaction (4-6):
//     4. filter_active_params: compact active params from expert assignments
//     5. scan_compacted: Mamba scan on compacted param subset
//     6. scatter_results: scatter compacted results back to full array
//   Expert Activation Frequency Counter (7-9):
//     7. count_expert_activations: atomic count of expert selections
//     8. compute_load_balance_loss: auxiliary load-balancing loss
//     9. apply_frequency_scaling: scale expert LRs by frequency

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace sm90 {

// =========================================================================
//  MoE dynamic expert load: determine active experts from gate logits
// =========================================================================

__device__ __forceinline__ void moe_dynamic_expert_load(
    const float* __restrict__ gate_logits,   // [num_experts] for one sample
    int* __restrict__ active_ids,            // output: active expert ids
    int* __restrict__ num_active,            // output: count
    const float threshold,
    const int num_experts
) {
    // TODO: Port full implementation from kernel
}

// =========================================================================
//  MoE dynamic expert forward: ReLU MLP through loaded experts
// =========================================================================

__device__ __forceinline__ void moe_dynamic_expert_fwd_element(
    const float* __restrict__ input,         // [input_dim]
    const float* __restrict__ gate_weights,  // [num_active]
    const float* __restrict__ loaded_W1,     // [expert_dim, input_dim]
    const float* __restrict__ loaded_b1,     // [expert_dim]
    const float* __restrict__ loaded_W2,     // [input_dim, expert_dim]
    const float* __restrict__ loaded_b2,     // [input_dim]
    float* __restrict__ output,              // [input_dim]
    const int input_dim,
    const int expert_dim,
    const int num_active
) {
    // TODO: Port full implementation from kernel
}

// =========================================================================
//  MoE filter active params: compact active params for scan
// =========================================================================

__device__ __forceinline__ void moe_filter_active_params(
    const float* __restrict__ all_params,
    const int* __restrict__ expert_assignment,
    float* __restrict__ compacted_params,
    int* __restrict__ compacted_indices,
    const int idx,
    const int target_expert
) {
    // TODO: Port full implementation from kernel
}

// =========================================================================
//  MoE frequency scaling: scale expert LR by activation frequency
// =========================================================================

__device__ __forceinline__ void moe_apply_frequency_scaling(
    float* __restrict__ expert_lr,
    const int* __restrict__ activation_counts,
    const float base_lr,
    const float total_samples,
    const int expert_id,
    const int num_experts
) {
    // TODO: Port full implementation from kernel
}

}}} // namespace sg::device::sm90
