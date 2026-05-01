#pragma once
// MoE Deep Optimization -- Device-function templates for gfx942 (CDNA3 / MI300X).
// Migrated from csrc/kernels/hip/gfx942/moe_gfx942.hip.cpp
//
// Nine operations for advanced Mixture-of-Experts (see moe_sm90.cuh for details).

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace gfx942 {

__device__ __forceinline__ void moe_dynamic_expert_load(
    const float* __restrict__ gate_logits,
    int* __restrict__ active_ids,
    int* __restrict__ num_active,
    const float threshold,
    const int num_experts
) {
    // TODO: Port full implementation from kernel
}

__device__ __forceinline__ void moe_dynamic_expert_fwd_element(
    const float* __restrict__ input,
    const float* __restrict__ gate_weights,
    const float* __restrict__ loaded_W1,
    const float* __restrict__ loaded_b1,
    const float* __restrict__ loaded_W2,
    const float* __restrict__ loaded_b2,
    float* __restrict__ output,
    const int input_dim,
    const int expert_dim,
    const int num_active
) {
    // TODO: Port full implementation from kernel
}

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

}}} // namespace sg::device::gfx942
