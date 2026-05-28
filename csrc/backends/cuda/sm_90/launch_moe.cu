// CUDA sm_90 launch stubs for MoE (Mixture of Experts).
//
// These are placeholder implementations that throw at runtime.
// The actual sm_90 kernels have not been implemented yet.

#include <torch/extension.h>
#include <stdexcept>

namespace sg {
namespace sm90 {

void moe_dynamic_expert_load(
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor active_mask,
    torch::Tensor smem_w1, torch::Tensor smem_b1,
    torch::Tensor smem_w2, torch::Tensor smem_b2) {
  throw std::runtime_error(
      "moe_dynamic_expert_load: sm_90 kernel not yet implemented.");
}

torch::Tensor moe_dynamic_expert_fwd(
    torch::Tensor input, torch::Tensor expert_indices,
    torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor output) {
  throw std::runtime_error(
      "moe_dynamic_expert_fwd: sm_90 kernel not yet implemented.");
  return torch::Tensor{};
}

void moe_dynamic_expert_bwd(
    torch::Tensor d_output, torch::Tensor input,
    torch::Tensor expert_indices, torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor d_input, torch::Tensor d_expert_w1,
    torch::Tensor d_expert_b1, torch::Tensor d_expert_w2,
    torch::Tensor d_expert_b2) {
  throw std::runtime_error(
      "moe_dynamic_expert_bwd: sm_90 kernel not yet implemented.");
}

void moe_filter_active_params(
    torch::Tensor params, torch::Tensor grads,
    torch::Tensor state_m, torch::Tensor state_v,
    torch::Tensor param_to_expert, torch::Tensor expert_active,
    torch::Tensor compact_params, torch::Tensor compact_grads,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices, torch::Tensor compact_count,
    int total_params) {
  throw std::runtime_error(
      "moe_filter_active_params: sm_90 kernel not yet implemented.");
}

void moe_scan_compacted(
    torch::Tensor compact_x, torch::Tensor compact_dt,
    torch::Tensor compact_B, torch::Tensor compact_C,
    torch::Tensor A_log, torch::Tensor D_param,
    torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor final_state,
    torch::Tensor initial_state,
    int compact_N, int d_inner, int d_state) {
  throw std::runtime_error(
      "moe_scan_compacted: sm_90 kernel not yet implemented.");
}

void moe_scatter_results(
    torch::Tensor compact_params,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices,
    torch::Tensor params,
    torch::Tensor state_m, torch::Tensor state_v,
    int compact_N) {
  throw std::runtime_error(
      "moe_scatter_results: sm_90 kernel not yet implemented.");
}

void moe_count_expert_activations(
    torch::Tensor gate_logits, torch::Tensor expert_counts,
    float threshold, int N, int num_experts) {
  throw std::runtime_error(
      "moe_count_expert_activations: sm_90 kernel not yet implemented.");
}

torch::Tensor moe_compute_load_balance_loss(
    torch::Tensor expert_counts, torch::Tensor gate_logits,
    int N, int num_experts) {
  throw std::runtime_error(
      "moe_compute_load_balance_loss: sm_90 kernel not yet implemented.");
  return torch::Tensor{};
}

void moe_apply_frequency_scaling(
    torch::Tensor expert_counts, torch::Tensor lr_scale,
    int num_experts, int total_activations,
    float min_scale, float max_scale, float smoothing) {
  throw std::runtime_error(
      "moe_apply_frequency_scaling: sm_90 kernel not yet implemented.");
}

} // namespace sm90
} // namespace sg
