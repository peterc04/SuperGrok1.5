// bindings/moe.cpp — runtime dispatch to per-arch MoE launchers.
//
// MoE covers dynamic expert load/forward/backward, active-param
// filtering, compacted scan, scatter, activation counting, load-balance
// loss, and frequency-based LR scaling. Functions in the canonical
// moe_<arch>.cu files are NOT prefixed with `launch_`; they are direct
// host-side wrappers, so the SG_DISPATCH calls below match those names.

#include "_dispatch_macro.h"

#include <vector>

namespace sg {

#define DECLARE_MOE(NS)                                                       \
    namespace NS {                                                            \
        void moe_dynamic_expert_load(                                         \
            torch::Tensor expert_w1, torch::Tensor expert_b1,                 \
            torch::Tensor expert_w2, torch::Tensor expert_b2,                 \
            torch::Tensor active_mask,                                        \
            torch::Tensor smem_w1, torch::Tensor smem_b1,                     \
            torch::Tensor smem_w2, torch::Tensor smem_b2);                    \
        torch::Tensor moe_dynamic_expert_fwd(                                 \
            torch::Tensor input, torch::Tensor expert_indices,                \
            torch::Tensor routing_weights,                                    \
            torch::Tensor expert_w1, torch::Tensor expert_b1,                 \
            torch::Tensor expert_w2, torch::Tensor expert_b2,                 \
            torch::Tensor output);                                            \
        void moe_dynamic_expert_bwd(                                          \
            torch::Tensor d_output, torch::Tensor input,                      \
            torch::Tensor expert_indices, torch::Tensor routing_weights,      \
            torch::Tensor expert_w1, torch::Tensor expert_b1,                 \
            torch::Tensor expert_w2, torch::Tensor expert_b2,                 \
            torch::Tensor d_input, torch::Tensor d_expert_w1,                 \
            torch::Tensor d_expert_b1, torch::Tensor d_expert_w2,             \
            torch::Tensor d_expert_b2);                                       \
        void moe_filter_active_params(                                        \
            torch::Tensor params, torch::Tensor grads,                        \
            torch::Tensor state_m, torch::Tensor state_v,                     \
            torch::Tensor param_to_expert, torch::Tensor expert_active,      \
            torch::Tensor compact_params, torch::Tensor compact_grads,        \
            torch::Tensor compact_state_m, torch::Tensor compact_state_v,    \
            torch::Tensor scatter_indices, torch::Tensor compact_count,      \
            int total_params);                                                \
        void moe_scan_compacted(                                              \
            torch::Tensor compact_x, torch::Tensor compact_dt,                \
            torch::Tensor compact_B, torch::Tensor compact_C,                 \
            torch::Tensor A_log, torch::Tensor D_param,                       \
            torch::Tensor rope_freq,                                          \
            torch::Tensor scan_output, torch::Tensor final_state,             \
            torch::Tensor initial_state,                                      \
            int compact_N, int d_inner, int d_state);                         \
        void moe_scatter_results(                                             \
            torch::Tensor compact_params,                                     \
            torch::Tensor compact_state_m, torch::Tensor compact_state_v,    \
            torch::Tensor scatter_indices,                                    \
            torch::Tensor params,                                             \
            torch::Tensor state_m, torch::Tensor state_v,                     \
            int compact_N);                                                   \
        void moe_count_expert_activations(                                    \
            torch::Tensor gate_logits, torch::Tensor expert_counts,           \
            float threshold, int N, int num_experts);                         \
        torch::Tensor moe_compute_load_balance_loss(                          \
            torch::Tensor expert_counts, torch::Tensor gate_logits,           \
            int N, int num_experts);                                          \
        void moe_apply_frequency_scaling(                                     \
            torch::Tensor expert_counts, torch::Tensor lr_scale,              \
            int num_experts, int total_activations,                           \
            float min_scale, float max_scale, float smoothing);               \
    }

DECLARE_MOE(sm80) DECLARE_MOE(sm89) DECLARE_MOE(sm90)
DECLARE_MOE(sm100) DECLARE_MOE(sm103) DECLARE_MOE(sm120)
DECLARE_MOE(gfx942) DECLARE_MOE(gfx950)
#undef DECLARE_MOE

// ---------------------------------------------------------------------
// Public Python entry points. Names match the m.def() registrations
// from the deleted csrc/common/ops.cpp.
// ---------------------------------------------------------------------

void moe_dynamic_expert_load(
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor active_mask,
    torch::Tensor smem_w1, torch::Tensor smem_b1,
    torch::Tensor smem_w2, torch::Tensor smem_b2)
{
    SG_DISPATCH(moe_dynamic_expert_load,
        expert_w1, expert_b1, expert_w2, expert_b2, active_mask,
        smem_w1, smem_b1, smem_w2, smem_b2);
}

torch::Tensor moe_dynamic_expert_fwd(
    torch::Tensor input, torch::Tensor expert_indices,
    torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor output)
{
    SG_DISPATCH(moe_dynamic_expert_fwd,
        input, expert_indices, routing_weights,
        expert_w1, expert_b1, expert_w2, expert_b2, output);
}

void moe_dynamic_expert_bwd(
    torch::Tensor d_output, torch::Tensor input,
    torch::Tensor expert_indices, torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor d_input, torch::Tensor d_expert_w1,
    torch::Tensor d_expert_b1, torch::Tensor d_expert_w2,
    torch::Tensor d_expert_b2)
{
    SG_DISPATCH(moe_dynamic_expert_bwd,
        d_output, input, expert_indices, routing_weights,
        expert_w1, expert_b1, expert_w2, expert_b2,
        d_input, d_expert_w1, d_expert_b1, d_expert_w2, d_expert_b2);
}

void moe_filter_active_params(
    torch::Tensor params, torch::Tensor grads,
    torch::Tensor state_m, torch::Tensor state_v,
    torch::Tensor param_to_expert, torch::Tensor expert_active,
    torch::Tensor compact_params, torch::Tensor compact_grads,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices, torch::Tensor compact_count,
    int total_params)
{
    SG_DISPATCH(moe_filter_active_params,
        params, grads, state_m, state_v, param_to_expert, expert_active,
        compact_params, compact_grads, compact_state_m, compact_state_v,
        scatter_indices, compact_count, total_params);
}

void moe_scan_compacted(
    torch::Tensor compact_x, torch::Tensor compact_dt,
    torch::Tensor compact_B, torch::Tensor compact_C,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor final_state,
    torch::Tensor initial_state,
    int compact_N, int d_inner, int d_state)
{
    SG_DISPATCH(moe_scan_compacted,
        compact_x, compact_dt, compact_B, compact_C,
        A_log, D_param, rope_freq,
        scan_output, final_state, initial_state,
        compact_N, d_inner, d_state);
}

void moe_scatter_results(
    torch::Tensor compact_params,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices,
    torch::Tensor params,
    torch::Tensor state_m, torch::Tensor state_v,
    int compact_N)
{
    SG_DISPATCH(moe_scatter_results,
        compact_params, compact_state_m, compact_state_v,
        scatter_indices, params, state_m, state_v, compact_N);
}

void moe_count_expert_activations(
    torch::Tensor gate_logits, torch::Tensor expert_counts,
    float threshold, int N, int num_experts)
{
    SG_DISPATCH(moe_count_expert_activations,
        gate_logits, expert_counts, threshold, N, num_experts);
}

torch::Tensor moe_compute_load_balance_loss(
    torch::Tensor expert_counts, torch::Tensor gate_logits,
    int N, int num_experts)
{
    SG_DISPATCH(moe_compute_load_balance_loss,
        expert_counts, gate_logits, N, num_experts);
}

void moe_apply_frequency_scaling(
    torch::Tensor expert_counts, torch::Tensor lr_scale,
    int num_experts, int total_activations,
    float min_scale, float max_scale, float smoothing)
{
    SG_DISPATCH(moe_apply_frequency_scaling,
        expert_counts, lr_scale, num_experts, total_activations,
        min_scale, max_scale, smoothing);
}

} // namespace sg
