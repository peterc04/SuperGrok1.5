// bindings/supergrok2.cpp — runtime dispatch to per-arch SG v2 launchers.
//
// SG v2 has the largest dispatcher surface in the codebase. The Python
// optimizer (grokking_optimizers/supergrok2.py) invokes six entry points
// that all dispatch to per-arch launchers in
// csrc/kernels/{cuda/<sm>,hip/<gfx>}/supergrok2_{fwd,bwd}_<arch>.{cu,hip.cpp}.
//
// Per-arch launcher names (signatures must match exactly across all 8
// arches — the wrap-baseline pass preserved bit-identical signatures):
//
//   sg::<arch>::launch_mamba3_peer_step                    (forward step)
//   sg::<arch>::launch_mamba3_peer_batched_step            (batched fwd step)
//   sg::<arch>::launch_mamba3_peer_bilevel_fwd_save        (bilevel fwd-save)
//   sg::<arch>::launch_mamba3_peer_bilevel_fwd_save_batched(batched fwd-save)
//   sg::<arch>::launch_mamba3_peer_backward                (bilevel bwd)
//   sg::<arch>::launch_mamba3_peer_backward_batched        (batched bwd)
//
// Public Python entry points (registered via _ops.<name> in module.cpp):
//   supergrok2_mamba_peer_step
//   supergrok2_mamba_peer_batched_step
//   supergrok2_bilevel_fwd_save
//   supergrok2_bilevel_fwd_save_batched
//   supergrok2_bilevel_backward
//   supergrok2_bilevel_backward_batched
//
// Reference: pre-refactor csrc/common/ops.cpp@682eab4^ (lines 908-1199 for
// forward/peer-step entries, lines 1499-1611 for bilevel registrations).
// The launcher signatures here are taken verbatim from the canonical sm_80
// translation units (supergrok2_fwd_sm80.cu / supergrok2_bwd_sm80.cu) —
// they are the linker contract.

#include "_dispatch_macro.h"
#include "_helpers.h"

#include <vector>

namespace sg {

// ---------------------------------------------------------------------
// Per-arch launcher forward declarations.
// Signatures verbatim from csrc/kernels/cuda/sm_80/supergrok2_{fwd,bwd}_sm80.cu.
// ---------------------------------------------------------------------

#define DECLARE_SG2(NS)                                                       \
    namespace NS {                                                            \
        void launch_mamba3_peer_step(                                         \
            torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness, \
            torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,\
            torch::Tensor gru_state,                                          \
            torch::Tensor mamba_fwd_state, torch::Tensor mamba_bwd_state,     \
            torch::Tensor input_proj_W, torch::Tensor input_proj_b,           \
            torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,    \
            torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,     \
            torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,    \
            torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,          \
            torch::Tensor mamba_fwd_out_proj,                                 \
            torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,    \
            torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,     \
            torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,    \
            torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,          \
            torch::Tensor mamba_bwd_out_proj,                                 \
            torch::Tensor gru_Wz, torch::Tensor gru_bz,                       \
            torch::Tensor gru_Wr, torch::Tensor gru_br,                       \
            torch::Tensor gru_Wh, torch::Tensor gru_bh,                       \
            torch::Tensor peer_query_Ws,                                      \
            torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,             \
            torch::Tensor expert_W1, torch::Tensor expert_b1,                 \
            torch::Tensor expert_W2, torch::Tensor expert_b2,                 \
            float rescale, float alpha_mu, float lamb_eff,                    \
            float beta1, float beta2, float lr, float wd_eff, float eps,      \
            float bc1, float bc2,                                             \
            int d_model, int d_state, int d_inner,                            \
            int gru_hidden, int num_heads, int pk_dim,                        \
            int expert_hidden, int num_experts,                               \
            torch::Tensor expert_counts);                                     \
        void launch_mamba3_peer_batched_step(                                 \
            std::vector<torch::Tensor> params,                                \
            std::vector<torch::Tensor> grads,                                 \
            std::vector<torch::Tensor> sharpness_list,                        \
            std::vector<torch::Tensor> exp_avgs,                              \
            std::vector<torch::Tensor> exp_avg_sqs,                           \
            std::vector<torch::Tensor> mus,                                   \
            std::vector<torch::Tensor> gru_states,                            \
            std::vector<torch::Tensor> mamba_fwd_states,                      \
            std::vector<torch::Tensor> mamba_bwd_states,                      \
            torch::Tensor input_proj_W, torch::Tensor input_proj_b,           \
            torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,    \
            torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,     \
            torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,    \
            torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,          \
            torch::Tensor mamba_fwd_out_proj,                                 \
            torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,    \
            torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,     \
            torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,    \
            torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,          \
            torch::Tensor mamba_bwd_out_proj,                                 \
            torch::Tensor gru_Wz, torch::Tensor gru_bz,                       \
            torch::Tensor gru_Wr, torch::Tensor gru_br,                       \
            torch::Tensor gru_Wh, torch::Tensor gru_bh,                       \
            torch::Tensor peer_query_Ws,                                      \
            torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,             \
            torch::Tensor expert_W1, torch::Tensor expert_b1,                 \
            torch::Tensor expert_W2, torch::Tensor expert_b2,                 \
            std::vector<float> alpha_mus, std::vector<float> lamb_effs,       \
            std::vector<float> beta1s,                                        \
            std::vector<float> bc1s, std::vector<float> bc2s,                 \
            float rescale, float beta2, float lr, float wd_eff, float eps,    \
            int d_model, int d_state, int d_inner,                            \
            int gru_hidden, int num_heads, int pk_dim,                        \
            int expert_hidden, int num_experts,                               \
            torch::Tensor expert_counts);                                     \
        void launch_mamba3_peer_bilevel_fwd_save(                             \
            torch::Tensor grad, torch::Tensor sharpness,                      \
            torch::Tensor input_proj_W, torch::Tensor input_proj_b,           \
            torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,    \
            torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,     \
            torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,    \
            torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,          \
            torch::Tensor mamba_fwd_out_proj,                                 \
            torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,    \
            torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,     \
            torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,    \
            torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,          \
            torch::Tensor mamba_bwd_out_proj,                                 \
            int d_model, int d_state, int d_inner,                            \
            torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out,           \
            torch::Tensor fwd_final_state, torch::Tensor bwd_final_state,     \
            torch::Tensor fwd_saved_states,                                   \
            torch::Tensor fwd_saved_x_branch,                                 \
            torch::Tensor fwd_saved_z, torch::Tensor fwd_saved_dt,            \
            torch::Tensor bwd_saved_states,                                   \
            torch::Tensor bwd_saved_x_branch,                                 \
            torch::Tensor bwd_saved_z, torch::Tensor bwd_saved_dt,            \
            torch::Tensor x_sorted, torch::Tensor sort_indices,               \
            torch::Tensor fwd_initial_state,                                  \
            torch::Tensor bwd_initial_state,                                  \
            int checkpoint_interval);                                         \
        void launch_mamba3_peer_bilevel_fwd_save_batched(                     \
            std::vector<torch::Tensor> grads,                                 \
            std::vector<torch::Tensor> sharpness_list,                        \
            torch::Tensor input_proj_W, torch::Tensor input_proj_b,           \
            torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,    \
            torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,     \
            torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,    \
            torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,          \
            torch::Tensor mamba_fwd_out_proj,                                 \
            torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,    \
            torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,     \
            torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,    \
            torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,          \
            torch::Tensor mamba_bwd_out_proj,                                 \
            int d_model, int d_state, int d_inner,                            \
            torch::Tensor fwd_scan_out_packed,                                \
            torch::Tensor bwd_scan_out_packed,                                \
            torch::Tensor fwd_saved_states_packed,                            \
            torch::Tensor fwd_saved_xb_packed,                                \
            torch::Tensor fwd_saved_z_packed,                                 \
            torch::Tensor fwd_saved_dt_packed,                                \
            torch::Tensor bwd_saved_states_packed,                            \
            torch::Tensor bwd_saved_xb_packed,                                \
            torch::Tensor bwd_saved_z_packed,                                 \
            torch::Tensor bwd_saved_dt_packed,                                \
            torch::Tensor x_sorted_packed, torch::Tensor offsets_t,           \
            torch::Tensor sort_indices_packed,                                \
            torch::Tensor fwd_initial_states,                                 \
            torch::Tensor bwd_initial_states,                                 \
            int checkpoint_interval);                                         \
        void launch_mamba3_peer_backward(                                     \
            torch::Tensor d_smart_grad,                                       \
            torch::Tensor grad, torch::Tensor sharpness, float rescale,       \
            torch::Tensor sort_indices, torch::Tensor x_sorted,               \
            torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out,           \
            torch::Tensor fwd_saved_states,                                   \
            torch::Tensor fwd_saved_x_branch,                                 \
            torch::Tensor fwd_saved_z, torch::Tensor fwd_saved_dt,            \
            torch::Tensor bwd_saved_states,                                   \
            torch::Tensor bwd_saved_x_branch,                                 \
            torch::Tensor bwd_saved_z, torch::Tensor bwd_saved_dt,            \
            torch::Tensor gru_input, torch::Tensor gru_h_old,                 \
            torch::Tensor gru_z_gate, torch::Tensor gru_r_gate,               \
            torch::Tensor gru_h_tilde,                                        \
            torch::Tensor peer_input, torch::Tensor expert_indices,           \
            torch::Tensor routing_weights, torch::Tensor saved_z_hidden,      \
            torch::Tensor saved_scores_a, torch::Tensor saved_scores_b,       \
            torch::Tensor saved_top_a_idx, torch::Tensor saved_top_b_idx,     \
            torch::Tensor saved_soft_a, torch::Tensor saved_soft_b,           \
            torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,    \
            torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,     \
            torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,    \
            torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,          \
            torch::Tensor mamba_fwd_out_proj,                                 \
            torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,    \
            torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,     \
            torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,    \
            torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,          \
            torch::Tensor mamba_bwd_out_proj,                                 \
            torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh, \
            torch::Tensor peer_query_Ws,                                      \
            torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,             \
            torch::Tensor expert_W1, torch::Tensor expert_W2,                 \
            torch::Tensor expert_b1_in, torch::Tensor expert_b2_in,           \
            torch::Tensor input_proj_W,                                       \
            torch::Tensor mamba_fwd_init_state,                               \
            torch::Tensor mamba_bwd_init_state,                               \
            torch::Tensor d_mamba_fwd_in_proj,                                \
            torch::Tensor d_mamba_fwd_dt_W,                                   \
            torch::Tensor d_mamba_fwd_dt_b,                                   \
            torch::Tensor d_mamba_fwd_B_proj,                                 \
            torch::Tensor d_mamba_fwd_C_proj,                                 \
            torch::Tensor d_mamba_fwd_A_log,                                  \
            torch::Tensor d_mamba_fwd_D,                                      \
            torch::Tensor d_mamba_fwd_rope,                                   \
            torch::Tensor d_mamba_fwd_out_proj,                               \
            torch::Tensor d_mamba_bwd_in_proj,                                \
            torch::Tensor d_mamba_bwd_dt_W,                                   \
            torch::Tensor d_mamba_bwd_dt_b,                                   \
            torch::Tensor d_mamba_bwd_B_proj,                                 \
            torch::Tensor d_mamba_bwd_C_proj,                                 \
            torch::Tensor d_mamba_bwd_A_log,                                  \
            torch::Tensor d_mamba_bwd_D,                                      \
            torch::Tensor d_mamba_bwd_rope,                                   \
            torch::Tensor d_mamba_bwd_out_proj,                               \
            torch::Tensor d_gru_Wz, torch::Tensor d_gru_bz,                   \
            torch::Tensor d_gru_Wr, torch::Tensor d_gru_br,                   \
            torch::Tensor d_gru_Wh, torch::Tensor d_gru_bh,                   \
            torch::Tensor d_peer_query_Ws,                                    \
            torch::Tensor d_prod_keys_A, torch::Tensor d_prod_keys_B,         \
            torch::Tensor d_expert_W1, torch::Tensor d_expert_b1,             \
            torch::Tensor d_expert_W2, torch::Tensor d_expert_b2,             \
            torch::Tensor d_input_proj_W, torch::Tensor d_input_proj_b,       \
            int d_model, int d_state, int d_inner,                            \
            int gru_hidden, int gru_input_dim,                                \
            int num_heads, int topk, int pk_dim,                              \
            int expert_hidden, int peer_input_dim, int num_experts,           \
            int checkpoint_interval);                                         \
        void launch_mamba3_peer_backward_batched(                             \
            torch::Tensor d_fwd_scan_out_packed,                              \
            torch::Tensor d_bwd_scan_out_packed,                              \
            torch::Tensor x_sorted_packed,                                    \
            torch::Tensor fwd_saved_states_packed,                            \
            torch::Tensor fwd_saved_xb_packed,                                \
            torch::Tensor fwd_saved_z_packed,                                 \
            torch::Tensor fwd_saved_dt_packed,                                \
            torch::Tensor bwd_saved_states_packed,                            \
            torch::Tensor bwd_saved_xb_packed,                                \
            torch::Tensor bwd_saved_z_packed,                                 \
            torch::Tensor bwd_saved_dt_packed,                                \
            torch::Tensor offsets_t,                                          \
            torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,    \
            torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,     \
            torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,    \
            torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,          \
            torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,    \
            torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,     \
            torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,    \
            torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,          \
            torch::Tensor d_mamba_fwd_in_proj,                                \
            torch::Tensor d_mamba_fwd_dt_W, torch::Tensor d_mamba_fwd_dt_b,   \
            torch::Tensor d_mamba_fwd_B_proj,                                 \
            torch::Tensor d_mamba_fwd_C_proj,                                 \
            torch::Tensor d_mamba_fwd_A_log,                                  \
            torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,      \
            torch::Tensor d_mamba_bwd_in_proj,                                \
            torch::Tensor d_mamba_bwd_dt_W, torch::Tensor d_mamba_bwd_dt_b,   \
            torch::Tensor d_mamba_bwd_B_proj,                                 \
            torch::Tensor d_mamba_bwd_C_proj,                                 \
            torch::Tensor d_mamba_bwd_A_log,                                  \
            torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,      \
            torch::Tensor d_x_sorted_packed,                                  \
            torch::Tensor fwd_initial_states,                                 \
            torch::Tensor bwd_initial_states,                                 \
            int d_model, int d_state, int d_inner, int num_params,            \
            int checkpoint_interval);                                         \
    }

DECLARE_SG2(sm80) DECLARE_SG2(sm89) DECLARE_SG2(sm90)
DECLARE_SG2(sm100) DECLARE_SG2(sm103) DECLARE_SG2(sm120)
DECLARE_SG2(gfx942) DECLARE_SG2(gfx950)
#undef DECLARE_SG2

// ---------------------------------------------------------------------
// Public entry points: thin SG_DISPATCH wrappers around the launchers.
// Each one matches the pre-refactor csrc/common/ops.cpp signature so
// existing Python callers (grokking_optimizers/supergrok2.py) keep
// working unchanged. Reference: ops.cpp@682eab4^ lines 908-1199 for the
// forward/peer-step entries, lines 1499-1611 for the bilevel entries.
// ---------------------------------------------------------------------

// Pre-refactor csrc/common/ops.cpp::supergrok2_mamba_peer_step.
void supergrok2_mamba_peer_step(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor gru_state,
    torch::Tensor mamba_fwd_state, torch::Tensor mamba_bwd_state,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws,
    torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts
) {
    if (grad.numel() == 0) return;
    SG_DISPATCH(launch_mamba3_peer_step,
        param, grad, sharpness, exp_avg, exp_avg_sq, mu,
        gru_state, mamba_fwd_state, mamba_bwd_state,
        input_proj_W, input_proj_b,
        mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
        mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
        mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
        mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
        mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
        mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
        gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
        peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_b1, expert_W2, expert_b2,
        rescale, alpha_mu, lamb_eff,
        beta1, beta2, lr, wd_eff, eps, bc1, bc2,
        d_model, d_state, d_inner,
        gru_hidden, num_heads, pk_dim,
        expert_hidden, num_experts, expert_counts);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_mamba_peer_batched_step.
void supergrok2_mamba_peer_batched_step(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws,
    torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s,
    std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts
) {
    if (params.empty()) return;
    SG_DISPATCH(launch_mamba3_peer_batched_step,
        params, grads, sharpness_list, exp_avgs, exp_avg_sqs, mus,
        gru_states, mamba_fwd_states, mamba_bwd_states,
        input_proj_W, input_proj_b,
        mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
        mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
        mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
        mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
        mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
        mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
        gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
        peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_b1, expert_W2, expert_b2,
        alpha_mus, lamb_effs, beta1s, bc1s, bc2s,
        rescale, beta2, lr, wd_eff, eps,
        d_model, d_state, d_inner,
        gru_hidden, num_heads, pk_dim,
        expert_hidden, num_experts, expert_counts);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_bilevel_fwd_save.
// Forward scan with state-saving for the meta-net's bilevel backward.
void supergrok2_bilevel_fwd_save(
    torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    int d_model, int d_state, int d_inner,
    torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out,
    torch::Tensor fwd_final_state, torch::Tensor bwd_final_state,
    torch::Tensor fwd_saved_states,
    torch::Tensor fwd_saved_x_branch,
    torch::Tensor fwd_saved_z, torch::Tensor fwd_saved_dt,
    torch::Tensor bwd_saved_states,
    torch::Tensor bwd_saved_x_branch,
    torch::Tensor bwd_saved_z, torch::Tensor bwd_saved_dt,
    torch::Tensor x_sorted, torch::Tensor sort_indices,
    torch::Tensor fwd_initial_state,
    torch::Tensor bwd_initial_state,
    int checkpoint_interval
) {
    if (grad.numel() == 0) return;
    SG_DISPATCH(launch_mamba3_peer_bilevel_fwd_save,
        grad, sharpness,
        input_proj_W, input_proj_b,
        mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
        mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
        mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
        mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
        mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
        mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
        d_model, d_state, d_inner,
        fwd_scan_out, bwd_scan_out,
        fwd_final_state, bwd_final_state,
        fwd_saved_states, fwd_saved_x_branch, fwd_saved_z, fwd_saved_dt,
        bwd_saved_states, bwd_saved_x_branch, bwd_saved_z, bwd_saved_dt,
        x_sorted, sort_indices,
        fwd_initial_state, bwd_initial_state,
        checkpoint_interval);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_bilevel_fwd_save_batched.
void supergrok2_bilevel_fwd_save_batched(
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    int d_model, int d_state, int d_inner,
    torch::Tensor fwd_scan_out_packed,
    torch::Tensor bwd_scan_out_packed,
    torch::Tensor fwd_saved_states_packed,
    torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed,
    torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed,
    torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed,
    torch::Tensor bwd_saved_dt_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor fwd_initial_states,
    torch::Tensor bwd_initial_states,
    int checkpoint_interval
) {
    if (grads.empty()) return;
    SG_DISPATCH(launch_mamba3_peer_bilevel_fwd_save_batched,
        grads, sharpness_list,
        input_proj_W, input_proj_b,
        mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
        mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
        mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
        mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
        mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
        mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
        d_model, d_state, d_inner,
        fwd_scan_out_packed, bwd_scan_out_packed,
        fwd_saved_states_packed, fwd_saved_xb_packed,
        fwd_saved_z_packed, fwd_saved_dt_packed,
        bwd_saved_states_packed, bwd_saved_xb_packed,
        bwd_saved_z_packed, bwd_saved_dt_packed,
        x_sorted_packed, offsets_t, sort_indices_packed,
        fwd_initial_states, bwd_initial_states,
        checkpoint_interval);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_bilevel_backward.
// Full backward through the meta-net, single-tensor entry.
void supergrok2_bilevel_backward(
    torch::Tensor d_smart_grad,
    torch::Tensor grad, torch::Tensor sharpness, float rescale,
    torch::Tensor sort_indices, torch::Tensor x_sorted,
    torch::Tensor fwd_scan_out, torch::Tensor bwd_scan_out,
    torch::Tensor fwd_saved_states,
    torch::Tensor fwd_saved_x_branch,
    torch::Tensor fwd_saved_z, torch::Tensor fwd_saved_dt,
    torch::Tensor bwd_saved_states,
    torch::Tensor bwd_saved_x_branch,
    torch::Tensor bwd_saved_z, torch::Tensor bwd_saved_dt,
    torch::Tensor gru_input, torch::Tensor gru_h_old,
    torch::Tensor gru_z_gate, torch::Tensor gru_r_gate,
    torch::Tensor gru_h_tilde,
    torch::Tensor peer_input, torch::Tensor expert_indices,
    torch::Tensor routing_weights, torch::Tensor saved_z_hidden,
    torch::Tensor saved_scores_a, torch::Tensor saved_scores_b,
    torch::Tensor saved_top_a_idx, torch::Tensor saved_top_b_idx,
    torch::Tensor saved_soft_a, torch::Tensor saved_soft_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh,
    torch::Tensor peer_query_Ws,
    torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_W2,
    torch::Tensor expert_b1_in, torch::Tensor expert_b2_in,
    torch::Tensor input_proj_W,
    torch::Tensor mamba_fwd_init_state,
    torch::Tensor mamba_bwd_init_state,
    torch::Tensor d_mamba_fwd_in_proj,
    torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b,
    torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj,
    torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D,
    torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_fwd_out_proj,
    torch::Tensor d_mamba_bwd_in_proj,
    torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b,
    torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj,
    torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D,
    torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_mamba_bwd_out_proj,
    torch::Tensor d_gru_Wz, torch::Tensor d_gru_bz,
    torch::Tensor d_gru_Wr, torch::Tensor d_gru_br,
    torch::Tensor d_gru_Wh, torch::Tensor d_gru_bh,
    torch::Tensor d_peer_query_Ws,
    torch::Tensor d_prod_keys_A, torch::Tensor d_prod_keys_B,
    torch::Tensor d_expert_W1, torch::Tensor d_expert_b1,
    torch::Tensor d_expert_W2, torch::Tensor d_expert_b2,
    torch::Tensor d_input_proj_W, torch::Tensor d_input_proj_b,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int gru_input_dim,
    int num_heads, int topk, int pk_dim,
    int expert_hidden, int peer_input_dim, int num_experts,
    int checkpoint_interval
) {
    if (d_smart_grad.numel() == 0) return;
    SG_DISPATCH(launch_mamba3_peer_backward,
        d_smart_grad, grad, sharpness, rescale,
        sort_indices, x_sorted,
        fwd_scan_out, bwd_scan_out,
        fwd_saved_states, fwd_saved_x_branch, fwd_saved_z, fwd_saved_dt,
        bwd_saved_states, bwd_saved_x_branch, bwd_saved_z, bwd_saved_dt,
        gru_input, gru_h_old, gru_z_gate, gru_r_gate, gru_h_tilde,
        peer_input, expert_indices, routing_weights, saved_z_hidden,
        saved_scores_a, saved_scores_b,
        saved_top_a_idx, saved_top_b_idx, saved_soft_a, saved_soft_b,
        mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
        mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
        mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
        mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
        mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
        mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
        gru_Wz, gru_Wr, gru_Wh,
        peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_W2, expert_b1_in, expert_b2_in, input_proj_W,
        mamba_fwd_init_state, mamba_bwd_init_state,
        d_mamba_fwd_in_proj, d_mamba_fwd_dt_W, d_mamba_fwd_dt_b,
        d_mamba_fwd_B_proj, d_mamba_fwd_C_proj, d_mamba_fwd_A_log,
        d_mamba_fwd_D, d_mamba_fwd_rope, d_mamba_fwd_out_proj,
        d_mamba_bwd_in_proj, d_mamba_bwd_dt_W, d_mamba_bwd_dt_b,
        d_mamba_bwd_B_proj, d_mamba_bwd_C_proj, d_mamba_bwd_A_log,
        d_mamba_bwd_D, d_mamba_bwd_rope, d_mamba_bwd_out_proj,
        d_gru_Wz, d_gru_bz, d_gru_Wr, d_gru_br, d_gru_Wh, d_gru_bh,
        d_peer_query_Ws, d_prod_keys_A, d_prod_keys_B,
        d_expert_W1, d_expert_b1, d_expert_W2, d_expert_b2,
        d_input_proj_W, d_input_proj_b,
        d_model, d_state, d_inner,
        gru_hidden, gru_input_dim,
        num_heads, topk, pk_dim,
        expert_hidden, peer_input_dim, num_experts,
        checkpoint_interval);
}

// Pre-refactor csrc/common/ops.cpp::supergrok2_bilevel_backward_batched.
void supergrok2_bilevel_backward_batched(
    torch::Tensor d_fwd_scan_out_packed,
    torch::Tensor d_bwd_scan_out_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor fwd_saved_states_packed,
    torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed,
    torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed,
    torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed,
    torch::Tensor bwd_saved_dt_packed,
    torch::Tensor offsets_t,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor d_mamba_fwd_in_proj,
    torch::Tensor d_mamba_fwd_dt_W, torch::Tensor d_mamba_fwd_dt_b,
    torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj,
    torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_bwd_in_proj,
    torch::Tensor d_mamba_bwd_dt_W, torch::Tensor d_mamba_bwd_dt_b,
    torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj,
    torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_x_sorted_packed,
    torch::Tensor fwd_initial_states,
    torch::Tensor bwd_initial_states,
    int d_model, int d_state, int d_inner, int num_params,
    int checkpoint_interval
) {
    if (num_params == 0) return;
    SG_DISPATCH(launch_mamba3_peer_backward_batched,
        d_fwd_scan_out_packed, d_bwd_scan_out_packed,
        x_sorted_packed,
        fwd_saved_states_packed, fwd_saved_xb_packed,
        fwd_saved_z_packed, fwd_saved_dt_packed,
        bwd_saved_states_packed, bwd_saved_xb_packed,
        bwd_saved_z_packed, bwd_saved_dt_packed,
        offsets_t,
        mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
        mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
        mamba_fwd_D, mamba_fwd_rope,
        mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
        mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
        mamba_bwd_D, mamba_bwd_rope,
        d_mamba_fwd_in_proj, d_mamba_fwd_dt_W, d_mamba_fwd_dt_b,
        d_mamba_fwd_B_proj, d_mamba_fwd_C_proj, d_mamba_fwd_A_log,
        d_mamba_fwd_D, d_mamba_fwd_rope,
        d_mamba_bwd_in_proj, d_mamba_bwd_dt_W, d_mamba_bwd_dt_b,
        d_mamba_bwd_B_proj, d_mamba_bwd_C_proj, d_mamba_bwd_A_log,
        d_mamba_bwd_D, d_mamba_bwd_rope,
        d_x_sorted_packed,
        fwd_initial_states, bwd_initial_states,
        d_model, d_state, d_inner, num_params,
        checkpoint_interval);
}

} // namespace sg
