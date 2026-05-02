// =====================================================================
//  csrc/kernels/hip/gfx942/supergrok2_bwd.hip.cpp
//
//  SuperGrok v2 backward launchers — gfx942 stub TU. Signatures verbatim
//  from csrc/bindings/supergrok2.cpp::DECLARE_SG2.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/supergrok2_bwd.hip.h"

namespace {
[[noreturn]] inline void sg2_unimplemented(const char* what) {
    throw std::runtime_error(
        std::string("supergrok2 backward (gfx942): ") + what +
        " is not implemented in this port. See "
        "csrc/kernels/hip/gfx942/supergrok2_bwd.hip.cpp.");
}
} // anon

namespace sg { namespace gfx942 {

void launch_mamba3_peer_backward(
    torch::Tensor /*d_smart_grad*/,
    torch::Tensor /*grad*/, torch::Tensor /*sharpness*/, float /*rescale*/,
    torch::Tensor /*sort_indices*/, torch::Tensor /*x_sorted*/,
    torch::Tensor /*fwd_scan_out*/, torch::Tensor /*bwd_scan_out*/,
    torch::Tensor /*fwd_saved_states*/,
    torch::Tensor /*fwd_saved_x_branch*/,
    torch::Tensor /*fwd_saved_z*/, torch::Tensor /*fwd_saved_dt*/,
    torch::Tensor /*bwd_saved_states*/,
    torch::Tensor /*bwd_saved_x_branch*/,
    torch::Tensor /*bwd_saved_z*/, torch::Tensor /*bwd_saved_dt*/,
    torch::Tensor /*gru_input*/, torch::Tensor /*gru_h_old*/,
    torch::Tensor /*gru_z_gate*/, torch::Tensor /*gru_r_gate*/,
    torch::Tensor /*gru_h_tilde*/,
    torch::Tensor /*peer_input*/, torch::Tensor /*expert_indices*/,
    torch::Tensor /*routing_weights*/, torch::Tensor /*saved_z_hidden*/,
    torch::Tensor /*saved_scores_a*/, torch::Tensor /*saved_scores_b*/,
    torch::Tensor /*saved_top_a_idx*/, torch::Tensor /*saved_top_b_idx*/,
    torch::Tensor /*saved_soft_a*/, torch::Tensor /*saved_soft_b*/,
    torch::Tensor /*mamba_fwd_in_proj*/, torch::Tensor /*mamba_fwd_dt_W*/,
    torch::Tensor /*mamba_fwd_dt_b*/, torch::Tensor /*mamba_fwd_B_proj*/,
    torch::Tensor /*mamba_fwd_C_proj*/, torch::Tensor /*mamba_fwd_A_log*/,
    torch::Tensor /*mamba_fwd_D*/, torch::Tensor /*mamba_fwd_rope*/,
    torch::Tensor /*mamba_fwd_out_proj*/,
    torch::Tensor /*mamba_bwd_in_proj*/, torch::Tensor /*mamba_bwd_dt_W*/,
    torch::Tensor /*mamba_bwd_dt_b*/, torch::Tensor /*mamba_bwd_B_proj*/,
    torch::Tensor /*mamba_bwd_C_proj*/, torch::Tensor /*mamba_bwd_A_log*/,
    torch::Tensor /*mamba_bwd_D*/, torch::Tensor /*mamba_bwd_rope*/,
    torch::Tensor /*mamba_bwd_out_proj*/,
    torch::Tensor /*gru_Wz*/, torch::Tensor /*gru_Wr*/, torch::Tensor /*gru_Wh*/,
    torch::Tensor /*peer_query_Ws*/,
    torch::Tensor /*prod_keys_A*/, torch::Tensor /*prod_keys_B*/,
    torch::Tensor /*expert_W1*/, torch::Tensor /*expert_W2*/,
    torch::Tensor /*expert_b1_in*/, torch::Tensor /*expert_b2_in*/,
    torch::Tensor /*input_proj_W*/,
    torch::Tensor /*mamba_fwd_init_state*/,
    torch::Tensor /*mamba_bwd_init_state*/,
    torch::Tensor /*d_mamba_fwd_in_proj*/,
    torch::Tensor /*d_mamba_fwd_dt_W*/,
    torch::Tensor /*d_mamba_fwd_dt_b*/,
    torch::Tensor /*d_mamba_fwd_B_proj*/,
    torch::Tensor /*d_mamba_fwd_C_proj*/,
    torch::Tensor /*d_mamba_fwd_A_log*/,
    torch::Tensor /*d_mamba_fwd_D*/,
    torch::Tensor /*d_mamba_fwd_rope*/,
    torch::Tensor /*d_mamba_fwd_out_proj*/,
    torch::Tensor /*d_mamba_bwd_in_proj*/,
    torch::Tensor /*d_mamba_bwd_dt_W*/,
    torch::Tensor /*d_mamba_bwd_dt_b*/,
    torch::Tensor /*d_mamba_bwd_B_proj*/,
    torch::Tensor /*d_mamba_bwd_C_proj*/,
    torch::Tensor /*d_mamba_bwd_A_log*/,
    torch::Tensor /*d_mamba_bwd_D*/,
    torch::Tensor /*d_mamba_bwd_rope*/,
    torch::Tensor /*d_mamba_bwd_out_proj*/,
    torch::Tensor /*d_gru_Wz*/, torch::Tensor /*d_gru_bz*/,
    torch::Tensor /*d_gru_Wr*/, torch::Tensor /*d_gru_br*/,
    torch::Tensor /*d_gru_Wh*/, torch::Tensor /*d_gru_bh*/,
    torch::Tensor /*d_peer_query_Ws*/,
    torch::Tensor /*d_prod_keys_A*/, torch::Tensor /*d_prod_keys_B*/,
    torch::Tensor /*d_expert_W1*/, torch::Tensor /*d_expert_b1*/,
    torch::Tensor /*d_expert_W2*/, torch::Tensor /*d_expert_b2*/,
    torch::Tensor /*d_input_proj_W*/, torch::Tensor /*d_input_proj_b*/,
    int /*d_model*/, int /*d_state*/, int /*d_inner*/,
    int /*gru_hidden*/, int /*gru_input_dim*/,
    int /*num_heads*/, int /*topk*/, int /*pk_dim*/,
    int /*expert_hidden*/, int /*peer_input_dim*/, int /*num_experts*/,
    int /*checkpoint_interval*/
) {
    sg2_unimplemented("launch_mamba3_peer_backward");
}

void launch_mamba3_peer_backward_batched(
    torch::Tensor /*d_fwd_scan_out_packed*/,
    torch::Tensor /*d_bwd_scan_out_packed*/,
    torch::Tensor /*x_sorted_packed*/,
    torch::Tensor /*fwd_saved_states_packed*/,
    torch::Tensor /*fwd_saved_xb_packed*/,
    torch::Tensor /*fwd_saved_z_packed*/,
    torch::Tensor /*fwd_saved_dt_packed*/,
    torch::Tensor /*bwd_saved_states_packed*/,
    torch::Tensor /*bwd_saved_xb_packed*/,
    torch::Tensor /*bwd_saved_z_packed*/,
    torch::Tensor /*bwd_saved_dt_packed*/,
    torch::Tensor /*offsets_t*/,
    torch::Tensor /*mamba_fwd_in_proj*/, torch::Tensor /*mamba_fwd_dt_W*/,
    torch::Tensor /*mamba_fwd_dt_b*/, torch::Tensor /*mamba_fwd_B_proj*/,
    torch::Tensor /*mamba_fwd_C_proj*/, torch::Tensor /*mamba_fwd_A_log*/,
    torch::Tensor /*mamba_fwd_D*/, torch::Tensor /*mamba_fwd_rope*/,
    torch::Tensor /*mamba_bwd_in_proj*/, torch::Tensor /*mamba_bwd_dt_W*/,
    torch::Tensor /*mamba_bwd_dt_b*/, torch::Tensor /*mamba_bwd_B_proj*/,
    torch::Tensor /*mamba_bwd_C_proj*/, torch::Tensor /*mamba_bwd_A_log*/,
    torch::Tensor /*mamba_bwd_D*/, torch::Tensor /*mamba_bwd_rope*/,
    torch::Tensor /*d_mamba_fwd_in_proj*/,
    torch::Tensor /*d_mamba_fwd_dt_W*/, torch::Tensor /*d_mamba_fwd_dt_b*/,
    torch::Tensor /*d_mamba_fwd_B_proj*/,
    torch::Tensor /*d_mamba_fwd_C_proj*/,
    torch::Tensor /*d_mamba_fwd_A_log*/,
    torch::Tensor /*d_mamba_fwd_D*/, torch::Tensor /*d_mamba_fwd_rope*/,
    torch::Tensor /*d_mamba_bwd_in_proj*/,
    torch::Tensor /*d_mamba_bwd_dt_W*/, torch::Tensor /*d_mamba_bwd_dt_b*/,
    torch::Tensor /*d_mamba_bwd_B_proj*/,
    torch::Tensor /*d_mamba_bwd_C_proj*/,
    torch::Tensor /*d_mamba_bwd_A_log*/,
    torch::Tensor /*d_mamba_bwd_D*/, torch::Tensor /*d_mamba_bwd_rope*/,
    torch::Tensor /*d_x_sorted_packed*/,
    torch::Tensor /*fwd_initial_states*/,
    torch::Tensor /*bwd_initial_states*/,
    int /*d_model*/, int /*d_state*/, int /*d_inner*/, int /*num_params*/,
    int /*checkpoint_interval*/
) {
    sg2_unimplemented("launch_mamba3_peer_backward_batched");
}

}} // namespace sg::gfx942
