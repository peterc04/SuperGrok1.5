// =====================================================================
//  csrc/kernels/hip/gfx942/supergrok2_fwd.hip.cpp
//
//  SuperGrok v2 forward launchers — gfx942. The full Mamba+GRU+PEER
//  pipeline is too large to port here as a "compile + link" milestone.
//  These entry points throw at runtime; the rest of the optimizer
//  catalogue is functional on gfx942.
//
//  Signatures here are copied verbatim from the DECLARE_SG2 macro in
//  csrc/bindings/supergrok2.cpp to ensure ABI / linker agreement.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/supergrok2_fwd.hip.h"

namespace {
[[noreturn]] inline void sg2_unimplemented(const char* what) {
    throw std::runtime_error(
        std::string("supergrok2 (gfx942): ") + what +
        " is not implemented in this port. The Mamba+GRU+PEER pipeline "
        "requires an MFMA + LDS-resident gfx942 implementation; see "
        "csrc/kernels/hip/gfx942/supergrok2_fwd.hip.cpp.");
}
} // anon

namespace sg { namespace gfx942 {

// -----------------------------------------------------------------
// 1) launch_mamba3_peer_step
// -----------------------------------------------------------------
void launch_mamba3_peer_step(
    torch::Tensor /*param*/, torch::Tensor /*grad*/, torch::Tensor /*sharpness*/,
    torch::Tensor /*exp_avg*/, torch::Tensor /*exp_avg_sq*/, torch::Tensor /*mu*/,
    torch::Tensor /*gru_state*/,
    torch::Tensor /*mamba_fwd_state*/, torch::Tensor /*mamba_bwd_state*/,
    torch::Tensor /*input_proj_W*/, torch::Tensor /*input_proj_b*/,
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
    torch::Tensor /*gru_Wz*/, torch::Tensor /*gru_bz*/,
    torch::Tensor /*gru_Wr*/, torch::Tensor /*gru_br*/,
    torch::Tensor /*gru_Wh*/, torch::Tensor /*gru_bh*/,
    torch::Tensor /*peer_query_Ws*/,
    torch::Tensor /*prod_keys_A*/, torch::Tensor /*prod_keys_B*/,
    torch::Tensor /*expert_W1*/, torch::Tensor /*expert_b1*/,
    torch::Tensor /*expert_W2*/, torch::Tensor /*expert_b2*/,
    float /*rescale*/, float /*alpha_mu*/, float /*lamb_eff*/,
    float /*beta1*/, float /*beta2*/, float /*lr*/, float /*wd_eff*/, float /*eps*/,
    float /*bc1*/, float /*bc2*/,
    int /*d_model*/, int /*d_state*/, int /*d_inner*/,
    int /*gru_hidden*/, int /*num_heads*/, int /*pk_dim*/,
    int /*expert_hidden*/, int /*num_experts*/,
    torch::Tensor /*expert_counts*/
) {
    sg2_unimplemented("launch_mamba3_peer_step");
}

// -----------------------------------------------------------------
// 2) launch_mamba3_peer_batched_step
// -----------------------------------------------------------------
void launch_mamba3_peer_batched_step(
    std::vector<torch::Tensor> /*params*/,
    std::vector<torch::Tensor> /*grads*/,
    std::vector<torch::Tensor> /*sharpness_list*/,
    std::vector<torch::Tensor> /*exp_avgs*/,
    std::vector<torch::Tensor> /*exp_avg_sqs*/,
    std::vector<torch::Tensor> /*mus*/,
    std::vector<torch::Tensor> /*gru_states*/,
    std::vector<torch::Tensor> /*mamba_fwd_states*/,
    std::vector<torch::Tensor> /*mamba_bwd_states*/,
    torch::Tensor /*input_proj_W*/, torch::Tensor /*input_proj_b*/,
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
    torch::Tensor /*gru_Wz*/, torch::Tensor /*gru_bz*/,
    torch::Tensor /*gru_Wr*/, torch::Tensor /*gru_br*/,
    torch::Tensor /*gru_Wh*/, torch::Tensor /*gru_bh*/,
    torch::Tensor /*peer_query_Ws*/,
    torch::Tensor /*prod_keys_A*/, torch::Tensor /*prod_keys_B*/,
    torch::Tensor /*expert_W1*/, torch::Tensor /*expert_b1*/,
    torch::Tensor /*expert_W2*/, torch::Tensor /*expert_b2*/,
    std::vector<float> /*alpha_mus*/, std::vector<float> /*lamb_effs*/,
    std::vector<float> /*beta1s*/,
    std::vector<float> /*bc1s*/, std::vector<float> /*bc2s*/,
    float /*rescale*/, float /*beta2*/, float /*lr*/, float /*wd_eff*/, float /*eps*/,
    int /*d_model*/, int /*d_state*/, int /*d_inner*/,
    int /*gru_hidden*/, int /*num_heads*/, int /*pk_dim*/,
    int /*expert_hidden*/, int /*num_experts*/,
    torch::Tensor /*expert_counts*/
) {
    sg2_unimplemented("launch_mamba3_peer_batched_step");
}

// -----------------------------------------------------------------
// 3) launch_mamba3_peer_bilevel_fwd_save
// -----------------------------------------------------------------
void launch_mamba3_peer_bilevel_fwd_save(
    torch::Tensor /*grad*/, torch::Tensor /*sharpness*/,
    torch::Tensor /*input_proj_W*/, torch::Tensor /*input_proj_b*/,
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
    int /*d_model*/, int /*d_state*/, int /*d_inner*/,
    torch::Tensor /*fwd_scan_out*/, torch::Tensor /*bwd_scan_out*/,
    torch::Tensor /*fwd_final_state*/, torch::Tensor /*bwd_final_state*/,
    torch::Tensor /*fwd_saved_states*/,
    torch::Tensor /*fwd_saved_x_branch*/,
    torch::Tensor /*fwd_saved_z*/, torch::Tensor /*fwd_saved_dt*/,
    torch::Tensor /*bwd_saved_states*/,
    torch::Tensor /*bwd_saved_x_branch*/,
    torch::Tensor /*bwd_saved_z*/, torch::Tensor /*bwd_saved_dt*/,
    torch::Tensor /*x_sorted*/, torch::Tensor /*sort_indices*/,
    torch::Tensor /*fwd_initial_state*/,
    torch::Tensor /*bwd_initial_state*/,
    int /*checkpoint_interval*/
) {
    sg2_unimplemented("launch_mamba3_peer_bilevel_fwd_save");
}

// -----------------------------------------------------------------
// 4) launch_mamba3_peer_bilevel_fwd_save_batched
// -----------------------------------------------------------------
void launch_mamba3_peer_bilevel_fwd_save_batched(
    std::vector<torch::Tensor> /*grads*/,
    std::vector<torch::Tensor> /*sharpness_list*/,
    torch::Tensor /*input_proj_W*/, torch::Tensor /*input_proj_b*/,
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
    int /*d_model*/, int /*d_state*/, int /*d_inner*/,
    torch::Tensor /*fwd_scan_out_packed*/,
    torch::Tensor /*bwd_scan_out_packed*/,
    torch::Tensor /*fwd_saved_states_packed*/,
    torch::Tensor /*fwd_saved_xb_packed*/,
    torch::Tensor /*fwd_saved_z_packed*/,
    torch::Tensor /*fwd_saved_dt_packed*/,
    torch::Tensor /*bwd_saved_states_packed*/,
    torch::Tensor /*bwd_saved_xb_packed*/,
    torch::Tensor /*bwd_saved_z_packed*/,
    torch::Tensor /*bwd_saved_dt_packed*/,
    torch::Tensor /*x_sorted_packed*/, torch::Tensor /*offsets_t*/,
    torch::Tensor /*sort_indices_packed*/,
    torch::Tensor /*fwd_initial_states*/,
    torch::Tensor /*bwd_initial_states*/,
    int /*checkpoint_interval*/
) {
    sg2_unimplemented("launch_mamba3_peer_bilevel_fwd_save_batched");
}

}} // namespace sg::gfx942
