// =====================================================================
//  csrc/kernels/cuda/sm_90/supergrok2_bwd.cu
//
//  sm_90 SuperGrok v2 BACKWARD instantiation TU.
//  Pulls the templated kernels from `supergrok2_bwd.cuh`, instantiates
//  the dtype matrix (ParamT × StateT × GradT), and bridges the
//  binding-side `sg::sm90::launch_mamba3_peer_backward{,_batched}`
//  symbols to the canonical implementations in
//  `sg::sm90::supergrok2::launch_mamba3_peer_backward{,_batched}_impl`.
//
//  The bridge is necessary because the binding macro
//  DECLARE_SG2(sm90) declares the launchers in namespace `sg::sm90`,
//  whereas the new kernel header places them in
//  `sg::sm90::supergrok2`. We define a thin wrapper here that the
//  binding's `SG_DISPATCH(launch_mamba3_peer_backward, ...)`
//  resolves to.
//
//  Instantiation count (input_proj_backward_kernel only — the rest of
//  the kernels operate on FP32 metanet weights and FP32 saved states
//  per the SG2 design and need no template instantiation):
//      GradT ∈ { float, __nv_bfloat16, __half }            : 3 instances
//      Plus FP8 e4m3 and e5m2 guarded on toolchain support : +2 instances
//      Total : up to 5 explicit instantiations.
//
//  Forward agent owns supergrok2_fwd.{cuh,cu} and
//  supergrok2_warp_specialized.{cuh,cu}; we do not touch those here.
// =====================================================================

#include "supergrok2_bwd.cuh"

#if GROK_CUDA

namespace sg { namespace sm90 { namespace supergrok2 {

// ---- Explicit kernel-template instantiations (input-proj-bwd only) ----
template __global__ void input_proj_backward_kernel<float>(
    const float*, const float*, const float*, float*, float*, int, int);
template __global__ void input_proj_backward_kernel<__nv_bfloat16>(
    const float*, const __nv_bfloat16*, const __nv_bfloat16*, float*, float*, int, int);
template __global__ void input_proj_backward_kernel<__half>(
    const float*, const __half*, const __half*, float*, float*, int, int);
#if SG2_BWD_HAS_FP8
template __global__ void input_proj_backward_kernel<__nv_fp8_e4m3>(
    const float*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, float*, float*, int, int);
template __global__ void input_proj_backward_kernel<__nv_fp8_e5m2>(
    const float*, const __nv_fp8_e5m2*, const __nv_fp8_e5m2*, float*, float*, int, int);
#endif

// ---- Dtype-coherence sanity asserts (fire at instantiation) ----
template struct sg2_dtype_check<float,         float,         float        >;
template struct sg2_dtype_check<float,         float,         __nv_bfloat16>;
template struct sg2_dtype_check<float,         float,         __half       >;
template struct sg2_dtype_check<__nv_bfloat16, float,         float        >;
template struct sg2_dtype_check<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
template struct sg2_dtype_check<__half,        float,         __half       >;

}}} // namespace sg::sm90::supergrok2

// ---- Binding-side bridge: re-export under namespace sg::sm90 ----
//
// The binding TU `csrc/bindings/supergrok2.cpp` declares:
//
//   namespace sg { namespace sm90 {
//     void launch_mamba3_peer_backward(...);
//     void launch_mamba3_peer_backward_batched(...);
//   } }
//
// We define those symbols here as thin forwarders to the canonical
// `sg::sm90::supergrok2::launch_*_impl` definitions in the header.

namespace sg { namespace sm90 {

void launch_mamba3_peer_backward(
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
    sg::sm90::supergrok2::launch_mamba3_peer_backward_impl(
        d_smart_grad, grad, sharpness, rescale, sort_indices, x_sorted,
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

void launch_mamba3_peer_backward_batched(
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
    sg::sm90::supergrok2::launch_mamba3_peer_backward_batched_impl(
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

}} // namespace sg::sm90

#endif // GROK_CUDA
