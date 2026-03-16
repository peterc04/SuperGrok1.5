/*
 * SuperGrok v2 — CDNA3-Optimized Launchers (gfx942, MI300X)
 *
 * MI300X-specific optimizations beyond CDNA2:
 *   - BF16 MFMA projection precompute: cdna3_precompute_bf16() casts
 *     inputs and weights to BF16 at the torch::mm boundary, causing
 *     rocBLAS to dispatch to MFMA_F32_32x32x8_BF16 instructions for
 *     ~2x projection throughput. GEMM accumulation is FP32. Output
 *     tensors are FP32 for scan numerical stability.
 *   - Refactored batched step pipeline: setup_and_sort → BF16 precompute
 *     → shared scan+fused_elem. This correctly applies BF16 at the GEMM
 *     boundary rather than round-tripping weights through BF16→FP32.
 *   - 256MB L2 cache: meta-net weights (~50KB) always L2-resident
 *   - 304 CUs (vs 220 MI250): more concurrent scan blocks
 *
 * Single-param step: delegates to CDNA2 (BF16 overhead exceeds MFMA
 * benefit for small single-param GEMMs).
 *
 * Bilevel fwd_save and backward: delegate to CDNA2 (wavefront-64
 * scan via platform.h WARP_SIZE=64).
 */

#include <torch/extension.h>
#include "platform.h"
#include "types.h"
#include "ops.h"

// ═══════════════════════════════════════════════════════════════════════
//  Forward declarations of CDNA2 launchers
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_step_cdna2(
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
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

void launch_mamba3_peer_batched_step_cdna2(
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
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

void launch_mamba3_peer_bilevel_fwd_save_batched_cdna2(
    std::vector<torch::Tensor> grads, std::vector<torch::Tensor> sharpness_list,
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
    torch::Tensor fwd_scan_out_packed, torch::Tensor bwd_scan_out_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int checkpoint_interval);

void launch_mamba3_peer_backward_batched_cdna2(
    torch::Tensor d_fwd_scan_out_packed, torch::Tensor d_bwd_scan_out_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor offsets_t,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor d_mamba_fwd_in_proj, torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b, torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj, torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_bwd_in_proj, torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b, torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj, torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_x_sorted_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int d_model, int d_state, int d_inner, int num_params,
    int checkpoint_interval);


// ═══════════════════════════════════════════════════════════════════════
//  BF16 MFMA Precompute
//
//  Runs projection GEMMs with BF16 inputs via torch::mm, which on
//  MI300X dispatches to rocBLAS MFMA_F32_32x32x8_BF16 for ~2x
//  throughput over FP32 MFMA. Outputs are FP32 for scan stability.
//
//  The BF16 conversion happens at the GEMM boundary: inputs and weights
//  are cast to BF16, the GEMM accumulates in FP32, and outputs remain
//  FP32. This avoids the old BF16→FP32 round-trip bug where weights
//  were converted to BF16 then back to FP32 before the GEMM.
// ═══════════════════════════════════════════════════════════════════════

static void cdna3_precompute_bf16(
    torch::Tensor x_sorted,          // [N, d_model] FP32
    torch::Tensor in_proj_W,         // [2*d_inner, d_model] FP32
    torch::Tensor dt_proj_W,         // [d_inner, d_inner]
    torch::Tensor dt_proj_b,         // [d_inner]
    torch::Tensor B_proj_W,          // [d_state, d_inner]
    torch::Tensor C_proj_W,          // [d_state, d_inner]
    torch::Tensor& pre_x,            // [N, d_inner] output FP32
    torch::Tensor& pre_z,            // [N, d_inner] output FP32
    torch::Tensor& pre_dt,           // [N, d_inner] output FP32
    torch::Tensor& pre_B,            // [N, d_state] output FP32
    torch::Tensor& pre_C,            // [N, d_state] output FP32
    int N, int d_model, int d_inner, int d_state
) {
    // Convert inputs to BF16 at the GEMM boundary
    // rocBLAS sees BF16 inputs → dispatches to MFMA_F32_32x32x8_BF16
    // torch::mm with BF16 inputs accumulates in FP32 on CDNA3
    auto x_bf16 = x_sorted.to(torch::kBFloat16);
    auto in_proj_x_bf16 = in_proj_W.narrow(0, 0, d_inner).to(torch::kBFloat16);
    auto in_proj_z_bf16 = in_proj_W.narrow(0, d_inner, d_inner).to(torch::kBFloat16);

    // x_branch = x_sorted @ in_proj_x.T → [N, d_inner]
    // BF16 GEMM → FP32 output via torch::mm accumulation
    pre_x = torch::mm(x_bf16, in_proj_x_bf16.t()).to(torch::kFloat32);

    // z_branch = x_sorted @ in_proj_z.T → [N, d_inner]
    pre_z = torch::mm(x_bf16, in_proj_z_bf16.t()).to(torch::kFloat32);

    // dt_proj: x_branch @ dt_proj_W.T + bias → softplus
    auto pre_x_bf16 = pre_x.to(torch::kBFloat16);
    auto dt_proj_W_bf16 = dt_proj_W.to(torch::kBFloat16);
    pre_dt = torch::mm(pre_x_bf16, dt_proj_W_bf16.t()).to(torch::kFloat32);
    pre_dt.add_(dt_proj_b.unsqueeze(0));
    pre_dt = torch::where(pre_dt > 20.0f, pre_dt, torch::log1p(torch::exp(pre_dt)));

    // B_proj: x_branch @ B_proj_W.T → [N, d_state]
    auto B_proj_W_bf16 = B_proj_W.to(torch::kBFloat16);
    pre_B = torch::mm(pre_x_bf16, B_proj_W_bf16.t()).to(torch::kFloat32);

    // C_proj: x_branch @ C_proj_W.T → [N, d_state]
    auto C_proj_W_bf16 = C_proj_W.to(torch::kBFloat16);
    pre_C = torch::mm(pre_x_bf16, C_proj_W_bf16.t()).to(torch::kFloat32);
}


// ═══════════════════════════════════════════════════════════════════════
//  CDNA3 Launchers
//
//  Single-param step: delegates to CDNA2 (BF16 conversion overhead
//  exceeds MFMA benefit for small single-param GEMMs).
//
//  Batched step: uses refactored pipeline with BF16 MFMA precompute:
//    1. batched_step_setup_and_sort() — shared setup
//    2. cdna3_precompute_bf16() — BF16 projection GEMMs via MFMA
//    3. batched_step_scan_and_fused_elem() — shared scan + fused_elem
//
//  Bilevel/backward: delegate to CDNA2 (wavefront-64 via platform.h).
//
//  MI300X L2 (256MB): meta-net weights (~50KB) are always L2-resident
//  after first access. No explicit L2 management needed.
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_step_cdna3(
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
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
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
    // Single-param: CDNA2 path (wavefront-64, FP32 MFMA)
    // BF16 conversion overhead exceeds benefit for small single-param GEMMs
    launch_mamba3_peer_step_cdna2(
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
        expert_hidden, num_experts,
        expert_counts);
}

void launch_mamba3_peer_batched_step_cdna3(
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
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts
) {
    // Phase 1: Shared setup — input projection, CUB sort, packing
    auto ctx = batched_step_setup_and_sort(
        grads, sharpness_list, mamba_fwd_states, mamba_bwd_states,
        input_proj_W, input_proj_b, d_model, d_state, d_inner);

    if (ctx.total_N == 0) return;

    // Phase 2: BF16 MFMA precompute — real BF16 GEMMs via rocBLAS
    // Inputs cast to BF16 at the GEMM boundary → MFMA_F32_32x32x8_BF16
    // Output is FP32 for scan numerical stability
    torch::Tensor fwd_pre_x, fwd_pre_z, fwd_pre_dt, fwd_pre_B, fwd_pre_C;
    torch::Tensor bwd_pre_x, bwd_pre_z, bwd_pre_dt, bwd_pre_B, bwd_pre_C;

    cdna3_precompute_bf16(
        ctx.x_sorted_packed,
        mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
        mamba_fwd_B_proj, mamba_fwd_C_proj,
        fwd_pre_x, fwd_pre_z, fwd_pre_dt, fwd_pre_B, fwd_pre_C,
        ctx.total_N, d_model, d_inner, d_state);

    cdna3_precompute_bf16(
        ctx.x_sorted_packed,
        mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
        mamba_bwd_B_proj, mamba_bwd_C_proj,
        bwd_pre_x, bwd_pre_z, bwd_pre_dt, bwd_pre_B, bwd_pre_C,
        ctx.total_N, d_model, d_inner, d_state);

    // Phase 3: Shared scan + fused_elem (CDNA2 wavefront-64 scan)
    batched_step_scan_and_fused_elem(
        ctx,
        fwd_pre_x, fwd_pre_z, fwd_pre_dt, fwd_pre_B, fwd_pre_C,
        bwd_pre_x, bwd_pre_z, bwd_pre_dt, bwd_pre_B, bwd_pre_C,
        params, grads, sharpness_list, exp_avgs, exp_avg_sqs, mus,
        gru_states, mamba_fwd_states, mamba_bwd_states,
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
        expert_hidden, num_experts,
        expert_counts);
}

void launch_mamba3_peer_bilevel_fwd_save_batched_cdna3(
    std::vector<torch::Tensor> grads, std::vector<torch::Tensor> sharpness_list,
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
    torch::Tensor fwd_scan_out_packed, torch::Tensor bwd_scan_out_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int checkpoint_interval
) {
    // Bilevel fwd_save: CDNA2 path (wavefront-64, FP32 MFMA)
    launch_mamba3_peer_bilevel_fwd_save_batched_cdna2(
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

void launch_mamba3_peer_backward_batched_cdna3(
    torch::Tensor d_fwd_scan_out_packed, torch::Tensor d_bwd_scan_out_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor fwd_saved_states_packed, torch::Tensor fwd_saved_xb_packed,
    torch::Tensor fwd_saved_z_packed, torch::Tensor fwd_saved_dt_packed,
    torch::Tensor bwd_saved_states_packed, torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed, torch::Tensor bwd_saved_dt_packed,
    torch::Tensor offsets_t,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor d_mamba_fwd_in_proj, torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b, torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj, torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D, torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_bwd_in_proj, torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b, torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj, torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D, torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_x_sorted_packed,
    torch::Tensor fwd_initial_states, torch::Tensor bwd_initial_states,
    int d_model, int d_state, int d_inner, int num_params,
    int checkpoint_interval
) {
    // Backward: CDNA2 path (wavefront-64, FP32 MFMA)
    launch_mamba3_peer_backward_batched_cdna2(
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
