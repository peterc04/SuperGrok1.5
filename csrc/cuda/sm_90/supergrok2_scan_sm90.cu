/*
 * SuperGrok v2 — Hopper Forward Kernels (sm_90+)
 *
 * Contains real FP8 E4M3 precompute code (hopper_precompute_fp8,
 * hopper_fp8_gemm) that uses cublasGemmEx with CUDA_R_8F_E4M3 inputs
 * and FP32 accumulation for projection GEMMs when N >= 4096.
 * Per-tensor absmax scaling: scale = max(|tensor|) / 448.0.
 *
 * Batched step uses the refactored pipeline: FP8 precompute (projection
 * GEMMs via hopper_precompute_fp8) followed by shared scan + fused_elem
 * from the Ampere path. The scan recurrence remains FP32 for numerical
 * stability.
 *
 * Single-param step delegates to Ampere — FP8 precompute does not
 * benefit small N because the .item() CPU-GPU sync for absmax scaling
 * dominates the GEMM speedup.
 *
 * Note on TMA: The Tensor Memory Accelerator is NOT used here because
 * the scan's per-timestep access pattern (scattered reads indexed by
 * sort order) is not suited to TMA's bulk copy model. TMA would require
 * descriptor setup per sort permutation, negating any benefit.
 *
 * Guarded with #if CUDA_VERSION >= 11080 for FP8 type availability.
 *
 * Dispatch: ops.cpp calls these on sm_90+ GPUs.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include "platform.h"
#include "types.h"
#include "dispatch.h"
#include "ops.h"

#if GROK_CUDA
#include <cuda_pipeline.h>
#endif

// ═══════════════════════════════════════════════════════════════════════
//  Forward declarations of Ampere launchers
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_step_ampere(
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

void launch_mamba3_peer_batched_step_ampere(
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


// ═══════════════════════════════════════════════════════════════════════
//  FP8 Precompute Helper
//
//  Converts projection inputs to FP8 E4M3 and runs cuBLAS GEMMs with
//  FP8 inputs and FP32 output. Per-tensor absmax scaling:
//    scale = max(|tensor|) / 448.0  (FP8 E4M3 max representable)
//
//  Note: .item() calls cause CPU-GPU sync. Acceptable for correctness;
//  production should use a CUDA max-reduce kernel for input scales and
//  cache weight scales per _weights_dirty flip.
// ═══════════════════════════════════════════════════════════════════════

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080

static void hopper_fp8_gemm(
    cublasHandle_t handle,
    torch::Tensor input,       // [M, K] FP32
    torch::Tensor weight,      // [N, K] FP32 (transposed in GEMM)
    torch::Tensor output,      // [M, N] FP32
    int M, int N, int K
) {
    // Per-tensor absmax scaling for FP8
    float input_scale = input.abs().max().item<float>() / 448.0f;
    float weight_scale = weight.abs().max().item<float>() / 448.0f;

    // Clamp scales to avoid division by zero
    if (input_scale < 1e-12f) input_scale = 1e-12f;
    if (weight_scale < 1e-12f) weight_scale = 1e-12f;

    // Convert to FP8 E4M3
    auto input_fp8 = (input / input_scale).to(torch::kFloat8_e4m3fn).contiguous();
    auto weight_fp8 = (weight / weight_scale).to(torch::kFloat8_e4m3fn).contiguous();

    // FP8 GEMM: output = (input_fp8 @ weight_fp8.T) * input_scale * weight_scale
    float alpha = input_scale * weight_scale;
    float beta = 0.0f;

    // cuBLAS: C = alpha * op(A) * op(B) + beta * C
    // We want: output[M,N] = input[M,K] @ weight[N,K].T
    // In column-major: C(N,M) = weight(N,K) * input(K,M)
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weight_fp8.data_ptr(), CUDA_R_8F_E4M3, K,
        input_fp8.data_ptr(), CUDA_R_8F_E4M3, K,
        &beta,
        output.data_ptr<float>(), CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

static void hopper_precompute_fp8(
    torch::Tensor x_sorted,          // [N, d_model] FP32
    torch::Tensor in_proj_W,         // [2*d_inner, d_model] FP32
    torch::Tensor dt_proj_W,         // [d_inner, d_inner]
    torch::Tensor dt_proj_b,         // [d_inner]
    torch::Tensor B_proj_W,          // [d_state, d_inner]
    torch::Tensor C_proj_W,          // [d_state, d_inner]
    torch::Tensor pre_x,             // [N, d_inner] output
    torch::Tensor pre_z,             // [N, d_inner] output
    torch::Tensor pre_dt,            // [N, d_inner] output
    torch::Tensor pre_B,             // [N, d_state] output
    torch::Tensor pre_C,             // [N, d_state] output
    int N, int d_model, int d_inner, int d_state
) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    // Split in_proj into x_branch and z_branch weights
    auto in_proj_x = in_proj_W.narrow(0, 0, d_inner);        // [d_inner, d_model]
    auto in_proj_z = in_proj_W.narrow(0, d_inner, d_inner);   // [d_inner, d_model]

    // x_branch = x_sorted @ in_proj_x.T → [N, d_inner]
    hopper_fp8_gemm(handle, x_sorted, in_proj_x, pre_x, N, d_inner, d_model);

    // z_branch = x_sorted @ in_proj_z.T → [N, d_inner]
    hopper_fp8_gemm(handle, x_sorted, in_proj_z, pre_z, N, d_inner, d_model);

    // dt_proj: x_branch @ dt_proj_W.T + bias → softplus
    hopper_fp8_gemm(handle, pre_x, dt_proj_W, pre_dt, N, d_inner, d_inner);

    // Add bias + softplus in-place
    pre_dt.add_(dt_proj_b.unsqueeze(0));
    // softplus: log(1 + exp(x)), with clamp for large values
    auto mask = pre_dt.le(20.0f);
    pre_dt.where(mask, pre_dt).log1p_().where(mask, pre_dt);
    // Simpler: use torch where
    pre_dt = torch::where(pre_dt > 20.0f, pre_dt, torch::log1p(torch::exp(pre_dt)));

    // B_proj: x_branch @ B_proj_W.T → [N, d_state]
    hopper_fp8_gemm(handle, pre_x, B_proj_W, pre_B, N, d_state, d_inner);

    // C_proj: x_branch @ C_proj_W.T → [N, d_state]
    hopper_fp8_gemm(handle, pre_x, C_proj_W, pre_C, N, d_state, d_inner);
}

#endif  // CUDA_VERSION >= 11080


// ═══════════════════════════════════════════════════════════════════════
//  Hopper Forward: Per-Parameter Step
//
//  Uses Ampere path (TF32 + cp.async) as baseline. The FP8 projection
//  optimization applies to the batched step path where N is large enough
//  to benefit from FP8 GEMMs.
//
//  For single-parameter steps, the overhead of FP8 scale computation
//  (CPU-GPU sync via .item()) outweighs the GEMM speedup on small
//  matrices, so we use Ampere TF32 directly.
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_step_hopper(
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
    // Single-parameter step: FP8 scale computation overhead dominates
    // for small matrices. Use Ampere TF32 + cp.async path.
    launch_mamba3_peer_step_ampere(
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


// ═══════════════════════════════════════════════════════════════════════
//  Hopper Forward: Batched Step
//
//  Uses the refactored batched step pipeline to inject FP8 E4M3
//  precompute via cublasGemmEx when total_N >= GEMM_PRECOMPUTE_THRESHOLD
//  and CUDA >= 11.8. Otherwise falls back to generic FP32 precompute.
//
//  Pipeline:
//    1. batched_step_setup_and_sort() — shared setup, CUB sort, packing
//    2. hopper_precompute_fp8() — FP8 projection GEMMs (this is Hopper-specific)
//       or generic_batched_precompute() — fallback for small N
//    3. batched_step_scan_and_fused_elem() — shared scan + fused_elem
//
//  TF32 math mode is set for any torch::mm calls in the scan/fused_elem path.
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_batched_step_hopper(
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
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    // Phase 1: Shared setup — input projection, CUB sort, packing
    auto ctx = batched_step_setup_and_sort(
        grads, sharpness_list, mamba_fwd_states, mamba_bwd_states,
        input_proj_W, input_proj_b, d_model, d_state, d_inner);

    if (ctx.total_N == 0) {
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        return;
    }

    // Phase 2: Precompute projections
    // Use FP8 when total_N is large enough to amortize FP8 scale overhead
    torch::Tensor fwd_pre_x, fwd_pre_z, fwd_pre_dt, fwd_pre_B, fwd_pre_C;
    torch::Tensor bwd_pre_x, bwd_pre_z, bwd_pre_dt, bwd_pre_B, bwd_pre_C;

    bool use_fp8 = false;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    use_fp8 = (ctx.total_N >= GEMM_PRECOMPUTE_THRESHOLD);
#endif

    if (use_fp8) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
        // Hopper FP8 E4M3 precompute — real cublasGemmEx with CUDA_R_8F_E4M3
        auto float_opts = torch::TensorOptions().device(ctx.x_sorted_packed.device()).dtype(torch::kFloat32);
        fwd_pre_x = torch::empty({ctx.total_N, d_inner}, float_opts);
        fwd_pre_z = torch::empty({ctx.total_N, d_inner}, float_opts);
        fwd_pre_dt = torch::empty({ctx.total_N, d_inner}, float_opts);
        fwd_pre_B = torch::empty({ctx.total_N, d_state}, float_opts);
        fwd_pre_C = torch::empty({ctx.total_N, d_state}, float_opts);
        bwd_pre_x = torch::empty({ctx.total_N, d_inner}, float_opts);
        bwd_pre_z = torch::empty({ctx.total_N, d_inner}, float_opts);
        bwd_pre_dt = torch::empty({ctx.total_N, d_inner}, float_opts);
        bwd_pre_B = torch::empty({ctx.total_N, d_state}, float_opts);
        bwd_pre_C = torch::empty({ctx.total_N, d_state}, float_opts);

        hopper_precompute_fp8(
            ctx.x_sorted_packed,
            mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
            mamba_fwd_B_proj, mamba_fwd_C_proj,
            fwd_pre_x, fwd_pre_z, fwd_pre_dt, fwd_pre_B, fwd_pre_C,
            ctx.total_N, d_model, d_inner, d_state);

        hopper_precompute_fp8(
            ctx.x_sorted_packed,
            mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
            mamba_bwd_B_proj, mamba_bwd_C_proj,
            bwd_pre_x, bwd_pre_z, bwd_pre_dt, bwd_pre_B, bwd_pre_C,
            ctx.total_N, d_model, d_inner, d_state);
#endif
    } else {
        // Fall back to generic FP32 precompute kernel for small N
        generic_batched_precompute(
            ctx, mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
            mamba_fwd_B_proj, mamba_fwd_C_proj,
            fwd_pre_x, fwd_pre_z, fwd_pre_dt, fwd_pre_B, fwd_pre_C,
            d_model, d_inner, d_state);

        generic_batched_precompute(
            ctx, mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
            mamba_bwd_B_proj, mamba_bwd_C_proj,
            bwd_pre_x, bwd_pre_z, bwd_pre_dt, bwd_pre_B, bwd_pre_C,
            d_model, d_inner, d_state);
    }

    // Phase 3: cp.async scan + cp.async fused_elem (via Ampere path)
    // Hopper uses Ampere's cp.async scan kernels which provide double-buffered
    // prefetch. Combined with FP8 precompute above, this gives Hopper:
    // FP8 precompute + cp.async scan + cp.async fused_elem.
    ampere_batched_scan_and_fused_elem(
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

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}
