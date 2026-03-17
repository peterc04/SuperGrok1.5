/*
 * Fused multi-tensor gradient preparation kernel.
 *
 * For ALL parameters in one kernel launch:
 *   1. Compute gradient norms
 *   2. Clip gradients that exceed threshold
 *   3. Replace non-finite values with zero
 *   4. Compute bias corrections (bc1, bc2, alpha_mu, lamb_eff)
 *
 * Replaces N per-parameter Python iterations + 4N kernel launches with 1 launch.
 */

#include "ops.h"
#include "dispatch.h"
#include <cmath>

__launch_bounds__(256, 8)
__global__ void multi_tensor_grad_prepare_kernel(
    // Arrays of pointers (one per parameter):
    float** __restrict__ grad_ptrs,         // [num_params] -> each param's grad
    const int* __restrict__ sizes,          // [num_params] -> numel per param
    const float* __restrict__ layer_alphas, // [num_params]
    const float* __restrict__ layer_beta1s, // [num_params]
    const int* __restrict__ steps,          // [num_params]
    // Outputs:
    float* __restrict__ alpha_mus_out,      // [num_params]
    float* __restrict__ bc1s_out,           // [num_params]
    float* __restrict__ bc2s_out,           // [num_params]
    float* __restrict__ lamb_effs_out,      // [num_params]
    // Scalars:
    float base_alpha, float gradient_clipping, float beta2,
    float lamb, float ramp, float gate_signal,
    int num_params
) {
    // Each block handles one parameter
    int param_idx = blockIdx.x;
    if (param_idx >= num_params) return;

    float* grad = grad_ptrs[param_idx];
    int N = sizes[param_idx];

    // Step 1: Compute grad norm (parallel reduction within block)
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float g = grad[i];
        local_sq += g * g;
    }
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);

    // Cross-warp reduce via shared memory
    __shared__ float s_partial[8];  // up to 8 warps (256 threads / 32)
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        s_partial[warp_id] = local_sq;
    }
    __syncthreads();

    __shared__ float s_norm;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            total += s_partial[w];
        }
        s_norm = sqrtf(total);
    }
    __syncthreads();
    float grad_norm = s_norm;

    // Step 2: Clip + finite check (fused)
    if (grad_norm > gradient_clipping) {
        float scale = gradient_clipping / (grad_norm + 1e-12f);
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float g = grad[i];
            g = isfinite(g) ? g * scale : 0.0f;  // Clip + finite in one pass
            grad[i] = g;
        }
    } else {
        // Just finite check
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float g = grad[i];
            if (!isfinite(g)) grad[i] = 0.0f;
        }
    }

    // Step 3: Compute per-parameter scalars (thread 0 only)
    if (threadIdx.x == 0) {
        float alpha_i = fmaxf(0.0f, fminf(1.0f, base_alpha * layer_alphas[param_idx]));
        float beta1_i = layer_beta1s[param_idx];
        int step_i = steps[param_idx];
        alpha_mus_out[param_idx] = alpha_i;
        bc1s_out[param_idx] = 1.0f - powf(beta1_i, (float)step_i);
        bc2s_out[param_idx] = 1.0f - powf(beta2, (float)step_i);
        lamb_effs_out[param_idx] = lamb * ramp * gate_signal;
    }
}


// ── C++ launcher: prepare + batched step ────────────────────────────

void supergrok2_prepare_and_batched_step(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> sharpnesses,
    std::vector<int64_t> steps,
    std::vector<double> layer_alphas,
    std::vector<double> layer_beta1s,
    double base_alpha, double gradient_clipping,
    double beta2, double lr, double eps, double wd,
    double lamb, double ramp, double gate_signal,
    // Meta-net weights
    torch::Tensor mamba_fwd_A, torch::Tensor mamba_fwd_B,
    torch::Tensor mamba_fwd_C, torch::Tensor mamba_fwd_D,
    torch::Tensor mamba_fwd_dt,
    torch::Tensor mamba_bwd_A, torch::Tensor mamba_bwd_B,
    torch::Tensor mamba_bwd_C, torch::Tensor mamba_bwd_D,
    torch::Tensor mamba_bwd_dt,
    torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh,
    torch::Tensor gru_bz, torch::Tensor gru_br, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws,
    torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor value_proj_W,
    int64_t d_inner, int64_t d_state, int64_t n_experts, int64_t topk
) {
    int num_params = static_cast<int>(params.size());
    if (num_params == 0) return;

    auto device = params[0].device();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(device);

    // Pack pointer arrays and metadata on device
    auto grad_ptrs_cpu = torch::zeros({num_params}, torch::TensorOptions().dtype(torch::kInt64));
    auto sizes_cpu = torch::zeros({num_params}, torch::TensorOptions().dtype(torch::kInt32));
    auto layer_alphas_t = torch::zeros({num_params}, torch::TensorOptions().dtype(torch::kFloat32));
    auto layer_beta1s_t = torch::zeros({num_params}, torch::TensorOptions().dtype(torch::kFloat32));
    auto steps_t = torch::zeros({num_params}, torch::TensorOptions().dtype(torch::kInt32));

    for (int i = 0; i < num_params; i++) {
        grad_ptrs_cpu.data_ptr<int64_t>()[i] = reinterpret_cast<int64_t>(grads[i].data_ptr<float>());
        sizes_cpu.data_ptr<int32_t>()[i] = static_cast<int32_t>(grads[i].numel());
        layer_alphas_t.data_ptr<float>()[i] = static_cast<float>(layer_alphas[i]);
        layer_beta1s_t.data_ptr<float>()[i] = static_cast<float>(layer_beta1s[i]);
        steps_t.data_ptr<int32_t>()[i] = static_cast<int32_t>(steps[i]);
    }

    auto grad_ptrs_d = grad_ptrs_cpu.to(device);
    auto sizes_d = sizes_cpu.to(device);
    auto layer_alphas_d = layer_alphas_t.to(device);
    auto layer_beta1s_d = layer_beta1s_t.to(device);
    auto steps_d = steps_t.to(device);

    // Output buffers
    auto alpha_mus_d = torch::zeros({num_params}, opts_f);
    auto bc1s_d = torch::zeros({num_params}, opts_f);
    auto bc2s_d = torch::zeros({num_params}, opts_f);
    auto lamb_effs_d = torch::zeros({num_params}, opts_f);

    // Launch preparation kernel (1 block per param)
    multi_tensor_grad_prepare_kernel<<<num_params, 256>>>(
        reinterpret_cast<float**>(grad_ptrs_d.data_ptr<int64_t>()),
        sizes_d.data_ptr<int32_t>(),
        layer_alphas_d.data_ptr<float>(),
        layer_beta1s_d.data_ptr<float>(),
        steps_d.data_ptr<int32_t>(),
        alpha_mus_d.data_ptr<float>(),
        bc1s_d.data_ptr<float>(),
        bc2s_d.data_ptr<float>(),
        lamb_effs_d.data_ptr<float>(),
        static_cast<float>(base_alpha),
        static_cast<float>(gradient_clipping),
        static_cast<float>(beta2),
        static_cast<float>(lamb),
        static_cast<float>(ramp),
        static_cast<float>(gate_signal),
        num_params);

    // Now launch the existing batched step with pre-computed scalars
    supergrok2_mamba_peer_batched_step(
        params, grads, exp_avgs, exp_avg_sqs,
        mamba_fwd_states, mamba_bwd_states, gru_states,
        mus, sharpnesses,
        alpha_mus_d.unbind(0),
        lamb_effs_d.unbind(0),
        bc1s_d.unbind(0),
        bc2s_d.unbind(0),
        steps,
        mamba_fwd_A, mamba_fwd_B, mamba_fwd_C, mamba_fwd_D, mamba_fwd_dt,
        mamba_bwd_A, mamba_bwd_B, mamba_bwd_C, mamba_bwd_D, mamba_bwd_dt,
        gru_Wz, gru_Wr, gru_Wh, gru_bz, gru_br, gru_bh,
        peer_query_Ws, prod_keys_A, prod_keys_B, value_proj_W,
        lr, beta2, eps, wd, d_inner, d_state, n_experts, topk
    );
}
