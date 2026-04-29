/*
 * SuperGrok v2 — CPU Reference Kernels
 *
 * OpenMP-parallelized CPU implementations for all optimizer components.
 * NOT for training — for debugging, CI testing, and checkpoint inspection.
 *
 * Performance target: N=65536 step < 5 seconds.
 *
 * Architecture: sequential scan over timesteps (O(N)), OpenMP over
 * d_inner for projections and over N for element-wise operations.
 */

#include <torch/extension.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

// ═══════════════════════════════════════════════════════════════════
//  Constants (matching CUDA types.h)
// ═══════════════════════════════════════════════════════════════════

static constexpr int MAX_D_STATE = 32;
static constexpr int MAX_D_INNER = 32;
static constexpr int MAX_D_MODEL = 16;
static constexpr int MAX_GRU_HIDDEN = 8;
static constexpr int MAX_TOPK = 4;

// ═══════════════════════════════════════════════════════════════════
//  CPU Mamba-3 Scan (Sequential + OpenMP over projections)
//
//  Processes timesteps sequentially (required by recurrence).
//  Parallelizes projections and state update over d_inner.
// ═══════════════════════════════════════════════════════════════════

static void mamba3_scan_cpu(
    const float* x_sorted,      // [N, d_model]
    const float* in_proj_W,     // [2*d_inner, d_model]
    const float* dt_proj_W,     // [d_inner, d_inner]
    const float* dt_proj_b,     // [d_inner]
    const float* B_proj_W,      // [d_state, d_inner]
    const float* C_proj_W,      // [d_state, d_inner]
    const float* A_log,         // [d_inner, d_state]
    const float* D_param,       // [d_inner]
    const float* rope_freq,     // [d_inner, d_state/2]
    float* scan_output,         // [N, d_inner]
    float* final_state,         // [d_inner, d_state]
    const float* initial_state, // [d_inner, d_state] or nullptr
    int N, int d_model, int d_inner, int d_state, int reverse
) {
    if (N == 0) return;

    const int half_d_state = d_state / 2;

    // Allocate workspace
    std::vector<float> h(d_inner * d_state, 0.0f);
    std::vector<float> x_branch(d_inner), z_branch(d_inner);

    // Initialize state
    if (initial_state) {
        std::copy(initial_state, initial_state + d_inner * d_state, h.data());
    }

    for (int step = 0; step < N; step++) {
        int i = reverse ? (N - 1 - step) : step;

        // Input projection: split into x_branch and z_branch
        #pragma omp parallel for schedule(static) if(d_inner > 4)
        for (int j = 0; j < d_inner; j++) {
            float xv = 0.0f, zv = 0.0f;
            for (int d = 0; d < d_model; d++) {
                float inp = x_sorted[i * d_model + d];
                xv += in_proj_W[j * d_model + d] * inp;
                zv += in_proj_W[(j + d_inner) * d_model + d] * inp;
            }
            x_branch[j] = xv;
            z_branch[j] = zv;
        }

        // dt projection (needs all x_branch values)
        std::vector<float> dt_val(d_inner);
        #pragma omp parallel for schedule(static) if(d_inner > 4)
        for (int j = 0; j < d_inner; j++) {
            float dt_raw = dt_proj_b[j];
            for (int k = 0; k < d_inner; k++)
                dt_raw += dt_proj_W[j * d_inner + k] * x_branch[k];
            // softplus
            dt_val[j] = (dt_raw > 20.0f) ? dt_raw : std::log(1.0f + std::exp(dt_raw));
        }

        // State update + output (parallelizable over j)
        #pragma omp parallel for schedule(static) if(d_inner > 4)
        for (int j = 0; j < d_inner; j++) {
            float dt = dt_val[j];
            float x_val = x_branch[j];
            float y_val = 0.0f;

            for (int p = 0; p < half_d_state; p++) {
                int se = 2 * p, so = 2 * p + 1;

                // A discretization (trapezoidal)
                float A_e = -std::exp(A_log[j * d_state + se]);
                float A_o = -std::exp(A_log[j * d_state + so]);
                float A_bar_e = (1.0f + dt * A_e / 2.0f)
                              / (1.0f - dt * A_e / 2.0f + 1e-8f);
                float A_bar_o = (1.0f + dt * A_o / 2.0f)
                              / (1.0f - dt * A_o / 2.0f + 1e-8f);

                // B projection
                float B_e = 0.0f, B_o = 0.0f;
                for (int k = 0; k < d_inner; k++) {
                    B_e += B_proj_W[se * d_inner + k] * x_branch[k];
                    B_o += B_proj_W[so * d_inner + k] * x_branch[k];
                }

                // RoPE rotation
                float freq_p = rope_freq[j * half_d_state + p];
                float cos_p = std::cos(dt * freq_p);
                float sin_p = std::sin(dt * freq_p);
                float h_e = h[j * d_state + se];
                float h_o = h[j * d_state + so];
                float h_rot_e = h_e * cos_p - h_o * sin_p;
                float h_rot_o = h_o * cos_p + h_e * sin_p;

                // State update
                h[j * d_state + se] = A_bar_e * h_rot_e + dt * B_e * x_val;
                h[j * d_state + so] = A_bar_o * h_rot_o + dt * B_o * x_val;

                // C projection for output
                float C_e = 0.0f, C_o = 0.0f;
                for (int k = 0; k < d_inner; k++) {
                    C_e += C_proj_W[se * d_inner + k] * x_branch[k];
                    C_o += C_proj_W[so * d_inner + k] * x_branch[k];
                }
                y_val += h[j * d_state + se] * C_e + h[j * d_state + so] * C_o;
            }

            // Gated output: y * silu(z) + D * x
            float z = z_branch[j];
            float silu_z = z / (1.0f + std::exp(-z));
            scan_output[i * d_inner + j] = y_val * silu_z + D_param[j] * x_val;
        }
    }

    // Copy final state
    std::copy(h.begin(), h.end(), final_state);
}


// ═══════════════════════════════════════════════════════════════════
//  CPU Fused Element Step (GRU + PEER + Expert + Adam)
//
//  Embarrassingly parallel over N elements.
//  Each element: GRU update, PEER routing, expert MLP, Adam.
// ═══════════════════════════════════════════════════════════════════

static void fused_elem_step_cpu(
    float* param,               // [N] modified in-place
    const float* grad,          // [N]
    const float* sharpness,     // [N]
    float* exp_avg,             // [N]
    float* exp_avg_sq,          // [N]
    float* mu,                  // [N]
    float* gru_state,           // [N, gru_hidden]
    const float* fwd_ctx,       // [N, d_model]
    const float* bwd_ctx,       // [N, d_model]
    // GRU weights
    const float* gru_Wz, const float* gru_bz,
    const float* gru_Wr, const float* gru_br,
    const float* gru_Wh, const float* gru_bh,
    // PEER weights
    const float* peer_query_Ws, const float* prod_keys_A, const float* prod_keys_B,
    // Expert weights
    const float* expert_W1, const float* expert_b1,
    const float* expert_W2, const float* expert_b2,
    // Expert tracking
    int* expert_counts,
    // Scalars
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int N, int d_model, int gru_hidden,
    int num_heads, int pk_dim, int expert_hidden, int num_experts
) {
    // GRU input dim = grad + sharpness + fwd_ctx + bwd_ctx = 2 + 2*d_model
    const int gru_input_dim = 2 + 2 * d_model;
    const int gru_total_dim = gru_input_dim + gru_hidden;
    // PEER input dim = gru_hidden + 2*d_model + 2
    const int peer_input_dim = gru_hidden + 2 * d_model + 2;
    const int half_d = d_model / 2;
    const int topk = MAX_TOPK;

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < N; idx++) {
        float g = grad[idx];
        float s = sharpness[idx];

        // Build GRU input: [g, s, fwd_ctx, bwd_ctx]
        float gru_inp[MAX_D_MODEL * 2 + MAX_GRU_HIDDEN + 2];
        gru_inp[0] = g;
        gru_inp[1] = s;
        for (int d = 0; d < d_model; d++) {
            gru_inp[2 + d] = fwd_ctx[idx * d_model + d];
            gru_inp[2 + d_model + d] = bwd_ctx[idx * d_model + d];
        }

        // Concat with h_old for GRU
        float xh[MAX_D_MODEL * 2 + MAX_GRU_HIDDEN * 2 + 4];
        for (int d = 0; d < gru_input_dim; d++)
            xh[d] = gru_inp[d];
        for (int d = 0; d < gru_hidden; d++)
            xh[gru_input_dim + d] = gru_state[idx * gru_hidden + d];

        // GRU: z_gate = sigmoid(Wz @ xh + bz)
        float z_gate[MAX_GRU_HIDDEN], r_gate[MAX_GRU_HIDDEN], h_tilde[MAX_GRU_HIDDEN];
        for (int h = 0; h < gru_hidden; h++) {
            float z_raw = gru_bz[h], r_raw = gru_br[h];
            for (int d = 0; d < gru_total_dim; d++) {
                z_raw += gru_Wz[h * gru_total_dim + d] * xh[d];
                r_raw += gru_Wr[h * gru_total_dim + d] * xh[d];
            }
            z_gate[h] = 1.0f / (1.0f + std::exp(-z_raw));
            r_gate[h] = 1.0f / (1.0f + std::exp(-r_raw));
        }

        // h_tilde = tanh(Wh @ [x, r*h] + bh)
        float xrh[MAX_D_MODEL * 2 + MAX_GRU_HIDDEN * 2 + 4];
        for (int d = 0; d < gru_input_dim; d++)
            xrh[d] = gru_inp[d];
        for (int d = 0; d < gru_hidden; d++)
            xrh[gru_input_dim + d] = r_gate[d] * gru_state[idx * gru_hidden + d];

        for (int h = 0; h < gru_hidden; h++) {
            float val = gru_bh[h];
            for (int d = 0; d < gru_total_dim; d++)
                val += gru_Wh[h * gru_total_dim + d] * xrh[d];
            h_tilde[h] = std::tanh(val);
        }

        // h_new = (1-z)*h_old + z*h_tilde
        for (int h = 0; h < gru_hidden; h++) {
            float h_old = gru_state[idx * gru_hidden + h];
            gru_state[idx * gru_hidden + h] =
                (1.0f - z_gate[h]) * h_old + z_gate[h] * h_tilde[h];
        }

        // Build PEER input: [gru_state, fwd_ctx, bwd_ctx, g, s]
        float peer_inp[MAX_GRU_HIDDEN + MAX_D_MODEL * 2 + 4];
        for (int d = 0; d < gru_hidden; d++)
            peer_inp[d] = gru_state[idx * gru_hidden + d];
        for (int d = 0; d < d_model; d++) {
            peer_inp[gru_hidden + d] = fwd_ctx[idx * d_model + d];
            peer_inp[gru_hidden + d_model + d] = bwd_ctx[idx * d_model + d];
        }
        peer_inp[gru_hidden + 2 * d_model] = g;
        peer_inp[gru_hidden + 2 * d_model + 1] = s;

        // PEER routing: multi-head product-key with hard routing
        float total_expert_out = 0.0f;

        for (int head = 0; head < num_heads; head++) {
            // Query projection: [d_model] = W[d_model, peer_input_dim] @ peer_inp
            float query[MAX_D_MODEL];
            const float* W_head = peer_query_Ws + head * d_model * peer_input_dim;
            for (int d = 0; d < d_model; d++) {
                float val = 0.0f;
                for (int k = 0; k < peer_input_dim; k++)
                    val += W_head[d * peer_input_dim + k] * peer_inp[k];
                query[d] = val;
            }

            // Product-key: split query into A and B halves
            const float* pka = prod_keys_A + head * pk_dim * half_d;
            const float* pkb = prod_keys_B + head * pk_dim * half_d;

            // Compute scores for A keys
            float scores_a[144];  // pk_dim max
            for (int k = 0; k < pk_dim; k++) {
                float val = 0.0f;
                for (int d = 0; d < half_d; d++)
                    val += pka[k * half_d + d] * query[d];
                scores_a[k] = val;
            }

            // Compute scores for B keys
            float scores_b[144];
            for (int k = 0; k < pk_dim; k++) {
                float val = 0.0f;
                for (int d = 0; d < half_d; d++)
                    val += pkb[k * half_d + d] * query[half_d + d];
                scores_b[k] = val;
            }

            // Top-k for each sub-key (hard routing: argmax top-k)
            int top_a[MAX_TOPK], top_b[MAX_TOPK];
            for (int t = 0; t < topk; t++) {
                int best_a = 0, best_b = 0;
                for (int k = 1; k < pk_dim; k++) {
                    if (scores_a[k] > scores_a[best_a]) best_a = k;
                    if (scores_b[k] > scores_b[best_b]) best_b = k;
                }
                top_a[t] = best_a;
                top_b[t] = best_b;
                scores_a[best_a] = -1e30f;
                scores_b[best_b] = -1e30f;
            }

            // Combine: each (a,b) pair selects expert a*pk_dim+b
            for (int ta = 0; ta < topk; ta++) {
                for (int tb = 0; tb < topk; tb++) {
                    int expert_id = top_a[ta] * pk_dim + top_b[tb];
                    if (expert_id >= num_experts) continue;

                    // Expert MLP: W1*g + b1 -> relu -> W2*z + b2
                    float hidden[16];  // MAX_EXPERT_HIDDEN
                    for (int h = 0; h < expert_hidden; h++) {
                        float val = expert_b1[expert_id * expert_hidden + h]
                                  + expert_W1[expert_id * expert_hidden + 0] * g;
                        hidden[h] = (val > 0.0f) ? val : 0.0f;  // ReLU
                    }

                    float expert_val = expert_b2[expert_id];
                    for (int h = 0; h < expert_hidden; h++)
                        expert_val += expert_W2[expert_id * expert_hidden + h] * hidden[h];

                    total_expert_out += expert_val / (float)(topk * topk * num_heads);

                    // Track expert usage (atomic in multi-threaded context)
                    #pragma omp atomic
                    expert_counts[expert_id] += 1;
                }
            }
        }

        // Smart grad
        float smart_g = g + rescale * total_expert_out;

        // Mu EMA
        mu[idx] = alpha_mu * mu[idx] + (1.0f - alpha_mu) * g;
        float effective = smart_g + lamb_eff * mu[idx];

        // Adam
        exp_avg[idx] = beta1 * exp_avg[idx] + (1.0f - beta1) * effective;
        exp_avg_sq[idx] = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * effective * effective;

        float step_size = lr / bc1;
        float denom = std::sqrt(exp_avg_sq[idx] / bc2) + eps;

        float p_val = param[idx];
        p_val = p_val * (1.0f - lr * wd_eff) - step_size * exp_avg[idx] / denom;
        param[idx] = p_val;
    }
}


// ═══════════════════════════════════════════════════════════════════
//  CPU Full Step Launcher
//
//  Combines: sort → scan (fwd+bwd) → unsort → output projection →
//  fused element step (GRU + PEER + Expert + Adam)
// ═══════════════════════════════════════════════════════════════════

void supergrok2_cpu_step(
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
    int N = grad.numel();
    if (N == 0) return;

    // Ensure contiguous FP32
    auto g_f = grad.to(torch::kFloat32).reshape(-1).contiguous();
    auto s_f = sharpness.to(torch::kFloat32).reshape(-1).contiguous();
    auto p_f = param.to(torch::kFloat32).reshape(-1).contiguous();

    // Sort by |gradient| magnitude
    auto sort_result = g_f.abs().sort(/*dim=*/0, /*descending=*/false);
    auto sort_indices = std::get<1>(sort_result);
    auto unsort_indices = sort_indices.argsort();

    // Input projection: [g_sorted, s_sorted] -> [N, d_model]
    auto g_sorted = g_f.index_select(0, sort_indices);
    auto s_sorted = s_f.index_select(0, sort_indices);
    auto inp = torch::stack({g_sorted, s_sorted}, /*dim=*/1);  // [N, 2]
    auto x_proj = torch::addmm(input_proj_b, inp, input_proj_W.t());  // [N, d_model]
    x_proj = x_proj.contiguous();

    // Forward Mamba scan
    auto fwd_scan_out = torch::empty({N, d_inner}, torch::kFloat32);
    auto fwd_final = torch::empty({d_inner, d_state}, torch::kFloat32);
    mamba3_scan_cpu(
        x_proj.data_ptr<float>(),
        mamba_fwd_in_proj.data_ptr<float>(),
        mamba_fwd_dt_W.data_ptr<float>(),
        mamba_fwd_dt_b.data_ptr<float>(),
        mamba_fwd_B_proj.data_ptr<float>(),
        mamba_fwd_C_proj.data_ptr<float>(),
        mamba_fwd_A_log.data_ptr<float>(),
        mamba_fwd_D.data_ptr<float>(),
        mamba_fwd_rope.data_ptr<float>(),
        fwd_scan_out.data_ptr<float>(),
        fwd_final.data_ptr<float>(),
        mamba_fwd_state.numel() > 0 ? mamba_fwd_state.data_ptr<float>() : nullptr,
        N, d_model, d_inner, d_state, 0);

    // Backward Mamba scan
    auto bwd_scan_out = torch::empty({N, d_inner}, torch::kFloat32);
    auto bwd_final = torch::empty({d_inner, d_state}, torch::kFloat32);
    mamba3_scan_cpu(
        x_proj.data_ptr<float>(),
        mamba_bwd_in_proj.data_ptr<float>(),
        mamba_bwd_dt_W.data_ptr<float>(),
        mamba_bwd_dt_b.data_ptr<float>(),
        mamba_bwd_B_proj.data_ptr<float>(),
        mamba_bwd_C_proj.data_ptr<float>(),
        mamba_bwd_A_log.data_ptr<float>(),
        mamba_bwd_D.data_ptr<float>(),
        mamba_bwd_rope.data_ptr<float>(),
        bwd_scan_out.data_ptr<float>(),
        bwd_final.data_ptr<float>(),
        mamba_bwd_state.numel() > 0 ? mamba_bwd_state.data_ptr<float>() : nullptr,
        N, d_model, d_inner, d_state, 1);

    // Update Mamba states
    mamba_fwd_state.copy_(fwd_final);
    mamba_bwd_state.copy_(bwd_final);

    // Output projection: unsort, then project scan output to d_model
    auto fwd_unsorted = fwd_scan_out.index_select(0, unsort_indices);
    auto bwd_unsorted = bwd_scan_out.index_select(0, unsort_indices);

    // Project scan output to context: [N, d_inner] @ [d_model, d_inner].T -> [N, d_model]
    auto fwd_ctx = torch::mm(fwd_unsorted, mamba_fwd_out_proj.t());  // [N, d_model]
    auto bwd_ctx = torch::mm(bwd_unsorted, mamba_bwd_out_proj.t());  // [N, d_model]
    fwd_ctx = fwd_ctx.contiguous();
    bwd_ctx = bwd_ctx.contiguous();

    // Fused element step
    fused_elem_step_cpu(
        p_f.data_ptr<float>(), g_f.data_ptr<float>(), s_f.data_ptr<float>(),
        exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
        mu.data_ptr<float>(), gru_state.data_ptr<float>(),
        fwd_ctx.data_ptr<float>(), bwd_ctx.data_ptr<float>(),
        gru_Wz.data_ptr<float>(), gru_bz.data_ptr<float>(),
        gru_Wr.data_ptr<float>(), gru_br.data_ptr<float>(),
        gru_Wh.data_ptr<float>(), gru_bh.data_ptr<float>(),
        peer_query_Ws.data_ptr<float>(),
        prod_keys_A.data_ptr<float>(), prod_keys_B.data_ptr<float>(),
        expert_W1.data_ptr<float>(), expert_b1.data_ptr<float>(),
        expert_W2.data_ptr<float>(), expert_b2.data_ptr<float>(),
        expert_counts.data_ptr<int>(),
        rescale, alpha_mu, lamb_eff,
        beta1, beta2, lr, wd_eff, eps, bc1, bc2,
        N, d_model, gru_hidden,
        num_heads, pk_dim, expert_hidden, num_experts);

    // Write back parameters (handle dtype conversion)
    param.reshape(-1).copy_(p_f);
}
