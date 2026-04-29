/*
 * Fused single-pass SG2 step for CPU.
 *
 * Sequential scan + output projection + GRU + PEER + Expert + Adam in one loop.
 * scan_output never leaves registers — consumed immediately after computation.
 * This saves one full read + write of the scan_output buffer through L3 cache.
 *
 * For 125M params at 4 bytes: saves 1 GB of cache traffic per step.
 *
 * Replaces the split path (mamba3_scan_cpu + fused_elem_step_cpu).
 */

#include <torch/extension.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int MAX_D_STATE = 32;
static constexpr int MAX_D_INNER = 32;
static constexpr int MAX_D_MODEL = 16;
static constexpr int MAX_GRU_HIDDEN = 8;
static constexpr int MAX_TOPK = 4;


void cpu_sg2_fused_scan_elem(
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

    const int half_d_state = d_state / 2;
    const int gru_input_dim = 2 + 2 * d_model;
    const int gru_total_dim = gru_input_dim + gru_hidden;
    const int peer_input_dim = gru_hidden + 2 * d_model + 2;
    const int half_d = d_model / 2;
    const int topk = MAX_TOPK;

    // Ensure contiguous FP32
    auto g_f = grad.to(torch::kFloat32).reshape(-1).contiguous();
    auto s_f = sharpness.to(torch::kFloat32).reshape(-1).contiguous();
    auto p_f = param.to(torch::kFloat32).reshape(-1).contiguous();

    // Sort by |gradient| magnitude
    auto sort_result = g_f.abs().sort(0, false);
    auto sort_indices = std::get<1>(sort_result);
    auto si = sort_indices.data_ptr<int64_t>();

    // Input projection: [g_sorted, s_sorted] -> [N, d_model]
    auto g_sorted = g_f.index_select(0, sort_indices);
    auto s_sorted = s_f.index_select(0, sort_indices);
    auto inp = torch::stack({g_sorted, s_sorted}, 1);
    auto x_proj = torch::addmm(input_proj_b, inp, input_proj_W.t()).contiguous();

    // Pre-compute projections for fwd and bwd scans (parallelizable)
    auto* xp = x_proj.data_ptr<float>();
    auto* fwd_in_proj = mamba_fwd_in_proj.data_ptr<float>();
    auto* fwd_dt_W = mamba_fwd_dt_W.data_ptr<float>();
    auto* fwd_dt_b = mamba_fwd_dt_b.data_ptr<float>();
    auto* fwd_B_proj = mamba_fwd_B_proj.data_ptr<float>();
    auto* fwd_C_proj = mamba_fwd_C_proj.data_ptr<float>();
    auto* fwd_A_log = mamba_fwd_A_log.data_ptr<float>();
    auto* fwd_D_param = mamba_fwd_D.data_ptr<float>();
    auto* fwd_rope = mamba_fwd_rope.data_ptr<float>();
    auto* fwd_out_proj = mamba_fwd_out_proj.data_ptr<float>();

    auto* bwd_in_proj = mamba_bwd_in_proj.data_ptr<float>();
    auto* bwd_dt_W = mamba_bwd_dt_W.data_ptr<float>();
    auto* bwd_dt_b = mamba_bwd_dt_b.data_ptr<float>();
    auto* bwd_B_proj = mamba_bwd_B_proj.data_ptr<float>();
    auto* bwd_C_proj = mamba_bwd_C_proj.data_ptr<float>();
    auto* bwd_A_log = mamba_bwd_A_log.data_ptr<float>();
    auto* bwd_D_param = mamba_bwd_D.data_ptr<float>();
    auto* bwd_rope = mamba_bwd_rope.data_ptr<float>();
    auto* bwd_out_proj = mamba_bwd_out_proj.data_ptr<float>();

    // Scan state (fwd and bwd) — stays on stack, never written to RAM
    std::vector<float> h_fwd(d_inner * d_state, 0.0f);
    std::vector<float> h_bwd(d_inner * d_state, 0.0f);

    if (mamba_fwd_state.numel() > 0)
        std::copy_n(mamba_fwd_state.data_ptr<float>(), d_inner * d_state, h_fwd.data());
    if (mamba_bwd_state.numel() > 0)
        std::copy_n(mamba_bwd_state.data_ptr<float>(), d_inner * d_state, h_bwd.data());

    // Pre-compute backward scan output for all timesteps (bwd goes in reverse)
    // This is needed because the bwd scan runs reverse while fwd runs forward.
    // We store bwd scan output but NOT fwd — fwd is consumed immediately.
    std::vector<float> bwd_scan_out(N * d_inner, 0.0f);

    for (int step = 0; step < N; step++) {
        int i = N - 1 - step;  // reverse

        std::vector<float> x_branch(d_inner), z_branch(d_inner);
        #pragma omp parallel for schedule(static) if(d_inner > 4)
        for (int j = 0; j < d_inner; j++) {
            float xv = 0.0f, zv = 0.0f;
            for (int d = 0; d < d_model; d++) {
                float val = xp[i * d_model + d];
                xv += bwd_in_proj[j * d_model + d] * val;
                zv += bwd_in_proj[(j + d_inner) * d_model + d] * val;
            }
            x_branch[j] = xv;
            z_branch[j] = zv;
        }

        std::vector<float> dt_val(d_inner);
        #pragma omp parallel for schedule(static) if(d_inner > 4)
        for (int j = 0; j < d_inner; j++) {
            float dt_raw = bwd_dt_b[j];
            for (int k = 0; k < d_inner; k++)
                dt_raw += bwd_dt_W[j * d_inner + k] * x_branch[k];
            dt_val[j] = (dt_raw > 20.0f) ? dt_raw : std::log(1.0f + std::exp(dt_raw));
        }

        #pragma omp parallel for schedule(static) if(d_inner > 4)
        for (int j = 0; j < d_inner; j++) {
            float dt = dt_val[j];
            float x_val = x_branch[j];
            float y_val = 0.0f;

            for (int p = 0; p < half_d_state; p++) {
                int se = 2 * p, so = 2 * p + 1;
                float A_e = -std::exp(bwd_A_log[j * d_state + se]);
                float A_o = -std::exp(bwd_A_log[j * d_state + so]);
                float A_bar_e = (1.0f + dt * A_e / 2.0f) / (1.0f - dt * A_e / 2.0f + 1e-8f);
                float A_bar_o = (1.0f + dt * A_o / 2.0f) / (1.0f - dt * A_o / 2.0f + 1e-8f);

                float B_e = 0.0f, B_o = 0.0f;
                for (int k = 0; k < d_inner; k++) {
                    B_e += bwd_B_proj[se * d_inner + k] * x_branch[k];
                    B_o += bwd_B_proj[so * d_inner + k] * x_branch[k];
                }

                float freq_p = bwd_rope[j * half_d_state + p];
                float cos_p = std::cos(dt * freq_p);
                float sin_p = std::sin(dt * freq_p);
                float h_e = h_bwd[j * d_state + se];
                float h_o = h_bwd[j * d_state + so];
                float h_rot_e = h_e * cos_p - h_o * sin_p;
                float h_rot_o = h_o * cos_p + h_e * sin_p;

                h_bwd[j * d_state + se] = A_bar_e * h_rot_e + dt * B_e * x_val;
                h_bwd[j * d_state + so] = A_bar_o * h_rot_o + dt * B_o * x_val;

                float C_e = 0.0f, C_o = 0.0f;
                for (int k = 0; k < d_inner; k++) {
                    C_e += bwd_C_proj[se * d_inner + k] * x_branch[k];
                    C_o += bwd_C_proj[so * d_inner + k] * x_branch[k];
                }
                y_val += h_bwd[j * d_state + se] * C_e + h_bwd[j * d_state + so] * C_o;
            }

            float z = z_branch[j];
            float silu_z = z / (1.0f + std::exp(-z));
            bwd_scan_out[i * d_inner + j] = y_val * silu_z + bwd_D_param[j] * x_val;
        }
    }

    // FUSED FORWARD PASS: scan + output projection + GRU + PEER + Expert + Adam
    // Forward scan_output stays in registers — consumed immediately per timestep
    auto* p_ptr = p_f.data_ptr<float>();
    auto* g_ptr = g_f.data_ptr<float>();
    auto* s_ptr = s_f.data_ptr<float>();
    auto* ea_ptr = exp_avg.data_ptr<float>();
    auto* eas_ptr = exp_avg_sq.data_ptr<float>();
    auto* mu_ptr = mu.data_ptr<float>();
    auto* gru_ptr = gru_state.data_ptr<float>();
    auto* ec_ptr = expert_counts.data_ptr<int>();

    auto* gWz = gru_Wz.data_ptr<float>();
    auto* gbz = gru_bz.data_ptr<float>();
    auto* gWr = gru_Wr.data_ptr<float>();
    auto* gbr = gru_br.data_ptr<float>();
    auto* gWh = gru_Wh.data_ptr<float>();
    auto* gbh = gru_bh.data_ptr<float>();
    auto* pqWs = peer_query_Ws.data_ptr<float>();
    auto* pkA = prod_keys_A.data_ptr<float>();
    auto* pkB = prod_keys_B.data_ptr<float>();
    auto* eW1 = expert_W1.data_ptr<float>();
    auto* eb1 = expert_b1.data_ptr<float>();
    auto* eW2 = expert_W2.data_ptr<float>();
    auto* eb2 = expert_b2.data_ptr<float>();

    for (int step = 0; step < N; step++) {
        int i = step;  // forward scan order

        // === Forward scan step ===
        std::vector<float> x_branch(d_inner), z_branch(d_inner);
        #pragma omp parallel for schedule(static) if(d_inner > 4)
        for (int j = 0; j < d_inner; j++) {
            float xv = 0.0f, zv = 0.0f;
            for (int d = 0; d < d_model; d++) {
                float val = xp[i * d_model + d];
                xv += fwd_in_proj[j * d_model + d] * val;
                zv += fwd_in_proj[(j + d_inner) * d_model + d] * val;
            }
            x_branch[j] = xv;
            z_branch[j] = zv;
        }

        std::vector<float> dt_val(d_inner);
        #pragma omp parallel for schedule(static) if(d_inner > 4)
        for (int j = 0; j < d_inner; j++) {
            float dt_raw = fwd_dt_b[j];
            for (int k = 0; k < d_inner; k++)
                dt_raw += fwd_dt_W[j * d_inner + k] * x_branch[k];
            dt_val[j] = (dt_raw > 20.0f) ? dt_raw : std::log(1.0f + std::exp(dt_raw));
        }

        // Compute fwd scan output — stays on stack, NOT written to scan_output array
        float fwd_y[MAX_D_INNER];
        for (int j = 0; j < d_inner; j++) {
            float dt = dt_val[j];
            float x_val = x_branch[j];
            float y_val = 0.0f;

            for (int p = 0; p < half_d_state; p++) {
                int se = 2 * p, so = 2 * p + 1;
                float A_e = -std::exp(fwd_A_log[j * d_state + se]);
                float A_o = -std::exp(fwd_A_log[j * d_state + so]);
                float A_bar_e = (1.0f + dt * A_e / 2.0f) / (1.0f - dt * A_e / 2.0f + 1e-8f);
                float A_bar_o = (1.0f + dt * A_o / 2.0f) / (1.0f - dt * A_o / 2.0f + 1e-8f);

                float B_e = 0.0f, B_o = 0.0f;
                for (int k = 0; k < d_inner; k++) {
                    B_e += fwd_B_proj[se * d_inner + k] * x_branch[k];
                    B_o += fwd_B_proj[so * d_inner + k] * x_branch[k];
                }

                float freq_p = fwd_rope[j * half_d_state + p];
                float cos_p = std::cos(dt * freq_p);
                float sin_p = std::sin(dt * freq_p);
                float h_e = h_fwd[j * d_state + se];
                float h_o = h_fwd[j * d_state + so];
                float h_rot_e = h_e * cos_p - h_o * sin_p;
                float h_rot_o = h_o * cos_p + h_e * sin_p;

                h_fwd[j * d_state + se] = A_bar_e * h_rot_e + dt * B_e * x_val;
                h_fwd[j * d_state + so] = A_bar_o * h_rot_o + dt * B_o * x_val;

                float C_e = 0.0f, C_o = 0.0f;
                for (int k = 0; k < d_inner; k++) {
                    C_e += fwd_C_proj[se * d_inner + k] * x_branch[k];
                    C_o += fwd_C_proj[so * d_inner + k] * x_branch[k];
                }
                y_val += h_fwd[j * d_state + se] * C_e + h_fwd[j * d_state + so] * C_o;
            }

            float z = z_branch[j];
            float silu_z = z / (1.0f + std::exp(-z));
            fwd_y[j] = y_val * silu_z + fwd_D_param[j] * x_val;
        }

        // === Output projection (in registers) ===
        // fwd_ctx and bwd_ctx for this timestep — stay on stack
        float fwd_ctx[MAX_D_MODEL], bwd_ctx[MAX_D_MODEL];

        // Unsort: this timestep in sorted order maps to orig_idx in original order
        int orig_idx = (int)si[i];

        // Project fwd scan output -> d_model context
        for (int d = 0; d < d_model; d++) {
            float val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                val += fwd_y[j] * fwd_out_proj[d * d_inner + j];
            fwd_ctx[d] = val;
        }

        // Get bwd scan output for this timestep (already computed)
        for (int d = 0; d < d_model; d++) {
            float val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                val += bwd_scan_out[i * d_inner + j] * bwd_out_proj[d * d_inner + j];
            bwd_ctx[d] = val;
        }

        // === GRU + PEER + Expert + Adam (all in registers) ===
        float g = g_ptr[orig_idx];
        float s = s_ptr[orig_idx];

        // Build GRU input
        float gru_inp[MAX_D_MODEL * 2 + 4];
        gru_inp[0] = g;
        gru_inp[1] = s;
        for (int d = 0; d < d_model; d++) {
            gru_inp[2 + d] = fwd_ctx[d];
            gru_inp[2 + d_model + d] = bwd_ctx[d];
        }

        float xh[MAX_D_MODEL * 2 + MAX_GRU_HIDDEN * 2 + 4];
        for (int d = 0; d < gru_input_dim; d++)
            xh[d] = gru_inp[d];
        for (int d = 0; d < gru_hidden; d++)
            xh[gru_input_dim + d] = gru_ptr[orig_idx * gru_hidden + d];

        // GRU gates
        float z_gate[MAX_GRU_HIDDEN], r_gate[MAX_GRU_HIDDEN];
        for (int h = 0; h < gru_hidden; h++) {
            float z_raw = gbz[h], r_raw = gbr[h];
            for (int d = 0; d < gru_total_dim; d++) {
                z_raw += gWz[h * gru_total_dim + d] * xh[d];
                r_raw += gWr[h * gru_total_dim + d] * xh[d];
            }
            z_gate[h] = 1.0f / (1.0f + std::exp(-z_raw));
            r_gate[h] = 1.0f / (1.0f + std::exp(-r_raw));
        }

        float xrh[MAX_D_MODEL * 2 + MAX_GRU_HIDDEN * 2 + 4];
        for (int d = 0; d < gru_input_dim; d++)
            xrh[d] = gru_inp[d];
        for (int d = 0; d < gru_hidden; d++)
            xrh[gru_input_dim + d] = r_gate[d] * gru_ptr[orig_idx * gru_hidden + d];

        float h_tilde[MAX_GRU_HIDDEN];
        for (int h = 0; h < gru_hidden; h++) {
            float val = gbh[h];
            for (int d = 0; d < gru_total_dim; d++)
                val += gWh[h * gru_total_dim + d] * xrh[d];
            h_tilde[h] = std::tanh(val);
        }

        for (int h = 0; h < gru_hidden; h++) {
            float h_old = gru_ptr[orig_idx * gru_hidden + h];
            gru_ptr[orig_idx * gru_hidden + h] =
                (1.0f - z_gate[h]) * h_old + z_gate[h] * h_tilde[h];
        }

        // PEER routing
        float peer_inp[MAX_GRU_HIDDEN + MAX_D_MODEL * 2 + 4];
        for (int d = 0; d < gru_hidden; d++)
            peer_inp[d] = gru_ptr[orig_idx * gru_hidden + d];
        for (int d = 0; d < d_model; d++) {
            peer_inp[gru_hidden + d] = fwd_ctx[d];
            peer_inp[gru_hidden + d_model + d] = bwd_ctx[d];
        }
        peer_inp[gru_hidden + 2 * d_model] = g;
        peer_inp[gru_hidden + 2 * d_model + 1] = s;

        float total_expert_out = 0.0f;
        for (int head = 0; head < num_heads; head++) {
            float query[MAX_D_MODEL];
            const float* W_head = pqWs + head * d_model * peer_input_dim;
            for (int d = 0; d < d_model; d++) {
                float val = 0.0f;
                for (int k = 0; k < peer_input_dim; k++)
                    val += W_head[d * peer_input_dim + k] * peer_inp[k];
                query[d] = val;
            }

            float scores_a[144], scores_b[144];
            const float* pka_head = pkA + head * pk_dim * half_d;
            const float* pkb_head = pkB + head * pk_dim * half_d;

            for (int k = 0; k < pk_dim; k++) {
                float va = 0.0f, vb = 0.0f;
                for (int d = 0; d < half_d; d++) {
                    va += pka_head[k * half_d + d] * query[d];
                    vb += pkb_head[k * half_d + d] * query[half_d + d];
                }
                scores_a[k] = va;
                scores_b[k] = vb;
            }

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

            for (int ta = 0; ta < topk; ta++) {
                for (int tb = 0; tb < topk; tb++) {
                    int expert_id = top_a[ta] * pk_dim + top_b[tb];
                    if (expert_id >= num_experts) continue;

                    float hidden[16];
                    for (int h = 0; h < expert_hidden; h++) {
                        float val = eb1[expert_id * expert_hidden + h]
                                  + eW1[expert_id * expert_hidden + 0] * g;
                        hidden[h] = (val > 0.0f) ? val : 0.0f;
                    }

                    float expert_val = eb2[expert_id];
                    for (int h = 0; h < expert_hidden; h++)
                        expert_val += eW2[expert_id * expert_hidden + h] * hidden[h];

                    total_expert_out += expert_val / (float)(topk * topk * num_heads);
                    ec_ptr[expert_id] += 1;
                }
            }
        }

        // Smart gradient + Mu EMA + Adam
        float smart_g = g + rescale * total_expert_out;
        mu_ptr[orig_idx] = alpha_mu * mu_ptr[orig_idx] + (1.0f - alpha_mu) * g;
        float effective = smart_g + lamb_eff * mu_ptr[orig_idx];

        ea_ptr[orig_idx] = beta1 * ea_ptr[orig_idx] + (1.0f - beta1) * effective;
        eas_ptr[orig_idx] = beta2 * eas_ptr[orig_idx] + (1.0f - beta2) * effective * effective;

        float step_size = lr / bc1;
        float denom = std::sqrt(eas_ptr[orig_idx] / bc2) + eps;
        float p_val = p_ptr[orig_idx];
        p_val = p_val * (1.0f - lr * wd_eff) - step_size * ea_ptr[orig_idx] / denom;
        p_ptr[orig_idx] = p_val;
    }

    // Update Mamba states
    mamba_fwd_state.copy_(torch::from_blob(h_fwd.data(), {d_inner, d_state}, torch::kFloat32));
    mamba_bwd_state.copy_(torch::from_blob(h_bwd.data(), {d_inner, d_state}, torch::kFloat32));

    // Write back parameters
    param.reshape(-1).copy_(p_f);
}


void cpu_sg2_fused_scan_elem_q4(
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
    // Config 4 variant: same fused loop but with INT8 quantized state read/write
    // for exp_avg and exp_avg_sq (stochastic rounding on write, dequant on read).
    // For now, delegates to the FP32 version — quantization can be added later
    // without changing the fused structure.
    cpu_sg2_fused_scan_elem(
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
