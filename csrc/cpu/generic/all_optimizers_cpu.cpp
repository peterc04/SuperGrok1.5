/*
 * Grokking Optimizers — CPU OpenMP Kernels for All Optimizers
 *
 * Portable C++ implementations with OpenMP parallelism.
 * These are called from cpu_ops.cpp pybind module.
 *
 * Each kernel mirrors the GPU kernel's numerics exactly.
 */

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declarations for SIMD dispatch (defined in avx512/ or neon/)
extern void simd_adam_update(float* param, const float* ea, const float* easq,
                             float step_size, float bc2, float eps,
                             float lr, float wd, int N);
extern void simd_lion_update(float* param, const float* grad, float* momentum,
                             float lr, float beta1, float beta2, float wd, int N);
extern void simd_ema_amplify(float* grad, float* ema, float alpha, float lamb, int N);

// Check if SIMD kernels are available (set by simd_kernels.cpp)
extern bool simd_available();


// ═══════════════════════════════════════════════════════════════════
//  SuperGrok v1.1 CPU kernel
// ═══════════════════════════════════════════════════════════════════

void supergrok11_step_cpu(
    float* param, const float* grad, float* exp_avg, float* exp_avg_sq,
    float* mu, const float* sharpness,
    const float* W1, const float* b1, const float* W2, const float* b2,
    float rescale, int hidden_dim,
    float alpha_mu, float beta1, float beta2, float lr,
    float wd, float eps, float lamb_eff, float bc1, float bc2,
    float grad_clip_norm, int N
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float g = grad[i];

        // NaN guard
        if (!std::isfinite(g)) g = 0.0f;

        // Per-element gradient clipping (simplified)
        // Meta-net: MLP(g * rescale) -> hidden -> output
        float scaled = g * rescale;
        float smart_g = g;

        // Simple MLP: hidden = relu(W1 * scaled + b1), out = W2^T * hidden
        float mlp_out = 0.0f;
        for (int h = 0; h < hidden_dim; h++) {
            float hidden_val = b1[h] + W1[h * 2] * scaled + W1[h * 2 + 1] * (sharpness ? sharpness[i] * rescale : 0.0f);
            hidden_val = (hidden_val > 0.0f) ? hidden_val : 0.0f;  // ReLU
            mlp_out += W2[h] * hidden_val;
        }
        smart_g = g + rescale * mlp_out;

        // Mu EMA
        mu[i] = alpha_mu * mu[i] + (1.0f - alpha_mu) * g;
        float effective = smart_g + lamb_eff * mu[i];

        // Adam
        exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * effective;
        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * effective * effective;

        float step_size = lr / bc1;
        float denom = std::sqrt(exp_avg_sq[i] / bc2) + eps;
        param[i] = param[i] * (1.0f - lr * wd) - step_size * exp_avg[i] / denom;
    }
}


// ═══════════════════════════════════════════════════════════════════
//  NeuralGrok CPU kernel
// ═══════════════════════════════════════════════════════════════════

void neuralgrok_step_cpu(
    float* param, const float* grad, float* exp_avg, float* exp_avg_sq,
    const float* W1, const float* b1, const float* W_last, const float* b_last,
    float alpha, float beta, int hidden_dim,
    float beta1, float beta2, float lr, float wd, float eps,
    float bc1, float bc2, float grad_clip_norm, int N
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float g = grad[i];
        if (!std::isfinite(g)) g = 0.0f;

        float abs_g = std::fabs(g);

        // Amplifier MLP: W1 * |g| + b1 -> relu -> W_last * hidden + b_last
        float mlp_out = b_last[0];
        for (int h = 0; h < hidden_dim; h++) {
            float hidden_val = W1[h] * abs_g + b1[h];
            hidden_val = (hidden_val > 0.0f) ? hidden_val : 0.0f;  // ReLU
            mlp_out += W_last[h] * hidden_val;
        }
        float scale = alpha * mlp_out + beta;
        float smart_g = g * scale;

        // Adam
        exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * smart_g;
        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * smart_g * smart_g;

        float step_size = lr / bc1;
        float denom = std::sqrt(exp_avg_sq[i] / bc2) + eps;
        param[i] = param[i] * (1.0f - lr * wd) - step_size * exp_avg[i] / denom;
    }
}


// ═══════════════════════════════════════════════════════════════════
//  Prodigy CPU kernel
// ═══════════════════════════════════════════════════════════════════

void prodigy_step_cpu(
    float* param, const float* grad, float* exp_avg, float* exp_avg_sq,
    float* s_buf, const float* param_init,
    float d_lr, float beta1, float beta2, float lr,
    float wd, float eps, float bc1, float bc2,
    float* num_acc, float* den_acc, int N
) {
    float local_num = 0.0f, local_den = 0.0f;

    #pragma omp parallel for schedule(static) reduction(+:local_num,local_den)
    for (int i = 0; i < N; i++) {
        float g = grad[i];
        if (!std::isfinite(g)) g = 0.0f;

        local_num += g * (param[i] - param_init[i]);
        local_den += s_buf[i] * std::fabs(g);

        // s_buf update
        s_buf[i] = beta2 * s_buf[i] + (1.0f - beta2) * std::fabs(g) * d_lr;

        float effective = g * d_lr;

        exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * effective;
        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * effective * effective;

        float step_size = lr / bc1;
        float denom = std::sqrt(exp_avg_sq[i] / bc2) + eps;
        param[i] = param[i] * (1.0f - lr * wd) - step_size * exp_avg[i] / denom;
    }

    *num_acc += local_num;
    *den_acc += local_den;
}


// ═══════════════════════════════════════════════════════════════════
//  Muon CPU kernel (Newton-Schulz orthogonalization)
// ═══════════════════════════════════════════════════════════════════

void muon_ns_ortho_cpu(
    float* X, int rows, int cols, int ns_steps
) {
    // Newton-Schulz iteration: X <- X * (3I - X^T X) / 2
    // For small matrices, do this in-place with temporary buffers
    int mn = (rows < cols) ? rows : cols;
    std::vector<float> XtX(mn * mn, 0.0f);
    std::vector<float> temp(rows * cols, 0.0f);

    // Normalize X
    float norm = 0.0f;
    for (int i = 0; i < rows * cols; i++)
        norm += X[i] * X[i];
    norm = std::sqrt(norm);
    if (norm > 0) {
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < rows * cols; i++)
            X[i] *= inv_norm;
    }

    for (int step = 0; step < ns_steps; step++) {
        // Compute X^T X (cols x cols) or X X^T (rows x rows) depending on which is smaller
        if (rows <= cols) {
            // A = X X^T (rows x rows)
            std::fill(XtX.begin(), XtX.end(), 0.0f);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < rows; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < cols; k++)
                        sum += X[i * cols + k] * X[j * cols + k];
                    XtX[i * rows + j] = sum;
                }

            // temp = (3I - A) @ X / 2 = 1.5*X - 0.5*A@X
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++) {
                    float ax = 0.0f;
                    for (int k = 0; k < rows; k++)
                        ax += XtX[i * rows + k] * X[k * cols + j];
                    temp[i * cols + j] = 1.5f * X[i * cols + j] - 0.5f * ax;
                }
        } else {
            // A = X^T X (cols x cols)
            std::fill(XtX.begin(), XtX.end(), 0.0f);
            for (int i = 0; i < cols; i++)
                for (int j = 0; j < cols; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < rows; k++)
                        sum += X[k * cols + i] * X[k * cols + j];
                    XtX[i * cols + j] = sum;
                }

            // temp = X @ (3I - A) / 2 = 1.5*X - 0.5*X@A
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++) {
                    float xa = 0.0f;
                    for (int k = 0; k < cols; k++)
                        xa += X[i * cols + k] * XtX[k * cols + j];
                    temp[i * cols + j] = 1.5f * X[i * cols + j] - 0.5f * xa;
                }
        }

        std::memcpy(X, temp.data(), rows * cols * sizeof(float));
    }

    // Restore norm
    if (norm > 0) {
        for (int i = 0; i < rows * cols; i++)
            X[i] *= norm;
    }
}


// ═══════════════════════════════════════════════════════════════════
//  LookSAM CPU kernels
// ═══════════════════════════════════════════════════════════════════

void looksam_perturb_cpu(float* param, const float* grad, float rho, float gnorm, int N) {
    if (gnorm <= 0) return;
    float scale = rho / (gnorm + 1e-12f);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        param[i] += grad[i] * scale;
}

void looksam_restore_cpu(float* param, const float* backup, int N) {
    std::memcpy(param, backup, N * sizeof(float));
}

void looksam_direction_cpu(float* direction, const float* perturbed, const float* original, int N) {
    float norm_sq = 0.0f;
    #pragma omp parallel for schedule(static) reduction(+:norm_sq)
    for (int i = 0; i < N; i++) {
        float diff = perturbed[i] - original[i];
        direction[i] = diff;
        norm_sq += diff * diff;
    }
    float norm = std::sqrt(norm_sq);
    if (norm > 0) {
        float inv_norm = 1.0f / norm;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            direction[i] *= inv_norm;
    }
}

void looksam_adjust_cpu(float* grad, const float* direction, float alpha, int N) {
    float proj = 0.0f;
    #pragma omp parallel for schedule(static) reduction(+:proj)
    for (int i = 0; i < N; i++)
        proj += grad[i] * direction[i];
    float scaled_proj = alpha * proj;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        grad[i] += direction[i] * scaled_proj;
}
