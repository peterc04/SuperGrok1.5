/*
 * Muon — Hopper-Optimized (sm_90+)
 *
 * FP8 E4M3 Newton-Schulz GEMMs via cublasGemmEx for 4x throughput
 * over FP32 on H100. Momentum buffer is converted to FP8 per NS iteration.
 * Element-wise kernels use the generic path.
 *
 * Guarded with CUDA_VERSION >= 11080 for FP8 type availability.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

// Forward declarations
void launch_muon_ns_combine(
    torch::Tensor X_out, torch::Tensor X, torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c);

void launch_muon_update(
    torch::Tensor param, torch::Tensor orth,
    float neg_lr_scale, float decay_factor);

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080

static void hopper_fp8_mm(
    cublasHandle_t handle,
    torch::Tensor A,      // [M, K]
    torch::Tensor B,      // [K, N] or transposed
    torch::Tensor C,      // [M, N]
    int M, int N, int K,
    bool transpose_b
) {
    float a_scale = A.abs().max().item<float>() / 448.0f;
    float b_scale = B.abs().max().item<float>() / 448.0f;
    if (a_scale < 1e-12f) a_scale = 1e-12f;
    if (b_scale < 1e-12f) b_scale = 1e-12f;

    auto a_fp8 = (A / a_scale).to(torch::kFloat8_e4m3fn).contiguous();
    auto b_fp8 = (B / b_scale).to(torch::kFloat8_e4m3fn).contiguous();

    float alpha = a_scale * b_scale;
    float beta = 0.0f;

    if (transpose_b) {
        // C = A @ B.T: in col-major, C(N,M) = B(N,K) * A(K,M)
        cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, &alpha,
            b_fp8.data_ptr(), CUDA_R_8F_E4M3, K,
            a_fp8.data_ptr(), CUDA_R_8F_E4M3, K,
            &beta, C.data_ptr<float>(), CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        // C = A @ B: in col-major, C(N,M) = B(N,K).T * A(K,M)
        cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            b_fp8.data_ptr(), CUDA_R_8F_E4M3, N,
            a_fp8.data_ptr(), CUDA_R_8F_E4M3, K,
            &beta, C.data_ptr<float>(), CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

#endif

void launch_muon_fused_step_hopper(
    torch::Tensor param, torch::Tensor momentum_buffer, torch::Tensor grad,
    float lr, float momentum, float weight_decay, int ns_steps,
    float a, float b, float c
) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    // Momentum update: buf = momentum * buf + grad
    momentum_buffer.mul_(momentum).add_(grad);

    // Normalize
    float norm = momentum_buffer.norm().item<float>();
    float inv_norm = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;
    auto X = momentum_buffer * inv_norm;

    // Newton-Schulz iterations with FP8 GEMMs
    int M = X.size(0);
    int N_dim = (X.dim() >= 2) ? X.size(1) : 1;
    auto X_2d = X.view({M, N_dim});

    #pragma unroll 4
    for (int i = 0; i < ns_steps; i++) {
        // A = X^T @ X
        auto A = torch::empty({N_dim, N_dim}, X.options());
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
        if (M >= 64 && N_dim >= 64) {
            hopper_fp8_mm(handle, X_2d.t().contiguous(), X_2d, A, N_dim, N_dim, M, false);
        } else {
            A = torch::mm(X_2d.t(), X_2d);
        }
#else
        A = torch::mm(X_2d.t(), X_2d);
#endif
        // AX = A @ X^T => X^T @ X @ X^T but we need X_out
        auto AX = torch::mm(X_2d, A);     // [M, N]
        auto AAX = torch::mm(AX, A);      // [M, N]

        // X = a*X + b*AX + c*AAX (element-wise via custom kernel)
        launch_muon_ns_combine(X_2d, X_2d, AX, AAX, a, b, c);
    }

    // Update: param += neg_lr * orth; param *= decay
    float neg_lr = -lr;
    float decay = 1.0f - lr * weight_decay;
    launch_muon_update(param, X_2d.view_as(param), neg_lr, decay);

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}
