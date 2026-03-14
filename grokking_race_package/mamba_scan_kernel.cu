/*
 * Mamba Selective Scan — Fused CUDA Kernel
 *
 * Replaces the Python for-loop over timesteps with a single kernel launch.
 * Each thread handles one (batch, d_inner) pair and loops over L timesteps
 * sequentially (scan dependency: h[t] depends on h[t-1]).
 *
 * State (h) is kept in registers — state_dim is typically 16 (64 bytes).
 * Supports FP32, FP16, and BF16.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int MAMBA_BLOCK_SIZE = 256;
constexpr int MAX_STATE_DIM = 64;

template <typename scalar_t>
__global__ void selective_scan_kernel(
    const scalar_t* __restrict__ x,       // [B, L, D]
    const scalar_t* __restrict__ dt,      // [B, L, D] (already softplus'd)
    const scalar_t* __restrict__ B_proj,  // [B, L, S]
    const scalar_t* __restrict__ C_proj,  // [B, L, S]
    const scalar_t* __restrict__ A,       // [D, S] (negated: -exp(A_log))
    scalar_t* __restrict__ y,             // [B, L, D]
    int batch, int L, int D, int S
) {
    // One thread per (batch_idx, inner_idx)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / D;
    int d = idx % D;
    if (b >= batch || d >= D) return;

    // State in registers
    float h[MAX_STATE_DIM];
    for (int s = 0; s < S; s++) h[s] = 0.0f;

    for (int t = 0; t < L; t++) {
        float dt_val = static_cast<float>(dt[b * L * D + t * D + d]);
        float x_val = static_cast<float>(x[b * L * D + t * D + d]);

        float y_val = 0.0f;
        for (int s = 0; s < S; s++) {
            float a = static_cast<float>(A[d * S + s]);
            float dA = expf(dt_val * a);
            float b_val = static_cast<float>(B_proj[b * L * S + t * S + s]);
            h[s] = dA * h[s] + dt_val * b_val * x_val;

            float c_val = static_cast<float>(C_proj[b * L * S + t * S + s]);
            y_val += h[s] * c_val;
        }
        y[b * L * D + t * D + d] = static_cast<scalar_t>(y_val);
    }
}

torch::Tensor selective_scan_cuda(
    torch::Tensor x,    // [B, L, D]
    torch::Tensor dt,   // [B, L, D]
    torch::Tensor B,    // [B, L, S]
    torch::Tensor C,    // [B, L, S]
    torch::Tensor A     // [D, S]
) {
    const int batch = x.size(0);
    const int L = x.size(1);
    const int D = x.size(2);
    const int S = A.size(1);

    TORCH_CHECK(S <= MAX_STATE_DIM, "state_dim exceeds MAX_STATE_DIM (", MAX_STATE_DIM, ")");

    auto y = torch::empty_like(x);
    const int total_threads = batch * D;
    const int grid = (total_threads + MAMBA_BLOCK_SIZE - 1) / MAMBA_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "selective_scan_cuda", ([&] {
        selective_scan_kernel<scalar_t><<<grid, MAMBA_BLOCK_SIZE>>>(
            x.data_ptr<scalar_t>(),
            dt.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            batch, L, D, S
        );
    }));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selective_scan_cuda", &selective_scan_cuda,
          "Mamba selective scan — fused CUDA kernel",
          py::arg("x"), py::arg("dt"), py::arg("B"), py::arg("C"), py::arg("A"));
}
