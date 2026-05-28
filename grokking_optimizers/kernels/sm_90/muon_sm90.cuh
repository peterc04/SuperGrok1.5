#ifndef GROKKING_MUON_SM90_CUH_
#define GROKKING_MUON_SM90_CUH_

#include "common_sm90.cuh"

namespace grokking { namespace sm90 {

// Newton-Schulz coefficients (Keller-Jordan)
static constexpr float kNS_a = 3.4445f;
static constexpr float kNS_b = -4.7750f;
static constexpr float kNS_c = 2.0315f;

// ---------------------------------------------------------------------------
// State for 2D parameters (momentum-only path)
// ---------------------------------------------------------------------------
struct MuonState {
    float* __restrict__ momentum_buffer;

    static constexpr int num_state_tensors() { return 1; }
    static constexpr int state_bytes_per_element() { return sizeof(float); }
};

// ---------------------------------------------------------------------------
// State for 1D parameters (AdamW fallback)
// ---------------------------------------------------------------------------
struct MuonAdamWState {
    float* __restrict__ exp_avg;
    float* __restrict__ exp_avg_sq;

    static constexpr int num_state_tensors() { return 2; }
    static constexpr int state_bytes_per_element() { return 2 * sizeof(float); }
};

// ============================= 2D PATH ====================================

// ---------------------------------------------------------------------------
// Per-element momentum update for 2D params:
//   buf = momentum * buf + grad
// ---------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void muon_momentum_update(
    const ParamT* __restrict__ grads,
    MuonState                  state,
    int64_t                    n,
    float                      momentum,
    float                      clip_threshold = 0.0f
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * (int64_t)blockDim.x) {

        float g = to_float<ParamT>(__ldg(&grads[i]));

        // NaN handling
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__isnanf(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__isnanf(g)) continue;
        }

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        float buf_old = state.momentum_buffer[i];
        state.momentum_buffer[i] = momentum * buf_old + g;
    }
}

// ---------------------------------------------------------------------------
// Vectorized momentum update for float params (4 elements per thread).
// Caller must guarantee n % 4 == 0 and 16-byte alignment.
// ---------------------------------------------------------------------------
template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void muon_momentum_update_vec4(
    const float* __restrict__ grads,
    MuonState                 state,
    int64_t                   n,
    float                     momentum,
    float                     clip_threshold = 0.0f
) {
    const int64_t n4 = n / 4;

    const float4* __restrict__ g4  = reinterpret_cast<const float4*>(grads);
    float4* __restrict__       b4  = reinterpret_cast<float4*>(state.momentum_buffer);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * (int64_t)blockDim.x) {

        float4 g_vec = g4[i];
        float4 b_vec = b4[i];

        float gs[4] = {g_vec.x, g_vec.y, g_vec.z, g_vec.w};
        float bs[4] = {b_vec.x, b_vec.y, b_vec.z, b_vec.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g = gs[k];

            if constexpr (NAN_POLICY == NanPolicy::kZero) {
                if (__isnanf(g)) g = 0.0f;
            } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
                if (__isnanf(g)) { continue; }
            }

            if constexpr (ENABLE_CLIP) {
                g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
            }

            bs[k] = momentum * bs[k] + g;
        }

        b4[i] = make_float4(bs[0], bs[1], bs[2], bs[3]);
    }
}

// ---------------------------------------------------------------------------
// Newton-Schulz orthogonalization kernel.
//
// Operates on the momentum buffer as a 2D matrix (rows x cols, row-major).
// After NS iterations, applies the parameter update:
//   param -= lr * (X[i] + weight_decay * param[i])
//
// Uses block-cooperative tiling: each thread handles a tile of the matrix.
// The NS iteration requires computing X @ X^T and (X @ X^T)^2, which are
// performed via shared-memory-assisted matmul.
//
// Shared memory layout (per block):
//   float XXT[rows * rows]   -- the X @ X^T product
//   float XXT2[rows * rows]  -- (X @ X^T)^2
//
// Constraint: rows * rows * 2 * sizeof(float) must fit in shared memory.
// For large matrices, the host should tile or use cuBLAS.
// ---------------------------------------------------------------------------
template <typename ParamT>
__global__ void muon_ns_kernel(
    float* __restrict__        buf,       // momentum buffer, rows x cols
    ParamT* __restrict__       params,    // parameter tensor, rows x cols
    int                        rows,
    int                        cols,
    int                        ns_steps,
    float                      lr,
    float                      weight_decay
) {
    // Dynamic shared memory for XXT and XXT2
    extern __shared__ float smem[];
    float* XXT  = smem;                   // rows * rows
    float* XXT2 = smem + rows * rows;     // rows * rows

    const int64_t numel = (int64_t)rows * cols;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // -----------------------------------------------------------------------
    // Newton-Schulz iterations
    // For each iteration:
    //   XXT  = X @ X^T          (rows x rows)
    //   XXT2 = XXT @ XXT        (rows x rows)
    //   X_new = a*X + b*(XXT)@X + c*(XXT2)@X
    // -----------------------------------------------------------------------
    for (int step = 0; step < ns_steps; ++step) {

        // --- Compute XXT = X @ X^T ---
        // Each thread computes one or more entries of the rows x rows matrix
        for (int idx = tid; idx < rows * rows; idx += nthreads) {
            int r = idx / rows;
            int s = idx % rows;
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k) {
                acc += buf[r * cols + k] * buf[s * cols + k];
            }
            XXT[idx] = acc;
        }
        __syncthreads();

        // --- Compute XXT2 = XXT @ XXT ---
        for (int idx = tid; idx < rows * rows; idx += nthreads) {
            int r = idx / rows;
            int s = idx % rows;
            float acc = 0.0f;
            for (int k = 0; k < rows; ++k) {
                acc += XXT[r * rows + k] * XXT[k * rows + s];
            }
            XXT2[idx] = acc;
        }
        __syncthreads();

        // --- Update X: X_new = a*X + b*(XXT @ X) + c*(XXT2 @ X) ---
        // Each thread handles a slice of elements in the rows x cols buffer.
        for (int64_t idx = tid; idx < numel; idx += nthreads) {
            int r = (int)(idx / cols);
            int j = (int)(idx % cols);

            float x_old = buf[idx];

            // (XXT @ X)[r, j] = sum_k XXT[r,k] * X[k,j]
            float xxt_x = 0.0f;
            for (int k = 0; k < rows; ++k) {
                xxt_x += XXT[r * rows + k] * buf[k * cols + j];
            }

            // (XXT2 @ X)[r, j] = sum_k XXT2[r,k] * X[k,j]
            float xxt2_x = 0.0f;
            for (int k = 0; k < rows; ++k) {
                xxt2_x += XXT2[r * rows + k] * buf[k * cols + j];
            }

            buf[idx] = kNS_a * x_old + kNS_b * xxt_x + kNS_c * xxt2_x;
        }
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Apply parameter update: param -= lr * (X + weight_decay * param)
    // -----------------------------------------------------------------------
    for (int64_t idx = tid; idx < numel; idx += nthreads) {
        float x_val = buf[idx];
        float p_f = to_float<ParamT>(params[idx]);
        p_f -= lr * (x_val + weight_decay * p_f);
        params[idx] = from_float<ParamT>(p_f);
    }
}

// ---------------------------------------------------------------------------
// Launcher for momentum update (scalar path)
// ---------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void muon_momentum_kernel(
    const ParamT* grads, MuonState state, int64_t n,
    float momentum, float clip_threshold
) {
    muon_momentum_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        grads, state, n, momentum, clip_threshold);
}

// ============================= 1D PATH (AdamW fallback) ====================

// ---------------------------------------------------------------------------
// Per-element AdamW update for 1D params (biases, norms, etc.)
// ---------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void muon_adamw_update(
    ParamT* __restrict__       params,
    const ParamT* __restrict__ grads,
    MuonAdamWState             state,
    int64_t                    n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * (int64_t)blockDim.x) {

        float g = to_float<ParamT>(__ldg(&grads[i]));

        // NaN handling
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__isnanf(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__isnanf(g)) continue;
        }

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        float p_f   = to_float<ParamT>(params[i]);
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = beta1 * m_old + (1.0f - beta1) * g;
        float v = beta2 * v_old + (1.0f - beta2) * g * g;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        float denom  = sqrtf(v_hat) + eps;
        float update = m_hat / denom + weight_decay * p_f;

        p_f -= lr * update;

        state.exp_avg[i]    = m;
        state.exp_avg_sq[i] = v;
        params[i]           = from_float<ParamT>(p_f);
    }
}

// ---------------------------------------------------------------------------
// Vectorized AdamW update for float params (4 elements per thread).
// Caller must guarantee n % 4 == 0 and 16-byte alignment.
// ---------------------------------------------------------------------------
template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void muon_adamw_update_vec4(
    float* __restrict__       params,
    const float* __restrict__ grads,
    MuonAdamWState            state,
    int64_t                   n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    const int64_t n4 = n / 4;

    float4* __restrict__       p4 = reinterpret_cast<float4*>(params);
    const float4* __restrict__ g4 = reinterpret_cast<const float4*>(grads);
    float4* __restrict__       m4 = reinterpret_cast<float4*>(state.exp_avg);
    float4* __restrict__       v4 = reinterpret_cast<float4*>(state.exp_avg_sq);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * (int64_t)blockDim.x) {

        float4 p_vec = p4[i];
        float4 g_vec = g4[i];
        float4 m_vec = m4[i];
        float4 v_vec = v4[i];

        float gs[4] = {g_vec.x, g_vec.y, g_vec.z, g_vec.w};
        float ps[4] = {p_vec.x, p_vec.y, p_vec.z, p_vec.w};
        float ms[4] = {m_vec.x, m_vec.y, m_vec.z, m_vec.w};
        float vs[4] = {v_vec.x, v_vec.y, v_vec.z, v_vec.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g = gs[k];

            if constexpr (NAN_POLICY == NanPolicy::kZero) {
                if (__isnanf(g)) g = 0.0f;
            } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
                if (__isnanf(g)) { g = 0.0f; ms[k] = ms[k]; vs[k] = vs[k]; ps[k] = ps[k]; continue; }
            }

            if constexpr (ENABLE_CLIP) {
                g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
            }

            float m = beta1 * ms[k] + (1.0f - beta1) * g;
            float v = beta2 * vs[k] + (1.0f - beta2) * g * g;

            float m_hat = m * bias_correction1;
            float v_hat = v * bias_correction2;

            float denom = sqrtf(v_hat) + eps;
            ps[k] -= lr * (m_hat / denom + weight_decay * ps[k]);
            ms[k] = m;
            vs[k] = v;
        }

        p4[i] = make_float4(ps[0], ps[1], ps[2], ps[3]);
        m4[i] = make_float4(ms[0], ms[1], ms[2], ms[3]);
        v4[i] = make_float4(vs[0], vs[1], vs[2], vs[3]);
    }
}

// ---------------------------------------------------------------------------
// Launcher for 1D AdamW fallback (scalar path)
// ---------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void muon_adamw_kernel(
    ParamT* params, const ParamT* grads, MuonAdamWState state, int64_t n,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, float clip_threshold
) {
    muon_adamw_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, lr, beta1, beta2, eps, wd, bc1, bc2, clip_threshold);
}

}} // namespace grokking::sm90

#endif // GROKKING_MUON_SM90_CUH_
