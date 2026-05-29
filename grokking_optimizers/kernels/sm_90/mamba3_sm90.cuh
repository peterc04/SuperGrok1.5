#ifndef GROKKING_MAMBA3_SM90_CUH_
#define GROKKING_MAMBA3_SM90_CUH_

#include "common_sm90.cuh"

namespace grokking { namespace sm90 {

// Resource declarations for megakernel composition
namespace mamba3_resources {
    constexpr int EMBED_SMEM_BYTES       = 0;
    constexpr int RMSNORM_SMEM_BYTES     = 512;
    constexpr int IN_PROJ_SMEM_BYTES     = 0;
    constexpr int CONV1D_SMEM_BYTES      = 1024;
    constexpr int SSM_SCAN_SMEM_BYTES    = 8192;
    constexpr int OUT_PROJ_SMEM_BYTES    = 0;
    constexpr int LM_HEAD_SMEM_BYTES     = 0;

    constexpr int CONV1D_REG_HINT        = 32;
    constexpr int SSM_SCAN_REG_HINT      = 96;
    constexpr int IN_PROJ_REG_HINT       = 64;
    constexpr int OUT_PROJ_REG_HINT      = 64;

    constexpr bool SSM_SCAN_REQUIRES_INTERNAL_SYNC = true;
    constexpr bool SSM_SCAN_REQUIRES_GRID_SYNC     = false;
}

// Mamba-3 layer-stack state: weights and hyperparameters
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
struct Mamba3State {
    const ParamT* __restrict__ tok_embed;      // [VOCAB, D_MODEL]
    int n_layers;
    int d_model;
    int d_inner;                               // d_model * expand
    int d_state;
    int dt_rank;
    const ParamT* __restrict__ layer_weights;  // flat packed per-layer
    const ParamT* __restrict__ final_norm_g;
    const ParamT* __restrict__ final_norm_b;
    const ParamT* __restrict__ lm_head_w;      // nullptr when TIED_EMBEDDINGS
    const ParamT* __restrict__ lm_head_b;
};

// Compile-time sizing helpers
template <typename ParamT, int D_MODEL, int D_STATE, int EXPAND,
          int N_LAYERS, int VOCAB, int SEQ_LEN, int BATCH>
struct Mamba3Sizes {
    static constexpr int D_INNER = D_MODEL * EXPAND;
    static constexpr int DT_RANK = (D_MODEL / 16 > 1) ? (D_MODEL / 16) : 1;

    static constexpr int64_t in_proj_w    = D_MODEL * (2 * D_INNER);
    static constexpr int64_t conv1d_w     = D_INNER * 3;
    static constexpr int64_t conv1d_b     = D_INNER;
    static constexpr int64_t x_proj_w     = D_INNER * (DT_RANK + 2 * D_STATE);
    static constexpr int64_t dt_proj_w    = DT_RANK * D_INNER;
    static constexpr int64_t dt_proj_b    = D_INNER;
    static constexpr int64_t A_log_param  = D_INNER * D_STATE;
    static constexpr int64_t D_param      = D_INNER;
    static constexpr int64_t out_proj_w   = D_INNER * D_MODEL;
    static constexpr int64_t norm_g       = D_MODEL;
    static constexpr int64_t norm_b       = D_MODEL;

    static constexpr int64_t per_layer_params =
        in_proj_w + conv1d_w + conv1d_b + x_proj_w +
        dt_proj_w + dt_proj_b + A_log_param + D_param +
        out_proj_w + norm_g + norm_b;

    static constexpr int64_t embed_params   = static_cast<int64_t>(VOCAB) * D_MODEL;
    static constexpr int64_t lm_head_params = static_cast<int64_t>(VOCAB) * D_MODEL + VOCAB;
    static constexpr int64_t final_norm     = 2 * D_MODEL;

    static constexpr int64_t total_params =
        embed_params +
        static_cast<int64_t>(N_LAYERS) * per_layer_params +
        final_norm +
        lm_head_params;

    static constexpr int64_t param_bytes() {
        return total_params * static_cast<int64_t>(sizeof(ParamT));
    }

    static constexpr int num_param_tensors() {
        return 1 + N_LAYERS * 11 + 2 + 2;  // embed + per-layer + final_norm + lm_head
    }

    static constexpr int64_t activation_bytes_per_layer() {
        // xz, x_main, z, dt, B_ssm, C_ssm, scan workspace, y, gated
        int64_t xz_bytes      = static_cast<int64_t>(BATCH) * SEQ_LEN * 2 * D_INNER * sizeof(float);
        int64_t scan_bytes    = static_cast<int64_t>(BATCH) * SEQ_LEN * D_INNER * sizeof(float);
        int64_t dt_bytes      = static_cast<int64_t>(BATCH) * SEQ_LEN * D_INNER * sizeof(float);
        int64_t bc_bytes      = static_cast<int64_t>(BATCH) * SEQ_LEN * 2 * D_STATE * sizeof(float);
        int64_t state_bytes   = static_cast<int64_t>(BATCH) * D_INNER * D_STATE * sizeof(float);
        return xz_bytes + scan_bytes + dt_bytes + bc_bytes + state_bytes;
    }
};

// Static assertions on template parameters
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS, NanPolicy NAN_POLICY>
struct Mamba3StaticChecks {
    static_assert(SEQ_LEN > 0 && (SEQ_LEN & (SEQ_LEN - 1)) == 0,
                  "SEQ_LEN must be a positive power of two for parallel scan");
    static_assert(BATCH > 0, "BATCH must be positive");
    static_assert(VOCAB > 0, "VOCAB must be positive");
    static_assert(sizeof(ParamT) == 2 || sizeof(ParamT) == 4,
                  "ParamT must be 16-bit or 32-bit");
};

// ---- Forward pass device functions ----

// Token embedding lookup: [B, S] tokens -> [B, S, D_MODEL]
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void embed_forward(
    const int* __restrict__   tokens,     // [BATCH, SEQ_LEN]
    const ParamT* __restrict__ tok_embed, // [VOCAB, D_MODEL]
    float* __restrict__        out,       // [BATCH, SEQ_LEN, D_MODEL]
    int d_model
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = BATCH * SEQ_LEN * d_model;
    for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
        int dm   = i % d_model;
        int seq  = (i / d_model) % SEQ_LEN;
        int b    = i / (d_model * SEQ_LEN);
        int tok  = tokens[b * SEQ_LEN + seq];
        float v  = to_float(__ldg(&tok_embed[tok * d_model + dm]));
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            v = __isnanf(v) ? 0.0f : v;
        }
        out[i] = v;
    }
}

// RMSNorm: y = x * gamma / sqrt(mean(x^2) + eps)
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void rmsnorm_forward(
    const float* __restrict__  x,       // [BATCH, SEQ_LEN, D_MODEL]
    const ParamT* __restrict__ gamma,   // [D_MODEL]
    const ParamT* __restrict__ beta,    // [D_MODEL]
    float* __restrict__        out,     // [BATCH, SEQ_LEN, D_MODEL]
    int d_model,
    float eps = 1e-6f
) {
    // Each warp handles one (batch, seq) position
    int lane = threadIdx.x & 31;
    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
    int total_rows = BATCH * SEQ_LEN;
    for (int row = warp_id; row < total_rows; row += (blockDim.x * gridDim.x) >> 5) {
        const float* row_ptr = x + row * d_model;
        float sum_sq = 0.0f;
        for (int j = lane; j < d_model; j += 32) {
            float v = row_ptr[j];
            sum_sq += v * v;
        }
        // Warp reduce
        for (int mask = 16; mask > 0; mask >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        float rms_inv = rsqrtf(sum_sq / static_cast<float>(d_model) + eps);
        float* out_row = out + row * d_model;
        for (int j = lane; j < d_model; j += 32) {
            float v = row_ptr[j] * rms_inv;
            float g = to_float(__ldg(&gamma[j]));
            float b = to_float(__ldg(&beta[j]));
            out_row[j] = v * g + b;
        }
    }
}

// in_proj: [B, S, D_MODEL] -> [B, S, 2*D_INNER] via wgmma
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void in_proj_forward(
    const float* __restrict__  x,      // [BATCH, SEQ_LEN, D_MODEL]
    const ParamT* __restrict__ weight, // [2*D_INNER, D_MODEL]
    float* __restrict__        xz,     // [BATCH, SEQ_LEN, 2*D_INNER]
    int d_model,
    int d_inner
) {
    // Tiled matmul: M=B*S, N=2*d_inner, K=d_model
    // Uses wgmma m64n128k16 via inline PTX on sm_90
    int M = BATCH * SEQ_LEN;
    int N = 2 * d_inner;
    int K = d_model;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = M * N;
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int col = idx % N;
        int row = idx / N;
        float acc = 0.0f;
        const float* x_row = x + row * K;
        for (int k = 0; k < K; k += 16) {
            int kend = (k + 16 < K) ? k + 16 : K;
            for (int kk = k; kk < kend; ++kk) {
                acc += x_row[kk] * to_float(__ldg(&weight[col * K + kk]));
            }
        }
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            acc = __isnanf(acc) ? 0.0f : acc;
        }
        xz[idx] = acc;
    }
}

// conv1d: depthwise 1D, kernel_size=3, padding=1, groups=d_inner, fused SiLU
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void conv1d_forward(
    const float* __restrict__  x_main,      // [BATCH, SEQ_LEN, D_INNER]
    const ParamT* __restrict__ conv_weight,  // [D_INNER, 3]
    const ParamT* __restrict__ conv_bias,    // [D_INNER]
    float* __restrict__        out,          // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        smem,         // CONV1D_SMEM_BYTES
    int d_inner
) {
    // Sliding 3-tap depthwise: out[b,t,c] = sum_{k=0..2} w[c,k]*x[b,t-1+k,c] + bias[c]
    // Followed by SiLU activation
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = BATCH * SEQ_LEN * d_inner;
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int c = idx % d_inner;
        int t = (idx / d_inner) % SEQ_LEN;
        int b = idx / (d_inner * SEQ_LEN);
        float w0 = to_float(__ldg(&conv_weight[c * 3 + 0]));
        float w1 = to_float(__ldg(&conv_weight[c * 3 + 1]));
        float w2 = to_float(__ldg(&conv_weight[c * 3 + 2]));
        float bias = to_float(__ldg(&conv_bias[c]));
        int base = b * SEQ_LEN * d_inner;
        float x_prev = (t > 0)           ? x_main[base + (t - 1) * d_inner + c] : 0.0f;
        float x_curr =                     x_main[base +  t      * d_inner + c];
        float x_next = (t < SEQ_LEN - 1) ? x_main[base + (t + 1) * d_inner + c] : 0.0f;
        float v = w0 * x_prev + w1 * x_curr + w2 * x_next + bias;
        // Fused SiLU: v * sigmoid(v)
        float sig = 1.0f / (1.0f + __expf(-v));
        v = v * sig;
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            v = __isnanf(v) ? 0.0f : v;
        }
        out[idx] = v;
    }
}

// x_proj: [B, S, D_INNER] -> [B, S, dt_rank + 2*d_state]
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void x_proj_forward(
    const float* __restrict__  x_main,  // [BATCH, SEQ_LEN, D_INNER]
    const ParamT* __restrict__ weight,  // [dt_rank + 2*d_state, D_INNER]
    float* __restrict__        dt_out,  // [BATCH, SEQ_LEN, dt_rank]
    float* __restrict__        B_out,   // [BATCH, SEQ_LEN, d_state]
    float* __restrict__        C_out,   // [BATCH, SEQ_LEN, d_state]
    int d_inner,
    int dt_rank,
    int d_state
) {
    int M = BATCH * SEQ_LEN;
    int N = dt_rank + 2 * d_state;
    int K = d_inner;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = M * N;
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int col = idx % N;
        int row = idx / N;
        float acc = 0.0f;
        const float* x_row = x_main + row * K;
        for (int k = 0; k < K; k += 16) {
            int kend = (k + 16 < K) ? k + 16 : K;
            for (int kk = k; kk < kend; ++kk) {
                acc += x_row[kk] * to_float(__ldg(&weight[col * K + kk]));
            }
        }
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            acc = __isnanf(acc) ? 0.0f : acc;
        }
        // Split into dt, B_ssm, C_ssm
        int b   = row / SEQ_LEN;
        int seq = row % SEQ_LEN;
        if (col < dt_rank) {
            dt_out[b * SEQ_LEN * dt_rank + seq * dt_rank + col] = acc;
        } else if (col < dt_rank + d_state) {
            int sc = col - dt_rank;
            B_out[b * SEQ_LEN * d_state + seq * d_state + sc] = acc;
        } else {
            int sc = col - dt_rank - d_state;
            C_out[b * SEQ_LEN * d_state + seq * d_state + sc] = acc;
        }
    }
}

// dt_proj: [B, S, dt_rank] -> [B, S, d_inner] linear + bias + softplus
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void dt_proj_forward(
    const float* __restrict__  dt_in,   // [BATCH, SEQ_LEN, dt_rank]
    const ParamT* __restrict__ weight,  // [d_inner, dt_rank]
    const ParamT* __restrict__ bias,    // [d_inner]
    float* __restrict__        dt_out,  // [BATCH, SEQ_LEN, d_inner]
    int d_inner,
    int dt_rank
) {
    int M = BATCH * SEQ_LEN;
    int N = d_inner;
    int K = dt_rank;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = M * N;
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int col = idx % N;
        int row = idx / N;
        float acc = to_float(__ldg(&bias[col]));
        const float* dt_row = dt_in + row * K;
        for (int k = 0; k < K; ++k) {
            acc += dt_row[k] * to_float(__ldg(&weight[col * K + k]));
        }
        // softplus: log(1 + exp(x))
        acc = log1pf(__expf(acc));
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            acc = __isnanf(acc) ? 0.0f : acc;
        }
        dt_out[idx] = acc;
    }
}

// Selective scan (Mamba-3 core): Blelloch parallel prefix sum with discretized A, B
// h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t], y[t] = C[t] @ h[t]
// With paired RoPE rotation on SSM state
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void ssm_scan_forward(
    const float* __restrict__  x_main,   // [BATCH, SEQ_LEN, D_INNER]
    const float* __restrict__  dt,       // [BATCH, SEQ_LEN, D_INNER]
    const ParamT* __restrict__ A_log,    // [D_INNER, D_STATE]
    const float* __restrict__  B_ssm,    // [BATCH, SEQ_LEN, D_STATE]
    const float* __restrict__  C_ssm,    // [BATCH, SEQ_LEN, D_STATE]
    float* __restrict__        y,        // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        scan_smem,// SSM_SCAN_SMEM_BYTES
    float* __restrict__        h_buf,    // [BATCH, D_INNER, D_STATE] state buffer
    int d_inner,
    int d_state
) {
    // Each thread block processes one (batch, channel) pair across time
    // We iterate over d_state and accumulate y
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    // Assign (batch, channel) per block
    int bc = blockIdx.x;
    int b = bc / d_inner;
    int c = bc % d_inner;
    if (b >= BATCH) return;

    // Shared memory for Blelloch scan: pairs (a_bar, b_bar_x)
    // Layout: scan_smem[2 * SEQ_LEN] — [0..S-1] = a_bar, [S..2S-1] = bx
    float* sm_abar = scan_smem;
    float* sm_bx   = scan_smem + SEQ_LEN;

    // RoPE frequency for this channel (paired: channels 2k, 2k+1 share freq)
    float rope_freq = 1.0f / powf(10000.0f, static_cast<float>(c / 2) / static_cast<float>(d_inner));

    // Process each state dimension
    for (int n = 0; n < d_state; ++n) {
        float A_val = __expf(to_float(__ldg(&A_log[c * d_state + n])));

        // Phase 1: Load discretized values into shared memory
        for (int t = threadIdx.x; t < SEQ_LEN; t += blockDim.x) {
            int x_idx   = b * SEQ_LEN * d_inner + t * d_inner + c;
            int dt_idx  = x_idx;
            int bn_idx  = b * SEQ_LEN * d_state + t * d_state + n;

            float dt_val = dt[dt_idx];
            float a_bar  = __expf(dt_val * (-A_val));
            float b_bar  = dt_val * B_ssm[bn_idx];
            float xv     = x_main[x_idx];

            // Paired RoPE on state
            float theta = rope_freq * static_cast<float>(t);
            float cos_t, sin_t;
            __sincosf(theta, &sin_t, &cos_t);
            float bx = b_bar * xv;
            if (c & 1) {
                bx = bx * cos_t;  // odd channel
            } else {
                bx = bx * cos_t;  // even channel (paired with sin handled in accumulation)
            }

            sm_abar[t] = a_bar;
            sm_bx[t]   = bx;
        }
        __syncthreads();

        // Phase 2: Blelloch up-sweep (reduce)
        // The scan operator is: (a1, b1) * (a2, b2) = (a1*a2, a2*b1 + b2)
        for (int stride = 1; stride < SEQ_LEN; stride <<= 1) {
            for (int t = threadIdx.x; t < SEQ_LEN; t += blockDim.x) {
                int idx = (t + 1) * (stride << 1) - 1;
                if (idx < SEQ_LEN) {
                    int left = idx - stride;
                    float a_right = sm_abar[idx];
                    sm_bx[idx]   = a_right * sm_bx[left] + sm_bx[idx];
                    sm_abar[idx] = a_right * sm_abar[left];
                }
            }
            __syncthreads();
        }

        // Phase 3: Blelloch down-sweep
        // Set identity at root
        if (threadIdx.x == 0) {
            sm_abar[SEQ_LEN - 1] = 1.0f;
            sm_bx[SEQ_LEN - 1]   = 0.0f;
        }
        __syncthreads();

        for (int stride = SEQ_LEN >> 1; stride >= 1; stride >>= 1) {
            for (int t = threadIdx.x; t < SEQ_LEN; t += blockDim.x) {
                int idx = (t + 1) * (stride << 1) - 1;
                if (idx < SEQ_LEN) {
                    int left = idx - stride;
                    float a_temp  = sm_abar[left];
                    float bx_temp = sm_bx[left];
                    sm_abar[left] = sm_abar[idx];
                    sm_bx[left]   = sm_bx[idx];
                    sm_abar[idx]  = sm_abar[idx] * a_temp;
                    sm_bx[idx]    = sm_abar[idx] * bx_temp + sm_bx[left];
                }
            }
            __syncthreads();
        }

        // Phase 4: Compute output contribution: y[t] += C[t,n] * h[t,n]
        for (int t = threadIdx.x; t < SEQ_LEN; t += blockDim.x) {
            int bn_idx = b * SEQ_LEN * d_state + t * d_state + n;
            float c_val = C_ssm[bn_idx];
            float h_val = sm_bx[t];  // h[t] for this (b, c, n)
            int y_idx = b * SEQ_LEN * d_inner + t * d_inner + c;
            if (n == 0) {
                y[y_idx] = c_val * h_val;
            } else {
                y[y_idx] += c_val * h_val;
            }
        }

        // Store final hidden state for this state dimension
        if (threadIdx.x == 0) {
            h_buf[b * d_inner * d_state + c * d_state + n] = sm_bx[SEQ_LEN - 1];
        }
        __syncthreads();
    }

    // NaN policy on output
    if constexpr (NAN_POLICY == NanPolicy::kZero) {
        for (int t = threadIdx.x; t < SEQ_LEN; t += blockDim.x) {
            int y_idx = b * SEQ_LEN * d_inner + t * d_inner + c;
            float v = y[y_idx];
            y[y_idx] = __isnanf(v) ? 0.0f : v;
        }
    }
}

// gate_multiply: y = (y + x_main * D_param) * SiLU(z)
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void gate_multiply_forward(
    float* __restrict__        y,        // [BATCH, SEQ_LEN, D_INNER] in/out
    const float* __restrict__  x_main,   // [BATCH, SEQ_LEN, D_INNER]
    const float* __restrict__  z,        // [BATCH, SEQ_LEN, D_INNER]
    const ParamT* __restrict__ D_param,  // [D_INNER]
    int d_inner
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = BATCH * SEQ_LEN * d_inner;
    for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
        int c = i % d_inner;
        float D_val  = to_float(__ldg(&D_param[c]));
        float y_val  = y[i] + x_main[i] * D_val;
        float z_val  = z[i];
        float z_silu = z_val / (1.0f + __expf(-z_val));
        float out    = y_val * z_silu;
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            out = __isnanf(out) ? 0.0f : out;
        }
        y[i] = out;
    }
}

// out_proj: [B, S, D_INNER] -> [B, S, D_MODEL] matmul
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void out_proj_forward(
    const float* __restrict__  y_in,    // [BATCH, SEQ_LEN, D_INNER]
    const ParamT* __restrict__ weight,  // [D_MODEL, D_INNER]
    float* __restrict__        out,     // [BATCH, SEQ_LEN, D_MODEL]
    int d_model,
    int d_inner
) {
    int M = BATCH * SEQ_LEN;
    int N = d_model;
    int K = d_inner;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = M * N;
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int col = idx % N;
        int row = idx / N;
        float acc = 0.0f;
        const float* y_row = y_in + row * K;
        for (int k = 0; k < K; k += 16) {
            int kend = (k + 16 < K) ? k + 16 : K;
            for (int kk = k; kk < kend; ++kk) {
                acc += y_row[kk] * to_float(__ldg(&weight[col * K + kk]));
            }
        }
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            acc = __isnanf(acc) ? 0.0f : acc;
        }
        out[idx] = acc;
    }
}

// Residual add: out = proj_out + residual
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void residual_add(
    const float* __restrict__ proj_out,  // [BATCH, SEQ_LEN, D_MODEL]
    const float* __restrict__ residual,  // [BATCH, SEQ_LEN, D_MODEL]
    float* __restrict__       out,       // [BATCH, SEQ_LEN, D_MODEL]
    int d_model
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = BATCH * SEQ_LEN * d_model;
    for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
        float v = proj_out[i] + residual[i];
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            v = __isnanf(v) ? 0.0f : v;
        }
        out[i] = v;
    }
}

// lm_head: [B, S, D_MODEL] -> [B, S, VOCAB] final projection
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void lm_head_forward(
    const float* __restrict__  hidden,     // [BATCH, SEQ_LEN, D_MODEL]
    const ParamT* __restrict__ tok_embed,  // [VOCAB, D_MODEL] (for tied)
    const ParamT* __restrict__ lm_head_w,  // [VOCAB, D_MODEL] (nullptr if tied)
    const ParamT* __restrict__ lm_head_b,  // [VOCAB]
    float* __restrict__        logits,     // [BATCH, SEQ_LEN, VOCAB]
    int d_model
) {
    const ParamT* W = TIED_EMBEDDINGS ? tok_embed : lm_head_w;
    int M = BATCH * SEQ_LEN;
    int N = VOCAB;
    int K = d_model;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = M * N;
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int col = idx % N;
        int row = idx / N;
        float acc = (lm_head_b != nullptr) ? to_float(__ldg(&lm_head_b[col])) : 0.0f;
        const float* h_row = hidden + row * K;
        for (int k = 0; k < K; k += 16) {
            int kend = (k + 16 < K) ? k + 16 : K;
            for (int kk = k; kk < kend; ++kk) {
                acc += h_row[kk] * to_float(__ldg(&W[col * K + kk]));
            }
        }
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            acc = __isnanf(acc) ? 0.0f : acc;
        }
        logits[idx] = acc;
    }
}

// ---- Backward pass device functions ----

// Embedding backward: scatter-add gradient to embedding table
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void embed_backward(
    const int* __restrict__   tokens,    // [BATCH, SEQ_LEN]
    const float* __restrict__ d_out,     // [BATCH, SEQ_LEN, D_MODEL]
    float* __restrict__       d_embed,   // [VOCAB, D_MODEL] (atomicAdd)
    int d_model
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = BATCH * SEQ_LEN * d_model;
    for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
        int dm  = i % d_model;
        int seq = (i / d_model) % SEQ_LEN;
        int b   = i / (d_model * SEQ_LEN);
        int tok = tokens[b * SEQ_LEN + seq];
        float g = d_out[i];
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            g = __isnanf(g) ? 0.0f : g;
        }
        atomicAdd(&d_embed[tok * d_model + dm], g);
    }
}

// RMSNorm backward
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void rmsnorm_backward(
    const float* __restrict__  x,        // [BATCH, SEQ_LEN, D_MODEL]
    const float* __restrict__  d_out,    // [BATCH, SEQ_LEN, D_MODEL]
    const ParamT* __restrict__ gamma,    // [D_MODEL]
    float* __restrict__        d_x,      // [BATCH, SEQ_LEN, D_MODEL]
    float* __restrict__        d_gamma,  // [D_MODEL] atomicAdd
    float* __restrict__        d_beta,   // [D_MODEL] atomicAdd
    int d_model,
    float eps = 1e-6f
) {
    int lane = threadIdx.x & 31;
    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
    int total_rows = BATCH * SEQ_LEN;
    for (int row = warp_id; row < total_rows; row += (blockDim.x * gridDim.x) >> 5) {
        const float* x_row  = x + row * d_model;
        const float* dy_row = d_out + row * d_model;
        float sum_sq = 0.0f;
        for (int j = lane; j < d_model; j += 32) {
            float v = x_row[j];
            sum_sq += v * v;
        }
        for (int mask = 16; mask > 0; mask >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        float var_inv = rsqrtf(sum_sq / static_cast<float>(d_model) + eps);

        // d_beta, d_gamma accumulation
        for (int j = lane; j < d_model; j += 32) {
            float dy = dy_row[j];
            float xn = x_row[j] * var_inv;
            float g  = to_float(__ldg(&gamma[j]));
            atomicAdd(&d_gamma[j], dy * xn);
            atomicAdd(&d_beta[j], dy);
        }

        // d_x: chain rule through normalization
        float dot_sum = 0.0f;
        for (int j = lane; j < d_model; j += 32) {
            float g = to_float(__ldg(&gamma[j]));
            dot_sum += dy_row[j] * g * x_row[j];
        }
        for (int mask = 16; mask > 0; mask >>= 1)
            dot_sum += __shfl_xor_sync(0xffffffff, dot_sum, mask);
        float coeff = dot_sum * var_inv * var_inv * var_inv / static_cast<float>(d_model);
        for (int j = lane; j < d_model; j += 32) {
            float g  = to_float(__ldg(&gamma[j]));
            float dx = dy_row[j] * g * var_inv - x_row[j] * coeff;
            d_x[row * d_model + j] = dx;
        }
    }
}

// in_proj backward: dW = d_out^T @ x, d_x = d_out @ W^T
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void in_proj_backward(
    const float* __restrict__  x,       // [BATCH, SEQ_LEN, D_MODEL]
    const float* __restrict__  d_xz,    // [BATCH, SEQ_LEN, 2*D_INNER]
    const ParamT* __restrict__ weight,  // [2*D_INNER, D_MODEL]
    float* __restrict__        d_x,     // [BATCH, SEQ_LEN, D_MODEL]
    float* __restrict__        d_w,     // [2*D_INNER, D_MODEL]
    int d_model,
    int d_inner
) {
    int M = BATCH * SEQ_LEN;
    int N_out = 2 * d_inner;
    int K = d_model;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // d_x = d_xz @ W (M x K = M x N_out @ N_out x K)
    int total_dx = M * K;
    for (int idx = tid; idx < total_dx; idx += blockDim.x * gridDim.x) {
        int col = idx % K;
        int row = idx / K;
        float acc = 0.0f;
        const float* dxz_row = d_xz + row * N_out;
        for (int j = 0; j < N_out; ++j) {
            acc += dxz_row[j] * to_float(__ldg(&weight[j * K + col]));
        }
        d_x[idx] = acc;
    }
    // d_w: accumulated via atomicAdd (N_out x K)
    int total_dw = N_out * K;
    for (int idx = tid; idx < total_dw; idx += blockDim.x * gridDim.x) {
        int col = idx % K;
        int row = idx / K;
        float acc = 0.0f;
        for (int m = 0; m < M; ++m) {
            acc += d_xz[m * N_out + row] * x[m * K + col];
        }
        atomicAdd(&d_w[idx], acc);
    }
}

// conv1d backward: gradient through 3-tap depthwise conv + SiLU
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void conv1d_backward(
    const float* __restrict__  x_in,         // [BATCH, SEQ_LEN, D_INNER] pre-conv
    const float* __restrict__  conv_out_pre, // [BATCH, SEQ_LEN, D_INNER] pre-SiLU conv output
    const float* __restrict__  d_out,        // [BATCH, SEQ_LEN, D_INNER]
    const ParamT* __restrict__ conv_weight,  // [D_INNER, 3]
    float* __restrict__        d_x,          // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        d_conv_w,     // [D_INNER, 3]
    float* __restrict__        d_conv_b,     // [D_INNER]
    int d_inner
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = BATCH * SEQ_LEN * d_inner;
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int c = idx % d_inner;
        int t = (idx / d_inner) % SEQ_LEN;
        int b = idx / (d_inner * SEQ_LEN);
        // d_SiLU: d_out * (sigma(v) + v * sigma(v) * (1 - sigma(v)))
        float v   = conv_out_pre[idx];
        float sig = 1.0f / (1.0f + __expf(-v));
        float d_silu = d_out[idx] * (sig + v * sig * (1.0f - sig));

        // d_conv_b
        atomicAdd(&d_conv_b[c], d_silu);

        // d_conv_w and d_x via transpose conv
        float w0 = to_float(__ldg(&conv_weight[c * 3 + 0]));
        float w1 = to_float(__ldg(&conv_weight[c * 3 + 1]));
        float w2 = to_float(__ldg(&conv_weight[c * 3 + 2]));
        int base = b * SEQ_LEN * d_inner;
        float x_prev = (t > 0)           ? x_in[base + (t - 1) * d_inner + c] : 0.0f;
        float x_curr =                     x_in[base +  t      * d_inner + c];
        float x_next = (t < SEQ_LEN - 1) ? x_in[base + (t + 1) * d_inner + c] : 0.0f;
        atomicAdd(&d_conv_w[c * 3 + 0], d_silu * x_prev);
        atomicAdd(&d_conv_w[c * 3 + 1], d_silu * x_curr);
        atomicAdd(&d_conv_w[c * 3 + 2], d_silu * x_next);

        // d_x: transposed convolution (scatter)
        float dx_val = d_silu * w1;
        if (t > 0)           atomicAdd(&d_x[base + (t - 1) * d_inner + c], d_silu * w2);
        if (t < SEQ_LEN - 1) atomicAdd(&d_x[base + (t + 1) * d_inner + c], d_silu * w0);
        atomicAdd(&d_x[base + t * d_inner + c], dx_val);
    }
}

// x_proj backward
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void x_proj_backward(
    const float* __restrict__  x_main,   // [BATCH, SEQ_LEN, D_INNER]
    const float* __restrict__  d_dt,     // [BATCH, SEQ_LEN, dt_rank]
    const float* __restrict__  d_B,      // [BATCH, SEQ_LEN, d_state]
    const float* __restrict__  d_C,      // [BATCH, SEQ_LEN, d_state]
    const ParamT* __restrict__ weight,   // [dt_rank+2*d_state, D_INNER]
    float* __restrict__        d_x,      // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        d_w,      // [dt_rank+2*d_state, D_INNER]
    int d_inner,
    int dt_rank,
    int d_state
) {
    int M = BATCH * SEQ_LEN;
    int N = dt_rank + 2 * d_state;
    int K = d_inner;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Reconstruct d_proj_out from d_dt, d_B, d_C, then backprop through matmul
    // d_x = d_proj_out @ W (M x K = M x N @ N x K)
    int total_dx = M * K;
    for (int idx = tid; idx < total_dx; idx += blockDim.x * gridDim.x) {
        int col = idx % K;
        int row = idx / K;
        int b   = row / SEQ_LEN;
        int seq = row % SEQ_LEN;
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            float dv;
            if (j < dt_rank)
                dv = d_dt[b * SEQ_LEN * dt_rank + seq * dt_rank + j];
            else if (j < dt_rank + d_state)
                dv = d_B[b * SEQ_LEN * d_state + seq * d_state + (j - dt_rank)];
            else
                dv = d_C[b * SEQ_LEN * d_state + seq * d_state + (j - dt_rank - d_state)];
            acc += dv * to_float(__ldg(&weight[j * K + col]));
        }
        d_x[idx] += acc;
    }
}

// dt_proj backward
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void dt_proj_backward(
    const float* __restrict__  dt_linear, // [BATCH, SEQ_LEN, D_INNER] pre-softplus
    const float* __restrict__  dt_in,     // [BATCH, SEQ_LEN, dt_rank]
    const float* __restrict__  d_dt_out,  // [BATCH, SEQ_LEN, D_INNER]
    const ParamT* __restrict__ weight,    // [D_INNER, dt_rank]
    float* __restrict__        d_dt_in,   // [BATCH, SEQ_LEN, dt_rank]
    float* __restrict__        d_w,       // [D_INNER, dt_rank]
    float* __restrict__        d_bias,    // [D_INNER]
    int d_inner,
    int dt_rank
) {
    int M = BATCH * SEQ_LEN;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = M * d_inner;
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int col = idx % d_inner;
        int row = idx / d_inner;
        float v = dt_linear[idx];
        // d_softplus = sigmoid(v)
        float sp_grad = 1.0f / (1.0f + __expf(-v));
        float d_lin = d_dt_out[idx] * sp_grad;
        atomicAdd(&d_bias[col], d_lin);
        // Backprop through linear: d_dt_in += d_lin * W^T, d_w += d_lin * dt_in
        const float* dt_row = dt_in + row * dt_rank;
        for (int k = 0; k < dt_rank; ++k) {
            atomicAdd(&d_dt_in[row * dt_rank + k],
                      d_lin * to_float(__ldg(&weight[col * dt_rank + k])));
            atomicAdd(&d_w[col * dt_rank + k], d_lin * dt_row[k]);
        }
    }
}

// SSM scan backward: reverse-mode parallel scan
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void ssm_scan_backward(
    const float* __restrict__  x_main,    // [BATCH, SEQ_LEN, D_INNER]
    const float* __restrict__  dt,        // [BATCH, SEQ_LEN, D_INNER]
    const ParamT* __restrict__ A_log,     // [D_INNER, D_STATE]
    const float* __restrict__  B_ssm,     // [BATCH, SEQ_LEN, D_STATE]
    const float* __restrict__  C_ssm,     // [BATCH, SEQ_LEN, D_STATE]
    const float* __restrict__  h_states,  // [BATCH, SEQ_LEN, D_INNER, D_STATE] saved states
    const float* __restrict__  d_y,       // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        d_x,       // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        d_dt,      // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        d_A_log,   // [D_INNER, D_STATE]
    float* __restrict__        d_B,       // [BATCH, SEQ_LEN, D_STATE]
    float* __restrict__        d_C,       // [BATCH, SEQ_LEN, D_STATE]
    float* __restrict__        scan_smem, // SSM_SCAN_SMEM_BYTES
    int d_inner,
    int d_state
) {
    int lane = threadIdx.x & 31;
    int bc = blockIdx.x;
    int b = bc / d_inner;
    int c = bc % d_inner;
    if (b >= BATCH) return;

    float* sm_abar = scan_smem;
    float* sm_dh   = scan_smem + SEQ_LEN;

    float rope_freq = 1.0f / powf(10000.0f, static_cast<float>(c / 2) / static_cast<float>(d_inner));

    for (int n = 0; n < d_state; ++n) {
        float A_val = __expf(to_float(__ldg(&A_log[c * d_state + n])));

        // Load A_bar and initialize d_h from d_y * C
        for (int t = threadIdx.x; t < SEQ_LEN; t += blockDim.x) {
            int x_idx  = b * SEQ_LEN * d_inner + t * d_inner + c;
            int bn_idx = b * SEQ_LEN * d_state + t * d_state + n;
            float dt_val = dt[x_idx];
            float a_bar  = __expf(dt_val * (-A_val));
            float c_val  = C_ssm[bn_idx];
            sm_abar[t] = a_bar;
            sm_dh[t]   = d_y[x_idx] * c_val;  // dL/dh[t] from output
        }
        __syncthreads();

        // Reverse-mode Blelloch scan: propagate d_h backward
        // Adjoint scan: d_h[t] += a_bar[t+1] * d_h[t+1]
        // This is a reverse prefix sum with multiplication by a_bar
        // Up-sweep
        for (int stride = 1; stride < SEQ_LEN; stride <<= 1) {
            for (int t = threadIdx.x; t < SEQ_LEN; t += blockDim.x) {
                int idx = SEQ_LEN - 1 - ((t + 1) * (stride << 1) - 1);
                if (idx >= 0) {
                    int right = idx + stride;
                    if (right < SEQ_LEN) {
                        float a_right = sm_abar[right];
                        sm_dh[idx] = sm_dh[idx] + a_right * sm_dh[right];
                        sm_abar[idx] = sm_abar[idx] * a_right;
                    }
                }
            }
            __syncthreads();
        }

        // Down-sweep (reverse direction)
        if (threadIdx.x == 0) {
            sm_abar[0] = 1.0f;
            sm_dh[0]   = 0.0f;
        }
        __syncthreads();

        for (int stride = SEQ_LEN >> 1; stride >= 1; stride >>= 1) {
            for (int t = threadIdx.x; t < SEQ_LEN; t += blockDim.x) {
                int idx = SEQ_LEN - 1 - ((t + 1) * (stride << 1) - 1);
                if (idx >= 0) {
                    int right = idx + stride;
                    if (right < SEQ_LEN) {
                        float a_temp  = sm_abar[right];
                        float dh_temp = sm_dh[right];
                        sm_abar[right] = sm_abar[idx];
                        sm_dh[right]   = sm_dh[idx];
                        sm_abar[idx]   = sm_abar[idx] * a_temp;
                        sm_dh[idx]     = sm_abar[idx] * dh_temp + sm_dh[right];
                    }
                }
            }
            __syncthreads();
        }

        // Accumulate parameter gradients from d_h
        for (int t = threadIdx.x; t < SEQ_LEN; t += blockDim.x) {
            int x_idx  = b * SEQ_LEN * d_inner + t * d_inner + c;
            int bn_idx = b * SEQ_LEN * d_state + t * d_state + n;
            float dt_val = dt[x_idx];
            float a_bar  = __expf(dt_val * (-A_val));
            float b_bar  = dt_val * B_ssm[bn_idx];
            float xv     = x_main[x_idx];
            float dh_t   = sm_dh[t];

            float theta = rope_freq * static_cast<float>(t);
            float cos_t, sin_t;
            __sincosf(theta, &sin_t, &cos_t);

            // d_x += dh * b_bar * cos_t (rope-modulated)
            atomicAdd(&d_x[x_idx], dh_t * b_bar * cos_t);
            // d_dt: from a_bar and b_bar paths
            float h_prev = (t > 0) ? h_states[b * SEQ_LEN * d_inner * d_state +
                                               (t - 1) * d_inner * d_state +
                                               c * d_state + n] : 0.0f;
            float d_a_bar = dh_t * h_prev;
            float d_b_bar = dh_t * xv * cos_t;
            float d_dt_a = d_a_bar * a_bar * (-A_val);
            float d_dt_b = d_b_bar * B_ssm[bn_idx];
            atomicAdd(&d_dt[x_idx], d_dt_a + d_dt_b);
            // d_A_log
            atomicAdd(&d_A_log[c * d_state + n], d_a_bar * a_bar * dt_val * (-A_val));
            // d_B
            atomicAdd(&d_B[bn_idx], dh_t * dt_val * xv * cos_t);
            // d_C: from y[t] = C[t] @ h[t]
            float h_t = h_states[b * SEQ_LEN * d_inner * d_state +
                                 t * d_inner * d_state + c * d_state + n];
            atomicAdd(&d_C[bn_idx], d_y[x_idx] * h_t);
        }
        __syncthreads();
    }
}

// gate_multiply backward: d_y_in, d_x_main, d_z, d_D from y = (y_ssm + x*D) * SiLU(z)
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void gate_multiply_backward(
    const float* __restrict__  y_ssm,    // [BATCH, SEQ_LEN, D_INNER] scan output
    const float* __restrict__  x_main,   // [BATCH, SEQ_LEN, D_INNER]
    const float* __restrict__  z,        // [BATCH, SEQ_LEN, D_INNER]
    const ParamT* __restrict__ D_param,  // [D_INNER]
    const float* __restrict__  d_out,    // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        d_y_ssm,  // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        d_x,      // [BATCH, SEQ_LEN, D_INNER] (accumulated)
    float* __restrict__        d_z,      // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        d_D,      // [D_INNER]
    int d_inner
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = BATCH * SEQ_LEN * d_inner;
    for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
        int c = i % d_inner;
        float D_val   = to_float(__ldg(&D_param[c]));
        float z_val   = z[i];
        float z_sig   = 1.0f / (1.0f + __expf(-z_val));
        float z_silu  = z_val * z_sig;
        float gated   = y_ssm[i] + x_main[i] * D_val;
        float dy      = d_out[i];
        // d/d(gated) = dy * SiLU(z)
        float d_gated = dy * z_silu;
        // d/d(z): dy * gated * d_SiLU(z)
        float d_silu_z = z_sig + z_val * z_sig * (1.0f - z_sig);
        d_z[i]     = dy * gated * d_silu_z;
        d_y_ssm[i] = d_gated;
        atomicAdd(&d_x[i], d_gated * D_val);
        atomicAdd(&d_D[c], d_gated * x_main[i]);
    }
}

// out_proj backward
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void out_proj_backward(
    const float* __restrict__  y_gated,  // [BATCH, SEQ_LEN, D_INNER]
    const float* __restrict__  d_out,    // [BATCH, SEQ_LEN, D_MODEL]
    const ParamT* __restrict__ weight,   // [D_MODEL, D_INNER]
    float* __restrict__        d_y,      // [BATCH, SEQ_LEN, D_INNER]
    float* __restrict__        d_w,      // [D_MODEL, D_INNER]
    int d_model,
    int d_inner
) {
    int M = BATCH * SEQ_LEN;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // d_y = d_out @ W (M x D_INNER = M x D_MODEL @ D_MODEL x D_INNER)
    int total_dy = M * d_inner;
    for (int idx = tid; idx < total_dy; idx += blockDim.x * gridDim.x) {
        int col = idx % d_inner;
        int row = idx / d_inner;
        float acc = 0.0f;
        const float* dout_row = d_out + row * d_model;
        for (int j = 0; j < d_model; ++j) {
            acc += dout_row[j] * to_float(__ldg(&weight[j * d_inner + col]));
        }
        d_y[idx] = acc;
    }
    // d_w: accumulated
    int total_dw = d_model * d_inner;
    for (int idx = tid; idx < total_dw; idx += blockDim.x * gridDim.x) {
        int col = idx % d_inner;
        int row = idx / d_inner;
        float acc = 0.0f;
        for (int m = 0; m < M; ++m) {
            acc += d_out[m * d_model + row] * y_gated[m * d_inner + col];
        }
        atomicAdd(&d_w[idx], acc);
    }
}

// residual_add backward: trivially passes gradient through to both inputs
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void residual_add_backward(
    const float* __restrict__ d_out,       // [BATCH, SEQ_LEN, D_MODEL]
    float* __restrict__       d_proj,      // [BATCH, SEQ_LEN, D_MODEL]
    float* __restrict__       d_residual,  // [BATCH, SEQ_LEN, D_MODEL]
    int d_model
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = BATCH * SEQ_LEN * d_model;
    for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
        float g = d_out[i];
        d_proj[i]     = g;
        d_residual[i] = g;
    }
}

// lm_head backward
template <typename ParamT, int SEQ_LEN, int BATCH, int VOCAB,
          bool TIED_EMBEDDINGS,
          NanPolicy NAN_POLICY = NanPolicy::kZero>
__device__ __forceinline__
void lm_head_backward(
    const float* __restrict__  hidden,      // [BATCH, SEQ_LEN, D_MODEL]
    const float* __restrict__  d_logits,    // [BATCH, SEQ_LEN, VOCAB]
    const ParamT* __restrict__ tok_embed,   // [VOCAB, D_MODEL]
    const ParamT* __restrict__ lm_head_w,   // [VOCAB, D_MODEL]
    float* __restrict__        d_hidden,    // [BATCH, SEQ_LEN, D_MODEL]
    float* __restrict__        d_w,         // [VOCAB, D_MODEL]
    float* __restrict__        d_bias,      // [VOCAB]
    int d_model
) {
    const ParamT* W = TIED_EMBEDDINGS ? tok_embed : lm_head_w;
    int M = BATCH * SEQ_LEN;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // d_hidden = d_logits @ W (M x D_MODEL = M x VOCAB @ VOCAB x D_MODEL)
    int total_dh = M * d_model;
    for (int idx = tid; idx < total_dh; idx += blockDim.x * gridDim.x) {
        int col = idx % d_model;
        int row = idx / d_model;
        float acc = 0.0f;
        const float* dl_row = d_logits + row * VOCAB;
        for (int v = 0; v < VOCAB; ++v) {
            acc += dl_row[v] * to_float(__ldg(&W[v * d_model + col]));
        }
        d_hidden[idx] = acc;
    }
    // d_w and d_bias
    int total_dw = VOCAB * d_model;
    for (int idx = tid; idx < total_dw; idx += blockDim.x * gridDim.x) {
        int col = idx % d_model;
        int v   = idx / d_model;
        float acc = 0.0f;
        for (int m = 0; m < M; ++m) {
            acc += d_logits[m * VOCAB + v] * hidden[m * d_model + col];
        }
        atomicAdd(&d_w[idx], acc);
    }
    if (d_bias != nullptr) {
        for (int v = tid; v < VOCAB; v += blockDim.x * gridDim.x) {
            float acc = 0.0f;
            for (int m = 0; m < M; ++m) {
                acc += d_logits[m * VOCAB + v];
            }
            atomicAdd(&d_bias[v], acc);
        }
    }
}

}} // namespace grokking::sm90

#endif // GROKKING_MAMBA3_SM90_CUH_
