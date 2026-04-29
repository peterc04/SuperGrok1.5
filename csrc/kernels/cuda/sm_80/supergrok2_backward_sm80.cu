
/*
 * SuperGrok v2 — Ampere-Optimized Backward Kernel (sm_80+)
 *
 * Real __global__ backward dh-propagation kernel with cp.async
 * double-buffered prefetch:
 *   - mamba3_backward_dh_cpasync_kernel: propagates d_h (hidden state
 *     gradient) backward through time, double-buffering saved states,
 *     x_branch, z, and dt in shared memory via __pipeline_memcpy_async.
 *     While computing on timestep t, timestep t-1's data is prefetched
 *     into the alternate buffer asynchronously.
 *
 * The TF32 cuBLAS-mode wraps that previously lived in
 * launch_mamba3_peer_bilevel_fwd_save_batched_ampere /
 * launch_mamba3_peer_backward_batched_ampere have been folded directly
 * into the canonical launchers in supergrok2_bwd_sm80.cu (which now
 * open CUBLAS_TF32_TENSOR_OP_MATH at entry and restore on exit).
 *
 * The cp.async kernel below is preserved for future activation. It is
 * not currently called from the canonical bwd launcher — the previous
 * Ampere wrapper allocated its own intermediate gradient buffers but
 * never propagated them; the data path is deferred to a hardware-validated
 * tuning pass.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "platform.h"
#include "types.h"

// cp.async intrinsics (sm_80+): asynchronous global->shared memory copy
// These are compiled conditionally and only used on Ampere+
#if GROK_CUDA
#include <cuda_pipeline.h>
#endif

namespace sg { namespace sm80 {

// =====================================================================
//  Backward dh-propagation kernel with cp.async double-buffered prefetch
//
//  Phase 1 of the backward pass: propagate d_h (hidden state gradient)
//  backward through time, using saved states from the forward pass.
//
//  Grid:  num_params blocks  (one block per parameter)
//  Block: d_inner threads    (one thread per inner dimension)
//
//  Double-buffer layout in dynamic shared memory:
//    buf0 states: smem[0 .. state_buf_size-1]
//    buf1 states: smem[state_buf_size .. 2*state_buf_size-1]
//    buf0 xb:     smem[2*state_buf_size .. 2*state_buf_size + d_inner - 1]
//    buf1 xb:     smem[2*state_buf_size + d_inner .. 2*state_buf_size + 2*d_inner - 1]
//    buf0 z:      smem[2*state_buf_size + 2*d_inner .. ...]
//    buf1 z:      ...
//    buf0 dt:     ...
//    buf1 dt:     ...
//
//  cp.async pattern:
//    Iteration start: commit prefetch of timestep t-1 into buf[1-cur]
//    __pipeline_wait_prior(1): ensures buf[cur] (committed last iteration)
//      is ready while buf[1-cur] loads continue in background.
//    Compute on buf[cur], then swap cur = 1-cur.
//
//  Each thread handles one d_inner index and loops over d_state for
//  state interactions. Running d_h gradient is held in registers
//  (MAX_D_STATE floats) and propagated backward through A_bar.
// =====================================================================

#ifdef __CUDACC__

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void mamba3_backward_dh_cpasync_kernel(
    const scalar_t* __restrict__ saved_states,   // [N, d_inner, d_state]
    const scalar_t* __restrict__ saved_x_branch, // [N, d_inner]
    const scalar_t* __restrict__ saved_z,        // [N, d_inner]
    const scalar_t* __restrict__ saved_dt,       // [N, d_inner]
    const scalar_t* __restrict__ d_scan_out,     // [N, d_inner] incoming gradient
    const float* __restrict__ A_log,             // [d_inner, d_state]
    const float* __restrict__ D_param,           // [d_inner]
    const float* __restrict__ C_proj_W,          // [d_state, d_inner]
    scalar_t* __restrict__ d_x_branch,           // [N, d_inner] output
    scalar_t* __restrict__ d_dt,                 // [N, d_inner] output
    float* __restrict__ d_B_accum,               // [N, d_state] output
    float* __restrict__ d_C_accum,               // [N, d_state] output
    const int N, const int d_inner, const int d_state
) {
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    const int state_buf_size = d_inner * d_state;

    // Double buffer pointers for states, x_branch, z, dt
    // Layout: [buf0_states][buf1_states][buf0_xb][buf1_xb][buf0_z][buf1_z][buf0_dt][buf1_dt]
    float* s_states[2] = {smem, smem + state_buf_size};
    float* s_xb[2]     = {smem + 2 * state_buf_size,
                           smem + 2 * state_buf_size + d_inner};
    float* s_z[2]      = {smem + 2 * state_buf_size + 2 * d_inner,
                           smem + 2 * state_buf_size + 3 * d_inner};
    float* s_dt[2]     = {smem + 2 * state_buf_size + 4 * d_inner,
                           smem + 2 * state_buf_size + 5 * d_inner};

    // Load A and D into registers for the full backward sweep
    float A[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[tid * d_state + s]);
    float D_val = D_param[tid];

    int buf = 0;

    // -- Prefetch last timestep (step = N-1) into buffer 0 --
    // Use cp.async to asynchronously copy from global to shared memory.
    // Each thread copies its d_state entries of the state matrix plus
    // its scalar entries for xb, z, dt.
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __pipeline_memcpy_async(
            &s_states[0][tid * d_state + s],
            &saved_states[((N - 1) * d_inner + tid) * d_state + s],
            sizeof(float));
#else
        s_states[0][tid * d_state + s] =
            static_cast<float>(saved_states[((N - 1) * d_inner + tid) * d_state + s]);
#endif
    }
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_memcpy_async(&s_xb[0][tid],
                            &saved_x_branch[(N - 1) * d_inner + tid],
                            sizeof(float));
    __pipeline_memcpy_async(&s_z[0][tid],
                            &saved_z[(N - 1) * d_inner + tid],
                            sizeof(float));
    __pipeline_memcpy_async(&s_dt[0][tid],
                            &saved_dt[(N - 1) * d_inner + tid],
                            sizeof(float));
    __pipeline_commit();
    __pipeline_wait_prior(0);  // first iteration: wait for everything
#else
    s_xb[0][tid] = static_cast<float>(saved_x_branch[(N - 1) * d_inner + tid]);
    s_z[0][tid]  = static_cast<float>(saved_z[(N - 1) * d_inner + tid]);
    s_dt[0][tid] = static_cast<float>(saved_dt[(N - 1) * d_inner + tid]);
#endif
    __syncthreads();

    // Running gradient for hidden state -- initialized to zero
    float d_h[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        d_h[s] = 0.0f;

    // -- Backward iteration: t = N-1 down to 0 --
    #pragma unroll 4
    for (int step = N - 1; step >= 0; step--) {

        // Prefetch step-1 into alternate buffer (skip on final iteration)
        if (step > 0) {
            int next = step - 1;
            #pragma unroll 4
            for (int s = 0; s < d_state; s++) {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                __pipeline_memcpy_async(
                    &s_states[1 - buf][tid * d_state + s],
                    &saved_states[(next * d_inner + tid) * d_state + s],
                    sizeof(float));
#else
                s_states[1 - buf][tid * d_state + s] =
                    static_cast<float>(saved_states[(next * d_inner + tid) * d_state + s]);
#endif
            }
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            __pipeline_memcpy_async(&s_xb[1 - buf][tid],
                                    &saved_x_branch[next * d_inner + tid],
                                    sizeof(float));
            __pipeline_memcpy_async(&s_z[1 - buf][tid],
                                    &saved_z[next * d_inner + tid],
                                    sizeof(float));
            __pipeline_memcpy_async(&s_dt[1 - buf][tid],
                                    &saved_dt[next * d_inner + tid],
                                    sizeof(float));
            __pipeline_commit();
#else
            s_xb[1 - buf][tid] = static_cast<float>(saved_x_branch[next * d_inner + tid]);
            s_z[1 - buf][tid]  = static_cast<float>(saved_z[next * d_inner + tid]);
            s_dt[1 - buf][tid] = static_cast<float>(saved_dt[next * d_inner + tid]);
#endif
        }

        // -- Compute on current buffer --
        float xb     = s_xb[buf][tid];
        float z_val  = s_z[buf][tid];
        float dt_val = s_dt[buf][tid];
        float d_out  = static_cast<float>(d_scan_out[step * d_inner + tid]);

        // d_output through SiLU gate: silu(z) = z * sigmoid(z)
        float sig_z  = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sig_z;
        float d_y    = d_out;

        // Contribution from D skip connection: D * x_branch
        float d_xb_D = d_y * D_val;

        // Gated gradient: d_y passed through SiLU gate
        float d_y_gated = d_y * silu_z;

        float d_xb_total = d_xb_D;
        float d_dt_total = 0.0f;

        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            float h_val = s_states[buf][tid * d_state + s];

            // Bilinear (Tustin) discretization: A_bar = (1 + dt*A/2) / (1 - dt*A/2)
            float half_dtA = dt_val * A[s] * 0.5f;
            float denom = 1.0f - half_dtA + 1e-8f;
            float A_bar = (1.0f + half_dtA) / denom;

            // B_bar (zero-order hold discretization)
            float B_bar = dt_val;

            // Accumulate d_h from output gradient through C projection
            d_h[s] += d_y_gated * C_proj_W[s * d_inner + tid];

            // d_C accumulation: d_C[step, s] += d_y_gated * h[step, s]
            // Multiple threads (different d_inner) contribute to the same
            // d_state slot, so we use atomicAdd. The full reduction happens
            // in Phase 2 weight gradient GEMMs.
            atomicAdd(&d_C_accum[step * d_state + s], d_y_gated * h_val);

            // d_x_branch contribution from B_bar * x term in state update
            d_xb_total += d_h[s] * B_bar;

            // d_dt contribution via quotient rule on A_bar:
            //   d(A_bar)/d(dt) = A[s] / (1 - dt*A[s]/2 + eps)^2
            //   d_dt += d_h[s] * (dA_bar_ddt * h_prev + xb)
            float dA_bar_ddt = A[s] / (denom * denom);
            d_dt_total += d_h[s] * (dA_bar_ddt * h_val + xb);

            // d_B accumulation
            atomicAdd(&d_B_accum[step * d_state + s], d_h[s] * xb * dt_val);

            // Propagate d_h backward through A_bar (chain rule)
            d_h[s] = d_h[s] * A_bar;
        }

        d_x_branch[step * d_inner + tid] = static_cast<scalar_t>(d_xb_total);
        d_dt[step * d_inner + tid]        = static_cast<scalar_t>(d_dt_total);

        // Wait for prefetch of next (earlier) timestep to complete
        if (step > 0) {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            __pipeline_wait_prior(0);
#endif
            __syncthreads();
        }

        // Flip double buffer AFTER wait
        buf = 1 - buf;
    }
}

// Explicit instantiations
template __global__ void mamba3_backward_dh_cpasync_kernel<float>(
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*,
    float*, float*, float*, float*,
    int, int, int);

template __global__ void mamba3_backward_dh_cpasync_kernel<at::Half>(
    const at::Half*, const at::Half*, const at::Half*, const at::Half*,
    const at::Half*, const float*, const float*, const float*,
    at::Half*, at::Half*, float*, float*,
    int, int, int);

template __global__ void mamba3_backward_dh_cpasync_kernel<at::BFloat16>(
    const at::BFloat16*, const at::BFloat16*, const at::BFloat16*, const at::BFloat16*,
    const at::BFloat16*, const float*, const float*, const float*,
    at::BFloat16*, at::BFloat16*, float*, float*,
    int, int, int);

#endif  // __CUDACC__



} } // namespace sg::sm80
