// =====================================================================
//  csrc/kernels/cuda/sm_90/supergrok2_fwd.cuh
//
//  sm_90 (Hopper) SuperGrok v2 — forward kernels + top-level dispatcher.
//
//  This header is NET-NEW: the previous baseline at
//  csrc/kernels/cuda/sm_90/supergrok2_fwd_sm90.cu (4798 lines) was
//  deleted in commit 5505b50 and is recoverable for reference via:
//    git show 5505b50^:csrc/kernels/cuda/sm_90/supergrok2_fwd_sm90.cu
//
//  Algorithm (per param batch — Mamba-3 + 4-Head PEER + GRU step):
//
//    Mamba-3 selective scan (bidirectional):
//      x_t  = W_x · ĝ;  z_t = W_z · ĝ
//      Δ_t  = softplus(W_Δ · x_t + b_Δ)         (softplus_ptx; CUTLASS
//                                                  EpilogueOp fuses
//                                                  softplus(x+bias) where
//                                                  total_N >= 1024)
//      Ā_t  = exp(Δ_t ⊙ A);  B̄_t = Δ_t ⊙ W_B · x_t
//      H_t  = Ā_t ⊙ R_θ(H_{t-1}) + B̄_t ⊗ x_t   (Affine2x2 + Blelloch)
//      y_t  = (W_C^T · H_t) ⊙ σ(z_t)
//
//    GRU forward (atop y_t):
//      r_t       = σ(W^(r) · [h^gru_{t-1}; y_t])
//      z^g_t     = σ(W^(u) · [h^gru_{t-1}; y_t])
//      ĥ_t       = tanh(W^(c) · [r_t ⊙ h^gru_{t-1}; y_t])
//      h^gru_t   = (1 - z^g_t) ⊙ h^gru_{t-1} + z^g_t ⊙ ĥ_t
//                                              (gru_gates_ptx)
//
//    PEER routing (4-head, √E sub-keys):
//      S_j  = TopK_k(Q₁·q^a_j + Q₂·q^b_j)
//      w_t  = softmax(S)
//      e_t  = Σ_{j,n} w_{t,j,n} · E_{π(j,n)}(h^gru_t)
//                                              (warp-cooperative
//                                               radix-select TopK)
//
//    Forward apply tail (mirrors v1.5):
//      α_t = clip(α₀ + τ·e_t)
//      g̃   = α_t ⊙ ĝ + (1-α_t) ⊙ e_t
//      Adam(g̃) → u_t  (with bc1, bc2)
//      TrustRatio(u_t) → θ_t
//
//  Decision tree (top-level dispatcher):
//      N < PSCAN_THRESHOLD  (256) : sequential scan kernel
//      256 ≤ N < GEMM_PRECOMPUTE_THRESHOLD (1024)
//                                : parallel precompute + parallel scan
//      N ≥ GEMM_PRECOMPUTE_THRESHOLD (1024)
//                                : bilevel_precompute_gemm (cuBLAS/
//                                  CUTLASS — Hopper FP8 fast path) +
//                                  parallel scan
//
//  Hopper FP8 fast path (preserved from deleted baseline): the five
//  projection matmuls (in_proj_x, in_proj_z, dt_proj, B_proj, C_proj)
//  are dispatched to hopper_fp8_gemm + hopper_precompute_fp8 when:
//
//      (CUDA_VERSION >= 11080)
//   && (total_N >= GEMM_PRECOMPUTE_THRESHOLD)
//   && (d_inner >= 64) && (d_state >= 64) && (d_model >= 64)
//
//  Warp-specialized scan activation (REFRESH §25.1):  the parallel
//  scan path checks `is_uniform_d_state(batch)` and routes to
//  launch_scan_warp_specialized (or _d16 for d_state=16) when true.
//  Expected ~1.5x over the unfused parallel scan kernel.
//
//  Optimizations honored:
//    - CUTLASS sm_90a WGMMA + TMA + producer/consumer for the GEMMs
//    - Affine2x2 + Blelloch parallel prefix scan (affine_combine_ptx)
//      with SMEM-resident state and warp-shuffle skip below WARP_SIZE
//    - DSMEM cluster reduction for global norms (REFRESH §25.7)
//    - CUDA Graph capture: launchers are stream-ordered + use
//      pre-allocated workspaces; stream pool is initialized once and
//      reused across captured graphs
//    - §25.3 fused softplus epilogue: applied via cutlass_dt_proj_fused
//      whenever the GEMM tile size threshold is met; falls back to the
//      separate softplus_bias_kernel pass otherwise (see
//      `apply_dt_proj_with_softplus`)
//
//  Sweep budget for the apply tail: 2 grid sweeps over per-element
//  data — sweep A computes meta-net forward + ‖ĝ-g‖² reduction +
//  e_t scratch; sweep B is register-resident smart_grad + Adam
//  moments + trust-ratio reduction + apply (cooperative grid sync
//  between phases).
//
//  Dtype matrix instantiated in the .cu TU:
//    ParamT in {float, __nv_bfloat16, __half}                     (3)
//    StateT in {float, __nv_bfloat16}                             (2)
//    GradT  in {float, __nv_bfloat16, __half,
//               __nv_fp8_e4m3, __nv_fp8_e5m2}                     (5)
//  Coherent combos only — FP8 grad with FP32 param is rejected via
//  static_assert (mirrors adamw.cu / supergrok11.cu).
//
//  NAMESPACE NOTE: csrc/bindings/supergrok2.cpp DECLARE_SG2 expects
//  per-tensor entry points
//    sg::sm90::launch_mamba3_peer_step
//    sg::sm90::launch_mamba3_peer_batched_step
//    sg::sm90::launch_mamba3_peer_bilevel_fwd_save
//    sg::sm90::launch_mamba3_peer_bilevel_fwd_save_batched
//    sg::sm90::supergrok2_prepare_and_batched_step
//  This header places the canonical implementations in
//  `sg::sm90::supergrok2`, and the .cu TU emits thin shim symbols at
//  the binding-expected `sg::sm90::launch_*` namespace. Mirrors the
//  supergrok11 / supergrok15 design.
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/ptx_intrinsics.cuh"
#include "csrc/kernels/cuda/sm_90/supergrok2_warp_specialized.cuh"

#ifdef WITH_CUTLASS
#include "csrc/kernels/cuda/_cutlass_gemm.cuh"
#endif

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

namespace sg { namespace sm90 { namespace supergrok2 {

// =====================================================================
//  Coherent-combo guard (mirrors adamw.cuh::is_coherent_combo).
//  FP8 grads with FP32 params silently lose dynamic range; reject.
// =====================================================================
template <typename ParamT, typename StateT, typename GradT>
struct is_coherent_combo {
#if defined(__CUDACC__)
    static constexpr bool value = !(
        std::is_same<ParamT, float>::value &&
        (std::is_same<GradT, __nv_fp8_e4m3>::value ||
         std::is_same<GradT, __nv_fp8_e5m2>::value));
#else
    static constexpr bool value = true;
#endif
};

// =====================================================================
//  Forward Kernel 1: Input projection + sort key
//
//  Per-element compute:
//    x_out[idx, d] = proj_W[d, 0] * grad[idx]
//                  + proj_W[d, 1] * sharpness[idx]
//                  + proj_b[d]
//    sort_keys[idx] = |grad[idx]|
//
//  Reused as the ingest stage for both per-tensor and batched flows.
// =====================================================================
template <typename GradT>
__launch_bounds__(SG2M_BLOCK, 4)
__global__ void input_proj_sort_kernel(
    const GradT* __restrict__ grad,
    const GradT* __restrict__ sharpness,
    float* __restrict__ x_out,
    float* __restrict__ sort_keys,
    int* __restrict__ sort_indices,
    const float* __restrict__ proj_W,
    const float* __restrict__ proj_b,
    int N, int d_model
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = static_cast<float>(grad[idx]);
    float s = static_cast<float>(sharpness[idx]);
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(s)) s = 0.0f;

    #pragma unroll 4
    for (int d = 0; d < d_model; d++) {
        x_out[idx * d_model + d] =
            proj_W[d * 2] * g + proj_W[d * 2 + 1] * s + proj_b[d];
    }
    sort_keys[idx] = fabsf(g);
    sort_indices[idx] = idx;
}

// =====================================================================
//  Forward Kernel 2 (sequential branch — N < PSCAN_THRESHOLD):
//      mamba3_scan_kernel
//
//  One thread per d_inner index. Sequential scan over N timesteps with
//  trapezoidal discretization + paired RoPE. State held in registers
//  (d_state floats per thread). Projection weights staged through smem.
//  Math identical to deleted baseline; weights cached cooperatively in
//  the d_inner threads of the block.
// =====================================================================
__launch_bounds__(MAX_D_INNER, 8)
__global__ void mamba3_scan_kernel(
    const float* __restrict__ x_sorted,
    const float* __restrict__ in_proj_W,
    const float* __restrict__ dt_proj_W,
    const float* __restrict__ dt_proj_b,
    const float* __restrict__ B_proj_W,
    const float* __restrict__ C_proj_W,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ rope_freq,
    float* __restrict__ scan_output,
    float* __restrict__ final_state,
    const float* __restrict__ initial_state,
    int N, int d_model, int d_inner, int d_state, int reverse
) {
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    extern __shared__ float smem[];
    float* s_x_branch = smem;
    float* s_in_proj_W = s_x_branch + d_inner;
    float* s_dt_proj_W = s_in_proj_W + 2 * d_inner * d_model;
    float* s_dt_proj_b = s_dt_proj_W + d_inner * d_inner;
    float* s_B_proj_W = s_dt_proj_b + d_inner;
    float* s_C_proj_W = s_B_proj_W + d_state * d_inner;

    #pragma unroll 4
    for (int i = tid; i < 2 * d_inner * d_model; i += d_inner)
        s_in_proj_W[i] = in_proj_W[i];
    #pragma unroll 4
    for (int i = tid; i < d_inner * d_inner; i += d_inner)
        s_dt_proj_W[i] = dt_proj_W[i];
    if (tid < d_inner) s_dt_proj_b[tid] = dt_proj_b[tid];
    #pragma unroll 4
    for (int i = tid; i < d_state * d_inner; i += d_inner)
        s_B_proj_W[i] = B_proj_W[i];
    #pragma unroll 4
    for (int i = tid; i < d_state * d_inner; i += d_inner)
        s_C_proj_W[i] = C_proj_W[i];
    __syncthreads();

    float h[MAX_D_STATE], h_snap[MAX_D_STATE];
    if (initial_state) {
        #pragma unroll 4
        for (int s = 0; s < d_state; s++) h[s] = initial_state[tid * d_state + s];
    } else {
        #pragma unroll 4
        for (int s = 0; s < d_state; s++) h[s] = 0.0f;
    }
    const int half_d_state = d_state / 2;
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -fast_exp_ptx(A_log[tid * d_state + s]);
    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[tid * half_d_state + p];
    const float D_val = D_param[tid];

    #pragma unroll 4
    for (int step = 0; step < N; step++) {
        const int i = reverse ? (N - 1 - step) : step;

        float x_val = 0.0f, z_val = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) {
            const float inp = x_sorted[i * d_model + d];
            x_val += s_in_proj_W[tid * d_model + d] * inp;
            z_val += s_in_proj_W[(tid + d_inner) * d_model + d] * inp;
        }
        s_x_branch[tid] = x_val;
        __syncthreads();

        float dt_raw = s_dt_proj_b[tid];
        #pragma unroll 4
        for (int j = 0; j < d_inner; j++)
            dt_raw += s_dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        const float dt_val = softplus_ptx(dt_raw);

        #pragma unroll 4
        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];

        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            const float A_bar = (1.0f + dt_val * A[s] / 2.0f)
                              / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
            float B_val = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < d_inner; j++)
                B_val += s_B_proj_W[s * d_inner + j] * s_x_branch[j];
            const float B_bar = dt_val * B_val;
            const int p = s / 2;
            float cos_p, sin_p;
            FAST_SINCOSF(dt_val * freq[p], &sin_p, &cos_p);
            const float h_rot = (s & 1) == 0
                ? h_snap[s] * cos_p - h_snap[s + 1] * sin_p
                : h_snap[s] * cos_p + h_snap[s - 1] * sin_p;
            h[s] = A_bar * h_rot + B_bar * x_val;
        }

        float y_val = 0.0f;
        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            float C_val = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < d_inner; j++)
                C_val += s_C_proj_W[s * d_inner + j] * s_x_branch[j];
            y_val += h[s] * C_val;
        }
        const float silu_z = z_val / (1.0f + __expf(-z_val));
        scan_output[i * d_inner + tid] = y_val * silu_z + D_val * x_val;
        __syncthreads();
    }

    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        final_state[tid * d_state + s] = h[s];
}

// =====================================================================
//  Forward Kernel 3 (parallel branch — 256 ≤ N < 1024):
//      mamba3_parallel_precompute_kernel
//
//  Pre-computes all cross-thread-dependent quantities for all N
//  timesteps. Each thread handles one timestep, computing x/z (input
//  proj), dt (full proj + softplus_ptx), B and C (full proj). Output
//  buffers feed the parallel scan kernel below. This is the FP32
//  fallback path; the FP8 fast path lives in `hopper_precompute_fp8`
//  under WITH_CUTLASS / CUDA_VERSION >= 11080.
// =====================================================================
__launch_bounds__(SG2M_BLOCK, 4)
__global__ void mamba3_parallel_precompute_kernel(
    const float* __restrict__ x_sorted,
    const float* __restrict__ in_proj_W,
    const float* __restrict__ dt_proj_W,
    const float* __restrict__ dt_proj_b,
    const float* __restrict__ B_proj_W,
    const float* __restrict__ C_proj_W,
    float* __restrict__ pre_x_val,
    float* __restrict__ pre_z_val,
    float* __restrict__ pre_dt_val,
    float* __restrict__ pre_B_val,
    float* __restrict__ pre_C_val,
    int N, int d_model, int d_inner, int d_state
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    float inp[MAX_D_MODEL];
    #pragma unroll 4
    for (int d = 0; d < d_model; d++) inp[d] = x_sorted[t * d_model + d];

    float x_branch[MAX_D_INNER];
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++) {
        float xv = 0.0f, zv = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) {
            xv += in_proj_W[j * d_model + d] * inp[d];
            zv += in_proj_W[(j + d_inner) * d_model + d] * inp[d];
        }
        x_branch[j] = xv;
        pre_x_val[t * d_inner + j] = xv;
        pre_z_val[t * d_inner + j] = zv;
    }

    #pragma unroll 4
    for (int j = 0; j < d_inner; j++) {
        float dt_raw = dt_proj_b[j];
        #pragma unroll 4
        for (int k = 0; k < d_inner; k++)
            dt_raw += dt_proj_W[j * d_inner + k] * x_branch[k];
        pre_dt_val[t * d_inner + j] = softplus_ptx(dt_raw);
    }

    #pragma unroll 4
    for (int s = 0; s < d_state; s++) {
        float Bv = 0.0f, Cv = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < d_inner; j++) {
            Bv += B_proj_W[s * d_inner + j] * x_branch[j];
            Cv += C_proj_W[s * d_inner + j] * x_branch[j];
        }
        pre_B_val[t * d_state + s] = Bv;
        pre_C_val[t * d_state + s] = Cv;
    }
}

// =====================================================================
//  §25.3 fused-softplus epilogue helper:
//
//  When CUTLASS is available + total_N >= GEMM_PRECOMPUTE_THRESHOLD,
//  the dt_proj produces (W·x + b) through cutlass_dt_proj_fused with
//  EpilogueOp = softplus(x + bias). When CUTLASS is not configured or
//  the size threshold is not met, we run the standalone GEMM and a
//  separate softplus_bias_kernel pass.
// =====================================================================
__launch_bounds__(SG2M_BLOCK, 4)
__global__ void softplus_bias_kernel(
    float* __restrict__ buf,           // [N, d_inner], in/out
    const float* __restrict__ bias,    // [d_inner]
    int N, int d_inner
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++) {
        float v = buf[t * d_inner + j] + bias[j];
        buf[t * d_inner + j] = softplus_ptx(v);
    }
}

// =====================================================================
//  Forward Kernel 4 (parallel branch): mamba3_parallel_scan_kernel
//
//  Block per d_inner index; each thread handles a contiguous chunk of
//  timesteps. Implements Affine2x2 + Blelloch parallel prefix scan over
//  the (Ā, B̄·x) recurrence, with paired RoPE composed into the matrix
//  factor. SMEM holds Affine2x2 entries per thread (6 floats × num
//  threads). The composition `affine_combine_ptx` is the 12-FMA hot
//  loop; the WARP_SIZE-aware sync skip avoids __syncthreads() while
//  stride is below WARP_SIZE.
//
//  Output gating + D skip:  scan_output[t,j] = y_t · σ(z_t) + D · x_t.
// =====================================================================
__launch_bounds__(PSCAN_BLOCK, 2)
__global__ void mamba3_parallel_scan_kernel(
    const float* __restrict__ pre_x_val,
    const float* __restrict__ pre_z_val,
    const float* __restrict__ pre_dt_val,
    const float* __restrict__ pre_B_val,
    const float* __restrict__ pre_C_val,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ rope_freq,
    float* __restrict__ scan_output,
    float* __restrict__ final_state,
    const float* __restrict__ initial_state,
    int N, int d_inner, int d_state, int reverse
) {
    const int j = blockIdx.x;
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float smem[];

    const int chunk_size = (N + num_threads - 1) / num_threads;
    const int my_start = ltid * chunk_size;
    const int my_end = min(my_start + chunk_size, N);
    const int my_count = max(my_end - my_start, 0);
    const int half_d_state = d_state / 2;

    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -fast_exp_ptx(A_log[j * d_state + s]);
    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[j * half_d_state + p];
    const float D_val = D_param[j];

    float h_init_all[MAX_D_STATE];
    if (initial_state) {
        #pragma unroll 4
        for (int s = 0; s < d_state; s++) h_init_all[s] = initial_state[j * d_state + s];
    } else {
        #pragma unroll 4
        for (int s = 0; s < d_state; s++) h_init_all[s] = 0.0f;
    }

    #define SG2_BUILD_AFFINE(t_idx, A_e, A_o, f_val, s_e, s_o, elem_out) do { \
        float dt = pre_dt_val[(t_idx) * d_inner + j]; \
        float xv = pre_x_val[(t_idx) * d_inner + j]; \
        float Be = pre_B_val[(t_idx) * d_state + (s_e)]; \
        float Bo = pre_B_val[(t_idx) * d_state + (s_o)]; \
        float Ae_b = (1.0f + dt * (A_e) / 2.0f) / (1.0f - dt * (A_e) / 2.0f + 1e-8f); \
        float Ao_b = (1.0f + dt * (A_o) / 2.0f) / (1.0f - dt * (A_o) / 2.0f + 1e-8f); \
        float cv, sv; FAST_SINCOSF(dt * (f_val), &sv, &cv); \
        (elem_out).m00 = Ae_b * cv; (elem_out).m01 = -Ae_b * sv; \
        (elem_out).m10 = Ao_b * sv; (elem_out).m11 = Ao_b * cv; \
        (elem_out).b0 = dt * Be * xv; (elem_out).b1 = dt * Bo * xv; \
    } while (0)

    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++) {
        const int s_e = 2 * p;
        const int s_o = 2 * p + 1;
        const float A_e = A[s_e], A_o = A[s_o];
        const float f_val = freq[p];
        const float h_init_e = h_init_all[s_e];
        const float h_init_o = h_init_all[s_o];

        Affine2x2 summary = affine_identity();
        #pragma unroll 4
        for (int step = 0; step < my_count; step++) {
            const int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
            Affine2x2 elem;
            SG2_BUILD_AFFINE(t, A_e, A_o, f_val, s_e, s_o, elem);
            summary = affine_combine_ptx(summary, elem);
        }

        const int base = ltid * 6;
        smem[base + 0] = summary.m00; smem[base + 1] = summary.m01;
        smem[base + 2] = summary.m10; smem[base + 3] = summary.m11;
        smem[base + 4] = summary.b0;  smem[base + 5] = summary.b1;
        __syncthreads();

        // Up-sweep
        #pragma unroll 4
        for (int stride = 1; stride < num_threads; stride *= 2) {
            const int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx-stride)*6], smem[(idx-stride)*6+1],
                                   smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                                   smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 right = {smem[idx*6], smem[idx*6+1], smem[idx*6+2],
                                   smem[idx*6+3], smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 c = affine_combine_ptx(left, right);
                smem[idx*6] = c.m00; smem[idx*6+1] = c.m01;
                smem[idx*6+2] = c.m10; smem[idx*6+3] = c.m11;
                smem[idx*6+4] = c.b0; smem[idx*6+5] = c.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        if (ltid == 0) {
            const int last = (num_threads - 1) * 6;
            smem[last]   = 1.0f; smem[last+1] = 0.0f;
            smem[last+2] = 0.0f; smem[last+3] = 1.0f;
            smem[last+4] = 0.0f; smem[last+5] = 0.0f;
        }
        __syncthreads();
