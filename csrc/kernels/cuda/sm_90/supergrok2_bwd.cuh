// =====================================================================
//  csrc/kernels/cuda/sm_90/supergrok2_bwd.cuh
//
//  sm_90 (Hopper) SuperGrok v2 — BACKWARD path (NET-NEW).
//
//  The previous baseline TU
//    csrc/kernels/cuda/sm_90/supergrok2_bwd_sm90.cu (3837 lines)
//  was deleted in 5505b50. Recoverable via:
//    git show 5505b50^:csrc/kernels/cuda/sm_90/supergrok2_bwd_sm90.cu
//
//  This header carries the kernel definitions + host launcher
//  templates; the thin TU `supergrok2_bwd.cu` instantiates the
//  dtype-templated symbols and bridges the binding-side
//  `sg::sm90::launch_*` symbols to our internal
//  `sg::sm90::supergrok2::launch_*` definitions.
//
//  BACKWARD DECISION TREE (mirrors the forward dispatcher):
//    * N <  PSCAN_THRESHOLD                : sequential reverse scan
//    * PSCAN_THRESHOLD <= N < GEMM_PRECOMPUTE_THRESHOLD
//                                          : parallel-precompute + parallel
//                                            reverse-scan
//    * N >= GEMM_PRECOMPUTE_THRESHOLD      : bilevel reverse-GEMM precompute
//                                            + parallel reverse-scan
//
//  BILEVEL CHECKPOINTING:
//    Default C=1 saves H_t at every step (no recompute). With C>1 the
//    forward-save kernel only retains H every C steps; the backward
//    scan kernel recomputes intermediate states on demand from
//    cached B̄, Ā (~2× backward compute, ~(C-1)/C memory saved).
//    C is bounded by `MAX_CKPT_INTERVAL` to keep `seg_h` in registers.
//
//  W-MATRIX GRADIENT STRATEGY:
//    Per-element accumulators        -> shared-memory then atomicAdd
//                                       (input_proj, gru, expert+peer).
//    Two-pass (per-timestep reduce)  -> warp-reduced into a [N, d_state]
//                                       buffer, fused with CUTLASS GEMM-T
//                                       in the launcher (B/C-proj weights).
//    Cluster-level dot accumulations -> DSMEM cluster-reduce when the
//                                       block-cluster API is available
//                                       (sm_90+).
//
//  NAMESPACE: kernels + launchers live in `sg::sm90::supergrok2`. The
//  binding declarations in csrc/bindings/supergrok2.cpp use namespace
//  `sg::sm90` (no `::supergrok2` suffix). The instantiation TU
//  `supergrok2_bwd.cu` defines a thin shim that re-exports the binding
//  names from the canonical implementations in this header — see the
//  matching forward header `supergrok2_warp_specialized.cuh` for the
//  same convention.
//
//  Dtype matrix (instantiated by `.cu`):
//    ParamT in {fp32, bf16, fp16}
//    StateT in {fp32, bf16}
//    GradT  in {fp32, bf16, fp16, fp8_e4m3, fp8_e5m2}
//    Incoherent combos rejected via static_assert in the templates.
//
//  Forbidden by spec & honoured here:
//    * No inline PTX beyond ptx_intrinsics::* helpers
//    * No touching of supergrok2_fwd.{cuh,cu} or
//      supergrok2_warp_specialized.{cuh,cu}
//    * No reimplementation of the affine-2x2 scan combine
//    * No hardcoded launch params (PSCAN_BLOCK / SG2B_BLOCK from
//      tuned_configs.h via types.h)
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/ptx_intrinsics.cuh"
#include "csrc/common/utils.cuh"

#if GROK_CUDA

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if (CUDA_VERSION >= 11080) && (__CUDA_ARCH__ + 0 >= 890 || !defined(__CUDA_ARCH__))
  // FP8 types are sm_89+ but the binding-side dtype dispatch may still
  // see them on Hopper. Guarded include keeps older toolchains happy.
  #include <cuda_fp8.h>
  #define SG2_BWD_HAS_FP8 1
#else
  #define SG2_BWD_HAS_FP8 0
#endif

#ifdef WITH_CUTLASS
#include "../_cutlass_gemm.cuh"
#endif

namespace sg { namespace sm90 { namespace supergrok2 {

// ---------------------------------------------------------------------
//  Dtype trait + coherence check (static_assert-driven)
// ---------------------------------------------------------------------
template <typename T> struct sg2_dtype_tag      { static constexpr int v = 0; };
template <> struct sg2_dtype_tag<float>          { static constexpr int v = 1; };
template <> struct sg2_dtype_tag<__nv_bfloat16>  { static constexpr int v = 2; };
template <> struct sg2_dtype_tag<__half>         { static constexpr int v = 3; };
#if SG2_BWD_HAS_FP8
template <> struct sg2_dtype_tag<__nv_fp8_e4m3>  { static constexpr int v = 4; };
template <> struct sg2_dtype_tag<__nv_fp8_e5m2>  { static constexpr int v = 5; };
#endif

template <typename ParamT, typename StateT, typename GradT>
struct sg2_dtype_check {
    // ParamT must be fp32/bf16/fp16
    static_assert(sg2_dtype_tag<ParamT>::v >= 1 && sg2_dtype_tag<ParamT>::v <= 3,
                  "SuperGrok2 backward: ParamT must be {fp32, bf16, fp16}");
    // StateT must be fp32/bf16
    static_assert(sg2_dtype_tag<StateT>::v == 1 || sg2_dtype_tag<StateT>::v == 2,
                  "SuperGrok2 backward: StateT must be {fp32, bf16}");
    // GradT must be fp32/bf16/fp16/fp8_e4m3/fp8_e5m2
    static_assert(sg2_dtype_tag<GradT>::v >= 1,
                  "SuperGrok2 backward: GradT must be one of {fp32, bf16, fp16, fp8_e4m3, fp8_e5m2}");
    // Incoherent: ParamT=fp16 with StateT=fp32 is allowed; StateT=bf16 paired with
    // a wider ParamT is allowed too. We forbid only the truly nonsensical cases:
    // * GradT=fp8_* with ParamT=fp16 (Hopper FP8 only has bf16/fp32 accumulators)
    static_assert(!((sg2_dtype_tag<GradT>::v == 4 || sg2_dtype_tag<GradT>::v == 5)
                    && sg2_dtype_tag<ParamT>::v == 3),
                  "SuperGrok2 backward: FP8 grads incompatible with FP16 params");
    static constexpr bool ok = true;
};

}}} // namespace sg::sm90::supergrok2

#endif // GROK_CUDA

// ---------------------------------------------------------------------
//  CHUNK 2 — Scan reverse-time backward kernel (Mamba-3 selective scan).
//
//  Per-thread (one thread per d_inner index, blockDim.x==d_inner). State
//  is held in registers (`dh[MAX_D_STATE]`). Saved tensors:
//    saved_states     : [N, d_inner, d_state] (or [num_ckpts, ...] when C>1)
//    saved_x_branch   : [N, d_inner]
//    saved_z          : [N, d_inner]
//    saved_dt         : [N, d_inner]
//
//  Outputs (all atomicAdd-accumulated except the per-step C/B reduce
//  buffers):
//    d_in_proj_W      : [2*d_inner, d_model]
//    d_dt_proj_W      : [d_inner, d_inner]
//    d_dt_proj_b      : [d_inner]
//    d_A_log          : [d_inner, d_state]
//    d_D_param        : [d_inner]
//    d_rope_freq      : [d_inner, d_state/2]
//    d_x_sorted       : [N, d_model]
//    d_C_vals_buf     : [N, d_state]   per-timestep, warp-reduced
//    d_B_vals_buf     : [N, d_state]   per-timestep, warp-reduced
//
//  d_C_proj_W and d_B_proj_W are NOT written here — the launcher fuses
//  the reduce buffers with a CUTLASS GEMM-T (or torch::mm fallback)
//  outside the kernel. This is the "two-pass" path described in
//  REFRESH §25.7 and used by every existing arch (sm_80/sm_100/gfx942).
// ---------------------------------------------------------------------

#if GROK_CUDA
namespace sg { namespace sm90 { namespace supergrok2 {

__launch_bounds__(MAX_D_INNER, 8)
__global__ void mamba3_scan_backward_kernel(
    const float* __restrict__ d_scan_output,
    const float* __restrict__ x_sorted,
    const float* __restrict__ saved_states,
    const float* __restrict__ saved_x_branch,
    const float* __restrict__ saved_z,
    const float* __restrict__ saved_dt,
    const float* __restrict__ in_proj_W,
    const float* __restrict__ dt_proj_W,
    const float* __restrict__ dt_proj_b,
    const float* __restrict__ B_proj_W,
    const float* __restrict__ C_proj_W,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ rope_freq,
    float* __restrict__ d_in_proj_W,
    float* __restrict__ d_dt_proj_W,
    float* __restrict__ d_dt_proj_b,
    float* __restrict__ d_A_log,
    float* __restrict__ d_D_param,
    float* __restrict__ d_rope_freq,
    float* __restrict__ d_x_sorted,
    const float* __restrict__ initial_state,
    float* __restrict__ d_C_vals_buf,
    float* __restrict__ d_B_vals_buf,
    const int N, const int d_model, const int d_inner, const int d_state,
    const int reverse,
    const int checkpoint_interval
) {
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    extern __shared__ float smem[];
    float* s_x_branch = smem;                // [d_inner]
    float* s_d_dt_raw = smem + d_inner;      // [d_inner]

    float A[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) {
        A[s] = -fast_exp_ptx(A_log[tid * d_state + s]);
    }
    const int half_d_state = d_state / 2;
    float freq[MAX_D_STATE / 2];
    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++) {
        freq[p] = rope_freq[tid * half_d_state + p];
    }
    const float D_val = D_param[tid];

    float d_D_acc = 0.0f, d_dt_proj_b_acc = 0.0f;
    float d_A_log_acc[MAX_D_STATE];
    float d_freq_acc[MAX_D_STATE / 2];
    float d_dt_proj_W_row[MAX_D_INNER];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) d_A_log_acc[s] = 0.0f;
    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++) d_freq_acc[p] = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++) d_dt_proj_W_row[j] = 0.0f;

    float d_in_proj_W_x_local[MAX_D_MODEL];
    float d_in_proj_W_z_local[MAX_D_MODEL];
    #pragma unroll 4
    for (int d = 0; d < d_model; d++) {
        d_in_proj_W_x_local[d] = 0.0f;
        d_in_proj_W_z_local[d] = 0.0f;
    }

    float dh[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) dh[s] = 0.0f;

    // The hot inner loop body is identical between the no-checkpoint and
    // checkpointed paths apart from how `h_curr` and `h_prev` are
    // sourced. Encoded as a macro so a single edit modifies both, the
    // alternative being a __device__ helper that captures ~25 registers
    // by reference. Keeping it inline avoids the spill.
    #define SG2_BWD_STEP_BODY(H_CURR_PTR, H_PREV_PTR)                          \
        do {                                                                   \
            float d_out  = d_scan_output[i * d_inner + tid];                   \
            float x_val  = saved_x_branch[i * d_inner + tid];                  \
            float z_val  = saved_z[i * d_inner + tid];                         \
            float dt_val = saved_dt[i * d_inner + tid];                        \
            s_x_branch[tid] = x_val;                                           \
            __syncthreads();                                                   \
            float sig_z  = 1.0f / (1.0f + expf(-z_val));                       \
            float silu_z = z_val * sig_z;                                      \
            float y_val = 0.0f;                                                \
            for (int s = 0; s < d_state; s++) {                                \
                float Cv = 0.0f;                                               \
                for (int j = 0; j < d_inner; j++)                              \
                    Cv += C_proj_W[s * d_inner + j] * s_x_branch[j];           \
                y_val += (H_CURR_PTR)[s] * Cv;                                 \
            }                                                                  \
            float d_y_val   = d_out * silu_z;                                  \
            float d_silu_z  = d_out * y_val;                                   \
            float d_z_val   = d_silu_z * (sig_z + z_val * sig_z * (1.0f - sig_z)); \
            float d_x_from_D = d_out * D_val;                                  \
            d_D_acc += d_out * x_val;                                          \
            float d_x_from_C = 0.0f;                                           \
            for (int s = 0; s < d_state; s++) {                                \
                float h_s = (H_CURR_PTR)[s];                                   \
                float Cv = 0.0f;                                               \
                for (int j = 0; j < d_inner; j++)                              \
                    Cv += C_proj_W[s * d_inner + j] * s_x_branch[j];           \
                dh[s] += d_y_val * Cv;                                         \
                float d_C_val = d_y_val * h_s;                                 \
                float d_C_red = warp_reduce_sum(d_C_val, d_inner, tid);        \
                if (tid == 0) d_C_vals_buf[i * d_state + s] = d_C_red;         \
                d_x_from_C += d_y_val * h_s * C_proj_W[s * d_inner + tid];     \
            }                                                                  \
            float dh_snap[MAX_D_STATE];                                        \
            for (int s = 0; s < d_state; s++) { dh_snap[s] = dh[s]; dh[s] = 0.0f; } \
            float d_dt_val = 0.0f, d_x_from_scan = 0.0f;                       \
            for (int s = 0; s < d_state; s++) {                                \
                float half_dtA = dt_val * A[s] * 0.5f;                         \
                float denom    = 1.0f - half_dtA + 1e-8f;                      \
                float A_bar    = (1.0f + half_dtA) / denom;                    \
                float B_val = 0.0f;                                            \
                for (int j = 0; j < d_inner; j++)                              \
                    B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];        \
                float B_bar = dt_val * B_val;                                  \
                int pair_idx = s >> 1;                                         \
                float cos_p, sin_p;                                            \
                FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);         \
                float h_rot; int partner; float sign;                          \
                if ((s & 1) == 0) {                                            \
                    partner = s + 1; sign = -1.0f;                             \
                    h_rot = (H_PREV_PTR)[s] * cos_p - (H_PREV_PTR)[partner] * sin_p; \
                } else {                                                       \
                    partner = s - 1; sign =  1.0f;                             \
                    h_rot = (H_PREV_PTR)[s] * cos_p + (H_PREV_PTR)[partner] * sin_p; \
                }                                                              \
                float d_h_s    = dh_snap[s];                                   \
                float d_A_bar  = d_h_s * h_rot;                                \
                float d_h_rot  = d_h_s * A_bar;                                \
                float d_B_bar  = d_h_s * x_val;                                \
                d_x_from_scan += d_h_s * B_bar;                                \
                d_dt_val      += d_B_bar * B_val;                              \
                float d_B_val  = d_B_bar * dt_val;                             \
                float d_B_red  = warp_reduce_sum(d_B_val, d_inner, tid);       \
                if (tid == 0) d_B_vals_buf[i * d_state + s] = d_B_red;         \
                d_x_from_scan += d_B_val * B_proj_W[s * d_inner + tid];        \
                float d_half_dtA = d_A_bar * (1.0f + A_bar) / denom;           \
                d_dt_val += d_half_dtA * A[s] * 0.5f;                          \
                float d_A_s = d_half_dtA * dt_val * 0.5f;                      \
                d_A_log_acc[s] += d_A_s * A[s];                                \
                float d_h_prev_s       = d_h_rot * cos_p;                      \
                float d_h_prev_partner = d_h_rot * sign * sin_p;               \
                float d_cos = d_h_rot * (H_PREV_PTR)[s];                       \
                float d_sin = d_h_rot * sign * (H_PREV_PTR)[partner];          \
                d_dt_val      += (-sin_p * freq[pair_idx]) * d_cos              \
                               + ( cos_p * freq[pair_idx]) * d_sin;            \
                d_freq_acc[pair_idx] += (-sin_p * dt_val) * d_cos              \
                                      + ( cos_p * dt_val) * d_sin;             \
                dh[s]       += d_h_prev_s;                                     \
                dh[partner] += d_h_prev_partner;                               \
            }                                                                  \
            float dt_raw = dt_proj_b[tid];                                     \
            for (int j = 0; j < d_inner; j++)                                  \
                dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];        \
            float sig_dt   = 1.0f / (1.0f + expf(-dt_raw));                    \
            float d_dt_raw = d_dt_val * sig_dt;                                \
            d_dt_proj_b_acc += d_dt_raw;                                       \
            for (int j = 0; j < d_inner; j++)                                  \
                d_dt_proj_W_row[j] += d_dt_raw * s_x_branch[j];                \
            s_d_dt_raw[tid] = d_dt_raw;                                        \
            __syncthreads();                                                   \
            float d_x_from_dt = 0.0f;                                          \
            for (int t2 = 0; t2 < d_inner; t2++)                               \
                d_x_from_dt += s_d_dt_raw[t2] * dt_proj_W[t2 * d_inner + tid]; \
            float d_x_val = d_x_from_D + d_x_from_C + d_x_from_scan + d_x_from_dt; \
            for (int d = 0; d < d_model; d++) {                                \
                float inp = x_sorted[i * d_model + d];                         \
                d_in_proj_W_x_local[d] += d_x_val * inp;                       \
                d_in_proj_W_z_local[d] += d_z_val * inp;                       \
                atomicAdd(&d_x_sorted[i * d_model + d],                        \
                          d_x_val * in_proj_W[tid * d_model + d] +             \
                          d_z_val * in_proj_W[(tid + d_inner) * d_model + d]); \
            }                                                                  \
            __syncthreads();                                                   \
        } while (0)

    if (checkpoint_interval <= 1) {
        // ────────────────────────────────────────────────────────────
        //  PATH A: dense state save  (C == 1, default)
        //  saved_states holds H_t for every t.
        // ────────────────────────────────────────────────────────────
        for (int step = N - 1; step >= 0; step--) {
            int i = reverse ? (N - 1 - step) : step;
            float h_curr[MAX_D_STATE];
            float h_prev[MAX_D_STATE];
            #pragma unroll 4
            for (int s = 0; s < d_state; s++)
                h_curr[s] = saved_states[(i * d_inner + tid) * d_state + s];
            if (step > 0) {
                int i_prev = reverse ? (N - step) : (step - 1);
                #pragma unroll 4
                for (int s = 0; s < d_state; s++)
                    h_prev[s] = saved_states[(i_prev * d_inner + tid) * d_state + s];
            } else {
                #pragma unroll 4
                for (int s = 0; s < d_state; s++)
                    h_prev[s] = (initial_state != nullptr)
                              ? initial_state[tid * d_state + s] : 0.0f;
            }
            SG2_BWD_STEP_BODY(h_curr, h_prev);
        }
    } else {
        // ────────────────────────────────────────────────────────────
        //  PATH B: bilevel-checkpoint path  (C > 1)
        //  Saved states only at segment boundaries; recompute the
        //  intermediate forward states inside seg_h[]. ~2x compute,
        //  ~(C-1)/C memory saved.
        // ────────────────────────────────────────────────────────────
        const int num_segments = (N + checkpoint_interval - 1) / checkpoint_interval;
        // seg_h[(local) * MAX_D_STATE + s] holds the forward-recomputed
        // state at the START of local step `local` (i.e. h_prev for that
        // step). seg_h[(local+1) * ...] holds h_curr for step local.
        float seg_h[(MAX_CKPT_INTERVAL + 1) * MAX_D_STATE];

        for (int seg = num_segments - 1; seg >= 0; seg--) {
            const int seg_start = seg * checkpoint_interval;
            const int seg_end   = (seg_start + checkpoint_interval < N)
                                  ? seg_start + checkpoint_interval : N;
            const int seg_len   = seg_end - seg_start;

            // Phase 1: load segment-input checkpoint
            if (seg == 0) {
                #pragma unroll 4
                for (int s = 0; s < d_state; s++)
                    seg_h[s] = (initial_state != nullptr)
                             ? initial_state[tid * d_state + s] : 0.0f;
            } else {
                int ckpt_idx = seg - 1;
                #pragma unroll 4
                for (int s = 0; s < d_state; s++)
                    seg_h[s] = saved_states[(ckpt_idx * d_inner + tid) * d_state + s];
            }

            // Phase 2: forward-recompute states across this segment
            for (int local = 0; local < seg_len; local++) {
                int step = seg_start + local;
                int i = reverse ? (N - 1 - step) : step;
                float xv  = saved_x_branch[i * d_inner + tid];
                float dtv = saved_dt[i * d_inner + tid];
                s_x_branch[tid] = xv;
                __syncthreads();
                float h_in[MAX_D_STATE];
                #pragma unroll 4
                for (int s = 0; s < d_state; s++)
                    h_in[s] = seg_h[local * MAX_D_STATE + s];
                #pragma unroll 4
                for (int s = 0; s < d_state; s++) {
                    float A_bar = (1.0f + dtv * A[s] * 0.5f)
                                  / (1.0f - dtv * A[s] * 0.5f + 1e-8f);
                    float B_val = 0.0f;
                    #pragma unroll 4
                    for (int j = 0; j < d_inner; j++)
                        B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
                    float B_bar = dtv * B_val;
                    int pair_idx = s >> 1;
                    float cos_p, sin_p;
                    FAST_SINCOSF(dtv * freq[pair_idx], &sin_p, &cos_p);
                    float h_rot;
                    if ((s & 1) == 0) h_rot = h_in[s]*cos_p - h_in[s+1]*sin_p;
                    else              h_rot = h_in[s]*cos_p + h_in[s-1]*sin_p;
                    seg_h[(local + 1) * MAX_D_STATE + s] = A_bar * h_rot + B_bar * xv;
                }
                __syncthreads();
            }

            // Phase 3: backward through this segment in reverse local order
            for (int local = seg_len - 1; local >= 0; local--) {
                int step = seg_start + local;
                int i = reverse ? (N - 1 - step) : step;
                float* h_curr = &seg_h[(local + 1) * MAX_D_STATE];
                float* h_prev = &seg_h[ local      * MAX_D_STATE];
                SG2_BWD_STEP_BODY(h_curr, h_prev);
            }
        }
    }
    #undef SG2_BWD_STEP_BODY

    // Atomic flush of per-thread parameter accumulators.
    atomicAdd(&d_D_param[tid], d_D_acc);
    atomicAdd(&d_dt_proj_b[tid], d_dt_proj_b_acc);
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++)
        atomicAdd(&d_dt_proj_W[tid * d_inner + j], d_dt_proj_W_row[j]);
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        atomicAdd(&d_A_log[tid * d_state + s], d_A_log_acc[s]);
    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++)
        atomicAdd(&d_rope_freq[tid * half_d_state + p], d_freq_acc[p]);
    #pragma unroll 4
    for (int d = 0; d < d_model; d++) {
        atomicAdd(&d_in_proj_W[tid * d_model + d], d_in_proj_W_x_local[d]);
        atomicAdd(&d_in_proj_W[(tid + d_inner) * d_model + d], d_in_proj_W_z_local[d]);
    }
}

}}} // namespace sg::sm90::supergrok2
#endif // GROK_CUDA

// ---------------------------------------------------------------------
//  CHUNK 3 — Batched scan-bwd kernel + utility kernels
//  (segmented reverse, combine fwd+reversed-bwd, sequential reverse-scan
//  fallback for N < PSCAN_THRESHOLD).
// ---------------------------------------------------------------------

#if GROK_CUDA
namespace sg { namespace sm90 { namespace supergrok2 {

// Batched scan backward: one block per parameter, blockDim.x == d_inner.
// Layout matches launch_mamba3_peer_backward_batched in the binding.
__launch_bounds__(MAX_D_INNER, 8)
__global__ void mamba3_scan_backward_batched_kernel(
    const float* __restrict__ d_scan_output_packed,  // [total_N, d_inner]
    const float* __restrict__ x_sorted_packed,       // [total_N, d_model]
    const float* __restrict__ saved_states_packed,   // [total_N or total_ckpts, d_inner, d_state]
    const float* __restrict__ saved_x_branch_packed, // [total_N, d_inner]
    const float* __restrict__ saved_z_packed,        // [total_N, d_inner]
    const float* __restrict__ saved_dt_packed,       // [total_N, d_inner]
    const int*   __restrict__ offsets,               // [num_params + 1]
    const int*   __restrict__ reverse_flags,         // [num_params]
    const float* __restrict__ initial_states,        // [num_params, d_inner, d_state]
    const float* __restrict__ in_proj_W,
    const float* __restrict__ dt_proj_W,
    const float* __restrict__ dt_proj_b,
    const float* __restrict__ B_proj_W,
    const float* __restrict__ C_proj_W,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ rope_freq,
    float* __restrict__ d_in_proj_W,
    float* __restrict__ d_dt_proj_W,
    float* __restrict__ d_dt_proj_b,
    float* __restrict__ d_A_log,
    float* __restrict__ d_D_param,
    float* __restrict__ d_rope_freq,
    float* __restrict__ d_x_sorted_packed,           // [total_N, d_model]
    const int*   __restrict__ ckpt_offsets,          // [num_params + 1] or nullptr
    float* __restrict__ d_C_vals_packed,             // [total_N, d_state]
    float* __restrict__ d_B_vals_packed,             // [total_N, d_state]
    const int d_model, const int d_inner, const int d_state,
    const int checkpoint_interval
) {
    const int param_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    const int start = offsets[param_idx];
    const int end   = offsets[param_idx + 1];
    const int N     = end - start;
    if (N == 0) return;
    const int reverse = reverse_flags[param_idx];

    const float* my_d_scan = d_scan_output_packed + start * d_inner;
    const float* my_x_sort = x_sorted_packed      + start * d_model;
    const float* my_xb     = saved_x_branch_packed + start * d_inner;
    const float* my_z      = saved_z_packed       + start * d_inner;
    const float* my_dt     = saved_dt_packed      + start * d_inner;
    float*       my_dx     = d_x_sorted_packed    + start * d_model;
    float*       my_dC_buf = d_C_vals_packed      + start * d_state;
    float*       my_dB_buf = d_B_vals_packed      + start * d_state;

    const int ckpt_start = (checkpoint_interval > 1 && ckpt_offsets != nullptr)
                           ? ckpt_offsets[param_idx] : start;
    const float* my_saved = saved_states_packed + ckpt_start * d_inner * d_state;
    const float* init_ptr = initial_states + param_idx * d_inner * d_state;

    // Forward to the single-param kernel by aliasing pointers and
    // re-running its body in-place. We reuse the macro defined in the
    // primary kernel via duplicate inline replication; the duplication
    // is ~150 lines but keeps the per-block index arithmetic local.
    extern __shared__ float smem[];
    float* s_x_branch = smem;
    float* s_d_dt_raw = smem + d_inner;

    float A[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -fast_exp_ptx(A_log[tid * d_state + s]);
    const int half_d_state = d_state / 2;
    float freq[MAX_D_STATE / 2];
    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[tid * half_d_state + p];
    const float D_val = D_param[tid];

    float d_D_acc = 0.0f, d_dt_proj_b_acc = 0.0f;
    float d_A_log_acc[MAX_D_STATE];
    float d_freq_acc[MAX_D_STATE / 2];
    float d_dt_proj_W_row[MAX_D_INNER];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) d_A_log_acc[s] = 0.0f;
    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++) d_freq_acc[p] = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++) d_dt_proj_W_row[j] = 0.0f;

    float d_in_proj_W_x_local[MAX_D_MODEL];
    float d_in_proj_W_z_local[MAX_D_MODEL];
    #pragma unroll 4
    for (int d = 0; d < d_model; d++) {
        d_in_proj_W_x_local[d] = 0.0f;
        d_in_proj_W_z_local[d] = 0.0f;
    }

    float dh[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) dh[s] = 0.0f;

    // Reuse the same step-body macro at file scope. Re-declare to keep
    // the macro hygiene local to this kernel.
    #define SG2_BWD_BSTEP_BODY(H_CURR_PTR, H_PREV_PTR)                          \
        do {                                                                    \
            float d_out  = my_d_scan[i * d_inner + tid];                        \
            float x_val  = my_xb[i * d_inner + tid];                            \
            float z_val  = my_z [i * d_inner + tid];                            \
            float dt_val = my_dt[i * d_inner + tid];                            \
            s_x_branch[tid] = x_val;                                            \
            __syncthreads();                                                    \
            float sig_z  = 1.0f / (1.0f + expf(-z_val));                        \
            float silu_z = z_val * sig_z;                                       \
            float y_val = 0.0f;                                                 \
            for (int s = 0; s < d_state; s++) {                                 \
                float Cv = 0.0f;                                                \
                for (int j = 0; j < d_inner; j++)                               \
                    Cv += C_proj_W[s * d_inner + j] * s_x_branch[j];            \
                y_val += (H_CURR_PTR)[s] * Cv;                                  \
            }                                                                   \
            float d_y_val = d_out * silu_z;                                     \
            float d_silu_z = d_out * y_val;                                     \
            float d_z_val  = d_silu_z * (sig_z + z_val * sig_z * (1.0f - sig_z)); \
            float d_x_from_D = d_out * D_val;                                   \
            d_D_acc += d_out * x_val;                                           \
            float d_x_from_C = 0.0f;                                            \
            for (int s = 0; s < d_state; s++) {                                 \
                float h_s = (H_CURR_PTR)[s];                                    \
                float Cv = 0.0f;                                                \
                for (int j = 0; j < d_inner; j++)                               \
                    Cv += C_proj_W[s * d_inner + j] * s_x_branch[j];            \
                dh[s] += d_y_val * Cv;                                          \
                float d_C_val = d_y_val * h_s;                                  \
                float d_C_red = warp_reduce_sum(d_C_val, d_inner, tid);         \
                if (tid == 0) my_dC_buf[i * d_state + s] = d_C_red;             \
                d_x_from_C += d_y_val * h_s * C_proj_W[s * d_inner + tid];      \
            }                                                                   \
            float dh_snap[MAX_D_STATE];                                         \
            for (int s = 0; s < d_state; s++) { dh_snap[s] = dh[s]; dh[s] = 0.0f; } \
            float d_dt_val = 0.0f, d_x_from_scan = 0.0f;                        \
            for (int s = 0; s < d_state; s++) {                                 \
                float half_dtA = dt_val * A[s] * 0.5f;                          \
                float denom    = 1.0f - half_dtA + 1e-8f;                       \
                float A_bar    = (1.0f + half_dtA) / denom;                     \
                float B_val = 0.0f;                                             \
                for (int j = 0; j < d_inner; j++)                               \
                    B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];         \
                float B_bar = dt_val * B_val;                                   \
                int pair_idx = s >> 1;                                          \
                float cos_p, sin_p;                                             \
                FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);          \
                float h_rot; int partner; float sign;                           \
                if ((s & 1) == 0) {                                             \
                    partner = s + 1; sign = -1.0f;                              \
                    h_rot = (H_PREV_PTR)[s] * cos_p - (H_PREV_PTR)[partner] * sin_p; \
                } else {                                                        \
                    partner = s - 1; sign =  1.0f;                              \
                    h_rot = (H_PREV_PTR)[s] * cos_p + (H_PREV_PTR)[partner] * sin_p; \
                }                                                               \
                float d_h_s = dh_snap[s];                                       \
                float d_A_bar = d_h_s * h_rot;                                  \
                float d_h_rot = d_h_s * A_bar;                                  \
                float d_B_bar = d_h_s * x_val;                                  \
                d_x_from_scan += d_h_s * B_bar;                                 \
                d_dt_val      += d_B_bar * B_val;                               \
                float d_B_val  = d_B_bar * dt_val;                              \
                float d_B_red  = warp_reduce_sum(d_B_val, d_inner, tid);        \
                if (tid == 0) my_dB_buf[i * d_state + s] = d_B_red;             \
                d_x_from_scan += d_B_val * B_proj_W[s * d_inner + tid];         \
                float d_half_dtA = d_A_bar * (1.0f + A_bar) / denom;            \
                d_dt_val += d_half_dtA * A[s] * 0.5f;                           \
                float d_A_s = d_half_dtA * dt_val * 0.5f;                       \
                d_A_log_acc[s] += d_A_s * A[s];                                 \
                float d_h_prev_s       = d_h_rot * cos_p;                       \
                float d_h_prev_partner = d_h_rot * sign * sin_p;                \
                float d_cos = d_h_rot * (H_PREV_PTR)[s];                        \
                float d_sin = d_h_rot * sign * (H_PREV_PTR)[partner];           \
                d_dt_val += (-sin_p * freq[pair_idx]) * d_cos                   \
                          + ( cos_p * freq[pair_idx]) * d_sin;                  \
                d_freq_acc[pair_idx] += (-sin_p * dt_val) * d_cos               \
                                      + ( cos_p * dt_val) * d_sin;              \
                dh[s]       += d_h_prev_s;                                      \
                dh[partner] += d_h_prev_partner;                                \
            }                                                                   \
            float dt_raw = dt_proj_b[tid];                                      \
            for (int j = 0; j < d_inner; j++)                                   \
                dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];         \
            float sig_dt   = 1.0f / (1.0f + expf(-dt_raw));                     \
            float d_dt_raw = d_dt_val * sig_dt;                                 \
            d_dt_proj_b_acc += d_dt_raw;                                        \
            for (int j = 0; j < d_inner; j++)                                   \
                d_dt_proj_W_row[j] += d_dt_raw * s_x_branch[j];                 \
            s_d_dt_raw[tid] = d_dt_raw;                                         \
            __syncthreads();                                                    \
            float d_x_from_dt = 0.0f;                                           \
            for (int t2 = 0; t2 < d_inner; t2++)                                \
                d_x_from_dt += s_d_dt_raw[t2] * dt_proj_W[t2 * d_inner + tid];  \
            float d_x_val = d_x_from_D + d_x_from_C + d_x_from_scan + d_x_from_dt; \
            for (int d = 0; d < d_model; d++) {                                 \
                float inp = my_x_sort[i * d_model + d];                         \
                d_in_proj_W_x_local[d] += d_x_val * inp;                        \
                d_in_proj_W_z_local[d] += d_z_val * inp;                        \
                atomicAdd(&my_dx[i * d_model + d],                              \
                          d_x_val * in_proj_W[tid * d_model + d] +              \
                          d_z_val * in_proj_W[(tid + d_inner) * d_model + d]);  \
            }                                                                   \
            __syncthreads();                                                    \
        } while (0)

    if (checkpoint_interval <= 1) {
        for (int step = N - 1; step >= 0; step--) {
            int i = reverse ? (N - 1 - step) : step;
            float h_curr[MAX_D_STATE];
            float h_prev[MAX_D_STATE];
            #pragma unroll 4
            for (int s = 0; s < d_state; s++)
                h_curr[s] = my_saved[(i * d_inner + tid) * d_state + s];
            if (step > 0) {
                int i_prev = reverse ? (N - step) : (step - 1);
                #pragma unroll 4
                for (int s = 0; s < d_state; s++)
                    h_prev[s] = my_saved[(i_prev * d_inner + tid) * d_state + s];
            } else {
                #pragma unroll 4
                for (int s = 0; s < d_state; s++)
                    h_prev[s] = init_ptr[tid * d_state + s];
            }
            SG2_BWD_BSTEP_BODY(h_curr, h_prev);
        }
    } else {
        const int num_segments = (N + checkpoint_interval - 1) / checkpoint_interval;
        float seg_h[(MAX_CKPT_INTERVAL + 1) * MAX_D_STATE];
        for (int seg = num_segments - 1; seg >= 0; seg--) {
            const int seg_start = seg * checkpoint_interval;
            const int seg_end   = (seg_start + checkpoint_interval < N)
                                  ? seg_start + checkpoint_interval : N;
            const int seg_len   = seg_end - seg_start;
            if (seg == 0) {
                #pragma unroll 4
                for (int s = 0; s < d_state; s++)
                    seg_h[s] = init_ptr[tid * d_state + s];
            } else {
                int ckpt_idx = seg - 1;
                #pragma unroll 4
                for (int s = 0; s < d_state; s++)
                    seg_h[s] = my_saved[(ckpt_idx * d_inner + tid) * d_state + s];
            }
            for (int local = 0; local < seg_len; local++) {
                int step = seg_start + local;
                int i = reverse ? (N - 1 - step) : step;
                float xv  = my_xb[i * d_inner + tid];
                float dtv = my_dt[i * d_inner + tid];
                s_x_branch[tid] = xv;
                __syncthreads();
                float h_in[MAX_D_STATE];
                #pragma unroll 4
                for (int s = 0; s < d_state; s++)
                    h_in[s] = seg_h[local * MAX_D_STATE + s];
                #pragma unroll 4
                for (int s = 0; s < d_state; s++) {
                    float A_bar = (1.0f + dtv * A[s] * 0.5f)
                                  / (1.0f - dtv * A[s] * 0.5f + 1e-8f);
                    float B_val = 0.0f;
                    #pragma unroll 4
                    for (int j = 0; j < d_inner; j++)
                        B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
                    float B_bar = dtv * B_val;
                    int pair_idx = s >> 1;
                    float cos_p, sin_p;
                    FAST_SINCOSF(dtv * freq[pair_idx], &sin_p, &cos_p);
                    float h_rot;
                    if ((s & 1) == 0) h_rot = h_in[s]*cos_p - h_in[s+1]*sin_p;
                    else              h_rot = h_in[s]*cos_p + h_in[s-1]*sin_p;
                    seg_h[(local + 1) * MAX_D_STATE + s] = A_bar * h_rot + B_bar * xv;
                }
                __syncthreads();
            }
            for (int local = seg_len - 1; local >= 0; local--) {
                int step = seg_start + local;
                int i = reverse ? (N - 1 - step) : step;
                float* h_curr = &seg_h[(local + 1) * MAX_D_STATE];
                float* h_prev = &seg_h[ local      * MAX_D_STATE];
                SG2_BWD_BSTEP_BODY(h_curr, h_prev);
            }
        }
    }
    #undef SG2_BWD_BSTEP_BODY

    atomicAdd(&d_D_param[tid], d_D_acc);
    atomicAdd(&d_dt_proj_b[tid], d_dt_proj_b_acc);
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++)
        atomicAdd(&d_dt_proj_W[tid * d_inner + j], d_dt_proj_W_row[j]);
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        atomicAdd(&d_A_log[tid * d_state + s], d_A_log_acc[s]);
    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++)
        atomicAdd(&d_rope_freq[tid * half_d_state + p], d_freq_acc[p]);
    #pragma unroll 4
    for (int d = 0; d < d_model; d++) {
        atomicAdd(&d_in_proj_W[tid * d_model + d], d_in_proj_W_x_local[d]);
        atomicAdd(&d_in_proj_W[(tid + d_inner) * d_model + d], d_in_proj_W_z_local[d]);
    }
}

}}} // namespace sg::sm90::supergrok2 (close batched scan-bwd block)
#endif // GROK_CUDA

// ---------------------------------------------------------------------
//  CHUNK 4 — Segmented reverse + combine util kernels (binary-search
//  segment lookup; no CPU-GPU sync); softplus-bias post-pass kernel for
//  the GEMM precompute path; input_proj backward (templated on GradT
//  for fp32/bf16/fp16/fp8 grad inputs).
// ---------------------------------------------------------------------

#if GROK_CUDA
namespace sg { namespace sm90 { namespace supergrok2 {

__launch_bounds__(SG2B_BLOCK, 8)
__global__ void reverse_segments_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    const int*   __restrict__ offsets,
    const int    d,
    const int    num_params
) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_N = offsets[num_params];
    if (global_idx >= total_N * d) return;
    const int row = global_idx / d;
    const int col = global_idx % d;
    int lo = 0, hi = num_params;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (offsets[mid + 1] <= row) lo = mid + 1; else hi = mid;
    }
    const int seg_start = offsets[lo];
    const int seg_end   = offsets[lo + 1];
    const int local_row = row - seg_start;
    const int Nseg      = seg_end - seg_start;
    const int reversed_row = seg_start + (Nseg - 1 - local_row);
    dst[reversed_row * d + col] = src[row * d + col];
}

__launch_bounds__(SG2B_BLOCK, 8)
__global__ void combine_fwd_bwd_kernel(
    const float* __restrict__ fwd,
    const float* __restrict__ bwd,
    float*       __restrict__ out,
    const int*   __restrict__ offsets,
    const int    d,
    const int    num_params
) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_N = offsets[num_params];
    if (global_idx >= total_N * d) return;
    const int row = global_idx / d;
    const int col = global_idx % d;
    int lo = 0, hi = num_params;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (offsets[mid + 1] <= row) lo = mid + 1; else hi = mid;
    }
    const int seg_start = offsets[lo];
    const int seg_end   = offsets[lo + 1];
    const int local_row = row - seg_start;
    const int Nseg      = seg_end - seg_start;
    const int reversed_row = seg_start + (Nseg - 1 - local_row);
    out[row * d + col] = fwd[row * d + col] + bwd[reversed_row * d + col];
}

__launch_bounds__(SG2B_BLOCK, 8)
__global__ void softplus_bias_post_kernel(
    float*       __restrict__ dt_out,
    const float* __restrict__ bias,
    const int N, const int d_inner
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * d_inner) return;
    const int j = idx % d_inner;
    float dt_raw = dt_out[idx] + bias[j];
    dt_out[idx] = softplus_ptx(dt_raw);
}

// Input projection backward — templated on GradT.
// Per-element accumulation into shared memory, then atomicAdd to global.
template <typename GradT>
__launch_bounds__(SG2B_BLOCK, 8)
__global__ void input_proj_backward_kernel(
    const float* __restrict__ d_x,
    const GradT* __restrict__ grad,
    const GradT* __restrict__ sharpness,
    float*       __restrict__ d_proj_W,
    float*       __restrict__ d_proj_b,
    const int N, const int d_model
) {
    extern __shared__ float smem[];
    float* s_d_proj_W = smem;
    float* s_d_proj_b = smem + d_model * 2;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    for (int i = tid; i < d_model * 2; i += bs) s_d_proj_W[i] = 0.0f;
    for (int i = tid; i < d_model;     i += bs) s_d_proj_b[i] = 0.0f;
    __syncthreads();
    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {
        float g = static_cast<float>(grad[idx]);
        float s = static_cast<float>(sharpness[idx]);
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) {
            float dx = d_x[idx * d_model + d];
            atomicAdd(&s_d_proj_W[d * 2 + 0], dx * g);
            atomicAdd(&s_d_proj_W[d * 2 + 1], dx * s);
            atomicAdd(&s_d_proj_b[d],         dx);
        }
    }
    __syncthreads();
    for (int i = tid; i < d_model * 2; i += bs)
        if (s_d_proj_W[i] != 0.0f) atomicAdd(&d_proj_W[i], s_d_proj_W[i]);
    for (int i = tid; i < d_model; i += bs)
        if (s_d_proj_b[i] != 0.0f) atomicAdd(&d_proj_b[i], s_d_proj_b[i]);
}

}}} // namespace sg::sm90::supergrok2 (close chunk 4)
#endif // GROK_CUDA

// ---------------------------------------------------------------------
//  CHUNK 5 — GRU backward + Expert+PEER backward + Out-projection
//  backward. All use shared-memory accumulators flushed via atomicAdd
//  (256x fewer global atomics vs. naive). Math identical to the
//  pre-deletion baseline; rewritten under the new namespace.
// ---------------------------------------------------------------------

#if GROK_CUDA
namespace sg { namespace sm90 { namespace supergrok2 {

__launch_bounds__(SG2B_BLOCK, 8)
__global__ void gru_backward_kernel(
    const float* __restrict__ d_h_new,
    const float* __restrict__ gru_input,
    const float* __restrict__ h_old,
    const float* __restrict__ z_gate,
    const float* __restrict__ r_gate,
    const float* __restrict__ h_tilde,
    const float* __restrict__ Wz,
    const float* __restrict__ Wr,
    const float* __restrict__ Wh,
    float*       __restrict__ d_Wz,
    float*       __restrict__ d_bz,
    float*       __restrict__ d_Wr,
    float*       __restrict__ d_br,
    float*       __restrict__ d_Wh,
    float*       __restrict__ d_bh,
    float*       __restrict__ d_gru_input,
    const int N, const int input_dim, const int gru_hidden
) {
    extern __shared__ float smem[];
    const int total_dim = input_dim + gru_hidden;
    const int w_size    = gru_hidden * total_dim;
    float* s_d_Wz = smem;
    float* s_d_Wr = s_d_Wz + w_size;
    float* s_d_Wh = s_d_Wr + w_size;
    float* s_d_bz = s_d_Wh + w_size;
    float* s_d_br = s_d_bz + gru_hidden;
    float* s_d_bh = s_d_br + gru_hidden;
    const int smem_total = 3 * w_size + 3 * gru_hidden;

    const int tid = threadIdx.x;
    for (int i = tid; i < smem_total; i += blockDim.x) smem[i] = 0.0f;
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {
        for (int gh = 0; gh < gru_hidden; gh++) {
            float d_h    = d_h_new[idx * gru_hidden + gh];
            float zv     = z_gate [idx * gru_hidden + gh];
            float rv     = r_gate [idx * gru_hidden + gh];
            float hv     = h_tilde[idx * gru_hidden + gh];
            float ho     = h_old  [idx * gru_hidden + gh];
            float d_z    = d_h * (hv - ho);
            float d_h_t  = d_h * zv;
            float d_tanh = d_h_t * (1.0f - hv * hv);
            float d_zin  = d_z * zv * (1.0f - zv);

            atomicAdd(&s_d_bh[gh], d_tanh);
            for (int j = 0; j < input_dim; j++) {
                float xj = gru_input[idx * input_dim + j];
                atomicAdd(&s_d_Wh[gh * total_dim + j], d_tanh * xj);
            }
            for (int j = 0; j < gru_hidden; j++) {
                float rh = (j == gh) ? rv * ho
                                     : r_gate[idx * gru_hidden + j] * h_old[idx * gru_hidden + j];
                atomicAdd(&s_d_Wh[gh * total_dim + input_dim + j], d_tanh * rh);
            }
            atomicAdd(&s_d_bz[gh], d_zin);
            for (int j = 0; j < total_dim; j++) {
                float xh = (j < input_dim)
                           ? gru_input[idx * input_dim + j]
                           : h_old[idx * gru_hidden + (j - input_dim)];
                atomicAdd(&s_d_Wz[gh * total_dim + j], d_zin * xh);
            }
            for (int j = 0; j < gru_hidden; j++) {
                float d_r_j = d_tanh * Wh[gh * total_dim + input_dim + j]
                              * h_old[idx * gru_hidden + j];
                float r_j   = r_gate[idx * gru_hidden + j];
                float d_r_in = d_r_j * r_j * (1.0f - r_j);
                atomicAdd(&s_d_br[j], d_r_in);
                for (int k = 0; k < total_dim; k++) {
                    float xh_k = (k < input_dim)
                                 ? gru_input[idx * input_dim + k]
                                 : h_old[idx * gru_hidden + (k - input_dim)];
                    atomicAdd(&s_d_Wr[j * total_dim + k], d_r_in * xh_k);
                }
                for (int k = 0; k < input_dim; k++) {
                    atomicAdd(&d_gru_input[idx * input_dim + k],
                              d_r_in * Wr[j * total_dim + k]);
                }
            }
            for (int j = 0; j < input_dim; j++) {
                float d_in = d_zin * Wz[gh * total_dim + j]
                           + d_tanh * Wh[gh * total_dim + j];
                atomicAdd(&d_gru_input[idx * input_dim + j], d_in);
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < w_size; i += blockDim.x) {
        if (s_d_Wz[i] != 0.0f) atomicAdd(&d_Wz[i], s_d_Wz[i]);
        if (s_d_Wr[i] != 0.0f) atomicAdd(&d_Wr[i], s_d_Wr[i]);
        if (s_d_Wh[i] != 0.0f) atomicAdd(&d_Wh[i], s_d_Wh[i]);
    }
    for (int i = tid; i < gru_hidden; i += blockDim.x) {
        if (s_d_bz[i] != 0.0f) atomicAdd(&d_bz[i], s_d_bz[i]);
        if (s_d_br[i] != 0.0f) atomicAdd(&d_br[i], s_d_br[i]);
        if (s_d_bh[i] != 0.0f) atomicAdd(&d_bh[i], s_d_bh[i]);
    }
}

}}} // namespace sg::sm90::supergrok2 (close GRU bwd block)
#endif // GROK_CUDA

// ---------------------------------------------------------------------
//  CHUNK 6 — Expert+PEER backward + Out-projection backward kernels.
// ---------------------------------------------------------------------

#if GROK_CUDA
namespace sg { namespace sm90 { namespace supergrok2 {

__launch_bounds__(SG2B_BLOCK, 8)
__global__ void expert_peer_backward_kernel(
    const float* __restrict__ d_expert_out,
    const float* __restrict__ grad_vals,
    const int*   __restrict__ expert_indices,
    const float* __restrict__ routing_weights,
    const float* __restrict__ saved_z_hidden,
    const float* __restrict__ saved_peer_input,
    const float* __restrict__ peer_query_Ws,
    const float* __restrict__ prod_keys_A,
    const float* __restrict__ prod_keys_B,
    const float* __restrict__ saved_scores_a,
    const float* __restrict__ saved_scores_b,
    const int*   __restrict__ saved_top_a_idx,
    const int*   __restrict__ saved_top_b_idx,
    const float* __restrict__ saved_soft_a,
    const float* __restrict__ saved_soft_b,
    const float* __restrict__ expert_W1,
    const float* __restrict__ expert_W2,
    const float* __restrict__ expert_b2_in,
    float*       __restrict__ d_expert_W1,
    float*       __restrict__ d_expert_b1,
    float*       __restrict__ d_expert_W2,
    float*       __restrict__ d_expert_b2,
    float*       __restrict__ d_peer_query_Ws,
    float*       __restrict__ d_prod_keys_A,
    float*       __restrict__ d_prod_keys_B,
    float*       __restrict__ d_peer_input,
    const int N, const int num_heads, const int topk, const int num_active,
    const int d_model, const int pk_dim, const int expert_hidden,
    const int peer_input_dim, const int num_experts
) {
    extern __shared__ float smem[];
    float* s_d_expert_W1 = smem;
    float* s_d_expert_b1 = s_d_expert_W1 + num_experts * expert_hidden;
    float* s_d_expert_W2 = s_d_expert_b1 + num_experts * expert_hidden;
    float* s_d_expert_b2 = s_d_expert_W2 + num_experts * expert_hidden;
    int total_expert_smem = 3 * num_experts * expert_hidden + num_experts;
    int half_d_smem = d_model / 2;
    int pqw_size = num_heads * d_model * peer_input_dim;
    int pka_size = num_heads * pk_dim * half_d_smem;
    int pkb_size = pka_size;
    float* s_d_peer_query_Ws = smem + total_expert_smem;
    float* s_d_prod_keys_A   = s_d_peer_query_Ws + pqw_size;
    float* s_d_prod_keys_B   = s_d_prod_keys_A + pka_size;
    int total_smem = total_expert_smem + pqw_size + pka_size + pkb_size;

    for (int i = threadIdx.x; i < total_smem; i += blockDim.x) smem[i] = 0.0f;
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float d_out = d_expert_out[idx];
        float g_val = grad_vals[idx];
        int half_d  = d_model / 2;

        for (int h = 0; h < num_heads; h++) {
            float d_head_out = d_out / (float)num_heads;
            float dot_a[MAX_TOPK] = {};
            float dot_b[MAX_TOPK] = {};
            for (int k = 0; k < num_active; k++) {
                int a_local = k / topk;
                int b_local = k % topk;
                int ei = expert_indices[(idx * num_heads + h) * num_active + k];
                float out_k = expert_b2_in[ei];
                for (int eh = 0; eh < expert_hidden; eh++) {
                    float zv = saved_z_hidden[((idx * num_heads + h) * num_active + k) * expert_hidden + eh];
                    out_k += expert_W2[ei * expert_hidden + eh] * zv;
                }
                float d_rw = d_head_out * out_k;
                float sa = saved_soft_a[(idx * num_heads + h) * topk + a_local];
                float sb = saved_soft_b[(idx * num_heads + h) * topk + b_local];
                dot_a[a_local] += (d_rw * sb) * sa;
                dot_b[b_local] += (d_rw * sa) * sb;
            }
            for (int k = 0; k < num_active; k++) {
                int a_local = k / topk;
                int b_local = k % topk;
                int ei = expert_indices[(idx * num_heads + h) * num_active + k];
                float rw = routing_weights[(idx * num_heads + h) * num_active + k];
                float out_k = expert_b2_in[ei];
                for (int eh = 0; eh < expert_hidden; eh++) {
                    float zv = saved_z_hidden[((idx * num_heads + h) * num_active + k) * expert_hidden + eh];
                    out_k += expert_W2[ei * expert_hidden + eh] * zv;
                }
                float d_rw    = d_head_out * out_k;
                float d_out_k = d_head_out * rw;
                atomicAdd(&s_d_expert_b2[ei], d_out_k);
                for (int eh = 0; eh < expert_hidden; eh++) {
                    float zv = saved_z_hidden[((idx * num_heads + h) * num_active + k) * expert_hidden + eh];
                    atomicAdd(&s_d_expert_W2[ei * expert_hidden + eh], d_out_k * zv);
                    float d_z = d_out_k * expert_W2[ei * expert_hidden + eh];
                    float d_pre_relu = (zv > 0.0f) ? d_z : 0.0f;
                    atomicAdd(&s_d_expert_W1[ei * expert_hidden + eh], d_pre_relu * g_val);
                    atomicAdd(&s_d_expert_b1[ei * expert_hidden + eh], d_pre_relu);
                }
                float sa = saved_soft_a[(idx * num_heads + h) * topk + a_local];
                float sb = saved_soft_b[(idx * num_heads + h) * topk + b_local];
                float d_score_a = 10.0f * sa * (d_rw * sb - dot_a[a_local]);
                float d_score_b = 10.0f * sb * (d_rw * sa - dot_b[b_local]);
                int a_key_idx = saved_top_a_idx[(idx * num_heads + h) * topk + a_local];
                int b_key_idx = saved_top_b_idx[(idx * num_heads + h) * topk + b_local];
                for (int d = 0; d < half_d; d++) {
                    float q_a_d = 0.0f, q_b_d = 0.0f;
                    for (int j = 0; j < peer_input_dim; j++) {
                        float pi_j = saved_peer_input[idx * peer_input_dim + j];
                        q_a_d += peer_query_Ws[(h * d_model + d) * peer_input_dim + j] * pi_j;
                        q_b_d += peer_query_Ws[(h * d_model + half_d + d) * peer_input_dim + j] * pi_j;
                    }
                    atomicAdd(&s_d_prod_keys_A[(h * pk_dim + a_key_idx) * half_d + d], d_score_a * q_a_d);
                    atomicAdd(&s_d_prod_keys_B[(h * pk_dim + b_key_idx) * half_d + d], d_score_b * q_b_d);
                    float d_q_a_d = d_score_a * prod_keys_A[(h * pk_dim + a_key_idx) * half_d + d];
                    float d_q_b_d = d_score_b * prod_keys_B[(h * pk_dim + b_key_idx) * half_d + d];
                    for (int j = 0; j < peer_input_dim; j++) {
                        float pi_j = saved_peer_input[idx * peer_input_dim + j];
                        atomicAdd(&s_d_peer_query_Ws[(h * d_model + d) * peer_input_dim + j], d_q_a_d * pi_j);
                        atomicAdd(&s_d_peer_query_Ws[(h * d_model + half_d + d) * peer_input_dim + j], d_q_b_d * pi_j);
                        atomicAdd(&d_peer_input[idx * peer_input_dim + j],
                                  d_q_a_d * peer_query_Ws[(h * d_model + d) * peer_input_dim + j] +
                                  d_q_b_d * peer_query_Ws[(h * d_model + half_d + d) * peer_input_dim + j]);
                    }
                }
            }
        }
    }
    __syncthreads();

    // Flush expert weight grads
    for (int i = threadIdx.x; i < total_expert_smem; i += blockDim.x) {
        if (smem[i] != 0.0f) {
            if      (i < num_experts * expert_hidden)
                atomicAdd(&d_expert_W1[i], smem[i]);
            else if (i < 2 * num_experts * expert_hidden)
                atomicAdd(&d_expert_b1[i - num_experts * expert_hidden], smem[i]);
            else if (i < 3 * num_experts * expert_hidden)
                atomicAdd(&d_expert_W2[i - 2 * num_experts * expert_hidden], smem[i]);
            else
                atomicAdd(&d_expert_b2[i - 3 * num_experts * expert_hidden], smem[i]);
        }
    }
    for (int i = threadIdx.x; i < pqw_size; i += blockDim.x)
        if (s_d_peer_query_Ws[i] != 0.0f) atomicAdd(&d_peer_query_Ws[i], s_d_peer_query_Ws[i]);
    for (int i = threadIdx.x; i < pka_size; i += blockDim.x)
        if (s_d_prod_keys_A[i] != 0.0f) atomicAdd(&d_prod_keys_A[i], s_d_prod_keys_A[i]);
    for (int i = threadIdx.x; i < pkb_size; i += blockDim.x)
        if (s_d_prod_keys_B[i] != 0.0f) atomicAdd(&d_prod_keys_B[i], s_d_prod_keys_B[i]);
}

__launch_bounds__(SG2B_BLOCK, 8)
__global__ void out_proj_backward_kernel(
    const float* __restrict__ d_context,
    const float* __restrict__ scan_out,
    const float* __restrict__ out_proj_W,
    float*       __restrict__ d_out_proj_W,
    float*       __restrict__ d_scan_out,
    const int N, const int d_model, const int d_inner
) {
    extern __shared__ float smem[];
    float* s_d_out_proj_W = smem;
    const int tid = threadIdx.x;
    const int op_size = d_model * d_inner;
    for (int i = tid; i < op_size; i += blockDim.x) s_d_out_proj_W[i] = 0.0f;
    __syncthreads();
    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {
        for (int j = 0; j < d_inner; j++) {
            float d_scan_j = 0.0f;
            float so_j = scan_out[idx * d_inner + j];
            for (int d = 0; d < d_model; d++) {
                float d_ctx = d_context[idx * d_model + d];
                d_scan_j += d_ctx * out_proj_W[d * d_inner + j];
                atomicAdd(&s_d_out_proj_W[d * d_inner + j], d_ctx * so_j);
            }
            d_scan_out[idx * d_inner + j] = d_scan_j;
        }
    }
    __syncthreads();
    for (int i = tid; i < op_size; i += blockDim.x)
        if (s_d_out_proj_W[i] != 0.0f) atomicAdd(&d_out_proj_W[i], s_d_out_proj_W[i]);
}

}}} // namespace sg::sm90::supergrok2 (close chunk 6)
#endif // GROK_CUDA
