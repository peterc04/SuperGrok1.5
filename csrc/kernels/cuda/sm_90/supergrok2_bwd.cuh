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
