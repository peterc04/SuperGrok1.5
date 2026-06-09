#pragma once
// Canonical header (de-inlined). Body is byte-identical to the
// formerly copy-pasted block; prerequisites are included so that
// platform macros precede their use.
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/scan/affine2x2.h"
#include "csrc/common/utils.cuh"

// ── inlined from former csrc/common/utils.cuh (Phase3 S0) ──
#if GROK_CUDA
#ifndef SG_INLINE_PTX_PTX_EXP2
#define SG_INLINE_PTX_PTX_EXP2
// Fast exp2 approximation via PTX ex2.approx.f32.
// Used in Mamba scan: exp(A * dt) = exp2(A * dt / ln2).
__device__ __forceinline__ float ptx_exp2(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
#endif  // SG_INLINE_PTX_PTX_EXP2

#ifndef SG_INLINE_PTX_PTX_EXPF
#define SG_INLINE_PTX_PTX_EXPF
// Fast exp via exp2: exp(x) = exp2(x * log2(e))
__device__ __forceinline__ float ptx_expf(float x) {
    return ptx_exp2(x * 1.4426950408889634f);  // log2(e)
}
#endif  // SG_INLINE_PTX_PTX_EXPF
#endif  // GROK_CUDA

#if GROK_HIP
#ifndef SG_INLINE_PTX_PTX_EXPF
#define SG_INLINE_PTX_PTX_EXPF
__device__ __forceinline__ float ptx_expf(float x) { return expf(x); }
#endif  // SG_INLINE_PTX_PTX_EXPF
#endif  // GROK_HIP

// csrc/scan/mamba_scan_adapter.cuh — CUDA scan adapter.
// Moved here in Phase 4 of the refactor because the Mamba selective scan is
// shared between the Mamba model kernels and the SuperGrok v2 optimizer.
//
// Thin adapter wrapping SG2's existing mamba3_* scan kernels for model-context
// use. No reimplementation of the core scan algorithm — reuses the Affine2x2
// parallel-prefix infrastructure from csrc/scan/affine2x2.h.
//
// The adapter packs model-level (x, dt, A_log, B, C) into Affine2x2 maps:
//   A_bar = exp(dt * A),  B_bar = dt * B
//   Affine2x2: M = diag(A_bar_s0, A_bar_s1),  b = (B_bar_s0*x, B_bar_s1*x)
// then calls the Blelloch parallel-prefix scan for medium/large N, or a
// simple sequential scan for small N.
//
// Decision tree (thresholds from csrc/common/types.h):
//   N < PSCAN_THRESHOLD (256)               -> sequential scan kernel
//   256 <= N < GEMM_PRECOMPUTE_THRESHOLD    -> parallel Blelloch scan
//   N >= GEMM_PRECOMPUTE_THRESHOLD (1024)   -> parallel Blelloch scan (same kernel)



namespace sg { namespace sm90 { namespace models { namespace mamba_adapter {

// ── Sequential scan kernel (N < PSCAN_THRESHOLD) ──────────────────────
// One thread per d_inner dimension, sequential over timesteps.

template <typename ActT>
__global__ void __launch_bounds__(128, 4)
sequential_scan_kernel(
    const ActT* __restrict__ x,       // [B, N, d_inner]
    const ActT* __restrict__ dt,      // [B, N, d_inner]
    const ActT* __restrict__ A_log,   // [d_inner, d_state]
    const ActT* __restrict__ B,       // [B, N, d_state]
    const ActT* __restrict__ C,       // [B, N, d_state]
    ActT* __restrict__ y,             // [B, N, d_inner]
    float* __restrict__ state_save,   // [B, d_inner, d_state] or nullptr
    int seq_len, int d_inner, int d_state
) {
    const int b = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_inner) return;

    const int bN  = b * seq_len;
    const int bDi = b * d_inner;

    float A[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -ptx_expf(static_cast<float>(A_log[j * d_state + s]));

    float h[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) h[s] = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        float x_val  = static_cast<float>(x[(bN + t) * d_inner + j]);
        float dt_val = static_cast<float>(dt[(bN + t) * d_inner + j]);
        float y_acc  = 0.0f;

        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            float A_bar = ptx_expf(A[s] * dt_val);
            float B_bar = dt_val * static_cast<float>(B[(bN + t) * d_state + s]);
            h[s] = A_bar * h[s] + B_bar * x_val;
            y_acc += static_cast<float>(C[(bN + t) * d_state + s]) * h[s];
        }
        y[(bN + t) * d_inner + j] = static_cast<ActT>(y_acc);
    }

    if (state_save != nullptr) {
        #pragma unroll 4
        for (int s = 0; s < d_state; s++)
            state_save[(bDi + j) * d_state + s] = h[s];
    }
}

// ── Parallel Blelloch scan kernel (N >= PSCAN_THRESHOLD) ──────────────
// One block per (batch, d_inner). Affine2x2 prefix scan across timesteps,
// processing d_state pairs two at a time through the 2x2 matrix machinery.

template <typename ActT>
__global__ void __launch_bounds__(256, 2)
parallel_scan_kernel(
    const ActT* __restrict__ x,
    const ActT* __restrict__ dt,
    const ActT* __restrict__ A_log,
    const ActT* __restrict__ B,
    const ActT* __restrict__ C,
    ActT* __restrict__ y,
    float* __restrict__ state_save,
    int seq_len, int d_inner, int d_state
) {
    const int b = blockIdx.y;
    const int j = blockIdx.x;
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int N = seq_len;
    const int bN = b * N;
    const int bDi = b * d_inner;

    extern __shared__ float smem[];  // 6 * nthreads

    const int chunk = (N + nthreads - 1) / nthreads;
    const int t0 = ltid * chunk;
    const int t1 = min(t0 + chunk, N);
    const int cnt = max(t1 - t0, 0);

    float A_coeff[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A_coeff[s] = -ptx_expf(static_cast<float>(A_log[j * d_state + s]));

    // Zero output for accumulation across d_state pairs
    for (int step = 0; step < cnt; step++) {
        y[(bN + t0 + step) * d_inner + j] = static_cast<ActT>(0.0f);
    }
    __syncthreads();

    const int half_ds = d_state / 2;

    for (int p = 0; p < half_ds; p++) {
        const int s0 = 2 * p, s1 = 2 * p + 1;

        // Phase 1: sequential scan within chunk -> summary Affine2x2
        Affine2x2 summary = affine_identity();
        #pragma unroll 4
        for (int step = 0; step < cnt; step++) {
            int t = t0 + step;
            float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
            float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
            Affine2x2 elem;
            elem.m00 = ptx_expf(A_coeff[s0] * dtv);  elem.m01 = 0.0f;
            elem.m10 = 0.0f;                          elem.m11 = ptx_expf(A_coeff[s1] * dtv);
            elem.b0  = dtv * static_cast<float>(B[(bN + t) * d_state + s0]) * xv;
            elem.b1  = dtv * static_cast<float>(B[(bN + t) * d_state + s1]) * xv;
            summary = affine_combine(summary, elem);
        }

        int base = ltid * 6;
        smem[base]   = summary.m00; smem[base+1] = summary.m01;
        smem[base+2] = summary.m10; smem[base+3] = summary.m11;
        smem[base+4] = summary.b0;  smem[base+5] = summary.b1;
        __syncthreads();

        // Phase 2: Blelloch up-sweep
        for (int stride = 1; stride < nthreads; stride *= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < nthreads) {
                Affine2x2 L = {smem[(idx-stride)*6],   smem[(idx-stride)*6+1],
                               smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                               smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 R = {smem[idx*6],   smem[idx*6+1],
                               smem[idx*6+2], smem[idx*6+3],
                               smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 c = affine_combine(L, R);
                smem[idx*6]=c.m00; smem[idx*6+1]=c.m01; smem[idx*6+2]=c.m10;
                smem[idx*6+3]=c.m11; smem[idx*6+4]=c.b0; smem[idx*6+5]=c.b1;
            }
            __syncthreads();
        }

        // Set last to identity (exclusive scan)
        if (ltid == 0) {
            int last = (nthreads - 1) * 6;
            smem[last]=1; smem[last+1]=0; smem[last+2]=0;
            smem[last+3]=1; smem[last+4]=0; smem[last+5]=0;
        }
        __syncthreads();

        // Down-sweep
        for (int stride = nthreads / 2; stride >= 1; stride /= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < nthreads) {
                Affine2x2 L = {smem[(idx-stride)*6],   smem[(idx-stride)*6+1],
                               smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                               smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 R = {smem[idx*6],   smem[idx*6+1],
                               smem[idx*6+2], smem[idx*6+3],
                               smem[idx*6+4], smem[idx*6+5]};
                smem[(idx-stride)*6]=R.m00; smem[(idx-stride)*6+1]=R.m01;
                smem[(idx-stride)*6+2]=R.m10; smem[(idx-stride)*6+3]=R.m11;
                smem[(idx-stride)*6+4]=R.b0; smem[(idx-stride)*6+5]=R.b1;
                Affine2x2 c = affine_combine(R, L);
                smem[idx*6]=c.m00; smem[idx*6+1]=c.m01; smem[idx*6+2]=c.m10;
                smem[idx*6+3]=c.m11; smem[idx*6+4]=c.b0; smem[idx*6+5]=c.b1;
            }
            __syncthreads();
        }

        // Phase 3: re-scan with prefix, accumulate output
        Affine2x2 run = {smem[ltid*6], smem[ltid*6+1], smem[ltid*6+2],
                         smem[ltid*6+3], smem[ltid*6+4], smem[ltid*6+5]};
        #pragma unroll 4
        for (int step = 0; step < cnt; step++) {
            int t = t0 + step;
            float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
            float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
            Affine2x2 elem;
            elem.m00 = ptx_expf(A_coeff[s0] * dtv);  elem.m01 = 0.0f;
            elem.m10 = 0.0f;                          elem.m11 = ptx_expf(A_coeff[s1] * dtv);
            elem.b0  = dtv * static_cast<float>(B[(bN + t) * d_state + s0]) * xv;
            elem.b1  = dtv * static_cast<float>(B[(bN + t) * d_state + s1]) * xv;
            run = affine_combine(run, elem);

            // h = run applied to zero initial state -> h = run.b
            float c0 = static_cast<float>(C[(bN + t) * d_state + s0]);
            float c1 = static_cast<float>(C[(bN + t) * d_state + s1]);
            float prev = static_cast<float>(y[(bN + t) * d_inner + j]);
            y[(bN + t) * d_inner + j] = static_cast<ActT>(prev + run.b0*c0 + run.b1*c1);
        }

        if (state_save != nullptr && t1 == N && cnt > 0) {
            state_save[(bDi + j) * d_state + s0] = run.b0;
            state_save[(bDi + j) * d_state + s1] = run.b1;
        }
        __syncthreads();
    }

    // Handle odd d_state
    if (d_state % 2 != 0) {
        const int s = d_state - 1;
        float hv = 0.0f;
        for (int step = 0; step < cnt; step++) {
            int t = t0 + step;
            float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
            float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
            hv = ptx_expf(A_coeff[s] * dtv) * hv
               + dtv * static_cast<float>(B[(bN + t) * d_state + s]) * xv;
            float prev = static_cast<float>(y[(bN + t) * d_inner + j]);
            float cv   = static_cast<float>(C[(bN + t) * d_state + s]);
            y[(bN + t) * d_inner + j] = static_cast<ActT>(prev + hv * cv);
        }
        if (state_save != nullptr && t1 == N && cnt > 0)
            state_save[(bDi + j) * d_state + s] = hv;
    }
}

// ── Forward dispatch ──────────────────────────────────────────────────

template <typename ActT>
cudaError_t selective_scan_forward(
    const ActT* x, const ActT* dt, const ActT* A_log,
    const ActT* B, const ActT* C,
    ActT* y, float* state_save,
    int batch, int seq_len, int d_inner, int d_state,
    cudaStream_t stream
) {
    if (seq_len < PSCAN_THRESHOLD) {
        int block = min(d_inner, 128);
        dim3 grid((d_inner + block - 1) / block, batch);
        sequential_scan_kernel<ActT><<<grid, block, 0, stream>>>(
            x, dt, A_log, B, C, y, state_save, seq_len, d_inner, d_state);
    } else {
        int block = min(PSCAN_BLOCK, 256);
        dim3 grid(d_inner, batch);
        int smem_bytes = 6 * block * sizeof(float);
        parallel_scan_kernel<ActT><<<grid, block, smem_bytes, stream>>>(
            x, dt, A_log, B, C, y, state_save, seq_len, d_inner, d_state);
    }
    return cudaGetLastError();
}

// ── Backward: adjoint scan ────────────────────────────────────────────
// Reverse-time sequential scan computing gradients through the recurrence.
// For each timestep t (in reverse):
//   grad_h += C[t] * grad_y[t]
//   grad_B[t] = dt[t] * x[t] * grad_h
//   grad_C[t] = h[t] * grad_y[t]   (h[t] recomputed via forward pass)
//   grad_x[t] = sum_s(B[t,s] * dt[t] * grad_h[s])
//   grad_dt[t] = sum_s(A[s]*A_bar*h[t-1,s] + B[t,s]*x[t]) * grad_h[s]
//   grad_A_log[j,s] += dt[t]*A[s]*A_bar * h[t-1,s] * grad_h[s]
//   grad_h = A_bar * grad_h   (backprop through recurrence)

// Gradient checkpointing replaces the O(seq_len * d_state) per-thread h_all
// buffer (128 KB at seq_len=256, d_state=128) with O(sqrt(seq_len) * d_state)
// checkpoint + segment buffers (~17 KB), enabling nvcc to compile the kernel.
static constexpr int BW_CKPT_STRIDE = 32;
static constexpr int BW_MAX_SEQ     = 1024;
static constexpr int BW_MAX_SEGS    = BW_MAX_SEQ / BW_CKPT_STRIDE;

template <typename ActT>
__global__ void __launch_bounds__(128, 2)
scan_backward_kernel(
    const ActT* __restrict__ grad_y,
    const ActT* __restrict__ x,
    const ActT* __restrict__ dt,
    const ActT* __restrict__ A_log,
    const ActT* __restrict__ B,
    const ActT* __restrict__ C,
    const float* __restrict__ state_save,
    ActT* __restrict__ grad_x,
    ActT* __restrict__ grad_dt,
    float* __restrict__ grad_A_log,  // [d_inner, d_state], atomicAdd
    ActT* __restrict__ grad_B,
    ActT* __restrict__ grad_C,
    int seq_len, int d_inner, int d_state
) {
    const int b = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_inner) return;

    const int bN  = b * seq_len;
    const int n_seg = (seq_len + BW_CKPT_STRIDE - 1) / BW_CKPT_STRIDE;

    float A[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -ptx_expf(static_cast<float>(A_log[j * d_state + s]));

    // Checkpoint buffer: h state at segment boundaries.
    // h_ckpt[seg] = hidden state BEFORE timestep (seg * BW_CKPT_STRIDE).
    float h_ckpt[BW_MAX_SEGS * MAX_D_STATE];

    // Segment recompute buffer: h[seg_start-1 .. seg_end].
    // h_seg[0] = checkpoint (h before first timestep), h_seg[i+1] = h after step i.
    float h_seg[(BW_CKPT_STRIDE + 1) * MAX_D_STATE];

    // Forward pass: store checkpoint at each segment boundary
    float h_run[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) {
        h_run[s] = 0.0f;
        h_ckpt[s] = 0.0f;
    }

    for (int t = 0; t < seq_len; t++) {
        float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
        float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            float A_bar = ptx_expf(A[s] * dtv);
            float B_bar = dtv * static_cast<float>(B[(bN + t) * d_state + s]);
            h_run[s] = A_bar * h_run[s] + B_bar * xv;
        }
        if ((t + 1) % BW_CKPT_STRIDE == 0) {
            int ci = (t + 1) / BW_CKPT_STRIDE;
            #pragma unroll 4
            for (int s = 0; s < d_state; s++)
                h_ckpt[ci * d_state + s] = h_run[s];
        }
    }

    // Reverse pass: process segments from last to first
    float grad_h[MAX_D_STATE];
    float grad_A_acc[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) { grad_h[s] = 0.0f; grad_A_acc[s] = 0.0f; }

    for (int seg = n_seg - 1; seg >= 0; seg--) {
        const int seg_start = seg * BW_CKPT_STRIDE;
        const int seg_end   = min(seg_start + BW_CKPT_STRIDE, seq_len);
        const int seg_len   = seg_end - seg_start;

        // Load checkpoint: h_seg[0] = h state before seg_start
        #pragma unroll 4
        for (int s = 0; s < d_state; s++)
            h_seg[s] = h_ckpt[seg * d_state + s];

        // Recompute forward through segment to fill h_seg[1..seg_len]
        for (int i = 0; i < seg_len; i++) {
            int t = seg_start + i;
            float xv  = static_cast<float>(x[(bN + t) * d_inner + j]);
            float dtv = static_cast<float>(dt[(bN + t) * d_inner + j]);
            #pragma unroll 4
            for (int s = 0; s < d_state; s++) {
                float A_bar = ptx_expf(A[s] * dtv);
                float B_bar = dtv * static_cast<float>(B[(bN + t) * d_state + s]);
                h_seg[(i + 1) * d_state + s] = A_bar * h_seg[i * d_state + s] + B_bar * xv;
            }
        }

        // Backward through segment (reverse)
        for (int i = seg_len - 1; i >= 0; i--) {
            int t = seg_start + i;
            float xv   = static_cast<float>(x[(bN + t) * d_inner + j]);
            float dtv  = static_cast<float>(dt[(bN + t) * d_inner + j]);
            float gy   = static_cast<float>(grad_y[(bN + t) * d_inner + j]);

            float grad_x_acc  = 0.0f;
            float grad_dt_acc = 0.0f;

            #pragma unroll 4
            for (int s = 0; s < d_state; s++) {
                float A_bar = ptx_expf(A[s] * dtv);
                float bv    = static_cast<float>(B[(bN + t) * d_state + s]);
                float cv    = static_cast<float>(C[(bN + t) * d_state + s]);
                float h_t   = h_seg[(i + 1) * d_state + s];
                float h_tm1 = h_seg[i * d_state + s];

                grad_C[(bN + t) * d_state + s] = static_cast<ActT>(h_t * gy);
                grad_h[s] += cv * gy;
                grad_B[(bN + t) * d_state + s] = static_cast<ActT>(dtv * xv * grad_h[s]);
                grad_x_acc += bv * dtv * grad_h[s];
                grad_dt_acc += (A[s] * A_bar * h_tm1 + bv * xv) * grad_h[s];
                grad_A_acc[s] += dtv * A[s] * A_bar * h_tm1 * grad_h[s];
                grad_h[s] = A_bar * grad_h[s];
            }

            grad_x[(bN + t) * d_inner + j]  = static_cast<ActT>(grad_x_acc);
            grad_dt[(bN + t) * d_inner + j] = static_cast<ActT>(grad_dt_acc);
        }
    }

    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        atomicAdd(&grad_A_log[j * d_state + s], grad_A_acc[s]);
}

// ── Backward dispatch ─────────────────────────────────────────────────

template <typename ActT>
cudaError_t selective_scan_backward(
    const ActT* grad_y,
    const ActT* x, const ActT* dt, const ActT* A_log,
    const ActT* B, const ActT* C,
    const float* state_save,
    ActT* grad_x, ActT* grad_dt, float* grad_A_log,
    ActT* grad_B, ActT* grad_C,
    int batch, int seq_len, int d_inner, int d_state,
    cudaStream_t stream
) {
    cudaMemsetAsync(grad_A_log, 0, d_inner * d_state * sizeof(float), stream);
    int block = min(d_inner, 128);
    dim3 grid((d_inner + block - 1) / block, batch);
    scan_backward_kernel<ActT><<<grid, block, 0, stream>>>(
        grad_y, x, dt, A_log, B, C, state_save,
        grad_x, grad_dt, grad_A_log, grad_B, grad_C,
        seq_len, d_inner, d_state);
    return cudaGetLastError();
}

}}}}  // namespace sg::sm90::models::mamba_adapter
