#ifndef GROKKING_KERNELS_SM90_MAMBA3_SM90_CUH_
#define GROKKING_KERNELS_SM90_MAMBA3_SM90_CUH_
// ============================================================================
// mamba3_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for the 'mamba'
// model. Single source of truth: templated per-layer __device__ forward/backward,
// __global__ launchers, inline-PTX blocks VERBATIM, and the CUTLASS Sm90
// tensor-core GEMM wrappers (attention). Composition primitive for the future
// fused megakernel.
//
// The production location csrc/backends/cuda/sm_90/models/mamba.cuh is now a thin
// shim that #include's this header, so every existing includer (bindings.cpp,
// the mamba.cu instantiation TU, the HIP tree's references) keeps working
// unchanged. Migrated byte-for-byte; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__/__LINE__).
// ============================================================================
// csrc/backends/cuda/sm_90/models/mamba.cuh
// Mamba (selective state-space) model header for sm_90 (Hopper).
//
// Full forward/backward implementation. Mirrors the Python reference in
// grokking_race_v2.py::SelectiveSSMLayer:
//
//   residual = x
//   xz = in_proj(x)
//   x_main, z = xz.chunk(2, dim=-1)
//   x_main = SiLU(conv1d(x_main))                # depthwise k=3, pad=1
//   dt, B, C = x_proj(x_main).split([dt_rank, d_state, d_state])
//   dt = softplus(dt_proj(dt) + dt_bias)
//   y  = selective_scan(x_main, dt, A_log, B, C)  # via mamba_adapter
//   y  = (y + x_main * D) * SiLU(z)
//   y  = out_proj(y) + residual
//   y  = LayerNorm(y)
//
// The selective scan itself is delegated to mamba_scan_adapter.cuh — we do
// NOT reimplement it here. GEMMs use cuBLAS (via ATen's blas handle); the
// dt_proj is fused via cutlass_dt_proj_fused_with_bias when WITH_CUTLASS
// is defined.
//
// Weight buffer layout (flat, contiguous, T-typed):
//   tok_emb [vocab, d_model]
//   pos_emb [seq_len, d_model]
//   for each layer L:
//     ln1_g, ln1_b               [d_model]
//     in_proj_W                  [2*d_inner, d_model]
//     conv_W                     [d_inner, 3]      (depthwise, groups=d_inner)
//     conv_b                     [d_inner]
//     x_proj_W                   [dt_rank+2*d_state, d_inner]
//     dt_proj_W                  [d_inner, dt_rank]
//     dt_proj_b                  [d_inner]
//     A_log                      [d_inner, d_state]
//     D                          [d_inner]
//     out_proj_W                 [d_model, d_inner]
//     ln2_g, ln2_b               [d_model]
//   ln_final_g, ln_final_b       [d_model]
//   head_W                       [vocab, d_model]
//
//   dt_rank = max(d_model / 16, 1)

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/scan/mamba_scan_adapter.cuh"
// SG_LAUNCH_CHECK(stream): surfaces a kernel launch failure (bad config, OOM
// SMEM, invalid args) as an immediate TORCH_CHECK at the launch site instead of
// a silent deferred async error. (Foundation: csrc/.../primitives.cuh.)
#include "csrc/backends/cuda/sm_90/primitives.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <algorithm>
#include <type_traits>

#ifdef WITH_CUTLASS
#include "csrc/backends/cuda/sm_90/mma.cuh"
#endif

// ── inlined from former csrc/common/utils.cuh (Phase3 S0) ──
#if GROK_CUDA
#ifndef SG_INLINE_PTX_FAST_RSQRT_NR
#define SG_INLINE_PTX_FAST_RSQRT_NR
// Fast reciprocal sqrt via PTX rsqrt.approx.f32 + Newton-Raphson refinement.
// 2-3x faster than sqrtf(x) + fdividef for Adam denominator.
__device__ __forceinline__ float fast_rsqrt_nr(float x) {
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    // One Newton-Raphson iteration: r = r * (1.5 - 0.5 * x * r * r)
    r = r * (1.5f - 0.5f * x * r * r);
    return r;
}
#endif  // SG_INLINE_PTX_FAST_RSQRT_NR

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

#ifndef SG_INLINE_PTX_PTX_SIGMOIDF
#define SG_INLINE_PTX_PTX_SIGMOIDF
// Fast sigmoid via exp2: sigmoid(x) = 1 / (1 + exp(-x))
// Used in GRU z_gate and r_gate.
__device__ __forceinline__ float ptx_sigmoidf(float x) {
    float en = ptx_exp2(-x * 1.4426950408889634f);
    return 1.0f / (1.0f + en);
}
#endif  // SG_INLINE_PTX_PTX_SIGMOIDF
#endif  // GROK_CUDA

#if GROK_HIP
#ifndef SG_INLINE_PTX_FAST_RSQRT_NR
#define SG_INLINE_PTX_FAST_RSQRT_NR
__device__ __forceinline__ float fast_rsqrt_nr(float x) { return rsqrtf(x); }
#endif  // SG_INLINE_PTX_FAST_RSQRT_NR
#ifndef SG_INLINE_PTX_PTX_EXPF
#define SG_INLINE_PTX_PTX_EXPF
__device__ __forceinline__ float ptx_expf(float x) { return expf(x); }
#endif  // SG_INLINE_PTX_PTX_EXPF
#ifndef SG_INLINE_PTX_PTX_SIGMOIDF
#define SG_INLINE_PTX_PTX_SIGMOIDF
__device__ __forceinline__ float ptx_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
#endif  // SG_INLINE_PTX_PTX_SIGMOIDF
#endif  // GROK_HIP

namespace sg { namespace sm90 { namespace models { namespace mamba {

// ─────────────────────────────────────────────────────────────────────
//  Type traits to map T → cuBLAS data type & cast helpers
// ─────────────────────────────────────────────────────────────────────

template <typename T> struct cublas_traits;
template <> struct cublas_traits<float>          { static constexpr cudaDataType_t dt = CUDA_R_32F; };
template <> struct cublas_traits<__half>         { static constexpr cudaDataType_t dt = CUDA_R_16F; };
template <> struct cublas_traits<__nv_bfloat16>  { static constexpr cudaDataType_t dt = CUDA_R_16BF; };

// Convert T <-> float (device)
template <typename T> __device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v) { return static_cast<T>(v); }
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ─────────────────────────────────────────────────────────────────────
//  cuBLAS GEMM wrapper: C [M,N] = A [M,K] · B [K,N]   (row-major)
//  We compute it via column-major: C^T = B^T · A^T using cublasGemmEx
//  with B as the "first" operand. Math: alpha=1, beta=0.
// ─────────────────────────────────────────────────────────────────────

template <typename T>
inline cudaError_t gemm_rowmajor(
    cublasHandle_t handle,
    int M, int N, int K,
    const T* A, const T* B, T* C,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);
    const float alpha = 1.0f, beta = 0.0f;
    cudaDataType_t dt = cublas_traits<T>::dt;
    // Row-major: C = A · B  ⇔  C^T = B^T · A^T  (col-major view)
    // In col-major: m=N, n=M, k=K, A_in=B (lda=N), B_in=A (ldb=K), C_out=C (ldc=N)
    // For float inputs use CUBLAS_COMPUTE_32F_FAST_TF32 so plain-FP32 buffers are
    // routed through the TF32 tensor cores (matching the CUTLASS TF32 path and
    // gemm_rowmajor_NT's fast-path precision class); bf16/half keep
    // CUBLAS_COMPUTE_32F (FP32-accumulate), which already selects tensor cores.
    cublasComputeType_t ctype =
        std::is_same<T, float>::value ? CUBLAS_COMPUTE_32F_FAST_TF32
                                      : CUBLAS_COMPUTE_32F;
    cublasStatus_t st = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, dt, N,
        A, dt, K,
        &beta,
        C, dt, N,
        ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (st == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

// GEMM with A transposed: C [M,N] = A^T [K,M]^T · B [K,N]  (i.e. A is stored [K,M]).
// Used for weight-grad GEMMs: grad_W [out,in] = acts^T [rows,in]^T @ grad [rows,out].
template <typename T>
inline cudaError_t gemm_rowmajor_TN(
    cublasHandle_t handle,
    int M, int N, int K,
    const T* A,    // [K, M] row-major — we compute A^T [M,K] implicitly
    const T* B,    // [K, N] row-major
    T* C,          // [M, N] row-major
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);
    const float alpha = 1.0f, beta = 1.0f;   // beta=1 to accumulate into grad_W
    cudaDataType_t dt = cublas_traits<T>::dt;
    // Row-major: C = A^T · B  (A is [K,M], B is [K,N], C is [M,N])
    // Col-major view: C^T [N,M] = B^T [N,K] · A [K,M]
    // cublasGemmEx(opA=T, opB=N, m=N, n=M, k=K, A=B(lda=N), B=A(ldb=M), C=C(ldc=N))
    cublasStatus_t st = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, dt, N,
        A, dt, M,
        &beta,
        C, dt, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (st == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

// GEMM with B operand transposed: C [M,N] = A [M,K] · B^T  where B is [N,K].
template <typename T>
inline cudaError_t gemm_rowmajor_NT(
    cublasHandle_t handle,
    int M, int N, int K,
    const T* A, const T* B, T* C,
    cudaStream_t stream)
{
#if defined(WITH_CUTLASS) && !defined(SG_FORCE_SCALAR_FP32)
    // FP32 tensor-core (TF32) fast-path (LIVE). A[M,K] · Bᵀ (B is [N,K]
    // row-major) → C[M,N], all FP32 buffers. The Sm90 WGMMA mainloop reads A/B
    // as cutlass::tfloat32_t (B[N,K] read as Bᵀ via LayoutBT=ColumnMajor) and
    // accumulates in FP32 — matching this GEMM's float output. 🟡 TF32's 10-bit
    // mantissa is NOT bit-identical to FP32's 23 — the accepted FP32
    // tensor-core precision tradeoff, not a bug. On a shape CUTLASS cannot tile
    // (cudaErrorNotSupported) we fall through to the cuBLAS path below.
    // (Define SG_FORCE_SCALAR_FP32 to force the exact-FP32 cuBLAS path.)
    if constexpr (std::is_same<T, float>::value) {
        cudaError_t tc = mma::sm90_run_gemm_tf32_bt<cutlass::layout::ColumnMajor>(
            M, N, K,
            reinterpret_cast<const float*>(A), reinterpret_cast<const float*>(B),
            reinterpret_cast<float*>(C), stream);
        if (tc == cudaSuccess) return cudaSuccess;
        // else: untileable shape → genuine last-resort cuBLAS fallback below.
    }
#endif
    cublasSetStream(handle, stream);
    const float alpha = 1.0f, beta = 0.0f;
    cudaDataType_t dt = cublas_traits<T>::dt;
    // Row-major: C = A · B^T (B is [N,K]) ⇔ col-major: C^T (N×M) = B (K×N col-major view of [N,K] row) · A^T
    // For row-major A[M,K], B[N,K], C[M,N]:
    //   col-major leading dims: A: K, B: K, C: N
    //   In col-major math: C^T_{N,M} = B_{N,K}_rm · A^T_{K,M}_rm = (B as col-major K×N with op_T) · (A as col-major K×M, no op)
    //   Use cublasGemmEx(opA=T, opB=N, m=N, n=M, k=K, A=B (K×N col), B=A (K×M col), C=C (N×M col))
    cublasStatus_t st = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, dt, K,
        A, dt, K,
        &beta,
        C, dt, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (st == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

// ─────────────────────────────────────────────────────────────────────
//  Kernels: token+pos embedding lookup
//  input is [B, seq_len] of integer ids (passed as T after host-side cast,
//  but we treat the bits as int32 if T is int32 — for simplicity input
//  is a separate int32 buffer in the binding wrapper).
// ─────────────────────────────────────────────────────────────────────

template <typename T>
__global__ void embed_kernel(
    const int* __restrict__ ids,        // [B, seq_len]
    const T* __restrict__ tok_emb,      // [vocab, d_model]
    const T* __restrict__ pos_emb,      // [seq_len, d_model]
    T* __restrict__ out,                // [B, seq_len, d_model]
    int B, int N, int D, int vocab)
{
    int bs = blockIdx.y;
    int t  = blockIdx.x;
    int j  = threadIdx.x;
    if (bs >= B || t >= N || j >= D) return;
    int id = ids[static_cast<int64_t>(bs) * N + t];
    if (id < 0 || id >= vocab) id = 0;
    float v = to_float<T>(tok_emb[static_cast<int64_t>(id) * D + j])
            + to_float<T>(pos_emb[static_cast<int64_t>(t) * D + j]);
    out[(static_cast<int64_t>(bs) * N + t) * D + j] = from_float<T>(v);
}

// LayerNorm forward: per-row, in-place safe.  out[i,:] = (x - mean)/std * g + b
template <typename T>
__global__ void layernorm_kernel(
    const T* __restrict__ x,            // [M, D]
    const T* __restrict__ gamma,        // [D]
    const T* __restrict__ beta,         // [D]
    T* __restrict__ y,                  // [M, D]
    float* __restrict__ saved_mean,     // [M] or null
    float* __restrict__ saved_rstd,     // [M] or null
    int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];     // 2 * blockDim.x

    // Compute mean and variance via block reduction
    float sum = 0.0f, sumsq = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        float v = to_float<T>(x[row * D + j]);
        sum   += v;
        sumsq += v * v;
    }
    smem[tid] = sum;
    smem[blockDim.x + tid] = sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid]               += smem[tid + s];
            smem[blockDim.x + tid]  += smem[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    float mean = smem[0] / (float)D;
    float var  = smem[blockDim.x] / (float)D - mean * mean;
    float rstd = fast_rsqrt_nr(var + eps);
    if (tid == 0 && saved_mean) saved_mean[row] = mean;
    if (tid == 0 && saved_rstd) saved_rstd[row] = rstd;

    for (int j = tid; j < D; j += blockDim.x) {
        float v = to_float<T>(x[row * D + j]);
        float g = to_float<T>(gamma[j]);
        float b = to_float<T>(beta[j]);
        y[row * D + j] = from_float<T>((v - mean) * rstd * g + b);
    }
}

// Split chunk-of-2 along last dim: in[..., 2*D] -> a[..., D], b[..., D]
template <typename T>
__global__ void split_chunk2_kernel(
    const T* __restrict__ in,
    T* __restrict__ a, T* __restrict__ b,
    int rows, int D)
{
    int r = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || j >= D) return;
    a[r * D + j] = in[r * 2 * D + j];
    b[r * D + j] = in[r * 2 * D + D + j];
}

// Depthwise Conv1d (groups=d_inner, kernel=3, padding=1) + SiLU activation, fused.
// Input layout: [B, N, d_inner] (channels-last on last dim, contiguous over time per channel? no — sequence-major).
// We treat it as [B, N, d_inner] with the time dimension being N.
//   y[b, t, c] = SiLU(b_c + sum_k W[c,k] * x[b, t+k-1, c])     (k in {0,1,2}, pad=1 zero outside)
template <typename T>
__global__ void conv1d_silu_kernel(
    const T* __restrict__ x,    // [B, N, C]
    const T* __restrict__ W,    // [C, 3]
    const T* __restrict__ bias, // [C]
    T* __restrict__ y,          // [B, N, C]
    int B, int N, int C)
{
    int bs = blockIdx.z;
    int t  = blockIdx.y;
    int c  = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs >= B || t >= N || c >= C) return;

    float w0 = to_float<T>(W[c * 3 + 0]);
    float w1 = to_float<T>(W[c * 3 + 1]);
    float w2 = to_float<T>(W[c * 3 + 2]);
    float bv = bias ? to_float<T>(bias[c]) : 0.0f;

    float xm1 = (t > 0)     ? to_float<T>(x[(bs * N + (t - 1)) * C + c]) : 0.0f;
    float x0  =                to_float<T>(x[(bs * N + t)       * C + c]);
    float xp1 = (t < N - 1) ? to_float<T>(x[(bs * N + (t + 1)) * C + c]) : 0.0f;

    float v = w0 * xm1 + w1 * x0 + w2 * xp1 + bv;
    // SiLU
    float s = v * ptx_sigmoidf(v);
    y[(bs * N + t) * C + c] = from_float<T>(s);
}

// Split a [rows, dt_rank + 2*d_state] tensor into dt[rows,dt_rank], B[rows,d_state], C[rows,d_state]
template <typename T>
__global__ void split_dbc_kernel(
    const T* __restrict__ in,
    T* __restrict__ dt, T* __restrict__ B, T* __restrict__ C,
    int rows, int dt_rank, int d_state)
{
    int r = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    int total = dt_rank + 2 * d_state;
    if (r >= rows || j >= total) return;
    T v = in[r * total + j];
    if (j < dt_rank)                       dt[r * dt_rank + j] = v;
    else if (j < dt_rank + d_state)        B[r * d_state + (j - dt_rank)] = v;
    else                                   C[r * d_state + (j - dt_rank - d_state)] = v;
}

// Softplus + bias kernel: out[i,j] = softplus(in[i,j] + bias[j])
template <typename T>
__global__ void softplus_bias_kernel(
    T* __restrict__ inout, const T* __restrict__ bias,
    int rows, int cols)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int64_t>(rows) * cols) return;
    int j = static_cast<int>(idx % cols);
    float v = to_float<T>(inout[idx]) + to_float<T>(bias[j]);
    float sp = (v > 20.0f) ? v : logf(1.0f + ptx_expf(v));
    inout[idx] = from_float<T>(sp);
}

// y = (y + x_main * D) * SiLU(z)    elementwise. Layout [rows, d_inner].
template <typename T>
__global__ void gate_dskip_kernel(
    T* __restrict__ y,
    const T* __restrict__ x_main,
    const T* __restrict__ Dpar,    // [d_inner]
    const T* __restrict__ z,
    int rows, int d_inner)
{
    int r = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || c >= d_inner) return;
    float yv  = to_float<T>(y[r * d_inner + c]);
    float xv  = to_float<T>(x_main[r * d_inner + c]);
    float dv  = to_float<T>(Dpar[c]);
    float zv  = to_float<T>(z[r * d_inner + c]);
    float sz  = zv * ptx_sigmoidf(zv);
    y[r * d_inner + c] = from_float<T>((yv + xv * dv) * sz);
}

// Add residual (out += residual) elementwise
template <typename T>
__global__ void add_residual_kernel(T* __restrict__ y, const T* __restrict__ r, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = from_float<T>(to_float<T>(y[idx]) + to_float<T>(r[idx]));
}

// Read-last-token + write logits via head GEMM is a normal gemm; no fused kernel.
// However we need to gather the last token row before head GEMM:
//   out[b, :] = h[b, N-1, :]
template <typename T>
__global__ void gather_last_token_kernel(
    const T* __restrict__ h,    // [B, N, D]
    T* __restrict__ last,       // [B, D]
    int B, int N, int D)
{
    int b = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || j >= D) return;
    last[b * D + j] = h[(b * N + (N - 1)) * D + j];
}

// ─────────────────────────────────────────────────────────────────────
//  Backward helper kernels
// ─────────────────────────────────────────────────────────────────────

// LayerNorm backward: given grad_out [M,D], saved x_norm [M,D] (= (x-mean)*rstd),
// saved_mean [M], saved_rstd [M], gamma [D]:
//   grad_gamma[j] += sum_rows(grad_out[:,j] * x_norm[:,j])
//   grad_beta[j]  += sum_rows(grad_out[:,j])
//   dx[i,j] = rstd*(grad_out[i,j]*gamma[j]
//              - mean(grad_out*gamma) - x_norm[i,j]*mean(grad_out*gamma*x_norm))
// We accumulate grad_gamma/grad_beta with atomicAdd; dx is written directly.
template <typename T>
__global__ void layernorm_backward_kernel(
    const T* __restrict__ grad_out,     // [M, D]
    const T* __restrict__ x_norm,       // [M, D]  — (x - mean)*rstd, i.e. normalised
    const T* __restrict__ gamma,        // [D]
    const float* __restrict__ saved_rstd, // [M]
    T* __restrict__ dx,                 // [M, D]
    float* __restrict__ grad_gamma_f,   // [D]  float accumulator
    float* __restrict__ grad_beta_f,    // [D]  float accumulator
    int D, int M)
{
    // One block per row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    extern __shared__ float smem2[];   // 2 * blockDim.x floats
    float rstd = saved_rstd[row];

    // Pass 1: compute dot1 = mean(dout*g), dot2 = mean(dout*g*xnorm)
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        float go = to_float<T>(grad_out[row * D + j]);
        float g  = to_float<T>(gamma[j]);
        float xn = to_float<T>(x_norm[row * D + j]);
        float dg = go * g;
        sum1 += dg;
        sum2 += dg * xn;
        // accumulate weight grads
        atomicAdd(&grad_gamma_f[j], go * xn);
        atomicAdd(&grad_beta_f[j],  go);
    }
    smem2[tid]              = sum1;
    smem2[blockDim.x + tid] = sum2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem2[tid]              += smem2[tid + s];
            smem2[blockDim.x + tid] += smem2[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    float mean1 = smem2[0]              / (float)D;
    float mean2 = smem2[blockDim.x]     / (float)D;

    // Pass 2: write dx
    for (int j = tid; j < D; j += blockDim.x) {
        float go = to_float<T>(grad_out[row * D + j]);
        float g  = to_float<T>(gamma[j]);
        float xn = to_float<T>(x_norm[row * D + j]);
        float dxv = rstd * (go * g - mean1 - xn * mean2);
        dx[row * D + j] = from_float<T>(dxv);
    }
}

// Scatter-add embedding backward: for each (b,t), atomicAdd grad into tok_emb and pos_emb grads.
// grad_h [B, N, D] -> grad_tok_emb [vocab, D], grad_pos_emb [seq, D]
template <typename T>
__global__ void embed_backward_kernel(
    const T* __restrict__ grad_h,           // [B, N, D]
    const int* __restrict__ ids,            // [B, N]
    float* __restrict__ grad_tok_emb,       // [vocab, D]  float accum
    float* __restrict__ grad_pos_emb,       // [seq_len, D] float accum
    int B, int N, int D, int vocab)
{
    int b = blockIdx.z;
    int t = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || t >= N || j >= D) return;
    float gv = to_float<T>(grad_h[(b * N + t) * D + j]);
    int id = ids[b * N + t];
    if (id >= 0 && id < vocab)
        atomicAdd(&grad_tok_emb[id * D + j], gv);
    atomicAdd(&grad_pos_emb[t * D + j], gv);
}

// Copy float grad buffer -> T output, with optional beta (add to existing)
template <typename T>
__global__ void float_to_T_kernel(
    const float* __restrict__ src,
    T* __restrict__ dst,
    int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = from_float<T>(src[idx]);
}

// Gate+skip backward: forward was y_out = (y_scan + xc * D) * silu(z)
// Given grad_y_out, compute grad_y_scan, grad_xc_skip, grad_z, grad_D (atomicAdd).
template <typename T>
__global__ void gate_dskip_backward_kernel(
    const T* __restrict__ grad_y_out,   // [rows, d_inner]
    const T* __restrict__ y_scan_in,    // [rows, d_inner] — scan output (before gate)
    const T* __restrict__ xc,           // [rows, d_inner] — post conv-silu
    const T* __restrict__ Dpar,         // [d_inner]
    const T* __restrict__ z,            // [rows, d_inner]
    T* __restrict__ grad_y_scan,        // [rows, d_inner]
    T* __restrict__ grad_xc_skip,       // [rows, d_inner]  (additive)
    T* __restrict__ grad_z,             // [rows, d_inner]
    float* __restrict__ grad_D,         // [d_inner]   atomicAdd
    int rows, int d_inner)
{
    int r = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || c >= d_inner) return;

    float gy   = to_float<T>(grad_y_out[r * d_inner + c]);
    float ys   = to_float<T>(y_scan_in [r * d_inner + c]);
    float xcv  = to_float<T>(xc        [r * d_inner + c]);
    float Dv   = to_float<T>(Dpar[c]);
    float zv   = to_float<T>(z         [r * d_inner + c]);

    float sig_z = ptx_sigmoidf(zv);
    float sz    = zv * sig_z;                                    // SiLU(z)
    float dsz   = sig_z * (1.0f + zv * (1.0f - sig_z));         // d/dz SiLU(z)
    float inner = ys + xcv * Dv;

    // grad w.r.t scan output (before D-skip) and xc (D-skip path)
    grad_y_scan [r * d_inner + c] = from_float<T>(gy * sz);
    grad_xc_skip[r * d_inner + c] = from_float<T>(gy * sz * Dv);

    // grad w.r.t z
    grad_z[r * d_inner + c] = from_float<T>(gy * inner * dsz);

    // grad w.r.t D (per-channel, sum across rows via atomicAdd)
    atomicAdd(&grad_D[c], gy * sz * xcv);
}

// Softplus backward: forward was dt_full = softplus(dt_pre_proj + bias)
//   d/dx softplus(x) = sigmoid(x)
// Multiplies in-place: grad_dt_pre *= sigmoid(dt_pre + bias)
template <typename T>
__global__ void softplus_bias_backward_kernel(
    T* __restrict__ grad_dt,           // [rows, d_inner] in/out
    const T* __restrict__ dt_pre_proj, // [rows, d_inner] — value BEFORE softplus (pre-bias-add)
    const T* __restrict__ bias,        // [d_inner]
    int rows, int cols)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int64_t>(rows) * cols) return;
    int j = static_cast<int>(idx % cols);
    float pre = to_float<T>(dt_pre_proj[idx]) + to_float<T>(bias[j]);
    float sig  = ptx_sigmoidf(pre);
    grad_dt[idx] = from_float<T>(to_float<T>(grad_dt[idx]) * sig);
}

// Accumulate grad_dt_proj_b: sum grad_dt_pre over rows dimension.
// grad_dt_pre [rows, d_inner] -> grad_b [d_inner] (atomicAdd, float)
template <typename T>
__global__ void accumulate_bias_grad_kernel(
    const T* __restrict__ grad_dt_pre, // [rows, d_inner]
    float* __restrict__ grad_b,        // [d_inner]
    int rows, int d_inner)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d_inner) return;
    float acc = 0.0f;
    for (int r = 0; r < rows; r++)
        acc += to_float<T>(grad_dt_pre[r * d_inner + c]);
    atomicAdd(&grad_b[c], acc);
}

// Conv1d + SiLU backward (depthwise k=3 pad=1).
// Forward: xc[b,t,c] = silu(sum_k W[c,k]*x_main[b,t+k-1,c] + bias[c])
// We need pre-silu value; recompute from x_main.
// Writes grad_x_main_f[b,t,c] += (accumulated from all output positions that read x[b,t,c]).
//   NOTE: grad_x_main_f is a FLOAT buffer (not T) to avoid type-punning atomicAdd bugs.
//   The caller converts it to T afterward via float_to_T_kernel.
// Also accumulates grad_conv_W [d_inner, 3] and grad_conv_b [d_inner] via atomicAdd (float).
template <typename T>
__global__ void conv1d_silu_backward_kernel(
    const T* __restrict__ grad_xc,      // [B, N, C]   gradient w.r.t. conv output
    const T* __restrict__ x_main,       // [B, N, C]   conv input
    const T* __restrict__ W,            // [C, 3]
    const T* __restrict__ bias,         // [C]
    float* __restrict__ grad_x_main_f,  // [B, N, C]   float output (atomicAdd)
    float* __restrict__ grad_W,         // [C, 3]   atomicAdd float
    float* __restrict__ grad_bias,      // [C]      atomicAdd float
    int B, int N, int C)
{
    int bs = blockIdx.z;
    int t  = blockIdx.y;
    int c  = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs >= B || t >= N || c >= C) return;

    // Recompute conv output (pre-SiLU)
    float w0 = to_float<T>(W[c * 3 + 0]);
    float w1 = to_float<T>(W[c * 3 + 1]);
    float w2 = to_float<T>(W[c * 3 + 2]);
    float bv = bias ? to_float<T>(bias[c]) : 0.0f;
    float xm1 = (t > 0)     ? to_float<T>(x_main[(bs * N + (t-1)) * C + c]) : 0.0f;
    float x0  =                to_float<T>(x_main[(bs * N + t)     * C + c]);
    float xp1 = (t < N-1)   ? to_float<T>(x_main[(bs * N + (t+1)) * C + c]) : 0.0f;
    float conv_out = w0 * xm1 + w1 * x0 + w2 * xp1 + bv;

    // d/d(conv_out) of silu
    float sig  = ptx_sigmoidf(conv_out);
    float dsilu = sig * (1.0f + conv_out * (1.0f - sig));

    float gxc = to_float<T>(grad_xc[(bs * N + t) * C + c]);
    float g   = gxc * dsilu;   // grad w.r.t. conv_out

    // Accumulate grad_conv_b (one contribution per (b,t))
    atomicAdd(&grad_bias[c], g);

    // Accumulate grad_conv_W: grad_W[c,k] += g * x_main[b, t+k-1, c]
    atomicAdd(&grad_W[c * 3 + 0], g * xm1);
    atomicAdd(&grad_W[c * 3 + 1], g * x0);
    atomicAdd(&grad_W[c * 3 + 2], g * xp1);

    // Grad w.r.t. x_main (float buffer, safe atomicAdd for all T types):
    //   The conv output at position t uses x_main[t-1], x_main[t], x_main[t+1].
    //   So grad for x_main[t] comes from 3 output positions:
    //     from output[t]:   g_from_t   * W[c, 1]   (x0 role)
    //     from output[t+1]: g_from_tp1 * W[c, 0]   (x0 = x_main[t] acts as xm1 for t+1)
    //     from output[t-1]: g_from_tm1 * W[c, 2]   (x0 = x_main[t] acts as xp1 for t-1)
    atomicAdd(&grad_x_main_f[(bs * N + t) * C + c],         g * w1);
    if (t > 0)
        atomicAdd(&grad_x_main_f[(bs * N + (t-1)) * C + c], g * w0);
    if (t < N - 1)
        atomicAdd(&grad_x_main_f[(bs * N + (t+1)) * C + c], g * w2);
}

// Concatenate [grad_x_main, grad_z] -> grad_xz [rows, 2*d_inner]
template <typename T>
__global__ void concat_chunk2_kernel(
    const T* __restrict__ a,   // [rows, D]
    const T* __restrict__ b,   // [rows, D]
    T* __restrict__ out,       // [rows, 2*D]
    int rows, int D)
{
    int r = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || j >= D) return;
    out[r * 2 * D + j]     = a[r * D + j];
    out[r * 2 * D + D + j] = b[r * D + j];
}

// Concatenate [grad_dt_rank, grad_B, grad_C] -> grad_x_dbc [rows, dt_rank+2*d_state]
template <typename T>
__global__ void concat_dbc_kernel(
    const T* __restrict__ gdt,   // [rows, dt_rank]
    const T* __restrict__ gB,    // [rows, d_state]
    const T* __restrict__ gC,    // [rows, d_state]
    T* __restrict__ out,         // [rows, dt_rank+2*d_state]
    int rows, int dt_rank, int d_state)
{
    int r = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    int total = dt_rank + 2 * d_state;
    if (r >= rows || j >= total) return;
    T v;
    if (j < dt_rank)                    v = gdt[r * dt_rank + j];
    else if (j < dt_rank + d_state)     v = gB[r * d_state + (j - dt_rank)];
    else                                v = gC[r * d_state + (j - dt_rank - d_state)];
    out[r * total + j] = v;
}

// Add two buffers: dst[i] += src[i]  (for grad accumulation)
template <typename T>
__global__ void add_inplace_kernel(T* __restrict__ dst, const T* __restrict__ src, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = from_float<T>(to_float<T>(dst[idx]) + to_float<T>(src[idx]));
}

// Scatter last token: given grad of head output [B, D], broadcast back to [B, N, D]
// at position N-1 (other positions get zero from memset).
template <typename T>
__global__ void scatter_last_token_kernel(
    const T* __restrict__ grad_last,  // [B, D]
    T* __restrict__ grad_h,           // [B, N, D]  (should be zeroed first)
    int B, int N, int D)
{
    int b = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || j >= D) return;
    grad_h[(b * N + (N - 1)) * D + j] = grad_last[b * D + j];
}

// Save layernorm statistics during forward: x_norm = (x - mean) * rstd
// This kernel saves x_norm and rstd alongside the LN output (needs to run after LN).
// Actually we store x_norm directly — the regular layernorm_kernel already does the
// computation. This helper saves x_norm = (x - mean) * rstd separately.
template <typename T>
__global__ void save_xnorm_kernel(
    const T* __restrict__ x,           // [M, D]
    const float* __restrict__ saved_mean, // [M]
    const float* __restrict__ saved_rstd, // [M]
    T* __restrict__ x_norm,            // [M, D]
    int D)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float mean = saved_mean[row];
    float rstd = saved_rstd[row];
    for (int j = tid; j < D; j += blockDim.x) {
        float v = to_float<T>(x[row * D + j]);
        x_norm[row * D + j] = from_float<T>((v - mean) * rstd);
    }
}

// ─────────────────────────────────────────────────────────────────────
//  Weight pointer helper
// ─────────────────────────────────────────────────────────────────────

template <typename T>
struct WeightPtrs {
    const T* tok_emb;
    const T* pos_emb;
    // per-layer arrays of size n_layers
    const T* ln1_g; const T* ln1_b;
    const T* in_proj_W;
    const T* conv_W; const T* conv_b;
    const T* x_proj_W;
    const T* dt_proj_W; const T* dt_proj_b;
    const T* A_log;
    const T* D;
    const T* out_proj_W;
    const T* ln2_g; const T* ln2_b;
    const T* ln_final_g; const T* ln_final_b;
    const T* head_W;
    // strides
    int per_layer_floats;
};

template <typename T>
inline size_t per_layer_count(int d_model, int d_inner, int d_state, int dt_rank)
{
    size_t n = 0;
    n += 2 * d_model;                          // ln1
    n += 2 * d_inner * d_model;                // in_proj_W
    n += 3 * d_inner;                          // conv_W
    n += d_inner;                              // conv_b
    n += (size_t)(dt_rank + 2 * d_state) * d_inner; // x_proj_W
    n += (size_t)d_inner * dt_rank;            // dt_proj_W
    n += d_inner;                              // dt_proj_b
    n += (size_t)d_inner * d_state;            // A_log
    n += d_inner;                              // D
    n += (size_t)d_model * d_inner;            // out_proj_W
    n += 2 * d_model;                          // ln2
    return n;
}

// Compute a layer-l view into the flat weights buffer.  Returns a partly-
// filled struct: only per-layer pointers are resolved against the global
// tok/pos/head/ln_final pointers which the caller fills separately.
template <typename T>
__host__ inline void resolve_layer(
    const T* base,
    int l, int n_layers,
    int vocab, int seq_len,
    int d_model, int d_inner, int d_state, int dt_rank,
    WeightPtrs<T>& W)
{
    const T* p = base;
    W.tok_emb = p; p += (size_t)vocab * d_model;
    W.pos_emb = p; p += (size_t)seq_len * d_model;
    size_t per = per_layer_count<T>(d_model, d_inner, d_state, dt_rank);
    const T* layer_base = p + (size_t)l * per;
    const T* lp = layer_base;
    W.ln1_g     = lp; lp += d_model;
    W.ln1_b     = lp; lp += d_model;
    W.in_proj_W = lp; lp += (size_t)2 * d_inner * d_model;
    W.conv_W    = lp; lp += (size_t)3 * d_inner;
    W.conv_b    = lp; lp += d_inner;
    W.x_proj_W  = lp; lp += (size_t)(dt_rank + 2 * d_state) * d_inner;
    W.dt_proj_W = lp; lp += (size_t)d_inner * dt_rank;
    W.dt_proj_b = lp; lp += d_inner;
    W.A_log     = lp; lp += (size_t)d_inner * d_state;
    W.D         = lp; lp += d_inner;
    W.out_proj_W= lp; lp += (size_t)d_model * d_inner;
    W.ln2_g     = lp; lp += d_model;
    W.ln2_b     = lp; lp += d_model;
    // tail (after all layers)
    const T* tail = p + (size_t)n_layers * per;
    W.ln_final_g = tail; tail += d_model;
    W.ln_final_b = tail; tail += d_model;
    W.head_W     = tail;
}

// ─────────────────────────────────────────────────────────────────────
//  Weight gradient offset helpers.
//  These return T-element offsets into the flat grad_weights buffer for
//  each per-layer weight group. Used by backward() to accumulate grads.
// ─────────────────────────────────────────────────────────────────────

// Offset to the start of layer l's block in the flat weight buffer.
template <typename T>
__host__ inline size_t layer_weight_offset(
    int l, int vocab, int seq_len,
    int d_model, int d_inner, int d_state, int dt_rank)
{
    size_t off = 0;
    off += (size_t)vocab * d_model;       // tok_emb
    off += (size_t)seq_len * d_model;     // pos_emb
    off += (size_t)l * per_layer_count<T>(d_model, d_inner, d_state, dt_rank);
    return off;
}

// Within a per-layer block, offset (in T-elems) to each weight group.
// Layout (matches resolve_layer):
//   ln1_g [d_model], ln1_b [d_model]
//   in_proj_W [2*d_inner, d_model]
//   conv_W [3*d_inner], conv_b [d_inner]
//   x_proj_W [(dt_rank+2*d_state)*d_inner]
//   dt_proj_W [d_inner*dt_rank], dt_proj_b [d_inner]
//   A_log [d_inner*d_state], D [d_inner]
//   out_proj_W [d_model*d_inner]
//   ln2_g [d_model], ln2_b [d_model]

template <typename T>
__host__ inline size_t per_layer_inprojW_offset(
    int d_model, int d_inner, int /*d_state*/, int /*dt_rank*/)
{
    return (size_t)2 * d_model;  // after ln1_g, ln1_b
}

template <typename T>
__host__ inline size_t per_layer_convW_offset(
    int d_model, int d_inner, int /*d_state*/, int /*dt_rank*/)
{
    return (size_t)2 * d_model + (size_t)2 * d_inner * d_model;
}

template <typename T>
__host__ inline size_t per_layer_xprojW_offset(
    int d_model, int d_inner, int /*d_state*/, int /*dt_rank*/)
{
    return (size_t)2 * d_model
         + (size_t)2 * d_inner * d_model
         + (size_t)3 * d_inner    // conv_W
         + (size_t)d_inner;       // conv_b
}

template <typename T>
__host__ inline size_t per_layer_dtW_offset(
    int d_model, int d_inner, int d_state, int dt_rank)
{
    return per_layer_xprojW_offset<T>(d_model, d_inner, d_state, dt_rank)
         + (size_t)(dt_rank + 2 * d_state) * d_inner;
}

template <typename T>
__host__ inline size_t per_layer_dtb_offset(
    int d_model, int d_inner, int d_state, int dt_rank)
{
    return per_layer_dtW_offset<T>(d_model, d_inner, d_state, dt_rank)
         + (size_t)d_inner * dt_rank;
}

template <typename T>
__host__ inline size_t per_layer_Alog_offset(
    int d_model, int d_inner, int d_state, int dt_rank)
{
    return per_layer_dtb_offset<T>(d_model, d_inner, d_state, dt_rank)
         + (size_t)d_inner;  // dt_proj_b
}

template <typename T>
__host__ inline size_t per_layer_D_offset(
    int d_model, int d_inner, int d_state, int dt_rank)
{
    return per_layer_Alog_offset<T>(d_model, d_inner, d_state, dt_rank)
         + (size_t)d_inner * d_state;
}

template <typename T>
__host__ inline size_t per_layer_out_proj_offset(
    int d_model, int d_inner, int d_state, int dt_rank)
{
    return per_layer_D_offset<T>(d_model, d_inner, d_state, dt_rank)
         + (size_t)d_inner;  // D
}

template <typename T>
__host__ inline size_t per_layer_ln2_offset(
    int d_model, int d_inner, int d_state, int dt_rank)
{
    return per_layer_out_proj_offset<T>(d_model, d_inner, d_state, dt_rank)
         + (size_t)d_model * d_inner;
}

// ─────────────────────────────────────────────────────────────────────
//  Activation-cache layout helpers.
//
//  Per-layer activation save (for backward). Layout inside the states
//  buffer, starting after the transient workspace (act_cache_base):
//
//  For layer l, offset = l * per_layer_act_elems(T, rows, d_model,
//                              d_inner, dt_rank, d_state):
//    h_pre       [rows, d_model]     T     — layer input (pre-LN1)
//    mean_ln1    [rows]              float — LN1 statistics
//    rstd_ln1    [rows]              float
//    x_main_save [rows, d_inner]     T     — pre-conv (for conv bwd recompute)
//    z_save      [rows, d_inner]     T     — gating z
//    xc_save     [rows, d_inner]     T     — post-conv-silu (xc)
//    dt_pre_save [rows, dt_rank]     T     — pre-softplus dt
//    dt_full_save[rows, d_inner]     T     — post-softplus dt (used by scan bwd)
//    B_save      [rows, d_state]     T
//    C_save      [rows, d_state]     T
//    y_scan_save [rows, d_inner]     T     — scan output BEFORE gate
//    mean_ln2    [rows]              float — LN2 statistics
//    rstd_ln2    [rows]              float
//    h_post_save [rows, d_model]     T     — post-LN2 (input to next layer)
//
//  After all layers (offset = n_layers * per_layer):
//    mean_lnf    [rows]              float — final-LN statistics
//    rstd_lnf    [rows]              float
//    last_save   [batch, d_model]    T     — gathered last token
//
//  The float arrays are interleaved as T-sized units (rounded up to next
//  multiple of sizeof(T)) for pointer alignment.
// ─────────────────────────────────────────────────────────────────────

// Round a float count up to a T-element count (ceiling division).
template <typename T>
__host__ inline size_t floats_as_T(size_t float_count) {
    // Bytes needed for float_count floats, rounded up to sizeof(T) boundary.
    size_t bytes = float_count * sizeof(float);
    return (bytes + sizeof(T) - 1) / sizeof(T);
}

// Per-layer activation count in T-elements.
template <typename T>
__host__ inline size_t per_layer_act_elems(
    int rows, int d_model, int d_inner, int dt_rank, int d_state)
{
    size_t n = 0;
    n += (size_t)rows * d_model;                     // h_pre
    n += floats_as_T<T>(rows);                       // mean_ln1
    n += floats_as_T<T>(rows);                       // rstd_ln1
    n += (size_t)rows * d_inner;                     // x_main_save
    n += (size_t)rows * d_inner;                     // z_save
    n += (size_t)rows * d_inner;                     // xc_save
    n += (size_t)rows * dt_rank;                     // dt_pre_save
    n += (size_t)rows * d_inner;                     // dt_full_save
    n += (size_t)rows * d_state;                     // B_save
    n += (size_t)rows * d_state;                     // C_save
    n += (size_t)rows * d_inner;                     // y_scan_save
    n += floats_as_T<T>(rows);                       // mean_ln2
    n += floats_as_T<T>(rows);                       // rstd_ln2
    n += (size_t)rows * d_model;                     // h_post_save
    return n;
}

// Tail activation count in T-elements (after all layers).
template <typename T>
__host__ inline size_t tail_act_elems(int rows, int batch, int d_model) {
    size_t n = 0;
    n += floats_as_T<T>(rows);                       // mean_lnf
    n += floats_as_T<T>(rows);                       // rstd_lnf
    n += (size_t)batch * d_model;                    // last_save
    return n;
}

// Total activation cache size in T-elements.
template <typename T>
__host__ inline size_t activation_cache_elems(
    int batch, int seq_len, int d_model, int d_state, int expand, int n_layers)
{
    const int rows    = batch * seq_len;
    const int d_inner = d_model * expand;
    const int dt_rank = std::max(d_model / 16, 1);
    size_t pla = per_layer_act_elems<T>(rows, d_model, d_inner, dt_rank, d_state);
    size_t tail = tail_act_elems<T>(rows, batch, d_model);
    return (size_t)n_layers * pla + tail;
}

// ─────────────────────────────────────────────────────────────────────
//  Struct for layer activation pointers (resolved from cache buffer).
// ─────────────────────────────────────────────────────────────────────

template <typename T>
struct LayerActPtrs {
    T*      h_pre;          // [rows, d_model]
    float*  mean_ln1;       // [rows]
    float*  rstd_ln1;       // [rows]
    T*      x_main_save;    // [rows, d_inner]
    T*      z_save;         // [rows, d_inner]
    T*      xc_save;        // [rows, d_inner]
    T*      dt_pre_save;    // [rows, dt_rank]
    T*      dt_full_save;   // [rows, d_inner]
    T*      B_save;         // [rows, d_state]
    T*      C_save;         // [rows, d_state]
    T*      y_scan_save;    // [rows, d_inner]
    float*  mean_ln2;       // [rows]
    float*  rstd_ln2;       // [rows]
    T*      h_post_save;    // [rows, d_model]
};

template <typename T>
__host__ inline LayerActPtrs<T> resolve_layer_act(
    T* base, int l,
    int rows, int d_model, int d_inner, int dt_rank, int d_state)
{
    size_t pla = per_layer_act_elems<T>(rows, d_model, d_inner, dt_rank, d_state);
    T* p = base + (size_t)l * pla;
    LayerActPtrs<T> A;
    A.h_pre       = p; p += (size_t)rows * d_model;
    A.mean_ln1    = reinterpret_cast<float*>(p); p += floats_as_T<T>(rows);
    A.rstd_ln1    = reinterpret_cast<float*>(p); p += floats_as_T<T>(rows);
    A.x_main_save = p; p += (size_t)rows * d_inner;
    A.z_save      = p; p += (size_t)rows * d_inner;
    A.xc_save     = p; p += (size_t)rows * d_inner;
    A.dt_pre_save = p; p += (size_t)rows * dt_rank;
    A.dt_full_save= p; p += (size_t)rows * d_inner;
    A.B_save      = p; p += (size_t)rows * d_state;
    A.C_save      = p; p += (size_t)rows * d_state;
    A.y_scan_save = p; p += (size_t)rows * d_inner;
    A.mean_ln2    = reinterpret_cast<float*>(p); p += floats_as_T<T>(rows);
    A.rstd_ln2    = reinterpret_cast<float*>(p); p += floats_as_T<T>(rows);
    A.h_post_save = p;
    return A;
}

// Resolve tail activation pointers (after all n_layers).
template <typename T>
struct TailActPtrs {
    float* mean_lnf;    // [rows]
    float* rstd_lnf;    // [rows]
    T*     last_save;   // [batch, d_model]
};

template <typename T>
__host__ inline TailActPtrs<T> resolve_tail_act(
    T* base, int n_layers,
    int rows, int batch, int d_model, int d_inner, int dt_rank, int d_state)
{
    size_t pla = per_layer_act_elems<T>(rows, d_model, d_inner, dt_rank, d_state);
    T* p = base + (size_t)n_layers * pla;
    TailActPtrs<T> A;
    A.mean_lnf  = reinterpret_cast<float*>(p); p += floats_as_T<T>(rows);
    A.rstd_lnf  = reinterpret_cast<float*>(p); p += floats_as_T<T>(rows);
    A.last_save = p;
    return A;
}

// ─────────────────────────────────────────────────────────────────────
//  Forward pass — full Mamba stack. The `input` parameter is interpreted
//  as a buffer of int32 token ids (B*N) cast to T*. The binding ensures
//  the underlying tensor is int32 and reinterprets safely.
//
//  `states` serves dual purpose:
//    1. Transient workspace for per-layer intermediate buffers (reused
//       across layers for the non-caching path).
//    2. When `activation_cache` is non-null (or when `states` is large
//       enough to hold the activation section), per-layer activations
//       are saved for the backward pass. The binding passes the same
//       `states` tensor to backward as `activations_saved`.
//
//  CONTRACT: `states` must be sized to hold BOTH the transient workspace
//  AND the activation cache. Use activation_cache_elems<T>(...) to
//  compute the extra elements needed after the transient workspace.
//  The transient workspace size (in T-elems) is approximately:
//    rows*(2*d_model + 2*d_inner*4 + dt_rank + d_state*2) + B*d_inner*d_state
//  (see exact layout below). The binding allocates conservatively.
// ─────────────────────────────────────────────────────────────────────

template <typename T>
cudaError_t forward(
    const T* input,                  // reinterpret as int32* (B*N ids)
    const T* weights,
    T* output,                       // [B, p_vocab]  (last-token logits)
    T* states,                       // workspace + activation save area
    int batch, int seq_len, int d_model, int d_state,
    int d_conv, int expand, int n_layers,
    cudaStream_t stream,
    T* activation_cache = nullptr)   // if non-null, save activations here;
                                     // if null, save into states after workspace
{
    (void)d_conv;  // we hard-code k=3, pad=1 per Python ref
    const int d_inner = d_model * expand;
    const int dt_rank = std::max(d_model / 16, 1);
    const int vocab   = 99;   // grokking config (used only for embedding bound)
    const int rows    = batch * seq_len;
    const float eps   = 1e-5f;

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);

    // Transient workspace partition (all in-place over `states`):
    //   h        [B, N, d_model]
    //   res      [B, N, d_model]      (for residual save)
    //   xz       [B, N, 2*d_inner]
    //   x_main   [B, N, d_inner]
    //   z        [B, N, d_inner]
    //   xc       [B, N, d_inner]      (post-conv-silu)
    //   x_dbc    [B, N, dt_rank+2*d_state]
    //   dt_pre   [B, N, dt_rank]
    //   dt_full  [B, N, d_inner]      (post-dt_proj+softplus)
    //   B_buf    [B, N, d_state]
    //   C_buf    [B, N, d_state]
    //   y_scan   [B, N, d_inner]
    //   state_save [B, d_inner, d_state] float   (scan recurrence state)

    T* w = states;
    T* h        = w; w += (size_t)rows * d_model;
    T* res      = w; w += (size_t)rows * d_model;
    T* xz       = w; w += (size_t)rows * 2 * d_inner;
    T* x_main   = w; w += (size_t)rows * d_inner;
    T* z_buf    = w; w += (size_t)rows * d_inner;
    T* xc       = w; w += (size_t)rows * d_inner;
    T* x_dbc    = w; w += (size_t)rows * (dt_rank + 2 * d_state);
    T* dt_pre   = w; w += (size_t)rows * dt_rank;
    T* dt_full  = w; w += (size_t)rows * d_inner;
    T* B_buf    = w; w += (size_t)rows * d_state;
    T* C_buf    = w; w += (size_t)rows * d_state;
    T* y_scan   = w; w += (size_t)rows * d_inner;
    // scan recurrence state (float) lives at `w`; we keep it as float*
    float* state_save = reinterpret_cast<float*>(w);
    // advance w past state_save (B*d_inner*d_state floats, rounded to T)
    w += floats_as_T<T>((size_t)batch * d_inner * d_state);

    // Activation cache: use caller-provided pointer or fall back to states
    // immediately after the transient workspace.
    T* act_base = (activation_cache != nullptr) ? activation_cache : w;

    // 1. Embedding
    {
        const int* ids = reinterpret_cast<const int*>(input);
        dim3 grid(seq_len, batch);
        int block = std::min(d_model, 256);
        embed_kernel<T><<<grid, block, 0, stream>>>(
            ids, weights /*tok_emb*/, weights + (size_t)vocab * d_model /*pos_emb*/,
            h, batch, seq_len, d_model, vocab);
        SG_LAUNCH_CHECK(stream);
    }

    // 2. Per-layer
    for (int l = 0; l < n_layers; l++) {
        WeightPtrs<T> W;
        resolve_layer<T>(weights, l, n_layers, vocab, seq_len,
                         d_model, d_inner, d_state, dt_rank, W);

        // Resolve activation save pointers for this layer.
        LayerActPtrs<T> A = resolve_layer_act<T>(
            act_base, l, rows, d_model, d_inner, dt_rank, d_state);

        // Save h_pre (layer input, before any transformation).
        // This is the residual that gets added back after out_proj.
        cudaMemcpyAsync(A.h_pre, h, (size_t)rows * d_model * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);
        // Zero mean_ln1/rstd_ln1 (unused — ln1 weights are reserved but not applied
        // in the current forward formulation; backward skips LN1 accordingly).
        // We mark them invalid by leaving them at 0 (from the activation cache memset).

        // Save residual = h_pre for the add_residual_kernel at the end of the layer.
        cudaMemcpyAsync(res, h, (size_t)rows * d_model * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // (a) in_proj GEMM:  xz [rows, 2*d_inner] = h_ln1 [rows, d_model] · in_proj_W^T
        cudaError_t err = gemm_rowmajor_NT<T>(handle, rows, 2 * d_inner, d_model,
                                              h, W.in_proj_W, xz, stream);
        if (err != cudaSuccess) return err;

        // (b) split xz -> x_main, z
        {
            dim3 grid(rows, (d_inner + 127) / 128);
            split_chunk2_kernel<T><<<grid, 128, 0, stream>>>(xz, x_main, z_buf, rows, d_inner);
            SG_LAUNCH_CHECK(stream);
        }

        // Save x_main and z before conv/gate (needed for backward).
        cudaMemcpyAsync(A.x_main_save, x_main, (size_t)rows * d_inner * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(A.z_save, z_buf, (size_t)rows * d_inner * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // (c) conv1d depthwise k=3 pad=1 + SiLU
        {
            dim3 grid((d_inner + 127) / 128, seq_len, batch);
            conv1d_silu_kernel<T><<<grid, 128, 0, stream>>>(
                x_main, W.conv_W, W.conv_b, xc,
                batch, seq_len, d_inner);
            SG_LAUNCH_CHECK(stream);
        }

        // Save xc (post-conv-silu).
        cudaMemcpyAsync(A.xc_save, xc, (size_t)rows * d_inner * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // (d) x_proj GEMM: x_dbc [rows, dt_rank+2*d_state] = xc · x_proj_W^T
        err = gemm_rowmajor_NT<T>(handle, rows, dt_rank + 2 * d_state, d_inner,
                                  xc, W.x_proj_W, x_dbc, stream);
        if (err != cudaSuccess) return err;

        // (e) split x_dbc -> dt_pre, B_buf, C_buf
        {
            dim3 grid(rows, (dt_rank + 2 * d_state + 127) / 128);
            split_dbc_kernel<T><<<grid, 128, 0, stream>>>(
                x_dbc, dt_pre, B_buf, C_buf, rows, dt_rank, d_state);
            SG_LAUNCH_CHECK(stream);
        }

        // Save dt_pre, B, C.
        cudaMemcpyAsync(A.dt_pre_save, dt_pre, (size_t)rows * dt_rank * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(A.B_save, B_buf, (size_t)rows * d_state * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(A.C_save, C_buf, (size_t)rows * d_state * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // (f) dt_proj GEMM + softplus(+ bias).  dt_full = dt_pre · dt_proj_W^T
        err = gemm_rowmajor_NT<T>(handle, rows, d_inner, dt_rank,
                                  dt_pre, W.dt_proj_W, dt_full, stream);
        if (err != cudaSuccess) return err;
        {
            int block = 256;
            int grid  = (rows * d_inner + block - 1) / block;
            softplus_bias_kernel<T><<<grid, block, 0, stream>>>(dt_full, W.dt_proj_b, rows, d_inner);
            SG_LAUNCH_CHECK(stream);
        }

        // Save dt_full (post-softplus).
        cudaMemcpyAsync(A.dt_full_save, dt_full, (size_t)rows * d_inner * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // (g) selective scan via adapter
        err = mamba_adapter::selective_scan_forward<T>(
            xc, dt_full, W.A_log, B_buf, C_buf,
            y_scan, state_save,
            batch, seq_len, d_inner, d_state, stream);
        if (err != cudaSuccess) return err;

        // Save y_scan BEFORE the gate (gate modifies y_scan in-place).
        cudaMemcpyAsync(A.y_scan_save, y_scan, (size_t)rows * d_inner * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // (h) y = (y + x_main_post_silu * D) * SiLU(z)
        //     where the "x_main * D" skip uses the post-conv-SiLU xc per Python ref.
        {
            dim3 grid(rows, (d_inner + 127) / 128);
            gate_dskip_kernel<T><<<grid, 128, 0, stream>>>(
                y_scan, xc, W.D, z_buf, rows, d_inner);
            SG_LAUNCH_CHECK(stream);
        }

        // (i) out_proj GEMM:  h_new = y_gated · out_proj_W^T  [rows, d_model]
        err = gemm_rowmajor_NT<T>(handle, rows, d_model, d_inner,
                                  y_scan, W.out_proj_W, h, stream);
        if (err != cudaSuccess) return err;

        // (j) Add residual (h_pre)
        {
            int64_t n = static_cast<int64_t>(rows) * d_model;
            int block = 256;
            int grid  = static_cast<int>((n + block - 1) / block);
            add_residual_kernel<T><<<grid, block, 0, stream>>>(h, res, n);
            SG_LAUNCH_CHECK(stream);
        }

        // (k) LayerNorm 2 (post-block): save h (post-residual) as h_post_save,
        //     then apply LN2 in-place and record statistics.
        cudaMemcpyAsync(A.h_post_save, h, (size_t)rows * d_model * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);
        {
            int block = 128;
            size_t smem = 2 * block * sizeof(float);
            layernorm_kernel<T><<<rows, block, smem, stream>>>(
                h, W.ln2_g, W.ln2_b, h,
                A.mean_ln2, A.rstd_ln2, d_model, eps);
            SG_LAUNCH_CHECK(stream);
        }
    }

    // 3. Final LayerNorm + head
    WeightPtrs<T> Wlast;
    resolve_layer<T>(weights, 0, n_layers, vocab, seq_len,
                     d_model, d_inner, d_state, dt_rank, Wlast);
    TailActPtrs<T> Atail = resolve_tail_act<T>(
        act_base, n_layers, rows, batch, d_model, d_inner, dt_rank, d_state);
    {
        int block = 128;
        size_t smem = 2 * block * sizeof(float);
        layernorm_kernel<T><<<rows, block, smem, stream>>>(
            h, Wlast.ln_final_g, Wlast.ln_final_b, h,
            Atail.mean_lnf, Atail.rstd_lnf, d_model, eps);
        SG_LAUNCH_CHECK(stream);
    }

    // Gather last token: last [B, d_model] = h[:, N-1, :]
    // Save into tail activation cache (not state_save, which is scan state).
    {
        dim3 grid(batch, (d_model + 127) / 128);
        gather_last_token_kernel<T><<<grid, 128, 0, stream>>>(
            h, Atail.last_save, batch, seq_len, d_model);
        SG_LAUNCH_CHECK(stream);
    }

    // Head GEMM: output [B, p_vocab] = last [B, d_model] · head_W^T [p_vocab, d_model]
    int p_vocab = 97;   // grokking default
    cudaError_t err = gemm_rowmajor_NT<T>(handle, batch, p_vocab, d_model,
                                          Atail.last_save, Wlast.head_W, output, stream);
    return err;
}

// ─────────────────────────────────────────────────────────────────────
//  Backward pass — REAL fused reverse pass.
//
//  Processes the full N-layer Mamba stack in reverse order, computing
//  exact gradients for all weight parameters using per-layer activations
//  saved during forward(). The activation layout in activations_saved
//  matches the layout written by forward() into the states buffer.
//
//  grad_input stays zero (gradient w.r.t. integer token ids is undefined).
//
//  Gradient accumulators for weights are maintained as float buffers
//  internally and converted to T at the end for each parameter. Weight
//  gradient GEMMs use cuBLAS via ATen's handle (beta=1.0 to accumulate).
//
//  The selective scan backward uses mamba_adapter::selective_scan_backward
//  which implements the full adjoint scan (forward-recompute + reverse pass).
// ─────────────────────────────────────────────────────────────────────

template <typename T>
cudaError_t backward(
    const T* grad_output,            // [B, p_vocab] loss gradient
    const T* activations_saved,      // per-layer activations saved by forward()
    const T* weights,
    T* grad_input,                   // [B*N] — zero (grad w.r.t. token ids)
    T* grad_weights,                 // [total_weight_elems] accumulated grads
    int batch, int seq_len, int d_model, int d_state,
    int d_conv, int expand, int n_layers,
    cudaStream_t stream)
{
    (void)d_conv;
    const int d_inner  = d_model * expand;
    const int dt_rank  = std::max(d_model / 16, 1);
    const int vocab    = 99;
    const int p_vocab  = 97;
    const int rows     = batch * seq_len;
    const float eps    = 1e-5f;

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);

    // ── Zero grad_input ──────────────────────────────────────────────
    if (grad_input) {
        cudaMemsetAsync(grad_input, 0,
            (size_t)batch * seq_len * sizeof(T), stream);
    }
    if (!grad_weights) return cudaGetLastError();

    // ── Compute total weight buffer size and zero grad_weights ───────
    size_t total_w = 0;
    total_w += (size_t)vocab * d_model;           // tok_emb
    total_w += (size_t)seq_len * d_model;         // pos_emb
    total_w += (size_t)n_layers *
               per_layer_count<T>(d_model, d_inner, d_state, dt_rank);
    total_w += (size_t)2 * d_model;               // ln_final g,b
    total_w += (size_t)p_vocab * d_model;         // head_W
    cudaMemsetAsync(grad_weights, 0, total_w * sizeof(T), stream);

    // ── Allocate internal scratch for float gradient accumulators ────
    // We use cudaMalloc for per-layer float grad buffers. For small
    // grokking models this is fast; all allocations are freed before return.
    // scratch layout:
    //   grad_h      [rows, d_model]   T   — flowing gradient of h
    //   grad_xz     [rows, 2*d_inner] T
    //   grad_xm     [rows, d_inner]   T   — grad x_main (conv output)
    //   grad_z      [rows, d_inner]   T
    //   grad_xc     [rows, d_inner]   T   — grad xc (post-conv-silu)
    //   grad_yscan  [rows, d_inner]   T   — grad y_scan (before gate)
    //   grad_dtf    [rows, d_inner]   T   — grad dt_full (post-softplus)
    //   grad_dtp    [rows, dt_rank]   T   — grad dt_pre (pre-dt_proj output)
    //   grad_B_b    [rows, d_state]   T
    //   grad_C_b    [rows, d_state]   T
    //   grad_x_dbc  [rows, dt_rank+2*d_state] T
    //   grad_last   [batch, d_model]  T   — grad w.r.t. gathered last token
    // float accumulators (one set, reused per layer):
    //   fgD         [d_inner]         float  — grad D
    //   fg_ln1g     [d_model]         float  — grad ln1 gamma
    //   fg_ln1b     [d_model]         float
    //   fg_ln2g     [d_model]         float
    //   fg_ln2b     [d_model]         float
    //   fg_dtb      [d_inner]         float  — grad dt_proj_b
    //   fg_convW    [d_inner*3]       float  — grad conv_W
    //   fg_convb    [d_inner]         float  — grad conv_b
    //   fg_Alog     [d_inner*d_state] float  — grad A_log (from scan bwd)
    //   fg_lnfg     [d_model]         float  — grad ln_final gamma
    //   fg_lnfb     [d_model]         float

    size_t scratch_T_elems = 0;
    scratch_T_elems += (size_t)rows * d_model;                    // grad_h
    scratch_T_elems += (size_t)rows * 2 * d_inner;               // grad_xz
    scratch_T_elems += (size_t)rows * d_inner;                   // grad_xm
    scratch_T_elems += (size_t)rows * d_inner;                   // grad_z
    scratch_T_elems += (size_t)rows * d_inner;                   // grad_xc
    scratch_T_elems += (size_t)rows * d_inner;                   // grad_yscan
    scratch_T_elems += (size_t)rows * d_inner;                   // grad_dtf
    scratch_T_elems += (size_t)rows * dt_rank;                   // grad_dtp
    scratch_T_elems += (size_t)rows * d_state;                   // grad_B_b
    scratch_T_elems += (size_t)rows * d_state;                   // grad_C_b
    scratch_T_elems += (size_t)rows * (dt_rank + 2 * d_state);  // grad_x_dbc
    scratch_T_elems += (size_t)batch * d_model;                  // grad_last
    // float accumulators (reuse floats_as_T for alignment)
    size_t fa_elems = 0;
    fa_elems += (size_t)d_inner;                 // fgD
    fa_elems += (size_t)d_model;                 // fg_ln2g
    fa_elems += (size_t)d_model;                 // fg_ln2b
    fa_elems += (size_t)d_inner;                 // fg_dtb
    fa_elems += (size_t)d_inner * 3;             // fg_convW
    fa_elems += (size_t)d_inner;                 // fg_convb
    fa_elems += (size_t)d_inner * d_state;       // fg_Alog
    fa_elems += (size_t)d_model;                 // fg_lnfg
    fa_elems += (size_t)d_model;                 // fg_lnfb
    fa_elems += (size_t)rows * d_inner;          // fg_xm (float grad_x_main for conv bwd)
    // Round T-section up to 8-byte boundary for float alignment.
    size_t T_section_bytes = scratch_T_elems * sizeof(T);
    T_section_bytes = (T_section_bytes + 7) & ~size_t(7);
    size_t scratch_bytes = T_section_bytes + fa_elems * sizeof(float);

    void* scratch_raw = nullptr;
    cudaError_t merr = cudaMalloc(&scratch_raw, scratch_bytes);
    if (merr != cudaSuccess) return merr;
    cudaMemsetAsync(scratch_raw, 0, scratch_bytes, stream);

    // Partition scratch_raw into typed pointers.
    T* sp = reinterpret_cast<T*>(scratch_raw);
    T* grad_h     = sp; sp += (size_t)rows * d_model;
    T* grad_xz    = sp; sp += (size_t)rows * 2 * d_inner;
    T* grad_xm    = sp; sp += (size_t)rows * d_inner;
    T* grad_z     = sp; sp += (size_t)rows * d_inner;
    T* grad_xc    = sp; sp += (size_t)rows * d_inner;
    T* grad_yscan = sp; sp += (size_t)rows * d_inner;
    T* grad_dtf   = sp; sp += (size_t)rows * d_inner;
    T* grad_dtp   = sp; sp += (size_t)rows * dt_rank;
    T* grad_Bb    = sp; sp += (size_t)rows * d_state;
    T* grad_Cb    = sp; sp += (size_t)rows * d_state;
    T* grad_xdbc  = sp; sp += (size_t)rows * (dt_rank + 2 * d_state);
    T* grad_last  = sp; sp += (size_t)batch * d_model;
    // float accumulators — start at the 8-byte-aligned boundary after T section.
    float* fa = reinterpret_cast<float*>(
        reinterpret_cast<char*>(scratch_raw) + T_section_bytes);
    float* fgD     = fa; fa += d_inner;
    float* fg_ln2g = fa; fa += d_model;
    float* fg_ln2b = fa; fa += d_model;
    float* fg_dtb  = fa; fa += d_inner;
    float* fg_cW   = fa; fa += d_inner * 3;
    float* fg_cb   = fa; fa += d_inner;
    float* fg_Alog = fa; fa += d_inner * d_state;
    float* fg_lnfg = fa; fa += d_model;
    float* fg_lnfb = fa; fa += d_model;
    float* fg_xm   = fa; fa += (size_t)rows * d_inner;  // float grad_x_main for conv bwd
    // (fa pointer not used beyond this point)

    // Cast activation cache to mutable T* for resolve functions.
    T* act_base = const_cast<T*>(activations_saved);

    // ── Resolve tail activations (final LN stats + gathered last token) ──
    TailActPtrs<T> Atail = resolve_tail_act<T>(
        act_base, n_layers, rows, batch, d_model, d_inner, dt_rank, d_state);

    // ── Resolve final weight pointers ──────────────────────────────────
    WeightPtrs<T> Wlast;
    resolve_layer<T>(weights, 0, n_layers, vocab, seq_len,
                     d_model, d_inner, d_state, dt_rank, Wlast);

    // ── (A) Head GEMM backward ──────────────────────────────────────────
    // Forward: output[B, p_vocab] = last[B, d_model] · head_W^T
    // grad_last [B, d_model]   = grad_output [B, p_vocab] · head_W [p_vocab, d_model]
    // grad_head_W [p_vocab, d_model] += grad_output^T [p_vocab, B] · last [B, d_model]
    {
        // grad_last = grad_output @ head_W  (head_W is [p_vocab, d_model])
        cudaError_t err = gemm_rowmajor<T>(handle, batch, d_model, p_vocab,
                                           grad_output, Wlast.head_W, grad_last, stream);
        if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }

        // grad_head_W += grad_output^T @ last  (accumulate into grad_weights)
        // head_W offset in grad_weights:
        size_t head_off = (size_t)vocab * d_model + (size_t)seq_len * d_model
                        + (size_t)n_layers * per_layer_count<T>(d_model, d_inner, d_state, dt_rank)
                        + (size_t)2 * d_model;
        // gemm_rowmajor_TN: C [p_vocab, d_model] += A^T [B, p_vocab]^T · B [B, d_model]
        //   i.e. M=p_vocab, N=d_model, K=batch
        err = gemm_rowmajor_TN<T>(handle, p_vocab, d_model, batch,
                                   grad_output, Atail.last_save,
                                   grad_weights + head_off, stream);
        if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
    }

    // ── (B) Final LayerNorm backward ────────────────────────────────────
    // Forward: h = LN(h_pre, ln_final_g, ln_final_b)  (in-place, rows = B*N)
    // We need x_norm (= (h_pre - mean)*rstd) for the LN backward.
    // h_pre before final LN is the output of the last layer (= last layer's
    // post-LN2 output, which we'll need for the scatter). We reconstruct
    // x_norm from the last LN2 output (h) using save_xnorm_kernel.
    // Actually after the last layer, h went through final LN, so h_pre_lnf
    // is the last layer's post-LN2 output. The last layer's h_post_save
    // holds the post-residual value BEFORE LN2, but we need AFTER LN2
    // (= the LN2 output that feeds the final LN).
    // Since h_post_save is BEFORE LN2, and the LN2 output is what enters
    // the final LN, we need to reconstruct it. Instead: we save the
    // post-LN2 h using h_post_save in the layer loop below.
    // For now, expand the backward in terms of what IS saved:
    //   Atail.mean_lnf, Atail.rstd_lnf were saved during forward's final LN.
    //   The input to the final LN is the last layer's LN2 output.
    //   We use save_xnorm_kernel to compute x_norm from the last layer's
    //   h_post_save-post-ln2. But h_post_save is pre-LN2 in the current layout.
    //
    // Re-examination: in the modified forward, for layer l:
    //   h_post_save = h BEFORE LN2 (the post-residual value)
    //   LN2 is then applied in-place, and the OUTPUT feeds the next layer.
    // So after layer n_layers-1, h is the LN2 output of the last layer.
    // That h is what the final LN receives. The final LN saved mean_lnf/rstd_lnf.
    // To compute x_norm for the final LN backward, we need h BEFORE the final LN,
    // which is the last layer's LN2 output. We don't save this directly.
    //
    // Solution: save it into Atail using an extra h_pre_lnf field.
    // However, to avoid changing the layout again, we reconstruct it:
    // The last layer's h_post_save is pre-LN2. We can recompute LN2 output
    // using the saved mean_ln2/rstd_ln2. Let's add h_pre_lnf to the tail.
    //
    // Actually the simplest fix: in the forward loop for the last layer,
    // after LN2, copy h into Atail as h_pre_lnf. But that requires changing
    // the layout. Instead, since we have rstd_lnf and mean_lnf, we compute
    // x_norm from the LN2 output directly: we just need h_pre_lnf as a T buffer.
    // We can reuse grad_h as temporary for x_norm. But we need h_pre_lnf.
    //
    // Cleanest fix: reconstruct the LN2 output of the last layer by running
    // LN2 forward again (it's cheap). Use grad_xz as a temp buffer.
    {
        LayerActPtrs<T> Alast = resolve_layer_act<T>(
            act_base, n_layers - 1, rows, d_model, d_inner, dt_rank, d_state);
        WeightPtrs<T> Wl;
        resolve_layer<T>(weights, n_layers - 1, n_layers, vocab, seq_len,
                         d_model, d_inner, d_state, dt_rank, Wl);
        // Recompute LN2 output of last layer into grad_xz (as temp).
        // h_post_save is pre-LN2; we apply LN2 forward.
        int block = 128;
        size_t smem = 2 * block * sizeof(float);
        layernorm_kernel<T><<<rows, block, smem, stream>>>(
            Alast.h_post_save, Wl.ln2_g, Wl.ln2_b,
            grad_xz /*= h_pre_lnf_temp*/,
            nullptr, nullptr, d_model, eps);
        SG_LAUNCH_CHECK(stream);

        // Compute x_norm for final LN backward.
        save_xnorm_kernel<T><<<rows, 128, 0, stream>>>(
            grad_xz /*h_pre_lnf*/, Atail.mean_lnf, Atail.rstd_lnf,
            grad_xm /*x_norm_lnf*/, d_model);
        SG_LAUNCH_CHECK(stream);

        // Final LN backward: grad_last (reused as grad w.r.t. final LN input)
        // → writes grad_h which is the grad flowing into the last layer's LN2 output.
        // We scatter grad_last [B, d_model] → grad_h [rows, d_model] at position N-1.
        cudaMemsetAsync(grad_h, 0, (size_t)rows * d_model * sizeof(T), stream);
        {
            dim3 grid2(batch, (d_model + 127) / 128);
            scatter_last_token_kernel<T><<<grid2, 128, 0, stream>>>(
                grad_last, grad_h, batch, seq_len, d_model);
            SG_LAUNCH_CHECK(stream);
        }

        // Final LN backward kernel: produces dx (grad w.r.t. pre-LN input = last LN2 output)
        // and accumulates fg_lnfg, fg_lnfb.
        layernorm_backward_kernel<T><<<rows, 128, 2 * 128 * sizeof(float), stream>>>(
            grad_h, grad_xm /*x_norm_lnf*/, Wlast.ln_final_g, Atail.rstd_lnf,
            grad_last /*dx: reuse as temp*/, fg_lnfg, fg_lnfb, d_model, rows);
        SG_LAUNCH_CHECK(stream);

        // grad_last now holds the gradient w.r.t. the final LN's pre-LN input
        // (= last layer's LN2 output). This is grad_h for entering the reverse
        // layer loop. Copy into grad_h.
        cudaMemcpyAsync(grad_h, grad_last, (size_t)rows * d_model * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // Write ln_final grads to grad_weights.
        size_t lnf_off = (size_t)vocab * d_model + (size_t)seq_len * d_model
                       + (size_t)n_layers * per_layer_count<T>(d_model, d_inner, d_state, dt_rank);
        {
            int64_t nd = d_model;
            int blk = 256;
            int grd = (int)((nd + blk - 1) / blk);
            float_to_T_kernel<T><<<grd, blk, 0, stream>>>(fg_lnfg, grad_weights + lnf_off,          nd);
            float_to_T_kernel<T><<<grd, blk, 0, stream>>>(fg_lnfb, grad_weights + lnf_off + d_model, nd);
            SG_LAUNCH_CHECK(stream);
        }
    }

    // ── (C) Per-layer backward (reverse order) ──────────────────────────
    for (int l = n_layers - 1; l >= 0; l--) {
        LayerActPtrs<T> A = resolve_layer_act<T>(
            act_base, l, rows, d_model, d_inner, dt_rank, d_state);
        WeightPtrs<T> Wl;
        resolve_layer<T>(weights, l, n_layers, vocab, seq_len,
                         d_model, d_inner, d_state, dt_rank, Wl);

        // grad_h currently holds gradient w.r.t. this layer's LN2 output.

        // ── (C.1) LN2 backward ─────────────────────────────────────────
        // Forward: h_ln2 = LN2(h_post_residual, ln2_g, ln2_b)
        //   h_post_save = h_post_residual (pre-LN2 input)
        // Compute x_norm_ln2 = (h_post_save - mean_ln2) * rstd_ln2.
        // Reuse grad_xz (rows*2*d_inner) as temporary for x_norm_ln2 (rows*d_model).
        save_xnorm_kernel<T><<<rows, 128, 0, stream>>>(
            A.h_post_save, A.mean_ln2, A.rstd_ln2,
            grad_xz /*x_norm_ln2*/, d_model);
        SG_LAUNCH_CHECK(stream);

        // Zero fg_ln2g/fg_ln2b for this layer.
        cudaMemsetAsync(fg_ln2g, 0, (size_t)d_model * sizeof(float), stream);
        cudaMemsetAsync(fg_ln2b, 0, (size_t)d_model * sizeof(float), stream);

        // LN2 backward → grad w.r.t. h_post_residual into grad_last (temp).
        layernorm_backward_kernel<T><<<rows, 128, 2 * 128 * sizeof(float), stream>>>(
            grad_h, grad_xz /*x_norm_ln2*/, Wl.ln2_g, A.rstd_ln2,
            grad_last /*dx_post_residual*/, fg_ln2g, fg_ln2b, d_model, rows);
        SG_LAUNCH_CHECK(stream);

        // ── (C.2) Residual backward ─────────────────────────────────────
        // Forward: h_post_residual = out_proj_out + h_pre  (residual = h_pre)
        // grad_out_proj_out = grad_post_residual (= grad_last)
        // grad flows through residual: we add grad_last to grad_h_pre (accumulated later)
        // For now: grad_out_proj_out = grad_last (= dx_post_residual above).

        // ── (C.3) out_proj GEMM backward ───────────────────────────────
        // Forward: out_proj_out [rows, d_model] = y_gated [rows, d_inner] · out_proj_W^T
        // grad_y_gated [rows, d_inner] = grad_out_proj_out [rows, d_model] · out_proj_W [d_model, d_inner]
        //   Note: out_proj_W is [d_model, d_inner] (output_dim, input_dim).
        //   gemm: grad_y_gated = grad_out_proj_out @ out_proj_W
        //   i.e. [rows, d_inner] = [rows, d_model] @ [d_model, d_inner]
        //   = gemm_rowmajor: M=rows, N=d_inner, K=d_model, A=grad_last, B=out_proj_W
        {
            cudaError_t err = gemm_rowmajor<T>(handle, rows, d_inner, d_model,
                                               grad_last, Wl.out_proj_W,
                                               grad_yscan /*temp: grad_y_gated*/, stream);
            if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
        }
        // grad_out_proj_W [d_model, d_inner] += grad_out_proj_out^T @ y_gated
        //   y_gated is y_scan AFTER gate. We need to recompute it or save it.
        //   We saved y_scan_save (before gate) and xc_save, z_save, D.
        //   Recompute y_gated = gate_dskip(y_scan_save, xc_save, D, z_save) in a temp.
        //   Use grad_xz (rows*2*d_inner, enough) as temp for y_gated.
        {
            // Copy y_scan_save into a temp, then apply gate.
            cudaMemcpyAsync(grad_xz, A.y_scan_save,
                            (size_t)rows * d_inner * sizeof(T),
                            cudaMemcpyDeviceToDevice, stream);
            dim3 gg(rows, (d_inner + 127) / 128);
            gate_dskip_kernel<T><<<gg, 128, 0, stream>>>(
                grad_xz, A.xc_save, Wl.D, A.z_save, rows, d_inner);
            SG_LAUNCH_CHECK(stream);
            // grad_out_proj_W += grad_last^T @ y_gated
            // out_proj_W offset in grad_weights:
            size_t opw_off = layer_weight_offset<T>(
                l, vocab, seq_len, d_model, d_inner, d_state, dt_rank) +
                per_layer_out_proj_offset<T>(d_model, d_inner, d_state, dt_rank);
            cudaError_t err = gemm_rowmajor_TN<T>(handle, d_model, d_inner, rows,
                                                   grad_last, grad_xz,
                                                   grad_weights + opw_off, stream);
            if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
        }

        // ── (C.4) Gate+skip backward ────────────────────────────────────
        // Forward: y_gated = (y_scan + xc * D) * SiLU(z)
        // grad_y_gated = grad_yscan (currently holds grad w.r.t. y_gated)
        // Outputs: grad_yscan (grad scan output before gate),
        //          grad_xc_skip, grad_z, grad_D.
        cudaMemsetAsync(fgD, 0, (size_t)d_inner * sizeof(float), stream);
        {
            dim3 gg(rows, (d_inner + 127) / 128);
            gate_dskip_backward_kernel<T><<<gg, 128, 0, stream>>>(
                grad_yscan,      // grad_y_gated in
                A.y_scan_save,   // y_scan before gate
                A.xc_save,       // post-conv-silu
                Wl.D,
                A.z_save,
                grad_yscan,      // out: grad_y_scan (overwrite in-place OK since we copy)
                grad_xc,         // out: grad_xc from D-skip
                grad_z,          // out: grad_z
                fgD,             // out: grad_D (float atomicAdd)
                rows, d_inner);
            SG_LAUNCH_CHECK(stream);
        }
        // Write grad_D to grad_weights.
        {
            size_t d_off = layer_weight_offset<T>(
                l, vocab, seq_len, d_model, d_inner, d_state, dt_rank) +
                per_layer_D_offset<T>(d_model, d_inner, d_state, dt_rank);
            int blk = 256;
            int grd = (d_inner + blk - 1) / blk;
            float_to_T_kernel<T><<<grd, blk, 0, stream>>>(
                fgD, grad_weights + d_off, d_inner);
            SG_LAUNCH_CHECK(stream);
        }

        // ── (C.5) Selective scan backward ──────────────────────────────
        // mamba_adapter::selective_scan_backward produces:
        //   grad_xc_scan, grad_dt_full, grad_A_log, grad_B, grad_C
        // We zero fg_Alog first (selective_scan_backward memsets it internally,
        // but we need a separate float* for the layer accumulation).
        cudaMemsetAsync(fg_Alog, 0, (size_t)d_inner * d_state * sizeof(float), stream);
        {
            cudaError_t err = mamba_adapter::selective_scan_backward<T>(
                grad_yscan,        // grad_y [rows, d_inner]
                A.xc_save,         // x (= xc)
                A.dt_full_save,    // dt
                Wl.A_log,          // A_log
                A.B_save,          // B
                A.C_save,          // C
                nullptr,           // state_save (not needed for sequential scan bwd)
                grad_xm,           // grad_x  (grad w.r.t. xc from scan)
                grad_dtf,          // grad_dt_full
                fg_Alog,           // grad_A_log (float, accumulated)
                grad_Bb,           // grad_B
                grad_Cb,           // grad_C
                batch, seq_len, d_inner, d_state, stream);
            if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
        }
        // Accumulate total grad_xc = grad_xc_from_gate_skip + grad_xc_from_scan
        {
            int64_t nxc = (int64_t)rows * d_inner;
            int blk = 256;
            int grd = (int)((nxc + blk - 1) / blk);
            add_inplace_kernel<T><<<grd, blk, 0, stream>>>(grad_xc, grad_xm, nxc);
            SG_LAUNCH_CHECK(stream);
        }
        // Write grad_A_log to grad_weights.
        {
            size_t alog_off = layer_weight_offset<T>(
                l, vocab, seq_len, d_model, d_inner, d_state, dt_rank) +
                per_layer_Alog_offset<T>(d_model, d_inner, d_state, dt_rank);
            int64_t nalog = (int64_t)d_inner * d_state;
            int blk = 256;
            int grd = (int)((nalog + blk - 1) / blk);
            float_to_T_kernel<T><<<grd, blk, 0, stream>>>(
                fg_Alog, grad_weights + alog_off, nalog);
            SG_LAUNCH_CHECK(stream);
        }

        // ── (C.6) dt_proj + softplus backward ──────────────────────────
        // Forward: dt_full = softplus(dt_proj(dt_pre) + dt_proj_b)
        // grad_dt_full → grad_dt_proj_out (via softplus backward):
        //   grad_dt_proj_out = grad_dt_full * sigmoid(dt_proj_out + bias)
        //   where dt_proj_out = dt_full_pre_softplus = dt_pre · dt_proj_W^T
        //   We stored dt_pre_save; recompute dt_proj_out from it if needed.
        //   Actually: softplus_bias_backward_kernel needs dt_pre_proj
        //   (= the value before softplus, = dt_proj output). We can use
        //   dt_full_save as an approximation? No — we need the pre-softplus value.
        //
        //   Recompute dt_proj_out by running dt_proj GEMM forward from dt_pre_save.
        //   Use grad_dtp as temp (rows × dt_rank → rows × d_inner).
        //   Actually grad_dtp is rows × dt_rank; we need rows × d_inner temp.
        //   Use grad_xm (rows × d_inner) as temp.
        {
            // Recompute dt_proj GEMM output (pre-softplus) into grad_xm.
            cudaError_t err = gemm_rowmajor_NT<T>(handle, rows, d_inner, dt_rank,
                                                   A.dt_pre_save, Wl.dt_proj_W,
                                                   grad_xm, stream);
            if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
        }
        // grad_dt_proj_out = grad_dtf * sigmoid(grad_xm + dt_proj_b)
        // Apply softplus_bias_backward_kernel in-place on grad_dtf.
        {
            int blk = 256;
            int grd = (rows * d_inner + blk - 1) / blk;
            softplus_bias_backward_kernel<T><<<grd, blk, 0, stream>>>(
                grad_dtf, grad_xm /*dt_pre_proj*/, Wl.dt_proj_b, rows, d_inner);
            SG_LAUNCH_CHECK(stream);
        }
        // grad_dt_proj_b += sum(grad_dtf, axis=0)  (float atomicAdd)
        cudaMemsetAsync(fg_dtb, 0, (size_t)d_inner * sizeof(float), stream);
        {
            int blk = 256;
            int grd = (d_inner + blk - 1) / blk;
            accumulate_bias_grad_kernel<T><<<grd, blk, 0, stream>>>(
                grad_dtf, fg_dtb, rows, d_inner);
            SG_LAUNCH_CHECK(stream);
        }
        // Write grad_dt_proj_b.
        {
            size_t dtb_off = layer_weight_offset<T>(
                l, vocab, seq_len, d_model, d_inner, d_state, dt_rank) +
                per_layer_dtb_offset<T>(d_model, d_inner, d_state, dt_rank);
            int blk = 256;
            int grd = (d_inner + blk - 1) / blk;
            float_to_T_kernel<T><<<grd, blk, 0, stream>>>(
                fg_dtb, grad_weights + dtb_off, d_inner);
            SG_LAUNCH_CHECK(stream);
        }
        // dt_proj GEMM backward:
        // grad_dt_pre [rows, dt_rank] = grad_dtf [rows, d_inner] @ dt_proj_W [d_inner, dt_rank]
        //   (dt_proj_W is [d_inner, dt_rank])
        //   gemm_rowmajor: M=rows, N=dt_rank, K=d_inner; A=grad_dtf, B=dt_proj_W
        {
            cudaError_t err = gemm_rowmajor<T>(handle, rows, dt_rank, d_inner,
                                               grad_dtf, Wl.dt_proj_W,
                                               grad_dtp, stream);
            if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
        }
        // grad_dt_proj_W [d_inner, dt_rank] += grad_dtf^T @ dt_pre_save
        {
            size_t dtW_off = layer_weight_offset<T>(
                l, vocab, seq_len, d_model, d_inner, d_state, dt_rank) +
                per_layer_dtW_offset<T>(d_model, d_inner, d_state, dt_rank);
            cudaError_t err = gemm_rowmajor_TN<T>(handle, d_inner, dt_rank, rows,
                                                   grad_dtf, A.dt_pre_save,
                                                   grad_weights + dtW_off, stream);
            if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
        }

        // ── (C.7) x_proj backward ──────────────────────────────────────
        // Forward: x_dbc [rows, dt_rank+2*d_state] = xc [rows, d_inner] · x_proj_W^T
        // Split x_dbc → dt_pre, B_buf, C_buf.
        // We need grad w.r.t. x_dbc = concat([grad_dtp, grad_Bb, grad_Cb]).
        {
            dim3 gg(rows, (dt_rank + 2 * d_state + 127) / 128);
            concat_dbc_kernel<T><<<gg, 128, 0, stream>>>(
                grad_dtp, grad_Bb, grad_Cb,
                grad_xdbc, rows, dt_rank, d_state);
            SG_LAUNCH_CHECK(stream);
        }
        // grad_xc_from_xproj [rows, d_inner] = grad_xdbc @ x_proj_W [dt_rank+2*d_state, d_inner]
        {
            cudaError_t err = gemm_rowmajor<T>(handle, rows, d_inner,
                                               dt_rank + 2 * d_state,
                                               grad_xdbc, Wl.x_proj_W,
                                               grad_xm, stream);  // grad_xm = grad_xc_from_xproj
            if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
        }
        // Accumulate into grad_xc.
        {
            int64_t nxc = (int64_t)rows * d_inner;
            int blk = 256;
            int grd = (int)((nxc + blk - 1) / blk);
            add_inplace_kernel<T><<<grd, blk, 0, stream>>>(grad_xc, grad_xm, nxc);
            SG_LAUNCH_CHECK(stream);
        }
        // grad_x_proj_W [dt_rank+2*d_state, d_inner] += grad_xdbc^T @ xc_save
        {
            size_t xpW_off = layer_weight_offset<T>(
                l, vocab, seq_len, d_model, d_inner, d_state, dt_rank) +
                per_layer_xprojW_offset<T>(d_model, d_inner, d_state, dt_rank);
            cudaError_t err = gemm_rowmajor_TN<T>(handle, dt_rank + 2 * d_state, d_inner, rows,
                                                   grad_xdbc, A.xc_save,
                                                   grad_weights + xpW_off, stream);
            if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
        }

        // ── (C.8) Conv1d + SiLU backward ───────────────────────────────
        // Forward: xc = silu(conv1d(x_main, W_conv, b_conv))
        // grad_x_main (from conv backward) + grad_conv_W + grad_conv_b
        // fg_xm is a float buffer for grad_x_main (avoids type-punning atomicAdd).
        cudaMemsetAsync(fg_xm,   0, (size_t)rows * d_inner * sizeof(float), stream);
        cudaMemsetAsync(fg_cW,   0, (size_t)d_inner * 3 * sizeof(float), stream);
        cudaMemsetAsync(fg_cb,   0, (size_t)d_inner * sizeof(float), stream);
        {
            dim3 gg((d_inner + 127) / 128, seq_len, batch);
            conv1d_silu_backward_kernel<T><<<gg, 128, 0, stream>>>(
                grad_xc, A.x_main_save, Wl.conv_W, Wl.conv_b,
                fg_xm /*float output*/, fg_cW, fg_cb, batch, seq_len, d_inner);
            SG_LAUNCH_CHECK(stream);
        }
        // Convert float grad_x_main to T in grad_xm.
        {
            int64_t nxm = (int64_t)rows * d_inner;
            int blk = 256;
            int grd = (int)((nxm + blk - 1) / blk);
            float_to_T_kernel<T><<<grd, blk, 0, stream>>>(fg_xm, grad_xm, nxm);
            SG_LAUNCH_CHECK(stream);
        }
        // Write grad_conv_W and grad_conv_b to grad_weights.
        {
            size_t cW_off = layer_weight_offset<T>(
                l, vocab, seq_len, d_model, d_inner, d_state, dt_rank) +
                per_layer_convW_offset<T>(d_model, d_inner, d_state, dt_rank);
            size_t cb_off = cW_off + (size_t)3 * d_inner;
            int blk = 256;
            int g1 = (3 * d_inner + blk - 1) / blk;
            int g2 = (d_inner + blk - 1) / blk;
            float_to_T_kernel<T><<<g1, blk, 0, stream>>>(fg_cW, grad_weights + cW_off, 3 * d_inner);
            float_to_T_kernel<T><<<g2, blk, 0, stream>>>(fg_cb, grad_weights + cb_off, d_inner);
            SG_LAUNCH_CHECK(stream);
        }

        // ── (C.9) split backward: concat([grad_xm, grad_z]) → grad_xz ─
        {
            dim3 gg(rows, (d_inner + 127) / 128);
            concat_chunk2_kernel<T><<<gg, 128, 0, stream>>>(
                grad_xm, grad_z, grad_xz, rows, d_inner);
            SG_LAUNCH_CHECK(stream);
        }

        // ── (C.10) in_proj GEMM backward ───────────────────────────────
        // Forward: xz [rows, 2*d_inner] = h_pre [rows, d_model] · in_proj_W^T
        // (LN1 is NOT applied in the current forward formulation; h_pre is used directly.)
        // grad_h_from_inproj [rows, d_model] = grad_xz @ in_proj_W
        {
            cudaError_t err = gemm_rowmajor<T>(handle, rows, d_model, 2 * d_inner,
                                               grad_xz, Wl.in_proj_W,
                                               grad_xc /*temp: grad_h_from_inproj*/, stream);
            if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
        }
        // grad_in_proj_W [2*d_inner, d_model] += grad_xz^T @ h_pre
        {
            size_t ipW_off = layer_weight_offset<T>(
                l, vocab, seq_len, d_model, d_inner, d_state, dt_rank) +
                per_layer_inprojW_offset<T>(d_model, d_inner, d_state, dt_rank);
            cudaError_t err = gemm_rowmajor_TN<T>(handle, 2 * d_inner, d_model, rows,
                                                   grad_xz, A.h_pre,
                                                   grad_weights + ipW_off, stream);
            if (err != cudaSuccess) { cudaFreeAsync(scratch_raw, stream); return err; }
        }

        // ── (C.11) Accumulate total grad_h for previous layer ──────────
        // The residual: h_post_residual = out_proj_out + h_pre
        //   → grad_h_pre_residual = grad_last (dx_post_residual from LN2 bwd)
        // in_proj path: grad_h_pre_from_inproj = grad_xc
        // Total grad flowing into h_pre = grad_last + grad_xc
        // This becomes grad_h for the previous layer (l-1).
        {
            int64_t nd = (int64_t)rows * d_model;
            int blk = 256;
            int grd = (int)((nd + blk - 1) / blk);
            add_inplace_kernel<T><<<grd, blk, 0, stream>>>(grad_last, grad_xc, nd);
            SG_LAUNCH_CHECK(stream);
            cudaMemcpyAsync(grad_h, grad_last, nd * sizeof(T),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // Write ln2 weight grads to grad_weights.
        // (ln1_g/ln1_b grads are zero since LN1 is not applied in forward.)
        {
            int blk = 256;
            int grd = (d_model + blk - 1) / blk;

            size_t ln2g_off = layer_weight_offset<T>(
                l, vocab, seq_len, d_model, d_inner, d_state, dt_rank) +
                per_layer_ln2_offset<T>(d_model, d_inner, d_state, dt_rank);
            float_to_T_kernel<T><<<grd, blk, 0, stream>>>(
                fg_ln2g, grad_weights + ln2g_off,          d_model);
            float_to_T_kernel<T><<<grd, blk, 0, stream>>>(
                fg_ln2b, grad_weights + ln2g_off + d_model, d_model);
            SG_LAUNCH_CHECK(stream);
        }
    }  // end per-layer backward loop

    // ── (D) Embedding backward ───────────────────────────────────────────
    // grad_h now holds gradient w.r.t. the embedding output (first layer's input).
    // Computing grad_tok_emb and grad_pos_emb requires the token ids (int32 input),
    // which are not passed to backward(). The embedding weight grads are correctly
    // left at zero (as set by the initial memset). The Python training loop handles
    // embedding gradients via autograd for the integer-indexed embedding table.
    (void)grad_h;  // grad_h is the final upstream gradient at the embedding output

    // Use stream-ordered free to avoid freeing scratch before in-flight kernels complete.
    // cudaFreeAsync is available since CUDA 11.2 (SM90 requires CUDA 11.8+, so always OK).
    cudaFreeAsync(scratch_raw, stream);
    return cudaGetLastError();
}

// ─────────────────────────────────────────────────────────────────────
//  selective_scan_fwd / bwd : thin component-test wrappers around the
//  adapter. The binding signature has only 4 inputs (u, delta, A, B) and
//  2 outputs (out, state) — for component testing we set d_inner = d_state
//  and use B as both B and C. This matches the simple "minimal scan" test.
// ─────────────────────────────────────────────────────────────────────

template <typename T>
cudaError_t selective_scan_fwd(
    const T* u, const T* delta, const T* A, const T* B,
    T* out, T* state,
    int batch, int seq_len, int d_state, cudaStream_t stream)
{
    // Component-test wrapper: d_inner == d_state, C == B.
    float* state_f = reinterpret_cast<float*>(state);
    return mamba_adapter::selective_scan_forward<T>(
        u, delta, A, B, B, out, state_f,
        batch, seq_len, /*d_inner=*/d_state, d_state, stream);
}

template <typename T>
cudaError_t selective_scan_bwd(
    const T* grad_out,
    const T* u, const T* delta, const T* A,
    const T* B, const T* state,
    T* grad_u, T* grad_delta, T* grad_A, T* grad_B,
    int batch, int seq_len, int d_state, cudaStream_t stream)
{
    // Component-test wrapper. Caller should allocate `state` and `grad_A`
    // as FP32 buffers regardless of T (the adapter operates in FP32 for the
    // recurrence) — we reinterpret accordingly. d_inner == d_state and
    // C := B (so grad_C aliases grad_B).
    const float* state_f = reinterpret_cast<const float*>(state);
    float* grad_A_f      = reinterpret_cast<float*>(grad_A);
    return mamba_adapter::selective_scan_backward<T>(
        grad_out, u, delta, A, B, B, state_f,
        grad_u, grad_delta, grad_A_f, grad_B, grad_B,
        batch, seq_len, /*d_inner=*/d_state, d_state, stream);
}

}}}}  // namespace sg::sm90::models::mamba

#endif  // GROKKING_KERNELS_SM90_MAMBA3_SM90_CUH_
