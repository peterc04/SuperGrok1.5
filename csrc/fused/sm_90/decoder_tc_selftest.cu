// csrc/fused/sm_90/decoder_tc_selftest.cu
// ─────────────────────────────────────────────────────────────────────────
// Test-only nvcc TU that micro-gates the GEMM engine of
// csrc/fused/sm_90/model_stage_decoder_tc.cuh (the decoder TC variant) in the
// THREE operand orientations the decoder fwd+bwd use, with the transposed
// HBM->smem staging that keeps every wgmma issue on the substrate-validated
// TransA=0/TransB=0 path:
//
//   (fwd) Y[M,Nout] = X[M,Kin] @ W[Nout,Kin]^T   (both K-major; no transpose)
//   (dX)  dX[M,Kin] = dY[M,Nout] @ W[Nout,Kin]    (W staged transposed: n=in,k=out)
//   (dW)  dW[Nout,Kin] = dY[T,Nout]^T @ X[T,Kin]  (BOTH staged transposed; K=T)
//
// Driven by tests/hw/test_decoder_tc.py via cpp_extension.load (NOT setup.py —
// test-only, never in the shipped glob). It is loaded for sm_90a so the wgmma
// PTX is real. Each entry point runs ONE tiny GEMM on a single CTA (256 thr,
// 1 producer + 1 consumer warpgroup the engine expects) and returns the fp32
// result for comparison against an fp32/fp64 reference of the bf16-rounded
// inputs.
//
// CRITICAL (DESIGN honesty rail + the scalar path's burned-in lesson at
// model_stages_decoder.cuh:619-625): the dW gate uses MULTIPLE k-steps with the
// transposed staging (K>=128), because a single-tile A=I gate catches a
// transpose-LAYOUT bug but NOT a k-step STRIDE bug in the transposed staging —
// the exact bug class that made "the forward stay exact while every grad was
// garbage" on the scalar path.
// ─────────────────────────────────────────────────────────────────────────
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "csrc/fused/sm_90/model_stage_decoder_tc.cuh"

namespace dectc = ::sg::fused::sm90::dectc;
namespace wgs   = ::sg::sm90::wgs;

// ── smem arena for one A(64x16) + one B(Nmax x16) bf16 tile (the unpipelined
//    engine reuses two buffers across k-steps). Nmax = 128. 16B aligned. ──────
// Full-shape fwd: M = TILE_M (token rows), N_total = Nout (tiled into N=128/96
// atoms by the caller loop), K = Kin. Exercises N-tiling (Nout=512 → 4 tiles).
template <int N>
__global__ void __launch_bounds__(256)
gemm_fwd_kernel(const __nv_bfloat16* __restrict__ X,   // [M, Kin] row-major
                const __nv_bfloat16* __restrict__ W,   // [Nout, Kin] row-major
                float* __restrict__ D,                 // [M, Nout] row-major out
                int Kin, int Nout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sA = smem;
    __nv_bfloat16* sB = sA + dectc::kDecTcStages * 64 * 16;   // B ring after the A ring          // A is 64x16
    const int M = dectc::kTileM;
    const int m_atoms = M / 64;
    const int k_steps = Kin / 16;
    // Loop N-tiles of width N over [0, Nout). (fwd) D[m,n] = Σ_k X[m,k]·W[n,k].
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return X[(int64_t)m * Kin + k]; };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return nn < Nout ? W[(int64_t)nn * Kin + k] : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) { if (m < M) D[(int64_t)m * Nout + n0 + n] = v; };
        dectc::tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/2>(
            0, m_atoms, n_real, k_steps, srcA, srcB, out, sA, sB);
    }
#else
    (void)X; (void)W; (void)D; (void)Kin; (void)Nout;
#endif
}

// (dX) dX[m,kin] = Σ_out dY[m,out]·W[out,kin]. GEMM: A=dY (m, out=K), B=W staged
// so n=kin, k=out → srcB(n=kin, k=out) reads W[out, kin] (PHYSICAL TRANSPOSE).
// Here N (the wgmma N) = Kin (the in_dim), and the contraction K = Nout.
template <int N>
__global__ void __launch_bounds__(256)
gemm_dx_kernel(const __nv_bfloat16* __restrict__ dY,   // [M, Nout] row-major
               const __nv_bfloat16* __restrict__ W,    // [Nout, N] row-major (in_dim=N)
               float* __restrict__ dX,                 // [M, N] row-major out
               int Nout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sA = smem;
    __nv_bfloat16* sB = sA + dectc::kDecTcStages * 64 * 16;   // B ring after the A ring
    const int M = dectc::kTileM;
    const int m_atoms = M / 64;
    const int k_steps = Nout / 16;
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return dY[(int64_t)m * Nout + k]; };
    // B[n=kin, k=out] = W[out, kin]  → transposed read.
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 { return W[(int64_t)k * N + n]; };
    auto out  = [&] (int m, int n, float v) { if (m < M) dX[(int64_t)m * N + n] = v; };
    dectc::tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/2>(
        0, m_atoms, /*n_real=*/N, k_steps, srcA, srcB, out, sA, sB);
#else
    (void)dY; (void)W; (void)dX; (void)Nout;
#endif
}

// (dW) dW[out,in] = Σ_t dY[t,out]·X[t,in]. GEMM: A=dY staged so m=out, k=t
// (dY is [t,out] → srcA(m=out,k=t)=dY[t,out], TRANSPOSE); B=X staged so n=in,
// k=t (X is [t,in] → srcB(n=in,k=t)=X[t,in], TRANSPOSE). K = T (the full token
// dim → MULTI-k-step, the stride-bug gate). M here is Nout (rows of dW), tiled
// as kAtomsM stacked m64 atoms; N is the wgmma N = in_dim.
// dW: M = Nout (rows of dW, up to 512 = 8 atoms), N = in_dim (<=128), K = T.
template <int N>
__global__ void __launch_bounds__(256)
gemm_dw_kernel(const __nv_bfloat16* __restrict__ dY,   // [T, Nout] row-major
               const __nv_bfloat16* __restrict__ X,    // [T, N] row-major (in_dim=N)
               float* __restrict__ dW,                 // [Nout, N] row-major out
               int Nout, int T) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sA = smem;
    __nv_bfloat16* sB = sA + dectc::kDecTcStages * 64 * 16;   // B ring after the A ring
    const int k_steps = T / 16;
    const int m_atoms = (Nout + 63) / 64;        // up to 8 for Nout=512
    // A[m=out, k=t] = dY[t, out]  → transposed read.
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
        return m < Nout ? dY[(int64_t)k * Nout + m] : __float2bfloat16(0.f); };
    // B[n=in, k=t] = X[t, in]     → transposed read.
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 { return X[(int64_t)k * N + n]; };
    auto out  = [&] (int m, int n, float v) { if (m < Nout) dW[(int64_t)m * N + n] = v; };
    dectc::tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/8>(
        0, m_atoms, /*n_real=*/N, k_steps, srcA, srcB, out, sA, sB);
#else
    (void)dY; (void)X; (void)dW; (void)Nout; (void)T;
#endif
}

// kDecTcStages tiles per operand (the GEMM K-loop double-buffer ring; S=2 → 2
// A-tiles + 2 B-tiles). Mirrors DecTcSmem.sA/sB; without this the pipelined
// tc_gemm_block_unpipelined writes ring slot 1 out of bounds (illegal access).
static size_t arena_bytes() {
    return (size_t)(dectc::kDecTcStages * 64 * 16
                    + dectc::kDecTcStages * 128 * 16) * sizeof(__nv_bfloat16) + 256;
}

template <int N>
static torch::Tensor run_fwd(torch::Tensor X, torch::Tensor W, int Kin, int Nout) {
    TORCH_CHECK(X.scalar_type() == torch::kBFloat16 && W.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.is_contiguous() && W.is_contiguous());
    auto D = torch::zeros({dectc::kTileM, Nout}, torch::dtype(torch::kFloat32).device(X.device()));
    size_t smem = arena_bytes();
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gemm_fwd_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gemm_fwd_kernel<N><<<1, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), D.data_ptr<float>(), Kin, Nout);
    C10_CUDA_CHECK(cudaGetLastError());
    return D;
}
template <int N>
static torch::Tensor run_dx(torch::Tensor dY, torch::Tensor W, int Nout) {
    auto dX = torch::zeros({dectc::kTileM, N}, torch::dtype(torch::kFloat32).device(dY.device()));
    size_t smem = arena_bytes();
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gemm_dx_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gemm_dx_kernel<N><<<1, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), dX.data_ptr<float>(), Nout);
    C10_CUDA_CHECK(cudaGetLastError());
    return dX;
}
template <int N>
static torch::Tensor run_dw(torch::Tensor dY, torch::Tensor X, int Nout, int T) {
    auto dW = torch::zeros({Nout, N}, torch::dtype(torch::kFloat32).device(dY.device()));
    size_t smem = arena_bytes();
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gemm_dw_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gemm_dw_kernel<N><<<1, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()), dW.data_ptr<float>(), Nout, T);
    C10_CUDA_CHECK(cudaGetLastError());
    return dW;
}

// Runtime-N dispatch to the compile-time template (decoder's N set: 128 in/out/
// ff0(dX)/ff2(dX); ff0 fwd N=dff=512 tiled as 4x128 by the real header, but the
// micro-gate exercises the atom widths the engine issues: 128 (and 96 for head)).
static torch::Tensor gemm_fwd(torch::Tensor X, torch::Tensor W, int N, int Kin, int Nout) {
    switch (N) { case 96: return run_fwd<96>(X, W, Kin, Nout); case 128: return run_fwd<128>(X, W, Kin, Nout);
                 default: TORCH_CHECK(false, "fwd N ", N); }
}
static torch::Tensor gemm_dx(torch::Tensor dY, torch::Tensor W, int N, int Nout) {
    switch (N) { case 96: return run_dx<96>(dY, W, Nout); case 128: return run_dx<128>(dY, W, Nout);
                 default: TORCH_CHECK(false, "dx N ", N); }
}
static torch::Tensor gemm_dw(torch::Tensor dY, torch::Tensor X, int N, int Nout, int T) {
    switch (N) { case 96: return run_dw<96>(dY, X, Nout, T); case 128: return run_dw<128>(dY, X, Nout, T);
                 default: TORCH_CHECK(false, "dw N ", N); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fwd", &gemm_fwd, "TC fwd Y=X@W^T (TILE_M x N), bf16/fp32acc");
    m.def("gemm_dx",  &gemm_dx,  "TC dX=dY@W (W transposed-staged), bf16/fp32acc");
    m.def("gemm_dw",  &gemm_dw,  "TC dW=dY^T@X (both transposed-staged, K=T), bf16/fp32acc");
    m.attr("TILE_M") = (int)dectc::kTileM;
}
