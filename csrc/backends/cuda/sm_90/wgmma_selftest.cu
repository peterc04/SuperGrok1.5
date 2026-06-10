// csrc/backends/cuda/sm_90/wgmma_selftest.cu
// ─────────────────────────────────────────────────────────────────────────
// Minimal nvcc test TU for the R2 tensor-core substrate (wgmma.cuh /
// tile_pipeline.cuh). Compiled via torch.utils.cpp_extension.load() by
// tests/hw/test_wgmma_substrate.py (NOT via setup.py — this TU is test-only
// and is never part of the shipped build glob).
//
// Exposes torch-callable entry points that run small, seconds-scale CORRECTNESS
// gates on a single CTA (one warpgroup = 128 threads, blockDim.x=128). These
// are contention-immune correctness checks, NOT timed benchmarks (perf is a
// later quiet-GPU phase): tiny grids, < 1 GB allocations.
//
// Gates exercised here (driven from python):
//   (a) single-tile bf16 GEMM (m64×N×16) vs an fp32 reference of the
//       bf16-rounded inputs, over multiple N incl. ragged edges;
//   (b) multi-tile K-loop accumulation (the k16 chain over K=128);
//   (c) pipelined variant (producer/consumer, 2-3 stages) — same numerics;
//   (d) determinism: same inputs -> bitwise-identical outputs;
//   (e) occupancy / refuse probe for an oversized dynamic-smem request.
//
// All operands are bf16 in smem (the design's input dtype), fp32 accumulators.
// ─────────────────────────────────────────────────────────────────────────
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/std/utility>   // cuda::std::pair (descriptor-pair return)
#include <cstdint>

#include "csrc/backends/cuda/sm_90/wgmma.cuh"
#include "csrc/backends/cuda/sm_90/tile_pipeline.cuh"

namespace wgs = sg::sm90::wgs;

// =========================================================================
//  Helpers shared by the kernels.
//
//  smem operand layout — the CUTLASS-canonical Major-K (INTERLEAVE/no-swizzle)
//  layout the ss-wgmma actually requires (NOT a plain row-major block). For an
//  operand of MN rows × K(=16) bf16, element (mn,k) lands at the bf16 index
//      idx(mn,k) = (k/8)*(MN*8) + mn*8 + (k%8)
//  i.e. K=16 is two stacked 8-wide core matrices, each an MN×8 row-major block.
//  Both A (MN=64) and B (MN=N) are staged this way; the wgmma then computes the
//  TN product  D[m,n] = sum_k A[m,k]·B[n,k]  with BOTH operands Major-K
//  (TransA=0, TransB=0 — Major::K==0 in cute). The python reference computes
//  exactly D = einsum("smk,snk->mn", A_bf16, B_bf16) over the SAME logical
//  [k_step,MN,16] tensors (global stays row-major; only the smem staging is
//  canonicalized). See make_desc_{A,B}_kmajor in wgmma.cuh.
// =========================================================================

// Canonical Major-K bf16 smem index for element (mn,k) of an MN×16 tile.
__device__ __forceinline__ int kmajor_idx(int mn, int k, int MN) {
    return (k >> 3) * (MN * 8) + mn * 8 + (k & 7);
}

template <int N>
__global__ void __launch_bounds__(128)
gemm_single_tile_kernel(
    const __nv_bfloat16* __restrict__ gA,   // [64*16] row-major M×K
    const __nv_bfloat16* __restrict__ gB,   // [N*16]  row-major N×K
    float* __restrict__ gD,                 // [64*N]  row-major M×N out
    int k_steps                              // number of 16-wide K tiles
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // smem operand tiles: one A (64×16) + one B (N×16) per K-step, all staged.
    // For the single-tile / K-loop gates we stage ALL k_steps contiguously so
    // the descriptor generator just advances a pointer (no pipelining).
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sA = smem;                       // [k_steps][64*16]
    __nv_bfloat16* sB = sA + (size_t)k_steps * 64 * 16; // [k_steps][N*16]

    const int tid = threadIdx.x;  // 0..127 (one warpgroup)

    // Cooperative load of all K-tiles into smem, SCATTERED from global
    // row-major ([k_step,MN,16]) into the canonical Major-K core-matrix layout
    // make_desc_*_kmajor requires. Each thread copies elements at its stride
    // and routes them via kmajor_idx. 64*16=1024 A elems + N*16 B elems / step.
    for (int k = 0; k < k_steps; ++k) {
        const __nv_bfloat16* aSrc = gA + (size_t)k * 64 * 16;   // row-major MxK
        const __nv_bfloat16* bSrc = gB + (size_t)k * N * 16;    // row-major NxK
        __nv_bfloat16* aDst = sA + (size_t)k * 64 * 16;
        __nv_bfloat16* bDst = sB + (size_t)k * N * 16;
        for (int i = tid; i < 64 * 16; i += 128) {
            int m = i >> 4, kk = i & 15;                 // global (m,k), K=16
            aDst[kmajor_idx(m, kk, 64)] = aSrc[i];
        }
        for (int i = tid; i < N * 16; i += 128) {
            int n = i / 16, kk = i % 16;
            bDst[kmajor_idx(n, kk, N)] = bSrc[i];
        }
    }
    __syncthreads();

    wgs::WgmmaAccum<N> acc;  // fp32 fragment, N/2 regs/thread

    // Descriptor generator: k-step k → {descA, descB} for the pre-staged tiles
    // (each tile already in canonical Major-K layout; the builder encodes the
    // canonical LBO=MN*16 / SBO=128 from the MN extent — see wgmma.cuh).
    auto gen = [&] (int k) {
        const __nv_bfloat16* aTile = sA + (size_t)k * 64 * 16;
        const __nv_bfloat16* bTile = sB + (size_t)k * N * 16;
        wgs::SmemDesc dA = wgs::make_desc_A_kmajor</*M=*/64, wgs::kSwizzleNone>(aTile);
        wgs::SmemDesc dB = wgs::make_desc_B_kmajor</*N=*/N,  wgs::kSwizzleNone>(bTile);
        return cuda::std::pair<wgs::SmemDesc, wgs::SmemDesc>(dA, dB);
    };

    // The full choreography: fence → (k=0 overwrite, k>0 accumulate) →
    // commit → wait_group<0>. Both operands Major-K ⇒ TransA=0, TransB=0
    // (cute Major::K == 0); the wgmma reads D[m,n] = Σ_k A[m,k]·B[n,k].
    wgs::wgmma_mainloop_kchain<N, /*TransA=*/0, /*TransB=*/0>(acc, k_steps, gen);

    // Epilogue: store each thread's N/2 fragment regs to the right (row,col)
    // of the M×N output using the authoritative fragment decode.
    #pragma unroll
    for (int i = 0; i < wgs::WgmmaAccum<N>::kRegs; ++i) {
        int row, col;
        wgs::wgmma_frag_decode(tid, i, N, row, col);
        gD[row * N + col] = acc.c[i];
    }
#else
    (void)gA; (void)gB; (void)gD; (void)k_steps;
#endif
}

// Pipelined variant: producer/consumer warpgroup split staging K-tiles through
// an mbarrier ring (tile_pipeline.cuh), proving the choreography. Same numerics
// as the unpipelined kernel above (gate c). Uses blockDim.x = 256 (WG0
// producer + WG1 consumer).
template <int N, int PipeDepth>
__global__ void __launch_bounds__(256)
gemm_pipelined_kernel(
    const __nv_bfloat16* __restrict__ gA,
    const __nv_bfloat16* __restrict__ gB,
    float* __restrict__ gD,
    int k_steps
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    wgs::tc_pipelined_gemm_m64nNk16<N, PipeDepth>(gA, gB, gD, k_steps);
#else
    (void)gA; (void)gB; (void)gD; (void)k_steps;
#endif
}

// =========================================================================
//  Host launchers (torch-callable).
// =========================================================================

static size_t single_tile_smem_bytes(int N, int k_steps) {
    return (size_t)k_steps * (64 * 16 + N * 16) * sizeof(__nv_bfloat16);
}

// Launch the unpipelined single-tile / K-loop GEMM for a compile-time N.
template <int N>
static torch::Tensor run_single_tile(torch::Tensor A, torch::Tensor B, int k_steps) {
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "B must be bf16");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(A.numel() == (int64_t)k_steps * 64 * 16, "A numel mismatch");
    TORCH_CHECK(B.numel() == (int64_t)k_steps * N * 16,  "B numel mismatch");

    auto D = torch::empty({64, N},
                          torch::dtype(torch::kFloat32).device(A.device()));
    size_t smem = single_tile_smem_bytes(N, k_steps);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Allow > 48 KB dynamic smem (opt-in attribute) for larger k_steps.
    cudaFuncSetAttribute(gemm_single_tile_kernel<N>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    gemm_single_tile_kernel<N><<<1, 128, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        D.data_ptr<float>(), k_steps);
    C10_CUDA_CHECK(cudaGetLastError());
    return D;
}

// Pipelined launcher (compile-time N + depth).
template <int N, int PipeDepth>
static torch::Tensor run_pipelined(torch::Tensor A, torch::Tensor B, int k_steps) {
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16 && B.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.is_contiguous() && B.is_contiguous());
    auto D = torch::empty({64, N},
                          torch::dtype(torch::kFloat32).device(A.device()));
    // Ring of PipeDepth stages, each one A(64×16)+B(N×16) bf16 tile + the
    // mbarrier words.
    size_t ring = (size_t)PipeDepth * (64 * 16 + N * 16) * sizeof(__nv_bfloat16);
    size_t bars = (size_t)PipeDepth * 2 * sizeof(unsigned long long); // full+empty
    size_t smem = ring + bars + 256;  // slack for alignment
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gemm_pipelined_kernel<N, PipeDepth>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gemm_pipelined_kernel<N, PipeDepth><<<1, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        D.data_ptr<float>(), k_steps);
    C10_CUDA_CHECK(cudaGetLastError());
    return D;
}

// Dispatch over the runtime N to the compile-time template (the design's set).
static torch::Tensor gemm_tile(torch::Tensor A, torch::Tensor B, int N, int k_steps) {
    switch (N) {
        case 8:   return run_single_tile<8>(A, B, k_steps);
        case 16:  return run_single_tile<16>(A, B, k_steps);
        case 32:  return run_single_tile<32>(A, B, k_steps);
        case 64:  return run_single_tile<64>(A, B, k_steps);
        case 96:  return run_single_tile<96>(A, B, k_steps);
        case 128: return run_single_tile<128>(A, B, k_steps);
        default: TORCH_CHECK(false, "unsupported N ", N);
    }
}

static torch::Tensor gemm_tile_pipelined(torch::Tensor A, torch::Tensor B,
                                         int N, int k_steps, int depth) {
    if (depth == 2) {
        switch (N) {
            case 64:  return run_pipelined<64, 2>(A, B, k_steps);
            case 128: return run_pipelined<128, 2>(A, B, k_steps);
            default: TORCH_CHECK(false, "pipelined: unsupported N ", N);
        }
    } else if (depth == 3) {
        switch (N) {
            case 64:  return run_pipelined<64, 3>(A, B, k_steps);
            case 128: return run_pipelined<128, 3>(A, B, k_steps);
            default: TORCH_CHECK(false, "pipelined: unsupported N ", N);
        }
    }
    TORCH_CHECK(false, "pipelined: unsupported depth ", depth);
}

// =========================================================================
//  Occupancy / refuse probe (gate e). The faithful "occupancy >= 1 or refuse"
//  contract: query cudaOccupancyMaxActiveBlocksPerMultiprocessor for a given
//  dynamic-smem request; if it is 0 the launch would not be resident → REFUSE
//  (return 0) WITHOUT launching (no hang). Also exercises the hard failure
//  path: a smem request beyond the device max returns
//  cudaErrorLaunchOutOfResources promptly (asserted in python).
// =========================================================================
static int64_t occupancy_for_smem(int64_t requested_smem_bytes) {
    int numBlocks = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, gemm_single_tile_kernel<128>, 128,
        (size_t)requested_smem_bytes);
    if (e != cudaSuccess) {
        // The query itself rejected the config (e.g. > device max): clear the
        // sticky error and report 0 occupancy (refuse). No kernel was launched.
        cudaGetLastError();
        return 0;
    }
    return (int64_t)numBlocks;
}

// Actually attempt an oversized launch and return the cuda error code (the
// test asserts it is cudaErrorLaunchOutOfResources, i.e. it fails fast, no
// hang). We deliberately request more dynamic smem than any H100 SM has.
static int64_t probe_oversized_launch(int64_t requested_smem_bytes) {
    auto stream = at::cuda::getCurrentCUDAStream();
    // Do NOT set the attribute high enough (the request exceeds the cap); the
    // launch must be rejected at launch time.
    gemm_single_tile_kernel<128><<<1, 128, (size_t)requested_smem_bytes, stream>>>(
        nullptr, nullptr, nullptr, 0);
    cudaError_t e = cudaGetLastError();
    return (int64_t)e;  // python compares against cudaErrorLaunchOutOfResources (7? -> see test)
}

// Report the H100 SM dynamic-smem cap so the test can size the oversized probe.
static int64_t device_max_dynamic_smem() {
    int v = 0;
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (int64_t)v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_tile", &gemm_tile,
          "single-tile / K-loop ss-wgmma bf16 GEMM (m64xNx16), fp32 accum");
    m.def("gemm_tile_pipelined", &gemm_tile_pipelined,
          "pipelined (producer/consumer) ss-wgmma GEMM, same numerics");
    m.def("occupancy_for_smem", &occupancy_for_smem,
          "max active blocks/SM for a dynamic-smem request (0 = refuse)");
    m.def("probe_oversized_launch", &probe_oversized_launch,
          "attempt an oversized-smem launch; returns cuda error code (no hang)");
    m.def("device_max_dynamic_smem", &device_max_dynamic_smem,
          "device cudaDevAttrMaxSharedMemoryPerBlockOptin (bytes)");
}
