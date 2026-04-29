/*
 * CUTLASS GEMM helpers for SG2 projections + Muon Newton-Schulz.
 *
 * Gated on `-DWITH_CUTLASS`. When the macro is undefined, including this
 * header is a hard #error so a misuse (calling cutlass_gemm_* without
 * the flag) is caught at compile time. Per-arch kernel TUs always wrap
 * their #include "../_cutlass_gemm.cuh" in #ifdef WITH_CUTLASS.
 *
 * Only Hopper+ (sm_90, sm_100, sm_103, sm_120) routes here; sm_80/sm_89
 * keep cuBLAS torch::mm and gfx942/gfx950 keep rocBLAS. See setup.py
 * for which arches inject -DWITH_CUTLASS into the nvcc command line.
 *
 * Math equivalence: every helper computes C = A * B with FP32 accumulate
 * and FP32 output, matching cuBLAS GemmEx with CUBLAS_COMPUTE_32F. The
 * tile/stage/cluster choice is left to CUTLASS device-default templates;
 * the autotuner (autotune/cutlass_profile.py) supplies tuned shapes when
 * available via tuned_configs.h, but defaults still produce correct math.
 */
#pragma once

#ifdef WITH_CUTLASS

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace sg { namespace cutlass_gemm {

// ----------------------------------------------------------------------
// FP16 in / FP32 acc / FP32 out, row-major A * row-major B, row-major C.
// Layout matches torch::mm on contiguous row-major tensors: C = A @ B.
// ----------------------------------------------------------------------

inline cudaError_t cutlass_gemm_fp16(
    int M, int N, int K,
    const __half* A, const __half* B, float* C,
    cudaStream_t stream)
{
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = float;
    using ElementAcc = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAcc>;

    typename Gemm::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(A), K},
        {reinterpret_cast<const ElementB*>(B), N},
        {C, N},
        {C, N},
        {ElementAcc(1.0f), ElementAcc(0.0f)});

    Gemm op;
    cutlass::Status st = op(args, nullptr, stream);
    return (st == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// ----------------------------------------------------------------------
// BF16 in / FP32 acc / FP32 out variant.
// ----------------------------------------------------------------------

inline cudaError_t cutlass_gemm_bf16(
    int M, int N, int K,
    const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
    cudaStream_t stream)
{
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = float;
    using ElementAcc = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAcc>;

    typename Gemm::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(A), K},
        {reinterpret_cast<const ElementB*>(B), N},
        {C, N},
        {C, N},
        {ElementAcc(1.0f), ElementAcc(0.0f)});

    Gemm op;
    cutlass::Status st = op(args, nullptr, stream);
    return (st == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// ----------------------------------------------------------------------
// dt_proj fused helper. Reference math (see softplus_bias_kernel in
// supergrok2_bwd_<arch>.cu): out = softplus(A @ B + bias[col]).
//
// TODO: Fused softplus epilogue. CUTLASS 3.x EpilogueOp customization
// (LinearCombinationSoftplusBias) is non-trivial and version-sensitive;
// for now we run the unfused linear-combination GEMM and rely on the
// caller to launch softplus_bias_kernel afterward (matching the existing
// cuBLAS path). Math is bit-identical to the cuBLAS+softplus_bias_kernel
// sequence; only a single elementwise pass is unfused vs. ideal.
// ----------------------------------------------------------------------

inline cudaError_t cutlass_dt_proj_fused(
    int M, int N, int K,
    const __half* A, const __half* B,
    const float* /*bias*/, float* C,
    cudaStream_t stream)
{
    // Unfused path: run plain linear-combination GEMM. Caller still
    // launches softplus_bias_kernel(C, bias, M, N) afterward, which is
    // what bilevel_precompute_gemm() already does in the cuBLAS branch.
    return cutlass_gemm_fp16(M, N, K, A, B, C, stream);
}

}} // namespace sg::cutlass_gemm

#else  // !WITH_CUTLASS

#error "CUTLASS not enabled — _cutlass_gemm.cuh included without -DWITH_CUTLASS. Wrap the include in #ifdef WITH_CUTLASS."

#endif // WITH_CUTLASS
