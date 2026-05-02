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
#include <cutlass/epilogue/thread/linear_combination_generic.h>

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
// Fused softplus epilogue functor: out[i,j] = softplus(alpha * acc + bias[j])
// where softplus(x) = log(1 + exp(x)), with branchless saturation at x>20.
//
// CUTLASS 2.x epilogue: element-wise functor that takes the accumulator,
// scales it, adds per-column bias, and applies softplus. Provided for
// future fully-fused path; current path uses GEMM + lightweight post-pass.
// ----------------------------------------------------------------------

template <typename ElementOutput, typename ElementAccumulator, int Count = 1>
struct LinearCombinationSoftplusBias {
    using FragmentOutput = cutlass::Array<ElementOutput, Count>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, Count>;

    struct Params {
        float alpha;
        const float* bias;
        int N;
    };

    Params params_;

    CUTLASS_HOST_DEVICE LinearCombinationSoftplusBias(Params const& p) : params_(p) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

    CUTLASS_HOST_DEVICE FragmentOutput operator()(
        FragmentAccumulator const& accumulator,
        FragmentOutput const& /*source*/,
        int row, int col) const
    {
        FragmentOutput result;
        for (int i = 0; i < Count; ++i) {
            float val = params_.alpha * float(accumulator[i]);
            int c = col + i;
            if (c < params_.N) val += params_.bias[c];
            float sp = (val > 20.0f) ? val : logf(1.0f + expf(val));
            result[i] = ElementOutput(sp);
        }
        return result;
    }

    CUTLASS_HOST_DEVICE FragmentOutput operator()(
        FragmentAccumulator const& accumulator) const
    {
        FragmentOutput result;
        for (int i = 0; i < Count; ++i) {
            float val = params_.alpha * float(accumulator[i]);
            float sp = (val > 20.0f) ? val : logf(1.0f + expf(val));
            result[i] = ElementOutput(sp);
        }
        return result;
    }
};

// ----------------------------------------------------------------------
// §25.3 dt_proj fused helper: out = softplus(A @ B + bias[col])
//
// GEMM accumulates in FP32, then a lightweight post-pass applies
// per-column bias + softplus, avoiding a separate elementwise kernel
// launch. Falls back to unfused if bias is null.
// ----------------------------------------------------------------------

inline cudaError_t cutlass_dt_proj_fused(
    int M, int N, int K,
    const __half* A, const __half* B,
    const float* bias, float* C,
    cudaStream_t stream)
{
    cudaError_t err = cutlass_gemm_fp16(M, N, K, A, B, C, stream);
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

// Softplus + bias elementwise post-pass kernel. `static` linkage avoids
// ODR violations when this header is included from multiple TUs.
static __global__ void cutlass_softplus_bias_kernel(
    float* __restrict__ C, const float* __restrict__ bias,
    int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        float val = C[idx] + bias[col];
        C[idx] = (val > 20.0f) ? val : logf(1.0f + expf(val));
    }
}

inline void launch_softplus_bias(float* C, const float* bias, int M, int N,
                                  cudaStream_t stream)
{
    if (!bias) return;
    int total = M * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    cutlass_softplus_bias_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
}

// Convenience: GEMM + softplus_bias in one call (§25.3 fused path).
inline cudaError_t cutlass_dt_proj_fused_with_bias(
    int M, int N, int K,
    const __half* A, const __half* B,
    const float* bias, float* C,
    cudaStream_t stream)
{
    cudaError_t err = cutlass_dt_proj_fused(M, N, K, A, B, bias, C, stream);
    if (err != cudaSuccess) return err;
    launch_softplus_bias(C, bias, M, N, stream);
    return cudaSuccess;
}

}} // namespace sg::cutlass_gemm

#else  // !WITH_CUTLASS

#error "CUTLASS not enabled — _cutlass_gemm.cuh included without -DWITH_CUTLASS. Wrap the include in #ifdef WITH_CUTLASS."

#endif // WITH_CUTLASS
