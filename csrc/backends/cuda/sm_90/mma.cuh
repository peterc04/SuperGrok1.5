#pragma once
// Canonical header (de-inlined). Body is byte-identical to the
// formerly copy-pasted block; prerequisites are included so that
// platform macros precede their use.
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/scan/affine2x2.h"
#include "csrc/common/utils.cuh"

// CUDA sm_90 matrix-multiply accelerator wrappers (CUTLASS).
// Renamed from csrc/kernels/cuda/_cutlass_gemm.cuh.
//
// Used by Muon (Newton-Schulz GEMMs) and SuperGrok v2 (dt_proj fused
// softplus). Gated behind -DWITH_CUTLASS. Without the flag, Muon falls
// back to cuBLAS (torch::mm) and SG2 uses cuBLAS + a separate softplus
// kernel — slightly slower but fully functional.
//
// Math equivalence: every helper computes C = A * B with FP32 accumulate
// and FP32 output, matching cuBLAS GemmEx with CUBLAS_COMPUTE_32F.

#ifdef WITH_CUTLASS

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>

// ── Sm90 (Hopper) warp-group collective GEMM headers ──────────────────────
// The previous code used the default device::Gemm (no arch tag) which, with no arch
// tag, silently defaults to the SIMT/Sm70 path — i.e. NO tensor cores, NO
// WGMMA, NO TMA. To actually emit Hopper WGMMA/TMA instructions we build a
// GemmUniversalAdapter from the Sm90 CollectiveBuilder mainloop + collective
// epilogue. FP32 accumulate throughout (matches cuBLAS CUBLAS_COMPUTE_32F).
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>

namespace sg { namespace sm90 { namespace mma {

// ─────────────────────────────────────────────────────────────────────────
//  Sm90 collective GEMM (TMA + WGMMA, FP32 accumulate)
//
//  Generic builder parameterised on the input element type. Row-major A,
//  row-major B, row-major C with FP32 output. C = alpha*A*B + beta*C.
//
//  Why this exists: the old the default device::Gemm (no arch tag) instantiation
//  carried no arch tag and therefore compiled the SIMT (Sm70) kernel — no
//  tensor cores at all. GemmUniversalAdapter<GemmUniversal<...>> built from
//  CollectiveBuilder<arch::Sm90, OpClassTensorOp, ...> guarantees the Hopper
//  warp-group MMA + TMA path is emitted.
//
//  GemmUniversalAdapter requires a workspace; we query get_workspace_size()
//  and serve it from a per-thread cached device buffer (grown on demand).
// ─────────────────────────────────────────────────────────────────────────

// Per-thread, lazily grown CUTLASS workspace. Avoids a cudaMalloc per call.
// Lifetime = process lifetime (intentional: the buffer is reused). Sized to
// the largest workspace any GEMM in this thread has requested so far.
inline void* sm90_get_workspace(size_t bytes) {
    static thread_local void*  ws_ptr   = nullptr;
    static thread_local size_t ws_bytes = 0;
    if (bytes > ws_bytes) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, bytes) != cudaSuccess) {
            ws_ptr = nullptr;
            ws_bytes = 0;
            return nullptr;
        }
        ws_bytes = bytes;
    }
    return ws_ptr;
}

template <typename ElementInput>
struct Sm90Gemm {
    using ElementA   = ElementInput;
    using ElementB   = ElementInput;
    using ElementC   = float;          // FP32 output (matches cuBLAS path)
    using ElementAcc = float;          // FP32 accumulate
    using LayoutA    = cutlass::layout::RowMajor;
    using LayoutB    = cutlass::layout::RowMajor;
    using LayoutC    = cutlass::layout::RowMajor;

    // 128-bit aligned access (16 bytes / sizeof(element)).
    static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    // Collective epilogue: linear combination (alpha/beta), FP32 accumulate.
    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            TileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAcc, ElementAcc,
            ElementC, LayoutC, AlignC,
            ElementC, LayoutC, AlignC,
            cutlass::epilogue::collective::EpilogueScheduleAuto
        >::CollectiveOp;

    // Collective mainloop: Sm90 TMA + WGMMA, auto stage count / schedule.
    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            ElementA, LayoutA, AlignA,
            ElementB, LayoutB, AlignB,
            ElementAcc,
            TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            cutlass::gemm::collective::KernelScheduleAuto
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,   // (M, N, K, L) problem shape
        CollectiveMainloop,
        CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Run a single row-major C = A*B with FP32 accumulate on the Sm90 collective.
template <typename ElementInput>
inline cudaError_t sm90_run_gemm(
    int M, int N, int K,
    const void* A, const void* B, float* C,
    cudaStream_t stream)
{
    using G          = Sm90Gemm<ElementInput>;
    using Gemm       = typename G::Gemm;
    using ElementA   = typename G::ElementA;
    using ElementB   = typename G::ElementB;
    using ElementAcc = typename G::ElementAcc;
    using StrideA    = typename Gemm::GemmKernel::StrideA;
    using StrideB    = typename Gemm::GemmKernel::StrideB;
    using StrideC    = typename Gemm::GemmKernel::StrideC;

    // Row-major strides via CUTLASS helpers (L=1 batch).
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { reinterpret_cast<const ElementA*>(A), stride_a,
          reinterpret_cast<const ElementB*>(B), stride_b },
        { {ElementAcc(1.0f), ElementAcc(0.0f)},
          C, stride_c, C, stride_c }
    };

    Gemm op;
    cutlass::Status st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess) return cudaErrorNotSupported;

    size_t ws_size = Gemm::get_workspace_size(args);
    void*  ws = (ws_size > 0) ? sm90_get_workspace(ws_size) : nullptr;
    if (ws_size > 0 && ws == nullptr) return cudaErrorMemoryAllocation;

    st = op.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) return cudaErrorUnknown;

    st = op.run(stream);
    return (st == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// ─────────────────────────────────────────────────────────────────────────
//  LayoutB-parameterised Sm90 collective GEMM (TMA + WGMMA, FP32 accumulate)
//
//  Identical builder to Sm90Gemm above, but the B operand layout is a
//  template parameter. This lets callers express both flavours of the
//  transformer/ViT matmuls with the SAME proven CollectiveBuilder path used
//  by the FMHA wrappers in attention_sm90.cuh:
//    - LayoutBT = cutlass::layout::RowMajor    => C = A · B   (physical [K,N])
//    - LayoutBT = cutlass::layout::ColumnMajor => C = A · Bᵀ  (physical [N,K]
//                                                 row-major, e.g. a Linear
//                                                 weight W[out,in] read as Wᵀ)
//  A is always RowMajor [M,K]; C is RowMajor [M,N]; FP32 accumulate + FP32
//  output (matches the cuBLAS CUBLAS_COMPUTE_32F numerics). strideB packs the
//  logical {N, K, 1} extents so it matches the physical row-major [N,K] (Bᵀ)
//  / [K,N] (B) buffer respectively — same convention as fmha_sm90_gemm.
// ─────────────────────────────────────────────────────────────────────────
template <typename ElementInput, typename LayoutBT>
struct Sm90GemmBT {
    using ElementA   = ElementInput;
    using ElementB   = ElementInput;
    using ElementC   = float;          // FP32 output (matches cuBLAS path)
    using ElementAcc = float;          // FP32 accumulate
    using LayoutA    = cutlass::layout::RowMajor;
    using LayoutB    = LayoutBT;
    using LayoutC    = cutlass::layout::RowMajor;

    static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            TileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAcc, ElementAcc,
            ElementC, LayoutC, AlignC,
            ElementC, LayoutC, AlignC,
            cutlass::epilogue::collective::EpilogueScheduleAuto
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            ElementA, LayoutA, AlignA,
            ElementB, LayoutB, AlignB,
            ElementAcc,
            TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            cutlass::gemm::collective::KernelScheduleAuto
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Run a single C[MxN] = A[MxK] · op(B) with FP32 accumulate on the Sm90
// collective. op(B) is B (LayoutBT=RowMajor) or Bᵀ (LayoutBT=ColumnMajor).
template <typename ElementInput, typename LayoutBT>
inline cudaError_t sm90_run_gemm_bt(
    int M, int N, int K,
    const void* A, const void* B, float* C,
    cudaStream_t stream)
{
    using G          = Sm90GemmBT<ElementInput, LayoutBT>;
    using Gemm       = typename G::Gemm;
    using ElementA   = typename G::ElementA;
    using ElementB   = typename G::ElementB;
    using ElementAcc = typename G::ElementAcc;
    using StrideA    = typename Gemm::GemmKernel::StrideA;
    using StrideB    = typename Gemm::GemmKernel::StrideB;
    using StrideC    = typename Gemm::GemmKernel::StrideC;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { reinterpret_cast<const ElementA*>(A), stride_a,
          reinterpret_cast<const ElementB*>(B), stride_b },
        { {ElementAcc(1.0f), ElementAcc(0.0f)},
          C, stride_c, C, stride_c }
    };

    Gemm op;
    cutlass::Status st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess) return cudaErrorNotSupported;

    size_t ws_size = Gemm::get_workspace_size(args);
    void*  ws = (ws_size > 0) ? sm90_get_workspace(ws_size) : nullptr;
    if (ws_size > 0 && ws == nullptr) return cudaErrorMemoryAllocation;

    st = op.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) return cudaErrorUnknown;

    st = op.run(stream);
    return (st == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// FP16 in / FP32 acc / FP32 out, row-major A * row-major B, row-major C.
// Sm90 collective (TMA+WGMMA), FP32 accumulate — replaces the old Gemm<>
// that silently defaulted to Sm70 SIMT (no tensor cores).
inline cudaError_t gemm_fp16(
    int M, int N, int K,
    const __half* A, const __half* B, float* C,
    cudaStream_t stream)
{
    return sm90_run_gemm<cutlass::half_t>(M, N, K, A, B, C, stream);
}

// BF16 in / FP32 acc / FP32 out variant.
// Sm90 collective (TMA+WGMMA), FP32 accumulate — replaces the old Gemm<>
// that silently defaulted to Sm70 SIMT (no tensor cores).
inline cudaError_t gemm_bf16(
    int M, int N, int K,
    const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
    cudaStream_t stream)
{
    return sm90_run_gemm<cutlass::bfloat16_t>(M, N, K, A, B, C, stream);
}

// Softplus+bias post-pass (used by SG2 dt_proj fused path).
static __global__ void softplus_bias_kernel(
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

inline void launch_softplus_bias(
    float* C, const float* bias, int M, int N,
    cudaStream_t stream)
{
    if (!bias) return;
    int total = M * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    softplus_bias_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
}

// Fused dt_proj: GEMM + softplus+bias in one call.
inline cudaError_t dt_proj_fused_with_bias(
    int M, int N, int K,
    const __half* A, const __half* B,
    const float* bias, float* C,
    cudaStream_t stream)
{
    cudaError_t err = gemm_fp16(M, N, K, A, B, C, stream);
    if (err != cudaSuccess) return err;
    launch_softplus_bias(C, bias, M, N, stream);
    return cudaSuccess;
}

}}} // namespace sg::sm90::mma

#else  // !WITH_CUTLASS

#error "CUTLASS not enabled. Use cuBLAS path or build with -DWITH_CUTLASS."

#endif // WITH_CUTLASS
