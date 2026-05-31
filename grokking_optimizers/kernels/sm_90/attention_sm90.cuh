#ifndef GROKKING_KERNELS_SM90_ATTENTION_SM90_CUH_
#define GROKKING_KERNELS_SM90_ATTENTION_SM90_CUH_
// ============================================================================
// attention_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for the 'attention'
// model. Single source of truth: templated per-layer __device__ forward/backward,
// __global__ launchers, inline-PTX blocks VERBATIM, and the CUTLASS Sm90
// tensor-core GEMM wrappers (attention). Composition primitive for the future
// fused megakernel.
//
// The production location csrc/backends/cuda/sm_90/models/attention.cuh is now a thin
// shim that #include's this header, so every existing includer (bindings.cpp,
// the attention.cu instantiation TU, the HIP tree's references) keeps working
// unchanged. Migrated byte-for-byte; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__/__LINE__).
// ============================================================================
// csrc/kernels/cuda/sm_90/models/attention.cuh
// Shared attention kernel for sm_90 (Hopper). Serves both Decoder (causal,
// seq_len=4) and ViT (non-causal, seq_len=17) via the kCausal template flag.
//
// At these tiny sequence lengths FlashAttention's block-wise tiling adds
// overhead with no benefit. Instead we compute the full QK^T score matrix
// in SMEM/registers, run softmax in-place, then multiply by V.
//
// BF16 activations, FP32 accumulation for matmuls.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

#ifdef WITH_CUTLASS
// ── Sm90 (Hopper) warp-group collective GEMM headers ──────────────────────
// Used by cutlass_fmha_forward below. The fused-MHA is realised as two Sm90
// GemmUniversal calls (S = Q·Kᵀ, then O = P·V) with a softmax kernel between,
// all FP32-accumulate. This emits real WGMMA/TMA instructions; the previous
// FMHA path relied on the default the default device::Gemm (no arch tag) which has
// no arch tag and silently compiled the Sm70 SIMT kernel (no tensor cores).
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>
#endif // WITH_CUTLASS

namespace sg { namespace sm90 { namespace models { namespace attention {

// ── Launch configuration descriptor ────────────────────────────────────
template <typename ActT, int kHeadDim, bool kCausal>
struct AttentionLaunchConfig {
    int block;
    int cluster_size;
    int smem_bytes;
    bool use_fa3;
    bool use_cutlass_fmha;
};

// ── Forward declaration for external backends ──────────────────────────
#ifdef WITH_FLASH_ATTN_3
template <typename ActT, int kHeadDim, bool kCausal>
cudaError_t flash_attn3_forward(
    const ActT* q, const ActT* k, const ActT* v, ActT* out, ActT* softmax_lse,
    int batch, int n_heads, int seq_len, float scale, cudaStream_t stream);
#endif

#ifdef WITH_CUTLASS
template <typename ActT, int kHeadDim, bool kCausal>
cudaError_t cutlass_fmha_forward(
    const ActT* q, const ActT* k, const ActT* v, ActT* out, ActT* softmax_lse,
    int batch, int n_heads, int seq_len, float scale, cudaStream_t stream);
#endif

// ── SMEM-based attention kernel (default for tiny seq_len) ─────────────
// One block per (batch, head). Each thread cooperates on the matmul.
// Max seq_len supported: 32 (17x17 scores = 289 floats in SMEM).

template <typename ActT, int kHeadDim, bool kCausal>
__global__ void __launch_bounds__(128, 4)
smem_attention_fwd_kernel(
    const ActT* __restrict__ q,     // [B, H, N, D]
    const ActT* __restrict__ k,
    const ActT* __restrict__ v,
    ActT* __restrict__ out,
    float* __restrict__ softmax_lse, // [B, H, N] or nullptr
    int seq_len, float scale
) {
    const int bh = blockIdx.x;              // flattened (batch, head) index
    const int tid = threadIdx.x;
    const int N = seq_len;
    const int D = kHeadDim;
    const int base = bh * N * D;

    // Shared memory: scores[N][N] + row_max[N] + row_sum[N]
    extern __shared__ float smem[];
    float* scores  = smem;                  // N * N
    float* row_max = scores + N * N;        // N
    float* row_sum = row_max + N;           // N

    // Step 1: Compute S = Q K^T * scale, with optional causal mask
    for (int idx = tid; idx < N * N; idx += blockDim.x) {
        int i = idx / N;  // query row
        int j = idx % N;  // key column
        if constexpr (kCausal) {
            if (j > i) { scores[idx] = -1e9f; continue; }
        }
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += static_cast<float>(q[base + i * D + d])
                 * static_cast<float>(k[base + j * D + d]);
        }
        scores[idx] = dot * scale;
    }
    __syncthreads();

    // Step 2: Row-wise softmax (online stable via max subtraction)
    for (int i = tid; i < N; i += blockDim.x) {
        float m = -1e9f;
        for (int j = 0; j < N; j++) m = fmaxf(m, scores[i * N + j]);
        row_max[i] = m;
        float s = 0.0f;
        for (int j = 0; j < N; j++) {
            float e = ptx_expf(scores[i * N + j] - m);
            scores[i * N + j] = e;
            s += e;
        }
        float inv_s = 1.0f / fmaxf(s, 1e-12f);
        for (int j = 0; j < N; j++) scores[i * N + j] *= inv_s;
        row_sum[i] = s;  // keep for lse
        if (softmax_lse != nullptr)
            softmax_lse[bh * N + i] = m + logf(fmaxf(s, 1e-12f));
    }
    __syncthreads();

    // Step 3: Out = Softmax(S) * V
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int i = idx / D;
        int d = idx % D;
        float acc = 0.0f;
        for (int j = 0; j < N; j++) {
            acc += scores[i * N + j] * static_cast<float>(v[base + j * D + d]);
        }
        out[base + idx] = static_cast<ActT>(acc);
    }
}

#ifdef WITH_CUTLASS
// ─────────────────────────────────────────────────────────────────────────
//  Sm90 collective FMHA forward (TMA + WGMMA, FP32 accumulate)
//
//  Implemented as two Sm90 GemmUniversal calls per (batch, head):
//      S = Q · Kᵀ          (M=N=seq_len, K=head_dim)   -> FP32 scores
//      softmax(S * scale)  (causal mask if kCausal)     -> FP32 probs P
//      O = P · V           (M=seq_len, N=head_dim, K=seq_len)
//  All matmuls accumulate in FP32 and emit Hopper WGMMA/TMA instructions via
//  CollectiveBuilder<arch::Sm90, OpClassTensorOp, ...>. This replaces the old
//  the default device::Gemm (no arch tag) default which compiled the Sm70 SIMT path.
//
//  A full single-kernel Sm90 fused-MHA collective is intentionally not inlined
//  here (too large); the two-GEMM + softmax decomposition is clearly correct
//  and shares the same WGMMA/TMA mainloop. The tiny-seq_len SMEM fallback
//  kernel (non-CUTLASS path) is left untouched.
// ─────────────────────────────────────────────────────────────────────────

// Element trait: map activation type -> CUTLASS input element.
template <typename ActT> struct cutlass_elem;
template <> struct cutlass_elem<__half>        { using type = cutlass::half_t; };
template <> struct cutlass_elem<__nv_bfloat16> { using type = cutlass::bfloat16_t; };

// The Sm90 collective FMHA path only supports the CUTLASS half/bf16 element
// types (above). FP32 activations (ActT=float, used by the decoder/ViT
// instantiation TUs) fall back to the SMEM attention kernel — see the dispatch
// in attention_forward. This trait lets that dispatch be resolved at compile
// time without instantiating cutlass_fmha_forward<float> (which would require a
// non-existent cutlass_elem<float> and an FP32-input CUTLASS GEMM).
template <typename ActT> struct cutlass_fmha_supported { static constexpr bool value = false; };
template <> struct cutlass_fmha_supported<__half>        { static constexpr bool value = true; };
template <> struct cutlass_fmha_supported<__nv_bfloat16> { static constexpr bool value = true; };

// Per-thread CUTLASS workspace (lazily grown). Reused across calls.
inline void* fmha_get_workspace(size_t bytes) {
    static thread_local void*  ws_ptr   = nullptr;
    static thread_local size_t ws_bytes = 0;
    if (bytes > ws_bytes) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, bytes) != cudaSuccess) {
            ws_ptr = nullptr; ws_bytes = 0; return nullptr;
        }
        ws_bytes = bytes;
    }
    return ws_ptr;
}

// Generic Sm90 GEMM: C[MxN] = A[MxK] * B[KxN], FP32 out, FP32 accumulate.
// A is RowMajor [M,K]. B layout is a template parameter:
//   - ColumnMajor B + physical [N,K] row-major data  => computes A·Bᵀ (QKᵀ)
//   - RowMajor    B + physical [K,N] row-major data  => computes A·B  (P·V)
// strideB_dims = logical {N, K, 1} so the packed stride matches the physical
// row-major [N,K] (QKᵀ) / [K,N] (PV) buffer respectively.
template <typename ElementInput, typename LayoutBT>
inline cudaError_t fmha_sm90_gemm(
    int M, int N, int K,
    const void* A, const void* B, float* C,
    cudaStream_t stream)
{
    using ElementA   = ElementInput;
    using ElementB   = ElementInput;
    using ElementC   = float;
    using ElementAcc = float;
    using LayoutA    = cutlass::layout::RowMajor;
    using LayoutB    = LayoutBT;
    using LayoutC    = cutlass::layout::RowMajor;

    constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;

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
        CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;

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
    if (op.can_implement(args) != cutlass::Status::kSuccess)
        return cudaErrorNotSupported;
    size_t ws_size = Gemm::get_workspace_size(args);
    void*  ws = (ws_size > 0) ? fmha_get_workspace(ws_size) : nullptr;
    if (ws_size > 0 && ws == nullptr) return cudaErrorMemoryAllocation;
    if (op.initialize(args, ws, stream) != cutlass::Status::kSuccess)
        return cudaErrorUnknown;
    return (op.run(stream) == cutlass::Status::kSuccess)
               ? cudaSuccess : cudaErrorUnknown;
}

// In-place row-wise softmax of an [N x N] FP32 score matrix, scaled, with
// optional causal mask. Casts the result into ActT probs buffer P.
template <typename ActT, bool kCausal>
__global__ void fmha_softmax_kernel(
    const float* __restrict__ S,   // [N, N] raw scores (Q·Kᵀ)
    ActT* __restrict__ P,          // [N, N] softmax probabilities (ActT)
    float* __restrict__ lse,       // [N] log-sum-exp or nullptr
    int N, float scale)
{
    int i = blockIdx.x;            // query row
    if (i >= N) return;
    extern __shared__ float sh[];  // N floats
    int tid = threadIdx.x;

    // load + mask + find max
    float m = -1e30f;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = S[i * N + j] * scale;
        if (kCausal && j > i) v = -1e30f;
        sh[j] = v;
        m = fmaxf(m, v);
    }
    __syncthreads();
    // block-wide max reduction (simple shared-mem tree over warps)
    __shared__ float red[32];
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        m = fmaxf(m, SHFL_DOWN_SYNC(FULL_WARP_MASK, m, o));
    if ((tid & (WARP_SIZE - 1)) == 0) red[tid / WARP_SIZE] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = -1e30f;
        int nwarp = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int w = 0; w < nwarp; w++) mm = fmaxf(mm, red[w]);
        red[0] = mm;
    }
    __syncthreads();
    float row_max = red[0];

    // exp + sum
    float s = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float e = ptx_expf(sh[j] - row_max);
        sh[j] = e;
        s += e;
    }
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        s += SHFL_DOWN_SYNC(FULL_WARP_MASK, s, o);
    if ((tid & (WARP_SIZE - 1)) == 0) red[tid / WARP_SIZE] = s;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        int nwarp = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int w = 0; w < nwarp; w++) ss += red[w];
        red[0] = ss;
    }
    __syncthreads();
    float row_sum = red[0];
    float inv = 1.0f / fmaxf(row_sum, 1e-12f);

    for (int j = tid; j < N; j += blockDim.x)
        P[i * N + j] = static_cast<ActT>(sh[j] * inv);
    if (lse != nullptr && tid == 0)
        lse[i] = row_max + logf(fmaxf(row_sum, 1e-12f));
}

// Cast FP32 O accumulator -> ActT output.
template <typename ActT>
__global__ void fmha_cast_kernel(const float* __restrict__ src,
                                 ActT* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = static_cast<ActT>(src[idx]);
}

template <typename ActT, int kHeadDim, bool kCausal>
cudaError_t cutlass_fmha_forward(
    const ActT* q, const ActT* k, const ActT* v, ActT* out, ActT* softmax_lse,
    int batch, int n_heads, int seq_len, float scale, cudaStream_t stream)
{
    using Elem = typename cutlass_elem<ActT>::type;
    const int N  = seq_len;
    const int D  = kHeadDim;
    const int BH = batch * n_heads;
    float* lse = reinterpret_cast<float*>(softmax_lse);

    // Scratch: FP32 scores [N*N], ActT probs [N*N], FP32 O [N*D].
    float* S = nullptr;  ActT* P = nullptr;  float* O = nullptr;
    if (cudaMalloc(&S, sizeof(float) * (size_t)N * N) != cudaSuccess)
        return cudaErrorMemoryAllocation;
    if (cudaMalloc(&P, sizeof(ActT) * (size_t)N * N) != cudaSuccess) {
        cudaFree(S); return cudaErrorMemoryAllocation;
    }
    if (cudaMalloc(&O, sizeof(float) * (size_t)N * D) != cudaSuccess) {
        cudaFree(S); cudaFree(P); return cudaErrorMemoryAllocation;
    }

    cudaError_t err = cudaSuccess;
    for (int bh = 0; bh < BH && err == cudaSuccess; ++bh) {
        const ActT* qh = q + (size_t)bh * N * D;
        const ActT* kh = k + (size_t)bh * N * D;
        const ActT* vh = v + (size_t)bh * N * D;
        ActT*       oh = out + (size_t)bh * N * D;

        // S[N,N] = Q[N,D] · Kᵀ[D,N]. K is physically row-major [N,D]. Declaring
        // the B operand ColumnMajor over logical {N(=GEMM-N), D(=GEMM-K)} makes
        // CUTLASS read that same buffer as Kᵀ, so we get S[i,j]=Σ_d Q[i,d]K[j,d].
        err = fmha_sm90_gemm<Elem, cutlass::layout::ColumnMajor>(
            N, N, D, qh, kh, S, stream);
        if (err != cudaSuccess) break;

        // softmax over rows (scaled, optional causal mask) -> P (ActT)
        int sm_block = 128;
        size_t sm_smem = sizeof(float) * (size_t)N;
        fmha_softmax_kernel<ActT, kCausal>
            <<<N, sm_block, sm_smem, stream>>>(
                S, P, lse ? lse + (size_t)bh * N : nullptr, N, scale);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;

        // O[N,D] = P[N,N] · V[N,D]. V is row-major [N,D] = logical [K=N, N=D],
        // so the B operand is RowMajor and read directly.
        err = fmha_sm90_gemm<Elem, cutlass::layout::RowMajor>(
            N, D, N, P, vh, O, stream);
        if (err != cudaSuccess) break;

        int total = N * D, cb = 256, cg = (total + cb - 1) / cb;
        fmha_cast_kernel<ActT><<<cg, cb, 0, stream>>>(O, oh, total);
        err = cudaGetLastError();
    }

    cudaFree(S); cudaFree(P); cudaFree(O);
    return err;
}
#endif // WITH_CUTLASS

// ── Forward dispatch ───────────────────────────────────────────────────
template <typename ActT, int kHeadDim, bool kCausal>
cudaError_t attention_forward(
    const ActT* q, const ActT* k, const ActT* v,
    ActT* out,
    ActT* softmax_lse_act,  // only used to locate the float buffer
    int batch, int n_heads, int seq_len,
    float scale,
    cudaStream_t stream
) {
    // softmax_lse is always FP32 regardless of ActT
    float* softmax_lse = reinterpret_cast<float*>(softmax_lse_act);

#ifdef WITH_FLASH_ATTN_3
    return flash_attn3_forward<ActT, kHeadDim, kCausal>(
        q, k, v, out, reinterpret_cast<ActT*>(softmax_lse),
        batch, n_heads, seq_len, scale, stream);
#elif defined(WITH_CUTLASS)
    // The Sm90 collective FMHA only supports half/bf16 inputs; FP32 activations
    // (ActT=float) use the SMEM attention path. Resolved at compile time so
    // cutlass_fmha_forward<float> is never instantiated.
    if constexpr (cutlass_fmha_supported<ActT>::value) {
        return cutlass_fmha_forward<ActT, kHeadDim, kCausal>(
            q, k, v, out, reinterpret_cast<ActT*>(softmax_lse),
            batch, n_heads, seq_len, scale, stream);
    } else {
        int grid = batch * n_heads;
        int block = 128;
        int N = seq_len;
        int smem_bytes = (N * N + 2 * N) * sizeof(float);
        smem_attention_fwd_kernel<ActT, kHeadDim, kCausal>
            <<<grid, block, smem_bytes, stream>>>(
                q, k, v, out, softmax_lse, seq_len, scale);
        return cudaGetLastError();
    }
#else
    int grid = batch * n_heads;
    int block = 128;
    int N = seq_len;
    int smem_bytes = (N * N + 2 * N) * sizeof(float);
    smem_attention_fwd_kernel<ActT, kHeadDim, kCausal>
        <<<grid, block, smem_bytes, stream>>>(
            q, k, v, out, softmax_lse, seq_len, scale);
    return cudaGetLastError();
#endif
}

// ── Backward kernel ────────────────────────────────────────────────────
// Recomputes attention weights from softmax_lse (log-sum-exp saved in fwd).
// dV = A^T dO, dA = dO V^T, backprop through softmax, dQ = dA' K * scale,
// dK = dA'^T Q * scale. All in SMEM for these tiny sequence lengths.

template <typename ActT, int kHeadDim, bool kCausal>
__global__ void __launch_bounds__(128, 4)
smem_attention_bwd_kernel(
    const ActT* __restrict__ grad_out,   // [B, H, N, D]
    const ActT* __restrict__ q,
    const ActT* __restrict__ k,
    const ActT* __restrict__ v,
    const ActT* __restrict__ attn_out,   // saved forward output
    const float* __restrict__ softmax_lse, // [B, H, N]
    ActT* __restrict__ grad_q,
    ActT* __restrict__ grad_k,
    ActT* __restrict__ grad_v,
    int seq_len, float scale
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int N = seq_len;
    const int D = kHeadDim;
    const int base = bh * N * D;

    extern __shared__ float smem[];
    float* scores = smem;               // N * N  (attention weights)
    float* dA     = scores + N * N;     // N * N  (grad through attn weights)

    // Recompute attention weights from softmax_lse
    for (int idx = tid; idx < N * N; idx += blockDim.x) {
        int i = idx / N, j = idx % N;
        if constexpr (kCausal) {
            if (j > i) { scores[idx] = 0.0f; continue; }
        }
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
            dot += static_cast<float>(q[base + i * D + d])
                 * static_cast<float>(k[base + j * D + d]);
        float lse = softmax_lse[bh * N + i];
        scores[idx] = ptx_expf(dot * scale - lse);
    }
    __syncthreads();

    // dV = A^T dO  (accumulated directly to global)
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int j = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int i = 0; i < N; i++) acc += scores[i * N + j]
            * static_cast<float>(grad_out[base + i * D + d]);
        grad_v[base + idx] = static_cast<ActT>(acc);
    }
    __syncthreads();

    // dA = dO V^T
    for (int idx = tid; idx < N * N; idx += blockDim.x) {
        int i = idx / N, j = idx % N;
        float acc = 0.0f;
        for (int d = 0; d < D; d++)
            acc += static_cast<float>(grad_out[base + i * D + d])
                 * static_cast<float>(v[base + j * D + d]);
        dA[idx] = acc;
    }
    __syncthreads();

    // Backprop through softmax: dS_ij = A_ij * (dA_ij - sum_k(A_ik * dA_ik))
    for (int i = tid; i < N; i += blockDim.x) {
        float dot_sum = 0.0f;
        for (int j = 0; j < N; j++) dot_sum += scores[i * N + j] * dA[i * N + j];
        for (int j = 0; j < N; j++) {
            float ds = scores[i * N + j] * (dA[i * N + j] - dot_sum) * scale;
            if constexpr (kCausal) { if (j > i) ds = 0.0f; }
            dA[i * N + j] = ds;   // reuse dA storage for dS
        }
    }
    __syncthreads();

    // dQ = dS K
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int i = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int j = 0; j < N; j++)
            acc += dA[i * N + j] * static_cast<float>(k[base + j * D + d]);
        grad_q[base + idx] = static_cast<ActT>(acc);
    }

    // dK = dS^T Q
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int j = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int i = 0; i < N; i++)
            acc += dA[i * N + j] * static_cast<float>(q[base + i * D + d]);
        grad_k[base + idx] = static_cast<ActT>(acc);
    }
}

// ── Backward dispatch ──────────────────────────────────────────────────
template <typename ActT, int kHeadDim, bool kCausal>
cudaError_t attention_backward(
    const ActT* grad_out,
    const ActT* q, const ActT* k, const ActT* v,
    const ActT* out, const ActT* softmax_lse_act,
    ActT* grad_q, ActT* grad_k, ActT* grad_v,
    int batch, int n_heads, int seq_len,
    float scale,
    cudaStream_t stream
) {
    const float* softmax_lse = reinterpret_cast<const float*>(softmax_lse_act);
    int grid = batch * n_heads;
    int block = 128;
    int N = seq_len;
    int smem_bytes = 2 * N * N * sizeof(float);  // scores + dA
    smem_attention_bwd_kernel<ActT, kHeadDim, kCausal>
        <<<grid, block, smem_bytes, stream>>>(
            grad_out, q, k, v, out, softmax_lse,
            grad_q, grad_k, grad_v, seq_len, scale);
    return cudaGetLastError();
}

}}}}  // namespace sg::sm90::models::attention

#endif  // GROKKING_KERNELS_SM90_ATTENTION_SM90_CUH_
