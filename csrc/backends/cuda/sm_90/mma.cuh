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

#include <cstdint>
#include <mutex>

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

// ─────────────────────────────────────────────────────────────────────────
//  §3.1 — TMA-descriptor / operator reuse (shape-keyed host cache)
//
//  WHAT IS REUSED (and WHY this is the correct CUTLASS-3.6 reuse):
//  ----------------------------------------------------------------
//  CUTLASS 3.6 builds the Hopper TMA tensor-map descriptors *inside*
//  `op.initialize(args, ws, stream)` → `GemmKernel::to_underlying_arguments`
//  → `CollectiveMainloop::to_underlying_arguments`, which calls
//  `make_tma_copy_A/B_sm90(...)` (sm90_mma_tma_gmma_ss_warpspecialized.hpp
//  lines 215-226). Those descriptors (one cuTensorMapEncode each for A and B,
//  plus the epilogue's C map) are baked into the operator's `params_` member.
//  GemmUniversalAdapter exposes `run(stream)` precisely "to re-launch the same
//  kernel without updating internal params" (gemm_universal_adapter.h:545) —
//  i.e. reusing the already-built descriptors.
//
//  So the supported, internals-respecting reuse is NOT to reach into CUTLASS's
//  TMA path, but to CACHE THE INITIALIZED OPERATOR keyed by everything its
//  descriptors encode, and on a recurring shape skip
//  make_cute_packed_stride + Arguments construction + get_workspace_size +
//  initialize (which is where the cuTensorMapEncode calls live) and just call
//  the cached `op.run(stream)`.
//
//  DESCRIPTOR ADDRESS-STABILITY (correctness invariant):
//  ----------------------------------------------------------------
//  A Hopper TMA tensor map encodes the *global base address* of the tensor
//  (make_tma_copy is fed `make_tensor(ptr_A, ...)`), the box/shape and the
//  strides. Therefore a cached operator is bit-for-bit equivalent to a freshly
//  rebuilt one IFF (M,N,K), the A/B/C base pointers, AND the strides all match.
//  Strides here are a pure function of (M,N,K) (row-major packed), so the key
//  is {M,N,K, A, B, C}. The optimizer/model buffers this serves (param/m/v
//  state, weights, and the grown-once FP32 scratch) are allocated ONCE and keep
//  the same address every step → stable key → cache hit. If any buffer is
//  reallocated to a new address, the key changes → MISS → rebuild, so a stale
//  descriptor can never be launched against a moved buffer (that would be wrong
//  results / an illegal access). This is the load-bearing safety property.
//
//  WORKSPACE LIFETIME:
//  ----------------------------------------------------------------
//  For this builder (1x1x1 cluster, kGemm, no split-K) the kernel reports
//  get_workspace_size()==0 and the adapter adds a barrier workspace only when
//  cute::size(ClusterShape)>1 — so the served GEMMs need no workspace and
//  `initialize` is called with ws==nullptr. We still query the size and, if a
//  future tile/cluster change makes it non-zero, each cache entry OWNS its own
//  workspace allocation (freed on eviction) so a cached operator's params can
//  never dangle into the shared grow-only `sm90_get_workspace` buffer.
//
//  EVICTION: fixed-size, direct-mapped (hash → slot). On collision the resident
//  entry is evicted (its owned workspace freed) and replaced — bounded memory,
//  no leak. Only a handful of distinct (shape,ptr) signatures recur per run, so
//  collisions are rare.
//
//  THREAD-SAFETY: one static cache per template instantiation, guarded by a
//  std::mutex. The launcher is single-threaded per stream in this codebase, but
//  the mutex makes concurrent host launchers safe and is off the hot path
//  (host-side, microseconds, dwarfed by the kernel it replaces rebuilding).
//
//  ELECT/MBARRIER BOUNDARY (honest note):
//  ----------------------------------------------------------------
//  §3.4's elect_one_sync()/Mbarrier (warp_specialize.cuh) compose a
//  HAND-WRITTEN producer/consumer TMA staging loop. The CUTLASS GEMM is a
//  self-contained kernel that OWNS its TMA mainloop — there is no hand-written
//  TMA staging in mma.cuh to inject a leader/barrier into. Pairing the elect/
//  mbarrier primitives with TMA therefore belongs to the Stage-6 megakernel's
//  hand-written TMA path, NOT here. Stage 4.1 is the host-side operator/
//  descriptor cache only.
// ─────────────────────────────────────────────────────────────────────────

// Problem signature the cached descriptors depend on. Pointers are part of the
// key because a TMA tensor map encodes the global base address (see above).
struct GemmCacheKey {
    int         M, N, K;
    const void* A;
    const void* B;
    void*       C;

    bool operator==(const GemmCacheKey& o) const {
        return M == o.M && N == o.N && K == o.K &&
               A == o.A && B == o.B && C == o.C;
    }
};

inline uint64_t sm90_gemm_key_hash(const GemmCacheKey& k) {
    // FNV-1a over the raw key bytes — cheap, decent spread for the small set
    // of distinct signatures that recur.
    auto mix = [](uint64_t h, uint64_t v) {
        h ^= v; h *= 0x100000001b3ull; return h;
    };
    uint64_t h = 0xcbf29ce484222325ull;
    h = mix(h, (uint64_t)(uint32_t)k.M);
    h = mix(h, (uint64_t)(uint32_t)k.N);
    h = mix(h, (uint64_t)(uint32_t)k.K);
    h = mix(h, (uint64_t)k.A);
    h = mix(h, (uint64_t)k.B);
    h = mix(h, (uint64_t)k.C);
    return h;
}

// Per-Gemm-instantiation, fixed-size, direct-mapped cache of INITIALIZED
// operators. Holds the operator (with its baked-in TMA descriptors / params_)
// and the workspace it was initialized with. The operator is stored by value;
// GemmUniversalAdapter is just a `Params params_` holder (copy/move-able) and
// is re-launched via op.run(stream), which reuses params_ unchanged.
template <typename Gemm>
struct Sm90GemmCache {
    static constexpr int kSlots = 16;   // bounded; covers the recurring shapes

    struct Entry {
        bool         valid = false;
        GemmCacheKey key{};
        Gemm         op{};
        void*        ws       = nullptr;   // OWNED by this entry (may be null)
        size_t       ws_bytes = 0;
    };

    Entry      slots[kSlots];
    std::mutex mu;

    static Sm90GemmCache& instance() {
        static Sm90GemmCache c;
        return c;
    }
};

template <typename Gemm>
inline int sm90_gemm_slot(const GemmCacheKey& k) {
    return (int)(sm90_gemm_key_hash(k) & (uint64_t)(Sm90GemmCache<Gemm>::kSlots - 1));
}

// Build a row-major-packed Arguments for (M,N,K) with the given A/B/C and run
// it, caching the initialized operator so a recurring (shape,ptr) signature
// skips make_stride + Arguments + initialize (and thus the cuTensorMapEncode
// calls) and re-launches the cached op directly.
template <typename Gemm, typename ElementA, typename ElementB, typename ElementAcc>
inline cudaError_t sm90_run_gemm_cached(
    int M, int N, int K,
    const void* A, const void* B, float* C,
    cudaStream_t stream)
{
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;

    GemmCacheKey key{M, N, K, A, B, C};
    auto& cache = Sm90GemmCache<Gemm>::instance();
    int   slot  = sm90_gemm_slot<Gemm>(key);

    std::lock_guard<std::mutex> guard(cache.mu);
    auto& e = cache.slots[slot];

    // HIT: identical (shape,ptr) signature → cached descriptors are valid.
    if (e.valid && e.key == key) {
        cutlass::Status st = e.op.run(stream);
        return (st == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
    }

    // MISS (empty or colliding signature): build + initialize a fresh operator.
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

    // Evict the resident colliding entry (free its OWNED workspace) before reuse.
    if (e.valid && e.ws) {
        cudaFree(e.ws);
        e.ws = nullptr;
        e.ws_bytes = 0;
    }

    void* ws = nullptr;
    if (ws_size > 0) {
        // Entry owns its workspace so the cached operator's params can never
        // dangle into the shared grow-only buffer.
        if (cudaMalloc(&ws, ws_size) != cudaSuccess) {
            e.valid = false;
            return cudaErrorMemoryAllocation;
        }
    }

    st = op.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) {
        if (ws) cudaFree(ws);
        e.valid = false;
        return cudaErrorUnknown;
    }

    st = op.run(stream);
    if (st != cutlass::Status::kSuccess) {
        if (ws) cudaFree(ws);
        e.valid = false;
        return cudaErrorUnknown;
    }

    // Install into the cache for subsequent recurring invocations.
    e.op       = op;          // copies params_ (baked TMA descriptors)
    e.key      = key;
    e.ws       = ws;
    e.ws_bytes = ws_size;
    e.valid    = true;
    return cudaSuccess;
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

    // §3.1: serve from the shape-keyed operator cache. A recurring
    // (M,N,K, A,B,C) signature reuses the already-built TMA descriptors and
    // skips make_stride + Arguments + initialize (the cuTensorMapEncode path);
    // a miss builds + initializes + installs. Numerically identical to the
    // rebuild-every-time path (same args → same params_).
    return sm90_run_gemm_cached<Gemm, ElementA, ElementB, ElementAcc>(
        M, N, K, A, B, C, stream);
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

    // §3.1: same shape-keyed operator cache as sm90_run_gemm. The cache is
    // per-Gemm-instantiation, so each (ElementInput, LayoutBT) flavour gets its
    // own slot table — a RowMajor-B descriptor is never confused with a
    // ColumnMajor-B (Bᵀ) one even at identical (M,N,K,ptrs).
    return sm90_run_gemm_cached<Gemm, ElementA, ElementB, ElementAcc>(
        M, N, K, A, B, C, stream);
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
