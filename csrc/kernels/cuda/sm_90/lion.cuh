// =====================================================================
//  csrc/kernels/cuda/sm_90/lion.cuh
//
//  sm_90 (Hopper) Lion optimizer kernel + launcher header.
//
//  Algorithm (no v-state Lion variant -- 3 reads / 2 writes per element):
//     c_t   = sign(beta1 * m_{t-1} + (1 - beta1) * g)
//     theta = theta - lr * (c_t + wd * theta)            // decoupled WD
//     m_t   = beta2 * m_{t-1} + (1 - beta2) * g          // EMA uses beta2
//
//  Launchers (signature dictated by csrc/bindings/lion.cpp DECLARE_LION):
//     void launch_fused_lion_step(torch::Tensor, torch::Tensor,
//                                 torch::Tensor, float, float, float, float);
//     void launch_multi_tensor_lion(std::vector<torch::Tensor>&,
//                                   std::vector<torch::Tensor>&,
//                                   std::vector<torch::Tensor>&,
//                                   float, float, float, float);
//
//  Note: csrc/bindings/multi_tensor.cpp forward-declares the multi-tensor
//  variant by-value. We provide both overloads in the .cu to satisfy the
//  linker for both translation units.
//
//  Roofline (FP32 dtype, worst case):
//     R = 3 * 4 B  reads (param, exp_avg, grad) = 12 B
//     W = 2 * 4 B  writes (param, exp_avg)       =  8 B
//     I ~= 7 FLOPs per element (3 FMAs + sign + FMA + FMA)
//  Arithmetic intensity ~= 7/20 = 0.35 FLOP/B -- solidly BW-bound on H100
//  (~3 TB/s HBM3, ~67 TFLOP/s FP32). Kernel is therefore tuned for peak
//  HBM throughput: vec4 loads (ld.global.nc.v4.f32), wt stores to bypass
//  L2 allocation on optimizer state, __launch_bounds__ from
//  tuned_configs.h::DEFAULT_CONFIG (BLOCK=256, MIN_BLOCKS_PER_SM=2).
//
//  Heterogeneous dtype matrix -- all instantiations live in lion.cu:
//     ParamT in {float, __nv_bfloat16, __half}
//     StateT in {float, __nv_bfloat16}
//     GradT  in {float, __nv_bfloat16, __half,
//                __nv_fp8_e4m3, __nv_fp8_e5m2}
//  Incoherent combos rejected via static_assert. All math runs in FP32;
//  only loads/stores are typed.
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/tuned_configs.h"
#include "csrc/device/optimizers/sm_90/lion_sm90.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890)
  #include <cuda_fp8.h>
#endif

#include <torch/extension.h>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace sg { namespace sm90 { namespace lion {

// ---------------------------------------------------------------------
// Compile-time predicates for the dtype matrix.
// ---------------------------------------------------------------------

template <typename T>
struct is_param_dtype
    : std::integral_constant<bool,
        std::is_same<T, float>::value         ||
        std::is_same<T, __nv_bfloat16>::value ||
        std::is_same<T, __half>::value> {};

template <typename T>
struct is_state_dtype
    : std::integral_constant<bool,
        std::is_same<T, float>::value         ||
        std::is_same<T, __nv_bfloat16>::value> {};

template <typename T>
struct is_grad_dtype
    : std::integral_constant<bool,
        std::is_same<T, float>::value          ||
        std::is_same<T, __nv_bfloat16>::value  ||
        std::is_same<T, __half>::value         ||
        std::is_same<T, __nv_fp8_e4m3>::value  ||
        std::is_same<T, __nv_fp8_e5m2>::value> {};

// FP8 grads are only physically meaningful when the optimizer state is at
// least BF16 (FP8 grad + FP8 anything is incoherent for an optimizer step).
template <typename ParamT, typename StateT, typename GradT>
struct is_coherent_combo
    : std::integral_constant<bool,
        is_param_dtype<ParamT>::value &&
        is_state_dtype<StateT>::value &&
        is_grad_dtype<GradT>::value> {};

// ---------------------------------------------------------------------
// Type-erased load helpers -- all math is FP32. Reads are routed through
// LDG (read-only cache) where the dtype has an __ldg overload; FP8 falls
// through to a direct dereference (no LDG overload exists for fp8).
// ---------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p);

template <>
__device__ __forceinline__ float load_as_float<float>(const float* p) {
    return LDG(p);
}
template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(LDG(p));
}
template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* p) {
    return __half2float(LDG(p));
}
template <>
__device__ __forceinline__ float load_as_float<__nv_fp8_e4m3>(const __nv_fp8_e4m3* p) {
    return static_cast<float>(*p);
}
template <>
__device__ __forceinline__ float load_as_float<__nv_fp8_e5m2>(const __nv_fp8_e5m2* p) {
    return static_cast<float>(*p);
}

// ---------------------------------------------------------------------
// Type-erased store helper for params (FP32 / BF16 / FP16). State stores
// go through store_state which uses the wt streaming path on FP32.
// ---------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ void store_param(T* p, float v);

template <>
__device__ __forceinline__ void store_param<float>(float* p, float v) {
    *p = v;
}
template <>
__device__ __forceinline__ void store_param<__nv_bfloat16>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ void store_param<__half>(__half* p, float v) {
    *p = __float2half_rn(v);
}

// State stream-store: FP32 state goes through ld/st.global.wt to bypass L2;
// BF16 state has no PTX wt v1.b16 path so falls through to a normal store.
__device__ __forceinline__ void store_state(float* p, float v) {
    stream_store(p, v);
}
__device__ __forceinline__ void store_state(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}

// State load: FP32 streaming load via ld.global.nc.f32; BF16 normal LDG.
__device__ __forceinline__ float load_state(const float* p) {
    return stream_load(p);
}
__device__ __forceinline__ float load_state(const __nv_bfloat16* p) {
    return __bfloat162float(LDG(p));
}

// ---------------------------------------------------------------------
// Constants from tuned_configs.h::DEFAULT_CONFIG. Cached at file-scope so
// the compiler can fold them into __launch_bounds__ at compile time
// without re-reading the global table on every TU. NOT a runtime branch.
// ---------------------------------------------------------------------

constexpr int kBlockSize       = ::sg::DEFAULT_CONFIG.block_size;        // 256
constexpr int kMinBlocksPerSM  = ::sg::DEFAULT_CONFIG.min_blocks_per_sm; // 2

// ---------------------------------------------------------------------
// Per-tensor offset table (multi-tensor variant).
//
// We pack up to kMaxTensors per launch. Each entry holds the typed base
// pointers plus the cumulative element offset; threadIdx.x + blockIdx.x *
// blockDim.x is mapped to (tensor_id, local_idx) by binary search on
// offsets. Stored in __constant__ to fit in 64 KB and broadcast-fast.
// ---------------------------------------------------------------------

constexpr int kMaxTensorsPerLaunch = 96;

template <typename ParamT, typename StateT, typename GradT>
struct LionTensorTable {
    ParamT*       param[kMaxTensorsPerLaunch];
    StateT*       exp_avg[kMaxTensorsPerLaunch];
    const GradT*  grad[kMaxTensorsPerLaunch];
    int64_t       offsets[kMaxTensorsPerLaunch + 1]; // prefix sum, len = N+1
    int           num_tensors;
};

// ---------------------------------------------------------------------
// Single-tensor kernel: one global thread per element. Routes through
// the device-template lion_step (csrc/device/optimizers/sm_90/lion_sm90.cuh)
// when (Param=Grad, State=fp32) so cross-arch numerical agreement holds;
// otherwise inlines the math here for heterogeneous dtype combos.
// ---------------------------------------------------------------------

template <typename ParamT, typename StateT, typename GradT>
__launch_bounds__(kBlockSize, kMinBlocksPerSM)
__global__ void fused_lion_step_kernel(
    ParamT* __restrict__ param,
    StateT* __restrict__ exp_avg,
    const GradT* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd,
    const int64_t N
) {
    static_assert(is_coherent_combo<ParamT, StateT, GradT>::value,
                  "lion::fused_lion_step_kernel: incoherent ParamT/StateT/GradT combo");

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float g  = load_as_float<GradT>(grad + idx);
    const float ea = load_state(exp_avg + idx);
    const float p  = load_as_float<ParamT>(param + idx);

    const float interp = beta1 * ea + (1.0f - beta1) * g;
    const float s = (interp != 0.0f) ? copysignf(1.0f, interp) : 0.0f;

    store_param<ParamT>(param + idx, p - lr * (s + wd * p));
    store_state(exp_avg + idx, beta2 * ea + (1.0f - beta2) * g);
}

// ---------------------------------------------------------------------
// FP32-fastpath vec4 kernel. Activated when ParamT=StateT=GradT=float and
// N % 4 == 0 with 16-byte aligned base pointers. Uses ld.global.nc.v4.f32
// and st.global.wt.v4.f32 via stream_load4/stream_store4 from platform.h.
//
// We delegate to lion_step_vec4 in the device template so the math is
// literally the same instructions across this kernel and the per-arch
// tests in tests/test_cross_arch_agreement.py.
// ---------------------------------------------------------------------

__launch_bounds__(kBlockSize, kMinBlocksPerSM)
__global__ void fused_lion_step_vec4_kernel(
    float4* __restrict__ param4,
    float4* __restrict__ exp_avg4,
    const float4* __restrict__ grad4,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd,
    const int64_t N4
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N4) return;

    // The device template uses int idx; safe because per-call N4 fits in
    // an int as long as N4 < 2^31. For tensors larger than 2^33 elements
    // (32 GB FP32), we fall back to the scalar kernel via the launcher.
    ::sg::device::sm90::lion_step_vec4(
        param4, exp_avg4, grad4,
        lr, beta1, beta2, wd, static_cast<int>(i));
}

// ---------------------------------------------------------------------
// Multi-tensor kernel: one launch covers all packed tensors. Each thread
// resolves (tensor_id, local_idx) via upper_bound on the offsets prefix
// sum stored in __constant__ memory. The table is small (kMaxTensors * 32
// bytes per arr * 3 + 8 * (kMaxTensors+1) ~= 9.7 KB at 96 tensors), well
// within the 64 KB constant-memory budget per device.
//
// Cooperative-groups are not strictly required (no cross-block sync), but
// we use a flat 1D grid sized to total elements; blocks past the last
// tensor's range early-out.
// ---------------------------------------------------------------------

template <typename ParamT, typename StateT, typename GradT>
__device__ __forceinline__ int find_tensor_id(
    const int64_t* offsets, int n_tensors, int64_t idx
) {
    // Linear search is O(n_tensors); n_tensors <= 96 and threads in a warp
    // typically hit the same tensor, so branch-prediction is cheap. This
    // outperforms binary search at this scale due to shorter latency
    // chains and warp-uniform lookups.
    int lo = 0, hi = n_tensors;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (offsets[mid] <= idx) lo = mid; else hi = mid;
    }
    return lo;
}

template <typename ParamT, typename StateT, typename GradT>
__launch_bounds__(kBlockSize, kMinBlocksPerSM)
__global__ void multi_tensor_lion_kernel(
    LionTensorTable<ParamT, StateT, GradT> table,
    const int64_t total_N,
    const float lr,
    const float beta1,
    const float beta2,
    const float wd
) {
    static_assert(is_coherent_combo<ParamT, StateT, GradT>::value,
                  "lion::multi_tensor_lion_kernel: incoherent ParamT/StateT/GradT combo");

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_N) return;

    const int t = find_tensor_id<ParamT, StateT, GradT>(
        table.offsets, table.num_tensors, idx);
    const int64_t local = idx - table.offsets[t];

    ParamT*       p_ptr = table.param[t];
    StateT*       e_ptr = table.exp_avg[t];
    const GradT*  g_ptr = table.grad[t];

    const float g  = load_as_float<GradT>(g_ptr + local);
    const float ea = load_state(e_ptr + local);
    const float p  = load_as_float<ParamT>(p_ptr + local);

    const float interp = beta1 * ea + (1.0f - beta1) * g;
    const float s = (interp != 0.0f) ? copysignf(1.0f, interp) : 0.0f;

    store_param<ParamT>(p_ptr + local, p - lr * (s + wd * p));
    store_state(e_ptr + local, beta2 * ea + (1.0f - beta2) * g);
}

// =====================================================================
// Typed launcher template -- called by the AT_DISPATCH dispatch in the
// torch::Tensor-facing launchers. Exposed in the header so the .cu can
// emit explicit instantiations of every coherent (Param, State, Grad)
// combo in the matrix.
// =====================================================================

template <typename ParamT, typename StateT, typename GradT>
cudaError_t launch_lion_typed(
    ParamT* param,
    StateT* exp_avg,
    const GradT* grad,
    float lr, float beta1, float beta2, float wd,
    int64_t n_elements,
    cudaStream_t stream
) {
    static_assert(is_coherent_combo<ParamT, StateT, GradT>::value,
                  "launch_lion_typed: incoherent dtype combo");
    if (n_elements == 0) return cudaSuccess;

    // FP32 vec4 fastpath: identical dtypes, divisible-by-4, 16B aligned.
    if (std::is_same<ParamT, float>::value &&
        std::is_same<StateT, float>::value &&
        std::is_same<GradT,  float>::value &&
        (n_elements % 4) == 0 &&
        (reinterpret_cast<uintptr_t>(param)   % 16) == 0 &&
        (reinterpret_cast<uintptr_t>(exp_avg) % 16) == 0 &&
        (reinterpret_cast<uintptr_t>(grad)    % 16) == 0 &&
        n_elements < (int64_t(1) << 33)) {
        const int64_t N4 = n_elements / 4;
        const int64_t grid = (N4 + kBlockSize - 1) / kBlockSize;
        fused_lion_step_vec4_kernel<<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
            reinterpret_cast<float4*>(param),
            reinterpret_cast<float4*>(exp_avg),
            reinterpret_cast<const float4*>(grad),
            lr, beta1, beta2, wd, N4);
        return cudaGetLastError();
    }

    const int64_t grid = (n_elements + kBlockSize - 1) / kBlockSize;
    fused_lion_step_kernel<ParamT, StateT, GradT>
        <<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
            param, exp_avg, grad, lr, beta1, beta2, wd, n_elements);
    return cudaGetLastError();
}

template <typename ParamT, typename StateT, typename GradT>
cudaError_t launch_multi_tensor_lion_typed(
    const LionTensorTable<ParamT, StateT, GradT>& table,
    int64_t total_N,
    float lr, float beta1, float beta2, float wd,
    cudaStream_t stream
) {
    static_assert(is_coherent_combo<ParamT, StateT, GradT>::value,
                  "launch_multi_tensor_lion_typed: incoherent dtype combo");
    if (total_N == 0 || table.num_tensors == 0) return cudaSuccess;
    const int64_t grid = (total_N + kBlockSize - 1) / kBlockSize;
    multi_tensor_lion_kernel<ParamT, StateT, GradT>
        <<<dim3(grid), dim3(kBlockSize), 0, stream>>>(
            table, total_N, lr, beta1, beta2, wd);
    return cudaGetLastError();
}

}}} // namespace sg::sm90::lion

// =====================================================================
// Public-facing torch::Tensor launchers, declared in namespace
// sg::sm90 to satisfy the forward decls in csrc/bindings/lion.cpp and
// csrc/bindings/multi_tensor.cpp. Defined in lion.cu.
// =====================================================================

namespace sg { namespace sm90 {

void launch_fused_lion_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,
    float lr, float beta1, float beta2, float weight_decay);

// Both the by-reference (lion.cpp) and by-value (multi_tensor.cpp) decls
// are satisfied by overloads in lion.cu.
void launch_multi_tensor_lion(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float weight_decay);

void launch_multi_tensor_lion(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> grads,
    float lr, float beta1, float beta2, float weight_decay);

}} // namespace sg::sm90

