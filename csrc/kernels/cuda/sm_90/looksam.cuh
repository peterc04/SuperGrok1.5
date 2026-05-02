// =====================================================================
//  csrc/kernels/cuda/sm_90/looksam.cuh
//
//  sm_90 (Hopper) LookSAM kernels + launcher header.
//
//  LookSAM (Liu et al., 2022) is AdamW + sharpness-aware direction
//  adjustment refreshed every k steps. The four operations exposed to
//  the binding are:
//
//    1. perturb        : param += rho_over_norm * grad   (writes backup)
//    2. restore        : param = backup
//    3. direction+adjust (fused) :
//                        v_dir = (sam_grad - normal_grad) * inv_norm
//                        grad += la_times_gnorm * v_dir
//                        — v_dir stays in registers, never round-trips.
//    4. norm_reduce    : compute [||sam_grad - normal_grad||^2,
//                                  ||grad||^2] in one pass with warp
//                        shuffle + atomicAdd.
//
//  Launcher signatures match csrc/bindings/looksam.cpp DECLARE_LOOKSAM(sm90)
//  — torch::Tensor in, AT_DISPATCH on dtype to the templated kernel.
//
//  Roofline (FP32 perturb):
//     R = 2 * 4 B (param, grad)
//     W = 2 * 4 B (param, backup)
//     I = 1 FMA  -> AI = 1/16 FLOP/B  -> hard BW-bound on H100 HBM3.
//  Optimisation strategy is therefore "saturate HBM": vec4 loads, wt
//  stores on backup/v_dir state to bypass L2, __launch_bounds__(256, 8)
//  matching the deleted sm_90 baseline (high occupancy hides BW stalls).
//
//  Heterogeneous dtype support (instantiations in looksam.cu):
//     ParamT in {float, __nv_bfloat16, __half}
//     StateT in {float, __nv_bfloat16}        (backup uses ParamT,
//                                              direction state uses StateT)
//     GradT  in {float, __nv_bfloat16, __half,
//                __nv_fp8_e4m3, __nv_fp8_e5m2}
//  All math runs in FP32; only loads/stores are typed.
//  The fused direction+adjust kernel additionally takes a kRefresh
//  template bool — kRefresh=true emits the v_dir update; kRefresh=false
//  skips it (cached path saves 2 reads / 1 write per element). The host
//  decides which to call based on (step % k).
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/tuned_configs.h"
#include "csrc/device/optimizers/sm_90/looksam_sm90.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
  #include <cuda_fp8.h>
#endif

#include <torch/extension.h>
#include <type_traits>

namespace sg { namespace sm90 { namespace looksam {

constexpr int LOOKSAM_BLOCK_SIZE = 256;
constexpr int LOOKSAM_MIN_BLOCKS_PER_SM = 8;

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

// ---------------------------------------------------------------------
// Type-erased load / store helpers (FP32 math; typed memory).
// Mirrors lion.cuh — kept local to avoid a cross-optimizer dependency
// and so each kernel header compiles standalone.
// ---------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p) {
    return static_cast<float>(*p);
}
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

template <typename T>
__device__ __forceinline__ void store_from_float(T* p, float v) {
    *p = static_cast<T>(v);
}
template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ void store_from_float<__half>(__half* p, float v) {
    *p = __float2half_rn(v);
}

// State stream-store: FP32 state goes through the wt path to bypass L2
// allocation; BF16 state has no PTX wt v1.b16 path so falls through.
__device__ __forceinline__ void store_state_nt(float* p, float v)         { stream_store(p, v); }
__device__ __forceinline__ void store_state_nt(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

// =====================================================================
//  Kernel 1: perturb — param += rho_over_norm * grad
//
//  The binding pre-clones the backup tensor on the host before calling
//  this launcher (looksam.cpp:73 — `backups.push_back(params[i].clone())`).
//  The kernel therefore only reads param + grad and writes param;
//  the backup pointer is accepted in the launcher signature for API
//  symmetry but is not touched in the inner loop.
// =====================================================================

template <typename ParamT, typename GradT>
__launch_bounds__(LOOKSAM_BLOCK_SIZE, LOOKSAM_MIN_BLOCKS_PER_SM)
__global__ void perturb_kernel(
    ParamT* __restrict__ param,
    const GradT* __restrict__ grad,
    const float rho_over_norm,
    const int64_t N)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float p = load_as_float<ParamT>(param + idx);
    float g = load_as_float<GradT>(grad + idx);
    store_from_float<ParamT>(param + idx, p + rho_over_norm * g);
}

// FP32-only float4 fast path (param + grad both FP32). 2 reads / 1 write
// per element — the vector path saturates HBM by issuing 16 B per lane
// per memory op.
__launch_bounds__(LOOKSAM_BLOCK_SIZE, LOOKSAM_MIN_BLOCKS_PER_SM)
__global__ void perturb_vec4_kernel(
    float4* __restrict__ param4,
    const float4* __restrict__ grad4,
    const float rho_over_norm,
    const int64_t N4)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N4) return;
    float4 p = param4[i];
    float4 g = stream_load4(grad4 + i);
    p.x += rho_over_norm * g.x;
    p.y += rho_over_norm * g.y;
    p.z += rho_over_norm * g.z;
    p.w += rho_over_norm * g.w;
    param4[i] = p;
}

// =====================================================================
//  Kernel 2: restore — param = backup
//
//  Pure copy; the backup tensor is dead after this call so its load is
//  also non-temporal in the FP32 fast path.
// =====================================================================

template <typename ParamT>
__launch_bounds__(LOOKSAM_BLOCK_SIZE, LOOKSAM_MIN_BLOCKS_PER_SM)
__global__ void restore_kernel(
    ParamT* __restrict__ param,
    const ParamT* __restrict__ backup,
    const int64_t N)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float b = load_as_float<ParamT>(backup + idx);
    store_from_float<ParamT>(param + idx, b);
}

__launch_bounds__(LOOKSAM_BLOCK_SIZE, LOOKSAM_MIN_BLOCKS_PER_SM)
__global__ void restore_vec4_kernel(
    float4* __restrict__ param4,
    const float4* __restrict__ backup4,
    const int64_t N4)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N4) return;
    param4[i] = stream_load4(backup4 + i);
}

// =====================================================================
//  Kernel 3: direction + adjust (fused)
//
//  Per the current binding contract (csrc/bindings/looksam.cpp lines
//  113, 132): the host pre-computes diff = (sam_grad - normal_grad)
//  cast to FP32 and passes it as the "v_dir" tensor along with
//  inv_norm = 1/||diff||. The launcher's job is therefore the apply:
//
//      grad[i] += lambda * grad_norm * (v_dir[i] * inv_norm)
//
//  kRefresh template bool:
//    - kRefresh = false (used by today's binding):
//        v_dir is loaded directly from GMEM (it's already FP32 diff).
//    - kRefresh = true  (compiled but not yet wired):
//        Future fused path that takes (grad, sam_grad, normal_grad)
//        and computes diff in-register, saving the .to(float32) +
//        materialisation of the diff tensor on the host. Reuses the
//        same kernel slot via the third-tensor pointer being read as
//        normal_grad of dtype GradT.
//
//  Two dtypes are exposed:
//    GradT — the dtype of grad[] (loaded + stored)
//    VDirT — the dtype of the third tensor (FP32 in cached, GradT
//             in refresh-future). All math FP32.
// =====================================================================

template <bool kRefresh, typename GradT, typename VDirT>
__launch_bounds__(LOOKSAM_BLOCK_SIZE, LOOKSAM_MIN_BLOCKS_PER_SM)
__global__ void direction_adjust_fused_kernel(
    GradT* __restrict__ grad,
    const GradT* __restrict__ sam_grad,         // refresh-future only
    const VDirT* __restrict__ v_dir_or_normal,  // cached: v_dir (FP32)
                                                // refresh: normal_grad
    const float inv_norm,
    const float la_times_gnorm,
    const int64_t N)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v;
    if constexpr (kRefresh) {
        float sg = load_as_float<GradT>(sam_grad + idx);
        float ng = load_as_float<VDirT>(v_dir_or_normal + idx);
        v = (sg - ng) * inv_norm;
    } else {
        // Cached: v_dir already pre-multiplied by 1 (raw diff). The
        // inv_norm scaling happens here so that lambda*grad_norm
        // sees a unit direction.
        v = load_as_float<VDirT>(v_dir_or_normal + idx) * inv_norm;
    }
    float g = load_as_float<GradT>(grad + idx);
    store_from_float<GradT>(grad + idx, g + la_times_gnorm * v);
}

// FP32-only float4 fast path. Both refresh and cached share the same
// vec4 layout (all three tensors FP32 + 16 B aligned).
template <bool kRefresh>
__launch_bounds__(LOOKSAM_BLOCK_SIZE, LOOKSAM_MIN_BLOCKS_PER_SM)
__global__ void direction_adjust_fused_vec4_kernel(
    float4* __restrict__ grad4,
    const float4* __restrict__ sam_grad4,
    const float4* __restrict__ v_dir_or_normal4,
    const float inv_norm,
    const float la_times_gnorm,
    const int64_t N4)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N4) return;
    float4 v;
    if constexpr (kRefresh) {
        float4 sg = sam_grad4[i];
        float4 ng = v_dir_or_normal4[i];
        v.x = (sg.x - ng.x) * inv_norm;
        v.y = (sg.y - ng.y) * inv_norm;
        v.z = (sg.z - ng.z) * inv_norm;
        v.w = (sg.w - ng.w) * inv_norm;
    } else {
        v = stream_load4(v_dir_or_normal4 + i);
        v.x *= inv_norm; v.y *= inv_norm;
        v.z *= inv_norm; v.w *= inv_norm;
    }
    float4 g = grad4[i];
    g.x += la_times_gnorm * v.x;
    g.y += la_times_gnorm * v.y;
    g.z += la_times_gnorm * v.z;
    g.w += la_times_gnorm * v.w;
    grad4[i] = g;
}

// =====================================================================
//  Kernel 4: fused norm reduce — [||sam_grad - normal_grad||^2,
//                                   ||grad||^2]
//
//  Single grid-stride pass; warp shuffle reduce + 1 atomicAdd per warp
//  per output. Per the binding contract this is the only reduction
//  inside the LookSAM kernels — the apply kernels never reduce.
// =====================================================================

template <typename GradT>
__launch_bounds__(LOOKSAM_BLOCK_SIZE, LOOKSAM_MIN_BLOCKS_PER_SM)
__global__ void norm_reduce_kernel(
    const GradT* __restrict__ grad,
    const GradT* __restrict__ sam_grad,
    float* __restrict__ results,    // [2]: {diff_norm_sq, grad_norm_sq}
    const int64_t N)
{
    float diff_sq = 0.0f, grad_sq = 0.0f;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < N; i += stride) {
        float sg = load_as_float<GradT>(sam_grad + i);
        float g  = load_as_float<GradT>(grad + i);
        float d  = sg - g;
        diff_sq += d * d;
        grad_sq += g * g;
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    diff_sq = cluster_dsmem_reduce_sum(diff_sq);
    grad_sq = cluster_dsmem_reduce_sum(grad_sq);
    if (threadIdx.x == 0) {
        namespace cg = cooperative_groups;
        auto cluster = cg::this_cluster();
        if (cluster.block_rank() == 0) {
            atomicAdd(&results[0], diff_sq);
            atomicAdd(&results[1], grad_sq);
        }
    }
#else
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        diff_sq += SHFL_DOWN(diff_sq, offset);
        grad_sq += SHFL_DOWN(grad_sq, offset);
    }
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        atomicAdd(&results[0], diff_sq);
        atomicAdd(&results[1], grad_sq);
    }
#endif
}

// =====================================================================
//  vec4 fast-path eligibility predicate — three pointers must be
//  16-byte aligned and N must be divisible by 4. FP32-only because
//  float4 is the only safe 16-B vector load type for our dtypes.
// =====================================================================

inline bool vec4_eligible_3(const torch::Tensor& a, const torch::Tensor& b,
                            const torch::Tensor& c)
{
    if (a.scalar_type() != at::ScalarType::Float) return false;
    if (a.numel() % 4 != 0) return false;
    return reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0
        && reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0
        && reinterpret_cast<uintptr_t>(c.data_ptr()) % 16 == 0;
}

inline bool vec4_eligible_2(const torch::Tensor& a, const torch::Tensor& b)
{
    if (a.scalar_type() != at::ScalarType::Float) return false;
    if (a.numel() % 4 != 0) return false;
    return reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0
        && reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0;
}

}}} // namespace sg::sm90::looksam

// =====================================================================
//  Public launchers — match csrc/bindings/looksam.cpp DECLARE_LOOKSAM(sm90)
//  exactly. They live in namespace sg::sm90 (not ::looksam) because that
//  is what the binding's SG_DISPATCH macro names.
// =====================================================================

namespace sg { namespace sm90 {

void launch_looksam_perturb(
    torch::Tensor param, torch::Tensor backup, torch::Tensor grad,
    float rho_over_norm);

void launch_looksam_restore(
    torch::Tensor param, torch::Tensor backup);

void launch_looksam_direction_adjust_fused(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor v_dir,
    float inv_norm, float lambda, float grad_norm);

void launch_looksam_norm_reduce(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor results);

}} // namespace sg::sm90
