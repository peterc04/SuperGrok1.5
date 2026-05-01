// =====================================================================
//  csrc/kernels/cuda/sm_90/looksam.cu
//
//  Thin instantiation translation unit for the sm_90 LookSAM kernels
//  declared in looksam.cuh. Provides the four public launchers that
//  csrc/bindings/looksam.cpp DECLARE_LOOKSAM(sm90) forward-declares.
//
//  The launchers AT_DISPATCH on the runtime tensor dtype to one of the
//  templated kernels in sg::sm90::looksam::. The dispatch is at the
//  launcher boundary only — kernels themselves are purely templated.
//
//  Refresh/cached doubling for direction_adjust_fused is a compile-time
//  template-bool kRefresh; the binding always calls with refresh=true
//  today (Python only invokes the refresh path), but both variants are
//  instantiated so the host can switch via SG_DISPATCH_CALL with a
//  different launcher in a future K-step caching pass.
// =====================================================================

#include "csrc/kernels/cuda/sm_90/looksam.cuh"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

namespace sg { namespace sm90 {

namespace lk = ::sg::sm90::looksam;

// ---------------------------------------------------------------------
// Helper: 1-D grid sizing for elementwise kernels.
// ---------------------------------------------------------------------

static inline int compute_grid(int64_t N, int block) {
    int64_t g = (N + block - 1) / block;
    // cap at 2^31-1 to fit a 32-bit grid.x; sm_90 max is 2^31-1.
    if (g > 2147483647LL) g = 2147483647LL;
    return static_cast<int>(g);
}

// =====================================================================
//  launch_looksam_perturb: param += rho_over_norm * grad; backup = param
// =====================================================================

void launch_looksam_perturb(
    torch::Tensor param, torch::Tensor backup, torch::Tensor grad,
    float rho_over_norm)
{
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    // backup is host-cloned by the binding before this call; the kernel
    // does not touch it. The arg is here for API symmetry only.
    (void)backup;

    // FP32 vec4 fast path: param + grad both FP32 and aligned.
    if (param.scalar_type() == at::ScalarType::Float &&
        grad.scalar_type() == at::ScalarType::Float &&
        lk::vec4_eligible_2(param, grad))
    {
        const int64_t N4 = N / 4;
        const int grid = compute_grid(N4, lk::LOOKSAM_BLOCK_SIZE);
        lk::perturb_vec4_kernel<<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            rho_over_norm, N4);
        return;
    }

    const int grid = compute_grid(N, lk::LOOKSAM_BLOCK_SIZE);

    // Helper macro to keep the 3 x 3 ParamT x GradT dispatch readable.
    // FP8 grads are not exposed at the launcher boundary today (torch
    // tensors don't carry FP8 scalar types in the binding) but the
    // perturb_kernel<ParamT, FP8> instantiations are still emitted in
    // this TU for future linkage.
    #define LOOKSAM_PERTURB_LAUNCH(ParamT, ParamPtr, GradT, GradPtr) \
        lk::perturb_kernel<ParamT, GradT> \
            <<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>( \
            (ParamPtr), (GradPtr), rho_over_norm, N)

    AT_DISPATCH_SWITCH(param.scalar_type(), "looksam_perturb_param",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            float* pp = param.data_ptr<float>();
            if (grad.scalar_type() == at::ScalarType::Float) {
                LOOKSAM_PERTURB_LAUNCH(float, pp, float, grad.data_ptr<float>());
            } else if (grad.scalar_type() == at::ScalarType::BFloat16) {
                LOOKSAM_PERTURB_LAUNCH(float, pp, __nv_bfloat16,
                    reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()));
            } else if (grad.scalar_type() == at::ScalarType::Half) {
                LOOKSAM_PERTURB_LAUNCH(float, pp, __half,
                    reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()));
            } else {
                TORCH_CHECK(false, "looksam_perturb: unsupported grad dtype");
            }
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            __nv_bfloat16* pp = reinterpret_cast<__nv_bfloat16*>(param.data_ptr<at::BFloat16>());
            if (grad.scalar_type() == at::ScalarType::Float) {
                LOOKSAM_PERTURB_LAUNCH(__nv_bfloat16, pp, float, grad.data_ptr<float>());
            } else if (grad.scalar_type() == at::ScalarType::BFloat16) {
                LOOKSAM_PERTURB_LAUNCH(__nv_bfloat16, pp, __nv_bfloat16,
                    reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()));
            } else if (grad.scalar_type() == at::ScalarType::Half) {
                LOOKSAM_PERTURB_LAUNCH(__nv_bfloat16, pp, __half,
                    reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()));
            } else {
                TORCH_CHECK(false, "looksam_perturb: unsupported grad dtype");
            }
        })
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            __half* pp = reinterpret_cast<__half*>(param.data_ptr<at::Half>());
            if (grad.scalar_type() == at::ScalarType::Float) {
                LOOKSAM_PERTURB_LAUNCH(__half, pp, float, grad.data_ptr<float>());
            } else if (grad.scalar_type() == at::ScalarType::BFloat16) {
                LOOKSAM_PERTURB_LAUNCH(__half, pp, __nv_bfloat16,
                    reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()));
            } else if (grad.scalar_type() == at::ScalarType::Half) {
                LOOKSAM_PERTURB_LAUNCH(__half, pp, __half,
                    reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()));
            } else {
                TORCH_CHECK(false, "looksam_perturb: unsupported grad dtype");
            }
        })
    );
    #undef LOOKSAM_PERTURB_LAUNCH
}

// =====================================================================
//  launch_looksam_restore: param = backup
// =====================================================================

void launch_looksam_restore(torch::Tensor param, torch::Tensor backup) {
    const int64_t N = param.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (param.scalar_type() == at::ScalarType::Float &&
        backup.scalar_type() == at::ScalarType::Float &&
        lk::vec4_eligible_2(param, backup))
    {
        const int64_t N4 = N / 4;
        const int grid = compute_grid(N4, lk::LOOKSAM_BLOCK_SIZE);
        lk::restore_vec4_kernel<<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<const float4*>(backup.data_ptr<float>()),
            N4);
        return;
    }

    const int grid = compute_grid(N, lk::LOOKSAM_BLOCK_SIZE);
    AT_DISPATCH_SWITCH(param.scalar_type(), "looksam_restore",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            lk::restore_kernel<float><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                param.data_ptr<float>(), backup.data_ptr<float>(), N);
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            lk::restore_kernel<__nv_bfloat16><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(param.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(backup.data_ptr<at::BFloat16>()),
                N);
        })
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            lk::restore_kernel<__half><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<__half*>(param.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(backup.data_ptr<at::Half>()),
                N);
        })
    );
}

// =====================================================================
//  launch_looksam_direction_adjust_fused
//
//  Binding contract (verified against csrc/bindings/looksam.cpp:113,132):
//    - host pre-computes diff = (sam_grad - normal_grad).to(float32)
//    - host pre-computes inv_norm = 1 / ||diff||
//    - the v_dir argument here IS that FP32 diff tensor
//    - sam_grad argument is the perturbed grad (unread by today's call;
//      reserved for the future kRefresh=true on-the-fly diff path)
//
//  This is the cached path (kRefresh=false): one read of v_dir + one
//  read+write of grad. The refresh path (kRefresh=true, on-the-fly
//  diff in registers) is compiled in but not yet wired to the binding.
// =====================================================================

void launch_looksam_direction_adjust_fused(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor v_dir,
    float inv_norm, float lambda, float grad_norm)
{
    const int64_t N = grad.numel();
    if (N == 0) return;
    const float la_times_gnorm = lambda * grad_norm;
    auto stream = at::cuda::getCurrentCUDAStream();
    (void)sam_grad;  // unused in cached path; binding still passes it.

    // FP32 vec4 fast path: grad + v_dir both FP32 + 16-B aligned.
    if (grad.scalar_type() == at::ScalarType::Float &&
        v_dir.scalar_type() == at::ScalarType::Float &&
        lk::vec4_eligible_2(grad, v_dir))
    {
        const int64_t N4 = N / 4;
        const int grid = compute_grid(N4, lk::LOOKSAM_BLOCK_SIZE);
        // sam_grad pointer is not read in cached vec4 path; reuse v_dir.
        lk::direction_adjust_fused_vec4_kernel<false><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<float4*>(grad.data_ptr<float>()),
            reinterpret_cast<const float4*>(v_dir.data_ptr<float>()),  // unused
            reinterpret_cast<const float4*>(v_dir.data_ptr<float>()),
            inv_norm, la_times_gnorm, N4);
        return;
    }

    // VDirT is fixed FP32 (binding casts diff to float32 before calling).
    TORCH_CHECK(v_dir.scalar_type() == at::ScalarType::Float,
                "looksam_direction_adjust_fused expects FP32 v_dir tensor");

    const int grid = compute_grid(N, lk::LOOKSAM_BLOCK_SIZE);
    AT_DISPATCH_SWITCH(grad.scalar_type(), "looksam_direction_adjust_fused",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            lk::direction_adjust_fused_kernel<false, float, float>
                <<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                grad.data_ptr<float>(), grad.data_ptr<float>() /*unused*/,
                v_dir.data_ptr<float>(), inv_norm, la_times_gnorm, N);
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            auto* g = reinterpret_cast<__nv_bfloat16*>(grad.data_ptr<at::BFloat16>());
            lk::direction_adjust_fused_kernel<false, __nv_bfloat16, float>
                <<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                g, g /*unused*/, v_dir.data_ptr<float>(),
                inv_norm, la_times_gnorm, N);
        })
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            auto* g = reinterpret_cast<__half*>(grad.data_ptr<at::Half>());
            lk::direction_adjust_fused_kernel<false, __half, float>
                <<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                g, g /*unused*/, v_dir.data_ptr<float>(),
                inv_norm, la_times_gnorm, N);
        })
    );
}

// =====================================================================
//  launch_looksam_norm_reduce — [diff_norm_sq, grad_norm_sq] in one pass
//
//  The binding owns the sqrt + 1/x post-processing on the host. The
//  apply kernels never reduce — by spec.
// =====================================================================

void launch_looksam_norm_reduce(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor results)
{
    const int64_t N = grad.numel();
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();

    // Persistent grid: cap so each block does meaningful work.
    int grid = compute_grid(N, lk::LOOKSAM_BLOCK_SIZE);
    if (grid > 1024) grid = 1024;

    AT_DISPATCH_SWITCH(grad.scalar_type(), "looksam_norm_reduce",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            lk::norm_reduce_kernel<float><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                grad.data_ptr<float>(), sam_grad.data_ptr<float>(),
                results.data_ptr<float>(), N);
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            lk::norm_reduce_kernel<__nv_bfloat16><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(sam_grad.data_ptr<at::BFloat16>()),
                results.data_ptr<float>(), N);
        })
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            lk::norm_reduce_kernel<__half><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(sam_grad.data_ptr<at::Half>()),
                results.data_ptr<float>(), N);
        })
    );
}

}} // namespace sg::sm90

// =====================================================================
//  Explicit template instantiations — dtype matrix.
//
//  ParamT in {float, __nv_bfloat16, __half}                        (3)
//  StateT in {float, __nv_bfloat16}                                (2)
//      — exposed via store_state_nt; LookSAM v_dir is FP32 in the
//        current binding so StateT only matters in a future pass.
//  GradT  in {float, __nv_bfloat16, __half,
//             __nv_fp8_e4m3, __nv_fp8_e5m2}                        (5)
//  Refresh/cached doubling for direction_adjust_fused              (x2)
//
//  Most instantiations are forced by the AT_DISPATCH blocks above; the
//  explicit `template ...;` lines below force the FP8 grad variants and
//  both kRefresh paths to be emitted into this TU so a future binding
//  pass can call them without a relink.
//
//  Counts:
//    perturb_kernel<ParamT, GradT>          : 3 * 5     = 15
//    restore_kernel<ParamT>                 :              3
//    direction_adjust_fused_kernel
//        <kRefresh, GradT, VDirT>           :              10
//        - cached  (kRefresh=false, VDirT=float): GradT ×5 = 5
//        - refresh (kRefresh=true,  VDirT=GradT): GradT ×5 = 5
//    norm_reduce_kernel<GradT>              :              5
//    direction_adjust_fused_vec4_kernel<R>  :              2
//    perturb_vec4_kernel, restore_vec4_kernel:             2
//  Total instantiations                     :              37
// =====================================================================

namespace sg { namespace sm90 { namespace looksam {

// perturb: ParamT × GradT (ParamT in {fp32, bf16, fp16},
//                          GradT  in {fp32, bf16, fp16, fp8e4m3, fp8e5m2})
#define INST_PERTURB(P, G) \
    template __global__ void perturb_kernel<P, G>( \
        P*, const G*, const float, const int64_t);
INST_PERTURB(float,         float)
INST_PERTURB(float,         __nv_bfloat16)
INST_PERTURB(float,         __half)
INST_PERTURB(float,         __nv_fp8_e4m3)
INST_PERTURB(float,         __nv_fp8_e5m2)
INST_PERTURB(__nv_bfloat16, float)
INST_PERTURB(__nv_bfloat16, __nv_bfloat16)
INST_PERTURB(__nv_bfloat16, __half)
INST_PERTURB(__nv_bfloat16, __nv_fp8_e4m3)
INST_PERTURB(__nv_bfloat16, __nv_fp8_e5m2)
INST_PERTURB(__half,        float)
INST_PERTURB(__half,        __nv_bfloat16)
INST_PERTURB(__half,        __half)
INST_PERTURB(__half,        __nv_fp8_e4m3)
INST_PERTURB(__half,        __nv_fp8_e5m2)
#undef INST_PERTURB

// restore: ParamT only
template __global__ void restore_kernel<float>(float*, const float*, const int64_t);
template __global__ void restore_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const int64_t);
template __global__ void restore_kernel<__half>(__half*, const __half*, const int64_t);

// direction+adjust: kRefresh × GradT × VDirT
//   - cached  (kRefresh=false): VDirT = float (binding casts diff -> fp32)
//   - refresh (kRefresh=true) : VDirT = GradT (computed in-register from
//                                              sam_grad, normal_grad)
#define INST_DAF(R, G, V) \
    template __global__ void direction_adjust_fused_kernel<R, G, V>( \
        G*, const G*, const V*, const float, const float, const int64_t);

// Cached: GradT in {fp32, bf16, fp16, fp8e4m3, fp8e5m2}, VDirT = float
INST_DAF(false, float,         float)
INST_DAF(false, __nv_bfloat16, float)
INST_DAF(false, __half,        float)
INST_DAF(false, __nv_fp8_e4m3, float)
INST_DAF(false, __nv_fp8_e5m2, float)

// Refresh: GradT × VDirT == GradT (homogeneous diff path)
INST_DAF(true,  float,         float)
INST_DAF(true,  __nv_bfloat16, __nv_bfloat16)
INST_DAF(true,  __half,        __half)
INST_DAF(true,  __nv_fp8_e4m3, __nv_fp8_e4m3)
INST_DAF(true,  __nv_fp8_e5m2, __nv_fp8_e5m2)
#undef INST_DAF

// vec4 direction+adjust: refresh + cached
template __global__ void direction_adjust_fused_vec4_kernel<true>(
    float4*, const float4*, const float4*, const float, const float, const int64_t);
template __global__ void direction_adjust_fused_vec4_kernel<false>(
    float4*, const float4*, const float4*, const float, const float, const int64_t);

// norm_reduce: GradT only
template __global__ void norm_reduce_kernel<float>(
    const float*, const float*, float*, const int64_t);
template __global__ void norm_reduce_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, float*, const int64_t);
template __global__ void norm_reduce_kernel<__half>(
    const __half*, const __half*, float*, const int64_t);
template __global__ void norm_reduce_kernel<__nv_fp8_e4m3>(
    const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, float*, const int64_t);
template __global__ void norm_reduce_kernel<__nv_fp8_e5m2>(
    const __nv_fp8_e5m2*, const __nv_fp8_e5m2*, float*, const int64_t);

}}} // namespace sg::sm90::looksam

