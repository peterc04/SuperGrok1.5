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

#include <ATen/Dispatch.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

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

    // FP32 vec4 fast path (param == backup == grad dtype, all aligned).
    if (param.scalar_type() == at::ScalarType::Float &&
        backup.scalar_type() == at::ScalarType::Float &&
        grad.scalar_type() == at::ScalarType::Float &&
        lk::vec4_eligible_3(param, backup, grad))
    {
        const int64_t N4 = N / 4;
        const int grid = compute_grid(N4, lk::LOOKSAM_BLOCK_SIZE);
        lk::perturb_vec4_kernel<<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<float4*>(param.data_ptr<float>()),
            reinterpret_cast<float4*>(backup.data_ptr<float>()),
            reinterpret_cast<const float4*>(grad.data_ptr<float>()),
            rho_over_norm, N4);
        return;
    }

    const int grid = compute_grid(N, lk::LOOKSAM_BLOCK_SIZE);
    // ParamT == backup dtype (assumed to match param). GradT may differ.
    AT_DISPATCH_SWITCH(param.scalar_type(), "looksam_perturb_param",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            using ParamT = float;
            AT_DISPATCH_SWITCH(grad.scalar_type(), "looksam_perturb_grad",
                AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
                    lk::perturb_kernel<ParamT, float><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                        param.data_ptr<ParamT>(), backup.data_ptr<ParamT>(),
                        grad.data_ptr<float>(), rho_over_norm, N);
                })
                AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
                    lk::perturb_kernel<ParamT, __nv_bfloat16><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                        param.data_ptr<ParamT>(), backup.data_ptr<ParamT>(),
                        reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()),
                        rho_over_norm, N);
                })
                AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                    lk::perturb_kernel<ParamT, __half><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                        param.data_ptr<ParamT>(), backup.data_ptr<ParamT>(),
                        reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()),
                        rho_over_norm, N);
                })
            );
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            using ParamT = __nv_bfloat16;
            auto* p = reinterpret_cast<ParamT*>(param.data_ptr<at::BFloat16>());
            auto* b = reinterpret_cast<ParamT*>(backup.data_ptr<at::BFloat16>());
            if (grad.scalar_type() == at::ScalarType::Float) {
                lk::perturb_kernel<ParamT, float><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                    p, b, grad.data_ptr<float>(), rho_over_norm, N);
            } else if (grad.scalar_type() == at::ScalarType::BFloat16) {
                lk::perturb_kernel<ParamT, __nv_bfloat16><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                    p, b, reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()),
                    rho_over_norm, N);
            } else if (grad.scalar_type() == at::ScalarType::Half) {
                lk::perturb_kernel<ParamT, __half><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                    p, b, reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()),
                    rho_over_norm, N);
            } else {
                TORCH_CHECK(false, "looksam_perturb: unsupported grad dtype");
            }
        })
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            using ParamT = __half;
            auto* p = reinterpret_cast<ParamT*>(param.data_ptr<at::Half>());
            auto* b = reinterpret_cast<ParamT*>(backup.data_ptr<at::Half>());
            if (grad.scalar_type() == at::ScalarType::Float) {
                lk::perturb_kernel<ParamT, float><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                    p, b, grad.data_ptr<float>(), rho_over_norm, N);
            } else if (grad.scalar_type() == at::ScalarType::BFloat16) {
                lk::perturb_kernel<ParamT, __nv_bfloat16><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                    p, b, reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()),
                    rho_over_norm, N);
            } else if (grad.scalar_type() == at::ScalarType::Half) {
                lk::perturb_kernel<ParamT, __half><<<grid, lk::LOOKSAM_BLOCK_SIZE, 0, stream>>>(
                    p, b, reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()),
                    rho_over_norm, N);
            } else {
                TORCH_CHECK(false, "looksam_perturb: unsupported grad dtype");
            }
        })
    );
}

}} // namespace sg::sm90
