// =====================================================================
//  csrc/kernels/cuda/sm_90/prodigy.cu
//
//  sm_90 (Hopper) Prodigy optimizer kernels — explicit instantiation TU.
//
//  All kernel + launcher logic lives in prodigy.cuh as templates. This
//  TU forces emission of the dtype matrix below into a single object
//  file so callers outside this TU can link against the per-tensor
//  entry point without dragging the whole header into a non-CUDA TU.
//
//  Dtype matrix (per main spec):
//    ParamT in {float, __nv_bfloat16, __half}
//    StateT in {float, __nv_bfloat16}
//    GradT  in {float, __nv_bfloat16, __half,
//               __nv_fp8_e4m3, __nv_fp8_e5m2}
//
//  Incoherent combos (FP8 grad with FP32 param) are statically rejected
//  by the templates — those rows are simply absent from the list below.
//  Total: 3 (FP32 param x 3 grads) * 2 states
//       + 5 (BF16 param x 5 grads) * 2 states
//       + 5 (FP16 param x 5 grads) * 2 states
//       = 6 + 10 + 10 = 26 instantiations.
// =====================================================================

#include "csrc/kernels/cuda/sm_90/prodigy.cuh"

namespace sg { namespace sm90 { namespace prodigy {

// Convenience macro for the launcher signature (verbose enough that the
// compiler can pick the right overload, but short enough to keep the
// instantiation list scannable).
#define SG_PRODIGY_INST(P, S, G)                                              \
    template cudaError_t launch_prodigy_fused_step<P, S, G>(                  \
        P*, P*, S*, S*, S*, S*, const G*,                                     \
        float*, float*, float*,                                               \
        float, float, float, float, float,                                    \
        int64_t, int64_t, cudaStream_t)

// ---------------------------------------------------------------------
// FP32 param family — FP8 grads excluded (is_coherent_combo).
// ---------------------------------------------------------------------
SG_PRODIGY_INST(float, float, float);
SG_PRODIGY_INST(float, float, __nv_bfloat16);
SG_PRODIGY_INST(float, float, __half);
SG_PRODIGY_INST(float, __nv_bfloat16, float);
SG_PRODIGY_INST(float, __nv_bfloat16, __nv_bfloat16);
SG_PRODIGY_INST(float, __nv_bfloat16, __half);

// ---------------------------------------------------------------------
// BF16 param family — full grad cross-section (incl. FP8 grads).
// ---------------------------------------------------------------------
SG_PRODIGY_INST(__nv_bfloat16, float, float);
SG_PRODIGY_INST(__nv_bfloat16, float, __nv_bfloat16);
SG_PRODIGY_INST(__nv_bfloat16, float, __half);
SG_PRODIGY_INST(__nv_bfloat16, float, __nv_fp8_e4m3);
SG_PRODIGY_INST(__nv_bfloat16, float, __nv_fp8_e5m2);
SG_PRODIGY_INST(__nv_bfloat16, __nv_bfloat16, float);
SG_PRODIGY_INST(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
SG_PRODIGY_INST(__nv_bfloat16, __nv_bfloat16, __half);
SG_PRODIGY_INST(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3);
SG_PRODIGY_INST(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e5m2);

// ---------------------------------------------------------------------
// FP16 param family — full grad cross-section.
// ---------------------------------------------------------------------
SG_PRODIGY_INST(__half, float, float);
SG_PRODIGY_INST(__half, float, __nv_bfloat16);
SG_PRODIGY_INST(__half, float, __half);
SG_PRODIGY_INST(__half, float, __nv_fp8_e4m3);
SG_PRODIGY_INST(__half, float, __nv_fp8_e5m2);
SG_PRODIGY_INST(__half, __nv_bfloat16, float);
SG_PRODIGY_INST(__half, __nv_bfloat16, __nv_bfloat16);
SG_PRODIGY_INST(__half, __nv_bfloat16, __half);
SG_PRODIGY_INST(__half, __nv_bfloat16, __nv_fp8_e4m3);
SG_PRODIGY_INST(__half, __nv_bfloat16, __nv_fp8_e5m2);

#undef SG_PRODIGY_INST

}}} // namespace sg::sm90::prodigy

// =====================================================================
// torch::Tensor binding shims in namespace sg::sm90.
//
// Forward-declared by csrc/bindings/prodigy.cpp DECLARE_PRODIGY(sm90).
// These accept torch::Tensors, dispatch on dtype, extract raw pointers,
// and delegate to the inner sg::sm90::prodigy:: templated launchers.
// =====================================================================

#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

namespace sg { namespace sm90 {

// ---------------------------------------------------------------------
// launch_fused_prodigy_step — per-tensor fused Prodigy apply step.
//
// The binding provides d_lr as a scalar float. We place it on-device,
// allocate temporary workspace for r_state / r_partial / s_partial,
// and delegate to prodigy::launch_prodigy_fused_step which runs
// reduce + d_update + apply. Because r_state is zero-init'd, the
// reduce contributes nothing, d_update yields max(d_lr, 0) == d_lr,
// and the apply kernel executes with the caller's d_lr.
// ---------------------------------------------------------------------
void launch_fused_prodigy_step(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor s,
    torch::Tensor param_init,
    torch::Tensor grad,
    float lr,
    float d_lr,
    float beta1,
    float beta2,
    float weight_decay,
    float eps,
    float bc1,
    float bc2
) {
    TORCH_CHECK(param.is_cuda(),       "prodigy_step sm90: param must be CUDA");
    TORCH_CHECK(exp_avg.is_cuda(),     "prodigy_step sm90: exp_avg must be CUDA");
    TORCH_CHECK(exp_avg_sq.is_cuda(),  "prodigy_step sm90: exp_avg_sq must be CUDA");
    TORCH_CHECK(s.is_cuda(),           "prodigy_step sm90: s must be CUDA");
    TORCH_CHECK(param_init.is_cuda(),  "prodigy_step sm90: param_init must be CUDA");
    TORCH_CHECK(grad.is_cuda(),        "prodigy_step sm90: grad must be CUDA");
    TORCH_CHECK(param.is_contiguous(),       "prodigy_step sm90: param must be contiguous");
    TORCH_CHECK(exp_avg.is_contiguous(),     "prodigy_step sm90: exp_avg must be contiguous");
    TORCH_CHECK(exp_avg_sq.is_contiguous(),  "prodigy_step sm90: exp_avg_sq must be contiguous");
    TORCH_CHECK(s.is_contiguous(),           "prodigy_step sm90: s must be contiguous");
    TORCH_CHECK(param_init.is_contiguous(),  "prodigy_step sm90: param_init must be contiguous");
    TORCH_CHECK(grad.is_contiguous(),        "prodigy_step sm90: grad must be contiguous");

    // State tensors are always FP32.
    TORCH_CHECK(exp_avg.scalar_type()    == at::kFloat,
                "prodigy_step sm90: exp_avg must be float32");
    TORCH_CHECK(exp_avg_sq.scalar_type() == at::kFloat,
                "prodigy_step sm90: exp_avg_sq must be float32");
    TORCH_CHECK(s.scalar_type()          == at::kFloat,
                "prodigy_step sm90: s must be float32");

    const int64_t N = param.numel();
    if (N == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Recover step from bc1 for PRNG seeding.
    int64_t step_count = 0;
    if (bc1 > 0.0f && beta1 > 0.0f && beta1 < 1.0f) {
        const float t = std::log(std::max(1.0f - bc1, 1e-30f))
                        / std::log(beta1);
        step_count = static_cast<int64_t>(std::lround(std::max(t, 1.0f)));
    }

    // Temporary device workspace: r_state (N floats), d_lr (1 float),
    // r_partial (1 float), s_partial (1 float).
    auto opts_f32 = torch::TensorOptions().device(param.device()).dtype(torch::kFloat32);
    auto r_state_tmp   = torch::zeros({N}, opts_f32);
    auto d_lr_dev      = torch::tensor({d_lr}, opts_f32);
    auto r_partial_tmp = torch::zeros({1}, opts_f32);
    auto s_partial_tmp = torch::zeros({1}, opts_f32);

    // Dispatch on param dtype. Grad dtype matches param; state is FP32.
    TORCH_CHECK(grad.scalar_type() == param.scalar_type(),
                "prodigy_step sm90: grad and param dtypes must match");

    // param_init may be FP32 (Python: p.clone().float()); the inner kernel
    // expects ParamT, so cast if needed. The cast is a no-op when param is
    // already FP32.
    torch::Tensor param_init_cast = param_init.to(param.scalar_type());

    auto dispatch_typed = [&](auto param_tag) {
        using ParamT = decltype(param_tag);
        auto err = ::sg::sm90::prodigy::launch_prodigy_fused_step<
            ParamT, float, ParamT>(
            reinterpret_cast<ParamT*>(param.data_ptr()),
            reinterpret_cast<ParamT*>(param_init_cast.data_ptr()),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            r_state_tmp.data_ptr<float>(),
            s.data_ptr<float>(),
            reinterpret_cast<const ParamT*>(grad.data_ptr()),
            d_lr_dev.data_ptr<float>(),
            r_partial_tmp.data_ptr<float>(),
            s_partial_tmp.data_ptr<float>(),
            lr, beta1, beta2, eps, weight_decay,
            N, step_count, stream);
        TORCH_CHECK(err == cudaSuccess,
            "launch_fused_prodigy_step (sm90): ", cudaGetErrorString(err));
    };

    switch (param.scalar_type()) {
        case at::kFloat:    dispatch_typed(float{});          break;
        case at::kBFloat16: dispatch_typed(__nv_bfloat16{}); break;
        case at::kHalf:     dispatch_typed(__half{});         break;
        default:
            TORCH_CHECK(false,
                "prodigy_step sm90: unsupported param dtype ",
                param.scalar_type());
    }
}

// ---------------------------------------------------------------------
// launch_prodigy_dlr_reduce — per-tensor d_lr reduction.
//
// Computes partial sums for the Prodigy learning rate update:
//   numerator   += d_prev^2 * g[i] * (p0[i] - p[i])
//   denominator += |d_prev*(1-sqrt(beta2))*g[i] + sqrt(beta2)*s[i]|
//
// The binding does not pass d_prev or beta2 explicitly. We set
// d_prev = 1.0 (unity scaling; the caller handles d_prev externally)
// and pass the eps argument as beta2 to the reduction kernel. The
// numerator and denominator tensors are expected to be pre-zeroed
// single-element FP32 accumulators; the kernel atomicAdd's into them.
// ---------------------------------------------------------------------
void launch_prodigy_dlr_reduce(
    torch::Tensor grad,
    torch::Tensor param,
    torch::Tensor param_init,
    torch::Tensor s,
    torch::Tensor numerator,
    torch::Tensor denominator,
    float eps
) {
    TORCH_CHECK(grad.is_cuda(),        "prodigy_dlr_reduce sm90: grad must be CUDA");
    TORCH_CHECK(param.is_cuda(),       "prodigy_dlr_reduce sm90: param must be CUDA");
    TORCH_CHECK(param_init.is_cuda(),  "prodigy_dlr_reduce sm90: param_init must be CUDA");
    TORCH_CHECK(s.is_cuda(),           "prodigy_dlr_reduce sm90: s must be CUDA");
    TORCH_CHECK(numerator.is_cuda(),   "prodigy_dlr_reduce sm90: numerator must be CUDA");
    TORCH_CHECK(denominator.is_cuda(), "prodigy_dlr_reduce sm90: denominator must be CUDA");
    TORCH_CHECK(grad.is_contiguous(),       "prodigy_dlr_reduce sm90: grad must be contiguous");
    TORCH_CHECK(param.is_contiguous(),      "prodigy_dlr_reduce sm90: param must be contiguous");
    TORCH_CHECK(param_init.is_contiguous(), "prodigy_dlr_reduce sm90: param_init must be contiguous");
    TORCH_CHECK(s.is_contiguous(),          "prodigy_dlr_reduce sm90: s must be contiguous");

    // State (s) and accumulators are always FP32.
    TORCH_CHECK(s.scalar_type()           == at::kFloat,
                "prodigy_dlr_reduce sm90: s must be float32");
    TORCH_CHECK(numerator.scalar_type()   == at::kFloat,
                "prodigy_dlr_reduce sm90: numerator must be float32");
    TORCH_CHECK(denominator.scalar_type() == at::kFloat,
                "prodigy_dlr_reduce sm90: denominator must be float32");

    const int64_t N = grad.numel();
    if (N == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // d_prev = 1.0 (unity; caller scales externally).
    auto opts_f32 = torch::TensorOptions().device(grad.device()).dtype(torch::kFloat32);
    auto d_prev_dev = torch::ones({1}, opts_f32);

    // Block size from tuned configs (same as the fused launcher uses).
    const ::sg::LaunchConfig cfg =
        ::sg::get_grokadamw_config(/*arch=*/90, static_cast<int>(N));
    const int block = cfg.block_size;

    constexpr int MAX_BLOCKS = 8192;
    const int64_t needed = (N + block - 1) / block;
    const int grid = static_cast<int>(needed < MAX_BLOCKS ? needed : MAX_BLOCKS);

    // The beta2 argument to the kernel controls the mixing between
    // grad and s_state in the denominator channel. We forward the
    // binding's eps parameter here — the caller is responsible for
    // interpreting the returned (numerator, denominator) accordingly.
    const float beta2 = eps;

    // Dispatch on param dtype. Grad dtype matches param; state is FP32.
    TORCH_CHECK(grad.scalar_type() == param.scalar_type(),
                "prodigy_dlr_reduce sm90: grad and param dtypes must match");

    // param_init may be FP32 (Python: p.clone().float()); the inner kernel
    // expects ParamT, so cast if needed.
    torch::Tensor param_init_cast = param_init.to(param.scalar_type());

    auto launch_reduce = [&](auto param_tag, auto block_constant) {
        using ParamT = decltype(param_tag);
        constexpr int BS = decltype(block_constant)::value;
        ::sg::sm90::prodigy::prodigy_dlr_reduce_kernel<ParamT, ParamT, float, BS>
            <<<grid, BS, 0, stream>>>(
                reinterpret_cast<const ParamT*>(param.data_ptr()),
                reinterpret_cast<const ParamT*>(param_init_cast.data_ptr()),
                s.data_ptr<float>(),
                reinterpret_cast<const ParamT*>(grad.data_ptr()),
                d_prev_dev.data_ptr<float>(),
                numerator.data_ptr<float>(),
                denominator.data_ptr<float>(),
                beta2, N);
    };

    auto dispatch_for_dtype = [&](auto param_tag) {
        if (block == 128) {
            launch_reduce(param_tag, std::integral_constant<int, 128>{});
        } else if (block == 512) {
            launch_reduce(param_tag, std::integral_constant<int, 512>{});
        } else {
            launch_reduce(param_tag, std::integral_constant<int, 256>{});
        }
    };

    switch (param.scalar_type()) {
        case at::kFloat:    dispatch_for_dtype(float{});          break;
        case at::kBFloat16: dispatch_for_dtype(__nv_bfloat16{}); break;
        case at::kHalf:     dispatch_for_dtype(__half{});         break;
        default:
            TORCH_CHECK(false,
                "prodigy_dlr_reduce sm90: unsupported param dtype ",
                param.scalar_type());
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "launch_prodigy_dlr_reduce (sm90): ", cudaGetErrorString(err));
}

}} // namespace sg::sm90
