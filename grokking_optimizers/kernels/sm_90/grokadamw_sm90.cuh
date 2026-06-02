#ifndef GROKKING_KERNELS_SM90_GROKADAMW_SM90_CUH_
#define GROKKING_KERNELS_SM90_GROKADAMW_SM90_CUH_
// ============================================================================
// grokadamw_sm90.cuh — CANONICAL SuperGrok sm_90 device kernels for 'grokadamw'.
//
// This header is the SINGLE source of truth for the sm_90 device logic:
// templated __forceinline__ __device__ update/_vec4 functions, the __global__
// launcher kernels, every inline-PTX (asm-volatile) block VERBATIM, and (for
// muon/supergrok2) the CUTLASS Sm90 tensor-core collectives. It is a
// composition primitive for the future fused megakernel.
//
// The production TU csrc/backends/cuda/sm_90/launch_grokadamw.cu now #include's
// this header and keeps only the host launcher(s) the pybind layer calls.
// Migrated byte-for-byte from that .cu; verified compile-neutral via the
// preprocessor-equivalence gate (nvcc -E, modulo __FILE__).
// ============================================================================
// CUDA sm_90 launch glue for GrokAdamW.
// Algorithm: csrc/algorithms/grokadamw.h

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/algorithms/grokadamw.h"
// ── Autotuner-consumable launch parameters (inlined; see compile.py) ──
#ifndef SG_TUNED_BLOCK_SIZE
#define SG_TUNED_BLOCK_SIZE 256
#endif
#ifndef SG_TUNED_VEC_WIDTH
#define SG_TUNED_VEC_WIDTH 4
#endif
#ifndef SG_TUNED_UNROLL
#define SG_TUNED_UNROLL 1
#endif
#ifndef SG_TUNED_ASYNC_DEPTH
#define SG_TUNED_ASYNC_DEPTH 2
#endif
#include "csrc/backends/cuda/sm_90/primitives.cuh"

// ── inlined from former csrc/common/utils.cuh (Phase3 S0) ──
#if GROK_CUDA
#ifndef SG_INLINE_PTX_PTX_INT8_STOCHASTIC_ROUND
#define SG_INLINE_PTX_PTX_INT8_STOCHASTIC_ROUND
// Stochastic rounding with PTX prmt (permute bytes) for fast bit extraction.
// Replaces the hash_prng shift+multiply chain with a single PTX instruction
// for extracting the random threshold from the hash output.
__device__ __forceinline__ int8_t ptx_int8_stochastic_round(
    float val, float scale, unsigned rand_bits
) {
    float scaled = val / fmaxf(scale, 1e-12f);
    float tr = truncf(scaled);
    float frac = fabsf(scaled - tr);
    // Extract lower 16 bits as threshold using prmt
    unsigned lo16;
    asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(lo16) : "r"(rand_bits));
    float threshold = (float)lo16 / 65536.0f;
    if (frac > threshold) tr += (scaled > 0) ? 1.0f : -1.0f;
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, tr));
}
#endif  // SG_INLINE_PTX_PTX_INT8_STOCHASTIC_ROUND
#endif  // GROK_CUDA

namespace sg { namespace sm90 {

namespace prim = ::sg::sm90::primitives;
using ::sg::algorithms::grokadamw_step;
using ::sg::algorithms::grokadamw_adam_tail;

template <typename ParamT, typename GradT>
__global__ void grokadamw_kernel(
    ParamT* param, float* exp_avg, float* exp_avg_sq, float* ema,
    const GradT* grad,
    float alpha, float lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        grokadamw_step(param, exp_avg, exp_avg_sq, ema, grad,
                       alpha, lamb, lr, beta1, beta2, eps, wd, bc1, bc2, i);
    }
}

void launch_grokadamw_step(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& ema,
    const torch::Tensor& grad,
    float alpha, float lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // §6.1: keep the Adam moments (m, v) L2-resident across the step.
    prim::L2PersistScope l2(stream,
        exp_avg.data_ptr(), exp_avg.nbytes(),
        exp_avg_sq.data_ptr(), exp_avg_sq.nbytes());

    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "grokadamw_step", [&] {
            grokadamw_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                ema.data_ptr<float>(),
                grad.data_ptr<scalar_t>(),
                alpha, lamb, lr, beta1, beta2, eps, wd, bc1, bc2, N);
        });
}


void launch_fused_grokadamw_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2
) {
    launch_grokadamw_step(param, exp_avg, exp_avg_sq, ema, grad,
                          alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                          bc1, bc2);
}

void launch_fused_grokadamw_clip_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2, float clip_threshold
) {
    if (clip_threshold > 0.0f) {
        auto gn = grad.norm().item<float>();
        if (gn > clip_threshold) {
            grad = grad.mul(clip_threshold / gn);
        }
    }
    launch_grokadamw_step(param, exp_avg, exp_avg_sq, ema, grad,
                          alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                          bc1, bc2);
}

// Quantized Config-3: INT8 exp_avg, BF16 exp_avg_sq & ema.
// Dequantize → update in FP32 → re-quantize with stochastic rounding.
template <typename ParamT, typename GradT>
__global__ void grokadamw_q3_kernel(
    ParamT* param,
    int8_t* ea_int8, float* ea_scales,
    __nv_bfloat16* eas_bf16, __nv_bfloat16* ema_bf16,
    const GradT* grad,
    float alpha, float lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, unsigned step, int N, int block_size
) {
    const int stride = prim::grid_stride();
    for (int i = prim::grid_stride_index(); i < N; i += stride) {
        int scale_idx = i / block_size;
        float scale = ea_scales[scale_idx];
        float ea_val = static_cast<float>(ea_int8[i]) * scale;
        float eas_val = __bfloat162float(eas_bf16[i]);
        float ema_val = __bfloat162float(ema_bf16[i]);

        float g = static_cast<float>(grad[i]);
        float p = static_cast<float>(param[i]);

        float ema_new = alpha * ema_val + (1.0f - alpha) * g;
        float g_amp = g + lamb * ema_new;
        // Canonical bias-corrected Adam tail (single-source: algorithms/grokadamw.h).
        // g_eff = g_amp; moments come from the dequantized state, results are
        // re-quantized below — the float math itself lives once in the header.
        float m, v, p_new;
        grokadamw_adam_tail(g_amp, p, ea_val, eas_val,
                            lr, beta1, beta2, eps, wd, bc1, bc2,
                            m, v, p_new);
        param[i] = static_cast<ParamT>(p_new);

        unsigned rng = hash_prng(step, static_cast<unsigned>(i));
        ea_int8[i] = ptx_int8_stochastic_round(m, scale, rng);
        eas_bf16[i] = float_to_bf16_stochastic(v, rng >> 16);
        ema_bf16[i] = float_to_bf16_stochastic(ema_new, rng ^ 0xDEADBEEFu);
    }
}

void launch_fused_grokadamw_step_q3(
    torch::Tensor param,
    torch::Tensor exp_avg_int8, torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_bf16, torch::Tensor ema_bf16,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2, unsigned global_step
) {
    const int64_t N = param.numel();
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int block = SG_TUNED_BLOCK_SIZE;
    const int grid = std::min<int>(65535, (N + block - 1) / block);
    const int q_block_size = std::max<int>(1,
        static_cast<int>(N / exp_avg_scales.numel()));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "grokadamw_q3", [&] {
            grokadamw_q3_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                param.data_ptr<scalar_t>(),
                exp_avg_int8.data_ptr<int8_t>(),
                exp_avg_scales.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(exp_avg_sq_bf16.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(ema_bf16.data_ptr()),
                grad.data_ptr<scalar_t>(),
                alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                bc1, bc2, global_step, N, q_block_size);
        });
}

void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float wd, float eps
) {
    for (size_t i = 0; i < params.size(); i++) {
        launch_grokadamw_step(params[i], exp_avgs[i], exp_avg_sqs[i],
                              emas[i], grads[i],
                              alpha, lamb, lr, beta1, beta2, eps, wd,
                              bc1s[i], bc2s[i]);
    }
}

// AdamW is GrokAdamW with lamb=0 (EMA amplification disabled).
// The ema buffer is not needed; allocate a dummy per-tensor.
void launch_fused_adamw_simple(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    std::vector<int64_t>& steps,
    float beta1, float beta2, float lr, float wd, float eps
) {
    for (size_t t = 0; t < params.size(); t++) {
        const int64_t N = params[t].numel();
        if (N == 0 || !grads[t].defined()) continue;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[t]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[t]));
        auto ema_dummy = torch::zeros_like(exp_avgs[t]);
        launch_grokadamw_step(params[t], exp_avgs[t], exp_avg_sqs[t],
                              ema_dummy, grads[t],
                              0.0f, 0.0f, lr, beta1, beta2, eps, wd,
                              bc1, bc2);
    }
}

}} // namespace sg::sm90

#endif  // GROKKING_KERNELS_SM90_GROKADAMW_SM90_CUH_
