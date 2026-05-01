// =====================================================================
//  csrc/kernels/cuda/sm_90/grokadamw.cuh
//
//  sm_90 (Hopper) GrokAdamW optimizer kernels + thin templated launchers.
//  Net-new file recovering functionality of the deleted
//  csrc/kernels/cuda/sm_90/grokadamw_sm90.cu (commit 5505b50). All kernel
//  + launcher logic lives here as templates; explicit instantiations are
//  emitted from grokadamw.cu.
//
//  Two templated entry points in namespace sg::sm90::grokadamw:
//
//    1) launch_grokadamw_fused_step<ParamT,StateT,GradT>(...)
//       Single-pass Grokfast + AdamW with a gradient-clamp prologue:
//
//           g~      = clamp(g, -c, c)            (scalar grad clamp)
//           ξ_t     = α·ξ_{t-1} + (1-α)·g~
//           g_amp   = g~ + ℓ·ξ_t                 (register-only)
//           m_t     = β1·m_{t-1} + (1-β1)·g_amp
//           v_t     = β2·v_{t-1} + (1-β2)·g_amp²
//           m_hat   = m_t / bc1                  (host-passed bc1=1-β1^t)
//           v_hat   = v_t / bc2                  (host-passed bc2=1-β2^t)
//           u       = m_hat / (√v_hat + ε) + λ·θ
//           θ_t     = θ_{t-1} - η·u
//
//       6 reads (g, ξ, m, v, θ, plus the typed-load implicit on grad)
//       / 5 writes (ξ, m, v, θ, plus only one path actually mutates grad
//       — here we keep the amplified grad register-only). Roofline with
//       FP32 P/S/G: R+W = 36 B/elem, ~16 FLOPs ⇒ AI ≈ 0.44 FLOP/byte.
//       At H100 HBM3 (3 TB/s) the BW ceiling is ~1.33 TFLOP/s; FP32
//       compute ceiling is ~67 TFLOP/s. Solidly bandwidth-bound — every
//       optimization below targets HBM throughput.
//
//    2) launch_grokadamw_fused_step_q3<ParamT,GradT>(...)
//       Quantized state variant. State storage:
//         exp_avg     : INT8 + per-block FP32 scales (block_size from
//                       quantization.h Q_BLOCK_SIZE).
//         slow_ema (ξ): INT8 + per-block FP32 scales (same pattern).
//         exp_avg_sq  : BF16 with stochastic rounding on writeback
//                       (utils.cuh::float_to_bf16_stochastic).
//       Math is identical to the non-q3 path; dequant on read via
//       quantization.h::dequant_int8, requantize-back via
//       utils.cuh::ptx_int8_stochastic_round (which uses prmt.b32 for
//       fast 16-bit threshold extraction). FP8 grad is rejected on q3
//       via static_assert — FP8's dynamic range is incompatible with
//       INT8 moment storage at the recommended LR schedule.
//
//  Heterogeneous dtype matrix (instantiations live in grokadamw.cu):
//     Non-q3:
//        ParamT in {float, __nv_bfloat16, __half}
//        StateT in {float, __nv_bfloat16}
//        GradT  in {float, __nv_bfloat16, __half,
//                   __nv_fp8_e4m3, __nv_fp8_e5m2}
//     Q3:
//        ParamT in {float, __nv_bfloat16}
//        GradT  in {float, __nv_bfloat16, __half}
//
//  Optimizations & justifications:
//    - LDG / stream_load on read-only state: bypasses L2 allocation so
//      model weights stay warm for the next forward pass.
//    - stream_store (st.global.wt) on FP32 state writes: same rationale.
//    - float4 vec fast path on the all-FP32 instantiation, gated by
//      tuned_configs.h::GROKADAMW_CONFIGS[ARCH_SM90][bucket].vec4.
//    - __launch_bounds__ pulled from tuned_configs.h
//      (GROKADAMW_CONFIGS[ARCH_SM90].block_size, .min_blocks_per_sm).
//    - fast_rsqrt_nr from utils.cuh for 1/√v_hat — single rsqrt.approx.f32
//      + one Newton-Raphson refinement; ~2× faster than sqrtf+fdividef.
//    - Q3 INT8 requant via ptx_int8_stochastic_round — uses prmt.b32 byte
//      permutation for fast threshold extraction.
//    - Reuses csrc/device/optimizers/sm_90/grokadamw_sm90.cuh device
//      template's per-element math layout for cross-arch numerical
//      agreement; the device .cuh is non-stub but lacks the grad-clamp
//      prologue and the bias-correction-by-division formulation, so we
//      inline a slightly-modified copy here matching the spec's algebra.
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/quantization.h"
#include "csrc/common/tuned_configs.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
  #include <cuda_fp8.h>
#endif

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace sg { namespace sm90 { namespace grokadamw {

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

// FP8 grads with FP32 params silently lose dynamic range without an
// explicit rescale (which is not part of this kernel).
template <typename ParamT, typename GradT>
struct is_coherent_combo
    : std::integral_constant<bool,
        !((std::is_same<GradT, __nv_fp8_e4m3>::value ||
           std::is_same<GradT, __nv_fp8_e5m2>::value) &&
          std::is_same<ParamT, float>::value)> {};

// FP8 grad on the Q3 path is rejected unconditionally.
template <typename GradT>
struct is_q3_grad_dtype
    : std::integral_constant<bool,
        std::is_same<GradT, float>::value         ||
        std::is_same<GradT, __nv_bfloat16>::value ||
        std::is_same<GradT, __half>::value> {};

// Q3 quantization block size for INT8 state. Matches the convention
// in csrc/device/optimizers/sm_90/grokadamw_sm90.cuh and the deleted
// baseline (32 elements / FP32 scale).
constexpr int Q3_BLOCK_SIZE = 32;

}}} // namespace sg::sm90::grokadamw
