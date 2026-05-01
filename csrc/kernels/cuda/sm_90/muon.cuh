// =====================================================================
//  csrc/kernels/cuda/sm_90/muon.cuh
//
//  sm_90 (Hopper) Muon optimizer kernel + launcher header.
//
//  Two public APIs in this header:
//
//   (A) Tensor-typed launchers in `namespace sg::sm90` — match the
//       forward-declarations in csrc/bindings/muon.cpp::DECLARE_MUON(sm90):
//          launch_muon_momentum_normalize(buf, X, grad, momentum, inv_norm)
//          launch_muon_ns_combine(X_out, X, AX, AAX, a, b, c)
//          launch_muon_update(param, orth, neg_lr_scale, decay_factor)
//          launch_muon_ns_combine_update_fused(
//              param, X, AX, AAX, a, b, c, neg_lr_scale, decay_factor)
//
//   (B) Templated raw-pointer entry in `namespace sg::sm90::muon`:
//          launch_muon_fused_step<ParamT, StateT, GradT>(...)
//       Forward-compatible API for a future single-call binding. The
//       quintic coefficients (a, b, c) are passed as kernel args (NOT
//       hardcoded). neg_lr_scale is host-computed as
//       -lr * 0.2 * sqrt(max(m, n)) and consumed verbatim — the polarity
//       is the corrected one (sqrt MULTIPLIED, not divided).
//
//  Algorithm — 2D path (per param matrix theta in R^{m x n}):
//      M_t = mu*M_{t-1} + g                   (momentum)
//      X0  = M_t / ||M_t||_F                  (Frobenius normalize)
//      For i = 1..N_s (~5 iterations):
//          A    = X_{i-1} . X_{i-1}^T          (m x m matmul)
//          AX   = A . X_{i-1}                  (m x n matmul)
//          A2X  = A . AX                       (m x n matmul)
//          X_i  = a*X_{i-1} + b*AX + c*A2X     (quintic Newton-Schulz)
//      theta_t = theta_{t-1}
//                - eta*sqrt(max(m,n))*(X_{N_s} + lambda*theta_{t-1})
//
//  GEMMs use the existing CUTLASS sm_90a wrappers in _cutlass_gemm.cuh
//  (BF16/FP16-in, FP32-acc, FP32-out). No inline `wgmma.mma_async` — the
//  Forbidden list is explicit on this point and CUTLASS BF16/FP16 GEMMs
//  for these shapes hit roofline on H100 without custom WGMMA.
//
//  A Hopper FP8 fast path is GATED on:
//      (1) leading dim (m, n) >= 64
//      (2) availability of sg::cutlass_gemm::hopper_fp8_gemm
//  Currently (2) is NOT defined in _cutlass_gemm.cuh — we leave the
//  threshold + gate wired so the FP8 path is a one-symbol drop-in.
//
//  Heterogeneous dtype matrix (instantiations live in muon.cu):
//     ParamT in {float, __nv_bfloat16, __half}
//     StateT in {float, __nv_bfloat16}
//     GradT  in {float, __nv_bfloat16, __half,
//                __nv_fp8_e4m3, __nv_fp8_e5m2}
//  Incoherent combos (FP8 grad + FP32 param without rescale, FP8 storage
//  on param/state) are statically rejected.
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/common/tuned_configs.h"
#include "csrc/device/optimizers/sm_90/muon_sm90.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) ? (__CUDA_ARCH__ >= 890) : 1
  #include <cuda_fp8.h>
#endif

#ifdef WITH_CUTLASS
#include "csrc/kernels/cuda/_cutlass_gemm.cuh"
#endif

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace sg { namespace sm90 { namespace muon {

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

// FP8 grads require a reduced-precision param (BF16/FP16). The FP8 GEMM
// fast path additionally requires leading dim >= 64; that is a runtime
// gate and is checked in the launcher.
template <typename ParamT, typename GradT>
struct is_coherent_combo
    : std::integral_constant<bool,
        !((std::is_same<GradT, __nv_fp8_e4m3>::value ||
           std::is_same<GradT, __nv_fp8_e5m2>::value) &&
          std::is_same<ParamT, float>::value)> {};

// The Muon Newton-Schulz path materialises X_i as a working copy of the
// param matrix; FP8 storage on the param itself (or on momentum state)
// is rejected — that would require an explicit per-tensor amax rescale
// which is not part of this kernel.
template <typename T>
struct is_fp8 : std::integral_constant<bool,
    std::is_same<T, __nv_fp8_e4m3>::value ||
    std::is_same<T, __nv_fp8_e5m2>::value> {};

}}} // namespace sg::sm90::muon
