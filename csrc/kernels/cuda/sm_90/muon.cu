// =====================================================================
//  csrc/kernels/cuda/sm_90/muon.cu
//
//  sm_90 (Hopper) Muon kernel — explicit instantiation TU.
//
//  All kernel + launcher logic lives in muon.cuh as templates. This TU
//  forces emission of the dtype matrix below into a single object file,
//  and defines the four torch::Tensor-based launchers that match the
//  forward declarations in csrc/bindings/muon.cpp::DECLARE_MUON(sm90):
//
//    sg::sm90::launch_muon_momentum_normalize
//    sg::sm90::launch_muon_ns_combine
//    sg::sm90::launch_muon_update
//    sg::sm90::launch_muon_ns_combine_update_fused
//
//  The vector / single-shot orchestrator (muon_fused_step) lives in
//  bindings/muon.cpp; it composes the four launchers per training step.
//  The templated raw-pointer entry sg::sm90::muon::launch_muon_fused_step
//  is also instantiated below for a future single-call binding (the
//  binding does not currently call it).
//
//  Dtype matrix (per main spec):
//    ParamT in {float, __nv_bfloat16, __half}
//    StateT in {float, __nv_bfloat16}
//    GradT  in {float, __nv_bfloat16, __half,
//               __nv_fp8_e4m3, __nv_fp8_e5m2}
//
//  Incoherent combos are statically rejected:
//    * FP8 grad + FP32 param (no rescale path)        -> rejected
//    * FP8 storage on param OR state                  -> rejected
//    * FP16 state                                     -> rejected (BF16-only)
//
//  Hopper FP8 fast path: gated on leading dim m, n >= 64
//  (sg::sm90::muon::kFp8MinLeadingDim) AND availability of
//  sg::cutlass_gemm::hopper_fp8_gemm. The latter is NOT yet wired into
//  _cutlass_gemm.cuh, so the gate currently falls through to the BF16
//  CUTLASS path. The threshold lives in muon.cuh as a one-symbol switch.
//
//  neg_lr_scale polarity: the binding (muon.cpp:124) computes
//  -lr * 0.2 * sqrt(max(m, n)) — sqrt MULTIPLIED, not divided. The
//  kernel consumes the scalar verbatim. The corrected polarity is
//  preserved here.
// =====================================================================

// Force the four sg::sm90:: launcher bodies to be emitted into THIS TU.
#define SG_SM90_MUON_DEFINE_LAUNCHER 1

#include "csrc/kernels/cuda/sm_90/muon.cuh"

namespace sg { namespace sm90 { namespace muon {

// ---------------------------------------------------------------------
// Templated raw-pointer entry — instantiated for future single-call
// binding. The active binding path uses the four launchers below.
// ---------------------------------------------------------------------
//
// FP32 param family. FP8 grad + FP32 param is rejected by static_assert
// (is_coherent_combo<float, FP8>) — those rows are absent below.

template cudaError_t launch_muon_fused_step<float, float, float>(
    float*, float*, const float*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<float, float, __nv_bfloat16>(
    float*, float*, const __nv_bfloat16*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<float, float, __half>(
    float*, float*, const __half*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

// FP32 param + BF16 state.
template cudaError_t launch_muon_fused_step<float, __nv_bfloat16, float>(
    float*, __nv_bfloat16*, const float*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<float, __nv_bfloat16, __nv_bfloat16>(
    float*, __nv_bfloat16*, const __nv_bfloat16*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<float, __nv_bfloat16, __half>(
    float*, __nv_bfloat16*, const __half*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

// ---------------------------------------------------------------------
// BF16 param family — full grad cross-section (incl. FP8 grads).
// ---------------------------------------------------------------------

template cudaError_t launch_muon_fused_step<__nv_bfloat16, float, float>(
    __nv_bfloat16*, float*, const float*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__nv_bfloat16, float, __nv_bfloat16>(
    __nv_bfloat16*, float*, const __nv_bfloat16*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__nv_bfloat16, float, __half>(
    __nv_bfloat16*, float*, const __half*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__nv_bfloat16, float, __nv_fp8_e4m3>(
    __nv_bfloat16*, float*, const __nv_fp8_e4m3*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__nv_bfloat16, float, __nv_fp8_e5m2>(
    __nv_bfloat16*, float*, const __nv_fp8_e5m2*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__nv_bfloat16, __nv_bfloat16, float>(
    __nv_bfloat16*, __nv_bfloat16*, const float*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__nv_bfloat16, __nv_bfloat16, __half>(
    __nv_bfloat16*, __nv_bfloat16*, const __half*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3>(
    __nv_bfloat16*, __nv_bfloat16*, const __nv_fp8_e4m3*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__nv_bfloat16, __nv_bfloat16, __nv_fp8_e5m2>(
    __nv_bfloat16*, __nv_bfloat16*, const __nv_fp8_e5m2*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

// ---------------------------------------------------------------------
// FP16 param family — full grad cross-section.
// ---------------------------------------------------------------------

template cudaError_t launch_muon_fused_step<__half, float, float>(
    __half*, float*, const float*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__half, float, __nv_bfloat16>(
    __half*, float*, const __nv_bfloat16*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__half, float, __half>(
    __half*, float*, const __half*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__half, float, __nv_fp8_e4m3>(
    __half*, float*, const __nv_fp8_e4m3*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__half, float, __nv_fp8_e5m2>(
    __half*, float*, const __nv_fp8_e5m2*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__half, __nv_bfloat16, float>(
    __half*, __nv_bfloat16*, const float*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__half, __nv_bfloat16, __nv_bfloat16>(
    __half*, __nv_bfloat16*, const __nv_bfloat16*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__half, __nv_bfloat16, __half>(
    __half*, __nv_bfloat16*, const __half*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__half, __nv_bfloat16, __nv_fp8_e4m3>(
    __half*, __nv_bfloat16*, const __nv_fp8_e4m3*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

template cudaError_t launch_muon_fused_step<__half, __nv_bfloat16, __nv_fp8_e5m2>(
    __half*, __nv_bfloat16*, const __nv_fp8_e5m2*,
    int, int, int,
    float, float, float,
    float, float, float,
    float, int64_t, cudaStream_t);

}}} // namespace sg::sm90::muon
