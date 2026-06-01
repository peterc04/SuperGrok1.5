// =====================================================================
//  csrc/backends/hip/gfx942/models/mamba.hip.cpp
//
//  gfx942 (CDNA3 / MI300X) Mamba — explicit template instantiation TU.
//  Mirrors csrc/backends/cuda/sm_90/models/mamba.cu but for the HIP build.
//
//  Instantiates:
//    1. gfx942 wrappers (sg::gfx942::models::mamba::{forward, backward,
//       selective_scan_fwd, selective_scan_bwd})
//    2. sm_90 templates (sg::sm90::models::mamba::*) — required because
//       the sm_90 mamba.cu instantiation TU is excluded from the HIP
//       build, while the binding csrc/bindings/models_mamba.cpp still
//       references sm_90 symbols at the call sites that compile under
//       both arches.
//
//  Dtype matrix: {float, __nv_bfloat16, __half}.
// =====================================================================

#include "csrc/backends/hip/gfx942/models/mamba.hip.h"

// ─── gfx942 wrapper instantiations ───────────────────────────────────
namespace sg { namespace gfx942 { namespace models { namespace mamba {

// forward / backward: full-stack
template cudaError_t forward<float>(
    const float*, const float*, float*, float*,
    int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t forward<__half>(
    const __half*, const __half*, __half*, __half*,
    int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t forward<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, cudaStream_t);

template cudaError_t backward<float>(
    const float*, const float*, const float*,
    float*, float*,
    int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t backward<__half>(
    const __half*, const __half*, const __half*,
    __half*, __half*,
    int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t backward<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, cudaStream_t);

// selective_scan_fwd / bwd: component-test wrappers
template cudaError_t selective_scan_fwd<float>(
    const float*, const float*, const float*, const float*,
    float*, float*,
    int, int, int, cudaStream_t);
template cudaError_t selective_scan_fwd<__half>(
    const __half*, const __half*, const __half*, const __half*,
    __half*, __half*,
    int, int, int, cudaStream_t);
template cudaError_t selective_scan_fwd<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, cudaStream_t);

template cudaError_t selective_scan_bwd<float>(
    const float*, const float*, const float*, const float*,
    const float*, const float*,
    float*, float*, float*, float*,
    int, int, int, cudaStream_t);
template cudaError_t selective_scan_bwd<__half>(
    const __half*, const __half*, const __half*, const __half*,
    const __half*, const __half*,
    __half*, __half*, __half*, __half*,
    int, int, int, cudaStream_t);
template cudaError_t selective_scan_bwd<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, cudaStream_t);

}}}}  // namespace sg::gfx942::models::mamba

// ─── sm_90 mamba instantiations ──────────────────────────────────────
// Required because csrc/bindings/models_mamba.cpp references sm_90
// symbols at the call site even when arch-dispatch routes to gfx942 at
// runtime, and the sm_90 mamba.cu instantiation TU is excluded from the
// HIP build.
namespace sg { namespace sm90 { namespace models { namespace mamba {

template cudaError_t forward<float>(
    const float*, const float*, float*, float*,
    int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t forward<__half>(
    const __half*, const __half*, __half*, __half*,
    int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t forward<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, cudaStream_t);

template cudaError_t backward<float>(
    const float*, const float*, const float*,
    float*, float*,
    int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t backward<__half>(
    const __half*, const __half*, const __half*,
    __half*, __half*,
    int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t backward<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, cudaStream_t);

template cudaError_t selective_scan_fwd<float>(
    const float*, const float*, const float*, const float*,
    float*, float*,
    int, int, int, cudaStream_t);
template cudaError_t selective_scan_fwd<__half>(
    const __half*, const __half*, const __half*, const __half*,
    __half*, __half*,
    int, int, int, cudaStream_t);
template cudaError_t selective_scan_fwd<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, cudaStream_t);

template cudaError_t selective_scan_bwd<float>(
    const float*, const float*, const float*, const float*,
    const float*, const float*,
    float*, float*, float*, float*,
    int, int, int, cudaStream_t);
template cudaError_t selective_scan_bwd<__half>(
    const __half*, const __half*, const __half*, const __half*,
    const __half*, const __half*,
    __half*, __half*, __half*, __half*,
    int, int, int, cudaStream_t);
template cudaError_t selective_scan_bwd<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, cudaStream_t);

}}}}  // namespace sg::sm90::models::mamba
