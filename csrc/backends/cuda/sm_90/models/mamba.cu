// csrc/backends/cuda/sm_90/models/mamba.cu
// Translation unit that instantiates the Mamba model templates declared in
// mamba.cuh for {float, __half, __nv_bfloat16}. The header contains the
// full implementation; this TU only forces explicit instantiation so the
// binding (csrc/bindings/models_mamba.cpp) can link against concrete
// symbols.

#include "csrc/backends/cuda/sm_90/models/mamba.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace sg { namespace sm90 { namespace models { namespace mamba {

// ── forward / backward: full-stack ──────────────────────────────────
// forward has an extra optional activation_cache parameter (default nullptr).
template cudaError_t forward<float>(
    const float*, const float*, float*, float*,
    int, int, int, int, int, int, int, cudaStream_t, float*);
template cudaError_t forward<__half>(
    const __half*, const __half*, __half*, __half*,
    int, int, int, int, int, int, int, cudaStream_t, __half*);
template cudaError_t forward<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, cudaStream_t, __nv_bfloat16*);

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

// ── selective_scan_fwd / bwd: component-test wrappers ────────────────
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
