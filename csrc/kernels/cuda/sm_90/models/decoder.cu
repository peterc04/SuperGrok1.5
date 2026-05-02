// =====================================================================
//  csrc/kernels/cuda/sm_90/models/decoder.cu
//
//  sm_90 (Hopper) Decoder Transformer — explicit template instantiation TU.
//
//  All kernel + orchestration logic lives in decoder.cuh as templates.
//  This TU forces emission of the dtype matrix below into a single
//  object file so callers outside this TU (the pybind layer in
//  csrc/bindings/models_decoder.cpp) can link without dragging the
//  whole header into a non-CUDA source.
//
//  Dtype matrix:
//    ActT == WeightT in {float, __nv_bfloat16, __half}
//
//  (The decoder uses tied dtypes for activations and weights — all
//  GEMMs use cuBLAS with matching CublasTraits<T>.)
// =====================================================================

#include "csrc/kernels/cuda/sm_90/models/decoder.cuh"

namespace sg { namespace sm90 { namespace models { namespace decoder {

// ── Forward instantiations ───────────────────────────────────────────
template cudaError_t forward<float, float>(
    const float*, const float*, float*, float*,
    int, int, int, int, int, int, int, int, cudaStream_t);

template cudaError_t forward<__nv_bfloat16, __nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, int, cudaStream_t);

template cudaError_t forward<__half, __half>(
    const __half*, const __half*, __half*, __half*,
    int, int, int, int, int, int, int, int, cudaStream_t);

// ── Backward instantiations ──────────────────────────────────────────
template cudaError_t backward<float, float>(
    const float*, const float*, const float*,
    float*, float*,
    int, int, int, int, int, int, int, int, cudaStream_t);

template cudaError_t backward<__nv_bfloat16, __nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, int, cudaStream_t);

template cudaError_t backward<__half, __half>(
    const __half*, const __half*, const __half*,
    __half*, __half*,
    int, int, int, int, int, int, int, int, cudaStream_t);

}}}}  // namespace sg::sm90::models::decoder

// ─── Attention kernel instantiations (called from bindings) ──────────
// The pybind layer (csrc/bindings/models_decoder.cpp) calls
// sg::sm90::models::attention::attention_forward/backward<T, 32, true>
// for component-level testing. We instantiate them here so the .cpp
// file (compiled by the host C++ compiler) can link without including
// the attention header (which contains __global__ kernels).

namespace sg { namespace sm90 { namespace models { namespace attention {

template cudaError_t attention_forward<float, 32, true>(
    const float*, const float*, const float*, float*, float*,
    int, int, int, float, cudaStream_t);
template cudaError_t attention_forward<__nv_bfloat16, 32, true>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, float, cudaStream_t);
template cudaError_t attention_forward<__half, 32, true>(
    const __half*, const __half*, const __half*, __half*, __half*,
    int, int, int, float, cudaStream_t);

template cudaError_t attention_backward<float, 32, true>(
    const float*, const float*, const float*, const float*,
    const float*, const float*,
    float*, float*, float*,
    int, int, int, float, cudaStream_t);
template cudaError_t attention_backward<__nv_bfloat16, 32, true>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, float, cudaStream_t);
template cudaError_t attention_backward<__half, 32, true>(
    const __half*, const __half*, const __half*, const __half*,
    const __half*, const __half*,
    __half*, __half*, __half*,
    int, int, int, float, cudaStream_t);

}}}}  // namespace sg::sm90::models::attention
