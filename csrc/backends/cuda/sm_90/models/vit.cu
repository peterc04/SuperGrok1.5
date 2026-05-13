// =====================================================================
//  csrc/backends/cuda/sm_90/models/vit.cu
//
//  sm_90 (Hopper) Vision Transformer — explicit template instantiation TU.
//
//  All kernel + launcher logic lives in vit.cuh as templates. This TU
//  forces emission of the {ActT, WeightT} matrix below into a single
//  object file so callers outside this TU (the pybind11 bindings in
//  csrc/bindings/models_vit.cpp) can link against the per-tensor entry
//  points without dragging the whole header into a non-CUDA source.
//
//  Dtype matrix (matches the ViT spec — homogeneous ActT==WeightT):
//      {float, __nv_bfloat16, __half}
//
//  Heterogeneous mixes (e.g. fp32 weights + bf16 activations) are not
//  exposed; the binding contract uses one type parameter (ActT==WeightT)
//  in practice via SG_DISPATCH on the input dtype.
// =====================================================================

#include "csrc/backends/cuda/sm_90/models/vit.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace sg { namespace sm90 { namespace models { namespace vit {

// ---------------------------------------------------------------------
// forward / backward — homogeneous dtype matrix
// ---------------------------------------------------------------------
#define INSTANTIATE_VIT(ActT, WeightT)                                          \
    template cudaError_t forward<ActT, WeightT>(                                \
        const ActT*, const WeightT*, ActT*, ActT*,                              \
        int, int, int, int, int, int, int, int, int, int, int,                  \
        cudaStream_t);                                                          \
    template cudaError_t backward<ActT, WeightT>(                               \
        const ActT*, const ActT*, const WeightT*, ActT*, WeightT*,              \
        int, int, int, int, int, int, int, int, int, int, int,                  \
        cudaStream_t);                                                          \
    template cudaError_t patch_project<ActT, WeightT>(                          \
        const ActT*, const WeightT*, const WeightT*, ActT*,                     \
        int, int, int, int, cudaStream_t)

INSTANTIATE_VIT(float, float);
INSTANTIATE_VIT(__nv_bfloat16, __nv_bfloat16);
INSTANTIATE_VIT(__half, __half);

#undef INSTANTIATE_VIT

}}}} // namespace sg::sm90::models::vit

// ---------------------------------------------------------------------
// Re-emit the non-causal attention<float/bf16/fp16, 32> instantiations
// so the binding TU (csrc/bindings/models_vit.cpp), which can only see
// forward declarations, resolves at link time. These mirror what
// vit::forward/backward already implicitly instantiate, but are made
// explicit here to harden against link order.
// ---------------------------------------------------------------------
namespace sg { namespace sm90 { namespace models { namespace attention {

template cudaError_t attention_forward<float, 32, false>(
    const float*, const float*, const float*, float*, float*,
    int, int, int, float, cudaStream_t);
template cudaError_t attention_forward<__nv_bfloat16, 32, false>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, float, cudaStream_t);
template cudaError_t attention_forward<__half, 32, false>(
    const __half*, const __half*, const __half*, __half*, __half*,
    int, int, int, float, cudaStream_t);

template cudaError_t attention_backward<float, 32, false>(
    const float*, const float*, const float*, const float*,
    const float*, const float*,
    float*, float*, float*,
    int, int, int, float, cudaStream_t);
template cudaError_t attention_backward<__nv_bfloat16, 32, false>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, float, cudaStream_t);
template cudaError_t attention_backward<__half, 32, false>(
    const __half*, const __half*, const __half*, const __half*,
    const __half*, const __half*,
    __half*, __half*, __half*,
    int, int, int, float, cudaStream_t);

}}}}
