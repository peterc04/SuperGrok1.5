// =====================================================================
//  csrc/backends/hip/gfx942/models/vit.hip.cpp
//
//  gfx942 (CDNA3 / MI300X) ViT — explicit template instantiation TU.
//  Mirrors csrc/backends/cuda/sm_90/models/vit.cu but for the HIP build.
//
//  Instantiates:
//    1. gfx942 wrappers (sg::gfx942::models::vit::forward/backward/patch_project)
//    2. sm_90 templates (sg::sm90::models::vit::* and the non-causal
//       sm_90 attention<T,32,false> kernels) — required because the sm_90
//       vit.cu instantiation TU is excluded from the HIP build, while the
//       binding csrc/bindings/models_vit.cpp still references sm_90
//       symbols at the call site even when arch-dispatch routes to gfx942.
//
//  Dtype matrix (homogeneous): {float, __nv_bfloat16, __half}.
// =====================================================================

#include "csrc/backends/hip/gfx942/models/vit.hip.h"
// vit.hip.h pulls in csrc/backends/cuda/sm_90/models/vit.cuh which itself
// includes attention.cuh, so the sm_90 attention templates are already
// visible here.

// ─── gfx942 wrapper instantiations ───────────────────────────────────
namespace sg { namespace gfx942 { namespace models { namespace vit {

#define INSTANTIATE_GFX942_VIT(ActT, WeightT)                                   \
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

INSTANTIATE_GFX942_VIT(float, float);
INSTANTIATE_GFX942_VIT(__nv_bfloat16, __nv_bfloat16);
INSTANTIATE_GFX942_VIT(__half, __half);

#undef INSTANTIATE_GFX942_VIT

}}}} // namespace sg::gfx942::models::vit

// ─── sm_90 ViT instantiations ────────────────────────────────────────
// The sm_90 vit.cu instantiation TU is not part of the HIP build, but
// csrc/bindings/models_vit.cpp references sg::sm90::models::vit symbols
// at the call site. Re-emit the dtype matrix here for the HIP link.
namespace sg { namespace sm90 { namespace models { namespace vit {

#define INSTANTIATE_SM90_VIT(ActT, WeightT)                                     \
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

INSTANTIATE_SM90_VIT(float, float);
INSTANTIATE_SM90_VIT(__nv_bfloat16, __nv_bfloat16);
INSTANTIATE_SM90_VIT(__half, __half);

#undef INSTANTIATE_SM90_VIT

}}}} // namespace sg::sm90::models::vit

// ─── sm_90 attention<causal=false, kHeadDim=32> instantiations ───────
// vit.cuh implicitly instantiates these from forward/backward, but
// emit them explicitly to harden against link order — mirroring what
// sm_90/models/vit.cu does on the CUDA side.
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

}}}} // namespace sg::sm90::models::attention
