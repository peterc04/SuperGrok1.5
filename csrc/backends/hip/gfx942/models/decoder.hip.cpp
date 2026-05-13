// =====================================================================
//  csrc/backends/hip/gfx942/models/decoder.hip.cpp
//
//  gfx942 (CDNA3 / MI300X) Decoder Transformer — explicit template
//  instantiation TU.
//
//  This TU mirrors csrc/backends/cuda/sm_90/models/decoder.cu but in the
//  HIP build. It instantiates:
//
//    1. The gfx942 wrappers (sg::gfx942::models::decoder::forward/backward)
//       declared in decoder.hip.h. These are thin inline shims that
//       delegate to the sm_90 implementation.
//
//    2. The underlying sm_90 templates (sg::sm90::models::decoder::forward
//       and ::backward, plus sm_90 attention forward/backward). Under the
//       HIP build the sm_90 .cu instantiation TU is NOT compiled, so the
//       HIP build must emit these symbols itself. The binding
//       (csrc/bindings/models_decoder.cpp) calls
//       sg::sm90::models::decoder::forward directly under the cuBLAS path
//       (which hipify remaps to rocBLAS via PyTorch's CUDAExtension).
//
//  Dtype matrix (homogeneous): {float, __nv_bfloat16, __half}.
//  __nv_bfloat16 / __half alias to __hip_bfloat16 / _Float16 on HIP via
//  PyTorch's hipification of <cuda_bf16.h> / <cuda_fp16.h>.
// =====================================================================

#include "csrc/backends/hip/gfx942/models/decoder.hip.h"
// decoder.hip.h pulls in csrc/backends/cuda/sm_90/models/decoder.cuh
// which itself includes attention.cuh, so the sm_90 attention templates
// are already visible here.

// ─── gfx942 wrapper instantiations ───────────────────────────────────
namespace sg { namespace gfx942 { namespace models { namespace decoder {

template cudaError_t forward<float, float>(
    const float*, const float*, float*, float*,
    int, int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t forward<__nv_bfloat16, __nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t forward<__half, __half>(
    const __half*, const __half*, __half*, __half*,
    int, int, int, int, int, int, int, int, cudaStream_t);

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

}}}}  // namespace sg::gfx942::models::decoder

// ─── sm_90 decoder instantiations ────────────────────────────────────
// Binding csrc/bindings/models_decoder.cpp calls sm_90 symbols directly.
// Since the sm_90 decoder.cu instantiation TU is not part of the HIP
// build, we instantiate the same symbols here so the HIP link succeeds.
namespace sg { namespace sm90 { namespace models { namespace decoder {

template cudaError_t forward<float, float>(
    const float*, const float*, float*, float*,
    int, int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t forward<__nv_bfloat16, __nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t forward<__half, __half>(
    const __half*, const __half*, __half*, __half*,
    int, int, int, int, int, int, int, int, cudaStream_t);

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

// ─── sm_90 attention<causal=true, kHeadDim=32> instantiations ────────
// The decoder binding TU also calls sg::sm90::models::attention::
// attention_forward/backward<T, 32, true> for component-level testing.
// Mirror the symbol set from sm_90/models/decoder.cu so the HIP link
// resolves these.
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
