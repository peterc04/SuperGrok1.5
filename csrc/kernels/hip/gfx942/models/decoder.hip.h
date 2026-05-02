// csrc/kernels/hip/gfx942/models/decoder.hip.h
// Decoder model header for gfx942 (CDNA3 / MI300X).
//
// Strategy: this is a thin wrapper around the sm_90 implementation. The
// sm_90 decoder.cuh is platform-portable — it uses cuBLAS (which hipify
// remaps to rocBLAS via PyTorch's CUDAExtension), at::cuda::* helpers
// (which work on both backends), and standard CUDA/HIP runtime calls. The
// only sm_90-only code paths are guarded by `WITH_CUTLASS`, which is NOT
// defined on the HIP build.
//
// At call time, `sg::gfx942::models::decoder::forward<ActT, WeightT>(...)`
// delegates directly to `sg::sm90::models::decoder::forward<ActT, WeightT>(...)`.
// The wrappers are inline so the compiler emits a single body per
// instantiation in the gfx942 namespace; the sm_90 instantiation lives in
// the same .hip.cpp TU (decoder.hip.cpp) since the sm_90 decoder.cu is not
// part of the HIP build.
//
// The bindings (csrc/bindings/models_decoder.cpp) forward-declare these
// gfx942 symbols using `cudaError_t` and `cudaStream_t`; under PyTorch's
// HIP build these resolve to `hipError_t` / `hipStream_t` automatically.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
// Bring in the full sm_90 decoder template implementation. On HIP the
// includes (cublas_v2.h, cuda_bf16.h, etc.) are remapped to their ROCm
// equivalents by PyTorch's hipification step.
#include "csrc/kernels/cuda/sm_90/models/decoder.cuh"

namespace sg { namespace gfx942 { namespace models { namespace decoder {

// -- Model configuration ------------------------------------------------------
// Kept for callers that want a structured config; the launcher accepts
// loose parameters to match the sm_90 contract used by the bindings.
struct ModelConfig {
    int d_model;
    int n_heads;
    int head_dim;
    int n_layers;
    int seq_len;
    int vocab_size;
    int batch;
    float attn_scale;       // typically 1/sqrt(head_dim)
    int lds_bytes;          // LDS allocation budget
    int waves_per_eu;       // occupancy hint for CDNA3
};

// -- Forward pass -------------------------------------------------------------
// Mirrors sm_90 decoder::forward<ActT, WeightT>; delegates to the sm_90
// implementation, which is platform-portable on the cuBLAS/rocBLAS path.
template <typename ActT, typename WeightT>
inline cudaError_t forward(
    const ActT* input,
    const WeightT* weights,
    ActT* output,
    ActT* activations,
    int batch, int seq_len, int d_model, int n_heads, int d_head,
    int n_layers, int vocab_size, int ffn_expansion,
    cudaStream_t stream
) {
    return sg::sm90::models::decoder::forward<ActT, WeightT>(
        input, weights, output, activations,
        batch, seq_len, d_model, n_heads, d_head,
        n_layers, vocab_size, ffn_expansion, stream);
}

// -- Backward pass ------------------------------------------------------------
template <typename ActT, typename WeightT>
inline cudaError_t backward(
    const ActT* grad_output,
    const ActT* activations_saved,
    const WeightT* weights,
    ActT* grad_input,
    WeightT* grad_weights,
    int batch, int seq_len, int d_model, int n_heads, int d_head,
    int n_layers, int vocab_size, int ffn_expansion,
    cudaStream_t stream
) {
    return sg::sm90::models::decoder::backward<ActT, WeightT>(
        grad_output, activations_saved, weights,
        grad_input, grad_weights,
        batch, seq_len, d_model, n_heads, d_head,
        n_layers, vocab_size, ffn_expansion, stream);
}

}}}}  // namespace sg::gfx942::models::decoder
