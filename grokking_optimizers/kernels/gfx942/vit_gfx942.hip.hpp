#ifndef GROKKING_KERNELS_GFX942_VIT_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_VIT_GFX942_HIP_HPP_
// ============================================================================
// vit_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 'vit' model logic.
//
// AMDGCN-asm status: NOT PRESENT in the production path. This path is ATen +
// rocBLAS (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm requires
// migrating the model TU from .hip.cpp to .hip (hipcc-routed); roadmap item 2.
//
// The production location csrc/backends/hip/gfx942/models/vit.hip.h is now a
// thin shim #include'ing this header, so its vit.hip.cpp TU resolves unchanged.
// Migrated byte-for-byte from that header.
// ============================================================================
// csrc/backends/hip/gfx942/models/vit.hip.h
// Vision Transformer model header for gfx942 (CDNA3 / MI300X).
//
// Strategy: this is a thin wrapper around the sm_90 implementation.
// vit.cuh is platform-portable — it uses cuBLAS (hipify-remapped to
// rocBLAS), at::cuda::* helpers, and standard runtime calls. There are
// no sm_90-only intrinsics in vit.cuh proper.
//
// The bindings (csrc/bindings/models_vit.cpp) call
// `sg::gfx942::models::vit::forward<ActT,ActT>` directly when
// `detect_arch() == 942`. The wrappers below provide those symbols and
// delegate to the sm_90 implementation.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
// Bring in the full sm_90 ViT template implementation. On HIP the cuBLAS
// includes are hipified to rocBLAS by PyTorch's CUDAExtension.
#include "csrc/backends/cuda/sm_90/models/vit.cuh"

namespace sg { namespace gfx942 { namespace models { namespace vit {

// -- Model configuration ------------------------------------------------------
struct ModelConfig {
    int d_model;
    int n_heads;
    int head_dim;
    int n_layers;
    int seq_len;        // num_patches + 1 (CLS token)
    int patch_size;
    int image_size;
    int num_classes;
    int batch;
    float attn_scale;   // typically 1/sqrt(head_dim)
    int lds_bytes;      // LDS allocation budget
    int waves_per_eu;   // occupancy hint for CDNA3
};

// -- Forward pass -------------------------------------------------------------
template <typename ActT, typename WeightT>
inline cudaError_t forward(
    const ActT* input,
    const WeightT* weights,
    ActT* output,
    ActT* activations,
    int batch, int channels, int height, int width,
    int patch_size, int d_model, int n_heads, int d_head,
    int n_layers, int n_classes, int ffn_expansion,
    cudaStream_t stream
) {
    return sg::sm90::models::vit::forward<ActT, WeightT>(
        input, weights, output, activations,
        batch, channels, height, width,
        patch_size, d_model, n_heads, d_head,
        n_layers, n_classes, ffn_expansion, stream);
}

// -- Backward pass ------------------------------------------------------------
template <typename ActT, typename WeightT>
inline cudaError_t backward(
    const ActT* grad_output,
    const ActT* activations_saved,
    const WeightT* weights,
    ActT* grad_input,
    WeightT* grad_weights,
    int batch, int channels, int height, int width,
    int patch_size, int d_model, int n_heads, int d_head,
    int n_layers, int n_classes, int ffn_expansion,
    cudaStream_t stream
) {
    return sg::sm90::models::vit::backward<ActT, WeightT>(
        grad_output, activations_saved, weights,
        grad_input, grad_weights,
        batch, channels, height, width,
        patch_size, d_model, n_heads, d_head,
        n_layers, n_classes, ffn_expansion, stream);
}

// -- Patch projection (component-level) ---------------------------------------
// Provided for parity with sm_90::vit::patch_project. The current binding
// explicitly errors for arch != 90, so this wrapper is unused at runtime
// but instantiated for symbol completeness.
template <typename ActT, typename WeightT>
inline cudaError_t patch_project(
    const ActT* input,
    const WeightT* weight,
    const WeightT* bias,
    ActT* output,
    int batch, int num_patches, int patch_dim, int d_model,
    cudaStream_t stream
) {
    return sg::sm90::models::vit::patch_project<ActT, WeightT>(
        input, weight, bias, output,
        batch, num_patches, patch_dim, d_model, stream);
}

}}}}  // namespace sg::gfx942::models::vit

#endif  // GROKKING_KERNELS_GFX942_VIT_GFX942_HIP_HPP_
