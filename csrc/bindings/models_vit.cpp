// =====================================================================
//  bindings/models_vit.cpp — pybind11 wrapper for Vision Transformer
//
//  Follows the SG_DISPATCH pattern used by optimizer bindings. Provides
//  forward/backward entry points for the full ViT, separate attention
//  entry points, and a patch projection entry point for testing.
//
//  Per-arch launchers live in csrc/kernels/{cuda/<sm>,hip/<gfx>}/ and
//  are forward-declared here.
//
//  Pybind11 registration is done from models_module.cpp.
// =====================================================================

#include "bindings.h"
#include "_dispatch_macro.h"

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

// ---------------------------------------------------------------------
// Forward declarations for per-arch launchers.
// ---------------------------------------------------------------------

namespace sg { namespace sm90 { namespace models { namespace vit {
    template <typename ActT, typename WeightT>
    cudaError_t forward(const ActT*, const WeightT*, ActT*, ActT*,
                       int batch, int channels, int height, int width,
                       int patch_size, int d_model, int n_heads, int d_head,
                       int n_layers, int n_classes, int ffn_expansion,
                       cudaStream_t);
    template <typename ActT, typename WeightT>
    cudaError_t backward(const ActT*, const ActT*, const WeightT*,
                        ActT*, WeightT*,
                        int batch, int channels, int height, int width,
                        int patch_size, int d_model, int n_heads, int d_head,
                        int n_layers, int n_classes, int ffn_expansion,
                        cudaStream_t);
    template <typename ActT, typename WeightT>
    cudaError_t patch_project(const ActT*, const WeightT*, const WeightT*, ActT*,
                             int batch, int num_patches, int patch_dim, int d_model,
                             cudaStream_t);
}}}}

namespace sg { namespace sm90 { namespace models { namespace attention {
    template <typename ActT, int kHeadDim, bool kCausal>
    cudaError_t attention_forward(
        const ActT* q, const ActT* k, const ActT* v, ActT* out,
        ActT* softmax_lse_act,
        int batch, int n_heads, int seq_len, float scale, cudaStream_t);
    template <typename ActT, int kHeadDim, bool kCausal>
    cudaError_t attention_backward(
        const ActT* grad_out, const ActT* q, const ActT* k, const ActT* v,
        const ActT* out, const ActT* softmax_lse_act,
        ActT* grad_q, ActT* grad_k, ActT* grad_v,
        int batch, int n_heads, int seq_len, float scale, cudaStream_t);
}}}}

namespace sg { namespace gfx942 { namespace models { namespace vit {
    template <typename ActT, typename WeightT>
    cudaError_t forward(const ActT*, const WeightT*, ActT*, ActT*,
                       int batch, int channels, int height, int width,
                       int patch_size, int d_model, int n_heads, int d_head,
                       int n_layers, int n_classes, int ffn_expansion,
                       cudaStream_t);
    template <typename ActT, typename WeightT>
    cudaError_t backward(const ActT*, const ActT*, const WeightT*,
                        ActT*, WeightT*,
                        int batch, int channels, int height, int width,
                        int patch_size, int d_model, int n_heads, int d_head,
                        int n_layers, int n_classes, int ffn_expansion,
                        cudaStream_t);
}}}}

// ---------------------------------------------------------------------
// Public entry points (called from models_module.cpp registration)
// ---------------------------------------------------------------------

namespace sg {

namespace {

inline cudaStream_t default_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

template <typename ActT>
cudaError_t do_forward(
    const void* input, const void* weights, void* output, void* activations,
    int batch, int channels, int height, int width,
    int patch_size, int d_model, int n_heads, int d_head,
    int n_layers, int n_classes, int ffn_expansion,
    cudaStream_t stream
) {
    int a = ::sg::detect_arch();
    if (a == 90) {
        return sg::sm90::models::vit::forward<ActT, ActT>(
            static_cast<const ActT*>(input),
            static_cast<const ActT*>(weights),
            static_cast<ActT*>(output),
            static_cast<ActT*>(activations),
            batch, channels, height, width,
            patch_size, d_model, n_heads, d_head,
            n_layers, n_classes, ffn_expansion, stream);
    }
    if (a == 942) {
        return sg::gfx942::models::vit::forward<ActT, ActT>(
            static_cast<const ActT*>(input),
            static_cast<const ActT*>(weights),
            static_cast<ActT*>(output),
            static_cast<ActT*>(activations),
            batch, channels, height, width,
            patch_size, d_model, n_heads, d_head,
            n_layers, n_classes, ffn_expansion, stream);
    }
    throw std::runtime_error("vit_forward: unsupported arch " + std::to_string(a));
}

template <typename ActT>
cudaError_t do_backward(
    const void* grad_output, const void* activations_saved, const void* weights,
    void* grad_input, void* grad_weights,
    int batch, int channels, int height, int width,
    int patch_size, int d_model, int n_heads, int d_head,
    int n_layers, int n_classes, int ffn_expansion,
    cudaStream_t stream
) {
    int a = ::sg::detect_arch();
    if (a == 90) {
        return sg::sm90::models::vit::backward<ActT, ActT>(
            static_cast<const ActT*>(grad_output),
            static_cast<const ActT*>(activations_saved),
            static_cast<const ActT*>(weights),
            static_cast<ActT*>(grad_input),
            static_cast<ActT*>(grad_weights),
            batch, channels, height, width,
            patch_size, d_model, n_heads, d_head,
            n_layers, n_classes, ffn_expansion, stream);
    }
    if (a == 942) {
        return sg::gfx942::models::vit::backward<ActT, ActT>(
            static_cast<const ActT*>(grad_output),
            static_cast<const ActT*>(activations_saved),
            static_cast<const ActT*>(weights),
            static_cast<ActT*>(grad_input),
            static_cast<ActT*>(grad_weights),
            batch, channels, height, width,
            patch_size, d_model, n_heads, d_head,
            n_layers, n_classes, ffn_expansion, stream);
    }
    throw std::runtime_error("vit_backward: unsupported arch " + std::to_string(a));
}

inline void check_cuda(cudaError_t err, const char* what) {
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

} // anonymous namespace

void vit_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor output,
    torch::Tensor activations,
    int batch_size, int channels, int height, int width,
    int patch_size, int d_model, int n_heads, int d_head,
    int n_layers, int n_classes, int ffn_expansion
) {
    TORCH_CHECK(input.is_cuda(), "vit_forward: input must be on CUDA");
    TORCH_CHECK(weights.scalar_type() == input.scalar_type(),
                "vit_forward: weights/input dtype mismatch");
    TORCH_CHECK(output.scalar_type() == input.scalar_type(),
                "vit_forward: output/input dtype mismatch");
    cudaStream_t stream = default_stream();
    cudaError_t err = cudaSuccess;
    auto t = input.scalar_type();
    if (t == torch::kFloat32) {
        err = do_forward<float>(input.data_ptr(), weights.data_ptr(),
                                output.data_ptr(), activations.data_ptr(),
                                batch_size, channels, height, width,
                                patch_size, d_model, n_heads, d_head,
                                n_layers, n_classes, ffn_expansion, stream);
    } else if (t == torch::kBFloat16) {
        err = do_forward<__nv_bfloat16>(input.data_ptr(), weights.data_ptr(),
                                output.data_ptr(), activations.data_ptr(),
                                batch_size, channels, height, width,
                                patch_size, d_model, n_heads, d_head,
                                n_layers, n_classes, ffn_expansion, stream);
    } else if (t == torch::kFloat16) {
        err = do_forward<__half>(input.data_ptr(), weights.data_ptr(),
                                output.data_ptr(), activations.data_ptr(),
                                batch_size, channels, height, width,
                                patch_size, d_model, n_heads, d_head,
                                n_layers, n_classes, ffn_expansion, stream);
    } else {
        TORCH_CHECK(false, "vit_forward: unsupported dtype ", t);
    }
    check_cuda(err, "vit_forward");
}

void vit_backward(
    torch::Tensor grad_output,
    torch::Tensor activations_saved,
    torch::Tensor weights,
    torch::Tensor grad_input,
    torch::Tensor grad_weights,
    int batch_size, int channels, int height, int width,
    int patch_size, int d_model, int n_heads, int d_head,
    int n_layers, int n_classes, int ffn_expansion
) {
    TORCH_CHECK(grad_output.is_cuda(), "vit_backward: grad_output must be on CUDA");
    TORCH_CHECK(weights.scalar_type() == grad_output.scalar_type(),
                "vit_backward: weights/grad_output dtype mismatch");
    cudaStream_t stream = default_stream();
    cudaError_t err = cudaSuccess;
    auto t = grad_output.scalar_type();
    if (t == torch::kFloat32) {
        err = do_backward<float>(
            grad_output.data_ptr(), activations_saved.data_ptr(), weights.data_ptr(),
            grad_input.data_ptr(), grad_weights.data_ptr(),
            batch_size, channels, height, width,
            patch_size, d_model, n_heads, d_head,
            n_layers, n_classes, ffn_expansion, stream);
    } else if (t == torch::kBFloat16) {
        err = do_backward<__nv_bfloat16>(
            grad_output.data_ptr(), activations_saved.data_ptr(), weights.data_ptr(),
            grad_input.data_ptr(), grad_weights.data_ptr(),
            batch_size, channels, height, width,
            patch_size, d_model, n_heads, d_head,
            n_layers, n_classes, ffn_expansion, stream);
    } else if (t == torch::kFloat16) {
        err = do_backward<__half>(
            grad_output.data_ptr(), activations_saved.data_ptr(), weights.data_ptr(),
            grad_input.data_ptr(), grad_weights.data_ptr(),
            batch_size, channels, height, width,
            patch_size, d_model, n_heads, d_head,
            n_layers, n_classes, ffn_expansion, stream);
    } else {
        TORCH_CHECK(false, "vit_backward: unsupported dtype ", t);
    }
    check_cuda(err, "vit_backward");
}

void vit_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor out, torch::Tensor softmax_lse,
    int batch, int n_heads, int seq_len, float scale
) {
    TORCH_CHECK(q.is_cuda(), "vit_attention_forward: tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32,
                "vit_attention_forward: only fp32 is wired through this entry; "
                "use vit_forward for bf16/fp16");
    cudaStream_t stream = default_stream();
    cudaError_t err = sg::sm90::models::attention::attention_forward<
        float, /*kHeadDim=*/32, /*kCausal=*/false>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(),
        softmax_lse.data_ptr<float>(),
        batch, n_heads, seq_len, scale, stream);
    check_cuda(err, "vit_attention_forward");
}

void vit_attention_backward(
    torch::Tensor grad_out,
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor out, torch::Tensor softmax_lse,
    torch::Tensor grad_q, torch::Tensor grad_k, torch::Tensor grad_v,
    int batch, int n_heads, int seq_len, float scale
) {
    TORCH_CHECK(q.is_cuda(), "vit_attention_backward: tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32,
                "vit_attention_backward: only fp32 is wired through this entry");
    cudaStream_t stream = default_stream();
    cudaError_t err = sg::sm90::models::attention::attention_backward<
        float, /*kHeadDim=*/32, /*kCausal=*/false>(
        grad_out.data_ptr<float>(),
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(),
        softmax_lse.data_ptr<float>(),
        grad_q.data_ptr<float>(), grad_k.data_ptr<float>(), grad_v.data_ptr<float>(),
        batch, n_heads, seq_len, scale, stream);
    check_cuda(err, "vit_attention_backward");
}

void vit_patch_project(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch, int channels, int height, int width,
    int patch_size, int d_model
) {
    TORCH_CHECK(input.is_cuda(), "vit_patch_project: input must be CUDA");
    TORCH_CHECK(height % patch_size == 0 && width % patch_size == 0,
                "vit_patch_project: image dims must be divisible by patch_size");
    int num_patches = (height / patch_size) * (width / patch_size);
    int patch_dim = patch_size * patch_size * channels;
    TORCH_CHECK(input.numel() == (int64_t)batch * num_patches * patch_dim,
                "vit_patch_project: input must be [B*num_patches*patch_dim]");
    TORCH_CHECK(output.numel() == (int64_t)batch * num_patches * d_model,
                "vit_patch_project: output must be [B*num_patches*d_model]");
    cudaStream_t stream = default_stream();
    cudaError_t err = cudaSuccess;
    auto t = input.scalar_type();
    const void* bias_ptr = bias.defined() && bias.numel() > 0 ? bias.data_ptr() : nullptr;
    int a = ::sg::detect_arch();
    if (a != 90) {
        throw std::runtime_error(
            "vit_patch_project: unsupported arch " + std::to_string(a));
    }
    if (t == torch::kFloat32) {
        err = sg::sm90::models::vit::patch_project<float, float>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            static_cast<const float*>(bias_ptr),
            output.data_ptr<float>(),
            batch, num_patches, patch_dim, d_model, stream);
    } else if (t == torch::kBFloat16) {
        err = sg::sm90::models::vit::patch_project<__nv_bfloat16, __nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            static_cast<const __nv_bfloat16*>(bias_ptr),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            batch, num_patches, patch_dim, d_model, stream);
    } else if (t == torch::kFloat16) {
        err = sg::sm90::models::vit::patch_project<__half, __half>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            static_cast<const __half*>(bias_ptr),
            reinterpret_cast<__half*>(output.data_ptr()),
            batch, num_patches, patch_dim, d_model, stream);
    } else {
        TORCH_CHECK(false, "vit_patch_project: unsupported dtype ", t);
    }
    check_cuda(err, "vit_patch_project");
}

} // namespace sg
