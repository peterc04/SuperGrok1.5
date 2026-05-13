// =====================================================================
// bindings/models_decoder.cpp — pybind11 wrapper for Decoder Transformer
//
// Follows the SG_DISPATCH pattern used by optimizer bindings. Provides
// forward/backward entry points for the full decoder and separate
// attention entry points for component-level testing.
//
// Per-arch launchers live in csrc/kernels/{cuda/<sm>,hip/<gfx>}/ and
// are forward-declared here. The full template definitions live in
// csrc/kernels/cuda/sm_90/models/decoder.cuh and are explicitly
// instantiated in csrc/kernels/cuda/sm_90/models/decoder.cu.
//
// Pybind11 registration is done from models_module.cpp.
// =====================================================================

#include "bindings.h"
#include "_dispatch_macro.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------
// Forward declarations for per-arch launchers.
// (We avoid including the .cuh headers here because they contain
// __global__ kernels that don't compile under the host C++ compiler.
// Instantiations live in csrc/kernels/cuda/sm_90/models/decoder.cu.)
// ---------------------------------------------------------------------

namespace sg { namespace sm90 { namespace models { namespace attention {
 template <typename ActT, int kHeadDim, bool kCausal>
 cudaError_t attention_forward(
 const ActT* q, const ActT* k, const ActT* v,
 ActT* out, ActT* softmax_lse_act,
 int batch, int n_heads, int seq_len,
 float scale, cudaStream_t stream);
 template <typename ActT, int kHeadDim, bool kCausal>
 cudaError_t attention_backward(
 const ActT* grad_out,
 const ActT* q, const ActT* k, const ActT* v,
 const ActT* out, const ActT* softmax_lse_act,
 ActT* grad_q, ActT* grad_k, ActT* grad_v,
 int batch, int n_heads, int seq_len,
 float scale, cudaStream_t stream);
}}}}

namespace sg { namespace sm90 { namespace models { namespace decoder {
 template <typename ActT, typename WeightT>
 cudaError_t forward(const ActT*, const WeightT*, ActT*, ActT*,
 int batch, int seq_len, int d_model,
 int n_heads, int d_head, int n_layers, int vocab_size,
 int ffn_expansion, cudaStream_t);
 template <typename ActT, typename WeightT>
 cudaError_t backward(const ActT*, const ActT*, const WeightT*,
 ActT*, WeightT*,
 int batch, int seq_len, int d_model,
 int n_heads, int d_head, int n_layers, int vocab_size,
 int ffn_expansion, cudaStream_t);
}}}}

namespace sg { namespace gfx942 { namespace models { namespace decoder {
 template <typename ActT, typename WeightT>
 cudaError_t forward(const ActT*, const WeightT*, ActT*, ActT*,
 int batch, int seq_len, int d_model,
 int n_heads, int d_head, int n_layers, int vocab_size,
 int ffn_expansion, cudaStream_t);
 template <typename ActT, typename WeightT>
 cudaError_t backward(const ActT*, const ActT*, const WeightT*,
 ActT*, WeightT*,
 int batch, int seq_len, int d_model,
 int n_heads, int d_head, int n_layers, int vocab_size,
 int ffn_expansion, cudaStream_t);
}}}}

// ---------------------------------------------------------------------
// Public entry points (called from models_module.cpp registration)
// ---------------------------------------------------------------------

namespace sg {

namespace {

inline void check_cuda_err(cudaError_t err, const char* what) {
 TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

} // anonymous namespace

void decoder_forward(
 torch::Tensor input,
 torch::Tensor weights,
 torch::Tensor output,
 torch::Tensor activations,
 int batch_size, int seq_len, int d_model, int n_heads,
 int d_head, int n_layers, int vocab_size, int ffn_expansion
) {
 auto stream = at::cuda::getCurrentCUDAStream().stream();
 auto t = weights.scalar_type();
 cudaError_t err;
 if (t == torch::kFloat32) {
 err = sg::sm90::models::decoder::forward<float, float>(
 input.data_ptr<float>(), weights.data_ptr<float>(),
 output.data_ptr<float>(), activations.data_ptr<float>(),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else if (t == torch::kBFloat16) {
 err = sg::sm90::models::decoder::forward<__nv_bfloat16, __nv_bfloat16>(
 reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(activations.data_ptr()),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else if (t == torch::kFloat16) {
 err = sg::sm90::models::decoder::forward<__half, __half>(
 reinterpret_cast<const __half*>(input.data_ptr()),
 reinterpret_cast<const __half*>(weights.data_ptr()),
 reinterpret_cast<__half*>(output.data_ptr()),
 reinterpret_cast<__half*>(activations.data_ptr()),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else {
 TORCH_CHECK(false, "decoder_forward: unsupported dtype ", t);
 }
 check_cuda_err(err, "decoder_forward");
}

void decoder_backward(
 torch::Tensor grad_output,
 torch::Tensor activations_saved,
 torch::Tensor weights,
 torch::Tensor grad_input,
 torch::Tensor grad_weights,
 int batch_size, int seq_len, int d_model, int n_heads,
 int d_head, int n_layers, int vocab_size, int ffn_expansion
) {
 auto stream = at::cuda::getCurrentCUDAStream().stream();
 auto t = weights.scalar_type();
 cudaError_t err;
 if (t == torch::kFloat32) {
 err = sg::sm90::models::decoder::backward<float, float>(
 grad_output.data_ptr<float>(), activations_saved.data_ptr<float>(),
 weights.data_ptr<float>(),
 grad_input.data_ptr<float>(), grad_weights.data_ptr<float>(),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else if (t == torch::kBFloat16) {
 err = sg::sm90::models::decoder::backward<__nv_bfloat16, __nv_bfloat16>(
 reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(activations_saved.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(grad_weights.data_ptr()),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else if (t == torch::kFloat16) {
 err = sg::sm90::models::decoder::backward<__half, __half>(
 reinterpret_cast<const __half*>(grad_output.data_ptr()),
 reinterpret_cast<const __half*>(activations_saved.data_ptr()),
 reinterpret_cast<const __half*>(weights.data_ptr()),
 reinterpret_cast<__half*>(grad_input.data_ptr()),
 reinterpret_cast<__half*>(grad_weights.data_ptr()),
 batch_size, seq_len, d_model, n_heads, d_head,
 n_layers, vocab_size, ffn_expansion, stream);
 } else {
 TORCH_CHECK(false, "decoder_backward: unsupported dtype ", t);
 }
 check_cuda_err(err, "decoder_backward");
}

void decoder_attention_forward(
 torch::Tensor q, torch::Tensor k, torch::Tensor v,
 torch::Tensor out, torch::Tensor softmax_lse,
 int batch, int n_heads, int seq_len, float scale
) {
 auto stream = at::cuda::getCurrentCUDAStream().stream();
 auto t = q.scalar_type();
 cudaError_t err;
 if (t == torch::kFloat32) {
 err = sg::sm90::models::attention::attention_forward<float, 32, true>(
 q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
 out.data_ptr<float>(),
 reinterpret_cast<float*>(softmax_lse.data_ptr()),
 batch, n_heads, seq_len, scale, stream);
 } else if (t == torch::kBFloat16) {
 err = sg::sm90::models::attention::attention_forward<__nv_bfloat16, 32, true>(
 reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(softmax_lse.data_ptr()),
 batch, n_heads, seq_len, scale, stream);
 } else if (t == torch::kFloat16) {
 err = sg::sm90::models::attention::attention_forward<__half, 32, true>(
 reinterpret_cast<const __half*>(q.data_ptr()),
 reinterpret_cast<const __half*>(k.data_ptr()),
 reinterpret_cast<const __half*>(v.data_ptr()),
 reinterpret_cast<__half*>(out.data_ptr()),
 reinterpret_cast<__half*>(softmax_lse.data_ptr()),
 batch, n_heads, seq_len, scale, stream);
 } else {
 TORCH_CHECK(false, "decoder_attention_forward: unsupported dtype ", t);
 }
 check_cuda_err(err, "decoder_attention_forward");
}

void decoder_attention_backward(
 torch::Tensor grad_out,
 torch::Tensor q, torch::Tensor k, torch::Tensor v,
 torch::Tensor out, torch::Tensor softmax_lse,
 torch::Tensor grad_q, torch::Tensor grad_k, torch::Tensor grad_v,
 int batch, int n_heads, int seq_len, float scale
) {
 auto stream = at::cuda::getCurrentCUDAStream().stream();
 auto t = q.scalar_type();
 cudaError_t err;
 if (t == torch::kFloat32) {
 err = sg::sm90::models::attention::attention_backward<float, 32, true>(
 grad_out.data_ptr<float>(),
 q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
 out.data_ptr<float>(),
 reinterpret_cast<const float*>(softmax_lse.data_ptr()),
 grad_q.data_ptr<float>(), grad_k.data_ptr<float>(), grad_v.data_ptr<float>(),
 batch, n_heads, seq_len, scale, stream);
 } else if (t == torch::kBFloat16) {
 err = sg::sm90::models::attention::attention_backward<__nv_bfloat16, 32, true>(
 reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(out.data_ptr()),
 reinterpret_cast<const __nv_bfloat16*>(softmax_lse.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(grad_q.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(grad_k.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(grad_v.data_ptr()),
 batch, n_heads, seq_len, scale, stream);
 } else if (t == torch::kFloat16) {
 err = sg::sm90::models::attention::attention_backward<__half, 32, true>(
 reinterpret_cast<const __half*>(grad_out.data_ptr()),
 reinterpret_cast<const __half*>(q.data_ptr()),
 reinterpret_cast<const __half*>(k.data_ptr()),
 reinterpret_cast<const __half*>(v.data_ptr()),
 reinterpret_cast<const __half*>(out.data_ptr()),
 reinterpret_cast<const __half*>(softmax_lse.data_ptr()),
 reinterpret_cast<__half*>(grad_q.data_ptr()),
 reinterpret_cast<__half*>(grad_k.data_ptr()),
 reinterpret_cast<__half*>(grad_v.data_ptr()),
 batch, n_heads, seq_len, scale, stream);
 } else {
 TORCH_CHECK(false, "decoder_attention_backward: unsupported dtype ", t);
 }
 check_cuda_err(err, "decoder_attention_backward");
}

} // namespace sg
