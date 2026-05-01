// =====================================================================
//  bindings/models_decoder.cpp — pybind11 wrapper for Decoder Transformer
//
//  Follows the SG_DISPATCH pattern used by optimizer bindings. Provides
//  forward/backward entry points for the full decoder and separate
//  attention entry points for component-level testing.
//
//  Per-arch launchers live in csrc/kernels/{cuda/<sm>,hip/<gfx>}/ and
//  are forward-declared here. Until those TUs are compiled, all entry
//  points raise with a clear diagnostic.
//
//  Pybind11 registration is done from models_module.cpp.
// =====================================================================

#include "bindings.h"
#include "_dispatch_macro.h"

#include <torch/extension.h>

// ---------------------------------------------------------------------
// Forward declarations for per-arch launchers.
// Signatures must match the definitions in per-arch .cu/.hip.cpp files.
// ---------------------------------------------------------------------

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

void decoder_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor output,
    torch::Tensor activations,
    int batch_size, int seq_len, int d_model, int n_heads,
    int d_head, int n_layers, int vocab_size, int ffn_expansion
) {
    TORCH_CHECK(false, "decoder_forward: per-arch kernel not yet compiled");
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
    TORCH_CHECK(false, "decoder_backward: per-arch kernel not yet compiled");
}

void decoder_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor out, torch::Tensor softmax_lse,
    int batch, int n_heads, int seq_len, float scale
) {
    TORCH_CHECK(false, "decoder_attention_forward: per-arch kernel not yet compiled");
}

void decoder_attention_backward(
    torch::Tensor grad_out,
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor out, torch::Tensor softmax_lse,
    torch::Tensor grad_q, torch::Tensor grad_k, torch::Tensor grad_v,
    int batch, int n_heads, int seq_len, float scale
) {
    TORCH_CHECK(false, "decoder_attention_backward: per-arch kernel not yet compiled");
}

} // namespace sg
