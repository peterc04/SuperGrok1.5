// =====================================================================
// bindings/models_module.cpp — model bindings aggregator
//
// Registers all model entry points (decoder, vit, mamba) into the _ops
// pybind11 module under a "models" submodule, so they appear as
// _ops.models.<name> in Python.
//
// Each per-model file (csrc/bindings/models_<model>.cpp) defines public
// entry points in namespace sg::; this file binds them to pybind11.
// =====================================================================

#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ---------------------------------------------------------------------
// Forward declarations — Decoder
// ---------------------------------------------------------------------
namespace sg {

void decoder_forward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int, int);
void decoder_backward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int, int);
void decoder_attention_forward(
 torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 int, int, int, float);
void decoder_attention_backward(
 torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, float);

// ---------------------------------------------------------------------
// Forward declarations — ViT
// ---------------------------------------------------------------------

void vit_forward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int, int, int, int, int);
void vit_backward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int, int, int, int, int);
void vit_attention_forward(
 torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 int, int, int, float);
void vit_attention_backward(
 torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, float);
void vit_patch_project(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int);

// ---------------------------------------------------------------------
// Forward declarations — Mamba
// ---------------------------------------------------------------------

void mamba_forward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int);
void mamba_backward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int, int);
void mamba_layer_forward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int, int, int, int);
void mamba_selective_scan_forward(
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 int, int, int);
void mamba_selective_scan_backward(
 torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor,
 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 int, int, int);

} // namespace sg

// ---------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------

void register_model_bindings(py::module_& m) {
 auto models = m.def_submodule("models",
 "Per-arch optimized model kernels (decoder, vit, mamba)");

 // Decoder Transformer
 models.def("decoder_forward", &sg::decoder_forward,
 "Decoder Transformer forward pass");
 models.def("decoder_backward", &sg::decoder_backward,
 "Decoder Transformer backward pass");
 models.def("decoder_attention_forward", &sg::decoder_attention_forward,
 "Decoder multi-head attention forward (component test)");
 models.def("decoder_attention_backward", &sg::decoder_attention_backward,
 "Decoder multi-head attention backward (component test)");

 // Vision Transformer
 models.def("vit_forward", &sg::vit_forward,
 "Vision Transformer forward pass");
 models.def("vit_backward", &sg::vit_backward,
 "Vision Transformer backward pass");
 models.def("vit_attention_forward", &sg::vit_attention_forward,
 "ViT multi-head attention forward (component test)");
 models.def("vit_attention_backward", &sg::vit_attention_backward,
 "ViT multi-head attention backward (component test)");
 models.def("vit_patch_project", &sg::vit_patch_project,
 "ViT patch projection (component test)");

 // Mamba (SSM)
 models.def("mamba_forward", &sg::mamba_forward,
 "Mamba SSM forward pass");
 models.def("mamba_backward", &sg::mamba_backward,
 "Mamba SSM backward pass");
 models.def("mamba_layer_forward", &sg::mamba_layer_forward,
 "Mamba single-layer forward (component test)");
 models.def("mamba_selective_scan_forward", &sg::mamba_selective_scan_forward,
 "Mamba selective scan forward (component test)");
 models.def("mamba_selective_scan_backward", &sg::mamba_selective_scan_backward,
 "Mamba selective scan backward (component test)");
}
