// =====================================================================
//  bindings/module.cpp — pybind11 module aggregator
//
//  Replaces the PYBIND11_MODULE block from csrc/common/ops.cpp.
//  Each per-optimizer file (csrc/bindings/<optimizer>.cpp) defines a
//  public C++ entry point in namespace sg::; this file registers them
//  with the Python module.
//
//  TODO(structural-refactor): the original ops.cpp registered a large
//  surface (~50 entry points). The list below is the worked-example
//  subset that matches the per-optimizer dispatchers added in this
//  refactor pass. The remaining entry points (SG2 batched fwd-save,
//  SG2 batched backward, MoE backward, etc.) should be added as the
//  per-arch launcher signatures are filled in.
//
//  The Python loader (grokking_optimizers/_ops_loader.py) imports this
//  module and exposes its entry points to the optimizer Python files.
// =====================================================================

#include "bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace sg {

// Forward declarations of the dispatcher entry points. Definitions are
// in the per-optimizer csrc/bindings/<optimizer>.cpp files.
//
// Arch detection
int detect_arch();

// GrokAdamW
void grokadamw_step(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, float, float, float, float, float, float,
                    float, float, float);
void grokadamw_clip_step(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, float, float, float, float, float, float,
                         float, float, float, float);
void grokadamw_step_q3(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor, float, float, float, float,
                       float, float, float, float, float, unsigned);

// Lion
void lion_step(torch::Tensor, torch::Tensor, torch::Tensor,
               float, float, float, float);

// Grokfast
void grokfast_ema(torch::Tensor, torch::Tensor, float, float);
void grokfast_adam(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                   torch::Tensor, float, float, float, float, float, float,
                   float, float, float);

// Prodigy
void prodigy_step(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, float, float, float, float,
                  float, float, float, float);
void prodigy_dlr_reduce(torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor, torch::Tensor, torch::Tensor, float);

// NeuralGrok
void neuralgrok_amplifier(torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor,
                          int, float, float);
void neuralgrok_adam(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                     float, float, float, float, float, float, float);

// LookSAM
void looksam_perturb(torch::Tensor, torch::Tensor, torch::Tensor, float);
void looksam_restore(torch::Tensor, torch::Tensor);
void looksam_direction_adjust_fused(torch::Tensor, torch::Tensor, torch::Tensor,
                                    float, float, float);
void looksam_norm_reduce(torch::Tensor, torch::Tensor, torch::Tensor);

// Muon
void muon_momentum_normalize(torch::Tensor, torch::Tensor, torch::Tensor,
                             float, float);
void muon_ns_combine(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                     float, float, float);
void muon_ns_combine_update_fused(torch::Tensor, torch::Tensor, torch::Tensor,
                                  torch::Tensor, float, float, float, float, float);
void muon_fused_step(torch::Tensor, torch::Tensor, torch::Tensor,
                     int, float, float, float);

// SuperGrok v1.5
void sg15_fused_step(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int, float, float, float,
    float, float, float, float, float, float, float);
void sg15_sam_perturb(torch::Tensor, torch::Tensor, float);

// SuperGrok v1.1
void sg11_fused_step(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int, float, float, float,
    float, float, float, float, float, float, float);
void sg11_cosine_gate_reduce(torch::Tensor, torch::Tensor, torch::Tensor);

// Multi-tensor
void multi_tensor_grokadamw(
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>,
    float, float, float, float, float, float, float, float, float);
void multi_tensor_lion(
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, float, float, float, float);

// Distributed scan (3 phases)
void distributed_scan_phase1(torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor);
void distributed_scan_phase2(torch::Tensor, torch::Tensor);
void distributed_scan_phase3(torch::Tensor, torch::Tensor);

// MoE (primary forward)
void moe_expert_fwd(torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor);

// Quantization
void fp8_e4m3_quantize(torch::Tensor, torch::Tensor, torch::Tensor);
void int8_symmetric_quantize(torch::Tensor, torch::Tensor, torch::Tensor);
void int4_gptq_quantize(torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor, int);
void mxfp4_quantize(torch::Tensor, torch::Tensor, torch::Tensor, int);

} // namespace sg

PYBIND11_MODULE(_ops, m) {
    m.doc() = "Grokking Optimizers — specialized per-arch C++/CUDA/HIP kernels";

    m.def("detect_arch", &sg::detect_arch,
          "Returns 80, 90, 100, or 942 for the detected GPU. Raises on unsupported.");

    // GrokAdamW
    m.def("grokadamw_step", &sg::grokadamw_step);
    m.def("grokadamw_clip_step", &sg::grokadamw_clip_step);
    m.def("grokadamw_step_q3", &sg::grokadamw_step_q3);

    // Lion
    m.def("lion_step", &sg::lion_step);

    // Grokfast
    m.def("grokfast_ema", &sg::grokfast_ema);
    m.def("grokfast_adam", &sg::grokfast_adam);

    // Prodigy
    m.def("prodigy_step", &sg::prodigy_step);
    m.def("prodigy_dlr_reduce", &sg::prodigy_dlr_reduce);

    // NeuralGrok
    m.def("neuralgrok_amplifier", &sg::neuralgrok_amplifier);
    m.def("neuralgrok_adam", &sg::neuralgrok_adam);

    // LookSAM
    m.def("looksam_perturb", &sg::looksam_perturb);
    m.def("looksam_restore", &sg::looksam_restore);
    m.def("looksam_direction_adjust_fused", &sg::looksam_direction_adjust_fused);
    m.def("looksam_norm_reduce", &sg::looksam_norm_reduce);

    // Muon
    m.def("muon_momentum_normalize", &sg::muon_momentum_normalize);
    m.def("muon_ns_combine", &sg::muon_ns_combine);
    m.def("muon_ns_combine_update_fused", &sg::muon_ns_combine_update_fused);
    m.def("muon_fused_step", &sg::muon_fused_step);

    // SG v1.5 / v1.1
    m.def("sg15_fused_step", &sg::sg15_fused_step);
    m.def("sg15_sam_perturb", &sg::sg15_sam_perturb);
    m.def("sg11_fused_step", &sg::sg11_fused_step);
    m.def("sg11_cosine_gate_reduce", &sg::sg11_cosine_gate_reduce);

    // Multi-tensor
    m.def("multi_tensor_grokadamw", &sg::multi_tensor_grokadamw);
    m.def("multi_tensor_lion", &sg::multi_tensor_lion);

    // Distributed scan
    m.def("distributed_scan_phase1", &sg::distributed_scan_phase1);
    m.def("distributed_scan_phase2", &sg::distributed_scan_phase2);
    m.def("distributed_scan_phase3", &sg::distributed_scan_phase3);

    // MoE
    m.def("moe_expert_fwd", &sg::moe_expert_fwd);

    // Quantization
    m.def("fp8_e4m3_quantize", &sg::fp8_e4m3_quantize);
    m.def("int8_symmetric_quantize", &sg::int8_symmetric_quantize);
    m.def("int4_gptq_quantize", &sg::int4_gptq_quantize);
    m.def("mxfp4_quantize", &sg::mxfp4_quantize);

    // TODO(structural-refactor): the original csrc/common/ops.cpp registered
    // additional entry points (SG2 batched fwd-save, SG2 batched backward,
    // MoE backward, secondary MoE kernels, gradient compression utilities,
    // CUDA graph helpers). These are still served by csrc/common/ops.cpp
    // until the per-optimizer bindings expand to cover them.
}
