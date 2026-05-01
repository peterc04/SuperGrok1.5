// =====================================================================
//  bindings/module.cpp — pybind11 module aggregator
//
//  Replaces the PYBIND11_MODULE block from the deleted csrc/common/ops.cpp.
//  Each per-optimizer file (csrc/bindings/<optimizer>.cpp) defines public
//  entry points in namespace sg::; this file registers them with the
//  Python module so they appear as _ops.<name>.
//
//  Two API surfaces are exposed:
//   - vector-signature entry points (the primary contract — these match
//     the pre-refactor ops.cpp signatures, taking std::vector<torch::Tensor>)
//   - per-tensor wrappers (the legacy small-API; available as escape
//     hatches for tests but not used by the Python optimizers)
// =====================================================================

#include "bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace sg {

// Arch detection
int detect_arch();

// Fused (model, optimizer, arch) dispatch
void fused_step(const std::string& model, const std::string& optimizer,
                torch::Tensor params, torch::Tensor input,
                torch::Tensor grad, torch::Tensor state, float lr);

// ── GrokAdamW ─────────────────────────────────────────────────────────
void grokadamw_step(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, float, float, float, float, float, float,
                    float, float, float);
void grokadamw_clip_step(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, float, float, float, float, float, float,
                         float, float, float, float);
void grokadamw_step_q3(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor, float, float, float, float,
                       float, float, float, float, float, unsigned);
void grokadamw_fused_step(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<int64_t>&,
    float, float, float, float, float, float, float, float);
void fused_adamw_simple_step(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<int64_t>&,
    float, float, float, float, float);

// ── Lion ──────────────────────────────────────────────────────────────
void lion_step(torch::Tensor, torch::Tensor, torch::Tensor,
               float, float, float, float);
void lion_fused_step(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&,
    float, float, float, float);

// ── Grokfast ──────────────────────────────────────────────────────────
void grokfast_ema(torch::Tensor, torch::Tensor, float, float);
void grokfast_adam(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                   torch::Tensor, float, float, float, float, float, float,
                   float, float, float);
void grokfast_fused_step(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    float, float);
void grokfast_fused_ema_adam_step(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<int64_t>&,
    float, float, float, float, float, float, float);

// ── Prodigy ───────────────────────────────────────────────────────────
void prodigy_step(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, float, float, float, float,
                  float, float, float, float);
void prodigy_dlr_reduce(torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor, torch::Tensor, torch::Tensor, float);
float prodigy_fused_step(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<int64_t>&,
    float, float, float, float, float, float);

// ── NeuralGrok ────────────────────────────────────────────────────────
void neuralgrok_amplifier(torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor,
                          int, float, float);
void neuralgrok_adam(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                     float, float, float, float, float, float, float);
void neuralgrok_fused_step(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<int64_t>&,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    float, float, int,
    float, float, float, float, float, float);

// ── LookSAM ───────────────────────────────────────────────────────────
void looksam_perturb(torch::Tensor, torch::Tensor, torch::Tensor, float);
void looksam_restore(torch::Tensor, torch::Tensor);
void looksam_direction_adjust_fused(torch::Tensor, torch::Tensor, torch::Tensor,
                                    float, float, float);
void looksam_norm_reduce(torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> looksam_perturb_all(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, float);
void looksam_restore_all(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&);
void looksam_compute_directions_and_adjust(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, float);

// ── Muon ──────────────────────────────────────────────────────────────
void muon_momentum_normalize(torch::Tensor, torch::Tensor, torch::Tensor,
                             float, float);
void muon_ns_combine(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                     float, float, float);
void muon_ns_combine_update_fused(torch::Tensor, torch::Tensor, torch::Tensor,
                                  torch::Tensor, float, float, float, float, float);
void muon_fused_step(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&,
    float, float, float, int);

// ── SuperGrok v1.5 ────────────────────────────────────────────────────
void sg15_sam_perturb(torch::Tensor, torch::Tensor, float);
void supergrok15_fused_step(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<int64_t>&, std::vector<float>&, std::vector<float>&,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    float, int, float, float, float, float,
    float, float, float, float);
std::vector<torch::Tensor> supergrok15_sam_perturb_all(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, float);
void supergrok15_sharpness_restore_all(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&);

// ── SuperGrok v1.1 ────────────────────────────────────────────────────
void supergrok11_fused_step(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<int64_t>&, std::vector<float>&, std::vector<float>&,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    float, int, float, float, float, float,
    float, float, float, float);
std::vector<torch::Tensor> supergrok11_sam_perturb_all(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, float);
void supergrok11_sharpness_restore_all(
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&);

// ── SuperGrok v2 ──────────────────────────────────────────────────────
void supergrok2_mamba_peer_step(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    float, float, float, float, float, float, float, float, float, float,
    int, int, int, int, int, int, int, int, torch::Tensor);
void supergrok2_mamba_peer_batched_step(
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    std::vector<float>, std::vector<float>,
    std::vector<float>, std::vector<float>, std::vector<float>,
    float, float, float, float, float,
    int, int, int, int, int, int, int, int, torch::Tensor);
void supergrok2_bilevel_fwd_save(
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int, int, int,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int);
void supergrok2_bilevel_fwd_save_batched(
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int, int, int,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int);
void supergrok2_bilevel_backward(
    torch::Tensor, torch::Tensor, torch::Tensor, float,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    int, int, int, int, int, int, int, int, int, int, int, int);
void supergrok2_bilevel_backward_batched(
    torch::Tensor, torch::Tensor,
    torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor,
    torch::Tensor, torch::Tensor,
    int, int, int, int, int);
void supergrok2_prepare_and_batched_step(
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>,
    std::vector<int64_t>, std::vector<double>, std::vector<double>,
    double, double, double, double, double, double,
    double, double, double,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t, int64_t);

// ── MoE ───────────────────────────────────────────────────────────────
void moe_dynamic_expert_load(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor moe_dynamic_expert_fwd(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void moe_dynamic_expert_bwd(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void moe_filter_active_params(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int);
void moe_scan_compacted(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);
void moe_scatter_results(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int);
void moe_count_expert_activations(
    torch::Tensor, torch::Tensor, float, int, int);
torch::Tensor moe_compute_load_balance_loss(
    torch::Tensor, torch::Tensor, int, int);
void moe_apply_frequency_scaling(
    torch::Tensor, torch::Tensor, int, int, float, float, float);

// ── Distributed scan / Quantization / Multi-tensor / Misc ─────────────
// Existing per-tensor entries from the original module.cpp stay registered
// for tests; they remain unchanged.
void distributed_scan_phase1(torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor);
void distributed_scan_phase2(torch::Tensor, torch::Tensor);
void distributed_scan_phase3(torch::Tensor, torch::Tensor);
void fp8_e4m3_quantize(torch::Tensor, torch::Tensor, torch::Tensor);
void int8_symmetric_quantize(torch::Tensor, torch::Tensor, torch::Tensor);
void int4_gptq_quantize(torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor, int);
void mxfp4_quantize(torch::Tensor, torch::Tensor, torch::Tensor, int);

} // namespace sg

// Forward declaration for model bindings aggregator (models_module.cpp)
void register_model_bindings(pybind11::module_& m);

PYBIND11_MODULE(_ops, m) {
    m.doc() = "Grokking Optimizers — specialized per-arch C++/CUDA/HIP kernels";

    m.def("detect_arch", &sg::detect_arch,
          "Returns 90 or 942 for the detected GPU (3-arch active set: "
          "sm_90, gfx942, tpu_v5p). TPU handled in Python.");

    // Fused (model, optimizer, arch) dispatch
    m.def("fused_step", &sg::fused_step,
          "Fused (model, optimizer, arch) kernel dispatch. Routes to the "
          "appropriate fused TU based on detected hardware.");

    // GrokAdamW
    m.def("grokadamw_step", &sg::grokadamw_step);
    m.def("grokadamw_clip_step", &sg::grokadamw_clip_step);
    m.def("grokadamw_step_q3", &sg::grokadamw_step_q3);
    m.def("grokadamw_fused_step", &sg::grokadamw_fused_step,
          "Pre-refactor ops.cpp::grokadamw_fused_step (vector signature)");
    m.def("fused_adamw_simple_step", &sg::fused_adamw_simple_step,
          "Shared simple AdamW step (vector signature)");

    // Lion
    m.def("lion_step", &sg::lion_step);
    m.def("lion_fused_step", &sg::lion_fused_step,
          "Pre-refactor ops.cpp::lion_fused_step (vector signature)");

    // Grokfast
    m.def("grokfast_ema", &sg::grokfast_ema);
    m.def("grokfast_adam", &sg::grokfast_adam);
    m.def("grokfast_fused_step", &sg::grokfast_fused_step,
          "EMA-only multi-tensor grokfast (vector signature)");
    m.def("grokfast_fused_ema_adam_step", &sg::grokfast_fused_ema_adam_step,
          "Per-param EMA + amplification + Adam (vector signature)");

    // Prodigy
    m.def("prodigy_step", &sg::prodigy_step);
    m.def("prodigy_dlr_reduce", &sg::prodigy_dlr_reduce);
    m.def("prodigy_fused_step", &sg::prodigy_fused_step,
          "Pre-refactor ops.cpp::prodigy_fused_step (returns updated d_lr)");

    // NeuralGrok
    m.def("neuralgrok_amplifier", &sg::neuralgrok_amplifier);
    m.def("neuralgrok_adam", &sg::neuralgrok_adam);
    m.def("neuralgrok_fused_step", &sg::neuralgrok_fused_step,
          "Pre-refactor ops.cpp::neuralgrok_fused_step (vector signature)");

    // LookSAM
    m.def("looksam_perturb", &sg::looksam_perturb);
    m.def("looksam_restore", &sg::looksam_restore);
    m.def("looksam_direction_adjust_fused", &sg::looksam_direction_adjust_fused);
    m.def("looksam_norm_reduce", &sg::looksam_norm_reduce);
    m.def("looksam_perturb_all", &sg::looksam_perturb_all,
          "Vector SAM perturb; returns clones to restore from");
    m.def("looksam_restore_all", &sg::looksam_restore_all);
    m.def("looksam_compute_directions_and_adjust",
          &sg::looksam_compute_directions_and_adjust,
          "Fused direction + adjust over a list of params");

    // Muon
    m.def("muon_momentum_normalize", &sg::muon_momentum_normalize);
    m.def("muon_ns_combine", &sg::muon_ns_combine);
    m.def("muon_ns_combine_update_fused", &sg::muon_ns_combine_update_fused);
    m.def("muon_fused_step", &sg::muon_fused_step,
          "Pre-refactor ops.cpp::muon_fused_step (vector signature)");

    // SG v1.5
    m.def("sg15_sam_perturb", &sg::sg15_sam_perturb);
    m.def("supergrok15_fused_step", &sg::supergrok15_fused_step);
    m.def("supergrok15_sam_perturb_all", &sg::supergrok15_sam_perturb_all);
    m.def("supergrok15_sharpness_restore_all", &sg::supergrok15_sharpness_restore_all);

    // SG v1.1
    m.def("supergrok11_fused_step", &sg::supergrok11_fused_step);
    m.def("supergrok11_sam_perturb_all", &sg::supergrok11_sam_perturb_all);
    m.def("supergrok11_sharpness_restore_all", &sg::supergrok11_sharpness_restore_all);

    // MoE
    m.def("moe_dynamic_expert_load", &sg::moe_dynamic_expert_load);
    m.def("moe_dynamic_expert_fwd", &sg::moe_dynamic_expert_fwd);
    m.def("moe_dynamic_expert_bwd", &sg::moe_dynamic_expert_bwd);
    m.def("moe_filter_active_params", &sg::moe_filter_active_params);
    m.def("moe_scan_compacted", &sg::moe_scan_compacted);
    m.def("moe_scatter_results", &sg::moe_scatter_results);
    m.def("moe_count_expert_activations", &sg::moe_count_expert_activations);
    m.def("moe_compute_load_balance_loss", &sg::moe_compute_load_balance_loss);
    m.def("moe_apply_frequency_scaling", &sg::moe_apply_frequency_scaling);

    // Distributed scan
    m.def("distributed_scan_phase1", &sg::distributed_scan_phase1);
    m.def("distributed_scan_phase2", &sg::distributed_scan_phase2);
    m.def("distributed_scan_phase3", &sg::distributed_scan_phase3);

    // Quantization
    m.def("fp8_e4m3_quantize", &sg::fp8_e4m3_quantize);
    m.def("int8_symmetric_quantize", &sg::int8_symmetric_quantize);
    m.def("int4_gptq_quantize", &sg::int4_gptq_quantize);
    m.def("mxfp4_quantize", &sg::mxfp4_quantize);

    // SuperGrok v2 — Mamba-3+PEER meta-net entries.
    // Wrappers in csrc/bindings/supergrok2.cpp dispatch to the per-arch
    // launchers in csrc/kernels/{cuda/<sm>,hip/<gfx>}/supergrok2_{fwd,bwd}_<arch>.
    m.def("supergrok2_mamba_peer_step", &sg::supergrok2_mamba_peer_step,
          "SuperGrok2: per-param Mamba-3+PEER meta-net + mu + Adam + WD");
    m.def("supergrok2_mamba_peer_batched_step", &sg::supergrok2_mamba_peer_batched_step,
          "SuperGrok2: batched Mamba-3+PEER step for all params at once");
    m.def("supergrok2_bilevel_fwd_save", &sg::supergrok2_bilevel_fwd_save,
          "SuperGrok2 bilevel: forward scan with state saving for backward");
    m.def("supergrok2_bilevel_fwd_save_batched", &sg::supergrok2_bilevel_fwd_save_batched,
          "SuperGrok2 bilevel: batched forward scan with state saving");
    m.def("supergrok2_bilevel_backward", &sg::supergrok2_bilevel_backward,
          "SuperGrok2 bilevel: full backward through meta-net");
    m.def("supergrok2_bilevel_backward_batched", &sg::supergrok2_bilevel_backward_batched,
          "SuperGrok2 bilevel: batched full backward through meta-net");
    m.def("supergrok2_prepare_and_batched_step", &sg::supergrok2_prepare_and_batched_step,
          "Fused multi-tensor grad prepare (clip, finite, bias corrections) "
          "+ batched mamba-3+PEER step. Replaces N per-parameter Python "
          "iterations with one kernel launch.");

    // Model bindings (decoder, vit, mamba) — registered as _ops.models.*
    register_model_bindings(m);
}
