// =====================================================================
// bindings.cpp — pybind11 module + all per-optimizer dispatchers
//
// Consolidated from 16 per-optimizer dispatcher files + module.cpp.
// Each section below preserves the original file's content verbatim;
// the only changes are: (1) #include lines for the deleted internal
// headers were stripped, (2) a single set of shared #includes lives at
// the top of this file.
//
// Pybind11 registration order is preserved from the original module.cpp.
// =====================================================================

#include "helpers.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <string>
#include <stdexcept>


// ─── csrc/bindings/supergrok2.cpp (SG2 megakernel tail only) ───────
// PURE L3-TC: every EAGER per-op binding (the per-optimizer *_fused_step +
// per-tensor helpers, MoE, and the eager SuperGrok2 CSA/HCA step / batched /
// bilevel / prepare wrappers + their DECLARE_* per-arch launcher forward-decls)
// is removed — the race runs the fused L3-TC megakernels only (ops.fused_step /
// ops.sg2_fused_step). The ONLY survivors of the old supergrok2.cpp section are
// the two extern decls below: sg2_meta_optimizer_tail + sg2_ws_stride, whose
// DEFINITIONS live in the nvcc-compiled csrc/fused/sm_90/sg2_meta_tail.cu (the
// SuperGrok2 full meta-net as ONE persistent megakernel). bindings.cpp is
// host-only so it cannot host the <<<>>> launch — it only extern-declares +
// PYBIND-registers them.
// ---------------------------------------------------------------------

namespace sg {

void sg2_meta_optimizer_tail(
    torch::Tensor params_packed, torch::Tensor grads_packed, torch::Tensor sharpness_packed,
    torch::Tensor exp_avg_packed, torch::Tensor exp_avg_sq_packed,
    torch::Tensor mu_packed, torch::Tensor slow_packed,
    torch::Tensor gru_state_packed, torch::Tensor perm_packed, torch::Tensor unsort_packed,
    torch::Tensor n_per_tensor, torch::Tensor row_off,
    torch::Tensor workspace, int64_t ws_stride,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, torch::Tensor csa_out_W,
    torch::Tensor csa_compress_w, torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_K,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, torch::Tensor hca_out_W,
    torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1, torch::Tensor expert_W2, torch::Tensor expert_b2,
    torch::Tensor alpha, torch::Tensor gru_decay, torch::Tensor lamb_eff,
    torch::Tensor beta1, torch::Tensor bc1, torch::Tensor bc2,
    double rescale, double beta2, double lr, double wd, double eps,
    torch::Tensor g_next_task, torch::Tensor g_arrived, torch::Tensor g_generation);

int64_t sg2_ws_stride(int64_t Nmax);
} // namespace sg





// ─── csrc/bindings/models_module.cpp ───────────────────────────────
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
// Registration
// ---------------------------------------------------------------------

// OWNER-DECIDED REMOVAL (2026-06-10): the `_ops.models.*` submodule exported
// 13 per-op model kernels (decoder/vit/mamba fwd+bwd + attention/scan/patch
// component entries) with ZERO Python consumers anywhere (race, tuner, tests —
// verified by the cleanup inventory and re-verified at deletion). They were
// pre-megakernel scaffolding; the portable reuse surface for model math is the
// stage-header layer (csrc/fused/sm_90/model_stage*.cuh, COMPONENT_CONTRACT.md),
// not pybind exports — an exported, untested, unused API is liability, not
// portability. The underlying sg::* host functions + kernels are excised in the
// dead-code pass that accompanies the dispatch-table prune (kept this commit so
// the binding removal is independently revertible).
void register_model_bindings(py::module_& m) {
 (void)m;  // intentionally registers nothing — see removal note above
}


// ─── csrc/bindings/module.cpp (PYBIND11 entry) ─────────────────
// Forward declaration for model bindings aggregator (models_module.cpp)
void register_model_bindings(pybind11::module_& m);

// ── Exported-ABI schema version (A5-F6) ──────────────────────────────
// Single integer that versions the SET of exported pybind signatures below
// (every m.def's argument list + return type, and the secondary-list/scalar
// contracts the Python optimizers rely on). BUMP THIS on ANY exported-signature
// change: adding/removing/reordering a fused-step argument, changing a tensor
// list into a scalar (or vice-versa), changing a return tuple, or renaming an
// exported symbol. A stale prebuilt .so paired with newer Python wrappers (or
// the reverse) otherwise mis-marshals arguments silently; a version mismatch
// must fail loudly instead.
//
// The Python-side assertion (compare _ops.__abi_schema__ against the value the
// wrappers were written for) lands SEPARATELY in grokking_optimizers/dispatch.py
// (sibling-owned). Until that check exists this attribute is exported but inert
// — that is intentional and harmless: it merely makes the version observable
// now so the Python guard can be added without a coordinated ABI bump.
constexpr int GROK_ABI_SCHEMA = 1;

// ── Module name (#12 tuner-JIT fix) ──────────────────────────────────
// The module must export the PyInit_<name> of the name it is BUILT as, or the
// importing side can never bind it. TORCH_EXTENSION_NAME is supplied by BOTH
// build paths:
//   • product/AOT: torch's BuildExtension defines it from the last component
//     of the extension name ("grokking_optimizers._ops" → _ops), so the
//     shipped .so still exports PyInit__ops — identical to the old pinned
//     name;
//   • JIT (compile.py autotuner variants): torch.utils.cpp_extension.load()
//     defines it to the variant module name (grokking_compiled_<opt>_<model>_
//     <arch>_<cfg>), so load()'s final import-by-name step — structurally
//     impossible under the old pinned `_ops` (#12: no JIT variant could ever
//     import; the autotuner could not time real variants) — now succeeds.
// SG_OPS_PYMODULE is a DELIBERATE level of indirection, not noise:
// setup.py::_collect and compile.py::_owns_extension_module_tu use the
// literal text `PYBIND11_MODULE(` immediately followed by the torch
// extension-name macro as the content marker for standalone self-test/driver
// TUs that must be EXCLUDED from _ops-style builds. Spelling that pattern
// literally here (in code OR in this comment) could make a filter drop
// bindings.cpp itself from every build. Do NOT "simplify" this away.
#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _ops  // bare compiles outside torch's builders
#endif
#define SG_OPS_PYMODULE TORCH_EXTENSION_NAME

PYBIND11_MODULE(SG_OPS_PYMODULE, m) {
 m.doc() = "Grokking Optimizers — specialized per-arch C++/CUDA/HIP kernels";

 // Exported-ABI schema version (see GROK_ABI_SCHEMA above). Bump on ANY
 // exported-signature change; the Python-side compatibility assert lands in
 // dispatch.py (sibling-owned) and is inert until then.
 m.attr("__abi_schema__") = GROK_ABI_SCHEMA;

 m.def("detect_arch", &sg::detect_arch,
 "Returns 90 or 942 for the detected GPU (3-arch active set: "
 "sm_90, gfx942, tpu_v6e). TPU handled in Python.");

 // Fused (model, optimizer, arch) dispatch. The trailing scalar args carry the
 // FULL optimizer-state scalar set (C2-gap fix) so the fused tail's real math
 // runs; py::arg defaults reproduce the pre-fix inert behavior, so a short call
 // ops.fused_step(model, opt, params, input, grad, state, lr) still works.
 // bc1/bc2 are un-inverted (= 1 - beta^step). opt_only=true → L1 faithful tail.
 m.def("fused_step", &sg::fused_step,
 "Fused (model, optimizer, arch) kernel dispatch. Routes to the "
 "appropriate fused TU based on detected hardware. Carries the full "
 "optimizer scalar set (beta1/beta2/eps/weight_decay/alpha/lamb/gamma/"
 "gate/d_factor/bc1/bc2/neg_lr_scale/decay_factor/beta/alpha_max/step); "
 "opt_only is a dead ABI slot (the L1 tail was removed in task #10 — pure "
 "L3-TC or hard-fail); default False selects the only surviving L3-REAL path.",
 py::arg("model"), py::arg("optimizer"), py::arg("params"),
 py::arg("input"), py::arg("grad"), py::arg("state"), py::arg("lr"),
 py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
 py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.01f,
 py::arg("alpha") = 0.98f, py::arg("lamb") = 2.0f,
 py::arg("gamma") = 0.0f, py::arg("gate") = 1.0f,
 py::arg("d_factor") = 1.0f, py::arg("bc1") = 1.0f, py::arg("bc2") = 1.0f,
 py::arg("neg_lr_scale") = 0.0f, py::arg("decay_factor") = 1.0f,
 py::arg("beta") = 0.0f, py::arg("alpha_max") = 1.0f,
 py::arg("step") = 1, py::arg("opt_only") = false,
 // GEMM-engine selector. "wgmma" is the ONLY engine (task #10 removed scalar);
 // the top gate in dispatch.cpp hard-CHECKs gemm_impl=="wgmma". Default "wgmma"
 // makes a bare call shape valid; the race wrapper passes it explicitly anyway.
 py::arg("gemm_impl") = "wgmma",
 // GrokAdamW GLOBAL grad-norm clip threshold (decoder L3-TC, mechanism (ii)).
 // Trailing defaulted arg → back-compat preserved (a stale _ops without it
 // trips the caller's one-shot TypeError latch, loud degrade). ≤0 ⇒ no clip
 // (inert for every non-GrokAdamW cell); the decoder GrokAdamW cell passes the
 // optimizer's grad_clip (=1.0) so the kernel's P2.5 global-norm clip fires.
 py::arg("grad_clip") = 0.0f,
 // Prodigy estimator scalars (decoder L3-TC, STAGED global-d). Trailing defaulted
 // args → back-compat preserved (eager/inert for every non-Prodigy cell). The
 // decoder Prodigy cell passes d0/d_coef/beta3 so the kernel's P2.6 d-reduction
 // matches the eager multi-tensor estimator.
 py::arg("d0") = 1e-6f, py::arg("d_coef") = 1.0f, py::arg("beta3") = 0.0f,
 // Muon 1D-group AdamW hyperparameters (ViT L3-TC, STAGED NS). Trailing defaulted
 // args → back-compat preserved (eager Muon adamw_* defaults, inert for every
 // non-Muon cell). The vit Muon cell passes adamw_lr/adamw_betas so the kernel's
 // Muon P3 1D AdamW tail matches the eager non-2D group.
 py::arg("aux_lr") = 1e-3f, py::arg("aux_beta1") = 0.9f,
 py::arg("aux_beta2") = 0.98f,
 // LookSAM SAM 2nd-backward scalars (decoder/vit/mamba L3-TC, MODEL-COUPLED).
 // Trailing defaulted args → back-compat preserved (inert 0.0 for every non-LookSAM
 // cell; a stale _ops without them trips the caller's one-shot TypeError latch, loud
 // degrade). The LookSAM cell passes rho (perturbation radius) + looksam_sam (the
 // every-k SAM-step gate, 1.0 on SAM steps) so the kernel's P2.4 perturb→2nd
 // fwd+bwd→sam_dir=g_sam−g phase fires on the right cadence.
 py::arg("rho") = 0.0f, py::arg("looksam_sam") = 0.0f,
 // SuperGrok11/15 meta-net rescale (decoder/vit L3-TC). Trailing defaulted arg →
 // back-compat preserved (inert 0.0 for every non-SG cell; a stale _ops without it
 // trips the caller's one-shot TypeError latch, loud degrade). The SG cell passes the
 // SharpnessMetaNet rescale so the kernel's P2.45 meta-net mu precompute (mu =
 // rescale·phi(g, sharpness)) runs the live mechanism; the phi weights + sharpness
 // buffer ride the STATE buffer (cell-scattered), so only this scalar is in the ABI.
 py::arg("sg_rescale") = 0.0f,
 // SuperGrok11 cosine-gate temperature (decoder/vit/mamba L3-TC). Trailing
 // defaulted arg → back-compat (inert 1.0 for every non-SG11 cell). The SG11
 // cell passes the SharpnessMetaNet gate_temperature so the kernel's P2.45
 // finalizer computes gate = sigmoid(gate_temp · cos(grad, mu)) (the cosine
 // SIGNAL preserved; only the final squashing is the temperature-scaled sigmoid).
 py::arg("gate_temp") = 1.0f);

 // SuperGrok2 DEDICATED L3-TC entry (decoder/vit). The FULL CSA/HCA/PEER/GRU
 // meta-net as the optimizer phase: in-kernel SEGMENTED SORT (STAGE -1) + SAM 2nd
 // backward → sharpness + sg2_meta_stages. Needs the meta-net weight bundle + the
 // per-tensor scalar arrays this generic fused_step ABI cannot carry, so it is a
 // PARALLEL entry (fused_step + the 28 byte-identical cells are UNTOUCHED).
 m.def("sg2_fused_step", &sg::sg2_fused_step,
 "SuperGrok2 L3-TC fused train step (decoder/vit): in-kernel segmented sort + "
 "SAM 2nd backward + full CSA/HCA/PEER/GRU meta-net as the optimizer phase. "
 "Writes the reduced grad into `grad` and the loss into state[3*total].",
 py::arg("model"), py::arg("params"), py::arg("input"), py::arg("grad"),
 py::arg("state"),
 py::arg("input_proj_W"), py::arg("input_proj_b"),
 py::arg("csa_q_W"), py::arg("csa_k_W"), py::arg("csa_v_W"), py::arg("csa_out_W"),
 py::arg("csa_compress_w"), py::arg("csa_idx_DQ"), py::arg("csa_idx_K"),
 py::arg("hca_q_W"), py::arg("hca_k_W"), py::arg("hca_v_W"), py::arg("hca_out_W"),
 py::arg("gru_Wz"), py::arg("gru_bz"), py::arg("gru_Wr"), py::arg("gru_br"),
 py::arg("gru_Wh"), py::arg("gru_bh"),
 py::arg("peer_query_Ws"), py::arg("prod_keys_A"), py::arg("prod_keys_B"),
 py::arg("expert_W1"), py::arg("expert_b1"), py::arg("expert_W2"), py::arg("expert_b2"),
 py::arg("sc_alpha"), py::arg("sc_gru_decay"), py::arg("sc_lamb_eff"),
 py::arg("sc_beta1"), py::arg("sc_bc1"), py::arg("sc_bc2"),
 py::arg("rescale"), py::arg("beta2"), py::arg("lr"), py::arg("wd"), py::arg("eps"),
 py::arg("rho"), py::arg("sam_on"), py::arg("step"));

 // PURE L3-TC: every EAGER per-op m.def is removed — the per-optimizer
 // *_fused_step + per-tensor helpers (GrokAdamW / Lion / Grokfast / Prodigy /
 // NeuralGrok / LookSAM / Muon / SuperGrok11 / SuperGrok15), the MoE entries, and
 // the eager SuperGrok2 CSA/HCA step / batched / bilevel / prepare entries (+ the
 // old "mamba_peer" aliases). The race runs the fused L3-TC megakernels only
 // (fused_step / sg2_fused_step above). The SuperGrok2 PERSISTENT megakernel tail
 // (sg2_meta_optimizer_tail + sg2_ws_stride, defined in sg2_meta_tail.cu) is kept.
 m.def("sg2_meta_optimizer_tail", &sg::sg2_meta_optimizer_tail,
 "SuperGrok2 full meta-net as ONE persistent megakernel (launch-elimination "
 "of csa_hca_step_one). Consumes the per-tensor PACKED flat buffers + the "
 "pre-computed |grad|-ascending sort perms; runs CSA/HCA/GRU/PEER/apply "
 "in-kernel.",
 py::arg("params_packed"),
 py::arg("grads_packed"),
 py::arg("sharpness_packed"),
 py::arg("exp_avg_packed"),
 py::arg("exp_avg_sq_packed"),
 py::arg("mu_packed"),
 py::arg("slow_packed"),
 py::arg("gru_state_packed"),
 py::arg("perm_packed"),
 py::arg("unsort_packed"),
 py::arg("n_per_tensor"),
 py::arg("row_off"),
 py::arg("workspace"),
 py::arg("ws_stride"),
 py::arg("input_proj_W"), py::arg("input_proj_b"),
 py::arg("csa_q_W"), py::arg("csa_k_W"), py::arg("csa_v_W"), py::arg("csa_out_W"),
 py::arg("csa_compress_w"), py::arg("csa_idx_DQ"), py::arg("csa_idx_K"),
 py::arg("hca_q_W"), py::arg("hca_k_W"), py::arg("hca_v_W"), py::arg("hca_out_W"),
 py::arg("gru_Wz"), py::arg("gru_bz"), py::arg("gru_Wr"), py::arg("gru_br"),
 py::arg("gru_Wh"), py::arg("gru_bh"),
 py::arg("peer_query_Ws"), py::arg("prod_keys_A"), py::arg("prod_keys_B"),
 py::arg("expert_W1"), py::arg("expert_b1"), py::arg("expert_W2"), py::arg("expert_b2"),
 py::arg("alpha"), py::arg("gru_decay"), py::arg("lamb_eff"),
 py::arg("beta1"), py::arg("bc1"), py::arg("bc2"),
 py::arg("rescale"), py::arg("beta2"), py::arg("lr"), py::arg("wd"), py::arg("eps"),
 py::arg("g_next_task"), py::arg("g_arrived"), py::arg("g_generation"));
 m.def("sg2_ws_stride", &sg::sg2_ws_stride,
 "Authoritative floats-per-CTA workspace stride for the SG2 megakernel "
 "(== sg2_ws_stride<SG2Dims<>>(Nmax)); the Python driver calls this so the "
 "host allocation can never drift from the kernel's workspace carve.",
 py::arg("Nmax"));
 // PURE L3-TC: the eager SuperGrok2 bilevel (fwd_save / backward, single + batched)
 // and prepare_and_batched_step entries are removed along with the rest of the
 // eager CSA/HCA path; only the persistent megakernel tail above survives.

 // Model bindings (decoder, vit, mamba) — registered as _ops.models.*
 register_model_bindings(m);
}
