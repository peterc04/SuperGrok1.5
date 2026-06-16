// =====================================================================
// helpers.h — shared bindings helpers + per-arch dispatch macro.
//
// Includes:
//   - int sg::detect_arch() forward decl (implemented in dispatch.cpp)
//   - SG_DISPATCH / SG_DISPATCH_CALL macros for runtime arch selection
//   - host-side gradient norm helpers extracted from the deleted
//     csrc/common/ops.cpp.
// =====================================================================

#pragma once

#include <torch/extension.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sg {

// ── Arch detection (impl in dispatch.cpp) ────────────────────────────
int detect_arch();
inline bool is_cuda_arch(int a) { return a == 90; }
inline bool is_hip_arch(int a)  { return a == 942; }

// ── Fused (model, optimizer, arch) megakernel dispatch (impl in
//    dispatch.cpp). Declared here so bindings.cpp can bind &sg::fused_step.
//
// The trailing scalar args carry the FULL optimizer-state scalar set so the
// fused tail's apply_optimizer<> runs its real math (C2-gap fix). Defaults
// reproduce the pre-fix inert behavior (bc1/bc2/gate/d_factor == 1.0), so a
// caller that passes only the first 7 args is unchanged. bc1/bc2 are un-inverted
// (= 1 - beta^step). `opt_only` selects L1 (faithful real-grad optimizer tail,
// the default + the race path) vs L3 (surrogate-model fwd+bwd+opt). Keep this
// declaration's defaults in sync with the definition in dispatch.cpp.
void fused_step(const std::string& model, const std::string& optimizer,
                torch::Tensor params, torch::Tensor input,
                torch::Tensor grad, torch::Tensor state, float lr,
                float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
                float weight_decay = 0.01f, float alpha = 0.98f,
                float lamb = 2.0f, float gamma = 0.0f, float gate = 1.0f,
                float d_factor = 1.0f, float bc1 = 1.0f, float bc2 = 1.0f,
                float neg_lr_scale = 0.0f, float decay_factor = 1.0f,
                float beta = 0.0f, float alpha_max = 1.0f,
                // opt_only is a dead ABI slot: the L1 real-grad optimizer TAIL was
                // removed (task #10 — pure L3-TC or hard-fail). Default false ⇒ the
                // L3-REAL (!opt_only) path, the only one that survives in dispatch.cpp.
                // Kept in the signature for pybind ABI stability (stale-ABI latch).
                int64_t step = 1, bool opt_only = false,
                // GEMM-engine selector: "wgmma" is the ONLY supported engine (task
                // #10 removed the scalar fp32 megakernel). The top gate in dispatch.cpp
                // hard-CHECKs gemm_impl=="wgmma"; the default makes a bare call shape
                // valid. Keep in sync with the definition in dispatch.cpp (which, per
                // the C++ rule, omits the default — it lives only on this declaration).
                const std::string& gemm_impl = "wgmma",
                // GrokAdamW GLOBAL grad-norm clip threshold (decoder L3-TC,
                // mechanism (ii)). Trailing defaulted arg (≤0 ⇒ no clip = inert for
                // every non-GrokAdamW cell). Keep in sync with the definition in
                // dispatch.cpp + the pybind py::arg list (bindings.cpp).
                float grad_clip = 0.0f,
                // Prodigy estimator scalars (decoder L3-TC, STAGED global-d).
                // Trailing defaulted args (eager/inert for every non-Prodigy cell:
                // d_coef=1, beta3=0). Keep in sync with the dispatch.cpp definition
                // + the pybind py::arg list (bindings.cpp).
                float d0 = 1e-6f, float d_coef = 1.0f, float beta3 = 0.0f,
                // Muon 1D-group AdamW hyperparameters (ViT L3-TC, STAGED NS). The
                // eager Muon's non-2D AdamW group has INDEPENDENT lr/betas
                // (adamw_lr/adamw_betas, muon.py:115-125); these carry them to the
                // kernel's Muon P3 1D tail. Trailing defaulted args = eager Muon
                // adamw_* defaults (inert for every non-Muon cell). Keep in sync
                // with the dispatch.cpp definition + the pybind py::arg list.
                float aux_lr = 1e-3f, float aux_beta1 = 0.9f,
                float aux_beta2 = 0.98f,
                // LookSAM SAM 2nd-backward scalars (decoder/vit/mamba L3-TC,
                // MODEL-COUPLED). rho = SAM perturbation radius; looksam_sam = the
                // every-k SAM-step gate (1.0 ⇒ run the in-kernel perturb→2nd
                // fwd+bwd→sam_dir=g_sam−g phase). Trailing defaulted args = inert
                // (0.0) for every non-LookSAM cell. Keep in sync with the
                // dispatch.cpp definition + the pybind py::arg list (bindings.cpp).
                float rho = 0.0f, float looksam_sam = 0.0f,
                // SuperGrok11/15 meta-net rescale (decoder/vit L3-TC). mu =
                // rescale·phi(g, sharpness). The phi weights + sharpness buffer ride
                // the STATE buffer (cell-scattered), so only this scalar is in the ABI.
                // Trailing defaulted arg = inert (0.0) for every non-SG cell. Keep in
                // sync with the dispatch.cpp definition + the pybind py::arg list.
                float sg_rescale = 0.0f,
                // SuperGrok11 cosine-gate temperature (decoder/vit/mamba L3-TC). The
                // P2.45 finalizer computes gate = sigmoid(gate_temp · cos(grad, mu))
                // (sg11_finalize_gate). Trailing defaulted arg = inert (1.0) for every
                // non-SG11 cell. Keep in sync with the dispatch.cpp definition + the
                // pybind py::arg list.
                float gate_temp = 1.0f);

// ── SuperGrok2 DEDICATED L3-TC entry (decoder/vit). SG2's optimizer phase is the
//    FULL CSA/HCA/PEER/GRU meta-net (in-kernel segmented sort + SAM 2nd backward →
//    sharpness + sg2_meta_stages), needing the meta-net weight bundle (26 tensors) +
//    per-tensor scalar ARRAYS (6, length P), none representable in fused_step's
//    FusedScalars POD. A PARALLEL entry; fused_step + the 28 cells are UNTOUCHED.
//    Definition in dispatch.cpp; pybind registration in bindings.cpp. ──
void sg2_fused_step(
    const std::string& model,
    torch::Tensor params, torch::Tensor input, torch::Tensor grad, torch::Tensor state,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, torch::Tensor csa_out_W,
    torch::Tensor csa_compress_w, torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_K,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, torch::Tensor hca_out_W,
    torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1, torch::Tensor expert_W2, torch::Tensor expert_b2,
    torch::Tensor sc_alpha, torch::Tensor sc_gru_decay, torch::Tensor sc_lamb_eff,
    torch::Tensor sc_beta1, torch::Tensor sc_bc1, torch::Tensor sc_bc2,
    double rescale, double beta2, double lr, double wd, double eps,
    double rho, double sam_on, int64_t step);

// ── Per-arch namespace handles ───────────────────────────────────────
namespace sm90 {}
namespace gfx942 {}

} // namespace sg

// ── Backend-gated dispatch case fragments ────────────────────────────
// A CUDA build links ONLY sg::sm90 symbols; a HIP build ONLY sg::gfx942. A
// `case` that calls the other arch's namespace is an undefined symbol that
// makes the .so fail to load (e.g. sg::gfx942::moe_filter_active_params in a
// CUDA-only build — the gfx942 .hip TUs are never compiled by nvcc). Gate each
// case by its backend macro, mirroring dispatch.cpp's WITH_CUDA/WITH_HIP split,
// so the Python front-end binds exactly the backend that was compiled in.
// (WITH_HIP wins if both are somehow defined, matching dispatch.cpp.)
#if defined(WITH_CUDA) && !defined(WITH_HIP)
#  define SG_CASE_SM90_RET(METHOD, ...)  case 90:  return ::sg::sm90::METHOD(__VA_ARGS__);
#  define SG_CASE_SM90_CALL(METHOD, ...) case 90:  ::sg::sm90::METHOD(__VA_ARGS__); break;
#else
#  define SG_CASE_SM90_RET(METHOD, ...)
#  define SG_CASE_SM90_CALL(METHOD, ...)
#endif
#if defined(WITH_HIP)
#  define SG_CASE_GFX942_RET(METHOD, ...)  case 942: return ::sg::gfx942::METHOD(__VA_ARGS__);
#  define SG_CASE_GFX942_CALL(METHOD, ...) case 942: ::sg::gfx942::METHOD(__VA_ARGS__); break;
#else
#  define SG_CASE_GFX942_RET(METHOD, ...)
#  define SG_CASE_GFX942_CALL(METHOD, ...)
#endif

// ── Dispatch macro: returns from enclosing function ──────────────────
#define SG_DISPATCH(METHOD, ...) \
    do { \
        const int sg_arch_ = ::sg::detect_arch(); \
        switch (sg_arch_) { \
            SG_CASE_SM90_RET(METHOD, __VA_ARGS__) \
            SG_CASE_GFX942_RET(METHOD, __VA_ARGS__) \
            default: \
                throw std::runtime_error( \
                    std::string(#METHOD) + " dispatch: unsupported arch " + \
                    std::to_string(sg_arch_)); \
        } \
    } while (0)

// ── Dispatch macro: same dispatch, no return ─────────────────────────
#define SG_DISPATCH_CALL(METHOD, ...) \
    do { \
        const int sg_arch_ = ::sg::detect_arch(); \
        switch (sg_arch_) { \
            SG_CASE_SM90_CALL(METHOD, __VA_ARGS__) \
            SG_CASE_GFX942_CALL(METHOD, __VA_ARGS__) \
            default: \
                throw std::runtime_error( \
                    std::string(#METHOD) + " dispatch: unsupported arch " + \
                    std::to_string(sg_arch_)); \
        } \
    } while (0)

namespace sg {

// ── Boundary validation for optimizer fused-step entrypoints ─────────
// The Python wrappers historically owned device/dtype/shape/contiguity
// invariants; these binding entrypoints trusted them. A non-contiguous param
// view, a device/dtype mismatch between param and grad, or a shape mismatch
// silently corrupts the in-place fused update (the launchers index a flat
// pointer with the param's numel). Validate at the C++ boundary so a bad call
// fails loudly. `where` is the entrypoint name for the error message.
//
// We check the param against its paired grad. Optimizer-state buffers (m, v,
// ema, …) are allocated by the wrapper from `torch.zeros_like(param)` so they
// inherit the param's device/dtype/shape/contiguity; the param check is the
// load-bearing one. A missing/empty grad (sparse-grad fallback) is skipped —
// the per-op loops already filter those out before dispatch.
inline void check_param_grad(
    const torch::Tensor& p,
    const torch::Tensor& g,
    const char* where
) {
    TORCH_CHECK(p.defined(), where, ": param tensor is undefined");
#if defined(WITH_HIP)
    TORCH_CHECK(p.is_hip(), where, ": param must be on a HIP device");
#else
    TORCH_CHECK(p.is_cuda(), where, ": param must be on a CUDA device");
#endif
    // A non-contiguous param view aliases a strided storage; the fused
    // launchers treat data_ptr() as a dense [numel] buffer, so a strided view
    // would read/write the wrong elements and silently corrupt the update.
    TORCH_CHECK(p.is_contiguous(), where,
                ": param must be contiguous (got a non-contiguous view); "
                "call .contiguous() in the Python wrapper before the fused step");
    if (!g.defined() || g.numel() == 0) return;  // sparse-grad: skip (filtered)
    TORCH_CHECK(g.device() == p.device(), where,
                ": grad device (", g.device().str(),
                ") != param device (", p.device().str(), ")");
    TORCH_CHECK(g.scalar_type() == p.scalar_type(), where,
                ": grad dtype (", toString(g.scalar_type()),
                ") != param dtype (", toString(p.scalar_type()), ")");
    TORCH_CHECK(g.sizes() == p.sizes(), where,
                ": grad shape (", g.sizes(), ") != param shape (",
                p.sizes(), ")");
    TORCH_CHECK(g.is_contiguous(), where, ": grad must be contiguous");
}

// Validate every (param, grad) pair in a multi-tensor fused-step entrypoint.
inline void check_params_grads(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& grads,
    const char* where
) {
    TORCH_CHECK(params.size() == grads.size(), where,
                ": params.size() (", params.size(),
                ") != grads.size() (", grads.size(), ")");
    for (size_t i = 0; i < params.size(); ++i)
        check_param_grad(params[i], grads[i], where);
}

// ── Secondary-list length guard for multi-tensor entrypoints ─────────
// check_params_grads validates the PRIMARY (params, grads) pairing. A
// multi-tensor fused step also receives several SECONDARY parallel lists
// (exp_avgs, exp_avg_sqs, emas, mus, slows, gru_states, sharpness caches, per-
// param step counters, per-layer scalar vectors, …). The launchers index those
// lists by the same i used for params[i], so a short/long secondary list either
// reads out of bounds or silently pairs the wrong buffer with a param — the same
// class of silent corruption check_param_grad guards against, but on the state
// side. Validate each secondary list's length == the expected element count.
//
// `what` names the offending list, `fn` the entrypoint, both in the message.
// A templated overload covers the scalar vectors (std::vector<float/int64_t/
// double>) so the SuperGrok per-layer alpha/beta1/step vectors are guarded too.
inline void check_list_len(
    const std::vector<torch::Tensor>& v,
    size_t expect,
    const char* what,
    const char* fn
) {
    TORCH_CHECK(v.size() == expect, fn, ": ", what, " list length (", v.size(),
                ") != expected (", expect, ") — every secondary list must be "
                "the same length as params/grads; the launcher indexes them in "
                "lockstep.");
}

template <typename Scalar>
inline void check_list_len(
    const std::vector<Scalar>& v,
    size_t expect,
    const char* what,
    const char* fn
) {
    TORCH_CHECK(v.size() == expect, fn, ": ", what, " list length (", v.size(),
                ") != expected (", expect, ") — every secondary list must be "
                "the same length as params/grads; the launcher indexes them in "
                "lockstep.");
}

// ── Device-side gradient clipping: single CPU sync instead of N ──────
inline void clip_grad_norms_device_side(
    std::vector<torch::Tensor>& grads,
    size_t n_params,
    float grad_clip_norm
) {
    if (grad_clip_norm <= 0.0f) return;

    // Collect the present grads once. Fused multi-tensor ops want a flat list.
    std::vector<torch::Tensor> present;
    present.reserve(n_params);
    for (size_t i = 0; i < n_params; i++) {
        if (grads[i].defined() && grads[i].numel() > 0)
            present.push_back(grads[i]);
    }
    if (present.empty()) return;

    // Per-tensor L2 norms via a single fused multi-tensor reduction
    // (torch::_foreach_norm), replacing the per-tensor upcast + dot + add_
    // (≈2N kernel launches) with one foreach launch. Numerics: the L2 norm of
    // each tensor is accumulated into a fp32 sum-of-squares (squaring the
    // per-tensor norm == that tensor's sum-of-squares), matching the original
    // fp32 accumulation of the global sum-of-squares. To preserve the original
    // precision when grads are NOT already fp32, upcast those (only) to fp32
    // before the norm; fp32 grads skip the upcast entirely.
    bool all_fp32 = true;
    for (auto& g : present)
        if (g.scalar_type() != torch::kFloat32) { all_fp32 = false; break; }

    std::vector<torch::Tensor> norm_inputs;
    if (all_fp32) {
        norm_inputs = present;
    } else {
        norm_inputs.reserve(present.size());
        for (auto& g : present)
            norm_inputs.push_back(g.scalar_type() == torch::kFloat32
                                      ? g
                                      : g.to(torch::kFloat32));
    }

    auto norms = torch::_foreach_norm(norm_inputs, /*p=*/2);
    // norm_sq = sum_i (||g_i||_2)^2, accumulated in fp32 on-device, one sync.
    auto stacked = torch::stack(norms).to(torch::kFloat32);
    float total_norm =
        std::sqrt(stacked.dot(stacked).item<float>());

    if (total_norm > grad_clip_norm) {
        float clip_coef = grad_clip_norm / (total_norm + 1e-6f);
        // Fused multi-tensor in-place scale of the present grads.
        torch::_foreach_mul_(present, clip_coef);
    }
}

// ── Device-side SAM grad-norm: single CPU sync instead of N ──────────
inline float compute_sam_grad_norm_device_side(
    std::vector<torch::Tensor>& grads,
    size_t n_grads
) {
    torch::Device dev(torch::kCPU);
    for (size_t i = 0; i < n_grads; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            dev = grads[i].device();
            break;
        }
    }
    auto norm_sq = torch::zeros(
        {1}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    for (size_t i = 0; i < n_grads; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            auto g_flat = grads[i].to(torch::kFloat32).reshape(-1);
            norm_sq.add_(g_flat.dot(g_flat));
        }
    }
    return std::sqrt(norm_sq.item<float>()) + 1e-12f;
}

} // namespace sg
