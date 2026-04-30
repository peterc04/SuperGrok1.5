// bindings/muon.cpp — runtime dispatch to per-arch Muon launchers.
//
// Per-arch namespaces define four launchers (signatures must match the
// definitions in csrc/kernels/{cuda/<sm>,hip/gfx{942,950}}/muon_<arch>.cu):
//
//   launch_muon_momentum_normalize(buf, X, grad, momentum, inv_norm)
//   launch_muon_ns_combine(X_out, X, AX, AAX, a, b, c)
//   launch_muon_update(param, orth, neg_lr_scale, decay_factor)
//   launch_muon_ns_combine_update_fused(
//       param, X, AX, AAX, a, b, c, neg_lr_scale, decay_factor)
//
// The vector-API muon_fused_step in this file orchestrates the per-tensor
// pipeline (norm reduction, NS iterations, fused final step) and is the
// only Python-callable Muon entry. The Newton-Schulz coefficients
// (3.4445, -4.7750, 2.0315) come from the pre-refactor ops.cpp constants
// and are passed as a, b, c.

#include "_dispatch_macro.h"
#include "_helpers.h"

#include <cmath>
#include <vector>

namespace sg {

#define DECLARE_MUON(NS)                                                      \
    namespace NS {                                                            \
        void launch_muon_momentum_normalize(                                  \
            torch::Tensor buf, torch::Tensor X, torch::Tensor grad,           \
            float momentum, float inv_norm);                                  \
        void launch_muon_ns_combine(                                          \
            torch::Tensor X_out, torch::Tensor X,                             \
            torch::Tensor AX, torch::Tensor AAX,                              \
            float a, float b, float c);                                       \
        void launch_muon_update(                                              \
            torch::Tensor param, torch::Tensor orth,                          \
            float neg_lr_scale, float decay_factor);                          \
        void launch_muon_ns_combine_update_fused(                             \
            torch::Tensor param, torch::Tensor X,                             \
            torch::Tensor AX, torch::Tensor AAX,                              \
            float a, float b, float c,                                        \
            float neg_lr_scale, float decay_factor);                          \
    }

DECLARE_MUON(sm80) DECLARE_MUON(sm89) DECLARE_MUON(sm90)
DECLARE_MUON(sm100) DECLARE_MUON(sm103) DECLARE_MUON(sm120)
DECLARE_MUON(gfx942) DECLARE_MUON(gfx950)
#undef DECLARE_MUON

// ---------------------------------------------------------------------
// Per-tensor wrappers (internal helpers).
// ---------------------------------------------------------------------

void muon_momentum_normalize(
    torch::Tensor buf, torch::Tensor X, torch::Tensor grad,
    float momentum, float inv_norm)
{
    SG_DISPATCH(launch_muon_momentum_normalize,
        buf, X, grad, momentum, inv_norm);
}

void muon_ns_combine(
    torch::Tensor X_out, torch::Tensor X,
    torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c)
{
    SG_DISPATCH(launch_muon_ns_combine, X_out, X, AX, AAX, a, b, c);
}

void muon_ns_combine_update_fused(
    torch::Tensor param, torch::Tensor X,
    torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c,
    float neg_lr_scale, float decay_factor)
{
    SG_DISPATCH(launch_muon_ns_combine_update_fused,
        param, X, AX, AAX, a, b, c, neg_lr_scale, decay_factor);
}

// ---------------------------------------------------------------------
// High-level vector-signature entry point — pre-refactor
// csrc/common/ops.cpp::muon_fused_step. The math is unchanged.
// Replaces N CPU syncs with one (torch::stack of bufs[i].norm()).
// ---------------------------------------------------------------------

void muon_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& bufs,
    float momentum, float lr, float wd, int ns_steps
) {
    constexpr float NS_A = 3.4445f;
    constexpr float NS_B = -4.7750f;
    constexpr float NS_C = 2.0315f;

    std::vector<torch::Tensor> norm_tensors;
    std::vector<size_t> valid_indices;
    norm_tensors.reserve(params.size());
    valid_indices.reserve(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        bufs[i].mul_(momentum).add_(grads[i]);
        norm_tensors.push_back(bufs[i].norm());
        valid_indices.push_back(i);
    }
    if (norm_tensors.empty()) return;

    auto norms_stacked = torch::stack(norm_tensors).cpu();
    auto* norms_ptr = norms_stacked.data_ptr<float>();

    for (size_t vi = 0; vi < valid_indices.size(); vi++) {
        size_t i = valid_indices[vi];
        auto& p = params[i];
        auto& buf = bufs[i];

        float buf_norm = norms_ptr[vi] + 1e-7f;
        float inv_norm = 1.0f / buf_norm;
        auto X = buf * inv_norm;

        int64_t rows = p.size(0);
        int64_t cols = p.size(1);
        float max_dim = static_cast<float>(std::max(rows, cols));
        float scale_factor = 0.2f * std::sqrt(max_dim);
        float neg_lr_scale = -lr * scale_factor;
        float decay_factor = 1.0f - lr * wd;

        for (int step = 0; step < ns_steps; step++) {
            auto A = torch::mm(X, X.t());
            auto AX = torch::mm(A, X);
            auto AAX = torch::mm(A, AX);
            if (step < ns_steps - 1) {
                auto X_new = torch::empty_like(X);
                SG_DISPATCH_CALL(launch_muon_ns_combine,
                    X_new, X, AX, AAX, NS_A, NS_B, NS_C);
                X = X_new;
            } else {
                SG_DISPATCH_CALL(launch_muon_ns_combine_update_fused,
                    p, X, AX, AAX, NS_A, NS_B, NS_C,
                    neg_lr_scale, decay_factor);
            }
        }
        if (ns_steps == 0) {
            auto X_typed = X.to(p.dtype());
            SG_DISPATCH_CALL(launch_muon_update,
                p, X_typed, neg_lr_scale, decay_factor);
        }
    }
}

} // namespace sg
