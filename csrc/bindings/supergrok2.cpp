// bindings/supergrok2.cpp — runtime dispatch to per-arch SG v2 launchers.
//
// SG v2 is the largest dispatcher. It covers:
// - mamba3_peer forward (per-element step + scan)
// - bilevel forward-save (state checkpointing)
// - bilevel backward (gradients of meta-net weights)
// - The batched + prepare entry point that ops.cpp called as
//   supergrok2_prepare_and_batched_step
//
// The CUTLASS migration for SG v2 projection GEMMs (sm_90 / sm_100) is
// inside the per-arch launcher (sg::sm90::launch_mamba3_peer_step etc.)
// and not visible at this layer.

#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_SG2(NS)                                                       \
    namespace NS {                                                            \
        void launch_mamba3_peer_step(                                         \
            std::vector<torch::Tensor> param_list,                            \
            std::vector<torch::Tensor> grad_list,                             \
            std::vector<torch::Tensor> exp_avg_list,                          \
            std::vector<torch::Tensor> exp_avg_sq_list,                       \
            std::vector<torch::Tensor> mu_list,                               \
            std::vector<torch::Tensor> sharpness_list,                        \
            std::vector<torch::Tensor> gru_state_list,                        \
            torch::Tensor mamba_fwd_states, torch::Tensor mamba_bwd_states,   \
            /* meta-net weights */                                            \
            torch::Tensor in_proj_W, torch::Tensor dt_proj_W,                 \
            torch::Tensor B_proj_W, torch::Tensor C_proj_W,                   \
            torch::Tensor A_log, torch::Tensor D_param,                       \
            torch::Tensor rope_freq,                                          \
            torch::Tensor gru_Wz, torch::Tensor gru_bz,                       \
            torch::Tensor gru_Wr, torch::Tensor gru_br,                       \
            torch::Tensor gru_Wh, torch::Tensor gru_bh,                       \
            torch::Tensor peer_query_Ws,                                      \
            torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,             \
            torch::Tensor expert_W1, torch::Tensor expert_b1,                 \
            torch::Tensor expert_W2, torch::Tensor expert_b2,                 \
            torch::Tensor expert_counts,                                      \
            /* scalars */                                                     \
            float rescale, float alpha, float lamb_eff,                       \
            float beta1, float beta2, float lr, float wd_eff,                 \
            float eps, float bc1, float bc2);                                 \
        void launch_mamba3_peer_bilevel_fwd_save_batched(                     \
            /* see supergrok2_mamba_peer_backward_kernels.cu for sig */);     \
        void launch_mamba3_peer_backward_batched(                             \
            /* see supergrok2_mamba_peer_backward_kernels.cu for sig */);     \
    }

DECLARE_SG2(sm80) DECLARE_SG2(sm90) DECLARE_SG2(sm100) DECLARE_SG2(gfx942)
DECLARE_SG2(sm89) DECLARE_SG2(sm103) DECLARE_SG2(sm120) DECLARE_SG2(gfx950)
#undef DECLARE_SG2

// TODO(structural-refactor): supergrok2 launcher signatures are large and
// not all argument tuples are inferable from the existing kernels without
// reading each launcher in detail. The dispatcher entry points below are
// stubs; flesh out each signature when wiring the bindings layer to
// pybind11. For now they are declared but not instantiated -- ops.cpp
// continues to be the entry point until the migration completes.

} // namespace sg
