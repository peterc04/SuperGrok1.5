// bindings/moe.cpp — runtime dispatch to per-arch MoE launchers.
//
// MoE layer covers dynamic expert load/forward/backward, active-param
// filtering, compacted scan, scatter, activation counting, load-balance
// loss. See csrc/cuda/generic/moe_deep_kernels.cu for the full surface.
//
// TODO(structural-refactor): the MoE launcher set is large and the
// signatures depend on tensor layouts that are easier to extract from
// the .cu file when a proper pass is done. The stubs below cover the
// primary forward path; secondary launchers (backward, count, lb-loss)
// follow the same pattern.

#include "_dispatch_macro.h"

#include <vector>

namespace sg {

#define DECLARE_MOE(NS)                                                       \
    namespace NS {                                                            \
        void moe_dynamic_expert_load(                                         \
            torch::Tensor expert_w1, torch::Tensor expert_b1,                 \
            torch::Tensor expert_w2, torch::Tensor expert_b2,                 \
            torch::Tensor active_mask,                                        \
            torch::Tensor smem_w1, torch::Tensor smem_b1,                     \
            torch::Tensor smem_w2, torch::Tensor smem_b2);                    \
        void moe_dynamic_expert_fwd(                                          \
            torch::Tensor input, torch::Tensor expert_indices,                \
            torch::Tensor routing_weights,                                    \
            torch::Tensor expert_w1, torch::Tensor expert_b1,                 \
            torch::Tensor expert_w2, torch::Tensor expert_b2,                 \
            torch::Tensor output);                                            \
    }

DECLARE_MOE(sm80) DECLARE_MOE(sm90) DECLARE_MOE(sm100) DECLARE_MOE(gfx942)
#undef DECLARE_MOE

void moe_expert_fwd(
    torch::Tensor input, torch::Tensor expert_indices,
    torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor output)
{
    SG_DISPATCH(moe_dynamic_expert_fwd,
        input, expert_indices, routing_weights,
        expert_w1, expert_b1, expert_w2, expert_b2, output);
}

} // namespace sg
