/*
 * CPU MoE Parameter Compaction
 *
 * For MoE models on CPU, stream-compact active-gradient elements
 * so that only parameters from routed experts are processed by
 * the scan, reducing scan length by ~30x for top-2/64 routing.
 */

#include <torch/extension.h>

void cpu_moe_filter_active_params(
    torch::Tensor all_grads,
    torch::Tensor active_mask,
    torch::Tensor compact_grads,
    torch::Tensor compact_indices,
    torch::Tensor N_active_out
) {
    int N = all_grads.numel();
    auto g = all_grads.data_ptr<float>();
    auto mask = active_mask.data_ptr<bool>();
    auto cg = compact_grads.data_ptr<float>();
    auto ci = compact_indices.data_ptr<int>();

    int count = 0;
    for (int i = 0; i < N; i++) {
        if (mask[i]) {
            cg[count] = g[i];
            ci[count] = i;
            count++;
        }
    }
    N_active_out.data_ptr<int>()[0] = count;
}
