// CUDA sm_90 distributed scan stubs.
// Resolves symbols declared in csrc/bindings/distributed_scan.cpp.

#include <torch/extension.h>
#include <stdexcept>

namespace sg { namespace sm90 {

void distributed_scan_local_with_summary(
    torch::Tensor x_sorted, torch::Tensor scan_out,
    torch::Tensor summary_out,
    torch::Tensor in_proj_W, torch::Tensor dt_proj_W,
    torch::Tensor B_proj_W, torch::Tensor C_proj_W,
    torch::Tensor A_log, torch::Tensor D_param,
    torch::Tensor rope_freq
) {
    throw std::runtime_error(
        "distributed_scan_local_with_summary: sm_90 kernel not yet implemented.");
}

void distributed_scan_summary_prefix(
    torch::Tensor summaries, torch::Tensor prefixes
) {
    throw std::runtime_error(
        "distributed_scan_summary_prefix: sm_90 kernel not yet implemented.");
}

void distributed_scan_apply_prefix(
    torch::Tensor scan_out, torch::Tensor prefix
) {
    throw std::runtime_error(
        "distributed_scan_apply_prefix: sm_90 kernel not yet implemented.");
}

}} // namespace sg::sm90
