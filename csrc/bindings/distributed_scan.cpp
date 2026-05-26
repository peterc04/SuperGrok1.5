// bindings/distributed_scan.cpp — dispatch for the multi-GPU Mamba-3 scan.
//
// Three-phase distributed scan:
// 1. local scan + summary on each GPU
// 2. summary prefix scan on rank 0 (or all-reduce)
// 3. apply per-GPU prefix correction to local output
// Backward variants follow the same shape.

#include "helpers.h"

namespace sg {

#define DECLARE_DSCAN(NS) \
 namespace NS { \
 void distributed_scan_local_with_summary( \
 torch::Tensor x_sorted, torch::Tensor scan_out, \
 torch::Tensor summary_out, \
 /* meta-net params */ \
 torch::Tensor in_proj_W, torch::Tensor dt_proj_W, \
 torch::Tensor B_proj_W, torch::Tensor C_proj_W, \
 torch::Tensor A_log, torch::Tensor D_param, \
 torch::Tensor rope_freq); \
 void distributed_scan_summary_prefix( \
 torch::Tensor summaries, torch::Tensor prefixes); \
 void distributed_scan_apply_prefix( \
 torch::Tensor scan_out, torch::Tensor prefix); \
 }

 DECLARE_DSCAN(sm90)
 DECLARE_DSCAN(gfx942)

#undef DECLARE_DSCAN

void distributed_scan_phase1(
 torch::Tensor x_sorted, torch::Tensor scan_out, torch::Tensor summary_out,
 torch::Tensor in_proj_W, torch::Tensor dt_proj_W,
 torch::Tensor B_proj_W, torch::Tensor C_proj_W,
 torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq)
{
 SG_DISPATCH(distributed_scan_local_with_summary,
 x_sorted, scan_out, summary_out,
 in_proj_W, dt_proj_W, B_proj_W, C_proj_W,
 A_log, D_param, rope_freq);
}

void distributed_scan_phase2(torch::Tensor summaries, torch::Tensor prefixes) {
 SG_DISPATCH(distributed_scan_summary_prefix, summaries, prefixes);
}

void distributed_scan_phase3(torch::Tensor scan_out, torch::Tensor prefix) {
 SG_DISPATCH(distributed_scan_apply_prefix, scan_out, prefix);
}

} // namespace sg
