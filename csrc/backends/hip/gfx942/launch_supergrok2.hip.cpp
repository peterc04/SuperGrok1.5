// HIP gfx942 launch glue for SuperGrok v2.
// Algorithm: csrc/algorithms/supergrok2.h
//
// HONEST STATUS: SuperGrok v2 is NOT implemented on gfx942. The full
// Mamba+GRU+PEER pipeline relies on Hopper-specific features (cluster
// reductions via DSMEM, WGMMA, warp-specialized scan with 4 schedulers)
// that have no direct CDNA3 equivalent. A complete port would require
// ~weeks of work using MFMA + LDS-resident scan + manual producer/consumer
// synchronization.
//
// This launcher raises std::runtime_error with a descriptive message.
// Non-SG2 workflows on MI300X are fully functional via the other 10
// launch files. Python frontends (grokking_optimizers/supergrok2.py)
// fall back to the pure-Python reference implementation when this
// runtime error is raised.

#include <torch/extension.h>
#include <stdexcept>
#include <string>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace hip_gfx942 {

[[noreturn]] static void sg2_not_implemented(const char* op) {
    throw std::runtime_error(
        std::string("SuperGrok v2 is not implemented on gfx942 (CDNA3). ") +
        "Requested op: " + op + ". " +
        "The Mamba + PEER + GRU pipeline requires Hopper-specific features " +
        "(DSMEM cluster reductions, WGMMA, 4-warp-scheduler specialization) " +
        "that have no direct CDNA3 equivalent. Use sm_90 for SG2 workflows, " +
        "or run with --no-fused to use the Python reference fallback.");
}

void launch_supergrok2_input_proj_sort(...) {
    sg2_not_implemented("input_proj_sort");
}

void launch_supergrok2_apply(...) {
    sg2_not_implemented("apply");
}

void launch_supergrok2_step(...) {
    sg2_not_implemented("step");
}

}} // namespace sg::hip_gfx942
