// =====================================================================
//  csrc/kernels/cuda/sm_90/supergrok2_warp_specialized.cuh
//
//  sm_90 (Hopper) SuperGrok v2 — warp-specialized scan kernels.
//
//  This header is NET-NEW: the previous baseline at
//  csrc/kernels/cuda/sm_90/supergrok2_warp_specialized_sm90.cu was
//  deleted in commit 5505b50. Recoverable for reference via:
//    git show 5505b50^:csrc/kernels/cuda/sm_90/supergrok2_warp_specialized_sm90.cu
//
//  Two scan kernels exploit Hopper's 4 warp schedulers per SM by
//  splitting the threadblock into a producer warp (issues data loads)
//  and consumer warps (run the SSM scan recurrence). The handoff
//  happens through double-buffered shared memory with atomic phase
//  flags. Memory latency is hidden behind compute, expected ~1.5x
//  speedup vs. the unfused parallel-scan kernel for uniform-d_state
//  param batches (see REFRESH §25.1 activation gate).
//
//  Kernels:
//    1. scan_warp_specialized_kernel       — generic d_state, FP32.
//                                              One block per (di, ds_pair)
//                                              tile; consumer warp 1 runs
//                                              the scan, producer warp 0
//                                              issues vec-loads, warps 2-3
//                                              are reserved for future use.
//    2. scan_warp_specialized_d16_kernel  — d_state=16 unrolled, FP32.
//                                              One block per d_inner index.
//                                              All 8 RoPE pairs unrolled
//                                              in registers in the
//                                              consumer warp.
//
//  Activation (REFRESH §25.1): the canonical SG2 scan launcher routes
//  here when `is_uniform_d_state(batch)` returns true and N >= the
//  parallel-scan threshold. The d16 variant takes precedence whenever
//  d_state == 16 (the common SG2 default).
//
//  NAMESPACE: kernels + launchers live in `sg::sm90::supergrok2`. The
//  binding declarations in `csrc/bindings/supergrok2.cpp` use namespace
//  `sg::sm90` (no `::supergrok2` suffix); a thin shim TU is required to
//  bridge this if the canonical SG2 dispatcher in
//  `supergrok2_fwd.{cuh,cu}` is to be invoked from the binding without
//  changes.  The dispatcher in supergrok2_fwd.cuh re-exports the
//  required `sg::sm90::launch_mamba3_peer_*` symbols.
// =====================================================================

#pragma once

#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

#if GROK_CUDA

#include <cuda_pipeline.h>

namespace sg { namespace sm90 { namespace supergrok2 {

// ---------------------------------------------------------------------
//  Warp-role assignment + double-buffer constants
// ---------------------------------------------------------------------

constexpr int WS_PRODUCER_WARP_ID = 0;
constexpr int WS_NUM_CONSUMER_WARPS = 3;
constexpr int WS_BLOCK_SIZE = 128;     // 4 warps × 32 threads
constexpr int WS_NUM_BUFFERS = 2;       // double-buffered staging

// ---------------------------------------------------------------------
//  Kernel 1: warp-specialized scan, generic d_state (FP32)
// ---------------------------------------------------------------------
__launch_bounds__(WS_BLOCK_SIZE, 4)
__global__ void scan_warp_specialized_kernel(
    const float* __restrict__ pre_x,     // [N, d_inner]
    const float* __restrict__ pre_z,     // [N, d_inner]
    const float* __restrict__ pre_dt,    // [N, d_inner]
    const float* __restrict__ pre_B,     // [N, d_state]
    const float* __restrict__ pre_C,     // [N, d_state]
    const float* __restrict__ A_log,     // [d_inner, d_state]
    const float* __restrict__ D_param,   // [d_inner]
    const float* __restrict__ rope_freq, // [d_state/2]
    float* __restrict__ scan_state,      // [d_inner, d_state] (in/out)
    float* __restrict__ scan_output,     // [N, d_inner]   (atomicAdd accum)
    int N, int d_inner, int d_state
);

// ---------------------------------------------------------------------
//  Kernel 2: warp-specialized scan, d_state=16 unrolled (FP32)
// ---------------------------------------------------------------------
__launch_bounds__(WS_BLOCK_SIZE, 4)
__global__ void scan_warp_specialized_d16_kernel(
    const float* __restrict__ pre_x,     // [N, d_inner]
    const float* __restrict__ pre_z,     // [N, d_inner]
    const float* __restrict__ pre_dt,    // [N, d_inner]
    const float* __restrict__ pre_B,     // [N, 16]
    const float* __restrict__ pre_C,     // [N, 16]
    const float* __restrict__ A_log,     // [d_inner, 16]
    const float* __restrict__ D_param,   // [d_inner]
    const float* __restrict__ rope_freq, // [8]
    float* __restrict__ scan_state,      // [d_inner, 16]
    float* __restrict__ scan_output,     // [N, d_inner]
    int N, int d_inner
);

// ---------------------------------------------------------------------
//  Host launchers (FP32 only; param batch must be FP32-state SG2)
// ---------------------------------------------------------------------
void launch_scan_warp_specialized(
    const float* pre_x, const float* pre_z, const float* pre_dt,
    const float* pre_B, const float* pre_C,
    const float* A_log, const float* D_param, const float* rope_freq,
    float* scan_state, float* scan_output,
    int N, int d_inner, int d_state,
    GpuStream_t stream
);

void launch_scan_warp_specialized_d16(
    const float* pre_x, const float* pre_z, const float* pre_dt,
    const float* pre_B, const float* pre_C,
    const float* A_log, const float* D_param, const float* rope_freq,
    float* scan_state, float* scan_output,
    int N, int d_inner,
    GpuStream_t stream
);

// ---------------------------------------------------------------------
//  Activation gate: returns true when every param in the batch shares
//  the same d_state. The canonical SG2 dispatcher in supergrok2_fwd.cuh
//  consults this before routing to the warp-specialized fast path.
//  REFRESH §25.1 expects this to fire on typical SG2 batches.
// ---------------------------------------------------------------------
__host__ __forceinline__ bool is_uniform_d_state(
    const std::vector<int>& d_state_per_param, int& d_state_out)
{
    if (d_state_per_param.empty()) {
        d_state_out = 0;
        return false;
    }
    const int ref = d_state_per_param[0];
    for (size_t i = 1; i < d_state_per_param.size(); ++i) {
        if (d_state_per_param[i] != ref) {
            d_state_out = 0;
            return false;
        }
    }
    d_state_out = ref;
    return true;
}

}}} // namespace sg::sm90::supergrok2

#endif // GROK_CUDA
