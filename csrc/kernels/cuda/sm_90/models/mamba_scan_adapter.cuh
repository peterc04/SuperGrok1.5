// csrc/kernels/cuda/sm_90/models/mamba_scan_adapter.cuh
// Thin adapter translating model-context arguments (B x N x d_inner activations,
// B/C/dt projections, A_log) into the optimizer-context calling convention that
// SG2's mamba3_* scan kernels already accept.
//
// NO reimplementation of the scan itself — this file delegates to:
//   sg::device::sm90::mamba3_scan_step              (sequential)
//   sg::device::sm90::scan_warp_specialized_*       (parallel)
// declared in csrc/device/optimizers/sm_90/supergrok2_sm90.cuh.
//
// Decision tree (from csrc/common/types.h thresholds):
//   N < PSCAN_THRESHOLD (256)      -> sequential scan
//   256 <= N < GEMM_PRECOMPUTE_THRESHOLD (1024)  -> parallel precompute + parallel scan
//   N >= 1024                       -> bilevel_precompute_gemm + parallel scan

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"

// Device-level scan primitives from the SG2 optimizer kernels
#include "csrc/device/optimizers/sm_90/supergrok2_sm90.cuh"

namespace sg { namespace sm90 { namespace models { namespace mamba_adapter {

// ── Scan strategy selector ────────────────────────────────────────────
enum class ScanStrategy {
    kSequential,              // N < 256
    kParallelPrecompute,      // 256 <= N < 1024
    kBilevelGemm              // N >= 1024
};

__host__ __forceinline__ ScanStrategy select_strategy(int seq_len) {
    if (seq_len < PSCAN_THRESHOLD)            return ScanStrategy::kSequential;
    if (seq_len < GEMM_PRECOMPUTE_THRESHOLD)  return ScanStrategy::kParallelPrecompute;
    return ScanStrategy::kBilevelGemm;
}

// ── Forward: selective scan ───────────────────────────────────────────
// Adapts model tensors [B, N, d_inner] etc. to the per-element scan
// interface of sg::device::sm90::mamba3_scan_step and friends.
//
// x:          [B, N, d_inner]   input activations
// dt:         [B, N, d_inner]   delta-t (post-softplus)
// A_log:      [d_inner, d_state] log of diagonal A matrix
// B:          [B, N, d_state]   input-dependent B projection
// C:          [B, N, d_state]   input-dependent C projection
// y:          [B, N, d_inner]   output activations
// state_save: [B, d_inner, d_state] saved SSM state for backward

template <typename ActT>
cudaError_t selective_scan_forward(
    const ActT* x,
    const ActT* dt,
    const ActT* A_log,
    const ActT* B,
    const ActT* C,
    ActT* y,
    ActT* state_save,
    int batch, int seq_len, int d_inner, int d_state,
    cudaStream_t stream
) {
    // TODO: Dispatch to sg::device::sm90 scan primitives based on strategy.
    //
    // ScanStrategy strat = select_strategy(seq_len);
    // switch (strat) {
    //   case ScanStrategy::kSequential:
    //     Launch kernel calling mamba3_scan_step per (batch, d_inner) thread.
    //   case ScanStrategy::kParallelPrecompute:
    //     Launch parallel precompute kernel, then parallel prefix scan.
    //   case ScanStrategy::kBilevelGemm:
    //     Launch bilevel_precompute_gemm, then parallel scan.
    // }
    //
    // The adapter's job: convert [B, N, d_inner] layout to per-element
    // pointers that mamba3_scan_step expects (one thread per d_inner dim,
    // sequential over N timesteps for the sequential path).
    return cudaErrorNotReady;  // stub
}

// ── Backward: selective scan ──────────────────────────────────────────
// Reverse-mode through the selective scan. Uses saved state_save from
// forward to avoid full recomputation.

template <typename ActT>
cudaError_t selective_scan_backward(
    const ActT* grad_y,
    const ActT* x, const ActT* dt, const ActT* A_log,
    const ActT* B, const ActT* C,
    const ActT* state_save,
    ActT* grad_x, ActT* grad_dt, ActT* grad_A_log,
    ActT* grad_B, ActT* grad_C,
    int batch, int seq_len, int d_inner, int d_state,
    cudaStream_t stream
) {
    // TODO: Reverse-mode selective scan.
    // The backward scan runs in reverse time order (t = N-1 .. 0).
    // Strategy selection mirrors forward:
    //   - Sequential: reverse loop per (batch, d_inner) thread
    //   - Parallel: reverse prefix scan with adjoint operators
    //   - Bilevel: reverse bilevel scan
    //
    // grad_A_log accumulates across all timesteps via atomic add.
    return cudaErrorNotReady;  // stub
}

}}}}  // namespace sg::sm90::models::mamba_adapter
