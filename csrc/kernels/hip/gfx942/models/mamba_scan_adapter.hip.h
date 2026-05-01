// csrc/kernels/hip/gfx942/models/mamba_scan_adapter.hip.h
// Adapter that exposes the gfx942 SG2 Mamba selective-scan kernels through
// a model-layer interface matching the sm_90 version.
//
// The actual scan device functions live in:
//   csrc/device/optimizers/gfx942/supergrok2_gfx942.hip.cuh
//
// The CDNA3 BF16 MFMA fast path is already inlined in the canonical
// mamba_peer_batched_step for gfx942 (active when leading dim >= 128).
//
// Wave size 64, LDS budget 64 KB per CU.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/device/optimizers/gfx942/supergrok2_gfx942.hip.cuh"

namespace sg { namespace gfx942 { namespace models { namespace mamba_scan {

// Max LDS per CU on gfx942
constexpr int kMaxLdsBytes = 65536;

// -- Scan configuration -------------------------------------------------------
struct MambaScanConfig {
    int d_model;
    int d_state;
    int d_inner;
    int seq_len;
    int batch;
    int lds_bytes;          // LDS allocation for the scan kernel
    int waves_per_eu;       // occupancy hint for CDNA3 scheduler
    bool use_bf16_mfma;     // enable BF16 MFMA fast path (leading dim >= 128)
};

// -- Forward selective scan ---------------------------------------------------
// Wraps the SG2 mamba_peer_batched_step forward scan for gfx942.
// BF16 MFMA is used automatically when d_inner >= 128.
template <typename ActT>
hipError_t mamba_scan_forward(
    const ActT* __restrict__ x,         // [B, L, D]
    const ActT* __restrict__ delta,     // [B, L, D_inner]
    const ActT* __restrict__ A,         // [D_inner, D_state]
    const ActT* __restrict__ B,         // [B, L, D_state]
    const ActT* __restrict__ C,         // [B, L, D_state]
    const float* __restrict__ D_skip,   // [D_inner] or nullptr
    ActT* __restrict__ out,             // [B, L, D]
    ActT* __restrict__ scan_states,     // [B, D_inner, D_state]
    const MambaScanConfig& cfg,
    hipStream_t stream
) {
    // TODO: wire up sg::device::gfx942 scan device functions
    // The batched scan launcher with BF16 MFMA dispatch lives in the
    // fused kernels (csrc/fused/gfx942/fused_mamba_*_gfx942.hip.cpp).
    return hipErrorNotReady;
}

// -- Backward selective scan --------------------------------------------------
template <typename ActT>
hipError_t mamba_scan_backward(
    const ActT* __restrict__ grad_out,  // [B, L, D]
    const ActT* __restrict__ x,
    const ActT* __restrict__ delta,
    const ActT* __restrict__ A,
    const ActT* __restrict__ B,
    const ActT* __restrict__ C,
    const ActT* __restrict__ scan_states,
    ActT* __restrict__ grad_x,
    ActT* __restrict__ grad_delta,
    ActT* __restrict__ grad_A,
    ActT* __restrict__ grad_B,
    ActT* __restrict__ grad_C,
    float* __restrict__ grad_D,
    const MambaScanConfig& cfg,
    hipStream_t stream
) {
    return hipErrorNotReady;
}

// -- Utility: compute LDS requirement -----------------------------------------
inline int compute_scan_lds_bytes(const MambaScanConfig& cfg) {
    // Scan state: d_inner * d_state floats
    // Work buffer: d_inner floats for reduction
    int state_bytes = cfg.d_inner * cfg.d_state * sizeof(float);
    int work_bytes  = cfg.d_inner * sizeof(float);
    int total = state_bytes + work_bytes;
    // Clamp to gfx942 LDS limit
    return (total <= kMaxLdsBytes) ? total : kMaxLdsBytes;
}

}}}}  // namespace sg::gfx942::models::mamba_scan
