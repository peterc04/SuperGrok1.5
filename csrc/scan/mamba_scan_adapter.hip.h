// csrc/scan/mamba_scan_adapter.hip.h — HIP scan adapter.
// Moved here in Phase 4 of the refactor (shared between Mamba model and SG2
// optimizer). Thin adapter wrapping SG2's gfx942 mamba3_* scan kernels for
// model-layer use. Same interface as the sm_90 adapter but delegates to
// gfx942 primitives.
//
// The actual scan device functions live in:
//   csrc/device/optimizers/gfx942/supergrok2_gfx942.hip.cuh
//
// Key differences from sm_90:
//   - Wave-64 scan butterfly (strides start at 32, not 16)
//   - No warp-specialized scan; use producer-consumer wavefront pair
//   - __shfl_xor instead of __shfl_down_sync (via platform.h SHFL_DOWN)
//   - BF16 MFMA fast path activates when leading dim >= 128
//   - rocBLAS for the GEMM precompute path (N >= 1024)
//
// The CDNA3 BF16 MFMA fast path is already inlined in the canonical
// mamba_peer_batched_step for SG2. LDS budget: 64 KB per CU on gfx942.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/device/optimizers/gfx942/supergrok2_gfx942.hip.cuh"

namespace sg { namespace gfx942 { namespace models { namespace mamba_adapter {

// Max LDS per CU on gfx942
constexpr int kMaxLdsBytes = 65536;

// -- Scan strategy selector ---------------------------------------------------
enum class ScanStrategy {
    kSequential,              // N < 256
    kParallelPrecompute,      // 256 <= N < 1024
    kBilevelGemm              // N >= 1024 (rocBLAS)
};

__host__ __forceinline__ ScanStrategy select_strategy(int seq_len) {
    if (seq_len < PSCAN_THRESHOLD)            return ScanStrategy::kSequential;
    if (seq_len < GEMM_PRECOMPUTE_THRESHOLD)  return ScanStrategy::kParallelPrecompute;
    return ScanStrategy::kBilevelGemm;
}

// -- Sequential scan kernel (N < 256) -----------------------------------------
// One thread per (batch, d_inner) pair. Sequential over seq_len timesteps.
// Uses wave-64 __shfl_xor for cross-lane communication.
template <typename ActT>
__global__
GROK_FLAT_WORK_GROUP_SIZE(64, 256)
GROK_WAVES_PER_EU(1, 4)
void selective_scan_seq_kernel(
    const ActT* __restrict__ x,       // [B, N, d_inner]
    const ActT* __restrict__ dt,      // [B, N, d_inner]
    const ActT* __restrict__ A_log,   // [d_inner, d_state]
    const ActT* __restrict__ B,       // [B, N, d_state]
    const ActT* __restrict__ C,       // [B, N, d_state]
    ActT* __restrict__ y,             // [B, N, d_inner]
    ActT* __restrict__ state_save,    // [B, d_inner, d_state]
    int seq_len, int d_inner, int d_state
) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = global_tid / d_inner;
    const int j = global_tid % d_inner;  // d_inner lane

    // State in registers
    float h[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) h[s] = 0.0f;

    // Preload A coefficients (log-space -> negative exp)
    float A[MAX_D_STATE];
    #pragma unroll 4
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(static_cast<float>(A_log[j * d_state + s]));

    // Sequential scan over timesteps
    for (int t = 0; t < seq_len; t++) {
        const int t_off = batch_idx * seq_len;
        float x_val = static_cast<float>(x[(t_off + t) * d_inner + j]);
        float dt_val = static_cast<float>(dt[(t_off + t) * d_inner + j]);

        float y_val = 0.0f;
        #pragma unroll 4
        for (int s = 0; s < d_state; s++) {
            // Trapezoidal discretization
            float dtA = dt_val * A[s];
            float A_bar = (1.0f + dtA * 0.5f) / (1.0f - dtA * 0.5f + 1e-8f);
            float B_val = static_cast<float>(B[(t_off + t) * d_state + s]);
            float B_bar = dt_val * B_val;

            h[s] = A_bar * h[s] + B_bar * x_val;

            float C_val = static_cast<float>(C[(t_off + t) * d_state + s]);
            y_val += h[s] * C_val;
        }

        y[(t_off + t) * d_inner + j] = static_cast<ActT>(y_val);
    }

    // Save final state
    if (state_save != nullptr) {
        #pragma unroll 4
        for (int s = 0; s < d_state; s++)
            state_save[(batch_idx * d_inner + j) * d_state + s] =
                static_cast<ActT>(h[s]);
    }
}

// -- Forward: selective scan --------------------------------------------------
template <typename ActT>
hipError_t selective_scan_forward(
    const ActT* x, const ActT* dt, const ActT* A_log,
    const ActT* B, const ActT* C,
    ActT* y, ActT* state_save,
    int batch, int seq_len, int d_inner, int d_state,
    hipStream_t stream
) {
    ScanStrategy strat = select_strategy(seq_len);

    if (strat == ScanStrategy::kSequential) {
        // One thread per (batch, d_inner); sequential over seq_len
        int total_threads = batch * d_inner;
        int block = 256;
        int grid = (total_threads + block - 1) / block;
        selective_scan_seq_kernel<ActT>
            <<<grid, block, 0, stream>>>(
                x, dt, A_log, B, C, y, state_save,
                seq_len, d_inner, d_state);
        return hipGetLastError();
    }

    // For parallel strategies (N >= 256), delegate to the SG2 fused kernel
    // infrastructure which handles precompute + Blelloch prefix scan.
    // The BF16 MFMA fast path activates when d_inner >= 128.
    // rocBLAS GEMM precompute activates for N >= 1024.
    //
    // TODO: Wire up parallel scan paths via sg::device::gfx942 primitives.
    // For now, fall back to the sequential kernel which is correct for all N.
    int total_threads = batch * d_inner;
    int block = 256;
    int grid = (total_threads + block - 1) / block;
    selective_scan_seq_kernel<ActT>
        <<<grid, block, 0, stream>>>(
            x, dt, A_log, B, C, y, state_save,
            seq_len, d_inner, d_state);
    return hipGetLastError();
}

// -- Backward: selective scan -------------------------------------------------
// Reverse-mode through the selective scan. Runs in reverse time order.
template <typename ActT>
hipError_t selective_scan_backward(
    const ActT* grad_y,
    const ActT* x, const ActT* dt, const ActT* A_log,
    const ActT* B, const ActT* C,
    const ActT* state_save,
    ActT* grad_x, ActT* grad_dt, ActT* grad_A_log,
    ActT* grad_B, ActT* grad_C,
    int batch, int seq_len, int d_inner, int d_state,
    hipStream_t stream
) {
    // TODO: Reverse-mode selective scan for gfx942.
    // The backward scan runs in reverse time order (t = N-1 .. 0).
    // Strategy mirrors forward: sequential for small N, parallel prefix
    // with adjoint operators for large N. grad_A_log accumulates across
    // all timesteps via atomic add.
    return hipErrorNotReady;  // stub
}

// -- Utility: compute LDS requirement ----------------------------------------
inline int compute_scan_lds_bytes(int d_inner, int d_state) {
    // Scan state: d_inner * d_state floats + d_inner work buffer
    int state_bytes = d_inner * d_state * sizeof(float);
    int work_bytes  = d_inner * sizeof(float);
    int total = state_bytes + work_bytes;
    return (total <= kMaxLdsBytes) ? total : kMaxLdsBytes;
}

}}}}  // namespace sg::gfx942::models::mamba_adapter
