#ifndef CSRC_BACKENDS_HIP_GFX942_MOE_COMPACTION_GFX942_HIP_HPP_
#define CSRC_BACKENDS_HIP_GFX942_MOE_COMPACTION_GFX942_HIP_HPP_
// ============================================================================
// moe_compaction_gfx942.hip.hpp — AMD-native gfx942 MoE compaction/scatter tail.
//
// This is the LAST "still ATen by design" AMD item: the MoE (Mixture-of-Experts)
// compaction/scatter path of MoEAwareSuperGrok2 (Stage 1B). The companion
// supergrok2_gfx942.hip.hpp keeps the ATen orchestration of these primitives
// (moe_filter_active_params / moe_scatter_results / moe_count_expert_activations,
// which reach rocPRIM/rocBLAS internally); this header adds the REAL hand-written
// AMDGCN device kernels for the three genuinely device-shaped ones, built on the
// shared, compiler-verified primitives:
//   csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp
//
// THREE device kernels (§5 below):
//   (1) moe_filter_active_kernel    — stream-compaction of active (row,expert)
//       pairs via a WARP-AGGREGATED ballot/mbcnt scan within each wavefront +
//       an LDS exclusive-scan ACROSS wavefronts + one atomic_add_agent_u32 to
//       claim the global write-cursor base. This is the rocPRIM-shaped part done
//       natively (one atomic per workgroup, not per kept element).
//   (2) moe_scatter_results_kernel  — grid-stride scatter of compacted expert
//       outputs back to their original row positions (amd::streaming_store),
//       accumulating with atomic_add_agent_f32 when several experts hit a row.
//   (3) moe_expert_histogram_kernel — per-expert activation counts via the DPP
//       wave reduction (amd::wave_reduce_add_u32_dpp) + atomic_add_agent_u32,
//       for dynamic load balancing / dead-expert recycling.
//
// COMPILE ROUTING (two passes, one header — the canonical grokfast pattern):
//   * HOST pass  (`!__AMDGCN__`): sees ONLY section (A) — the launch entry-point
//     shims documenting the hipLaunchKernelGGL wiring. The thin host
//     supergrok2 `.hip.cpp` TU keeps the proven ATen compaction as the fallback.
//   * DEVICE pass (`__AMDGCN__` — scripts/amdgcn_check.sh — or `__HIPCC__`):
//     sees ONLY section (B), the three device kernels.
//
// COMPILE-TIME-CONSTANT contract: every DPP-ctrl / ds_swizzle / cvt byte-index
// arg reached through amd:: is already a template parameter there (the gate
// REQUIRES it; runtime args fail). This header introduces no new runtime-arg
// builtins; the ballot/mbcnt/readfirstlane it adds take no such immediates.
//
// HARDWARE-GATED 🟡: device-compile-verified for gfx942 via
// scripts/amdgcn_check.sh ONLY. MI300X numeric parity vs the ATen reference and
// the live hipLaunchKernelGGL link are DEFERRED on-silicon (no hipcc / no device
// in this environment) — see HARDWARE_VALIDATION.md, Stage 5. Until that runs
// the ATen path in supergrok2_gfx942.hip.hpp stays the active fallback.
// ============================================================================

// ════════════════════════════════════════════════════════════════════════════
// (A) HOST orchestration — launch entry-point shims. Compiled by the HOST pass
// only (torch/ATen, invisible to the bare device gate). On a real hipcc build
// these launch the §5 kernels via hipLaunchKernelGGL (see §5.LAUNCH); for now
// the supergrok2 host TU keeps the ATen compaction as the validated fallback.
// ════════════════════════════════════════════════════════════════════════════
#if !defined(__AMDGCN__)

// ── §5.LAUNCH (host-side wiring note) ────────────────────────────────────────
// On a real `.hip` (hipcc) build, the MoE compaction launchers in
// supergrok2_gfx942.hip.hpp launch these §5 kernels instead of the ATen
// nonzero/index_select/index_copy_ chain:
//
//   // (1) stream-compaction of active (row,expert) pairs — P rows, E experts:
//   dim3 g1(min(4096u,(P+255)/256)), b1(256);              // 4 wavefronts/block
//   *d_cursor = 0;  // global write cursor, one u32 (hipMemsetAsync)
//   hipLaunchKernelGGL(moe_filter_active_kernel, g1, b1, 0, stream,
//                      param_to_expert, expert_active, scatter_indices,
//                      d_cursor, P);
//   // compact_count[0] = *d_cursor after the launch.
//
//   // (2) scatter compacted [K,row_stride] outputs back to dense rows:
//   dim3 g2(min(4096u,(K*row_stride+255)/256)), b2(256);
//   hipLaunchKernelGGL(moe_scatter_results_kernel, g2, b2, 0, stream,
//                      compact_out, scatter_indices, dense_out,
//                      K, row_stride, /*accumulate=*/1);
//
//   // (3) per-expert activation histogram — N rows × E experts gate logits:
//   dim3 g3(min(4096u,(N+255)/256)), b3(256);
//   hipMemsetAsync(expert_counts, 0, E*sizeof(unsigned), stream);
//   hipLaunchKernelGGL(moe_expert_histogram_kernel, g3, b3, 0, stream,
//                      gate_logits, expert_counts, threshold, N, E);
//
// 🟡 DEFERRED: the live launch + hipcc link is MI300X-gated. The supergrok2 host
// TU currently keeps the ATen path (numerics-correct, rocPRIM-backed); these §5
// kernels are COMPILER-VERIFIED for gfx942 via scripts/amdgcn_check.sh and ready
// to wire in once the model TU migrates .hip.cpp -> .hip on hardware.
//
// WHAT STAYS ATen/HOST (by design, and why):
//   * moe_dynamic_expert_fwd/bwd — dense per-token batched-bmm expert MLPs;
//     these are GEMM/MFMA-shaped (rocBLAS-backed via ATen bmm), not
//     compaction-shaped, so they live with the §5 MFMA attention path, not here.
//   * moe_compute_load_balance_loss / moe_apply_frequency_scaling — tiny
//     [E]-vector softmax/clamp reductions; ATen is already optimal, a custom
//     kernel would not help.
//   * moe_dynamic_expert_load — masked weight gather; same nonzero+gather shape
//     as (1) but operating on whole expert weight matrices, kept ATen for now.
//   * moe_scan_compacted — VESTIGIAL (Mamba-era; SG2's mixer is CSA/HCA).

#endif  // !defined(__AMDGCN__)  — end host orchestration (A)

// ════════════════════════════════════════════════════════════════════════════
// (B) DEVICE pass — real hand-written AMDGCN MoE compaction kernels (§5).
// Compiled by the AMDGCN device pass only: the Stage-5 gate (__AMDGCN__, no
// hipcc) AND the hipcc device pass (__HIPCC__). The host `.hip.cpp` TU never
// sees it — that pass keeps the ATen orchestration (which LAUNCHES these kernels
// via hipLaunchKernelGGL, see §5.LAUNCH).
// ════════════════════════════════════════════════════════════════════════════
#if defined(__AMDGCN__) || defined(__HIPCC__) || defined(GROK_HIP_DEVICE)
#include "csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp"
// ── gate-only launch-builtin shim (verbatim from the grokfast/looksam/mamba3
// exemplar). The free-standing AMDGCN device gate stubs out <hip/hip_runtime.h>,
// so the launch builtins (threadIdx/blockIdx/blockDim/gridDim, __global__,
// __shared__) HIP normally provides are absent. Model them with the AMDGCN
// workitem ISA builtins so the device bodies type-check. Active ONLY on the
// bare gate.
#if defined(__AMDGCN__) && !defined(__HIPCC__)
#ifndef GROK_GFX942_LAUNCH_SHIM_
#define GROK_GFX942_LAUNCH_SHIM_
struct GrokTidX { __device__ operator unsigned() const { return __builtin_amdgcn_workitem_id_x(); } };
struct GrokBidX { __device__ operator unsigned() const { return __builtin_amdgcn_workgroup_id_x(); } };
struct GrokBdimX{ __device__ operator unsigned() const { return __builtin_amdgcn_workgroup_size_x(); } };
struct GrokGdimX{ __device__ operator unsigned() const { return __builtin_amdgcn_grid_size_x()
                                                              / __builtin_amdgcn_workgroup_size_x(); } };
struct GrokThreadIdx { GrokTidX  x; };
struct GrokBlockIdx  { GrokBidX  x; };
struct GrokBlockDim  { GrokBdimX x; };
struct GrokGridDim   { GrokGdimX x; };
static GrokThreadIdx threadIdx;
static GrokBlockIdx  blockIdx;
static GrokBlockDim  blockDim;
static GrokGridDim   gridDim;
#ifndef __global__
#define __global__ __attribute__((amdgpu_kernel))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#endif  // GROK_GFX942_LAUNCH_SHIM_
#endif  // bare gate
// ============================================================================
// §5  AMD-NATIVE device kernels (Stage 5 hand-written AMDGCN MoE compaction).
//
// Built on the shared, compiler-verified primitives in
//   csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp  (namespace amd = …):
//   amd::streaming_load / streaming_store (§2.7, read-once VMEM),
//   amd::wave_reduce_add_u32_dpp (§2.6 DPP butterfly),
//   amd::atomic_add_agent_u32 / atomic_add_agent_f32 (§2.13 cross-XCD atomics),
//   amd::workgroup_barrier_release / _acquire (§2.13 LDS handoff).
// CDNA3 wavefront width = 64.
// ============================================================================
namespace sg { namespace gfx942 { namespace native {

namespace amd = ::sg::gfx942::amdgcn;

static constexpr int kWave    = 64;    // == amd::kWave (CDNA3 wavefront width)
static constexpr int kMaxWpb  = 64;    // LDS slots: blockDim.x <= kWave*kWave

// ── lane / wave helpers (the looksam/attention idiom) ───────────────────────
__device__ __forceinline__ int moe_lane()   { return static_cast<int>(threadIdx.x) % kWave; }
__device__ __forceinline__ int moe_waveId() { return static_cast<int>(threadIdx.x) / kWave; }
__device__ __forceinline__ int moe_wpb()    { return static_cast<int>(blockDim.x) / kWave; }

// ── §5.0  intra-wave exclusive ballot prefix (warp-aggregated compaction) ────
// Given this lane's boolean `pred`, return (lane_prefix, wave_total): the number
// of set predicates in lanes < this lane, and the wavefront popcount. The
// canonical AMD compaction primitive: a wave-64 ballot mask, then mbcnt_lo/hi
// gives each lane its exclusive prefix-popcount in ONE pair of instructions (no
// LDS, no DPP scan). ballot/mbcnt/popcount take no compile-time-immediate args,
// so they are gate-safe with no template params (verified, scripts/amdgcn_check).
__device__ __forceinline__ void wave_ballot_exclusive(
    int pred, unsigned& lane_prefix, unsigned& wave_total)
{
    const unsigned long long bal = __builtin_amdgcn_ballot_w64(pred);
    const unsigned lo = static_cast<unsigned>(bal & 0xffffffffull);
    const unsigned hi = static_cast<unsigned>(bal >> 32);
    // mbcnt_lo(mask, 0) counts set bits of `mask` strictly below this lane in the
    // low 32 lanes; mbcnt_hi continues the accumulation into the high 32 lanes.
    lane_prefix = __builtin_amdgcn_mbcnt_hi(hi, __builtin_amdgcn_mbcnt_lo(lo, 0u));
    wave_total  = static_cast<unsigned>(__builtin_popcountll(bal));
}

// ── §5.1  stream-compaction of active (row,expert) pairs ────────────────────
// Keep workitem `i` (a flattened param-row index 0..P) iff the expert it is
// assigned to is active: expert_active[param_to_expert[i]] != 0. The kept `i`
// values are written densely into out_indices[] in (mostly) ascending order;
// *cursor ends at the total kept count (== ATen compact_count[0]).
//
// Compaction algorithm (one atomic per workgroup, not per kept element):
//   1. each lane computes pred + its intra-wave exclusive ballot prefix (§5.0);
//   2. lane 0 of each wave publishes its wave_total to LDS;
//   3. wave 0, lane 0 does a serial exclusive scan of the <=64 wave totals AND
//      claims a single global base via atomic_add_agent_u32(cursor, block_total);
//   4. every kept lane writes to out_indices[base + wave_off + lane_prefix].
// The serial scan over <=64 slots is cheap (the cross-wave step the looksam
// reduction does with a second DPP reduce; here it must be a *prefix*, so it is
// an explicit running sum). Matches the ATen filter's (out -> original i)
// scatter contract; ascending order within a wave, wave-ordered across the block.
__device__ __forceinline__ void moe_filter_active_block(
    const int* __restrict__ param_to_expert,  // [P]  expert id per param row
    const int* __restrict__ expert_active,     // [E]  nonzero == active
    int*       __restrict__ out_indices,       // [P]  compacted kept row ids
    unsigned*  __restrict__ cursor,            // [1]  global write cursor (u32)
    int P)
{
    __shared__ unsigned lds_tot[kMaxWpb];   // per-wave kept counts
    __shared__ unsigned lds_base[1];        // this block's global base
    __shared__ unsigned lds_woff[kMaxWpb];  // per-wave intra-block offsets

    const int tid    = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                     + static_cast<int>(threadIdx.x);
    const int lane   = moe_lane();
    const int waveId = moe_waveId();
    const int wpb    = moe_wpb();

    // (1) predicate + intra-wave exclusive prefix.
    int pred = 0, row = -1;
    if (tid < P) {
        row = tid;
        const int e = static_cast<int>(amd::streaming_load(&param_to_expert[row]));
        pred = (e >= 0) ? (expert_active[e] != 0) : 0;
    }
    unsigned lane_prefix = 0u, wave_total = 0u;
    wave_ballot_exclusive(pred, lane_prefix, wave_total);

    // (2) publish per-wave totals to LDS.
    if (lane == 0) lds_tot[waveId] = wave_total;
    amd::workgroup_barrier_release();

    // (3) wave 0 / lane 0: exclusive scan of wave totals + one global atomic.
    if (waveId == 0 && lane == 0) {
        unsigned run = 0u;
        for (int w = 0; w < wpb; ++w) {
            const unsigned c = lds_tot[w];
            lds_woff[w] = run;       // exclusive offset of wave w within the block
            run += c;
        }
        // claim a contiguous [base, base+run) slice of the global output. The
        // §2.13 amd::atomic_add_agent_u32 wrapper returns void (side-effecting);
        // we need the PRIOR cursor value as our base, so use the AGENT-scope
        // atomic-fetch-add builtin directly (the gate maps __hip_atomic_fetch_add
        // -> __atomic_fetch_add; on hipcc this is the same agent-scope RMW).
        lds_base[0] = (run > 0u)
            ? __hip_atomic_fetch_add(cursor, run, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT)
            : 0u;
    }
    amd::workgroup_barrier_acquire();

    // (4) kept lanes write their original row id to the compacted slot.
    if (pred) {
        const unsigned dst = lds_base[0] + lds_woff[waveId] + lane_prefix;
        amd::streaming_store(&out_indices[dst], row);
    }
}

extern "C" __global__ void moe_filter_active_kernel(
    const int* __restrict__ param_to_expert,
    const int* __restrict__ expert_active,
    int*       __restrict__ out_indices,
    unsigned*  __restrict__ cursor,
    int P)
{
    moe_filter_active_block(param_to_expert, expert_active, out_indices, cursor, P);
}

// ── §5.2  scatter compacted expert outputs back to original rows ────────────
// dense_out[scatter_indices[j] * row_stride + c] (+)= compact_out[j*row_stride+c]
// for j in [0,K), c in [0,row_stride). Grid-stride over the flattened K*row_stride
// element space; read-once load of the compacted value (amd::streaming_load).
// `accumulate`: when several experts route to the same row (top-k MoE), the
// writes collide, so accumulate via amd::atomic_add_agent_f32 (cross-XCD safe).
// When the caller guarantees a 1:1 (row -> single expert) mapping it passes
// accumulate=0 and we use a plain streaming_store (the ATen index_copy_ case).
__device__ __forceinline__ void moe_scatter_results_block(
    const float* __restrict__ compact_out,    // [K, row_stride] f32, compacted
    const int*   __restrict__ scatter_indices,// [K] original dense row per slot
    float*       __restrict__ dense_out,       // [R, row_stride] f32, dense dst
    int K, int row_stride, int accumulate)
{
    const long total  = static_cast<long>(K) * static_cast<long>(row_stride);
    const long stride = static_cast<long>(gridDim.x) * static_cast<long>(blockDim.x);
    for (long e = static_cast<long>(blockIdx.x) * static_cast<long>(blockDim.x)
                  + static_cast<long>(threadIdx.x);
         e < total; e += stride) {
        const int j   = static_cast<int>(e / row_stride);   // compacted slot
        const int c   = static_cast<int>(e % row_stride);   // column within row
        const int dst = static_cast<int>(amd::streaming_load(&scatter_indices[j]));
        if (dst < 0) continue;
        const float v = amd::streaming_load(&compact_out[e]);
        float* addr = &dense_out[static_cast<long>(dst) * row_stride + c];
        if (accumulate) amd::atomic_add_agent_f32(addr, v);
        else            amd::streaming_store(addr, v);
    }
}

extern "C" __global__ void moe_scatter_results_kernel(
    const float* __restrict__ compact_out,
    const int*   __restrict__ scatter_indices,
    float*       __restrict__ dense_out,
    int K, int row_stride, int accumulate)
{
    moe_scatter_results_block(compact_out, scatter_indices, dense_out,
                              K, row_stride, accumulate);
}

// ── §5.3  per-expert activation histogram (dynamic load balancing) ──────────
// expert_counts[e] += #rows with gate_logits[row, e] > threshold, for the
// dynamic load-balance / dead-expert-recycling controller (the ATen
// moe_count_expert_activations). One wavefront cooperates per expert column:
// the 64 lanes stride the N rows, each lane tallies its local count, the
// wavefront DPP-reduces it (amd::wave_reduce_add_u32_dpp), and lane 0 commits
// the column total with a single atomic_add_agent_u32. gridDim.y selects the
// expert column e; blockDim.x is a single wavefront (64) striding rows.
__device__ __forceinline__ void moe_expert_histogram_block(
    const float* __restrict__ gate_logits,   // [N, E] row-major f32
    unsigned*    __restrict__ expert_counts,  // [E] u32, pre-zeroed
    float threshold, int N, int E, int e)
{
    if (e >= E) return;
    const int lane   = moe_lane();
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    unsigned local = 0u;
    for (int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
         row < N; row += stride) {
        const float g = amd::streaming_load(&gate_logits[static_cast<long>(row) * E + e]);
        local += (g > threshold) ? 1u : 0u;
    }
    // every lane holds the wavefront total after the DPP butterfly (§2.6).
    local = amd::wave_reduce_add_u32_dpp(local);
    if (lane == 0 && local != 0u) amd::atomic_add_agent_u32(&expert_counts[e], local);
}

extern "C" __global__ void moe_expert_histogram_kernel(
    const float* __restrict__ gate_logits,
    unsigned*    __restrict__ expert_counts,
    float threshold, int N, int E)
{
    // gridDim.y picks the expert column; under the bare gate gridDim has only .x,
    // so derive e from blockIdx (the host passes a 2-D grid on hipcc — see
    // §5.LAUNCH; the gate path just needs to type-check the body).
    const int e = static_cast<int>(__builtin_amdgcn_workgroup_id_y());
    moe_expert_histogram_block(gate_logits, expert_counts, threshold, N, E, e);
}

}}} // namespace sg::gfx942::native
#endif  // (B) device pass

#endif  // CSRC_BACKENDS_HIP_GFX942_MOE_COMPACTION_GFX942_HIP_HPP_
