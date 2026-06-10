#pragma once
// csrc/fused/megakernel_common_hip.hip.hpp
// ─────────────────────────────────────────────────────────────────────────
// Stage 6 — gfx942 (CDNA3 / MI300X) L3 persistent-megakernel internals
// (§1.1–§1.4, §1.13, §1.14). The AMD twin of megakernel_common.cuh.
//
// Same four pieces (task queue, work-stealing, SM/CU pinning, hand-built grid
// barrier), expressed with the CDNA3 primitives instead of the Hopper ones:
//
//   §1.1  TaskQueue   — a global work counter pulled with an AGENT-scope
//                       atomic_fetch_add (visible across all 8 XCDs, §2.13).
//   §1.2  work-steal  — identical pattern: idle workgroups keep pulling.
//   §1.3  cu_id()     — __builtin_amdgcn_s_getreg of HW_ID gives the CU /
//                       XCD the wavefront runs on, so the host can pin one
//                       persistent workgroup per CU and keep its optimizer
//                       slice resident in that CU's L2 partition.
//   §1.4  GridBarrier — two AGENT-scope global atomics + a sense-reversing
//                       generation, with __builtin_amdgcn_s_barrier for the
//                       intra-workgroup join and AGENT-scope fences for the
//                       device-wide release/acquire.
//
// §1.13  SCHEDULING NOTE — the AMD megakernel is NOT warp-specialized.
//   Hopper's producer/consumer split (a TMA warp-group + a WGMMA warp-group,
//   setmaxnreg register reallocation) has no CDNA analog: there is no async
//   TMA proxy, no mbarrier transaction barrier, and no per-warp-group register
//   reallocation. Instead the CDNA3 megakernel uses **ping-pong / 4-wave
//   interleave** scheduling: 4 wavefronts per SIMD take turns so that while one
//   wave issues MFMA the others issue VMEM/DS, hiding latency via occupancy
//   rather than via a dedicated producer warp-group. The §2.11
//   sched_group_barrier hints (in amdgcn_primitives.hip.hpp) pin that
//   interleave. So the compute STAGES differ by arch even though the SCHEDULER
//   substrate below (queue + grid barrier) is identical in shape.
//
// §1.14  fences: explicit + minimal. One AGENT release fence before publishing
//   arrival, one AGENT acquire fence after the generation advances. The
//   intra-workgroup join uses s_barrier (no global fence needed for it).
//
// HARDWARE-GATED: no hipcc / MI300X here. Device code is compile-checked via
// scripts/amdgcn_check.sh (clang free-standing amdgcn); full launch is 🟡 until
// MI300X (HARDWARE_VALIDATION.md Stage 6).
// ─────────────────────────────────────────────────────────────────────────

#include <hip/hip_runtime.h>

#if defined(__has_include)
#  if __has_include(<cstdint>)
#    include <cstdint>
#  else
typedef unsigned int uint32_t;
#  endif
#else
typedef unsigned int uint32_t;
#endif

// Under real hipcc, blockIdx/threadIdx/blockDim and the __global__/__shared__
// qualifiers are provided by the HIP runtime. The free-standing amdgcn compile
// gate (scripts/amdgcn_check.sh) has only a stub hip_runtime.h (just __device__
// / __forceinline__), so we supply minimal stand-ins there ONLY so the device
// code type-checks. The real definitions win under hipcc (this branch is dead
// there). amdgpu_kernel / shared are the canonical amdgcn lowerings of
// __global__ / __shared__.
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC__)
// On the free-standing gate these qualifiers carry no semantics (we only need
// the bodies to type-check). __global__ becomes a no-op so a templated kernel
// type-checks without the amdgpu_kernel-only attribute machinery; under hipcc
// the real __global__ applies and the kernels are emitted correctly.
#  ifndef __global__
#    define __global__
#  endif
#  ifndef __shared__
#    define __shared__
#  endif
// The flat-work-group-size attribute is kernel-only; drop it on the gate path.
#  define SG_MK_FWGS(lo, hi)
namespace sg_mk_detail {
struct _Dim3Stub { unsigned x, y, z; };
}
static __attribute__((unused)) sg_mk_detail::_Dim3Stub threadIdx{0u, 0u, 0u};
static __attribute__((unused)) sg_mk_detail::_Dim3Stub blockIdx{0u, 0u, 0u};
static __attribute__((unused)) sg_mk_detail::_Dim3Stub blockDim{1u, 1u, 1u};
#else
// Real hipcc path: the flat-work-group-size attribute pins the persistent
// workgroup size (256 = 4 wavefronts of 64) for the §1.13 ping-pong interleave.
#  define SG_MK_FWGS(lo, hi) __attribute__((amdgpu_flat_work_group_size(lo, hi)))
#endif

// SG_TUNED_MEGA_BLOCK — threads per persistent megakernel workgroup, the AMD
// twin of the token in megakernel_common.cuh. Default 256 (= 4 wavefronts of 64
// → byte-identical untuned build). NEEDS-PARITY before a non-default winner
// ships: the §1.13 ping-pong interleave + shared staging assume the 256-thread
// (4-wavefront) shape. Defined here (the HIP path does NOT see the .cuh guard);
// not registered on gfx942 (no AMD silicon in CI + sm_90-led roll-out) so the
// HIP build always takes this default — see report.
#ifndef SG_TUNED_MEGA_BLOCK
#define SG_TUNED_MEGA_BLOCK 256
#endif

namespace sg { namespace fused { namespace gfx942 {

// CDNA3 wavefront width (matches amdgcn_primitives.hip.hpp::kWave).
static constexpr int kWave = 64;

// =========================================================================
//  §1.3  CU/XCD pinning: read the hardware id register.
//
//  s_getreg HW_ID exposes the CU (compute unit) and SE (shader engine) the
//  wavefront runs on. On CDNA3 the XCD a workgroup landed on is derivable from
//  this; the host pins one persistent workgroup per CU so its optimizer-state
//  slice stays warm in that CU's L2 slice for the whole step (§1.3, the AMD
//  analog of %smid pinning). HW_ID encoding: bits [11:8] = CU_ID, [13] = SE.
// =========================================================================
__device__ __forceinline__ unsigned cu_id() {
#if defined(__AMDGCN__)
    // s_getreg_b32 dst, hwreg(HW_REG_HW_ID=4, offset=0, width=32).
    // The intrinsic packs (reg<<0)|(offset<<6)|((width-1)<<11); HW_ID=4.
    unsigned hwid = __builtin_amdgcn_s_getreg(/*hwreg=*/((4) | (0 << 6) | ((32 - 1) << 11)));
    return (hwid >> 8) & 0xfu;   // CU_ID field
#else
    return 0u;
#endif
}

// AGENT-scope (device-wide, cross-XCD) atomic add. Mirrors §2.13. Returns the
// previous value (fetch-add semantics) so the grid barrier can detect the last
// arriver. Under the free-standing amdgcn gate __hip_atomic_fetch_add is macro-
// shimmed to __atomic_fetch_add (see scripts/amdgcn_check.sh); under real hipcc
// it carries the AGENT memory scope.
__device__ __forceinline__ unsigned atomic_add_agent_u32(unsigned* addr,
                                                         unsigned v) {
    return __hip_atomic_fetch_add(addr, v, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ int atomic_add_agent_i32(int* addr, int v) {
    return __hip_atomic_fetch_add(addr, v, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
}
// AGENT-scope exchange/load. Under real hipcc these carry the AGENT scope; the
// free-standing gate shims __hip_atomic_load to a scopeless __atomic_load_n and
// has no __hip_atomic_exchange shim, so the exchange routes through the scopeless
// __atomic_exchange_n on the gate path (the AGENT-scoped form is the hipcc path).
__device__ __forceinline__ unsigned atomic_exch_agent_u32(unsigned* addr,
                                                          unsigned v) {
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
    return __hip_atomic_exchange(addr, v, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
#else
    return __atomic_exchange_n(addr, v, __ATOMIC_RELAXED);
#endif
}
__device__ __forceinline__ unsigned atomic_load_agent_u32(unsigned* addr) {
    return __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// =========================================================================
//  §1.1 / §1.2  TaskQueue — global work counter + work-stealing.
//
//  Identical shape to the sm_90 version: one global int the host zero-inits,
//  every persistent workgroup pulls indices via atomic_add_agent until the
//  counter passes n_tasks. AGENT scope makes the counter coherent across all 8
//  XCDs (a CU on XCD 3 and a CU on XCD 7 see one consistent sequence).
// =========================================================================
struct TaskQueue {
    int* g_next_task;
    int  n_tasks;

    __device__ __forceinline__ TaskQueue(int* counter, int total)
        : g_next_task(counter), n_tasks(total) {}

    __device__ __forceinline__ int next() {
        return atomic_add_agent_i32(g_next_task, 1);
    }

    // Per-workgroup pull: lane 0 of the workgroup does the single atomic and
    // broadcasts via LDS; the whole workgroup then cooperates on that tensor.
    __device__ __forceinline__ int next_block(int* lds_slot) {
        if (threadIdx.x == 0) {
            *lds_slot = atomic_add_agent_i32(g_next_task, 1);
        }
        __builtin_amdgcn_s_barrier();
        int t = *lds_slot;
        __builtin_amdgcn_s_barrier();
        return t;
    }
};

// =========================================================================
//  §1.4  GridBarrier — hand-built, reusable, sense-reversing grid barrier.
//
//  Same algorithm as the sm_90 GridBarrier (see that header for the full reuse-
//  hazard argument): each workgroup samples the generation, AGENT-releases its
//  phase writes, joins the arrival count; the last arriver resets the count and
//  advances the generation; everyone else spins on the generation; all then
//  AGENT-acquire. CDNA has no DSMEM/grid::sync, so AGENT-scope global atomics +
//  fences are the only cross-workgroup mechanism (§2.13).
//
//  §1.14 fences: one release fence before publish, one acquire fence after the
//  generation advances. The intra-workgroup join is s_barrier (no global fence).
// =========================================================================
struct GridBarrier {
    unsigned* g_arrived;
    unsigned* g_generation;
    unsigned  n_groups;   // number of persistent workgroups (one per CU)

    __device__ __forceinline__ GridBarrier(unsigned* arrived,
                                            unsigned* generation,
                                            unsigned groups)
        : g_arrived(arrived), g_generation(generation), n_groups(groups) {}

    __device__ __forceinline__ void sync() const {
        // Whole workgroup must reach the barrier before lane 0 publishes.
        __builtin_amdgcn_s_barrier();

        if (threadIdx.x == 0) {
            unsigned my_gen = atomic_load_agent_u32(g_generation);

            // §1.14 RELEASE: phase writes visible device-wide before arrival.
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");

            unsigned prev = atomic_add_agent_u32(g_arrived, 1u);
            if (prev + 1u == n_groups) {
                // Last arriver: clear count for next phase, then advance gen.
                atomic_exch_agent_u32(g_arrived, 0u);
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
                atomic_add_agent_u32(g_generation, 1u);
            } else {
                while (atomic_load_agent_u32(g_generation) == my_gen) {
                    // §1.14b BACKOFF: the AGENT-scope generation poll crosses
                    // the Infinity-Fabric between XCDs every iteration; a tight
                    // spin saturates the fabric and starves the laggard XCD's
                    // arrival traffic. s_sleep parks the wavefront for a short
                    // HW nap (the operand is a compile-time constant; CDNA3
                    // sleeps ~64*N cycles) so the cross-XCD poll cadence drops.
                    // Behavior-preserving: still exits the instant the
                    // generation advances; only the poll rate changes.
#if defined(__AMDGCN__) || defined(__HIP_DEVICE_COMPILE__)
                    __builtin_amdgcn_s_sleep(2);
#endif
                }
            }

            // §1.14 ACQUIRE: see peers' released phase writes.
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        }

        __builtin_amdgcn_s_barrier();
    }

    // #4 — barrier with a FUSED task-counter reset (AMD twin). The L3 phase loop
    // used 4 grid barriers + 2 standalone counter resets; this folds the reset
    // into the last-arriver critical section (cleared BEFORE the generation
    // advance, so observers of the new gen see the counter == 0), cutting the L3
    // step to 2 grid barriers with identical ordering/visibility. Pass
    // ctx.g_next_task as `reset_counter`.
    __device__ __forceinline__ void sync_reset(int* reset_counter) const {
        __builtin_amdgcn_s_barrier();

        if (threadIdx.x == 0) {
            unsigned my_gen = atomic_load_agent_u32(g_generation);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            unsigned prev = atomic_add_agent_u32(g_arrived, 1u);
            if (prev + 1u == n_groups) {
                atomic_exch_agent_u32(g_arrived, 0u);
                atomic_exch_agent_u32(
                    reinterpret_cast<unsigned*>(reset_counter), 0u);
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
                atomic_add_agent_u32(g_generation, 1u);
            } else {
                while (atomic_load_agent_u32(g_generation) == my_gen) {
#if defined(__AMDGCN__) || defined(__HIP_DEVICE_COMPILE__)
                    __builtin_amdgcn_s_sleep(2);
#endif
                }
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        }

        __builtin_amdgcn_s_barrier();
    }
};

// =========================================================================
//  PersistentContext — uniform global-scratch bundle (AMD twin of the sm_90
//  one), so the generated gfx942 megakernel signatures match across cells.
// =========================================================================
struct PersistentContext {
    int*      g_next_task;
    unsigned* g_arrived;
    unsigned* g_generation;
    int       n_tasks;
    unsigned  n_groups;

    __device__ __forceinline__ TaskQueue queue() const {
        return TaskQueue(g_next_task, n_tasks);
    }
    __device__ __forceinline__ GridBarrier barrier() const {
        return GridBarrier(g_arrived, g_generation, n_groups);
    }
};

}}} // namespace sg::fused::gfx942
