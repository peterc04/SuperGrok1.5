#ifndef SG_FUSED_SM90_TP_TRANSPORT_CUH_
#define SG_FUSED_SM90_TP_TRANSPORT_CUH_
// ============================================================================
// csrc/fused/sm_90/tp_transport.cuh — the TENSOR-PARALLEL TRANSPORT SEAM.
//
// IMPLEMENTS: /workspace/.parallelism_design.md §5.2 (device-initiated all-reduce
// inside the persistent megakernel, FIXED-ORDER so the cross-rank A/A/A gate
// §6.4 holds) + §5.3/§5.4 (the transport go/no-go is an IMPLEMENTATION choice;
// the TP math is transport-agnostic). Authored 1-GPU/CPU per design §7; task #25
// phase-2 authoring (Lane C).
//
// ──────────────────────────────────────────────────────────────────────────
//  THE TRANSPORT SWAP POINT (read this first)
//
//  The TP all-reduce math (tp_layer.cuh) is written against ONE device-side
//  concept, `TpTransport`:
//
//      int     my_pe()  const            — this rank's index in the TP group
//      int     n_pes()  const            — TP group size
//      float*  local(int64_t off)        — THIS pe's buffer at symmetric offset
//      const float* peer(int64_t off,p)  — pe p's buffer at the SAME offset
//      (rendezvous)                      — the caller's GridBarrier + (real
//                                          multi-GPU only) the cross-GPU barrier
//                                          via Transport::rendezvous(bar)
//
//  TWO implementations live here:
//
//  * `LoopbackTransport` — the SINGLE-PROCESS, single-GPU loopback. The
//    "symmetric heap" is ONE device allocation of n_pes * stride floats; virtual
//    pe p's region is base + p*stride. All virtual PEs run in ONE grid (the CTAs
//    are partitioned into contiguous per-PE groups), so the cross-PE rendezvous
//    IS the existing GridBarrier over the whole grid. This is an HONEST
//    simulation: the TP math (shard GEMMs, partial publish, fixed-order
//    ascending-pe fp32 reduce) executes the IDENTICAL code path it executes on
//    NVSHMEM — the only thing swapped is (a) the symmetric-address translation
//    and (b) the rendezvous scope, which is precisely the transport abstraction.
//    It is NOT fake success: the loopback test (tests/hw/test_tp_loopback.py)
//    asserts bit-exact cross-virtual-rank identity + A/A/A + an exact-slice dW
//    check against the unsharded reference — failures fail loudly.
//
//  * `NvshmemTransport` — the REAL device-initiated NVSHMEM transport (design
//    §5.2). NVSHMEM IS NOT INSTALLED ON THIS BOX (verified 2026-06-12: no
//    headers/libs/pip/ldconfig hits), so this struct compiles ONLY under
//    -DSG_HAS_NVSHMEM=1 with the NVSHMEM toolkit on the include/link path. It is
//    NOT a stub that pretends to work: without the flag it does not exist, and
//    any attempt to instantiate the megakernel TP point with it is a loud
//    compile error pointing here. Wiring + validating it is the ONE genuinely
//    8×H100 task (design §7.5) — swap `LoopbackTransport` → `NvshmemTransport`
//    at the call sites and run the §5.4 go/no-go gate.
//
//  WHY offset-based symmetric addressing (not raw pointers): NVSHMEM's
//  symmetric heap guarantees the SAME offset is valid on every PE
//  (nvshmem_malloc returns symmetric addresses; nvshmem_ptr(sym, pe) translates
//  for NVLink load/store). The loopback reproduces exactly that property with
//  base + pe*stride. One addressing model ⇒ one math code path.
//
//  DETERMINISM (design §5.2, non-negotiable): the fixed-order all-reduce below
//  reads every PE's partial in ASCENDING pe order and accumulates fp32. The
//  summation order is STRUCTURAL (never timing-dependent), so the result is
//  bit-identical on every PE and across reruns — the cross-rank analogue of the
//  in-kernel ascending-CTA reduce that made the single-GPU path A/A/A.
//  A `nvshmem_float_sum_reduce` collective is deliberately NOT used (its
//  reduction order is unspecified) — on NVSHMEM the same ascending-pe loop runs
//  over nvshmem_ptr-translated peer pointers (NVLink direct load/store).
// ──────────────────────────────────────────────────────────────────────────
//
// HEADER-ONLY, self-contained: CUDA runtime + the GridBarrier substrate only.
// No torch, no NCCL. Compiles for sm_90a with plain nvcc.
// ============================================================================

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>   // __nv_bfloat16 (the sharded partial-GEMM fwd-decl signatures)

#include "csrc/fused/megakernel_common.cuh"   // GridBarrier (the in-kernel rendezvous)

#if defined(SG_HAS_NVSHMEM)
// Real NVSHMEM headers — present ONLY when the toolkit is installed and the
// build opts in. NOT installed on the authoring box (see header comment).
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace sg { namespace fused { namespace sm90 { namespace tp {

// ─────────────────────────────────────────────────────────────────────────
//  FORWARD DECLARATIONS of the sharded partial-GEMM helpers (DEFINED in
//  tp_layer.cuh — which includes model_stage_decoder_tc.cuh for the dectc_gemm_*
//  bodies). The decoder tile header (model_stage_decoder_tc.cuh) includes THIS
//  header and references these by qualified name inside dectc_*_tile_impl, so
//  the first-phase name lookup needs the declarations here even though the
//  definitions land later in tp_layer.cuh (the include cycle: tp_layer →
//  model_stage_decoder_tc → tp_transport). Signatures MUST match tp_layer.cuh.
// ─────────────────────────────────────────────────────────────────────────
template <int N, class Transport, class WT>
__device__ __forceinline__ void tp_rowparallel_fwd_partial_tile(
        const Transport& tr, int64_t slot_off,
        const __nv_bfloat16* __restrict__ Xown, const WT* __restrict__ Wshard,
        int Kin_local, int Nout,
        __nv_bfloat16* sA, __nv_bfloat16* sB);
template <int N, class Transport, class WT>
__device__ __forceinline__ void tp_colparallel_dx_partial_tile(
        const Transport& tr, int64_t slot_off,
        const __nv_bfloat16* __restrict__ dYown, const WT* __restrict__ Wshard,
        int Kin, int Nout_local,
        __nv_bfloat16* sA, __nv_bfloat16* sB);

// ─────────────────────────────────────────────────────────────────────────
//  LoopbackTransport — single-process symmetric-heap simulation (the 1-GPU
//  honest test transport). See the swap-point doc above.
//
//  Host side: allocate ONE device buffer of `n_pes * stride_floats` floats and
//  pass (base, stride_floats, n_pes). Device side: each CTA derives its virtual
//  pe from its blockIdx (contiguous CTA groups per PE — `pe_of_cta`).
//  The rendezvous is the caller's whole-grid GridBarrier (all virtual PEs share
//  the grid), so `rendezvous()` simply forwards to bar.sync().
// ─────────────────────────────────────────────────────────────────────────
struct LoopbackTransport {
    float*  heap_base;       // [n_pes * stride_floats] device alloc
    int64_t stride_floats;   // per-PE symmetric region size
    int     n_pes_;          // TP degree being simulated
    int     my_pe_;          // this CTA-group's virtual rank

    __device__ __forceinline__ int my_pe() const { return my_pe_; }
    __device__ __forceinline__ int n_pes() const { return n_pes_; }

    // THIS pe's buffer at symmetric offset `off`.
    __device__ __forceinline__ float* local(int64_t off) const {
        return heap_base + (int64_t)my_pe_ * stride_floats + off;
    }
    // Peer pe's buffer at the SAME symmetric offset (ascending-pe reads in the
    // fixed-order reduce). On NVSHMEM this is nvshmem_ptr(sym, pe); here it is
    // the strided region — same contract, swapped translation.
    __device__ __forceinline__ const float* peer(int64_t off, int pe) const {
        return heap_base + (int64_t)pe * stride_floats + off;
    }

    // Cross-PE rendezvous. Loopback: every virtual PE lives in THIS grid, so the
    // whole-grid GridBarrier IS the cross-PE barrier (publish partials → sync →
    // read peers → sync → reuse slots).
    __device__ __forceinline__ void rendezvous(const ::sg::fused::GridBarrier& bar) const {
        bar.sync();
    }

    // Static CTA→PE partition helper: contiguous CTA groups (CTA c of nCTA, P
    // pes → pe = c / (nCTA/P); caller guarantees nCTA % P == 0, asserted in the
    // launchers). Contiguity keeps each virtual rank's tile traversal identical
    // to a real per-GPU grid (ascending tiles within the rank).
    __host__ __device__ __forceinline__ static int pe_of_cta(int cta, int nCTA, int P) {
        return cta / (nCTA / P);
    }
    __host__ __device__ __forceinline__ static int cta_within_pe(int cta, int nCTA, int P) {
        return cta % (nCTA / P);
    }
    __host__ __device__ __forceinline__ static int ctas_per_pe(int nCTA, int P) {
        return nCTA / P;
    }
};

#if defined(SG_HAS_NVSHMEM)
// ─────────────────────────────────────────────────────────────────────────
//  NvshmemTransport — the REAL device-initiated transport (design §5.2).
//  COMPILED ONLY under -DSG_HAS_NVSHMEM=1 (NVSHMEM 3.7.0 IS installed:
//  NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem; the TU
//  that instantiates a kTPComm point must be built -rdc=true and device-linked
//  against libnvshmem_device_sm_90.bc — see /workspace/impl_diffs/tp_kernel.md
//  §6). The math code path is IDENTICAL to the loopback's; only the address
//  translation + the rendezvous scope differ:
//    * `heap_base` is an nvshmem_malloc'd SYMMETRIC allocation (same offset
//      valid on every PE) — a plain cudaMalloc pointer is NOT addressable via
//      nvshmem_ptr, so the TP slots MUST live on the symmetric heap the
//      launcher carves (tp_kernel.md §2/EDIT E). `tp_team_` is the TP-group
//      team (NVSHMEM_TEAM_WORLD when TP == world); `peer()` is called with the
//      TEAM-LOCAL pe and translated to the global pe for nvshmem_ptr.
//    * `peer()` translates via nvshmem_ptr (NVLink direct load/store — the
//      single-node 8×H100 case; a multi-node fabric would use nvshmem_get,
//      the documented extension point, NOT silently emulated);
//    * `rendezvous()` = in-GPU GridBarrier + nvshmem_quiet (drain THIS PE's
//      published partials so peers' NVLink loads see them) + ONE CTA running
//      the cross-GPU TEAM-scoped block barrier + GridBarrier again. The NVSHMEM
//      barrier is entered by exactly one CTA per GPU, NEVER concurrently with
//      the GridBarrier spin (the design's "two barrier systems must not
//      deadlock" rule, tp_nvshmem.md §1(3)).
//    * TEAM SCOPE: nvshmemx_barrier_block(tp_team_) joins ONLY the TP group, so
//      under a 4D mesh DP/PP replicas are not dragged into the TP barrier.
//      NVSHMEM 3.7.0 exposes nvshmemx_barrier_block (host/nvshmemx_coll_api.h:112),
//      so the tp_nvshmem.md §3.1 "version-dependent follow-up" is RESOLVED — no
//      whole-world fallback needed.
//  GO/NO-GO (§5.4): parity vs host-NCCL TP, A/A/A, MFU, ZeRO-3/DP composition.
// ─────────────────────────────────────────────────────────────────────────
struct NvshmemTransport {
    float*         heap_base;  // nvshmem_malloc'd symmetric base (LOCAL pointer)
    nvshmem_team_t tp_team_;   // the TP-group team (NVSHMEM_TEAM_WORLD if TP==world)
    int            n_pes_;     // TP group size  (nvshmem_team_n_pes(tp_team_))
    int            my_pe_;     // this GPU's pe-in-team (nvshmem_team_my_pe(tp_team_))

    __device__ __forceinline__ int my_pe() const { return my_pe_; }
    __device__ __forceinline__ int n_pes() const { return n_pes_; }

    __device__ __forceinline__ float* local(int64_t off) const {
        return heap_base + off;
    }
    __device__ __forceinline__ const float* peer(int64_t off, int pe_in_team) const {
        // peer() is called with the TEAM-LOCAL pe index (ascending 0..n_pes()-1,
        // the fixed reduce order); map it to the GLOBAL pe nvshmem_ptr needs.
        const int global_pe = nvshmem_team_translate_pe(tp_team_, pe_in_team,
                                                         NVSHMEM_TEAM_WORLD);
        return static_cast<const float*>(nvshmem_ptr(heap_base, global_pe)) + off;
    }

    __device__ __forceinline__ void rendezvous(const ::sg::fused::GridBarrier& bar) const {
        bar.sync();                       // all CTAs of THIS GPU arrived
        if (blockIdx.x == 0) {            // one CTA crosses the GPU boundary
            // Drain THIS PE's outstanding symmetric stores so the ascending-pe
            // NVLink loads on every peer observe our published partial, THEN
            // join the TP-group barrier (team-scoped — not the whole world).
            nvshmem_quiet();
            nvshmemx_barrier_block(tp_team_);
        }
        bar.sync();                       // release every CTA after the cross-GPU join
    }
};
#endif  // SG_HAS_NVSHMEM

// ─────────────────────────────────────────────────────────────────────────
//  tp_allreduce_sum_fixed_order — THE all-reduce both transports share.
//
//  PRE: every PE has PUBLISHED its fp32 partial at symmetric offset `slot_off`
//  (length n) and the caller has crossed `tr.rendezvous(bar)` once (so all
//  publishes are visible). POST: `dst[i] = Σ_pe partial_pe[i]`, summed in
//  ASCENDING pe order in fp32 — bit-identical on every PE and across reruns
//  (design §5.2/§2.7: structural order, the A/A/A non-negotiable). The caller
//  crosses `tr.rendezvous(bar)` again before REUSING the slot.
//
//  `tid`/`nthreads` define this PE's cooperative element loop. In the loopback
//  every virtual PE redundantly computes its own copy (each PE strides its own
//  CTA group's threads) — exactly what real per-GPU ranks do.
//
//  This is deliberately NOT nvshmem_float_sum_reduce: that collective's
//  reduction order is unspecified ⇒ ULP drift run-to-run ⇒ A/A/A failure
//  (design §5.2). The hand-rolled ascending-pe loop IS the deliverable.
// ─────────────────────────────────────────────────────────────────────────
template <class Transport>
__device__ __forceinline__ void tp_allreduce_sum_fixed_order(
        const Transport& tr, int64_t slot_off, float* __restrict__ dst,
        int64_t n, int tid, int nthreads) {
    const int P = tr.n_pes();
    for (int64_t i = tid; i < n; i += nthreads) {
        float acc = 0.0f;
        #pragma unroll 1
        for (int pe = 0; pe < P; ++pe) {        // ASCENDING pe — fixed order
            acc += tr.peer(slot_off, pe)[i];
        }
        dst[i] = acc;
    }
}

// Variant writing the reduced value back INTO this PE's symmetric slot (in-place
// consumers); same fixed order. Safe because every PE reads ALL peers before any
// PE can pass the post-reduce rendezvous the caller crosses before reuse — the
// read-before-overwrite hazard is what that second rendezvous exists for, so
// this variant must write to a DIFFERENT slot than it reads. Kept explicit:
template <class Transport>
__device__ __forceinline__ void tp_allreduce_sum_fixed_order_to_slot(
        const Transport& tr, int64_t src_slot_off, int64_t dst_slot_off,
        int64_t n, int tid, int nthreads) {
    float* dst = tr.local(dst_slot_off);
    tp_allreduce_sum_fixed_order(tr, src_slot_off, dst, n, tid, nthreads);
}

// ─────────────────────────────────────────────────────────────────────────
//  make_transport_from_comm<Par> — the ONE place that selects the transport
//  from a populated par::CommCtx, so the megakernel constructs `tr` with a
//  single call (no #if clutter at the call site). Returns:
//    * NvshmemTransport       when -DSG_HAS_NVSHMEM (the real 8×H100 path),
//    * LoopbackTransport      otherwise (the single-GPU honest simulation).
//  Both read the SAME CommCtx fields (sym-heap base, stride, team pe/size), so
//  the launcher populates ONE struct and either transport binds it. Only ever
//  called under `if constexpr (Par::kTPComm)` (folded away on SingleGPU), so it
//  is never instantiated on the byte-identical default path.
//
//  The CommCtx type is forward-declared opaquely here to avoid pulling
//  parallel_config.cuh's include into every tp_transport.cuh consumer; the real
//  definition (par::CommCtx) is included by the megakernel before this is used.
//  The fields read are POD (float*/int64_t/int) — no NVSHMEM type crosses the
//  CommCtx boundary (the team handle is a void* cast to nvshmem_team_t here).
// ─────────────────────────────────────────────────────────────────────────
template <class Par, class CommT>
__device__ __forceinline__ auto make_transport_from_comm(const CommT& comm) {
#if defined(SG_HAS_NVSHMEM)
    return NvshmemTransport{
        reinterpret_cast<float*>(comm.tp_sym_heap),
        static_cast<nvshmem_team_t>(reinterpret_cast<intptr_t>(comm.tp_comm_handle)),
        comm.tp_team_n_pes, comm.tp_team_local_pe };
#else
    return LoopbackTransport{
        reinterpret_cast<float*>(comm.tp_sym_heap), comm.tp_heap_stride_floats,
        comm.tp_team_n_pes, comm.tp_team_local_pe };
#endif
}

}}}}  // namespace sg::fused::sm90::tp

#endif  // SG_FUSED_SM90_TP_TRANSPORT_CUH_
