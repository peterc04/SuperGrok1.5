#ifndef SG_FUSED_SM90_PARALLEL_CONFIG_CUH_
#define SG_FUSED_SM90_PARALLEL_CONFIG_CUH_
// ============================================================================
// csrc/fused/sm_90/parallel_config.cuh — the COMPILE-TIME parallelism point
// (the metaprogrammed template the L3-TC megakernel is specialized over).
//
// IMPLEMENTS: /workspace/.parallelism_design.md §1.1 (the ParConfig<DP,TP,PP,
// SP,ZeROStage> struct + the SingleGPU alias) and §1.2 (the bit-identical
// single-GPU contract — the kEmitComm gate that makes every comm branch fold
// away to ZERO code when the config is degenerate).
//
// HEADER-ONLY, SELF-CONTAINED, device+host. Included by every megakernel TU
// (decoder/vit/mamba) so the SAME translation unit can hold BOTH the single-GPU
// point AND a multi-GPU flagship point in one fat binary (design §1.2 — the
// discriminator is a TEMPLATE PARAMETER, not a TU-wide macro, because the
// campaign ships the single-GPU artifact alongside the 4D+ZeRO-3 flagship).
//
// WHY constexpr-everything: ALL fields are `static constexpr`, so every branch
// that reads them (`if constexpr (Par::kEmitComm)`, `if constexpr (Par::
// kShardOptGrad)`, …) folds at compile time. For the degenerate SingleGPU point
// the compiler emits NO NVSHMEM symbols, NO extra barriers, NO `comm` reads ⇒
// `fused_decoder_megakernel_tc<Opt, SingleGPU>` is byte-for-byte identical to
// today's `fused_decoder_megakernel_tc<Opt>` (design §1.2, the PTX-diff gate).
//
// THIS FILE ADDS NO MATH AND NO COMM. It is the pure compile-time config carrier
// + an empty CommCtx POD (the megakernel's default `comm` argument; populated
// with NCCL/NVSHMEM handles ONLY on the kEmitComm path, which a kernel builder
// wires later — out of scope here). Authored 1-GPU/CPU per design §7.
// ============================================================================

#include <cstdint>

namespace sg { namespace fused { namespace par {

// ─────────────────────────────────────────────────────────────────────────
//  ZeRO stage (design §1.1). Z0=none, Z1=opt-state shard, Z2=+grad shard,
//  Z3=+param shard. The flagship ships Z3 (design DELIVERABLE line); Z2 is the
//  SAME code path with param-sharding off (a stage flag), the first bring-up
//  increment (design §3.1).
// ─────────────────────────────────────────────────────────────────────────
enum class ZeROStage : int { Z0 = 0, Z1 = 1, Z2 = 2, Z3 = 3 };

// ─────────────────────────────────────────────────────────────────────────
//  ParConfig — the compile-time 4D(+ZeRO) parallelism point (design §1.1).
//
//  The (DP, TP, PP, SP, ZeRO) tuple maps 1:1 to the Python-side ParallelConfig
//  (distributed.py) per design §1.3. ALL fields constexpr ⇒ every consumer
//  branch folds; the degenerate point emits zero comm code (design §1.2).
//
//  SP (sequence-parallel) is EXPRESSIBLE but pinned to 1 this campaign: at the
//  race's seq 4-17 a sequence split is moot (design §1.1 / PARALLELISM-FINAL),
//  so the static_assert below makes any SP!=1 instantiation a loud BUILD error
//  rather than a silently-broken path.
// ─────────────────────────────────────────────────────────────────────────
template <int DP, int TP, int PP, int SP, ZeROStage Z>
struct ParConfig {
    static constexpr int        kDP   = DP;   // data-parallel replicas
    static constexpr int        kTP   = TP;   // tensor-parallel ranks (Megatron col/row split)
    static constexpr int        kPP   = PP;   // pipeline stages
    static constexpr int        kSP   = SP;   // sequence-parallel (kept EXPRESSIBLE, fixed 1)
    static constexpr ZeROStage  kZeRO = Z;

    // ── Derived compile-time gates (design §1.1). These are the predicates the
    //    megakernel + sharded-opt kernel branch on with `if constexpr`. ────────
    static constexpr bool kIsSingleGPU = (DP == 1 && TP == 1 && PP == 1 && SP == 1);
    static constexpr bool kEmitComm     = !kIsSingleGPU;        // gate ALL NVSHMEM/NCCL
    static constexpr bool kShardParams  = (Z == ZeROStage::Z3);  // ZeRO-3 param residency shard
    static constexpr bool kShardOptGrad = (Z >= ZeROStage::Z2);  // ZeRO>=2 grad+opt-state shard
                                                                 // ⇒ kernel early-exits at B2 (§2.2)
    static constexpr bool kTPComm       = (TP > 1);             // in-kernel TP all-reduce (§5)
    static constexpr bool kPPStage      = (PP > 1);             // pipeline stage cut (§4)

    // SP is expressible but must be 1 this campaign (design §1.1 static_assert).
    static_assert(SP == 1, "SP axis is expressible but must be 1 this campaign "
                           "(seq 4-17 makes a seq split moot; #14 / PARALLELISM-FINAL).");
    static_assert(DP >= 1 && TP >= 1 && PP >= 1 && SP >= 1, "degrees must be >= 1");
};

// ─────────────────────────────────────────────────────────────────────────
//  THE single-GPU guarantee, named once so the static_asserts read cleanly
//  (design §1.1). `fused_decoder_megakernel_tc<Opt, SingleGPU>` MUST be
//  byte-identical to the legacy `<Opt>` overload — enforced by kEmitComm==false
//  folding every comm branch away (design §1.2). This is the default template
//  arg of the megakernel, so existing call sites compile unchanged.
// ─────────────────────────────────────────────────────────────────────────
using SingleGPU = ParConfig<1, 1, 1, 1, ZeROStage::Z0>;

// ─────────────────────────────────────────────────────────────────────────
//  CommCtx — the megakernel's default trailing argument (design §1.1 kernel
//  signature: `par::CommCtx comm /* default-constructed, unused when
//  !kEmitComm */`). It is an EMPTY POD here on purpose:
//
//   * On the SingleGPU / kEmitComm==false path the kernel never reads it, so the
//     ABI of the `<Opt>`-only overload is preserved (the old non-Par launcher
//     forwards a default-constructed CommCtx) and the PTX is unchanged.
//   * On the kEmitComm==true path a kernel builder widens this POD with the
//     NVSHMEM team handle / NCCL comm / tp_group rank table / peer param-shard
//     base pointers the in-kernel collectives need (design §5.2/§6.1). That
//     widening is GPU-window work and is intentionally NOT authored here (this
//     file is 1-GPU/CPU pre-work per design §7); the empty struct is the stable
//     seam those fields hang off without touching any single-GPU call site.
//
//  Trivially constructible/copyable so it is a valid by-value kernel parameter
//  and a no-cost default argument.
// ─────────────────────────────────────────────────────────────────────────
struct CommCtx {
    // rank within the world / the per-axis groups (filled only when kEmitComm).
    // Defaults describe the degenerate single-GPU world so a default-constructed
    // CommCtx is the correct "no comm" value the single-GPU path forwards.
    int world_size = 1;
    int world_rank = 0;
    int dp_size = 1, dp_rank = 0;
    int tp_size = 1, tp_rank = 0;
    int pp_size = 1, pp_rank = 0;
    // Opaque handle slots (NVSHMEM team / NCCL comm) — left as nullptr-able
    // pointers a builder casts to the concrete handle type behind kEmitComm, so
    // this header need NOT include <nccl.h>/<nvshmem.h> (kept CPU-compilable).
    // tp_comm_handle carries the NVSHMEM TP team id stored as
    // reinterpret_cast<void*>((intptr_t)nvshmem_team_t) (tp_team_t is int32_t);
    // tp::make_transport_from_comm reverses the cast (tp_kernel.md §3 A.2).
    void* tp_comm_handle = nullptr;   // NCCL comm / NVSHMEM TP team id (§5)
    void* dp_comm_handle = nullptr;   // NCCL comm for the DP group (reduce-scatter / all-gather)
    // ── In-kernel TP all-reduce wiring (filled ONLY on kEmitComm; nullptr/0 on
    //    the SingleGPU path, so a default-constructed CommCtx forwards "no TP
    //    heap" and the kEmitComm=false megakernel never reads these — the ABI of
    //    the default <Opt,SingleGPU> instantiation is preserved, the §6 PTX gate).
    //  * tp_sym_heap: the nvshmem_malloc'd SYMMETRIC base for the TP reduce slots
    //    (NOT the cudaMalloc workspace — /workspace/impl_diffs/tp_kernel.md §2/EDIT E).
    //    Opaque float* here; NvshmemTransport reinterprets it as heap_base. On the
    //    loopback build it is the strided cudaMalloc heap (LoopbackTransport).
    //  * tp_heap_stride_floats: per-PE symmetric stride (== tp::tp_heap_stride_floats).
    //  * tp_team_local_pe / tp_team_n_pes: the team-local pe index + team size
    //    (== nvshmem_team_my_pe / _n_pes on the TP team; == tp_rank / tp_size).
    void*   tp_sym_heap           = nullptr;  // nvshmem_malloc'd symmetric TP-slot base
    int64_t tp_heap_stride_floats = 0;        // per-PE symmetric stride (floats)
    int     tp_team_local_pe      = 0;        // pe-in-TP-team (== tp_rank)
    int     tp_team_n_pes         = 1;        // TP team size  (== tp_size)
};

}}}  // namespace sg::fused::par

#endif  // SG_FUSED_SM90_PARALLEL_CONFIG_CUH_
