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
//  ParConfig — the compile-time ADAPTIVE 3D–5D(+ZeRO) parallelism point.
//
//  The (DP, TP, PP, SP, EP) tuple + ZeRO maps 1:1 to the Python-side
//  ParallelConfig (distributed.py: data/tensor/pipeline/expert_parallel +
//  zero_stage) per design §1.3. ALL fields constexpr ⇒ every consumer branch
//  folds; the degenerate point emits zero comm code (design §1.2).
//
//  ADAPTIVE DEGREE (the auto-3D–5D contract): the FRONT-END picks the degree and
//  instantiates the matching point (grokking_optimizers.parallel.auto_config,
//  /workspace/impl_diffs/adaptive_parallelism.md §3):
//    * base 3D  = DP × TP × PP                (every model);
//    * +SP (4th) iff the model is a SEQUENCE model (decoder / ViT-patches /
//      Mamba) — EXPRESSIBLE but pinned 1 this campaign (seq 4-17 makes a split
//      moot; the static_assert below is the loud gate);
//    * +EP (5th) iff the model declares EXPERTS (a model-level MoE, num_experts>1).
//      EP is the 6th TEMPLATE PARAM and is DEFAULTED to 1 so EVERY existing
//      5-arg ParConfig<DP,TP,PP,SP,Z> instantiation (the SingleGPU alias, the
//      §7.2 allow-list, the megakernel's Par=SingleGPU default) is BYTE-IDENTICAL
//      — EP=1 ⇒ kEPComm==false ⇒ every expert-dispatch branch folds to ZERO code
//      (the design §1.2 PTX-diff gate, extended to EP). EP is positioned AFTER Z
//      precisely so the trailing default preserves every current call site.
//
//  SP (sequence-parallel) is EXPRESSIBLE but pinned to 1 this campaign: at the
//  race's seq 4-17 a sequence split is moot (design §1.1 / PARALLELISM-FINAL),
//  so the static_assert below makes any SP!=1 instantiation a loud BUILD error
//  rather than a silently-broken path. EP is gated the SAME way TP is (a runtime-
//  degree axis, no fixed-to-1 assert) so a future MoE model instantiates EP>1
//  cleanly; the current dense roster never does (it has no model experts), so EP
//  stays 1 and folds away — honest inertness, not a stub that pretends to run.
// ─────────────────────────────────────────────────────────────────────────
template <int DP, int TP, int PP, int SP, ZeROStage Z, int EP = 1>
struct ParConfig {
    static constexpr int        kDP   = DP;   // data-parallel replicas
    static constexpr int        kTP   = TP;   // tensor-parallel ranks (Megatron col/row split)
    static constexpr int        kPP   = PP;   // pipeline stages
    static constexpr int        kSP   = SP;   // sequence-parallel (kept EXPRESSIBLE, fixed 1)
    static constexpr int        kEP   = EP;   // expert-parallel ranks (MoE; DEFAULT 1 = dense)
    static constexpr ZeROStage  kZeRO = Z;

    // ── Derived compile-time gates (design §1.1). These are the predicates the
    //    megakernel + sharded-opt kernel branch on with `if constexpr`. ────────
    //    EP folds into kIsSingleGPU/kEmitComm so an EP>1 point is (correctly) a
    //    multi-GPU point (kEmitComm==true) and a dense EP==1 point with all other
    //    axes 1 is still the byte-identical SingleGPU (kIsSingleGPU==true).
    static constexpr bool kIsSingleGPU = (DP == 1 && TP == 1 && PP == 1 && SP == 1 && EP == 1);
    static constexpr bool kEmitComm     = !kIsSingleGPU;        // gate ALL NVSHMEM/NCCL
    static constexpr bool kShardParams  = (Z == ZeROStage::Z3);  // ZeRO-3 param residency shard
    static constexpr bool kShardOptGrad = (Z >= ZeROStage::Z2);  // ZeRO>=2 grad+opt-state shard
                                                                 // ⇒ kernel early-exits at B2 (§2.2)
    static constexpr bool kTPComm       = (TP > 1);             // in-kernel TP all-reduce (§5)
    static constexpr bool kPPStage      = (PP > 1);             // pipeline stage cut (§4)
    static constexpr bool kEPComm       = (EP > 1);             // in-kernel expert all-to-all (§2.4)
                                                                 // — folds to ZERO when EP==1 (dense)

    // SP is expressible but must be 1 this campaign (design §1.1 static_assert).
    static_assert(SP == 1, "SP axis is expressible but must be 1 this campaign "
                           "(seq 4-17 makes a seq split moot; #14 / PARALLELISM-FINAL).");
    static_assert(DP >= 1 && TP >= 1 && PP >= 1 && SP >= 1 && EP >= 1, "degrees must be >= 1");
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
//  SizeConfig — the COMPILE-TIME *workload-size* point, the SECOND specialization
//  axis the megakernel is templated over (alongside ParConfig). It is the SAME
//  if-constexpr-on-config mechanism that gates the distributed all-reduce
//  (if constexpr (Par::kTPComm)), now keyed on workload SIZE rather than
//  parallelism: a LARGE config folds in CTA-TILING (occupancy>1 / clusters — more
//  CTAs per output tile to fill the SMs and kill the ~20% grid-barrier idle that
//  cross-CTA load imbalance causes at d=2048, bottleneck LEVER 2); a SMALL config
//  folds in NONE of it (the persistent 1-CTA/SM shape wins — overhead dominates).
//
//  ALL fields `static constexpr` ⇒ every consumer branch (`if constexpr
//  (Sz::kCtaTile)`, `if constexpr (Sz::kClusterDim > 1)`, …) folds at compile time.
//  The degenerate `SizeSmall` point sets every predicate to the SHIPPED value, so
//  `fused_decoder_megakernel_tc<Opt, Par, SizeSmall>` is byte-for-byte the pre-Sz
//  kernel — the PTX-diff gate (test_decoder_tc.py). This mirrors ParConfig's §1.2
//  byte-identical-when-degenerate contract exactly.
//
//  The selector that PICKS a SizeConfig from (d, layers, T, n_sms) lives host-side
//  in grokking_optimizers/megakernel_codegen.py::decoder_knobs_for_size (it also
//  emits the matching -DSG_TUNED_* tile knobs), keeping ONE source of truth for the
//  size→knobs map the same way ParallelConfig (distributed.py) is the source for
//  the parallelism degrees. This header carries ONLY the compile-time point + the
//  predicates the kernel branches on; it adds NO math and NO launch policy.
//
//  CtaTile           : the master gate — emit the CTA-tiled (occupancy>1) body
//                      (false on SizeSmall ⇒ the 1-CTA/SM persistent shape).
//  CtasPerTile       : how many CTAs cooperate on ONE output tile when CtaTile
//                      (1 on SizeSmall; e.g. 2 on SizeLarge). Drives the split of
//                      the per-tile GEMM N-range across cooperating CTAs (§7).
//  ClusterDim        : the Hopper thread-block-cluster size (1 = no cluster). A
//                      cluster lets cooperating CTAs share a distributed-smem tile
//                      + a cluster barrier instead of the grid barrier (§7); 1 on
//                      SizeSmall keeps the non-cluster launch byte-identical.
//  TileN             : the wgmma N-tile the body sizes from. Defaults to the
//                      SG_TUNED_TILE_N macro so SizeSmall == the shipped tile; a
//                      large tier may pin a different TileN as a constexpr (the
//                      kernel reads Sz::kTileN where it reads SG_TUNED_TILE_N
//                      today — see EDIT C note; on SizeSmall they are EQUAL so the
//                      substitution is byte-identical).
// ─────────────────────────────────────────────────────────────────────────
template <bool CtaTile, int CtasPerTile, int ClusterDim, int TileN>
struct SizeConfig {
    static constexpr bool kCtaTile     = CtaTile;     // master gate: occupancy>1 body
    static constexpr int  kCtasPerTile = CtasPerTile; // CTAs cooperating per output tile
    static constexpr int  kClusterDim  = ClusterDim;  // Hopper cluster size (1 = none)
    static constexpr int  kTileN       = TileN;       // wgmma N-tile the body sizes from

    // Derived gate: a cluster launch is only meaningful when CTA-tiling is on AND
    // the cluster dim > 1. SizeSmall ⇒ false ⇒ no cluster launch attribute emitted.
    static constexpr bool kUseCluster = (CtaTile && ClusterDim > 1);

    static_assert(CtasPerTile >= 1, "CtasPerTile must be >= 1");
    static_assert(ClusterDim  >= 1, "ClusterDim must be >= 1 (1 = no cluster)");
    static_assert(TileN       >= 1, "TileN must be >= 1");
    // CtaTile==false MUST be the degenerate (byte-identical) point: no cooperation,
    // no cluster. Guard it so a SizeSmall-shaped config can never silently request
    // tiling without flipping the master gate (the §1.2-style invariant).
    static_assert(CtaTile || (CtasPerTile == 1 && ClusterDim == 1),
                  "SizeConfig: CtaTile==false is the degenerate 1-CTA/SM point and "
                  "MUST have CtasPerTile==1 && ClusterDim==1 (byte-identical default).");
};

// ─────────────────────────────────────────────────────────────────────────
//  SizeSmall — the DEGENERATE size point (the analogue of SingleGPU). CtaTile OFF,
//  1 CTA/tile, no cluster, TileN == the shipped SG_TUNED_TILE_N. This is the
//  DEFAULT template arg of the megakernel ⇒ every existing call site compiles
//  unchanged AND `<Opt, Par, SizeSmall>` is byte-for-byte the legacy kernel.
//
//  SizeSmall reads SG_TUNED_TILE_N so the size point inherits whatever tile the
//  autotuner pinned for the small tier; if SG_TUNED_TILE_N is unset it is the
//  in-header #ifndef default (128), matching model_stage_decoder_tc.cuh.
// ─────────────────────────────────────────────────────────────────────────
#ifndef SG_TUNED_TILE_N
#define SG_TUNED_TILE_N 128   // keep in sync with model_stage_decoder_tc.cuh #ifndef
#endif
using SizeSmall = SizeConfig</*CtaTile=*/false, /*CtasPerTile=*/1,
                             /*ClusterDim=*/1,   /*TileN=*/SG_TUNED_TILE_N>;

// SizeLarge — the LARGE tier point: CTA-tiling ON, 2 CTAs per output tile, a 2-CTA
// Hopper cluster, the shipped TileN. This is the point the launcher selects for the
// d=2048 bench / flagship-class workloads (the codegen selector decides WHEN; see
// megakernel_codegen.decoder_knobs_for_size). It only ever instantiates the §7
// CTA-tiled body — NOT compiled until §7 lands, but the alias is defined now so the
// dispatch allow-list + the autotuner can name it. (Until §7, the kernel's
// `if constexpr (Sz::kCtaTile)` arm is the NO-OP stub of EDIT C.3 — instantiating
// SizeLarge compiles to the same body as SizeSmall, just reachable for wiring.)
using SizeLarge = SizeConfig</*CtaTile=*/true, /*CtasPerTile=*/2,
                             /*ClusterDim=*/2,  /*TileN=*/SG_TUNED_TILE_N>;

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
    // ── In-kernel EXPERT-PARALLEL (MoE) wiring — the EP analogue of the TP block,
    //    filled ONLY when kEPComm (EP>1); nullptr/0 on every dense path. Mirrors
    //    the TP fields EXACTLY so a future expert all-to-all/dispatch reads its
    //    own symmetric heap + team the SAME way the TP reduce reads the TP team
    //    (/workspace/impl_diffs/adaptive_parallelism.md §2.4). A default-constructed
    //    CommCtx forwards "no EP" and the kEPComm=false megakernel never reads
    //    these — so the dense (EP==1) ABI is byte-identical (the EP PTX gate).
    //  * ep_comm_handle: the NVSHMEM EP team id stored as
    //    reinterpret_cast<void*>((intptr_t)nvshmem_team_t) (int32 team), reversed
    //    by a future ep::make_transport_from_comm — the SAME opaque-void* trick
    //    tp_comm_handle uses so this header needs no <nvshmemx.h>.
    //  * ep_sym_heap / ep_heap_stride_floats: the symmetric base + per-PE stride
    //    for the expert dispatch/combine token buffers (the all-to-all operands).
    //  * ep_team_local_pe / ep_team_n_pes: the EP-team-local pe + team size
    //    (== nvshmem_team_my_pe / _n_pes on the EP team; == ep_rank / ep_size).
    //    The EP team is the set of DP peers that together hold one full expert set
    //    (distributed.py _build_mesh ep_ranks — EP sub-divides DP, never enlarges
    //    world_size), so on the dense roster ep_team_n_pes==1 and this is inert.
    void*   ep_comm_handle        = nullptr;  // NCCL comm / NVSHMEM EP team id (§2.4)
    void*   ep_sym_heap           = nullptr;  // nvshmem_malloc'd symmetric EP token base
    int64_t ep_heap_stride_floats = 0;        // per-PE symmetric stride (floats)
    int     ep_team_local_pe      = 0;        // pe-in-EP-team (== ep_rank)
    int     ep_team_n_pes         = 1;        // EP team size  (== ep_size)
};

}}}  // namespace sg::fused::par

#endif  // SG_FUSED_SM90_PARALLEL_CONFIG_CUH_
