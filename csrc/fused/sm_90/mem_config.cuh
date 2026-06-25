#ifndef SG_FUSED_SM90_MEM_CONFIG_CUH_
#define SG_FUSED_SM90_MEM_CONFIG_CUH_
// csrc/fused/sm_90/mem_config.cuh — the MEMORY-STRATEGY compile-time descriptor.
//
// Sibling of parallel_config.cuh's ParConfig: where ParConfig describes the
// PARALLELISM axes (DP/TP/PP/SP/ZeRO), MemConfig describes the per-rank MEMORY
// strategy (optimizer host-offload | activation recompute | layer streaming). It is
// templated on three independent bools + a ring depth, ALL `static constexpr`, so
// every consumer branch (`if constexpr (Mem::kRecomputeActs)`, …) folds at compile
// time. The DEFAULT InHbm point sets every gate false ⇒ the megakernel is
// BYTE-IDENTICAL to the shipped in-HBM build (the same guarantee SingleGPU gives for
// ParConfig — parallel_config.cuh:18-22,80-86). The chosen point comes from the
// Python planner (mem_strategy.plan_memory_strategy -> gate_macros) via -D macros, so
// the host fit-decision and the emitted machinery are ONE source of truth.
//
// SCOPING NOTE: like CommCtx (parallel_config.cuh:96-101), the staging POD that the
// ON paths need (pinned-host base pointers, cudaMemcpyAsync stream handles, the ring
// slot table) is a GPU-window widening hung off MemRuntime below; this header authors
// only the stable seam + the constexpr gates, not the device-window bodies.

#include <cstdint>

namespace sg { namespace fused { namespace mem {

template <bool OffloadOpt, bool RecomputeActs, bool StreamLayers, int StreamDepth>
struct MemConfig {
    static constexpr bool kOffloadOpt    = OffloadOpt;     // optimizer state in pinned host RAM, staged
    static constexpr bool kRecomputeActs = RecomputeActs;  // checkpoint layer boundaries; recompute fwd in bwd
    static constexpr bool kStreamLayers  = StreamLayers;   // weights in pinned host RAM, ring of kStreamDepth
    static constexpr int  kStreamDepth   = StreamDepth;    // resident layers when streaming (>=2)
    static constexpr bool kAnyOffHbm     = OffloadOpt || RecomputeActs || StreamLayers;
    static_assert(!StreamLayers || StreamDepth >= 2,
                  "layer streaming needs a ring depth >= 2 (one compute, one prefetch)");
};

// The byte-identical default: every strategy OFF. The megakernel's default Mem arg, so
// every existing call site compiles unchanged (mirrors parallel_config.cuh's SingleGPU).
using InHbm = MemConfig<false, false, false, 0>;

// Runtime staging seam (filled ONLY when a gate is ON; all-null on the InHbm path so the
// kAnyOffHbm==false kernel never reads it — the ABI of the default instantiation is
// preserved, the same PTX-gate discipline as CommCtx). Empty-by-default POD.
struct MemRuntime {
    // OPT-OFFLOAD: pinned-host optimizer state base + the staging stream/event handles.
    void*   host_state_base   = nullptr;   // cudaHostAlloc'd [k*total] fp32 (pinned)
    void*   stage_stream      = nullptr;   // cudaStream_t for async H2D/D2H of state tiles
    int64_t stage_tile_floats = 0;         // per-tile stage size (one tensor-group)
    // LAYER-STREAM: pinned-host weight base + the ring of device slots.
    void*   host_param_base   = nullptr;   // cudaHostAlloc'd [total] fp32 (pinned)
    void*   ring_slots        = nullptr;   // device [kStreamDepth] layer-weight slots
    void*   prefetch_stream   = nullptr;   // cudaStream_t for layer prefetch
};

}}}  // namespace sg::fused::mem
#endif  // SG_FUSED_SM90_MEM_CONFIG_CUH_
