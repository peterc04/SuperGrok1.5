# adaptive_parallelism — APPLY-READY edits: ADAPTIVE 3D–5D mesh (EP as the 5th axis on `ParConfig` + the front-end→`ParConfig` inference)

AREA: `csrc/fused/sm_90/parallel_config.cuh` + `grokking_optimizers/parallel/*`
(new `auto_config.py`) + the front-end model config in `grokking_race_v2.py`
(`MODEL_SCALES_BY_MODEL` + the model definitions).

READ-ONLY analysis of `/workspace/SuperGrok1.5`; this file is APPLY-READY: every
edit has a VERBATIM OLD snippet copied from the LIVE file plus a NEW replacement,
and new files are given in full. I read in full this session, before writing one
line: `csrc/fused/sm_90/parallel_config.cuh`, `csrc/fused/sm_90/tp_transport.cuh`,
`grokking_optimizers/distributed.py`, `grokking_optimizers/parallel/__init__.py`,
`grokking_optimizers/parallel/shard_map.py`, `grokking_optimizers/parallel/distributed_step.py` (head),
`grokking_race_v2.py` (MODEL_SCALES / MODEL_SCALES_BY_MODEL / DEFAULT_CONFIG / the
DecoderBlock/Transformer/ViT/Mamba defs / `_raw_model` / `train_supergrok2`),
`tests/hw/test_parallel_instantiation.py`, `tests/hw/test_3d_parallel.py`,
`tests/hw/test_tp_loopback.py`, and the three cited specs IN FULL
(`tp_kernel.md`, `dist_step.md`, `run_harness.md`).

---

## §0 — THE VERIFIED LIVE STATE (what is already applied — do NOT re-author it)

The codebase has moved PAST the apply-now state several specs describe. Confirmed
by reading the live files this session:

| fact | live evidence |
|---|---|
| The decoder megakernel IS templated `<OptId Opt, class Par=par::SingleGPU>` with a trailing `CommCtx comm={}`. | `fused_decoder_megakernel.cuh:674-681` |
| `ParConfig` is the 5-template-param `<int DP,int TP,int PP,int SP,ZeROStage Z>` (a 4D mesh + ZeRO; SP pinned 1). | `parallel_config.cuh:55` |
| `CommCtx` is ALREADY widened with the TP-reduce fields (`tp_sym_heap`, `tp_heap_stride_floats`, `tp_team_local_pe`, `tp_team_n_pes`, `tp_comm_handle`). | `parallel_config.cuh:106-138` |
| `tp_transport.cuh` ALREADY has the hardened `NvshmemTransport` (team-scoped `nvshmemx_barrier_block` + `nvshmem_quiet`) AND `make_transport_from_comm<Par>`. | `tp_transport.cuh:168-277` |
| The Python `distributed.py` ALREADY has FULL EP plumbing: `ParallelConfig.expert_parallel`, the `ep` alias, the `data_parallel % expert_parallel` validation, `_RankMesh.ep_ranks`, `_build_mesh` EP-block math, `_init_groups` EP `new_group` loop, `ep_group`/`ep_rank`/`ep_world_size` accessors, and `all_reduce_ep`. | `distributed.py:64-128,228-292,403-418,491-555` |
| EVERY `ParConfig<...>` instantiation in the tree uses EXACTLY 5 args (`<DP,TP,PP,SP,Z>`): the `SingleGPU` alias, `test_parallel_instantiation.py`'s allow-list, and the SP!=1 negative test. | `parallel_config.cuh:86`, `test_parallel_instantiation.py:115,120,197` |
| The 3 race models (decoder / vit / mamba) carry NO model-level MoE. `num_experts`/`expert_*` in `grokking_race_v2.py` are ALL `sg2_*` — the SuperGrok2 OPTIMIZER's PEER meta-net (144 product-key-routed experts INSIDE the optimizer), never a model layer. | `grokking_race_v2.py:1409-1411` (sg2_num_experts), `_raw_model:528-545` (Transformer/ViT/Mamba3 — no experts), `optimizers/supergrok2.py` (PEER routing is the optimizer) |

**The two load-bearing consequences these facts force on this design:**

1. **EP MUST be the 6th template param, defaulted, AFTER `Z`** — `ParConfig<DP,TP,
   PP,SP,Z, int EP=1>`. Adding it BEFORE `Z` (or as a non-defaulted positional)
   would break all five live 5-arg instantiations and the `test_parallel_
   instantiation` allow-list, violating the byte-identical-when-OFF requirement.
   A trailing defaulted `EP=1` makes `SingleGPU`, the allow-list points, and the
   megakernel's `Par=SingleGPU` default ALL resolve to `EP=1` ⇒ `kEPComm==false`
   ⇒ every EP branch folds ⇒ byte-identical. (§2.)

2. **EP is HONESTLY INERT for the current 3-model roster** — no race model declares
   model-level experts, so the front-end inference (§3) will pick `EP=1` for
   decoder/vit/mamba ALWAYS, and the EP kernel branch is never instantiated. EP is
   wired as the FUTURE-MODEL seam (a model that adds a `num_experts>1` MoE layer),
   and the kernel-side expert dispatch body is SCOPED as a flagged follow-on (§2.4)
   — the ParConfig axis + CommCtx fields + team + gate points land now; the
   expert-all-to-all megakernel body is the GPU-window deliverable. This is the
   same honest split tp_kernel.md uses for the TP reduce (seam now, on-silicon body
   later), and it is stated plainly rather than pretending decoder MoE exists.

---

## §1 — THE FIVE EDITS AT A GLANCE

| # | file | edit | apply-able now? | byte-identical when OFF? |
|---|------|------|-----------------|--------------------------|
| A | `parallel_config.cuh` | add `EP` as the 6th (trailing, defaulted) `ParConfig` template param + `kEP`/`kEPComm`; fold EP into `kIsSingleGPU`/`kEmitComm`; add the `EP>=1` assert | YES (header, CPU-compilable; all 5-arg sites unchanged) | YES (`EP=1` default ⇒ `kEPComm=false`) |
| B | `parallel_config.cuh` | widen `CommCtx` with the EP-team wiring (`ep_sym_heap`/`ep_heap_stride_floats`/`ep_team_local_pe`/`ep_team_n_pes`/`ep_comm_handle`), mirroring the TP fields EXACTLY | YES (POD, single-GPU defaults) | YES (default-constructed = inert) |
| C | `grokking_optimizers/parallel/auto_config.py` | NEW: the front-end→ParConfig inference (`infer_parallel_config`) — base 3D `DP×TP×PP`, +SP iff sequence model, +EP iff model has experts; honors 8 H100s + run_harness.md mesh math; maps to the template instantiation the launcher dispatches | YES (pure Python, no torch/GPU; CPU-testable) | N/A (new file) |
| D | `grokking_optimizers/parallel/__init__.py` | export the `auto_config` symbols | YES | N/A |
| E | `tests/hw/test_parallel_instantiation.py` | extend the allow-list with the EP point `ParConfig<…,EP=8>` + EP-gate static_asserts, proving the 6th param compiles AND folds | YES (CPU `nvcc -c`, SKIPs without nvcc) | the existing 5-arg points are UNCHANGED |

EDITS A, B, E are byte-exact against the live file and compile-safe on this box
today (header + CPU nvcc gate). EDIT C/D are pure-Python apply-now. The kernel-side
EP expert-dispatch BODY is the scoped follow-on (§2.4), not authored here.

---

## §2 — EDIT A + B: EP as the 5th axis on `ParConfig` + the EP `CommCtx` wiring

### §2.1 — EDIT A.1: the `ParConfig` template + derived gates

VERBATIM OLD (copied from `csrc/fused/sm_90/parallel_config.cuh` lines 43–77):
```cpp
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
```

NEW (add `EP` as the TRAILING, DEFAULTED 6th param; mirror TP's gate exactly for
EP; fold EP into the single-GPU/emit-comm predicates; add the EP degree assert):
```cpp
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
```

> WHY this is byte-identical when OFF (EDIT A.1 PTX-diff proof): every live
> `ParConfig<...>` site supplies 5 args and now resolves `EP` to the default `1`.
> `kEP==1` ⇒ `kEPComm==false`; `kIsSingleGPU` gains `&& EP==1` which is TRUE for
> the `<1,1,1,1,Z0>` alias (1==1), so `SingleGPU` keeps `kIsSingleGPU==true` and
> `kEmitComm==false` unchanged. No existing derived flag changes value for any
> 5-arg point. The static_assert gains `&& EP>=1` (always true for the default).
> ⇒ `fused_decoder_megakernel_tc<Opt, SingleGPU>` is byte-for-byte unchanged. GATE:
> `test_parallel_instantiation.py` (the 5-arg allow-list must still compile +
> static_assert green) and `test_decoder_tc.py` (SingleGPU PTX identity).

### §2.2 — EDIT B: the EP `CommCtx` wiring (mirror the TP fields EXACTLY)

The TP all-reduce already carries its team via `tp_sym_heap` / `tp_heap_stride_
floats` / `tp_team_local_pe` / `tp_team_n_pes` / `tp_comm_handle`. EP gets the
SAME five fields (renamed `ep_*`) so the expert all-to-all / dispatch reads its
own symmetric heap + team. The fields are POD with single-GPU defaults, so a
default-constructed `CommCtx` is inert (the `kEPComm==false` megakernel never
reads them — the ABI of the default `<Opt,SingleGPU>` instantiation is preserved).

VERBATIM OLD (copied from `csrc/fused/sm_90/parallel_config.cuh` lines 123–138 —
the TP wiring block + the struct close):
```cpp
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
```

NEW (append the EP wiring fields + the `ep_comm_handle` opaque team-id slot, then
close the struct):
```cpp
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
```

> WHY byte-identical when OFF (EDIT B): the new fields are POD with single-GPU
> defaults (`nullptr`/`0`/`1`). A default-constructed `CommCtx` (the megakernel's
> `comm = {}` default arg) sets `ep_team_n_pes=1` ⇒ no EP team. The `kEPComm==false`
> kernel never references any `ep_*` field (the §2.4 dispatch is `if constexpr
> (Par::kEPComm)`'d out), so the dense `<Opt,SingleGPU>` kernel reads zero new
> bytes. `CommCtx` stays trivially-copyable (all POD) ⇒ still a valid by-value
> kernel parameter. The struct grows by 5 fields (32 B), but since no dense path
> READS them, the emitted PTX for the dense kernel is unchanged (the larger
> by-value param is constructed `{}` and its EP bytes are never loaded — confirm
> with the `test_decoder_tc.py` PTX-diff gate, same as the TP-field widening did).

### §2.3 — How the EP gate threads into the megakernel (the gate points, NOW)

EP needs the SAME three seam points TP has, all `if constexpr (Par::kEPComm)`'d so
they fold to zero on the dense default. These are PLACEMENTS the kernel track wires
(mirroring the TP construction at `fused_decoder_megakernel.cuh:700`), specified
here so the seam is unambiguous:

1. **Transport construction (mirror line 700).** Next to the existing
   `auto tr = ::sg::fused::sm90::tp::make_transport_from_comm<Par>(comm);`, a future
   EP transport is built ONLY under the gate:
   ```cpp
   // EP transport — folds to nothing on the dense roster (kEPComm==false). Built
   // ONLY when EP>1; reads the ep_* CommCtx fields (the EP team's symmetric heap).
   if constexpr (Par::kEPComm) {
       auto ep_tr = ::sg::fused::sm90::ep::make_ep_transport_from_comm<Par>(comm);
       (void)ep_tr;  // consumed by the §2.4 dispatch/combine, below
   }
   ```
   (`ep::make_ep_transport_from_comm` is the EP analogue of `tp::make_transport_
   from_comm`, a §2.4 follow-on; it reads `comm.ep_sym_heap`/`ep_heap_stride_
   floats`/`ep_team_n_pes`/`ep_team_local_pe`/`ep_comm_handle` exactly as the TP
   helper reads the tp_* fields. It does not exist yet — see §2.4.)

2. **The MoE layer dispatch/combine (the expert FFN, gated).** Where a dense model
   runs the FF GEMM pair, an EP model would, under `if constexpr (Par::kEPComm)`:
   (a) route tokens to experts (top-k gate), (b) all-to-all dispatch the routed
   tokens to the owning EP rank's symmetric heap, (c) run the local expert FFN on
   the received tokens, (d) all-to-all combine back, (e) weighted-sum by the gate.
   The `else` arm is the literal dense FF — byte-identical. On the current roster
   the `if constexpr` arm is never instantiated (no model declares experts), so
   this is purely the future seam.

3. **The grad/opt path.** Expert weights are EP-sharded (each EP rank owns a
   disjoint expert subset), so their grads need NO cross-EP reduce (each expert's
   grad is local to its owner) — only the GATE/router weights (replicated across
   EP) get a fixed-order `all_reduce` over the EP team, the EP analogue of the TP
   replicated-tensor reduce (`tp_kernel.md §9.4`). Gated `if constexpr (Par::kEPComm)`.

### §2.4 — HONEST SCOPE of the kernel-side expert routing (the follow-on)

What lands NOW (this spec, apply-ready): the `ParConfig` EP axis (A.1) + the EP
`CommCtx` fields (B) + the EP team (already in `distributed.py`'s mesh: `ep_ranks`/
`ep_group`/`all_reduce_ep`) + the gate points (§2.3, specified placements). This is
the complete COMPILE-TIME + HOST-SIDE seam: an EP>1 `ParConfig` point compiles, the
EP team is built, the CommCtx carries the EP wiring, and `if constexpr (Par::kEPComm)`
is the canonical fold.

What is the FLAGGED FOLLOW-ON (NOT authored here — it needs a model with experts +
a GPU build loop, exactly like tp_kernel.md scopes the TP reduce body):
1. **`ep_transport.cuh`** — the `EpTransport` (Loopback + Nvshmem) + `make_ep_
   transport_from_comm<Par>`, the literal EP twin of `tp_transport.cuh`. The
   all-to-all primitive is `ep_dispatch_alltoall` / `ep_combine_alltoall` (token
   shuffle by expert ownership) — fixed-order so A/A/A holds, the SAME determinism
   discipline as `tp_allreduce_sum_fixed_order`.
2. **The expert-FFN megakernel body** — the routed GEMM (variable tokens-per-expert
   ⇒ a capacity-padded or grouped GEMM), inside the `if constexpr (Par::kEPComm)`
   arm of the FF stage. This is the genuinely hard kernel work and is the GPU-window
   deliverable.
3. **A model that declares experts** — none exists in `grokking_race_v2.py` today.
   The front-end inference (§3) returns `EP>1` ONLY when the model config carries
   `num_experts>1` / an `moe`/`is_moe` flag, so until such a model is added EP stays
   1 and ALL of the above is dead/un-instantiated — the byte-identical-when-OFF
   guarantee holds trivially because the EP code path is never reached.

This split is the honest analogue of `tp_kernel.md`'s "the ParConfig/CommCtx/team/
gate now; the in-kernel reduce body as the on-silicon follow-on" — stated plainly
so no one mistakes the seam for a working expert dispatch.

---

## §3 — EDIT C: the FRONT-END → `ParConfig` inference (NEW `auto_config.py`)

A pure-Python function that reads a model config dict (the `grokking_race_v2.py`
config the race builds models from) and PICKS the adaptive degree:
- base 3D = `DP × TP × PP` (every model),
- `+SP` (4th) iff the model is a SEQUENCE model (decoder / ViT-patches / Mamba are
  all sequence-shaped ⇒ SP-eligible; SP is then EXPRESSIBLE but pinned 1 this
  campaign, matching the kernel static_assert — so the returned `sp` is reported as
  *eligible* but the instantiated degree stays 1),
- `+EP` (5th) iff the model has model-level EXPERTS (`num_experts>1` or an MoE flag),

honoring the 8-H100 device count and the run_harness.md mesh math
(`DP·TP·PP == world`, TP=8 the saturation recommendation), and mapping to the
template instantiation string the launcher dispatches.

It lives in `grokking_optimizers/parallel/` (alongside `shard_map.py` /
`distributed_step.py`), is torch-free / GPU-free (CPU-testable like the rest of the
package), and reuses `grokking_optimizers.distributed.ParallelConfig` as the
return-carrier so the degree it picks is the SAME object `DistributedContext.
from_config` consumes (no parallel mesh abstraction).

NEW FILE — `grokking_optimizers/parallel/auto_config.py` (in full):
```python
"""grokking_optimizers/parallel/auto_config.py — the FRONT-END → ParConfig
ADAPTIVE 3D–5D inference.

Reads a model config (the grokking_race_v2 config dict the race builds models
from) and PICKS the parallelism degree, then maps it to the compile-time
ParConfig<DP,TP,PP,SP,Z,EP> template instantiation the launcher dispatches.

THE ADAPTIVE CONTRACT (the auto-3D–5D rule, /workspace/impl_diffs/
adaptive_parallelism.md §3):

  * base 3D  = DP × TP × PP                         (EVERY model);
  * +SP (4th) iff the model is a SEQUENCE model     (decoder / ViT-patches /
               Mamba are sequence-shaped ⇒ SP-eligible). SP is EXPRESSIBLE but
               PINNED to 1 this campaign (the kernel static_assert in
               parallel_config.cuh; seq 4-17 makes a seq split moot), so the
               returned degree carries sp_eligible=True but sp=1 — the 4th axis
               is *unlocked* for a future long-seq model, not silently broken;
  * +EP (5th) iff the model declares model-level EXPERTS (num_experts>1 or an
               moe/is_moe flag). EP sub-divides the DP group (it does NOT enlarge
               world_size — distributed.py: data_parallel % expert_parallel == 0),
               so it is the 5th axis a MoE model engages. The current race roster
               (decoder/vit/mamba) has NO model experts (num_experts there is the
               SuperGrok2 OPTIMIZER's PEER meta-net, never a model layer), so EP
               stays 1 for them and the kernel's kEPComm folds away — honest
               inertness, the byte-identical-when-OFF guarantee.

PURE PYTHON: no torch, no torch.distributed, no GPU — so it imports and unit-tests
on any box (design §7: maximize what's validated on CPU/1-GPU). It returns a
grokking_optimizers.distributed.ParallelConfig (the SAME carrier
DistributedContext.from_config consumes) plus a small AdaptivePlan describing the
chosen degree + the ParConfig<...> template string for the launcher dispatch.

SOURCES (read in full this session):
  * model defs + config keys : grokking_race_v2.py
        (_raw_model: model_type decoder/vit/mamba; DEFAULT_CONFIG; MODEL_SCALES /
         MODEL_SCALES_BY_MODEL flagship; no model-level num_experts on the roster)
  * mesh + EP rules          : grokking_optimizers/distributed.py
        (ParallelConfig.world_size = DP*TP*PP; expert_parallel sub-divides DP)
  * 8-H100 saturation mesh   : /workspace/impl_diffs/run_harness.md §0
        (TP8 × DP1 × PP1 + ZeRO-3 is the recommended flagship config)
  * kernel ParConfig axis    : csrc/fused/sm_90/parallel_config.cuh
        (template <int DP,int TP,int PP,int SP,ZeROStage Z,int EP=1>; SP pinned 1)
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional, Tuple

from grokking_optimizers.distributed import ParallelConfig

# ── The set of model_types that are SEQUENCE models (SP-eligible). Decoder is
#    token-sequence; ViT is a patch-sequence (+ cls token); Mamba is a state-space
#    sequence. All three are sequence-shaped, so all three are SP-eligible (the 4th
#    axis is unlocked for them) — even though SP is pinned 1 this campaign. A
#    non-sequence model (e.g. a pure MLP probe) would be SP-INeligible. ──
_SEQUENCE_MODEL_TYPES = frozenset({"decoder", "vit", "mamba"})

# ── ZeRO stage spelling → the kernel ZeROStage enumerator (parallel_config.cuh). ──
_ZERO_ENUM = {0: "Z0", 1: "Z1", 2: "Z2", 3: "Z3"}

DEFAULT_WORLD = 8  # 8× H100 (the saturation target, run_harness.md §0)


@dataclasses.dataclass(frozen=True)
class AdaptivePlan:
    """The chosen adaptive mesh + the kernel template instantiation it maps to.

    `degree` is 3/4/5 (3D base, +1 if SP engaged, +1 if EP engaged). Note SP is
    pinned 1 this campaign, so `degree` counts SP only as an *eligible* unlock:
    `degree_eligible` is the would-be degree if SP were active; `degree` is the
    EFFECTIVE degree actually instantiated (SP collapses out when sp==1). EP
    counts toward both only when ep>1.
    """

    dp: int
    tp: int
    pp: int
    sp: int                  # EFFECTIVE SP degree instantiated (pinned 1 this campaign)
    ep: int                  # EFFECTIVE EP degree instantiated (1 = dense)
    zero_stage: int
    sp_eligible: bool        # is the model a sequence model (4th axis unlockable)?
    has_experts: bool        # does the model declare a model-level MoE (5th axis)?
    world: int

    @property
    def degree(self) -> int:
        """EFFECTIVE adaptive degree actually instantiated (3, 4, or 5)."""
        d = 3
        if self.sp > 1:
            d += 1
        if self.ep > 1:
            d += 1
        return d

    @property
    def degree_eligible(self) -> int:
        """The degree the model is ELIGIBLE for (counts SP-eligibility even when
        SP is pinned 1) — the 'auto 3D–5D' label the front-end reports."""
        d = 3
        if self.sp_eligible:
            d += 1
        if self.has_experts:
            d += 1
        return d

    def parconfig_template(self) -> str:
        """The C++ ParConfig<...> instantiation the launcher dispatches.

        Always emits the EP arg (even when 1) for an unambiguous 6-arg point; the
        EP=1 form is byte-identical to the legacy 5-arg point (the trailing
        default), so this is safe to hand to the §7.2 allow-list / dispatch.
        """
        z = _ZERO_ENUM[self.zero_stage]
        return (f"::sg::fused::par::ParConfig<{self.dp}, {self.tp}, {self.pp}, "
                f"{self.sp}, ::sg::fused::par::ZeROStage::{z}, {self.ep}>")

    def to_parallel_config(self, **overrides: Any) -> ParallelConfig:
        """Build the distributed.ParallelConfig this plan describes (the carrier
        DistributedContext.from_config consumes). EP rides as expert_parallel
        (it sub-divides DP, never enlarges world)."""
        return ParallelConfig(
            data_parallel=self.dp, tensor_parallel=self.tp,
            pipeline_parallel=self.pp, expert_parallel=self.ep,
            zero_stage=self.zero_stage, **overrides)


def _model_type(cfg: Mapping[str, Any]) -> str:
    return str(cfg.get("model_type", "decoder")).strip().lower()


def is_sequence_model(cfg: Mapping[str, Any]) -> bool:
    """True iff the model is a SEQUENCE model (SP-eligible). Decoder / ViT-patches
    / Mamba are all sequence-shaped (grokking_race_v2._raw_model)."""
    return _model_type(cfg) in _SEQUENCE_MODEL_TYPES


def model_num_experts(cfg: Mapping[str, Any]) -> int:
    """The MODEL-level expert count (the MoE width), or 1 if dense.

    HONEST DISAMBIGUATION (the trap): grokking_race_v2.py's `num_experts` keys are
    ALL `sg2_num_experts` — the SuperGrok2 OPTIMIZER's PEER meta-net experts, NOT a
    model layer. So we read ONLY the MODEL's own MoE keys (`num_experts` /
    `model_num_experts` / a `moe`/`is_moe` flag), and DELIBERATELY ignore any
    `sg2_*` / optimizer key. The current roster has none of the model keys, so this
    returns 1 (dense) for decoder/vit/mamba — EP folds away (kEPComm==false)."""
    # An explicit MoE flag forces experts on (a future model may set is_moe=True
    # and carry its width under model_num_experts).
    flag = cfg.get("is_moe", cfg.get("moe", False))
    for key in ("model_num_experts", "num_experts"):
        # Guard against the optimizer key bleeding in: only honor a top-level
        # MODEL key, never an sg2_-prefixed one (those are never named plainly
        # `num_experts` in the race config, but be explicit).
        if key.startswith("sg2_"):
            continue
        v = cfg.get(key)
        if isinstance(v, int) and v > 1:
            return v
    if flag:
        # MoE declared but no width given → conservative 2 (loud-enough to engage
        # EP; a real MoE model should set model_num_experts explicitly).
        return 2
    return 1


def has_model_experts(cfg: Mapping[str, Any]) -> bool:
    """True iff the model declares a model-level MoE (EP-eligible, the 5th axis)."""
    return model_num_experts(cfg) > 1


def _pick_base_3d(world: int, *, prefer_tp: int = 0) -> Tuple[int, int, int]:
    """Pick (DP, TP, PP) for the base 3D mesh honoring DP·TP·PP == world.

    DEFAULT POLICY (run_harness.md §0): for the flagship single-model saturation
    the recommendation is TP = world (TP8 × DP1 × PP1) — TP spreads ONE model
    across all GPUs and is what shrinks per-rank Nmax so the staged-opt scratch
    fits. `prefer_tp` overrides (0 ⇒ the default TP=world). PP stays 1 (PP is
    owner-locked overhead at this depth — run_harness.md §5 / pipeline.py HONEST
    SCOPE). DP is whatever is left after TP·PP.
    """
    tp = prefer_tp if prefer_tp > 0 else world
    if world % tp != 0:
        raise ValueError(f"prefer_tp={tp} does not divide world={world}")
    pp = 1
    dp = world // (tp * pp)
    return dp, tp, pp


def infer_parallel_config(
    model_cfg: Mapping[str, Any],
    *,
    world: int = DEFAULT_WORLD,
    zero_stage: int = 3,
    prefer_tp: int = 0,
    expert_parallel: int = 0,
) -> AdaptivePlan:
    """Infer the ADAPTIVE 3D–5D mesh from a model config (the auto-rule).

    Parameters
    ----------
    model_cfg : the grokking_race_v2 config dict (model_type + any MoE keys).
    world     : device count (default 8 = 8×H100, run_harness.md §0).
    zero_stage: ZeRO stage (default 3 — the flagship ships ZeRO-3).
    prefer_tp : override the base-3D TP degree (0 ⇒ TP=world, the saturation mesh).
    expert_parallel : override the EP degree (0 ⇒ AUTO: world//... when the model
                has experts, else 1). EP sub-divides DP (distributed.py), so it
                must divide the chosen DP.

    Returns an :class:`AdaptivePlan` with the chosen per-axis degrees + the
    ParConfig<...> template string. The returned plan is what the launcher uses to
    pick the instantiation; build the runtime mesh via plan.to_parallel_config().
    """
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")

    seq_eligible = is_sequence_model(model_cfg)
    experts = has_model_experts(model_cfg)
    n_experts = model_num_experts(model_cfg)

    # ── base 3D = DP × TP × PP ──
    dp, tp, pp = _pick_base_3d(world, prefer_tp=prefer_tp)

    # ── +SP (4th): eligible for sequence models, but PINNED 1 this campaign (the
    #    kernel static_assert). We report eligibility but instantiate sp=1, so the
    #    4th axis is unlocked-but-inert (a future long-seq model flips it on by
    #    relaxing the parallel_config.cuh SP assert). ──
    sp = 1  # EXPRESSIBLE but pinned 1 (parallel_config.cuh SP==1 static_assert)

    # ── +EP (5th): engages ONLY when the model declares experts. EP sub-divides
    #    DP (it must divide DP and never enlarges world — distributed.py). AUTO:
    #    spread the experts over as many DP peers as evenly divide DP (cap at DP),
    #    so on a TP=world / DP=1 mesh EP would be 1 even for a MoE model unless DP
    #    is freed up (lower TP). The caller can force EP via expert_parallel. ──
    if not experts:
        ep = 1
    elif expert_parallel > 0:
        if dp % expert_parallel != 0:
            raise ValueError(
                f"expert_parallel={expert_parallel} must divide DP={dp} "
                f"(EP sub-divides the DP group — distributed.py)")
        ep = expert_parallel
    else:
        # AUTO: use the largest divisor of DP that does not exceed n_experts.
        ep = 1
        for cand in range(min(dp, n_experts), 1, -1):
            if dp % cand == 0:
                ep = cand
                break

    if zero_stage not in _ZERO_ENUM:
        raise ValueError(f"zero_stage must be 0..3, got {zero_stage}")

    return AdaptivePlan(
        dp=dp, tp=tp, pp=pp, sp=sp, ep=ep, zero_stage=zero_stage,
        sp_eligible=seq_eligible, has_experts=experts, world=world)


__all__ = [
    "AdaptivePlan",
    "DEFAULT_WORLD",
    "infer_parallel_config",
    "is_sequence_model",
    "has_model_experts",
    "model_num_experts",
]
```

> WHY EP is honestly inert for the current roster: `model_num_experts` reads ONLY
> the MODEL's own MoE keys and explicitly skips any optimizer (`sg2_*`) key, so for
> decoder/vit/mamba (which have NO model MoE) it returns 1 ⇒ `has_experts=False` ⇒
> `ep=1` ⇒ the plan's `parconfig_template()` emits `…, ZeROStage::Z3, 1>` which is
> byte-identical to the legacy 5-arg point. The 5th axis only activates for a future
> model that sets `model_num_experts>1` / `is_moe=True` — exactly the auto-rule.

---

## §4 — EDIT D: export the inference from the package

VERBATIM OLD (copied from `grokking_optimizers/parallel/__init__.py` — the import
block + `__all__`):
```python
from grokking_optimizers.parallel.shard_map import (
    ShardPlan,
    TensorPlacement,
    even_partition,
    partition_elementwise_even,
    partition_tensor_granular,
    shard_mode_for_optimizer,
)

__all__ = [
    "ShardPlan",
    "TensorPlacement",
    "even_partition",
    "partition_elementwise_even",
    "partition_tensor_granular",
    "shard_mode_for_optimizer",
]
```

NEW (append the `auto_config` re-exports; keep the package torch-OPTIONAL —
`auto_config` imports only `distributed.ParallelConfig`, and `distributed` imports
torch at module load, so to keep `import grokking_optimizers.parallel` from
REQUIRING torch we import `auto_config` lazily-safe: it is fine because torch IS a
hard package dep here — `distributed.py` does `import torch` at top — and the gate
`python -c "import grokking_optimizers.parallel"` runs on this box where torch is
present; verified this session that `import grokking_optimizers.parallel` succeeds):
```python
from grokking_optimizers.parallel.shard_map import (
    ShardPlan,
    TensorPlacement,
    even_partition,
    partition_elementwise_even,
    partition_tensor_granular,
    shard_mode_for_optimizer,
)
from grokking_optimizers.parallel.auto_config import (
    AdaptivePlan,
    infer_parallel_config,
    is_sequence_model,
    has_model_experts,
    model_num_experts,
)

__all__ = [
    "ShardPlan",
    "TensorPlacement",
    "even_partition",
    "partition_elementwise_even",
    "partition_tensor_granular",
    "shard_mode_for_optimizer",
    "AdaptivePlan",
    "infer_parallel_config",
    "is_sequence_model",
    "has_model_experts",
    "model_num_experts",
]
```

> IMPORT-SAFETY NOTE: `auto_config` imports `from grokking_optimizers.distributed
> import ParallelConfig`, and `distributed.py` does `import torch` at top (torch is
> a hard dep of the package — the optimizers need it). So adding this re-export
> makes `import grokking_optimizers.parallel` pull torch. That is ALREADY the case
> on this box (the gate `python -c "import grokking_optimizers.parallel"` passed
> this session WITH torch present), and the package's other modules
> (`distributed_step`/`zero3`) already import torch lazily. If a torch-free import
> of the parallel package is ever required, move the `from …distributed import
> ParallelConfig` INSIDE `infer_parallel_config`/`to_parallel_config` (call-time)
> — `auto_config`'s pure inference (`is_sequence_model`/`model_num_experts`) needs
> no torch. Recommended-but-optional; the current gate passes as written.

---

## §5 — EDIT E: extend the instantiation allow-list with the EP point (`test_parallel_instantiation.py`)

The existing allow-list test (`tests/hw/test_parallel_instantiation.py`) emits
`ParConfig<dp,tp,pp,sp, ZeROStage::Z>` (5-arg) and static_asserts the derived
flags. To prove the 6th EP param (a) COMPILES and (b) FOLDS correctly, add one EP
point and the EP-gate asserts. The existing 5-arg points stay EXACTLY as they are
(they exercise the trailing-default path — the byte-identity proof).

### §5.1 — add the EP point to the allow-list

VERBATIM OLD (copied from `tests/hw/test_parallel_instantiation.py` lines 70–78):
```python
# ─────────────────────────────── the allow-list (§7.2) ───────────────────────
# (name, DP, TP, PP, SP, ZeROStage-enumerator). SP is pinned 1 (design §1.1).
ALLOW_LIST: List[Tuple[str, int, int, int, int, str]] = [
    ("SingleGPU", 1, 1, 1, 1, "Z0"),  # bit-identical baseline (§1.2)
    ("DP8_ZeRO2", 8, 1, 1, 1, "Z2"),  # increment 1
    ("DP8_ZeRO3", 8, 1, 1, 1, "Z3"),  # increment 2
    ("DP4_PP2_ZeRO3", 4, 1, 2, 1, "Z3"),  # increment 3 (+PP)
    ("DP2_TP4_ZeRO3", 2, 4, 1, 1, "Z3"),  # increment 4 (+TP, frontier)
]
```

NEW (add EP as a trailing 7th tuple field defaulted to 1, so every existing entry
is unchanged in meaning; append the EP frontier point. The 6-int tuple keeps the
5-arg entries emitting the 5-arg template (EP omitted = the C++ default), and the
EP point emits the 6-arg template):
```python
# ─────────────────────────────── the allow-list (§7.2) ───────────────────────
# (name, DP, TP, PP, SP, ZeROStage-enumerator, EP). SP is pinned 1 (design §1.1);
# EP defaults to 1 (the dense, byte-identical trailing-default — adaptive_parallelism.md §2).
ALLOW_LIST: List[Tuple[str, int, int, int, int, str, int]] = [
    ("SingleGPU", 1, 1, 1, 1, "Z0", 1),  # bit-identical baseline (§1.2)
    ("DP8_ZeRO2", 8, 1, 1, 1, "Z2", 1),  # increment 1
    ("DP8_ZeRO3", 8, 1, 1, 1, "Z3", 1),  # increment 2
    ("DP4_PP2_ZeRO3", 4, 1, 2, 1, "Z3", 1),  # increment 3 (+PP)
    ("DP2_TP4_ZeRO3", 2, 4, 1, 1, "Z3", 1),  # increment 4 (+TP, frontier)
    ("DP1_TP1_EP8_ZeRO3", 1, 1, 1, 1, "Z3", 8),  # increment 5 (+EP, MoE frontier)
]
```

### §5.2 — emit EP in the derived-flag check + the harness template

VERBATIM OLD (copied from `tests/hw/test_parallel_instantiation.py` lines 81–125):
```python
def _expected_flags(dp: int, tp: int, pp: int, sp: int, zero: str) -> List[str]:
    """The derived ParConfig predicates each point MUST satisfy (design §1.1).

    These mirror the constexpr derivations in parallel_config.cuh, so the harness
    static_asserts both that the point compiles AND that its gates fold to the
    right values (a typo in the header's derivation fails the build loudly).
    """
    is_single = dp == 1 and tp == 1 and pp == 1 and sp == 1
    zero_rank = {"Z0": 0, "Z1": 1, "Z2": 2, "Z3": 3}[zero]
    return [
        f"P::kDP == {dp}",
        f"P::kTP == {tp}",
        f"P::kPP == {pp}",
        f"P::kSP == {sp}",
        f"P::kIsSingleGPU == {'true' if is_single else 'false'}",
        f"P::kEmitComm == {'false' if is_single else 'true'}",
        f"P::kShardParams == {'true' if zero_rank == 3 else 'false'}",
        f"P::kShardOptGrad == {'true' if zero_rank >= 2 else 'false'}",
        f"P::kTPComm == {'true' if tp > 1 else 'false'}",
        f"P::kPPStage == {'true' if pp > 1 else 'false'}",
    ]


def _emit_harness(points: List[Tuple[str, int, int, int, int, str]]) -> str:
    """Generate the minimal harness .cu source for the given allow-list points."""
    lines = [
        "// AUTO-GENERATED by tests/hw/test_parallel_instantiation.py (design §7.2).",
        "// Minimal harness: include parallel_config.cuh and static_assert the",
        "// derived gates of every allow-listed ParConfig point. nvcc -c only.",
        '#include "csrc/fused/sm_90/parallel_config.cuh"',
        "namespace par = ::sg::fused::par;",
    ]
    for (name, dp, tp, pp, sp, zero) in points:
        lines.append(
            f"using P_{name} = par::ParConfig<{dp},{tp},{pp},{sp}, par::ZeROStage::{zero}>;"
        )
        for flag in _expected_flags(dp, tp, pp, sp, zero):
            expr = flag.replace("P::", f"P_{name}::")
            lines.append(f'static_assert({expr}, "{name}: {flag}");')
    # SingleGPU must be exactly ParConfig<1,1,1,1,Z0> (the named alias).
    lines.append(
        'static_assert(par::SingleGPU::kIsSingleGPU, "SingleGPU alias must be single-GPU");'
    )
    lines.append("int main() { return 0; }")
    return "\n".join(lines) + "\n"
```

NEW (thread `ep` through `_expected_flags` + emit the 6-arg template; an EP point
asserts `kEP`/`kEPComm`/the EP contribution to `kIsSingleGPU`/`kEmitComm`):
```python
def _expected_flags(dp: int, tp: int, pp: int, sp: int, zero: str, ep: int = 1) -> List[str]:
    """The derived ParConfig predicates each point MUST satisfy (design §1.1).

    These mirror the constexpr derivations in parallel_config.cuh, so the harness
    static_asserts both that the point compiles AND that its gates fold to the
    right values (a typo in the header's derivation fails the build loudly). EP
    folds into kIsSingleGPU/kEmitComm and drives kEPComm (adaptive_parallelism.md §2).
    """
    is_single = dp == 1 and tp == 1 and pp == 1 and sp == 1 and ep == 1
    zero_rank = {"Z0": 0, "Z1": 1, "Z2": 2, "Z3": 3}[zero]
    return [
        f"P::kDP == {dp}",
        f"P::kTP == {tp}",
        f"P::kPP == {pp}",
        f"P::kSP == {sp}",
        f"P::kEP == {ep}",
        f"P::kIsSingleGPU == {'true' if is_single else 'false'}",
        f"P::kEmitComm == {'false' if is_single else 'true'}",
        f"P::kShardParams == {'true' if zero_rank == 3 else 'false'}",
        f"P::kShardOptGrad == {'true' if zero_rank >= 2 else 'false'}",
        f"P::kTPComm == {'true' if tp > 1 else 'false'}",
        f"P::kPPStage == {'true' if pp > 1 else 'false'}",
        f"P::kEPComm == {'true' if ep > 1 else 'false'}",
    ]


def _emit_harness(points: List[Tuple[str, int, int, int, int, str, int]]) -> str:
    """Generate the minimal harness .cu source for the given allow-list points."""
    lines = [
        "// AUTO-GENERATED by tests/hw/test_parallel_instantiation.py (design §7.2).",
        "// Minimal harness: include parallel_config.cuh and static_assert the",
        "// derived gates of every allow-listed ParConfig point. nvcc -c only.",
        '#include "csrc/fused/sm_90/parallel_config.cuh"',
        "namespace par = ::sg::fused::par;",
    ]
    for (name, dp, tp, pp, sp, zero, ep) in points:
        # EP==1 emits the 5-arg template (the trailing default — proves the legacy
        # point is byte-compatible); EP>1 emits the explicit 6-arg template.
        if ep == 1:
            tmpl = f"par::ParConfig<{dp},{tp},{pp},{sp}, par::ZeROStage::{zero}>"
        else:
            tmpl = f"par::ParConfig<{dp},{tp},{pp},{sp}, par::ZeROStage::{zero}, {ep}>"
        lines.append(f"using P_{name} = {tmpl};")
        for flag in _expected_flags(dp, tp, pp, sp, zero, ep):
            expr = flag.replace("P::", f"P_{name}::")
            lines.append(f'static_assert({expr}, "{name}: {flag}");')
    # SingleGPU must be exactly ParConfig<1,1,1,1,Z0> (the named alias).
    lines.append(
        'static_assert(par::SingleGPU::kIsSingleGPU, "SingleGPU alias must be single-GPU");'
    )
    # The EP default must be 1: a 5-arg point must equal the same point with EP=1.
    lines.append(
        'static_assert(par::ParConfig<1,1,1,1,par::ZeROStage::Z0>::kEP == 1, '
        '"EP must default to 1 (dense) for the legacy 5-arg point");'
    )
    lines.append(
        'static_assert(par::ParConfig<1,1,1,1,par::ZeROStage::Z0>::kEPComm == false, '
        '"EP=1 must fold kEPComm to false (byte-identical dense path)");'
    )
    lines.append("int main() { return 0; }")
    return "\n".join(lines) + "\n"
```

> The `name` indexing in `test_each_point_compiles_in_isolation` (`point[0]`) is
> unchanged (still the 0th tuple field); the only structural change is the tuple
> now has 7 fields, all consumed by the updated `_emit_harness`/`_expected_flags`.
> No other test function needs editing.

---

## §6 — DETERMINISM / BYTE-IDENTITY / A/A/A PRESERVATION

1. **Dense (EP==1) PTX-diff gate.** EP is a trailing-defaulted template param; all
   five live 5-arg `ParConfig<...>` sites resolve `EP=1` ⇒ `kEPComm==false` ⇒ the
   §2.3 EP branches fold to ZERO code. `kIsSingleGPU` gains `&& EP==1` which is TRUE
   for the `<1,1,1,1,Z0>` alias, so `SingleGPU` is unchanged. `CommCtx` grows by 5
   POD `ep_*` fields, but no dense path READS them (the by-value `comm={}` default
   sets them inert), so the dense kernel's emitted PTX is unchanged. GATE:
   `test_decoder_tc.py` (SingleGPU byte-identity) + `test_parallel_instantiation.py`
   (the 5-arg allow-list compiles + folds).

2. **EP A/A/A (when a MoE model lands).** The follow-on `ep_dispatch_alltoall` /
   `ep_combine_alltoall` (§2.4) must be FIXED-ORDER (ascending team-pe, like
   `tp_allreduce_sum_fixed_order`) so the token shuffle + the gate-weight EP reduce
   are bit-identical on every rank and across reruns. The router top-k is a
   deterministic function of the gate logits (a fixed tie-break), so routing is
   reproducible. This is the SAME determinism discipline the TP reduce already
   satisfies (`tp_transport.cuh` §DETERMINISM); it is stated here as the contract
   the §2.4 body must meet, not authored.

3. **Front-end inference determinism.** `infer_parallel_config` is a pure function
   of `(model_cfg, world, zero_stage, prefer_tp, expert_parallel)` — no randomness,
   no environment reads — so the chosen mesh is reproducible (the design §2.7
   deterministic-shard-boundary discipline, extended to the degree choice).

4. **gfx942 / tpu untouched.** Every edit is `parallel_config.cuh` (sm_90 header,
   EP behind `kEPComm`), a pure-Python module, or a CPU nvcc test. No HIP/TPU file
   is touched; the gfx942 path has no `ParConfig` instantiation (grep confirmed).

---

## §7 — GATE COMMANDS (the task's three, mapped to what they prove)

1. `python -c "import grokking_optimizers.parallel"`
   — PROVES EDIT C+D import-clean: the new `auto_config` module + its re-exports
   load without error. Verified this session that `import grokking_optimizers.
   parallel` succeeds today (torch present), and `auto_config` imports only
   `distributed.ParallelConfig` (already importable) + stdlib, so it stays green.

2. `python -m pytest tests/ -k "parallel or config" -q`
   — runs the 9 currently-collected `parallel`/`config` tests
   (`test_parallel_instantiation.py` ×3, `test_3d_parallel.py` ×6 — confirmed by
   `--co` this session). After EDIT E the `test_parallel_instantiation` trio
   exercises the 6-arg EP point + the EP-fold static_asserts; the `test_3d_parallel`
   sextet (CPU sizing/efficiency math + skip-guarded HW smoke) is untouched and
   stays green. The EP point compiles via `nvcc -c` only (no GPU, SKIPs without
   nvcc), so this is green on this box. NOTE: a NEW CPU unit test for
   `infer_parallel_config` (asserting decoder→EP=1, a MoE-flagged cfg→EP>1,
   sequence→sp_eligible) would also be selected by `-k "parallel"` if added under
   `tests/` named `test_*parallel*`/`test_*config*`; recommended as a follow-on
   (it is pure Python, CPU-green) but not required by these three gates.

3. `bash scripts/compile_to_object.sh tests/hw/tp_loopback_binding.cu`
   — PROVES the kernel-side header edits (A.1 EP template param + B EP CommCtx
   fields) compile clean WITHOUT NVSHMEM on this box. `tp_loopback_binding.cu`
   transitively includes `parallel_config.cuh` (via `tp_transport.cuh` /
   `model_stage_decoder_tc.cuh`), so the 6-arg `ParConfig` + the widened `CommCtx`
   must compile under the production flags. The EP fields are POD and the EP
   template param is defaulted, so the loopback TU (which instantiates no EP point)
   is byte-compatible — `COMPILE_OK` expected (baseline today).

---

## §8 — APPLY ORDER + CONFIDENCE + RISKS

Apply order: **A.1 + B** (the header — one file, both edits land together so the
EP gate + the EP CommCtx are coherent) → **E** (the allow-list test, so the EP
point is gated by CI) → **C + D** (the Python inference + export). All five are
byte-exact / new-file and apply on this box today; none needs a GPU.

- **A.1 (EP template param + gates):** HIGH. The trailing-defaulted 6th param is
  the ONLY placement that preserves all five live 5-arg sites (verified by grep:
  every `ParConfig<...>` uses 5 args). `kEPComm`/`kEP` mirror `kTPComm`/`kTP`
  exactly; the `kIsSingleGPU` `&& EP==1` and `static_assert … && EP>=1` are inert
  for the default. CPU-compilable header. Risk: NONE beyond the PTX-diff gate,
  which is structural (EP=1 folds).
- **B (EP CommCtx fields):** HIGH. POD twins of the TP fields with single-GPU
  defaults; trivially-copyable preserved. The struct grows 32 B but no dense path
  reads it. Matches the TP-widening pattern the live file already shipped.
- **C (front-end inference):** HIGH for the inference LOGIC (pure function, the
  auto-3D–5D rule mapped 1:1 to the kernel axis + the run_harness.md TP=world mesh),
  MEDIUM on the EP-AUTO heuristic (the `world//divisor-of-DP` EP pick is a sensible
  default but a real MoE model should pass `expert_parallel=` explicitly). The
  honest disambiguation (model `num_experts` vs the SG2 OPTIMIZER's PEER experts)
  is the load-bearing correctness point and is explicit in `model_num_experts`.
- **D (export):** HIGH. Append-only re-export; import-safety verified (torch is a
  hard package dep, the gate `import grokking_optimizers.parallel` passes today).
- **E (allow-list EP point):** HIGH. Mechanical extension of the existing harness;
  the 5-arg points are unchanged (the EP==1 branch emits the SAME 5-arg template),
  so the byte-identity proof is preserved AND the 6-arg EP point is newly gated.
- **The kernel-side EP expert-dispatch BODY (§2.4):** SCOPED FOLLOW-ON, not
  authored — by necessity (it needs a model with experts + a GPU build loop, the
  same reason tp_kernel.md scopes the TP reduce body on-silicon). The seam
  (ParConfig axis + CommCtx + EP team in distributed.py + the §2.3 gate points) is
  complete and compile-checked; the all-to-all + routed GEMM is the GPU-window
  deliverable. RISK is integration effort, not architecture — every EP construct
  is `if constexpr (Par::kEPComm)`'d so it is dead until a MoE model instantiates
  EP>1.
- **gfx942 / tpu:** UNTOUCHED. No cross-arch risk.

The single biggest honest caveat: **EP is inert for the entire current model
roster** — decoder/vit/mamba declare no model-level experts, so the front-end
inference returns EP=1 for all of them and the EP kernel path is never
instantiated. EP is the FUTURE-MoE-model seam, delivered as the byte-identical-
when-OFF axis the task asks for; it is NOT a working expert dispatch and is not
presented as one. The 3D (decoder today) / 4D (SP-eligible-but-pinned) / would-be-
5D (a MoE model) ladder is exactly the adaptive contract, with the current roster
sitting at effective-3D and the 4th/5th axes unlocked-but-inert.
```