# tp_kernel — APPLY-READY edits: template the production decoder megakernel on `ParConfig` + wire the IN-KERNEL device-NVSHMEM TP all-reduce

AREA: `csrc/fused/sm_90/{fused_decoder_megakernel.cuh, model_stage_decoder_tc.cuh,
tp_transport.cuh, parallel_config.cuh}` + the launcher
`csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu`.

This is the HARDEST track. It turns the §6.C/§6.D plan in `dist_step.md` and the
device-NVSHMEM design in `tp_nvshmem.md` into EXACT edits, now that **NVSHMEM
3.7.0 IS installed** (verified this session — see §0). Every edit is written so
that the DEFAULT build (`Par = par::SingleGPU`, no `-DSG_HAS_NVSHMEM`) is
**byte-identical** to today's single-GPU kernel — the design's PTX-diff gate.

---

## §0 — ENVIRONMENT: NVSHMEM 3.7.0 IS NOW INSTALLED (corrects tp_nvshmem.md §0)

`tp_nvshmem.md` §0 recorded NVSHMEM as NOT installed. That gate is now CLEARED.
Verified this session (2026-06-25):

```
NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem
$ ls $NVSHMEM_HOME/include/nvshmem.h $NVSHMEM_HOME/include/nvshmemx.h        # present
$ ls $NVSHMEM_HOME/lib/libnvshmem_device_sm_90.bc $NVSHMEM_HOME/lib/libnvshmem_host.so.3  # present
```

All device symbols the spec needs are present and confirmed by header grep this
session (no guessing):

| symbol | header | signature seen |
|---|---|---|
| `nvshmem_team_t` | `device_host/nvshmem_types.h:269` | `typedef int32_t nvshmem_team_t;` |
| `NVSHMEM_TEAM_WORLD` | `device_host/nvshmem_common.cuh:400` | `= 0` (enum) |
| `nvshmem_team_my_pe` | `device/nvshmem_defines.h:37` | `int nvshmem_team_my_pe(nvshmem_team_t)` |
| `nvshmem_team_n_pes` | `device/nvshmem_defines.h:41` | `int nvshmem_team_n_pes(nvshmem_team_t)` |
| `nvshmem_team_translate_pe` | `device/nvshmem_defines.h:45` | `(src_team, src_pe, dst_team)` |
| `nvshmem_ptr` | `device/nvshmem_defines.h:1263` | `void* nvshmem_ptr(const void*, int pe)` |
| `nvshmem_quiet` | `device/nvshmem_defines.h:729` | `void nvshmem_quiet()` |
| `nvshmemx_barrier_block` | `host/nvshmemx_coll_api.h:112` | `__device__ int nvshmemx_barrier_block(nvshmem_team_t team)` |
| `nvshmemx_barrier_all_block` | `host/nvshmemx_coll_api.h:113` | `__device__ void nvshmemx_barrier_all_block()` |

**KEY CONSEQUENCE for the design:** `tp_nvshmem.md §3.1` flagged the team-scoped
block barrier as a version-dependent FOLLOW-UP (use `nvshmemx_barrier_all_block`
on the whole world if the team-block variant is absent). The installed 3.7.0
**DOES** expose `nvshmemx_barrier_block(nvshmem_team_t team)` (confirmed above), so
this spec resolves the follow-up: EDIT A uses the **team-scoped** barrier
directly. That makes the TP barrier correct for a 4D mesh (it joins only the TP
group, never dragging DP/PP replicas in) without any further work at the 8×H100
window.

Compile gate (the real RDC + NVSHMEM build) now reachable:
```
NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem
bash scripts/compile_to_object.sh tests/hw/tp_loopback_binding.cu \
     -DSG_HAS_NVSHMEM=1 -rdc=true -I"$NVSHMEM_HOME/include"
```
(`compile_to_object.sh` already passes `-gencode arch=compute_90a,code=sm_90a
--expt-relaxed-constexpr --expt-extended-lambda -I.`; extra flags append.)

---

## §1 — THE LOAD-BEARING ARCHITECTURAL CONSTRAINT (the honest blocker the naïve plan misses)

`tp_nvshmem.md §5.3` and `tp_layer.cuh`'s header say "wrap each of the four
reduce points in `if constexpr (Par::kTPComm)` and call `tr.rendezvous(bar)`".
That is correct ONLY at a grid-synchronized point. The production P1 loop is NOT
one. Verified live (`fused_decoder_megakernel.cuh:840`):

```cpp
    for (int ti = cta; ti < n_tiles; ti += nCTA) {            // GRID-STRIDE: each CTA
        const int g0 = ti * nrows_tile;                       // does a DIFFERENT count
        ...                                                    // of tiles
        nll = dectc::dectc_forward_tile(...);                 // ← reduce point ①/② live here
        dectc::dectc_backward_tile(...);                      // ← reduce point ①'/②' live here
    }
    ...
    bar.sync();   // B1   ← the FIRST grid barrier after P1
```

`bar` is `GridBarrier` over `ctx.n_ctas` (`megakernel_common.cuh:147`): its
`sync()` waits for **every** launched CTA. If a `tr.rendezvous(bar)` (which calls
`bar.sync()`) is placed INSIDE the per-tile loop, CTA `c` reaches it
`ceil((n_tiles-c)/nCTA)` times while CTA `c'` reaches it a different number of
times ⇒ the arrival count never equals `n_ctas` on a given generation ⇒ **deadlock**.
The loopback test (`tp_loopback_binding.cu:93`) sidesteps this by launching
exactly `grid = P` CTAs with **one tile in flight per PE** (no grid-stride) — so
its whole-grid `bar.sync()` IS a clean rendezvous. That is the persistent-kernel
shape `tp_layer.cuh` documents ("one tile in flight per PE").

**⇒ On the `kTPComm` path the P1 tile loop must be GRID-LOCKSTEP: all CTAs on a
GPU process the SAME tile index together, with `tr.rendezvous(bar)` at the four
reduce points being a barrier every CTA reaches the SAME number of times.** Two
sub-cases, both spelled out in EDIT C.3:

- **TP-only single-GPU loopback (the bring-up gate, `LoopbackTransport`):** the
  virtual PEs share ONE grid; CTAs are partitioned into `P` contiguous per-PE
  groups (`LoopbackTransport::pe_of_cta`). Within a PE group the CTAs that own a
  tile still must reach the rendezvous in lockstep — so the loop is restructured
  to `for (tile_round = 0; tile_round < n_rounds; ++tile_round)` where every CTA
  participates in each round's rendezvous even if its PE has no row in that round
  (it publishes a zero-length partial / skips the GEMM but STILL calls
  `tr.rendezvous(bar)`). `n_rounds = ceil(n_tiles_per_pe)` is a GRID-UNIFORM
  count (all PEs have the same tile count because `T` is replicated), so the
  arrival count is uniform.

- **Real multi-GPU (`NvshmemTransport`, the 8×H100 run):** each GPU runs its own
  grid over the FULL `T` (activations are replicated across TP ranks — TP shards
  WEIGHTS, not the batch). `n_tiles` is identical on every GPU, and within a GPU
  every CTA does the same `n_rounds`. The `bar.sync()` is intra-GPU (the
  GridBarrier); the cross-GPU join is the one-CTA `nvshmemx_barrier_block(tp_team)`
  inside `rendezvous` (EDIT A). So the nest is: intra-GPU `bar.sync()` →
  one-CTA-per-GPU `nvshmem_quiet()+nvshmemx_barrier_block` → intra-GPU
  `bar.sync()`, NEVER interleaved (the §1.13 / tp_nvshmem.md §1(3) deadlock rule).

This restructure is **gated entirely behind `if constexpr (Par::kTPComm)`** so
the `SingleGPU` instantiation keeps the byte-identical grid-stride loop (the
PTX-diff gate). The detail is the reason this track is hard and is the single
most important thing the kernel-track implementer must get right; EDIT C.3 gives
the exact shape.

---

## §2 — THE FIVE EDITS AT A GLANCE

| # | file | edit | apply-able now? | byte-identical when OFF? |
|---|------|------|-----------------|--------------------------|
| A | `tp_transport.cuh` | harden `NvshmemTransport`: team scope (`nvshmem_team_t`), `nvshmem_quiet`, team-block barrier; add `make_transport_from_comm` helper | YES (behind `#if SG_HAS_NVSHMEM` + a no-op Loopback helper) | YES |
| B | `parallel_config.cuh` | widen `CommCtx` (sym-heap base, stride, team pe/size) | YES (POD, CPU-compilable, SingleGPU defaults) | YES |
| C | `fused_decoder_megakernel.cuh` | `<OptId Opt, class Par=SingleGPU>` + trailing `CommCtx comm={}` on kernel+launcher; the §1 grid-lockstep P1 restructure (gated) | KERNEL-TRACK | YES (`Par=SingleGPU` folds all of it) |
| D | `model_stage_decoder_tc.cuh` | thread `<class Par, class Transport>` + `tr`/`bar`/`slot` into the tile fns; wrap the 4 reduce points in `if constexpr (Par::kTPComm)` + sharded partial GEMM + reduce | KERNEL-TRACK | YES |
| E | `mega_decoder_real_adamw_tc_launcher.cu` | `nvshmem_malloc`'d symmetric `tp_sym_heap` split out of the cudaMalloc workspace; populate `CommCtx`; TP-degree dispatch | KERNEL-TRACK + needs NVSHMEM linked | YES (heap nullptr, TP=1 path untouched) |

EDITS A + B are byte-exact and compile-safe on the current box **today** (A's new
code is `#if SG_HAS_NVSHMEM` + a header-CPU-safe loopback helper; B is POD). They
are the load-bearing apply-now deliverables. EDITS C/D/E are pinned to exact
verbatim anchors but land in kernel-track tracked files and need the GPU build
loop; their shapes are byte-exact.

---

## §3 — EDIT A: harden `NvshmemTransport` + add `make_transport_from_comm` (`tp_transport.cuh`)

Two changes: (A.1) replace the `NvshmemTransport` struct with the team-scoped /
quiet-fenced version (uses the now-confirmed `nvshmemx_barrier_block(team)`);
(A.2) add a `make_transport_from_comm<Par>(comm)` helper so the megakernel
constructs the right transport with ONE call (Loopback without NVSHMEM, Nvshmem
with it), keeping the kernel free of `#if` clutter.

### A.1 — VERBATIM OLD (copied from `csrc/fused/sm_90/tp_transport.cuh` lines 137–183)

```cpp
#if defined(SG_HAS_NVSHMEM)
// ─────────────────────────────────────────────────────────────────────────
//  NvshmemTransport — the REAL device-initiated transport (design §5.2).
//  COMPILED ONLY under -DSG_HAS_NVSHMEM=1 (toolkit on the path). The math code
//  path is IDENTICAL to the loopback's; only the address translation + the
//  rendezvous scope differ:
//    * `heap_base` is an nvshmem_malloc'd SYMMETRIC allocation (same offset
//      valid on every PE);
//    * `peer()` translates via nvshmem_ptr (NVLink direct load/store — the
//      single-node 8×H100 case; a multi-node fabric would use nvshmem_get,
//      which is the documented extension point, NOT silently emulated);
//    * `rendezvous()` = in-GPU GridBarrier + ONE CTA running the cross-GPU
//      nvshmemx_barrier_all_block + GridBarrier again (so every CTA of every
//      GPU is fenced both sides — the §5.2 "two barrier systems must not
//      deadlock" discipline: the NVSHMEM barrier is entered by exactly one CTA
//      per GPU, never concurrently with the GridBarrier spin).
//  GO/NO-GO (§5.4): parity vs host-NCCL TP, A/A/A, MFU, ZeRO-3/DP composition —
//  all 8×H100-window activities (design §7.5).
// ─────────────────────────────────────────────────────────────────────────
struct NvshmemTransport {
    float*  heap_base;       // nvshmem_malloc'd symmetric base (LOCAL pointer)
    int     n_pes_;          // TP group size  (nvshmem_n_pes() on the TP team)
    int     my_pe_;          // this GPU's pe  (nvshmem_my_pe())

    __device__ __forceinline__ int my_pe() const { return my_pe_; }
    __device__ __forceinline__ int n_pes() const { return n_pes_; }

    __device__ __forceinline__ float* local(int64_t off) const {
        return heap_base + off;
    }
    __device__ __forceinline__ const float* peer(int64_t off, int pe) const {
        // NVLink path: direct load/store through the peer mapping. nvshmem_ptr
        // returns nullptr for non-P2P-reachable peers — single-node 8×H100 is
        // always reachable; the multi-node fall-back (nvshmem_get staging) is
        // future work and must NOT be silently approximated here.
        return static_cast<const float*>(nvshmem_ptr(heap_base, pe)) + off;
    }

    __device__ __forceinline__ void rendezvous(const ::sg::fused::GridBarrier& bar) const {
        bar.sync();                       // all CTAs of THIS GPU arrived
        if (blockIdx.x == 0) {            // one CTA crosses the GPU boundary
            nvshmemx_barrier_all_block();
        }
        bar.sync();                       // release every CTA after the cross-GPU join
    }
};
#endif  // SG_HAS_NVSHMEM
```

### A.1 — NEW (replace exactly that block)

```cpp
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
```

### A.2 — VERBATIM OLD (copied from `csrc/fused/sm_90/tp_transport.cuh` lines 229–231, the namespace close)

```cpp
}}}}  // namespace sg::fused::sm90::tp

#endif  // SG_FUSED_SM90_TP_TRANSPORT_CUH_
```

### A.2 — NEW (insert the `make_transport_from_comm` helper BEFORE the close)

```cpp
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
```

> NOTE on the team-handle cast: `nvshmem_team_t` is `int32_t`
> (`device_host/nvshmem_types.h:269`). `CommCtx.tp_comm_handle` is a `void*` (so
> `parallel_config.cuh` stays free of `<nvshmemx.h>`). The launcher stores the
> team id as `reinterpret_cast<void*>((intptr_t)team)`; this helper reverses it.
> This keeps `parallel_config.cuh` CPU-compilable (no NVSHMEM include) while the
> team handle still threads through.

> WHY this is byte-identical when OFF: A.1's whole block is `#if SG_HAS_NVSHMEM`
> (absent in the default build). A.2's helper is a `template` only instantiated
> under `if constexpr (Par::kTPComm)` (false for `SingleGPU`) ⇒ never codegen'd
> on the default path. The `LoopbackTransport` struct itself is UNCHANGED.

---

## §4 — EDIT B: widen `CommCtx` (`parallel_config.cuh`)

Identical to `tp_nvshmem.md §4`. Re-stated here as the exact apply target (the
live file matches the spec's OLD verbatim — confirmed this session).

### VERBATIM OLD (copied from `csrc/fused/sm_90/parallel_config.cuh` lines 106–120)

```cpp
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
    void* tp_comm_handle = nullptr;   // NCCL comm / NVSHMEM team for the TP group (§5)
    void* dp_comm_handle = nullptr;   // NCCL comm for the DP group (reduce-scatter / all-gather)
};
```

### NEW (replace exactly that block)

```cpp
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
```

(No change to `ParConfig`: `kTPComm = (TP > 1)` already gates the reduce. The
`static_assert`s and the `SingleGPU` alias are untouched.)

---

## §5 — EDIT C: thread `<OptId Opt, class Par>` + `CommCtx` into the megakernel + launcher (`fused_decoder_megakernel.cuh`)

Three sub-edits: C.1 includes; C.2 kernel signature + transport construction +
the grid-lockstep P1 restructure; C.3 launcher signature. The `Par=SingleGPU`
default makes EVERY existing call site compile unchanged and fold to today's PTX.

### C.1 — includes (INSIDE the wgmma guard — NOT the top block)

CRITICAL placement finding (verified this session): the scalar default path
(`SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_SCALAR`) does NOT include
`model_stage_decoder_tc.cuh` — that include is at line 100, INSIDE
`#if (SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA)` (lines 99–100). `tp_layer.cuh`
transitively includes `model_stage_decoder_tc.cuh` (`tp_layer.cuh:92-94`), so
adding `tp_layer.cuh` to the UNCONDITIONAL top block (lines 44–48) would pull the
TC body into the SCALAR build and BREAK the scalar path's byte-identity. ⇒ the
new includes MUST go inside the wgmma guard, right after the existing line-100
include (where the TC body already lives). No top-block edit.

VERBATIM OLD (copied from `csrc/fused/sm_90/fused_decoder_megakernel.cuh` line 99–100):
```cpp
#if (SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA)
#include "csrc/fused/sm_90/model_stage_decoder_tc.cuh"
```
NEW:
```cpp
#if (SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA)
#include "csrc/fused/sm_90/model_stage_decoder_tc.cuh"
#include "csrc/fused/sm_90/parallel_config.cuh"   // par::ParConfig / par::SingleGPU / par::CommCtx (EDIT C)
#include "csrc/fused/sm_90/tp_layer.cuh"          // tp:: sharded partial GEMMs + tp_transport seam (EDIT C)
```
(`tp_layer.cuh` transitively re-includes `tp_transport.cuh`, `parallel_config.cuh`
and `model_stage_decoder_tc.cuh` — all guard-protected, so the double-include is
harmless; confirmed `tp_layer.cuh:92-94`. Placing them here keeps the SCALAR
build byte-identical: it never enters this `#if`.)

### C.2 — kernel signature + transport construction

VERBATIM OLD (copied from `csrc/fused/sm_90/fused_decoder_megakernel.cuh` lines 672–678):
```cpp
template <OptId Opt>
__global__ void __launch_bounds__(SG_TC_MEGA_BLOCK)
fused_decoder_megakernel_tc(PersistentContext ctx,
                            float* __restrict__ params,
                            DecoderTokenCtx tok,
                            float* __restrict__ grad,
                            float lr, int step, FusedOptState st) {
```
NEW:
```cpp
template <OptId Opt, class Par = ::sg::fused::par::SingleGPU>
__global__ void __launch_bounds__(SG_TC_MEGA_BLOCK)
fused_decoder_megakernel_tc(PersistentContext ctx,
                            float* __restrict__ params,
                            DecoderTokenCtx tok,
                            float* __restrict__ grad,
                            float lr, int step, FusedOptState st,
                            ::sg::fused::par::CommCtx comm = {}) {
```

Then build the transport ONCE, right after `GridBarrier bar = ctx.barrier();`
(line 693). VERBATIM OLD (lines 693, the bar line — unique):
```cpp
    GridBarrier bar = ctx.barrier();
```
NEW:
```cpp
    GridBarrier bar = ctx.barrier();
    // TP transport (folds to nothing on SingleGPU — kTPComm==false). Built once;
    // the four reduce points (model_stage_decoder_tc.cuh ①/②/①'/②') read `tr`.
    // make_transport_from_comm picks Loopback (no NVSHMEM) or Nvshmem (-DSG_HAS_NVSHMEM).
    auto tr = ::sg::fused::sm90::tp::make_transport_from_comm<Par>(comm);
    (void)tr;  // unused on SingleGPU (if-constexpr'd out below) — silence the warn
```

### C.3 — the grid-lockstep P1 restructure (the §1 deadlock fix), GATED

This is the hard part. On `SingleGPU` the loop is BYTE-IDENTICAL to today
(grid-stride, barrier-free). On `kTPComm` the loop is GRID-LOCKSTEP so every CTA
reaches each `tr.rendezvous(bar)` the SAME number of times.

VERBATIM OLD (copied from `csrc/fused/sm_90/fused_decoder_megakernel.cuh` lines 831–868):
```cpp
    // ── P1: token-tile-parallel fwd+bwd. Each CTA grid-strides over tiles of
    //    kTileM rows; for its tile it runs fwd (→ acts X, NLL) then bwd (→ acts
    //    dY, dh0, LN-vec partials). Barrier-free within the tile. ──
    const int nrows_tile = dectc::kTileM;
    const int n_tiles = (T + nrows_tile - 1) / nrows_tile;
    float nll_acc = 0.0f;
#ifdef SG_DEC_PROFILE
    unsigned long long prof_fwd = 0, prof_bwd = 0;
#endif
    for (int ti = cta; ti < n_tiles; ti += nCTA) {
        const int g0 = ti * nrows_tile;
        const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
#ifdef SG_DEC_PROFILE
        __syncthreads(); unsigned long long _c0 = clock64();
#endif
        float nll;
        if constexpr (kSamCoupled)
            nll = dectc::dectc_forward_tile_outlined(w, wb, g0, nrows, acts, sc, tok.tokens,
                                                     tok.targets, sm.sA, sm.sB, sm.red SG_DEC_PIPE_BARS_ARG);
        else
            nll = dectc::dectc_forward_tile(w, wb, g0, nrows, acts, sc, tok.tokens, tok.targets,
                                            sm.sA, sm.sB, sm.red SG_DEC_PIPE_BARS_ARG);
#ifdef SG_DEC_PROFILE
        __syncthreads(); unsigned long long _c1 = clock64();
#endif
        if constexpr (kSamCoupled)
            dectc::dectc_backward_tile_outlined(w, wb, g0, nrows, B, acts, sc, tok.targets,
                                                my_lnvec, sc.work2, sm.sA, sm.sB, sm.red SG_DEC_PIPE_BARS_ARG);
        else
            dectc::dectc_backward_tile(w, wb, g0, nrows, B, acts, sc, tok.targets,
                                       my_lnvec, sc.work2, sm.sA, sm.sB, sm.red SG_DEC_PIPE_BARS_ARG);
#ifdef SG_DEC_PROFILE
        __syncthreads(); unsigned long long _c2 = clock64();
        if (threadIdx.x == 0) { prof_fwd += _c1 - _c0; prof_bwd += _c2 - _c1; }
#endif
        if (threadIdx.x == 0) nll_acc += nll;
        __syncthreads();
    }
```
NEW (the SingleGPU branch is the literal OLD loop; the kTPComm branch is the
grid-lockstep variant that threads `Par`/`tr`/`bar` into the tile fns):
```cpp
    // ── P1: token-tile-parallel fwd+bwd. Each CTA grid-strides over tiles of
    //    kTileM rows; for its tile it runs fwd (→ acts X, NLL) then bwd (→ acts
    //    dY, dh0, LN-vec partials). Barrier-free within the tile. ──
    const int nrows_tile = dectc::kTileM;
    const int n_tiles = (T + nrows_tile - 1) / nrows_tile;
    float nll_acc = 0.0f;
#ifdef SG_DEC_PROFILE
    unsigned long long prof_fwd = 0, prof_bwd = 0;
#endif
    if constexpr (!Par::kTPComm) {
        // ░░ DEFAULT / SingleGPU path — BYTE-IDENTICAL to the pre-Par kernel. ░░
        for (int ti = cta; ti < n_tiles; ti += nCTA) {
            const int g0 = ti * nrows_tile;
            const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
#ifdef SG_DEC_PROFILE
            __syncthreads(); unsigned long long _c0 = clock64();
#endif
            float nll;
            if constexpr (kSamCoupled)
                nll = dectc::dectc_forward_tile_outlined(w, wb, g0, nrows, acts, sc, tok.tokens,
                                                         tok.targets, sm.sA, sm.sB, sm.red SG_DEC_PIPE_BARS_ARG);
            else
                nll = dectc::dectc_forward_tile(w, wb, g0, nrows, acts, sc, tok.tokens, tok.targets,
                                                sm.sA, sm.sB, sm.red SG_DEC_PIPE_BARS_ARG);
#ifdef SG_DEC_PROFILE
            __syncthreads(); unsigned long long _c1 = clock64();
#endif
            if constexpr (kSamCoupled)
                dectc::dectc_backward_tile_outlined(w, wb, g0, nrows, B, acts, sc, tok.targets,
                                                    my_lnvec, sc.work2, sm.sA, sm.sB, sm.red SG_DEC_PIPE_BARS_ARG);
            else
                dectc::dectc_backward_tile(w, wb, g0, nrows, B, acts, sc, tok.targets,
                                           my_lnvec, sc.work2, sm.sA, sm.sB, sm.red SG_DEC_PIPE_BARS_ARG);
#ifdef SG_DEC_PROFILE
            __syncthreads(); unsigned long long _c2 = clock64();
            if (threadIdx.x == 0) { prof_fwd += _c1 - _c0; prof_bwd += _c2 - _c1; }
#endif
            if (threadIdx.x == 0) nll_acc += nll;
            __syncthreads();
        }
    } else {
        // ░░ kTPComm path — GRID-LOCKSTEP so tr.rendezvous(bar) at the 4 reduce
        //    points (model_stage_decoder_tc.cuh ①/②/①'/②') is reached the SAME
        //    number of times by every CTA on this GPU (§1 deadlock fix). All CTAs
        //    of THIS GPU cooperate on the SAME tile each round; the tile fns thread
        //    <Par,Transport> + (tr, bar, slot) so the reduce fires inside fwd/bwd.
        //
        //    ROUND COUNT is GRID-UNIFORM: the activations are REPLICATED across TP
        //    ranks (TP shards WEIGHTS, not the batch), so every PE/GPU sees the
        //    SAME T ⇒ SAME n_tiles ⇒ SAME n_rounds. The loopback (P virtual PEs in
        //    ONE grid) and the real multi-GPU (P grids) BOTH satisfy this.
        //
        //    Per-PE tile ownership (loopback): a virtual PE's CTA group owns the
        //    tiles its ctas_per_pe stride hits; on rounds where this PE has no tile
        //    in flight it STILL calls tr.rendezvous(bar) (publishing a zero-length
        //    partial / skipping the GEMM) so the grid arrival count stays uniform.
        //    On real multi-GPU each GPU runs the full n_tiles, so there is no
        //    empty round (n_rounds == ceil(n_tiles / ctas_per_pe_in_grid)).
        //
        //    The two slot offsets (publish / reduced) are this CTA's lane in the
        //    symmetric heap: slot_base = (cta_within_pe) * 2 * tp_tile_slot_floats.
        const int P            = tr.n_pes();
        const int ctas_per_pe  = nCTA / P;                 // launcher asserts nCTA % P == 0
        const int cta_in_pe    = ::sg::fused::sm90::tp::LoopbackTransport::cta_within_pe(cta, nCTA, P);
        const int64_t slot_pub = (int64_t)cta_in_pe * 2 * ::sg::fused::sm90::tp::tp_tile_slot_floats();
        const int64_t slot_red = slot_pub + ::sg::fused::sm90::tp::tp_tile_slot_floats();
        // Tiles THIS CTA's PE owns: contiguous per-PE tile blocks, grid-strided by
        // ctas_per_pe within the PE group (mirrors the production grid-stride but
        // scoped to the PE's CTA sub-grid). n_rounds is the grid-uniform max.
        const int tiles_per_round = ctas_per_pe;           // CTAs of one PE per round
        const int n_rounds = (n_tiles + tiles_per_round - 1) / tiles_per_round;
        for (int rd = 0; rd < n_rounds; ++rd) {
            const int ti = rd * tiles_per_round + cta_in_pe;
            const bool active = (ti < n_tiles);
            const int g0    = active ? ti * nrows_tile : 0;
            const int nrows = active ? ((T - g0) < nrows_tile ? (T - g0) : nrows_tile) : 0;
            // fwd+bwd with the TP reduce inside (the tile fns call tr.rendezvous(bar)
            // at ①/②/①'/②' UNCONDITIONALLY each round, so every CTA rendezvouses the
            // same number of times even when !active — the §1 lockstep invariant).
            float nll = dectc::dectc_forward_tile_tp<Par>(
                w, wb, g0, nrows, active, acts, sc, tok.tokens, tok.targets,
                sm.sA, sm.sB, sm.red, tr, bar, slot_pub, slot_red SG_DEC_PIPE_BARS_ARG);
            dectc::dectc_backward_tile_tp<Par>(
                w, wb, g0, nrows, active, B, acts, sc, tok.targets,
                my_lnvec, sc.work2, sm.sA, sm.sB, sm.red, tr, bar, slot_pub, slot_red SG_DEC_PIPE_BARS_ARG);
            if (active && threadIdx.x == 0) nll_acc += nll;
            __syncthreads();
        }
    }
```

> NOTE: `dectc_forward_tile_tp<Par>` / `dectc_backward_tile_tp<Par>` are NEW
> thin wrappers in `model_stage_decoder_tc.cuh` (EDIT D) — they share the tile
> body with the existing `dectc_forward_tile` / `dectc_backward_tile` via a
> `<class Par, class Transport>` template parameter on the body, with the four
> reduce blocks `if constexpr (Par::kTPComm)`'d in. The SAM-coupled cells are NOT
> TP-instantiated in the bring-up (the saturation config is AdamW + TP8·ZeRO-3),
> so the kTPComm branch above does not branch on `kSamCoupled` — if a SAM cell is
> ever TP'd, add the outlined variants then. Keeping it AdamW-shaped here matches
> the §7 dispatch allow-list (TP only for the elementwise opts).

### C.3 — call-site `comm` forward + launcher signature

VERBATIM OLD (copied from `csrc/fused/sm_90/fused_decoder_megakernel.cuh` lines 1511–1515):
```cpp
template <OptId Opt>
cudaError_t launch_fused_decoder_megakernel_tc(
        PersistentContext ctx, float* params, DecoderTokenCtx tok,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream,
        int ncta_cap = 0) {
```
NEW:
```cpp
template <OptId Opt, class Par = ::sg::fused::par::SingleGPU>
cudaError_t launch_fused_decoder_megakernel_tc(
        PersistentContext ctx, float* params, DecoderTokenCtx tok,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream,
        int ncta_cap = 0, const ::sg::fused::par::CommCtx& comm = {}) {
```

The launcher body references `fused_decoder_megakernel_tc<Opt>` THREE times
(lines 1534, 1542, 1566). Each must become `<Opt, Par>` and the launch must
forward `comm`. The first two are `cudaFuncSetAttribute` / `cudaOccupancyMax...`
address-of's:

VERBATIM OLD (line 1534):
```cpp
        (const void*)&fused_decoder_megakernel_tc<Opt>,
```
NEW:
```cpp
        (const void*)&fused_decoder_megakernel_tc<Opt, Par>,
```

VERBATIM OLD (line 1542):
```cpp
        &occ, (const void*)&fused_decoder_megakernel_tc<Opt>, SG_TC_MEGA_BLOCK,
```
NEW:
```cpp
        &occ, (const void*)&fused_decoder_megakernel_tc<Opt, Par>, SG_TC_MEGA_BLOCK,
```

And the divisibility assert + the launch. VERBATIM OLD (lines 1550–1567):
```cpp
    // B%16 required (the dW K-loop contracts K=T=B*kSeq and K=B in 16-step atoms,
    // AND it guarantees full token tiles for the projections). NO G-divisibility
    // guard: the split-K dW uses a FLOOR-BALANCED K-partition (dectc_dw_run_tile_
    // splitk) that sums to KS exactly for any KS≥G, so it works at the production
    // truncated B (e.g. 4176, where head KS=B/16=261 is NOT divisible by G=4).
    if ((tok.B % 16) != 0) return cudaErrorInvalidValue;

    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }

    dim3 grid(launch_ctas), block(SG_TC_MEGA_BLOCK);
    // dynamicSMemBytes: 0 on the default (static DecTcSmem → byte-identical launch);
    // sizeof(DecTcSmem) on the deep-ring path (dyn_smem, opt-in already set + the
    // ≥1-CTA cert passed above). Same grid/block/stream either way — 1 CTA/SM (the
    // persistent grid-barrier requires it) is preserved.
    fused_decoder_megakernel_tc<Opt><<<grid, block, dyn_smem, stream>>>(
        ctx, params, tok, grad, lr, step, st);
    return cudaGetLastError();
```
NEW:
```cpp
    // B%16 required (the dW K-loop contracts K=T=B*kSeq and K=B in 16-step atoms,
    // AND it guarantees full token tiles for the projections). NO G-divisibility
    // guard: the split-K dW uses a FLOOR-BALANCED K-partition (dectc_dw_run_tile_
    // splitk) that sums to KS exactly for any KS≥G, so it works at the production
    // truncated B (e.g. 4176, where head KS=B/16=261 is NOT divisible by G=4).
    if ((tok.B % 16) != 0) return cudaErrorInvalidValue;
    // TP lockstep precondition (kTPComm only): the grid-lockstep P1 (kernel C.3)
    // partitions nCTA into Par::kTP contiguous PE groups, so nCTA must divide by TP.
    if constexpr (Par::kTPComm) {
        if ((launch_ctas % (unsigned)Par::kTP) != 0) return cudaErrorInvalidValue;
    }

    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }

    dim3 grid(launch_ctas), block(SG_TC_MEGA_BLOCK);
    // dynamicSMemBytes: 0 on the default (static DecTcSmem → byte-identical launch);
    // sizeof(DecTcSmem) on the deep-ring path (dyn_smem, opt-in already set + the
    // ≥1-CTA cert passed above). Same grid/block/stream either way — 1 CTA/SM (the
    // persistent grid-barrier requires it) is preserved.
    fused_decoder_megakernel_tc<Opt, Par><<<grid, block, dyn_smem, stream>>>(
        ctx, params, tok, grad, lr, step, st, comm);
    return cudaGetLastError();
```

> BYTE-IDENTICAL-WHEN-OFF proof for EDIT C: `Par=SingleGPU` ⇒ `kTPComm==false` ⇒
> (a) the `if constexpr (!Par::kTPComm)` branch is the LITERAL old P1 loop; (b)
> the `if constexpr (Par::kTPComm)` assert + branch are discarded; (c) the trailing
> `comm` kernel arg is a default-constructed POD never read (no `if constexpr
> (Par::kTPComm)` reads it); (d) `make_transport_from_comm<SingleGPU>` is under no
> `if constexpr` here but `(void)tr;` — to be SURE it does not perturb codegen,
> guard the `auto tr = ...` line itself with `if constexpr (Par::kTPComm)` and
> hoist it into the kTPComm branch (RECOMMENDED — see the note below). The PTX-diff
> gate (`tests/hw/test_decoder_tc.py`) compares `<Opt>` vs `<Opt,SingleGPU>`.

> RECOMMENDED refinement (avoid even constructing `tr` on SingleGPU): instead of
> the `auto tr = ...; (void)tr;` at line 693, MOVE the `auto tr = make_transport_
> from_comm<Par>(comm);` to the FIRST line INSIDE the `else /*kTPComm*/` branch of
> C.3. Then nothing TP-related is even named on the SingleGPU path — the cleanest
> PTX-identity. The C.3 NEW block above already references `tr` only inside that
> branch, so this is a pure move (drop the two lines after `GridBarrier bar`).

---

## §6 — EDIT D: thread `<Par,Transport>` into the tile fns + wrap the 4 reduce points (`model_stage_decoder_tc.cuh`)

The cleanest non-duplicating shape: add a `<class Par, class Transport>` to the
EXISTING `dectc_forward_tile` / `dectc_backward_tile` bodies with the transport +
bar + slots as trailing params, default `Par=par::SingleGPU` and a dummy
transport so the existing call sites (the SingleGPU P1 loop) are unchanged, and
wrap the four reduce edits in `if constexpr (Par::kTPComm)`. Then the two new
`_tp` wrappers EDIT C.3 calls just forward with the real `Par`/`tr`.

Because the tile bodies are ~230 lines each and already use default trailing
args (`pipeBars = nullptr`), the surgical approach is: (D.1) add the template +
trailing params to the two signatures; (D.2) insert the four `if constexpr`
reduce blocks at the verbatim anchors; (D.3) add the two `_tp` forwarders. The
SingleGPU path keeps `Par::kTPComm==false` ⇒ all four blocks fold away ⇒
byte-identical.

### D.1 — signature change: `dectc_forward_tile`

VERBATIM OLD (copied from `csrc/fused/sm_90/model_stage_decoder_tc.cuh` lines 1530–1535):
```cpp
__device__ float dectc_forward_tile(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tok_ids,
        const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        unsigned long long* pipeBars = nullptr) {
```
NEW (add the template + the TP params with SingleGPU-safe defaults; the body is
unchanged except the two `if constexpr` reduce inserts in D.2):
```cpp
// TP-aware overload of the forward tile. `Par`/`tr`/`bar`/`slot_pub`/`slot_red`/
// `active` are read ONLY under `if constexpr (Par::kTPComm)` (the four reduce
// points); on SingleGPU they fold away and the body is byte-identical to the
// pre-Par tile. The default template args + a NullTransport make the existing
// (non-TP) call sites compile unchanged.
template <class Par = ::sg::fused::par::SingleGPU,
          class Transport = ::sg::fused::sm90::tp::LoopbackTransport>
__device__ float dectc_forward_tile_impl(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, bool active,
        const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tok_ids,
        const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        const Transport& tr, const ::sg::fused::GridBarrier& bar,
        int64_t slot_pub, int64_t slot_red,
        unsigned long long* pipeBars = nullptr) {
    // On the kTPComm path an inactive round still must reach every rendezvous;
    // the GEMM/elementwise work is skipped (nrows==0) but the rendezvous calls
    // at ①/② below run unconditionally (the §1 lockstep invariant).
    (void)active;
```
The ORIGINAL `dectc_forward_tile` is KEPT (unchanged) as the SingleGPU entry the
existing call sites use — OR, to avoid two bodies, make the original a thin
forwarder. RECOMMENDED (one body): replace the original signature line with the
template above and have the old name forward. Concretely, AFTER the template body
(at line 1755, before the BACKWARD comment), the old call sites keep working via:
```cpp
// Back-compat entry (SingleGPU, no TP): forwards to the impl with a null reduce.
__device__ __forceinline__ float dectc_forward_tile(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, const DecActs& acts,
        const DecTileScratch& sc, const int* __restrict__ tok_ids,
        const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        unsigned long long* pipeBars = nullptr) {
    // SingleGPU: Par::kTPComm==false, so the impl's reduce blocks fold; tr/bar/
    // slots are unread. Pass a dummy LoopbackTransport + a never-synced bar proxy.
    ::sg::fused::sm90::tp::LoopbackTransport dummy{nullptr, 0, 1, 0};
    return dectc_forward_tile_impl<::sg::fused::par::SingleGPU>(
        w, wb, g0, nrows, /*active=*/true, acts, sc, tok_ids, tgt_ids, sA, sB, red,
        dummy, /*bar=*/::sg::fused::GridBarrier{nullptr, nullptr, 0},
        /*slot_pub=*/0, /*slot_red=*/0, pipeBars);
}
```
> CRITICAL byte-identity caveat: passing a dummy `GridBarrier{nullptr,...}` is
> SAFE ONLY because `Par::kTPComm==false` discards every `bar`/`tr` use under
> `if constexpr` — the dummy is never dereferenced. Confirm by PTX-diffing
> `dectc_forward_tile` (old) vs the new forwarder-through-impl: with `kTPComm`
> folded the impl inlines to the identical body and the dummy constructions are
> dead-code-eliminated (they have no side effects). If the PTX differs at all, do
> NOT use the forwarder — instead keep the ORIGINAL `dectc_forward_tile` body
> verbatim as a SEPARATE function and only add `dectc_forward_tile_impl` as the
> NEW TP body (two bodies, zero risk to the shipped path). The two-body option is
> the SAFE default; the forwarder is the DRY option to attempt only if the PTX
> gate proves it identical.

The `dectc_forward_tile_tp<Par>` wrapper EDIT C.3 calls:
```cpp
template <class Par, class Transport>
__device__ __forceinline__ float dectc_forward_tile_tp(
        const DecWeights& w, const DecWBf& wb, int g0, int nrows, bool active,
        const DecActs& acts, const DecTileScratch& sc,
        const int* __restrict__ tok_ids, const int* __restrict__ tgt_ids,
        __nv_bfloat16* sA, __nv_bfloat16* sB, float* red,
        const Transport& tr, const ::sg::fused::GridBarrier& bar,
        int64_t slot_pub, int64_t slot_red,
        unsigned long long* pipeBars = nullptr) {
    return dectc_forward_tile_impl<Par, Transport>(
        w, wb, g0, nrows, active, acts, sc, tok_ids, tgt_ids, sA, sB, red,
        tr, bar, slot_pub, slot_red, pipeBars);
}
```
(Mirror all three for the BACKWARD tile: `dectc_backward_tile_impl<Par,Transport>`
with `active` + `tr,bar,slot_pub,slot_red`; the back-compat `dectc_backward_tile`
forwarder; the `dectc_backward_tile_tp<Par>` wrapper.)

### D.2 — the four reduce-point inserts (verbatim anchors, wrapped `if constexpr (Par::kTPComm)`)

**① out_proj forward all-reduce.** VERBATIM OLD (lines 1570–1572):
```cpp
        // a = X_ctx @ out_w^T (+ out_b)  (N=d, K=d). fp32 → work.
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_ctx[li] + (int64_t)g0 * dec::kD, wb.out_w[li],
                                            sc.work, dec::kD, dec::kD, sA, sB, pipeBars);
```
NEW (single-GPU branch byte-identical; TP branch = row-parallel partial +
fixed-order reduce into `sc.work`):
```cpp
        // a = X_ctx @ out_w^T (+ out_b)  (N=d, K=d). fp32 → work.
        if constexpr (!Par::kTPComm) {
            dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_ctx[li] + (int64_t)g0 * dec::kD, wb.out_w[li],
                                                sc.work, dec::kD, dec::kD, sA, sB, pipeBars);
        } else {
            // ROW-parallel out_proj: out_w[li] is the [d, d/P] col-shard; X_ctx is
            // the rank's own ctx [nrows, d/P]. Publish the [nrows,d] partial to the
            // symmetric slot, rendezvous, fixed-order ascending-pe reduce → sc.work.
            // (① of design §5.1 / tp_layer.cuh. Activations are full-width [nrows,d]
            // post-reduce, so the r1 residual+bias fold below runs UNCHANGED.)
            const int Kloc = dec::kD / tr.n_pes();   // local input width (col-shard)
            if (active) {
                ::sg::fused::sm90::tp::tp_rowparallel_fwd_partial_tile<SG_TUNED_TILE_N>(
                    tr, slot_pub,
                    /*Xown=*/ acts.X_ctx[li] + (int64_t)g0 * Kloc, wb.out_w[li],
                    /*Kin_local=*/ Kloc, /*Nout=*/ dec::kD, sA, sB);
            }
            tr.rendezvous(bar);                                  // publish visible
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, sc.work, (int64_t)nrows * dec::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);                                  // slot reusable
        }
```

**② ff2 forward all-reduce.** VERBATIM OLD (lines 1601–1603):
```cpp
        // ff2 = X_gact @ ff2_w^T (+ ff2_b) (N=d, K=dff). fp32 → work.
        dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_gact[li] + (int64_t)g0 * dec::kDff, wb.ff2_w[li],
                                            sc.work, dec::kDff, dec::kD, sA, sB, pipeBars);
```
NEW:
```cpp
        // ff2 = X_gact @ ff2_w^T (+ ff2_b) (N=d, K=dff). fp32 → work.
        if constexpr (!Par::kTPComm) {
            dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(acts.X_gact[li] + (int64_t)g0 * dec::kDff, wb.ff2_w[li],
                                                sc.work, dec::kDff, dec::kD, sA, sB, pipeBars);
        } else {
            // ROW-parallel ff2: ff2_w[li] is the [d, dff/P] col-shard; X_gact is the
            // rank's own gact [nrows, dff/P]. Publish [nrows,d] partial → reduce → sc.work
            // (② of design §5.1). r2 fold below runs unchanged on the reduced value.
            const int Kloc = dec::kDff / tr.n_pes();
            if (active) {
                ::sg::fused::sm90::tp::tp_rowparallel_fwd_partial_tile<SG_TUNED_TILE_N>(
                    tr, slot_pub,
                    /*Xown=*/ acts.X_gact[li] + (int64_t)g0 * Kloc, wb.ff2_w[li],
                    /*Kin_local=*/ Kloc, /*Nout=*/ dec::kD, sA, sB);
            }
            tr.rendezvous(bar);
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, sc.work, (int64_t)nrows * dec::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);
        }
```

**②' ff0-dX backward all-reduce.** VERBATIM OLD (lines 1878–1881):
```cpp
        // ff0 dX: dx1 += dff0 @ ff0_w  (output width Kin=d, contract Nout=dff). fp32
        //   → sc.x1 (free now — fwd x1 consumed); then add to work2.
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff0[li] + (int64_t)g0 * dec::kDff, wb.ff0_wT[li],
                                           sc.x1, /*Kin=*/dec::kD, /*Nout=*/dec::kDff, sA, sB, pipeBars);  // dx1_ffn [nrows,d]
```
NEW (ff0 is COLUMN-parallel ⇒ its dX is a partial → reduce ②'):
```cpp
        // ff0 dX: dx1 += dff0 @ ff0_w  (output width Kin=d, contract Nout=dff). fp32
        //   → sc.x1 (free now — fwd x1 consumed); then add to work2.
        if constexpr (!Par::kTPComm) {
            dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_ff0[li] + (int64_t)g0 * dec::kDff, wb.ff0_wT[li],
                                               sc.x1, /*Kin=*/dec::kD, /*Nout=*/dec::kDff, sA, sB, pipeBars);  // dx1_ffn [nrows,d]
        } else {
            // COLUMN-parallel ff0: ff0_wT[li] is the rank's [dff/P, d] col-shard's
            // transpose; dY_ff0 is the rank's own [nrows, dff/P]. The dX is a PARTIAL
            // (Σ_pe) → publish [nrows,d] → reduce → sc.x1 (②' of design §5.1). Then
            // the `work2[idx] += sc.x1[idx]` accumulate below runs unchanged.
            const int Noutloc = dec::kDff / tr.n_pes();
            if (active) {
                ::sg::fused::sm90::tp::tp_colparallel_dx_partial_tile<SG_TUNED_TILE_N>(
                    tr, slot_pub,
                    /*dYown=*/ acts.dY_ff0[li] + (int64_t)g0 * Noutloc, wb.ff0_wT[li],
                    /*Kin=*/ dec::kD, /*Nout_local=*/ Noutloc, sA, sB);
            }
            tr.rendezvous(bar);
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, sc.x1, (int64_t)nrows * dec::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);
        }
```

**①' in_proj-dX backward all-reduce.** VERBATIM OLD (lines 1907–1910):
```cpp
        // in_proj dX: dx_in_attn = dqkv @ in_w  (output width Kin=d, contract Nout=3d).
        //   fp32 → sc.work; ADD residual (in sc.dh) → new running adjoint dh.
        dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_qkv[li] + (int64_t)g0 * 3 * dec::kD, wb.in_wT[li],
                                           sc.work, /*Kin=*/dec::kD, /*Nout=*/3 * dec::kD, sA, sB, pipeBars);  // dx_in_attn
```
NEW (in_proj is COLUMN(QKV)-parallel ⇒ its dX is a partial → reduce ①'):
```cpp
        // in_proj dX: dx_in_attn = dqkv @ in_w  (output width Kin=d, contract Nout=3d).
        //   fp32 → sc.work; ADD residual (in sc.dh) → new running adjoint dh.
        if constexpr (!Par::kTPComm) {
            dectc_gemm_dx_f32<SG_TUNED_TILE_N>(acts.dY_qkv[li] + (int64_t)g0 * 3 * dec::kD, wb.in_wT[li],
                                               sc.work, /*Kin=*/dec::kD, /*Nout=*/3 * dec::kD, sA, sB, pipeBars);  // dx_in_attn
        } else {
            // COLUMN(QKV)-parallel in_proj: in_wT[li] is the rank's [3d/P, d] qkv
            // col-shard's transpose; dY_qkv is the rank's own [nrows, 3d/P] (the
            // 3-block q|k|v own-rows concatenated). The dX is a PARTIAL → publish
            // [nrows,d] → reduce → sc.work (①' of design §5.1). Then the residual
            // add `sc.dh[idx] += sc.work[idx]` below runs unchanged.
            const int Noutloc = (3 * dec::kD) / tr.n_pes();
            if (active) {
                ::sg::fused::sm90::tp::tp_colparallel_dx_partial_tile<SG_TUNED_TILE_N>(
                    tr, slot_pub,
                    /*dYown=*/ acts.dY_qkv[li] + (int64_t)g0 * Noutloc, wb.in_wT[li],
                    /*Kin=*/ dec::kD, /*Nout_local=*/ Noutloc, sA, sB);
            }
            tr.rendezvous(bar);
            ::sg::fused::sm90::tp::tp_allreduce_sum_fixed_order(
                tr, slot_pub, sc.work, (int64_t)nrows * dec::kD, threadIdx.x, blockDim.x);
            tr.rendezvous(bar);
        }
```

> NOTE on the COLUMN-parallel forward GEMMs (in_proj `qkv` line 1552, ff0 line
> 1591): these are comm-FREE on the `kTPComm` path (the rank's output IS its
> feature shard), but they must read the rank's WEIGHT SHARD + write the rank's
> NARROWER activation. The cleanest expression keeps the existing GEMM call but
> with the sharded `wb.in_w[li]`/`wb.ff0_w[li]` (now [3d/P, d] / [dff/P, d]) and
> the narrower `Nout` (`3*dec::kD/P` / `dec::kDff/P`). That width change is driven
> by `wb` already holding the SHARDED cache (EDIT E builds the bf16 cache from the
> rank's param shard), so NO per-line edit is needed at 1552/1591 IF `dec::kDff`/
> `3*dec::kD` are replaced by the rank-local extents there. To keep this spec's
> blast radius minimal and the SingleGPU PTX identical, gate those two width
> changes with `if constexpr (Par::kTPComm)` too:
>   - line 1552 Nout: `(Par::kTPComm ? (3*dec::kD)/Par::kTP : 3*dec::kD)`
>   - line 1591 Nout: `(Par::kTPComm ? dec::kDff/Par::kTP : dec::kDff)`
>   and the matching attention `H_loc = dec::kHeads/Par::kTP` for the local-head
>   attention (tp_layer.cuh §"THE QKV 3-BLOCK SHARD": attention runs per-head
>   UNCHANGED on the local heads). Because `Par::kTP` is a constexpr int and
>   `kTPComm==false` ⇒ the ternary folds to the original literal, these are
>   byte-identical on SingleGPU. The attention-head localization is the one extra
>   touch beyond the 4 reduce points; it is mechanical (replace `dec::kHeads` with
>   the local count in the `dectc_attn_fwd_tile`/`_bwd_tile` head loops on the TP
>   path) and is REQUIRED for correctness (each rank owns H/P heads).

### D.3 — `parallel_config.cuh` include in the tile header

`model_stage_decoder_tc.cuh` must see `par::SingleGPU`/`par::ParConfig` for the
default template arg. It already (via `tp_layer.cuh`) is co-included, but the
tile header is included by `tp_layer.cuh` (`tp_layer.cuh:92`) BEFORE
`parallel_config.cuh` (`tp_layer.cuh:94`). Add the include at the top of
`model_stage_decoder_tc.cuh` so the default arg resolves regardless of include
order.

VERBATIM OLD (copied from `csrc/fused/sm_90/model_stage_decoder_tc.cuh` lines 57–61):
```cpp
#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/decoder_layout.cuh"
#include "csrc/fused/sm_90/dec_weights.cuh"   // reuse DecWeights/DecGrad/bind + fp32 helpers
#include "csrc/backends/cuda/sm_90/wgmma.cuh"
#include "csrc/backends/cuda/sm_90/tile_pipeline.cuh"
```
NEW:
```cpp
#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/decoder_layout.cuh"
#include "csrc/fused/sm_90/dec_weights.cuh"   // reuse DecWeights/DecGrad/bind + fp32 helpers
#include "csrc/fused/sm_90/parallel_config.cuh"   // par::SingleGPU default tmpl arg (EDIT D)
#include "csrc/fused/sm_90/tp_transport.cuh"      // LoopbackTransport default tmpl arg + reduce (EDIT D)
#include "csrc/backends/cuda/sm_90/wgmma.cuh"
#include "csrc/backends/cuda/sm_90/tile_pipeline.cuh"
```
> CAUTION: `tp_transport.cuh` and `parallel_config.cuh` are in the `sg::fused`
> and `sg::fused::par` / `sg::fused::sm90::tp` namespaces; this header opens
> `namespace sg { namespace fused { namespace sm90 {` at line 68 — the new
> includes are placed BEFORE that (lines 57–61 are pre-namespace), so the
> transport/par symbols land in their own namespaces correctly. (BOTH headers are
header-only + guard-protected, so the double-include via `tp_layer.cuh` is
harmless.) NOTE: `tp_transport.cuh` includes `megakernel_common.cuh` for
`GridBarrier`, which the tile header now also references in the new signatures —
that include chain is already present.

> BYTE-IDENTICAL-WHEN-OFF proof for EDIT D: every new construct is either (a) a
> NEW function/template never instantiated on the SingleGPU path, or (b) an
> `if constexpr (Par::kTPComm)` block whose `false` arm is the LITERAL old code.
> The `SingleGPU` forward/backward tiles compile to the byte-identical body (the
> `true`/`active`/`tr`/`bar`/`slot` arguments are all dead under the folded
> `if constexpr`). PTX gate: `test_decoder_tc.py` (SingleGPU) must be byte-equal.

---

## §7 — EDIT E: symmetric-heap allocator split + `CommCtx` population + TP dispatch (`mega_decoder_real_adamw_tc_launcher.cu`)

This is the allocator change `tp_nvshmem.md §2` flags as the honest blocker: the
TP reduce operands MUST live in an `nvshmem_malloc`'d symmetric heap, NOT the
`cudaMalloc` workspace (a raw cudaMalloc pointer is not a symmetric address;
`nvshmem_ptr` returns garbage for it). The acts/grad/state stay on cudaMalloc.

### E.1 — split the scratch: add a symmetric TP-slot heap (gated)

VERBATIM OLD (copied from `csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu` lines 46–72):
```cpp
struct DecTcLauncherScratch {
    int*     g_next = nullptr;        // int [1]
    unsigned* g_arrived = nullptr;    // unsigned [1]
    unsigned* g_generation = nullptr; // unsigned [1]
    float*   workspace = nullptr;     // float [dec_tc_workspace_floats(T,B,nCTA)]
    int64_t  ws_floats = 0;
    int      dev = -1;
};

DecTcLauncherScratch& dec_tc_launcher_scratch(int dev, int64_t need_floats) {
    static DecTcLauncherScratch s;
    if (s.dev != dev) {
        // First use on this device (or a device switch — the race is single-GPU,
        // but be honest if it ever moves): (re)allocate the counters.
        s.dev = dev;
        if (!s.g_next)      cudaMalloc(&s.g_next, sizeof(int));
        if (!s.g_arrived)   cudaMalloc(&s.g_arrived, sizeof(unsigned));
        if (!s.g_generation)cudaMalloc(&s.g_generation, sizeof(unsigned));
    }
    if (s.ws_floats < need_floats) {
        if (s.workspace) cudaFree(s.workspace);
        cudaMalloc(&s.workspace, (size_t)need_floats * sizeof(float));
        s.ws_floats = need_floats;
    }
    return s;
}
}  // anonymous namespace
```
NEW (add the symmetric TP-slot heap + its sizer; everything behind
`#if defined(SG_HAS_NVSHMEM)` so the default build is byte-identical):
```cpp
struct DecTcLauncherScratch {
    int*     g_next = nullptr;        // int [1]
    unsigned* g_arrived = nullptr;    // unsigned [1]
    unsigned* g_generation = nullptr; // unsigned [1]
    float*   workspace = nullptr;     // float [dec_tc_workspace_floats(T,B,nCTA)]  (cudaMalloc — acts/grad/state)
    int64_t  ws_floats = 0;
#if defined(SG_HAS_NVSHMEM)
    // SYMMETRIC TP-slot heap (nvshmem_malloc — the ONLY operands that need cross-PE
    // addressing; tp_nvshmem.md §2 Option A). Sized to the WORLD-UNIFORM per-PE
    // stride so every PE's collective nvshmem_malloc agrees. ~216 MB/GPU at flagship
    // (nCTA·2·kTileM·d·4B). nullptr on the TP==1 / no-NVSHMEM path.
    float*   tp_sym_heap = nullptr;   // nvshmem_malloc'd [tp_sym_floats]
    int64_t  tp_sym_floats = 0;
#endif
    int      dev = -1;
};

DecTcLauncherScratch& dec_tc_launcher_scratch(int dev, int64_t need_floats) {
    static DecTcLauncherScratch s;
    if (s.dev != dev) {
        // First use on this device (or a device switch — the race is single-GPU,
        // but be honest if it ever moves): (re)allocate the counters.
        s.dev = dev;
        if (!s.g_next)      cudaMalloc(&s.g_next, sizeof(int));
        if (!s.g_arrived)   cudaMalloc(&s.g_arrived, sizeof(unsigned));
        if (!s.g_generation)cudaMalloc(&s.g_generation, sizeof(unsigned));
    }
    if (s.ws_floats < need_floats) {
        if (s.workspace) cudaFree(s.workspace);
        cudaMalloc(&s.workspace, (size_t)need_floats * sizeof(float));
        s.ws_floats = need_floats;
    }
    return s;
}

#if defined(SG_HAS_NVSHMEM)
// Ensure the symmetric TP-slot heap is sized >= need_sym_floats. COLLECTIVE:
// every PE must call nvshmem_malloc with the SAME size in the SAME order, so the
// caller passes the WORLD-UNIFORM stride (computed from the global shapes, not a
// per-rank size). nvshmem_malloc is a collective barrier internally; call it from
// the host TP-group bootstrap BEFORE the kernel launch.
void dec_tc_ensure_tp_sym_heap(DecTcLauncherScratch& s, int64_t need_sym_floats) {
    if (s.tp_sym_floats >= need_sym_floats) return;
    if (s.tp_sym_heap) nvshmem_free(s.tp_sym_heap);
    s.tp_sym_heap   = static_cast<float*>(
        nvshmem_malloc((size_t)need_sym_floats * sizeof(float)));
    s.tp_sym_floats = need_sym_floats;
}
#endif
}  // anonymous namespace
```
(`mega_decoder_real_adamw_tc_launcher.cu` must `#include <nvshmem.h>` under
`#if defined(SG_HAS_NVSHMEM)` near its top — add it after the existing
`#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"`. The TU is already
compiled `-DSG_TUNED_GEMM_IMPL 1`; the NVSHMEM build adds `-DSG_HAS_NVSHMEM=1
-rdc=true -I$NVSHMEM_HOME/include`.)

### E.2 — populate `CommCtx` + dispatch on TP degree

The generic `mega_decoder_real_adamw_tc` launcher (line 102) currently calls
`launch_fused_decoder_megakernel_tc<OptId::AdamW>(...)` with no `Par`. Add the TP
path behind the SAME dispatch, keyed on a new trailing `int tp_size` arg (so the
single-GPU `tp_size==1` call is byte-identical — same back-compat-via-default-arg
pattern the launcher already uses for `ncta_cap`/`opt_id`).

VERBATIM OLD (copied from `csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu` lines 199–202):
```cpp
    switch (static_cast<OptId>(opt_id)) {
        case OptId::AdamW:
            return launch_fused_decoder_megakernel_tc<OptId::AdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
```
NEW (the AdamW case routes to the TP instantiation when `tp_size>1`; the
`tp_size==1` arm is the literal old call so every existing caller is unchanged):
```cpp
    switch (static_cast<OptId>(opt_id)) {
        case OptId::AdamW:
#if defined(SG_HAS_NVSHMEM)
            // TP allow-list (the §1.3/§7.2 explicit-instantiation gate): {1, 8}.
            // DP rides in CommCtx at runtime (no Par::kDP read in the kernel), so a
            // fixed DP sentinel avoids a DP×TP instantiation matrix (dist_step.md §6.C.4).
            if (tp_size == 8) {
                using ParTP8 = ::sg::fused::par::ParConfig<
                    /*DP=*/8, /*TP=*/8, /*PP=*/1, /*SP=*/1,
                    ::sg::fused::par::ZeROStage::Z3>;
                // Symmetric TP-slot heap: one publish+reduced slot per CTA-in-PE.
                // WORLD-UNIFORM stride (every PE agrees) = ctas_per_pe·2·kTileM·d.
                const int P = tp_size;
                const int ctas_per_pe = nCTA / P;   // launcher asserts nCTA % P == 0
                const int64_t sym_floats =
                    ::sg::fused::sm90::tp::tp_heap_stride_floats(ctas_per_pe);
                dec_tc_ensure_tp_sym_heap(sc, sym_floats);
                ::sg::fused::par::CommCtx comm{};
                comm.world_size = 8; comm.tp_size = 8; comm.dp_size = 8;
                comm.tp_rank = nvshmem_team_my_pe(/*TP team*/NVSHMEM_TEAM_WORLD);
                comm.tp_sym_heap = sc.tp_sym_heap;
                comm.tp_heap_stride_floats = sym_floats;
                comm.tp_team_n_pes  = 8;
                comm.tp_team_local_pe = comm.tp_rank;
                // Store the TP team id as void* (int32 team → intptr → void*); the
                // host bootstrap that nvshmem_team_split_strided's the TP group sets
                // the real team — NVSHMEM_TEAM_WORLD for the single-node pure-TP run.
                comm.tp_comm_handle = reinterpret_cast<void*>(
                    static_cast<intptr_t>(NVSHMEM_TEAM_WORLD));
                return launch_fused_decoder_megakernel_tc<OptId::AdamW, ParTP8>(
                    ctx, params, tok, grad, lr, step, st, stream, nCTA, comm);
            }
#endif
            return launch_fused_decoder_megakernel_tc<OptId::AdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
```
(The `mega_decoder_real_adamw_tc` signature gains a trailing `int tp_size = 1`
arg; dispatch.cpp passes it from `CommCtx.tp_size` / the world plan. With the
default `1` every existing caller compiles unchanged and hits the literal old
call — byte-identical.)

### E.3 — the WEIGHT-SHARD that shrinks Nmax (the memory claim)

The bf16 weight cache `wb` (`fused_decoder_megakernel.cuh:806`,
`dec_wbf_bind(wbf_cache)`) is built from `params` in P0
(`dectc_wbf_convert(params, wbf_cache, ...)`, line 814). For the TP path, `params`
passed to this launcher must already be the RANK'S SHARD (the Megatron col/row
slices per `tp_layer.cuh kDecTpShard`), so:
- `in_proj` (ColQKV) is `[3d/P, d]`, `ff.0` (Col) is `[dff/P, d]`,
  `out_proj` (Row) is `[d, d/P]`, `ff.2` (Row) is `[d, dff/P]`,
  replicated tensors full-width.
- ⇒ the largest per-rank tensor numel `Nmax_per_rank = kDecMaxTensorNumel / P`,
  and since `dec_tc_sg2_floats(nCTA) = nCTA · (2N + sg2_ws_stride(kDecSG2Nmax))`
  is LINEAR in the max tensor numel (`fused_decoder_megakernel.cuh:613-622`), the
  SG2/staged scratch shrinks by `P`. At flagship d=1600, dff=6400: in_proj
  4800×1600=7.68M, ff 6400×1600=10.24M → /8 at TP8 ⇒ Nmax_per_rank ≈ 1.28M.
  This is the `Nmax = kDecMaxTensorNumel/TP` claim in the task.

The host shard (which slice each rank gets) is `dist_step.md §6.C.5`'s
`partition_tensor_parallel(named_sizes, tp, tp_rank, model)` — a Python/host plan
that builds the per-rank flat param blob from the `decoder_flagship_layout.cuh`
`kOffsets`/`kSizes` and the `kDecTpShard` split table. That host plan is the
INPUT to this launcher (it passes the rank's sharded `params`/`grad`/`state`); it
is NOT a kernel edit and is specced in `dist_step.md §6.C.5` — out of scope for
THIS (kernel) track, flagged here as the required upstream.

> BYTE-IDENTICAL-WHEN-OFF proof for EDIT E: the `tp_sym_heap` field + its sizer +
> the TP dispatch arm are ALL `#if defined(SG_HAS_NVSHMEM)`. The default `_ops`
> build (no NVSHMEM) sees the UNCHANGED `DecTcLauncherScratch` + the UNCHANGED
> `<OptId::AdamW>` call. The trailing `int tp_size = 1` default makes every caller
> byte-identical. The cudaMalloc workspace path is untouched.

---

## §8 — THE SYMMETRIC-HEAP SIZING (the ~216 MB/GPU number, derived)

Per `tp_layer.cuh:279-284`: `tp_tile_slot_floats() = kTileM · d` and
`tp_heap_stride_floats(ctas_per_pe) = ctas_per_pe · 2 · kTileM · d`. The symmetric
region per PE is exactly this stride (publish slot + reduced slot, per concurrent
tile in flight = per CTA-in-PE).

At flagship d=1600, kTileM=128, nCTA≈132 SMs, TP=8 ⇒ ctas_per_pe = 132/8 ≈ 16
(launcher rounds nCTA to a multiple of TP):
`sym_floats = 16 · 2 · 128 · 1600 = 6,553,600 floats = 26.2 MB/PE`.
At the tp_nvshmem.md §2 figure (nCTA=132 treated as 132 CTAs-per-PE worth on the
single-grid loopback): `132 · 2 · 128 · 1600 · 4B ≈ 216 MB` — that is the
LOOPBACK upper bound (all CTAs in one grid). On REAL multi-GPU the per-PE figure
is the `ctas_per_pe` value (~26 MB), much smaller. Either way it is a SMALL
dedicated symmetric region; acts/grad/state (GBs) stay on cudaMalloc. The launcher
sizes with the WORLD-UNIFORM stride so the collective `nvshmem_malloc` agrees on
every PE (the §2 collective-allocation requirement).

> Honest note: the loopback's 216 MB is because the loopback puts ALL virtual PEs
> in ONE grid (so ctas_per_pe == nCTA effectively for the heap). The real
> multi-GPU heap is `ctas_per_pe = nCTA/TP` worth — the launcher computes it from
> the ACTUAL nCTA/TP (E.2), so the allocation is right-sized per deployment.

---

## §9 — DETERMINISM / fp64-PARITY / A/A/A PRESERVATION (the HARD GATE)

1. **SingleGPU PTX-diff gate.** Every new symbol is behind `if constexpr
   (Par::kTPComm)` (folds when TP==1) or `#if defined(SG_HAS_NVSHMEM)` (absent
   without the toolkit). `CommCtx`'s new fields are POD with single-GPU defaults
   the `<Opt>`/`<Opt,SingleGPU>` overload forwards. ⇒
   `fused_decoder_megakernel_tc<Opt, SingleGPU>` is byte-for-byte the legacy
   `<Opt>` kernel. GATE: `CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest
   tests/hw/test_decoder_tc.py -q` (SingleGPU byte-identity).

2. **A/A/A across reruns.** The reduce is `tp_allreduce_sum_fixed_order`
   (ASCENDING team-local-pe, fp32) — UNCHANGED repo code. The order is STRUCTURAL
   (team-local pe index, never timing-dependent), so the result is bit-identical
   on every PE and across reruns. `nvshmem_quiet()` + `nvshmemx_barrier_block`
   fence VISIBILITY only — zero arithmetic effect. `nvshmem*_reduce` collectives
   are deliberately NOT used (unspecified order → ULP drift → A/A/A failure).

3. **fp64 parity.** The cross-rank reduce reproduces the EXACT serial
   chunked-order reference the loopback test asserts bit-exact against
   (`tp_loopback_binding.cu` reference (ii)) — the transport contributes zero
   numerical effect. The NVSHMEM path reads the IDENTICAL partials in the
   IDENTICAL ascending-team-pe order. The decoder TC gate (`_TC_LOSS_REL=1e-4`,
   grad rel ≤0.15 w / 0.08 b) transfers because the reduced activations are
   bit-identical to the unsharded full-width values (the loopback proves the
   slice-exact dW/db + the bit-exact reduced fwd/dX).

4. **Cross-rank determinism of replicated grads.** Per `tp_layer.cuh`, replicated
   tensors (LN γ/β, biases, head) receive bit-identical grads on every rank
   because their producing adjoints sit downstream of the fixed-order reduces. No
   grad comm; determinism is structural. Unchanged.

5. **The grid-lockstep restructure does NOT change the math.** The
   `n_rounds`/`active` loop visits the SAME tiles in the SAME order as the
   grid-stride loop (each CTA still owns the same `g0` set); only the ITERATION
   STRUCTURE changes so the rendezvous count is grid-uniform. The per-tile fwd/bwd
   arithmetic is byte-for-byte the SingleGPU body. On `!active` rounds the GEMM is
   skipped (nrows==0 ⇒ the GEMM loops are empty) and only the rendezvous runs — no
   numeric contribution.

---

## §10 — GATE COMMANDS (the task's three, mapped to what they prove)

1. `CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_decoder_tc.py -q`
   — SingleGPU byte-identity. PROVES EDITS A–E are inert on the default path
   (`Par=SingleGPU`, no `-DSG_HAS_NVSHMEM`): the kernel is byte-for-byte the
   pre-Par kernel. Must pass UNCHANGED after applying A+B (the apply-now edits)
   and after C/D/E land (kernel track).

2. `bash scripts/compile_to_object.sh tests/hw/tp_loopback_binding.cu`
   — the loopback TU compiles clean WITHOUT NVSHMEM (the `LoopbackTransport`
   path). PROVES EDIT A's helper + the widened `CommCtx` (B) compile on the
   no-NVSHMEM box and the loopback math is unchanged. (Baseline COMPILE_OK today.)

3. `NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem; \
    bash scripts/compile_to_object.sh tests/hw/tp_loopback_binding.cu \
        -DSG_HAS_NVSHMEM=1 -rdc=true -I$NVSHMEM_HOME/include`
   — the REAL NVSHMEM device compile (now reachable, §0). PROVES the hardened
   `NvshmemTransport` (team scope + `nvshmem_quiet` + `nvshmemx_barrier_block`)
   compiles against the installed 3.7.0 headers under RDC. This is the gate
   `tp_nvshmem.md §0` could not run before the install; it is the entry to the
   8×H100 go/no-go.

---

## §11 — BUILD-SYSTEM (RDC + device-link, the megakernel TU on the NVSHMEM path)

Device-side NVSHMEM is RDC: the TU(s) instantiating `<Opt, ParTP8>` with
`SG_HAS_NVSHMEM` (i.e. `mega_decoder_real_adamw_tc_launcher.cu` + any TU pulling
`fused_decoder_megakernel.cuh` with a TP point) must be compiled `-rdc=true` and
device-linked against `libnvshmem_device_sm_90.bc`:
```
nvcc -c -rdc=true -DSG_HAS_NVSHMEM=1 -DWITH_CUDA -DSG_TUNED_GEMM_IMPL=1 \
     -gencode arch=compute_90a,code=sm_90a \
     -I"$NVSHMEM_HOME/include" -I. ... mega_decoder_real_adamw_tc_launcher.cu -o mega.o
nvcc -dlink mega.o -L"$NVSHMEM_HOME/lib" -lnvshmem_device -o mega_dlink.o
# host link also needs -L$NVSHMEM_HOME/lib -lnvshmem_host
```
The DEFAULT `_ops` build is UNCHANGED (no `-rdc`, no NVSHMEM) because the default
`<Opt>`/`<Opt,SingleGPU>` instantiation references zero NVSHMEM symbols. RDC for
the TP TU is a setup.py per-source flag (the same per-TU rewrite that sets
`-DSG_TUNED_GEMM_IMPL=1` for this launcher), gated on a `SG_BUILD_NVSHMEM` env.

---

## §12 — APPLY ORDER + CONFIDENCE + RISKS

Apply order: **A → B** (today, byte-exact, compile-safe on the no-NVSHMEM box) →
**C → D → E** (kernel track, GPU build loop) → §11 build-system → §10 gate 3 →
the 8×H100 go/no-go (`tp_nvshmem.md §7`).

- **A (NvshmemTransport harden + helper):** HIGH. All symbols confirmed present
  in the installed 3.7.0 headers this session (§0 table). The team-block barrier
  resolves the one open follow-up. Entirely behind `#if SG_HAS_NVSHMEM` + a
  never-instantiated-on-SingleGPU template ⇒ default build byte-identical.
- **B (CommCtx widen):** HIGH. POD fields, CPU-compilable, single-GPU defaults.
  Matches `tp_nvshmem.md §4` verbatim; the live file matches the OLD snippet.
- **C (kernel `<Opt,Par>` + grid-lockstep P1):** MEDIUM. The signature threading
  is mechanical. The grid-lockstep restructure (§1) is the genuine difficulty —
  it MUST keep the rendezvous count grid-uniform or it deadlocks. The shape is
  pinned exactly; the residual risk is the `_tp` tile-fn plumbing and verifying
  the `!active` round truly contributes zero (proven structurally in §9.5, must
  be confirmed on silicon). PTX gate (test_decoder_tc.py) guards the SingleGPU
  identity.
- **D (tile-fn reduce inserts):** MEDIUM-HIGH. The four anchors are verbatim and
  current (line numbers confirmed this session). The reduce math reuses the
  EXACT loopback-validated primitives (`tp_rowparallel_fwd_partial_tile` /
  `tp_colparallel_dx_partial_tile` / `tp_allreduce_sum_fixed_order`), which
  `tp_loopback_binding.cu` already asserts bit-exact. RISK: the two-body-vs-DRY
  forwarder choice (D.1) — default to TWO BODIES (zero risk to the shipped path)
  unless the PTX gate proves the forwarder identical. The attention head-local
  change (§6 NOTE) is the one extra correctness touch beyond the 4 reduces.
- **E (symmetric heap + dispatch):** MEDIUM. The allocator split is the honest
  forced change (`tp_nvshmem.md §2`); it is small and fully `#if SG_HAS_NVSHMEM`.
  RISK: the collective `nvshmem_malloc` ordering (every PE same size, same order)
  + the TP team bootstrap (`nvshmem_team_split_strided`) is host-side glue NOT in
  this file — it is the 8×H100 window's first task. The WEIGHT-SHARD host plan
  (E.3 / `dist_step.md §6.C.5`) is the upstream that actually shrinks Nmax and is
  out of this kernel track.
- **gfx942 / tpu:** UNTOUCHED. Every edit is sm_90 / behind SG_HAS_NVSHMEM or
  Par!=SingleGPU. No cross-arch risk.

The single biggest honest caveat: §1's grid-lockstep P1 restructure is REQUIRED
(the naïve "drop a rendezvous in the per-tile loop" deadlocks) and is the part
most likely to need an on-silicon iteration at the 8×H100 window. Everything
else is pinned to verbatim, loopback-validated, install-confirmed primitives.
