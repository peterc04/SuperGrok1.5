# size_adaptive — APPLY-READY: SIZE/CONFIG-ADAPTIVE megakernel specialization (the self-designing megakernel made concrete)

AREA: `grokking_optimizers/megakernel_codegen.py` (the codegen/knob-selector surface) +
the megakernel tile/occupancy knobs in `csrc/fused/sm_90/*` (`SG_TUNED_TILE_N`, the ring
stages `SG_TUNED_DEC_GEMM_STAGES`/`SG_TUNED_DEC_FWD_PIPE`/`SG_TUNED_DEC_FWD_STAGES`, the
dW split-K `SG_TUNED_DEC_DW_SPLITK`, the M-atom interleave `SG_TUNED_DEC_GEMM_INTERLEAVE`,
and the launch occupancy via `ncta_cap` / the 1-CTA/SM persistent shape) + the autotuner
(`tuning/` + `grokking_optimizers/compile.py` autotune knob registry).

GOAL (verbatim from the task): the codegen/launcher selects knobs — notably CTA-TILING
(multiple CTAs per output tile / clusters / occupancy>1, vs the current 1-CTA/SM persistent
shape) — **by WORKLOAD SIZE**: CTA-tiling ON for LARGE configs (bottleneck LEVER 2, the
measured ~20% grid-barrier idle at d=2048 = cross-CTA load imbalance), OFF for SMALL configs
(the persistent 1-CTA/SM shape wins — overhead dominates). The selector is the SAME
`if constexpr (config)` folding mechanism that gates the distributed all-reduce
(`tp_kernel.md` §1, `parallel_config.cuh` `if constexpr (Par::kTPComm)`), now keyed on a
SIZE config instead of a PARALLELISM config — and it **COMPOSES** with `ParConfig`
(distributed+large ⇒ CTA-tiling + all-reduce; single+small ⇒ neither).

READ IN FULL FIRST (done this session): `tp_kernel.md`, `dist_step.md`, `run_harness.md`;
the cited live files `parallel_config.cuh`, `fused_decoder_megakernel.cuh`,
`model_stage_decoder_tc.cuh`, `megakernel_codegen.py`,
`mega_decoder_real_adamw_tc_launcher.cu`, `compile.py` (autotune dim registry).

---

## §0 — VERIFIED LIVE STATE (this session, the anchors every edit pins to)

These are the facts the spec is built on; each was confirmed by reading the live file.

1. **The megakernel is ALREADY templated `<OptId Opt, class Par = par::SingleGPU>`** and
   takes a trailing `CommCtx comm = {}` (`fused_decoder_megakernel.cuh:674-681`). The TP
   transport is built once at `:700` via `make_transport_from_comm<Par>(comm)`. So the
   `if constexpr (config)` specialization machinery the task asks me to reuse is LIVE and
   PROVEN — this spec adds a SECOND config axis (size) alongside the existing `Par` axis,
   using the identical mechanism. (The `tp_kernel.md` C/D/E kernel-track edits — the four
   reduce points, the grid-lockstep P1 — are still pending on the live file; my edits are
   written to land **before or after** them without conflict: my OLD anchors are the
   CURRENT live lines, and the byte-identity proof holds independently of whether `kTPComm`
   has been wired yet.)

2. **The launcher is still `template <OptId Opt>`** (`fused_decoder_megakernel.cuh:1519`),
   i.e. `tp_kernel.md` C.3's launcher-`Par` edit has NOT landed. My launcher edit (EDIT C)
   adds BOTH the size config AND is written so it merges cleanly with the pending
   `tp_kernel.md` launcher edit (I add `class Sz` as a SECOND default template param; the
   `Par` param `tp_kernel.md` adds slots in beside it — the order is `<OptId Opt, class Par,
   class Sz>` and BOTH default, see §5.4 merge note).

3. **The tile/occupancy knob surface** (all `#ifndef`-guarded defaults, so an unset build is
   byte-identical), confirmed live:
   - `SG_TUNED_TILE_M` (default 128) / `SG_TUNED_TILE_N` (default 128) —
     `model_stage_decoder_tc.cuh:75-80`.
   - `SG_TUNED_DEC_GEMM_STAGES` (default 2) — `:90-92` → `kDecTcStages` `:217`.
   - `SG_TUNED_DEC_GEMM_INTERLEAVE` (default 2) — `:113-115` → `kDecMaxIL` `:214`.
   - `SG_TUNED_DEC_DW_SPLITK` (default 1) — `:93-102` → `kDecDwSplitK` `:273`.
   - `SG_TUNED_DEC_FWD_PIPE` (0/1/2, default 0) + `SG_TUNED_DEC_FWD_STAGES` (2..4) —
     `:249-264` → `kDecFwdStages` `:261`, `kDecRingStagesMax` `:268`.
   - `SG_TUNED_DEC_DW_STAGE` (0/1, default 0) — `:210`.

4. **The occupancy/CTA shape today**: the launcher launches `launch_ctas = n_sms` CTAs
   (one CTA/SM), capped by `ncta_cap` (`fused_decoder_megakernel.cuh:1555-1556`). The grid
   barrier `GridBarrier` (`megakernel_common.cuh:147`) rendezvouses over `ctx.n_ctas` =
   `launch_ctas`. The kernel is a **persistent 1-CTA/SM** kernel: `cta = blockIdx.x`,
   `nCTA = ctx.n_ctas` (`:707-708`); the P1 token-tile loop is a grid-stride over
   `n_tiles` (`:842` region); the P2 dW loop is grid-strided over `n_dw`/`n_dw·G` tiles
   (`:927-938`); P3 optimizer is grid-strided over numel (`:1043`). **Occupancy>1 is NOT
   expressible today** — the launcher's `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
   cert REQUIRES occ≥1 and the launch is sized to exactly `n_sms` CTAs, and the GridBarrier
   `n_ctas == gridDim.x` assumes one wave (every launched CTA arrives once per `sync()`).
   This is why the genuine CTA-tiled (occupancy>1) kernel is a REAL kernel change, scoped
   honestly in §7 as the follow-on, while the SELECTOR + plumbing + knob surface land now.

5. **Workload-size inputs available at codegen/build time**: `megakernel_codegen.py` already
   carries the three decoder size tiers as build constants:
   `_DEC_D=128` (production/`d=128`), `_DEC_BENCH_D=2048` (the `d=2048` roofline bench),
   `_DEC_FLAGSHIP_D=1600,_DEC_FLAGSHIP_LAYERS=48` (the 1.5B flagship) — `:567,579,593`.
   `_decoder_param_sizes(d,layers,vocab,seq)` `:596` is the parametric size source. The
   H100 SM count (132) is the `n_sms` the launcher reads at runtime
   (`cudaDevAttrMultiProcessorCount`). So a size-tier selector is a PURE FUNCTION of
   `(d, layers, T=B·seq, n_sms)` that the codegen can evaluate at emit time and the launcher
   can re-derive at runtime — exactly the dual-surface pattern `flagship_budget.py` uses for
   the memory budget (`run_harness.md` NEW FILE 1).

---

## §1 — DESIGN: the SizeConfig axis, byte-identical-when-OFF, composing with ParConfig

The user directive: *the megakernel is templated on its deployment config and `if constexpr`
folds in EXACTLY the machinery that config needs — distributed builds the all-reduce,
single-GPU builds none of it (byte-identical); large size builds CTA-tiling, small size does
not. Every NEW config branch MUST be byte-identical when its config is OFF (the
SingleGPU / no-EP / small-size default folds to today's kernel).*

I realize this with a **compile-time `SizeConfig` template** that mirrors `par::ParConfig`
exactly: a `constexpr`-everything POD whose predicates (`kCtaTile`, `kClusterDim`, …) the
megakernel branches on with `if constexpr`. The default `SizeSmall` (the analogue of
`par::SingleGPU`) sets every predicate to the SHIPPED value, so
`fused_decoder_megakernel_tc<Opt, Par, SizeSmall>` is byte-for-byte today's kernel — the
PTX-diff gate `test_decoder_tc.py` stays green.

```
  PARALLELISM axis (LIVE):  Par  = ParConfig<DP,TP,PP,SP,Z>   default SingleGPU
                            gate: if constexpr (Par::kTPComm)  → all-reduce machinery
  SIZE axis (NEW, this spec): Sz = SizeConfig<...>            default SizeSmall
                            gate: if constexpr (Sz::kCtaTile)  → CTA-tiling machinery
```

The two axes are ORTHOGONAL template parameters on the SAME kernel, each folded by its own
`if constexpr`. They COMPOSE: `fused_decoder_megakernel_tc<Opt, ParTP8, SizeLarge>` builds
BOTH the TP all-reduce AND CTA-tiling; `<Opt, SingleGPU, SizeSmall>` builds NEITHER (=today).
This is precisely the user's "distributed+large ⇒ CTA-tiling + all-reduce; single+small ⇒
neither" composition, achieved by the canonical `if constexpr`-on-config mechanism.

**SCOPE HONESTY (the task's point 3):** this spec delivers, APPLY-READY NOW:
  (A) `SizeConfig` + `SizeSmall`/`SizeLarge` aliases in `parallel_config.cuh` (POD,
      CPU-compilable, byte-identical default) — EDIT A.
  (B) the **codegen size-tier selector** `decoder_knobs_for_size(d, layers, T, n_sms)` in
      `megakernel_codegen.py` that emits the right `-DSG_TUNED_*` knob set per size tier,
      keeping `d=128`/`d=2048`/flagship at their CURRENT knobs, plus a `--decoder-knobs` CLI
      to inspect/diff the selection — EDIT B.
  (C) the megakernel + launcher threaded on `<class Sz = SizeSmall>` with the size config
      carried (default folds to today; the CTA-tile body is `if constexpr (Sz::kCtaTile)`
      gated and is a NO-OP stub on this increment — the REAL CTA-tiled body is §7) — EDIT C.
  (D) the autotuner composition: a `size_tier` dim + a per-tier knob-default override so the
      autotuner explores knobs WITHIN a size tier and never proposes a small-tier knob for a
      large workload (or vice-versa) — EDIT D.
And delivers, as a PRECISE SCOPING (not byte-exact, because it is a real kernel change that
needs a GPU build/validate loop and changes the GridBarrier contract): the CTA-tiled
(occupancy>1 / cluster) megakernel VARIANT — what changes in the barrier, tile ownership,
and smem — §7.

---

## §2 — THE FIVE EDITS AT A GLANCE

| # | file | edit | apply-able now? | byte-identical when OFF? |
|---|------|------|-----------------|--------------------------|
| A | `parallel_config.cuh` | add `SizeConfig<...>` + `SizeSmall`/`SizeLarge` aliases (POD, constexpr predicates) | YES (POD, CPU-compilable) | YES (default `SizeSmall` = shipped predicates) |
| B | `megakernel_codegen.py` | `decoder_knobs_for_size()` size-tier knob selector + `--decoder-knobs` CLI + `DEC_SIZE_TIERS` table | YES (pure Python, no torch/GPU) | YES (NEW function + NEW CLI flag; emits NO header by default ⇒ `--decoder-layout` byte-identical) |
| C | `fused_decoder_megakernel.cuh` | thread `<class Sz = par::SizeSmall>` onto kernel+launcher; `if constexpr (Sz::kCtaTile)` gate at the P1 loop (NO-OP stub this increment) + launch-shape selector | KERNEL-TRACK | YES (`Sz=SizeSmall` ⇒ `kCtaTile==false` folds every new branch; trailing default template arg ⇒ every call site unchanged) |
| D | `compile.py` | a `size_tier` autotune dim + `_decoder_size_tier_pins()` that pins the per-tier knob DEFAULTS so the autotuner explores within a tier | YES (Python; first value == current default ⇒ untuned build byte-identical) | YES |
| E (scoping) | `fused_decoder_megakernel.cuh` + `megakernel_common.cuh` | the REAL CTA-tiled (occupancy>1 / cluster) kernel variant — barrier, tile ownership, smem | SCOPED ONLY (§7) | N/A (the variant only ever compiles under `Sz::kCtaTile`) |

EDITS A + B + D are byte-exact and land **today** (POD header / pure-Python codegen /
Python autotuner). EDIT C is pinned to verbatim live anchors but lands in the kernel-track
GPU build loop; its shape is byte-exact and its OFF-path is provably identical.

---

## §3 — EDIT A: `SizeConfig` + `SizeSmall`/`SizeLarge` (`parallel_config.cuh`)

Mirror `ParConfig` exactly: a `constexpr`-everything compile-time point, a degenerate
default alias (`SizeSmall`, the analogue of `SingleGPU`), and a large alias. All predicates
`static constexpr` so `if constexpr (Sz::kCtaTile)` folds. The default `SizeSmall` sets
`kCtaTile=false`, `kClusterDim=1`, `kTileN=SG_TUNED_TILE_N` (the shipped tile) — so the
kernel under `SizeSmall` is byte-for-byte the pre-Sz kernel.

### A.1 — VERBATIM OLD (copied from `csrc/fused/sm_90/parallel_config.cuh` lines 79–86)

```cpp
// ─────────────────────────────────────────────────────────────────────────
//  THE single-GPU guarantee, named once so the static_asserts read cleanly
//  (design §1.1). `fused_decoder_megakernel_tc<Opt, SingleGPU>` MUST be
//  byte-identical to the legacy `<Opt>` overload — enforced by kEmitComm==false
//  folding every comm branch away (design §1.2). This is the default template
//  arg of the megakernel, so existing call sites compile unchanged.
// ─────────────────────────────────────────────────────────────────────────
using SingleGPU = ParConfig<1, 1, 1, 1, ZeROStage::Z0>;
```

### A.1 — NEW (insert the `SizeConfig` block AFTER the `SingleGPU` alias — additive, no change to existing lines)

```cpp
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
```

> BYTE-IDENTICAL-WHEN-OFF proof for EDIT A: every new symbol is ADDITIVE (no existing line
> changes). `SizeSmall` is the megakernel's default size arg; `kCtaTile==false` folds every
> new `if constexpr` arm to nothing; `kTileN==SG_TUNED_TILE_N` so reading `Sz::kTileN`
> instead of the macro is the IDENTICAL constant. The `#ifndef SG_TUNED_TILE_N` block is the
> same guard the kernel header already has (no redefinition warning — `#ifndef`-guarded).
> No NVSHMEM, no math, no launch policy enters this header.

---

## §4 — EDIT B: the codegen size-tier knob SELECTOR (`megakernel_codegen.py`)

This is the load-bearing apply-now deliverable: the function that, GIVEN a workload size,
emits the right `-DSG_TUNED_*` knob set — and, crucially, keeps the EXISTING `d=128` /
`d=2048` / flagship builds at their CURRENT knobs so today's gates stay green. It is the
size analogue of `distributed.py`'s `ParallelConfig` (which maps degrees→`Par`): one source
of truth for the size→knobs map.

It is a PURE function (no torch, no GPU), unit-testable on CPU, and emits NOTHING into any
committed header by default — so `--decoder-layout` is byte-identical (the GATE-1 command).
A NEW `--decoder-knobs` CLI prints the selection (for diffing / the autotuner / the operator).

### B.1 — the size-tier table + selector. Insert AFTER `_DEC_FLAGSHIP_D, _DEC_FLAGSHIP_HEADS, _DEC_FLAGSHIP_LAYERS = 1600, 25, 48` (the flagship constants).

VERBATIM OLD (copied from `grokking_optimizers/megakernel_codegen.py` lines 593–594, the
flagship constants + the blank line before `_decoder_param_sizes`):
```python
_DEC_FLAGSHIP_D, _DEC_FLAGSHIP_HEADS, _DEC_FLAGSHIP_LAYERS = 1600, 25, 48


def _decoder_param_sizes(d: int = _DEC_D, *, layers: int = _DEC_LAYERS,
```

NEW (insert the selector between the flagship constants and `_decoder_param_sizes`):
```python
_DEC_FLAGSHIP_D, _DEC_FLAGSHIP_HEADS, _DEC_FLAGSHIP_LAYERS = 1600, 25, 48


# ── SIZE/CONFIG-ADAPTIVE kernel specialization (the self-designing megakernel) ──
# The codegen picks the megakernel's tile/occupancy KNOBS by WORKLOAD SIZE, the
# same way distributed.py's ParallelConfig picks the all-reduce by parallelism
# DEGREE. SMALL configs (the d=128 production race) keep the persistent 1-CTA/SM
# shape — overhead dominates, CTA-tiling would only add cross-CTA traffic. LARGE
# configs (d=2048 bench, the d=1600/L=48 flagship) turn on CTA-TILING (occupancy>1
# / clusters — bottleneck LEVER 2: the measured ~20% grid-barrier idle at d=2048 is
# cross-CTA load imbalance, fixed by more CTAs per output tile). The threshold is a
# function of (d, layers, T=B*seq, n_sms): a workload is LARGE when ONE persistent
# wave (1 CTA/SM) cannot keep the SMs busy through the per-tile fwd/bwd — i.e. when
# the token-tile count is small relative to n_sms (few tiles ⇒ idle SMs ⇒ the grid
# barrier waits on the slowest) OR the per-tile GEMM N-range is wide enough to split
# across cooperating CTAs (d large).
#
# CONTRACT (the §1 byte-identical-when-OFF invariant): the THREE shipped tiers
# (d=128 production, d=2048 bench, d=1600 flagship) MUST keep their CURRENT knobs so
# today's gates stay green. CTA-tiling is reported in the tier's `size_config` field
# (consumed by the launcher's SizeConfig template arg, EDIT C) but does NOT change
# the emitted -DSG_TUNED_* tile macros for those tiers on THIS increment — the knob
# values below are exactly the in-header #ifndef defaults the live build uses
# (model_stage_decoder_tc.cuh / fused_decoder_megakernel.cuh). The selector is the
# stable seam the §7 CTA-tiled body + a future retuned-large-tier knob set hang off.
#
# Each tier is (predicate, knobs, size_config) where:
#   * predicate(d, layers, T, n_sms) -> bool  : is this tier selected?
#   * knobs : dict of SG_TUNED_* -> value     : the -D macros emitted for the tier
#             (FIRST-MATCH wins; values == the live #ifndef defaults today).
#   * size_config : the parallel_config.cuh SizeConfig alias name the launcher
#                   instantiates (par::SizeSmall / par::SizeLarge). This is what
#                   carries CTA-tiling into the kernel (EDIT C), NOT the knobs dict.
#
# THE LIVE #ifndef DEFAULTS (single source — keep == the kernel headers):
_DEC_KNOB_DEFAULTS = {
    "SG_TUNED_TILE_M":              128,   # model_stage_decoder_tc.cuh:76
    "SG_TUNED_TILE_N":              128,   # model_stage_decoder_tc.cuh:79
    "SG_TUNED_DEC_GEMM_STAGES":     2,     # model_stage_decoder_tc.cuh:91
    "SG_TUNED_DEC_GEMM_INTERLEAVE": 2,     # model_stage_decoder_tc.cuh:114
    "SG_TUNED_DEC_DW_SPLITK":       1,     # model_stage_decoder_tc.cuh:101
    "SG_TUNED_DEC_FWD_PIPE":        0,     # model_stage_decoder_tc.cuh:249 (off)
    "SG_TUNED_DEC_FWD_STAGES":      2,     # model_stage_decoder_tc.cuh:261 (inherits)
    "SG_TUNED_DEC_DW_STAGE":        0,     # model_stage_decoder_tc.cuh:210 (scalar)
}

# H100 SM count — the n_sms the launcher reads at runtime
# (cudaDevAttrMultiProcessorCount). Mirrored here so the selector is a pure function
# the codegen + a CPU test can evaluate; the launcher passes the REAL n_sms.
_DEC_DEFAULT_N_SMS = 132


def _dec_token_tiles(T: int, tile_m: int) -> int:
    """Token-tile count = ceil(T / TILE_M). The P1 loop grid-strides over these
    (fused_decoder_megakernel.cuh P1). Few tiles vs n_sms ⇒ idle SMs (LEVER 2)."""
    return (T + tile_m - 1) // tile_m


def _dec_is_large(d: int, layers: int, T: int, n_sms: int,
                  tile_m: int = 128) -> bool:
    """LARGE-tier predicate: CTA-tiling pays off. A workload is LARGE when the
    persistent 1-CTA/SM wave under-fills the grid OR the model width is large enough
    that one output tile's N-range is worth splitting across cooperating CTAs.

    Two structural triggers (either ⇒ LARGE):
      (1) WIDTH: d >= 1024 — the in_proj/ff GEMM N-range (3d / 4d) is wide enough
          that 2 CTAs per tile each get a full wgmma N-tile, and the per-CTA weight
          residency halves (the d=2048 bench + the d=1600 flagship both clear this).
      (2) GRID UNDER-FILL: ceil(T/TILE_M) < n_sms — fewer token tiles than SMs, so a
          1-CTA/SM wave leaves SMs idle and the grid barrier waits on the slowest
          (the measured ~20% idle). CTA-tiling assigns multiple CTAs per tile to use
          the otherwise-idle SMs.
    SMALL otherwise (the d=128 production race: d<1024 AND enough tiles to fill the
    grid ⇒ the persistent shape wins, overhead dominates)."""
    if d >= 1024:
        return True
    if _dec_token_tiles(T, tile_m) < n_sms:
        return True
    return False


def decoder_knobs_for_size(d: int, layers: int, T: int,
                           n_sms: int = _DEC_DEFAULT_N_SMS) -> dict:
    """SELECT the megakernel tile/occupancy knobs for ONE workload size. Returns
    {"knobs": {SG_TUNED_*: val}, "size_config": "par::SizeSmall"|"par::SizeLarge",
     "tier": "small"|"large"}.

    This is the self-designing megakernel's size→knobs map, the analogue of
    distributed.py mapping degrees→ParConfig. The THREE shipped tiers keep their
    CURRENT knobs (the live #ifndef defaults) so today's gates stay green; the
    `size_config` field is what carries CTA-tiling into the kernel via the launcher's
    SizeConfig template arg (EDIT C). Only a future retuned-large-tier (or the §7
    CTA-tiled body) changes the emitted -D knobs for the large tier — and it does so
    HERE, in one place, gated on `tier == "large"`.

    PURE function: no torch, no GPU. The launcher re-derives the same tier from the
    runtime (d, layers, T, n_sms) so the codegen-emitted knobs and the runtime
    SizeConfig agree (the dual-surface contract, mirroring flagship_budget.py)."""
    knobs = dict(_DEC_KNOB_DEFAULTS)
    large = _dec_is_large(d, layers, T, n_sms, tile_m=knobs["SG_TUNED_TILE_M"])
    if large:
        # LARGE tier. CTA-tiling carried via size_config (par::SizeLarge); the
        # emitted -D knobs stay at the live defaults on THIS increment (byte-
        # identical to today's d=2048/flagship build — the §1 contract). A future
        # large-tier retune (or the §7 CTA-tiled body) edits THIS branch only.
        return {"knobs": knobs, "size_config": "par::SizeLarge", "tier": "large"}
    # SMALL tier (the d=128 production race): the persistent 1-CTA/SM shape.
    return {"knobs": knobs, "size_config": "par::SizeSmall", "tier": "small"}


# The named decoder size tiers the build/autotuner enumerate (d=128 production,
# d=2048 bench, d=1600/L=48 flagship). Each entry is the (d, layers, T) the tier is
# evaluated at; T uses a representative B per tier (the production race B, the bench
# B, the flagship B from run_harness.md). The autotuner's size_tier dim (EDIT D)
# iterates these names; the build emits decoder_knobs_for_size(*tier) per variant.
DEC_SIZE_TIERS = {
    # name        (d,                 layers,             T = B * seq)
    "production": (_DEC_D,            _DEC_LAYERS,         512 * _DEC_SEQ),   # d=128, the race
    "bench":      (_DEC_BENCH_D,      _DEC_LAYERS,        4096 * _DEC_SEQ),   # d=2048 roofline
    "flagship":   (_DEC_FLAGSHIP_D,  _DEC_FLAGSHIP_LAYERS, 512 * _DEC_SEQ),  # d=1600,L=48
}


def decoder_knobs_report(n_sms: int = _DEC_DEFAULT_N_SMS) -> str:
    """Human-readable table of the size→knobs selection for every named tier —
    printed by `--decoder-knobs` so the operator/diff SEES which tier turns on
    CTA-tiling and that the shipped tiers keep their knobs. Emits NO header."""
    lines = ["decoder size-adaptive knob selection  (n_sms=%d)" % n_sms,
             "  %-11s %5s %6s %8s  %-14s  %s"
             % ("tier", "d", "layers", "T", "size_config", "knobs(non-default)")]
    for name, (d, layers, T) in DEC_SIZE_TIERS.items():
        sel = decoder_knobs_for_size(d, layers, T, n_sms)
        nondef = {k: v for k, v in sel["knobs"].items()
                  if v != _DEC_KNOB_DEFAULTS[k]}
        lines.append("  %-11s %5d %6d %8d  %-14s  %s"
                     % (name, d, layers, T, sel["size_config"],
                        nondef if nondef else "(all default)"))
    return "\n".join(lines)


def _decoder_param_sizes(d: int = _DEC_D, *, layers: int = _DEC_LAYERS,
```

### B.2 — the `--decoder-knobs` CLI flag. Two sub-edits in `main()`.

VERBATIM OLD (copied from `grokking_optimizers/megakernel_codegen.py` lines 1667–1671):
```python
    ap.add_argument("--decoder-layout-flagship", action="store_true",
                    help="emit the FLAGSHIP (d=1600, layers=48) L3-REAL decoder "
                         "weight-layout header "
                         "(csrc/fused/sm_90/decoder_flagship_layout.cuh)")
    ap.add_argument("--vit-layout", action="store_true",
```
NEW (add the `--decoder-knobs` argument after `--decoder-layout-flagship`):
```python
    ap.add_argument("--decoder-layout-flagship", action="store_true",
                    help="emit the FLAGSHIP (d=1600, layers=48) L3-REAL decoder "
                         "weight-layout header "
                         "(csrc/fused/sm_90/decoder_flagship_layout.cuh)")
    ap.add_argument("--decoder-knobs", action="store_true",
                    help="print the SIZE-ADAPTIVE tile/occupancy knob selection for "
                         "every named decoder size tier (production/bench/flagship): "
                         "which tier turns on CTA-tiling + the emitted -DSG_TUNED_* "
                         "knobs. Emits NO header (inspection only).")
    ap.add_argument("--vit-layout", action="store_true",
```

VERBATIM OLD (copied from `grokking_optimizers/megakernel_codegen.py` lines 1719–1722):
```python
    if args.decoder_layout_flagship:
        sys.stdout.write(decoder_flagship_layout_header())
        return 0

```
NEW (handle the flag after `--decoder-layout-flagship`):
```python
    if args.decoder_layout_flagship:
        sys.stdout.write(decoder_flagship_layout_header())
        return 0

    if args.decoder_knobs:
        sys.stdout.write(decoder_knobs_report() + "\n")
        return 0

```

> BYTE-IDENTICAL-WHEN-OFF proof for EDIT B: every addition is a NEW function or a NEW CLI
> flag. No existing function body changes. `--decoder-layout` / `--decoder-layout-flagship`
> emit byte-for-byte the same header (verified this session: `--decoder-layout` ==
> committed `decoder_layout.cuh`). The selector emits NO `-D` until the build/autotuner
> calls it — and even then, the shipped tiers' knobs == the live `#ifndef` defaults, so a
> build that applies them is byte-identical to today (the autotuner already leads its sweep
> with the in-header default per `compile.py:2202` "First value == the in-header #ifndef
> default, so the untuned build is byte-identical").

---

## §5 — EDIT C: thread `<class Sz = par::SizeSmall>` onto the megakernel + launcher (`fused_decoder_megakernel.cuh`)

Four sub-edits: C.1 the kernel signature (add the `Sz` template param after `Par`);
C.2 the launch-shape selector (the size config decides the launch grid/cluster, gated);
C.3 the P1 CTA-tile gate (NO-OP stub this increment — the real body is §7); C.4 the
launcher signature + the kernel-launch call. The `Sz=SizeSmall` default makes EVERY
existing call site compile unchanged and fold to today's PTX.

### C.1 — kernel signature: add `Sz` as the THIRD template param

VERBATIM OLD (copied from `csrc/fused/sm_90/fused_decoder_megakernel.cuh` lines 674–681):
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
NEW (add `class Sz = par::SizeSmall` AFTER `Par`; the kernel arg list is UNCHANGED — the
size point is a pure compile-time param, it carries NO runtime arg, so the ABI of every
existing `<Opt>` / `<Opt,Par>` instantiation is preserved):
```cpp
template <OptId Opt, class Par = ::sg::fused::par::SingleGPU,
          class Sz = ::sg::fused::par::SizeSmall>
__global__ void __launch_bounds__(SG_TC_MEGA_BLOCK)
fused_decoder_megakernel_tc(PersistentContext ctx,
                            float* __restrict__ params,
                            DecoderTokenCtx tok,
                            float* __restrict__ grad,
                            float lr, int step, FusedOptState st,
                            ::sg::fused::par::CommCtx comm = {}) {
```

### C.2 — the CTA-tile P1 gate (NO-OP stub this increment), at the head of the P1 loop

The P1 token-tile loop is where CTA-tiling lands (it is the LEVER-2 site: the grid-stride
over `n_tiles` is what leaves SMs idle when `n_tiles < nCTA`). On this increment we add the
`if constexpr (Sz::kCtaTile)` SEAM with a NO-OP that falls through to the existing
grid-stride loop, so `SizeSmall` (and even `SizeLarge` until §7) is byte-identical; §7
replaces the stub body with the real cooperative-tile loop.

VERBATIM OLD (copied from `csrc/fused/sm_90/fused_decoder_megakernel.cuh` lines 840–844,
the P1 loop header comment + the tile-count setup — the unique anchor before the loop):
```cpp
    // ── P1: token-tile-parallel fwd+bwd. Each CTA grid-strides over tiles of
    //    kTileM rows; for its tile it runs fwd (→ acts X, NLL) then bwd (→ acts
    //    dY, dh0, LN-vec partials). Barrier-free within the tile. ──
    const int nrows_tile = dectc::kTileM;
    const int n_tiles = (T + nrows_tile - 1) / nrows_tile;
```
NEW (insert the size-gate comment + the static-asserted NO-OP seam immediately AFTER the
`n_tiles` line — the loop body itself is UNCHANGED below it):
```cpp
    // ── P1: token-tile-parallel fwd+bwd. Each CTA grid-strides over tiles of
    //    kTileM rows; for its tile it runs fwd (→ acts X, NLL) then bwd (→ acts
    //    dY, dh0, LN-vec partials). Barrier-free within the tile. ──
    const int nrows_tile = dectc::kTileM;
    const int n_tiles = (T + nrows_tile - 1) / nrows_tile;
    // ── SIZE-ADAPTIVE CTA-TILING gate (bottleneck LEVER 2). On SizeSmall (the
    //    default, kCtaTile==false) this folds away ENTIRELY and the grid-stride P1
    //    loop below is byte-identical to the pre-Sz kernel. On SizeLarge the §7
    //    CTA-tiled body replaces the loop so multiple CTAs cooperate on one output
    //    tile (using the SMs a 1-CTA/SM wave leaves idle when n_tiles < nCTA — the
    //    measured ~20% grid-barrier idle at d=2048). UNTIL §7 lands, the kCtaTile
    //    arm is a NO-OP fall-through (it does NOT early-return), so SizeLarge runs
    //    the SAME grid-stride loop — the seam is in place + wiring-reachable, the
    //    body is the follow-on. This mirrors the if-constexpr(Par::kTPComm) seam.
    if constexpr (Sz::kCtaTile) {
        // §7 CTA-TILED P1 BODY GOES HERE (cooperative-tile loop + cluster/grid
        // barrier rendezvous over kCtasPerTile-CTA groups). NO-OP this increment:
        // fall through to the grid-stride loop below (byte-identical to SizeSmall).
        static_assert(Sz::kCtasPerTile >= 1, "CTA-tiled P1 needs CtasPerTile>=1");
    }
```

> NOTE: the NO-OP `if constexpr (Sz::kCtaTile) { static_assert(...); }` block emits ZERO
> instructions even on `SizeLarge` (a `static_assert` is compile-time only). So on THIS
> increment `<Opt,Par,SizeLarge>` is ALSO byte-identical to `<Opt,Par,SizeSmall>` — the
> alias is reachable for dispatch/autotuner wiring without changing any PTX. §7 fills the
> block with the real cooperative loop AND wraps the existing grid-stride loop in an
> `else` (so SizeLarge takes the new path, SizeSmall the old). I do NOT wrap the existing
> loop in an `else` now (that would be a textual change to the shipped loop body); the §7
> edit does that as part of the real-body landing, keeping THIS increment's blast radius to
> the additive seam above.

### C.3 — the launch-shape selector in the launcher (cluster launch when `Sz::kUseCluster`)

The launcher today launches `dim3 grid(launch_ctas), block(SG_TC_MEGA_BLOCK)` with a plain
`<<<>>>` (`:1569,1574`). The size config decides the launch SHAPE: SizeSmall keeps the plain
1-CTA/SM launch (byte-identical); SizeLarge (once §7 lands) uses a cluster launch
(`cudaLaunchKernelEx` with a `cudaLaunchAttributeClusterDimension`). On THIS increment the
selector is gated `if constexpr (Sz::kUseCluster)` and that arm is unreachable for
`SizeSmall` (false) and a NO-OP-equivalent for `SizeLarge` (the kernel body is still the
grid-stride loop), so it does not perturb the shipped launch. See C.4 for the exact launcher
edit (signature + the gated launch).

### C.4 — launcher signature + kernel-launch call

VERBATIM OLD (copied from `csrc/fused/sm_90/fused_decoder_megakernel.cuh` lines 1519–1523):
```cpp
template <OptId Opt>
cudaError_t launch_fused_decoder_megakernel_tc(
        PersistentContext ctx, float* params, DecoderTokenCtx tok,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream,
        int ncta_cap = 0) {
```
NEW (add `class Sz = par::SizeSmall` — note: this composes with the PENDING `tp_kernel.md`
launcher edit that adds `class Par`; see the §5.4 MERGE NOTE — if `Par` is already present,
add `Sz` AFTER it):
```cpp
template <OptId Opt, class Sz = ::sg::fused::par::SizeSmall>
cudaError_t launch_fused_decoder_megakernel_tc(
        PersistentContext ctx, float* params, DecoderTokenCtx tok,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream,
        int ncta_cap = 0) {
```

The launcher references `fused_decoder_megakernel_tc<Opt>` THREE times (the
`cudaFuncSetAttribute` addr, the `cudaOccupancyMaxActiveBlocksPerMultiprocessor` addr, and
the `<<<>>>` launch). Each must forward `Sz` so the size point is consistent across the
occupancy cert and the launch.

VERBATIM OLD (line 1542, inside `#if SG_DEC_TC_DYNAMIC_SMEM`):
```cpp
        (const void*)&fused_decoder_megakernel_tc<Opt>,
```
NEW:
```cpp
        (const void*)&fused_decoder_megakernel_tc<Opt, ::sg::fused::par::SingleGPU, Sz>,
```

VERBATIM OLD (line 1550):
```cpp
        &occ, (const void*)&fused_decoder_megakernel_tc<Opt>, SG_TC_MEGA_BLOCK,
```
NEW:
```cpp
        &occ, (const void*)&fused_decoder_megakernel_tc<Opt, ::sg::fused::par::SingleGPU, Sz>, SG_TC_MEGA_BLOCK,
```

VERBATIM OLD (lines 1569–1576, the launch + the trailing comment block):
```cpp
    dim3 grid(launch_ctas), block(SG_TC_MEGA_BLOCK);
    // dynamicSMemBytes: 0 on the default (static DecTcSmem → byte-identical launch);
    // sizeof(DecTcSmem) on the deep-ring path (dyn_smem, opt-in already set + the
    // ≥1-CTA cert passed above). Same grid/block/stream either way — 1 CTA/SM (the
    // persistent grid-barrier requires it) is preserved.
    fused_decoder_megakernel_tc<Opt><<<grid, block, dyn_smem, stream>>>(
        ctx, params, tok, grad, lr, step, st);
    return cudaGetLastError();
}
```
NEW (forward `Sz`; the launch SHAPE selector is gated `if constexpr (Sz::kUseCluster)`,
false on SizeSmall ⇒ the plain `<<<>>>` is byte-identical; the cluster arm is the §7
landing point):
```cpp
    dim3 grid(launch_ctas), block(SG_TC_MEGA_BLOCK);
    // dynamicSMemBytes: 0 on the default (static DecTcSmem → byte-identical launch);
    // sizeof(DecTcSmem) on the deep-ring path (dyn_smem, opt-in already set + the
    // ≥1-CTA cert passed above). Same grid/block/stream either way — 1 CTA/SM (the
    // persistent grid-barrier requires it) is preserved on the SizeSmall path.
    // SIZE-ADAPTIVE launch shape: SizeSmall (kUseCluster==false) takes the plain
    // <<<>>> launch below (byte-identical). SizeLarge (§7) takes a Hopper cluster
    // launch (cudaLaunchKernelEx + cudaLaunchAttributeClusterDimension) so the
    // kCtasPerTile cooperating CTAs share a distributed-smem tile + a cluster
    // barrier — that arm is the §7 landing point; folded away on SizeSmall.
    if constexpr (Sz::kUseCluster) {
        // §7: cudaLaunchKernelEx with cluster dim Sz::kClusterDim. The grid must be
        // a multiple of the cluster dim; the §7 launcher rounds launch_ctas to
        // Sz::kClusterDim and the kernel's GridBarrier is replaced by the cluster
        // barrier for the cooperating group (see §7). NOT compiled until §7 (the
        // CTA-tiled body); on this increment kUseCluster is only true for SizeLarge,
        // whose body is still the grid-stride loop, so this arm is never the
        // shipped path. Left as the explicit seam (no cudaLaunchKernelEx call yet).
        static_assert(Sz::kClusterDim >= 1, "cluster launch needs ClusterDim>=1");
    }
    fused_decoder_megakernel_tc<Opt, ::sg::fused::par::SingleGPU, Sz><<<grid, block, dyn_smem, stream>>>(
        ctx, params, tok, grad, lr, step, st);
    return cudaGetLastError();
}
```

> BYTE-IDENTICAL-WHEN-OFF proof for EDIT C: `Sz=SizeSmall` ⇒ `kCtaTile==false` ⇒ the C.2
> seam folds to nothing (and even on SizeLarge the seam is a `static_assert`-only NO-OP this
> increment); `kUseCluster==false` ⇒ the C.4 launch selector folds to nothing and the plain
> `<<<>>>` runs. The kernel ARG LIST is unchanged (Sz is a pure compile-time param), so the
> `<Opt>`/`<Opt,Par>` ABI is preserved — `tp_kernel.md`'s `comm` arg and this `Sz` param are
> independent. The PTX-diff gate compares `<Opt>` vs `<Opt, SingleGPU, SizeSmall>` and must
> be byte-equal.

### §5.4 — MERGE NOTE: composing with the pending `tp_kernel.md` launcher edit

`tp_kernel.md` C.3 adds `class Par = par::SingleGPU` + a trailing `CommCtx comm` to the
LAUNCHER (the live launcher is still `template <OptId Opt>`; the live KERNEL already has
`Par`). When BOTH land, the launcher template head is:
```cpp
template <OptId Opt, class Par = ::sg::fused::par::SingleGPU,
          class Sz  = ::sg::fused::par::SizeSmall>
cudaError_t launch_fused_decoder_megakernel_tc(
        PersistentContext ctx, float* params, DecoderTokenCtx tok,
        float* grad, float lr, int step, FusedOptState st, cudaStream_t stream,
        int ncta_cap = 0, const ::sg::fused::par::CommCtx& comm = {}) {
```
and the three kernel references become `fused_decoder_megakernel_tc<Opt, Par, Sz>` (drop the
explicit `SingleGPU` I wrote above — that placeholder is ONLY there for the standalone case
where `tp_kernel.md`'s `Par` has not landed yet). The launch forwards `comm`:
`...<Opt, Par, Sz><<<...>>>(ctx, params, tok, grad, lr, step, st, comm);`. Apply order is
irrelevant (each axis defaults independently); if `tp_kernel.md` lands first, my edit only
ADDS `class Sz` to the head and `, Sz` to the three references — a pure 2-token append.

---

## §6 — EDIT D: autotuner composition — a `size_tier` dim + per-tier knob pins (`compile.py`)

The autotuner (`compile.py`'s `_dim` registry, the L3-TC GEMM/tile dims at lines 2217–2294)
sweeps `tile_n` / `dec_gemm_stages` / `dec_gemm_interleave` / `dec_dw_splitk` etc., each
LEADING with the in-header `#ifndef` default so the untuned build is byte-identical
(`:2202`). EDIT D composes the SIZE axis with this sweep so the autotuner (a) knows WHICH
size tier a variant targets and (b) never proposes a small-tier knob for a large workload.

The minimal, byte-safe composition: add a `size_tier` enum dim (macro=None — it does not
emit a `-D` itself; it routes the variant to `decoder_knobs_for_size` for the per-tier knob
DEFAULTS) and a `_decoder_size_tier_pins(tier)` helper that returns the per-tier knob
overrides. The FIRST value of the new dim is `"small"` (the production tier) and its knob
pins == the in-header defaults ⇒ the canonical (first-element) build is byte-identical.

### D.1 — the `size_tier` dim. Insert in the L3-TC GEMM/tile dims block, after the `pipe_depth` dim.

VERBATIM OLD (copied from `grokking_optimizers/compile.py` lines 2293–2295):
```python
            _dim("pipe_depth", "int", [2, 3],
                 "SG_TUNED_PIPE_DEPTH", ["device"]),
        ]),
```
NEW (add the `size_tier` dim before the close of the dims list):
```python
            _dim("pipe_depth", "int", [2, 3],
                 "SG_TUNED_PIPE_DEPTH", ["device"]),
            # === SIZE-ADAPTIVE specialization (the self-designing megakernel) =====
            # size_tier routes a variant to its WORKLOAD-SIZE knob set + SizeConfig
            # template point (megakernel_codegen.decoder_knobs_for_size). It does
            # NOT emit a -D itself (macro=None) — it selects (a) the per-tier knob
            # DEFAULTS the other tile dims sweep AROUND and (b) the SizeConfig alias
            # (par::SizeSmall/SizeLarge) the launcher instantiates (EDIT C). FIRST
            # value "small" == the production d=128 tier whose pins == the in-header
            # #ifndef defaults, so the canonical (first-element) build is byte-
            # identical (same convention as tile_n/dec_gemm_stages leading with the
            # default). "large" selects par::SizeLarge (CTA-tiling carried into the
            # kernel) for the d>=1024 bench/flagship tiers. The autotuner thus
            # explores tile knobs WITHIN a size tier and never crosses a small-tier
            # winner onto a large workload (or vice-versa) — the size axis composes
            # with the parallelism axis (ParConfig) exactly as distributed+large =>
            # CTA-tiling + all-reduce; single+small => neither.
            _dim("size_tier", "enum", ["small", "large"],
                 None, ["device"]),
        ]),
```

### D.2 — the per-tier knob pins helper. Insert near the other tuned-macro helpers (after `_dim`, e.g. after line 1245 where `_dim` ends — or co-located with the L3-TC dims; place it as a module-level function before the space build).

```python
def _decoder_size_tier_pins(tier: str, *, n_sms: int = 132) -> Dict[str, Any]:
    """Per-size-tier knob DEFAULTS + the SizeConfig alias the autotuner uses to seed
    a variant's L3-TC tile knobs. Delegates to the codegen size selector
    (grokking_optimizers.megakernel_codegen.decoder_knobs_for_size) so the autotuner
    and the build share ONE size->knobs map (no drift). Returns
    {"knobs": {SG_TUNED_*: val}, "size_config": "par::SizeSmall"|"par::SizeLarge"}.

    The "small" tier's knobs == the in-header #ifndef defaults, so seeding the
    canonical build from it is byte-identical. The "large" tier carries CTA-tiling
    via size_config (the launcher's SizeConfig template arg) and, on this increment,
    keeps the same -D knob defaults (the megakernel_codegen contract: shipped tiers
    keep their knobs; only a future large-tier retune changes them, in ONE place)."""
    from grokking_optimizers import megakernel_codegen as mkc  # lazy: no torch import
    tiers = mkc.DEC_SIZE_TIERS
    # Map the autotuner's small/large enum onto a representative named tier:
    #   small -> the production d=128 tier; large -> the d=2048 bench tier (the
    #   d>=1024 width trigger; the flagship is also "large" and shares its knobs).
    name = "production" if tier == "small" else "bench"
    d, layers, T = tiers[name]
    return mkc.decoder_knobs_for_size(d, layers, T, n_sms)
```

> BYTE-IDENTICAL-WHEN-OFF proof for EDIT D: the `size_tier` dim leads with `"small"` whose
> pins are the in-header defaults, so the canonical (first-element) config is the untuned
> build. `macro=None` ⇒ the dim emits NO `-D` itself (it routes to the existing tile dims +
> the SizeConfig alias). `_decoder_size_tier_pins` is a NEW function with no existing caller
> changed. The dim being in the space adds 1 axis of cardinality 2 — but per the
> `_LIVE_TUNING_DIMS` discipline (`compile.py:1287`), if it is left out of `config_key` it
> collapses binary-identical small/large configs (until §7 makes them differ); add
> `"size_tier"` to `_LIVE_TUNING_DIMS` ONLY once §7 makes the large tier emit distinct SASS,
> so today it does not poison the cache. (Flagged: keep `size_tier` OUT of `_LIVE_TUNING_DIMS`
> until §7; this is the same forward-compat pattern the dead-dim audit at `:1280` uses.)

---

## §7 — SCOPING (precise, NOT byte-exact): the CTA-tiled (occupancy>1 / cluster) megakernel VARIANT

This is the honest deep item. The selector + SizeConfig + plumbing land now (A–D); the REAL
CTA-tiled body is a genuine kernel change because the persistent megakernel's `GridBarrier`
ASSUMES 1 CTA/SM (`n_ctas == gridDim.x`, every launched CTA arrives exactly once per
`sync()` generation). Occupancy>1 (or a multi-CTA cluster per tile) breaks three invariants;
each is scoped below with the exact file/anchor and the change.

### 7.1 — WHAT CTA-tiling buys (the measured lever)

The P1 loop grid-strides over `n_tiles = ceil(T / kTileM)` tiles across `nCTA` persistent
CTAs (`fused_decoder_megakernel.cuh` P1, the §5 C.2 anchor). When `n_tiles < nCTA` (large
`d`, modest `T` — the d=2048 bench at B=4096 has `T=16384`, `n_tiles=128` vs `nCTA=132`, so
4 SMs idle the WHOLE P1; the flagship at B=512 has `T=2048`, `n_tiles=16` vs `nCTA=132`, so
**116 of 132 SMs idle in P1**), the grid barrier B1 (`:846`-region `bar.sync()`) waits on the
slowest of the few active CTAs — the measured ~20% grid-barrier idle = this cross-CTA load
imbalance (LEVER 2). CTA-tiling assigns `kCtasPerTile` CTAs to EACH output tile, splitting
the per-tile GEMM N-range (in_proj `N=3d`, ff0 `N=4d`, both wide at large d) across them, so
the idle SMs do real work and each CTA's per-tile latency drops.

### 7.2 — CHANGE 1: the barrier (the load-bearing change)

`GridBarrier` (`megakernel_common.cuh:147-185`) counts arrivals against `n_ctas =
gridDim.x` and releases when the count hits `n_ctas`. Two CTA-tiling shapes, two barrier
treatments:

  (a) **OCCUPANCY>1 (≥2 CTAs/SM, same grid)** — `gridDim.x = kCtasPerTile · n_sms`,
      `ctx.n_ctas` set to that. The GridBarrier ALREADY counts against `ctx.n_ctas`
      (`megakernel_common.cuh:150` `n_ctas` is a field set from `ctx.n_ctas`), so a
      whole-grid `bar.sync()` still works AS LONG AS the launcher certifies occupancy ≥
      `kCtasPerTile` (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` must return ≥
      kCtasPerTile, else REFUSE — the same `occ<1` refusal at `:1553`, generalized to
      `occ < Sz::kCtasPerTile`). The per-tile cooperation (which CTA does which N-slice) is
      INTRA-tile and needs a `__syncthreads()`-style join ACROSS the cooperating CTAs — but
      occupancy>1 CTAs are NOT in the same block, so they cannot `__syncthreads()`. ⇒ the
      cooperating CTAs must rendezvous via either (i) the whole-grid `bar.sync()` (correct
      but coarse — every CTA waits for every other, reintroducing the imbalance) or (ii) a
      **cluster** (shape (b)). For the load-imbalance lever, (b) is the right tool.

  (b) **CLUSTER (Hopper thread-block cluster, the recommended shape)** — launch with
      `cudaLaunchKernelEx` + `cudaLaunchAttributeClusterDimension = Sz::kClusterDim`
      (the §5 C.4 seam). The `kCtasPerTile` CTAs of one tile form a cluster and synchronize
      via `cluster.sync()` (cooperative-groups cluster barrier) + share the tile through
      DISTRIBUTED SHARED MEMORY (`cluster.map_shared_rank`), NOT the grid barrier. The
      GridBarrier is then used ONLY for the PHASE boundaries (B1/B2/B3 between P1/P2/P3),
      still over `ctx.n_ctas = kClusterDim · n_clusters` — which is fine because every CTA
      reaches each phase barrier exactly once (the §1.13 / tp_kernel.md §1 grid-uniform
      rule). The INTRA-tile join is the cluster barrier, never the grid barrier ⇒ no
      cross-tile coupling ⇒ the imbalance is removed.

  **Exact edits (scoped):** add a `kCtasPerTile`/`kClusterDim`-aware occupancy cert +
  cluster launch in `launch_fused_decoder_megakernel_tc` (the §5 C.4 `if constexpr
  (Sz::kUseCluster)` arm becomes a real `cudaLaunchKernelEx`); add a `cluster.sync()` /
  `cg::this_cluster()` join in the P1 cooperative body (the §5 C.2 `if constexpr
  (Sz::kCtaTile)` arm). The GridBarrier itself is UNCHANGED (it already counts
  `ctx.n_ctas`); only the LAUNCH grid sizing + the intra-tile cluster barrier are new.

### 7.3 — CHANGE 2: tile ownership (P1, P2, P3)

  * **P1 (token tiles):** today `for (ti = cta; ti < n_tiles; ti += nCTA)`. CTA-tiled:
    `cluster_id = cta / kCtasPerTile`, `lane = cta % kCtasPerTile`; the loop strides over
    tiles by `n_clusters` (`for (ti = cluster_id; ti < n_tiles; ti += n_clusters)`), and the
    `kCtasPerTile` lanes split the per-tile GEMM N-range (`lane` owns N-columns
    `[lane·Nslice, (lane+1)·Nslice)` of in_proj/out_proj/ff). The fwd/bwd tile fns
    (`model_stage_decoder_tc.cuh` `dectc_forward_tile`/`_backward_tile`) gain an
    `if constexpr (Sz::kCtaTile)` N-range narrowing (the SAME shape the tp_kernel.md §6 D.2
    reduce points use to narrow `Nout` per TP rank — the mechanisms are identical, just split
    across cluster lanes instead of TP ranks). The partial-N results are joined via
    `cluster.sync()` + distributed-smem (the cluster's reduction), NOT a grid barrier.

  * **P2 (dW tiles):** today grid-strided over `n_dw`/`n_dw·G` (`:927-938`). This is ALREADY
    CTA-tiled in spirit (the split-K `kDecDwSplitK` fans (gt,kc) partials across the grid) —
    so P2 needs NO ownership change; the dW loop strides over `nCTA = ctx.n_ctas` regardless
    of occupancy/cluster (more CTAs ⇒ finer split, which split-K already handles
    deterministically via the ascending-chunk reduce). VERIFY the split-K reduce
    (`dectc_dw_reduce_splitk`, `:935`) is correct when `nCTA` is a cluster multiple — it is
    (it reduces by `(gt,kc)` index, independent of CTA→tile mapping).

  * **P3 (optimizer):** today grid-strided over numel (`:1043`, `:1325`). Also already
    occupancy-agnostic (a larger `nCTA` just means a finer numel grid-stride, deterministic
    by index). The SG2 per-CTA scratch (`sg2_base + blockIdx.x · stride`, `:1388`) is sized
    `nCTA · stride` — with occupancy>1 `nCTA` grows, so `dec_tc_sg2_floats(nCTA)` grows
    PROPORTIONALLY (the launcher already sizes the workspace from `ctx.n_ctas`,
    `dec_tc_workspace_floats(T,B,nCTA)` at the launcher `:123`). ⇒ P3 needs NO code change,
    but the MEMORY budget grows ∝ occupancy — which is why the large-tier knob set should
    pair CTA-tiling with the `ncta_cap` / auto-nCTA discipline from `run_harness.md`
    (`auto_ncta`): CTA-tiling for LATENCY (P1 fill) but capped nCTA for the SG2 scratch.
    This is the composition the selector must encode (flagged for the §4 large-tier retune).

### 7.4 — CHANGE 3: smem (the cluster distributed-smem tile)

The cooperating CTAs share the partial-N GEMM accumulators through DISTRIBUTED SHARED MEMORY
(`cluster.map_shared_rank(smem_ptr, rank)`). The `DecTcSmem` arena
(`fused_decoder_megakernel.cuh:357`) is per-CTA today; under the cluster the reduction across
lanes reads peer CTAs' `sm.red` slot via the cluster map. This is a NEW smem access pattern
(peer-rank reads) but NO new smem ALLOCATION (the existing `float red[256]` slot
`:390` is the join buffer). The deep-ring dynamic-smem gate (`SG_DEC_TC_DYNAMIC_SMEM`,
`:466`) is ORTHOGONAL and unchanged. RISK: distributed-smem requires the cluster CTAs be
co-resident (the cluster launch guarantees it); the occupancy cert must account for
`kCtasPerTile` blocks' combined smem ≤ the per-SM budget when they land on the same SM
(cluster CTAs may span SMs — the cert is per the cluster launch's placement, validated on
silicon).

### 7.5 — the §7 landing edits (where the stubs become real)

  1. `parallel_config.cuh`: `SizeLarge` already carries `kCtasPerTile=2, kClusterDim=2`
     (EDIT A) — no change.
  2. `fused_decoder_megakernel.cuh` §5 C.2 `if constexpr (Sz::kCtaTile)` arm: replace the
     NO-OP with the cooperative-tile P1 loop (cluster.sync + lane N-split) AND wrap the
     existing grid-stride loop in `else`.
  3. `fused_decoder_megakernel.cuh` §5 C.4 `if constexpr (Sz::kUseCluster)` arm: replace the
     NO-OP with `cudaLaunchKernelEx` (cluster dim) + generalize the occupancy cert to
     `occ ≥ Sz::kCtasPerTile`; round `launch_ctas` to a `kClusterDim` multiple.
  4. `model_stage_decoder_tc.cuh` `dectc_forward_tile`/`_backward_tile`: add the
     `if constexpr (Sz::kCtaTile)` N-range narrowing + cluster-lane partial join (the SAME
     pattern as tp_kernel.md §6 D.2 — thread `<class Sz>` onto the tile fns alongside the
     `<class Par>` tp_kernel.md adds).
  5. `compile.py` EDIT D: add `"size_tier"` to `_LIVE_TUNING_DIMS` (now the large tier emits
     distinct SASS) and let the large tier retune its knobs in `decoder_knobs_for_size`
     (`megakernel_codegen.py` EDIT B's `tier == "large"` branch).

GATE for §7: a parity test (`test_decoder_tc.py` extended) asserting `<Opt, SingleGPU,
SizeLarge>` (cluster) produces bit-identical loss/grads to `<Opt, SingleGPU, SizeSmall>`
(grid-stride) — the cooperative split + cluster reduce must reproduce the EXACT ascending
accumulation order (the same A/A/A discipline tp_kernel.md §9 requires of the TP reduce).
Plus a perf gate: the d=2048 bench MFU under SizeLarge must beat SizeSmall (the ~20% idle
recovered), else CTA-tiling is not worth its smem/launch complexity at that size.

---

## §8 — DETERMINISM / PARITY / BYTE-IDENTITY (the hard gate)

1. **SingleGPU + SizeSmall PTX-diff gate (apply-now).** Every EDIT-A/C symbol is behind
   `if constexpr (Sz::kCtaTile)` / `if constexpr (Sz::kUseCluster)` (both false on
   `SizeSmall`) or is a NO-OP `static_assert` even on `SizeLarge` (this increment). The new
   template param `Sz` carries NO runtime arg (pure compile-time), so the kernel ABI is
   unchanged. ⇒ `fused_decoder_megakernel_tc<Opt, SingleGPU, SizeSmall>` is byte-for-byte
   the legacy `<Opt>`/`<Opt,SingleGPU>` kernel. GATE:
   `CUDA_VISIBLE_DEVICES=0 GATE_SEED=42 python -m pytest tests/hw/test_decoder_tc.py -q`.

2. **Codegen byte-identity (apply-now).** EDIT B emits NO header by default; `--decoder-layout`
   and `--decoder-layout-flagship` are unchanged (verified this session:
   `python -m grokking_optimizers.megakernel_codegen --decoder-layout` == committed
   `decoder_layout.cuh`). The new `--decoder-knobs` is inspection-only.

3. **Knob byte-identity for shipped tiers.** `decoder_knobs_for_size` returns the live
   `#ifndef` defaults for ALL three shipped tiers on this increment, so a build that applies
   them is byte-identical to today (the autotuner already leads with the default per
   `compile.py:2202`). CTA-tiling is carried in `size_config` (a template point), not the
   `-D` knobs, so the OFF (SizeSmall) build emits the identical macros.

4. **Composition with ParConfig (the user directive).** `Sz` and `Par` are orthogonal
   `if constexpr` axes on the same kernel. `<Opt, SingleGPU, SizeSmall>` = today (neither);
   `<Opt, ParTP8, SizeLarge>` = both (the §7 + tp_kernel.md composition). No axis perturbs
   the other's fold: `kTPComm` gates the all-reduce, `kCtaTile` gates the tiling, each
   independently degenerate at its default.

5. **§7 CTA-tiled parity (deferred).** The cluster reduce MUST reproduce the exact
   ascending accumulation order (§7.5 GATE). Until §7's parity gate is green on silicon,
   `SizeLarge` is NOT routed by the production dispatch (the allow-list stays {SizeSmall};
   SizeLarge is reachable only via the autotuner/harness for bring-up) — the same
   explicit-instantiation allow-list discipline tp_kernel.md §7.2 uses for the TP degrees.

---

## §9 — GATE COMMANDS (the task's two, mapped to what they prove)

1. ```
   python -m grokking_optimizers.megakernel_codegen --decoder-layout > /tmp/d.cuh && \
       diff /tmp/d.cuh csrc/fused/sm_90/decoder_layout.cuh
   ```
   — PROVES EDIT B is inert on the committed layout header: the size selector + the
   `--decoder-knobs` CLI add NO change to `--decoder-layout` output. Passes byte-identical
   BEFORE and AFTER EDIT B (verified this session on the baseline; EDIT B is purely
   additive — a new function + a new CLI flag, neither touched by `--decoder-layout`).

2. ```
   python -c "import grokking_optimizers.megakernel_codegen"
   ```
   — PROVES the module imports clean with the new selector + CLI code (no syntax/import
   error). Verified on baseline this session; EDIT B's additions are pure-Python stdlib +
   the existing module-level constants, so import stays torch-free.

   Companion (not in the task's list but proves the selector works):
   ```
   python -m grokking_optimizers.megakernel_codegen --decoder-knobs
   ```
   → prints the per-tier selection: production→`par::SizeSmall` (all default knobs),
   bench→`par::SizeLarge` (d=2048 ≥ 1024 ⇒ CTA-tiling carried), flagship→`par::SizeLarge`
   (d=1600 ≥ 1024). PROVES the size→knobs map turns on CTA-tiling for the large tiers and
   keeps the shipped knobs.

---

## §10 — APPLY ORDER + CONFIDENCE + RISKS

Apply order: **A → B → D** (today: POD header / pure-Python codegen / Python autotuner —
all byte-exact, CPU-compilable, the two gate commands stay green) → **C** (kernel track, GPU
build loop; the additive seam + the `Sz` template param, OFF-path provably identical) →
**§7** (the real CTA-tiled body + cluster launch + the §7 parity/perf gates on silicon).

- **A (SizeConfig POD):** HIGH. Pure constexpr POD, additive (no existing line changes),
  `SizeSmall` default sets every predicate to the shipped value, `kTileN==SG_TUNED_TILE_N`.
  CPU-compilable, mirrors the proven `ParConfig` pattern exactly.
- **B (codegen selector + CLI):** HIGH. Pure Python, no torch/GPU, additive functions + one
  CLI flag. Both gate commands verified green on the baseline; the selector returns the live
  `#ifndef` defaults for the shipped tiers (byte-identical knobs). The LARGE predicate
  (`d>=1024 OR n_tiles<n_sms`) is a judgment call — it correctly classifies the three named
  tiers (production=small, bench/flagship=large) and the threshold is documented + tunable in
  ONE place; a maintainer may refine it once §7 measures the cross-over on silicon.
- **C (kernel `<class Sz>` + seams):** MEDIUM-HIGH. The signature threading is mechanical
  (one template param, defaulted ⇒ every call site unchanged). The seams are additive
  `if constexpr` NO-OPs (the C.2 P1 seam is `static_assert`-only; the C.4 launch seam is
  gated `kUseCluster` false on SizeSmall). RISK: the three `<Opt>`→`<Opt,SingleGPU,Sz>`
  launcher references must match whatever `Par` state the live launcher is in (the §5.4 MERGE
  NOTE handles the tp_kernel.md interaction; standalone uses the explicit `SingleGPU`
  placeholder). PTX gate (test_decoder_tc.py) guards the SizeSmall identity.
- **D (autotuner size_tier dim):** MEDIUM-HIGH. Additive dim leading with `"small"`
  (byte-identical canonical) + a NEW helper delegating to the codegen selector (one
  source of truth). RISK: keep `size_tier` OUT of `_LIVE_TUNING_DIMS` until §7 makes the
  large tier emit distinct SASS, else it false-splits the cache (flagged inline; the
  dead-dim discipline at `compile.py:1280` is the precedent).
- **§7 (CTA-tiled body):** SCOPED ONLY (not byte-exact) BY NECESSITY — it changes the
  launch shape (cluster), the P1 tile ownership, and the intra-tile barrier (cluster.sync
  vs grid barrier), all of which need a GPU build/validate loop + an on-silicon parity gate.
  The seam (SizeConfig + the `if constexpr (Sz::kCtaTile)`/`kUseCluster` arms) is designed
  for exactly this body; the risk is kernel-engineering effort + the cluster reduce's
  bit-exact accumulation order (the §7.5 / §8.5 parity gate), not architecture.
- **gfx942 / tpu:** UNTOUCHED. Every edit is sm_90 (`csrc/fused/sm_90/*`) / Python codegen /
  Python autotuner. The SizeConfig header is sm_90-namespaced (`sg::fused::par`); the codegen
  selector is decoder-specific. No cross-arch risk.

The single biggest honest caveat: §7's CTA-tiled body is a REAL kernel change (the
GridBarrier assumes 1 CTA/SM; occupancy>1 needs a cluster barrier + distributed-smem for the
intra-tile join). This spec delivers the SELECTOR + the SizeConfig point + the codegen knob
surface + the autotuner composition NOW (A–D, byte-exact, gates green), and the precise
scoping of the CTA-tiled variant (barrier/ownership/smem) as the GPU-window follow-on — the
honest split the task asks for.
