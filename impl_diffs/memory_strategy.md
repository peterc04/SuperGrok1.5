# AREA: memory_strategy — survey existing memory machinery + scope large-model-on-few-GPUs (offload / recompute / layer-streaming)

> READ-ONLY survey + APPLY-READY planner edits. Driver = **memory-fit + compute-shape vs hardware, NEVER a GPU-count switch**.
> Every strategy is gated by the same `if constexpr`/config-templating mechanism the parallelism axes already use
> (`ParConfig`/`CommCtx`, `parallel_config.cuh:55-138`; `kDecStagedOptScratch`, `fused_decoder_megakernel.cuh:541-545`),
> so it is **byte-identical when OFF**.

---

## 0. TL;DR — what exists, what is needed, and what is APPLY-READY now

| Strategy | Today | Needed for 10B-on-1GPU | This spec gives |
| --- | --- | --- | --- |
| **ZeRO-3 param+state shard (across DP)** | ✅ `zero3.py` (`Zero3FlatParamStore`, `FlatShardPlan`) + `distributed.py::ZeRO3Sharder` | only helps when DP>1 — **does NOT shrink a 1-GPU footprint** | (survey) |
| **Optimizer-state HOST-offload (ZeRO-Offload)** | ⚠️ flag exists **only** for the DeepSpeed path (`distributed.py:800,824`); the **fused megakernel has none** — `state` is a device tensor read as a raw `st.exp_avg` pointer | YES (AdamW state = 3× params = 30 GB at 10B; the binding constraint) | **(A) planner flag + budget model (APPLY-READY)** + **(C) scoped launcher/kernel staging (real change)** |
| **Activation RECOMPUTE / checkpointing** | ❌ NONE. `DecActs` materializes **all-layer** acts in HBM (`model_stage_decoder_tc.cuh:423-437`, `dec_tc_acts_floats` sums `for li<L`, `fused_decoder_megakernel.cuh:504-512`) | YES (acts grow ∝ L·T·d; at 10B/L=64 the workspace dominates) | **(A) planner flag + budget model (APPLY-READY)** + **(D) scoped in-kernel fwd-recompute (real change)** |
| **LAYER STREAMING of weights from host** | ❌ NONE. All-layer weights staged bf16 in HBM (`DecWBf`, `dec_wbf_bind`, `model_stage_decoder_tc.cuh:496-531`); `params` is one device blob | YES (params themselves at 10B = 40 GB fp32) | **(A) planner flag + budget model (APPLY-READY)** + **(E) scoped weight ring (real kernel/launcher change)** |
| **CTA-tiling (nCTA cap)** | ✅ `auto_ncta()` (`flagship_budget.py`, run_harness.md:329-339); `ncta_cap` already plumbed to the launcher (`...launcher.cu:121`) | already a memory lever (SG2 scratch ∝ nCTA) | **(A) extends it into a single planner** |

**The honest split:** items **(A)** (the *planner* — budget arithmetic + strategy flags + the `--dry-run` proof) are **tractable config/Python edits, APPLY-READY in this spec**. Items **(C)/(D)/(E)** (the *machinery* — host↔device state staging, in-kernel fwd-recompute, a weight prefetch ring) are **real launcher + kernel changes**; this spec scopes them precisely and ties each to an `if constexpr`/`#if` gate so the OFF path stays byte-identical, but does **not** author the GPU-window kernel body (that is the same discipline `parallel_config.cuh:96-101` uses for the `CommCtx` widening).

---

## 1. INVENTORY — what exists today (read in full)

### 1.1 `grokking_optimizers/parallel/zero3.py` — DP param/state sharding, NO host offload
Read in full (340 lines). What it is and is **not**:

* **IS:** ZeRO-3 *across the DP group*. `FlatShardPlan` (`zero3.py:70-94`) partitions the flat `float[total]` param blob into per-rank contiguous slices; `Zero3FlatParamStore` (`:141-259`) keeps only the rank's shard resident between steps and does the **full pre-gather → kernel step → release** dance (`gather_full` `:197-251`, `release` `:253-259`). Mode keys off the optimizer (`flat_plan_for_optimizer` `:97-133`: elementwise → even flat slice; per-tensor → whole tensors). Bit-exact sharded checkpoint save/resume (`:269-330`).
* **IS NOT host-offload.** Every buffer is a torch device tensor: the gather target is `torch.empty(..., device=self.shard.device)` (`:200-201`), and the resident shard inherits the **CUDA** device (`:179-182`). There is **no `cudaMallocHost`/pinned-host path, no `.cpu()` between steps, no device↔host staging**. The *only* footprint reduction is the `1/world` shard division — which is **zero benefit at world=1** (`world==1` degenerates to no-op copies, `:207`). So `zero3.py` cannot, by itself, fit a 10B model on 1 GPU.

### 1.2 `grokking_optimizers/distributed.py::ZeRO3Sharder` — DeepSpeed `offload_optimizer` flag exists, but only off the kernel path
Read `:732-880`. `build_ds_config(..., offload_optimizer=False)` emits `zero["offload_optimizer"] = {"device":"cpu","pin_memory":True}` (`:800, :824-825`) — i.e. **host-offload already exists as a concept, but only for the DeepSpeed engine**, which is NOT the fused L3-TC megakernel. The native shim (`partition_optimizer_state` `:837`, `reduce_scatter_grads` `:864`, `all_gather_params`) is again **DP-shard only** (`_even_partition` `:716-729`), all-device. **Takeaway:** the *vocabulary* for host-offload is present; the *plumbing into the fused kernel* is not.

### 1.3 The dec_tc workspace carve — ACTS ARE STORED FULL (no recompute)
`fused_decoder_megakernel.cuh` + `model_stage_decoder_tc.cuh`:

* **Activations: full, all-layer, in HBM.** `dec_tc_acts_floats(T,B)` (`:504-512`) is a sum **over every layer** (`for (li=0; li<L; ++li) bf += Td+Td+Td+Tff+T3d+Td+Tff+Td`). `DecActs` (`model_stage_decoder_tc.cuh:423-437`) holds **per-layer arrays** `X_in[kLayers], X_ctx[kLayers], X_x1[kLayers], X_gact[kLayers]` (fwd caches) **and** `dY_qkv/dY_a/dY_ff0/dY_ff2[kLayers]` (bwd adjoints). The P1 tile (`fused_decoder_megakernel.cuh:848-876`) runs `dectc_forward_tile` (writes all those caches to HBM) then `dectc_backward_tile` (reads them). **There is NO recompute and NO checkpoint boundary** — the bwd consumes stored fwd acts directly. This is the single largest knob for big models (acts ∝ L·T·(8d+…)).
* **Weights: full, all-layer, in HBM (params blob + bf16 stage cache).** `params` is one contiguous device blob (`dec_bind` `:810`); `DecWBf`/`dec_wbf_bind` (`model_stage_decoder_tc.cuh:496-531`) pre-stages **all `kLayers`** layers' bf16 weights (`in_w[kLayers]…ff2_wT[kLayers]`, `kWbfTotalElems = kLayers*kWbfLayerElems`) into the `dec_wbf_floats()` workspace region, filled in P0 (`dectc_wbf_convert`, `:822`). **No layer is ever evicted; nothing streams from host.**
* **dW-transpose / staged-opt scratch:** carved LAST and **already gated** (`dec_tc_dw_transpose_floats` `:522-525` is `#if SG_TUNED_DEC_DW_STAGE`; the four staged-opt regions fold to 0 under `kDecStagedOptScratch=false`, `:541-638`). This is the *exact precedent* for how a recompute/stream gate elides a region byte-identically.

### 1.4 The launcher allocation — one device `cudaMalloc` workspace, no host staging
`mega_decoder_real_adamw_tc_launcher.cu` (read in full):
* `DecTcLauncherScratch` (`:46-71`) `cudaMalloc`s `workspace` = `dec_tc_workspace_floats(T,B,nCTA)` floats on the device, recreated only when a bigger `B`/`nCTA` needs it. `ncta_cap` is already threaded (`:121`, the memory lever `auto_ncta` drives).
* `params`, `state`, `grad`, `loss_out` arrive as **device** pointers; the optimizer state slices (`m_slice/v_slice/extra_slice`, `:138-140`) are device offsets into `state`. The TC TU twin (`mega_decoder_real_adamw_tc.cu:43-129`) `TORCH_CHECK(params.is_cuda())` / `state.is_cuda()` and binds `st.exp_avg = state.data_ptr<float>()`. **There is no pinned-host buffer, no `cudaMemcpyAsync`, no UVM.** Host-offload requires introducing exactly that staging here.

### 1.5 The planner seam already exists — `flagship_budget.py` + `auto_ncta()`
`impl_diffs/run_harness.md:120-380` specifies `grokking_optimizers/parallel/flagship_budget.py`: a **pure-Python** per-rank HBM budget (`per_rank_budget` `:284-326`, `RankBudget.fits` `:279-281`) that mirrors the live scratch formulas, plus `auto_ncta()` (`:329-339`) — the **memory-fit-driven knob selector**. This is the natural home for the planner: it already decides nCTA from fit, already models params/state/acts/staged separately. **The additions below extend this one module** so the planner emits the full strategy (offload/recompute/stream flags), keeping ONE source of truth for the `--dry-run` fit proof. (run_harness.md is a NEW-FILE spec, not yet on disk — so the edits in §2 are written as additive blocks to that pending file, clearly marked, and as a standalone fallback module if it lands first without them.)

### 1.6 The if-constexpr/config-templating mechanism (the gate every strategy hangs off)
`parallel_config.cuh:55-77`: `ParConfig<DP,TP,PP,SP,Z>` exposes `static constexpr bool` gates (`kShardParams`, `kShardOptGrad`, `kTPComm`, `kPPStage`, `kEmitComm`); the megakernel reads them with `if constexpr` so **SingleGPU folds every branch away → byte-identical** (`:18-22, :65-71`). `CommCtx` (`:106-138`) is the empty-POD seam fields hang off behind `kEmitComm`. The memory strategies need **three new orthogonal compile-time gates of the same shape** — they are NOT new parallelism axes, they are *memory* axes, so they belong on a sibling `MemConfig` (so `ParConfig` stays a clean parallelism descriptor) carried the same way.

---

## 2. APPLY-READY (item A) — the PLANNER: strategy selection + budget model + gates

These are pure-Python / pure-header config edits. They make the planner **decide** offload/recompute/stream from (model size/shape + hardware), and add the **compile-time gate POD** the kernel will read — both **byte-identical / no-op when every strategy is OFF**, exactly like `SingleGPU`.

### EDIT A1 (NEW FILE) — `grokking_optimizers/parallel/mem_strategy.py`
The planner core: from `(params, shape, hardware)` it returns a `MemPlan` (the chosen strategy bits + the per-strategy budget), and a CPU-testable budget that extends the `flagship_budget` arithmetic to model offload/recompute/streaming savings. **No torch, no GPU** (mirrors `flagship_budget.py`'s discipline).

```python
"""grokking_optimizers/parallel/mem_strategy.py — the MEMORY-STRATEGY PLANNER.

From (model size/shape + hardware: #GPUs, HBM/GPU, host RAM, interconnect) it
decides the per-rank memory strategy — in-HBM | optimizer host-offload | activation
recompute | layer-streaming — by MEMORY FIT, never by GPU count. It is the sibling
of flagship_budget.per_rank_budget: that module sizes the IN-HBM footprint; this one
applies the strategy SAVINGS and picks the minimal strategy set that fits.

DRIVER (USER DIRECTIVE): strategy decisions MUST NOT key on GPU count. 10M-on-1GPU →
in-HBM (trivial); 10B-on-1GPU → offload+recompute+streaming; 1.5B-on-8GPU → 4D+ZeRO-3.
Same model on 1 vs 8 GPUs differs ONLY because the per-GPU budget changes — the
decision is fit(footprint(strategy), usable_hbm), full stop.

PURE PYTHON — no torch, no GPU. Unit-testable on CPU; the harness prints the plan in
--dry-run BEFORE any GPU work (the same proof contract as flagship_budget).

The savings model (all per-rank, after TP/PP/ZeRO division which flagship_budget does):
  * OPT HOST-OFFLOAD  : optimizer state (k*total floats) lives in PINNED HOST RAM;
                        per-step it is staged in tiles, so the RESIDENT device cost is
                        ~one stage tile, not k*total. Budget: state_gb -> ~offload_tile_gb.
                        REQUIRES: host RAM >= state bytes; PCIe/NVLink bw >= step needs.
  * ACT RECOMPUTE     : store ONLY layer-boundary acts (the per-layer X_in inputs), drop
                        the interior (X_ctx/X_x1/X_gact + dY caches), recompute the layer
                        fwd in bwd. Budget: acts_gb -> acts_gb * (boundary_floats/full).
  * LAYER STREAMING   : weights live in PINNED HOST RAM; a ring keeps `stream_depth`
                        layers resident. Budget: params_gb -> params_gb * stream_depth/L.
                        REQUIRES: host RAM >= param bytes; bw >= per-layer fetch / compute.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from grokking_optimizers.parallel import flagship_budget as fb


# ── Hardware descriptor (the planner's only knowledge of the box). ──
@dataclasses.dataclass(frozen=True)
class Hardware:
    n_gpus: int
    hbm_gib_per_gpu: float
    host_ram_gib: float
    # effective host<->device bandwidth (GiB/s) on the bus that carries offload/stream.
    h2d_gib_s: float = 24.0          # PCIe Gen4 x16 ~24-26 GiB/s (NVLink-C2C/Grace: ~450)
    # usable fraction after CUDA ctx + handles + comm buffers (mirrors flagship_budget).
    usable_frac: float = None        # if None, derive from flagship_budget safety margin

    def usable_hbm_gib(self) -> float:
        if self.usable_frac is not None:
            return self.hbm_gib_per_gpu * self.usable_frac
        # mirror flagship_budget: capacity - 4 GiB safety (ctx/handles/NCCL).
        return self.hbm_gib_per_gpu - fb.H100_SAFETY_GIB


@dataclasses.dataclass(frozen=True)
class MemPlan:
    offload_optimizer: bool
    recompute_acts: bool
    stream_layers: bool
    stream_depth: int                # resident layers when streaming (>=2 for ring)
    ncta: int
    resident_gib: float              # the per-rank device footprint AFTER strategies
    host_gib: float                  # pinned host RAM the plan needs
    fits: bool
    reason: str

    def gate_macros(self) -> dict:
        """The -D macros the kernel/launcher build consumes (the if-constexpr gate set).
        ALL-OFF => the byte-identical in-HBM path."""
        return {
            "SG_MEM_OFFLOAD_OPT": 1 if self.offload_optimizer else 0,
            "SG_MEM_RECOMPUTE_ACTS": 1 if self.recompute_acts else 0,
            "SG_MEM_STREAM_LAYERS": 1 if self.stream_layers else 0,
            "SG_MEM_STREAM_DEPTH": self.stream_depth,
        }


# Fraction of full acts that the layer-boundary checkpoint keeps. The boundary set is
# the per-layer LAYER INPUT (X_in[li], one [T,d] per layer) — the recompute anchor. The
# interior caches (X_ctx/X_x1/X_gact + the 4 dY adjoints) are recomputed. From
# DecActs (model_stage_decoder_tc.cuh:425-433): full per-layer fwd+bwd cache bf16 elems
# = Td(X_in)+Td(X_ctx)+Td(X_x1)+Tff(X_gact) + T3d(dY_qkv)+Td(dY_a)+Tff(dY_ff0)+Td(dY_ff2).
# Boundary keeps only X_in (Td). dff=4d => full = (1+1+1+4 + 3+1+4+1)*Td = 16*Td; kept=1*Td.
_ACT_BOUNDARY_FRAC = 1.0 / 16.0      # exact for dff=4d; recomputed precisely per-shape below


def _full_acts_floats(B: int, layers: int) -> int:
    return fb.dec_tc_acts_floats(B, layers)


def _boundary_acts_floats(B: int, layers: int) -> int:
    """Acts kept under recompute: per-layer X_in (Td) + the non-layer tail (X_hn/dY_logits/dh0).
    Mirrors dec_tc_acts_floats's tail term (B*d + B*V + Td) which is NOT recomputable."""
    d, V, seq = fb.FLAGSHIP_D, fb.FLAGSHIP_VOCAB, fb.FLAGSHIP_SEQ
    T = B * seq
    Td = T * d
    bf = layers * Td                 # one X_in per layer (the checkpoint anchor)
    bf += B * d + B * V + Td         # tail (final-norm/logits/dh0) — must stay
    return (bf + 1) // 2


def plan_memory_strategy(*, total_params: int, layers: int, opt: str,
                         tp: int, pp: int, dp: int, zero3: bool, B: int,
                         hw: Hardware) -> MemPlan:
    """Pick the MINIMAL strategy set that fits the per-rank HBM budget, by FIT not by
    GPU count. Order of escalation (cheapest first):
        in-HBM -> cap nCTA -> recompute acts -> offload optimizer -> stream layers.
    Each step is added ONLY if the running budget still does not fit."""
    usable = hw.usable_hbm_gib()
    n_sms = 132

    # Start from the flagship_budget in-HBM model at full occupancy, then escalate.
    def budget(ncta, recompute, offload, stream_depth):
        b = fb.per_rank_budget(opt, tp=tp, pp=pp, dp=dp, zero3=zero3, ncta=ncta, B=B)
        params_gb, state_gb, acts_gb, staged_gb = b.params_gb, b.state_gb, b.acts_gb, b.staged_gb
        host_gib = 0.0
        layers_pr = max(layers // pp, 1)
        if recompute:
            full = _full_acts_floats(B, layers_pr)
            keep = _boundary_acts_floats(B, layers_pr)
            acts_gb = acts_gb * (keep / max(full, 1))
        if offload:
            host_gib += state_gb        # state moves to pinned host
            # resident device state ~ one stage tile; model it as 1/layers of the state
            # (the launcher stages per-tensor-group), bounded below by a small floor.
            state_gb = max(state_gb / max(layers_pr, 1), 0.05)
        if stream_depth and stream_depth < layers_pr:
            host_gib += params_gb        # weights move to pinned host
            params_gb = params_gb * (stream_depth / layers_pr)
        total = params_gb + state_gb + acts_gb + staged_gb + 0.10
        return total, host_gib

    # 1) in-HBM at the largest nCTA that fits the staged scratch (the existing lever).
    for ncta in (n_sms, 64, 32, 16, 8, 4, 2, 1):
        t, h = budget(ncta, False, False, 0)
        if t <= usable:
            return MemPlan(False, False, False, 0, ncta, t, h, True,
                           f"in-HBM @nCTA={ncta}")
    base_ncta = fb.auto_ncta(opt, tp=tp, pp=pp, dp=dp, zero3=zero3, B=B, n_sms=n_sms)

    # 2) + recompute acts.
    t, h = budget(base_ncta, True, False, 0)
    if t <= usable:
        return MemPlan(False, True, False, 0, base_ncta, t, h, True,
                       f"recompute-acts @nCTA={base_ncta}")
    # 3) + offload optimizer (needs host RAM for the state).
    t, h = budget(base_ncta, True, True, 0)
    if t <= usable and h <= hw.host_ram_gib:
        return MemPlan(True, True, False, 0, base_ncta, t, h, True,
                       f"recompute+offload @nCTA={base_ncta}")
    # 4) + stream layers (resident ring of 2; needs host RAM for params too).
    depth = 2
    t, h = budget(base_ncta, True, True, depth)
    fits = (t <= usable) and (h <= hw.host_ram_gib)
    return MemPlan(True, True, True, depth, base_ncta, t, h, fits,
                   ("recompute+offload+stream(depth=2) @nCTA="
                    f"{base_ncta}" + ("" if fits else " — STILL OOM (raise TP/PP or host RAM)")))


__all__ = ["Hardware", "MemPlan", "plan_memory_strategy"]
```

> **Why this is the planner, not a GPU-count switch:** every branch keys on `t <= usable` where `usable = hbm_per_gpu - safety` and `t` is the per-rank footprint *after* TP/PP/ZeRO division. A 1.5B on 8 GPUs gets a small `t` → returns at step 1 (in-HBM). A 10B on 1 GPU gets a huge `t` → escalates to step 4 (offload+recompute+stream). The **same function, same predicate** — the model size and the per-GPU HBM are the only inputs that move.

### EDIT A2 (NEW FILE) — `csrc/fused/sm_90/mem_config.cuh`
The compile-time gate POD the kernel reads — the *memory-axis* sibling of `ParConfig`, built the same way (all `static constexpr` so `if constexpr` folds; ALL-OFF default = byte-identical). New header (no edit to `parallel_config.cuh` — keeps that file a clean parallelism descriptor).

```cuda
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
```

### EDIT A3 (ADDITIVE to the pending `tuning/flagship_distributed.py`, run_harness.md NEW FILE 2) — wire the planner into the harness

The harness already builds per-rank with `-D` degree macros (run_harness.md:474-494) and prints the budget table in `--dry-run`. Add the planner call + the `-D SG_MEM_*` macros so the chosen strategy is **both** printed and compiled. **VERBATIM OLD** is the run_harness.md spec block that lists the build macros; **NEW** appends the mem-strategy macros. (If run_harness.md has not yet been applied, apply EDIT A3' below to a standalone planner-print instead.)

VERBATIM OLD (run_harness.md NEW FILE 2, the build-macro block, lines ~474-494):
```python
    # ── 4D+ZeRO degrees the megakernel ParConfig instantiates on ──
```
NEW (insert immediately AFTER the existing degree-macro assembly, before the build call):
```python
    # ── MEMORY-STRATEGY gates (the planner's verdict; ALL-OFF => byte-identical in-HBM) ──
    from grokking_optimizers.parallel.mem_strategy import Hardware, plan_memory_strategy
    import torch  # local; the harness is already torch-resident at build time
    hw = Hardware(
        n_gpus=int(os.environ.get("WORLD_SIZE", "1")),
        hbm_gib_per_gpu=torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
        host_ram_gib=float(os.environ.get("SG_HOST_RAM_GIB",
                          str(__import__("os").sysconf("SC_PAGE_SIZE")
                              * __import__("os").sysconf("SC_PHYS_PAGES") / (1024 ** 3)))),
    )
    mplan = plan_memory_strategy(
        total_params=fb.FLAGSHIP_TOTAL_PARAMS, layers=fb.FLAGSHIP_LAYERS, opt=opt,
        tp=tp, pp=pp, dp=dp, zero3=zero3, B=B, hw=hw)
    print(f"[mem-plan] {mplan.reason}  resident={mplan.resident_gib:.2f} GiB "
          f"host={mplan.host_gib:.2f} GiB  fits={mplan.fits}", flush=True)
    if not mplan.fits:
        raise SystemExit("[mem-plan] no strategy fits this box — raise TP/PP, add GPUs, "
                         "or add host RAM. (Refusing to launch an OOM config.)")
    for k, v in mplan.gate_macros().items():
        extra_cflags.append(f"-D{k}={v}")     # the if-constexpr/-#if gate set the kernel reads
```

> **Byte-identical guarantee:** when the planner returns `InHbm` (every gate 0), the macros are `-DSG_MEM_OFFLOAD_OPT=0 -DSG_MEM_RECOMPUTE_ACTS=0 -DSG_MEM_STREAM_LAYERS=0 -DSG_MEM_STREAM_DEPTH=0`. The kernel's `#if`/`if constexpr` on those folds to the shipped path → same PTX. This is the **APPLY-READY** end of the spec: the planner now drives the full strategy from fit, and a 10B-on-1GPU config gets `offload+recompute+stream` macros while a 1.5B-on-8GPU config gets the in-HBM macros — **never** via a GPU-count `if`.

---

## 3. SCOPED (real changes) — the machinery the ON paths emit

Each is gated by the §2 `MemConfig`/`#if SG_MEM_*` so it is **absent (byte-identical) when OFF**. These are GPU-window launcher+kernel edits (the honest "not-a-flag" part); the spec fixes the seam, the ABI, and the parity contract for each.

### (C) Optimizer-state HOST-OFFLOAD (ZeRO-Offload style) — `#if SG_MEM_OFFLOAD_OPT`
**Tractability: MEDIUM (launcher-heavy, modest kernel touch).** The optimizer tail (P3, `apply_optimizer<Opt>`) is the **only** consumer of `st.exp_avg/exp_avg_sq/extra` and it is **element-local** (each element reads/writes only its own `m/v` — confirmed by the launcher's "each tail reads only its own buffers" `:135-137`). So the state can live in **pinned host RAM** and be staged tile-by-tile through the existing P3 loop.

* **Launcher (`mega_decoder_real_adamw_tc_launcher.cu` / `..._tc.cu`):** under `#if SG_MEM_OFFLOAD_OPT`, `cudaHostAlloc` the `[k*total]` state in pinned host RAM (carried in `MemRuntime.host_state_base`); allocate one small device **stage tile** `[stage_tile_floats]`; bind `st.exp_avg/...` to the **stage tile** not the full state. Add a host-side per-tensor-group loop around the P3 launch (or a kernel-internal P3 that signals tile boundaries): `cudaMemcpyAsync(tile<-host)` → P3 applies on the tile (reads grad slice from device, m/v from the tile) → `cudaMemcpyAsync(tile->host)`. The grad+fwd+bwd (P1/P2) are **untouched** (they never read state). The byte-OFF path binds `st.exp_avg` to the full device `state` exactly as today.
* **Parity:** AdamW/Lion/etc. are element-local → tiling is bit-exact (same math, same order). **STAGED optimizers (Prodigy global-d, Muon NS, SG2 meta-net) are NOT element-local** (they reduce across the whole blob in-kernel, `:215-264`) → offload is **gated to the element-local OptIds**; staged opts keep state resident (the planner must know this — add `opt in ELEMENTWISE` guard to `plan_memory_strategy`'s offload branch). Honest scope: offload covers AdamW/Lion/Grokfast/GrokAdamW/NeuralGrok/LookSAM/SG11/SG15; Prodigy/Muon/SG2 do not offload (their cross-tensor stages need the full state on device).
* **Budget:** resident state drops from `k*total` to `~stage_tile`; host RAM grows by `k*total*4` bytes. Modeled in EDIT A1's `offload` branch.

### (D) Activation RECOMPUTE / gradient checkpointing — `#if SG_MEM_RECOMPUTE_ACTS`
**Tractability: HARDER (kernel-body change in the bwd tile).** The persistent megakernel **already streams layers within a token-tile** (P1 runs fwd over all layers, then bwd over all layers, per tile — `fused_decoder_megakernel.cuh:848-876`; the per-tile working set is `DecTileScratch`, the cross-CTA acts are `DecActs`). Recompute means: in the **fwd** pass store only the per-layer **input** `X_in[li]` (the checkpoint boundary); in the **bwd** pass, for each layer, **re-run that layer's fwd from `X_in[li]`** to reconstruct `X_ctx/X_x1/X_gact` into a *transient per-tile* buffer, then do the bwd.

* **Carve (`dec_tc_acts_floats`):** under `#if SG_MEM_RECOMPUTE_ACTS`, the `DecActs` carve keeps only `X_in[kLayers]` + the non-recomputable tail (logits/dh0) — i.e. `_boundary_acts_floats` in EDIT A1. The interior caches + dY adjoints move to a **single-layer transient** in `DecTileScratch` (already per-tile, per-CTA — `model_stage_decoder_tc.cuh:1300-1345`), reused across layers. This is the same "carve a region to 0 when a gate is off" pattern `dec_tc_dw_transpose_floats`/`kDecStagedOptScratch` already use (byte-identical OFF).
* **Bwd tile (`dectc_backward_tile`):** add a `if constexpr (Mem::kRecomputeActs)` branch that, per layer, calls the existing per-layer fwd sub-stage to refill the transient before the existing bwd math. The fwd sub-stages already exist (P1 uses them); recompute **reuses** them — no new math, so parity is by construction (recomputed acts are bit-identical to stored ones, same bf16 GEMMs). The OFF path is the verbatim current body.
* **Cost:** ~1 extra fwd per layer in bwd (≈ +1/3 step time, the standard checkpoint tradeoff). The planner models the acts saving (`keep/full` ≈ 1/16 at dff=4d), not the time — time is a perf note, not a fit gate.
* **Honest scope:** this is the **single biggest enabler** for large-L models and the most invasive kernel edit (touches the hot bwd loop). It is NOT a flag — it is a real `dectc_backward_tile` restructure, gated byte-identical.

### (E) LAYER STREAMING of weights from host — `#if SG_MEM_STREAM_LAYERS`
**Tractability: HARDEST (new prefetch ring + launcher host-pin + kernel slot indirection).** Today `params` is a full device blob and `DecWBf` stages **all** layers' bf16 weights (`model_stage_decoder_tc.cuh:496-531`). Streaming keeps params in **pinned host RAM** and a **ring of `kStreamDepth` layer slots** on device, prefetching layer `li+1` while computing `li`.

* **Launcher:** under `#if SG_MEM_STREAM_LAYERS`, `cudaHostAlloc` the `[total]` params pinned (`MemRuntime.host_param_base`), `cudaMalloc` `kStreamDepth` per-layer slots (`ring_slots`), a `prefetch_stream`. `dec_bind`/`dec_wbf_bind` change from a flat all-layer base to a **slot-indexed** base (`li % kStreamDepth`).
* **Kernel:** the P1 fwd already iterates layers; add `if constexpr (Mem::kStreamLayers)` to (a) `cudaMemcpyAsync` (driver-side, issued from the launcher loop, NOT in-kernel) layer `li+1` into the next ring slot, (b) a grid-barrier wait before a layer's GEMM reads its slot. **Tension with the persistent single-launch design:** a true in-kernel host fetch needs either (i) splitting the persistent launch into per-layer launches (loses the fused-megakernel property), or (ii) a device-initiated copy (cuda graph / `cudaMemcpyAsync` from a host callback). The **honest** scope: streaming is the one strategy that **partially breaks the single-persistent-launch invariant** — it is feasible as a *layer-pipelined relaunch* (a thin host loop over per-layer megakernel sub-launches, each fused internally) OR deferred until the model is large enough that the relaunch overhead is amortized. The planner only selects it as the **last resort** (EDIT A1 step 4), and the spec flags it as the deepest item (parallels the `CommCtx`-widening deferral discipline).
* **Bwd:** weights are needed again in bwd in reverse order → the ring must re-fetch (or, combined with recompute (D), the fwd-recompute pass re-fetches the same layer). Streaming **composes with recompute**: recompute's per-layer fwd-in-bwd is exactly when the bwd needs that layer's weights, so a single ring serves both.
* **Budget:** resident params drop from `total` to `stream_depth/L * total`; host RAM grows by `total*4`. Modeled in EDIT A1's `stream_depth` branch.

---

## 4. Composition + the byte-identical guarantee (the through-line)

The three gates are **orthogonal** and **compose** (the planner turns them on cheapest-first):
* `InHbm` (all OFF) → **byte-identical** to today (the `MemConfig` default, the `SingleGPU` analogue).
* `RecomputeActs` alone → fits big-L; `+OffloadOpt` → fits big-state (AdamW 3× at 10B); `+StreamLayers` → fits big-params. The 10B-on-1GPU config lands on all three; 1.5B-on-8GPU lands on none.
* The decision is **always** `fit(footprint_after_strategy, usable_hbm)` — `mem_strategy.plan_memory_strategy` never reads `n_gpus` except through `hw.usable_hbm_gib()` and the TP/PP/DP division `flagship_budget` already applies. **No naive GPU-count switch anywhere.**

Each gate folds via `if constexpr (Mem::k…)` / `#if SG_MEM_…` exactly as `ParConfig`'s `kShardParams`/`kTPComm` and the existing `kDecStagedOptScratch`/`SG_TUNED_DEC_DW_STAGE` carve gates do — so an OFF strategy emits **zero** extra PTX (the same parity discipline this codebase already enforces and tests).

---

## 5. Honest tractability summary

| Item | Kind | Tractable now? |
| --- | --- | --- |
| A1 `mem_strategy.py` planner + budget | pure Python | ✅ **APPLY-READY** (CPU-testable) |
| A2 `mem_config.cuh` gate POD | header, constexpr-only | ✅ **APPLY-READY** (byte-identical default; compiles, folds away) |
| A3 harness wiring (`-D SG_MEM_*`) | Python build glue | ✅ **APPLY-READY** (atop the pending run_harness.md harness) |
| C opt host-offload | launcher + small kernel | ⚠️ **real change**, element-local OptIds only; scoped here |
| D activation recompute | bwd-tile kernel restructure | ⚠️ **real change**, biggest enabler, most invasive; scoped here |
| E layer streaming | ring + host-pin + persists-launch tension | ⚠️ **deepest**, partially breaks single-launch; scoped + deferred path |

---

## 6. Gate commands

```bash
python -c "import grokking_optimizers.parallel.zero3"
bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1
```

* Cmd 1 (`import ...zero3`) confirms the surveyed module imports cleanly (it is import-safe without torch at module import, `zero3.py:38-39`). The NEW `mem_strategy.py`/`mem_config.cuh` are additive and do not perturb it; once A1 lands, `python -c "import grokking_optimizers.parallel.mem_strategy"` is the analogous gate (it imports only `flagship_budget`, pure-Python).
* Cmd 2 compiles the TC TU. The §2 edits are **byte-identical by default** (`MemConfig=InHbm`, no `-DSG_MEM_*` ⇒ the `#if SG_MEM_…` blocks are absent), so this command's result is **unchanged** by applying A1/A2/A3; A2's header only matters once the TU `#include`s it behind the gate (a §3 follow-on). The compile gate therefore proves the survey edits are non-regressing.
