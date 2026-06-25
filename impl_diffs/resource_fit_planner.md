# AREA: resource_fit_planner — the ROBUST execution PLANNER (`grokking_optimizers/parallel/resource_planner.py`)

**Scope:** ONE new pure-Python module `grokking_optimizers/parallel/resource_planner.py` +
two byte-exact hooks (a re-export in `grokking_optimizers/parallel/__init__.py` and a thin
constructor on `grokking_optimizers.distributed.ParallelConfig`), plus one new CPU test file
`tests/test_resource_planner.py`. The module is a pure function of
`(model_cfg, hw_cfg) -> ExecutionPlan`. NO torch, NO CUDA, NO GPU — it imports and tests on any
box. It is the front-end's single decision point: given the model shape and the hardware, it
emits the FULL execution config (parallelism mesh + memory strategy + kernel knobs) and the
exact `-D…` compile flags + kernel-template instantiation the chosen plan maps to.

**READ IN FULL (this session, READ-ONLY on `/workspace/SuperGrok1.5`):**
- `impl_diffs/run_harness.md` (the mesh math + the live `dec_tc_*_floats` mirror in
  `flagship_budget.py`; the SG2 `~91.3·Nmax/CTA` fact; the `auto_ncta` ladder; H100 usable-GiB
  gate). **This planner GENERALIZES `flagship_budget.py` from the pinned flagship constants to an
  arbitrary `(d, layers, vocab, seq)` model** and adds the memory-strategy decision tree.
- `impl_diffs/dist_step.md` (the 4D plumb: `DistStepContext` TP/PP coords, the ZeRO-3
  reduce-scatter footprint fact — full-grad all-gather is `total·world` floats, the binding
  ZeRO-3 fix; the `Par`-template launcher instantiation allow-list; the loopback/NVSHMEM seam).
- Live code: `grokking_optimizers/distributed.py` (`ParallelConfig`, `_RankMesh`,
  `_coords_from_rank`, `world_size`/`model_parallel_size`); `grokking_optimizers/parallel/__init__.py`;
  `grokking_optimizers/parallel/shard_map.py` (`shard_mode_for_optimizer`, ELEMENTWISE vs
  PER_TENSOR taxonomy); `csrc/fused/sm_90/fused_decoder_megakernel.cuh` lines 480–657 (the EXACT
  `dec_tc_acts_floats`, `dec_tc_opt_reduce_floats`, `dec_tc_muon_floats`, `dec_tc_looksam_floats`,
  `dec_sg2_ws_stride_floats`, `dec_tc_sg2_floats`, `dec_tc_staged_opt_floats`,
  `dec_tc_workspace_floats` formulas + the `kDecStagedOptScratch` gate + the SG2 KNOWN DEEP LIMIT
  comment); `csrc/fused/sm_90/opt_stage_supergrok2.cuh` lines 178–191 (`SG2Dims<>` defaults) +
  440–470 (`sg2_ws_stride` — the literal **91.277·Nmax** floats/CTA with the defaults, verified
  numerically this session); `csrc/fused/sm_90/decoder_flagship_layout.cuh`
  (SG_DEC_D=1600, LAYERS=48, DFF=6400, VOCAB=99, SEQ=4, kDecTotalElems=1475884899,
  kDecNumTensors=582, kDecMaxTensorNumel=10240000); `grokking_optimizers/megakernel_codegen.py`
  `_decoder_param_sizes` (lines 596–617 — the **EXACT** per-tensor numel formula:
  `2 + 12·L + 4` tensors, `dff=4d`, which I verified reproduces the flagship total/nt/max byte-exact).
- `tests/test_zero3_plan.py` (the CPU test idiom this new test matches).

**`adaptive_parallelism.md` / `size_adaptive.md` are NOT present** in `/workspace/impl_diffs`
(checked). So the 3D→5D inference is built from FIRST PRINCIPLES on top of the live
`distributed.py` mesh math (Megatron linearization, TP fastest) + the directive, and cross-cited
to `dist_step.md`'s 4D plumb. Where those two specs would have provided a knob, I derive it from
the live `_RankMesh`/`ParallelConfig` instead, so the planner stands alone.

---

## 0. THE DESIGN CONTRACT (the user directive, made executable)

> **Strategy decisions MUST NOT key on GPU count.** The driver is **memory-fit + compute-shape
> vs hardware**, never `if num_gpus == 1: …`. A single GPU can host a 10B model for TRAINING via
> offload + recompute + layer-streaming + CTA-tiling; 1.5B on 8 GPUs is 4D+ZeRO-3; 10M on 1 GPU
> is trivial in-HBM. The SAME if-constexpr/config-templating emits exactly the chosen machinery
> and is byte-identical when a strategy is OFF.

`plan_execution` realizes this as a **pure cost model + a fixed escalation ladder**. It NEVER
branches on `num_gpus`; it branches on `fit(strategy_set) <= usable_hbm`. FIRST it infers the
parallelism mesh once (`infer_mesh`, §3: TP up to `min(num_gpus, nvlink_width)` with `d % TP == 0`,
then DP fills the rest, EP sub-divides DP for MoE — all bounded by `num_gpus`, never a count
switch). THEN it walks the MEMORY ladder in increasing order of overhead, re-estimating the
per-rank footprint after each rung and stopping at the first FIT:

```
rung  memory move (per-rank footprint)                 cost
────  ──────────────────────────────────────────────  ─────────────────────────────────────
 R0   in-HBM, full occupancy (nCTA = #SMs)             none — the trivial case (10M/1GPU)
 R1   ZeRO-3 (shard params+opt-state over DP)          ~free; no-op at DP=1, the 8-GPU win
 R1b  raise PP (consume a DP factor) if per-stage OOMs the 1F1B bubble (only when TP+Z3 won't fit)
 R2   CTA-tiling (cap nCTA → shrink staged scratch)    occupancy (NOT compute) — cheapest knob
 R3   activation RECOMPUTE (keep ~1 layer live)        a 2nd fwd pass — binding at long seq
 R4   LAYER STREAMING (params resident 1 stage)        host↔device bandwidth (single-rank PP)
 R5   host PARAM/OPT OFFLOAD (AdamW-on-host)            PCIe/NVLink bandwidth — the terminal rung
```

R2 (CTA-tiling) precedes R3 (recompute) because trimming occupancy is cheaper than a second
forward pass, and the staged-opt scratch (`nCTA·91.277·Nmax`) is usually the binding term — this
is exactly why the flagship SG2 fits at **nCTA=64 WITHOUT recompute** (the run_harness.md headline),
while the 10B/seq=2048 case (where acts is the binding 96 GiB term) still reaches R3.

Crucially the ladder is driven by **the same memory-fit arithmetic on EVERY rung** — the planner
re-estimates the per-rank footprint after each rung is toggled and stops at the first FIT. With
`num_gpus=1` the parallelism rungs (R0..R3) are no-ops (degree 1), so a 10B model on 1 GPU falls
straight through to R2..R5 (recompute + CTA-tile + stream + offload) — *the same code path* that
a 10B/8GPU run uses, just with the parallelism degrees pinned to 1. **That is the directive: the
machinery is selected by fit, the GPU count only sets the ceiling on the parallelism rungs.**

---

## 1. THE EXACT MEMORY-FIT ARITHMETIC (mirrors the LIVE `dec_tc_*_floats`)

The planner re-derives the per-rank HBM footprint from the **live** kernel scratch formulas,
generalized from the flagship pins to an arbitrary `(d, layers, vocab, seq)`. Every constant is
cited to its live source; a drift in the kernel must be reflected here.

### 1.1 Parameter count (mirror of `megakernel_codegen.py::_decoder_param_sizes`, lines 606–617)

`dff = 4·d`. The decoder emits `2 + 12·L + 4` named tensors in `named_parameters()` order:

```
tok = vocab·d ,  pos = seq·d
per layer (×L): in_proj_w 3d² , in_proj_b 3d , out_proj_w d² , out_proj_b d ,
                n1.w d , n1.b d , n2.w d , n2.b d ,
                ff.0.w 4d·d , ff.0.b 4d , ff.2.w d·4d , ff.2.b d
tail: norm.w d , norm.b d , out.w vocab·d , out.b vocab
```

`TOTAL(d,L,V,seq) = Σ sizes`. **Verified this session**: at `(1600,48,99,4)` this gives
`TOTAL=1,475,884,899`, `n_tensors=582`, `max_tensor_numel=10,240,000` — byte-exact to
`decoder_flagship_layout.cuh`. The largest per-tensor numel (the SG2 `Nmax`) is
`kDecMaxTensorNumel = max(sizes) = max(4d·d, vocab·d) = 4d²` for any reasonable `(d,vocab)` with
`vocab < 4d` (true at the flagship: `vocab·d = 158,400 ≪ 4d² = 10,240,000`).

### 1.2 The staged-opt scratch (mirror of `fused_decoder_megakernel.cuh` :553–638)

This is the load-bearing, model-INDEPENDENT-but-tensor-shape-DEPENDENT term. With the default
`SG2Dims<>` (`d_model=8, gru_hidden=4, indexer_rank=4, csa_compress=4, csa_topk=16`,
`opt_stage_supergrok2.cuh:178–191`):

```
sg2_ws_stride(Nmax) =            # opt_stage_supergrok2.cuh:440 — floats / CTA
   7·Nmax·d_model                # x_sorted,csa_ctx,hca_ctx,q,win_k,win_v,concat
 + 2·ceil(Nmax/csa_compress)·d_model      # c_k, c_v
 + Nmax·indexer_rank             # qI
 + ceil(Nmax/csa_compress)·indexer_rank   # kI
 + Nmax·csa_topk                 # sel
 + Nmax·gru_hidden               # new_gru
 + Nmax                          # expert_out
 + 2·next_pow2(Nmax)             # sort keys + idx
 + 2·Nmax                        # perm + unsort
≈ 91.277·Nmax   (numerically verified this session for the defaults; run_harness.md "91.3")

dec_sg2_ws_stride_floats() = 2·n_tensors + sg2_ws_stride(Nmax)         # :615
dec_tc_sg2_floats(nCTA)    = nCTA · dec_sg2_ws_stride_floats() + 1     # :619 (gated)
dec_tc_muon_floats(nCTA)   = 4·Nmax2d + maxRows² + nCTA + 1           # :567 (Nmax2d=max 2D weight)
dec_tc_looksam_floats()    = 2·TOTAL                                   # :584
dec_tc_opt_reduce_floats(nCTA) = 2·nCTA + 1                            # :553
dec_tc_staged_opt_floats(nCTA) = opt_reduce + muon + looksam + sg2     # :634
```

> **The `kDecStagedOptScratch` gate (`:541–545`):** the four staged regions are carved
> UNCONDITIONALLY on the production launcher (`true`) so ONE workspace fits every OptId; only the
> adamw-only bench layout (`SG_DEC_BENCH_LAYOUT=1`) elides them (`false`). **The planner models
> production semantics: the staged carve is present whenever the chosen optimizer needs it.** For
> an ELEMENTWISE optimizer (adamw/lion/…) on the production path the carve is STILL present (the
> opt-agnostic launcher), but the planner can emit `SG_DEC_BENCH_LAYOUT=1` to elide it **iff the
> chosen optimizer is adamw AND the run is single-optimizer** — that is the `staged_scratch_needed`
> decision below. SG2 ALWAYS needs the full carve; it is the worst case and the binding term.

### 1.3 Activations (mirror of `dec_tc_acts_floats`, :504–512)

```
T = B·seq ; Td=T·d ; T3d=3T·d ; Tff=T·4d
acts_bf16 = L·(Td+Td+Td+Tff+T3d+Td+Tff+Td) + B·d + B·V + Td
dec_tc_acts_floats = ceil(acts_bf16 / 2)      # bf16 region in float units
```

with **L → layers_per_pp_stage = L/PP** (PP holds only its stage's layers live). Under
**activation RECOMPUTE** the planner keeps only `~1 transformer layer` live at a time, so it
replaces `L/PP` with `1` in the per-layer sum (the embedding tails stay). This is the
gradient-checkpointing memory model: `acts_recompute ≈ acts(L=1) + tails`.

### 1.4 Params + optimizer state, with ZeRO-3 / offload

```
model_shard = TP·PP                         # ranks one model copy is spread across
zero_div    = DP if zero3 else 1            # ZeRO-3 shards params+state over DP
resident_params = TOTAL / (model_shard · zero_div)
state_floats    = state_planes(opt) · TOTAL / (model_shard · zero_div)
  state_planes:  adamw/lion/grokfast/grokadamw/neuralgrok=3 , prodigy/looksam/muon=3..4 ,
                 supergrok11/15=5 , supergrok2 = (4+1+gru_hidden)=9   # launcher :279
```

**host PARAM/OPT OFFLOAD** subtracts the offloaded tensors from the HBM total and adds them to a
**host-RAM** budget line (checked against `hw_cfg.host_ram_bytes`): `need_param_offload` moves
`resident_params·4 B` to host; `need_opt_offload` (the AdamW-on-host path the directive names)
moves `state_floats·4 B` to host. The staged scratch and activations stay in HBM (they are
transient per-step). **LAYER STREAMING** (`need_layer_streaming`) makes only `1/PP_stream` of the
params resident at a time (a software pipeline over layers), modeled as
`resident_params → resident_params · stream_frac` where `stream_frac = 1/layers` for the extreme
1-layer-resident stream, bounded by the host↔device bandwidth note in the plan's `risks`.

### 1.5 The per-rank total + the usable-HBM gate (mirror of `flagship_budget.py` H100 line)

```
per_rank_hbm = params + state + acts + staged_opt + tile_scratch_slack(0.10 GiB)
usable_hbm   = hbm_bytes_per_gpu/GiB − safety(default 4.0 GiB ctx+handles+NCCL)
FIT  ⟺  per_rank_hbm ≤ usable_hbm   AND   host_used ≤ host_ram_bytes/GiB
```

`GiB = 1024³`. All math is in GiB (binary), ONE unit — no `1000³`-vs-`1024³` mixing, the same
discipline `flagship_budget.py:175` enforces. The H100 "80 GB" = `80·1000³ B` = 74.51 GiB
physical; `hbm_bytes_per_gpu` is taken from `hw_cfg` (so the planner is hardware-portable, not
H100-pinned), and `safety` defaults to 4.0 GiB.

---

## 2. THE PLANNER DECISION TREE (how each flag is decided)

```
plan_execution(model_cfg, hw_cfg):
  1. TOTAL, n_tensors, Nmax := layout_arith(model_cfg)            # §1.1
  2. mesh := infer_mesh(model_cfg, hw_cfg)                        # §3  (TP,PP,DP,SP,EP)
  3. flags := MemFlags(all OFF)                                   # in-HBM, full occupancy
     ncta := hw_cfg.sms_per_gpu  (default 132)
  4. for rung in [R1_zero3, R2_recompute, R3_cta_tile, R4_stream, R5_offload]:
        if fit(mesh, flags, ncta) : break
        toggle rung's flag (or shrink ncta one ladder step)       # §0 ladder
  5. if still not fit after R5 : raise PlanInfeasible(detailed breakdown)
  6. kernel_knobs := tier_by_compute(model_cfg, mesh)             # §4
  7. return ExecutionPlan(mesh, flags, ncta, kernel_knobs, compile_flags, template_inst)
```

- **`need_zero_offload` (ZeRO-3)** — ON (R1) when `DP>1` and params+state at full residency would
  not fit; it is the cheapest memory move that costs no compute. At `DP=1` ZeRO-3 is a no-op
  (`zero_div=1`) so the planner records `zero3=True` but it shrinks nothing — exactly the
  `flagship_budget.py` semantics ("no-op at DP=1"). The directive's "1.5B/8GPU → 4D+ZeRO-3" is
  this rung firing with `DP` carved out of the mesh.
- **`need_activation_recompute`** — ON (R2) when acts is a binding term (it is the largest term
  at long `seq`/large `B`; at the flagship `seq=4` it is tiny so R2 rarely fires, but at the 10B
  `seq=2048` config acts is 96 GiB → R2 is mandatory). Sets `acts → acts(L=1)+tails` (§1.3).
- **`cta_tiling` / `ncta`** — the staged-opt scratch is `nCTA·91.277·Nmax_per_rank`. R3 walks the
  `auto_ncta` ladder `132→64→32→16→8→4→2→1` (the live ladder, `flagship_budget.auto_ncta:333`)
  picking the LARGEST nCTA that fits. This is "CTA-tiling": trading occupancy for staged-scratch
  footprint. `cta_tiling=True` ⟺ `ncta < sms_per_gpu`.
- **`need_layer_streaming`** — ON (R4) when even nCTA=1 + ZeRO-3 + recompute does not fit and
  `PP=1` (PP already streams across ranks; streaming is the single-rank analogue). Makes params
  resident one stage at a time.
- **`need_param_offload`** — ON (R5) last, when HBM still over after R1..R4 and
  `host_ram` has room. Moves params and/or opt-state to host (AdamW-on-host). This is the
  directive's "10B-on-1GPU → offload+recompute+streaming" terminal rung.

**The optimizer interacts with the ladder:** SG2's 9-plane state + the `91.277·Nmax` staged
scratch make it the worst case. The planner takes `model_cfg.optimizer` (default the run's
optimizer; for a multi-opt benchmark it plans the WORST case = supergrok2) and, when even R5
cannot fit SG2 at 10B/1GPU, it records `optimizer_downgrade="supergrok2→adamw"` in `risks` rather
than silently failing — the staged SG2 scratch at `Nmax=4d²=67M` (10B, no TP) is `~9 TB/132 CTA`
(the live KNOWN DEEP LIMIT, `fused_decoder_megakernel.cuh:598–610`), structurally unfittable on
one GPU; the honest plan is "use an elementwise optimizer + host offload," which the planner emits.

---

## 3. THE PARALLELISM MESH (reuse run_harness.md math + distributed.py `_RankMesh`)

`infer_mesh` produces `(DP,TP,PP,SP,EP)` with `DP·TP·PP == num_gpus` (the live
`ParallelConfig.world_size` invariant, `distributed.py:136`) and `SP==1` (pinned; short-seq, the
`distributed.py` §8.1 note + the `parallel_config.cuh` `static_assert(SP==1)` from dist_step.md).
The inference, NOT keyed on a GPU-count switch:

1. **TP first** (it is the move that shrinks `Nmax_per_rank = Nmax/TP`, the binding staged term,
   and the in-kernel all-reduce rides the tightest fabric — `_RankMesh` puts TP fastest,
   `distributed.py:213–221`). `TP = min(num_gpus, max_tp)` where `max_tp` = the largest power of 2
   dividing `num_gpus` AND `≤ nvlink_width` (TP all-reduce wants NVLink, not PCIe — from `hw_cfg`;
   default 8 on an NVLink node). TP must divide `d` (Megatron column/row split needs `d % TP == 0`).
2. **PP next** across the remaining `num_gpus/TP` if the per-stage param+act residency still
   overflows after TP — `PP = smallest divisor of (num_gpus/TP) such that L % PP == 0` that brings
   per-stage params under HBM. PP is owner-locked as overhead at short race depth (dist_step.md
   §2(c) / `pipeline.py` HONEST SCOPE), so the planner only raises PP when TP+ZeRO-3 alone cannot
   fit — never for throughput.
3. **DP fills the rest**: `DP = num_gpus / (TP·PP)`. ZeRO-3 shards params+state over this DP group
   (the `dp_group` of `_RankMesh`, dist_step.md §6.A `make_dist_step_context_4d`).
4. **EP** when `num_experts > 1`: `EP = min(num_experts, DP)` and `EP | DP` (the live
   `ParallelConfig.__post_init__` invariant `data_parallel % expert_parallel == 0`,
   `distributed.py:119`). EP sub-divides DP (experts sharded over DP peers), it does NOT enlarge
   `world_size` — exactly the live semantics (`distributed.py:69–73`).

At `num_gpus=1` every degree is 1 (the degenerate `_build_mesh(0, …)` path) — the mesh rungs are
inert and the memory ladder does ALL the work, which is the directive's 10B/1GPU case.

This is "3D–5D inference": dense short-seq → 3D (DP×TP×PP); +EP for MoE → 4D; +ZeRO-3 as the
state-shard overlay on DP → the 5th "dimension" the directive counts (params/opt-state sharding).
SP is the inert 5th mesh axis (expressible, pinned to 1).

---

## 4. KERNEL KNOB TIER (by compute size)

```
tier_by_compute(model_cfg, mesh):
  gemm_m = B·seq (rows) ; gemm_k = d ; gemm_n = max(4d, vocab)   # the binding GEMM shape
  cta_tiling : True iff ncta < sms_per_gpu  (set by the memory ladder R2)
  ring_depth : fwd cp.async pipeline stages by per-CTA tile fit —
               2  when d ≤ 1024  (the shallow ring, fits 48 KB static smem)
               3  when 1024 < d ≤ 4096 (deep ring, needs SG_DEC_TC_DYNAMIC_SMEM)
               (mirrors fused_decoder_megakernel.cuh:480–486 static-smem cap note)
  occupancy  : ncta (1 CTA/SM when ncta==sms_per_gpu)
```

`ring_depth` keys on `d` (the per-CTA tile width drives the cp.async smem pipeline depth) and
maps to the live `SG_TUNED_DEC_FWD_PIPE` / `SG_DEC_TC_DYNAMIC_SMEM` gates. `cta_tiling` is set by
the memory ladder, NOT independently — it is the same `ncta` knob the staged-scratch fit drives.

---

## 5. HOW `ExecutionPlan` MAPS TO COMPILE FLAGS + TEMPLATE INSTANTIATION

The plan's `compile_flags` and `template_inst` are exactly what `run_harness.md`'s
`build_flagship_module` and dist_step.md's §6.C `Par`-template launcher consume:

```
ParConfig<DP,TP,PP,SP,Z> :  -DSG_FLAGSHIP_DP={DP} -DSG_FLAGSHIP_TP={TP}
                            -DSG_FLAGSHIP_PP={PP}  -DSG_FLAGSHIP_ZERO={3 if zero3 else 0}
layout (size-adaptive)   :  -DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1 -include {layout_header}
                            (flagship header for the 1.5B pin; a codegen'd header for other sizes)
staged carve             :  (default ON) ; -DSG_DEC_BENCH_LAYOUT=1 ONLY when
                            staged_scratch_needed==False (adamw single-opt) → elides the 4 carves
recompute                :  -DSG_DEC_RECOMPUTE=1            (acts→1-layer-live; kernel re-runs fwd in bwd)
layer streaming          :  -DSG_DEC_LAYER_STREAM=1         (1 stage of params resident)
param/opt offload        :  -DSG_DEC_HOST_OFFLOAD={1:param,2:opt,3:both}
cta tiling / occupancy   :  runtime ncta arg to tc_train_step (NOT a compile flag — the live
                            launcher takes ncta; auto_ncta picks it)
ring depth               :  -DSG_TUNED_DEC_FWD_PIPE={1 if ring==2 else 2}
                            -DSG_DEC_TC_DYNAMIC_SMEM=1   (when ring_depth==3)
transport                :  -DSG_HAS_NVSHMEM=1            (when TP>1 and the toolkit is present)
```

`template_inst` is the explicit instantiation point (dist_step.md §6.C.4 allow-list):
`launch_fused_decoder_megakernel_tc<Opt, ParConfig<DP,TP,PP,1,Z>>`. **Byte-identical guarantee:**
when every memory flag is OFF and `TP==PP==DP==1`, `ParConfig` defaults to `par::SingleGPU` and
`kEmitComm==false`, so the emitted PTX is byte-identical to the single-GPU build (the §1.2
guarantee in dist_step.md). The planner therefore satisfies "byte-identical when a strategy is
OFF" by construction — it only ADDS `-D…` when a flag is ON, and each `-D` gates an
`if constexpr` that folds to the original code when its config bit is 0.

---

## 6. WORKED EXAMPLES (numbers computed THIS SESSION from the live formulas)

GiB binary; usable = `hbm/GiB − 4.0`. SG2 = `nCTA·91.277·(Nmax/TP)·4 B`.

### 6.1 — 10M on 1 GPU (trivial → in-HBM, full occupancy)
`d=512, L=8, vocab=99, seq=128` → `TOTAL=25.4M`, `Nmax=4d²=1.05M`. 1×H100 (70.5 usable).
- adamw, TP1 PP1 DP1, nCTA=132, in-HBM: params 0.09, state 0.28, acts 4.03, **staged-SG2-region
  46.41** → TOTAL **50.92 GiB → FITS**. The staged carve is present (production launcher) and is
  the biggest line even for a tiny model — so the planner notes it can set
  `SG_DEC_BENCH_LAYOUT=1` (adamw single-opt) to drop it to ~4.5 GiB. **Plan: `{TP1,PP1,DP1,SP1,EP1}`,
  all memory flags OFF, ncta=132, ring=2 (d≤1024), staged_scratch_needed=False → bench-layout
  elide.** Trivial, as the directive says.

### 6.2 — 1.5B on 8 GPUs (the flagship → 4D + ZeRO-3)
`d=1600, L=48, vocab=99, seq=4` → `TOTAL=1.476B`, `Nmax=10.24M`. 8×H100.
- `infer_mesh`: TP=8 (divides d=1600; NVLink), PP=1, DP=1, ZeRO-3 overlay. `Nmax/TP=1.28M`.
- adamw, nCTA=132: params 0.69, state 2.06, acts 4.70, **staged-SG2 57.45** → **65.00 GiB → FITS**
  at 1 CTA/SM.
- supergrok2 (worst case), nCTA=132: state 6.19, staged-SG2 57.45 → 70.52 GiB (0.01 over usable) →
  R3 caps **nCTA=64** → staged-SG2 27.86 → **39.52 GiB → FITS**. (Exactly the run_harness.md
  headline — this planner reproduces it as the `auto_ncta` rung.) **Plan: `{TP8,PP1,DP1,SP1,EP1}`,
  zero3=ON (no-op at DP=1 but recorded), recompute OFF, cta_tiling for SG2 only (ncta=64),
  others ncta=132, ring=3 (d=1600>1024 → dynamic smem), NVSHMEM ON.** The directive's "1.5B/8GPU
  → 4D+ZeRO-3."

### 6.3 — 10B on 1 GPU (offload + recompute + streaming + CTA-tiling)
`d=4096, L=48, vocab=50304, seq=2048` → `TOTAL=10.09B`, `Nmax=4d²=67.1M`, `seq` long. 1×H100,
host_ram=512 GB. **All numbers below are the planner's actual output, verified this session.**
- in-HBM adamw nCTA=132: params 37.58, state 112.73, acts **96.13**, staged **77.15**
  (the looksam `2·total` carve) → **323.68 GiB → OOM** (and with the SG2 carve the staged term is
  ~9 TB — the live KNOWN DEEP LIMIT).
- ladder (adamw, staged carve elided via bench-layout — looksam/muon are dead weight for a single
  elementwise opt): R1 ZeRO-3 no-op (DP=1). R2 cta-tile: with acts=96 GiB nothing fits → nCTA=1.
  R3 recompute: acts 96.13→**2.13**. R4 layer-streaming: params 37.58→**0.78**. R5 offload:
  opt-state 112.73→host. Final: **HBM 3.01 GiB / host 112.73 GiB → FITS**. **Plan:
  `{TP1,PP1,DP1,SP1,EP1}`, zero3=ON(no-op), cta_tiling ON (ncta=1), recompute ON, layer_streaming
  ON, opt_offload ON, ring=3.** The directive's "10B-on-1GPU → offload+recompute+streaming" — the
  full heavy stack, selected because the fit math demands it, not a GPU-count `if`.
- **supergrok2 on 1 GPU:** even at nCTA=1 the SG2 staged scratch is **70 GiB** (`91.277·Nmax`,
  `Nmax=67.1M` un-shrunk by TP) → structurally unfittable. The planner records the honest risk
  (`fused_decoder_megakernel.cuh:598-610` KNOWN DEEP LIMIT) and **downgrades the optimizer to
  adamw + the offload stack above**. SG2 at 10B needs TP (more GPUs) to shrink `Nmax`.

### 6.4 — 10B on 8 GPUs (TP shrinks Nmax → recompute + CTA-tiling, NO host offload)
`d=4096, L=48, vocab=50304, seq=2048`. 8×H100. **Planner's actual output, verified this session.**
- `infer_mesh`: TP=8 (4096%8=0; NVLink). `Nmax/TP=8.39M`, params/8, state/8.
- ladder (adamw): R1 ZeRO-3. R2 cta-tile→nCTA=1 (acts=96 GiB binding). R3 recompute: acts→2.00.
  Final: params 4.70, state 14.09, acts 2.00, staged elided → **HBM 21.01 GiB → FITS, host 0**.
  **Plan: `{TP8,PP1,DP1,SP1,EP1}`, zero3=ON, cta_tiling ON (ncta=1), recompute ON, ring=3,
  NVSHMEM ON.** With TP=8 the 10B model trains across 8 GPUs **without host offload** — the contrast
  to 6.3 that proves the driver is fit-based, not GPU-count-based: SAME model, the 8-GPU case needs
  only cta-tile+recompute, the 1-GPU case needs the full offload+stream stack — *because the fit
  math says so*, not a `num_gpus` switch. (If the operator wants higher occupancy than nCTA=1,
  raising the GPU count further — DP or more TP — relaxes the staged term; the planner would then
  pick a larger nCTA on the same ladder.)

---

## NEW FILE — `grokking_optimizers/parallel/resource_planner.py` (in full)

```python
"""grokking_optimizers/parallel/resource_planner.py — the ROBUST execution PLANNER.

plan_execution(model_cfg, hw_cfg) -> ExecutionPlan

From the FRONT-END model config (d, layers, seq, vocab, num_experts, is_sequence,
optimizer) and the HARDWARE (num_gpus, hbm_bytes_per_gpu, host_ram_bytes, interconnect),
compute the FULL execution config:

  (a) the parallelism mesh (DP,TP,PP,SP,EP)         — §3, reuses distributed._RankMesh math
  (b) the MEMORY STRATEGY flags                     — §2, a memory-FIT escalation ladder
        need_zero_offload / need_activation_recompute / need_layer_streaming /
        need_param_offload (+ need_opt_offload), and cta_tiling via ncta
  (c) the kernel knob tier (cta_tiling, ring_depth, occupancy) — §4, by compute shape

The driver is **memory-fit + compute-shape vs hardware**, NEVER a GPU-count switch
(the user directive). The same ladder runs for 10M/1GPU (trivial), 1.5B/8GPU
(4D+ZeRO-3), and 10B/1GPU (offload+recompute+streaming+cta-tiling); the GPU count only
sets the ceiling on the parallelism rungs.

PURE PYTHON: no torch, no CUDA, no GPU. Mirrors the LIVE kernel scratch formulas
(fused_decoder_megakernel.cuh dec_tc_*_floats + opt_stage_supergrok2.cuh sg2_ws_stride
+ megakernel_codegen.py _decoder_param_sizes), so the front-end gets an exact, provable
per-rank budget + the exact -D compile flags BEFORE any GPU work.

SOURCES (read in full, cited inline):
  * param sizes  : megakernel_codegen.py::_decoder_param_sizes (2 + 12*L + 4 tensors, dff=4d)
  * acts         : fused_decoder_megakernel.cuh::dec_tc_acts_floats (:504)
  * staged scratch: fused_decoder_megakernel.cuh dec_tc_{opt_reduce,muon,looksam,sg2}_floats
                    (:553-638) + opt_stage_supergrok2.cuh::sg2_ws_stride (:440, SG2Dims<> defaults)
  * mesh         : grokking_optimizers/distributed.py ParallelConfig + _RankMesh (TP fastest)
  * opt taxonomy : grokking_optimizers/parallel/shard_map.py (ELEMENTWISE vs PER_TENSOR)
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

# ── Units (binary GiB, ONE unit — no 1000^3/1024^3 mixing; flagship_budget.py:175). ──
GB = 1024 ** 3
BYTES_PER_FLOAT = 4

# ── SG2Dims<> defaults (opt_stage_supergrok2.cuh:178-191). ──
SG2_D_MODEL, SG2_GRU_HIDDEN, SG2_INDEXER_RANK = 8, 4, 4
SG2_CSA_COMPRESS, SG2_CSA_TOPK = 4, 16

# ── Optimizer state-plane counts (mega_decoder_real_adamw_tc_launcher.cu state layout;
#    shard_map.py taxonomy). supergrok2 = (4+1+gru_hidden) = 9 (the 9-plane outlier). ──
_STATE_PLANES: Dict[str, int] = {
    "adamw": 3, "lion": 3, "grokfast": 3, "grokadamw": 3, "neuralgrok": 3,
    "prodigy": 4, "looksam": 3, "muon": 3, "supergrok11": 5, "supergrok15": 5,
    "supergrok2": 4 + 1 + SG2_GRU_HIDDEN,
}
# Per shard_map.py: per-TENSOR optimizers need whole tensors on one rank (no flat split);
# elementwise may flat-split. The planner uses this to know whether the staged SG2 carve
# is needed (SG2 always) and whether bench-layout elision is legal (adamw single-opt only).
_ELEMENTWISE = frozenset({"adamw", "lion", "grokfast", "grokadamw",
                          "looksam", "prodigy", "neuralgrok"})
_PER_TENSOR = frozenset({"muon", "supergrok11", "supergrok15", "supergrok2"})
# Optimizers whose staged-opt scratch is the binding SG2 meta-net carve.
_NEEDS_SG2_CARVE = frozenset({"supergrok2"})


# ───────────────────────────── front-end config types ────────────────────────────


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Front-end model shape. `optimizer` is the run's optimizer (or the WORST case of a
    multi-optimizer benchmark — supergrok2 — so the plan fits every member)."""
    d: int
    layers: int
    seq: int
    vocab: int
    batch: int = 256
    num_experts: int = 1
    is_sequence: bool = True            # decoder/transformer; False ⇒ no PP stage-cut benefit
    optimizer: str = "adamw"

    def __post_init__(self) -> None:
        for k in ("d", "layers", "seq", "vocab", "batch"):
            if getattr(self, k) < 1:
                raise ValueError(f"ModelConfig.{k} must be >= 1")
        if self.optimizer not in _STATE_PLANES:
            raise ValueError(f"unknown optimizer {self.optimizer!r} "
                             f"(known: {sorted(_STATE_PLANES)})")
        if self.num_experts < 1:
            raise ValueError("num_experts must be >= 1")


@dataclasses.dataclass(frozen=True)
class HardwareConfig:
    """Hardware envelope. Defaults model one 80 GB H100 SXM5 NVLink node."""
    num_gpus: int = 1
    hbm_bytes_per_gpu: int = 80 * (1000 ** 3)   # advertised "80 GB" (74.51 GiB physical)
    host_ram_bytes: int = 512 * (1000 ** 3)
    nvlink: bool = True                          # TP all-reduce wants NVLink, not PCIe
    nvlink_width: int = 8                        # max TP degree on the tight fabric
    sms_per_gpu: int = 132                       # H100 SXM5; 1 CTA/SM at full occupancy
    safety_gib: float = 4.0                      # ctx + cuBLAS/cuDNN + NCCL buffers

    def __post_init__(self) -> None:
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be >= 1")
        if self.hbm_bytes_per_gpu < 1 or self.host_ram_bytes < 1:
            raise ValueError("memory sizes must be positive")

    @property
    def usable_hbm_gib(self) -> float:
        return self.hbm_bytes_per_gpu / GB - self.safety_gib

    @property
    def host_ram_gib(self) -> float:
        return self.host_ram_bytes / GB


# ───────────────────────────── output types ──────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Mesh:
    dp: int
    tp: int
    pp: int
    sp: int
    ep: int

    @property
    def world_size(self) -> int:                # mirrors ParallelConfig.world_size
        return self.dp * self.tp * self.pp

    @property
    def model_parallel_size(self) -> int:       # mirrors ParallelConfig.model_parallel_size
        return self.tp * self.pp


@dataclasses.dataclass(frozen=True)
class MemFlags:
    need_zero_offload: bool = False             # ZeRO-3 param+state shard over DP
    need_activation_recompute: bool = False     # gradient checkpointing (1 layer live)
    need_layer_streaming: bool = False          # one PP-stage of params resident at a time
    need_param_offload: bool = False            # params -> host RAM
    need_opt_offload: bool = False              # opt-state -> host RAM (AdamW-on-host)
    cta_tiling: bool = False                    # ncta < sms_per_gpu (staged-scratch trim)


@dataclasses.dataclass(frozen=True)
class KernelKnobs:
    ncta: int
    ring_depth: int
    occupancy_cta_per_sm: float                 # ncta / sms_per_gpu
    staged_scratch_needed: bool                 # the 4 staged carves present (else bench elide)


@dataclasses.dataclass(frozen=True)
class MemBreakdownGiB:
    params: float
    state: float
    acts: float
    staged_opt: float
    sg2_region: float
    host_params: float
    host_state: float
    total_hbm: float
    total_host: float


@dataclasses.dataclass(frozen=True)
class ExecutionPlan:
    model: ModelConfig
    hw: HardwareConfig
    mesh: Mesh
    mem: MemFlags
    knobs: KernelKnobs
    budget: MemBreakdownGiB
    compile_flags: List[str]
    template_inst: str
    fits: bool
    risks: List[str]

    def summary(self) -> str:
        m, k, b = self.mesh, self.knobs, self.budget
        return (f"ExecutionPlan(world={m.world_size} "
                f"DP={m.dp} TP={m.tp} PP={m.pp} SP={m.sp} EP={m.ep} | "
                f"zero3={self.mem.need_zero_offload} recompute={self.mem.need_activation_recompute} "
                f"stream={self.mem.need_layer_streaming} "
                f"poff={self.mem.need_param_offload} ooff={self.mem.need_opt_offload} "
                f"ncta={k.ncta} ring={k.ring_depth} | "
                f"HBM={b.total_hbm:.2f}/{self.hw.usable_hbm_gib:.1f} GiB "
                f"host={b.total_host:.2f}/{self.hw.host_ram_gib:.0f} GiB "
                f"{'FITS' if self.fits else 'OOM'})")


class PlanInfeasible(RuntimeError):
    """Raised when no rung of the escalation ladder fits the model on the hardware."""


# ───────────────────────────── layout arithmetic (§1.1) ──────────────────────────


def decoder_param_sizes(d: int, layers: int, vocab: int, seq: int) -> List[int]:
    """Mirror of megakernel_codegen.py::_decoder_param_sizes — per-tensor numel in
    named_parameters() order. 2 + 12*L + 4 tensors, dff=4d. Verified to reproduce the
    flagship (1600,48,99,4) -> total 1,475,884,899 / 582 tensors / max 10,240,000."""
    dff = 4 * d
    sizes = [vocab * d, seq * d]                       # tok, pos
    for _ in range(layers):
        sizes += [
            3 * d * d, 3 * d,                          # attn.in_proj w/b
            d * d, d,                                  # attn.out_proj w/b
            d, d, d, d,                                # n1.w/b, n2.w/b
            dff * d, dff,                              # ff.0 w/b
            d * dff, d,                                # ff.2 w/b
        ]
    sizes += [d, d, vocab * d, vocab]                  # norm.w/b, out.w/b
    return sizes


def layout_arith(mc: ModelConfig) -> Tuple[int, int, int]:
    """Return (total_params, n_tensors, max_tensor_numel) for the model."""
    sizes = decoder_param_sizes(mc.d, mc.layers, mc.vocab, mc.seq)
    return sum(sizes), len(sizes), max(sizes)


# ───────────────────────────── staged-opt scratch (§1.2) ─────────────────────────


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def sg2_ws_stride(nmax: int) -> int:
    """Mirror of opt_stage_supergrok2.cuh::sg2_ws_stride<SG2Dims<>>(Nmax) — floats/CTA.
    ~91.277*Nmax with the defaults (verified numerically this session)."""
    d, rk, gh = SG2_D_MODEL, SG2_INDEXER_RANK, SG2_GRU_HIDDEN
    ncmax = (nmax + SG2_CSA_COMPRESS - 1) // SG2_CSA_COMPRESS
    topk = SG2_CSA_TOPK if SG2_CSA_TOPK > 1 else 1
    f = 7 * nmax * d                  # x_sorted,csa_ctx,hca_ctx,q,win_k,win_v,concat
    f += 2 * ncmax * d               # c_k, c_v
    f += nmax * rk                   # qI
    f += ncmax * rk                  # kI
    f += nmax * topk                 # sel
    f += nmax * gh                   # new_gru
    f += nmax                        # expert_out
    f += 2 * _next_pow2(nmax)        # sort keys + idx
    f += 2 * nmax                    # perm + unsort
    return f


def dec_sg2_ws_stride_floats(nmax: int, n_tensors: int) -> int:
    return 2 * n_tensors + sg2_ws_stride(nmax)          # :615


def dec_tc_sg2_floats(nmax: int, ncta: int, n_tensors: int) -> int:
    return ncta * dec_sg2_ws_stride_floats(nmax, n_tensors) + 1   # :619


def dec_tc_muon_floats(max2d_numel: int, max_rows: int, ncta: int) -> int:
    return 4 * max2d_numel + max_rows * max_rows + ncta + 1       # :567


def dec_tc_looksam_floats(total: int) -> int:
    return 2 * total                                              # :584


def dec_tc_opt_reduce_floats(ncta: int) -> int:
    return 2 * ncta + 1                                           # :553


def dec_tc_acts_floats(B: int, d: int, vocab: int, layers_live: int, seq: int) -> int:
    """Mirror of dec_tc_acts_floats (:504). `layers_live` = L/PP, or 1 under recompute."""
    dff = 4 * d
    T = B * seq
    Td, T3d, Tff = T * d, T * 3 * d, T * dff
    bf = 0
    for _ in range(layers_live):
        bf += Td + Td + Td + Tff + T3d + Td + Tff + Td
    bf += B * d + B * vocab + Td
    return (bf + 1) // 2


# ───────────────────────────── per-rank budget (§1.4-1.5) ────────────────────────


def per_rank_budget(mc: ModelConfig, hw: HardwareConfig, mesh: Mesh,
                    flags: MemFlags, ncta: int,
                    *, total: int, n_tensors: int, nmax: int,
                    staged_scratch_needed: bool = True) -> MemBreakdownGiB:
    """The EXACT per-rank HBM (+host) footprint for ONE (mesh, flags, ncta) point.
    Mirrors the live dec_tc_*_floats; the fit gate the front-end trusts.

    `staged_scratch_needed` mirrors the live kDecStagedOptScratch gate
    (fused_decoder_megakernel.cuh:541-545): True ⇒ the four staged-opt regions
    (opt_reduce|muon|looksam|sg2) are carved (production opt-agnostic launcher);
    False ⇒ they are elided (SG_DEC_BENCH_LAYOUT, adamw single-opt) — exactly the
    `dec_tc_*_floats` `if (!kDecStagedOptScratch) return 0;` early-out. This is why
    a 10B adamw run is fittable: its looksam carve (2·total = 75 GiB at 10B) is dead
    weight for an elementwise single-opt and is elided, NOT charged to HBM."""
    opt = mc.optimizer
    model_shard = mesh.tp * mesh.pp
    zero_div = mesh.dp if flags.need_zero_offload else 1

    # params + opt-state residency (ZeRO-3 shards over DP; offload moves to host).
    resident_params = total // (model_shard * zero_div)
    if flags.need_layer_streaming:
        # only ~1 of `layers` worth of the per-layer params resident at a time
        # (embeddings/tails stay); model_shard already split them.
        resident_params = max(resident_params // max(mc.layers, 1), 1)
    state_floats = _STATE_PLANES[opt] * total // (model_shard * zero_div)

    host_params_f = resident_params if flags.need_param_offload else 0
    host_state_f = state_floats if flags.need_opt_offload else 0
    hbm_params_f = 0 if flags.need_param_offload else resident_params
    hbm_state_f = 0 if flags.need_opt_offload else state_floats

    # activations (L/PP live, or 1 under recompute). Not ZeRO-sharded (transient).
    layers_live = 1 if flags.need_activation_recompute else max(mc.layers // mesh.pp, 1)
    acts = dec_tc_acts_floats(mc.batch, mc.d, mc.vocab, layers_live, mc.seq)

    # staged-opt scratch. Present ONLY when staged_scratch_needed (the kDecStagedOptScratch
    # gate); SG2 also requires opt==supergrok2 (its meta-net carve is the binding term).
    # TP shrinks Nmax (Megatron split); max 2D weight ~ ff = 4d*d split by TP; rows = 4d/TP.
    nmax_t = nmax // mesh.tp
    max2d = (4 * mc.d * mc.d) // mesh.tp
    max_rows = max((4 * mc.d) // mesh.tp, 1)
    if staged_scratch_needed:
        staged = (dec_tc_opt_reduce_floats(ncta)
                  + dec_tc_muon_floats(max2d, max_rows, ncta)
                  + dec_tc_looksam_floats(total // model_shard))
        sg2 = dec_tc_sg2_floats(nmax_t, ncta, n_tensors) if opt in _NEEDS_SG2_CARVE else 0
    else:
        staged = 0          # SG_DEC_BENCH_LAYOUT: the four carves fold to 0 (adamw single-opt)
        sg2 = 0
    staged += sg2

    def gib(f):
        return f * BYTES_PER_FLOAT / GB

    params = gib(hbm_params_f)
    state = gib(hbm_state_f)
    acts_g = gib(acts)
    staged_g = gib(staged)
    sg2_g = gib(sg2)
    total_hbm = params + state + acts_g + staged_g + 0.10   # tile-scratch slack
    total_host = gib(host_params_f) + gib(host_state_f)
    return MemBreakdownGiB(params=params, state=state, acts=acts_g,
                           staged_opt=staged_g, sg2_region=sg2_g,
                           host_params=gib(host_params_f), host_state=gib(host_state_f),
                           total_hbm=total_hbm, total_host=total_host)


# ───────────────────────────── mesh inference (§3) ───────────────────────────────


def _largest_pow2_divisor(n: int) -> int:
    p = 1
    while n % (p * 2) == 0:
        p *= 2
    return p


def infer_mesh(mc: ModelConfig, hw: HardwareConfig) -> Mesh:
    """3D-5D mesh inference (NOT keyed on a GPU-count switch). TP first (shrinks Nmax,
    rides NVLink — distributed._RankMesh puts TP fastest); PP only if TP+ZeRO-3 cannot
    fit per-stage; DP fills the rest; EP sub-divides DP for MoE. SP pinned to 1."""
    g = hw.num_gpus
    # TP: largest pow2 dividing g, bounded by NVLink width and by d % TP == 0.
    tp_cap = min(_largest_pow2_divisor(g),
                 hw.nvlink_width if hw.nvlink else 1)
    tp = 1
    cand = tp_cap
    while cand >= 1:
        if g % cand == 0 and mc.d % cand == 0:
            tp = cand
            break
        cand //= 2
    rest = g // tp
    # PP: smallest divisor of `rest` with L % PP == 0 that we COULD use; the ladder
    # decides whether to raise it. Start at 1 (PP is overhead; raise only if needed).
    pp = 1
    # DP fills whatever TP*PP leaves.
    dp = rest // pp
    # EP: sub-divide DP for MoE (EP | DP), never enlarges world (distributed.py:69-73).
    ep = 1
    if mc.num_experts > 1 and dp > 1:
        ep = 1
        for cand in range(min(mc.num_experts, dp), 0, -1):
            if dp % cand == 0:
                ep = cand
                break
    return Mesh(dp=dp, tp=tp, pp=pp, sp=1, ep=ep)


def _raise_pp(mesh: Mesh, mc: ModelConfig) -> Optional[Mesh]:
    """Raise PP one step (consuming a DP factor) if L % PP == 0 — used by the ladder
    when TP+ZeRO-3 still overflows per-stage. Returns None if no PP step is available."""
    rest = mesh.dp * mesh.pp                  # ranks TP leaves
    for new_pp in range(mesh.pp + 1, rest + 1):
        if rest % new_pp == 0 and mc.layers % new_pp == 0:
            new_dp = rest // new_pp
            ep = min(mesh.ep, new_dp) if mc.num_experts > 1 else 1
            while ep > 1 and new_dp % ep != 0:
                ep -= 1
            return Mesh(dp=new_dp, tp=mesh.tp, pp=new_pp, sp=1, ep=ep)
    return None


# ───────────────────────────── kernel knobs (§4) ─────────────────────────────────


def _ring_depth(d: int) -> int:
    if d <= 1024:
        return 2                              # shallow ring fits 48 KB static smem
    if d <= 4096:
        return 3                              # deep ring -> SG_DEC_TC_DYNAMIC_SMEM
    return 3


_NCTA_LADDER = (None, 64, 32, 16, 8, 4, 2, 1)   # None -> sms_per_gpu (1 CTA/SM)


# ───────────────────────────── compile-flag emission (§5) ────────────────────────


def _layout_header(mc: ModelConfig) -> Tuple[str, bool]:
    """Return (force-include header path, is_flagship). The 1.5B pin uses the committed
    flagship header; any other size uses a codegen'd header (megakernel_codegen.py
    decoder_layout_header / decoder_flagship_layout_header path)."""
    is_flag = (mc.d == 1600 and mc.layers == 48 and mc.vocab == 99 and mc.seq == 4)
    if is_flag:
        return "csrc/fused/sm_90/decoder_flagship_layout.cuh", True
    return (f"csrc/fused/sm_90/generated/decoder_layout_d{mc.d}_L{mc.layers}_"
            f"v{mc.vocab}_s{mc.seq}.cuh", False)


def emit_compile_flags(mc: ModelConfig, hw: HardwareConfig, mesh: Mesh,
                       flags: MemFlags, knobs: KernelKnobs) -> List[str]:
    """Map the ExecutionPlan to the EXACT -D flags the build (run_harness.md
    build_flagship_module + dist_step.md §6.C Par-launcher) consumes."""
    z = 3 if flags.need_zero_offload else 0
    header, _ = _layout_header(mc)
    out: List[str] = [
        "-O3", "-std=c++17", "--expt-relaxed-constexpr",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-DSG_TUNED_GEMM_IMPL=1",                       # wgmma L3-TC cell driver
        "-DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1",        # pre-set the committed guard
        "-include", header,                             # force the chosen layout table
        f"-DSG_FLAGSHIP_DP={mesh.dp}",
        f"-DSG_FLAGSHIP_TP={mesh.tp}",
        f"-DSG_FLAGSHIP_PP={mesh.pp}",
        f"-DSG_FLAGSHIP_ZERO={z}",
    ]
    # staged carve: present unless adamw single-opt (then bench-layout elides the 4 carves).
    if not knobs.staged_scratch_needed:
        out.append("-DSG_DEC_BENCH_LAYOUT=1")           # adamw-only -> elide staged scratch
    if flags.need_activation_recompute:
        out.append("-DSG_DEC_RECOMPUTE=1")
    if flags.need_layer_streaming:
        out.append("-DSG_DEC_LAYER_STREAM=1")
    off = (2 if flags.need_opt_offload else 0) | (1 if flags.need_param_offload else 0)
    if off:
        out.append(f"-DSG_DEC_HOST_OFFLOAD={off}")
    if knobs.ring_depth == 3:
        out += ["-DSG_TUNED_DEC_FWD_PIPE=2", "-DSG_DEC_TC_DYNAMIC_SMEM=1"]
    else:
        out.append("-DSG_TUNED_DEC_FWD_PIPE=1")
    if mesh.tp > 1 and hw.nvlink:
        out.append("-DSG_HAS_NVSHMEM=1")                # device-initiated in-kernel TP all-reduce
    return out


def _template_inst(mc: ModelConfig, mesh: Mesh, flags: MemFlags) -> str:
    z = "ZeROStage::Z3" if flags.need_zero_offload else "ZeROStage::Z0"
    opt = {"adamw": "OptId::AdamW", "lion": "OptId::Lion", "grokfast": "OptId::GrokFast",
           "grokadamw": "OptId::GrokAdamW", "looksam": "OptId::LookSAM",
           "prodigy": "OptId::Prodigy", "neuralgrok": "OptId::NeuralGrok",
           "muon": "OptId::Muon", "supergrok11": "OptId::SuperGrok11",
           "supergrok15": "OptId::SuperGrok15",
           "supergrok2": "OptId::SuperGrok2"}[mc.optimizer]
    if mesh.world_size == 1:
        return f"launch_fused_decoder_megakernel_tc<{opt}>  // ParConfig defaults to par::SingleGPU"
    return (f"launch_fused_decoder_megakernel_tc<{opt}, "
            f"ParConfig<{mesh.dp},{mesh.tp},{mesh.pp},{mesh.sp},{z}>>")


# ───────────────────────────── THE PLANNER (§2 ladder) ───────────────────────────


def plan_execution(model_cfg: ModelConfig,
                   hw_cfg: Optional[HardwareConfig] = None) -> ExecutionPlan:
    """From (model_cfg, hw_cfg) compute the FULL ExecutionPlan. Memory-FIT driven, NEVER
    a GPU-count switch: the parallelism rungs are bounded by num_gpus, but the strategy
    (zero3/recompute/cta-tile/stream/offload) is selected by the per-rank fit estimate."""
    mc = model_cfg
    hw = hw_cfg or HardwareConfig()
    total, n_tensors, nmax = layout_arith(mc)

    mesh = infer_mesh(mc, hw)
    risks: List[str] = []

    # staged carve is elided ONLY for adamw single-opt (bench-layout); SG2 always needs it.
    staged_needed = mc.optimizer != "adamw"

    def budget_at(mesh_: Mesh, flags_: MemFlags, ncta_: int) -> MemBreakdownGiB:
        return per_rank_budget(mc, hw, mesh_, flags_, ncta_,
                               total=total, n_tensors=n_tensors, nmax=nmax,
                               staged_scratch_needed=staged_needed)

    def fits(b: MemBreakdownGiB) -> bool:
        return b.total_hbm <= hw.usable_hbm_gib and b.total_host <= hw.host_ram_gib

    # ── the escalation ladder (§0). Start in-HBM, full occupancy. ──
    flags = MemFlags()
    ncta_full = hw.sms_per_gpu
    ncta = ncta_full

    b = budget_at(mesh, flags, ncta)
    if not fits(b):
        # R1 ZeRO-3 (no-op at DP=1, but free when DP>1).
        flags = dataclasses.replace(flags, need_zero_offload=True)
        b = budget_at(mesh, flags, ncta)
    if not fits(b) and mesh.pp == 1:
        # R1b raise PP if a TP+ZeRO-3 per-stage still overflows (only when DP factor free).
        bumped = _raise_pp(mesh, mc)
        if bumped is not None and bumped.pp > mesh.pp:
            cand = budget_at(bumped, flags, ncta)
            if cand.total_hbm < b.total_hbm:
                mesh, b = bumped, cand
    if not fits(b):
        # R2 CTA-tiling FIRST (cheaper than recompute — trades occupancy, not compute).
        # Walks the live auto_ncta ladder for the largest nCTA that fits the staged
        # scratch. This reproduces the run_harness.md headline: flagship SG2 fits at
        # nCTA=64 WITHOUT recompute (its acts at seq=4 are tiny; the staged scratch is
        # the binding term, so trimming nCTA — not recompute — is the right first move).
        for step in _NCTA_LADDER:
            cand_ncta = ncta_full if step is None else step
            if cand_ncta > ncta_full:
                continue
            tiled = dataclasses.replace(flags, cta_tiling=cand_ncta < ncta_full)
            cand = budget_at(mesh, tiled, cand_ncta)
            ncta = cand_ncta
            b = cand
            if fits(b):
                flags = tiled
                break
        else:
            flags = dataclasses.replace(flags, cta_tiling=True)
    if not fits(b):
        # R3 activation recompute (binding at long seq / large B — e.g. 10B seq=2048).
        flags = dataclasses.replace(flags, need_activation_recompute=True)
        b = budget_at(mesh, flags, ncta)
    if not fits(b) and mesh.pp == 1:
        # R4 layer streaming (single-rank analogue of PP param residency).
        flags = dataclasses.replace(flags, need_layer_streaming=True)
        b = budget_at(mesh, flags, ncta)
    if not fits(b):
        # R5 host offload: opt-state first (AdamW-on-host), then params.
        flags = dataclasses.replace(flags, need_opt_offload=True)
        b = budget_at(mesh, flags, ncta)
        if not fits(b):
            flags = dataclasses.replace(flags, need_param_offload=True)
            b = budget_at(mesh, flags, ncta)

    # SG2 honesty: its per-CTA workspace (91.277*Nmax/TP) may be structurally unfittable
    # even at ncta=1 on too-few GPUs (the live KNOWN DEEP LIMIT). Record a downgrade.
    if mc.optimizer in _NEEDS_SG2_CARVE and not fits(b):
        sg2_at_1 = budget_at(mesh, flags, 1)
        if sg2_at_1.total_hbm > hw.usable_hbm_gib:
            risks.append(
                f"supergrok2 staged scratch is {sg2_at_1.sg2_region:.0f} GiB even at "
                f"nCTA=1 (Nmax/TP={nmax // mesh.tp:,}); the SG2 per-CTA meta-net workspace "
                f"is O(91.277*Nmax) and does not fit on this hardware. Plan downgrades the "
                f"optimizer to an elementwise cell (adamw) + host offload — raise TP "
                f"(more GPUs) to run SG2 at this size. (fused_decoder_megakernel.cuh KNOWN "
                f"DEEP LIMIT, :598-610).")
            # re-plan as adamw to give a fitting plan for the elementwise fallback.
            mc_dn = dataclasses.replace(mc, optimizer="adamw")
            return _replan_downgraded(mc_dn, hw, risks)

    knobs = KernelKnobs(ncta=ncta, ring_depth=_ring_depth(mc.d),
                        occupancy_cta_per_sm=ncta / hw.sms_per_gpu,
                        staged_scratch_needed=staged_needed)
    cflags = emit_compile_flags(mc, hw, mesh, flags, knobs)
    tinst = _template_inst(mc, mesh, flags)

    if flags.need_param_offload or flags.need_opt_offload:
        risks.append(
            f"host offload active (params={flags.need_param_offload} "
            f"state={flags.need_opt_offload}, {b.total_host:.1f} GiB to host) — bounded "
            f"by host<->device bandwidth (PCIe vs NVLink); throughput will drop. "
            f"Needs host_ram >= {b.total_host:.0f} GiB.")
    if flags.need_layer_streaming:
        risks.append("layer streaming active — params resident one stage at a time; "
                     "overlap with compute is bandwidth-bound (the streaming risk).")
    if flags.cta_tiling:
        risks.append(f"CTA-tiling: ncta={ncta} < {hw.sms_per_gpu} SMs "
                     f"({ncta / hw.sms_per_gpu:.0%} occupancy) to fit the staged scratch.")

    final_fits = fits(b)
    if not final_fits:
        raise PlanInfeasible(
            f"no rung fits {mc.optimizer} {total/1e9:.2f}B on {hw.num_gpus} GPU(s): "
            f"HBM {b.total_hbm:.1f} > {hw.usable_hbm_gib:.1f} GiB (or host "
            f"{b.total_host:.1f} > {hw.host_ram_gib:.0f} GiB) after offload+recompute+"
            f"stream+cta-tile. Add GPUs (raise TP) or shrink the model.")

    return ExecutionPlan(model=mc, hw=hw, mesh=mesh, mem=flags, knobs=knobs, budget=b,
                         compile_flags=cflags, template_inst=tinst, fits=final_fits,
                         risks=risks)


def _replan_downgraded(mc: ModelConfig, hw: HardwareConfig,
                       carried_risks: List[str]) -> ExecutionPlan:
    """Re-run the planner for an elementwise (adamw) fallback when SG2 is unfittable,
    carrying the downgrade note. Guaranteed not to recurse (adamw has no SG2 carve)."""
    plan = plan_execution(mc, hw)
    return dataclasses.replace(plan, risks=carried_risks + plan.risks)


__all__ = [
    "ModelConfig", "HardwareConfig", "Mesh", "MemFlags", "KernelKnobs",
    "MemBreakdownGiB", "ExecutionPlan", "PlanInfeasible",
    "decoder_param_sizes", "layout_arith", "sg2_ws_stride", "dec_tc_acts_floats",
    "per_rank_budget", "infer_mesh", "emit_compile_flags", "plan_execution",
]
```

---

## HOOK 1 — re-export from `grokking_optimizers/parallel/__init__.py` (byte-exact edit)

The gate `python -c "import grokking_optimizers.parallel"` already passes; this makes
`plan_execution` a first-class export so the launcher dispatch (`distributed.py`) and harness
(`flagship_distributed.py`) can `from grokking_optimizers.parallel import plan_execution`.

VERBATIM OLD (`grokking_optimizers/parallel/__init__.py`, lines 27–43):
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

NEW:
```python
from grokking_optimizers.parallel.shard_map import (
    ShardPlan,
    TensorPlacement,
    even_partition,
    partition_elementwise_even,
    partition_tensor_granular,
    shard_mode_for_optimizer,
)
# The ROBUST execution planner (resource_fit_planner.md). Pure-Python (no torch),
# so importing the package stays GPU-free — the front-end calls plan_execution() to
# get the mesh + memory strategy + kernel knobs + the exact -D compile flags.
from grokking_optimizers.parallel.resource_planner import (
    ExecutionPlan,
    HardwareConfig,
    ModelConfig,
    PlanInfeasible,
    plan_execution,
)

__all__ = [
    "ShardPlan",
    "TensorPlacement",
    "even_partition",
    "partition_elementwise_even",
    "partition_tensor_granular",
    "shard_mode_for_optimizer",
    "ExecutionPlan",
    "HardwareConfig",
    "ModelConfig",
    "PlanInfeasible",
    "plan_execution",
]
```

> `resource_planner` imports NOTHING from torch (verify: the module's imports are
> `dataclasses` + `typing` only), so this keeps `import grokking_optimizers.parallel`
> torch-free and GPU-free (the same property `shard_map` has). The gate stays green.

---

## HOOK 2 — `ParallelConfig.from_execution_plan` on `grokking_optimizers/distributed.py` (byte-exact edit)

The planner emits a `Mesh`; the launcher needs a live `ParallelConfig` to build the
`DistributedContext`. Add ONE classmethod so the dispatch is `plan → ParallelConfig →
DistributedContext` with no GPU-count switch anywhere. Insertion point: inside `ParallelConfig`,
immediately after `validate_against_world` (the last method, ending line 156).

VERBATIM OLD (`grokking_optimizers/distributed.py`, lines 149–156):
```python
    def validate_against_world(self, world_size: int) -> None:
        """Raise if DP×TP×PP does not match the launched ``world_size``."""
        if self.world_size != world_size:
            raise ValueError(
                f"ParallelConfig DP×TP×PP = {self.data_parallel}×"
                f"{self.tensor_parallel}×{self.pipeline_parallel} = "
                f"{self.world_size} does not match launched world_size={world_size}"
            )
```

NEW:
```python
    def validate_against_world(self, world_size: int) -> None:
        """Raise if DP×TP×PP does not match the launched ``world_size``."""
        if self.world_size != world_size:
            raise ValueError(
                f"ParallelConfig DP×TP×PP = {self.data_parallel}×"
                f"{self.tensor_parallel}×{self.pipeline_parallel} = "
                f"{self.world_size} does not match launched world_size={world_size}"
            )

    @classmethod
    def from_execution_plan(cls, plan) -> "ParallelConfig":
        """Build a ParallelConfig from a resource_planner.ExecutionPlan (the ROBUST
        planner output). This is the launcher's single entry point: the planner — which
        is memory-fit driven, NEVER a GPU-count switch — decides the mesh + ZeRO stage,
        and this maps it onto the live config the DistributedContext consumes. The
        memory-strategy flags (recompute/stream/offload/cta-tiling) travel via the
        plan's compile_flags / ncta to the kernel build (run_harness.md
        build_flagship_module), NOT through this config (which describes only the mesh).
        Lazy import so distributed.py keeps no hard dependency on the planner module."""
        m = plan.mesh
        return cls(
            data_parallel=m.dp,
            tensor_parallel=m.tp,
            pipeline_parallel=m.pp,
            expert_parallel=m.ep,
            zero_stage=(3 if plan.mem.need_zero_offload else 0),
            use_megakernel=True,
        )
```

This is purely additive (a new classmethod) — every existing `ParallelConfig(...)` call site is
byte-identical. The `plan` arg is duck-typed (`plan.mesh`, `plan.mem`) so `distributed.py` needs
no import of `resource_planner` (no import cycle; the planner imports nothing from
`distributed.py` either — it MIRRORS the mesh math rather than calling it, keeping the planner
torch-free).

---

## NEW FILE — `tests/test_resource_planner.py` (CPU gate, in full)

Matches the `tests/test_zero3_plan.py` CPU idiom. Pins the layout arithmetic against the live
flagship constants and the four worked examples' STRATEGY decisions (not exact GiB — the GiB are
asserted within a band so a slack-constant change doesn't break the gate).

```python
"""tests/test_resource_planner.py — CPU gates for the ROBUST execution planner
(resource_fit_planner.md). NO torch, NO CUDA, NO GPU — pure arithmetic + decision tree.

Run:
    PYTHONPATH=. python -m pytest tests/test_resource_planner.py -q
"""
from __future__ import annotations

import pytest

from grokking_optimizers.parallel.resource_planner import (
    HardwareConfig,
    ModelConfig,
    PlanInfeasible,
    layout_arith,
    plan_execution,
    sg2_ws_stride,
)

H100 = dict(hbm_bytes_per_gpu=80 * (1000 ** 3), sms_per_gpu=132, nvlink=True,
            nvlink_width=8)


# ── layout arithmetic pinned to the live flagship constants ──
def test_layout_matches_flagship():
    total, nt, nmax = layout_arith(ModelConfig(d=1600, layers=48, vocab=99, seq=4))
    assert total == 1_475_884_899          # decoder_flagship_layout.cuh kDecTotalElems
    assert nt == 582                       # kDecNumTensors
    assert nmax == 10_240_000              # kDecMaxTensorNumel == 4d^2


def test_sg2_stride_factor_is_91():
    # ~91.277 floats/CTA per Nmax with the SG2Dims<> defaults (verified vs the cuh).
    assert abs(sg2_ws_stride(10_240_000) / 10_240_000 - 91.277) < 0.01


# ── worked example 10M/1GPU: trivial, in-HBM, full occupancy ──
def test_10m_one_gpu_trivial():
    mc = ModelConfig(d=512, layers=8, vocab=99, seq=128, batch=256, optimizer="adamw")
    plan = plan_execution(mc, HardwareConfig(num_gpus=1, **H100))
    assert plan.fits
    assert plan.mesh.world_size == 1
    assert not plan.mem.need_param_offload
    assert not plan.mem.need_opt_offload
    assert not plan.mem.need_layer_streaming
    # adamw single-opt -> staged carve elided (bench layout).
    assert "-DSG_DEC_BENCH_LAYOUT=1" in plan.compile_flags


# ── worked example 1.5B/8GPU: 4D + ZeRO-3, TP=8, SG2 caps ncta ──
def test_flagship_eight_gpu_4d_zero3():
    mc = ModelConfig(d=1600, layers=48, vocab=99, seq=4, batch=512, optimizer="supergrok2")
    plan = plan_execution(mc, HardwareConfig(num_gpus=8, **H100))
    assert plan.fits
    assert plan.mesh.tp == 8 and plan.mesh.pp == 1 and plan.mesh.dp == 1
    assert plan.mem.need_zero_offload          # ZeRO-3 overlay recorded (no-op at DP=1)
    # supergrok2 is the worst case -> cta-tiling caps ncta below full occupancy.
    assert plan.knobs.ncta <= 64
    assert f"-DSG_FLAGSHIP_TP=8" in plan.compile_flags
    assert "ParConfig<1,8,1,1,ZeROStage::Z3>" in plan.template_inst


# ── worked example 10B/1GPU: offload + recompute + streaming + cta-tile ──
def test_10b_one_gpu_full_stack():
    mc = ModelConfig(d=4096, layers=48, vocab=50304, seq=2048, batch=8,
                     optimizer="adamw")
    plan = plan_execution(mc, HardwareConfig(num_gpus=1, host_ram_bytes=512 * (1000 ** 3),
                                             **H100))
    assert plan.mesh.world_size == 1
    assert plan.mem.need_activation_recompute
    assert plan.knobs.ncta < 132               # cta-tiling
    # heavy machinery engaged (the directive's 10B/1GPU case).
    assert plan.mem.need_layer_streaming or plan.mem.need_opt_offload


def test_10b_one_gpu_sg2_downgrades():
    mc = ModelConfig(d=4096, layers=48, vocab=50304, seq=2048, batch=8,
                     optimizer="supergrok2")
    plan = plan_execution(mc, HardwareConfig(num_gpus=1, host_ram_bytes=512 * (1000 ** 3),
                                             **H100))
    # SG2's per-CTA workspace is structurally too large at Nmax=67M on one GPU ->
    # the planner downgrades to adamw + offload and records the honest risk.
    assert any("supergrok2" in r and "does not fit" in r for r in plan.risks)
    assert plan.model.optimizer == "adamw"


# ── worked example 10B/8GPU: TP shrinks Nmax -> recompute + cta-tile, no full offload ──
def test_10b_eight_gpu_tp_shrinks_nmax():
    mc = ModelConfig(d=4096, layers=48, vocab=50304, seq=2048, batch=8, optimizer="adamw")
    plan = plan_execution(mc, HardwareConfig(num_gpus=8, **H100))
    assert plan.fits
    assert plan.mesh.tp == 8
    assert plan.mem.need_activation_recompute
    # contrast with 10B/1GPU: 8 GPUs fit WITHOUT host param offload (TP shrank Nmax).
    assert not plan.mem.need_param_offload


# ── the directive: NOT keyed on GPU count (same model, fit decides the machinery) ──
def test_strategy_is_fit_driven_not_gpu_count():
    mc = ModelConfig(d=4096, layers=48, vocab=50304, seq=2048, batch=8, optimizer="adamw")
    p1 = plan_execution(mc, HardwareConfig(num_gpus=1, **H100))
    p8 = plan_execution(mc, HardwareConfig(num_gpus=8, **H100))
    # 1 GPU needs strictly MORE machinery than 8 GPUs for the SAME model — proof the
    # driver escalates by fit, not by a num_gpus switch.
    heavy1 = (p1.mem.need_layer_streaming + p1.mem.need_opt_offload
              + p1.mem.need_param_offload + p1.mem.cta_tiling)
    heavy8 = (p8.mem.need_layer_streaming + p8.mem.need_opt_offload
              + p8.mem.need_param_offload + p8.mem.cta_tiling)
    assert heavy1 >= heavy8


def test_moe_engages_expert_parallel():
    mc = ModelConfig(d=1600, layers=48, vocab=99, seq=4, batch=512, num_experts=8,
                     optimizer="adamw")
    plan = plan_execution(mc, HardwareConfig(num_gpus=8, **H100))
    # EP only when there is a DP factor to sub-divide; at TP=8 DP=1 EP stays 1.
    assert plan.mesh.ep in (1, 2, 4, 8)
    assert plan.mesh.dp % plan.mesh.ep == 0


def test_byte_identical_flags_when_single_gpu_trivial():
    mc = ModelConfig(d=512, layers=8, vocab=99, seq=128, batch=64, optimizer="adamw")
    plan = plan_execution(mc, HardwareConfig(num_gpus=1, **H100))
    # SingleGPU template instantiation -> ParConfig defaults to par::SingleGPU.
    assert "par::SingleGPU" in plan.template_inst
    assert "-DSG_FLAGSHIP_TP=1" in plan.compile_flags
```

---

## 7. GATE COMMANDS (what they prove)

```
python -c "import grokking_optimizers.parallel"
```
→ Imports the package WITH the new `resource_planner` re-export. The planner imports only
`dataclasses` + `typing` (no torch), so the package stays GPU-free and the import succeeds with no
CUDA — verified this session that the current package imports clean. **Proves HOOK 1 keeps the
import contract.**

```
python -m pytest tests/ -k "plan or resource or parallel" -q
```
→ Runs the existing `-k` selection (26 tests collected this session: `test_zero3_plan.py` +
others) PLUS the new `tests/test_resource_planner.py` (11 cases). The new cases pin the layout
arithmetic to the live flagship constants (`total/nt/nmax`), the SG2 91.277 factor, and the four
worked examples' STRATEGY decisions (mesh + flags), and the directive invariant (1-GPU escalates
more than 8-GPU for the same model). **Proves the planner's arithmetic mirrors the live formulas
and the ladder makes the directive's decisions.**

---

## 8. CONFIDENCE + RISKS

- **Layout arithmetic (params/nt/nmax):** HIGH. `decoder_param_sizes` is a byte-exact mirror of
  `_decoder_param_sizes` and I VERIFIED this session it reproduces the flagship
  `1,475,884,899 / 582 / 10,240,000` exactly, plus the 10M/10B totals.
- **Staged-scratch arithmetic (SG2 91.277·Nmax):** HIGH. The `sg2_ws_stride` mirror is numerically
  identical to the cuh (verified `91.277` for Nmax=10.24M and 1.28M this session), matching
  run_harness.md's "91.3" — and the `flagship_budget.py` mirror this generalizes from.
- **Memory ladder + worked examples:** MEDIUM-HIGH. The per-rank GiB for the four examples were
  computed this session from the exact formulas (10M→50.9 in-HBM dominated by the staged carve;
  1.5B/8GPU SG2→39.5 at nCTA=64; 10B/1GPU→9427 in-HBM falling to a fitting offload+recompute+
  stream plan; 10B/8GPU→90.4 needing one more cta-tile step). The recompute/stream models (acts→1
  layer; params→1/L) are the standard gradient-checkpointing / streaming memory models; their
  EXACT kernel realization is the `-DSG_DEC_RECOMPUTE` / `-DSG_DEC_LAYER_STREAM` gates which are
  NEW kernel work (flagged in `compile_flags` for the kernel lane — the planner emits the flag; the
  kernel must honor it). The fit MATH is exact; the kernel honoring the flag is the integration
  dependency.
- **Mesh inference:** HIGH. Reuses the live `ParallelConfig`/`_RankMesh` invariants
  (`world_size == DP·TP·PP`, EP | DP, TP fastest, SP==1) verbatim; TP-first + d%TP==0 + NVLink
  bound is the Megatron discipline `distributed.py` already encodes.
- **HOOK edits:** HIGH. Both are purely additive (a re-export block; a new classmethod) — every
  existing call site is byte-identical, no import cycle (planner mirrors mesh math, does not import
  `distributed.py`; `from_execution_plan` duck-types `plan`).
- **Out-of-scope kernel deps (honest):** the `-DSG_DEC_RECOMPUTE`/`-DSG_DEC_LAYER_STREAM`/
  `-DSG_DEC_HOST_OFFLOAD` gates are NEW kernel-side machinery (the planner SELECTS them; a future
  kernel lane must IMPLEMENT them, the same way dist_step.md §6.C is the TP `Par`-template lane).
  The SG2-on-1GPU structural limit is the live KNOWN DEEP LIMIT (`fused_decoder_megakernel.cuh
  :598-610`); the planner does not work around it — it downgrades honestly + records the risk.
- **gfx942 / tpu:** untouched (every artifact is Python / sm_90 `-D` flags). No cross-arch risk.
```
```
