# Crux deep-dive (first-hand, lead): the config-derivation system (the "robust workload×hardware→config")

Read `grokking_optimizers/parallel/resource_planner.py` (627L) in full + grepped the kernel for the flags it
emits + read `tests/test_resource_planner.py`. This is the subsystem the user emphasized ("CTA tiling etc. are
NOT a hard-set thing; the codebase needs a robust system to get the right config"). VERDICT: the design is
exactly that, the PLANNER is real and rigorous, but it is AHEAD OF THE KERNEL on the harder memory strategies.

## What the planner is (real, rigorous)
`plan_execution(ModelConfig, HardwareConfig) -> ExecutionPlan`. Pure Python, no torch/CUDA. It MIRRORS the live
kernel scratch formulas (`dec_tc_*_floats`, `sg2_ws_stride` ≈ 91.277·Nmax, `_decoder_param_sizes`) so it computes
an EXACT per-rank HBM/host budget BEFORE any GPU work and emits the exact `-D` flags + `ParConfig<dp,tp,pp,sp,z>`
template instantiation. Verified faithful: `layout_arith(1600,48,99,4)` → 1,475,884,899 params / 582 tensors /
nmax 10,240,000 (== `decoder_flagship_layout.cuh` constants).

### The derivation is FIT-DRIVEN, not a GPU-count switch (user directive upheld)
- `infer_mesh`: 3D-5D mesh — TP = largest pow2 | num_gpus, bounded by NVLink width AND d%TP==0 (TP first, rides
  NVLink, shrinks Nmax); PP starts at 1 (overhead, raised only if needed); DP fills the rest; EP sub-divides DP
  for MoE; SP pinned to 1 (present in the mesh type but not yet activated). `resource_planner.py:364-393`.
- The escalation ladder `plan_execution:514-564` (the heart): start full-occupancy in-HBM →
  **R1** ZeRO-3 → **R1b** raise PP → **R2 CTA-tiling** (walk nCTA ladder 64..1 — the "CTA tiling isn't fixed"
  the user meant; chosen by fit, before recompute because it trades occupancy not compute) → **R3** recompute →
  **R4** layer-streaming → **R5** host-offload (opt-state then params). Same ladder for 10M/1GPU, 1.5B/8GPU,
  10B/1GPU. SG2 honesty: if its 91.277·Nmax/TP carve is unfittable even at nCTA=1, it records a downgrade and
  re-plans as adamw + "raise TP (more GPUs) to run SG2 at this size."

## ⚠️ The design-vs-implementation gap (grep-verified against csrc)
The planner emits flags; not all are consumed by the decoder kernel:
| Rung / flag                | Decoder kernel honors it? | Evidence |
|----------------------------|---------------------------|----------|
| R1 ZeRO-3 (`ParConfig Z3`) | **YES (real)**            | `parallel_config.cuh:41,74` ZeROStage enum + ParConfig template; `zero3.py` FlatShardPlan |
| R2 CTA-tiling (nCTA)       | **YES (real)**            | megakernel parameterized on nCTA (per-CTA grad partials, cta-major) |
| bench-layout elision      | **YES (real)**            | `SG_DEC_BENCH_LAYOUT` — 5 csrc refs |
| dynamic smem (deep ring)  | **YES (real)**            | `SG_DEC_TC_DYNAMIC_SMEM` — `fused_decoder_megakernel.cuh:467` |
| R1b raise PP              | **partial**               | `pp_stage_decoder_tc.cuh` exists; PP=2 loopback gated, not exercised at flagship |
| R3 `SG_DEC_RECOMPUTE`     | **NO (unwired)**          | **0 csrc references** — the decoder does not implement activation recompute |
| R4 `SG_DEC_LAYER_STREAM`  | **NO for decoder**        | **0 csrc references**; layer-streaming IS real for **Mamba** (`kMbStreamSmem`, the smem redesign) but the decoder has no equivalent gated by this flag |
| R5 `SG_DEC_HOST_OFFLOAD`  | **NO (scaffold only)**    | **0 csrc references in the kernel**; only `mem_config.cuh` + `dispatch.cpp` mention offload config |

`tests/test_resource_planner.py` (10/10) asserts only the planner's ARITHMETIC + decision tree — NOT that the
kernel honors the emitted flags. So "10/10 planner tests" ≠ the strategies run end-to-end.

## Net assessment (what to tell the user)
- The robust config-derivation system the user described EXISTS and is genuinely good: a fit-driven ladder,
  faithful to the kernel's real memory formulas, emitting exact flags. CTA-tiling, ZeRO-3, the 3D-5D mesh are
  real. This is NOT the naive "10M→1GPU / 1.5B→multiGPU" I first stated.
- BUT the planner is AHEAD OF THE KERNEL: for the decoder, recompute / layer-streaming / host-offload are
  emitted-but-unimplemented. So the planner can return a "fits" plan for, e.g., 10B-on-1-GPU that the decoder
  kernel cannot actually execute (those flags are no-ops; the kernel would run full-memory and the launcher
  OOM-guard would return cudaErrorMemoryAllocation). Mamba is further along (real layer-streaming).
- Genuinely working memory/parallel levers for the decoder TODAY: TP mesh (currently REPLICATED-compute per the
  WIP data-path fix, not real sharding), ZeRO-3 over DP, CTA-tiling (nCTA), bench-layout elision. That is the
  honest envelope of "what can actually run across the 8×H100 right now."
