# Phase-2 dormant-scaffolding REVIEW + ADVANCE — findings log

**Agent:** Opus 4.8 (1M), CPU-ONLY Phase-2-prep, 2026-06-16, branch `claude/h100-audit-maximal`, local commits only.
**Scope (disjoint from GPU-driver #11):** `csrc/fused/sm_90/{parallel_config,tp_layer,tp_transport,sharded_optimizer_kernel}.cuh`, `tests/hw/{sharded_optimizer_binding.cu,test_sharded_optimizer.py}`, plus the rest of the dormant Phase-2 surface (`tests/hw/{tp_loopback_binding.cu,pp_stage_binding.cu,test_tp_loopback.py,test_pp2_loopback_determinism.py,test_dp2_loopback_determinism.py}`, `grokking_optimizers/parallel/*`), and `.phase2/`.
**Method:** line-by-line static review against `/workspace/.parallelism_design.md` (§0.2 ABI, §0.4 opt taxonomy, §2.1-2.7 decomposition, §3.4 shard partition, §5.1-5.2 TP geometry/determinism, §6.4 cross-rank A/A/A, §7 1-GPU test plan). NO GPU runs (device owned by #11). Syntax-only host `nvcc -c` to /tmp scratch where useful.

**APPEND findings the instant they are discovered. Severity: BLOCKER (wrong math, would ship a bug to 8×) / MAJOR (correctness risk under some config) / MINOR (clarity/robustness) / NIT.**

---

## CONTEXT CORRECTION vs the task brief

The task brief states prior work "authored increment 0-1 scaffolding" and points at two stale worktrees. In fact the MAIN tree already contains a much larger, **committed** Phase-2 surface authored by Lane C (#25):

- `26517ac` — parallel_config + sharded-opt kernel + shard_map (increment 0-1)
- `ef433ac` — sharded-opt bit-parity test + DP=2 loopback determinism test + step graph capture (§7)
- `ab8c313` — TP loopback transport+layer, PP stage kernels+1F1B, ZeRO-3 gather/release+ckpt, §6.2 dist step (Lane C)

Implication for my task list:
- `test_sharded_optimizer.py` is **already complete** (not a stub) — full DP=1 bit-parity gate, 9 cells, CUDA-skip-guarded. ✔
- A **DP=2 loopback determinism test already exists** (`tests/hw/test_dp2_loopback_determinism.py`, 24 KB) and per REPORT.md ran 16/16 green on HW previously. ✔
- So the "write the DP=2 loopback determinism test + complete test_sharded_optimizer.py" task items are, on inspection, **already satisfied by committed work**. My value-add is therefore the RIGOROUS STATIC REVIEW of the shard math (the brief's task 2) — which has NOT been independently re-audited — plus any CPU-authorable gaps the prior pass left, plus honest flagging.

This matches the brief's "Honesty" clause: if scaffolding is already complete, say so plainly. I will still audit every line of the shard math before concluding.

---

## STATUS MAP (per increment / component)

Legend: DONE (authored + committed + reviewed-correct) · STUB · PENDING-GPU (authored, needs GPU to verify) · FOLLOW-UP (1-GPU-authorable, deliberately not done) · 8x (genuinely needs 8 GPUs).

| Increment / Component | Design § | File(s) | Status (pre-review) |
|---|---|---|---|
| Compile-time ParConfig<DP,TP,PP,SP,Z> + SingleGPU + CommCtx | §1.1/§1.2 | parallel_config.cuh | DONE |
| Allow-list instantiation compile gate | §7.2 | tests/hw/test_parallel_instantiation.py | DONE (CPU/nvcc) |
| Sharded-opt kernel (flat grid-stride, elementwise) | §2.3 | sharded_optimizer_kernel.cuh | DONE |
| Sharded-opt DP=1 bit-parity gate | §7.3 | test_sharded_optimizer.py (+binding.cu) | PENDING-GPU |
| ZeRO shard map (even + tensor-granular) | §3.4 | parallel/shard_map.py | DONE (review pending) |
| ZeRO-3 flat plan / gather / release / checkpoint | §3.2(a) | parallel/zero3.py | DONE (review pending) |
| §6.2 fused_train_step_distributed | §6.2 | parallel/distributed_step.py | DONE (review pending) |
| DP=2 loopback cross-rank A/A/A | §6.4/§7.1 | tests/hw/test_dp2_loopback_determinism.py | PENDING-GPU |
| §6.2 distributed-step gate (world1 identity + world2 loopback) | §7.1 | tests/hw/test_distributed_step.py | PENDING-GPU |
| ZeRO-3 round-trip gate | §3.2(a)+ckpt | tests/hw/test_zero3_roundtrip.py | PENDING-GPU |
| TP transport seam (loopback + NVSHMEM surface + fixed-order reduce) | §5.2/§5.3 | tp_transport.cuh | DONE (review pending) |
| TP shard geometry/table/pack + sharded wgmma tiles + reduce-point map | §5.1 | tp_layer.cuh | DONE (review pending) |
| TP loopback gate (TP∈{2,4}) | §7 | tests/hw/test_tp_loopback.py (+binding.cu) | PENDING-GPU |
| PP stage kernels + 1F1B + handoff | §4.1/§4.2 | pp_stage_decoder_tc.cuh + parallel/pipeline.py | DONE (review pending) |
| PP=2 loopback gate | §4 | tests/hw/test_pp2_loopback_determinism.py (+binding.cu) | PENDING-GPU |
| PP layer-range patch (tracked-file change) | §4.1 | .phase2/patches/0001-*.patch | DONE (as patch, not applied) |
| CPU gates (pipeline schedule, zero3 plan) | §7 | tests/test_pipeline_schedule.py, tests/test_zero3_plan.py | DONE (CPU, were green) |
| TP insertion into production kernel body | §5 | (4 marked points in tp_layer.cuh) | 8x (transport-choice-dependent) |
| NVSHMEM-TP transport validation + §5.4 go/no-go | §5.4 | tests/hw/test_tp_nvshmem_gate.py | 8x (NVSHMEM not installed) |
| Real 1→8 scaling / ZeRO-3 OOM threshold / PP bubble | §8 | — | 8x |
| vit/mamba PP/TP twins | §4/§5 | — | FOLLOW-UP |

---

## DESIGN EQUATIONS — the invariants the shard math MUST satisfy (review checklist)

1. **ABI (§0.2):** params/grad = flat `float[total]`, `total = Σ kDecSizes[i]` over 30 tensors, concat in `named_parameters()` order. state = `[m|v|extra]+loss`. `rebase_state<Opt>` only ADDS the per-tensor offset to per-element pointers — so flat index `i` pairs `params[i]↔grad[i]↔m[i]↔v[i]↔…` identically on the in-kernel and sharded paths.
2. **Decomposition (§2.1):** `[fwd+bwd]→reduce_scatter(grad)→[sharded-opt over owned shard]→all_gather(params)`. Each rank ends reduce-scatter with the reduced grad for ITS shard only; sharded-opt writes only its shard; all-gather re-replicates.
3. **B2 cut (§2.2):** ZeRO≥2 ⇒ kernel `if constexpr (Par::kShardOptGrad) return;` after B2; sharded-opt kernel is the tail. ZeRO≤1 ⇒ P3 in-kernel as today.
4. **Determinism (§2.7/§5.2/§6.4):** cross-rank reduction MUST be FIXED-ORDER (ascending rank/pe, fp32/fp64 accumulate) — the structural order, not timing. Mirrors the in-kernel ascending-CTA fp64 sum. NON-NEGOTIABLE for A/A/A.
5. **Per-tensor constraint (§0.4/§3.4):** muon/SG11/SG15/SG2 are per-TENSOR/per-MATRIX — a flat-element shard that splits a tensor across ranks BREAKS them ⇒ they require tensor-granular partition. Elementwise {adamw,lion,grokfast,+global-scalar cells} may use flat-even.
6. **TP geometry (§5.1):** attn in_proj/QKV column-parallel (3d/TP cols, head-aligned), out_proj row-parallel (d/TP rows), all-reduce after out_proj; ff0 col-parallel (d→dff), ff2 row-parallel (dff→d), all-reduce after ff2. Two all-reduces per layer. Backward mirrors with all-reduce on dX of the column-parallel pieces. dW shards are EXACT slices of the unsharded grad.
7. **Single-GPU guarantee (§1.2):** SingleGPU = ParConfig<1,1,1,1,Z0>, kEmitComm==false ⇒ every comm branch folds to zero code ⇒ PTX-identical to legacy `<Opt>`.

Below: line-by-line findings against these.

---

## FINDINGS (append live)

(none yet — review in progress)
