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

### F1 — parallel_config.cuh — REVIEWED CLEAN
`csrc/fused/sm_90/parallel_config.cuh` (all 124 lines). ParConfig<DP,TP,PP,SP,Z> derived gates are arithmetically correct: `kIsSingleGPU = (DP==1&&TP==1&&PP==1&&SP==1)` (correctly EXCLUDES ZeRO from the single-GPU test — ZeRO with all degrees 1 is still a degenerate no-comm world, and `kEmitComm=!kIsSingleGPU` correctly stays false; ZeRO-vs-comm is separately gated by kShardOptGrad/kShardParams), `kShardParams=(Z==Z3)`, `kShardOptGrad=(Z>=Z2)`, `kTPComm=(TP>1)`, `kPPStage=(PP>1)`. static_assert SP==1 and degrees>=1 present. SingleGPU=ParConfig<1,1,1,1,Z0> matches the §1.2 byte-identity contract. CommCtx is an empty POD with single-GPU defaults — correct stable seam. **No bug.** Severity: n/a (clean).

### F2 — sharded_optimizer_kernel.cuh — REVIEWED CLEAN
`csrc/fused/sm_90/sharded_optimizer_kernel.cuh` (167 lines). The flat grid-stride sharded-opt kernel correctly: sets `st_shard.lr=lr` (single rate source, matches P3), strides `[0,shard_numel)` in int64 (no overflow at d=2048 scale = 25.4M), calls the verbatim `apply_optimizer<Opt>` from opt_components.cuh (zero new math → bit-parity with P3, §2.3). No GridBarrier (correct: each element written once, single phase). Launcher caps grid at max_grid with int64 want then narrows — `int grid = (int)(want<max_grid?want:max_grid)` is safe (want is clamped < 65535 before the cast). `shard_numel<=0` early-returns cudaSuccess. **No bug.** The per-tensor-vs-elementwise taxonomy in the header comment matches design §0.4 exactly. Severity: n/a (clean).

### F3 — shard_map.py — REVIEWED CLEAN (byte-for-byte verified vs distributed.py)
`grokking_optimizers/parallel/shard_map.py` (273 lines). `even_partition` is **byte-for-byte identical** to `distributed.py::_even_partition` (:717): same `per=ceil(numel/world)`, same `start=min(rank*per,numel)`, `end=min(start+per,numel)` — covers [0,numel) exactly, no overlap, empty for out-of-range ranks. Verified by direct comparison. `partition_tensor_granular` LPT is deterministic (sort by `(-numel, name)`, assign to `min((load,rank))`, then re-sort slices into named_parameters() order) — stable shard boundaries per §2.7. Taxonomy ELEMENTWISE/PER_TENSOR matches §0.4 (muon/SG11/SG15/SG2 per-tensor; everything else elementwise; global-scalar cells correctly elementwise since the scalar all-reduce is separate from the partition). `shard_mode_for_optimizer` raises on unknown (loud, no silent default). **No bug.** Severity: n/a (clean).

### F4 — zero3.py — REVIEWED CLEAN (offset arithmetic verified)
`grokking_optimizers/parallel/zero3.py` (332 lines). FlatShardPlan flat-coordinate mapping verified: tensor-granular maps each whole-tensor slice via prefix-sum `offsets[name]` (`offsets[name]+s, offsets[name]+e`, then sorted) — correct. Elementwise maps to ONE contiguous `even_partition(total, world, r)` slice of [0,total) — matches the DP-loopback/sharded-kernel convention (NOT per-tensor even split). `owned()` shard-coordinate mapping `(fs,fe,ss,se)` with running `o` is correct and consistent between gather_full/release. `gather_full` elementwise collective path uses the SAME padded all_gather_into_tensor convention as distributed.py::all_gather_params (`per=ceil(total/world)`, pad to per*world, slice back with min clamps) — verified match. world=1 degenerate copy correct. peers path validates every-rank-once + fingerprint match (loud). No-peers-no-dist raises (no stale gather). Checkpoint save/load fingerprint-guards mode/world/rank/total (loud refuse on drift). **No bug.** Severity: n/a (clean).
NOTE (not a bug, flagged for the 8x builder): `gather_full` tensor-granular branch issues one `dist.broadcast` per slice over all 30 tensors — correct but O(tensors) collectives; the design (§3.4) accepts this lumpiness. Left as-is.

### F5 — tp_transport.cuh — REVIEWED CLEAN
`csrc/fused/sm_90/tp_transport.cuh` (233 lines). `tp_allreduce_sum_fixed_order` reads ascending-pe (`for pe=0..P`) accumulating in fp32 with `#pragma unroll 1` (prevents compiler reassociation) — exactly the §5.2 structural-order determinism requirement (fp32 accumulate is sanctioned by §5.2 "fp32 or fp64 accumulate"; the analogized in-kernel CTA-0 reduce uses fp64 but the order, not the width, is what makes A/A/A hold). Symmetric-heap addressing (`base + pe*stride + off`) is identical between LoopbackTransport and (the -DSG_HAS_NVSHMEM-only) NvshmemTransport — one math path, swapped translation. `pe_of_cta`/`cta_within_pe`/`ctas_per_pe` require nCTA%P==0 (asserted in launchers per comment). NvshmemTransport.rendezvous fences GridBarrier→single-CTA nvshmemx_barrier_all_block→GridBarrier (the §5.2 two-barrier-no-deadlock discipline). NVSHMEM surface compiles only under the flag (honest absence). **No bug.** Severity: n/a (clean).

### F6 — tp_layer.cuh — Megatron geometry REVIEWED CLEAN; one DOC-DRIFT finding
`csrc/fused/sm_90/tp_layer.cuh` (280 lines). The TP shard MATH is correct:
- Shard table (lines 107-138): 30 tensors. in_proj/in_proj_b=ColQKV, out_proj=Row, out_b=Replicated, ff0/ff0_b=Col, ff2=Row, ff2_b=Replicated, tok/pos/LN/head=Replicated. **Matches design §5.1 table exactly.**
- `tp_colqkv_full_row(d,P,r,i)` (lines 174-179): `per=d/P; blk=i/per∈{0,1,2}; off=i%per; return blk*d + r*per + off`. VERIFIED correct: full in_proj is [3d,d] rows [q(d)|k(d)|v(d)]; rank r owns rows [r*per,r*per+per) within each block; dense buffer is [q_own|k_own|v_own]=3*(d/P) rows. The head-aligned claim holds because d/P with kHeads=4, kDhead=32 keeps heads whole for P∈{2,4} (d/P = 64 or 32 = whole multiples of kDhead=32).
- `tp_shard_extents` (lines 249-258): Col/ColQKV→(Nout/P, Kin); Row→(Nout, Kin/P); default→(Nout,Kin). Correct Megatron geometry. (ColQKV Nout_full=3d → Nout_local=3d/P, consistent with the 3-block dense pack.)
- dW exact-slice property (header lines 236-247): col-parallel dW_own = rows[own_lo,own_hi) of full dW; row-parallel dW_own = cols[own_lo,own_hi). Comm-free, bit-exact slices. Correct.
- The per-tile block functions correctly route to the production `dectc::dectc_gemm_fwd_f32<N>` / `dectc_gemm_dx_f32<N>` **fp32-W overloads** (which model_stage_decoder_tc.cuh:730-810 added specifically for the TP path — confirmed present). Reuses production wgmma tiles, zero new GEMM math.
- Divisibility: kD=128, kDff=512, kHeads=4 (dec_weights.cuh:103-109) all divide TP∈{2,4}. `tp_own_range`/`tp_heads_per_rank` assert P|extent (host plan). Correct.

**F6a [MINOR / DOC-DRIFT]** — `tp_layer.cuh` lines 56-65 (the §5.2 insertion map for the 8x megakernel builder) cite STALE line numbers in `model_stage_decoder_tc.cuh`. That production header grew from ~1100 to 1946 lines since the Lane C pass (split-K / GEMM-interleave kernel-track edits). Correct current lines:
  - ① out_proj fwd reduce: cited `:766-768` → ACTUAL GEMM `a = X_ctx @ out_w^T` at **1085-1087**, residual fold at 1093.
  - ② ff2 fwd reduce: cited `:797-799` → ACTUAL GEMM `ff2 = X_gact @ ff2_w^T` at **1116-1118**, fold at 1124.
  - ②' ff0 dX bwd reduce: cited `:1075` → ACTUAL `dx1 += dff0 @ ff0_w` GEMM at **1392-1395**.
  - ①' in_proj dX bwd reduce: cited `:1104` → ACTUAL `dx_in_attn = dqkv @ in_w` GEMM at **1421-1423**, residual fold `sc.dh += sc.work` at 1427.
  FIX: update the comment in tp_layer.cuh (my-scope file) to the current lines so the 8x TP-insertion builder lands at the right call sites. This is documentation only — NO math change, NO production-TU edit. Severity: MINOR (the math/transport are gated correctly; this is builder guidance that would otherwise send the 8x author to the wrong lines inside a GEMM helper).
