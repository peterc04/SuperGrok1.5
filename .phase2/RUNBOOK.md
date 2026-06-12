# Phase-2 RUNBOOK — single-GPU validation of the #25 parallelism authoring (Lane C)

**Written 2026-06-12 on a CPU-only lane.** Every test below was AUTHORED + its
CUDA compiled (nvcc sm_90a, compile-proof in `.phase2/REPORT.md`) but **NOT
RUN** (GPU owned by another lane). Run top-to-bottom on the 1×H100 box first;
the 8×-only residual is the last section. Wall-times are estimates calibrated
to the ef433ac pre-tests (same JIT/build/cell-build machinery): first
invocation of each suite pays a one-time JIT build (~2-4 min, cached in
`/workspace/.torch_ext` afterwards).

Environment for every command:

```bash
cd /workspace/SuperGrok1.5
export SCCACHE_DIR=/dev/shm/sccache TMPDIR=/dev/shm/tmp
export TORCH_CUDA_ARCH_LIST=9.0a
# PYTHONPATH=. on every python invocation (as the ef433ac suites do).
```

PRE-FLIGHT (the ef433ac discipline): `CUDA_MPS_PIPE_DIRECTORY=/nonexistent
python wiring_check.py --require-all` must still report 33/33 L3-TC — none of
the phase-2 files touch the production `_ops` build, so any regression here is
NOT from this lane.

---

## 0. CPU gates (already run GREEN on the authoring box, 2026-06-12)

```bash
PYTHONPATH=. python -m pytest tests/test_pipeline_schedule.py tests/test_zero3_plan.py -q
# EXPECTED: 55 passed  (measured: 55 passed in ~2 s)
PYTHONPATH=. python -m pytest tests/hw/test_parallel_instantiation.py -q
# EXPECTED: 3 passed (nvcc-only; pre-existing §7.2 gate, unaffected)
```

Re-run them on the GPU box only as a sanity echo; they cannot regress from
GPU presence.

## 1. TP loopback gate — `tests/hw/test_tp_loopback.py`  [NEW, GPU, ~4 min]

```bash
PYTHONPATH=. python -m pytest tests/hw/test_tp_loopback.py -q -s
# or: PYTHONPATH=. python tests/hw/test_tp_loopback.py
```

* **What it proves:** the Megatron column→row TP pair (col-parallel wgmma GEMM
  → gelu → row-parallel partial GEMM → publish → rendezvous → FIXED-ORDER
  ascending-pe fp32 all-reduce, + the conjugate backward + dW/db shards) on
  TP ∈ {2,4} virtual ranks over the `LoopbackTransport` symmetric heap —
  i.e. **every part of in-kernel TP except the physical transport** (design
  §5.1/§5.2). Uses the production `dectc_gemm_*` wgmma tiles.
* **Pass criteria (per TP degree):** (a) cross-virtual-rank outputs
  BIT-IDENTICAL; (b) 3 reruns BIT-IDENTICAL (A/A/A); (c) TP+transport ==
  serial chunked-order reference BIT-EXACT (transport-neutrality — the
  loopback's honesty assert); (d) dW0/db0/dW1/db1 shards == exact slices of
  the unsharded grads BIT-EXACT; (e) vs unsharded full-K reference rel < 3e-5
  (reported, reassociation-only).
* **Expected wall:** JIT ~3 min (first run), then <10 s for both degrees.
* **On failure:** (c) failing = transport bug (heap stride / barrier);
  (d) failing = shard-pack map bug (`tp_pack_pair_shards` / tp_layer maps);
  (a,b) failing = a timing-dependent order leaked into the reduce.

## 2. PP=2 loopback gate — `tests/hw/test_pp2_loopback_determinism.py`  [NEW, GPU, ~8 min]

```bash
# REQUIRED FIRST (tracked-file change shipped as a patch, lane discipline):
git apply .phase2/patches/0001-dectc-layer-range-pp.patch
# sanity: production cell still compiles + PTX-identity claim spot-check
#   (optional, ~2 min): see REPORT §"PP patch" for the exact nvcc -ptx diff cmd.

PYTHONPATH=. python -m pytest tests/hw/test_pp2_loopback_determinism.py -q -s
```

* **What it proves:** the PP stage cut (design §4.1): stage0-fwd →
  stage1-fwd+bwd (loss + fp32 dh handoff) → stage0-fwd-recompute+bwd, both
  stages on cuda:0 sharing one workspace (zero-copy acts boundary) —
  **the stage composition is BIT-IDENTICAL to the single-launch fused step**:
  (a) grad bitwise == production `return_grad` grad (all 30 tensors);
  (b) loss bitwise; (c) A/A/A ×3; (d) closure: sharded-opt(PP grad) ==
  production P3 params bitwise. Cells: decoder × {adamw, grokfast}.
  Also `test_pp2_stage_ownership_matches_python_plan`: kernel-side
  `PPStageSpec::owns_tensor` == `pipeline.stage_tensor_ownership` (single
  source cross-check), disjoint + complete over the 30 tensors.
* **Pass criteria:** all asserts bitwise (`torch.equal`), no tol.
* **Expected wall:** JIT ~4 min (the binding pulls the full TC header chain),
  then ~1-2 min/cell (two fused-scale launches + production step per A/A/A
  rep ×3).
* **SKIP behavior:** without the patch the test SKIPS loudly with the apply
  instruction (and the JIT would `#error` — no silent path).
* **On failure:** per-tensor delta breakdown prints; non-owned-tensor deltas
  = ownership filter bug; layer-0-only deltas = dh handoff (fp32 carrier)
  bug; everything-deltas = nCTA mismatch (production launcher cap vs the
  binding's `ncta_cap=0` — both must resolve to #SMs; check
  `wiring_check`-reported launch shape).
* **Afterwards:** `git checkout csrc/fused/sm_90/model_stage_decoder_tc.cuh`
  to unapply if the lane must stay patch-free, or keep it applied for the 8×
  window (the patch is the intended merge).

## 3. ZeRO-3 round-trip gate — `tests/hw/test_zero3_roundtrip.py`  [NEW, GPU, ~6 min]

```bash
PYTHONPATH=. python -m pytest tests/hw/test_zero3_roundtrip.py -q -s
```

* **What it proves (design §3.2(a) + checkpoint/resume):** virtual world=2 on
  one GPU; the sharded 2-step chain (fused step → per-rank
  `sharded_opt_step` on the owned slice via `Zero3FlatParamStore` →
  gather_full) == the production 2-step chain BIT-EXACTLY; save-at-step-1 →
  resume → step-2 BIT-IDENTICAL to the uninterrupted run (params + the
  [m|v|extra] state planes); gather/release keeps shards consistent;
  drifted-plan (world=4) and wrong-rank loads RAISE. Cells: decoder ×
  {adamw, grokfast}. Reuses ef433ac's `sharded_optimizer_binding` (cached
  JIT).
* **Pass criteria:** every `torch.equal` assert + both loud-guard raises.
* **Expected wall:** ~2 min/cell (4 fused-scale steps + sharded applies).
* **On failure:** `step1_eq` false = slice plumbing (owned() ranges);
  `resume_*` false = checkpoint serialization dropped bits (must be
  impossible — tensors round-trip via torch.save CPU clones);
  guard false = fingerprint regression.

## 4. §6.2 distributed-step gate — `tests/hw/test_distributed_step.py`  [NEW, GPU, ~10 min]

```bash
PYTHONPATH=. python -m pytest tests/hw/test_distributed_step.py -q -s
```

* **What it proves:** the production-shaped
  `grokking_optimizers.parallel.distributed_step.fused_train_step_distributed`
  (design §6.2, the per-rank [0]-[5] sequence promoted out of the ef433ac
  worker into ONE importable implementation, sharded apply dependency-injected
  from the DELIVERABLE-1 binding):
  - `test_world1_decomposed_identity` (adamw, grokfast): world=1 FORCED
    decomposition == plain `fused_train_step`, BIT-IDENTICAL params over 2
    consecutive steps (the §7.1 degenerate-collective identity).
  - `test_world2_loopback_through_module` (adamw): torchrun 2-rank-on-cuda:0
    (NCCL_HOSTID trick) 2-step chain — cross-rank params bitwise identical per
    step; final params within 3e-5 rel of the plain single-GPU 2-step chain
    (the batch-shard reduce reassociation, same class the DP=2 gate measured).
* **Pass criteria:** all `torch.equal` asserts in gate 1; cross-rank bitwise +
  rel < 3e-5 in gate 2; worker exits 0.
* **Expected wall:** ~3 min (world=1, 2 cells) + ~5 min (torchrun spawn).
* **On failure:** world=1 diverging = orchestration bug in the module ([0]
  gather/scatter or scalar plumbing); world=2 cross-rank diverging = the
  fixed-order reduce regressed (compare with test_dp2_loopback, which is the
  same flow inline).

## 5. Regression echo of the ef433ac pre-tests  [GPU, ~15 min]

The phase-2 files share machinery with the 16/16-green pre-tests; re-run them
to prove no interference:

```bash
PYTHONPATH=. python -m pytest tests/hw/test_sharded_optimizer.py -q -s          # 9 cells, bit-exact
PYTHONPATH=. python -m pytest tests/hw/test_dp2_loopback_determinism.py -q -s   # 4 cells, A/A/A
PYTHONPATH=. python -m pytest tests/hw/test_step_graph_capture.py -q -s         # graph capture
CUDA_MPS_PIPE_DIRECTORY=/nonexistent python wiring_check.py --require-all       # 33/33 L3-TC
```

## 6. Suggested order & total budget

0 → 1 → 2 → 3 → 4 → 5. Total ≈ 45-60 min including first-time JIT builds.
Each suite is independent; on any failure, capture the full `-s` output into
`.phase2/` and continue with the rest (failures are attributable per layer —
that is the point of the bring-up decomposition).

---

## GENUINELY-NEEDS-8× residual (design §7.5, updated by this authoring)

1. **NVSHMEM-TP transport validation** — the ONE go/no-go (design §5.4):
   install NVSHMEM, build with `-DSG_HAS_NVSHMEM=1`, swap
   `LoopbackTransport` → `NvshmemTransport` in `tp_loopback_binding.cu` (the
   declared swap point; zero math changes), then: parity vs loopback/host-NCCL
   TP, A/A/A across real ranks, MFU vs host-NCCL-split TP, coexistence with
   GridBarrier + NCCL + (optional) graph capture. NVSHMEM is NOT on the
   authoring box — nothing of this can be pre-run.
2. **TP insertion into the production megakernel body** — mechanical now (the
   4 reduce points are marked in `tp_layer.cuh` with file:line; the math +
   transport are gated here), but it edits `model_stage_decoder_tc.cuh` +
   `fused_decoder_megakernel.cuh` (Par threading) and is transport-choice-
   dependent (§5.4), so it is deliberately 8×-window work per the design's
   build order (TP last).
3. **Real multi-GPU scaling measurements** — DP weak-scaling 1→8 (≥70% bar),
   ZeRO-3 max-fit/OOM threshold (§3.3), PP bubble/1F1B throughput at real
   stage counts + microbatch sweeps (mb>1 grad-parity is tol-level, not
   bitwise — reassociation across microbatches; measure, don't assume).
4. **Cross-rank graph capture with collectives** — ef433ac established the
   megakernel+NCCL-collective mix does NOT capture on the 1-GPU loopback
   (SM contention); the per-piece captures are green. The mixed capture
   retry needs real per-GPU ranks.
5. **PP P2P transport** — the loopback handoff is zero-copy by construction;
   the real path is `dist.batch_isend_irecv` on `pp_group` of exactly the
   `HandoffSpec` payloads (bf16 X_in[L] fwd, fp32 dh bwd) — wire + validate
   on 2+ real GPUs (the `LoopbackP2P` → torch.distributed swap point in
   `pipeline.py`).
6. **vit/mamba PP/TP twins** — the decoder is the flagship target (design §4/§5
   are decoder-specified); porting the stage-range patch + shard tables to the
   vit/mamba stage headers follows the identical recipe once the decoder
   pattern is hardware-validated (1-GPU-authorable later, listed for honesty:
   not authored in this pass).
