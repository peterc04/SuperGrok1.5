# Crux deep-dive (first-hand, lead): the TP data-path fix — what it REALLY does

Read `phase6/tp_datapath_fix_WIP.patch` in full + traced it against `model_stage_decoder_tc.cuh` and
`mega_decoder_real_adamw_tc_launcher.cu`.

## What the 8-GPU run originally did (the buggy Megatron path)
On `Par::kTPComm`, the megakernel implemented a genuine Megatron tensor-parallel shard:
column-parallel QKV/ff0 (rank owns `3d/P` / `dff/P` out-cols), head-localized attention
(`Hloc=kHeads/P`, `Dloc=d/P`), row-parallel out_proj/ff2 (publish partial → device-NVSHMEM all-reduce),
mirrored in backward. It surfaced 3 bugs:
- **B (head divisibility):** `kHeads=25 % TP=8 ≠ 0` → `Hloc=3,Dloc=200` violates `Dloc==Hloc·kDhead=192`.
- **A (rank divergence):** host `_tp8_run.py` gives every rank the FULL replicated params + does NO host grad
  sync. Megatron makes each rank a distinct dW slice → replicated params DIVERGE without a full-weight grad
  all-reduce (which the tile-sized symmetric heap can't carry).
- **C (the +0x1dc30 IMA):** (i) shard-width writes go OOB when invariants break; (ii) the flagship workspace
  cudaMalloc is hundreds of GB → fails on 80 GB → null base → wild-pointer write.

## What the fix actually does  ⚠️ IMPORTANT NUANCE
The patch **abandons real sharding on the kTPComm path and computes FULL-WIDTH REPLICATED** — every rank runs
the identical full-width fwd/bwd (same math as SingleGPU, full kHeads=25 attention). To still exercise the
device-NVSHMEM path, the 4 reduce points publish `(full_result / P)` and do an ascending-pe fixed-order sum of
P identical copies → reconstructs `full_result`. That sum is a **mathematical identity** (Σ P·(x/P)=x).
- Fixes B (no head-split), A (identical full grad on every rank → params stay bit-consistent), C-OOB
  (only full-width buffers written).
- Launcher fix: workspace cudaMalloc made OOM-SAFE — on failure leave `workspace=nullptr/ws_floats=0` and
  return `cudaErrorMemoryAllocation` instead of launching through a null base (kills the +0x1dc30 IMA).

## My assessment (the part the headline glosses over)
1. **This is a VALIDATION SCAFFOLD, not real model parallelism.** Full-width replicated compute = every rank
   computes & stores the WHOLE model; the all-reduce of identical copies is pure overhead (an identity). So
   this path yields **zero compute reduction and zero per-rank memory reduction** from TP. It proves the
   NvshmemTransport NVLink all-reduce + megakernel TP plumbing work end-to-end on 8 GPUs with bit-consistent,
   descending loss — but it does NOT shard a 1.5B model across 8 GPUs.
2. The **OOM guard does not make the flagship fit** — it converts a wild-pointer IMA into a clean error. At the
   production layout the workspace is still hundreds of GB, so `mega_decoder_real_adamw_tc` will just RETURN
   `cudaErrorMemoryAllocation` unless nCTA is hard-capped / staged-opt carves removed / real sharding added
   (consistent with the single-GPU smoke needing ncta_cap=8 + AdamW-only + SG_DEC_BENCH_LAYOUT=1 to fit 80 GB).
3. **Genuine "one big model across 8, each holding 1/8" remains future work** — the patch comment is explicit:
   "a future genuine weight-shard needs the host to pre-pack per-rank shards AND a whole-weight grad all-reduce;
   that is scoped, not done here." So the north-star capability is plumbed + validated, not yet delivered.

## Resume de-risked
- The patch **applies cleanly on HEAD** (verified: HEAD blobs `e84ec16`/`973115b` == patch base; `git apply
  --check` and `--3way` both clean). It is UNGATED + uncommitted.
- It touches exactly 2 files; SingleGPU path is unchanged (the `Par::kTPComm` branches fold away when not TP),
  so the decoder 19/19 byte-identical gate should hold — that's the first gate to re-run after applying.
- Bug C's IMA-clear still needs the live `compute-sanitizer` 8-GPU run to confirm (the OOM guard is the
  hypothesized root, but the wild write must be witnessed gone).
