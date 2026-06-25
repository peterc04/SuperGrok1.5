---
name: flagship-distributed-config
description: TP8·ZeRO-3 is the mesh that saturates 8 H100s with one flagship 1.5B model and fits all 11 optimizers
metadata: 
  node_type: memory
  type: project
  originSessionId: 6354dc07-b50f-40a0-8748-5189102539d3
---

For the flagship 1.5B benchmark (one model across all 8 GPUs constantly working — the north star),
the winning 4D mesh is **TP=8 · DP=1 · PP=1 · ZeRO-3** (computed from the live scratch formulas
2026-06-25, `/workspace/impl_diffs/run_harness.md`).

WHY: the staged-opt per-CTA scratch `dec_tc_sg2_floats = nCTA · ~91·Nmax` is **linear in Nmax**
(`kDecMaxTensorNumel` = the largest tensor = ff weight `dff·d` = 10.24M at d=1600). TP=t column/row-shards
that weight → per-rank `Nmax = 10.24M / t`. At **TP=8, Nmax=1.28M**, so the SG2 scratch shrinks
**509 GB → ~58 GB/rank**. Result: **10 of 11 optimizers run at 1-CTA/SM (nCTA=132, 66–68 GiB/rank);
SuperGrok2 auto-caps to nCTA=64 (40.9 GiB)** — the full 11-optimizer ranking benchmark fits at flagship
size, all 8 GPUs saturated (TP, not nCTA, is what spreads the model). Usable budget ~70.5 GiB/GPU.

Single-GPU dense does NOT fit the staged opts (509 GB SG2 scratch) — only AdamW with the staged-opt
scratch elided (`SG_DEC_BENCH_LAYOUT=1`) + ncta_cap. So the flagship GENUINELY REQUIRES 4D sharding.

Distributed state (verified): DP + host-ZeRO-3 are WIRED (`fused_train_step_distributed`); **TP/PP are
STUBBED** — the production launcher `launch_fused_decoder_megakernel_tc<OptId>` is NOT yet templated on
`ParConfig`/`CommCtx` (zero refs in fused_decoder_megakernel.cuh). Wiring TP needs: template the
kernel+launcher on `Par`, a symmetric-heap TP-comm-slot allocator, and the in-kernel
`tp_allreduce_sum_fixed_order` via device NVSHMEM ([[nvshmem-installed]]). Apply-ready specs:
`/workspace/impl_diffs/{dist_step,tp_nvshmem,run_harness}.md`.
