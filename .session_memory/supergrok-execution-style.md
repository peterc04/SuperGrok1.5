---
name: supergrok-execution-style
description: "How the user wants me to work on SuperGrok2 — understand the goal, execute confidently, saturate GPUs with REAL 1.5B work not toys, stop over-asking"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6354dc07-b50f-40a0-8748-5189102539d3
---

Hard-won feedback from the SuperGrok2 session (user got repeatedly frustrated):

- **The REAL product is a portable, self-adapting, max-performance training stack** (PyTorch-shaped:
  high-level Python over a CUDA/C++ backend). Two core properties: (1) PORTABILITY — every component
  (3 models, 11 optimizers, the compile file, the profiling files) drops into ANYONE's project and fits,
  like nvcc/PyTorch (config/declaration-driven, never SuperGrok-hardcoded). (2) SELF-DESIGNING megakernels
  (meta-programming) — the kernel + autotuner co-generate the OPTIMAL kernel for whatever workload:
  10M params on 1 GPU -> specializes for that; 1.5B with full parallelism -> specializes for that, at max perf.
- **We VALIDATE it on** the 11-optimizer ranking benchmark (lowest val loss / most stable per fixed step
  budget) across **three ~1.5B models** (decoder/ViT/Mamba-3) on **real datasets**
  (FineWeb-Edu/ImageNet-1k/GiftEvalPretrain), with **4D parallelism = DP x TP x PP x SP (sequence
  parallelism is the 4th axis) + ZeRO-3**. SP is active at scale (long real-dataset sequences; was pinned
  1 only for the seq=4 toy). The `ParConfig<DP,TP,PP,SP,Z>` template (parallel_config.cuh) = those 4 axes + ZeRO stage.
- **4D+ZeRO-3 is HOW 8 GPUs get saturated**: ONE 1.5B model is distributed across all 8 (TP/PP/SP shard
  the model, ZeRO-3 shards opt state, DP replicates) — NOT 8 independent toy trainings. `.parallelism_design.md`
  (DP+ZeRO2 -> ZeRO3 -> +PP -> +TP, device-NVSHMEM-TP stretch / host-NCCL-TP fallback) is the contract; it is
  currently DESIGN-ONLY (ParConfig template + sharded_optimizer_kernel + ZeRO3Sharder + tp_transport scaffolded, not wired+validated on 8 GPUs).
- **Critical path = flagship kernel regen -> datasets -> real 1.5B training.** This is BOTH the
  deliverable AND what actually saturates 8xH100. Race-scale (mod-97, d=128) is too tiny to load an
  H100 — do NOT keep patching GPU-idle with toy-scale workloads; build the flagship and run real training.
- **Stop over-asking / stop flailing.** The user wants confident execution, not repeated AskUserQuestion
  or launch-and-check churn. Understand the goal, execute the critical path, surface only genuine forks.
- **GPU saturation is the standing requirement** (instance billed by the hour, CPU==GPU price): keep all
  8 GPUs on real work. My ad-hoc launchers kept dying (silent process deaths / concurrent .so-load races);
  the durable fix is real 1.5B training, not toy loops.
- Mamba-3 flagship hits the smem-per-SM cap above ~d=142 (needs activation-draining/chunking) — known blocker.

**Why:** I repeatedly mis-scaled (toy vs 1.5B), fumbled GPU orchestration, and over-asked. **How to
apply:** lead with the goal, drive the flagship+dataset critical path, saturate with real training.
See [[supergrok-working-prefs]], [[ncu-blocked-runpod]].
