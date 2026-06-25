---
name: supergrok-adaptive-parallelism
description: SuperGrok2 parallelism must auto-select 3D–5D from front-end params; 5th axis = expert parallelism; kernels self-specialize by workload size
metadata: 
  node_type: memory
  type: project
  originSessionId: 6354dc07-b50f-40a0-8748-5189102539d3
---

User directive (2026-06-25), TWO parts:

**1. Adaptive 3D–5D parallelism, degree INFERRED from front-end params.**
The parallelism degree is NOT fixed at 4D — it auto-selects from the model definition:
- base **3D** = DP × TP × PP (always).
- **+SP** (4th, sequence parallelism) IF the model is a sequence model.
- **+EP** (5th, **EXPERT PARALLELISM**) IF the model uses experts (MoE).
- "and so on" — so a non-sequence non-MoE model is 3D; a sequence model is 4D; a sequence MoE is 5D.
The inference is from the FRONT-END parameters (the model config the user passes), not hardcoded.
Current state: ParConfig<DP,TP,PP,SP,Z> exists (parallel_config.cuh) — **EP is a NEW 5th axis to add**, plus
a front-end → ParConfig inference function. The 3 current flagships (decoder/ViT/Mamba) are sequence,
non-MoE → 4D; EP only engages when a model declares experts. EP is ADDITIVE on top of the TP work in
[[flagship-distributed-config]] — the TP integration is not wasted.

**2. Kernels self-specialize by SIZE/config, not just for the flagship.**
The megakernel codegen/autotuner must pick knobs (notably **CTA-tiling**) by workload size: CTA-tiling
HELPS at large sizes (more SMs to fill, occupancy via clusters) but HURTS at smaller sizes (overhead) —
and similarly for other configurations. So the "self-designing megakernel" property generalizes: the kernel
co-generates the OPTIMAL config for ANY project's size (10M → 1.5B → bigger), not a flagship-pinned build.
This is the meta-programming/autotuner self-design property the README/refined-goal describes, made concrete:
size-thresholded knob selection in megakernel_codegen.py. The CTA-tiling lever ties to bottleneck LEVER ②
(the 20% grid-barrier idle measured at d=2048 — CTA-tiling/work-balance is its fix at large sizes).

**UNIFYING PRINCIPLE (user clarification 2026-06-25):** size/config specialization is the SAME mechanism
as distributed-vs-single. The megakernel is TEMPLATED on its deployment config and `if constexpr` folds in
exactly the machinery that config needs: distributed build → emits the all-reduce / TP / parallelism
machinery; single-GPU → none of it (byte-identical). Likewise large size → CTA-tiling ON; small → OFF.
This is ALREADY the pattern the TP work uses: `if constexpr (Par::kTPComm)` builds the in-kernel
device-NVSHMEM all-reduce only when distributed, and the SingleGPU instantiation is byte-for-byte the
legacy kernel. So EP (5th axis), SP (4th), and CTA-tiling are all just more `if constexpr`-gated config
branches on the SAME templated kernel — co-generated per workload, never hardcoded.

**ROBUST RESOURCE-FIT, NOT GPU-COUNT (user refinement 2026-06-25):** the strategy decisions must NOT key
on "1 vs N GPUs." A SINGLE GPU can host a 10B+ model, even for TRAINING — which then needs heavy machinery
(optimizer/activation OFFLOAD to host, RECOMPUTE / gradient checkpointing, LAYER STREAMING, possibly
CTA-tiling for the big GEMMs) all on one GPU. So the stack is a ROBUST PLANNER: given (model size/shape +
hardware capacity: #GPUs, HBM/GPU, host RAM, interconnect), it decides the FULL execution config:
  - parallelism degree (3D-5D),
  - memory strategy (in-HBM | ZeRO-offload | activation-recompute | layer-streaming | host-offload),
  - kernel knobs (CTA-tiling, ring depth, occupancy).
Examples the planner must handle: 10M-on-1-GPU → trivial (1-CTA/SM, all HBM, no offload); 10B-on-1-GPU →
offload + recompute + streaming + maybe CTA-tiling; 1.5B-on-8-GPU → 4D + ZeRO-3. The driver is the
MEMORY-FIT + COMPUTE-SHAPE analysis vs the hardware, never a naive GPU-count switch. Same if-constexpr /
config-templating mechanism emits exactly the chosen machinery (byte-identical when a strategy is OFF).

Apply: extend ParConfig + add EP + the front-end inference; add size-adaptive (CTA-tiling etc.) knob
selection to the codegen; add the ROBUST RESOURCE-FIT PLANNER (workload x hardware -> parallelism + memory
strategy + kernel knobs) — all via the config-templating + if-constexpr mechanism already in use for TP.
See [[flagship-distributed-config]], [[supergrok-autonomy]].
