---
name: supergrok-frontend-api
description: "SuperGrok2 front-end is PyTorch-shaped — call one of the 3 models at any size + plug YOUR OWN dataset + pick an optimizer, and the backend self-specializes/compiles; datasets are NOT confined to the 3 provided"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6354dc07-b50f-40a0-8748-5189102539d3
---

User directive/clarification (2026-06-25): the front-end must be PyTorch-shaped and usable like a library.

WHAT IS FIXED (the library surface): the **3 model ARCHITECTURES** (decoder / ViT / Mamba-3) and the
**11 optimizers**. The megakernels are hand-built/codegen'd for those 3 architectures, so the library is
confined to them — BUT parameterized by SIZE (d, layers, vocab, seq), so you can call ANY size of those 3
(10M → 1.5B → 10B+), not just the flagship.

WHAT IS NOT FIXED — DATASETS: the user explicitly does NOT want the stack confined to the 3 provided
datasets (FineWeb-Edu / ImageNet-1k / GiftEvalPretrain). Those are PROVIDED implementations of a PLUGGABLE
dataset interface — a user must be able to **connect their own dataset** (a streaming train iterator + a
fixed eval probe, per the datasets.md Layer-A design dispatching on data_source). Ensure the dataset
interface is a generic PROTOCOL (bring-your-own), not a 3-way hardcode.

THE FLOW (PyTorch-shaped): instantiate a model (1 of 3, any size) + pick an optimizer (1 of 11) + pass a
dataset → the backend SELF-SPECIALIZES: codegen emits the layout for that size, the [[supergrok-adaptive-parallelism]]
resource planner decides parallelism + memory strategy + CTA-tiling from (model x hardware), and it COMPILES
(cached via the compile file). The user noted "obviously it would all have to be compiled together" — correct:
the megakernel is size-pinned at compile time, so a config change triggers a (cached) recompile, not a runtime
reshape.

CAVEAT to honor in the dataset interface: the user dataset must map to the model's INPUT CONTRACT (tokens/
targets at the model's vocab/seq for decoder; patches for ViT; series for Mamba). The Layer-A interface should
expose that contract + an adapter, so "bring your own data" = data shaped to the chosen model's inputs (or a
config change + recompile if the shape differs). Kernels are currently size-pinned (mod-97 transition note).
See [[flagship-distributed-config]], [[supergrok-adaptive-parallelism]].
