---
name: supergrok-queued-deliverables
description: Two user-requested SuperGrok2 deliverables queued for after the flagship build — a 33-cell flagship roofline graph and a line-by-line dead-code cleanup + LOC report
metadata: 
  node_type: memory
  type: project
  originSessionId: 6354dc07-b50f-40a0-8748-5189102539d3
---

User requested (2026-06-25), explicitly queued ("after you get all of this done" / "when soonest convenient"):

1. **Roofline ceiling test of all 33 cells (11 optimizers × 3 models) for the FLAGSHIP variants → a GRAPH.**
   Do this AFTER the flagship build is solid. ncu HW counters are DENIED ([[ncu-blocked-runpod]]), so build the
   roofline ncu-FREE: analytical arithmetic intensity (FLOP/byte per cell from the known GEMM shapes) on the
   x-axis vs ACHIEVED TF/s (CUDA-event wallclock + analytical FLOPs, the method already used: decoder d=2048
   measured 64 TF/s = 6.5% of the 989 TF/s bf16 ceiling) on the y, plotted against the H100 bf16 roofline.
   Needs each of the 33 flagship cells built + run + measured. Deliver the graph (matplotlib PNG) to the user.

2. **Comprehensive LINE-BY-LINE dead-code cleanup of the whole codebase, then report total LOC + per-language
   LOC and percentage.** Do this when the TREE IS STABLE — NOT while the parallel integration agents are
   editing (conflict risk). Provably-dead only (reachability-checked), each removal gated (the math-drift guard
   + the parity/determinism gates must stay green; the prebuilt artifacts + the 33 _tc cells stay). After
   cleanup, run a cloc-style count: total lines, and lines + % per language (CUDA/C++/Python/etc).

Both are post-build deliverables; keep driving the flagship 8-GPU build + benchmark first. See [[supergrok-autonomy]].
