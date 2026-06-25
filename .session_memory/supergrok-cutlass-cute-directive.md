---
name: supergrok-cutlass-cute-directive
description: User directive — replace the hand-rolled in-kernel wgmma with CUTLASS/CuTe device-side atoms for the SuperGrok2 megakernel GEMMs (the
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6354dc07-b50f-40a0-8748-5189102539d3
---

User directive (SuperGrok2): **use CUTLASS/CuTe** for the megakernel GEMMs instead of the hand-rolled
ss-wgmma engine (`csrc/backends/cuda/sm_90/wgmma.cuh`). This is the #1 perf lever — the hand-rolled
wgmma runs well below cuBLAS/CUTLASS-class (decoder GEMM phases ~462ms vs ~40ms ideal; 6.48% roofline).

**Critical nuance (do it right):** you CANNOT drop in CUTLASS's host-launched *collective* GEMM
(`CollectiveMma`) inside the persistent megakernel — it owns its own grid/launch, incompatible with the
1-CTA/SM persistent kernel that runs fwd->bwd->opt between hand-built grid barriers (this is exactly why
the codebase hand-rolled wgmma). The right move = use **CuTe DEVICE-SIDE ATOMS** composed inside the
megakernel's GEMM phases: the wgmma MMA atom (SM90_64xNx16_F32BF16BF16_SS), SM90_TMA_LOAD copy atoms
(real TMA, replacing the current cp.async ring), CuTe swizzle layouts, and cutlass::pipeline for
multi-stage + warp-specialization — all device-callable between the grid barriers, preserving the
one-launch fusion. This is a substantial rewrite of the GEMM substrate, not a drop-in.

It is ARCH-level work (the user deferred "specific archs"), so it's the LEAD task for the arch phase
after the general "everything else" (compile additions, bug-fixes, dead-code, datasets Layer A,
4D+ZeRO-3 generality + in-kernel NVSHMEM all-reduce, verification, profiling). See [[supergrok-execution-style]].
