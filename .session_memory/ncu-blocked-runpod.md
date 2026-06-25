---
name: ncu-blocked-runpod
description: ncu/CUPTI hardware perf counters are blocked in this RunPod 8xH100 container and cannot be enabled from inside; needs a pod relaunch with CAP_SYS_ADMIN
metadata: 
  node_type: memory
  type: reference
  originSessionId: 6354dc07-b50f-40a0-8748-5189102539d3
---

On the RunPod 8×H100 pod used for SuperGrok2 work: `ncu` returns `ERR_NVGPUCTRPERM`. **Unfixable from
inside the container** — `CAP_SYS_ADMIN`/`CAP_PERFMON`/`CAP_SYS_PTRACE` are NOT in the container's
capability bounding set, and `/sys/module/nvidia/parameters/NVreg_RestrictProfilingToAdminUsers` is not
exposed. nsys CUPTI *tracing* DOES work (no counters).

**To enable (user action, host/launch level):** relaunch the pod with `--cap-add=SYS_ADMIN` (ideally
also SYS_PTRACE, PERFMON) or privileged — on RunPod usually a custom template / support request; OR if
they control the host, set `NVreg_RestrictProfilingToAdminUsers=0` in /etc/modprobe.d + reload the
nvidia module. Verify: `ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed python -c "..."`.

**Counter-free fallbacks** when blocked: nsys timeline (one-launch-per-step fusion proof), cuobjdump
-sass / nvdisasm / ptxas -v (HGMMA/TMA/regs/smem/spills, static), CUDA-event wall-clock → throughput +
analytical-FLOP roofline, cudaOccupancyMaxActiveBlocksPerMultiprocessor. Blocks only: measured
occupancy, warp-stall breakdown, L2 hit-rate, issue-slot/WGMMA-pipe util, ncu achieved-FLOP/s.
See [[supergrok-working-prefs]].
