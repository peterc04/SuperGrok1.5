---
name: h100-mps-max-parallelism
description: Owner wants MAX hardware parallelism — run NVIDIA MPS + a large worker fleet; single runs leave the H100 ~60-70% idle
metadata: 
  node_type: memory
  type: project
  originSessionId: 122b3aee-fa74-43c0-936d-39ff1eca854f
---

Owner directive (2026-06-09): "max parallelism" includes the HARDWARE, not just
agents. On this H100 + tiny-grokking workload, a single training run uses only
~40% util; even 9 plain concurrent processes only reached 33% util / 175 W
(context time-slicing). With NVIDIA **MPS** + 14 tuner workers: **88% util /
375 W** (~2.7× utilization).

**How to apply:**
- Start daemon (root): `export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
  CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY
  $CUDA_MPS_LOG_DIRECTORY; nvidia-cuda-mps-control -d`
- CAUTION: `echo quit | nvidia-cuda-mps-control` SHUTS THE DAEMON DOWN — it is
  the shutdown command, not a status probe (made this mistake once).
- Clients must CREATE their CUDA context AFTER the daemon is up and with the
  pipe env var exported — restart worker fleets after starting MPS.
- Fleet shape that works: per-model launchers sharing one Optuna journal
  (decoder 6w + vit 4w + mamba 4w = 14 workers). Memory ~2.5 GB/worker.
- Clean-timing measurements (roofline, knee) still need the GPU quiet —
  pause fleets for those windows; gradient-step trajectories are
  contention-immune so tuning/analysis correctness is unaffected by sharing.
- MIG (up to 7 hardware slices) exists as the heavier alternative but needs a
  GPU reset and breaks full-GPU measurement windows — not used.
- Shell gotcha in this harness: `pkill -f tune_optimizers` matches the shell's
  own command line and kills it (exit 144) — kill by `pgrep -f 'tune_opt'` PID
  loop in a SEPARATE short command instead.
- Optuna JournalStorage gotcha: a `kill -9` during an append leaves the
  symlink lock (`journal.log.lock -> journal.log`) held forever → ALL workers
  spin at 0% GPU. Fix: `rm results/tuning/journal.log.lock` (no restart needed;
  blocked workers acquire on their next retry).
- MPS "Teardown in progress" wedge (3× on 2026-06-10): short-lived test
  clients exiting under fleet load wedge the server → all new clients get 807
  AND existing clients stall at 0%. Recovery: TERM fleet → kill -9 mps procs →
  relaunch (~3 min). PREVENTION (standing): run ALL test/validation processes
  with `CUDA_MPS_PIPE_DIRECTORY=/nonexistent` so they never become MPS
  clients — fleet keeps the daemon to itself.
