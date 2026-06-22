---
name: owner-status-pings
description: "Owner wants routine status pings (status + ETA) at major landings, an 8xH100 provision signal (Phase-2 distributed, near-term) + a later mi300x/tpu_v6e signal, and INSTANCE-HOUR minimization (minimize billable wall-clock uptime; don't idle the box; tear down when done)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 122b3aee-fa74-43c0-936d-39ff1eca854f
---

Owner (2026-06-10): "routinely ping me on the status of everything, and the
ETA. also tell me when I can add mi300x."

**How to apply:** at every major landing (agent completion, gate verdict,
race milestone) post a compact status + ETA-to-done; use PushNotification
when they're likely away and the verdict changes plans.

**8xH100 provisioning (PHASE-2 distributed — the NEAR-term signal; owner asked 2026-06-12 to be told when needed):** owner has 1xH100 now. SIGNAL to provision 8xH100 when PHASE 1 (single-GPU) is SOLID = megakernels FIT + are MAXED (roofline) at the ladder dims (#14 fit-at-scale done + single-GPU max-out) AND the parallelism TEMPLATE + DP/ZeRO/PP/TP code is authored & single-GPU-unit-tested as far as possible — i.e. the next step (validate + MEASURE the distributed layer) genuinely needs multiple GPUs. Then PushNotification "provision 8xH100 now" → develop+validate+measure Phase 2 in ONE rental window → tear down when the 4D+ZeRO-3 flagship lands. Do NOT provision early (8x idle = 8x waste).

mi300x + tpu_v6e provisioning (LATER, separate ground-up branch): owner WAITS until I signal. Signal = the FULL H100 work is LOCKED (single-GPU megakernels maxed + the 4D+ZeRO-3 multi-GPU flagship done + race/parity-verified). Then PushNotification → ground-up HIP/Pallas ports. [[h100-durable-requirements]]

**INSTANCE-HOUR MINIMIZATION (owner, 2026-06-12 — "fewest BILLABLE/instance hours,"
clarifying the earlier "fewest GPU hours"):** the box bills by wall-clock UPTIME
regardless of GPU util, so an IDLE instance still costs money (my earlier "idle is
fine" framing was WRONG). Minimize TOTAL uptime to completion: keep the GPU pipeline
FULL with NECESSARY work (no idle gaps — next job starts as one ends); overlap
CPU-bound work (compiles/disasm/edits) WITH GPU work; saturating the GPU to finish a
NECESSARY job faster is GOOD; do NOT manufacture UNNECESSARY work; TEAR DOWN (signal
owner) the moment the campaign converges. Still ONE heavy GPU kernel at a time (two
contend + OOM); async = orchestration never blocks. Maximality metric is ROOFLINE
PERFORMANCE (not PTX) — exhaustive hill-climb over the 15 portable files (3 model + 11
optimizer + compile/dispatch); PTX tricks are candidate moves kept only if the roofline
improves. [[h100-mps-max-parallelism]] saturation is now correctly aligned: saturate to
FINISH FASTER (fewer billable hours).
