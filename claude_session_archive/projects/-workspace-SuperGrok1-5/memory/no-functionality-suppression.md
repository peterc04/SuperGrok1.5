---
name: no-functionality-suppression
description: "User directive — never fix by disabling; all functionality (SG meta/lamb/SAM/bilevel, everything in the codebase) must stay ACTIVE and be made to work"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 122b3aee-fa74-43c0-936d-39ff1eca854f
---

User (2026-06-09, H100 grokking-race session): "None of the SG functionality should be
suppressed. In fact, none of the functionality in this entire codebase should be suppressed."
Said in response to my "tamed" SG configs (memorization-gate ratchet driving meta→0
post-memorization, lamb=0, alpha=0 isolations proposed as shipping fixes).

**Why:** The SuperGrok optimizers are the point of the project — showing them working at
full capability (CSA/HCA+PEER+GRU meta, grokfast lamb, SAM, bilevel) is the deliverable.
A config that wins by zeroing components is a glorified AdamW and proves nothing. The user
accepts honest negative results ("as long as that is shown, that is fine") but not
engineering-by-disabling.

**How to apply:**
- Fix root causes so components are stable while ACTIVE (e.g., bounded/lookahead meta
  objectives instead of unbounded alignment; continuous grokfast schedules instead of
  discontinuous spikes), then TUNE magnitudes (>0 lower bounds in the tuner).
- Zeroed/stripped configs are allowed only as DIAGNOSIS (isolating a destabilizer), never
  as the shipped fix. Label them as such.
- Defaults must not suppress (remove gate-on-by-default once real fixes land).
- The megakernel ABI must carry FULL SG state (meta-net weights + all hyperparams) so the
  fused path runs the real algorithms, not placeholders.
- Exceptions must be loud — silently-swallowed component failures (try/except warn) are
  de-facto suppression (SG2 ran for weeks with SAM 100% dead and nobody knew).
