---
name: h100-durable-requirements
description: "Owner's standing requirements that must hold across ALL work — PTX-maximal kernels, AdamW-grade Python callability, line-by-line verification discipline"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 122b3aee-fa74-43c0-936d-39ff1eca854f
---

Owner (2026-06-10), after repeatedly having to re-specify things: every claim
must be verified line-by-line, and these requirements are STANDING (apply to
all current and future work without being re-asked):

1. **Maximal inline-PTX optimization** in every kernel (H100/sm_90 now;
   mi300x/tpu_v6e when those phases open). Verified at the SASS level
   (cuobjdump), not assumed from source.
2. **AdamW-grade Python callability for every optimizer**: `opt = X(params,
   lr=...)`, plain `loss.backward(); opt.step()` must work standalone AND
   exercise the full advertised machinery (no component silently idle in the
   naive pattern — no-suppression applies to API ergonomics too). External
   choreography (meta optimizers, sam/bilevel side-calls, special kwargs) may
   remain as an advanced path but must not be REQUIRED.
3. **Hardware max-parallelism** in every throughput phase ([[h100-mps-max-parallelism]]).
4. **Roofline fraction is the optimization metric** (not watts); tuning metric
   is gradient-steps-to-val-grok (test-confirmed); race tracks every 10 steps,
   tuner every step.
5. **No CUDA graphs** — true persistent megakernels ([[portable-megakernel-components]]).
6. **No functionality suppression** ([[no-functionality-suppression]]).

7. **Routine cleanups are a standing cadence** (owner, 2026-06-10): recurring
   dead-code/bug/optimization sweeps keeping the WHOLE codebase
   production/portfolio-ready — not a one-time pass. Run one after every major
   phase (integration landings, retrofits, race milestones).
9. **ALL kernel perf tuning belongs to compile.py** (owner, 2026-06-10): hand
   work writes CORRECTNESS only (PTX sequences, barriers, parity). Every
   perf-shaped constant anywhere in the kernel tree (tile shapes, pipeline
   depth, warpgroup counts, block sizes, samples-per-CTA, unrolls) must be a
   registered `SG_TUNED_*` autotuner dimension searched by compile.py's
   JIT-build-and-time flow (`_kernel_tuned.json` → baked flags). Hand
   perf-iteration after correctness gates is BANNED — register the dim and
   run the sweep instead. Audit existing hand constants when touching a file.
10. **No problem-specific hardcoding, codebase-wide** (owner, 2026-06-10):
   "I prefer not to have values hardcoded for all problems, but rather a
   system that works for every use case — that philosophy goes for everything
   in the codebase." Shapes, vocab sizes, seq lens, batch sizes, arch counts,
   thresholds must derive from config/model introspection/device queries —
   compile-time specialization is fine (templates/codegen) when the
   SPECIALIZER is general (e.g. layout headers generated from
   named_parameters(), not hand-typed totals). Flag hardcoded-value
   violations in every review/audit.
11. **Optuna tuned configs are hardware-agnostic** (owner, 2026-06-10): the
   pre-race optimizer-hyperparameter results (tuned_configs_{model}.json)
   transfer to mi300x and tpu_v6e — gradient-steps-to-grok is algorithmic,
   not hardware-bound. Do NOT re-tune per arch; keep precision policy
   consistent across archs so trajectories stay comparable.
12. **compile.py serves ALL archs, first-class** (owner, 2026-06-10): every
   NVIDIA, AMD, and TPU arch in the ARCH_TABLE (~25-30 canonical archs) must
   work maximally — not just the three race targets — AND the file stays
   portable to any other project (the custom-project self-tests are the
   guard). Per-arch search spaces/flag emission must not lag the table.
13. **Auto hardware detection + backend selection in the optimizer API**
   (owner, 2026-06-10, extends req #2): `opt = X(params, lr=...)` must
   detect the hardware (NVIDIA arch family / AMD / CPU; JAX context for TPU)
   and route to the proper compiled backend automatically — today's binary
   `_ops if cuda else _ops_cpu` selection is insufficient. No user-side
   backend flags required.
8. **mi300x phase protocol** (owner planning to grant access): gfx942 work on
   its own branch; per-arch dirs only (csrc/fused/gfx942, kernels/gfx942) +
   additive dispatch branches; canonical math headers FROZEN during bring-up
   (adapt gfx942 TO them, never the reverse); any shared-surface change must
   re-pass the full H100 gate battery; the race runs from a PINNED checkout so
   parallel work can never contaminate it.

**How to apply:** when finishing any work item, check it against this list
before reporting it done. Periodic full-codebase line-by-line audits (6-agent
fan-out pattern, 2026-06-10) are the enforcement mechanism — repeat after major
phases land.
14. **Production/race path = the fastest VALIDATED implementation per cell**
   (owner, 2026-06-11). Eager/L1/L3-scalar remain in the tree for development
   + gates only; the race ships whatever path measures fastest at race
   precision (currently L3-TC for decoder/vit; mamba per measurement — its
   scan-dominated cells may keep scalar). Selection by measurement/autotuner,
   never by default; if a faster path lands later, it becomes production.
