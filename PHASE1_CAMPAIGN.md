# SuperGrok1.5 — Phase-1 Completion Campaign & 8×H100 Hand-off

**Living document.** Master plan + locked decisions + status for finishing all single-H100
work and prepping Phase 2, so the (expensive) 8×H100 instance boots straight into the
remaining multi-GPU bring-up. Branch `claude/h100-audit-maximal`. Commits are local-only.
Cross-refs: `HANDOFF.md`, `BUILD_AND_VALIDATE.md`, `AUTOTUNE_LINKAGE.md`,
`/workspace/.campaign_plan.md`, `/workspace/.parallelism_design.md`.

Owner goal: **maximize work on the single H100 before the 8×H100 clock starts** (the 8×
instance is ~8× the billing), then hand off cleanly.

---

## 0. Project phases (only two)

- **Phase 1 — single-GPU foundation** (THIS campaign): the pure bf16 wgmma L3-TC persistent
  fused-megakernel for all 33 cells (decoder/vit/mamba × 11 optimizers), fp64-gate-validated,
  roofline-converged; the portable autotuner (compile.py); the trained models at canonical
  sizes; pre-race optimizer-hyperparameter tuning.
- **Phase 2 — 4D + ZeRO-3 multi-GPU** (8×H100): DP×TP×PP (+ SP axis expressible, pinned 1
  for the short seqs) + ZeRO stage 3, max batch, all 3 models. Design contract:
  `/workspace/.parallelism_design.md`. SG2 meta-model is DeepSeek-V4-derived (CSA/HCA), NOT Mamba-3.

There is **no formal "Phase 3"**; post-Phase-2 work (grok-science demos, datasets, the
separate ground-up AMD/RCCL + TPU/Pallas ports) is tracked in `/workspace/.campaign_plan.md`.

---

## 1. The 3 trained models — canonical published sizes (owner-locked 2026-06-16)

Use **recognized published configs**, not hand-tuned dimensions (peer-review credibility).
Each model at its OWN canonical size (different param counts is honest + normal).

| Model | Architecture (honest name) | Flagship config | Params |
|-------|----------------------------|-----------------|--------|
| decoder | GPT-2 XL | d=1600, L=48, h=25 | ~1.5 B |
| vit | ViT-G/14 | d=1664, L=48, h=16, MLP=8192 | ~1.8 B |
| mamba | **Mamba-3** (arXiv 2603.15569, ICLR 2026) | d=2048, L=**24**, state=128, head_dim=64 (SISO base) | **1.473 B** (paper's own 1.5B config) |

- **Grokking science RACE** stays at the toy config (modular arithmetic p=97, seq_len=8,
  d=128 → decoder ~0.42M / vit ~0.42M / mamba ~0.26M) — that's the *science*; the flagship
  sizes are a separate roofline/scaling config from the same portable code.
- Register the flagship tier in `grokking_race_v2.MODEL_SCALES` (today only small/medium/large).

---

## 2. Mamba-3 upgrade (trained model: Mamba-1 → Mamba-3)  — IN PROGRESS

The trained `mamba` model is being upgraded from Mamba-1 (`SelectiveSSMLayer`) to genuine
**Mamba-3** so the canonical `mamba3` name becomes accurate. SG2 meta-model UNCHANGED (DeepSeek-V4).

Mamba-3 (paper text cached at `/tmp/mamba3_paper.txt`): **exponential-trapezoidal
discretization** (subsumes + drops the conv1d via explicit B,C bias terms), **complex-valued
state** (→ state-tracking: parity/arithmetic — fits grokking), **SISO base** / MIMO optional.
Canonical 1.5B = d=2048, state ∈ {64,128}.

Phases (CHECKPOINT before the megakernel — expensive/irreversible):
1. ☑ **DONE + validated** — Reference model (`grokking_optimizers/mamba3_block.py`) + fp64
   oracle (`tests/hw/mamba3_oracle.py`) + writeup (`MAMBA3_REFERENCE.md`). SISO base, complex
   state as 2×2 real rotations (Eq 25 per-step form), conv1d dropped, exponential-trapezoidal.
   Oracle PASS: fp64 finite + all 35 params differentiable + fp32≈fp64 (1.25e-6) + FD≈autograd
   (4.8e-8). 1.473B at d=2048/L=24/state=128 (paper-faithful). 2 ambiguities open for review:
   (a) SiLU on SSM input — paper says obviate conv "and its accompanying activation" → lean DROP;
   (b) rotation dt — head-shared mean vs per-head dt (Mamba-2/3 use per-head dt) → settle in the
   multi-head megakernel layout.
2. ☐ L3-TC megakernel (`model_stage_mamba3.cuh`, `mamba3_layout.cuh`, launchers) — bf16
   transcription matching the oracle; hand-derive the complex-trapezoidal backward.
3. ☐ Re-gate all 11 mamba cells (fp64 parity gate, 3 seeds).
4. ☐ HIP (gfx942) + Pallas (TPU) mamba paths + register the canonical tier.

---

## 3. Optimization LOOP (compile.py first, then kernels)  — IN PROGRESS

The optimization process is **LOOPED**, not one-shot (owner 2026-06-16). Per track:

```
repeat:
  1. DISCOVER: read-only agent swarm (opt-discovery workflow) → exhaustive neutral/positive
     candidate list, EXCLUDING everything already tried this campaign.
  2. RATCHET each candidate serially (feedback-patch-protocol): apply → verify → KEEP/REVERT.
until  (a) a discovery round finds NO new viable candidate  [dry well], OR
       (b) 3 candidates IN A ROW come back NOT-positive (reverted/skipped) — counter is
           CONSECUTIVE across rounds.
```

**Order (owner): do the compile.py track FIRST — loop it to termination — then the kernel track.**

- **compile.py track** — gated by the in-file `--self-test` (correctness) + build/tune time.
  Candidates are bit-neutral host-side speed opts; "positive" = self-test stays green AND the
  change is a real speed/quality improvement (not a no-op). Run self-test CPU-only
  (`CUDA_VISIBLE_DEVICES=""`).
- **kernel track** — gated by the fp64 parity hard-gate + 3-seed step timing **at d=2048**
  (toy d=128 is physics-inert). Runs AFTER Mamba-3 lands (so the final kernels are tuned).

**STOP criteria (owner):** (a) genuinely no more neutral/positive candidates for that part →
stop that part; (b) 3 not-positive in a row → stop the process.

Prior P-series result: KEPT decoder SAM-outline (0002), mamba scope-noinline (0004),
decoder cp.async ring (0005, −14.2%); REVERTED vit SAM-outline (0003, +5%).

Verdict ledger (this campaign): _appended as the loop runs. Round-1 compile.py candidates:
`.opt_candidates.json` (10 compile + 9 kernel + 69 dropped-as-dead-code/already-done)._

**compile.py track — round 1** (gate: in-file `--self-test`, CPU-only; baseline
**236 passed / 6 failed**, the 6 are pre-existing #10-aftermath drift-guards on deleted
files; final tally held at **236 passed / 6 failed**, identical failing set, no new fails):

| # | candidate id | verdict | reason |
|---|---|---|---|
| 1 | json-indent-removal | **KEEP** | `indent=2`→`separators=(",",":")` in `_save_locked`, `sort_keys=True` kept; cache read via `json.loads` (whitespace-agnostic) ⇒ roundtrip + determinism identical. |
| 2 | variant-build-sig-hash-redundancy | **KEEP** | Hoisted `_hash_sources(sources)`→once-per-sweep `_base_sources_hash`; common path reuses it, only poly/synth (`_sources_replaced`) recomputes. build_sig byte-identical. |
| 3 | version-gated-flags-cache | **SKIP** | Already satisfied: `_VERSION_GATED_FLAGS_CACHE` (l.12018) + memoized `_version_gated_flags_for_hash` (l.12021-31) already cache the per-arch nvcc probe/flags. |
| 4 | prefilter-early-exit-cartesian | **SKIP** | Not actionable: in-place reorder changes `hash_space` (serializes rules list in order)→invalidates AOT keys (not neutral); in-closure reorder needs a speculative AST-cost heuristic; embedded rules already cheap-first. |
| 5 | measured-ms-window-algo | **KEEP** | `measured_ms` list+`del`-slice → `collections.deque(maxlen=2000)` (init+append+fallback). Last-2000 append-order identical, no consumer slices it ⇒ quantile bit-identical. |
| 6 | multi-fidelity-finite-sort-cache | **KEEP** (reduced) | Sort was already gated behind `<8`; instead hoisted cheap `isfinite(ms_pred)` + raw-`len(measured_ms)<8` guards above the O(n) finite-list build. Every guard returns identical `(False,None)` ⇒ decision bit-identical. |
| 7 | featurize-config-dim-index | **SKIP** | Premise false: `_values_of` called exactly 3× (vec/unroll/num_stages), each distinct, each once — no repeated lookup to hoist. |
| 8 | seed-trials-validation-early-exit | **SKIP** | Premise false: `dim_names`/`dim_value_sets` (l.6338-41) already built **before** the `for t in seed_trials` loop (l.6342), not per-iteration. |
| 9 | early-stop-window-slice-cache | **SKIP** | Premise false for the real driver: `should_stop()` runs exactly once per `observe()` (1:1, l.6400/6461)→window/patience change every poll→slice-cache hits 0% (net pessimization). |
| 10 | cost-model-quantile-stability | **SKIP** | Not bit-neutral: `np.percentile(linear)`'s `(1-g)·lo+g·hi` ≠ manual `lo+(hi-lo)·g` (1-ULP)→can flip the prune-boundary compare→trajectory drift vs baseline. Also likely perf-negative + contradicts the deliberate numpy-free hot path. |

**Round-1 result: 4 KEEP (1,2,5,6), 6 SKIP (3,4,7,8,9,10), 0 REVERT.** No applied edit
failed the gate; STOP-on-3-consecutive-reverts never triggered (skips of false-premise /
already-done survivors are not reverts). All four kept edits are bit-neutral host-side
autotuner hoists — no kernel-codegen, cache-key, or search-trajectory change.

---

## 4. Autotuner (compile.py) — status

Re-anchored + Wave-1-merged + validated (236/6 self-test = 6 pre-existing #10-aftermath
drift-guards only; 33/33 fp64 gate; production build green). See
`project-campaign-state` memory + `AUTOTUNE_LINKAGE.md`.
- **Build-cost reduction (quality-neutral/positive, owner-requested):** incremental variant
  build (`--incremental-variant-build`, done); single-cell build (rebuild only the tuned
  cell's TU, reuse other models' AOT objects — they link for dlopen, never timed); ccache-for-nvcc.
- **JIT search:** NO fixed trial count — `--bayesian-trials` omitted ⇒ multi-criterion
  early-stop (plateau + coverage saturation + wall-clock).
- **#11 validation:** tuned-vs-default + vs-regular-nvcc, meaningful only at d=2048 scale
  (toy d=128 is physics-inert — autotuner correctly found no win there).

---

## 5. Pre-race optimizer hyperparameter tuning (for the RACE)  — QUEUED (run at end)

`tuning/tune_optimizers.py` — Optuna, all 11 optimizers, tuning seed 1001 (disjoint from race
seeds), output → `results/tuning/tuned_configs.json`. The grokking race later just loads that
file (zero race-time impact). **Run this near the end** and commit the resulting
`tuned_configs.json` so the race uses owner-blessed hyperparameters.

Status: _TBD — tuned_configs.json not yet generated this campaign._

---

## 6. Phase-1 close-out checklist

- ☐ Mamba-3 trained model live + 11 mamba cells re-gated (3 seeds)
- ☐ Optimization ratchet complete (both tracks; stop criteria hit)
- ☐ All 33 cells parity-clean (fp64 gate) on seeds {42,7,123}
- ☐ Roofline-converged at d=2048 (baseline-before-mods → hill-climb → re-measure)
- ☐ Autotuner validated at scale (#11 tuned-vs-default-vs-nvcc)
- ☐ Pre-race `tuned_configs.json` generated + committed
- ☐ Pre-existing #10-aftermath self-test drift-guards fixed (OPTS/BINDING_FUNCS/manifest)
- ☐ Everything persisted to /workspace (§7) + hand-off doc complete (§8)

---

## 7. Persist for fast recompile / immediate use (/workspace)

Everything needed to recompile fast or run immediately on the 8×H100, kept on the persistent
volume `/workspace`:
- Built `_ops` extension + the persistent build caches (`/dev/shm/ccache` → also mirror to
  `/workspace`, `/workspace/.sccache`, the autotuner CompileCache + `tuned_configs`).
- `results/tuning/tuned_configs.json` (race hyperparameters).
- The tuned kernel configs (`grokking_optimizers/_kernel_tuned*.json`) if a scaled tune lands.
- Env to reproduce: `cd /workspace/SuperGrok1.5 && source /workspace/venv/bin/activate &&
  export PATH=/workspace/.local/bin:$PATH && source .regpressure/env.sh && export PYTHONPATH="$PWD"`.
- Build: `./build.sh` (compile.py/setup.py only; never raw nvcc). ~5.5 min full build.

---

## 8. 8×H100 / Phase-2 hand-off

When the 8×H100 is provisioned, the single-GPU foundation is done and the instance should
boot straight into the multi-GPU bring-up. Authoritative contract:
`/workspace/.parallelism_design.md` (4D+ZeRO-3, owner-locked).

**Done on the single H100 (Phase-2 prep):** _TBD — list the CPU/1-GPU-authorable pieces built
+ unit-tested here (parallel_config.cuh / tp_layer.cuh / sharded_optimizer_kernel.cuh, ZeRO
shard math, DP=2 loopback determinism)._

**Left for the 8×H100 window (minimize time here):** the NVSHMEM device-initiated TP (or
host-NCCL fallback), the graph-captured distributed step, and the 1→8 scaling measurements.
Bring-up order: DP+ZeRO-2 → ZeRO-3 → +PP → +TP (validation gates between).

**To resume:** read this file → §6 checklist → `/workspace/.parallelism_design.md` → run the
distributed tests under `torchrun --nproc_per_node=8` (they skip on WORLD_SIZE≤1 today).
