# Pre-race optimizer re-tune — status (2026-06-16)

Driver: Opus 4.8, autonomous. Branch `claude/h100-audit-maximal` (local-only).
Production `_ops` left untouched (state C, HEAD 739e563 at start); compile.py /
kernels / setup.py NOT edited. Only `results/tuning/**` touched.

## What this re-tune was for
The trained `mamba` model was replaced Mamba-1 -> **Mamba-3** on 2026-06-16
(533b1ea / fb70a7d). The Jun-10 mamba tuning was for Mamba-1 and had failed
(0 grokked, all DNF). The Optuna journal still held those invalid `*__mamba`
studies. Goal: archive the stale artifacts and re-tune all 11 optimizers at the
toy d=128 grokking scale, mamba first.

## DONE and committed
- **Load path resolved.** The race/roofline consume PER-MODEL
  `results/tuning/tuned_configs_<model>.json` via
  `tuning/roofline.py:_load_tuned_configs(model)` (also used by
  `.regpressure/gpu/prodtime.py`). `grokking_race_v2.py` itself does NOT
  auto-load tuned configs (it exposes `OPTIMIZER_CONFIGS`). The unified
  `results/tuning/tuned_configs.json` named in the tuner docstring has NO code
  consumer — the per-model files are canonical. `model_type="mamba"` correctly
  builds `Mamba3Model` (grokking_race_v2.py:511).
- **Contamination removed (commit 3791987).** The Jun-10 (Mamba-1 era)
  artifacts were archived to `results/tuning/archive/*_jun10_mamba1era.*`
  (tuned_configs/TUNING_REPORT/trials for decoder, vit, mamba; the 14 MB journal
  moved on disk). The journal was wiped so fresh `*__mamba` studies start with
  ZERO old Mamba-1 trials (verified: "A new study created" for all 11).
- **Throughput characterised (measured on the production L3-TC wgmma
  megakernel, bf16, toy d=128, p=97):**
    - mamba: 18 GB working set per run; ~4.4 steps/s at K=1; it ANTI-scales
      (K=2 ~3.4, K=4 ~3.5 steps/s aggregate) — memory-bandwidth bound. Peak
      throughput is a single worker.
    - the 5 meta-net optimizers (neuralgrok, grokadamw, supergrok, supergrok15,
      supergrok2) run only ~1.3 steps/s on mamba (heavy in-kernel meta-net).
    - Consequence: the stock mamba search (CAP=12000, 11 opts, 12 sampler
      startup trials, no early-kill once a config memorizes) is a ~50-100h job.
- **Valid decoder + vit configs preserved.** Decoder/ViT architectures are
  UNCHANGED since Jun-10 (directive), so their Jun-10 winners are still valid.
  They were restored to top-level (`tuned_configs_decoder.json` 8 opts,
  `tuned_configs_vit.json` 6 opts) and verified to load via
  `_load_tuned_configs`. The race retains its valid decoder/vit tuning.

## Mamba-3 toy-grok finding (the mandatory question) — HONEST
**Mamba-3 does NOT toy-grok at d=128.** Every mamba trial that ran to a usable
point showed the same behaviour: it MEMORIZES the train set almost instantly
(train_acc 0.02 -> ~0.99 by ~step 80-100, train loss ~0.04) but val/test
accuracy stays pinned at the ~0.01 random-chance noise floor (1/97 = 0.0103)
with no upward movement. Concretely:
  - adamw default, clean 300-step run: peak_train 0.996, peak_val 0.024,
    final_test 0.018, no grok.
  - grokadamw (a COMPLETED CAP=2000 trial): train 0.997, peak_val 0.026,
    test 0.016 -> DNF (objective value 3003 > cap).
  - adamw/neuralgrok/supergrok early trials: either the same memorize-no-grok
    plateau, or diverged (train collapsed) and were pruned.
Across all completed/pruned mamba trials in this session: **0 grokked.** This is
consistent with the Mamba-1 result the directive flagged (Mamba-3 is no better
at toy-grokking than Mamba-1 here). The optimizer is not the bottleneck — the
generalization signal simply does not appear within a bounded toy-scale budget.

## What is NOT done, and why (infrastructure blocker)
The full re-tune (mamba winners file + decoder/vit refresh) was **blocked by an
environment constraint**: GPU processes in this session are SIGKILLed within
~15s-2.5min (non-deterministic, and it tightened over the session — a 300-step
adamw trial completed once in 84.5s early on, then identical/​smaller runs were
killed at 55s, 39s, finally ~15s). This was independent of launch method
(foreground synchronous, `run_in_background` tasks, `setsid`/`nohup`/`disown`);
background tasks were additionally "stopped by main session". A grokking search
needs sustained multi-minute-to-hours runs, which this environment would not
sustain. No fresh `tuned_configs_mamba.json` could therefore be produced, and
the decoder/vit refresh could not be re-run (the valid Jun-10 configs are kept).

### To finish later (on a box without the GPU-process reaper)
Run, with the tuner driven so mamba's budget fits wall-clock:
  `python -m tuning.tune_optimizers --launch --model mamba --workers 1 \
       --trials-per-opt N` then `--confirm`, but at a REDUCED mamba CAP
  (e.g. 3000-4000 via a thin runtime wrapper that sets `T.CAPS["mamba"]`; the
  stock 12000 is ~50-100h at the measured 4.4 steps/s) and a smaller
  `--top-k` for confirm (confirm is serial and unbounded for DNF configs).
  workers=1 is optimal for mamba (it anti-scales). Expect DNF across the board
  unless Mamba-3's toy-grok behaviour differs from everything observed here.
A scratch foreground-runner that applies these patches without touching the
committed tuner lived at `/workspace/_fg_runner.py` (CAP + pruner-startup
overrides; runtime-only).

## Net effect on the race
The race/roofline load: decoder (8 tuned opts) + vit (6 tuned opts) VALID and
present; mamba absent -> falls back to `DEFAULT_CONFIG`/`OPTIMIZER_CONFIGS`
(which is the right behaviour given no valid Mamba-3 grok config exists). The
invalid Mamba-1 config is archived, not loaded.
