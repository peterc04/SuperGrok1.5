"""Offline pre-race hyperparameter tuner — Optuna, all 11 optimizers in parallel.

Owner-specified design (2026-06-09):
  • Runs BEFORE the race; zero impact on race FLOPs/wall (race later just loads
    results/tuning/tuned_configs.json).
  • Bayesian (TPE) + early-stopping pruner, REAL-size p=97 task (no tiny
    proxies — grokking phase transitions don't transfer across scale); the
    cheap axis is the step budget, not model/data size.
  • SINGLE standardized objective for every (optimizer, model) study:
    **fewest gradient steps to val_acc ≥ 0.95, TEST-CONFIRMED** (wall-clock
    explicitly does not matter — this tunes the high-level algorithm). A val-grok
    counts only if it transferred to the held-out TEST split (≥ 0.90); val-trained
    meta-nets can otherwise fake-grok val.
      grokked:    value = grokking_step  (∈ [1, CAP])
      fake-grok:  value = CAP + (1 − final_test_acc)·CAP   (val-grok, test < 0.90)
      DNF:        value = CAP + (1 − max(peak_val, 0.5·peak_train))·CAP
                  (gives TPE a gradient through the memorize-but-no-grok region)
      crashed:    config-fault only → value = 2·CAP + (1 − last peak_val)·100
                  (gradient-carrying); INFRA crashes (OOM / CUDA / ABI / pybind
                  TypeError) are PRUNED instead, so they never poison the region.
  • Max parallelism: N worker processes share the one H100 (a single run uses
    ~40% util / ~2.4 GB, measured) and round-robin across ALL studies so every
    optimizer advances simultaneously. Each worker startup-sweeps stale RUNNING
    trials (dead-worker orphans > 2h) back to FAIL so the budget can't wedge.
  • Robustness: trials run tuning seed 1001 (disjoint from race seeds); the
    top-K (default 5) configs per study are CONFIRMED on seeds 1002/1003/1004/1005
    and the winner is the config grokking (test-confirmed) ALL confirm seeds,
    ranked by MEDIAN steps (fallback: most seeds grokked, then median).
  • No-suppression bounds (owner directive): every component-strength dimension
    has a lower bound > 0 — the tuner may weaken a mechanism but can never
    switch one off. Architecture/compiled-shape knobs (meta dims, expert
    counts, hidden sizes) are FIXED at spec: they are the algorithm's identity.

Usage:
  python -m tuning.tune_optimizers --measure 1 2 4 6 8        # concurrency knee
  python -m tuning.tune_optimizers --launch --workers 6 --model decoder \
         --trials-per-opt 40                                   # run the studies
  python -m tuning.tune_optimizers --worker --model decoder ...# (internal)
  python -m tuning.tune_optimizers --confirm --model decoder   # seeds 1002/1003
  python -m tuning.tune_optimizers --report --model decoder    # files + report
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUNE_DIR = ROOT / "results" / "tuning"
LOG_DIR = TUNE_DIR / "logs"

# Step caps per model (the "fidelity" axis — real task, bounded budget).
# Decoder groks ~1k-4k for working optimizers; mamba is a slow grokker
# (AdamW itself needs >6k), so it gets the largest budget.
CAPS = {"decoder": 5000, "vit": 6000, "mamba": 12000}
EVAL_EVERY = 1            # owner: tuner tracks EVERY gradient step (exact
                          # grokking_step resolution; the race itself uses 10)
TUNING_SEED = 1001        # trial seed — disjoint from race seeds
# [A6] 4 confirm seeds (was 2): tighter robustness gate — the winner must grok
# ALL of them. Disjoint from the trial seed and the race seeds.
CONFIRM_SEEDS = (1002, 1003, 1004, 1005)
OPTS = ["adamw", "neuralgrok", "grokadamw", "supergrok", "supergrok15",
        "supergrok2", "grokfast", "muon", "lion", "looksam", "prodigy"]

# ─────────────────────────────────────────────────────────────────────────────
# Search spaces — runtime scalars only; >0 lower bounds on component strengths
# (no-suppression); compiled-shape/architecture knobs fixed at spec.
# Entry: (config_key, kind, lo, hi)  kind ∈ {"log", "lin", "int"}
# ─────────────────────────────────────────────────────────────────────────────
SPACES: dict[str, list[tuple[str, str, float, float]]] = {
    "adamw": [
        ("lr", "log", 2e-4, 5e-3), ("beta1", "lin", 0.85, 0.95),
        ("beta2", "lin", 0.95, 0.9995), ("weight_decay", "log", 0.2, 4.0),
    ],
    "lion": [
        ("lion_lr", "log", 5e-5, 1e-3), ("lion_wd", "log", 0.5, 10.0),
        ("beta1", "lin", 0.85, 0.95), ("beta2", "lin", 0.9, 0.999),
    ],
    "muon": [
        ("muon_lr", "log", 5e-3, 8e-2), ("muon_momentum", "lin", 0.85, 0.99),
        ("lr", "log", 2e-4, 5e-3),  # aux AdamW lr for non-2D params
        ("weight_decay", "log", 0.2, 4.0),
    ],
    "prodigy": [
        ("prodigy_lr", "lin", 0.5, 2.0), ("weight_decay", "log", 0.2, 4.0),
        ("beta1", "lin", 0.85, 0.95), ("beta2", "lin", 0.95, 0.9995),
    ],
    "grokfast": [
        ("lr", "log", 2e-4, 5e-3), ("grokfast_alpha", "lin", 0.9, 0.999),
        ("grokfast_lamb", "log", 0.5, 8.0), ("weight_decay", "log", 0.2, 4.0),
    ],
    "grokadamw": [
        # gamma (LAYER-WISE β1 decay, β1_i=β1*(1-gamma)**i) and kappa
        # (grokking_signal_decay_rate, the α schedule rate) are WIRED mechanisms
        # of the PUBLISHED GrokAdamW (cognitivecomputations/grokadamw), so they
        # ARE tuned. gamma>0 makes later layers lower-momentum; kappa controls
        # how fast α decays with the train/val grokking signal.
        ("lr", "log", 2e-4, 5e-3), ("grokadamw_alpha", "lin", 0.9, 0.999),
        ("grokadamw_lamb", "log", 0.5, 10.0), ("grokadamw_grad_clip", "lin", 0.5, 3.0),
        ("grokadamw_gamma", "lin", 0.02, 0.5), ("grokadamw_kappa", "log", 0.02, 1.0),
        ("weight_decay", "log", 0.2, 4.0),
    ],
    "neuralgrok": [
        # neural_layers / neural_hidden fixed (architecture identity).
        # inner_steps REMOVED: it is a dead constructor arg — never reaches the
        # NeuralGrok kernel or step() (source-verified), so tuning it is pure
        # search noise.
        ("lr", "log", 2e-4, 5e-3), ("neural_alpha", "log", 2.0, 30.0),
        ("neural_beta", "lin", 1.0, 10.0),
        ("neural_grad_clip", "lin", 0.5, 3.0), ("weight_decay", "log", 0.2, 4.0),
    ],
    "looksam": [
        ("lr", "log", 2e-4, 5e-3), ("looksam_rho", "log", 0.01, 0.3),
        ("looksam_k", "int", 2, 10), ("looksam_alpha", "lin", 0.3, 0.95),
        ("weight_decay", "log", 0.2, 4.0),
    ],
    # ANTI-TAMING NOTE (owner directive): the meta-net's `rescale` is
    # ZERO-INITIALIZED, so a floor-hugging meta_lr is a de-facto meta
    # off-switch (rescale never leaves ~0 within a trial). Floors are 1e-5
    # (not 1e-6) for that reason, and the confirm stage must surface the
    # winner's meta-contribution magnitude so a config that silently zeroed
    # its own meta is VISIBLE, never an accidental "tamed" champion.
    # SuperGrok11 — meta fully active; meta_lr is the application-drift control
    # (collapse analysis: per-element phi outgrows per-element g within ~1k
    # steps at meta_lr=1e-4; the workable region is the tuner's to find).
    "supergrok": [
        # lamb is now WIRED (Option B): the SG11 binding makes it a live
        # multiplier on the canonical (1-gate)*alpha correction
        # (lamb_eff = ramp*lamb*(1-gate)*alpha), so it is a genuine dial again.
        ("supergrok_lamb", "log", 0.25, 8.0),  # LIVE post-rebuild (binding multiplies it in)
        ("lr", "log", 2e-4, 5e-3), ("weight_decay", "log", 0.2, 4.0),
        ("supergrok_alpha", "lin", 0.5, 0.995),
        ("supergrok_gamma", "lin", 0.005, 0.3),  # >0: layer-decay stays active
        ("supergrok_kappa", "log", 0.01, 1.0),
        ("supergrok_grad_clip", "lin", 0.5, 3.0),
        ("supergrok_gate_temp", "lin", 1.0, 20.0),
        ("supergrok_meta_update_freq", "int", 2, 10),
        ("supergrok_meta_lr", "log", 1e-5, 3e-4),
        ("supergrok_warmup", "int", 50, 300),
    ],
    "supergrok15": [
        # lamb NOT tuned: SG15's kernel mixing is g + gate·a·mu — no lamb slot
        # (audit find).
        ("lr", "log", 2e-4, 5e-3), ("weight_decay", "log", 0.2, 4.0),
        ("supergrok15_alpha", "lin", 0.5, 0.995),
        ("supergrok15_gamma", "lin", 0.005, 0.3),
        ("supergrok15_kappa", "log", 0.01, 1.0),
        ("supergrok15_grad_clip", "lin", 0.5, 3.0),
        ("supergrok15_meta_lr", "log", 1e-5, 3e-4),
        ("supergrok15_sam_rho", "log", 0.01, 0.2),
        ("supergrok15_gate_thresh", "lin", 0.3, 0.9),
        ("supergrok15_sam_freq_min", "int", 2, 6),
        ("supergrok15_bilevel_freq_min", "int", 2, 8),
    ],
    # SuperGrok2 — gate_thresh lower bound 0.0 keeps the gate MECHANISM in the
    # kernel while letting the tuner make amplification continuous-from-warmup
    # if that's what groks. All components on. lamb is now WIRED: the canonical
    # apply (csrc/algorithms/supergrok2.h) restored the grokfast term
    # smart_grad = g + alpha*mu + lamb_eff*slow with lamb_eff = lamb*ramp*gate;
    # the launcher previously (void)-ed lamb_eff (the drop that made earlier
    # lamb sweeps inert). It is a genuine amplification dial again.
    "supergrok2": [
        ("sg2_lamb", "log", 0.25, 8.0),  # LIVE post-rebuild (kernel consumes lamb_eff·slow)
        ("lr", "log", 2e-4, 5e-3), ("weight_decay", "log", 0.2, 4.0),
        ("sg2_alpha", "lin", 0.5, 0.995),
        ("sg2_gamma", "lin", 0.005, 0.3),
        ("sg2_meta_rescale", "log", 0.01, 0.5),
        ("sg2_meta_lr", "log", 1e-5, 3e-4),
        ("sg2_gate_scale", "lin", 2.0, 30.0),
        ("sg2_gate_thresh", "lin", 0.0, 0.85),
        ("sg2_sam_rho", "log", 0.01, 0.2),
        ("sg2_grad_clip", "lin", 0.5, 3.0),
        ("sg2_bilevel_freq_min", "int", 2, 8),
    ],
}

TRAIN_FN = {  # optimizer name -> grokking_race_v2 train function name
    "adamw": "train_adamw", "neuralgrok": "train_neuralgrok",
    "grokadamw": "train_grokadamw", "supergrok": "train_supergrok",
    "supergrok15": "train_supergrok15", "supergrok2": "train_supergrok2",
    "grokfast": "train_grokfast", "muon": "train_muon", "lion": "train_lion",
    "looksam": "train_looksam", "prodigy": "train_prodigy",
}

# ─────────────────────────────────────────────────────────────────────────────
# Trial execution (worker side)
# ─────────────────────────────────────────────────────────────────────────────
_G = None          # lazily-imported grokking_race_v2 (after CUDA ctx exists)
_DATA_CACHE = {}   # (model, seed) -> (data tuple on GPU, INIT state dict)


def _g():
    global _G
    if _G is None:
        sys.path.insert(0, str(ROOT))
        import grokking_race_v2 as g  # noqa: PLC0415
        _G = g
    return _G


def _base_config(opt: str, model: str, seed: int, cap: int) -> dict:
    g = _g()
    c = dict(g.DEFAULT_CONFIG)
    c.update(g.OPTIMIZER_CONFIGS[opt])
    c.update({
        "seed": seed, "model_type": model, "frac_train": 0.5, "val_ratio": 0.10,
        "p": 97, "use_amp": False, "use_fused": True, "compile_model": False,
        "max_steps": cap, "early_stop_max_steps": cap,
        "eval_every": EVAL_EVERY,
        # stop AT grok on the VALIDATION criterion (the standardized metric)
        "early_stop_on": "val", "early_stop_threshold": 0.95,
        "early_stop_patience": 1,
        # loud-failure mode: a config that crashes a side-component (sam/meta/
        # bilevel) must score as a CRASH, not quietly degrade to its base
        # optimizer and win as a fake config (no-suppression directive)
        "strict_components": True,
        # every-step VAL check for the stopper (1-step grokking_step
        # resolution, the owner's metric) with the expensive full triple +
        # pruner callback on the 10-step grid — ~9x cheaper than triple-eval
        # every step (vit trials were ~1.2 steps/s under the fleet)
        "fast_val_eval": True,
    })
    return c


def _data_init(model: str, seed: int):
    import torch
    key = (model, seed)
    if key not in _DATA_CACHE:
        g = _g()
        c = _base_config("adamw", model, seed, CAPS[model])
        dev = torch.device("cuda")
        data = tuple(d.to(dev) for d in g.make_data_for_task(c, seed))
        m0 = g.build_model(c, dev)
        init = {k: v.detach().cpu().clone() for k, v in m0.state_dict().items()}
        del m0
        _DATA_CACHE[key] = (data, init)
    return _DATA_CACHE[key]


def run_trial_config(opt: str, model: str, params: dict, seed: int,
                     eval_cb=None) -> dict:
    """Run one full training with `params` applied; return outcome metrics."""
    import torch
    g = _g()
    cap = CAPS[model]
    c = _base_config(opt, model, seed, cap)
    c.update(params)
    if eval_cb is not None:
        c["_eval_callback"] = eval_cb
    data, init = _data_init(model, seed)
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    fn = getattr(g, TRAIN_FN[opt])
    r = fn(c, init, *data, dev, bp=0)
    peak_val = max(r.val_accs) if r.val_accs else 0.0
    peak_train = max(r.train_accs) if r.train_accs else 0.0
    return {
        "grokking_step": r.grokking_step,
        "peak_val_acc": float(peak_val),
        "peak_train_acc": float(peak_train),  # [A6] memorize-region gradient for TPE
        "final_val_acc": float(r.final_val_acc),
        "final_test_acc": float(r.final_test_acc),
        "total_steps": int(r.total_steps),
    }


def make_objective(opt: str, model: str):
    import optuna

    cap = CAPS[model]
    # [pruner intent] dead_by = cap*0.5 (no memorization by the half-budget ⇒
    # hopeless): decoder 2500, vit 3000, mamba 6000 (the prior fixed mamba 6000
    # already equalled cap*0.5; decoder/vit move onto the same rule). Distinct
    # from the MedianPruner's first prunable rung at step 500 (n_warmup_steps
    # 490 + the every-10-step cb.report cadence) — dead_by is the never-memorized
    # hard kill, the pruner handles slow-but-progressing trials.
    dead_by = cap // 2

    def objective(trial: "optuna.Trial") -> float:
        params = {}
        for key, kind, lo, hi in SPACES[opt]:
            if kind == "log":
                params[key] = trial.suggest_float(key, lo, hi, log=True)
            elif kind == "lin":
                params[key] = trial.suggest_float(key, lo, hi)
            else:
                params[key] = trial.suggest_int(key, int(lo), int(hi))
        # Owner decision (2026-06-10): precision dimension REMOVED — industry
        # standard fixed bf16 everywhere (race DEFAULT_CONFIG); trials inherit it.

        # [A6] last-known peak val, updated live in cb, so a CONFIG crash mid-run
        # still gets a gradient-carrying score (vs a flat worst constant).
        peak_val_cell = [0.0]

        def cb(step, train_acc, val_acc, test_acc):
            # Composite progress signal: pre-grok, val sits ≈0 even for GOOD
            # configs while train memorizes first — pruning on val alone would
            # kill them. memorize→0.5, grok→≈1.45.
            #
            # Cadence note: metrics are EVALUATED every gradient step (owner
            # directive — grokking_step resolution comes from the val-stopper
            # at full fidelity), but trial.report hits the lock-guarded SHARED
            # journal — with 14 MPS workers, per-step appends serialized the
            # fleet on the file lock (GPU fell to ~23% util). Report every
            # 10th step instead; the pruner compares rungs at matched steps,
            # so pruning decisions lose nothing.
            if val_acc > peak_val_cell[0]:
                peak_val_cell[0] = val_acc
            if step % 10 == 0:
                trial.report(val_acc + 0.5 * train_acc, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if step >= dead_by and train_acc < 0.30:
                raise optuna.TrialPruned()  # never even memorized

        try:
            out = run_trial_config(opt, model, params, TUNING_SEED, eval_cb=cb)
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # noqa: BLE001
            # [A6-HIGH] Crash classification: INFRA failures (CUDA OOM, generic
            # CUDA errors, stale-ABI / pybind-boundary TypeErrors) are NOT the
            # config's fault — poisoning the trial with a crash score would make
            # TPE flee a perfectly good region. Prune them (no score). Only
            # GENUINE config failures (everything else — bad shapes, NaNs that
            # raise, component asserts) get the crash score, and that score
            # carries a gradient via the last-known peak val so TPE still learns
            # the bad region's shape rather than seeing a flat wall.
            msg = f"{type(exc).__name__}: {exc}"
            low = msg.lower()
            is_infra = (
                "out of memory" in low or "cuda error" in low or "cuda runtime" in low
                or "abi" in low or "cublas" in low or "cudnn" in low or "nccl" in low
                # Optuna JournalStorage lock blips (e.g. "Error: did not possess
                # lock" when a sibling held the symlink lock across an MPS stall)
                # are storage infrastructure, not the config — observed scored as
                # a 10100 config-DNF in the v5 era (adamw__decoder #18).
                or "possess lock" in low or "lock file" in low
                or "mps server" in low  # MPS daemon teardown/restart window
                or isinstance(exc, TypeError)  # pybind-boundary signature mismatch
            )
            if is_infra:
                trial.set_user_attr("infra_crash", msg[:300])
                raise optuna.TrialPruned()
            trial.set_user_attr("crash", msg[:300])
            return 2.0 * cap + (1.0 - peak_val_cell[0]) * 100.0
        trial.set_user_attr("peak_val_acc", out["peak_val_acc"])
        trial.set_user_attr("peak_train_acc", out["peak_train_acc"])
        trial.set_user_attr("final_test_acc", out["final_test_acc"])
        if out["grokking_step"] is not None:
            # Fake-grok guard: the SuperGrok meta-nets TRAIN on the val split,
            # so "val ≥ 0.95" alone is a circular success criterion for them —
            # a config could overfit val while test stays low. Credit the
            # val-grok only if it transferred to the held-out TEST split;
            # otherwise score it as a DNF. (Audit finding; applies uniformly to
            # all optimizers so the metric stays standardized.)
            if out["final_test_acc"] >= 0.90:
                return float(out["grokking_step"])
            trial.set_user_attr("fake_grok", True)
            return cap + (1.0 - out["final_test_acc"]) * cap
        # [A6-HIGH] DNF (no val-grok): shape the score with the better of peak
        # val and HALF peak train. Pre-grok, val sits ≈0 while train memorizes,
        # so peak-val alone is a flat wall across the whole memorize-but-no-grok
        # region; 0.5*peak_train gives TPE a gradient toward configs that at
        # least learned the train set (the necessary precursor to grokking).
        return cap + (1.0 - max(out["peak_val_acc"],
                                0.5 * out.get("peak_train_acc", 0.0))) * cap

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Storage / studies
# ─────────────────────────────────────────────────────────────────────────────

def get_storage():
    import optuna
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    return optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(TUNE_DIR / "journal.log")))


def get_study(opt: str, model: str, worker_seed: int = 0):
    import optuna
    return optuna.create_study(
        study_name=f"{opt}__{model}",
        storage=get_storage(),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=10_000 + worker_seed, multivariate=True, n_startup_trials=12),  # [A6] 8→12: more random seeding before TPE kicks in
        pruner=optuna.pruners.MedianPruner(
            # [A6] n_startup_trials 8→12 (match sampler). [pruner intent]
            # n_warmup_steps 999→490: the first prunable rung is the report at
            # step 500 (cb reports every 10th step), so 490 lets step-500 rungs
            # prune (was 999 = effectively no pruning until step 1000).
            n_startup_trials=12, n_warmup_steps=490, interval_steps=1),
        load_if_exists=True,
    )


def _finished(study) -> int:
    import optuna
    return sum(1 for t in study.trials
               if t.state in (optuna.trial.TrialState.COMPLETE,
                              optuna.trial.TrialState.PRUNED,
                              optuna.trial.TrialState.FAIL))


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def _sweep_orphan_trials(studies, wid, max_age_hours=2.0):
    """[A6-MED] Startup orphan sweep: a worker that died mid-trial (OOM kill,
    node preemption) leaves its trial stuck in RUNNING forever, and
    _finished() never counts it, so the study can never reach its target. On
    startup, fail any RUNNING trial older than `max_age_hours` so the budget
    accounting recovers. Per-trial try/except + loud print so one storage
    hiccup can't abort the sweep.

    NOTE (accepted-benign): a trial legitimately still running past 2h on a
    genuinely slow cell could be failed here and re-run — wasted compute, but
    NOT a correctness problem (the re-run just adds another sample to the
    study). We accept that over the alternative (orphans wedging the budget)."""
    import optuna
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    for o, s in studies.items():
        storage = s._storage
        try:
            trials = s.get_trials(deepcopy=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker {wid}] orphan-sweep {o}: get_trials failed: {exc!r}", flush=True)
            continue
        for t in trials:
            if t.state != optuna.trial.TrialState.RUNNING:
                continue
            started = t.datetime_start
            if started is not None and started > cutoff:
                continue  # still within budget — leave it
            try:
                storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
                print(f"[worker {wid}] orphan-sweep {o}: FAILED stale RUNNING "
                      f"trial#{t.number} (started {started})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[worker {wid}] orphan-sweep {o}: could not fail "
                      f"trial#{t.number}: {exc!r}", flush=True)


def mode_worker(args):
    """One worker process: round-robin one trial per study until all done."""
    wid = args.worker_id
    studies = {o: get_study(o, args.model, worker_seed=wid) for o in args.opts}
    objectives = {o: make_objective(o, args.model) for o in args.opts}
    _sweep_orphan_trials(studies, wid)  # [A6-MED] recover budget from dead workers
    while True:
        progressed = False
        for o in args.opts:
            s = studies[o]
            if _finished(s) >= args.trials_per_opt:
                continue
            progressed = True
            try:
                s.optimize(objectives[o], n_trials=1, gc_after_trial=True)
            except Exception as exc:  # storage hiccup etc. — keep the worker alive
                print(f"[worker {wid}] {o}: {exc!r}", flush=True)
                time.sleep(2)
        if not progressed:
            break
    print(f"[worker {wid}] all studies at target — exiting", flush=True)


def mode_launch(args):
    """Spawn N workers, wait, then write the report."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(args.workers):
        log = open(LOG_DIR / f"worker_{args.model}_{i}.log", "w")
        cmd = [sys.executable, "-m", "tuning.tune_optimizers", "--worker",
               "--worker-id", str(i), "--model", args.model,
               "--trials-per-opt", str(args.trials_per_opt),
               "--opts", ",".join(args.opts)]
        procs.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log), log))
        time.sleep(3)  # stagger CUDA context creation
    print(f"launched {len(procs)} workers (logs: {LOG_DIR})", flush=True)
    for p, log in procs:
        p.wait(); log.close()
    print("all workers done", flush=True)
    mode_report(args)


def _confirm_score(out, cap):
    """[A6-HIGH] TEST-CONFIRMED steps-to-grok score for one confirm run, mirroring
    the search objective exactly:
      • grokked (val-grok AND test ≥ 0.90)        → grokking_step  (≤ cap)
      • val-grok but test < 0.90 (fake-grok)      → cap + (1−final_test_acc)·cap
      • no val-grok                               → cap + (1−peak_val_acc)·cap
    Returns (grokked: bool, score: float)."""
    grokked = out["grokking_step"] is not None and out["final_test_acc"] >= 0.90
    if grokked:
        return True, float(out["grokking_step"])
    if out["grokking_step"] is not None:  # val-grok that did NOT confirm on test
        return False, cap + (1.0 - out["final_test_acc"]) * cap
    return False, cap + (1.0 - out["peak_val_acc"]) * cap


def mode_confirm(args):
    """Re-run each study's top-K configs on the confirm seeds; pick winners.

    [A6] The winner is the config grokking (TEST-CONFIRMED) ALL confirm seeds,
    ranked by MEDIAN steps. Fall back to the config grokking the MOST seeds, then
    median. TEST-confirmation (not val-grok alone) is the gate at every seed —
    val-trained meta-nets can fake-grok val."""
    import optuna
    cap = CAPS[args.model]
    winners = {}
    for o in args.opts:
        s = get_study(o, args.model)
        done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not done:
            print(f"[confirm] {o}: no completed trials"); continue
        top = sorted(done, key=lambda t: t.value)[:args.top_k]
        rows = []
        for t in top:
            # Seed 1001 (the trial seed) is re-derived from the trial's own
            # outcome so its grokked flag uses the SAME test-confirmation gate.
            t1001_grokked = (t.value is not None and t.value <= cap
                             and float(t.user_attrs.get("final_test_acc") or 0.0) >= 0.90)
            steps = {TUNING_SEED: float(t.value)}
            grokked_by_seed = {TUNING_SEED: bool(t1001_grokked)}
            test_acc_by_seed = {TUNING_SEED: float(t.user_attrs.get("final_test_acc") or 0.0)}
            for seed in CONFIRM_SEEDS:
                out = run_trial_config(o, args.model, t.params, seed)
                gk, sc = _confirm_score(out, cap)
                steps[seed] = sc
                grokked_by_seed[seed] = gk
                test_acc_by_seed[seed] = float(out["final_test_acc"])
                print(f"[confirm] {o} trial#{t.number} seed{seed}: "
                      f"{sc:.0f} {'GROK' if gk else 'DNF'} "
                      f"(test={out['final_test_acc']:.3f})", flush=True)
            # Robustness gate is over the held-out CONFIRM_SEEDS (test-confirmed).
            confirm_grokked = [grokked_by_seed[seed] for seed in CONFIRM_SEEDS]
            n_confirm_grokked = sum(confirm_grokked)
            grokked_all = all(confirm_grokked)
            rows.append({"trial": t.number, "params": t.params,
                         "steps_by_seed": steps,
                         "grokked_by_seed": grokked_by_seed,
                         "test_acc_by_seed": test_acc_by_seed,
                         "grokked_all": grokked_all,
                         "n_confirm_grokked": n_confirm_grokked,
                         "median_steps": statistics.median(steps.values()),
                         "mean_steps": sum(steps.values()) / len(steps)})
        # [A6] Winner: maximize confirm seeds grokked (so "all confirm seeds"
        # wins outright), break ties by MEDIAN steps. This is exactly "grok ALL
        # → most seeds → median" as a single key.
        best = min(rows, key=lambda r: (-r["n_confirm_grokked"], r["median_steps"]))
        winners[o] = best
        print(f"[confirm] {o}: winner trial#{best['trial']} "
              f"median={best['median_steps']:.0f} "
              f"confirm_grokked={best['n_confirm_grokked']}/{len(CONFIRM_SEEDS)} "
              f"robust_all={best['grokked_all']}", flush=True)
    out_path = TUNE_DIR / f"tuned_configs_{args.model}.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    for o, w in winners.items():
        existing[o] = {"model": args.model, "params": w["params"],
                       "steps_by_seed": {str(k): v for k, v in w["steps_by_seed"].items()},
                       "grokked_by_seed": {str(k): v for k, v in w["grokked_by_seed"].items()},
                       "test_acc_by_seed": {str(k): v for k, v in w["test_acc_by_seed"].items()},  # [A6] persist per-seed final_test_acc
                       "robust_all_seeds": w["grokked_all"],
                       "n_confirm_grokked": w["n_confirm_grokked"],
                       "median_steps": w["median_steps"],
                       "mean_steps": w["mean_steps"], "trial": w["trial"]}
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"wrote {out_path}", flush=True)


def mode_report(args):
    import optuna
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# Tuning report — model={args.model}",
             f"Objective: steps to val_acc ≥ 0.95 (cap {CAPS[args.model]}; "
             f"DNF = cap + (1−peak_val)·cap). Seed {TUNING_SEED}.", ""]
    csv_path = TUNE_DIR / f"trials_{args.model}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["optimizer", "trial", "state", "value", "peak_val_acc",
                    "final_test_acc", "params"])
        for o in args.opts:
            s = get_study(o, args.model)
            ts = s.trials
            comp = [t for t in ts if t.state == optuna.trial.TrialState.COMPLETE]
            pruned = [t for t in ts if t.state == optuna.trial.TrialState.PRUNED]
            for t in ts:
                w.writerow([o, t.number, t.state.name, t.value,
                            t.user_attrs.get("peak_val_acc"),
                            t.user_attrs.get("final_test_acc"),
                            json.dumps(t.params)])
            lines.append(f"## {o} — {len(ts)} trials ({len(comp)} complete, "
                         f"{len(pruned)} pruned)")
            if comp:
                best = min(comp, key=lambda t: t.value)
                grokked = best.value <= CAPS[args.model]
                lines.append(f"- best: trial#{best.number} value={best.value:.0f} "
                             f"({'GROKKED' if grokked else 'DNF'})")
                lines.append(f"- params: `{json.dumps(best.params)}`")
            lines.append("")
    (TUNE_DIR / f"TUNING_REPORT_{args.model}.md").write_text("\n".join(lines))
    print(f"wrote {TUNE_DIR / f'TUNING_REPORT_{args.model}.md'} and {csv_path}", flush=True)


def mode_measure(args):
    """Aggregate-throughput knee: K concurrent plain-AdamW runs."""
    script = (
        "import sys, time, torch; sys.path.insert(0, %r);\n"
        "import grokking_race_v2 as g\n"
        "c = dict(g.DEFAULT_CONFIG); c.update(g.OPTIMIZER_CONFIGS['adamw'])\n"
        "c.update({'seed': 1001, 'model_type': %r, 'frac_train': 0.5, 'val_ratio': 0.10,\n"
        "          'p': 97, 'use_amp': False, 'compile_model': False, 'max_steps': 600,\n"
        "          'early_stop_max_steps': 600, 'eval_every': 10**9,\n"
        "          'early_stop_threshold': 1.01, 'early_stop_patience': 10**9})\n"
        "data = tuple(d.to('cuda') for d in g.make_data_for_task(c, 1001))\n"
        "m0 = g.build_model(c, torch.device('cuda'))\n"
        "init = {k: v.detach().cpu().clone() for k, v in m0.state_dict().items()}\n"
        "del m0; torch.manual_seed(1001); t0 = time.time()\n"
        "g.train_adamw(c, init, *data, torch.device('cuda'), bp=0)\n"
        "torch.cuda.synchronize(); print('WALL', time.time() - t0)\n"
    ) % (str(ROOT), args.model)
    for k in args.measure:
        t0 = time.time()
        procs = [subprocess.Popen([sys.executable, "-c", script], cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                 for _ in range(k)]
        walls = []
        for p in procs:
            out, _ = p.communicate()
            for line in out.decode().splitlines():
                if line.startswith("WALL"):
                    walls.append(float(line.split()[1]))
        total = time.time() - t0
        agg = 600.0 * k / total
        per = [600.0 / w for w in walls]
        print(f"K={k}: aggregate={agg:7.1f} steps/s  per-run="
              f"{min(per):5.1f}-{max(per):5.1f}  wall={total:5.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="decoder", choices=list(CAPS))
    ap.add_argument("--opts", default=",".join(OPTS))
    ap.add_argument("--trials-per-opt", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)  # [A6] 3→5: confirm more candidates per study
    ap.add_argument("--measure", type=int, nargs="*", default=None)
    args = ap.parse_args()
    args.opts = [o.strip() for o in args.opts.split(",") if o.strip()]
    for o in args.opts:
        assert o in SPACES, f"unknown optimizer {o}"
    if args.measure is not None:
        mode_measure(args)
    elif args.worker:
        mode_worker(args)
    elif args.launch:
        mode_launch(args)
    elif args.confirm:
        mode_confirm(args)
    elif args.report:
        mode_report(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
