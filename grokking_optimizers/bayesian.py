"""grokking_optimizers.bayesian — Optuna TPE-driven autotune.

Two-stage Bayesian sweep:

  **Stage 1** — Optuna TPE sampler explores the full search space. The
  first 10% of trials are Latin-Hypercube-like (random) to seed the
  surrogate; ~3% random throughout to escape local minima.

  **Stage 2** — for the top-K trials, generate a ±2-step neighbour grid
  per dim (deduplicated), and benchmark each survivor at full
  precision. The overall winner is ``min(timing_ms)`` across both
  stages.

This module is GPU-agnostic. It receives a ``timer`` callable that
takes a config dict and returns ``{"timing_ms", "min_ms", "max_ms",
"n"}`` (or ``None`` on failure). The caller (``compile.py``) provides
the timer that builds the variant .so and dispatches to the
``TimingWorker``.

Persistence
===========
If ``storage`` is given, the Optuna study is created with a SQLite
backend so it can be resumed across `compile.py` runs::

    storage = Path("build/optuna_lion_mamba_sm_90.db")
    trials = run_bayesian(arch, space, storage=storage, ...)
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import optuna
from optuna.samplers import TPESampler

from grokking_optimizers.search_space import (
    config_key as _ckey,
)


# ---------------------------------------------------------------------------
# §12 A1 — Pruner factory (Hyperband / Median / None)
# ---------------------------------------------------------------------------

def _make_pruner(name: str):
    """Return an Optuna pruner. ``name`` ∈ {none, median, hyperband}."""
    name = (name or "none").lower()
    if name == "none":
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=2)
    if name == "hyperband":
        # min_resource=1 rep, max_resource=21 reps (matches our default
        # iters), reduction_factor=3 (standard SHA).
        return optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=21, reduction_factor=3)
    raise ValueError(f"unknown pruner {name!r}")


TimerResult = Optional[Dict[str, Any]]
Timer = Callable[[Dict[str, Any]], TimerResult]
ProgressCb = Optional[Callable[[int, int, Dict[str, Any]], None]]


def _suggest(trial: optuna.Trial, dims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate dim specs into Optuna ``suggest_*`` calls."""
    cfg: Dict[str, Any] = {}
    for dim in dims:
        name = dim["name"]
        values = dim["values"]
        # Optuna's categorical wants hashable values. Tuples are fine;
        # lists are not, so convert.
        suggest_vals = [tuple(v) if isinstance(v, list) else v for v in values]
        cfg[name] = trial.suggest_categorical(name, suggest_vals)
    return cfg


def _make_trial_record(stage: str, trial_num: int, cfg: Dict[str, Any],
                       result: TimerResult, host: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    return {
        "trial_num":   trial_num,
        "stage":       stage,
        "config":      {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in cfg.items()},
        "config_key":  _ckey(cfg),
        "timing_ms":   result["timing_ms"] if result else None,
        "min_ms":      result["min_ms"]    if result else None,
        "max_ms":      result["max_ms"]    if result else None,
        "n":           result["n"]         if result else None,
        "host":        host,
        "recorded_at": datetime.datetime.now().isoformat(),
    }


def run_bayesian(
    arch: str,
    space: Dict[str, Any],
    *,
    n_trials: int = 500,
    seed: int = 0,
    storage: Optional[Path] = None,
    study_name: str = "sg_tune",
    timer: Timer,
    progress: ProgressCb = None,
    host: Optional[Dict[str, Any]] = None,
    prefiltered: Optional[List[Dict[str, Any]]] = None,
    pruner: str = "none",
    seed_trials: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run the TPE stage. Returns the list of trial records.

    If ``prefiltered`` is given, every suggested config that is not in
    that set is reported as a failed trial (Optuna's TPE only operates
    over the categorical product; static feasibility is enforced here).

    ``pruner`` selects an Optuna pruner: ``"none"`` (default),
    ``"median"``, or ``"hyperband"`` (§12 A1 — Hyperband / Successive
    Halving). Pruning only triggers if the timer reports intermediate
    values; the current single-shot timer does not, so the pruner is
    effectively a no-op for now. Code path is landed for forward
    compatibility.

    ``seed_trials`` (§12 A2 — transfer learning) is a list of prior
    trial records (same shape as the return value) used to warm-start
    the TPE surrogate via ``study.add_trials``. Typical use: pass
    AdamW's winners when tuning SuperGrok-2 on the same (model, arch).
    """
    dims = space[arch]["dims"]
    feasible = {_ckey(c) for c in (prefiltered or [])}

    sampler = TPESampler(
        seed=seed,
        n_startup_trials=max(10, n_trials // 10),
        multivariate=True,
        constant_liar=False,
    )

    storage_url = None
    if storage is not None:
        storage = Path(storage)
        storage.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{storage}"

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=_make_pruner(pruner),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    # §12 A2 — Transfer learning: seed the study with sibling-optimizer
    # winners. Optuna treats these as completed trials, so the TPE
    # surrogate sees them as prior knowledge before any new trial fires.
    if seed_trials:
        dim_names = [d["name"] for d in dims]
        dim_value_sets = {d["name"]: [tuple(v) if isinstance(v, list) else v
                                      for v in d["values"]]
                          for d in dims}
        seeded = 0
        for t in seed_trials:
            cfg = t.get("config") or {}
            tms = t.get("timing_ms")
            if tms is None:
                continue
            # Only seed if every dim is present in our current space.
            params: Dict[str, Any] = {}
            distributions: Dict[str, optuna.distributions.BaseDistribution] = {}
            ok = True
            for name in dim_names:
                if name not in cfg:
                    ok = False
                    break
                val = cfg[name]
                val = tuple(val) if isinstance(val, list) else val
                if val not in dim_value_sets[name]:
                    ok = False
                    break
                params[name] = val
                distributions[name] = optuna.distributions.CategoricalDistribution(
                    dim_value_sets[name])
            if not ok:
                continue
            try:
                study.add_trial(optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    value=float(tms),
                ))
                seeded += 1
            except Exception:
                continue

    records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        cfg = _suggest(trial, dims)
        if feasible and _ckey(cfg) not in feasible:
            # Static infeasible config; record a failed trial and tell
            # Optuna to assign infinity so TPE steers away.
            rec = _make_trial_record("tpe", trial.number, cfg, None, host=host)
            rec["status"] = "infeasible"
            records.append(rec)
            if progress:
                progress(trial.number + 1, n_trials, cfg)
            return math.inf
        result = timer(cfg)
        rec = _make_trial_record("tpe", trial.number, cfg, result, host=host)
        rec["status"] = "ok" if result else "build_or_time_fail"
        records.append(rec)
        if progress:
            progress(trial.number + 1, n_trials, cfg)
        if result is None:
            return math.inf
        return float(result["timing_ms"])

    # n_jobs=1 — building/timing variants is not parallelisable
    # (single GPU is exclusive).
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True,
                   show_progress_bar=False)
    return records


def _step_neighbours(value: Any, values: List[Any], radius: int = 2
                     ) -> List[Any]:
    """Return up to ``2*radius`` neighbours of ``value`` along its dim's
    ordered ``values`` list (radius-2 = ±2 steps)."""
    # YAML lists may contain other lists (tuples); normalise.
    normed = [tuple(v) if isinstance(v, list) else v for v in values]
    val = tuple(value) if isinstance(value, list) else value
    if val not in normed:
        return []
    idx = normed.index(val)
    lo = max(0, idx - radius)
    hi = min(len(normed), idx + radius + 1)
    return [normed[i] for i in range(lo, hi) if normed[i] != val]


def topk_refine(
    bayes_trials: List[Dict[str, Any]],
    space: Dict[str, Any],
    arch: str,
    *,
    top_k: int = 20,
    radius: int = 2,
    timer: Timer,
    progress: ProgressCb = None,
    host: Optional[Dict[str, Any]] = None,
    prefiltered: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """For each of the top-K TPE trials, time the ±radius-step
    neighbours along every dim. Returns the refine-stage records."""
    dims = space[arch]["dims"]
    feasible = {_ckey(c) for c in (prefiltered or [])}

    # Pick the top K successful trials by timing.
    successes = [t for t in bayes_trials if t["timing_ms"] is not None]
    successes.sort(key=lambda t: t["timing_ms"])
    seeds = successes[:top_k]

    seen_keys: set = {t["config_key"] for t in bayes_trials}
    candidate_cfgs: List[Dict[str, Any]] = []

    for seed_trial in seeds:
        base = {k: (tuple(v) if isinstance(v, list) else v)
                for k, v in seed_trial["config"].items()}
        for dim in dims:
            name = dim["name"]
            for nb in _step_neighbours(base.get(name), dim["values"], radius):
                cfg = dict(base)
                cfg[name] = nb
                k = _ckey(cfg)
                if k in seen_keys:
                    continue
                if feasible and k not in feasible:
                    continue
                seen_keys.add(k)
                candidate_cfgs.append(cfg)

    records: List[Dict[str, Any]] = []
    total = len(candidate_cfgs)
    for i, cfg in enumerate(candidate_cfgs, 1):
        result = timer(cfg)
        rec = _make_trial_record("refine", i, cfg, result, host=host)
        rec["status"] = "ok" if result else "build_or_time_fail"
        records.append(rec)
        if progress:
            progress(i, total, cfg)
    return records


def pick_winner(all_trials: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Lowest timing across all stages. Returns the winning trial record
    (with ``config`` and ``timing_ms``) or ``None``."""
    finished = [t for t in all_trials if t["timing_ms"] is not None]
    if not finished:
        return None
    return min(finished, key=lambda t: t["timing_ms"])


__all__ = ["run_bayesian", "topk_refine", "pick_winner"]
