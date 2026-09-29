"""Early stopping, per-run results, the console summary and the JSON record."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np


class EarlyStopper:
    """Grokked = the watched accuracy stays >= ``threshold`` for ``patience`` consecutive evals.

    ``grok_step`` is the first step of that held streak. ``first_cross_step``
    records the first eval that touched the threshold at all, held or not. (The
    previous driver called any single crossing a grok and never cleared it.)
    """

    def __init__(self, threshold: float = 0.95, patience: int = 50):
        self.threshold = threshold
        self.patience = patience
        self.best = 0.0
        self.first_cross_step: int | None = None
        self.grok_step: int | None = None
        self._streak = 0
        self._streak_start: int | None = None

    def update(self, metric: float, step: int) -> bool:
        """Feed one eval; returns True when the run should stop (grokking confirmed)."""
        self.best = max(self.best, metric)
        if metric < self.threshold:
            self._streak = 0
            return False
        if self.first_cross_step is None:
            self.first_cross_step = step
        if self._streak == 0:
            self._streak_start = step
        self._streak += 1
        if self._streak >= self.patience:
            self.grok_step = self._streak_start
            return True
        return False


@dataclass
class RunResult:
    optimizer: str
    seed: int
    task: str
    frac_train: float
    val_ratio: float
    preset: str = ""
    dtype: str = "fp32"
    grok_metric: str = "test"
    hparams: dict = field(default_factory=dict)
    model_params: int = 0
    active_params: int = 0
    optimizer_state_bytes: int = 0
    # curves, one entry per eval
    steps: list = field(default_factory=list)
    train_losses: list = field(default_factory=list)
    train_accs: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    val_accs: list = field(default_factory=list)
    test_losses: list = field(default_factory=list)
    test_accs: list = field(default_factory=list)
    eval_train_time: list = field(default_factory=list)  # cumulative training seconds at each eval
    eval_flops: list = field(default_factory=list)  # cumulative training FLOPs at each eval
    diagnostics: list = field(default_factory=list)  # optimizer-internal signals at each eval (if any)
    # outcome
    total_steps: int = 0
    stopping_reason: str | None = None
    grokked: bool = False
    grok_step: int | None = None
    grok_train_time: float | None = None
    grok_flops: float | None = None
    grok_val_acc: float | None = None
    first_cross_step: int | None = None
    best_metric_acc: float = 0.0
    final_train_acc: float = 0.0
    final_val_acc: float = 0.0
    final_test_acc: float = 0.0
    final_test_loss: float = 0.0
    # cost
    wall_time: float = 0.0
    train_time: float = 0.0
    train_flops: float = 0.0
    flops_per_step_kind: dict = field(default_factory=dict)
    ms_per_step: float = 0.0
    tokens_per_sec: float = 0.0
    mfu: float | None = None
    peak_mem_bytes: int | None = None
    error: str | None = None

    @property
    def crashed(self) -> bool:
        return self.error is not None


def _median(xs):
    return float(np.median(xs)) if xs else None


def summarize(results: dict[str, list[RunResult]]) -> list[dict]:
    """One row per optimizer, ranked by median grok step (grokked seeds only); all-DNF rows last."""
    rows = []
    for name, runs in results.items():
        live = [r for r in runs if not r.crashed]
        grokked = [r for r in live if r.grokked]
        rows.append(
            {
                "optimizer": name,
                "seeds": len(runs),
                "crashed": len(runs) - len(live),
                "grokked": len(grokked),
                "grok_step_median": _median([r.grok_step for r in grokked]),
                "grok_time_median": _median([r.grok_train_time for r in grokked]),
                "grok_flops_median": _median([r.grok_flops for r in grokked]),
                "test_acc_mean": float(np.mean([r.final_test_acc for r in live])) if live else None,
                "test_acc_std": float(np.std([r.final_test_acc for r in live])) if live else None,
                "ms_per_step": float(np.mean([r.ms_per_step for r in live])) if live else None,
                "flops_per_step": float(np.mean([r.train_flops / max(r.total_steps, 1) for r in live]))
                if live
                else None,
                "mfu": float(np.mean([r.mfu for r in live])) if live and all(r.mfu is not None for r in live) else None,
                "state_mb": live[0].optimizer_state_bytes / 2**20 if live else None,
            }
        )
    rows.sort(key=lambda r: (-r["grokked"], r["grok_step_median"] if r["grok_step_median"] is not None else math.inf))
    return rows


def _fmt(x, spec, missing="-"):
    return missing if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def _flops(x):
    if x is None:
        return "-"
    for unit, scale in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6)):
        if x >= scale:
            return f"{x / scale:.2f}{unit}"
    return f"{x:.0f}"


def print_summary(
    results: dict[str, list[RunResult]], title: str, total_wall: float | None = None, metric: str = "test"
) -> None:
    rows = summarize(results)
    width = 124
    print("\n" + "=" * width)
    print(f"  GROKKING RACE | {title}")
    print("=" * width)
    print(
        f"  {'#':>2} {'Optimizer':<12} {'Grokked':>8} {'Steps':>8} {'Seconds':>9} {'FLOPs':>9} "
        f"{'Final test acc':>16} {'ms/step':>9} {'FLOPs/step':>11} {'MFU':>6} {'State MB':>9}"
    )
    print("  " + "-" * (width - 2))
    for i, r in enumerate(rows, 1):
        ok = r["seeds"] - r["crashed"]
        test = "-" if r["test_acc_mean"] is None else f"{r['test_acc_mean']:.4f}±{r['test_acc_std']:.4f}"
        mfu = "-" if r["mfu"] is None else f"{100 * r['mfu']:.1f}%"
        print(
            f"  {i:>2} {r['optimizer']:<12} {r['grokked']:>4}/{ok:<3} {_fmt(r['grok_step_median'], ',.0f'):>8} "
            f"{_fmt(r['grok_time_median'], '.1f'):>9} {_flops(r['grok_flops_median']):>9} {test:>16} "
            f"{_fmt(r['ms_per_step'], '.2f'):>9} {_flops(r['flops_per_step']):>11} {mfu:>6} "
            f"{_fmt(r['state_mb'], '.1f'):>9}"
        )
    print("  " + "-" * (width - 2))
    print(f"  Steps / Seconds / FLOPs: to grok, median over grokked seeds. Grokked = {metric} accuracy held >= the")
    print("  threshold for `patience` evals. Seconds = training iterations only (evals excluded). FLOPs = matmul")
    print("  FLOPs of everything each iteration runs (extra SAM/meta passes, Newton-Schulz, amplifiers).")
    crashes = [(n, r) for n, runs in results.items() for r in runs if r.crashed]
    if crashes:
        print(f"  CRASHES ({len(crashes)}):")
        for n, r in crashes:
            print(f"    {n} seed={r.seed}: {r.error}")
    if total_wall is not None:
        print(f"  Total wall: {total_wall:.1f}s")
    print("=" * width)


def run_tag(task: str, frac_train: float, preset: str) -> str:
    return f"{task}_ft{round(frac_train * 100)}_{preset}"


def save_json(results: dict[str, list[RunResult]], path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {"_meta": meta, "_summary": summarize(results)}
    for name, runs in results.items():
        out[name] = [asdict(r) for r in runs]
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def load_json(path: str) -> tuple[dict, dict[str, list[RunResult]]]:
    """Read a results.json back into RunResult objects (for re-plotting)."""
    with open(path) as f:
        raw = json.load(f)
    fields_ = RunResult.__dataclass_fields__
    results = {
        k: [RunResult(**{f: v for f, v in r.items() if f in fields_}) for r in v]
        for k, v in raw.items()
        if not k.startswith("_")
    }
    return raw.get("_meta", {}), results
