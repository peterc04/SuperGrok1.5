"""Race plots (matplotlib, PNG, 150 dpi, white background).

``race_<tag>.png``      cost to grok in steps, training seconds and training FLOPs, side by side
``test_acc_<tag>.png``  test accuracy against steps, seconds and FLOPs (median over seeds)
``detail_<tag>.png``    one panel per optimizer: train / val / test accuracy
``loss_<tag>.png``      one panel per optimizer: train / val / test loss (log scale)

Each optimizer keeps one color in every chart: the eight hues of a palette
validated for color-vision deficiency (adjacent pairs), and AdamW, the
baseline, in dark gray with a dashed line. Nine series is one past what hue
alone can separate, so the baseline is told apart by line style as well, and
every bar and line is labelled. A control run of an optimizer (SuperGrok 1.1
with its meta-net frozen) shares its parent's hue and is told apart by a
dash-dot line and dotted bar hatching.
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .results import RunResult, summarize  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BASELINE = "#52514e"
FIXED = {
    "muon": 0,
    "grokfast": 1,
    "lion": 2,
    "grokadamw": 3,
    "prodigy": 4,
    "looksam": 5,
    "neuralgrok": 6,
    "supergrok11": 7,
}
GRID, AXIS, INK, INK_2 = "#e4e3df", "#8f8e8a", "#0b0b0b", "#52514e"
NAMES = {
    "adamw": "AdamW",
    "lion": "Lion",
    "grokfast": "Grokfast",
    "grokadamw": "GrokAdamW",
    "prodigy": "Prodigy",
    "looksam": "LookSAM",
    "neuralgrok": "NeuralGrok",
    "muon": "Muon",
    "supergrok11": "SuperGrok 1.1",
    "supergrok11_frozen": "SuperGrok 1.1, frozen meta",
}
CONTROLS = {"supergrok11_frozen": "supergrok11"}  # control run -> the optimizer whose hue it shares
BUDGET_ATTR = {"steps": "total_steps", "seconds": "train_time", "flops": "train_flops"}


def color(name: str) -> str:
    name = CONTROLS.get(name, name)
    if name == "adamw":
        return BASELINE
    if name in FIXED:
        return PALETTE[FIXED[name]]
    return PALETTE[sum(map(ord, name)) % len(PALETTE)]


def label(name: str) -> str:
    return NAMES.get(name, name)


def linestyle(name: str) -> str:
    return "--" if name == "adamw" else "-." if name in CONTROLS else "-"


def hatch(name: str) -> str | None:
    return "//" if name == "adamw" else ".." if name in CONTROLS else None


def _style(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def si(x, _pos=None) -> str:
    """1.23e15 -> '1.23P' (FLOP counts)."""
    for unit, scale in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= scale:
            return f"{x / scale:.3g}{unit}"
    return f"{x:.3g}"


AXES = [  # kind, axis label, per-run attribute, summary key, tick formatter
    ("steps", "gradient steps", "grok_step", "grok_step_median", None),
    ("seconds", "training time (s)", "grok_train_time", "grok_time_median", None),
    ("flops", "training FLOPs", "grok_flops", "grok_flops_median", si),
]
TITLES = {"steps": "Steps", "seconds": "Wall-clock", "flops": "Compute"}


def _x(r: RunResult, kind: str):
    return {"steps": r.steps, "seconds": r.eval_train_time, "flops": r.eval_flops}[kind]


def _median_curve(runs, kind, attr, n=400):
    live = [r for r in runs if not r.crashed and r.steps and len(_x(r, kind)) == len(getattr(r, attr))]
    if not live:
        return None, None
    end = max(_x(r, kind)[-1] for r in live)
    grid = np.linspace(0, end, n)
    curves = [np.interp(grid, _x(r, kind), getattr(r, attr)) for r in live]
    return grid, np.median(np.stack(curves), axis=0)


def plot_race(results: dict[str, list[RunResult]], path: str, title: str, metric: str):
    rows = summarize(results)
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.55 * len(rows) + 2.6), sharey=True)
    fig.suptitle(title, fontsize=13, color=INK, x=0.01, ha="left", y=0.995)
    live = [x for runs in results.values() for x in runs if not x.crashed]
    for ax, (kind, xlabel, per_run, key, fmt) in zip(axes, AXES):
        _style(ax, "x")
        spent = [getattr(x, BUDGET_ATTR[kind]) for x in live]
        finite = [r[key] for r in rows if r[key] is not None]
        span = max(finite + spent) if finite + spent else 1.0
        for i, r in enumerate(rows):
            name = r["optimizer"]
            runs = [x for x in results[name] if not x.crashed]
            if r[key] is None:  # no seed grokked: hatched outline over the budget actually spent
                budget = max([getattr(x, BUDGET_ATTR[kind]) for x in runs] or [0])
                ax.barh(i, budget, height=0.6, color="white", edgecolor=color(name), hatch="///", linewidth=1.0)
                ax.text(budget, i, "  DNF", va="center", fontsize=8.5, color=INK_2)
                continue
            ax.barh(
                i,
                r[key],
                height=0.6,
                color=color(name),
                alpha=0.92,
                edgecolor="white",
                linewidth=0.8,
                hatch=hatch(name),
            )
            vals = [getattr(x, per_run) for x in runs if x.grokked]
            ax.scatter(vals, [i] * len(vals), s=22, color=INK, zorder=3, linewidths=0)
            text = si(r[key]) if fmt else (f"{r[key]:,.0f}" if kind == "steps" else f"{r[key]:.1f}")
            ax.text(max(vals + [r[key]]), i, f"  {text}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, span * 1.2 if span > 0 else 1)
        if fmt:
            ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.set_xlabel(f"{xlabel} to grok", fontsize=9.5, color=INK_2)
        ax.set_title(TITLES[kind], fontsize=11, color=INK, loc="left")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(
        [f"{label(r['optimizer'])}  ({r['grokked']}/{r['seeds'] - r['crashed']})" for r in rows],
        fontsize=9.5,
        color=INK,
    )
    axes[0].invert_yaxis()
    fig.text(
        0.01,
        0.012,
        f"Grokked = {metric} accuracy held at or above the threshold (count of grokked seeds in brackets). "
        "Bars: median over grokked seeds; dots: individual seeds; hatched outline: no seed grokked "
        "(bar = budget spent).\nSeconds count training iterations only (evaluation excluded). FLOPs count "
        "every matrix multiply an iteration runs, including extra SAM / meta passes and Newton-Schulz.",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_test_curves(results: dict[str, list[RunResult]], path: str, title: str, threshold: float):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), sharey=True)
    fig.suptitle(title, fontsize=13, color=INK, x=0.01, ha="left")
    order = [r["optimizer"] for r in summarize(results)]
    for ax, (kind, xlabel, *_rest) in zip(axes, AXES):
        _style(ax)
        for name in order:
            g, m = _median_curve(results[name], kind, "test_accs")
            if g is None:
                continue
            ax.plot(g, m, color=color(name), linewidth=2.2, linestyle=linestyle(name), label=label(name))
        ax.axhline(threshold, color=AXIS, linewidth=1, linestyle=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(left=0)
        if kind == "flops":
            ax.xaxis.set_major_formatter(FuncFormatter(si))
        ax.set_xlabel(xlabel, fontsize=9.5, color=INK_2)
        ax.set_title(TITLES[kind], fontsize=11, color=INK, loc="left")
    axes[0].set_ylabel("test accuracy (median over seeds)", fontsize=9.5, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(len(labels), 9), frameon=False, fontsize=9.5, labelcolor=INK
        )
    fig.text(0.99, 0.935, f"dotted line: {threshold:.0%} threshold", fontsize=8, color=INK_2, ha="right")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_detail(results: dict[str, list[RunResult]], path: str, title: str, threshold: float, loss: bool = False):
    names = [r["optimizer"] for r in summarize(results)]
    cols = 3
    rows_n = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.4 * cols, 3.5 * rows_n), sharey=True, squeeze=False)
    fig.suptitle(title, fontsize=13, color=INK, x=0.01, ha="left")
    series = [("train", "#1baf7a"), ("val", "#2a78d6"), ("test", "#eb6834")]  # the all-pairs-safe first 3 slots
    for i, name in enumerate(names):
        ax = axes[i // cols][i % cols]
        _style(ax)
        for split, c in series:
            g, m = _median_curve(results[name], "steps", f"{split}_losses" if loss else f"{split}_accs")
            if g is not None:
                ax.plot(g, m, color=c, linewidth=1.8, label=split)
        if loss:
            ax.set_yscale("log")
        else:
            ax.axhline(threshold, color=AXIS, linewidth=1, linestyle=":")
            ax.set_ylim(-0.02, 1.02)
        s = summarize({name: results[name]})[0]
        ax.set_title(
            f"{label(name)}  ({s['grokked']}/{s['seeds'] - s['crashed']} grokked)", fontsize=10, color=INK, loc="left"
        )
        ax.plot(
            [0.0, 0.06],
            [1.03, 1.03],
            transform=ax.transAxes,
            color=color(name),
            linewidth=4,
            linestyle=linestyle(name),
            clip_on=False,
            solid_capstyle="butt",
        )  # identity swatch next to the title
        ax.set_xlabel("gradient steps", fontsize=8.5, color=INK_2)
    for j in range(len(names), rows_n * cols):
        axes[j // cols][j % cols].axis("off")
    h, lab = axes[0][0].get_legend_handles_labels()
    if lab:
        fig.legend(h, lab, loc="upper right", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def write_plots(
    results: dict[str, list[RunResult]], out_dir: str, tag: str, title: str, threshold: float, metric: str = "test"
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, f"{k}_{tag}.png") for k in ("race", "test_acc", "detail", "loss")}
    plot_race(results, paths["race"], f"{title}: cost to grok", metric)
    plot_test_curves(results, paths["test_acc"], f"{title}: test accuracy vs steps, time and compute", threshold)
    plot_detail(results, paths["detail"], f"{title}: accuracy per optimizer", threshold)
    plot_detail(results, paths["loss"], f"{title}: loss per optimizer", threshold, loss=True)
    return paths
