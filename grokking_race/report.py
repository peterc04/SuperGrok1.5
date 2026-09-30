"""Race plots (matplotlib, PNG, white background).

``race_<tag>.png``      cost to grok in steps, training seconds and training FLOPs, side by side
``test_acc_<tag>.png``  test accuracy against steps, seconds and FLOPs: one row per optimizer
``detail_<tag>.png``    one panel per optimizer: train / test accuracy
``loss_<tag>.png``      one panel per optimizer: train / test loss (log scale)

Each optimizer keeps one color in every chart: the eight hues of a palette
validated for color-vision deficiency (adjacent pairs), and AdamW, the
baseline, in dark gray with a dashed line. Two series share a hue and are told
apart by pattern: the control run of SuperGrok 1.1 with its meta-net frozen
(dash-dot line, dotted hatching) and Shampoo, which shares Muon's (dash-dot-dot
line, cross hatching). No chart asks the reader to tell
ten hues apart by color alone (eight hues cannot all be pairwise distinct): bars
are labelled on their axis, and curves are drawn as small multiples, one
optimizer highlighted per panel.
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .results import RunResult, seeds_label, summarize  # noqa: E402

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
TRAIN_GRAY = "#a3a29d"
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
    "shampoo": "Shampoo",
    "supergrok11": "SuperGrok 1.1",
    "supergrok11_frozen": "SuperGrok 1.1, frozen meta",
}
CONTROLS = {"supergrok11_frozen": "supergrok11"}  # control run -> the optimizer whose hue it shares
# Eight hues for ten optimizers: Shampoo shares Muon's (both precondition each weight matrix as a whole; Muon is
# Shampoo without accumulated statistics) and is told apart by pattern, like a control
SHARED_HUE = {**CONTROLS, "shampoo": "muon"}
BUDGET_ATTR = {"steps": "total_steps", "seconds": "train_time", "flops": "train_flops"}


def color(name: str) -> str:
    name = SHARED_HUE.get(name, name)
    if name == "adamw":
        return BASELINE
    if name in FIXED:
        return PALETTE[FIXED[name]]
    return PALETTE[sum(map(ord, name)) % len(PALETTE)]


def label(name: str) -> str:
    return NAMES.get(name, name)


def linestyle(name: str):
    if name == "adamw":
        return "--"
    if name in CONTROLS:
        return "-."
    return (0, (4, 1.5, 1, 1.5, 1, 1.5)) if name in SHARED_HUE else "-"  # dash-dot-dot


def hatch(name: str) -> str | None:
    return "//" if name == "adamw" else ".." if name in CONTROLS else "xx" if name in SHARED_HUE else None


def dnf_hatch(name: str) -> str:
    return "..." if name in CONTROLS else "xxx" if name in SHARED_HUE else "///"


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
TITLES = {"steps": "Steps", "seconds": "Training time", "flops": "Compute"}


def _x(r: RunResult, kind: str):
    return {"steps": r.steps, "seconds": r.eval_train_time, "flops": r.eval_flops}[kind]


def _median_curve(runs, kind, attr, n=400):
    live = [r for r in runs if not r.crashed and r.steps and len(_x(r, kind)) == len(getattr(r, attr))]
    if not live:
        return None, None
    end = max(_x(r, kind)[-1] for r in live)
    grid = np.linspace(0, end, n)
    # a seed that stopped because it grokked keeps its last value; one that diverged counts as failed from then on
    dead = 0.0 if attr.endswith("accs") else math.inf
    curves = [
        np.interp(grid, _x(r, kind), getattr(r, attr), right=dead if r.stopping_reason == "non_finite_loss" else None)
        for r in live
    ]
    return grid, np.median(np.stack(curves), axis=0)


def _value_text(v, kind, fmt):
    return si(v) if fmt else (f"{v:,.0f}" if kind == "steps" else f"{v:.1f}")


def plot_race(results: dict[str, list[RunResult]], path: str, title: str):
    rows = summarize(results)
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.55 * len(rows) + 2.8), sharey=True)
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
            done = [getattr(x, per_run) for x in runs if x.grokked]
            open_ = [
                (getattr(x, BUDGET_ATTR[kind]), x.stopping_reason == "non_finite_loss") for x in runs if not x.grokked
            ]
            budget = max([getattr(x, BUDGET_ATTR[kind]) for x in runs] or [0])
            if r[key] is None:  # the (censored) median seed never grokked: outline over the largest budget spent
                ax.barh(
                    i,
                    budget,
                    height=0.6,
                    color="white",
                    edgecolor=color(name),
                    hatch=dnf_hatch(name),
                    linewidth=1.0,
                    linestyle=linestyle(name),
                )
                text = "crashed" if r["crashed"] == r["seeds"] else "DNF"
                text += f", {r['diverged']} diverged" if r["diverged"] else ""
            else:
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
                text = _value_text(r[key], kind, fmt)
            ax.scatter(done, [i] * len(done), s=22, color=INK, zorder=3, linewidths=0)
            for x_, diverged in open_:  # seeds that did not grok, at the budget they used
                style = (
                    dict(marker="x", color=INK) if diverged else dict(marker="o", facecolors="white", edgecolors=INK)
                )
                ax.scatter(
                    [x_],
                    [i],
                    s=30,
                    zorder=3,
                    linewidths=1.0,
                    **style,
                )
            # the value sits by its bar unless a marker would cover it; a DNF label sits past the budget spent
            if r[key] is None:
                right = budget
            else:
                right = max(done + [r[key]])
                if any(right < x_ <= right + 0.15 * span for x_, _ in open_):
                    right = max([right] + [x_ for x_, _ in open_])
            ax.text(right, i, f"  {text}", va="center", fontsize=8.5, color=INK if r[key] is not None else INK_2)
        ax.set_xlim(0, span * 1.25 if span > 0 else 1)
        if fmt:
            ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.set_xlabel(f"{xlabel} to grok", fontsize=9.5, color=INK_2)
        ax.set_title(TITLES[kind], fontsize=11, color=INK, loc="left")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(
        [f"{label(r['optimizer'])}  ({seeds_label(r)})" for r in rows],
        fontsize=9.5,
        color=INK,
    )
    axes[0].invert_yaxis()
    fig.text(
        0.01,
        0.012,
        "Grokked = test accuracy held at or above the threshold; grokked seeds / seeds in brackets. Bars: median "
        "over all seeds, a seed that did not grok (or crashed) counting as never, so the bar is DNF (hatched "
        "outline over the budget spent)\nwhen half or more did not grok. Filled dots: seeds that grokked. Open "
        "circles: seeds that did not, at the budget they used; crosses: seeds that diverged (non-finite loss).\n"
        "Training time counts training iterations only (evaluation excluded). FLOPs count every matrix multiply "
        "an iteration runs, including extra SAM / meta passes, Newton-Schulz and Shampoo's eigendecompositions.",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_test_curves(results: dict[str, list[RunResult]], path: str, title: str, threshold: float):
    """Small multiples: one row per optimizer (ranked), one column per cost axis. Each panel shows that
    optimizer's median test-accuracy curve in its color over every other optimizer's in light gray, so
    identity never rests on telling ten overlapping hues apart."""
    rows = summarize(results)
    order = [r["optimizer"] for r in rows]
    curves = {(n, kind): _median_curve(results[n], kind, "test_accs") for n in order for kind, *_ in AXES}
    ends = {
        kind: max([g[-1] for (n, k), (g, _) in curves.items() if k == kind and g is not None] or [1.0])
        for kind, *_ in AXES
    }
    fig, axes = plt.subplots(len(order), 3, figsize=(15, 1.55 * len(order) + 1.4), sharey=True, squeeze=False)
    fig.suptitle(title, fontsize=13, color=INK, x=0.01, ha="left")
    for i, (name, row) in enumerate(zip(order, rows)):
        for j, (kind, xlabel, _per_run, key, fmt) in enumerate(AXES):
            ax = axes[i][j]
            _style(ax)
            for other in order:
                g, m = curves[(other, kind)]
                if other != name and g is not None:
                    ax.plot(g, m, color=GRID, linewidth=1.0, zorder=1)
            g, m = curves[(name, kind)]
            if g is not None:
                ax.plot(g, m, color=color(name), linewidth=2.0, linestyle=linestyle(name), zorder=3)
            ax.axhline(threshold, color=AXIS, linewidth=0.8, linestyle=":", zorder=2)
            if row[key] is not None:  # the (censored) median cost to grok
                ax.axvline(row[key], color=INK_2, linewidth=0.8, linestyle="--", zorder=2)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0, ends[kind] * 1.02)
            ax.set_yticks([0, 0.5, 1])
            if fmt:
                ax.xaxis.set_major_formatter(FuncFormatter(fmt))
            if i == 0:
                ax.set_title(TITLES[kind], fontsize=11, color=INK, loc="left")
            if i == len(order) - 1:
                ax.set_xlabel(xlabel, fontsize=9.5, color=INK_2)
            else:
                ax.tick_params(labelbottom=False)
        axes[i][0].set_ylabel(
            f"{label(name)}\n({seeds_label(row)})",
            fontsize=9,
            color=INK,
            rotation=0,
            ha="right",
            va="center",
            labelpad=10,
        )
    fig.text(
        0.01,
        0.006,
        f"Median test accuracy over seeds (color) against every other optimizer (gray). Dotted: "
        f"the {threshold:.0%} threshold. Dashed: median cost to grok (a seed that did not grok counts as never).",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def plot_detail(results: dict[str, list[RunResult]], path: str, title: str, threshold: float, loss: bool = False):
    names = [r["optimizer"] for r in summarize(results)]
    cols = 3
    rows_n = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.4 * cols, 3.5 * rows_n), sharey=True, squeeze=False)
    fig.suptitle(title, fontsize=13, color=INK, x=0.01, ha="left")
    for i, name in enumerate(names):
        ax = axes[i // cols][i % cols]
        _style(ax)
        # test in the optimizer's own color (as everywhere else), train in a neutral dashed gray
        for split, c, ls in (("train", TRAIN_GRAY, "--"), ("test", color(name), "-")):
            g, m = _median_curve(results[name], "steps", f"{split}_losses" if loss else f"{split}_accs")
            if g is not None:
                ax.plot(g, m, color=c, linewidth=1.8, linestyle=ls, label=split)
        if loss:
            ax.set_yscale("log")
        else:
            ax.axhline(threshold, color=AXIS, linewidth=1, linestyle=":")
            ax.set_ylim(-0.02, 1.02)
        s = summarize({name: results[name]})[0]
        # identity swatch, then the title beside it (not on top of it)
        ax.plot(
            [0.0, 0.07],
            [1.055, 1.055],
            transform=ax.transAxes,
            color=color(name),
            linewidth=4,
            linestyle=linestyle(name),
            clip_on=False,
            solid_capstyle="butt",
        )
        ax.text(
            0.09,
            1.03,
            f"{label(name)}  ({seeds_label(s)} grokked)",
            transform=ax.transAxes,
            fontsize=10,
            color=INK,
            ha="left",
            va="bottom",
        )
        ax.set_xlabel("gradient steps", fontsize=8.5, color=INK_2)
    for j in range(len(names), rows_n * cols):
        axes[j // cols][j % cols].axis("off")
    handles = [
        Line2D([], [], color=TRAIN_GRAY, linestyle="--", linewidth=1.8, label="train (the examples it trains on)"),
        Line2D([], [], color=INK_2, linestyle="-", linewidth=1.8, label="test (in the optimizer's color)"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def write_plots(
    results: dict[str, list[RunResult]], out_dir: str, tag: str, title: str, threshold: float
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, f"{k}_{tag}.png") for k in ("race", "test_acc", "detail", "loss")}
    plot_race(results, paths["race"], f"{title}: cost to grok")
    plot_test_curves(results, paths["test_acc"], f"{title}: test accuracy vs steps, time and compute", threshold)
    plot_detail(results, paths["detail"], f"{title}: accuracy per optimizer", threshold)
    plot_detail(results, paths["loss"], f"{title}: loss per optimizer", threshold, loss=True)
    return paths
