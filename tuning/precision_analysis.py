"""Matmul-precision analysis — FULL grid (owner spec).

7 precision arms × 33 cells (3 models × ALL 11 optimizers) × 3 seeds = 693 runs.
Arms: fp32 (reference), tf32, bf16, fp16amp (fp16 autocast + GradScaler), and
the H100-native sub-16-bit modes implemented in grokking_optimizers/lowprec.py:
fp8 (E4M3 fwd / E5M2 grads), fp8e5m2 (E5M2 acts × E4M3 weights), int8
(IMMA forward / bf16 backward). In EVERY mode weights, optimizer state, grads
at the optimizer boundary, meta/SAM side-steps and evaluate() are fp32.

Engineering for a multi-hour sweep:
  • RESUMABLE: every finished run is appended (flock'd) to
    results/h100_grokking_race/precision_analysis_full.jsonl; on restart the
    grid is diffed against recorded (precision, model, optimizer, seed) keys.
  • SHARDED WORKERS (no locking): worker i runs grid entries where
    index % nworkers == i. Launch N workers with --launch N (MPS-aware).
  • DEAD-STOP: a run that hasn't memorized (train_acc < 0.30) by half-cap is
    aborted and recorded as an early DNF (same heuristic as the tuner).
  • Verdicts count TEST-CONFIRMED groks only (val-grok with test < 0.90 is a
    fake-grok, recorded as such); partial-crash cells are counted explicitly
    (audit A6 fix — a 1-of-3-crash cell must not masquerade as low-accuracy).

Usage:
  python -m tuning.precision_analysis --launch 4      # spawn 4 shard workers
  python -m tuning.precision_analysis --worker 0 --nworkers 4   # (internal)
  python -m tuning.precision_analysis --report        # md + png + verdicts
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "h100_grokking_race"
JSONL = OUT / "precision_analysis_full.jsonl"

PRECISIONS = ("fp32", "tf32", "bf16", "fp16amp", "fp8", "fp8e5m2", "int8")
MODELS = (("decoder", 5000), ("vit", 6000), ("mamba", 12000))
OPTS = ("adamw", "neuralgrok", "grokadamw", "supergrok", "supergrok15",
        "supergrok2", "grokfast", "muon", "lion", "looksam", "prodigy")
SEEDS = (42, 123, 456)
TRAIN_FN = {"adamw": "train_adamw", "neuralgrok": "train_neuralgrok",
            "grokadamw": "train_grokadamw", "supergrok": "train_supergrok",
            "supergrok15": "train_supergrok15", "supergrok2": "train_supergrok2",
            "grokfast": "train_grokfast", "muon": "train_muon",
            "lion": "train_lion", "looksam": "train_looksam",
            "prodigy": "train_prodigy"}

_G = None

def _g():
    global _G
    if _G is None:
        sys.path.insert(0, str(ROOT))
        import grokking_race_v2 as g
        _G = g
    return _G


def grid():
    out = []
    for precision in PRECISIONS:
        for model, cap in MODELS:
            for opt in OPTS:
                for seed in SEEDS:
                    out.append((precision, model, cap, opt, seed))
    return out


def recorded_keys():
    keys = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                r = json.loads(line)
                keys.add((r["precision"], r["model"], r["optimizer"], r["seed"]))
            except Exception:
                continue  # tolerate a truncated trailing record
    return keys


def append_row(row: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(JSONL, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


class _DeadStop(Exception):
    pass


_DATA = {}

def _data_init(model, seed):
    import torch
    key = (model, seed)
    if key not in _DATA:
        g = _g()
        c = dict(g.DEFAULT_CONFIG)
        c.update({"model_type": model, "p": 97, "frac_train": 0.5,
                  "val_ratio": 0.10, "seed": seed})
        dev = torch.device("cuda")
        data = tuple(d.to(dev) for d in g.make_data_for_task(c, seed))
        m0 = g.build_model(c, dev)
        init = {k: v.detach().cpu().clone() for k, v in m0.state_dict().items()}
        del m0
        _DATA[key] = (data, init)
    return _DATA[key]


def run_one(precision, model, cap, opt, seed):
    import torch
    g = _g()
    c = dict(g.DEFAULT_CONFIG)
    c.update(g.OPTIMIZER_CONFIGS[opt])
    c.update({"seed": seed, "model_type": model, "frac_train": 0.5,
              "val_ratio": 0.10, "p": 97, "compile_model": False,
              "use_fused": True, "weight_decay": 1.0,
              "max_steps": cap, "early_stop_max_steps": cap,
              # 10-step record grid; patience is in EVALS → hold window =
              # 50 evals × 10 steps = 500 steps post-grok (collapse check)
              "eval_every": 10, "early_stop_on": "val",
              "early_stop_threshold": 0.95, "early_stop_patience": 50})
    if precision == "fp16amp":
        c["use_amp"] = True  # fp16 autocast + GradScaler (matmul_precision inert here)
    else:
        c.update({"use_amp": False, "matmul_precision": precision})
    dead_by = cap // 2

    def cb(step, train_acc, val_acc, test_acc):
        if step >= dead_by and train_acc < 0.30:
            raise _DeadStop()
    c["_eval_callback"] = cb

    data, init = _data_init(model, seed)
    dev = torch.device("cuda")
    fn = getattr(g, TRAIN_FN[opt])
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    dead = False
    try:
        r = fn(c, init, *data, dev, bp=0)
    except _DeadStop:
        dead = True
        r = None
    wall = time.perf_counter() - t0
    row = {"precision": precision, "model": model, "optimizer": opt,
           "seed": seed, "cap": cap, "wall_s": wall}
    if dead or r is None:
        row.update({"dead_stopped": True, "grok_step_val": None,
                    "test_confirmed": False, "final_test_acc": 0.0,
                    "peak_val_acc": 0.0})
        return row
    te = r.test_accs or [0.0]
    va = r.val_accs or [0.0]
    row.update({
        "dead_stopped": False,
        "grok_step_val": r.grokking_step,
        "test_confirmed": (r.grokking_step is not None
                           and r.final_test_acc >= 0.90),
        "final_test_acc": float(r.final_test_acc),
        "peak_test_acc": float(max(te)), "peak_val_acc": float(max(va)),
        "total_steps": int(r.total_steps),
        "component_failures": dict(getattr(r, "component_failures", {}) or {}),
    })
    return row


def mode_worker(args):
    g_ = grid()
    done = recorded_keys()
    mine = [e for i, e in enumerate(g_) if i % args.nworkers == args.worker]
    todo = [e for e in mine if (e[0], e[1], e[3], e[4]) not in done]
    print(f"[w{args.worker}] {len(todo)}/{len(mine)} runs pending", flush=True)
    for precision, model, cap, opt, seed in todo:
        try:
            row = run_one(precision, model, cap, opt, seed)
        except Exception as e:
            import traceback; traceback.print_exc()
            row = {"precision": precision, "model": model, "optimizer": opt,
                   "seed": seed, "cap": cap, "crashed": repr(e)[:300]}
        append_row(row)
        gs = row.get("grok_step_val")
        print(f"[w{args.worker}] {precision:8s} {opt:11s}/{model:7s} s{seed} "
              f"grok={gs} test={row.get('final_test_acc', 0):.3f} "
              f"{'DEAD' if row.get('dead_stopped') else ''}"
              f"{'CRASH' if 'crashed' in row else ''}", flush=True)
    print(f"[w{args.worker}] shard complete", flush=True)


def mode_launch(args):
    procs = []
    for i in range(args.launch):
        p = subprocess.Popen(
            [sys.executable, "-m", "tuning.precision_analysis",
             "--worker", str(i), "--nworkers", str(args.launch)],
            cwd=ROOT)
        procs.append(p)
        time.sleep(3)
    print(f"launched {len(procs)} analysis shard workers", flush=True)
    for p in procs:
        p.wait()
    print("all shards complete", flush=True)
    mode_report(args)


def mode_report(args):
    import statistics as st
    rows = []
    for line in JSONL.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    lines = ["# Matmul-precision analysis — FULL grid (7 arms × 33 cells × 3 seeds)",
             "", "Weights/opt-state/grads-at-boundary/meta/SAM/eval are fp32 in",
             "ALL modes. Verdicts count TEST-CONFIRMED groks only.", ""]
    summary = {}
    for precision in PRECISIONS:
        ok_cells = total_cells = 0
        lines.append(f"## {precision}")
        lines.append("| cell | confirmed grok | median grok step | final test (mean, non-crashed) | dead | crashed |")
        lines.append("|---|---|---|---|---|---|")
        for model, cap in MODELS:
            for opt in OPTS:
                sub = [r for r in rows if r.get("precision") == precision
                       and r.get("model") == model and r.get("optimizer") == opt]
                if not sub:
                    continue
                total_cells += 1
                crash = [r for r in sub if "crashed" in r]
                live = [r for r in sub if "crashed" not in r]
                ok = [r for r in live if r.get("test_confirmed")]
                steps = [r["grok_step_val"] for r in ok if r.get("grok_step_val")]
                fts = [r.get("final_test_acc", 0.0) for r in live]
                deads = sum(1 for r in live if r.get("dead_stopped"))
                lines.append(
                    f"| {opt}/{model} | {len(ok)}/{len(sub)} | "
                    f"{int(st.median(steps)) if steps else '—'} | "
                    f"{st.mean(fts):.3f}" + (f" ({len(live)}/{len(sub)})" if crash else "")
                    + f" | {deads} | {len(crash)} |")
                if len(ok) == len(sub) and sub:
                    ok_cells += 1
        summary[precision] = (ok_cells, total_cells)
        lines.append("")
    lines.append("## Verdicts")
    for p, (okc, tot) in summary.items():
        lines.append(f"- **{p}**: {okc}/{tot} cells at full confirmed-grok rate")
    (OUT / "PRECISION_ANALYSIS_FULL.md").write_text("\n".join(lines))
    print(f"wrote {OUT/'PRECISION_ANALYSIS_FULL.md'} ({len(rows)} rows)", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        # heatmap: cells × precisions, value = confirmed-grok count (0-3)
        cells = [f"{o}/{m}" for m, _ in MODELS for o in OPTS]
        mat = np.full((len(cells), len(PRECISIONS)), np.nan)
        for pi, precision in enumerate(PRECISIONS):
            ci = 0
            for model, _ in MODELS:
                for opt in OPTS:
                    sub = [r for r in rows if r.get("precision") == precision
                           and r.get("model") == model and r.get("optimizer") == opt]
                    if sub:
                        mat[ci, pi] = sum(1 for r in sub if r.get("test_confirmed"))
                    ci += 1
        fig, ax = plt.subplots(figsize=(9, 14))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=3, aspect="auto")
        ax.set_xticks(range(len(PRECISIONS))); ax.set_xticklabels(PRECISIONS, rotation=45)
        ax.set_yticks(range(len(cells))); ax.set_yticklabels(cells, fontsize=7)
        for i in range(len(cells)):
            for j in range(len(PRECISIONS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=7)
        ax.set_title("Test-confirmed groks per cell (of 3 seeds) × precision")
        fig.colorbar(im, shrink=0.5); fig.tight_layout()
        fig.savefig(OUT / "precision_analysis_full.png", dpi=140)
        print(f"wrote {OUT/'precision_analysis_full.png'}", flush=True)
    except Exception as e:
        print(f"figure skipped: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", type=int, default=0)
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.launch:
        mode_launch(args)
    elif args.worker >= 0:
        mode_worker(args)
    elif args.report:
        mode_report(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
