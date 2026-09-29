"""Command line: ``python -m grokking_race [options]``.

Default: DeepSeek-V4.1-Flash (``tiny`` preset) on modular division, 50% train,
seeds 42/123/456, all nine optimizers plus the SuperGrok 1.1 frozen-meta control, single device.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .notify import Notifier
from .report import write_plots
from .results import RunResult, print_summary, run_tag, save_json
from .tasks import TASKS, default_val_ratio, make_task
from .trainer import RunConfig, train_one


def _parse_splits(s: str) -> list[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        out.append(int(part.split("/")[0]) / 100 if "/" in part else float(part))
    return out


def main(argv=None):
    from grokking_optimizers import OPTIMIZERS

    ap = argparse.ArgumentParser(
        prog="python -m grokking_race", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--optimizers", default="all", help=f"comma list or 'all' ({','.join(OPTIMIZERS)})")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--tasks", default="moddiv", help=f"comma list from {TASKS}")
    ap.add_argument("--splits", default="0.5", help="train fractions, e.g. '0.5' or '10/90,25/75,50/50'")
    ap.add_argument("--p", type=int, default=97, help="modulus")
    ap.add_argument("--chain-length", type=int, default=3)
    ap.add_argument("--val-ratio", type=float, default=None, help="default: 0.05 at <=10%% train, else 0.10")
    ap.add_argument("--preset", default="tiny", help="DeepSeek-V4.1-Flash size preset (see deepseek_v41.PRESETS)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.95, help="accuracy that counts as grokked (default 0.95)")
    ap.add_argument(
        "--patience",
        type=int,
        default=50,
        help="evals the accuracy must stay >= threshold (default 50 evals x 10 steps = 500 steps)",
    )
    ap.add_argument(
        "--grok-metric",
        default="test",
        choices=["test", "val"],
        help="which split's accuracy defines grokking and stops the run (default: test)",
    )
    ap.add_argument("--eval-batch", type=int, default=0, help="chunk size for evaluation (0 = one pass)")
    ap.add_argument(
        "--micro-batches",
        type=int,
        default=1,
        help="accumulate the full-batch gradient over this many chunks (memory only; same math)",
    )
    ap.add_argument("--meta-frac", type=float, default=0.10, help="share of train held out for meta-learners")
    ap.add_argument("--peak-tflops", type=float, default=None, help="device peak TFLOP/s, enables MFU")
    ap.add_argument("--hparams", default=None, help="JSON {optimizer: {name: value}} overrides")
    ap.add_argument(
        "--param-policy",
        default="uniform",
        choices=["uniform", "deepseek"],
        help="uniform: one group, everything decayed (grokking convention); deepseek: DeepSeek's "
        "per-role weight decay and 5x Engram learning rate",
    )
    ap.add_argument("--model-overrides", default=None, help="JSON of deepseek_v41.Config fields for the preset")
    ap.add_argument("--output", default="results")
    ap.add_argument("--ntfy", default=os.environ.get("NTFY_TOPIC"), help="ntfy.sh topic (or env NTFY_TOPIC)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--replot",
        default=None,
        metavar="RESULTS_JSON",
        help="only redraw the summary and plots from a saved results.json",
    )
    args = ap.parse_args(argv)

    if args.replot:
        from .results import load_json

        meta, results = load_json(args.replot)
        tag = run_tag(meta["task"], meta["frac_train"], meta["preset"])
        title = (
            f"DeepSeek-V4.1-Flash ({meta['preset']}) | {meta['task']} p={meta['p']} | "
            f"train {round(meta['frac_train'] * 100)}%"
        )
        print_summary(results, title, metric=meta.get("grok_metric", "test"))
        paths = write_plots(
            results, os.path.dirname(args.replot), tag, title, meta["threshold"], meta.get("grok_metric", "test")
        )
        print("  wrote " + ", ".join(paths.values()))
        return

    names = list(OPTIMIZERS) if args.optimizers == "all" else [s.strip() for s in args.optimizers.split(",")]
    unknown = [n for n in names if n not in OPTIMIZERS]
    if unknown:
        ap.error(f"unknown optimizers {unknown}; choose from {list(OPTIMIZERS)}")
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = [s.strip() for s in args.tasks.split(",")]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RunConfig(
        preset=args.preset,
        device=device,
        dtype=args.dtype,
        compile=args.compile,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        threshold=args.threshold,
        patience=args.patience,
        grok_metric=args.grok_metric,
        eval_batch=args.eval_batch,
        micro_batches=args.micro_batches,
        meta_frac=args.meta_frac,
        peak_flops=args.peak_tflops * 1e12 if args.peak_tflops else None,
        hparams=json.loads(args.hparams) if args.hparams else {},
        progress=not args.quiet,
        param_policy=args.param_policy,
        model_overrides=json.loads(args.model_overrides) if args.model_overrides else {},
    )
    notify = Notifier(args.ntfy)
    notify(
        f"{len(names)} optimizers x {len(seeds)} seeds x {len(tasks)} task(s), preset {args.preset} on {device}",
        title="Grokking race started",
        tags=["rocket"],
    )
    t_all = time.time()
    for task_name in tasks:
        for ft in _parse_splits(args.splits):
            vr = args.val_ratio if args.val_ratio is not None else default_val_ratio(ft)
            results: dict[str, list[RunResult]] = {n: [] for n in names}
            tag = run_tag(task_name, ft, args.preset)
            first = make_task(task_name, args.p, ft, vr, seeds[0], args.chain_length)
            print(
                f"\n== {tag}: train/val/test = {first.sizes()}  seq_len {first.seq_len}  device {device}  "
                f"dtype {args.dtype}  seeds {seeds}"
            )
            for name in names:
                for seed in seeds:
                    task = make_task(task_name, args.p, ft, vr, seed, args.chain_length)
                    t0 = time.time()
                    try:
                        r = train_one(name, seed, task, cfg)
                    except Exception as e:  # a crash is recorded, never dropped
                        import traceback

                        traceback.print_exc()
                        r = RunResult(
                            optimizer=name,
                            seed=seed,
                            task=task_name,
                            frac_train=ft,
                            val_ratio=vr,
                            preset=args.preset,
                            dtype=args.dtype,
                            error=f"{type(e).__name__}: {e}"[:300],
                        )
                        notify(
                            f"{name} seed {seed} crashed: {r.error}", title="Run crashed", priority=4, tags=["warning"]
                        )
                    r.frac_train, r.val_ratio = ft, vr
                    results[name].append(r)
                    status = f"grokked at step {r.grok_step}" if r.grokked else (r.stopping_reason or "crashed")
                    print(
                        f"  {name:<12} seed {seed:<5} {status:<26} test {r.final_test_acc:.4f}  "
                        f"{r.total_steps} steps  {r.train_time:.1f}s train  {r.train_flops:.3g} FLOPs  "
                        f"({time.time() - t0:.1f}s wall)"
                    )
                    if r.grokked:
                        notify(
                            f"{name} grokked at step {r.grok_step} (seed {seed}, {tag})",
                            title="Grokked",
                            tags=["white_check_mark"],
                        )
            title = f"DeepSeek-V4.1-Flash ({args.preset}) | {task_name} p={args.p} | train {round(ft * 100)}%"
            print_summary(results, title, time.time() - t_all, metric=args.grok_metric)
            out = os.path.join(args.output, tag)
            meta = {
                "task": task_name,
                "p": args.p,
                "frac_train": ft,
                "val_ratio": vr,
                "seeds": seeds,
                "preset": args.preset,
                "device": device,
                "dtype": args.dtype,
                "max_steps": args.max_steps,
                "eval_every": args.eval_every,
                "threshold": args.threshold,
                "patience": args.patience,
                "grok_metric": args.grok_metric,
                "meta_frac": args.meta_frac,
                "param_policy": args.param_policy,
                "model_overrides": cfg.model_overrides,
                "torch": torch.__version__,
            }
            save_json(results, os.path.join(out, "results.json"), meta)
            paths = write_plots(results, out, tag, title, args.threshold, args.grok_metric)
            print(f"  wrote {out}/results.json and {', '.join(os.path.basename(p) for p in paths.values())}")
    notify(
        f"Finished in {time.time() - t_all:.0f}s", title="Grokking race complete", priority=4, tags=["checkered_flag"]
    )


if __name__ == "__main__":
    main()
