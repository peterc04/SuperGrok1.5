"""prodtime.py — per-cell PRODUCTION step timing at the roofline operating point.

Reuses tuning.roofline's exact machinery (_data_init, _tile_train_data, _cfg,
wall pass = discard-25 + diff(15, 15+timed)/timed) so the numbers are the same
quantity roofline.json records (wall_per_step_ms), but:
  * repeats the (15, 115) diff pair R times -> noise floor per cell
  * asserts EVERY l3 dispatch fired engine=="wgmma" (the wiring TRAP guard;
    a non-wgmma step aborts the cell measurement loudly)
  * no json clobber, no profiler/eval passes (timing only)

Usage:
  python .regpressure/gpu/prodtime.py --models decoder --reps 3 \
      --out .regpressure/gpu/prod_decoder_baseline.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tuning"))

import torch  # noqa: E402

import tuning.roofline as R  # noqa: E402

OPTS = ["adamw", "neuralgrok", "grokadamw", "supergrok", "supergrok15",
        "supergrok2", "grokfast", "muon", "lion", "looksam", "prodigy"]


def time_cell(opt: str, model: str, timed_steps: int, reps: int, max_batch: int):
    g = R._g()
    data, init, n_params = R._data_init(model)
    if max_batch:
        data = R._tile_train_data(data, max_batch)
    dev = torch.device("cuda")
    tuned = R._load_tuned_configs(model)
    # production race precision: bf16 pinned (the roofline default)
    mp_str, use_amp, prec_label = "bf16", False, "bf16"

    counters = {"l3": 0, "bad_engine": 0}
    engines = set()
    orig_l3 = g._try_fused_train_step

    def wrap_l3(*a, **k):
        r = orig_l3(*a, **k)
        if r is not None:
            counters["l3"] += 1
            if g.LAST_L3_ENGINE is not None:
                e = g.LAST_L3_ENGINE["engine"]
                engines.add(e)
                if e != "wgmma":
                    counters["bad_engine"] += 1
        return r

    def cfgp(steps):
        c = R._cfg(opt, model, steps, 10**9)
        c["matmul_precision"] = mp_str
        c["use_amp"] = use_amp
        return c

    fn = getattr(g, {
        "adamw": "train_adamw", "neuralgrok": "train_neuralgrok",
        "grokadamw": "train_grokadamw", "supergrok": "train_supergrok",
        "supergrok15": "train_supergrok15", "supergrok2": "train_supergrok2",
        "grokfast": "train_grokfast", "muon": "train_muon", "lion": "train_lion",
        "looksam": "train_looksam", "prodigy": "train_prodigy"}[opt])

    g._try_fused_train_step = wrap_l3
    walls_ms = []
    expected_l3 = 0
    try:
        torch.manual_seed(42)
        fn(cfgp(25), init, *data, dev, bp=0)   # discard: one-time costs
        torch.cuda.synchronize()
        expected_l3 += 25
        for _ in range(reps):
            w = []
            for steps in (15, 15 + timed_steps):
                torch.manual_seed(42)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn(cfgp(steps), init, *data, dev, bp=0)
                torch.cuda.synchronize()
                w.append(time.perf_counter() - t0)
                expected_l3 += steps
            walls_ms.append((w[1] - w[0]) / timed_steps * 1e3)
    finally:
        g._try_fused_train_step = orig_l3

    fired_all = (counters["l3"] == expected_l3)
    ok = fired_all and engines == {"wgmma"} and counters["bad_engine"] == 0
    return {
        "opt": opt, "model": model, "B": max_batch, "timed_steps": timed_steps,
        "walls_ms": walls_ms,
        "median_ms": statistics.median(walls_ms),
        "min_ms": min(walls_ms), "max_ms": max(walls_ms),
        "spread_ms": max(walls_ms) - min(walls_ms),
        "l3_fired": counters["l3"], "l3_expected": expected_l3,
        "engines": sorted(engines), "wgmma_every_step": ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="decoder")
    ap.add_argument("--opts", default=",".join(OPTS))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-batch", type=int, default=16384)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for model in [m for m in args.models.split(",") if m]:
        for opt in [o for o in args.opts.split(",") if o]:
            try:
                r = time_cell(opt, model, args.steps, args.reps, args.max_batch)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                r = {"opt": opt, "model": model, "error": str(e)}
            rows.append(r)
            if "error" in r:
                print(f"  {opt:11s}/{model:7s} FAILED: {r['error']}", flush=True)
            else:
                flag = "" if r["wgmma_every_step"] else "  ***NOT-WGMMA***"
                print(f"  {opt:11s}/{model:7s} median={r['median_ms']:9.3f} ms "
                      f"spread={r['spread_ms']:7.3f} ms "
                      f"walls={[f'{x:.3f}' for x in r['walls_ms']]}"
                      f" eng={r['engines']}{flag}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"[prodtime] wrote {args.out}", flush=True)
    bad = [r for r in rows if r.get("error") or not r.get("wgmma_every_step")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
