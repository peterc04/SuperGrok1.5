"""scripts/time_cell.py — median step-time of a production L3-TC cell.

Times `dispatch.fused_train_step(model, opt, ...)` end-to-end (the production race
path) with CUDA events: `warmup` un-timed steps, then `iters` timed steps repeated
`reps` times; reports the median per-step wall-ms (and steps/s). Built for the
register-pressure patch ratchet (P3/P4/P5): the SAM-coupled cells (looksam/
supergrok11/supergrok15/supergrok2) run the tile fwd+bwd TWICE, which is exactly
what the `__noinline__` outlining patches target, so we force the SAM path on.

Usage:
  python scripts/time_cell.py --model vit --opts looksam,supergrok11,supergrok15,supergrok2 \
                              --seed 42 --warmup 10 --iters 30 --reps 5
Prints one JSON line per (opt) so a caller can diff baseline-vs-patched across seeds.
"""
import argparse, json, statistics, os
import torch

from grokking_optimizers.dispatch import fused_train_step, canonicalize_model
import tests.hw.test_l3tc_tail_gate as G


def _build(model, opt, seed):
    """Reuse the gate's exact cell construction (model + config + data + optimizer)
    so timing matches what the parity gate validated."""
    os.environ["GATE_SEED"] = str(seed)
    g, c, m, data, dev = G._build_cell(model)
    tx, ty = data[0], data[1]
    spec = G._CELLS[f"{opt}/{model}"]
    opt_obj = spec["factory"](g, m.parameters(), c)
    return g, c, m, data, dev, tx, ty, opt_obj


def time_cell(model, opt, seed, warmup, iters, reps):
    g, c, m, data, dev, tx, ty, opt_obj = _build(model, opt, seed)
    canon = canonicalize_model(model)
    state_cache = {}

    def step():
        # force the SAM path on (looksam_sam=1) for SAM-coupled cells so the double
        # tile pass the outlining patches optimize is exercised every timed step.
        fused_train_step(canon, opt, m, opt_obj, tx, ty, state_cache=state_cache,
                         step=1, gemm_impl="wgmma")

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    walls = []
    for _ in range(reps):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            step()
        ev1.record()
        torch.cuda.synchronize()
        walls.append(ev0.elapsed_time(ev1) / iters)   # ms/step
    return {
        "model": model, "opt": opt, "seed": seed,
        "median_ms": statistics.median(walls),
        "min_ms": min(walls),
        "walls_ms": [round(w, 4) for w in walls],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--opts", required=True, help="comma-separated optimizer keys")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    for opt in args.opts.split(","):
        opt = opt.strip()
        if not opt:
            continue
        try:
            r = time_cell(args.model, opt, args.seed, args.warmup, args.iters, args.reps)
            print("TIMING " + json.dumps(r), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"TIMING_ERROR {opt}/{args.model} seed={args.seed}: "
                  f"{type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
