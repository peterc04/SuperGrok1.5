"""scripts/roofline_bench.py — 3-seed roofline timer for the d-scaled bench variants.

Wraps tuning/{decoder,vit,mamba}_bench.py's build_variant() so the kernel-opt ratchet
(feedback-patch-protocol) gets PROTOCOL-COMPLIANT 3-seed median step-ms at the roofline
scale WITHOUT editing the bench drivers or any kernel file. It:
  * builds the coexisting bench variant ONCE (own module name + build dir + sccache —
    never touches the production _ops.so or the 33/33 gate),
  * optionally applies hill-climb -D KEY=VAL overrides (e.g. -D SG_TUNED_DEC_DW_SPLITK=8),
  * times `reps` timed windows per seed across `--seeds` (random params/tokens reseeded
    per seed; shapes/work identical so this is a timing-variance probe, the protocol's
    3-seed guard), and reports the per-seed median ms + the across-seed median.

The candidate KEEP rule (owner ratchet): faster on >=3 seeds (every seed) AND the fp64
parity gate stays green. This harness produces the timing leg; the gate is run separately
on the production _ops.so at the toy d=128 config (the correctness oracle).

USAGE
  PYTHONPATH=. python scripts/roofline_bench.py --model decoder --d 2048 --B 16384 \
      --seeds 42,7,123 --reps 3
  PYTHONPATH=. python scripts/roofline_bench.py --model decoder --d 2048 --B 16384 \
      --seeds 42,7,123 --reps 3 -D SG_TUNED_DEC_DW_SPLITK=8
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def _import_bench(model: str):
    if model == "decoder":
        import tuning.decoder_bench as B
        return B
    if model == "vit":
        import tuning.vit_bench as B
        return B
    if model == "mamba":
        import tuning.mamba_bench as B
        return B
    raise ValueError(f"unknown model {model!r}")


def _build(model: str, B, d: int, defines, verbose: bool):
    # decoder_bench.build_variant has a distinct (d, profile, ...) signature; vit/mamba
    # are (d, ...). Normalize.
    if model == "decoder":
        return B.build_variant(d, profile=False, verbose=verbose, defines=defines)
    return B.build_variant(d, verbose=verbose, defines=defines)


def _inputs(model: str, mod, d: int, batch: int, seed: int, dev):
    """Reproduce each bench's measure() input construction, reseeded per `seed`."""
    total = int(mod.TOTAL)
    g = torch.Generator(device=dev).manual_seed(seed)
    params = (torch.randn(total, generator=g, device=dev) * 0.02).contiguous()
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    if model == "decoder":
        vocab, seq = int(mod.VOCAB), int(mod.SEQ)
        tok = torch.randint(0, vocab, (batch, seq), generator=g, device=dev).to(torch.int32).contiguous()
        tgt = torch.randint(0, vocab, (batch,), generator=g, device=dev).to(torch.int32).contiguous()
        return params, tok, tgt, state
    if model == "vit":
        from tests.hw.vit_oracle import NUM_PATCHES, PATCH_DIM, VOCAB
        x = torch.randn(batch, NUM_PATCHES, PATCH_DIM, generator=g, device=dev).to(torch.float32).contiguous()
        tgt = torch.randint(0, VOCAB, (batch,), generator=g, device=dev).to(torch.int32).contiguous()
        return params, x, tgt, state
    if model == "mamba":
        from tests.hw.mamba_oracle import VOCAB, P_HEAD, SEQ
        tok = torch.randint(0, VOCAB, (batch, SEQ), generator=g, device=dev).to(torch.int32).contiguous()
        tgt = torch.randint(0, P_HEAD, (batch,), generator=g, device=dev).to(torch.int32).contiguous()
        return params, tok, tgt, state
    raise ValueError(model)


def _flops(model: str, B, mod, d: int, batch: int) -> float:
    if model == "decoder":
        return B._decoder_gemm_flops_per_step(d, batch, int(mod.LAYERS), int(mod.SEQ), int(mod.VOCAB))
    # vit/mamba compute flops inline in measure(); recompute here from exposed attrs.
    if model == "vit":
        from tests.hw.vit_oracle import VOCAB
        layers, seq, dff = int(mod.LAYERS), int(mod.SEQ), 4 * d
        T = batch * seq
        fwd = 0.0
        for _ in range(layers):
            fwd += 2.0 * T * (3 * d) * d + 2.0 * T * d * d + 2.0 * T * dff * d + 2.0 * T * d * dff
        fwd += 2.0 * batch * VOCAB * d
        return 3.0 * fwd
    if model == "mamba":
        from tests.hw.mamba_oracle import P_HEAD
        layers, seq = int(mod.LAYERS), int(mod.SEQ)
        di, dtr, st = int(mod.DINNER), int(mod.DTRANK), int(mod.STATE)
        dbc = dtr + 2 * st
        T = batch * seq
        fwd = 0.0
        for _ in range(layers):
            fwd += 2.0 * T * (2 * di) * d + 2.0 * T * dbc * di + 2.0 * T * di * dtr + 2.0 * T * d * di
        fwd += 2.0 * batch * P_HEAD * d
        return 3.0 * fwd
    raise ValueError(model)


def time_seed(model, mod, args, seed, dev):
    params, a, b, state = _inputs(model, mod, args.d, args.B, seed, dev)
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0
    bc1, bc2 = 1.0 - beta1, 1.0 - beta2
    ncta = 0

    def call():
        return mod.tc_train_step(params, a, b, state, lr, beta1, beta2, eps, wd, bc1, bc2, 1, ncta)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    walls = []
    for _ in range(args.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            call()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) / args.iters)
    return statistics.median(walls), [w * 1e3 for w in walls]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["decoder", "vit", "mamba"])
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--B", type=int, default=16384)
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("-D", dest="defines", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()
    assert args.B % 16 == 0, "B must be %16==0"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    B = _import_bench(args.model)
    t0 = time.perf_counter()
    mod = _build(args.model, B, args.d, args.defines, args.verbose_build)
    build_s = time.perf_counter() - t0
    assert int(mod.D) == args.d, f"TU compiled at D={int(mod.D)}, expected {args.d}"
    dev = torch.device("cuda")
    flops = _flops(args.model, B, mod, args.d, args.B)

    per_seed = {}
    for s in seeds:
        med, walls = time_seed(args.model, mod, args, s, dev)
        per_seed[s] = {"median_ms": med * 1e3, "walls_ms": [round(w, 4) for w in walls],
                       "tf_s": flops / med / 1e12}
    across = statistics.median([per_seed[s]["median_ms"] for s in seeds])
    out = {
        "model": args.model, "d": args.d, "B": args.B,
        "total_params": int(mod.TOTAL), "defines": args.defines,
        "build_s": round(build_s, 1),
        "per_seed_median_ms": {s: round(per_seed[s]["median_ms"], 4) for s in seeds},
        "across_seed_median_ms": round(across, 4),
        "achieved_tf_s": round(flops / (across / 1e3) / 1e12, 3),
        "pct_roofline": round(100.0 * (flops / (across / 1e3) / 1e12) / 989.0, 3),
        "per_seed_walls_ms": {s: per_seed[s]["walls_ms"] for s in seeds},
    }
    print("ROOFLINE " + json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    main()
