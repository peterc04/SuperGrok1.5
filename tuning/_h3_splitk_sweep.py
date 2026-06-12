"""H3 Move-2: dW split-K G sweep. Builds the decoder TC bench variant at each
G in {1,2,4,8} (via -DSG_TUNED_DEC_DW_SPLITK=G) and reports wall + P2_dW_GEMM ms,
at BOTH d=1024 (roofline) and d=128 (production). The winner is kept as the new
in-header default ONLY if it beats the current default (G=4) beyond noise at BOTH
scales (the GATE discipline). Reuses decoder_bench.build_variant/measure verbatim.

Run: python tuning/_h3_splitk_sweep.py [--Gs 1,2,4,8] [--ds 1024,128] [--B 16384]
"""
from __future__ import annotations
import argparse, sys, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tuning.decoder_bench as db  # noqa: E402

PHASE = db.PHASE_NAMES  # P2_dW_GEMM is index 3


def run_one(d, B, G, reps=3, profile=False):
    mod = db.build_variant(d, profile=profile, defines=[f"SG_TUNED_DEC_DW_SPLITK={G}"])
    res = db.measure(mod, d, B, reps=reps, warmup=4, iters=10, profile=profile, ncta_cap=0)
    dw_ms = float("nan")
    if "phase_cycles" in res:
        dw_ms = res["phase_cycles"][3] / (db.SM_GHZ * 1e9) * 1e3
    return res["wall_ms"], res["achieved_tf_s"], dw_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Gs", type=str, default="1,2,4,8")
    ap.add_argument("--ds", type=str, default="1024,128")
    ap.add_argument("--B", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile", action="store_true",
                    help="also read P2_dW_GEMM phase ms (slower: doubles the measure loop)")
    args = ap.parse_args()
    Gs = [int(x) for x in args.Gs.split(",")]
    ds = [int(x) for x in args.ds.split(",")]

    results = {}  # (d) -> list of (G, wall, tf, dw_ms)
    for d in ds:
        results[d] = []
        print(f"\n{'='*64}\n[splitk-sweep] d={d} B={args.B}\n{'='*64}", flush=True)
        for G in Gs:
            print(f"  -- building+timing G={G} ...", flush=True)
            wall, tf, dw = run_one(d, args.B, G, args.reps, args.profile)
            results[d].append((G, wall, tf, dw))
            print(f"  G={G:<2d}  wall={wall:8.3f} ms  {tf:6.3f} TF/s  "
                  f"P2_dW_GEMM={dw:8.3f} ms", flush=True)

    print(f"\n{'='*64}\n[splitk-sweep] SUMMARY (wall ms / P2_dW_GEMM ms)\n{'='*64}", flush=True)
    for d in ds:
        print(f"  d={d}:", flush=True)
        best_wall = min(results[d], key=lambda r: r[1])
        for G, wall, tf, dw in results[d]:
            mark = "  <-- min wall" if (G, wall) == (best_wall[0], best_wall[1]) else ""
            print(f"    G={G:<2d}  wall={wall:8.3f}  dW={dw:8.3f}  {tf:.3f}TF/s{mark}", flush=True)
    # Cross-scale winner: minimal wall that wins at BOTH ds (if any single G does).
    print("\n  [winner-at-both] G with min wall per scale:", flush=True)
    for d in ds:
        bw = min(results[d], key=lambda r: r[1])
        print(f"    d={d}: G={bw[0]} (wall {bw[1]:.3f} ms)", flush=True)


if __name__ == "__main__":
    main()
