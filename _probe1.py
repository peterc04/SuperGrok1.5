"""Run ONE optimizer's single mamba trial (default config) at a small CAP that
fits the environment's ~1-2min GPU-process kill window, and append the outcome
to /workspace/mamba_probe_results.json. Foreground-only.

Usage: python /workspace/_probe1.py <opt> <cap>
"""
import sys, json, time, os
sys.path.insert(0, "/workspace/SuperGrok1.5")
import tuning.tune_optimizers as T

OUT = "/workspace/mamba_probe_results.json"

def main():
    opt = sys.argv[1]
    cap = int(sys.argv[2])
    T.CAPS["mamba"] = cap
    t0 = time.time()
    out = T.run_trial_config(opt, "mamba", {}, T.TUNING_SEED)
    dt = time.time() - t0
    rec = {
        "opt": opt, "cap": cap, "wall_s": round(dt, 1),
        "grokking_step": out["grokking_step"],
        "peak_val_acc": round(out["peak_val_acc"], 4),
        "peak_train_acc": round(out["peak_train_acc"], 4),
        "final_val_acc": round(out["final_val_acc"], 4),
        "final_test_acc": round(out["final_test_acc"], 4),
        "grokked": (out["grokking_step"] is not None and out["final_test_acc"] >= 0.90),
    }
    data = {}
    if os.path.exists(OUT):
        try:
            data = json.loads(open(OUT).read())
        except Exception:
            data = {}
    data[opt] = rec
    open(OUT, "w").write(json.dumps(data, indent=2))
    print(f"PROBE_DONE {opt} cap={cap} wall={dt:.1f}s grok={rec['grokked']} "
          f"peak_train={rec['peak_train_acc']} peak_val={rec['peak_val_acc']} "
          f"test={rec['final_test_acc']}", flush=True)

if __name__ == "__main__":
    main()
