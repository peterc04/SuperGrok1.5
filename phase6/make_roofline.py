#!/usr/bin/env python3
"""Aggregate CSV rows from the per-model bench logs into roofline_flagship.csv and
plot the H100 bf16 roofline PNG.

CSV row format emitted by the bench scripts:
  CSV,<model>,<opt>,<d>,<params>,<step_ms>,<tf_s>,<pct_peak>,<intensity>,<flops>,<bytes>
"""
import os, sys, csv, glob, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_CSV = "/workspace/phase6/roofline_flagship.csv"
OUT_PNG = "/workspace/phase6/roofline_flagship.png"
LOGS = ["/workspace/phase6/dec_probe.log",
        "/workspace/phase6/decoder_run.log",
        "/workspace/phase6/vit_run.log",
        "/workspace/phase6/mamba_run.log"]
# saturation-sweep logs hold the BEST-achievable (ceiling) config per model (AdamW).
SAT_LOGS = ["/workspace/phase6/dec_sat.log", "/workspace/phase6/vit_sat.log"]

# H100 SXM bf16 dense tensor-core peak + HBM3 bandwidth.
PEAK_TFS = 989.0        # TF/s bf16 dense (the ceiling the bench % is vs)
HBM_TBS = 3.35          # TB/s HBM3
HBM_BYTES_S = HBM_TBS * 1e12

def collect():
    rows = {}  # (model,opt) -> row dict (last wins)
    for lg in LOGS:
        if not os.path.exists(lg):
            continue
        with open(lg) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("CSV,"):
                    continue
                parts = line.split(",")
                if len(parts) < 11:
                    continue
                _, model, opt, d, params, step_ms, tf_s, pct, intensity, flops, bytes_ = parts[:11]
                rows[(model, opt)] = {
                    "model": model, "opt": opt, "d": int(d), "params": int(params),
                    "step_ms": float(step_ms), "tf_s": float(tf_s), "pct_peak": float(pct),
                    "intensity": float(intensity), "flops": float(flops), "bytes": float(bytes_),
                }
    return rows

def main():
    rows = collect()
    if not rows:
        print("NO CSV ROWS FOUND", flush=True)
        return
    ordered = sorted(rows.values(), key=lambda r: (r["model"], r["opt"]))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "opt", "d", "params", "step_ms", "achieved_tf_s",
                    "pct_of_989tfs_peak", "arith_intensity_flop_per_byte",
                    "gemm_flops_per_step", "bytes_moved_per_step"])
        for r in ordered:
            w.writerow([r["model"], r["opt"], r["d"], r["params"], f"{r['step_ms']:.4f}",
                        f"{r['tf_s']:.3f}", f"{r['pct_peak']:.3f}", f"{r['intensity']:.4f}",
                        f"{r['flops']:.4e}", f"{r['bytes']:.4e}"])
    print(f"wrote {OUT_CSV} ({len(ordered)} cells)", flush=True)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 7))
    # x range from data
    xs = [r["intensity"] for r in ordered]
    ys = [r["tf_s"] for r in ordered]
    xmin = min(xs) / 3.0
    xmax = max(xs) * 3.0
    # roofline: y = min(PEAK, HBM_BYTES_S * I) in TF/s
    import numpy as np
    I = np.logspace(np.log10(max(xmin, 1e-2)), np.log10(max(xmax, 1e3)), 400)
    roof = np.minimum(PEAK_TFS, (HBM_BYTES_S * I) / 1e12)
    ax.plot(I, roof, "k-", lw=2, label=f"H100 bf16 roofline (ridge {PEAK_TFS:.0f} TF/s, HBM3 {HBM_TBS} TB/s)")
    ax.axhline(PEAK_TFS, color="gray", ls="--", lw=1, alpha=0.6)
    # ridge point
    ridge_I = PEAK_TFS * 1e12 / HBM_BYTES_S
    ax.axvline(ridge_I, color="gray", ls=":", lw=1, alpha=0.5)
    ax.annotate(f"ridge I={ridge_I:.0f}", (ridge_I, PEAK_TFS*0.5), fontsize=8, rotation=90, va="center")

    markers = {"decoder": "o", "vit": "s", "mamba": "^"}
    colors = {"adamw": "#1f77b4", "lion": "#ff7f0e", "grokfast": "#2ca02c",
              "grokadamw": "#d62728", "neuralgrok": "#9467bd"}
    for r in ordered:
        m = markers.get(r["model"], "x")
        c = colors.get(r["opt"], "#333333")
        ax.scatter(r["intensity"], r["tf_s"], marker=m, c=c, s=90, edgecolors="k",
                   linewidths=0.6, zorder=5)
        ax.annotate(f"{r['model']}:{r['opt']}", (r["intensity"], r["tf_s"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (analytic GEMM FLOP / byte moved, bf16)")
    ax.set_ylabel("Achieved throughput (TF/s, analytic GEMM FLOPs / wall step)")
    ax.set_title("Flagship cells roofline vs H100 bf16 ceiling (ncu-free; CUDA-event timed)\n"
                 "decoder d=1600/L48, ViT d=1664/L48, Mamba d=2048/L24")
    # legend: model markers
    from matplotlib.lines import Line2D
    mh = [Line2D([0],[0], marker=markers[k], color="w", markerfacecolor="gray",
                 markeredgecolor="k", markersize=9, label=k) for k in markers]
    oh = [Line2D([0],[0], marker="o", color="w", markerfacecolor=colors[k],
                 markeredgecolor="k", markersize=9, label=k) for k in colors]
    leg1 = ax.legend(handles=mh, title="model", loc="lower right", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=oh, title="optimizer", loc="upper left", fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {OUT_PNG}", flush=True)

    # summary
    pcts = [r["pct_peak"] for r in ordered]
    print(f"cells={len(ordered)}  peak% range {min(pcts):.2f}..{max(pcts):.2f}", flush=True)
    for r in ordered:
        print(f"  {r['model']:8s}:{r['opt']:11s} {r['tf_s']:7.2f} TF/s ({r['pct_peak']:5.2f}%) I={r['intensity']:.2f}", flush=True)

if __name__ == "__main__":
    main()
