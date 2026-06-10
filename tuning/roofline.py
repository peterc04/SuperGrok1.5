"""Roofline analysis for every race pipeline (optimizer × model) on H100.

Owner directive: the optimization target is distance-to-roofline (not watts),
tracked at 10-step granularity.

Methodology (ncu SpeedOfLight is unavailable in this container —
ERR_NVGPUCTRPERM/RmProfilingAdminOnly=1 — so this uses first-principles
measurement, stated explicitly):
  • achieved FLOP/s  = FLOPs_per_step / wall_per_step
      - wall_per_step: the REAL race train function run with evaluation
        disabled (pure train steps), CUDA-synchronized wall over 100 steps
        after 25 warmup steps. A second pass with eval_every=10 records the
        per-10-step time series via the race's _eval_callback hook (labelled
        "incl. eval overhead" — the race's own tracking cadence).
      - FLOPs_per_step: torch.profiler(with_flops=True) over 20 steps of the
        same pipeline (GEMM/conv-registered FLOPs; elementwise ops register 0,
        so this slightly UNDERcounts — conservative for "how close to roof").
  • arithmetic intensity AI = FLOPs_per_step / bytes_per_step, with an
    analytical traffic model (documented in bytes_per_step below): weights are
    read for fwd+bwd, grads written+read, optimizer state read+written
    (per-optimizer state-tensor counts), activations written in fwd and read
    in bwd.
  • roofline ceiling at AI = min(PEAK_COMPUTE, AI × HBM_BW); closeness =
    achieved / ceiling. PEAK_COMPUTE is chosen by what eager fp32 matmul can
    legally hit on this torch build (TF32 tensor cores if
    torch.backends.cuda.matmul.allow_tf32 else FP32 CUDA cores); both ceilings
    are drawn on the plot regardless.

Outputs: results/h100_grokking_race/roofline.json, roofline.png, ROOFLINE.md.
Run:  python -m tuning.roofline [--models decoder,vit,mamba] [--steps 100]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "h100_grokking_race"

# H100 SXM5 datasheet ceilings (dense, no sparsity)
PEAKS = {
    "fp32_cuda": 66.9e12,    # FP32 CUDA cores
    "tf32_tc": 494.7e12,     # TF32 tensor cores
    "bf16_tc": 989.4e12,     # BF16/FP16 tensor cores
    # [A6-HIGH] FP8/INT8 tensor-core dense rate. H100 datasheet quotes 3958
    # TFLOP/s FP8 *with sparsity*; dense = 3958/2 = 1979 ≈ 1978.9e12. INT8 TOPS
    # matches the FP8 rate on H100, so both share this ceiling.
    "fp8_tc": 1978.9e12,     # FP8 (e4m3/e5m2) tensor cores, dense
    "int8_tc": 1978.9e12,    # INT8 tensor cores, dense (= FP8 rate)
    "hbm_bw": 3.35e12,       # HBM3 bytes/s
}

# [A6-HIGH] precision label -> compute ceiling. fp16amp rides the BF16/FP16 TC
# carrier (same peak as bf16); fp8/fp8e5m2/int8 use the low-precision TC rate.
_PRECISION_PEAK = {
    "bf16": PEAKS["bf16_tc"], "fp16amp": PEAKS["bf16_tc"],
    "tf32": PEAKS["tf32_tc"], "fp32": PEAKS["fp32_cuda"],
    "fp8": PEAKS["fp8_tc"], "fp8e5m2": PEAKS["fp8_tc"], "int8": PEAKS["int8_tc"],
}

OPTS = ["adamw", "neuralgrok", "grokadamw", "supergrok", "supergrok15",
        "supergrok2", "grokfast", "muon", "lion", "looksam", "prodigy"]

# [A6-HIGH] Optimizer state tensors read+written per step, in units of
# model-param-sized tensors (4 bytes/elem fp32), source-verified against each
# optimizer's state init (grokking_optimizers/optimizers/*.py):
#   adamw      = 2  : exp_avg, exp_avg_sq                       (adamw.py L134-135)
#   lion       = 1  : exp_avg only                              (lion.py L109)
#   grokfast   = 3  : ema, exp_avg, exp_avg_sq                  (grokfast.py L143-146)
#   grokadamw  = 3  : exp_avg, exp_avg_sq, ema                  (grokadamw.py L162-167)
#   muon       = 2  : 2D→momentum_buffer (+NS scratch); 1D→adamw m,v — the 2D
#                     momentum is 1 tensor, the upper bound 2 covers the 1D aux
#                     AdamW params                              (muon.py L167 / L192-193)
#   prodigy    = 4  : exp_avg, exp_avg_sq, s, param_init — s AND param_init are
#                     FULL param-sized tensors, not scalars (was wrongly 2)
#                                                               (prodigy.py L123-126)
#   neuralgrok = 2  : exp_avg, exp_avg_sq (+tiny amplifier MLP, negligible)
#                                                               (neuralgrok.py L213-214)
#   looksam    = 3  : exp_avg, exp_avg_sq, sam_direction        (looksam.py L113-115)
#   supergrok  = 4  : _flat_exp_avgs, _flat_exp_avg_sqs, _flat_mus, _flat_sharpness
#                                                               (supergrok11.py L198-201)
#   supergrok15= 4  : same four flat buffers                    (supergrok15.py L229-232)
#   supergrok2 = 9  : exp_avg, exp_avg_sq, mu, slow, sharpness  (= 5 param-sized)
#                     + gru_states of shape (N, gru_hidden=4) = 4 elem/param ⇒ 5+4=9
#                     (was 8 — missed the restored _flat_slows grokfast buffer)
#                                                               (supergrok2.py L1412-1427)
STATE_TENSORS = {
    "adamw": 2, "lion": 1, "grokfast": 3, "grokadamw": 3, "muon": 2,
    "prodigy": 4, "neuralgrok": 2, "looksam": 3,
    "supergrok": 4, "supergrok15": 4, "supergrok2": 9,
}

_G = None

def _g():
    global _G
    if _G is None:
        sys.path.insert(0, str(ROOT))
        import grokking_race_v2 as g
        _G = g
    return _G


def _cfg(opt, model, steps, eval_every):
    g = _g()
    c = dict(g.DEFAULT_CONFIG)
    c.update(g.OPTIMIZER_CONFIGS[opt])
    c.update({"seed": 42, "model_type": model, "frac_train": 0.5, "val_ratio": 0.10,
              "p": 97, "use_amp": False, "use_fused": True, "compile_model": False,
              "max_steps": steps, "early_stop_max_steps": steps,
              "eval_every": eval_every, "early_stop_threshold": 1.01,
              "early_stop_patience": 10**9, "weight_decay": 1.0})
    return c


_CACHE = {}

def _data_init(model):
    import torch
    if model not in _CACHE:
        g = _g()
        c = _cfg("adamw", model, 10, 10**9)
        dev = torch.device("cuda")
        data = tuple(d.to(dev) for d in g.make_data_for_task(c, 42))
        m0 = g.build_model(c, dev)
        init = {k: v.detach().cpu().clone() for k, v in m0.state_dict().items()}
        n_params = sum(p.numel() for p in m0.parameters())
        del m0
        _CACHE[model] = (data, init, n_params)
    return _CACHE[model]


def bytes_per_step(opt, model, n_params, batch, seq, dim, layers):
    """Analytical main-memory traffic per training step (bytes, fp32).

    weights: fwd read + bwd read           = 2 · 4N
    grads:   bwd write + optimizer read    = 2 · 4N
    params @ optimizer: read + write       = 2 · 4N
    state:   read + write per state tensor = 2 · 4N · S_opt
    activations: fwd write + bwd read      = 2 · 4 · A
      A ≈ batch · seq · dim · layers · k, k≈14 for attention blocks
      (qkv/attn-matrix/proj/mlp saves), k≈10 for mamba blocks.
    """
    k = 10 if model == "mamba" else 14
    acts = batch * seq * dim * layers * k
    s = STATE_TENSORS[opt]
    return 4.0 * (2 * n_params + 2 * n_params + 2 * n_params
                  + 2 * n_params * s + 2 * acts)


_TUNED_CACHE = {}

def _load_tuned_configs(model, path=None):
    """[A6-HIGH] Load the tuner's winners for `model` from
    results/tuning/tuned_configs_{model}.json (default). Returns {} if absent so
    the caller falls back to the auto-resolved default precision."""
    if path is None:
        path = ROOT / "results" / "tuning" / f"tuned_configs_{model}.json"
    path = Path(path)
    key = str(path)
    if key not in _TUNED_CACHE:
        try:
            _TUNED_CACHE[key] = json.loads(path.read_text()) if path.exists() else {}
        except Exception:  # noqa: BLE001 — a malformed file just means "no tuned precision"
            _TUNED_CACHE[key] = {}
    return _TUNED_CACHE[key]


def _resolve_precision(opt, model, tuned_configs):
    """[A6-HIGH] (matmul_precision_str, use_amp, precision_label) for (opt, model).

    If the tuner picked a winner for this cell, use ITS precision/AMP; else fall
    back to grokking_race_v2._resolve_matmul_precision on the default config (the
    per-model auto policy). The label is what gets drawn on the plot and chosen
    from _PRECISION_PEAK: fp16amp (use_amp=True) rides the bf16 carrier."""
    g = _g()
    params = (tuned_configs.get(opt) or {}).get("params") if tuned_configs else None
    if params:
        if params.get("use_amp"):
            return "fp32", True, "fp16amp"      # fp16 autocast; carrier peak = bf16
        mp = params.get("matmul_precision")
        if mp:
            return mp, False, mp
    # Fallback: the race's auto policy for this model (decoder→bf16, etc.).
    dc = dict(g.DEFAULT_CONFIG); dc["model_type"] = model
    mp = g._resolve_matmul_precision(dc)
    return mp, False, mp


def measure_pipeline(opt, model, timed_steps, tuned_configs=None):
    import torch
    g = _g()
    data, init, n_params = _data_init(model)
    dev = torch.device("cuda")
    # [A6-HIGH] per-pipeline precision: prefer the tuner's winning precision for
    # this (opt, model); fall back to the auto-resolved default. Applied to EVERY
    # run config below so wall/AI measure the tuned precision, and the ceiling
    # uses that precision's peak.
    if tuned_configs is None:
        tuned_configs = _load_tuned_configs(model)
    mp_str, use_amp, prec_label = _resolve_precision(opt, model, tuned_configs)
    def _cfgp(steps, ee):
        c = _cfg(opt, model, steps, ee)
        c["matmul_precision"] = mp_str; c["use_amp"] = use_amp
        return c
    fn = getattr(g, {
        "adamw": "train_adamw", "neuralgrok": "train_neuralgrok",
        "grokadamw": "train_grokadamw", "supergrok": "train_supergrok",
        "supergrok15": "train_supergrok15", "supergrok2": "train_supergrok2",
        "grokfast": "train_grokfast", "muon": "train_muon", "lion": "train_lion",
        "looksam": "train_looksam", "prodigy": "train_prodigy"}[opt])

    # ── pass 1: pure-train wall (eval disabled). One DISCARD run first absorbs
    # process-global one-time costs (cuBLAS/cuDNN handles, kernel ABI caches),
    # then two lengths are timed and differenced — per-run setup (model build +
    # state-dict load) cancels in the diff, leaving pure marginal step cost.
    c = _cfgp(25, 10**9)
    torch.manual_seed(42)
    fn(c, init, *data, dev, bp=0)          # discard (warmup)
    torch.cuda.synchronize()
    walls = []
    for steps in (15, 15 + timed_steps):
        c = _cfgp(steps, 10**9)
        torch.manual_seed(42); torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(c, init, *data, dev, bp=0)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
    wall_per_step = (walls[1] - walls[0]) / timed_steps

    # ── pass 2: per-10-step series via the race's own eval hook (the owner's
    # tracking cadence; includes eval overhead, labelled as such).
    series = []
    last = [None]
    def cb(step, ta, va, tea):
        now = time.perf_counter()
        if last[0] is not None:
            series.append({"step": step, "wall_per_step_incl_eval":
                           (now - last[0]) / 10.0})
        last[0] = now
    c = _cfgp(60, 10)
    c["_eval_callback"] = cb
    torch.manual_seed(42)
    fn(c, init, *data, dev, bp=0)

    # ── pass 3: FLOPs/step from torch.profiler (GEMM-registered)
    from torch.profiler import profile, ProfilerActivity
    c = _cfgp(20, 10**9)
    # [A6-MED] disable gradient checkpointing for the FLOP count ONLY: with
    # checkpointing on, every block's forward GEMMs run TWICE (once in fwd, once
    # recomputed in bwd), so with_flops would DOUBLE-COUNT the forward GEMM FLOPs
    # and overstate FLOPs/step (and thus achieved FLOP/s). The model builders
    # honor this key (grokking_race_v2._raw_model: gc = c.get("grad_checkpoint",
    # True)). Wall/AI passes keep the race's real (checkpointed) setting.
    c["grad_checkpoint"] = False
    torch.manual_seed(42)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 with_flops=True) as prof:
        fn(c, init, *data, dev, bp=0)
    flops_total = sum(getattr(e, "flops", 0) or 0 for e in prof.key_averages())
    flops_per_step = flops_total / 20.0

    dc = _cfgp(10, 10**9)
    batch = data[0].shape[0]
    # [A6-HIGH] per-model sequence length — the decoder build HARDCODES seq=4
    # (grokking_race_v2._raw_model: Transformer(..., 4, ...)), so the old
    # dc["seq_len"] (=8, the mamba chain length) over-counted decoder activation
    # traffic 2×. vit uses num_patches+CLS handled in-model; the activation
    # model here counts the patch tokens (num_patches=16). mamba's true length
    # is 2*chain_length+2 (=8 at chain_length=3).
    seq = {
        "decoder": 4,
        "vit": dc.get("num_patches", 16),
        "mamba": 2 * dc.get("chain_length", 3) + 2,
    }[model]
    bps = bytes_per_step(opt, model, n_params, batch, seq,
                         dc.get("dim_model", 128), dc.get("num_layers", 2))

    achieved = flops_per_step / wall_per_step
    ai = flops_per_step / bps
    # [A6-HIGH] Compute ceiling follows THIS pipeline's precision (tuner's winner
    # for the cell, else the model's auto default): bf16/fp16amp -> BF16 TC peak;
    # tf32 -> TF32 TC; fp32 -> CUDA core; fp8/fp8e5m2/int8 -> low-precision TC.
    peak_c = _PRECISION_PEAK.get(prec_label, PEAKS["bf16_tc"])
    ceiling = min(peak_c, ai * PEAKS["hbm_bw"])
    bound = "compute" if peak_c < ai * PEAKS["hbm_bw"] else "memory"
    return {
        "optimizer": opt, "model": model,
        "precision": prec_label,          # [A6-HIGH] per-pipeline precision label
        "use_amp": bool(use_amp),
        "steps_per_s": 1.0 / wall_per_step,
        "wall_per_step_ms": wall_per_step * 1e3,
        "flops_per_step": flops_per_step,
        "bytes_per_step_analytical": bps,
        "achieved_flops_per_s": achieved,
        "arithmetic_intensity": ai,
        "matmul_peak_used": peak_c,
        "roofline_ceiling_flops_per_s": ceiling,
        "fraction_of_roofline": achieved / ceiling,
        "bound_regime_at_AI": bound,
        "achieved_bytes_per_s_analytical": bps / wall_per_step,
        "fraction_of_hbm_bw": (bps / wall_per_step) / PEAKS["hbm_bw"],
        "per10step_series_incl_eval": series,
    }


def make_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    # ── classic roofline (log-log)
    ais = np.logspace(-2, 4, 256)
    # [A6-HIGH] draw every precision ceiling (incl. FP8/INT8 low-precision TC) so
    # a point tuned to fp8 is read against its own roof.
    for key, label, ls in [("fp32_cuda", "FP32 CUDA-core peak", "--"),
                           ("tf32_tc", "TF32 tensor-core peak", "-."),
                           ("bf16_tc", "BF16/FP16 tensor-core peak", ":"),
                           ("fp8_tc", "FP8/INT8 tensor-core peak", (0, (1, 1)))]:
        if PEAKS.get(key) is None:  # fp8/int8 peaks retired with the precision program
            continue
        # linestyle= keyword required: tuple dash specs (0,(1,1)) passed
        # positionally are parsed as a data series and crash matplotlib.
        ax.plot(ais, np.minimum(PEAKS[key], ais * PEAKS["hbm_bw"]),
                linestyle=ls, lw=1.2, label=label)
    colors = {"decoder": "#1f77b4", "vit": "#2ca02c", "mamba": "#d62728"}
    for r in rows:
        ax.scatter(r["arithmetic_intensity"], r["achieved_flops_per_s"],
                   s=42, color=colors[r["model"]], zorder=5)
        # [A6-HIGH] label each point with its precision (the ceiling it's read against)
        ax.annotate(f'{r["optimizer"]} ({r.get("precision","?")})',
                    (r["arithmetic_intensity"], r["achieved_flops_per_s"]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    for mdl, col in colors.items():
        ax.scatter([], [], color=col, label=mdl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte, analytical)")
    ax.set_ylabel("Achieved FLOP/s (measured)")
    ax.set_title("H100 roofline — race pipelines (eager model + fused optimizer)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="upper left")

    # ── % of attainable roofline, sorted
    rows_s = sorted(rows, key=lambda r: r["fraction_of_roofline"])
    # [A6-HIGH] label each bar with the precision its ceiling came from
    names = [f'{r["optimizer"]}/{r["model"]} [{r.get("precision","?")}]' for r in rows_s]
    fracs = [100 * r["fraction_of_roofline"] for r in rows_s]
    cols = [colors[r["model"]] for r in rows_s]
    ax2.barh(range(len(rows_s)), fracs, color=cols)
    ax2.set_yticks(range(len(rows_s))); ax2.set_yticklabels(names, fontsize=7)
    ax2.set_xlabel("% of attainable roofline ceiling  min(compute, AI·BW)")
    ax2.set_title("Distance to roofline per pipeline")
    for i, f in enumerate(fracs):
        ax2.text(f, i, f" {f:.2f}%", va="center", fontsize=6.5)
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="decoder,vit,mamba")
    ap.add_argument("--opts", default=",".join(OPTS))
    ap.add_argument("--steps", type=int, default=100)
    # [A6-HIGH] optional override for the tuned-configs file; default per model is
    # results/tuning/tuned_configs_{model}.json (resolved in _load_tuned_configs).
    ap.add_argument("--tuned-configs", default=None,
                    help="path to a tuned_configs JSON; overrides the per-model default")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    opts = [o for o in args.opts.split(",") if o]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        tuned = _load_tuned_configs(model, args.tuned_configs)
        for opt in opts:
            try:
                r = measure_pipeline(opt, model, args.steps, tuned_configs=tuned)
                rows.append(r)
                print(f"  {opt:11s}/{model:7s} [{r['precision']:7s}] {r['steps_per_s']:7.1f} steps/s  "
                      f"{r['flops_per_step']/1e9:7.2f} GF/step  "
                      f"achieved={r['achieved_flops_per_s']/1e12:6.3f} TF/s  "
                      f"AI={r['arithmetic_intensity']:6.1f}  "
                      f"roof%={100*r['fraction_of_roofline']:6.2f}  "
                      f"({r['bound_regime_at_AI']}-bound at this AI)", flush=True)
            except Exception as e:
                import traceback
                print(f"  {opt}/{model} FAILED: {e}", flush=True)
                traceback.print_exc()
    (OUT / "roofline.json").write_text(json.dumps(
        {"peaks": PEAKS, "rows": rows}, indent=2))
    print(f"wrote {OUT/'roofline.json'}", flush=True)
    make_plot(rows, OUT / "roofline.png")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
