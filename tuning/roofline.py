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

# [R2.4 item 2] BATCH-SATURATION SWEEP (decoder, min-of-3 wall, quiet GPU). The
# operating-point evidence: the one-CTA-per-SM TC megakernel is occupancy-pinned, so
# its throughput (samples/s = achieved FLOP/s, FLOPs linear in B) saturates at B≈2k
# (one 16-row dW atom × 132 SMs ≈ 2112) and DECLINES past 16k — batch does NOT buy
# roofline fraction for it. Eager (cuBLAS batched GEMMs) saturates much later (≈32-65k)
# because it is not occupancy-capped. B=16384 is the chosen shared point (owner's
# "B≈16k+" floor; megakernel within 3.5% of its peak there; eager at its knee). This
# is the honest negative result (no-suppression): the path to higher megakernel
# fraction is multi-CTA-per-tensor tiling, NOT a larger batch. Tables stored in the
# run JSON as `batch_saturation_sweep`.
BATCH_SATURATION_SWEEP = {
    "note": ("decoder, min-of-3 wall, quiet GPU (tuner fleet SIGSTOPped). "
             "throughput = samples/s ∝ achieved FLOP/s. megakernel = TC "
             "tc_train_step (one CTA/SM); eager = lion fwd+bwd+step bf16-autocast."),
    "megakernel_tc": [
        {"B": 1024,   "wall_ms": 8.820,    "throughput_samples_per_s": 116098, "peak_vram_gb": 0.250},
        {"B": 2048,   "wall_ms": 14.374,   "throughput_samples_per_s": 142477, "peak_vram_gb": 0.524},
        {"B": 4096,   "wall_ms": 28.714,   "throughput_samples_per_s": 142646, "peak_vram_gb": 0.630},
        {"B": 8192,   "wall_ms": 58.173,   "throughput_samples_per_s": 140821, "peak_vram_gb": 0.840},
        {"B": 16384,  "wall_ms": 119.077,  "throughput_samples_per_s": 137591, "peak_vram_gb": 1.260},
        {"B": 32768,  "wall_ms": 255.077,  "throughput_samples_per_s": 128463, "peak_vram_gb": 2.103},
        {"B": 65536,  "wall_ms": 505.165,  "throughput_samples_per_s": 129732, "peak_vram_gb": 3.787},
        {"B": 131072, "wall_ms": 1022.129, "throughput_samples_per_s": 128234, "peak_vram_gb": 7.154},
    ],
    "eager_lion": [
        {"B": 4096,   "wall_ms": 6.472,   "throughput_samples_per_s": 632904,  "peak_vram_gb": 0.220},
        {"B": 8192,   "wall_ms": 14.245,  "throughput_samples_per_s": 575065,  "peak_vram_gb": 0.368},
        {"B": 16384,  "wall_ms": 14.353,  "throughput_samples_per_s": 1141497, "peak_vram_gb": 0.583},
        {"B": 32768,  "wall_ms": 19.007,  "throughput_samples_per_s": 1724025, "peak_vram_gb": 1.096},
        {"B": 65536,  "wall_ms": 35.794,  "throughput_samples_per_s": 1830900, "peak_vram_gb": 2.120},
        {"B": 131072, "wall_ms": 69.709,  "throughput_samples_per_s": 1880273, "peak_vram_gb": 4.170},
        {"B": 262144, "wall_ms": 137.169, "throughput_samples_per_s": 1911100, "peak_vram_gb": 8.269},
    ],
    "verdict": ("megakernel saturates ~8× below the owner's 16k estimate (occupancy "
                "pinned at 1 CTA/SM); eager saturates ~32-65k. VRAM is NOT the binding "
                "constraint at d=128 (literal VRAM-max ≈ B~1-2M yields no fraction gain). "
                "16384 = fair shared point + megakernel ceiling."),
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


def _tile_train_data(data, B):
    """DETERMINISTIC repeat-tile the TRAIN split (tx, ty) of a race data tuple up
    to exactly B rows (owner directive R2.4 item 2: "tile data if dataset<max;
    full-batch repeated/tiled deterministically"). The race train functions consume
    the WHOLE (tx, ty) every step (full-batch grokking), so a larger tx/ty IS a
    larger train batch — this is the only surface needed to drive the megakernel /
    eager paths at B without touching the train loop.

    Tiling is index-only (``idx = arange(B) % n_train``) — NO RNG, NO new samples:
    sample i appears ⌈B/n⌉ times, byte-identical across runs, so the wall, FLOPs/step
    and peak VRAM are reproducible. If B <= n_train we truncate (never upsample past
    the requested B). The eval/val/test splits (vax..tey) pass through UNCHANGED —
    the roofline disables eval (eval_every=1e9) so they are never touched; keeping
    them intact preserves the train-fn signature.

    data = (tx, ty, vax, vay, tex, tey). Returns the same tuple with tx, ty tiled.
    """
    import torch
    tx, ty = data[0], data[1]
    n = tx.shape[0]
    if B is None or B == n:
        return data
    idx = torch.arange(B, device=tx.device) % n     # deterministic, RNG-free
    return (tx[idx], ty[idx]) + tuple(data[2:])


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


def measure_pipeline(opt, model, timed_steps, tuned_configs=None,
                     force_precision=None, max_batch=None):
    import torch
    g = _g()
    data, init, n_params = _data_init(model)
    # [R2.4-MAXBATCH item 2] train every cell at the per-model max batch (same B
    # within a model, deterministic repeat-tile of the native train split). B is
    # chosen at the SM-saturation operating point (16384 by default — one 128-row
    # tile per 132 SMs; the megakernel's one-CTA-per-SM throughput is already within
    # ~3.5% of its 4k-batch peak there, and eager is at its utilization knee). The
    # wall, FLOPs/step and peak VRAM below are then all measured AT this batch.
    if max_batch is not None:
        data = _tile_train_data(data, max_batch)
    dev = torch.device("cuda")
    # [A6-HIGH] per-pipeline precision: prefer the tuner's winning precision for
    # this (opt, model); fall back to the auto-resolved default. Applied to EVERY
    # run config below so wall/AI measure the tuned precision, and the ceiling
    # uses that precision's peak.
    if tuned_configs is None:
        tuned_configs = _load_tuned_configs(model)
    mp_str, use_amp, prec_label = _resolve_precision(opt, model, tuned_configs)
    # [ROOFLINE-FP32] OWNER OVERRIDE: when force_precision is set (e.g. "fp32"),
    # pin EVERY run config to it, overriding the tuner/auto resolution. This is the
    # whole point of the re-measure: the decoder auto-default is bf16, which makes
    # _try_fused_train_step DECLINE the L3 persistent megakernel (precision gate in
    # grokking_race_v2: L3 fires only when resolved precision == fp32). Forcing fp32
    # makes decoder/vit/mamba × adamw run the TRUE L3 real fwd+bwd+opt megakernel
    # (validated: the run banner prints "L3-REAL fused-train path ENABLED" and the
    # fused_train_counter advances every step). The other 8 optimizers run
    # eager-fp32 + the L1 fused optimizer tail (L1 has no precision gate). Every row
    # is then read against the fp32 CUDA-core ceiling (66.9 TF) — the uniform basis
    # the old bf16 chart lacked.
    if force_precision is not None:
        mp_str, use_amp, prec_label = force_precision, False, force_precision

    # [ROOFLINE-PATH] Verify which path ACTUALLY executes (owner: "label paths
    # explicitly"; don't infer from gate logic). We wrap the dispatch entry points
    # to count real L3 / L1 firings for THIS pipeline, and read the process-global
    # _FUSED_ABI_STALE flag — if it tripped, L3 silently degraded to eager and the
    # "L3 measurement" would secretly be eager-fp32 (wrong thing measured).
    _counters = {"l3": 0, "l1": 0}
    # [ROOFLINE-ENGINE] capture the ACTUAL GEMM engine the L3 path ran (task 2):
    # decoder/vit bf16 → "wgmma" (TC), mamba bf16 + any fp32 → "scalar". The engine
    # — NOT the requested precision — drives the row's path label, precision label,
    # AND compute ceiling (a scalar engine computes fp32 → 66.9 TF peak even when
    # the config precision string is "bf16"; without this, a scalar-executing cell's
    # achieved-fp32 ÷ bf16-ceiling craters its fraction ~15× silently).
    _last_engine = {"engine": None}
    _orig_l3 = g._try_fused_train_step
    _orig_l1 = g._try_fused_step
    def _wrap_l3(*a, **k):
        r = _orig_l3(*a, **k)
        if r is not None:
            _counters["l3"] += 1
            if g.LAST_L3_ENGINE is not None:
                _last_engine["engine"] = g.LAST_L3_ENGINE["engine"]
        return r
    def _wrap_l1(*a, **k):
        r = _orig_l1(*a, **k)
        if r:
            _counters["l1"] += 1
        return r

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
    # [ROOFLINE-PATH] the path counters wrap the dispatch entries for the WHOLE
    # wall pass so we record what actually ran (L3 / L1 / eager), not what the gate
    # would permit. use_fused stays True + the forced precision is in effect, so the
    # REAL megakernel is what is timed.
    g._try_fused_train_step = _wrap_l3
    g._try_fused_step = _wrap_l1
    # [R2.4-PEAKVRAM item 2] capture PEAK VRAM of the PRODUCTION path only. This MUST
    # be measured during the wall pass (use_fused=True → the real megakernel/L1 path),
    # NOT across the whole function: pass 3 (FLOP count) runs use_fused=False → EAGER
    # for every cell, and eager allocates the full activation graph while the
    # megakernel keeps activations CTA-local. A whole-function max_memory_allocated
    # would report the eager-flop-pass VRAM for the 3 adamw megakernel cells —
    # misrepresenting exactly the production paths this table exists to characterise.
    # Reset before the warmup so the reading is steady-state (warmup transients
    # included is fine — they ARE part of the path's footprint), capture right after
    # the timed walls and BEFORE pass 2/3.
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    peak_vram_bytes = None
    try:
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
        peak_vram_bytes = int(torch.cuda.max_memory_allocated())  # production-path peak
    finally:
        g._try_fused_train_step = _orig_l3
        g._try_fused_step = _orig_l1
    wall_per_step = (walls[1] - walls[0]) / timed_steps
    # Steps that fired L3/L1 across the two timed runs (15 + (15+timed)).
    _wall_total_steps = 15 + (15 + timed_steps)
    l3_fired = _counters["l3"]; l1_fired = _counters["l1"]
    l3_engine = _last_engine["engine"]   # "wgmma" | "scalar" | None (task 2)
    if g._FUSED_ABI_STALE:           # the stale-ABI guard tripped → L3 degraded
        actual_path = "eager+L1(ABI-stale)" if l1_fired else "eager(ABI-stale)"
    elif l3_fired > 0:
        # Distinguish the directly-wired TC path from the scalar megakernel (the
        # owner directive's path labels): L3-TC bf16 vs L3-scalar fp32.
        if l3_engine == "wgmma":
            actual_path = "L3-TC-megakernel(wgmma)"
        elif l3_engine == "scalar":
            actual_path = "L3-scalar-megakernel"
        else:
            actual_path = "L3-real-megakernel"  # engine unknown (should not happen)
    elif l1_fired > 0:
        actual_path = "eager+L1-fused-tail"
    else:
        actual_path = "eager"
    # ENGINE-DRIVEN precision + ceiling (task 2): when an L3 engine actually ran, the
    # row's precision label and compute peak follow the ENGINE, not the requested
    # mp_str — wgmma computes bf16 (989 TF peak), scalar computes fp32 (66.9 TF). For
    # eager/L1 rows keep the requested-precision label/peak (those DO run at mp_str).
    eng_prec_label = prec_label
    if l3_engine == "wgmma":
        eng_prec_label = "bf16"
    elif l3_engine == "scalar":
        eng_prec_label = "fp32"

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
    # [ROOFLINE-FLOP-FIX] CRITICAL: disable the fused megakernel for the FLOP
    # count ONLY. torch.profiler(with_flops=True) counts ONLY registered aten
    # GEMMs/convs; a fused custom megakernel is ONE opaque kernel that registers
    # ZERO FLOPs. With use_fused=True on the L3 cells (decoder/vit/mamba × adamw
    # under forced fp32) the profiler would see ~0 FLOPs/step and crater achieved
    # FLOP/s for exactly the cells we are trying to measure. EMPIRICALLY VERIFIED:
    # decoder×adamw fp32 profiled 2.23e9 FLOPs/step with use_fused=True vs
    # 4.23e10 with use_fused=False (~19×). FLOPs/step is dtype- AND path-
    # independent (the math is identical), so the eager count is the CORRECT
    # FLOPs/step for the L3 megakernel; achieved = (eager-counted FLOPs)/(L3 wall)
    # from pass 1 is valid. Non-fused cells are unaffected (same path either way).
    c["use_fused"] = False
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
    # [A6-HIGH / task 2] Compute ceiling follows the ACTUAL engine when an L3 cell
    # ran (eng_prec_label: wgmma→bf16 989TF, scalar→fp32 66.9TF), else the requested
    # precision for eager/L1 rows. Reading a scalar-fp32 cell against the bf16 roof
    # would understate its fraction ~15×; this keeps the ceiling honest to what ran.
    peak_c = _PRECISION_PEAK.get(eng_prec_label, PEAKS["bf16_tc"])
    ceiling = min(peak_c, ai * PEAKS["hbm_bw"])
    bound = "compute" if peak_c < ai * PEAKS["hbm_bw"] else "memory"
    # path_family: "tc" for the wired bf16 tensor-core megakernel, else "scalar"
    # (covers eager/L1 race rows AND the L3-scalar megakernel rows like mamba).
    _path_family = "tc" if l3_engine == "wgmma" else "scalar"
    return {
        "optimizer": opt, "model": model,
        "precision": eng_prec_label,      # task 2: label = ACTUAL engine precision
        "precision_requested": prec_label,  # what the config/tuner asked for
        "l3_engine": l3_engine,           # "wgmma" | "scalar" | None (task 2)
        "use_amp": bool(use_amp),
        # [ROOFLINE-PATH] what ACTUALLY executed during the timed wall pass (owner:
        # label paths explicitly). family distinguishes the scalar-race rows from
        # the directly-measured bf16-TC cells.
        "path": actual_path,
        "path_family": _path_family,      # "tc" (wgmma) | "scalar" (everything else)
        "l3_steps_fired": l3_fired,
        "l1_steps_fired": l1_fired,
        "wall_pass_total_steps": _wall_total_steps,
        "abi_stale": bool(g._FUSED_ABI_STALE),
        "steps_per_s": 1.0 / wall_per_step,
        "wall_per_step_ms": wall_per_step * 1e3,
        "flops_per_step": flops_per_step,
        "batch": int(batch),              # the ACTUAL train batch this row ran at
        # [R2.4 item 2] PEAK VRAM (torch.cuda.max_memory_allocated) of the PRODUCTION
        # path, measured during the wall pass (use_fused=True), NOT the eager flop
        # pass. None only if the wall pass raised before capture.
        "peak_vram_bytes": peak_vram_bytes,
        "max_batch_requested": (int(max_batch) if max_batch is not None else None),
        "seq_for_acts": int(seq), "dim_for_acts": int(dc.get("dim_model", 128)),
        "layers_for_acts": int(dc.get("num_layers", 2)),
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


# ── TASK 2+3: directly-measured TC cells (mega_*_real_adamw_tc.cu) ──────────
# Each TC TU exposes BOTH tc_train_step (bf16 wgmma fwd+bwd+AdamW persistent
# megakernel) AND scalar_train_step (the fp32 persistent megakernel — the same
# kernel family the race's L3-real path runs). We JIT-load each TU exactly like
# its test (tests/hw/test_{decoder,vit,mamba}_tc.py::_build_tc_module) and time
# both paths at the SAME full batch (truncated to B%16==0, which the decoder TC
# TORCH_CHECK requires; the dW K-loop is 16-step atoms). ncta_cap=0 → one CTA/SM
# = the shipped saturating config (NOT the gate's capped witness).
_TC_TU_REL = {
    "decoder": "csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu",
    "vit":     "csrc/fused/sm_90/mega_vit_real_adamw_tc.cu",
    "mamba":   "csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu",
}
_TC_MOD_CACHE = {}


def _build_tc_module(model, verbose=False):
    """JIT-build the model's TC TU (mirrors the test _build_tc_module: 9.0a,
    -O3 -std=c++17 --expt-relaxed-constexpr, sm_90a gencode). Cached per model.
    Raises on build failure (the caller skips TC rows per the owner's "skip if
    JIT-path still blocked")."""
    import os
    if model in _TC_MOD_CACHE:
        return _TC_MOD_CACHE[model]
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    from torch.utils.cpp_extension import load
    tu = str(ROOT / _TC_TU_REL[model])
    mod = load(
        name=f"mega_{model}_real_adamw_tc",
        sources=[tu], extra_include_paths=[str(ROOT)],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--expt-relaxed-constexpr",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-gencode=arch=compute_90a,code=compute_90a", "-lineinfo"],
        extra_cflags=["-O3", "-std=c++17"], verbose=verbose)
    _TC_MOD_CACHE[model] = mod
    return mod


def _tc_inputs(model, B, dev):
    """Per-model (params, input, targets) for one TC/scalar step at batch B. The
    flat params are random of the TC TU's TOTAL size — step TIME is independent of
    param VALUES (correctness is already validated by the cells' gates: decoder
    13/13+5, vit 21/21, mamba 5/5), so we don't need the exact named layout here.
    decoder/mamba take int32 tokens [B,seq]; vit takes float32 patches [B,16,49]."""
    import torch
    mod = _build_tc_module(model)
    total = int(mod.TOTAL)
    gpar = torch.Generator(device=dev).manual_seed(42)
    params = (torch.randn(total, generator=gpar, device=dev) * 0.02).contiguous()
    gin = torch.Generator(device=dev).manual_seed(7)
    if model == "vit":
        from tests.hw.vit_oracle import NUM_PATCHES, PATCH_DIM, VOCAB
        inp = torch.randn(B, NUM_PATCHES, PATCH_DIM, generator=gin, device=dev).to(torch.float32)
        tgt = torch.randint(0, VOCAB, (B,), generator=gin, device=dev).to(torch.int32)
    elif model == "mamba":
        from tests.hw.mamba_oracle import VOCAB, P_HEAD, SEQ
        inp = torch.randint(0, VOCAB, (B, SEQ), generator=gin, device=dev).to(torch.int32)
        tgt = torch.randint(0, P_HEAD, (B,), generator=gin, device=dev).to(torch.int32)
    else:  # decoder
        from tests.hw.decoder_oracle import VOCAB, SEQ
        inp = torch.randint(0, VOCAB, (B, SEQ), generator=gin, device=dev).to(torch.int32)
        tgt = torch.randint(0, VOCAB, (B,), generator=gin, device=dev).to(torch.int32)
    return mod, params, inp, tgt, total


def _time_tc_kernel(call, warmup=8, iters=40):
    import torch
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def measure_tc_cell(model, eager_flops_full, eager_B_full, n_params,
                    seq, dim, layers):
    """Measure the directly-loaded TC TU for `model` (Task 2) AND its scalar
    sibling (Task 3) at the same B%16 full batch, ncta_cap=0 (full saturation).

    eager_flops_full / eager_B_full: FLOPs/step and batch from the eager adamw
    profile (pass 3 of measure_pipeline, use_fused=False — the path-independent
    GEMM count). FLOPs scale linearly in B, so flops_at(B) = eager_flops_full ·
    B / eager_B_full. Returns a list of two rows (tc, scalar) or [] if the JIT
    build is blocked.
    """
    import torch
    dev = torch.device("cuda")
    B = eager_B_full - (eager_B_full % 16)          # decoder TC requires B%16==0
    try:
        mod, params, inp, tgt, total = _tc_inputs(model, B, dev)
    except Exception as e:  # noqa: BLE001 — JIT blocked → owner says skip TC rows
        print(f"  TC[{model}] BUILD/SETUP FAILED (skipping TC rows): {e}", flush=True)
        return []
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    flops_per_step = eager_flops_full * (B / float(eager_B_full))
    beta1, beta2 = 0.9, 0.98
    bc1, bc2 = 1.0 - beta1, 1.0 - beta2
    lr, eps, wd = 1e-3, 1e-8, 0.0
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)

    def _tc_call():
        return mod.tc_train_step(params.clone(), inp, tgt, state, lr, beta1, beta2,
                                 eps, wd, bc1, bc2, 1, 0)
    def _sc_call():
        return mod.scalar_train_step(params.clone(), inp, tgt, state, lr, beta1,
                                     beta2, eps, wd, bc1, bc2, 1)

    rows = []
    for label, family, prec, call, has in (
        ("tc_train_step",     "tc",     "bf16-TC", _tc_call, True),
        ("scalar_train_step", "scalar-mega", "fp32", _sc_call,
         hasattr(mod, "scalar_train_step")),
    ):
        if not has:
            continue
        try:
            wall = _time_tc_kernel(call)
        except Exception as e:  # noqa: BLE001
            print(f"  TC[{model}] {label} FAILED: {e}", flush=True)
            continue
        # Bytes/AI: keep the SAME analytical traffic model as the scalar rows for
        # cross-row comparability (adamw S=2), BUT the TC kernel stores activations
        # in bf16 (2B not 4B). The activation term dominates at these B, so an
        # all-fp32 byte model understates TC AI ~2×. We report BOTH: the fp32-model
        # AI (comparable to every other row) and a TC-faithful AI (activations 2B).
        # CAVEAT (load-bearing): the TC point lands MEMORY-bound in this analytical
        # model (AI·BW < bf16 peak), so the ceiling — and thus the fraction —
        # depends on this byte assumption. The achieved TF/s (the honest axis) does
        # NOT. The writeup leads with achieved TF/s and reports BOTH AIs for TC.
        bps_fp32 = bytes_per_step("adamw", model, n_params, B, seq, dim, layers)
        # TC-faithful: halve the activation byte term (acts stored bf16).
        acts = B * seq * dim * layers * (10 if model == "mamba" else 14)
        bps_tc = bps_fp32 - 2 * 4.0 * acts + 2 * 2.0 * acts  # acts 4B→2B
        bps = bps_tc if family == "tc" else bps_fp32
        achieved = flops_per_step / wall
        ai = flops_per_step / bps
        ai_fp32model = flops_per_step / bps_fp32
        peak_c = PEAKS["bf16_tc"] if family == "tc" else PEAKS["fp32_cuda"]
        ceiling = min(peak_c, ai * PEAKS["hbm_bw"])
        bound = "compute" if peak_c < ai * PEAKS["hbm_bw"] else "memory"
        rows.append({
            "optimizer": "adamw", "model": model,
            "precision": prec, "use_amp": False,
            "path": f"L3-real-megakernel({label})", "path_family": family,
            "tc_batch": B, "ncta_cap": 0,
            "steps_per_s": 1.0 / wall, "wall_per_step_ms": wall * 1e3,
            "flops_per_step": flops_per_step,
            "flops_basis": "eager-profiled (use_fused=False) scaled by B/B_full",
            "bytes_per_step_analytical": bps,
            "bytes_model": ("tc-faithful (acts bf16 2B)" if family == "tc"
                            else "fp32 (all 4B)"),
            "achieved_flops_per_s": achieved,
            "arithmetic_intensity": ai,
            "arithmetic_intensity_fp32model": ai_fp32model,
            "matmul_peak_used": peak_c,
            "roofline_ceiling_flops_per_s": ceiling,
            "fraction_of_roofline": achieved / ceiling,
            "bound_regime_at_AI": bound,
            "achieved_bytes_per_s_analytical": bps / wall,
            "fraction_of_hbm_bw": (bps / wall) / PEAKS["hbm_bw"],
            "per10step_series_incl_eval": [],
        })
    if len(rows) >= 2:
        tc_r = next((r for r in rows if r["path_family"] == "tc"), None)
        sc_r = next((r for r in rows if r["path_family"] == "scalar-mega"), None)
        if tc_r and sc_r:
            ratio = sc_r["wall_per_step_ms"] / tc_r["wall_per_step_ms"]
            print(f"  TC[{model}] B={B} ncta=0  TC={tc_r['wall_per_step_ms']:.3f}ms "
                  f"scalar={sc_r['wall_per_step_ms']:.3f}ms  speedup(scalar/TC)="
                  f"{ratio:.2f}×  ({'TC wins' if ratio > 1 else 'scalar wins'})",
                  flush=True)
    return rows


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
    # [ROOFLINE-PATH] marker shape encodes the measured path FAMILY so the two
    # paths the owner asked for are visually separable: scalar race rows (o),
    # the directly-measured fp32 scalar megakernel (s), and the bf16-TC megakernel
    # (^, hollow). L3-real scalar race rows get a heavier edge so the adamw cells
    # that now fire the real persistent megakernel stand out from eager rows.
    def _marker(r):
        fam = r.get("path_family", "scalar")
        if fam == "tc":
            return dict(marker="^", facecolor="none", edgecolor=colors[r["model"]],
                        linewidths=1.6, s=70)
        if fam == "scalar-mega":
            return dict(marker="s", color=colors[r["model"]], s=46)
        is_l3 = str(r.get("path", "")).startswith("L3-real")
        return dict(marker="o", color=colors[r["model"]], s=46,
                    edgecolor="black" if is_l3 else "none",
                    linewidths=1.2 if is_l3 else 0)
    for r in rows:
        ax.scatter(r["arithmetic_intensity"], r["achieved_flops_per_s"],
                   zorder=5, **_marker(r))
        # [A6-HIGH] label each point with its precision (the ceiling it's read against)
        ax.annotate(f'{r["optimizer"]} ({r.get("precision","?")})',
                    (r["arithmetic_intensity"], r["achieved_flops_per_s"]),
                    fontsize=6.0, xytext=(3, 3), textcoords="offset points")
    for mdl, col in colors.items():
        ax.scatter([], [], color=col, label=mdl)
    # path-family legend handles
    ax.scatter([], [], marker="o", color="gray", edgecolor="black", linewidths=1.2,
               label="scalar race (L3-real fp32 = black edge)")
    ax.scatter([], [], marker="s", color="gray", label="scalar megakernel (fp32, direct)")
    ax.scatter([], [], marker="^", facecolor="none", edgecolor="gray", linewidths=1.6,
               label="TC megakernel (bf16, direct)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte, analytical)")
    ax.set_ylabel("Achieved FLOP/s (measured)")
    ax.set_title("H100 roofline — REAL megakernel paths "
                 "(scalar fp32 L3 + bf16-TC), fp32-forced")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, loc="upper left")

    # ── % of attainable roofline, sorted
    rows_s = sorted(rows, key=lambda r: r["fraction_of_roofline"])
    # [A6-HIGH] label each bar with the precision its ceiling came from + path tag
    def _tag(r):
        fam = r.get("path_family", "scalar")
        if fam == "tc":
            return "TC"
        if fam == "scalar-mega":
            return "mega"
        return "L3" if str(r.get("path", "")).startswith("L3-real") else "eag"
    names = [f'{r["optimizer"]}/{r["model"]} [{r.get("precision","?")}·{_tag(r)}]'
             for r in rows_s]
    fracs = [100 * r["fraction_of_roofline"] for r in rows_s]
    cols = [colors[r["model"]] for r in rows_s]
    # TC bars hatched so they read distinctly from the scalar bars
    hatches = ["///" if r.get("path_family") == "tc" else None for r in rows_s]
    bars = ax2.barh(range(len(rows_s)), fracs, color=cols)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    ax2.set_yticks(range(len(rows_s))); ax2.set_yticklabels(names, fontsize=7)
    ax2.set_xlabel("% of attainable roofline ceiling  min(compute, AI·BW)")
    ax2.set_title("Distance to roofline per pipeline")
    for i, f in enumerate(fracs):
        ax2.text(f, i, f" {f:.2f}%", va="center", fontsize=6.5)
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"wrote {path}", flush=True)


def _delta_vs_baseline(rows, baseline_path):
    """[ROOFLINE-DELTA] Diff the new scalar rows against the archived eager
    baseline (same (optimizer, model) key, path_family scalar). Reports the change
    in roofline-fraction / achieved-TF/s / precision/path so the owner can see what
    moved when the L3 megakernel actually fired."""
    try:
        base = json.loads(Path(baseline_path).read_text())
    except Exception:  # noqa: BLE001
        return []
    bmap = {(r["optimizer"], r["model"]): r
            for r in base.get("rows", []) if r.get("path_family", "scalar") == "scalar"}
    out = []
    for r in rows:
        if r.get("path_family", "scalar") != "scalar":
            continue                       # deltas only defined for the scalar race rows
        b = bmap.get((r["optimizer"], r["model"]))
        if not b:
            continue
        # [ROOFLINE-DELTA] Decompose WHY the fraction moved (advisor): a fraction
        # jump can come from (a) a real PATH change (adamw: eager+L1 → L3 real
        # megakernel) or (b) a pure CEILING relabel (the other opts only changed
        # precision bf16→fp32, dropping the ceiling 285→66.9 TF, so the same
        # achieved TF/s reads as a higher fraction — NOT more efficient). We record
        # the OLD path + a path_changed flag + the ceiling ratio so the writeup
        # never implies fp32 made 10 optimizers faster (it did not).
        # The eager baseline predates path labelling AND the extension build fix
        # (per the commit timeline the fused path was broken then → every baseline
        # row ran eager / eager+L1). So "did the PATH genuinely change" = did the
        # NEW row fire the L3-real megakernel (the only path this fp32-force newly
        # enables). The non-adamw rows ran eager+L1 in BOTH (ceiling-relabel only).
        path_old = b.get("path") or "eager (unlabelled pre-fix)"
        # A real L3 megakernel fired iff the path starts "L3-" (now "L3-TC-..."/
        # "L3-scalar-..." after the bf16/wgmma wiring; was "L3-real-..." pre-wiring).
        path_changed = str(r.get("path", "")).startswith("L3-")
        ceil_old = b.get("roofline_ceiling_flops_per_s", 0.0) or 0.0
        ceil_new = r.get("roofline_ceiling_flops_per_s", 0.0) or 0.0
        tf_old = b.get("achieved_flops_per_s", 0) / 1e12
        tf_new = r.get("achieved_flops_per_s", 0) / 1e12
        out.append({
            "optimizer": r["optimizer"], "model": r["model"],
            "precision_old": b.get("precision"), "precision_new": r.get("precision"),
            "path_old": path_old, "path_new": r.get("path"),
            "path_changed": path_changed,
            "roof_frac_old": b.get("fraction_of_roofline"),
            "roof_frac_new": r.get("fraction_of_roofline"),
            "roof_frac_delta": (r.get("fraction_of_roofline", 0)
                                - b.get("fraction_of_roofline", 0)),
            "achieved_tf_old": tf_old, "achieved_tf_new": tf_new,
            "achieved_tf_ratio_new_over_old": (tf_new / tf_old) if tf_old else None,
            "ceiling_tf_old": ceil_old / 1e12, "ceiling_tf_new": ceil_new / 1e12,
            "ceiling_ratio_old_over_new": (ceil_old / ceil_new) if ceil_new else None,
            "steps_per_s_old": b.get("steps_per_s"),
            "steps_per_s_new": r.get("steps_per_s"),
            # the honest tag: did the underlying WORK change, or just the ceiling label?
            "interpretation": ("PATH CHANGED (real megakernel now measured)"
                               if path_changed else
                               "ceiling-relabel only (precision changed, work same)"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="decoder,vit,mamba")
    ap.add_argument("--opts", default=",".join(OPTS))
    ap.add_argument("--steps", type=int, default=100)
    # [A6-HIGH] optional override for the tuned-configs file; default per model is
    # results/tuning/tuned_configs_{model}.json (resolved in _load_tuned_configs).
    ap.add_argument("--tuned-configs", default=None,
                    help="path to a tuned_configs JSON; overrides the per-model default")
    # [ROOFLINE-BF16] OWNER DIRECTIVE (task 1 wired): pin every row to RACE precision
    # bf16. The GEMM_IMPL=wgmma wiring is now live, so at bf16 the decoder/vit × adamw
    # cells take the bf16 TENSOR-CORE L3 megakernel on the RACE PATH (engine="wgmma",
    # 989 TF ceiling), and mamba × adamw takes the scalar L3 megakernel (the measured
    # carve-out; engine="scalar", read against its real fp32 66.9 TF ceiling). This
    # REPLACES the old fp32-force, which existed ONLY because the pre-wiring precision
    # gate declined L3 at bf16 → eager (the exact gate task 1 replaced). Default
    # "bf16"; pass "" for tuner/auto (verified null for adamw, so adamw still bf16).
    ap.add_argument("--force-precision", default="bf16",
                    help="pin every row to this matmul precision (default bf16, the race "
                         "precision; '' = tuner/auto)")
    # [ROOFLINE-TC] Standalone TC-cell measurement (Task 2/3 of the PRE-WIRING era):
    # JIT-loaded the *_tc.cu modules to get bf16-TC numbers BEFORE the race path could
    # reach them. Now that the race rows THEMSELVES run the bf16 TC megakernel (via the
    # wired wgmma path), this is redundant — it would add a DUPLICATE "tc" row per model
    # to the plot. Default OFF (""). Pass a model list to also emit the standalone
    # TC + scalar-megakernel sibling rows (e.g. for a direct TC-vs-scalar-mega ratio).
    ap.add_argument("--tc-models", default="",
                    help="models to ALSO measure via the standalone TC + scalar megakernel "
                         "TUs (default off; the bf16 race rows already exercise the TC path)")
    # [R2.4 item 2] MAX-BATCH operating point. Every cell trains at this batch (same
    # B within a model, deterministic repeat-tile of the native ~4.2k-pair train
    # split). Default 16384 = the owner's "B≈16k+ fills 132 SMs" floor (≈ one 128-row
    # tile per 132 SMs). MEASURED: the one-CTA-per-SM megakernel already saturates at
    # B≈2k (within 96% of peak from 2048 on; one 16-row dW atom per SM ≈ 2112) and
    # DECLINES past 16k, while eager (cuBLAS batched GEMMs) is at its utilization knee
    # at 16k (~60% of its 32-65k plateau). 16384 is the shared point that is fair to
    # both and is the megakernel ceiling (larger only hurts it). Pass 0 for the native
    # per-model batch (no tiling).
    ap.add_argument("--max-batch", type=int, default=16384,
                    help="train every cell at this batch (deterministic repeat-tile; "
                         "default 16384 = SM-saturation operating point; 0 = native batch)")
    ap.add_argument("--baseline", default=str(ROOT / "results" / "h100_grokking_race"
                    / "archive" / "roofline_eager_baseline_20260610.json"),
                    help="archived eager roofline.json to diff against for deltas")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    opts = [o for o in args.opts.split(",") if o]
    tc_models = [m for m in args.tc_models.split(",") if m]
    force_prec = args.force_precision or None
    max_batch = args.max_batch if args.max_batch and args.max_batch > 0 else None
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[roofline] force_precision={force_prec!r}  max_batch={max_batch}  "
          f"TASK1 race rows + TASK2/3 TC cells for {tc_models}", flush=True)
    rows = []
    # ── TASK 1: scalar race rows, fp32-forced (adamw → real L3 megakernel) ──
    for model in models:
        tuned = _load_tuned_configs(model, args.tuned_configs)
        for opt in opts:
            try:
                r = measure_pipeline(opt, model, args.steps, tuned_configs=tuned,
                                     force_precision=force_prec, max_batch=max_batch)
                rows.append(r)
                _pv = r.get("peak_vram_bytes")
                _pvs = f"{_pv/1e9:5.2f}GB" if _pv else "  n/a "
                print(f"  {opt:11s}/{model:7s} [{r['precision']:5s}|{r['path']:22s}] "
                      f"B={r['batch']:>6} "
                      f"{r['steps_per_s']:7.1f} steps/s  "
                      f"achieved={r['achieved_flops_per_s']/1e12:6.3f} TF/s  "
                      f"roof%={100*r['fraction_of_roofline']:6.2f}  "
                      f"VRAM={_pvs}  ({r['bound_regime_at_AI']})", flush=True)
            except Exception as e:
                import traceback
                print(f"  {opt}/{model} FAILED: {e}", flush=True)
                traceback.print_exc()

    # ── TASK 2+3: directly-measured TC + scalar megakernel cells ──
    print("[roofline] TASK 2/3 — directly-loaded TC cells (bf16-TC + fp32 scalar "
          "megakernel), ncta_cap=0 full saturation", flush=True)
    for model in tc_models:
        # FLOP basis: the eager-profiled adamw row for this model (use_fused=False
        # in pass 3 → path-independent GEMM count) at full batch.
        adamw_row = next((r for r in rows if r["optimizer"] == "adamw"
                          and r["model"] == model and r.get("path_family") == "scalar"), None)
        if adamw_row is None:
            print(f"  TC[{model}] skipped: no adamw scalar row for FLOP basis", flush=True)
            continue
        try:
            tc_rows = measure_tc_cell(
                model, adamw_row["flops_per_step"], adamw_row["batch"],
                _data_init(model)[2], adamw_row["seq_for_acts"],
                adamw_row["dim_for_acts"], adamw_row["layers_for_acts"])
            for tr in tc_rows:
                rows.append(tr)
                print(f"  {tr['optimizer']:11s}/{model:7s} [{tr['precision']:7s}|"
                      f"{tr['path_family']:11s}] {tr['steps_per_s']:7.1f} steps/s  "
                      f"achieved={tr['achieved_flops_per_s']/1e12:6.3f} TF/s  "
                      f"AI={tr['arithmetic_intensity']:6.1f}  "
                      f"roof%={100*tr['fraction_of_roofline']:6.2f}  "
                      f"({tr['bound_regime_at_AI']})", flush=True)
        except Exception as e:
            import traceback
            print(f"  TC[{model}] FAILED: {e}", flush=True)
            traceback.print_exc()

    deltas = _delta_vs_baseline(rows, args.baseline)
    # [R2.4 item 2] PEAK-VRAM table per pipeline (model × optimizer), production-path
    # peak measured during the wall pass. Grouped by model with the per-model batch.
    vram_table = []
    for r in rows:
        pv = r.get("peak_vram_bytes")
        vram_table.append({
            "model": r["model"], "optimizer": r["optimizer"],
            "path_family": r.get("path_family"), "path": r.get("path"),
            "batch": r.get("batch"),
            "peak_vram_bytes": pv,
            "peak_vram_gb": (round(pv / 1e9, 4) if pv else None),
        })
    # [R2.4 item 2] the baseline JSON predates max-batch tiling (native ~4.2k batch),
    # so `deltas_vs_eager` is BATCH-CONFOUNDED — a fraction move there mixes path +
    # precision + ceiling + BATCH. ROOFLINE.md leads with absolute TF/s and ignores
    # these deltas; flag it so no reader mistakes a batch effect for a path effect.
    _baseline_batch_differs = (max_batch is not None)
    (OUT / "roofline.json").write_text(json.dumps(
        {"peaks": PEAKS, "force_precision": force_prec,
         "max_batch": max_batch,
         "baseline_for_deltas": args.baseline,
         "baseline_batch_differs": _baseline_batch_differs,
         "deltas_batch_confounded_note": (
             "deltas_vs_eager compares max-batch rows against a native-batch baseline; "
             "a fraction move there mixes batch with path/precision — read absolute "
             "achieved_flops_per_s, not the delta." if _baseline_batch_differs else None),
         "batch_saturation_sweep": BATCH_SATURATION_SWEEP,
         "peak_vram_table": vram_table,
         "rows": rows, "deltas_vs_eager": deltas}, indent=2))
    print(f"wrote {OUT/'roofline.json'}", flush=True)
    # ── PEAK-VRAM table to stdout (per pipeline, grouped by model) ──
    print(f"\n[roofline] PEAK VRAM per pipeline (production path, B={max_batch}, "
          f"torch.cuda.max_memory_allocated during the wall pass):", flush=True)
    for model in models:
        mrows = [v for v in vram_table if v["model"] == model]
        if not mrows:
            continue
        b = next((v["batch"] for v in mrows if v["batch"]), "?")
        print(f"  ── {model} (B={b}) ──", flush=True)
        for v in mrows:
            pg = f"{v['peak_vram_gb']:6.3f} GB" if v["peak_vram_gb"] is not None else "   n/a  "
            print(f"     {v['optimizer']:11s} [{str(v['path_family']):6s}] "
                  f"peak={pg}  ({v['path']})", flush=True)
    if deltas:
        print("\n[roofline] DELTAS vs eager baseline — fraction moves confound PATH "
              "+ PRECISION + CEILING; achieved TF/s is the honest axis:", flush=True)
        for d in sorted(deltas, key=lambda x: -x["roof_frac_delta"]):
            tag = "PATH-CHANGED" if d["path_changed"] else "ceiling-relabel"
            print(f"  {d['optimizer']:11s}/{d['model']:7s} [{tag:14s}] "
                  f"{d['precision_old']}→{d['precision_new']}  "
                  f"roof% {100*d['roof_frac_old']:.2f}→{100*d['roof_frac_new']:.2f}  "
                  f"achieved-TF/s {d['achieved_tf_old']:.2f}→{d['achieved_tf_new']:.2f} "
                  f"(ceiling {d['ceiling_tf_old']:.0f}→{d['ceiling_tf_new']:.0f} TF)  "
                  f"[{d['path_new']}]", flush=True)
    # ── HONEST HEADLINE: same-precision (fp32) absolute comparison for the cells
    # where the REAL megakernel path now fires (adamw) vs the eager+L1 path the
    # other optimizers run. Identical ceiling (66.9 TF) → apples-to-apples.
    print("\n[roofline] SAME-PRECISION (fp32) megakernel vs eager — absolute "
          "achieved TF/s (identical 66.9 TF ceiling, apples-to-apples):", flush=True)
    for model in tc_models:
        l3 = next((r for r in rows if r["optimizer"] == "adamw" and r["model"] == model
                   and r.get("path_family") == "scalar"), None)
        mega = next((r for r in rows if r["model"] == model
                     and r.get("path_family") == "scalar-mega"), None)
        tc = next((r for r in rows if r["model"] == model
                   and r.get("path_family") == "tc"), None)
        # an eager fp32 reference: the fastest non-adamw scalar row (eager+L1) for this model
        eager = max((r for r in rows if r["model"] == model
                     and r.get("path_family") == "scalar"
                     and not str(r.get("path", "")).startswith("L3-real")),
                    key=lambda r: r["achieved_flops_per_s"], default=None)
        parts = []
        if l3:
            parts.append(f"L3-scalar(adamw)={l3['steps_per_s']:.1f} st/s "
                         f"{l3['achieved_flops_per_s']/1e12:.3f} TF/s")
        if eager:
            parts.append(f"eager+L1({eager['optimizer']})={eager['steps_per_s']:.1f} st/s "
                         f"{eager['achieved_flops_per_s']/1e12:.3f} TF/s")
        if tc:
            parts.append(f"TC(bf16)={tc['steps_per_s']:.1f} st/s "
                         f"{tc['achieved_flops_per_s']/1e12:.3f} TF/s")
        if l3 and eager:
            r = eager["achieved_flops_per_s"] / max(l3["achieved_flops_per_s"], 1e-30)
            parts.append(f"→ eager is {r:.1f}× the L3 megakernel at fp32")
        print(f"  {model:8s}: " + "  |  ".join(parts), flush=True)
    make_plot(rows, OUT / "roofline.png")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
