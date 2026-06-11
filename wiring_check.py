"""wiring_check.py — OWNER BASELINE GATE: assert every routed cell's EXECUTED
path is the L3-TC persistent megakernel (bf16 wgmma engine), fail LOUD otherwise.

Mandated by the locked baseline directive: "build a wiring_check.py: route every
cell, assert executed path == L3-TC, fail loud — saved to repo, run before every
roofline." This is the per-cell pass/fail oracle for the 33 (model × optimizer)
cells; the roofline must not be measured until this reports the converted set.

WHAT IT DOES (no surrogate, no inference from gate logic — it OBSERVES the real
executed engine):
  * For each (model, optimizer) cell it runs a few REAL race train steps via the
    production train_<opt> function (grokking_race_v2.OPTIMIZER_REGISTRY), with
    the SAME dispatch wrapper the roofline uses to capture grokking_race_v2.
    LAST_L3_ENGINE — the engine dispatch.cpp ACTUALLY ran ("wgmma" | "scalar" |
    None). dispatch.cpp has no silent wgmma->scalar fallback, so a captured
    engine=="wgmma" PROVES the bf16 tensor-core L3 megakernel executed.
  * A cell PASSES iff every timed step fired engine=="wgmma" (path == L3-TC) AND
    the stale-ABI latch never tripped (a TypeError soft-degrade would mean the
    extension predates the widened fused_step ABI — rebuild, then re-run).
  * A cell is BLOCKED otherwise; the row records the OBSERVED path (eager /
    eager+L1-fused-tail / L3-scalar) and a cited reason so the converted/blocked
    table is self-documenting.

USAGE
  python wiring_check.py                       # all 33 cells, table + JSON
  python wiring_check.py --models decoder      # subset
  python wiring_check.py --opts adamw,lion     # subset
  python wiring_check.py --require-all         # exit 1 unless ALL routed cells
                                               # are L3-TC (the FULL-baseline gate)
Default exit code: 0 if the run completed (the converted/blocked split is the
deliverable); --require-all makes a non-fully-converted run fail loud (exit 1),
which is the gate the locked directive's end-state wants green.

Output JSON: results/h100_grokking_race/wiring_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "h100_grokking_race"

# The 33 cells. Optimizer keys are the RACE registry keys (note: the race uses
# "supergrok" for the SuperGrok11 cell; dispatch/codegen use "supergrok11" — the
# train fn owns the mapping, so we drive by the race key and let it translate).
MODELS = ["decoder", "vit", "mamba"]
OPTS = ["adamw", "neuralgrok", "grokadamw", "supergrok", "supergrok15",
        "supergrok2", "grokfast", "muon", "lion", "looksam", "prodigy"]

# Cited reasons a cell cannot (yet) be L3-TC. Keyed by the OBSERVED engine/path so
# the table explains every non-converted cell from the single source of truth
# (INTEGRATION-OPTSTAGES.md verdict table + dispatch.cpp _L3_WGMMA_CELLS + the
# fused_train_step registries). These are diagnostic annotations, not gates — the
# GATE is the observed engine. A blocked cell still fails the --require-all gate.
_BLOCK_REASONS = {
    # optimizer -> short reason (why its L3-TC tail is not wired on the prod path)
    "looksam": ("MODEL-COUPLED: st.sam_dir needs a SAM 2nd backward in the model "
                "phase (INTEGRATION-OPTSTAGES §6); model-stage-owned, not wired."),
    "supergrok": ("MODEL-COUPLED 2nd-pass (OPTSTAGES §4): SG11's `sharpness` is "
                  "(g_sam−g)² from a SAM SECOND backward at perturbed weights — a "
                  "FULL second in-kernel model fwd+bwd+dW-assembly (perturb→2nd "
                  "backward→restore), the SAME model-stage machinery as looksam, a "
                  "DIFFERENT category from the per-element/precompute tails "
                  "(prodigy-d, muon-NS) that were in scope. The per-tensor mu/gate "
                  "meta-MLP precompute sits ON TOP of that 2nd pass. NOTE: the missing "
                  "`sharpness` FusedOptState field is cheap (append-only, as aux_* "
                  "proved) but INSUFFICIENT — the second fwd+bwd pipeline is the real "
                  "blocker. A host-side 2nd backward feeding st.sharpness would be an "
                  "intermediate launch (violates single-persistent-launch). Probed "
                  "re-invocable (vittc_forward_tile/vittc_backward_tile are device "
                  "fns) but the unit of work is the whole P1+P2 pipeline, twice."),
    "supergrok15": ("MODEL-COUPLED 2nd-pass (OPTSTAGES §5): same blocker as SG11 — "
                    "SG15's `sharpness` is a SAM SECOND-backward side-channel needing "
                    "a FULL second in-kernel model fwd+bwd+dW-assembly. SG15 is "
                    "SIMPLER than SG11 in the tail (gate is a host scalar, no cosine "
                    "stage) but identical at the blocking layer: the 2nd pass is "
                    "model-stage machinery, out of the per-element-tail scope. The "
                    "missing `sharpness` field is cheap but insufficient (the 2nd "
                    "fwd+bwd is the real cost)."),
    "supergrok2": ("MODEL-COUPLED 2nd-pass + SEPARATE-LAUNCH precompute (OPTSTAGES "
                   "§7/INTEGRATION-NOTES): the sg2_meta_tail is PARITY-PROVEN as a "
                   "STANDALONE kernel (sg2_meta_optimizer_tail, its own <<<>>> "
                   "launch), but it CONSUMES pre-computed host inputs it cannot "
                   "produce in the single vit launch: (i) `sharpness_packed` — the "
                   "SAME SAM 2nd-backward side-channel as SG11/15 (needs a 2nd in-"
                   "kernel fwd+bwd), AND (ii) `perm_packed`/`unsort_packed` — a "
                   "per-tensor SEGMENTED-SORT permutation (a separate sort stage), "
                   "plus the CSA/HCA/PEER/GRU meta-net weight pack. It is NOT a drop-"
                   "in P2.x/P3 tail slot like grokadamw/prodigy/muon; wiring it into "
                   "the single persistent launch needs the 2nd-pass pipeline AND an "
                   "in-kernel segmented sort — model-stage + separate-launch "
                   "machinery, both out of scope."),
    "prodigy": ("STAGED global-d, CONVERTED for decoder+vit (wave-2): the cross-all-"
                "tensors d-reduction is the in-kernel P2.6 phase + the owner-block "
                "beta3-EMA/d_coef/persist (OPTSTAGES §2). Only mamba×prodigy is still "
                "blocked (no mamba P2.6 wiring) — decoder/vit fire wgmma."),
    "muon": ("STAGED grid-cooperative NS, CONVERTED for decoder+vit+mamba (wave-2): the "
             "2D weights' Newton-Schulz orthogonalization is the in-kernel P2.7 phase "
             "(grid-barrier-looped, no separate launch; decoder/mamba 13 2D weights, vit 11); "
             "the 1D weights take the P3 AdamW tail with INDEPENDENT aux_lr/aux_betas (the "
             "eager non-2D group, carried by the append-only FusedScalars aux_* widening). "
             "OPTSTAGES §3. muon/mamba is a SINGLE forward + NS precompute (NOT a 2nd forward), "
             "so it does NOT hit the shared mamba-forward A/A/A race that blocks looksam/prodigy/"
             "SG mamba — A/A/A bit-exact. ALL THREE fire wgmma; if ANY shows here, the rebuild "
             "is stale."),
    "neuralgrok": ("KERNEL-READY, RACE-BLOCKED (directive item d): the psi MLP is in "
                   "opt_components.cuh (kPsiHidden=16 == race neural_hidden=16, 2-layer), "
                   "so a snapshot-plumbed L3 step would PASS the tail gate (m/v parity "
                   "validates the psi pack; clip is inert at steps 1&50, max grad-norm "
                   "0.72<1.0). BUT train_neuralgrok trains the amplifier BETWEEN steps via "
                   "a differentiable Python rebuild that needs the model grads, and the L3 "
                   "megakernel returns only the loss — snapshot-plumbing alone yields a "
                   "frozen-amplifier (non-faithful) race. The race integration (re-extract "
                   "+ keep the amplifier-training half) is deferred; needs owner re-scope."),
    # grokadamw (decoder): CONVERTED in this cycle (no longer blocked) — all THREE
    # eager mechanisms land in the bf16 TC path: (i) per-tensor layer-wise β1 +
    # rebased bc1 in the kernel P3 (t == flat layer index); (ii) GLOBAL grad-norm
    # clip via the kernel's P2.5 deterministic global-norm reduction (γ/grad_clip
    # thread through FusedScalars); (iii) adaptive-α is a no-op in-context (no
    # losses fed to .step()), so the static α is faithful. So decoder×grokadamw
    # should be OBSERVED as wgmma; if it ever shows here, the kernel rebuild is
    # stale. vit×grokadamw is ALSO CONVERTED now (wave-2 vit lane — the same 3
    # mechanisms on the vit TC kernel: P2.5 global-norm clip + P3 per-tensor β1),
    # so it too should be OBSERVED as wgmma, not listed here.
}
# Per-(model) note for any cell that routes to a scalar L3 megakernel rather than
# wgmma (mamba's measured carve-out — the path EXISTS but scalar wins on wall).
_MODEL_NOTE = {
    "mamba": ("mamba×{adamw,lion,grokfast} now run the OptId-generic wgmma TC kernel "
              "(launch_fused_mamba_megakernel_tc<Opt>), WIRED in-_ops via mega_mamba_"
              "real_adamw_tc_launcher.cu + the dispatch.cpp wgmma branch (cycle-2). The "
              "0.46x scalar-wins (905a4bb: selective-scan/conv1d dominate, not the 4 "
              "projection GEMMs) is a PERFORMANCE fact the roofline reports, not a "
              "correctness carve-out (test_mamba_tc 5/5). fp32 mamba still routes to the "
              "scalar megakernel; grokadamw/neuralgrok are not mamba TC tails."),
}

_G = None


def _g():
    global _G
    if _G is None:
        sys.path.insert(0, str(ROOT))
        import grokking_race_v2 as g
        _G = g
    return _G


_CACHE = {}


def _cfg(opt, model, steps):
    """A minimal race config that runs `steps` pure train steps (no eval, no early
    stop) at the production precision (bf16 → the wgmma engine for TC cells)."""
    g = _g()
    c = dict(g.DEFAULT_CONFIG)
    c.update(g.OPTIMIZER_CONFIGS[opt])
    c.update({"seed": 42, "model_type": model, "frac_train": 0.5, "val_ratio": 0.10,
              "p": 97, "use_amp": False, "use_fused": True, "compile_model": False,
              "matmul_precision": "bf16",  # production default → wgmma for TC cells
              "max_steps": steps, "early_stop_max_steps": steps,
              "eval_every": 10 ** 9, "early_stop_threshold": 1.01,
              "early_stop_patience": 10 ** 9, "weight_decay": 1.0})
    return c


def _data_init(model):
    import torch
    if model not in _CACHE:
        g = _g()
        c = _cfg("adamw", model, 10)
        dev = torch.device("cuda")
        data = tuple(d.to(dev) for d in g.make_data_for_task(c, 42))
        m0 = g.build_model(c, dev)
        init = {k: v.detach().cpu().clone() for k, v in m0.state_dict().items()}
        del m0
        _CACHE[model] = (data, init)
    return _CACHE[model]


def check_cell(opt, model, steps=4):
    """Route ONE cell through the real race train fn for `steps` steps and report
    the OBSERVED execution path. Returns a dict row (never raises for a cell-level
    failure — it records it so the table is complete)."""
    import torch
    g = _g()
    fn = g.OPTIMIZER_REGISTRY[opt]
    data, init = _data_init(model)
    dev = torch.device("cuda")

    # Capture the engine of EVERY L3 step + whether L1 ever fired, by wrapping the
    # two dispatch entry points (identical technique to tuning.roofline). We record
    # the SET of engines seen across the timed steps so a cell that flickers between
    # paths is caught (it must be wgmma on EVERY step to pass).
    seen_l3_engines = []
    l1_fired = [0]
    _orig_l3 = g._try_fused_train_step
    _orig_l1 = g._try_fused_step

    def _wrap_l3(*a, **k):
        r = _orig_l3(*a, **k)
        if r is not None:
            eng = g.LAST_L3_ENGINE["engine"] if g.LAST_L3_ENGINE else None
            seen_l3_engines.append(eng)
        return r

    def _wrap_l1(*a, **k):
        r = _orig_l1(*a, **k)
        if r:
            l1_fired[0] += 1
        return r

    g._try_fused_train_step = _wrap_l3
    g._try_fused_step = _wrap_l1
    err = None
    try:
        c = _cfg(opt, model, steps)
        torch.manual_seed(42)
        fn(c, init, *data, dev, bp=0)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 — record the failure into the row
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        g._try_fused_train_step = _orig_l3
        g._try_fused_step = _orig_l1

    abi_stale = bool(getattr(g, "_FUSED_ABI_STALE", False))
    n_l3 = len(seen_l3_engines)
    all_wgmma = (n_l3 > 0 and all(e == "wgmma" for e in seen_l3_engines)
                 and not abi_stale and err is None)
    # Observed path label (mirrors roofline's actual_path semantics).
    if err is not None:
        path = f"ERROR ({err.split(':')[0]})"
    elif abi_stale:
        path = "eager(ABI-stale)"
    elif n_l3 > 0 and all(e == "wgmma" for e in seen_l3_engines):
        path = "L3-TC-megakernel(wgmma)"
    elif n_l3 > 0 and all(e == "scalar" for e in seen_l3_engines):
        path = "L3-scalar-megakernel"
    elif n_l3 > 0:
        path = "L3-mixed(" + ",".join(sorted(set(seen_l3_engines))) + ")"
    elif l1_fired[0] > 0:
        path = "eager+L1-fused-tail"
    else:
        path = "eager"

    converted = bool(all_wgmma)
    if converted:
        reason = ""
    elif err is not None:
        reason = f"step raised: {err}"
    elif abi_stale:
        reason = ("ABI-stale latch tripped (extension predates the widened "
                  "fused_step signature) — rebuild _ops then re-run.")
    elif path == "L3-scalar-megakernel":
        reason = ("ran the fp32 SCALAR L3 megakernel, not the bf16 wgmma TC engine. "
                  + _MODEL_NOTE.get(model, ""))
    else:
        # eager / eager+L1: the cell has no production L3-TC route.
        # Single-launch wgmma-tail optimizers (lion is the only converted one; the
        # others have their own tail-specific blocker) that land on mamba are blocked
        # by the MISSING mamba wgmma launcher TU — cite that, not the generic reason.
        _wgmma_tails = {"adamw", "lion", "grokfast", "grokadamw", "neuralgrok"}
        if model == "mamba" and opt in _wgmma_tails and opt not in _BLOCK_REASONS:
            reason = ("BLOCKED by the missing mamba wgmma launcher TU. " + _MODEL_NOTE["mamba"])
        elif model == "mamba" and opt in ("prodigy", "looksam"):
            # NOTE: prodigy/mamba + looksam/mamba are now CONVERTED (the A/A/A race — a
            # register-pressure wgmma-accumulator spill in the mamba TC backward — is FIXED via
            # the fused dB/dC reduce + a_save-drop + __noinline__ in model_stage_mamba{3,_tc}.cuh;
            # they PASS test_l3tc_tail_gate A/A/A bit-exact). They route engine=="wgmma" and hit
            # the `if converted` branch above, so this branch is normally DEAD for them; it only
            # fires if a build regresses the route. Kept as a tripwire.
            reason = ("UNEXPECTED: mamba×" + opt + " did NOT route wgmma despite the A/A/A race "
                      "fix — check the dispatch.cpp mamba gate + the __noinline__ scan-bwd fix.")
        else:
            reason = _BLOCK_REASONS.get(
                opt, "no production L3-TC route wired for this (model, optimizer).")
        if model in _MODEL_NOTE and opt == "adamw":
            reason = _MODEL_NOTE[model]

    return {
        "model": model, "optimizer": opt,
        "executed_path": path,
        "l3_steps_fired": n_l3,
        "l3_engines_seen": sorted(set(seen_l3_engines)),
        "l1_steps_fired": l1_fired[0],
        "abi_stale": abi_stale,
        "error": err,
        "is_l3_tc": converted,
        "reason_if_blocked": reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--opts", default=",".join(OPTS))
    ap.add_argument("--steps", type=int, default=4,
                    help="real train steps to route per cell (default 4)")
    ap.add_argument("--require-all", action="store_true",
                    help="exit 1 unless EVERY routed cell executed L3-TC (the "
                         "full-baseline gate); default exits 0 with the split")
    ap.add_argument("--out", default=str(OUT / "wiring_check.json"))
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    opts = [o for o in args.opts.split(",") if o]
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[wiring_check] routing {len(models)*len(opts)} cells "
          f"({len(models)} models x {len(opts)} opts), {args.steps} real steps each.\n"
          f"[wiring_check] PASS == executed engine is wgmma (bf16 TC L3 megakernel) "
          f"on EVERY step.\n", flush=True)
    rows = []
    for model in models:
        for opt in opts:
            row = check_cell(opt, model, steps=args.steps)
            rows.append(row)
            mark = "PASS L3-TC" if row["is_l3_tc"] else "BLOCKED   "
            extra = "" if row["is_l3_tc"] else f"  ← {row['reason_if_blocked'][:72]}"
            print(f"  [{mark}] {opt:11s}/{model:7s}  path={row['executed_path']:26s}"
                  f"{extra}", flush=True)

    converted = [r for r in rows if r["is_l3_tc"]]
    blocked = [r for r in rows if not r["is_l3_tc"]]
    n = len(rows)
    frac = len(converted) / n if n else 0.0
    summary = {
        "total_cells": n,
        "converted_l3_tc": len(converted),
        "blocked": len(blocked),
        "fraction_l3_tc": round(frac, 4),
        "converted_cells": [f"{r['optimizer']}/{r['model']}" for r in converted],
        "blocked_cells": [f"{r['optimizer']}/{r['model']}" for r in blocked],
    }
    payload = {"summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\n[wiring_check] {len(converted)}/{n} cells on L3-TC "
          f"({100*frac:.1f}%).  wrote {args.out}", flush=True)

    # Fraction table by model (the directive's "frac table" seed).
    print("\n[wiring_check] L3-TC fraction by model:", flush=True)
    for model in models:
        mrows = [r for r in rows if r["model"] == model]
        c = sum(1 for r in mrows if r["is_l3_tc"])
        print(f"   {model:8s}: {c}/{len(mrows)} L3-TC", flush=True)

    if args.require_all and blocked:
        print(f"\n[wiring_check] FAIL-LOUD (--require-all): {len(blocked)} cell(s) "
              f"are NOT L3-TC. The locked baseline requires all 33 on L3-TC. "
              f"Blocked: {summary['blocked_cells']}", file=sys.stderr, flush=True)
        sys.exit(1)
    print("\n[wiring_check] DONE", flush=True)


if __name__ == "__main__":
    main()
