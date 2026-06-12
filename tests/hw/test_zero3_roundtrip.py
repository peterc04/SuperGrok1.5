"""tests/hw/test_zero3_roundtrip.py — ZeRO-3 gather/release + sharded-state
CHECKPOINT/RESUME round-trip around the REAL fused step (1 GPU).

IMPLEMENTS /workspace/.parallelism_design.md §3.2(a) + the checkpoint/resume
requirement (task #25 phase-2, Lane C), building DIRECTLY on ef433ac's
DELIVERABLE 1 (the sharded-opt-kernel == in-kernel-P3 bit-parity) and the DP=2
loopback's shard conventions: VIRTUAL world=2 on one GPU, elementwise-even
flat plan, shard-local [m|v|extra] state planes.

THE TWO-STEP PROTOCOL (decoder × {adamw, grokfast}):
  step 1: production fused step (return_grad) on params0 → grad1; the in-kernel
          P3 advances the model to params1_ref. The ZeRO-3 path applies
          sharded_opt_step per VIRTUAL rank on its (param slice, grad slice,
          zero state slice) via Zero3FlatParamStore; gather_full(peers) must
          equal params1_ref BIT-EXACTLY (the DELIVERABLE-1 parity, now driven
          through the zero3 store/plan machinery — proves the slice plumbing).
  save:   each rank persists ONLY its param+state shard (+ plan fingerprint).
  step 2 LIVE: fused step on params1 → grad2; sharded opt with the LIVE state
          shards → params2_live (must equal the model's params2_ref bitwise —
          the 2-step sharded chain == the 2-step production chain).
  RESUME: fresh stores ← load_sharded_checkpoint; loaded shards must equal the
          save-time shards bitwise; re-apply step 2 on the LOADED state with
          the SAME grad2 → params2_resumed + state2_resumed must equal the
          LIVE step-2 results BIT-EXACTLY. That is the save/resume round-trip:
          a resumed run is indistinguishable from an uninterrupted one.
  guards: loading under a drifted plan (world=4) or wrong rank RAISES (no
          silent re-shard); gather/release keeps shards consistent on device.

Run (GPU box):
    PYTHONPATH=. python -m pytest tests/hw/test_zero3_roundtrip.py -q -s
    PYTHONPATH=. python tests/hw/test_zero3_roundtrip.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

_THIS = os.path.abspath(__file__)
_REPO = os.path.abspath(os.path.join(os.path.dirname(_THIS), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_WORLD = 2          # virtual ranks on one GPU (the loopback shape)
_OPTID = {"adamw": 0, "lion": 1, "grokfast": 2}
_CELLS = ["adamw", "grokfast"]


def _sm90a_available() -> bool:
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


_GATE = pytest.mark.skipif(
    not _sm90a_available(),
    reason="ZeRO-3 round-trip gate needs an sm_90 (Hopper) GPU")


def _build_cell(seed=42):
    import grokking_race_v2 as g
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": "decoder", "use_amp": False, "use_fused": True,
              "compile_model": False, "matmul_precision": "bf16", "seed": seed,
              "frac_train": 0.5, "val_ratio": 0.10, "p": 97})
    dev = torch.device("cuda")
    data = tuple(d.to(dev) for d in g.make_data_for_task(c, seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = g.build_model(c, dev)
    return g, c, m, data, dev


def _make_optimizer(opt, g, params, c):
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                                 weight_decay=c["weight_decay"])
    if opt == "grokfast":
        return g.Grokfast(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                          weight_decay=c["weight_decay"],
                          grokfast_alpha=c.get("grokfast_alpha", 0.98),
                          grokfast_lamb=c.get("grokfast_lamb", 2.0))
    raise ValueError(opt)


def _flat_params(named, total, dev):
    out = torch.empty(total, device=dev)
    off = 0
    for _, p in named:
        n = p.numel()
        out[off:off + n].copy_(p.data.reshape(-1))
        off += n
    return out


def _sharded_step(mod, opt, stores, states, grad_full, sc, step):
    """Apply sharded_opt_step on every virtual rank's owned slice (in place on
    the stores' shards + state planes), mirroring the DP loopback's per-rank
    [3]. grad_full is the deterministically-reduced full grad."""
    for r, store in enumerate(stores):
        for fs, fe, ss, se in store.owned():
            params_shard = store.shard[ss:se].contiguous()
            grad_shard = grad_full[fs:fe].contiguous()
            state_shard = states[r][:, ss:se].reshape(-1).contiguous()
            mod.sharded_opt_step(
                _OPTID[opt], params_shard, grad_shard, state_shard,
                sc["lr"], sc["beta1"], sc["beta2"], sc["eps"], sc["weight_decay"],
                sc["bc1"], sc["bc2"], sc.get("alpha", 0.98), sc.get("lamb", 2.0),
                step)
            # write back (contiguous() may have copied).
            store.shard[ss:se].copy_(params_shard)
            states[r][:, ss:se].copy_(state_shard.view(3, se - ss))
    torch.cuda.synchronize()


def run_zero3_roundtrip(opt, verbose=True):
    from grokking_optimizers.dispatch import (fused_train_step, canonicalize_model,
                                              has_l3_real, gemm_impl_for_cell,
                                              _opt_scalars_from)
    from grokking_optimizers.parallel.zero3 import (
        Zero3FlatParamStore, flat_plan_for_optimizer, load_sharded_checkpoint,
        save_sharded_checkpoint)
    from tests.hw.test_sharded_optimizer import _build_binding
    mod = _build_binding()
    g, c, m, data, dev = _build_cell()
    canon = canonicalize_model("decoder")
    assert has_l3_real(canon, opt) and gemm_impl_for_cell(canon, opt, "bf16") == "wgmma"

    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    named_sizes = [(n, p.numel()) for n, p in named]
    total = sum(n for _, n in named_sizes)
    plan = flat_plan_for_optimizer(named_sizes, _WORLD, opt)
    opt_obj = _make_optimizer(opt, g, m.parameters(), c)
    cache = {}

    # ── step 1 ──────────────────────────────────────────────────────────────
    params0 = _flat_params(named, total, dev)
    _, grad1 = fused_train_step(canon, opt, m, opt_obj, data[0], data[1],
                                state_cache=cache, step=1, return_grad=True,
                                gemm_impl="wgmma")
    params1_ref = _flat_params(named, total, dev)

    stores = [Zero3FlatParamStore(plan, r, full_flat=params0) for r in range(_WORLD)]
    # shard-local [m|v|extra] planes per rank, [3, shard_numel] (zero cold-start).
    states = [torch.zeros(3, s.shard.numel(), device=dev) for s in stores]
    _sharded_step(mod, opt, stores, states, grad1, _opt_scalars_from(opt_obj, 1), 1)
    params1 = stores[0].gather_full(peers=stores[1:])
    step1_eq = torch.equal(params1, params1_ref)

    # gather/release consistency on device: release(updated full) → re-gather.
    probe = params1.clone()
    for st in stores:
        st.release(probe)
    regather_eq = torch.equal(stores[0].gather_full(peers=stores[1:]), params1)

    # ── checkpoint (each rank persists ONLY its shard) ──────────────────────
    td = tempfile.mkdtemp(prefix="zero3_ckpt_")
    saved_shards = []
    for r in range(_WORLD):
        save_sharded_checkpoint(
            os.path.join(td, f"rank{r}.pt"), plan=plan, rank=r, step=1,
            param_shard=stores[r].shard, opt_state_shard=states[r].reshape(-1),
            extra_meta={"opt": opt, "model": "decoder"})
        saved_shards.append((stores[r].shard.clone(), states[r].clone()))

    # ── step 2 LIVE ─────────────────────────────────────────────────────────
    _, grad2 = fused_train_step(canon, opt, m, opt_obj, data[0], data[1],
                                state_cache=cache, step=2, return_grad=True,
                                gemm_impl="wgmma")
    params2_ref = _flat_params(named, total, dev)
    _sharded_step(mod, opt, stores, states, grad2, _opt_scalars_from(opt_obj, 2), 2)
    params2_live = stores[0].gather_full(peers=stores[1:])
    states2_live = [s.clone() for s in states]
    step2_eq = torch.equal(params2_live, params2_ref)

    # ── RESUME from the step-1 checkpoint ───────────────────────────────────
    stores_r = [Zero3FlatParamStore(plan, r) for r in range(_WORLD)]
    states_r = []
    loaded_eq = True
    for r in range(_WORLD):
        obj = load_sharded_checkpoint(os.path.join(td, f"rank{r}.pt"),
                                      plan=plan, rank=r)
        stores_r[r].shard.copy_(obj["param_shard"].to(dev))
        st = obj["opt_state_shard"].to(dev).view(3, -1).clone()
        states_r.append(st)
        loaded_eq = loaded_eq and torch.equal(stores_r[r].shard, saved_shards[r][0]) \
                              and torch.equal(st, saved_shards[r][1])
    _sharded_step(mod, opt, stores_r, states_r, grad2,
                  _opt_scalars_from(opt_obj, 2), 2)
    params2_resumed = stores_r[0].gather_full(peers=stores_r[1:])
    resume_params_eq = torch.equal(params2_resumed, params2_live)
    resume_state_eq = all(torch.equal(states_r[r], states2_live[r])
                          for r in range(_WORLD))

    # ── loud-guard checks: drifted plan / wrong rank refuse to load ─────────
    plan4 = flat_plan_for_optimizer(named_sizes, 4, opt)
    guard_plan = guard_rank = False
    try:
        load_sharded_checkpoint(os.path.join(td, "rank0.pt"), plan=plan4, rank=0)
    except RuntimeError:
        guard_plan = True
    try:
        load_sharded_checkpoint(os.path.join(td, "rank0.pt"), plan=plan, rank=1)
    except RuntimeError:
        guard_rank = True

    ok = (step1_eq and regather_eq and step2_eq and loaded_eq
          and resume_params_eq and resume_state_eq and guard_plan and guard_rank)
    if verbose:
        print(f"  [zero3/{opt}] step1 sharded==P3: {step1_eq} | re-gather: "
              f"{regather_eq} | step2 sharded==P3: {step2_eq} | ckpt loaded "
              f"bit-eq: {loaded_eq} | RESUME params bit-eq: {resume_params_eq} "
              f"state bit-eq: {resume_state_eq} | guards plan/rank: "
              f"{guard_plan}/{guard_rank}", flush=True)
    return ok, dict(step1_eq=step1_eq, regather_eq=regather_eq, step2_eq=step2_eq,
                    loaded_eq=loaded_eq, resume_params_eq=resume_params_eq,
                    resume_state_eq=resume_state_eq, guard_plan=guard_plan,
                    guard_rank=guard_rank)


@_GATE
@pytest.mark.parametrize("opt", _CELLS)
def test_zero3_checkpoint_resume_roundtrip(opt):
    """ZeRO-3 virtual-world=2: sharded 2-step chain == production 2-step chain
    bit-exactly; save-at-step-1 → resume → step 2 is bit-identical to the
    uninterrupted run; drifted-plan/rank loads refuse loudly. Design §3 (§7
    pre-work, building on ef433ac DELIVERABLE 1)."""
    ok, d = run_zero3_roundtrip(opt, verbose=True)
    assert d["step1_eq"], f"zero3/{opt}: sharded step-1 != fused P3 (slice plumbing broken)"
    assert d["step2_eq"], f"zero3/{opt}: sharded step-2 != fused P3 (state continuity broken)"
    assert d["loaded_eq"], f"zero3/{opt}: checkpoint loaded shards != saved shards"
    assert d["resume_params_eq"] and d["resume_state_eq"], (
        f"zero3/{opt}: RESUMED step-2 differs from the uninterrupted run "
        f"(params={d['resume_params_eq']} state={d['resume_state_eq']}) — the "
        f"checkpoint round-trip is not bit-exact.")
    assert d["regather_eq"], f"zero3/{opt}: gather/release round-trip drifted"
    assert d["guard_plan"] and d["guard_rank"], (
        f"zero3/{opt}: a mismatched plan/rank checkpoint loaded SILENTLY — "
        f"the fingerprint guard is broken.")
    assert ok


def main() -> int:
    if not _sm90a_available():
        print("[zero3-roundtrip] SKIP: no sm_90 GPU")
        return 0
    fails = []
    for opt in _CELLS:
        ok, d = run_zero3_roundtrip(opt, verbose=True)
        if not ok:
            fails.append((opt, d))
    if fails:
        print(f"[zero3-roundtrip] FAIL: {[o for o, _ in fails]}")
        return 1
    print(f"[zero3-roundtrip] OK: {_CELLS} — sharded chain == production chain; "
          f"checkpoint/resume bit-exact; guards loud.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
