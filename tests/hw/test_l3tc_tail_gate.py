"""tests/hw/test_l3tc_tail_gate.py — per-cell L3-TC conversion gate (owner baseline).

The owner baseline directive's per-cell gates for a newly-converted L3-TC cell:
  (1) megakernel-vs-eager single-step  — the in-kernel optimizer TAIL must match a
      reference apply of the SAME canonical update on the SAME TC-reduced grad.
  (2) A/A/A determinism                — the production L3-TC step is bit-identical
      across 3 runs from the same init (fixed tile ownership + ascending-k/t reduce).

WHY THIS SHAPE (not a full loss-trajectory-vs-eager oracle): the bf16 wgmma fwd+bwd
and the deterministic grad reduction are SHARED with the already-gated adamw TC cell
(decoder 13/13+5, vit 21/21, mamba 5/5 — grad parity vs the bf16-faithful oracle).
A converted cell changes ONLY the per-element tail (apply_optimizer<Opt>), which is
the 14/0-apply-parity math. So the cell-specific NEW surface is exactly the tail, and
the right gate is "did the kernel's tail apply the canonical update to the reduced
grad correctly" — which is (1). The grad itself is validated by the adamw gates +
the determinism check (2) proves no race was introduced by the tail swap.

The reference apply is the canonical CPU math (transcribed from csrc/algorithms/<opt>.h
— the SAME single source the kernel's apply_optimizer<Opt> calls), run on the grad the
kernel returns (fused_train_step(return_grad=True)). fp32 both sides → tight tol.

Run:
  PYTHONPATH=. python -m tests.hw.test_l3tc_tail_gate                 # all converted
  PYTHONPATH=. python -m tests.hw.test_l3tc_tail_gate --cell lion/decoder
  PYTHONPATH=. python -m pytest tests/hw/test_l3tc_tail_gate.py -q
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _g():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import grokking_race_v2 as g
    return g


# Cell spec: model, race optimizer key, and a factory that builds the REAL eager
# optimizer with the EXACT production hyperparameters (the SAME ones train_<opt> uses).
# The gate references this LIVE optimizer (not a header re-transcription), so it catches
# any mechanism the real optimizer has that apply_optimizer<Opt> drops (grad_clip,
# per-layer schedules, adaptive alpha) — the "megakernel-vs-REAL-eager" the directive
# means. State layout in fused_train_step is [m|v|extra].
def _adamw_factory(g, params, c):
    # train_adamw: torch.optim.AdamW(lr, betas=(beta1,beta2), weight_decay, fused=True)
    import torch
    return torch.optim.AdamW(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                             weight_decay=c["weight_decay"])


def _lion_factory(g, params, c):
    # train_lion: Lion(lr=lion_lr, betas=(beta1, 0.99), weight_decay=lion_wd)
    return g.Lion(params, lr=c.get("lion_lr", 3e-4), betas=(c["beta1"], 0.99),
                  weight_decay=c.get("lion_wd", 3.0))


def _grokfast_factory(g, params, c):
    # train_grokfast: Grokfast(lr, betas=(beta1,beta2), weight_decay,
    #                          grokfast_alpha, grokfast_lamb)
    return g.Grokfast(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                      weight_decay=c["weight_decay"],
                      grokfast_alpha=c.get("grokfast_alpha", 0.98),
                      grokfast_lamb=c.get("grokfast_lamb", 2.0))


def _grokadamw_factory(g, params, c):
    # train_grokadamw: GrokAdamW(lr, betas, weight_decay, alpha, gamma, kappa,
    #                            lamb, grad_clip, ...) — has grad_clip + gamma + kappa
    return g.GrokAdamW(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                       weight_decay=c["weight_decay"],
                       alpha=c.get("grokadamw_alpha", 0.98),
                       gamma=c.get("grokadamw_gamma", 0.1),
                       kappa=c.get("grokadamw_kappa", 0.1),
                       lamb=c.get("grokadamw_lamb", 2.0),
                       grad_clip=c.get("grokadamw_grad_clip", 1.0))


# CONVERTED cells (state-gate clean → production-registered). lion's exp_avg cold-
# starts at ZEROS (lion.py), matching the kernel's zero-init state cache, so its tail
# AND state match the real optimizer exactly. adamw is included as the REGRESSION
# guard for the OptId-generic launcher refactor: the production adamw path now flows
# through the same opt_id-switched launcher (opt_id=0) with the unconditional
# ema/psi-net binding — this re-validates that change against the real AdamW (params
# AND m/v state), which the standalone tc_train_step gates (13/13+5, 21/21) do NOT
# cover (they call the cell's own pybind, not the launcher TU dispatch.cpp uses).
_CELLS = {
    "adamw/decoder": dict(model="decoder", opt="adamw", factory=_adamw_factory),
    "adamw/vit":     dict(model="vit",     opt="adamw", factory=_adamw_factory),
    "lion/decoder":  dict(model="decoder", opt="lion", factory=_lion_factory),
    "lion/vit":      dict(model="vit",     opt="lion", factory=_lion_factory),
}

# BLOCKED cells — kept here (commented) with the state-gate evidence that blocks them,
# so the reason is reproducible. NOT registered in _FUSED_L3_REAL; the gate's
# has_l3_real precondition would (correctly) refuse to run them as "converted".
#  * grokfast/{decoder,vit}: state-gate ema rel ≈ 0.98 — the optimizer cold-starts
#    ema=grad0 (grokfast.py:143) while the TC P3 cache inits ema=0. PARAMS match at
#    step 1 (Adam→sign masks it) but the ema/m/v STATE diverges. Needs a step==1
#    ema=grad init in the shared TC P3 (cold-start-STAGED).
#  * grokadamw/decoder: per-tensor beta1 = beta1·(1-gamma)^i (gamma=0.1) is not
#    representable in one global FusedScalars. Inert at step 1, diverges multi-step.
_BLOCKED_EVIDENCE = {
    "grokfast/decoder":  dict(model="decoder", opt="grokfast", factory=_grokfast_factory),
    "grokfast/vit":      dict(model="vit",     opt="grokfast", factory=_grokfast_factory),
    "grokadamw/decoder": dict(model="decoder", opt="grokadamw", factory=_grokadamw_factory),
}


def _build_cell(model, seed=42):
    """Build the live race model + one real batch + the L3-REAL flat layout.

    DETERMINISM (load-bearing for gate 2): build_model does NOT reseed from
    c["seed"] internally — it draws from the global RNG — so the A/A/A determinism
    check MUST reseed here, or each rebuild yields different random init weights and
    the gate would compare runs with DIFFERENT inputs (a test bug, not kernel
    non-determinism). We torch.manual_seed(seed) immediately before build_model so
    every _build_cell(model, seed) returns byte-identical initial params."""
    g = _g()
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": model, "use_amp": False, "use_fused": True,
              "compile_model": False, "matmul_precision": "bf16", "seed": seed,
              "frac_train": 0.5, "val_ratio": 0.10, "p": 97})
    dev = torch.device("cuda")
    data = tuple(d.to(dev) for d in g.make_data_for_task(c, seed))
    torch.manual_seed(seed)            # pin build_model's global-RNG draw
    torch.cuda.manual_seed_all(seed)
    m = g.build_model(c, dev)
    return g, c, m, data, dev


def run_cell_gate(cell_key, verbose=True):
    """Run gates (1)+(2) for one converted cell. Returns (ok, detail)."""
    from grokking_optimizers.dispatch import (has_l3_real, gemm_impl_for_cell,
                                              canonicalize_model, fused_train_step)
    spec = _CELLS[cell_key]
    model, opt = spec["model"], spec["opt"]
    g, c, m, data, dev = _build_cell(model)
    canon = canonicalize_model(model)
    # Precondition: the cell must actually be wired L3-TC (wgmma) — else the gate is
    # meaningless. has_l3_real + gemm_impl_for_cell are the production gates.
    assert has_l3_real(canon, opt), f"{cell_key}: not L3-REAL (wire it first)"
    eng = gemm_impl_for_cell(canon, opt, "bf16")
    assert eng == "wgmma", f"{cell_key}: bf16 engine is {eng!r}, expected wgmma"

    tx, ty = data[0], data[1]
    # A live optimizer carrying the EXACT production hyperparameters (drives the
    # kernel's scalars via _opt_scalars_from inside fused_train_step).
    opt_obj = spec["factory"](g, m.parameters(), c)

    # Snapshot the initial flat params (named_parameters order) BEFORE the step.
    named0 = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named0)
    p_before = torch.empty(total, device=dev)
    off = 0
    for _, p in named0:
        n = p.numel(); p_before[off:off + n].copy_(p.data.reshape(-1)); off += n

    # ── (1) megakernel-vs-REAL-EAGER single-step: run ONE production L3-TC step (the
    # SAME route the race takes), capturing the TC-reduced grad + the post-step
    # params. The state cache starts at zero (m=v=extra=0).
    cache = {}
    loss, grad = fused_train_step(canon, opt, m, opt_obj, tx, ty,
                                  state_cache=cache, step=1, return_grad=True,
                                  gemm_impl="wgmma")
    # params after the kernel step (named order → flat).
    p_after = torch.empty(total, device=dev)
    off = 0
    for _, p in named0:
        n = p.numel(); p_after[off:off + n].copy_(p.data.reshape(-1)); off += n

    # REFERENCE = the REAL eager optimizer (not a header re-transcription). Build a
    # FRESH model with the SAME init (same seed), INJECT the kernel's TC-reduced grad
    # into each p.grad (so both sides consume the BYTE-IDENTICAL gradient — the bf16-TC
    # grad — isolating the optimizer TAIL math), then run ONE real opt.step() from
    # zero state. A mechanism the real optimizer has but apply_optimizer<Opt> drops
    # (grad_clip, per-layer schedule, adaptive alpha) SHOWS UP here as a param mismatch.
    g_r, c_r, m_ref, data_r, dev_r = _build_cell(model)
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    # confirm the fresh model's init matches p_before (same seed) — the reference is
    # only valid if it starts from the kernel's input params.
    p_ref_before = torch.cat([p.data.reshape(-1) for _, p in named_ref])
    assert torch.equal(p_ref_before, p_before), \
        f"{cell_key}: reference model init != kernel init (seed mismatch)"
    opt_ref = spec["factory"](g_r, m_ref.parameters(), c_r)
    # Scatter the kernel's flat TC grad into the reference params' .grad (layout order).
    off = 0
    for _, p in named_ref:
        n = p.numel()
        p.grad = grad[off:off + n].reshape(p.shape).clone()
        off += n
    opt_ref.step()
    p_ref = torch.cat([p.data.reshape(-1) for _, p in named_ref])

    # The kernel and the real optimizer consumed the SAME grad; the only legitimate
    # difference is fp32 rounding order in the elementwise tail. A DROPPED mechanism
    # (e.g. grad_clip scaling every grad, or a per-layer beta1) makes this large.
    abs_err = (p_after - p_ref).abs().max().item()
    denom = p_ref.abs().max().item() + 1e-30
    rel_err = abs_err / denom
    param_ok = rel_err < 1e-4   # fp32 reorder tol; a dropped mechanism blows past this
    if verbose:
        print(f"  (1a) params vs REAL eager: max|Δp|={abs_err:.3e} rel={rel_err:.3e} "
              f"(tol 1e-4) {'OK' if param_ok else 'FAIL'}  loss={loss:.5f}", flush=True)

    # ── (1b) STATE vs REAL eager — THE DECISIVE CHECK. At step 1 every Adam-family
    # tail collapses to ≈sign(g_amp) in the PARAM update (m/bc1=g_amp, sqrt(v/bc2)=
    # |g_amp|), so (1a) is BLIND to per-layer beta1 (gamma), amplification magnitude,
    # and STATE INITIALIZATION — all of which live in the optimizer state, not the
    # sign. So we compare the kernel's post-step state slices [m|v|ema] (in the cache
    # buffer) against the REAL optimizer's opt.state[p]. A state mismatch == the kernel
    # is NOT running the real optimizer (e.g. grokfast's ema cold-start: real inits
    # ema=g0, the kernel inits ema=0 → the ema slice diverges 50x even when params
    # match). This is what makes the single-step verdict mean "runs the real optimizer".
    kstate = cache[canon]["state"]              # flat [m|v|extra] (3*total)+loss
    k_m = kstate[0:total]; k_v = kstate[total:2 * total]; k_ema = kstate[2 * total:3 * total]
    # Gather the real optimizer's per-parameter state into flat buffers (layout order).
    r_m = torch.zeros(total, device=dev); r_v = torch.zeros(total, device=dev)
    r_ema = torch.zeros(total, device=dev)
    have_v = have_ema = False
    off = 0
    for _, p in named_ref:
        n = p.numel(); stt = opt_ref.state.get(p, {})
        if "exp_avg" in stt:
            r_m[off:off + n].copy_(stt["exp_avg"].reshape(-1))
        if "exp_avg_sq" in stt:
            r_v[off:off + n].copy_(stt["exp_avg_sq"].reshape(-1)); have_v = True
        if "ema" in stt:
            r_ema[off:off + n].copy_(stt["ema"].reshape(-1)); have_ema = True
        off += n
    def _rel(a, b):
        d = (a - b).abs().max().item(); return d / (b.abs().max().item() + 1e-30)
    m_rel = _rel(k_m, r_m)
    v_rel = _rel(k_v, r_v) if have_v else 0.0
    ema_rel = _rel(k_ema, r_ema) if have_ema else None
    # The kernel and real opt consumed the SAME grad; their state must match to fp32
    # reorder tol. ema only checked when the optimizer HAS an ema buffer.
    state_ok = (m_rel < 1e-4 and v_rel < 1e-4
                and (ema_rel is None or ema_rel < 1e-4))
    if verbose:
        ema_s = f"ema={ema_rel:.3e}" if ema_rel is not None else "ema=n/a"
        print(f"  (1b) STATE vs REAL eager: m={m_rel:.3e} v={v_rel:.3e} {ema_s} "
              f"(tol 1e-4) {'OK' if state_ok else 'FAIL — kernel state != real optimizer'}",
              flush=True)
    tail_ok = param_ok and state_ok

    # ── (2) A/A/A determinism: re-run the production step 3× from the SAME init,
    # fresh state each time → loss + grad + params must be BIT-identical.
    def _one():
        g2, c2, m2, data2, dev2 = _build_cell(model)
        tx2, ty2 = data2[0], data2[1]
        opt2 = spec["factory"](g2, m2.parameters(), c2)
        cc = {}
        L, G = fused_train_step(canon, opt, m2, opt2, tx2, ty2, state_cache=cc,
                                step=1, return_grad=True, gemm_impl="wgmma")
        P = torch.empty(total, device=dev2)
        o = 0
        for _, p in m2.named_parameters():
            if p.requires_grad:
                k = p.numel(); P[o:o + k].copy_(p.data.reshape(-1)); o += k
        return L, G.clone(), P
    L1, G1, P1 = _one(); L2, G2, P2 = _one(); L3, G3, P3 = _one()
    det_ok = (L1 == L2 == L3 and torch.equal(G1, G2) and torch.equal(G2, G3)
              and torch.equal(P1, P2) and torch.equal(P2, P3))
    if verbose:
        print(f"  (2) A/A/A determinism: loss {L1:.6f}/{L2:.6f}/{L3:.6f}  "
              f"grad-eq={torch.equal(G1,G2) and torch.equal(G2,G3)}  "
              f"param-eq={torch.equal(P1,P2) and torch.equal(P2,P3)}  "
              f"{'OK' if det_ok else 'FAIL'}", flush=True)
    ok = tail_ok and det_ok
    return ok, dict(rel_err=rel_err, loss=loss, det=det_ok)


def _gpu_ready():
    try:
        from grokking_optimizers.dispatch import get_ops
        return torch.cuda.is_available() and hasattr(get_ops(), "fused_step")
    except Exception:
        return False


# ── pytest entry (one test per converted cell; skips cleanly off-GPU) ──
import pytest  # noqa: E402


@pytest.mark.hw
@pytest.mark.skipif(not _gpu_ready(), reason="needs built extension on GPU")
@pytest.mark.parametrize("cell", list(_CELLS))
def test_l3tc_cell_gate(cell):
    ok, detail = run_cell_gate(cell, verbose=True)
    assert ok, f"{cell} L3-TC gate failed: {detail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="", help="single cell e.g. lion/decoder")
    args = ap.parse_args()
    if not _gpu_ready():
        print("SKIP: no built extension / GPU", flush=True)
        return
    cells = [args.cell] if args.cell else list(_CELLS)
    n_ok = 0
    for cell in cells:
        print(f"[l3tc-gate] {cell}", flush=True)
        ok, _ = run_cell_gate(cell, verbose=True)
        print(f"  => {'PASS' if ok else 'FAIL'}\n", flush=True)
        n_ok += int(ok)
    print(f"[l3tc-gate] {n_ok}/{len(cells)} cells passed", flush=True)
    sys.exit(0 if n_ok == len(cells) else 1)


if __name__ == "__main__":
    main()
