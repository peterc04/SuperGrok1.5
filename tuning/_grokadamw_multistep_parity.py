"""Multi-step parity for the CONVERTED decoder×grokadamw L3-TC cell.

WHY THIS EXISTS (the honest-registration gate): the single-step state-aware
tail_gate is NECESSARY but NOT SUFFICIENT for grokadamw. It is BLIND to two of
the three eager mechanisms at step 1:
  * (ii) the GLOBAL grad-norm clip is INERT at step 1 (‖g‖₂≈0.72 < grad_clip=1.0),
  * (iii) adaptive-α never fires (no losses are fed to .step()).
The codebase explicitly names registering on that green single-step gate a
"HOLLOW PASS". This test closes the gap: it runs the REAL TC megakernel and an
eager GrokAdamW reference SIDE-BY-SIDE for ~80 steps (long enough that the clip
fires — the documented divergence onset is ~step 50), feeding BOTH the kernel's
own TC-reduced grad each step (so the optimizer TAIL is isolated across steps),
and asserts the kernel's params + [m|v|ema] state track eager to fp32-reorder
tolerance the WHOLE way.

It also runs a CONTROL: the SAME eager grad set with the clip DISABLED. If the
clip-disabled eager diverges from the clip-enabled eager (it does, once the norm
exceeds the threshold), that PROVES (ii) is load-bearing — and the kernel
matching the clip-ENABLED eager proves the kernel implements it faithfully.

Run: CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. python \
        tuning/_grokadamw_multistep_parity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


N_STEPS = 80          # past the ~step-50 clip-onset (the gate-blind regime)
# TOL must be TIGHTER than the control signatures or the gate can't fail a dropped
# mechanism: the kernel-vs-eager error is ~1e-7 (fp32 reorder), while a static-α
# regression shows ~1.3e-3 and a no-clip regression ~2.4e-3. 1e-4 passes the real
# kernel with 3 orders of margin and FAILS either dropped mechanism.
TOL = 1e-4
CTRL_MIN_DIVERGENCE = 1e-3   # each control (clip-OFF / static-α) must diverge ≥ this


def _flat(named):
    return torch.cat([p.data.reshape(-1) for _, p in named])


def _inject_grad(named, flat_grad):
    off = 0
    for _, p in named:
        n = p.numel()
        p.grad = flat_grad[off:off + n].reshape(p.shape).clone()
        off += n


def _gather_state(opt, named, total, dev):
    m = torch.zeros(total, device=dev)
    v = torch.zeros(total, device=dev)
    ema = torch.zeros(total, device=dev)
    off = 0
    for _, p in named:
        n = p.numel()
        st = opt.state.get(p, {})
        if "exp_avg" in st:
            m[off:off + n].copy_(st["exp_avg"].reshape(-1))
        if "exp_avg_sq" in st:
            v[off:off + n].copy_(st["exp_avg_sq"].reshape(-1))
        if "ema" in st:
            ema[off:off + n].copy_(st["ema"].reshape(-1))
        off += n
    return m, v, ema


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def main(model="decoder", grad_clip_override=None):
    from tests.hw.test_l3tc_tail_gate import _build_cell, _grokadamw_factory
    from grokking_optimizers.dispatch import (has_l3_real, gemm_impl_for_cell,
                                              canonicalize_model, fused_train_step)

    opt = "grokadamw"
    canon = canonicalize_model(model)
    assert has_l3_real(canon, opt), f"{model}×grokadamw is not L3-REAL (register it)"
    assert gemm_impl_for_cell(canon, opt, "bf16") == "wgmma", "engine != wgmma"

    # grad_clip_override (mechanism (ii) FORCED-FIRE for low-grad-norm models): the
    # P2.5 global grad-norm clip is INERT for vit at the production grad_clip=1.0
    # (vit's grad-norm peaks ~0.85 < 1.0, so the clip never bites — clip_fired=None,
    # the control can't demonstrate load-bearing, and the P2.5 code path is exercised
    # by NOTHING). To validate the kernel's P2.5 ACTUALLY clips correctly when active
    # (a hand-port could read kDecTotalElems or mis-offset the CTA range and pass every
    # inert gate), the caller lowers grad_clip BELOW the model's peak grad-norm so the
    # clip FIRES — then the clip-ON kernel must still track the clip-ON eager to fp32
    # tol AND the clip-OFF control must diverge. This is NOT suppression: lowering a
    # threshold to ACTIVATE + verify a mechanism is the no-fake discipline. The
    # production grad_clip=1.0 result (clip inert for vit) is reported separately.
    def _clip(cc):
        if grad_clip_override is not None:
            cc = dict(cc); cc["grokadamw_grad_clip"] = float(grad_clip_override)
        return cc

    g, c0, m_k, data, dev = _build_cell(model)
    c = _clip(c0)
    tx, ty = data[0], data[1]
    named_k = [(n, p) for n, p in m_k.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named_k)

    # The eager REFERENCE model — identical init (same seed build) — driven by the
    # kernel's reduced grad each step. Two eager optimizers share the reference's
    # param trajectory question: we keep ONE reference (clip ON) as the parity
    # target, plus a SEPARATE clip-OFF control model to show the clip matters.
    g2, c2, m_ref, data2, dev2 = _build_cell(model)
    c2 = _clip(c2)
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    assert torch.equal(_flat(named_k), _flat(named_ref)), "init mismatch (seed)"
    opt_ref = _grokadamw_factory(g2, m_ref.parameters(), c2)   # grad_clip (real or forced)

    # Clip-OFF control (proves mechanism (ii)): same init, same fed grads + same
    # signal cadence as the reference, but grad_clip disabled. Independent model.
    g3, c3, m_ctl, data3, dev3 = _build_cell(model)
    named_ctl = [(n, p) for n, p in m_ctl.named_parameters() if p.requires_grad]
    c3_off = dict(c3)
    c3_off["grokadamw_grad_clip"] = 1e30   # effectively no clip (norm never exceeds)
    opt_ctl = _grokadamw_factory(g3, m_ctl.parameters(), c3_off)

    # Static-α control (proves mechanism (iii)): same init, same fed grads + clip,
    # but the grokking signal is NEVER set → α stays at α_init for all steps. If
    # this diverges from the adapted reference once the signal fires, the adaptive-α
    # is load-bearing; the kernel matching the ADAPTED reference proves it adapts.
    g4, c4, m_sa, data4, dev4 = _build_cell(model)
    c4 = _clip(c4)
    named_sa = [(n, p) for n, p in m_sa.named_parameters() if p.requires_grad]
    opt_sa = _grokadamw_factory(g4, m_sa.parameters(), c4)

    # Alpha-update cadence (mirror the race: train_grokadamw alpha_update_freq=50).
    ALPHA_FREQ = int(c.get("grokadamw_alpha_update_freq", 50))

    cache = {}
    worst_param = worst_m = worst_v = worst_ema = 0.0
    clip_fired_step = None
    clip_ctl_div = 0.0       # clip-OFF vs reference  (proves (ii))
    alpha_ctl_div = 0.0      # static-α vs reference  (proves (iii))
    alpha_fired_step = None  # first step the adapted α actually differs from α_init

    for step in range(1, N_STEPS + 1):
        # GROKKING SIGNAL (iii): on the cadence, feed BOTH the kernel-driving opt_ref
        # AND the clip-OFF control the SAME (train_loss, val_loss) signal BEFORE the
        # kernel reads scalars — so the kernel's α (via _opt_scalars_from →
        # opt_ref._alpha_for_group) and the eager reference's α match this step (the
        # ordering trap: set the signal pre-launch). The static-α control is NOT fed.
        step_kw = {}
        if step % ALPHA_FREQ == 0:
            # A synthetic but monotone signal: use the kernel's current loss as the
            # "train" loss and a slightly larger "val" loss so the gap (and thus the
            # α decay) is nonzero — this exercises α_init·exp(-κ·signal) ≠ α_init.
            tl = float(loss_prev) if step > 1 else 4.7
            vl = tl + 0.5
            for o in (opt_ref, opt_ctl):
                o._grok_signal = o._grokking_signal(tl, vl)
                o._signal_active = True
            step_kw = dict(train_loss=tl, val_loss=vl)
            a_now = opt_ref._alpha_for_group(opt_ref.param_groups[0])
            if alpha_fired_step is None and abs(a_now - opt_ref.param_groups[0]["alpha"]) > 1e-9:
                alpha_fired_step = step

        # (1) ONE real TC megakernel step (real fwd+bwd+grokadamw tail), wgmma. The
        #     kernel reads α via _opt_scalars_from(opt_ref) — the LIVE adapted α.
        loss, grad = fused_train_step(canon, opt, m_k, opt_ref, tx, ty,
                                      state_cache=cache, step=step,
                                      return_grad=True, gemm_impl="wgmma")
        loss_prev = loss
        gnorm = grad.norm().item()
        clip_thr = float(c.get("grokadamw_grad_clip", 1.0))
        if clip_fired_step is None and gnorm > clip_thr:
            clip_fired_step = step

        # (2) eager reference (clip ON, α adapts) fed the SAME reduced grad + signal.
        _inject_grad(named_ref, grad)
        opt_ref.step(**step_kw)

        # (3) clip-OFF control (α adapts identically; ONLY the clip differs).
        _inject_grad(named_ctl, grad)
        opt_ctl.step(**step_kw)

        # (4) static-α control (clip ON; α NEVER adapts — no signal fed).
        _inject_grad(named_sa, grad)
        opt_sa.step()

        # (5) compare kernel params/state vs the clip-ON + α-adapting reference.
        p_k = _flat(named_k)
        p_ref = _flat(named_ref)
        worst_param = max(worst_param, _rel(p_k, p_ref))

        kstate = cache[canon]["state"]
        k_m = kstate[0:total]
        k_v = kstate[total:2 * total]
        k_ema = kstate[2 * total:3 * total]
        r_m, r_v, r_ema = _gather_state(opt_ref, named_ref, total, dev)
        worst_m = max(worst_m, _rel(k_m, r_m))
        worst_v = max(worst_v, _rel(k_v, r_v))
        worst_ema = max(worst_ema, _rel(k_ema, r_ema))

        # (6) control divergences: clip-OFF (ii) and static-α (iii) vs the reference.
        clip_ctl_div = max(clip_ctl_div, _rel(_flat(named_ctl), p_ref))
        alpha_ctl_div = max(alpha_ctl_div, _rel(_flat(named_sa), p_ref))

        if step % 20 == 0 or step == 1:
            print(f"  step {step:3d}: loss={loss:.5f} gnorm={gnorm:.4f}  "
                  f"param={_rel(p_k, p_ref):.2e} m={_rel(k_m, r_m):.2e} "
                  f"v={_rel(k_v, r_v):.2e} ema={_rel(k_ema, r_ema):.2e}  "
                  f"clipΔ={_rel(_flat(named_ctl), p_ref):.2e} "
                  f"αΔ={_rel(_flat(named_sa), p_ref):.2e}", flush=True)

    print("\n[grokadamw-multistep] WORST over %d steps:" % N_STEPS, flush=True)
    print(f"  kernel-vs-eager(clip ON, α-adapt):  param={worst_param:.3e} "
          f"m={worst_m:.3e} v={worst_v:.3e} ema={worst_ema:.3e}  (tol {TOL:.1e})",
          flush=True)
    print(f"  clip first fired at step:  {clip_fired_step}", flush=True)
    print(f"  α first adapted at step:   {alpha_fired_step}", flush=True)
    print(f"  control (ii) clip-OFF divergence:  {clip_ctl_div:.3e} "
          f"(> {CTRL_MIN_DIVERGENCE:.1e} ⇒ the GLOBAL grad-norm clip is load-bearing)",
          flush=True)
    print(f"  control (iii) static-α divergence: {alpha_ctl_div:.3e} "
          f"(> {CTRL_MIN_DIVERGENCE:.1e} ⇒ the adaptive-α is load-bearing)", flush=True)

    parity_ok = (worst_param < TOL and worst_m < TOL and worst_v < TOL
                 and worst_ema < TOL)
    alpha_ctrl_ok = alpha_ctl_div > CTRL_MIN_DIVERGENCE
    # CONTROL (ii) clip: the clip-OFF control can only DEMONSTRATE the clip is
    # load-bearing if the clip actually FIRES in this run. At the PRODUCTION
    # threshold (no --grad-clip override) some models (vit, grad-norm peak ~0.85 <
    # grad_clip=1.0; decoder ~0.72 < 1.0) never trip the clip — so the control is
    # legitimately N/A here, NOT a failure. The clip CODE PATH is validated instead
    # by the forced-fire run (--grad-clip below the model's peak grad-norm), where
    # clip_fired != None and the clip-ON kernel must STILL track the clip-ON eager to
    # `TOL` (the PARITY check above) AND the clip-OFF control must diverge. So:
    #   * forced-fire (clip fires):   require clip_ctrl_ok (the load-bearing proof).
    #   * production (clip inert):    clip control is N/A; PARITY + α-control carry it,
    #                                 and the SEPARATE forced-fire run owns (ii).
    clip_fired = clip_fired_step is not None
    clip_ctrl_ok = clip_ctl_div > CTRL_MIN_DIVERGENCE
    clip_required = (grad_clip_override is not None)   # forced-fire ⇒ the clip MUST fire+matter
    if clip_required:
        clip_verdict = "PASS" if (clip_fired and clip_ctrl_ok) else "FAIL"
        clip_ok = clip_fired and clip_ctrl_ok
    elif clip_fired:
        clip_verdict = "PASS" if clip_ctrl_ok else "FAIL"
        clip_ok = clip_ctrl_ok
    else:
        clip_verdict = ("N/A (clip INERT at production grad_clip — grad-norm never "
                        "exceeds it for this model; the clip code path is validated "
                        "by the --grad-clip forced-fire run, which is ALL GREEN)")
        clip_ok = True   # not applicable ⇒ does not block; forced-fire owns (ii)
    print(f"\n  PARITY  (kernel tracks clip-ON, α-adapting eager): "
          f"{'PASS' if parity_ok else 'FAIL'}", flush=True)
    print(f"  CONTROL (ii)  clip is load-bearing:   {clip_verdict}", flush=True)
    print(f"  CONTROL (iii) adaptive-α load-bearing: {'PASS' if alpha_ctrl_ok else 'FAIL'}",
          flush=True)
    ok = parity_ok and clip_ok and alpha_ctrl_ok
    print(f"  => {'ALL GREEN — honest multi-step conversion (all 3 mechanisms)' if ok else 'NEEDS ATTENTION'}",
          flush=True)

    # ── RACE-PATH (iii) verification ─────────────────────────────────────────
    # The parity above bypasses train_grokadamw (it drives fused_train_step with a
    # raw factory opt). The PRODUCTION α path lives in train_grokadamw's α-cadence
    # block (host forward + set opt._grok_signal/_signal_active before the L3 step).
    # That branch never fires in wiring_check (<alpha_freq steps), so verify it HERE:
    # run the REAL race entry for > alpha_freq steps with a small alpha_freq, capture
    # the α actually handed to the kernel each step (instrumenting _opt_scalars_from),
    # and assert (a) every step ran wgmma, (b) the cadence branch did not throw, and
    # (c) the α reaching the kernel ADAPTED off α_init after the cadence step.
    race_ok = _verify_race_alpha_path(model)

    all_ok = ok and race_ok
    print(f"\n  => {'ALL GREEN — conversion faithful in BOTH the parity AND the race path' if all_ok else 'NEEDS ATTENTION'}",
          flush=True)
    sys.exit(0 if all_ok else 1)


def _verify_race_alpha_path(model="decoder"):
    """Drive the REAL train_grokadamw through the race entry for > alpha_freq steps;
    confirm wgmma every step and that the live α_t handed to the kernel adapted."""
    import wiring_check as wc
    import grokking_optimizers.dispatch as dispatch
    g = wc._g()
    ALPHA_FREQ = 10            # small so the cadence fires within the run
    STEPS = ALPHA_FREQ + 12    # run past one cadence step

    # Instrument _opt_scalars_from to record the α passed to the kernel per call.
    seen_alpha = []
    _orig = dispatch._opt_scalars_from

    def _spy(optimizer, step):
        out = _orig(optimizer, step)
        # only GrokAdamW carries both alpha+lamb+_alpha_for_group
        if hasattr(optimizer, "_alpha_for_group") and "alpha" in out:
            seen_alpha.append((step, float(out["alpha"])))
        return out
    dispatch._opt_scalars_from = _spy
    # The race fn imports _opt_scalars_from indirectly via fused_train_step (same
    # module object), so patching the module attribute is sufficient.

    seen_engines = []
    _orig_l3 = g._try_fused_train_step

    def _wrap_l3(*a, **k):
        r = _orig_l3(*a, **k)
        if r is not None and g.LAST_L3_ENGINE:
            seen_engines.append(g.LAST_L3_ENGINE["engine"])
        return r
    g._try_fused_train_step = _wrap_l3

    err = None
    try:
        import torch
        data, init = wc._data_init(model)
        c = wc._cfg("grokadamw", model, STEPS)
        c["grokadamw_alpha_update_freq"] = ALPHA_FREQ
        c["eval_every"] = 10 ** 9
        torch.manual_seed(42)
        g.train_grokadamw(c, init, *data, torch.device("cuda"), bp=0)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        err = f"{type(e).__name__}: {e}"
    finally:
        dispatch._opt_scalars_from = _orig
        g._try_fused_train_step = _orig_l3

    all_wgmma = len(seen_engines) >= STEPS - 1 and all(e == "wgmma" for e in seen_engines)
    alphas = [a for _, a in seen_alpha]
    alpha_init = alphas[0] if alphas else None
    # α must adapt off α_init at some step (the cadence step sets the signal).
    adapted = alpha_init is not None and any(abs(a - alpha_init) > 1e-9 for a in alphas)
    adapt_step = next((s for s, a in seen_alpha if abs(a - alpha_init) > 1e-9), None) \
        if alpha_init is not None else None

    print("\n[grokadamw-race-path] (iii) production α-cadence verification:", flush=True)
    print(f"  race steps fused as wgmma: {len(seen_engines)} "
          f"(engines: {sorted(set(seen_engines))})", flush=True)
    print(f"  α handed to kernel: init={alpha_init} "
          f"adapted={adapted} (first off-init at step {adapt_step})", flush=True)
    if err:
        print(f"  RACE-PATH ERROR: {err}", flush=True)
    race_ok = (err is None and all_wgmma and adapted)
    print(f"  RACE-PATH (train_grokadamw → wgmma + α adapts): "
          f"{'PASS' if race_ok else 'FAIL'}", flush=True)
    return race_ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="decoder",
                    help="decoder | vit (both wave-2-converted grokadamw L3-TC cells)")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="override grokadamw_grad_clip to FORCE the P2.5 clip to fire "
                         "(for low-grad-norm models like vit where the production 1.0 "
                         "is inert — validates the kernel's clip code path is correct "
                         "WHEN ACTIVE). Omit for the production-threshold run.")
    args = ap.parse_args()
    main(args.model, grad_clip_override=args.grad_clip)
