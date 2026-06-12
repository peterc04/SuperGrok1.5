"""Multi-step race probe: confirm SG11/SG15 now run as REAL SG on the L3 path.

Drives the REAL train_supergrok / train_supergrok15 (the exact edited code paths)
for a few hundred steps with the production config (DEFAULT_CONFIG + OPTIMIZER_CONFIGS,
bf16, decoder, fused ON). Instruments:
  * meta_net.rescale  — must MOVE off its init 0 (degenerate ⇒ mu≡0 ⇒ ≈AdamW).
  * ||mu|| (||meta correction||) at each host meta/bilevel step — must be nonzero.
  * which path actually executed (LAST_L3_ENGINE) — must be the L3 wgmma kernel.
  * train/val/test acc trajectory — the documented REAL-SG memorize-then-collapse.

Honest: the goal is FIDELITY (real algorithm), not grokking. We REPORT the trajectory.
"""
import sys, copy, numpy as np, torch
import grokking_race_v2 as R

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
SEED = 42
dev = torch.device("cuda")

def build_cfg(opt_name):
    base = dict(R.DEFAULT_CONFIG)
    base["model_type"] = "decoder"
    cfg = R._merge(base, R.OPTIMIZER_CONFIGS[opt_name])
    cfg["model_type"] = "decoder"
    cfg["frac_train"] = base["frac_train"]; cfg["val_ratio"] = base["val_ratio"]
    cfg["seq_len"] = base["seq_len"]
    cfg["seed"] = SEED
    # Short probe: cap the step budget; keep eval cadence frequent for trajectory.
    cfg["max_steps"] = STEPS
    cfg["early_stop_max_steps"] = STEPS
    cfg["eval_every"] = 25
    cfg["use_fused"] = True            # L3 path ON (the path under test)
    cfg["matmul_precision"] = "bf16"   # bf16 wgmma TC kernel (the L3-REAL carrier)
    return cfg

def run(opt_name, train_fn):
    cfg = build_cfg(opt_name)
    torch.manual_seed(SEED); np.random.seed(SEED)
    tx, ty, vax, vay, tex, tey = R.make_data_for_task(cfg, SEED)
    tx, ty = tx.to(dev), ty.to(dev); vax, vay = vax.to(dev), vay.to(dev); tex, tey = tex.to(dev), tey.to(dev)
    ist = R.get_init_state(dict(cfg), dev)

    # rescale / ||mu|| history, captured directly from the host meta-net forward
    # (the bilevel/meta_step call into meta_net is what the host training drives).
    hist = []  # list of (||mu_correction||, rescale_at_that_forward)
    orig_forward = R.SuperGrok11.__mro__  # noqa (keep ref for clarity)

    # acc trajectory via the eval callback (step, train_acc, val_acc, test_acc)
    traj = []
    cfg["_eval_callback"] = lambda step, ta, va, tea: traj.append((step, ta, va, tea))

    # Wrap the meta-net forward to record ||rescale*MLP|| (== ||mu host-correction||)
    # and the live rescale at each host training forward. We attach AFTER the opt is
    # built; train_fn builds the opt internally, so hook via a one-shot module patch:
    holder = {}
    import grokking_optimizers.optimizers.supergrok11 as sg11mod
    import grokking_optimizers.optimizers.supergrok15 as sg15mod
    NetCls = sg11mod.SharpnessMetaNet if opt_name == "supergrok" else sg15mod.SharpnessMetaNet
    real_fwd = NetCls.forward
    def patched_fwd(self, grad, sharpness):
        out = real_fwd(self, grad, sharpness)
        # correction = out - grad  (== rescale * MLP); ||.|| is the meta contribution
        try:
            corr = (out.detach() - grad.detach())
            hist.append((float(corr.norm().item()), float(self.rescale.detach().item())))
        except Exception:
            pass
        return out
    NetCls.forward = patched_fwd
    try:
        res = train_fn(cfg, ist, tx, ty, vax, vay, tex, tey, dev, 0)
    finally:
        NetCls.forward = real_fwd

    path = R.LAST_L3_ENGINE
    return res, traj, hist, path

def fmt_traj(traj):
    lines = []
    for step, ta, va, tea in traj:
        lines.append(f"    step {step:5d} | train {ta:.3f} | val {va:.3f} | test {tea:.3f}")
    return "\n".join(lines)

def summarize(name, res, traj, hist, path):
    print(f"\n{'='*72}\n{name}: REAL-SG L3 probe ({STEPS} steps)\n{'='*72}")
    print(f"  LAST_L3_ENGINE (path actually executed last step): {path}")
    if hist:
        rescales = [r for _, r in hist]
        mus = [m for m, _ in hist]
        print(f"  host meta/bilevel forwards captured: {len(hist)}")
        print(f"  rescale: first={rescales[0]:+.6e}  last={rescales[-1]:+.6e}  "
              f"min={min(rescales):+.6e}  max={max(rescales):+.6e}")
        print(f"  ||mu (meta correction)||: first={mus[0]:.6e}  last={mus[-1]:.6e}  "
              f"max={max(mus):.6e}")
        moved = abs(rescales[-1]) > 0.0 or any(abs(r) > 0.0 for r in rescales)
        print(f"  >>> rescale MOVED off 0: {moved}  (degenerate-AdamW would be rescale==0 ⇒ mu==0)")
    else:
        print("  WARNING: no host meta/bilevel forwards captured "
              "(meta-net never trained — would be the OLD degenerate behavior).")
    # final live rescale read straight off the optimizer's meta-net is in res? No —
    # res is a TrainResult. Report the acc trajectory (the SG dynamics).
    print(f"  acc trajectory (memorize-then-collapse is the documented REAL-SG result):")
    print(fmt_traj(traj))
    print(f"  final: train_acc={res.final_train_acc:.3f} val_acc={res.final_val_acc:.3f} "
          f"test_acc={res.final_test_acc:.3f}  grokking_step={res.grokking_step}")

if __name__ == "__main__":
    r1, t1, h1, p1 = run("supergrok",   R.train_supergrok)
    summarize("SuperGrok1.1", r1, t1, h1, p1)
    r2, t2, h2, p2 = run("supergrok15", R.train_supergrok15)
    summarize("SuperGrok1.5", r2, t2, h2, p2)
