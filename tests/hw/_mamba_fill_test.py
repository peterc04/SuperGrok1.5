#!/usr/bin/env python
"""Read-before-write discriminator for the looksam mamba TC race.

Fills the cached TC workspace with a SENTINEL before each launch. If the kernel
output depends on the fill value, a region is read-before-written. Then narrows
the fill to one region at a time to localize the unwritten buffer.

Usage: CUDA_MPS_PIPE_DIRECTORY=/nonexistent python tests/hw/_mamba_fill_test.py [--sam 1|0] [--ncta 8]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/nonexistent")
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
import torch
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "probe", os.path.join(os.path.dirname(__file__), "_mamba_race_probe.py"))
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)


def looksam_once(mod, params0, tokens, targets, ncta, sam, fill=None, region=None):
    total = int(mod.TOTAL)
    dev = "cuda"
    if fill is None:
        mod.set_ws_fill(False, 0.0)
    elif region is None:
        mod.set_ws_fill(True, float(fill))
    else:
        mod.set_ws_fill(True, float(fill), int(region[0]), int(region[1]))
    params = params0.clone()
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    loss, grad, samdir = mod.looksam_step(
        params, tokens, targets, state,
        1e-3, 0.9, 0.98, 1e-8, 0.0, 1 - 0.9, 1 - 0.98, 0.05, 0.7, float(sam), 1, ncta)
    return float(loss), grad.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam", type=float, default=1.0)
    ap.add_argument("--ncta", type=int, default=8)
    ap.add_argument("--B", type=int, default=64)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 9:
        print("SKIP: no sm_90 GPU"); return
    torch.backends.cuda.matmul.allow_tf32 = False
    from tests.hw.mamba_oracle import VOCAB, P_HEAD, SEQ
    mod = P.build()
    dev = "cuda"
    g = torch.Generator().manual_seed(9)
    tokens = torch.randint(0, VOCAB, (args.B, SEQ), generator=g).to(dev).to(torch.int32).contiguous()
    targets = torch.randint(0, P_HEAD, (args.B,), generator=g).to(dev).to(torch.int32).contiguous()
    params0 = P.flat(P.eager_named(seed=3)).to(dev).contiguous()

    # Effective nCTA the launcher uses (for ws_layout) — mirror eff via a probe call:
    # ncta arg 0 means full; ws_layout needs the actual nCTA. Use the arg if >0 else
    # query n_sms.
    nCTA = args.ncta if args.ncta > 0 else torch.cuda.get_device_properties(0).multi_processor_count
    lay = mod.ws_layout(args.B, nCTA)

    print(f"== FILL TEST looksam sam={args.sam} ncta={args.ncta} (nCTA={nCTA}) B={args.B} ==")
    # 1) Whole-workspace fill sweep: does output depend on fill value?
    base_l, base_g = looksam_once(mod, params0, tokens, targets, args.ncta, args.sam, fill=None)
    print(f"  no-fill (cached residue): loss={base_l:.8f} grad_absmax={base_g.abs().max().item():.4e}")
    refs = {}
    for fv in (0.0, 1.0, -7.0, 1e30):
        L, G = looksam_once(mod, params0, tokens, targets, args.ncta, args.sam, fill=fv)
        nan = bool(torch.isnan(G).any().item())
        refs[fv] = (L, G)
        print(f"  fill={fv:>10}: loss={L:.8f} grad_absmax={G.abs().max().item():.4e} grad_nan={nan}")
    # Compare fill=0 vs fill=1e30
    dl = abs(refs[0.0][0] - refs[1e30][0])
    dg = (refs[0.0][1] - refs[1e30][1]).abs()
    dg = dg[~torch.isnan(dg)].max().item() if (~torch.isnan(dg)).any() else float('nan')
    depends = (dl != 0.0 or dg != 0.0 or torch.isnan(refs[1e30][1]).any().item() != torch.isnan(refs[0.0][1]).any().item())
    print(f"  -> output depends on fill: {depends}  (loss Δ={dl:.3e}, grad Δ={dg:.3e})")
    if not depends:
        print("  CONCLUSION: NOT read-before-write (fill-invariant) → timing/barrier race.")
        return

    print("  CONCLUSION: READ-BEFORE-WRITE confirmed. Localizing region (fill=1e30 each):")
    # 2) Per-region fill: fill ONLY one region with a sentinel; compare to clean(0-fill).
    clean_l, clean_g = refs[0.0]
    order = ["acts", "scratch", "part", "loss", "lossout", "dwpart", "optreduce", "sam_backup", "sam_grad"]
    span = {
        "acts": (lay["acts_begin"], lay["acts_end"]),
        "scratch": (lay["scratch_begin"], lay["scratch_end"]),
        "part": (lay["part_begin"], lay["part_end"]),
        "loss": (lay["loss_begin"], lay["loss_end"]),
        "lossout": (lay["lossout_begin"], lay["lossout_end"]),
        "dwpart": (lay["lossout_end"], lay["dwpart_end"]),
        "optreduce": (lay["dwpart_end"], lay["optreduce_end"]),
        "sam_backup": (lay["sam_backup_begin"], lay["sam_backup_end"]),
        "sam_grad": (lay["sam_grad_begin"], lay["sam_grad_end"]),
    }
    for name in order:
        a, b = span[name]
        if b <= a:
            print(f"    region {name:11} [{a},{b}) EMPTY — skip"); continue
        L, G = looksam_once(mod, params0, tokens, targets, args.ncta, args.sam, fill=1e30, region=(a, b))
        dl = abs(L - clean_l)
        diff = (G - clean_g).abs()
        nan = bool(torch.isnan(G).any().item())
        dgv = diff[~torch.isnan(diff)].max().item() if (~torch.isnan(diff)).any() else float('nan')
        flag = "  <== READ-BEFORE-WRITE" if (dl != 0.0 or (dgv == dgv and dgv != 0.0) or nan) else ""
        print(f"    region {name:11} [{a},{b}): loss Δ={dl:.3e} grad Δ={dgv:.3e} nan={nan}{flag}")


if __name__ == "__main__":
    main()
