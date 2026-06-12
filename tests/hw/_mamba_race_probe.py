#!/usr/bin/env python
"""Diagnostic A/A/A probe for the shared mamba-forward race (mamba x prodigy TC).

Runs the Prodigy mamba TC step N times from a BIT-IDENTICAL init and reports
loss spread + per-tensor grad max|delta|. Under compute-sanitizer this wraps a
SINGLE step so racecheck/initcheck can localize the hazard.

Usage:
  CUDA_MPS_PIPE_DIRECTORY=/nonexistent python tests/hw/_mamba_race_probe.py [--steps N] [--nruns K] [--ncta C]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build():
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    from torch.utils.cpp_extension import load
    min_blocks = os.environ.get("SG_PROBE_MIN_BLOCKS", "")  # "" = header default (1); "2" = occ=2
    cuda_flags = [
        "-O3", "-std=c++17", "--expt-relaxed-constexpr",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-gencode=arch=compute_90a,code=compute_90a",
        "-lineinfo",
    ]
    name = "_mamba_prodigy_probe"
    if min_blocks:
        cuda_flags.append(f"-DSG_MBTC_MIN_BLOCKS_PER_SM={min_blocks}")
        name = f"_mamba_prodigy_probe_mb{min_blocks}"
    extra_def = os.environ.get("SG_PROBE_DEFS", "")  # e.g. "SG_MBTC_BYPASS_SCAN=1,SG_MBTC_BYPASS_DW_GEMM=1"
    if extra_def:
        tag = ""
        for d in extra_def.split(","):
            d = d.strip()
            if d:
                cuda_flags.append(f"-D{d}")
                tag += "_" + d.replace("=", "").replace("SG_MBTC_BYPASS_", "bp").lower()
        name = name + tag
    return load(
        name=name,
        sources=[os.path.join(_REPO, "tests", "hw", "_mamba_prodigy_probe.cu")],
        extra_include_paths=[_REPO],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


def eager_named(seed=3):
    from tests.hw.mamba_oracle import mamba_param_spec
    torch.manual_seed(seed)
    named = {}
    for name, shape in mamba_param_spec():
        named[name] = torch.randn(*shape) * 0.02 if len(shape) > 1 else torch.zeros(*shape)
    return named


def flat(named):
    from tests.hw.mamba_oracle import mamba_param_spec
    return torch.cat([named[n].reshape(-1) for n, _ in mamba_param_spec()])


def run(mod, params0, tokens, targets, steps, ncta, opt="prodigy", sam=1.0):
    total = int(mod.TOTAL)
    dev = "cuda"
    params = params0.clone()
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    param_init = params.clone()
    persist = torch.zeros(3, dtype=torch.float32, device=dev)
    last = None
    for step in range(1, steps + 1):
        if opt == "prodigy":
            loss, grad, _ = mod.prodigy_step(
                params, tokens, targets, state, param_init, persist,
                1e-3, 0.9, 0.98, 1e-8, 0.0, 1 - 0.9, 1 - 0.98,
                1e-6, 1.0, 0.98 ** 0.5, step, ncta)
        else:  # looksam
            loss, grad, _ = mod.looksam_step(
                params, tokens, targets, state,
                1e-3, 0.9, 0.98, 1e-8, 0.0, 1 - 0.9, 1 - 0.98,
                0.05, 0.7, sam, step, ncta)
        last = (float(loss.item()), grad.clone())
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--nruns", type=int, default=3)
    ap.add_argument("--ncta", type=int, default=int(os.environ.get("SG_TC_NCTA_CAP", "8")))
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--opt", default="prodigy", choices=["prodigy", "looksam"])
    ap.add_argument("--sam", type=float, default=1.0, help="looksam_sam flag (1=SAM step, 0=SAM-off)")
    args = ap.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 9:
        print("SKIP: no sm_90 GPU"); return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from tests.hw.mamba_oracle import VOCAB, P_HEAD, SEQ
    mod = build()
    dev = "cuda"
    g = torch.Generator().manual_seed(9)
    tokens = torch.randint(0, VOCAB, (args.B, SEQ), generator=g).to(dev).to(torch.int32).contiguous()
    targets = torch.randint(0, P_HEAD, (args.B,), generator=g).to(dev).to(torch.int32).contiguous()
    params0 = flat(eager_named(seed=3)).to(dev).contiguous()

    print(f"== mamba x {args.opt} TC A/A/A probe: steps={args.steps} nruns={args.nruns} "
          f"ncta={args.ncta} B={args.B} sam={args.sam} ==")
    outs = []
    for r in range(args.nruns):
        loss, grad = run(mod, params0, tokens, targets, args.steps, args.ncta,
                         opt=args.opt, sam=args.sam)
        nan = bool(torch.isnan(grad).any().item())
        outs.append((loss, grad, nan))
        print(f"  run {r}: loss={loss:.8f} grad_has_nan={nan} grad_absmax={grad.abs().max().item():.4e}")

    base_l, base_g, _ = outs[0]
    max_dl = max(abs(o[0] - base_l) for o in outs)
    max_dg = 0.0
    for o in outs[1:]:
        d = (o[1] - base_g).abs().max().item()
        max_dg = max(max_dg, d)
    any_nan = any(o[2] for o in outs)
    print(f"== loss spread={max_dl:.3e}  grad max|delta|={max_dg:.3e}  any_nan={any_nan} ==")
    bit_identical = (max_dl == 0.0 and max_dg == 0.0 and not any_nan)
    print("RESULT:", "A/A/A BIT-IDENTICAL" if bit_identical else "NON-DETERMINISTIC (race present)")
    return 0 if bit_identical else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
