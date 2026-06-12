#!/usr/bin/env python
"""AUTHORITATIVE production-path A/A/A probe for mamba x prodigy (and looksam).

Drives the REAL fused_train_step("mamba3","prodigy", gemm_impl="wgmma") path —
the exact code dispatch routes in production — three times from a bit-identical
init, and reports loss/grad/param determinism. This is the path the carve-out
claims is racy. It temporarily registers the cell (monkeypatch) + sets the C++
probe env so dispatch.cpp routes the landed-dormant prodigy/looksam TC tail.

SG_MAMBA_PRODIGY_PROBE / SG_MAMBA_LOOKSAM_PROBE are read by dispatch.cpp at static
init, so they MUST be set before _ops imports — this script sets them at the top.

Usage:
  CUDA_MPS_PIPE_DIRECTORY=/nonexistent python tests/hw/_mamba_prodigy_production_probe.py [--opt prodigy|looksam] [--nruns 3]
"""
import os, sys, argparse

# Set the C++ probe env BEFORE any import that pulls in _ops.
os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/nonexistent")
ap0 = argparse.ArgumentParser(add_help=False)
ap0.add_argument("--opt", default="prodigy")
ap0.add_argument("--nruns", type=int, default=3)
_a, _ = ap0.parse_known_args()
if _a.opt == "prodigy":
    os.environ["SG_MAMBA_PRODIGY_PROBE"] = "1"
elif _a.opt == "looksam":
    os.environ["SG_MAMBA_LOOKSAM_PROBE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 9:
        print("SKIP: no sm_90 GPU"); return 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from grokking_optimizers import dispatch as D
    # Temporarily REGISTER the cell so has_l3_real / gemm_impl_for_cell accept it
    # (the C++ probe env already lifts the dispatch.cpp carve-out at static init).
    cell = ("mamba3", _a.opt)
    D._FUSED_L3_REAL = frozenset(set(D._FUSED_L3_REAL) | {cell})
    D._L3_WGMMA_CELLS = frozenset(set(D._L3_WGMMA_CELLS) | {cell})

    # Reuse the gate's cell builder + factory so the init is the EXACT production
    # gate init (and bit-identical across the 3 re-runs by the gate's own contract).
    from tests.hw import test_l3tc_tail_gate as G
    spec = G._CELLS.get(f"{_a.opt}/mamba")
    if spec is None:
        # prodigy/mamba & looksam/mamba are intentionally NOT in _CELLS (blocked);
        # synthesize a spec from the registered factories.
        fac = {"prodigy": G._prodigy_factory, "looksam": G._looksam_factory}[_a.opt]
        spec = dict(model="mamba", opt=_a.opt, factory=fac)

    from grokking_optimizers.dispatch import canonicalize_model, fused_train_step
    canon = canonicalize_model("mamba")

    print(f"== PRODUCTION-PATH A/A/A: fused_train_step(mamba3, {_a.opt}, wgmma) "
          f"nruns={_a.nruns} ==", flush=True)

    def _one():
        g2, c2, m2, data2, dev2 = G._build_cell("mamba")
        tx2, ty2 = data2[0], data2[1]
        opt2 = spec["factory"](g2, m2.parameters(), c2)
        cc = {}
        L, Gr = fused_train_step(canon, _a.opt, m2, opt2, tx2, ty2, state_cache=cc,
                                 step=1, return_grad=True, gemm_impl="wgmma")
        total = sum(p.numel() for p in m2.parameters() if p.requires_grad)
        P = torch.empty(total, device=dev2)
        o = 0
        for _, p in m2.named_parameters():
            if p.requires_grad:
                k = p.numel(); P[o:o + k].copy_(p.data.reshape(-1)); o += k
        return float(L), Gr.clone(), P

    outs = [_one() for _ in range(_a.nruns)]
    for i, (L, Gr, _) in enumerate(outs):
        nan = bool(torch.isnan(Gr).any().item())
        print(f"  run {i}: loss={L:.8f} grad_has_nan={nan} grad_absmax={Gr.abs().max().item():.4e}",
              flush=True)
    L0, G0, P0 = outs[0]
    max_dl = max(abs(o[0] - L0) for o in outs)
    max_dg = max((o[1] - G0).abs().max().item() for o in outs[1:]) if len(outs) > 1 else 0.0
    max_dp = max((o[2] - P0).abs().max().item() for o in outs[1:]) if len(outs) > 1 else 0.0
    any_nan = any(bool(torch.isnan(o[1]).any().item()) for o in outs)
    print(f"== loss spread={max_dl:.3e}  grad max|delta|={max_dg:.3e}  "
          f"param max|delta|={max_dp:.3e}  any_nan={any_nan} ==", flush=True)
    ok = (max_dl == 0.0 and max_dg == 0.0 and max_dp == 0.0 and not any_nan)
    print("RESULT:", "A/A/A BIT-IDENTICAL" if ok else "NON-DETERMINISTIC (race present)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
