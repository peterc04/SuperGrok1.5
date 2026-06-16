"""tests/hw/test_mb3_scalar.py — Milestone A.1/B.1: the Mamba-3 SCALAR kernel
fwd+bwd vs the fp64 oracle. JIT-builds tests/hw/_mb3_scalar_probe.cu, runs one
batch, compares loss (fp32-vs-fp64 floor) + per-tensor grads. Run:
    python tests/hw/test_mb3_scalar.py
"""
from __future__ import annotations
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch

CFG = dict(p=97, ntok=99, seq_len=8, d=128, nl=2, state_dim=128, head_dim=64,
           expand_factor=2, mlp_ratio=2)
B = 64


def _build_mod():
    from torch.utils.cpp_extension import load
    src = os.path.join(_ROOT, "tests", "hw", "_mb3_scalar_probe.cu")
    return load(name="mb3_scalar_probe", sources=[src],
                extra_include_paths=[_ROOT],
                extra_cuda_cflags=["-std=c++17", "-arch=sm_90a", f"-I{_ROOT}",
                                   "--use_fast_math", "-diag-suppress=177"],
                verbose=True)


def _named_order():
    """The 45 named-parameter names in named_parameters() order (== flat layout)."""
    from grokking_optimizers.mamba3_block import Mamba3Model
    m = Mamba3Model(**CFG)
    return [n for n, _ in m.named_parameters()]


def _flat(model):
    return torch.cat([p.detach().reshape(-1).float()
                      for _, p in model.named_parameters()]).contiguous()


def main():
    torch.manual_seed(0)
    from grokking_optimizers.mamba3_block import Mamba3Model
    import tests.hw.mamba3_oracle as orc

    dev = torch.device("cuda")
    model = Mamba3Model(**CFG).to(dev)
    names = [n for n, _ in model.named_parameters()]
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}

    g = torch.Generator().manual_seed(1234)
    toks = torch.randint(0, CFG["ntok"], (B, CFG["seq_len"]), generator=g).to(dev)
    tgts = torch.randint(0, CFG["p"], (B,), generator=g).to(dev)

    # --- fp64 oracle (ground truth), CPU fp64 ---
    m64 = Mamba3Model(**CFG).to(torch.float64)
    m64.load_state_dict({k: v.detach().double().cpu() for k, v in model.state_dict().items()},
                        strict=True)
    loss64, cache = orc.model_forward(m64, toks.cpu(), tgts.cpu())
    grads64 = orc.model_backward(m64, cache)

    # --- kernel (fp32 scalar) ---
    mod = _build_mod()
    flat = _flat(model).to(dev)
    assert flat.numel() == mod.TOTAL, f"flat {flat.numel()} != kernel TOTAL {mod.TOTAL}"
    kloss, kgrad, _dbg = mod.scalar_fwdbwd(flat, toks, tgts)
    kloss = float(kloss.item())

    # --- compare ---
    print(f"\nloss: kernel={kloss:.8e}  oracle={loss64:.8e}  "
          f"rel={abs(kloss-loss64)/(abs(loss64)+1e-30):.3e}")
    off = 0
    worst = 0.0; worst_name = ""
    print(f"  {'tensor':34s} {'shape':16s} {'max-rel':>11s} {'max-abs':>11s}")
    for n in names:
        k = 1
        for s in shapes[n]:
            k *= s
        kg = kgrad[off:off + k].double().cpu().reshape(shapes[n])
        og = grads64[n].double().cpu()
        num = (kg - og).abs().max().item()
        den = og.abs().max().item() + 1e-30
        rel = num / den
        if rel > worst:
            worst = rel; worst_name = n
        flag = "  <<<" if rel > 5e-3 else ""
        print(f"  {n:34s} {str(shapes[n]):16s} {rel:11.3e} {num:11.3e}{flag}")
        off += k
    loss_rel = abs(kloss - loss64) / (abs(loss64) + 1e-30)
    print(f"\nWORST grad rel-err = {worst:.3e}  ({worst_name})")
    print(f"loss rel-err = {loss_rel:.3e}")
    # fp32 scalar vs fp64 oracle: expect ~1e-5..1e-3 (the scalar path is fp32).
    ok = (loss_rel < 5e-3) and (worst < 5e-2)
    print("\nMILESTONE A.1/B.1 (scalar):", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
