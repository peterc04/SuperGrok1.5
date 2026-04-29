"""
profile_smoke.py — minimal kernel-exercising harness for `ncu` profiling.

Imports each of the eleven public optimizers from grokking_optimizers,
constructs a tiny parameter tensor, and runs 5 optimizer steps. Sequential,
no asserts, no metric checks. The point is to give ncu / nsys a payload of
real optimizer kernel launches in one process.

Invoked by build.sh --profile under:
    ncu --set full --target-processes all -o profile_output/baseline \
        python benchmarks/profile_smoke.py
"""

from __future__ import annotations

import sys
import traceback

import torch
import torch.nn as nn

import grokking_optimizers as go


N_STEPS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _tiny_model():
    # 2-layer MLP, ~1k params; exercises both bias and weight kernels.
    m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).to(DEVICE)
    return m


def _step_loop(opt, model, name, needs_closure=False):
    x = torch.randn(4, 16, device=DEVICE)
    y = torch.randn(4, 8, device=DEVICE)
    loss_fn = nn.MSELoss()
    for i in range(N_STEPS):
        opt.zero_grad(set_to_none=True)
        if needs_closure:
            def closure():
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                return loss
            opt.step(closure)
        else:
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _run(name, build_fn, needs_closure=False):
    print(f"[profile_smoke] {name}", flush=True)
    try:
        model = _tiny_model()
        opt = build_fn(model.parameters())
        _step_loop(opt, model, name, needs_closure=needs_closure)
    except Exception as exc:
        # Profiling smoke: log and continue. ncu still captures completed work.
        print(f"[profile_smoke] {name} skipped: {exc}", file=sys.stderr)
        traceback.print_exc()


def main():
    # Eleven public optimizers as exported by grokking_optimizers.__init__.
    _run("GrokAdamW",   lambda p: go.GrokAdamW(p, lr=1e-3))
    _run("NeuralGrok",  lambda p: go.NeuralGrok(p, lr=1e-3))
    _run("Prodigy",     lambda p: go.Prodigy(p, lr=1.0))
    _run("Grokfast",    lambda p: go.Grokfast(p, lr=1e-3))
    _run("Lion",        lambda p: go.Lion(p, lr=1e-4))
    _run("Muon",        lambda p: go.Muon(p, lr=1e-3))
    _run("LookSAM",     lambda p: go.LookSAM(p, lr=1e-3), needs_closure=True)
    _run("SuperGrok11", lambda p: go.SuperGrok11(p, lr=1e-3))
    _run("SuperGrok15", lambda p: go.SuperGrok15(p, lr=1e-3))
    _run("SuperGrok2",  lambda p: go.SuperGrok2(p, lr=1e-3))

    # CUDAGraphOptimizer wraps another optimizer; exercise the wrap path.
    def _build_graph_opt(p):
        params = list(p)
        inner = go.GrokAdamW(params, lr=1e-3)
        return go.CUDAGraphOptimizer(inner)
    _run("CUDAGraphOptimizer", _build_graph_opt)


if __name__ == "__main__":
    main()
