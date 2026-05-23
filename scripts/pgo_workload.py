"""scripts/pgo_workload.py — default PGO workload for the optimizer hot path.

Run N gradient-descent steps on a small parameter tensor using the
optimizer class produced by the most recent compile. The script is
invoked by ``grokking_optimizers.pgo.collect_workload`` between the
"instrument" and "use" AOT passes; its only purpose is to exercise the
optimizer's hot path so the profile data is representative.

The .so produced by the instrumented build is loaded by absolute path
(``--so``) so this script does not need ``pip install -e .``.

Usage::

    python scripts/pgo_workload.py \\
        --so build/compiled/.../grokking_compiled_lion_mamba_sm_90.so \\
        --opt Lion --model mamba --arch sm_90 --steps 1000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def load_so(so_path: Path) -> None:
    """Load ``so_path`` as ``grokking_optimizers._ops`` into the live process."""
    if "grokking_optimizers._ops" in sys.modules:
        del sys.modules["grokking_optimizers._ops"]
    spec = importlib.util.spec_from_file_location(
        "grokking_optimizers._ops", str(so_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load .so: {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    sys.modules["grokking_optimizers._ops"] = mod


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Default PGO workload — runs N optimizer steps")
    parser.add_argument("--so", type=Path, required=True,
                        help="Path to the instrumented .so")
    parser.add_argument("--opt", required=True,
                        help="Optimizer class name (e.g. Lion, AdamW)")
    parser.add_argument("--model", default="mamba",
                        help="Model identifier (informational; not loaded)")
    parser.add_argument("--arch", default="sm_90",
                        help="Arch identifier (informational; not loaded)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of optimizer.step() calls")
    parser.add_argument("--size", type=int, default=2048,
                        help="Parameter size (size × size float32 tensor)")
    args = parser.parse_args()

    if args.so:
        load_so(args.so)

    import torch
    from importlib import import_module
    grok = import_module("grokking_optimizers")
    OptCls = getattr(grok, args.opt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    p = torch.nn.Parameter(
        torch.randn(args.size, args.size, device=device, dtype=torch.float32))
    target = torch.randn_like(p)

    opt = OptCls([p], lr=1e-3)
    for _ in range(args.steps):
        loss = ((p - target) ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"PGO workload OK ({args.opt}/{args.model}/{args.arch}): "
          f"{args.steps} steps, size={args.size}, device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
