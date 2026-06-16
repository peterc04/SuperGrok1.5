"""examples/autotune_demo/build_variant.py — build ONE GEMM variant .so in a
fresh subprocess and print its path.

run_autotune.py shells out to this once per search config so that each variant
is COMPILED in its own process (the GEMM op self-registers TORCH_LIBRARY(
sg_demo, ...), which can only happen once per process — so building+loading
multiple variants in one process would double-register and abort).

We use torch.utils.cpp_extension.load with is_python_module=False — the SAME
builder grokking_optimizers/compile.py::_torch_load wraps — handed the
-DSG_TUNED_* macro flags that compile.py's resolve_macros() produced. Prints
``SO_PATH <abs path to the built .so>`` on success.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_CU = HERE / "gemm_kernel.cu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="unique module/build name")
    ap.add_argument("--cflags", required=True,
                    help="JSON list of extra_cuda_cflags")
    args = ap.parse_args()
    cflags = json.loads(args.cflags)

    # Pin the arch the way compile.py's _torch_load does (avoid torch's
    # detected '9.0' gencode that rejects sm_90a-only instructions).
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

    import torch.utils.cpp_extension as cpp_ext

    cpp_ext.load(name=args.name, sources=[str(KERNEL_CU)],
                 extra_cuda_cflags=cflags, verbose=False,
                 is_python_module=False)
    bd = cpp_ext._get_build_directory(args.name, False)
    so = next((p for p in os.listdir(bd) if p.endswith(".so")), None)
    if so is None:
        sys.stderr.write(f"no .so produced in {bd}\n")
        return 1
    sys.stdout.write("SO_PATH " + str(Path(bd) / so) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
