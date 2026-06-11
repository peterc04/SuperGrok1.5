"""Minimal single-launch driver for ncu profiling of the vit TC megakernel.
Builds the vit TC TU, runs ONE tc_train_step at a small batch (enough to fill
the grid: >= 132 tiles → B*kSeq/1088 >= 132 → B >= 8448; use B=8704 = 512*17
which gives exactly 136 tiles, >132 SMs, so every SM gets >=1 tile but the run
is short). ncu wraps this; we want warp-stall reasons + SOL + tensor-pipe %.

NOTE: grid-barrier persistent kernels DEADLOCK under ncu default kernel-replay;
the caller MUST pass --replay-mode application.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import tuning.roofline as R  # noqa: E402


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 8704  # 512 samples → 8704 rows? no: 512*17=8704
    dev = torch.device("cuda")
    mod, params, inp, tgt, total = R._tc_inputs("vit", B, dev)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    beta1, beta2 = 0.9, 0.98
    bc1, bc2 = 1.0 - beta1, 1.0 - beta2
    lr, eps, wd = 1e-3, 1e-8, 0.0
    # warmup once (build caches / one-time) OUTSIDE the profiled region is not
    # possible with ncu launch-count=1 capturing the first launch; ncu --launch-skip
    # handles that. Here we just issue exactly the calls; caller controls skip/count.
    for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3):
        mod.tc_train_step(params.clone(), inp, tgt, state, lr, beta1, beta2,
                          eps, wd, bc1, bc2, 1, 0)
    torch.cuda.synchronize()
    print(f"[ncu-driver] done B={B} (rows={B})", flush=True)


if __name__ == "__main__":
    main()
