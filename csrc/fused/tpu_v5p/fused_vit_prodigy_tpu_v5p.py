"""Fused (vit + prodigy) instantiation for TPU v5p (Pallas / JAX).

This file is the fusion point that wires the model forward/backward
with the optimizer launch glue. Real implementations live in:

  csrc/algorithms/prodigy.h                      C++ math spec (mirrored in primitives.py)
  csrc/models/vit.h                           model contract
  csrc/backends/pallas/launch_prodigy.py         optimizer launchers
  csrc/backends/pallas/models/vit.py          model kernels
  csrc/backends/pallas/primitives.py            shared JAX/Pallas helpers

At this stage the fused step is a placeholder that raises NotImplementedError.
The race driver routes around it via the per-optimizer launch_*.py files.
"""

from csrc.backends.pallas.launch_prodigy import (
    launch_prodigy_step,
)


def fused_vit_prodigy_step(*args, **kwargs):
    raise NotImplementedError(
        "fused vit+prodigy for tpu_v5p is not implemented; "
        "use csrc.backends.pallas.launch_prodigy.launch_prodigy_step directly."
    )
