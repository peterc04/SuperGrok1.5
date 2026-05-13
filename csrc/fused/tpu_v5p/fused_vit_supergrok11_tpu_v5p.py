"""Fused (vit + supergrok11) instantiation for TPU v5p (Pallas / JAX).

This file is the fusion point that wires the model forward/backward
with the optimizer launch glue. Real implementations live in:

  csrc/algorithms/supergrok11.h                      C++ math spec (mirrored in primitives.py)
  csrc/models/vit.h                           model contract
  csrc/backends/pallas/launch_supergrok11.py         optimizer launchers
  csrc/backends/pallas/models/vit.py          model kernels
  csrc/backends/pallas/primitives.py            shared JAX/Pallas helpers

At this stage the fused step is a placeholder that raises NotImplementedError.
The race driver routes around it via the per-optimizer launch_*.py files.
"""

from csrc.backends.pallas.launch_supergrok11 import (
    launch_supergrok11_step,
)


def fused_vit_supergrok11_step(*args, **kwargs):
    raise NotImplementedError(
        "fused vit+supergrok11 for tpu_v5p is not implemented; "
        "use csrc.backends.pallas.launch_supergrok11.launch_supergrok11_step directly."
    )
