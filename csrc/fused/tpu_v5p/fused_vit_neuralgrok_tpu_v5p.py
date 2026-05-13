"""Fused (vit + neuralgrok) instantiation for TPU v5p (Pallas / JAX).

This file is the fusion point that wires the model forward/backward
with the optimizer launch glue. Real implementations live in:

  csrc/algorithms/neuralgrok.h                      C++ math spec (mirrored in primitives.py)
  csrc/models/vit.h                           model contract
  csrc/backends/pallas/launch_neuralgrok.py         optimizer launchers
  csrc/backends/pallas/models/vit.py          model kernels
  csrc/backends/pallas/primitives.py            shared JAX/Pallas helpers

At this stage the fused step is a placeholder that raises NotImplementedError.
The race driver routes around it via the per-optimizer launch_*.py files.
"""

from csrc.backends.pallas.launch_neuralgrok import (
    launch_neuralgrok_step,
)


def fused_vit_neuralgrok_step(*args, **kwargs):
    raise NotImplementedError(
        "fused vit+neuralgrok for tpu_v5p is not implemented; "
        "use csrc.backends.pallas.launch_neuralgrok.launch_neuralgrok_step directly."
    )
