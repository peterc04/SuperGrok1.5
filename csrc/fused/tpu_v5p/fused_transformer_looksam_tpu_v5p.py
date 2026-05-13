"""Fused (transformer + looksam) instantiation for TPU v5p (Pallas / JAX).

This file is the fusion point that wires the model forward/backward
with the optimizer launch glue. Real implementations live in:

  csrc/algorithms/looksam.h                      C++ math spec (mirrored in primitives.py)
  csrc/models/decoder.h                           model contract
  csrc/backends/pallas/launch_looksam.py         optimizer launchers
  csrc/backends/pallas/models/decoder.py          model kernels
  csrc/backends/pallas/primitives.py            shared JAX/Pallas helpers

At this stage the fused step is a placeholder that raises NotImplementedError.
The race driver routes around it via the per-optimizer launch_*.py files.
"""

from csrc.backends.pallas.launch_looksam import (
    launch_looksam_step,
)


def fused_transformer_looksam_step(*args, **kwargs):
    raise NotImplementedError(
        "fused decoder+looksam for tpu_v5p is not implemented; "
        "use csrc.backends.pallas.launch_looksam.launch_looksam_step directly."
    )
