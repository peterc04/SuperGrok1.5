"""Fused (mamba + muon) instantiation for TPU v5p (Pallas / JAX).

This file is the fusion point that wires the model forward/backward
with the optimizer launch glue. Real implementations live in:

  csrc/algorithms/muon.h                      C++ math spec (mirrored in primitives.py)
  csrc/models/mamba.h                           model contract
  csrc/backends/pallas/launch_muon.py         optimizer launchers
  csrc/backends/pallas/models/mamba.py          model kernels
  csrc/backends/pallas/primitives.py            shared JAX/Pallas helpers

At this stage the fused step is a placeholder that raises NotImplementedError.
The race driver routes around it via the per-optimizer launch_*.py files.
"""

from csrc.backends.pallas.launch_muon import (
    launch_muon_step,
)


def fused_mamba_muon_step(*args, **kwargs):
    raise NotImplementedError(
        "fused mamba+muon for tpu_v5p is not implemented; "
        "use csrc.backends.pallas.launch_muon.launch_muon_step directly."
    )
