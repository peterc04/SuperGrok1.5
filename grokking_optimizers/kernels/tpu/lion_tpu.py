"""TPU/Pallas kernel for Lion — base optimizer (Phase 4 WS5 symmetry fill).

# ----------------------------------------------------------------------------
# TPU TREE STATUS: REFERENCE / SPEC (single source of truth).
# The AUTHORITATIVE, executed Pallas path for tpu_v5p is
# csrc/backends/pallas/launch_lion.py (resolved by the dispatch layer and
# imported by csrc/backends/pallas/_pallas_fused.py). The four base optimizers
# (adamw, lion, grokfast, grokadamw) inline their per-tensor math THERE; this
# module re-exports it so grokking_optimizers/kernels/tpu/ carries all 11
# optimizers symmetrically (previously only the 7 non-base ones lived here).
# There is NO duplicated math — editing launch_lion.py is the single edit
# point; this shim follows automatically. See WORKSTREAM 5 in the Phase 4 report
# and WORKSTREAM 4 (math single-source-of-truth).
# ----------------------------------------------------------------------------
"""

from __future__ import annotations

from csrc.backends.pallas.launch_lion import launch_lion_step

# Naming-symmetry alias matching the other kernels/tpu/<opt>_tpu.py modules
# (e.g. neuralgrok_update, prodigy_update). Same callable, single source.
lion_update = launch_lion_step

__all__ = ["launch_lion_step", "lion_update"]
