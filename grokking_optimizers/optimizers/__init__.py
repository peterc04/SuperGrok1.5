"""The 11 core optimizers, each a torch.optim.Optimizer subclass.

Math: each optimizer's per-element step is documented in csrc/algorithms/.
On GPU, .step() dispatches to the compiled C++ extension via
grokking_optimizers.dispatch.get_ops. The kernel is the only execution
path; there is no Python reference fallback.

Public classes (re-exported by grokking_optimizers/__init__.py):
  SuperGrok2, SuperGrok15, SuperGrok11
  GrokAdamW, NeuralGrok, Prodigy, Grokfast
  Lion, LookSAM, Muon
  MoEAwareSuperGrok2   (the MoE/Adam multi-tensor variant)
"""

from .supergrok2 import SuperGrok2, CompiledSuperGrok2, MoEAwareSuperGrok2
from .supergrok15 import SuperGrok15
from .supergrok11 import SuperGrok11
from .grokadamw import GrokAdamW
from .grokfast import Grokfast
from .lion import Lion
from .looksam import LookSAM
from .muon import Muon
from .neuralgrok import NeuralGrok
from .prodigy import Prodigy

__all__ = [
    "SuperGrok2", "CompiledSuperGrok2",
    "SuperGrok15",
    "SuperGrok11",
    "GrokAdamW",
    "Grokfast",
    "Lion",
    "LookSAM",
    "Muon",
    "NeuralGrok",
    "Prodigy",
    "MoEAwareSuperGrok2",
]
