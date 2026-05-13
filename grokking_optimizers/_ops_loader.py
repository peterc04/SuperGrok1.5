"""Backward-compatibility shim. The actual implementation lives in
grokking_optimizers.dispatch.get_ops as of the Phase 9 refactor.
"""

from grokking_optimizers.dispatch import get_ops

__all__ = ["get_ops"]
