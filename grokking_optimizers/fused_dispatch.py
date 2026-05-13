"""Backward-compatibility shim. The fused-kernel dispatch surface
moved into grokking_optimizers.dispatch in Phase 9 of the refactor.
"""

from grokking_optimizers.dispatch import (
    MODELS,
    OPTIMIZERS,
    register_fused,
    has_fused,
    dispatch_fused,
)

__all__ = ["MODELS", "OPTIMIZERS", "register_fused", "has_fused", "dispatch_fused"]
