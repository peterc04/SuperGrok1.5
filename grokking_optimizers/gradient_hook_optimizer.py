"""Backward-compatibility shim.
GradientHookOptimizer moved to grokking_optimizers._gradient_hook
in Phase 9 of the refactor.
"""
from grokking_optimizers._gradient_hook import GradientHookOptimizer

__all__ = ["GradientHookOptimizer"]
