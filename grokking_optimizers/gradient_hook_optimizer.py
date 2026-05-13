"""Backward-compatibility shim.
GradientHookOptimizer moved to grokking_optimizers.extensions.gradient_hook_optimizer
in Phase 9 of the refactor.
"""
from grokking_optimizers.extensions.gradient_hook_optimizer import GradientHookOptimizer

__all__ = ["GradientHookOptimizer"]
