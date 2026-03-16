"""
CUDA Graph Optimizer Wrapper

Wraps any optimizer step in a CUDA graph for reduced kernel launch overhead.
Automatically handles:
  - Static buffer allocation for graph-captured tensors
  - Adaptive scheduling invalidation (re-records when hyperparams change)
  - Graceful fallback to eager mode when graph capture fails

Usage:
    opt = SuperGrok15(model.parameters(), ...)
    graph_opt = CUDAGraphOptimizer(opt)

    for step in range(n_steps):
        loss.backward()
        graph_opt.step()  # First call records, subsequent calls replay

When hyperparameters change (e.g., adaptive SAM frequency, progressive WD),
call graph_opt.invalidate() to force re-recording on the next step.
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Dict, Any


class CUDAGraphOptimizer:
    """Wraps an optimizer step in a CUDA graph for reduced launch overhead.

    Args:
        optimizer: The underlying optimizer to wrap.
        warmup_steps: Number of eager-mode steps before graph capture (default: 3).
            This ensures the optimizer state is fully initialized.
        max_graph_age: Maximum steps a graph can be replayed before forced
            re-recording (default: 0 = never force re-record).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 3,
        max_graph_age: int = 0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_graph_age = max_graph_age

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._step_count = 0
        self._graph_step_count = 0
        self._valid = False

        # Static buffers for captured tensors
        self._static_grads: Dict[int, torch.Tensor] = {}

    def step(self, **kwargs):
        """Execute an optimizer step, using CUDA graph when possible.

        Falls back to eager mode during warmup or when kwargs are provided
        (since kwargs typically vary per step, breaking graph assumptions).
        """
        self._step_count += 1

        # Eager mode during warmup or when variable kwargs are provided
        if self._step_count <= self.warmup_steps or kwargs:
            self._valid = False
            return self.optimizer.step(**kwargs)

        # Age-based invalidation
        if self.max_graph_age > 0 and self._graph_step_count >= self.max_graph_age:
            self._valid = False

        if not self._valid:
            self._record_graph()
            self._graph_step_count = 0

        if self._graph is not None:
            # Copy gradients into static buffers
            self._copy_grads_to_static()
            self._graph.replay()
            self._graph_step_count += 1
        else:
            # Graph capture failed — stay in eager mode
            self.optimizer.step()

    def invalidate(self):
        """Force re-recording of the CUDA graph on the next step.

        Call this when optimizer hyperparameters change (e.g., adaptive
        scheduling updates like SAM frequency, weight decay, etc.).
        """
        self._valid = False

    def _record_graph(self):
        """Record the optimizer step as a CUDA graph.

        No fallback — if graph capture fails, the error is raised.
        """
        # Allocate static gradient buffers
        self._allocate_static_grads()
        self._copy_grads_to_static()

        # Temporarily swap grads to static buffers
        orig_grads = self._swap_grads(self._static_grads)

        # Record
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.optimizer.step()

        # Restore original grads
        self._swap_grads(orig_grads)
        self._valid = True

    def _allocate_static_grads(self):
        """Allocate static buffers that mirror the gradient tensors."""
        self._static_grads.clear()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    pid = id(p)
                    if pid not in self._static_grads:
                        self._static_grads[pid] = torch.empty_like(p.grad)

    def _copy_grads_to_static(self):
        """Copy current gradients into static buffers."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                pid = id(p)
                if p.grad is not None and pid in self._static_grads:
                    self._static_grads[pid].copy_(p.grad)

    def _swap_grads(self, new_grads: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Swap parameter grads with provided tensors, return old grads."""
        old_grads = {}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                pid = id(p)
                old_grads[pid] = p.grad
                if pid in new_grads:
                    p.grad = new_grads[pid]
        return old_grads

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        self.invalidate()

    def __getattr__(self, name):
        """Forward attribute access to the wrapped optimizer."""
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self.optimizer, name)
