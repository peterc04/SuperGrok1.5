"""
AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2017).

Per-element update (FP32 accumulator, BF16/FP16 storage allowed):

    m_t = beta1 * m_{t-1} + (1 - beta1) * g
    v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    bc1 = 1 - beta1^t                                # un-inverted
    bc2 = 1 - beta2^t                                # un-inverted
    m_hat = m_t / bc1
    v_hat = v_t / bc2
    p_{t+1} = p_t - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p_t)

All computation is dispatched to the fused C++/CUDA kernel via _ops.
Math: csrc/algorithms/adamw.h. Launchers:
  csrc/backends/cuda/sm_90/launch_adamw.cu
  csrc/backends/hip/gfx942/launch_adamw.hip.cpp
  csrc/backends/pallas/launch_adamw.py
"""

from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from grokking_optimizers.dispatch import get_ops

_ops = get_ops()  # Fails loudly on first attribute access if extension not built


def _validate_grad(p):
    """Validate a gradient before handing its data_ptr to the fused kernel.

    The fused kernels index raw contiguous memory and dispatch on the
    parameter's dtype (treating the grad buffer as the same scalar type),
    so a sparse, dtype-mismatched, or non-contiguous gradient would
    silently corrupt the parameter and optimizer state. There is no Python
    fallback, so reject the unsupported cases loudly and densify the rest.
    """
    g = p.grad
    if g.is_sparse:
        raise RuntimeError(
            "fused optimizer kernel does not support sparse gradients")
    if not g.is_cuda:
        raise RuntimeError(
            "fused optimizer kernel requires CUDA tensors; got a CPU gradient. "
            "These are GPU-only kernels with no CPU fallback — move the model "
            "and gradients to a CUDA device before step().")
    if g.dtype != p.dtype:
        raise RuntimeError(
            f"grad dtype {g.dtype} != param dtype {p.dtype}; cast gradients "
            "to the parameter dtype before step()")
    return g if g.is_contiguous() else g.contiguous()


class AdamW(Optimizer):
    """Adam with decoupled weight decay (multi-tensor fused step).

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: (beta1, beta2) for the first/second moment EMAs
            (default: (0.9, 0.999)).
        eps: Numerical stabiliser added to the denominator (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 1e-2).
        use_grad_hooks: When True, register
            ``register_post_accumulate_grad_hook`` on each parameter so the
            update runs while gradients are still L2-warm (requires
            PyTorch >= 2.1).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        use_grad_hooks: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"Invalid betas (expected a 2-tuple): {betas}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Per-group cache of the static tensor lists handed to the fused kernel.
        # Params and their state buffers keep a fixed identity across steps, so
        # we build params/exp_avg/exp_avg_sq once (avoiding the per-step
        # self.state[p] re-hash and list rebuilds) and only refresh the dynamic
        # grads_list + step counters. Invalidated by add_param_group.
        self._static_cache: dict = {}
        # Lazily-bound fused kernel callable (resolved once at first step()).
        self._fused_step = None

        self._use_grad_hooks = use_grad_hooks
        if use_grad_hooks:
            _register_grad_hooks(self)

    def add_param_group(self, param_group) -> None:
        # Static tensor lists are derived from the param groups; drop the cache
        # so the next step() rebuilds them.
        self._static_cache = {}
        super().add_param_group(param_group)

    def _group_cache(self, group):
        """Return cached (params, exp_avg, exp_avg_sq, step_counts) for *group*.

        The cache is keyed on the tuple of grad-bearing parameter ids; if that
        set changes (a param newly receives or loses a grad) the cache is
        rebuilt so semantics match the original per-step scan exactly.
        """
        key = tuple(id(p) for p in group["params"] if p.grad is not None)
        cached = self._static_cache.get(id(group))
        if cached is not None and cached[0] == key:
            return cached[1]

        params_list = []
        exp_avg_list = []
        exp_avg_sq_list = []
        step_counts = []  # mutable per-param step counters (1-element lists)
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            params_list.append(p)
            exp_avg_list.append(state["exp_avg"])
            exp_avg_sq_list.append(state["exp_avg_sq"])
            step_counts.append(state)
        entry = (params_list, exp_avg_list, exp_avg_sq_list, step_counts)
        self._static_cache[id(group)] = (key, entry)
        return entry

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    def _single_param_step(self, param, group, state):
        """Per-parameter eager step (use_grad_hooks path) — REMOVED (pure L3-TC)."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")


# ── Shared (inlined) helper: register post_accumulate_grad_hook on each param.
# Duplicated across every optimizer file by design (self-containment);
# requires PyTorch >= 2.1.
def _register_grad_hooks(optimizer):
    _pt = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if _pt < (2, 1):
        raise RuntimeError(
            f"use_grad_hooks requires PyTorch >= 2.1 for "
            f"register_post_accumulate_grad_hook. Current: {torch.__version__}.")
    optimizer._grad_hook_handles = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not p.requires_grad:
                continue
            def _hook(param, _g=group, _opt=optimizer):
                if param.grad is None:
                    return
                _opt._single_param_step(param, _g, _opt.state[param])
            optimizer._grad_hook_handles.append(
                p.register_post_accumulate_grad_hook(_hook))
