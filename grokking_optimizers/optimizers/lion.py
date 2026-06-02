"""
Lion — Sign-based optimizer with interpolated momentum.

Lion (EvoLved Sign Momentum) uses the sign of an interpolation between the
gradient and the momentum buffer to compute parameter updates. This yields
uniform update magnitudes and strong implicit regularisation, often
outperforming AdamW with significantly less memory.

All computation is dispatched to the fused C++/CUDA kernel via _ops.
"""

from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from grokking_optimizers.dispatch import get_ops

_ops = get_ops()  # Fails loudly if C++ extension not built


def _validate_grad(p):
    """Validate a gradient before handing its data_ptr to the fused kernel.

    The fused kernel indexes raw contiguous memory and dispatches on the
    parameter's dtype, so a sparse, dtype-mismatched, or non-contiguous
    gradient would silently corrupt the parameter and momentum state.
    There is no Python fallback, so reject the unsupported cases loudly
    and densify the rest.
    """
    g = p.grad
    if g.is_sparse:
        raise RuntimeError(
            "fused optimizer kernel does not support sparse gradients")
    if g.dtype != p.dtype:
        raise RuntimeError(
            f"grad dtype {g.dtype} != param dtype {p.dtype}; cast gradients "
            "to the parameter dtype before step()")
    return g if g.is_contiguous() else g.contiguous()


class Lion(Optimizer):
    """Sign-based optimiser with interpolated momentum (Lion).

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 3e-4).
        betas: Coefficients for the interpolation between gradient and
            momentum for the update (beta1) and the momentum EMA decay
            (beta2) (default: (0.9, 0.99)).
        weight_decay: Decoupled weight decay coefficient (default: 3.0).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 3.0,
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

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Per-group cache of static tensor lists (params + exp_avg buffers keep
        # a fixed identity across steps); only grads_list is rebuilt per step.
        # Invalidated by add_param_group.
        self._static_cache: dict = {}
        # Lazily-bound fused kernel callable (resolved once at first step()).
        self._fused_step = None

        self._use_grad_hooks = use_grad_hooks
        if use_grad_hooks:
            _register_grad_hooks(self)

    def add_param_group(self, param_group) -> None:
        self._static_cache = {}
        super().add_param_group(param_group)

    def _group_cache(self, group):
        """Return cached (params, exp_avg) for *group*, keyed on the set of
        grad-bearing param ids so semantics match the per-step scan."""
        key = tuple(id(p) for p in group["params"] if p.grad is not None)
        cached = self._static_cache.get(id(group))
        if cached is not None and cached[0] == key:
            return cached[1]

        params_list = []
        exp_avg_list = []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            params_list.append(p)
            exp_avg_list.append(state["exp_avg"])
        entry = (params_list, exp_avg_list)
        self._static_cache[id(group)] = (key, entry)
        return entry

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Perform a single optimisation step.

        Args:
            closure: A closure that re-evaluates the model and returns the loss
                (optional).

        Returns:
            The loss value if *closure* is provided, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._use_grad_hooks:
            return loss

        fused_step = self._fused_step
        if fused_step is None:
            fused_step = self._fused_step = _ops.bind("lion_fused_step")

        for group in self.param_groups:
            params_list, exp_avg_list = self._group_cache(group)

            if len(params_list) == 0:
                continue

            grads_list = [_validate_grad(p) for p in params_list]

            fused_step(
                params_list,
                grads_list,
                exp_avg_list,
                group["lr"],
                group["betas"][0],
                group["betas"][1],
                group["weight_decay"],
            )

        return loss

    def _single_param_step(self, param, group, state):
        """Per-parameter step used by the `use_grad_hooks=True` path."""
        if param.grad is None:
            return
        grad = _validate_grad(param)
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
        _ops.lion_fused_step(
            [param], [grad], [state["exp_avg"]],
            group["lr"], group["betas"][0], group["betas"][1],
            group["weight_decay"],
        )


# ── Shared (inlined) helper: register post_accumulate_grad_hook on each param.
# Each hook calls back into the optimizer's `_single_param_step` so the update
# runs while gradient data is still L2-warm. Duplicated across every optimizer
# file by design (self-containment); requires PyTorch >= 2.1.
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
