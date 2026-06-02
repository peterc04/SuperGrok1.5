"""
Grokfast — AdamW with EMA gradient amplification.

Maintains an exponential moving average of gradients and amplifies the
current gradient by adding a scaled version of the EMA. This encourages
the optimiser to follow persistent gradient directions, which has been
shown to accelerate grokking (delayed generalisation).

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
    gradient would silently corrupt the parameter, EMA, and Adam state.
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


class Grokfast(Optimizer):
    """AdamW with EMA gradient amplification (Grokfast).

    The step proceeds in two phases:
      1. **Grokfast filter**: update the per-parameter gradient EMA and
         amplify the raw gradient in-place.
      2. **AdamW update**: apply a standard AdamW step using the
         amplified gradients.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for the AdamW running averages of gradient
            and its square (default: (0.9, 0.98)).
        eps: Numerical stability term (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 1.0).
        grokfast_alpha: EMA decay factor for gradient filtering
            (default: 0.98).
        grokfast_lamb: Amplification factor applied to the EMA signal
            (default: 2.0).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 1.0,
        grokfast_alpha: float = 0.98,
        grokfast_lamb: float = 2.0,
        use_grad_hooks: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"Invalid betas (expected a 2-tuple): {betas}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= grokfast_alpha < 1.0:
            raise ValueError(f"Invalid grokfast_alpha: {grokfast_alpha}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grokfast_alpha=grokfast_alpha,
            grokfast_lamb=grokfast_lamb,
        )
        super().__init__(params, defaults)

        # Per-group cache of static tensor lists (params + ema/exp_avg/exp_avg_sq
        # buffers keep a fixed identity across steps); only grads_list and step
        # counters are refreshed per step. Invalidated by add_param_group.
        self._static_cache: dict = {}
        # Lazily-bound fused kernel callable (resolved once at first step()).
        self._fused_step = None

        self._use_grad_hooks = use_grad_hooks
        if use_grad_hooks:
            _register_grad_hooks(self)

    def add_param_group(self, param_group) -> None:
        self._static_cache = {}
        super().add_param_group(param_group)

    def _group_cache(self, group, grads_by_id):
        """Return cached (params, grads, ema, exp_avg, exp_avg_sq, states).

        Keyed on the grad-bearing param ids. ``grads_by_id`` maps id(p) -> the
        validated grad, used to seed the EMA on first init (matching the
        canonical Grokfast first-gradient seed)."""
        key = tuple(id(p) for p in group["params"] if p.grad is not None)
        cached = self._static_cache.get(id(group))
        if cached is not None and cached[0] == key:
            return cached[1]

        params_list = []
        ema_list = []
        exp_avg_list = []
        exp_avg_sq_list = []
        states = []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                # Seed the gradient EMA with the first gradient (not zeros) to
                # match the canonical Grokfast filter. A zero seed heavily damps
                # the amplification term for the first ~1/(1-alpha) steps —
                # exactly the early phase Grokfast exists to accelerate — and
                # there is no EMA bias correction in the kernel to compensate.
                state["ema"] = grads_by_id[id(p)].detach().to(
                    torch.float32).clone()
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            params_list.append(p)
            ema_list.append(state["ema"])
            exp_avg_list.append(state["exp_avg"])
            exp_avg_sq_list.append(state["exp_avg_sq"])
            states.append(state)
        entry = (params_list, ema_list, exp_avg_list, exp_avg_sq_list, states)
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
            fused_step = self._fused_step = _ops.bind(
                "grokfast_fused_ema_adam_step")

        for group in self.param_groups:
            # Validate grads up front; this also provides the first-gradient
            # seed for any params whose EMA state is initialised on this step.
            grads_by_id = {
                id(p): _validate_grad(p)
                for p in group["params"] if p.grad is not None
            }
            params_list, ema_list, exp_avg_list, exp_avg_sq_list, states = \
                self._group_cache(group, grads_by_id)

            if len(params_list) == 0:
                continue

            grads_list = [grads_by_id[id(p)] for p in params_list]
            step_list = []
            for state in states:
                state["step"] += 1
                step_list.append(state["step"])

            # Fused EMA + amplification + Adam in a single CUDA pass
            fused_step(
                params_list, grads_list, ema_list,
                exp_avg_list, exp_avg_sq_list, step_list,
                group["grokfast_alpha"], group["grokfast_lamb"],
                group["betas"][0], group["betas"][1],
                group["lr"], group["weight_decay"], group["eps"],
            )

        return loss

    def _single_param_step(self, param, group, state):
        """Per-parameter step used by the `use_grad_hooks=True` path."""
        if param.grad is None:
            return
        grad = _validate_grad(param)
        if len(state) == 0:
            state["step"] = 0
            state["ema"] = grad.detach().to(torch.float32).clone()
            state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
        state["step"] += 1
        # Fused EMA + amplification + Adam in a single CUDA pass
        _ops.grokfast_fused_ema_adam_step(
            [param], [grad], [state["ema"]],
            [state["exp_avg"]], [state["exp_avg_sq"]], [state["step"]],
            group["grokfast_alpha"], group["grokfast_lamb"],
            group["betas"][0], group["betas"][1],
            group["lr"], group["weight_decay"], group["eps"],
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
