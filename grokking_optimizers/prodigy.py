"""
Prodigy — Distance-aware self-tuning Adam.

Prodigy automatically estimates the optimal learning rate by tracking the
distance between the current parameters and their initial values. The
recommended default learning rate is ``1.0``; the internal ``d_lr`` scalar
is adjusted adaptively so that manual LR tuning is largely unnecessary.

All computation is dispatched to the fused C++/CUDA kernel via _ops.
"""

from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from grokking_optimizers._ops_loader import get_ops

_ops = get_ops()  # Fails loudly if C++ extension not built


class Prodigy(Optimizer):
    """Distance-aware self-tuning Adam optimiser.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Outer learning rate multiplier. The recommended default is
            ``1.0`` because Prodigy self-tunes its effective step size
            (default: 1.0).
        betas: Coefficients for running averages of gradient and its
            square (default: (0.9, 0.999)).
        eps: Numerical stability term (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 1.0).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Global adaptive learning rate shared across all parameter groups.
        self._d_lr: float = 1e-6

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

        for group in self.param_groups:
            params_list = []
            grads_list = []
            exp_avg_list = []
            exp_avg_sq_list = []
            s_list = []
            param_init_list = []
            step_list = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Lazy state initialisation
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["s"] = torch.zeros_like(p, dtype=torch.float32)
                    state["param_init"] = p.detach().clone().float()

                state["step"] += 1

                params_list.append(p)
                grads_list.append(p.grad)
                exp_avg_list.append(state["exp_avg"])
                exp_avg_sq_list.append(state["exp_avg_sq"])
                s_list.append(state["s"])
                param_init_list.append(state["param_init"])
                step_list.append(state["step"])

            if len(params_list) == 0:
                continue

            # The kernel returns the updated d_lr value.
            self._d_lr = _ops.prodigy_fused_step(
                params_list,
                grads_list,
                exp_avg_list,
                exp_avg_sq_list,
                s_list,
                param_init_list,
                step_list,
                self._d_lr,
                group["betas"][0],
                group["betas"][1],
                group["lr"],
                group["weight_decay"],
                group["eps"],
            )

        return loss

    def _single_param_step(self, param, group, state):
        """Per-parameter step for GradientHookOptimizer integration."""
        if param.grad is None:
            return
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            state["s"] = torch.zeros_like(param, dtype=torch.float32)
            state["param_init"] = param.data.clone()
        state["step"] += 1
        bc1 = 1 - group["betas"][0] ** state["step"]
        bc2 = 1 - group["betas"][1] ** state["step"]
        d_lr = getattr(self, '_d_lr', 1.0)
        _ops.prodigy_fused_step(
            [param], [param.grad], [state["exp_avg"]], [state["exp_avg_sq"]],
            [state["s"]], d_lr,
            group["betas"][0], group["betas"][1], group["lr"],
            group["weight_decay"], group["eps"], bc1, bc2,
        )

    @property
    def d_lr(self) -> float:
        """Current adaptive learning rate estimated by Prodigy."""
        return self._d_lr
