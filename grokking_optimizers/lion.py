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

from grokking_optimizers import _HAS_OPS
if _HAS_OPS:
    from grokking_optimizers import _ops
else:
    from grokking_optimizers import _python_fallback as _ops


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
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
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

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Lazy state initialisation
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)

                params_list.append(p)
                grads_list.append(p.grad)
                exp_avg_list.append(state["exp_avg"])

            if len(params_list) == 0:
                continue

            _ops.lion_fused_step(
                params_list,
                grads_list,
                exp_avg_list,
                group["lr"],
                group["betas"][0],
                group["betas"][1],
                group["weight_decay"],
            )

        return loss
