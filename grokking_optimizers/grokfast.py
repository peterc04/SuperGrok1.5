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

from grokking_optimizers import _HAS_OPS
if _HAS_OPS:
    from grokking_optimizers import _ops
else:
    from grokking_optimizers import _python_fallback as _ops
from grokking_optimizers._adamw_helper import adamw_step


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
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= grokfast_alpha < 1.0:
            raise ValueError(f"Invalid grokfast_alpha: {grokfast_alpha}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grokfast_alpha=grokfast_alpha,
            grokfast_lamb=grokfast_lamb,
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
            ema_list = []
            exp_avg_list = []
            exp_avg_sq_list = []
            step_list = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Lazy state initialisation
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1

                params_list.append(p)
                grads_list.append(p.grad)
                ema_list.append(state["ema"])
                exp_avg_list.append(state["exp_avg"])
                exp_avg_sq_list.append(state["exp_avg_sq"])
                step_list.append(state["step"])

            if len(params_list) == 0:
                continue

            # Phase 1: Grokfast EMA amplification (modifies grads in-place)
            _ops.grokfast_fused_step(
                grads_list,
                ema_list,
                group["grokfast_alpha"],
                group["grokfast_lamb"],
            )

            # Phase 2: AdamW update with the amplified gradients
            adamw_step(
                params_list,
                grads_list,
                exp_avg_list,
                exp_avg_sq_list,
                step_list,
                group["lr"],
                group["betas"][0],
                group["betas"][1],
                group["eps"],
                group["weight_decay"],
            )

        return loss
