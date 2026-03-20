"""
GrokAdamW — Adam with grokking-aware EMA gradient filtering and amplification.

Combines AdamW with an exponential moving average (EMA) gradient filter that
detects and amplifies slow-learning gradient signals, accelerating the
transition from memorisation to generalisation (grokking).

All computation is dispatched to the fused C++/CUDA kernel via _ops.
"""

from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from grokking_optimizers._ops_loader import get_ops

_ops = get_ops()  # Fails loudly if C++ extension not built


class GrokAdamW(Optimizer):
    """Adam with grokking-aware EMA gradient filtering and amplification.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient
            and its square (default: (0.9, 0.98)).
        eps: Term added to the denominator for numerical stability
            (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 1.0).
        alpha: EMA decay factor for gradient filter (default: 0.98).
        lamb: Amplification factor applied to the filtered gradient
            signal (default: 5.0).
        gamma: Deprecated — unused. Kept for API backward compatibility.
        decay: Deprecated — unused. Kept for API backward compatibility.
        grad_clip: Maximum gradient norm for per-parameter clipping
            (default: 1.0).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 1.0,
        alpha: float = 0.98,
        lamb: float = 5.0,
        gamma: float = 0.1,
        decay: float = 0.1,
        grad_clip: float = 1.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            lamb=lamb,
            gamma=gamma,
            decay=decay,
            grad_clip=grad_clip,
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
            exp_avg_sq_list = []
            ema_list = []
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
                    state["ema"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1

                params_list.append(p)
                grads_list.append(p.grad)
                exp_avg_list.append(state["exp_avg"])
                exp_avg_sq_list.append(state["exp_avg_sq"])
                ema_list.append(state["ema"])
                step_list.append(state["step"])

            if len(params_list) == 0:
                continue

            _ops.grokadamw_fused_step(
                params_list,
                grads_list,
                exp_avg_list,
                exp_avg_sq_list,
                ema_list,
                step_list,
                group["alpha"],
                group["lamb"],
                group["betas"][0],
                group["betas"][1],
                group["lr"],
                group["weight_decay"],
                group["eps"],
                group["grad_clip"],
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
            state["ema"] = torch.zeros_like(param, dtype=torch.float32)
        state["step"] += 1
        _ops.grokadamw_fused_step(
            [param], [param.grad], [state["exp_avg"]], [state["exp_avg_sq"]],
            [state["ema"]], [state["step"]],
            group["alpha"], group["lamb"],
            group["betas"][0], group["betas"][1], group["lr"],
            group["weight_decay"], group["eps"],
            group["grad_clip"],
        )
