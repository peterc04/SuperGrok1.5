"""
NeuralGrok — Adam with a learned MLP gradient amplifier.

Uses a small neural network (the *amplifier*) to learn per-element gradient
scaling factors. The amplifier is a lightweight MLP that maps gradient
magnitudes to scale factors, enabling adaptive amplification that can be
jointly trained alongside the main model.

All heavy computation is dispatched to the fused C++/CUDA kernel via _ops.
"""

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

from grokking_optimizers import _HAS_OPS
if _HAS_OPS:
    from grokking_optimizers import _ops
else:
    from grokking_optimizers import _python_fallback as _ops


class _Amplifier(nn.Module):
    """MLP gradient amplifier: grad magnitude -> scale factor.

    A small feed-forward network that takes the absolute gradient value
    (shape ``[N, 1]``) and produces a multiplicative scale factor.

    Args:
        num_layers: Total number of linear layers (default: 3).
        hidden_dim: Width of hidden layers (default: 128).
    """

    def __init__(self, num_layers: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 (input + output layer)")

        self.hidden_dim = hidden_dim
        layers: List[nn.Module] = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: gradient magnitude -> scale factor."""
        return self.net(x)

    def get_weights(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return first- and last-layer weights for the C++ kernel.

        NOTE: The C++ kernel only evaluates a 2-layer MLP (first + last layer).
        If num_layers > 2, intermediate layers are skipped in the CUDA path.
        Set num_layers=2 for exact CUDA/Python parity.

        Returns:
            (W1, b1, W_last, b_last) — all as contiguous FP32 tensors.
        """
        first_linear = self.net[0]
        last_linear = self.net[-1]

        W1 = first_linear.weight.detach().float().contiguous()
        b1 = first_linear.bias.detach().float().contiguous()
        W_last = last_linear.weight.detach().float().contiguous()
        b_last = last_linear.bias.detach().float().contiguous()

        return W1, b1, W_last, b_last


class NeuralGrok(Optimizer):
    """Adam with a learned MLP gradient amplifier.

    The amplifier is a small ``_Amplifier`` network whose weights can be
    trained with a separate optimiser (see :meth:`get_amplifier_optimizer`).

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for running averages of gradient and its
            square (default: (0.9, 0.98)).
        eps: Numerical stability term (default: 1e-8).
        weight_decay: Decoupled weight decay (default: 1.0).
        alpha: Amplification scale factor (default: 10.0).
        beta: Secondary amplification scale factor (default: 4.0).
        num_layers: Number of layers in the amplifier MLP (default: 3).
        hidden_dim: Hidden dimension of the amplifier MLP (default: 128).
        inner_steps: Number of inner amplifier update steps per
            optimizer step (default: 1).
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
        alpha: float = 10.0,
        beta: float = 4.0,
        num_layers: int = 3,
        hidden_dim: int = 128,
        inner_steps: int = 1,
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

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            beta=beta,
            inner_steps=inner_steps,
            grad_clip=grad_clip,
        )

        # The amplifier is constructed before super().__init__ so that it
        # lives on the correct device once parameters are registered.
        self.amplifier = _Amplifier(num_layers=num_layers, hidden_dim=hidden_dim)

        super().__init__(params, defaults)

        self._meta_weights_dirty = True
        self._cached_meta_weights = None

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

        # Extract amplifier weights for the fused kernel (cached)
        if self._meta_weights_dirty or self._cached_meta_weights is None:
            self._cached_meta_weights = self.amplifier.get_weights()
            self._meta_weights_dirty = False
        W1, b1, W_last, b_last = self._cached_meta_weights

        for group in self.param_groups:
            params_list = []
            grads_list = []
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
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1

                params_list.append(p)
                grads_list.append(p.grad)
                exp_avg_list.append(state["exp_avg"])
                exp_avg_sq_list.append(state["exp_avg_sq"])
                step_list.append(state["step"])

            if len(params_list) == 0:
                continue

            # Move amplifier weights to same device as params
            device = params_list[0].device
            W1_d = W1.to(device)
            b1_d = b1.to(device)
            W_last_d = W_last.to(device)
            b_last_d = b_last.to(device)

            _ops.neuralgrok_fused_step(
                params_list,
                grads_list,
                exp_avg_list,
                exp_avg_sq_list,
                step_list,
                W1_d,
                b1_d,
                W_last_d,
                b_last_d,
                group["alpha"],
                group["beta"],
                self.amplifier.hidden_dim,
                group["betas"][0],
                group["betas"][1],
                group["lr"],
                group["weight_decay"],
                group["eps"],
                group["grad_clip"],
            )

        return loss

    def mark_amplifier_dirty(self) -> None:
        """Mark the cached amplifier weights as stale.

        Call this after updating the amplifier weights (e.g. after
        ``amplifier_optimizer.step()``) so that the next optimizer step
        re-extracts FP32 contiguous copies.
        """
        self._meta_weights_dirty = True

    def get_amplifier(self) -> _Amplifier:
        """Return the gradient amplifier module."""
        return self.amplifier

    def get_amplifier_optimizer(self, lr: float = 1e-4) -> torch.optim.Adam:
        """Create an Adam optimiser for the amplifier's parameters.

        Args:
            lr: Learning rate for training the amplifier (default: 1e-4).

        Returns:
            A ``torch.optim.Adam`` instance targeting the amplifier weights.
        """
        return torch.optim.Adam(self.amplifier.parameters(), lr=lr)
