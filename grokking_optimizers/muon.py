"""
Muon — Newton-Schulz orthogonalisation optimizer for 2D weights.

Muon applies orthogonalised gradient updates to 2D (matrix) parameters using
iterative Newton-Schulz orthogonalisation, while falling back to standard
AdamW for 1D parameters (biases, layer norms, embeddings). This encourages
weight matrices to stay near the orthogonal manifold, improving training
dynamics for deep networks.

All computation is dispatched to fused C++/CUDA kernels via _ops.
"""

from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from grokking_optimizers._ops_loader import get_ops

_ops = get_ops()  # Fails loudly if C++ extension not built


class Muon(Optimizer):
    """Orthogonalised update optimizer for 2D params, AdamW for others.

    This optimizer manages two distinct parameter groups internally:
      - **2D params** (e.g. weight matrices): updated with Muon
        (momentum + Newton-Schulz orthogonalisation).
      - **1D params** (biases, norms, etc.): updated with AdamW.

    Args:
        params_2d: Iterable of 2D parameters for Muon updates.
        params_1d: Iterable of non-2D parameters for AdamW updates
            (optional).
        lr: Learning rate for 2D Muon updates (default: 0.02).
        momentum: Momentum coefficient for 2D updates (default: 0.95).
        weight_decay: Weight decay for 2D params (default: 1.0).
        adamw_lr: Learning rate for AdamW (1D) updates (default: 1e-3).
        adamw_betas: Beta coefficients for AdamW (default: (0.9, 0.98)).
        adamw_eps: Epsilon for AdamW (default: 1e-8).
        ns_steps: Number of Newton-Schulz iteration steps for
            orthogonalisation (default: 5).
    """

    def __init__(
        self,
        params_2d: Iterable,
        params_1d: Optional[Iterable] = None,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 1.0,
        adamw_lr: float = 1e-3,
        adamw_betas: Tuple[float, float] = (0.9, 0.98),
        adamw_eps: float = 1e-8,
        ns_steps: int = 5,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if adamw_lr < 0.0:
            raise ValueError(f"Invalid AdamW learning rate: {adamw_lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= adamw_betas[0] < 1.0:
            raise ValueError(f"Invalid AdamW beta at index 0: {adamw_betas[0]}")
        if not 0.0 <= adamw_betas[1] < 1.0:
            raise ValueError(f"Invalid AdamW beta at index 1: {adamw_betas[1]}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        params_2d_list = list(params_2d)
        params_1d_list = list(params_1d) if params_1d is not None else []

        # Build param groups: group 0 = 2D (Muon), group 1 = 1D (AdamW)
        param_groups = [
            {
                "params": params_2d_list,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "ns_steps": ns_steps,
                "group_type": "muon",
            },
        ]
        if params_1d_list:
            param_groups.append(
                {
                    "params": params_1d_list,
                    "lr": adamw_lr,
                    "betas": adamw_betas,
                    "eps": adamw_eps,
                    "weight_decay": weight_decay,
                    "group_type": "adamw",
                },
            )

        # Use a minimal defaults dict; per-group settings override.
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            group_type="muon",
        )
        super().__init__(param_groups, defaults)

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
            group_type = group.get("group_type", "muon")

            if group_type == "muon":
                self._step_muon(group)
            else:
                self._step_adamw(group)

        return loss

    def _step_muon(self, group: dict) -> None:
        """Apply Muon (momentum + Newton-Schulz orthogonalisation) to 2D params."""
        params_list = []
        grads_list = []
        momentum_buf_list = []

        for p in group["params"]:
            if p.grad is None:
                continue

            # Lazy state initialisation
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(
                    p, dtype=torch.float32
                )

            params_list.append(p)
            grads_list.append(p.grad)
            momentum_buf_list.append(state["momentum_buffer"])

        if len(params_list) == 0:
            return

        _ops.muon_fused_step(
            params_list,
            grads_list,
            momentum_buf_list,
            group["momentum"],
            group["lr"],
            group["weight_decay"],
            group["ns_steps"],
        )

    def _step_adamw(self, group: dict) -> None:
        """Apply standard AdamW to non-2D params."""
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
            return

        betas = group.get("betas", (0.9, 0.98))
        eps = group.get("eps", 1e-8)

        _ops.fused_adamw_simple_step(
            params_list,
            grads_list,
            exp_avg_list,
            exp_avg_sq_list,
            step_list,
            betas[0],
            betas[1],
            group["lr"],
            group["weight_decay"],
            eps,
        )

    def _single_param_step(self, param, group, state):
        """Per-parameter step for GradientHookOptimizer integration."""
        if param.grad is None:
            return
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
        _ops.muon_fused_step(
            [param], [param.grad], [state["momentum_buffer"]],
            group.get("momentum", 0.95),
            group["lr"],
            group["weight_decay"],
            group.get("ns_steps", 5),
        )
