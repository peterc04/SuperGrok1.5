"""
LookSAM — AdamW with sharpness-aware direction adjustment.

LookSAM periodically computes sharpness-aware perturbations (every *k*
steps) and uses them to adjust gradient directions, steering the optimiser
toward flatter minima without the full cost of two forward/backward passes
at every step.

All heavy computation is dispatched to fused C++/CUDA kernels via _ops.
"""

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from grokking_optimizers._ops_loader import get_ops

_ops = get_ops()  # Fails loudly if C++ extension not built
from grokking_optimizers._adamw_helper import adamw_step


class LookSAM(Optimizer):
    """AdamW with sharpness-aware direction adjustment (LookSAM).

    The standard ``step()`` performs a regular AdamW update. Sharpness-aware
    perturbation and gradient direction adjustment are handled separately via
    :meth:`sam_step`, which should be called periodically (every *k* steps)
    by the training loop.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for running averages of gradient and its
            square (default: (0.9, 0.98)).
        eps: Numerical stability term (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 1.0).
        rho: SAM perturbation radius (default: 0.05).
        k: SAM step frequency — perform sharpness-aware computation
            every *k* steps (default: 5).
        alpha: Direction adjustment interpolation weight (default: 0.7).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 1.0,
        rho: float = 0.05,
        k: int = 5,
        alpha: float = 0.7,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if k < 1:
            raise ValueError(f"Invalid k value: {k}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rho=rho,
            k=k,
            alpha=alpha,
        )
        super().__init__(params, defaults)
        self._global_step: int = 0

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Perform a standard AdamW optimisation step.

        SAM perturbation is **not** applied here; call :meth:`sam_step`
        separately when sharpness-aware updates are desired.

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

        self._global_step += 1

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
                    state["sam_direction"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1

                params_list.append(p)
                grads_list.append(p.grad)
                exp_avg_list.append(state["exp_avg"])
                exp_avg_sq_list.append(state["exp_avg_sq"])
                step_list.append(state["step"])

            if len(params_list) == 0:
                continue

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

    @torch.no_grad()
    def sam_step(
        self,
        model: torch.nn.Module,
        train_x: Tensor,
        train_y: Tensor,
        criterion: Callable,
    ) -> None:
        """Perform a LookSAM sharpness-aware gradient direction adjustment.

        This method:
          1. Saves current gradients and backs up parameters.
          2. Perturbs parameters in the sharpness direction.
          3. Computes gradients at the perturbed point.
          4. Restores original parameters.
          5. Computes and stores sharpness-aware directions.
          6. Adjusts current gradients using those directions.

        Should be called every *k* training steps (see constructor).

        Args:
            model: The model whose parameters are being optimised.
            train_x: Input batch for re-evaluating loss at the perturbed point.
            train_y: Target batch.
            criterion: Loss function ``criterion(model(train_x), train_y)``.
        """
        # Collect per-group data in a single pass so it survives across phases
        group_data = []
        for group in self.param_groups:
            params_list = []
            orig_grads_list = []
            direction_list = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["sam_direction"] = torch.zeros_like(p, dtype=torch.float32)
                params_list.append(p)
                orig_grads_list.append(p.grad.clone())
                direction_list.append(state["sam_direction"])

            if len(params_list) == 0:
                group_data.append(None)
                continue

            # Step 1: Perturb parameters — returns backups
            backups = _ops.looksam_perturb_all(
                params_list,
                orig_grads_list,
                group["rho"],
            )
            group_data.append((params_list, orig_grads_list, direction_list, backups, group))

        # Step 2: Forward pass at the perturbed point
        model.zero_grad()
        with torch.enable_grad():
            perturbed_loss = criterion(model(train_x), train_y)
            perturbed_loss.backward()

        # Step 3 & 4: Restore parameters and compute/adjust directions
        for gd in group_data:
            if gd is None:
                continue
            params_list, orig_grads_list, direction_list, backups, group = gd

            # Collect perturbed gradients
            perturbed_grads_list = [p.grad.clone() for p in params_list]

            # Restore original parameters
            _ops.looksam_restore_all(params_list, backups)

            # Compute sharpness-aware directions: (v_dirs, sam_grads, normal_grads)
            _ops.looksam_compute_directions(
                direction_list,
                perturbed_grads_list,
                orig_grads_list,
            )

            # Adjust current gradients using the sharpness-aware directions
            _ops.looksam_adjust_grads(
                orig_grads_list,
                direction_list,
                group["alpha"],
            )

            # Write adjusted gradients back to parameters
            for p, g in zip(params_list, orig_grads_list):
                p.grad.copy_(g)

    @property
    def global_step(self) -> int:
        """Current global step count."""
        return self._global_step

    def should_sam_step(self) -> bool:
        """Return ``True`` if SAM should be applied at the current step."""
        k = self.param_groups[0]["k"]
        return self._global_step % k == 0
