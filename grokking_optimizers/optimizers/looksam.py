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

from grokking_optimizers.dispatch import get_ops

_ops = get_ops()  # Fails loudly if C++ extension not built


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
        use_grad_hooks: bool = False,
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

        # Per-group cache of static tensor lists (params + exp_avg/exp_avg_sq
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

    def _group_cache(self, group):
        """Cached (params, exp_avg, exp_avg_sq, states) for the AdamW step(),
        keyed on the grad-bearing param ids."""
        key = tuple(id(p) for p in group["params"] if p.grad is not None)
        cached = self._static_cache.get(id(group))
        if cached is not None and cached[0] == key:
            return cached[1]
        params_list = []
        exp_avg_list = []
        exp_avg_sq_list = []
        states = []
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
            exp_avg_list.append(state["exp_avg"])
            exp_avg_sq_list.append(state["exp_avg_sq"])
            states.append(state)
        entry = (params_list, exp_avg_list, exp_avg_sq_list, states)
        self._static_cache[id(group)] = (key, entry)
        return entry

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

        if self._use_grad_hooks:
            return loss

        fused_step = self._fused_step
        if fused_step is None:
            fused_step = self._fused_step = _ops.bind("fused_adamw_simple_step")

        for group in self.param_groups:
            params_list, exp_avg_list, exp_avg_sq_list, states = \
                self._group_cache(group)

            if len(params_list) == 0:
                continue

            # Canonical LookSAM ``apply`` (csrc/algorithms/looksam.h
            # ``looksam_apply_step``): on EVERY step blend the cached SAM
            # direction into the gradient before the AdamW update,
            #   g_adj = (1 - alpha) * g + alpha * sam_dir,
            # where ``sam_dir`` was cached by the most recent ``sam_step``.
            # This is the amortization that makes the k-1 intervening steps
            # sharpness-aware. Before the first ``sam_step`` the cached
            # direction is zero, so g_adj = (1 - alpha) * g — identical to the
            # kernel, which applies the same blend unconditionally.
            alpha = group["alpha"]
            grads_list = []
            step_list = []
            for p, state in zip(params_list, states):
                sam_dir = state["sam_direction"]
                grads_list.append((1.0 - alpha) * p.grad + alpha * sam_dir)
                state["step"] += 1
                step_list.append(state["step"])

            fused_step(
                params_list,
                grads_list,
                exp_avg_list,
                exp_avg_sq_list,
                step_list,
                group["betas"][0],
                group["betas"][1],
                group["lr"],
                group["weight_decay"],
                group["eps"],
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

        # Step 3 & 4: Restore parameters, then CACHE the SAM direction.
        for gd in group_data:
            if gd is None:
                continue
            params_list, orig_grads_list, direction_list, backups, group = gd

            # Collect perturbed gradients (∇L at the perturbed point).
            perturbed_grads_list = [p.grad.clone() for p in params_list]

            # Restore original parameters
            _ops.looksam_restore_all(params_list, backups)

            # Canonical LookSAM ``set_direction`` (csrc/algorithms/looksam.h:
            # ``sam_dir = g_sam - g``): cache the gradient difference into the
            # per-parameter ``sam_direction`` state. This is the whole point of
            # LookSAM — the direction is REUSED by ``step()`` on the k-1
            # intervening (non-SAM) steps via the cached-direction blend. The
            # previous code adjusted ``p.grad`` in place for the current step
            # only and never wrote ``sam_direction``, so it was dead state and
            # every intervening step degenerated to plain AdamW (no SAM at all).
            #
            # We also restore ``p.grad`` to the ORIGINAL gradient so the
            # immediately-following ``step()`` blends from a clean g (matching
            # the kernel contract where ``looksam_apply_step`` receives the
            # unperturbed grad and the freshly cached direction).
            for p, og, pg, sam_dir in zip(
                    params_list, orig_grads_list, perturbed_grads_list,
                    direction_list):
                sam_dir.copy_((pg - og).to(sam_dir.dtype))
                p.grad.copy_(og)

    @property
    def global_step(self) -> int:
        """Current global step count."""
        return self._global_step

    def should_sam_step(self) -> bool:
        """Return ``True`` if SAM should be applied at the current step."""
        k = self.param_groups[0]["k"]
        return self._global_step % k == 0

    def _single_param_step(self, param, group, state):
        """Per-parameter step used by the `use_grad_hooks=True` path.

        Performs the AdamW step only (SAM perturbation must be done separately).
        """
        if param.grad is None:
            return
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            state["sam_direction"] = torch.zeros_like(param, dtype=torch.float32)
        state["step"] += 1
        # Same canonical LookSAM blend as the batched step(): apply the cached
        # SAM direction every step (g_adj = (1-alpha)*g + alpha*sam_dir) so the
        # hooks path is also sharpness-aware on intervening steps, not plain
        # AdamW. sam_direction is refreshed by sam_step on the SAM steps.
        sam_dir = state.setdefault(
            "sam_direction", torch.zeros_like(param, dtype=torch.float32))
        alpha = group["alpha"]
        g_adj = (1.0 - alpha) * param.grad + alpha * sam_dir
        _ops.fused_adamw_simple_step(
            [param], [g_adj], [state["exp_avg"]], [state["exp_avg_sq"]],
            [state["step"]], group["betas"][0], group["betas"][1],
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
