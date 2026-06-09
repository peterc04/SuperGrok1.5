"""
Prodigy — Distance-aware self-tuning Adam.

Prodigy automatically estimates the optimal learning rate by tracking the
distance between the current parameters and their initial values. The
recommended default learning rate is ``1.0``; the internal ``d_lr`` scalar
is adjusted adaptively so that manual LR tuning is largely unnecessary.

All computation is dispatched to the fused C++/CUDA kernel via _ops.
"""

import math
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from grokking_optimizers.dispatch import get_ops

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
        d0: float = 1e-6,
        d_coef: float = 1.0,
        use_grad_hooks: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if d0 <= 0.0:
            raise ValueError(f"Invalid d0 value: {d0}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            d0=d0,
            d_coef=d_coef,
        )
        super().__init__(params, defaults)

        # Global adaptive learning rate shared across all parameter groups,
        # initialised to d0 (the Prodigy D-estimate's starting value).
        self._d_lr: float = d0
        # Persistent EMA accumulators for the Prodigy D-estimate (canonical
        # Mishchenko & Defazio 2023 / prodigyopt form). r/s are NO LONGER
        # recomputed instantaneously each step: they are running EMAs decayed
        # by beta3 = sqrt(beta2) across steps. r_ema tracks Σ d²·<g, p0−p>
        # (numerator) and s_ema tracks Σ d²·‖g‖₁ (denominator) accumulated over
        # the whole trajectory, so the estimate d_hat = d_coef·r_ema/|s_ema|
        # PLATEAUS once the parameters stop drifting from init instead of
        # ratcheting up forever on post-memorization gradient-noise spikes
        # (the instantaneous form's `max()` locked those spikes in → d blew up
        # → training collapsed). See the fused launcher for the decay-then-
        # accumulate wiring.
        self._r_ema: float = 0.0
        self._s_ema: float = 0.0

        # Per-group cache of static tensor lists (params + exp_avg/exp_avg_sq/s/
        # param_init buffers keep a fixed identity across steps); only grads_list
        # and step counters are refreshed per step. Invalidated by
        # add_param_group.
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
        """Return cached (params, exp_avg, exp_avg_sq, s, param_init, states)
        for *group*, keyed on the grad-bearing param ids."""
        key = tuple(id(p) for p in group["params"] if p.grad is not None)
        cached = self._static_cache.get(id(group))
        if cached is not None and cached[0] == key:
            return cached[1]

        params_list = []
        exp_avg_list = []
        exp_avg_sq_list = []
        s_list = []
        param_init_list = []
        states = []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["s"] = torch.zeros_like(p, dtype=torch.float32)
                state["param_init"] = p.detach().clone().float()
            params_list.append(p)
            exp_avg_list.append(state["exp_avg"])
            exp_avg_sq_list.append(state["exp_avg_sq"])
            s_list.append(state["s"])
            param_init_list.append(state["param_init"])
            states.append(state)
        entry = (params_list, exp_avg_list, exp_avg_sq_list, s_list,
                 param_init_list, states)
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
            fused_step = self._fused_step = _ops.bind("prodigy_fused_step")

        for group in self.param_groups:
            (params_list, exp_avg_list, exp_avg_sq_list, s_list,
             param_init_list, states) = self._group_cache(group)

            if len(params_list) == 0:
                continue

            grads_list = [p.grad for p in params_list]
            step_list = []
            for state in states:
                state["step"] += 1
                step_list.append(state["step"])

            beta1, beta2 = group["betas"]
            # beta3 governs the persistent-EMA decay of the D-estimate's
            # numerator/denominator (canonical Prodigy uses sqrt(beta2)).
            beta3 = math.sqrt(beta2)
            # The kernel decays the persistent (r_ema, s_ema) by beta3, adds
            # this step's reduction, updates d = max(d_prev, d_coef·r_ema/|s_ema|),
            # applies, and returns the new (d_lr, r_ema, s_ema).
            self._d_lr, self._r_ema, self._s_ema = fused_step(
                params_list,
                grads_list,
                exp_avg_list,
                exp_avg_sq_list,
                s_list,
                param_init_list,
                step_list,
                self._d_lr,
                self._r_ema,
                self._s_ema,
                beta1,
                beta2,
                beta3,
                group["lr"],
                group["weight_decay"],
                group["eps"],
                group["d0"],
                group["d_coef"],
            )

        return loss

    def _single_param_step(self, param, group, state):
        """Per-parameter step used by the `use_grad_hooks=True` path.

        KNOWN LIMITATION: the persistent-EMA D-estimate decays (r_ema, s_ema) by
        beta3 ONCE PER fused_step CALL. On this per-parameter hook path that is
        once per param per optimizer step, so a model with K parameters over-
        decays the shared EMA by beta3**K each step (vs the intended beta3). The
        D-estimate trajectory therefore differs from the standard (use_grad_hooks
        =False) path. The grokking race uses use_grad_hooks=False, so this does
        not affect it; fixing the hook path would require a per-step (not
        per-param) decay barrier and is left out of scope.
        """
        if param.grad is None:
            return
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            state["s"] = torch.zeros_like(param, dtype=torch.float32)
            state["param_init"] = param.data.clone().float()
        state["step"] += 1
        beta1, beta2 = group["betas"]
        beta3 = math.sqrt(beta2)
        self._d_lr, self._r_ema, self._s_ema = _ops.prodigy_fused_step(
            [param], [param.grad], [state["exp_avg"]], [state["exp_avg_sq"]],
            [state["s"]], [state["param_init"]], [state["step"]],
            getattr(self, '_d_lr', group["d0"]),
            getattr(self, '_r_ema', 0.0), getattr(self, '_s_ema', 0.0),
            beta1, beta2, beta3, group["lr"],
            group["weight_decay"], group["eps"],
            group["d0"], group["d_coef"],
        )

    @property
    def d_lr(self) -> float:
        """Current adaptive learning rate estimated by Prodigy."""
        return self._d_lr


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
