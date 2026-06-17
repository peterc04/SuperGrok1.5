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
        """Eager optimiser step — REMOVED (pure L3-TC)."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    def _single_param_step(self, param, group, state):
        """Per-parameter eager step (use_grad_hooks path) — REMOVED (pure L3-TC)."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")


# ── Shared (inlined) helper: register post_accumulate_grad_hook on each param.
# Each hook calls back into the optimizer's `_single_param_step` so the update
# runs while gradient data is still L2-warm. Duplicated across every optimizer
# file by design (self-containment); requires PyTorch >= 2.1.
def _register_grad_hooks(optimizer):
    # Fail-fast (construction time): the per-parameter eager `_single_param_step`
    # was removed under pure L3-TC (the megakernel owns the optimizer update via
    # fused_train_step), so the grad-hook path cannot run. Raise HERE — when the
    # optimizer is constructed with use_grad_hooks=True — rather than mid-backward
    # from inside the hook, so `--grad-hooks` fails immediately with a clear
    # message. The plumbing below is retained for when an eager path returns.
    raise NotImplementedError(
        "use_grad_hooks is not supported: the per-parameter eager step was "
        "removed under pure L3-TC (the megakernel owns the optimizer update via "
        "fused_train_step). Construct the optimizer with use_grad_hooks=False "
        "(the default) and drive training through the fused L3-TC megakernel.")
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
