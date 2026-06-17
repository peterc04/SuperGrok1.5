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
from grokking_optimizers.optimizers.adamw import _validate_grad

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
        """Eager optimiser step — REMOVED (pure L3-TC).

        On the L3-TC path the SAM-step cadence is host-computed by
        ``dispatch._opt_scalars_from`` (the every-k gate on the fused-train step
        counter) and the LookSAM blend + 2nd backward run IN the megakernel; the
        eager blend + AdamW launch here is gone. ``should_sam_step`` is kept as a
        host-side helper (no longer on the production path)."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    @torch.no_grad()
    def sam_step(
        self,
        model: torch.nn.Module,
        train_x: Tensor,
        train_y: Tensor,
        criterion: Callable,
    ) -> None:
        """Eager LookSAM 2nd-backward direction step — REMOVED (pure L3-TC).

        The model-coupled SAM perturb / 2nd forward+backward / restore /
        ``sam_dir = g_sam - g`` cache now run IN the megakernel (the in-kernel P2.4
        SAM phase, gated by FusedScalars.looksam_sam). The eager kernel body
        (``_ops.looksam_perturb_all`` / ``_ops.looksam_restore_all``) is removed."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    @property
    def global_step(self) -> int:
        """Current global step count."""
        return self._global_step

    def should_sam_step(self) -> bool:
        """Return ``True`` if SAM should be applied at the current step."""
        k = self.param_groups[0]["k"]
        return self._global_step % k == 0

    def state_dict(self) -> dict:
        """Include the SAM-cadence counter in the checkpoint.

        ``_global_step`` (which drives ``should_sam_step`` every k steps) lives on
        the optimizer instance, not the per-parameter ``state`` dict, so the base
        ``Optimizer.state_dict`` would drop it and a resumed run would mis-phase
        its SAM cadence. Stash it under a private ``"_looksam"`` key.
        """
        sd = super().state_dict()
        sd["_looksam"] = {"_global_step": self._global_step}
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the SAM-cadence counter, then the base state.

        Pops the private ``"_looksam"`` blob (default-absent so OLD checkpoints
        load cleanly with the constructor-initialised ``_global_step``).
        """
        sd = dict(state_dict)
        blob = sd.pop("_looksam", None)
        if blob is not None:
            self._global_step = blob.get("_global_step", self._global_step)
        super().load_state_dict(sd)

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
