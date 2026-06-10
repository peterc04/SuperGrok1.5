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

from grokking_optimizers.dispatch import get_ops
from grokking_optimizers.optimizers.adamw import _validate_grad

_ops = get_ops()  # Fails loudly if C++ extension not built


class Muon(Optimizer):
    """Orthogonalised update optimizer for 2D params, AdamW for others.

    This optimizer manages two distinct parameter groups internally:
      - **2D params** (e.g. weight matrices): updated with Muon
        (momentum + Newton-Schulz orthogonalisation).
      - **1D params** (biases, norms, etc.): updated with AdamW.

    Two construction forms:
      - **Explicit split** ``Muon(params_2d, params_1d, ...)``: you pre-route the
        2D (matrix) params and the 1D/other params yourself.
      - **Single iterable** ``Muon(params, ...)`` (``params_1d`` left as None):
        the constructor AUTO-SPLITS ``params`` by ``p.ndim`` — every 2D tensor
        goes to the Muon group, everything else (1D biases/norms, >=3D conv
        weights) to the AdamW group. This mirrors the standard PyTorch optimizer
        signature ``Opt(model.parameters(), ...)``.

    Back-compat: an all-2D iterable is a no-op under auto-split (every param
    lands in the 2D group, exactly as if passed as ``params_2d``). The only
    behavioural change is mixed-ndim input passed positionally as the first arg:
    previously those 1D params reached the Newton-Schulz path and tripped the
    ``muon_fused_step`` ``p.dim()==2`` assert; now they are routed to AdamW —
    strictly an improvement, not a regression.

    Args:
        params_2d: Iterable of 2D parameters for Muon updates. If ``params_1d``
            is omitted, this is treated as a single combined iterable and split
            by ``ndim`` into the 2D (Muon) and non-2D (AdamW) groups.
        params_1d: Iterable of non-2D parameters for AdamW updates
            (optional). When omitted, ``params_2d`` is auto-split by ``ndim``.
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
        use_grad_hooks: bool = False,
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

        if params_1d is None:
            # Single-iterable form: consume once, then split by ndim. 2D matrices
            # → Muon (Newton-Schulz); everything else → AdamW. (Consuming into a
            # list first is required so a generator is not exhausted by the
            # split scan.)
            all_p = list(params_2d)
            params_2d_list = [p for p in all_p if getattr(p, "ndim", None) == 2]
            params_1d_list = [p for p in all_p if getattr(p, "ndim", None) != 2]
        else:
            # Explicit split: caller pre-routed the two groups (verbatim path).
            params_2d_list = list(params_2d)
            params_1d_list = list(params_1d)

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

        # Per-group cache of static tensor lists (params + state buffers keep a
        # fixed identity across steps); only grads_list and step counters are
        # refreshed per step. Invalidated by add_param_group.
        self._static_cache: dict = {}
        # Lazily-bound fused kernel callables (resolved once at first use).
        self._muon_fused_step = None
        self._adamw_fused_step = None

        self._use_grad_hooks = use_grad_hooks
        if use_grad_hooks:
            _register_grad_hooks(self)

    def add_param_group(self, param_group) -> None:
        self._static_cache = {}
        super().add_param_group(param_group)

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

        for group in self.param_groups:
            group_type = group.get("group_type", "muon")

            if group_type == "muon":
                self._step_muon(group)
            else:
                self._step_adamw(group)

        return loss

    def _muon_cache(self, group: dict):
        """Cached (params, momentum_buffers) for a Muon group, keyed on the
        grad-bearing param ids."""
        key = tuple(id(p) for p in group["params"] if p.grad is not None)
        cached = self._static_cache.get(id(group))
        if cached is not None and cached[0] == key:
            return cached[1]
        params_list = []
        momentum_buf_list = []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(
                    p, dtype=torch.float32)
            params_list.append(p)
            momentum_buf_list.append(state["momentum_buffer"])
        entry = (params_list, momentum_buf_list)
        self._static_cache[id(group)] = (key, entry)
        return entry

    def _adamw_cache(self, group: dict):
        """Cached (params, exp_avg, exp_avg_sq, states) for an AdamW group,
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
            params_list.append(p)
            exp_avg_list.append(state["exp_avg"])
            exp_avg_sq_list.append(state["exp_avg_sq"])
            states.append(state)
        entry = (params_list, exp_avg_list, exp_avg_sq_list, states)
        self._static_cache[id(group)] = (key, entry)
        return entry

    def _step_muon(self, group: dict) -> None:
        """Apply Muon (momentum + Newton-Schulz orthogonalisation) to 2D params."""
        params_list, momentum_buf_list = self._muon_cache(group)

        if len(params_list) == 0:
            return

        grads_list = [_validate_grad(p) for p in params_list]

        fused_step = self._muon_fused_step
        if fused_step is None:
            fused_step = self._muon_fused_step = _ops.bind("muon_fused_step")
        fused_step(
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
        params_list, exp_avg_list, exp_avg_sq_list, states = \
            self._adamw_cache(group)

        if len(params_list) == 0:
            return

        grads_list = [_validate_grad(p) for p in params_list]
        step_list = []
        for state in states:
            state["step"] += 1
            step_list.append(state["step"])

        betas = group.get("betas", (0.9, 0.98))
        eps = group.get("eps", 1e-8)

        fused_step = self._adamw_fused_step
        if fused_step is None:
            fused_step = self._adamw_fused_step = _ops.bind(
                "fused_adamw_simple_step")
        fused_step(
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
        """Per-parameter step used by the `use_grad_hooks=True` path."""
        if param.grad is None:
            return
        grad = _validate_grad(param)
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
        _ops.muon_fused_step(
            [param], [grad], [state["momentum_buffer"]],
            group.get("momentum", 0.95),
            group["lr"],
            group["weight_decay"],
            group.get("ns_steps", 5),
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
