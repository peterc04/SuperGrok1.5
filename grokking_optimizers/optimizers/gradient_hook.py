"""
Phase 5B: GradientHookOptimizer — Cache-warm optimizer updates via
post_accumulate_grad_hook.

Instead of the traditional pattern where gradients go cold in L2 cache
between backward() and step():

    loss.backward()   # writes gradients to L2
    optimizer.step()  # reads gradients — likely evicted from L2

GradientHookOptimizer registers a per-parameter hook that fires immediately
after each gradient is accumulated, running the optimizer update while the
gradient data is still L2-warm. This eliminates the cold-cache penalty
entirely for bandwidth-bound optimizer kernels.

Requires PyTorch >= 2.1 (register_post_accumulate_grad_hook).
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Dict, Any, List

# Minimum PyTorch version check — post_accumulate_grad_hook added in 2.1
_PT_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
_HAS_GRAD_HOOK = _PT_VERSION >= (2, 1)


class GradientHookOptimizer(Optimizer):
    """Runs optimizer step per-parameter via post_accumulate_grad_hook.

    Each parameter's update runs immediately after its gradient is
    accumulated, while gradient data is L2-warm. This eliminates the
    cold-cache penalty of a separate opt.step() call.

    Works with any underlying optimizer that supports per-parameter stepping.
    Primarily designed for use with SuperGrok2's CUDA fused kernels.

    Usage::

        base_opt = SuperGrok2(model.parameters(), lr=1e-3)
        hook_opt = GradientHookOptimizer(model, base_opt)

        # Training loop — no explicit opt.step() needed!
        for batch in dataloader:
            hook_opt.zero_grad()
            loss = model(batch)
            loss.backward()
            # Updates happen automatically during backward!
            hook_opt.finish_step()  # bookkeeping: increment step count, etc.

    Notes:
        - The hook fires during backward(), so step() is a no-op for compute.
        - finish_step() handles step counting, LR scheduling, etc.
        - Not compatible with naive gradient accumulation; use the
          accumulate_steps parameter to handle accumulation correctly.
        - Thread-safe: hooks are serialized on the same CUDA stream, so
          no atomics are needed for the accumulation counter.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        base_optimizer: Optimizer,
        accumulate_steps: int = 1,
    ):
        if not _HAS_GRAD_HOOK:
            raise RuntimeError(
                f"GradientHookOptimizer requires PyTorch >= 2.1 for "
                f"register_post_accumulate_grad_hook support. "
                f"Current version: {torch.__version__}. "
                f"Please upgrade PyTorch or use the standard optimizer.step() pattern."
            )

        if accumulate_steps < 1:
            raise ValueError(
                f"accumulate_steps must be >= 1, got {accumulate_steps}"
            )

        self.model = model
        self.base_optimizer = base_optimizer
        self.accumulate_steps = accumulate_steps

        # Step tracking
        self._global_step: int = 0
        self._backward_count: int = 0  # counts backward passes for accumulation

        # Build a mapping from parameter id -> (param_group_idx, param_idx_in_group)
        # so the hook knows which group/index a parameter belongs to.
        self._param_to_location: Dict[int, tuple] = {}
        for group_idx, group in enumerate(base_optimizer.param_groups):
            for param_idx, p in enumerate(group["params"]):
                self._param_to_location[id(p)] = (group_idx, param_idx)

        # Register hooks on all parameters that require grad
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        self._hooked_params: List[torch.nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) not in self._param_to_location:
                # Parameter exists in the model but wasn't passed to the
                # base optimizer (e.g., frozen layers). Skip it.
                continue

            group_idx, param_idx = self._param_to_location[id(param)]
            hook_fn = self._make_hook(param, group_idx, param_idx)
            handle = param.register_post_accumulate_grad_hook(hook_fn)
            self._hook_handles.append(handle)
            self._hooked_params.append(param)

        if len(self._hook_handles) == 0:
            import warnings
            warnings.warn(
                "GradientHookOptimizer: no parameters were hooked. "
                "Ensure the model has parameters with requires_grad=True "
                "that are also in the base_optimizer's param_groups.",
                stacklevel=2,
            )

        # We inherit from Optimizer for interface compatibility but delegate
        # all real state to base_optimizer. Pass an empty param list.
        super().__init__([{"params": []}], defaults={})

    def _make_hook(self, param, param_group_idx, param_idx_in_group):
        """Create the post_accumulate_grad_hook for one parameter.

        The returned callable receives the parameter tensor as its sole
        argument (called by PyTorch after gradient accumulation).
        """
        base_opt = self.base_optimizer

        def hook(p: torch.Tensor) -> None:
            # If gradient is None (e.g., unused parameter in this backward),
            # skip the update.
            if p.grad is None:
                return

            # Gradient accumulation: only step when we've accumulated enough.
            # We check the *current* backward count (incremented after all
            # hooks fire in finish_step), so during intermediate accumulation
            # passes the count won't match and we skip.
            if self.accumulate_steps > 1:
                # During accumulation, _backward_count is 0..(accumulate_steps-1).
                # We only step when _backward_count == accumulate_steps - 1
                # (the last accumulation pass). The count is incremented in
                # finish_step() after backward completes.
                if self._backward_count < self.accumulate_steps - 1:
                    return

                # Scale the gradient by 1/accumulate_steps so the effective
                # gradient magnitude matches non-accumulated training.
                if p.grad is not None:
                    p.grad.div_(self.accumulate_steps)

            # Attempt to use the base optimizer's CUDA kernel directly for
            # maximum performance (avoids Python overhead of full step()).
            group = base_opt.param_groups[param_group_idx]
            state = base_opt.state[p]

            # If the base optimizer has a per-parameter step method, use it.
            # SuperGrok2 and similar optimizers store state keyed by parameter.
            if hasattr(base_opt, '_single_param_step'):
                base_opt._single_param_step(p, group, state)
            elif hasattr(base_opt, '_step_param'):
                base_opt._step_param(p, group, state)
            else:
                # Generic fallback: perform a minimal AdamW-style update
                # directly. This covers the case where the base optimizer
                # doesn't expose a per-parameter step API.
                _adamw_per_param(p, group, state, self._global_step + 1)

        return hook

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients on the base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def finish_step(self) -> None:
        """Called after backward(). Handles accumulation counting and step bookkeeping.

        In a standard training loop, call this after loss.backward():

            hook_opt.zero_grad()
            loss = model(batch)
            loss.backward()
            hook_opt.finish_step()

        With gradient accumulation (accumulate_steps > 1):

            for micro_step, batch in enumerate(micro_batches):
                loss = model(batch) / accumulate_steps
                loss.backward()
                hook_opt.finish_step()
                # zero_grad only after a full accumulation cycle
                if (micro_step + 1) % accumulate_steps == 0:
                    hook_opt.zero_grad()
        """
        self._backward_count += 1

        if self._backward_count >= self.accumulate_steps:
            # A full accumulation cycle completed — the hooks already ran
            # the optimizer updates on the last backward pass.
            self._backward_count = 0
            self._global_step += 1

            # Propagate the step count to the base optimizer if it tracks it.
            if hasattr(self.base_optimizer, '_global_step'):
                self.base_optimizer._global_step = self._global_step
            if hasattr(self.base_optimizer, '_step_counter'):
                self.base_optimizer._step_counter = self._global_step

    def step(self, closure=None) -> None:
        """No-op for compute — updates happen in the gradient hooks.

        Calls finish_step() for convenience so users can drop this in as a
        replacement for a standard optimizer.step() call. However, the
        recommended pattern is to call finish_step() explicitly.

        If a closure is provided, it is evaluated (this triggers backward
        and the hooks), and then finish_step() is called.
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        self.finish_step()

    def remove_hooks(self) -> None:
        """Remove all registered hooks. Call before deleting the optimizer.

        After calling this, the optimizer will no longer update parameters
        during backward(). You must call step() on the base_optimizer
        manually or create a new GradientHookOptimizer.
        """
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._hooked_params.clear()

    def state_dict(self) -> Dict[str, Any]:
        """Return the state dict of the base optimizer plus hook metadata."""
        base_state = self.base_optimizer.state_dict()
        base_state['gradient_hook_meta'] = {
            'global_step': self._global_step,
            'backward_count': self._backward_count,
            'accumulate_steps': self.accumulate_steps,
        }
        return base_state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state into the base optimizer and restore hook metadata."""
        meta = state_dict.get('gradient_hook_meta', None)
        # Pass a copy without hook metadata to avoid mutating the caller's dict
        base_state = {k: v for k, v in state_dict.items() if k != 'gradient_hook_meta'}
        self.base_optimizer.load_state_dict(base_state)

        if meta is not None:
            self._global_step = meta.get('global_step', 0)
            self._backward_count = meta.get('backward_count', 0)
            loaded_accum = meta.get('accumulate_steps', self.accumulate_steps)
            if loaded_accum != self.accumulate_steps:
                import warnings
                warnings.warn(
                    f"GradientHookOptimizer: loaded accumulate_steps={loaded_accum} "
                    f"differs from current accumulate_steps={self.accumulate_steps}. "
                    f"Using current value.",
                    stacklevel=2,
                )

    @property
    def param_groups(self):
        """Proxy to base optimizer's param_groups for LR scheduler compatibility."""
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value

    def __repr__(self) -> str:
        base_repr = repr(self.base_optimizer)
        return (
            f"GradientHookOptimizer(\n"
            f"  accumulate_steps={self.accumulate_steps},\n"
            f"  global_step={self._global_step},\n"
            f"  hooked_params={len(self._hook_handles)},\n"
            f"  base_optimizer={base_repr}\n"
            f")"
        )

    def __del__(self):
        """Clean up hooks when the optimizer is garbage-collected."""
        self.remove_hooks()


def _adamw_per_param(
    param: torch.Tensor,
    group: Dict[str, Any],
    state: Dict[str, Any],
    step: int,
) -> None:
    """Minimal AdamW update for a single parameter.

    Used as a generic fallback when the base optimizer doesn't expose a
    per-parameter step API. Implements decoupled weight decay AdamW
    (Loshchilov & Hutter, 2019).
    """
    grad = param.grad
    if grad is None:
        return

    lr = group.get('lr', 1e-3)
    beta1, beta2 = group.get('betas', (0.9, 0.999))
    eps = group.get('eps', 1e-8)
    weight_decay = group.get('weight_decay', 0.0)

    # Lazy state initialization
    if len(state) == 0:
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

    state['step'] = step
    exp_avg = state['exp_avg']
    exp_avg_sq = state['exp_avg_sq']

    # Bias correction
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    # Decoupled weight decay
    if weight_decay != 0.0:
        param.data.mul_(1.0 - lr * weight_decay)

    # Momentum updates
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    # Bias-corrected step
    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
    step_size = lr / bias_correction1
    param.data.addcdiv_(exp_avg, denom, value=-step_size)
