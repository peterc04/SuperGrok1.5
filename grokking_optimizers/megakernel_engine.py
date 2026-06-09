"""grokking_optimizers.megakernel_engine — §8.4 fused-kernel ↔ framework adapter.

THE TRICKY PART (§8.4). DeepSpeed and Megatron-LM are both built around a
**separate** forward / backward / `optimizer.step()` interface:

    out  = engine(batch)        # forward
    engine.backward(loss)       # backward — populates .grad on every param
    engine.step()               # optimizer.step() + grad clear

But the Stage-6 megakernel (``grokking_optimizers.megakernel``) is a **fused L3
single launch**: one persistent kernel does forward **and** backward **and** the
optimizer update inside one kernel invocation (``FusionTier.L3_FWD_BWD_OPT``).
There is no separate backward to hook, and no separate `.step()` — the fused
launch already did all three. The framework's contract and the fused kernel's
contract are therefore in direct conflict.

This module is the **adapter** that reconciles them. It does two things:

1. :class:`MegakernelOptimizer` — a ``torch.optim.Optimizer`` subclass that
   presents the fused step as a standard ``.step()`` so DeepSpeed/Megatron can
   drive it as a ``client_optimizer``. Its ``.step()`` is where the actual fused
   launch is invoked (for L3) or where the inner optimizer's real ``.step()``
   runs (for L1/L2). It degrades to a plain pass-through optimizer when no
   framework is present.

2. :class:`FusedBackwardHook` — intercepts / no-ops the framework's separate
   ``.backward()`` **when the fused path owns fwd+bwd+opt** (L3). When the fused
   kernel already produced gradients-and-update in the forward launch, letting
   the framework run its own autograd backward would (a) waste a full backward
   pass and (b) double-apply the update. The hook short-circuits the framework
   backward and instead just marks gradients as "owned by the fused kernel" so
   ``.step()`` knows not to re-run them.

The decision of *which* mode a given (model, optimizer, arch) cell is in comes
from the Stage-6 solver: :func:`grokking_optimizers.megakernel.solve` returns a
:class:`~grokking_optimizers.megakernel.FusionPlan` whose ``tier`` tells us how
much the fused kernel owns. The control flow below is keyed entirely off that
tier.

────────────────────────────── CONTROL FLOW ──────────────────────────────────

  query solve(model, optimizer, arch) → plan.tier

  ┌─ L3_FWD_BWD_OPT  (fused kernel owns fwd + bwd + opt) ─────────────────────┐
  │  forward(batch):                                                          │
  │     • the fused L3 launch runs fwd+bwd+opt in ONE kernel.                 │
  │     • returns loss; gradients + the parameter update are ALREADY applied. │
  │     • adapter sets `_fused_owned = True`.                                 │
  │  engine.backward(loss):                                                   │
  │     • FusedBackwardHook.intercepts() → True → **NO-OP**. The framework's  │
  │       autograd backward is suppressed (it would redo work + double-apply).│
  │  engine.step()  →  MegakernelOptimizer.step():                           │
  │     • sees `_fused_owned` → the update already happened in forward → it   │
  │       only does framework bookkeeping (zero_grad bookkeeping, lr-sched    │
  │       coupling) and returns. It does NOT re-run the optimizer math.       │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ L2_BWD_OPT  (fused kernel owns bwd + opt; framework owns fwd) ───────────┐
  │  forward(batch):  framework's normal forward (autograd graph built).      │
  │  engine.backward(loss):                                                   │
  │     • FusedBackwardHook.intercepts() → True for the FUSED region: the     │
  │       fused bwd+opt launch runs, producing grads + applying the update.   │
  │     • framework autograd backward is suppressed for the fused params.     │
  │  engine.step():  update already applied by the fused bwd+opt → no-op math.│
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ L1_OPT_ONLY  (fused kernel owns ONLY the optimizer step) ────────────────┐
  │  forward(batch):  framework forward.                                      │
  │  engine.backward(loss):                                                   │
  │     • FusedBackwardHook.intercepts() → False → framework runs its OWN     │
  │       autograd backward normally (grads populated the usual way).         │
  │  engine.step()  →  MegakernelOptimizer.step():                           │
  │     • runs the fused optimizer-only kernel via the wrapped optimizer's    │
  │       real `.step()` (which itself dispatches to the L1 fused tail).      │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ L0_UNFUSED  (no fusion — solver would have errored for a megakernel) ────┐
  │  Pure pass-through: framework fwd, framework bwd, wrapped optimizer step. │
  └───────────────────────────────────────────────────────────────────────────┘

IMPORT SAFETY. This module imports only ``torch`` and the pure-Python Stage-6
solver at module load. The DeepSpeed / Megatron framework objects are *never*
imported here; the adapter is shaped to fit their duck-typed `client_optimizer`
/ backward-hook protocols, so it works when they are present and degrades to a
plain torch optimizer when they are absent. No GPU, no launched job, and no
DeepSpeed are required to import or unit-test the dispatch logic.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim import Optimizer

from grokking_optimizers.megakernel import FusionPlan, FusionTier, solve


# ─────────────────────── unified cross-arch fused dispatch ───────────────────

def dispatch_fused_megakernel(model: str, optimizer: str, *, params=None,
                              grads=None, opt_state=None, inputs=None,
                              lr: float = 1e-3, state=None):
    """Route a fused L3/L1 megakernel step to the backend for the detected arch.

    The single cross-arch entry that unifies the three real composition paths
    (Phase 3):
      * tpu_v6e → csrc.backends.pallas._pallas_fused.fused_step (jax.jit fused
        program; the 33 tpu cells bind to it).
      * sm_90 / gfx942 → the C++ `fused_step` pybind (dispatch.cpp routes to the
        real composed `mega_<model>_<opt>` launcher; gfx942 host launch is 🟡).

    Returns the backend's result (TPU: (new_params, new_state)). Raises a clear
    error if the cell/binding is unavailable on this build (no silent no-op).
    """
    from grokking_optimizers.dispatch import detect_arch, normalize_arch
    arch = detect_arch()
    # detect_arch() now reports the REAL arch (e.g. sm_80 / sm_90a / gfx942).
    # The megakernel solver keys on the impl-family label ("sm_90"/"gfx942"/
    # "tpu_v6e"), so normalise the real arch to its family for the solve() call.
    impl = normalize_arch(arch)
    solver_arch = (impl if isinstance(impl, str)
                   else ("sm_90" if impl == 90 else "gfx942"))
    tier = ("L3" if solve(model, optimizer, solver_arch).tier
            == FusionTier.L3_FWD_BWD_OPT else "L1")
    if arch == "tpu_v6e":
        from csrc.backends.pallas._pallas_fused import fused_step as tpu_fused
        return tpu_fused(model, optimizer, params=params, grads=grads,
                         opt_state=opt_state, inputs=inputs, lr=lr, tier=tier)
    # NVIDIA / AMD: the C++ megakernel dispatch (real composition per dispatch.cpp).
    from grokking_optimizers.dispatch import get_ops
    ops = get_ops()
    if not hasattr(ops, "fused_step"):
        raise RuntimeError(
            "C++ fused_step binding unavailable; build the extension "
            "(WITH_CUDA/WITH_HIP) to use the GPU fused megakernel path.")
    if params is None or state is None:
        raise ValueError("GPU fused_step needs params, grads, and a state tensor "
                         "([m|v|extra] per param-tensor).")
    ops.fused_step(model, optimizer, params,
                   inputs if inputs is not None else params,
                   grads if grads is not None else params, state, float(lr))
    return params


# ─────────────────────────── fused-ownership query ───────────────────────────


def resolve_fusion_plan(
    model_name: str, optimizer_name: str, arch: str
) -> Optional[FusionPlan]:
    """Query the Stage-6 solver for the (model, optimizer, arch) fusion plan.

    Returns ``None`` (not an error) when the cell is infeasible or the names are
    unknown — the adapter then falls back to the fully-unfused pass-through path,
    which is always correct. This keeps the adapter import-safe and robust to a
    model/optimizer the solver doesn't model.
    """
    try:
        return solve(model_name, optimizer_name, arch)
    except (KeyError, RuntimeError):
        return None


def _tier_of(plan: Optional[FusionPlan]) -> FusionTier:
    return plan.tier if plan is not None else FusionTier.L0_UNFUSED


def fused_owns_backward(plan: Optional[FusionPlan]) -> bool:
    """True iff the fused kernel owns the backward pass (tiers L2/L3).

    When True, the framework's separate ``.backward()`` must be suppressed by
    :class:`FusedBackwardHook` to avoid redundant work + a double parameter
    update.
    """
    return _tier_of(plan) >= FusionTier.L2_BWD_OPT


def fused_owns_step(plan: Optional[FusionPlan]) -> bool:
    """True iff the fused kernel already applied the optimizer update inside the
    fused launch (tiers L2/L3). When True, ``.step()`` must NOT re-run the math.
    """
    return _tier_of(plan) >= FusionTier.L2_BWD_OPT


# ─────────────────────────── FusedBackwardHook ───────────────────────────────


class FusedBackwardHook:
    """Intercepts / no-ops the framework's separate backward when fused (§8.4).

    Usage shapes (both duck-typed, framework-agnostic):

    * **Explicit guard** — wrap the framework backward call::

          if not hook.intercept_backward(loss):
              engine.backward(loss)   # framework path (L0/L1 only)

      ``intercept_backward`` returns True when the fused kernel already owns the
      backward (L2/L3), in which case the framework backward is skipped and the
      fused launch is invoked instead.

    * **Tensor hook** — :meth:`as_tensor_hook` returns a callable suitable for
      ``loss.register_hook(...)`` that zeroes the incoming grad when the fused
      path owns it, so an accidental ``loss.backward()`` is neutralized rather
      than double-applying.
    """

    def __init__(
        self,
        plan: Optional[FusionPlan],
        fused_launch: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.plan = plan
        # The callable that actually performs the fused bwd(+opt) launch. In the
        # real build this routes into the dispatch.cpp fused TU; here it is
        # injected so the control flow is testable without a kernel.
        self._fused_launch = fused_launch
        # Set True after a fused launch has applied the update this iteration,
        # so MegakernelOptimizer.step() knows the math already ran.
        self.fused_update_applied = False

    @property
    def owns_backward(self) -> bool:
        return fused_owns_backward(self.plan)

    def intercept_backward(self, loss: "torch.Tensor", **kwargs: Any) -> bool:
        """Decide whether to suppress the framework backward.

        Returns
        -------
        bool
            True  → the fused kernel owns backward (L2/L3); the framework's
                    ``.backward()`` was suppressed and the fused launch (if
                    provided) was invoked here instead.
            False → L0/L1; the caller must run the framework backward itself.
        """
        if not self.owns_backward:
            return False
        # Fused path owns backward: run the fused launch (bwd[+opt]) if wired,
        # and mark the update as applied so .step() is a no-op on the math.
        if self._fused_launch is not None:
            self._fused_launch(loss, **kwargs)
        self.fused_update_applied = fused_owns_step(self.plan)
        return True

    def as_tensor_hook(self) -> Callable[["torch.Tensor"], "torch.Tensor"]:
        """Return a ``Tensor.register_hook`` callable that neutralizes grads.

        If the fused path owns backward and an autograd backward is nonetheless
        triggered, this zeroes the gradient flowing back so nothing is
        double-counted. A pass-through (identity) when the framework legitimately
        owns backward (L0/L1).
        """

        owns = self.owns_backward

        def _hook(grad: "torch.Tensor") -> "torch.Tensor":
            if owns:
                return torch.zeros_like(grad)
            return grad

        return _hook

    def reset(self) -> None:
        """Clear per-iteration state (call at the top of each training step)."""
        self.fused_update_applied = False


# ─────────────────────────── MegakernelOptimizer ─────────────────────────────


class MegakernelOptimizer(Optimizer):
    """Present the fused L3 megakernel step as a standard ``torch.optim`` step.

    Wraps an *inner* optimizer (one of the 11 race optimizers — an
    ``Optimizer`` whose real ``.step()`` dispatches into the fused kernel for the
    L1 optimizer-only tail). Depending on the fusion tier reported by the
    Stage-6 solver, ``.step()`` either:

    * **L3 / L2** — the fused launch (driven through :class:`FusedBackwardHook`
      during the forward/backward phase) already applied the parameter update;
      ``.step()`` performs only framework bookkeeping and returns. It does NOT
      re-run the optimizer math (that would double-apply).
    * **L1 / L0** — calls the inner optimizer's real ``.step()`` (which itself
      fuses the optimizer-only tail at L1, or runs unfused at L0).

    The class satisfies DeepSpeed's ``client_optimizer`` shape: it subclasses
    ``torch.optim.Optimizer``, exposes ``param_groups`` / ``state`` (delegated to
    the inner optimizer), and a no-arg-compatible ``.step()``. With no framework
    present it behaves as a transparent wrapper around the inner optimizer.
    """

    def __init__(
        self,
        inner_optimizer: Optimizer,
        model_name: str,
        arch: str,
        optimizer_name: Optional[str] = None,
        hook: Optional[FusedBackwardHook] = None,
    ) -> None:
        if not isinstance(inner_optimizer, Optimizer):
            raise TypeError(
                "inner_optimizer must be a torch.optim.Optimizer, got "
                f"{type(inner_optimizer).__name__}"
            )
        self.inner = inner_optimizer
        self.model_name = model_name
        self.arch = arch
        self.optimizer_name = optimizer_name or _infer_optimizer_name(inner_optimizer)
        self.plan = resolve_fusion_plan(self.model_name, self.optimizer_name, self.arch)
        self.hook = hook or FusedBackwardHook(self.plan)
        # We deliberately do NOT call super().__init__ with fresh defaults — we
        # delegate param_groups/state to the inner optimizer so DeepSpeed sees
        # the real parameters, learning rates, and state. The Optimizer
        # protocol only requires param_groups + state + step + zero_grad, all of
        # which we forward.
        self.defaults = getattr(inner_optimizer, "defaults", {})

    # ── Optimizer protocol delegation (DeepSpeed client_optimizer shape) ─────

    @property
    def param_groups(self):  # type: ignore[override]
        return self.inner.param_groups

    @param_groups.setter
    def param_groups(self, value):  # type: ignore[override]
        self.inner.param_groups = value

    @property
    def state(self):  # type: ignore[override]
        return self.inner.state

    @state.setter
    def state(self, value):  # type: ignore[override]
        self.inner.state = value

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state_dict):
        return self.inner.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = True):  # type: ignore[override]
        return self.inner.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        return self.inner.add_param_group(param_group)

    # ── the fused-aware step (the §8.4 reconciliation point) ─────────────────

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):  # type: ignore[override]
        """Fused-aware optimizer step.

        Control flow (mirrors the module docstring):

        * If the fused kernel already applied the update this iteration
          (``hook.fused_update_applied`` — set by :class:`FusedBackwardHook` for
          L2/L3), the math is **already done**: we only run the closure (for
          loss reporting) and reset the hook, then return. Re-running
          ``inner.step()`` here would double-apply the update.
        * Otherwise (L0/L1, or no fused launch happened), delegate to the inner
          optimizer's real ``.step()`` — which fuses the optimizer-only tail at
          L1 or runs unfused at L0.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if fused_owns_step(self.plan) and self.hook.fused_update_applied:
            # L2/L3 fused path already updated the params in the fused launch.
            # Bookkeeping only — do NOT re-run the optimizer math.
            self.hook.reset()
            return loss

        # L0/L1 (or fused launch did not run this step): real optimizer step.
        result = self.inner.step()
        self.hook.reset()
        return loss if loss is not None else result

    # ── convenience for the framework-driven training loop ───────────────────

    def backward(self, loss: "torch.Tensor", **kwargs: Any) -> bool:
        """Framework-facing backward shim.

        Returns True if the fused kernel owned (and performed) the backward, in
        which case the caller must NOT also run the framework autograd backward.
        Returns False for L0/L1, signalling the caller to run its own backward.
        """
        return self.hook.intercept_backward(loss, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        tier = _tier_of(self.plan).name
        return (
            f"MegakernelOptimizer(inner={type(self.inner).__name__}, "
            f"model={self.model_name!r}, arch={self.arch!r}, "
            f"optimizer={self.optimizer_name!r}, tier={tier})"
        )


def _infer_optimizer_name(opt: Optimizer) -> str:
    """Best-effort map from an optimizer instance to its solver name key.

    The Stage-6 solver keys on the lower-cased canonical names in
    ``megakernel.OPTIMIZERS``. We match by class name, tolerating the common
    ``SuperGrok2`` → ``supergrok2`` spelling. Unknown ⇒ returns the lowercased
    class name (the solver will report it as unknown and the adapter falls back
    to the unfused path).
    """
    from grokking_optimizers.megakernel import OPTIMIZERS  # lazy, pure-Python

    cls = type(opt).__name__.lower()
    # Strip common decorator prefixes.
    for prefix in ("compiled", "moeaware"):
        if cls.startswith(prefix):
            cls = cls[len(prefix):]
    if cls in OPTIMIZERS:
        return cls
    # Map e.g. "grokadamw" already matches; "looksam" matches; fall through.
    for canon in OPTIMIZERS:
        if cls == canon:
            return canon
    return cls


def build_client_optimizer(
    inner_optimizer: Optimizer,
    model_name: str,
    arch: str,
    optimizer_name: Optional[str] = None,
) -> MegakernelOptimizer:
    """Factory: wrap ``inner_optimizer`` as a fused-aware DeepSpeed client_optimizer.

    Hand the returned object to ``deepspeed.initialize(optimizer=...)`` (or use
    it directly in a plain torch loop). It carries its own
    :class:`FusedBackwardHook`, reachable as ``.hook`` for the backward-shim
    wiring in the training loop.
    """
    return MegakernelOptimizer(
        inner_optimizer=inner_optimizer,
        model_name=model_name,
        arch=arch,
        optimizer_name=optimizer_name,
    )


def fused_training_step(
    optimizer: MegakernelOptimizer,
    loss: "torch.Tensor",
    framework_backward: Optional[Callable[["torch.Tensor"], None]] = None,
) -> None:
    """Reference glue for one fused-aware training step (§8.4 control flow).

    Encapsulates the backward/step reconciliation so a framework loop (or the
    test harness) can call a single function:

        1. Ask the hook whether the fused kernel owns backward.
        2. If not (L0/L1), run ``framework_backward(loss)`` (the engine's
           autograd backward).
        3. Call ``optimizer.step()`` — which is a no-op on the math for L2/L3
           (already applied) and the real step for L0/L1.

    ``framework_backward`` defaults to ``loss.backward`` when not supplied.
    """
    if not optimizer.backward(loss):
        bwd = framework_backward or (lambda x: x.backward())
        bwd(loss)
    optimizer.step()


def all_reduce_optimizer_hooks(
    optimizer: Optimizer, params: Optional[Iterable["torch.Tensor"]] = None
) -> None:
    """Invoke an optimizer's existing data-parallel allreduce hooks if present.

    Several race optimizers (notably SuperGrok2 via ``_allreduce_meta_grads`` /
    ``_allreduce_expert_counts``) already carry DP-aware allreduce hooks. The
    distributed layer calls this before the fused step so those hooks fire under
    the launched job; it is a guarded no-op when the hooks/job are absent.
    """
    inner = getattr(optimizer, "inner", optimizer)
    for hook_name in ("_allreduce_meta_grads", "_allreduce_expert_counts"):
        fn = getattr(inner, hook_name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                # Hooks self-guard on dist.is_initialized(); never fatal here.
                pass
    _ = params


__all__ = [
    "MegakernelOptimizer",
    "FusedBackwardHook",
    "resolve_fusion_plan",
    "fused_owns_backward",
    "fused_owns_step",
    "build_client_optimizer",
    "fused_training_step",
    "all_reduce_optimizer_hooks",
    "dispatch_fused_megakernel",
]
