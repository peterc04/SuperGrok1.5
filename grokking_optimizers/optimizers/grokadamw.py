"""
GrokAdamW — Adam with grokking-aware EMA gradient filtering and amplification.

Combines AdamW with an exponential moving average (EMA) gradient filter that
detects and amplifies slow-learning gradient signals, accelerating the
transition from memorisation to generalisation (grokking).

This is the PUBLISHED GrokAdamW algorithm (QuixiAI/grokadamw aka
cognitivecomputations/grokadamw; also HF transformers PR #32521). Two
mechanisms beyond plain EMA-filtered AdamW make it "grokking-aware", BOTH wired
here (they were previously dead constructor args):

  (1) LAYER-WISE β1 MOMENTUM DECAY.  Per the reference (grokadamw.py):
          layer_beta1 = beta1 * (1 - gamma) ** i
      where ``i`` is the enumeration index of the parameter tensor. Earlier
      layers keep the full β1; later layers get a progressively smaller β1
      (less momentum), which the paper motivates as promoting early-layer
      generalisation. gamma default 0.1. Implemented with the SAME flat-index
      pattern as supergrok11.py's ``_flat_layer_beta1s``.

  (2) GROKKING-SIGNAL-DRIVEN ADAPTIVE α.  α is the EMA decay of the slow-grad
      filter; the reference modulates it from a grokking signal. VERBATIM from
      the reference (grokadamw.py):
          # _default_grokking_signal(train_loss, eval_loss):
          diff      = max(0, eval_loss - train_loss)
          max_loss  = max(eval_loss, train_loss)
          signal    = diff / max_loss if max_loss > 0 else 0.0
          # then, per param-group:
          alpha = alpha_init * math.exp(-grokking_signal_decay_rate * signal)
      i.e. ALPHA SCHEDULE  α_t = alpha_init · exp(−κ · signal),  with the
      signal the *max-normalised* (NOT train-normalised) clipped val−train gap
      and κ == ``grokking_signal_decay_rate`` (default 0.1). The slow-grad
      filter + amplification it feeds is unchanged and identical to the kernel
      math (csrc/algorithms/grokadamw.h):
          grok_ema  = α·grok_ema + (1−α)·grad
          grok_grad = grad + lamb·grok_ema
      lamb default 2.0 in the reference. (Our historical default was 5.0; we
      keep the constructor default at the reference value 2.0 — see __init__.)

INTERPRETATION / SOURCE NOTES (read LOUDLY):
  * The reference α update lives in ``_default_grokking_signal`` +
    the per-group ``alpha = alpha_init * math.exp(...)`` line of
    cognitivecomputations/grokadamw grokadamw.py. We transcribe THAT formula
    (max-normalised signal, exp decay), NOT SuperGrok11's train-normalised
    variant — the two differ only in the signal denominator and this one is
    the published GrokAdamW.
  * The fused kernel consumes a SCALAR α per multi-tensor call, so the live
    α_t is simply passed through. β1 is per-tensor: until the vector ABI lands
    we partition the fused call per tensor (each with its own β1_i); post
    rebuild a single call passes the whole β1 vector. See ``step``.

All hot-path computation is dispatched to the fused C++/CUDA kernel via _ops.
"""

import math
import warnings
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


@torch.no_grad()
def _adamw_step_reference(params, grads, exp_avgs, exp_avg_sqs, steps,
                          lr, beta1, beta2, eps, wd):
    """Pure-Python AdamW reference step (decoupled weight decay).

    Kept as documentation of the math the fused kernel implements; not
    invoked on the hot path (the C++ extension's grokadamw_fused_step
    handles every GPU/CPU case).
    """
    for p, g, ea, easq, step in zip(params, grads, exp_avgs, exp_avg_sqs, steps):
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        ea.mul_(beta1).add_(g, alpha=1 - beta1)
        easq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        step_size = lr / bc1
        denom = (easq / bc2).sqrt().add_(eps)
        p.mul_(1 - lr * wd)
        p.addcdiv_(ea, denom, value=-step_size)


class GrokAdamW(Optimizer):
    """Adam with grokking-aware EMA gradient filtering and amplification.

    Implements the PUBLISHED GrokAdamW (cognitivecomputations/grokadamw):
    layer-wise β1 decay + grokking-signal-driven adaptive α (see module
    docstring for the transcribed formulas and source lines).

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient
            and its square (default: (0.9, 0.98)). The reference uses
            (0.9, 0.999); we keep our (0.9, 0.98) project default.
        eps: Term added to the denominator for numerical stability
            (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 1.0).
        alpha: ``alpha_init`` — the BASE EMA decay of the slow-grad filter
            (default: 0.98). When ``step`` is given both ``train_loss`` and
            ``val_loss``, the LIVE decay becomes
            ``alpha_t = alpha * exp(-kappa * signal)`` (reference formula).
        lamb: Amplification factor applied to the filtered gradient
            signal (default: 2.0 — the reference default; was 5.0 historically).
        gamma: LAYER-WISE β1 decay rate (WIRED, not deprecated). β1 for the
            i-th parameter tensor is ``betas[0] * (1 - gamma) ** i`` (reference
            grokadamw.py). gamma=0 ⇒ a single global β1 (identical layers);
            gamma>0 ⇒ later layers get less momentum (default: 0.1).
        kappa: ``grokking_signal_decay_rate`` — the exponential decay rate of
            α vs the grokking signal (reference name; default: 0.1). Only
            active when ``step`` receives both train_loss and val_loss.
        decay: Back-compat alias accepted but NOT used by the published
            algorithm (the reference has no second decay knob). Kept so old
            call sites do not break; silently ignored (default: 0.1).
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
        alpha: float = 0.98,
        lamb: float = 2.0,
        gamma: float = 0.1,
        kappa: float = 0.1,
        decay: float = 0.1,
        grad_clip: float = 1.0,
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
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if grad_clip <= 0.0:
            raise ValueError(f"Invalid grad_clip (must be > 0): {grad_clip}")
        if gamma is None:
            gamma = 0.0
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma (layer-wise β1 decay, want "
                             f"0<=gamma<1): {gamma}")
        if kappa is None:
            kappa = 0.0
        if kappa < 0.0:
            raise ValueError(f"Invalid kappa (grokking_signal_decay_rate, want "
                             f">=0): {kappa}")
        # `decay` is the one remaining back-compat-only arg: the PUBLISHED
        # GrokAdamW has no second decay knob, so we accept it and ignore it
        # (no DeprecationWarning churn — it simply has no published role).

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            lamb=lamb,
            gamma=gamma,
            kappa=kappa,
            decay=decay,
            grad_clip=grad_clip,
        )
        super().__init__(params, defaults)

        # Per-group cache of static tensor lists (params + exp_avg/exp_avg_sq/ema
        # buffers keep a fixed identity across steps); only grads_list and step
        # counters are refreshed per step. Invalidated by add_param_group.
        self._static_cache: dict = {}
        # Lazily-bound fused kernel callable (resolved once at first step()).
        self._fused_step = None

        # ── Layer-wise β1 (PUBLISHED GrokAdamW): β1_i = β1 * (1 - gamma)**i,
        # i the flat enumeration index of the param tensor — the SAME pattern as
        # supergrok11.py::_flat_layer_beta1s. Precompute the per-param-id β1 so
        # the hot path is a dict lookup. Index runs across ALL groups (global
        # layer order), matching the reference's single enumerate(params).
        self._layer_beta1_by_id: dict = {}
        idx = 0
        for group in self.param_groups:
            beta1 = group["betas"][0]
            g_gamma = group["gamma"]
            for p in group["params"]:
                self._layer_beta1_by_id[id(p)] = beta1 * ((1.0 - g_gamma) ** idx)
                idx += 1

        # Grokking signal (one scalar per step — the reference computes a single
        # train/val signal shared across groups) and whether it has ever been
        # set. α is derived PER GROUP from it (alpha_init & kappa are group-level
        # in the reference), so we keep the signal, not a pre-baked α.
        self._grok_signal: float = 0.0
        self._signal_active: bool = False

        self._use_grad_hooks = use_grad_hooks
        if use_grad_hooks:
            _register_grad_hooks(self)

    def _layer_beta1(self, p) -> float:
        """Per-tensor β1 (= β1 * (1-gamma)**layer_index). Falls back to the
        group β1 for a param added after construction (defensive: a freshly
        ``add_param_group``'d tensor not yet indexed gets gamma^0 = full β1)."""
        return self._layer_beta1_by_id.get(
            id(p), self.param_groups[0]["betas"][0])

    @staticmethod
    def _grokking_signal(train_loss, val_loss) -> float:
        """Reference _default_grokking_signal (max-normalised clipped gap):
            diff = max(0, val - train); m = max(val, train); signal = diff/m.
        Returns 0.0 if either loss is None or m<=0."""
        if train_loss is None or val_loss is None:
            return 0.0
        diff = max(0.0, float(val_loss) - float(train_loss))
        max_loss = max(float(val_loss), float(train_loss))
        return diff / max_loss if max_loss > 0.0 else 0.0

    def _alpha_for_group(self, group) -> float:
        """Live α_t for a group (PUBLISHED GrokAdamW, per group):
            α = alpha_init * exp(-kappa * signal).
        Until a (train_loss, val_loss) pair has ever been supplied, returns the
        group's alpha_init unchanged. alpha_init == group["alpha"] and
        kappa == group["kappa"] (grokking_signal_decay_rate)."""
        if not self._signal_active:
            return group["alpha"]
        return group["alpha"] * math.exp(-group["kappa"] * self._grok_signal)

    def add_param_group(self, param_group) -> None:
        self._static_cache = {}
        super().add_param_group(param_group)
        # Layer indices shift when a group is appended, so recompute the
        # per-param-id β1 across ALL groups (global enumeration order).
        self._layer_beta1_by_id = {}
        idx = 0
        for group in self.param_groups:
            beta1 = group["betas"][0]
            g_gamma = group["gamma"]
            for p in group["params"]:
                self._layer_beta1_by_id[id(p)] = beta1 * ((1.0 - g_gamma) ** idx)
                idx += 1

    def _group_cache(self, group, grads_by_id):
        """Return cached (params, exp_avg, exp_avg_sq, ema, states).

        Keyed on the grad-bearing param ids. ``grads_by_id`` provides the
        first-gradient EMA seed on first init."""
        key = tuple(id(p) for p in group["params"] if p.grad is not None)
        cached = self._static_cache.get(id(group))
        if cached is not None and cached[0] == key:
            return cached[1]

        params_list = []
        exp_avg_list = []
        exp_avg_sq_list = []
        ema_list = []
        states = []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                # Seed the gradient EMA with the first gradient (not zeros) —
                # see Grokfast; a zero seed under-amplifies the early grokking
                # phase and the kernel applies no EMA bias correction.
                state["ema"] = grads_by_id[id(p)].detach().to(
                    torch.float32).clone()
            params_list.append(p)
            exp_avg_list.append(state["exp_avg"])
            exp_avg_sq_list.append(state["exp_avg_sq"])
            ema_list.append(state["ema"])
            states.append(state)
        entry = (params_list, exp_avg_list, exp_avg_sq_list, ema_list, states)
        self._static_cache[id(group)] = (key, entry)
        return entry

    @torch.no_grad()
    def step(self, closure=None, train_loss=None, val_loss=None
             ) -> Optional[float]:
        """Perform a single optimisation step.

        Args:
            closure: A closure that re-evaluates the model and returns the loss
                (optional).
            train_loss: Current training loss (scalar / float). When BOTH this
                and ``val_loss`` are given, the live grokking signal is
                recomputed and α_t = alpha_init * exp(-kappa * signal) drives the
                slow-grad EMA decay this step (PUBLISHED GrokAdamW). Pass them on
                the cadence the caller wants α to adapt; on steps where either is
                None, α stays at its previous value (alpha_init on the first).
            val_loss: Current validation/eval loss (scalar / float).

        Returns:
            The loss value if *closure* is provided, otherwise ``None``.
        """
        # PURE L3-TC: eager .step() execution is REMOVED. The L3-TC production path
        # (grokking_race_v2.train_grokadamw) sets opt._grok_signal / _signal_active
        # HOST-SIDE before the fused launch; _alpha_for_group + _layer_beta1 stay as
        # host-side helpers _opt_scalars_from reads. The eager kernel dispatch is gone.
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
