"""
SuperGrok v1.1 — C++/CUDA Accelerated Grokking Optimizer

Gate = sigmoid(gate_temperature * cos_sim(grad, momentum)) — a SIGMOID of the
per-parameter cosine similarity between grad and momentum. What distinguishes
v1.1 from v1.5 is the gate SIGNAL (the per-parameter cosine here vs. v1.5's
training-accuracy scalar), not the squashing function (both end in a sigmoid).
The gate (and the cosine reductions it needs) is computed in the C++/CUDA
kernel — see compute_cosine_gate_fused / sg11_finalize_gate
(csrc/algorithms/supergrok11.h). Otherwise similar to SuperGrok v1.5 (meta-net,
SAM, bilevel).
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint as _grad_checkpoint
from typing import Optional, Callable, Dict, Tuple

from grokking_optimizers.dispatch import get_ops

_ops = get_ops()  # Fails loudly if C++ extension not built
_ops_cpu = _ops  # CPU ops are part of the same extension


# Activation/gradient checkpointing helper (mirrors grokking_race_v2.py).
# Only checkpoints when training AND grad is being tracked — under
# torch.no_grad()/eval there is nothing to recompute, so we call directly,
# avoiding checkpoint's overhead and its "no input requires grad" warning.
# This keeps the eval/inference and CUDA fused-step paths completely unaffected.
def _maybe_checkpoint(module, x, enabled):
    """Run ``module(x)`` (single-tensor input), optionally checkpointed."""
    if enabled and module.training and torch.is_grad_enabled():
        return _grad_checkpoint(module, x, use_reentrant=False)
    return module(x)


class SharpnessMetaNet(nn.Module):
    """Element-wise gradient transformation conditioned on sharpness.

    Architecture: output = grad + rescale * MLP(grad, sharpness)
    MLP: Linear(2, H) -> GELU -> Linear(H, 1)

    Duplicated verbatim from supergrok15.py so each optimizer file is fully
    self-contained. If the metanet architecture changes, update both copies.
    """

    def __init__(self, hidden_dim: int = 32, grad_checkpoint: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grad_checkpoint = grad_checkpoint
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.rescale = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.net[0].weight.normal_(0, 0.01)
            self.net[0].bias.zero_()
            self.net[2].weight.normal_(0, 0.01)
            self.net[2].bias.zero_()

    def forward(self, grad: torch.Tensor, sharpness: torch.Tensor) -> torch.Tensor:
        if grad.numel() == 0:
            return grad
        shape = grad.shape
        flat_g = grad.reshape(-1, 1)
        flat_s = sharpness.reshape(-1, 1)
        inp = torch.cat([flat_g, flat_s], dim=1)
        # The MLP (Linear -> GELU -> Linear) is the only non-trivial
        # self-contained sub-computation here, so it is checkpointed as a
        # single block; the surrounding cat/rescale/reshape ops are negligible.
        correction = self.rescale * _maybe_checkpoint(
            self.net, inp, self.grad_checkpoint)
        return (flat_g + correction).reshape(shape)

    def get_weights(self):
        W1 = self.net[0].weight.data.contiguous()
        b1 = self.net[0].bias.data.contiguous()
        W2 = self.net[2].weight.data.contiguous()
        b2 = self.net[2].bias.data.contiguous()
        rescale = self.rescale.data.item()
        return W1, b1, W2, b2, rescale


class SuperGrok11(Optimizer):
    r"""SuperGrok v1.1 — C++/CUDA with sigmoid-of-cosine-similarity gating.

    gate = sigmoid(gate_temperature * cos_sim(grad, momentum)); the per-parameter
    cosine is the gate SIGNAL (vs. SG15's accuracy scalar). Computed in-kernel."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1.0,
        alpha_init: float = 0.98,
        lamb: float = 5.0,
        gamma: float = 0.1,
        gamma_alpha: float = 0.0,
        kappa: float = 0.1,
        warmup_steps: int = 100,
        warmup_ramp: int = 100,
        gradient_clipping: float = 1.0,
        meta_net: Optional[nn.Module] = None,
        meta_hidden_dim: int = 32,
        gate_temperature: float = 5.0,
        alpha_update_freq: int = 50,
        meta_update_freq: int = 5,
        zero_loss_threshold: float = 1e-4,
        zero_acc_threshold: float = 0.995,
        sam_rho: float = 0.05,
        sam_enable_threshold: float = 0.0,
        use_grad_hooks: bool = False,
        grad_checkpoint: bool = True,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.alpha_init = alpha_init
        self.sam_enable_threshold = sam_enable_threshold
        self.lamb = lamb
        self.gamma = gamma
        self.gamma_alpha = gamma_alpha
        self.kappa = kappa
        self.warmup_steps = warmup_steps
        self.warmup_ramp = max(1, warmup_ramp)
        self.gradient_clipping = gradient_clipping
        self.meta_hidden_dim = meta_hidden_dim
        self.gate_temperature = gate_temperature
        self.alpha_update_freq = alpha_update_freq
        self.meta_update_freq = meta_update_freq
        self.zero_loss_threshold = zero_loss_threshold
        self.zero_acc_threshold = zero_acc_threshold
        self.sam_rho = sam_rho
        # NOTE (owner directive — no functionality suppression): two earlier
        # collapse mitigations were REMOVED from this class rather than left as
        # dormant knobs: a `rescale_clamp` on the meta-net output scale and a
        # `meta_gate_power` memorization ratchet that multiplied the meta
        # contribution toward 0 post-memorization. Both "fixed" the collapse by
        # suppressing the meta. The functionality-preserving replacement is the
        # two-term lookahead meta objective in `meta_step` (val_CE + train_CE at
        # a virtual step), which keeps the meta fully active for the whole run.

        if meta_net is None:
            self.meta_net = SharpnessMetaNet(meta_hidden_dim,
                                             grad_checkpoint=grad_checkpoint)
        else:
            self.meta_net = meta_net

        try:
            first_param = next(iter(self.param_groups[0]["params"]))
            self.meta_net = self.meta_net.to(first_param.device)
        except (StopIteration, IndexError):
            pass

        self._global_step = 0
        self._cached_alpha = alpha_init
        self._cached_train_acc = 0.0

        self._flat_params = []
        self._flat_steps = []
        self._flat_layer_alphas = []
        self._flat_layer_beta1s = []
        self._flat_exp_avgs = []
        self._flat_exp_avg_sqs = []
        self._flat_mus = []
        self._flat_sharpness = []
        self._param_to_idx = {}

        idx = 0
        num_params = sum(1 for g in self.param_groups for _ in g["params"])
        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                self._flat_params.append(p)
                self._flat_steps.append(0)
                lb1 = beta1 * ((1.0 - gamma) ** idx)
                self._flat_layer_beta1s.append(lb1)
                if gamma_alpha == 0.0:
                    la_factor = 1.0
                else:
                    max_idx = max(num_params - 1, 1)
                    la_factor = (1.0 - gamma_alpha) ** (max_idx - idx)
                self._flat_layer_alphas.append(la_factor)
                self._param_to_idx[id(p)] = idx
                idx += 1

        self._num_params = num_params
        self._state_initialized = False
        self._flat_param_data = [p.data for p in self._flat_params]
        self._weights_dirty = True
        self._cached_weights = None

        self._use_grad_hooks = use_grad_hooks
        if use_grad_hooks:
            _register_grad_hooks(self)

    def _ensure_state(self):
        if self._state_initialized:
            return
        for p in self._flat_params:
            self._flat_exp_avgs.append(torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
            self._flat_exp_avg_sqs.append(torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
            self._flat_mus.append(torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
            self._flat_sharpness.append(torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
        self._state_initialized = True

    def _update_alpha(self, train_loss, val_loss, train_acc):
        if train_loss is None and train_acc is None:
            return
        signal = 0.0
        if (train_acc is not None and train_acc >= self.zero_acc_threshold) or \
           (train_loss is not None and train_loss < self.zero_loss_threshold):
            signal = 10.0
        elif val_loss is not None and train_loss is not None and train_loss > 1e-12:
            signal = max(0.0, (val_loss - train_loss) / train_loss)
        self._cached_alpha = self.alpha_init * math.exp(-self.kappa * signal)

    def _get_ramp_factor(self):
        if self._global_step <= self.warmup_steps:
            return 0.0
        return min(1.0, (self._global_step - self.warmup_steps) / self.warmup_ramp)

    def _get_effective_sam_freq(self):
        """Check sam_enable_threshold before allowing SAM steps."""
        if self._cached_train_acc < self.sam_enable_threshold:
            return 999999  # effectively disabled
        return 1  # caller controls actual frequency

    @torch.no_grad()
    def step(self, closure=None, train_loss=None, val_loss=None, train_acc=None):
        """Eager optimiser step — REMOVED (pure L3-TC).

        On the L3-TC path the SAM 2nd backward + per-tensor meta-net mu/gate
        precompute + the SG11 apply run IN the megakernel; the meta-net is trained
        host-side via ``meta_step``/``sam_step`` (pure-PyTorch autograd, kept) and
        fused_train_step re-extracts its phi pack. The eager kernel dispatch here is
        gone (``_update_alpha`` / ``_get_gate_signal`` stay as host-side helpers)."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    def sam_step(self, model, train_x, train_y, criterion):
        """SAM ascent step → per-parameter sharpness |g(w+e) − g(w)|.

        Recomputes the gradient at the SAM-perturbed point e(w)=rho·g/‖g‖ via
        ``torch.func.functional_call`` and stores the elementwise sharpness into
        ``_flat_sharpness`` — the buffer that both ``meta_step`` and the fused
        kernel read as the meta-net's 2nd MLP input. Two correctness points the
        previous version got wrong:
          • the perturbed params are built from ``.detach()`` tensors, so they
            must be ``.requires_grad_(True)`` or autograd has nothing to
            differentiate (the old ``sam_loss.backward()`` raised
            "element 0 … does not require grad" on every call → sharpness was
            never written and stayed at its zero init);
          • ``functional_call`` substitutes the perturbed tensors as the
            module's params, so gradients live on THEM, not on
            ``model.parameters().grad``. We take them with ``autograd.grad`` and
            leave each ``p.grad`` (the original backward result that ``step()``
            consumes) completely untouched.
        """
        self._ensure_state()
        train_grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                train_grads[name] = p.grad.detach().clone()
        if not train_grads:
            return 0.0

        flat_grads = [train_grads[n] for n, _ in model.named_parameters() if n in train_grads]
        total_norm_sq = sum(g.norm().pow(2) for g in flat_grads if g.numel() > 0)
        grad_norm = total_norm_sq.sqrt() + 1e-12
        rho_over_norm = self.sam_rho / grad_norm

        named_params = dict(model.named_parameters())
        pert_names = [n for n in named_params if n in train_grads]
        perturbed = {}
        for name, p in named_params.items():
            if name in train_grads:
                perturbed[name] = (p.detach() + rho_over_norm * train_grads[name]).requires_grad_(True)
            else:
                perturbed[name] = p.detach()

        with torch.enable_grad():
            sam_logits = torch.func.functional_call(model, perturbed, (train_x,))
            sam_loss = criterion(sam_logits, train_y)
            sam_grads = torch.autograd.grad(
                sam_loss, [perturbed[n] for n in pert_names], allow_unused=True)
        sam_loss_val = sam_loss.item()

        for name, sg in zip(pert_names, sam_grads):
            if sg is None:
                continue
            pidx = self._param_to_idx.get(id(named_params[name]))
            if pidx is not None:
                # flatten to match the (numel,) zero-init buffer the kernel reshapes
                self._flat_sharpness[pidx] = (sg.detach() - train_grads[name]).abs().reshape(-1)

        return sam_loss_val

    def meta_step(self, model, val_x, val_y, criterion, meta_optimizer,
                  train_x=None, train_y=None):
        """Bilevel meta-net training — two-term lookahead descent objective.

        Trains the meta-net so the smart gradient it emits improves validation
        WITHOUT leaving the train manifold, one virtual step ahead:

            w⁺ = w·(1−lr·wd) − lr·meta_net(g, sharpness)
            meta_loss = val_CE(w⁺) + train_CE(w⁺)

        differentiated through the virtual step into the meta-net weights via
        ``torch.func.functional_call`` (real params untouched — virtual params
        are built from ``p.detach()``; ``p.grad`` is snapshotted/restored).

        History of this objective (both predecessors collapsed training):
        • unbounded alignment ``−Σ(smart_grad·vg_unit)`` rewarded sheer
          magnitude along the val-gradient → post-memorization the unopposed
          push walked the weights off the minimum (collapse @~900).
        • val-only lookahead ``val_CE(w⁺)`` self-regulates against single-step
          damage, but at a memorized-not-grokked point the val-optimal push is
          LARGE, and the kernel re-applies the cached meta output every step
          between meta updates (freq=5) → overshoot past the lookahead horizon
          (SG11 collapse @~1200; SG15 oscillated to test 0.92 then collapsed).
        The train term anchors the manifold: off-manifold pushes raise
        train_CE sharply at a memorized point, so the optimum is movement
        ALONG the train level-set toward better val — the grokking direction —
        and after any slip the train term dominates and pulls back
        (self-healing). The meta stays FULLY ACTIVE the entire run; restraint
        is learned, not gated (no functionality suppression). Kernel path
        unchanged: it consumes refreshed meta weights via ``_weights_dirty``.

        Batched meta-net forward — one call instead of per-parameter."""
        self._ensure_state()
        named_params = list(model.named_parameters())

        saved_grads = {}
        for name, p in named_params:
            if p.grad is not None:
                saved_grads[name] = p.grad.detach().clone()

        # Batched meta-net forward — one call instead of per-parameter
        all_grads = []
        all_sharps = []
        all_names = []
        all_sizes = []
        for name, p in named_params:
            if name in saved_grads:
                pidx = self._param_to_idx.get(id(p))
                all_grads.append(saved_grads[name].reshape(-1))
                all_sharps.append(
                    self._flat_sharpness[pidx].reshape(-1) if pidx is not None
                    else torch.zeros(p.data.numel(), device=p.device))
                all_names.append(name)
                all_sizes.append(saved_grads[name].numel())

        smart_grads = {}
        if all_grads:
            cat_grads = torch.cat(all_grads)
            cat_sharps = torch.cat(all_sharps)
            cat_smart = self.meta_net(cat_grads, cat_sharps)
            offset = 0
            for name, size in zip(all_names, all_sizes):
                smart_grads[name] = cat_smart[offset:offset+size].reshape(saved_grads[name].shape)
                offset += size

        # Virtual step at the optimizer's own lr/wd (decoupled-WD shrink + the
        # smart gradient), then evaluate validation CE there. Gradients flow
        # back through smart_grads into the meta-net only.
        group = self.param_groups[0]
        lr_eff = float(group["lr"])
        wd_eff = float(group.get("weight_decay", 0.0))
        virtual = {}
        for name, p in named_params:
            base = p.detach()
            if name in smart_grads:
                virtual[name] = base * (1.0 - lr_eff * wd_eff) - lr_eff * smart_grads[name]
            else:
                virtual[name] = base

        meta_optimizer.zero_grad()
        with torch.enable_grad():
            val_loss = criterion(
                torch.func.functional_call(model, virtual, (val_x,)), val_y)
            meta_loss = val_loss
            if train_x is not None and train_y is not None:
                meta_loss = meta_loss + criterion(
                    torch.func.functional_call(model, virtual, (train_x,)), train_y)
        meta_loss.backward()
        meta_optimizer.step()
        self._weights_dirty = True

        for name, p in named_params:
            p.grad = saved_grads.get(name)

        return val_loss.item()

    def _single_param_step(self, param, group, state):
        """Per-parameter eager step (use_grad_hooks path) — REMOVED (pure L3-TC)."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    def get_global_step(self):
        return self._global_step

    # ── Checkpoint fidelity ──────────────────────────────────────────────────
    # The base Optimizer.state_dict() serializes only param_groups + self.state
    # — but ALL of SuperGrok1.1's evolving state lives outside self.state (the
    # flat Adam moments, the meta-net mu inputs, the per-param step counters,
    # the trained meta-net itself). Without these overrides a checkpoint/resume
    # silently reset momentum AND the trained meta-net to init — a suppression-
    # class bug in API form. The derived per-param constants (_flat_layer_*) are
    # NOT serialized: they are pure functions of the config, rebuilt in __init__.
    _EXTRA_KEY = "supergrok11_extra"

    def state_dict(self):
        sd = super().state_dict()
        self._ensure_state()
        sd[self._EXTRA_KEY] = {
            "global_step": self._global_step,
            "cached_alpha": self._cached_alpha,
            "cached_train_acc": self._cached_train_acc,
            "flat_steps": list(self._flat_steps),
            "flat_exp_avgs": [t.detach().clone() for t in self._flat_exp_avgs],
            "flat_exp_avg_sqs": [t.detach().clone() for t in self._flat_exp_avg_sqs],
            "flat_mus": [t.detach().clone() for t in self._flat_mus],
            "flat_sharpness": [t.detach().clone() for t in self._flat_sharpness],
            "meta_net": self.meta_net.state_dict(),
        }
        # The step_full path owns an internal Adam over the meta-net; its
        # momentum is part of the training state (dropping it makes a resumed
        # run diverge by one meta-step's worth, ~3e-8 after 2 steps — measured).
        if hasattr(self, "_auto_meta_opt"):
            sd[self._EXTRA_KEY]["auto_meta_opt"] = self._auto_meta_opt.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        extra = state_dict.get(self._EXTRA_KEY)
        base = {k: v for k, v in state_dict.items() if k != self._EXTRA_KEY}
        super().load_state_dict(base)
        if extra is None:
            return  # plain/base checkpoint: behave like the base class
        self._ensure_state()
        n = len(self._flat_params)
        for key in ("flat_exp_avgs", "flat_exp_avg_sqs", "flat_mus",
                    "flat_sharpness"):
            saved = extra[key]
            live = getattr(self, "_" + key)
            if len(saved) != n:
                raise ValueError(
                    f"SuperGrok11.load_state_dict: checkpoint has {len(saved)} "
                    f"{key} buffers but the optimizer holds {n} params — "
                    f"the model architecture changed.")
            for dst, src in zip(live, saved):
                if dst.numel() != src.numel():
                    raise ValueError(
                        f"SuperGrok11.load_state_dict: {key} numel mismatch "
                        f"({dst.numel()} vs checkpoint {src.numel()}).")
                dst.copy_(src.to(dst.device, dtype=dst.dtype))
        self._flat_steps = list(extra["flat_steps"])
        self._global_step = int(extra["global_step"])
        self._cached_alpha = float(extra["cached_alpha"])
        self._cached_train_acc = float(extra["cached_train_acc"])
        self.meta_net.load_state_dict(extra["meta_net"])
        if "auto_meta_opt" in extra:
            if not hasattr(self, "_auto_meta_opt"):
                self._auto_meta_opt = torch.optim.Adam(
                    self.meta_net.parameters(), lr=1e-4)
            self._auto_meta_opt.load_state_dict(extra["auto_meta_opt"])
        self._weights_dirty = True   # re-extract kernel weights from the loaded net

    def step_full(self, model, train_x, train_y, val_x, val_y, criterion=None):
        """Complete training step: forward + backward + SAM + meta-learning + optimizer.

        The AdamW-grade self-contained path (mirrors SuperGrok15/SuperGrok2
        ``step_full``): owns the forward/backward, runs ``sam_step`` and
        ``meta_step`` on their cadences with an internally-managed meta
        optimizer, and feeds ``step()`` the train/val statistics it needs — so
        the FULL machinery is exercised without any external choreography.
        External drivers (the race loop) may still call sam_step/meta_step/
        step() directly; this wrapper makes that advanced wiring optional,
        not required.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if not hasattr(self, '_auto_meta_opt'):
            self._auto_meta_opt = torch.optim.Adam(self.meta_net.parameters(), lr=1e-4)

        model.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()

        metrics: Dict[str, float] = {}
        step_num = self._global_step + 1

        sam_freq_eff = self._get_effective_sam_freq()
        if step_num % sam_freq_eff == 0:
            try:
                metrics["sam_loss"] = self.sam_step(model, train_x, train_y, criterion)
            except Exception as e:
                import warnings
                warnings.warn(f"SuperGrok1.1 sam_step failed at step {step_num}: {e}")

        if step_num % self.meta_update_freq == 0:
            try:
                # Two-term lookahead objective: pass the train batch so the
                # meta-net descends val_CE + train_CE at the virtual step
                # (train_x/train_y omitted would silently drop the train
                # anchor that stabilizes the meta — keep them explicit).
                metrics["val_loss"] = self.meta_step(
                    model, val_x, val_y, criterion, self._auto_meta_opt,
                    train_x=train_x, train_y=train_y)
            except Exception as e:
                import warnings
                warnings.warn(f"SuperGrok1.1 meta_step failed at step {step_num}: {e}")

        kw: Dict[str, float] = {}
        alpha_freq = self.alpha_update_freq
        if (step_num % alpha_freq == 0) or step_num == 1:
            with torch.no_grad():
                train_loss_val = loss.item()
                train_acc = (logits.detach().argmax(-1) == train_y).float().mean().item()
            kw["train_loss"] = train_loss_val
            kw["train_acc"] = train_acc
            metrics["train_loss"] = train_loss_val
            metrics["train_acc"] = train_acc
            if step_num % alpha_freq == 0:
                with torch.no_grad():
                    val_loss_val = criterion(model(val_x), val_y).item()
                kw["val_loss"] = val_loss_val
                if "val_loss" not in metrics:
                    metrics["val_loss"] = val_loss_val

        self.step(**kw)
        return metrics


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
