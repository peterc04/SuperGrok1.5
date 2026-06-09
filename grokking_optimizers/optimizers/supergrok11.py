"""
SuperGrok v1.1 — C++/CUDA Accelerated Grokking Optimizer

Uses cosine similarity gating (per-parameter) instead of sigmoid gating.
Otherwise similar to SuperGrok v1.5 (meta-net, SAM, bilevel).
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
    r"""SuperGrok v1.1 — C++/CUDA with cosine similarity gating."""

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
        rescale_clamp: Optional[float] = None,
        meta_gate_power: Optional[float] = None,
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
        # Bound the learned meta-net output scale. The meta-loss
        # (−Σ smart_grad·vg_unit) is linear in `rescale` with no magnitude
        # penalty, so Adam grows it without bound until the correction
        # rescale·MLP(grad,sharpness) buries the real gradient and training
        # collapses (~step 900). Clamping rescale after each meta_step keeps the
        # correction a bounded *steer* on the update instead of a runaway term.
        # None = unbounded (legacy). Set via `supergrok_rescale_clamp`.
        self.rescale_clamp = rescale_clamp
        # Memorization-gated meta correction. The collapse is NOT a magnitude
        # runaway — it's that the val-aligned meta push is *applied at all* after
        # the model memorizes, walking it off the memorized minimum (verified:
        # with the meta weight α≈0 training is stable; re-enabling α once
        # memorized collapses it). The natural signal is train PROGRESS, not
        # ‖grad‖ (which is pinned at gradient_clipping). The legacy reactive
        # alpha schedule oscillates (a train dip snaps α back up → re-collapse),
        # so we gate by a MONOTONIC ratchet: scale α by (1 − max_train_acc)^p.
        # Once the model memorizes the meta contribution ratchets to ~0 and stays
        # there → SG reduces to its (grokking) AdamW core through the plateau.
        # None = off (legacy). Set via `supergrok_meta_gate_power`.
        self.meta_gate_power = meta_gate_power
        self._max_train_acc = 0.0

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
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._use_grad_hooks:
            self._global_step += 1
            return loss

        self._ensure_state()
        self._global_step += 1

        if train_acc is not None:
            self._cached_train_acc = train_acc

        if self._global_step % self.alpha_update_freq == 0 or self._global_step == 1:
            self._update_alpha(train_loss, val_loss, train_acc)

        base_alpha = self._cached_alpha
        ramp = self._get_ramp_factor()
        layer_alphas = [max(0.0, min(1.0, base_alpha * f)) for f in self._flat_layer_alphas]

        group = self.param_groups[0]
        lr = group["lr"]
        beta2 = group["betas"][1]
        eps = group["eps"]
        wd = group["weight_decay"]

        grads = []
        for p in self._flat_params:
            grads.append(p.grad.data if p.grad is not None else torch.Tensor())

        # Own the step counter in Python. The fused binding receives `steps` BY
        # VALUE (pybind copies the list[int]), so its internal `steps[i] += 1`
        # never persists back to this list. Without owning it here, _flat_steps
        # stays frozen at 0 -> bc1/bc2 are pinned at t=1 -> the Adam denominator
        # stays ~50x inflated forever and the model can never memorize. Mirrors
        # adamw.py / grokadamw.py, which advance the counter in Python.
        for i, p in enumerate(self._flat_params):
            if p.grad is not None and p.grad.numel() > 0:
                self._flat_steps[i] += 1

        # Memorization-gated meta correction (monotonic ratchet). Scale α by
        # (1 − max_train_acc)^p so the val-aligned meta push vanishes as the model
        # memorizes and — unlike the reactive alpha schedule — never re-enables on
        # a transient train dip. See __init__ for the why.
        if self.meta_gate_power is not None:
            self._max_train_acc = max(self._max_train_acc, self._cached_train_acc)
            gate = max(0.0, 1.0 - self._max_train_acc) ** self.meta_gate_power
            layer_alphas = [a * gate for a in layer_alphas]

        if self._weights_dirty:
            self._cached_weights = self.meta_net.get_weights()
            self._weights_dirty = False
        W1, b1, W2, b2, rescale = self._cached_weights

        ops_impl = _ops if self._flat_params[0].is_cuda else _ops_cpu
        ops_impl.supergrok11_fused_step(
            self._flat_param_data,
            grads,
            self._flat_exp_avgs,
            self._flat_exp_avg_sqs,
            self._flat_mus,
            self._flat_sharpness,
            self._flat_steps,
            layer_alphas,
            self._flat_layer_beta1s,
            W1, b1, W2, b2, rescale, self.meta_hidden_dim,
            beta2, lr, wd, eps,
            self.lamb, ramp, self.gate_temperature,
            self.gradient_clipping,
        )

        return loss

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

    def meta_step(self, model, val_x, val_y, criterion, meta_optimizer):
        """Bilevel meta-net training. Batched meta-net forward."""
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

        model.zero_grad()
        with torch.enable_grad():
            val_loss = criterion(model(val_x), val_y)
            val_loss.backward()

        meta_optimizer.zero_grad()
        device = val_x.device
        meta_loss = torch.tensor(0.0, device=device)
        for name, p in named_params:
            if name in smart_grads and p.grad is not None:
                vg = p.grad.detach()
                vg_norm = vg.norm()
                vg_unit = vg / vg_norm if vg_norm > 1e-12 else vg
                meta_loss = meta_loss - (smart_grads[name] * vg_unit).sum()

        meta_loss.backward()
        meta_optimizer.step()
        if self.rescale_clamp is not None:
            self.meta_net.rescale.data.clamp_(-self.rescale_clamp, self.rescale_clamp)
        self._weights_dirty = True

        for name, p in named_params:
            p.grad = saved_grads.get(name)

        return val_loss.item()

    def _single_param_step(self, param, group, state):
        """Per-parameter step used by the `use_grad_hooks=True` path."""
        if param.grad is None:
            return
        self._ensure_state()
        pidx = self._param_to_idx.get(id(param))
        if pidx is None:
            return
        self._flat_steps[pidx] += 1
        base_alpha = self._cached_alpha
        ramp = self._get_ramp_factor()
        layer_alpha = max(0.0, min(1.0, base_alpha * self._flat_layer_alphas[pidx]))

        if self._weights_dirty:
            self._cached_weights = self.meta_net.get_weights()
            self._weights_dirty = False
        W1, b1, W2, b2, rescale = self._cached_weights

        ops_impl = _ops if param.is_cuda else _ops_cpu
        ops_impl.supergrok11_fused_step(
            [param.data],
            [param.grad.data],
            [self._flat_exp_avgs[pidx]],
            [self._flat_exp_avg_sqs[pidx]],
            [self._flat_mus[pidx]],
            [self._flat_sharpness[pidx]],
            [self._flat_steps[pidx]],
            [layer_alpha],
            [self._flat_layer_beta1s[pidx]],
            W1, b1, W2, b2, rescale, self.meta_hidden_dim,
            group["betas"][1], group["lr"], group["weight_decay"], group["eps"],
            self.lamb, ramp, self.gate_temperature,
            self.gradient_clipping,
        )

    def get_global_step(self):
        return self._global_step


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
