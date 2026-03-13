"""
SuperGrok v1.1 — C++/CUDA Accelerated Grokking Optimizer

Uses cosine similarity gating (per-parameter) instead of sigmoid gating.
Otherwise similar to SuperGrok v1.5 (meta-net, SAM, bilevel).
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Callable, Dict, Tuple

from grokking_optimizers import _ops
from .supergrok15 import SharpnessMetaNet


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
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.alpha_init = alpha_init
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

        if meta_net is None:
            self.meta_net = SharpnessMetaNet(meta_hidden_dim)
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

    def _ensure_state(self):
        if self._state_initialized:
            return
        for p in self._flat_params:
            self._flat_exp_avgs.append(torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
            self._flat_exp_avg_sqs.append(torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
            self._flat_mus.append(torch.zeros_like(p.data))
            self._flat_sharpness.append(torch.zeros_like(p.data))
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

    @torch.no_grad()
    def step(self, closure=None, train_loss=None, val_loss=None, train_acc=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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

        W1, b1, W2, b2, rescale = self.meta_net.get_weights()

        _ops.supergrok11_fused_step(
            [p.data for p in self._flat_params],
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

    def meta_step(self, model, val_x, val_y, criterion, meta_optimizer):
        """Bilevel meta-net training."""
        self._ensure_state()
        named_params = list(model.named_parameters())

        saved_grads = {}
        for name, p in named_params:
            if p.grad is not None:
                saved_grads[name] = p.grad.detach().clone()

        smart_grads = {}
        for name, p in named_params:
            if name in saved_grads:
                pidx = self._param_to_idx.get(id(p))
                sharp = self._flat_sharpness[pidx] if pidx is not None else torch.zeros_like(p.data)
                smart_grads[name] = self.meta_net(saved_grads[name], sharp)

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

        for name, p in named_params:
            p.grad = saved_grads.get(name)

        return val_loss.item()

    def get_global_step(self):
        return self._global_step
