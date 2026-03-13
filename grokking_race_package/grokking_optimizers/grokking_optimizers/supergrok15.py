"""
SuperGrok v1.5 — C++/CUDA Accelerated Grokking Optimizer

Dispatches to fused C++/CUDA kernels. No Python fallback.

Key features:
  - Fused CUDA kernel for meta-net inference (per-element MLP)
  - Fused CUDA kernel for Adam + progressive weight decay
  - Sigmoid gating (metric-driven, based on training accuracy)
  - Adaptive SAM frequency (sigmoid-driven schedule)
  - Decoupled SAM and bilevel optimization (independent schedules)
  - Progressive weight decay (sigmoid-driven)
  - BF16/FP16 support
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Callable, Dict, Any, Tuple

from grokking_optimizers import _ops


class SharpnessMetaNet(nn.Module):
    """Element-wise gradient transformation conditioned on sharpness.

    Architecture: output = grad + rescale * MLP(grad, sharpness)
    MLP: Linear(2, H) -> GELU -> Linear(H, 1)
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
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
        correction = self.rescale * self.net(inp)
        return (flat_g + correction).reshape(shape)

    def get_weights(self):
        """Extract contiguous weight tensors for the C++ kernel."""
        W1 = self.net[0].weight.data.contiguous()
        b1 = self.net[0].bias.data.contiguous()
        W2 = self.net[2].weight.data.contiguous()
        b2 = self.net[2].bias.data.contiguous()
        rescale = self.rescale.data.item()
        return W1, b1, W2, b2, rescale


class SuperGrok15(Optimizer):
    r"""SuperGrok v1.5 — C++/CUDA Accelerated Grokking Optimizer.

    All computations dispatched to fused CUDA kernels.
    Supports FP32, FP16, and BF16 parameters.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1.0,
        alpha_init: float = 0.98,
        lamb: float = 2.0,
        gamma: float = 0.1,
        gamma_alpha: float = 0.0,
        kappa: float = 0.1,
        warmup_steps: int = 100,
        warmup_ramp: int = 100,
        gradient_clipping: float = 1.0,
        meta_net: Optional[nn.Module] = None,
        meta_hidden_dim: int = 32,
        alpha_update_freq: int = 100,
        zero_loss_threshold: float = 1e-4,
        zero_acc_threshold: float = 0.995,
        sam_rho: float = 0.05,
        gate_scale: float = 20.0,
        gate_thresh: float = 0.8,
        sam_freq_min: int = 3,
        sam_freq_max: int = 20,
        sam_scale: float = 20.0,
        sam_thresh: float = 0.85,
        bilevel_freq_min: int = 5,
        bilevel_freq_max: int = 30,
        bilevel_scale: float = 20.0,
        bilevel_thresh: float = 0.9,
        wd_ramp: float = 4.0,
        wd_scale: float = 20.0,
        wd_thresh: float = 0.9,
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
        self.alpha_update_freq = alpha_update_freq
        self.zero_loss_threshold = zero_loss_threshold
        self.zero_acc_threshold = zero_acc_threshold
        self.sam_rho = sam_rho
        self.meta_hidden_dim = meta_hidden_dim

        self.gate_scale = gate_scale
        self.gate_thresh = gate_thresh
        self.sam_freq_min = sam_freq_min
        self.sam_freq_max = sam_freq_max
        self.sam_scale = sam_scale
        self.sam_thresh = sam_thresh
        self.bilevel_freq_min = bilevel_freq_min
        self.bilevel_freq_max = bilevel_freq_max
        self.bilevel_scale = bilevel_scale
        self.bilevel_thresh = bilevel_thresh
        self.wd_ramp = wd_ramp
        self.wd_scale = wd_scale
        self.wd_thresh = wd_thresh

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

        # Build flat parameter lists for C++
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

    def _ensure_state(self):
        if self._state_initialized:
            return
        for p in self._flat_params:
            self._flat_exp_avgs.append(torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
            self._flat_exp_avg_sqs.append(torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
            self._flat_mus.append(torch.zeros_like(p.data))
            self._flat_sharpness.append(torch.zeros_like(p.data))
        self._state_initialized = True

    def _sigmoid(self, scale, value, thresh):
        return 1.0 / (1.0 + math.exp(-scale * (value - thresh)))

    def _update_alpha(self, train_loss, val_loss, train_acc):
        if train_loss is None and train_acc is None:
            return
        signal = 0.0
        memorized = False
        if train_acc is not None and train_acc >= self.zero_acc_threshold:
            memorized = True
        elif train_loss is not None and train_loss < self.zero_loss_threshold:
            memorized = True
        if memorized:
            signal = 10.0
        elif val_loss is not None and train_loss is not None and train_loss > 1e-12:
            signal = max(0.0, (val_loss - train_loss) / train_loss)
        self._cached_alpha = self.alpha_init * math.exp(-self.kappa * signal)

    def _get_ramp_factor(self):
        step = self._global_step
        if step <= self.warmup_steps:
            return 0.0
        elapsed = step - self.warmup_steps
        return min(1.0, elapsed / self.warmup_ramp)

    def _get_effective_wd(self, base_wd):
        acc = self._cached_train_acc
        sigmoid_val = self._sigmoid(self.wd_scale, acc, self.wd_thresh)
        return base_wd * (1.0 + self.wd_ramp * sigmoid_val)

    def _get_gate_signal(self):
        return self._sigmoid(self.gate_scale, self._cached_train_acc, self.gate_thresh)

    def _get_effective_sam_freq(self):
        acc = self._cached_train_acc
        sam_heat = self._sigmoid(self.sam_scale, acc, self.sam_thresh)
        freq = self.sam_freq_max - (self.sam_freq_max - self.sam_freq_min) * sam_heat
        return max(1, round(freq))

    def _get_effective_bilevel_freq(self):
        acc = self._cached_train_acc
        bilevel_heat = self._sigmoid(self.bilevel_scale, acc, self.bilevel_thresh)
        freq = self.bilevel_freq_max - (self.bilevel_freq_max - self.bilevel_freq_min) * bilevel_heat
        return max(1, round(freq))

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
        elif train_acc is not None and train_acc >= self.zero_acc_threshold:
            self._update_alpha(train_loss, val_loss, train_acc)
        elif train_loss is not None and train_loss < self.zero_loss_threshold:
            self._update_alpha(train_loss, val_loss, train_acc)

        base_alpha = self._cached_alpha
        ramp = self._get_ramp_factor()
        layer_alphas = [max(0.0, min(1.0, base_alpha * f)) for f in self._flat_layer_alphas]

        group = self.param_groups[0]
        lr = group["lr"]
        beta2 = group["betas"][1]
        eps = group["eps"]
        wd_eff = self._get_effective_wd(group["weight_decay"])
        gate_signal = self._get_gate_signal()

        grads = []
        for p in self._flat_params:
            grads.append(p.grad.data if p.grad is not None else torch.Tensor())

        if self._weights_dirty:
            self._cached_weights = self.meta_net.get_weights()
            self._weights_dirty = False
        W1, b1, W2, b2, rescale = self._cached_weights

        _ops.supergrok15_fused_step(
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
            beta2, lr, wd_eff, eps,
            self.lamb, ramp, gate_signal,
            self.gradient_clipping,
        )

        return loss

    def sam_step(self, model, train_x, train_y, criterion):
        """SAM perturbation + sharpness computation. No bilevel."""
        self._ensure_state()
        train_grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                train_grads[name] = p.grad.detach().clone()
        if not train_grads:
            return 0.0

        named_params = list(model.named_parameters())
        flat_grads = [train_grads.get(n, torch.Tensor()) for n, _ in named_params]
        flat_params = [p.data for _, p in named_params]

        backups = _ops.supergrok15_sam_perturb_all(flat_params, flat_grads, self.sam_rho)

        model.zero_grad()
        with torch.enable_grad():
            sam_loss = criterion(model(train_x), train_y)
            sam_loss.backward()
        sam_loss_val = sam_loss.item()

        sam_grads = []
        for _, p in named_params:
            sam_grads.append(p.grad.detach().clone() if p.grad is not None else torch.Tensor())

        sharpness_out = [torch.zeros_like(p.data) for _, p in named_params]
        _ops.supergrok15_sharpness_restore_all(flat_params, sharpness_out, backups, sam_grads, flat_grads)

        for i, (name, p) in enumerate(named_params):
            pidx = self._param_to_idx.get(id(p))
            if pidx is not None and sharpness_out[i].numel() > 0:
                self._flat_sharpness[pidx] = sharpness_out[i]

        for name, p in named_params:
            p.grad = train_grads.get(name)

        return sam_loss_val

    def bilevel_step(self, model, train_x, train_y, val_x, val_y, criterion, meta_optimizer):
        """Bilevel meta-net training. Uses cached sharpness."""
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
        self._weights_dirty = True

        for name, p in named_params:
            p.grad = saved_grads.get(name)

        return val_loss.item()

    def sam_meta_step(self, model, train_x, train_y, val_x, val_y, criterion, meta_optimizer):
        """Combined SAM + bilevel (backward-compatible)."""
        sam_loss = self.sam_step(model, train_x, train_y, criterion)
        val_loss = self.bilevel_step(model, train_x, train_y, val_x, val_y, criterion, meta_optimizer)
        return sam_loss, val_loss

    def get_global_step(self):
        return self._global_step

    def get_cached_alpha(self):
        return self._cached_alpha

    def get_effective_wd(self):
        if self.param_groups:
            return self._get_effective_wd(self.param_groups[0]["weight_decay"])
        return 0.0

    def step_full(self, model, train_x, train_y, val_x, val_y, criterion=None):
        """Complete training step: forward + backward + SAM + meta-learning + optimizer."""
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
            except Exception:
                pass

        bilevel_freq_eff = self._get_effective_bilevel_freq()
        if step_num % bilevel_freq_eff == 0:
            try:
                metrics["val_loss"] = self.bilevel_step(
                    model, train_x, train_y, val_x, val_y, criterion, self._auto_meta_opt)
            except Exception:
                pass

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
