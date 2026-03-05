"""
SuperGrok v1.5 — Low-Data Grokking Optimizer (Pure Python)

Builds on v1.1 with optimizations targeting the low-data regime
(ft10/ft25) where grokking fails because the memorization->generalization
signal is too weak for temporal filtering alone.

Key features:
  1. 2D SharpnessMetaNet: input is (gradient, sharpness_signal) per element.
     The sharpness signal tells the meta-net which gradient components sit
     in sharp basins (memorization) vs flat basins (generalization).

  2. LookSAM integration: adaptive SAM frequency (sigmoid-driven) to
     compute sharpness directions when they matter most.

  3. Decoupled SAM and bilevel optimization: independent sigmoid-driven
     schedules for SAM perturbation and meta-net training.

  4. Sigmoid gating: metric-driven gate (based on training accuracy)
     replaces cosine similarity gating for faster computation.

  5. Progressive weight decay: gentle during feature learning, aggressive
     after memorization. wd_eff = wd * (1 + wd_ramp * sigmoid(s * (acc - th))).

All dynamic schedules use the same sigmoid pattern, making them
fully configurable via (scale, threshold, min, max) parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Optional, Callable, Dict, Any, Tuple


# ═══════════════════════════════════════════════════════════════════════
#  2D Sharpness-Aware Meta-Net
# ═══════════════════════════════════════════════════════════════════════

class SharpnessMetaNet(nn.Module):
    """Element-wise gradient transformation conditioned on sharpness.

    Architecture::

        correction = rescale * MLP([grad, sharp])
        output     = grad + correction

    The skip connection means the network starts as identity (rescale=0)
    and learns a correction informed by landscape geometry.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
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
        out = flat_g + correction
        return out.reshape(shape)


# ═══════════════════════════════════════════════════════════════════════
#  SuperGrok v1.5 Optimizer (Pure Python)
# ═══════════════════════════════════════════════════════════════════════

class SuperGrok15(Optimizer):
    r"""SuperGrok v1.5 — Sharpness-Aware Grokking Optimizer.

    Features:
      - Sigmoid gating (metric-driven, based on training accuracy)
      - Adaptive SAM frequency (sigmoid-driven schedule)
      - Decoupled SAM and bilevel optimization (independent schedules)
      - 2D meta-net with sharpness signal
      - Progressive weight decay (sigmoid-driven)
      - Layer-wise beta1 and alpha decay

    All sigmoid-driven dynamics use the pattern:
        heat = sigmoid(scale * (metric - threshold))
        effective_value = max_val - (max_val - min_val) * heat

    Args:
        params: Model parameters.
        lr: Learning rate. Default: 1e-3.
        betas: Adam momentum coefficients. Default: (0.9, 0.999).
        eps: Numerical stability. Default: 1e-8.
        weight_decay: Base weight decay. Default: 1.0.
        alpha_init: EMA momentum for memory buffer. Default: 0.98.
        lamb: Peak amplification factor. Default: 2.0.
        gamma: Layer-wise beta1 decay. Default: 0.1.
        gamma_alpha: Layer-wise alpha decay. Default: 0.0.
        kappa: Grokking signal decay rate. Default: 0.1.
        warmup_steps: Steps before amplification. Default: 100.
        warmup_ramp: Ramp length after warmup. Default: 100.
        gradient_clipping: Max gradient norm. Default: 1.0.
        meta_hidden_dim: Hidden dim for meta-net. Default: 32.
        alpha_update_freq: Steps between alpha updates. Default: 100.
        zero_loss_threshold: Memorization loss threshold. Default: 1e-4.
        zero_acc_threshold: Memorization accuracy threshold. Default: 0.995.
        sam_rho: SAM perturbation radius. Default: 0.05.
        gate_scale: Sigmoid steepness for gating. Default: 20.0.
        gate_thresh: Accuracy threshold for gate activation. Default: 0.8.
        sam_freq_min: Most aggressive SAM frequency. Default: 3.
        sam_freq_max: Laziest SAM frequency. Default: 20.
        sam_scale: Sigmoid steepness for SAM schedule. Default: 20.0.
        sam_thresh: Accuracy threshold for SAM activation. Default: 0.85.
        bilevel_freq_min: Most aggressive bilevel frequency. Default: 5.
        bilevel_freq_max: Laziest bilevel frequency. Default: 30.
        bilevel_scale: Sigmoid steepness for bilevel schedule. Default: 20.0.
        bilevel_thresh: Accuracy threshold for bilevel activation. Default: 0.9.
        wd_ramp: Max weight decay multiplier above base. Default: 4.0.
        wd_scale: Sigmoid steepness for progressive wd. Default: 20.0.
        wd_thresh: Accuracy threshold for wd ramp onset. Default: 0.9.
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
        # ── Sigmoid gating (replaces cosine gating) ──────────────────
        gate_scale: float = 20.0,
        gate_thresh: float = 0.8,
        # ── Adaptive SAM frequency (sigmoid-driven) ──────────────────
        sam_freq_min: int = 3,
        sam_freq_max: int = 20,
        sam_scale: float = 20.0,
        sam_thresh: float = 0.85,
        # ── Decoupled bilevel optimization ────────────────────────────
        bilevel_freq_min: int = 5,
        bilevel_freq_max: int = 30,
        bilevel_scale: float = 20.0,
        bilevel_thresh: float = 0.9,
        # ── Progressive weight decay ─────────────────────────────────
        wd_ramp: float = 4.0,
        wd_scale: float = 20.0,
        wd_thresh: float = 0.9,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= alpha_init <= 1.0):
            raise ValueError(f"Invalid alpha_init: {alpha_init}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if lamb < 0.0:
            raise ValueError(f"Invalid lamb: {lamb}")
        if sam_rho < 0.0:
            raise ValueError(f"Invalid sam_rho: {sam_rho}")
        if sam_freq_min < 1:
            raise ValueError(f"Invalid sam_freq_min: {sam_freq_min}")
        if sam_freq_max < sam_freq_min:
            raise ValueError(f"sam_freq_max ({sam_freq_max}) must be >= sam_freq_min ({sam_freq_min})")
        if bilevel_freq_min < 1:
            raise ValueError(f"Invalid bilevel_freq_min: {bilevel_freq_min}")
        if bilevel_freq_max < bilevel_freq_min:
            raise ValueError(f"bilevel_freq_max ({bilevel_freq_max}) must be >= bilevel_freq_min ({bilevel_freq_min})")
        if wd_ramp < 0.0:
            raise ValueError(f"Invalid wd_ramp: {wd_ramp}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Core hyperparameters
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

        # Sigmoid gating
        self.gate_scale = gate_scale
        self.gate_thresh = gate_thresh

        # Adaptive SAM frequency
        self.sam_freq_min = sam_freq_min
        self.sam_freq_max = sam_freq_max
        self.sam_scale = sam_scale
        self.sam_thresh = sam_thresh

        # Decoupled bilevel
        self.bilevel_freq_min = bilevel_freq_min
        self.bilevel_freq_max = bilevel_freq_max
        self.bilevel_scale = bilevel_scale
        self.bilevel_thresh = bilevel_thresh

        # Progressive weight decay
        self.wd_ramp = wd_ramp
        self.wd_scale = wd_scale
        self.wd_thresh = wd_thresh

        # Meta-net
        if meta_net is None:
            self.meta_net = SharpnessMetaNet(meta_hidden_dim)
        else:
            self.meta_net = meta_net

        # Auto-move meta-net to same device as parameters
        try:
            first_param = next(iter(self.param_groups[0]["params"]))
            self.meta_net = self.meta_net.to(first_param.device)
        except (StopIteration, IndexError):
            pass

        # Internal state
        self._global_step = 0
        self._cached_alpha = alpha_init
        self._cached_train_acc = 0.0
        self._layer_beta1_cache: Dict[torch.nn.Parameter, float] = {}
        self._layer_alpha_cache: Dict[torch.nn.Parameter, float] = {}
        self._sharpness_cache: Dict[int, torch.Tensor] = {}

        # Build parameter index
        self._param_index: Dict[torch.nn.Parameter, int] = {}
        self._num_params = 0
        for group in self.param_groups:
            for p in group["params"]:
                self._param_index[p] = self._num_params
                self._num_params += 1

    # ══════════════════════════════════════════════════════════════════
    #  Sigmoid-driven dynamics
    # ══════════════════════════════════════════════════════════════════

    def _sigmoid(self, scale, value, thresh):
        """Compute sigmoid: 1 / (1 + exp(-scale * (value - thresh)))"""
        return 1.0 / (1.0 + math.exp(-scale * (value - thresh)))

    def _get_layer_beta1(self, p: torch.nn.Parameter, beta1: float, gamma: float) -> float:
        if p in self._layer_beta1_cache:
            return self._layer_beta1_cache[p]
        idx = self._param_index.get(p, 0)
        val = beta1 * ((1.0 - gamma) ** idx)
        self._layer_beta1_cache[p] = val
        return val

    def _get_layer_alpha(self, p: torch.nn.Parameter, alpha: float) -> float:
        if self.gamma_alpha == 0.0:
            return alpha
        if p not in self._layer_alpha_cache:
            idx = self._param_index.get(p, 0)
            max_idx = max(self._num_params - 1, 1)
            inverted = max_idx - idx
            self._layer_alpha_cache[p] = (1.0 - self.gamma_alpha) ** inverted
        return max(0.0, min(1.0, alpha * self._layer_alpha_cache[p]))

    def _get_ramp_factor(self) -> float:
        step = self._global_step
        if step <= self.warmup_steps:
            return 0.0
        elapsed = step - self.warmup_steps
        return min(1.0, elapsed / self.warmup_ramp)

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

    def _get_effective_wd(self, base_wd: float) -> float:
        acc = self._cached_train_acc
        sigmoid_val = self._sigmoid(self.wd_scale, acc, self.wd_thresh)
        return base_wd * (1.0 + self.wd_ramp * sigmoid_val)

    def _get_gate_signal(self) -> float:
        """Sigmoid gating based on training accuracy."""
        acc = self._cached_train_acc
        return self._sigmoid(self.gate_scale, acc, self.gate_thresh)

    def _get_effective_sam_freq(self) -> int:
        """Sigmoid-driven SAM frequency."""
        acc = self._cached_train_acc
        sam_heat = self._sigmoid(self.sam_scale, acc, self.sam_thresh)
        freq = self.sam_freq_max - (self.sam_freq_max - self.sam_freq_min) * sam_heat
        return max(1, round(freq))

    def _get_effective_bilevel_freq(self) -> int:
        """Sigmoid-driven bilevel frequency."""
        acc = self._cached_train_acc
        bilevel_heat = self._sigmoid(self.bilevel_scale, acc, self.bilevel_thresh)
        freq = self.bilevel_freq_max - (self.bilevel_freq_max - self.bilevel_freq_min) * bilevel_heat
        return max(1, round(freq))

    def _get_sharpness(self, p: torch.nn.Parameter) -> torch.Tensor:
        pidx = self._param_index.get(p, -1)
        if pidx in self._sharpness_cache:
            return self._sharpness_cache[pidx]
        return torch.zeros_like(p.data)

    # ══════════════════════════════════════════════════════════════════
    #  Main optimizer step
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
    ):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        if train_acc is not None:
            self._cached_train_acc = train_acc

        # Alpha update
        if self._global_step % self.alpha_update_freq == 0 or self._global_step == 1:
            self._update_alpha(train_loss, val_loss, train_acc)
        elif train_acc is not None and train_acc >= self.zero_acc_threshold:
            self._update_alpha(train_loss, val_loss, train_acc)
        elif train_loss is not None and train_loss < self.zero_loss_threshold:
            self._update_alpha(train_loss, val_loss, train_acc)

        base_alpha = self._cached_alpha
        ramp = self._get_ramp_factor()

        # Sigmoid gate signal (computed once, not per-parameter)
        gate_signal = self._get_gate_signal()
        lamb_eff = ramp * gate_signal * self.lamb if ramp > 0 else 0.0

        # Gradient clipping
        if self.gradient_clipping > 0:
            all_params = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, self.gradient_clipping)

        # Per-parameter update
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            base_wd = group["weight_decay"]
            wd_eff = self._get_effective_wd(base_wd)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Lazy state init
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["mu"] = torch.zeros_like(p.data)

                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                mu = state["mu"]

                layer_beta1 = self._get_layer_beta1(p, beta1, self.gamma)
                layer_alpha = self._get_layer_alpha(p, base_alpha)

                # Decoupled Memory: mu tracks raw gradients only
                mu.mul_(layer_alpha).add_(grad, alpha=1.0 - layer_alpha)

                # 2D Meta-net transformation
                sharpness = self._get_sharpness(p)
                smart_grad = self.meta_net(grad, sharpness)

                # Sigmoid gating: final_grad = smart_grad + lamb_eff * mu
                final_grad = smart_grad + lamb_eff * mu

                # AdamW update
                exp_avg.mul_(layer_beta1).add_(final_grad, alpha=1.0 - layer_beta1)
                exp_avg_sq.mul_(beta2).addcmul_(final_grad, final_grad, value=1.0 - beta2)

                bc1 = 1.0 - layer_beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]

                step_size = lr / bc1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)

                p.data.mul_(1.0 - lr * wd_eff)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    # ══════════════════════════════════════════════════════════════════
    #  Decoupled SAM + Bilevel
    # ══════════════════════════════════════════════════════════════════

    def sam_step(
        self,
        model: nn.Module,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        criterion: Callable,
    ) -> float:
        """SAM perturbation + sharpness computation ONLY. No bilevel.

        Returns sam_loss value.
        """
        # Save current training gradients
        train_grads: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                train_grads[name] = p.grad.detach().clone()

        if not train_grads:
            return 0.0

        # Compute grad norm for SAM perturbation
        grad_norm = 0.0
        for tg in train_grads.values():
            grad_norm += tg.norm().item() ** 2
        grad_norm = math.sqrt(grad_norm) + 1e-12

        # Perturb parameters
        param_backups: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            param_backups[name] = p.data.clone()
            if name in train_grads:
                p.data.add_(train_grads[name], alpha=self.sam_rho / grad_norm)

        # Forward + backward at perturbed point
        model.zero_grad()
        with torch.enable_grad():
            sam_loss = criterion(model(train_x), train_y)
            sam_loss.backward()

        # Cache sharpness direction: d = |sam_grad - normal_grad|
        for name, p in model.named_parameters():
            if p.grad is not None and name in train_grads:
                d = (p.grad.detach() - train_grads[name]).abs()
                pidx = self._param_index.get(p, -1)
                if pidx >= 0:
                    self._sharpness_cache[pidx] = d

        sam_loss_val = sam_loss.item()

        # Restore parameters
        for name, p in model.named_parameters():
            if name in param_backups:
                p.data.copy_(param_backups[name])

        # Restore training gradients
        for name, p in model.named_parameters():
            if name in train_grads:
                p.grad = train_grads[name]
            else:
                p.grad = None

        return sam_loss_val

    def bilevel_step(
        self,
        model: nn.Module,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        criterion: Callable,
        meta_optimizer: Optimizer,
    ) -> float:
        """Bilevel meta-net training ONLY. Uses cached sharpness.

        Returns val_loss value.
        """
        # Save current gradients
        saved_grads: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                saved_grads[name] = p.grad.detach().clone()

        # Compute smart gradients using cached sharpness
        smart_grads: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in saved_grads:
                pidx = self._param_index.get(p, -1)
                sharpness = self._sharpness_cache.get(pidx, torch.zeros_like(p.data))
                smart_grads[name] = self.meta_net(saved_grads[name], sharpness)

        # Validation gradients
        model.zero_grad()
        with torch.enable_grad():
            val_loss = criterion(model(val_x), val_y)
            val_loss.backward()

        # Meta-loss = -<smart_grad, val_grad_unit>
        meta_optimizer.zero_grad()
        device = val_x.device
        meta_loss = torch.tensor(0.0, device=device)
        for name, p in model.named_parameters():
            if name in smart_grads and p.grad is not None:
                vg = p.grad.detach()
                vg_norm = vg.norm()
                vg_unit = vg / vg_norm if vg_norm > 1e-12 else vg
                meta_loss = meta_loss - (smart_grads[name] * vg_unit).sum()

        meta_loss.backward()
        meta_optimizer.step()

        # Restore saved gradients
        for name, p in model.named_parameters():
            if name in saved_grads:
                p.grad = saved_grads[name]
            else:
                p.grad = None

        return val_loss.item()

    def sam_meta_step(
        self,
        model: nn.Module,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        criterion: Callable,
        meta_optimizer: Optimizer,
    ) -> Tuple[float, float]:
        """Combined SAM + bilevel (backward-compatible API)."""
        sam_loss_val = self.sam_step(model, train_x, train_y, criterion)
        val_loss_val = self.bilevel_step(
            model, train_x, train_y, val_x, val_y, criterion, meta_optimizer)
        return sam_loss_val, val_loss_val

    # ══════════════════════════════════════════════════════════════════
    #  Inspection helpers
    # ══════════════════════════════════════════════════════════════════

    def get_global_step(self) -> int:
        return self._global_step

    def get_cached_alpha(self) -> float:
        return self._cached_alpha

    def get_effective_wd(self) -> float:
        if self.param_groups:
            return self._get_effective_wd(self.param_groups[0]["weight_decay"])
        return 0.0

    def get_state_summary(self) -> Dict[str, Any]:
        mu_norms, ea_norms, sharp_norms = [], [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state:
                    s = self.state[p]
                    if "mu" in s:
                        mu_norms.append(s["mu"].norm().item())
                    if "exp_avg" in s:
                        ea_norms.append(s["exp_avg"].norm().item())
                pidx = self._param_index.get(p, -1)
                if pidx in self._sharpness_cache:
                    sharp_norms.append(self._sharpness_cache[pidx].norm().item())
        return {
            "global_step": self._global_step,
            "cached_alpha": self._cached_alpha,
            "cached_train_acc": self._cached_train_acc,
            "ramp_factor": self._get_ramp_factor(),
            "effective_wd": self.get_effective_wd(),
            "gate_signal": self._get_gate_signal(),
            "effective_sam_freq": self._get_effective_sam_freq(),
            "effective_bilevel_freq": self._get_effective_bilevel_freq(),
            "avg_mu_norm": sum(mu_norms) / max(len(mu_norms), 1),
            "avg_exp_avg_norm": sum(ea_norms) / max(len(ea_norms), 1),
            "avg_sharpness_norm": sum(sharp_norms) / max(len(sharp_norms), 1),
            "sharpness_cached": len(self._sharpness_cache) > 0,
        }

    def __repr__(self) -> str:
        lines = ["SuperGrok v1.5 ("]
        for group in self.param_groups:
            lines.append(f"  lr={group['lr']}, betas={group['betas']}, "
                         f"eps={group['eps']}, wd={group['weight_decay']}")
        lines.append(f"  alpha_init={self.alpha_init}, lamb={self.lamb}, "
                     f"gamma={self.gamma}")
        lines.append(f"  gate: scale={self.gate_scale}, thresh={self.gate_thresh}")
        lines.append(f"  sam: rho={self.sam_rho}, freq=[{self.sam_freq_min},{self.sam_freq_max}], "
                     f"scale={self.sam_scale}, thresh={self.sam_thresh}")
        lines.append(f"  bilevel: freq=[{self.bilevel_freq_min},{self.bilevel_freq_max}], "
                     f"scale={self.bilevel_scale}, thresh={self.bilevel_thresh}")
        lines.append(f"  wd: ramp={self.wd_ramp}, scale={self.wd_scale}, "
                     f"thresh={self.wd_thresh}")
        lines.append(")")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════
    #  Drop-in API: step_full()
    # ══════════════════════════════════════════════════════════════════

    def step_full(
        self,
        model: nn.Module,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        criterion: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Complete training step with automatic SAM/bilevel scheduling.

        Usage:
            opt = SuperGrok15(model.parameters(), lr=1e-3)
            for batch_x, batch_y in dataloader:
                metrics = opt.step_full(model, batch_x, batch_y, val_x, val_y)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        if not hasattr(self, '_auto_meta_opt'):
            self._auto_meta_opt = torch.optim.Adam(
                self.meta_net.parameters(), lr=1e-4)

        # Forward + backward
        model.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()

        metrics: Dict[str, float] = {}
        step_num = self._global_step + 1

        # Adaptive SAM (sigmoid-driven frequency)
        sam_freq_eff = self._get_effective_sam_freq()
        if step_num % sam_freq_eff == 0:
            try:
                sam_loss = self.sam_step(model, train_x, train_y, criterion)
                metrics["sam_loss"] = sam_loss
            except Exception:
                pass

        # Adaptive bilevel (independent sigmoid-driven frequency)
        bilevel_freq_eff = self._get_effective_bilevel_freq()
        if step_num % bilevel_freq_eff == 0:
            try:
                val_loss = self.bilevel_step(
                    model, train_x, train_y, val_x, val_y,
                    criterion, self._auto_meta_opt)
                metrics["val_loss"] = val_loss
            except Exception:
                pass

        # Deferred metrics (only compute .item() when needed)
        kw: Dict[str, float] = {}
        alpha_freq = self.alpha_update_freq
        needs_metrics = (step_num % alpha_freq == 0) or step_num == 1
        if needs_metrics:
            with torch.no_grad():
                train_loss_val = loss.item()
                train_acc = (logits.detach().argmax(-1) == train_y).float().mean().item()
            kw["train_loss"] = train_loss_val
            kw["train_acc"] = train_acc
            metrics["train_loss"] = train_loss_val
            metrics["train_acc"] = train_acc

            if step_num % alpha_freq == 0:
                with torch.no_grad():
                    val_logits = model(val_x)
                    val_loss_val = criterion(val_logits, val_y).item()
                kw["val_loss"] = val_loss_val
                if "val_loss" not in metrics:
                    metrics["val_loss"] = val_loss_val

        self.step(**kw)

        return metrics
