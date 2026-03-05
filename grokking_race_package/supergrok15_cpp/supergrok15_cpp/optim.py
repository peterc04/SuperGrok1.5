"""
SuperGrok v1.5 C++ — Python Optimizer Wrapper

Thin wrapper around the C++/CUDA extension. Identical API to the pure-Python
SuperGrok15, but step() and sam_meta_step() dispatch to fused C++/CUDA
operations that eliminate Python loop overhead and fuse per-element ops.

The meta-net lives in Python (as an nn.Module) for bilevel backward
compatibility.  During step(), its weights are extracted as flat tensors
and passed to the fused CUDA kernel which evaluates the 2→H→1 MLP
per-element in parallel without reshape/matmul overhead.

Key v1.5 optimizations:
  - Sigmoid gating (metric-driven, replaces cosine similarity gating)
  - Adaptive SAM frequency (sigmoid-driven, based on training accuracy)
  - Decoupled SAM and bilevel optimization (independent schedules)
  - Progressive weight decay (sigmoid-driven)
  - All dynamics are fully configurable via sigmoid parameters
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Optional, Callable, Dict, Any, Tuple

# ── Import C++ extension (with fallback) ──────────────────────────────
try:
    from supergrok15_cpp import _ops
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    import warnings
    warnings.warn(
        "supergrok15_cpp._ops not found — falling back to pure Python. "
        "Build with: pip install -e . (from the supergrok15_cpp directory)",
        RuntimeWarning,
    )


# ═══════════════════════════════════════════════════════════════════════
#  2D Sharpness-Aware Meta-Net (Python — used for bilevel backward)
# ═══════════════════════════════════════════════════════════════════════

class SharpnessMetaNet(nn.Module):
    """Element-wise gradient transformation conditioned on sharpness.

    Architecture: output = grad + rescale · MLP(grad, sharpness)
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
        out = flat_g + correction
        return out.reshape(shape)

    def get_weights(self):
        """Extract contiguous weight tensors for the C++ kernel."""
        W1 = self.net[0].weight.data.contiguous()  # (H, 2)
        b1 = self.net[0].bias.data.contiguous()     # (H,)
        W2 = self.net[2].weight.data.contiguous()   # (1, H)
        b2 = self.net[2].bias.data.contiguous()     # (1,)
        rescale = self.rescale.data.item()
        return W1, b1, W2, b2, rescale


# ═══════════════════════════════════════════════════════════════════════
#  SuperGrok v1.5 C++ Optimizer
# ═══════════════════════════════════════════════════════════════════════

class SuperGrok15(Optimizer):
    r"""SuperGrok v1.5 — C++/CUDA Accelerated Grokking Optimizer.

    Features:
      - Fused CUDA kernel for meta-net inference (per-element MLP)
      - Fused CUDA kernel for Adam + progressive weight decay
      - C++ parameter loop (no Python for-loop overhead)
      - C++ SAM perturbation + sharpness computation
      - Sigmoid gating (metric-driven, based on training accuracy)
      - Adaptive SAM frequency (sigmoid-driven schedule)
      - Decoupled SAM and bilevel optimization (independent schedules)
      - Progressive weight decay (sigmoid-driven)

    Falls back to pure Python if C++ extension is not built.
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

        # Hyperparameters
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

        # Meta-net (Python module for bilevel backward)
        if meta_net is None:
            self.meta_net = SharpnessMetaNet(meta_hidden_dim)
        else:
            self.meta_net = meta_net

        # Auto-move meta-net to same device as parameters
        try:
            first_param = next(iter(self.param_groups[0]["params"]))
            self.meta_net = self.meta_net.to(first_param.device)
        except (StopIteration, IndexError):
            pass  # no params yet, will be moved manually

        # Internal state
        self._global_step = 0
        self._cached_alpha = alpha_init
        self._cached_train_acc = 0.0
        self._sam_call_count = 0

        # Build flat parameter lists for C++ (avoids dict lookups per step)
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
        num_params = 0
        for group in self.param_groups:
            for p in group["params"]:
                num_params += 1
        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                self._flat_params.append(p)
                self._flat_steps.append(0)
                # Layer-wise beta1
                lb1 = beta1 * ((1.0 - gamma) ** idx)
                self._flat_layer_beta1s.append(lb1)
                # Layer-wise alpha (computed at step time, placeholder)
                if gamma_alpha == 0.0:
                    la_factor = 1.0
                else:
                    max_idx = max(num_params - 1, 1)
                    inverted = max_idx - idx
                    la_factor = (1.0 - gamma_alpha) ** inverted
                self._flat_layer_alphas.append(la_factor)
                self._param_to_idx[id(p)] = idx
                idx += 1

        self._num_params = num_params
        self._state_initialized = False

    def _ensure_state(self):
        """Lazily initialize flat state tensors on first step."""
        if self._state_initialized:
            return
        for i, p in enumerate(self._flat_params):
            self._flat_exp_avgs.append(torch.zeros_like(p.data))
            self._flat_exp_avg_sqs.append(torch.zeros_like(p.data))
            self._flat_mus.append(torch.zeros_like(p.data))
            self._flat_sharpness.append(torch.zeros_like(p.data))
        self._state_initialized = True

    # ══════════════════════════════════════════════════════════════════
    #  Sigmoid-driven dynamics (all use the same pattern as weight decay)
    # ══════════════════════════════════════════════════════════════════

    def _sigmoid(self, scale, value, thresh):
        """Compute sigmoid: 1 / (1 + exp(-scale * (value - thresh)))"""
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
        """Sigmoid gating based on training accuracy (replaces cosine similarity gating)."""
        acc = self._cached_train_acc
        return self._sigmoid(self.gate_scale, acc, self.gate_thresh)

    def _get_effective_sam_freq(self):
        """Sigmoid-driven SAM frequency. More SAM during transition, less early/late."""
        acc = self._cached_train_acc
        sam_heat = self._sigmoid(self.sam_scale, acc, self.sam_thresh)
        freq = self.sam_freq_max - (self.sam_freq_max - self.sam_freq_min) * sam_heat
        return max(1, round(freq))

    def _get_effective_bilevel_freq(self):
        """Sigmoid-driven bilevel frequency. Independent of SAM schedule."""
        acc = self._cached_train_acc
        bilevel_heat = self._sigmoid(self.bilevel_scale, acc, self.bilevel_thresh)
        freq = self.bilevel_freq_max - (self.bilevel_freq_max - self.bilevel_freq_min) * bilevel_heat
        return max(1, round(freq))

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

        self._ensure_state()
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

        # Compute layer-wise alphas (combine base_alpha with cached factor)
        layer_alphas = [
            max(0.0, min(1.0, base_alpha * f))
            for f in self._flat_layer_alphas
        ]

        # Get hyperparams from first group (C++ path only supports single group)
        if len(self.param_groups) > 1:
            import warnings
            warnings.warn("SuperGrok15 C++ path uses only the first param group's "
                          "hyperparameters. Multiple param groups will be ignored.",
                          RuntimeWarning, stacklevel=2)
        group = self.param_groups[0]
        lr = group["lr"]
        beta2 = group["betas"][1]
        eps = group["eps"]
        base_wd = group["weight_decay"]
        wd_eff = self._get_effective_wd(base_wd)

        # Sigmoid gate signal (computed once, not per-parameter)
        gate_signal = self._get_gate_signal()

        # Collect gradients (handle missing)
        grads = []
        for p in self._flat_params:
            if p.grad is not None:
                grads.append(p.grad.data)
            else:
                grads.append(torch.Tensor())  # empty

        # Extract meta-net weights for kernel
        W1, b1, W2, b2, rescale = self.meta_net.get_weights()

        if _HAS_CPP:
            # ── C++/CUDA path ─────────────────────────────────────────
            _ops.fused_step(
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
                beta2, lr, wd_eff, eps,
                self.lamb, ramp, gate_signal,
                self.gradient_clipping,
            )
        else:
            # ── Pure Python fallback ──────────────────────────────────
            self._python_step(grads, layer_alphas, lr, beta2, eps,
                              wd_eff, ramp, gate_signal,
                              W1, b1, W2, b2, rescale)

        return loss

    def _python_step(self, grads, layer_alphas, lr, beta2, eps,
                     wd_eff, ramp, gate_signal,
                     W1, b1, W2, b2, rescale):
        """Pure Python fallback when C++ extension is not available."""
        # Gradient clipping (manual norm computation on raw tensors)
        if self.gradient_clipping > 0:
            valid = [g for g in grads if g.numel() > 0]
            if valid:
                total_norm = torch.sqrt(sum(g.norm() ** 2 for g in valid))
                clip_coef = self.gradient_clipping / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    for g in valid:
                        g.mul_(clip_coef)

        # Compute lamb_eff once (sigmoid gating — no per-param computation)
        lamb_eff = ramp * gate_signal * self.lamb if ramp > 0 else 0.0

        for i in range(self._num_params):
            g = grads[i]
            if g.numel() == 0:
                continue

            self._flat_steps[i] += 1
            step = self._flat_steps[i]
            alpha = layer_alphas[i]
            beta1 = self._flat_layer_beta1s[i]
            mu = self._flat_mus[i]
            ea = self._flat_exp_avgs[i]
            easq = self._flat_exp_avg_sqs[i]
            sharp = self._flat_sharpness[i]
            p = self._flat_params[i]

            # mu EMA
            mu.mul_(alpha).add_(g, alpha=1.0 - alpha)

            # Meta-net
            smart_grad = self.meta_net(g, sharp)

            # Final gradient = smart_grad + lambda * mu
            fg = smart_grad + lamb_eff * mu
            ea.mul_(beta1).add_(fg, alpha=1.0 - beta1)
            easq.mul_(beta2).addcmul_(fg, fg, value=1.0 - beta2)

            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            step_size = lr / bc1
            denom = (easq.sqrt() / math.sqrt(bc2)).add_(eps)

            p.data.mul_(1.0 - lr * wd_eff)
            p.data.addcdiv_(ea, denom, value=-step_size)

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

        Perturbs parameters in the sharpness-ascent direction, computes
        sharpness signal |sam_grad - normal_grad|, then restores parameters.
        The cached sharpness is used by the meta-net during step().

        Returns sam_loss value.
        """
        self._ensure_state()

        # ── Step 1: Save training gradients ───────────────────────────
        train_grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                train_grads[name] = p.grad.detach().clone()

        if not train_grads:
            return 0.0

        # Build flat grad list for C++ SAM
        flat_grads_for_sam = []
        named_params = list(model.named_parameters())
        for name, p in named_params:
            if name in train_grads:
                flat_grads_for_sam.append(train_grads[name])
            else:
                flat_grads_for_sam.append(torch.Tensor())

        flat_params_for_sam = [p.data for _, p in named_params]

        # ── Step 2: SAM perturbation (C++ or Python) ──────────────────
        if _HAS_CPP:
            backups = _ops.sam_perturb_all(flat_params_for_sam,
                                            flat_grads_for_sam,
                                            self.sam_rho)
        else:
            grad_norm = 0.0
            for tg in train_grads.values():
                grad_norm += tg.norm().item() ** 2
            grad_norm = math.sqrt(grad_norm) + 1e-12
            backups = []
            for name, p in named_params:
                backups.append(p.data.clone())
                if name in train_grads:
                    p.data.add_(train_grads[name], alpha=self.sam_rho / grad_norm)

        # Forward + backward at perturbed point
        model.zero_grad()
        with torch.enable_grad():
            sam_loss = criterion(model(train_x), train_y)
            sam_loss.backward()
        sam_loss_val = sam_loss.item()

        # Collect SAM gradients
        sam_grads = []
        for name, p in named_params:
            if p.grad is not None:
                sam_grads.append(p.grad.detach().clone())
            else:
                sam_grads.append(torch.Tensor())

        # ── Step 3: Compute sharpness + restore (C++ or Python) ───────
        if _HAS_CPP:
            sharpness_out = [torch.zeros_like(p.data) for _, p in named_params]
            _ops.sharpness_restore_all(
                flat_params_for_sam, sharpness_out, backups,
                sam_grads, flat_grads_for_sam
            )
            for i, (name, p) in enumerate(named_params):
                pidx = self._param_to_idx.get(id(p))
                if pidx is not None and sharpness_out[i].numel() > 0:
                    self._flat_sharpness[pidx] = sharpness_out[i]
        else:
            for i, (name, p) in enumerate(named_params):
                p.data.copy_(backups[i])
                if name in train_grads and sam_grads[i].numel() > 0:
                    d = (sam_grads[i] - train_grads[name]).abs()
                    pidx = self._param_to_idx.get(id(p))
                    if pidx is not None:
                        self._flat_sharpness[pidx] = d

        # Restore training gradients
        for name, p in named_params:
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

        Computes smart gradients via meta-net, then aligns them with
        validation gradient direction to train the meta-net.

        Returns val_loss value.
        """
        self._ensure_state()
        named_params = list(model.named_parameters())

        # Save current gradients
        saved_grads = {}
        for name, p in named_params:
            if p.grad is not None:
                saved_grads[name] = p.grad.detach().clone()

        # Compute smart gradients using cached sharpness
        smart_grads = {}
        for name, p in named_params:
            if name in saved_grads:
                pidx = self._param_to_idx.get(id(p))
                sharp = self._flat_sharpness[pidx] if pidx is not None else torch.zeros_like(p.data)
                smart_grads[name] = self.meta_net(saved_grads[name], sharp)

        # Validation gradients
        model.zero_grad()
        with torch.enable_grad():
            val_loss = criterion(model(val_x), val_y)
            val_loss.backward()

        # Meta-loss = -<smart_grad, val_grad_unit>
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

        # Restore saved gradients
        for name, p in named_params:
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
        """Combined SAM + bilevel (backward-compatible API).

        Calls sam_step() then bilevel_step() sequentially.
        """
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
        mu_norms = [m.norm().item() for m in self._flat_mus if m.numel() > 0] if self._state_initialized else []
        sharp_norms = [s.norm().item() for s in self._flat_sharpness if s.numel() > 0 and s.norm().item() > 0] if self._state_initialized else []
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
            "avg_sharpness_norm": sum(sharp_norms) / max(len(sharp_norms), 1),
            "sharpness_cached": any(s.norm().item() > 0 for s in self._flat_sharpness) if self._state_initialized else False,
            "cpp_backend": _HAS_CPP,
        }

    def __repr__(self) -> str:
        backend = "C++/CUDA" if _HAS_CPP else "Python"
        lines = [f"SuperGrok v1.5 [{backend}] ("]
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
    #  Drop-in API: step_full() — complete pipeline in one call
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
        """Complete training step: forward + backward + SAM + meta-learning + optimizer.

        This is the fully self-contained API. Call this instead of manually
        orchestrating loss.backward(), sam_step(), bilevel_step(), and step().

        SAM and bilevel frequencies are automatically managed via sigmoid schedules.
        Metrics are only computed when needed (deferred .item() calls).

        Usage:
            opt = SuperGrok15(model.parameters(), lr=1e-3)
            for batch_x, batch_y in dataloader:
                metrics = opt.step_full(model, batch_x, batch_y, val_x, val_y)
                print(f"loss={metrics['train_loss']:.4f}")

        Returns a dict with: train_loss, train_acc (when computed),
        val_loss (when computed), sam_loss (when SAM step runs).
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Auto-create meta optimizer on first call
        if not hasattr(self, '_auto_meta_opt'):
            self._auto_meta_opt = torch.optim.Adam(
                self.meta_net.parameters(), lr=1e-4)

        # ── Forward + backward ────────────────────────────────────────
        model.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()

        metrics: Dict[str, float] = {}
        step_num = self._global_step + 1  # Will be incremented in step()

        # ── Adaptive SAM (sigmoid-driven frequency) ───────────────────
        sam_freq_eff = self._get_effective_sam_freq()
        if step_num % sam_freq_eff == 0:
            try:
                sam_loss = self.sam_step(model, train_x, train_y, criterion)
                metrics["sam_loss"] = sam_loss
            except Exception:
                pass

        # ── Adaptive bilevel (independent sigmoid-driven frequency) ───
        bilevel_freq_eff = self._get_effective_bilevel_freq()
        if step_num % bilevel_freq_eff == 0:
            try:
                val_loss = self.bilevel_step(
                    model, train_x, train_y, val_x, val_y,
                    criterion, self._auto_meta_opt)
                metrics["val_loss"] = val_loss
            except Exception:
                pass

        # ── Deferred metrics (only compute .item() when needed) ───────
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
