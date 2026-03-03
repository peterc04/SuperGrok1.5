"""
SuperGrok v1.5 — Low-Data Grokking Optimizer

Builds on v1.1 with three modifications targeting the low-data regime
(ft10/ft25) where grokking fails because the memorization→generalization
signal is too weak for temporal filtering alone.

Changes from v1.1:
  1. 2D SharpnessMetaNet: input is (gradient, sharpness_signal) per element.
     The sharpness signal tells the meta-net which gradient components sit
     in sharp basins (memorization) vs flat basins (generalization), giving
     it a direct structural signal instead of relying on temporal patterns.

  2. LookSAM integration: every k steps, compute SAM perturbation to get
     the sharpness direction, cache it for the k−1 intermediate steps.
     Synced with meta_step cadence to amortize overhead.

  3. Progressive weight decay: gentle during feature learning, aggressive
     after memorization.  wd_eff = wd · (1 + wd_ramp · σ(s · (acc − θ))).

Inherits from v1.1:
  - Decoupled Memory (Rule 1): mu tracks raw gradients only
  - Soft Sigmoid Gating (Rule 2): smooth amplification control
  - Memorization Fix (Rule 3): adaptive alpha via train_acc / train_loss
  - Layer-wise β₁ and α decay
  - Smooth warmup ramp
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Optional, Callable, Dict, Any, Tuple


# ═══════════════════════════════════════════════════════════════════════
#  2D Sharpness-Aware Meta-Net  (v1.5 NEW)
# ═══════════════════════════════════════════════════════════════════════

class SharpnessMetaNet(nn.Module):
    """Element-wise gradient transformation conditioned on sharpness.

    Takes two signals per element:
      - ``grad``:  the raw gradient value
      - ``sharp``: the sharpness signal (magnitude of gradient change
        under SAM perturbation)

    Architecture::

        correction = rescale · MLP([grad, sharp])
        output     = grad + correction

    The skip connection over the raw gradient means the network starts
    as identity (``rescale`` initialised to 0) and learns a *correction*
    informed by landscape geometry.

    The 2D input space has clean geometry:
      - High sharpness + large grad → memorization → suppress
      - Low sharpness  + large grad → generalization → amplify
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start at 0 so skip connection dominates → near-identity
        self.rescale = nn.Parameter(torch.zeros(1))

        # Small random init
        with torch.no_grad():
            self.net[0].weight.normal_(0, 0.01)
            self.net[0].bias.zero_()
            self.net[2].weight.normal_(0, 0.01)
            self.net[2].bias.zero_()

    def forward(self, grad: torch.Tensor, sharpness: torch.Tensor) -> torch.Tensor:
        """Transform gradient using sharpness context.

        Args:
            grad: Raw gradient tensor of any shape.
            sharpness: Sharpness signal, same shape as grad.
                Typically ``|sam_grad − normal_grad|`` per element.

        Returns:
            Transformed gradient, same shape as input.
        """
        if grad.numel() == 0:
            return grad
        shape = grad.shape
        # Stack into (N, 2) input
        flat_g = grad.reshape(-1, 1)
        flat_s = sharpness.reshape(-1, 1)
        inp = torch.cat([flat_g, flat_s], dim=1)  # (N, 2)
        correction = self.rescale * self.net(inp)   # (N, 1)
        out = flat_g + correction
        return out.reshape(shape)


# ═══════════════════════════════════════════════════════════════════════
#  SuperGrok v1.5 Optimizer
# ═══════════════════════════════════════════════════════════════════════

class SuperGrok15(Optimizer):
    r"""SuperGrok v1.5 — Sharpness-Aware Grokking Optimizer.

    Extends v1.1 with LookSAM-based sharpness signals, a 2D meta-net,
    and progressive weight decay for improved low-data grokking.

    **New in v1.5:**

    **LookSAM Integration**
        Every ``sam_freq`` steps, compute the SAM perturbation to find
        the worst-case gradient direction.  Cache the *sharpness direction*
        ``d = sam_grad − normal_grad`` for use in intermediate steps.
        On intermediate steps, the cached ``d`` provides the sharpness
        signal to the meta-net at negligible cost.

    **2D Meta-Net**
        The meta-net receives ``(gradient, |cached_d|)`` per element.
        The sharpness channel gives it a direct structural signal about
        whether each gradient component sits in a sharp basin
        (memorization) or flat basin (generalization).

    **Progressive Weight Decay**
        Weight decay ramps up after memorization is detected:

            wd_eff = wd · (1 + wd_ramp · sigmoid(wd_scale · (acc − wd_thresh)))

        Below the accuracy threshold, decay stays near ``wd``.
        Above it, decay ramps up to ``wd · (1 + wd_ramp)``.

    Args:
        params: Model parameters.
        lr (float): Learning rate.  Default: ``1e-3``.
        betas (tuple): Adam momentum coefficients.  Default: ``(0.9, 0.999)``.
        eps (float): Numerical stability.  Default: ``1e-8``.
        weight_decay (float): Base weight decay.  Default: ``1.0``.
        alpha_init (float): EMA momentum for Grokfast buffer.
            Default: ``0.98``.
        lamb (float): Peak amplification factor.  Default: ``2.0``.
        gamma (float): Layer-wise β₁ decay.  Default: ``0.1``.
        gamma_alpha (float): Layer-wise α decay.  Default: ``0.0``.
        kappa (float): Grokking signal decay rate.  Default: ``0.1``.
        warmup_steps (int): Steps before amplification.  Default: ``100``.
        warmup_ramp (int): Ramp length after warmup.  Default: ``100``.
        gradient_clipping (float): Max gradient norm.  Default: ``1.0``.
        meta_net (nn.Module | None): Custom meta-net.  Default: auto-created.
        meta_hidden_dim (int): Hidden dim for auto meta-net.  Default: ``32``.
        alpha_update_freq (int): Steps between alpha updates.  Default: ``100``.
        gate_temperature (float): Sigmoid gate temperature.  Default: ``5.0``.
        zero_loss_threshold (float): Memorization loss threshold.
            Default: ``1e-4``.
        zero_acc_threshold (float): Memorization accuracy threshold.
            Default: ``0.995``.
        sam_rho (float): SAM perturbation radius.  Default: ``0.05``.
        sam_freq (int): Steps between SAM computations.  Default: ``5``.
        wd_ramp (float): Max weight decay multiplier above base.
            wd_eff ranges from wd to wd·(1+wd_ramp).  Default: ``4.0``.
        wd_scale (float): Sigmoid steepness for progressive wd.
            Default: ``20.0``.
        wd_thresh (float): Accuracy threshold for wd ramp onset.
            Default: ``0.9``.
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
        gate_temperature: float = 5.0,
        zero_loss_threshold: float = 1e-4,
        zero_acc_threshold: float = 0.995,
        # v1.5 new params
        sam_rho: float = 0.05,
        sam_freq: int = 5,
        wd_ramp: float = 4.0,
        wd_scale: float = 20.0,
        wd_thresh: float = 0.9,
    ):
        # ── Validation ────────────────────────────────────────────────
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
        if gate_temperature <= 0.0:
            raise ValueError(f"Invalid gate_temperature: {gate_temperature}")
        if sam_rho < 0.0:
            raise ValueError(f"Invalid sam_rho: {sam_rho}")
        if sam_freq < 1:
            raise ValueError(f"Invalid sam_freq: {sam_freq}")
        if wd_ramp < 0.0:
            raise ValueError(f"Invalid wd_ramp: {wd_ramp}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # ── v1.1 hyperparameters ──────────────────────────────────────
        self.alpha_init = alpha_init
        self.lamb = lamb
        self.gamma = gamma
        self.gamma_alpha = gamma_alpha
        self.kappa = kappa
        self.warmup_steps = warmup_steps
        self.warmup_ramp = max(1, warmup_ramp)
        self.gradient_clipping = gradient_clipping
        self.alpha_update_freq = alpha_update_freq
        self.gate_temperature = gate_temperature
        self.zero_loss_threshold = zero_loss_threshold
        self.zero_acc_threshold = zero_acc_threshold

        # ── v1.5 hyperparameters ──────────────────────────────────────
        self.sam_rho = sam_rho
        self.sam_freq = sam_freq
        self.wd_ramp = wd_ramp
        self.wd_scale = wd_scale
        self.wd_thresh = wd_thresh

        # ── Meta-net (2D sharpness-aware) ─────────────────────────────
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

        # ── Internal state ────────────────────────────────────────────
        self._global_step = 0
        self._cached_alpha = alpha_init
        self._cached_train_acc = 0.0
        self._layer_beta1_cache: Dict[torch.nn.Parameter, float] = {}
        self._layer_alpha_cache: Dict[torch.nn.Parameter, float] = {}
        # Cached sharpness directions from LookSAM (keyed by param index)
        self._sharpness_cache: Dict[int, torch.Tensor] = {}

        # Build parameter index
        self._param_index: Dict[torch.nn.Parameter, int] = {}
        self._num_params = 0
        for group in self.param_groups:
            for p in group["params"]:
                self._param_index[p] = self._num_params
                self._num_params += 1

    # ══════════════════════════════════════════════════════════════════
    #  Layer-wise helpers (same as v1.1)
    # ══════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════
    #  Alpha update (grokking signal) — same as v1.1
    # ══════════════════════════════════════════════════════════════════

    def _update_alpha(
        self,
        train_loss: Optional[float],
        val_loss: Optional[float],
        train_acc: Optional[float],
    ) -> None:
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

    # ══════════════════════════════════════════════════════════════════
    #  Progressive Weight Decay  (v1.5 NEW)
    # ══════════════════════════════════════════════════════════════════

    def _get_effective_wd(self, base_wd: float) -> float:
        """Compute progressive weight decay based on train accuracy.

        Below ``wd_thresh``, decay ≈ base_wd.
        Above ``wd_thresh``, decay ramps toward base_wd · (1 + wd_ramp).

        With defaults (wd=1.0, wd_ramp=4.0, wd_thresh=0.9, wd_scale=20):
          - acc=0.5:  wd_eff ≈ 1.00  (barely above base)
          - acc=0.85: wd_eff ≈ 1.54
          - acc=0.90: wd_eff ≈ 3.00  (midpoint of ramp)
          - acc=0.95: wd_eff ≈ 4.46
          - acc=1.00: wd_eff ≈ 4.97  (near max of 5.0)
        """
        acc = self._cached_train_acc
        sigmoid_val = 1.0 / (1.0 + math.exp(-self.wd_scale * (acc - self.wd_thresh)))
        return base_wd * (1.0 + self.wd_ramp * sigmoid_val)

    # ══════════════════════════════════════════════════════════════════
    #  Sharpness helper
    # ══════════════════════════════════════════════════════════════════

    def _get_sharpness(self, p: torch.nn.Parameter) -> torch.Tensor:
        """Return cached sharpness signal for parameter, or zeros."""
        pidx = self._param_index.get(p, -1)
        if pidx in self._sharpness_cache:
            return self._sharpness_cache[pidx]
        return torch.zeros_like(p.data)

    # ══════════════════════════════════════════════════════════════════
    #  Main optimiser step
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
    ):
        """Perform a single optimization step.

        Args:
            closure: Re-evaluates loss (standard PyTorch).
            train_loss: Current training loss.
            val_loss: Current validation loss.
            train_acc: Current training accuracy in [0, 1].
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        # Cache train_acc for progressive weight decay
        if train_acc is not None:
            self._cached_train_acc = train_acc

        # ── Update alpha ──────────────────────────────────────────────
        if self._global_step % self.alpha_update_freq == 0 or self._global_step == 1:
            self._update_alpha(train_loss, val_loss, train_acc)
        elif train_acc is not None and train_acc >= self.zero_acc_threshold:
            self._update_alpha(train_loss, val_loss, train_acc)
        elif train_loss is not None and train_loss < self.zero_loss_threshold:
            self._update_alpha(train_loss, val_loss, train_acc)

        base_alpha = self._cached_alpha
        ramp = self._get_ramp_factor()

        # ── Gradient clipping ─────────────────────────────────────────
        if self.gradient_clipping > 0:
            all_params = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, self.gradient_clipping)

        # ── Per-parameter update ──────────────────────────────────────
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

                # ── Rule 1: Decoupled Memory ──────────────────────────
                mu.mul_(layer_alpha).add_(grad, alpha=1.0 - layer_alpha)

                # ── 2D Meta-net transformation (v1.5) ─────────────────
                sharpness = self._get_sharpness(p)
                smart_grad = self.meta_net(grad, sharpness)

                # ── Rule 2: Soft Sigmoid Gating ───────────────────────
                if ramp > 0 and mu.norm() > 1e-12 and smart_grad.norm() > 1e-12:
                    cos_sim = F.cosine_similarity(
                        smart_grad.flatten().unsqueeze(0),
                        mu.flatten().unsqueeze(0),
                    ).item()
                    gate = 1.0 / (1.0 + math.exp(-self.gate_temperature * cos_sim))
                    lamb_eff = ramp * gate * self.lamb
                    final_grad = smart_grad + lamb_eff * mu
                else:
                    final_grad = smart_grad

                # ── AdamW update ──────────────────────────────────────
                exp_avg.mul_(layer_beta1).add_(final_grad, alpha=1.0 - layer_beta1)
                exp_avg_sq.mul_(beta2).addcmul_(final_grad, final_grad, value=1.0 - beta2)

                bc1 = 1.0 - layer_beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]

                step_size = lr / bc1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)

                # Progressive weight decay (v1.5)
                p.data.mul_(1.0 - lr * wd_eff)
                # Adam step
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    # ══════════════════════════════════════════════════════════════════
    #  LookSAM + Bilevel Meta-Step  (v1.5 NEW)
    # ══════════════════════════════════════════════════════════════════

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
        """Combined LookSAM perturbation + bilevel meta-net update.

        Performs three operations in sequence:
          1. **LookSAM**: Perturb parameters in the worst-case direction,
             compute gradients at the perturbed point, cache the sharpness
             direction ``d = sam_grad − normal_grad`` for intermediate steps.
          2. **Bilevel**: Train the 2D meta-net to align its transformed
             gradients (conditioned on sharpness) with validation gradient.
          3. **Restore**: Put original parameters and training gradients back.

        **Call sequence** (in training loop)::

            # 1. Forward + backward on training data
            loss = criterion(model(train_x), train_y)
            optimizer.zero_grad()
            loss.backward()

            # 2. Combined SAM + bilevel (every sam_freq steps)
            if step % optimizer.sam_freq == 0:
                optimizer.sam_meta_step(model, train_x, train_y,
                                        val_x, val_y, criterion, meta_opt)

            # 3. Optimizer step
            optimizer.step(train_loss=loss.item(), train_acc=acc)

        Args:
            model: The model being trained.
            train_x: Training inputs (for SAM perturbation).
            train_y: Training targets.
            val_x: Validation inputs (for bilevel).
            val_y: Validation targets.
            criterion: Loss function.
            meta_optimizer: Optimizer for meta-net parameters.

        Returns:
            Tuple of (sam_loss, val_loss) for logging.
        """
        # ── Step 1: Save current training gradients ───────────────────
        train_grads: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                train_grads[name] = p.grad.detach().clone()

        if not train_grads:
            return 0.0, 0.0

        # ── Step 2: LookSAM — compute sharpness directions ───────────
        # Compute per-parameter ascent direction: ε_p = ρ · g_p / ‖g‖
        grad_norm = 0.0
        for tg in train_grads.values():
            grad_norm += tg.norm().item() ** 2
        grad_norm = math.sqrt(grad_norm) + 1e-12

        # Perturb parameters: θ → θ + ε
        param_backups: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            param_backups[name] = p.data.clone()
            if name in train_grads:
                eps_p = self.sam_rho * train_grads[name] / grad_norm
                p.data.add_(eps_p)

        # Forward + backward at perturbed point
        model.zero_grad()
        with torch.enable_grad():
            sam_logits = model(train_x)
            sam_loss = criterion(sam_logits, train_y)
            sam_loss.backward()

        # Cache sharpness direction: d[p] = sam_grad[p] − normal_grad[p]
        for name, p in model.named_parameters():
            if p.grad is not None and name in train_grads:
                d = p.grad.detach() - train_grads[name]
                pidx = self._param_index.get(p, -1)
                if pidx >= 0:
                    self._sharpness_cache[pidx] = d.abs()

        sam_loss_val = sam_loss.item()

        # Restore parameters: θ → θ − ε (back to original)
        for name, p in model.named_parameters():
            if name in param_backups:
                p.data.copy_(param_backups[name])

        # ── Step 3: Bilevel — train 2D meta-net ──────────────────────
        # Transform training grads through meta-net WITH sharpness
        # (with gradient tracking for backprop through meta-net)
        smart_grads: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in train_grads:
                pidx = self._param_index.get(p, -1)
                sharpness = self._sharpness_cache.get(pidx, torch.zeros_like(p.data))
                smart_grads[name] = self.meta_net(train_grads[name], sharpness)

        # Compute validation gradients
        model.zero_grad()
        with torch.enable_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y)
            val_loss.backward()

        # Meta-loss = −⟨smart_grad, val_grad_unit⟩
        meta_optimizer.zero_grad()

        device = val_x.device
        meta_loss = torch.tensor(0.0, device=device)
        for name, p in model.named_parameters():
            if name in smart_grads and p.grad is not None:
                vg = p.grad.detach()
                vg_norm = vg.norm()
                if vg_norm > 1e-12:
                    vg_unit = vg / vg_norm
                else:
                    vg_unit = vg
                meta_loss = meta_loss - (smart_grads[name] * vg_unit).sum()

        meta_loss.backward()
        meta_optimizer.step()

        # ── Step 4: Restore training gradients ────────────────────────
        for name, p in model.named_parameters():
            if name in train_grads:
                p.grad = train_grads[name]
            else:
                p.grad = None

        return sam_loss_val, val_loss.item()

    # ══════════════════════════════════════════════════════════════════
    #  Inspection helpers
    # ══════════════════════════════════════════════════════════════════

    def get_global_step(self) -> int:
        return self._global_step

    def get_cached_alpha(self) -> float:
        return self._cached_alpha

    def get_effective_wd(self) -> float:
        """Current effective weight decay (for logging)."""
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
            "avg_mu_norm": sum(mu_norms) / max(len(mu_norms), 1),
            "avg_exp_avg_norm": sum(ea_norms) / max(len(ea_norms), 1),
            "avg_sharpness_norm": sum(sharp_norms) / max(len(sharp_norms), 1),
            "sharpness_cached": len(self._sharpness_cache) > 0,
        }

    def __repr__(self) -> str:
        lines = [f"SuperGrok v1.5 ("]
        for group in self.param_groups:
            lines.append(f"  lr={group['lr']}, betas={group['betas']}, "
                         f"eps={group['eps']}, wd={group['weight_decay']}")
        lines.append(f"  alpha_init={self.alpha_init}, lamb={self.lamb}, "
                     f"gamma={self.gamma}, gamma_alpha={self.gamma_alpha}")
        lines.append(f"  kappa={self.kappa}, warmup={self.warmup_steps}+{self.warmup_ramp}, "
                     f"gate_temp={self.gate_temperature}")
        lines.append(f"  sam_rho={self.sam_rho}, sam_freq={self.sam_freq}")
        lines.append(f"  wd_ramp={self.wd_ramp}, wd_scale={self.wd_scale}, "
                     f"wd_thresh={self.wd_thresh}")
        lines.append(f"  meta_net={self.meta_net.__class__.__name__}")
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
        orchestrating loss.backward(), sam_meta_step(), and step().

        Usage:
            opt = SuperGrok15(model.parameters(), lr=1e-3)
            for batch_x, batch_y in dataloader:
                metrics = opt.step_full(model, batch_x, batch_y, val_x, val_y)
                print(f"loss={metrics['train_loss']:.4f}")

        Returns a dict with: train_loss, train_acc, val_loss (when computed),
        sam_loss (when SAM step runs).
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

        train_loss_val = loss.item()
        with torch.no_grad():
            train_acc = (logits.detach().argmax(-1) == train_y).float().mean().item()

        metrics = {"train_loss": train_loss_val, "train_acc": train_acc}

        # ── SAM + bilevel (automatic scheduling) ──────────────────────
        step_num = self._global_step + 1  # Will be incremented in step()
        if step_num % self.sam_freq == 0:
            try:
                sam_loss, val_loss = self.sam_meta_step(
                    model, train_x, train_y, val_x, val_y,
                    criterion, self._auto_meta_opt)
                metrics["sam_loss"] = sam_loss
            except Exception:
                pass  # SAM failure is non-fatal

        # ── Optimizer step with signals ───────────────────────────────
        kw: Dict[str, float] = {
            "train_loss": train_loss_val,
            "train_acc": train_acc,
        }
        alpha_freq = self.alpha_update_freq
        if step_num % alpha_freq == 0:
            with torch.no_grad():
                val_logits = model(val_x)
                val_loss_val = criterion(val_logits, val_y).item()
            kw["val_loss"] = val_loss_val
            metrics["val_loss"] = val_loss_val

        self.step(**kw)

        return metrics
