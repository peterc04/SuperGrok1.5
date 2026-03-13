"""
SuperGrok v2 — C++/CUDA Accelerated Grokking Optimizer with DeepSeek Sparse Attention

Same as SuperGrok v1.5 but replaces the MLP meta-net with a
DeepSeek-style Sparse Attention meta-net that captures cross-element
gradient correlations.

The lightning indexer selects top-k most relevant gradient elements
per query, enabling O(N*k) attention instead of O(N^2).
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Callable, Dict, Tuple

from grokking_optimizers import _ops


class SparseAttentionMetaNet(nn.Module):
    """DeepSeek Sparse Attention meta-net for gradient correction.

    Architecture:
        1. Project each (grad, sharpness) pair to Q, K, V
        2. Lightning indexer scores element pairs
        3. Top-k selection per query element
        4. Sparse attention computes cross-element correction
        5. Skip connection: smart_grad = grad + rescale * correction
    """

    def __init__(self, d_head: int = 16, n_idx_heads: int = 4,
                 top_k: int = 64, rescale_init: float = 0.0):
        super().__init__()
        self.d_head = d_head
        self.n_idx_heads = n_idx_heads
        self.top_k = top_k

        # Projection matrices: (grad, sharpness) -> Q, K, V
        self.W_q = nn.Linear(2, d_head, bias=True)
        self.W_k = nn.Linear(2, d_head, bias=True)
        self.W_v = nn.Linear(2, d_head, bias=True)

        # Lightning indexer projections
        self.W_iq = nn.Linear(2, n_idx_heads, bias=False)
        self.W_ik = nn.Linear(2, n_idx_heads, bias=False)
        self.w_idx = nn.Parameter(torch.ones(n_idx_heads) / n_idx_heads)

        # Output projection: d_head -> 1 (scalar correction)
        self.W_out = nn.Linear(d_head, 1, bias=True)

        # Skip connection scale
        self.rescale = nn.Parameter(torch.tensor(rescale_init))

        # Initialize small
        for m in [self.W_q, self.W_k, self.W_v, self.W_iq, self.W_ik, self.W_out]:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 0, 0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, grad: torch.Tensor, sharpness: torch.Tensor) -> torch.Tensor:
        """Forward pass (Python path for bilevel backward)."""
        if grad.numel() == 0:
            return grad
        shape = grad.shape
        N = grad.numel()
        eff_k = min(self.top_k, N)

        flat_g = grad.reshape(-1, 1)
        flat_s = sharpness.reshape(-1, 1)
        inp = torch.cat([flat_g, flat_s], dim=1)  # [N, 2]

        # Project to Q, K, V
        q = self.W_q(inp)   # [N, d_head]
        k = self.W_k(inp)   # [N, d_head]
        v = self.W_v(inp)   # [N, d_head]

        # Lightning indexer
        idx_q = self.W_iq(inp)  # [N, n_idx_heads]
        idx_k = self.W_ik(inp)  # [N, n_idx_heads]

        # Index scores: I[i,j] = sum_h(w_h * ReLU(idx_q[i,h] * idx_k[j,h]))
        # For each query, compute scores against all keys
        # scores[i, j] = sum_h w_h * relu(idx_q[i,h] * idx_k[j,h])
        scores = torch.zeros(N, N, device=grad.device)
        for h in range(self.n_idx_heads):
            outer = idx_q[:, h:h+1] * idx_k[:, h:h+1].t()  # [N, N]
            scores += self.w_idx[h] * torch.relu(outer)

        # Top-k selection
        _, topk_indices = scores.topk(eff_k, dim=-1)  # [N, k]

        # Sparse attention
        scale = 1.0 / math.sqrt(self.d_head)
        # Gather selected keys and values
        k_selected = k[topk_indices]  # [N, k, d_head]
        v_selected = v[topk_indices]  # [N, k, d_head]

        # Attention weights
        attn_scores = torch.bmm(q.unsqueeze(1), k_selected.transpose(1, 2)).squeeze(1) * scale  # [N, k]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [N, k]

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), v_selected).squeeze(1)  # [N, d_head]

        # Project to scalar correction
        correction = self.W_out(context)  # [N, 1]

        return (flat_g + self.rescale * correction).reshape(shape)

    def get_weights(self):
        """Extract weight tensors for C++ CUDA kernels."""
        return (
            self.W_q.weight.data.contiguous(),
            self.W_q.bias.data.contiguous(),
            self.W_k.weight.data.contiguous(),
            self.W_k.bias.data.contiguous(),
            self.W_v.weight.data.contiguous(),
            self.W_v.bias.data.contiguous(),
            self.W_iq.weight.data.contiguous(),
            self.W_ik.weight.data.contiguous(),
            self.w_idx.data.contiguous(),
            self.W_out.weight.data.contiguous(),
            self.W_out.bias.data.contiguous(),
            self.rescale.data.item(),
        )


class SuperGrok2(Optimizer):
    r"""SuperGrok v2 — C++/CUDA Grokking Optimizer with Sparse Attention.

    Same dynamics as SuperGrok v1.5 (sigmoid gating, adaptive SAM/bilevel,
    progressive WD) but with a DeepSeek Sparse Attention meta-net that
    captures cross-element gradient correlations.
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
        d_head: int = 16,
        n_idx_heads: int = 4,
        top_k: int = 64,
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
        self.d_head = d_head
        self.n_idx_heads = n_idx_heads
        self.top_k = top_k

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
            self.meta_net = SparseAttentionMetaNet(d_head, n_idx_heads, top_k)
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

        # Build flat parameter lists
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

    def _get_effective_wd(self, base_wd):
        return base_wd * (1.0 + self.wd_ramp * self._sigmoid(self.wd_scale, self._cached_train_acc, self.wd_thresh))

    def _get_gate_signal(self):
        return self._sigmoid(self.gate_scale, self._cached_train_acc, self.gate_thresh)

    def _get_effective_sam_freq(self):
        sam_heat = self._sigmoid(self.sam_scale, self._cached_train_acc, self.sam_thresh)
        return max(1, round(self.sam_freq_max - (self.sam_freq_max - self.sam_freq_min) * sam_heat))

    def _get_effective_bilevel_freq(self):
        bilevel_heat = self._sigmoid(self.bilevel_scale, self._cached_train_acc, self.bilevel_thresh)
        return max(1, round(self.bilevel_freq_max - (self.bilevel_freq_max - self.bilevel_freq_min) * bilevel_heat))

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
        wd_eff = self._get_effective_wd(group["weight_decay"])
        gate_signal = self._get_gate_signal()

        grads = []
        for p in self._flat_params:
            grads.append(p.grad.data if p.grad is not None else torch.Tensor())

        if self._weights_dirty:
            self._cached_weights = self.meta_net.get_weights()
            self._weights_dirty = False
        W_q, b_q, W_k, b_k, W_v, b_v, W_iq, W_ik, w_idx, W_out, b_out, rescale = self._cached_weights

        _ops.supergrok2_fused_step(
            self._flat_param_data,
            grads,
            self._flat_exp_avgs,
            self._flat_exp_avg_sqs,
            self._flat_mus,
            self._flat_sharpness,
            self._flat_steps,
            layer_alphas,
            self._flat_layer_beta1s,
            W_q, b_q, W_k, b_k, W_v, b_v,
            W_iq, W_ik, w_idx, W_out, b_out,
            rescale, self.d_head, self.n_idx_heads, self.top_k,
            beta2, lr, wd_eff, eps,
            self.lamb, ramp, gate_signal,
            self.gradient_clipping,
        )

        return loss

    def sam_step(self, model, train_x, train_y, criterion):
        """SAM perturbation + sharpness computation."""
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

        # Reuse v1.5 SAM kernels
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
        """Bilevel meta-net training using cached sharpness."""
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
        """Combined SAM + bilevel."""
        sam_loss = self.sam_step(model, train_x, train_y, criterion)
        val_loss = self.bilevel_step(model, train_x, train_y, val_x, val_y, criterion, meta_optimizer)
        return sam_loss, val_loss

    def get_global_step(self):
        return self._global_step

    def step_full(self, model, train_x, train_y, val_x, val_y, criterion=None):
        """Complete training step with adaptive SAM/bilevel scheduling."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if not hasattr(self, '_auto_meta_opt'):
            self._auto_meta_opt = torch.optim.Adam(self.meta_net.parameters(), lr=1e-4)

        model.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()

        metrics = {}
        step_num = self._global_step + 1

        if step_num % self._get_effective_sam_freq() == 0:
            try:
                metrics["sam_loss"] = self.sam_step(model, train_x, train_y, criterion)
            except Exception:
                pass

        if step_num % self._get_effective_bilevel_freq() == 0:
            try:
                metrics["val_loss"] = self.bilevel_step(
                    model, train_x, train_y, val_x, val_y, criterion, self._auto_meta_opt)
            except Exception:
                pass

        kw = {}
        if (step_num % self.alpha_update_freq == 0) or step_num == 1:
            with torch.no_grad():
                kw["train_loss"] = loss.item()
                kw["train_acc"] = (logits.detach().argmax(-1) == train_y).float().mean().item()
            metrics.update(kw)

        self.step(**kw)
        return metrics
