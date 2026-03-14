"""
SuperGrok v2 — Mamba-3 + 4-Head PEER + GRU Meta-Net Grokking Optimizer

Replaces the old ISAB+PEER+Recurrent architecture with Mamba-3 based meta-net
that captures cross-element gradient correlations via bidirectional selective
state space scans.

Key features:
  - Mamba-3 bidirectional scan (sorted by |gradient| magnitude)
  - 4-Head PEER product-key expert routing (144 experts, 4 active per element)
  - Per-element GRU for temporal gradient memory
  - Dynamic expert recycling (dead experts cloned from top performer)
  - All adaptive scheduling from v1.5 (sigmoid SAM/bilevel/WD, alpha updates)
  - functional_call SAM (no parameter modification)
  - Python fallback path (CUDA kernels in Phase C)
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Dict

from grokking_optimizers.mamba3_peer_metanet import Mamba3PEERMetaNet

try:
    from grokking_optimizers import _ops
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False


class SuperGrok2(Optimizer):
    r"""SuperGrok v2 — Mamba-3+PEER Grokking Optimizer.

    Same dynamics as SuperGrok v1.5 (sigmoid gating, adaptive SAM/bilevel,
    progressive WD) but with a Mamba-3+PEER meta-net that captures
    cross-element gradient correlations via bidirectional selective scans.
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
        d_model: int = 8,
        d_state: int = 16,
        mamba_expand: int = 2,
        num_peer_heads: int = 4,
        num_experts: int = 144,
        expert_hidden: int = 16,
        gru_hidden: int = 4,
        meta_rescale: float = 0.1,
        recycle_interval: int = 100,
        recycle_threshold: float = 0.001,
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
        sam_enable_threshold: float = 0.0,
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
        self.alpha_update_freq = alpha_update_freq
        self.zero_loss_threshold = zero_loss_threshold
        self.zero_acc_threshold = zero_acc_threshold
        self.sam_rho = sam_rho

        # Meta-net hyperparams
        self.d_model = d_model
        self.d_state = d_state
        self.num_experts = num_experts
        self.expert_hidden = expert_hidden
        self.gru_hidden = gru_hidden
        self.meta_rescale = meta_rescale

        # Adaptive scheduling params
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

        # Meta-net: Mamba-3 + 4-Head PEER + GRU
        if meta_net is None:
            self.meta_net = Mamba3PEERMetaNet(
                d_model=d_model,
                d_state=d_state,
                mamba_expand=mamba_expand,
                num_peer_heads=num_peer_heads,
                num_experts=num_experts,
                expert_hidden=expert_hidden,
                gru_hidden=gru_hidden,
                rescale=meta_rescale,
                recycle_interval=recycle_interval,
                recycle_threshold=recycle_threshold,
            )
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
        self._flat_gru_states = []
        self._flat_mamba_fwd_states = []
        self._flat_mamba_bwd_states = []
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
            self._flat_exp_avgs.append(
                torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
            self._flat_exp_avg_sqs.append(
                torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
            self._flat_mus.append(
                torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
            self._flat_sharpness.append(
                torch.zeros(p.data.shape, dtype=torch.float32, device=p.device))
            # Per-parameter GRU state
            self._flat_gru_states.append(
                torch.zeros(p.data.numel(), self.gru_hidden,
                            dtype=torch.float32, device=p.device))
            # Per-parameter Mamba states (initialized on first use)
            self._flat_mamba_fwd_states.append(None)
            self._flat_mamba_bwd_states.append(None)
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
        if self._cached_train_acc < self.sam_enable_threshold:
            return 999999  # effectively disabled
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
        gate_signal = self._get_gate_signal()

        group = self.param_groups[0]
        lr = group["lr"]
        beta2 = group["betas"][1]
        eps = group["eps"]
        wd_eff = self._get_effective_wd(group["weight_decay"])

        # Per-parameter pre-processing: clip grads, compute per-param scalars, init states
        lamb_eff = self.lamb * ramp * gate_signal
        active_indices = []
        clipped_grads = []
        alpha_mus_list = []
        lamb_effs_list = []
        beta1s_list = []
        bc1s_list = []
        bc2s_list = []

        for i, p in enumerate(self._flat_params):
            if p.grad is None:
                continue

            self._flat_steps[i] += 1
            grad = p.grad.data

            # Gradient clipping (per-parameter) + NaN guard
            grad_norm = grad.norm()
            if grad_norm > self.gradient_clipping:
                grad = grad * (self.gradient_clipping / (grad_norm + 1e-12))
            if not torch.isfinite(grad).all():
                grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))

            alpha_i = max(0.0, min(1.0, base_alpha * self._flat_layer_alphas[i]))
            beta1_i = self._flat_layer_beta1s[i]
            step_i = self._flat_steps[i]
            bc1 = 1.0 - beta1_i ** step_i
            bc2 = 1.0 - beta2 ** step_i

            # Initialize Mamba states if needed
            if self._flat_mamba_fwd_states[i] is None:
                d_inner = self.meta_net.mamba_fwd.d_inner
                d_state = self.meta_net.d_state
                self._flat_mamba_fwd_states[i] = torch.zeros(
                    d_inner, d_state, dtype=torch.float32, device=p.device)
                self._flat_mamba_bwd_states[i] = torch.zeros(
                    d_inner, d_state, dtype=torch.float32, device=p.device)

            active_indices.append(i)
            clipped_grads.append(grad)
            alpha_mus_list.append(float(alpha_i))
            lamb_effs_list.append(float(lamb_eff))
            beta1s_list.append(float(beta1_i))
            bc1s_list.append(float(bc1))
            bc2s_list.append(float(bc2))

        if not active_indices:
            return loss

        # Check if we can use CUDA batched path
        use_cuda = _HAS_CUDA and self._flat_params[active_indices[0]].is_cuda

        if use_cuda:
            # Ensure weights are extracted and cached
            if self._weights_dirty:
                w = self.meta_net.get_weights()
                self._cached_peer_query_Ws = torch.stack(
                    [q.weight.data.float().contiguous() for q in self.meta_net.peer_queries])
                self._cached_prod_keys_A = torch.stack(
                    [k.data.float().contiguous() for k in self.meta_net.product_keys_A])
                self._cached_prod_keys_B = torch.stack(
                    [k.data.float().contiguous() for k in self.meta_net.product_keys_B])
                self._cached_weights = w
                self._weights_dirty = False
            w = self._cached_weights

            _ops.supergrok2_mamba_peer_batched_step(
                [self._flat_params[i].data for i in active_indices],
                clipped_grads,
                [self._flat_sharpness[i] for i in active_indices],
                [self._flat_exp_avgs[i] for i in active_indices],
                [self._flat_exp_avg_sqs[i] for i in active_indices],
                [self._flat_mus[i] for i in active_indices],
                [self._flat_gru_states[i] for i in active_indices],
                [self._flat_mamba_fwd_states[i] for i in active_indices],
                [self._flat_mamba_bwd_states[i] for i in active_indices],
                # Input proj
                w['input_proj_W'], w['input_proj_b'],
                # Mamba forward
                w['mamba_fwd_in_proj'], w['mamba_fwd_dt_proj_W'],
                w['mamba_fwd_dt_proj_b'], w['mamba_fwd_B_proj'],
                w['mamba_fwd_C_proj'], w['mamba_fwd_A_log'],
                w['mamba_fwd_D'], w['mamba_fwd_rope_freq'],
                w['mamba_fwd_out_proj'],
                # Mamba backward
                w['mamba_bwd_in_proj'], w['mamba_bwd_dt_proj_W'],
                w['mamba_bwd_dt_proj_b'], w['mamba_bwd_B_proj'],
                w['mamba_bwd_C_proj'], w['mamba_bwd_A_log'],
                w['mamba_bwd_D'], w['mamba_bwd_rope_freq'],
                w['mamba_bwd_out_proj'],
                # GRU
                w['gru_W_z'], w['gru_b_z'],
                w['gru_W_r'], w['gru_b_r'],
                w['gru_W_h'], w['gru_b_h'],
                # PEER (stacked)
                self._cached_peer_query_Ws,
                self._cached_prod_keys_A,
                self._cached_prod_keys_B,
                # Experts
                w['expert_W1'].reshape(self.num_experts, -1),
                w['expert_b1'],
                w['expert_W2'].reshape(self.num_experts, -1),
                w['expert_b2'].reshape(-1),
                # Per-param scalars
                alpha_mus_list, lamb_effs_list,
                beta1s_list, bc1s_list, bc2s_list,
                # Shared scalars
                float(self.meta_net.rescale),
                float(beta2), float(lr), float(wd_eff), float(eps),
                # Dims
                self.meta_net.d_model, self.meta_net.d_state,
                self.meta_net.mamba_fwd.d_inner,
                self.meta_net.gru_hidden, self.meta_net.num_peer_heads,
                self.meta_net.pk_dim, self.meta_net.expert_hidden,
                self.meta_net.num_experts,
                self.meta_net.expert_counts,
            )
            # Expert recycling: increment step counter and periodically recycle
            self.meta_net.step_counter += 1
            if (self.meta_net.recycle_interval > 0 and
                    self.meta_net.step_counter % self.meta_net.recycle_interval == 0):
                self.meta_net._recycle_dead_experts()
        else:
            # Python fallback — per-parameter
            for idx, i in enumerate(active_indices):
                p = self._flat_params[i]
                grad = clipped_grads[idx]
                alpha_i = alpha_mus_list[idx]
                beta1_i = beta1s_list[idx]
                bc1 = bc1s_list[idx]
                bc2 = bc2s_list[idx]

                flat_grad = grad.reshape(-1)
                flat_sharp = self._flat_sharpness[i].reshape(-1)

                smart_grad, new_gru, new_fwd, new_bwd = self.meta_net(
                    flat_grad, flat_sharp,
                    self._flat_gru_states[i],
                    self._flat_mamba_fwd_states[i],
                    self._flat_mamba_bwd_states[i])
                self._flat_gru_states[i] = new_gru.detach()
                self._flat_mamba_fwd_states[i] = new_fwd.detach()
                self._flat_mamba_bwd_states[i] = new_bwd.detach()

                mu = self._flat_mus[i]
                mu.mul_(alpha_i).add_(grad, alpha=1.0 - alpha_i)
                effective_grad = smart_grad.reshape(grad.shape) + lamb_effs_list[idx] * mu
                self._flat_mus[i] = mu

                fg = effective_grad.reshape(-1).float()
                ea = self._flat_exp_avgs[i]
                easq = self._flat_exp_avg_sqs[i]
                ea.mul_(beta1_i).add_(fg, alpha=1 - beta1_i)
                easq.mul_(beta2).addcmul_(fg, fg, value=1 - beta2)
                step_size = lr / bc1
                denom = (easq / bc2).sqrt().add_(eps)
                p.data.mul_(1 - lr * wd_eff)
                p.data.addcdiv_(ea.reshape(p.data.shape), denom.reshape(p.data.shape), value=-step_size)

            # Expert recycling for Python fallback (once per step, not per param)
            self.meta_net.step_counter += 1
            if (self.meta_net.recycle_interval > 0 and
                    self.meta_net.step_counter % self.meta_net.recycle_interval == 0):
                self.meta_net._recycle_dead_experts()

        return loss

    def sam_step(self, model, train_x, train_y, criterion):
        """SAM perturbation + sharpness computation via functional_call (no param modification)."""
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
        perturbed_params = {}
        for name, p in named_params.items():
            if name in train_grads:
                perturbed_params[name] = p.detach() + rho_over_norm * train_grads[name]
            else:
                perturbed_params[name] = p.detach()

        model.zero_grad()
        with torch.enable_grad():
            sam_logits = torch.func.functional_call(model, perturbed_params, (train_x,))
            sam_loss = criterion(sam_logits, train_y)
            sam_loss.backward()
        sam_loss_val = sam_loss.item()

        for name, p in model.named_parameters():
            pidx = self._param_to_idx.get(id(p))
            if pidx is not None and p.grad is not None and name in train_grads:
                sam_grad = p.grad.detach()
                normal_grad = train_grads[name]
                self._flat_sharpness[pidx] = (sam_grad - normal_grad).abs()

        for name, p in model.named_parameters():
            p.grad = train_grads.get(name)

        return sam_loss_val

    def bilevel_step(self, model, train_x, train_y, val_x, val_y, criterion, meta_optimizer):
        """Bilevel meta-net training. Uses forward_for_bilevel (soft routing) for gradient flow."""
        self._ensure_state()
        named_params = list(model.named_parameters())

        saved_grads = {}
        for name, p in named_params:
            if p.grad is not None:
                saved_grads[name] = p.grad.detach().clone()

        # Per-parameter meta-net forward with soft routing for gradient flow
        smart_grads = {}
        for name, p in named_params:
            if name in saved_grads:
                pidx = self._param_to_idx.get(id(p))
                if pidx is None:
                    continue
                sg, _, _, _ = self.meta_net.forward_for_bilevel_cuda(
                    saved_grads[name].reshape(-1),
                    self._flat_sharpness[pidx].reshape(-1),
                    self._flat_gru_states[pidx],
                    self._flat_mamba_fwd_states[pidx],
                    self._flat_mamba_bwd_states[pidx])
                smart_grads[name] = sg.reshape(saved_grads[name].shape)

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
