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

Performance:
  - Blelloch parallel prefix scan for N >= 256 (O(N/P + log N) vs O(N))
  - Expert weights in shared memory (eliminates global reads during PEER eval)
  - Pre-allocated BilevelWorkspace (eliminates per-step temporary allocations)
  - ATen GEMM for bilevel precompute projections (N >= 1024)
  - Gradient checkpointing for bilevel (bilevel_checkpoint_interval parameter)
  - Shared-memory reduction for backward expert gradients (256x fewer atomics)
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
        bilevel_checkpoint_interval: int = 1,
        projection_precision: str = 'auto',
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

        # Precision configuration for projection GEMMs
        from .quantization import PrecisionConfig
        self.precision_config = PrecisionConfig(
            projection_precision=projection_precision)

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
        self.bilevel_checkpoint_interval = max(1, bilevel_checkpoint_interval)

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
        self._step_counter = 0  # Python int for expert recycling (avoids GPU sync)
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
                torch.zeros(p.data.numel(), dtype=torch.float32, device=p.device))
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

            # Cast sharpness to match grad dtype (FP32 sharpness + FP16 grad would
            # cause the CUDA kernel to reinterpret FP32 bits as FP16)
            sharpness_list = [
                self._flat_sharpness[i].to(clipped_grads[k].dtype)
                for k, i in enumerate(active_indices)
            ]
            _ops.supergrok2_mamba_peer_batched_step(
                [self._flat_params[i].data for i in active_indices],
                clipped_grads,
                sharpness_list,
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
            self._step_counter += 1
            if (self.meta_net.recycle_interval > 0 and
                    self._step_counter % self.meta_net.recycle_interval == 0):
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
                mu.mul_(alpha_i).add_(grad.reshape(-1), alpha=1.0 - alpha_i)
                effective_grad = smart_grad.reshape(-1) + lamb_effs_list[idx] * mu
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
            self._step_counter += 1
            if (self.meta_net.recycle_interval > 0 and
                    self._step_counter % self.meta_net.recycle_interval == 0):
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
        """Bilevel meta-net training.

        CUDA path: uses _ops.supergrok2_bilevel_fwd_save for fast scan forward
        and _ops.supergrok2_bilevel_backward for full backward through meta-net.
        Python fallback: uses forward_for_bilevel with autograd.
        """
        self._ensure_state()
        named_params = list(model.named_parameters())

        # 1. Save training gradients
        saved_grads = {}
        for name, p in named_params:
            if p.grad is not None:
                saved_grads[name] = p.grad.detach().clone()

        # Decide whether to use CUDA bilevel path
        use_cuda = (
            _HAS_CUDA
            and hasattr(_ops, 'supergrok2_bilevel_fwd_save')
            and hasattr(_ops, 'supergrok2_bilevel_backward')
            and any(p.is_cuda for _, p in named_params if p.grad is not None)
        )

        if not use_cuda:
            return self._bilevel_step_python(
                model, named_params, saved_grads,
                val_x, val_y, criterion, meta_optimizer,
            )

        # ═══════════════════════════════════════════════════════════════
        #  CUDA BILEVEL PATH
        # ═══════════════════════════════════════════════════════════════
        device = next(p for _, p in named_params if p.grad is not None).device
        mn = self.meta_net
        w = mn.get_weights()

        d_model = mn.d_model
        d_state = mn.d_state
        d_inner = mn.mamba_fwd.d_inner
        gru_hidden = mn.gru_hidden
        gru_input_dim = 2 + 2 * d_model  # [g, s, fwd_ctx, bwd_ctx]
        num_heads = mn.num_peer_heads
        pk_dim = mn.pk_dim
        expert_hidden = mn.expert_hidden
        num_experts = mn.num_experts
        peer_input_dim = gru_hidden + 2 * d_model + 2  # [h, fwd_ctx, bwd_ctx, g, s]
        topk = 4
        num_active = topk * topk
        rescale = float(mn.rescale)

        # Stack PEER weights
        peer_query_Ws = torch.stack(
            [q.weight.data.float().contiguous() for q in mn.peer_queries])
        prod_keys_A = torch.stack(
            [k.data.float().contiguous() for k in mn.product_keys_A])
        prod_keys_B = torch.stack(
            [k.data.float().contiguous() for k in mn.product_keys_B])

        # GRU weights
        gru_Wz = mn.gru.W_z.weight.data.float().contiguous()
        gru_bz = mn.gru.W_z.bias.data.float().contiguous()
        gru_Wr = mn.gru.W_r.weight.data.float().contiguous()
        gru_br = mn.gru.W_r.bias.data.float().contiguous()
        gru_Wh = mn.gru.W_h.weight.data.float().contiguous()
        gru_bh = mn.gru.W_h.bias.data.float().contiguous()

        # Expert weights (flattened for CUDA kernel)
        expert_W1_flat = w['expert_W1'].reshape(num_experts, expert_hidden)
        expert_b1 = w['expert_b1']  # [E, H]
        expert_W2_flat = w['expert_W2'].reshape(num_experts, expert_hidden)
        expert_b2_flat = w['expert_b2'].reshape(num_experts)

        half_d = d_model // 2

        # Pre-allocate gradient accumulators (zeroed, accumulated across params)
        d_mamba_fwd_in_proj = torch.zeros_like(w['mamba_fwd_in_proj'])
        d_mamba_fwd_dt_W = torch.zeros_like(w['mamba_fwd_dt_proj_W'])
        d_mamba_fwd_dt_b = torch.zeros_like(w['mamba_fwd_dt_proj_b'])
        d_mamba_fwd_B_proj = torch.zeros_like(w['mamba_fwd_B_proj'])
        d_mamba_fwd_C_proj = torch.zeros_like(w['mamba_fwd_C_proj'])
        d_mamba_fwd_A_log = torch.zeros_like(w['mamba_fwd_A_log'])
        d_mamba_fwd_D = torch.zeros_like(w['mamba_fwd_D'])
        d_mamba_fwd_rope = torch.zeros_like(w['mamba_fwd_rope_freq'])
        d_mamba_fwd_out_proj = torch.zeros_like(w['mamba_fwd_out_proj'])
        d_mamba_bwd_in_proj = torch.zeros_like(w['mamba_bwd_in_proj'])
        d_mamba_bwd_dt_W = torch.zeros_like(w['mamba_bwd_dt_proj_W'])
        d_mamba_bwd_dt_b = torch.zeros_like(w['mamba_bwd_dt_proj_b'])
        d_mamba_bwd_B_proj = torch.zeros_like(w['mamba_bwd_B_proj'])
        d_mamba_bwd_C_proj = torch.zeros_like(w['mamba_bwd_C_proj'])
        d_mamba_bwd_A_log = torch.zeros_like(w['mamba_bwd_A_log'])
        d_mamba_bwd_D = torch.zeros_like(w['mamba_bwd_D'])
        d_mamba_bwd_rope = torch.zeros_like(w['mamba_bwd_rope_freq'])
        d_mamba_bwd_out_proj = torch.zeros_like(w['mamba_bwd_out_proj'])
        gru_total_dim = gru_input_dim + gru_hidden
        d_gru_Wz = torch.zeros(gru_hidden, gru_total_dim, device=device)
        d_gru_bz = torch.zeros(gru_hidden, device=device)
        d_gru_Wr = torch.zeros(gru_hidden, gru_total_dim, device=device)
        d_gru_br = torch.zeros(gru_hidden, device=device)
        d_gru_Wh = torch.zeros(gru_hidden, gru_total_dim, device=device)
        d_gru_bh = torch.zeros(gru_hidden, device=device)
        d_peer_query_Ws = torch.zeros_like(peer_query_Ws)
        d_prod_keys_A = torch.zeros_like(prod_keys_A)
        d_prod_keys_B = torch.zeros_like(prod_keys_B)
        d_expert_W1 = torch.zeros(num_experts, expert_hidden, device=device)
        d_expert_b1 = torch.zeros(num_experts, expert_hidden, device=device)
        d_expert_W2 = torch.zeros(num_experts, expert_hidden, device=device)
        d_expert_b2 = torch.zeros(num_experts, device=device)
        d_input_proj_W = torch.zeros_like(w['input_proj_W'])
        d_input_proj_b = torch.zeros_like(w['input_proj_b'])

        # ── Collect active parameters for bilevel ──
        param_info = []  # (name, p, pidx, grad_flat, sharp_flat, N)
        for name, p in named_params:
            if name not in saved_grads:
                continue
            pidx = self._param_to_idx.get(id(p))
            if pidx is None:
                continue
            grad_flat = saved_grads[name].reshape(-1).float().contiguous()
            sharp_flat = self._flat_sharpness[pidx].reshape(-1).float().contiguous()
            N = grad_flat.numel()
            if N == 0:
                continue
            param_info.append((name, p, pidx, grad_flat, sharp_flat, N))

        num_bilevel_params = len(param_info)
        if num_bilevel_params == 0:
            return 0.0

        use_batched = hasattr(_ops, 'supergrok2_bilevel_fwd_save_batched')

        if use_batched:
            # ═════════════════════════════════════════════════════════
            #  BATCHED BILEVEL PATH — single kernel launch per phase
            # ═════════════════════════════════════════════════════════
            Ns = [info[5] for info in param_info]
            total_N = sum(Ns)
            offsets_list = [0]
            for n in Ns:
                offsets_list.append(offsets_list[-1] + n)

            # Stack persistent Mamba states [num_params, d_inner, d_state]
            fwd_init_states = torch.stack(
                [self._flat_mamba_fwd_states[info[2]].float().contiguous()
                 for info in param_info])
            bwd_init_states = torch.stack(
                [self._flat_mamba_bwd_states[info[2]].float().contiguous()
                 for info in param_info])

            # Pre-allocate packed output tensors
            ckpt_int = self.bilevel_checkpoint_interval
            fwd_scan_out_packed = torch.zeros(total_N, d_inner, device=device)
            bwd_scan_out_packed = torch.zeros(total_N, d_inner, device=device)
            if ckpt_int <= 1:
                ckpt_total_N = total_N
            else:
                ckpt_total_N = sum((n + ckpt_int - 1) // ckpt_int for n in Ns)
            fwd_saved_states_p = torch.zeros(ckpt_total_N, d_inner, d_state, device=device)
            fwd_saved_xb_p = torch.zeros(total_N, d_inner, device=device)
            fwd_saved_z_p = torch.zeros(total_N, d_inner, device=device)
            fwd_saved_dt_p = torch.zeros(total_N, d_inner, device=device)
            bwd_saved_states_p = torch.zeros(ckpt_total_N, d_inner, d_state, device=device)
            bwd_saved_xb_p = torch.zeros(total_N, d_inner, device=device)
            bwd_saved_z_p = torch.zeros(total_N, d_inner, device=device)
            bwd_saved_dt_p = torch.zeros(total_N, d_inner, device=device)
            x_sorted_packed = torch.zeros(total_N, d_model, device=device)
            offsets_t = torch.zeros(num_bilevel_params + 1, dtype=torch.int32, device=device)
            sort_indices_packed = torch.zeros(total_N, dtype=torch.int32, device=device)

            # 1a. Single batched forward-save call
            _ops.supergrok2_bilevel_fwd_save_batched(
                [info[3] for info in param_info],
                [info[4] for info in param_info],
                w['input_proj_W'], w['input_proj_b'],
                w['mamba_fwd_in_proj'], w['mamba_fwd_dt_proj_W'],
                w['mamba_fwd_dt_proj_b'], w['mamba_fwd_B_proj'],
                w['mamba_fwd_C_proj'], w['mamba_fwd_A_log'],
                w['mamba_fwd_D'], w['mamba_fwd_rope_freq'],
                w['mamba_fwd_out_proj'],
                w['mamba_bwd_in_proj'], w['mamba_bwd_dt_proj_W'],
                w['mamba_bwd_dt_proj_b'], w['mamba_bwd_B_proj'],
                w['mamba_bwd_C_proj'], w['mamba_bwd_A_log'],
                w['mamba_bwd_D'], w['mamba_bwd_rope_freq'],
                w['mamba_bwd_out_proj'],
                d_model, d_state, d_inner,
                fwd_scan_out_packed, bwd_scan_out_packed,
                fwd_saved_states_p, fwd_saved_xb_p,
                fwd_saved_z_p, fwd_saved_dt_p,
                bwd_saved_states_p, bwd_saved_xb_p,
                bwd_saved_z_p, bwd_saved_dt_p,
                x_sorted_packed, offsets_t,
                sort_indices_packed,
                fwd_init_states, bwd_init_states,
                ckpt_int,
            )

            # 1b. Per-parameter GRU + PEER forward (using packed scan outputs)
            smart_grads = {}
            per_param_fwd = {}

            for idx, (name, p, pidx, grad_flat, sharp_flat, N) in enumerate(param_info):
                start = offsets_list[idx]

                fwd_scan_out = fwd_scan_out_packed[start:start+N]
                bwd_scan_out = bwd_scan_out_packed[start:start+N]
                sort_indices = sort_indices_packed[start:start+N]

                fwd_ctx_sorted = fwd_scan_out @ w['mamba_fwd_out_proj'].T
                bwd_ctx_reversed = bwd_scan_out @ w['mamba_bwd_out_proj'].T
                bwd_ctx_sorted = bwd_ctx_reversed.flip(0)

                sort_idx_long = sort_indices.long()
                unsort_idx = sort_idx_long.argsort()
                fwd_ctx = fwd_ctx_sorted[unsort_idx]
                bwd_ctx = bwd_ctx_sorted[unsort_idx]

                g = grad_flat
                s = sharp_flat

                # GRU forward
                gru_inp = torch.cat([
                    g.unsqueeze(-1), s.unsqueeze(-1), fwd_ctx, bwd_ctx
                ], dim=-1)
                h_old = self._flat_gru_states[pidx].float()
                xh = torch.cat([gru_inp, h_old], dim=-1)
                z_gate = torch.sigmoid(xh @ gru_Wz.T + gru_bz)
                r_gate = torch.sigmoid(xh @ gru_Wr.T + gru_br)
                xrh = torch.cat([gru_inp, r_gate * h_old], dim=-1)
                h_tilde = torch.tanh(xrh @ gru_Wh.T + gru_bh)
                h_new = (1 - z_gate) * h_old + z_gate * h_tilde

                # PEER routing + expert MLP forward
                peer_inp = torch.cat([
                    h_new, fwd_ctx, bwd_ctx, g.unsqueeze(-1), s.unsqueeze(-1)
                ], dim=-1)

                all_expert_indices = torch.zeros(
                    N, num_heads, num_active, dtype=torch.int32, device=device)
                all_routing_weights = torch.zeros(
                    N, num_heads, num_active, device=device)
                all_z_hidden = torch.zeros(
                    N, num_heads, num_active, expert_hidden, device=device)
                total_expert_out = torch.zeros(N, device=device)

                for h in range(num_heads):
                    query = peer_inp @ peer_query_Ws[h].T
                    q_a = query[:, :half_d]
                    q_b = query[:, half_d:]
                    scores_a = q_a @ prod_keys_A[h].T
                    scores_b = q_b @ prod_keys_B[h].T
                    top_a_vals, top_a_idx = scores_a.topk(topk, dim=-1)
                    top_b_vals, top_b_idx = scores_b.topk(topk, dim=-1)
                    soft_a = torch.softmax(top_a_vals * 10.0, dim=-1)
                    soft_b = torch.softmax(top_b_vals * 10.0, dim=-1)
                    expert_idx = (
                        top_a_idx.unsqueeze(2) * pk_dim + top_b_idx.unsqueeze(1)
                    ).reshape(N, num_active)
                    routing_w = (
                        soft_a.unsqueeze(2) * soft_b.unsqueeze(1)
                    ).reshape(N, num_active)
                    all_expert_indices[:, h] = expert_idx.int()
                    all_routing_weights[:, h] = routing_w

                    ew1 = expert_W1_flat[expert_idx.long()]
                    eb1_sel = expert_b1[expert_idx.long()]
                    ew2 = expert_W2_flat[expert_idx.long()]
                    eb2_sel = expert_b2_flat[expert_idx.long()]
                    g_exp = g.unsqueeze(-1).unsqueeze(1).expand(-1, num_active, -1)
                    z_hidden = torch.relu(ew1 * g_exp + eb1_sel)
                    all_z_hidden[:, h] = z_hidden
                    out_k = (ew2 * z_hidden).sum(-1) + eb2_sel
                    head_out = (routing_w * out_k).sum(-1)
                    total_expert_out = total_expert_out + head_out / num_heads

                smart_grad = g + rescale * total_expert_out
                smart_grads[name] = smart_grad.reshape(saved_grads[name].shape)

                per_param_fwd[name] = {
                    'sort_indices': sort_indices, 'unsort_idx': unsort_idx,
                    'fwd_scan_out': fwd_scan_out, 'bwd_scan_out': bwd_scan_out,
                    'gru_input': gru_inp.contiguous(),
                    'gru_h_old': h_old.contiguous(),
                    'gru_z_gate': z_gate.contiguous(),
                    'gru_r_gate': r_gate.contiguous(),
                    'gru_h_tilde': h_tilde.contiguous(),
                    'peer_input': peer_inp.contiguous(),
                    'expert_indices': all_expert_indices.contiguous(),
                    'routing_weights': all_routing_weights.contiguous(),
                    'saved_z_hidden': all_z_hidden.contiguous(),
                    'grad_flat': grad_flat, 'sharp_flat': sharp_flat,
                }

            # 2. Compute validation gradients
            model.zero_grad()
            with torch.enable_grad():
                val_loss = criterion(model(val_x), val_y)
                val_loss.backward()

            # 3. Per-parameter backward through expert/PEER/GRU/out_proj
            #    → pack d_fwd/bwd_scan_out for batched scan backward
            d_fwd_scan_packed = torch.zeros(total_N, d_inner, device=device)
            d_bwd_scan_packed = torch.zeros(total_N, d_inner, device=device)

            for idx, (name, p, pidx, grad_flat, sharp_flat, N) in enumerate(param_info):
                if p.grad is None:
                    continue
                start = offsets_list[idx]
                sv = per_param_fwd[name]

                vg = p.grad.detach().reshape(-1).float()
                if not torch.isfinite(vg).all():
                    continue
                vg_norm = vg.norm()
                vg_unit = vg / vg_norm if vg_norm > 1e-12 else vg
                d_smart = -vg_unit

                # Expert/PEER backward
                d_expert_out = d_smart * rescale  # [N]
                d_peer_input = torch.zeros(N, peer_input_dim, device=device)

                for h in range(num_heads):
                    eidx = sv['expert_indices'][:, h].long()  # [N, K²]
                    rw = sv['routing_weights'][:, h]           # [N, K²]
                    zh = sv['saved_z_hidden'][:, h]            # [N, K², H]
                    ew2 = expert_W2_flat[eidx]                 # [N, K², H]
                    eb2_sel = expert_b2_flat[eidx]             # [N, K²]
                    ew1 = expert_W1_flat[eidx]                 # [N, K², H]

                    # Recompute out_k for routing gradient
                    out_k = (ew2 * zh).sum(-1) + eb2_sel       # [N, K²]

                    d_head = d_expert_out / num_heads          # [N]
                    d_out_k = d_head.unsqueeze(-1) * rw        # [N, K²]
                    d_rw = d_head.unsqueeze(-1) * out_k        # [N, K²]

                    # Expert MLP backward
                    d_eb2 = d_out_k
                    d_ew2_zh = d_out_k.unsqueeze(-1).expand_as(zh)
                    d_ew2 = d_ew2_zh * zh
                    d_zh = d_ew2_zh * ew2
                    relu_mask = (zh > 0).float()
                    d_pre_relu = d_zh * relu_mask
                    d_ew1 = d_pre_relu * grad_flat.unsqueeze(-1).unsqueeze(1).expand_as(d_pre_relu)
                    d_eb1 = d_pre_relu

                    # Accumulate expert weight gradients
                    eidx_flat = eidx.reshape(-1)
                    d_expert_W1.index_add_(0, eidx_flat, d_ew1.reshape(-1, expert_hidden))
                    d_expert_b1.index_add_(0, eidx_flat, d_eb1.reshape(-1, expert_hidden))
                    d_expert_W2.index_add_(0, eidx_flat, d_ew2.reshape(-1, expert_hidden))
                    d_expert_b2.index_add_(0, eidx_flat, d_eb2.reshape(-1))

                    # Routing backward → product-key softmax → query → peer_input
                    # Recompute soft_a, soft_b from scores
                    query = sv['peer_input'] @ peer_query_Ws[h].T
                    q_a = query[:, :half_d]
                    q_b = query[:, half_d:]
                    scores_a = q_a @ prod_keys_A[h].T
                    scores_b = q_b @ prod_keys_B[h].T
                    top_a_vals, top_a_idx = scores_a.topk(topk, dim=-1)
                    top_b_vals, top_b_idx = scores_b.topk(topk, dim=-1)
                    soft_a = torch.softmax(top_a_vals * 10.0, dim=-1)
                    soft_b = torch.softmax(top_b_vals * 10.0, dim=-1)

                    # routing_w = soft_a[:, :, None] * soft_b[:, None, :]
                    # d_soft_a = sum_j(d_rw[i,j] * soft_b[j])
                    # d_soft_b = sum_i(d_rw[i,j] * soft_a[i])
                    d_rw_reshaped = d_rw.reshape(N, topk, topk)
                    d_soft_a = (d_rw_reshaped * soft_b.unsqueeze(1)).sum(-1)
                    d_soft_b = (d_rw_reshaped * soft_a.unsqueeze(2)).sum(-2)

                    # softmax backward: d_pre = soft * (d - sum(soft*d))
                    sa_dot = (soft_a * d_soft_a).sum(-1, keepdim=True)
                    d_top_a = soft_a * (d_soft_a - sa_dot) * 10.0
                    sb_dot = (soft_b * d_soft_b).sum(-1, keepdim=True)
                    d_top_b = soft_b * (d_soft_b - sb_dot) * 10.0

                    # topk backward: scatter into full score gradients
                    d_scores_a = torch.zeros(N, pk_dim, device=device)
                    d_scores_b = torch.zeros(N, pk_dim, device=device)
                    d_scores_a.scatter_add_(1, top_a_idx, d_top_a)
                    d_scores_b.scatter_add_(1, top_b_idx, d_top_b)

                    # scores = q @ keys.T → d_q, d_keys
                    d_q_a = d_scores_a @ prod_keys_A[h]
                    d_q_b = d_scores_b @ prod_keys_B[h]
                    d_prod_keys_A[h].addmm_(d_scores_a.T, q_a)
                    d_prod_keys_B[h].addmm_(d_scores_b.T, q_b)

                    d_query = torch.cat([d_q_a, d_q_b], dim=-1)
                    d_peer_query_Ws[h].addmm_(d_query.T, sv['peer_input'])
                    d_peer_input.addmm_(d_query, peer_query_Ws[h])

                # GRU backward
                d_gru_out = d_peer_input[:, :gru_hidden]
                d_h_new = d_gru_out

                z_g = sv['gru_z_gate']
                r_g = sv['gru_r_gate']
                h_til = sv['gru_h_tilde']
                h_o = sv['gru_h_old']
                gru_in = sv['gru_input']
                xh = torch.cat([gru_in, h_o], dim=-1)

                d_z = d_h_new * (h_til - h_o)
                d_h_tilde = d_h_new * z_g
                d_h_old = d_h_new * (1 - z_g)

                d_pre_tanh = d_h_tilde * (1 - h_til ** 2)
                xrh = torch.cat([gru_in, r_g * h_o], dim=-1)
                d_Wh_contrib = d_pre_tanh.T @ xrh
                d_gru_Wh.add_(d_Wh_contrib)
                d_gru_bh.add_(d_pre_tanh.sum(0))
                d_xrh = d_pre_tanh @ gru_Wh
                d_gru_inp_1 = d_xrh[:, :gru_input_dim]
                d_r_h_old = d_xrh[:, gru_input_dim:]
                d_r = d_r_h_old * h_o
                d_h_old = d_h_old + d_r_h_old * r_g

                d_pre_r = d_r * r_g * (1 - r_g)
                d_gru_Wr.add_(d_pre_r.T @ xh)
                d_gru_br.add_(d_pre_r.sum(0))
                d_xh_r = d_pre_r @ gru_Wr

                d_pre_z = d_z * z_g * (1 - z_g)
                d_gru_Wz.add_(d_pre_z.T @ xh)
                d_gru_bz.add_(d_pre_z.sum(0))
                d_xh_z = d_pre_z @ gru_Wz

                d_xh_total = d_xh_r + d_xh_z
                d_gru_inp_2 = d_xh_total[:, :gru_input_dim]

                d_gru_input = d_gru_inp_1 + d_gru_inp_2

                # Combine d_fwd_ctx, d_bwd_ctx from GRU + PEER inputs
                d_fwd_ctx = (d_gru_input[:, 2:2+d_model]
                             + d_peer_input[:, gru_hidden:gru_hidden+d_model])
                d_bwd_ctx = (d_gru_input[:, 2+d_model:2+2*d_model]
                             + d_peer_input[:, gru_hidden+d_model:gru_hidden+2*d_model])

                # Re-sort for out_proj backward
                sort_idx_long = sv['sort_indices'].long()
                d_fwd_sorted = d_fwd_ctx.index_select(0, sort_idx_long)
                d_bwd_sorted = d_bwd_ctx.index_select(0, sort_idx_long).flip(0)

                # Out-proj backward: ctx = scan_out @ out_proj.T
                # d_scan_out = d_ctx @ out_proj; d_out_proj += scan_out.T @ d_ctx
                d_fwd_scan = d_fwd_sorted @ w['mamba_fwd_out_proj']
                d_mamba_fwd_out_proj.addmm_(sv['fwd_scan_out'].T, d_fwd_sorted)
                d_bwd_scan = d_bwd_sorted @ w['mamba_bwd_out_proj']
                d_mamba_bwd_out_proj.addmm_(sv['bwd_scan_out'].T, d_bwd_sorted)

                # Pack into batched tensors
                d_fwd_scan_packed[start:start+N] = d_fwd_scan
                d_bwd_scan_packed[start:start+N] = d_bwd_scan

            # 4. Single batched backward scan call
            d_x_sorted_packed = torch.zeros(total_N, d_model, device=device)
            _ops.supergrok2_bilevel_backward_batched(
                d_fwd_scan_packed, d_bwd_scan_packed,
                x_sorted_packed,
                fwd_saved_states_p, fwd_saved_xb_p,
                fwd_saved_z_p, fwd_saved_dt_p,
                bwd_saved_states_p, bwd_saved_xb_p,
                bwd_saved_z_p, bwd_saved_dt_p,
                offsets_t,
                w['mamba_fwd_in_proj'], w['mamba_fwd_dt_proj_W'],
                w['mamba_fwd_dt_proj_b'], w['mamba_fwd_B_proj'],
                w['mamba_fwd_C_proj'], w['mamba_fwd_A_log'],
                w['mamba_fwd_D'], w['mamba_fwd_rope_freq'],
                w['mamba_bwd_in_proj'], w['mamba_bwd_dt_proj_W'],
                w['mamba_bwd_dt_proj_b'], w['mamba_bwd_B_proj'],
                w['mamba_bwd_C_proj'], w['mamba_bwd_A_log'],
                w['mamba_bwd_D'], w['mamba_bwd_rope_freq'],
                d_mamba_fwd_in_proj, d_mamba_fwd_dt_W,
                d_mamba_fwd_dt_b, d_mamba_fwd_B_proj,
                d_mamba_fwd_C_proj, d_mamba_fwd_A_log,
                d_mamba_fwd_D, d_mamba_fwd_rope,
                d_mamba_bwd_in_proj, d_mamba_bwd_dt_W,
                d_mamba_bwd_dt_b, d_mamba_bwd_B_proj,
                d_mamba_bwd_C_proj, d_mamba_bwd_A_log,
                d_mamba_bwd_D, d_mamba_bwd_rope,
                d_x_sorted_packed,
                fwd_init_states, bwd_init_states,
                d_model, d_state, d_inner, num_bilevel_params,
                ckpt_int,
            )

            # 5. Per-parameter input_proj backward from d_x_sorted
            for idx, (name, p, pidx, grad_flat, sharp_flat, N) in enumerate(param_info):
                start = offsets_list[idx]
                sv = per_param_fwd.get(name)
                if sv is None:
                    continue

                d_x_sorted = d_x_sorted_packed[start:start+N]
                unsort_idx = sv['unsort_idx']
                d_x_unsorted = d_x_sorted.index_select(0, unsort_idx)

                # input_proj: x = [g, s] @ W.T + b → d_W += d_x.T @ [g, s], d_b += d_x.sum(0)
                gs = torch.stack([grad_flat, sharp_flat], dim=-1)  # [N, 2]
                d_input_proj_W.addmm_(d_x_unsorted.T, gs)
                d_input_proj_b.add_(d_x_unsorted.sum(0))

        else:
            # ═════════════════════════════════════════════════════════
            #  SERIAL BILEVEL FALLBACK — per-parameter kernel launches
            # ═════════════════════════════════════════════════════════
            smart_grads = {}
            per_param_saved = {}
            ckpt_int = self.bilevel_checkpoint_interval

            for idx, (name, p, pidx, grad_flat, sharp_flat, N) in enumerate(param_info):
                # Allocate scan output + saved buffers
                fwd_scan_out = torch.zeros(N, d_inner, device=device)
                bwd_scan_out = torch.zeros(N, d_inner, device=device)
                fwd_final = torch.zeros(d_inner, d_state, device=device)
                bwd_final = torch.zeros(d_inner, d_state, device=device)
                num_ckpts = (N + ckpt_int - 1) // ckpt_int if ckpt_int > 1 else N
                fwd_saved_states = torch.zeros(num_ckpts, d_inner, d_state, device=device)
                fwd_saved_xb = torch.zeros(N, d_inner, device=device)
                fwd_saved_z = torch.zeros(N, d_inner, device=device)
                fwd_saved_dt = torch.zeros(N, d_inner, device=device)
                bwd_saved_states = torch.zeros(num_ckpts, d_inner, d_state, device=device)
                bwd_saved_xb = torch.zeros(N, d_inner, device=device)
                bwd_saved_z = torch.zeros(N, d_inner, device=device)
                bwd_saved_dt = torch.zeros(N, d_inner, device=device)
                x_sorted = torch.zeros(N, d_model, device=device)
                sort_indices = torch.zeros(N, dtype=torch.int32, device=device)

                # Pass persistent Mamba states (not zeros)
                fwd_init = self._flat_mamba_fwd_states[pidx].float().contiguous()
                bwd_init = self._flat_mamba_bwd_states[pidx].float().contiguous()

                _ops.supergrok2_bilevel_fwd_save(
                    grad_flat, sharp_flat,
                    w['input_proj_W'], w['input_proj_b'],
                    w['mamba_fwd_in_proj'], w['mamba_fwd_dt_proj_W'],
                    w['mamba_fwd_dt_proj_b'], w['mamba_fwd_B_proj'],
                    w['mamba_fwd_C_proj'], w['mamba_fwd_A_log'],
                    w['mamba_fwd_D'], w['mamba_fwd_rope_freq'],
                    w['mamba_fwd_out_proj'],
                    w['mamba_bwd_in_proj'], w['mamba_bwd_dt_proj_W'],
                    w['mamba_bwd_dt_proj_b'], w['mamba_bwd_B_proj'],
                    w['mamba_bwd_C_proj'], w['mamba_bwd_A_log'],
                    w['mamba_bwd_D'], w['mamba_bwd_rope_freq'],
                    w['mamba_bwd_out_proj'],
                    d_model, d_state, d_inner,
                    fwd_scan_out, bwd_scan_out,
                    fwd_final, bwd_final,
                    fwd_saved_states, fwd_saved_xb, fwd_saved_z, fwd_saved_dt,
                    bwd_saved_states, bwd_saved_xb, bwd_saved_z, bwd_saved_dt,
                    x_sorted, sort_indices,
                    fwd_init, bwd_init,
                    ckpt_int,
                )

                fwd_ctx_sorted = fwd_scan_out @ w['mamba_fwd_out_proj'].T
                bwd_ctx_reversed = bwd_scan_out @ w['mamba_bwd_out_proj'].T
                bwd_ctx_sorted = bwd_ctx_reversed.flip(0)

                sort_idx_long = sort_indices.long()
                unsort_idx = sort_idx_long.argsort()
                fwd_ctx = fwd_ctx_sorted[unsort_idx]
                bwd_ctx = bwd_ctx_sorted[unsort_idx]

                g = grad_flat
                s = sharp_flat

                # GRU forward
                gru_inp = torch.cat([
                    g.unsqueeze(-1), s.unsqueeze(-1), fwd_ctx, bwd_ctx
                ], dim=-1)
                h_old = self._flat_gru_states[pidx].float()
                xh = torch.cat([gru_inp, h_old], dim=-1)
                z_gate = torch.sigmoid(xh @ gru_Wz.T + gru_bz)
                r_gate = torch.sigmoid(xh @ gru_Wr.T + gru_br)
                xrh = torch.cat([gru_inp, r_gate * h_old], dim=-1)
                h_tilde = torch.tanh(xrh @ gru_Wh.T + gru_bh)
                h_new = (1 - z_gate) * h_old + z_gate * h_tilde

                # PEER routing + expert MLP forward
                peer_inp = torch.cat([
                    h_new, fwd_ctx, bwd_ctx, g.unsqueeze(-1), s.unsqueeze(-1)
                ], dim=-1)

                all_expert_indices = torch.zeros(
                    N, num_heads, num_active, dtype=torch.int32, device=device)
                all_routing_weights = torch.zeros(
                    N, num_heads, num_active, device=device)
                all_z_hidden = torch.zeros(
                    N, num_heads, num_active, expert_hidden, device=device)
                all_scores_a = torch.zeros(N, num_heads, pk_dim, device=device)
                all_scores_b = torch.zeros(N, num_heads, pk_dim, device=device)
                all_top_a_idx = torch.zeros(
                    N, num_heads, topk, dtype=torch.int32, device=device)
                all_top_b_idx = torch.zeros(
                    N, num_heads, topk, dtype=torch.int32, device=device)
                all_soft_a = torch.zeros(N, num_heads, topk, device=device)
                all_soft_b = torch.zeros(N, num_heads, topk, device=device)
                total_expert_out = torch.zeros(N, device=device)

                for h in range(num_heads):
                    query = peer_inp @ peer_query_Ws[h].T
                    q_a = query[:, :half_d]
                    q_b = query[:, half_d:]
                    scores_a = q_a @ prod_keys_A[h].T
                    scores_b = q_b @ prod_keys_B[h].T
                    all_scores_a[:, h] = scores_a
                    all_scores_b[:, h] = scores_b
                    top_a_vals, top_a_idx = scores_a.topk(topk, dim=-1)
                    top_b_vals, top_b_idx = scores_b.topk(topk, dim=-1)
                    all_top_a_idx[:, h] = top_a_idx.int()
                    all_top_b_idx[:, h] = top_b_idx.int()
                    soft_a = torch.softmax(top_a_vals * 10.0, dim=-1)
                    soft_b = torch.softmax(top_b_vals * 10.0, dim=-1)
                    all_soft_a[:, h] = soft_a
                    all_soft_b[:, h] = soft_b
                    expert_idx = (
                        top_a_idx.unsqueeze(2) * pk_dim + top_b_idx.unsqueeze(1)
                    ).reshape(N, num_active)
                    routing_w = (
                        soft_a.unsqueeze(2) * soft_b.unsqueeze(1)
                    ).reshape(N, num_active)
                    all_expert_indices[:, h] = expert_idx.int()
                    all_routing_weights[:, h] = routing_w

                    ew1 = expert_W1_flat[expert_idx.long()]
                    eb1_sel = expert_b1[expert_idx.long()]
                    ew2 = expert_W2_flat[expert_idx.long()]
                    eb2_sel = expert_b2_flat[expert_idx.long()]
                    g_exp = g.unsqueeze(-1).unsqueeze(1).expand(-1, num_active, -1)
                    z_hidden = torch.relu(ew1 * g_exp + eb1_sel)
                    all_z_hidden[:, h] = z_hidden
                    out_k = (ew2 * z_hidden).sum(-1) + eb2_sel
                    head_out = (routing_w * out_k).sum(-1)
                    total_expert_out = total_expert_out + head_out / num_heads

                smart_grad = g + rescale * total_expert_out
                smart_grads[name] = smart_grad.reshape(saved_grads[name].shape)

                per_param_saved[name] = {
                    'sort_indices': sort_indices, 'x_sorted': x_sorted,
                    'fwd_scan_out': fwd_scan_out, 'bwd_scan_out': bwd_scan_out,
                    'fwd_saved_states': fwd_saved_states,
                    'fwd_saved_xb': fwd_saved_xb,
                    'fwd_saved_z': fwd_saved_z,
                    'fwd_saved_dt': fwd_saved_dt,
                    'bwd_saved_states': bwd_saved_states,
                    'bwd_saved_xb': bwd_saved_xb,
                    'bwd_saved_z': bwd_saved_z,
                    'bwd_saved_dt': bwd_saved_dt,
                    'gru_input': gru_inp.contiguous(),
                    'gru_h_old': h_old.contiguous(),
                    'gru_z_gate': z_gate.contiguous(),
                    'gru_r_gate': r_gate.contiguous(),
                    'gru_h_tilde': h_tilde.contiguous(),
                    'peer_input': peer_inp.contiguous(),
                    'expert_indices': all_expert_indices.contiguous(),
                    'routing_weights': all_routing_weights.contiguous(),
                    'saved_z_hidden': all_z_hidden.contiguous(),
                    'saved_scores_a': all_scores_a.contiguous(),
                    'saved_scores_b': all_scores_b.contiguous(),
                    'saved_top_a_idx': all_top_a_idx.contiguous(),
                    'saved_top_b_idx': all_top_b_idx.contiguous(),
                    'saved_soft_a': all_soft_a.contiguous(),
                    'saved_soft_b': all_soft_b.contiguous(),
                    'grad_flat': grad_flat, 'sharp_flat': sharp_flat,
                    'fwd_init': fwd_init, 'bwd_init': bwd_init,
                }

            # 2. Compute validation gradients
            model.zero_grad()
            with torch.enable_grad():
                val_loss = criterion(model(val_x), val_y)
                val_loss.backward()

            # 3-4. Per-parameter: d_smart_grad → CUDA backward → accumulate
            for name, p in named_params:
                if name not in per_param_saved or p.grad is None:
                    continue

                vg = p.grad.detach().reshape(-1).float()
                if not torch.isfinite(vg).all():
                    continue
                vg_norm = vg.norm()
                vg_unit = vg / vg_norm if vg_norm > 1e-12 else vg
                d_smart_grad = -vg_unit
                sv = per_param_saved[name]

                _ops.supergrok2_bilevel_backward(
                    d_smart_grad, sv['grad_flat'], sv['sharp_flat'], rescale,
                    sv['sort_indices'], sv['x_sorted'],
                    sv['fwd_scan_out'], sv['bwd_scan_out'],
                    sv['fwd_saved_states'], sv['fwd_saved_xb'],
                    sv['fwd_saved_z'], sv['fwd_saved_dt'],
                    sv['bwd_saved_states'], sv['bwd_saved_xb'],
                    sv['bwd_saved_z'], sv['bwd_saved_dt'],
                    sv['gru_input'], sv['gru_h_old'],
                    sv['gru_z_gate'], sv['gru_r_gate'], sv['gru_h_tilde'],
                    sv['peer_input'],
                    sv['expert_indices'], sv['routing_weights'],
                    sv['saved_z_hidden'],
                    sv['saved_scores_a'], sv['saved_scores_b'],
                    sv['saved_top_a_idx'], sv['saved_top_b_idx'],
                    sv['saved_soft_a'], sv['saved_soft_b'],
                    w['mamba_fwd_in_proj'], w['mamba_fwd_dt_proj_W'],
                    w['mamba_fwd_dt_proj_b'], w['mamba_fwd_B_proj'],
                    w['mamba_fwd_C_proj'], w['mamba_fwd_A_log'],
                    w['mamba_fwd_D'], w['mamba_fwd_rope_freq'],
                    w['mamba_fwd_out_proj'],
                    w['mamba_bwd_in_proj'], w['mamba_bwd_dt_proj_W'],
                    w['mamba_bwd_dt_proj_b'], w['mamba_bwd_B_proj'],
                    w['mamba_bwd_C_proj'], w['mamba_bwd_A_log'],
                    w['mamba_bwd_D'], w['mamba_bwd_rope_freq'],
                    w['mamba_bwd_out_proj'],
                    gru_Wz, gru_Wr, gru_Wh,
                    peer_query_Ws, prod_keys_A, prod_keys_B,
                    expert_W1_flat, expert_W2_flat,
                    w['expert_b1'], expert_b2_flat,
                    w['input_proj_W'],
                    sv['fwd_init'], sv['bwd_init'],
                    d_mamba_fwd_in_proj, d_mamba_fwd_dt_W,
                    d_mamba_fwd_dt_b, d_mamba_fwd_B_proj,
                    d_mamba_fwd_C_proj, d_mamba_fwd_A_log,
                    d_mamba_fwd_D, d_mamba_fwd_rope, d_mamba_fwd_out_proj,
                    d_mamba_bwd_in_proj, d_mamba_bwd_dt_W,
                    d_mamba_bwd_dt_b, d_mamba_bwd_B_proj,
                    d_mamba_bwd_C_proj, d_mamba_bwd_A_log,
                    d_mamba_bwd_D, d_mamba_bwd_rope, d_mamba_bwd_out_proj,
                    d_gru_Wz, d_gru_bz, d_gru_Wr, d_gru_br,
                    d_gru_Wh, d_gru_bh,
                    d_peer_query_Ws, d_prod_keys_A, d_prod_keys_B,
                    d_expert_W1, d_expert_b1, d_expert_W2, d_expert_b2,
                    d_input_proj_W, d_input_proj_b,
                    d_model, d_state, d_inner,
                    gru_hidden, gru_input_dim,
                    num_heads, topk, pk_dim,
                    expert_hidden, peer_input_dim, num_experts,
                    ckpt_int,
                )

        # 5. Map accumulated gradients to meta-net parameters and step
        meta_optimizer.zero_grad()
        # Mamba forward
        mn.mamba_fwd.in_proj.weight.grad = d_mamba_fwd_in_proj.to(
            mn.mamba_fwd.in_proj.weight.dtype)
        mn.mamba_fwd.dt_proj.weight.grad = d_mamba_fwd_dt_W.to(
            mn.mamba_fwd.dt_proj.weight.dtype)
        mn.mamba_fwd.dt_proj.bias.grad = d_mamba_fwd_dt_b.to(
            mn.mamba_fwd.dt_proj.bias.dtype)
        mn.mamba_fwd.B_proj.weight.grad = d_mamba_fwd_B_proj.to(
            mn.mamba_fwd.B_proj.weight.dtype)
        mn.mamba_fwd.C_proj.weight.grad = d_mamba_fwd_C_proj.to(
            mn.mamba_fwd.C_proj.weight.dtype)
        mn.mamba_fwd.A_log.grad = d_mamba_fwd_A_log.to(mn.mamba_fwd.A_log.dtype)
        mn.mamba_fwd.D.grad = d_mamba_fwd_D.to(mn.mamba_fwd.D.dtype)
        mn.mamba_fwd.rope_freq.grad = d_mamba_fwd_rope.to(
            mn.mamba_fwd.rope_freq.dtype)
        mn.mamba_fwd.out_proj.weight.grad = d_mamba_fwd_out_proj.to(
            mn.mamba_fwd.out_proj.weight.dtype)
        # Mamba backward
        mn.mamba_bwd.in_proj.weight.grad = d_mamba_bwd_in_proj.to(
            mn.mamba_bwd.in_proj.weight.dtype)
        mn.mamba_bwd.dt_proj.weight.grad = d_mamba_bwd_dt_W.to(
            mn.mamba_bwd.dt_proj.weight.dtype)
        mn.mamba_bwd.dt_proj.bias.grad = d_mamba_bwd_dt_b.to(
            mn.mamba_bwd.dt_proj.bias.dtype)
        mn.mamba_bwd.B_proj.weight.grad = d_mamba_bwd_B_proj.to(
            mn.mamba_bwd.B_proj.weight.dtype)
        mn.mamba_bwd.C_proj.weight.grad = d_mamba_bwd_C_proj.to(
            mn.mamba_bwd.C_proj.weight.dtype)
        mn.mamba_bwd.A_log.grad = d_mamba_bwd_A_log.to(mn.mamba_bwd.A_log.dtype)
        mn.mamba_bwd.D.grad = d_mamba_bwd_D.to(mn.mamba_bwd.D.dtype)
        mn.mamba_bwd.rope_freq.grad = d_mamba_bwd_rope.to(
            mn.mamba_bwd.rope_freq.dtype)
        mn.mamba_bwd.out_proj.weight.grad = d_mamba_bwd_out_proj.to(
            mn.mamba_bwd.out_proj.weight.dtype)
        # GRU
        mn.gru.W_z.weight.grad = d_gru_Wz.to(mn.gru.W_z.weight.dtype)
        mn.gru.W_z.bias.grad = d_gru_bz.to(mn.gru.W_z.bias.dtype)
        mn.gru.W_r.weight.grad = d_gru_Wr.to(mn.gru.W_r.weight.dtype)
        mn.gru.W_r.bias.grad = d_gru_br.to(mn.gru.W_r.bias.dtype)
        mn.gru.W_h.weight.grad = d_gru_Wh.to(mn.gru.W_h.weight.dtype)
        mn.gru.W_h.bias.grad = d_gru_bh.to(mn.gru.W_h.bias.dtype)
        # PEER queries (unstacked back to per-head)
        for h in range(num_heads):
            mn.peer_queries[h].weight.grad = d_peer_query_Ws[h].to(
                mn.peer_queries[h].weight.dtype)
            mn.product_keys_A[h].grad = d_prod_keys_A[h].to(
                mn.product_keys_A[h].dtype)
            mn.product_keys_B[h].grad = d_prod_keys_B[h].to(
                mn.product_keys_B[h].dtype)
        # Experts (reshape gradients to match parameter shapes)
        mn.expert_W1.grad = d_expert_W1.reshape(
            mn.expert_W1.shape).to(mn.expert_W1.dtype)
        mn.expert_b1.grad = d_expert_b1.to(mn.expert_b1.dtype)
        mn.expert_W2.grad = d_expert_W2.reshape(
            mn.expert_W2.shape).to(mn.expert_W2.dtype)
        mn.expert_b2.grad = d_expert_b2.reshape(
            mn.expert_b2.shape).to(mn.expert_b2.dtype)
        # Input projection
        mn.input_proj.weight.grad = d_input_proj_W.to(
            mn.input_proj.weight.dtype)
        mn.input_proj.bias.grad = d_input_proj_b.to(mn.input_proj.bias.dtype)

        meta_optimizer.step()
        self._weights_dirty = True

        # 6. Restore original training gradients
        for name, p in named_params:
            p.grad = saved_grads.get(name)

        return val_loss.item()

    def _bilevel_step_python(self, model, named_params, saved_grads,
                             val_x, val_y, criterion, meta_optimizer):
        """Python autograd fallback for bilevel meta-net training."""
        smart_grads = {}
        for name, p in named_params:
            if name in saved_grads:
                pidx = self._param_to_idx.get(id(p))
                if pidx is None:
                    continue
                sg, _, _, _ = self.meta_net.forward_for_bilevel(
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
