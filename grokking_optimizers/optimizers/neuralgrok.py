"""
NeuralGrok — Adam with a learned MLP gradient amplifier.

Uses a small neural network (the *amplifier*) to learn per-element gradient
scaling factors. The amplifier is a lightweight MLP that maps gradient
magnitudes to scale factors, enabling adaptive amplification that can be
jointly trained alongside the main model.

All heavy computation is dispatched to the fused C++/CUDA kernel via _ops.
"""

import warnings
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from grokking_optimizers.dispatch import get_ops
from grokking_optimizers.optimizers.adamw import _validate_grad

_ops = get_ops()  # Fails loudly if C++ extension not built

# The L3-TC megakernel evaluates the psi-net at a COMPILE-TIME hidden width
# (csrc/fused/sm_90/opt_components.cuh: static constexpr int kPsiHidden = 16, and
# the eager per-op full_step is instantiated <16> too). The amplifier MUST use this
# width for the L3 path so the trained net == the deployed net (no A3-class
# train/deploy split). OPTIMIZER_CONFIGS["neuralgrok"] pins neural_hidden=16 to
# match. Mirror it here as the single Python-side source of truth (psi_pack asserts
# against it); keep byte-identical to the kernel constant.
KERNEL_PSI_HIDDEN = 16


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


class _Amplifier(nn.Module):
    """MLP gradient amplifier: grad magnitude -> scale factor.

    A small feed-forward network that takes the absolute gradient value
    (shape ``[N, 1]``) and produces a multiplicative scale factor.

    Args:
        num_layers: Total number of linear layers (default: 3).
        hidden_dim: Width of hidden layers (default: 128).
    """

    def __init__(self, num_layers: int = 3, hidden_dim: int = 128,
                 grad_checkpoint: bool = True) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 (input + output layer)")

        self.hidden_dim = hidden_dim
        self.grad_checkpoint = grad_checkpoint
        layers: List[nn.Module] = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        # Group the flat Sequential into natural sub-blocks for checkpointing:
        # each (Linear, ReLU) pair is one block, plus the trailing output Linear.
        # Activations are recomputed in the backward pass per-block rather than
        # held in memory, which is the point of gradient checkpointing.
        self._blocks = nn.ModuleList()
        i = 0
        net_layers = list(self.net)
        while i < len(net_layers):
            if i + 1 < len(net_layers) and isinstance(net_layers[i + 1], nn.ReLU):
                self._blocks.append(nn.Sequential(net_layers[i], net_layers[i + 1]))
                i += 2
            else:
                self._blocks.append(nn.Sequential(net_layers[i]))
                i += 1

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: gradient magnitude -> scale factor.

        Each (Linear, ReLU) sub-block is independently checkpointed in
        train mode so its activations are recomputed during backward.
        """
        for block in self._blocks:
            x = _maybe_checkpoint(block, x, self.grad_checkpoint)
        return x

    def get_weights(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return first- and last-layer weights for the C++ kernel.

        NOTE: The C++ kernel only evaluates a 2-layer MLP (first + last layer).
        If num_layers > 2, intermediate layers are skipped in the CUDA path.
        Set num_layers=2 for exact CUDA/Python parity.

        Returns:
            (W1, b1, W_last, b_last) — all as contiguous FP32 tensors.
        """
        first_linear = self.net[0]
        last_linear = self.net[-1]

        W1 = first_linear.weight.detach().float().contiguous()
        b1 = first_linear.bias.detach().float().contiguous()
        W_last = last_linear.weight.detach().float().contiguous()
        b_last = last_linear.bias.detach().float().contiguous()

        return W1, b1, W_last, b_last

    def psi_pack(self, device=None) -> Tensor:
        """The psi-net weights flattened into the L3-TC kernel's `extra`-slice layout.

        The L3-TC megakernel (mega_{decoder,vit}_real_adamw_tc_launcher.cu) does NOT
        take the amplifier weights as a separate ABI arg — it binds st.psi_W1/b1/W2
        from the HEAD of the per-tensor `extra` state slice, in the EXACT packing
        csrc/fused/sm_90/opt_components.cuh (#5) documents:

            extra[0          .. H)        = psi_W1  [H]
            extra[H          .. 2H)       = psi_b1  [H]
            extra[2H         .. 3H)       = psi_W2  [H]
            extra[3H]                     = psi_b2  (scalar; read on-device as W2[H])

        with H == kPsiHidden == 16 (the kernel's COMPILE-TIME psi width). This method
        returns that contiguous ``[3*H + 1]`` fp32 vector so the host (dispatch.
        fused_train_step) can scatter the FRESH amplifier weights into the L3 state
        cache's extra slice every step — the seam that makes the kernel read the
        host-trained amplifier instead of a null/stale pointer.

        The vector is exactly the same psi math the kernel evaluates: with H==16 and
        num_layers==2, ``_Amplifier.net`` is ``Linear(1,H) -> ReLU -> Linear(H,1)``,
        so get_weights' (W1[H,1], b1[H], W_last[1,H], b_last[1]) flatten to
        (psi_W1[H], psi_b1[H], psi_W2[H], psi_b2) — the per-hidden weights
        neuralgrok_psi_forward<H> reads positionally.

        Raises if hidden_dim != KERNEL_PSI_HIDDEN: the kernel's psi width is fixed at
        compile time, so a mismatched amplifier would train/deploy DIFFERENT functions
        (the A3-class train/deploy split). Loud, not silent.
        """
        if self.hidden_dim != KERNEL_PSI_HIDDEN:
            raise ValueError(
                f"NeuralGrok.psi_pack: amplifier hidden_dim={self.hidden_dim} but the "
                f"L3-TC kernel's psi width is fixed at kPsiHidden={KERNEL_PSI_HIDDEN} "
                f"(compile-time). Build the optimizer with hidden_dim="
                f"{KERNEL_PSI_HIDDEN} for the L3 path (OPTIMIZER_CONFIGS pins "
                f"neural_hidden=16), or the kernel reads a different-width MLP than "
                f"the one trained.")
        if len(self.net) != 3:
            raise ValueError(
                "NeuralGrok.psi_pack: the kernel evaluates a 2-layer MLP "
                "(Linear-ReLU-Linear); amplifier has num_layers != 2 so the pack "
                "would not match the deployed psi function. Set num_layers=2.")
        W1, b1, W_last, b_last = self.get_weights()
        H = self.hidden_dim
        pack = torch.empty(3 * H + 1, dtype=torch.float32)
        pack[0:H].copy_(W1.reshape(-1))          # psi_W1[H]  (Linear(1,H).weight)
        pack[H:2 * H].copy_(b1.reshape(-1))      # psi_b1[H]
        pack[2 * H:3 * H].copy_(W_last.reshape(-1))  # psi_W2[H]  (Linear(H,1).weight)
        pack[3 * H:3 * H + 1].copy_(b_last.reshape(-1))  # psi_b2 (scalar, read on-device)
        if device is not None:
            pack = pack.to(device)
        return pack


class NeuralGrok(Optimizer):
    """Adam with a learned MLP gradient amplifier.

    The amplifier is a small ``_Amplifier`` network whose weights can be
    trained with a separate optimiser (see :meth:`get_amplifier_optimizer`).

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for running averages of gradient and its
            square (default: (0.9, 0.98)).
        eps: Numerical stability term (default: 1e-8).
        weight_decay: Decoupled weight decay (default: 1.0).
        alpha: Amplification scale factor (default: 10.0).
        beta: Secondary amplification scale factor (default: 4.0).
        num_layers: Number of layers in the amplifier MLP (default: 3).
        hidden_dim: Hidden dimension of the amplifier MLP (default: 128).
        inner_steps: Deprecated — dead arg with no kernel slot; ignored. The
            fused kernel performs exactly one amplifier evaluation per step.
            Kept for API backward compatibility; passing a non-default value
            emits a DeprecationWarning (default: 1).
        grad_clip: Maximum gradient norm for per-parameter clipping
            (default: 1.0).
        amplifier_lr: Learning rate for the INTERNALLY-owned Adam that trains
            the amplifier in :meth:`train_amplifier_step` (default: 1e-3). The
            amplifier is trained inside the optimizer (AdamW-parity directive:
            the caller does not have to own an optimizer), though an external
            one may still be passed to ``train_amplifier_step``.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 1.0,
        alpha: float = 10.0,
        beta: float = 4.0,
        num_layers: int = 3,
        hidden_dim: int = 128,
        inner_steps: int = 1,
        grad_clip: float = 1.0,
        amplifier_lr: float = 1e-3,
        use_grad_hooks: bool = False,
        grad_checkpoint: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        # `inner_steps` is a dead arg: the fused kernel does ONE amplifier
        # evaluation per step (no kernel slot for inner iterations). Warn if a
        # non-default value is passed so callers know it is ignored; still
        # accepted for back-compat.
        if inner_steps != 1:
            warnings.warn(
                "NeuralGrok 'inner_steps' is deprecated and has no kernel slot; "
                "it is ignored (the fused kernel performs one amplifier "
                "evaluation per step).",
                DeprecationWarning, stacklevel=2)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            beta=beta,
            inner_steps=inner_steps,
            grad_clip=grad_clip,
        )

        # The amplifier is constructed before super().__init__ so that it
        # lives on the correct device once parameters are registered.
        self.amplifier = _Amplifier(num_layers=num_layers, hidden_dim=hidden_dim,
                                    grad_checkpoint=grad_checkpoint)

        super().__init__(params, defaults)

        self._meta_weights_dirty = True
        self._cached_meta_weights = None

        # Internally-owned Adam over the amplifier params, lazily constructed on
        # first train_amplifier_step (AdamW-parity directive: the caller does
        # not have to own an amplifier optimizer). amplifier_lr is its LR.
        self._amplifier_lr = amplifier_lr
        self._amp_opt = None

        # Per-group cache of static tensor lists (params + exp_avg/exp_avg_sq
        # buffers keep a fixed identity across steps); only grads_list and step
        # counters are refreshed per step. Invalidated by add_param_group.
        self._static_cache: dict = {}
        # Lazily-bound fused kernel callable (resolved once at first step()).
        self._fused_step = None

        self._use_grad_hooks = use_grad_hooks
        if use_grad_hooks:
            _register_grad_hooks(self)

    def add_param_group(self, param_group) -> None:
        self._static_cache = {}
        super().add_param_group(param_group)

    def _group_cache(self, group):
        """Cached (params, exp_avg, exp_avg_sq, states), keyed on the
        grad-bearing param ids."""
        key = tuple(id(p) for p in group["params"] if p.grad is not None)
        cached = self._static_cache.get(id(group))
        if cached is not None and cached[0] == key:
            return cached[1]
        params_list = []
        exp_avg_list = []
        exp_avg_sq_list = []
        states = []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            params_list.append(p)
            exp_avg_list.append(state["exp_avg"])
            exp_avg_sq_list.append(state["exp_avg_sq"])
            states.append(state)
        entry = (params_list, exp_avg_list, exp_avg_sq_list, states)
        self._static_cache[id(group)] = (key, entry)
        return entry

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Perform a single optimisation step.

        Args:
            closure: A closure that re-evaluates the model and returns the loss
                (optional).

        Returns:
            The loss value if *closure* is provided, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._use_grad_hooks:
            return loss

        fused_step = self._fused_step
        if fused_step is None:
            fused_step = self._fused_step = _ops.bind("neuralgrok_fused_step")

        # Extract amplifier weights for the fused kernel (cached)
        if self._meta_weights_dirty or self._cached_meta_weights is None:
            self._cached_meta_weights = self.amplifier.get_weights()
            self._meta_weights_dirty = False
        W1, b1, W_last, b_last = self._cached_meta_weights

        for group in self.param_groups:
            params_list, exp_avg_list, exp_avg_sq_list, states = \
                self._group_cache(group)

            if len(params_list) == 0:
                continue

            grads_list = [_validate_grad(p) for p in params_list]
            step_list = []
            for state in states:
                state["step"] += 1
                step_list.append(state["step"])

            # Move amplifier weights to same device as params
            device = params_list[0].device
            W1_d = W1.to(device)
            b1_d = b1.to(device)
            W_last_d = W_last.to(device)
            b_last_d = b_last.to(device)

            fused_step(
                params_list,
                grads_list,
                exp_avg_list,
                exp_avg_sq_list,
                step_list,
                W1_d,
                b1_d,
                W_last_d,
                b_last_d,
                group["alpha"],
                group["beta"],
                self.amplifier.hidden_dim,
                group["betas"][0],
                group["betas"][1],
                group["lr"],
                group["weight_decay"],
                group["eps"],
                group["grad_clip"],
            )

        return loss

    def mark_amplifier_dirty(self) -> None:
        """Mark the cached amplifier weights as stale.

        Call this after updating the amplifier weights (e.g. after
        ``amplifier_optimizer.step()``) so that the next optimizer step
        re-extracts FP32 contiguous copies.
        """
        self._meta_weights_dirty = True

    def get_amplifier(self) -> _Amplifier:
        """Return the gradient amplifier module."""
        return self.amplifier

    def psi_pack(self, device=None) -> Tensor:
        """The amplifier psi-net weights in the L3-TC kernel's `extra`-slice layout.

        Thin delegate to :meth:`_Amplifier.psi_pack` (the amplifier owns the weights).
        dispatch.fused_train_step calls this on the live NeuralGrok every step to
        scatter the FRESH (possibly just host-trained) psi weights into the L3 state
        cache's extra slice, so the megakernel evaluates the host-owned amplifier.
        Returns a contiguous ``[3*kPsiHidden + 1]`` fp32 vector [W1|b1|W2|b2].
        """
        return self.amplifier.psi_pack(device=device)

    def train_amplifier_step(self, model, val_x, val_y, criterion,
                             amplifier_optimizer=None):
        """Train the gradient amplifier via a one-step lookahead objective.

        AUDIT A3 (the defect this fixes): the amplifier MLP was AUTOGRAD-
        UNREACHABLE on every path that ever existed. The fused kernel consumes
        it only through DETACHED ``get_weights()`` snapshots (step()/_single_
        param_step), and the race's old ``aloss = neural_beta*CE(m(vax), vay)``
        contained ONLY the model ``m`` — the amplifier params were in neither
        that graph nor any optimizer's param_groups, so ``aloss.backward()``
        left them grad-less and ``aopt.step()`` was a permanent no-op. The
        amplifier was therefore frozen at random init for the entire run, and
        ``mark_amplifier_dirty()`` faithfully refreshed a snapshot of weights
        that never changed.

        THE FIX (mirrors the proven SuperGrok11.meta_step lookahead template):
        rebuild the amplified update DIFFERENTIABLY THROUGH THE AMPLIFIER in
        Python, take one virtual optimizer step, evaluate validation loss
        there, and backprop into the amplifier weights. The graph now reaches
        the amplifier, so its parameters actually move; the kernel then picks
        up the trained weights via the snapshot refresh (mark_amplifier_dirty).

        Steps (each annotated with the SG11.meta_step line it mirrors):
          (a) snapshot current per-param grads g, DETACHED (they exist right
              after the caller's backward; we differentiate w.r.t. the
              amplifier, not w.r.t. g).
          (b) amplified_g = (psi*alpha + beta) * g, where
              psi = amplifier(|g|) is built KEEPING THE GRAPH to the amplifier
              weights. FEATURIZATION PARITY IS THE CORRECTNESS CRUX: the
              deployed kernel feeds the per-element scalar |g| (fabsf(grad[idx])
              — csrc/fused/sm_90/opt_components.cuh:225; raw |g|, NO norm) into a
              2-layer ReLU MLP psi = sum_j W2[j]*relu(W1[j]*|g| + b1[j]) + b2
              (csrc/algorithms/neuralgrok.h:38-45), then applies
              g_amp = (psi*alpha + beta)*g (neuralgrok.h:70). With num_layers=2
              (the race's setting, and the kernel's documented 2-layer contract
              — get_weights' note), _Amplifier.forward is EXACTLY that MLP, so
              feeding it |g| reshaped to [N,1] trains the deployed function.
          (c) virtual step at the optimizer's OWN lr/wd (decoupled-WD shrink +
              the amplified gradient), built from p.detach() — same recipe as
              SG11.meta_step's lookahead:
                  w⁺ = p·(1 − lr·wd) − lr·amplified_g
          (d) amp_loss = criterion(functional_call(model, w⁺, val_x), val_y),
              backward into the amplifier optimizer, step it.

              OBJECTIVE CHOICE — VAL-ONLY (not SG11's two-term val+train): the
              SG meta-net adds a learned correction to the gradient
              (grad + rescale·MLP), so it can REDIRECT the step in an
              ARBITRARY direction off the train manifold; SG11 needed the train
              anchor because the val-optimal additive push walks the weights
              off a memorized minimum and collapses. The amplifier is far more
              constrained: it is a PER-COORDINATE MULTIPLICATIVE rescale —
              g_amp = (psi·alpha + beta)·g — that acts INDEPENDENTLY on each
              coordinate and cannot introduce cross-coordinate rotation the way
              the additive meta can. (It is not strictly sign-preserving: psi is
              an unbounded linear output, so psi·alpha+beta can go negative and
              flip an individual coordinate; but it still cannot synthesize an
              off-manifold direction out of coordinates the gradient leaves at
              zero, and the search space is a diagonal rescale of g, not all of
              R^n.) The off-manifold collapse mode the train anchor exists to
              prevent is therefore much weaker here, so we start val-only — the
              simplest objective that makes the amplifier learn a useful
              per-coordinate scale. If a future run shows val-only overshoot,
              add the train term exactly as in SG11.meta_step (the hook for it
              is one line below).
          (e) mark_amplifier_dirty() so the next kernel step() re-extracts the
              just-trained weights (the snapshot refresh — necessary but
              useless until now, since the weights finally change).
          (f) restore each p.grad untouched (snapshot/restore, like
              SG11.meta_step) so the subsequent fused step() consumes the
              caller's original backward result, not our intermediate state.

        Args:
            model: the model whose validation loss defines the objective.
            val_x, val_y: held-out batch (the same held-out convention the
                SuperGrok bilevel/meta nets use).
            criterion: loss (e.g. nn.CrossEntropyLoss()).
            amplifier_optimizer: OPTIONAL external optimizer over the amplifier
                params (e.g. the race's existing aopt). If None, the
                internally-owned Adam (amplifier_lr) is lazily built and used —
                so the caller is not required to own one.

        Returns:
            The scalar validation loss at the virtual step (float).
        """
        named_params = list(model.named_parameters())

        # (a) Snapshot DETACHED grads. These are the caller's backward result;
        # we differentiate the objective w.r.t. the AMPLIFIER, with g held
        # constant (detached) — exactly SG11.meta_step's `saved_grads`.
        saved_grads = {}
        for name, p in named_params:
            if p.grad is not None:
                saved_grads[name] = p.grad.detach().clone()
        if not saved_grads:
            # No grads to amplify (caller has not run backward, or all grads are
            # None). Nothing to train against — fail LOUDLY rather than silently
            # no-op, per the honesty rails (a silent no-op here would recreate
            # exactly the A3 defect this method exists to kill).
            raise RuntimeError(
                "train_amplifier_step called with no parameter gradients — run "
                "loss.backward() before training the amplifier (A3: a silent "
                "no-op here is the defect this method fixes).")

        # Pick the optimizer: external if supplied, else the lazily-built
        # internal Adam over the amplifier params (AdamW-parity directive).
        if amplifier_optimizer is None:
            if self._amp_opt is None:
                self._amp_opt = torch.optim.Adam(
                    self.amplifier.parameters(), lr=self._amplifier_lr)
            amplifier_optimizer = self._amp_opt

        # (b) Build the amplified gradient DIFFERENTIABLY through the amplifier.
        # Batched: one amplifier forward over all params' |g| concatenated, then
        # scatter back (mirrors SG11.meta_step's batched meta-net forward). The
        # amplifier weights require grad, so the graph reaches them; g itself is
        # detached, so we differentiate ONLY the amplifier (psi*alpha+beta).
        group = self.param_groups[0]
        alpha = float(group["alpha"])
        beta = float(group["beta"])
        lr_eff = float(group["lr"])
        wd_eff = float(group.get("weight_decay", 0.0))

        all_abs = []
        all_names = []
        all_sizes = []
        for name, _p in named_params:
            if name in saved_grads:
                g = saved_grads[name]
                # KERNEL FEATURIZATION: per-element scalar |g| (raw, no norm).
                # Shape [N, 1] is the _Amplifier input contract (Linear(1, H)).
                all_abs.append(g.reshape(-1, 1).abs())
                all_names.append(name)
                all_sizes.append(g.numel())

        with torch.enable_grad():
            cat_abs = torch.cat(all_abs, dim=0)            # [sum(N), 1]
            # _Amplifier.forward == the kernel's 2-layer ReLU MLP when
            # num_layers==2 (the deployed config). If num_layers>2 the kernel
            # only evaluates first+last layer (get_weights' documented contract)
            # while this trains the full stack — flag LOUDLY so we never
            # silently optimize a function the kernel does not run.
            if len(self.amplifier.net) != 3:
                warnings.warn(
                    "NeuralGrok amplifier has num_layers != 2; the fused kernel "
                    "evaluates only a 2-layer (first+last) MLP, so "
                    "train_amplifier_step is optimizing a DIFFERENT function "
                    "than the kernel deploys. Set num_layers=2 for parity.",
                    RuntimeWarning, stacklevel=2)
            cat_psi = self.amplifier(cat_abs).reshape(-1)  # [sum(N)] psi(|g|)

            # (c) Virtual step at the optimizer's own lr/wd, from p.detach():
            #     w⁺ = p·(1 − lr·wd) − lr·(psi·alpha + beta)·g
            # amplified_g keeps the graph to the amplifier (via psi); g detached.
            virtual = {}
            offset = 0
            for name, p in named_params:
                base = p.detach()
                if name in saved_grads:
                    idx = all_names.index(name)
                    size = all_sizes[idx]
                    psi = cat_psi[offset:offset + size].reshape(saved_grads[name].shape)
                    offset += size
                    amplified_g = (psi * alpha + beta) * saved_grads[name]
                    virtual[name] = base * (1.0 - lr_eff * wd_eff) - lr_eff * amplified_g
                else:
                    virtual[name] = base

            # (d) Validation loss at the virtual step. functional_call swaps the
            # virtual tensors in as the module's params, so grads flow back
            # through amplified_g into the amplifier ONLY (real params untouched).
            amplifier_optimizer.zero_grad()
            val_logits = torch.func.functional_call(model, virtual, (val_x,))
            amp_loss = criterion(val_logits, val_y)

        amp_loss.backward()
        amplifier_optimizer.step()

        # (e) Refresh the kernel's weight snapshot — now meaningful because the
        # weights actually changed (pre-A3-fix this was a no-op on frozen
        # weights).
        self.mark_amplifier_dirty()

        # (f) Restore the caller's grads untouched so the subsequent fused
        # step() consumes the original backward result (SG11.meta_step parity).
        for name, p in named_params:
            p.grad = saved_grads.get(name)

        return amp_loss.item()

    def maybe_train_amplifier(self, model, val_x, val_y, criterion, step,
                              every=1, amplifier_optimizer=None):
        """Host-side amplifier-train CADENCE for the L3-TC path (eager, every k steps).

        The L3-TC megakernel runs the model fwd+bwd+amplified-AdamW in ONE persistent
        kernel and CANNOT call back into Python autograd to train the amplifier (the
        amplifier's lookahead objective needs functional_call + a second graph). So the
        amplifier is trained HOST-SIDE on a cadence, EAGERLY, here: every ``every``
        steps this runs one ``train_amplifier_step`` (which itself snapshots/restores
        p.grad and marks the snapshot dirty). The caller then re-extracts the fresh
        psi_pack() into the kernel's `extra` state slice (dispatch.fused_train_step does
        this every step, so the just-trained weights are picked up on the very next
        kernel launch).

        CADENCE PARITY: train_neuralgrok trains the amplifier EVERY step (every==1 — the
        default here reproduces that). A larger ``every`` is the host-cost amortization
        the directive allows ("every k steps"); the math each time it fires is byte-
        identical to train_amplifier_step (== the deployed psi function at num_layers=2,
        hidden=16).

        REQUIRES the caller to have run model backward already (so p.grad is populated
        for the lookahead's detached-grad snapshot) — on the L3-TC path the kernel does
        NOT leave eager grads, so the caller runs a SEPARATE eager fwd+bwd on the train
        batch to seed p.grad before calling this (the host-side amplifier train is a
        side computation, distinct from the kernel's in-fused grad). Returns the scalar
        val loss when it fired this step, else None.

        Args mirror train_amplifier_step; ``step`` is the 1-based global step and
        ``every`` the cadence. ``amplifier_optimizer`` optional (else the internal Adam).
        """
        if every < 1:
            raise ValueError(f"maybe_train_amplifier: every must be >= 1 (got {every})")
        if step % every != 0:
            return None
        return self.train_amplifier_step(model, val_x, val_y, criterion,
                                         amplifier_optimizer=amplifier_optimizer)

    def _single_param_step(self, param, group, state):
        """Per-parameter step used by the `use_grad_hooks=True` path."""
        if param.grad is None:
            return
        grad = _validate_grad(param)
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
        state["step"] += 1
        if self._meta_weights_dirty or self._cached_meta_weights is None:
            self._cached_meta_weights = self.amplifier.get_weights()
            self._meta_weights_dirty = False
        W1, b1, W_last, b_last = self._cached_meta_weights
        device = param.device
        _ops.neuralgrok_fused_step(
            [param], [grad], [state["exp_avg"]], [state["exp_avg_sq"]],
            [state["step"]],
            W1.to(device), b1.to(device), W_last.to(device), b_last.to(device),
            group["alpha"], group["beta"],
            self.amplifier.hidden_dim,
            group["betas"][0], group["betas"][1],
            group["lr"], group["weight_decay"], group["eps"],
            group["grad_clip"],
        )

    def get_amplifier_optimizer(self, lr: float = 1e-4) -> torch.optim.Adam:
        """Create an Adam optimiser for the amplifier's parameters.

        Args:
            lr: Learning rate for training the amplifier (default: 1e-4).

        Returns:
            A ``torch.optim.Adam`` instance targeting the amplifier weights.
        """
        return torch.optim.Adam(self.amplifier.parameters(), lr=lr)


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
