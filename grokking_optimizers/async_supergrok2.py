"""
AsyncSuperGrok2 — Double-buffered async SuperGrok2 optimizer (Phase 5A).

Overlaps meta-net computation with the next forward+backward pass using a
dedicated CUDA stream and double-buffered meta-net weights.  Smart gradients
are 1-step stale by design, which is acceptable for the meta-net signal.

Architecture
------------
    stream_main   : forward + backward + Adam update
    stream_meta   : meta-net forward using gradient snapshot

Each step:
    1. Main stream records event_grad_ready after gradients are available.
    2. Meta stream waits for event_grad_ready, then runs meta-net on the
       *snapshot* of the previous step's gradients using the inactive weight
       buffer.
    3. Meta stream records event_meta_done.
    4. Main stream waits for event_meta_done from the *previous* iteration,
       then applies the Adam update using the (1-step stale) smart gradients.
    5. Gradient snapshot is taken for the next meta-net pass.
    6. Weight buffers are swapped.

The first step runs synchronously because there are no previous smart
gradients to consume.
"""

import copy
import torch
from torch.optim import Optimizer

from grokking_optimizers.supergrok2 import SuperGrok2


class AsyncSuperGrok2(Optimizer):
    """Double-buffered async SuperGrok2 -- overlaps meta-net with next fwd+bwd.

    The meta-net runs on a dedicated CUDA stream using a snapshot of the
    previous step's gradients, while the main stream runs the next
    forward+backward pass.  Smart gradients are 1-step stale.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-3).
        **sg2_kwargs: All remaining keyword arguments are forwarded to the
            underlying :class:`SuperGrok2` constructor (e.g. ``d_model``,
            ``betas``, ``weight_decay``, etc.).

    Usage::

        opt = AsyncSuperGrok2(model.parameters(), lr=1e-3, d_inner=16)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            opt.step()  # internally overlaps meta-net with next iteration
    """

    def __init__(self, params, lr=1e-3, **sg2_kwargs):
        # Materialize the parameter list so we can pass it to both the
        # base Optimizer and the internal SuperGrok2.
        param_list = list(params)
        if len(param_list) == 0:
            raise ValueError("AsyncSuperGrok2 requires at least one parameter")

        # We need to figure out if the first parameter lives on a CUDA device
        # so we know whether to set up streams.
        first_param = param_list[0] if not isinstance(param_list[0], dict) else param_list[0]["params"].__iter__().__next__()
        self._use_cuda = first_param.is_cuda and torch.cuda.is_available()

        # Defaults dict for the base Optimizer (mirrors SuperGrok2 defaults
        # that are needed for param_groups).
        defaults = dict(lr=lr)
        super().__init__(param_list, defaults)

        # ---- Internal SuperGrok2 instance (does the real work) ----
        # We pass our already-constructed param_groups so SuperGrok2 sees
        # the same parameter tensors.
        sg2_params = [
            {"params": list(group["params"]), **{k: v for k, v in group.items() if k != "params"}}
            for group in self.param_groups
        ]
        self._inner = SuperGrok2(sg2_params, lr=lr, **sg2_kwargs)

        # ---- CUDA streams and events ----
        if self._use_cuda:
            self._stream_meta = torch.cuda.Stream(device=first_param.device)
            self._event_grad_ready = torch.cuda.Event()
            self._event_meta_done = torch.cuda.Event()
        else:
            self._stream_meta = None
            self._event_grad_ready = None
            self._event_meta_done = None

        # ---- Double-buffered meta-net weights ----
        # buf_A and buf_B are deep copies of the meta-net state dict.
        # The inner optimizer always references self._inner.meta_net; we
        # swap the underlying parameters by loading state dicts.
        self._meta_weights_a = copy.deepcopy(self._inner.meta_net.state_dict())
        self._meta_weights_b = copy.deepcopy(self._inner.meta_net.state_dict())
        self._active_buf = "a"  # "a" is currently loaded in meta_net

        # ---- Gradient snapshot buffers ----
        # Lazily allocated on first step, matching each parameter's shape.
        self._grad_snapshots = None  # list[Tensor | None], one per param

        # ---- Previous smart gradients (1-step stale) ----
        # After the meta stream finishes, the smart grads are stored here
        # for the *next* main-stream Adam update.
        self._prev_smart_grads = None  # list[Tensor | None]

        # ---- Step bookkeeping ----
        self._async_step_count = 0
        self._meta_launched = False  # True while a meta-net pass is in flight

    # ------------------------------------------------------------------
    # Gradient snapshot
    # ------------------------------------------------------------------

    def _init_snapshot_buffers(self):
        """Lazily allocate gradient snapshot buffers matching each parameter."""
        self._grad_snapshots = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self._grad_snapshots.append(
                        torch.zeros_like(p.data, memory_format=torch.contiguous_format)
                    )
                else:
                    self._grad_snapshots.append(None)

    def _snapshot_gradients(self):
        """Copy current gradients to snapshot buffer.

        On CUDA this uses ``copy_()`` which is asynchronous on the current
        stream, so the copy is overlapped with subsequent work on other
        streams.
        """
        if self._grad_snapshots is None:
            self._init_snapshot_buffers()

        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    snap = self._grad_snapshots[idx]
                    if p.grad is not None:
                        snap.copy_(p.grad.data)
                    else:
                        snap.zero_()
                idx += 1

    # ------------------------------------------------------------------
    # Double-buffer swap
    # ------------------------------------------------------------------

    def _swap_buffers(self):
        """Swap active/inactive weight buffers.

        Saves the current meta-net state into the active buffer, then
        loads the inactive buffer into the meta-net for the next meta-net
        pass.
        """
        meta_net = self._inner.meta_net

        if self._active_buf == "a":
            # Save current weights to buf_A, load buf_B
            self._meta_weights_a = {
                k: v.clone() for k, v in meta_net.state_dict().items()
            }
            meta_net.load_state_dict(self._meta_weights_b)
            self._active_buf = "b"
        else:
            # Save current weights to buf_B, load buf_A
            self._meta_weights_b = {
                k: v.clone() for k, v in meta_net.state_dict().items()
            }
            meta_net.load_state_dict(self._meta_weights_a)
            self._active_buf = "a"

        # Mark inner optimizer's weight cache as dirty so it re-extracts
        # weights on the next call.
        if hasattr(self._inner, '_weights_dirty'):
            self._inner._weights_dirty = True

    # ------------------------------------------------------------------
    # Meta-net launch (on meta stream)
    # ------------------------------------------------------------------

    def _launch_meta_on_stream(self):
        """Run the meta-net forward pass on the meta stream using snapshot
        gradients.  Results are stored in ``self._prev_smart_grads``.

        This method must be called while the meta stream is the active
        stream context (or synchronously when CUDA is unavailable).
        """
        inner = self._inner
        inner._ensure_state()

        smart_grads = []
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    smart_grads.append(None)
                    idx += 1
                    continue

                snap = self._grad_snapshots[idx]
                if snap is None or snap.abs().max().item() == 0:
                    smart_grads.append(None)
                    idx += 1
                    continue

                # Flatten for meta-net
                param_idx = inner._param_to_idx.get(id(p))
                if param_idx is None:
                    smart_grads.append(None)
                    idx += 1
                    continue

                flat_grad = snap.reshape(-1)
                flat_sharp = inner._flat_sharpness[param_idx].reshape(-1)

                gru_state = inner._flat_gru_states[param_idx]
                fwd_state = inner._flat_mamba_fwd_states[param_idx]
                bwd_state = inner._flat_mamba_bwd_states[param_idx]

                # Initialize Mamba states if needed
                if fwd_state is None:
                    d_inner = inner.meta_net.mamba_fwd.d_inner
                    d_state = inner.meta_net.d_state
                    fwd_state = torch.zeros(
                        d_inner, d_state, dtype=torch.float32, device=p.device)
                    bwd_state = torch.zeros(
                        d_inner, d_state, dtype=torch.float32, device=p.device)
                    inner._flat_mamba_fwd_states[param_idx] = fwd_state
                    inner._flat_mamba_bwd_states[param_idx] = bwd_state

                with torch.no_grad():
                    sg, new_gru, new_fwd, new_bwd = inner.meta_net(
                        flat_grad, flat_sharp,
                        gru_state, fwd_state, bwd_state,
                    )

                inner._flat_gru_states[param_idx] = new_gru.detach()
                inner._flat_mamba_fwd_states[param_idx] = new_fwd.detach()
                inner._flat_mamba_bwd_states[param_idx] = new_bwd.detach()

                smart_grads.append(sg.detach().reshape(p.data.shape))
                idx += 1

        self._prev_smart_grads = smart_grads

    # ------------------------------------------------------------------
    # Adam update using stale smart gradients
    # ------------------------------------------------------------------

    def _adam_update_with_stale_grads(self, closure_loss):
        """Apply the Adam update using current raw gradients blended with
        1-step-stale smart gradients from the meta stream.

        This mirrors the inner loop of SuperGrok2.step() but substitutes
        the synchronous meta-net call with the pre-computed stale smart
        gradients.
        """
        import math

        inner = self._inner
        inner._ensure_state()
        inner._global_step += 1

        base_alpha = inner._cached_alpha
        ramp = inner._get_ramp_factor()
        gate_signal = inner._get_gate_signal()

        group = inner.param_groups[0]
        lr = group["lr"]
        beta2 = group["betas"][1]
        eps = group["eps"]
        wd_eff = inner._get_effective_wd(group["weight_decay"])
        lamb_eff = inner.lamb * ramp * gate_signal

        sg_idx = 0
        for i, p in enumerate(inner._flat_params):
            if p.grad is None:
                sg_idx += 1
                continue

            inner._flat_steps[i] += 1
            grad = p.grad.data

            # Gradient clipping + NaN guard (same as SuperGrok2)
            grad_norm = grad.norm()
            if grad_norm > inner.gradient_clipping:
                grad = grad * (inner.gradient_clipping / (grad_norm + 1e-12))
            if not torch.isfinite(grad).all():
                grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))

            alpha_i = max(0.0, min(1.0, base_alpha * inner._flat_layer_alphas[i]))
            beta1_i = inner._flat_layer_beta1s[i]
            step_i = inner._flat_steps[i]
            bc1 = 1.0 - beta1_i ** step_i
            bc2 = 1.0 - beta2 ** step_i

            # Momentum (mu) update
            mu = inner._flat_mus[i]
            if mu.dtype != torch.float32:
                mu_fp32 = mu.float()
                mu_fp32.mul_(alpha_i).add_(grad.reshape(-1), alpha=1.0 - alpha_i)
                inner._flat_mus[i] = mu_fp32.to(mu.dtype)
            else:
                mu.mul_(alpha_i).add_(grad.reshape(-1), alpha=1.0 - alpha_i)

            # Build effective gradient: use stale smart grad if available,
            # otherwise fall back to raw gradient.
            smart_grad = None
            if self._prev_smart_grads is not None and sg_idx < len(self._prev_smart_grads):
                smart_grad = self._prev_smart_grads[sg_idx]

            if smart_grad is not None:
                effective_grad = smart_grad.reshape(-1) + lamb_eff * inner._flat_mus[i].float()
            else:
                # No stale smart grad -- use raw gradient (first step or None grad snapshot)
                effective_grad = grad.reshape(-1).float() + lamb_eff * inner._flat_mus[i].float()

            # Adam update
            fg = effective_grad.float()
            ea = inner._flat_exp_avgs[i]
            easq = inner._flat_exp_avg_sqs[i]

            ea.mul_(beta1_i).add_(fg, alpha=1 - beta1_i)
            easq.mul_(beta2).addcmul_(fg, fg, value=1 - beta2)
            step_size = lr / bc1
            denom = (easq / bc2).sqrt().add_(eps)
            p.data.mul_(1 - lr * wd_eff)
            p.data.addcdiv_(ea.reshape(p.data.shape), denom.reshape(p.data.shape), value=-step_size)

            sg_idx += 1

        # Expert recycling
        inner._step_counter += 1
        if (inner.meta_net.recycle_interval > 0 and
                inner._step_counter % inner.meta_net.recycle_interval == 0):
            if inner.expert_allreduce_before_recycle:
                inner._allreduce_expert_counts()
            inner.meta_net._recycle_dead_experts()

        # Periodic mamba state sync
        if (inner.mamba_state_sync_interval > 0 and
                inner._step_counter % inner.mamba_state_sync_interval == 0):
            inner._sync_mamba_states()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None, train_loss=None, val_loss=None, train_acc=None):
        """Perform an asynchronous optimizer step.

        Pipeline overview:

        1. If a meta-net pass from the previous step is in-flight, wait
           for it to complete (``event_meta_done``).
        2. Apply the Adam update using current gradients blended with
           1-step-stale smart gradients from the previous meta-net pass.
        3. Snapshot current gradients for the next meta-net pass.
        4. Launch meta-net forward on the meta stream using the snapshot.
        5. Swap double-buffered meta-net weights.

        The first step runs synchronously (delegates to the inner
        :class:`SuperGrok2`) because there are no stale smart gradients.

        Args:
            closure: An optional closure that re-evaluates the model and
                returns the loss.
            train_loss: Current training loss (for alpha scheduling).
            val_loss: Current validation loss (for alpha scheduling).
            train_acc: Current training accuracy (for alpha scheduling).

        Returns:
            The loss from the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Forward scheduling signals to the inner optimizer
        if train_acc is not None:
            self._inner._cached_train_acc = train_acc
        if self._inner._global_step % max(1, self._inner.alpha_update_freq) == 0 or self._inner._global_step == 0:
            self._inner._update_alpha(train_loss, val_loss, train_acc)
        elif train_acc is not None and train_acc >= self._inner.zero_acc_threshold:
            self._inner._update_alpha(train_loss, val_loss, train_acc)
        elif train_loss is not None and train_loss < self._inner.zero_loss_threshold:
            self._inner._update_alpha(train_loss, val_loss, train_acc)

        self._async_step_count += 1

        # ---- CPU / no-CUDA fallback: run synchronously ----
        if not self._use_cuda:
            return self._inner.step(
                closure=None,  # already evaluated above
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
            )

        # ---- First CUDA step: run synchronously ----
        if self._async_step_count == 1:
            result = self._inner.step(
                closure=None,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
            )
            # Snapshot gradients for the next meta-net pass
            self._snapshot_gradients()
            # Launch meta-net on the meta stream for the NEXT step
            self._event_grad_ready.record(torch.cuda.current_stream())
            self._stream_meta.wait_event(self._event_grad_ready)
            with torch.cuda.stream(self._stream_meta):
                self._launch_meta_on_stream()
                self._event_meta_done.record(self._stream_meta)
            self._meta_launched = True
            self._swap_buffers()
            return result if loss is None else loss

        # ---- Steady-state async step (step >= 2) ----

        # Step A: Wait for the previous meta-net pass to complete.
        if self._meta_launched:
            torch.cuda.current_stream().wait_event(self._event_meta_done)
            self._meta_launched = False

        # Step B: Adam update using current grads + 1-step-stale smart grads.
        self._adam_update_with_stale_grads(loss)

        # Step C: Snapshot current gradients for the next meta-net pass.
        self._snapshot_gradients()

        # Step D: Launch meta-net on meta stream for the NEXT step.
        self._event_grad_ready.record(torch.cuda.current_stream())
        self._stream_meta.wait_event(self._event_grad_ready)
        with torch.cuda.stream(self._stream_meta):
            self._launch_meta_on_stream()
            self._event_meta_done.record(self._stream_meta)
        self._meta_launched = True

        # Step E: Swap double-buffered meta-net weights.
        self._swap_buffers()

        return loss

    def synchronize(self):
        """Block until all pending async meta-net work completes.

        Call this before checkpointing, evaluation, or any operation that
        reads meta-net state to ensure consistency.
        """
        if self._use_cuda and self._meta_launched:
            torch.cuda.current_stream().wait_event(self._event_meta_done)
            self._meta_launched = False

    # ------------------------------------------------------------------
    # State dict (for checkpointing)
    # ------------------------------------------------------------------

    def state_dict(self):
        """Return the full state dict including inner optimizer and buffers."""
        self.synchronize()
        return {
            "inner": self._inner.state_dict(),
            "meta_weights_a": self._meta_weights_a,
            "meta_weights_b": self._meta_weights_b,
            "active_buf": self._active_buf,
            "async_step_count": self._async_step_count,
        }

    def load_state_dict(self, state_dict):
        """Load state dict and restore double-buffer state."""
        self._inner.load_state_dict(state_dict["inner"])
        self._meta_weights_a = state_dict["meta_weights_a"]
        self._meta_weights_b = state_dict["meta_weights_b"]
        self._active_buf = state_dict["active_buf"]
        self._async_step_count = state_dict["async_step_count"]

        # Reload the correct buffer into the meta-net
        if self._active_buf == "a":
            self._inner.meta_net.load_state_dict(self._meta_weights_a)
        else:
            self._inner.meta_net.load_state_dict(self._meta_weights_b)

        if hasattr(self._inner, '_weights_dirty'):
            self._inner._weights_dirty = True

        # Reset async state -- next step will run synchronously
        self._meta_launched = False
        self._prev_smart_grads = None
        self._grad_snapshots = None

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    @property
    def param_groups(self):
        """Proxy to inner optimizer's param_groups for LR schedulers."""
        return self._inner.param_groups

    @param_groups.setter
    def param_groups(self, value):
        # During super().__init__ the base Optimizer sets param_groups.
        # After that we want writes to go to the inner optimizer.
        if hasattr(self, '_inner'):
            self._inner.param_groups = value
        else:
            # Called from Optimizer.__init__ before _inner exists
            super().__setattr__('_param_groups_init', value)

    def zero_grad(self, set_to_none=True):
        """Zero gradients on all parameters."""
        self._inner.zero_grad(set_to_none=set_to_none)

    def sam_step(self, model, train_x, train_y, criterion):
        """Delegate SAM step to the inner optimizer."""
        self.synchronize()
        return self._inner.sam_step(model, train_x, train_y, criterion)

    def __repr__(self):
        return (
            f"AsyncSuperGrok2(\n"
            f"  async_step_count={self._async_step_count},\n"
            f"  use_cuda={self._use_cuda},\n"
            f"  active_buf={self._active_buf!r},\n"
            f"  inner={self._inner!r}\n"
            f")"
        )
