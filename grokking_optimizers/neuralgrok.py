"""NeuralGrok (Zhou, Fan, Jaggi, Fu, "NeuralGrok: Accelerate Grokking by Neural Gradient Transformation",
arXiv 2504.17243; official code github.com/Blackzxy/NeuralOptGrok, branch ``neuralgrad``).

A small MLP, the *amplifier*, reweights each gradient tensor before Adam
(paper Eq. 2, applied per parameter tensor)::

    m  = MLP(g)                  g = the tensor's signed entries as an [N, 1] column
    p  = softmax(m) over the N entries of that tensor
    g' = c * (p * g) / ||p * g||_2

followed by a global gradient-norm clip at 1.0 and an Adam step. Every
``meta_every`` steps the amplifier is trained by one bilevel "Learn-Amplifier"
step (Algorithm 2): recompute the gradient at the new parameters on the same
batch, take a virtual SGD step ``theta' = theta - virtual_lr * G(g)`` through the
amplifier, and update the amplifier with Adam on the loss of ``theta'`` on a
held-out slice of the training data.

Matches the official code step for step (tests). Two defaults differ from it
without changing any finite result: an all-zero gradient tensor (an MoE expert
that got no tokens) returns zeros instead of 0/0 = NaN, and parameters without a
gradient are skipped instead of crashing.

Training-loop contract: ``needs_closure`` (the fresh gradient at theta_{t+1})
and ``needs_meta_loss`` (the held-out loss as a function of substituted
parameters). ``amplifier_mode="identity"`` is the paper's own control
(``g' = c * g / ||g||``, Section 3.2): normalized Adam without the learned part.

Cost: the amplifier runs on every gradient entry, and the meta step
back-propagates through it, so memory is roughly (hidden widths) x (number of
parameters) floats, 2.8x the model's own forward+backward at the paper's
scale. It does not scale to billions of parameters as published.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ._core import adam_state, adamw_update_, check_common


class NeuralAmplifier(nn.Module):
    """``g -> c * softmax(MLP(g)) * g / ||softmax(MLP(g)) * g||`` over one tensor (PyTorch default init)."""

    def __init__(self, hidden_dims: Sequence[int] = (128, 128), c: float = 1.0):
        super().__init__()
        layers, prev = [], 1
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Softmax(dim=0)]
        self.network = nn.Sequential(*layers)
        self.c = float(c)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        x = g.reshape(-1, 1).float()
        p = self.network(x)
        n = torch.norm(p * x)
        n = torch.where(n > 0, n, torch.ones_like(n))  # all-zero tensor -> zeros, not NaN
        return (self.c * p * x / n).reshape(g.shape).to(g.dtype)


class NeuralGrok(torch.optim.Optimizer):
    flops_depend_on_routing = True  # the amplifier runs on each gradient present; the meta step's backward only
    needs_closure = True  # reaches the experts its held-out batch routes to
    needs_meta_loss = True

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=1e-3,
        *,
        decoupled_weight_decay=False,
        max_grad_norm: float | None = 1.0,
        amp_hidden_dims: Sequence[int] = (128, 128),
        amp_c=1.0,
        amplifier_mode="learned",
        meta_every=4,
        meta_lr=1e-4,
        meta_betas=(0.9, 0.98),
        meta_eps=1e-8,
        meta_weight_decay=1e-3,
        virtual_lr: float | None = None,
    ):
        check_common(lr, betas, eps, weight_decay)
        if amplifier_mode not in ("learned", "identity"):
            raise ValueError(f"amplifier_mode must be 'learned' or 'identity', got {amplifier_mode!r}")
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled_weight_decay=decoupled_weight_decay),
        )
        self.amplifier_mode = amplifier_mode
        self.amplifier = NeuralAmplifier(amp_hidden_dims, amp_c)
        self.amp_c = float(amp_c)
        self.max_grad_norm = max_grad_norm
        self.meta_every = int(meta_every)
        self.virtual_lr = float(lr if virtual_lr is None else virtual_lr)  # official: the constant base lr
        self.meta_opt = torch.optim.Adam(
            self.amplifier.parameters(), lr=meta_lr, betas=meta_betas, eps=meta_eps, weight_decay=meta_weight_decay
        )
        self.global_step = 0
        self.last_meta_loss: torch.Tensor | None = None

    def upcoming_step_kind(self) -> str:
        """What the next ``step()`` will do (the race counts FLOPs per kind)."""
        due = self.amplifier_mode == "learned" and self.meta_every > 0 and (self.global_step + 1) % self.meta_every == 0
        return "inner+meta" if due else "inner"

    def diagnostics(self) -> dict:
        return {"meta_loss": None if self.last_meta_loss is None else float(self.last_meta_loss)}

    def transform(self, g: torch.Tensor) -> torch.Tensor:
        if self.amplifier_mode == "identity":
            n = g.norm()
            return self.amp_c * g / torch.where(n > 0, n, torch.ones_like(n))
        return self.amplifier(g)

    def _params(self):
        return [p for group in self.param_groups for p in group["params"]]

    @torch.no_grad()
    def step(self, closure=None, meta_loss=None):
        """Inner step on the current ``p.grad``; then, when due, the Learn-Amplifier meta step.

        ``closure`` recomputes the training loss and gradients at the current parameters;
        ``meta_loss(params)`` is the held-out loss with ``params`` substituted (aligned with
        the optimizer's parameters, group by group). Unlike most optimizers, the closure is
        *not* called before the update: the caller has already produced ``p.grad``.
        """
        for p in self._params():
            if p.grad is not None:
                p.grad = self.transform(p.grad)
        if self.max_grad_norm is not None:
            with_grad = [p for p in self._params() if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(with_grad, self.max_grad_norm, foreach=False)
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            wd, coupled = group["weight_decay"], not group["decoupled_weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                exp_avg, exp_avg_sq = adam_state(state, p)
                state["step"] += 1
                g = p.grad.add(p, alpha=wd) if coupled and wd != 0 else p.grad  # torch.optim.Adam L2
                adamw_update_(
                    p,
                    g,
                    exp_avg,
                    exp_avg_sq,
                    int(state["step"].item()),
                    lr=group["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    eps=group["eps"],
                    weight_decay=0.0 if coupled else wd,
                )
        self.global_step += 1
        if (
            self.amplifier_mode == "learned"
            and self.meta_every > 0
            and self.global_step % self.meta_every == 0
            and closure is not None
            and meta_loss is not None
        ):
            self.meta_step(closure, meta_loss)
        return None

    def meta_step(self, closure, meta_loss):
        """Learn-Amplifier (Algorithm 2) at the current parameters theta_{t+1}."""
        with torch.enable_grad():
            closure()  # fresh gradient g~ at theta_{t+1} on the same batch, into p.grad
            virtual = [
                p.detach() if p.grad is None else p.detach() - self.virtual_lr * self.amplifier(p.grad.detach())
                for p in self._params()
            ]
            outer = meta_loss(virtual)
            self.meta_opt.zero_grad(set_to_none=True)
            outer.backward()
        self.meta_opt.step()
        self.last_meta_loss = outer.detach()

    def extra_state_bytes(self) -> int:
        amp = sum(p.numel() * p.element_size() for p in self.amplifier.parameters())
        return 3 * amp  # weights + Adam moments

    def state_dict(self):
        sd = super().state_dict()
        sd["neuralgrok"] = {
            "amplifier": self.amplifier.state_dict(),
            "meta_opt": self.meta_opt.state_dict(),
            "global_step": self.global_step,
        }
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)  # do not consume the caller's dict
        extra = state_dict.pop("neuralgrok", None)
        super().load_state_dict(state_dict)
        if extra is not None:
            self.amplifier.load_state_dict(extra["amplifier"])
            self.meta_opt.load_state_dict(extra["meta_opt"])
            self.global_step = extra["global_step"]
