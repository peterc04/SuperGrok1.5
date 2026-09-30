"""SuperGrok 1.1: the project's own optimizer, with every component live.

AdamW whose gradient is corrected element by element by a small learned
network, ``phi``, that sees each gradient entry together with a local sharpness
estimate. There is no paper; this is the algorithm the legacy repository
(commit 19c9d39, ``supergrok11.py`` + ``csrc/algorithms/supergrok11.h``) declares.
That repository could only run it inside a fused CUDA kernel which silently
dropped the layer-wise momentum, the clip and the adaptive alpha, so the
executed optimizer was mostly AdamW. Here everything runs.

One step ``t`` (1-based), for every tensor ``i`` with a gradient::

    alpha  <- alpha_init * exp(-kappa * signal)          every alpha_update_freq steps (and t = 1), from
                                                         set_losses(); signal = 10 once memorized
                                                         (train acc >= 0.995 or loss < 1e-4), else
                                                         max(0, (heldout - train) / train)
    meta step                                            every meta_update_freq steps (below), first
    s_i    <- |grad L(w + eps) - g_i|                    SAM probe at t = 1 and every sam_every steps,
                                                         eps = rho * g / ||g|| (global norm)
    g_i    <- g_i * min(1, clip / (||g_i|| + 1e-6))      per-tensor clip
    mu_i    = r * phi(g_i, s_i)                          phi: Linear(2,H) -> GELU -> Linear(H,1) per element
    gate_i  = 1 - sigmoid(T * cos(g_i, m_i))             m_i = Adam first moment before this step
    ramp_t  = 0 for t <= warmup_steps, then linear to 1 over warmup_ramp steps
    g^_i    = g_i + gate_i * ramp_t * lamb * alpha_i * mu_i
    AdamW on g^_i with beta1_i = beta1 * (1 - gamma) ** layer_i   (bias correction with beta1_i)

The meta step trains ``phi`` and ``r`` through one virtual step,
``w+ = w (1 - lr wd) - lr (g + r phi(g, s))``, minimizing the held-out loss plus
the training loss at ``w+`` (Adam, lr 1e-4). At initialization ``r = 0``, so
the correction is exactly zero until the meta step moves it.

Reductions: ``lamb = 0`` (or ``r`` never trained), ``gamma = 0``, no clip and no
SAM probe is ``torch.optim.AdamW`` bit for bit (tests).

**Known behaviour (measured; see docs/ALGORITHMS.md).** The meta objective is
effectively linear in the correction, so ``phi`` learns a same-sign constant
that Adam grows every meta step, and the one-step lookahead (69-530x smaller
than the real step) cannot see the damage. Once the constant outgrows the
post-memorization gradient, training collapses. Remove the constant and the
meta-net learns nothing. The frozen-meta control (``OPTIMIZERS["supergrok11_frozen"]``)
isolates what the meta-net contributes.

Opt-in variants (defaults are the canonical algorithm):

* ``gate_mode``: ``"momentum"`` (canonical), ``"atlas"`` (``sigmoid(T cos(g, mu))``,
  the owner's written description and the pre-``49453a2`` kernel), ``"none"``.
* ``gate_eps``: 0 = true cosine; ``1e-12`` reproduces the legacy kernel's floor,
  which made the gate a constant 0.5 after memorization.
* ``sharpness_transform``: ``"abs"`` (canonical), ``"square"`` (legacy kernel).
* ``meta_objective``: ``"lookahead"`` (held-out + train), ``"lookahead_val"``
  (the earlier held-out-only version), ``"align"`` (the original objective,
  ``-<g + r phi, grad L_val / ||grad L_val||>``, known to collapse).
* ``meta_grad``: ``"exact"`` or ``"first_order"`` (linear in the correction;
  matches exact to cosine 1.0 at the scales measured, and needs only
  ``meta_chunk``-sized memory: use it beyond ~1e8 parameters).
* ``layer_ids``: what ``i`` counts in ``beta1_i``. ``None`` = tensor index (the
  legacy definition, which leaves most tensors of a deep model with almost no
  momentum); the race passes the transformer-block index.
* ``zero_grad_policy``: ``"apply"`` (canonical) or ``"mask"`` (no correction where
  ``g == 0`` exactly; otherwise unused embedding / Engram rows drift by ~lr per step).
* ``max_correction_ratio``: **not part of the design.** Caps each tensor's applied
  correction at this multiple of ``||g_i||``, a bound the canonical algorithm lacks.

Training-loop contract (``grokking_race/trainer.py``): ``needs_closure`` (SAM
probe), ``needs_meta_loss`` (held-out loss at substituted parameters),
``needs_train_meta_loss`` (training loss at substituted parameters) and
``wants_losses`` (``set_losses(train_loss, heldout_loss, train_acc=...)``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import nn

from ._core import adam_state, adamw_update_, check_common


class SharpnessMetaNet(nn.Module):
    """``correction(g, s) = rescale * MLP([g, s])`` element-wise; ``forward(g, s) = g + correction``.

    Module names, init order and init values are those of the legacy class, so a
    seeded construction gives the same weights and ``state_dict`` keys.
    """

    def __init__(self, hidden_dim: int = 32, init_std: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.rescale = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.net[0].weight.normal_(0, init_std)
            self.net[0].bias.zero_()
            self.net[2].weight.normal_(0, init_std)
            self.net[2].bias.zero_()

    def phi(self, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        dt = self.net[0].weight.dtype
        x = torch.stack([g.reshape(-1).to(dt), s.reshape(-1).to(dt)], dim=1)
        return self.net(x).reshape(g.shape)

    def correction(self, g: torch.Tensor, s: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        if chunk is None or g.numel() <= chunk:
            return self.rescale * self.phi(g, s)
        gf, sf = g.reshape(-1), s.reshape(-1)
        out = torch.cat([self.phi(gf[i : i + chunk], sf[i : i + chunk]) for i in range(0, gf.numel(), chunk)])
        return (self.rescale * out).reshape(g.shape)

    def forward(self, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return g + self.correction(g, s)


def _wdt(t: torch.Tensor) -> torch.dtype:
    """Working precision: at least fp32."""
    return torch.promote_types(t.dtype, torch.float32)


def clip_tensor(g: torch.Tensor, max_norm: float) -> torch.Tensor:
    """``torch.nn.utils.clip_grad_norm_`` applied to one tensor, out of place (off for ``max_norm <= 0``)."""
    if max_norm <= 0:
        return g
    return g * torch.clamp(max_norm / (torch.linalg.vector_norm(g) + 1e-6), max=1.0)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """``<a, b> / (||a|| ||b||)`` (0 if either is 0); ``eps > 0``: ``<a, b> / sqrt(|a|^2 |b|^2 + eps)``."""
    num = (a * b).sum()
    if eps > 0:
        return num / torch.sqrt((a * a).sum() * (b * b).sum() + eps)
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(num))


def block_layer_ids(names: Sequence[str]) -> list[int]:
    """Transformer-block index per parameter name: ``layers.k.*`` -> k + 1, anything before
    the first block (embeddings) -> 0, anything after the last (final norm, head) -> last + 2."""
    import re

    ids, last = [], None
    for n in names:
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", n)
        if m:
            k = int(m.group(1))
            last = k if last is None else max(last, k)
            ids.append(k + 1)
        else:
            ids.append(0 if last is None else -1)
    top = (last if last is not None else -1) + 2
    return [top if i == -1 else i for i in ids]


class SuperGrok11(torch.optim.Optimizer):
    needs_closure = True
    needs_meta_loss = True
    needs_train_meta_loss = True
    wants_losses = True

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1.0,
        *,
        alpha_init=0.98,
        lamb=5.0,
        gamma=0.1,
        gamma_alpha=0.0,
        kappa=0.1,
        warmup_steps=100,
        warmup_ramp=100,
        gradient_clipping=1.0,
        gate_temperature=5.0,
        gate_eps=0.0,
        gate_mode="momentum",
        alpha_update_freq=50,
        zero_loss_threshold=1e-4,
        zero_acc_threshold=0.995,
        sam_rho=0.05,
        sam_every=10,
        sharpness_transform="abs",
        meta_hidden_dim=32,
        meta_init_std=0.01,
        meta_lr=1e-4,
        meta_betas=(0.9, 0.999),
        meta_update_freq=5,
        meta_objective="lookahead",
        meta_grad="exact",
        meta_chunk: int | None = None,
        layer_ids: Sequence[int] | None = None,
        zero_grad_policy="apply",
        max_correction_ratio: float | None = None,
        meta_net: nn.Module | None = None,
        track_stats=True,
    ):
        check_common(lr, betas, eps, weight_decay)
        for name, val, ok in [
            ("gate_mode", gate_mode, ("momentum", "atlas", "none")),
            ("sharpness_transform", sharpness_transform, ("abs", "square")),
            ("meta_objective", meta_objective, ("lookahead", "lookahead_val", "align")),
            ("meta_grad", meta_grad, ("exact", "first_order")),
            ("zero_grad_policy", zero_grad_policy, ("apply", "mask")),
        ]:
            if val not in ok:
                raise ValueError(f"{name} must be one of {ok}, got {val!r}")
        if not (0.0 <= gamma < 1.0 and 0.0 <= gamma_alpha < 1.0):
            raise ValueError("need 0 <= gamma, gamma_alpha < 1")
        if warmup_steps < 0 or warmup_ramp < 1 or sam_every < 0 or meta_update_freq < 0 or alpha_update_freq < 0:
            raise ValueError("need warmup_steps >= 0, warmup_ramp >= 1 and non-negative frequencies")
        if max_correction_ratio is not None and max_correction_ratio < 0:
            raise ValueError("max_correction_ratio must be >= 0")
        if max_correction_ratio is not None and meta_grad == "first_order" and meta_chunk is not None:
            raise ValueError("max_correction_ratio needs whole-tensor norms: use meta_chunk=None with first_order")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.alpha_init, self.lamb, self.kappa = alpha_init, lamb, kappa
        self.gamma, self.gamma_alpha = gamma, gamma_alpha
        self.warmup_steps, self.warmup_ramp = int(warmup_steps), int(warmup_ramp)
        self.gradient_clipping = float(gradient_clipping)
        self.gate_temperature, self.gate_eps, self.gate_mode = gate_temperature, gate_eps, gate_mode
        self.alpha_update_freq = int(alpha_update_freq)
        self.zero_loss_threshold, self.zero_acc_threshold = zero_loss_threshold, zero_acc_threshold
        self.sam_rho, self.sam_every, self.sharpness_transform = sam_rho, int(sam_every), sharpness_transform
        self.meta_update_freq, self.meta_objective, self.meta_grad = int(meta_update_freq), meta_objective, meta_grad
        self.meta_chunk = meta_chunk
        self.zero_grad_policy, self.max_correction_ratio = zero_grad_policy, max_correction_ratio
        self.track_stats = track_stats

        allp = self._all_params()
        self._n = len(allp)
        if layer_ids is None:
            layer_ids = range(self._n)
        layer_ids = [int(x) for x in layer_ids]
        if len(layer_ids) != self._n:
            raise ValueError(f"layer_ids has {len(layer_ids)} entries for {self._n} parameters")
        self._layer = {p: li for p, li in zip(allp, layer_ids)}
        self._index = {p: i for i, p in enumerate(allp)}
        self.meta_net = meta_net if meta_net is not None else SharpnessMetaNet(meta_hidden_dim, meta_init_std)
        self.meta_net.to(allp[0].device)
        self.meta_opt = torch.optim.Adam(self.meta_net.parameters(), lr=meta_lr, betas=meta_betas)
        self.global_step = 0
        self.alpha = float(alpha_init)
        self._losses: tuple | None = None  # (train_loss, heldout_loss, train_acc)
        self.last_meta_loss: torch.Tensor | None = None
        self._stats: dict | None = None

    # ---- schedules ----
    def _all_params(self):
        return [p for group in self.param_groups for p in group["params"]]

    def ramp(self, t: int) -> float:
        return 0.0 if t <= self.warmup_steps else min(1.0, (t - self.warmup_steps) / self.warmup_ramp)

    def layer_alpha(self, i: int) -> float:
        """``alpha_i = clamp(alpha * (1 - gamma_alpha) ** (max(n - 1, 1) - i), 0, 1)``, ``i`` the tensor index
        (the legacy formula, including its ``max``)."""
        scale = 1.0 if self.gamma_alpha == 0.0 else (1.0 - self.gamma_alpha) ** (max(self._n - 1, 1) - i)
        return max(0.0, min(1.0, self.alpha * scale))

    def _corrects(self, t: int) -> bool:
        return self.ramp(t) > 0.0 and self.lamb != 0.0

    def _sam_due(self, t: int) -> bool:
        return self.sam_rho > 0 and self.sam_every > 0 and (t == 1 or t % self.sam_every == 0)

    def _meta_due(self, t: int) -> bool:
        return self.meta_update_freq > 0 and t % self.meta_update_freq == 0

    def upcoming_step_kind(self) -> str:
        """What the next ``step()`` will run (the race counts FLOPs per kind)."""
        t = self.global_step + 1
        kind = "corrected" if self._corrects(t) else "plain"
        return kind + ("+meta" if self._meta_due(t) else "") + ("+sam" if self._sam_due(t) else "")

    def set_losses(self, train_loss=None, heldout_loss=None, train_acc=None) -> None:
        """Losses (and train accuracy) for the adaptive alpha, applied at the next refresh."""
        self._losses = (train_loss, heldout_loss, train_acc)

    def update_alpha(self, train_loss=None, heldout_loss=None, train_acc=None) -> None:
        """The legacy ``_update_alpha``, verbatim in meaning."""
        if train_loss is None and train_acc is None:
            return
        signal = 0.0
        if (train_acc is not None and train_acc >= self.zero_acc_threshold) or (
            train_loss is not None and train_loss < self.zero_loss_threshold
        ):
            signal = 10.0
        elif heldout_loss is not None and train_loss is not None and train_loss > 1e-12:
            signal = max(0.0, (heldout_loss - train_loss) / train_loss)
        self.alpha = self.alpha_init * math.exp(-self.kappa * signal)

    def _sharpness(self, p):
        st = self.state[p]
        if "sharpness" not in st:
            st["sharpness"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return st["sharpness"]

    # ---- SAM probe ----
    @torch.no_grad()
    def sam_probe(self, closure: Callable) -> torch.Tensor:
        """``s <- |grad L(w + rho g / ||g||) - g|`` element-wise; ``p`` and every ``p.grad`` are restored."""
        allp = self._all_params()
        ps = [p for p in allp if p.grad is not None]
        g0 = [p.grad.detach().clone() for p in ps]
        norm = torch.sqrt(sum((g.to(_wdt(g)) ** 2).sum() for g in g0)) + 1e-12
        ron = self.sam_rho / norm
        backup = [p.detach().clone() for p in ps]
        for p, g in zip(ps, g0):
            p.add_((ron * g.to(_wdt(g))).to(p.dtype))
        with torch.enable_grad():
            loss = closure()
        for p, g, b in zip(ps, g0, backup):
            gs = p.grad if p.grad is not None else torch.zeros_like(g)
            d = gs - g
            self._sharpness(p).copy_(d.abs() if self.sharpness_transform == "abs" else d * d)
            p.copy_(b)
        restored = dict(zip(ps, g0))
        for p in allp:  # a parameter without a gradient at w keeps none
            p.grad = restored.get(p)
        return loss

    # ---- meta step ----
    def _meta_inputs(self):
        """Per parameter: (lr, wd) of its own group, the clipped gradient and the sharpness, in the meta-net dtype."""
        allp = self._all_params()
        hp = {p: (group["lr"], group["weight_decay"]) for group in self.param_groups for p in group["params"]}
        dt = next(self.meta_net.parameters()).dtype
        G = [None if p.grad is None else clip_tensor(p.grad.detach().to(dt), self.gradient_clipping) for p in allp]
        S = [None if p.grad is None else self._sharpness(p).to(dt) for p in allp]
        return allp, hp, G, S

    def _shape_correction(self, g: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """The zero-gradient mask and the optional ratio cap, applied alike in the real and the virtual step."""
        if self.zero_grad_policy == "mask":
            corr = torch.where(g == 0, torch.zeros_like(corr), corr)
        if self.max_correction_ratio is not None:
            # scale down only where the cap binds; written so that the backward stays finite at a zero
            # correction (r = 0 at initialization), where bound / ||corr|| would overflow
            bound = self.max_correction_ratio * torch.linalg.vector_norm(g)
            n = torch.linalg.vector_norm(corr)
            over = n > bound
            corr = torch.where(over, corr * (bound / torch.where(over, n, torch.ones_like(n))), corr)
        return corr

    def meta_step(self, meta_loss: Callable, train_meta_loss: Callable | None = None) -> torch.Tensor:
        """Train the meta-net at the current parameters; never touches ``p`` or ``p.grad``."""
        if self.meta_objective == "lookahead" and train_meta_loss is None:
            raise RuntimeError("SuperGrok11: meta_objective='lookahead' needs train_meta_loss")
        allp, hp, G, S = self._meta_inputs()
        idx = [k for k, g in enumerate(G) if g is not None]
        self.meta_opt.zero_grad(set_to_none=True)
        if not idx:
            return self.last_meta_loss
        with torch.enable_grad():
            if self.meta_objective == "align":
                w = [p.detach().requires_grad_(True) for p in allp]
                held = meta_loss(w)
                vg = torch.autograd.grad(held, [w[k] for k in idx], allow_unused=True)
                vg = [torch.zeros_like(G[k]) if v is None else v.float() for k, v in zip(idx, vg)]
                vnorm = torch.sqrt(sum((v**2).sum() for v in vg)) + 1e-12
                smart = self.meta_net(
                    torch.cat([G[k].reshape(-1) for k in idx]), torch.cat([S[k].reshape(-1) for k in idx])
                )
                (-(smart * torch.cat([v.reshape(-1) for v in vg]) / vnorm).sum()).backward()
            elif self.meta_grad == "exact":
                corr = self.meta_net.correction(
                    torch.cat([G[k].reshape(-1) for k in idx]), torch.cat([S[k].reshape(-1) for k in idx])
                )
                virtual, off = [], 0
                for p, g in zip(allp, G):
                    if g is None:
                        virtual.append(p.detach())
                        continue
                    lr, wd = hp[p]
                    sm = g + self._shape_correction(g, corr[off : off + p.numel()].reshape(p.shape))
                    off += p.numel()
                    wdt = torch.promote_types(p.dtype, sm.dtype)
                    virtual.append((p.detach().to(wdt) * (1.0 - lr * wd) - lr * sm.to(wdt)).to(p.dtype))
                held = meta_loss(virtual)
                obj = held + train_meta_loss(virtual) if self.meta_objective == "lookahead" else held
                obj.backward()
            else:  # first order: exact in -lr g, linear in the correction; memory O(meta_chunk)
                base = [
                    p.detach()
                    if g is None
                    else (
                        p.detach().to(torch.promote_types(p.dtype, g.dtype)) * (1.0 - hp[p][0] * hp[p][1])
                        - hp[p][0] * g
                    )
                    .to(p.dtype)
                    .requires_grad_(True)
                    for p, g in zip(allp, G)
                ]
                held = meta_loss(base)
                obj = held + train_meta_loss(base) if self.meta_objective == "lookahead" else held
                u = torch.autograd.grad(obj, [base[k] for k in idx], allow_unused=True)
                chunk = self.meta_chunk or max(G[k].numel() for k in idx)
                for k, uk in zip(idx, u):
                    if uk is None:
                        continue
                    lr = hp[allp[k]][0]
                    uf, gf, sf = uk.detach().to(G[k].dtype).reshape(-1), G[k].reshape(-1), S[k].reshape(-1)
                    for a in range(0, gf.numel(), chunk):
                        corr = self.meta_net.correction(gf[a : a + chunk], sf[a : a + chunk])
                        corr = self._shape_correction(gf[a : a + chunk], corr)  # whole tensor if capped
                        (-lr * (uf[a : a + chunk] * corr).sum()).backward()
        self.meta_opt.step()
        self.last_meta_loss = held.detach()
        return self.last_meta_loss

    # ---- the step ----
    @torch.no_grad()
    def step(self, closure=None, meta_loss=None, train_meta_loss=None):
        """One update from the gradients in ``p.grad`` (left unchanged)."""
        t = self.global_step + 1
        if self._losses is not None and (t == 1 or (self.alpha_update_freq > 0 and t % self.alpha_update_freq == 0)):
            self.update_alpha(*self._losses)
        if self._meta_due(t):
            if meta_loss is None:
                raise RuntimeError(f"SuperGrok11: meta step due at step {t} but no meta_loss given")
            self.meta_step(meta_loss, train_meta_loss)
        if self._sam_due(t):
            if closure is None:
                raise RuntimeError(f"SuperGrok11: SAM probe due at step {t} but no closure given")
            self.sam_probe(closure)
        corrects, ramp = self._corrects(t), self.ramp(t)
        stats = [] if self.track_stats and corrects else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                exp_avg, exp_avg_sq = adam_state(state, p)
                g = clip_tensor(p.grad, self.gradient_clipping)
                if corrects:
                    mu = self.meta_net.correction(g, self._sharpness(p), self.meta_chunk).to(g.dtype)
                    if self.gate_mode == "momentum":
                        gate = 1.0 - torch.sigmoid(self.gate_temperature * cosine(g, exp_avg, self.gate_eps))
                    elif self.gate_mode == "atlas":
                        gate = torch.sigmoid(self.gate_temperature * cosine(g, mu, self.gate_eps))
                    else:
                        gate = torch.ones((), dtype=g.dtype, device=g.device)
                    corr = self._shape_correction(
                        g, (gate * (ramp * self.lamb * self.layer_alpha(self._index[p]))) * mu
                    )
                    g_hat = g + corr
                    if stats is not None:
                        stats.append(
                            (
                                gate.detach(),
                                (corr.float() ** 2).sum(),
                                (g.float() ** 2).sum(),
                                ((torch.sign(g_hat) != torch.sign(g)) & (g != 0)).sum(),
                                (g != 0).sum(),
                            )
                        )
                else:
                    g_hat = g
                state["step"] += 1
                beta1_i = beta1 * (1.0 - self.gamma) ** self._layer[p]
                adamw_update_(
                    p,
                    g_hat,
                    exp_avg,
                    exp_avg_sq,
                    int(state["step"].item()),
                    lr=group["lr"],
                    beta1=beta1_i,
                    beta2=beta2,
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                )
        if stats is not None:
            self._stats = {"t": t, "ramp": ramp, "rows": stats}
        self.global_step = t
        return None

    # ---- introspection ----
    def diagnostics(self) -> dict:
        """State of every component, for the race log (syncs the device; call it at evals only).

        ``meta_loss`` is the held-out loss at the last meta step's virtual point: with the correction for
        ``meta_grad="exact"``, without it for ``"first_order"``. The gate / correction statistics are from the
        last step run with ``track_stats`` on (the race turns it on only for the step before an evaluation).
        """
        net = self.meta_net
        out = {
            "alpha": self.alpha,
            "rescale": float(net.rescale.detach()) if hasattr(net, "rescale") else None,
            "meta_loss": None if self.last_meta_loss is None else float(self.last_meta_loss),
        }
        if hasattr(net, "net"):
            out["phi_bias"] = float(net.net[2].bias.detach())
        sharp = [st["sharpness"] for st in self.state.values() if "sharpness" in st]
        if sharp:
            out["sharpness_mean"] = float(sum(s.float().abs().sum() for s in sharp) / sum(s.numel() for s in sharp))
        if self._stats is not None:
            rows = self._stats["rows"]
            gates = torch.stack([r[0].double() for r in rows])
            corr2, g2 = sum(r[1] for r in rows), sum(r[2] for r in rows)
            out.update(
                stats_step=self._stats["t"],
                ramp=self._stats["ramp"],
                gate_mean=float(gates.mean()),
                gate_min=float(gates.min()),
                gate_max=float(gates.max()),
                correction_ratio=float(torch.sqrt(corr2 / torch.clamp(g2, min=1e-300))),
                sign_flip_frac=float(sum(r[3] for r in rows) / max(int(sum(r[4] for r in rows)), 1)),
            )
        return out

    def extra_state_bytes(self) -> int:
        return 3 * sum(p.numel() * p.element_size() for p in self.meta_net.parameters())  # weights + Adam moments

    def state_dict(self):
        sd = super().state_dict()
        sd["supergrok11"] = dict(
            global_step=self.global_step,
            alpha=self.alpha,
            losses=self._losses,
            meta_net=self.meta_net.state_dict(),
            meta_opt=self.meta_opt.state_dict(),
        )
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop("supergrok11", None)
        super().load_state_dict(state_dict)
        if extra is not None:
            self.global_step, self.alpha, self._losses = extra["global_step"], extra["alpha"], extra["losses"]
            self.meta_net.load_state_dict(extra["meta_net"])
            self.meta_opt.load_state_dict(extra["meta_opt"])
