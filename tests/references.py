"""Independent reference implementations the optimizers are checked against.

Each is a literal transcription of the published algorithm or of the authors'
own code (cited per function), kept deliberately naive: plain loops, no shared
helpers with the package under test.
"""

from __future__ import annotations

import torch


def lion_automl_step(params, grads, moms, *, lr, beta1, beta2, wd):
    """google/automl lion/lion_pytorch.py, Lion.step (operation order preserved)."""
    for p, g, m in zip(params, grads, moms):
        p.mul_(1 - lr * wd)
        update = m * beta1 + g * (1 - beta1)
        p.add_(update.sign_(), alpha=-lr)
        m.mul_(beta2).add_(g, alpha=1 - beta2)


def gradfilter_ema(model, grads=None, alpha=0.98, lamb=2.0):
    """ironjr/grokfast grokfast.py, gradfilter_ema (verbatim logic)."""
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb
    return grads


def grokadamw_published_step(
    params, state, *, lr, betas, eps, wd, alpha_init, lamb, gamma, kappa, clip, train_loss=None, eval_loss=None
):
    """pip grokadamw 0.1.2 / github.com/QuixiAI/grokadamw GrokAdamW.step, one param group, transcribed."""
    import math

    beta1, beta2 = betas
    if train_loss is None or eval_loss is None:
        signal = 0.0
    else:
        mx = max(eval_loss, train_loss)
        signal = max(0, eval_loss - train_loss) / mx if mx > 0 else 0.0
    alpha = alpha_init * math.exp(-kappa * signal)
    for i, p in enumerate([p for p in params if p.grad is not None]):
        st = state.setdefault(
            p, {"step": 0, "m": torch.zeros_like(p), "v": torch.zeros_like(p), "ema": torch.zeros_like(p)}
        )
        st["step"] += 1
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(p, clip)
        g = p.grad
        b1 = beta1 * (1 - gamma) ** i
        st["ema"].mul_(alpha).add_(g, alpha=1 - alpha)
        gg = g + lamb * st["ema"]
        st["m"].mul_(b1).add_(gg, alpha=1 - b1)
        st["v"].mul_(beta2).addcmul_(gg, gg, value=1 - beta2)
        step_size = lr * math.sqrt(1 - beta2 ** st["step"]) / (1 - beta1 ** st["step"])
        p.mul_(1 - lr * wd)
        p.addcdiv_(st["m"], st["v"].sqrt().add_(eps), value=-step_size)


class ProdigyReference:
    """prodigyopt 1.1.2 Prodigy.step (decoupled weight decay, no bias correction), transcribed with per-tensor
    .item() accumulation exactly as the original."""

    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, d0=1e-6):
        import math

        self.params, self.lr, self.betas, self.eps, self.wd, self.d0 = list(params), lr, betas, eps, weight_decay, d0
        self.beta3 = math.sqrt(betas[1])
        self.d, self.d_max, self.d_numerator, self.k, self.state = d0, d0, 0.0, 0, {}

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.betas
        d, d0, lr = self.d, self.d0, self.lr
        dlr = d * lr
        self.d_numerator *= self.beta3
        delta_numerator, d_denom = 0.0, 0.0
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state.setdefault(
                p,
                {
                    "s": torch.zeros_like(p.flatten()),
                    "p0": p.flatten().clone(),
                    "m": torch.zeros_like(p),
                    "v": torch.zeros_like(p),
                },
            )
            g = p.grad
            delta_numerator += (d / d0) * dlr * torch.dot(g.flatten(), st["p0"] - p.flatten()).item()
            st["m"].mul_(beta1).add_(g, alpha=d * (1 - beta1))
            st["v"].mul_(beta2).addcmul_(g, g, value=d * d * (1 - beta2))
            st["s"].mul_(self.beta3).add_(g.flatten(), alpha=(d / d0) * dlr)
            d_denom += st["s"].abs().sum().item()
        self.d_numerator += delta_numerator
        d_hat = self.d_numerator / d_denom
        if self.d == d0:
            self.d = max(self.d, d_hat)
        self.d_max = max(self.d_max, d_hat)
        self.d = min(self.d_max, self.d * float("inf"))
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[p]
            denom = st["v"].sqrt().add_(self.d * self.eps)
            p.add_(p, alpha=-self.wd * dlr)
            p.addcdiv_(st["m"], denom, value=-dlr)
        self.k += 1


def muon_reference_step(params, bufs, *, lr, wd, momentum, update_rms=0.2):
    """torch.optim.Muon's update (EMA momentum, Nesterov, Keller quintic x5, 'match_rms_adamw' scale) in fp32.

    Its EMA momentum is (1 - momentum) times the Moonlight/Kimi heavy-ball buffer, a scale that the
    Frobenius normalization in Newton-Schulz removes, so both forms give the same update."""
    import math

    for p, buf in zip(params, bufs):
        g = p.grad
        buf.lerp_(g, 1 - momentum)
        u = g.lerp(buf, momentum)
        x = u.T if u.size(0) > u.size(1) else u
        x = x / (x.norm() + 1e-7)
        for _ in range(5):
            a, b, c = 3.4445, -4.7750, 2.0315
            s = x @ x.T
            x = a * x + (b * s + c * s @ s) @ x
        if u.size(0) > u.size(1):
            x = x.T
        p.mul_(1 - lr * wd)
        p.add_(x, alpha=-lr * update_rms * math.sqrt(max(p.shape)))


class NeuralGrokReference:
    """Official NeuralOptGrok (branch neuralgrad) training step, transcribed: transform_grads with the
    softmax amplifier, global clip 1.0, torch.optim.Adam (coupled L2), and amp_update every T steps."""

    def __init__(self, model, amplifier, lr=1e-3, wd=1e-3, meta_every=4, meta_lr=1e-4):
        self.model, self.amp, self.lr, self.meta_every = model, amplifier, lr, meta_every
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=wd)
        self.meta_opt = torch.optim.Adam(
            amplifier.parameters(), lr=meta_lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-3
        )
        self.t = 0

    def amplify(self, g):
        x = g.reshape(-1, 1)
        p = self.amp(x)
        return (self.amp_c * p * x / torch.norm(p * x)).reshape(g.shape)

    def train_step(self, loss_fn, x, y, xo, yo):
        from torch.func import functional_call

        self.opt.zero_grad()
        loss_fn(self.model(x), y).backward()
        with torch.no_grad():
            for p in self.model.parameters():
                p.grad = self.amplify(p.grad)
        torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), 1.0, foreach=False)
        self.opt.step()
        self.t += 1
        if self.t % self.meta_every == 0:
            names, params = zip(*self.model.named_parameters())
            grads = torch.autograd.grad(loss_fn(self.model(x), y), params)
            virtual = {n: p.detach() - self.lr * self.amplify(g.detach()) for n, p, g in zip(names, params, grads)}
            outer = loss_fn(functional_call(self.model, virtual, (xo,)), yo)
            self.meta_opt.zero_grad()
            outer.backward()
            self.meta_opt.step()


class LookSAMReference:
    """Liu et al. 2022, Algorithm 1, transcribed with plain flattened vectors in float64, then AdamW
    (torch.optim.AdamW, foreach=False) on the resulting gradient. SAM step on the first step and every k-th."""

    def __init__(self, model, loss_fn, *, rho, k, alpha, **adamw):
        self.model, self.loss_fn, self.rho, self.k, self.alpha = model, loss_fn, rho, k, alpha
        self.opt = torch.optim.AdamW(model.parameters(), foreach=False, **adamw)
        self.t, self.g_v = 0, None

    def _flat_grad(self):
        return torch.cat([p.grad.reshape(-1).double() for p in self.model.parameters()])

    def _grad_at(self, x, y):
        self.model.zero_grad(set_to_none=True)
        self.loss_fn(self.model(x), y).backward()
        return self._flat_grad()

    def step(self, x, y):
        g = self._grad_at(x, y)
        if self.t % self.k == 0:
            params = list(self.model.parameters())
            backup = [p.detach().clone() for p in params]
            eps = self.rho * g / (g.norm() + 1e-12)
            with torch.no_grad():
                i = 0
                for p in params:
                    n = p.numel()
                    p.add_(eps[i : i + n].view_as(p).to(p.dtype))
                    i += n
            g_s = self._grad_at(x, y)
            with torch.no_grad():
                for p, b in zip(params, backup):
                    p.copy_(b)
            cos = (g @ g_s) / (g.norm() * g_s.norm())
            self.g_v = g_s - g_s.norm() * cos * g / g.norm()
            update = g_s
        else:
            update = g + self.alpha * (g.norm() / self.g_v.norm()) * self.g_v
        i = 0
        for p in self.model.parameters():
            n = p.numel()
            p.grad = update[i : i + n].view_as(p).to(p.dtype)
            i += n
        self.opt.step()
        self.t += 1


class SuperGrok11Reference:
    """SuperGrok 1.1 as declared by the legacy repository (commit 19c9d39), every knob live, transcribed
    naively with one parameter group: ``_update_alpha``, ``_get_ramp_factor``, ``sam_step`` (functional SAM,
    sharpness ``|g_sam - g|``) and ``meta_step`` (two-term lookahead) from
    ``grokking_optimizers/optimizers/supergrok11.py``; the per-element update and the Adam tail from
    ``csrc/algorithms/supergrok11.h`` (``sg11_sweep_b_step``: ``p - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)``).
    Declared semantics kept where the executed kernel dropped them: layer-wise beta1 (bias correction with
    beta1_i), the per-tensor clip (also on the meta-net's input), the adaptive alpha, and a true cosine."""

    def __init__(
        self,
        model,
        meta_net,
        *,
        lr,
        betas,
        eps,
        wd,
        alpha_init,
        lamb,
        gamma,
        kappa,
        warmup,
        ramp_len,
        clip,
        temperature,
        alpha_every,
        rho,
        sam_every,
        meta_every,
        meta_lr,
        meta_betas,
    ):
        import math

        self.math = math
        self.model, self.meta_net = model, meta_net
        self.names = [n for n, _ in model.named_parameters()]
        self.params = [p for _, p in model.named_parameters()]
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, wd
        self.alpha_init, self.lamb, self.gamma, self.kappa = alpha_init, lamb, gamma, kappa
        self.warmup, self.ramp_len, self.clip, self.T = warmup, ramp_len, clip, temperature
        self.alpha_every, self.rho, self.sam_every, self.meta_every = alpha_every, rho, sam_every, meta_every
        self.meta_opt = torch.optim.Adam(meta_net.parameters(), lr=meta_lr, betas=meta_betas)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.s = [torch.zeros_like(p) for p in self.params]
        self.t, self.alpha = 0, alpha_init

    def _clip(self, g):
        n = torch.sqrt((g * g).sum())
        return g * min(1.0, float(self.clip / (n + 1e-6))) if self.clip > 0 else g

    def _mlp(self, g, s):  # rescale * Linear(H,1)(GELU(Linear(2,H)([g, s]))), one element per row
        net = self.meta_net.net
        h = torch.nn.functional.gelu(torch.stack([g.reshape(-1), s.reshape(-1)], 1) @ net[0].weight.T + net[0].bias)
        return (self.meta_net.rescale * (h @ net[2].weight.T + net[2].bias)).reshape(g.shape)

    def step(self, loss_fn, x, y, xo, yo, losses=None):
        from torch.func import functional_call

        grads = torch.autograd.grad(loss_fn(self.model(x), y), self.params)
        self.t += 1
        t = self.t
        if losses is not None and (t == 1 or t % self.alpha_every == 0):
            train_loss, heldout_loss, train_acc = losses
            signal = 0.0
            if train_acc >= 0.995 or train_loss < 1e-4:
                signal = 10.0
            elif train_loss > 1e-12:
                signal = max(0.0, (heldout_loss - train_loss) / train_loss)
            self.alpha = self.alpha_init * self.math.exp(-self.kappa * signal)
        if t % self.meta_every == 0:
            G = [self._clip(g) for g in grads]
            virtual = {
                n: p.detach() * (1 - self.lr * self.wd) - self.lr * (g + self._mlp(g, s))
                for n, p, g, s in zip(self.names, self.params, G, self.s)
            }
            meta = loss_fn(functional_call(self.model, virtual, (xo,)), yo) + loss_fn(
                functional_call(self.model, virtual, (x,)), y
            )
            self.meta_opt.zero_grad()
            meta.backward()
            self.meta_opt.step()
        if t == 1 or t % self.sam_every == 0:
            norm = torch.sqrt(sum((g * g).sum() for g in grads)) + 1e-12
            perturbed = {
                n: (p.detach() + self.rho / norm * g).requires_grad_(True)
                for n, p, g in zip(self.names, self.params, grads)
            }
            g_sam = torch.autograd.grad(
                loss_fn(functional_call(self.model, perturbed, (x,)), y), list(perturbed.values())
            )
            self.s = [(a - b).abs() for a, b in zip(g_sam, grads)]
        ramp = 0.0 if t <= self.warmup else min(1.0, (t - self.warmup) / self.ramp_len)
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(self.params, grads)):
                g = self._clip(g)
                if ramp > 0:
                    mu = self._mlp(g, self.s[i])
                    gm = torch.sqrt((g * g).sum()) * torch.sqrt((self.m[i] * self.m[i]).sum())
                    cos = float((g * self.m[i]).sum() / gm) if gm > 0 else 0.0
                    gate = 1.0 - 1.0 / (1.0 + self.math.exp(-self.T * cos))
                    g = g + gate * ramp * self.lamb * min(1.0, max(0.0, self.alpha)) * mu
                b1 = self.b1 * (1 - self.gamma) ** i
                self.m[i] = b1 * self.m[i] + (1 - b1) * g
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
                m_hat, v_hat = self.m[i] / (1 - b1**t), self.v[i] / (1 - self.b2**t)
                p.copy_(p - self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.wd * p))
