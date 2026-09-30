"""Shampoo (Gupta, Koren, Singer, ICML 2018; at scale: Anil et al. 2020, Shi et al. 2023) with Adam grafting.

Implements, for one device, exactly what Meta's Distributed Shampoo does
(github.com/facebookresearch/optimizers, ``distributed_shampoo``, the
implementation that won the 2024 AlgoPerf training-algorithms benchmark) in its
recommended "replace Adam" configuration: ``RootInvShampooPreconditionerConfig``
with eigendecomposition roots, ``AdamPreconditionerConfig`` grafting, decoupled
weight decay, bias correction. Tests compare it with that code step for step.

For every parameter:

1. **Blocking.** Dimensions of size 1 are dropped, adjacent dimensions are merged
   while their product stays <= ``max_preconditioner_dim`` (so small matrices
   become vectors), and the result is cut into blocks of at most
   ``max_preconditioner_dim`` along every dimension.
2. **Factor matrices**, per block and dimension ``k``, from the raw gradient
   ``G``: ``L_k <- beta2 L_k + (1 - beta2) G_(k) G_(k)^T``, where ``G_(k)``
   contracts every dimension but ``k`` (for a matrix: ``L = G G^T``, ``R = G^T G``).
3. **Inverse roots**, every ``precondition_frequency`` steps from
   ``start_preconditioning_step`` on: ``A = L_k / (1 - beta2^t) + epsilon I``,
   ``A = Q diag(lam) Q^T`` (``torch.linalg.eigh``; eigenvalues below ``epsilon``
   are shifted up), ``L_k^(-1/(2 order)) = Q diag(lam^(-1/(2 order))) Q^T``:
   ``-1/4`` per side for a matrix, ``-1/2`` for a vector (full-matrix AdaGrad).
4. **Direction.** ``m`` is the Adam first moment of ``G`` (bias corrected).
   Shampoo's direction is ``L^(-1/4) m R^(-1/4)`` (a vector: ``L^(-1/2) m``);
   **grafting** rescales it, per block, to the norm of Adam's direction
   ``m / (sqrt(v / (1 - beta2_g^t)) + eps_g)``. Before
   ``start_preconditioning_step`` the direction is Adam's.
5. **Update.** ``W <- W - lr (direction + weight_decay W)`` (decoupled decay).

So the step size of every block is Adam's, and Shampoo only chooses its
direction: with the race's shared lr, betas and weight decay, a difference from
AdamW is attributable to the preconditioner.

Parameter groups may set ``use_shampoo=False``: their blocks take Adam's
direction throughout (the race does this for embeddings, tables, norms and
other non-matrix parameters, like Muon). Parameters without a gradient are
skipped on that step, as in the reference.

Cost: the factor matrices take ``sum_k d_k^2`` floats per block (for a
1024-element block merged to a vector: 1M floats), their inverse roots as much
again, plus Adam's ``m`` and ``v``. Each root costs one ``d x d``
eigendecomposition (about ``9 d^3`` FLOPs), which the race counts even though
PyTorch's FLOP counter does not. ``upcoming_step_kind()`` separates Adam-only,
preconditioned and root-computing steps for that accounting.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import reduce

import torch
from torch import Tensor

from ._core import check_common


def merge_small_dims(shape: tuple[int, ...], threshold: float, target_dimensionality: int = 1) -> tuple[int, ...]:
    """``distributed_shampoo.utils.shampoo_utils.merge_small_dims``: drop size-1 dimensions, then merge
    adjacent dimensions (from the last) while the product stays <= ``threshold``."""
    if 0 in shape:
        return (0,)
    squeezed = [d for d in reversed(shape) if d != 1] or [1]
    new = [squeezed[0]]
    for processed, nxt in enumerate(squeezed[1:], start=1):
        potential = len(new) + len(squeezed) - processed
        if potential > target_dimensionality and new[-1] * nxt <= threshold:
            new[-1] *= nxt
        else:
            new.append(nxt)
    return tuple(reversed(new))


def multi_dim_split(tensor: Tensor, split_size: float) -> tuple[Tensor, ...]:
    """Cut ``tensor`` into blocks of at most ``split_size`` along every dimension (views)."""
    if split_size == math.inf:
        return (tensor,)
    return reduce(
        lambda parts, dim: tuple(s for t in parts for s in torch.split(t, int(split_size), dim=dim)),
        range(tensor.dim()),
        (tensor,),
    )


def _blocks(t: Tensor, max_dim: float) -> tuple[Tensor, ...]:
    return multi_dim_split(t.view(merge_small_dims(tuple(t.shape), max_dim)), max_dim)


def _symmetric_from_upper(m: Tensor) -> Tensor:
    """The reference stores symmetric matrices as their packed upper triangle; this is its unpacking."""
    return torch.triu(m) + torch.triu(m, 1).T


def matrix_inverse_root(a: Tensor, root: float, epsilon: float) -> Tensor:
    """``(a + epsilon I)^(-1/root)`` by ``torch.linalg.eigh``, with the reference's eigenvalue floor."""
    a_ridge = a.add(torch.eye(a.shape[0], dtype=a.dtype, device=a.device), alpha=epsilon) if epsilon else a
    lam, q = torch.linalg.eigh(a_ridge)
    lam_min = torch.min(lam).item()
    if lam_min < epsilon:  # not positive definite in floating point: shift the spectrum up
        lam = (lam + (-lam_min)) + epsilon
    return q * lam.pow(-1.0 / Fraction(root)).unsqueeze(0) @ q.T


class Shampoo(torch.optim.Optimizer):
    flops_depend_on_routing = True  # factor updates and preconditioning run for each parameter with a gradient

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        epsilon=1e-12,
        weight_decay=0.0,
        *,
        max_preconditioner_dim: float = 1024,
        precondition_frequency: int = 1,
        start_preconditioning_step: int = -1,
        use_bias_correction=True,
        grafting_beta2: float | None = None,
        grafting_epsilon=1e-8,
        use_shampoo=True,
        factor_dtype=torch.float32,
    ):
        check_common(lr, betas, epsilon, weight_decay)
        defaults = dict(
            lr=lr,
            betas=tuple(betas),
            epsilon=epsilon,
            weight_decay=weight_decay,
            max_preconditioner_dim=max_preconditioner_dim,
            precondition_frequency=int(precondition_frequency),
            start_preconditioning_step=int(start_preconditioning_step),
            use_bias_correction=use_bias_correction,
            grafting_beta2=betas[1] if grafting_beta2 is None else grafting_beta2,
            grafting_epsilon=grafting_epsilon,
            use_shampoo=use_shampoo,
            step=0,
        )
        super().__init__(params, defaults)
        self.factor_dtype = factor_dtype
        for group in self.param_groups:
            if group["precondition_frequency"] < 1:
                raise ValueError("precondition_frequency must be >= 1")
            if group["start_preconditioning_step"] == -1:
                group["start_preconditioning_step"] = group["precondition_frequency"]
            if group["start_preconditioning_step"] < group["precondition_frequency"]:
                raise ValueError("start_preconditioning_step must be >= precondition_frequency (or -1)")
            if not 0.0 < group["grafting_beta2"] <= 1.0 or group["grafting_epsilon"] <= 0:
                raise ValueError("need 0 < grafting_beta2 <= 1 and grafting_epsilon > 0")
        schedules = {
            (g["precondition_frequency"], g["start_preconditioning_step"])
            for g in self.param_groups
            if g["use_shampoo"]
        }
        if len(schedules) > 1:
            raise ValueError("all Shampoo groups must share precondition_frequency and start_preconditioning_step")

    # ---- schedule ----
    @staticmethod
    def _amortized(group, t: int) -> bool:
        freq, start = group["precondition_frequency"], group["start_preconditioning_step"]
        return (t % freq == 0 and t > start) or t == start

    def upcoming_step_kind(self) -> str:
        """``adam`` (before preconditioning starts), ``shampoo`` or ``shampoo+root`` (inverse roots recomputed)."""
        for group in self.param_groups:
            if group["use_shampoo"]:
                t = group["step"] + 1
                if t < group["start_preconditioning_step"]:
                    return "adam"
                return "shampoo+root" if self._amortized(group, t) else "shampoo"
        return "adam"

    # ---- state ----
    def _block_state(self, p: Tensor, group) -> dict:
        st = self.state[p]
        if "blocks" not in st:
            blocks = _blocks(p.detach(), group["max_preconditioner_dim"])
            st["blocks"] = []
            for b in blocks:
                bs = {"m": torch.zeros_like(b), "v": torch.zeros_like(b)}
                if group["use_shampoo"]:
                    dims = b.shape
                    order = len(dims)
                    bs["factors"] = [torch.zeros(d, d, dtype=self.factor_dtype, device=p.device) for d in dims]
                    bs["inv_factors"] = [torch.eye(d, dtype=self.factor_dtype, device=p.device) for d in dims]
                    bs["root"] = 1 / (1 / (2 * max(order, 1)))
                st["blocks"].append(bs)
        return st

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["step"] += 1
            step = torch.tensor(group["step"], dtype=torch.int64)  # the reference's CPU step tensor
            t = group["step"]
            beta1, beta2 = group["betas"]
            beta2_g, eps_g = group["grafting_beta2"], group["grafting_epsilon"]
            use_bc = group["use_bias_correction"]
            shampoo = group["use_shampoo"]
            amortized = shampoo and self._amortized(group, t)
            graft_only = (not shampoo) or t < group["start_preconditioning_step"]
            # bias corrections as the reference computes them (float32 CPU scalars)
            bc1 = 1.0 - beta1 * beta1 ** (step - 1) if (beta1 != 0.0 and use_bc) else None
            bc2 = torch.tensor(1.0) - beta2**step if (use_bc and beta2 < 1.0) else torch.tensor(1.0)
            bc2_g = torch.tensor(1.0) - beta2_g**step if beta2_g < 1.0 else torch.tensor(1.0)
            lr = torch.tensor(group["lr"], dtype=torch.float)
            wd = group["weight_decay"]
            w_factor = 1.0 if beta2 == 1.0 else 1 - beta2
            for p in group["params"]:
                if p.grad is None or p.grad.numel() == 0:
                    continue
                st = self._block_state(p, group)
                grads = _blocks(p.grad, group["max_preconditioner_dim"])
                params = _blocks(p.detach(), group["max_preconditioner_dim"])
                for bs, g, w in zip(st["blocks"], grads, params):
                    # 1. preconditioner statistics from the raw gradient
                    if shampoo:
                        order = g.dim()
                        for k, factor in enumerate(bs["factors"]):
                            outer = torch.tensordot(g, g, dims=[[*range(k), *range(k + 1, order)]] * 2)
                            if beta2 != 1.0:
                                factor.mul_(beta2)
                            factor.add_(outer, alpha=w_factor)  # summed in the gradient's precision, rounded once
                        if amortized:
                            for i, factor in enumerate(bs["factors"]):
                                a = _symmetric_from_upper(factor / bc2)
                                if not torch.isfinite(a).all():
                                    raise FloatingPointError("Shampoo: non-finite factor matrix")
                                inv = matrix_inverse_root(a, bs["root"], group["epsilon"]).to(self.factor_dtype)
                                bs["inv_factors"][i] = _symmetric_from_upper(inv)
                    v = bs["v"]
                    if beta2_g != 1.0:
                        v.mul_(beta2_g)
                    v.addcmul_(g, g, value=1 - beta2_g if beta2_g != 1.0 else 1.0)
                    # 2. first moment (bias corrected)
                    if beta1 != 0.0:
                        m = bs["m"]
                        m.lerp_(g, 1 - beta1)
                        m_hat = m / bc1 if bc1 is not None else m
                    else:
                        m_hat = g
                    # 3. direction: Adam's, or Shampoo's grafted to Adam's norm
                    adam_dir = m_hat / ((v / bc2_g).sqrt_().add_(eps_g))
                    if graft_only:
                        d = adam_dir
                    else:
                        target = bs["inv_factors"][0].dtype
                        pre = reduce(
                            lambda x, inv: torch.tensordot(x.to(target), inv, dims=([0], [0])), bs["inv_factors"], m_hat
                        ).to(m_hat.dtype)
                        graft_norm, pre_norm = torch._foreach_norm([adam_dir, pre])
                        d = pre * (graft_norm / (pre_norm + 1e-16))
                    # 4. decoupled weight decay folded into the direction, then the step
                    if wd != 0.0:
                        d = d.add(w, alpha=wd)
                    w.add_(d * -lr)
        return loss
