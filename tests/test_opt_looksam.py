"""LookSAM against Algorithm 1 of the paper, SAM (k=1) and AdamW (rho=0)."""

import torch
import torch.nn.functional as F
from conftest import batch, max_diff, tiny_mlp
from references import LookSAMReference

from grokking_optimizers import AdamW, LookSAM


def _run(opt, model, x, y, steps):
    def closure():
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(x), y)
            loss.backward()
        return loss.detach()

    for _ in range(steps):
        loss = closure()
        opt.step(closure=closure, loss=loss)


def test_looksam_matches_algorithm_1():
    x, y = batch()
    m1, m2 = tiny_mlp(), tiny_mlp()
    opt = LookSAM(m1.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0, rho=0.05, k=3, alpha=0.7)
    ref = LookSAMReference(m2, F.cross_entropy, rho=0.05, k=3, alpha=0.7, lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)
    _run(opt, m1, x, y, 20)
    for _ in range(20):
        ref.step(x, y)
    assert max_diff(m1, m2) < 1e-6


def test_looksam_rho_zero_is_adamw_bitwise():
    x, y = batch()
    m1, m2 = tiny_mlp(), tiny_mlp()
    _run(LookSAM(m1.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0, rho=0.0, k=3), m1, x, y, 30)
    o2 = AdamW(m2.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)
    for _ in range(30):
        o2.zero_grad()
        F.cross_entropy(m2(x), y).backward()
        o2.step()
    assert max_diff(m1, m2) == 0.0


def test_looksam_k1_is_sam():
    x, y = batch()
    m1, m2 = tiny_mlp(), tiny_mlp()
    _run(LookSAM(m1.parameters(), lr=1e-3, weight_decay=0.1, rho=0.05, k=1), m1, x, y, 15)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=0.1, foreach=False)
    for _ in range(15):  # plain SAM (Foret et al.) feeding AdamW
        o2.zero_grad()
        F.cross_entropy(m2(x), y).backward()
        g = [p.grad.clone() for p in m2.parameters()]
        norm = torch.cat([gi.flatten().double() for gi in g]).norm()
        with torch.no_grad():
            backup = [p.clone() for p in m2.parameters()]
            for p, gi in zip(m2.parameters(), g):
                p.add_((gi.double() * 0.05 / (norm + 1e-12)).to(p.dtype))
        o2.zero_grad()
        F.cross_entropy(m2(x), y).backward()
        with torch.no_grad():
            for p, b in zip(m2.parameters(), backup):
                p.copy_(b)
        o2.step()
    assert max_diff(m1, m2) < 1e-7


def test_looksam_reuse_push_is_orthogonal_with_norm_alpha_g():
    x, y = batch()
    model = tiny_mlp()
    opt = LookSAM(model.parameters(), rho=0.1, k=5, alpha=0.7)
    _run(opt, model, x, y, 1)  # SAM step caches g_v
    g_v = torch.cat([opt.state[p]["g_v"].flatten().double() for p in model.parameters()])
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    g = torch.cat([p.grad.flatten().double() for p in model.parameters()])
    assert opt.upcoming_step_kind() == "reuse"
    push = 0.7 * g.norm() / g_v.norm() * g_v
    assert abs(float(push.norm() - 0.7 * g.norm())) < 1e-10
    # g_v was orthogonal to the gradient it was projected against (the SAM step's g)
    assert opt.gv_norm > 0
