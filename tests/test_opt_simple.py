"""AdamW, Lion and Grokfast against their references, bit for bit."""

import torch
import torch.nn.functional as F
from conftest import batch, max_diff, tiny_mlp
from references import gradfilter_ema, lion_automl_step

from grokking_optimizers import AdamW, Grokfast, Lion


def _grads(model, x, y):
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()


def test_adamw_matches_torch_bitwise():
    x, y = batch()
    m1, m2 = tiny_mlp(), tiny_mlp()
    o1 = AdamW(m1.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0, foreach=False)
    for _ in range(200):
        for m, o in ((m1, o1), (m2, o2)):
            _grads(m, x, y)
            o.step()
    assert max_diff(m1, m2) == 0.0
    o2.load_state_dict(o1.state_dict())  # state is torch-compatible


def test_adamw_skips_params_without_grad():
    p, q = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
    opt = AdamW([p, q], lr=0.1, weight_decay=0.5)
    p.grad = torch.ones(3)
    opt.step()
    assert torch.equal(q.detach(), torch.ones(3)) and q not in opt.state
    assert int(opt.state[p]["step"]) == 1


def test_lion_matches_automl_bitwise():
    x, y = batch()
    m1, m2 = tiny_mlp(), tiny_mlp()
    o1 = Lion(m1.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3.0)
    moms = [torch.zeros_like(p) for p in m2.parameters()]
    for _ in range(200):
        _grads(m1, x, y)
        o1.step()
        _grads(m2, x, y)
        with torch.no_grad():
            lion_automl_step(
                list(m2.parameters()), [p.grad for p in m2.parameters()], moms, lr=3e-4, beta1=0.9, beta2=0.99, wd=3.0
            )
    assert max_diff(m1, m2) == 0.0


def test_lion_update_is_sign_sized():
    p = torch.nn.Parameter(torch.zeros(5))
    opt = Lion([p], lr=0.01)
    p.grad = torch.tensor([3.0, -0.001, 0.0, 1e6, -2.0])
    opt.step()
    assert torch.equal(p.detach(), torch.tensor([-0.01, 0.01, 0.0, -0.01, 0.01]))


def test_grokfast_matches_gradfilter_ema_then_adamw_bitwise():
    x, y = batch()
    m1, m2 = tiny_mlp(), tiny_mlp()
    o1 = Grokfast(m1.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0, alpha=0.98, lamb=2.0)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0, foreach=False)
    grads = None
    for _ in range(200):
        _grads(m1, x, y)
        o1.step()
        _grads(m2, x, y)
        grads = gradfilter_ema(m2, grads, alpha=0.98, lamb=2.0)
        o2.step()
    assert max_diff(m1, m2) == 0.0


def test_grokfast_lamb_zero_is_adamw():
    x, y = batch()
    m1, m2 = tiny_mlp(), tiny_mlp()
    o1 = Grokfast(m1.parameters(), lr=1e-3, weight_decay=1.0, lamb=0.0)
    o2 = AdamW(m2.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)
    for _ in range(50):
        for m, o in ((m1, o1), (m2, o2)):
            _grads(m, x, y)
            o.step()
    assert max_diff(m1, m2) == 0.0
