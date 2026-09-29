"""GrokAdamW, Prodigy, Muon and NeuralGrok against their references."""

import copy

import pytest
import torch
import torch.nn.functional as F
from conftest import batch, max_diff, tiny_mlp
from references import (
    NeuralGrokReference,
    ProdigyReference,
    grokadamw_published_step,
    muon_reference_step,
)
from torch.func import functional_call

from grokking_optimizers import OPTIMIZERS, GrokAdamW, Muon, NeuralGrok, Prodigy, build_optimizer, muon_param_groups
from grokking_optimizers.muon import newton_schulz


def deep_mlp(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)
    )


def _grads(model, x, y):
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()


@pytest.mark.parametrize("losses", [None, (0.5, 1.5)])
def test_grokadamw_matches_published_code(losses):
    x, y = batch()
    m1, m2 = deep_mlp(), deep_mlp()
    hp = dict(lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, alpha_init=0.98, lamb=2.0, gamma=0.1)
    opt = GrokAdamW(m1.parameters(), **hp, grokking_signal_decay_rate=0.1, gradient_clipping=1.0)
    if losses:
        opt.set_losses(*losses)
    state = {}
    for _ in range(100):
        _grads(m1, x, y)
        opt.step()
        _grads(m2, x, y)
        with torch.no_grad():
            grokadamw_published_step(
                list(m2.parameters()),
                state,
                lr=1e-3,
                betas=(0.9, 0.98),
                eps=1e-8,
                wd=1.0,
                alpha_init=0.98,
                lamb=2.0,
                gamma=0.1,
                kappa=0.1,
                clip=1.0,
                train_loss=losses[0] if losses else None,
                eval_loss=losses[1] if losses else None,
            )
    assert max_diff(m1, m2) == 0.0


def test_grokadamw_layer_bias_correction_limits_first_step():
    """The published bias correction inflates a deep tensor's first step by (1-beta1_i)/(1-beta1)."""
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.zeros(4)) for _ in range(20)]
    for p in ps:
        p.grad = torch.ones(4)
    GrokAdamW(ps, lr=1.0, weight_decay=0, gradient_clipping=0, lamb=0.0).step()
    qs = [torch.nn.Parameter(torch.zeros(4)) for _ in range(20)]
    for q in qs:
        q.grad = torch.ones(4)
    GrokAdamW(qs, lr=1.0, weight_decay=0, gradient_clipping=0, lamb=0.0, bias_correction1="layer").step()
    assert abs(qs[-1].abs().max().item() - 1.0) < 1e-4  # corrected: a normal Adam first step (size lr)
    assert ps[-1].abs().max().item() > 5.0  # published: about (1 - 0.9 * 0.9**19) / 0.1 = 9.1x


@pytest.mark.parametrize("wd", [0.0, 1.0])
def test_prodigy_matches_prodigyopt(wd):
    x, y = batch()
    m1, m2 = tiny_mlp(), tiny_mlp()
    opt = Prodigy(m1.parameters(), weight_decay=wd)
    ref = ProdigyReference(m2.parameters(), weight_decay=wd)
    for _ in range(150):
        _grads(m1, x, y)
        opt.step()
        _grads(m2, x, y)
        ref.step()
    assert max_diff(m1, m2) == 0.0
    assert opt.d == ref.d and opt.d > 1e-3  # d grew from 1e-6 to a sensible step size


def test_prodigy_state_dict_round_trip():
    x, y = batch()
    m1 = tiny_mlp()
    opt = Prodigy(m1.parameters(), weight_decay=0.1)
    for _ in range(20):
        _grads(m1, x, y)
        opt.step()
    m2 = copy.deepcopy(m1)
    opt2 = Prodigy(m2.parameters(), weight_decay=0.1)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))  # a checkpoint holds copies, not references
    for m, o in ((m1, opt), (m2, opt2)):
        for _ in range(10):
            _grads(m, x, y)
            o.step()
    assert max_diff(m1, m2) == 0.0 and opt2.d == opt.d


def test_muon_matches_torch_semantics_in_fp32():
    torch.manual_seed(0)
    ws = [torch.nn.Parameter(torch.randn(16, 24) * 0.1), torch.nn.Parameter(torch.randn(32, 8) * 0.1)]
    ref = [w.detach().clone() for w in ws]
    bufs = [torch.zeros_like(w) for w in ref]
    opt = Muon(ws, lr=1e-2, weight_decay=0.1, ns_dtype=None)
    for i in range(50):
        for w, r in zip(ws, ref):
            w.grad = torch.sin(w.detach() * 3 + i)
            r.grad = torch.sin(r * 3 + i)
        opt.step()
        muon_reference_step(ref, bufs, lr=1e-2, wd=0.1, momentum=0.95)
    assert max((w - r).abs().max().item() for w, r in zip(ws, ref)) < 1e-5


def test_muon_one_step_matches_torch_optim_muon_bf16():
    """Same update as torch.optim.Muon(adjust_lr_fn='match_rms_adamw') up to bf16 Newton-Schulz rounding."""
    torch.manual_seed(0)
    w0 = torch.randn(24, 16) * 0.1
    w1, w2 = torch.nn.Parameter(w0.clone()), torch.nn.Parameter(w0.clone())
    g = torch.randn(24, 16)
    w1.grad, w2.grad = g.clone(), g.clone()
    Muon([w1], lr=1e-2, weight_decay=0.1).step()
    torch.optim.Muon([w2], lr=1e-2, weight_decay=0.1, adjust_lr_fn="match_rms_adamw").step()
    d1, d2 = (w1 - w0).detach(), (w2 - w0).detach()
    assert (d1 - d2).norm() / d2.norm() < 0.06  # bf16 NS alone moves the update ~3.5% from fp32


def test_muon_per_block_orthogonalization_and_scale():
    torch.manual_seed(0)
    g = torch.randn(64, 32)
    for blocks in (1, 4):
        p = torch.nn.Parameter(torch.zeros(64, 32))
        p.grad = g.clone()
        Muon([{"params": [p], "row_blocks": blocks}], lr=1.0, weight_decay=0.0, momentum=0.0, ns_dtype=None).step()
        rms = p.detach().pow(2).mean().sqrt().item()
        assert 0.14 < rms < 0.22  # ~update_rms whatever the split (scale uses the block shape)
        o = newton_schulz(g.view(blocks, 64 // blocks, 32), dtype=None)
        assert torch.allclose(p.detach(), -0.2 * max(64 // blocks, 32) ** 0.5 * o.view(64, 32), atol=1e-6)


def test_muon_groups_for_deepseek_follow_param_roles():
    from deepseek_v41 import build_model

    model = build_model("tiny", vocab_size=20, max_seq_len=8)
    roles = model.param_roles()
    groups = muon_param_groups(model, lr=1e-3, weight_decay=0.1, per_head=True)
    placed = {id(p): g for g in groups for p in g["params"]}
    for name, p in model.named_parameters():
        g, role = placed[id(p)], roles[name]
        assert g["use_muon"] == (role.kind == "matrix"), name
        if g["use_muon"]:
            assert g["row_blocks"] == role.blocks * role.head_blocks, name
    cfg = model.cfg
    by_name = dict(model.named_parameters())
    assert placed[id(by_name["layers.0.attn.wq_b.weight"])]["row_blocks"] == cfg.n_heads
    assert placed[id(by_name["layers.0.attn.wo_a.weight"])]["row_blocks"] == cfg.o_groups
    assert not placed[id(by_name["layers.1.engram.q_weight"])]["use_muon"]  # 2-D but an elementwise gain
    off = muon_param_groups(model, lr=1e-3, weight_decay=0.1, per_head=False)
    assert {g.get("row_blocks") for g in off if g["use_muon"]} == {1, cfg.o_groups}  # wo_a split stays


def test_neuralgrok_matches_official_training_step():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(64, 8, generator=g), torch.randint(0, 4, (64,), generator=g)
    xo, yo = torch.randn(16, 8, generator=g), torch.randint(0, 4, (16,), generator=g)
    m1, m2 = tiny_mlp(), tiny_mlp()
    torch.manual_seed(5)
    opt = NeuralGrok(m1.parameters(), amp_hidden_dims=(16, 16), meta_every=2)
    torch.manual_seed(5)
    ref = NeuralGrokReference(m2, copy.deepcopy(opt.amplifier.network), meta_every=2)
    ref.amp_c = 1.0
    names = [n for n, _ in m1.named_parameters()]

    def closure():
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = F.cross_entropy(m1(x), y)
            loss.backward()
        return loss.detach()

    def meta_loss(ps):
        return F.cross_entropy(functional_call(m1, dict(zip(names, ps)), (xo,)), yo)

    for _ in range(8):
        closure()
        opt.step(closure=closure, meta_loss=meta_loss)
        ref.train_step(F.cross_entropy, x, y, xo, yo)
    assert max_diff(m1, m2) < 1e-6
    amp_pairs = zip(opt.amplifier.network.parameters(), ref.amp.parameters())
    assert max((a - b).abs().max().item() for a, b in amp_pairs) < 1e-6


def test_neuralgrok_zero_gradient_tensor_gives_zeros_not_nan():
    p = torch.nn.Parameter(torch.ones(5))
    p.grad = torch.zeros(5)
    opt = NeuralGrok([p], amp_hidden_dims=(4,), weight_decay=0.0)
    opt.step()
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_registry_builds_every_optimizer_for_the_model(name):
    from deepseek_v41 import build_model

    model = build_model("tiny", vocab_size=20, max_seq_len=8, engram_vocab_size=64)
    for policy in ("uniform", "deepseek"):
        opt = build_optimizer(name, model, policy=policy)
        seen = [p for g in opt.param_groups for p in g["params"]]
        assert len(seen) == len(set(map(id, seen))) == len(list(model.parameters()))
