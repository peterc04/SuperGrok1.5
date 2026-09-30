"""Shampoo: bit-exact against Meta's Distributed Shampoo when it is installed, and against the textbook
formulas, Adam before preconditioning starts, per-block grafting, its schedule and its FLOP accounting."""

import copy
import logging
import math

import pytest
import torch
import torch.nn.functional as F
from conftest import batch, max_diff, tiny_mlp
from torch.utils.flop_counter import FlopCounterMode

from grokking_optimizers import OPTIMIZERS, Shampoo, build_optimizer
from grokking_optimizers.shampoo import merge_small_dims, multi_dim_split

REFERENCE_CONFIGS = [
    dict(max_dim=1024, freq=1, start=1, steps=30),  # small matrices merged into vectors: full-matrix AdaGrad
    dict(max_dim=12, freq=1, start=1, steps=30),  # 2-D factors, blocking
    dict(max_dim=12, freq=5, start=10, steps=40),  # Adam first, then amortized roots
    dict(max_dim=1024, freq=3, start=6, steps=40, wd=0.0),
    dict(max_dim=12, freq=2, start=4, steps=30, adam_group=True),  # a group that stays Adam
    dict(max_dim=8, freq=1, start=1, steps=30, betas=(0.0, 0.98)),  # no first moment
    dict(max_dim=12, freq=1, start=1, steps=20, dtype=torch.float64),
]


def _step(model, opt, x, y):
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    opt.step()


@pytest.mark.parametrize("cfg", REFERENCE_CONFIGS, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()))
def test_matches_meta_distributed_shampoo(cfg):
    """facebookresearch/optimizers (the AlgoPerf-winning implementation), single device, Adam grafting."""
    ds = pytest.importorskip("distributed_shampoo")
    logging.getLogger("distributed_shampoo").setLevel(logging.ERROR)
    dtype, betas, wd = cfg.get("dtype", torch.float32), cfg.get("betas", (0.9, 0.98)), cfg.get("wd", 0.5)
    m1 = tiny_mlp().to(dtype)
    m2 = copy.deepcopy(m1)
    x, y = batch()
    x = x.to(dtype)
    hp = dict(
        lr=1e-2,
        betas=betas,
        epsilon=1e-12,
        weight_decay=wd,
        max_preconditioner_dim=cfg["max_dim"],
        precondition_frequency=cfg["freq"],
        start_preconditioning_step=cfg["start"],
    )
    graft = ds.AdamPreconditionerConfig(beta2=betas[1], epsilon=1e-8)
    ref_kw = dict(weight_decay_type=ds.WeightDecayType.DECOUPLED, use_bias_correction=True, grafting_config=graft)
    if cfg.get("adam_group"):
        split = (
            [[p for p in m.parameters() if p.ndim == 2] for m in (m1, m2)],
            [[p for p in m.parameters() if p.ndim == 1] for m in (m1, m2)],
        )
        (mats1, mats2), (vecs1, vecs2) = split
        ours = Shampoo([{"params": mats1}, {"params": vecs1, "use_shampoo": False}], grafting_epsilon=1e-8, **hp)
        adam_only = ds.AdamPreconditionerConfig(beta2=betas[1], epsilon=1e-8)
        ref = ds.DistributedShampoo(
            [{"params": mats2}, {"params": vecs2, "preconditioner_config": adam_only, "grafting_config": None}],
            **ref_kw,
            **hp,
        )
    else:
        ours = Shampoo(m1.parameters(), grafting_epsilon=1e-8, **hp)
        ref = ds.DistributedShampoo(m2.parameters(), **ref_kw, **hp)
    for _ in range(cfg["steps"]):
        _step(m1, ours, x, y)
        _step(m2, ref, x, y)
        assert max_diff(m1, m2) == 0.0


def test_merging_and_blocking_follow_the_reference_examples():
    assert merge_small_dims((1, 2, 5, 1), 10) == (10,)
    assert merge_small_dims((1, 2, 5, 1), 1) == (2, 5)
    assert merge_small_dims((32, 3, 64, 64), 8192) == (96, 4096)
    assert merge_small_dims((32, 3, 64, 64), 1_000_000, 2) == (32, 12_288)
    assert merge_small_dims((1, 1), 4) == (1,) and merge_small_dims((3, 0), 4) == (0,)
    blocks = multi_dim_split(torch.arange(15.0).view(5, 3), 2)
    assert [tuple(b.shape) for b in blocks] == [(2, 2), (2, 1), (2, 2), (2, 1), (1, 2), (1, 1)]
    assert len(multi_dim_split(torch.zeros(5, 3), math.inf)) == 1


def _inv_root(a, p):
    lam, q = torch.linalg.eigh(a)
    return q @ torch.diag(lam.pow(-1.0 / p)) @ q.T


@pytest.mark.parametrize("shape", [(5, 3), (5,)])
def test_one_step_is_the_shampoo_formula(shape):
    """One step from zero state (beta1 = 0): P = (G G^T + eps I)^(-1/4) G (G^T G + eps I)^(-1/4), or
    (g g^T + eps I)^(-1/2) g for a vector, rescaled to the norm of Adam's step, plus decoupled decay."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(*shape, dtype=torch.float64))
    g = torch.randn(*shape, dtype=torch.float64)
    # values exact in fp32: the reference keeps lr and the bias corrections as fp32 scalars
    lr, wd, eps, eps_g = 0.125, 0.25, 1e-6, 1e-8
    opt = Shampoo(
        [w],
        lr=lr,
        betas=(0.0, 0.5),
        epsilon=eps,
        weight_decay=wd,
        max_preconditioner_dim=5,
        precondition_frequency=1,
        grafting_epsilon=eps_g,
        factor_dtype=torch.float64,
    )
    w0 = w.detach().clone()
    w.grad = g.clone()
    opt.step()
    if len(shape) == 2:
        eye_l, eye_r = torch.eye(shape[0], dtype=g.dtype), torch.eye(shape[1], dtype=g.dtype)
        p = _inv_root(g @ g.T + eps * eye_l, 4) @ g @ _inv_root(g.T @ g + eps * eye_r, 4)
    else:
        p = _inv_root(torch.outer(g, g) + eps * torch.eye(shape[0], dtype=g.dtype), 2) @ g
    adam = g / (g.abs() + eps_g)  # v_hat = g^2 after one bias-corrected step
    want = w0 - lr * (p * (adam.norm() / p.norm()) + wd * w0)
    assert torch.allclose(w.detach(), want, rtol=0, atol=1e-10)


def test_before_preconditioning_it_is_adamw():
    x, y = batch()
    m1, m2 = tiny_mlp(), tiny_mlp()
    opt = Shampoo(
        m1.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=1.0,
        precondition_frequency=100,
        start_preconditioning_step=100,
        grafting_epsilon=1e-8,
    )
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, foreach=False)
    for _ in range(30):
        _step(m1, opt, x, y)
        _step(m2, o2, x, y)
    assert opt.upcoming_step_kind() == "adam"
    assert max_diff(m1, m2) < 1e-6  # same algorithm; the reference's operation order differs from torch's


def test_grafting_gives_each_block_adams_step_length():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(10, 6))  # max dim 5: blocks (5,5) (5,1) (5,5) (5,1)
    opt = Shampoo([w], lr=0.01, betas=(0.9, 0.95), max_preconditioner_dim=5, precondition_frequency=1)
    shadow = Shampoo(
        [torch.nn.Parameter(w.detach().clone())],
        lr=0.01,
        betas=(0.9, 0.95),
        max_preconditioner_dim=5,
        precondition_frequency=10**6,
        start_preconditioning_step=10**6,
    )
    for _ in range(3):
        g = torch.randn(10, 6)
        before = w.detach().clone()
        s_param = shadow.param_groups[0]["params"][0]
        s_before = s_param.detach().clone()
        w.grad, s_param.grad = g.clone(), g.clone()
        opt.step()
        shadow.step()  # the same statistics, Adam's direction
        for rows in (slice(0, 5), slice(5, 10)):
            for cols in (slice(0, 5), slice(5, 6)):
                ours = (w.detach() - before)[rows, cols].norm()
                adam = (s_param.detach() - s_before)[rows, cols].norm()
                assert ours == pytest.approx(adam.item(), rel=1e-5)


def test_step_kinds_follow_the_schedule():
    opt = Shampoo([torch.nn.Parameter(torch.ones(3, 3))], precondition_frequency=2, start_preconditioning_step=4)
    kinds = []
    for _ in range(8):
        kinds.append(opt.upcoming_step_kind())
        opt.param_groups[0]["params"][0].grad = torch.randn(3, 3)
        opt.step()
    assert kinds == ["adam", "adam", "adam", "shampoo+root", "shampoo", "shampoo+root", "shampoo", "shampoo+root"]


def test_parameters_without_gradients_are_skipped():
    a, b = torch.nn.Parameter(torch.randn(4, 4)), torch.nn.Parameter(torch.ones(3))
    opt = Shampoo([a, b], precondition_frequency=1)
    a.grad = torch.randn(4, 4)
    opt.step()
    assert torch.equal(b, torch.ones(3)) and b not in opt.state and opt.param_groups[0]["step"] == 1


def test_checkpoint_round_trip_is_exact():
    x, y = batch()
    m1 = tiny_mlp()
    hp = dict(
        lr=1e-2,
        betas=(0.9, 0.98),
        weight_decay=0.5,
        max_preconditioner_dim=12,
        precondition_frequency=3,
        start_preconditioning_step=3,
    )
    o1 = Shampoo(m1.parameters(), **hp)
    for _ in range(7):
        _step(m1, o1, x, y)
    sd, snap = copy.deepcopy(o1.state_dict()), copy.deepcopy(m1.state_dict())
    for _ in range(8):
        _step(m1, o1, x, y)
    m2 = tiny_mlp(seed=5)
    m2.load_state_dict(snap)
    o2 = Shampoo(m2.parameters(), **hp)
    o2.load_state_dict(sd)
    for _ in range(8):
        _step(m2, o2, x, y)
    assert max_diff(m1, m2) == 0.0


def test_root_steps_count_the_eigendecompositions():
    from grokking_race.trainer import EXTRA_FLOP_FORMULAS

    w = torch.nn.Parameter(torch.randn(40, 30))  # 1200 > 1024: stays a matrix, factors 40x40 and 30x30
    opt = Shampoo([w], precondition_frequency=1)
    w.grad = torch.randn(40, 30)
    with FlopCounterMode(display=False, custom_mapping=EXTRA_FLOP_FORMULAS) as fc:
        opt.step()
    counts = fc.get_flop_counts()["Global"]
    assert counts[torch.ops.aten._linalg_eigh] == 9 * (40**3 + 30**3)


def test_race_build_preconditions_the_hidden_matrices_only():
    from deepseek_v41 import build_model

    torch.manual_seed(0)
    model = build_model("tiny", vocab_size=13, max_seq_len=8)
    roles = model.param_roles()
    for policy in ("uniform", "deepseek"):
        opt = build_optimizer("shampoo", model, policy=policy)
        name_of = {p: n for n, p in model.named_parameters()}
        for group in opt.param_groups:
            for p in group["params"]:
                assert group["use_shampoo"] == (roles[name_of[p]].kind == "matrix"), name_of[p]
    assert list(OPTIMIZERS).index("shampoo") == list(OPTIMIZERS).index("supergrok11") - 1
