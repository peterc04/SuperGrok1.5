"""SuperGrok 1.1: the full algorithm against a naive transcription of its declared math, every component
shown to act on the update, the exact reductions to AdamW, and the training-loop contract."""

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from conftest import batch, max_diff, tiny_mlp
from references import SuperGrok11Reference
from torch.func import functional_call

from grokking_optimizers import OPTIMIZERS, SharpnessMetaNet, SuperGrok11, block_layer_ids, build_optimizer
from grokking_optimizers.supergrok11 import cosine

# Small enough cadences that every component acts within 30 steps (fp64 so the comparison is exact to rounding).
FULL = dict(
    lr=1e-2,
    betas=(0.9, 0.98),
    eps=1e-8,
    weight_decay=0.5,
    alpha_init=0.98,
    lamb=5.0,
    gamma=0.2,
    kappa=0.5,
    warmup_steps=3,
    warmup_ramp=4,
    gradient_clipping=0.05,
    gate_temperature=5.0,
    alpha_update_freq=3,
    sam_rho=0.05,
    sam_every=3,
    meta_update_freq=2,
    meta_lr=1e-2,
)


def data():
    x, y = batch()
    xo, yo = batch(seed=7)
    return x.double(), y, xo.double(), yo


def losses_at(model, x, y, xo, yo):
    with torch.no_grad():
        logits = model(x)
        return (
            F.cross_entropy(logits, y).item(),
            F.cross_entropy(model(xo), yo).item(),
            (logits.argmax(-1) == y).float().mean().item(),
        )


def meta_losses(model, x, y, xo, yo):
    """(held-out loss, training loss) as functions of substituted parameters."""
    names = [n for n, _ in model.named_parameters()]
    return (
        lambda ps: F.cross_entropy(functional_call(model, dict(zip(names, ps)), (xo,)), yo),
        lambda ps: F.cross_entropy(functional_call(model, dict(zip(names, ps)), (x,)), y),
    )


def run(model, opt, x, y, xo, yo, steps, feed_losses=True):
    names = [n for n, _ in model.named_parameters()]

    def closure():
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(x), y)
            loss.backward()
        return loss.detach()

    def meta_loss(params):
        return F.cross_entropy(functional_call(model, dict(zip(names, params)), (xo,)), yo)

    def train_meta_loss(params):
        return F.cross_entropy(functional_call(model, dict(zip(names, params)), (x,)), y)

    for _ in range(steps):
        if feed_losses:
            opt.set_losses(*losses_at(model, x, y, xo, yo))
        closure()
        opt.step(closure=closure, meta_loss=meta_loss, train_meta_loss=train_meta_loss)


def make(overrides=None, seed=0):
    model = tiny_mlp().double()
    torch.manual_seed(seed)
    net = SharpnessMetaNet(32).double()
    hp = {**FULL, **(overrides or {})}
    return model, SuperGrok11(model.parameters(), meta_net=net, **hp), net


def test_matches_declared_algorithm_with_every_component_on():
    x, y, xo, yo = data()
    m1, opt, net = make()
    m2 = tiny_mlp().double()
    ref_net = copy.deepcopy(net)
    ref = SuperGrok11Reference(
        m2,
        ref_net,
        lr=1e-2,
        betas=(0.9, 0.98),
        eps=1e-8,
        wd=0.5,
        alpha_init=0.98,
        lamb=5.0,
        gamma=0.2,
        kappa=0.5,
        warmup=3,
        ramp_len=4,
        clip=0.05,
        temperature=5.0,
        alpha_every=3,
        rho=0.05,
        sam_every=3,
        meta_every=2,
        meta_lr=1e-2,
        meta_betas=(0.9, 0.999),
    )
    run(m1, opt, x, y, xo, yo, 30)
    for _ in range(30):
        ref.step(F.cross_entropy, x, y, xo, yo, losses=losses_at(m2, x, y, xo, yo))
    assert max_diff(m1, m2) < 1e-12
    assert max((a - b).abs().max().item() for a, b in zip(net.parameters(), ref_net.parameters())) < 1e-12
    assert opt.alpha == pytest.approx(ref.alpha, rel=1e-12)
    assert max((opt.state[p]["sharpness"] - s).abs().max().item() for p, s in zip(m1.parameters(), ref.s)) < 1e-12
    # and the run is not AdamW in disguise: the learned correction is a sizeable part of the update
    d = opt.diagnostics()
    assert abs(d["rescale"]) > 1e-3 and d["correction_ratio"] > 0.05 and 0 < d["gate_mean"] < 1


# Each component switched off in turn must change the trajectory (it is live), relative to the full run.
# Measured effect (max parameter difference / max movement, 30 steps): correction 0.12, meta step 0.12, train
# term 0.034, gate 1.4, ramp 1e-3, layer-wise beta1 0.64, clip 0.16, adaptive alpha 0.10, and sharpness only
# 1.9e-5: phi sees s ~ 3e-4 through near-linear weights of size ~0.01, so its second input barely registers.
WEAK = {"sharpness input (no SAM probe)": 1e-6}
ABLATIONS = {
    "learned correction (lamb=0)": dict(lamb=0.0),
    "meta step (never trained)": dict(meta_update_freq=0),
    "train term of the meta objective": dict(meta_objective="lookahead_val"),
    "sharpness input (no SAM probe)": dict(sam_rho=0.0),
    "cosine gate": dict(gate_mode="none"),
    "warm-up ramp": dict(warmup_ramp=1),
    "layer-wise beta1": dict(gamma=0.0),
    "per-tensor clip": dict(gradient_clipping=0.0),
}


@pytest.mark.parametrize("name", list(ABLATIONS) + ["adaptive alpha"])
def test_every_component_acts_on_the_update(name):
    x, y, xo, yo = data()
    m_full, opt_full, _ = make()
    run(m_full, opt_full, x, y, xo, yo, 30)
    moved = max((a - b).abs().max().item() for a, b in zip(m_full.parameters(), tiny_mlp().double().parameters()))
    m_abl, opt_abl, _ = make(ABLATIONS.get(name))
    run(m_abl, opt_abl, x, y, xo, yo, 30, feed_losses=name != "adaptive alpha")
    assert max_diff(m_full, m_abl) > WEAK.get(name, 5e-4) * moved, name


def test_frozen_meta_no_clip_no_layerwise_is_adamw_bitwise():
    x, y = batch()
    m1, m2 = tiny_mlp(), tiny_mlp()
    opt = SuperGrok11(
        m1.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=1.0,
        gamma=0.0,
        gradient_clipping=0.0,
        meta_update_freq=0,
        sam_rho=0.0,
        warmup_steps=0,
    )
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0, foreach=False)
    for _ in range(50):
        for m, o in ((m1, opt), (m2, o2)):
            o.zero_grad()
            F.cross_entropy(m(x), y).backward()
            o.step()
    assert opt.meta_net.rescale.item() == 0.0  # r = 0: the correction is exactly zero, the gate multiplies zero
    assert max_diff(m1, m2) == 0.0


def test_frozen_control_is_adamw_with_blockwise_beta1_and_clip():
    x, y = batch()
    model = tiny_mlp()
    ref = copy.deepcopy(model)
    hp = {**OPTIMIZERS["supergrok11_frozen"].race_defaults, "layer_ids": [0, 0, 1, 1]}
    opt = SuperGrok11(model.parameters(), **hp)
    assert opt.lamb == 0 and opt.meta_update_freq == 0 and opt.sam_rho == 0
    state = [{"m": torch.zeros_like(p), "v": torch.zeros_like(p)} for p in ref.parameters()]
    for t in range(1, 21):
        for m in (model, ref):
            m.zero_grad()
            F.cross_entropy(m(x), y).backward()
        opt.step()
        with torch.no_grad():
            for i, p in enumerate(ref.parameters()):
                g = p.grad * min(1.0, 1.0 / (p.grad.norm().item() + 1e-6))
                b1 = 0.9 * 0.9 ** [0, 0, 1, 1][i]
                st = state[i]
                st["m"] = b1 * st["m"] + (1 - b1) * g
                st["v"] = 0.98 * st["v"] + 0.02 * g * g
                p.mul_(1 - 1e-3 * 1.0)
                p.sub_(1e-3 * (st["m"] / (1 - b1**t)) / ((st["v"] / (1 - 0.98**t)).sqrt() + 1e-8))
    assert max_diff(model, ref) < 1e-6


def test_ramp_and_layer_alpha_schedule():
    opt = SuperGrok11(
        [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)],
        warmup_steps=100,
        warmup_ramp=100,
        gamma_alpha=0.5,
        meta_update_freq=0,
    )
    assert [opt.ramp(t) for t in (1, 100, 101, 150, 200, 500)] == [0.0, 0.0, 0.01, 0.5, 1.0, 1.0]
    opt.alpha = 0.8
    assert [opt.layer_alpha(i) for i in range(3)] == pytest.approx([0.2, 0.4, 0.8])  # (1-0.5)^(n-1-i)
    opt.alpha = 3.0
    assert opt.layer_alpha(2) == 1.0  # clamped to [0, 1]


@pytest.mark.parametrize(
    "losses,expected",
    [
        ((0.5, 1.5, 0.3), 0.98 * math.exp(-0.1 * 2.0)),  # gap signal (val - train) / train
        ((0.5, 0.2, 0.3), 0.98),  # no gap
        ((0.5, 1.5, 0.995), 0.98 * math.exp(-1.0)),  # memorized by accuracy: signal 10
        ((5e-5, 1.5, 0.3), 0.98 * math.exp(-1.0)),  # memorized by loss
    ],
)
def test_adaptive_alpha_table_and_cadence(losses, expected):
    opt = SuperGrok11([torch.nn.Parameter(torch.ones(3))], alpha_update_freq=50, meta_update_freq=0, sam_rho=0.0)
    opt.update_alpha(*losses)
    assert opt.alpha == pytest.approx(expected, rel=1e-12)
    p = opt.param_groups[0]["params"][0]
    opt2 = SuperGrok11([p], alpha_update_freq=50, meta_update_freq=0, sam_rho=0.0)
    opt2.set_losses(*losses)
    seen = []
    for t in range(1, 101):
        p.grad = torch.ones(3)
        before = opt2.alpha
        opt2.alpha = 0.0  # mark: a refresh overwrites it
        opt2.step()
        if opt2.alpha != 0.0:
            seen.append(t)
        else:
            opt2.alpha = before
    assert seen == [1, 50, 100]


def test_gate_values():
    g = torch.randn(1000, dtype=torch.float64)
    assert cosine(g, torch.zeros_like(g)).item() == 0.0  # first step: m = 0, gate = 1 - sigmoid(0) = 0.5
    for scale in (1.0, 1e-6, 1e-12):
        assert cosine(scale * g, scale * g).item() == pytest.approx(1.0, abs=1e-12)
        assert cosine(scale * g, -scale * g).item() == pytest.approx(-1.0, abs=1e-12)
    # the legacy kernel's floor, sqrt(|g|^2 |m|^2 + 1e-12): aligned tiny gradients read as unaligned
    assert cosine(1e-6 * g, 1e-6 * g, eps=1e-12).item() < 0.01

    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    opt = SuperGrok11(
        [p],
        warmup_steps=0,
        warmup_ramp=1,
        meta_update_freq=0,
        sam_rho=0.0,
        gamma=0.0,
        gradient_clipping=0.0,
        meta_net=SharpnessMetaNet().double(),
    )
    p.grad = torch.tensor([1.0, -2.0, 3.0, 0.5], dtype=torch.float64)
    opt.step()
    assert opt.diagnostics()["gate_mean"] == 0.5
    opt.step()  # same gradient again: aligned with the momentum
    assert opt.diagnostics()["gate_mean"] == pytest.approx(1 - 1 / (1 + math.exp(-5.0)), rel=1e-12)


def test_meta_step_uses_each_groups_lr_and_wd():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float64))
    b = torch.nn.Parameter(torch.randn(4, dtype=torch.float64))
    net = SharpnessMetaNet().double()
    with torch.no_grad():
        net.rescale.fill_(0.3)
        net.net[2].bias.fill_(0.1)
    groups = [{"params": [a], "lr": 0.1, "weight_decay": 0.5}, {"params": [b], "lr": 0.02, "weight_decay": 0.0}]
    opt = SuperGrok11(groups, meta_net=net, gradient_clipping=0.0, meta_objective="lookahead_val", meta_lr=0.0)
    a.grad, b.grad = torch.randn_like(a), torch.randn_like(b)
    seen = {}

    def meta_loss(params):
        seen["virtual"] = [q.detach().clone() for q in params]
        return sum((q**2).sum() for q in params)

    opt.meta_step(meta_loss)
    with torch.no_grad():
        want_a = a * (1 - 0.1 * 0.5) - 0.1 * net(a.grad, torch.zeros_like(a))
        want_b = b * (1 - 0.02 * 0.0) - 0.02 * net(b.grad, torch.zeros_like(b))
    assert torch.allclose(seen["virtual"][0], want_a, rtol=0, atol=1e-15)
    assert torch.allclose(seen["virtual"][1], want_b, rtol=0, atol=1e-15)


def test_first_order_meta_gradient_matches_exact():
    x, y, xo, yo = data()
    grads = {}
    for mode in ("exact", "first_order"):
        model, opt, net = make(dict(meta_grad=mode, meta_lr=0.0, lr=1e-3))  # lr 0: the meta grads stay in .grad
        with torch.no_grad():
            net.rescale.fill_(0.05)
            net.net[2].bias.fill_(0.02)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.meta_step(*meta_losses(model, x, y, xo, yo))
        grads[mode] = torch.cat([q.grad.reshape(-1) for q in net.parameters()])
    e, f = grads["exact"], grads["first_order"]
    assert F.cosine_similarity(e, f, dim=0).item() > 0.9999
    assert (e - f).norm() / e.norm() < 1e-2


def test_first_order_chunking_is_exact():
    x, y, xo, yo = data()
    out = []
    for chunk in (None, 7):
        model, opt, net = make(dict(meta_grad="first_order", meta_lr=0.0, meta_chunk=chunk))
        with torch.no_grad():
            net.rescale.fill_(0.05)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.meta_step(*meta_losses(model, x, y, xo, yo))
        out.append(torch.cat([q.grad.reshape(-1) for q in net.parameters()]))
    assert torch.allclose(out[0], out[1], rtol=1e-10, atol=1e-18)


def test_sam_probe_restores_parameters_and_gradients():
    x, y = batch()
    model = tiny_mlp()
    extra = torch.nn.Parameter(torch.ones(3))  # no gradient at w
    opt = SuperGrok11(list(model.parameters()) + [extra], meta_update_freq=0, sam_rho=0.5, sam_every=1)

    def closure():
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(x), y) + 0.0 * extra.sum()  # the perturbed pass does touch it
            loss.backward()
        return loss

    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    before = [p.detach().clone() for p in model.parameters()]
    grads = [p.grad.clone() for p in model.parameters()]
    opt.sam_probe(closure)
    assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before))
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), grads))
    assert extra.grad is None
    assert all(opt.state[p]["sharpness"].abs().sum() > 0 for p in model.parameters())


def test_zero_gradient_elements_drift_unless_masked():
    x, y = batch()
    out = {}
    for policy in ("apply", "mask"):
        model = tiny_mlp()
        idle = torch.nn.Parameter(torch.zeros(5))  # gradient exactly zero, like an unused embedding row
        net = SharpnessMetaNet()
        with torch.no_grad():
            net.rescale.fill_(0.1)
            net.net[2].bias.fill_(0.1)  # phi(0, 0) = b2 != 0: a learned constant
        opt = SuperGrok11(
            [{"params": list(model.parameters())}, {"params": [idle], "weight_decay": 0.0}],
            meta_net=net,
            warmup_steps=0,
            warmup_ramp=1,
            meta_update_freq=0,
            sam_rho=0.0,
            zero_grad_policy=policy,
        )
        for _ in range(10):
            opt.zero_grad()
            (F.cross_entropy(model(x), y) + 0.0 * idle.sum()).backward()
            opt.step()
        out[policy] = idle.detach().abs().max().item()
    assert out["mask"] == 0.0 and out["apply"] > 1e-3


def test_max_correction_ratio_bounds_the_correction():
    x, y = batch()
    model = tiny_mlp()
    net = SharpnessMetaNet()
    with torch.no_grad():
        net.rescale.fill_(10.0)
        net.net[2].bias.fill_(10.0)
    opt = SuperGrok11(
        model.parameters(),
        meta_net=net,
        warmup_steps=0,
        warmup_ramp=1,
        meta_update_freq=0,
        sam_rho=0.0,
        gate_mode="none",
        max_correction_ratio=0.5,
    )
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    opt.step()
    assert opt.diagnostics()["correction_ratio"] <= 0.5 + 1e-6


def test_contract_errors_and_untouched_params():
    x, y = batch()
    model = tiny_mlp()
    frozen = torch.nn.Parameter(torch.ones(2))
    opt = SuperGrok11(list(model.parameters()) + [frozen], meta_update_freq=2, sam_every=3)
    F.cross_entropy(model(x), y).backward()
    with pytest.raises(RuntimeError, match="closure"):
        opt.step()  # t = 1 probes sharpness
    opt.global_step = 1
    with pytest.raises(RuntimeError, match="meta_loss"):
        opt.step()  # t = 2 is a meta step
    with pytest.raises(RuntimeError, match="train_meta_loss"):
        opt.meta_step(lambda ps: sum(q.sum() for q in ps))  # default objective has the train term
    opt.global_step = 4
    opt.step()  # t = 5: neither due
    assert torch.equal(frozen, torch.ones(2)) and frozen not in opt.state
    with pytest.raises(ValueError):
        SuperGrok11(model.parameters(), gate_mode="bogus")
    with pytest.raises(ValueError):
        SuperGrok11(model.parameters(), layer_ids=[0])


def test_step_kinds_follow_the_schedule():
    opt = SuperGrok11([torch.nn.Parameter(torch.ones(2))], warmup_steps=5, meta_update_freq=5, sam_every=10)
    kinds = []
    for t in range(1, 21):
        kinds.append(opt.upcoming_step_kind())
        opt.global_step = t
    assert kinds[0] == "plain+sam" and kinds[4] == "plain+meta" and kinds[5] == "corrected"
    assert kinds[9] == "corrected+meta+sam" and kinds[14] == "corrected+meta" and kinds[19] == "corrected+meta+sam"


def test_checkpoint_round_trip_is_exact():
    x, y, xo, yo = data()
    m1, o1, _ = make()
    run(m1, o1, x, y, xo, yo, 7)
    sd, snap = copy.deepcopy(o1.state_dict()), copy.deepcopy(m1.state_dict())
    run(m1, o1, x, y, xo, yo, 8)
    m2, o2, _ = make(seed=123)  # different meta-net init: everything must come from the checkpoint
    m2.load_state_dict(snap)
    o2.load_state_dict(sd)
    run(m2, o2, x, y, xo, yo, 8)
    assert max_diff(m1, m2) == 0.0


def test_block_layer_ids():
    names = [
        "embed.weight",
        "layers.0.attn.w",
        "layers.0.ffn.w",
        "layers.1.attn.w",
        "layers.10.x",
        "norm.weight",
        "head.weight",
    ]
    assert block_layer_ids(names) == [0, 1, 1, 2, 11, 12, 12]


def test_registry_builds_blockwise_on_deepseek():
    from deepseek_v41 import build_model

    torch.manual_seed(0)
    model = build_model("tiny", vocab_size=13, max_seq_len=8)
    for policy in ("uniform", "deepseek"):
        opt = build_optimizer("supergrok11", model, policy=policy)
        name_of = {p: n for n, p in model.named_parameters()}
        ids = {name_of[p]: opt._layer[p] for p in opt._layer}
        assert ids["embed.weight"] == 0 and ids["layers.0.attn_norm.weight"] == 1
        assert ids["layers.5.ffn_norm.weight"] == 6 and ids["head.weight"] == 7


def test_virtual_step_masks_zero_gradients_like_the_real_step():
    idle = torch.nn.Parameter(torch.randn(5, dtype=torch.float64))
    net = SharpnessMetaNet().double()
    with torch.no_grad():
        net.rescale.fill_(0.3)
        net.net[2].bias.fill_(0.1)  # phi(0, 0) != 0
    opt = SuperGrok11([idle], meta_net=net, zero_grad_policy="mask", meta_objective="lookahead_val", meta_lr=0.0)
    idle.grad = torch.zeros_like(idle)
    seen = {}

    def meta_loss(params):
        seen["v"] = params[0].detach().clone()
        return (params[0] ** 2).sum()

    opt.meta_step(meta_loss)
    assert torch.equal(seen["v"], idle.detach() * (1 - 1e-3 * 1.0))  # decay only: no correction where g == 0


def test_load_state_dict_does_not_consume_the_checkpoint():
    from grokking_optimizers import LookSAM, NeuralGrok

    for cls in (LookSAM, NeuralGrok, SuperGrok11):
        opt = cls(tiny_mlp().parameters())
        sd = opt.state_dict()
        keys = set(sd)
        cls(tiny_mlp().parameters()).load_state_dict(sd)
        assert set(sd) == keys, cls.__name__


def test_tensor_layer_ids_follow_model_order_under_grouping():
    from deepseek_v41 import build_model

    torch.manual_seed(0)
    model = build_model("tiny", vocab_size=13, max_seq_len=8)
    opt = build_optimizer("supergrok11", model, policy="deepseek", layer_ids="tensor")
    order = {p: i for i, (_, p) in enumerate(model.named_parameters())}
    assert all(opt._layer[p] == order[p] for p in opt._layer)


@pytest.mark.parametrize("meta_grad", ["exact", "first_order"])
def test_correction_cap_keeps_the_first_meta_step_finite(meta_grad):
    """fp32, default init (r = 0, so the correction is exactly zero): the cap must not NaN the meta-gradient."""
    x, y = batch()
    model = tiny_mlp()
    xo, yo = batch(seed=7)
    opt = SuperGrok11(
        model.parameters(),
        max_correction_ratio=1.0,
        meta_update_freq=1,
        meta_grad=meta_grad,
        sam_rho=0.0,
        warmup_steps=0,
    )
    for _ in range(3):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step(meta_loss=meta_losses(model, x, y, xo, yo)[0], train_meta_loss=meta_losses(model, x, y, xo, yo)[1])
    assert all(torch.isfinite(q).all() for q in opt.meta_net.parameters())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert opt.meta_net.rescale.item() != 0.0  # it did learn
