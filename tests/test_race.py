"""Race harness: grokking rule, data, cost accounting, gradient accumulation, end-to-end CLI."""

import json
import math
import os

import pytest
import torch
import torch.nn.functional as F
from references import original_make_data, original_make_sequential_division_data
from torch.utils.flop_counter import FlopCounterMode

from deepseek_v41 import build_model
from grokking_optimizers import OPTIMIZERS
from grokking_race.__main__ import main
from grokking_race.results import EarlyStopper
from grokking_race.tasks import carve_meta_split, chained_division, make_task, modular_division
from grokking_race.trainer import RunConfig, train_one


def test_grok_requires_a_held_streak():
    st = EarlyStopper(threshold=0.95, patience=3)
    seq = [0.1, 0.96, 0.5, 0.97, 0.98, 0.99, 0.99]  # a spike at step 2, then a held streak from step 4
    stopped = [st.update(a, s) for s, a in enumerate(seq, 1)]
    assert st.first_cross_step == 2
    assert st.grok_step == 4 and stopped.index(True) == 5  # confirmed at the 3rd eval of the streak


def test_tasks_shapes_labels_and_meta_split():
    t = modular_division(11, 0.5, seed=3)
    assert sum(t.sizes()) == 11 * 10 and t.seq_len == 4 and t.vocab_size == 13
    x = torch.cat([t.x_train, t.x_test])
    y = torch.cat([t.y_train, t.y_test])
    assert torch.equal((x[:, 2] * y) % 11, x[:, 0])  # label * b == a (mod p)
    c = chained_division(11, 3, 0.5, seed=3)
    assert c.seq_len == 8 and len(set(map(tuple, c.x_train.tolist()))) == len(c.x_train)
    xi, yi, xm, ym = carve_meta_split(t.x_train, t.y_train, 0.2, seed=0)
    assert len(yi) + len(ym) == len(t.y_train) and len(ym) == round(0.2 * len(t.y_train))


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("frac", [0.1, 0.25, 0.5])
def test_splits_are_the_original_scripts_train_test_split(seed, frac):
    """Same examples, order and train/test cut as make_data / make_sequential_division_data of the original
    race scripts (no validation split)."""
    for task, ref in (
        ("moddiv", original_make_data(97, frac, seed)),
        ("chaindiv", original_make_sequential_division_data(97, 3, frac, seed)),
    ):
        t = make_task(task, 97, frac, seed)
        for got, want in zip((t.x_train, t.y_train, t.x_test, t.y_test), ref):
            assert torch.equal(got, want), task


def test_meta_slice_comes_out_of_train_only():
    t = modular_division(23, 0.5, seed=42)
    xi, yi, xm, ym = carve_meta_split(t.x_train, t.y_train, 0.1, seed=42)
    train = set(map(tuple, t.x_train.tolist()))
    test = set(map(tuple, t.x_test.tolist()))
    inner, meta = set(map(tuple, xi.tolist())), set(map(tuple, xm.tolist()))
    assert inner | meta == train and not inner & meta and not (inner | meta) & test


def test_micro_batches_give_the_full_batch_gradient():
    """The trainer's chunk weighting: CE and the balance loss are both per-sequence means, so weighting each
    chunk by its size reproduces the full batch exactly, also for unequal chunks (55 examples in 4)."""
    torch.manual_seed(0)
    model = build_model("tiny", vocab_size=13, max_seq_len=4, balance_loss_alpha=1.0)  # make the aux term count
    t = modular_division(11, 0.5, seed=0)
    grads = []
    for n in (1, 4):
        model.zero_grad()
        for xb, yb in zip(t.x_train.tensor_split(n), t.y_train.tensor_split(n)):
            logits, aux = model(xb, return_aux=True)
            ((F.cross_entropy(logits, yb) + aux) * (len(yb) / len(t.y_train))).backward()
        grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    worst = max((a - b).abs().max().item() for a, b in zip(*grads))
    assert worst < 1e-6


def test_router_stats_frozen_inside_optimizer_forwards():
    model = build_model("tiny", vocab_size=13, max_seq_len=4)
    x = torch.randint(0, 13, (8, 4))
    with model.frozen_router_stats():
        model(x).sum().backward()
    assert all(float(layer.ffn.gate.load.sum()) == 0 for layer in model.layers)
    model(x).sum().backward()
    assert all(float(layer.ffn.gate.load.sum()) > 0 for layer in model.layers)


def _cfg(**kw):
    base = dict(max_steps=4, eval_every=2, patience=2, progress=False)
    base.update(kw)
    return RunConfig(**base)


def test_flop_accounting_matches_a_direct_count():
    t = modular_division(11, 0.5, seed=0)
    r = train_one("adamw", 0, t, _cfg())
    torch.manual_seed(0)
    model = build_model("tiny", vocab_size=13, max_seq_len=4)
    with FlopCounterMode(display=False) as fc:
        logits, aux = model(t.x_train, return_aux=True)
        (F.cross_entropy(logits, t.y_train) + aux).backward()
    assert r.flops_per_step_kind == {"step": fc.get_total_flops()}
    assert r.eval_flops == [s * fc.get_total_flops() for s in r.steps]
    assert r.train_flops == r.total_steps * fc.get_total_flops()
    assert all(b > a for a, b in zip(r.eval_train_time, r.eval_train_time[1:]))


def test_flop_accounting_per_step_kind():
    t = modular_division(11, 0.5, seed=0)
    r = train_one(
        "neuralgrok",
        0,
        t,
        _cfg(max_steps=5, eval_every=1, hparams={"neuralgrok": {"amp_hidden_dims": (8,), "meta_every": 2}}),
    )
    kinds = r.flops_per_step_kind
    assert set(kinds) == {"inner", "inner+meta"} and kinds["inner+meta"] > 2 * kinds["inner"]
    # every iteration charged exactly what counting it would give (routing can differ between meta steps)
    exact = train_one(
        "neuralgrok",
        0,
        t,
        _cfg(
            max_steps=5,
            eval_every=1,
            flop_count_every_step=True,
            hparams={"neuralgrok": {"amp_hidden_dims": (8,), "meta_every": 2}},
        ),
    )
    assert r.eval_flops == exact.eval_flops or r.flops_approx_steps > 0


def test_every_optimizer_trains_end_to_end(tmp_path):
    main(
        [
            "--optimizers",
            "all",
            "--seeds",
            "0",
            "--p",
            "11",
            "--max-steps",
            "3",
            "--eval-every",
            "1",
            "--patience",
            "2",
            "--output",
            str(tmp_path),
            "--quiet",
            "--hparams",
            json.dumps({"neuralgrok": {"amp_hidden_dims": [8], "meta_every": 2}}),
        ]
    )
    out = tmp_path / "moddiv_ft50_tiny"
    data = json.loads((out / "results.json").read_text())
    from grokking_optimizers import OPTIMIZERS

    for name in OPTIMIZERS:
        run = data[name][0]
        assert run["error"] is None, (name, run["error"])
        assert run["total_steps"] == 3 and run["train_flops"] > 0 and len(run["test_accs"]) == 3
    for kind in ("race", "test_acc", "detail", "loss"):
        assert os.path.getsize(out / f"{kind}_moddiv_ft50_tiny.png") > 10_000


def test_censored_median_counts_dnf_seeds_as_never():
    from grokking_race.results import RunResult, censored_median

    def run(step):
        r = RunResult(optimizer="x", seed=0, task="t", frac_train=0.5)
        r.grokked, r.grok_step = step is not None, step
        return r

    assert censored_median([run(1000), run(None), run(2000)], "grok_step") == 2000
    assert censored_median([run(1000), run(None), run(None)], "grok_step") is None  # its luckiest seed is not enough
    assert censored_median([run(1000), run(3000)], "grok_step") == 2000
    assert censored_median([run(1000), run(None)], "grok_step") is None


def test_same_train_data_and_own_train_split():
    t = modular_division(11, 0.5, seed=0)
    plain = train_one("adamw", 0, t, _cfg())
    matched = train_one("adamw", 0, t, _cfg(same_train_data=True))
    meta = train_one("grokadamw", 0, t, _cfg())
    assert plain.train_examples == len(t.y_train) and plain.meta_examples == 0
    assert matched.train_examples == meta.train_examples == len(t.y_train) - meta.meta_examples
    assert meta.meta_examples == round(0.1 * len(t.y_train))


def test_bf16_autocast_runs_on_cpu():
    t = modular_division(11, 0.5, seed=0)
    r = train_one("adamw", 0, t, _cfg(dtype="bf16"))
    assert r.error is None and r.total_steps == 4 and all(math.isfinite(x) for x in r.train_losses)


# small settings that reach every kind of step within a few iterations
QUICK = {
    "neuralgrok": {"amp_hidden_dims": (8,), "meta_every": 2},
    "supergrok11": {"warmup_steps": 1, "warmup_ramp": 1, "meta_update_freq": 2, "sam_every": 3},
    "shampoo": {"precondition_frequency": 2},
    "looksam": {"k": 2},
}


@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_flop_accounting_equals_counting_every_iteration(name):
    """At p = 11 some MoE experts get no tokens on some steps. That never changes the model's FLOPs, but it
    changes the work of optimizers that declare flops_depend_on_routing. The race's accounting must equal
    counting every iteration, or say where it could not."""
    t = modular_division(11, 0.5, seed=0)
    kw = dict(max_steps=6, eval_every=1, patience=100, hparams=QUICK)
    fast = train_one(name, 0, t, _cfg(**kw))
    exact = train_one(name, 0, t, _cfg(**kw, flop_count_every_step=True))
    assert fast.test_accs == exact.test_accs  # counting does not change the numbers
    assert fast.eval_flops == exact.eval_flops or fast.flops_approx_steps > 0
    if not getattr(OPTIMIZERS[name].cls, "flops_depend_on_routing", False):
        assert fast.eval_flops == exact.eval_flops and fast.flops_approx_steps == 0
