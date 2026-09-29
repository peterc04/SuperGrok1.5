"""Race harness: grokking rule, data, cost accounting, gradient accumulation, end-to-end CLI."""

import json
import os

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from deepseek_v41 import build_model
from grokking_race.__main__ import main
from grokking_race.results import EarlyStopper
from grokking_race.tasks import carve_meta_split, chained_division, modular_division
from grokking_race.trainer import RunConfig, train_one


def test_grok_requires_a_held_streak():
    st = EarlyStopper(threshold=0.95, patience=3)
    seq = [0.1, 0.96, 0.5, 0.97, 0.98, 0.99, 0.99]  # a spike at step 2, then a held streak from step 4
    stopped = [st.update(a, s) for s, a in enumerate(seq, 1)]
    assert st.first_cross_step == 2
    assert st.grok_step == 4 and stopped.index(True) == 5  # confirmed at the 3rd eval of the streak


def test_tasks_shapes_labels_and_meta_split():
    t = modular_division(11, 0.5, 0.1, seed=3)
    assert sum(t.sizes()) == 11 * 10 and t.seq_len == 4 and t.vocab_size == 13
    x = torch.cat([t.x_train, t.x_val, t.x_test])
    y = torch.cat([t.y_train, t.y_val, t.y_test])
    assert torch.equal((x[:, 2] * y) % 11, x[:, 0])  # label * b == a (mod p)
    c = chained_division(11, 3, 0.5, 0.1, seed=3)
    assert c.seq_len == 8 and len(set(map(tuple, c.x_train.tolist()))) == len(c.x_train)
    xi, yi, xm, ym = carve_meta_split(t.x_train, t.y_train, 0.2, seed=0)
    assert len(yi) + len(ym) == len(t.y_train) and len(ym) == round(0.2 * len(t.y_train))


def test_micro_batches_give_the_full_batch_gradient():
    torch.manual_seed(0)
    model = build_model("tiny", vocab_size=13, max_seq_len=4)
    t = modular_division(11, 0.5, 0.1, seed=0)
    grads = []
    for n in (1, 4):
        model.zero_grad()
        for xb, yb in zip(t.x_train.tensor_split(n), t.y_train.tensor_split(n)):
            logits, aux = model(xb, return_aux=True)
            (F.cross_entropy(logits, yb) * len(yb) / len(t.y_train) + aux / n).backward()
        grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    # the balance term is a per-chunk sequence mean, so it matches exactly only for equal chunks
    worst = max((a - b).abs().max().item() for a, b in zip(*grads))
    assert worst < 1e-5


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
    t = modular_division(11, 0.5, 0.1, seed=0)
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
    t = modular_division(11, 0.5, 0.1, seed=0)
    r = train_one(
        "neuralgrok",
        0,
        t,
        _cfg(max_steps=5, eval_every=1, hparams={"neuralgrok": {"amp_hidden_dims": (8,), "meta_every": 2}}),
    )
    kinds = r.flops_per_step_kind
    assert set(kinds) == {"inner", "inner+meta"} and kinds["inner+meta"] > 2 * kinds["inner"]
    expect, acc = [], 0
    for step in range(1, 6):
        acc += kinds["inner+meta" if step % 2 == 0 else "inner"]
        expect.append(acc)
    assert r.eval_flops == expect


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
