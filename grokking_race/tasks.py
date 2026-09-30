"""Grokking tasks: token sequences for a causal LM, answer read at the last position.

Two splits, as in the original race scripts: **train** (the first
``frac_train`` of a seeded shuffle) and **test** (everything else). There is no
validation split: grokking is judged on test accuracy, and nothing that trains
ever sees test data. Optimizers that need held-out data for themselves carve it
out of their own training split (:func:`carve_meta_split`).

Both builders are ports of ``make_data`` and ``make_sequential_division_data``
in the original race scripts (v1 and v2, identical there). For the same
``(p, frac_train, seed)`` they produce the same examples in the same order with
the same train/test split (tests). The later repository driver (commit 19c9d39)
carved a validation split out of train; that is gone.

Vocabulary: numbers ``0..p-1``, then ``op = p`` and ``eq = p + 1``. The model
predicts the answer from its logits at the final ``eq`` position; accuracy is
argmax over the first ``p`` logits (the number tokens).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

TASKS = ("moddiv", "chaindiv")


@dataclass
class Task:
    name: str
    p: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return self.p + 2

    @property
    def num_classes(self) -> int:
        return self.p

    @property
    def seq_len(self) -> int:
        return self.x_train.shape[1]

    def sizes(self) -> tuple[int, int]:
        return len(self.y_train), len(self.y_test)

    def to(self, device) -> "Task":
        t = {k: getattr(self, k).to(device) for k in ("x_train", "y_train", "x_test", "y_test")}
        return Task(self.name, self.p, **t)


def _split(pairs, labels, frac_train, rng):
    combined = list(zip(pairs, labels))
    rng.shuffle(combined)
    pairs, labels = zip(*combined)
    n = int(len(pairs) * frac_train)
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return x[:n], y[:n], x[n:], y[n:]


def modular_division(p: int = 97, frac_train: float = 0.5, seed: int = 42) -> Task:
    """All ``(a, b)`` with ``b != 0``: input ``[a, op, b, eq]``, label ``a * b^-1 mod p``."""
    rng = random.Random(seed)
    op_tok, eq_tok = p, p + 1
    pairs, labels = [], []
    for a in range(p):
        for b in range(1, p):
            pairs.append([a, op_tok, b, eq_tok])
            labels.append((a * pow(b, p - 2, p)) % p)
    return Task("moddiv", p, *_split(pairs, labels, frac_train, rng))


def chained_division(p: int = 97, chain_length: int = 3, frac_train: float = 0.5, seed: int = 42) -> Task:
    """``p * (p - 1)`` distinct random chains ``[a, op, b1, op, b2, ..., eq]``, label ``a / b1 / b2 / ... mod p``.

    The chain space (``p * (p-1)^chain_length``) is far larger than the sample, so
    unlike modular division this is not a closed "all pairs" task.
    """
    rng = random.Random(seed)
    op_tok, eq_tok = p, p + 1
    target_size = p * (p - 1)
    seen = set()
    pairs, labels = [], []
    while len(pairs) < target_size:
        a = rng.randint(0, p - 1)
        bs = tuple(rng.randint(1, p - 1) for _ in range(chain_length))
        key = (a, *bs)
        if key in seen:
            continue
        seen.add(key)
        result = a
        for b in bs:
            result = (result * pow(b, p - 2, p)) % p
        seq = [a]
        for b in bs:
            seq.extend([op_tok, b])
        seq.append(eq_tok)
        pairs.append(seq)
        labels.append(result)
    return Task("chaindiv", p, *_split(pairs, labels, frac_train, rng))


def make_task(name: str, p: int = 97, frac_train: float = 0.5, seed: int = 42, chain_length: int = 3) -> Task:
    if name == "moddiv":
        return modular_division(p, frac_train, seed)
    if name == "chaindiv":
        return chained_division(p, chain_length, frac_train, seed)
    raise ValueError(f"unknown task {name!r}; choose from {TASKS}")


def carve_meta_split(x: torch.Tensor, y: torch.Tensor, frac: float, seed: int):
    """Split a train set into (inner, meta) for optimizers that learn from held-out data of their own.

    NeuralGrok, GrokAdamW and SuperGrok 1.1 need a loss on examples they do not
    take gradient steps on. That slice comes out of the optimizer's own training
    split (as the original NeuralGrok race did, 90/10), never out of test: they
    train on ``inner`` and use ``meta`` for their held-out signal.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"meta fraction must be in (0, 1), got {frac}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    n_meta = max(1, int(round(len(y) * frac)))
    meta, inner = perm[:n_meta], perm[n_meta:]
    return x[inner], y[inner], x[meta], y[meta]
