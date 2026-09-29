"""Grokking tasks: token sequences for a causal LM, answer read at the last position.

Both builders are ports of the previous race driver (``grokking_race_v2.py`` at
commit 19c9d39, ``make_data`` and ``make_sequential_division_data``). For the
same ``(p, frac_train, val_ratio, seed)`` they produce the same examples, in the
same order and with the same splits, so results stay comparable with earlier
races.

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
    x_val: torch.Tensor
    y_val: torch.Tensor
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

    def sizes(self) -> tuple[int, int, int]:
        return len(self.y_train), len(self.y_val), len(self.y_test)

    def to(self, device) -> "Task":
        t = {k: getattr(self, k).to(device) for k in ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")}
        return Task(self.name, self.p, **t)


def _split(pairs, labels, frac_train, val_ratio, rng):
    combined = list(zip(pairs, labels))
    rng.shuffle(combined)
    pairs, labels = zip(*combined)
    n_train_total = int(len(pairs) * frac_train)
    n_val = int(n_train_total * val_ratio)
    n_train = n_train_total - n_val
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return (
        x[:n_train],
        y[:n_train],
        x[n_train:n_train_total],
        y[n_train:n_train_total],
        x[n_train_total:],
        y[n_train_total:],
    )


def modular_division(p: int = 97, frac_train: float = 0.5, val_ratio: float = 0.10, seed: int = 42) -> Task:
    """All ``(a, b)`` with ``b != 0``: input ``[a, op, b, eq]``, label ``a * b^-1 mod p``."""
    rng = random.Random(seed)
    op_tok, eq_tok = p, p + 1
    pairs, labels = [], []
    for a in range(p):
        for b in range(1, p):
            pairs.append([a, op_tok, b, eq_tok])
            labels.append((a * pow(b, p - 2, p)) % p)
    return Task("moddiv", p, *_split(pairs, labels, frac_train, val_ratio, rng))


def chained_division(
    p: int = 97, chain_length: int = 3, frac_train: float = 0.5, val_ratio: float = 0.10, seed: int = 42
) -> Task:
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
    return Task("chaindiv", p, *_split(pairs, labels, frac_train, val_ratio, rng))


def make_task(
    name: str, p: int = 97, frac_train: float = 0.5, val_ratio: float = 0.10, seed: int = 42, chain_length: int = 3
) -> Task:
    if name == "moddiv":
        return modular_division(p, frac_train, val_ratio, seed)
    if name == "chaindiv":
        return chained_division(p, chain_length, frac_train, val_ratio, seed)
    raise ValueError(f"unknown task {name!r}; choose from {TASKS}")


def default_val_ratio(frac_train: float) -> float:
    """The previous driver used 0.05 at a 10% train split (too few examples otherwise), else 0.10."""
    return 0.05 if frac_train <= 0.10 else 0.10


def carve_meta_split(x: torch.Tensor, y: torch.Tensor, frac: float, seed: int):
    """Split a train set into (inner, meta) for optimizers that meta-learn on held-out data.

    The meta slice comes out of the optimizer's own training data, so every
    optimizer sees the same total number of labelled examples, and the val split
    stays reserved for early stopping.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    n_meta = max(1, int(round(len(y) * frac)))
    meta, inner = perm[:n_meta], perm[n_meta:]
    return x[inner], y[inner], x[meta], y[meta]
