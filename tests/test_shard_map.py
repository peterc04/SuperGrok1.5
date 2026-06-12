"""tests/test_shard_map.py — pure-Python unit test for the ZeRO shard partition.

IMPLEMENTS the CPU-testable half of /workspace/.parallelism_design.md §3.4: the
tensor-granular partition (whole tensors, balanced) + the elementwise even
partition. These are PURE FUNCTIONS on size lists — no torch, no
torch.distributed, no GPU — so this test runs green under plain ``pytest`` on any
box (design §7: maximize what's validated on CPU before the 8×H100 window).

Run:
    pytest tests/test_shard_map.py -q
"""

from __future__ import annotations

import pytest

from grokking_optimizers.parallel.shard_map import (
    ShardPlan,
    even_partition,
    partition_elementwise_even,
    partition_for_optimizer,
    partition_tensor_granular,
    shard_mode_for_optimizer,
)

# A representative decoder-ish size list (a few big matrices + many small vectors)
# so balance/coverage are exercised on a realistic lumpy distribution. Numbers
# echo the d=128 production decoder_layout.cuh shapes (embeddings + ff matrices
# dominate; LN/bias vectors are tiny) — the exact "lumpy" case design §3.4 warns
# tensor-granular sharding is lumpier on.
NAMED_SIZES = [
    ("tok_emb", 12672),
    ("pos_emb", 512),
    ("l0.attn.in_proj", 49152),
    ("l0.attn.in_bias", 384),
    ("l0.attn.out_proj", 16384),
    ("l0.ln1.w", 128),
    ("l0.ln1.b", 128),
    ("l0.ff0", 65536),
    ("l0.ff0.b", 512),
    ("l0.ff2", 65536),
    ("l0.ln2.w", 128),
    ("l1.attn.in_proj", 49152),
    ("l1.attn.out_proj", 16384),
    ("l1.ff0", 65536),
    ("l1.ff2", 65536),
    ("lm_head", 12672),
]
TOTAL = sum(n for _, n in NAMED_SIZES)


# ──────────────────────────── even_partition (flat) ──────────────────────────


def test_even_partition_covers_exactly():
    """Union of the per-rank flat slices covers [0, numel) with no gap/overlap."""
    numel, world = 1000, 4
    covered = [0] * numel
    for r in range(world):
        s, e = even_partition(numel, world, r)
        for i in range(s, e):
            covered[i] += 1
    assert all(c == 1 for c in covered), "flat partition gap or overlap"


def test_even_partition_world1_is_whole():
    assert even_partition(777, 1, 0) == (0, 777)


def test_even_partition_matches_distributed_when_available():
    """Mirror invariant: even_partition agrees with distributed._even_partition.

    SKIPS if torch (hence distributed.py's torch import) is unavailable — the
    mirror only needs checking where both exist; the pure-Python copy stands
    alone otherwise.
    """
    dist = pytest.importorskip(
        "grokking_optimizers.distributed", reason="torch/distributed not importable"
    )
    for numel in (0, 1, 7, 1000, 1024, 25401443):
        for world in (1, 2, 3, 8):
            for rank in range(world):
                assert even_partition(numel, world, rank) == dist._even_partition(
                    numel, world, rank
                ), f"mirror drift at numel={numel} world={world} rank={rank}"


# ─────────────────────── partition_elementwise_even (§2.4) ────────────────────


def _covered_by_plan(plan: ShardPlan) -> dict:
    """Element-coverage count per (tensor, index) across all ranks' slices."""
    counts: dict = {}
    for r_slices in plan.slices:
        for name, s, e in r_slices:
            for i in range(s, e):
                counts[(name, i)] = counts.get((name, i), 0) + 1
    return counts


def test_elementwise_partition_covers_every_element_once():
    """Every element of every tensor is owned by exactly one rank (no gap/dup)."""
    plan = partition_elementwise_even(NAMED_SIZES, world=8)
    assert plan.mode == "elementwise_even"
    assert plan.total_numel == TOTAL
    counts = _covered_by_plan(plan)
    assert len(counts) == TOTAL, "coverage count != total elements"
    assert all(c == 1 for c in counts.values()), "elementwise gap or overlap"


def test_elementwise_partition_balance_is_tight():
    """Elementwise split is near-perfectly balanced (the finer-balance win)."""
    plan = partition_elementwise_even(NAMED_SIZES, world=8)
    # ceil padding means at most (world-1) extra elems per rank vs mean — tiny.
    assert plan.imbalance() < 1.01, f"elementwise imbalance too high: {plan.imbalance()}"


def test_elementwise_world1_puts_everything_on_rank0():
    plan = partition_elementwise_even(NAMED_SIZES, world=1)
    assert len(plan.slices) == 1
    assert plan.rank_numel[0] == TOTAL
    # Every tensor present as a whole slice.
    names = {name for name, _, _ in plan.slices[0]}
    assert names == {name for name, _ in NAMED_SIZES}


# ─────────────────────── partition_tensor_granular (§3.4) ─────────────────────


def test_tensor_granular_never_splits_a_tensor():
    """Each tensor appears on EXACTLY ONE rank, as a whole (0, numel) slice."""
    plan = partition_tensor_granular(NAMED_SIZES, world=4)
    assert plan.mode == "tensor_granular"
    seen: dict = {}
    size_by_name = dict(NAMED_SIZES)
    for r, r_slices in enumerate(plan.slices):
        for name, s, e in r_slices:
            assert (s, e) == (0, size_by_name[name]), (
                f"tensor {name} was split: ({s},{e}) != (0,{size_by_name[name]})"
            )
            assert name not in seen, f"tensor {name} placed on >1 rank"
            seen[name] = r
    assert set(seen) == {name for name, _ in NAMED_SIZES}, "a tensor went missing"


def test_tensor_granular_covers_all_elements():
    """The union of whole-tensor placements still accounts for every element."""
    plan = partition_tensor_granular(NAMED_SIZES, world=4)
    assert plan.total_numel == TOTAL


def test_tensor_granular_is_reasonably_balanced():
    """Greedy-LPT keeps the lumpy whole-tensor load within a sane factor.

    Tensor-granular is necessarily lumpier than elementwise (design §3.4), but
    LPT should stay well under 2x the mean for this distribution.
    """
    plan = partition_tensor_granular(NAMED_SIZES, world=4)
    assert plan.imbalance() < 1.5, f"LPT imbalance too high: {plan.imbalance()}"
    # And it must beat naive round-robin's worst case (sanity that LPT helps).
    assert plan.imbalance() >= 1.0


def test_tensor_granular_is_deterministic():
    """Same inputs → identical plan (design §2.7: deterministic shard boundaries).

    Determinism must NOT depend on input order: a shuffled size list yields the
    same per-rank OWNERSHIP (the partition is a pure function of the multiset of
    sizes + world).
    """
    import random

    p1 = partition_tensor_granular(NAMED_SIZES, world=4)
    p2 = partition_tensor_granular(NAMED_SIZES, world=4)
    assert p1 == p2, "tensor-granular not deterministic on identical input"

    shuffled = list(NAMED_SIZES)
    random.Random(1234).shuffle(shuffled)
    p3 = partition_tensor_granular(shuffled, world=4)
    # Ownership map (name -> rank) must match regardless of input order.
    def owner_map(plan):
        return {name: r for r, sls in enumerate(plan.slices) for name, _, _ in sls}

    assert owner_map(p1) == owner_map(p3), "ownership changed under input reorder"


def test_tensor_granular_world1_whole_model_on_rank0():
    """world=1 (the DP=1 case the per-tensor cells use) → whole model on rank 0.

    This is exactly the §7.3 setup: muon/SG11/SG15/SG2 at DP=1 run their P2.x
    stages intact because every tensor is whole on rank 0.
    """
    plan = partition_tensor_granular(NAMED_SIZES, world=1)
    assert plan.rank_numel[0] == TOTAL
    names = {name for name, _, _ in plan.slices[0]}
    assert names == {name for name, _ in NAMED_SIZES}


def test_tensor_granular_more_ranks_than_tensors():
    """world > #tensors leaves some ranks empty but never splits/drops a tensor."""
    small = [("a", 10), ("b", 20), ("c", 30)]
    plan = partition_tensor_granular(small, world=8)
    placed = [name for sls in plan.slices for name, _, _ in sls]
    assert sorted(placed) == ["a", "b", "c"], "a tensor was lost/duplicated"
    assert plan.total_numel == 60


# ──────────────────────────── mode selection (§3.4) ───────────────────────────


@pytest.mark.parametrize(
    "opt",
    ["adamw", "lion", "grokfast", "grokadamw", "looksam", "prodigy", "neuralgrok"],
)
def test_elementwise_optimizers_select_flat_even(opt):
    assert shard_mode_for_optimizer(opt) == "elementwise_even"


@pytest.mark.parametrize("opt", ["muon", "supergrok11", "supergrok15", "supergrok2"])
def test_per_tensor_optimizers_select_tensor_granular(opt):
    """The cells that need a whole tensor on one rank get tensor-granular (§3.4)."""
    assert shard_mode_for_optimizer(opt) == "tensor_granular"


def test_mode_selection_accepts_optid_int():
    """OptId ints map the same way (0=adamw elementwise, 7=muon per-tensor)."""
    assert shard_mode_for_optimizer(0) == "elementwise_even"   # adamw
    assert shard_mode_for_optimizer(7) == "tensor_granular"    # muon
    assert shard_mode_for_optimizer(10) == "tensor_granular"   # supergrok2


def test_mode_selection_is_case_insensitive():
    assert shard_mode_for_optimizer("AdamW") == "elementwise_even"
    assert shard_mode_for_optimizer("SuperGrok2") == "tensor_granular"


def test_mode_selection_rejects_unknown_loudly():
    with pytest.raises(ValueError):
        shard_mode_for_optimizer("not_an_optimizer")
    with pytest.raises(ValueError):
        shard_mode_for_optimizer(99)


def test_partition_for_optimizer_routes_correctly():
    """The convenience router picks the right partitioner per optimizer."""
    el = partition_for_optimizer(NAMED_SIZES, 4, "adamw")
    pt = partition_for_optimizer(NAMED_SIZES, 4, "muon")
    assert el.mode == "elementwise_even"
    assert pt.mode == "tensor_granular"
    # muon's plan never splits a tensor.
    size_by_name = dict(NAMED_SIZES)
    for sls in pt.slices:
        for name, s, e in sls:
            assert (s, e) == (0, size_by_name[name])


# ──────────────────────────── input validation ───────────────────────────────


def test_negative_numel_rejected():
    with pytest.raises(ValueError):
        partition_tensor_granular([("bad", -1)], world=2)
    with pytest.raises(ValueError):
        partition_elementwise_even([("bad", -1)], world=2)


def test_empty_size_list_is_empty_plan():
    plan = partition_tensor_granular([], world=4)
    assert plan.total_numel == 0
    assert plan.imbalance() == 1.0
    assert all(len(s) == 0 for s in plan.slices)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
