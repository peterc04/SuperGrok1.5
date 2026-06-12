"""tests/test_zero3_plan.py — CPU gates for the ZeRO-3 flat-blob plan,
gather/release, and the sharded checkpoint round-trip (design §3 via §7;
task #25 phase-2, Lane C).

CPU-ONLY (torch CPU tensors, no CUDA init, no GPU) — the device round-trip
around the REAL fused step is tests/hw/test_zero3_roundtrip.py (GPU box).

Run:
    PYTHONPATH=. python -m pytest tests/test_zero3_plan.py -q
"""
from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from grokking_optimizers.parallel.zero3 import (  # noqa: E402
    FlatShardPlan,
    Zero3FlatParamStore,
    flat_plan_for_optimizer,
    load_sharded_checkpoint,
    save_sharded_checkpoint,
)

# A small named-size list shaped like a real layout (mixed magnitudes).
_SIZES = [("tok", 1000), ("pos", 16), ("w0", 4096), ("b0", 64),
          ("w1", 4096), ("b1", 64), ("head", 990)]
_TOTAL = sum(n for _, n in _SIZES)


# ─────────────────────────────── plan geometry ───────────────────────────────


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_elementwise_plan_partitions_flat_blob(world):
    plan = flat_plan_for_optimizer(_SIZES, world, "adamw")
    assert plan.mode == "elementwise_even"
    assert plan.total == _TOTAL
    covered = []
    for r in range(world):
        for s, e in plan.slices[r]:
            covered.append((s, e))
    covered.sort()
    # union covers [0,total) exactly, no overlap.
    pos = 0
    for s, e in covered:
        assert s == pos and e > s
        pos = e
    assert pos == _TOTAL
    # elementwise = ONE contiguous slice per rank (the kernel-binding convention).
    assert all(len(plan.slices[r]) <= 1 for r in range(world))


@pytest.mark.parametrize("world", [2, 4])
def test_tensor_granular_plan_keeps_tensors_whole(world):
    plan = flat_plan_for_optimizer(_SIZES, world, "muon")
    assert plan.mode == "tensor_granular"
    # every slice must be a whole tensor's flat range.
    offsets = {}
    off = 0
    for name, n in _SIZES:
        offsets[(off, off + n)] = name
        off += n
    seen = []
    for r in range(world):
        for s, e in plan.slices[r]:
            assert (s, e) in offsets, f"slice ({s},{e}) splits a tensor"
            seen.append((s, e))
    assert sorted(seen) == sorted(offsets)   # complete + disjoint


def test_fingerprint_pins_every_boundary():
    p2 = flat_plan_for_optimizer(_SIZES, 2, "adamw")
    p4 = flat_plan_for_optimizer(_SIZES, 4, "adamw")
    pt = flat_plan_for_optimizer(_SIZES, 2, "muon")
    assert p2.fingerprint() != p4.fingerprint()
    assert p2.fingerprint() != pt.fingerprint()
    assert p2.fingerprint() == flat_plan_for_optimizer(_SIZES, 2, "adamw").fingerprint()


# ─────────────────────────── gather / release dance ─────────────────────────


@pytest.mark.parametrize("opt,world", [("adamw", 2), ("adamw", 4), ("muon", 2)])
def test_gather_release_roundtrip_bit_exact(opt, world):
    """Virtual-rank (in-process) gather_full reconstitutes the original flat
    blob BIT-EXACTLY; release after an owned-slice update keeps shards
    consistent (the §3.2(a) full-pre-gather dance, CPU edition)."""
    g = torch.Generator().manual_seed(7)
    full0 = torch.randn(_TOTAL, generator=g)
    plan = flat_plan_for_optimizer(_SIZES, world, opt)
    stores = [Zero3FlatParamStore(plan, r, full_flat=full0) for r in range(world)]

    for r in range(world):
        peers = [stores[k] for k in range(world) if k != r]
        full = stores[r].gather_full(peers=peers)
        assert torch.equal(full, full0), f"rank {r}: gather_full != original"

    # simulate a step: each rank updates ONLY its owned slices of the full
    # buffer (what the sharded-opt does on its shard), then releases.
    full = stores[0].gather_full(peers=stores[1:])
    updated = full0.clone()
    for r in range(world):
        for fs, fe, ss, se in stores[r].owned():
            updated[fs:fe] += (r + 1) * 0.5
    for r in range(world):
        f = updated.clone()
        stores[r].release(f)
    full2 = stores[0].gather_full(peers=stores[1:])
    assert torch.equal(full2, updated)


def test_gather_peers_must_cover_world():
    plan = flat_plan_for_optimizer(_SIZES, 4, "adamw")
    full0 = torch.randn(_TOTAL)
    stores = [Zero3FlatParamStore(plan, r, full_flat=full0) for r in range(4)]
    with pytest.raises(ValueError):
        stores[0].gather_full(peers=[stores[1]])       # ranks 2,3 missing
    with pytest.raises(RuntimeError):
        stores[0].gather_full()                        # world>1, no peers, no dist


# ─────────────────────────── checkpoint round-trip ──────────────────────────


def test_sharded_checkpoint_roundtrip_bit_exact():
    plan = flat_plan_for_optimizer(_SIZES, 2, "adamw")
    g = torch.Generator().manual_seed(11)
    full0 = torch.randn(_TOTAL, generator=g)
    with tempfile.TemporaryDirectory() as td:
        shards = []
        for r in range(2):
            store = Zero3FlatParamStore(plan, r, full_flat=full0)
            n = store.shard.numel()
            state = torch.randn(3 * n, generator=g)    # [m|v|extra] shard-local
            save_sharded_checkpoint(
                os.path.join(td, f"rank{r}.pt"), plan=plan, rank=r, step=17,
                param_shard=store.shard, opt_state_shard=state,
                extra_meta={"opt": "adamw"})
            shards.append((store.shard.clone(), state.clone()))
        for r in range(2):
            obj = load_sharded_checkpoint(os.path.join(td, f"rank{r}.pt"),
                                          plan=plan, rank=r)
            assert obj["step"] == 17
            assert torch.equal(obj["param_shard"], shards[r][0])
            assert torch.equal(obj["opt_state_shard"], shards[r][1])
            assert obj["extra_meta"]["opt"] == "adamw"


def test_checkpoint_refuses_mismatched_plan_or_rank():
    plan2 = flat_plan_for_optimizer(_SIZES, 2, "adamw")
    plan4 = flat_plan_for_optimizer(_SIZES, 4, "adamw")
    full0 = torch.randn(_TOTAL)
    store = Zero3FlatParamStore(plan2, 0, full_flat=full0)
    n = store.shard.numel()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "r0.pt")
        save_sharded_checkpoint(path, plan=plan2, rank=0, step=1,
                                param_shard=store.shard,
                                opt_state_shard=torch.zeros(3 * n))
        with pytest.raises(RuntimeError):
            load_sharded_checkpoint(path, plan=plan4, rank=0)   # world drifted
        with pytest.raises(RuntimeError):
            load_sharded_checkpoint(path, plan=plan2, rank=1)   # wrong rank
        # the happy path still loads.
        assert load_sharded_checkpoint(path, plan=plan2, rank=0)["step"] == 1


def test_checkpoint_rejects_misaligned_state_planes():
    plan = flat_plan_for_optimizer(_SIZES, 2, "adamw")
    full0 = torch.randn(_TOTAL)
    store = Zero3FlatParamStore(plan, 0, full_flat=full0)
    n = store.shard.numel()
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError):
            save_sharded_checkpoint(os.path.join(td, "bad.pt"), plan=plan,
                                    rank=0, step=1, param_shard=store.shard,
                                    opt_state_shard=torch.zeros(3 * n + 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
