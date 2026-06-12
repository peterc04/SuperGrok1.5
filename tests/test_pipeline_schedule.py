"""tests/test_pipeline_schedule.py — CPU gates for the 1F1B pipeline schedule,
stage partition, and hand-off plan (design §4 via §7; task #25 phase-2, Lane C).

PURE CPU (no torch, no GPU) — runs on any box, mirroring tests/test_shard_map.py's
discipline (design §7: maximize what's validated before the 8×H100 window).
The GPU-side stage kernels are gated separately by
tests/hw/test_pp2_loopback_determinism.py (which also cross-checks the tensor
ownership here against the kernel-side PPStageSpec single source).

Run:
    PYTHONPATH=. python -m pytest tests/test_pipeline_schedule.py -q
"""
from __future__ import annotations

import pytest

from grokking_optimizers.parallel.pipeline import (
    LoopbackP2P,
    bubble_fraction,
    build_1f1b_schedule,
    handoff_plan,
    max_inflight_microbatches,
    run_1f1b,
    stage_layer_ranges,
    stage_tensor_ownership,
    validate_1f1b_schedule,
)

_GRID = [(p, m) for p in (1, 2, 3, 4, 8) for m in (1, 2, 4, 8, 16) if m >= 1]


# ─────────────────────────────── stage partition ─────────────────────────────


def test_stage_layer_ranges_even_split():
    assert stage_layer_ranges(2, 2) == [(0, 1), (1, 2)]
    assert stage_layer_ranges(8, 4) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert stage_layer_ranges(48, 8)[0] == (0, 6)
    assert stage_layer_ranges(48, 8)[-1] == (42, 48)


def test_stage_layer_ranges_rejects_uneven_and_oversplit():
    with pytest.raises(ValueError):
        stage_layer_ranges(3, 2)        # 2 ∤ 3 — must be loud, never silent
    with pytest.raises(ValueError):
        stage_layer_ranges(2, 3)        # more stages than layers
    with pytest.raises(ValueError):
        stage_layer_ranges(0, 1)


def test_decoder_tensor_ownership_pp2():
    """The decoder 30-tensor map at PP=2: stage0 = embeddings + L0 block
    {0..13}; stage1 = L1 block + final norm + head {14..29}. This is the host
    mirror of PPStageSpec::owns_tensor — the kernel-side cross-check lives in
    tests/hw/test_pp2_loopback_determinism.py."""
    own = stage_tensor_ownership("decoder", 2)
    assert own[0] == list(range(0, 14))
    assert own[1] == list(range(14, 30))
    # partition: disjoint + complete (also asserted inside, belt-and-braces).
    assert sorted(own[0] + own[1]) == list(range(30))


def test_decoder_tensor_ownership_pp1_degenerate():
    own = stage_tensor_ownership("decoder", 1)
    assert own == [list(range(30))]


def test_unknown_model_is_loud():
    with pytest.raises(ValueError):
        stage_tensor_ownership("vit", 2)   # not yet declared — must not default


# ─────────────────────────────── 1F1B schedule ───────────────────────────────


@pytest.mark.parametrize("P,M", _GRID)
def test_1f1b_schedule_valid(P, M):
    """Every (P,M) point: dependency-feasible (no deadlock), every microbatch
    fwd+bwd exactly once per stage, in-flight ≤ the 1F1B bound."""
    sched = build_1f1b_schedule(P, M)
    validate_1f1b_schedule(sched, M)   # raises on any violation
    assert len(sched) == P
    for ops in sched:
        assert len(ops) == 2 * M


def test_1f1b_last_stage_strictly_alternates():
    """The LAST stage has zero warmup: fwd(mb) immediately followed by bwd(mb)
    — the property that lets the PP stage kernel keep the last stage's fwd+bwd
    FUSED in one launch (pp_decoder_stage_bwd_tc)."""
    for M in (1, 2, 4, 8):
        sched = build_1f1b_schedule(4, M)
        last = sched[-1]
        expect = []
        for mb in range(M):
            expect += [("fwd", mb), ("bwd", mb)]
        assert last == expect


def test_1f1b_warmup_counts():
    """Leading fwds before the FIRST bwd = warmup + 1 steady fwd (when any
    steady pair exists): warmup = min(P-1-s, M), then the steady loop issues
    one more fwd before its first bwd (the 1F1B pairing)."""
    P, M = 4, 8
    sched = build_1f1b_schedule(P, M)
    for s in range(P):
        leading = 0
        for kind, _ in sched[s]:
            if kind == "fwd":
                leading += 1
            else:
                break
        warmup = min(P - 1 - s, M)
        steady = M - warmup
        assert leading == warmup + (1 if steady > 0 else 0)


def test_bubble_fraction_formula():
    assert bubble_fraction(1, 8) == 0.0
    assert abs(bubble_fraction(4, 8) - 3 / 11) < 1e-12
    assert abs(bubble_fraction(2, 1) - 1 / 2) < 1e-12


def test_inflight_bound():
    assert max_inflight_microbatches(4, 8, 0) == 4
    assert max_inflight_microbatches(4, 8, 3) == 1
    assert max_inflight_microbatches(4, 2, 0) == 2   # clamped by M


# ─────────────────────────────── the driver ─────────────────────────────────


@pytest.mark.parametrize("P,M", [(2, 1), (2, 4), (4, 8), (3, 2), (8, 16)])
def test_run_1f1b_executes_with_loopback_transport(P, M):
    """run_1f1b against recording stubs + LoopbackP2P: completes, every handoff
    produced/consumed exactly once, payloads flow stage→stage correctly."""
    log = []

    def fwd(s, mb, fwd_in):
        if s == 0:
            assert fwd_in is None
        else:
            assert fwd_in == ("act", s - 1, mb)   # upstream's payload arrived
        log.append(("fwd", s, mb))
        return ("act", s, mb)

    def bwd(s, mb, dh_in):
        if s == P - 1:
            assert dh_in is None                   # last stage starts from CE
        else:
            assert dh_in == ("dh", s + 1, mb)
        log.append(("bwd", s, mb))
        return ("dh", s, mb)

    tr = run_1f1b(P, M, fwd, bwd)
    assert tr.drained()
    assert tr.sends == tr.recvs == 2 * (P - 1) * M   # act + dh per boundary per mb
    # every (stage, mb) ran exactly once per direction.
    assert sorted(e for e in log if e[0] == "fwd") == \
        sorted(("fwd", s, mb) for s in range(P) for mb in range(M))
    assert sorted(e for e in log if e[0] == "bwd") == \
        sorted(("bwd", s, mb) for s in range(P) for mb in range(M))
    # dependency order in the executed log.
    pos = {e: i for i, e in enumerate(log)}
    for s in range(P):
        for mb in range(M):
            if s > 0:
                assert pos[("fwd", s, mb)] > pos[("fwd", s - 1, mb)]
            if s < P - 1:
                assert pos[("bwd", s, mb)] > pos[("bwd", s + 1, mb)]
            assert pos[("bwd", s, mb)] > pos[("fwd", s, mb)]


def test_loopback_p2p_is_honest():
    """recv-before-send and double-send are LOUD failures (a real P2P would
    deadlock / corrupt — the loopback must surface, not mask)."""
    tr = LoopbackP2P()
    with pytest.raises(AssertionError):
        tr.recv(0, 1, 0, "act")
    tr.send(0, 1, 0, "act", "x")
    with pytest.raises(AssertionError):
        tr.send(0, 1, 0, "act", "y")
    assert tr.recv(0, 1, 0, "act") == "x"
    assert tr.drained()


# ─────────────────────────────── hand-off plan ──────────────────────────────


def test_handoff_plan_decoder_pp2():
    """One boundary at layer 1; fwd = bf16 [T,d] (2 B/elem), bwd = fp32 (4 B)."""
    specs = handoff_plan(d_model=128, seq=4, microbatch_rows=4176,
                         n_layers=2, num_stages=2)
    assert len(specs) == 1
    sp = specs[0]
    t = 4176 * 4
    assert sp.boundary_layer == 1
    assert sp.fwd_numel == sp.bwd_numel == t * 128
    assert sp.fwd_bytes == 2 * t * 128       # bf16 X_in[1]
    assert sp.bwd_bytes == 4 * t * 128       # fp32 dh (bit-preserving carrier)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
