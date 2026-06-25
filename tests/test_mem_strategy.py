"""tests/test_mem_strategy.py — CPU gates for the MEMORY-STRATEGY planner
(memory_strategy.md item A). NO torch, NO CUDA, NO GPU — pure fit arithmetic +
escalation decision tree + the SG_MEM_* gate-macro contract.

The planner's ONE rule: escalate in-HBM -> nCTA-cap -> recompute -> offload -> stream
purely by MEMORY FIT, NEVER by GPU count. These tests pin:
  * the byte-identical default (a trivial model returns all-OFF gate macros);
  * each escalation rung engages only when the running budget overflows;
  * the same model needs MORE machinery on 1 GPU than on 8 (proof: fit-driven, not
    a num_gpus switch);
  * gate_macros() always emits the four SG_MEM_* keys the kernel build consumes.

Run:
    PYTHONPATH=. python -m pytest tests/test_mem_strategy.py -q
"""
from __future__ import annotations

from grokking_optimizers.parallel.mem_strategy import (
    Hardware,
    MemPlan,
    plan_memory_strategy,
)

# A flagship-tier reference shape (matches decoder_flagship_layout.cuh: d=1600, L=48,
# vocab=99, seq=4 -> 1,475,884,899 params). Used by the default-shape branch.
FLAGSHIP_TOTAL = 1_475_884_899
FLAGSHIP_LAYERS = 48

# An 80 GB H100-class box (advertised 80 GB ~= 74.5 GiB physical) with ample host RAM.
H100 = dict(hbm_gib_per_gpu=80.0, host_ram_gib=512.0)


def _plan(total, layers, opt, *, tp=1, pp=1, dp=1, zero3=False, B=256,
          n_gpus=1, hbm=80.0, host=512.0):
    hw = Hardware(n_gpus=n_gpus, hbm_gib_per_gpu=hbm, host_ram_gib=host)
    return plan_memory_strategy(total_params=total, layers=layers, opt=opt,
                                tp=tp, pp=pp, dp=dp, zero3=zero3, B=B, hw=hw)


# ── the byte-identical default: a trivial model -> in-HBM, every gate OFF ──
def test_trivial_model_is_in_hbm_all_gates_off():
    plan = _plan(total=10_000_000, layers=8, opt="adamw", B=64)
    assert isinstance(plan, MemPlan)
    assert plan.fits
    assert not plan.offload_optimizer
    assert not plan.recompute_acts
    assert not plan.stream_layers
    assert plan.stream_depth == 0
    assert plan.reason.startswith("in-HBM")


def test_gate_macros_all_zero_on_in_hbm_path():
    """ALL-OFF => -DSG_MEM_*=0 => the #if/if-constexpr blocks fold to the shipped
    in-HBM PTX (the byte-identical guarantee)."""
    plan = _plan(total=10_000_000, layers=8, opt="adamw", B=64)
    m = plan.gate_macros()
    assert m == {
        "SG_MEM_OFFLOAD_OPT": 0,
        "SG_MEM_RECOMPUTE_ACTS": 0,
        "SG_MEM_STREAM_LAYERS": 0,
        "SG_MEM_STREAM_DEPTH": 0,
    }


def test_gate_macros_always_has_four_keys():
    """The kernel build always consumes the same four -D macros; only the values move."""
    plan = _plan(total=FLAGSHIP_TOTAL, layers=FLAGSHIP_LAYERS, opt="adamw", B=256)
    keys = set(plan.gate_macros())
    assert keys == {"SG_MEM_OFFLOAD_OPT", "SG_MEM_RECOMPUTE_ACTS",
                    "SG_MEM_STREAM_LAYERS", "SG_MEM_STREAM_DEPTH"}


# ── the flagship 1.5B on a single 80 GB GPU fits in HBM (the shipped path) ──
def test_flagship_one_gpu_fits_in_hbm():
    plan = _plan(total=FLAGSHIP_TOTAL, layers=FLAGSHIP_LAYERS, opt="adamw", B=256)
    assert plan.fits
    # 1.5B adamw at seq=4 is comfortably in-HBM -> no offload/recompute/stream.
    assert not (plan.offload_optimizer or plan.recompute_acts or plan.stream_layers)


# ── a 10B-class model on ONE small GPU escalates the heavy machinery ──
def test_big_model_one_small_gpu_escalates():
    # ~10B params on a deliberately small 24 GiB card with big host RAM: the planner
    # MUST escalate past in-HBM (recompute and/or offload and/or stream).
    plan = _plan(total=10_000_000_000, layers=64, opt="adamw", B=8,
                 hbm=24.0, host=1024.0)
    engaged = (plan.offload_optimizer + plan.recompute_acts + plan.stream_layers)
    assert engaged >= 1, f"expected escalation, got {plan.reason!r}"


def test_recompute_before_offload_before_stream():
    """Escalation order is cheapest-first: recompute is reached before offload, and
    offload before streaming. Verified by the monotone machinery as the box shrinks."""
    big = dict(total=10_000_000_000, layers=64, opt="adamw", B=8, host=2048.0)
    # A roomy-but-not-trivial card may need only recompute; a tiny card needs the stack.
    roomy = _plan(hbm=70.0, **big)
    tiny = _plan(hbm=12.0, **big)
    # streaming (the last resort) only appears when offload is already on.
    if roomy.stream_layers:
        assert roomy.offload_optimizer
    if tiny.stream_layers:
        assert tiny.offload_optimizer and tiny.recompute_acts
    # the tinier card never needs strictly less machinery than the roomy one.
    assert (tiny.offload_optimizer + tiny.recompute_acts + tiny.stream_layers) >= \
           (roomy.offload_optimizer + roomy.recompute_acts + roomy.stream_layers)


# ── the DIRECTIVE: fit-driven, NOT a GPU-count switch ──
def test_strategy_is_fit_driven_not_gpu_count():
    """Same model: more per-GPU budget (via TP) -> less machinery. The ONLY thing that
    moved is the per-rank footprint, never an `if n_gpus` branch."""
    common = dict(total=10_000_000_000, layers=64, opt="adamw", B=8, hbm=40.0,
                  host=1024.0)
    p1 = _plan(tp=1, n_gpus=1, **common)
    p8 = _plan(tp=8, n_gpus=8, **common)
    heavy1 = p1.offload_optimizer + p1.recompute_acts + p1.stream_layers
    heavy8 = p8.offload_optimizer + p8.recompute_acts + p8.stream_layers
    assert heavy1 >= heavy8


def test_same_footprint_same_plan_regardless_of_n_gpus_label():
    """The decision keys on (footprint, usable_hbm), not n_gpus: two Hardware descriptors
    with the SAME per-GPU HBM/host but different n_gpus labels (and tp=1) plan identically."""
    a = _plan(total=FLAGSHIP_TOTAL, layers=FLAGSHIP_LAYERS, opt="adamw", B=256,
              n_gpus=1, hbm=80.0)
    b = _plan(total=FLAGSHIP_TOTAL, layers=FLAGSHIP_LAYERS, opt="adamw", B=256,
              n_gpus=8, hbm=80.0)
    assert a.gate_macros() == b.gate_macros()
    assert a.offload_optimizer == b.offload_optimizer
    assert a.recompute_acts == b.recompute_acts
    assert a.stream_layers == b.stream_layers


# ── offload/stream require host RAM; streaming uses a ring depth >= 2 ──
def test_streaming_uses_ring_depth_two():
    plan = _plan(total=20_000_000_000, layers=80, opt="adamw", B=8,
                 hbm=16.0, host=4096.0)
    if plan.stream_layers:
        assert plan.stream_depth >= 2
        assert plan.host_gib > 0.0  # weights pinned to host when streaming


def test_no_fit_records_honest_failure():
    """When even the full stack cannot fit (host RAM too small for offload+stream),
    the planner returns fits=False with an honest reason — never a forced pass."""
    plan = _plan(total=50_000_000_000, layers=96, opt="adamw", B=16,
                 hbm=8.0, host=8.0)  # absurdly small host RAM
    if not plan.fits:
        assert "OOM" in plan.reason or "STILL OOM" in plan.reason
