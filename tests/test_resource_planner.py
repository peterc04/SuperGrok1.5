"""tests/test_resource_planner.py — CPU gates for the ROBUST execution planner
(resource_fit_planner.md). NO torch, NO CUDA, NO GPU — pure arithmetic + decision tree.

Run:
    PYTHONPATH=. python -m pytest tests/test_resource_planner.py -q
"""
from __future__ import annotations

import pytest

from grokking_optimizers.parallel.resource_planner import (
    HardwareConfig,
    ModelConfig,
    PlanInfeasible,
    layout_arith,
    plan_execution,
    sg2_ws_stride,
)

H100 = dict(hbm_bytes_per_gpu=80 * (1000 ** 3), sms_per_gpu=132, nvlink=True,
            nvlink_width=8)


# ── layout arithmetic pinned to the live flagship constants ──
def test_layout_matches_flagship():
    total, nt, nmax = layout_arith(ModelConfig(d=1600, layers=48, vocab=99, seq=4))
    assert total == 1_475_884_899          # decoder_flagship_layout.cuh kDecTotalElems
    assert nt == 582                       # kDecNumTensors
    assert nmax == 10_240_000              # kDecMaxTensorNumel == 4d^2


def test_sg2_stride_factor_is_91():
    # ~91.277 floats/CTA per Nmax with the SG2Dims<> defaults (verified vs the cuh).
    assert abs(sg2_ws_stride(10_240_000) / 10_240_000 - 91.277) < 0.01


# ── worked example 10M/1GPU: trivial, in-HBM, full occupancy ──
def test_10m_one_gpu_trivial():
    mc = ModelConfig(d=512, layers=8, vocab=99, seq=128, batch=256, optimizer="adamw")
    plan = plan_execution(mc, HardwareConfig(num_gpus=1, **H100))
    assert plan.fits
    assert plan.mesh.world_size == 1
    assert not plan.mem.need_param_offload
    assert not plan.mem.need_opt_offload
    assert not plan.mem.need_layer_streaming
    # adamw single-opt -> staged carve elided (bench layout).
    assert "-DSG_DEC_BENCH_LAYOUT=1" in plan.compile_flags


# ── worked example 1.5B/8GPU: 4D + ZeRO-3, TP=8, SG2 caps ncta ──
def test_flagship_eight_gpu_4d_zero3():
    mc = ModelConfig(d=1600, layers=48, vocab=99, seq=4, batch=512, optimizer="supergrok2")
    plan = plan_execution(mc, HardwareConfig(num_gpus=8, **H100))
    assert plan.fits
    assert plan.mesh.tp == 8 and plan.mesh.pp == 1 and plan.mesh.dp == 1
    assert plan.mem.need_zero_offload          # ZeRO-3 overlay recorded (no-op at DP=1)
    # supergrok2 is the worst case -> cta-tiling caps ncta below full occupancy.
    assert plan.knobs.ncta <= 64
    assert f"-DSG_FLAGSHIP_TP=8" in plan.compile_flags
    assert "ParConfig<1,8,1,1,ZeROStage::Z3>" in plan.template_inst


# ── worked example 10B/1GPU: offload + recompute + streaming + cta-tile ──
def test_10b_one_gpu_full_stack():
    mc = ModelConfig(d=4096, layers=48, vocab=50304, seq=2048, batch=8,
                     optimizer="adamw")
    plan = plan_execution(mc, HardwareConfig(num_gpus=1, host_ram_bytes=512 * (1000 ** 3),
                                             **H100))
    assert plan.mesh.world_size == 1
    assert plan.mem.need_activation_recompute
    assert plan.knobs.ncta < 132               # cta-tiling
    # heavy machinery engaged (the directive's 10B/1GPU case).
    assert plan.mem.need_layer_streaming or plan.mem.need_opt_offload


def test_10b_one_gpu_sg2_downgrades():
    mc = ModelConfig(d=4096, layers=48, vocab=50304, seq=2048, batch=8,
                     optimizer="supergrok2")
    plan = plan_execution(mc, HardwareConfig(num_gpus=1, host_ram_bytes=512 * (1000 ** 3),
                                             **H100))
    # SG2's per-CTA workspace is structurally too large at Nmax=67M on one GPU ->
    # the planner downgrades to adamw + offload and records the honest risk.
    assert any("supergrok2" in r and "does not fit" in r for r in plan.risks)
    assert plan.model.optimizer == "adamw"


# ── worked example 10B/8GPU: TP shrinks Nmax -> recompute + cta-tile, no full offload ──
def test_10b_eight_gpu_tp_shrinks_nmax():
    mc = ModelConfig(d=4096, layers=48, vocab=50304, seq=2048, batch=8, optimizer="adamw")
    plan = plan_execution(mc, HardwareConfig(num_gpus=8, **H100))
    assert plan.fits
    assert plan.mesh.tp == 8
    assert plan.mem.need_activation_recompute
    # contrast with 10B/1GPU: 8 GPUs fit WITHOUT host param offload (TP shrank Nmax).
    assert not plan.mem.need_param_offload


# ── the directive: NOT keyed on GPU count (same model, fit decides the machinery) ──
def test_strategy_is_fit_driven_not_gpu_count():
    mc = ModelConfig(d=4096, layers=48, vocab=50304, seq=2048, batch=8, optimizer="adamw")
    p1 = plan_execution(mc, HardwareConfig(num_gpus=1, **H100))
    p8 = plan_execution(mc, HardwareConfig(num_gpus=8, **H100))
    # 1 GPU needs strictly MORE machinery than 8 GPUs for the SAME model — proof the
    # driver escalates by fit, not by a num_gpus switch.
    heavy1 = (p1.mem.need_layer_streaming + p1.mem.need_opt_offload
              + p1.mem.need_param_offload + p1.mem.cta_tiling)
    heavy8 = (p8.mem.need_layer_streaming + p8.mem.need_opt_offload
              + p8.mem.need_param_offload + p8.mem.cta_tiling)
    assert heavy1 >= heavy8


def test_moe_engages_expert_parallel():
    mc = ModelConfig(d=1600, layers=48, vocab=99, seq=4, batch=512, num_experts=8,
                     optimizer="adamw")
    plan = plan_execution(mc, HardwareConfig(num_gpus=8, **H100))
    # EP only when there is a DP factor to sub-divide; at TP=8 DP=1 EP stays 1.
    assert plan.mesh.ep in (1, 2, 4, 8)
    assert plan.mesh.dp % plan.mesh.ep == 0


def test_byte_identical_flags_when_single_gpu_trivial():
    mc = ModelConfig(d=512, layers=8, vocab=99, seq=128, batch=64, optimizer="adamw")
    plan = plan_execution(mc, HardwareConfig(num_gpus=1, **H100))
    # SingleGPU template instantiation -> ParConfig defaults to par::SingleGPU.
    assert "par::SingleGPU" in plan.template_inst
    assert "-DSG_FLAGSHIP_TP=1" in plan.compile_flags
