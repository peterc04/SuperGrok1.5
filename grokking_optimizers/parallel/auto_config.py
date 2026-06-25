"""grokking_optimizers/parallel/auto_config.py — the FRONT-END → ParConfig
ADAPTIVE 3D–5D inference.

Reads a model config (the grokking_race_v2 config dict the race builds models
from) and PICKS the parallelism degree, then maps it to the compile-time
ParConfig<DP,TP,PP,SP,Z,EP> template instantiation the launcher dispatches.

THE ADAPTIVE CONTRACT (the auto-3D–5D rule, /workspace/impl_diffs/
adaptive_parallelism.md §3):

  * base 3D  = DP × TP × PP                         (EVERY model);
  * +SP (4th) iff the model is a SEQUENCE model     (decoder / ViT-patches /
               Mamba are sequence-shaped ⇒ SP-eligible). SP is EXPRESSIBLE but
               PINNED to 1 this campaign (the kernel static_assert in
               parallel_config.cuh; seq 4-17 makes a seq split moot), so the
               returned degree carries sp_eligible=True but sp=1 — the 4th axis
               is *unlocked* for a future long-seq model, not silently broken;
  * +EP (5th) iff the model declares model-level EXPERTS (num_experts>1 or an
               moe/is_moe flag). EP sub-divides the DP group (it does NOT enlarge
               world_size — distributed.py: data_parallel % expert_parallel == 0),
               so it is the 5th axis a MoE model engages. The current race roster
               (decoder/vit/mamba) has NO model experts (num_experts there is the
               SuperGrok2 OPTIMIZER's PEER meta-net, never a model layer), so EP
               stays 1 for them and the kernel's kEPComm folds away — honest
               inertness, the byte-identical-when-OFF guarantee.

PURE PYTHON: no torch, no torch.distributed, no GPU — so it imports and unit-tests
on any box (design §7: maximize what's validated on CPU/1-GPU). It returns a
grokking_optimizers.distributed.ParallelConfig (the SAME carrier
DistributedContext.from_config consumes) plus a small AdaptivePlan describing the
chosen degree + the ParConfig<...> template string for the launcher dispatch.

SOURCES (read in full this session):
  * model defs + config keys : grokking_race_v2.py
        (_raw_model: model_type decoder/vit/mamba; DEFAULT_CONFIG; MODEL_SCALES /
         MODEL_SCALES_BY_MODEL flagship; no model-level num_experts on the roster)
  * mesh + EP rules          : grokking_optimizers/distributed.py
        (ParallelConfig.world_size = DP*TP*PP; expert_parallel sub-divides DP)
  * 8-H100 saturation mesh   : /workspace/impl_diffs/run_harness.md §0
        (TP8 × DP1 × PP1 + ZeRO-3 is the recommended flagship config)
  * kernel ParConfig axis    : csrc/fused/sm_90/parallel_config.cuh
        (template <int DP,int TP,int PP,int SP,ZeROStage Z,int EP=1>; SP pinned 1)
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional, Tuple

from grokking_optimizers.distributed import ParallelConfig

# ── The set of model_types that are SEQUENCE models (SP-eligible). Decoder is
#    token-sequence; ViT is a patch-sequence (+ cls token); Mamba is a state-space
#    sequence. All three are sequence-shaped, so all three are SP-eligible (the 4th
#    axis is unlocked for them) — even though SP is pinned 1 this campaign. A
#    non-sequence model (e.g. a pure MLP probe) would be SP-INeligible. ──
_SEQUENCE_MODEL_TYPES = frozenset({"decoder", "vit", "mamba"})

# ── ZeRO stage spelling → the kernel ZeROStage enumerator (parallel_config.cuh). ──
_ZERO_ENUM = {0: "Z0", 1: "Z1", 2: "Z2", 3: "Z3"}

DEFAULT_WORLD = 8  # 8× H100 (the saturation target, run_harness.md §0)


@dataclasses.dataclass(frozen=True)
class AdaptivePlan:
    """The chosen adaptive mesh + the kernel template instantiation it maps to.

    `degree` is 3/4/5 (3D base, +1 if SP engaged, +1 if EP engaged). Note SP is
    pinned 1 this campaign, so `degree` counts SP only as an *eligible* unlock:
    `degree_eligible` is the would-be degree if SP were active; `degree` is the
    EFFECTIVE degree actually instantiated (SP collapses out when sp==1). EP
    counts toward both only when ep>1.
    """

    dp: int
    tp: int
    pp: int
    sp: int                  # EFFECTIVE SP degree instantiated (pinned 1 this campaign)
    ep: int                  # EFFECTIVE EP degree instantiated (1 = dense)
    zero_stage: int
    sp_eligible: bool        # is the model a sequence model (4th axis unlockable)?
    has_experts: bool        # does the model declare a model-level MoE (5th axis)?
    world: int

    @property
    def degree(self) -> int:
        """EFFECTIVE adaptive degree actually instantiated (3, 4, or 5)."""
        d = 3
        if self.sp > 1:
            d += 1
        if self.ep > 1:
            d += 1
        return d

    @property
    def degree_eligible(self) -> int:
        """The degree the model is ELIGIBLE for (counts SP-eligibility even when
        SP is pinned 1) — the 'auto 3D–5D' label the front-end reports."""
        d = 3
        if self.sp_eligible:
            d += 1
        if self.has_experts:
            d += 1
        return d

    def parconfig_template(self) -> str:
        """The C++ ParConfig<...> instantiation the launcher dispatches.

        Always emits the EP arg (even when 1) for an unambiguous 6-arg point; the
        EP=1 form is byte-identical to the legacy 5-arg point (the trailing
        default), so this is safe to hand to the §7.2 allow-list / dispatch.
        """
        z = _ZERO_ENUM[self.zero_stage]
        return (f"::sg::fused::par::ParConfig<{self.dp}, {self.tp}, {self.pp}, "
                f"{self.sp}, ::sg::fused::par::ZeROStage::{z}, {self.ep}>")

    def to_parallel_config(self, **overrides: Any) -> ParallelConfig:
        """Build the distributed.ParallelConfig this plan describes (the carrier
        DistributedContext.from_config consumes). EP rides as expert_parallel
        (it sub-divides DP, never enlarges world)."""
        return ParallelConfig(
            data_parallel=self.dp, tensor_parallel=self.tp,
            pipeline_parallel=self.pp, expert_parallel=self.ep,
            zero_stage=self.zero_stage, **overrides)


def _model_type(cfg: Mapping[str, Any]) -> str:
    return str(cfg.get("model_type", "decoder")).strip().lower()


def is_sequence_model(cfg: Mapping[str, Any]) -> bool:
    """True iff the model is a SEQUENCE model (SP-eligible). Decoder / ViT-patches
    / Mamba are all sequence-shaped (grokking_race_v2._raw_model)."""
    return _model_type(cfg) in _SEQUENCE_MODEL_TYPES


def model_num_experts(cfg: Mapping[str, Any]) -> int:
    """The MODEL-level expert count (the MoE width), or 1 if dense.

    HONEST DISAMBIGUATION (the trap): grokking_race_v2.py's `num_experts` keys are
    ALL `sg2_num_experts` — the SuperGrok2 OPTIMIZER's PEER meta-net experts, NOT a
    model layer. So we read ONLY the MODEL's own MoE keys (`num_experts` /
    `model_num_experts` / a `moe`/`is_moe` flag), and DELIBERATELY ignore any
    `sg2_*` / optimizer key. The current roster has none of the model keys, so this
    returns 1 (dense) for decoder/vit/mamba — EP folds away (kEPComm==false)."""
    # An explicit MoE flag forces experts on (a future model may set is_moe=True
    # and carry its width under model_num_experts).
    flag = cfg.get("is_moe", cfg.get("moe", False))
    for key in ("model_num_experts", "num_experts"):
        # Guard against the optimizer key bleeding in: only honor a top-level
        # MODEL key, never an sg2_-prefixed one (those are never named plainly
        # `num_experts` in the race config, but be explicit).
        if key.startswith("sg2_"):
            continue
        v = cfg.get(key)
        if isinstance(v, int) and v > 1:
            return v
    if flag:
        # MoE declared but no width given → conservative 2 (loud-enough to engage
        # EP; a real MoE model should set model_num_experts explicitly).
        return 2
    return 1


def has_model_experts(cfg: Mapping[str, Any]) -> bool:
    """True iff the model declares a model-level MoE (EP-eligible, the 5th axis)."""
    return model_num_experts(cfg) > 1


def _pick_base_3d(world: int, *, prefer_tp: int = 0) -> Tuple[int, int, int]:
    """Pick (DP, TP, PP) for the base 3D mesh honoring DP·TP·PP == world.

    DEFAULT POLICY (run_harness.md §0): for the flagship single-model saturation
    the recommendation is TP = world (TP8 × DP1 × PP1) — TP spreads ONE model
    across all GPUs and is what shrinks per-rank Nmax so the staged-opt scratch
    fits. `prefer_tp` overrides (0 ⇒ the default TP=world). PP stays 1 (PP is
    owner-locked overhead at this depth — run_harness.md §5 / pipeline.py HONEST
    SCOPE). DP is whatever is left after TP·PP.
    """
    tp = prefer_tp if prefer_tp > 0 else world
    if world % tp != 0:
        raise ValueError(f"prefer_tp={tp} does not divide world={world}")
    pp = 1
    dp = world // (tp * pp)
    return dp, tp, pp


def infer_parallel_config(
    model_cfg: Mapping[str, Any],
    *,
    world: int = DEFAULT_WORLD,
    zero_stage: int = 3,
    prefer_tp: int = 0,
    expert_parallel: int = 0,
) -> AdaptivePlan:
    """Infer the ADAPTIVE 3D–5D mesh from a model config (the auto-rule).

    Parameters
    ----------
    model_cfg : the grokking_race_v2 config dict (model_type + any MoE keys).
    world     : device count (default 8 = 8×H100, run_harness.md §0).
    zero_stage: ZeRO stage (default 3 — the flagship ships ZeRO-3).
    prefer_tp : override the base-3D TP degree (0 ⇒ TP=world, the saturation mesh).
    expert_parallel : override the EP degree (0 ⇒ AUTO: world//... when the model
                has experts, else 1). EP sub-divides DP (distributed.py), so it
                must divide the chosen DP.

    Returns an :class:`AdaptivePlan` with the chosen per-axis degrees + the
    ParConfig<...> template string. The returned plan is what the launcher uses to
    pick the instantiation; build the runtime mesh via plan.to_parallel_config().
    """
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")

    seq_eligible = is_sequence_model(model_cfg)
    experts = has_model_experts(model_cfg)
    n_experts = model_num_experts(model_cfg)

    # ── base 3D = DP × TP × PP ──
    dp, tp, pp = _pick_base_3d(world, prefer_tp=prefer_tp)

    # ── +SP (4th): eligible for sequence models, but PINNED 1 this campaign (the
    #    kernel static_assert). We report eligibility but instantiate sp=1, so the
    #    4th axis is unlocked-but-inert (a future long-seq model flips it on by
    #    relaxing the parallel_config.cuh SP assert). ──
    sp = 1  # EXPRESSIBLE but pinned 1 (parallel_config.cuh SP==1 static_assert)

    # ── +EP (5th): engages ONLY when the model declares experts. EP sub-divides
    #    DP (it must divide DP and never enlarges world — distributed.py). AUTO:
    #    spread the experts over as many DP peers as evenly divide DP (cap at DP),
    #    so on a TP=world / DP=1 mesh EP would be 1 even for a MoE model unless DP
    #    is freed up (lower TP). The caller can force EP via expert_parallel. ──
    if not experts:
        ep = 1
    elif expert_parallel > 0:
        if dp % expert_parallel != 0:
            raise ValueError(
                f"expert_parallel={expert_parallel} must divide DP={dp} "
                f"(EP sub-divides the DP group — distributed.py)")
        ep = expert_parallel
    else:
        # AUTO: use the largest divisor of DP that does not exceed n_experts.
        ep = 1
        for cand in range(min(dp, n_experts), 1, -1):
            if dp % cand == 0:
                ep = cand
                break

    if zero_stage not in _ZERO_ENUM:
        raise ValueError(f"zero_stage must be 0..3, got {zero_stage}")

    return AdaptivePlan(
        dp=dp, tp=tp, pp=pp, sp=sp, ep=ep, zero_stage=zero_stage,
        sp_eligible=seq_eligible, has_experts=experts, world=world)


__all__ = [
    "AdaptivePlan",
    "DEFAULT_WORLD",
    "infer_parallel_config",
    "is_sequence_model",
    "has_model_experts",
    "model_num_experts",
]
