"""grokking_optimizers/parallel/flagship_budget.py — per-rank memory budget for the
FLAGSHIP 1.5B decoder under 4D + ZeRO-3, mirroring the LIVE kernel scratch formulas.

This module is the single source of truth for the harness's fit decision. It is a pure
function of (TP, PP, DP, ZeRO-stage, optimizer, batch, nCTA) — NO torch, NO GPU — so the
8x driver can print an exact, provable per-rank budget BEFORE it launches anything (the
`--dry-run` / `--help` contract) and a CPU unit test can pin the numbers.

EVERY constant below is mirrored from a cited live source; a drift in the kernel formulas
should be reflected here (the harness re-derives nothing it cannot show).

SOURCES (read in full):
  * flagship dims + totals : csrc/fused/sm_90/decoder_flagship_layout.cuh
        (SG_DEC_D=1600, SG_DEC_LAYERS=48, SG_DEC_DFF=6400, kDecNumTensors=582,
         kDecTotalElems=1475884899, max kDecSizes = dff*d = 10,240,000)
  * SG2 per-CTA stride     : csrc/fused/sm_90/opt_stage_supergrok2.cuh::sg2_ws_stride
        with the DEFAULT SG2Dims<> (d_model=8, gru_hidden=4, indexer_rank=4,
        csa_compress=4, csa_topk=16)
  * staged-opt aggregate   : csrc/fused/sm_90/fused_decoder_megakernel.cuh
        dec_tc_opt_reduce_floats / dec_tc_muon_floats / dec_tc_looksam_floats /
        dec_tc_sg2_floats / dec_tc_acts_floats (lines 502-650)
  * SG2 state layout       : csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu:279
        [m|v|mu|loss|sharpness|slow|gru_state(total*GH)] => (4+1+GH)*total+1, GH=4
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

# ── Flagship layout constants (decoder_flagship_layout.cuh — single source). ──
FLAGSHIP_D      = 1600
FLAGSHIP_DFF    = 4 * FLAGSHIP_D          # 6400
FLAGSHIP_VOCAB  = 99
FLAGSHIP_SEQ    = 4
FLAGSHIP_LAYERS = 48
FLAGSHIP_TOTAL_PARAMS = 1_475_884_899     # kDecTotalElems
FLAGSHIP_NUM_TENSORS  = 582               # kDecNumTensors
# max per-tensor numel == ff weight dff*d (the kDecMaxTensorNumel the SG2 stride uses).
FLAGSHIP_NMAX = FLAGSHIP_DFF * FLAGSHIP_D  # 10,240,000

# ── SG2Dims<> defaults (opt_stage_supergrok2.cuh template defaults). ──
SG2_D_MODEL      = 8
SG2_GRU_HIDDEN   = 4          # GH — also the state gru_state width
SG2_INDEXER_RANK = 4
SG2_CSA_COMPRESS = 4
SG2_CSA_TOPK     = 16

BYTES_PER_FLOAT = 4
GB = 1024 ** 3                             # report everything in GiB (binary) — ONE unit, no mixing
# H100 SXM5 advertises "80 GB" = 80*1000^3 bytes = 74.5 GiB; the usable capacity after the
# CUDA context + cuBLAS/cuDNN handles + NCCL comm buffers is ~70 GiB. We budget against a
# single, consistent GiB number (no 1000^3-vs-1024^3 mixing) so the fit verdict is honest.
H100_CAPACITY_GIB = 80 * (1000 ** 3) / GB  # 74.51 GiB physical
H100_SAFETY_GIB   = 4.0                     # ctx + handles + NCCL buffers headroom
H100_USABLE_GIB   = H100_CAPACITY_GIB - H100_SAFETY_GIB   # ~70.5 GiB


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def sg2_ws_stride(nmax: int) -> int:
    """Mirror of opt_stage_supergrok2.cuh::sg2_ws_stride<SG2Dims<>>(Nmax) — floats per CTA.
    Conservatively sized for the LARGEST tensor (Nmax). Exact, NOT the ~50 estimate."""
    d, rk, gh = SG2_D_MODEL, SG2_INDEXER_RANK, SG2_GRU_HIDDEN
    ncmax = (nmax + SG2_CSA_COMPRESS - 1) // SG2_CSA_COMPRESS
    topk = SG2_CSA_TOPK if SG2_CSA_TOPK > 1 else 1
    f = 0
    # x_sorted, csa_ctx, hca_ctx, q, win_k, win_v, concat  (7 planes of Nmax*d)
    f += 7 * nmax * d
    f += 2 * ncmax * d          # c_k, c_v
    f += nmax * rk              # qI
    f += ncmax * rk             # kI
    f += nmax * topk            # sel
    f += nmax * gh              # new_gru
    f += nmax                   # expert_out
    f += 2 * _next_pow2(nmax)   # sort keys + idx
    f += 2 * nmax               # perm + unsort
    return f


def dec_sg2_ws_stride_floats(nmax: int = FLAGSHIP_NMAX,
                             n_tensors: int = FLAGSHIP_NUM_TENSORS) -> int:
    # fused_decoder_megakernel.cuh:613 — 2*kDecNumTensors (row_off64 staging) + the stride.
    return 2 * n_tensors + sg2_ws_stride(nmax)


def dec_tc_sg2_floats(nmax: int, ncta: int) -> int:
    return ncta * dec_sg2_ws_stride_floats(nmax) + 1            # :617


def dec_tc_muon_floats(nmax: int, rows: int, ncta: int) -> int:
    return 4 * nmax + rows * rows + ncta + 1                    # :565


def dec_tc_looksam_floats(per_rank_total: int) -> int:
    return 2 * per_rank_total                                   # :582 (sized by the on-rank blob)


def dec_tc_opt_reduce_floats(ncta: int) -> int:
    return 2 * ncta + 1                                         # :551


def dec_tc_acts_floats(B: int, layers: int) -> int:
    """fused_decoder_megakernel.cuh:502 — bf16 acts region in floats (ceil/2)."""
    d, dff, V, seq = FLAGSHIP_D, FLAGSHIP_DFF, FLAGSHIP_VOCAB, FLAGSHIP_SEQ
    T = B * seq
    Td, T3d, Tff = T * d, T * 3 * d, T * dff
    bf = 0
    for _ in range(layers):
        bf += Td + Td + Td + Tff + T3d + Td + Tff + Td
    bf += B * d + B * V + Td
    return (bf + 1) // 2


def sg2_state_floats(total: int) -> int:
    # [m|v|mu|loss|sharpness|slow|gru_state(total*GH)] => (4+1+GH)*total+1
    return (4 + 1 + SG2_GRU_HIDDEN) * total + 1


# Elementwise/per-tensor "extra" state plane counts (state = k*total floats), keyed off
# the launcher state-layout comments. AdamW/Lion/grokfast/grokadamw/neuralgrok/prodigy/
# looksam/muon/SG11/SG15 all fit the 3*total ([m|v|extra]) or 4*total (prodigy/SG) layout;
# SG2 is the 9*total outlier above. We size conservatively per optimizer.
_STATE_PLANES: Dict[str, int] = {
    "adamw": 3, "lion": 3, "grokfast": 3, "grokadamw": 3, "neuralgrok": 3,
    "prodigy": 4, "looksam": 3, "muon": 3, "supergrok11": 5, "supergrok15": 5,
    # supergrok2 handled specially via sg2_state_floats().
}


@dataclasses.dataclass(frozen=True)
class RankBudget:
    opt: str
    tp: int
    pp: int
    dp: int
    zero3: bool
    ncta: int
    B: int
    nmax_per_rank: int
    resident_params: int
    params_gb: float
    state_gb: float
    acts_gb: float
    staged_gb: float
    sg2_gb: float
    total_gb: float

    @property
    def fits(self) -> bool:
        return self.total_gb <= H100_USABLE_GIB     # ~70.5 GiB usable on an 80 GB H100


def per_rank_budget(opt: str, *, tp: int, pp: int, dp: int, zero3: bool,
                    ncta: int, B: int) -> RankBudget:
    """The exact per-rank HBM budget (GiB) for ONE flagship config. The fit gate the
    harness trusts. TP shrinks Nmax (model split); ZeRO-3 shards params+state over DP;
    the transient workspace (acts + staged-opt scratch) is per-rank and Nmax(TP)-sized."""
    opt = opt.lower()
    model_shard = tp * pp
    zero_div = dp if zero3 else 1
    resident_params = FLAGSHIP_TOTAL_PARAMS // (model_shard * zero_div)
    nmax_t = FLAGSHIP_NMAX // tp
    muon_rows = max(FLAGSHIP_DFF // tp, 1)
    layers_per_rank = FLAGSHIP_LAYERS // pp

    if opt == "supergrok2":
        state_floats = sg2_state_floats(FLAGSHIP_TOTAL_PARAMS) // (model_shard * zero_div)
    else:
        planes = _STATE_PLANES.get(opt)
        if planes is None:
            raise ValueError(f"unknown optimizer {opt!r} for budget")
        state_floats = planes * FLAGSHIP_TOTAL_PARAMS // (model_shard * zero_div)

    acts = dec_tc_acts_floats(B, layers_per_rank)
    # looksam scratch is sized by the on-rank (TP/PP-resident, pre-ZeRO) blob.
    per_rank_resident_pretotal = FLAGSHIP_TOTAL_PARAMS // model_shard
    staged = (dec_tc_opt_reduce_floats(ncta)
              + dec_tc_muon_floats(nmax_t, muon_rows, ncta)
              + dec_tc_looksam_floats(per_rank_resident_pretotal)
              + dec_tc_sg2_floats(nmax_t, ncta))
    sg2_only = dec_tc_sg2_floats(nmax_t, ncta)

    def gb(floats_or_params, bpf=BYTES_PER_FLOAT):
        return floats_or_params * bpf / GB

    params_gb = gb(resident_params)
    state_gb = gb(state_floats)
    acts_gb = gb(acts)
    staged_gb = gb(staged)
    sg2_gb = gb(sg2_only)
    total_gb = params_gb + state_gb + acts_gb + staged_gb + 0.10  # + tile-scratch slack
    return RankBudget(opt=opt, tp=tp, pp=pp, dp=dp, zero3=zero3, ncta=ncta, B=B,
                      nmax_per_rank=nmax_t, resident_params=resident_params,
                      params_gb=params_gb, state_gb=state_gb, acts_gb=acts_gb,
                      staged_gb=staged_gb, sg2_gb=sg2_gb, total_gb=total_gb)


def auto_ncta(opt: str, *, tp: int, pp: int, dp: int, zero3: bool, B: int,
              n_sms: int = 132) -> int:
    """Largest nCTA (down from one-CTA/SM) whose per-rank budget fits 80 GB for `opt`.
    Returns n_sms when the full-occupancy config already fits (the common case at TP=8)."""
    for ncta in (n_sms, 64, 32, 16, 8, 4, 2, 1):
        if ncta > n_sms:
            continue
        b = per_rank_budget(opt, tp=tp, pp=pp, dp=dp, zero3=zero3, ncta=ncta, B=B)
        if b.fits:
            return ncta
    return 1


# The 11-optimizer ranking benchmark roster (opt_components.cuh OptId order). SG2 is the
# memory worst case; the harness budgets EVERY opt and reports the binding one.
ALL_OPTIMIZERS: Tuple[str, ...] = (
    "adamw", "lion", "grokfast", "grokadamw", "looksam", "prodigy",
    "neuralgrok", "muon", "supergrok11", "supergrok15", "supergrok2",
)

# The recommended flagship config (the §0 headline).
RECOMMENDED = dict(tp=8, pp=1, dp=1, zero3=True)


def format_budget_table(*, tp: int, pp: int, dp: int, zero3: bool, B: int,
                        ncta: Optional[int] = None, n_sms: int = 132) -> str:
    """Human-readable per-optimizer budget table for the chosen mesh — printed by
    `--help`/`--dry-run` so the operator SEES the fit proof before any GPU work."""
    lines: List[str] = []
    lines.append(f"flagship 1.5B decoder per-rank budget  (mesh TP={tp} PP={pp} DP={dp} "
                 f"ZeRO-3={zero3}  B={B}  n_sms={n_sms})")
    lines.append(f"  world_size = TP*PP*DP = {tp*pp*dp}")
    lines.append(f"  {'optimizer':<13}{'nCTA':>6}{'Nmax/rank':>12}{'par':>7}{'state':>7}"
                 f"{'acts':>7}{'staged':>9}{'(sg2)':>9}{'TOTAL':>9}  fit")
    for opt in ALL_OPTIMIZERS:
        nc = ncta if ncta is not None else auto_ncta(opt, tp=tp, pp=pp, dp=dp,
                                                     zero3=zero3, B=B, n_sms=n_sms)
        b = per_rank_budget(opt, tp=tp, pp=pp, dp=dp, zero3=zero3, ncta=nc, B=B)
        lines.append(f"  {opt:<13}{nc:>6}{b.nmax_per_rank:>12,}{b.params_gb:>7.2f}"
                     f"{b.state_gb:>7.2f}{b.acts_gb:>7.2f}{b.staged_gb:>9.2f}"
                     f"{b.sg2_gb:>9.2f}{b.total_gb:>9.2f}  {'FITS' if b.fits else 'OOM'}")
    return "\n".join(lines)


__all__ = [
    "FLAGSHIP_D", "FLAGSHIP_DFF", "FLAGSHIP_VOCAB", "FLAGSHIP_SEQ", "FLAGSHIP_LAYERS",
    "FLAGSHIP_TOTAL_PARAMS", "FLAGSHIP_NUM_TENSORS", "FLAGSHIP_NMAX",
    "sg2_ws_stride", "dec_sg2_ws_stride_floats", "dec_tc_sg2_floats", "dec_tc_acts_floats",
    "sg2_state_floats", "RankBudget", "per_rank_budget", "auto_ncta",
    "ALL_OPTIMIZERS", "RECOMMENDED", "format_budget_table",
]
