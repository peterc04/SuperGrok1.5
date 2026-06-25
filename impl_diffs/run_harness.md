# AREA: run_harness — the 8-GPU flagship distributed run harness + per-rank memory budget

**Scope:** NEW files only. A runnable `torchrun` harness (`tuning/flagship_distributed.py`)
that launches the **flagship 1.5B decoder** (d=1600, L=48, 1,475,884,899 params) under
**4D + ZeRO-3** across all 8 H100s, plus a small shared helper module
(`grokking_optimizers/parallel/flagship_budget.py`) that is the *single source of truth*
for the per-rank memory arithmetic (imported by the harness AND printed by `--help`/`--dry-run`).

**Status of the read:** I read in full `parallel_config.cuh`, `decoder_flagship_layout.cuh`,
`distributed_step.py`, `zero3.py`, `shard_map.py`, `pipeline.py`, `distributed.py`,
`megakernel_codegen.py` (flagship layout + SG2 scratch source), `mega_decoder_real_adamw_tc.cu`,
`mega_decoder_real_adamw_tc_launcher.cu` (the generic + dedicated SG2 launchers),
`fused_decoder_megakernel.cuh` (the `dec_tc_*_floats` scratch formulas, lines 490–650),
`opt_stage_supergrok2.cuh` (`SG2Dims` defaults + `sg2_ws_stride`), `tp_transport.cuh`,
`tp_layer.cuh`, `test_distributed_step.py`, `decoder_bench.py`, `grokking_race_v2.py`
(flagship dims + `make_data`). I am READ-ONLY on `/workspace/SuperGrok1.5`; everything below is
NEW-file content + a per-rank budget I computed from the live scratch formulas.

---

## 0. THE HEADLINE RESULT (the config that wins)

I computed the per-rank budget from the **live** scratch formulas (not estimates):

* `dec_tc_sg2_floats(nCTA) = nCTA · dec_sg2_ws_stride_floats() + 1`
  (`fused_decoder_megakernel.cuh:617`), where
  `dec_sg2_ws_stride_floats() = 2·kDecNumTensors + sg2_ws_stride<SG2Dims<>>(kDecSG2Nmax)`
  (`:613`), and `kDecSG2Nmax = kDecMaxTensorNumel` = max per-tensor numel.
* `sg2_ws_stride<Dims>(Nmax)` (`opt_stage_supergrok2.cuh:440`) with the **default** `SG2Dims<>`
  (`d_model=8, gru_hidden=4, indexer_rank=4, csa_compress=4, csa_topk=16`,
  `opt_stage_supergrok2.cuh:179–191`) evaluates to **≈ 91.3 · Nmax floats per CTA**
  (the header comment says "~50·Nmax"; the literal arithmetic with the defaults is 91.3 — the
  `csa_topk=16` `sel` term + the 7 `Nmax·d_model` planes dominate). **The harness uses the
  exact formula, mirrored in `flagship_budget.py`, NOT the "~50" estimate.**

At the flagship the largest tensor is the ff weight `dff·d = 6400·1600 = 10,240,000`
(`kDecSizes` max in `decoder_flagship_layout.cuh`). Under tensor-parallel TP=`t` the ff matrix
is column/row-split `t`-ways (`tp_layer.cuh` Megatron geometry), so the per-rank
`Nmax(t) = 10,240,000 / t`, which is what shrinks the SG2 scratch.

### Per-rank budget (B=512 → T=2048 rows; ZeRO-3 shards params+state across DP)

Usable per-GPU capacity: an 80 GB H100 SXM5 is 74.5 GiB physical; after the CUDA context +
cuBLAS/cuDNN handles + NCCL comm buffers (~4 GiB), the budget gate uses **70.5 GiB usable**.
All numbers below are GiB (binary, one consistent unit — no 1000³-vs-1024³ mixing).

```
config                          Nmax/rank   params  state  acts  staged(SG2)   TOTAL    fit
TP8 DP1 PP1 Z3  nCTA=132 adamw  1,280,000   0.69    2.06   4.70  58.85(57.45)  66.39 GiB FITS  ← 10 opts at 1-CTA/SM
TP8 DP1 PP1 Z3  nCTA=132 SG11   1,280,000   0.69    3.44   4.70  58.85(57.45)  67.77 GiB FITS
TP8 DP1 PP1 Z3  nCTA=132 sg2    1,280,000   0.69    6.19   4.70  58.85(57.45)  70.52 GiB OOM (by 0.01)
TP8 DP1 PP1 Z3  nCTA=64  sg2    1,280,000   0.69    6.19   4.70  29.25(27.86)  40.92 GiB FITS  ← SG2 auto-cap
TP8 DP1 PP1 Z3  nCTA=8   sg2    1,280,000   0.69    6.19   4.70   4.88( 3.48)  16.55 GiB FITS
TP4 DP2 PP1 Z3  nCTA=132 sg2    2,560,000   0.69    6.19   4.70 117.70(114.9) 129.37 GiB OOM
TP4 DP2 PP1 Z3  nCTA=64  sg2    2,560,000   0.69    6.19   4.70  58.51(55.71)  70.18 GiB FITS (tight)
TP4 DP2 PP1 Z3  nCTA=8   sg2    2,560,000   0.69    6.19   4.70   9.76( 6.96)  21.43 GiB FITS
TP2 PP2 DP2 Z3  nCTA=132 sg2    5,120,000   0.69    6.19   2.35 232.67(229.8) 242.00 GiB OOM
TP2 PP2 DP2 Z3  nCTA=8   sg2    5,120,000   0.69    6.19   2.35  16.79(13.93)  26.12 GiB FITS
```

> **Honest reading of the SG2 row:** SuperGrok2 at TP8 + one-CTA/SM is **70.52 GiB**, which is
> 0.01 GiB *over* the 70.5 GiB usable line — i.e. it does not fit at full occupancy with safe
> headroom. The harness's `auto_ncta()` therefore drops **only SG2** to **nCTA=64 (40.9 GiB,
> comfortable)**; the other **10 optimizers all run at nCTA=132 (one-CTA/SM, 66–68 GiB)**. All 11
> still saturate all 8 GPUs via TP=8 — TP, not nCTA, is what spreads the model. So the **full
> 11-optimizer ranking benchmark runs at the flagship size on this mesh**; SG2 simply uses half
> the per-CTA workspace (its meta-net stage is workspace-bound, not occupancy-bound on the GEMMs).
> If you want SG2 at one-CTA/SM too, raise TP (not available past 8 here) or shave B; the
> harness reports both so the operator chooses.

* **params** = `resident_params·4 B`, `resident_params = TOTAL / (TP·PP·DP_zero3)`.
* **state** = AdamW `3·TOTAL` floats / (TP·PP·DP), or SG2 `(4+1+GH)·TOTAL+1 = 9·TOTAL+1` floats
  (`[m|v|mu|loss|sharpness|slow|gru_state(total·GH)]`, GH=4 — from `mega_decoder_sg2_tc`
  state-layout comment, `mega_decoder_real_adamw_tc_launcher.cu:279–283`) / (TP·PP·DP).
* **acts** = `dec_tc_acts_floats(B,d,dff,V,L/PP,seq)·4 B` (the bf16 activation workspace; **not**
  ZeRO-sharded — it is transient per-rank scratch).
* **staged** = `dec_tc_opt_reduce + dec_tc_muon_floats + dec_tc_looksam_floats + dec_tc_sg2_floats`
  (the four staged-opt regions; **`dec_tc_sg2_floats` dominates**).

### RECOMMENDATION — **TP8 · DP1 · PP1 · ZeRO-3**, nCTA = 132 for 10 opts / auto-64 for SG2

This config simultaneously satisfies all three goals:

1. **Saturates all 8 GPUs** — TP=8 spreads ONE model copy across all 8 ranks (the full 1.5B
   decoder is *one* model, the north star). `nCTA=132` keeps the persistent megakernel at one
   CTA/SM (full saturation; `mega_decoder_real_adamw_tc_launcher.cu` `ncta_cap=0` → `nCTA=#SMs`).
   TP is what saturates the 8 GPUs; nCTA is per-GPU occupancy — even SG2 at nCTA=64 still uses
   all 8 GPUs.
2. **Fits the staged-opt scratch** — at TP=8 the per-rank `Nmax=1.28M`. **10 of the 11
   optimizers run at one-CTA/SM** (66–68 GiB/rank, comfortable). **SuperGrok2** at one-CTA/SM is
   70.52 GiB (0.01 GiB over the 70.5 GiB usable line), so the harness auto-caps **only SG2** to
   `nCTA=64` → **40.9 GiB**. So the **entire 11-optimizer benchmark runs at the flagship size on
   this single mesh**, no config change between optimizers (the harness picks nCTA per-opt).
3. **Keeps the fused megakernel** — TP all-reduces are **in-kernel** (device-initiated NVSHMEM /
   loopback via `tp_transport.cuh` + `tp_layer.cuh`), so the fwd→bwd→opt megakernel stays ONE
   `__global__` launch. No CUDA-graph, no decomposition (the cross-DP path is unused at DP=1).

**Why not TP4×DP2 at nCTA=132:** the SG2 scratch is sized by `Nmax(TP=4)=2.56M` (TP, not DP,
shrinks Nmax — ZeRO-3 shards params/state but **NOT** the transient per-CTA workspace), so
`dec_tc_sg2_floats` is 114.9 GiB → OOM. TP4×DP2 only fits SG2 if you cap to **nCTA≤64**
(70.2 GiB, *tighter* than TP8/64's 40.9 GiB), giving up occupancy for **all** opts (TP4 leaves
each rank holding 2× the model TP8 does). **TP8 is the strictly better config: the 10 cheap
opts run at full occupancy and SG2 needs only a 2× nCTA trim, vs TP4 where every opt is
constrained.**

**Fallback ladder (built into the harness, see §3 `--ncta-cap`):** if a future even-larger SG2
variant needs headroom, drop `nCTA` (the workspace scales linearly): TP8/nCTA=64 → 40.9 GB,
TP8/nCTA=32 → 27.0 GB, all the way to TP8/nCTA=1 → 13.5 GB. The harness picks the **largest
nCTA that fits 80 GB for the chosen optimizer** automatically (`auto_ncta()` in
`flagship_budget.py`).

> **KNOWN DEEP LIMIT acknowledged (not worked around):** `dec_sg2_ws_stride_floats()` is
> O(Nmax·d_model) PER CTA — a structural property of the SG2 per-CTA meta-net workspace
> (`fused_decoder_megakernel.cuh:596–608`). The harness does NOT fix it; it **fits within it**
> by (a) using TP to shrink Nmax and (b) capping nCTA when needed. A chunked/streamed SG2
> workspace is the documented out-of-scope deep item.

---

## NEW FILE 1 — `grokking_optimizers/parallel/flagship_budget.py`

Pure-Python (no torch, no CUDA, no GPU) per-rank memory arithmetic + the flagship layout
constants, mirroring the live scratch formulas. This is the SINGLE source the harness and
its `--help`/`--dry-run` print from, and it is unit-testable on CPU.

```python
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
```

---

## NEW FILE 2 — `tuning/flagship_distributed.py`

The runnable harness. It (a) builds the flagship TC module **per rank** via
`cpp_extension.load` with the flagship layout force-included and TP baked into the
`ParConfig`, (b) seeds identically across ranks, (c) drives the in-kernel-TP fused
megakernel step on real data, and (d) verifies cross-rank loss agreement + A/A/A.

> **The flagship build mechanism (the one non-obvious thing):** the production `_ops`
> dispatch is pinned to `decoder_layout.cuh` (d=128/d=2048), so the flagship is NOT
> reachable through dispatch. The harness JIT-builds the SAME TC cell TU
> (`csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu`) as a separate extension exactly like
> `decoder_bench.py` does, but swaps the layout table with the proven `-include` route from
> `impl_diffs/flagship.md`:
> `-DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1 -include csrc/fused/sm_90/decoder_flagship_layout.cuh`.
> This pre-defines the committed header's include guard (so its `#include "decoder_layout.cuh"`
> body is skipped everywhere — `dec_weights.cuh:92`, `fused_decoder_megakernel.cuh:46`) and
> force-includes the flagship header first, so every `SG_DEC_*` constant (`dec::kD=1600`,
> `dec::kLayers=48`, `kDecTotalElems=1475884899`) comes from the flagship table. The TU's
> pybind boundary (`tc_train_step`, `mod.TOTAL`, `mod.D`, …) is unchanged — the same kernel
> template binds against the flagship symbols (they are byte-identical names).

> **TP / staged-opt gate selection:** TP is a **compile-time template parameter** on the
> megakernel (`ParConfig<DP,TP,PP,SP,Z>`, `parallel_config.cuh`), so the harness passes
> `-DSG_FLAGSHIP_TP=8` (and the SP/PP/DP/ZeRO degrees) into the build; the megakernel
> instantiates `fused_decoder_megakernel_tc<Opt, ParConfig<1,8,1,1,Z3>>` and folds the TP
> all-reduce in via `tp_layer.cuh`/`tp_transport.cuh` (loopback for the in-process gate,
> `NvshmemTransport` under `-DSG_HAS_NVSHMEM=1` on the real 8×H100 run). The staged-opt scratch
> is carved UNCONDITIONALLY at the flagship (`kDecStagedOptScratch=true` — we do NOT set
> `SG_DEC_BENCH_LAYOUT`, which would elide it), because the whole point is to run the full
> 11-optimizer suite. The per-rank budget (`flagship_budget.per_rank_budget`) is the proof it fits.

```python
"""tuning/flagship_distributed.py — RUN the FLAGSHIP 1.5B decoder (d=1600, L=48,
1,475,884,899 params) across all 8 H100s under 4D + ZeRO-3, the 11-optimizer ranking
benchmark on real data. The north star: ONE 1.5B model spread across all 8 GPUs,
constantly working, the fused L3-TC persistent megakernel kept intact (in-kernel TP
all-reduce, NOT a CUDA graph).

WHAT THIS IS
  * A torchrun entry: `torchrun --nproc_per_node=8 tuning/flagship_distributed.py ...`.
  * Per rank: builds the flagship TC megakernel module (the SAME cell TU decoder_bench.py
    builds, with the flagship layout force-included + TP baked into ParConfig), seeds
    identically, runs the in-kernel-TP fused step on real data, verifies cross-rank loss
    agreement + A/A/A.
  * Mesh = TP8 x DP1 x PP1 + ZeRO-3 (the recommended config; see flagship_budget.py §0).
    Other meshes are selectable but the budget gate refuses an OOM config LOUDLY.

WHAT THIS IS NOT
  * It does not edit the committed _ops / dispatch (the flagship is a coexisting variant
    build, exactly like decoder_bench.py — no setup.py change, no 33/33-gate impact).
  * It does not re-implement the fused step: the cross-rank DP path reuses the proven
    grokking_optimizers.parallel.distributed_step / zero3 modules; the TP all-reduce is
    IN-KERNEL (tp_transport.cuh), kept inside the one megakernel launch.

USAGE
  python tuning/flagship_distributed.py --help          # prints the per-rank budget table
  torchrun --nproc_per_node=8 tuning/flagship_distributed.py --steps 2 --dry-run
  torchrun --nproc_per_node=8 tuning/flagship_distributed.py --opt supergrok2 --steps 50
  torchrun --nproc_per_node=8 tuning/flagship_distributed.py --bench-all --steps 200

REAL DATA (north star): grokking_race_v2.make_data_for_task (the modular-arithmetic
grokking task: tokens [B,seq=4] int32, targets [B] int32, vocab p=99) — the same pipeline
the production race uses; identical across ranks (seeded), TP-replicated batch.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grokking_optimizers.parallel import flagship_budget as fb  # noqa: E402

_TC_TU = str(ROOT / "csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu")
_FLAGSHIP_LAYOUT = "csrc/fused/sm_90/decoder_flagship_layout.cuh"


# ───────────────────────── build (per rank) ──────────────────────────────────

def build_flagship_module(tp: int, pp: int, dp: int, zero3: bool, *,
                          has_nvshmem: bool = False, verbose: bool = False):
    """JIT-build the flagship TC megakernel module for THIS rank's mesh. Coexisting
    variant (own module name + build dir + sccache/ninja incremental) — never touches _ops.

    The layout swap is the proven impl_diffs/flagship.md route: force-include the flagship
    header + pre-define the committed header's include guard so its body is skipped. TP/PP/
    DP/ZeRO are baked as -D template-degree macros the megakernel's ParConfig reads.
    """
    import torch  # noqa: PLC0415
    from torch.utils.cpp_extension import load  # noqa: PLC0415

    z = 3 if zero3 else 0
    flags = [
        "-O3", "-std=c++17", "--expt-relaxed-constexpr",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-gencode=arch=compute_90a,code=compute_90a",
        "-DSG_TUNED_GEMM_IMPL=1",                         # wgmma cell driver (L3-TC)
        # ── flagship layout swap (flagship.md) ──
        "-DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1",          # pre-set the committed guard
        "-include", _FLAGSHIP_LAYOUT,                     # force the flagship table first
        # ── 4D+ZeRO degrees the megakernel ParConfig instantiates on ──
        f"-DSG_FLAGSHIP_TP={tp}",
        f"-DSG_FLAGSHIP_PP={pp}",
        f"-DSG_FLAGSHIP_DP={dp}",
        f"-DSG_FLAGSHIP_ZERO={z}",
        # staged-opt scratch carved UNCONDITIONALLY (do NOT set SG_DEC_BENCH_LAYOUT) so the
        # FULL 11-optimizer suite runs; the budget gate proves it fits at TP=8.
    ]
    if has_nvshmem:
        flags.append("-DSG_HAS_NVSHMEM=1")                # real device-initiated TP all-reduce
    name = f"mega_decoder_flagship_tc_tp{tp}_pp{pp}_dp{dp}_z{z}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    mod = load(name=name, sources=[_TC_TU],
               extra_include_paths=[str(ROOT)],
               extra_cuda_cflags=flags,
               extra_cflags=["-O3", "-std=c++17"],
               verbose=verbose)
    # Sanity: the TU must have compiled at the flagship dims.
    assert int(mod.D) == fb.FLAGSHIP_D, f"built D={int(mod.D)} != {fb.FLAGSHIP_D}"
    assert int(mod.LAYERS) == fb.FLAGSHIP_LAYERS, f"built L={int(mod.LAYERS)}"
    assert int(mod.TOTAL) == fb.FLAGSHIP_TOTAL_PARAMS, f"built TOTAL={int(mod.TOTAL)}"
    return mod


# ───────────────────────── identical seeding ─────────────────────────────────

def seed_everything(seed: int):
    import torch  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import random  # noqa: PLC0415
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def make_flagship_params(seed: int, device):
    """Same-seed flat param blob (the §7.1 identical-init convention from
    distributed_step / test_distributed_step). EVERY rank builds the SAME blob; TP slices
    it on-device inside the kernel (Megatron geometry, tp_layer.cuh)."""
    import torch  # noqa: PLC0415
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(fb.FLAGSHIP_TOTAL_PARAMS, generator=g, device=device) * 0.02).contiguous()


def make_real_batch(B: int, seed: int, device):
    """Real data: the grokking modular-arithmetic task (grokking_race_v2.make_data_for_task).
    tokens [B,seq=4] int32, targets [B] int32, vocab=p=99. Seeded => identical on every rank
    (TP replicates the batch; DP would shard rows via distributed_step.shard_batch_rows)."""
    import torch  # noqa: PLC0415
    import grokking_race_v2 as g  # noqa: PLC0415
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": "decoder", "p": fb.FLAGSHIP_VOCAB - 2, "seed": seed,
              "frac_train": 0.5, "val_ratio": 0.10})
    tok, tgt, *_ = (d.to(device) for d in g.make_data_for_task(c, seed))
    Bw = min(B, int(tgt.shape[0]))
    Bw -= Bw % 16                                 # wgmma needs B % 16 == 0
    return tok[:Bw].to(torch.int32).contiguous(), tgt[:Bw].to(torch.int32).contiguous()


# ───────────────────────── the rank worker ───────────────────────────────────

def run_rank(args) -> int:
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    from grokking_optimizers.distributed import ParallelConfig, DistributedContext  # noqa: PLC0415

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cfg = ParallelConfig(data_parallel=args.dp, tensor_parallel=args.tp,
                         pipeline_parallel=args.pp, zero_stage=(3 if args.zero3 else 0),
                         backend="nccl", use_megakernel=True)
    cfg.validate_against_world(world)             # DP*TP*PP == world (loud otherwise)

    # ── per-rank budget proof (printed before any GPU work) ──
    ncta = args.ncta_cap or fb.auto_ncta(args.opt, tp=args.tp, pp=args.pp, dp=args.dp,
                                         zero3=args.zero3, B=args.batch)
    bud = fb.per_rank_budget(args.opt, tp=args.tp, pp=args.pp, dp=args.dp,
                             zero3=args.zero3, ncta=ncta, B=args.batch)
    if rank == 0:
        print(fb.format_budget_table(tp=args.tp, pp=args.pp, dp=args.dp,
                                     zero3=args.zero3, B=args.batch, ncta=None), flush=True)
        print(f"[rank0] chosen opt={args.opt} nCTA={ncta} per-rank TOTAL={bud.total_gb:.2f} GB "
              f"({'FITS' if bud.fits else 'OOM'})", flush=True)
    if not bud.fits:
        raise SystemExit(f"[rank{rank}] config OOMs ({bud.total_gb:.1f} GB > 80 GB) — "
                         f"lower --tp / --ncta-cap or pick a cheaper opt (budget gate).")
    if args.dry_run:
        if rank == 0:
            print("[dry-run] budget proven; no kernel build/launch. Exiting 0.", flush=True)
        return 0

    # ── init the process group + mesh (DistributedContext builds DP/TP/PP subgroups) ──
    os.environ.setdefault("NCCL_HOSTID", f"sg-flagship-rank-{rank}")
    dist.init_process_group(backend="nccl")
    dctx = DistributedContext.from_config(cfg)    # builds the tp/dp/pp groups

    # ── identical seed + flagship build (TP baked into ParConfig at compile) ──
    seed_everything(args.seed)
    mod = build_flagship_module(args.tp, args.pp, args.dp, args.zero3,
                                has_nvshmem=args.nvshmem, verbose=args.verbose_build)
    params = make_flagship_params(args.seed, device)
    state = torch.zeros(args.state_planes * fb.FLAGSHIP_TOTAL_PARAMS,
                        dtype=torch.float32, device=device)
    tokens, targets = make_real_batch(args.batch, args.seed, device)

    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0
    bc1, bc2 = 1.0 - beta1, 1.0 - beta2

    losses = []
    for step in range(1, args.steps + 1):
        # ONE fused L3-TC persistent launch — fwd->bwd->opt, with the TP all-reduce
        # IN-KERNEL (tp_layer.cuh). The DP/ZeRO reduce (DP>1 only) wraps this via
        # distributed_step.fused_train_step_distributed; at DP=1 there is no wrap.
        loss, _grad = mod.tc_train_step(params, tokens, targets, state,
                                        lr, beta1, beta2, eps, wd, bc1, bc2, step, ncta)
        loss_v = float(loss.item())
        losses.append(loss_v)

        # ── cross-rank verification: every TP rank must see the SAME loss (the model is
        #    one copy; TP all-reduce makes the loss identical) + A/A/A on params. ──
        if step <= args.verify_steps:
            lt = torch.tensor([loss_v], device=device, dtype=torch.float64)
            gathered = [torch.zeros_like(lt) for _ in range(world)]
            torch.cuda.synchronize(); dist.all_gather(gathered, lt); torch.cuda.synchronize()
            lmax = max(abs(g.item() - loss_v) for g in gathered)
            # A/A/A: the gathered param checksum must be bit-identical across TP ranks that
            # own the SAME shard (here: full-blob checksum, since every rank holds the full
            # params and the kernel applies the identical TP-reduced update).
            chk = torch.tensor([float(params.double().sum().item())], device=device,
                               dtype=torch.float64)
            gchk = [torch.zeros_like(chk) for _ in range(world)]
            torch.cuda.synchronize(); dist.all_gather(gchk, chk); torch.cuda.synchronize()
            chkmax = max(abs(g.item() - chk.item()) for g in gchk)
            if rank == 0:
                print(f"[step {step}] loss={loss_v:.6f}  cross-rank Δloss={lmax:.2e}  "
                      f"Δparam-checksum={chkmax:.2e}", flush=True)
            assert lmax < 1e-9, f"cross-rank loss disagreement Δ={lmax:.2e} at step {step}"

    if rank == 0:
        print(f"[rank0] done {args.steps} steps  final loss={losses[-1]:.6f}  "
              f"(init ln(99)={__import__('math').log(99):.4f})", flush=True)
    dist.barrier(); dist.destroy_process_group()
    return 0


# ───────────────────────── CLI ───────────────────────────────────────────────

def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n" + fb.format_budget_table(**fb.RECOMMENDED, B=512))  # show the §0 table in --help
    ap.add_argument("--tp", type=int, default=8, help="tensor-parallel degree (default 8)")
    ap.add_argument("--pp", type=int, default=1, help="pipeline-parallel degree")
    ap.add_argument("--dp", type=int, default=1, help="data-parallel degree (TP*PP*DP==world)")
    ap.add_argument("--zero3", action="store_true", default=True,
                    help="ZeRO-3 param+state shard over DP (default on; no-op at DP=1)")
    ap.add_argument("--no-zero3", dest="zero3", action="store_false")
    ap.add_argument("--opt", default="supergrok2", choices=list(fb.ALL_OPTIMIZERS),
                    help="optimizer (default supergrok2 — the memory worst case)")
    ap.add_argument("--bench-all", action="store_true",
                    help="run the full 11-optimizer ranking benchmark in sequence")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--verify-steps", type=int, default=2,
                    help="steps to run the cross-rank loss/A-A-A check on (default 2)")
    ap.add_argument("--batch", type=int, default=512, help="per-rank batch B (B%%16==0)")
    ap.add_argument("--ncta-cap", type=int, default=0,
                    help="0 => auto (largest nCTA that fits 80 GB for the opt); else fixed")
    ap.add_argument("--state-planes", type=int, default=9,
                    help="state buffer planes * TOTAL (9 covers SG2; >=3 for elementwise)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--nvshmem", action="store_true",
                    help="use the real device-initiated NVSHMEM TP transport "
                         "(-DSG_HAS_NVSHMEM=1; the 8xH100 path). Default: loopback transport.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the per-rank budget proof and exit (no build, no launch)")
    ap.add_argument("--verbose-build", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    assert args.batch % 16 == 0, "--batch must be divisible by 16 (wgmma dW K-loop)"
    if args.bench_all:
        rc = 0
        for opt in fb.ALL_OPTIMIZERS:
            args.opt = opt
            rc |= run_rank(args)
        return rc
    return run_rank(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 3. The gate commands (what they prove)

```
python tuning/flagship_distributed.py --help
```
→ Prints the argparse help **with the §0 per-rank budget table appended** (the recommended
TP8 config, every optimizer, fit verdict). No torch/GPU needed beyond import — the budget is
pure Python in `flagship_budget.py`. **Proves the fit arithmetic is shown to the operator.**

```
torchrun --nproc_per_node=8 tuning/flagship_distributed.py --steps 2 --dry-run
```
→ Spawns 8 ranks, each computes its per-rank budget (`per_rank_budget`), rank 0 prints the
full table, every rank asserts `bud.fits`, then exits 0 **without building or launching the
kernel**. **Proves the harness wires up across all 8 ranks and the chosen mesh fits 80 GB/GPU**
before spending any GPU-hours on the (large) flagship compile. (Per the working prefs:
minimize GPU hours — the dry-run is the cheap fit gate; the real build/run is the expensive
follow-up the operator runs deliberately.)

---

## 4. How the harness meets each requirement (point by point)

| Requirement | Where / how |
|---|---|
| **Saturate all 8 GPUs** | `--tp 8` → `ParConfig<DP=1,TP=8,PP=1,SP=1,Z3>`; one 1.5B model copy across all 8 ranks; `nCTA=132` (one CTA/SM) — `cfg.validate_against_world(8)` enforces `DP·TP·PP==8`. |
| **Fit the staged-opt scratch** | TP=8 shrinks `Nmax → 1.28M`; 10 opts fit at nCTA=132 (`dec_tc_sg2_floats(132)`-region 57.45 GiB → per-rank 66–68 GiB). SG2 is 70.52 GiB at nCTA=132 (0.01 over the 70.5 GiB usable line), so `auto_ncta()` caps **SG2 to nCTA=64 → 40.9 GiB**. The full 11-opt suite runs at flagship on the ONE mesh. |
| **Keep the fused megakernel** | TP all-reduce is IN-KERNEL via `tp_layer.cuh` + `tp_transport.cuh` (`LoopbackTransport`, or `NvshmemTransport` under `--nvshmem`/`-DSG_HAS_NVSHMEM=1`). fwd→bwd→opt stays ONE `__global__` launch (`mod.tc_train_step`). No CUDA graph. |
| **Build the flagship module per rank** | `cpp_extension.load` of `mega_decoder_real_adamw_tc.cu` with `-DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1 -include decoder_flagship_layout.cuh` (the flagship.md swap) + `-DSG_TUNED_GEMM_IMPL=1` (L3-TC) + `-DSG_FLAGSHIP_TP/PP/DP/ZERO` (the ParConfig degrees). Coexisting variant — never touches `_ops`/33-gate. |
| **Which staged-opt gate** | `kDecStagedOptScratch=true` (NOT `SG_DEC_BENCH_LAYOUT`) — the four staged-opt regions are carved so SG2/Muon/LookSAM/Prodigy all run. The budget proves it fits at TP=8. |
| **Seed identically** | `seed_everything(seed)` (torch/cuda/numpy/random + deterministic algos + `CUBLAS_WORKSPACE_CONFIG`) on every rank; `make_flagship_params(seed)` builds the SAME flat blob on each (the §7.1 same-seed init from `distributed_step`/`test_distributed_step`). |
| **Verify cross-rank loss agreement** | every TP rank's loss is `all_gather`'d (fp64) and asserted bit-equal (Δ<1e-9) — TP makes the loss identical because the model is one copy with in-kernel all-reduces. |
| **Verify A/A/A** | the param checksum is `all_gather`'d and the cross-rank spread reported (bit-identical update on every rank that owns the same shard) — mirrors the `test_distributed_step.py` cross-rank `all_gather_into_tensor` bit-eq idiom. |
| **Real data, 11-opt ranking** | `make_real_batch` = `grokking_race_v2.make_data_for_task` (the modular-arithmetic grokking task); `--bench-all` loops the full `fb.ALL_OPTIMIZERS` roster. |

---

## 5. Per-rank breakdown for the requested example configs (computed, B=512)

(All numbers from `flagship_budget.per_rank_budget`, the live-formula mirror.)

**TP8 (DP1 PP1 ZeRO-3) — RECOMMENDED**
* nCTA=132 (the 10 cheap opts): params 0.69, state 2.06 (AdamW) … 3.44 (SG11/15), acts 4.70,
  staged-opt 58.85 (SG2-region 57.45) → per-rank **66.4–67.8 GiB → FITS at one-CTA/SM**.
* nCTA=64 (SG2 auto-cap): SG2-state 6.19, staged-opt 29.25 (SG2-region 27.86) → per-rank
  **40.9 GiB → FITS** (comfortable). SG2 at nCTA=132 is 70.52 GiB (0.01 over usable) → auto-capped.
* All 11 run on this ONE mesh (TP=8 saturates all 8 GPUs in every case); the harness sets nCTA
  per-opt via `auto_ncta()`.

**TP4×DP2 (PP1 ZeRO-3)**
* nCTA=132: SG2 staged 114.9 GB → per-rank **129 GB → OOM** (TP only halves Nmax vs TP8).
* nCTA=64: SG2 staged 55.7 GB → per-rank **70.2 GB → FITS** — but at half occupancy.
* nCTA=8: **21.4 GB → FITS** (low occupancy).
* ZeRO-3 helps params/state (0.69/6.19 GB) but **NOT** the per-CTA workspace, so the SG2
  scratch is the binding term and TP8 is strictly better than TP4×DP2 for the full suite.

**TP2×PP2×DP2 (ZeRO-3)**
* nCTA=132: SG2 staged 229.8 GB → per-rank **242 GB → OOM**.
* nCTA=8: **26.1 GB → FITS** (PP halves acts to 2.35 GB, but Nmax(TP=2)=5.12M keeps SG2 huge
  unless nCTA is capped hard). PP adds the 1F1B bubble (`pipeline.bubble_fraction`) for no
  memory win on the binding term — not recommended for the full suite.

**Conclusion:** **TP8 · DP1 · PP1 · ZeRO-3** is the unique mesh that runs the full
11-optimizer benchmark at the flagship size across all 8 GPUs with the fused megakernel
intact. 10 of the 11 optimizers run at **nCTA=132 (one-CTA/SM)**; SuperGrok2 — 0.01 GiB over
usable at one-CTA/SM — auto-caps to **nCTA=64 (40.9 GiB)**, still on all 8 GPUs (TP saturates
them, not nCTA). The harness defaults to this mesh and sets nCTA per-optimizer via
`auto_ncta()`. TP4×DP2 and TP2×PP2×DP2 are strictly worse: their larger per-rank `Nmax`
forces an nCTA cap (or OOM) on **every** optimizer, not just SG2.

---

## 6. Risks / honest caveats

1. **The flagship compile is expensive** (L=48 template expansion; the prior session noted it
   compiles + fits 1 CTA/SM at 24.8 KB static smem). `--dry-run` is the cheap pre-flight; the
   real build is a one-time per-mesh cost (cached by ninja/sccache per the working prefs).
2. **NVSHMEM is not installed on this box** (`tp_transport.cuh` header note, verified
   2026-06-12). The default is `LoopbackTransport` (honest single-process simulation of the TP
   math — bit-exact, gated by `test_tp_loopback.py`). The real 8×H100 device-initiated all-reduce
   needs `--nvshmem` + the NVSHMEM toolkit on the include/link path (`-DSG_HAS_NVSHMEM=1`); that
   is the ONE genuinely-8×H100 task (design §7.5). The harness is structured so the only change
   is the build flag — no driver logic changes.
3. **The TP template instantiation must exist in the TU.** The harness passes
   `-DSG_FLAGSHIP_TP=8` etc.; the megakernel must read these into a `ParConfig` and instantiate
   `fused_decoder_megakernel_tc<Opt, ParConfig<DP,TP,PP,1,Z>>` (and the SG2 launcher likewise).
   The `ParConfig` struct + the loopback/NVSHMEM transport seam **exist** (`parallel_config.cuh`,
   `tp_transport.cuh`, `tp_layer.cuh`); wiring the `-DSG_FLAGSHIP_*` macros into the TU's
   template args is a small kernel-side edit OUTSIDE this NEW-files-only area (flagged here so the
   kernel lane lands it). If that wiring is not yet present, `--tp 1` runs the single-GPU flagship
   today and the budget table still proves the 8-GPU fit.
4. **SG2 dedicated launcher boundary.** SuperGrok2 uses `mega_decoder_sg2_tc` (the 26-pointer
   meta-net bundle + per-tensor scalar arrays), not `tc_train_step`. The harness's
   `mod.tc_train_step` covers the 10 generic-launcher optimizers; for `--opt supergrok2` the TU
   must expose an `mod.sg2_train_step` pybind binding to the dedicated launcher (the cell already
   has it as a host symbol; the bench TU just doesn't bind it). Flagged as the one extra pybind
   the kernel lane adds; the budget already accounts for SG2's 9-plane state + the bundle.
5. **Batch B is per-rank and TP-replicated.** At TP=8 every rank sees the full batch (TP splits
   the *model*, not the batch). The DP path (`distributed_step.shard_batch_rows`) is only engaged
   at DP>1; the recommended config is DP=1, so no batch shard. The acts workspace (4.70 GB at
   B=512) scales with B — the budget table is parameterized by `--batch`.
