"""grokking_optimizers/parallel/resource_planner.py — the ROBUST execution PLANNER.

plan_execution(model_cfg, hw_cfg) -> ExecutionPlan

From the FRONT-END model config (d, layers, seq, vocab, num_experts, is_sequence,
optimizer) and the HARDWARE (num_gpus, hbm_bytes_per_gpu, host_ram_bytes, interconnect),
compute the FULL execution config:

  (a) the parallelism mesh (DP,TP,PP,SP,EP)         — §3, reuses distributed._RankMesh math
  (b) the MEMORY STRATEGY flags                     — §2, a memory-FIT escalation ladder
        need_zero_offload / need_activation_recompute / need_layer_streaming /
        need_param_offload (+ need_opt_offload), and cta_tiling via ncta
  (c) the kernel knob tier (cta_tiling, ring_depth, occupancy) — §4, by compute shape

The driver is **memory-fit + compute-shape vs hardware**, NEVER a GPU-count switch
(the user directive). The same ladder runs for 10M/1GPU (trivial), 1.5B/8GPU
(4D+ZeRO-3), and 10B/1GPU (offload+recompute+streaming+cta-tiling); the GPU count only
sets the ceiling on the parallelism rungs.

PURE PYTHON: no torch, no CUDA, no GPU. Mirrors the LIVE kernel scratch formulas
(fused_decoder_megakernel.cuh dec_tc_*_floats + opt_stage_supergrok2.cuh sg2_ws_stride
+ megakernel_codegen.py _decoder_param_sizes), so the front-end gets an exact, provable
per-rank budget + the exact -D compile flags BEFORE any GPU work.

SOURCES (read in full, cited inline):
  * param sizes  : megakernel_codegen.py::_decoder_param_sizes (2 + 12*L + 4 tensors, dff=4d)
  * acts         : fused_decoder_megakernel.cuh::dec_tc_acts_floats (:504)
  * staged scratch: fused_decoder_megakernel.cuh dec_tc_{opt_reduce,muon,looksam,sg2}_floats
                    (:553-638) + opt_stage_supergrok2.cuh::sg2_ws_stride (:440, SG2Dims<> defaults)
  * mesh         : grokking_optimizers/distributed.py ParallelConfig + _RankMesh (TP fastest)
  * opt taxonomy : grokking_optimizers/parallel/shard_map.py (ELEMENTWISE vs PER_TENSOR)
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

# ── Units (binary GiB, ONE unit — no 1000^3/1024^3 mixing; flagship_budget.py:175). ──
GB = 1024 ** 3
BYTES_PER_FLOAT = 4

# ── SG2Dims<> defaults (opt_stage_supergrok2.cuh:178-191). ──
SG2_D_MODEL, SG2_GRU_HIDDEN, SG2_INDEXER_RANK = 8, 4, 4
SG2_CSA_COMPRESS, SG2_CSA_TOPK = 4, 16

# ── Optimizer state-plane counts (mega_decoder_real_adamw_tc_launcher.cu state layout;
#    shard_map.py taxonomy). supergrok2 = (4+1+gru_hidden) = 9 (the 9-plane outlier). ──
_STATE_PLANES: Dict[str, int] = {
    "adamw": 3, "lion": 3, "grokfast": 3, "grokadamw": 3, "neuralgrok": 3,
    "prodigy": 4, "looksam": 3, "muon": 3, "supergrok11": 5, "supergrok15": 5,
    "supergrok2": 4 + 1 + SG2_GRU_HIDDEN,
}
# Per shard_map.py: per-TENSOR optimizers need whole tensors on one rank (no flat split);
# elementwise may flat-split. The planner uses this to know whether the staged SG2 carve
# is needed (SG2 always) and whether bench-layout elision is legal (adamw single-opt only).
_ELEMENTWISE = frozenset({"adamw", "lion", "grokfast", "grokadamw",
                          "looksam", "prodigy", "neuralgrok"})
_PER_TENSOR = frozenset({"muon", "supergrok11", "supergrok15", "supergrok2"})
# Optimizers whose staged-opt scratch is the binding SG2 meta-net carve.
_NEEDS_SG2_CARVE = frozenset({"supergrok2"})


# ───────────────────────────── front-end config types ────────────────────────────


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Front-end model shape. `optimizer` is the run's optimizer (or the WORST case of a
    multi-optimizer benchmark — supergrok2 — so the plan fits every member)."""
    d: int
    layers: int
    seq: int
    vocab: int
    batch: int = 256
    num_experts: int = 1
    is_sequence: bool = True            # decoder/transformer; False ⇒ no PP stage-cut benefit
    optimizer: str = "adamw"

    def __post_init__(self) -> None:
        for k in ("d", "layers", "seq", "vocab", "batch"):
            if getattr(self, k) < 1:
                raise ValueError(f"ModelConfig.{k} must be >= 1")
        if self.optimizer not in _STATE_PLANES:
            raise ValueError(f"unknown optimizer {self.optimizer!r} "
                             f"(known: {sorted(_STATE_PLANES)})")
        if self.num_experts < 1:
            raise ValueError("num_experts must be >= 1")


@dataclasses.dataclass(frozen=True)
class HardwareConfig:
    """Hardware envelope. Defaults model one 80 GB H100 SXM5 NVLink node."""
    num_gpus: int = 1
    hbm_bytes_per_gpu: int = 80 * (1000 ** 3)   # advertised "80 GB" (74.51 GiB physical)
    host_ram_bytes: int = 512 * (1000 ** 3)
    nvlink: bool = True                          # TP all-reduce wants NVLink, not PCIe
    nvlink_width: int = 8                        # max TP degree on the tight fabric
    sms_per_gpu: int = 132                       # H100 SXM5; 1 CTA/SM at full occupancy
    safety_gib: float = 4.0                      # ctx + cuBLAS/cuDNN + NCCL buffers

    def __post_init__(self) -> None:
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be >= 1")
        if self.hbm_bytes_per_gpu < 1 or self.host_ram_bytes < 1:
            raise ValueError("memory sizes must be positive")

    @property
    def usable_hbm_gib(self) -> float:
        return self.hbm_bytes_per_gpu / GB - self.safety_gib

    @property
    def host_ram_gib(self) -> float:
        return self.host_ram_bytes / GB


# ───────────────────────────── output types ──────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Mesh:
    dp: int
    tp: int
    pp: int
    sp: int
    ep: int

    @property
    def world_size(self) -> int:                # mirrors ParallelConfig.world_size
        return self.dp * self.tp * self.pp

    @property
    def model_parallel_size(self) -> int:       # mirrors ParallelConfig.model_parallel_size
        return self.tp * self.pp


@dataclasses.dataclass(frozen=True)
class MemFlags:
    need_zero_offload: bool = False             # ZeRO-3 param+state shard over DP
    need_activation_recompute: bool = False     # gradient checkpointing (1 layer live)
    need_layer_streaming: bool = False          # one PP-stage of params resident at a time
    need_param_offload: bool = False            # params -> host RAM
    need_opt_offload: bool = False              # opt-state -> host RAM (AdamW-on-host)
    cta_tiling: bool = False                    # ncta < sms_per_gpu (staged-scratch trim)


@dataclasses.dataclass(frozen=True)
class KernelKnobs:
    ncta: int
    ring_depth: int
    occupancy_cta_per_sm: float                 # ncta / sms_per_gpu
    staged_scratch_needed: bool                 # the 4 staged carves present (else bench elide)


@dataclasses.dataclass(frozen=True)
class MemBreakdownGiB:
    params: float
    state: float
    acts: float
    staged_opt: float
    sg2_region: float
    host_params: float
    host_state: float
    total_hbm: float
    total_host: float


@dataclasses.dataclass(frozen=True)
class ExecutionPlan:
    model: ModelConfig
    hw: HardwareConfig
    mesh: Mesh
    mem: MemFlags
    knobs: KernelKnobs
    budget: MemBreakdownGiB
    compile_flags: List[str]
    template_inst: str
    fits: bool
    risks: List[str]

    def summary(self) -> str:
        m, k, b = self.mesh, self.knobs, self.budget
        return (f"ExecutionPlan(world={m.world_size} "
                f"DP={m.dp} TP={m.tp} PP={m.pp} SP={m.sp} EP={m.ep} | "
                f"zero3={self.mem.need_zero_offload} recompute={self.mem.need_activation_recompute} "
                f"stream={self.mem.need_layer_streaming} "
                f"poff={self.mem.need_param_offload} ooff={self.mem.need_opt_offload} "
                f"ncta={k.ncta} ring={k.ring_depth} | "
                f"HBM={b.total_hbm:.2f}/{self.hw.usable_hbm_gib:.1f} GiB "
                f"host={b.total_host:.2f}/{self.hw.host_ram_gib:.0f} GiB "
                f"{'FITS' if self.fits else 'OOM'})")


class PlanInfeasible(RuntimeError):
    """Raised when no rung of the escalation ladder fits the model on the hardware."""


# ───────────────────────────── layout arithmetic (§1.1) ──────────────────────────


def decoder_param_sizes(d: int, layers: int, vocab: int, seq: int) -> List[int]:
    """Mirror of megakernel_codegen.py::_decoder_param_sizes — per-tensor numel in
    named_parameters() order. 2 + 12*L + 4 tensors, dff=4d. Verified to reproduce the
    flagship (1600,48,99,4) -> total 1,475,884,899 / 582 tensors / max 10,240,000."""
    dff = 4 * d
    sizes = [vocab * d, seq * d]                       # tok, pos
    for _ in range(layers):
        sizes += [
            3 * d * d, 3 * d,                          # attn.in_proj w/b
            d * d, d,                                  # attn.out_proj w/b
            d, d, d, d,                                # n1.w/b, n2.w/b
            dff * d, dff,                              # ff.0 w/b
            d * dff, d,                                # ff.2 w/b
        ]
    sizes += [d, d, vocab * d, vocab]                  # norm.w/b, out.w/b
    return sizes


def layout_arith(mc: ModelConfig) -> Tuple[int, int, int]:
    """Return (total_params, n_tensors, max_tensor_numel) for the model."""
    sizes = decoder_param_sizes(mc.d, mc.layers, mc.vocab, mc.seq)
    return sum(sizes), len(sizes), max(sizes)


# ───────────────────────────── staged-opt scratch (§1.2) ─────────────────────────


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def sg2_ws_stride(nmax: int) -> int:
    """Mirror of opt_stage_supergrok2.cuh::sg2_ws_stride<SG2Dims<>>(Nmax) — floats/CTA.
    ~91.277*Nmax with the defaults (verified numerically this session)."""
    d, rk, gh = SG2_D_MODEL, SG2_INDEXER_RANK, SG2_GRU_HIDDEN
    ncmax = (nmax + SG2_CSA_COMPRESS - 1) // SG2_CSA_COMPRESS
    topk = SG2_CSA_TOPK if SG2_CSA_TOPK > 1 else 1
    f = 7 * nmax * d                  # x_sorted,csa_ctx,hca_ctx,q,win_k,win_v,concat
    f += 2 * ncmax * d               # c_k, c_v
    f += nmax * rk                   # qI
    f += ncmax * rk                  # kI
    f += nmax * topk                 # sel
    f += nmax * gh                   # new_gru
    f += nmax                        # expert_out
    f += 2 * _next_pow2(nmax)        # sort keys + idx
    f += 2 * nmax                    # perm + unsort
    return f


def dec_sg2_ws_stride_floats(nmax: int, n_tensors: int) -> int:
    return 2 * n_tensors + sg2_ws_stride(nmax)          # :615


def dec_tc_sg2_floats(nmax: int, ncta: int, n_tensors: int) -> int:
    return ncta * dec_sg2_ws_stride_floats(nmax, n_tensors) + 1   # :619


def dec_tc_muon_floats(max2d_numel: int, max_rows: int, ncta: int) -> int:
    return 4 * max2d_numel + max_rows * max_rows + ncta + 1       # :567


def dec_tc_looksam_floats(total: int) -> int:
    return 2 * total                                              # :584


def dec_tc_opt_reduce_floats(ncta: int) -> int:
    return 2 * ncta + 1                                           # :553


def dec_tc_acts_floats(B: int, d: int, vocab: int, layers_live: int, seq: int) -> int:
    """Mirror of dec_tc_acts_floats (:504). `layers_live` = L/PP, or 1 under recompute."""
    dff = 4 * d
    T = B * seq
    Td, T3d, Tff = T * d, T * 3 * d, T * dff
    bf = 0
    for _ in range(layers_live):
        bf += Td + Td + Td + Tff + T3d + Td + Tff + Td
    bf += B * d + B * vocab + Td
    return (bf + 1) // 2


# ───────────────────────────── per-rank budget (§1.4-1.5) ────────────────────────


def per_rank_budget(mc: ModelConfig, hw: HardwareConfig, mesh: Mesh,
                    flags: MemFlags, ncta: int,
                    *, total: int, n_tensors: int, nmax: int,
                    staged_scratch_needed: bool = True) -> MemBreakdownGiB:
    """The EXACT per-rank HBM (+host) footprint for ONE (mesh, flags, ncta) point.
    Mirrors the live dec_tc_*_floats; the fit gate the front-end trusts.

    `staged_scratch_needed` mirrors the live kDecStagedOptScratch gate
    (fused_decoder_megakernel.cuh:541-545): True ⇒ the four staged-opt regions
    (opt_reduce|muon|looksam|sg2) are carved (production opt-agnostic launcher);
    False ⇒ they are elided (SG_DEC_BENCH_LAYOUT, adamw single-opt) — exactly the
    `dec_tc_*_floats` `if (!kDecStagedOptScratch) return 0;` early-out. This is why
    a 10B adamw run is fittable: its looksam carve (2·total = 75 GiB at 10B) is dead
    weight for an elementwise single-opt and is elided, NOT charged to HBM."""
    opt = mc.optimizer
    model_shard = mesh.tp * mesh.pp
    zero_div = mesh.dp if flags.need_zero_offload else 1

    # params + opt-state residency (ZeRO-3 shards over DP; offload moves to host).
    resident_params = total // (model_shard * zero_div)
    if flags.need_layer_streaming:
        # only ~1 of `layers` worth of the per-layer params resident at a time
        # (embeddings/tails stay); model_shard already split them.
        resident_params = max(resident_params // max(mc.layers, 1), 1)
    state_floats = _STATE_PLANES[opt] * total // (model_shard * zero_div)

    host_params_f = resident_params if flags.need_param_offload else 0
    host_state_f = state_floats if flags.need_opt_offload else 0
    hbm_params_f = 0 if flags.need_param_offload else resident_params
    hbm_state_f = 0 if flags.need_opt_offload else state_floats

    # activations (L/PP live, or 1 under recompute). Not ZeRO-sharded (transient).
    layers_live = 1 if flags.need_activation_recompute else max(mc.layers // mesh.pp, 1)
    acts = dec_tc_acts_floats(mc.batch, mc.d, mc.vocab, layers_live, mc.seq)

    # staged-opt scratch. Present ONLY when staged_scratch_needed (the kDecStagedOptScratch
    # gate); SG2 also requires opt==supergrok2 (its meta-net carve is the binding term).
    # TP shrinks Nmax (Megatron split); max 2D weight ~ ff = 4d*d split by TP; rows = 4d/TP.
    nmax_t = nmax // mesh.tp
    max2d = (4 * mc.d * mc.d) // mesh.tp
    max_rows = max((4 * mc.d) // mesh.tp, 1)
    if staged_scratch_needed:
        staged = (dec_tc_opt_reduce_floats(ncta)
                  + dec_tc_muon_floats(max2d, max_rows, ncta)
                  + dec_tc_looksam_floats(total // model_shard))
        sg2 = dec_tc_sg2_floats(nmax_t, ncta, n_tensors) if opt in _NEEDS_SG2_CARVE else 0
    else:
        staged = 0          # SG_DEC_BENCH_LAYOUT: the four carves fold to 0 (adamw single-opt)
        sg2 = 0
    staged += sg2

    def gib(f):
        return f * BYTES_PER_FLOAT / GB

    params = gib(hbm_params_f)
    state = gib(hbm_state_f)
    acts_g = gib(acts)
    staged_g = gib(staged)
    sg2_g = gib(sg2)
    total_hbm = params + state + acts_g + staged_g + 0.10   # tile-scratch slack
    total_host = gib(host_params_f) + gib(host_state_f)
    return MemBreakdownGiB(params=params, state=state, acts=acts_g,
                           staged_opt=staged_g, sg2_region=sg2_g,
                           host_params=gib(host_params_f), host_state=gib(host_state_f),
                           total_hbm=total_hbm, total_host=total_host)


# ───────────────────────────── mesh inference (§3) ───────────────────────────────


def _largest_pow2_divisor(n: int) -> int:
    p = 1
    while n % (p * 2) == 0:
        p *= 2
    return p


def infer_mesh(mc: ModelConfig, hw: HardwareConfig) -> Mesh:
    """3D-5D mesh inference (NOT keyed on a GPU-count switch). TP first (shrinks Nmax,
    rides NVLink — distributed._RankMesh puts TP fastest); PP only if TP+ZeRO-3 cannot
    fit per-stage; DP fills the rest; EP sub-divides DP for MoE. SP pinned to 1."""
    g = hw.num_gpus
    # TP: largest pow2 dividing g, bounded by NVLink width and by d % TP == 0.
    tp_cap = min(_largest_pow2_divisor(g),
                 hw.nvlink_width if hw.nvlink else 1)
    tp = 1
    cand = tp_cap
    while cand >= 1:
        if g % cand == 0 and mc.d % cand == 0:
            tp = cand
            break
        cand //= 2
    rest = g // tp
    # PP: smallest divisor of `rest` with L % PP == 0 that we COULD use; the ladder
    # decides whether to raise it. Start at 1 (PP is overhead; raise only if needed).
    pp = 1
    # DP fills whatever TP*PP leaves.
    dp = rest // pp
    # EP: sub-divide DP for MoE (EP | DP), never enlarges world (distributed.py:69-73).
    ep = 1
    if mc.num_experts > 1 and dp > 1:
        ep = 1
        for cand in range(min(mc.num_experts, dp), 0, -1):
            if dp % cand == 0:
                ep = cand
                break
    return Mesh(dp=dp, tp=tp, pp=pp, sp=1, ep=ep)


def _raise_pp(mesh: Mesh, mc: ModelConfig) -> Optional[Mesh]:
    """Raise PP one step (consuming a DP factor) if L % PP == 0 — used by the ladder
    when TP+ZeRO-3 still overflows per-stage. Returns None if no PP step is available."""
    rest = mesh.dp * mesh.pp                  # ranks TP leaves
    for new_pp in range(mesh.pp + 1, rest + 1):
        if rest % new_pp == 0 and mc.layers % new_pp == 0:
            new_dp = rest // new_pp
            ep = min(mesh.ep, new_dp) if mc.num_experts > 1 else 1
            while ep > 1 and new_dp % ep != 0:
                ep -= 1
            return Mesh(dp=new_dp, tp=mesh.tp, pp=new_pp, sp=1, ep=ep)
    return None


# ───────────────────────────── kernel knobs (§4) ─────────────────────────────────


def _ring_depth(d: int) -> int:
    if d <= 1024:
        return 2                              # shallow ring fits 48 KB static smem
    if d <= 4096:
        return 3                              # deep ring -> SG_DEC_TC_DYNAMIC_SMEM
    return 3


_NCTA_LADDER = (None, 64, 32, 16, 8, 4, 2, 1)   # None -> sms_per_gpu (1 CTA/SM)


# ───────────────────────────── compile-flag emission (§5) ────────────────────────


def _layout_header(mc: ModelConfig) -> Tuple[str, bool]:
    """Return (force-include header path, is_flagship). The 1.5B pin uses the committed
    flagship header; any other size uses a codegen'd header (megakernel_codegen.py
    decoder_layout_header / decoder_flagship_layout_header path)."""
    is_flag = (mc.d == 1600 and mc.layers == 48 and mc.vocab == 99 and mc.seq == 4)
    if is_flag:
        return "csrc/fused/sm_90/decoder_flagship_layout.cuh", True
    return (f"csrc/fused/sm_90/generated/decoder_layout_d{mc.d}_L{mc.layers}_"
            f"v{mc.vocab}_s{mc.seq}.cuh", False)


def emit_compile_flags(mc: ModelConfig, hw: HardwareConfig, mesh: Mesh,
                       flags: MemFlags, knobs: KernelKnobs) -> List[str]:
    """Map the ExecutionPlan to the EXACT -D flags the build (run_harness.md
    build_flagship_module + dist_step.md §6.C Par-launcher) consumes."""
    z = 3 if flags.need_zero_offload else 0
    header, _ = _layout_header(mc)
    out: List[str] = [
        "-O3", "-std=c++17", "--expt-relaxed-constexpr",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-DSG_TUNED_GEMM_IMPL=1",                       # wgmma L3-TC cell driver
        "-DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1",        # pre-set the committed guard
        "-include", header,                             # force the chosen layout table
        f"-DSG_FLAGSHIP_DP={mesh.dp}",
        f"-DSG_FLAGSHIP_TP={mesh.tp}",
        f"-DSG_FLAGSHIP_PP={mesh.pp}",
        f"-DSG_FLAGSHIP_ZERO={z}",
    ]
    # staged carve: present unless adamw single-opt (then bench-layout elides the 4 carves).
    if not knobs.staged_scratch_needed:
        out.append("-DSG_DEC_BENCH_LAYOUT=1")           # adamw-only -> elide staged scratch
    if flags.need_activation_recompute:
        out.append("-DSG_DEC_RECOMPUTE=1")
    if flags.need_layer_streaming:
        out.append("-DSG_DEC_LAYER_STREAM=1")
    off = (2 if flags.need_opt_offload else 0) | (1 if flags.need_param_offload else 0)
    if off:
        out.append(f"-DSG_DEC_HOST_OFFLOAD={off}")
    if knobs.ring_depth == 3:
        out += ["-DSG_TUNED_DEC_FWD_PIPE=2", "-DSG_DEC_TC_DYNAMIC_SMEM=1"]
    else:
        out.append("-DSG_TUNED_DEC_FWD_PIPE=1")
    if mesh.tp > 1 and hw.nvlink:
        out.append("-DSG_HAS_NVSHMEM=1")                # device-initiated in-kernel TP all-reduce
    return out


def _template_inst(mc: ModelConfig, mesh: Mesh, flags: MemFlags) -> str:
    z = "ZeROStage::Z3" if flags.need_zero_offload else "ZeROStage::Z0"
    opt = {"adamw": "OptId::AdamW", "lion": "OptId::Lion", "grokfast": "OptId::GrokFast",
           "grokadamw": "OptId::GrokAdamW", "looksam": "OptId::LookSAM",
           "prodigy": "OptId::Prodigy", "neuralgrok": "OptId::NeuralGrok",
           "muon": "OptId::Muon", "supergrok11": "OptId::SuperGrok11",
           "supergrok15": "OptId::SuperGrok15",
           "supergrok2": "OptId::SuperGrok2"}[mc.optimizer]
    if mesh.world_size == 1:
        return f"launch_fused_decoder_megakernel_tc<{opt}>  // ParConfig defaults to par::SingleGPU"
    return (f"launch_fused_decoder_megakernel_tc<{opt}, "
            f"ParConfig<{mesh.dp},{mesh.tp},{mesh.pp},{mesh.sp},{z}>>")


# ───────────────────────────── THE PLANNER (§2 ladder) ───────────────────────────


def plan_execution(model_cfg: ModelConfig,
                   hw_cfg: Optional[HardwareConfig] = None) -> ExecutionPlan:
    """From (model_cfg, hw_cfg) compute the FULL ExecutionPlan. Memory-FIT driven, NEVER
    a GPU-count switch: the parallelism rungs are bounded by num_gpus, but the strategy
    (zero3/recompute/cta-tile/stream/offload) is selected by the per-rank fit estimate."""
    mc = model_cfg
    hw = hw_cfg or HardwareConfig()
    total, n_tensors, nmax = layout_arith(mc)

    mesh = infer_mesh(mc, hw)
    risks: List[str] = []

    # staged carve is elided ONLY for adamw single-opt (bench-layout); SG2 always needs it.
    staged_needed = mc.optimizer != "adamw"

    def budget_at(mesh_: Mesh, flags_: MemFlags, ncta_: int) -> MemBreakdownGiB:
        return per_rank_budget(mc, hw, mesh_, flags_, ncta_,
                               total=total, n_tensors=n_tensors, nmax=nmax,
                               staged_scratch_needed=staged_needed)

    def fits(b: MemBreakdownGiB) -> bool:
        return b.total_hbm <= hw.usable_hbm_gib and b.total_host <= hw.host_ram_gib

    # ── the escalation ladder (§0). Start in-HBM, full occupancy. ──
    flags = MemFlags()
    ncta_full = hw.sms_per_gpu
    ncta = ncta_full

    b = budget_at(mesh, flags, ncta)
    if not fits(b):
        # R1 ZeRO-3 (no-op at DP=1, but free when DP>1).
        flags = dataclasses.replace(flags, need_zero_offload=True)
        b = budget_at(mesh, flags, ncta)
    if not fits(b) and mesh.pp == 1:
        # R1b raise PP if a TP+ZeRO-3 per-stage still overflows (only when DP factor free).
        bumped = _raise_pp(mesh, mc)
        if bumped is not None and bumped.pp > mesh.pp:
            cand = budget_at(bumped, flags, ncta)
            if cand.total_hbm < b.total_hbm:
                mesh, b = bumped, cand
    if not fits(b):
        # R2 CTA-tiling FIRST (cheaper than recompute — trades occupancy, not compute).
        # Walks the live auto_ncta ladder for the largest nCTA that fits the staged
        # scratch. This reproduces the run_harness.md headline: flagship SG2 fits at
        # nCTA=64 WITHOUT recompute (its acts at seq=4 are tiny; the staged scratch is
        # the binding term, so trimming nCTA — not recompute — is the right first move).
        for step in _NCTA_LADDER:
            cand_ncta = ncta_full if step is None else step
            if cand_ncta > ncta_full:
                continue
            tiled = dataclasses.replace(flags, cta_tiling=cand_ncta < ncta_full)
            cand = budget_at(mesh, tiled, cand_ncta)
            ncta = cand_ncta
            b = cand
            if fits(b):
                flags = tiled
                break
        else:
            flags = dataclasses.replace(flags, cta_tiling=True)
    if not fits(b):
        # R3 activation recompute (binding at long seq / large B — e.g. 10B seq=2048).
        flags = dataclasses.replace(flags, need_activation_recompute=True)
        b = budget_at(mesh, flags, ncta)
    if not fits(b) and mesh.pp == 1:
        # R4 layer streaming (single-rank analogue of PP param residency).
        flags = dataclasses.replace(flags, need_layer_streaming=True)
        b = budget_at(mesh, flags, ncta)
    if not fits(b):
        # R5 host offload: opt-state first (AdamW-on-host), then params.
        flags = dataclasses.replace(flags, need_opt_offload=True)
        b = budget_at(mesh, flags, ncta)
        if not fits(b):
            flags = dataclasses.replace(flags, need_param_offload=True)
            b = budget_at(mesh, flags, ncta)

    # SG2 honesty: its per-CTA workspace (91.277*Nmax/TP) may be structurally unfittable
    # even at ncta=1 on too-few GPUs (the live KNOWN DEEP LIMIT). Record a downgrade.
    if mc.optimizer in _NEEDS_SG2_CARVE and not fits(b):
        sg2_at_1 = budget_at(mesh, flags, 1)
        if sg2_at_1.total_hbm > hw.usable_hbm_gib:
            risks.append(
                f"supergrok2 staged scratch is {sg2_at_1.sg2_region:.0f} GiB even at "
                f"nCTA=1 (Nmax/TP={nmax // mesh.tp:,}); the SG2 per-CTA meta-net workspace "
                f"is O(91.277*Nmax) and does not fit on this hardware. Plan downgrades the "
                f"optimizer to an elementwise cell (adamw) + host offload — raise TP "
                f"(more GPUs) to run SG2 at this size. (fused_decoder_megakernel.cuh KNOWN "
                f"DEEP LIMIT, :598-610).")
            # re-plan as adamw to give a fitting plan for the elementwise fallback.
            mc_dn = dataclasses.replace(mc, optimizer="adamw")
            return _replan_downgraded(mc_dn, hw, risks)

    knobs = KernelKnobs(ncta=ncta, ring_depth=_ring_depth(mc.d),
                        occupancy_cta_per_sm=ncta / hw.sms_per_gpu,
                        staged_scratch_needed=staged_needed)
    cflags = emit_compile_flags(mc, hw, mesh, flags, knobs)
    tinst = _template_inst(mc, mesh, flags)

    if flags.need_param_offload or flags.need_opt_offload:
        risks.append(
            f"host offload active (params={flags.need_param_offload} "
            f"state={flags.need_opt_offload}, {b.total_host:.1f} GiB to host) — bounded "
            f"by host<->device bandwidth (PCIe vs NVLink); throughput will drop. "
            f"Needs host_ram >= {b.total_host:.0f} GiB.")
    if flags.need_layer_streaming:
        risks.append("layer streaming active — params resident one stage at a time; "
                     "overlap with compute is bandwidth-bound (the streaming risk).")
    if flags.cta_tiling:
        risks.append(f"CTA-tiling: ncta={ncta} < {hw.sms_per_gpu} SMs "
                     f"({ncta / hw.sms_per_gpu:.0%} occupancy) to fit the staged scratch.")

    final_fits = fits(b)
    if not final_fits:
        raise PlanInfeasible(
            f"no rung fits {mc.optimizer} {total/1e9:.2f}B on {hw.num_gpus} GPU(s): "
            f"HBM {b.total_hbm:.1f} > {hw.usable_hbm_gib:.1f} GiB (or host "
            f"{b.total_host:.1f} > {hw.host_ram_gib:.0f} GiB) after offload+recompute+"
            f"stream+cta-tile. Add GPUs (raise TP) or shrink the model.")

    return ExecutionPlan(model=mc, hw=hw, mesh=mesh, mem=flags, knobs=knobs, budget=b,
                         compile_flags=cflags, template_inst=tinst, fits=final_fits,
                         risks=risks)


def _replan_downgraded(mc: ModelConfig, hw: HardwareConfig,
                       carried_risks: List[str]) -> ExecutionPlan:
    """Re-run the planner for an elementwise (adamw) fallback when SG2 is unfittable,
    carrying the downgrade note. Guaranteed not to recurse (adamw has no SG2 carve)."""
    plan = plan_execution(mc, hw)
    return dataclasses.replace(plan, risks=carried_risks + plan.risks)


__all__ = [
    "ModelConfig", "HardwareConfig", "Mesh", "MemFlags", "KernelKnobs",
    "MemBreakdownGiB", "ExecutionPlan", "PlanInfeasible",
    "decoder_param_sizes", "layout_arith", "sg2_ws_stride", "dec_tc_acts_floats",
    "per_rank_budget", "infer_mesh", "emit_compile_flags", "plan_execution",
]
