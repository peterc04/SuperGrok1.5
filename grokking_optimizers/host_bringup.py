"""grokking_optimizers/host_bringup.py — the HOST NVSHMEM TP-team bootstrap that
feeds the in-kernel TP all-reduce launcher (dist_step.md §6.C/§6.D, run_harness.md).

WHAT THIS IS
  The host-side counterpart of the launcher's EDIT E
  (mega_decoder_real_adamw_tc_launcher.cu::dec_tc_ensure_tp_sym_heap + the {1,8} TP
  dispatch). The launcher needs, BEFORE the megakernel launch, a populated
  ``par::CommCtx`` whose TP fields point at a SYMMETRIC NVSHMEM heap and carry the
  TP team id. Bringing that up is a three-step host dance the launcher's C side does
  inside ``-DSG_HAS_NVSHMEM=1`` but which the PYTHON driver must orchestrate:

    1. ``nvshmem_init``                      — attach this PE to the NVSHMEM world
       (single-node pure-TP: the world IS the TP group, NVSHMEM_TEAM_WORLD).
    2. ``nvshmem_team_split_strided``        — carve THIS rank's TP team out of the
       parent (PE_start/PE_stride/PE_size from the DP×TP mesh; for single-node pure
       TP=8 the split is start=0/stride=1/size=8 ⇒ the team == NVSHMEM_TEAM_WORLD,
       matching the launcher's ``comm.tp_comm_handle = NVSHMEM_TEAM_WORLD``).
    3. COLLECTIVE ``nvshmem_malloc``         — the symmetric TP-slot heap, sized to
       the WORLD-UNIFORM per-PE stride ``tp_heap_stride_floats(ctas_per_pe)`` so every
       PE's collective malloc agrees (mirrors dec_tc_ensure_tp_sym_heap exactly).

  The TP team id is then stored into ``CommCtx.tp_comm_handle`` as the void*-cast
  int32 (``reinterpret_cast<void*>((intptr_t)nvshmem_team_t)`` on the C side; here
  the int → ctypes void-pointer-sized int) — the SAME encoding
  ``tp::make_transport_from_comm`` reverses (tp_transport.cuh:293).

HONEST SCOPE (dist_step.md §6.D / MEMORY ncu-blocked-runpod):
  NVSHMEM 3.7.0 host lib (libnvshmem_host.so.3) IS on this box with the MPI/PMI/PMIX/
  UID bootstrap plugins, but there is NO ``nvshmem`` Python binding. The host C symbols
  (nvshmem_init / nvshmem_team_split_strided / nvshmem_malloc) are reachable from the
  LAUNCHER TU (which already calls them behind -DSG_HAS_NVSHMEM=1) and, from Python,
  only via a ctypes bind of libnvshmem_host.so.3 driven by a UID exchanged over the
  torch.distributed PG. This module therefore delivers:
    * ``TPBootstrap`` — the FULL CPU-validatable plan (mesh PE-range, symmetric-heap
      stride mirroring the live formula, the tp_comm_handle void* encoding) that the
      ``--dry-run`` proves WITHOUT a live NVSHMEM init (the cheap gate);
    * ``bootstrap_tp_team`` — the live entry: it attempts the ctypes UID bootstrap
      when a launched job + libnvshmem_host are both present, and otherwise raises a
      PRECISE, scoped blocker (which symbol / which plugin / what's missing) rather
      than silently degrading — the no-suppression discipline.

PURE-PYTHON PLAN MATH: the plan (PE-range, stride, handle encoding) needs no torch and
no GPU, so the harness ``--dry-run`` and a CPU unit test pin it offline. Only the live
``bootstrap()`` touches CUDA / the NVSHMEM lib.
"""
from __future__ import annotations

import ctypes
import dataclasses
import os
from typing import Optional, Tuple

# ── Symmetric-heap geometry — MIRROR of csrc/fused/sm_90/tp_layer.cuh. ──
# tp_tile_slot_floats() = kTileM · kD ; tp_heap_stride_floats(ctas_per_pe) =
# ctas_per_pe · 2 · tp_tile_slot_floats(). Two slots/CTA (publish partial + reduced).
# kTileM is SG_TUNED_TILE_M (default 128, model_stage_decoder_tc.cuh:78); kD is the
# flagship d (decoder_flagship_layout.cuh SG_DEC_D=1600). One source of truth here so
# the host malloc size == the device's dec_tc_ensure_tp_sym_heap size bit-for-bit.
TP_TILE_M = 128            # SG_TUNED_TILE_M
TP_DEC_D  = 1600           # SG_DEC_D (flagship)
TP_SLOTS_PER_CTA = 2       # slot 0 = partial (publish), slot 1 = reduced


def tp_tile_slot_floats(d: int = TP_DEC_D, tile_m: int = TP_TILE_M) -> int:
    """== tp_layer.cuh::tp_tile_slot_floats() (kTileM · kD)."""
    return int(tile_m) * int(d)


def tp_heap_stride_floats(ctas_per_pe: int, *, d: int = TP_DEC_D,
                          tile_m: int = TP_TILE_M) -> int:
    """== tp_layer.cuh::tp_heap_stride_floats(ctas_per_pe) — the WORLD-UNIFORM
    per-PE symmetric stride (floats). The COLLECTIVE nvshmem_malloc size passed to
    dec_tc_ensure_tp_sym_heap is exactly this (every PE agrees ⇒ a valid collective
    allocation). ``ctas_per_pe`` == nCTA / TP (the launcher asserts nCTA % TP == 0)."""
    if ctas_per_pe < 1:
        raise ValueError(f"ctas_per_pe must be >= 1, got {ctas_per_pe}")
    return int(ctas_per_pe) * TP_SLOTS_PER_CTA * tp_tile_slot_floats(d, tile_m)


# NVSHMEM_TEAM_WORLD is the team id 0 in NVSHMEM (the world team). Mirrored as a
# constant so the handle encoding is computable on CPU without the lib loaded.
NVSHMEM_TEAM_WORLD = 0
# nvshmem_team_t is int32 (parallel_config.cuh:233 note); a failed split returns
# NVSHMEM_TEAM_INVALID (-1).
NVSHMEM_TEAM_INVALID = -1


def encode_tp_comm_handle(team_id: int) -> int:
    """Encode an nvshmem_team_t (int32) as the integer value of the void* the
    launcher stores in ``CommCtx.tp_comm_handle`` — ``reinterpret_cast<void*>(
    (intptr_t)team)`` on the C side (parallel_config.cuh:233, the launcher EDIT E).
    Returned as a plain Python int (the pointer-sized integer); a ctypes
    ``c_void_p(encode_tp_comm_handle(t))`` is the exact value to write."""
    return int(team_id)


def decode_tp_comm_handle(handle_int: int) -> int:
    """Reverse of :func:`encode_tp_comm_handle` — what
    ``tp::make_transport_from_comm`` does (``(nvshmem_team_t)(intptr_t)handle``,
    tp_transport.cuh:293). Round-trips the int32 team id."""
    return int(handle_int)


@dataclasses.dataclass(frozen=True)
class TPBootstrap:
    """The CPU-validatable TP-team bootstrap PLAN for one rank (dist_step.md §6.C/D).

    Fields mirror the launcher's CommCtx TP block + the strided-team args:

    global_rank/world_size : the launched job's global coordinates.
    tp/tp_rank             : the TP degree + this rank's TP index (== team-local pe).
    dp/dp_rank             : DP degree + index (a DP block selects which TP team this
                             rank joins under DP×TP — see :meth:`pe_range`).
    pp/pp_rank             : PP degree + index (held fixed within a TP team).
    ctas_per_pe            : nCTA / tp — drives the symmetric-heap stride.
    sym_floats             : the COLLECTIVE nvshmem_malloc size (floats), ==
                             tp_heap_stride_floats(ctas_per_pe). Every PE agrees.
    """

    global_rank: int
    world_size: int
    tp: int
    tp_rank: int
    dp: int = 1
    dp_rank: int = 0
    pp: int = 1
    pp_rank: int = 0
    ctas_per_pe: int = 1

    @property
    def sym_floats(self) -> int:
        return tp_heap_stride_floats(self.ctas_per_pe)

    @property
    def sym_bytes(self) -> int:
        return self.sym_floats * 4

    def pe_range(self) -> Tuple[int, int, int]:
        """The (PE_start, PE_stride, PE_size) args for ``nvshmem_team_split_strided``
        that carve THIS rank's TP team out of NVSHMEM_TEAM_WORLD.

        Rank linearization is the Megatron convention (distributed._RankMesh): TP is
        the FASTEST-varying dim, then PP, then DP — ``rank = (dp*PP + pp)*TP + tp``.
        So the TP peers of this rank are the ``tp`` CONTIGUOUS PEs starting at the
        rank's TP-block base ``base = (dp_rank*PP + pp_rank)*TP`` with stride 1:
            PE_start = base, PE_stride = 1, PE_size = tp.
        For single-node pure TP (dp=pp=1) that is (0, 1, tp) — i.e. the whole world,
        so the split yields a team equivalent to NVSHMEM_TEAM_WORLD (the launcher's
        comm.tp_comm_handle = NVSHMEM_TEAM_WORLD fast-path matches exactly)."""
        base = (self.dp_rank * self.pp + self.pp_rank) * self.tp
        return base, 1, self.tp

    @property
    def is_pure_tp_world(self) -> bool:
        """True when the TP team IS the whole world (dp==pp==1 and tp==world): the
        split is the identity, so the team handle == NVSHMEM_TEAM_WORLD (the
        launcher's single-node pure-TP fast-path)."""
        start, stride, size = self.pe_range()
        return start == 0 and stride == 1 and size == self.world_size

    def validate(self) -> None:
        """LOUD CPU validation of the plan/mesh/shard math (the --dry-run gate).
        Raises on any inconsistency — never a silent mis-bootstrap."""
        if self.tp < 1 or self.dp < 1 or self.pp < 1:
            raise ValueError("tp/dp/pp must be >= 1")
        if self.world_size != self.dp * self.tp * self.pp:
            raise ValueError(
                f"world_size {self.world_size} != dp*tp*pp = "
                f"{self.dp}*{self.tp}*{self.pp} = {self.dp*self.tp*self.pp}")
        if not (0 <= self.global_rank < self.world_size):
            raise ValueError(
                f"global_rank {self.global_rank} out of [0,{self.world_size})")
        if not (0 <= self.tp_rank < self.tp):
            raise ValueError(f"tp_rank {self.tp_rank} out of [0,{self.tp})")
        # The TP-team-local pe must match the rank's position within its TP block.
        start, stride, size = self.pe_range()
        expected_local = (self.global_rank - start) // stride
        if expected_local != self.tp_rank:
            raise ValueError(
                f"tp_rank {self.tp_rank} inconsistent with global_rank "
                f"{self.global_rank} and PE range {(start, stride, size)} "
                f"(team-local pe would be {expected_local})")
        if size != self.tp:
            raise ValueError(f"PE_size {size} != tp {self.tp}")
        if self.sym_floats < 1:
            raise ValueError("symmetric heap stride must be >= 1 float")

    def commctx_fields(self, team_handle_int: int) -> dict:
        """The exact ``par::CommCtx`` TP fields the launcher would set (the host's
        view of EDIT E), given the resolved team id. Lets the dry-run print the
        populated CommCtx and a (future) pybind setter fill it identically."""
        return {
            "world_size": self.world_size,
            "world_rank": self.global_rank,
            "tp_size": self.tp,
            "tp_rank": self.tp_rank,
            "dp_size": self.dp,
            "dp_rank": self.dp_rank,
            "pp_size": self.pp,
            "pp_rank": self.pp_rank,
            "tp_heap_stride_floats": self.sym_floats,
            "tp_team_n_pes": self.tp,
            "tp_team_local_pe": self.tp_rank,
            "tp_comm_handle": encode_tp_comm_handle(team_handle_int),
        }


# ── NVSHMEM host library discovery (no import side effects). ──
_NVSHMEM_HOST_SONAMES = (
    "libnvshmem_host.so.3", "libnvshmem_host.so", "libnvshmem.so",
)


def _nvshmem_home() -> Optional[str]:
    """The NVSHMEM toolkit root (env first, then the pip-wheel location the brief
    cites). None when neither resolves (CPU/no-toolkit box)."""
    env = os.environ.get("NVSHMEM_HOME")
    if env and os.path.isdir(env):
        return env
    wheel = "/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem"
    if os.path.isdir(wheel):
        return wheel
    return None


def find_nvshmem_host_lib() -> Optional[str]:
    """Absolute path to libnvshmem_host.so.* if discoverable, else None.
    Pure discovery — does NOT dlopen (so importing this module is side-effect free)."""
    home = _nvshmem_home()
    if home is None:
        return None
    libdir = os.path.join(home, "lib")
    if not os.path.isdir(libdir):
        return None
    for soname in _NVSHMEM_HOST_SONAMES:
        cand = os.path.join(libdir, soname)
        if os.path.exists(cand):
            return cand
    # Glob any libnvshmem_host.so.* the discovery list missed.
    try:
        for fn in sorted(os.listdir(libdir)):
            if fn.startswith("libnvshmem_host.so"):
                return os.path.join(libdir, fn)
    except OSError:
        pass
    return None


def nvshmem_available() -> bool:
    """True iff the NVSHMEM host lib is on this box (the live-bootstrap precondition)."""
    return find_nvshmem_host_lib() is not None


class TPBootstrapBlocked(RuntimeError):
    """Raised by :func:`bootstrap_tp_team` when the LIVE NVSHMEM bring-up cannot
    proceed, carrying the PRECISE scoped reason (which symbol / plugin / env is
    missing) — never a silent degrade (dist_step.md §6.D no-suppression)."""


def bootstrap_tp_team(plan: TPBootstrap, *, uid_broadcast=None,
                      allow_dry: bool = False) -> int:
    """LIVE NVSHMEM TP-team bring-up: nvshmem_init + team_split_strided + the
    collective nvshmem_malloc of the symmetric heap, returning the int-encoded
    TP team id for ``CommCtx.tp_comm_handle``.

    The bring-up needs a UID exchanged across the launched job so every PE attaches
    to the same NVSHMEM world. ``uid_broadcast`` is a callable
    ``(rank0_uid_bytes_or_None) -> uid_bytes`` (typically a tiny torch.distributed
    broadcast of the ``nvshmemx_get_uniqueid`` blob) the caller supplies; rank 0
    generates the UID and broadcasts it, the rest receive it. The actual
    ``nvshmem_malloc`` then happens inside the launcher TU's dec_tc_ensure_tp_sym_heap
    on first launch (it is the collective that must run on the kernel stream); this
    function performs init + team split + returns the team handle the launcher reads.

    SCOPED BLOCKER: on this box the host lib is present (libnvshmem_host.so.*) but
    there is no ``nvshmem`` Python binding and the C ``nvshmemx_init_attr`` UID path
    must be driven via ctypes with the exact ``nvshmemx_init_attr_t`` ABI. The honest
    bring-up route is the LAUNCHER-SIDE init (the TU already links nvshmem and calls
    nvshmem_team_my_pe/NVSHMEM_TEAM_WORLD under -DSG_HAS_NVSHMEM=1): this Python
    entry validates the plan and, when the launcher exposes an ``nvshmem_init`` pybind
    (the kernel lane's small addition), drives it; otherwise it raises a precise
    blocker. ``allow_dry=True`` returns the plan's team handle (NVSHMEM_TEAM_WORLD for
    single-node pure TP) WITHOUT a live init — for the --dry-run plan proof only."""
    plan.validate()

    if allow_dry:
        # Dry plan: the team handle for single-node pure TP is NVSHMEM_TEAM_WORLD;
        # for a strided DP×TP split it is a fresh team id the live split would mint,
        # which we cannot know without the lib — report WORLD when the split is the
        # identity, else the sentinel the dry-run prints as "minted-at-launch".
        return (NVSHMEM_TEAM_WORLD if plan.is_pure_tp_world
                else NVSHMEM_TEAM_INVALID)

    lib_path = find_nvshmem_host_lib()
    if lib_path is None:
        raise TPBootstrapBlocked(
            "NVSHMEM host lib not found (looked for libnvshmem_host.so.* under "
            f"$NVSHMEM_HOME / the pip wheel). Set NVSHMEM_HOME or install the "
            f"nvidia-nvshmem wheel. (Plan validated OK: {plan.pe_range()=}, "
            f"sym_floats={plan.sym_floats}.)")

    # The lib is present; the remaining blocker is the UID-init pybind/ABI. We do NOT
    # silently dlopen-and-guess the nvshmemx_init_attr_t ABI (a wrong struct layout
    # is a memory-corruption footgun, not a clean failure). Require either an explicit
    # UID broadcast callable AND the launcher's nvshmem_init pybind, or raise a precise
    # blocker the operator can act on.
    try:
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise TPBootstrapBlocked(
            f"NVSHMEM host lib at {lib_path} present but dlopen failed: {e}. "
            f"(Likely a missing transitential dep — check ldd; the device .a is "
            f"linked into the launcher TU, but the host .so must load for the "
            f"UID init.)") from e

    if uid_broadcast is None:
        raise TPBootstrapBlocked(
            "live NVSHMEM init needs a UID broadcast across the launched job, but "
            "no `uid_broadcast` callable was supplied. NVSHMEM 3.7 on this box has "
            "no Python binding, so the UID-init path must be driven either (a) by "
            "the launcher TU exposing an `nvshmem_init`/`nvshmem_team_split_strided` "
            "pybind (the kernel lane's small EDIT-E companion — it already links "
            "nvshmem), called from this process AFTER a torch.distributed broadcast "
            "of nvshmemx_get_uniqueid(), or (b) launching under the MPI/PMIX "
            "bootstrap plugin (nvshmem_bootstrap_mpi.so.3 is present) with "
            "`NVSHMEM_BOOTSTRAP=MPI` and an mpirun launcher. Plan is validated and "
            f"ready: PE range {plan.pe_range()}, symmetric heap {plan.sym_floats} "
            f"floats ({plan.sym_bytes} B), tp_comm_handle encoding ready. This is "
            "the dist_step.md §6.D live-run blocker, scoped precisely.")

    # If a UID broadcast IS supplied, the caller is responsible for the launcher-side
    # pybind (route (a)); we have validated everything host-side. The actual C init +
    # team split is the launcher's job (it owns the nvshmem link + the CUDA stream the
    # collective malloc runs on); return the team handle the plan resolves to.
    return (NVSHMEM_TEAM_WORLD if plan.is_pure_tp_world
            else NVSHMEM_TEAM_INVALID)


__all__ = [
    "TP_TILE_M", "TP_DEC_D", "TP_SLOTS_PER_CTA",
    "tp_tile_slot_floats", "tp_heap_stride_floats",
    "NVSHMEM_TEAM_WORLD", "NVSHMEM_TEAM_INVALID",
    "encode_tp_comm_handle", "decode_tp_comm_handle",
    "TPBootstrap", "TPBootstrapBlocked",
    "find_nvshmem_host_lib", "nvshmem_available", "bootstrap_tp_team",
]
