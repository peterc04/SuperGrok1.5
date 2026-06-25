"""grokking_optimizers/nvshmem_bringup_ext.py — the LIVE NVSHMEM bring-up driver.

This is the runtime companion of grokking_optimizers/host_bringup.py (the CPU plan).
host_bringup.py validates the plan offline and, with no Python NVSHMEM binding, can
only raise a scoped TPBootstrapBlocked. THIS module is the missing piece the GOAL
asks for: it JIT-builds the host NVSHMEM bring-up pybind
(csrc/fused/sm_90/nvshmem_bringup_pybind.cpp, linked host-only against
libnvshmem_host.so.3) and drives the real three-step bring-up:

    1. nvshmem_init via the UID bootstrap (nvshmemx_get_uniqueid +
       nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID)), the UID broadcast
       over torch.distributed (NCCL) — MPI/PMI are absent; the nvshmem_bootstrap_uid
       plugin IS present.
    2. nvshmem_team_split_strided for the TP team (PE range from the TPBootstrap plan).
    3. the COLLECTIVE nvshmem_malloc of the symmetric tp_sym_heap (sized to
       host_bringup.tp_heap_stride_floats(ctas_per_pe), matching the launcher's
       dec_tc_ensure_tp_sym_heap exactly).

CONTAINER NVLS BLOCKER (scoped, mitigated): NVSHMEM 3.7 + NCCL both default to NVLS
(NVLink-SHARP multicast) for the world team, which needs a multicast cuMemMap that
this RunPod container forbids (cuMemMap of the MC group fails with CUDA 'invalid
argument' — the SAME class of container capability restriction as the ncu
perf-counter blocker). The mitigation is to disable NVLS for BOTH stacks:
NVSHMEM_DISABLE_NVLS=1 and NCCL_NVLS_ENABLE=0. We set these defaults at import (the
operator can override) so the UID + P2P/NVLink path — which works — is taken.

The built module exposes (see the pybind TU): get_uniqueid / uniqueid_size /
init_with_uniqueid / team_split_strided / malloc_symmetric_heap / tp_allreduce_smoke
/ barrier_world / finalize / is_initialized / world_pe / world_npes / tp_local_pe /
tp_n_pes / sym_heap_ptr / sym_heap_floats.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from grokking_optimizers.host_bringup import (
    TPBootstrap, TPBootstrapBlocked, tp_heap_stride_floats, find_nvshmem_host_lib,
)

ROOT = Path(__file__).resolve().parents[1]
_PYBIND_SRC = ROOT / "csrc/fused/sm_90/nvshmem_bringup_pybind.cpp"

# Container NVLS blocker mitigation — set BEFORE NCCL/NVSHMEM init (idempotent;
# operator-overridable). See module docstring.
def set_nvls_disabled_defaults() -> None:
    os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


# Apply at import so a `torch.distributed.init_process_group` later in the same
# process (NCCL) also picks up NCCL_NVLS_ENABLE=0 — the 8-GPU NCCL collective hits
# the same NVLS cuMemMap blocker otherwise.
set_nvls_disabled_defaults()

_MODULE = None


def _nvshmem_home() -> str:
    return os.environ.get(
        "NVSHMEM_HOME", "/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem")


def _nvshmem_soname(libdir: str) -> Optional[str]:
    # The wheel ships only the versioned soname (libnvshmem_host.so.3) with no
    # unversioned .so symlink, so we link the exact filename via ld's -l:NAME form.
    for cand in ("libnvshmem_host.so.3", "libnvshmem_host.so"):
        if os.path.exists(os.path.join(libdir, cand)):
            return cand
    try:
        for fn in sorted(os.listdir(libdir)):
            if fn.startswith("libnvshmem_host.so"):
                return fn
    except OSError:
        pass
    return None


def build_bringup_module(*, verbose: bool = False):
    """JIT-build (or load cached) the host NVSHMEM bring-up pybind. Separate module
    name + build dir — never touches _ops or the committed launcher TUs. Raises
    TPBootstrapBlocked with a precise reason if the toolkit is not discoverable."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not _PYBIND_SRC.is_file():
        raise TPBootstrapBlocked(f"pybind source missing: {_PYBIND_SRC}")
    if find_nvshmem_host_lib() is None:
        raise TPBootstrapBlocked(
            "NVSHMEM host lib not found (libnvshmem_host.so.* under $NVSHMEM_HOME / "
            "the pip wheel) — cannot build the bring-up pybind.")
    home = _nvshmem_home()
    libdir = os.path.join(home, "lib")
    soname = _nvshmem_soname(libdir)
    if soname is None:
        raise TPBootstrapBlocked(f"libnvshmem_host.so* not found under {libdir}")
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")

    import torch  # noqa: PLC0415  (only needed for the live build)
    from torch.utils.cpp_extension import load  # noqa: PLC0415

    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    bd = os.path.join(os.environ.get("SG_TORCH_EXT_DIR", "/workspace/.torch_ext"),
                      "sg_nvshmem_bringup")
    os.makedirs(bd, exist_ok=True)
    _MODULE = load(
        name="sg_nvshmem_bringup",
        sources=[str(_PYBIND_SRC)],
        build_directory=bd,
        extra_include_paths=[str(ROOT), os.path.join(home, "include"),
                             os.path.join(cuda, "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[f"-L{libdir}", f"-l:{soname}",
                       f"-L{os.path.join(cuda, 'lib64')}", "-lcudart",
                       f"-Wl,-rpath,{libdir}"],
        verbose=verbose,
    )
    return _MODULE


def bringup_tp_team_live(plan: TPBootstrap, *, uid_broadcast, device_ordinal: int,
                         verbose_build: bool = False):
    """Drive the REAL NVSHMEM TP-team bring-up for THIS rank and return a dict of
    the resolved CommCtx fields (the launcher's host view of EDIT E) PLUS the live
    module so the caller can run a smoke/allocate further. The collective ops
    (init / team split / malloc) are entered on every rank in lock-step.

    ``uid_broadcast`` is ``(uid_bytes_or_None) -> uid_bytes`` (a tiny
    torch.distributed broadcast of the 128-byte UID blob): rank 0 supplies the
    minted blob, the rest supply None and receive it. ``device_ordinal`` is the
    local CUDA device (== LOCAL_RANK) NVSHMEM binds its symmetric heap to.

    Returns (mod, commctx_fields). Raises TPBootstrapBlocked on any bring-up failure
    (carrying the precise NVSHMEM/CUDA error)."""
    plan.validate()
    set_nvls_disabled_defaults()
    try:
        mod = build_bringup_module(verbose=verbose_build)
    except Exception as e:  # build/link failure is a precise, actionable blocker
        raise TPBootstrapBlocked(f"NVSHMEM bring-up pybind build failed: {e}") from e

    n_uid = mod.uniqueid_size()
    is_rank0 = (plan.global_rank == 0)
    uid_local = mod.get_uniqueid() if is_rank0 else None
    uid = uid_broadcast(uid_local)
    if uid is None or len(uid) != n_uid:
        raise TPBootstrapBlocked(
            f"UID broadcast returned {0 if uid is None else len(uid)} bytes, "
            f"expected {n_uid} (the torch.distributed broadcast of the 128-byte "
            f"nvshmemx_uniqueid_t blob is the bring-up's rendezvous).")
    try:
        mod.init_with_uniqueid(plan.global_rank, plan.world_size, uid, device_ordinal)
    except Exception as e:
        raise TPBootstrapBlocked(
            f"nvshmem_init (UID bootstrap) failed: {e}. If this is an NVLS "
            f"cuMemMap error, the container forbids NVLink-SHARP multicast — "
            f"NVSHMEM_DISABLE_NVLS=1 is set by default; check it survived.") from e

    pe_start, pe_stride, pe_size = plan.pe_range()
    try:
        team = mod.team_split_strided(pe_start, pe_stride, pe_size)
    except Exception as e:
        raise TPBootstrapBlocked(
            f"nvshmem_team_split_strided({pe_start},{pe_stride},{pe_size}) "
            f"failed: {e}") from e

    sym_floats = tp_heap_stride_floats(plan.ctas_per_pe)
    try:
        heap_ptr = mod.malloc_symmetric_heap(sym_floats)
    except Exception as e:
        raise TPBootstrapBlocked(
            f"collective nvshmem_malloc({sym_floats} floats) failed: {e}") from e

    # The launcher's CommCtx view (host_bringup.TPBootstrap.commctx_fields), with
    # the LIVE team handle + symmetric heap pointer filled in.
    fields = plan.commctx_fields(team)
    fields["tp_sym_heap_ptr"] = heap_ptr
    fields["tp_team_n_pes"] = mod.tp_n_pes()
    fields["tp_team_local_pe"] = mod.tp_local_pe()
    return mod, fields


__all__ = [
    "set_nvls_disabled_defaults",
    "build_bringup_module",
    "bringup_tp_team_live",
]
