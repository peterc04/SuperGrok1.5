"""tests/hw/nvshmem_smoke.py — the CROSS-GPU NVSHMEM bring-up SMOKE (capability track).

WHAT THIS PROVES (the genuine multi-GPU gate, not a 1-GPU loopback):
  Launch N processes (one per GPU) under torchrun, and on each:
    1. JIT-build the host NVSHMEM bring-up pybind (csrc/fused/sm_90/
       nvshmem_bringup_pybind.cpp), linked host-only against libnvshmem_host.so.3.
    2. Bootstrap NVSHMEM via the UID path: rank 0 calls get_uniqueid(); the 128-byte
       blob is broadcast over torch.distributed (NCCL); EVERY rank calls
       init_with_uniqueid(rank, world, uid) — the NVSHMEMX_INIT_WITH_UNIQUEID
       bootstrap the present nvshmem_bootstrap_uid.so.3 plugin implements (MPI/PMI
       are absent on this box).
    3. Carve THIS rank's TP team via nvshmem_team_split_strided with the
       (PE_start, PE_stride, PE_size) the host_bringup.TPBootstrap plan resolves
       (single-node pure TP ⇒ (0, 1, world) ⇒ the team == NVSHMEM_TEAM_WORLD).
    4. COLLECTIVE nvshmem_malloc of the symmetric heap (sized to
       host_bringup.tp_heap_stride_floats(ctas_per_pe), matching the launcher's
       dec_tc_ensure_tp_sym_heap exactly).
    5. A tiny TEAM-scoped all-reduce of (world_pe + 1) and ASSERT the cross-rank
       result == Σ_{pe in team}(pe+1) bit-exact on every rank. This is the genuine
       cross-GPU correctness assert (the symmetric heap addressing + a real
       collective over the carved team).

USAGE
    torchrun --nproc_per_node=2 tests/hw/nvshmem_smoke.py
    torchrun --nproc_per_node=8 tests/hw/nvshmem_smoke.py --ctas-per-pe 16 --count 4096

  (the script auto-sets NVSHMEM_DISABLE_NVLS=1 / NCCL_NVLS_ENABLE=0 — the RunPod
   container forbids NVLink-SHARP multicast cuMemMap; see the module-top note.)

It exits 0 on success (rank 0 prints PASS) and non-zero with a precise message on
any failure (build / init / team / malloc / reduce mismatch).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Container NVLS blocker mitigation, set at module load BEFORE any torch/NCCL/NVSHMEM
# import or init: the RunPod container forbids NVLink-SHARP multicast cuMemMap, which
# hits BOTH NCCL's NVLS path and NVSHMEM's (incl. NVSHMEM's internal NCCL for team
# setup). Disabling NVLS routes through the working P2P/NVLink path. (Operator can
# override by exporting these before launch.)
os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def build_pybind(verbose: bool = False):
    """JIT-build the host NVSHMEM bring-up pybind (delegates to the shared driver
    in grokking_optimizers.nvshmem_bringup_ext — ONE build path the harness shares).
    Also applies the NVLS-disabled container mitigation."""
    from grokking_optimizers.nvshmem_bringup_ext import build_bringup_module  # noqa: PLC0415
    return build_bringup_module(verbose=verbose)


def broadcast_uid(uid_bytes: bytes, *, src: int, device) -> bytes:
    """Broadcast the fixed-length UID blob over torch.distributed (NCCL). Every
    rank ends with rank-src's blob. The length is fixed (uniqueid_size); rank src
    supplies the bytes, the rest a zero buffer."""
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415

    n = len(uid_bytes) if uid_bytes is not None else None
    # length is a compile-time constant from the module; pass it in via the buffer
    # size the caller sized. We assume all ranks call with the same n (they do —
    # n == module.uniqueid_size()).
    assert n is not None
    buf = torch.zeros(n, dtype=torch.uint8, device=device)
    if dist.get_rank() == src:
        buf.copy_(torch.frombuffer(bytearray(uid_bytes), dtype=torch.uint8).to(device))
    dist.broadcast(buf, src=src)
    return bytes(buf.cpu().numpy().tobytes())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tp", type=int, default=0,
                    help="TP degree (0 => world_size: single-node pure TP)")
    ap.add_argument("--ctas-per-pe", type=int, default=1,
                    help="ctas_per_pe driving the symmetric heap stride")
    ap.add_argument("--count", "--n", dest="count", type=int, default=1024,
                    help="all-reduce element count for the smoke (<= heap/2)")
    ap.add_argument("--verbose-build", action="store_true")
    args = ap.parse_args(argv)

    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    # Import the ext driver FIRST: it sets NVSHMEM_DISABLE_NVLS / NCCL_NVLS_ENABLE=0
    # at import (the container's NVLS cuMemMap blocker hits BOTH NCCL and NVSHMEM, so
    # the knob must be set before dist.init_process_group / nvshmem_init).
    from grokking_optimizers import nvshmem_bringup_ext  # noqa: F401,PLC0415
    from grokking_optimizers.host_bringup import (  # noqa: PLC0415
        TPBootstrap, tp_heap_stride_floats, NVSHMEM_TEAM_WORLD,
    )

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world < 2:
        print("[smoke] WORLD_SIZE < 2 — this is a CROSS-GPU smoke; "
              "launch with torchrun --nproc_per_node=2 (or more).", flush=True)
        return 2

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    tp = args.tp or world
    tp_rank = rank % tp
    dp = world // tp
    dp_rank = rank // tp
    plan = TPBootstrap(global_rank=rank, world_size=world, tp=tp, tp_rank=tp_rank,
                       dp=dp, dp_rank=dp_rank, ctas_per_pe=args.ctas_per_pe)
    plan.validate()
    pe_start, pe_stride, pe_size = plan.pe_range()
    sym_floats = tp_heap_stride_floats(args.ctas_per_pe)

    if rank == 0:
        print(f"[smoke] world={world} tp={tp} dp={dp}; "
              f"per-rank PE range=({pe_start},{pe_stride},{pe_size}); "
              f"sym_floats={sym_floats:,} (>= 2*n={2*args.count})", flush=True)

    # 1. build the pybind (every rank builds; ninja serializes via the lock — rank 0
    #    typically wins the compile, the rest load the cached .so).
    mod = build_pybind(verbose=args.verbose_build and rank == 0)
    n_uid = mod.uniqueid_size()

    # 2. UID bootstrap: rank 0 mints the UID, broadcast it, everyone inits.
    uid = mod.get_uniqueid() if rank == 0 else (b"\x00" * n_uid)
    uid = broadcast_uid(uid, src=0, device=device)
    assert len(uid) == n_uid, f"UID length {len(uid)} != {n_uid}"
    dist.barrier()
    mod.init_with_uniqueid(rank, world, uid, local_rank)
    assert mod.is_initialized()
    assert mod.world_pe() == rank, f"nvshmem pe {mod.world_pe()} != rank {rank}"
    assert mod.world_npes() == world
    if rank == 0:
        print(f"[smoke] nvshmem_init OK: world_pe={mod.world_pe()} "
              f"world_npes={mod.world_npes()}", flush=True)

    # 3. team split (collective over WORLD — all ranks call with their own pe_range;
    #    for single-node pure TP every rank's pe_range is identical).
    team = mod.team_split_strided(pe_start, pe_stride, pe_size)
    assert mod.tp_n_pes() == tp, f"team n_pes {mod.tp_n_pes()} != tp {tp}"
    assert mod.tp_local_pe() == tp_rank, \
        f"team local_pe {mod.tp_local_pe()} != tp_rank {tp_rank}"
    if rank == 0:
        is_world = (team == NVSHMEM_TEAM_WORLD)
        print(f"[smoke] team_split OK: handle={team} "
              f"(== NVSHMEM_TEAM_WORLD? {is_world}); "
              f"local_pe={mod.tp_local_pe()} n_pes={mod.tp_n_pes()}", flush=True)

    # 4. collective symmetric malloc.
    heap = mod.malloc_symmetric_heap(sym_floats)
    assert heap != 0, "symmetric heap malloc returned null"
    assert mod.sym_heap_floats() >= sym_floats
    if rank == 0:
        print(f"[smoke] nvshmem_malloc OK: heap_ptr=0x{heap:x} "
              f"floats={mod.sym_heap_floats():,}", flush=True)

    # 5. the cross-rank all-reduce assert.
    out = mod.tp_allreduce_smoke(args.count)
    # Expected: Σ over team members of (global_pe + 1). For single-node pure TP the
    # team is the whole world ⇒ Σ_{p=0..world-1}(p+1) = world*(world+1)/2.
    team_pes = [pe_start + i * pe_stride for i in range(pe_size)]
    expected = float(sum(p + 1 for p in team_pes))
    got_min = float(out.min().item())
    got_max = float(out.max().item())
    ok = (got_min == expected and got_max == expected)
    # Cross-rank agreement: gather every rank's first element + ok flag.
    flag = torch.tensor([1.0 if ok else 0.0, got_min, got_max, expected],
                        device=device, dtype=torch.float64)
    gathered = [torch.zeros_like(flag) for _ in range(world)]
    dist.all_gather(gathered, flag)
    all_ok = all(g[0].item() == 1.0 for g in gathered)
    # every rank in the same team must have the IDENTICAL reduced value.
    vals = [g[1].item() for g in gathered]
    cross_consistent = (max(vals) == min(vals) == expected)

    if rank == 0:
        print(f"[smoke] all-reduce: expected={expected} "
              f"got=[{got_min},{got_max}] per-rank-ok={ok}", flush=True)
        for r, g in enumerate(gathered):
            print(f"    rank {r}: ok={int(g[0].item())} "
                  f"min={g[1].item()} max={g[2].item()} expected={g[3].item()}",
                  flush=True)

    mod.barrier_world()
    dist.barrier()
    mod.finalize()

    if not (all_ok and cross_consistent):
        if rank == 0:
            print("[smoke] FAIL: cross-rank all-reduce mismatch", flush=True)
        dist.destroy_process_group()
        return 1

    if rank == 0:
        print(f"[smoke] PASS: NVSHMEM UID bootstrap + team split + symmetric malloc "
              f"+ cross-rank all-reduce correct across {world} GPUs.", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
