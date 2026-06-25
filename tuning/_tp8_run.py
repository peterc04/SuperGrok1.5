"""tuning/_tp8_run.py — NON-COMMITTED scratch runner: ONE flagship 1.5B decoder
across all 8 H100s via TP8 + ZeRO-3, driving the IN-KERNEL device-NVSHMEM all-reduce.

Per rank (torchrun --nproc_per_node=8):
  1. import the bringup ext (sets NVSHMEM_DISABLE_NVLS=1 / NCCL_NVLS_ENABLE=0),
     init torch.distributed (NCCL).
  2. NVSHMEM UID bootstrap + team_split(0,1,8) + a collective sym malloc via the
     validated sg_nvshmem_bringup pybind → brings up the process-global NVSHMEM world
     the megakernel's in-kernel nvshmem_* calls share.
  3. JIT-build the scratch TP8 pybind (_tp8_scratch_pybind.cu) with the flagship
     layout force-included + -DSG_HAS_NVSHMEM=1 -rdc=true + device-linked nvshmem.
  4. TP-shard the 1.476B weights (partition_tensor_parallel) → report per-rank
     resident params; ZeRO-3 shards the AdamW state.
  5. run N steps of tc_train_step_tp8 (the ParTP8 in-kernel all-reduce path) on a
     fixed TP-replicated real batch; verify cross-rank loss AGREES + loss DESCENDS.

USAGE: torchrun --nproc_per_node=8 tuning/_tp8_run.py --steps 30 --ncta 64
"""
from __future__ import annotations
import argparse, glob, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SCRATCH_TU = str(ROOT / "tuning/_tp8_scratch_pybind.cu")
_FLAGSHIP_LAYOUT = "csrc/fused/sm_90/decoder_flagship_layout.cuh"


def glob_headers():
    """Every sm_90 .cuh/.cu the TP8 scratch TU transitively includes — so the
    rebuild freshness check fires when a megakernel/decoder HEADER is edited (the
    bug fixes touch headers, not the pybind .cu)."""
    return glob.glob(str(ROOT / "csrc/fused/sm_90/*.cuh")) + \
           glob.glob(str(ROOT / "csrc/fused/sm_90/*.cu")) + \
           glob.glob(str(ROOT / "csrc/fused/*.cuh"))


def build_tp8_module(verbose=False, only_rank0_build=True, rank=0):
    """Build the scratch TP8 pybind via the manual driver (_tp8_build.sh) which adds
    the device-link (-dlink) step torch's JIT load() omits for -rdc=true objects,
    then import the resulting .so. torch's load() cannot device-link an RDC TU
    against libnvshmem_device.a (it leaves __cudaRegisterLinkedBinary unresolved)."""
    import torch  # noqa  (ensure libtorch is in the process before dlopen of the .so)
    import importlib.util  # noqa
    import subprocess  # noqa
    home = os.environ.get("NVSHMEM_HOME",
                          "/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem")
    # WORKTREE build dir + ROOT (matches tuning/_tp8_build.sh's SG_TP8_BD/SG_TP8_ROOT
    # defaults) so the EDITED megakernel headers compile into a SEPARATE .so that
    # does NOT clobber the main repo's cached sg_tp8_scratch.so.
    bd = os.environ.get(
        "SG_TP8_BD",
        os.path.join(os.environ.get("SG_TORCH_EXT_DIR", "/workspace/.torch_ext"),
                     "sg_tp8_scratch_wt"))
    so = os.path.join(bd, "sg_tp8_scratch.so")
    env = dict(os.environ); env["NVSHMEM_HOME"] = home
    env["SG_TP8_BD"] = bd
    env["SG_TP8_ROOT"] = str(ROOT)
    # Freshness keys on the pybind .cu AND every megakernel/decoder header the TU
    # transitively includes (the bug fixes edit the HEADERS, not the .cu) so an
    # edited kernel is never silently skipped.
    _dep_globs = [str(_SCRATCH_TU)] + glob_headers()
    src_mtime = max((os.path.getmtime(p) for p in _dep_globs if os.path.exists(p)),
                    default=os.path.getmtime(_SCRATCH_TU))
    fresh = os.path.exists(so) and os.path.getmtime(so) >= src_mtime
    # Only one rank compiles (avoid 8 concurrent nvcc on the same .o). Others wait.
    # Skip the (slow) rebuild when the cached .so is newer than the source.
    if (rank == 0) and not fresh:
        subprocess.run(["bash", str(ROOT / "tuning/_tp8_build.sh")], check=True,
                       env=env, stdout=(None if verbose else subprocess.DEVNULL),
                       stderr=(None if verbose else subprocess.STDOUT))
    spec = importlib.util.spec_from_file_location("sg_tp8_scratch", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seed_everything(seed):
    import torch, numpy as np, random  # noqa
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_real_batch(B, seed, vocab, device):
    import torch  # noqa
    import grokking_race_v2 as g  # noqa
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": "decoder", "p": vocab - 2, "seed": seed,
              "frac_train": 0.5, "val_ratio": 0.10})
    tok, tgt, *_ = (d.to(device) for d in g.make_data_for_task(c, seed))
    Bw = min(B, int(tgt.shape[0])); Bw -= Bw % 16
    return tok[:Bw].to(torch.int32).contiguous(), tgt[:Bw].to(torch.int32).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--verify-steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ncta", type=int, default=64)  # multiple of TP=8
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--state-planes", type=int, default=3)
    ap.add_argument("--verbose-build", action="store_true")
    args = ap.parse_args()
    assert args.batch % 16 == 0 and args.ncta % 8 == 0

    import grokking_optimizers.nvshmem_bringup_ext as ext  # sets NVLS-disable defaults
    import torch  # noqa
    import torch.distributed as dist  # noqa
    from grokking_optimizers.host_bringup import TPBootstrap, tp_heap_stride_floats
    from grokking_optimizers.distributed import partition_tensor_parallel
    from grokking_optimizers.parallel import flagship_budget as fb
    from tuning.flagship_distributed import flagship_named_shapes

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert world == 8, "this is the 8-GPU TP8 run"
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    tp = 8
    ctas_per_pe = args.ncta // tp
    plan = TPBootstrap(global_rank=rank, world_size=world, tp=tp, tp_rank=rank,
                       dp=1, dp_rank=0, ctas_per_pe=ctas_per_pe)
    plan.validate()
    pe_start, pe_stride, pe_size = plan.pe_range()

    # ---- 0. BUILD + dlopen the TP8 megakernel module BEFORE nvshmem_init ----
    # CRITICAL (in-kernel device-NVSHMEM): nvshmem_init populates the device state
    # symbol (nvshmemi_device_state_d) in EVERY CUDA module loaded AT init time. The
    # TP8 scratch .so is device-linked against its OWN copy of libnvshmem_device, so
    # its in-kernel nvshmemx_barrier_block / nvshmem_ptr read that module's device
    # state — which stays NULL (→ in-kernel IMA in the team barrier) unless the .so
    # is dlopen'd BEFORE nvshmem_init. So load it here, before init_with_uniqueid.
    # rank-0 compiles; all ranks dlopen (build_tp8_module gates the compile on rank).
    if rank == 0:
        build_tp8_module(verbose=args.verbose_build, rank=0)
    dist.barrier()
    mod = build_tp8_module(verbose=False, rank=(1 if rank != 0 else 0))

    # ---- 1. NVSHMEM bring-up (UID bootstrap + team split + a small sym malloc) ----
    mod_nv = ext.build_bringup_module(verbose=args.verbose_build and rank == 0)
    n_uid = mod_nv.uniqueid_size()
    uid = mod_nv.get_uniqueid() if rank == 0 else (b"\x00" * n_uid)
    buf = torch.zeros(n_uid, dtype=torch.uint8, device=device)
    if rank == 0:
        buf.copy_(torch.frombuffer(bytearray(uid), dtype=torch.uint8).to(device))
    dist.broadcast(buf, src=0)
    uid = bytes(buf.cpu().numpy().tobytes())
    dist.barrier()
    mod_nv.init_with_uniqueid(rank, world, uid, local_rank)
    assert mod_nv.is_initialized() and mod_nv.world_pe() == rank
    team = mod_nv.team_split_strided(pe_start, pe_stride, pe_size)
    # The launcher uses NVSHMEM_TEAM_WORLD (team 0) for the pure-TP8 in-kernel path;
    # we still split (matches the smoke) to confirm the world is consistent.
    sym_floats = tp_heap_stride_floats(ctas_per_pe)
    heap = mod_nv.malloc_symmetric_heap(sym_floats)
    if rank == 0:
        print(f"[tp8] NVSHMEM up: world_pe={mod_nv.world_pe()} npes={mod_nv.world_npes()} "
              f"team={team} tp_n_pes={mod_nv.tp_n_pes()} sym_heap=0x{heap:x} "
              f"sym_floats={sym_floats:,}", flush=True)

    # ---- 1b. Register the TP8 megakernel module's NVSHMEM DEVICE STATE ----
    # The megakernel's in-kernel nvshmemx_barrier_block / nvshmem_ptr read
    # nvshmemi_device_state_d, which nvshmem_init does NOT populate in this
    # separately-device-linked dlopen'd .so. nvshmemx_cumodule_init (via the .so's
    # CUmodule) registers it. MUST run after nvshmem_init, before the first step.
    mod.init_device_state()
    dist.barrier()
    if rank == 0:
        print("[tp8] device-state registered (nvshmemx_cumodule_init) on all ranks", flush=True)

    # ---- 2. TP weight-shard report (resident params per rank) ----
    tp_plan = partition_tensor_parallel(flagship_named_shapes(), tp, rank, model="decoder")
    if rank == 0:
        print(f"[tp8] TP weight-shard: {len(tp_plan.shards)} tensors, "
              f"{tp_plan.n_split} split / {len(tp_plan.shards)-tp_plan.n_split} replicated; "
              f"per-rank Nmax={tp_plan.max_shard_numel:,} (== {fb.FLAGSHIP_NMAX//tp:,}? "
              f"{tp_plan.max_shard_numel == fb.FLAGSHIP_NMAX//tp}); "
              f"per-rank resident params={tp_plan.total_shard_numel:,} of {tp_plan.total_full_numel:,}",
              flush=True)

    # ---- 3. the scratch TP8 megakernel was built+dlopen'd at step 0 (BEFORE
    #         nvshmem_init, so its module device state got populated). Just verify. ----
    seed_everything(args.seed)
    assert int(mod.D) == fb.FLAGSHIP_D and int(mod.TOTAL) == fb.FLAGSHIP_TOTAL_PARAMS, \
        f"built D={int(mod.D)} TOTAL={int(mod.TOTAL)} (layout swap failed)"
    if rank == 0:
        print(f"[tp8] flagship module built: D={int(mod.D)} L={int(mod.LAYERS)} "
              f"TOTAL={int(mod.TOTAL):,} NUMT={int(mod.NUMT)}", flush=True)

    # ---- 4. params (replicated, identical seed) + ZeRO-3 sharded state ----
    g = torch.Generator(device=device).manual_seed(args.seed)
    params = (torch.randn(fb.FLAGSHIP_TOTAL_PARAMS, generator=g, device=device) * 0.02).contiguous()
    total = fb.FLAGSHIP_TOTAL_PARAMS
    # state buffer: >= 3*total + 1 (m|v|extra|loss). state-planes>=3.
    state = torch.zeros(max(args.state_planes, 3) * total + 8, dtype=torch.float32, device=device)
    tokens, targets = make_real_batch(args.batch, args.seed, fb.FLAGSHIP_VOCAB, device)

    # per-rank GiB (resident)
    def gib():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**3)

    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0
    bc1, bc2 = 1.0 - b1, 1.0 - b2
    losses = []
    cross_ok = True
    for step in range(1, args.steps + 1):
        loss, _g = mod.tc_train_step_tp8(params, tokens, targets, state,
                                         lr, b1, b2, eps, wd, bc1, bc2, step, args.ncta)
        lv = float(loss.item()); losses.append(lv)
        if step <= args.verify_steps:
            lt = torch.tensor([lv], device=device, dtype=torch.float64)
            gath = [torch.zeros_like(lt) for _ in range(world)]
            torch.cuda.synchronize(); dist.all_gather(gath, lt); torch.cuda.synchronize()
            dloss = max(abs(x.item() - lv) for x in gath)
            if rank == 0:
                vals = [x.item() for x in gath]
                print(f"[step {step}] loss={lv:.6f} cross-rank dloss={dloss:.3e} "
                      f"min={min(vals):.6f} max={max(vals):.6f}", flush=True)
            if dloss >= 1e-6:
                cross_ok = False
                if rank == 0:
                    print(f"[tp8] WARNING cross-rank loss disagree d={dloss:.3e} step {step}", flush=True)

    per_rank_gib = gib()
    all_gib = [None] * world
    dist.all_gather_object(all_gib, per_rank_gib)
    descends = len(losses) >= 2 and losses[-1] < losses[0]
    if rank == 0:
        print(f"[tp8] DONE steps={len(losses)} loss[0]={losses[0]:.6f} -> "
              f"loss[-1]={losses[-1]:.6f}  init ln(99)={math.log(99):.4f}", flush=True)
        print(f"[tp8] descends={descends} cross_rank_agree={cross_ok} "
              f"per_rank_GiB={per_rank_gib:.2f} all_ranks_GiB={[round(x,2) for x in all_gib]}",
              flush=True)
        print(f"[tp8] RESULT_JSON {{'ran_8gpu': true, 'steps': {len(losses)}, "
              f"'loss0': {losses[0]:.6f}, 'lossN': {losses[-1]:.6f}, "
              f"'descends': {str(descends).lower()}, 'cross_rank_agree': {str(cross_ok).lower()}, "
              f"'per_rank_gib': {per_rank_gib:.3f}}}", flush=True)
    dist.barrier()
    mod_nv.barrier_world()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
