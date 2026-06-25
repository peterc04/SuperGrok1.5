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
  * The HOST bring-up (the TP weight-shard + the NVSHMEM TP-team bootstrap) comes from
    grokking_optimizers.distributed.partition_tensor_parallel + .TPBootstrap /
    .bootstrap_tp_team — the launcher EDIT-E companion (dist_step.md §6.C/§6.D).
  * Mesh = TP8 x DP1 x PP1 + ZeRO-3 (the recommended config; see flagship_budget.py §0).
    Other meshes are selectable but the budget gate refuses an OOM config LOUDLY.

WHAT THIS IS NOT
  * It does not edit the committed _ops / dispatch (the flagship is a coexisting variant
    build, exactly like decoder_bench.py — no setup.py change, no 33/33-gate impact).
  * It does not re-implement the fused step: the cross-rank DP path reuses the proven
    grokking_optimizers.parallel.distributed_step / zero3 modules; the TP all-reduce is
    IN-KERNEL (tp_transport.cuh), kept inside the one megakernel launch.

USAGE
  python tuning/flagship_distributed.py --help                     # per-rank budget table
  python tuning/flagship_distributed.py --dry-run --model decoder --gpus 8  # CPU plan proof
  torchrun --nproc_per_node=8 tuning/flagship_distributed.py --steps 2 --dry-run
  torchrun --nproc_per_node=8 tuning/flagship_distributed.py --opt supergrok2 --steps 50
  torchrun --nproc_per_node=8 tuning/flagship_distributed.py --bench-all --steps 200

DRY-RUN CONTRACT (the cheap gate, no GPU/NVSHMEM): validates the full 4D mesh, the TP
Megatron weight-shard (per-rank Nmax = kDecMaxTensorNumel/TP), the symmetric-heap stride
+ tp_comm_handle encoding, and the per-rank HBM budget for EVERY rank — on CPU — so the
operator SEES the fit + bring-up proof before spending GPU-hours on the (large) flagship
compile. The single-process `--gpus 8` form validates all 8 ranks; the torchrun form
validates this rank.

REAL DATA (north star): grokking_race_v2.make_data_for_task (the modular-arithmetic
grokking task: tokens [B,seq=4] int32, targets [B] int32, vocab p=99) — the same pipeline
the production race uses; identical across ranks (seeded), TP-replicated batch.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grokking_optimizers.parallel import flagship_budget as fb  # noqa: E402
from grokking_optimizers.distributed import (  # noqa: E402
    ParallelConfig,
    partition_tensor_parallel,
    TPBootstrap,
    bootstrap_tp_team,
)
from grokking_optimizers.host_bringup import (  # noqa: E402
    nvshmem_available,
    find_nvshmem_host_lib,
    TPBootstrapBlocked,
    decode_tp_comm_handle,
)

_TC_TU = str(ROOT / "csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu")
_FLAGSHIP_LAYOUT = "csrc/fused/sm_90/decoder_flagship_layout.cuh"


# ───────────────────────── flagship named-sizes (host TP shard input) ─────────
# The decoder's named_parameters() order + per-tensor 2D shapes (so the Megatron
# col/row split knows WHICH dim to divide). Built from the flagship dims WITHOUT
# importing torch or the kernel (pure shapes), so the dry-run validates the TP shard
# on CPU. This mirrors decoder_flagship_layout.cuh's kSizes ordering: per layer
# [in_proj 3d×d, out_proj d×d, ff.0 4d×d, ff.2 d×4d] + the norms/biases, plus the
# embeddings + final unembed. The shard math only needs the SPLIT tensors' shapes to
# be exact; the replicated remainder is summed as a single "replicated" bucket so the
# total matches kDecTotalElems.

def flagship_named_shapes():
    """[(name, (out,in) | (numel,))] for the flagship decoder, named_parameters()
    order, faithful to decoder_flagship_layout.cuh's kSizes histogram so the TP shard's
    per-rank Nmax matches the real kernel (kDecMaxTensorNumel/TP, NOT an artifact of
    lumping the replicated tensors). The four large per-layer matrices carry their 2D
    shape (the Megatron col/row split needs it to pick the split dim); the replicated
    tensors (embeddings/unembed/norms/biases) are emitted INDIVIDUALLY (their real
    sizes), so the largest replicated tensor is the embedding (158,400 = vocab*d-ish),
    far below a per-rank split shard (10.24M/8 = 1.28M). Grand total == kDecTotalElems.

    REAL replicated histogram (decoder_flagship_layout.cuh kSizes): {158400:2, 6400:49,
    4800:48, 1600:290, 99:1} (sum 1,324,899). We reproduce it EXACTLY so the per-rank
    resident-params and Nmax are the live numbers, not estimates."""
    d, dff, L = fb.FLAGSHIP_D, fb.FLAGSHIP_DFF, fb.FLAGSHIP_LAYERS
    shapes = []
    split_total = 0
    for li in range(L):
        # attention: in_proj (QKV, column-parallel) 3d×d ; out_proj (row-parallel) d×d
        shapes.append((f"layers.{li}.attn.in_proj.weight", (3 * d, d)))
        shapes.append((f"layers.{li}.attn.out_proj.weight", (d, d)))
        # mlp: ff.0 (column-parallel) 4d×d ; ff.2 (row-parallel) d×4d
        shapes.append((f"layers.{li}.mlp.ff.0.weight", (dff, d)))
        shapes.append((f"layers.{li}.mlp.ff.2.weight", (d, dff)))
        split_total += (3 * d * d) + (d * d) + (dff * d) + (d * dff)
    # Replicated tensors, INDIVIDUALLY (faithful to the layout histogram). Names are
    # representative; only the SIZES are load-bearing (the TP shard leaves these whole).
    repl_hist = [(158400, 2), (6400, 49), (4800, 48), (1600, 290), (99, 1)]
    ri = 0
    for sz, count in repl_hist:
        for _ in range(count):
            shapes.append((f"replicated.{ri}.weight", (sz,)))
            ri += 1
    repl_total = sum(sz * count for sz, count in repl_hist)
    grand = split_total + repl_total
    if grand != fb.FLAGSHIP_TOTAL_PARAMS:
        raise AssertionError(
            f"flagship named-shapes total {grand:,} != kDecTotalElems "
            f"{fb.FLAGSHIP_TOTAL_PARAMS:,} (split {split_total:,} + replicated "
            f"{repl_total:,}) — shape table drifted from the layout header")
    return shapes


# ───────────────────────── host bring-up plan (CPU, dry-run) ──────────────────

def build_host_plan(args, *, global_rank: int, world: int):
    """Assemble the per-rank HOST bring-up plan: TP weight-shard + budget + the
    NVSHMEM TP-team bootstrap plan. Pure CPU — no torch, no GPU, no NVSHMEM init.
    Returns (tp_plan, budget, ncta, ctas_per_pe, tp_boot)."""
    # mesh coords (Megatron linearization: TP fastest, then PP, then DP).
    tp, pp, dp = args.tp, args.pp, args.dp
    tp_rank = global_rank % tp
    pp_rank = (global_rank // tp) % pp
    dp_rank = global_rank // (tp * pp)

    # TP weight-shard (the load-bearing Nmax/TP shrink).
    tp_plan = partition_tensor_parallel(flagship_named_shapes(), tp, tp_rank,
                                        model="decoder")

    # nCTA + budget (the fit gate).
    ncta = args.ncta_cap or fb.auto_ncta(args.opt, tp=tp, pp=pp, dp=dp,
                                         zero3=args.zero3, B=args.batch)
    budget = fb.per_rank_budget(args.opt, tp=tp, pp=pp, dp=dp, zero3=args.zero3,
                                ncta=ncta, B=args.batch)
    if ncta % tp != 0:
        # The launcher asserts nCTA % TP == 0 (symmetric heap = ctas_per_pe slots).
        # Round nCTA DOWN to a multiple of TP for the heap math (the fit only improves).
        ncta = (ncta // tp) * tp
        ncta = max(ncta, tp)
    ctas_per_pe = max(ncta // tp, 1)

    tp_boot = TPBootstrap(global_rank=global_rank, world_size=world,
                          tp=tp, tp_rank=tp_rank, dp=dp, dp_rank=dp_rank,
                          pp=pp, pp_rank=pp_rank, ctas_per_pe=ctas_per_pe)
    return tp_plan, budget, ncta, ctas_per_pe, tp_boot


def print_rank_plan(args, *, global_rank, world, tp_plan, budget, ncta,
                    ctas_per_pe, tp_boot):
    """Print the CPU plan proof for one rank (the --dry-run output)."""
    tp_boot.validate()
    team_handle = bootstrap_tp_team(tp_boot, allow_dry=True)
    cc = tp_boot.commctx_fields(team_handle)
    nmax = tp_plan.max_shard_numel
    expect_nmax = fb.FLAGSHIP_NMAX // args.tp
    ok_nmax = (nmax == expect_nmax)
    print(f"[rank {global_rank}/{world}] tp={args.tp} tp_rank={tp_boot.tp_rank} "
          f"dp={args.dp} dp_rank={tp_boot.dp_rank} pp={args.pp} pp_rank={tp_boot.pp_rank}")
    print(f"    TP weight-shard: {len(tp_plan.shards)} tensors, "
          f"{tp_plan.n_split} split / {len(tp_plan.shards) - tp_plan.n_split} replicated")
    print(f"    per-rank Nmax = {nmax:,}  (== kDecMaxTensorNumel/TP = {expect_nmax:,}? "
          f"{'YES' if ok_nmax else 'NO'})")
    print(f"    per-rank resident params = {tp_plan.total_shard_numel:,}  "
          f"(of full {tp_plan.total_full_numel:,})")
    print(f"    opt={args.opt} nCTA={ncta} ctas_per_pe={ctas_per_pe}  "
          f"budget TOTAL={budget.total_gb:.2f} GiB ({'FITS' if budget.fits else 'OOM'})")
    print(f"    NVSHMEM TP-team: PE range {tp_boot.pe_range()} "
          f"(pure-TP-world={tp_boot.is_pure_tp_world}); "
          f"sym heap = {tp_boot.sym_floats:,} floats ({tp_boot.sym_bytes/1e6:.1f} MB)")
    print(f"    CommCtx.tp_comm_handle = {cc['tp_comm_handle']} "
          f"(decode -> team {decode_tp_comm_handle(cc['tp_comm_handle'])}); "
          f"tp_team_n_pes={cc['tp_team_n_pes']} tp_team_local_pe={cc['tp_team_local_pe']}")
    if not ok_nmax:
        raise SystemExit(
            f"[rank {global_rank}] TP shard Nmax {nmax} != expected "
            f"{expect_nmax} — the Megatron col/row split did not shrink the max "
            f"tensor by TP (shard plan drift).")
    if not budget.fits:
        raise SystemExit(
            f"[rank {global_rank}] config OOMs ({budget.total_gb:.1f} GiB > usable) "
            f"— lower --tp / --ncta-cap or pick a cheaper opt (budget gate).")


def run_dry_plan(args) -> int:
    """CPU-only validation of the whole job's plan/mesh/shard/budget/bootstrap math.
    Single-process `--gpus N` validates ALL N ranks; under torchrun validates this rank.
    NO torch, NO GPU, NO NVSHMEM init."""
    world = args.gpus if args.gpus else (args.tp * args.pp * args.dp)
    cfg = ParallelConfig(data_parallel=args.dp, tensor_parallel=args.tp,
                         pipeline_parallel=args.pp,
                         zero_stage=(3 if args.zero3 else 0),
                         backend="nccl", use_megakernel=True)
    cfg.validate_against_world(world)  # DP*TP*PP == world (loud otherwise)

    print(fb.format_budget_table(tp=args.tp, pp=args.pp, dp=args.dp,
                                 zero3=args.zero3, B=args.batch, ncta=None))
    print()
    print(f"NVSHMEM host lib: {find_nvshmem_host_lib() or 'NOT FOUND'} "
          f"(available={nvshmem_available()})")
    print(f"4D mesh: TP={args.tp} PP={args.pp} DP={args.dp} ZeRO-3={args.zero3} "
          f"=> world_size={world}; model={args.model!r}")
    print()

    env_rank = os.environ.get("RANK")
    if env_rank is not None and not args.gpus:
        ranks = [int(env_rank)]   # torchrun: just this rank
    else:
        ranks = list(range(world))  # single-process: prove every rank's plan

    for r in ranks:
        tp_plan, budget, ncta, ctas_per_pe, tp_boot = build_host_plan(
            args, global_rank=r, world=world)
        print_rank_plan(args, global_rank=r, world=world, tp_plan=tp_plan,
                        budget=budget, ncta=ncta, ctas_per_pe=ctas_per_pe,
                        tp_boot=tp_boot)
    print()
    print(f"[dry-run] plan/mesh/TP-shard/budget/bootstrap validated for "
          f"{len(ranks)} rank(s); no kernel build/launch, no NVSHMEM init. Exit 0.")
    return 0


# ───────────────────────── build (per rank, live run) ─────────────────────────

def build_flagship_module(tp: int, pp: int, dp: int, zero3: bool, *,
                          has_nvshmem: bool = False, verbose: bool = False):
    """JIT-build the flagship TC megakernel module for THIS rank's mesh. Coexisting
    variant (own module name + build dir) — never touches _ops. The layout swap is the
    proven impl_diffs/flagship.md route: force-include the flagship header + pre-define
    the committed header's include guard so its body is skipped."""
    import torch  # noqa: PLC0415
    from torch.utils.cpp_extension import load  # noqa: PLC0415

    z = 3 if zero3 else 0
    flags = [
        "-O3", "-std=c++17", "--expt-relaxed-constexpr",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-gencode=arch=compute_90a,code=compute_90a",
        "-DSG_TUNED_GEMM_IMPL=1",
        "-DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1",
        "-include", _FLAGSHIP_LAYOUT,
        f"-DSG_FLAGSHIP_TP={tp}",
        f"-DSG_FLAGSHIP_PP={pp}",
        f"-DSG_FLAGSHIP_DP={dp}",
        f"-DSG_FLAGSHIP_ZERO={z}",
    ]
    extra_ldflags = []
    if has_nvshmem:
        flags.append("-DSG_HAS_NVSHMEM=1")
        flags.append("-rdc=true")
        home = os.environ.get("NVSHMEM_HOME",
                              "/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem")
        flags.append(f"-I{home}/include")
        extra_ldflags = [f"-L{home}/lib", "-lnvshmem_host", "-lnvshmem_device"]
    name = f"mega_decoder_flagship_tc_tp{tp}_pp{pp}_dp{dp}_z{z}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    mod = load(name=name, sources=[_TC_TU],
               extra_include_paths=[str(ROOT)],
               extra_cuda_cflags=flags,
               extra_cflags=["-O3", "-std=c++17"],
               extra_ldflags=extra_ldflags,
               verbose=verbose)
    assert int(mod.D) == fb.FLAGSHIP_D, f"built D={int(mod.D)} != {fb.FLAGSHIP_D}"
    assert int(mod.LAYERS) == fb.FLAGSHIP_LAYERS, f"built L={int(mod.LAYERS)}"
    assert int(mod.TOTAL) == fb.FLAGSHIP_TOTAL_PARAMS, f"built TOTAL={int(mod.TOTAL)}"
    return mod


def seed_everything(seed: int):
    import torch  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import random  # noqa: PLC0415
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def make_flagship_params(seed: int, device):
    import torch  # noqa: PLC0415
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(fb.FLAGSHIP_TOTAL_PARAMS, generator=g, device=device)
            * 0.02).contiguous()


def make_real_batch(B: int, seed: int, device):
    import torch  # noqa: PLC0415
    import grokking_race_v2 as g  # noqa: PLC0415
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": "decoder", "p": fb.FLAGSHIP_VOCAB - 2, "seed": seed,
              "frac_train": 0.5, "val_ratio": 0.10})
    tok, tgt, *_ = (d.to(device) for d in g.make_data_for_task(c, seed))
    Bw = min(B, int(tgt.shape[0]))
    Bw -= Bw % 16
    return tok[:Bw].to(torch.int32).contiguous(), tgt[:Bw].to(torch.int32).contiguous()


# ───────────────────────── the rank worker (live run) ─────────────────────────

def run_rank(args) -> int:
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    from grokking_optimizers.distributed import DistributedContext  # noqa: PLC0415

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cfg = ParallelConfig(data_parallel=args.dp, tensor_parallel=args.tp,
                         pipeline_parallel=args.pp,
                         zero_stage=(3 if args.zero3 else 0),
                         backend="nccl", use_megakernel=True)
    cfg.validate_against_world(world)

    # Host plan (the same plan the dry-run prints — the single source).
    tp_plan, bud, ncta, ctas_per_pe, tp_boot = build_host_plan(
        args, global_rank=rank, world=world)
    if rank == 0:
        print(fb.format_budget_table(tp=args.tp, pp=args.pp, dp=args.dp,
                                     zero3=args.zero3, B=args.batch, ncta=None),
              flush=True)
        print(f"[rank0] opt={args.opt} nCTA={ncta} per-rank TOTAL={bud.total_gb:.2f} "
              f"GiB ({'FITS' if bud.fits else 'OOM'}); Nmax/rank={tp_plan.max_shard_numel:,}",
              flush=True)
    if not bud.fits:
        raise SystemExit(f"[rank{rank}] config OOMs ({bud.total_gb:.1f} GiB) — "
                         f"lower --tp/--ncta-cap or pick a cheaper opt.")

    # init the process group + mesh.
    os.environ.setdefault("NCCL_HOSTID", f"sg-flagship-rank-{rank}")
    dist.init_process_group(backend="nccl")
    _dctx = DistributedContext.from_config(cfg)

    # NVSHMEM TP-team bootstrap (only on the real device-initiated path).
    if args.nvshmem:
        def _uid_bcast(uid):  # tiny torch.distributed broadcast of the UID blob
            import torch as _t  # noqa: PLC0415
            buf = _t.zeros(1, dtype=_t.uint8, device=device)  # placeholder seam
            dist.broadcast(buf, src=0)
            return uid
        try:
            team = bootstrap_tp_team(tp_boot, uid_broadcast=_uid_bcast)
            if rank == 0:
                print(f"[rank0] NVSHMEM TP team bootstrapped (handle={team})",
                      flush=True)
        except TPBootstrapBlocked as e:
            raise SystemExit(f"[rank{rank}] NVSHMEM bring-up blocked: {e}")

    # identical seed + flagship build.
    seed_everything(args.seed)
    mod = build_flagship_module(args.tp, args.pp, args.dp, args.zero3,
                                has_nvshmem=args.nvshmem,
                                verbose=args.verbose_build)
    params = make_flagship_params(args.seed, device)
    state = torch.zeros(args.state_planes * fb.FLAGSHIP_TOTAL_PARAMS,
                        dtype=torch.float32, device=device)
    tokens, targets = make_real_batch(args.batch, args.seed, device)

    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0
    bc1, bc2 = 1.0 - beta1, 1.0 - beta2

    losses = []
    for step in range(1, args.steps + 1):
        loss, _grad = mod.tc_train_step(params, tokens, targets, state,
                                        lr, beta1, beta2, eps, wd, bc1, bc2,
                                        step, ncta)
        loss_v = float(loss.item())
        losses.append(loss_v)

        if step <= args.verify_steps:
            lt = torch.tensor([loss_v], device=device, dtype=torch.float64)
            gathered = [torch.zeros_like(lt) for _ in range(world)]
            torch.cuda.synchronize(); dist.all_gather(gathered, lt)
            torch.cuda.synchronize()
            lmax = max(abs(g.item() - loss_v) for g in gathered)
            chk = torch.tensor([float(params.double().sum().item())], device=device,
                               dtype=torch.float64)
            gchk = [torch.zeros_like(chk) for _ in range(world)]
            torch.cuda.synchronize(); dist.all_gather(gchk, chk)
            torch.cuda.synchronize()
            chkmax = max(abs(g.item() - chk.item()) for g in gchk)
            if rank == 0:
                print(f"[step {step}] loss={loss_v:.6f}  cross-rank dloss={lmax:.2e}  "
                      f"dparam-checksum={chkmax:.2e}", flush=True)
            assert lmax < 1e-9, \
                f"cross-rank loss disagreement d={lmax:.2e} at step {step}"

    if rank == 0:
        print(f"[rank0] done {args.steps} steps  final loss={losses[-1]:.6f}  "
              f"(init ln(99)={math.log(99):.4f})", flush=True)
    dist.barrier(); dist.destroy_process_group()
    return 0


# ───────────────────────── CLI ───────────────────────────────────────────────

def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n" + fb.format_budget_table(**fb.RECOMMENDED, B=512))
    ap.add_argument("--model", default="decoder", choices=["decoder"],
                    help="model cell (only 'decoder' has the Megatron TP roster wired)")
    ap.add_argument("--gpus", type=int, default=0,
                    help="world size for a SINGLE-PROCESS --dry-run plan proof (validate "
                         "all N ranks on CPU); ignored under torchrun (uses WORLD_SIZE)")
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
                         "(-DSG_HAS_NVSHMEM=1; the 8xH100 path). Default: loopback.")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the per-rank plan/mesh/TP-shard/budget/bootstrap on "
                         "CPU and exit (no build, no launch, no NVSHMEM init)")
    ap.add_argument("--verbose-build", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    assert args.batch % 16 == 0, "--batch must be divisible by 16 (wgmma dW K-loop)"
    if args.dry_run:
        if args.bench_all:
            rc = 0
            for opt in fb.ALL_OPTIMIZERS:
                args.opt = opt
                rc |= run_dry_plan(args)
            return rc
        return run_dry_plan(args)
    if args.bench_all:
        rc = 0
        for opt in fb.ALL_OPTIMIZERS:
            args.opt = opt
            rc |= run_rank(args)
        return rc
    return run_rank(args)


if __name__ == "__main__":
    raise SystemExit(main())
