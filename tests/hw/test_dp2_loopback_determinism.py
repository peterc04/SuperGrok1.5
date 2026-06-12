"""tests/hw/test_dp2_loopback_determinism.py — DP=2 NCCL-loopback cross-rank A/A/A.

IMPLEMENTS /workspace/.parallelism_design.md §7.1 (single-rank/multi-rank NCCL
loopback correctness of the B2-seam decomposition) + §6.4 (THE cross-rank
bit-determinism gate) at **DP=2 on ONE GPU** (task #25 DELIVERABLE 2). Both ranks
run on cuda:0 (``CUDA_VISIBLE_DEVICES=0`` for both), giving a REAL 2-rank
reduce/all-gather sharing one GPU — the collective wiring + the cross-rank
determinism gate validated WITHOUT a second GPU, before the 8×H100 window.

THE DECOMPOSED STEP per rank (design §2.1 [1]..[4], cut at the B2 seam §0.1):
    [1] production fused step on THIS rank's batch shard -> local grad[total]
        (the dispatch.py fused path returns grad_out; the in-kernel P3 runs too,
         but we IGNORE its param update and re-apply via the sharded kernel on the
         REDUCED grad — the ZeRO>=2 "kernel ends at local grad" cut, design §2.2)
    [2] reduce the per-rank grads across DP  -> each rank's OWNED grad shard
    [3] sharded_optimizer_kernel<Opt> (DELIVERABLE 1) over the OWNED shard
    [4] all-gather params  -> full updated params on every rank

NCCL MULTI-RANK-ONE-GPU (the loopback enabler). NCCL 2.20.5 REFUSES two ranks on
one physical device by default ("Duplicate GPU detected : rank 0 and rank 1 both
on CUDA device …", ncclInvalidUsage) — its communicator init dedups by (hostid,
PCI-busid). The documented loopback workaround is to give each rank a DISTINCT
``NCCL_HOSTID`` so the pair differs and NCCL treats the ranks as different
"nodes"; the collectives then run for-real on the shared device (correctness, not
perf — exactly the design §7.1 [SPEC] note). We set it per rank in
``_init_dist`` BEFORE init_process_group. (Measured: with the hostid trick,
all_gather_into_tensor + reduce_scatter_tensor both succeed on one H100.)

FIXED-ORDER REDUCE (the A/A/A non-negotiable, design §2.7/§6.4). NCCL's
ring/tree ``reduce_scatter`` float-summation order is NOT guaranteed bit-identical
run-to-run, which would break the cross-rank A/A/A gate. So for the grad reduce we
use the design's mandated **fixed-order/reduce-to-rank pattern**: NCCL
``all_gather`` the full per-rank grad (a PURE data movement → bit-exact, order-
free), then sum the gathered grads in **ascending rank order** locally in fp32 —
the cross-rank analogue of the in-kernel ascending-CTA fp32 sum that made the
single-GPU path A/A/A (dispatch.py:805-833). This is what makes (b) below pass.
We use ``ZeRO3Sharder``/``even_partition`` for the shard-ownership convention
(design §0.5 reuse) but do the reduce fixed-order (the §2.7 correction the raw
NCCL reduce-scatter needs).

THE THREE ASSERTS (design §6.4):
  (a) the final all-gathered params are IDENTICAL on both ranks (the shard dance
      reconstitutes the same full param on every rank);
  (b) repeating the WHOLE run 3× is BIT-IDENTICAL (the cross-rank A/A/A gate —
      fixed-order reduce + deterministic shard boundaries make it structural);
  (c) DP=2-loopback vs the single-GPU fused step on the UNSHARDED batch: NOT
      required bitwise (the grad reduce order differs — the unsharded kernel sums
      all B rows in one ascending-CTA pass; the DP=2 path sums two half-batch
      grads). Asserted within the parity tol (SG-family ~3e-5 fp32, campaign
      baseline) and the ACTUAL delta is reported.

GRAPH CAPTURE (design §7.4) lives in test_step_graph_capture.py (a sibling file).

IMPORT / SUITE SAFETY (matches test_3d_parallel.py): imports cleanly with no GPU
and no launched job; the pytest entry SKIPS when there is no Hopper GPU /
torchrun, and otherwise SPAWNS torchrun --nproc_per_node=2 as a subprocess and
asserts it exits 0 (so ``pytest`` runs the real 2-rank job). The worker logic runs
only under the launched ``__main__``/``--worker`` path.

Run:
    PYTHONPATH=. python -m pytest tests/hw/test_dp2_loopback_determinism.py -q -s
    # or drive the worker directly:
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 \
        tests/hw/test_dp2_loopback_determinism.py --worker --opt adamw --model decoder
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

import pytest

_THIS = os.path.abspath(__file__)
_REPO = os.path.abspath(os.path.join(os.path.dirname(_THIS), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Parity tol for assert (c): the campaign's SG-family fp32 baseline (~3e-5), per
# /workspace/.campaign_plan.md:12 and design §6.4. (c) need NOT be bitwise.
_PARITY_TOL = 3e-5


def _sm90a_available() -> bool:
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


# ════════════════════════════════════════════════════════════════════════════
#  WORKER side (runs under torchrun --nproc_per_node=2, both ranks on cuda:0)
# ════════════════════════════════════════════════════════════════════════════


def _init_dist():
    """Init a 2-rank NCCL group with BOTH ranks on cuda:0 (the loopback trick)."""
    import torch.distributed as dist
    rank = int(os.environ["RANK"])
    # DISTINCT NCCL_HOSTID per rank defeats NCCL 2.20.5's duplicate-GPU dedup so
    # two ranks can share cuda:0 (design §7.1 multi-rank-one-GPU). Must precede
    # init_process_group. P2P/SHM off is harmless here (one device, no real peer).
    os.environ["NCCL_HOSTID"] = f"sg-loopback-rank-{rank}"
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    # Block collectives on the HOST (not a GPU busy-spin) so a momentarily-slower
    # rank does not starve the other's full-GPU persistent megakernel on the shared
    # device — the second guard (with the symmetric-work structure) against the
    # collective-vs-megakernel contention of the loopback config.
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl")
    return dist, dist.get_rank(), dist.get_world_size()


def _build_cell(model, seed=1234):
    """Build the live race model + the FIXED full batch (SAME init on every rank —
    same seed). The batch is identical across ranks; each rank shards it in [1]."""
    import grokking_race_v2 as g
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": model, "use_amp": False, "use_fused": True,
              "compile_model": False, "matmul_precision": "bf16", "seed": seed,
              "frac_train": 0.5, "val_ratio": 0.10, "p": 97})
    dev = torch.device("cuda:0")
    data = tuple(d.to(dev) for d in g.make_data_for_task(c, seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = g.build_model(c, dev)
    return g, c, m, data, dev


def _make_optimizer(opt, g, params, c):
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                                 weight_decay=c["weight_decay"])
    if opt == "lion":
        return g.Lion(params, lr=c.get("lion_lr", 3e-4), betas=(c["beta1"], 0.99),
                      weight_decay=c.get("lion_wd", 3.0))
    if opt == "grokfast":
        return g.Grokfast(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                          weight_decay=c["weight_decay"],
                          grokfast_alpha=c.get("grokfast_alpha", 0.98),
                          grokfast_lamb=c.get("grokfast_lamb", 2.0))
    raise ValueError(f"unhandled elementwise opt {opt!r}")


_OPTID = {"adamw": 0, "lion": 1, "grokfast": 2}


def _build_sharded_binding():
    """Load the DELIVERABLE-1 sharded-opt JIT binding (shared build cache)."""
    from tests.hw.test_sharded_optimizer import _build_binding
    return _build_binding()


def _shard_batch(tokens, targets, world, rank):
    """Deterministically split a fixed batch's ROWS across ranks via the
    shard_map even-partition helper (design §7.1 "shard a fixed batch
    deterministically"). Each shard is truncated to a multiple of 16 so the wgmma
    dW K-loop accepts it (dispatch.py truncates B%16 internally too, but we make
    each shard ÷16 up front so the two shards are balanced + comparable)."""
    from grokking_optimizers.parallel.shard_map import even_partition
    B = int(tokens.shape[0])
    # Round the whole batch down to a multiple of 16*world so each rank's slice is
    # a multiple of 16 (clean wgmma tiles, no per-rank truncation drift).
    step = 16 * world
    Bw = B - (B % step)
    s, e = even_partition(Bw, world, rank)
    return tokens[s:e].contiguous(), targets[s:e].contiguous(), Bw


def _fixed_order_allreduce_grad(dist, grad, world):
    """FIXED-ORDER cross-rank grad reduce (design §2.7/§6.4): NCCL all_gather the
    full per-rank grad (pure movement, bit-exact), then sum in ASCENDING rank
    order in fp32. Returns the full reduced grad (same on every rank). Averaged by
    world (DDP/ZeRO convention — matches ZeRO3Sharder.reduce_scatter_grads's div)."""
    total = grad.numel()
    gathered = torch.empty(total * world, dtype=grad.dtype, device=grad.device)
    # Sync the local megakernel to completion BEFORE the collective so the two
    # never overlap on the shared device (the loopback lockstep).
    torch.cuda.synchronize()
    dist.all_gather_into_tensor(gathered, grad.contiguous())
    torch.cuda.synchronize()
    acc = torch.zeros(total, dtype=torch.float32, device=grad.device)
    for r in range(world):                       # ascending rank order — fixed
        acc += gathered[r * total:(r + 1) * total]
    acc /= float(world)
    return acc


def _run_decomposed_step(dist, rank, world, g, c, m, data, dev, opt, mod,
                         init_flat=None):
    """ONE DP=2 decomposed step. Returns (full_params_after, loss_local, Bw).

    If ``init_flat`` is given, the model's params are RESET to it before the step
    (so an A/A/A re-run starts from byte-identical init WITHOUT rebuilding the
    model — faster, and an equally-rigorous A/A/A: the whole decomposed step
    re-runs from the same init). When None, the model's current params are the
    init (the first run)."""
    from grokking_optimizers.dispatch import (fused_train_step, canonicalize_model,
                                              _opt_scalars_from)
    from grokking_optimizers.parallel.shard_map import even_partition
    canon = canonicalize_model(model_of(c))
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named)

    # Reset params to the shared init (if provided) so every A/A/A run + every rank
    # starts from the SAME weights. Then snapshot params_before.
    if init_flat is not None:
        off = 0
        for _, p in named:
            n = p.numel(); p.data.reshape(-1).copy_(init_flat[off:off + n]); off += n
    pb = torch.empty(total, device=dev)
    off = 0
    for _, p in named:
        n = p.numel(); pb[off:off + n].copy_(p.data.reshape(-1)); off += n

    # [1] production fused step on THIS rank's batch shard -> local grad[total].
    #     (in-kernel P3 also runs and mutates p.data; we restore pb below and
    #      re-apply via the sharded kernel on the REDUCED grad — the §2.2 cut.)
    tok_sh, tgt_sh, Bw = _shard_batch(data[0], data[1], world, rank)
    opt_obj = _make_optimizer(opt, g, m.parameters(), c)
    cache = {}
    loss, grad_local = fused_train_step(canon, opt, m, opt_obj, tok_sh, tgt_sh,
                                        state_cache=cache, step=1, return_grad=True,
                                        gemm_impl="wgmma")

    # [2] FIXED-ORDER reduce of the per-rank grads across DP -> full reduced grad.
    grad_full = _fixed_order_allreduce_grad(dist, grad_local, world)

    # [3] sharded_optimizer_kernel<Opt> over THIS rank's OWNED param/grad/state
    #     shard. ZeRO-3 element-even partition (design §2.4 — elementwise opt).
    s, e = even_partition(total, world, rank)
    shard_numel = e - s
    sc = _opt_scalars_from(opt_obj, 1)
    full = pb.clone()                             # start from params_before
    params_shard = full[s:e].contiguous()
    grad_shard = grad_full[s:e].contiguous()
    # Owned shard of the zero-init [m|v|extra] state (laid out [m|v|ema] over the
    # shard so the binding's m=state[0:n]/v=state[n:2n]/ema=state[2n:3n] aligns).
    state_shard = torch.zeros(3 * shard_numel, dtype=torch.float32, device=dev)
    mod.sharded_opt_step(
        _OPTID[opt], params_shard, grad_shard, state_shard,
        sc["lr"], sc["beta1"], sc["beta2"], sc["eps"], sc["weight_decay"],
        sc["bc1"], sc["bc2"], sc.get("alpha", 0.98), sc.get("lamb", 2.0), 1)

    # [4] all-gather the updated param shards -> full params on every rank. Use a
    #     PADDED all_gather_into_tensor over the even partition (uniform per-rank
    #     length), then stitch (mirrors ZeRO3Sharder.all_gather_params' convention).
    per = (total + world - 1) // world
    buf = torch.zeros(per, dtype=torch.float32, device=dev)
    buf[:shard_numel].copy_(params_shard)
    gathered = torch.empty(per * world, dtype=torch.float32, device=dev)
    torch.cuda.synchronize()                      # sharded-opt done before collective
    dist.all_gather_into_tensor(gathered, buf)
    torch.cuda.synchronize()
    out = torch.empty(total, dtype=torch.float32, device=dev)
    for r in range(world):
        rs = min(r * per, total); re = min(rs + per, total)
        n = re - rs
        if n > 0:
            out[rs:re].copy_(gathered[r * per:r * per + n])
    return out, float(loss), Bw


# Helper: the canonical model name is carried on c (set in _build_cell).
def model_of(c):
    return c["model_type"]


def _single_gpu_reference(g, c, m, data, dev, opt, world, init_flat):
    """The single-GPU fused step on the UNSHARDED (full) batch — assert (c). Resets
    ``m`` to the SAME init the workers used, then runs the in-kernel-P3 fused step
    on the SAME truncated batch the two shards together cover (Bw rows). The result
    is the in-kernel P3 update over the FULL-batch grad (vs the DP=2 path's
    averaged two-half-batch grad — they agree up to float reduction order)."""
    from grokking_optimizers.dispatch import fused_train_step, canonicalize_model
    canon = canonicalize_model(model_of(c))
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named)
    # Reset to the shared init.
    off = 0
    for _, p in named:
        n = p.numel(); p.data.reshape(-1).copy_(init_flat[off:off + n]); off += n
    # Same Bw the workers cover (B rounded to 16*world).
    B = int(data[0].shape[0]); step = 16 * world; Bw = B - (B % step)
    opt_obj = _make_optimizer(opt, g, m.parameters(), c)
    cache = {}
    fused_train_step(canon, opt, m, opt_obj, data[0][:Bw], data[1][:Bw],
                     state_cache=cache, step=1, return_grad=True, gemm_impl="wgmma")
    out = torch.empty(total, device=dev)
    off = 0
    for _, p in named:
        n = p.numel(); out[off:off + n].copy_(p.data.reshape(-1)); off += n
    return out


def worker_main(opt, model):
    """The torchrun worker: run the DP=2 decomposed step 3× (A/A/A), assert (a)+(b)
    on every rank and (c) on rank 0. Prints a per-rank summary; exits non-zero on
    any failure (the pytest harness asserts the exit code)."""
    dist, rank, world = _init_dist()
    assert world == 2, f"expected world=2, got {world}"
    mod = _build_sharded_binding()

    # Build the cell ONCE (same seed on every rank ⇒ byte-identical init + the SAME
    # fixed batch), and snapshot the initial flat params. Each A/A/A run RESETS the
    # model to this snapshot before the decomposed step (faster than rebuilding,
    # and an equally-rigorous A/A/A — the whole step re-runs from the same init).
    g, c, m, data, dev = _build_cell(model)
    named0 = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    init_flat = torch.cat([p.data.reshape(-1) for _, p in named0]).clone()

    # Run the whole decomposed pipeline 3× from the SAME init (the A/A/A gate: the
    # only thing under test is run-to-run non-determinism).
    runs = []
    losses = []
    for _ in range(3):
        full, loss, Bw = _run_decomposed_step(dist, rank, world, g, c, m, data,
                                               dev, opt, mod, init_flat=init_flat)
        runs.append(full)
        losses.append(loss)

    # (b) A/A/A: the 3 full-param results are BIT-IDENTICAL on THIS rank.
    aaa_ok = bool(torch.equal(runs[0], runs[1]) and torch.equal(runs[1], runs[2]))

    # (a) cross-rank identity: the final params match the OTHER rank's, bit-for-bit.
    # all_gather the run-0 full params from both ranks and compare.
    total = runs[0].numel()
    gathered = torch.empty(total * world, dtype=torch.float32, device=runs[0].device)
    torch.cuda.synchronize()
    dist.all_gather_into_tensor(gathered, runs[0].contiguous())
    torch.cuda.synchronize()
    rank0_params = gathered[0:total]
    rank1_params = gathered[total:2 * total]
    cross_ok = bool(torch.equal(rank0_params, rank1_params))

    # (c) vs single-GPU unsharded reference. BOTH ranks compute it (only rank 0 uses
    # it for the assert) so the megakernel work stays SYMMETRIC across ranks — this
    # is the multi-rank-one-GPU lockstep that avoids the NCCL-collective-vs-persistent-
    # megakernel starvation: one rank busy-spinning in a collective while the other is
    # still in a full-GPU megakernel deadlocks the shared device. Keeping every rank's
    # device timeline identical (same fused steps, same collectives, in the same order)
    # is what makes the loopback progress (the simple fused→all_gather probe proved
    # lockstep works; the asymmetric rank0-only reference is what hung).
    ref = _single_gpu_reference(g, c, m, data, dev, opt, world, init_flat)
    delta_c = (runs[0] - ref).abs().max().item()
    denom = ref.abs().max().item() + 1e-30
    rel_c = delta_c / denom
    parity_c = bool(rel_c < _PARITY_TOL)
    if rank == 0:
        print(f"[dp2-loopback rank0] {opt}/{model}: (a) cross-rank bit-eq={cross_ok} | "
              f"(b) A/A/A bit-eq={aaa_ok} | (c) vs single-GPU unsharded: "
              f"maxd={delta_c:.3e} rel={rel_c:.3e} (tol {_PARITY_TOL:.0e}) "
              f"parity={parity_c} | loss_shard={losses[0]:.5f}",
              flush=True)
    else:
        print(f"[dp2-loopback rank1] {opt}/{model}: (a) cross-rank bit-eq={cross_ok} | "
              f"(b) A/A/A bit-eq={aaa_ok} | loss_shard={losses[0]:.5f}", flush=True)

    # Reduce a pass/fail bit across ranks (the design §6.3 "all ranks agree" check):
    # (a)+(b) must hold on EVERY rank; (c) is gated on rank 0's comparison (runs[0]
    # is rank 0's full params vs rank 0's reference) and broadcast so every rank
    # exits consistently.
    local_ok = 1 if (aaa_ok and cross_ok) else 0
    ok_t = torch.tensor([local_ok], dtype=torch.int32, device=runs[0].device)
    torch.cuda.synchronize()                       # reference megakernel done first
    dist.all_reduce(ok_t)                          # sum; must == world
    all_ranks_ok = int(ok_t.item()) == world
    c_t = torch.tensor([1 if parity_c else 0], dtype=torch.int32, device=runs[0].device)
    dist.broadcast(c_t, src=0)                      # rank-0's (c) verdict wins
    c_ok_all = int(c_t.item()) == 1

    dist.barrier()
    dist.destroy_process_group()

    if not (all_ranks_ok and c_ok_all):
        if rank == 0:
            print(f"[dp2-loopback] FAIL: all_ranks_ok={all_ranks_ok} "
                  f"(a+b) c_ok={c_ok_all} (c)", flush=True)
        return 1
    if rank == 0:
        print(f"[dp2-loopback] OK: {opt}/{model} — cross-rank identity + A/A/A "
              f"bit-exact on both ranks; (c) within parity tol.", flush=True)
    return 0


# ════════════════════════════════════════════════════════════════════════════
#  PYTEST side — spawn torchrun and assert it exits 0 (the REAL 2-rank run)
# ════════════════════════════════════════════════════════════════════════════


def _torchrun_available() -> bool:
    import shutil
    return shutil.which("torchrun") is not None


def _skip_reason():
    if not _sm90a_available():
        return "DP=2 loopback gate needs an sm_90 (Hopper) GPU"
    if not _torchrun_available():
        return "torchrun not on PATH"
    return None


def _spawn_worker(opt, model, port):
    """Spawn torchrun --nproc_per_node=2 on cuda:0 for one cell; return CompletedProcess."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"           # BOTH ranks on cuda:0 (loopback)
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    env["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    cmd = [
        "torchrun", "--nproc_per_node=2", f"--master_port={port}",
        _THIS, "--worker", "--opt", opt, "--model", model,
    ]
    return subprocess.run(cmd, env=env, cwd=_REPO, capture_output=True, text=True,
                          timeout=900)


# Deterministic per-cell master_port (avoid PYTHONHASHSEED-dependent collisions).
_DP2_CELLS = [
    ("adamw", "decoder"), ("lion", "decoder"), ("grokfast", "decoder"),
    ("adamw", "vit"),
]


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
@pytest.mark.parametrize("opt,model", _DP2_CELLS)
def test_dp2_loopback_cross_rank_aaa(opt, model):
    """torchrun --nproc_per_node=2 (both on cuda:0): the DP=2 decomposed step is
    cross-rank identical + A/A/A bit-exact, and within parity tol of the single-
    GPU unsharded step. Design §7.1 + §6.4."""
    port = 29610 + _DP2_CELLS.index((opt, model))
    proc = _spawn_worker(opt, model, port)
    # Surface the worker's stdout (the per-rank summary lines) for -s visibility.
    tail = "\n".join(proc.stdout.splitlines()[-25:])
    print(f"\n--- torchrun {opt}/{model} (rc={proc.returncode}) ---\n{tail}")
    if proc.returncode != 0:
        err_tail = "\n".join(proc.stderr.splitlines()[-40:])
        raise AssertionError(
            f"DP=2 loopback worker failed for {opt}/{model} (rc={proc.returncode}).\n"
            f"--- stdout tail ---\n{tail}\n--- stderr tail ---\n{err_tail}")


# ════════════════════════════════════════════════════════════════════════════
#  __main__ — dispatch worker vs a local all-cells driver
# ════════════════════════════════════════════════════════════════════════════


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true",
                    help="run as a torchrun worker (set by the spawned job)")
    ap.add_argument("--opt", default="adamw", choices=list(_OPTID))
    ap.add_argument("--model", default="decoder", choices=["decoder", "vit", "mamba"])
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.worker:
        return worker_main(args.opt, args.model)
    # Non-worker __main__: spawn the worker for the requested cell (a convenience
    # for manual runs; mirrors the pytest path).
    reason = _skip_reason()
    if reason is not None:
        print(f"[dp2-loopback] SKIP: {reason}")
        return 0
    proc = _spawn_worker(args.opt, args.model, 29599)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
