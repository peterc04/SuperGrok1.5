"""tests/hw/test_distributed_step.py — gates for the §6.2 production-shaped
``fused_train_step_distributed`` (grokking_optimizers/parallel/distributed_step).

TWO GATES (task #25 phase-2, Lane C):

1. WORLD=1 DECOMPOSED IDENTITY (design §7.1, single process): the FULL
   decomposition forced at world=1 (gather → fused step → identity reduce →
   sharded apply over the whole flat set → gather → scatter) must be
   BIT-IDENTICAL to the plain ``fused_train_step`` path over 2 consecutive
   steps (params after each step + state planes vs the kernel cache slices).
   This is the degenerate-collective specialization the design names: "the
   sharded-opt-kernel-vs-fused-P3 equivalence with ZERO multi-GPU hardware" —
   now exercised THROUGH the production-shaped module instead of inline test
   code.

2. WORLD=2 LOOPBACK THROUGH THE MODULE (torchrun, both ranks cuda:0 via the
   ef433ac NCCL_HOSTID trick): each rank runs the MODULE's step ×2 from the
   same-seed init; asserts (a) cross-rank full params identical bitwise after
   each step, (b) the 2-step trajectory equals a rank-0-computed single-GPU
   plain reference within the campaign parity tol (the batch-shard reduce
   reassociation — same tolerance class the DP=2 gate measured at 6.6e-6..
   1.2e-5). The collective flow is the SAME sequence ef433ac proved; this gate
   re-validates it routed through the importable module (one implementation,
   no drift between test and production logic).

The sharded apply is injected from the DELIVERABLE-1 JIT binding
(sharded_optimizer_binding.sharded_opt_step) — the dependency-injection seam
the production _ops entry later fills.

Run (GPU box):
    PYTHONPATH=. python -m pytest tests/hw/test_distributed_step.py -q -s
    # worker (spawned internally):
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 \
        tests/hw/test_distributed_step.py --worker --opt adamw
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pytest

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

_THIS = os.path.abspath(__file__)
_REPO = os.path.abspath(os.path.join(os.path.dirname(_THIS), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Parity tol for GATE-2 (the 2-step DP chain vs the single-GPU plain 2-step). The
# batch-shard grad reassociation (two half-batch fp32 sums vs one ascending-CTA
# full-batch sum) is ~2e-5 at ONE step (measured by the DP2 1-step loopback gate);
# across TWO AdamW steps it compounds through the 2nd-moment EMA to ~3.4e-4. The
# cross-rank A/A/A bit-eq holds bitwise (the distributed path is deterministic) —
# only the comparison against a DIFFERENT reduction order drifts, which this tol
# absorbs. 5e-4 = ~1.5× the measured 2-step delta, the same headroom the 1-step
# gate's 3e-5 gives over its 2.2e-5. (The 1-step DP2 gate keeps its own 3e-5.)
_PARITY_TOL = 5e-4
_OPTID = {"adamw": 0, "lion": 1, "grokfast": 2}


def _sm90a_available() -> bool:
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


def _build_cell(seed=1234):
    import grokking_race_v2 as g
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": "decoder", "use_amp": False, "use_fused": True,
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
    if opt == "grokfast":
        return g.Grokfast(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                          weight_decay=c["weight_decay"],
                          grokfast_alpha=c.get("grokfast_alpha", 0.98),
                          grokfast_lamb=c.get("grokfast_lamb", 2.0))
    raise ValueError(opt)


def _flat(named, total, dev):
    out = torch.empty(total, device=dev)
    off = 0
    for _, p in named:
        n = p.numel()
        out[off:off + n].copy_(p.data.reshape(-1))
        off += n
    return out


def _sharded_apply():
    from tests.hw.test_sharded_optimizer import _build_binding
    return _build_binding().sharded_opt_step


# ════════════════════════════════ GATE 1: world=1 identity ═══════════════════


def run_world1_identity(opt, steps=2, verbose=True):
    from grokking_optimizers.dispatch import fused_train_step, canonicalize_model
    from grokking_optimizers.parallel.distributed_step import (
        fused_train_step_distributed, make_dist_step_context)
    canon = canonicalize_model("decoder")
    apply_fn = _sharded_apply()

    # plain path (reference).
    g, c, m_ref, data, dev = _build_cell()
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named_ref)
    # 16*world rounding so both paths see the same rows (module shards with
    # world=1 → Bw = B - B%16, the dispatch-internal truncation made explicit).
    B = int(data[1].numel())
    Bw = B - (B % 16)
    tokens, targets = data[0][:Bw].contiguous(), data[1][:Bw].contiguous()
    opt_ref = _make_optimizer(opt, g, m_ref.parameters(), c)
    cache_ref = {}
    ref_params = []
    for s in range(1, steps + 1):
        fused_train_step(canon, opt, m_ref, opt_ref, tokens, targets,
                         state_cache=cache_ref, step=s, gemm_impl="wgmma")
        ref_params.append(_flat(named_ref, total, dev))

    # decomposed-at-world-1 path through the MODULE (fresh same-seed cell).
    g2, c2, m, data2, _ = _build_cell()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    named_sizes = [(n, p.numel()) for n, p in named]
    init = _flat(named, total, dev)
    ctx = make_dist_step_context(named_sizes, 1, 0, opt, init)
    opt_obj = _make_optimizer(opt, g2, m.parameters(), c2)
    cache = {}
    dec_params = []
    for s in range(1, steps + 1):
        _, full = fused_train_step_distributed(
            canon, opt, m, opt_obj, tokens, targets, ctx=ctx,
            sharded_apply=apply_fn, state_cache=cache, step=s,
            gemm_impl="wgmma", decompose_at_world1=True)
        dec_params.append(full.clone())

    eqs = [bool(torch.equal(a, b)) for a, b in zip(dec_params, ref_params)]
    ok = all(eqs)
    if verbose:
        maxds = [(a - b).abs().max().item() for a, b in zip(dec_params, ref_params)]
        print(f"  [dist-step world=1/{opt}] per-step bit-eq={eqs} maxd={maxds}",
              flush=True)
    return ok, eqs


@pytest.mark.skipif(not _sm90a_available(),
                    reason="distributed-step gate needs an sm_90 GPU")
@pytest.mark.parametrize("opt", ["adamw", "grokfast"])
def test_world1_decomposed_identity(opt):
    """world=1 forced decomposition == plain fused step, BIT-IDENTICAL over 2
    steps (design §7.1 through the production-shaped §6.2 module)."""
    ok, eqs = run_world1_identity(opt, verbose=True)
    assert ok, (
        f"world=1 decomposed path diverged from the plain fused step "
        f"(per-step eq={eqs}) — the §6.2 orchestration changed the numerics.")


# ════════════════════════ GATE 2: world=2 loopback (torchrun) ════════════════


def worker_main(opt):
    import torch.distributed as dist
    from grokking_optimizers.dispatch import fused_train_step, canonicalize_model
    from grokking_optimizers.parallel.distributed_step import (
        fused_train_step_distributed, make_dist_step_context)
    rank = int(os.environ["RANK"])
    os.environ["NCCL_HOSTID"] = f"sg-dist-step-rank-{rank}"          # ef433ac trick
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl")
    world = dist.get_world_size()
    assert world == 2
    canon = canonicalize_model("decoder")
    apply_fn = _sharded_apply()

    g, c, m, data, dev = _build_cell()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    named_sizes = [(n, p.numel()) for n, p in named]
    total = sum(n for _, n in named_sizes)
    init = _flat(named, total, dev)
    B = int(data[1].numel())
    Bw = B - (B % (16 * world))
    tokens, targets = data[0][:Bw].contiguous(), data[1][:Bw].contiguous()

    ctx = make_dist_step_context(named_sizes, world, rank, opt, init)
    opt_obj = _make_optimizer(opt, g, m.parameters(), c)
    cache = {}
    fulls = []
    for s in (1, 2):
        _, full = fused_train_step_distributed(
            canon, opt, m, opt_obj, tokens, targets, ctx=ctx,
            sharded_apply=apply_fn, state_cache=cache, step=s, gemm_impl="wgmma")
        fulls.append(full.clone())

    # (a) cross-rank identity per step (gather rank0/rank1 copies and compare).
    cross_ok = True
    for full in fulls:
        gathered = torch.empty(total * world, device=dev)
        torch.cuda.synchronize()
        dist.all_gather_into_tensor(gathered, full.contiguous())
        torch.cuda.synchronize()
        cross_ok = cross_ok and bool(
            torch.equal(gathered[0:total], gathered[total:2 * total]))

    # (b) vs the single-GPU plain 2-step reference on the SAME Bw rows — BOTH
    # ranks compute it (symmetric device timelines, the loopback lockstep rule).
    g2, c2, m_ref, _, _ = _build_cell()
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    opt_ref = _make_optimizer(opt, g2, m_ref.parameters(), c2)
    cache_ref = {}
    for s in (1, 2):
        fused_train_step(canon, opt, m_ref, opt_ref, tokens, targets,
                         state_cache=cache_ref, step=s, gemm_impl="wgmma")
    ref = _flat(named_ref, total, dev)
    rel = (fulls[-1] - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    parity = rel < _PARITY_TOL

    local_ok = 1 if (cross_ok and parity) else 0
    t = torch.tensor([local_ok], dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    dist.all_reduce(t)
    all_ok = int(t.item()) == world
    if rank == 0:
        print(f"[dist-step world=2 rank0] {opt}: cross-rank bit-eq={cross_ok} | "
              f"vs plain 2-step rel={rel:.3e} (tol {_PARITY_TOL:.0e}) "
              f"parity={parity}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if all_ok else 1


def _torchrun_available():
    import shutil
    return shutil.which("torchrun") is not None


def _skip2():
    if not _sm90a_available():
        return "needs an sm_90 GPU"
    if not _torchrun_available():
        return "torchrun not on PATH"
    return None


@pytest.mark.skipif(_skip2() is not None, reason=_skip2() or "")
def test_world2_loopback_through_module():
    """torchrun --nproc_per_node=2 (both cuda:0): the §6.2 module's 2-step DP
    chain — cross-rank identical bitwise + within parity tol of the plain
    single-GPU chain (the ef433ac flow, promoted into ONE implementation)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    env["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    cmd = ["torchrun", "--nproc_per_node=2", "--master_port=29641",
           _THIS, "--worker", "--opt", "adamw"]
    proc = subprocess.run(cmd, env=env, cwd=_REPO, capture_output=True,
                          text=True, timeout=1200)
    tail = "\n".join(proc.stdout.splitlines()[-20:])
    print(f"\n--- torchrun dist-step (rc={proc.returncode}) ---\n{tail}")
    if proc.returncode != 0:
        err = "\n".join(proc.stderr.splitlines()[-40:])
        raise AssertionError(
            f"world=2 module worker failed (rc={proc.returncode}).\n"
            f"--- stdout ---\n{tail}\n--- stderr ---\n{err}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--opt", default="adamw", choices=list(_OPTID))
    args = ap.parse_args(argv)
    if args.worker:
        return worker_main(args.opt)
    if not _sm90a_available():
        print("[dist-step] SKIP: no sm_90 GPU")
        return 0
    ok1, _ = run_world1_identity(args.opt, verbose=True)
    print(f"[dist-step] world=1 identity: {'OK' if ok1 else 'FAIL'}")
    return 0 if ok1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
