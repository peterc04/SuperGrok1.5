"""H3 dW-interleave A/A/A + parity probe (d=1024 bench layout).

Two checks on the d=1024 decoder TC cell:
  (A) DETERMINISM: 3 runs of one tc_train_step from the SAME inputs → params+state
      bit-identical (sha256). The dW move must not introduce any nondeterminism.
  (P) PARITY vs the pre-change BASELINE: rebuilds the UNMODIFIED header from a git
      checkout of the named baseline ref into a SEPARATE module, runs the same one
      step, and asserts the post-step (params,state,loss) is BIT-IDENTICAL. The
      M-atom interleave only reorders independent-fragment MMAs + restages the
      shared operand (ascending-k preserved per atom) → must be bitwise-exact.

Usage:
  python tuning/_h3_dw_aaa.py --baseline-ref c67cb66 [--d 1024 --B 16384]
"""
from __future__ import annotations
import argparse, hashlib, os, subprocess, sys, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

TU_REL = "csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu"


def _flags(d: int, profile: bool = False):
    f = ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
         "-gencode=arch=compute_90a,code=sm_90a",
         "-gencode=arch=compute_90a,code=compute_90a",
         "-DSG_TUNED_GEMM_IMPL=1"]
    if d != 128:
        f += ["-DSG_DEC_BENCH_LAYOUT=1", "-DSG_DEC_SCALAR_MEGAKERNEL=0"]
    return f


def _build(name, sources_root, d):
    """Build the TU rooted at `sources_root` (a checkout dir or the live tree)."""
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    tu = str(Path(sources_root) / TU_REL)
    bdir = Path(tempfile.gettempdir()) / f"build_{name}"
    bdir.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[tu],
                extra_include_paths=[str(sources_root)],
                extra_cuda_cflags=_flags(d),
                extra_cflags=["-O3", "-std=c++17"],
                build_directory=str(bdir),
                verbose=False)


def _one_step_hash(mod, d, B, seed=42):
    dev = torch.device("cuda")
    total = int(mod.TOTAL)
    vocab, seq = int(mod.VOCAB), int(mod.SEQ)
    assert int(mod.D) == d
    g = torch.Generator(device=dev).manual_seed(seed)
    params = (torch.randn(total, generator=g, device=dev) * 0.02).contiguous()
    tokens = torch.randint(0, vocab, (B, seq), generator=g, device=dev).to(torch.int32).contiguous()
    targets = torch.randint(0, vocab, (B,), generator=g, device=dev).to(torch.int32).contiguous()
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0
    bc1, bc2 = 1.0 - b1, 1.0 - b2
    loss = mod.tc_train_step(params, tokens, targets, state, lr, b1, b2, eps, wd, bc1, bc2, 1, 0)
    torch.cuda.synchronize()
    h = hashlib.sha256()
    h.update(params.detach().cpu().numpy().tobytes())
    h.update(state.detach().cpu().numpy().tobytes())
    # loss may come back as a python float, a tensor, or a 1-element list/tuple.
    def _scalar(x):
        if x is None: return float("nan")
        if isinstance(x, (list, tuple)): x = x[0]
        try: return float(x.item()) if hasattr(x, "item") else float(x)
        except Exception: return float("nan")
    return h.hexdigest(), _scalar(loss), params.clone(), state.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--B", type=int, default=16384)
    ap.add_argument("--baseline-ref", type=str, default="c67cb66")
    args = ap.parse_args()

    print(f"[h3-aaa] building LIVE (modified) d={args.d} ...", flush=True)
    mod = _build("dec_h3_live", str(ROOT), args.d)

    # (A) determinism A/A/A
    hashes = []
    for i in range(3):
        hx, lv, _, _ = _one_step_hash(mod, args.d, args.B)
        hashes.append(hx)
        print(f"  [A/A/A] run{i}: sha256={hx[:16]}  loss={lv:.8f}", flush=True)
    aaa_ok = len(set(hashes)) == 1
    print(f"  => A/A/A bit-identical: {'PASS' if aaa_ok else 'FAIL'}", flush=True)

    # (P) parity vs baseline ref (checkout the TU + its includes into a temp dir)
    print(f"\n[h3-aaa] checking out baseline ref {args.baseline_ref} for parity ...", flush=True)
    base_dir = Path(tempfile.mkdtemp(prefix="h3_base_"))
    # archive the whole tree at the ref (need the includes), extract to base_dir.
    arch = subprocess.run(["git", "archive", args.baseline_ref], cwd=str(ROOT),
                          capture_output=True)
    if arch.returncode != 0:
        print(f"  git archive failed: {arch.stderr.decode()[:200]}", flush=True)
        sys.exit(2)
    tar = base_dir / "tree.tar"
    tar.write_bytes(arch.stdout)
    subprocess.run(["tar", "xf", str(tar), "-C", str(base_dir)], check=True)
    print(f"  building BASELINE d={args.d} from {base_dir} ...", flush=True)
    base_mod = _build("dec_h3_base", str(base_dir), args.d)
    bh, blv, _, _ = _one_step_hash(base_mod, args.d, args.B)
    lh = hashes[0]
    print(f"  [parity] baseline sha256={bh[:16]}  loss={blv:.8f}", flush=True)
    print(f"  [parity] live     sha256={lh[:16]}", flush=True)
    par_ok = (bh == lh)
    print(f"  => PARITY (live == baseline {args.baseline_ref}) bit-identical: "
          f"{'PASS' if par_ok else 'FAIL'}", flush=True)
    shutil.rmtree(base_dir, ignore_errors=True)

    print(f"\n[h3-aaa] RESULT: A/A/A={'PASS' if aaa_ok else 'FAIL'}  "
          f"PARITY={'PASS' if par_ok else 'FAIL'}", flush=True)
    sys.exit(0 if (aaa_ok and par_ok) else 1)


if __name__ == "__main__":
    main()
