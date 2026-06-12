"""tests/hw/test_pp2_loopback_determinism.py — PP=2 single-GPU loopback A/A/A.

IMPLEMENTS /workspace/.parallelism_design.md §4 (PP stage cut) via §7 (task #25
phase-2, Lane C): BOTH stages of a PP=2 split of the decoder L3-TC megakernel
run on ONE GPU as sequential launches sharing one workspace — the virtual-
boundary discipline ef433ac used for DP=2, applied to the PIPELINE axis.

THE DECOMPOSED STEP (1F1B at mb=1 — the correctness configuration):
    [stage0 fwd]          → boundary acts X_in[1] (bf16 [T,d], zero-copy)
    [stage1 fwd+bwd]      → owned grads {tensors 14..29} + loss + dh (fp32 [T,d])
    [stage0 fwd-rec+bwd]  → owned grads {tensors 0..13}
    [sharded-opt kernel]  → the established B2-seam optimizer (DELIVERABLE 1)

WHY BIT-EXACT vs the single-launch fused step (the strong assert): the stage
kernels run the SAME tile partition (one CTA/SM, same nCTA), the SAME per-tile
math (the layer-range templates fold to the identical per-layer bodies), the
SAME P2 dW machinery (identical global work-item space + split-K slot bases,
non-owned items skipped), and the boundary adjoint crosses in fp32 (the fused
path never rounds it to bf16). The fwd-recompute in a bwd launch is
deterministic ⇒ reproduces the fwd launch's scratch bit-for-bit. So the
CONCATENATED PP grad must equal the production fused step's return_grad
BITWISE, and loss must match bitwise — asserted, not assumed.

THE ASSERTS:
  (a) PP grad == production fused-step grad, BIT-IDENTICAL (all 30 tensors;
      per-tensor breakdown printed on failure);
  (b) loss bit-identical;
  (c) A/A/A: 3 PP reruns bit-identical (grad + loss + dh handoff);
  (d) END-TO-END closure: sharded_optimizer_kernel (DELIVERABLE-1 binding) on
      (params_before, PP grad, zero state, production scalars) == the
      production step's params_after, BIT-IDENTICAL (PP stages + B2-seam opt
      == the full fused step).

PATCH GATE: requires .phase2/patches/0001-dectc-layer-range-pp.patch applied to
csrc/fused/sm_90/model_stage_decoder_tc.cuh (the SG_DECTC_LAYER_RANGE marker).
Unpatched tree ⇒ SKIP with the apply instruction (the JIT build would #error
loudly anyway — no silent degradation).

Run (GPU box, after applying the patch):
    git apply .phase2/patches/0001-dectc-layer-range-pp.patch
    PYTHONPATH=. python -m pytest tests/hw/test_pp2_loopback_determinism.py -q -s
    PYTHONPATH=. python tests/hw/test_pp2_loopback_determinism.py
"""
from __future__ import annotations

import os
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
_BINDING_CU = os.path.join(_REPO, "tests", "hw", "pp_stage_binding.cu")
_STAGE_HEADER = os.path.join(_REPO, "csrc", "fused", "sm_90", "model_stage_decoder_tc.cuh")
_TORCH_EXT_DIR = os.environ.get("SG_TORCH_EXT_DIR", "/workspace/.torch_ext")
_BUILD_DIR = os.path.join(_TORCH_EXT_DIR, "pp_stage_bind")


def _sm90a_available() -> bool:
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


def _patch_applied() -> bool:
    """The layer-range patch defines SG_DECTC_LAYER_RANGE in the stage header."""
    try:
        with open(_STAGE_HEADER) as f:
            return "SG_DECTC_LAYER_RANGE" in f.read()
    except OSError:
        return False


def _skip_reason():
    if not _sm90a_available():
        return "PP=2 loopback gate needs an sm_90 (Hopper) GPU"
    if not _patch_applied():
        return ("layer-range patch NOT applied — run `git apply "
                ".phase2/patches/0001-dectc-layer-range-pp.patch` first "
                "(see .phase2/RUNBOOK.md). Skipping loudly, not faking success.")
    return None


_MODULE = None


def _build_binding():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.makedirs(_BUILD_DIR, exist_ok=True)
    from torch.utils.cpp_extension import load
    # SAME flags as the production TC TUs — required for bit-parity of the math.
    _MODULE = load(
        name="pp_stage_bind",
        sources=[_BINDING_CU],
        extra_include_paths=[_REPO],
        build_directory=_BUILD_DIR,
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
            "-DWITH_CUDA",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-gencode=arch=compute_90a,code=compute_90a",
            "-lineinfo",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )
    return _MODULE


def _g():
    import grokking_race_v2 as g
    return g


def _build_cell(seed=42):
    """The live race decoder + one real batch (the ef433ac recipe)."""
    g = _g()
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": "decoder", "use_amp": False, "use_fused": True,
              "compile_model": False, "matmul_precision": "bf16", "seed": seed,
              "frac_train": 0.5, "val_ratio": 0.10, "p": 97})
    dev = torch.device("cuda")
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
    raise ValueError(f"unhandled opt {opt!r}")


_OPTID = {"adamw": 0, "lion": 1, "grokfast": 2}


def _flat_params(named, total, dev):
    out = torch.empty(total, device=dev)
    off = 0
    for _, p in named:
        n = p.numel()
        out[off:off + n].copy_(p.data.reshape(-1))
        off += n
    return out


def run_pp2_gate(opt, verbose=True):
    """One full PP=2-vs-fused gate for the decoder. Returns (ok, detail)."""
    from grokking_optimizers.dispatch import (fused_train_step, canonicalize_model,
                                              has_l3_real, gemm_impl_for_cell,
                                              _opt_scalars_from)
    from tests.hw.test_sharded_optimizer import _build_binding as _build_sharded
    mod = _build_binding()
    sharded = _build_sharded()
    g, c, m, data, dev = _build_cell()
    canon = canonicalize_model("decoder")
    assert has_l3_real(canon, opt), f"{opt}/decoder: not L3-REAL (wire it first)"
    assert gemm_impl_for_cell(canon, opt, "bf16") == "wgmma"

    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named)
    assert total == mod.TOTAL, f"layout drift: model total {total} != binding {mod.TOTAL}"

    # Truncate the batch to a multiple of 16 UP FRONT so the production step and
    # the PP step see the identical token rows (dispatch truncates internally).
    B = int(data[1].numel())
    Bw = B - (B % 16)
    tokens = data[0][:Bw].contiguous()
    targets = data[1][:Bw].contiguous()

    params_before = _flat_params(named, total, dev)

    # ── production single-launch fused step (in-kernel P3), harvest grad+after.
    opt_obj = _make_optimizer(opt, g, m.parameters(), c)
    cache = {}
    loss_ref, grad_ref = fused_train_step(canon, opt, m, opt_obj, tokens, targets,
                                          state_cache=cache, step=1,
                                          return_grad=True, gemm_impl="wgmma")
    params_after = _flat_params(named, total, dev)

    # ── PP=2 loopback ×3 (A/A/A): same flat params, same truncated batch.
    tok_flat = tokens.reshape(-1).to(torch.int32).contiguous()
    tgt_flat = targets.reshape(-1).to(torch.int32).contiguous()
    runs = []
    for _ in range(3):
        loss_pp, grad_pp, dh = mod.pp2_step(params_before, tok_flat, tgt_flat, 0)
        torch.cuda.synchronize()
        runs.append((loss_pp.clone(), grad_pp.clone(), dh.clone()))
    loss_pp, grad_pp, dh = runs[0]

    # (c) A/A/A bit-identical (grad + loss + the dh handoff itself).
    aaa = all(torch.equal(runs[0][k], r[k]) for r in runs[1:] for k in range(3))

    # (a) PP grad == production grad, BIT-IDENTICAL.
    grad_eq = torch.equal(grad_pp, grad_ref)
    # (b) loss bit-identical.
    loss_eq = bool(loss_pp.item() == loss_ref)

    per_tensor = None
    if not grad_eq:
        lay = mod.pp2_layout()
        per_tensor = []
        for t in range(30):
            off, sz = int(lay[2 * t]), int(lay[2 * t + 1])
            d = (grad_pp[off:off + sz] - grad_ref[off:off + sz]).abs().max().item()
            if d > 0:
                per_tensor.append((t, d))

    # (d) end-to-end closure: PP grad + DELIVERABLE-1 sharded opt == fused P3.
    sc = _opt_scalars_from(opt_obj, 1)
    params_pp = params_before.clone()
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    sharded.sharded_opt_step(
        _OPTID[opt], params_pp, grad_pp, state,
        sc["lr"], sc["beta1"], sc["beta2"], sc["eps"], sc["weight_decay"],
        sc["bc1"], sc["bc2"], sc.get("alpha", 0.98), sc.get("lamb", 2.0), 1)
    torch.cuda.synchronize()
    closure_eq = torch.equal(params_pp, params_after)

    ok = grad_eq and loss_eq and aaa and closure_eq
    if verbose:
        gmax = (grad_pp - grad_ref).abs().max().item()
        print(f"  [pp2/{opt}] (a) grad bit-eq={grad_eq} maxd={gmax:.3e} | "
              f"(b) loss bit-eq={loss_eq} ({loss_pp.item():.6f} vs {loss_ref:.6f}) | "
              f"(c) A/A/A={aaa} | (d) closure params bit-eq={closure_eq} | "
              f"Bw={Bw} total={total}", flush=True)
        if per_tensor:
            print(f"    non-bit-exact tensors (idx, maxd): {per_tensor}", flush=True)
    return ok, dict(grad_eq=grad_eq, loss_eq=loss_eq, aaa=aaa,
                    closure_eq=closure_eq, per_tensor=per_tensor)


_PP_CELLS = ["adamw", "grokfast"]

_REASON = None


def _gate():
    global _REASON
    if _REASON is None:
        _REASON = _skip_reason() or ""
    return pytest.mark.skipif(bool(_REASON), reason=_REASON)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
@pytest.mark.parametrize("opt", _PP_CELLS)
def test_pp2_loopback_bit_exact_aaa(opt):
    """PP=2 stage composition == single-launch fused step, BIT-EXACT (grad,
    loss, end-to-end params via the B2-seam sharded opt) + A/A/A. Design §4
    via §7 (the PP analogue of ef433ac's DP=2 gate)."""
    ok, d = run_pp2_gate(opt, verbose=True)
    assert d["grad_eq"], (
        f"pp2/{opt}: PP-composed grad NOT bit-identical to the fused step "
        f"(per-tensor deltas: {d['per_tensor']}). The stage cut changed the "
        f"math — check the dh fp32 handoff and the owned-spec filters.")
    assert d["loss_eq"], f"pp2/{opt}: loss differs (CE runs only on the last stage)"
    assert d["aaa"], f"pp2/{opt}: A/A/A failed — non-determinism in the stage kernels"
    assert d["closure_eq"], (
        f"pp2/{opt}: sharded-opt(PP grad) != fused P3 params. Grad matched, so "
        f"this is a DELIVERABLE-1 regression — rerun test_sharded_optimizer.py.")
    assert ok


def test_pp2_stage_ownership_matches_python_plan():
    """The kernel-side PPStageSpec tensor ownership == the Python pipeline
    plan's (single-source cross-check). Needs the JIT binding (GPU box) but no
    kernel launch."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    mod = _build_binding()
    from grokking_optimizers.parallel.pipeline import stage_tensor_ownership
    for stage in (0, 1):
        kernel_side = sorted(int(t) for t in mod.pp2_owned_tensors(stage))
        py_side = sorted(stage_tensor_ownership("decoder", 2)[stage])
        assert kernel_side == py_side, (
            f"stage {stage}: kernel ownership {kernel_side} != python plan {py_side}")
    # Disjoint + complete.
    s0 = set(mod.pp2_owned_tensors(0))
    s1 = set(mod.pp2_owned_tensors(1))
    assert not (s0 & s1) and (s0 | s1) == set(range(30))


def main() -> int:
    reason = _skip_reason()
    if reason:
        print(f"[pp2-loopback] SKIP: {reason}")
        return 0
    fails = []
    for opt in _PP_CELLS:
        ok, d = run_pp2_gate(opt, verbose=True)
        if not ok:
            fails.append((opt, d))
    if fails:
        print(f"[pp2-loopback] FAIL: {[o for o, _ in fails]}")
        return 1
    print(f"[pp2-loopback] OK: {_PP_CELLS} — PP=2 stage composition bit-identical "
          f"to the fused step (grad+loss+params) and A/A/A on one GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
