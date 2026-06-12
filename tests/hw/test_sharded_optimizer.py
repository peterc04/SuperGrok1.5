"""tests/hw/test_sharded_optimizer.py — Phase-2 sharded-optimizer bit-parity gate.

IMPLEMENTS /workspace/.parallelism_design.md §7.3 (DELIVERABLE 1, task #25): at
DP=1 the new ZeRO-2/3 ``sharded_optimizer_kernel<Opt>`` over the FULL param set
must produce **BIT-IDENTICAL** params to the in-kernel P3 fused tail
(``apply_optimizer<Opt>`` work-steal loop) on the SAME ``(params, grad, state)``.
This proves the B2-seam cut (design §2.2 — the ``if constexpr (Par::kShardOptGrad)
return;`` that ends the fused kernel at "local grad assembled" so a SEPARATE
sharded-opt kernel consumes the reduce-scattered grad) is correctness-preserving
PER OPTIMIZER, entirely on one GPU.

THE EQUIVALENT CHECK (design §7.3 / the task's harvesting note). Harvesting the
pre-P3 (params, grad, state) through the public ABI is awkward (the fused step is
ONE launch that runs P0..P3), so we use the equivalence the task spells out:

  1. Run ONE production fused step (in-kernel P3) on a freshly-built cell, with
     ``return_grad=True`` → ``params_after`` (scattered back into the model) AND
     the deterministically-reduced ``grad`` the P3 tail consumed (the dispatch.py
     fused path returns ``grad_out``). Snapshot ``params_before`` first.
  2. Re-run the SAME optimizer step routed through the sharded kernel: feed
     ``params_before`` + the captured ``grad`` + a ZERO-init ``[m|v|extra]`` state
     (the production state cache zero-inits at step 1) + the SAME scalars
     (``_opt_scalars_from`` — the single source the production path uses) into
     ``sharded_optimizer_kernel<Opt>`` over the whole flat ``[0,total)``.
  3. Assert ``params_sharded`` is BITWISE-equal to ``params_after`` (and the
     updated m/v state matches the kernel's state slices bit-for-bit).

WHY IT IS BIT-IDENTICAL, NOT MERELY ULP-EQUAL (so the assert is ``torch.equal``,
not a tol): the sharded kernel includes ``opt_components.cuh`` VERBATIM and calls
the EXACT SAME ``apply_optimizer<Opt>`` device function the P3 tail calls. The
only structural difference is the iteration shape — P3 work-steals the 30 tensors
and ``rebase_state``-offsets the per-element pointers per tensor; the sharded
kernel is a flat grid-stride over ``[0,total)`` with the un-rebased flat slices.
But ``rebase_state`` only ADDS the tensor offset to the per-element pointers
(``exp_avg``/``exp_avg_sq``/``ema``) and passes every scalar through unchanged
(fused_decoder_megakernel.cuh:147), so element ``i`` always pairs ``params[i]`` ↔
``grad[i]`` ↔ ``m[i]`` ↔ ``v[i]`` ↔ ``ema[i]`` with the IDENTICAL scalars on both
paths. Same inputs + same device op (same ``--use_fast_math``/fp32 contraction) ⇒
same bits. The grokfast cold-start (``if (step==1) ema[idx]=grad[idx]``) is in
``apply_optimizer`` too, so it replicates exactly. MEASURED: maxd == 0.0 for every
cell below (no loosening was needed — see the run output).

SCOPE — the ELEMENTWISE OptIds (design §2.3 / §0.4): ``{adamw, lion, grokfast}``.
These are the cells whose sharded-opt kernel is the flat grid-stride (their
``apply_optimizer<Opt>`` is purely per-element). The per-TENSOR cells
(muon/SG11/SG15/SG2) and the global-scalar cells (prodigy/grokadamw/looksam) need
an UPSTREAM stage / all-reduce before the flat apply (design §2.3/§2.4), out of
scope for this flat-kernel gate; their sharded path is the *full* persistent
kernel restricted to the owned tensors, validated separately. We run the three
elementwise ids across ALL THREE models (decoder/vit/mamba) so the cut is gated
on every L3-TC fwd+bwd substrate.

JIT BUILD (task #25): the sharded kernel is bound via
``torch.utils.cpp_extension.load`` into its OWN build dir under
``/workspace/.torch_ext`` — it does NOT touch the production ``_ops`` build. The
binding TU (tests/hw/sharded_optimizer_binding.cu) owns its own
``PYBIND11_MODULE`` and is excluded from the setup.py glob by
``_owns_extension_module``.

Run:
    PYTHONPATH=. python -m pytest tests/hw/test_sharded_optimizer.py -q -s
    PYTHONPATH=. python tests/hw/test_sharded_optimizer.py
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
_BINDING_CU = os.path.join(_REPO, "tests", "hw", "sharded_optimizer_binding.cu")
# Own JIT build dir — NEVER the production _ops build (task #25 mandate).
_TORCH_EXT_DIR = os.environ.get("SG_TORCH_EXT_DIR", "/workspace/.torch_ext")
_BUILD_DIR = os.path.join(_TORCH_EXT_DIR, "sharded_opt_bind")


def _sm90a_available() -> bool:
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


# hw + skipif gate — mirrors test_decoder_tc / test_l3tc_tail_gate discipline.
_GATE = pytest.mark.skipif(
    not _sm90a_available(),
    reason="sharded-optimizer bit-parity gate needs an sm_90 (Hopper) GPU + bf16")

# OptId integers (mirror opt_components.cuh::OptId) for the elementwise ids.
_OPTID = {"adamw": 0, "lion": 1, "grokfast": 2}

# (optimizer, model) cells — the 3 elementwise ids × the 3 L3-TC models. All run
# on the wgmma (bf16 tensor-core) production path (the only L3-TC engine).
_CELLS = [
    (opt, model)
    for model in ("decoder", "vit", "mamba")
    for opt in ("adamw", "lion", "grokfast")
]

_MODULE = None


def _build_binding():
    """JIT-build the sharded-opt binding into /workspace/.torch_ext (NOT _ops)."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.makedirs(_BUILD_DIR, exist_ok=True)
    from torch.utils.cpp_extension import load
    # The SAME flags the production build uses for the TC TUs (setup.py): the math
    # must compile identically (--use_fast_math, -DWITH_CUDA, --expt-relaxed-
    # constexpr) or apply_optimizer<Opt> could round differently and break bit-
    # parity. Repo-root include for the "csrc/..."-relative #includes.
    _MODULE = load(
        name="sharded_opt_bind",
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
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    import grokking_race_v2 as g
    return g


def _build_cell(model, seed=42):
    """Build the live race model + one real batch (the SAME recipe the tail gate
    uses). Reseeds immediately before build_model so the init is byte-stable."""
    g = _g()
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": model, "use_amp": False, "use_fused": True,
              "compile_model": False, "matmul_precision": "bf16", "seed": seed,
              "frac_train": 0.5, "val_ratio": 0.10, "p": 97})
    dev = torch.device("cuda")
    data = tuple(d.to(dev) for d in g.make_data_for_task(c, seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = g.build_model(c, dev)
    return g, c, m, data, dev


def _make_optimizer(opt, g, params, c):
    """The REAL production optimizer with the EXACT race hyperparameters (the
    source of the scalars _opt_scalars_from feeds the kernel)."""
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


def _flat_params(named, total, dev):
    out = torch.empty(total, device=dev)
    off = 0
    for _, p in named:
        n = p.numel()
        out[off:off + n].copy_(p.data.reshape(-1))
        off += n
    return out


def run_sharded_parity(opt, model, verbose=True):
    """Run the DP=1 sharded-opt-vs-P3 bit-parity check for one cell.

    Returns (ok, detail). ok is True iff params AND m/v state are BITWISE-equal.
    """
    from grokking_optimizers.dispatch import (fused_train_step, canonicalize_model,
                                              has_l3_real, gemm_impl_for_cell,
                                              _opt_scalars_from)
    mod = _build_binding()
    g, c, m, data, dev = _build_cell(model)
    canon = canonicalize_model(model)
    # Precondition: the cell must really be wired L3-TC wgmma — else the gate is
    # comparing against the wrong tail.
    assert has_l3_real(canon, opt), f"{opt}/{model}: not L3-REAL (wire it first)"
    eng = gemm_impl_for_cell(canon, opt, "bf16")
    assert eng == "wgmma", f"{opt}/{model}: bf16 engine is {eng!r}, expected wgmma"

    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named)
    params_before = _flat_params(named, total, dev)

    # ── (1) ONE production fused step (in-kernel P3), harvest grad + params_after.
    opt_obj = _make_optimizer(opt, g, m.parameters(), c)
    cache = {}
    loss, grad = fused_train_step(canon, opt, m, opt_obj, data[0], data[1],
                                  state_cache=cache, step=1, return_grad=True,
                                  gemm_impl="wgmma")
    params_after = _flat_params(named, total, dev)
    kstate = cache[canon]["state"]            # the kernel's [m|v|extra]+loss buffer

    # ── (2) re-run the SAME step through the sharded kernel on the captured grad.
    #    ZERO-init [m|v|extra] state (matches the production cache cold-start);
    #    SAME scalars via _opt_scalars_from (the single source dispatch.py uses).
    sc = _opt_scalars_from(opt_obj, 1)
    params_sharded = params_before.clone()
    grad_in = grad.clone()                    # also lets us assert grad is NOT mutated
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    mod.sharded_opt_step(
        _OPTID[opt], params_sharded, grad_in, state,
        sc["lr"], sc["beta1"], sc["beta2"], sc["eps"], sc["weight_decay"],
        sc["bc1"], sc["bc2"], sc.get("alpha", 0.98), sc.get("lamb", 2.0), 1)

    # ── (3) BITWISE-equality asserts (no tol — the math is the same device op).
    param_eq = torch.equal(params_sharded, params_after)
    param_maxd = (params_sharded - params_after).abs().max().item()
    m_eq = torch.equal(state[0:total], kstate[0:total])
    v_eq = torch.equal(state[total:2 * total], kstate[total:2 * total])
    grad_unmutated = torch.equal(grad_in, grad)   # the sharded apply must NOT write grad
    ok = param_eq and m_eq and v_eq and grad_unmutated
    if verbose:
        print(f"  [{opt}/{model}] params bit-eq={param_eq} maxd={param_maxd:.3e} | "
              f"m_eq={m_eq} v_eq={v_eq} | grad_unmutated={grad_unmutated} | "
              f"loss={loss:.5f} total={total}", flush=True)
    return ok, dict(param_eq=param_eq, param_maxd=param_maxd, m_eq=m_eq, v_eq=v_eq,
                    grad_unmutated=grad_unmutated, loss=loss, total=total)


@_GATE
@pytest.mark.parametrize("opt,model", _CELLS)
def test_sharded_optimizer_bit_parity(opt, model):
    """DP=1: sharded_optimizer_kernel<Opt> == in-kernel P3 fused tail, BIT-EXACT
    (params + m/v state) on the SAME (params, grad, state). Design §7.3."""
    ok, d = run_sharded_parity(opt, model, verbose=True)
    assert d["param_eq"], (
        f"{opt}/{model}: sharded-opt params NOT bit-identical to the P3 tail "
        f"(maxd={d['param_maxd']:.3e}). The B2-seam cut changed the result — the "
        f"sharded apply must call the SAME apply_optimizer<{opt}> on the SAME "
        f"flat (params, grad, state).")
    assert d["m_eq"] and d["v_eq"], (
        f"{opt}/{model}: sharded-opt m/v state NOT bit-identical (m_eq={d['m_eq']} "
        f"v_eq={d['v_eq']}). The per-element moment update diverged.")
    assert d["grad_unmutated"], (
        f"{opt}/{model}: the sharded apply mutated grad (it must read-only consume "
        f"the reduce-scattered grad, like the P3 tail).")
    assert ok


# ─────────────────────────────────────────────────────────────── script main ──


def main() -> int:
    if not _sm90a_available():
        print("[sharded-opt] SKIP: no sm_90 GPU")
        return 0
    fails = []
    for opt, model in _CELLS:
        ok, d = run_sharded_parity(opt, model, verbose=True)
        if not ok:
            fails.append((opt, model, d))
    if fails:
        print(f"[sharded-opt] FAIL: {len(fails)} cell(s) not bit-identical: "
              f"{[(o, m) for o, m, _ in fails]}")
        return 1
    print(f"[sharded-opt] OK: {len(_CELLS)} cells bit-identical "
          f"(sharded_optimizer_kernel<Opt> == P3 fused tail, maxd=0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
