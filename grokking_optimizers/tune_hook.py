"""grokking_optimizers/tune_hook.py — the grokking implementation of compile.py's
portable autotuner ``tune_hook`` provider.

WHAT THIS IS
------------
compile.py (the portable AOT-then-JIT autotuner) tunes kernel *variants* by
building one ``.so`` per candidate config and then, in a FRESH SUBPROCESS per
built ``.so``, calling a project-supplied provider::

    def run(*, so_path: str, model: str, optimizer: str, arch: str,
            regime: str, seed: int) -> dict

and expecting it to return::

    {"output": <np.ndarray, float>, "elapsed_ms": <float>}

compile.py saves ``output`` to a ``.npy`` and compares the variant build's
output against a STRICT-MATH (fast-math-OFF) build's output for the SAME
``(model, optimizer, arch, regime, seed)`` — that is the numerical-correctness
gate. ``elapsed_ms`` is the perf signal the search optimizes. So a provider
``run`` must, deterministically given its arguments:

  1. Load ``so_path`` as THIS project's CUDA extension and make
     ``grokking_optimizers._ops`` resolve to it — so the production dispatch
     (``grokking_optimizers.dispatch.fused_train_step``) drives THIS variant
     binary, not the installed ``_ops*.so``. We use the proven idiom from
     compile.py's ``_WORKER_BODY._load_so`` / scripts/time_cell.py: derive the
     module name from the filename (everything before the first '.'), evict any
     stale ``grokking_optimizers._ops`` (+ submodules) from ``sys.modules``,
     ``exec_module`` the ``.so``, and alias it as ``sys.modules
     ['grokking_optimizers._ops']`` BEFORE dispatch resolves it.

  2. Build a DETERMINISTIC canonical workload for ``(model, optimizer)`` seeded
     by ``(regime, seed)`` — REUSING the L3-TC tail gate's exact cell
     construction (``tests/hw/test_l3tc_tail_gate._build_cell`` +
     ``_CELLS[f"{optimizer}/{model}"]["factory"]``), exactly as
     scripts/time_cell.py does. supergrok2 has no ``_CELLS`` factory, so it is
     built via ``tests/hw/_sg2_l3tc_gate`` (``_build`` + ``_sg2_factory``).

  3. Run the production L3-TC path ``fused_train_step(canon_model, optimizer,
     module, opt_obj, tokens, targets, state_cache={}, step=1,
     gemm_impl="wgmma")`` for a few warmup steps, then time N steps with CUDA
     Events (median ms/step) — mirroring scripts/time_cell.py::time_cell.

  4. Return ``output`` = the concatenated UPDATED params after one production
     step (``torch.cat([p.data.reshape(-1) for _,p in named_params if
     p.requires_grad]).float().cpu().numpy()``) — the representative post-step
     tensor that captures kernel correctness (this is what compile.py compares
     strict-vs-variant), and ``elapsed_ms`` = the median ms/step.

REGIME / SEED MAPPING
---------------------
The whole input distribution of a grokking cell (the synthesized grad/params)
is a deterministic function of ONE integer seed: ``_build_cell(model, seed)``
draws the data via ``make_data_for_task(c, seed)`` and reseeds the global RNG
(``torch.manual_seed(seed)``) right before ``build_model``, so a given seed
yields byte-identical data AND init weights. We therefore realize compile.py's
input regimes (``normal`` / ``large`` / ``small`` / ``adversarial`` — see
compile.py ``_INPUT_REGIMES``) as deterministic, fixed PER-REGIME SEED OFFSETS
on top of the caller's ``seed``:

    normal -> seed            large -> seed + 101
    small  -> seed + 202      adversarial -> seed + 303
    (any other/unknown regime -> a stable hash-derived offset)

This is intentionally simple and fully deterministic: because the effective
seed is a pure function of ``(regime, seed)``, the strict-math build and every
variant build synthesize the IDENTICAL workload for the SAME regime, so their
post-step ``output`` tensors are directly comparable (the ONLY difference is
the kernel's numerics) — which is exactly what compile.py's oracle compare
needs. (We do not perturb scale/dtype: a different per-regime seed already
moves the grad/param magnitudes deterministically, and keeping params fp32 is a
hard requirement of the L3-REAL fused path.)

DETERMINISM / ROBUSTNESS
------------------------
``run`` sets ``torch.manual_seed`` / ``torch.cuda.manual_seed_all`` and
cudnn-deterministic, and re-pins ``GATE_SEED`` so any gate-internal rebuild
agrees with our effective seed. There is NO silent fallback: if a
``(model, optimizer)`` cannot be driven (unknown cell, not L3-REAL, missing
factory, no GPU/extension), ``run`` raises with a clear message. torch / numpy /
grokking_optimizers are imported LAZILY inside ``run`` so this module imports
with no CUDA present.
"""
from __future__ import annotations

import importlib.util
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict

# Repo root (…/SuperGrok1.5) — two levels up from this file
# (grokking_optimizers/tune_hook.py).
_ROOT = Path(__file__).resolve().parents[1]

# compile.py's input regimes -> deterministic seed offsets. "normal" is the
# caller's seed unchanged; the others are fixed, distinct offsets so each regime
# is a different-but-reproducible workload. Mirrors compile.py._INPUT_REGIMES.
_REGIME_SEED_OFFSET: Dict[str, int] = {
    "normal": 0,
    "large": 101,
    "small": 202,
    "adversarial": 303,
}

# Timing knobs — same shape as scripts/time_cell.py defaults (warmup un-timed
# steps, then `iters` timed steps repeated `reps` times; median per-step ms).
_WARMUP = 8
_ITERS = 20
_REPS = 5


def _effective_seed(regime: str, seed: int) -> int:
    """Deterministic effective seed for (regime, seed). Known regimes use their
    fixed offset; an unknown regime gets a stable, collision-resistant offset
    derived from its name (so a new compile.py regime still maps reproducibly
    instead of silently colliding with "normal")."""
    if regime in _REGIME_SEED_OFFSET:
        off = _REGIME_SEED_OFFSET[regime]
    else:
        # Stable per-name offset in a high band so it can't collide with the
        # known offsets above; not RNG, so it's reproducible across processes.
        off = 1000 + (sum(ord(ch) for ch in str(regime)) % 9000)
    return int(seed) + off


def _alias_variant_ops(so_path: str) -> Any:
    """Load ``so_path`` and make ``grokking_optimizers._ops`` resolve to it.

    Proven idiom (compile.py `_WORKER_BODY._load_so` / scripts/time_cell.py):
    an extension .so exports ``PyInit_<name-it-was-BUILT-as>`` and
    ExtensionFileLoader looks up ``PyInit_<last dot-component of the spec
    name>`` — so the spec/module name MUST be the FILENAME's pre-first-dot stem
    (``_ops.cpython-311-….so`` -> ``_ops``; a JIT variant
    ``grokking_compiled_<opt>_<model>_<arch>_<cfg>.so`` -> that variant name).
    We evict the package + all its submodules first (dispatch._LazyOps memoizes
    the resolved _ops module AND per-attribute kernel fns, and optimizers bind()
    kernel callables onto instances), so a stale import can't make us silently
    drive the wrong binary. We DON'T import grokking_optimizers before this — the
    alias must be in sys.modules before dispatch's lazy loader resolves _ops."""
    sop = str(so_path)
    if not os.path.exists(sop):
        raise FileNotFoundError(f"tune_hook.run: so_path does not exist: {sop!r}")
    mod_name = os.path.basename(sop).split(".", 1)[0]
    # Evict the package and every submodule so nothing keeps a handle to a
    # previously-resolved _ops (this provider runs in a fresh subprocess per .so,
    # but be defensive — a parent might have imported grokking_optimizers).
    for _k in [k for k in list(sys.modules)
               if k == "grokking_optimizers" or k.startswith("grokking_optimizers.")]:
        del sys.modules[_k]
    spec = importlib.util.spec_from_file_location(mod_name, sop)
    if spec is None or spec.loader is None:
        raise ImportError(f"tune_hook.run: could not create import spec for {sop!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["grokking_optimizers._ops"] = mod
    return mod


def _ensure_root_on_path() -> None:
    """tests.hw.* and grokking_race_v2 both need the repo root importable."""
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def run(*, so_path: str, model: str, optimizer: str, arch: str,
        regime: str, seed: int) -> Dict[str, Any]:
    """Drive the production L3-TC step for ``(model, optimizer)`` against the
    variant extension at ``so_path`` and return its post-step params + median
    step time. See the module docstring for the full contract.

    Args:
        so_path:   path to the variant CUDA extension .so to time/validate.
        model:     "decoder" | "vit" | "mamba".
        optimizer: one of the 11 race optimizer keys (adamw, grokadamw,
                   grokfast, lion, looksam, muon, neuralgrok, prodigy,
                   supergrok11, supergrok15, supergrok2).
        arch:      target arch token (e.g. "sm_90a") — accepted for the portable
                   contract; the workload itself is arch-independent (the .so
                   IS the arch-specialized artifact).
        regime:    input distribution selector (see _REGIME_SEED_OFFSET).
        seed:      base seed; the effective seed is _effective_seed(regime, seed).

    Returns:
        {"output": np.ndarray (float32, flat post-step params),
         "elapsed_ms": float (median ms/step)}.
    """
    del arch  # the .so is already the arch-specialized artifact; recorded by caller.

    # 1) Alias the variant .so as grokking_optimizers._ops BEFORE anything imports
    #    the package, so dispatch.fused_train_step drives THIS binary.
    _ensure_root_on_path()
    _alias_variant_ops(so_path)

    # 2) Lazy imports (this module must import without CUDA). grokking_optimizers
    #    now binds the aliased variant _ops on first kernel access.
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "tune_hook.run: torch.cuda.is_available() is False — the L3-TC "
            "production path needs a CUDA device.")

    from grokking_optimizers.dispatch import (  # noqa: WPS433 (lazy by contract)
        fused_train_step, canonicalize_model, get_ops, has_l3_real,
        gemm_impl_for_cell)

    # The aliased variant must actually expose the fused kernels — otherwise we'd
    # be timing nothing / the wrong artifact. Fail loud (no silent fallback).
    ops = get_ops()
    if not hasattr(ops, "fused_step"):
        raise RuntimeError(
            f"tune_hook.run: the loaded extension at {so_path!r} does not expose "
            f"ops.fused_step — it is not a usable grokking_optimizers kernel build.")

    eff_seed = _effective_seed(regime, int(seed))

    # 3) Determinism: pin seeds + cudnn-deterministic, and re-pin GATE_SEED so any
    #    gate-internal rebuild (none on our happy path, but be consistent) agrees.
    os.environ["GATE_SEED"] = str(eff_seed)
    torch.manual_seed(eff_seed)
    torch.cuda.manual_seed_all(eff_seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001 — cudnn may be unavailable; not fatal
        pass

    canon = canonicalize_model(model)

    # 4) Build the DETERMINISTIC canonical workload by REUSING the L3-TC tail
    #    gate's exact construction (model + config + data + the production
    #    optimizer), seeded by the effective seed — exactly as time_cell.py does.
    if optimizer == "supergrok2":
        # supergrok2 has no factory in _CELLS — build it via the dedicated SG2
        # gate helpers (_build + _sg2_factory), which carry the gate-friendly
        # production hyperparameters (warmup_steps=0, nonzero meta_rescale,
        # sam_freq_min=1) and place the meta-net on the param device.
        try:
            import tests.hw._sg2_l3tc_gate as SG2  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"tune_hook.run: cannot import the supergrok2 gate helpers "
                f"(tests/hw/_sg2_l3tc_gate.py) needed to build the supergrok2 "
                f"workload: {exc}") from exc
        g, c, module, data, dev = SG2._build(model, seed=eff_seed)
        opt_obj = SG2._sg2_factory(g, module.parameters(), c)
    else:
        try:
            import tests.hw.test_l3tc_tail_gate as G  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"tune_hook.run: cannot import the L3-TC tail gate "
                f"(tests/hw/test_l3tc_tail_gate.py) needed to build the workload: "
                f"{exc}") from exc
        cell_key = f"{optimizer}/{model}"
        if cell_key not in G._CELLS:
            raise NotImplementedError(
                f"tune_hook.run: no canonical workload for cell {cell_key!r}. "
                f"Known cells: {sorted(G._CELLS)}.")
        spec = G._CELLS[cell_key]
        factory = spec.get("factory")
        if factory is None:
            # Defensive: every non-SG2 _CELLS entry should carry a factory; SG2 is
            # handled above. If a future None-factory cell appears, surface it.
            raise NotImplementedError(
                f"tune_hook.run: cell {cell_key!r} has no factory in _CELLS "
                f"(only supergrok2 is expected to be factory-less, and it is "
                f"handled via the SG2 gate). Cannot drive it.")
        g, c, module, data, dev = G._build_cell(model, seed=eff_seed)
        opt_obj = factory(g, module.parameters(), c)

    tx, ty = data[0], data[1]

    # Precondition: the cell must actually be wired L3-TC on the wgmma engine —
    # otherwise fused_train_step would throw and the timing would be meaningless.
    if not has_l3_real(canon, optimizer):
        raise RuntimeError(
            f"tune_hook.run: cell ({canon}, {optimizer}) is not L3-REAL — the "
            f"production fused_train_step path cannot drive it.")
    eng = gemm_impl_for_cell(canon, optimizer, "bf16")
    if eng != "wgmma":
        raise RuntimeError(
            f"tune_hook.run: cell ({canon}, {optimizer}) bf16 engine is {eng!r}, "
            f"expected 'wgmma' (the L3-TC tensor-core path the autotuner times).")

    # A single persistent state cache across warmup + timed steps (the megakernel
    # owns the flat param + [m|v|extra]+loss buffers in it; reallocating per step
    # would reset momentum). This mirrors scripts/time_cell.py exactly.
    state_cache: Dict[str, Any] = {}

    def step() -> None:
        # Force the production L3-TC race path (gemm_impl="wgmma"); for the
        # SAM-coupled cells (looksam / supergrok11 / supergrok15 / supergrok2) the
        # factory pins the SAM-on settings so the double tile fwd+bwd runs every
        # timed step — exactly what scripts/time_cell.py exercises.
        fused_train_step(canon, optimizer, module, opt_obj, tx, ty,
                         state_cache=state_cache, step=1, gemm_impl="wgmma")

    # 5) Warmup (un-timed), then time N steps with CUDA Events; median over reps.
    for _ in range(_WARMUP):
        step()
    torch.cuda.synchronize()
    walls = []
    for _ in range(_REPS):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(_ITERS):
            step()
        ev1.record()
        torch.cuda.synchronize()
        walls.append(ev0.elapsed_time(ev1) / _ITERS)  # ms/step
    median_ms = float(statistics.median(walls))

    # 6) output = the concatenated UPDATED params after the (last) production step —
    #    the representative post-step tensor compile.py compares strict-vs-variant.
    #    The in-place megakernel has scattered the new weights back into p.data, so
    #    named_parameters() carries the post-step values (named order == flat layout).
    params_np = (torch.cat([p.data.reshape(-1)
                            for _, p in module.named_parameters() if p.requires_grad])
                 .float().cpu().numpy().astype(np.float32))

    return {"output": params_np, "elapsed_ms": median_ms}
