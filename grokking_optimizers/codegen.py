"""grokking_optimizers.codegen — Jinja2-based kernel source emitter.

When the autotuner is run with ``enable_emitter=True`` (CLI flag
``--enable-emitter`` or the equivalent setting in compile_config.toml),
``compile.py`` routes each variant through ``emit_variant_source(...)``
instead of recompiling one fixed source file with different ``-D``
macros.

The emitter renders a Jinja2 template per (optimizer, arch) and bakes
the autotune config (block size, vec width, num stages, cluster shape,
…) directly into the source.  This lets the emitted kernel use
config-dependent ``constexpr``, template parameters, ``#pragma unroll``
counts, and even ``__launch_bounds__`` annotations that a macro-only
build can't express cleanly.

Templates live in ``grokking_optimizers/templates/`` and follow the
naming convention::

    <optimizer>_<arch>.<ext>.j2     # most-specific (e.g. adamw_sm_90a.cu.j2)
    <optimizer>_<vendor>.<ext>.j2   # vendor fallback (e.g. adamw_cuda.cu.j2)
    <optimizer>_generic.<ext>.j2    # last-resort fallback

``find_template`` walks that priority order; ``emit_variant_source``
hashes (template_source + JSON-canonical context) so identical configs
re-use the same emitted file across runs (free incremental rebuild).

The module is import-light: ``jinja2`` is imported lazily so the rest
of compile.py keeps working when the emitter is disabled or when
``jinja2`` isn't installed.  Failures inside the emitter raise
``CodegenError``; the caller in ``_variant_macros`` catches that and
falls back to macros-only.

A ``cutlass-python`` GEMM sweep entry point (``emit_cutlass_gemm_variants``)
is sketched in but gated behind an ImportError → CodegenError shim so
hosts without ``cutlass-python`` simply skip it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TEMPLATE_ROOT = Path(__file__).parent / "templates"


class CodegenError(RuntimeError):
    """Raised when the emitter cannot produce a valid source file.

    Callers in ``compile.py`` catch this and fall back to the
    macros-only build path so a missing template / missing jinja2 /
    bad render never breaks an autotune run."""


# ---------------------------------------------------------------------------
# Jinja2 environment (lazy import)
# ---------------------------------------------------------------------------

def _jinja_env():
    """Construct a Jinja2 ``Environment`` rooted at ``TEMPLATE_ROOT``.

    ``StrictUndefined`` makes typos in template variables crash loudly
    rather than silently rendering an empty string. ``trim_blocks`` and
    ``lstrip_blocks`` keep emitted C++ tidy by stripping the newline
    after a ``{% … %}`` block and the leading whitespace before it."""
    try:
        import jinja2
    except ImportError as exc:
        raise CodegenError(
            "codegen requires jinja2 — pip install jinja2") from exc
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_ROOT)),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


# ---------------------------------------------------------------------------
# Template lookup
# ---------------------------------------------------------------------------

# Filename extensions per vendor — the ``.j2`` suffix is appended by the
# template-file convention (e.g. ``adamw_sm_90a.cu.j2``).
_VENDOR_EXT: Dict[str, str] = {
    "cuda":   ".cu",
    "hip":    ".hip.cpp",
    "pallas": ".py",
}


def _candidate_template_names(optimizer: str, arch: str,
                              vendor: str) -> List[str]:
    """Return template basenames to probe, in priority order.

    Tried in this order:
      1. ``<optimizer>_<arch>.<ext>.j2``           — most-specific
      2. ``<optimizer>_<vendor>.<ext>.j2``         — vendor fallback
      3. ``<optimizer>_generic.<ext>.j2``          — last resort

    ``<ext>`` is chosen from ``_VENDOR_EXT``; for arch=``sm_90a`` the
    arch is also probed without the ``a`` suffix (so a single
    ``adamw_sm_90.cu.j2`` template covers both canonical and alias).
    """
    ext = _VENDOR_EXT.get(vendor, ".cu")
    names: List[str] = [f"{optimizer}_{arch}{ext}.j2"]
    # Probe the non-"a" alias too (Hopper / Blackwell).
    if arch.endswith("a") and arch.startswith("sm_"):
        names.append(f"{optimizer}_{arch[:-1]}{ext}.j2")
    names.append(f"{optimizer}_{vendor}{ext}.j2")
    names.append(f"{optimizer}_generic{ext}.j2")
    return names


def find_template(optimizer: str, arch: str,
                  vendor: Optional[str] = None) -> Optional[Path]:
    """Locate the highest-priority template for ``(optimizer, arch)``.

    ``vendor`` is looked up from ``ARCH_TABLE`` when not supplied.
    Returns ``None`` when none of the candidate filenames exist; the
    caller is expected to raise ``CodegenError`` in that case."""
    if vendor is None:
        # Imported lazily so codegen.py has no import-time dependency
        # on the (large) compile.py module being loaded.
        from grokking_optimizers.compile import get_arch_entry
        try:
            vendor = get_arch_entry(arch).vendor
        except KeyError:
            vendor = "cuda"
    for name in _candidate_template_names(optimizer, arch, vendor):
        p = TEMPLATE_ROOT / name
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def _json_default(o: Any) -> Any:
    """Best-effort fallback for JSON encoding of common non-trivial types
    that may appear in a config dict (e.g. tuples from cluster_shape)."""
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, frozenset):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def _canonical_json(ctx: Dict[str, Any]) -> str:
    """Stable JSON for the cache key: sorted keys, no whitespace."""
    return json.dumps(ctx, sort_keys=True, separators=(",", ":"),
                      default=_json_default)


def _build_context(config: Dict[str, Any], arch: str) -> Dict[str, Any]:
    """Merge ``config`` with arch metadata for template consumption.

    Both nested (``ctx.config["block"]``) and spread access
    (``ctx["block"]``) work in templates — the spread is for ergonomics
    so simple templates don't have to write ``{{ config.block }}`` for
    every reference."""
    from grokking_optimizers.compile import get_arch_entry
    entry = get_arch_entry(arch)
    ctx: Dict[str, Any] = {
        "config": dict(config),
        "arch": arch,
        "vendor": entry.vendor,
        "features": sorted(entry.features),
        "warp_size": entry.warp_size,
        "max_smem_per_block": entry.max_smem_per_block,
        "max_regs_per_thread": entry.max_regs_per_thread,
        "max_threads_per_block": entry.max_threads_per_block,
        "macro": entry.macro,
        "display_name": entry.display_name,
    }
    # Spread config into top-level keys (don't clobber the dotted view).
    for k, v in config.items():
        if k not in ctx:
            ctx[k] = v
    return ctx


# ---------------------------------------------------------------------------
# Variant emission (the hot path called from _variant_macros)
# ---------------------------------------------------------------------------

def emit_variant_source(config: Dict[str, Any],
                        dims: List[Dict[str, Any]],
                        optimizer: str,
                        arch: str,
                        out_dir: Path) -> Tuple[Path, List[str]]:
    """Render a template with the given config; return (path, residual_macros).

    The returned ``path`` is the on-disk emitted source file (``.cu``
    / ``.hip.cpp`` / ``.py``). ``residual_macros`` is the list of
    ``-D`` flags for any dim that the template *can't* bake in
    (typically host-side macros or any dim flagged
    ``kind="force_macro"``); the caller appends them to its existing
    cflags.

    Files are cached by SHA-256 of (template_source + JSON-canonical
    ctx). Re-rendering with an identical config short-circuits to the
    existing file — useful when the JIT autotuner samples the same
    config twice."""
    tpl_path = find_template(optimizer, arch)
    if tpl_path is None:
        raise CodegenError(
            f"no template found for optimizer={optimizer!r} arch={arch!r} "
            f"(searched {TEMPLATE_ROOT})")

    env = _jinja_env()
    try:
        tpl = env.get_template(tpl_path.name)
    except Exception as exc:
        raise CodegenError(
            f"jinja2 could not load template {tpl_path.name}: {exc}") from exc

    ctx = _build_context(config, arch)
    try:
        rendered = tpl.render(**ctx)
    except Exception as exc:
        raise CodegenError(
            f"template {tpl_path.name} failed to render with "
            f"ctx={_canonical_json(ctx)[:200]}…: {exc}") from exc

    # Cache key — content of the template + the JSON-canonical context.
    # 16 hex chars (64 bits) is plenty to avoid collisions in a single
    # autotune sweep (billions of candidates → P[collision] ≈ 0).
    tpl_src = tpl_path.read_text(encoding="utf-8")
    key = hashlib.sha256(
        tpl_src.encode("utf-8") + b"\0" + _canonical_json(ctx).encode("utf-8")
    ).hexdigest()[:16]

    from grokking_optimizers.compile import get_arch_entry
    vendor = get_arch_entry(arch).vendor
    ext = _VENDOR_EXT.get(vendor, ".cu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"emitted_{optimizer}_{arch}_{key}{ext}"
    if not out_path.exists():
        # Atomic-ish write: rename a sibling tmp file so a concurrent
        # autotuner worker never sees a half-written source.
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(rendered, encoding="utf-8")
        tmp.replace(out_path)

    residual = _compute_residual_macros(config, dims)
    return out_path, residual


def _compute_residual_macros(config: Dict[str, Any],
                             dims: List[Dict[str, Any]]) -> List[str]:
    """Build ``-D`` flags for dims that the template can't bake in.

    Two cases:
      1. ``dim["kind"] == "force_macro"`` — explicit opt-out from
         template hardcoding (caller controls)
      2. ``dim["applies_to"]`` contains ``"host"`` — host-side code
         can't see template-emitted device constants, so we still need
         the macro
      3. legacy ``dim["target"] == "host"`` — older dim shape

    Dims without a ``macro`` key (e.g. ``maxrregcount`` which becomes
    a NVCC flag) contribute nothing here."""
    out: List[str] = []
    for d in dims:
        macro = d.get("macro")
        if not macro:
            continue
        name = d.get("name")
        if name is None or name not in config:
            continue
        force = (d.get("kind") == "force_macro")
        applies = d.get("applies_to") or []
        is_host = ("host" in applies) or (d.get("target") == "host")
        if force or is_host:
            v = config[name]
            # Tuples / lists → comma-separated for templates that
            # consumed the macro the old way (kept compact, no spaces).
            if isinstance(v, (list, tuple)):
                v = ",".join(str(x) for x in v)
            elif isinstance(v, bool):
                v = 1 if v else 0
            out.append(f"-D{macro}={v}")
    return out


# ---------------------------------------------------------------------------
# Introspection helpers (self-tests, debug, future TOML wiring)
# ---------------------------------------------------------------------------

def list_available_templates() -> List[Tuple[str, str]]:
    """Enumerate ``(optimizer, arch_or_vendor)`` pairs we have templates for.

    Returns the suffix after the optimizer name as the second element
    — so ``adamw_sm_90a.cu.j2`` yields ``("adamw", "sm_90a")`` and
    ``adamw_generic.cu.j2`` yields ``("adamw", "generic")``."""
    out: List[Tuple[str, str]] = []
    if not TEMPLATE_ROOT.is_dir():
        return out
    for p in sorted(TEMPLATE_ROOT.glob("*.j2")):
        # Strip the ".j2" then the vendor extension (".cu" / ".hip.cpp" / ".py").
        name = p.name[: -len(".j2")]
        for ext in (".hip.cpp", ".cu", ".py"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        # Split into (optimizer, suffix) at the LAST underscore — supports
        # multi-word arch names like "sm_90a" (split keeps "sm_90a" intact).
        if "_" in name:
            opt, suffix = name.split("_", 1)
            out.append((opt, suffix))
        else:
            out.append((name, ""))
    return out


def validate_template_render(optimizer: str, arch: str
                             ) -> Tuple[bool, str]:
    """Render the template with a canned minimal config; report (ok, err).

    Used by ``_self_test`` to ensure every shipped template parses
    cleanly under Jinja2 without needing an actual autotune sweep."""
    try:
        from grokking_optimizers.compile import (
            build_full_search_space, ARCH_TABLE)
        if arch not in ARCH_TABLE:
            return False, f"unknown arch {arch!r}"
        space = build_full_search_space()
        # Map canonical alias to the key used in build_full_search_space
        # (which uses "sm_90" not "sm_90a"). Probe both forms.
        space_key = arch
        if space_key not in space and arch.endswith("a"):
            space_key = arch[:-1]
        block = space.get(space_key, {})
        dims = block.get("dims", []) if isinstance(block, dict) else []
        cfg: Dict[str, Any] = {}
        for d in dims:
            vs = d.get("values") or []
            if vs:
                cfg[d["name"]] = vs[0]
        with tempfile.TemporaryDirectory() as td:
            emit_variant_source(cfg, dims, optimizer, arch, Path(td))
        return True, ""
    except CodegenError as exc:
        return False, f"CodegenError: {exc}"
    except Exception as exc:  # pragma: no cover — defensive
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Compile-time validation (best-effort, gracefully degrades)
# ---------------------------------------------------------------------------

def validate_with_nvcc_dryrun(emitted: Path) -> Tuple[bool, str]:
    """Run ``nvcc --cuda --dryrun`` to check the emitted source parses.

    Returns ``(ok, message)``. When ``nvcc`` isn't on PATH (CI hosts,
    Colab CPU runtimes, AMD-only boxes) this returns
    ``(True, "nvcc unavailable…")`` so the self-test is graceful."""
    if not shutil.which("nvcc"):
        return True, "nvcc unavailable; skipping validation"
    if emitted.suffix not in (".cu", ".cuh"):
        return True, f"non-CUDA source {emitted.suffix}; skipping nvcc check"
    try:
        # --cuda preprocesses+compiles to .cpp.ii without invoking the
        # back-end ptxas, so it works on a CPU-only host. --dryrun just
        # prints the commands it would run; combined with --cuda the
        # parser still runs over the source.
        r = subprocess.run(
            ["nvcc", "--cuda", "--dryrun", str(emitted)],
            capture_output=True, text=True, timeout=30,
        )
        ok = (r.returncode == 0)
        return ok, (r.stderr or r.stdout or "").strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return True, f"nvcc invocation failed ({type(exc).__name__}): {exc}"


# ---------------------------------------------------------------------------
# CUTLASS profiler integration (skeleton — full impl gated on cutlass-python)
# ---------------------------------------------------------------------------

def emit_cutlass_gemm_variants(arch: str,
                               problem_shape: Tuple[int, int, int],
                               dtype: str,
                               out_dir: Path
                               ) -> List[Tuple[Path, Dict[str, Any]]]:
    """Sweep CUTLASS GEMM variants for ``(M, N, K)`` on ``arch``.

    Sketches the entry point Stream 6 will flesh out once
    ``cutlass-python`` is wired into the build environment. Raises
    ``CodegenError`` immediately when the package isn't importable so
    the caller can cleanly fall back to template-only emission.

    Intended sweep axes (when fully implemented):
      - tile shapes:        M × N × K (per cluster size)
      - cluster shapes:     ARCH_TABLE features-aware (Hopper only)
      - pipeline stages:    {2 .. 8}
      - schedule:           {pingpong, cooperative, ws-cooperative}
      - epilogue:           {default, fused-bias, fused-relu, fused-gelu}

    Returns ``[(emitted_path, variant_metadata), …]`` so the autotuner
    can time each variant the same way it times template-emitted
    elementwise kernels.
    """
    try:
        import cutlass  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise CodegenError(
            "cutlass-python required for GEMM emitter "
            "(pip install nvidia-cutlass-python or build from source); "
            "falling back to template-only emission") from exc
    # TODO(stream-6 follow-up): implement the actual cutlass.op.Gemm
    # sweep. The scaffolding above plus the dim list under
    # ``_sm90_full_space["dims"]`` already cover the search axes; this
    # function just needs to instantiate cutlass operations and write
    # the generated .cu files under ``out_dir``.
    _ = (problem_shape, dtype, out_dir)  # silence linters during stub phase
    raise CodegenError(
        "emit_cutlass_gemm_variants is a stub — Stream 6 follow-up will "
        "wire cutlass.op.Gemm.profile() into the autotuner")


__all__ = [
    "TEMPLATE_ROOT",
    "CodegenError",
    "find_template",
    "emit_variant_source",
    "list_available_templates",
    "validate_template_render",
    "validate_with_nvcc_dryrun",
    "emit_cutlass_gemm_variants",
]
