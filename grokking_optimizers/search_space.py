"""grokking_optimizers.search_space — YAML-driven autotune search space.

Single source of truth for the per-arch tunable dimensions used by
``grokking_optimizers.compile``. The YAML at ``configs/search_space.yaml``
is loaded once at build start; this module turns it into:

  * a list of dim specs (name / type / values / macro / applies_to)
  * a cartesian list of candidate config dicts
  * a pre-filtered subset (static rules eliminate infeasible configs
    before any compile is attempted)
  * the host/device ``-D`` macro list resolved from a chosen config
  * a stable SHA-256 hash of the resolved space (cache invalidation)

No GPU access — entire module is pure CPU.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class SearchSpaceError(ValueError):
    """Raised when the YAML is missing required keys or has a bad type."""


_VALID_TYPES = {"int", "bool", "enum", "tuple"}
_VALID_TARGETS = {"host", "device"}


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load the YAML file and validate the per-arch shape."""
    path = Path(path)
    if not path.exists():
        raise SearchSpaceError(f"search-space YAML not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise SearchSpaceError(
            f"search-space YAML must be a top-level dict; got {type(raw).__name__}")
    for arch, block in raw.items():
        _validate_arch(arch, block)
    return raw


def _validate_arch(arch: str, block: Any) -> None:
    if not isinstance(block, dict):
        raise SearchSpaceError(f"{arch}: expected dict, got {type(block).__name__}")
    dims = block.get("dims", [])
    if not isinstance(dims, list):
        raise SearchSpaceError(f"{arch}.dims must be a list")
    seen_names: set = set()
    for dim in dims:
        _validate_dim(arch, dim, seen_names)
        seen_names.add(dim["name"])
    prefilter = block.get("prefilter", {})
    if prefilter and not isinstance(prefilter, dict):
        raise SearchSpaceError(f"{arch}.prefilter must be a dict")
    rules = prefilter.get("rules", []) if prefilter else []
    if rules and not isinstance(rules, list):
        raise SearchSpaceError(f"{arch}.prefilter.rules must be a list")
    for rule in rules:
        if not isinstance(rule, dict) or "expr" not in rule:
            raise SearchSpaceError(
                f"{arch}.prefilter.rules each entry must be a dict with 'expr'")


def _validate_dim(arch: str, dim: Any, seen: set) -> None:
    if not isinstance(dim, dict):
        raise SearchSpaceError(f"{arch}: dim entry must be a dict, got {type(dim).__name__}")
    for required in ("name", "type", "values"):
        if required not in dim:
            raise SearchSpaceError(f"{arch}: dim missing '{required}': {dim}")
    name = dim["name"]
    if not isinstance(name, str):
        raise SearchSpaceError(f"{arch}: dim 'name' must be a str: {dim}")
    if name in seen:
        raise SearchSpaceError(f"{arch}: duplicate dim name {name!r}")
    if dim["type"] not in _VALID_TYPES:
        raise SearchSpaceError(
            f"{arch}.{name}: type {dim['type']!r} not in {_VALID_TYPES}")
    if not isinstance(dim["values"], list) or not dim["values"]:
        raise SearchSpaceError(f"{arch}.{name}: values must be a non-empty list")
    applies_to = dim.get("applies_to", ["host", "device"])
    if not isinstance(applies_to, list):
        raise SearchSpaceError(f"{arch}.{name}: applies_to must be a list")
    bad = set(applies_to) - _VALID_TARGETS
    if bad:
        raise SearchSpaceError(
            f"{arch}.{name}: applies_to has unknown targets {bad}")


# ---------------------------------------------------------------------------
# Cartesian + pre-filter
# ---------------------------------------------------------------------------

def cartesian(space: Dict[str, Any], arch: str) -> List[Dict[str, Any]]:
    """Return the full cartesian product as a list of {dim_name: value} dicts."""
    if arch not in space:
        return []
    dims = space[arch].get("dims", [])
    if not dims:
        return []
    names = [d["name"] for d in dims]
    values = [d["values"] for d in dims]
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        cfg: Dict[str, Any] = {}
        for n, v in zip(names, combo):
            # YAML loads JSON-style lists as Python lists; convert tuple-typed
            # dims back to tuples so expressions like cluster_shape[0] work.
            cfg[n] = tuple(v) if isinstance(v, list) else v
        out.append(cfg)
    return out


def prefilter(configs: List[Dict[str, Any]],
              prefilter_spec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """Apply the static pruning rules. Returns (survivors, eliminated_count)."""
    if not prefilter_spec:
        return list(configs), 0
    rules: List[Dict[str, Any]] = prefilter_spec.get("rules", []) or []
    survivors: List[Dict[str, Any]] = []
    eliminated = 0
    # Compile expressions once for speed.
    compiled = [(r.get("name", f"rule_{i}"), compile(r["expr"], "<prefilter>", "eval"))
                for i, r in enumerate(rules)]
    for cfg in configs:
        ok = True
        for rname, code in compiled:
            try:
                # Eval in a restricted namespace: dim values + a tiny set of
                # safe builtins.
                env = dict(cfg)
                env["__builtins__"] = {
                    "len": len, "min": min, "max": max, "abs": abs,
                    "int": int, "bool": bool, "True": True, "False": False,
                }
                if not bool(eval(code, env, env)):  # noqa: S307 — sandboxed
                    ok = False
                    break
            except Exception:
                # A rule that errors on a config is treated as eliminating
                # that config (defensive default).
                ok = False
                break
        if ok:
            survivors.append(cfg)
        else:
            eliminated += 1
    return survivors, eliminated


# ---------------------------------------------------------------------------
# Macro / flag resolution
# ---------------------------------------------------------------------------

def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, tuple):
        # cluster_shape=(2,2,1) -> "2,2,1"  (caller wraps in {…} if needed)
        return ",".join(str(v) for v in value)
    return str(value)


def resolve_macros(config: Dict[str, Any], dim_specs: List[Dict[str, Any]],
                   target: str) -> List[str]:
    """Return the ``-DFOO=VAL`` flags for a config restricted to ``target``."""
    if target not in _VALID_TARGETS:
        raise ValueError(f"target={target!r} not in {_VALID_TARGETS}")
    out: List[str] = []
    for spec in dim_specs:
        macro = spec.get("macro")
        applies_to = spec.get("applies_to", ["host", "device"])
        if target not in applies_to:
            continue
        if not macro:
            continue
        name = spec["name"]
        if name not in config:
            continue
        out.append(f"-D{macro}={_format_value(config[name])}")
    return out


def resolve_extra_nvcc_flags(config: Dict[str, Any],
                             dim_specs: List[Dict[str, Any]]) -> List[str]:
    """Some dims become bare NVCC/HIPCC flags rather than ``-D`` macros.
    Currently: ``maxrregcount`` -> ``--maxrregcount=N``."""
    out: List[str] = []
    for spec in dim_specs:
        if spec.get("macro") is None and spec["name"] == "maxrregcount":
            v = config.get("maxrregcount")
            if v is not None:
                out.append(f"--maxrregcount={int(v)}")
    return out


def resolve_extra_hipcc_flags(config: Dict[str, Any],
                              dim_specs: List[Dict[str, Any]]) -> List[str]:
    """HIPCC analogue. ``maxrregcount`` -> ``-mllvm -amdgpu-max-num-vgprs=N``."""
    out: List[str] = []
    for spec in dim_specs:
        if spec.get("macro") is None and spec["name"] == "maxrregcount":
            v = config.get("maxrregcount")
            if v is not None:
                out.extend(["-mllvm", f"-amdgpu-max-num-vgprs={int(v)}"])
    return out


# ---------------------------------------------------------------------------
# Hashing (for cache invalidation when the YAML changes)
# ---------------------------------------------------------------------------

def hash_space(space: Dict[str, Any], arch: str) -> str:
    """Stable SHA-256 of the per-arch space (sorted JSON, no whitespace)."""
    if arch not in space:
        return hashlib.sha256(b"").hexdigest()
    block = space[arch]
    payload = json.dumps(block, sort_keys=True, separators=(",", ":"),
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Config key (stable string for cache lookup)
# ---------------------------------------------------------------------------

def config_key(config: Dict[str, Any]) -> str:
    """Compact, deterministic key — used as cache.variant_artifacts subkey."""
    parts = []
    for k in sorted(config.keys()):
        parts.append(f"{k}={_format_value(config[k])}")
    return "_".join(parts)


__all__ = [
    "SearchSpaceError",
    "load_yaml",
    "cartesian",
    "prefilter",
    "resolve_macros",
    "resolve_extra_nvcc_flags",
    "resolve_extra_hipcc_flags",
    "hash_space",
    "config_key",
]
