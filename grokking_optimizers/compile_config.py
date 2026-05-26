"""TOML project config loader for grokking_optimizers.compile.

Search order:
  1. Path passed via build(config=...) / main(--config)
  2. ./compile_config.toml in current working directory
  3. <repo_root>/grokking_optimizers/compile_config.toml (default)

Returns a plain dict. All keys optional; callers should use dict.get(...)
with sensible defaults.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import os
import sys

DEFAULT_CONFIG_PATH = Path(__file__).parent / "compile_config.toml"
CWD_CONFIG_NAME = "compile_config.toml"


def _load_toml(path: Path) -> Dict[str, Any]:
    """Use stdlib tomllib (3.11+) or fall back to tomli."""
    if sys.version_info >= (3, 11):
        import tomllib
        with path.open("rb") as f:
            return tomllib.load(f)
    else:
        try:
            import tomli
            with path.open("rb") as f:
                return tomli.load(f)
        except ImportError:
            # Last resort: very crude regex parser for our flat config
            import re
            data: Dict[str, Any] = {}
            section = None
            for line in path.read_text().splitlines():
                line = line.split("#", 1)[0].rstrip()
                if not line: continue
                m = re.match(r"\[([\w.]+)\]", line)
                if m:
                    section = m.group(1)
                    cur = data
                    for part in section.split("."):
                        cur = cur.setdefault(part, {})
                    continue
                m = re.match(r"(\w+)\s*=\s*(.+)", line)
                if m and section is not None:
                    key, raw = m.group(1), m.group(2).strip()
                    cur = data
                    for part in section.split("."):
                        cur = cur.setdefault(part, {})
                    try:
                        v = eval(raw, {"__builtins__": {}}, {"true": True, "false": False})
                    except Exception:
                        v = raw.strip('"').strip("'")
                    cur[key] = v
            return data


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the project config, merging in this priority order:
      1. The argument `path` (if given and exists)
      2. ./compile_config.toml in CWD (if exists)
      3. The packaged default at DEFAULT_CONFIG_PATH

    Returns the deep-merged result. Caller-passed config wins over CWD config
    which wins over packaged defaults.
    """
    layers = []
    if DEFAULT_CONFIG_PATH.exists():
        layers.append(_load_toml(DEFAULT_CONFIG_PATH))
    cwd_path = Path.cwd() / CWD_CONFIG_NAME
    if cwd_path.exists() and cwd_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        layers.append(_load_toml(cwd_path))
    if path is not None:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--config: {p} not found")
        if p != DEFAULT_CONFIG_PATH.resolve():
            layers.append(_load_toml(p))
    return _deep_merge(*layers)


def _deep_merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        for k, v in d.items():
            if (k in out and isinstance(out[k], dict)
                    and isinstance(v, dict)):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
    return out


def apply_to_buildspec(spec, config: Dict[str, Any]) -> None:
    """Mutate spec in-place: apply config fields if not already set
    by explicit CLI args. Caller is responsible for tracking which
    spec fields were CLI-set vs defaulted."""
    # Codegen
    if "codegen" in config:
        if not getattr(spec, "enable_emitter", False) and \
                config["codegen"].get("enable_emitter"):
            spec.enable_emitter = True
    # Runtime spec
    if "runtime_specialization" in config:
        if not getattr(spec, "enable_runtime_specialization", False) and \
                config["runtime_specialization"].get("enable"):
            spec.enable_runtime_specialization = True
    # Device PGO
    if "device_pgo" in config:
        if not getattr(spec, "enable_device_pgo", False) and \
                config["device_pgo"].get("enable"):
            spec.enable_device_pgo = True
    # Numerics
    if "numerics" in config:
        if not getattr(spec, "strict_numerics", False) and \
                config["numerics"].get("strict"):
            spec.strict_numerics = True
    # Cache
    if "cache" in config:
        if not config["cache"].get("auto_prune_after_jit", True):
            spec.prune_after_autotune = False
        spec.prune_max_age_days = int(
            config["cache"].get("max_age_days", 30))
        spec.prune_keep_top_n = int(
            config["cache"].get("keep_top_n", 100))


def project_sources(config: Dict[str, Any], vendor: str, arch: str) -> Dict[str, Path]:
    """Return resolved source paths for a (vendor, arch). Used by
    _resolve_sources to override the default csrc/backends/<vendor>/<arch> layout."""
    src = config.get("sources", {})
    root_key = {"cuda": "cuda_root", "hip": "hip_root",
                 "pallas": "pallas_root"}[vendor]
    root = src.get(root_key, f"csrc/backends/{vendor}")
    if vendor == "pallas":
        return {"backend": Path(root)}
    return {"backend": Path(root) / arch}


def allowed_optimizers(config: Dict[str, Any]) -> Optional[list]:
    opts = config.get("optimizers", {}).get("enabled", [])
    return opts if opts else None


def allowed_models(config: Dict[str, Any]) -> Optional[list]:
    mds = config.get("models", {}).get("enabled", [])
    return mds if mds else None


def allowed_archs(config: Dict[str, Any]) -> Optional[list]:
    ar = config.get("archs", {}).get("allowed", [])
    return ar if ar else None


def default_arch(config: Dict[str, Any]) -> Optional[str]:
    return config.get("archs", {}).get("default") or None
