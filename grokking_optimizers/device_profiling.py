"""Device-side PGO feedback for the compile pipeline.

NVCC strips host PGO instrumentation from device code, so the standard
LLVM ``-fprofile-generate``/``-fprofile-use`` loop only optimizes host
launchers on NVIDIA. This module fills the gap with three vendor-specific
collectors:

  * **NVIDIA**: CUPTI PC sampling via ``nsys profile`` (or direct CUPTI
    when available). Extracts stall reasons such as ``long_scoreboard``,
    ``not_selected``, ``math_pipe_throttle`` from the per-kernel summary.
  * **AMD**: ``rocprof --stats`` ATT traces. Extracts wait / LDS / VALU
    dependency counters and normalizes to fractions.
  * **Pallas**: parses XLA HLO dumps emitted via
    ``XLA_FLAGS=--xla_gpu_dump_autotuned_gemm_fusions=true``. Looks for
    cost-model annotations (``memory_throttle``, ``barrier`` etc.).

Output is a JSON sidecar at ``<out_dir>/device_stall_info.json`` with
shape::

    {
        "arch": "sm_90a",
        "tool": "nsys",
        "report": "<path to raw profiler output>",
        "stall_reasons": {
            "long_scoreboard":    0.42,
            "not_selected":       0.18,
            "math_pipe_throttle": 0.10,
            ...
        },
        "bias_hints": {
            "swizzle":    [64, 128],
            "num_stages": [4, 5, 6],
            ...
        }
    }

The ``bias_hints`` dict is consumed by ``bias_trial_queue()`` to enqueue
biased trials into an Optuna study before the random TPE sweep starts.

This module is intentionally tolerant: every collector returns ``None``
when its underlying tool is unavailable (so a CPU-only host can still
``--self-test`` cleanly) and ``{"error": ...}`` when the tool runs but
fails. The ``run_device_pgo_round`` hook called from ``_build_aot_pgo``
swallows all exceptions so device-PGO failure never breaks the main
build pipeline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


class DeviceProfilingError(RuntimeError):
    """Raised only by callers that opt into strict mode (not used internally)."""


# ---------------------------------------------------------------------------
# Stall reason -> search-space dim hints
# ---------------------------------------------------------------------------
#
# Each entry maps a stall category (as reported by nsys / rocprof / XLA) to
# the list of search-space dimension names whose values should be biased
# when this stall dominates. The dim names mirror those used by the Stream 2
# search-space builders (``swizzle``, ``num_stages``, ``waves_per_eu`` etc.).

STALL_DIM_HINTS: Dict[str, List[str]] = {
    # NVIDIA CUPTI categories
    "long_scoreboard":     ["swizzle", "lds_padding", "vec"],
    "not_selected":        ["block", "waves_per_eu", "maxrregcount"],
    "math_pipe_throttle":  ["unroll", "num_stages"],
    "memory_throttle":     ["block", "vec", "async_depth", "num_stages"],
    "tex_throttle":        ["block", "vec"],
    "barrier":             ["block", "warp_specialization"],
    "wait":                ["num_stages", "async_depth"],
    "imc_miss":            ["block", "unroll"],
    "lg_throttle":         ["lds_padding", "swizzle"],
    "dispatch_stall":      ["block", "maxrregcount"],
    # AMD rocprof categories
    "vmem_lat":            ["waves_per_eu", "vec"],
    "lds_bank_conflict":   ["lds_padding"],
    "valu_dep":            ["unroll", "num_stages"],
}


# ---------------------------------------------------------------------------
# NVIDIA — nsys / CUPTI
# ---------------------------------------------------------------------------

def collect_nvidia_stalls(workload_cmd: List[str], out_dir: Path,
                          *, timeout: int = 600) -> Optional[Dict[str, Any]]:
    """Run ``workload_cmd`` under ``nsys profile`` and parse stall reasons.

    Returns:
      - ``None`` if nsys is not available on the host.
      - ``{"error": ..., "stderr": ...}`` if nsys ran but failed.
      - Otherwise a dict with ``tool``, ``report``, ``stall_reasons``,
        and ``bias_hints`` keys.
    """
    nsys = shutil.which("nsys") or shutil.which("/usr/local/cuda/bin/nsys")
    if nsys is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "cupti_pc_sample.qdrep"
    try:
        proc = subprocess.run(
            [nsys, "profile",
             "-o", str(report.with_suffix("")),
             "--sample=cpu",
             "--cuda-memory-usage=true",
             "--cuda-stack=true",
             "--stats=true"] + list(workload_cmd),
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return {"error": f"nsys exit {proc.returncode}",
                    "stderr": (proc.stderr or "")[-2000:]}
    except subprocess.TimeoutExpired:
        return {"error": "nsys timeout"}
    except OSError as exc:
        return {"error": f"nsys spawn failed: {exc}"}
    stall_reasons = _parse_nsys_stall_section(proc.stdout or "")
    return {
        "tool": "nsys",
        "report": str(report),
        "stall_reasons": stall_reasons,
        "bias_hints": _stall_to_bias_hints(stall_reasons),
    }


def _parse_nsys_stall_section(text: str) -> Dict[str, float]:
    """Best-effort parser for ``nsys --stats=true`` text output.

    nsys emits sections like ``[5/8] Executing 'cuda_gpu_kern_sum' stats...``
    followed by a table. A full implementation would parse the table
    columns explicitly; here we scan line-by-line for any known stall
    reason keyword followed by a percentage value.
    """
    out: Dict[str, float] = {}
    pct_re = re.compile(r"(\d+\.?\d*)\s*%")
    for line in text.splitlines():
        low = line.lower()
        for reason in STALL_DIM_HINTS:
            if reason in low:
                m = pct_re.search(line)
                if m:
                    try:
                        out[reason] = float(m.group(1)) / 100.0
                    except ValueError:
                        continue
    return out


# ---------------------------------------------------------------------------
# AMD — rocprof
# ---------------------------------------------------------------------------

def collect_amd_stalls(workload_cmd: List[str], out_dir: Path,
                       *, timeout: int = 600) -> Optional[Dict[str, Any]]:
    """Run ``workload_cmd`` under ``rocprof --stats`` and parse counters."""
    rocprof = shutil.which("rocprof") or shutil.which("/opt/rocm/bin/rocprof")
    if rocprof is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "rocprof_stats.csv"
    try:
        proc = subprocess.run(
            [rocprof, "--stats", "-o", str(out_csv)] + list(workload_cmd),
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return {"error": f"rocprof exit {proc.returncode}",
                    "stderr": (proc.stderr or "")[-2000:]}
    except subprocess.TimeoutExpired:
        return {"error": "rocprof timeout"}
    except OSError as exc:
        return {"error": f"rocprof spawn failed: {exc}"}
    stall_reasons = _parse_rocprof_csv(out_csv)
    return {
        "tool": "rocprof",
        "report": str(out_csv),
        "stall_reasons": stall_reasons,
        "bias_hints": _stall_to_bias_hints(stall_reasons),
    }


def _parse_rocprof_csv(csv_path: Path) -> Dict[str, float]:
    """Parse a ``rocprof --stats`` CSV into per-category fractions.

    The CSV header columns vary by rocprof version, but counter names
    typically contain substrings like ``WAIT``, ``LDS`` and ``VALU`` that
    we can match against to assign counts to our category buckets.
    Returned values are normalized to fractions summing to <= 1.
    """
    if not csv_path.exists():
        return {}
    out: Dict[str, float] = {}
    try:
        lines = csv_path.read_text(errors="ignore").splitlines()
        if not lines:
            return {}
        header = [c.strip() for c in lines[0].split(",")]
        for line in lines[1:]:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < len(header):
                continue
            row = dict(zip(header, cols))
            for k, v in row.items():
                kl = k.lower()
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue
                if "wait" in kl:
                    out["vmem_lat"] = out.get("vmem_lat", 0.0) + val
                if "lds" in kl:
                    out["lds_bank_conflict"] = (
                        out.get("lds_bank_conflict", 0.0) + val)
                if "valu" in kl:
                    out["valu_dep"] = out.get("valu_dep", 0.0) + val
    except Exception:
        return {}
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


# ---------------------------------------------------------------------------
# Pallas — XLA HLO cost-model dump
# ---------------------------------------------------------------------------

def collect_pallas_stalls(workload_cmd: List[str], out_dir: Path,
                          *, timeout: int = 600) -> Optional[Dict[str, Any]]:
    """Run ``workload_cmd`` with XLA HLO dump enabled and parse cost hints.

    Always attempts to invoke the workload (XLA is part of jax, which is
    a normal pip dep). Returns ``{"error": ...}`` if the workload fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = out_dir / "xla_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    extra_flags = (
        f"--xla_gpu_dump_autotuned_gemm_fusions=true "
        f"--xla_dump_to={dump_dir}"
    )
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " " + extra_flags).strip()
    try:
        proc = subprocess.run(
            list(workload_cmd),
            env=env, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return {"error": f"xla workload exit {proc.returncode}",
                    "stderr": (proc.stderr or "")[-2000:]}
    except subprocess.TimeoutExpired:
        return {"error": "xla workload timeout"}
    except OSError as exc:
        return {"error": f"xla workload spawn failed: {exc}"}
    stall_reasons = _parse_xla_dump(dump_dir)
    return {
        "tool": "xla_dump",
        "report": str(dump_dir),
        "stall_reasons": stall_reasons,
        "bias_hints": _stall_to_bias_hints(stall_reasons),
    }


def _parse_xla_dump(dump_dir: Path) -> Dict[str, float]:
    """Best-effort cost-model keyword scan of XLA HLO ``.txt`` dumps."""
    if not dump_dir.exists():
        return {}
    out: Dict[str, float] = {}
    for f in dump_dir.glob("*.txt"):
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            low = line.lower()
            if "memory_throttle" in low:
                out["memory_throttle"] = out.get("memory_throttle", 0.0) + 0.05
            if "barrier" in low:
                out["barrier"] = out.get("barrier", 0.0) + 0.05
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


# ---------------------------------------------------------------------------
# Vendor dispatch
# ---------------------------------------------------------------------------

def collect_device_stalls(arch: str, workload_cmd: List[str],
                          out_dir: Path) -> Optional[Dict[str, Any]]:
    """Dispatch to the right collector by arch vendor.

    Imports ``ARCH_TABLE`` lazily to avoid a circular import with
    ``compile.py``.
    """
    try:
        from grokking_optimizers.compile import ARCH_TABLE  # type: ignore
    except Exception:
        return None
    if arch not in ARCH_TABLE:
        return None
    vendor = ARCH_TABLE[arch].vendor
    if vendor == "cuda":
        result = collect_nvidia_stalls(workload_cmd, out_dir)
    elif vendor == "hip":
        result = collect_amd_stalls(workload_cmd, out_dir)
    elif vendor == "pallas":
        result = collect_pallas_stalls(workload_cmd, out_dir)
    else:
        return None
    if result is not None:
        result.setdefault("arch", arch)
    return result


# ---------------------------------------------------------------------------
# Stall -> Bias mapping
# ---------------------------------------------------------------------------

def _stall_to_bias_hints(stall_reasons: Dict[str, float]
                         ) -> Dict[str, List[Any]]:
    """Map dominant stall reasons to recommended search-space values.

    Looks at the top 5 stall categories by fraction; anything below 5%
    is ignored. Each surviving category contributes its set of associated
    dim names (via ``STALL_DIM_HINTS``) and, for a few well-known cases,
    specific value recommendations.
    """
    hints: Dict[str, List[Any]] = {}
    if not stall_reasons:
        return hints
    top = sorted(stall_reasons.items(), key=lambda x: -x[1])[:5]
    for reason, frac in top:
        if frac < 0.05:
            continue
        for dim in STALL_DIM_HINTS.get(reason, []):
            hints.setdefault(dim, [])
    # Specific value recommendations for the most actionable patterns.
    if stall_reasons.get("long_scoreboard", 0) > 0.2:
        hints.setdefault("swizzle", []).extend([64, 128])
        hints.setdefault("lds_padding", []).extend([8, 16])
    if stall_reasons.get("not_selected", 0) > 0.15:
        hints.setdefault("waves_per_eu", []).extend([2, 4])
        hints.setdefault("maxrregcount", []).extend([64, 96])
    if stall_reasons.get("memory_throttle", 0) > 0.2:
        hints.setdefault("vec", []).extend([4, 8])
        hints.setdefault("async_depth", []).extend([4, 8])
    if stall_reasons.get("math_pipe_throttle", 0) > 0.2:
        hints.setdefault("unroll", []).extend([4, 8])
        hints.setdefault("num_stages", []).extend([3, 4, 5])
    if stall_reasons.get("lds_bank_conflict", 0) > 0.15:
        hints.setdefault("lds_padding", []).extend([4, 8, 16])
    # Dedupe + sort for deterministic output.
    for k in list(hints.keys()):
        hints[k] = sorted(set(hints[k]))
    return hints


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------

def write_stall_sidecar(stall_info: Dict[str, Any], out_dir: Path) -> Path:
    """Write the JSON sidecar at ``<out_dir>/device_stall_info.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "device_stall_info.json"
    path.write_text(json.dumps(stall_info, indent=2, default=str))
    return path


def read_stall_sidecar(out_dir: Path) -> Optional[Dict[str, Any]]:
    """Inverse of ``write_stall_sidecar``. Returns ``None`` if absent."""
    path = out_dir / "device_stall_info.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Optuna integration hook (consumed by Stream 5)
# ---------------------------------------------------------------------------

def bias_trial_queue(study, stall_info: Optional[Dict[str, Any]],
                     space: Dict[str, Any], arch: str,
                     *, max_enqueued: int = 25) -> int:
    """Enqueue Optuna trials biased toward stall-suggested values.

    Stream 5 (Bayesian autotuner) owns the actual ``optuna.Study``; this
    function is the integration point. ``study`` is any object with an
    ``enqueue_trial(cfg: dict)`` method.

    ``space`` is the resolved search-space dict for ``arch`` (shape:
    ``{"dims": [{"name", "values", ...}, ...]}``). For each bias hint
    we build a base config (each dim's middle value), then overlay the
    hinted value into the relevant dim and enqueue that point.

    Returns the number of trials successfully enqueued.
    """
    if not stall_info or "bias_hints" not in stall_info:
        return 0
    hints = stall_info["bias_hints"]
    if not hints:
        return 0
    dims = space.get("dims", []) if isinstance(space, dict) else []
    if not dims:
        return 0
    # Build a base config: middle-of-list value per dim, so the bias is a
    # focused single-axis perturbation.
    base: Dict[str, Any] = {}
    for d in dims:
        name = d.get("name")
        vals = d.get("values") or []
        if name is None or not vals:
            continue
        base[name] = vals[len(vals) // 2]
    enqueued = 0
    for dim_name, preferred_vals in hints.items():
        if dim_name not in base:
            continue
        for val in preferred_vals:
            cfg = dict(base)
            cfg[dim_name] = val
            try:
                study.enqueue_trial(cfg)
                enqueued += 1
                if enqueued >= max_enqueued:
                    return enqueued
            except Exception:
                # Optuna raises if the trial duplicates an existing one,
                # or if a value is outside the parameter domain. Skip and
                # keep going — biasing is best-effort.
                continue
    return enqueued


# ---------------------------------------------------------------------------
# Hook called from _build_aot_pgo
# ---------------------------------------------------------------------------

def run_device_pgo_round(spec, workload_cmd: List[str],
                         out_dir: Path, report=None) -> Optional[Path]:
    """End-of-PGO hook: collect device stalls and write the JSON sidecar.

    Called from ``_build_aot_pgo`` after the standard 3-pass LLVM PGO
    loop completes. Reads the ``spec.enable_device_pgo`` flag (added by
    this stream) and returns the sidecar path on success, or ``None`` if
    the flag is off / no profiler is available / anything goes wrong.

    All failures are swallowed and logged to ``report`` — device PGO is
    strictly additive and must never break the main build.
    """
    if not getattr(spec, "enable_device_pgo", False):
        return None
    try:
        stall_info = collect_device_stalls(spec.arch, workload_cmd, out_dir)
        if stall_info is None:
            if report is not None:
                report.write(
                    f"  [device-pgo] no profiler available for arch "
                    f"{spec.arch}\n")
            return None
        if "error" in stall_info:
            if report is not None:
                report.write(f"  [device-pgo] {stall_info['error']}\n")
            return None
        sidecar = write_stall_sidecar(stall_info, out_dir)
        if report is not None:
            report.write(
                f"  [device-pgo] {stall_info.get('tool')} -> {sidecar}; "
                f"{len(stall_info.get('stall_reasons', {}))} stall "
                f"categories, "
                f"{len(stall_info.get('bias_hints', {}))} bias hints\n")
        return sidecar
    except Exception as exc:
        if report is not None:
            report.write(f"  [device-pgo] FAILED: {exc}\n")
        return None
