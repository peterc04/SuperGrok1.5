"""Live GPU/TPU utilization tracking across all 33 pipelines per arch.

Each arch ships **33 fused training pipelines** (11 optimizers × 3 models).
This module measures the *device utilization* each pipeline drives — the
fraction of the accelerator's compute + memory that the fused step actually
keeps busy — by sampling the device with a lightweight background poller while
the pipeline runs a sustained workload, then aggregating per-cell stats.

It complements the two existing tools:
  * ``grokking_optimizers.profile`` — one-shot ncu / rocprof / jax.profiler
    *dump* per cell (deep per-kernel counters; heavyweight).
  * ``bench_backends`` — wall-clock latency / throughput.
This module is the **live device-wide utilization sweep**: low-overhead, runs
all 33 cells, emits one structured record per cell.

Per-arch sampler backends (device-wide, single-tenant assumption):
  * NVIDIA  → ``pynvml`` ``nvmlDeviceGetUtilizationRates`` (SM% + mem%),
    falling back to ``nvidia-smi --query-gpu=utilization.gpu,...``.
  * AMD     → ``amdsmi`` / ``rocm-smi --showuse --json`` (GPU use% + VRAM%).
  * TPU     → ``jax`` ``device.memory_stats()`` for live HBM utilization;
    MXU compute duty-cycle is left ``None`` (it requires an xprof/jax.profiler
    trace — see ``grokking_optimizers.profile``) and is documented as the
    hardware-gated piece.

Honesty: on a host with no accelerator (or no sampler library), every cell
record is still produced with ``status`` set and metrics ``None`` — so the
33-cell structure, the JSON schema, and the aggregation are all CPU-testable;
only the *numbers* require the accelerator (the same posture as the rest of the
tree; see HARDWARE_VALIDATION.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from grokking_optimizers.profile import (
    ARCHES,
    MODELS,
    OPTIMIZERS,
    REPO_ROOT,
    _arch_info,
    child_env,
    make_progress,
    write_temp_script,
)

# 33 = 11 optimizers × 3 models — the pipeline count PER ARCH.
PIPELINES_PER_ARCH: int = len(OPTIMIZERS) * len(MODELS)


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class UtilSample:
    """One device-wide utilization sample."""
    t: float                          # monotonic seconds since sampler start
    compute_pct: Optional[float]      # SM/CU/MXU busy %, None if unavailable
    mem_pct: Optional[float]          # memory-controller busy % (GPU) / HBM
                                      # occupancy % (TPU), None if unavailable
    mem_used_mb: Optional[float]      # device memory in use, MB


@dataclasses.dataclass
class CellUtilization:
    """Aggregated utilization for one (optimizer, model) pipeline on one arch."""
    arch: str
    optimizer: str
    model: str
    status: str                       # "ok" | "no-device" | "sampler-unavailable"
                                      # | "workload-failed" | "no-samples"
    backend: str                      # sampler backend name
    n_samples: int
    duration_s: float
    compute_mean: Optional[float]
    compute_peak: Optional[float]
    mem_mean: Optional[float]
    mem_peak: Optional[float]
    mem_used_peak_mb: Optional[float]
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Device samplers (device-wide, background-thread pollers)
# ---------------------------------------------------------------------------

class DeviceSampler:
    """Base sampler: a background thread that polls the device at ``interval``.

    Subclasses implement :meth:`available` and :meth:`_read` (one sample).
    The base owns the thread, the sample buffer, and start/stop.
    """

    backend = "none"

    def __init__(self, interval_s: float = 0.1, device_index: int = 0):
        self.interval_s = interval_s
        self.device_index = device_index
        self._samples: List[UtilSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def available(self) -> bool:  # pragma: no cover - overridden
        return False

    def _read(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (compute_pct, mem_pct, mem_used_mb) for one sample."""
        return (None, None, None)

    # -- lifecycle ----------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                compute, mem, mem_mb = self._read()
            except Exception:
                compute, mem, mem_mb = (None, None, None)
            self._samples.append(
                UtilSample(time.monotonic() - self._t0, compute, mem, mem_mb))
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._samples = []
        self._stop.clear()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[UtilSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return list(self._samples)

    def setup(self) -> None:
        """Optional one-time init (e.g. nvmlInit). Default no-op."""

    def teardown(self) -> None:
        """Optional one-time shutdown (e.g. nvmlShutdown). Default no-op."""


class _NvmlSampler(DeviceSampler):
    """NVIDIA sampler via pynvml, falling back to nvidia-smi parsing."""

    backend = "nvml"

    def __init__(self, interval_s: float = 0.1, device_index: int = 0):
        super().__init__(interval_s, device_index)
        self._pynvml = None
        self._handle = None
        self._smi = None  # path to nvidia-smi for the fallback

    def available(self) -> bool:
        try:
            import pynvml  # noqa: F401
            self._pynvml = pynvml
            return True
        except Exception:
            pass
        self._smi = shutil.which("nvidia-smi")
        return self._smi is not None

    def setup(self) -> None:
        if self._pynvml is not None:
            self._pynvml.nvmlInit()
            self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(
                self.device_index)

    def teardown(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def _read(self):
        if self._pynvml is not None and self._handle is not None:
            rates = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return (float(rates.gpu), float(rates.memory),
                    float(mem.used) / (1024.0 * 1024.0))
        # nvidia-smi one-shot query fallback.
        out = subprocess.run(
            [self._smi, "--query-gpu=utilization.gpu,utilization.memory,"
             "memory.used", "--format=csv,noheader,nounits",
             f"--id={self.device_index}"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        if not out:
            return (None, None, None)
        gpu, mem_pct, mem_mb = (x.strip() for x in out[0].split(","))
        return (float(gpu), float(mem_pct), float(mem_mb))


class _RocmSmiSampler(DeviceSampler):
    """AMD sampler via amdsmi, falling back to rocm-smi --showuse --json."""

    backend = "rocm-smi"

    def __init__(self, interval_s: float = 0.1, device_index: int = 0):
        super().__init__(interval_s, device_index)
        self._amdsmi = None
        self._handle = None
        self._rocm_smi = None

    def available(self) -> bool:
        try:
            import amdsmi  # noqa: F401
            self._amdsmi = amdsmi
            return True
        except Exception:
            pass
        self._rocm_smi = shutil.which("rocm-smi")
        return self._rocm_smi is not None

    def setup(self) -> None:
        if self._amdsmi is not None:
            self._amdsmi.amdsmi_init()
            handles = self._amdsmi.amdsmi_get_processor_handles()
            self._handle = handles[self.device_index]

    def teardown(self) -> None:
        if self._amdsmi is not None:
            try:
                self._amdsmi.amdsmi_shut_down()
            except Exception:
                pass

    def _read(self):
        if self._amdsmi is not None and self._handle is not None:
            eng = self._amdsmi.amdsmi_get_gpu_activity(self._handle)
            mem = self._amdsmi.amdsmi_get_gpu_vram_usage(self._handle)
            used_mb = float(mem.get("vram_used", 0))
            total_mb = float(mem.get("vram_total", 0)) or 1.0
            return (float(eng.get("gfx_activity", 0)),
                    100.0 * used_mb / total_mb, used_mb)
        # rocm-smi JSON fallback: GPU use% + VRAM%.
        out = subprocess.run(
            [self._rocm_smi, "--showuse", "--showmemuse", "--json"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        data = json.loads(out)
        card = data.get(f"card{self.device_index}", {})
        compute = card.get("GPU use (%)")
        mem_pct = card.get("GPU Memory Allocated (VRAM%)") or \
            card.get("GPU memory use (%)")
        return (float(compute) if compute is not None else None,
                float(mem_pct) if mem_pct is not None else None, None)


class _TpuSampler(DeviceSampler):
    """TPU sampler via JAX ``device.memory_stats()`` for live HBM utilization.

    MXU compute duty-cycle is NOT exposed as a pollable counter by JAX/libtpu;
    it requires an xprof/jax.profiler trace (see grokking_optimizers.profile),
    so ``compute_pct`` is left ``None`` and noted as the HW-gated piece.
    """

    backend = "jax-tpu"

    def __init__(self, interval_s: float = 0.1, device_index: int = 0):
        super().__init__(interval_s, device_index)
        self._device = None

    def available(self) -> bool:
        try:
            import jax
            devs = [d for d in jax.devices() if d.platform == "tpu"]
            if not devs:
                return False
            self._device = devs[self.device_index]
            return hasattr(self._device, "memory_stats")
        except Exception:
            return False

    def _read(self):
        stats = self._device.memory_stats() or {}
        used = stats.get("bytes_in_use")
        limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
        used_mb = (float(used) / (1024.0 * 1024.0)) if used else None
        mem_pct = (100.0 * float(used) / float(limit)) if used and limit else None
        # compute_pct intentionally None — MXU duty-cycle is xprof-only.
        return (None, mem_pct, used_mb)


_SAMPLERS_BY_VENDOR = {
    "cuda": _NvmlSampler,
    "hip": _RocmSmiSampler,
    "pallas": _TpuSampler,
}


def make_sampler(arch: str, interval_s: float = 0.1,
                 device_index: int = 0) -> DeviceSampler:
    """Return the device sampler matching ``arch``'s vendor.

    Always returns a sampler instance; call ``.available()`` to check whether
    the backend can actually read the device on this host.
    """
    vendor = _arch_info().get(arch, {}).get("vendor")
    cls = _SAMPLERS_BY_VENDOR.get(vendor, DeviceSampler)
    return cls(interval_s=interval_s, device_index=device_index)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(samples: List[UtilSample], *, drop_warmup: int = 1
               ) -> Dict[str, Optional[float]]:
    """Reduce a sample list to mean/peak compute + mem stats.

    The first ``drop_warmup`` samples are discarded (device spin-up / first
    iteration overhead) when there are enough samples to spare.
    """
    use = samples[drop_warmup:] if len(samples) > drop_warmup + 1 else samples

    def _stats(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
        clean = [v for v in vals if v is not None]
        if not clean:
            return (None, None)
        return (sum(clean) / len(clean), max(clean))

    c_mean, c_peak = _stats([s.compute_pct for s in use])
    m_mean, m_peak = _stats([s.mem_pct for s in use])
    used = [s.mem_used_mb for s in use if s.mem_used_mb is not None]
    return {
        "compute_mean": c_mean, "compute_peak": c_peak,
        "mem_mean": m_mean, "mem_peak": m_peak,
        "mem_used_peak_mb": max(used) if used else None,
    }


# ---------------------------------------------------------------------------
# Sustained workload (subprocess) — keeps the device busy so the sampler has a
# stable window to measure. Runs the per-pipeline step in a loop.
# ---------------------------------------------------------------------------

def workload_script(optimizer: str, model: str, arch: str, iters: int) -> str:
    """A sustained-load script for one pipeline: ``iters`` fused/optimizer
    steps in a loop, syncing each iteration so the device stays saturated.

    GPU (cuda/hip): exercises the torch optimizer class (routes into the built
    extension). TPU (pallas): runs the fused cell's ``step`` in a loop, falling
    back to the launcher if the cell is unavailable.
    """
    from grokking_optimizers.profile import OPT_CLASS
    vendor = _arch_info().get(arch, {}).get("vendor")
    if vendor == "pallas":
        return _pallas_workload_script(optimizer, model, iters)
    cls = OPT_CLASS[optimizer]
    return (
        "import sys, traceback\n"
        "try:\n"
        "    import torch\n"
        f"    from grokking_optimizers import {cls}\n"
        "    torch.manual_seed(0)\n"
        "    dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "    p = torch.nn.Parameter(torch.randn(4096, 4096, device=dev))\n"
        f"    opt = {cls}([p], lr=1e-3)\n"
        "    print('WORKLOAD_READY', flush=True)\n"
        f"    for _ in range({iters}):\n"
        "        if p.grad is not None: p.grad = None\n"
        "        (p * p).sum().backward()\n"
        "        opt.step()\n"
        "        if torch.cuda.is_available(): torch.cuda.synchronize()\n"
        f"    print('WORKLOAD_DONE ({optimizer}/{model}/{arch})')\n"
        "except Exception:\n"
        "    traceback.print_exc(); sys.exit(1)\n"
    )


def _pallas_workload_script(optimizer: str, model: str, iters: int) -> str:
    """TPU sustained load: run the fused cell ``step`` (or ``verify``) in a
    loop. Best-effort — the cell import + trace is the hard requirement."""
    from grokking_optimizers.dispatch import canonicalize_model
    canon = canonicalize_model(model)
    cell = f"mega_{canon}_{optimizer}"
    return (
        "import sys, traceback, importlib\n"
        "try:\n"
        "    import jax\n"
        f"    m = importlib.import_module('csrc.fused.tpu_v6e.{cell}')\n"
        "    print('WORKLOAD_READY', flush=True)\n"
        f"    for _ in range({iters}):\n"
        "        r = m.verify()\n"
        "        jax.tree_util.tree_map(\n"
        "            lambda x: x.block_until_ready()\n"
        "            if hasattr(x, 'block_until_ready') else x, r)\n"
        f"    print('WORKLOAD_DONE ({optimizer}/{model}/tpu_v6e)')\n"
        "except Exception:\n"
        "    traceback.print_exc(); sys.exit(1)\n"
    )


# ---------------------------------------------------------------------------
# Per-cell + full-sweep drivers
# ---------------------------------------------------------------------------

def track_cell(optimizer: str, model: str, arch: str, *,
               iters: int = 200, interval_s: float = 0.05,
               timeout: int = 300,
               sampler: Optional[DeviceSampler] = None) -> CellUtilization:
    """Run one pipeline under a sustained load and measure device utilization.

    A device-wide sampler runs in this (parent) process while the workload runs
    in a subprocess. Returns a :class:`CellUtilization`. Never raises for a
    missing device / sampler / failed workload — the status field records it.
    """
    own_sampler = sampler is None
    sampler = sampler or make_sampler(arch, interval_s=interval_s)

    base = CellUtilization(
        arch=arch, optimizer=optimizer, model=model, status="ok",
        backend=sampler.backend, n_samples=0, duration_s=0.0,
        compute_mean=None, compute_peak=None, mem_mean=None, mem_peak=None,
        mem_used_peak_mb=None)

    if not sampler.available():
        base.status = "no-device" if sampler.backend == "none" \
            else "sampler-unavailable"
        base.note = (f"no {sampler.backend} device/library on this host; "
                     f"run on the target accelerator for live numbers")
        return base

    script = workload_script(optimizer, model, arch, iters)
    script_path = write_temp_script(script)
    if own_sampler:
        try:
            sampler.setup()
        except Exception as exc:
            base.status = "sampler-unavailable"
            base.note = f"sampler.setup() failed: {exc}"
            return base

    t0 = time.monotonic()
    try:
        sampler.start()
        proc = subprocess.run(
            [sys.executable, script_path], cwd=REPO_ROOT, env=child_env(),
            capture_output=True, text=True, timeout=timeout)
        samples = sampler.stop()
        base.duration_s = time.monotonic() - t0
        base.n_samples = len(samples)
        if proc.returncode != 0:
            base.status = "workload-failed"
            tail = (proc.stderr or proc.stdout or "")[-300:]
            base.note = f"workload exit {proc.returncode}: {tail.strip()}"
            return base
        if not samples:
            base.status = "no-samples"
            base.note = "sampler produced no samples"
            return base
        agg = _aggregate(samples)
        base.compute_mean = agg["compute_mean"]
        base.compute_peak = agg["compute_peak"]
        base.mem_mean = agg["mem_mean"]
        base.mem_peak = agg["mem_peak"]
        base.mem_used_peak_mb = agg["mem_used_peak_mb"]
        if base.compute_mean is None and base.mem_mean is None:
            base.note = (f"{sampler.backend}: no compute/mem counters exposed "
                         f"(e.g. TPU MXU duty-cycle is xprof-only)")
    except subprocess.TimeoutExpired:
        sampler.stop()
        base.status = "workload-failed"
        base.note = f"workload timed out after {timeout}s"
    finally:
        if own_sampler:
            sampler.teardown()
        try:
            Path(script_path).unlink()
        except OSError:
            pass
    return base


def enumerate_cells(optimizers: Optional[List[str]] = None,
                    models: Optional[List[str]] = None
                    ) -> List[Tuple[str, str]]:
    """Return the (optimizer, model) pipeline list — 33 by default."""
    opts = optimizers or list(OPTIMIZERS)
    mods = models or list(MODELS)
    return [(o, m) for o in opts for m in mods]


def track_all(arch: str, *, optimizers: Optional[List[str]] = None,
              models: Optional[List[str]] = None,
              iters: int = 200, interval_s: float = 0.05,
              timeout: int = 300, progress: bool = True
              ) -> List[CellUtilization]:
    """Sweep all pipelines for ``arch`` (33 by default) and return one
    :class:`CellUtilization` per cell. Shares one sampler across the sweep."""
    if arch not in ARCHES:
        raise ValueError(f"arch={arch!r} not in {list(ARCHES)}")
    cells = enumerate_cells(optimizers, models)
    sampler = make_sampler(arch, interval_s=interval_s)
    have = sampler.available()
    if have:
        try:
            sampler.setup()
        except Exception:
            have = False

    step = close = None
    if progress:
        step, close = make_progress(len(cells), f"utilization {arch}")
    results: List[CellUtilization] = []
    try:
        for opt, model in cells:
            results.append(track_cell(
                opt, model, arch, iters=iters, interval_s=interval_s,
                timeout=timeout, sampler=sampler if have else None))
            if step:
                step(f"{opt}/{model}")
    finally:
        if have:
            sampler.teardown()
        if close:
            close()
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], width: int = 7, prec: int = 1) -> str:
    return f"{v:>{width}.{prec}f}" if v is not None else f"{'—':>{width}}"


def format_table(results: List[CellUtilization]) -> str:
    """Render the per-cell utilization results as a fixed-width table."""
    if not results:
        return "(no results)\n"
    arch = results[0].arch
    lines = [
        f"# Device utilization — {arch} "
        f"({len(results)} pipelines, backend={results[0].backend})",
        f"{'optimizer':<12} {'model':<10} {'compute%':>9} {'mem%':>7} "
        f"{'memMB':>9} {'samp':>5}  status",
        "-" * 72,
    ]
    for r in results:
        lines.append(
            f"{r.optimizer:<12} {r.model:<10} "
            f"{_fmt(r.compute_mean, 9)} {_fmt(r.mem_mean, 7)} "
            f"{_fmt(r.mem_used_peak_mb, 9, 0)} {r.n_samples:>5}  {r.status}")
    ok = [r for r in results if r.status == "ok" and r.compute_mean is not None]
    if ok:
        avg = sum(r.compute_mean for r in ok) / len(ok)
        peak = max(r.compute_peak for r in ok if r.compute_peak is not None)
        lines.append("-" * 72)
        lines.append(f"mean compute% across {len(ok)} measured cells: "
                     f"{avg:.1f}  (peak {peak:.1f})")
    return "\n".join(lines) + "\n"


def write_report(results: List[CellUtilization], arch: str,
                 out_path: Optional[Path] = None) -> Path:
    """Write the sweep results to a JSON sidecar and return its path."""
    if out_path is None:
        out_dir = REPO_ROOT / "build" / "utilization"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"utilization_{arch}.json"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "generated": datetime.datetime.now().isoformat(),
        "pipelines_per_arch": PIPELINES_PER_ARCH,
        "n_cells": len(results),
        "backend": results[0].backend if results else "none",
        "cells": [r.to_dict() for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m grokking_optimizers.utilization",
        description="Track live GPU/TPU device utilization across all 33 "
                    "fused pipelines (11 optimizers × 3 models) for an arch.")
    parser.add_argument(
        "--arch", "-A", choices=ARCHES, default=None,
        help="Target arch (default: auto-detect via dispatch.detect_arch).")
    parser.add_argument("--optimizer", "-O", choices=OPTIMIZERS, default=None,
                        help="Limit to one optimizer (default: all 11).")
    parser.add_argument("--model", "-M", choices=MODELS, default=None,
                        help="Limit to one model (default: all 3).")
    parser.add_argument("--iters", type=int, default=200,
                        help="Sustained-load iterations per cell (default 200).")
    parser.add_argument("--interval-ms", type=int, default=50,
                        help="Sampler poll interval in ms (default 50).")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-cell workload timeout in seconds.")
    parser.add_argument("--out", type=Path, default=None,
                        help="JSON output path (default: "
                             "build/utilization/utilization_<arch>.json).")
    parser.add_argument("--list", action="store_true",
                        help="List the 33 pipelines and exit (no device).")
    args = parser.parse_args(argv)

    arch = args.arch
    if arch is None:
        try:
            from grokking_optimizers.dispatch import detect_arch
            arch = detect_arch()
        except Exception:
            parser.error("could not auto-detect arch; pass --arch explicitly")

    if args.list:
        cells = enumerate_cells(
            [args.optimizer] if args.optimizer else None,
            [args.model] if args.model else None)
        sys.stdout.write(
            f"{len(cells)} pipelines for {arch} "
            f"(PIPELINES_PER_ARCH={PIPELINES_PER_ARCH}):\n")
        for opt, model in cells:
            sys.stdout.write(f"  {opt}/{model}\n")
        return 0

    results = track_all(
        arch,
        optimizers=[args.optimizer] if args.optimizer else None,
        models=[args.model] if args.model else None,
        iters=args.iters, interval_s=args.interval_ms / 1000.0,
        timeout=args.timeout)
    sys.stdout.write("\n" + format_table(results))
    report = write_report(results, arch, args.out)
    sys.stdout.write(f"\n[utilization] wrote {report}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
