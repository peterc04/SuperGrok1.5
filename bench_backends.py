#!/usr/bin/env python3
"""bench_backends.py — Build and profile each backend launcher with arch-native tooling.

Outputs a single text report (default: ``bench_report.txt``) containing
the build log, smoke-test stdout/stderr, and a native profiler dump per
backend launch file. Progress is shown on stderr via a tqdm bar with ETA;
nothing is written to stdout except the final report path on completion.

Profilers (auto-selected from the active backend):

    CUDA / sm_90    → ncu (Nsight Compute) with ``--set full``,
                      ``--target-processes all``, source-line metrics,
                      ComputeWorkloadAnalysis / LaunchStats /
                      MemoryWorkloadAnalysis / SchedulerStats /
                      WarpStateStats / InstructionStats sections.

    HIP / gfx942    → rocprof-compute (preferred) or rocprofv2 / rocprof
                      with ``--hip-trace --hsa-trace --stats
                      --basenames on --timestamp on``.

    Pallas / TPU    → jax.profiler.start_trace / stop_trace in-process,
                      XLA HLO + op-level capture.

Build:
    Invokes ``pip install -e .`` with ``MAX_JOBS`` set to the detected
    CPU-core count so the underlying torch cpp_extension build uses
    ninja with maximal parallelism. Build logs (nvcc / hipcc / ninja
    diagnostics) are captured verbatim into the report.

Diagnostics enabled:

    CUDA: -Xptxas=-v --resource-usage --generate-line-info -DSG_VERBOSE=1
    HIP:  -Rpass-analysis=kernel-resource-usage -ggdb -DSG_VERBOSE=1

Usage:
    python bench_backends.py
    python bench_backends.py --backend cuda
    python bench_backends.py --output /tmp/r.txt --filter supergrok
    python bench_backends.py --skip-build   # reuse already-built _ops
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = REPO_ROOT / "bench_report.txt"
NCPUS = multiprocessing.cpu_count()

# (launcher_short_name, optimizer_class, source_module)
OPTIMIZERS: List[Tuple[str, str, str]] = [
    ("adamw",       "AdamW",       "grokking_optimizers"),
    ("grokadamw",   "GrokAdamW",   "grokking_optimizers"),
    ("grokfast",    "Grokfast",    "grokking_optimizers"),
    ("lion",        "Lion",        "grokking_optimizers"),
    ("looksam",     "LookSAM",     "grokking_optimizers"),
    ("muon",        "Muon",        "grokking_optimizers"),
    ("neuralgrok",  "NeuralGrok",  "grokking_optimizers"),
    ("prodigy",     "Prodigy",     "grokking_optimizers"),
    ("supergrok11", "SuperGrok11", "grokking_optimizers"),
    ("supergrok15", "SuperGrok15", "grokking_optimizers"),
    ("supergrok2",  "SuperGrok2",  "grokking_optimizers"),
]

NVCC_DIAG = [
    "-Xptxas=-v",
    "--resource-usage",
    "--generate-line-info",
    "-DSG_VERBOSE=1",
]

HIPCC_DIAG = [
    "-Rpass-analysis=kernel-resource-usage",
    "-ggdb",
    "-DSG_VERBOSE=1",
]

NCU_FLAGS = [
    "--set", "full",
    "--target-processes", "all",
    "--import-source", "yes",
    "--source-folders", str(REPO_ROOT / "csrc"),
    "--section", "ComputeWorkloadAnalysis",
    "--section", "LaunchStats",
    "--section", "MemoryWorkloadAnalysis",
    "--section", "SchedulerStats",
    "--section", "WarpStateStats",
    "--section", "InstructionStats",
    "--section", "Occupancy",
]

ROCPROF_FLAGS = [
    "--hip-trace",
    "--hsa-trace",
    "--stats",
    "--basenames", "on",
    "--timestamp", "on",
]


def detect_backend() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                return "hip"
            return "cuda"
    except Exception:
        pass
    try:
        import jax
        if any(d.platform == "tpu" for d in jax.devices()):
            return "pallas"
    except Exception:
        pass
    return "cuda"


def list_launchers(backend: str) -> List[Path]:
    if backend == "cuda":
        return sorted((REPO_ROOT / "csrc/backends/cuda/sm_90").glob("launch_*.cu"))
    if backend == "hip":
        cpp = list((REPO_ROOT / "csrc/backends/hip/gfx942").glob("launch_*.hip.cpp"))
        native = list((REPO_ROOT / "csrc/backends/hip/gfx942").glob("launch_*.hip"))
        return sorted(cpp + native)
    if backend == "pallas":
        return sorted((REPO_ROOT / "csrc/backends/pallas").glob("launch_*.py"))
    raise ValueError(f"Unknown backend: {backend}")


def section(report, title: str, char: str = "=") -> None:
    bar = char * 78
    report.write(f"\n{bar}\n {title}\n{bar}\n")


def build_extension(backend: str, report) -> bool:
    section(report, f"BUILD PHASE — ninja -j{NCPUS} (backend={backend})", char="=")
    env = os.environ.copy()
    env["MAX_JOBS"] = str(NCPUS)
    env["TORCH_CUDA_VERBOSE_BUILD"] = "1"
    env["NINJA_STATUS"] = "[%f/%t %es] "
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(NCPUS)
    if backend == "cuda":
        env["NVCC_APPEND_FLAGS"] = " ".join(NVCC_DIAG)
    elif backend == "hip":
        env["HIPCC_COMPILE_FLAGS_APPEND"] = " ".join(HIPCC_DIAG)

    cmd = [
        sys.executable, "-m", "pip", "install", "-e", ".",
        "--no-deps", "--force-reinstall", "-v",
    ]
    report.write(f"$ {' '.join(cmd)}\n  env.MAX_JOBS={NCPUS}\n\n")
    report.flush()

    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    report.write(proc.stdout or "")
    report.write(f"\n[build exit: {proc.returncode}]\n")
    return proc.returncode == 0


def make_smoke_script(opt_short: str, opt_class: str, source_module: str,
                      backend: str) -> str:
    device_pref = "tpu" if backend == "pallas" else "cuda"
    return textwrap.dedent(f"""
        import sys, traceback
        try:
            import torch
            from {source_module} import {opt_class}
            torch.manual_seed(0)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            p = torch.nn.Parameter(torch.randn(64, 64, device=device, dtype=torch.float32))
            opt = {opt_class}([p], lr=1e-3)
            (p * p).sum().backward()
            opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print('{opt_short}: smoke OK on', device)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    """).strip()


def write_temp_script(script: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        return f.name


def _child_env() -> dict:
    """Env for subprocess calls — ensures repo root is on PYTHONPATH so the
    smoke scripts can `import grokking_optimizers` without an install."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )
    return env


def run_capture(cmd: List[str], report, *, timeout: int = 900,
                env: Optional[dict] = None) -> int:
    if env is None:
        env = _child_env()
    report.write(f"\n$ {' '.join(cmd)}\n")
    report.flush()
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        report.write(proc.stdout or "")
        report.write(f"\n[exit: {proc.returncode}]\n")
        return proc.returncode
    except subprocess.TimeoutExpired as exc:
        report.write(f"\n[TIMEOUT after {timeout}s]\n")
        report.write((exc.stdout or "") if isinstance(exc.stdout, str) else "")
        return -1
    except FileNotFoundError as exc:
        report.write(f"\n[command not found: {exc}]\n")
        return -2


def profile_cuda(script: str, report, opt_name: str) -> None:
    if shutil.which("ncu") is None:
        report.write(f"  [skip] ncu not in PATH; cannot profile {opt_name}\n")
        smoke_script = write_temp_script(script)
        try:
            run_capture([sys.executable, smoke_script], report)
        finally:
            os.unlink(smoke_script)
        return

    smoke_script = write_temp_script(script)
    try:
        cmd = ["ncu"] + NCU_FLAGS + [sys.executable, smoke_script]
        run_capture(cmd, report)
    finally:
        os.unlink(smoke_script)


def profile_hip(script: str, report, opt_name: str) -> None:
    tool = (shutil.which("rocprof-compute")
            or shutil.which("rocprofv2")
            or shutil.which("rocprof"))
    smoke_script = write_temp_script(script)
    if tool is None:
        report.write("  [skip] no rocprof tool in PATH; running smoke only\n")
        try:
            run_capture([sys.executable, smoke_script], report)
        finally:
            os.unlink(smoke_script)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / f"{opt_name}.csv"
        try:
            cmd = [tool] + ROCPROF_FLAGS + ["-o", str(out_csv),
                                            sys.executable, smoke_script]
            run_capture(cmd, report)
            for f in sorted(Path(tmpdir).iterdir()):
                report.write(f"\n--- {f.name} ---\n")
                try:
                    report.write(f.read_text())
                except UnicodeDecodeError:
                    report.write(f"[binary {f.stat().st_size}B]\n")
        finally:
            os.unlink(smoke_script)


def profile_pallas(script: str, report, opt_name: str) -> None:
    """In-process JAX profiler trace; writes XLA traces to a tempdir."""
    wrapper = textwrap.dedent(f"""
        import sys, os, tempfile, traceback
        try:
            import jax.profiler
        except ImportError:
            print('[skip] jax.profiler not available', file=sys.stderr)
            sys.exit(0)
        with tempfile.TemporaryDirectory() as tdir:
            jax.profiler.start_trace(tdir)
            try:
                exec(compile({script!r}, '<smoke>', 'exec'))
            except Exception:
                traceback.print_exc()
            finally:
                jax.profiler.stop_trace()
            print('--- trace dir contents ---')
            for root, _, files in os.walk(tdir):
                for f in files:
                    fp = os.path.join(root, f)
                    size = os.path.getsize(fp)
                    print(f'{{f}}: {{size}} bytes')
    """).strip()
    run_capture([sys.executable, "-c", wrapper], report)


def opt_for_launcher(path: Path) -> Optional[Tuple[str, str, str]]:
    """Map csrc/.../launch_<NAME>(.cu|.hip.cpp|.hip|.py) → optimizer tuple."""
    stem = path.stem
    if stem.endswith(".hip"):
        stem = stem[: -len(".hip")]
    stem = stem.replace("launch_", "")
    for tup in OPTIMIZERS:
        if tup[0] == stem:
            return tup
    for tup in OPTIMIZERS:
        if stem.startswith(tup[0] + "_") or stem.startswith(tup[0]):
            return tup
    return None


# ===========================================================================
#  Real latency/throughput measurement (the timing half — runs on CPU now,
#  uses CUDA events when a GPU is present). Times the fused optimizer step
#  against an ATen-baseline reference path on realistic parameter shapes.
# ===========================================================================

# Realistic per-step workloads: (label, [tensor shapes]) — a transformer-ish
# mix of 2D weight matrices + 1D bias/norm vectors, several sizes. NOT 64x64.
BENCH_WORKLOADS: List[Tuple[str, List[Tuple[int, ...]]]] = [
    ("tiny",   [(256, 256), (256,)]),
    ("small",  [(1024, 1024), (1024,), (4096, 1024), (4096,)]),
    ("medium", [(4096, 4096), (4096,), (4096, 11008), (11008,)]),
    ("wide",   [(8192, 2048), (2048, 8192), (8192,), (2048,)]),
]


def _percentile(samples: List[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0, 100]) over a list of latencies."""
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _time_callable(fn: Callable[[], None], *, warmup: int, iters: int,
                   cuda: bool) -> List[float]:
    """Return a list of `iters` per-call latencies in milliseconds.

    Warms up `warmup` times first. On CUDA uses torch.cuda.Event timing (so the
    async launch queue is measured correctly); on CPU uses time.perf_counter.
    """
    import torch

    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()

    samples: List[float] = []
    if cuda:
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))   # ms
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)   # ms
    return samples


def _make_params(shapes: List[Tuple[int, ...]], device: str, dtype) -> list:
    import torch

    params = []
    for shape in shapes:
        p = torch.nn.Parameter(torch.randn(*shape, device=device, dtype=dtype))
        p.grad = torch.randn_like(p)
        params.append(p)
    return params


def _refresh_grads(params: list) -> None:
    import torch

    for p in params:
        p.grad = torch.randn_like(p)


# ATen baseline: stock torch.optim equivalents for the kernels we can compare
# against 1:1. The fused path is the project's optimizer; the baseline is the
# vanilla ATen implementation of the same math.
ATEN_BASELINE: Dict[str, Callable] = {}


def _init_aten_baselines() -> None:
    import torch

    ATEN_BASELINE["adamw"] = lambda params: torch.optim.AdamW(
        params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    # Lion landed in torch.optim in newer releases; fall back to None if absent.
    if hasattr(torch.optim, "Lion"):
        ATEN_BASELINE["lion"] = lambda params: torch.optim.Lion(  # type: ignore[attr-defined]
            params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)


def _build_fused_optimizer(opt_class: str, params: list):
    """Construct the project's fused optimizer; returns None (and the reason)
    if the extension isn't built / the class can't run on this device."""
    try:
        import grokking_optimizers as go
        cls = getattr(go, opt_class)
        return cls(params, lr=1e-3), None
    except Exception as exc:  # noqa: BLE001 — surface any construction failure
        return None, f"{type(exc).__name__}: {exc}"


def run_benchmark(args, report) -> dict:
    """Time fused vs ATen-baseline optimizer steps over realistic shapes.

    CPU paths run now; GPU paths only when CUDA is available. Returns a JSON-
    serialisable results dict (also written to `report` as text)."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        report.write(f"[bench] torch import failed: {exc}\n")
        return {"error": f"torch import failed: {exc}"}

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float32
    _init_aten_baselines()

    results: dict = {
        "schema": "bench_backends/v1",
        "generated": datetime.datetime.now().isoformat(),
        "device": device,
        "torch": torch.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "workloads": [w[0] for w in BENCH_WORKLOADS],
        "measurements": [],
    }

    section(report, f"BENCHMARK — device={device} warmup={args.warmup} "
                    f"iters={args.iters}", char="=")
    if not cuda:
        report.write(
            "[bench] no CUDA device — timing the CPU dispatch path. GPU latency "
            "numbers require a real accelerator (committed baselines are env-"
            "specific and intentionally not checked in).\n")

    opts = [o for o in OPTIMIZERS
            if not args.filter or args.filter in o[0]]

    for opt_short, opt_class, _src in opts:
        for wl_name, shapes in BENCH_WORKLOADS:
            entry: dict = {
                "optimizer": opt_short,
                "workload": wl_name,
                "n_params": len(shapes),
                "n_elements": int(sum(_numel(s) for s in shapes)),
            }

            # ── fused (project) path ──
            params = _make_params(shapes, device, dtype)
            opt, why = _build_fused_optimizer(opt_class, params)
            if opt is None:
                entry["fused"] = {"skipped": why}
            else:
                def _fused_step(_opt=opt, _params=params):
                    _refresh_grads(_params)
                    _opt.step()

                try:
                    lat = _time_callable(_fused_step, warmup=args.warmup,
                                         iters=args.iters, cuda=cuda)
                    entry["fused"] = _summarize(lat, entry["n_elements"])
                except Exception as exc:  # noqa: BLE001
                    entry["fused"] = {"error": f"{type(exc).__name__}: {exc}"}

            # ── ATen baseline (only where a 1:1 stock optimizer exists) ──
            base_factory = ATEN_BASELINE.get(opt_short)
            if base_factory is not None:
                bparams = _make_params(shapes, device, dtype)
                try:
                    bopt = base_factory(bparams)

                    def _base_step(_opt=bopt, _params=bparams):
                        _refresh_grads(_params)
                        _opt.step()

                    blat = _time_callable(_base_step, warmup=args.warmup,
                                          iters=args.iters, cuda=cuda)
                    entry["aten_baseline"] = _summarize(blat,
                                                        entry["n_elements"])
                    # speedup = baseline_median / fused_median (>1 => fused wins)
                    f = entry.get("fused", {})
                    if "median_ms" in f and f["median_ms"] > 0:
                        entry["speedup_vs_aten"] = (
                            entry["aten_baseline"]["median_ms"] / f["median_ms"])
                except Exception as exc:  # noqa: BLE001
                    entry["aten_baseline"] = {"error": f"{type(exc).__name__}: {exc}"}

            results["measurements"].append(entry)
            _write_bench_row(report, entry)

    return results


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _summarize(lat_ms: List[float], n_elements: int) -> dict:
    median = statistics.median(lat_ms)
    out = {
        "median_ms": median,
        "p90_ms": _percentile(lat_ms, 90),
        "min_ms": min(lat_ms),
        "samples": len(lat_ms),
    }
    # Throughput: elements updated per second at the median latency.
    if median > 0:
        out["throughput_melem_per_s"] = (n_elements / (median / 1e3)) / 1e6
    return out


def _write_bench_row(report, entry: dict) -> None:
    fused = entry.get("fused", {})
    base = entry.get("aten_baseline")
    line = (f"  {entry['optimizer']:<12} {entry['workload']:<7} "
            f"elems={entry['n_elements']:>11,} ")
    if "median_ms" in fused:
        line += (f"fused med={fused['median_ms']:.4f}ms "
                 f"p90={fused['p90_ms']:.4f}ms "
                 f"thr={fused.get('throughput_melem_per_s', 0):.1f}Melem/s")
    elif "skipped" in fused:
        line += f"fused [skip: {fused['skipped']}]"
    elif "error" in fused:
        line += f"fused [err: {fused['error']}]"
    if base and "median_ms" in base:
        line += f" | aten med={base['median_ms']:.4f}ms"
        if "speedup_vs_aten" in entry:
            line += f" speedup={entry['speedup_vs_aten']:.2f}x"
    report.write(line + "\n")
    report.flush()


def _run_bench_mode(args) -> int:
    """Bench mode entry: build (optional), time fused vs ATen, emit text + JSON."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend = args.backend or detect_backend()
    with open(args.output, "w") as report:
        report.write("# SuperGrok optimizer bench (timing mode)\n")
        report.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
        report.write(f"# Backend:   {backend}\n")

        if not args.skip_build and backend in ("cuda", "hip"):
            try:
                import torch
                if torch.cuda.is_available():
                    build_extension(backend, report)
                else:
                    report.write("\n[bench] no accelerator — skipping build; "
                                 "timing the importable dispatch path.\n")
            except Exception:  # noqa: BLE001
                report.write("\n[bench] torch unavailable for build phase.\n")
        else:
            report.write("\n[bench] build phase skipped.\n")
        report.flush()

        results = run_benchmark(args, report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as jf:
            json.dump(results, jf, indent=2, sort_keys=True)
        sys.stderr.write(f"[bench] JSON written: {args.json_out}\n")

    sys.stdout.write(f"{args.output}\n")
    return 0 if "error" not in results else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=("cuda", "hip", "pallas"),
                        default=None,
                        help="Force a backend (default: auto-detect)")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_REPORT,
                        help=f"Report file (default: {DEFAULT_REPORT.name})")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip the pip-install build phase")
    parser.add_argument("--filter", default="",
                        help="Only run launchers whose name contains this substring")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Per-command timeout in seconds (default 900)")
    parser.add_argument("--mode", choices=("bench", "profile"), default="bench",
                        help="bench: real warmup+timed latency/throughput "
                             "measurement vs an ATen baseline (default). "
                             "profile: the legacy ncu/rocprof/jax profiler dump.")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations before timing (bench mode)")
    parser.add_argument("--iters", type=int, default=50,
                        help="Timed iterations per (optimizer, workload) "
                             "(bench mode)")
    parser.add_argument("--json", dest="json_out", type=Path, default=None,
                        help="Write the bench results to this JSON file "
                             "(committed-baseline format; numbers are env-"
                             "specific so don't commit GPU numbers from CI)")
    args = parser.parse_args()

    if args.mode == "bench":
        return _run_bench_mode(args)

    backend = args.backend or detect_backend()
    launchers = list_launchers(backend)
    if args.filter:
        launchers = [p for p in launchers if args.filter in p.name]

    if not launchers:
        sys.stderr.write(
            f"No launchers matched (backend={backend}, filter={args.filter!r})\n"
        )
        return 1

    try:
        from tqdm import tqdm
        progress = lambda it: tqdm(it, desc=f"Profile {backend}",
                                    file=sys.stderr, unit="launcher",
                                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                               "[{elapsed}<{remaining}, {rate_fmt}]")
    except ImportError:
        # Built-in ETA fallback: emits "[i/N elapsed=Xs eta=Ys] <name>" per launcher
        import time as _t

        def progress(it):
            items = list(it)
            n = len(items)
            t0 = _t.monotonic()
            for i, x in enumerate(items, 1):
                elapsed = _t.monotonic() - t0
                eta = (elapsed / max(i - 1, 1)) * (n - i + 1) if i > 1 else 0.0
                sys.stderr.write(
                    f"\r[{i}/{n} elapsed={elapsed:5.1f}s eta={eta:5.1f}s] "
                    f"{getattr(x, 'name', str(x)):40s}"
                )
                sys.stderr.flush()
                yield x
            sys.stderr.write("\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiler_name = {
        "cuda": "ncu (Nsight Compute)",
        "hip":  "rocprof-compute / rocprofv2",
        "pallas": "jax.profiler",
    }[backend]

    with open(args.output, "w") as report:
        report.write("# SuperGrok backend bench report\n")
        report.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
        report.write(f"# Backend:   {backend}\n")
        report.write(f"# CPU cores: {NCPUS} (used for ninja -j)\n")
        report.write(f"# Profiler:  {profiler_name}\n")
        report.write(f"# Launchers: {len(launchers)}\n")
        report.write(f"# Repo:      {REPO_ROOT}\n")

        if not args.skip_build:
            ok = build_extension(backend, report)
            if not ok:
                report.write(
                    "\n[!] Build returned non-zero — continuing to per-launcher "
                    "phase regardless (profiler runs may still surface errors).\n"
                )
        else:
            report.write("\n[skip-build] reusing already-installed _ops\n")

        report.flush()

        for path in progress(launchers):
            tup = opt_for_launcher(path)
            section(report, f"LAUNCHER: {path.relative_to(REPO_ROOT)}", char="-")
            if tup is None:
                report.write(f"[!] No optimizer mapping for {path.name}; skipping\n")
                continue
            opt_short, opt_class, source_mod = tup
            report.write(f"  optimizer:  {opt_class}\n")
            report.write(f"  source:     {source_mod}\n")
            script = make_smoke_script(opt_short, opt_class, source_mod, backend)
            report.write(f"  smoke script:\n{textwrap.indent(script, '    ')}\n")

            if backend == "cuda":
                profile_cuda(script, report, opt_short)
            elif backend == "hip":
                profile_hip(script, report, opt_short)
            else:
                profile_pallas(script, report, opt_short)

            report.flush()

    sys.stdout.write(f"{args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
