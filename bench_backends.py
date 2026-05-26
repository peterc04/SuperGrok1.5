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
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

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
        report.write(f"  [skip] no rocprof tool in PATH; running smoke only\n")
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
    args = parser.parse_args()

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
