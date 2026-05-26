"""grokking_optimizers.profile — Standalone arch-native profile capture.

Companion to ``grokking_optimizers.compile``: where ``compile`` builds the
kernels, ``profile`` runs the matching native profiler against an already-
built artefact (or a launcher source file used as an identifier) and
writes a single diagnostic report.

Profilers, auto-selected from the target arch:

    sm_90    -> ncu (Nsight Compute) ``--set full --target-processes all
                --import-source yes --source-folders csrc`` + 7 sections
                (ComputeWorkloadAnalysis, LaunchStats,
                MemoryWorkloadAnalysis, SchedulerStats, WarpStateStats,
                InstructionStats, Occupancy).

    gfx942   -> rocprof-compute >> rocprofv2 >> rocprof (first available on
                PATH) with ``--hip-trace --hsa-trace --stats --basenames on
                --timestamp on``. CSV/JSON outputs are inlined into the
                report.

    tpu_v5p  -> ``jax.profiler.start_trace / stop_trace`` in-process; XLA
                HLO + op-level capture. The trace directory contents are
                summarised in the report.

Path inference (when ``--path`` is given):

    csrc/backends/cuda/sm_90/launch_<opt>.cu          -> sm_90, <opt>
    csrc/backends/hip/gfx942/launch_<opt>.hip.cpp     -> gfx942, <opt>
    csrc/backends/hip/gfx942/launch_<opt>.hip         -> gfx942, <opt>
    csrc/backends/pallas/launch_<opt>.py              -> tpu_v5p, <opt>
    build/compiled/grokking_compiled_<opt>_<model>_<arch>/*.so
                                                       -> arch, opt, model
    any other .py                                     -> need --arch

The profiler runs the standard smoke (import the optimizer class, run one
``opt.step()``) — the path is the *identifier* of what to profile; the
actual kernels exercised come from the installed ``grokking_optimizers._ops``
(or, for tpu_v5p, the matching ``launch_*.py``).

Usage (CLI):
    python -m grokking_optimizers.profile \\
        --path csrc/backends/cuda/sm_90/launch_supergrok2.cu

    python -m grokking_optimizers.profile \\
        --optimizer supergrok2 --model mamba --arch sm_90

Usage (importable):
    from grokking_optimizers.profile import profile
    report = profile(path="csrc/backends/cuda/sm_90/launch_supergrok2.cu")
    # or
    report = profile(optimizer="lion", arch="gfx942")
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
NCPUS = multiprocessing.cpu_count()

OPTIMIZERS: Tuple[str, ...] = (
    "adamw", "grokadamw", "grokfast", "lion", "looksam", "muon",
    "neuralgrok", "prodigy", "supergrok11", "supergrok15", "supergrok2",
)

MODELS: Tuple[str, ...] = ("mamba", "decoder", "vit")

ARCHES: Tuple[str, ...] = ("sm_90", "gfx942", "tpu_v5p")

OPT_CLASS = {
    "adamw":       "AdamW",
    "grokadamw":   "GrokAdamW",
    "grokfast":    "Grokfast",
    "lion":        "Lion",
    "looksam":     "LookSAM",
    "muon":        "Muon",
    "neuralgrok":  "NeuralGrok",
    "prodigy":     "Prodigy",
    "supergrok11": "SuperGrok11",
    "supergrok15": "SuperGrok15",
    "supergrok2":  "SuperGrok2",
}

ARCH_INFO = {
    "sm_90": {
        "vendor": "cuda",
        "subdir": "cuda/sm_90",
        "launcher_glob": ("launch_*.cu",),
        "model_glob":    ("*.cu",),
        "macro":         "SG_BUILD_ARCH_SM90",
        "host_define":   "WITH_CUDA",
    },
    "gfx942": {
        "vendor": "hip",
        "subdir": "hip/gfx942",
        "launcher_glob": ("launch_*.hip.cpp", "launch_*.hip"),
        "model_glob":    ("*.hip.cpp",),
        "macro":         "SG_BUILD_ARCH_GFX942",
        "host_define":   "WITH_HIP",
    },
    "tpu_v5p": {
        "vendor": "pallas",
        "subdir": "pallas",
        "launcher_glob": ("launch_*.py",),
        "model_glob":    (),
        "macro":         "SG_BUILD_ARCH_TPU_V5P",
        "host_define":   None,
    },
}

NCU_FLAGS: List[str] = [
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

ROCPROF_FLAGS: List[str] = [
    "--hip-trace", "--hsa-trace", "--stats",
    "--basenames", "on", "--timestamp", "on",
]


# ---------------------------------------------------------------------------
# Shared subprocess / env helpers (re-used by grokking_optimizers.compile)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def env_overlay(**kw):
    """Temporarily overlay environment variables; restore on exit."""
    saved = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def child_env() -> dict:
    """Subprocess env with REPO_ROOT prepended to PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )
    return env


def run_capture(cmd: List[str], report, timeout: int = 900) -> int:
    """Run a subprocess; pipe stdout+stderr into the report file."""
    report.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    report.flush()
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        report.write(proc.stdout or "")
        report.write(f"\n[exit: {proc.returncode}]\n")
        return proc.returncode
    except subprocess.TimeoutExpired:
        report.write(f"\n[TIMEOUT after {timeout}s]\n")
        return -1
    except FileNotFoundError as exc:
        report.write(f"\n[command not found: {exc}]\n")
        return -2


def make_progress(total: int, title: str) -> Tuple[Callable[[str], None],
                                                   Callable[[], None]]:
    """Return (step, close). Uses tqdm if available; falls back to a
    plain ``[i/N elapsed=Xs eta=Ys]`` stderr line."""
    try:
        from tqdm import tqdm
        bar = tqdm(
            total=total, desc=title, file=sys.stderr, unit="phase",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )

        def step(label: str) -> None:
            bar.set_postfix_str(label, refresh=True)
            bar.update(1)

        def close() -> None:
            bar.close()

        return step, close
    except ImportError:
        t0 = time.monotonic()
        done = [0]

        def step(label: str) -> None:
            done[0] += 1
            elapsed = time.monotonic() - t0
            eta = (elapsed / done[0]) * (total - done[0]) if done[0] else 0.0
            sys.stderr.write(
                f"\r[{done[0]}/{total} elapsed={elapsed:5.1f}s "
                f"eta={eta:5.1f}s] {label:40s}"
            )
            sys.stderr.flush()

        def close() -> None:
            sys.stderr.write("\n")

        return step, close


# ---------------------------------------------------------------------------
# Smoke-script generator (one opt.step() in the active backend)
# ---------------------------------------------------------------------------

def smoke_script(optimizer: str, model: str, arch: str) -> str:
    cls = OPT_CLASS[optimizer]
    return textwrap.dedent(f"""\
        import sys, traceback
        try:
            import torch
            from grokking_optimizers import {cls}
            torch.manual_seed(0)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            p = torch.nn.Parameter(
                torch.randn(64, 64, device=device, dtype=torch.float32))
            (p * p).sum().backward()
            opt = {cls}([p], lr=1e-3)
            opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print('smoke OK ({optimizer}/{model}/{arch})')
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    """)


def write_temp_script(body: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    os.write(fd, body.encode())
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# Path inference: launcher source / .so / generic .py -> (opt, model, arch)
# ---------------------------------------------------------------------------

def infer_from_path(path: Path) -> dict:
    """Return a dict with keys ``optimizer``, ``model``, ``arch`` (each
    optionally ``None``) inferred from ``path``."""
    path = Path(path).resolve()
    parts = path.parts
    arch: Optional[str] = None
    optimizer: Optional[str] = None
    model: Optional[str] = None

    if "sm_90" in parts:
        arch = "sm_90"
    elif "gfx942" in parts:
        arch = "gfx942"
    elif "pallas" in parts:
        arch = "tpu_v5p"

    name = path.name

    if name.startswith("launch_"):
        stem = name[len("launch_"):]
        for ext in (".hip.cpp", ".hip.h", ".cuh", ".cu", ".hip", ".py"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        if stem in OPTIMIZERS:
            optimizer = stem

    elif name.startswith("grokking_compiled_"):
        bare = name[len("grokking_compiled_"):]
        for sep in (".cpython", ".abi3", ".so"):
            i = bare.find(sep)
            if i >= 0:
                bare = bare[:i]
                break
        for a in ARCHES:
            suffix = "_" + a
            if bare.endswith(suffix):
                arch = a
                trimmed = bare[: -len(suffix)]
                for m in MODELS:
                    if trimmed.endswith("_" + m):
                        model = m
                        opt_candidate = trimmed[: -len("_" + m)]
                        if opt_candidate in OPTIMIZERS:
                            optimizer = opt_candidate
                        break
                break

    return {"optimizer": optimizer, "model": model, "arch": arch}


# ---------------------------------------------------------------------------
# Per-arch profile drivers
# ---------------------------------------------------------------------------

def profile_cuda(optimizer: str, model: str, arch: str, report,
                 timeout: int = 900, ncu_flags: Optional[List[str]] = None
                 ) -> int:
    flags = ncu_flags if ncu_flags is not None else NCU_FLAGS
    script = write_temp_script(smoke_script(optimizer, model, arch))
    try:
        if shutil.which("ncu") is None:
            report.write("  [profile] ncu not in PATH; running smoke only.\n")
            return run_capture([sys.executable, script], report,
                               timeout=timeout)
        cmd = ["ncu"] + flags + [sys.executable, script]
        return run_capture(cmd, report, timeout=timeout)
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def profile_hip(optimizer: str, model: str, arch: str, report,
                timeout: int = 900,
                rocprof_flags: Optional[List[str]] = None) -> int:
    flags = rocprof_flags if rocprof_flags is not None else ROCPROF_FLAGS
    script = write_temp_script(smoke_script(optimizer, model, arch))
    tool = (shutil.which("rocprof-compute")
            or shutil.which("rocprofv2")
            or shutil.which("rocprof"))
    try:
        if tool is None:
            report.write(
                "  [profile] no rocprof tool in PATH; running smoke only.\n")
            return run_capture([sys.executable, script], report,
                               timeout=timeout)
        with tempfile.TemporaryDirectory() as tdir:
            out_csv = Path(tdir) / f"{optimizer}.csv"
            cmd = [tool] + flags + [
                "-o", str(out_csv), sys.executable, script,
            ]
            rc = run_capture(cmd, report, timeout=timeout)
            for f in sorted(Path(tdir).iterdir()):
                report.write(f"\n--- {f.name} ---\n")
                try:
                    report.write(f.read_text())
                except UnicodeDecodeError:
                    report.write(f"[binary, {f.stat().st_size}B]\n")
            return rc
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def profile_pallas(optimizer: str, model: str, arch: str, report,
                   timeout: int = 900) -> int:
    body = smoke_script(optimizer, model, arch)
    wrapper = textwrap.dedent(f"""\
        import sys, os, tempfile, traceback
        try:
            import jax.profiler
        except ImportError:
            print('[skip] jax.profiler not available', file=sys.stderr)
            sys.exit(0)
        with tempfile.TemporaryDirectory() as tdir:
            jax.profiler.start_trace(tdir)
            try:
                exec(compile({body!r}, '<smoke>', 'exec'))
            except Exception:
                traceback.print_exc()
            finally:
                jax.profiler.stop_trace()
            print('--- jax.profiler trace contents ---')
            for root, _, files in os.walk(tdir):
                for f in files:
                    fp = os.path.join(root, f)
                    print(f'{{f}}: {{os.path.getsize(fp)}} bytes')
    """)
    return run_capture([sys.executable, "-c", wrapper], report,
                       timeout=timeout)


def _dispatch_profile(optimizer: str, model: str, arch: str, report,
                      timeout: int = 900) -> int:
    vendor = ARCH_INFO[arch]["vendor"]
    if vendor == "cuda":
        return profile_cuda(optimizer, model, arch, report, timeout=timeout)
    if vendor == "hip":
        return profile_hip(optimizer, model, arch, report, timeout=timeout)
    return profile_pallas(optimizer, model, arch, report, timeout=timeout)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

class _DebugTee:
    """File-like wrapper that mirrors every write to a secondary stream.

    Used when debug=True to stream the profile report to stderr in real
    time without changing existing report.write(...) call sites.
    """
    __slots__ = ("primary", "mirror")

    def __init__(self, primary, mirror):
        self.primary = primary
        self.mirror = mirror

    def write(self, s):
        self.primary.write(s)
        self.primary.flush()
        try:
            self.mirror.write(s)
            self.mirror.flush()
        except Exception:
            pass
        return len(s) if isinstance(s, str) else 0

    def flush(self):
        self.primary.flush()
        try:
            self.mirror.flush()
        except Exception:
            pass


def profile(
    *,
    path: Optional[Path] = None,
    optimizer: Optional[str] = None,
    model: Optional[str] = None,
    arch: Optional[str] = None,
    report_path: Optional[Path] = None,
    timeout: int = 900,
    debug: bool = False,
) -> Path:
    """Run the arch-native profiler against the target and write a report.

    Either pass ``path`` (a launcher source, a built ``.so`` from
    ``grokking_optimizers.compile``, or any Python script — with ``--arch``
    in the last case) **or** pass the ``(optimizer, model, arch)`` triple
    explicitly. Returns the path to the written report.

    ``debug=True`` mirrors the report to stderr in real time so every
    ncu/rocprof/jax.profiler subcommand and its output is visible live.
    """
    inferred = {"optimizer": None, "model": None, "arch": None}
    path_obj: Optional[Path] = None
    if path is not None:
        path_obj = Path(path).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(path_obj)
        inferred = infer_from_path(path_obj)

    optimizer = optimizer or inferred["optimizer"]
    model = model or inferred["model"] or "mamba"
    arch = arch or inferred["arch"]

    if optimizer is None:
        raise ValueError(
            "Could not determine optimizer; pass --optimizer or a path "
            "that includes a known launcher name (e.g. "
            "csrc/backends/cuda/sm_90/launch_supergrok2.cu).")
    if arch is None:
        raise ValueError(
            "Could not determine arch; pass --arch or a path under "
            "csrc/backends/{cuda/sm_90, hip/gfx942, pallas}/.")
    if optimizer not in OPTIMIZERS:
        raise ValueError(
            f"optimizer={optimizer!r} not in {list(OPTIMIZERS)}")
    if model not in MODELS:
        raise ValueError(f"model={model!r} not in {list(MODELS)}")
    if arch not in ARCHES:
        raise ValueError(f"arch={arch!r} not in {list(ARCHES)}")

    if report_path is None:
        out_dir = REPO_ROOT / "build" / "profiled"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"profile_{optimizer}_{model}_{arch}.txt"
    else:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

    if debug:
        bar = "=" * 72
        _ts = datetime.datetime.now().isoformat()
        sys.stderr.write(
            f"\n{bar}\n"
            f"[debug] grokking_optimizers.profile starting at {_ts}\n"
            f"[debug] target:   {optimizer}/{model}/{arch} "
            f"(vendor={ARCH_INFO[arch]['vendor']})\n"
            f"[debug] report:   {report_path}\n"
            f"[debug] timeout:  {timeout}s\n"
            f"{bar}\n\n"
        )
        sys.stderr.flush()

    step, close = make_progress(2, f"profile {optimizer}/{model}/{arch}")
    try:
        with open(report_path, "w") as report_file:
            report = _DebugTee(report_file, sys.stderr) if debug else report_file
            report.write("# grokking_optimizers.profile — capture report\n")
            report.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
            report.write(f"# Optimizer: {optimizer}\n")
            report.write(f"# Model:     {model}\n")
            report.write(f"# Arch:      {arch} "
                         f"(vendor={ARCH_INFO[arch]['vendor']})\n")
            if path_obj is not None:
                report.write(f"# Path:      {path_obj}\n")
            report.write(f"# CPU cores: {NCPUS}\n")

            tool_name = {
                "cuda":   "ncu (Nsight Compute)",
                "hip":    "rocprof-compute / rocprofv2 / rocprof",
                "pallas": "jax.profiler",
            }[ARCH_INFO[arch]["vendor"]]
            report.write(f"# Profiler:  {tool_name}\n")
            step("setup")

            report.write("\n--- PROFILE PASS ---\n")
            _dispatch_profile(optimizer, model, arch, report, timeout=timeout)
            step("profile")
    finally:
        close()

    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m grokking_optimizers.profile",
        description="Run the arch-native profiler against a kernel target.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path", "-p", type=Path, default=None,
        help="Path to a launcher source, a compile.py-produced .so, or a "
             "Python script. Optimizer / arch are inferred when possible.")
    parser.add_argument(
        "--optimizer", "-O", choices=OPTIMIZERS, default=None,
        help="Optimizer name (overrides inference)")
    parser.add_argument(
        "--model", "-M", choices=MODELS, default=None,
        help="Model name (default: mamba)")
    parser.add_argument(
        "--arch", "-A", choices=ARCHES, default=None,
        help="Target arch (overrides inference; required when --path is "
             "an arbitrary .py)")
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Report file path (default: build/profiled/profile_<O>_<M>_<A>.txt)")
    parser.add_argument(
        "--timeout", type=int, default=900,
        help="Per-command timeout in seconds (default 900)")
    parser.add_argument(
        "--debug", action="store_true",
        help="Mirror the full profile report to stderr in real time so "
             "every ncu/rocprof/jax.profiler subcommand and its output is "
             "visible live.")
    args = parser.parse_args(argv)

    if args.path is None and args.optimizer is None:
        parser.error("either --path or --optimizer must be provided")

    report = profile(
        path=args.path,
        optimizer=args.optimizer,
        model=args.model,
        arch=args.arch,
        report_path=args.report,
        timeout=args.timeout,
        debug=args.debug,
    )
    sys.stdout.write(f"{report}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
