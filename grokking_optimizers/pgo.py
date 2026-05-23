"""grokking_optimizers.pgo — Profile-Guided Optimisation driver.

Three-phase loop that piggybacks on ``compile.py``:

  1. **Instrument**: AOT-build with ``-fprofile-generate=<dir>`` so the
     host code is instrumented; ``-Xcompiler -fprofile-generate`` for
     NVCC and ``-fprofile-generate`` for HIPCC.
  2. **Collect**: run a workload script that imports the instrumented
     ``.so`` and exercises the optimizer for N steps; this populates
     ``<dir>/*.gcda`` (host) and ``<dir>/*.profraw`` (NVCC/HIPCC where
     supported).
  3. **Use**: AOT-rebuild with ``-fprofile-use=<dir>`` and let JIT
     autotune run on the PGO binary.

JIT autotune **must run on the PGO binary** (do not reuse a non-PGO
sweep) because instruction selection and inlining decisions change
with the profile data.

Cache key
=========
The PGO data is hashed by ``(workload_file_sha256, steps)``. Whenever
the user changes the workload, the hash changes and the PGO cache is
invalidated. ``is_aot_fresh()`` includes ``pgo_enabled`` so a non-PGO
build is never confused with a PGO build.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Flag manipulation
# ---------------------------------------------------------------------------

def instrument_flags(
    arch: str,
    profile_dir: Path,
    host_cflags: List[str],
    device_cflags: List[str],
    ldflags: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Append profile-generate flags. ``profile_dir`` is the .gcda output dir."""
    profile_dir = Path(profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    h = list(host_cflags) + [
        f"-fprofile-generate={profile_dir}",
        # The default profile name would collide across TUs; per-pid is safer.
        "-fprofile-update=atomic",
    ]
    d: List[str] = list(device_cflags)
    if arch == "sm_90":
        d += [
            "-Xcompiler", f"-fprofile-generate={profile_dir}",
            "-Xcompiler", "-fprofile-update=atomic",
        ]
    elif arch == "gfx942":
        d += [
            f"-fprofile-generate={profile_dir}",
            "-fprofile-update=atomic",
        ]
    l = list(ldflags) + [f"-fprofile-generate={profile_dir}"]
    return h, d, l


def use_flags(
    arch: str,
    profile_dir: Path,
    host_cflags: List[str],
    device_cflags: List[str],
    ldflags: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Append profile-use flags."""
    profile_dir = Path(profile_dir).resolve()

    h = list(host_cflags) + [
        f"-fprofile-use={profile_dir}",
        "-fprofile-correction",
    ]
    d: List[str] = list(device_cflags)
    if arch == "sm_90":
        d += [
            "-Xcompiler", f"-fprofile-use={profile_dir}",
            "-Xcompiler", "-fprofile-correction",
        ]
    elif arch == "gfx942":
        d += [
            f"-fprofile-use={profile_dir}",
            "-fprofile-correction",
        ]
    l = list(ldflags) + [f"-fprofile-use={profile_dir}"]
    return h, d, l


# ---------------------------------------------------------------------------
# Workload runner
# ---------------------------------------------------------------------------

def hash_workload(workload_script: Path, steps: int) -> str:
    """SHA-256 of (workload file contents, step count)."""
    h = hashlib.sha256()
    try:
        h.update(Path(workload_script).read_bytes())
    except OSError:
        h.update(b"<missing-workload>")
    h.update(b"\0")
    h.update(str(int(steps)).encode("ascii"))
    return h.hexdigest()


def collect_workload(
    workload_script: Path,
    *,
    so_path: Path,
    opt_class: str,
    model: str,
    arch: str,
    profile_dir: Path,
    steps: int = 1000,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 600,
    report=None,
) -> bool:
    """Run the workload subprocess and confirm profile files appear.

    Returns ``True`` if at least one ``.gcda`` (or any file) was created
    in ``profile_dir`` and the workload exited cleanly.
    """
    workload_script = Path(workload_script).resolve()
    if not workload_script.exists():
        if report:
            report.write(f"  [pgo collect] workload not found: {workload_script}\n")
        return False
    profile_dir = Path(profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_size for p in profile_dir.glob("**/*") if p.is_file()}

    sub_env = (env or os.environ).copy()
    sub_env["LLVM_PROFILE_FILE"] = str(profile_dir / "default_%p.profraw")
    sub_env["GCOV_PREFIX"] = str(profile_dir)

    cmd = [
        sys.executable, str(workload_script),
        "--so",   str(so_path),
        "--opt",  opt_class,
        "--model", model,
        "--arch", arch,
        "--steps", str(int(steps)),
    ]
    if report:
        report.write(f"\n$ {' '.join(cmd)}\n")
        report.flush()
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout, env=sub_env,
        )
    except subprocess.TimeoutExpired:
        if report:
            report.write(f"  [pgo collect] TIMEOUT after {timeout}s\n")
        return False
    elapsed = time.monotonic() - t0
    if report:
        report.write(proc.stdout or "")
        report.write(f"\n[exit {proc.returncode} in {elapsed:.1f}s]\n")
    if proc.returncode != 0:
        return False

    after = {p: p.stat().st_size for p in profile_dir.glob("**/*") if p.is_file()}
    new_or_grown = [p for p, sz in after.items()
                    if before.get(p, -1) != sz]
    if not new_or_grown:
        if report:
            report.write(
                f"  [pgo collect] no profile files appeared under {profile_dir}\n")
        return False
    if report:
        report.write(
            f"  [pgo collect] {len(new_or_grown)} profile file(s) updated\n")
    return True


__all__ = [
    "instrument_flags",
    "use_flags",
    "hash_workload",
    "collect_workload",
]
