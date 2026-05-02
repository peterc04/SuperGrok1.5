#!/usr/bin/env python3
"""Verification gate runner — Phase 2 / Phase 3 release gates.

Runs the build + test matrix that gates a release candidate and prints
a final pass/fail summary. Exits 0 iff every required gate passes; any
failure short-circuits to a non-zero exit code so the gate is safe to
chain into CI.

Usage
-----
    # Run the full Phase 3 gate (default — covers all phases up to 3):
    python scripts/verification_gate.py

    # Phase 2 only (build + core optimizer tests; no model bindings):
    python scripts/verification_gate.py --phase 2

    # Phase 3 explicitly (adds models_sm_90, cross-arch, optional cutlass):
    python scripts/verification_gate.py --phase 3

    # Skip the build step (use the wheel already on PYTHONPATH):
    python scripts/verification_gate.py --skip-build

Gate composition
----------------
Phase 2:
    build               — ``bash build.sh`` (stderr → ``build.log``)
    test_supergrok2     — pytest tests/test_supergrok2.py
    test_cutlass_parity — pytest tests/test_cutlass_parity.py
                          (only if WITH_CUTLASS=1 in env; else N/A)

Phase 3 (everything in Phase 2 plus):
    test_models_sm_90               — pytest tests/test_models_sm_90.py
    test_cross_arch_agreement       — pytest tests/test_cross_arch_agreement.py
                                      (filtered to ``-k sm_90``)

Output format mirrors the rest of the project's gate scripts:

    Verification gate:
      build:                              PASS / FAIL
      test_supergrok2:                    PASS (N) / FAIL
      test_models_sm_90:                  PASS (N) / FAIL
      test_cross_arch_agreement (sm_90):  PASS (N) / FAIL
      test_cutlass_parity:                PASS (N) / FAIL / N/A
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# ── repo layout ─────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_LOG = REPO_ROOT / "build.log"


# ── result types ────────────────────────────────────────────────────


@dataclass
class GateResult:
    """One row of the final summary table."""

    name: str           # display label
    status: str         # one of PASS, FAIL, N/A
    count: Optional[int] = None   # passing test count if applicable
    detail: str = ""    # short note appended to the row

    @property
    def ok(self) -> bool:
        # N/A is non-blocking (e.g. cutlass without WITH_CUTLASS=1).
        return self.status in ("PASS", "N/A")

    def render(self, label_width: int) -> str:
        label = f"  {self.name}:".ljust(label_width + 4)
        if self.status == "PASS" and self.count is not None:
            body = f"PASS ({self.count})"
        else:
            body = self.status
        if self.detail:
            body += f"  [{self.detail}]"
        return f"{label}{body}"


# ── pytest output parser ────────────────────────────────────────────

# Matches the tail of a typical pytest run. We pull "passed" out of
# either of these forms, both of which pytest emits depending on
# version + plugins:
#     ====== 4 passed in 0.21s ======
#     ====== 3 passed, 1 skipped in 0.05s ======
_PYTEST_PASS_RE = re.compile(r"(\d+)\s+passed")
_PYTEST_FAIL_RE = re.compile(r"(\d+)\s+failed")
_PYTEST_ERROR_RE = re.compile(r"(\d+)\s+error")


def _parse_pytest_counts(stdout: str) -> tuple[int, int, int]:
    """Return ``(passed, failed, errored)`` counts from pytest stdout."""
    passed = sum(int(m.group(1)) for m in _PYTEST_PASS_RE.finditer(stdout))
    failed = sum(int(m.group(1)) for m in _PYTEST_FAIL_RE.finditer(stdout))
    errored = sum(int(m.group(1)) for m in _PYTEST_ERROR_RE.finditer(stdout))
    return passed, failed, errored


# ── individual gate runners ─────────────────────────────────────────


def run_build() -> GateResult:
    """Run ``bash build.sh`` from repo root, send stderr to ``build.log``.

    Returns PASS on exit code 0, FAIL otherwise. The full stderr is
    always written to ``build.log`` regardless of outcome so a failed
    build can be inspected without re-running.
    """
    print("─── build ──────────────────────────────────────────────")
    print(f"  $ bash build.sh   (stderr → {BUILD_LOG.name})")

    build_script = REPO_ROOT / "build.sh"
    if not build_script.exists():
        return GateResult("build", "FAIL", detail="build.sh missing")

    with BUILD_LOG.open("w") as logf:
        proc = subprocess.run(
            ["bash", str(build_script)],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=logf,
            text=True,
        )
    # Mirror stdout to the console so the operator sees real-time-ish
    # output of the build (we couldn't tee directly because subprocess
    # only lets us pipe one stream at a time without threading).
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.returncode == 0:
        return GateResult("build", "PASS")
    return GateResult(
        "build", "FAIL",
        detail=f"exit {proc.returncode}; see {BUILD_LOG.name}"
    )


def run_pytest(label: str, args: List[str]) -> GateResult:
    """Run pytest with ``args``; produce a ``GateResult`` row.

    ``args`` is the argv handed to pytest — typically a path + ``-v``
    plus optional ``-k <expr>``. The result row reports passed counts
    when the run succeeds; failed/errored counts go into ``detail`` on
    failure.
    """
    print(f"─── {label} ─────────────────────────────────────────────")
    cmdline = ["pytest", *args]
    print(f"  $ {' '.join(shlex.quote(a) for a in cmdline)}")

    proc = subprocess.run(
        cmdline,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    passed, failed, errored = _parse_pytest_counts(proc.stdout)

    if proc.returncode == 0:
        return GateResult(label, "PASS", count=passed)

    detail_parts = []
    if failed:
        detail_parts.append(f"{failed} failed")
    if errored:
        detail_parts.append(f"{errored} errored")
    if not detail_parts:
        detail_parts.append(f"exit {proc.returncode}")
    return GateResult(label, "FAIL", detail=", ".join(detail_parts))


def run_cutlass_parity() -> GateResult:
    """Optional gate — only runs if ``WITH_CUTLASS=1`` is set in env.

    When CUTLASS support is not requested, the gate is reported as
    ``N/A`` (not a failure). This matches how the rest of the build
    treats the CUTLASS path: opt-in.
    """
    if os.environ.get("WITH_CUTLASS") != "1":
        print("─── test_cutlass_parity (skipped — WITH_CUTLASS != 1) ───")
        return GateResult(
            "test_cutlass_parity", "N/A",
            detail="WITH_CUTLASS=1 not set"
        )
    return run_pytest(
        "test_cutlass_parity",
        ["tests/test_cutlass_parity.py", "-v"],
    )


# ── phase orchestration ─────────────────────────────────────────────


def gates_for_phase(phase: int, skip_build: bool) -> list:
    """Return the ordered list of gate-runner callables for ``phase``.

    Each callable returns a ``GateResult``. Phases stack — Phase 3 runs
    everything Phase 2 does, plus the model + cross-arch tests.
    """
    runners = []

    if not skip_build:
        runners.append(run_build)

    # Phase 2: core optimizer parity gate.
    runners.append(lambda: run_pytest(
        "test_supergrok2",
        ["tests/test_supergrok2.py", "-v"],
    ))

    if phase >= 3:
        runners.append(lambda: run_pytest(
            "test_models_sm_90",
            ["tests/test_models_sm_90.py", "-v"],
        ))
        runners.append(lambda: run_pytest(
            "test_cross_arch_agreement (sm_90)",
            ["tests/test_cross_arch_agreement.py", "-v", "-k", "sm_90"],
        ))

    runners.append(run_cutlass_parity)
    return runners


# ── reporting ───────────────────────────────────────────────────────


def print_summary(phase: int, results: List[GateResult]) -> None:
    """Pretty-print the final gate table."""
    label_width = max(len(r.name) for r in results)
    print()
    print("=" * 60)
    print(f"Verification gate (Phase {phase}):")
    for r in results:
        print(r.render(label_width))
    print("=" * 60)

    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_na = sum(1 for r in results if r.status == "N/A")
    print(f"  {n_pass} pass, {n_fail} fail, {n_na} N/A")
    print("=" * 60)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="verification_gate",
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--phase", type=int, default=3, choices=(2, 3),
        help="Which release phase's gate to run (default: 3 — full).",
    )
    p.add_argument(
        "--skip-build", action="store_true",
        help="Skip ``bash build.sh`` (use a pre-built wheel on PYTHONPATH).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    print("=" * 60)
    print(f"  Phase {args.phase} verification gate")
    print(f"  repo: {REPO_ROOT}")
    print("=" * 60)

    results: List[GateResult] = []
    for runner in gates_for_phase(args.phase, args.skip_build):
        result = runner()
        results.append(result)
        # Short-circuit on build failure: the rest of the gates need
        # the freshly-built wheel and will produce noise without it.
        if result.name == "build" and result.status == "FAIL":
            print("  build failed — skipping remaining gates")
            break

    print_summary(args.phase, results)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
