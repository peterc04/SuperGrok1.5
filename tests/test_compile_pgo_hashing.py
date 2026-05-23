"""tests/test_compile_pgo_hashing.py — workload hash determinism + flag composition."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grokking_optimizers import pgo  # noqa: E402


def test_hash_workload_deterministic(tmp_path: Path) -> None:
    script = tmp_path / "w.py"
    script.write_text("print('hi')")
    h1 = pgo.hash_workload(script, 1000)
    h2 = pgo.hash_workload(script, 1000)
    assert h1 == h2


def test_hash_workload_changes_on_content(tmp_path: Path) -> None:
    script = tmp_path / "w.py"
    script.write_text("a = 1")
    h1 = pgo.hash_workload(script, 1000)
    script.write_text("a = 2")
    h2 = pgo.hash_workload(script, 1000)
    assert h1 != h2


def test_hash_workload_changes_on_steps(tmp_path: Path) -> None:
    script = tmp_path / "w.py"
    script.write_text("a = 1")
    h1 = pgo.hash_workload(script, 1000)
    h2 = pgo.hash_workload(script, 2000)
    assert h1 != h2


def test_hash_workload_missing_file(tmp_path: Path) -> None:
    # Missing file still produces a stable hash (so the cache doesn't crash).
    missing = tmp_path / "nope.py"
    assert pgo.hash_workload(missing, 100) == pgo.hash_workload(missing, 100)


def test_instrument_flags_cuda(tmp_path: Path) -> None:
    h, d, l = pgo.instrument_flags(
        "sm_90", tmp_path,
        ["-O3"], ["-O3", "-Xcompiler", "-fPIC"], ["-flto"],
    )
    assert any("-fprofile-generate" in f for f in h)
    assert any("-fprofile-generate" in f for f in d)
    assert any("-fprofile-generate" in f for f in l)


def test_instrument_flags_hip(tmp_path: Path) -> None:
    h, d, l = pgo.instrument_flags(
        "gfx942", tmp_path,
        ["-O3"], ["-O3"], ["-flto"],
    )
    # HIP path: device cflags get the bare -fprofile-generate (no -Xcompiler)
    assert any("-fprofile-generate" in f for f in d)
    assert not any(f == "-Xcompiler" for f in d)


def test_use_flags_round_trip(tmp_path: Path) -> None:
    h, d, l = pgo.use_flags(
        "sm_90", tmp_path,
        ["-O3"], ["-O3", "-Xcompiler", "-fPIC"], ["-flto"],
    )
    assert any("-fprofile-use" in f for f in h)
    assert any("-fprofile-use" in f for f in d)
    assert any("-fprofile-use" in f for f in l)
    assert any("-fprofile-correction" in f for f in h)


def test_collect_workload_validates_no_output(tmp_path: Path) -> None:
    # Script exits cleanly but produces no profile data → should return False.
    script = tmp_path / "noop.py"
    script.write_text(
        "import argparse, sys\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--so'); p.add_argument('--opt')\n"
        "p.add_argument('--model'); p.add_argument('--arch')\n"
        "p.add_argument('--steps', type=int)\n"
        "p.parse_args()\n"
        "print('noop'); sys.exit(0)\n"
    )
    profile_dir = tmp_path / "pdata"
    profile_dir.mkdir()
    ok = pgo.collect_workload(
        script, so_path=tmp_path / "fake.so",
        opt_class="Lion", model="mamba", arch="sm_90",
        profile_dir=profile_dir, steps=10,
    )
    assert ok is False  # no profile data written


def test_collect_workload_validates_with_output(tmp_path: Path) -> None:
    # Script writes a fake .gcda → should return True.
    script = tmp_path / "writer.py"
    profile_dir = tmp_path / "pdata"
    profile_dir.mkdir()
    script.write_text(
        f"import argparse, sys\n"
        f"p = argparse.ArgumentParser()\n"
        f"p.add_argument('--so'); p.add_argument('--opt')\n"
        f"p.add_argument('--model'); p.add_argument('--arch')\n"
        f"p.add_argument('--steps', type=int)\n"
        f"p.parse_args()\n"
        f"from pathlib import Path\n"
        f"Path({str(profile_dir)!r}).joinpath('default.profraw').write_bytes(b'\\x00' * 32)\n"
        f"sys.exit(0)\n"
    )
    ok = pgo.collect_workload(
        script, so_path=tmp_path / "fake.so",
        opt_class="Lion", model="mamba", arch="sm_90",
        profile_dir=profile_dir, steps=10,
    )
    assert ok is True
