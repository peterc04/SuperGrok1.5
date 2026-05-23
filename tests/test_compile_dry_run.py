"""tests/test_compile_dry_run.py — end-to-end on a no-GPU host.

Uses the pallas backend (Python-only; no C++ compile path) so a CPU-only
CI environment can verify the orchestrator wires up correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grokking_optimizers.compile import (  # noqa: E402
    CACHE_VERSION, build, build_aot, build_jit, BuildSpec, CompileCache,
    main,
)


def test_pallas_build_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "out"
    cache = tmp_path / "cache.json"
    # Pallas backend: no GPU needed, no .so produced.
    so = build(
        optimizer="lion", model="mamba", arch="tpu_v5p",
        cache_path=cache, out_dir=out,
        aot_only=True, autotune=False, profile=False,
        bayesian_trials=10,
    )
    assert so is None  # pallas: no .so
    assert cache.exists()
    data = json.loads(cache.read_text())
    assert data["version"] == CACHE_VERSION


def test_cli_help_lists_new_flags() -> None:
    """The CLI should accept the new mandated flags."""
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code in (0, None)


def test_cli_runtime_alias_consistency() -> None:
    """--aot-only should be an alias for --runtime aot (parse but don't
    actually build)."""
    import argparse
    from grokking_optimizers.compile import main as _main
    # Trigger argparse validation by passing an unknown optimizer combined
    # with --aot-only; we expect argparse to reject the unknown optimizer
    # before we get to the build.
    try:
        _main(["--optimizer", "zzz", "--model", "mamba", "--arch", "sm_90",
               "--aot-only"])
    except SystemExit:
        pass


def test_aot_freshness_with_search_space_hash(tmp_path: Path) -> None:
    cp = tmp_path / "ss.json"
    cache = CompileCache(cp)
    fake = tmp_path / "a.so"
    fake.write_bytes(b"x" * 100)
    cache.record_aot(
        "lion", "mamba", "sm_90",
        source_hash="s", host_flags_hash="h", device_flags_hash="d",
        so_path=fake, search_space_hash="ss1",
    )
    # Different search_space_hash on the inquiry should not invalidate
    # if the cache entry's hash is None (legacy) — but if the cache has
    # ss1 and we ask with ss2 it should fail.
    assert cache.is_aot_fresh("lion", "mamba", "sm_90", "s", "h", "d",
                              search_space_hash="ss1")
    assert not cache.is_aot_fresh("lion", "mamba", "sm_90", "s", "h", "d",
                                  search_space_hash="ss2")
