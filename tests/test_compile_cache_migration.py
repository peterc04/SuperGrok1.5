"""tests/test_compile_cache_migration.py — v2 → v3 forward migration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grokking_optimizers.compile import (  # noqa: E402
    CACHE_VERSION, CompileCache, _V3_DEFAULTS,
)


def _v2_payload() -> dict:
    """Hand-crafted v2 cache shape."""
    return {
        "version": 2,
        "created_at": "2026-04-10T12:00:00",
        "host_history": [{"platform": "old"}],
        "entries": {
            "lion/mamba/sm_90": {
                "source_hash": "abc",
                "host_cflags_hash": "def",
                "device_cflags_hash": "ghi",
                "primary_artifact": None,
                "variant_artifacts": {"b256_v4_u8": {"path": "/a", "size": 1, "mtime": 0}},
                "sweep_history": [{"config": {"block": 256, "vec": 4, "unroll": 8},
                                   "timing_ms": 1.23}],
                "tuned_config": {"block": 256, "vec": 4, "unroll": 8, "timing_ms": 1.23},
                "aot_completed_at": "2026-04-10T12:00:00",
                "jit_completed_at": "2026-04-10T12:01:00",
                "aot_host": {"platform": "old"},
                "jit_host": {"platform": "old"},
            },
            "adamw/decoder/gfx942": {
                "source_hash": "xxx",
                "host_cflags_hash": "yyy",
                "device_cflags_hash": "zzz",
                "primary_artifact": None,
                "variant_artifacts": {},
                "sweep_history": [],
                "tuned_config": None,
                "aot_completed_at": "2026-04-11T08:00:00",
                "jit_completed_at": None,
                "aot_host": {"platform": "old"},
                "jit_host": None,
            },
        },
    }


def test_v2_to_v3_migration(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(_v2_payload(), indent=2))

    cache = CompileCache(cache_path)
    assert cache._data["version"] == CACHE_VERSION
    # Migration backup exists
    assert (cache_path.with_suffix(cache_path.suffix + ".v2.bak")).exists()
    # Original entries are preserved
    entries = cache._data["entries"]
    assert set(entries.keys()) == {"lion/mamba/sm_90", "adamw/decoder/gfx942"}
    e = entries["lion/mamba/sm_90"]
    assert e["source_hash"] == "abc"
    assert e["tuned_config"] == {"block": 256, "vec": 4, "unroll": 8,
                                 "timing_ms": 1.23}
    assert e["sweep_history"][0]["timing_ms"] == 1.23
    # v3 defaults are filled in
    for k in _V3_DEFAULTS:
        assert k in e
    assert e["pgo_enabled"] is False
    assert e["bayesian_trials"] == []
    assert e["mode"] is None

    # Save and reload — must come back as v3.
    cache.save()
    reloaded = json.loads(cache_path.read_text())
    assert reloaded["version"] == CACHE_VERSION
    assert reloaded["entries"]["lion/mamba/sm_90"]["pgo_enabled"] is False
    assert reloaded["entries"]["lion/mamba/sm_90"]["bayesian_trials"] == []


def test_v3_cache_round_trips(tmp_path: Path) -> None:
    cp = tmp_path / "fresh.json"
    cache = CompileCache(cp)
    cache.record_aot(
        "lion", "mamba", "sm_90",
        source_hash="s", host_flags_hash="h", device_flags_hash="d",
        so_path=None, pgo_enabled=True,
        pgo_workload_hash="w", search_space_hash="ss",
    )
    cache.record_trial("lion", "mamba", "sm_90", {
        "trial_num": 1, "stage": "tpe",
        "config": {"block": 256, "vec": 4, "unroll": 8},
        "config_key": "block=256_unroll=8_vec=4",
        "timing_ms": 0.42, "min_ms": 0.40, "max_ms": 0.45, "n": 21,
        "status": "ok", "host": {"platform": "test"},
        "recorded_at": "2026-05-23T12:00:00",
    })
    cache.set_tuned("lion", "mamba", "sm_90",
                    {"block": 256, "vec": 4, "unroll": 8, "timing_ms": 0.42},
                    mode="bayesian", search_space_hash="ss")
    cache.save()

    reloaded = CompileCache(cp)
    e = reloaded._data["entries"]["lion/mamba/sm_90"]
    assert e["pgo_enabled"] is True
    assert e["pgo_workload_hash"] == "w"
    assert e["search_space_hash"] == "ss"
    assert e["mode"] == "bayesian"
    assert len(e["bayesian_trials"]) == 1
    assert e["bayesian_trials"][0]["timing_ms"] == 0.42
    assert reloaded.is_jit_fresh("lion", "mamba", "sm_90",
                                 search_space_hash="ss")


def test_aot_freshness_with_pgo_toggle(tmp_path: Path) -> None:
    cp = tmp_path / "pgo.json"
    cache = CompileCache(cp)
    # Record an AOT entry with a fake artifact in tmp_path.
    fake = tmp_path / "fake.so"
    fake.write_bytes(b"\0" * 16)
    cache.record_aot(
        "lion", "mamba", "sm_90",
        source_hash="s", host_flags_hash="h", device_flags_hash="d",
        so_path=fake, pgo_enabled=True,
        pgo_workload_hash="w1", search_space_hash="ss",
    )
    # Same params → fresh.
    assert cache.is_aot_fresh("lion", "mamba", "sm_90", "s", "h", "d",
                              pgo_enabled=True, pgo_workload_hash="w1",
                              search_space_hash="ss")
    # Toggling PGO off → not fresh.
    assert not cache.is_aot_fresh("lion", "mamba", "sm_90", "s", "h", "d",
                                  pgo_enabled=False,
                                  search_space_hash="ss")
    # Different workload hash → not fresh.
    assert not cache.is_aot_fresh("lion", "mamba", "sm_90", "s", "h", "d",
                                  pgo_enabled=True, pgo_workload_hash="w2",
                                  search_space_hash="ss")


def test_corrupt_cache_archives(tmp_path: Path) -> None:
    cp = tmp_path / "corrupt.json"
    cp.write_text("{ not valid json ")
    cache = CompileCache(cp)
    assert (cp.with_suffix(cp.suffix + ".corrupt.bak")).exists()
    assert cache._data["version"] == CACHE_VERSION
    assert cache._data["entries"] == {}
