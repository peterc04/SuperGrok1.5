"""tests/test_compile_optimizations.py — §12 A1/A2 + C1 wiring tests.

Validates the auto-enabled candidates from docs/optimization_matrix.md:
- C1: NVCC version probe + ``--split-compile`` injection
- C2: ccache fallback alongside sccache in ``_sccache_env``
- A1: Hyperband pruner option in ``run_bayesian``
- A2: Transfer-learning seeding via sibling-optimizer trials
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grokking_optimizers import bayesian, search_space as ss  # noqa: E402
from grokking_optimizers.compile import (  # noqa: E402
    CompileCache, _collect_sibling_trials, _newer_compiler_flags,
    _probe_nvcc_version, _probe_hipcc_version, _sccache_env,
)


# ── C1 — toolchain probe ────────────────────────────────────────────────────

def test_nvcc_probe_returns_none_when_absent() -> None:
    # No nvcc on this CI host → probe returns None (does not crash).
    assert _probe_nvcc_version() is None


def test_hipcc_probe_returns_none_when_absent() -> None:
    assert _probe_hipcc_version() is None


def test_newer_compiler_flags_no_op_on_old_toolchain() -> None:
    # Without nvcc on PATH the call is a no-op.
    h, d = _newer_compiler_flags("sm_90", report=None)
    assert h == []
    assert d == []


def test_newer_compiler_flags_pallas_arch() -> None:
    h, d = _newer_compiler_flags("tpu_v5p", report=None)
    assert h == []
    assert d == []


# ── C2 / I1 — sccache & ccache wiring ──────────────────────────────────────

def test_sccache_env_safe_when_no_wrappers(monkeypatch) -> None:
    # No ccache/sccache binary should produce an empty (no-op) dict
    # except possibly Redis env propagation.
    monkeypatch.delenv("SCCACHE_REDIS_ENDPOINT", raising=False)
    monkeypatch.delenv("SCCACHE_REDIS",          raising=False)
    monkeypatch.delenv("SCCACHE_S3_BUCKET",      raising=False)
    out = _sccache_env()
    # On this CPU-only host neither is on PATH, so out is {} or only
    # contains a host wrapper if a stub somewhere caught.
    assert "SCCACHE_REDIS_ENDPOINT" not in out


def test_sccache_redis_env_propagates(monkeypatch) -> None:
    """If SCCACHE_REDIS_ENDPOINT is set and sccache is on PATH, we
    propagate it. If sccache is NOT on PATH, the env var still doesn't
    appear (we only propagate alongside the binary)."""
    monkeypatch.setenv("SCCACHE_REDIS_ENDPOINT", "redis://localhost:6379")
    out = _sccache_env()
    # On this host, sccache isn't installed, so the var won't be in out.
    # This is the correct conservative behaviour: we don't lie about
    # sccache being wired.
    import shutil
    if shutil.which("sccache") is None:
        assert "SCCACHE_REDIS_ENDPOINT" not in out
    else:
        assert out.get("SCCACHE_REDIS_ENDPOINT") == "redis://localhost:6379"


# ── A1 — Pruner factory ────────────────────────────────────────────────────

def test_pruner_factory() -> None:
    import optuna
    p = bayesian._make_pruner("none")
    assert isinstance(p, optuna.pruners.NopPruner)
    p = bayesian._make_pruner("median")
    assert isinstance(p, optuna.pruners.MedianPruner)
    p = bayesian._make_pruner("hyperband")
    assert isinstance(p, optuna.pruners.HyperbandPruner)


def test_run_bayesian_accepts_pruner(tmp_path: Path) -> None:
    """The pruner kwarg must thread through to the study without error."""
    space = {
        "sm_90": {
            "dims": [
                {"name": "block", "type": "int", "values": [128, 256],
                 "macro": "SG_TUNED_BLOCK_SIZE", "applies_to": ["host"]},
                {"name": "vec", "type": "int", "values": [1, 2, 4],
                 "macro": "SG_TUNED_VEC_WIDTH", "applies_to": ["host"]},
            ],
            "prefilter": {"rules": []},
        }
    }
    def t(cfg):
        return {"timing_ms": float(cfg["block"]) + cfg["vec"], "min_ms": 0,
                "max_ms": 0, "n": 21}
    trials = bayesian.run_bayesian(
        "sm_90", space, n_trials=5, seed=0,
        storage=tmp_path / "p.db",
        timer=t, pruner="hyperband",
    )
    assert len(trials) == 5


# ── A2 — Transfer-learning seeding ─────────────────────────────────────────

def test_collect_sibling_trials(tmp_path: Path) -> None:
    cache = CompileCache(tmp_path / "c.json")
    # Add two entries on the same (model, arch), different optimizers.
    cache.record_trial("adamw", "mamba", "sm_90", {
        "trial_num": 1, "stage": "tpe",
        "config": {"block": 256, "vec": 4, "unroll": 8},
        "config_key": "k1",
        "timing_ms": 0.42, "min_ms": 0.40, "max_ms": 0.45, "n": 21,
        "host": {}, "recorded_at": ""
    })
    cache.record_trial("lion", "mamba", "sm_90", {
        "trial_num": 1, "stage": "tpe",
        "config": {"block": 128, "vec": 2, "unroll": 4},
        "config_key": "k2",
        "timing_ms": 0.51, "min_ms": 0.50, "max_ms": 0.55, "n": 21,
        "host": {}, "recorded_at": ""
    })
    # Failed trial — must NOT be returned
    cache.record_trial("muon", "mamba", "sm_90", {
        "trial_num": 1, "stage": "tpe",
        "config": {"block": 64, "vec": 1, "unroll": 1},
        "config_key": "k3",
        "timing_ms": None, "min_ms": None, "max_ms": None, "n": None,
        "host": {}, "recorded_at": ""
    })

    siblings = _collect_sibling_trials(cache, "supergrok2", "mamba", "sm_90")
    assert len(siblings) == 2  # adamw + lion; muon skipped (failed)
    siblings_ck = sorted(t["config_key"] for t in siblings)
    assert siblings_ck == ["k1", "k2"]


def test_collect_sibling_trials_excludes_self(tmp_path: Path) -> None:
    cache = CompileCache(tmp_path / "c.json")
    cache.record_trial("lion", "mamba", "sm_90", {
        "trial_num": 1, "stage": "tpe",
        "config": {"block": 256, "vec": 4, "unroll": 8},
        "config_key": "k", "timing_ms": 0.4, "min_ms": 0, "max_ms": 0,
        "n": 21, "host": {}, "recorded_at": ""
    })
    # Querying for lion itself should return [] (no siblings).
    assert _collect_sibling_trials(cache, "lion", "mamba", "sm_90") == []


def test_collect_sibling_trials_isolates_arch(tmp_path: Path) -> None:
    cache = CompileCache(tmp_path / "c.json")
    cache.record_trial("adamw", "mamba", "gfx942", {
        "trial_num": 1, "stage": "tpe",
        "config": {"block": 256, "vec": 4, "unroll": 8},
        "config_key": "k", "timing_ms": 0.4, "min_ms": 0, "max_ms": 0,
        "n": 21, "host": {}, "recorded_at": ""
    })
    # Querying for a different arch should not return cross-arch trials.
    assert _collect_sibling_trials(cache, "lion", "mamba", "sm_90") == []
    # Same arch returns it.
    assert len(_collect_sibling_trials(cache, "lion", "mamba", "gfx942")) == 1


def test_run_bayesian_seed_trials_threaded(tmp_path: Path) -> None:
    space = {
        "sm_90": {
            "dims": [
                {"name": "block", "type": "int", "values": [128, 256, 512],
                 "macro": "SG_TUNED_BLOCK_SIZE", "applies_to": ["host"]},
                {"name": "vec", "type": "int", "values": [1, 2, 4],
                 "macro": "SG_TUNED_VEC_WIDTH", "applies_to": ["host"]},
            ],
            "prefilter": {"rules": []},
        }
    }
    seeds = [
        {"config": {"block": 256, "vec": 4}, "timing_ms": 0.1,
         "min_ms": 0.09, "max_ms": 0.11, "n": 21, "stage": "tpe",
         "config_key": "block=256_vec=4", "host": {}, "recorded_at": ""},
    ]
    def t(cfg):
        # Synthetic: prefer 256/4 (matches the seed)
        return {"timing_ms": abs(cfg["block"] - 256) + abs(cfg["vec"] - 4),
                "min_ms": 0, "max_ms": 0, "n": 21}
    trials = bayesian.run_bayesian(
        "sm_90", space, n_trials=8, seed=0,
        storage=tmp_path / "p.db",
        timer=t, seed_trials=seeds,
    )
    # The study should have at least one trial; seed_trials add to the
    # study but n_trials is the new-trial budget.
    assert len(trials) == 8
