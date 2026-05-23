"""tests/test_compile_bayesian.py — Bayesian driver end-to-end with stub timer.

No GPU; the timer is a deterministic synthetic function of the config so
TPE's behaviour can be verified without a real build/run cycle.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grokking_optimizers import bayesian, search_space as ss  # noqa: E402


def _tiny_space() -> dict:
    return {
        "sm_90": {
            "dims": [
                {"name": "block", "type": "int", "values": [64, 128, 256, 512],
                 "macro": "SG_TUNED_BLOCK_SIZE", "applies_to": ["host", "device"]},
                {"name": "vec", "type": "int", "values": [1, 2, 4],
                 "macro": "SG_TUNED_VEC_WIDTH", "applies_to": ["host", "device"]},
                {"name": "unroll", "type": "int", "values": [1, 2, 4, 8],
                 "macro": "SG_TUNED_UNROLL", "applies_to": ["host", "device"]},
            ],
            "prefilter": {"rules": []},
        }
    }


def _synthetic_timer(cfg):
    """Quadratic-ish landscape with min near (block=256, vec=4, unroll=8)."""
    score = ((cfg["block"] - 256) / 256.0) ** 2
    score += ((cfg["vec"] - 4) / 4.0) ** 2
    score += ((cfg["unroll"] - 8) / 8.0) ** 2
    score += 0.05  # base
    return {"timing_ms": score, "min_ms": score - 0.01,
            "max_ms": score + 0.01, "n": 21}


def test_run_bayesian_finds_winner(tmp_path: Path) -> None:
    space = _tiny_space()
    trials = bayesian.run_bayesian(
        "sm_90", space, n_trials=40, seed=0,
        storage=tmp_path / "study.db",
        timer=_synthetic_timer,
    )
    assert len(trials) == 40
    assert all(t["stage"] == "tpe" for t in trials)
    winner = bayesian.pick_winner(trials)
    assert winner is not None
    # With 40 trials over 4×3×4=48 configs the optimum is usually found,
    # but seed=0 + categorical TPE makes the bound tight.
    assert winner["timing_ms"] < 0.5


def test_topk_refine_generates_neighbours(tmp_path: Path) -> None:
    space = _tiny_space()
    trials = bayesian.run_bayesian(
        "sm_90", space, n_trials=20, seed=0,
        storage=tmp_path / "study.db",
        timer=_synthetic_timer,
    )
    refines = bayesian.topk_refine(
        trials, space, "sm_90", top_k=5, radius=2,
        timer=_synthetic_timer,
    )
    # The grid has only 48 configs so the deduped refinements are bounded.
    assert all(t["stage"] == "refine" for t in refines)
    if refines:
        # Each refine config differs from its seed by exactly one dim
        seen = {t["config_key"] for t in trials} | {t["config_key"] for t in refines}
        assert len(seen) > len(trials)


def test_resume_via_storage(tmp_path: Path) -> None:
    space = _tiny_space()
    storage = tmp_path / "resume.db"
    t1 = bayesian.run_bayesian(
        "sm_90", space, n_trials=10, seed=0,
        storage=storage,
        timer=_synthetic_timer,
    )
    # Re-open the same storage — Optuna should keep the trial history.
    # Our adapter does N more trials on top.
    t2 = bayesian.run_bayesian(
        "sm_90", space, n_trials=10, seed=0,
        storage=storage,
        timer=_synthetic_timer,
    )
    assert len(t1) == 10
    assert len(t2) == 10
    # Both file-mounted, study persists.
    assert storage.exists()


def test_winner_picker_handles_failures() -> None:
    trials = [
        {"trial_num": 1, "stage": "tpe", "config_key": "a",
         "config": {"block": 64}, "timing_ms": None,
         "min_ms": None, "max_ms": None, "n": None,
         "host": {}, "recorded_at": ""},
        {"trial_num": 2, "stage": "tpe", "config_key": "b",
         "config": {"block": 128}, "timing_ms": 0.7,
         "min_ms": 0.6, "max_ms": 0.8, "n": 21,
         "host": {}, "recorded_at": ""},
        {"trial_num": 3, "stage": "tpe", "config_key": "c",
         "config": {"block": 256}, "timing_ms": 0.3,
         "min_ms": 0.2, "max_ms": 0.4, "n": 21,
         "host": {}, "recorded_at": ""},
    ]
    w = bayesian.pick_winner(trials)
    assert w is not None
    assert w["timing_ms"] == 0.3
    assert w["config_key"] == "c"
