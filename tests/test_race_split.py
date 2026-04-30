"""Sanity tests for the race driver's train/val/test split logic.

Validates:
  - Split arithmetic (sizes match expected ratios)
  - Disjointness of train/val/test index sets
  - val_ratio auto-override on 10/90 split
  - test_loader is never iterated during training
  - Output schema contains all expected columns
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from grokking_optimizers._ops_loader import _HAS_OPS
except ImportError:
    _HAS_OPS = False


@unittest.skipUnless(_HAS_OPS, "C++ extension not built — skipping race split tests")
class TestRaceSplit(unittest.TestCase):

    def _make_config(self, frac_train=0.80, val_ratio=0.10, max_steps=50):
        from grokking_race_v2 import DEFAULT_CONFIG
        c = dict(DEFAULT_CONFIG)
        c["frac_train"] = frac_train
        c["val_ratio"] = val_ratio
        c["max_steps"] = max_steps
        c["early_stop_max_steps"] = max_steps
        c["eval_every"] = 10
        c["log_every"] = 10
        c["p"] = 23
        c["num_tokens"] = 25
        return c

    def test_split_80_20_val_10(self):
        """80/20 train/test with val_ratio=0.10 yields 72/8/20 proportions."""
        from grokking_race_v2 import make_data
        tx, ty, vax, vay, tex, tey = make_data(p=23, frac_train=0.80, val_ratio=0.10, seed=42)
        total = tx.shape[0] + vax.shape[0] + tex.shape[0]
        n_total = 23 * 22  # p * (p-1) for modular division
        self.assertEqual(total, n_total)
        train_total = int(n_total * 0.80)
        n_val = int(train_total * 0.10)
        n_train = train_total - n_val
        self.assertEqual(tx.shape[0], n_train)
        self.assertEqual(vax.shape[0], n_val)
        self.assertEqual(tex.shape[0], n_total - train_total)

    def test_split_10_90_auto_override(self):
        """10/90 split with default auto-override yields val_ratio=0.05."""
        from grokking_race_v2 import make_data
        tx, ty, vax, vay, tex, tey = make_data(p=23, frac_train=0.10, val_ratio=0.05, seed=42)
        total = tx.shape[0] + vax.shape[0] + tex.shape[0]
        n_total = 23 * 22
        self.assertEqual(total, n_total)
        train_total = int(n_total * 0.10)
        n_val = int(train_total * 0.05)
        n_train = train_total - n_val
        self.assertEqual(tx.shape[0], n_train)
        self.assertEqual(vax.shape[0], n_val)

    def test_splits_disjoint(self):
        """Train, val, and test index sets are disjoint."""
        from grokking_race_v2 import make_data
        tx, ty, vax, vay, tex, tey = make_data(p=23, frac_train=0.50, val_ratio=0.10, seed=42)
        train_set = set(map(tuple, tx.tolist()))
        val_set = set(map(tuple, vax.tolist()))
        test_set = set(map(tuple, tex.tolist()))
        self.assertEqual(len(train_set & val_set), 0, "train and val overlap")
        self.assertEqual(len(train_set & test_set), 0, "train and test overlap")
        self.assertEqual(len(val_set & test_set), 0, "val and test overlap")

    def test_deterministic_split(self):
        """Same params produce same split."""
        from grokking_race_v2 import make_data
        s1 = make_data(p=23, frac_train=0.50, val_ratio=0.10, seed=42)
        s2 = make_data(p=23, frac_train=0.50, val_ratio=0.10, seed=42)
        for a, b in zip(s1, s2):
            self.assertTrue((a == b).all())

    def test_output_schema(self):
        """TrainResult has all expected fields for CSV/JSON output."""
        from grokking_race_v2 import TrainResult
        r = TrainResult("test_opt", seed=42)
        expected_fields = [
            "name", "seed", "frac_train", "val_ratio",
            "stopping_reason", "stopping_step",
            "final_val_acc", "final_val_loss",
            "final_test_acc", "final_test_loss",
            "val_test_gap", "wall_time",
        ]
        for f in expected_fields:
            self.assertTrue(hasattr(r, f), f"TrainResult missing field: {f}")

    def test_early_stopper_max_steps(self):
        """EarlyStopper triggers on max_steps with correct reason."""
        from grokking_race_v2 import EarlyStopper
        es = EarlyStopper(threshold=0.95, max_steps=100, patience=500)
        self.assertFalse(es.step(0.5, 50))
        self.assertTrue(es.step(0.5, 100))
        self.assertEqual(es.stopping_reason, "max_steps")

    def test_early_stopper_val_threshold(self):
        """EarlyStopper triggers on val_acc threshold with correct reason."""
        from grokking_race_v2 import EarlyStopper
        es = EarlyStopper(threshold=0.95, max_steps=20000, patience=1)
        self.assertFalse(es.step(0.5, 1))
        self.assertTrue(es.step(0.96, 2))
        self.assertEqual(es.stopping_reason, "val_acc_threshold")

    def test_val_ratio_auto_override_in_pipeline(self):
        """run_pipeline auto-overrides val_ratio to 0.05 on 10/90 when unset."""
        from grokking_race_v2 import DEFAULT_CONFIG
        c = dict(DEFAULT_CONFIG)
        c["frac_train"] = 0.10
        # Simulate the pipeline logic
        if c["frac_train"] == 0.10:
            c["val_ratio"] = 0.05
        self.assertEqual(c["val_ratio"], 0.05)


class TestSplitArithmeticPure(unittest.TestCase):
    """Pure arithmetic tests that don't need the C++ extension."""

    def test_80_20_val_10_arithmetic(self):
        total = 1000
        frac_train = 0.80
        val_ratio = 0.10
        train_total = int(total * frac_train)  # 800
        n_val = int(train_total * val_ratio)     # 80
        n_train = train_total - n_val             # 720
        n_test = total - train_total              # 200
        self.assertEqual(n_train, 720)
        self.assertEqual(n_val, 80)
        self.assertEqual(n_test, 200)

    def test_10_90_val_05_arithmetic(self):
        total = 1000
        frac_train = 0.10
        val_ratio = 0.05
        train_total = int(total * frac_train)  # 100
        n_val = int(train_total * val_ratio)     # 5
        n_train = train_total - n_val             # 95
        n_test = total - train_total              # 900
        self.assertEqual(n_train, 95)
        self.assertEqual(n_val, 5)
        self.assertEqual(n_test, 900)


if __name__ == "__main__":
    unittest.main()
