#!/usr/bin/env python3
"""AMD HIP tests for the all-specialized policy.

Supported AMD arches (per REFACTOR_PLAN.md §10):
  - gfx942 (CDNA3, MI300X / MI300A)
  - gfx950 (CDNA4, MI350X / MI355X)

Legacy AMD arches (gfx908 MI100, gfx90a MI200) are unsupported.

This test file replaces the previous tier-based test suite. The tests:

  1. platform.h is the only CUDA/HIP abstraction in csrc/kernels/. No
     raw cuda* / hip* APIs leak into the per-arch kernel files.
  2. dispatch.get_gpu_arch() returns 942/950 under FORCE_ARCH and raises
     UnsupportedArchError on legacy CDNA values.
  3. dispatch.get_arch_label() returns gfx942/gfx950 labels.
  4. csrc/kernels/hip/gfx942/ and csrc/kernels/hip/gfx950/ both have the
     17 per-optimizer baseline kernels.
"""

import os
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _set_force_arch(arch):
    if arch is None:
        os.environ.pop("FORCE_ARCH", None)
    else:
        os.environ["FORCE_ARCH"] = arch
    try:
        from grokking_optimizers import dispatch
        dispatch.get_gpu_arch.cache_clear()
        dispatch.get_gpu_vendor.cache_clear()
        dispatch.get_backend.cache_clear()
        dispatch.get_warp_size.cache_clear()
    except Exception:
        pass


class AMDDispatchTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed in this environment")

    def test_force_arch_942_returns_942(self):
        _set_force_arch("942")
        from grokking_optimizers.dispatch import get_gpu_arch
        self.assertEqual(get_gpu_arch(), 942)

    def test_force_arch_950_returns_950(self):
        _set_force_arch("950")
        from grokking_optimizers.dispatch import get_gpu_arch
        self.assertEqual(get_gpu_arch(), 950)

    def test_legacy_amd_arches_rejected(self):
        from grokking_optimizers.dispatch import UnsupportedArchError, get_gpu_arch
        # gfx908 (MI100), gfx90a (MI200), gfx1200 are not supported.
        for legacy in ("908", "90", "94", "1200"):
            _set_force_arch(legacy)
            with self.assertRaises(UnsupportedArchError, msg=f"FORCE_ARCH={legacy}"):
                get_gpu_arch()

    def test_arch_label_gfx942(self):
        _set_force_arch("942")
        from grokking_optimizers.dispatch import get_arch_label
        label = get_arch_label()
        self.assertIn("gfx942", label)

    def test_arch_label_gfx950(self):
        _set_force_arch("950")
        from grokking_optimizers.dispatch import get_arch_label
        label = get_arch_label()
        self.assertIn("gfx950", label)

    def test_supports_fp4_mfma_only_on_gfx950(self):
        from grokking_optimizers.dispatch import supports_fp4_mfma
        # FP4 MFMA is only available on gfx950, not gfx942.
        # This predicate also depends on get_gpu_vendor() which inspects
        # torch.version.hip; in a CPU-only test environment it returns
        # False under either FORCE_ARCH. The test skips if not on AMD.
        try:
            import torch
            if not (hasattr(torch.version, 'hip') and torch.version.hip):
                self.skipTest("not an AMD/HIP torch build")
        except ImportError:
            self.skipTest("torch missing")
        _set_force_arch("942")
        self.assertFalse(supports_fp4_mfma())
        _set_force_arch("950")
        self.assertTrue(supports_fp4_mfma())


class PlatformAbstractionTest(unittest.TestCase):
    """Per-arch kernel sources should not raw-call cuda*/hip* runtime APIs."""

    FORBIDDEN = (
        "cudaMemcpyAsync(",
        "cudaStreamSynchronize(",
        "hipMemcpyAsync(",
        "hipStreamSynchronize(",
    )

    def test_platform_h_used_in_gfx942_kernels(self):
        self._check_dir(REPO_ROOT / "csrc" / "kernels" / "hip" / "gfx942")

    def test_platform_h_used_in_gfx950_kernels(self):
        self._check_dir(REPO_ROOT / "csrc" / "kernels" / "hip" / "gfx950")

    def _check_dir(self, root):
        kernels = list(root.glob("*.hip.cpp"))
        kernels = [k for k in kernels if "_overlay" not in k.name]
        self.assertGreater(len(kernels), 0,
                           f"Expected at least one kernel under {root}")
        for path in kernels:
            text = path.read_text()
            for forbidden in self.FORBIDDEN:
                self.assertNotIn(
                    forbidden, text,
                    f"{path.name} uses {forbidden} directly; should go via "
                    f"platform.h abstraction.")


class AMDKernelLayoutTest(unittest.TestCase):
    """Every generic optimizer has baseline files in both supported AMD arches."""

    OPTIMIZERS = [
        "grokadamw", "grokfast", "lion", "looksam", "moe",
        "multi_tensor", "multi_tensor_prepare", "muon",
        "neuralgrok", "prodigy",
        "supergrok11", "supergrok15",
        "supergrok2_fwd", "supergrok2_bwd",
        "distributed_pipeline", "distributed_scan",
        "distributed_scan_pipeline",
    ]

    def test_all_gfx942_baselines_present(self):
        root = REPO_ROOT / "csrc" / "kernels" / "hip" / "gfx942"
        for opt in self.OPTIMIZERS:
            name = f"{opt}_gfx942.hip.cpp"
            self.assertTrue((root / name).exists(),
                            f"missing gfx942 baseline: {name}")

    def test_all_gfx950_baselines_present(self):
        root = REPO_ROOT / "csrc" / "kernels" / "hip" / "gfx950"
        for opt in self.OPTIMIZERS:
            name = f"{opt}_gfx950.hip.cpp"
            self.assertTrue((root / name).exists(),
                            f"missing gfx950 baseline: {name}")

    def test_gfx950_has_cdna4_overlay(self):
        """The recovered FP4/FP6/2:4 sparsity overlay should be in place."""
        overlay = (REPO_ROOT / "csrc" / "kernels" / "hip" / "gfx950" /
                   "cdna4_kernels_gfx950_overlay.hip.cpp")
        self.assertTrue(overlay.exists(),
                        "Expected the recovered CDNA4 FP4/FP6/2:4 overlay "
                        "at csrc/kernels/hip/gfx950/cdna4_kernels_gfx950_overlay.hip.cpp")

    def test_no_legacy_cdna_dirs(self):
        for legacy in ("cdna2", "cdna3", "cdna4"):
            self.assertFalse(
                (REPO_ROOT / "csrc" / "kernels" / "hip" / legacy).exists(),
                f"legacy {legacy} dir should not exist; the supported AMD "
                f"set is gfx942 + gfx950 only.")


if __name__ == "__main__":
    unittest.main()
