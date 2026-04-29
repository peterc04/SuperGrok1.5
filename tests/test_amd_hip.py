#!/usr/bin/env python3
"""AMD HIP / gfx942 tests for the all-specialized policy.

Under the all-specialized refactor (REFACTOR_PLAN.md), the only supported
AMD arch is gfx942 (MI300X / CDNA3). gfx908, gfx90a, and gfx950 are no
longer supported and the build drops them from --offload-arch.

This test file replaces the previous tier-based test suite. The tests:

  1. platform.h is the only CUDA/HIP abstraction in csrc/kernels/. No
     raw cuda* / hip* APIs leak into the per-arch kernel files.
  2. dispatch.get_gpu_arch() returns 942 under FORCE_ARCH=942 and raises
     UnsupportedArchError on legacy CDNA values.
  3. dispatch.get_arch_label() returns the gfx942 label.
  4. The csrc/kernels/hip/gfx942/ directory has all the per-optimizer
     baseline kernels.
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


class GFX942DispatchTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed in this environment")

    def test_force_arch_942_returns_942(self):
        _set_force_arch("942")
        from grokking_optimizers.dispatch import get_gpu_arch
        self.assertEqual(get_gpu_arch(), 942)

    def test_legacy_amd_arches_rejected(self):
        from grokking_optimizers.dispatch import UnsupportedArchError, get_gpu_arch
        for legacy in ("908", "90", "94", "950", "1200"):
            _set_force_arch(legacy)
            with self.assertRaises(UnsupportedArchError, msg=f"FORCE_ARCH={legacy}"):
                get_gpu_arch()

    def test_arch_label_gfx942(self):
        _set_force_arch("942")
        from grokking_optimizers.dispatch import get_arch_label
        label = get_arch_label()
        self.assertIn("gfx942", label)


class PlatformAbstractionTest(unittest.TestCase):
    """Per-arch kernel sources should not raw-call cuda*/hip* runtime APIs."""

    FORBIDDEN = (
        "cudaMemcpyAsync(",
        "cudaStreamSynchronize(",
        "hipMemcpyAsync(",
        "hipStreamSynchronize(",
    )

    def test_platform_h_used_in_kernels(self):
        kernels = list((REPO_ROOT / "csrc" / "kernels" / "hip" / "gfx942").glob("*.hip.cpp"))
        kernels = [k for k in kernels if "_overlay" not in k.name]
        self.assertGreater(len(kernels), 0,
                           "Expected at least one gfx942 kernel under "
                           "csrc/kernels/hip/gfx942/")
        for path in kernels:
            text = path.read_text()
            for forbidden in self.FORBIDDEN:
                self.assertNotIn(
                    forbidden, text,
                    f"{path.name} uses {forbidden} directly; should go via "
                    f"platform.h abstraction.")


class GFX942KernelLayoutTest(unittest.TestCase):
    """Every generic optimizer should have a gfx942 baseline file."""

    EXPECTED_BASELINES = [
        "grokadamw_gfx942.hip.cpp",
        "grokfast_gfx942.hip.cpp",
        "lion_gfx942.hip.cpp",
        "looksam_gfx942.hip.cpp",
        "moe_gfx942.hip.cpp",
        "multi_tensor_gfx942.hip.cpp",
        "multi_tensor_prepare_gfx942.hip.cpp",
        "muon_gfx942.hip.cpp",
        "neuralgrok_gfx942.hip.cpp",
        "prodigy_gfx942.hip.cpp",
        "supergrok11_gfx942.hip.cpp",
        "supergrok15_gfx942.hip.cpp",
        "supergrok2_fwd_gfx942.hip.cpp",
        "supergrok2_bwd_gfx942.hip.cpp",
        "distributed_pipeline_gfx942.hip.cpp",
        "distributed_scan_gfx942.hip.cpp",
        "distributed_scan_pipeline_gfx942.hip.cpp",
    ]

    def test_all_baselines_present(self):
        gfx942 = REPO_ROOT / "csrc" / "kernels" / "hip" / "gfx942"
        for name in self.EXPECTED_BASELINES:
            self.assertTrue(
                (gfx942 / name).exists(),
                f"missing baseline: {name}")

    def test_no_legacy_cdna_dirs(self):
        for legacy in ("cdna2", "cdna4"):
            self.assertFalse(
                (REPO_ROOT / "csrc" / "kernels" / "hip" / legacy).exists(),
                f"legacy {legacy} dir should not exist; gfx908/gfx90a/gfx950 "
                f"are no longer supported.")


if __name__ == "__main__":
    unittest.main()
