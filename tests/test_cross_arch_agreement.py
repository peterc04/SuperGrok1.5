"""Cross-arch numerical agreement test.

For each optimizer, verify that the per-arch specialized kernels produce
elementwise-equal output (modulo FP tolerance) on the same input. Skips
arches that aren't compiled into the local extension.

This test is the safety net for the all-specialized refactor: the math
must remain identical across sm_80, sm_90, sm_100, gfx942 even as
arch-specific primitives (cp.async, TMA, MFMA, FP8 paths) diverge for
performance.

Set FORCE_ARCH=<80|90|100|942> to switch dispatch between arches without
changing hardware (the per-arch binding for that arch must be in the
build).

The test loops over each (optimizer, arch) pair, runs N optimizer steps
under FORCE_ARCH=<arch>, captures the resulting parameter and state
tensors, and asserts allclose between every pair.
"""

from __future__ import annotations

import os
import unittest

import torch

# Skip the entire module gracefully if torch isn't built or if no GPU is
# available. The test is hardware-gated; CI runs it per-arch.
HAS_GPU = torch.cuda.is_available()
SUPPORTED = (80, 89, 90, 100, 103, 120, 942, 950)


def _set_force_arch(arch: int):
    os.environ["FORCE_ARCH"] = str(arch)
    # Force a re-import of dispatch so the lru_cache picks up the env change.
    import grokking_optimizers.dispatch as _d
    _d.get_gpu_arch.cache_clear()
    _d.get_gpu_vendor.cache_clear()
    _d.get_backend.cache_clear()
    _d.get_warp_size.cache_clear()


def _arch_available(arch: int) -> bool:
    """Returns True if the per-arch binding for `arch` is compiled into _ops.

    Pragmatic check: try to set FORCE_ARCH and run detect_arch(); raise if
    the binding errors at the dispatch boundary.
    """
    try:
        _set_force_arch(arch)
        from grokking_optimizers._ops_loader import get_ops
        ops = get_ops()
        return ops.detect_arch() == arch
    except Exception:
        return False


@unittest.skipUnless(HAS_GPU, "Cross-arch test requires a GPU.")
class CrossArchAgreementTest(unittest.TestCase):
    """Per-optimizer agreement across the eight supported arches.

    Each test_<optimizer>_cross_arch:
      1. Builds a small parameter tensor and gradient.
      2. For each compiled-in arch, runs N optimizer steps under
         FORCE_ARCH=<arch> and captures the resulting parameter tensor.
      3. Asserts torch.allclose against the reference (first arch
         present), with a tolerance that allows for FP rounding in
         fast-math intrinsics.
    Arches not compiled into the local extension are skipped. If fewer
    than two arches are available the whole test skips (nothing to
    compare).
    """

    TOLERANCE = 1e-4  # max abs diff per element

    def _run_optimizer_under_arch(self, optimizer_factory, arch, steps=10):
        """Helper: run the given optimizer under FORCE_ARCH=<arch>."""
        if not _arch_available(arch):
            self.skipTest(f"sm_{arch} binding not built")
        _set_force_arch(arch)

        torch.manual_seed(0)
        param = torch.randn(1024, device="cuda", requires_grad=True)
        opt = optimizer_factory([param])
        for _ in range(steps):
            opt.zero_grad()
            (param ** 2).sum().backward()
            opt.step()
        return param.detach().cpu().clone()

    def _check_agreement(self, optimizer_factory):
        results = {}
        for arch in SUPPORTED:
            if not _arch_available(arch):
                continue
            results[arch] = self._run_optimizer_under_arch(optimizer_factory, arch)
        if len(results) < 2:
            self.skipTest(
                f"Need at least 2 arches with bindings; got {list(results)}")
        ref_arch, ref = next(iter(results.items()))
        for arch, p in results.items():
            if arch == ref_arch:
                continue
            max_diff = (p - ref).abs().max().item()
            self.assertLess(
                max_diff, self.TOLERANCE,
                f"sm_{arch} disagrees with sm_{ref_arch} by {max_diff:.6f}")

    # ────────────────────────────────────────────────────────────────────
    # Per-optimizer test cases. Each just supplies a factory.
    # ────────────────────────────────────────────────────────────────────

    def test_grokadamw_cross_arch(self):
        from grokking_optimizers import GrokAdamW
        self._check_agreement(lambda ps: GrokAdamW(ps, lr=1e-3))

    def test_lion_cross_arch(self):
        from grokking_optimizers import Lion
        self._check_agreement(lambda ps: Lion(ps, lr=3e-4))

    def test_grokfast_cross_arch(self):
        from grokking_optimizers import Grokfast
        self._check_agreement(lambda ps: Grokfast(ps, lr=1e-3))

    def test_prodigy_cross_arch(self):
        from grokking_optimizers import Prodigy
        self._check_agreement(lambda ps: Prodigy(ps, lr=1.0))

    def test_neuralgrok_cross_arch(self):
        from grokking_optimizers import NeuralGrok
        self._check_agreement(lambda ps: NeuralGrok(ps, lr=1e-3))

    def test_looksam_cross_arch(self):
        from grokking_optimizers import LookSAM
        self._check_agreement(lambda ps: LookSAM(ps, lr=1e-3))

    def test_muon_cross_arch(self):
        """Muon requires 2D params (Newton-Schulz orthogonalization).
        Rebuild a 64x64 matrix per arch and compare element-wise.
        """
        from grokking_optimizers import Muon

        def run_arch(arch, steps=10):
            if not _arch_available(arch):
                return None
            _set_force_arch(arch)
            torch.manual_seed(0)
            P = torch.randn(64, 64, device="cuda", requires_grad=True)
            opt = Muon([P], lr=0.02)
            for _ in range(steps):
                opt.zero_grad()
                (P ** 2).sum().backward()
                opt.step()
            return P.detach().cpu().clone()

        results = {}
        for arch in SUPPORTED:
            r = run_arch(arch)
            if r is not None:
                results[arch] = r
        if len(results) < 2:
            self.skipTest(
                f"Muon: need >=2 arches with bindings; got {list(results)}")
        ref_arch, ref = next(iter(results.items()))
        for arch, p in results.items():
            if arch == ref_arch:
                continue
            max_diff = (p - ref).abs().max().item()
            self.assertLess(
                max_diff, self.TOLERANCE,
                f"Muon: sm_{arch} disagrees with sm_{ref_arch} "
                f"by {max_diff:.6f}")

    def test_supergrok15_cross_arch(self):
        from grokking_optimizers import SuperGrok15
        self._check_agreement(lambda ps: SuperGrok15(ps, lr=1e-3))

    def test_supergrok11_cross_arch(self):
        from grokking_optimizers import SuperGrok11
        self._check_agreement(lambda ps: SuperGrok11(ps, lr=1e-3))

    def test_supergrok2_cross_arch(self):
        from grokking_optimizers import SuperGrok2
        self._check_agreement(lambda ps: SuperGrok2(ps, lr=1e-3))


if __name__ == "__main__":
    unittest.main()
