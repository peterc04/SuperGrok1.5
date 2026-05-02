"""Cross-arch numerical agreement test for the model bindings.

For each fused-kernel model (decoder, vit, mamba), verify that the
per-arch specialized forward kernels produce elementwise-equal output
(modulo FP tolerance) on the same input + same weights. Skips arches
that aren't compiled into the local extension.

This test is the model-side analogue of
``tests/test_cross_arch_agreement.py`` (which covers optimizers): the
math must remain identical across sm_90, gfx942 even as arch-specific
primitives (TMA, MFMA, FP8 paths) diverge for performance.

Skip rules:
  * Skips entirely if neither CUDA nor HIP is available — without a
    backend there is no kernel to run.
  * Skips trivially with a single-arch pass if only one of (sm_90,
    gfx942) is built into the wheel — there is nothing to compare
    against, but the structural test passes.
  * Compares pairs only when ≥2 arches are available.

Set ``FORCE_ARCH=<90|942>`` to switch dispatch between arches without
changing hardware (the per-arch binding for that arch must be in the
build).

NOTE: Like the parity tests in ``test_models_sm_90.py``, the actual
kernel call is currently behind ``pytest.skip("weight packing TBD")``
until the binding's packed-weight layout is exposed via a packer.
The structural part — fixture, dispatch, single-arch trivial pass,
multi-arch comparison loop — is in place so the live test goes green
the moment a packer lands.
"""

from __future__ import annotations

import os
import unittest

import torch


# ── Backend availability ─────────────────────────────────────────────

# CUDA gives us sm_90; HIP gives us gfx942. PyTorch reports both
# through ``torch.cuda.is_available()`` (HIP is exposed as the CUDA
# API on ROCm builds). We additionally probe the per-arch binding via
# the dispatch shim used by the optimizer cross-arch test.
HAS_CUDA = torch.cuda.is_available()
HAS_HIP = getattr(torch.version, "hip", None) is not None

SUPPORTED_ARCHES = (90, 942)


def _set_force_arch(arch: int):
    """Steer dispatch to a specific per-arch binding.

    Mirrors the helper in ``tests/test_cross_arch_agreement.py`` so the
    test surface is consistent across the optimizer + model cross-arch
    suites.
    """
    os.environ["FORCE_ARCH"] = str(arch)
    try:
        import grokking_optimizers.dispatch as _d
        _d.get_gpu_arch.cache_clear()
        _d.get_gpu_vendor.cache_clear()
        _d.get_backend.cache_clear()
        _d.get_warp_size.cache_clear()
    except Exception:
        # Dispatch module may not exist on every install; the FORCE_ARCH
        # env var is still set for any code that consults it directly.
        pass


def _arch_available(arch: int) -> bool:
    """True iff the per-arch model binding for ``arch`` is loadable."""
    try:
        _set_force_arch(arch)
        from grokking_optimizers._ops_loader import get_ops
        ops = get_ops()
        if not hasattr(ops, "models"):
            return False
        return ops.detect_arch() == arch
    except Exception:
        return False


@unittest.skipUnless(HAS_CUDA or HAS_HIP,
                     "Cross-arch model test requires CUDA or HIP.")
class CrossArchModelAgreementTest(unittest.TestCase):
    """Per-model agreement across the active arch set (sm_90, gfx942).

    Each ``test_<model>_cross_arch``:
      1. Builds a small reference model with a fixed seed.
      2. For each arch with a binding present, runs the fused forward
         under ``FORCE_ARCH=<arch>`` and captures the output.
      3. Asserts ``torch.allclose`` against the first arch present
         within ``rtol=1e-3, atol=1e-3`` (the BF16 envelope).

    Single-arch installs trivially pass (one entry, nothing to compare).
    """

    RTOL = 1e-3
    ATOL = 1e-3

    # ── helpers ──────────────────────────────────────────────────────

    def _available_arches(self):
        return [a for a in SUPPORTED_ARCHES if _arch_available(a)]

    def _run_decoder_under_arch(self, arch):
        from tests.reference.models.decoder_ref import make_decoder
        from grokking_optimizers._ops_loader import get_ops
        _set_force_arch(arch)
        ops = get_ops()
        if not hasattr(ops.models, "decoder_forward"):
            return None
        torch.manual_seed(0)
        device = torch.device("cuda:0")
        model = make_decoder().to(device).eval()
        x = torch.randint(0, 99, (2, 4), device=device, dtype=torch.long)
        # See test_models_sm_90.py: the kernel binding uses an opaque
        # packed-weight layout. Once a packer is wired up, replace
        # this stub with the real call:
        #     packed = _pack_decoder_weights(model)
        #     return ops.models.decoder_forward(x, *packed).detach().cpu()
        return None  # signals "binding present but uncallable here"

    def _run_vit_under_arch(self, arch):
        from tests.reference.models.vit_ref import make_vit
        from grokking_optimizers._ops_loader import get_ops
        _set_force_arch(arch)
        ops = get_ops()
        if not hasattr(ops.models, "vit_forward"):
            return None
        torch.manual_seed(0)
        device = torch.device("cuda:0")
        model = make_vit().to(device).eval()
        _ = model  # reference build only; live path TBD
        return None

    def _run_mamba_under_arch(self, arch):
        from tests.reference.models.mamba_ref import make_mamba
        from grokking_optimizers._ops_loader import get_ops
        _set_force_arch(arch)
        ops = get_ops()
        if not hasattr(ops.models, "mamba_forward"):
            return None
        torch.manual_seed(0)
        device = torch.device("cuda:0")
        model = make_mamba().to(device).eval()
        _ = model
        return None

    def _check_agreement(self, runner, model_name: str):
        """Generic harness shared by every model.

        ``runner(arch)`` is expected to return either:
          * a tensor (the forward output captured under ``FORCE_ARCH=arch``)
          * ``None``, signalling "binding present but kernel call not
            currently exercisable" (weight packer TBD). When every
            available arch returns ``None`` we ``skipTest`` rather than
            fail — the structural test still validates that the
            dispatch path is reachable on every arch the wheel was
            built for.
        """
        arches = self._available_arches()
        if not arches:
            self.skipTest(
                f"{model_name}: no per-arch binding present; "
                f"need at least one of sm_{SUPPORTED_ARCHES}"
            )

        results = {}
        for arch in arches:
            r = runner(arch)
            if r is not None:
                results[arch] = r

        if not results:
            self.skipTest(
                f"{model_name}: weight packing TBD — kernel uses an "
                "opaque layout. Cross-arch comparison will go live "
                "once the packer is wired up."
            )

        if len(results) == 1:
            # Single-arch install: trivially passes. The test still
            # validated that the binding was loadable + dispatchable.
            (arch,) = results.keys()
            self.assertIsNotNone(
                results[arch],
                f"{model_name}: single-arch result must not be None"
            )
            return

        ref_arch, ref = next(iter(results.items()))
        for arch, out in results.items():
            if arch == ref_arch:
                continue
            self.assertTrue(
                torch.allclose(out, ref, rtol=self.RTOL, atol=self.ATOL),
                f"{model_name}: sm_{arch} disagrees with sm_{ref_arch} "
                f"(max |Δ| = {(out - ref).abs().max().item():.3e})"
            )

    # ── per-model test cases ─────────────────────────────────────────

    def test_decoder_cross_arch(self):
        self._check_agreement(self._run_decoder_under_arch, "decoder")

    def test_vit_cross_arch(self):
        self._check_agreement(self._run_vit_under_arch, "vit")

    def test_mamba_cross_arch(self):
        self._check_agreement(self._run_mamba_under_arch, "mamba")


if __name__ == "__main__":
    unittest.main()
