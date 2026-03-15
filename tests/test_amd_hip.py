#!/usr/bin/env python3
"""
AMD HIP Port — Test Suite

Tests for:
  - platform.h: no raw CUDA calls in generic kernels
  - AMD tier detection: get_amd_tier() with FORCE_ARCH
  - PrecisionConfig: AMD auto-detection selects BF16 for CDNA2/3
  - dispatch.py: get_amd_label() returns correct labels

These tests run on any platform (CPU-only is fine for dispatch/precision tests).
"""

import os
import sys
import traceback
import functools

# ─── Test infrastructure ────────────────────────────────────────────

results = []


def run_test(name, fn):
    """Run a test function, record PASS/FAIL."""
    try:
        fn()
        results.append((name, True, ""))
        print(f"  PASS: {name}")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, False, str(e)))
        print(f"  FAIL: {name} — {e}")
        print(tb)


# ═══════════════════════════════════════════════════════════════════════
#  Test 1: platform.h — no raw CUDA calls in generic kernels
# ═══════════════════════════════════════════════════════════════════════

def test_platform_h_no_raw_cuda():
    """Verify generic kernels use platform.h macros, not raw CUDA APIs."""
    import re
    from pathlib import Path

    generic_dir = Path(__file__).parent.parent / "csrc" / "cuda" / "generic"
    if not generic_dir.exists():
        raise RuntimeError(f"Generic kernel dir not found: {generic_dir}")

    # Patterns that should NOT appear in generic kernels
    # (they should use platform.h macros instead)
    raw_cuda_patterns = [
        r'\bcudaMemsetAsync\b',
        r'\bcudaStream_t\b',
        r'\bcudaStreamCreate\b',
        r'\bcudaStreamSynchronize\b',
        r'\bcudaStreamDestroy\b',
        r'\bcudaFuncSetAttribute\b',
        r'\bcudaGetLastError\b',
        r'\bcudaGetErrorString\b',
        r'\bcudaDeviceSynchronize\b',
    ]

    violations = []
    for cu_file in sorted(generic_dir.glob("*.cu")):
        content = cu_file.read_text()
        for pattern in raw_cuda_patterns:
            matches = re.findall(pattern, content)
            if matches:
                violations.append(f"{cu_file.name}: {matches[0]} ({len(matches)}x)")

    if violations:
        raise AssertionError(
            f"Raw CUDA API calls found in generic kernels:\n" +
            "\n".join(f"  {v}" for v in violations)
        )


# ═══════════════════════════════════════════════════════════════════════
#  Test 2: AMD tier detection with FORCE_ARCH
# ═══════════════════════════════════════════════════════════════════════

def test_amd_tier_detection():
    """Test get_amd_tier() returns correct tier for various FORCE_ARCH values."""
    from grokking_optimizers import dispatch
    from unittest.mock import patch

    test_cases = [
        # (FORCE_ARCH, expected_tier)
        ("942", "cdna3"),
        ("90", "cdna2"),
        ("908", "generic"),  # MI100 = generic
        ("94", "cdna3"),     # capability format
    ]

    for force_arch, expected in test_cases:
        # Clear all caches
        dispatch.get_gpu_arch.cache_clear()
        dispatch.get_gpu_vendor.cache_clear()
        dispatch.get_amd_tier.cache_clear()

        old_force = os.environ.get("FORCE_ARCH")
        os.environ["FORCE_ARCH"] = force_arch

        # Patch get_gpu_vendor to return 'amd'
        with patch.object(dispatch, 'get_gpu_vendor',
                          new=functools.lru_cache(maxsize=1)(lambda: 'amd')):
            dispatch.get_amd_tier.cache_clear()
            tier = dispatch.get_amd_tier()
            assert tier == expected, (
                f"FORCE_ARCH={force_arch}: expected '{expected}', got '{tier}'"
            )

        # Restore
        dispatch.get_gpu_vendor.cache_clear()
        dispatch.get_gpu_arch.cache_clear()
        dispatch.get_amd_tier.cache_clear()
        if old_force is not None:
            os.environ["FORCE_ARCH"] = old_force
        else:
            os.environ.pop("FORCE_ARCH", None)


# ═══════════════════════════════════════════════════════════════════════
#  Test 3: PrecisionConfig AMD auto-detection
# ═══════════════════════════════════════════════════════════════════════

def test_precision_auto_amd():
    """Test PrecisionConfig auto-selects BF16 for CDNA2/3, FP32 for generic."""
    from grokking_optimizers import dispatch
    from grokking_optimizers import quantization as quant_mod
    from unittest.mock import patch

    test_cases = [
        # (FORCE_ARCH, expected_proj_precision, bf16_supported)
        ("942", "bf16", True),     # CDNA3: BF16 MFMA
        ("90", "bf16", True),      # CDNA2: BF16 MFMA
        ("908", "fp32", False),    # MI100: no BF16 MFMA
    ]

    for force_arch, expected_proj, bf16_avail in test_cases:
        # Clear caches
        dispatch.get_gpu_arch.cache_clear()
        dispatch.get_gpu_vendor.cache_clear()
        dispatch.get_amd_tier.cache_clear()

        old_force = os.environ.get("FORCE_ARCH")
        os.environ["FORCE_ARCH"] = force_arch

        mock_vendor = functools.lru_cache(maxsize=1)(lambda: 'amd')
        mock_bf16 = lambda: bf16_avail

        # Patch in both dispatch and quantization modules
        with patch.object(dispatch, 'get_gpu_vendor', new=mock_vendor), \
             patch.object(quant_mod, 'get_gpu_vendor', new=mock_vendor), \
             patch.object(dispatch, 'supports_bf16', new=mock_bf16), \
             patch.object(quant_mod, 'supports_bf16', new=mock_bf16):
            dispatch.get_amd_tier.cache_clear()
            # Also patch get_amd_tier in quantization module
            quant_mod.get_amd_tier = dispatch.get_amd_tier

            from grokking_optimizers.quantization import PrecisionConfig
            pc = PrecisionConfig(projection_precision='auto')
            assert pc.projection_precision == expected_proj, (
                f"FORCE_ARCH={force_arch}: expected proj='{expected_proj}', "
                f"got '{pc.projection_precision}'"
            )

        # Restore
        dispatch.get_gpu_vendor.cache_clear()
        dispatch.get_gpu_arch.cache_clear()
        dispatch.get_amd_tier.cache_clear()
        if old_force is not None:
            os.environ["FORCE_ARCH"] = old_force
        else:
            os.environ.pop("FORCE_ARCH", None)


# ═══════════════════════════════════════════════════════════════════════
#  Test 4: get_amd_label returns correct labels
# ═══════════════════════════════════════════════════════════════════════

def test_amd_label():
    """Test get_amd_label() returns human-readable labels."""
    from grokking_optimizers import dispatch
    from unittest.mock import patch

    test_cases = [
        ("942", "MI300X"),
        ("90", "MI250"),
    ]

    for force_arch, expected_substr in test_cases:
        dispatch.get_gpu_arch.cache_clear()
        dispatch.get_gpu_vendor.cache_clear()
        dispatch.get_amd_tier.cache_clear()

        old_force = os.environ.get("FORCE_ARCH")
        os.environ["FORCE_ARCH"] = force_arch

        with patch.object(dispatch, 'get_gpu_vendor',
                          new=functools.lru_cache(maxsize=1)(lambda: 'amd')):
            dispatch.get_amd_tier.cache_clear()
            label = dispatch.get_amd_label()
            assert expected_substr in label, (
                f"FORCE_ARCH={force_arch}: expected '{expected_substr}' in label, "
                f"got '{label}'"
            )

        dispatch.get_gpu_vendor.cache_clear()
        dispatch.get_gpu_arch.cache_clear()
        dispatch.get_amd_tier.cache_clear()
        if old_force is not None:
            os.environ["FORCE_ARCH"] = old_force
        else:
            os.environ.pop("FORCE_ARCH", None)


# ═══════════════════════════════════════════════════════════════════════
#  Test 5: Quantization platform guards
# ═══════════════════════════════════════════════════════════════════════

def test_quantization_platform_guards():
    """Verify FP8 and NVFP4 kernels are guarded with #if GROK_CUDA."""
    from pathlib import Path
    import re

    quant_file = Path(__file__).parent.parent / "csrc" / "quantization" / "quantization_kernels.cu"
    if not quant_file.exists():
        raise RuntimeError(f"Quantization file not found: {quant_file}")

    content = quant_file.read_text()

    # FP8 and NVFP4 sections should be inside #if GROK_CUDA
    # Check that FP8 kernel definition is guarded
    fp8_section = content.find("quantize_fp8_e4m3_kernel")
    if fp8_section == -1:
        raise RuntimeError("FP8 kernel not found in quantization file")

    # Find the nearest #if before the FP8 kernel
    preceding = content[:fp8_section]
    last_if = preceding.rfind("#if GROK_CUDA")
    if last_if == -1:
        raise AssertionError("FP8 kernel not guarded with #if GROK_CUDA")

    # Check no #endif between the guard and the kernel
    between = preceding[last_if:]
    endif_count = between.count("#endif")
    if endif_count > 0:
        raise AssertionError(
            f"FP8 kernel guard has {endif_count} #endif before kernel definition"
        )

    # Similarly for NVFP4
    nvfp4_section = content.find("quantize_nvfp4_kernel")
    if nvfp4_section == -1:
        raise RuntimeError("NVFP4 kernel not found in quantization file")

    preceding_nvfp4 = content[:nvfp4_section]
    last_if_nvfp4 = preceding_nvfp4.rfind("#if GROK_CUDA")
    if last_if_nvfp4 == -1:
        raise AssertionError("NVFP4 kernel not guarded with #if GROK_CUDA")


# ═══════════════════════════════════════════════════════════════════════
#  Test 6: CDNA source files exist
# ═══════════════════════════════════════════════════════════════════════

def test_cdna_sources_exist():
    """Verify CDNA2 and CDNA3 specialization files exist."""
    from pathlib import Path

    base = Path(__file__).parent.parent / "csrc" / "hip"

    cdna2 = base / "cdna2" / "supergrok2_scan_cdna2.hip.cpp"
    cdna3 = base / "cdna3" / "supergrok2_cdna3.hip.cpp"

    assert cdna2.exists(), f"CDNA2 file not found: {cdna2}"
    assert cdna3.exists(), f"CDNA3 file not found: {cdna3}"

    # Verify they contain the expected launcher functions
    cdna2_content = cdna2.read_text()
    cdna3_content = cdna3.read_text()

    for suffix in ["_cdna2"]:
        assert f"launch_mamba3_peer_step{suffix}" in cdna2_content
        assert f"launch_mamba3_peer_batched_step{suffix}" in cdna2_content
        assert f"launch_mamba3_peer_bilevel_fwd_save_batched{suffix}" in cdna2_content
        assert f"launch_mamba3_peer_backward_batched{suffix}" in cdna2_content

    for suffix in ["_cdna3"]:
        assert f"launch_mamba3_peer_step{suffix}" in cdna3_content
        assert f"launch_mamba3_peer_batched_step{suffix}" in cdna3_content
        assert f"launch_mamba3_peer_bilevel_fwd_save_batched{suffix}" in cdna3_content
        assert f"launch_mamba3_peer_backward_batched{suffix}" in cdna3_content


# ═══════════════════════════════════════════════════════════════════════
#  Test 7: dispatch.h has AmdTier enum
# ═══════════════════════════════════════════════════════════════════════

def test_dispatch_h_amd_tier():
    """Verify dispatch.h contains AmdTier enum and get_amd_tier()."""
    from pathlib import Path

    dispatch_h = Path(__file__).parent.parent / "csrc" / "common" / "dispatch.h"
    content = dispatch_h.read_text()

    assert "enum class AmdTier" in content, "AmdTier enum not found in dispatch.h"
    assert "CDNA2" in content, "CDNA2 not found in AmdTier enum"
    assert "CDNA3" in content, "CDNA3 not found in AmdTier enum"
    assert "get_amd_tier()" in content, "get_amd_tier() not found in dispatch.h"
    assert "get_amd_tier_name()" in content, "get_amd_tier_name() not found in dispatch.h"


# ═══════════════════════════════════════════════════════════════════════
#  Test 8: ops.h has WITH_HIP guards
# ═══════════════════════════════════════════════════════════════════════

def test_ops_h_hip_guards():
    """Verify ops.h has proper WITH_HIP guards for AMD tier declarations."""
    from pathlib import Path

    ops_h = Path(__file__).parent.parent / "csrc" / "common" / "ops.h"
    content = ops_h.read_text()

    # Check outer guard includes WITH_HIP
    assert "#if defined(WITH_CUDA) || defined(WITH_HIP)" in content, \
        "ops.h missing combined WITH_CUDA/WITH_HIP guard"

    # Check AMD tier declarations are guarded
    assert "#ifdef WITH_HIP" in content, "ops.h missing #ifdef WITH_HIP"

    # Check CDNA2/CDNA3 launchers declared
    assert "launch_mamba3_peer_step_cdna2" in content
    assert "launch_mamba3_peer_step_cdna3" in content
    assert "launch_mamba3_peer_batched_step_cdna2" in content
    assert "launch_mamba3_peer_batched_step_cdna3" in content


# ═══════════════════════════════════════════════════════════════════════
#  Run all tests
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("AMD HIP Port — Test Suite")
    print("=" * 60)

    run_test("1. platform.h no raw CUDA in generic", test_platform_h_no_raw_cuda)
    run_test("2. AMD tier detection (FORCE_ARCH)", test_amd_tier_detection)
    run_test("3. PrecisionConfig AMD auto", test_precision_auto_amd)
    run_test("4. AMD labels", test_amd_label)
    run_test("5. Quantization platform guards", test_quantization_platform_guards)
    run_test("6. CDNA source files exist", test_cdna_sources_exist)
    run_test("7. dispatch.h AmdTier enum", test_dispatch_h_amd_tier)
    run_test("8. ops.h WITH_HIP guards", test_ops_h_hip_guards)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"Results: {passed} passed, {failed} failed, {len(results)} total")

    if failed > 0:
        print("\nFailed tests:")
        for name, ok, msg in results:
            if not ok:
                print(f"  {name}: {msg}")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
